"""LangGraph workflows for simulator diagnosis and approved experiments."""

import json
import math
import os
import secrets
import time
from copy import deepcopy
from threading import Lock
from typing import Annotated, Literal, TypedDict
from urllib.request import Request, urlopen
from uuid import uuid4

from fastapi import Depends, FastAPI, Header, HTTPException
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field

from gpu_sim import serving_scheduler
from gpu_sim.analytics.benchmark import (
    BenchmarkCriteria,
    BenchmarkSample,
    analyze_benchmark,
)
from gpu_sim.analytics.diagnostics import diagnose, last
from gpu_sim.analytics.measurements import Evidence, ObservationWindow, collect
from gpu_sim.analytics.prometheus import Prometheus, Scope
from gpu_sim.simulator import Config


class Run(BaseModel):
    model_config = ConfigDict(extra="forbid")
    goal: str = Field(default="", max_length=1000)
    role: Literal["profiler", "scheduler"] = "profiler"


class Sweep(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    goal: str = Field(default="", max_length=1000)
    start_rps: float = Field(ge=0, le=120)
    end_rps: float = Field(ge=0, le=120)
    step_rps: float = Field(ge=0.1)
    settle_sec: int = Field(ge=1, le=180)


class ServingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    gpus: int = Field(ge=1, le=4)
    name: str = Field(
        min_length=1, max_length=40, pattern=r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?$"
    )


class Decision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    decision: Literal["approve", "reject"]


class GraphState(TypedDict, total=False):
    run: Run
    evidence: Evidence
    incident_report: dict
    actions: list
    scheduler_summary: str
    serving: dict
    plan: dict
    result: dict


def sim_request(path: str, payload: dict | None = None) -> dict:
    request = Request(
        os.environ["SIM_URL"].rstrip("/") + path,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=10) as response:
        return json.load(response)


def evidence() -> Evidence:
    state = sim_request("/state")
    scopes = [
        Scope(
            job=os.environ[f"ANALYTICS_{name}_JOB"],
            instance=os.environ[f"ANALYTICS_{name}_INSTANCE"],
        )
        for name in ("DCGM", "SERVING")
    ]
    result = collect(
        Prometheus(os.environ["PROM_URL"]),
        state,
        sim_request("/cluster"),
        *scopes,
        ObservationWindow(end=time.time(), duration=60, step=2, max_age=10),
    )
    if baseline(state) != baseline(sim_request("/state")):
        raise RuntimeError("Simulator state changed during evidence collection")
    return result


def baseline(state: dict) -> dict:
    return {
        key: state[key]
        for key in ("config_revision_seconds", "config", "workload", "fault")
    }


def summarize_with_llm(role: str, evidence: dict) -> dict:
    url, model = os.getenv("OPENAI_BASE_URL"), os.getenv("MAIN_MODEL")
    if not url or not model:
        return {"status": "unavailable", "text": "LLM summary unavailable: configure OPENAI_BASE_URL and MAIN_MODEL on the coordinator."}
    headers = {"Content-Type": "application/json"}
    if token := os.getenv("OPENAI_API_KEY"):
        headers["Authorization"] = f"Bearer {token}"
    instructions = (
        "Summarize only the supplied simulator evidence for an operator. "
        "Identify uncertainty and missing measurements; missing data is not evidence of a fault. "
        "Do not claim actions were executed unless execution is explicitly confirmed."
    )
    if role == "scheduler":
        instructions += (
            " The waiting queue is one shared serving measurement, not a per-node count; "
            "do not add repeated values. If proposals are present, describe each as pending "
            "human approval, including its reason, impact and proposed changes. "
            "If there are no proposals, state that no change was proposed."
        )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": instructions},
            {"role": "user", "content": json.dumps({"agent": role, "evidence": evidence})},
        ],
        "temperature": 0,
    }
    try:
        request = Request(url.rstrip("/") + "/chat/completions", data=json.dumps(payload).encode(), headers=headers)
        with urlopen(request, timeout=30) as response:
            content = json.load(response)["choices"][0]["message"]["content"]
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Empty model response")
        return {"status": "ok", "text": content.strip()}
    except Exception:  # noqa: BLE001 - summaries must not change approval outcomes
        return {"status": "unavailable", "text": "LLM summary unavailable: check coordinator configuration and model server connectivity."}


def summary_evidence(result: dict, goal: str) -> dict:
    report = result["incident_report"]
    findings = deepcopy(report.get("findings", []))
    proposals = deepcopy(result.get("actions", []))
    for finding in findings + [item for plan in proposals for item in plan.get("evidence", [])]:
        values = finding.get("evidence", {})
        if "waiting_queue" in values:
            values["shared_serving_waiting_queue"] = values.pop("waiting_queue")
    return {
        "goal": goal,
        "findings": findings,
        "measurement_quality": report.get("measurement_quality", {}),
        "limitations": report.get("limitations", []),
        "scheduler_summary": result.get("scheduler_summary", ""),
        "proposals": proposals,
    }


def profile(state: GraphState) -> dict:
    observed = evidence().for_agent()
    return {
        "evidence": observed,
        "incident_report": {
            "goal": state["run"].goal,
            "findings": [item.model_dump() for item in diagnose(observed)],
            "measurement_quality": {
                key: item.quality for key, item in observed.measurements.items()
            },
            "cluster": observed.cluster,
            "limitations": observed.limitations,
        },
    }


def schedule(state: GraphState) -> dict:
    observed = state["evidence"]
    live = sim_request("/state")
    config = live["config"]
    findings = state["incident_report"]["findings"]
    pressure = {item["kind"] for item in findings}
    directions = {"prefill_compute_pressure": 1, "decode_bandwidth_pressure": -1}
    selected = pressure & directions.keys()
    action = None
    if (
        live["config_revision_seconds"] == observed.context["config_revision_seconds"]
        and live["workload"] == observed.context["workload"]
        and live["fault"]["type"] == "none"
        and config["disaggregated"]
        and len(selected) == 1
    ):
        reason = selected.pop()
        shift = directions[reason] * config["tp_size"]
        patch = {
            "prefill_gpus": config["prefill_gpus"] + shift,
            "decode_gpus": config["decode_gpus"] - shift,
        }
        try:
            Config.model_validate(config | patch)
        except ValueError:
            pass
        else:
            action = {
                "kind": "rebalance",
                "baseline": baseline(live),
                "payload": patch,
                "reason": reason,
                "impact": "Move one TP group and reset in-flight requests",
                "evidence": [item for item in findings if item["kind"] == reason],
            }
    return {
        "serving": serving_scheduler.audit(),
        "actions": [action] if action else [],
        "scheduler_summary": (
            "Review the simulator GPU rebalance proposal."
            if action
            else "No supported change justified by current evidence."
        ),
    }


def scalar(observed: Evidence, name: str) -> float | None:
    values = last(observed.measurements[name])
    return values[0][1] if len(values) == 1 and math.isfinite(values[0][1]) else None


def benchmark(state: GraphState) -> dict:
    plan = state["plan"]
    original = plan["baseline"]
    expected = original
    samples = []
    restored = False
    try:
        for rps in plan["rates"]:
            if baseline(sim_request("/state")) != expected:
                raise RuntimeError("Simulator changed outside the approved experiment")
            expected = original | {"workload": original["workload"] | {"rps": rps}}
            sim_request("/workload", {"rps": rps})
            time.sleep(plan["hold_seconds"])
            observed = evidence()
            if baseline(observed.context) != expected:
                raise RuntimeError("Simulator changed during the approved experiment")
            window = observed.measurements["waiting_queue"]
            samples.append(
                BenchmarkSample(
                    offered_rps=rps,
                    completed_rps=scalar(observed, "completed_rps"),
                    queue_growth_rps=scalar(observed, "queue_growth_rps"),
                    ttft_p95_seconds=scalar(observed, "ttft_p95"),
                    measurement_start=window.start,
                    measurement_end=window.evaluated_at,
                    valid=scalar(observed, "completion_counter_resets") == 0
                    and scalar(observed, "offered_rps") == rps,
                )
            )
    finally:
        if baseline(sim_request("/state")) == expected:
            restored = (
                baseline(sim_request("/workload", original["workload"])) == original
            )
    analysis = analyze_benchmark(
        samples, BenchmarkCriteria.model_validate(plan["criteria"])
    )
    lines = [
        "# Simulator capacity report",
        "",
        f"Classification: {analysis.status}",
        f"Sustainable RPS: {analysis.sustainable_rps}",
        f"First failing RPS: {analysis.first_failure_rps}",
        "",
        "| Offered RPS | Completed RPS | Queue growth | TTFT p95 (s) | Assessment |",
        "|---:|---:|---:|---:|---|",
    ]
    for row in analysis.samples:
        lines.append(
            "| "
            + " | ".join(
                str(row[key])
                for key in (
                    "offered_rps",
                    "completed_rps",
                    "queue_growth_rps",
                    "ttft_p95_seconds",
                    "assessment",
                )
            )
            + " |"
        )
    lines.extend(["", *analysis.limitations])
    return {
        "result": analysis.model_dump()
        | {
            "workload_restored": restored,
            "markdown_report": "\n".join(lines),
        }
    }


diagnosis_graph = StateGraph(GraphState)
diagnosis_graph.add_node("ProfilerAgent", profile)
diagnosis_graph.add_node("SchedulerAgent", schedule)
diagnosis_graph.add_edge(START, "ProfilerAgent")
diagnosis_graph.add_conditional_edges(
    "ProfilerAgent",
    lambda state: state["run"].role,
    {"profiler": END, "scheduler": "SchedulerAgent"},
)
diagnosis_graph.add_edge("SchedulerAgent", END)
diagnosis = diagnosis_graph.compile()

benchmark_graph = StateGraph(GraphState)
benchmark_graph.add_node("BenchmarkAgent", benchmark)
benchmark_graph.add_edge(START, "BenchmarkAgent")
benchmark_graph.add_edge("BenchmarkAgent", END)
experiment = benchmark_graph.compile()

app = FastAPI(title="GPU simulation agents", docs_url=None, redoc_url=None)
records: dict[str, dict] = {}
gate = Lock()
execution = Lock()


def authorize(authorization: Annotated[str | None, Header()] = None) -> None:
    token = os.getenv("AGENT_OPERATOR_TOKEN")
    if not token or not secrets.compare_digest(
        (authorization or "").encode(), f"Bearer {token}".encode()
    ):
        raise HTTPException(401, "Operator token required")


Operator = Annotated[None, Depends(authorize)]


def propose(plan: dict) -> dict:
    with gate:
        for record in records.values():
            if record["status"] == "pending" and record["plan"] == plan:
                return record.copy()
        item = {"id": uuid4().hex, "status": "pending", "plan": plan}
        records[item["id"]] = item
        return item.copy()


@app.get("/approvals")
def approvals(_: Operator) -> dict:
    with gate:
        return {
            "human_in_the_loop": True,
            "pending": [
                row.copy() for row in records.values() if row["status"] == "pending"
            ],
            "history": [
                row.copy() for row in records.values() if row["status"] != "pending"
            ],
        }


@app.post("/run")
def run(command: Run, _: Operator) -> dict:
    result = diagnosis.invoke({"run": command})
    report = result["incident_report"]
    return {
        "incident_report": report,
        "serving": result.get("serving"),
        "scheduler_summary": result.get("scheduler_summary", ""),
        "approvals": [propose(plan) for plan in result.get("actions", [])],
        "llm_summary": summarize_with_llm(command.role, summary_evidence(result, command.goal)),
    }


@app.post("/benchmark")
def plan_benchmark(command: Sweep, _: Operator) -> dict:
    count = (
        math.floor((command.end_rps - command.start_rps) / command.step_rps + 1e-9) + 1
    )
    if not 1 <= count <= 6:
        raise HTTPException(422, "Sweep requires one to six ascending rates")
    current = baseline(sim_request("/state"))
    if current["fault"]["type"] != "none":
        raise HTTPException(
            409, "Clear the active simulator fault before planning a sweep"
        )
    return propose(
        {
            "kind": "benchmark",
            "baseline": current,
            "goal": command.goal,
            "rates": [
                round(command.start_rps + i * command.step_rps, 9) for i in range(count)
            ],
            "hold_seconds": max(130, command.settle_sec),
            "criteria": BenchmarkCriteria(
                min_throughput_fraction=0.9,
                max_queue_growth_rps=1,
                ttft_slo_seconds=5,
                simulation_tick_seconds=1,
            ).model_dump(),
            "impact": "Change offered load temporarily and restore the approved baseline",
        }
    )


@app.get("/serving")
def inspect_serving(_: Operator) -> dict:
    return serving_scheduler.audit()


@app.post("/serving")
def plan_serving(command: ServingRequest, _: Operator) -> dict:
    try:
        return propose(serving_scheduler.plan_service(command.name, command.gpus))
    except (RuntimeError, OSError) as error:
        raise HTTPException(409, str(error)) from error


@app.post("/approvals/decide")
def decide(command: Decision, _: Operator) -> dict:
    with gate:
        record = records.get(command.id)
        if not record or record["status"] != "pending":
            raise HTTPException(409, "Approval is unknown or already decided")
        if command.decision == "reject":
            record["status"] = "rejected"
            return record.copy()
        if not execution.acquire(blocking=False):
            raise HTTPException(409, "An approved action is already executing")
        record["status"] = "executing"
    outcome = {"status": "failed", "error": "Execution interrupted"}
    try:
        plan = record["plan"]
        if (
            plan["kind"] != "kserve_submit"
            and baseline(sim_request("/state")) != plan["baseline"]
        ):
            raise RuntimeError("Approved baseline changed. Request a new proposal.")
        if plan["kind"] == "kserve_submit":
            result = serving_scheduler.submit(plan)
        elif plan["kind"] == "benchmark":
            result = experiment.invoke({"plan": plan})["result"]
            if not result["workload_restored"]:
                raise RuntimeError(
                    "Original workload could not be verified after the sweep"
                )
            result["llm_summary"] = summarize_with_llm("benchmark", result)
        else:
            # Require the same pressure and payload at execution time.
            refreshed = diagnosis.invoke({"run": Run(role="scheduler")})
            if not any(
                all(
                    candidate[key] == plan[key]
                    for key in ("baseline", "reason", "payload")
                )
                for candidate in refreshed["actions"]
            ):
                raise RuntimeError(
                    "Scheduling evidence changed. Request a new proposal."
                )
            applied = sim_request("/config", plan["payload"])
            verified = sim_request("/state")
            if any(
                verified["config"][key] != value
                for key, value in plan["payload"].items()
            ):
                raise RuntimeError("Could not verify the approved configuration")
            result = {
                "config": verified["config"],
                "requests_reset": applied["requests_reset"],
            }
        outcome = {"status": "executed", "result": result}
    except Exception as error:  # noqa: BLE001 - record execution failures at the approval boundary
        outcome = {"status": "failed", "error": str(error)}
    finally:
        with gate:
            record.update(outcome)
            execution.release()
    return record.copy()
