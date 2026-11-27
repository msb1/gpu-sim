"""Scoped simulator metric queries and stack-aware measurements."""

from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field

from gpu_sim.analytics.freshness import (
    gate_scrape,
    gpu_identities,
    revision_error,
    sources,
    verify_source,
)
from gpu_sim.analytics.prometheus import Prometheus, QueryResult, Scope


class ObservationWindow(BaseModel):
    model_config = ConfigDict(extra="forbid")
    end: float
    duration: float = Field(gt=0)
    step: float = Field(gt=0)
    max_age: float = Field(gt=0)


class Evidence(BaseModel):
    context: dict[str, Any]
    cluster: dict[str, Any]
    measurements: dict[str, QueryResult]
    limitations: list[str] = Field(default_factory=list)
    source_checks: dict[str, QueryResult] = Field(default_factory=dict)

    def for_agent(self) -> Self:
        return self.model_copy(
            update={"context": {k: v for k, v in self.context.items() if k != "fault"}},
            deep=True,
        )


def queries(
    config: dict[str, Any], dcgm: Scope, serving: Scope, window: ObservationWindow
) -> dict[str, tuple[str, str, bool]]:
    d, s = dcgm.selector(), serving.selector()
    model = config["model_name"]
    sm = serving.selector(model_name=model)
    role = lambda metric: f"avg by (node,role) ({metric}{d})"
    by_node = lambda metric: f"max by (node) ({metric}{d})"
    rate_window = f"{window.duration:g}s"
    vllm = config["backend_engine"] == "vllm"
    queue = (
        f"vllm:num_requests_waiting{sm}"
        if vllm
        else f"nv_trt_llm_request_metrics{serving.selector(model_name=model, request_type='context')}"
    )
    running = (
        f"vllm:num_requests_running{sm}"
        if vllm
        else f"nv_trt_llm_request_metrics{serving.selector(model_name=model, request_type='generation')}"
    )
    tokens = (
        "vllm:generation_tokens_total" if vllm else "nv_trt_llm_generation_tokens_total"
    )
    ttft = (
        "vllm:time_to_first_token_seconds_bucket"
        if vllm
        else "sim_trt_llm_time_to_first_token_seconds_bucket"
    )
    if vllm:
        kv = f"vllm:gpu_cache_usage_perc{sm}"
    else:
        used = f"nv_trt_llm_kv_cache_block_metrics{serving.selector(model_name=model, kv_metric='used_blocks')}"
        free = f"nv_trt_llm_kv_cache_block_metrics{serving.selector(model_name=model, kv_metric='free_blocks')}"
        kv = f"sum({used}) / (sum({used}) + sum({free}))"
    return {
        "dcgm_scrape_up": (f"up{d}", "boolean", True),
        "serving_scrape_up": (f"up{s}", "boolean", True),
        "offered_rps": (f"sim_workload_target_rps{s}", "requests/s", True),
        "completed_rps": (
            f"sum(rate(sim_completed_requests_total{s}[{rate_window}]))",
            "requests/s",
            False,
        ),
        "completion_counter_resets": (
            f"sum(resets(sim_completed_requests_total{s}[{rate_window}]))",
            "resets",
            False,
        ),
        "generated_tokens_s": (
            f"sum(rate({tokens}{sm}[{rate_window}]))",
            "tokens/s",
            False,
        ),
        "prefix_cache_hit_tokens_s": (
            f"sum(rate(dynamo_prefix_cache_hit_tokens_total{sm}[{rate_window}]))",
            "tokens/s",
            False,
        ),
        "nixl_transfer_bytes_s": (
            f"sum(rate(dynamo_nixl_kv_transfer_bytes_total{sm}[{rate_window}]))",
            "bytes/s",
            False,
        ),
        "waiting_queue": (queue, "requests", True),
        "queue_growth_rps": (f"deriv({queue}[{rate_window}])", "requests/s", False),
        "running_requests": (running, "requests", True),
        "kv_occupancy": (kv, "ratio", True),
        "ttft_p95": (
            f"histogram_quantile(0.95, sum by (le) (rate({ttft}{sm}[{rate_window}])))",
            "seconds",
            False,
        ),
        "sm_occupancy": (role("DCGM_FI_PROF_SM_OCCUPANCY"), "ratio", True),
        "tensor_active": (role("DCGM_FI_PROF_PIPE_TENSOR_ACTIVE"), "ratio", True),
        "dram_active": (role("DCGM_FI_PROF_DRAM_ACTIVE"), "ratio", True),
        "gpu_temp": (by_node("DCGM_FI_DEV_GPU_TEMP"), "celsius", True),
        "throttle": (by_node("DCGM_FI_DEV_CLOCK_THROTTLE_REASONS"), "bitmask", True),
        "xid": (by_node("DCGM_FI_DEV_XID_ERRORS"), "code", True),
        "xid_gpu": (f"DCGM_FI_DEV_XID_ERRORS{d}", "code", True),
        "nvlink_xid_events": (f"sim_nvlink_xid_event_info{d}", "event", True),
        "nvlink_peer_observations": (
            f"sim_nvlink_peer_observation_info{d}",
            "observation",
            True,
        ),
        "nvlink_crc_rate": (
            f"sum by (node) (rate(sim_nvlink_crc_errors_total{d}[{rate_window}]))",
            "errors/s",
            True,
        ),
        "nvlink_crc_total": (
            f"sum by (node) (sim_nvlink_crc_errors_total{d})",
            "errors",
            True,
        ),
        **{
            f"{name}_source_age": (
                f"max by ({'reporter_node' if name == 'nvlink_peer_observations' else 'node'}) (time() - timestamp({metric}{d}))",
                "seconds",
                True,
            )
            for name, metric in (
                ("gpu_temp", "DCGM_FI_DEV_GPU_TEMP"),
                ("throttle", "DCGM_FI_DEV_CLOCK_THROTTLE_REASONS"),
                ("xid", "DCGM_FI_DEV_XID_ERRORS"),
                ("xid_gpu", "DCGM_FI_DEV_XID_ERRORS"),
                ("nvlink_xid_events", "sim_nvlink_xid_event_info"),
                ("nvlink_peer_observations", "sim_nvlink_peer_observation_info"),
                ("nvlink_crc_rate", "sim_nvlink_crc_errors_total"),
                ("nvlink_crc_total", "sim_nvlink_crc_errors_total"),
            )
        },
        "nvlink_tx": (f"sum(DCGM_FI_PROF_NVLINK_TX_BYTES{d})", "bytes/s", False),
        "ib_tx": (f"sum(dynamo_inter_node_ib_tx_bytes_per_sec{sm})", "bytes/s", False),
        "power": (f"sum(DCGM_FI_DEV_POWER_USAGE{d})", "watts", False),
    }


def collect(
    client: Prometheus,
    state: dict[str, Any],
    cluster: dict[str, Any],
    dcgm: Scope,
    serving: Scope,
    window: ObservationWindow,
) -> Evidence:
    context = {
        k: state.get(k)
        for k in (
            "config",
            "workload",
            "fault",
            "capacity_mode",
            "hardware_assumptions",
            "config_revision_seconds",
            "inventory",
            "replicas",
        )
    }
    result = {}
    for name, (query, unit, ranged) in queries(
        state["config"], dcgm, serving, window
    ).items():
        result[name] = client.query(
            query,
            window.end,
            unit,
            start=window.end - window.duration if ranged else None,
            step=window.step if ranged else None,
            max_age=window.max_age,
        )
    source_specs = sources(state["config"], dcgm, serving)
    source_checks = {}
    for name, (selector, _, _) in source_specs.items():
        source_checks[name] = client.query(
            f"time() - timestamp({selector})",
            window.end,
            "seconds",
            start=window.end - 2 * window.duration,
            step=window.step,
            max_age=window.max_age,
        )
    for name, scope in (("dcgm_up", dcgm), ("serving_up", serving)):
        source_checks[name] = client.query(
            f"up{scope.selector()}",
            window.end,
            "boolean",
            start=window.end - 2 * window.duration,
            step=window.step,
            max_age=window.max_age,
        )
    revision_selector = f"sim_config_revision_seconds{serving.selector()}"
    source_checks["config_revision"] = client.query(
        revision_selector,
        window.end,
        "seconds",
        start=window.end - 2 * window.duration,
        step=window.step,
        max_age=window.max_age,
    )
    source_checks["config_revision_age"] = client.query(
        f"time() - timestamp({revision_selector})",
        window.end,
        "seconds",
        start=window.end - 2 * window.duration,
        step=window.step,
        max_age=window.max_age,
    )
    for source in ("completed", "tokens", "prefix", "nixl", "ttft", "crc"):
        selector = source_specs[source][0]
        source_checks[f"{source}_resets"] = client.query(
            f"sum(resets({selector}[{2 * window.duration:g}s]))",
            window.end,
            "resets",
            max_age=window.max_age,
        )
    dependencies = {
        "offered_rps": ["offered"],
        "completed_rps": ["completed"],
        "completion_counter_resets": ["completed"],
        "generated_tokens_s": ["tokens"],
        "prefix_cache_hit_tokens_s": ["prefix"],
        "nixl_transfer_bytes_s": ["nixl"],
        "waiting_queue": ["queue"],
        "queue_growth_rps": ["queue"],
        "running_requests": ["running"],
        "kv_occupancy": ["kv"]
        if state["config"]["backend_engine"] == "vllm"
        else ["kv_used_blocks", "kv_free_blocks"],
        "ttft_p95": ["ttft"],
        "sm_occupancy": ["occupancy"],
        "tensor_active": ["tensor"],
        "dram_active": ["dram"],
        "gpu_temp": ["temperature"],
        "throttle": ["throttle"],
        "xid": ["xid"],
        "xid_gpu": ["xid"],
        "nvlink_xid_events": ["event"],
        "nvlink_peer_observations": ["peer"],
        "nvlink_crc_rate": ["crc"],
        "nvlink_crc_total": ["crc"],
        "nvlink_tx": ["nvlink"],
        "ib_tx": ["ib"],
        "power": ["power"],
    }
    expected = gpu_identities(state)
    revision_problem = revision_error(
        source_checks["config_revision"], state.get("config_revision_seconds")
    ) or verify_source(source_checks["config_revision_age"], max_age=window.max_age)
    for name, needed in dependencies.items():
        if result[name].quality != "ok":
            continue
        problem = revision_problem
        for source in needed:
            _, target, gpu_scoped = source_specs[source]
            problem = problem or gate_scrape(source_checks[f"{target}_up"])
            problem = problem or verify_source(
                source_checks[source],
                max_age=window.max_age,
                expected_gpus=expected if gpu_scoped else None,
            )
            if name != "completion_counter_resets" and source in (
                "completed",
                "tokens",
                "prefix",
                "nixl",
                "ttft",
                "crc",
            ):
                resets = source_checks[f"{source}_resets"]
                if (
                    resets.quality != "ok"
                    or resets.warnings
                    or len(resets.series) != 1
                    or resets.series[0].samples[-1].value != 0
                ):
                    problem = (
                        problem
                        or f"{source} counter reset or reset history unavailable"
                    )
        if problem:
            result[name].quality = (
                "stale"
                if problem.startswith("source samples are stale")
                else "insufficient_history"
            )
            result[name].error = problem
    for name in (
        "gpu_temp",
        "xid",
        "xid_gpu",
        "nvlink_xid_events",
        "nvlink_peer_observations",
        "throttle",
        "nvlink_crc_rate",
        "nvlink_crc_total",
    ):
        age = result[f"{name}_source_age"]
        if result[name].quality == "ok" and (
            age.quality != "ok"
            or any(
                sample.value is None
                or sample.value < 0
                or sample.value > window.max_age
                for series in age.series
                for sample in series.samples
            )
        ):
            result[name].quality = "stale"
            result[
                name
            ].error = "Source freshness could not be verified for the full window"
    return Evidence(
        context=context,
        cluster=cluster,
        measurements=result,
        source_checks=source_checks,
        limitations=[
            "Simulation metrics only; real GPU exporters and serving adapters require validation.",
            "Source checks use two observation windows; configuration revision history must cover that period.",
            "Simulation uses fixed inventory independently of KWOK worker replica count.",
        ],
    )
