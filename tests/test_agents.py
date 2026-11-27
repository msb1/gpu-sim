"""HTTP workflow integration tests for the LangGraph coordinator."""

import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from gpu_sim import agents
from gpu_sim.analytics.measurements import Evidence
from gpu_sim.simulator import Config


class AgentWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.state = {
            "config_revision_seconds": 1,
            "config": Config().model_dump(),
            "workload": {"rps": 5, "prompt_tokens": 4096, "output_tokens": 512},
            "fault": {"type": "none"},
        }
        self.writes = []
        agents.records.clear()
        self.addCleanup(agents.records.clear)
        self.enterContext(
            patch.dict("os.environ", {"AGENT_OPERATOR_TOKEN": "test-operator"})
        )
        self.enterContext(patch.object(agents, "sim_request", self.sim))
        self.client = self.enterContext(
            TestClient(agents.app, headers={"Authorization": "Bearer test-operator"})
        )

    def sim(self, path: str, payload: dict | None = None) -> dict:
        if payload is not None:
            self.writes.append((path, payload))
            key = "/workload" if path == "/workload" else "/config"
            self.state[key[1:]].update(payload)
        return deepcopy(self.state) | {"requests_reset": path == "/config"}

    def test_llm_summary_unavailable_does_not_raise(self) -> None:
        with patch.dict("os.environ", {"OPENAI_BASE_URL": "", "MAIN_MODEL": ""}):
            summary = agents.summarize_with_llm("profiler", {"findings": []})
        self.assertEqual(summary["status"], "unavailable")

    def test_llm_summary_uses_model_response(self) -> None:
        from io import BytesIO

        response = BytesIO(b'{"choices":[{"message":{"content":"Evidence summary"}}]}')
        with (patch.dict("os.environ", {"OPENAI_BASE_URL": "http://localhost/v1", "MAIN_MODEL": "test", "OPENAI_API_KEY": "local"}),
              patch.object(agents, "urlopen", return_value=response) as send):
            summary = agents.summarize_with_llm("profiler", {"findings": []})
        self.assertEqual(summary, {"status": "ok", "text": "Evidence summary"})
        self.assertEqual(send.call_args.args[0].full_url, "http://localhost/v1/chat/completions")

    def test_summary_identifies_shared_queue_without_changing_proposal(self) -> None:
        finding = {"node": "tray-1", "evidence": {"waiting_queue": 42}}
        result = {"incident_report": {"findings": [finding]},
                  "actions": [{"evidence": [finding]}]}
        summary = agents.summary_evidence(result, "test")
        self.assertEqual(summary["findings"][0]["evidence"], {"shared_serving_waiting_queue": 42})
        self.assertEqual(summary["proposals"][0]["evidence"][0]["evidence"],
                         {"shared_serving_waiting_queue": 42})
        self.assertEqual(finding["evidence"], {"waiting_queue": 42})

    def plan(self) -> dict:
        response = self.client.post(
            "/benchmark",
            json={
                "start_rps": 2,
                "end_rps": 5,
                "step_rps": 3,
                "settle_sec": 10,
            },
        )
        self.assertEqual(response.status_code, 200)
        return response.json()

    def decide(self, item: dict, decision: str = "approve"):
        return self.client.post(
            "/approvals/decide",
            json={
                "id": item["id"],
                "decision": decision,
            },
        )

    def test_approval_runs_sweep_and_restores_once(self) -> None:
        def observe():
            return SimpleNamespace(
                context=deepcopy(self.state),
                measurements={
                    "waiting_queue": SimpleNamespace(start=10, evaluated_at=70)
                },
            )

        def scalar(observed, name):
            rps = observed.context["workload"]["rps"]
            return {
                "offered_rps": rps,
                "completed_rps": rps,
                "queue_growth_rps": 0,
                "ttft_p95": 1,
                "completion_counter_resets": 0,
            }[name]

        item = self.plan()
        self.assertEqual(item["plan"]["hold_seconds"], 130)
        self.assertEqual(self.writes, [])
        with (
            patch.object(agents, "evidence", observe),
            patch.object(agents, "scalar", scalar),
            patch.object(agents.time, "sleep"),
        ):
            outcome = self.decide(item).json()
        self.assertEqual(outcome["status"], "executed")
        self.assertEqual(outcome["result"]["status"], "lower_bound")
        self.assertTrue(outcome["result"]["workload_restored"])
        self.assertEqual([payload["rps"] for _, payload in self.writes], [2, 5, 5])
        self.assertEqual(self.decide(item).status_code, 409)

    def test_reject_and_changed_baseline_make_no_writes(self) -> None:
        self.assertEqual(TestClient(agents.app).get("/approvals").status_code, 401)
        self.assertEqual(
            self.decide(self.plan(), "reject").json()["status"], "rejected"
        )
        item = self.plan()
        self.state["workload"]["rps"] = 9
        self.assertEqual(self.decide(item).json()["status"], "failed")
        self.assertEqual(self.writes, [])

    def test_profiler_to_scheduler_rebalance(self) -> None:
        self.state["config"] = Config(
            topology="rack_72gpu_nvl72",
            gpu_profile="blackwell_gb200_nvl72",
            disaggregated=True,
            tp_size=8,
            prefill_gpus=24,
            decode_gpus=48,
        ).model_dump()
        observed = Evidence(context=deepcopy(self.state), cluster={}, measurements={})
        finding = SimpleNamespace(
            model_dump=lambda: {"kind": "prefill_compute_pressure"}
        )
        with (
            patch.object(agents, "evidence", return_value=observed),
            patch.object(agents, "diagnose", return_value=[finding]),
        ):
            profiler = self.client.post("/run", json={"role": "profiler"}).json()
            self.assertEqual(profiler["approvals"], [])
            scheduler = self.client.post("/run", json={"role": "scheduler"}).json()
            self.assertEqual(self.writes, [])
            self.assertEqual(
                self.decide(scheduler["approvals"][0]).json()["status"], "executed"
            )
        self.assertEqual(
            self.writes, [("/config", {"prefill_gpus": 32, "decode_gpus": 40})]
        )

    def test_kserve_submission_requires_approval(self) -> None:
        commands = []

        def kube(context: str, *args: str, payload: dict | None = None) -> dict:
            commands.append((context, args, payload))
            if args[0] == "get":
                return {"metadata": {"uid": "queue-1"}}
            return {"metadata": {"name": "serving-smoke", "uid": "service-1"}}

        with (
            patch.dict("os.environ", {"AGENT_KUBE_CONTEXT": "kind-test"}),
            patch.object(agents.serving_scheduler, "kube", side_effect=kube),
        ):
            proposal = self.client.post(
                "/serving", json={"name": "serving-smoke", "gpus": 3}
            ).json()
            self.assertEqual([args[0] for _, args, _ in commands], ["get"])
            outcome = self.decide(proposal).json()
            self.assertEqual(outcome["status"], "executed")
            self.assertEqual(outcome["result"]["state"], "submitted")
            self.assertEqual(self.decide(proposal).status_code, 409)
        self.assertEqual([args[0] for _, args, _ in commands], ["get", "get", "create"])
        manifest = commands[-1][2]
        self.assertEqual(
            manifest["metadata"]["annotations"]["serving.kserve.io/autoscalerClass"],
            "hpa",
        )
        self.assertEqual(
            manifest["metadata"]["labels"]["kueue.x-k8s.io/queue-name"], "gpu-serving"
        )
        self.assertEqual(
            manifest["spec"]["predictor"]["schedulerName"], "default-scheduler"
        )

        resources = manifest["spec"]["predictor"]["containers"][0]["resources"]
        self.assertEqual(resources["requests"]["nvidia.com/gpu"], "3")
        self.assertEqual(resources["limits"]["nvidia.com/gpu"], "3")
