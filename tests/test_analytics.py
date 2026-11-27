import io
import json
import unittest
from unittest.mock import patch

from prometheus_client import CollectorRegistry, generate_latest

from gpu_sim.analytics.benchmark import (
    BenchmarkCriteria,
    BenchmarkSample,
    analyze_benchmark,
)
from gpu_sim.analytics.diagnostics import diagnose, map_node, recovery
from gpu_sim.analytics.measurements import Evidence, ObservationWindow, queries
from gpu_sim.analytics.prometheus import Prometheus, QueryResult, Sample, Scope, Series
from gpu_sim.metrics import DCGMCollector, Metrics
from gpu_sim.simulator import Fault, Simulator, Workload


def metric(value, *, node="tray-3", quality="ok", start=0, end=10):
    return QueryResult(
        query="test",
        evaluated_at=end,
        start=start,
        step=5,
        unit="test",
        quality=quality,
        series=[
            Series(
                labels={"node": node},
                samples=[Sample(timestamp=t, value=value) for t in (0, 5, 10)],
            )
        ],
    )


class PrometheusTests(unittest.TestCase):
    def test_instant_zero_and_missing_and_error(self):
        client = Prometheus("http://localhost:9090")
        body = {
            "status": "success",
            "data": {
                "resultType": "vector",
                "result": [{"metric": {"node": "tray-3"}, "value": [100, "0"]}],
            },
        }
        with patch(
            "gpu_sim.analytics.prometheus.urlopen",
            return_value=io.BytesIO(json.dumps(body).encode()),
        ):
            result = client.query("x", 100, "ratio", max_age=5)
        self.assertEqual(result.quality, "ok")
        self.assertEqual(result.series[0].samples[0].value, 0)
        body["data"]["result"] = []
        with patch(
            "gpu_sim.analytics.prometheus.urlopen",
            return_value=io.BytesIO(json.dumps(body).encode()),
        ):
            self.assertEqual(
                client.query("x", 100, "ratio", max_age=5).quality, "missing"
            )
        with patch("gpu_sim.analytics.prometheus.urlopen", side_effect=TimeoutError):
            self.assertEqual(
                client.query("x", 100, "ratio", max_age=5).quality, "error"
            )

    def test_range_stale_and_insufficient_history(self):
        client = Prometheus("http://localhost:9090")
        body = {
            "status": "success",
            "data": {
                "resultType": "matrix",
                "result": [{"metric": {}, "values": [[9, "1"], [10, "2"]]}],
            },
        }
        with patch(
            "gpu_sim.analytics.prometheus.urlopen",
            return_value=io.BytesIO(json.dumps(body).encode()),
        ):
            self.assertEqual(
                client.query("x", 20, "x", start=0, step=1, max_age=2).quality, "stale"
            )
        with patch(
            "gpu_sim.analytics.prometheus.urlopen",
            return_value=io.BytesIO(json.dumps(body).encode()),
        ):
            self.assertEqual(
                client.query("x", 10, "x", start=0, step=1, max_age=2).quality,
                "insufficient_history",
            )

    def test_scope_and_stack_matrix(self):
        dcgm = Scope(job="dcgm", instance="a:9400")
        serving = Scope(job="serving", instance="a:8002")
        window = ObservationWindow(end=100, duration=30, step=2, max_age=5)
        for backend in ("vllm", "tensorrt_llm"):
            for wrapper in (True, False):
                q = queries(
                    {
                        "backend_engine": backend,
                        "triton_wrapper": wrapper,
                        "model_name": 'model"x',
                    },
                    dcgm,
                    serving,
                    window,
                )
                self.assertIn('instance="a:8002"', q["waiting_queue"][0])
                self.assertIn('model\\"x', q["waiting_queue"][0])
                self.assertIn(
                    "vllm:" if backend == "vllm" else "nv_trt_llm_",
                    q["waiting_queue"][0],
                )
                self.assertNotIn("nv_inference_pending", q["waiting_queue"][0])
                self.assertIn("resets(", q["completion_counter_resets"][0])


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.cluster = {
            "nodes": [
                {"name": "gb200-tray-03", "virtual": True, "unschedulable": True}
            ],
            "errors": {},
        }
        self.measurements = {
            name: metric(0)
            for name in (
                "xid",
                "throttle",
                "nvlink_crc_rate",
                "waiting_queue",
                "sm_occupancy",
                "tensor_active",
                "dram_active",
            )
        }
        self.measurements["gpu_temp"] = metric(60)
        for name in ("gpu_temp", "xid", "throttle", "nvlink_crc_rate"):
            self.measurements[f"{name}_source_age"] = metric(1)
        self.evidence = Evidence(
            context={"hardware_assumptions": {"thermal_throttle_bitmask": 32}},
            cluster=self.cluster,
            measurements=self.measurements,
        )

    def test_thermal_xid_crc_and_mapping(self):
        self.evidence.measurements.update(
            gpu_temp=metric(91),
            throttle=metric(32),
            xid=metric(79),
            nvlink_crc_rate=metric(1),
        )
        findings = diagnose(self.evidence)
        self.assertEqual(
            {f.kind for f in findings}
            & {"thermal_throttle", "xid_hardware_fault", "nvlink_crc_growth"},
            {"thermal_throttle", "xid_hardware_fault", "nvlink_crc_growth"},
        )
        self.assertEqual(map_node("tray-3", self.cluster), "gb200-tray-03")
        self.assertIsNone(map_node("tray-4", self.cluster))

    def test_crc_only_finding_clear_and_missing_evidence(self):
        self.evidence.measurements["nvlink_crc_rate"] = metric(120)
        findings = [f for f in diagnose(self.evidence) if f.kind == "nvlink_crc_growth"]
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].kubernetes_node, "gb200-tray-03")
        self.assertEqual(findings[0].evidence["crc_errors_per_s"], 120)
        self.evidence.measurements["nvlink_crc_rate"] = metric(0)
        self.assertFalse(
            any(f.kind == "nvlink_crc_growth" for f in diagnose(self.evidence))
        )
        self.evidence.measurements["nvlink_crc_rate"] = metric(120, quality="stale")
        self.assertFalse(
            any(f.kind == "nvlink_crc_growth" for f in diagnose(self.evidence))
        )

    def test_xid_reports_per_gpu_and_does_not_infer_peer_source(self):
        self.evidence.context["fault"] = {"target_gpu": 14}
        self.assertNotIn("fault", self.evidence.for_agent().context)
        self.assertIn("fault", self.evidence.context)
        self.evidence.measurements["xid"] = metric(149)
        self.evidence.measurements["xid_gpu"] = QueryResult(
            query="gpu",
            evaluated_at=10,
            start=0,
            step=5,
            unit="code",
            quality="ok",
            series=[
                Series(
                    labels={"node": "tray-3", "gpu": "14", "UUID": "GPU-tray-3-14"},
                    samples=[Sample(timestamp=t, value=149) for t in (0, 5, 10)],
                )
            ],
        )
        self.evidence.measurements["nvlink_xid_events"] = QueryResult(
            query="event",
            evaluated_at=10,
            start=0,
            step=5,
            unit="event",
            quality="ok",
            series=[
                Series(
                    labels={
                        "node": "tray-3",
                        "gpu": "14",
                        "xid": "149",
                        "peer_type": "unknown",
                    },
                    samples=[Sample(timestamp=t, value=1) for t in (0, 5, 10)],
                )
            ],
        )
        finding = next(
            f for f in diagnose(self.evidence) if f.kind == "xid_hardware_fault"
        )
        self.assertEqual(finding.evidence["reporting_gpus"][0]["gpu"], "14")
        self.assertEqual(finding.evidence["nvlink_events"][0]["peer_type"], "unknown")
        self.assertIn("not proof", finding.interpretation)
        self.evidence.measurements["xid_gpu"].quality = "stale"
        finding = next(
            f for f in diagnose(self.evidence) if f.kind == "xid_hardware_fault"
        )
        self.assertEqual(finding.evidence["reporting_gpus"], [])

    def test_peer_observation_requires_matching_fresh_event(self):
        self.evidence.measurements["xid"] = metric(149)
        event_labels = {
            "node": "tray-3",
            "gpu": "14",
            "xid": "149",
            "link_id": "l0",
            "peer_type": "gpu",
            "peer_id": "18",
        }
        peer_labels = {
            "reporter_node": "tray-3",
            "reporter_gpu": "14",
            "link_id": "l0",
            "peer_type": "gpu",
            "peer_id": "18",
            "observation": "link_unhealthy",
        }
        for name, labels in (
            ("nvlink_xid_events", event_labels),
            ("nvlink_peer_observations", peer_labels),
        ):
            self.evidence.measurements[name] = QueryResult(
                query=name,
                evaluated_at=10,
                start=0,
                step=5,
                unit="event",
                quality="ok",
                series=[
                    Series(
                        labels=labels,
                        samples=[Sample(timestamp=t, value=1) for t in (0, 5, 10)],
                    )
                ],
            )
        finding = next(
            f for f in diagnose(self.evidence) if f.kind == "xid_hardware_fault"
        )
        self.assertEqual(finding.evidence["peer_observations"][0]["peer_id"], "18")
        self.evidence.measurements["nvlink_peer_observations"].quality = "stale"
        finding = next(
            f for f in diagnose(self.evidence) if f.kind == "xid_hardware_fault"
        )
        self.assertEqual(finding.evidence["peer_observations"], [])
        self.evidence.measurements["nvlink_peer_observations"].quality = "ok"
        self.evidence.measurements["nvlink_peer_observations"].series[0].labels[
            "link_id"
        ] = "other"
        finding = next(
            f for f in diagnose(self.evidence) if f.kind == "xid_hardware_fault"
        )
        self.assertEqual(finding.evidence["peer_observations"], [])

    def test_simulator_crc_counter_stops_but_survives_clear(self):
        simulator = Simulator(1)
        simulator.workload = Workload(rps=0)
        exporter = Metrics(simulator)
        simulator.inject(Fault(type="nvlink_degraded", target_node="tray-3"))
        simulator.tick()
        first = dict(simulator.crc)
        simulator.tick()
        self.assertEqual(len(first), 4)
        self.assertTrue(
            all(simulator.crc[key] > value > 0 for key, value in first.items())
        )
        retained = dict(simulator.crc)
        simulator.inject(Fault(type="clear"))
        simulator.tick()
        self.assertEqual(simulator.crc, retained)
        self.assertTrue(all(row["xid"] == 0 for row in simulator.telemetry))
        exporter.publish(simulator)
        registry = CollectorRegistry()
        registry.register(DCGMCollector(exporter))
        output = generate_latest(registry).decode()
        self.assertIn("sim_nvlink_crc_errors_total", output)

    def test_recovery_needs_sustained_data_and_known_remediation(self):
        args = {"min_window": 10, "temp_limit": 80, "max_gap": 5}
        self.assertEqual(
            recovery(self.evidence, "tray-3", known_remediation=True, **args).status,
            "eligible_evidence",
        )
        self.assertEqual(
            recovery(self.evidence, "tray-3", known_remediation=False, **args).status,
            "insufficient",
        )
        self.evidence.measurements["xid"] = metric(0, quality="stale")
        self.assertEqual(
            recovery(self.evidence, "tray-3", known_remediation=True, **args).status,
            "insufficient",
        )
        self.evidence.measurements["xid"] = metric(79)
        self.assertEqual(
            recovery(self.evidence, "tray-3", known_remediation=True, **args).status,
            "unhealthy",
        )

    def test_recovery_rejects_stale_sources_and_unrelated_cordon(self):
        args = {"min_window": 10, "temp_limit": 80, "max_gap": 5}
        self.evidence.measurements["gpu_temp_source_age"] = metric(8)
        self.assertEqual(
            recovery(self.evidence, "tray-3", known_remediation=True, **args).status,
            "insufficient",
        )
        self.evidence.measurements["gpu_temp_source_age"] = metric(1)
        self.cluster["nodes"][0]["unschedulable"] = False
        self.assertEqual(
            recovery(self.evidence, "tray-3", known_remediation=True, **args).status,
            "insufficient",
        )

    def test_recovery_rejects_thermal_mask_and_missing_semantics(self):
        args = {"min_window": 10, "temp_limit": 80, "max_gap": 5}
        self.evidence.measurements["throttle"] = metric(32)
        self.assertEqual(
            recovery(self.evidence, "tray-3", known_remediation=True, **args).status,
            "unhealthy",
        )
        self.evidence.context = {}
        self.assertEqual(
            recovery(self.evidence, "tray-3", known_remediation=True, **args).status,
            "insufficient",
        )


class BenchmarkTests(unittest.TestCase):
    def test_knee_and_missing_and_tick_resolution(self):
        criteria = BenchmarkCriteria(
            min_throughput_fraction=0.9, max_queue_growth_rps=1, ttft_slo_seconds=2
        )

        def sample(rps, completed, queue=0, ttft=1):
            return BenchmarkSample(
                offered_rps=rps,
                completed_rps=completed,
                queue_growth_rps=queue,
                ttft_p95_seconds=ttft,
                measurement_start=0,
                measurement_end=10,
            )

        self.assertEqual(
            analyze_benchmark([sample(10, 10), sample(20, 10, 4)], criteria).status,
            "bracketed",
        )
        self.assertEqual(
            analyze_benchmark([sample(10, 10)], criteria).status, "lower_bound"
        )
        self.assertEqual(
            analyze_benchmark([sample(10, 1)], criteria).status, "all_failed"
        )
        self.assertEqual(
            analyze_benchmark([sample(10, 10, ttft=None)], criteria).status,
            "insufficient",
        )
        criteria.simulation_tick_seconds = 1
        criteria.ttft_slo_seconds = 0.2
        self.assertEqual(
            analyze_benchmark([sample(10, 10)], criteria).status, "insufficient"
        )
