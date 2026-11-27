import asyncio
import unittest
from importlib import import_module
from itertools import product

from fastapi import HTTPException
from prometheus_client import CollectorRegistry, generate_latest
from pydantic import ValidationError

main = import_module("gpu_sim.main")
from gpu_sim.metrics import DCGMCollector, Metrics, ServingCollector
from gpu_sim.simulator import Config, Fault, Request, Simulator, Workload


def configuration(total: int = 72, **changes: object) -> Config:
    presets = {
        1: ("single_1gpu", 1, 1, 1),
        8: ("node_8gpu", 2, 6, 2),
        16: ("cluster_16gpu", 4, 12, 4),
        72: ("rack_72gpu_nvl72", 24, 48, 8),
    }
    topology, p, d, tp = presets[total]
    return Config.model_validate(
        {
            "topology": topology,
            "params_billions": 1,
            "prefill_gpus": p,
            "decode_gpus": d,
            "tp_size": tp,
            "disaggregated": total > 1,
            "prefix_cache_hit_rate": 0,
        }
        | changes
    )


class V2Tests(unittest.TestCase):
    def test_inventory_groups(self) -> None:
        for total in (1, 8, 16, 72):
            sim = Simulator(1)
            sim.reset(configuration(total))
            self.assertEqual(sum(len(g.gpus) for g in sim.replicas), total)
            self.assertEqual(sim.state()["capacity_mode"], "fixed_inventory")
        self.assertEqual(len(sim.pool("prefill")), 3)
        self.assertEqual(len(sim.pool("decode")), 6)
        for k, group in enumerate(sim.state()["replicas"]):
            self.assertEqual(group["nodes"], [f"tray-{2 * k}", f"tray-{2 * k + 1}"])

    def test_stack_matrix_completes_and_exports(self) -> None:
        for backend, wrapper, disagg in product(
            ("vllm", "tensorrt_llm"), (False, True), (False, True)
        ):
            with self.subTest(backend=backend, wrapper=wrapper, disagg=disagg):
                sim = Simulator(1)
                sim.reset(
                    configuration(
                        8,
                        backend_engine=backend,
                        triton_wrapper=wrapper,
                        disaggregated=disagg,
                    )
                )
                sim.workload = Workload(rps=0)
                metrics = Metrics(sim)
                sim.waiting.append(Request(-1, 16, 4))
                for _ in range(5):
                    sim.tick()
                    metrics.publish(sim)
                self.assertEqual(sim.completed, 1)
                self.assertEqual(sim.generated, 4)
                self.assertEqual(sim.nixl_bytes > 0, disagg)
                reg = CollectorRegistry()
                reg.register(ServingCollector(metrics))
                text = generate_latest(reg).decode()
                self.assertEqual("vllm:num_requests_waiting" in text, backend == "vllm")
                self.assertEqual(
                    "nv_trt_llm_request_metrics" in text, backend == "tensorrt_llm"
                )
                self.assertEqual("nv_inference_pending_request_count" in text, wrapper)
                self.assertIn("time_to_first_token_seconds_count", text)
                self.assertIn("sim_completed_requests_total 1.0", text)

    def test_prefix_cache_and_replica_budgets(self) -> None:
        sim = Simulator(1)
        sim.reset(configuration(8, prefix_cache_hit_rate=0.5, chunked_prefill_size=10))
        sim.workload = Workload(rps=0)
        requests = [Request(-1, 100, 2) for _ in range(2)]
        sim.waiting.extend(requests)
        sim.tick()
        self.assertEqual(sim.prefix_hits, 100)
        self.assertEqual(sum(r.prefilled - r.cached for r in requests), 10)
        self.assertTrue(all(r.generated == 0 for r in requests))

    def test_admission_uses_multiple_replicas(self) -> None:
        sim = Simulator(1)
        sim.reset(configuration())
        sim.workload = Workload(rps=0)
        sim.waiting.extend(Request(-1, 100, 100) for _ in range(12))
        sim.tick()
        self.assertEqual(len({r.prefill_replica for r in sim.active + sim.transfer}), 3)
        self.assertEqual(len({r.decode_replica for r in sim.active + sim.transfer}), 6)

    def test_fault_targets_clear_and_counters(self) -> None:
        for kind, factor in [
            ("thermal_throttle", 0.65),
            ("nvlink_degraded", 0.55),
            ("xid_error", 0.5),
        ]:
            sim = Simulator(1)
            sim.workload = Workload(rps=0)
            sim.inject(Fault(type=kind, target_node="tray-3"))
            sim.tick()
            self.assertEqual(sim.fault_factor, factor)
            rows = [r for r in sim.telemetry if r["node"] == "tray-3"]
            healthy = [r for r in sim.telemetry if r["node"] != "tray-3"]
            self.assertTrue(
                all(
                    r["xid"] == 0 and r["throttle"] == 0 and r["crc"] == 0
                    for r in healthy
                )
            )
            if kind == "thermal_throttle":
                self.assertTrue(
                    all(89 <= r["temp"] <= 93 and r["throttle"] == 32 for r in rows)
                )
            elif kind == "nvlink_degraded":
                self.assertTrue(all(r["crc"] > 0 for r in rows))
            else:
                self.assertTrue(all(r["xid"] == 79 and r["util"] == 0 for r in rows))
            previous = dict(sim.crc)
            sim.inject(Fault(type="clear"))
            sim.tick()
            self.assertEqual(sim.fault_factor, 1)
            self.assertEqual(sim.crc, previous)
            self.assertTrue(all(r["throttle"] == r["xid"] == 0 for r in sim.telemetry))

    def test_invalid_configuration_and_faults(self) -> None:
        for patch in [
            {"prefill_gpus": 36, "decode_gpus": 36},
            {"active_params_billions": 1001},
            {"thermal_throttle_factor": 0},
            {"tp_size": 7},
            {"prefix_cache_hit_rate": float("nan")},
        ]:
            with self.assertRaises(ValidationError):
                Config.model_validate(Config().model_dump() | patch)
        sim = Simulator()
        for target in [
            {"target_node": "tray-03"},
            {"target_gpu": 71, "target_node": "tray-0"},
        ]:
            with self.assertRaises(ValueError):
                sim.inject(Fault(type="thermal_throttle", **target))
        sim.reset(configuration(1))
        with self.assertRaises(ValueError):
            sim.inject(Fault(type="xid_error", target_gpu=1))

    def test_metrics_switch_and_crc(self) -> None:
        sim = Simulator(1)
        metrics = Metrics(sim)
        sim.inject(Fault(type="nvlink_degraded", target_gpu=0))
        sim.tick()
        metrics.publish(sim)
        reg = CollectorRegistry()
        reg.register(DCGMCollector(metrics))
        self.assertIn(
            "DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_TOTAL{",
            generate_latest(reg).decode(),
        )
        sim.reset(configuration(8, backend_engine="vllm", triton_wrapper=False))
        metrics.publish(sim)
        reg = CollectorRegistry()
        reg.register(ServingCollector(metrics))
        text = generate_latest(reg).decode()
        self.assertNotIn("nv_trt_llm_request_metrics", text)
        self.assertNotIn("nv_inference_request_success", text)

    def test_api_patch_validation_and_reset(self) -> None:
        sim = Simulator()
        sim.waiting.append(Request(-1, 100, 10))
        result = asyncio.run(main.workload(main.WorkloadPatch(rps=80), sim))
        self.assertEqual(len(sim.waiting), 1)
        self.assertEqual(result["workload"]["prompt_tokens"], 4096)
        with self.assertRaises(HTTPException):
            asyncio.run(main.config(main.ConfigPatch(tp_size=7), sim))
        self.assertEqual(len(sim.waiting), 1)
        result = asyncio.run(
            main.config(main.ConfigPatch(chunked_prefill_size=1024), sim)
        )
        self.assertTrue(result["requests_reset"])
        self.assertEqual(len(sim.waiting), 0)
        self.assertEqual(result["config"]["decode_gpus"], 48)

    def test_rack_and_inter_node_transfers(self) -> None:
        for total in (16, 72):
            sim = Simulator(1)
            sim.reset(configuration(total))
            sim.workload = Workload(rps=0)
            sim.waiting.extend(Request(-1, 4, 10) for _ in range(4))
            sim.tick()
            self.assertGreater(sim.nixl_bytes, 0)
            if total == 72:
                self.assertEqual(sim.ib_tx, 0)
            else:
                self.assertGreater(sim.ib_tx, 0)


class PhysicsRegressionTests(unittest.TestCase):
    def test_compute_roofline_and_fault_multipliers(self) -> None:
        counts = []
        for kind in ("none", "thermal_throttle", "nvlink_degraded", "xid_error"):
            sim = Simulator(1)
            sim.reset(
                configuration(8, params_billions=70, chunked_prefill_size=1000000)
            )
            sim.workload = Workload(rps=0)
            req = Request(-1, 1000000, 1)
            sim.waiting.append(req)
            if kind != "none":
                sim.inject(Fault(type=kind, target_gpu=0))
            sim.tick()
            counts.append(req.prefilled)
        for actual, factor in zip(counts[1:], (0.65, 0.55, 0.50)):
            self.assertAlmostEqual(actual / counts[0], factor, delta=0.001)

    def test_model_resident_weights_use_tp_not_pool(self) -> None:
        with self.assertRaises(ValidationError):
            configuration(
                72,
                gpu_profile="ampere_a100_80gb",
                params_billions=671,
                precision="fp16",
            )

    def test_fifo_oversized_head_does_not_bypass(self) -> None:
        sim = Simulator(1)
        sim.reset(configuration(1, kv_bytes_per_token=10000000))
        sim.workload = Workload(rps=0)
        sim.waiting.extend([Request(-1, 1000000, 1000000), Request(-1, 1, 1)])
        sim.tick()
        self.assertEqual(len(sim.waiting), 2)
        self.assertFalse(sim.active)

    def test_cache_does_not_reduce_full_kv_reservation(self) -> None:
        sim = Simulator(1)
        sim.reset(
            configuration(1, prefix_cache_hit_rate=0.95, kv_bytes_per_token=10000000)
        )
        sim.workload = Workload(rps=0)
        sim.waiting.append(Request(-1, 1000000, 1000000))
        sim.tick()
        self.assertEqual(len(sim.waiting), 1)
        self.assertEqual(sim.prefix_hits, 0)

    def test_decode_bandwidth_is_fault_scaled(self) -> None:
        progress = []
        for fault in (False, True):
            sim = Simulator(1)
            sim.reset(configuration(8, params_billions=70))
            sim.workload = Workload(rps=0)
            req = Request(-1, 1, 100000)
            sim.waiting.append(req)
            sim.tick()
            sim.tick()
            sim.tick()
            if fault:
                sim.inject(Fault(type="xid_error", target_gpu=0))
            before = req.generated
            sim.tick()
            progress.append(req.generated - before)
        self.assertGreater(progress[0], 0)
        self.assertAlmostEqual(progress[1] / progress[0], 0.5, delta=0.01)

    def test_counter_totals_survive_reset_and_backend_round_trip(self) -> None:
        sim = Simulator(1)
        metrics = Metrics(sim)
        for backend in ("vllm", "tensorrt_llm", "vllm"):
            sim.reset(configuration(8, backend_engine=backend))
            metrics.publish(sim)
            sim.workload = Workload(rps=0)
            sim.waiting.append(Request(-1, 2, 2))
            for _ in range(5):
                sim.tick()
                metrics.publish(sim)
        self.assertEqual(sim.completed, 3)
        self.assertEqual(metrics.totals[("vllm", sim.config.model_name)][0], 4)

    def test_clear_preserves_inflight_requests(self) -> None:
        sim = Simulator(1)
        sim.waiting.append(Request(-1, 1000000, 1000000))
        sim.inject(Fault(type="xid_error", target_gpu=0))
        sim.inject(Fault(type="clear"))
        self.assertEqual(len(sim.waiting), 1)
        for dt in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                sim.tick(dt)
