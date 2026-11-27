import unittest

from pydantic import ValidationError

from gpu_sim.simulator import Config, Request, Simulator, Workload


def legacy(**changes: object) -> Config:
    return Config.model_validate(
        {
            "topology": "node_8gpu",
            "gpu_profile": "hopper_h100_80gb",
            "backend_engine": "vllm",
            "model_name": "llama-3-70b",
            "params_billions": 70,
            "precision": "fp8",
            "prefill_gpus": 4,
            "decode_gpus": 4,
            "tp_size": 4,
            "prefix_cache_hit_rate": 0,
            "kv_bytes_per_token": 327680,
        }
        | changes
    )


class SimulatorTests(unittest.TestCase):
    def test_model_must_fit(self) -> None:
        with self.assertRaises(ValidationError):
            Config(gpu_profile="ampere_a100_80gb", precision="fp16")

    def test_prefill_then_decode_and_complete(self) -> None:
        sim = Simulator(1)
        sim.reset(legacy(disaggregated=False, chunked_prefill_size=100))
        sim.workload = Workload(rps=0)
        req = Request(-1, 150, 2)
        sim.waiting.append(req)
        sim.tick()
        self.assertEqual((req.prefilled, req.generated), (100, 0))
        sim.tick()
        self.assertEqual((req.prefilled, req.generated), (150, 0))
        sim.tick()
        self.assertEqual(sim.completed, 1)
        self.assertEqual(sim.generated, 2)
        self.assertEqual(len(sim.ttft), 1)

    def test_transfer_delay_and_workload_update(self) -> None:
        sim = Simulator(1)
        sim.reset(legacy(kv_transfer_latency_ms=1500))
        sim.workload = Workload(rps=0)
        req = Request(-1, 10, 2)
        sim.waiting.append(req)
        sim.tick()
        self.assertEqual(len(sim.transfer), 1)
        sim.workload = Workload(rps=0, prompt_tokens=10, output_tokens=2)
        sim.tick()
        sim.tick()
        self.assertEqual(req.generated, 0)
        sim.tick()
        sim.tick()
        self.assertEqual(sim.completed, 1)

    def test_reservation_blocks_admission(self) -> None:
        sim = Simulator(1)
        sim.reset(legacy())
        sim.workload = Workload(rps=0)
        sim.waiting.append(Request(0, 1000000, 1000000))
        sim.tick()
        self.assertEqual(len(sim.waiting), 1)
        self.assertFalse(sim.active)


if __name__ == "__main__":
    unittest.main()
