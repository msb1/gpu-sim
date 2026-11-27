import json
import unittest
from io import StringIO

from gpu_sim.simulator import Fault, Request, Simulator, Workload
from scripts.check_simulator import SoakLog, invariants


class CLIDiagnosticsTests(unittest.TestCase):
    def test_queue_limit_reports_state_without_dropping_requests(self) -> None:
        sim = Simulator(1)
        sim.workload = Workload(rps=0)
        sim.tick()
        sim.workload = Workload(rps=45)
        sim.waiting.extend([Request(0, 1, 1), Request(0, 1, 1)])
        with self.assertRaisesRegex(
            AssertionError, "waiting=2, limit=1, offered_rps=45"
        ):
            invariants(sim, 1)
        self.assertEqual(len(sim.waiting), 2)

    def test_transitions_cannot_be_hidden_by_sample_interval(self) -> None:
        sim = Simulator(1)
        sim.workload = Workload(rps=0)
        output = StringIO()
        log = SoakLog(output, sample_every=30)
        for tick, fault in enumerate(
            (
                "none",
                "thermal_throttle",
                "none",
                "nvlink_degraded",
                "none",
                "xid_error",
                "none",
            ),
            1,
        ):
            sim.inject(Fault() if fault == "none" else Fault(type=fault, target_gpu=0))
            sim.tick()
            log.record(sim, tick, final=tick == 7)
        log.summary()
        rows = [json.loads(line) for line in output.getvalue().splitlines()]
        transitions = [row for row in rows if row["test"] == "fault_transition"]
        self.assertEqual(len(transitions), 6)
        xid = next(row for row in transitions if row["to_fault"] == "xid_error")
        self.assertEqual(xid["target_telemetry"]["xid"], 79)
        self.assertEqual(xid["target_telemetry"]["util"], 0)
        cleared = transitions[-1]
        self.assertEqual(cleared["target_telemetry"]["xid"], 0)
        self.assertEqual(cleared["fault_factor"], 1)
        self.assertEqual(
            rows[-1]["fault_ticks"],
            {"none": 4, "thermal_throttle": 1, "nvlink_degraded": 1, "xid_error": 1},
        )
        self.assertEqual(rows[-1]["total_ticks"], 7)

    def test_ttft_counts_cover_entire_logging_interval(self) -> None:
        sim = Simulator(1)
        sim.workload = Workload(rps=0)
        output = StringIO()
        log = SoakLog(output, sample_every=30)
        for tick in range(1, 4):
            sim.tick()
            sim.ttft = [float(tick)]
            log.record(sim, tick, final=tick == 3)
        rows = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(sum(r["observed_ttft_count"] for r in rows), 3)
        self.assertEqual(rows[-1]["observed_ttft_mean_seconds"], 2.5)
        self.assertEqual(rows[-1]["observed_ttft_max_seconds"], 3)
        self.assertEqual(rows[-1]["observation_interval_ticks"], 2)
