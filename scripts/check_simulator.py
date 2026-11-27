"""User-run live smoke, offline scenario matrix and deterministic soak checks."""

import argparse
import json
import time
import tracemalloc
from itertools import product
from pathlib import Path
from typing import Any, TextIO
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from gpu_sim.simulator import PROFILES, TOPOLOGIES, Config, Fault, Simulator, Workload


def request(url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload).encode()
    req = Request(url, data=body, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=10) as response:
        return json.load(response)


def invariants(sim: Simulator, max_queued: int) -> None:
    state = sim.state()
    assert len(sim.telemetry) == sim.config.total_gpus, "Inventory telemetry mismatch"
    assert 0 <= state["kv_ratio"] <= 1, "KV capacity exceeded"
    if len(sim.waiting) > max_queued:
        raise AssertionError(
            f"Queue safety limit reached at simulated second {sim.time:g}: "
            f"waiting={len(sim.waiting)}, limit={max_queued}, "
            f"offered_rps={sim.workload.rps:g}, completed={sim.completed}, "
            f"active={len(sim.active)}, transferring={len(sim.transfer)}, "
            f"kv_ratio={state['kv_ratio']:.4f}. "
            "Inspect throughput and prefill token budgets before increasing the limit."
        )
    assert sim.generated >= 0 and sim.completed >= 0
    for group in sim.replicas:
        assert (
            sim.kv(sim.members(group, group.role != "prefill"), True) <= sim.kv_capacity
        ), "Replica over-admission"
    for row in sim.telemetry:
        assert all(0 <= row[key] <= 1 for key in ("sm", "occupancy", "tensor", "dram"))
        assert row["fb"] >= 0 and row["power"] >= 0
    for req in sim.active + sim.transfer:
        assert 0 <= req.prefilled <= req.prompt and 0 <= req.generated <= req.output
    if TOPOLOGIES[sim.config.topology]["nvl72_domain"]:
        assert sim.ib_tx == 0, "Unexpected intra-rack IB traffic"


def smoke(args: argparse.Namespace, out: Any) -> None:
    baseline = request(args.url + "/state")
    try:
        state = request(args.url + "/workload", {"rps": 0})
        assert state["config"] == baseline["config"]
        assert (
            state["workload"]["prompt_tokens"] == baseline["workload"]["prompt_tokens"]
        )
        for kind in ("thermal_throttle", "nvlink_degraded", "xid_error"):
            request(args.url + "/fault", {"type": kind, "target_gpu": 0})
            time.sleep(2)
            state = request(args.url + "/state")
            rows = [r for r in state["telemetry"] if r["gpu"] == "0"]
            assert len(rows) == 1
            row = rows[0]
            assert state["fault_factor"] == state["config"][kind + "_factor"]
            if kind == "thermal_throttle":
                assert row["temp"] >= 89 and row["throttle"] != 0
            elif kind == "nvlink_degraded":
                assert row["crc"] > 0
            else:
                assert row["xid"] == 79 and row["util"] == 0
            out.write(
                json.dumps({"test": "live_fault", "kind": kind, "row": row}) + "\n"
            )
            request(args.url + "/fault", {"type": "clear"})
        try:
            request(
                args.url + "/fault", {"type": "xid_error", "target_node": "invalid"}
            )
        except HTTPError as exc:
            assert exc.code == 422
        else:
            raise AssertionError("Invalid fault accepted")
        for endpoint in (args.dcgm, args.serving):
            with urlopen(endpoint + "/metrics", timeout=10) as response:
                text = response.read().decode()
            assert "# TYPE" in text
        out.write(json.dumps({"test": "live_smoke", "status": "passed"}) + "\n")
    finally:
        request(args.url + "/fault", {"type": "clear"})
        try:
            request(args.url + "/workload", baseline["workload"])
        finally:
            if baseline["fault"]["type"] != "none":
                request(args.url + "/fault", baseline["fault"])


def matrix(args: argparse.Namespace, out: Any) -> None:
    presets = {
        "single_1gpu": (1, 1, 1),
        "node_8gpu": (2, 6, 2),
        "cluster_16gpu": (4, 12, 4),
        "rack_72gpu_nvl72": (24, 48, 8),
    }
    for topology, hardware, precision, backend, wrapper, disagg in product(
        TOPOLOGIES,
        PROFILES,
        ("fp16", "fp8", "fp4"),
        ("vllm", "tensorrt_llm"),
        (False, True),
        (False, True),
    ):
        if topology == "single_1gpu" and disagg:
            continue
        p, d, tp = presets[topology]
        sim = Simulator(args.seed)
        sim.reset(
            Config(
                topology=topology,
                gpu_profile=hardware,
                params_billions=1,
                prefill_gpus=p,
                decode_gpus=d,
                tp_size=tp,
                precision=precision,
                backend_engine=backend,
                triton_wrapper=wrapper,
                disaggregated=disagg,
            )
        )
        sim.workload = Workload(rps=5, prompt_tokens=32, output_tokens=16)
        for tick in range(args.ticks):
            if tick % 20 == 5:
                sim.inject(Fault(type="nvlink_degraded", target_gpu=0))
            elif tick % 20 == 10:
                sim.inject(Fault(type="clear"))
            sim.tick()
            invariants(sim, args.max_queued)
        assert sim.completed > 0, "No completed requests"
        out.write(
            json.dumps(
                {
                    "test": "matrix",
                    "config": sim.config.model_dump(),
                    "completed": sim.completed,
                    "generated": sim.generated,
                }
            )
            + "\n"
        )
        out.flush()


class SoakLog:
    def __init__(self, out: TextIO, sample_every: int) -> None:
        self.out = out
        self.sample_every = sample_every
        self.previous_fault = "none"
        self.fault_ticks = dict.fromkeys(
            ("none", "thermal_throttle", "nvlink_degraded", "xid_error"), 0
        )
        self.transitions = 0
        self.interval_ticks = self.ttft_count = 0
        self.ttft_sum = self.ttft_max = 0.0
        self.started = time.monotonic()

    def write(self, record: dict[str, Any]) -> None:
        self.out.write(json.dumps(record) + "\n")
        self.out.flush()

    def record(self, sim: Simulator, tick: int, final: bool = False) -> None:
        fault = sim.fault.type
        self.fault_ticks[fault] += 1
        self.interval_ticks += 1
        self.ttft_count += len(sim.ttft)
        self.ttft_sum += sum(sim.ttft)
        self.ttft_max = max(self.ttft_max, max(sim.ttft, default=0))
        sim.ttft.clear()
        changed = fault != self.previous_fault
        if changed:
            self.transitions += 1
            self.write(
                {
                    "test": "fault_transition",
                    "tick": tick,
                    "from_fault": self.previous_fault,
                    "to_fault": fault,
                    "fault_factor": sim.fault_factor,
                    "target_gpu": 0,
                    "target_telemetry": next(
                        row for row in sim.telemetry if row["gpu"] == "0"
                    ),
                }
            )
        self.previous_fault = fault
        if (tick - 1) % self.sample_every != 0 and not changed and not final:
            return
        current, peak = tracemalloc.get_traced_memory()
        self.write(
            {
                "test": "soak",
                "tick": tick,
                "elapsed_seconds": time.monotonic() - self.started,
                "waiting": len(sim.waiting),
                "active": len(sim.active),
                "transferring": len(sim.transfer),
                "generated": sim.generated,
                "completed": sim.completed,
                "kv_ratio": sim.state()["kv_ratio"],
                "observed_ttft_count": self.ttft_count,
                "observed_ttft_mean_seconds": self.ttft_sum / self.ttft_count
                if self.ttft_count
                else None,
                "observed_ttft_max_seconds": self.ttft_max if self.ttft_count else None,
                "observation_interval_ticks": self.interval_ticks,
                "fault": fault,
                "fault_ticks": dict(self.fault_ticks),
                "memory_bytes": current,
                "peak_memory_bytes": peak,
            }
        )
        self.interval_ticks = self.ttft_count = 0
        self.ttft_sum = self.ttft_max = 0.0

    def summary(self) -> None:
        self.write(
            {
                "test": "soak_summary",
                "fault_ticks": dict(self.fault_ticks),
                "fault_transitions": self.transitions,
                "total_ticks": sum(self.fault_ticks.values()),
            }
        )


def soak(args: argparse.Namespace, out: TextIO) -> None:
    sim = Simulator(args.seed)
    if args.config:
        sim.reset(Config.model_validate_json(args.config.read_text()))
    sim.workload = Workload(rps=args.rps)
    log = SoakLog(out, args.sample_every)
    log.write(
        {
            "test": "soak_configuration",
            "log_schema_version": 2,
            "config": sim.config.model_dump(),
            "workload": sim.workload.model_dump(),
            "seed": args.seed,
            "ticks": args.ticks,
            "max_queued": args.max_queued,
            "sample_every": args.sample_every,
            "fault_cycle": args.fault_cycle,
        }
    )
    tracemalloc.start()
    last_generated = last_completed = 0
    try:
        for tick in range(args.ticks):
            if args.fault_cycle and tick % 100 == 20:
                sim.inject(
                    Fault(
                        type=("thermal_throttle", "nvlink_degraded", "xid_error")[
                            (tick // 100) % 3
                        ],
                        target_gpu=0,
                    )
                )
            elif args.fault_cycle and tick % 100 == 40:
                sim.inject(Fault(type="clear"))
            sim.tick()
            invariants(sim, args.max_queued)
            assert sim.generated >= last_generated and sim.completed >= last_completed
            last_generated, last_completed = sim.generated, sim.completed
            log.record(sim, tick + 1, final=tick == args.ticks - 1)
        log.summary()
    finally:
        tracemalloc.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["smoke", "matrix", "soak"])
    parser.add_argument("--url", default="http://localhost:8080")
    parser.add_argument("--dcgm", default="http://localhost:9400")
    parser.add_argument("--serving", default="http://localhost:8002")
    parser.add_argument("--ticks", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--rps", type=float, default=45)
    parser.add_argument("--sample-every", type=int, default=10)
    parser.add_argument("--max-queued", type=int, default=100000)
    parser.add_argument("--fault-cycle", action="store_true")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.ticks < 5 or args.sample_every < 1 or args.max_queued < 1:
        parser.error("ticks must be >=5; sample-every and max-queued must be positive")
    with args.output.open("w") as out:
        try:
            {"smoke": smoke, "matrix": matrix, "soak": soak}[args.mode](args, out)
        except Exception as exc:
            out.write(json.dumps({"status": "failed", "error": str(exc)}) + "\n")
            raise
        out.write(json.dumps({"status": "passed", "mode": args.mode}) + "\n")
    print(f"{args.mode} passed; results: {args.output}")


if __name__ == "__main__":
    main()
