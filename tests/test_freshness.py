"""Regression tests for scoped source quality and configuration epochs."""

import io
import json
import unittest
from typing import Any
from unittest.mock import patch

from gpu_sim.analytics.freshness import gpu_identities, revision_error, verify_source
from gpu_sim.analytics.measurements import ObservationWindow, collect
from gpu_sim.analytics.prometheus import Prometheus, QueryResult, Sample, Scope, Series
from gpu_sim.metrics import Metrics
from gpu_sim.simulator import Simulator


def result(values: list[float], *, labels: dict[str, str] | None = None) -> QueryResult:
    return QueryResult(
        query="test",
        evaluated_at=10,
        start=0,
        step=5,
        unit="s",
        quality="ok",
        series=[
            Series(
                labels=labels or {},
                samples=[
                    Sample(timestamp=i * 5, value=v) for i, v in enumerate(values)
                ],
            )
        ],
    )


class SourceChecks(unittest.TestCase):
    def test_inventory_and_source_coverage(self) -> None:
        state = {
            "inventory": {"total_gpus": 2, "gpus_per_node": 2, "nvl72_domain": False}
        }
        expected = gpu_identities(state)
        source = result(
            [1, 1, 1], labels={"node": "node-0", "gpu": "0", "UUID": "GPU-node-0-00"}
        )
        self.assertIsNotNone(verify_source(source, max_age=5, expected_gpus=expected))
        source.series.append(
            Series(
                labels={"node": "node-0", "gpu": "1", "UUID": "GPU-node-0-01"},
                samples=[Sample(timestamp=t, value=1) for t in (0, 5, 10)],
            )
        )
        self.assertIsNone(verify_source(source, max_age=5, expected_gpus=expected))
        source.series[1].samples[1].value = 8
        self.assertIsNotNone(verify_source(source, max_age=5, expected_gpus=expected))

    def test_revision_change_and_missing(self) -> None:
        self.assertIsNone(revision_error(result([42, 42, 42]), 42))
        self.assertIsNotNone(revision_error(result([41, 41, 42]), 42))
        self.assertIsNotNone(revision_error(result([42, 42, 42]), None))

    def test_range_gaps_and_nonfinite_samples_are_not_healthy(self) -> None:
        client = Prometheus("http://localhost:9090")
        body = {
            "status": "success",
            "data": {
                "resultType": "matrix",
                "result": [{"metric": {}, "values": [[0, "0"], [10, "1"]]}],
            },
        }
        with patch(
            "gpu_sim.analytics.prometheus.urlopen",
            return_value=io.BytesIO(json.dumps(body).encode()),
        ):
            outcome = client.query("q", 10, "x", start=0, step=5, max_age=10)
            self.assertEqual(outcome.quality, "insufficient_history")
            self.assertIn("gaps", outcome.error)
        body["data"]["result"][0]["values"] = [[0, "0"], [5, "NaN"], [10, "1"]]
        with patch(
            "gpu_sim.analytics.prometheus.urlopen",
            return_value=io.BytesIO(json.dumps(body).encode()),
        ):
            outcome = client.query("q", 10, "x", start=0, step=5, max_age=10)
            self.assertEqual(outcome.quality, "insufficient_history")
            self.assertIn("non-finite", outcome.error)

    def test_scope_cannot_override_target(self) -> None:
        with self.assertRaises(ValueError):
            Scope(
                job="correct", instance="right", labels={"instance": "wrong"}
            ).selector()
        with self.assertRaises(ValueError):
            Scope(job="correct", instance="right").selector(job="wrong")

    def test_revision_changes_on_reset_and_is_exported(self) -> None:
        sim = Simulator(1)
        first = sim.config_revision_seconds
        with patch("gpu_sim.simulator.time.time", return_value=first + 10):
            sim.reset(sim.config)
        self.assertEqual(sim.config_revision_seconds, first + 10)
        metrics = Metrics(sim)
        metrics.publish(sim)
        names = [m.name for m in metrics.serving()]
        self.assertIn("sim_config_revision_seconds", names)


class FakePrometheus:
    def __init__(
        self,
        revision: float,
        *,
        stale: bool = False,
        missing_gpu: bool = False,
        down: bool = False,
        reset: bool = False,
    ) -> None:
        self.revision, self.stale, self.missing_gpu, self.down, self.reset = (
            revision,
            stale,
            missing_gpu,
            down,
            reset,
        )

    def query(
        self,
        query: str,
        at: float,
        unit: str,
        *,
        start: float | None = None,
        step: float | None = None,
        max_age: float,
    ) -> QueryResult:
        n = int((at - start) / step) + 1 if start is not None else 1
        times = [start + i * step for i in range(n)] if start is not None else [at]
        gpu = (
            "time() - timestamp(DCGM_" in query
            or "time() - timestamp(sim_nvlink_crc_errors_total" in query
        )
        labels: list[dict[str, str]] = (
            [
                {"node": "node-0", "gpu": str(i), "UUID": f"GPU-node-0-{i:02d}"}
                for i in range(1 if self.missing_gpu else 2)
            ]
            if gpu
            else [{}]
        )
        value = 1.0
        if query.startswith("sum(resets("):
            value = 1.0 if self.reset else 0.0
        elif query.startswith("sim_config_revision_seconds"):
            value = self.revision
        elif query.startswith("up{"):
            value = 0.0 if self.down else 1.0
        elif self.stale and "timestamp(DCGM_FI_PROF_SM_OCCUPANCY" in query:
            value = 99.0
        return QueryResult(
            query=query,
            evaluated_at=at,
            start=start,
            step=step,
            unit=unit,
            quality="ok",
            series=[
                Series(
                    labels=label,
                    samples=[Sample(timestamp=t, value=value) for t in times],
                )
                for label in labels
            ],
        )


class CollectionChecks(unittest.TestCase):
    def test_measurement_gate_blocks_stale_missing_down_reset_and_revision(
        self,
    ) -> None:
        state: dict[str, Any] = {
            "config": {"model_name": "m", "backend_engine": "vllm"},
            "config_revision_seconds": 42,
            "inventory": {"total_gpus": 2, "gpus_per_node": 2, "nvl72_domain": False},
        }
        scope = Scope(job="test", instance="test:1")
        window = ObservationWindow(end=100, duration=10, step=5, max_age=5)

        def run(client: FakePrometheus) -> dict[str, QueryResult]:
            return collect(client, state, {}, scope, scope, window).measurements

        self.assertEqual(run(FakePrometheus(42))["sm_occupancy"].quality, "ok")
        self.assertNotEqual(
            run(FakePrometheus(42, stale=True))["sm_occupancy"].quality, "ok"
        )
        self.assertNotEqual(
            run(FakePrometheus(42, missing_gpu=True))["xid"].quality, "ok"
        )
        self.assertNotEqual(
            run(FakePrometheus(42, down=True))["waiting_queue"].quality, "ok"
        )
        self.assertNotEqual(
            run(FakePrometheus(42, reset=True))["completed_rps"].quality, "ok"
        )
        self.assertNotEqual(run(FakePrometheus(43))["waiting_queue"].quality, "ok")
