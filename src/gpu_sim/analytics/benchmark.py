"""Analysis of completed, externally approved benchmark measurements."""

from typing import Literal

from pydantic import BaseModel, Field


class BenchmarkSample(BaseModel):
    offered_rps: float = Field(ge=0)
    completed_rps: float | None = Field(default=None, ge=0)
    queue_growth_rps: float | None = None
    ttft_p95_seconds: float | None = Field(default=None, ge=0)
    measurement_start: float
    measurement_end: float
    valid: bool = True


class BenchmarkCriteria(BaseModel):
    min_throughput_fraction: float = Field(gt=0, le=1)
    max_queue_growth_rps: float = Field(ge=0)
    ttft_slo_seconds: float = Field(gt=0)
    simulation_tick_seconds: float | None = Field(default=None, gt=0)


class BenchmarkAnalysis(BaseModel):
    status: Literal["bracketed", "lower_bound", "all_failed", "insufficient"]
    sustainable_rps: float | None = None
    first_failure_rps: float | None = None
    samples: list[dict]
    limitations: list[str]


def analyze_benchmark(
    samples: list[BenchmarkSample], criteria: BenchmarkCriteria
) -> BenchmarkAnalysis:
    limitations = []
    if (
        criteria.simulation_tick_seconds
        and criteria.ttft_slo_seconds < criteria.simulation_tick_seconds
    ):
        limitations.append("TTFT SLO is below the simulator time resolution.")
    rows = []
    for sample in samples:
        usable = (
            sample.valid
            and sample.measurement_end > sample.measurement_start
            and all(
                value is not None
                for value in (
                    sample.completed_rps,
                    sample.queue_growth_rps,
                    sample.ttft_p95_seconds,
                )
            )
        )
        if not usable or limitations:
            state = "unavailable"
        elif (
            sample.completed_rps
            >= sample.offered_rps * criteria.min_throughput_fraction
            and sample.queue_growth_rps <= criteria.max_queue_growth_rps
            and sample.ttft_p95_seconds <= criteria.ttft_slo_seconds
        ):
            state = "sustainable"
        else:
            state = "failed"
        rows.append(sample.model_dump() | {"assessment": state})
    rows.sort(key=lambda row: row["offered_rps"])
    if not rows or any(row["assessment"] == "unavailable" for row in rows):
        return BenchmarkAnalysis(
            status="insufficient",
            samples=rows,
            limitations=limitations + ["All sweep points require valid measurements."],
        )
    first_failure = next(
        (i for i, row in enumerate(rows) if row["assessment"] == "failed"), None
    )
    if first_failure is None:
        return BenchmarkAnalysis(
            status="lower_bound",
            sustainable_rps=rows[-1]["offered_rps"],
            samples=rows,
            limitations=limitations,
        )
    if any(row["assessment"] == "sustainable" for row in rows[first_failure + 1 :]):
        return BenchmarkAnalysis(
            status="insufficient",
            samples=rows,
            limitations=["Non-monotonic results require further investigation."],
        )
    if first_failure == 0:
        return BenchmarkAnalysis(
            status="all_failed",
            first_failure_rps=rows[0]["offered_rps"],
            samples=rows,
            limitations=limitations,
        )
    return BenchmarkAnalysis(
        status="bracketed",
        sustainable_rps=rows[first_failure - 1]["offered_rps"],
        first_failure_rps=rows[first_failure]["offered_rps"],
        samples=rows,
        limitations=limitations,
    )
