"""Prometheus HTTP queries and typed telemetry results."""

import json
import math
import re
from itertools import pairwise
from typing import Literal
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field


class Sample(BaseModel):
    timestamp: float
    value: float | None


class Series(BaseModel):
    labels: dict[str, str]
    samples: list[Sample]


class QueryResult(BaseModel):
    query: str
    evaluated_at: float
    start: float | None = None
    step: float | None = None
    unit: str
    quality: Literal[
        "ok", "missing", "stale", "insufficient_history", "error", "unsupported"
    ]
    series: list[Series] = Field(default_factory=list)
    error: str | None = None
    warnings: list[str] = Field(default_factory=list)


class Scope(BaseModel):
    model_config = ConfigDict(extra="forbid")
    job: str = Field(min_length=1)
    instance: str = Field(min_length=1)
    labels: dict[str, str] = Field(default_factory=dict)

    def selector(self, **labels: str) -> str:
        if set(labels) & {"job", "instance"} or set(self.labels) & {"job", "instance"}:
            raise ValueError("Scope job and instance cannot be overridden")
        matchers = self.labels | {"job": self.job, "instance": self.instance} | labels
        if any(not re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*", key) for key in matchers):
            raise ValueError("Invalid Prometheus label name")
        return (
            "{"
            + ",".join(
                f"{key}={json.dumps(value)}" for key, value in sorted(matchers.items())
            )
            + "}"
        )


class Prometheus:
    def __init__(self, url: str, timeout: float = 5) -> None:
        parsed = urlsplit(url)
        if (
            parsed.scheme not in ("http", "https")
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "PROM_URL must be an HTTP(S) URL without credentials, query, or fragment"
            )
        self.url = url.rstrip("/")
        self.timeout = timeout

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
        result = QueryResult(
            query=query,
            evaluated_at=at,
            start=start,
            step=step,
            unit=unit,
            quality="error",
        )
        if max_age <= 0 or (
            start is not None and (start >= at or step is None or step <= 0)
        ):
            result.error = "Invalid observation window, step or freshness limit"
            return result
        params = {"query": query}
        if start is None:
            params["time"] = str(at)
            endpoint = "query"
        else:
            params.update(start=str(start), end=str(at), step=str(step))
            endpoint = "query_range"
        try:
            request = Request(
                f"{self.url}/api/v1/{endpoint}?{urlencode(params)}",
                headers={"Accept": "application/json"},
            )
            with urlopen(request, timeout=self.timeout) as response:
                data = json.load(response)
            if data.get("status") != "success":
                result.error = str(data.get("error", "Prometheus query failed"))
                return result
            payload = data["data"]
            if payload["resultType"] not in ("vector", "matrix"):
                result.quality = "unsupported"
                result.error = "Unsupported Prometheus result type"
                return result
            for item in payload["result"]:
                pairs = item.get("values", [item["value"]] if "value" in item else [])
                samples = []
                for timestamp, raw in pairs:
                    value = float(raw)
                    samples.append(
                        Sample(
                            timestamp=float(timestamp),
                            value=value if math.isfinite(value) else None,
                        )
                    )
                result.series.append(
                    Series(labels=item.get("metric", {}), samples=samples)
                )
            result.warnings = data.get("warnings", [])
            if not result.series or any(not s.samples for s in result.series):
                result.quality = "missing"
            elif any(
                s.samples[-1].value is None
                or at - s.samples[-1].timestamp > max_age
                or s.samples[-1].timestamp > at + max_age
                for s in result.series
            ):
                result.quality = "stale"
            elif start is not None and any(
                len(s.samples) < 2
                or s.samples[0].timestamp > start + (step or 0) + 0.001
                or any(v.value is None for v in s.samples)
                or any(
                    b.timestamp - a.timestamp > (step or 0) * 1.01 + 0.001
                    for a, b in pairwise(s.samples)
                )
                for s in result.series
            ):
                result.quality = "insufficient_history"
                reasons = []
                if any(
                    len(s.samples) < 2
                    or s.samples[0].timestamp > start + (step or 0) + 0.001
                    for s in result.series
                ):
                    reasons.append("incomplete history")
                if any(any(v.value is None for v in s.samples) for s in result.series):
                    reasons.append("non-finite samples")
                if any(
                    any(
                        b.timestamp - a.timestamp > (step or 0) * 1.01 + 0.001
                        for a, b in pairwise(s.samples)
                    )
                    for s in result.series
                ):
                    reasons.append("sample gaps")
                result.error = "Range query has " + ", ".join(reasons)
            else:
                result.quality = "ok"
        except (
            HTTPError,
            URLError,
            OSError,
            TimeoutError,
            ValueError,
            KeyError,
            TypeError,
        ) as exc:
            result.error = (
                f"Prometheus request or response failed: {type(exc).__name__}"
            )
        return result
