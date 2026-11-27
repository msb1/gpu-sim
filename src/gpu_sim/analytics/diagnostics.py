"""Evidence-based diagnostic and recovery checks."""

import re
from itertools import pairwise
from typing import Any, Literal

from pydantic import BaseModel, Field

from gpu_sim.analytics.measurements import Evidence
from gpu_sim.analytics.prometheus import QueryResult


class Finding(BaseModel):
    kind: str
    node: str | None = None
    kubernetes_node: str | None = None
    evidence: dict[str, Any] = Field(default_factory=dict)
    interpretation: str


def last(result: QueryResult, **labels: str) -> list[tuple[dict[str, str], float]]:
    if result.quality != "ok":
        return []
    return [
        (s.labels, s.samples[-1].value)
        for s in result.series
        if all(s.labels.get(k) == v for k, v in labels.items())
        and s.samples[-1].value is not None
    ]


def map_node(node: str, cluster: dict[str, Any]) -> str | None:
    match = re.fullmatch(r"tray-(\d+)", node)
    if not match or cluster.get("errors", {}).get("nodes"):
        return None
    target = f"gb200-tray-{int(match.group(1)):02d}"
    return (
        target
        if any(
            n.get("name") == target and n.get("virtual")
            for n in cluster.get("nodes", [])
        )
        else None
    )


def diagnose(evidence: Evidence) -> list[Finding]:
    m, cluster = evidence.measurements, evidence.cluster
    mask = (evidence.context.get("hardware_assumptions") or {}).get(
        "thermal_throttle_bitmask"
    )
    findings = []
    gpu_xids = m.get("xid_gpu")
    events = m.get("nvlink_xid_events")
    peer_observations = m.get("nvlink_peer_observations")
    for labels, xid in last(m["xid"]):
        node = labels.get("node")
        if node and xid > 0:
            findings.append(
                Finding(
                    kind="xid_hardware_fault",
                    node=node,
                    kubernetes_node=map_node(node, cluster),
                    evidence={
                        "xid": xid,
                        "reporting_gpus": [
                            {"gpu": gl.get("gpu"), "UUID": gl.get("UUID"), "xid": value}
                            for gl, value in (
                                last(gpu_xids, node=node) if gpu_xids else []
                            )
                            if value > 0
                        ],
                        "nvlink_events": [
                            gl
                            for gl, value in (last(events, node=node) if events else [])
                            if value > 0 and gl.get("xid") == str(int(xid))
                        ],
                        "peer_observations": [
                            pl
                            for pl, value in (
                                last(peer_observations, reporter_node=node)
                                if peer_observations
                                else []
                            )
                            if value > 0
                            and any(
                                el.get("gpu") == pl.get("reporter_gpu")
                                and el.get("link_id") == pl.get("link_id")
                                and el.get("peer_type") == pl.get("peer_type")
                                and el.get("peer_id") == pl.get("peer_id")
                                and el.get("xid") == str(int(xid))
                                for el, event_value in (
                                    last(events, node=node) if events else []
                                )
                                if event_value > 0
                            )
                        ],
                    },
                    interpretation="XID reported; determine whether it is current or historical. A reporting GPU or observed peer is not proof of the fault source.",
                )
            )
    throttles = {labels.get("node"): value for labels, value in last(m["throttle"])}
    for labels, temp in last(m["gpu_temp"]):
        node = labels.get("node")
        throttle = throttles.get(node)
        if (
            node
            and temp > 85
            and throttle is not None
            and isinstance(mask, int)
            and mask > 0
            and int(throttle) & mask
        ):
            findings.append(
                Finding(
                    kind="thermal_throttle",
                    node=node,
                    kubernetes_node=map_node(node, cluster),
                    evidence={"temperature_c": temp, "throttle_mask": throttle},
                    interpretation="Simulated thermal-throttle signal; verify platform bit semantics for real hardware.",
                )
            )
    for labels, value in last(m["nvlink_crc_rate"]):
        node = labels.get("node")
        if node and value > 0:
            findings.append(
                Finding(
                    kind="nvlink_crc_growth",
                    node=node,
                    kubernetes_node=map_node(node, cluster),
                    evidence={"crc_errors_per_s": value},
                    interpretation="CRC growth is a link-health signal; corroborate with transfer or collective degradation.",
                )
            )
    queues = last(m["waiting_queue"])
    if len(queues) == 1 and queues[0][1] > 15:
        for metric, role, threshold, kind in [
            ("tensor_active", "prefill", 0.82, "prefill_compute_pressure"),
            ("dram_active", "decode", 0.85, "decode_bandwidth_pressure"),
        ]:
            for labels, value in last(m[metric], role=role):
                if value > threshold:
                    findings.append(
                        Finding(
                            kind=kind,
                            node=labels.get("node"),
                            evidence={metric: value, "waiting_queue": queues[0][1]},
                            interpretation="Potential bottleneck; correlate remaining pool, cache and workload evidence.",
                        )
                    )
    if len(queues) == 1 and queues[0][1] <= 15:
        for labels, value in last(m["sm_occupancy"]):
            if value < 0.25:
                findings.append(
                    Finding(
                        kind="low_occupancy",
                        node=labels.get("node"),
                        evidence={"sm_occupancy": value, "waiting_queue": queues[0][1]},
                        interpretation="Possible small batch or insufficient parallelism; not proof of memory starvation.",
                    )
                )
    return findings


class RecoveryResult(BaseModel):
    status: Literal["eligible_evidence", "insufficient", "unhealthy"]
    node: str
    explanation: str


def recovery(
    evidence: Evidence,
    dcgm_node: str,
    *,
    known_remediation: bool,
    min_window: float,
    temp_limit: float,
    max_gap: float,
) -> RecoveryResult:
    target = map_node(dcgm_node, evidence.cluster)
    if (
        not known_remediation
        or not target
        or evidence.cluster.get("errors", {}).get("nodes")
    ):
        return RecoveryResult(
            status="insufficient",
            node=dcgm_node,
            explanation="Known remediation and verified node identity are required.",
        )
    node = next(n for n in evidence.cluster["nodes"] if n["name"] == target)
    if not node.get("unschedulable"):
        return RecoveryResult(
            status="insufficient", node=dcgm_node, explanation="Node is not cordoned."
        )
    mask = (evidence.context.get("hardware_assumptions") or {}).get(
        "thermal_throttle_bitmask"
    )
    if not isinstance(mask, int) or mask <= 0:
        return RecoveryResult(
            status="insufficient",
            node=dcgm_node,
            explanation="Thermal throttle mask is unavailable.",
        )
    checks = [
        ("gpu_temp", lambda v: v < temp_limit),
        ("xid", lambda v: v == 0),
        ("throttle", lambda v: int(v) & mask == 0),
        ("nvlink_crc_rate", lambda v: v == 0),
    ]
    for name, healthy in checks:
        result = evidence.measurements[name]
        ages = evidence.measurements.get(f"{name}_source_age")
        age_series = (
            [s for s in ages.series if s.labels.get("node") == dcgm_node]
            if ages
            else []
        )
        if (
            ages is None
            or ages.quality != "ok"
            or ages.start is None
            or ages.evaluated_at - ages.start < min_window
            or len(age_series) != 1
            or len(age_series[0].samples) < 2
            or age_series[0].samples[0].timestamp > ages.start + max_gap
            or ages.evaluated_at - age_series[0].samples[-1].timestamp > max_gap
            or any(
                b.timestamp - a.timestamp > max_gap
                for a, b in pairwise(age_series[0].samples)
            )
            or any(
                s.value is None or s.value < 0 or s.value > max_gap
                for s in age_series[0].samples
            )
        ):
            return RecoveryResult(
                status="insufficient",
                node=dcgm_node,
                explanation=f"{name}: underlying source freshness is unverified.",
            )
        series = [s for s in result.series if s.labels.get("node") == dcgm_node]
        if (
            result.quality != "ok"
            or result.start is None
            or result.evaluated_at - result.start < min_window
            or len(series) != 1
        ):
            return RecoveryResult(
                status="insufficient",
                node=dcgm_node,
                explanation=f"{name}: missing, stale or insufficient window.",
            )
        samples = series[0].samples
        if (
            len(samples) < 2
            or samples[0].timestamp > result.start + max_gap
            or result.evaluated_at - samples[-1].timestamp > max_gap
            or any(b.timestamp - a.timestamp > max_gap for a, b in pairwise(samples))
        ):
            return RecoveryResult(
                status="insufficient",
                node=dcgm_node,
                explanation=f"{name}: incomplete observation window.",
            )
        if any(s.value is None or not healthy(s.value) for s in samples):
            return RecoveryResult(
                status="unhealthy",
                node=dcgm_node,
                explanation=f"{name}: unhealthy during observation window.",
            )
    return RecoveryResult(
        status="eligible_evidence",
        node=dcgm_node,
        explanation="Sustained healthy evidence; human approval and revalidation are still required.",
    )
