"""Read-only view of the local Kubernetes control plane."""

import asyncio
import json
import os
import ssl
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import BaseModel, Field

TOKEN = Path("/var/run/secrets/kubernetes.io/serviceaccount/token")
CA = Path("/var/run/secrets/kubernetes.io/serviceaccount/ca.crt")
PATHS = {
    "nodes": "/api/v1/nodes",
    "pods": "/api/v1/namespaces/default/pods",
    "allocation_pods": "/api/v1/pods",
    "events": "/api/v1/namespaces/default/events",
    "control_plane_pods": "/api/v1/namespaces/kube-system/pods",
    "deployments": "/apis/apps/v1/namespaces/default/deployments",
    "hpas": "/apis/autoscaling/v2/namespaces/default/horizontalpodautoscalers",
    "scaledobjects": "/apis/keda.sh/v1alpha1/namespaces/default/scaledobjects",
    "keda_deployments": "/apis/apps/v1/namespaces/keda/deployments",
}


class ClusterSnapshot(BaseModel):
    available: bool
    nodes: list[dict[str, Any]] = Field(default_factory=list)
    pods: list[dict[str, Any]] = Field(default_factory=list)
    allocation_pods: list[dict[str, Any]] = Field(default_factory=list)
    events: list[dict[str, Any]] = Field(default_factory=list)
    control_plane_pods: list[dict[str, Any]] = Field(default_factory=list)
    deployments: list[dict[str, Any]] = Field(default_factory=list)
    hpas: list[dict[str, Any]] = Field(default_factory=list)
    scaledobjects: list[dict[str, Any]] = Field(default_factory=list)
    keda_deployments: list[dict[str, Any]] = Field(default_factory=list)
    errors: dict[str, str] = Field(default_factory=dict)


def fetch_resource(path: str) -> list[dict[str, Any]]:
    host = os.environ.get("KUBERNETES_SERVICE_HOST")
    port = os.environ.get("KUBERNETES_SERVICE_PORT_HTTPS", "443")
    if not host or not TOKEN.is_file() or not CA.is_file():
        raise RuntimeError("Kubernetes service account is not available")
    authority = f"[{host}]" if ":" in host else host
    req = Request(
        f"https://{authority}:{port}{path}",
        headers={
            "Authorization": f"Bearer {TOKEN.read_text().strip()}",
            "Accept": "application/json",
        },
    )
    try:
        with urlopen(
            req, timeout=5, context=ssl.create_default_context(cafile=str(CA))
        ) as response:
            return json.load(response)["items"]
    except HTTPError as exc:
        raise RuntimeError(f"Kubernetes API returned HTTP {exc.code}") from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise RuntimeError("Kubernetes API is unreachable") from exc


def metadata(item: dict[str, Any]) -> dict[str, Any]:
    meta = item.get("metadata", {})
    return {
        "name": meta.get("name"),
        "namespace": meta.get("namespace"),
        "created": meta.get("creationTimestamp"),
    }


def summarize(name: str, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for item in items:
        meta, spec, status = (
            metadata(item),
            item.get("spec", {}),
            item.get("status", {}),
        )
        resource_name = meta["name"] or ""
        if name == "nodes":
            conditions = status.get("conditions", [])
            result.append(
                meta
                | {
                    "virtual": item.get("metadata", {})
                    .get("annotations", {})
                    .get("kwok.x-k8s.io/node")
                    == "fake",
                    "ready": next(
                        (
                            c.get("status")
                            for c in conditions
                            if c.get("type") == "Ready"
                        ),
                        "Unknown",
                    ),
                    "unschedulable": spec.get("unschedulable", False),
                    "capacity": status.get("capacity", {}),
                    "allocatable": status.get("allocatable", {}),
                }
            )
        elif name == "control_plane_pods":
            if not resource_name.startswith(
                (
                    "etcd-",
                    "kube-apiserver-",
                    "kube-controller-manager-",
                    "kube-scheduler-",
                    "kwok-controller-",
                    "coredns-",
                )
            ):
                continue
            result.append(
                meta
                | {
                    "node": spec.get("nodeName"),
                    "phase": status.get("phase", "Unknown"),
                    "conditions": status.get("conditions", []),
                }
            )
        elif name in ("pods", "allocation_pods"):
            if name == "pods" and item.get("metadata", {}).get("labels", {}).get("app") not in (
                "dynamo-inference-workers",
                "gpu-simulator-engine",
            ):
                continue
            result.append(
                meta
                | {
                    "app": item.get("metadata", {}).get("labels", {}).get("app"),
                    "node": spec.get("nodeName"),
                    "phase": status.get("phase", "Unknown"),
                    "reason": status.get("reason"),
                    "conditions": status.get("conditions", []),
                    "gpu_requests": sum(
                        int(
                            c.get("resources", {})
                            .get("requests", {})
                            .get("nvidia.com/gpu", 0)
                        )
                        for c in spec.get("containers", [])
                    ),
                }
            )
        elif name == "events":
            involved = item.get("involvedObject", {})
            if not involved.get("name", "").startswith(
                (
                    "dynamo-inference-workers",
                    "gpu-simulator-engine",
                    "keda-hpa-dynamo-worker-autoscaler",
                )
            ):
                continue
            result.append(
                meta
                | {
                    "reason": item.get("reason"),
                    "message": item.get("message"),
                    "type": item.get("type"),
                    "object": involved.get("name"),
                    "last_seen": item.get("lastTimestamp")
                    or item.get("eventTime")
                    or meta["created"],
                    "count": item.get("count"),
                }
            )
        elif name in ("deployments", "keda_deployments"):
            if name == "deployments" and resource_name not in (
                "gpu-simulator-engine",
                "dynamo-inference-workers",
            ):
                continue
            result.append(
                meta
                | {
                    "desired": spec.get("replicas", 1),
                    "ready": status.get("readyReplicas", 0),
                    "available": status.get("availableReplicas", 0),
                    "updated": status.get("updatedReplicas", 0),
                    "conditions": status.get("conditions", []),
                }
            )
        elif name == "hpas":
            if resource_name != "keda-hpa-dynamo-worker-autoscaler":
                continue
            result.append(
                meta
                | {
                    "target": spec.get("scaleTargetRef"),
                    "min": spec.get("minReplicas"),
                    "max": spec.get("maxReplicas"),
                    "current": status.get("currentReplicas"),
                    "desired": status.get("desiredReplicas"),
                    "current_metrics": status.get("currentMetrics", []),
                    "metrics": spec.get("metrics", []),
                    "conditions": status.get("conditions", []),
                    "last_scale_time": status.get("lastScaleTime"),
                }
            )
        elif name == "scaledobjects":
            if resource_name != "dynamo-worker-autoscaler":
                continue
            result.append(
                meta
                | {
                    "target": spec.get("scaleTargetRef"),
                    "min": spec.get("minReplicaCount"),
                    "max": spec.get("maxReplicaCount"),
                    "polling_interval": spec.get("pollingInterval"),
                    "cooldown_period": spec.get("cooldownPeriod"),
                    "triggers": [
                        {
                            "type": trigger.get("type"),
                            "threshold": trigger.get("metadata", {}).get("threshold"),
                        }
                        for trigger in spec.get("triggers", [])
                    ],
                    "conditions": status.get("conditions", []),
                    "last_active_time": status.get("lastActiveTime"),
                }
            )
    if name == "events":
        result.sort(key=lambda row: row.get("last_seen") or "", reverse=True)
        return result[:30]
    return result


async def cluster_snapshot() -> ClusterSnapshot:
    results = await asyncio.gather(
        *(asyncio.to_thread(fetch_resource, path) for path in PATHS.values()),
        return_exceptions=True,
    )
    data: dict[str, Any] = {"errors": {}}
    for name, result in zip(PATHS, results):
        if isinstance(result, BaseException):
            data["errors"][name] = (
                str(result)
                if isinstance(result, RuntimeError)
                else "Unable to read resource"
            )
            data[name] = []
        else:
            data[name] = summarize(name, result)
    data["available"] = not bool(data["errors"])
    return ClusterSnapshot(**data)
