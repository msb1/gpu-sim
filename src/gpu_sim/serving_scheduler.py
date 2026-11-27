"""KServe submission and Kueue placement observations for the KWOK lab."""

import json
import os
import subprocess

NAMESPACE = "gpu-serving"
QUEUE_LABEL = "kueue.x-k8s.io/queue-name"
SERVICE = "inferenceservices.serving.kserve.io"
QUEUE = "localqueues.kueue.x-k8s.io"


def kube(
    context: str, *args: str, payload: dict | None = None, namespace: str = NAMESPACE
) -> dict:
    result = subprocess.run(
        ["kubectl", "--context", context, "-n", namespace, *args, "-o", "json"],
        input=json.dumps(payload) if payload is not None else None,
        capture_output=True,
        text=True,
        timeout=25,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "kubectl failed")
    return json.loads(result.stdout) if result.stdout.strip() else {}


def audit() -> dict:
    context = os.getenv("AGENT_KUBE_CONTEXT")
    if not context:
        return {"available": False, "reason": "AGENT_KUBE_CONTEXT is unset"}
    result = {"available": True, "context": context, "namespace": NAMESPACE}
    for key, resource in (
        ("services", SERVICE),
        ("queues", QUEUE),
        ("workloads", "workloads.kueue.x-k8s.io"),
        ("pods", "pods"),
        ("deployments", "deployments.apps"),
        ("cluster_queues", "clusterqueues.kueue.x-k8s.io"),
        ("routes", "httproutes.gateway.networking.k8s.io"),
        ("gateways", "gateways.gateway.networking.k8s.io"),
    ):
        try:
            items = kube(
                context,
                "get",
                resource,
                namespace="kserve" if key == "gateways" else NAMESPACE,
            )["items"]
        except (RuntimeError, OSError, subprocess.TimeoutExpired) as error:
            result[key] = {"error": str(error)}
            result["available"] = False
            continue
        result[key] = [
            {
                "name": item["metadata"]["name"],
                "owners": item["metadata"].get("ownerReferences", []),
                "queue": item["metadata"].get("labels", {}).get(QUEUE_LABEL)
                or item.get("spec", {}).get("queueName"),
                "scheduler": item.get("spec", {}).get("schedulerName"),
                "gates": item.get("spec", {}).get("schedulingGates", []),
                "node": item.get("spec", {}).get("nodeName"),
                "conditions": item.get("status", {}).get("conditions", []),
                "spec": {
                    key: value
                    for key, value in item.get("spec", {}).items()
                    if key
                    in {
                        "resourceGroups",
                        "clusterQueue",
                        "hostnames",
                        "parentRefs",
                        "listeners",
                        "replicas",
                    }
                },
                "status": {
                    key: value
                    for key, value in item.get("status", {}).items()
                    if key
                    in {
                        "url",
                        "parents",
                        "addresses",
                        "listeners",
                        "admission",
                        "pendingWorkloads",
                        "admittedWorkloads",
                        "flavorsReservation",
                        "readyReplicas",
                        "phase",
                    }
                },
                "service": item["metadata"]
                .get("labels", {})
                .get("serving.kserve.io/inferenceservice"),
                "gpu_requests": [
                    container.get("resources", {})
                    .get("requests", {})
                    .get("nvidia.com/gpu", "0")
                    for container in item.get("spec", {}).get("containers", [])
                ],
            }
            for item in items
        ]
    return result


def plan_service(name: str, gpus: int) -> dict:
    context = os.environ.get("AGENT_KUBE_CONTEXT")
    if not context:
        raise RuntimeError("Set AGENT_KUBE_CONTEXT on the coordinator")
    queue = kube(context, "get", QUEUE, NAMESPACE)
    payload = {
        "apiVersion": "serving.kserve.io/v1beta1",
        "kind": "InferenceService",
        "metadata": {
            "name": name,
            "namespace": NAMESPACE,
            "labels": {QUEUE_LABEL: NAMESPACE},
            "annotations": {
                "serving.kserve.io/deploymentMode": "RawDeployment",
                "serving.kserve.io/autoscalerClass": "hpa",
            },
        },
        "spec": {
            "predictor": {
                "minReplicas": 1,
                "maxReplicas": 1,
                "schedulerName": "default-scheduler",
                "nodeSelector": {"type": "kwok"},
                "tolerations": [
                    {
                        "key": "kwok.x-k8s.io/node",
                        "operator": "Equal",
                        "value": "fake",
                        "effect": "NoSchedule",
                    }
                ],
                "containers": [
                    {
                        "name": "kserve-container",
                        "image": "registry.k8s.io/pause:3.10",
                        "resources": {
                            key: {"nvidia.com/gpu": str(gpus)}
                            for key in ("requests", "limits")
                        },
                    }
                ],
            }
        },
    }
    return {
        "kind": "kserve_submit",
        "context": context,
        "queue_uid": queue["metadata"]["uid"],
        "payload": payload,
        "impact": f"Submit one simulated predictor requesting {gpus} GPUs to Kueue",
    }


def submit(plan: dict) -> dict:
    context = plan["context"]
    if context != os.getenv("AGENT_KUBE_CONTEXT"):
        raise RuntimeError("Configured Kubernetes context changed")
    queue = kube(context, "get", QUEUE, NAMESPACE)
    if queue["metadata"]["uid"] != plan["queue_uid"]:
        raise RuntimeError("Approved LocalQueue changed")
    created = kube(context, "create", "-f", "-", payload=plan["payload"])
    return {
        "name": created["metadata"]["name"],
        "uid": created["metadata"]["uid"],
        "state": "submitted",
        "status_source": "SchedulerAgent audit",
    }
