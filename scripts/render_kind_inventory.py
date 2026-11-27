"""Render KWOK inventory and worker autoscaler for the selected simulator topology."""

import argparse
import json
from typing import Any
from urllib.parse import urlsplit

PRESETS = {
    "node_8gpu": (1, 8, "gpu-node", 2, 6, 2),
    "cluster_16gpu": (2, 8, "gpu-node", 4, 12, 4),
    "rack_72gpu_nvl72": (18, 4, "gb200-tray", 24, 48, 8),
}


def inventory(topology: str) -> list[dict[str, Any]]:
    count, _, prefix, _, _, _ = PRESETS[topology]
    return [
        {
            "apiVersion": "v1",
            "kind": "Node",
            "metadata": {
                "name": f"{prefix}-{i:02d}",
                "annotations": {"kwok.x-k8s.io/node": "fake"},
                "labels": {
                    "type": "kwok",
                    "gpu-sim.io/inventory": topology.replace("_", "-"),
                    "kubernetes.io/hostname": f"{prefix}-{i:02d}",
                    "kubernetes.io/os": "linux",
                },
            },
            "spec": {
                "taints": [
                    {
                        "key": "kwok.x-k8s.io/node",
                        "value": "fake",
                        "effect": "NoSchedule",
                    }
                ]
            },
        }
        for i in range(count)
    ]


def node_status(topology: str) -> dict[str, Any]:
    _, gpus, _, _, _, _ = PRESETS[topology]
    resources = {
        "cpu": "64",
        "memory": "512Gi",
        "pods": "110",
        "nvidia.com/gpu": str(gpus),
    }
    return {"status": {"capacity": resources, "allocatable": resources}}


def worker_resources(topology: str, prometheus_url: str) -> list[dict[str, Any]]:
    url = urlsplit(prometheus_url)
    if (
        url.scheme not in ("http", "https")
        or not url.hostname
        or url.username
        or url.password
    ):
        raise ValueError(
            "Prometheus URL must be HTTP(S), with a host and without embedded credentials"
        )
    count, gpus, _, _, _, _ = PRESETS[topology]
    worker_gpu = 4 if topology == "rack_72gpu_nvl72" else 8
    replicas = 2 if topology == "rack_72gpu_nvl72" else 1
    deployment = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": "dynamo-inference-workers"},
        "spec": {
            "replicas": replicas,
            "selector": {"matchLabels": {"app": "dynamo-inference-workers"}},
            "template": {
                "metadata": {"labels": {"app": "dynamo-inference-workers"}},
                "spec": {
                    "nodeSelector": {
                        "gpu-sim.io/inventory": topology.replace("_", "-")
                    },
                    "tolerations": [
                        {
                            "key": "kwok.x-k8s.io/node",
                            "operator": "Exists",
                            "effect": "NoSchedule",
                        }
                    ],
                    "containers": [
                        {
                            "name": "virtual-worker",
                            "image": "registry.k8s.io/pause:3.10",
                            "resources": {
                                "requests": {"nvidia.com/gpu": str(worker_gpu)},
                                "limits": {"nvidia.com/gpu": str(worker_gpu)},
                            },
                        }
                    ],
                },
            },
        },
    }
    scaled = {
        "apiVersion": "keda.sh/v1alpha1",
        "kind": "ScaledObject",
        "metadata": {"name": "dynamo-worker-autoscaler"},
        "spec": {
            "scaleTargetRef": {"name": "dynamo-inference-workers"},
            "minReplicaCount": replicas,
            "maxReplicaCount": count * gpus // worker_gpu,
            "pollingInterval": 5,
            "cooldownPeriod": 30,
            "triggers": [
                {
                    "type": "prometheus",
                    "metadata": {
                        "serverAddress": prometheus_url,
                        "metricName": "simulated_request_queue",
                        "threshold": "15",
                        "query": 'sum(nv_trt_llm_request_metrics{request_type="context"}) or sum(vllm:num_requests_waiting) or vector(0)',
                    },
                }
            ],
        },
    }
    return [deployment, scaled]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("topology", choices=PRESETS)
    parser.add_argument("part", choices=["nodes", "status", "workers", "config"])
    parser.add_argument(
        "--prometheus-url", help="Existing Prometheus URL reachable from KEDA"
    )
    args = parser.parse_args()
    if args.part == "nodes":
        data = {"apiVersion": "v1", "kind": "List", "items": inventory(args.topology)}
    elif args.part == "status":
        data = node_status(args.topology)
    elif args.part == "workers":
        if not args.prometheus_url:
            parser.error("workers requires --prometheus-url")
        data = {
            "apiVersion": "v1",
            "kind": "List",
            "items": worker_resources(args.topology, args.prometheus_url),
        }
    else:
        _, _, _, p, d, tp = PRESETS[args.topology]
        data = {
            "topology": args.topology,
            "prefill_gpus": p,
            "decode_gpus": d,
            "tp_size": tp,
        }
    print(json.dumps(data))


if __name__ == "__main__":
    main()
