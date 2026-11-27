import asyncio
import os
import unittest
from unittest.mock import patch

from gpu_sim.cluster import PATHS, cluster_snapshot, summarize


class ClusterTests(unittest.TestCase):
    def test_allocation_pods_include_other_namespaces(self) -> None:
        pod = {
            "metadata": {"name": "predictor", "namespace": "gpu-serving"},
            "spec": {"nodeName": "gb200-tray-10", "containers": [
                {"resources": {"requests": {"nvidia.com/gpu": "4"}}}
            ]},
            "status": {"phase": "Running"},
        }
        result = summarize("allocation_pods", [pod])
        self.assertEqual((result[0]["namespace"], result[0]["node"], result[0]["gpu_requests"]),
                         ("gpu-serving", "gb200-tray-10", 4))
        self.assertEqual(summarize("pods", [pod]), [])

    def test_snapshot_groups_named_resources_and_marks_virtual_nodes(self) -> None:
        fixtures = {
            "nodes": [
                {
                    "metadata": {
                        "name": "gb200-tray-03",
                        "annotations": {"kwok.x-k8s.io/node": "fake"},
                    },
                    "spec": {"unschedulable": True},
                    "status": {
                        "allocatable": {"nvidia.com/gpu": "4"},
                        "conditions": [{"type": "Ready", "status": "True"}],
                    },
                }
            ],
            "pods": [
                {
                    "metadata": {
                        "name": "worker-a",
                        "namespace": "default",
                        "labels": {"app": "dynamo-inference-workers"},
                    },
                    "spec": {
                        "nodeName": "gb200-tray-03",
                        "containers": [
                            {"resources": {"requests": {"nvidia.com/gpu": "4"}}}
                        ],
                    },
                    "status": {"phase": "Running"},
                }
            ],
            "allocation_pods": [],
            "events": [
                {
                    "metadata": {"name": "event-a"},
                    "involvedObject": {"name": "dynamo-inference-workers-a"},
                    "reason": "FailedScheduling",
                    "message": "Insufficient nvidia.com/gpu",
                    "type": "Warning",
                }
            ],
            "control_plane_pods": [
                {
                    "metadata": {"name": "kube-scheduler-control-plane"},
                    "status": {
                        "phase": "Running",
                        "conditions": [{"type": "Ready", "status": "True"}],
                    },
                },
                {"metadata": {"name": "unrelated"}},
            ],
            "deployments": [
                {
                    "metadata": {"name": "gpu-simulator-engine"},
                    "spec": {"replicas": 1},
                    "status": {"readyReplicas": 1},
                }
            ],
            "hpas": [
                {
                    "metadata": {"name": "keda-hpa-dynamo-worker-autoscaler"},
                    "spec": {"minReplicas": 2, "maxReplicas": 18},
                    "status": {
                        "currentReplicas": 2,
                        "desiredReplicas": 4,
                        "currentMetrics": [
                            {"external": {"current": {"averageValue": "20"}}}
                        ],
                    },
                }
            ],
            "scaledobjects": [
                {
                    "metadata": {"name": "dynamo-worker-autoscaler"},
                    "spec": {
                        "triggers": [
                            {
                                "type": "prometheus",
                                "metadata": {
                                    "threshold": "15",
                                    "serverAddress": "secret-should-not-leak",
                                },
                            }
                        ]
                    },
                    "status": {"conditions": [{"type": "Ready", "status": "True"}]},
                }
            ],
            "keda_deployments": [
                {
                    "metadata": {"name": "keda-operator", "namespace": "keda"},
                    "status": {"readyReplicas": 1},
                }
            ],
        }
        reverse = {path: name for name, path in PATHS.items()}
        with patch(
            "gpu_sim.cluster.fetch_resource",
            side_effect=lambda path: fixtures[reverse[path]],
        ):
            result = asyncio.run(cluster_snapshot()).model_dump()
        self.assertTrue(result["available"])
        self.assertTrue(result["nodes"][0]["virtual"])
        self.assertTrue(result["nodes"][0]["unschedulable"])
        self.assertEqual(result["pods"][0]["gpu_requests"], 4)
        self.assertEqual(result["events"][0]["reason"], "FailedScheduling")
        self.assertEqual(
            [pod["name"] for pod in result["control_plane_pods"]],
            ["kube-scheduler-control-plane"],
        )
        self.assertEqual(result["hpas"][0]["desired"], 4)
        self.assertEqual(
            result["scaledobjects"][0]["triggers"],
            [{"type": "prometheus", "threshold": "15"}],
        )
        self.assertNotIn("secret-should-not-leak", str(result))

    def test_partial_permissions_are_reported_not_fabricated(self) -> None:
        def fetch(path: str) -> list:
            if path == PATHS["hpas"]:
                raise RuntimeError("Kubernetes API returned HTTP 403")
            return []

        with patch("gpu_sim.cluster.fetch_resource", side_effect=fetch):
            result = asyncio.run(cluster_snapshot())
        self.assertFalse(result.available)
        self.assertEqual(result.hpas, [])
        self.assertEqual(result.errors["hpas"], "Kubernetes API returned HTTP 403")
        self.assertNotIn("nodes", result.errors)

    def test_non_kubernetes_environment_is_explicitly_unavailable(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            result = asyncio.run(cluster_snapshot())
        self.assertFalse(result.available)
        self.assertEqual(set(result.errors), set(PATHS))
        self.assertTrue(
            all(
                value == "Kubernetes service account is not available"
                for value in result.errors.values()
            )
        )

    def test_event_filter_excludes_unrelated_workloads(self) -> None:
        events = [
            {"metadata": {"name": "other"}, "involvedObject": {"name": "unrelated"}},
            {
                "metadata": {"name": "ours"},
                "involvedObject": {"name": "gpu-simulator-engine-abc"},
                "message": "scheduled",
            },
        ]
        self.assertEqual([row["name"] for row in summarize("events", events)], ["ours"])
