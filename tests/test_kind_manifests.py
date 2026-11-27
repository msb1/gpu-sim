import unittest

from gpu_sim.simulator import Config
from scripts.render_kind_inventory import (
    PRESETS,
    inventory,
    node_status,
    worker_resources,
)


class KindManifestTests(unittest.TestCase):
    def test_inventory_and_simulator_configuration_agree(self) -> None:
        for topology, (count, gpus, prefix, prefill, decode, tp) in PRESETS.items():
            with self.subTest(topology=topology):
                nodes = inventory(topology)
                self.assertEqual(len(nodes), count)
                self.assertEqual(
                    len({node["metadata"]["name"] for node in nodes}), count
                )
                self.assertEqual(nodes[0]["metadata"]["name"], f"{prefix}-00")
                self.assertTrue(
                    all(
                        n["metadata"]["annotations"]["kwok.x-k8s.io/node"] == "fake"
                        for n in nodes
                    )
                )
                status = node_status(topology)["status"]
                self.assertEqual(
                    int(status["capacity"]["nvidia.com/gpu"]) * count, prefill + decode
                )
                self.assertEqual(status["capacity"], status["allocatable"])
                config = Config.model_validate(
                    Config().model_dump()
                    | {
                        "topology": topology,
                        "prefill_gpus": prefill,
                        "decode_gpus": decode,
                        "tp_size": tp,
                    }
                )
                self.assertEqual(config.total_gpus, count * gpus)

    def test_worker_capacity_and_scaler_target(self) -> None:
        for topology, (count, gpus, _, _, _, _) in PRESETS.items():
            with self.subTest(topology=topology):
                deployment, scaled = worker_resources(
                    topology, "http://172.18.0.1:9090"
                )
                self.assertEqual(
                    scaled["spec"]["scaleTargetRef"]["name"],
                    deployment["metadata"]["name"],
                )
                pod = deployment["spec"]["template"]["spec"]
                self.assertEqual(
                    pod["nodeSelector"]["gpu-sim.io/inventory"],
                    topology.replace("_", "-"),
                )
                self.assertEqual(pod["tolerations"][0]["key"], "kwok.x-k8s.io/node")
                resources = pod["containers"][0]["resources"]
                self.assertEqual(resources["requests"], resources["limits"])
                requested = int(resources["requests"]["nvidia.com/gpu"])
                self.assertLessEqual(requested, gpus)
                self.assertEqual(
                    scaled["spec"]["maxReplicaCount"] * requested, count * gpus
                )
                self.assertEqual(
                    scaled["spec"]["minReplicaCount"], deployment["spec"]["replicas"]
                )
                self.assertIn(
                    "172.18.0.1:9090",
                    scaled["spec"]["triggers"][0]["metadata"]["serverAddress"],
                )
        self.assertEqual(
            worker_resources("rack_72gpu_nvl72", "http://172.18.0.1:9090")[1]["spec"][
                "minReplicaCount"
            ],
            2,
        )
        self.assertEqual(
            worker_resources("rack_72gpu_nvl72", "http://172.18.0.1:9090")[1]["spec"][
                "maxReplicaCount"
            ],
            18,
        )
        self.assertEqual(
            worker_resources("cluster_16gpu", "http://172.18.0.1:9090")[1]["spec"][
                "minReplicaCount"
            ],
            1,
        )
        self.assertEqual(
            worker_resources("cluster_16gpu", "http://172.18.0.1:9090")[1]["spec"][
                "maxReplicaCount"
            ],
            2,
        )

    def test_prometheus_url_rejects_missing_host_and_credentials(self) -> None:
        for url in ("", "localhost:9090", "http://user:secret@example.test:9090"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                worker_resources("rack_72gpu_nvl72", url)
