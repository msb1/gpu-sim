"""Regression coverage for modeled NVLink XID scenarios."""

import unittest

from prometheus_client import CollectorRegistry, generate_latest
from pydantic import ValidationError

from gpu_sim.metrics import DCGMCollector, Metrics
from gpu_sim.simulator import Config, Fault, Simulator, Workload


class NvlinkXidTests(unittest.TestCase):
    def test_peer_observation_independent_and_cleared(self) -> None:
        sim = Simulator(1)
        sim.workload = Workload(rps=0)
        sim.inject(
            Fault(
                type="nvlink_xid",
                target_gpu=14,
                xid_code=149,
                link_id="logical-0",
                subcode="NETIR_LINK_DOWN",
                severity="fatal",
                link_down=True,
                peer_type="gpu",
                peer_id="18",
                peer_observation="link_unhealthy",
            )
        )
        sim.tick()
        metrics = Metrics(sim)
        registry = CollectorRegistry()
        registry.register(DCGMCollector(metrics))
        output = generate_latest(registry).decode()
        self.assertIn("sim_nvlink_peer_observation_info{", output)
        self.assertIn('peer_node="tray-4"', output)
        self.assertIn('reporter_gpu="14"', output)
        self.assertEqual(sim.crc, {})
        sim.inject(Fault(type="clear"))
        sim.tick()
        metrics.publish(sim)
        self.assertNotIn(
            "sim_nvlink_peer_observation_info{", generate_latest(registry).decode()
        )

    def test_unidentified_peer_observation_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            Fault(
                type="nvlink_xid",
                target_gpu=14,
                xid_code=149,
                link_id="logical-0",
                subcode="NETIR_LINK_DOWN",
                severity="fatal",
                link_down=True,
                peer_observation="link_unhealthy",
            )

    def test_profile_variants_and_clear(self) -> None:
        scenarios = [
            ("hopper_h100_80gb", 74, {}),
            (
                "blackwell_gb200_nvl72",
                144,
                {
                    "subcode": "SAW_MVB",
                    "severity": "nonfatal",
                    "error_status": "0x00000001",
                },
            ),
            (
                "blackwell_gb200_nvl72",
                144,
                {
                    "subcode": "SAW_MVB",
                    "severity": "fatal",
                    "error_status": "0x00000002",
                },
            ),
            (
                "blackwell_gb200_nvl72",
                149,
                {"subcode": "NETIR_LINK_DOWN", "severity": "fatal", "link_down": True},
            ),
        ]
        for profile, code, variant in scenarios:
            with self.subTest(profile=profile, code=code, variant=variant):
                sim = Simulator(1)
                sim.reset(
                    Config.model_validate(
                        sim.config.model_dump()
                        | {
                            "gpu_profile": profile,
                            "params_billions": 1,
                            "active_params_billions": 1,
                        }
                    )
                )
                sim.workload = Workload(rps=0)
                sim.inject(
                    Fault(
                        type="nvlink_xid",
                        target_gpu=14,
                        xid_code=code,
                        link_id="logical-0",
                        peer_type="switch",
                        peer_id="switch-1",
                        crc_errors=True,
                        **variant,
                    )
                )
                sim.tick()
                self.assertEqual(sim.fault_factor, 0.55)
                self.assertEqual([r["gpu"] for r in sim.telemetry if r["xid"]], ["14"])
                row = sim.telemetry[14]
                self.assertEqual(row["xid"], code)
                self.assertGreater(row["crc"], 0)
                self.assertTrue(
                    all(
                        r["xid"] == 0 and r["crc"] == 0
                        for r in sim.telemetry
                        if r["gpu"] != "14"
                    )
                )
                metrics = Metrics(sim)
                registry = CollectorRegistry()
                registry.register(DCGMCollector(metrics))
                output = generate_latest(registry).decode()
                self.assertIn("sim_nvlink_xid_event_info{", output)
                self.assertIn(f'xid="{code}"', output)
                self.assertIn('gpu="14"', output)
                counts = dict(sim.crc)
                sim.inject(Fault(type="clear"))
                sim.tick()
                metrics.publish(sim)
                self.assertEqual(sim.crc, counts)
                self.assertEqual(sim.fault_factor, 1)
                self.assertNotIn(
                    "sim_nvlink_xid_event_info{", generate_latest(registry).decode()
                )

    def test_invalid_variants_and_incompatible_profiles(self) -> None:
        for data in [
            {
                "type": "nvlink_xid",
                "target_gpu": 14,
                "xid_code": 149,
                "link_id": "l0",
                "subcode": "NETIR_LINK_DOWN",
                "severity": "fatal",
            },
            {
                "type": "nvlink_xid",
                "target_gpu": 14,
                "xid_code": 144,
                "link_id": "l0",
                "subcode": "SAW_MVB",
                "severity": "fatal",
                "error_status": "0x00000001",
            },
            {
                "type": "nvlink_xid",
                "target_gpu": 14,
                "xid_code": 74,
                "link_id": "l0",
                "peer_type": "gpu",
            },
            {"type": "xid_error", "target_gpu": 14, "xid_code": 149},
        ]:
            with self.subTest(data=data), self.assertRaises(ValidationError):
                Fault(**data)
        sim = Simulator(1)
        with self.assertRaises(ValueError):
            sim.inject(
                Fault(type="nvlink_xid", target_gpu=14, xid_code=74, link_id="l0")
            )
        with self.assertRaises(ValueError):
            sim.inject(
                Fault(
                    type="nvlink_xid",
                    target_gpu=14,
                    xid_code=149,
                    link_id="l0",
                    subcode="NETIR_LINK_DOWN",
                    severity="fatal",
                    link_down=True,
                    peer_type="gpu",
                    peer_id="14",
                )
            )
