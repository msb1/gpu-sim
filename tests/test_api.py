import asyncio
import json
import unittest
from importlib import import_module
from typing import Any

from gpu_sim.simulator import Simulator

api = import_module("gpu_sim.main")


async def call(
    method: str, path: str, payload: dict[str, Any] | None = None
) -> tuple[int, dict[str, Any]]:
    messages = []
    body = json.dumps(payload).encode() if payload is not None else b""

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message: dict[str, Any]) -> None:
        messages.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": [(b"content-type", b"application/json")],
        "client": ("test", 1),
        "server": ("test", 80),
        "root_path": "",
    }
    await api.app(scope, receive, send)
    return messages[0]["status"], json.loads(
        b"".join(m.get("body", b"") for m in messages[1:])
    )


class APITests(unittest.TestCase):
    def setUp(self) -> None:
        self.sim = Simulator(1)
        api.app.dependency_overrides[api.get_simulator] = lambda: self.sim

    def tearDown(self) -> None:
        api.app.dependency_overrides.clear()

    def test_state_and_partial_patch_over_asgi(self) -> None:
        code, state = asyncio.run(call("GET", "/state"))
        self.assertEqual(code, 200)
        self.assertEqual(state["config"]["active_params_billions"], 37)
        code, state = asyncio.run(call("POST", "/workload", {"rps": 20}))
        self.assertEqual(code, 200)
        self.assertEqual(state["workload"]["prompt_tokens"], 4096)
        code, state = asyncio.run(
            call("POST", "/config", {"backend_engine": "vllm", "triton_wrapper": False})
        )
        self.assertEqual(code, 200)
        self.assertTrue(state["requests_reset"])
        self.assertEqual(state["config"]["tp_size"], 8)

    def test_invalid_requests_do_not_mutate_state(self) -> None:
        baseline = self.sim.state()
        for path, payload in [
            ("/config", {"tp_size": 7}),
            ("/config", {"rps": 4}),
            ("/config", {"prefill_gpus": None}),
            ("/workload", {"rps": -1}),
            ("/workload", {"rps": None}),
            ("/fault", {"type": "unknown"}),
            ("/fault", {"type": "xid_error", "target_gpu": 72}),
            ("/fault", {"type": "thermal_throttle", "target_node": "tray-03"}),
        ]:
            with self.subTest(path=path, payload=payload):
                code, _ = asyncio.run(call("POST", path, payload))
                self.assertEqual(code, 422)
                self.assertEqual(self.sim.state(), baseline)

    def test_fault_response_and_clear(self) -> None:
        code, state = asyncio.run(
            call(
                "POST",
                "/fault",
                {"type": "xid_error", "target_gpu": 14, "xid_code": 48},
            )
        )
        self.assertEqual(code, 200)
        self.assertEqual(state["fault_factor"], 0.5)
        self.sim.tick()
        self.assertEqual(self.sim.telemetry[14]["xid"], 48)
        code, state = asyncio.run(call("POST", "/fault", {"type": "clear"}))
        self.assertEqual(code, 200)
        self.assertEqual(state["fault"]["type"], "none")
        self.assertEqual(state["fault_factor"], 1)
