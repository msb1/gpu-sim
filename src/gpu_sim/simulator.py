"""Fixed-inventory analytical serving model adapted from the supplied scaling samples."""

import math
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

PROFILES = {
    "ampere_a100_80gb": ("NVIDIA A100 80GB", 80, 2.04, 312, 312, 312, 400, 600, 65),
    "hopper_h100_80gb": ("NVIDIA H100 80GB", 80, 3.35, 989, 1979, 1979, 700, 900, 75),
    "blackwell_b200_192gb": (
        "NVIDIA B200 192GB",
        192,
        8.0,
        2250,
        4500,
        9000,
        1000,
        1800,
        120,
    ),
    "blackwell_gb200_nvl72": (
        "NVIDIA GB200 Grace Blackwell (NVL72)",
        192,
        8.0,
        2500,
        5000,
        10000,
        1200,
        1800,
        140,
    ),
}
TOPOLOGIES = {
    "single_1gpu": {"total_gpus": 1, "gpus_per_node": 1, "nvl72_domain": False},
    "node_8gpu": {"total_gpus": 8, "gpus_per_node": 8, "nvl72_domain": False},
    "cluster_16gpu": {"total_gpus": 16, "gpus_per_node": 8, "nvl72_domain": False},
    "rack_72gpu_nvl72": {"total_gpus": 72, "gpus_per_node": 4, "nvl72_domain": True},
}


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    topology: Literal[
        "single_1gpu", "node_8gpu", "cluster_16gpu", "rack_72gpu_nvl72"
    ] = "rack_72gpu_nvl72"
    gpu_profile: Literal[
        "ampere_a100_80gb",
        "hopper_h100_80gb",
        "blackwell_b200_192gb",
        "blackwell_gb200_nvl72",
    ] = "blackwell_gb200_nvl72"
    backend_engine: Literal["vllm", "tensorrt_llm"] = "tensorrt_llm"
    triton_wrapper: bool = True
    model_name: str = Field(default="deepseek-r1-671b", min_length=1, max_length=100)
    params_billions: float = Field(default=671, gt=0, le=1000)
    active_params_billions: float | None = Field(default=None, gt=0, le=1000)
    precision: Literal["fp16", "fp8", "fp4"] = "fp4"
    kv_bytes_per_token: int = Field(default=131072, gt=0, le=10000000)
    disaggregated: bool = True
    prefill_gpus: int = Field(default=24, ge=1, le=72)
    decode_gpus: int = Field(default=48, ge=1, le=72)
    tp_size: int = Field(default=8, ge=1, le=72)
    chunked_prefill_size: int = Field(default=8192, ge=1, le=1000000)
    prefix_cache_hit_rate: float = Field(default=0.45, ge=0, le=0.95)
    kv_transfer_latency_ms: float = Field(default=15, ge=0, le=60000)
    thermal_throttle_factor: float = Field(default=0.65, gt=0, le=1)
    nvlink_degraded_factor: float = Field(default=0.55, gt=0, le=1)
    xid_error_factor: float = Field(default=0.50, gt=0, le=1)

    @model_validator(mode="after")
    def fits(self) -> Config:
        total = self.total_gpus
        if (
            self.active_params_billions is not None
            and self.active_params_billions > self.params_billions
        ):
            raise ValueError("Active parameters exceed total parameters")
        if self.disaggregated:
            if self.prefill_gpus + self.decode_gpus != total:
                raise ValueError("Prefill plus decode GPUs must equal inventory")
            pools = [self.prefill_gpus, self.decode_gpus]
        else:
            pools = [total]
        if any(n % self.tp_size for n in pools):
            raise ValueError("Every pool must contain complete TP groups")
        if self.weights_per_gpu >= PROFILES[self.gpu_profile][1] * 0.88 * 1e9:
            raise ValueError("Model weights exceed usable GPU memory")
        return self

    @property
    def total_gpus(self) -> int:
        return int(TOPOLOGIES[self.topology]["total_gpus"])

    @property
    def weights_per_gpu(self) -> float:
        return (
            self.params_billions
            * {"fp16": 2, "fp8": 1, "fp4": 0.5}[self.precision]
            * 1e9
            / self.tp_size
        )

    @property
    def active_params(self) -> float:
        return (
            self.active_params_billions
            if self.active_params_billions is not None
            else self.params_billions
        )


class Workload(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    rps: float = Field(default=45, ge=0, le=1000)
    prompt_tokens: int = Field(default=4096, ge=1, le=1000000)
    output_tokens: int = Field(default=512, ge=1, le=1000000)


class Fault(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal[
        "none",
        "clear",
        "thermal_throttle",
        "nvlink_degraded",
        "xid_error",
        "nvlink_xid",
    ] = "none"
    target_node: str | None = None
    target_gpu: int | None = Field(default=None, ge=0, le=71)
    xid_code: Literal[79, 48, 31, 74, 144, 149] = 79
    link_id: str | None = Field(
        default=None, min_length=1, max_length=32, pattern=r"^[A-Za-z0-9_-]+$"
    )
    peer_type: Literal["unknown", "gpu", "switch"] = "unknown"
    peer_id: str | None = Field(
        default=None, min_length=1, max_length=32, pattern=r"^[A-Za-z0-9_-]+$"
    )
    severity: Literal["unknown", "nonfatal", "fatal"] = "unknown"
    subcode: Literal["unknown", "SAW_MVB", "NETIR_LINK_DOWN"] = "unknown"
    error_status: Literal["unknown", "0x00000001", "0x00000002"] = "unknown"
    link_down: bool = False
    crc_errors: bool = False
    peer_observation: Literal["none", "link_unhealthy"] = "none"

    @model_validator(mode="after")
    def target(self) -> Fault:
        targets = int(self.target_node is not None) + int(self.target_gpu is not None)
        if targets != (0 if self.type in ("none", "clear") else 1):
            raise ValueError(
                "Injection requires exactly one target; clear requires no target"
            )
        if self.type == "nvlink_xid":
            if (
                self.target_gpu is None
                or self.link_id is None
                or self.xid_code not in (74, 144, 149)
            ):
                raise ValueError(
                    "NVLink XID requires a reporting GPU, logical link and XID 74, 144 or 149"
                )
            if (self.peer_type == "unknown") != (self.peer_id is None):
                raise ValueError(
                    "A known peer requires an ID; unknown peer cannot have an ID"
                )
            if self.peer_observation != "none" and self.peer_type == "unknown":
                raise ValueError("Peer observation requires an identified peer")
            variants = {
                74: {("unknown", "unknown", "unknown")},
                144: {
                    ("SAW_MVB", "0x00000001", "nonfatal"),
                    ("SAW_MVB", "0x00000002", "fatal"),
                },
                149: {("NETIR_LINK_DOWN", "unknown", "fatal")},
            }
            if (self.subcode, self.error_status, self.severity) not in variants[
                self.xid_code
            ]:
                raise ValueError("Invalid XID scenario variant")
            if self.xid_code == 149 and not self.link_down:
                raise ValueError("NETIR_LINK_DOWN requires link_down")
        elif self.xid_code not in (79, 48, 31):
            raise ValueError("NVLink XIDs require nvlink_xid fault type")
        elif self.peer_observation != "none":
            raise ValueError("Peer observations require an NVLink XID scenario")
        return self


@dataclass(eq=False)
class Request:
    arrival: float
    prompt: int
    output: int
    prefilled: int = 0
    generated: int = 0
    transfer_until: float = 0
    first_token: bool = False
    cached: int = 0
    prefill_replica: int | None = None
    decode_replica: int | None = None


@dataclass
class Replica:
    id: int
    role: str
    gpus: list[int]


class Simulator:
    def __init__(self, seed: int | None = None) -> None:
        self.random = random.Random(seed)
        self.config = Config(active_params_billions=37)
        self.workload = Workload()
        self.waiting: deque[Request] = deque()
        self.active: list[Request] = []
        self.transfer: list[Request] = []
        self.time = 0.0
        self.generated = self.completed = self.prefix_hits = self.nixl_bytes = 0
        self.ttft: list[float] = []
        self.telemetry: list[dict[str, Any]] = []
        self.fault = Fault()
        self.crc: dict[str, int] = {}
        self.ib_tx = 0.0
        self.replicas: list[Replica] = []
        self.reset(self.config)

    def node(self, gpu: int) -> str:
        topo = TOPOLOGIES[self.config.topology]
        prefix = "tray" if topo["nvl72_domain"] else "node"
        return f"{prefix}-{gpu // int(topo['gpus_per_node'])}"

    def reset(self, config: Config) -> None:
        self.config = config
        self.config_revision_seconds = time.time()
        self.waiting.clear()
        self.active.clear()
        self.transfer.clear()
        self.ttft.clear()
        self.telemetry = []
        self.fault = Fault()
        self.ib_tx = 0.0
        self.replicas = [
            Replica(
                i // config.tp_size,
                ("prefill" if i < config.prefill_gpus else "decode")
                if config.disaggregated
                else "colocated",
                list(range(i, i + config.tp_size)),
            )
            for i in range(0, config.total_gpus, config.tp_size)
        ]

    def inject(self, fault: Fault) -> None:
        if fault.type == "nvlink_xid":
            allowed = {
                74: ("hopper_h100_80gb", "ampere_a100_80gb"),
                144: ("blackwell_gb200_nvl72",),
                149: ("blackwell_gb200_nvl72",),
            }
            if self.config.gpu_profile not in allowed[fault.xid_code]:
                raise ValueError("XID is not supported by this simulated GPU profile")
            if fault.peer_type == "gpu" and (
                not fault.peer_id.isdigit()
                or int(fault.peer_id) >= self.config.total_gpus
                or int(fault.peer_id) == fault.target_gpu
            ):
                raise ValueError("GPU peer must be a different GPU in the inventory")
        if fault.target_gpu is not None and fault.target_gpu >= self.config.total_gpus:
            raise ValueError("GPU target outside inventory")
        if fault.target_node is not None and fault.target_node not in {
            self.node(i) for i in range(self.config.total_gpus)
        }:
            raise ValueError("Unknown node target")
        self.fault = Fault() if fault.type == "clear" else fault

    @property
    def fault_factor(self) -> float:
        return (
            getattr(
                self.config,
                f"{'nvlink_degraded' if self.fault.type == 'nvlink_xid' else self.fault.type}_factor",
            )
            if self.fault.type not in ("none", "clear")
            else 1.0
        )

    @property
    def kv_capacity(self) -> float:
        return (
            PROFILES[self.config.gpu_profile][1] * 0.88 * 1e9
            - self.config.weights_per_gpu
        ) * self.config.tp_size

    def kv(self, requests: list[Request], reserved: bool = False) -> int:
        return sum(
            ((r.prompt + r.output) if reserved else (r.prefilled + r.generated))
            * self.config.kv_bytes_per_token
            for r in requests
        )

    def pool(self, role: str) -> list[Replica]:
        return [r for r in self.replicas if r.role in (role, "colocated")]

    def members(self, replica: Replica, decode: bool = False) -> list[Request]:
        return [
            r
            for r in self.active + self.transfer
            if (r.decode_replica if decode else r.prefill_replica) == replica.id
        ]

    def admit(self) -> None:
        while self.waiting:
            req = self.waiting[0]
            needed = self.kv([req], reserved=True)
            p = next(
                (
                    g
                    for g in sorted(
                        self.pool("prefill"),
                        key=lambda g: self.kv(self.members(g), True),
                    )
                    if self.kv(self.members(g), True) + needed <= self.kv_capacity
                ),
                None,
            )
            d = next(
                (
                    g
                    for g in sorted(
                        self.pool("decode"),
                        key=lambda g: self.kv(self.members(g, True), True),
                    )
                    if self.kv(self.members(g, True), True) + needed <= self.kv_capacity
                ),
                None,
            )
            if p is None or d is None:
                break
            self.waiting.popleft()
            req.cached = int(req.prompt * self.config.prefix_cache_hit_rate)
            req.prefilled = req.cached
            self.prefix_hits += req.cached
            req.prefill_replica, req.decode_replica = p.id, d.id
            self.active.append(req)

    def tick(self, dt: float = 1.0) -> None:
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("Tick duration must be finite and positive")
        c, w, rng = self.config, self.workload, self.random
        hw = PROFILES[c.gpu_profile]
        t = 0.0
        while w.rps and (t := t + rng.expovariate(w.rps)) < dt:
            self.waiting.append(
                Request(
                    self.time + t,
                    max(1, round(rng.gauss(w.prompt_tokens, w.prompt_tokens * 0.15))),
                    max(1, round(rng.gauss(w.output_tokens, w.output_tokens * 0.15))),
                )
            )
        self.admit()
        ready = [r for r in self.transfer if r.transfer_until <= self.time]
        self.active.extend(ready)
        self.transfer = [r for r in self.transfer if r not in ready]
        prefilling = {id(r) for r in self.active if r.prefilled < r.prompt}
        peak = hw[{"fp16": 3, "fp8": 4, "fp4": 5}[c.precision]]
        rate = (
            peak
            * c.tp_size
            * 1e12
            * (0.72 if c.backend_engine == "tensorrt_llm" else 0.58)
            * self.fault_factor
            / (2 * c.active_params * 1e9)
        )
        duties: dict[int, tuple[float, float, float, float]] = {}
        handoff = 0
        self.ib_tx = 0.0
        for group in self.replicas:
            processed = 0
            budget = min(rate * dt, c.chunked_prefill_size)
            for req in self.active:
                if req.prefill_replica != group.id or id(req) not in prefilling:
                    continue
                n = min(req.prompt - req.prefilled, int(budget))
                req.prefilled += n
                processed += n
                budget -= n
                if req.prefilled == req.prompt and c.disaggregated:
                    req.transfer_until = (
                        self.time + dt + c.kv_transfer_latency_ms / 1000
                    )
                    self.transfer.append(req)
                    handoff += req.prompt * c.kv_bytes_per_token
                    pnode = self.node(self.replicas[req.prefill_replica].gpus[0])
                    dnode = self.node(self.replicas[req.decode_replica].gpus[0])
                    if not TOPOLOGIES[c.topology]["nvl72_domain"] and pnode != dnode:
                        self.ib_tx += req.prompt * c.kv_bytes_per_token / dt
            pd = processed / rate / dt
            decode = [
                r
                for r in self.active
                if r.decode_replica == group.id
                and r.prefilled == r.prompt
                and r not in self.transfer
                and r not in ready
                and id(r) not in prefilling
                and r.arrival < self.time
            ]
            kv = self.kv(self.members(group, group.role != "prefill")) / c.tp_size
            active_weights = c.weights_per_gpu * c.active_params / c.params_billions
            bw_rate = (
                hw[2]
                * 1e12
                * (0.85 if c.backend_engine == "tensorrt_llm" else 0.74)
                * self.fault_factor
            )
            step = (active_weights + kv) / bw_rate
            available = dt * (1 - pd) if not c.disaggregated else dt
            steps = min(
                int(available / step),
                max((r.output - r.generated for r in decode), default=0),
            )
            bd = steps * step / dt
            for req in decode:
                n = min(steps, req.output - req.generated)
                if n and not req.first_token:
                    self.ttft.append(self.time + dt - req.arrival)
                    req.first_token = True
                req.generated += n
                self.generated += n
            duties[group.id] = (pd, bd, min(1, math.log2(len(decode) + 1) / 7), kv)
        self.active = [r for r in self.active if r not in self.transfer]
        self.completed += sum(r.generated >= r.output for r in self.active)
        self.active = [r for r in self.active if r.generated < r.output]
        self.nixl_bytes += handoff
        self.emit(duties, handoff / dt, dt)
        self.time += dt

    def emit(
        self,
        duties: dict[int, tuple[float, float, float, float]],
        transfer_rate: float,
        dt: float,
    ) -> None:
        c, hw = self.config, PROFILES[self.config.gpu_profile]
        self.telemetry = []
        for group in self.replicas:
            pd, bd, batch, kv = duties[group.id]
            for gpu in group.gpus:
                noise = lambda x: max(0, min(1, x * self.random.uniform(0.98, 1.02)))
                sm = noise(min(1, pd * 0.98 + bd * 0.93))
                occupancy = noise(min(0.96, pd * 0.90 + bd * batch * 0.78))
                tensor = noise(
                    min(
                        0.95,
                        pd * (0.94 if c.backend_engine == "tensorrt_llm" else 0.86)
                        + bd * (0.14 + 0.26 * batch),
                    )
                )
                dram = noise(
                    min(
                        0.96,
                        pd * 0.46
                        + bd * (0.95 if c.backend_engine == "tensorrt_llm" else 0.88),
                    )
                )
                util = min(100, sm * 100)
                power = hw[8] + (hw[6] - hw[8]) * min(1, pd + 0.45 * bd + 0.42 * batch)
                temp, throttle, xid = 36 + int(util * 0.38), 0, 0
                tx = (
                    transfer_rate / c.total_gpus + (pd + bd * batch) * 32768
                    if c.tp_size > 1
                    else transfer_rate / c.total_gpus
                )
                node = self.node(gpu)
                uuid = f"GPU-{node}-{gpu:02d}"
                targeted = (
                    self.fault.target_gpu == gpu or self.fault.target_node == node
                )
                if targeted and self.fault.type == "thermal_throttle":
                    temp, throttle = self.random.randint(89, 93), 32
                    occupancy *= 0.45
                    tensor *= 0.45
                    util *= 0.5
                    power = hw[6] * 0.62
                elif targeted and self.fault.type == "nvlink_degraded":
                    tx *= 0.12
                    tensor *= 0.55
                    occupancy *= 0.60
                    self.crc[uuid] = self.crc.get(uuid, 0) + round(
                        self.random.randint(15, 60) * dt
                    )
                elif targeted and self.fault.type == "xid_error":
                    xid = self.fault.xid_code
                    sm = occupancy = tensor = dram = util = tx = 0.0
                    power = hw[8] * 0.5
                elif targeted and self.fault.type == "nvlink_xid":
                    xid = self.fault.xid_code
                    if self.fault.link_down:
                        tx *= 0.12
                    if self.fault.crc_errors:
                        self.crc[uuid] = self.crc.get(uuid, 0) + round(
                            self.random.randint(15, 60) * dt
                        )
                self.telemetry.append(
                    {
                        "gpu": str(gpu),
                        "UUID": uuid,
                        "device": f"nvidia{gpu % int(TOPOLOGIES[c.topology]['gpus_per_node'])}",
                        "node": node,
                        "role": group.role,
                        "replica_id": str(group.id),
                        "modelName": hw[0],
                        "util": util,
                        "mem": dram * 100,
                        "fb": (c.weights_per_gpu + kv) / 1048576,
                        "power": power,
                        "temp": temp,
                        "throttle": throttle,
                        "xid": xid,
                        "crc": self.crc.get(uuid, 0),
                        "occupancy": occupancy,
                        "sm": sm,
                        "tensor": tensor,
                        "dram": dram,
                        "tx": tx,
                        "rx": tx,
                    }
                )

    def state(self) -> dict[str, Any]:
        decode = self.pool("decode")
        capacity = len(decode) * self.kv_capacity
        return {
            "config": self.config.model_dump(),
            "workload": self.workload.model_dump(),
            "fault": self.fault.model_dump(),
            "config_revision_seconds": self.config_revision_seconds,
            "capacity_mode": "fixed_inventory",
            "fault_factor": self.fault_factor,
            "inventory": TOPOLOGIES[self.config.topology],
            "replicas": [
                {
                    "id": r.id,
                    "role": r.role,
                    "gpus": r.gpus,
                    "nodes": list(dict.fromkeys(self.node(g) for g in r.gpus)),
                }
                for r in self.replicas
            ],
            "time": self.time,
            "waiting": len(self.waiting),
            "running": len(self.active),
            "transferring": len(self.transfer),
            "generated_tokens": self.generated,
            "completed": self.completed,
            "prefix_hits": self.prefix_hits,
            "nixl_bytes": self.nixl_bytes,
            "ib_tx": self.ib_tx,
            "kv_ratio": self.kv(self.active + self.transfer) / capacity,
            "kv_capacity_bytes": capacity,
            "hardware_assumptions": {
                "illustrative_profile": True,
                "precision_fallback": (
                    self.config.gpu_profile == "ampere_a100_80gb"
                    and self.config.precision in ("fp8", "fp4")
                )
                or (
                    self.config.gpu_profile == "hopper_h100_80gb"
                    and self.config.precision == "fp4"
                ),
                "fault_scope": "cluster_wide",
                "thermal_throttle_bitmask": 32,
            },
            "telemetry": self.telemetry,
        }
