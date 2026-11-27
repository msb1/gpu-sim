"""FastAPI control and snapshot exporters for the fixed-inventory simulator."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, HTTPException
from prometheus_client import CollectorRegistry, start_http_server
from pydantic import BaseModel, ConfigDict, ValidationError, create_model

from gpu_sim.cluster import ClusterSnapshot, cluster_snapshot
from gpu_sim.metrics import DCGMCollector, Metrics, ServingCollector
from gpu_sim.simulator import Config, Fault, Simulator, Workload

sim = Simulator()
exports = Metrics(sim)
dcgm, serving = CollectorRegistry(), CollectorRegistry()
dcgm.register(DCGMCollector(exports))
serving.register(ServingCollector(exports))


def patch_model(name: str, base: type[BaseModel]) -> type[BaseModel]:
    return create_model(
        name,
        __config__=ConfigDict(extra="forbid", allow_inf_nan=False),
        **{
            key: (field.annotation | None, None)
            for key, field in base.model_fields.items()
        },
    )


ConfigPatch = patch_model("ConfigPatch", Config)
WorkloadPatch = patch_model("WorkloadPatch", Workload)


class State(BaseModel):
    config: Config
    config_revision_seconds: float
    workload: Workload
    fault: Fault
    capacity_mode: Literal["fixed_inventory"]
    fault_factor: float
    inventory: dict[str, Any]
    replicas: list[dict[str, Any]]
    time: float
    waiting: int
    running: int
    transferring: int
    generated_tokens: int
    completed: int
    prefix_hits: int
    nixl_bytes: int
    ib_tx: float
    kv_ratio: float
    kv_capacity_bytes: float
    hardware_assumptions: dict[str, Any]
    telemetry: list[dict[str, Any]]
    status: str | None = None
    requests_reset: bool = False


async def loop() -> None:
    while True:
        sim.tick()
        exports.publish(sim)
        await asyncio.sleep(1)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    servers = []
    task = None
    try:
        servers.append(start_http_server(9400, registry=dcgm))
        servers.append(start_http_server(8002, registry=serving))
        task = asyncio.create_task(loop())
        yield
    finally:
        if task is not None:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        for server, thread in servers:
            server.shutdown()
            server.server_close()
            thread.join()


app = FastAPI(title="GPU simulator", lifespan=lifespan)


def get_simulator() -> Simulator:
    return sim


Simulation = Annotated[Simulator, Depends(get_simulator)]


def merge(current: BaseModel, patch: BaseModel) -> BaseModel:
    try:
        return type(current).model_validate(
            current.model_dump() | patch.model_dump(exclude_unset=True)
        )
    except ValidationError as exc:
        raise HTTPException(422, detail=exc.errors(include_context=False)) from exc


@app.get("/state", response_model=State)
async def state(engine: Simulation) -> dict[str, Any]:
    return engine.state()


@app.post("/config", response_model=State)
async def config(value: ConfigPatch, engine: Simulation) -> dict[str, Any]:
    candidate = merge(engine.config, value)
    engine.reset(candidate)
    exports.publish(engine)
    return engine.state() | {"status": "config_updated", "requests_reset": True}


@app.post("/workload", response_model=State)
async def workload(value: WorkloadPatch, engine: Simulation) -> dict[str, Any]:
    engine.workload = merge(engine.workload, value)
    exports.publish(engine)
    return engine.state() | {"status": "workload_ramped"}


@app.post("/fault", response_model=State)
async def fault(value: Fault, engine: Simulation) -> dict[str, Any]:
    try:
        engine.inject(value)
    except ValueError as exc:
        raise HTTPException(422, detail=str(exc)) from exc
    return engine.state() | {"status": "fault_updated"}


@app.get("/cluster", response_model=ClusterSnapshot)
async def cluster() -> ClusterSnapshot:
    return await cluster_snapshot()
