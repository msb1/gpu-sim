"""GPU simulation package."""


def main() -> None:
    import uvicorn

    uvicorn.run("gpu_sim.main:app", host="0.0.0.0", port=8080)
