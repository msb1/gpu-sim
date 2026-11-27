"""Read-only live analytics inspection using the project's environment settings."""

import argparse
import json
import os
import time
from urllib.request import urlopen

from gpu_sim.analytics.diagnostics import diagnose
from gpu_sim.analytics.measurements import ObservationWindow, collect
from gpu_sim.analytics.prometheus import Prometheus, Scope


def get_json(url: str) -> dict:
    with urlopen(url, timeout=5) as response:
        return json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--window", type=float, required=True, help="Observation window in seconds"
    )
    parser.add_argument(
        "--step", type=float, required=True, help="Range query step in seconds"
    )
    parser.add_argument(
        "--max-age", type=float, required=True, help="Maximum sample age in seconds"
    )
    args = parser.parse_args()
    names = (
        "PROM_URL",
        "SIM_URL",
        "ANALYTICS_DCGM_JOB",
        "ANALYTICS_DCGM_INSTANCE",
        "ANALYTICS_SERVING_JOB",
        "ANALYTICS_SERVING_INSTANCE",
    )
    missing = [name for name in names if not os.getenv(name)]
    if missing:
        parser.error("Set these variables in the root .env: " + ", ".join(missing))
    sim_url = os.environ["SIM_URL"].rstrip("/")
    state = get_json(sim_url + "/state")
    cluster = get_json(sim_url + "/cluster")
    window = ObservationWindow(
        end=time.time(), duration=args.window, step=args.step, max_age=args.max_age
    )
    result = collect(
        Prometheus(os.environ["PROM_URL"]),
        state,
        cluster,
        Scope(
            job=os.environ["ANALYTICS_DCGM_JOB"],
            instance=os.environ["ANALYTICS_DCGM_INSTANCE"],
        ),
        Scope(
            job=os.environ["ANALYTICS_SERVING_JOB"],
            instance=os.environ["ANALYTICS_SERVING_INSTANCE"],
        ),
        window,
    )
    after = get_json(sim_url + "/state")
    if after.get("config_revision_seconds") != state.get("config_revision_seconds"):
        for measurement in result.measurements.values():
            if measurement.quality == "ok":
                measurement.quality = "insufficient_history"
                measurement.error = "Simulator configuration changed during collection"
    print(
        json.dumps(
            {
                "evidence": result.model_dump(),
                "findings": [f.model_dump() for f in diagnose(result)],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
