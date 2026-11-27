"""Source-level freshness and coverage checks for scoped Prometheus observations."""

from typing import Any

from gpu_sim.analytics.prometheus import QueryResult, Scope


def sources(
    config: dict[str, Any], dcgm: Scope, serving: Scope
) -> dict[str, tuple[str, str, bool]]:
    d, s = dcgm.selector(), serving.selector()
    model = config["model_name"]
    sm = serving.selector(model_name=model)
    vllm = config["backend_engine"] == "vllm"
    queue = (
        f"vllm:num_requests_waiting{sm}"
        if vllm
        else f"nv_trt_llm_request_metrics{serving.selector(model_name=model, request_type='context')}"
    )
    running = (
        f"vllm:num_requests_running{sm}"
        if vllm
        else f"nv_trt_llm_request_metrics{serving.selector(model_name=model, request_type='generation')}"
    )
    tokens = (
        "vllm:generation_tokens_total" if vllm else "nv_trt_llm_generation_tokens_total"
    )
    ttft = (
        "vllm:time_to_first_token_seconds_bucket"
        if vllm
        else "sim_trt_llm_time_to_first_token_seconds_bucket"
    )
    mapping = {
        "offered": (f"sim_workload_target_rps{s}", "serving", False),
        "completed": (f"sim_completed_requests_total{s}", "serving", False),
        "tokens": (f"{tokens}{sm}", "serving", False),
        "prefix": (f"dynamo_prefix_cache_hit_tokens_total{sm}", "serving", False),
        "nixl": (f"dynamo_nixl_kv_transfer_bytes_total{sm}", "serving", False),
        "queue": (queue, "serving", False),
        "running": (running, "serving", False),
        "ttft": (f"{ttft}{sm}", "serving", False),
        "ib": (f"dynamo_inter_node_ib_tx_bytes_per_sec{sm}", "serving", False),
    }
    if vllm:
        mapping["kv"] = (f"vllm:gpu_cache_usage_perc{sm}", "serving", False)
    else:
        for kind in ("used_blocks", "free_blocks"):
            mapping[f"kv_{kind}"] = (
                f"nv_trt_llm_kv_cache_block_metrics{serving.selector(model_name=model, kv_metric=kind)}",
                "serving",
                False,
            )
    for name, metric in {
        "occupancy": "DCGM_FI_PROF_SM_OCCUPANCY",
        "tensor": "DCGM_FI_PROF_PIPE_TENSOR_ACTIVE",
        "dram": "DCGM_FI_PROF_DRAM_ACTIVE",
        "temperature": "DCGM_FI_DEV_GPU_TEMP",
        "throttle": "DCGM_FI_DEV_CLOCK_THROTTLE_REASONS",
        "xid": "DCGM_FI_DEV_XID_ERRORS",
        "crc": "sim_nvlink_crc_errors_total",
        "nvlink": "DCGM_FI_PROF_NVLINK_TX_BYTES",
        "power": "DCGM_FI_DEV_POWER_USAGE",
    }.items():
        mapping[name] = (f"{metric}{d}", "dcgm", True)
    mapping["event"] = (f"sim_nvlink_xid_event_info{d}", "dcgm", False)
    mapping["peer"] = (f"sim_nvlink_peer_observation_info{d}", "dcgm", False)
    return mapping


def gpu_identities(state: dict[str, Any]) -> set[tuple[str, str, str]]:
    inventory = state.get("inventory") or {}
    total, per_node = inventory.get("total_gpus"), inventory.get("gpus_per_node")
    if (
        not isinstance(total, int)
        or not isinstance(per_node, int)
        or total <= 0
        or per_node <= 0
    ):
        return set()
    prefix = "tray" if inventory.get("nvl72_domain") else "node"
    return {
        (
            f"{prefix}-{gpu // per_node}",
            str(gpu),
            f"GPU-{prefix}-{gpu // per_node}-{gpu:02d}",
        )
        for gpu in range(total)
    }


def verify_source(
    result: QueryResult,
    *,
    max_age: float,
    expected_gpus: set[tuple[str, str, str]] | None = None,
) -> str | None:
    if result.quality != "ok" or result.warnings:
        detail = f": {result.error}" if result.error else ""
        return f"source data quality: {result.quality}{detail}"
    if any(
        v.value is None or v.value < 0 or v.value > max_age
        for s in result.series
        for v in s.samples
    ):
        return "source samples are stale, invalid, or have a future timestamp"
    if expected_gpus is not None:
        identities = [
            (s.labels.get("node"), s.labels.get("gpu"), s.labels.get("UUID"))
            for s in result.series
        ]
        if (
            not expected_gpus
            or len(identities) != len(expected_gpus)
            or set(identities) != expected_gpus
        ):
            return "GPU source series do not match the current simulator inventory"
    return None


def gate_scrape(result: QueryResult) -> str | None:
    if result.quality != "ok" or result.warnings or len(result.series) != 1:
        return f"scrape status unavailable: {result.quality}"
    if any(s.value != 1 for s in result.series[0].samples):
        return "scrape target was down during the observation window"
    return None


def revision_error(result: QueryResult, expected: float | None) -> str | None:
    if (
        result.quality != "ok"
        or result.warnings
        or len(result.series) != 1
        or expected is None
    ):
        return "configuration revision or its observation history is unavailable"
    if any(s.value != expected for s in result.series[0].samples):
        return (
            "simulator configuration or process changed within the observation window"
        )
    return None
