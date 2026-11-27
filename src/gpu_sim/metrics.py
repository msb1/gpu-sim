"""Snapshot-based simulated DCGM, serving and orchestration metric collectors."""

from collections.abc import Iterator

from prometheus_client import CollectorRegistry, Histogram
from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily, Metric

from gpu_sim.simulator import Simulator

GPU_FIELDS = {
    "DCGM_FI_DEV_GPU_UTIL": "util",
    "DCGM_FI_DEV_MEM_COPY_UTIL": "mem",
    "DCGM_FI_DEV_FB_USED": "fb",
    "DCGM_FI_DEV_POWER_USAGE": "power",
    "DCGM_FI_DEV_GPU_TEMP": "temp",
    "DCGM_FI_DEV_CLOCK_THROTTLE_REASONS": "throttle",
    "DCGM_FI_DEV_XID_ERRORS": "xid",
    "DCGM_FI_PROF_SM_OCCUPANCY": "occupancy",
    "DCGM_FI_PROF_SM_ACTIVE": "sm",
    "DCGM_FI_PROF_PIPE_TENSOR_ACTIVE": "tensor",
    "DCGM_FI_PROF_DRAM_ACTIVE": "dram",
    "DCGM_FI_PROF_NVLINK_TX_BYTES": "tx",
    "DCGM_FI_PROF_NVLINK_RX_BYTES": "rx",
}
GPU_LABELS = ["gpu", "UUID", "device", "modelName", "node", "role", "replica_id"]


class Metrics:
    def __init__(self, simulator: Simulator) -> None:
        self.snapshot = simulator.state()
        self.last_generated, self.last_completed = (
            simulator.generated,
            simulator.completed,
        )
        self.totals: dict[tuple[str, str], list[int]] = {}
        self.wrapper_totals: dict[str, int] = {}
        self.histograms: dict[tuple[str, str], tuple[CollectorRegistry, Histogram]] = {}

    def publish(self, simulator: Simulator) -> None:
        state = simulator.state()
        c = state["config"]
        key = (c["backend_engine"], c["model_name"])
        totals = self.totals.setdefault(key, [0, 0])
        generated = simulator.generated - self.last_generated
        completed = simulator.completed - self.last_completed
        totals[0] += generated
        totals[1] += completed
        if c["triton_wrapper"]:
            self.wrapper_totals[c["model_name"]] = (
                self.wrapper_totals.get(c["model_name"], 0) + completed
            )
        self.last_generated, self.last_completed = (
            simulator.generated,
            simulator.completed,
        )
        if key not in self.histograms:
            registry = CollectorRegistry()
            prefix = "vllm:" if key[0] == "vllm" else "sim_trt_llm_"
            histogram = Histogram(
                prefix + "time_to_first_token_seconds",
                "Simulated first generated token latency seconds",
                ["model_name"],
                registry=registry,
            )
            self.histograms[key] = registry, histogram
        for latency in simulator.ttft:
            self.histograms[key][1].labels(model_name=c["model_name"]).observe(latency)
        simulator.ttft.clear()
        self.snapshot = {
            **state,
            "backend_totals": tuple(totals),
            "wrapper_completed": self.wrapper_totals.get(c["model_name"], 0),
        }

    def dcgm(self) -> Iterator[Metric]:
        rows = self.snapshot["telemetry"]
        fault = self.snapshot["fault"]
        if fault["type"] == "nvlink_xid":
            metric = GaugeMetricFamily(
                "sim_nvlink_xid_event_info",
                "Active simulated NVLink event observations",
                labels=[
                    "node",
                    "gpu",
                    "UUID",
                    "xid",
                    "link_id",
                    "peer_type",
                    "peer_id",
                    "severity",
                    "subcode",
                    "error_status",
                    "link_down",
                ],
            )
            for row in rows:
                if int(row["gpu"]) == fault["target_gpu"]:
                    metric.add_metric(
                        [
                            row["node"],
                            row["gpu"],
                            row["UUID"],
                            str(fault["xid_code"]),
                            fault["link_id"],
                            fault["peer_type"],
                            fault["peer_id"] or "unknown",
                            fault["severity"],
                            fault["subcode"],
                            fault["error_status"],
                            str(fault["link_down"]).lower(),
                        ],
                        1,
                    )
            yield metric
            if fault["peer_observation"] != "none":
                peer = next(
                    (
                        r
                        for r in rows
                        if fault["peer_type"] == "gpu" and r["gpu"] == fault["peer_id"]
                    ),
                    None,
                )
                observation = GaugeMetricFamily(
                    "sim_nvlink_peer_observation_info",
                    "Active synthetic peer-side observation; not a confirmed root cause",
                    labels=[
                        "reporter_node",
                        "reporter_gpu",
                        "link_id",
                        "peer_type",
                        "peer_id",
                        "peer_node",
                        "observation",
                    ],
                )
                reporter = next(r for r in rows if int(r["gpu"]) == fault["target_gpu"])
                observation.add_metric(
                    [
                        reporter["node"],
                        reporter["gpu"],
                        fault["link_id"],
                        fault["peer_type"],
                        fault["peer_id"],
                        peer["node"] if peer else "unknown",
                        fault["peer_observation"],
                    ],
                    1,
                )
                yield observation
        for name, key in GPU_FIELDS.items():
            metric = GaugeMetricFamily(name, "Simulated " + key, labels=GPU_LABELS)
            for row in rows:
                metric.add_metric([row[label] for label in GPU_LABELS], row[key])
            yield metric
        errors = GaugeMetricFamily(
            "DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_TOTAL",
            "Simulated cumulative CRC errors (DCGM-compatible name)",
            labels=GPU_LABELS,
        )
        counter = CounterMetricFamily(
            "sim_nvlink_crc_errors",
            "Simulated cumulative CRC errors",
            labels=GPU_LABELS,
        )
        for row in rows:
            values = [row[label] for label in GPU_LABELS]
            errors.add_metric(values, row["crc"])
            counter.add_metric(values, row["crc"])
        yield errors
        yield counter

    def serving(self) -> Iterator[Metric]:
        s, c = self.snapshot, self.snapshot["config"]
        yield self.gauge("sim_config_revision_seconds", s["config_revision_seconds"])
        model = c["model_name"]
        waiting = s["waiting"] + s["transferring"]
        running = s["running"]
        backend = c["backend_engine"]
        for field in ["rps", "prompt_tokens", "output_tokens"]:
            yield self.gauge(
                "sim_workload_" + ("target_rps" if field == "rps" else field),
                s["workload"][field],
            )
        yield self.gauge("sim_fault_capacity_factor", s["fault_factor"])
        yield self.gauge("sim_completed_requests_total", s["completed"], counter=True)
        if backend == "vllm":
            for name, value in [
                ("num_requests_running", running),
                ("num_requests_waiting", waiting),
                ("gpu_cache_usage_perc", s["kv_ratio"]),
            ]:
                yield self.gauge("vllm:" + name, value, {"model_name": model})
            token_name = "vllm:generation_tokens_total"
        else:
            metric = GaugeMetricFamily(
                "nv_trt_llm_request_metrics",
                "Simulated request counts",
                labels=["model_name", "request_type"],
            )
            metric.add_metric([model, "context"], waiting)
            metric.add_metric([model, "generation"], running)
            yield metric
            blocks = int(s["kv_capacity_bytes"] / (64 * c["kv_bytes_per_token"]))
            used = int(blocks * s["kv_ratio"])
            metric = GaugeMetricFamily(
                "nv_trt_llm_kv_cache_block_metrics",
                "Simulated KV blocks",
                labels=["model_name", "kv_metric"],
            )
            metric.add_metric([model, "used_blocks"], used)
            metric.add_metric([model, "free_blocks"], max(0, blocks - used))
            yield metric
            token_name = "nv_trt_llm_generation_tokens_total"
        yield self.gauge(
            token_name,
            s.get("backend_totals", (0, 0))[0],
            {"model_name": model},
            counter=True,
        )
        if c["triton_wrapper"]:
            yield self.gauge(
                "nv_inference_pending_request_count",
                waiting,
                {"model": model, "version": "1"},
            )
            yield self.gauge(
                "nv_inference_request_success_total",
                s.get("wrapper_completed", 0),
                {"model": model, "version": "1"},
                counter=True,
            )
        yield self.gauge(
            "dynamo_prefix_cache_hit_tokens_total",
            s["prefix_hits"],
            {"model_name": model},
            counter=True,
        )
        yield self.gauge(
            "dynamo_nixl_kv_transfer_bytes_total",
            s["nixl_bytes"],
            {"model_name": model},
            counter=True,
        )
        yield self.gauge(
            "dynamo_inter_node_ib_tx_bytes_per_sec", s["ib_tx"], {"model_name": model}
        )
        key = (backend, model)
        if key in self.histograms:
            yield from self.histograms[key][0].collect()

    @staticmethod
    def gauge(
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
        counter: bool = False,
    ) -> Metric:
        labels = labels or {}
        family = CounterMetricFamily if counter else GaugeMetricFamily
        metric = family(name, "Simulated " + name, labels=list(labels))
        metric.add_metric(list(labels.values()), value)
        return metric


class DCGMCollector:
    def __init__(self, metrics: Metrics) -> None:
        self.metrics = metrics

    def collect(self) -> Iterator[Metric]:
        return self.metrics.dcgm()


class ServingCollector:
    def __init__(self, metrics: Metrics) -> None:
        self.metrics = metrics

    def collect(self) -> Iterator[Metric]:
        return self.metrics.serving()
