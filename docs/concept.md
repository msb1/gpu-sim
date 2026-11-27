# GPU simulator concepts

## Purpose

GPU-sim is a lab for exploring how inference workload demand, modeled GPU resources, telemetry, Kubernetes scheduling and human-reviewed agent proposals interact. Its outputs are simulated evidence for experiments and software integration, not measurements or capacity predictions for physical GPUs. A hands-on operator can configure workloads and faults, inspect telemetry, run agent workflows, review proposed changes and test queueing behavior.

## Simulation model

The simulator owns a fixed inventory: a single GPU, an eight-GPU node, two eight-GPU nodes, or an NVL72-style rack of 18 four-GPU trays. Hardware profiles, precision, model sizes, token lengths and serving-stack options parameterize an analytical model. Tensor-parallel (TP) groups form replicas within the configured inventory. A disaggregated configuration divides the inventory between prefill and decode pools; a colocated configuration uses shared groups.

Requests arrive probabilistically, wait for admission, prefill, optionally transfer KV state, decode and complete. Full requested KV capacity is reserved for admission. Modeled compute and memory bandwidth constrain progress; the model includes configurable prefix-cache effectiveness, chunked prefill and transfer delay. GPU activity, temperature, power, fabric and serving metrics are synthetic. The available vLLM, TensorRT-LLM, Triton and Dynamo-style options model characteristics and expose compatible metric families; they are not those inference servers running a model.

Fault injection produces simulated thermal, NVLink degradation and XID-related signals. Interpret a finding as evidence to investigate: an event or peer observation does not by itself establish physical fault origin, and missing telemetry is not a fault. See [design.md](design.md) for the current measurement and agent rules.

## Kubernetes and serving

The kind lab uses real Kubernetes controllers with KWOK virtual GPU nodes. The simulator runs as a real container on the control-plane node; virtual worker Pods and KServe predictors on KWOK nodes do not run containers. KEDA observes Prometheus and scales virtual worker replicas, while simulator physics keeps its own fixed inventory. Kubernetes replica count is therefore not a measurement of simulator capacity.

KServe creates a simulated predictor from an approved InferenceService. Kueue reserves admission quota for its Pod; kube-scheduler then places an admitted Pod on a suitable virtual node. Quota is an admission budget, not proof of a free node. Envoy Gateway and Cloud Provider KIND expose a Gateway and route configuration, but the pause-image KWOK predictor does not serve real inference traffic.

## Agents and human control

The separate coordinator exposes three roles. Profiler diagnoses scoped Prometheus evidence; Scheduler can propose a limited simulator prefill/decode rebalance; Benchmark plans and runs an approved load sweep. Findings and proposal rules are deterministic. A configured OpenAI-compatible model writes advisory summaries after profiler and scheduler runs and after completed benchmarks; it does not decide or execute changes. The operator reviews any supported proposal before execution. Approval records are in memory for the lifetime of the coordinator.

Benchmark classifications describe the tested simulator observations against configured throughput, queue-growth and time-to-first-token (TTFT) criteria. A passing single point is a lower bound within its measurement window, not a proven maximum or long-term guarantee. Existing backlog can make completed throughput exceed current arrivals and distort latency. The LLM narrative can misstate data; inspect structured findings and the benchmark report.

## Scope and next work

This lab is not a calibrated real-GPU performance predictor or a deployed inference service. Real exporter semantics, network behavior, security, fault origin and real model serving require independent validation. [next-steps.md](next-steps.md) records remaining work; [operator-runbook.md](operator-runbook.md) gives setup and live tests.

## What a KWOK node represents

KWOK means **Kubernetes WithOut Kubelet**. A normal Kubernetes node corresponds to a machine or VM running a kubelet and container runtime. A KWOK node is an API object whose lifecycle and Pod status are simulated by the KWOK controller. The Kubernetes scheduler can see its labels, taints, readiness and advertised resources and bind Pods to it, but no kubelet on that virtual node starts their containers.

For example, `gb200-tray-10` represents one four-GPU tray in the lab. A predictor can request four `nvidia.com/gpu` resources and be assigned there; KWOK can then make the Pod appear Running and Ready. That proves API-level admission and placement, not execution of a model or use of physical GPUs. The one real kind control-plane node hosts the simulator container and controllers. A request shown as `4 / 4` in inventory is the sum of scheduled Pod requests versus allocatable GPU resources, not utilization.

## KServe, Kueue and kube-scheduler: distinct decisions

- **KServe** is the Kubernetes controller that reconciles an `InferenceService`, a declaration of a desired prediction service. In this lab it uses RawDeployment mode to create a predictor Deployment and related service and route resources. The predictor is a pause-image fixture on KWOK; KServe's Ready condition is therefore simulated here.
- **Kueue** is the admission and quota controller. A named LocalQueue in `gpu-serving` points to the cluster-scoped ClusterQueue. A ResourceFlavor describes the eligible KWOK labels and taint. The ClusterQueue defines a budget for CPU, memory and GPUs. Kueue creates a Workload for the queued Pod, decides whether quota can be reserved, and controls the Pod's admission scheduling gate. It does not choose which particular node hosts the Pod.
- **kube-scheduler** is the standard Kubernetes scheduler. Once admission has removed the gate, it evaluates the Pod's requests, node selectors, taints/tolerations and available node resources and binds the Pod to a suitable node. Admission is not a guarantee that such a node exists.
- **KEDA** is a separate event-driven autoscaler. Here it reads an existing Prometheus service and adjusts the Dynamo worker Deployment's desired replica count. Its virtual workers share the KWOK node inventory with serving fixtures, but changing their count does not change the simulator's modeled GPU inventory or capacity.

### Submission, admission and placement in this project

1. In the dashboard the operator enters a service name and GPUs per predictor and selects **Propose service**. The coordinator looks up the LocalQueue and creates a reviewable plan. The plan contains an InferenceService manifest, the target Kubernetes context and the queue UID; planning does not create a Kubernetes resource.
2. The operator approves. The coordinator verifies the context and queue UID and submits the InferenceService using `kubectl`. This is a serving request, not a Kubernetes `Job` or a batch-job submission. KServe reconciles it into a Deployment and predictor Pod with the Kueue queue label. KServe adds CPU/memory defaults as applicable.
3. Kueue's Pod integration creates a Workload and holds the predictor with an admission gate while its resource requests cannot fit the ClusterQueue budget. For example, with a four-GPU budget already reserved, a second four-GPU request waits even if other simulated GPUs exist elsewhere in the rack.
4. When quota is released, Kueue reserves it, marks the Workload admitted and releases the gate. The default Kubernetes scheduler then chooses a virtual node that satisfies the Pod's selector, toleration and resource requirements. KWOK supplies simulated status after placement.
5. KServe and Envoy Gateway publish the service and HTTPRoute state. Inspect the dashboard's **Workload admission** and **Predictor Pod placement** separately: submitted, admitted, scheduled and Ready are different states. The Gateway routes to a service, but this pause fixture cannot answer real inference requests.

Kueue's Workload is a queueing object owned by the predictor Pod in this integration. The user-facing word “job” should not be confused with a Kubernetes `Job`, which this path does not create. The human approval is permission to submit; it is not Kueue admission or node placement.

## Simulator calculation model

Each simulated tick defaults to one second. Arrivals use exponentially distributed interarrival times based on offered RPS; prompt and output lengths are normal draws around configured values with 15% standard deviation and a minimum of one token. For each request the full prompt plus output KV requirement is reserved at admission. Groups with the least reserved KV are considered first; the FIFO head waits if suitable prefill and decode groups cannot fit it. Prefix-cache tokens are credited at admission using the configured hit fraction.

Each TP group models prefill token capacity as `peak_TFLOPS × TP × 10^12 × backend_efficiency × fault_factor / (2 × active_parameter_count)`, limited per tick by the configured chunked-prefill budget. Backend efficiency is 0.72 for TensorRT-LLM and 0.58 for vLLM. Decode estimates a step as `(active weight bytes per GPU + current KV bytes per GPU) / effective HBM bytes per second`; modeled bandwidth efficiency is 0.85 for TensorRT-LLM and 0.74 for vLLM. A disaggregated request waits for its modeled KV transfer and the decode pool can operate concurrently with the prefill pool. Colocated work shares the tick budget. First-token latency spans arrival, queueing, prefill, transfer and the tick when the first token is produced.

These formulas use illustrative profile values, not calibration. The engine creates one replica per TP-sized group. It does not implement a real block allocator, hardware kernel scheduling, detailed network contention or an actual vLLM/TensorRT-LLM process. Fault factors modify the modeled capacity; per-GPU telemetry also reflects the injected target. See `simulator.py` for the exact validation and edge cases.

## Synthetic DCGM-style GPU metrics

Each row comes from the simulator's current per-GPU snapshot and carries GPU, synthetic UUID/device, model, node, role and replica labels. The following definitions describe this simulator, not NVIDIA's measurement guarantees. Let `P` be prefill duty, `D` decode bandwidth duty, `B = min(1, log2(number of decode requests + 1) / 7)`, and `noise(x)` multiply by a random factor from 0.98 to 1.02 and clamp to 0–1. Duty values derive from simulated work in the tick.

| Exported name | Unit / meaning in gpu-sim | Calculation or source |
|---|---|---|
| `DCGM_FI_DEV_GPU_UTIL` | percent; heuristic GPU busy indicator | `min(100, SM_active × 100)`; selected faults override it. |
| `DCGM_FI_DEV_MEM_COPY_UTIL` | percent; memory-activity proxy | `DRAM_active × 100`; it is not a measured copy-engine utilization. |
| `DCGM_FI_DEV_FB_USED` | MiB; modeled per-GPU framebuffer use | `(weights_per_gpu + current_group_KV_per_gpu) / 1,048,576`. |
| `DCGM_FI_DEV_POWER_USAGE` | W; heuristic power | Profile idle power plus `(max_power − idle_power) × min(1, P + 0.45D + 0.42B)`; faults may override it. |
| `DCGM_FI_DEV_GPU_TEMP` | °C; heuristic temperature | Normally `36 + int(GPU_util × 0.38)`; a targeted thermal fault substitutes a random 89–93 °C. |
| `DCGM_FI_DEV_CLOCK_THROTTLE_REASONS` | synthetic bitmask | Normally zero; targeted thermal fault emits bit value 32. Verify real-platform semantics independently. |
| `DCGM_FI_DEV_XID_ERRORS` | synthetic XID code | Zero normally; selected XID/NVLink-XID fault code on the target. It is an event indication, not a root-cause diagnosis. |
| `DCGM_FI_PROF_SM_ACTIVE` | ratio; heuristic streaming-multiprocessor activity | `noise(min(1, 0.98P + 0.93D))`. |
| `DCGM_FI_PROF_SM_OCCUPANCY` | ratio; heuristic occupancy | `noise(min(0.96, 0.90P + 0.78DB))`; targeted faults can reduce it. |
| `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE` | ratio; heuristic tensor-pipe activity | `noise(min(0.95, P × backend_coefficient + D × (0.14 + 0.26B)))`, with prefill coefficient 0.94 for TensorRT-LLM or 0.86 for vLLM. |
| `DCGM_FI_PROF_DRAM_ACTIVE` | ratio; heuristic DRAM activity | `noise(min(0.96, 0.46P + D × backend_coefficient))`, with decode coefficient 0.95 for TensorRT-LLM or 0.88 for vLLM. |
| `DCGM_FI_PROF_NVLINK_TX_BYTES` | bytes/s; synthetic transmit rate | Transfer bytes/s divided by total GPUs, plus `(P + DB) × 32768` when TP > 1. It is a proxy, not a byte counter. |
| `DCGM_FI_PROF_NVLINK_RX_BYTES` | bytes/s; synthetic receive rate | Set equal to simulated TX for that GPU. |
| `DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_TOTAL` | cumulative synthetic error gauge | Per-GPU accumulated CRC count, increased by selected degraded-link faults. |
| `sim_nvlink_crc_errors_total` | cumulative synthetic counter | Same modeled CRC count exported as a counter for rate calculations. |
| `sim_nvlink_xid_event_info` | event-presence gauge | Emitted for the targeted active NVLink-XID scenario, with event and peer labels. |
| `sim_nvlink_peer_observation_info` | observation-presence gauge | Emitted only when an independent simulated peer observation is configured. A peer is not proof of origin. |

Thermal, degraded-link, generic XID and NVLink-XID faults alter selected values and may also change the common capacity factor. The exact substitutions are implemented in `Simulator.emit` and `Simulator.fault_factor`.

## Serving and orchestration metrics

The active backend determines which metric family is emitted. The queue is **waiting plus transferring** requests; running is the simulator's active list. These names resemble real systems but the data comes from the analytical simulation.

| Metric | Meaning and calculation |
|---|---|
| `vllm:num_requests_waiting` / `vllm:num_requests_running` | For vLLM mode, simulated waiting-plus-transfer and active request counts. |
| `vllm:gpu_cache_usage_perc` | For vLLM mode, current active/transfer KV bytes divided by total modeled decode-group KV capacity; a ratio despite the name. |
| `vllm:generation_tokens_total` | Cumulative generated-token counter attributed to the active vLLM backend/model. |
| `vllm:time_to_first_token_seconds_{bucket,sum,count}` | Histogram of simulated arrival-to-first-generated-token latency in vLLM mode. |
| `nv_trt_llm_request_metrics` | In TensorRT-LLM mode, `request_type=context` is waiting plus transfer and `generation` is active requests; these labels are the simulator's mapping. |
| `nv_trt_llm_kv_cache_block_metrics` | In TensorRT-LLM mode, total modeled blocks are `int(KV_capacity / (64 × kv_bytes_per_token))`; `used_blocks=int(total × kv_ratio)`, `free_blocks=max(0,total−used)`. |
| `nv_trt_llm_generation_tokens_total` | Cumulative generated-token counter attributed to the active TensorRT-LLM backend/model. |
| `sim_trt_llm_time_to_first_token_seconds_{bucket,sum,count}` | Simulator-specific TTFT histogram for TensorRT-LLM mode. |
| `nv_inference_pending_request_count` | When the Triton wrapper is enabled, the same waiting-plus-transfer count, labeled by model/version; do not add it to the backend queue. |
| `nv_inference_request_success_total` | When the wrapper is enabled, cumulative simulated completions attributed to the wrapper/model. |
| `dynamo_prefix_cache_hit_tokens_total` | Cumulative prompt tokens credited as cache hits at admission. |
| `dynamo_nixl_kv_transfer_bytes_total` | Cumulative simulated handoff bytes, based on completed prompt tokens times KV bytes per token. |
| `dynamo_inter_node_ib_tx_bytes_per_sec` | Current simulated cross-node transfer rate outside the NVL72 domain. |
| `sim_completed_requests_total` | Cumulative simulator completions; Prometheus `rate` gives completed requests/s. |
| `sim_workload_target_rps`, `sim_workload_prompt_tokens`, `sim_workload_output_tokens` | Current configured workload gauges. |
| `sim_config_revision_seconds` | Timestamp of the most recent simulator config reset, used to validate evidence windows. |
| `sim_fault_capacity_factor` | Current modeled capacity multiplier due to the active fault. |

The dashboard and agents compute rates over Prometheus windows; the agents' queue growth uses `deriv` and p95 TTFT uses `histogram_quantile` over bucket rates. Values depend on the selected observation window, scrape freshness and histogram bucket resolution. Missing series or source checks should remain visible as missing or insufficient, not be interpreted as zero.

## How the three agents use the system

**Profiler** obtains simulator state and cluster snapshot, queries scoped Prometheus history, validates source quality, and applies deterministic diagnosis rules. Its report includes findings, measurement quality and limitations. **Scheduler** invokes that profiling step, examines pressure and current baseline, audits KServe/Kueue resources and may offer a narrow prefill/decode rebalance for approval. **Benchmark** first proposes a configured sweep; after approval it adjusts offered load, waits, collects measured samples, classifies them against criteria and restores the approved baseline when possible. The coordinator asks the configured LLM to summarize role results, but its prose can be inaccurate. Structured evidence, current approval state and the benchmark report remain the basis for operator decisions. Full interactions and controls are described in [dashboard.md](dashboard.md), and executable steps are in [operator-runbook.md](operator-runbook.md).
