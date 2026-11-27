# GPU simulator design

This document describes the implementation in the current repository, not the earlier upgrade proposal. [concept.md](concept.md) explains the model; [architecture.md](architecture.md) covers component boundaries; [operator-runbook.md](operator-runbook.md) provides operational steps.

## Engine and API

`Simulator` maintains a one-second-tick analytical request lifecycle: waiting, prefill, optional transfer, decode and completion. Arrivals use a Poisson process with variable token lengths. Fixed inventory is divided into complete TP groups, and requests are admitted using full prompt/output KV reservations. A profile, precision and modeled efficiency determine compute and bandwidth estimates. These are heuristic, not measurements of real kernels, fabric contention or a real serving deployment.

The default simulator initializes an NVL72-style 72-GPU inventory, GB200 profile, TP8, a 24/48 prefill/decode split, TensorRT-LLM with a Triton wrapper and a DeepSeek-like model. Workload defaults are code defaults, not guaranteed live state. A config must fit the inventory, TP group and model memory constraints.

| Endpoint | Behavior |
|---|---|
| `GET /state` | Current config, workload, fault, inventory, replicas and state. |
| `POST /config` | Validates a partial configuration merged with current values; resets in-flight requests, simulated fault and related state. |
| `POST /workload` | Validates and updates workload without clearing queued requests. |
| `POST /fault` | Injects or clears a supported simulated fault. |
| `GET /cluster` | Read-only Kubernetes snapshot; per-resource errors remain visible. |

FastAPI lifespan owns the simulation loop and separate metrics exporters. The simulator state is process-local; run a single engine instance for this lab. Restarting or reapplying config resets simulation state as described by the operator runbook.

## Telemetry and evidence

Prometheus scrapes synthetic DCGM-style and serving metrics. Analytics queries explicitly configured DCGM and serving job/instance scopes, selects stack-appropriate metric families and checks freshness, source health, identities, counter resets and configuration revision. Measurement quality and limitations accompany findings; unavailable data is not silently converted to healthy zeros. The profiler recognizes supported XID, thermal throttle, NVLink CRC, prefill compute pressure, decode bandwidth pressure and low occupancy signals. Some findings repeat the same scoped serving waiting queue for each affected tray; it is one shared value, not a count per tray.

The dashboard reports GPU **requests** for bound nonterminal Pods from a cluster-wide Pod snapshot and allocatable node inventory. This does not measure actual GPU use. Failure to list Pods must appear as unavailable. KServe-specific admission and placement are separately read through the coordinator, so access and refresh behavior differ.

## Coordinator API and roles

`GET /approvals`, `POST /run`, `POST /benchmark`, `GET/POST /serving` and `POST /approvals/decide` require the configured operator bearer token. Profiler collects scoped telemetry and returns structured findings. Scheduler runs after profiling when selected and can propose moving one TP group between simulator pools only under its supported pressure, baseline and configuration checks. Its proposal records the impact of resetting in-flight requests. It separately audits serving resources; it is not a general Kubernetes scheduler.

Benchmark planning uses explicit start, end, step and hold values, permits one to six ascending points, enforces at least 130 seconds per point, and captures a baseline. Approved execution checks for outside state changes, applies offered rates, samples Prometheus and verifies restoration where possible. Current criteria require completed throughput of at least 90% of offered rate, queue growth at most 1 RPS and TTFT p95 at most 5 seconds. Classifications include `lower_bound`, `bracketed`, `all_failed` and `insufficient`. A 60-second observation and histogram-based p95 limit interpretation; a clean single passing point is not a long-term capacity guarantee.

`OPENAI_BASE_URL`, `OPENAI_API_KEY` and `MAIN_MODEL` configure optional OpenAI-compatible chat-completions summaries. The scheduler summary receives a labeled shared queue and instructions to report pending proposals accurately. LLM text is advisory, not a deterministic finding or execution record, and may be wrong. Benchmark narratives have made unsupported claims and need improvement. The dashboard sanitizes rendered Markdown.

Approval records and concurrency control are process-local and non-durable. A stale baseline blocks execution. A restarted coordinator loses history and pending records. See [next-steps.md](next-steps.md) for durability and reliability work.

## KServe integration

The serving installer configures a RawDeployment KServe fixture, Envoy Gateway and Kueue Pod integration. The coordinator proposes a labeled InferenceService with one simulated pause-image predictor, 1–4 GPUs, KWOK selection/toleration and default scheduler. After approval, KServe creates a Deployment and Pod, Kueue reserves CPU, memory and GPU quota, and Kubernetes schedules the admitted Pod. Serving inspection returns resources and per-resource errors, including workload conditions, Pod gates/placement, Gateway and HTTPRoute conditions.

The virtual predictor is not a real serving process. A Gateway address, accepted route or KWOK Ready condition demonstrates controller state, not a successful inference request. Cloud Provider KIND is run as a host foreground process in the demonstrated environment; Gateway channel is disabled there to avoid the observed ReferenceGrant CRD mismatch.

## Validation status

Repository tests cover simulator behavior, API, analytics, freshness, agents, cluster snapshots, manifests and dashboard helpers. Live lab observations confirmed profiler baseline and thermal-fault detection; scheduler proposal/rejection and improved proposal summary; a one-point 5 RPS benchmark with verified workload restoration; and dashboard-created KServe submission, Kueue quota waiting/release and KWOK placement. The clean 5 RPS point passed configured criteria during its window; it did not establish a maximum. The clean-install runbook, real-GPU telemetry and real inference serving have not been validated end to end.

## Detailed agent workflows and contracts

### Shared evidence and the Profiler

`POST /run` accepts a role (`profiler` or `scheduler`) and goal. The graph invokes `profile`, which reads `/state` and `/cluster`, then collects job/instance-scoped Prometheus measurements in a 60-second observation with a two-second step and ten-second freshness threshold. Source checks use two observation windows for freshness and revision validation. It reads simulator state again to reject a changed configuration/workload/fault baseline during collection. The profiler removes the explicit fault from the agent-facing context and diagnoses observed metrics rather than simply reporting the injected setting.

`incident_report` contains the user goal, findings, `measurement_quality`, cluster snapshot and limitations. The LLM summary receives a selected evidence representation, not an unrestricted copy of every UI table. Nodes may lack an unambiguous Kubernetes mapping; null is not a successful correlation. Missing source data, stale series, missing history and reset checks can make a result unavailable or insufficient. A finding represents a rule match, not a validated causal conclusion.

### Scheduler and service submission

For `role=scheduler`, the graph proceeds from profiling to `schedule`. The scheduler reads live simulator state and considers prefill compute pressure or decode bandwidth pressure only when exactly one supported direction is selected, the collected baseline remains consistent, no fault is active and the configuration is disaggregated. It tries a one-TP-group shift and validates the resulting configuration. A qualifying plan contains baseline, proposed prefill/decode counts, reason, impact and supporting findings. The approval executor rechecks current baseline and re-runs the deterministic diagnosis before applying a matching proposal. Config application resets in-flight simulator requests. Unsupported or ambiguous conditions produce no rebalance proposal.

Serving inspection is a read-only `kubectl` audit of services, Deployments, queues, Workloads, Pods, HTTPRoutes and Gateways; individual reads can fail independently. The dashboard's service proposal captures a queue UID and context. Execution checks both before creating the approved InferenceService. Submission is not admission, scheduling or model execution. More complete KServe/Kueue concepts and the exact live sequence are in [concept.md](concept.md) and [operator-runbook.md](operator-runbook.md).

### Benchmark lifecycle

The dashboard's Benchmark workflow creates a pending `/benchmark` plan with exact rates, hold, criteria and simulator baseline. The baseline includes configuration revision, configuration, workload and fault; an active fault prevents planning. A changed baseline causes approval execution to fail rather than applying a stale plan. Approval claims the in-memory record and a single-process execution lock. At each approved point the coordinator changes RPS, waits, obtains a new evidence window and derives completed RPS, queue growth and TTFT p95. It validates the observation, uses the `BenchmarkCriteria` model to classify it, and attempts baseline workload restoration if its expected state still matches.

The result carries a machine-readable classification and samples, a Markdown report and `workload_restored`; a failed restoration is surfaced as an execution failure. A reset of configuration is not part of normal benchmark restoration, so prior queue backlog can remain. A single passing point yields `lower_bound`, and `all_failed` only means none of the tested points passed. The benchmark may be long-running, and approval/history remain in coordinator memory. A summary from the model is added after the structured result; it is not itself the benchmark calculation.

### Model summaries and safety boundaries

The coordinator uses the configured OpenAI-compatible `/chat/completions` endpoint and `MAIN_MODEL`. Its system instructions ask for an evidence-based summary with uncertainty, and scheduler-specific instructions identify the shared queue and pending proposals. A request timeout or error returns a visible unavailable summary, not permission to execute a change. The dashboard displays sanitized Markdown. Model wording is not validated against the structured result; known benchmark errors are listed in [next-steps.md](next-steps.md). Agent goals are context, not commands that expand action authority.

The in-memory `records` map holds pending, executing, rejected, executed and failed decisions; a process lock serializes approved execution. The coordinator checks an operator token but does not provide durable history, multi-process coordination or a production identity system. Remote process restart loses those records. For current controls, polling and security scope, see [dashboard.md](dashboard.md).
