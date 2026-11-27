# Dashboard implementation and operator guide

The dashboard is a Vite application in `ui/`, with HTML in `index.html`, orchestration and D3 charts in `src/main.js`, Kubernetes and serving tables in `src/cluster.js`, and styles in `src/style.css`. It is a client for the simulator, Prometheus and coordinator; it does not implement the simulator physics or execute Kueue admission. [operator-runbook.md](operator-runbook.md) gives startup, authentication and live validation steps.

## Connectivity and modes

Vite loads the project-root `.env` and supplies development-server proxies: `/api/sim` → `SIM_URL`, `/api/prom` → `PROM_URL`, `/api/agent` → `AGENT_URL` when configured, and `/api/real-prom` → `REAL_PROM_URL` when configured. The dashboard computer's `.env` is separate from the remote coordinator's `.env`. A browser sends its operator token to the coordinator through the Vite agent proxy; the coordinator verifies the bearer token. The token input is not a durable approval store.

In the demonstrated lab, Vite runs on the Mac, simulator/Prometheus run on the remote host, and `AGENT_URL` points at a local SSH tunnel to the remote loopback coordinator. The LLM server can run on the Mac, but the remote coordinator must be able to reach it directly. The Developer server's root, model and authentication settings are configured on the coordinator, not in the browser.

The environment selector chooses **Simulated** or **Real GPU**. Simulated mode enables simulator configuration, faults, agent actions and sweeps. Real mode reads `REAL_PROM_URL` if configured and hides or disables simulation-only actions; it is an observational path, not a validated production integration. The node-label control chooses the Prometheus label used to group heatmap rows (`node` in simulation, initially `Hostname` in real mode). Missing Prometheus data is shown as unavailable rather than proof that utilization is zero.

The UI requests telemetry approximately every three seconds, refreshes the Kubernetes snapshot approximately every five seconds while its view is active, refreshes serving data approximately every ten seconds, and polls approvals approximately every five seconds. Timers initiate requests; completion depends on the network and backend. **Refresh** requests Kubernetes and serving updates. These are independent snapshots, not a transaction spanning all components.

## Sidebar controls

### Cluster & software stack

Topology, GPU profile, backend and wrapper, model parameters, precision, TP size, orchestration, prefill/decode GPUs, chunk budget, cache hit ratio, KV bytes/token and transfer latency configure the analytical engine. Changing topology updates suggested UI values; the API remains responsible for validation. **Apply config & reset requests** posts the form to `/config`; it clears in-flight simulated requests and resets the simulated fault as part of applying config. The displayed values are what will be submitted.

### Workload and sweep

**Workload** sets arrival RPS and prompt/output token means through `/workload`, affecting future arrivals without clearing the existing queue. A growing or shrinking queue depends on both current arrivals and previous backlog.

**Automated RPS sweep** fields specify start, end, step and hold. The sidebar **Start sweep** is a direct dashboard-driven simulator sweep, separate from the agent's human-approved benchmark: it changes offered workload, collects chart samples, and leaves the last workload active when finished; **Stop** cancels it. The chart can export those direct-sweep samples as CSV or Markdown. Do not confuse this control with the benchmark agent.

When **Benchmark** is selected under Agents, **Run agent workflow** reads the same sweep fields to construct an approval plan. Inspect its exact `rates`, `hold_seconds` and baseline before approving: changing the form later does not edit an existing proposal. The coordinator requires one to six points and applies a minimum 130-second hold per point. Rejection does not start a sweep. Approval starts the coordinator-managed benchmark, which attempts to restore its original workload. Review the reported restoration result.

### Fault injection

The selector injects a simulated thermal throttle, NVLink degradation, generic XID or profile-specific NVLink XID. Controls select node/tray or GPU and, for NVLink XID, relevant link, peer, severity and observation details. **Inject** posts to `/fault`; **Clear** removes the simulated fault. These are simulator controls, not real GPU operations. A fault can modify model capacity and specific synthetic telemetry.

### Environment & endpoints

The endpoint fields show dashboard proxy paths and the environment and node-label choices. The status message reports current connectivity or errors. Updating project `.env` proxy targets requires restarting Vite; changing a browser endpoint field affects the current client session. Do not place secrets in URL fields.

## Simulator view

### GPU telemetry heatmap

The metric selector chooses SM occupancy, tensor activity, DRAM activity, GPU utilization, temperature or power. D3 groups Prometheus samples by node and GPU, colors values with a sequential scale and marks the simulated group role. Hover text identifies the model, node, GPU, role and value. These are synthetic metrics in simulated mode; the formulas and caveats are in [concept.md](concept.md).

### Time-series charts

The charts query a rolling Prometheus window and show the last value per returned series. Missing data displays as unavailable. The implemented charts are:

| Chart | Meaning |
|---|---|
| SM occupancy, tensor activity, DRAM activity | Averages grouped by role. |
| Waiting / running requests | Backend-appropriate queue and active counts; waiting includes transfers. |
| Offered / completed RPS | Configured target and the rate of cumulative simulator completions. A backlog can make completions exceed current arrivals. |
| Generated tokens/s | Rate of the active backend's cumulative token counter. |
| KV utilization | vLLM ratio or TensorRT-LLM used/(used+free) simulated blocks. |
| TTFT p95 | Prometheus histogram quantile of time to **first token**, subject to bucket and window limits. |
| Rack power | Sum of simulated GPU watts converted to kW. |
| NVLink TX / inter-node IB | Sum of synthetic transfer-rate proxies, converted to GB/s. |
| Dynamo cache tokens/s / KV transfer GB/s | Rates of cumulative simulated cache-credit and handoff counters. |
| Triton pending requests | Wrapper queue when the simulated wrapper is enabled; it is not additional demand to add to the backend queue. |

### Latency–throughput sweep

The graph plots sampled completed requests/s against TTFT p95 for the direct sidebar sweep. It is not the coordinator's benchmark classification. CSV and Markdown exports contain the direct sweep's collected points; they are not a durable database. A direct sweep leaves its last workload active. The agent benchmark's structured report and approval state appear in the agent panel instead.

## Kubernetes & KEDA view

The upper status describes snapshot connectivity and errors. Summary cards show ready virtual nodes, advertised GPUs, Dynamo worker GPU requests, running workers and pending workers. The worker request summary is intentionally scoped to the Dynamo worker set; it is not the all-Pod inventory total.

- **Worker replicas & HPA:** current/desired replicas, min/max, metric values and conditions from the default namespace HPA.
- **KEDA autoscaler:** ScaledObject target, thresholds, poll/cooldown, conditions and last-active time.
- **Control-plane components and pods:** observed simulator/KEDA deployments and selected control-plane Pod readiness.
- **GPU inventory and placement:** node kind, Ready/schedulable state and requested versus allocatable GPUs. Requests are summed for bound, nonterminal Pods across namespaces; this is not utilization. Cluster-wide Pod read permission is required. If unavailable, the table displays unavailable rather than zero.
- **Simulator & worker pods:** the selected default-namespace simulator and Dynamo Pods, placement, state and requests.
- **Recent scheduling events:** relevant default-namespace events from the snapshot; this is not a full-cluster event audit.
- **KServe, Kueue & Gateway:** read through the authenticated agent coordinator. Tables show InferenceServices, predictor Deployments, ClusterQueues, LocalQueues, Workload admission, predictor Pod placement, Gateways and HTTPRoutes, including conditions. Each table may report its own access error. A programmed Gateway and accepted route are control-plane status; a KWOK Ready predictor does not prove HTTP inference works.

The **Agent summary** above the tables is the latest displayed LLM summary or workflow status. It is advisory: compare it with detailed findings, proposals and benchmark report. Markdown is rendered after sanitization. Missing data, global queues repeated in per-node findings and finite latency buckets require careful interpretation.

## Agents & human approvals

Choose an agent, enter an objective, supply the operator token, and click **Run agent workflow**. The role and objective are sent to the coordinator. The objective provides context for the LLM summary; it does not create arbitrary tools or change the coordinator's deterministic execution rules.

- **Profiler** reads simulator and Prometheus evidence and produces a diagnosis, measurement quality and limitations. It does not mutate simulator or Kubernetes state.
- **Scheduler** uses profiler evidence and the current state to decide whether its limited simulator rebalance rule can propose moving one TP group. Its serving audit observes KServe and Kueue; the separate service controls create a KServe submission proposal. A proposal is not an executed action.
- **Benchmark** creates a pending plan from the sidebar sweep settings. After human approval it runs the sweep, collects evidence and reports classification and restoration; it can change offered load while running.

The page displays **Profiler diagnosis**, **Scheduler proposals & trace**, **Benchmark report**, service inspection and pending approvals. Running, preparing, pending, rejected, failed and completed feedback depends on the workflow stage. Use **Approve & execute** or **Reject** only after inspecting the exact plan and impact. Applying a simulator rebalance resets in-flight requests. Agent summaries are advisory; a local LLM may omit or infer details incorrectly and cannot approve actions.

The **KServe and Kueue** controls let the operator propose a uniquely named 1–4 GPU simulated service and inspect admission/placement. Approval submits the manifest; Kueue and kube-scheduler subsequently make independent decisions. The serving route/status tables do not run an inference test.

**Approval history** is collapsed by default and reflects only the current coordinator process. It is neither Postgres nor RustFS persistence; coordinator restart clears pending and completed records. The copy icons in output blocks copy their text when browser clipboard access is available. The approvals list polls without rebuilding unchanged entries so reading position is retained.

## Known limits

The dashboard is a Vite development application, not a complete production identity or deployment platform. Agent/coordinator state is ephemeral; browser reload can reset form values and selections. A simulator restart or config apply resets in-flight state. LLM summaries are not validated diagnoses and benchmark wording has shown errors. See [next-steps.md](next-steps.md) for unresolved usability, persistence, validation and real-serving work.
