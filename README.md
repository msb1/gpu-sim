# GPU simulator

GPU-sim is an interactive lab for exploring GPU inference workloads, performance telemetry, and scheduling. Its FastAPI simulator models fixed GPU inventories, tensor-parallel replicas, prefill and decode processing, request queues, configurable workloads, and injected faults. It exports simulated DCGM-style and inference-serving metrics to Prometheus; the Vite dashboard provides configuration controls, charts, and Kubernetes views. The model is an analytical approximation, not calibrated hardware telemetry or a real inference server.

The Kubernetes lab runs the simulator in kind and uses KWOK virtual nodes to represent GPU inventory. KEDA scales simulated workers, while KServe, Kueue, and kube-scheduler demonstrate how an approved inference-service request is submitted, admitted against quota, and placed. The predictor Pods are simulated fixtures: their Ready status does not mean a model container is serving inference.

Three LangGraph agent roles work with the dashboard through a separate coordinator: **Profiler** analyzes scoped telemetry and diagnoses supported conditions; **Scheduler** audits placement and can propose a simulator prefill/decode rebalance; and **Benchmark** plans and runs approved workload sweeps. A configured local LLM provides advisory summaries, while deterministic checks and human approval govern proposed actions. Approval history is held in coordinator memory and is lost on restart.

For a complete remote kind, KServe, Kueue, dashboard and agent setup, follow the [operator runbook](docs/operator-runbook.md). It includes live validation and the current setup-script packaging workaround. Current documentation: [concept](docs/concept.md), [architecture](docs/architecture.md), [design](docs/design.md), [dashboard](docs/dashboard.md), and [next steps](docs/next-steps.md).

For a standalone simulator, run `uv sync` and `uv run gpu-sim`. In `ui`, run `npm install` then `npm run dev` and open the displayed Vite URL. The control API is on 8080, DCGM metrics on 9400, and serving metrics on 8002. Vite proxies the API and an existing Prometheus server.

POST `/config` validates a merged configuration update and clears in-flight requests. POST `/workload` updates the workload without clearing requests. GET `/state` exposes current settings and state; `/docs` provides schemas. This is an analytical approximation, not calibrated hardware telemetry.

## Simulator capabilities

The backend now models fixed single/8/16/72-GPU inventories, explicit TP replicas
and NVL72 tray membership. The default is GB200 NVL72, 24P/48D, TP8, DeepSeek
671B/37B active, FP4. vLLM and TensorRT-LLM are selectable, each direct or
Triton-wrapped, with colocated or Dynamo-style disaggregated execution.

/config and /workload now accept partial objects merged into current validated
settings. Configuration updates report `requests_reset: true`; workload changes
preserve in-flight work. Changing topology requires supplying a valid P/D/TP
allocation (and model fit); omitted counts are not silently guessed. Colocated
execution uses the entire selected inventory. GET /state includes replica/node
membership, fixed-inventory mode, fault factor and modeling assumptions.
POST /fault supports thermal_throttle, nvlink_degraded and xid_error targeted at
one node or global GPU index; clear restores the normal factor without resetting
cumulative counters. See /docs for request/response schemas.

Runtime is the FastAPI simulator, not an executable inference server. The
dashboard, Kubernetes integration and agent coordinator are separate components.

Extended validation is available through `scripts/check_simulator.py` (smoke, matrix and soak modes); live setup and test procedures are in the operator runbook.

## Remote kind deployment

`scripts/setup-cluster.sh` is intended to run on the remote Docker host. It builds the
engine image with the build-only Compose service and creates one kind node
container containing the real engine, KWOK and KEDA. It uses the existing
Prometheus server on the remote PC; no Prometheus container is created. The autoscaled
inference-worker pods are virtual KWOK objects; their containers do not run.
Physics remains fixed-inventory while KEDA scales those virtual workers.

The setup supports `node_8gpu`, `cluster_16gpu` and `rack_72gpu_nvl72` (default).
It requires Docker with Compose, kind v0.27.0 or newer, kubectl, Python 3 and
curl on the remote machine, plus network access to the pinned images and release
manifests. The script refuses to replace an existing cluster with the same name.
It binds simulator control port 8080 and metric ports 9400 and 8002 to all
remote-host interfaces. Add the jobs in deploy/kind/prometheus-scrape.yaml to
the existing Prometheus configuration. KEDA defaults to the kind Docker bridge
host gateway on port 9090; configure KEDA_PROM_URL in the remote root .env if
the existing Prometheus is reachable at a different address from inside kind. Put dashboard `SIM_URL` and
`PROM_URL` in the existing root `.env`, never in a Docker image. The agent
coordinator currently runs as a separate host process; no agent image is included.

Failed setup can leave a partially created cluster for inspection. The cluster does not make the engine or KEDA production
ready; validate the deployment on the remote host before relying on its behavior.

### LangGraph agent coordinator

The simulator dashboard can use a separate LangGraph service. Keep it on
loopback and run one process so each approval is claimed once. Pending and
completed approvals are in memory and are lost on restart.
The dashboard's collapsed approval history covers only the current coordinator
session; it is not stored in PostgreSQL or RustFS.

Set `AGENT_URL=http://127.0.0.1:8081` and a private
`AGENT_OPERATOR_TOKEN` in the root `.env`. The coordinator also uses the
existing `SIM_URL`, `PROM_URL`, and `ANALYTICS_{DCGM,SERVING}_{JOB,INSTANCE}`
settings.

For advisory agent summaries, configure `OPENAI_BASE_URL` as the base URL of an
OpenAI-compatible API, `MAIN_MODEL` as its model name, and `OPENAI_API_KEY` if
the server requires one. These settings belong in the remote coordinator's
`.env`. The model summarizes profiler and scheduler
results and completed benchmark results; deterministic checks and human approval
continue to govern actions. An unavailable model is reported in the dashboard.

```bash
uv run --env-file .env uvicorn gpu_sim.agents:app --host 127.0.0.1 --port 8081
```

Restart the Vite dev server and enter the operator token in the dashboard.
Profiler reads scoped telemetry. Scheduler audits the cluster snapshot and can
propose moving a tensor-parallel group between simulator pools. Benchmark
publishes up to six exact load steps with a hold of at least 130 seconds per
point. Approve or reject using the pending approval card. Keep the dashboard tab
open for the result and avoid manual simulator edits during a sweep.

Run the coordinator HTTP workflow tests with
`uv run python -m unittest discover -s tests -p test_agents.py`.
For a live smoke test, run Profiler, then Scheduler. Plan a one-point sweep
at the baseline RPS, reject it, and verify that the workload stays unchanged.
Plan it again and approve it. After the hold, check the Markdown report and
verified workload restoration in approval history. Repeating a decision on the
same ID returns HTTP 409.

### KServe and Kueue on the remote kind lab

On the remote PC, from a checkout with these files and Helm installed, run:

```bash
bash deploy/serving/install.sh kind-gpu-sim-cluster 4
```

The installer is for this lab's dedicated controllers and configures Kueue for
the `gpu-serving` namespace. The existing kind control plane supplies
`default-scheduler`. Configure `AGENT_KUBE_CONTEXT=kind-gpu-sim-cluster`
where the coordinator runs, with a kubeconfig that can reach that context.
The coordinator needs list access to InferenceServices, LocalQueues, Workloads
and Pods in `gpu-serving`, get access to the LocalQueue and create access to
InferenceServices there.

The second argument sets the serving quota independently of the 72-GPU inventory.
For the smoke test, use quota 4 and request 4 GPUs per predictor in the dashboard.
The single predictor can request 1–4 GPUs, matching one tray.
Propose `serving-smoke`. Verify the service is absent before
approval. Approve once, then inspect admission and placement. For a quota test,
approve a second uniquely named service while the first still holds the four-GPU
quota. Inspect its Workload and Pod scheduling gates. Use a rack with free
physical scheduling capacity in the virtual inventory, since the existing
KEDA workers share those nodes and are outside this LocalQueue.

```bash
kubectl --context kind-gpu-sim-cluster -n gpu-serving get inferenceservices,localqueues,workloads,pods -o wide
kubectl --context kind-gpu-sim-cluster -n gpu-serving get pods -L kueue.x-k8s.io/queue-name -o yaml
kubectl --context kind-gpu-sim-cluster -n gpu-serving describe workloads
```

Confirm KServe propagated the queue label to the predictor Pod. Confirm the
first Workload is admitted and its Pod has a node bound by the default
scheduler. Confirm the second Workload is waiting for quota. A submitted
InferenceService alone does not establish admission or placement. The KWOK
predictor is a placement fixture and has no model-serving endpoint.

To change the quota after installation, render and apply only the queue
manifests. For example, to set a 16-GPU serving quota:

```bash
sed -e 's/__GPU_QUOTA__/16/' -e 's/__CPU_QUOTA__/64/' \
  -e 's/__MEMORY_QUOTA__/512Gi/' deploy/serving/queues.yaml |
  kubectl --context kind-gpu-sim-cluster apply -f -
```

A Kueue quota is an admission budget. Pod placement still depends on free
GPUs on individual trays and any capacity used by the existing KEDA workers.

The serving installer accepts optional CPU-core and memory-GiB quota arguments after the GPU quota. Defaults are 64 cores and 512 GiB. These are Kueue admission budgets and should be selected for the intended workload.

The **Kubernetes & KEDA** dashboard includes KServe, Kueue and Gateway status.
The shared agent panel connects the Profiler, Scheduler and Benchmark agents
to both dashboard views. Authenticate with the operator token to load serving
data. In addition to its original permissions, the coordinator needs list
access to Deployments and HTTPRoutes in `gpu-serving`, ClusterQueues at cluster
scope, and Gateways in `kserve`. Individual resource errors appear in the
corresponding dashboard section.
