# Operator runbook: kind, dashboard and agents

This runbook describes the current **simulation lab**, not a production GPU cluster. One Ubuntu host runs Docker, kind, a real simulator process inside the kind control-plane node, and a separate agent coordinator. KWOK nodes and their worker and KServe predictor Pods are simulated: a `Running` or `Ready` KWOK Pod does not run an inference container. The dashboard and LM Studio may run on a separate computer. The examples use `192.168.1.50` for the remote host and `192.168.1.60` for the dashboard/LM Studio host; replace these addresses and the SSH user for your environment.

## 1. Prerequisites and files

On the remote Ubuntu host, obtain the project files, Docker and the Compose plugin, kind v0.27.0 or newer, kubectl, Helm, Python 3, curl, uv, an existing Prometheus server, and the `cloud-provider-kind` binary. The user running setup needs Docker access and a kubeconfig writable by kind. The host needs access to the container images and release manifests used by the scripts. Project Python dependencies require Python 3.14; `uv sync` can provision it if configured to do so.

On the dashboard computer, obtain the same project revision, Node.js and npm, an SSH client, and LM Studio with a model available through its OpenAI-compatible server. The coordinator must be able to reach LM Studio over the network. The dashboard needs network access to the remote simulator and Prometheus, plus an SSH tunnel for the coordinator.

**Current packaging mismatch:** the repository stores the Compose file and Dockerfile in `docker/`, while `scripts/setup-cluster.sh` currently checks for both in the project root and runs Compose there. For a fresh remote checkout, create the files it expects without overwriting existing root files:

```bash
cd ~/Docker/gpu-sim
cp -n docker/Dockerfile Dockerfile
cp -n docker/compose.yaml compose.yaml
```

If root files already exist, inspect them and confirm they match `docker/Dockerfile` and `docker/compose.yaml` before proceeding. The Compose build context must be the project root. This workaround does not modify the setup script.

## 2. Remote simulator and Prometheus

On the remote host, prepare the existing Prometheus configuration using `deploy/kind/prometheus-scrape.yaml`. Add the two jobs under `scrape_configs`, replacing `REMOTE_PC_IP` with an address reachable by Prometheus, then reload or restart Prometheus using the procedure for that installation. The job names and target addresses in that file are examples: the agent environment below must match the **actual** Prometheus `job` and `instance` labels exactly. KEDA must be able to reach Prometheus from the kind network; the script uses the kind Docker bridge gateway on port 9090 unless `KEDA_PROM_URL` is set in the remote `.env`.

For a **new cluster only**, run on the remote host:

```bash
cd ~/Docker/gpu-sim
bash scripts/setup-cluster.sh rack_72gpu_nvl72
```

The script refuses to replace an existing `gpu-sim-cluster`. It installs pinned KWOK and KEDA versions, builds and loads the image, creates the simulated nodes, and deploys the simulator and worker pool. A failure may leave partial resources; inspect them before retrying. The cluster exposes simulator control on remote port 8080 and metrics on 9400 and 8002. These ports bind to host interfaces; restrict access to the trusted lab network.

Verify on the remote host:

```bash
kubectl --context kind-gpu-sim-cluster get nodes
kubectl --context kind-gpu-sim-cluster -n default get deployments,pods,hpa,scaledobjects -o wide
curl --noproxy '*' -fsS http://127.0.0.1:8080/state
curl --noproxy '*' -fsS http://127.0.0.1:9090/-/ready
```

In Prometheus, check **Status → Targets** and verify both simulator targets are up; record their exact job and instance labels. The simulator configuration endpoint resets in-flight requests; changing workload does not clear them. Restarting the simulator engine also resets its in-memory state.

## 3. Install KServe, Kueue and the Gateway

On the remote host, after the kind cluster is ready:

```bash
cd ~/Docker/gpu-sim
bash deploy/serving/install.sh kind-gpu-sim-cluster 4
```

The example sets the Kueue serving quota to four GPUs, with defaults of 64 CPU cores and 512 GiB memory. These are admission budgets, separate from the simulated 72-GPU inventory. The installer uses KServe RawDeployment, Envoy Gateway and Kueue; it installs pinned controllers and may require network access and time for webhooks to become ready. If it fails, inspect its error and controller readiness before retrying.

For this lab, run the installed Cloud Provider KIND **in a foreground terminal on the remote computer**, using the kubeconfig of the user who created the cluster:

```bash
sudo env KUBECONFIG="$HOME/.kube/config" \
  "$HOME/go/bin/cloud-provider-kind" --gateway-channel=disabled --enable-default-ingress=false
```

Adjust the binary path if installed elsewhere. Leave this process running. The disabled Gateway channel avoids the observed `ReferenceGrant` version mismatch with the lab's existing Gateway API CRDs; Envoy Gateway continues to manage the Gateway. A process launched inside SSH may stop when the session ends.

In another remote terminal:

```bash
kubectl --context kind-gpu-sim-cluster -n kserve get gateway gpu-sim-gateway
kubectl --context kind-gpu-sim-cluster -n envoy-gateway-system get services
kubectl --context kind-gpu-sim-cluster get clusterqueue gpu-serving
kubectl --context kind-gpu-sim-cluster -n gpu-serving get localqueue gpu-serving
```

Expect a programmed Gateway and an assigned LoadBalancer address. Its HTTP listener is port **80** on the displayed address, not necessarily on the host's LAN address. An HTTP 404 for `/` without a matching hostname/route does not establish a Gateway failure. The KWOK pause-image predictor cannot validate real inference traffic.

## 4. Configure and run the agent coordinator

On the remote host, run `uv sync` from the project root. Set values in its root `.env`; replace the example IPs, job labels, instance labels, token and model. Do not paste real credentials into shared documentation.

```dotenv
SIM_URL=http://127.0.0.1:8080
PROM_URL=http://127.0.0.1:9090
AGENT_KUBE_CONTEXT=kind-gpu-sim-cluster
AGENT_OPERATOR_TOKEN=replace-with-a-private-token
ANALYTICS_DCGM_JOB=gpu-sim-dcgm
ANALYTICS_DCGM_INSTANCE=192.168.1.50:9400
ANALYTICS_SERVING_JOB=gpu-sim-serving
ANALYTICS_SERVING_INSTANCE=192.168.1.50:8002
OPENAI_BASE_URL=http://192.168.1.60:1234/v1
OPENAI_API_KEY=local-not-required
MAIN_MODEL=replace-with-the-loaded-model-id
```

The LM Studio server must listen on a network interface reachable from the remote host and have the specified model loaded. The coordinator calls `/chat/completions` on the configured base URL. Model output summarizes deterministic agent results; it does not authorize or execute actions. A model connection failure is shown as an unavailable summary.

Run one coordinator process in a remote terminal:

```bash
cd ~/Docker/gpu-sim
uv sync
uv run --env-file .env uvicorn gpu_sim.agents:app --host 127.0.0.1 --port 8081
```

It uses the remote user's kubeconfig and `AGENT_KUBE_CONTEXT`. That user needs permission to read the serving resources and create InferenceServices. All pending approvals and approval history are in this process's memory; they are **not** persisted to Postgres or RustFS and are lost when it restarts. Review pending approvals before stopping it.

## 5. Connect and start the dashboard

On the dashboard computer, set its own project-root `.env` to use addresses reachable from that computer. It is a separate file from the remote `.env`:

```dotenv
SIM_URL=http://192.168.1.50:8080
PROM_URL=http://192.168.1.50:9090
AGENT_URL=http://127.0.0.1:18081
```

Open an SSH tunnel **on the dashboard computer** and leave that terminal open:

```bash
ssh -N -L 18081:127.0.0.1:8081 barnwaldo@192.168.1.50
```

The tunnel is normally silent. If local port 18081 is in use, choose another local port and update the local `.env` `AGENT_URL`; the remote coordinator can remain on 8081.

In another local terminal:

```bash
cd /Users/msb/Code/gpu-sim/ui
npm install
npm run dev
```

Use the URL printed by Vite, usually `http://localhost:5173`. Open **Kubernetes & KEDA**, enter the remote coordinator's operator token, and click outside the field to connect. The page shows the cluster, KEDA, KServe, queues, Gateway and agent controls. `Refresh` requests current Kubernetes data. Agent results appear in the shared agent panel; the LLM summary appears near the top of the Kubernetes page. The approval history is collapsed and session-only. Changes to Vite environment settings require restarting Vite.

## 6. Live validation: KServe → Kueue → scheduler

This test changes lab resources. Use unique names and a four-GPU serving quota. In the dashboard's **KServe and Kueue** controls, propose a four-GPU service named `serving-check-1`; inspect its pending approval and approve it. Verify from the remote host:

```bash
kubectl --context kind-gpu-sim-cluster -n gpu-serving get inferenceservices,workloads,pods -o wide
kubectl --context kind-gpu-sim-cluster describe clusterqueue gpu-serving
```

Expect a Kueue workload admitted in `gpu-serving` and a predictor assigned to a KWOK node. The dashboard's **Workload admission** and **Predictor Pod placement** tables should agree. GPU inventory counts bound Pod requests across namespaces; it reports requests, not GPU utilization.

Propose and approve another four-GPU service, `serving-check-2`. With the first service consuming the entire queue quota, inspect:

```bash
kubectl --context kind-gpu-sim-cluster -n gpu-serving get workloads,pods -o wide
kubectl --context kind-gpu-sim-cluster -n gpu-serving describe workloads
```

Expect the second Pod to be `SchedulingGated` and its workload pending for insufficient GPU quota. When ready to remove the first test service and release its quota:

```bash
kubectl --context kind-gpu-sim-cluster -n gpu-serving delete inferenceservice serving-check-1
kubectl --context kind-gpu-sim-cluster -n gpu-serving get workloads -o wide -w
```

After the second workload shows `ADMITTED=True`, stop the watch with Ctrl+C and inspect its node:

```bash
kubectl --context kind-gpu-sim-cluster -n gpu-serving get inferenceservices,workloads,pods,httproutes -o wide
kubectl --context kind-gpu-sim-cluster -n gpu-serving describe httproutes
kubectl --context kind-gpu-sim-cluster -n kserve describe gateway gpu-sim-gateway
```

Expect admitted status and placement, accepted and resolved routes, and a programmed Gateway. Delete `serving-check-2` when the fixture is no longer needed. Since its predictor is a KWOK-simulated Pod with a pause image, this does not prove HTTP inference works.

## 7. Live validation: three agents

Run these from **Agents & human approvals** after authenticating. The objective is advisory text; supported findings, benchmark settings and action eligibility are determined by the application. Check the **Agent summary** near the top of the Kubernetes page and the detailed output below. If the model is not configured or reachable, the summary explicitly says it is unavailable.

### Profiler

1. With no fault active, choose **Profiler**, enter “Inspect GPU health and serving telemetry; summarize findings and missing measurements,” and click **Run agent workflow**. Inspect the summary, diagnosis, measurement quality and limitations. An empty finding set can be valid.
2. For a fault test, use **Fault injection** to inject **Thermal throttling** on a chosen tray. Allow telemetry to be collected, rerun the profiler with an objective to identify thermal throttling, and check for a `thermal_throttle` finding and the affected target. Clear the fault after the test. A missing or stale metric can prevent detection.

### Scheduler

1. With faults cleared, choose **Scheduler**, enter “Inspect scheduling and GPU pressure; propose a change only when supported by evidence,” and run it. If supported pressure conditions are absent, **No supported change justified by current evidence** and no proposal is a successful result.
2. If a rebalance proposal appears during sustained simulator pressure, inspect its reason, GPU counts, baseline and impact in **Pending approvals**. Reject it to test the human gate without changing the simulator. Check approval history and verify the simulator configuration remains unchanged. A proposal is not guaranteed. Approving a rebalance can reset in-flight simulator requests.

### Benchmark

The separate **Automated RPS sweep** sidebar controls are read when **Run agent workflow** creates a benchmark proposal. For a one-point 5 RPS test, enter **Start 5, End 5, Step 1, Hold 130**; choose **Benchmark** and run the workflow. Before any decision, inspect the new approval and verify it contains `"rates": [5]` and `"hold_seconds": 130`. Edits to fields do not update an existing proposal; reject a mismatched plan.

1. Reject one matching plan. Confirm no sweep runs and the decision appears in approval history.
2. Create a new matching plan, ensure the queue has settled and faults are clear, then approve it. Keep the coordinator running and avoid manual changes. Wait at least 130 seconds plus collection and model response time. Check the one-row benchmark report, LLM summary, and `workload_restored` in approval history. The point may pass or fail its criteria; either is a test result. Restoration changes offered workload back but does not necessarily clear accumulated queue backlog.

## 8. Restart and troubleshooting

`scripts/stop-cluster.sh` stops the kind node container while retaining cluster data. To restart that stopped node, run `docker start gpu-sim-cluster-control-plane` on the remote host, then verify node and Deployment readiness. Start the coordinator and Cloud Provider KIND again if they are not running. A new cluster should use the setup procedure, not `docker start`.

- **Coordinator unavailable:** verify its remote terminal, the local SSH tunnel and the local `AGENT_URL`. Check the operator token; do not share it in logs.
- **LLM summary unavailable:** verify `OPENAI_BASE_URL`, `MAIN_MODEL`, LM Studio's server binding and reachability from the remote PC. The coordinator's other outputs do not prove the LLM call succeeded.
- **Missing telemetry:** check Prometheus targets and ensure the four `ANALYTICS_*` labels match the actual target labels. Allow sufficient observation time.
- **Gateway unavailable:** inspect the Gateway and Envoy Service above, confirm Cloud Provider KIND is running, and check coordinator permissions for Gateways in `kserve`.
- **Kueue pending:** `describe workloads` for the precise reason. CPU and memory requests must also be covered by ClusterQueue quota. A quota reservation does not guarantee a suitable node has free GPUs.
- **Inventory says Unavailable:** ensure the simulator runs the updated image and the observer service account can list Pods across namespaces. Avoid interpreting missing data as zero.
