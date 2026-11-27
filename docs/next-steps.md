# Next steps

This file records remaining work after the current simulation-lab implementation and live tests. Items are opportunities for a future work cycle, not claims that they are implemented or that the present lab cannot be used.

## Improve operator confidence

1. **Benchmark interpretation.** Improve the benchmark-specific LLM prompt and evidence package. The live model attributed a draining backlog to possible miscounting/retries and expanded TTFT as time to first byte. Include criteria, workload and relevant context; distinguish a 60-second passing point from a proven long-term capacity. Test summaries for backlog drainage, failing latency, missing measurements and restoration. Keep deterministic measurements and approval outcomes authoritative; prompts alone cannot guarantee accuracy.
2. **Latency measurement limits.** Verify and document the behavior of the TTFT histogram at its largest finite bucket and its implications for `histogram_quantile`. Surface clipped or indeterminate results and source/measurement limitations where appropriate; do not silently treat a bucket boundary as an exact latency.
3. **Dashboard usability.** Preserve non-sensitive benchmark form values across reloads and make the exact proposed rates prominent before approval. Prevent confusing stale reports and distinguish planning, awaiting approval, running, failed, rejected and completed states. Validate the current status indicators and Markdown rendering in a live browser, including long-running benchmarks and lost coordinator connections.
4. **Approval history.** Decide whether session-only history meets operator needs. If durable audit is required, design a persistent store, retention, access controls, restart recovery and cross-process claiming before selecting Postgres, RustFS or another backend. Existing approvals and history are in memory only; do not imply durability or exactly-once behavior across restarts.

## Deployment and validation

5. **Clean installation.** Resolve the mismatch between `scripts/setup-cluster.sh`, which expects root-level Docker files, and the repository's `docker/` directory. Update scripts, build context and runbook together without breaking the established remote layout. Have an independent operator follow [operator-runbook.md](operator-runbook.md) from a clean checkout and record failures and fixes.
6. **Live environment coverage.** Recheck after deployment or component version changes: all namespace Pod-request accounting and RBAC; Prometheus target scope/freshness; dashboard connection and recovery; all three agent roles; approval rejection, stale baselines, execution failures and restoration; KServe/Kueue contention and release; and Gateway/HTTPRoute conditions. Test the simulator state reset and queue behavior explicitly.
7. **Serving data plane.** The present KWOK/pause-image fixture validates controllers and placement only. If actual inference is in scope, choose a real runnable predictor and node, then test HTTP routing, hostname/path, readiness and a real inference response. Review Cloud Provider KIND behavior with fake-node backends and reconcile Gateway API version compatibility before changing controller flags or CRDs.
8. **Real hardware.** Calibrate the analytical model and validate exporter fields, source freshness, fault semantics, node/GPU identity mapping and network behavior against a real GPU environment before using results for operations or capacity planning.
9. **Services and access.** Decide whether to supervise the coordinator, Cloud Provider KIND and dashboard for restart recovery. Document and implement appropriately scoped Kubernetes permissions, token handling, secure network exposure and operational observability for any environment beyond this trusted lab.

## Future experiments and agent work

10. **Capacity experiments.** Compare 24/48 and 16/56 prefill/decode configurations under identical workload, fault-free conditions and controlled starting queues. A configuration apply resets in-flight requests, so plan comparable baselines and record limitations. A single passing 5 RPS point is only a lower bound within the sampled window.
11. **Additional agent capabilities.** Capture new user requirements before expanding agent reasoning, tools or coordination. Preserve human approval and deterministic validation of mutations; distinguish model suggestions from measured evidence and executed actions.
12. **Reference cleanup.** Historical upgrade samples in `upgrade-2-reference/` are retained for provenance. Review their continuing value separately; they are not runtime code or the current design. Update the operator runbook and current architecture, concept and design documents as behavior changes.
