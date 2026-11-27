#!/usr/bin/env bash
set -euo pipefail
dir="$(cd "$(dirname "$0")" && pwd)"
context="${1:-kind-gpu-sim-cluster}"
quota="${2:-}"
cpu_quota="${3:-64}"
memory_gi="${4:-512}"
if [[ ! "$cpu_quota" =~ ^[1-9][0-9]*$ || ! "$memory_gi" =~ ^[1-9][0-9]*$ ]]; then
  echo "CPU and memory quotas must be positive integers (cores and GiB)." >&2
  exit 2
fi
if [[ ! "$quota" =~ ^([1-9]|[1-6][0-9]|7[0-2])$ ]]; then
  echo "Usage: bash $0 <kind-context> <serving-gpu-quota: 1-72> [cpu-cores] [memory-GiB]" >&2
  exit 2
fi
k=(kubectl --context "$context")
h=(helm --kube-context "$context")

"${h[@]}" upgrade --install gpu-sim-cert-manager oci://quay.io/jetstack/charts/cert-manager \
  --version v1.17.4 --namespace cert-manager --create-namespace --set crds.enabled=true --wait
"${h[@]}" upgrade --install gpu-sim-envoy oci://docker.io/envoyproxy/gateway-helm \
  --version v1.3.3 --namespace envoy-gateway-system --create-namespace --wait
"${k[@]}" wait --for=condition=Available deployment/envoy-gateway -n envoy-gateway-system --timeout=180s
"${k[@]}" create namespace kserve --dry-run=client -o yaml | "${k[@]}" apply -f -
"${k[@]}" apply -f "$dir/gateway.yaml"
"${k[@]}" wait --for=condition=Accepted gatewayclass/gpu-sim-envoy --timeout=180s
"${h[@]}" upgrade --install kserve-crd oci://ghcr.io/kserve/charts/kserve-crd \
  --version v0.15.0 --namespace kserve --create-namespace --wait
"${h[@]}" upgrade --install kserve oci://ghcr.io/kserve/charts/kserve \
  --version v0.15.0 --namespace kserve \
  --set kserve.controller.deploymentMode=RawDeployment \
  --set kserve.controller.gateway.ingressGateway.enableGatewayApi=true \
  --set kserve.controller.gateway.ingressGateway.kserveGateway=kserve/gpu-sim-gateway --wait

"${k[@]}" apply --server-side -f https://github.com/kubernetes-sigs/kueue/releases/download/v0.13.0/manifests.yaml
"${k[@]}" -n kueue-system patch configmap kueue-manager-config --type merge --patch-file "$dir/kueue-config.yaml"
"${k[@]}" -n kueue-system rollout restart deployment/kueue-controller-manager
"${k[@]}" -n kueue-system rollout status deployment/kueue-controller-manager --timeout=180s
sed -e "s/__GPU_QUOTA__/$quota/" -e "s/__CPU_QUOTA__/$cpu_quota/" -e "s/__MEMORY_QUOTA__/${memory_gi}Gi/" "$dir/queues.yaml" | "${k[@]}" apply -f -
