#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
CLUSTER_NAME="gpu-sim-cluster"
CONTEXT="kind-${CLUSTER_NAME}"
TOPOLOGY="${1:-rack_72gpu_nvl72}"
if (( $# > 1 )); then
  printf 'Usage: %s [node_8gpu|cluster_16gpu|rack_72gpu_nvl72]\n' "$0" >&2
  exit 2
fi
KWOK_VERSION="v0.7.0"
KEDA_VERSION="2.17.2"
KIND_NODE_IMAGE="kindest/node:v1.32.2@sha256:f226345927d7e348497136874b6d207e0b32cc52154ad8323129352923a3142f"

case "$TOPOLOGY" in
  node_8gpu|cluster_16gpu|rack_72gpu_nvl72) ;;
  *) printf 'Usage: %s [node_8gpu|cluster_16gpu|rack_72gpu_nvl72]\n' "$0" >&2; exit 2 ;;
esac
for path in compose.yaml Dockerfile pyproject.toml uv.lock README.md src/gpu_sim/main.py scripts/render_kind_inventory.py deploy/kind/cluster.yaml deploy/kind/services.yaml deploy/kind/observability-rbac.yaml deploy/kind/prometheus-scrape.yaml; do
  if [[ ! -f "$ROOT_DIR/$path" ]]; then
    printf 'Missing deployment file: %s\n' "$ROOT_DIR/$path" >&2
    printf 'Copy the complete gpu-sim deployment package to this directory before running setup.\n' >&2
    exit 1
  fi
done
for command in docker kind kubectl python3 curl; do
  command -v "$command" >/dev/null || { printf 'Missing required command: %s\n' "$command" >&2; exit 1; }
done
docker compose version >/dev/null
kind_version="$(kind version)"
if [[ ! "$kind_version" =~ v([0-9]+)\.([0-9]+)\.([0-9]+) ]] || (( BASH_REMATCH[1] == 0 && BASH_REMATCH[2] < 27 )); then
  printf 'kind v0.27.0 or newer is required for the pinned node image; found: %s\n' "$kind_version" >&2
  exit 1
fi
if kind get clusters | grep -Fxq "$CLUSTER_NAME"; then
  printf 'Cluster %s already exists. No changes made.\n' "$CLUSTER_NAME" >&2
  exit 1
fi
printf '%s\n' 'Building engine image on this host...'
docker compose build gpu-simulator
printf '%s\n' 'Creating the single-node kind cluster...'
kind create cluster --name "$CLUSTER_NAME" --image "$KIND_NODE_IMAGE" --config deploy/kind/cluster.yaml
kind load docker-image gpu-simulator:kind --name "$CLUSTER_NAME"
K=(kubectl --context "$CONTEXT")

printf '%s\n' 'Installing pinned KWOK and KEDA...'
"${K[@]}" apply -f "https://github.com/kubernetes-sigs/kwok/releases/download/${KWOK_VERSION}/kwok.yaml"
"${K[@]}" apply -f "https://github.com/kubernetes-sigs/kwok/releases/download/${KWOK_VERSION}/stage-fast.yaml"
"${K[@]}" rollout status deployment/kwok-controller -n kube-system --timeout=180s
"${K[@]}" apply --server-side -f "https://github.com/kedacore/keda/releases/download/v${KEDA_VERSION}/keda-${KEDA_VERSION}.yaml"
"${K[@]}" rollout status deployment/keda-operator -n keda --timeout=180s
"${K[@]}" rollout status deployment/keda-metrics-apiserver -n keda --timeout=180s
"${K[@]}" rollout status deployment/keda-admission -n keda --timeout=180s

printf '%s\n' 'Creating virtual GPU nodes...'
python3 scripts/render_kind_inventory.py "$TOPOLOGY" nodes | "${K[@]}" apply -f -
node_patch="$(python3 scripts/render_kind_inventory.py "$TOPOLOGY" status)"
node_names="$(python3 -c 'import json,sys; print("\n".join(n["metadata"]["name"] for n in json.load(sys.stdin)["items"]))' < <(python3 scripts/render_kind_inventory.py "$TOPOLOGY" nodes))"
while IFS= read -r node; do
  "${K[@]}" patch node "$node" --subresource=status --type=merge -p "$node_patch"
  "${K[@]}" wait --for=condition=Ready "node/$node" --timeout=120s
done <<< "$node_names"

printf '%s\n' 'Deploying the real simulator engine...'
"${K[@]}" apply -f deploy/kind/observability-rbac.yaml
"${K[@]}" apply -f deploy/kind/services.yaml
"${K[@]}" rollout status deployment/gpu-simulator-engine --timeout=180s
config_payload="$(python3 scripts/render_kind_inventory.py "$TOPOLOGY" config)"
curl --fail-with-body --silent --show-error -H 'Content-Type: application/json' -X POST -d "$config_payload" http://127.0.0.1:8080/config >/dev/null

# The remote .env can override the address KEDA uses from inside kind.
keda_prom_url="$(python3 -c '
from pathlib import Path
import shlex
p = Path(".env")
if p.is_file():
    for line in p.read_text().splitlines():
        key, sep, value = line.partition("=")
        if sep and key.strip() == "KEDA_PROM_URL":
            parts = shlex.split(value, comments=True)
            print(parts[0] if parts else "")
            break
')"
if [[ -z "$keda_prom_url" ]]; then
  gateway="$(docker network inspect kind | python3 -c '
import ipaddress, json, sys
for config in json.load(sys.stdin)[0]["IPAM"]["Config"]:
    gateway = config.get("Gateway")
    if gateway and ipaddress.ip_address(gateway).version == 4:
        print(gateway)
        break
')"
  if [[ -z "$gateway" ]]; then
    printf 'Could not determine the kind network host gateway; set KEDA_PROM_URL in .env.\n' >&2
    exit 1
  fi
  keda_prom_url="http://${gateway}:9090"
fi
printf '%s\n' 'Deploying virtual worker pool and KEDA autoscaler...'
python3 scripts/render_kind_inventory.py "$TOPOLOGY" workers --prometheus-url "$keda_prom_url" | "${K[@]}" apply -f -
"${K[@]}" get nodes -l "gpu-sim.io/inventory=${TOPOLOGY//_/-}"
"${K[@]}" get deployment,scaledobject,hpa,pods -o wide
printf '\nCluster created: %s (topology: %s)\n' "$CLUSTER_NAME" "$TOPOLOGY"
printf '%s\n' 'Simulator control :8080 and metrics :9400/:8002 listen on remote-host interfaces.'
printf '%s\n' 'Existing Prometheus must scrape the simulator metrics; see deploy/kind/prometheus-scrape.yaml.'
printf '%s\n' 'Set SIM_URL and PROM_URL in the dashboard .env to the remote host address.'
