#!/usr/bin/env bash
set -euo pipefail

CLUSTER_NAME="gpu-sim-cluster"
NODE_NAME="${CLUSTER_NAME}-control-plane"

command -v docker >/dev/null || { printf 'Docker is required on the remote cluster host.\n' >&2; exit 1; }
if ! docker inspect "$NODE_NAME" >/dev/null 2>&1; then
  printf 'Container %s was not found on this host.\n' "$NODE_NAME" >&2
  exit 1
fi

cluster_label="$(docker inspect --format '{{ index .Config.Labels "io.x-k8s.kind.cluster" }}' "$NODE_NAME")"
if [[ "$cluster_label" != "$CLUSTER_NAME" ]]; then
  printf 'Container %s is not the expected kind cluster node.\n' "$NODE_NAME" >&2
  exit 1
fi

if [[ "$(docker inspect --format '{{.State.Running}}' "$NODE_NAME")" != 'true' ]]; then
  printf 'Cluster %s is already stopped.\n' "$CLUSTER_NAME"
  exit 0
fi

printf 'Stopping cluster %s...\n' "$CLUSTER_NAME"
docker stop --time 30 "$NODE_NAME" >/dev/null
printf 'Cluster stopped. Its container and cluster data have been retained.\n'
