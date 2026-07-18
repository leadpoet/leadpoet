#!/bin/bash
# Reclaim validator-builder Docker storage, including an orphaned data root.

set -euo pipefail

MIN_FREE_BYTES="${VALIDATOR_DOCKER_MIN_FREE_BYTES:-30000000000}"
ALLOW_DATA_ROOT_RESET="${VALIDATOR_DOCKER_ALLOW_DATA_ROOT_RESET:-0}"

available_bytes() {
  df --output=avail -B1 / | tail -1 | tr -d '[:space:]'
}

docker image prune --all --force >/dev/null
docker builder prune --all --force >/dev/null
docker system prune --all --force --volumes >/dev/null

AVAILABLE="$(available_bytes)"
if [ "$AVAILABLE" -ge "$MIN_FREE_BYTES" ]; then
  echo "Docker storage ready: free_bytes=$AVAILABLE"
  exit 0
fi

CONTAINER_COUNT="$(docker ps -aq | wc -l | tr -d '[:space:]')"
DOCKER_ROOT="$(docker info --format '{{.DockerRootDir}}')"
DOCKER_ROOT_BYTES="$(sudo du -sx -B1 "$DOCKER_ROOT" | awk '{print $1}')"

if [ "$ALLOW_DATA_ROOT_RESET" != "1" ]; then
  echo "ERROR: Docker prune left only $AVAILABLE free bytes; Docker root uses $DOCKER_ROOT_BYTES bytes" >&2
  echo "ERROR: rerun only after stopping all validator containers with VALIDATOR_DOCKER_ALLOW_DATA_ROOT_RESET=1" >&2
  exit 1
fi
if [ "$CONTAINER_COUNT" -ne 0 ]; then
  echo "ERROR: refusing Docker data-root reset while $CONTAINER_COUNT container(s) exist" >&2
  exit 1
fi
if [ "$DOCKER_ROOT" != "/var/lib/docker" ]; then
  echo "ERROR: refusing unexpected Docker data-root reset: $DOCKER_ROOT" >&2
  exit 1
fi

echo "Resetting orphaned Docker data root after guarded empty-container check"
sudo systemctl stop docker.service docker.socket
sudo rm -rf --one-file-system "$DOCKER_ROOT"
sudo install -d -m 0711 -o root -g root "$DOCKER_ROOT"
sudo systemctl start docker.service

AVAILABLE="$(available_bytes)"
if [ "$AVAILABLE" -lt "$MIN_FREE_BYTES" ]; then
  echo "ERROR: Docker data-root reset left only $AVAILABLE free bytes" >&2
  exit 1
fi
echo "Docker storage recovered: free_bytes=$AVAILABLE"
