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
docker system prune --all --force >/dev/null

AVAILABLE="$(available_bytes)"
CONTAINER_COUNT="$(docker ps -aq | wc -l | tr -d '[:space:]')"
IMAGE_COUNT="$(docker image ls -aq | sort -u | sed '/^$/d' | wc -l | tr -d '[:space:]')"
VOLUME_COUNT="$(docker volume ls -q | wc -l | tr -d '[:space:]')"
DOCKER_ROOT="$(docker info --format '{{.DockerRootDir}}')"
DOCKER_ROOT_BYTES="$(sudo du -sx -B1 "$DOCKER_ROOT" | awk '{print $1}')"
LAYERDB_IMAGE_COUNT=0
LAYERDB_MOUNT_COUNT=0
OVERLAY_DIRECTORY_COUNT=0
if sudo test -d "$DOCKER_ROOT/image/overlay2/layerdb/sha256"; then
  LAYERDB_IMAGE_COUNT="$(
    sudo find "$DOCKER_ROOT/image/overlay2/layerdb/sha256" \
      -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d '[:space:]'
  )"
fi
if sudo test -d "$DOCKER_ROOT/image/overlay2/layerdb/mounts"; then
  LAYERDB_MOUNT_COUNT="$(
    sudo find "$DOCKER_ROOT/image/overlay2/layerdb/mounts" \
      -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d '[:space:]'
  )"
fi
if sudo test -d "$DOCKER_ROOT/overlay2"; then
  OVERLAY_DIRECTORY_COUNT="$(
    sudo find "$DOCKER_ROOT/overlay2" \
      -mindepth 1 -maxdepth 1 -type d ! -name l \
      | wc -l | tr -d '[:space:]'
  )"
fi

ORPHANED_DOCKER_STATE=0
if [ "$CONTAINER_COUNT" -eq 0 ] \
    && [ "$IMAGE_COUNT" -eq 0 ] \
    && { [ "$LAYERDB_IMAGE_COUNT" -ne 0 ] \
      || [ "$LAYERDB_MOUNT_COUNT" -ne 0 ] \
      || [ "$OVERLAY_DIRECTORY_COUNT" -ne 0 ]; }; then
  ORPHANED_DOCKER_STATE=1
fi

echo "Docker storage state: free_bytes=$AVAILABLE root_bytes=$DOCKER_ROOT_BYTES containers=$CONTAINER_COUNT images=$IMAGE_COUNT volumes=$VOLUME_COUNT layerdb_images=$LAYERDB_IMAGE_COUNT layerdb_mounts=$LAYERDB_MOUNT_COUNT overlay_directories=$OVERLAY_DIRECTORY_COUNT orphaned=$ORPHANED_DOCKER_STATE"
if [ "$AVAILABLE" -ge "$MIN_FREE_BYTES" ] \
    && [ "$ORPHANED_DOCKER_STATE" -eq 0 ]; then
  echo "Docker storage ready: free_bytes=$AVAILABLE"
  exit 0
fi

if [ "$ALLOW_DATA_ROOT_RESET" != "1" ]; then
  echo "ERROR: Docker storage requires a guarded reset after prune" >&2
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
if [ "$IMAGE_COUNT" -ne 0 ]; then
  echo "ERROR: refusing Docker data-root reset while $IMAGE_COUNT image(s) remain" >&2
  exit 1
fi
if [ "$VOLUME_COUNT" -ne 0 ]; then
  echo "ERROR: refusing Docker data-root reset while $VOLUME_COUNT volume(s) remain" >&2
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
