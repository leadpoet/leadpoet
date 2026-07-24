#!/bin/bash
# Reclaim validator-builder Docker storage, including an orphaned data root.

set -euo pipefail

MIN_FREE_BYTES="${VALIDATOR_DOCKER_MIN_FREE_BYTES:-30000000000}"
ALLOW_DATA_ROOT_RESET="${VALIDATOR_DOCKER_ALLOW_DATA_ROOT_RESET:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

. "$SCRIPT_DIR/docker_operation_lock_v2.sh"
leadpoet_acquire_docker_operation_lock_v2
PYTHONPATH="$REPO_ROOT" python3 \
  -m validator_tee.host.docker_operation_guard_v2 \
  --wait \
  --timeout-seconds 1800 \
  --interval-seconds 3 \
  --proc-root "${LEADPOET_PROC_ROOT:-/proc}"

available_bytes() {
  df --output=avail -B1 / | tail -1 | tr -d '[:space:]'
}

if ! docker info >/dev/null 2>&1; then
  echo "Docker is unavailable; recovering builder daemons before inventory"
  sudo systemctl start containerd.service docker.service
  DAEMON_READY=0
  for _attempt in $(seq 1 30); do
    if docker info >/dev/null 2>&1 \
        && sudo ctr -n moby containers list -q >/dev/null 2>&1; then
      DAEMON_READY=1
      break
    fi
    sleep 1
  done
  if [ "$DAEMON_READY" -ne 1 ]; then
    echo "ERROR: Docker/containerd did not recover before storage inventory" >&2
    exit 1
  fi
fi

docker image prune --all --force >/dev/null
docker builder prune --all --force >/dev/null
docker system prune --all --force >/dev/null

AVAILABLE="$(available_bytes)"
CONTAINER_COUNT="$(docker ps -aq | wc -l | tr -d '[:space:]')"
IMAGE_COUNT="$(docker image ls -aq | sort -u | sed '/^$/d' | wc -l | tr -d '[:space:]')"
VOLUME_COUNT="$(docker volume ls -q | wc -l | tr -d '[:space:]')"
DOCKER_ROOT="$(docker info --format '{{.DockerRootDir}}')"
CONTAINERD_ROOT="${VALIDATOR_CONTAINERD_ROOT:-/var/lib/containerd}"
DOCKER_ROOT_BYTES="$(sudo du -sx -B1 "$DOCKER_ROOT" | awk '{print $1}')"
CONTAINERD_CONTAINER_COUNT="$(
  sudo ctr -n moby containers list -q | sed '/^$/d' | wc -l | tr -d '[:space:]'
)"
CONTAINERD_TASK_COUNT="$(
  sudo ctr -n moby tasks list -q | sed '/^$/d' | wc -l | tr -d '[:space:]'
)"
CONTAINERD_RUNNING_TASK_COUNT="$(
  sudo ctr -n moby tasks list \
    | awk 'NR > 1 && $3 == "RUNNING" { count += 1 } END { print count + 0 }'
)"
NON_MOBY_NAMESPACE_COUNT="$(
  sudo ctr namespaces list -q \
    | sed '/^$/d; /^moby$/d' \
    | wc -l \
    | tr -d '[:space:]'
)"
MOBY_SHIM_COUNT="$(
  pgrep -fc '^/usr/bin/containerd-shim-runc-v2 -namespace moby ' || true
)"
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
    && { [ "$CONTAINERD_CONTAINER_COUNT" -ne 0 ] \
      || [ "$CONTAINERD_TASK_COUNT" -ne 0 ] \
      || [ "$MOBY_SHIM_COUNT" -ne 0 ] \
      || { [ "$IMAGE_COUNT" -eq 0 ] \
        && { [ "$LAYERDB_IMAGE_COUNT" -ne 0 ] \
          || [ "$LAYERDB_MOUNT_COUNT" -ne 0 ] \
          || [ "$OVERLAY_DIRECTORY_COUNT" -ne 0 ]; }; }; }; then
  ORPHANED_DOCKER_STATE=1
fi

echo "Docker storage state: free_bytes=$AVAILABLE root_bytes=$DOCKER_ROOT_BYTES containers=$CONTAINER_COUNT images=$IMAGE_COUNT volumes=$VOLUME_COUNT containerd_containers=$CONTAINERD_CONTAINER_COUNT containerd_tasks=$CONTAINERD_TASK_COUNT containerd_running_tasks=$CONTAINERD_RUNNING_TASK_COUNT moby_shims=$MOBY_SHIM_COUNT non_moby_namespaces=$NON_MOBY_NAMESPACE_COUNT layerdb_images=$LAYERDB_IMAGE_COUNT layerdb_mounts=$LAYERDB_MOUNT_COUNT overlay_directories=$OVERLAY_DIRECTORY_COUNT orphaned=$ORPHANED_DOCKER_STATE"
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
if [ "$CONTAINERD_ROOT" != "/var/lib/containerd" ]; then
  echo "ERROR: refusing unexpected containerd data-root reset: $CONTAINERD_ROOT" >&2
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
if [ "$CONTAINERD_RUNNING_TASK_COUNT" -ne 0 ]; then
  echo "ERROR: refusing containerd reset while $CONTAINERD_RUNNING_TASK_COUNT moby task(s) are running" >&2
  exit 1
fi
if [ "$NON_MOBY_NAMESPACE_COUNT" -ne 0 ]; then
  echo "ERROR: refusing containerd reset while $NON_MOBY_NAMESPACE_COUNT non-moby namespace(s) exist" >&2
  exit 1
fi

echo "Resetting orphaned Docker and containerd roots after guarded empty-runtime check"
sudo systemctl stop docker.service docker.socket containerd.service
sudo pkill -TERM -f '^/usr/bin/containerd-shim-runc-v2 -namespace moby ' 2>/dev/null || true
sleep 2
sudo pkill -KILL -f '^/usr/bin/containerd-shim-runc-v2 -namespace moby ' 2>/dev/null || true

# Stale overlay mounts can survive daemon shutdown even though every guarded
# runtime inventory above is empty. Unmount only descendants of the two exact
# validated data roots, deepest paths first, and refuse a lazy/forced unmount.
while IFS= read -r mount_target; do
  echo "Unmounting stale empty-runtime mount: $mount_target"
  sudo umount "$mount_target"
done < <(
  findmnt -rn -o TARGET \
    | awk -v docker_root="$DOCKER_ROOT/" -v containerd_root="$CONTAINERD_ROOT/" \
        'index($0, docker_root) == 1 || index($0, containerd_root) == 1' \
    | awk '{ print length($0), $0 }' \
    | sort -rn \
    | cut -d' ' -f2-
)
if findmnt -rn -o TARGET \
    | awk -v docker_root="$DOCKER_ROOT/" -v containerd_root="$CONTAINERD_ROOT/" \
        'index($0, docker_root) == 1 || index($0, containerd_root) == 1' \
    | grep -q .; then
  echo "ERROR: stale Docker/containerd mounts remain after guarded unmount" >&2
  exit 1
fi

sudo rm -rf --one-file-system "$DOCKER_ROOT"
sudo rm -rf --one-file-system "$CONTAINERD_ROOT"
sudo install -d -m 0711 -o root -g root "$DOCKER_ROOT"
sudo install -d -m 0711 -o root -g root "$CONTAINERD_ROOT"
sudo systemctl start containerd.service docker.service

RUNTIME_READY=0
for _attempt in $(seq 1 30); do
  if docker info >/dev/null 2>&1 \
      && sudo ctr -n moby containers list -q >/dev/null 2>&1; then
    RUNTIME_READY=1
    break
  fi
  sleep 1
done
if [ "$RUNTIME_READY" -ne 1 ]; then
  echo "ERROR: Docker/containerd did not become ready after reset" >&2
  exit 1
fi

if [ "$(sudo ctr -n moby containers list -q | sed '/^$/d' | wc -l | tr -d '[:space:]')" -ne 0 ] \
    || [ "$(sudo ctr -n moby tasks list -q | sed '/^$/d' | wc -l | tr -d '[:space:]')" -ne 0 ] \
    || [ "$(pgrep -fc '^/usr/bin/containerd-shim-runc-v2 -namespace moby ' || true)" -ne 0 ]; then
  echo "ERROR: Docker/containerd reset left stale moby runtime state" >&2
  exit 1
fi

AVAILABLE="$(available_bytes)"
if [ "$AVAILABLE" -lt "$MIN_FREE_BYTES" ]; then
  echo "ERROR: Docker data-root reset left only $AVAILABLE free bytes" >&2
  exit 1
fi
echo "Docker storage recovered: free_bytes=$AVAILABLE"
