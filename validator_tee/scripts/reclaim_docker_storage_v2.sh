#!/bin/bash
# Reclaim validator-builder Docker storage, including an orphaned data root.

set -euo pipefail

MIN_FREE_BYTES="${VALIDATOR_DOCKER_MIN_FREE_BYTES:-30000000000}"
LIVE_RUNTIME_MIN_FREE_BYTES="${VALIDATOR_DOCKER_LIVE_RUNTIME_MIN_FREE_BYTES:-18000000000}"
ALLOW_DATA_ROOT_RESET="${VALIDATOR_DOCKER_ALLOW_DATA_ROOT_RESET:-0}"
PRUNE_ATTEMPTS="${VALIDATOR_DOCKER_PRUNE_ATTEMPTS:-5}"
SETTLE_ATTEMPTS="${VALIDATOR_DOCKER_SETTLE_ATTEMPTS:-30}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

for setting in PRUNE_ATTEMPTS SETTLE_ATTEMPTS; do
  value="${!setting}"
  if ! [[ "$value" =~ ^[1-9][0-9]*$ ]] || [ "$value" -gt 300 ]; then
    echo "ERROR: $setting must be between 1 and 300" >&2
    exit 2
  fi
done
for setting in MIN_FREE_BYTES LIVE_RUNTIME_MIN_FREE_BYTES; do
  value="${!setting}"
  if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: $setting must be a positive integer byte count" >&2
    exit 2
  fi
done

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

run_prune_with_retry() {
  local label="$1"
  shift
  local attempt
  for attempt in $(seq 1 "$PRUNE_ATTEMPTS"); do
    if "$@" >/dev/null; then
      return 0
    fi
    if [ "$attempt" -lt "$PRUNE_ATTEMPTS" ]; then
      echo "Docker $label prune did not settle; retrying ($attempt/$PRUNE_ATTEMPTS)" >&2
      sleep 2
    fi
  done
  echo "ERROR: Docker $label prune failed after $PRUNE_ATTEMPTS attempts" >&2
  return 1
}

run_prune_with_retry image docker image prune --all --force
run_prune_with_retry builder docker builder prune --all --force
run_prune_with_retry system docker system prune --all --force

# The release builder invokes this helper while the deployed validator
# containers are intentionally running. Wait for containerd teardown only
# after Docker reports an empty runtime, which is the restart/data-root
# recovery path. Healthy live containers are inventoried and preserved below.
if ! INITIAL_CONTAINER_IDS="$(docker ps -aq 2>/dev/null)"; then
  echo "ERROR: Docker container inventory is unreadable after prune" >&2
  exit 1
fi
INITIAL_CONTAINER_COUNT="$(
  printf '%s\n' "$INITIAL_CONTAINER_IDS" \
    | awk 'NF { count += 1 } END { print count + 0 }'
)"
if [ "$INITIAL_CONTAINER_COUNT" -ne 0 ]; then
  echo "Reclaiming stale Docker overlay mounts without disturbing live containers"
  PYTHONPATH="$REPO_ROOT" python3 \
    -m validator_tee.host.docker_stale_mount_reclaimer_v2
  # A stale Nitro build mount can keep otherwise unreferenced layers alive.
  # Retry the normal Docker-owned reclamation only after the guarded unmount.
  run_prune_with_retry image docker image prune --all --force
  run_prune_with_retry builder docker builder prune --all --force
  run_prune_with_retry system docker system prune --all --force
fi
if [ "$INITIAL_CONTAINER_COUNT" -eq 0 ]; then
  TEARDOWN_SETTLED=0
  TEARDOWN_PROBE_FAILED=0
  CONTAINER_COUNT=-1
  CONTAINERD_RUNNING_TASK_COUNT=-1
  for attempt in $(seq 1 "$SETTLE_ATTEMPTS"); do
    TEARDOWN_PROBE_FAILED=0
    if ! CONTAINER_IDS="$(docker ps -aq 2>/dev/null)"; then
      TEARDOWN_PROBE_FAILED=1
    elif ! CONTAINERD_TASKS="$(sudo ctr -n moby tasks list 2>/dev/null)"; then
      TEARDOWN_PROBE_FAILED=1
    else
      CONTAINER_COUNT="$(printf '%s\n' "$CONTAINER_IDS" | awk 'NF { count += 1 } END { print count + 0 }')"
      CONTAINERD_RUNNING_TASK_COUNT="$(
        printf '%s\n' "$CONTAINERD_TASKS" \
          | awk 'NR > 1 && $3 == "RUNNING" { count += 1 } END { print count + 0 }'
      )"
    fi
    if [ "$TEARDOWN_PROBE_FAILED" -eq 0 ] \
        && [ "$CONTAINER_COUNT" -eq 0 ] \
        && [ "$CONTAINERD_RUNNING_TASK_COUNT" -eq 0 ]; then
      TEARDOWN_SETTLED=1
      break
    fi
    if [ "$attempt" -lt "$SETTLE_ATTEMPTS" ]; then
      if [ "$attempt" -eq 1 ] || [ $((attempt % 10)) -eq 0 ]; then
        if [ "$TEARDOWN_PROBE_FAILED" -eq 1 ]; then
          echo "Waiting for Docker teardown state to become readable ($attempt/$SETTLE_ATTEMPTS)" >&2
        else
          echo "Waiting for Docker teardown to settle: containers=$CONTAINER_COUNT containerd_running_tasks=$CONTAINERD_RUNNING_TASK_COUNT ($attempt/$SETTLE_ATTEMPTS)" >&2
        fi
      fi
      sleep 1
    fi
  done
  if [ "$TEARDOWN_SETTLED" -ne 1 ]; then
    if [ "$TEARDOWN_PROBE_FAILED" -eq 1 ]; then
      echo "ERROR: Docker teardown state remained unreadable after $SETTLE_ATTEMPTS settle attempts" >&2
    elif [ "$CONTAINER_COUNT" -ne 0 ]; then
      echo "ERROR: refusing Docker recovery while $CONTAINER_COUNT container(s) remain after $SETTLE_ATTEMPTS settle attempts" >&2
    else
      echo "ERROR: refusing containerd reset while $CONTAINERD_RUNNING_TASK_COUNT moby task(s) are running after $SETTLE_ATTEMPTS settle attempts" >&2
    fi
    exit 1
  fi
fi

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
REQUIRED_FREE_BYTES="$MIN_FREE_BYTES"
RUNTIME_MODE="empty"
if [ "$CONTAINER_COUNT" -ne 0 ]; then
  REQUIRED_FREE_BYTES="$LIVE_RUNTIME_MIN_FREE_BYTES"
  RUNTIME_MODE="live"
fi
echo "Docker storage requirement: runtime_mode=$RUNTIME_MODE required_free_bytes=$REQUIRED_FREE_BYTES"
if [ "$AVAILABLE" -ge "$REQUIRED_FREE_BYTES" ] \
    && [ "$ORPHANED_DOCKER_STATE" -eq 0 ]; then
  echo "Docker storage ready: free_bytes=$AVAILABLE required_free_bytes=$REQUIRED_FREE_BYTES runtime_mode=$RUNTIME_MODE"
  exit 0
fi

if [ "$CONTAINER_COUNT" -ne 0 ]; then
  echo "ERROR: live validator runtime has only $AVAILABLE free bytes; $REQUIRED_FREE_BYTES are required for an independent recovery build" >&2
  echo "ERROR: refusing to stop containers or reset Docker storage from the release builder" >&2
  exit 1
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
