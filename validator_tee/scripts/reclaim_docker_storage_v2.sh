#!/bin/bash
# Reclaim validator-builder Docker storage, including an orphaned data root.

set -euo pipefail

MIN_FREE_BYTES="${VALIDATOR_DOCKER_MIN_FREE_BYTES:-30000000000}"
LIVE_RUNTIME_MIN_FREE_BYTES="${VALIDATOR_DOCKER_LIVE_RUNTIME_MIN_FREE_BYTES:-18000000000}"
ALLOW_DATA_ROOT_RESET="${VALIDATOR_DOCKER_ALLOW_DATA_ROOT_RESET:-0}"
ALLOW_LIVE_HOST_GATEWAY_PRUNE="${VALIDATOR_DOCKER_ALLOW_LIVE_HOST_GATEWAY_PRUNE:-0}"
REQUIRE_ZERO_RUNTIME_RECONCILE="${REQUIRE_ZERO_RUNTIME_RECONCILE-0}"
PRUNE_ATTEMPTS="${VALIDATOR_DOCKER_PRUNE_ATTEMPTS:-5}"
SETTLE_ATTEMPTS="${VALIDATOR_DOCKER_SETTLE_ATTEMPTS:-30}"
DAEMON_READY_ATTEMPTS="${VALIDATOR_DOCKER_DAEMON_READY_ATTEMPTS:-30}"
DAEMON_PROBE_TIMEOUT_SECONDS="${VALIDATOR_DOCKER_DAEMON_PROBE_TIMEOUT_SECONDS:-10}"
DAEMON_CONTROL_TIMEOUT_SECONDS="${VALIDATOR_DOCKER_DAEMON_CONTROL_TIMEOUT_SECONDS:-30}"
DAEMON_COMMAND_TIMEOUT_SECONDS="${VALIDATOR_DOCKER_DAEMON_COMMAND_TIMEOUT_SECONDS:-600}"
PROC_ROOT="${LEADPOET_PROC_ROOT:-/proc}"
ALLOW_NONSTANDARD_PROC_ROOT="${LEADPOET_DOCKER_ALLOW_NONSTANDARD_PROC_ROOT:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MAX_SHELL_INTEGER="9223372036854775807"

is_bounded_unsigned_shell_integer() {
  local value="$1"
  local LC_ALL=C

  if ! [[ "$value" =~ ^(0|[1-9][0-9]*)$ ]]; then
    return 1
  fi
  if [ "${#value}" -gt "${#MAX_SHELL_INTEGER}" ]; then
    return 1
  fi
  if [ "${#value}" -eq "${#MAX_SHELL_INTEGER}" ] \
      && [[ "$value" > "$MAX_SHELL_INTEGER" ]]; then
    return 1
  fi
  return 0
}

for setting in PRUNE_ATTEMPTS SETTLE_ATTEMPTS DAEMON_READY_ATTEMPTS DAEMON_PROBE_TIMEOUT_SECONDS DAEMON_CONTROL_TIMEOUT_SECONDS; do
  value="${!setting}"
  if ! is_bounded_unsigned_shell_integer "$value" \
      || [ "$value" -eq 0 ] \
      || [ "$value" -gt 300 ]; then
    echo "ERROR: $setting must be between 1 and 300" >&2
    exit 2
  fi
done
if [ "$PROC_ROOT" != "/proc" ] \
    && [ "$ALLOW_NONSTANDARD_PROC_ROOT" != "1" ]; then
  echo "ERROR: nonstandard process roots are allowed only in explicit rehearsal" >&2
  exit 2
fi
if ! is_bounded_unsigned_shell_integer "$DAEMON_COMMAND_TIMEOUT_SECONDS" \
    || [ "$DAEMON_COMMAND_TIMEOUT_SECONDS" -eq 0 ] \
    || [ "$DAEMON_COMMAND_TIMEOUT_SECONDS" -gt 3600 ]; then
  echo "ERROR: DAEMON_COMMAND_TIMEOUT_SECONDS must be between 1 and 3600" >&2
  exit 2
fi
for setting in MIN_FREE_BYTES LIVE_RUNTIME_MIN_FREE_BYTES; do
  value="${!setting}"
  if ! is_bounded_unsigned_shell_integer "$value" || [ "$value" -eq 0 ]; then
    echo "ERROR: $setting must be a positive bounded integer byte count" >&2
    exit 2
  fi
done
if [ "$ALLOW_LIVE_HOST_GATEWAY_PRUNE" != "0" ] \
    && [ "$ALLOW_LIVE_HOST_GATEWAY_PRUNE" != "1" ]; then
  echo "ERROR: VALIDATOR_DOCKER_ALLOW_LIVE_HOST_GATEWAY_PRUNE must be 0 or 1" >&2
  exit 2
fi
if [ "$REQUIRE_ZERO_RUNTIME_RECONCILE" != "0" ] \
    && [ "$REQUIRE_ZERO_RUNTIME_RECONCILE" != "1" ]; then
  echo "ERROR: REQUIRE_ZERO_RUNTIME_RECONCILE must be 0 or 1" >&2
  exit 2
fi
if [ "$REQUIRE_ZERO_RUNTIME_RECONCILE" = "1" ] \
    && [ "$ALLOW_LIVE_HOST_GATEWAY_PRUNE" != "1" ]; then
  echo "ERROR: REQUIRE_ZERO_RUNTIME_RECONCILE requires VALIDATOR_DOCKER_ALLOW_LIVE_HOST_GATEWAY_PRUNE=1" >&2
  exit 2
fi

. "$SCRIPT_DIR/docker_operation_lock_v2.sh"
leadpoet_acquire_docker_operation_lock_v2
PYTHONPATH="$REPO_ROOT" python3 \
  -m validator_tee.host.docker_operation_guard_v2 \
  --wait \
  --timeout-seconds 1800 \
  --interval-seconds 3 \
  --proc-root "$PROC_ROOT"

available_bytes() {
  local value

  if ! value="$(df --output=avail -B1 / | tail -1 | tr -d '[:space:]')"; then
    echo "ERROR: filesystem free-space state is unreadable" >&2
    return 1
  fi
  if ! is_bounded_unsigned_shell_integer "$value"; then
    echo "ERROR: filesystem free-space state is malformed" >&2
    return 1
  fi
  printf '%s\n' "$value"
}

inspect_exact_host_gateway_runtime() {
  local phase="$1"

  if ! HOST_GATEWAY_REPORT="$(
    PYTHONPATH="$REPO_ROOT" python3 \
      -m validator_tee.host.docker_operation_guard_v2 \
      --detect-exact-host-gateway \
      --proc-root "$PROC_ROOT"
  )"; then
    echo "ERROR: exact host gateway process state is unreadable; refusing Docker maintenance" >&2
    exit 1
  fi
  printf '%s\n' "$HOST_GATEWAY_REPORT"
  if ! HOST_GATEWAY_STATE="$(
    printf '%s' "$HOST_GATEWAY_REPORT" | python3 -c '
import json
import sys

document = json.load(sys.stdin)
if document.get("schema_version") != "leadpoet.host_gateway_process_guard.v1":
    raise SystemExit("invalid host gateway process report schema")
status = document.get("status")
process = document.get("gateway_process")
if status == "live":
    if not isinstance(process, dict) or not isinstance(process.get("pid"), int):
        raise SystemExit("invalid live host gateway process report")
    print("1 {}".format(process["pid"]))
elif status == "absent":
    if process is not None:
        raise SystemExit("invalid absent host gateway process report")
    print("0 0")
else:
    raise SystemExit("invalid host gateway process status")
'
  )"; then
    echo "ERROR: exact host gateway process report is invalid; refusing Docker maintenance" >&2
    exit 1
  fi
  read -r HOST_GATEWAY_LIVE HOST_GATEWAY_PID <<< "$HOST_GATEWAY_STATE"
  if ! is_bounded_unsigned_shell_integer "$HOST_GATEWAY_PID" \
      || { [ "$HOST_GATEWAY_LIVE" -eq 1 ] && [ "$HOST_GATEWAY_PID" -eq 0 ]; }; then
    echo "ERROR: exact host gateway process identity is out of range; refusing Docker maintenance" >&2
    exit 1
  fi
  HOST_GATEWAY_START_TIME_TICKS=0
  if [ "$HOST_GATEWAY_LIVE" -eq 1 ]; then
    if ! HOST_GATEWAY_START_TIME_TICKS="$(
      python3 - "$PROC_ROOT" "$HOST_GATEWAY_PID" <<'PY'
from pathlib import Path
import re
import sys

proc_root = Path(sys.argv[1])
pid = sys.argv[2]
if re.fullmatch(r"[1-9][0-9]*", pid) is None:
    raise SystemExit("invalid host gateway pid")
raw = (proc_root / pid / "stat").read_text(encoding="utf-8")
if len(raw) > 4096:
    raise SystemExit("host gateway stat exceeds its size bound")
closing = raw.rfind(")")
if closing < 0:
    raise SystemExit("host gateway stat is malformed")
fields = raw[closing + 2 :].split()
if len(fields) < 20 or re.fullmatch(r"[1-9][0-9]*", fields[19]) is None:
    raise SystemExit("host gateway start time is malformed")
print(fields[19])
PY
    )"; then
      echo "ERROR: exact host gateway start identity is unreadable; refusing Docker maintenance" >&2
      exit 1
    fi
    if ! is_bounded_unsigned_shell_integer "$HOST_GATEWAY_START_TIME_TICKS" \
        || [ "$HOST_GATEWAY_START_TIME_TICKS" -eq 0 ]; then
      echo "ERROR: exact host gateway start identity is out of range; refusing Docker maintenance" >&2
      exit 1
    fi
  fi
}

protect_exact_host_gateway_runtime() {
  local phase="$1"

  inspect_exact_host_gateway_runtime "$phase"
  if [ "$HOST_GATEWAY_LIVE" -eq 1 ]; then
    AVAILABLE="$(available_bytes)"
    echo "Docker storage requirement: runtime_mode=host-gateway-live phase=$phase required_free_bytes=$LIVE_RUNTIME_MIN_FREE_BYTES"
    if [ "$AVAILABLE" -ge "$LIVE_RUNTIME_MIN_FREE_BYTES" ] \
        && [ "$REQUIRE_ZERO_RUNTIME_RECONCILE" != "1" ]; then
      echo "Docker storage maintenance deferred while the exact host gateway is live: phase=$phase free_bytes=$AVAILABLE required_free_bytes=$LIVE_RUNTIME_MIN_FREE_BYTES"
      exit 0
    fi
    if [ "$phase" = "entry" ] \
        && [ "$ALLOW_LIVE_HOST_GATEWAY_PRUNE" = "1" ]; then
      echo "Docker storage online prune admitted under the exclusive operation lock: phase=$phase free_bytes=$AVAILABLE required_free_bytes=$LIVE_RUNTIME_MIN_FREE_BYTES"
      return 0
    fi
    echo "ERROR: exact host gateway runtime has only $AVAILABLE free bytes; $LIVE_RUNTIME_MIN_FREE_BYTES are required for an independent release build" >&2
    echo "ERROR: refusing Docker maintenance, daemon stop, or data-root reset while the host gateway is live" >&2
    exit 1
  fi
  if [ "$REQUIRE_ZERO_RUNTIME_RECONCILE" = "1" ]; then
    echo "Docker storage reconciliation admitted with the exact host gateway absent: phase=$phase"
  fi
}

require_exact_host_gateway_absent() {
  local phase="$1"

  inspect_exact_host_gateway_runtime "$phase"
  if [ "$HOST_GATEWAY_LIVE" -ne 0 ]; then
    echo "ERROR: exact host gateway state changed during absent-runtime Docker reconciliation: phase=$phase" >&2
    return 1
  fi
}

protect_exact_host_gateway_runtime "entry"

run_bounded_subprocess() {
  local timeout_seconds="$1"
  shift
  python3 - "$timeout_seconds" "$@" <<'PY'
import subprocess
import os
import signal
import sys
import time

timeout_seconds = int(sys.argv[1])
command = sys.argv[2:]
try:
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        return_code = process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            group_exists = False
        else:
            group_exists = True
        if group_exists:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            raise SystemExit(125)
        group_deadline = time.monotonic() + 5
        while True:
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                break
            if time.monotonic() >= group_deadline:
                raise SystemExit(125)
            time.sleep(0.01)
        raise SystemExit(124)
except OSError:
    raise SystemExit(127)
raise SystemExit(return_code)
PY
}

run_bounded_daemon_probe() {
  run_bounded_subprocess "$DAEMON_PROBE_TIMEOUT_SECONDS" "$@" \
    >/dev/null 2>&1
}

run_bounded_daemon_inventory() {
  run_bounded_subprocess "$DAEMON_PROBE_TIMEOUT_SECONDS" "$@"
}

run_bounded_daemon_control() {
  run_bounded_subprocess "$DAEMON_CONTROL_TIMEOUT_SECONDS" "$@"
}

run_bounded_daemon_command() {
  run_bounded_subprocess "$DAEMON_COMMAND_TIMEOUT_SECONDS" "$@"
}

reconcile_empty_docker_runtime() {
  local reconcile_result

  if ! reconcile_result="$(
    run_bounded_daemon_command sudo env PYTHONSAFEPATH=1 python3 \
      "$REPO_ROOT/validator_tee/host/docker_zero_runtime_reconciler_v2.py" \
      --docker-lock-file "$LEADPOET_DOCKER_OPERATION_LOCK_FILE" \
      --docker-admission-lock-file "$LEADPOET_DOCKER_OPERATION_ADMISSION_LOCK_FILE" \
      --docker-lock-owner-pid "$LEADPOET_DOCKER_OPERATION_LOCK_OWNER_PID" \
      --ready-attempts "$DAEMON_READY_ATTEMPTS" \
      --timeout-seconds 480
  )"; then
    echo "ERROR: guarded empty-runtime dockerd reconciliation failed" >&2
    return 1
  fi
  printf '%s\n' "$reconcile_result"
  if ! printf '%s' "$reconcile_result" | python3 -c '
import json
import re
import sys

document = json.load(sys.stdin)
if document.get("schema_version") != "leadpoet.docker_zero_runtime_reconcile.v1":
    raise SystemExit("invalid dockerd reconciliation schema")
if document.get("status") != "ready" or document.get("restart_performed") is not True:
    raise SystemExit("dockerd reconciliation did not report ready")
if document.get("docker_root") != "/var/lib/docker":
    raise SystemExit("dockerd reconciliation returned an invalid data-root")
for field in ("container_count", "containerd_container_count", "containerd_task_count", "moby_shim_count"):
    if type(document.get(field)) is not int or document[field] != 0:
        raise SystemExit("dockerd reconciliation returned a nonempty runtime")
if type(document.get("image_count")) is not int or document["image_count"] < 0:
    raise SystemExit("dockerd reconciliation returned malformed image count")
for field in ("root_device", "root_inode"):
    if type(document.get(field)) is not int or document[field] <= 0:
        raise SystemExit("dockerd reconciliation returned malformed root identity")
manifest = document.get("image_manifest_hash")
if not isinstance(manifest, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", manifest) is None:
    raise SystemExit("dockerd reconciliation returned malformed image identity")
'; then
    echo "ERROR: guarded empty-runtime dockerd reconciliation returned malformed evidence" >&2
    return 1
  fi
}

docker_daemons_ready() {
  run_bounded_daemon_probe docker info \
    && run_bounded_daemon_probe sudo ctr -n moby containers list -q
}

if ! docker_daemons_ready; then
  if [ "$HOST_GATEWAY_LIVE" -eq 1 ]; then
    echo "ERROR: Docker/containerd is unavailable while the exact host gateway is live; refusing daemon maintenance" >&2
    exit 1
  fi
  echo "Docker is unavailable; recovering builder daemons before inventory"
  run_bounded_daemon_control \
    sudo systemctl start containerd.service docker.service
  DAEMON_READY=0
  for _attempt in $(seq 1 "$DAEMON_READY_ATTEMPTS"); do
    if docker_daemons_ready; then
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

start_docker_daemons_and_wait() {
  local attempt
  if ! run_bounded_daemon_control \
      sudo systemctl start containerd.service docker.service; then
    echo "Docker/containerd start command failed; continuing bounded readiness recovery" >&2
  fi
  for attempt in $(seq 1 "$DAEMON_READY_ATTEMPTS"); do
    if docker_daemons_ready; then
      return 0
    fi
    if [ "$attempt" -lt "$DAEMON_READY_ATTEMPTS" ]; then
      sleep 1
    fi
  done
  return 1
}

DOCKER_RESET_STARTED=0
recover_docker_daemons_on_exit() {
  local exit_status=$?
  trap - EXIT
  if [ "$DOCKER_RESET_STARTED" -eq 1 ]; then
    echo "Recovering Docker/containerd readiness after failed data-root reset" >&2
    if start_docker_daemons_and_wait; then
      echo "Docker/containerd readiness recovered after failed data-root reset" >&2
    else
      echo "ERROR: Docker/containerd recovery remained unavailable after failed data-root reset" >&2
    fi
  fi
  exit "$exit_status"
}

run_prune_with_retry() {
  local label="$1"
  shift
  local attempt
  for attempt in $(seq 1 "$PRUNE_ATTEMPTS"); do
    if run_bounded_daemon_command "$@" >/dev/null; then
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

bounded_moby_shim_count() {
  local count status

  if count="$(
    run_bounded_daemon_inventory \
      pgrep -fc '^/usr/bin/containerd-shim-runc-v2 -namespace moby ' \
      2>/dev/null
  )"; then
    status=0
  else
    status=$?
  fi
  if [ "$status" -ne 0 ] && [ "$status" -ne 1 ]; then
    echo "ERROR: moby shim inventory is unreadable" >&2
    return 1
  fi
  if ! is_bounded_unsigned_shell_integer "$count"; then
    echo "ERROR: moby shim inventory is malformed" >&2
    return 1
  fi
  if [ "$status" -eq 1 ] && [ "$count" -ne 0 ]; then
    echo "ERROR: moby shim inventory status is inconsistent" >&2
    return 1
  fi
  # procps pgrep reports no matches as status 1, while the strict restart
  # rehearsal adapter reports the same successful count probe as status 0.
  # Both zero-count outcomes are readable.  A nonzero count remains valid only
  # with pgrep's match status 0; status 2+ is unreadable above.
  printf '%s\n' "$count"
}

inventory_empty_online_runtime() {
  local phase="$1"
  local docker_container_ids containerd_container_ids containerd_task_ids
  local docker_container_count containerd_container_count containerd_task_count
  local moby_shim_count

  if ! docker_container_ids="$(
    run_bounded_daemon_inventory docker ps -aq 2>/dev/null
  )"; then
    echo "ERROR: Docker container inventory is unreadable during online prune: phase=$phase" >&2
    return 1
  fi
  if ! containerd_container_ids="$(
    run_bounded_daemon_inventory sudo ctr -n moby containers list -q 2>/dev/null
  )"; then
    echo "ERROR: containerd container inventory is unreadable during online prune: phase=$phase" >&2
    return 1
  fi
  if ! containerd_task_ids="$(
    run_bounded_daemon_inventory sudo ctr -n moby tasks list -q 2>/dev/null
  )"; then
    echo "ERROR: containerd task inventory is unreadable during online prune: phase=$phase" >&2
    return 1
  fi
  docker_container_count="$(
    printf '%s\n' "$docker_container_ids" \
      | awk 'NF { count += 1 } END { print count + 0 }'
  )"
  containerd_container_count="$(
    printf '%s\n' "$containerd_container_ids" \
      | awk 'NF { count += 1 } END { print count + 0 }'
  )"
  containerd_task_count="$(
    printf '%s\n' "$containerd_task_ids" \
      | awk 'NF { count += 1 } END { print count + 0 }'
  )"
  moby_shim_count="$(bounded_moby_shim_count)"
  echo "Docker online-prune runtime state: phase=$phase containers=$docker_container_count containerd_containers=$containerd_container_count containerd_tasks=$containerd_task_count moby_shims=$moby_shim_count"
  if [ "$docker_container_count" -ne 0 ] \
      || [ "$containerd_container_count" -ne 0 ] \
      || [ "$containerd_task_count" -ne 0 ] \
      || [ "$moby_shim_count" -ne 0 ]; then
    echo "ERROR: refusing online Docker prune unless the exact container runtime is empty: phase=$phase" >&2
    return 1
  fi
}

empty_runtime_metadata_is_clear() {
  local image_count

  if ! image_count="$(
    run_bounded_daemon_inventory docker image ls -aq \
      | sed '/^$/d' | sort -u | wc -l | tr -d '[:space:]'
  )"; then
    echo "ERROR: Docker image inventory is unreadable after reconciliation" >&2
    return 1
  fi
  if [ "$image_count" -ne 0 ]; then
    echo "Docker images remain after guarded reconciliation: images=$image_count" >&2
    return 1
  fi
  # Missing directories are empty; access and traversal errors are not.
  # This function runs in an if condition, so do not rely on shell errexit.
  if ! sudo python3 - "$DOCKER_ROOT" <<'PY'
import os
import sys

root = sys.argv[1]
try:
    for relative in ("image/overlay2/layerdb/sha256", "image/overlay2/layerdb/mounts", "overlay2"):
        try:
            with os.scandir(os.path.join(root, relative)) as entries:
                if any(relative != "overlay2" or entry.name != "l" for entry in entries):
                    raise RuntimeError("Docker metadata remains after reconciliation")
        except FileNotFoundError:
            continue
except (OSError, RuntimeError) as exc:
    print("ERROR: Docker metadata is not proven empty: " + str(exc), file=sys.stderr)
    sys.exit(1)
PY
  then
    echo "ERROR: Docker metadata is not proven empty after guarded reconciliation" >&2
    return 1
  fi
  return 0
}

online_image_ids() {
  local raw_image_ids

  if ! raw_image_ids="$(
    run_bounded_daemon_inventory docker image ls -aq --no-trunc 2>/dev/null
  )"; then
    echo "ERROR: Docker image inventory is unreadable during online prune" >&2
    return 1
  fi
  printf '%s\n' "$raw_image_ids" | python3 -c '
import re
import sys

rows = [row.strip() for row in sys.stdin if row.strip()]
if len(rows) != len(set(rows)):
    raise SystemExit("Docker image inventory contains duplicate identities")
if any(re.fullmatch(r"sha256:[0-9a-f]{64}", row) is None for row in rows):
    raise SystemExit("Docker image inventory contains a malformed identity")
print("\n".join(sorted(rows)))
'
}

online_docker_root() {
  local observed_root

  if ! observed_root="$(
    run_bounded_daemon_inventory \
      docker info --format '{{.DockerRootDir}}' 2>/dev/null
  )"; then
    echo "ERROR: Docker data-root is unreadable during online prune" >&2
    return 1
  fi
  if [ "$observed_root" != "/var/lib/docker" ]; then
    echo "ERROR: refusing online prune for unexpected Docker data-root: $observed_root" >&2
    return 1
  fi
  printf '%s\n' "$observed_root"
}

require_same_online_gateway() {
  local phase="$1"

  inspect_exact_host_gateway_runtime "$phase"
  if [ "$HOST_GATEWAY_LIVE" -ne 1 ] \
      || [ "$HOST_GATEWAY_PID" -ne "$ONLINE_GATEWAY_PID" ] \
      || [ "$HOST_GATEWAY_START_TIME_TICKS" -ne "$ONLINE_GATEWAY_START_TIME_TICKS" ]; then
    echo "ERROR: exact host gateway identity changed during online Docker reclaim: phase=$phase" >&2
    return 1
  fi
}

require_same_online_docker_root() {
  local phase="$1"
  local observed_root

  if ! observed_root="$(online_docker_root)"; then
    return 1
  fi
  if [ "$observed_root" != "$ONLINE_DOCKER_ROOT" ]; then
    echo "ERROR: Docker data-root changed during online reclaim: phase=$phase" >&2
    return 1
  fi
}

require_same_online_images() {
  local phase="$1"
  local observed_image_ids

  if ! observed_image_ids="$(online_image_ids)"; then
    return 1
  fi
  if [ "$observed_image_ids" != "$ONLINE_IMAGE_IDS" ]; then
    echo "ERROR: Docker image identity changed during online reclaim: phase=$phase" >&2
    return 1
  fi
}

online_overlay_metadata_layout() {
  local directory
  local present=0
  local absent=0
  local status

  for directory in \
      "$ONLINE_DOCKER_ROOT/image/overlay2/layerdb/sha256" \
      "$ONLINE_DOCKER_ROOT/image/overlay2/layerdb/mounts" \
      "$ONLINE_DOCKER_ROOT/overlay2"; do
    if sudo test -d "$directory"; then
      present=$((present + 1))
    else
      status=$?
      if [ "$status" -ne 1 ]; then
        echo "ERROR: Docker overlay metadata layout is unreadable: $directory" >&2
        return 1
      fi
      absent=$((absent + 1))
    fi
  done
  if [ "$present" -eq 3 ]; then
    printf '%s\n' "initialized"
    return 0
  fi
  if [ "$absent" -eq 3 ] && [ -z "$ONLINE_IMAGE_IDS" ]; then
    printf '%s\n' "absent"
    return 0
  fi
  echo "ERROR: Docker overlay metadata layout is partial or inconsistent with image inventory" >&2
  return 1
}

online_stale_audit_manifest() {
  python3 -c '
import json
import re
import sys

document = json.load(sys.stdin)
count_fields = (
    "active_container_count",
    "active_image_count",
    "active_layer_count",
    "active_mount_count",
    "active_overlay_dir_count",
    "mounted_overlay_count",
    "stale_layer_record_count",
    "stale_mount_record_count",
    "stale_overlay_dir_count",
    "stale_overlay_link_count",
)
zero_fields = (
    "active_container_count",
    "active_mount_count",
    "mounted_overlay_count",
    "stale_layer_record_count",
    "stale_mount_record_count",
    "stale_overlay_dir_count",
    "stale_overlay_link_count",
)
if document.get("schema_version") != "leadpoet.docker_stale_mount_audit.v3":
    raise SystemExit("invalid online Docker audit schema")
if document.get("status") != "ready" or document.get("docker_root") != "/var/lib/docker":
    raise SystemExit("invalid online Docker audit identity")
if any(type(document.get(field)) is not int or document[field] < 0 for field in count_fields):
    raise SystemExit("online Docker audit returned malformed counts")
if any(document[field] != 0 for field in zero_fields):
    raise SystemExit("online Docker audit did not observe clean empty-runtime state")
manifest = document.get("active_manifest_hash")
if not isinstance(manifest, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", manifest) is None:
    raise SystemExit("online Docker audit returned malformed active identity")
print(manifest)
'
}

online_fully_empty_stale_audit_manifest() {
  python3 -c '
import json
import re
import sys

document = json.load(sys.stdin)
count_fields = (
    "active_container_count",
    "active_image_count",
    "active_layer_count",
    "active_mount_count",
    "active_overlay_dir_count",
    "mounted_overlay_count",
    "stale_layer_record_count",
    "stale_mount_record_count",
    "stale_overlay_dir_count",
    "stale_overlay_link_count",
)
if document.get("schema_version") != "leadpoet.docker_stale_mount_audit.v3":
    raise SystemExit("invalid empty Docker audit schema")
if document.get("status") != "ready" or document.get("docker_root") != "/var/lib/docker":
    raise SystemExit("invalid empty Docker audit identity")
if any(type(document.get(field)) is not int or document[field] != 0 for field in count_fields):
    raise SystemExit("empty Docker audit returned nonzero or malformed counts")
manifest = document.get("active_manifest_hash")
if not isinstance(manifest, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", manifest) is None:
    raise SystemExit("empty Docker audit returned malformed active identity")
print(manifest)
'
}

online_stale_audit_manifest_allowing_stale() {
  python3 -c '
import json
import re
import sys

document = json.load(sys.stdin)
count_fields = (
    "active_container_count",
    "active_image_count",
    "active_layer_count",
    "active_mount_count",
    "active_overlay_dir_count",
    "mounted_overlay_count",
    "stale_layer_record_count",
    "stale_mount_record_count",
    "stale_overlay_dir_count",
    "stale_overlay_link_count",
)
if document.get("schema_version") != "leadpoet.docker_stale_mount_audit.v3":
    raise SystemExit("invalid pre-reclaim Docker audit schema")
if document.get("status") != "ready" or document.get("docker_root") != "/var/lib/docker":
    raise SystemExit("invalid pre-reclaim Docker audit identity")
if any(type(document.get(field)) is not int or document[field] < 0 for field in count_fields):
    raise SystemExit("pre-reclaim Docker audit returned malformed counts")
if document["active_container_count"] != 0 or document["active_mount_count"] != 0:
    raise SystemExit("pre-reclaim Docker audit observed a nonempty runtime")
manifest = document.get("active_manifest_hash")
if not isinstance(manifest, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", manifest) is None:
    raise SystemExit("pre-reclaim Docker audit returned malformed active identity")
print(manifest)
'
}

online_stale_reclaim_performed() {
  python3 -c '
import json
import sys

raw = sys.stdin.read(65537)
if len(raw) > 65536:
    raise SystemExit("online Docker reclaim evidence exceeds its size bound")
lines = raw.splitlines()
if len(lines) != 2:
    raise SystemExit("online Docker reclaim evidence must contain exactly two documents")
try:
    before, result = (json.loads(line) for line in lines)
except (TypeError, ValueError) as exc:
    raise SystemExit("online Docker reclaim evidence is not valid JSON") from exc
audit_counts = (
    "active_container_count",
    "active_image_count",
    "active_layer_count",
    "active_mount_count",
    "active_overlay_dir_count",
    "mounted_overlay_count",
    "stale_layer_record_count",
    "stale_mount_record_count",
    "stale_overlay_dir_count",
    "stale_overlay_link_count",
)
result_counts = (
    "active_container_count",
    "active_image_count",
    "active_layer_count",
    "active_mount_count",
    "mounted_overlay_count",
    "reclaimed_layer_record_count",
    "reclaimed_mount_count",
    "reclaimed_mount_record_count",
    "reclaimed_overlay_dir_count",
    "reclaimed_overlay_link_count",
)
if before.get("schema_version") != "leadpoet.docker_stale_mount_audit.v3":
    raise SystemExit("invalid pre-reclaim Docker audit schema")
if result.get("schema_version") != "leadpoet.docker_stale_mount_reclaim.v3":
    raise SystemExit("invalid online Docker reclaim result schema")
if any(document.get("status") != "ready" for document in (before, result)):
    raise SystemExit("online Docker reclaim evidence did not report ready")
if any(document.get("docker_root") != "/var/lib/docker" for document in (before, result)):
    raise SystemExit("online Docker reclaim evidence returned an invalid data-root")
if any(type(before.get(field)) is not int or before[field] < 0 for field in audit_counts):
    raise SystemExit("pre-reclaim Docker audit returned malformed counts")
if any(type(result.get(field)) is not int or result[field] < 0 for field in result_counts):
    raise SystemExit("online Docker reclaim returned malformed counts")
if before["active_container_count"] != 0 or before["active_mount_count"] != 0:
    raise SystemExit("pre-reclaim Docker audit observed a nonempty runtime")
active_pairs = (
    ("active_container_count", "active_container_count"),
    ("active_image_count", "active_image_count"),
    ("active_layer_count", "active_layer_count"),
    ("active_mount_count", "active_mount_count"),
    ("mounted_overlay_count", "mounted_overlay_count"),
)
reclaimed_pairs = (
    ("stale_layer_record_count", "reclaimed_layer_record_count"),
    ("mounted_overlay_count", "reclaimed_mount_count"),
    ("stale_mount_record_count", "reclaimed_mount_record_count"),
    ("stale_overlay_dir_count", "reclaimed_overlay_dir_count"),
    ("stale_overlay_link_count", "reclaimed_overlay_link_count"),
)
if any(before[audit_field] != result[result_field] for audit_field, result_field in active_pairs):
    raise SystemExit("active Docker counts changed during guarded stale reclaim")
if any(before[audit_field] != result[result_field] for audit_field, result_field in reclaimed_pairs):
    raise SystemExit("Docker stale-state mutation counts do not match the pre-reclaim audit")
stale_fields = tuple(audit_field for audit_field, _ in reclaimed_pairs)
print(int(any(before[field] for field in stale_fields)))
'
}

online_stale_audit_identity_preserved() {
  python3 -c '
import json
import re
import sys

raw = sys.stdin.read(65537)
if len(raw) > 65536:
    raise SystemExit("online Docker audit evidence exceeds its size bound")
lines = raw.splitlines()
if len(lines) != 2:
    raise SystemExit("online Docker audit evidence must contain exactly two documents")
try:
    before, after = (json.loads(line) for line in lines)
except (TypeError, ValueError) as exc:
    raise SystemExit("online Docker audit evidence is not valid JSON") from exc
count_fields = (
    "active_container_count",
    "active_image_count",
    "active_layer_count",
    "active_mount_count",
    "active_overlay_dir_count",
    "mounted_overlay_count",
    "stale_layer_record_count",
    "stale_mount_record_count",
    "stale_overlay_dir_count",
    "stale_overlay_link_count",
)
active_fields = (
    "active_container_count",
    "active_image_count",
    "active_layer_count",
    "active_mount_count",
    "active_overlay_dir_count",
)
zero_after_fields = (
    "active_container_count",
    "active_mount_count",
    "mounted_overlay_count",
    "stale_layer_record_count",
    "stale_mount_record_count",
    "stale_overlay_dir_count",
    "stale_overlay_link_count",
)
for document in (before, after):
    if document.get("schema_version") != "leadpoet.docker_stale_mount_audit.v3":
        raise SystemExit("invalid online Docker audit schema")
    if document.get("status") != "ready" or document.get("docker_root") != "/var/lib/docker":
        raise SystemExit("invalid online Docker audit identity")
    if any(type(document.get(field)) is not int or document[field] < 0 for field in count_fields):
        raise SystemExit("online Docker audit returned malformed counts")
    manifest = document.get("active_manifest_hash")
    if not isinstance(manifest, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", manifest) is None:
        raise SystemExit("online Docker audit returned malformed active identity")
if any(after[field] != 0 for field in zero_after_fields):
    raise SystemExit("post-reclaim Docker audit did not observe clean empty-runtime state")
if any(before[field] != after[field] for field in active_fields):
    raise SystemExit("active Docker counts changed during guarded stale reclaim")
if before["active_manifest_hash"] != after["active_manifest_hash"]:
    raise SystemExit("active Docker manifest changed during guarded stale reclaim")
'
}

if [ "$HOST_GATEWAY_LIVE" -eq 1 ]; then
  ONLINE_GATEWAY_PID="$HOST_GATEWAY_PID"
  ONLINE_GATEWAY_START_TIME_TICKS="$HOST_GATEWAY_START_TIME_TICKS"
  inventory_empty_online_runtime "pre-prune"
  ONLINE_DOCKER_ROOT="$(online_docker_root)"
  ONLINE_IMAGE_IDS="$(online_image_ids)"
  run_prune_with_retry builder docker builder prune --all --force
  inventory_empty_online_runtime "post-builder-prune"
  require_same_online_gateway "post-builder-prune"
  require_same_online_docker_root "post-builder-prune"
  require_same_online_images "post-builder-prune"
  AVAILABLE="$(available_bytes)"
  ONLINE_RAW_RECLAIM_PERFORMED=0
  if [ "$AVAILABLE" -lt "$LIVE_RUNTIME_MIN_FREE_BYTES" ] \
      || [ "$REQUIRE_ZERO_RUNTIME_RECONCILE" = "1" ]; then
    inventory_empty_online_runtime "pre-stale-reclaim"
    require_same_online_gateway "pre-stale-reclaim"
    require_same_online_docker_root "pre-stale-reclaim"
    require_same_online_images "pre-stale-reclaim"
    ONLINE_PRE_RECLAIM_METADATA_LAYOUT="$(online_overlay_metadata_layout)"
    if ! ONLINE_PRE_RECLAIM_AUDIT="$(
      run_bounded_daemon_command sudo env PYTHONSAFEPATH=1 python3 \
        "$REPO_ROOT/validator_tee/host/docker_stale_mount_reclaimer_v2.py" \
        --audit-only
    )"; then
      echo "ERROR: Docker audit failed before guarded stale reclaim" >&2
      exit 1
    fi
    printf '%s\n' "$ONLINE_PRE_RECLAIM_AUDIT"
    if ! ONLINE_PRE_RECLAIM_MANIFEST="$(
      printf '%s' "$ONLINE_PRE_RECLAIM_AUDIT" \
        | online_stale_audit_manifest_allowing_stale
    )"; then
      echo "ERROR: Docker audit was invalid before guarded stale reclaim" >&2
      exit 1
    fi
    if ! ONLINE_RECLAIM_RESULT="$(
      run_bounded_daemon_command sudo env PYTHONSAFEPATH=1 python3 \
        "$REPO_ROOT/validator_tee/host/docker_stale_mount_reclaimer_v2.py"
    )"; then
      echo "ERROR: validated stale Docker overlay reclaim failed while the exact host gateway is live" >&2
      exit 1
    fi
    printf '%s\n' "$ONLINE_RECLAIM_RESULT"
    if ! ONLINE_RAW_RECLAIM_PERFORMED="$(
      printf '%s\n%s\n' "$ONLINE_PRE_RECLAIM_AUDIT" "$ONLINE_RECLAIM_RESULT" \
        | online_stale_reclaim_performed
    )"; then
      echo "ERROR: validated stale Docker overlay reclaim returned malformed evidence" >&2
      exit 1
    fi
    if ! ONLINE_PRE_RECONCILE_AUDIT="$(
      run_bounded_daemon_command sudo env PYTHONSAFEPATH=1 python3 \
        "$REPO_ROOT/validator_tee/host/docker_stale_mount_reclaimer_v2.py" \
        --audit-only
    )"; then
      echo "ERROR: post-reclaim Docker audit failed before daemon reconciliation" >&2
      exit 1
    fi
    printf '%s\n' "$ONLINE_PRE_RECONCILE_AUDIT"
    ONLINE_PRE_RECONCILE_METADATA_LAYOUT="$(online_overlay_metadata_layout)"
    if [ "$ONLINE_PRE_RECONCILE_METADATA_LAYOUT" != "$ONLINE_PRE_RECLAIM_METADATA_LAYOUT" ]; then
      echo "ERROR: Docker overlay metadata layout changed during guarded stale reclaim" >&2
      exit 1
    fi
    if [ "$ONLINE_PRE_RECONCILE_METADATA_LAYOUT" = "absent" ]; then
      ONLINE_PRE_RECONCILE_MANIFEST="$(
        printf '%s' "$ONLINE_PRE_RECONCILE_AUDIT" \
          | online_fully_empty_stale_audit_manifest
      )" || {
        echo "ERROR: post-reclaim empty Docker audit was invalid before daemon reconciliation" >&2
        exit 1
      }
    elif ! ONLINE_PRE_RECONCILE_MANIFEST="$(
      printf '%s' "$ONLINE_PRE_RECONCILE_AUDIT" | online_stale_audit_manifest
    )"; then
      echo "ERROR: post-reclaim Docker audit was invalid before daemon reconciliation" >&2
      exit 1
    fi
    if [ "$ONLINE_PRE_RECONCILE_MANIFEST" != "$ONLINE_PRE_RECLAIM_MANIFEST" ]; then
      echo "ERROR: active Docker image/layer identity changed during guarded stale reclaim" >&2
      exit 1
    fi
    if ! printf '%s\n%s\n' "$ONLINE_PRE_RECLAIM_AUDIT" "$ONLINE_PRE_RECONCILE_AUDIT" \
        | online_stale_audit_identity_preserved; then
      echo "ERROR: active Docker inventory changed during guarded stale reclaim" >&2
      exit 1
    fi
  fi
  if [ "$ONLINE_RAW_RECLAIM_PERFORMED" -eq 1 ] \
      || [ "$REQUIRE_ZERO_RUNTIME_RECONCILE" = "1" ]; then
      inventory_empty_online_runtime "pre-daemon-reconcile"
      require_same_online_gateway "pre-daemon-reconcile"
      require_same_online_docker_root "pre-daemon-reconcile"
      require_same_online_images "pre-daemon-reconcile"
      PYTHONPATH="$REPO_ROOT" python3 \
        -m validator_tee.host.docker_operation_guard_v2 \
        --wait \
        --timeout-seconds 1800 \
        --interval-seconds 3 \
        --proc-root "$PROC_ROOT"
      echo "Reconciling dockerd metadata under the guarded empty runtime"
      reconcile_empty_docker_runtime
      inventory_empty_online_runtime "post-daemon-reconcile"
      require_same_online_gateway "post-daemon-reconcile"
      require_same_online_docker_root "post-daemon-reconcile"
      require_same_online_images "post-daemon-reconcile"
      if ! ONLINE_POST_RECONCILE_AUDIT="$(
        run_bounded_daemon_command sudo env PYTHONSAFEPATH=1 python3 \
          "$REPO_ROOT/validator_tee/host/docker_stale_mount_reclaimer_v2.py" \
          --audit-only
      )"; then
        echo "ERROR: Docker audit failed after daemon reconciliation" >&2
        exit 1
      fi
      printf '%s\n' "$ONLINE_POST_RECONCILE_AUDIT"
      ONLINE_POST_RECONCILE_METADATA_LAYOUT="$(online_overlay_metadata_layout)"
      if [ "$ONLINE_PRE_RECONCILE_METADATA_LAYOUT" = "initialized" ] \
          && [ "$ONLINE_POST_RECONCILE_METADATA_LAYOUT" != "initialized" ]; then
        echo "ERROR: initialized Docker overlay metadata disappeared during daemon reconciliation" >&2
        exit 1
      fi
      if [ "$ONLINE_PRE_RECONCILE_METADATA_LAYOUT" = "absent" ]; then
        ONLINE_POST_RECONCILE_MANIFEST="$(
          printf '%s' "$ONLINE_POST_RECONCILE_AUDIT" \
            | online_fully_empty_stale_audit_manifest
        )" || {
          echo "ERROR: Docker empty-state audit was invalid after daemon reconciliation" >&2
          exit 1
        }
      elif ! ONLINE_POST_RECONCILE_MANIFEST="$(
        printf '%s' "$ONLINE_POST_RECONCILE_AUDIT" | online_stale_audit_manifest
      )"; then
        echo "ERROR: Docker audit was invalid after daemon reconciliation" >&2
        exit 1
      fi
      if [ "$ONLINE_PRE_RECONCILE_METADATA_LAYOUT" = "initialized" ]; then
        if [ "$ONLINE_POST_RECONCILE_MANIFEST" != "$ONLINE_PRE_RECONCILE_MANIFEST" ]; then
          echo "ERROR: active Docker image/layer identity changed during daemon reconciliation" >&2
          exit 1
        fi
        if ! printf '%s\n%s\n' "$ONLINE_PRE_RECLAIM_AUDIT" "$ONLINE_POST_RECONCILE_AUDIT" \
            | online_stale_audit_identity_preserved; then
          echo "ERROR: active Docker inventory changed during daemon reconciliation" >&2
          exit 1
        fi
      fi
      run_prune_with_retry builder docker builder prune --all --force
      inventory_empty_online_runtime "post-reconcile-builder-prune"
      require_same_online_gateway "post-reconcile-builder-prune"
      require_same_online_docker_root "post-reconcile-builder-prune"
      require_same_online_images "post-reconcile-builder-prune"
      if ! ONLINE_POST_PRUNE_AUDIT="$(
        run_bounded_daemon_command sudo env PYTHONSAFEPATH=1 python3 \
          "$REPO_ROOT/validator_tee/host/docker_stale_mount_reclaimer_v2.py" \
          --audit-only
      )"; then
        echo "ERROR: Docker audit failed after reconciled builder prune" >&2
        exit 1
      fi
      printf '%s\n' "$ONLINE_POST_PRUNE_AUDIT"
      if [ "$ONLINE_PRE_RECONCILE_METADATA_LAYOUT" = "absent" ]; then
        ONLINE_POST_PRUNE_MANIFEST="$(
          printf '%s' "$ONLINE_POST_PRUNE_AUDIT" \
            | online_fully_empty_stale_audit_manifest
        )" || {
          echo "ERROR: Docker empty-state audit was invalid after reconciled builder prune" >&2
          exit 1
        }
      elif ! ONLINE_POST_PRUNE_MANIFEST="$(
        printf '%s' "$ONLINE_POST_PRUNE_AUDIT" | online_stale_audit_manifest
      )"; then
        echo "ERROR: Docker audit was invalid after reconciled builder prune" >&2
        exit 1
      fi
      if [ "$ONLINE_POST_PRUNE_MANIFEST" != "$ONLINE_POST_RECONCILE_MANIFEST" ]; then
        echo "ERROR: active Docker image/layer identity changed during reconciled builder prune" >&2
        exit 1
      fi
      if ! printf '%s\n%s\n' "$ONLINE_POST_RECONCILE_AUDIT" "$ONLINE_POST_PRUNE_AUDIT" \
          | online_stale_audit_identity_preserved; then
        echo "ERROR: active Docker inventory changed during reconciled builder prune" >&2
        exit 1
      fi
  fi
  inventory_empty_online_runtime "post-reclaim"
  require_same_online_gateway "post-reclaim"
  require_same_online_docker_root "post-reclaim"
  require_same_online_images "post-reclaim"
  AVAILABLE="$(available_bytes)"
  if [ "$AVAILABLE" -lt "$LIVE_RUNTIME_MIN_FREE_BYTES" ]; then
    echo "ERROR: bounded online Docker reclaim left only $AVAILABLE free bytes; $LIVE_RUNTIME_MIN_FREE_BYTES are required for an independent release build" >&2
    echo "ERROR: refusing daemon stop or data-root reset while the exact host gateway is live" >&2
    exit 1
  fi
  echo "Docker storage ready after bounded online reclaim: free_bytes=$AVAILABLE required_free_bytes=$LIVE_RUNTIME_MIN_FREE_BYTES runtime_mode=host-gateway-live"
  exit 0
fi

run_prune_with_retry image docker image prune --all --force
run_prune_with_retry builder docker builder prune --all --force
run_prune_with_retry system docker system prune --all --force

# The release builder invokes this helper while the deployed validator
# containers are intentionally running. Wait for containerd teardown only
# after Docker reports an empty runtime, which is the restart/data-root
# recovery path. Healthy live containers are inventoried and preserved below.
if ! INITIAL_CONTAINER_IDS="$(
  run_bounded_daemon_inventory docker ps -aq 2>/dev/null
)"; then
  echo "ERROR: Docker container inventory is unreadable after prune" >&2
  exit 1
fi
INITIAL_CONTAINER_COUNT="$(
  printf '%s\n' "$INITIAL_CONTAINER_IDS" \
    | awk 'NF { count += 1 } END { print count + 0 }'
)"
if [ "$INITIAL_CONTAINER_COUNT" -ne 0 ]; then
  echo "Reclaiming unreachable Docker overlay state without disturbing live containers"
  # This runs before release dependencies are installed. Execute the
  # stdlib-only helper directly so validator_tee/__init__.py cannot import
  # wallet/runtime packages that are intentionally unavailable at this stage.
  RECLAIM_RESULT="$(
    sudo env PYTHONSAFEPATH=1 python3 \
      "$REPO_ROOT/validator_tee/host/docker_stale_mount_reclaimer_v2.py"
  )"
  printf '%s\n' "$RECLAIM_RESULT"
  RAW_RECLAIM_PERFORMED="$(
    printf '%s' "$RECLAIM_RESULT" | python3 -c '
import json
import sys

document = json.load(sys.stdin)
fields = (
    "reclaimed_layer_record_count",
    "reclaimed_mount_count",
    "reclaimed_mount_record_count",
    "reclaimed_overlay_dir_count",
    "reclaimed_overlay_link_count",
)
if document.get("status") != "ready" or any(
    not isinstance(document.get(field), int) or document[field] < 0
    for field in fields
):
    raise SystemExit("invalid Docker stale-state reclaim result")
print(int(any(document[field] for field in fields)))
'
  )"
  if ! LIVE_RESTORE_ENABLED="$(
    run_bounded_daemon_inventory \
      docker info --format '{{json .LiveRestoreEnabled}}' 2>/dev/null
  )"; then
    echo "ERROR: Docker live-restore status is unreadable" >&2
    exit 1
  fi
  if [ "$LIVE_RESTORE_ENABLED" != "true" ] \
      && [ "$LIVE_RESTORE_ENABLED" != "false" ]; then
    echo "ERROR: Docker live-restore status is malformed" >&2
    exit 1
  fi
  if [ "$RAW_RECLAIM_PERFORMED" -eq 1 ] \
      || [ "$LIVE_RESTORE_ENABLED" != "true" ]; then
    echo "Reconciling dockerd metadata while preserving the live runtime"
    sudo env PYTHONSAFEPATH=1 python3 \
      "$REPO_ROOT/validator_tee/host/docker_live_restore_reconciler_v2.py"
  fi
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
    if ! CONTAINER_IDS="$(
      run_bounded_daemon_inventory docker ps -aq 2>/dev/null
    )"; then
      TEARDOWN_PROBE_FAILED=1
    elif ! CONTAINERD_TASKS="$(
      run_bounded_daemon_inventory \
        sudo ctr -n moby tasks list 2>/dev/null
    )"; then
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
CONTAINER_COUNT="$(run_bounded_daemon_inventory docker ps -aq | wc -l | tr -d '[:space:]')"
IMAGE_COUNT="$(run_bounded_daemon_inventory docker image ls -aq | sort -u | sed '/^$/d' | wc -l | tr -d '[:space:]')"
VOLUME_COUNT="$(run_bounded_daemon_inventory docker volume ls -q | wc -l | tr -d '[:space:]')"
DOCKER_ROOT="$(run_bounded_daemon_inventory docker info --format '{{.DockerRootDir}}')"
CONTAINERD_ROOT="${VALIDATOR_CONTAINERD_ROOT:-/var/lib/containerd}"
DOCKER_ROOT_BYTES="$(sudo du -sx -B1 "$DOCKER_ROOT" | awk '{print $1}')"
CONTAINERD_CONTAINER_COUNT="$(
  run_bounded_daemon_inventory sudo ctr -n moby containers list -q \
    | sed '/^$/d' | wc -l | tr -d '[:space:]'
)"
CONTAINERD_TASK_COUNT="$(
  run_bounded_daemon_inventory sudo ctr -n moby tasks list -q \
    | sed '/^$/d' | wc -l | tr -d '[:space:]'
)"
CONTAINERD_RUNNING_TASK_COUNT="$(
  run_bounded_daemon_inventory sudo ctr -n moby tasks list \
    | awk 'NR > 1 && $3 == "RUNNING" { count += 1 } END { print count + 0 }'
)"
NON_MOBY_NAMESPACE_COUNT="$(
  run_bounded_daemon_inventory sudo ctr namespaces list -q \
    | sed '/^$/d; /^moby$/d' \
    | wc -l \
    | tr -d '[:space:]'
)"
MOBY_SHIM_COUNT="$(bounded_moby_shim_count)"
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
  if [ "$REQUIRE_ZERO_RUNTIME_RECONCILE" = "1" ]; then
    if [ "$RUNTIME_MODE" != "empty" ]; then
      echo "ERROR: required Docker reconciliation reached a nonempty runtime" >&2
      exit 1
    fi
    inventory_empty_online_runtime "pre-absent-daemon-reconcile"
    require_exact_host_gateway_absent "pre-absent-daemon-reconcile"
    PYTHONPATH="$REPO_ROOT" python3 \
      -m validator_tee.host.docker_operation_guard_v2 \
      --wait \
      --timeout-seconds 1800 \
      --interval-seconds 3 \
      --proc-root "$PROC_ROOT"
    require_exact_host_gateway_absent "pre-absent-daemon-reconcile-apply"
    echo "Reconciling dockerd metadata under the guarded absent empty runtime"
    reconcile_empty_docker_runtime
    inventory_empty_online_runtime "post-absent-daemon-reconcile"
    require_exact_host_gateway_absent "post-absent-daemon-reconcile"
    AVAILABLE="$(available_bytes)"
    if [ "$AVAILABLE" -lt "$REQUIRED_FREE_BYTES" ]; then
      echo "ERROR: required absent-runtime Docker reconciliation left only $AVAILABLE free bytes; $REQUIRED_FREE_BYTES are required" >&2
      exit 1
    fi
  fi
  echo "Docker storage ready: free_bytes=$AVAILABLE required_free_bytes=$REQUIRED_FREE_BYTES runtime_mode=$RUNTIME_MODE"
  exit 0
fi

if [ "$CONTAINER_COUNT" -ne 0 ]; then
  echo "ERROR: live validator runtime has only $AVAILABLE free bytes; $REQUIRED_FREE_BYTES are required for an independent recovery build" >&2
  echo "ERROR: refusing to stop containers or reset Docker storage from the release builder" >&2
  exit 1
fi

# A named volume is independent state.  When the runtime is otherwise empty
# and capacity is already sufficient, first refresh only dockerd metadata.  A
# successful refresh avoids a destructive data-root reset. Verify the refreshed
# state in this process, under the same exclusive lock. A persistent orphan
# still follows the existing fail-closed reset checks.
if [ "$AVAILABLE" -ge "$REQUIRED_FREE_BYTES" ] \
    && [ "$ORPHANED_DOCKER_STATE" -eq 1 ] \
    && [ "$VOLUME_COUNT" -ne 0 ] \
    && [ "$IMAGE_COUNT" -eq 0 ] \
    && [ "$CONTAINERD_CONTAINER_COUNT" -eq 0 ] \
    && [ "$CONTAINERD_TASK_COUNT" -eq 0 ] \
    && [ "$CONTAINERD_RUNNING_TASK_COUNT" -eq 0 ] \
    && [ "$MOBY_SHIM_COUNT" -eq 0 ] \
    && [ "$NON_MOBY_NAMESPACE_COUNT" -eq 0 ]; then
  inventory_empty_online_runtime "pre-orphan-metadata-reconcile"
  require_exact_host_gateway_absent "pre-orphan-metadata-reconcile"
  PYTHONPATH="$REPO_ROOT" python3 \
    -m validator_tee.host.docker_operation_guard_v2 \
    --wait \
    --timeout-seconds 1800 \
    --interval-seconds 3 \
    --proc-root "$PROC_ROOT"
  require_exact_host_gateway_absent "pre-orphan-metadata-reconcile-apply"
  echo "Reconciling orphaned dockerd metadata before considering a data-root reset"
  reconcile_empty_docker_runtime
  inventory_empty_online_runtime "post-orphan-metadata-reconcile"
  require_exact_host_gateway_absent "post-orphan-metadata-reconcile"
  if empty_runtime_metadata_is_clear; then
    AVAILABLE="$(available_bytes)"
    if [ "$AVAILABLE" -ge "$REQUIRED_FREE_BYTES" ]; then
      echo "Docker storage ready after orphaned metadata reconciliation: free_bytes=$AVAILABLE required_free_bytes=$REQUIRED_FREE_BYTES runtime_mode=empty"
      exit 0
    fi
  fi
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

# Inventory/prune can take long enough for the deployed N-1 gateway to start
# after the entry scan. Re-read the bounded exact process identity at the last
# possible boundary before arming recovery or stopping either daemon.
protect_exact_host_gateway_runtime "pre-reset"

echo "Resetting orphaned Docker and containerd roots after guarded empty-runtime check"
DOCKER_RESET_STARTED=1
trap recover_docker_daemons_on_exit EXIT
run_bounded_daemon_control \
  sudo systemctl stop docker.service docker.socket containerd.service
run_bounded_daemon_control \
  sudo pkill -TERM -f '^/usr/bin/containerd-shim-runc-v2 -namespace moby ' \
  2>/dev/null || true
sleep 2
run_bounded_daemon_control \
  sudo pkill -KILL -f '^/usr/bin/containerd-shim-runc-v2 -namespace moby ' \
  2>/dev/null || true

# Stale overlay mounts can survive daemon shutdown even though every guarded
# runtime inventory above is empty. Unmount only descendants of the two exact
# validated data roots, deepest paths first, and refuse a lazy/forced unmount.
while IFS= read -r mount_target; do
  echo "Unmounting stale empty-runtime mount: $mount_target"
  run_bounded_daemon_command sudo umount "$mount_target"
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

run_bounded_daemon_command \
  sudo rm -rf --one-file-system "$DOCKER_ROOT"
run_bounded_daemon_command \
  sudo rm -rf --one-file-system "$CONTAINERD_ROOT"
run_bounded_daemon_command \
  sudo install -d -m 0711 -o root -g root "$DOCKER_ROOT"
run_bounded_daemon_command \
  sudo install -d -m 0711 -o root -g root "$CONTAINERD_ROOT"
if ! start_docker_daemons_and_wait; then
  echo "ERROR: Docker/containerd did not become ready after reset" >&2
  exit 1
fi

POST_RESET_MOBY_SHIM_COUNT="$(bounded_moby_shim_count)"
if [ "$(run_bounded_daemon_inventory sudo ctr -n moby containers list -q | sed '/^$/d' | wc -l | tr -d '[:space:]')" -ne 0 ] \
    || [ "$(run_bounded_daemon_inventory sudo ctr -n moby tasks list -q | sed '/^$/d' | wc -l | tr -d '[:space:]')" -ne 0 ] \
    || [ "$POST_RESET_MOBY_SHIM_COUNT" -ne 0 ]; then
  echo "ERROR: Docker/containerd reset left stale moby runtime state" >&2
  exit 1
fi

AVAILABLE="$(available_bytes)"
if [ "$AVAILABLE" -lt "$MIN_FREE_BYTES" ]; then
  echo "ERROR: Docker data-root reset left only $AVAILABLE free bytes" >&2
  exit 1
fi
DOCKER_RESET_STARTED=0
trap - EXIT
echo "Docker storage recovered: free_bytes=$AVAILABLE"
