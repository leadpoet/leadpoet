#!/bin/bash
set -euo pipefail

GATEWAY_GIT_DEPLOY_PROTOCOL="1"
LEADPOET_REPO_ROOT="${LEADPOET_REPO_ROOT:-/home/ec2-user/leadpoet_repo}"
GATEWAY_ROOT="${GATEWAY_ROOT:-$LEADPOET_REPO_ROOT/gateway}"
GATEWAY_LOG_ROOT="${GATEWAY_LOG_ROOT:-/home/ec2-user/gateway}"
GATEWAY_LOG_FILE="${GATEWAY_LOG_FILE:-$GATEWAY_LOG_ROOT/gateway.log}"
LAB_ARENA_SERVICE_LOG_FILE="${LAB_ARENA_SERVICE_LOG_FILE:-$GATEWAY_LOG_ROOT/lab_arena_service.log}"
LAB_ARENA_SERVICE_STATE_FILE="${LAB_ARENA_SERVICE_STATE_FILE:-/home/ec2-user/.config/leadpoet/lab-arena-service-process.json}"
GATEWAY_PRIVATE_KEY_PATH="${GATEWAY_PRIVATE_KEY_PATH:-$GATEWAY_LOG_ROOT/secrets/gateway_private_key.pem}"
ARWEAVE_KEYFILE_PATH="${ARWEAVE_KEYFILE_PATH:-$GATEWAY_LOG_ROOT/secrets/arweave_keyfile.json}"
GATEWAY_RESTART_GIT_SSH_COMMAND="${GATEWAY_RESTART_GIT_SSH_COMMAND:-}"
GATEWAY_ENV_FILE="${GATEWAY_ENV_FILE:-/home/ec2-user/.config/leadpoet/gateway.env}"
LEADPOET_GATEWAY_ENV_SECRET_ID="${LEADPOET_GATEWAY_ENV_SECRET_ID:-leadpoet/prod/gateway/env}"
# Interpreter for all gateway V2 tooling and long-lived processes. Production
# uses the isolated Python 3.11 environment so Bittensor 10's Cyscale namespace
# cannot collide with legacy py-scale-codec packages in the system interpreter.
GATEWAY_PYTHON_BIN="${GATEWAY_PYTHON_BIN:-/home/ec2-user/venv311/bin/python3}"
ENV_CLONE="/tmp/gw_env_clone.sh"
ENV_SECRET="/tmp/gw_env_secret.sh"
MIN_FREE_KB=$((10 * 1024 * 1024))
EXPECTED_AWS_ACCOUNT="493765492819"
LEGACY_FOUR_FILE_CONTROLLER_BOUNDARY="202cd66a41f17f3030bcf6889d381cd3ecfd8f1c"
HISTORICAL_THREE_ROLE_TOPOLOGY_HASH="sha256:a13a1b16fb1501f953b2396aba88b87d7e5e0d3cfac4079b9230ea6165a88f34"
HISTORICAL_THREE_ROLE_TOPOLOGY_BLOB="f79cf108e4a98ca950a0087d786958f92c5f691f"
GATEWAY_HISTORICAL_TOPOLOGY_HASH=""
ENV_BACKUP_DIR="/home/ec2-user/.config/leadpoet/env-backups"
GATEWAY_RESTART_CONTROLLER_ROOT="${GATEWAY_RESTART_CONTROLLER_ROOT:-/home/ec2-user/.config/leadpoet/restart-controller/gateway}"
GATEWAY_RESTART_CONTROLLER_CURRENT="$GATEWAY_RESTART_CONTROLLER_ROOT/current"
GATEWAY_RESTART_AUTHORITY_ROOT="${GATEWAY_RESTART_AUTHORITY_ROOT:-}"
GATEWAY_RESTART_AUTHORITY_COMMIT="${GATEWAY_RESTART_AUTHORITY_COMMIT:-}"
if [ -n "$GATEWAY_RESTART_AUTHORITY_ROOT" ]; then
  GATEWAY_CONTROLLER_PROCESS_HELPER="$GATEWAY_RESTART_AUTHORITY_ROOT/scripts/manage_owned_process_group.py"
else
  GATEWAY_CONTROLLER_PROCESS_HELPER="${LAB_ARENA_PROCESS_HELPER:-$LEADPOET_REPO_ROOT/scripts/manage_owned_process_group.py}"
fi
LAB_ARENA_PROCESS_HELPER="$GATEWAY_CONTROLLER_PROCESS_HELPER"
GATEWAY_CONTROLLER_PROCESS_STATE_FILE="$LAB_ARENA_SERVICE_STATE_FILE"
GATEWAY_ACTIVE_RELEASE_RESTART_INVOCATION_ID="${GATEWAY_ACTIVE_RELEASE_RESTART_INVOCATION_ID:-}"
GATEWAY_ACTIVE_RELEASE_COMPONENT="${GATEWAY_ACTIVE_RELEASE_COMPONENT:-gateway}"
GATEWAY_GIT_HELPER_DEFAULT="$LEADPOET_REPO_ROOT/scripts/gateway_git_deploy.py"
GATEWAY_HOST_MEMORY_GUARD_DEFAULT="$LEADPOET_REPO_ROOT/gateway/tee/host_memory_guard_v2.py"
if [ -n "$GATEWAY_RESTART_AUTHORITY_ROOT" ] \
    && [ -r "$GATEWAY_RESTART_AUTHORITY_ROOT/scripts/gateway_git_deploy.py" ]; then
  GATEWAY_GIT_HELPER_DEFAULT="$GATEWAY_RESTART_AUTHORITY_ROOT/scripts/gateway_git_deploy.py"
  GATEWAY_HOST_MEMORY_GUARD_DEFAULT="$GATEWAY_RESTART_AUTHORITY_ROOT/gateway/tee/host_memory_guard_v2.py"
elif [ -r "$GATEWAY_RESTART_CONTROLLER_CURRENT/scripts/gateway_git_deploy.py" ]; then
  GATEWAY_GIT_HELPER_DEFAULT="$GATEWAY_RESTART_CONTROLLER_CURRENT/scripts/gateway_git_deploy.py"
  GATEWAY_HOST_MEMORY_GUARD_DEFAULT="$GATEWAY_RESTART_CONTROLLER_CURRENT/gateway/tee/host_memory_guard_v2.py"
fi
GATEWAY_GIT_HELPER="${GATEWAY_GIT_HELPER:-$GATEWAY_GIT_HELPER_DEFAULT}"
GATEWAY_RESTART_PHASE="${GATEWAY_RESTART_PHASE:-prepare}"
GATEWAY_RECLAIMABLE_MEMORY_SAFETY_MARGIN_MIB=2048
GATEWAY_V2_HEALTH_MAX_ATTEMPTS="${GATEWAY_V2_HEALTH_MAX_ATTEMPTS:-120}"
GATEWAY_V2_HEALTH_RETRY_SECONDS="${GATEWAY_V2_HEALTH_RETRY_SECONDS:-5}"
GATEWAY_V2_HEALTH_DEADLINE_SECONDS="${GATEWAY_V2_HEALTH_DEADLINE_SECONDS:-600}"
GATEWAY_DEPENDENCY_INSTALL_FINGERPRINT="${GATEWAY_DEPENDENCY_INSTALL_FINGERPRINT:-}"
GATEWAY_RESTART_STARTED_EPOCH="${GATEWAY_RESTART_STARTED_EPOCH:-$(date -u +%s)}"
GATEWAY_RESTART_INVOCATION_ID="${GATEWAY_RESTART_INVOCATION_ID:-gateway-${GATEWAY_RESTART_STARTED_EPOCH}-$$}"
GATEWAY_MIGRATION_203_SQL_SHA256="${GATEWAY_MIGRATION_203_SQL_SHA256:-}"
GATEWAY_MIGRATION_203_BARRIER_FILE="${GATEWAY_MIGRATION_203_BARRIER_FILE:-/home/ec2-user/.config/leadpoet/migration-203-barrier.json}"
GATEWAY_MIGRATION_203_COMPLETION_FILE="${GATEWAY_MIGRATION_203_COMPLETION_FILE:-/home/ec2-user/.config/leadpoet/migration-203-complete.json}"
GATEWAY_MIGRATION_203_WAIT_SECONDS="${GATEWAY_MIGRATION_203_WAIT_SECONDS:-900}"
export GATEWAY_MIGRATION_203_SQL_SHA256 GATEWAY_MIGRATION_203_BARRIER_FILE
export GATEWAY_MIGRATION_203_COMPLETION_FILE GATEWAY_MIGRATION_203_WAIT_SECONDS
GATEWAY_ACTIVE_RELEASE_RESTART_INVOCATION_ID="${GATEWAY_ACTIVE_RELEASE_RESTART_INVOCATION_ID:-$GATEWAY_RESTART_INVOCATION_ID}"
GATEWAY_RELEASE_ATTEMPTS_USED="${GATEWAY_RELEASE_ATTEMPTS_USED:-0}"
GATEWAY_RESTART_TIMING_DIR="${GATEWAY_RESTART_TIMING_DIR:-/home/ec2-user/.config/leadpoet/restart-timings}"
GATEWAY_RESTART_TIMING_FILE="${GATEWAY_RESTART_TIMING_FILE:-$GATEWAY_RESTART_TIMING_DIR/gateway-${GATEWAY_RESTART_STARTED_EPOCH}-$$.jsonl}"
GATEWAY_RESTART_TIMING_INITIALIZED="${GATEWAY_RESTART_TIMING_INITIALIZED:-0}"
PREPARED_GATEWAY_SHA="${PREPARED_GATEWAY_SHA:-}"
LAB_ARENA_RESTART_GUARD_GENERATION="${LAB_ARENA_RESTART_GUARD_GENERATION:-}"
GATEWAY_MINER_MAINTENANCE_BOOTSTRAP_PLAN=""
GATEWAY_MINER_MAINTENANCE_BOOTSTRAP_ROOT=""
GATEWAY_MINER_MAINTENANCE_HANDOFF_FILE=""
GATEWAY_MINER_MAINTENANCE_HANDOFF_NONCE=""

wait_for_exact_migration_203() {
  local deadline
  [ -n "$GATEWAY_MIGRATION_203_SQL_SHA256" ] || return 0
  if ! [[ "$GATEWAY_MIGRATION_203_SQL_SHA256" =~ ^[0-9a-f]{64}$ ]] \
      || ! [[ "$GATEWAY_MIGRATION_203_WAIT_SECONDS" =~ ^[1-9][0-9]{0,3}$ ]]; then
    echo "ERROR: migration 203 barrier configuration is invalid" >&2
    return 1
  fi
  mkdir -p "$(dirname "$GATEWAY_MIGRATION_203_BARRIER_FILE")"
  rm -f -- "$GATEWAY_MIGRATION_203_COMPLETION_FILE"
  "$GATEWAY_PYTHON_BIN" - "$GATEWAY_MIGRATION_203_BARRIER_FILE" \
    "$PREPARED_GATEWAY_SHA" "$GATEWAY_MIGRATION_203_SQL_SHA256" \
    "$GATEWAY_RESTART_INVOCATION_ID" <<'PY'
import json, os, sys, tempfile
from pathlib import Path
path = Path(sys.argv[1])
doc = {"schema_version":"leadpoet.gateway.migration_203_barrier.v1",
       "candidate_commit":sys.argv[2], "sql_sha256":sys.argv[3],
       "restart_invocation_id":sys.argv[4], "old_producers_stopped":True}
fd, name = tempfile.mkstemp(prefix=".migration-203.", dir=str(path.parent))
try:
    with os.fdopen(fd, "w", encoding="ascii") as handle:
        json.dump(doc, handle, sort_keys=True, separators=(",", ":")); handle.write("\n")
        handle.flush(); os.fsync(handle.fileno())
    os.chmod(name, 0o600); os.replace(name, path)
finally:
    try: os.unlink(name)
    except FileNotFoundError: pass
PY
  echo "Migration 203 barrier ready; old incentive producers are stopped"
  deadline=$((SECONDS + GATEWAY_MIGRATION_203_WAIT_SECONDS))
  while [ "$SECONDS" -lt "$deadline" ]; do
    if "$GATEWAY_PYTHON_BIN" - "$GATEWAY_MIGRATION_203_COMPLETION_FILE" \
      "$PREPARED_GATEWAY_SHA" "$GATEWAY_MIGRATION_203_SQL_SHA256" \
      "$GATEWAY_RESTART_INVOCATION_ID" <<'PY'
import json, os, stat, sys
path = sys.argv[1]
try:
    info = os.lstat(path)
    if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600 or info.st_uid != os.geteuid() or info.st_size > 4096:
        raise ValueError("unsafe completion file")
    with open(path, "r", encoding="ascii") as handle: doc = json.load(handle)
except (OSError, ValueError, json.JSONDecodeError):
    raise SystemExit(1)
expected = {"schema_version":"leadpoet.gateway.migration_203_complete.v1",
            "candidate_commit":sys.argv[2], "sql_sha256":sys.argv[3],
            "restart_invocation_id":sys.argv[4], "migration_203_verified":True}
raise SystemExit(0 if doc == expected else 1)
PY
    then
      echo "Exact migration 203 completion received"
      return 0
    fi
    sleep 2
  done
  echo "ERROR: exact migration 203 completion was not received; gateway remains stopped" >&2
  return 1
}

if [ -n "$GATEWAY_RESTART_AUTHORITY_ROOT" ]; then
  if ! [[ "$GATEWAY_RESTART_AUTHORITY_ROOT" =~ ^/tmp/gateway-restart-controller-bootstrap\.[A-Za-z0-9]+/authority$|^/tmp/gateway-miner-maintenance-bootstrap\.[A-Za-z0-9]+/authority$ ]] \
      || ! [[ "$GATEWAY_RESTART_AUTHORITY_COMMIT" =~ ^[0-9a-f]{40}$ ]] \
      || [ ! -r "$GATEWAY_RESTART_AUTHORITY_ROOT/gw_restart.sh" ] \
      || [ -L "$GATEWAY_RESTART_AUTHORITY_ROOT/gw_restart.sh" ]; then
    echo "ERROR: gateway restart authority controller is invalid" >&2
    exit 2
  fi
fi
gateway_restart_invocation_id_from_timing_file() {
  local ledger_name ledger_epoch ledger_pid expected_ledger
  if [ -L "$GATEWAY_RESTART_TIMING_FILE" ]; then
    echo "ERROR: gateway restart timing ledger must not be a symlink" >&2
    return 1
  fi
  if [ ! -f "$GATEWAY_RESTART_TIMING_FILE" ]; then
    echo "ERROR: gateway restart timing ledger is unavailable" >&2
    return 1
  fi
  if [ ! -s "$GATEWAY_RESTART_TIMING_FILE" ]; then
    echo "ERROR: gateway restart timing ledger is empty" >&2
    return 1
  fi
  ledger_name="${GATEWAY_RESTART_TIMING_FILE##*/}"
  if ! [[ "$ledger_name" =~ ^gateway-([0-9]+)-([0-9]+)\.jsonl$ ]]; then
    echo "ERROR: gateway restart timing ledger name is invalid" >&2
    return 1
  fi
  ledger_epoch="${BASH_REMATCH[1]}"
  ledger_pid="${BASH_REMATCH[2]}"
  if [ "$ledger_epoch" != "$GATEWAY_RESTART_STARTED_EPOCH" ]; then
    echo "ERROR: gateway restart timing ledger epoch differs from active restart" >&2
    return 1
  fi
  if [ "$ledger_pid" != "$$" ]; then
    echo "ERROR: gateway restart timing ledger PID differs from active restart" >&2
    return 1
  fi
  expected_ledger="${GATEWAY_RESTART_TIMING_DIR%/}/$ledger_name"
  if [ "$GATEWAY_RESTART_TIMING_FILE" != "$expected_ledger" ]; then
    echo "ERROR: gateway restart timing ledger is outside the canonical directory" >&2
    return 1
  fi
  printf 'gateway-%s-%s\n' "$ledger_epoch" "$ledger_pid"
}

bind_gateway_restart_invocation_to_timing_file() {
  local authoritative_invocation_id
  if [ "$GATEWAY_RESTART_TIMING_INITIALIZED" != "1" ]; then
    return 0
  fi
  if ! authoritative_invocation_id="$(
      gateway_restart_invocation_id_from_timing_file
    )"; then
    return 1
  fi
  GATEWAY_RESTART_INVOCATION_ID="$authoritative_invocation_id"
  LEADPOET_RESTART_INVOCATION_ID="$authoritative_invocation_id"
  export GATEWAY_RESTART_INVOCATION_ID
  export LEADPOET_RESTART_INVOCATION_ID
}

wait_for_gateway_v2_authority() {
  local attempt deadline
  if ! [[ "$GATEWAY_V2_HEALTH_MAX_ATTEMPTS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: GATEWAY_V2_HEALTH_MAX_ATTEMPTS must be a positive integer" >&2
    return 1
  fi
  if ! [[ "$GATEWAY_V2_HEALTH_RETRY_SECONDS" =~ ^[0-9]+$ ]]; then
    echo "ERROR: GATEWAY_V2_HEALTH_RETRY_SECONDS must be a non-negative integer" >&2
    return 1
  fi
  if ! [[ "$GATEWAY_V2_HEALTH_DEADLINE_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: GATEWAY_V2_HEALTH_DEADLINE_SECONDS must be a positive integer" >&2
    return 1
  fi
  deadline=$((SECONDS + GATEWAY_V2_HEALTH_DEADLINE_SECONDS))
  for attempt in $(seq 1 "$GATEWAY_V2_HEALTH_MAX_ATTEMPTS"); do
    GATEWAY_PID="$(
      pgrep -f "^$GATEWAY_PYTHON_BIN -u -m gateway[.]main$" | head -1 || true
    )"
    if [ -z "$GATEWAY_PID" ]; then
      tail -160 "$GATEWAY_LOG_FILE"
      echo "ERROR: gateway exited while authoritative V2 readiness was pending" >&2
      return 1
    fi
    if timeout 5 curl -fsS http://localhost:8000/health/v2-authority >/dev/null 2>&1; then
      echo "Gateway authoritative V2 health ready after attempt $attempt"
      return 0
    fi
    if [ "$attempt" -ge "$GATEWAY_V2_HEALTH_MAX_ATTEMPTS" ] \
        || [ "$SECONDS" -ge "$deadline" ]; then
      break
    fi
    sleep "$GATEWAY_V2_HEALTH_RETRY_SECONDS"
  done
  tail -160 "$GATEWAY_LOG_FILE"
  echo "ERROR: authoritative V2 enclave/worker readiness did not become ready before the bounded deadline" >&2
  return 1
}

stop_lab_arena_service() {
  local process_helper="${1:-$GATEWAY_CONTROLLER_PROCESS_HELPER}"
  if ! verify_controller_process_helper "$process_helper"; then
    echo "ERROR: verified controller Lab Arena stop helper is unavailable" >&2
    return 1
  fi
  "$GATEWAY_PYTHON_BIN" "$process_helper" stop \
    --state-file "$GATEWAY_CONTROLLER_PROCESS_STATE_FILE" \
    --cwd "$LEADPOET_REPO_ROOT" \
    --uid "$(id -u)" \
    -- \
    "$GATEWAY_PYTHON_BIN" -u scripts/run_lab_arena_service.py \
    --environment-file "$GATEWAY_ENV_FILE" \
    --host 127.0.0.1 --port 8792
}

verify_controller_process_helper() {
  local process_helper="${1:-}"
  if [ "$process_helper" = "/proc/self/fd/195" ]; then
    "$GATEWAY_PYTHON_BIN" - "$process_helper" <<'PY'
import fcntl
import hashlib
import json
import os
import stat
import sys

path = sys.argv[1]
if path != "/proc/self/fd/195":
    raise SystemExit("controller process helper descriptor path is invalid")
required_seals = sum(
    int(getattr(fcntl, name))
    for name in ("F_SEAL_WRITE", "F_SEAL_GROW", "F_SEAL_SHRINK", "F_SEAL_SEAL")
)
try:
    original = os.fstat(195)
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(original.st_mode)
            or original.st_uid != os.geteuid()
            or original.st_gid != os.getegid()
            or stat.S_IMODE(original.st_mode) != 0o400
            or not os.get_inheritable(195)
            or os.readlink(path) != "/memfd:leadpoet-process-helper (deleted)"
            or (original.st_dev, original.st_ino) != (opened.st_dev, opened.st_ino)
            or int(fcntl.fcntl(195, fcntl.F_GET_SEALS)) & required_seals
            != required_seals
        ):
            raise SystemExit("controller process helper descriptor identity is unsafe")
        if os.environ.get("GATEWAY_MINER_MAINTENANCE_PROOF_FD") != "190":
            raise SystemExit("controller process helper proof descriptor is unavailable")
        proof = os.fstat(190)
        if (
            not stat.S_ISREG(proof.st_mode)
            or proof.st_uid != os.geteuid()
            or proof.st_gid != os.getegid()
            or stat.S_IMODE(proof.st_mode) != 0o400
            or not os.get_inheritable(190)
            or int(fcntl.fcntl(190, fcntl.F_GET_SEALS)) & required_seals
            != required_seals
        ):
            raise SystemExit("controller process helper proof identity is unsafe")
        process_payload = os.pread(195, 4 * 1024 * 1024 + 1, 0)
        proof_payload = os.pread(190, 32 * 1024 + 1, 0)
        document = json.loads(proof_payload.decode("ascii"))
        expected = document.get("controller_process_helper_sha256")
        observed = "sha256:" + hashlib.sha256(process_payload).hexdigest()
        if expected != observed:
            raise SystemExit("controller process helper proof commitment differs")
    finally:
        os.close(descriptor)
except OSError as exc:
    raise SystemExit("controller process helper descriptor is unavailable") from exc
PY
    return $?
  fi
  [ -r "$process_helper" ] && [ ! -L "$process_helper" ]
}

start_lab_arena_service() {
  local mode pid
  mode="${LAB_ARENA_MODE:-off}"
  case "$mode" in
    off)
      echo "Lab Arena service is disabled"
      return 0
      ;;
    shadow|live) ;;
    *)
      echo "ERROR: LAB_ARENA_MODE must be off, shadow, or live" >&2
      return 1
      ;;
  esac
  if [ ! -r "$LEADPOET_REPO_ROOT/scripts/run_lab_arena_service.py" ]; then
    echo "ERROR: Lab Arena service entrypoint is unavailable" >&2
    return 1
  fi
  if ! verify_controller_process_helper "$GATEWAY_CONTROLLER_PROCESS_HELPER"; then
    echo "ERROR: verified controller Lab Arena process helper is unavailable" >&2
    return 1
  fi
  mkdir -p "$(dirname "$LAB_ARENA_SERVICE_LOG_FILE")"
  cd "$LEADPOET_REPO_ROOT"
  env -u GATEWAY_MINER_MAINTENANCE_PROOF_FD \
    -u GATEWAY_CONTROLLER_PROCESS_HELPER \
    -u LAB_ARENA_PROCESS_HELPER \
    -u GATEWAY_RESTART_AUTHORITY_ROOT \
    -u GATEWAY_RESTART_AUTHORITY_COMMIT \
    -u PREPARED_GATEWAY_SHA \
    -u LAB_ARENA_RESTART_GUARD_GENERATION \
    setsid "$GATEWAY_PYTHON_BIN" -u scripts/run_lab_arena_service.py \
      --environment-file "$GATEWAY_ENV_FILE" \
      --host 127.0.0.1 --port 8792 \
      > "$LAB_ARENA_SERVICE_LOG_FILE" 2>&1 < /dev/null \
      9>&- 190>&- 191>&- 192>&- 193>&- 194>&- 195>&- &
  pid="$!"
  if ! "$GATEWAY_PYTHON_BIN" "$GATEWAY_CONTROLLER_PROCESS_HELPER" record \
      --state-file "$GATEWAY_CONTROLLER_PROCESS_STATE_FILE" \
      --cwd "$LEADPOET_REPO_ROOT" \
      --uid "$(id -u)" \
      --launch-pgid "$pid" \
      --discover-timeout-seconds 5 \
      -- \
      "$GATEWAY_PYTHON_BIN" -u scripts/run_lab_arena_service.py \
      --environment-file "$GATEWAY_ENV_FILE" \
      --host 127.0.0.1 --port 8792; then
    kill -TERM -- "-$pid" 2>/dev/null || true
    sleep 1
    kill -KILL -- "-$pid" 2>/dev/null || true
    echo "ERROR: Lab Arena service ownership could not be recorded" >&2
    return 1
  fi
  for attempt in $(seq 1 30); do
    if ! kill -0 "$pid" 2>/dev/null; then
      tail -120 "$LAB_ARENA_SERVICE_LOG_FILE" >&2 || true
      echo "ERROR: Lab Arena service exited during startup" >&2
      wait "$pid" 2>/dev/null || true
      return 1
    fi
    if timeout 5 curl -fsS http://127.0.0.1:8792/arena/v1/current >/dev/null 2>&1 \
        && timeout 5 curl -fsS http://127.0.0.1:8000/arena/v1/current >/dev/null 2>&1; then
      echo "Lab Arena service ready after attempt $attempt"
      return 0
    fi
    sleep 2
  done
  stop_lab_arena_service || true
  wait "$pid" 2>/dev/null || true
  tail -120 "$LAB_ARENA_SERVICE_LOG_FILE" >&2 || true
  echo "ERROR: Lab Arena service did not become ready" >&2
  return 1
}

run_lab_arena_restart_guard() {
  local source_root="$1"
  shift
  local guard_args=("$@" --environment-file "$GATEWAY_ENV_FILE")
  local -a guard_environment=(
    env
    -u LAB_ARENA_SUPABASE_URL
    -u LAB_ARENA_SUPABASE_ANON_KEY
    -u LAB_ARENA_SERVICE_KEY
    -u LAB_ARENA_SERVICE_JWT
  )
  if [ ! -r "$source_root/scripts/lab_arena_restart_claim_guard.py" ]; then
    echo "ERROR: exact Lab Arena restart guard helper is unavailable" >&2
    return 1
  fi
  "${guard_environment[@]}" PYTHONPATH="$source_root" "$GATEWAY_PYTHON_BIN" \
    "$source_root/scripts/lab_arena_restart_claim_guard.py" "${guard_args[@]}" \
    --candidate "$PREPARED_GATEWAY_SHA" \
    --invocation "$GATEWAY_ACTIVE_RELEASE_RESTART_INVOCATION_ID"
}

validate_post_activate_arena_guard_authority() {
  if ! [[ "$PREPARED_GATEWAY_SHA" =~ ^[0-9a-f]{40}$ ]]; then
    echo "ERROR: post-activation Lab Arena guard candidate is invalid" >&2
    return 1
  fi
  if ! [[ "$LAB_ARENA_RESTART_GUARD_GENERATION" =~ ^[1-9][0-9]{0,18}$ ]] \
      || { [ "${#LAB_ARENA_RESTART_GUARD_GENERATION}" -eq 19 ] \
        && [[ "$LAB_ARENA_RESTART_GUARD_GENERATION" > "9223372036854775807" ]]; }; then
    echo "ERROR: post-activation Lab Arena guard generation is invalid" >&2
    return 1
  fi
}

bind_activated_gateway_guard_candidate() {
  GATEWAY_DEPLOY_SHA="$(deployment_field target_sha)"
  GATEWAY_DEPLOY_BRANCH="$(deployment_field branch)"
  GATEWAY_DEPLOY_REMOTE="$(deployment_field remote_url)"
  if [ "$PREPARED_GATEWAY_SHA" != "$GATEWAY_DEPLOY_SHA" ] \
      || [ "$(git -C "$LEADPOET_REPO_ROOT" rev-parse HEAD)" != "$GATEWAY_DEPLOY_SHA" ]; then
    echo "ERROR: canonical gateway checkout, prepared guard candidate, and activated deployment differ" >&2
    return 1
  fi
}

abort_lab_arena_restart_guard_before_destructive() {
  local source_root
  if [ -z "$LAB_ARENA_RESTART_GUARD_GENERATION" ] \
      || [ "$GATEWAY_DESTRUCTIVE_PHASE_STARTED" = "1" ]; then
    return 0
  fi
  source_root="${GATEWAY_PREFLIGHT_TREE:-$LEADPOET_REPO_ROOT}"
  run_lab_arena_restart_guard "$source_root" abort \
    --generation "$LAB_ARENA_RESTART_GUARD_GENERATION" >/dev/null 2>&1 || true
  LAB_ARENA_RESTART_GUARD_GENERATION=""
}

drain_lab_arena_for_restart() {
  local source_root="$1" report
  report="$(run_lab_arena_restart_guard "$source_root" drain \
    --scope "$GATEWAY_ACTIVE_RELEASE_COMPONENT")" || return 1
  LAB_ARENA_RESTART_GUARD_GENERATION="$(
    "$GATEWAY_PYTHON_BIN" -c \
      'import json,sys; value=json.load(sys.stdin); generation=value.get("guard_generation"); assert type(generation) is int and generation > 0; print(generation)' \
      <<<"$report"
  )" || return 1
  echo "Lab Arena claims paused and existing leases have durable completion receipts"
}

# The N-1 controller can carry a stale process environment across the exact
# candidate exec.  Once a timing ledger exists, its validated basename is the
# per-invocation authority shared by both controller generations.
bind_gateway_restart_invocation_to_timing_file
GATEWAY_RELEASE_FOLLOW_ROOT="${GATEWAY_RELEASE_FOLLOW_ROOT:-}"
GATEWAY_RELEASE_SUPERSESSION_COUNT="${GATEWAY_RELEASE_SUPERSESSION_COUNT:-0}"
GATEWAY_RELEASE_SUPERSESSION_MAX="${GATEWAY_RELEASE_SUPERSESSION_MAX:-20}"
GATEWAY_OFFLINE_ARTIFACT_PREPARE_PID=""
GATEWAY_OFFLINE_ARTIFACT_PREPARE_LOG="${GATEWAY_OFFLINE_ARTIFACT_PREPARE_LOG:-${GATEWAY_RESTART_TIMING_FILE%.jsonl}.offline-artifacts.log}"
GATEWAY_RESTART_EPOCH_REPORT=""
GATEWAY_RESTART_START_PATH="/home/ec2-user/.config/leadpoet/restart-start-v1.json"
GATEWAY_RESTART_LOCK_FILE="${GATEWAY_RESTART_LOCK_FILE:-/home/ec2-user/.config/leadpoet/gateway-restart.lock}"
GATEWAY_RESTART_RECOVERY_LOCK_FILE="${GATEWAY_RESTART_RECOVERY_LOCK_FILE:-${GATEWAY_RESTART_LOCK_FILE}.recovery}"
GATEWAY_DEPLOY_PLAN_FILE="${GATEWAY_DEPLOY_PLAN_FILE:-/tmp/gateway_git_deploy.$$.json}"
GATEWAY_DEPLOYMENT_DIR="${GATEWAY_DEPLOYMENT_DIR:-/home/ec2-user/.config/leadpoet/deployments}"
GATEWAY_DEPLOYMENT_MANIFEST="${GATEWAY_DEPLOYMENT_MANIFEST:-$GATEWAY_DEPLOYMENT_DIR/gateway-current.json}"
GATEWAY_LAST_GOOD_MANIFEST="${GATEWAY_LAST_GOOD_MANIFEST:-$GATEWAY_DEPLOYMENT_DIR/gateway-last-good.json}"
GATEWAY_HOST_RESTART_SCRIPT="${GATEWAY_HOST_RESTART_SCRIPT:-/home/ec2-user/gw_restart.sh}"
GATEWAY_TEE_EIF_ROOT="${GATEWAY_TEE_EIF_ROOT:-/home/ec2-user/tee}"
GATEWAY_V2_RELEASE_ARCHIVE_ROOT="${GATEWAY_V2_RELEASE_ARCHIVE_ROOT:-$GATEWAY_TEE_EIF_ROOT/releases-v2}"
GATEWAY_RESTART_TEMP_CLEANUP_MIN_AGE_SECONDS="${GATEWAY_RESTART_TEMP_CLEANUP_MIN_AGE_SECONDS:-86400}"
GATEWAY_RESTART_EMERGENCY_BACKUP_MIN_AGE_SECONDS="${GATEWAY_RESTART_EMERGENCY_BACKUP_MIN_AGE_SECONDS:-604800}"
GATEWAY_RESTART_CLEANUP_MAX_CANDIDATES="${GATEWAY_RESTART_CLEANUP_MAX_CANDIDATES:-64}"
RESEARCH_LAB_TEE_PROTOCOL="${RESEARCH_LAB_TEE_PROTOCOL:-}"
GATEWAY_V2_CONFIG_DIR="${GATEWAY_V2_CONFIG_DIR:-/home/ec2-user/.config/leadpoet/v2}"
GATEWAY_V2_RELEASE_MANIFEST="${GATEWAY_V2_RELEASE_MANIFEST:-$GATEWAY_TEE_EIF_ROOT/gateway-v2-release-manifest.json}"
GATEWAY_V2_RELEASE_LINEAGE="${GATEWAY_V2_RELEASE_LINEAGE:-$GATEWAY_TEE_EIF_ROOT/gateway-v2-release-lineage.json}"
# Release acquisition happens while the existing gateway is still serving.
# Keep candidate evidence restart-scoped so its fail-closed verifier remains
# bound to the release that actually booted it until destructive cutover.
GATEWAY_PREPARED_V2_RELEASE_MANIFEST="${GATEWAY_PREPARED_V2_RELEASE_MANIFEST:-${GATEWAY_RESTART_TIMING_FILE%.jsonl}.candidate-release.json}"
GATEWAY_PREPARED_V2_RELEASE_LINEAGE="${GATEWAY_PREPARED_V2_RELEASE_LINEAGE:-${GATEWAY_RESTART_TIMING_FILE%.jsonl}.candidate-release-lineage.json}"
GATEWAY_V2_ARTIFACT_POLICY="${GATEWAY_V2_ARTIFACT_POLICY:-$GATEWAY_V2_CONFIG_DIR/encrypted-artifact-policy.json}"
GATEWAY_V2_RELEASE_BUCKET="${GATEWAY_V2_RELEASE_BUCKET:-leadpoet-attested-v2-artifacts-493765492819}"
RESEARCH_LAB_ATTESTED_V2_ARTIFACT_BUCKET="${RESEARCH_LAB_ATTESTED_V2_ARTIFACT_BUCKET:-$GATEWAY_V2_RELEASE_BUCKET}"
GATEWAY_V2_RELEASE_PREFIX="${GATEWAY_V2_RELEASE_PREFIX:-attested-v2/releases}"
GATEWAY_V2_KMS_KEY_ID="${GATEWAY_V2_KMS_KEY_ID:-arn:aws:kms:us-east-1:493765492819:key/c5412928-093e-4bf5-aafc-7b27c02f1445}"
export GATEWAY_V2_OFFLINE_ARTIFACT_ROOT="${GATEWAY_V2_OFFLINE_ARTIFACT_ROOT:-$HOME/.cache/leadpoet-v2-artifacts}"
GATEWAY_DEPLOY_STAGE="${GATEWAY_DEPLOY_STAGE:-bootstrap}"
GATEWAY_DEPLOY_COMPLETED=0
GATEWAY_DESTRUCTIVE_PHASE_STARTED="${GATEWAY_DESTRUCTIVE_PHASE_STARTED:-0}"
GATEWAY_PREFLIGHT_TREE=""
GATEWAY_HOST_MEMORY_GUARD_PATH="${GATEWAY_HOST_MEMORY_GUARD_PATH:-$GATEWAY_HOST_MEMORY_GUARD_DEFAULT}"
LEADPOET_DOCKER_OPERATION_LOCK_FILE="${LEADPOET_DOCKER_OPERATION_LOCK_FILE:-/home/ec2-user/.config/leadpoet/docker-operation-v2.lock}"

# Environment documents may contain stale copies of restart-only paths. Keep
# the invocation's already-resolved controller boundary authoritative across
# every later parent environment reload.
GATEWAY_RESTART_PATH_AUTHORITY_KEYS=(
  GATEWAY_ENV_FILE
  GATEWAY_PRIVATE_KEY_PATH
  ARWEAVE_KEYFILE_PATH
  GATEWAY_RESTART_GIT_SSH_COMMAND
  LEADPOET_GATEWAY_ENV_SECRET_ID
  GATEWAY_RESTART_CONTROLLER_ROOT
  GATEWAY_RESTART_AUTHORITY_ROOT
  GATEWAY_RESTART_AUTHORITY_COMMIT
  GATEWAY_CONTROLLER_PROCESS_HELPER
  GATEWAY_CONTROLLER_PROCESS_STATE_FILE
  LAB_ARENA_PROCESS_HELPER
  LAB_ARENA_SERVICE_STATE_FILE
  GATEWAY_ACTIVE_RELEASE_RESTART_INVOCATION_ID
  GATEWAY_RESTART_RECOVERY_LOCK_FILE
  LEADPOET_DOCKER_OPERATION_LOCK_FILE
  GATEWAY_V2_CONFIG_DIR
  GATEWAY_V2_RELEASE_MANIFEST
  GATEWAY_V2_RELEASE_LINEAGE
  GATEWAY_PREPARED_V2_RELEASE_MANIFEST
  GATEWAY_PREPARED_V2_RELEASE_LINEAGE
  GATEWAY_V2_ARTIFACT_POLICY
  GATEWAY_V2_OFFLINE_ARTIFACT_ROOT
  GATEWAY_V2_RELEASE_PREFIX
  GATEWAY_V2_RELEASE_BUCKET
  GATEWAY_V2_KMS_KEY_ID
)
GATEWAY_RESTART_PATH_AUTHORITY_VALUES=()
for authority_key in "${GATEWAY_RESTART_PATH_AUTHORITY_KEYS[@]}"; do
  GATEWAY_RESTART_PATH_AUTHORITY_VALUES+=("${!authority_key}")
done
readonly -a GATEWAY_RESTART_PATH_AUTHORITY_KEYS
readonly -a GATEWAY_RESTART_PATH_AUTHORITY_VALUES

restore_gateway_restart_path_authority() {
  local authority_index authority_key
  for ((authority_index = 0; authority_index < ${#GATEWAY_RESTART_PATH_AUTHORITY_KEYS[@]}; authority_index++)); do
    authority_key="${GATEWAY_RESTART_PATH_AUTHORITY_KEYS[$authority_index]}"
    printf -v "$authority_key" '%s' \
      "${GATEWAY_RESTART_PATH_AUTHORITY_VALUES[$authority_index]}"
    export "$authority_key"
  done
  if [ -n "$GATEWAY_RESTART_GIT_SSH_COMMAND" ]; then
    GIT_SSH_COMMAND="$GATEWAY_RESTART_GIT_SSH_COMMAND"
    export GIT_SSH_COMMAND
  fi
}

restore_gateway_restart_path_authority
REQUESTED_GATEWAY_DEPLOY_COMMIT="${GATEWAY_DEPLOY_COMMIT:-}"
unset GATEWAY_DEPLOY_COMMIT
export GATEWAY_RESTART_INVOCATION_ID
export LEADPOET_RESTART_INVOCATION_ID="$GATEWAY_RESTART_INVOCATION_ID"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --commit)
      if [ "$#" -lt 2 ] || [ -z "${2:-}" ]; then
        echo "ERROR: --commit requires a full 40-character SHA" >&2
        exit 2
      fi
      if [ -n "$REQUESTED_GATEWAY_DEPLOY_COMMIT" ] \
          && [ "$REQUESTED_GATEWAY_DEPLOY_COMMIT" != "$2" ]; then
        echo "ERROR: --commit conflicts with GATEWAY_DEPLOY_COMMIT" >&2
        exit 2
      fi
      REQUESTED_GATEWAY_DEPLOY_COMMIT="$2"
      shift 2
      ;;
    --commit=*)
      requested_commit="${1#--commit=}"
      if [ -z "$requested_commit" ]; then
        echo "ERROR: --commit requires a full 40-character SHA" >&2
        exit 2
      fi
      if [ -n "$REQUESTED_GATEWAY_DEPLOY_COMMIT" ] \
          && [ "$REQUESTED_GATEWAY_DEPLOY_COMMIT" != "$requested_commit" ]; then
        echo "ERROR: --commit conflicts with GATEWAY_DEPLOY_COMMIT" >&2
        exit 2
      fi
      REQUESTED_GATEWAY_DEPLOY_COMMIT="$requested_commit"
      shift
      ;;
    --miner-maintenance-bootstrap-plan)
      if [ "$#" -lt 2 ] || [ -z "${2:-}" ]; then
        echo "ERROR: --miner-maintenance-bootstrap-plan requires a path" >&2
        exit 2
      fi
      GATEWAY_MINER_MAINTENANCE_BOOTSTRAP_PLAN="$2"
      shift 2
      ;;
    --miner-maintenance-bootstrap-root)
      if [ "$#" -lt 2 ] || [ -z "${2:-}" ]; then
        echo "ERROR: --miner-maintenance-bootstrap-root requires a path" >&2
        exit 2
      fi
      GATEWAY_MINER_MAINTENANCE_BOOTSTRAP_ROOT="$2"
      shift 2
      ;;
    --miner-maintenance-handoff-file)
      if [ "$#" -lt 2 ] || [ -z "${2:-}" ]; then
        echo "ERROR: --miner-maintenance-handoff-file requires a path" >&2
        exit 2
      fi
      GATEWAY_MINER_MAINTENANCE_HANDOFF_FILE="$2"
      shift 2
      ;;
    --miner-maintenance-handoff-nonce)
      if [ "$#" -lt 2 ] || [ -z "${2:-}" ]; then
        echo "ERROR: --miner-maintenance-handoff-nonce requires a value" >&2
        exit 2
      fi
      GATEWAY_MINER_MAINTENANCE_HANDOFF_NONCE="$2"
      shift 2
      ;;
    *)
      echo "ERROR: unsupported gateway restart argument: $1" >&2
      exit 2
      ;;
  esac
done
if [ -n "$REQUESTED_GATEWAY_DEPLOY_COMMIT" ] \
    && ! [[ "$REQUESTED_GATEWAY_DEPLOY_COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
  echo "ERROR: --commit must be a lowercase full 40-character SHA" >&2
  exit 2
fi
miner_maintenance_bootstrap_values=(
  "$GATEWAY_MINER_MAINTENANCE_BOOTSTRAP_PLAN"
  "$GATEWAY_MINER_MAINTENANCE_BOOTSTRAP_ROOT"
  "$GATEWAY_MINER_MAINTENANCE_HANDOFF_FILE"
  "$GATEWAY_MINER_MAINTENANCE_HANDOFF_NONCE"
)
miner_maintenance_bootstrap_count=0
for bootstrap_value in "${miner_maintenance_bootstrap_values[@]}"; do
  [ -n "$bootstrap_value" ] && miner_maintenance_bootstrap_count=$((miner_maintenance_bootstrap_count + 1))
done
if [ "$miner_maintenance_bootstrap_count" -ne 0 ] \
    && [ "$miner_maintenance_bootstrap_count" -ne 4 ]; then
  echo "ERROR: miner-maintenance bootstrap arguments must be supplied together" >&2
  exit 2
fi
if [ -n "${GATEWAY_MINER_MAINTENANCE_PROOF_FD:-}" ]; then
  if [ "$miner_maintenance_bootstrap_count" -ne 0 ] \
      || [ "$GATEWAY_MINER_MAINTENANCE_PROOF_FD" != "190" ] \
      || [ ! -r "/proc/$$/fd/190" ]; then
    echo "ERROR: miner-maintenance invocation proof descriptor is invalid" >&2
    exit 2
  fi
fi
if [ "$miner_maintenance_bootstrap_count" -eq 4 ]; then
  if [ -z "$REQUESTED_GATEWAY_DEPLOY_COMMIT" ]; then
    echo "ERROR: miner-maintenance bootstrap requires --commit" >&2
    exit 2
  fi
  if ! [[ "$GATEWAY_MINER_MAINTENANCE_BOOTSTRAP_ROOT" =~ ^/tmp/gateway-miner-maintenance-bootstrap\.[A-Za-z0-9]+$ ]] \
      || [ "$GATEWAY_MINER_MAINTENANCE_BOOTSTRAP_PLAN" != \
        "$GATEWAY_MINER_MAINTENANCE_BOOTSTRAP_ROOT/plan.json" ] \
      || ! [[ "$GATEWAY_MINER_MAINTENANCE_HANDOFF_FILE" =~ ^/tmp/leadpoet-gateway-miner-maintenance-handoff\.[A-Za-z0-9._-]+$ ]] \
      || ! [[ "$GATEWAY_MINER_MAINTENANCE_HANDOFF_NONCE" =~ ^[0-9a-f]{64}$ ]]; then
    echo "ERROR: miner-maintenance bootstrap authority is invalid" >&2
    exit 2
  fi
fi
if ! [[ "$GATEWAY_RELEASE_SUPERSESSION_COUNT" =~ ^[0-9]+$ ]] \
    || ! [[ "$GATEWAY_RELEASE_SUPERSESSION_MAX" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: gateway release supersession counters are invalid" >&2
  exit 2
fi
V2_CREDENTIAL_ENVELOPES=(
  "$GATEWAY_V2_CONFIG_DIR/artifact_master_key.json"
  "$GATEWAY_V2_CONFIG_DIR/openrouter.json"
  "$GATEWAY_V2_CONFIG_DIR/exa.json"
  "$GATEWAY_V2_CONFIG_DIR/scrapingdog.json"
  "$GATEWAY_V2_CONFIG_DIR/deepline.json"
  "$GATEWAY_V2_CONFIG_DIR/supabase_service_role.json"
  "$GATEWAY_V2_CONFIG_DIR/truelist.json"
)
GATEWAY_HOST_EXTRA_PYTHON_PACKAGES=(
  minio
  awscli
)

record_gateway_restart_timing() {
  local stage="$1"
  local status="${2:-reached}"
  if ! mkdir -p "$GATEWAY_RESTART_TIMING_DIR" \
      || ! chmod 700 "$GATEWAY_RESTART_TIMING_DIR"; then
    echo "WARNING: gateway restart timing directory is unavailable" >&2
    return 0
  fi
  if ! python3 - \
    "$GATEWAY_RESTART_TIMING_FILE" \
    "$GATEWAY_RESTART_STARTED_EPOCH" \
    "$stage" \
    "$status" \
    "${GATEWAY_DEPLOY_SHA:-${PREPARED_GATEWAY_SHA:-}}" <<'PY'
import datetime
import json
import os
import sys
import time

path, started, stage, status, commit = sys.argv[1:]
now = time.time()
record = {
    "schema_version": "leadpoet.gateway_restart_timing.v1",
    "stage": stage,
    "status": status,
    "observed_at": datetime.datetime.fromtimestamp(
        now, datetime.timezone.utc
    ).isoformat().replace("+00:00", "Z"),
    "elapsed_seconds": round(now - int(started), 3),
    "commit_sha": commit or None,
}
with open(path, "a", encoding="utf-8") as handle:
    handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
os.chmod(path, 0o600)
print(
    "GATEWAY_RESTART_TIMING "
    f"stage={stage} status={status} elapsed_seconds={record['elapsed_seconds']}"
)
PY
  then
    echo "WARNING: gateway restart timing event could not be recorded: $stage" >&2
  fi
  return 0
}

emit_gateway_restart_sentry_summary() {
  local status="$1" candidate_sha summary_status shutdown_flag=()
  command -v timeout >/dev/null 2>&1 || return 0
  [ -x "$GATEWAY_PYTHON_BIN" ] || return 0
  [ -r "$LEADPOET_REPO_ROOT/leadpoet_observability/sentry_cli.py" ] || return 0
  candidate_sha="${GATEWAY_DEPLOY_SHA:-${PREPARED_GATEWAY_SHA:-}}"
  summary_status="failed"
  [ "$status" -eq 0 ] && summary_status="passed"
  if [ "${GATEWAY_DESTRUCTIVE_PHASE_STARTED:-0}" = "1" ]; then
    shutdown_flag=(--shutdown-started)
  fi
  PYTHONPATH="$LEADPOET_REPO_ROOT" timeout 2 \
    "$GATEWAY_PYTHON_BIN" -m leadpoet_observability.sentry_cli \
    restart-summary \
    --component gateway \
    --status "$summary_status" \
    --stage "${GATEWAY_DEPLOY_STAGE:-unknown}" \
    --ledger "$GATEWAY_RESTART_TIMING_FILE" \
    --restart-invocation-id "$GATEWAY_RESTART_INVOCATION_ID" \
    --release-attempts "$GATEWAY_RELEASE_ATTEMPTS_USED" \
    --candidate-sha "$candidate_sha" \
    --evidence "$GATEWAY_OFFLINE_ARTIFACT_PREPARE_LOG" \
    "${shutdown_flag[@]}" >/dev/null 2>&1 || true
  return 0
}

wait_for_gateway_owned_process_group() {
  local process_pid="$1" ready_marker="$2" label="$3"
  local observed_pid=""
  for _ in $(seq 1 100); do
    if [ -s "$ready_marker" ]; then
      IFS= read -r observed_pid < "$ready_marker" || observed_pid=""
      # The child writes this only after setsid.  It may legitimately finish
      # before the parent observes the marker; as an unreaped child its PID
      # cannot be reused before the later wait.
      if [ "$observed_pid" = "$process_pid" ]; then
        return 0
      fi
      break
    fi
    if ! kill -0 "$process_pid" 2>/dev/null; then
      break
    fi
    sleep 0.01
  done
  kill -TERM "$process_pid" 2>/dev/null || true
  wait "$process_pid" 2>/dev/null || true
  rm -f -- "$ready_marker"
  echo "ERROR: $label did not establish its owned process group" >&2
  return 1
}

cancel_gateway_owned_process_group() {
  local process_pid="$1" ready_marker="$2"
  if kill -0 -- "-$process_pid" 2>/dev/null; then
    kill -TERM -- "-$process_pid" 2>/dev/null || true
  else
    # The direct-PID fallback closes the small launch window before setsid.
    kill -TERM "$process_pid" 2>/dev/null || true
  fi
  for _ in $(seq 1 20); do
    if ! kill -0 -- "-$process_pid" 2>/dev/null \
        && ! kill -0 "$process_pid" 2>/dev/null; then
      break
    fi
    sleep 0.1
  done
  kill -KILL -- "-$process_pid" 2>/dev/null || true
  kill -KILL "$process_pid" 2>/dev/null || true
  wait "$process_pid" 2>/dev/null || true
  rm -f -- "$ready_marker"
}

stop_failed_miner_maintenance_runtime() {
  local process_pid=""
  echo "Stopping the newly launched gateway after miner-maintenance verification failure" >&2
  for process_pid in \
      "${GATEWAY_LAUNCHER_PID:-}" \
      "${TEE_EGRESS_FORWARDER_PID:-}" \
      "${INTER_ENCLAVE_RELAY_PID:-}"; do
    if ! [[ "$process_pid" =~ ^[1-9][0-9]*$ ]]; then
      continue
    fi
    kill -TERM -- "-$process_pid" 2>/dev/null || true
    kill -TERM "$process_pid" 2>/dev/null || true
  done
  for _ in $(seq 1 50); do
    local running=0
    for process_pid in \
        "${GATEWAY_LAUNCHER_PID:-}" \
        "${TEE_EGRESS_FORWARDER_PID:-}" \
        "${INTER_ENCLAVE_RELAY_PID:-}"; do
      if [[ "$process_pid" =~ ^[1-9][0-9]*$ ]] \
          && { kill -0 -- "-$process_pid" 2>/dev/null \
            || kill -0 "$process_pid" 2>/dev/null; }; then
        running=1
      fi
    done
    [ "$running" -eq 0 ] && break
    sleep 0.1
  done
  for process_pid in \
      "${GATEWAY_LAUNCHER_PID:-}" \
      "${TEE_EGRESS_FORWARDER_PID:-}" \
      "${INTER_ENCLAVE_RELAY_PID:-}"; do
    if ! [[ "$process_pid" =~ ^[1-9][0-9]*$ ]]; then
      continue
    fi
    kill -KILL -- "-$process_pid" 2>/dev/null || true
    kill -KILL "$process_pid" 2>/dev/null || true
    wait "$process_pid" 2>/dev/null || true
  done
  sudo systemctl stop leadpoet-tee-egress-forwarder.service 2>/dev/null || true
  if [ -r "$GATEWAY_ROOT/tee/stop_enclave.sh" ]; then
    sudo bash "$GATEWAY_ROOT/tee/stop_enclave.sh" >/dev/null 2>&1 || true
  else
    sudo nitro-cli terminate-enclave --all >/dev/null 2>&1 || true
  fi
}

start_gateway_offline_artifact_prepare() {
  local prepare_script process_group_marker
  local -a prepare_command
  prepare_script="$GATEWAY_PREFLIGHT_TREE/gateway/tee/prepare_offline_artifacts_v2.sh"
  if [ -n "$GATEWAY_OFFLINE_ARTIFACT_PREPARE_PID" ]; then
    echo "ERROR: V2 offline artifact preparation is already running" >&2
    return 1
  fi
  if [ ! -r "$prepare_script" ]; then
    echo "ERROR: prepared V2 offline artifact helper is unavailable" >&2
    return 1
  fi
  if ! mkdir -p "$(dirname "$GATEWAY_OFFLINE_ARTIFACT_PREPARE_LOG")" \
      || ! : > "$GATEWAY_OFFLINE_ARTIFACT_PREPARE_LOG" \
      || ! chmod 600 "$GATEWAY_OFFLINE_ARTIFACT_PREPARE_LOG"; then
    echo "ERROR: V2 offline artifact preparation log is unavailable" >&2
    return 1
  fi
  process_group_marker="${GATEWAY_OFFLINE_ARTIFACT_PREPARE_LOG}.process-group"
  rm -f -- "$process_group_marker"

  record_gateway_restart_timing "offline_artifact_prepare_started"
  prepare_command=(bash "$prepare_script")
  if command -v ionice >/dev/null 2>&1; then
    prepare_command=(ionice -c2 -n7 "${prepare_command[@]}")
  fi
  # Own the whole helper process group so an interrupted curl/pip/rsync child
  # cannot outlive the candidate tree.  Keep this release-independent work at
  # low CPU and I/O priority while the attestation runner is building.
  env -u GATEWAY_MINER_MAINTENANCE_PROOF_FD \
    -u GATEWAY_CONTROLLER_PROCESS_HELPER \
    -u LAB_ARENA_PROCESS_HELPER \
    -u GATEWAY_GIT_HELPER \
    -u GATEWAY_HOST_MEMORY_GUARD_PATH \
    -u GATEWAY_RESTART_AUTHORITY_ROOT \
    -u GATEWAY_RESTART_AUTHORITY_COMMIT \
    -u GATEWAY_ACTIVE_RELEASE_RESTART_INVOCATION_ID \
    -u GATEWAY_ACTIVE_RELEASE_COMPONENT \
    python3 -c '
import os
import sys

os.chdir(sys.argv[1])
os.setsid()
os.nice(10)
marker_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
marker = os.open(sys.argv[2], marker_flags, 0o600)
with os.fdopen(marker, "w", encoding="ascii") as handle:
    handle.write(str(os.getpid()) + "\n")
    handle.flush()
    os.fsync(handle.fileno())
os.execvp(sys.argv[3], sys.argv[3:])
' "$GATEWAY_PREFLIGHT_TREE" "$process_group_marker" "${prepare_command[@]}" \
    >"$GATEWAY_OFFLINE_ARTIFACT_PREPARE_LOG" 2>&1 \
    190>&- 191>&- 192>&- 193>&- 194>&- 195>&- &
  GATEWAY_OFFLINE_ARTIFACT_PREPARE_PID="$!"
  if ! wait_for_gateway_owned_process_group \
      "$GATEWAY_OFFLINE_ARTIFACT_PREPARE_PID" \
      "$process_group_marker" \
      "V2 offline artifact preparation"; then
    GATEWAY_OFFLINE_ARTIFACT_PREPARE_PID=""
    return 1
  fi
  echo "Started candidate V2 offline artifact preparation as PID $GATEWAY_OFFLINE_ARTIFACT_PREPARE_PID"
  echo "Offline artifact preparation log: $GATEWAY_OFFLINE_ARTIFACT_PREPARE_LOG"
}

wait_for_gateway_offline_artifact_prepare() {
  local prepare_pid status=0
  prepare_pid="$GATEWAY_OFFLINE_ARTIFACT_PREPARE_PID"
  if [ -z "$prepare_pid" ]; then
    echo "ERROR: V2 offline artifact preparation was not started" >&2
    return 1
  fi
  if wait "$prepare_pid"; then
    status=0
  else
    status="$?"
  fi
  GATEWAY_OFFLINE_ARTIFACT_PREPARE_PID=""
  rm -f -- "${GATEWAY_OFFLINE_ARTIFACT_PREPARE_LOG}.process-group"
  cat "$GATEWAY_OFFLINE_ARTIFACT_PREPARE_LOG"
  if [ "$status" -ne 0 ]; then
    record_gateway_restart_timing "offline_artifact_prepare_complete" "failed"
    echo "ERROR: V2 offline artifact preparation failed before shutdown" >&2
    return "$status"
  fi
  record_gateway_restart_timing "offline_artifact_prepare_complete" "passed"
  return 0
}

cancel_gateway_offline_artifact_prepare() {
  local prepare_pid process_group_marker
  prepare_pid="$GATEWAY_OFFLINE_ARTIFACT_PREPARE_PID"
  [ -n "$prepare_pid" ] || return 0
  process_group_marker="${GATEWAY_OFFLINE_ARTIFACT_PREPARE_LOG}.process-group"
  cancel_gateway_owned_process_group "$prepare_pid" "$process_group_marker"
  GATEWAY_OFFLINE_ARTIFACT_PREPARE_PID=""
}

follow_superseding_gateway_release() {
  local helper latest_sha next_count superseding_tree

  if [ -n "$REQUESTED_GATEWAY_DEPLOY_COMMIT" ]; then
    return 0
  fi
  helper="$GATEWAY_PREFLIGHT_TREE/Leadpoet/utils/restart_release_supersession_v2.py"
  if [ ! -r "$helper" ]; then
    echo "ERROR: forward restart authority helper is unavailable" >&2
    return 1
  fi
  if ! latest_sha="$(
      "$GATEWAY_PYTHON_BIN" "$helper" \
        --repo-root "$LEADPOET_REPO_ROOT" \
        --expected-commit "$PREPARED_GATEWAY_SHA" \
        --branch main
    )"; then
    echo "Forward restart authority is temporarily unreadable; retaining the running gateway" >&2
    return 1
  fi
  if [ "$latest_sha" = "$PREPARED_GATEWAY_SHA" ]; then
    return 0
  fi

  next_count=$((GATEWAY_RELEASE_SUPERSESSION_COUNT + 1))
  if [ "$next_count" -gt "$GATEWAY_RELEASE_SUPERSESSION_MAX" ]; then
    echo "ERROR: gateway release changed too many times during one restart invocation" >&2
    return 1
  fi
  if [ -z "$GATEWAY_RELEASE_FOLLOW_ROOT" ]; then
    GATEWAY_RELEASE_FOLLOW_ROOT="$(
      mktemp -d /tmp/gateway-release-follow.XXXXXX
    )"
    chmod 700 "$GATEWAY_RELEASE_FOLLOW_ROOT"
  fi
  superseding_tree="$GATEWAY_RELEASE_FOLLOW_ROOT/$latest_sha"
  mkdir -p "$superseding_tree"
  if ! git -C "$LEADPOET_REPO_ROOT" archive "$latest_sha" \
      | tar -xf - -C "$superseding_tree"; then
    echo "ERROR: superseding gateway commit could not be materialized" >&2
    return 1
  fi
  if ! grep -Fq 'GATEWAY_GIT_DEPLOY_PROTOCOL="1"' \
      "$superseding_tree/gw_restart.sh" \
      || [ ! -r "$superseding_tree/scripts/gateway_git_deploy.py" ] \
      || [ ! -r "$superseding_tree/Leadpoet/utils/restart_release_supersession_v2.py" ]; then
    echo "ERROR: superseding gateway commit lacks the restart handoff contract" >&2
    return 1
  fi

  echo "Forward gateway release moved from $PREPARED_GATEWAY_SHA to $latest_sha; re-executing before shutdown"
  record_gateway_restart_timing "release_superseded"
  cancel_gateway_offline_artifact_prepare
  rm -rf -- "$GATEWAY_PREFLIGHT_TREE"
  GATEWAY_PREFLIGHT_TREE=""

  LEADPOET_REPO_ROOT="$LEADPOET_REPO_ROOT" \
    GATEWAY_ROOT="$GATEWAY_ROOT" \
    GATEWAY_LOG_ROOT="$GATEWAY_LOG_ROOT" \
    GATEWAY_LOG_FILE="$GATEWAY_LOG_FILE" \
    GATEWAY_ENV_FILE="$GATEWAY_ENV_FILE" \
    LEADPOET_GATEWAY_ENV_SECRET_ID="$LEADPOET_GATEWAY_ENV_SECRET_ID" \
    GATEWAY_PYTHON_BIN="$GATEWAY_PYTHON_BIN" \
    GATEWAY_RESTART_CONTROLLER_ROOT="$GATEWAY_RESTART_CONTROLLER_ROOT" \
    GATEWAY_RESTART_AUTHORITY_ROOT="$GATEWAY_RESTART_AUTHORITY_ROOT" \
    GATEWAY_RESTART_AUTHORITY_COMMIT="$GATEWAY_RESTART_AUTHORITY_COMMIT" \
    GATEWAY_ACTIVE_RELEASE_RESTART_INVOCATION_ID="$GATEWAY_ACTIVE_RELEASE_RESTART_INVOCATION_ID" \
    GATEWAY_ACTIVE_RELEASE_COMPONENT="$GATEWAY_ACTIVE_RELEASE_COMPONENT" \
    GATEWAY_GIT_HELPER="$superseding_tree/scripts/gateway_git_deploy.py" \
    GATEWAY_HOST_MEMORY_GUARD_PATH="$superseding_tree/gateway/tee/host_memory_guard_v2.py" \
    GATEWAY_RESTART_PHASE=prepare \
    GATEWAY_RESTART_LOCK_HELD=1 \
    GATEWAY_RESTART_LOCK_FILE="$GATEWAY_RESTART_LOCK_FILE" \
    GATEWAY_RESTART_RECOVERY_LOCK_FILE="$GATEWAY_RESTART_RECOVERY_LOCK_FILE" \
    GATEWAY_RESTART_STARTED_EPOCH="$GATEWAY_RESTART_STARTED_EPOCH" \
    GATEWAY_RESTART_INVOCATION_ID="${GATEWAY_RESTART_INVOCATION_ID:-gateway-${GATEWAY_RESTART_STARTED_EPOCH:-unknown}-$$}" \
    GATEWAY_RELEASE_ATTEMPTS_USED="${GATEWAY_RELEASE_ATTEMPTS_USED:-0}" \
    GATEWAY_RESTART_TIMING_DIR="$GATEWAY_RESTART_TIMING_DIR" \
    GATEWAY_RESTART_TIMING_FILE="$GATEWAY_RESTART_TIMING_FILE" \
    GATEWAY_RESTART_TIMING_INITIALIZED="$GATEWAY_RESTART_TIMING_INITIALIZED" \
    GATEWAY_RELEASE_FOLLOW_ROOT="$GATEWAY_RELEASE_FOLLOW_ROOT" \
    GATEWAY_RELEASE_SUPERSESSION_COUNT="$next_count" \
    GATEWAY_RELEASE_SUPERSESSION_MAX="$GATEWAY_RELEASE_SUPERSESSION_MAX" \
    GATEWAY_DEPLOY_PLAN_FILE="$GATEWAY_DEPLOY_PLAN_FILE" \
    GATEWAY_DEPLOYMENT_DIR="$GATEWAY_DEPLOYMENT_DIR" \
    GATEWAY_DEPLOYMENT_MANIFEST="$GATEWAY_DEPLOYMENT_MANIFEST" \
    GATEWAY_LAST_GOOD_MANIFEST="$GATEWAY_LAST_GOOD_MANIFEST" \
    GATEWAY_HOST_RESTART_SCRIPT="$GATEWAY_HOST_RESTART_SCRIPT" \
    GATEWAY_TEE_EIF_ROOT="$GATEWAY_TEE_EIF_ROOT" \
    GATEWAY_V2_RELEASE_ARCHIVE_ROOT="$GATEWAY_V2_RELEASE_ARCHIVE_ROOT" \
    GATEWAY_V2_CONFIG_DIR="$GATEWAY_V2_CONFIG_DIR" \
    GATEWAY_V2_RELEASE_MANIFEST="$GATEWAY_V2_RELEASE_MANIFEST" \
    GATEWAY_V2_RELEASE_LINEAGE="$GATEWAY_V2_RELEASE_LINEAGE" \
    GATEWAY_PREPARED_V2_RELEASE_MANIFEST="$GATEWAY_PREPARED_V2_RELEASE_MANIFEST" \
    GATEWAY_PREPARED_V2_RELEASE_LINEAGE="$GATEWAY_PREPARED_V2_RELEASE_LINEAGE" \
    GATEWAY_V2_RELEASE_BUCKET="$GATEWAY_V2_RELEASE_BUCKET" \
    GATEWAY_V2_RELEASE_PREFIX="$GATEWAY_V2_RELEASE_PREFIX" \
    GATEWAY_V2_ARTIFACT_POLICY="$GATEWAY_V2_ARTIFACT_POLICY" \
    RESEARCH_LAB_ATTESTED_V2_ARTIFACT_BUCKET="$RESEARCH_LAB_ATTESTED_V2_ARTIFACT_BUCKET" \
    GATEWAY_V2_OFFLINE_ARTIFACT_ROOT="$GATEWAY_V2_OFFLINE_ARTIFACT_ROOT" \
    GATEWAY_DEPLOY_STAGE=bootstrap \
    exec bash "$superseding_tree/gw_restart.sh"
}

install_gateway_python_dependencies() {
  local dependency_fingerprint legacy_project_metadata pip_scope=() requirements_file
  if [ -n "${GATEWAY_PREFLIGHT_TREE:-}" ] \
      && [ -f "$GATEWAY_PREFLIGHT_TREE/requirements.txt" ]; then
    requirements_file="$GATEWAY_PREFLIGHT_TREE/requirements.txt"
  else
    requirements_file="$LEADPOET_REPO_ROOT/requirements.txt"
  fi
  if [ ! -r "$requirements_file" ]; then
    echo "ERROR: exact gateway requirements are unavailable: $requirements_file" >&2
    return 1
  fi
  if ! "$GATEWAY_PYTHON_BIN" -m pip --version >/dev/null 2>&1; then
    curl -fsS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
    "$GATEWAY_PYTHON_BIN" /tmp/get-pip.py
    rm -f /tmp/get-pip.py
  fi
  if ! "$GATEWAY_PYTHON_BIN" -c \
      'import sys; raise SystemExit(0 if sys.prefix != sys.base_prefix else 1)'; then
    pip_scope=(--user)
  fi
  dependency_fingerprint="$(
    "$GATEWAY_PYTHON_BIN" - "$requirements_file" \
      "${GATEWAY_HOST_EXTRA_PYTHON_PACKAGES[@]}" <<'PY'
import hashlib
from pathlib import Path
import sys

requirements = Path(sys.argv[1]).read_bytes()
identity = "\0".join(
    (
        str(Path(sys.executable).resolve()),
        sys.version,
        *sys.argv[2:],
    )
).encode("utf-8")
print(hashlib.sha256(requirements + b"\0" + identity).hexdigest())
PY
  )"
  export PATH="$HOME/.local/bin:$PATH"
  if [ "$GATEWAY_DEPENDENCY_INSTALL_FINGERPRINT" = "$dependency_fingerprint" ]; then
    echo "Reusing exact candidate dependency installation from pre-shutdown validation"
  else
    # These legacy distributions install the same `scalecodec` import
    # namespace as Cyscale and make Bittensor 10 fail at import time.
    "$GATEWAY_PYTHON_BIN" -m pip uninstall -y \
      leadpoet-subnet substrate-interface py-scale-codec scalecodec \
      >/dev/null 2>&1 || true
    "$GATEWAY_PYTHON_BIN" -m pip install \
      "${pip_scope[@]}" \
      --requirement "$requirements_file" \
      "${GATEWAY_HOST_EXTRA_PYTHON_PACKAGES[@]}"
    GATEWAY_DEPENDENCY_INSTALL_FINGERPRINT="$dependency_fingerprint"
    export GATEWAY_DEPENDENCY_INSTALL_FINGERPRINT
  fi
  legacy_project_metadata="$LEADPOET_REPO_ROOT/leadpoet_subnet.egg-info"
  if [ -d "$legacy_project_metadata" ]; then
    echo "Removing generated legacy project metadata before dependency validation"
    rm -rf -- "$legacy_project_metadata"
  fi
  "$GATEWAY_PYTHON_BIN" -m pip check
}

select_gateway_python_runtime() {
  local configured resolved version
  configured="$(
    set -a
    . "$ENV_CLONE"
    set +a
    printf '%s' "${GATEWAY_PYTHON_BIN:-/home/ec2-user/venv311/bin/python3}"
  )"
  if [[ "$configured" = /* ]]; then
    resolved="$configured"
  else
    resolved="$(command -v "$configured" || true)"
  fi
  if [ -z "$resolved" ] || [ ! -x "$resolved" ]; then
    echo "ERROR: configured gateway Python is unavailable: $configured" >&2
    return 1
  fi
  version="$(
    "$resolved" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'
  )"
  if ! "$resolved" -c \
      'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)'; then
    echo "ERROR: gateway V2 requires Python 3.11 or newer; observed $version" >&2
    return 1
  fi
  GATEWAY_PYTHON_BIN="$resolved"
  export GATEWAY_PYTHON_BIN
  printf 'export GATEWAY_PYTHON_BIN=%q\n' "$GATEWAY_PYTHON_BIN" >> "$ENV_CLONE"
  echo "Gateway Python runtime: $GATEWAY_PYTHON_BIN ($version)"
}

report_gateway_v2_bootstrap_pending() {
  local missing=() path
  for path in \
    "$GATEWAY_PREPARED_V2_RELEASE_MANIFEST" \
    "$GATEWAY_V2_ARTIFACT_POLICY" \
    "${V2_CREDENTIAL_ENVELOPES[@]}"; do
    [ -e "$path" ] || missing+=("$path")
  done
  if [ "${#missing[@]}" -eq 0 ]; then
    return 1
  fi
  python3 - "${missing[@]}" <<'PY'
import json
import sys
print(json.dumps({
    "schema_version": "leadpoet.gateway_v2_first_activation.v1",
    "status": "bootstrap_pending",
    "production_shutdown_started": False,
    "missing_paths": sys.argv[1:],
    "required_external_approvals": [],
}, sort_keys=True, indent=2))
PY
  echo "Gateway remains untouched. Complete the V2 bootstrap ceremony, then rerun this restart." >&2
  return 0
}

deployment_field() {
  python3 "$GATEWAY_GIT_HELPER" field \
    --plan-file "$GATEWAY_DEPLOY_PLAN_FILE" \
    --name "$1"
}

finalize_deployment_record() {
  local status="$1"
  local stage="$2"
  python3 "$GATEWAY_GIT_HELPER" finalize \
    --plan-file "$GATEWAY_DEPLOY_PLAN_FILE" \
    --status "$status" \
    --stage "$stage" \
    --eif-root "$GATEWAY_TEE_EIF_ROOT"
}

cleanup_gateway_miner_maintenance_bootstrap() {
  if [ -n "${GATEWAY_MINER_MAINTENANCE_HANDOFF_FILE:-}" ] \
      && [[ "$GATEWAY_MINER_MAINTENANCE_HANDOFF_FILE" =~ ^/tmp/leadpoet-gateway-miner-maintenance-handoff\.[A-Za-z0-9._-]+$ ]]; then
    rm -f -- "$GATEWAY_MINER_MAINTENANCE_HANDOFF_FILE" 2>/dev/null || true
  fi
  if [ -n "${GATEWAY_MINER_MAINTENANCE_BOOTSTRAP_ROOT:-}" ] \
      && [[ "$GATEWAY_MINER_MAINTENANCE_BOOTSTRAP_ROOT" =~ ^/tmp/gateway-miner-maintenance-bootstrap\.[A-Za-z0-9]+$ ]]; then
    rm -rf -- "$GATEWAY_MINER_MAINTENANCE_BOOTSTRAP_ROOT" 2>/dev/null || true
  fi
}

scrub_gateway_bootstrap_aws_environment() {
  unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN \
    AWS_SECURITY_TOKEN AWS_PROFILE AWS_DEFAULT_PROFILE \
    AWS_SHARED_CREDENTIALS_FILE AWS_WEB_IDENTITY_TOKEN_FILE AWS_ROLE_ARN \
    AWS_ROLE_SESSION_NAME AWS_CONTAINER_CREDENTIALS_FULL_URI \
    AWS_CONTAINER_CREDENTIALS_RELATIVE_URI AWS_CONFIG_FILE AWS_CA_BUNDLE \
    AWS_ENDPOINT_URL AWS_ENDPOINT_URL_S3 AWS_ENDPOINT_URL_STS \
    AWS_ENDPOINT_URL_SECRETSMANAGER AWS_EC2_METADATA_SERVICE_ENDPOINT \
    AWS_EC2_METADATA_SERVICE_ENDPOINT_MODE AWS_METADATA_SERVICE_TIMEOUT \
    AWS_METADATA_SERVICE_NUM_ATTEMPTS BOTO_CONFIG HTTP_PROXY HTTPS_PROXY \
    ALL_PROXY http_proxy https_proxy all_proxy
  export AWS_REGION=us-east-1
  export AWS_DEFAULT_REGION=us-east-1
}

validate_gateway_aws_authority() {
  local name value
  for name in \
    AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN \
    AWS_SECURITY_TOKEN AWS_PROFILE AWS_DEFAULT_PROFILE \
    AWS_SHARED_CREDENTIALS_FILE AWS_WEB_IDENTITY_TOKEN_FILE AWS_ROLE_ARN \
    AWS_ROLE_SESSION_NAME AWS_CONTAINER_CREDENTIALS_FULL_URI \
    AWS_CONTAINER_CREDENTIALS_RELATIVE_URI AWS_CONFIG_FILE AWS_CA_BUNDLE \
    AWS_ENDPOINT_URL AWS_ENDPOINT_URL_S3 AWS_ENDPOINT_URL_STS \
    AWS_ENDPOINT_URL_SECRETSMANAGER AWS_EC2_METADATA_SERVICE_ENDPOINT \
    AWS_EC2_METADATA_SERVICE_ENDPOINT_MODE AWS_METADATA_SERVICE_TIMEOUT \
    AWS_METADATA_SERVICE_NUM_ATTEMPTS BOTO_CONFIG HTTP_PROXY HTTPS_PROXY \
    ALL_PROXY http_proxy https_proxy all_proxy; do
    value="${!name-}"
    if [ -n "$value" ]; then
      echo "ERROR: gateway restart inherited delegated AWS authority: $name" >&2
      return 1
    fi
  done
  if { [ -n "${AWS_REGION:-}" ] && [ "$AWS_REGION" != "us-east-1" ]; } \
      || { [ -n "${AWS_DEFAULT_REGION:-}" ] \
        && [ "$AWS_DEFAULT_REGION" != "us-east-1" ]; }; then
    echo "ERROR: gateway restart AWS region differs from us-east-1" >&2
    return 1
  fi
  if [ -n "${LEADPOET_AWS_INSTANCE_ROLE_ONLY:-}" ] \
      && [ "${LEADPOET_AWS_INSTANCE_ROLE_ONLY,,}" != "true" ]; then
    echo "ERROR: gateway restart instance-role-only authority differs" >&2
    return 1
  fi
  scrub_gateway_bootstrap_aws_environment
  export LEADPOET_AWS_INSTANCE_ROLE_ONLY=true
}

on_gateway_restart_exit() {
  local status="$?"
  if [ "$status" -ne 0 ]; then
    abort_lab_arena_restart_guard_before_destructive
  fi
  if [ "$status" -ne 0 ]; then
    record_gateway_restart_timing "${GATEWAY_DEPLOY_STAGE:-unknown}" "failed" \
      >/dev/null 2>&1 || true
  fi
  emit_gateway_restart_sentry_summary "$status"
  cancel_gateway_offline_artifact_prepare
  cleanup_gateway_miner_maintenance_bootstrap
  rm -f -- \
    "$GATEWAY_PREPARED_V2_RELEASE_MANIFEST" \
    "$GATEWAY_PREPARED_V2_RELEASE_LINEAGE" \
    2>/dev/null || true
  if [ -n "${GATEWAY_PREFLIGHT_TREE:-}" ]; then
    rm -rf "$GATEWAY_PREFLIGHT_TREE"
  fi
  if [ -n "${GATEWAY_RELEASE_FOLLOW_ROOT:-}" ]; then
    rm -rf "$GATEWAY_RELEASE_FOLLOW_ROOT"
  fi
  if [ "$status" -ne 0 ] \
      && [ "$GATEWAY_DEPLOY_COMPLETED" != "1" ] \
      && [ -f "$GATEWAY_DEPLOY_PLAN_FILE" ] \
      && [ -f "$GATEWAY_GIT_HELPER" ]; then
    finalize_deployment_record failed "$GATEWAY_DEPLOY_STAGE" >/dev/null 2>&1 || true
  fi
}

gateway_memory_ready_after_running_gateway_shutdown() {
  "$GATEWAY_PYTHON_BIN" - \
    "$1" "$2" "$LEADPOET_REPO_ROOT" "$GATEWAY_PYTHON_BIN" \
    "$GATEWAY_RECLAIMABLE_MEMORY_SAFETY_MARGIN_MIB" "${3:-/proc}" <<'PY'
import json
import os
from pathlib import Path
import sys

report_path = Path(sys.argv[1])
pid_text = sys.argv[2]
repo_root = Path(sys.argv[3]).resolve()
python_bin = Path(sys.argv[4]).resolve()
safety_margin_mib = int(sys.argv[5])
proc_root = Path(sys.argv[6])
if not pid_text.isdigit() or int(pid_text) <= 1:
    raise SystemExit("running gateway PID is invalid")

report = json.loads(report_path.read_text(encoding="utf-8"))
if (
    not isinstance(report, dict)
    or report.get("schema_version") != "leadpoet.gateway_host_memory_guard.v2"
    or report.get("status") != "blocked"
    or report.get("minimum_available_memory_mib") != 16384
):
    raise SystemExit("blocked gateway memory report is invalid")
available_mib = report.get("available_memory_mib")
if (
    not isinstance(available_mib, int)
    or isinstance(available_mib, bool)
    or available_mib < 0
):
    raise SystemExit("available gateway memory is invalid")

process_root = proc_root / pid_text


def read_process(selected_process_root):
    status = (selected_process_root / "status").read_text(encoding="utf-8")
    stat_fields = (selected_process_root / "stat").read_text(encoding="utf-8").split()
    argv = tuple(
        value.decode("utf-8", errors="strict")
        for value in (selected_process_root / "cmdline").read_bytes().split(b"\0")
        if value
    )
    cwd = Path(os.readlink(selected_process_root / "cwd")).resolve()
    fields = {}
    for line in status.splitlines():
        key, separator, value = line.partition(":")
        if separator:
            fields[key] = value.strip()
    return {
        "ppid": int(stat_fields[3]),
        "start_ticks": int(stat_fields[21]),
        "uid": int(fields["Uid"].split()[0]),
        "rss_kib": int(fields["VmRSS"].split()[0]),
        "argv": argv,
        "cwd": cwd,
    }


try:
    first = read_process(process_root)
    second = read_process(process_root)
except (IndexError, KeyError, OSError, UnicodeError, ValueError) as exc:
    raise SystemExit("running gateway process identity is unavailable") from exc
if first != second:
    raise SystemExit("running gateway process identity changed")
if first["uid"] != os.getuid() or not first["argv"]:
    raise SystemExit("running gateway process owner is invalid")
if Path(first["argv"][0]).resolve() != python_bin:
    raise SystemExit("running gateway interpreter differs")
suffix = first["argv"][1:]
if suffix == ("-u", "-m", "gateway.main"):
    allowed_cwds = {repo_root, repo_root / "gateway"}
elif suffix == ("-u", "main.py"):
    allowed_cwds = {repo_root / "gateway"}
else:
    raise SystemExit("running gateway command differs")
if first["cwd"] not in allowed_cwds:
    raise SystemExit("running gateway working directory differs")

worker_script = (repo_root / "gateway" / "research_lab" / "worker_process.py").resolve()
worker_rss_kib = 0
worker_count = 0
for child_root in proc_root.iterdir():
    if not child_root.name.isdigit() or child_root.name == pid_text:
        continue
    try:
        child_first = read_process(child_root)
        child_second = read_process(child_root)
    except (IndexError, KeyError, OSError, UnicodeError, ValueError):
        continue
    if child_first != child_second or child_first["ppid"] != int(pid_text):
        continue
    child_argv = child_first["argv"]
    if (
        child_first["uid"] != os.getuid()
        or child_first["cwd"] != repo_root
        or len(child_argv) != 12
        or Path(child_argv[0]).resolve() != python_bin
        or Path(child_argv[1]).resolve() != worker_script
        or child_argv[2] != "--kind"
        or child_argv[3] not in {"hosted", "scoring"}
        or child_argv[4] != "--worker-index"
        or not child_argv[5].isdigit()
        or child_argv[6] != "--total-workers"
        or not child_argv[7].isdigit()
        or child_argv[8] != "--worker-prefix"
        or not child_argv[9]
        or child_argv[10] != "--log-level"
        or child_argv[11] != "INFO"
    ):
        continue
    worker_rss_kib += child_first["rss_kib"]
    worker_count += 1

reclaimable_gateway_mib = first["rss_kib"] // 1024
reclaimable_worker_mib = worker_rss_kib // 1024
reclaimable_mib = reclaimable_gateway_mib + reclaimable_worker_mib
required_mib = int(report["minimum_available_memory_mib"])
if available_mib + reclaimable_mib < required_mib + safety_margin_mib:
    raise SystemExit("gateway shutdown would not recover enough build memory")
print(
    json.dumps(
        {
            "available_memory_mib": available_mib,
            "minimum_available_memory_mib": required_mib,
            "reclaimable_gateway_memory_mib": reclaimable_mib,
            "reclaimable_gateway_parent_memory_mib": reclaimable_gateway_mib,
            "reclaimable_gateway_worker_count": worker_count,
            "reclaimable_gateway_worker_memory_mib": reclaimable_worker_mib,
            "safety_margin_mib": safety_margin_mib,
            "schema_version": "leadpoet.gateway_reclaimable_memory.v1",
            "status": "ready_after_gateway_shutdown",
        },
        sort_keys=True,
    )
)
PY
}

wait_for_gateway_build_memory() {
  local allow_running_gateway_reclaim="${1:-0}"
  local max_attempts="${2:-300}"
  local guard="$GATEWAY_HOST_MEMORY_GUARD_PATH"
  local report
  if ! [[ "$max_attempts" =~ ^[1-9][0-9]*$ ]] \
      || { [ "$allow_running_gateway_reclaim" != "0" ] \
        && [ "$allow_running_gateway_reclaim" != "1" ]; }; then
    echo "ERROR: gateway memory wait configuration is invalid" >&2
    return 1
  fi
  if [ ! -r "$guard" ]; then
    echo "ERROR: gateway host memory guard is unavailable: $guard" >&2
    return 1
  fi
  report="$(mktemp /tmp/gateway-memory-ready.XXXXXX.json)"
  for attempt in $(seq 1 "$max_attempts"); do
    if python3 "$guard" \
        --cleanup-disposable-tests \
        --cleanup-stale-vsock-probes \
        --minimum-available-mib 16384 >"$report"; then
      cat "$report"
      rm -f "$report"
      return 0
    fi
    if [ "$allow_running_gateway_reclaim" = "1" ] \
        && gateway_memory_ready_after_running_gateway_shutdown \
          "$report" "${PID:-}"; then
      rm -f "$report"
      return 0
    fi
    if [ "$attempt" -eq 1 ] || [ $((attempt % 10)) -eq 0 ]; then
      echo "Waiting for 16 GiB available memory (${attempt}/${max_attempts})"
      cat "$report"
    fi
    sleep 6
  done
  echo "ERROR: gateway build memory did not recover within the bounded wait" >&2
  cat "$report" >&2
  rm -f "$report"
  return 1
}

run_prepared_gateway_module() {
  (
    cd "$GATEWAY_PREFLIGHT_TREE"
    PYTHONPATH="$GATEWAY_PREFLIGHT_TREE" "$GATEWAY_PYTHON_BIN" -m "$@"
  )
}

validate_runtime_secret_paths() {
  local key value
  for key in GATEWAY_PRIVATE_KEY_PATH ARWEAVE_KEYFILE_PATH; do
    value="${!key:-}"
    if [ -z "$value" ] && [ "$key" = "GATEWAY_PRIVATE_KEY_PATH" ]; then
      value="$GATEWAY_LOG_ROOT/secrets/gateway_private_key.pem"
    elif [ -z "$value" ]; then
      value="$GATEWAY_LOG_ROOT/secrets/arweave_keyfile.json"
    fi
    printf -v "$key" '%s' "$value"
    export "$key"
    if [[ "$value" != /* ]]; then
      echo "ERROR: $key must be configured as an absolute path for Git-checkout deployment" >&2
      return 1
    fi
    if [ ! -f "$value" ]; then
      echo "ERROR: configured $key file does not exist" >&2
      return 1
    fi
  done
}

enforce_deployment_environment() {
  unset BUILD_ID BUILD_TIME_UTC BUILD_TIMESTAMP GITHUB_TAG GIT_TAG
  export LEADPOET_REPO_ROOT GATEWAY_ROOT GATEWAY_LOG_ROOT GATEWAY_LOG_FILE
  export GATEWAY_TEE_EIF_ROOT GATEWAY_V2_RELEASE_ARCHIVE_ROOT
  export GATEWAY_RESTART_TEMP_CLEANUP_MIN_AGE_SECONDS
  export GATEWAY_RESTART_EMERGENCY_BACKUP_MIN_AGE_SECONDS
  export GATEWAY_RESTART_CLEANUP_MAX_CANDIDATES
  export GATEWAY_MIGRATION_203_SQL_SHA256 GATEWAY_MIGRATION_203_BARRIER_FILE
  export GATEWAY_MIGRATION_203_COMPLETION_FILE GATEWAY_MIGRATION_203_WAIT_SECONDS
  export RESEARCH_LAB_TEE_PROTOCOL
  export GATEWAY_V2_CONFIG_DIR GATEWAY_V2_RELEASE_MANIFEST GATEWAY_V2_RELEASE_LINEAGE
  export GATEWAY_V2_ARTIFACT_POLICY
  export RESEARCH_LAB_ATTESTED_V2_ARTIFACT_BUCKET
  export LEADPOET_LOCAL_RELEASE_COMMIT_SHA LEADPOET_LOCAL_GATEWAY_RELEASE
  export GATEWAY_TEE_FALLBACK_LOG_DIR="$GATEWAY_LOG_ROOT/gateway/logs/tee_fallback"
  export PYTHONPATH="$LEADPOET_REPO_ROOT"
  export GITHUB_SHA="$GATEWAY_DEPLOY_SHA"
  export GITHUB_COMMIT="$GATEWAY_DEPLOY_SHA"
  export GITHUB_REF_NAME="$GATEWAY_DEPLOY_BRANCH"
  export GIT_BRANCH="$GATEWAY_DEPLOY_BRANCH"
  export GITHUB_BRANCH="$GATEWAY_DEPLOY_BRANCH"
  export GITHUB_REPO_URL="$GATEWAY_DEPLOY_REMOTE"
  export GATEWAY_BUILD_INFO_GIT_ROOT="$LEADPOET_REPO_ROOT"
  export GATEWAY_BUILD_INFO_FILE="$GATEWAY_ROOT/BUILD_INFO.json"
  export RESEARCH_LAB_RUNTIME_SOURCE_ROOT="$LEADPOET_REPO_ROOT"
  export RESEARCH_LAB_DEV_SNAPSHOT_AUTO_REFRESH_ENABLED="${RESEARCH_LAB_DEV_SNAPSHOT_AUTO_REFRESH_ENABLED:-true}"
  export RESEARCH_LAB_DEV_SNAPSHOT_RECORD_ENABLED="${RESEARCH_LAB_DEV_SNAPSHOT_RECORD_ENABLED:-true}"
  export RESEARCH_LAB_DEV_SNAPSHOT_KMS_KEY_ID="${RESEARCH_LAB_DEV_SNAPSHOT_KMS_KEY_ID:-alias/leadpoet-research-lab-artifact-signing}"
  export ATTESTED_RUNTIME_COMMIT_SHA="$GATEWAY_DEPLOY_SHA"
  export ATTESTED_RUNTIME_GIT_REPO_URL="$GATEWAY_DEPLOY_REMOTE"
}

install_successful_restart_script() {
  local controller_sha release_dir temporary_dir temporary_link
  local controller_source_root source_script
  local target_script="$GATEWAY_HOST_RESTART_SCRIPT"
  local target_dir temporary
  controller_sha="${GATEWAY_RESTART_AUTHORITY_COMMIT:-$GATEWAY_DEPLOY_SHA}"
  controller_source_root="${GATEWAY_RESTART_AUTHORITY_ROOT:-$LEADPOET_REPO_ROOT}"
  mkdir -p "$GATEWAY_RESTART_CONTROLLER_ROOT/releases"
  chmod 700 \
    "$(dirname "$GATEWAY_RESTART_CONTROLLER_ROOT")" \
    "$GATEWAY_RESTART_CONTROLLER_ROOT" \
    "$GATEWAY_RESTART_CONTROLLER_ROOT/releases"
  release_dir="$GATEWAY_RESTART_CONTROLLER_ROOT/releases/$controller_sha"
  temporary_dir="$(mktemp -d "$GATEWAY_RESTART_CONTROLLER_ROOT/.release.XXXXXX")"
  mkdir -p \
    "$temporary_dir/Leadpoet/utils" \
    "$temporary_dir/gateway/tee" \
    "$temporary_dir/scripts"
  install -m 700 "$controller_source_root/gw_restart.sh" \
    "$temporary_dir/gw_restart.sh"
  install -m 600 "$controller_source_root/scripts/gateway_git_deploy.py" \
    "$temporary_dir/scripts/gateway_git_deploy.py"
  install -m 600 "$controller_source_root/gateway/tee/host_memory_guard_v2.py" \
    "$temporary_dir/gateway/tee/host_memory_guard_v2.py"
  install -m 600 "$controller_source_root/scripts/manage_owned_process_group.py" \
    "$temporary_dir/scripts/manage_owned_process_group.py"
  if [ -e "$release_dir" ] || [ -L "$release_dir" ]; then
    if [ ! -d "$release_dir" ] \
        || [ -L "$release_dir" ] \
        || [ "$(stat -c '%u:%g:%a' "$release_dir")" != "$(id -u):$(id -g):700" ] \
        || [ "$(stat -c '%u:%g:%a' "$release_dir/gw_restart.sh")" != "$(id -u):$(id -g):700" ] \
        || [ "$(stat -c '%u:%g:%a' "$release_dir/scripts/gateway_git_deploy.py")" != "$(id -u):$(id -g):600" ] \
        || [ "$(stat -c '%u:%g:%a' "$release_dir/gateway/tee/host_memory_guard_v2.py")" != "$(id -u):$(id -g):600" ] \
        || ! cmp -s "$temporary_dir/gw_restart.sh" "$release_dir/gw_restart.sh" \
        || ! cmp -s "$temporary_dir/scripts/gateway_git_deploy.py" "$release_dir/scripts/gateway_git_deploy.py" \
        || ! cmp -s "$temporary_dir/gateway/tee/host_memory_guard_v2.py" "$release_dir/gateway/tee/host_memory_guard_v2.py"; then
      rm -rf -- "$temporary_dir"
      echo "ERROR: installed gateway restart controller release differs from the exact candidate" >&2
      return 1
    fi
    if [ -e "$release_dir/scripts/manage_owned_process_group.py" ] \
        || [ -L "$release_dir/scripts/manage_owned_process_group.py" ]; then
      if [ -L "$release_dir/scripts/manage_owned_process_group.py" ] \
          || [ "$(stat -c '%u:%g:%a' "$release_dir/scripts/manage_owned_process_group.py")" != "$(id -u):$(id -g):600" ] \
          || ! cmp -s "$temporary_dir/scripts/manage_owned_process_group.py" \
              "$release_dir/scripts/manage_owned_process_group.py"; then
        rm -rf -- "$temporary_dir"
        echo "ERROR: installed gateway restart controller process helper differs from the exact candidate" >&2
        return 1
      fi
    elif ! git -C "$LEADPOET_REPO_ROOT" merge-base --is-ancestor \
        "$controller_sha" "$LEGACY_FOUR_FILE_CONTROLLER_BOUNDARY" \
        || ! git -C "$LEADPOET_REPO_ROOT" cat-file -e \
        "$controller_sha:scripts/manage_owned_process_group.py" \
        || ! git -C "$LEADPOET_REPO_ROOT" show \
        "$controller_sha:scripts/manage_owned_process_group.py" \
        | cmp -s - "$temporary_dir/scripts/manage_owned_process_group.py"; then
      rm -rf -- "$temporary_dir"
      echo "ERROR: installed gateway restart controller lacks a bounded exact process helper transition" >&2
      return 1
    fi
    rm -rf -- "$temporary_dir"
  else
    chmod 700 "$temporary_dir"
    mv -- "$temporary_dir" "$release_dir"
  fi
  if [ "$(readlink "$GATEWAY_RESTART_CONTROLLER_CURRENT" 2>/dev/null || true)" != \
      "releases/$controller_sha" ]; then
    temporary_link="$GATEWAY_RESTART_CONTROLLER_ROOT/.current.$$"
    rm -f -- "$temporary_link"
    ln -s "releases/$controller_sha" "$temporary_link"
    mv -Tf "$temporary_link" "$GATEWAY_RESTART_CONTROLLER_CURRENT"
  fi
  source_script="$GATEWAY_RESTART_CONTROLLER_CURRENT/gw_restart.sh"
  if [ "$(cd "$(dirname "$source_script")" && pwd)/$(basename "$source_script")" = \
      "$(cd "$(dirname "$target_script")" && pwd)/$(basename "$target_script")" ]; then
    return 0
  fi
  target_dir="$(dirname "$target_script")"
  mkdir -p "$target_dir"
  if [ -f "$target_script" ] \
      && [ ! -L "$target_script" ] \
      && [ "$(stat -c '%u:%g:%a' "$target_script")" = "$(id -u):$(id -g):700" ] \
      && cmp -s "$source_script" "$target_script"; then
    return 0
  fi
  temporary="$(mktemp "$target_dir/.gw_restart.sh.XXXXXX")"
  if ! install -m 700 "$source_script" "$temporary"; then
    rm -f "$temporary"
    return 1
  fi
  if ! mv -f "$temporary" "$target_script"; then
    rm -f "$temporary"
    return 1
  fi
}

root_free_kb() {
  df --output=avail / | tail -1 | tr -d ' '
}

docker_storage_counts() {
  OVERLAY_DIRS="$(sudo find /var/lib/docker/overlay2 -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l | tr -d ' ')"
  OVERLAY_KB="$(sudo du -sxk /var/lib/docker/overlay2 2>/dev/null | awk '{print $1}' || true)"
  OVERLAY_KB="${OVERLAY_KB:-0}"
  DOCKER_ROOT_KB="$(sudo du -sxk /var/lib/docker 2>/dev/null | awk '{print $1}' || true)"
  DOCKER_ROOT_KB="${DOCKER_ROOT_KB:-0}"
  IMAGE_COUNT="$(sudo docker images -q 2>/dev/null | wc -l | tr -d ' ')"
  CONTAINER_COUNT="$(sudo docker ps -aq 2>/dev/null | wc -l | tr -d ' ')"
  VOLUME_COUNT="$(sudo docker volume ls -q 2>/dev/null | wc -l | tr -d ' ')"
}

reset_orphaned_docker_storage_if_needed() {
  local free_kb_after_prune="$1"
  local reason="${2:-orphaned Docker storage}"
  local reclaim_script
  docker_storage_counts

  if [ "${IMAGE_COUNT:-0}" -eq 0 ] \
     && [ "${CONTAINER_COUNT:-0}" -eq 0 ] \
     && [ "${VOLUME_COUNT:-0}" -eq 0 ] \
     && { [ "${free_kb_after_prune:-0}" -lt "$MIN_FREE_KB" ] || [ "${DOCKER_ROOT_KB:-0}" -gt 1024 ] || [ "${OVERLAY_DIRS:-0}" -gt 0 ]; }; then
    echo "Detected ${reason} with no tracked Docker objects; resetting full Docker data root"
    echo "docker root usage: ${DOCKER_ROOT_KB:-0} KiB; overlay usage: ${OVERLAY_KB:-0} KiB across ${OVERLAY_DIRS:-0} dirs"
    reclaim_script="$LEADPOET_REPO_ROOT/validator_tee/scripts/reclaim_docker_storage_v2.sh"
    if [ ! -r "$reclaim_script" ]; then
      echo "ERROR: guarded Docker storage reclaim helper is unavailable" >&2
      return 1
    fi
    VALIDATOR_DOCKER_MIN_FREE_BYTES=$((MIN_FREE_KB * 1024)) \
      VALIDATOR_DOCKER_LIVE_RUNTIME_MIN_FREE_BYTES="${GATEWAY_DOCKER_LIVE_RUNTIME_MIN_FREE_BYTES:-18000000000}" \
      VALIDATOR_DOCKER_ALLOW_DATA_ROOT_RESET=1 \
      bash "$reclaim_script"
  fi
}

ensure_docker_ready() {
  if sudo docker info >/dev/null 2>&1; then
    return 0
  fi
  echo "Starting Docker and waiting for the daemon"
  sudo systemctl start docker
  for _ in $(seq 1 30); do
    if sudo docker info >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  echo "ERROR: Docker did not become ready" >&2
  return 1
}

foreign_docker_build_processes() {
  python3 - "$LEADPOET_REPO_ROOT" <<'PY'
import os
import sys
from pathlib import Path

repo_root = Path(sys.argv[1]).resolve()
for entry in Path("/proc").iterdir():
    if not entry.name.isdigit():
        continue
    try:
        argv = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        cwd = (entry / "cwd").resolve()
    except (OSError, UnicodeDecodeError):
        continue
    is_build = (
        "docker build" in argv
        or "docker-buildx" in argv
        or "pip install --no-cache-dir -r requirements.txt" in argv
    )
    if is_build and cwd != repo_root and repo_root not in cwd.parents:
        print(f"{entry.name}\t{cwd}\t{argv}")
PY
}

wait_for_foreign_docker_builds() {
  local active
  for attempt in $(seq 1 300); do
    active="$(foreign_docker_build_processes)"
    if [ -z "$active" ]; then
      return 0
    fi
    if [ "$attempt" -eq 1 ]; then
      echo "Waiting for co-located foreign Docker builds before production shutdown"
      printf '%s\n' "$active"
    fi
    sleep 6
  done
  echo "ERROR: foreign Docker builds remained active for 30 minutes" >&2
  return 1
}

stop_local_stale_build_processes() {
  local signal="$1"
  python3 - "$LEADPOET_REPO_ROOT" "$signal" <<'PY'
import os
import signal
import sys
from pathlib import Path

repo_root = Path(sys.argv[1]).resolve()
requested_signal = getattr(signal, "SIG" + sys.argv[2])
for entry in Path("/proc").iterdir():
    if not entry.name.isdigit():
        continue
    try:
        argv = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        cwd = (entry / "cwd").resolve()
    except (OSError, UnicodeDecodeError):
        continue
    is_stale_build = (
        "docker build -f " in argv
        and "/validator_models/containerizing/Dockerfile" in argv
    ) or "pip install --no-cache-dir -r requirements.txt" in argv
    if is_stale_build and (cwd == repo_root or repo_root in cwd.parents):
        try:
            os.kill(int(entry.name), requested_signal)
        except (ProcessLookupError, PermissionError):
            pass
PY
}

run_bounded_restart_artifact_cleanup() {
  local gateway_build_root validator_build_root
  gateway_build_root="${GATEWAY_V2_BUILD_WORK_ROOT:-$HOME/.cache/leadpoet/gateway-release-build-v2}"
  validator_build_root="${VALIDATOR_V2_BUILD_WORK_ROOT:-$HOME/.cache/leadpoet/validator-pcr0-normalizer-v2}"
  if [ ! -r "$LEADPOET_REPO_ROOT/validator_tee/host/restart_artifact_cleanup_v2.py" ]; then
    echo "WARNING: bounded restart artifact cleanup helper is unavailable" >&2
    return 0
  fi
  if ! sudo env PYTHONPATH="$LEADPOET_REPO_ROOT" "$GATEWAY_PYTHON_BIN" \
      -m validator_tee.host.restart_artifact_cleanup_v2 \
      --apply \
      --temporary-root /tmp \
      --gateway-build-root "$gateway_build_root" \
      --validator-build-root "$validator_build_root" \
      --gateway-eif-root "$GATEWAY_TEE_EIF_ROOT" \
      --emergency-backup-root "$HOME/.config/leadpoet" \
      --gateway-archive-root "$GATEWAY_V2_RELEASE_ARCHIVE_ROOT" \
      --gateway-last-good-manifest "$GATEWAY_LAST_GOOD_MANIFEST" \
      --docker-lock-file "$LEADPOET_DOCKER_OPERATION_LOCK_FILE" \
      --docker-lock-owner-pid "${LEADPOET_DOCKER_OPERATION_LOCK_OWNER_PID:-$$}" \
      --temp-min-age-seconds "$GATEWAY_RESTART_TEMP_CLEANUP_MIN_AGE_SECONDS" \
      --emergency-min-age-seconds "$GATEWAY_RESTART_EMERGENCY_BACKUP_MIN_AGE_SECONDS" \
      --max-candidates "$GATEWAY_RESTART_CLEANUP_MAX_CANDIDATES" \
      --allowed-owner-uid "$(id -u)"; then
    echo "WARNING: bounded restart artifact cleanup failed closed" >&2
  fi
}

emergency_disk_preflight() {
  local free_kb emergency_lock_helper
  free_kb="$(root_free_kb)"
  if [ "${free_kb:-0}" -ge "$MIN_FREE_KB" ]; then
    return 0
  fi

  echo "Low disk before env hydration: $(df -h / | tail -1)"
  echo "Running emergency cleanup so restart can reach the full Docker cleanup path"
  mkdir -p "$ENV_BACKUP_DIR" 2>/dev/null || true
  rm -f /tmp/gateway_secret_env.* "$ENV_CLONE" "$ENV_SECRET" 2>/dev/null || true
  find "$ENV_BACKUP_DIR" -maxdepth 1 -type f \
    \( -name "gateway.env.before-gw-restart.*.bak" -o -name "gateway.env.before-secret-hydrate.*" \) \
    -delete 2>/dev/null || true
  sudo journalctl --vacuum-size=200M 2>/dev/null || true
  emergency_lock_helper="$LEADPOET_REPO_ROOT/validator_tee/scripts/docker_operation_lock_v2.sh"
  if [ ! -r "$emergency_lock_helper" ]; then
    echo "ERROR: emergency Docker lock helper is unavailable" >&2
    return 1
  fi
  . "$emergency_lock_helper"
  if ! leadpoet_acquire_docker_operation_lock_v2; then
    echo "ERROR: emergency Docker cleanup could not acquire its operation lock" >&2
    return 1
  fi
  run_bounded_restart_artifact_cleanup
  if ! VALIDATOR_DOCKER_MIN_FREE_BYTES=$((MIN_FREE_KB * 1024)) \
      VALIDATOR_DOCKER_LIVE_RUNTIME_MIN_FREE_BYTES="${GATEWAY_DOCKER_LIVE_RUNTIME_MIN_FREE_BYTES:-18000000000}" \
      VALIDATOR_DOCKER_ALLOW_DATA_ROOT_RESET=1 \
      bash "$LEADPOET_REPO_ROOT/validator_tee/scripts/reclaim_docker_storage_v2.sh"; then
    leadpoet_release_docker_operation_lock_v2 || true
    echo "ERROR: guarded emergency Docker cleanup failed closed" >&2
    return 1
  fi
  leadpoet_release_docker_operation_lock_v2

  echo "Disk after emergency cleanup"
  df -h / /var/lib/docker 2>/dev/null || df -h /
}

acquire_gateway_restart_lock() {
  local lock_holder_found=0
  local lock_holder_is_stale=1
  local fd_path holder_pid holder_command stale_lock_file

  exec 8>"$GATEWAY_RESTART_RECOVERY_LOCK_FILE"
  chmod 600 "$GATEWAY_RESTART_RECOVERY_LOCK_FILE"
  flock 8

  exec 9>"$GATEWAY_RESTART_LOCK_FILE"
  chmod 600 "$GATEWAY_RESTART_LOCK_FILE"
  if flock -n 9; then
    flock -u 8
    exec 8>&-
    return 0
  fi

  for fd_path in /proc/[0-9]*/fd/*; do
    if [ "$(readlink "$fd_path" 2>/dev/null || true)" != "$GATEWAY_RESTART_LOCK_FILE" ]; then
      continue
    fi
    holder_pid="${fd_path#/proc/}"
    holder_pid="${holder_pid%%/*}"
    if [ "$holder_pid" = "$$" ]; then
      continue
    fi
    lock_holder_found=1
    holder_command="$(tr '\0' ' ' < "/proc/$holder_pid/cmdline" 2>/dev/null || true)"
    case "$holder_command" in
      *"gateway.utils.tee_inter_enclave_relay"*|\
      *"gateway.utils.tee_egress_forwarder"*|\
      *" -m gateway.main "*)
        ;;
      *)
        lock_holder_is_stale=0
        ;;
    esac
  done

  if [ "$lock_holder_found" -ne 1 ] || [ "$lock_holder_is_stale" -ne 1 ]; then
    echo "ERROR: another gateway restart is already running" >&2
    exit 1
  fi

  stale_lock_file="${GATEWAY_RESTART_LOCK_FILE}.stale.$$"
  echo "Recovering gateway restart lock inherited by a detached runtime process"
  mv -- "$GATEWAY_RESTART_LOCK_FILE" "$stale_lock_file"
  exec 9>&-
  exec 9>"$GATEWAY_RESTART_LOCK_FILE"
  chmod 600 "$GATEWAY_RESTART_LOCK_FILE"
  if ! flock -n 9; then
    echo "ERROR: gateway restart lock recovery lost a concurrency race" >&2
    exit 1
  fi
  rm -f -- "$stale_lock_file"
  flock -u 8
  exec 8>&-
}

if [ "$GATEWAY_RESTART_PHASE" = "prepare" ]; then
  # Fresh preparation never trusts guard authority inherited from a caller.
  # The exact drain below establishes and replaces this value.
  PREPARED_GATEWAY_SHA=""
  LAB_ARENA_RESTART_GUARD_GENERATION=""
  mkdir -p \
    "$(dirname "$GATEWAY_RESTART_LOCK_FILE")" \
    "$(dirname "$GATEWAY_RESTART_RECOVERY_LOCK_FILE")" \
    "$GATEWAY_DEPLOYMENT_DIR"
  chmod 700 "$(dirname "$GATEWAY_RESTART_LOCK_FILE")" "$GATEWAY_DEPLOYMENT_DIR"
  command -v flock >/dev/null 2>&1 || {
    echo "ERROR: flock is required for gateway Git deployments" >&2
    exit 1
  }
  if [ "${GATEWAY_RESTART_LOCK_HELD:-0}" = "1" ]; then
    if [ ! -e "/proc/$$/fd/9" ] \
        || [ "$(readlink "/proc/$$/fd/9" 2>/dev/null || true)" != "$GATEWAY_RESTART_LOCK_FILE" ]; then
      echo "ERROR: re-executed gateway restart lost the deployment lock" >&2
      exit 1
    fi
  else
    acquire_gateway_restart_lock
    export GATEWAY_RESTART_LOCK_HELD=1
  fi
elif [ "$GATEWAY_RESTART_PHASE" = "post_activate" ]; then
  validate_post_activate_arena_guard_authority || exit 1
  if [ "${GATEWAY_RESTART_LOCK_HELD:-0}" != "1" ] || [ ! -e "/proc/$$/fd/9" ]; then
    echo "ERROR: post-activation gateway restart lost the deployment lock" >&2
    exit 1
  fi
  DOCKER_LOCK_HELPER="$LEADPOET_REPO_ROOT/validator_tee/scripts/docker_operation_lock_v2.sh"
  if [ ! -r "$DOCKER_LOCK_HELPER" ]; then
    echo "ERROR: activated Docker operation lock helper is unavailable" >&2
    exit 1
  fi
  . "$DOCKER_LOCK_HELPER"
  leadpoet_ensure_post_activation_docker_operation_lock_v2
else
  echo "ERROR: unsupported GATEWAY_RESTART_PHASE=$GATEWAY_RESTART_PHASE" >&2
  exit 1
fi

if [ "$GATEWAY_RESTART_PHASE" = "prepare" ]; then
  validate_gateway_aws_authority
fi

if [ "$miner_maintenance_bootstrap_count" -eq 4 ]; then
  bootstrap_script_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
  bootstrap_candidate_root="$GATEWAY_MINER_MAINTENANCE_BOOTSTRAP_ROOT/candidate"
  if [ "$GATEWAY_RESTART_PHASE" != "prepare" ] \
      || [ "${GATEWAY_RESTART_LOCK_HELD:-0}" != "1" ] \
      || [ ! -e "/proc/$$/fd/9" ] \
      || [ "$(readlink "/proc/$$/fd/9" 2>/dev/null || true)" != "$GATEWAY_RESTART_LOCK_FILE" ] \
      || [ "$bootstrap_script_root" != "$bootstrap_candidate_root" ] \
      || [ -e "$GATEWAY_MINER_MAINTENANCE_HANDOFF_FILE" ] \
      || [ "$LEADPOET_GATEWAY_ENV_SECRET_ID" != "leadpoet/prod/gateway/env" ] \
      || [ "${AWS_REGION:-us-east-1}" != "us-east-1" ] \
      || [ "${AWS_DEFAULT_REGION:-us-east-1}" != "us-east-1" ] \
      || [ ! -x "$GATEWAY_HOST_RESTART_SCRIPT" ]; then
    echo "ERROR: miner-maintenance bootstrap did not acquire an isolated canonical handoff" >&2
    exit 1
  fi

  GATEWAY_DEPLOY_STAGE="miner_maintenance_pre_hydration"
  export GATEWAY_DEPLOY_STAGE
  echo "Preparing disabled miner submissions from the exact candidate under the canonical restart lock"
  export GATEWAY_RESTART_STARTED_EPOCH
  export GATEWAY_RESTART_TIMING_DIR
  export GATEWAY_RESTART_TIMING_FILE
  export GATEWAY_RESTART_TIMING_INITIALIZED
  export PYTHONDONTWRITEBYTECODE=1
  export PYTHONPATH="$bootstrap_candidate_root"
  cd /
  exec env \
    -u GATEWAY_RESTART_AUTHORITY_ROOT \
    -u GATEWAY_RESTART_AUTHORITY_COMMIT \
    "$GATEWAY_PYTHON_BIN" \
    -P -m gateway.tee.gateway_miner_maintenance_restart_v1 \
    --bootstrap-exec \
    --expected-commit "$REQUESTED_GATEWAY_DEPLOY_COMMIT" \
    --repo-root "$LEADPOET_REPO_ROOT" \
    --plan-file "$GATEWAY_MINER_MAINTENANCE_BOOTSTRAP_PLAN" \
    --bootstrap-root "$GATEWAY_MINER_MAINTENANCE_BOOTSTRAP_ROOT" \
    --controller-current "$GATEWAY_RESTART_CONTROLLER_CURRENT" \
    --host-restart-path "$GATEWAY_HOST_RESTART_SCRIPT" \
    --handoff-file "$GATEWAY_MINER_MAINTENANCE_HANDOFF_FILE" \
    --handoff-nonce "$GATEWAY_MINER_MAINTENANCE_HANDOFF_NONCE"
fi

if [ "$GATEWAY_RESTART_PHASE" = "prepare" ]; then
cd "$GATEWAY_ROOT"

PID="$(pgrep -f "python3 -u main.py|python3 -u -m gateway.main" | head -1 || true)"
echo "main pid before: ${PID:-none}"
if [ -z "${PID:-}" ]; then
  echo "main.py not currently running; continuing with Secrets Manager env only"
fi

export AWS_REGION="${AWS_REGION:-us-east-1}"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"

emergency_disk_preflight

echo "Hydrating gateway env from Secrets Manager before stopping processes"
mkdir -p "$(dirname "$GATEWAY_ENV_FILE")" "$ENV_BACKUP_DIR"
chmod 700 "$(dirname "$GATEWAY_ENV_FILE")" "$ENV_BACKUP_DIR"
if [ -f "$GATEWAY_ENV_FILE" ]; then
  find "$ENV_BACKUP_DIR" -maxdepth 1 -type f -name "gateway.env.before-gw-restart.*.bak" \
    -printf "%T@ %p\n" 2>/dev/null \
    | sort -nr \
    | awk 'NR > 5 {print substr($0, index($0,$2))}' \
    | xargs -r rm -f
  BACKUP_PATH="$ENV_BACKUP_DIR/gateway.env.before-gw-restart.$(date -u +%Y%m%dT%H%M%SZ).bak"
  if cp -p "$GATEWAY_ENV_FILE" "$BACKUP_PATH"; then
    echo "Backed up cached gateway env to $BACKUP_PATH"
  else
    echo "WARNING: failed to back up cached gateway env; continuing with Secrets Manager hydration"
  fi
fi

SECRET_TMP="$(mktemp /tmp/gateway_secret_env.XXXXXX)"
aws secretsmanager get-secret-value \
  --secret-id "$LEADPOET_GATEWAY_ENV_SECRET_ID" \
  --query SecretString \
  --output text > "$SECRET_TMP"

python3 - "$SECRET_TMP" "$GATEWAY_ENV_FILE" <<'PY'
import json
import shlex
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
raw = src.read_text()
restart_only_keys = {
    "GATEWAY_DEPLOY_COMMIT",
    "GATEWAY_ENV_FILE",
    "GATEWAY_PRIVATE_KEY_PATH",
    "ARWEAVE_KEYFILE_PATH",
    "GATEWAY_RESTART_GIT_SSH_COMMAND",
    "GATEWAY_PYTHON_BIN",
    "GATEWAY_RESTART_CONTROLLER_ROOT",
    "GATEWAY_RESTART_AUTHORITY_ROOT",
    "GATEWAY_RESTART_AUTHORITY_COMMIT",
    "GATEWAY_ACTIVE_RELEASE_RESTART_INVOCATION_ID",
    "GATEWAY_ACTIVE_RELEASE_COMPONENT",
    "PREPARED_GATEWAY_SHA",
    "LAB_ARENA_RESTART_GUARD_GENERATION",
    "GATEWAY_RESTART_RECOVERY_LOCK_FILE",
    "GATEWAY_RESTART_INVOCATION_ID",
    "GATEWAY_MINER_MAINTENANCE_PROOF_FD",
    "GATEWAY_CONTROLLER_PROCESS_HELPER",
    "LAB_ARENA_PROCESS_HELPER",
    "GATEWAY_GIT_HELPER",
    "GATEWAY_HOST_MEMORY_GUARD_PATH",
    "GATEWAY_V2_ARTIFACT_POLICY",
    "GATEWAY_V2_CONFIG_DIR",
    "GATEWAY_V2_DEFER_WORKER_FLEETS",
    "GATEWAY_V2_KMS_KEY_ID",
    "GATEWAY_V2_OFFLINE_ARTIFACT_ROOT",
    "GATEWAY_V2_RELEASE_BUCKET",
    "GATEWAY_V2_RELEASE_ARCHIVE_ROOT",
    "GATEWAY_V2_RELEASE_LINEAGE",
    "GATEWAY_V2_RELEASE_MANIFEST",
    "GATEWAY_PREPARED_V2_RELEASE_MANIFEST",
    "GATEWAY_PREPARED_V2_RELEASE_LINEAGE",
    "GATEWAY_V2_RELEASE_PREFIX",
    "GATEWAY_RESTART_TEMP_CLEANUP_MIN_AGE_SECONDS",
    "GATEWAY_RESTART_EMERGENCY_BACKUP_MIN_AGE_SECONDS",
    "GATEWAY_RESTART_CLEANUP_MAX_CANDIDATES",
    "LEADPOET_DOCKER_OPERATION_LOCK_FILE",
    "LEADPOET_GATEWAY_ENV_SECRET_ID",
    "LEADPOET_RESTART_INVOCATION_ID",
    "LEADPOET_SENTRY_API_TOKEN",
}

try:
    parsed = json.loads(raw)
except Exception:
    parsed = None

if isinstance(parsed, dict):
    lines = []
    for key, value in parsed.items():
        if key in restart_only_keys:
            continue
        if isinstance(value, (dict, list)):
            value = json.dumps(value, separators=(",", ":"))
        elif value is None:
            value = ""
        lines.append(f"{key}={value}")
    raw = "\n".join(lines) + "\n"
else:
    lines = []
    for raw_line in raw.replace("\x00", "\n").splitlines():
        line = raw_line.strip()
        candidate = line[7:].strip() if line.startswith("export ") else line
        try:
            parts = shlex.split(candidate, posix=True)
        except ValueError:
            parts = [candidate]
        assignment = parts[0] if len(parts) == 1 else candidate
        key = assignment.split("=", 1)[0].strip() if "=" in assignment else ""
        if key in restart_only_keys:
            continue
        lines.append(raw_line)
    raw = "\n".join(lines)
    if lines:
        raw += "\n"

dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(raw)
PY
chmod 600 "$GATEWAY_ENV_FILE"
rm -f "$SECRET_TMP"

python3 - "$GATEWAY_ENV_FILE" "$ENV_SECRET" <<'PY'
import re
import shlex
import sys
from pathlib import Path

env_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
skip_keys = {
    "AWS_ACCESS_KEY_ID",
    "AWS_CA_BUNDLE",
    "AWS_CONFIG_FILE",
    "AWS_CONTAINER_CREDENTIALS_FULL_URI",
    "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
    "AWS_DEFAULT_PROFILE",
    "AWS_DEFAULT_REGION",
    "AWS_EC2_METADATA_SERVICE_ENDPOINT",
    "AWS_EC2_METADATA_SERVICE_ENDPOINT_MODE",
    "AWS_ENDPOINT_URL",
    "AWS_ENDPOINT_URL_S3",
    "AWS_ENDPOINT_URL_SECRETSMANAGER",
    "AWS_ENDPOINT_URL_STS",
    "AWS_METADATA_SERVICE_NUM_ATTEMPTS",
    "AWS_METADATA_SERVICE_TIMEOUT",
    "AWS_REGION",
    "AWS_ROLE_ARN",
    "AWS_ROLE_SESSION_NAME",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SHARED_CREDENTIALS_FILE",
    "AWS_SESSION_TOKEN",
    "AWS_SECURITY_TOKEN",
    "AWS_PROFILE",
    "AWS_WEB_IDENTITY_TOKEN_FILE",
    "BOTO_CONFIG",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "LEADPOET_AWS_INSTANCE_ROLE_ONLY",
    "GATEWAY_DEPLOY_COMMIT",
    "GATEWAY_RESTART_INVOCATION_ID",
    "GATEWAY_V2_DEFER_WORKER_FLEETS",
    "LEADPOET_REPO_ROOT",
    "GATEWAY_ROOT",
    "GATEWAY_LOG_ROOT",
    "GATEWAY_LOG_FILE",
    "GATEWAY_TEE_EIF_ROOT",
    "GATEWAY_ENV_FILE",
    "GATEWAY_PRIVATE_KEY_PATH",
    "ARWEAVE_KEYFILE_PATH",
    "GATEWAY_RESTART_GIT_SSH_COMMAND",
    "GATEWAY_PYTHON_BIN",
    "GATEWAY_RESTART_CONTROLLER_ROOT",
    "GATEWAY_RESTART_AUTHORITY_ROOT",
    "GATEWAY_RESTART_AUTHORITY_COMMIT",
    "GATEWAY_ACTIVE_RELEASE_RESTART_INVOCATION_ID",
    "GATEWAY_ACTIVE_RELEASE_COMPONENT",
    "PREPARED_GATEWAY_SHA",
    "LAB_ARENA_RESTART_GUARD_GENERATION",
    "GATEWAY_RESTART_RECOVERY_LOCK_FILE",
    "GATEWAY_V2_ARTIFACT_POLICY",
    "GATEWAY_V2_CONFIG_DIR",
    "GATEWAY_V2_KMS_KEY_ID",
    "GATEWAY_V2_OFFLINE_ARTIFACT_ROOT",
    "GATEWAY_V2_RELEASE_BUCKET",
    "GATEWAY_V2_RELEASE_ARCHIVE_ROOT",
    "GATEWAY_V2_RELEASE_LINEAGE",
    "GATEWAY_V2_RELEASE_MANIFEST",
    "GATEWAY_PREPARED_V2_RELEASE_MANIFEST",
    "GATEWAY_PREPARED_V2_RELEASE_LINEAGE",
    "GATEWAY_V2_RELEASE_PREFIX",
    "GATEWAY_RESTART_TEMP_CLEANUP_MIN_AGE_SECONDS",
    "GATEWAY_RESTART_EMERGENCY_BACKUP_MIN_AGE_SECONDS",
    "GATEWAY_RESTART_CLEANUP_MAX_CANDIDATES",
    "GATEWAY_TEE_FALLBACK_LOG_DIR",
    "GATEWAY_GIT_HELPER",
    "GATEWAY_HOST_MEMORY_GUARD_PATH",
    "GATEWAY_CONTROLLER_PROCESS_HELPER",
    "LAB_ARENA_PROCESS_HELPER",
    "GATEWAY_MINER_MAINTENANCE_PROOF_FD",
    "GATEWAY_DEPENDENCY_INSTALL_FINGERPRINT",
    "GATEWAY_RESTART_PHASE",
    "GATEWAY_RESTART_STARTED_EPOCH",
    "GATEWAY_RESTART_TIMING_DIR",
    "GATEWAY_RESTART_TIMING_FILE",
    "GATEWAY_RESTART_TIMING_INITIALIZED",
    "GATEWAY_RELEASE_FOLLOW_ROOT",
    "GATEWAY_RELEASE_SUPERSESSION_COUNT",
    "GATEWAY_RELEASE_SUPERSESSION_MAX",
    "LEADPOET_RESTART_INVOCATION_ID",
    "LEADPOET_DOCKER_OPERATION_LOCK_FILE",
    "LEADPOET_GATEWAY_ENV_SECRET_ID",
    "LEADPOET_SENTRY_API_TOKEN",
    "GATEWAY_RESTART_LOCK_HELD",
    "GATEWAY_RESTART_LOCK_FILE",
    "GATEWAY_DEPLOY_PLAN_FILE",
    "GATEWAY_DEPLOYMENT_DIR",
    "GATEWAY_DEPLOYMENT_MANIFEST",
    "GATEWAY_LAST_GOOD_MANIFEST",
    "GATEWAY_HOST_RESTART_SCRIPT",
    "GATEWAY_DEPLOY_STAGE",
    "GATEWAY_DEPLOY_COMPLETED",
    "PYTHONPATH",
    "GITHUB_SHA",
    "GITHUB_COMMIT",
    "GITHUB_REF_NAME",
    "GITHUB_TAG",
    "GIT_BRANCH",
    "GIT_TAG",
    "BUILD_ID",
    "BUILD_TIME_UTC",
    "BUILD_TIMESTAMP",
    "GATEWAY_BUILD_INFO_GIT_ROOT",
    "GATEWAY_BUILD_INFO_FILE",
    "RESEARCH_LAB_RUNTIME_SOURCE_ROOT",
    "ATTESTED_RUNTIME_COMMIT_SHA",
    "ATTESTED_RUNTIME_GIT_REPO_URL",
}

out = []
for raw_line in env_path.read_text(errors="replace").replace("\x00", "\n").splitlines():
    line = raw_line.strip()
    if not line or line.startswith("#"):
        continue
    if line.startswith("export "):
        line = line[len("export "):].strip()
    try:
        parts = shlex.split(line, posix=True)
    except ValueError:
        parts = [line]
    if len(parts) != 1 or "=" not in parts[0]:
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
    else:
        key, value = parts[0].split("=", 1)
    key = key.strip()
    if key in skip_keys:
        continue
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key):
        continue
    out.append(f"export {key}={shlex.quote(value)}")

out_path.write_text("\n".join(out) + "\n")
print(f"hydrated env cache and prepared {len(out)} secret env vars")
PY

if [ -n "${PID:-}" ] && [ -r "/proc/$PID/environ" ]; then
  echo "Cloning live gateway env before stopping processes"
  python3 - "$PID" "$ENV_CLONE" <<'PY'
import re
import shlex
import sys

pid = sys.argv[1]
out_path = sys.argv[2]
skip_keys = {
    "AWS_ACCESS_KEY_ID",
    "AWS_CA_BUNDLE",
    "AWS_CONFIG_FILE",
    "AWS_CONTAINER_CREDENTIALS_FULL_URI",
    "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
    "AWS_DEFAULT_PROFILE",
    "AWS_DEFAULT_REGION",
    "AWS_EC2_METADATA_SERVICE_ENDPOINT",
    "AWS_EC2_METADATA_SERVICE_ENDPOINT_MODE",
    "AWS_ENDPOINT_URL",
    "AWS_ENDPOINT_URL_S3",
    "AWS_ENDPOINT_URL_SECRETSMANAGER",
    "AWS_ENDPOINT_URL_STS",
    "AWS_METADATA_SERVICE_NUM_ATTEMPTS",
    "AWS_METADATA_SERVICE_TIMEOUT",
    "AWS_REGION",
    "AWS_ROLE_ARN",
    "AWS_ROLE_SESSION_NAME",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SHARED_CREDENTIALS_FILE",
    "AWS_SESSION_TOKEN",
    "AWS_SECURITY_TOKEN",
    "AWS_PROFILE",
    "AWS_WEB_IDENTITY_TOKEN_FILE",
    "BOTO_CONFIG",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "LEADPOET_AWS_INSTANCE_ROLE_ONLY",
    "GATEWAY_DEPLOY_COMMIT",
    "GATEWAY_RESTART_INVOCATION_ID",
    "GATEWAY_V2_DEFER_WORKER_FLEETS",
    "LEADPOET_REPO_ROOT",
    "GATEWAY_ROOT",
    "GATEWAY_LOG_ROOT",
    "GATEWAY_LOG_FILE",
    "GATEWAY_TEE_EIF_ROOT",
    "GATEWAY_ENV_FILE",
    "GATEWAY_PRIVATE_KEY_PATH",
    "ARWEAVE_KEYFILE_PATH",
    "GATEWAY_RESTART_GIT_SSH_COMMAND",
    "GATEWAY_PYTHON_BIN",
    "GATEWAY_RESTART_CONTROLLER_ROOT",
    "GATEWAY_RESTART_AUTHORITY_ROOT",
    "GATEWAY_RESTART_AUTHORITY_COMMIT",
    "GATEWAY_ACTIVE_RELEASE_RESTART_INVOCATION_ID",
    "GATEWAY_ACTIVE_RELEASE_COMPONENT",
    "PREPARED_GATEWAY_SHA",
    "LAB_ARENA_RESTART_GUARD_GENERATION",
    "GATEWAY_RESTART_RECOVERY_LOCK_FILE",
    "GATEWAY_V2_ARTIFACT_POLICY",
    "GATEWAY_V2_CONFIG_DIR",
    "GATEWAY_V2_KMS_KEY_ID",
    "GATEWAY_V2_OFFLINE_ARTIFACT_ROOT",
    "GATEWAY_V2_RELEASE_BUCKET",
    "GATEWAY_V2_RELEASE_ARCHIVE_ROOT",
    "GATEWAY_V2_RELEASE_LINEAGE",
    "GATEWAY_V2_RELEASE_MANIFEST",
    "GATEWAY_PREPARED_V2_RELEASE_MANIFEST",
    "GATEWAY_PREPARED_V2_RELEASE_LINEAGE",
    "GATEWAY_V2_RELEASE_PREFIX",
    "GATEWAY_RESTART_TEMP_CLEANUP_MIN_AGE_SECONDS",
    "GATEWAY_RESTART_EMERGENCY_BACKUP_MIN_AGE_SECONDS",
    "GATEWAY_RESTART_CLEANUP_MAX_CANDIDATES",
    "GATEWAY_TEE_FALLBACK_LOG_DIR",
    "GATEWAY_GIT_HELPER",
    "GATEWAY_HOST_MEMORY_GUARD_PATH",
    "GATEWAY_CONTROLLER_PROCESS_HELPER",
    "LAB_ARENA_PROCESS_HELPER",
    "GATEWAY_MINER_MAINTENANCE_PROOF_FD",
    "GATEWAY_DEPENDENCY_INSTALL_FINGERPRINT",
    "GATEWAY_RESTART_PHASE",
    "GATEWAY_RESTART_STARTED_EPOCH",
    "GATEWAY_RESTART_TIMING_DIR",
    "GATEWAY_RESTART_TIMING_FILE",
    "GATEWAY_RESTART_TIMING_INITIALIZED",
    "GATEWAY_RELEASE_FOLLOW_ROOT",
    "GATEWAY_RELEASE_SUPERSESSION_COUNT",
    "GATEWAY_RELEASE_SUPERSESSION_MAX",
    "LEADPOET_RESTART_INVOCATION_ID",
    "LEADPOET_DOCKER_OPERATION_LOCK_FILE",
    "LEADPOET_GATEWAY_ENV_SECRET_ID",
    "GATEWAY_RESTART_LOCK_HELD",
    "GATEWAY_RESTART_LOCK_FILE",
    "GATEWAY_DEPLOY_PLAN_FILE",
    "GATEWAY_DEPLOYMENT_DIR",
    "GATEWAY_DEPLOYMENT_MANIFEST",
    "GATEWAY_LAST_GOOD_MANIFEST",
    "GATEWAY_HOST_RESTART_SCRIPT",
    "GATEWAY_DEPLOY_STAGE",
    "GATEWAY_DEPLOY_COMPLETED",
    "PYTHONPATH",
    "GITHUB_SHA",
    "GITHUB_COMMIT",
    "GITHUB_REF_NAME",
    "GITHUB_TAG",
    "GIT_BRANCH",
    "GIT_TAG",
    "BUILD_ID",
    "BUILD_TIME_UTC",
    "BUILD_TIMESTAMP",
    "GATEWAY_BUILD_INFO_GIT_ROOT",
    "GATEWAY_BUILD_INFO_FILE",
    "RESEARCH_LAB_RUNTIME_SOURCE_ROOT",
    "ATTESTED_RUNTIME_COMMIT_SHA",
    "ATTESTED_RUNTIME_GIT_REPO_URL",
}
data = open(f"/proc/{pid}/environ", "rb").read()
out = []
for kv in data.split(b"\0"):
    if not kv:
        continue
    s = kv.decode("utf-8", "replace")
    if "=" not in s:
        continue
    k, v = s.split("=", 1)
    if k in skip_keys:
        continue
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", k):
        continue
    out.append(f"export {k}={shlex.quote(v)}")
open(out_path, "w").write("\n".join(out) + "\n")
print(f"cloned {len(out)} env vars")
PY
else
  echo "No live gateway env available; using hydrated Secrets Manager env only"
  : > "$ENV_CLONE"
fi

cat "$ENV_SECRET" >> "$ENV_CLONE"
# Per-invocation telemetry identity belongs to this restart controller, never
# to a cached secret or the previously running gateway.  Keep the authoritative
# values last so every later environment reload and launched process agrees
# with the active timing ledger.
printf 'export GATEWAY_RESTART_INVOCATION_ID=%q\n' \
  "$GATEWAY_RESTART_INVOCATION_ID" >> "$ENV_CLONE"
printf 'export LEADPOET_RESTART_INVOCATION_ID=%q\n' \
  "$GATEWAY_RESTART_INVOCATION_ID" >> "$ENV_CLONE"
printf 'export AWS_REGION=us-east-1\n' >> "$ENV_CLONE"
printf 'export AWS_DEFAULT_REGION=us-east-1\n' >> "$ENV_CLONE"
printf 'export LEADPOET_AWS_INSTANCE_ROLE_ONLY=true\n' >> "$ENV_CLONE"
printf 'export GATEWAY_PRIVATE_KEY_PATH=%q\n' "$GATEWAY_PRIVATE_KEY_PATH" >> "$ENV_CLONE"
printf 'export ARWEAVE_KEYFILE_PATH=%q\n' "$ARWEAVE_KEYFILE_PATH" >> "$ENV_CLONE"
if [ -n "$GATEWAY_RESTART_GIT_SSH_COMMAND" ]; then
  printf 'export GATEWAY_RESTART_GIT_SSH_COMMAND=%q\n' \
    "$GATEWAY_RESTART_GIT_SSH_COMMAND" >> "$ENV_CLONE"
  printf 'export GIT_SSH_COMMAND=%q\n' \
    "$GATEWAY_RESTART_GIT_SSH_COMMAND" >> "$ENV_CLONE"
fi

grep -q "SUPABASE_SERVICE_ROLE_KEY" "$ENV_CLONE" || {
  echo "ERROR: hydrated/cloned env missing SUPABASE_SERVICE_ROLE_KEY"
  exit 1
}

echo "Selecting one Python runtime for V2 preflight, bootstrap, and gateway processes"
if ! select_gateway_python_runtime; then
  echo "Gateway remains running; production shutdown has not started." >&2
  exit 1
fi

# Read the protocol only after the live environment and Secrets Manager have
# been merged. Authoritative V2 is the sole production protocol.
RESEARCH_LAB_TEE_PROTOCOL="$(
  set -a
  . "$ENV_CLONE"
  set +a
  printf '%s' "${RESEARCH_LAB_TEE_PROTOCOL:-v2}"
)"
RESEARCH_LAB_TEE_PROTOCOL="$(
  printf '%s' "$RESEARCH_LAB_TEE_PROTOCOL" | tr '[:upper:]' '[:lower:]'
)"
case "$RESEARCH_LAB_TEE_PROTOCOL" in
  v2|authoritative_v2)
    RESEARCH_LAB_TEE_PROTOCOL="v2"
    ;;
  *)
    echo "ERROR: RESEARCH_LAB_TEE_PROTOCOL must be v2; V1 authority is retired" >&2
    exit 1
    ;;
esac
export RESEARCH_LAB_TEE_PROTOCOL
# Keep every later environment reload on the normalized V2 value.
printf 'export RESEARCH_LAB_TEE_PROTOCOL=%q\n' \
  "$RESEARCH_LAB_TEE_PROTOCOL" >> "$ENV_CLONE"
echo "Research Lab TEE protocol: $RESEARCH_LAB_TEE_PROTOCOL"

echo "Validating absolute gateway secret paths for canonical Git checkout"
(
  set -a
  . "$ENV_CLONE"
  set +a
  validate_runtime_secret_paths
)

if [ ! -f "$GATEWAY_GIT_HELPER" ]; then
  echo "ERROR: gateway Git deployment helper is missing: $GATEWAY_GIT_HELPER" >&2
  exit 1
fi

echo "Preparing exact gateway commit from configured GitHub branch"
GATEWAY_DEPLOY_STAGE="git_prepare"
export GATEWAY_DEPLOY_STAGE
PREPARED_GATEWAY_SHA="$(
  python3 "$GATEWAY_GIT_HELPER" prepare \
    --repo-root "$LEADPOET_REPO_ROOT" \
    --env-file "$GATEWAY_ENV_FILE" \
    --deploy-commit "$REQUESTED_GATEWAY_DEPLOY_COMMIT" \
    --plan-file "$GATEWAY_DEPLOY_PLAN_FILE" \
    --manifest-file "$GATEWAY_DEPLOYMENT_MANIFEST" \
    --last-good-file "$GATEWAY_LAST_GOOD_MANIFEST"
)"
echo "Prepared gateway commit: $PREPARED_GATEWAY_SHA"
PREPARED_GATEWAY_TOPOLOGY_ENTRY="$(
  git -C "$LEADPOET_REPO_ROOT" ls-tree \
    "$PREPARED_GATEWAY_SHA" -- gateway/tee/topology.json
)" || {
  echo "ERROR: prepared gateway topology Git identity is unavailable" >&2
  exit 1
}
if ! [[ "$PREPARED_GATEWAY_TOPOLOGY_ENTRY" =~ ^100644\ blob\ [0-9a-f]{40}$'\t'gateway/tee/topology.json$ ]]; then
  echo "ERROR: prepared gateway topology is not one regular Git blob" >&2
  exit 1
fi
PREPARED_GATEWAY_TOPOLOGY_BLOB="${PREPARED_GATEWAY_TOPOLOGY_ENTRY#100644 blob }"
PREPARED_GATEWAY_TOPOLOGY_BLOB="${PREPARED_GATEWAY_TOPOLOGY_BLOB%%$'\t'*}"
if [ "$PREPARED_GATEWAY_TOPOLOGY_BLOB" = "$HISTORICAL_THREE_ROLE_TOPOLOGY_BLOB" ] \
    && ! git -C "$LEADPOET_REPO_ROOT" cat-file -e \
      "$PREPARED_GATEWAY_SHA:gateway/tee/build_local_release_v2.sh" 2>/dev/null \
    && ! git -C "$LEADPOET_REPO_ROOT" cat-file -e \
      "$PREPARED_GATEWAY_SHA:gateway/tee/local_release_v2.py" 2>/dev/null; then
  GATEWAY_HISTORICAL_TOPOLOGY_HASH="$HISTORICAL_THREE_ROLE_TOPOLOGY_HASH"
fi
POST_ACTIVATE_GATEWAY_HOST_RESTART_SCRIPT="$GATEWAY_HOST_RESTART_SCRIPT"
ORIGIN_MAIN_GATEWAY_SHA="$(git -C "$LEADPOET_REPO_ROOT" rev-parse origin/main)"
if [ -n "$REQUESTED_GATEWAY_DEPLOY_COMMIT" ] \
    && [ "$PREPARED_GATEWAY_SHA" != "$ORIGIN_MAIN_GATEWAY_SHA" ] \
    && [ -z "$GATEWAY_RESTART_AUTHORITY_ROOT" ]; then
  echo "Preserving the newer exact-commit gateway restart controller"
  POST_ACTIVATE_GATEWAY_HOST_RESTART_SCRIPT="$LEADPOET_REPO_ROOT/gw_restart.sh"
fi

echo "Materializing the prepared commit for pre-shutdown V2 tooling"
GATEWAY_PREFLIGHT_TREE="$(mktemp -d /tmp/gateway-v2-preflight.XXXXXX)"
if ! git -C "$LEADPOET_REPO_ROOT" archive "$PREPARED_GATEWAY_SHA" \
    | tar -xf - -C "$GATEWAY_PREFLIGHT_TREE"; then
  echo "ERROR: unable to materialize the prepared commit for V2 preflight" >&2
  exit 1
fi
echo "Verifying the prepared gateway tree with the preserved restart controller"
GATEWAY_DEPLOY_STAGE="git_prepared_tree_verification"
export GATEWAY_DEPLOY_STAGE
"$GATEWAY_PYTHON_BIN" "$GATEWAY_GIT_HELPER" \
  verify-tree \
  --plan-file "$GATEWAY_DEPLOY_PLAN_FILE" \
  --materialized-root "$GATEWAY_PREFLIGHT_TREE" \
  --phase prepared_archive \
  --strict-extras

echo "Cleaning stale read-only gateway vsock probes before V2 preflight"
"$GATEWAY_PYTHON_BIN" \
  "$GATEWAY_PREFLIGHT_TREE/gateway/tee/host_memory_guard_v2.py" \
  --cleanup-stale-vsock-probes \
  --minimum-available-mib 1024

RESTART_GATE_ARGS=(
  --network "${BITTENSOR_NETWORK:-finney}"
  --netuid "${BITTENSOR_NETUID:-71}"
)
echo "Capturing the official subnet restart window before release acquisition"
if ! GATEWAY_RESTART_EPOCH_REPORT="$(
    run_prepared_gateway_module Leadpoet.utils.restart_epoch_gate \
      "${RESTART_GATE_ARGS[@]}"
  )"; then
  echo "Gateway remains running; production shutdown has not started." >&2
  exit 75
fi
printf '%s\n' "$GATEWAY_RESTART_EPOCH_REPORT"

echo "Preparing exact hash-locked V2 build artifacts during release acquisition"
if ! start_gateway_offline_artifact_prepare; then
  echo "Gateway remains running; production shutdown has not started." >&2
  exit 75
fi

if ! wait_for_gateway_offline_artifact_prepare; then
  GATEWAY_DEPLOY_STAGE="v2_offline_artifact_prepare"
  export GATEWAY_DEPLOY_STAGE
  echo "Gateway remains running; production shutdown has not started." >&2
  exit 75
fi

if ! follow_superseding_gateway_release; then
  echo "Gateway remains running; production shutdown has not started." >&2
  exit 75
fi
if [ -f "$GATEWAY_LAST_GOOD_MANIFEST" ] \
    && [ -f "$GATEWAY_V2_RELEASE_ARCHIVE_ROOT/index.json" ]; then
  echo "Verifying and repairing retired last-good gateway role measurements"
  PYTHONPATH="$GATEWAY_PREFLIGHT_TREE" python3 \
    "$GATEWAY_PREFLIGHT_TREE/scripts/gateway_git_deploy.py" \
    repair-last-good-role-pcr0s \
    --last-good-file "$GATEWAY_LAST_GOOD_MANIFEST" \
    --archive-root "$GATEWAY_V2_RELEASE_ARCHIVE_ROOT"
fi
GATEWAY_LOCAL_RELEASE_SCRIPT="$GATEWAY_PREFLIGHT_TREE/gateway/tee/build_local_release_v2.sh"
GATEWAY_LOCAL_RELEASE_MODULE="$GATEWAY_PREFLIGHT_TREE/gateway/tee/local_release_v2.py"
GATEWAY_HISTORICAL_RELEASE_MODULE="$GATEWAY_PREFLIGHT_TREE/gateway/tee/release_channel_v2.py"
if [ -f "$GATEWAY_LOCAL_RELEASE_SCRIPT" ] \
    && [ -r "$GATEWAY_LOCAL_RELEASE_SCRIPT" ] \
    && [ ! -L "$GATEWAY_LOCAL_RELEASE_SCRIPT" ] \
    && [ -f "$GATEWAY_LOCAL_RELEASE_MODULE" ] \
    && [ -r "$GATEWAY_LOCAL_RELEASE_MODULE" ] \
    && [ ! -L "$GATEWAY_LOCAL_RELEASE_MODULE" ]; then
  echo "Building the exact local gateway runtime identity"
  GATEWAY_DEPLOY_STAGE="local_release_build"
  export GATEWAY_DEPLOY_STAGE
  if ! PYTHONPATH="$GATEWAY_PREFLIGHT_TREE" \
      GATEWAY_V2_BUILD_WORK_ROOT="${GATEWAY_V2_BUILD_WORK_ROOT:-$HOME/.cache/leadpoet/gateway-release-build-v2}" \
      GATEWAY_V2_OFFLINE_ARTIFACT_ROOT="$GATEWAY_V2_OFFLINE_ARTIFACT_ROOT" \
      bash "$GATEWAY_PREFLIGHT_TREE/gateway/tee/build_local_release_v2.sh" \
        --repository "$LEADPOET_REPO_ROOT" \
        --revision "$PREPARED_GATEWAY_SHA" \
        --gateway-output "$GATEWAY_PREPARED_V2_RELEASE_MANIFEST"; then
    echo "ERROR: exact local runtime identity build failed" >&2
    echo "Gateway remains running; production shutdown has not started." >&2
    exit 75
  fi
  export LEADPOET_LOCAL_RELEASE_COMMIT_SHA="$PREPARED_GATEWAY_SHA"
  export LEADPOET_LOCAL_GATEWAY_RELEASE="$GATEWAY_PREPARED_V2_RELEASE_MANIFEST"
  if [ -e "$GATEWAY_V2_RELEASE_LINEAGE" ] \
      || [ -L "$GATEWAY_V2_RELEASE_LINEAGE" ]; then
    export LEADPOET_LOCAL_PRIOR_RELEASE_LINEAGE="$GATEWAY_V2_RELEASE_LINEAGE"
  else
    unset LEADPOET_LOCAL_PRIOR_RELEASE_LINEAGE
  fi
  if ! run_prepared_gateway_module gateway.tee.release_channel_v2 \
        --ensure \
        --expected-commit "$PREPARED_GATEWAY_SHA" \
        --gateway-output "$GATEWAY_PREPARED_V2_RELEASE_MANIFEST" \
        --lineage-output "$GATEWAY_PREPARED_V2_RELEASE_LINEAGE" \
        --lineage-repository "$LEADPOET_REPO_ROOT" \
        --lineage-authority-commit "$PREPARED_GATEWAY_SHA" \
        --lineage-required-commit "$PREPARED_GATEWAY_SHA"; then
    echo "ERROR: exact local gateway release lineage is invalid" >&2
    echo "Gateway remains running; production shutdown has not started." >&2
    exit 75
  fi
  record_gateway_restart_timing "local_release_ready"
elif [ ! -e "$GATEWAY_LOCAL_RELEASE_SCRIPT" ] \
    && [ ! -L "$GATEWAY_LOCAL_RELEASE_SCRIPT" ] \
    && [ ! -e "$GATEWAY_LOCAL_RELEASE_MODULE" ] \
    && [ ! -L "$GATEWAY_LOCAL_RELEASE_MODULE" ] \
    && [ -f "$GATEWAY_HISTORICAL_RELEASE_MODULE" ] \
    && [ -r "$GATEWAY_HISTORICAL_RELEASE_MODULE" ] \
    && [ ! -L "$GATEWAY_HISTORICAL_RELEASE_MODULE" ] \
    && [ "$GATEWAY_HISTORICAL_TOPOLOGY_HASH" = "$HISTORICAL_THREE_ROLE_TOPOLOGY_HASH" ] \
    && [ -n "$REQUESTED_GATEWAY_DEPLOY_COMMIT" ] \
    && [ "$PREPARED_GATEWAY_SHA" != "$ORIGIN_MAIN_GATEWAY_SHA" ]; then
  echo "Acquiring the exact historical attested V2 release channel"
  GATEWAY_DEPLOY_STAGE="historical_release_acquisition"
  export GATEWAY_DEPLOY_STAGE
  unset LEADPOET_LOCAL_RELEASE_COMMIT_SHA LEADPOET_LOCAL_GATEWAY_RELEASE
  record_gateway_restart_timing "release_wait_started"
  V2_RELEASE_READY=0
  for attempt in $(seq 1 300); do
    GATEWAY_RELEASE_ATTEMPTS_USED="$attempt"
    if follow_superseding_gateway_release \
        && run_prepared_gateway_module gateway.tee.release_channel_v2 \
        --ensure \
        --expected-commit "$PREPARED_GATEWAY_SHA" \
        --bucket "$GATEWAY_V2_RELEASE_BUCKET" \
        --prefix "$GATEWAY_V2_RELEASE_PREFIX" \
        --gateway-output "$GATEWAY_PREPARED_V2_RELEASE_MANIFEST"; then
      V2_RELEASE_READY=1
      break
    fi
    echo "Exact historical V2 release is not published yet; waiting inside the valid restart invocation (${attempt}/300)"
    sleep 12
  done
  if [ "$V2_RELEASE_READY" != "1" ]; then
    echo "ERROR: exact historical attested V2 release is unavailable for $PREPARED_GATEWAY_SHA" >&2
    echo "Gateway remains running; production shutdown has not started." >&2
    exit 75
  fi
  record_gateway_restart_timing "release_ready"
  record_gateway_restart_timing "historical_release_ready"
else
  echo "ERROR: selected release has an incomplete or unsupported V2 release acquisition contract" >&2
  echo "Gateway remains running; production shutdown has not started." >&2
  exit 75
fi
if ! follow_superseding_gateway_release; then
  echo "Gateway remains running; production shutdown has not started." >&2
  exit 75
fi

echo "Preparing commit-bound KMS credential envelopes"
GATEWAY_DEPLOY_STAGE="v2_credential_envelope_preparation"
export GATEWAY_DEPLOY_STAGE
gateway_envelope_schema_args=()
if [ -n "$GATEWAY_MIGRATION_203_SQL_SHA256" ]; then
  gateway_envelope_schema_args+=(--defer-incentive-retirement-schema)
fi
if ! run_prepared_gateway_module gateway.tee.prepare_gateway_envelopes_v2 \
    --install \
    --env-file "$ENV_CLONE" \
    --kms-key-id "$GATEWAY_V2_KMS_KEY_ID" \
    --deploy-commit "$PREPARED_GATEWAY_SHA" \
    --output-dir "$GATEWAY_V2_CONFIG_DIR" \
    "${gateway_envelope_schema_args[@]}"; then
  echo "ERROR: gateway V2 credential envelope preparation failed before shutdown" >&2
  exit 75
fi

(
  cd "$GATEWAY_PREFLIGHT_TREE"
  PYTHONPATH="$GATEWAY_PREFLIGHT_TREE" \
  "$GATEWAY_PYTHON_BIN" - "$ENV_CLONE" "$GATEWAY_V2_CONFIG_DIR/gateway-v2-env-transition.json" <<'PY'
import sys

from gateway.tee.prepare_gateway_envelopes_v2 import (
    scrub_parent_environment_file_v2,
)

scrub_parent_environment_file_v2(
    environment_path=sys.argv[1],
    transition_report_path=sys.argv[2],
)
print("Scrubbed commit-bound provider plaintext from prepared parent environment")
PY
)

if [ ! -e "$GATEWAY_V2_ARTIFACT_POLICY" ]; then
  echo "Installing the public production V2 artifact policy"
  python3 - "$GATEWAY_V2_ARTIFACT_POLICY" <<'PY'
import json
import os
from pathlib import Path
import sys
import tempfile

destination = Path(sys.argv[1])
destination.parent.mkdir(parents=True, exist_ok=True)
value = {
    "schema_version": "leadpoet.encrypted_artifact_policy.v2",
    "bucket_host": (
        "leadpoet-attested-v2-artifacts-493765492819."
        "s3.us-east-1.amazonaws.com"
    ),
    "key_prefix": "/encrypted-artifacts/",
    "minimum_retention_days": 365,
}
descriptor, temporary_name = tempfile.mkstemp(
    prefix=".artifact-policy.", dir=str(destination.parent)
)
try:
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary_name, 0o600)
    os.replace(temporary_name, destination)
finally:
    Path(temporary_name).unlink(missing_ok=True)
PY
fi

if report_gateway_v2_bootstrap_pending; then
  exit 75
fi

echo "Installing gateway host Python dependencies before production shutdown"
GATEWAY_DEPLOY_STAGE="dependency_preflight"
export GATEWAY_DEPLOY_STAGE
if ! install_gateway_python_dependencies; then
  echo "ERROR: gateway host dependency installation failed before shutdown" >&2
  echo "Gateway remains running; production shutdown has not started." >&2
  exit 1
fi
record_gateway_restart_timing "dependency_preflight_complete"

echo "Validating the prepared V2 release before production shutdown"
  GATEWAY_DEPLOY_STAGE="v2_pre_shutdown_preflight"
  export GATEWAY_DEPLOY_STAGE
  V2_PREFLIGHT_CREDENTIAL_ARGS=()
  for envelope in "${V2_CREDENTIAL_ENVELOPES[@]}"; do
    V2_PREFLIGHT_CREDENTIAL_ARGS+=(--credential-envelope "$envelope")
  done
  V2_PREFLIGHT_ACCEPTANCE_ARGS=()
  if [ -n "${GATEWAY_HISTORICAL_TOPOLOGY_HASH:-}" ]; then
    V2_PREFLIGHT_ACCEPTANCE_ARGS=(
      --acceptance-corpus-manifest \
        "$GATEWAY_V2_CONFIG_DIR/acceptance-corpus-v2.json"
      --acceptance-corpus-root \
        "$GATEWAY_V2_CONFIG_DIR/acceptance-corpus-v2"
    )
  fi
  if ! run_prepared_gateway_module gateway.tee.restart_preflight_v2 \
      --deploy-commit "$PREPARED_GATEWAY_SHA" \
      --release-manifest "$GATEWAY_PREPARED_V2_RELEASE_MANIFEST" \
      --topology-manifest "$GATEWAY_PREFLIGHT_TREE/gateway/tee/topology.json" \
      --artifact-policy "$GATEWAY_V2_ARTIFACT_POLICY" \
      --config-dir "$GATEWAY_V2_CONFIG_DIR" \
      --parent-env-file "$ENV_CLONE" \
      "${V2_PREFLIGHT_ACCEPTANCE_ARGS[@]}" \
      --topology-mode "${GATEWAY_TEE_TOPOLOGY_MODE:-full}" \
      "${V2_PREFLIGHT_CREDENTIAL_ARGS[@]}"; then
    rm -rf "$GATEWAY_PREFLIGHT_TREE"
    echo "ERROR: prepared V2 release failed before-shutdown validation" >&2
    exit 1
  fi
DOCKER_LOCK_HELPER="$GATEWAY_PREFLIGHT_TREE/validator_tee/scripts/docker_operation_lock_v2.sh"
if [ ! -r "$DOCKER_LOCK_HELPER" ]; then
  echo "ERROR: prepared Docker operation lock helper is unavailable" >&2
  exit 1
fi
. "$DOCKER_LOCK_HELPER"
leadpoet_acquire_docker_operation_lock_v2
run_prepared_gateway_module validator_tee.host.docker_operation_guard_v2 \
  --wait \
  --timeout-seconds 1800 \
  --interval-seconds 3
wait_for_foreign_docker_builds
wait_for_gateway_build_memory 1
record_gateway_restart_timing "pre_shutdown_checks_complete"

GATEWAY_DEPLOY_STAGE="lab_arena_claim_drain"
export GATEWAY_DEPLOY_STAGE
if ! drain_lab_arena_for_restart "$GATEWAY_PREFLIGHT_TREE"; then
  echo "ERROR: Lab Arena leases did not drain before gateway shutdown" >&2
  echo "Gateway remains running; production shutdown has not started." >&2
  exit 1
fi

echo "Rechecking shared miner maintenance at the destructive boundary"
GATEWAY_DEPLOY_STAGE="miner_maintenance_shutdown_verification"
export GATEWAY_DEPLOY_STAGE
if ! (
    set -a
    . "$ENV_CLONE"
    set +a
    run_prepared_gateway_module \
      gateway.tee.gateway_miner_maintenance_restart_v1 \
      --verify-shutdown-quiescence \
      --expected-commit "$PREPARED_GATEWAY_SHA"
  ); then
  echo "ERROR: shared miner-maintenance authority changed before shutdown" >&2
  echo "Gateway remains running; production shutdown has not started." >&2
  exit 1
fi

GATEWAY_DEPLOY_STAGE="lab_arena_destructive_authorization"
export GATEWAY_DEPLOY_STAGE
if ! run_lab_arena_restart_guard "$GATEWAY_PREFLIGHT_TREE" authorize \
    --generation "$LAB_ARENA_RESTART_GUARD_GENERATION" \
    --phase gateway_destructive >/dev/null; then
  echo "ERROR: Lab Arena drain changed before gateway shutdown" >&2
  echo "Gateway remains running; production shutdown has not started." >&2
  exit 1
fi

GATEWAY_LAB_ARENA_STOP_PROCESS_HELPER="$GATEWAY_CONTROLLER_PROCESS_HELPER"
if ! verify_controller_process_helper "$GATEWAY_LAB_ARENA_STOP_PROCESS_HELPER"; then
  echo "ERROR: verified controller Lab Arena stop helper is unavailable" >&2
  echo "Gateway remains running; production shutdown has not started." >&2
  exit 1
fi
echo "Stopping existing gateway and Research Lab worker processes"
GATEWAY_DESTRUCTIVE_PHASE_STARTED=1
export GATEWAY_DESTRUCTIVE_PHASE_STARTED
sudo systemctl stop leadpoet-tee-egress-forwarder.service 2>/dev/null || true
sudo systemctl reset-failed leadpoet-tee-egress-forwarder.service 2>/dev/null || true
pkill -9 -f "python3 main.py" 2>/dev/null || true
pkill -9 -f "python3 -u main.py" 2>/dev/null || true
pkill -9 -f "python3 -u -m gateway.main" 2>/dev/null || true
pkill -9 -f "uvicorn" 2>/dev/null || true
pkill -9 -f "run_research_lab_hosted_worker" 2>/dev/null || true
pkill -9 -f "/gateway/research_lab/worker_process[.]py" 2>/dev/null || true
pkill -9 -f "gateway.research_lab.provider_evidence_proxy" 2>/dev/null || true
pkill -9 -f "provider_evidence_proxy" 2>/dev/null || true
pkill -9 -f "gateway.utils.tee_inter_enclave_relay" 2>/dev/null || true
pkill -9 -f "gateway.utils.tee_egress_forwarder" 2>/dev/null || true
stop_lab_arena_service "$GATEWAY_LAB_ARENA_STOP_PROCESS_HELPER"
rm -rf "$GATEWAY_PREFLIGHT_TREE"
GATEWAY_PREFLIGHT_TREE=""

wait_for_exact_migration_203

echo "Stopping stuck local validator Docker builds or pip installs"
stop_local_stale_build_processes TERM
sleep 3
stop_local_stale_build_processes KILL
sleep 2

echo "Confirming real build memory after gateway shutdown"
if ! wait_for_gateway_build_memory 0 10; then
  echo "ERROR: gateway shutdown did not recover the required build memory" >&2
  exit 1
fi

echo "Waiting for :8000 to free"
for i in $(seq 1 25); do
  if ! sudo ss -tulpn 2>/dev/null | grep -q ":8000 "; then
    echo ":8000 free after ${i}s"
    break
  fi
  sleep 1
done

echo "Activating prepared gateway Git commit after process shutdown"
GATEWAY_DEPLOY_STAGE="git_activate"
export GATEWAY_DEPLOY_STAGE
ACTIVATED_GATEWAY_SHA="$(
  python3 "$GATEWAY_GIT_HELPER" activate \
    --plan-file "$GATEWAY_DEPLOY_PLAN_FILE"
)"
if [ "$ACTIVATED_GATEWAY_SHA" != "$PREPARED_GATEWAY_SHA" ]; then
  echo "ERROR: activated gateway commit differs from prepared commit" >&2
  exit 1
fi
echo "Activated gateway commit: $ACTIVATED_GATEWAY_SHA"

GATEWAY_POST_ACTIVATE_REEXEC_SCRIPT="$LEADPOET_REPO_ROOT/gw_restart.sh"
if [ -n "$GATEWAY_RESTART_AUTHORITY_ROOT" ]; then
  GATEWAY_POST_ACTIVATE_REEXEC_SCRIPT="$GATEWAY_RESTART_AUTHORITY_ROOT/gw_restart.sh"
fi
if [ ! -r "$GATEWAY_POST_ACTIVATE_REEXEC_SCRIPT" ] \
    || [ -L "$GATEWAY_POST_ACTIVATE_REEXEC_SCRIPT" ]; then
  echo "ERROR: exact restart authority disappeared before post-activation reexec" >&2
  exit 1
fi

GATEWAY_DEPLOY_STAGE="restart_reexec"
export GATEWAY_DEPLOY_STAGE
unset GATEWAY_DEPLOY_COMMIT
exec env \
  GATEWAY_RESTART_PHASE=post_activate \
  GATEWAY_RESTART_LOCK_HELD=1 \
  LEADPOET_REPO_ROOT="$LEADPOET_REPO_ROOT" \
  GATEWAY_ROOT="$GATEWAY_ROOT" \
  GATEWAY_LOG_ROOT="$GATEWAY_LOG_ROOT" \
  GATEWAY_LOG_FILE="$GATEWAY_LOG_FILE" \
  GATEWAY_ENV_FILE="$GATEWAY_ENV_FILE" \
  GATEWAY_PRIVATE_KEY_PATH="$GATEWAY_PRIVATE_KEY_PATH" \
  ARWEAVE_KEYFILE_PATH="$ARWEAVE_KEYFILE_PATH" \
  GATEWAY_RESTART_GIT_SSH_COMMAND="$GATEWAY_RESTART_GIT_SSH_COMMAND" \
  GATEWAY_GIT_HELPER="$GATEWAY_GIT_HELPER" \
  GATEWAY_RESTART_CONTROLLER_ROOT="$GATEWAY_RESTART_CONTROLLER_ROOT" \
  GATEWAY_RESTART_AUTHORITY_ROOT="$GATEWAY_RESTART_AUTHORITY_ROOT" \
  GATEWAY_RESTART_AUTHORITY_COMMIT="$GATEWAY_RESTART_AUTHORITY_COMMIT" \
  GATEWAY_ACTIVE_RELEASE_RESTART_INVOCATION_ID="$GATEWAY_ACTIVE_RELEASE_RESTART_INVOCATION_ID" \
  GATEWAY_ACTIVE_RELEASE_COMPONENT="$GATEWAY_ACTIVE_RELEASE_COMPONENT" \
  GATEWAY_DEPLOY_PLAN_FILE="$GATEWAY_DEPLOY_PLAN_FILE" \
  GATEWAY_DEPLOYMENT_DIR="$GATEWAY_DEPLOYMENT_DIR" \
  GATEWAY_DEPLOYMENT_MANIFEST="$GATEWAY_DEPLOYMENT_MANIFEST" \
  GATEWAY_LAST_GOOD_MANIFEST="$GATEWAY_LAST_GOOD_MANIFEST" \
  GATEWAY_HOST_RESTART_SCRIPT="$POST_ACTIVATE_GATEWAY_HOST_RESTART_SCRIPT" \
  LEADPOET_DOCKER_OPERATION_LOCK_FILE="$LEADPOET_DOCKER_OPERATION_LOCK_FILE" \
  LEADPOET_DOCKER_OPERATION_LOCK_HELD=1 \
  GATEWAY_TEE_EIF_ROOT="$GATEWAY_TEE_EIF_ROOT" \
  GATEWAY_V2_RELEASE_ARCHIVE_ROOT="$GATEWAY_V2_RELEASE_ARCHIVE_ROOT" \
  GATEWAY_PYTHON_BIN="$GATEWAY_PYTHON_BIN" \
  GATEWAY_DEPENDENCY_INSTALL_FINGERPRINT="$GATEWAY_DEPENDENCY_INSTALL_FINGERPRINT" \
  GATEWAY_RESTART_STARTED_EPOCH="$GATEWAY_RESTART_STARTED_EPOCH" \
  GATEWAY_RESTART_INVOCATION_ID="${GATEWAY_RESTART_INVOCATION_ID:-gateway-${GATEWAY_RESTART_STARTED_EPOCH:-unknown}-$$}" \
  GATEWAY_RELEASE_ATTEMPTS_USED="${GATEWAY_RELEASE_ATTEMPTS_USED:-0}" \
  GATEWAY_DESTRUCTIVE_PHASE_STARTED="$GATEWAY_DESTRUCTIVE_PHASE_STARTED" \
  PREPARED_GATEWAY_SHA="$PREPARED_GATEWAY_SHA" \
  LAB_ARENA_RESTART_GUARD_GENERATION="$LAB_ARENA_RESTART_GUARD_GENERATION" \
  GATEWAY_RESTART_TIMING_DIR="$GATEWAY_RESTART_TIMING_DIR" \
  GATEWAY_RESTART_TIMING_FILE="$GATEWAY_RESTART_TIMING_FILE" \
  GATEWAY_RESTART_TIMING_INITIALIZED="$GATEWAY_RESTART_TIMING_INITIALIZED" \
  GATEWAY_RELEASE_FOLLOW_ROOT="$GATEWAY_RELEASE_FOLLOW_ROOT" \
  GATEWAY_RELEASE_SUPERSESSION_COUNT="$GATEWAY_RELEASE_SUPERSESSION_COUNT" \
  GATEWAY_RELEASE_SUPERSESSION_MAX="$GATEWAY_RELEASE_SUPERSESSION_MAX" \
  RESEARCH_LAB_TEE_PROTOCOL="$RESEARCH_LAB_TEE_PROTOCOL" \
  GATEWAY_V2_CONFIG_DIR="$GATEWAY_V2_CONFIG_DIR" \
  GATEWAY_V2_RELEASE_MANIFEST="$GATEWAY_V2_RELEASE_MANIFEST" \
  GATEWAY_V2_RELEASE_LINEAGE="$GATEWAY_V2_RELEASE_LINEAGE" \
  GATEWAY_PREPARED_V2_RELEASE_MANIFEST="$GATEWAY_PREPARED_V2_RELEASE_MANIFEST" \
  GATEWAY_PREPARED_V2_RELEASE_LINEAGE="$GATEWAY_PREPARED_V2_RELEASE_LINEAGE" \
  GATEWAY_V2_RELEASE_BUCKET="$GATEWAY_V2_RELEASE_BUCKET" \
  GATEWAY_V2_RELEASE_PREFIX="$GATEWAY_V2_RELEASE_PREFIX" \
  GATEWAY_V2_ARTIFACT_POLICY="$GATEWAY_V2_ARTIFACT_POLICY" \
  RESEARCH_LAB_ATTESTED_V2_ARTIFACT_BUCKET="$RESEARCH_LAB_ATTESTED_V2_ARTIFACT_BUCKET" \
  GATEWAY_V2_OFFLINE_ARTIFACT_ROOT="$GATEWAY_V2_OFFLINE_ARTIFACT_ROOT" \
  GATEWAY_DEPLOY_STAGE="$GATEWAY_DEPLOY_STAGE" \
  bash "$GATEWAY_POST_ACTIVATE_REEXEC_SCRIPT" "$@"
fi

bind_activated_gateway_guard_candidate || exit 1
record_gateway_restart_timing "candidate_activated"
echo "Installing the preflighted gateway manifest and lineage"
PYTHONPATH="$LEADPOET_REPO_ROOT" "$GATEWAY_PYTHON_BIN" \
  -m gateway.tee.install_gateway_release_state_v2 \
  --prepared-manifest "$GATEWAY_PREPARED_V2_RELEASE_MANIFEST" \
  --prepared-lineage "$GATEWAY_PREPARED_V2_RELEASE_LINEAGE" \
  --active-manifest "$GATEWAY_V2_RELEASE_MANIFEST" \
  --active-lineage "$GATEWAY_V2_RELEASE_LINEAGE" \
  --expected-commit "$GATEWAY_DEPLOY_SHA"
echo "Cleaning stale read-only gateway vsock probes"
"$GATEWAY_PYTHON_BIN" \
  "$LEADPOET_REPO_ROOT/gateway/tee/host_memory_guard_v2.py" \
  --cleanup-stale-vsock-probes \
  --minimum-available-mib 1024
echo "Verifying prepared and activated gateway trees against exact Git blobs"
GATEWAY_DEPLOY_STAGE="git_tree_verification"
export GATEWAY_DEPLOY_STAGE
"$GATEWAY_PYTHON_BIN" \
  "$LEADPOET_REPO_ROOT/scripts/gateway_git_deploy.py" \
  verify-tree-pair \
  --plan-file "$GATEWAY_DEPLOY_PLAN_FILE" \
  --prepared-evidence \
    "$GATEWAY_V2_CONFIG_DIR/gateway-candidate-tree-preflight.json" \
  --activated-root "$LEADPOET_REPO_ROOT"
enforce_deployment_environment

echo "Recording exact gateway Git build provenance"
GATEWAY_DEPLOY_STAGE="build_provenance"
export GATEWAY_DEPLOY_STAGE
python3 "$LEADPOET_REPO_ROOT/scripts/write_gateway_build_info.py" \
  --repo-root "$LEADPOET_REPO_ROOT" \
  --output "$GATEWAY_ROOT/BUILD_INFO.json" \
  --require-git-commit
printf '%s\n' "$GATEWAY_DEPLOY_SHA" > "$GATEWAY_ROOT/.source_commit"

echo "Clearing Python caches"
GATEWAY_DEPLOY_STAGE="python_cache_cleanup"
export GATEWAY_DEPLOY_STAGE
cd "$GATEWAY_ROOT"
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -delete 2>/dev/null || true
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
rm -rf ~/.cache/Python* 2>/dev/null || true
find ~/.local/lib/python3.9/site-packages -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

echo "Preflight disk cleanup for Docker/PCR0/Research Lab builds"
GATEWAY_DEPLOY_STAGE="docker_disk_cleanup"
export GATEWAY_DEPLOY_STAGE
ensure_docker_ready
df -h / /var/lib/docker 2>/dev/null || df -h /
sudo journalctl --vacuum-size=200M 2>/dev/null || true
run_bounded_restart_artifact_cleanup

sudo docker container prune -f 2>/dev/null || true
sudo docker builder prune -af 2>/dev/null || true
sudo docker system prune -af --volumes 2>/dev/null || true

FREE_KB_AFTER_PRUNE="$(df --output=avail / | tail -1 | tr -d ' ')"
reset_orphaned_docker_storage_if_needed "$FREE_KB_AFTER_PRUNE" "stale Docker storage after cleanup"

echo "Disk after cleanup"
df -h / /var/lib/docker 2>/dev/null || df -h /
sudo docker system df 2>/dev/null || true

FREE_KB="$(df --output=avail / | tail -1 | tr -d ' ')"
if [ "${FREE_KB:-0}" -lt "$MIN_FREE_KB" ]; then
  echo "ERROR: insufficient free disk after cleanup: $(df -h / | tail -1)"
  echo "Need at least 10GiB free before starting gateway Research Lab Docker workloads."
  exit 1
fi

echo "Loading gateway runtime env for AWS/ECR checks"
GATEWAY_DEPLOY_STAGE="runtime_env_and_ecr"
export GATEWAY_DEPLOY_STAGE
set -a
. "$ENV_CLONE"
set +a
restore_gateway_restart_path_authority
enforce_deployment_environment
validate_runtime_secret_paths
GATEWAY_DEPLOY_STAGE="runtime_env_and_ecr"
export GATEWAY_DEPLOY_STAGE
export AWS_REGION="${AWS_REGION:-us-east-1}"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"
unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_PROFILE AWS_SESSION_TOKEN AWS_SECURITY_TOKEN

ACTUAL_AWS_ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
if [ "$ACTUAL_AWS_ACCOUNT" != "$EXPECTED_AWS_ACCOUNT" ]; then
  echo "ERROR: gateway AWS account is $ACTUAL_AWS_ACCOUNT, expected $EXPECTED_AWS_ACCOUNT"
  exit 1
fi

aws ecr get-login-password --region "$AWS_REGION" | sudo docker login --username AWS --password-stdin "${EXPECTED_AWS_ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com"

echo "Building/restarting TEE enclave"
GATEWAY_DEPLOY_STAGE="attested_runtime_and_enclave_build"
export GATEWAY_DEPLOY_STAGE
cd "$GATEWAY_ROOT/tee"
sudo mkdir -p "$GATEWAY_TEE_EIF_ROOT"
rm -f "$GATEWAY_ROOT/tee/tee-enclave.eif"
sudo docker rmi tee-enclave:latest 2>/dev/null || true
bash "$GATEWAY_ROOT/tee/stage_attested_runtime.sh"
record_gateway_restart_timing "attested_runtime_staged"

# Preflight: verify the runtime import graph against the freshly staged
# runtime before building the enclave or relaunching anything.
# A gateway/ tree that imports names the staged top-level packages do not
# export would otherwise crash-loop every worker on its next respawn
# This catches missing or incorrectly staged imports before shutdown.
echo "Preflight: importing gateway dependencies from the canonical Git checkout"
GATEWAY_DEPLOY_STAGE="dependency_import_preflight"
export GATEWAY_DEPLOY_STAGE
if ! PYTHONSAFEPATH=1 LEADPOET_REPO_ROOT="$LEADPOET_REPO_ROOT" PYTHONPATH="$LEADPOET_REPO_ROOT" "$GATEWAY_PYTHON_BIN" - <<'PREFLIGHT_HOST'
import importlib
import os
from pathlib import Path

import bittensor as bt
from async_substrate_interface import SubstrateInterface
import scalecodec

if str(bt.__version__) != "10.5.0":
    raise RuntimeError(f"gateway Bittensor SDK mismatch: {bt.__version__}")
repo_root = Path(os.environ["LEADPOET_REPO_ROOT"]).resolve()
modules = (
    "gateway.research_lab.config",
    "leadpoet_canonical",
    "qualification",
    "validator_models",
    "Leadpoet",
)
for module_name in modules:
    module = importlib.import_module(module_name)
    origin = Path(module.__file__).resolve()
    if not origin.is_relative_to(repo_root):
        raise RuntimeError(f"{module_name} resolved outside canonical checkout: {origin}")
print("canonical host imports OK")
PREFLIGHT_HOST
then
  echo "ERROR: gateway dependency import preflight failed against canonical Git checkout." >&2
  exit 1
fi

echo "Preflight: importing gateway dependencies from staged attested runtime"
if ! GATEWAY_ROOT="$GATEWAY_ROOT" LEADPOET_REPO_ROOT="$LEADPOET_REPO_ROOT" "$GATEWAY_PYTHON_BIN" - <<'PREFLIGHT_ATTESTED'
import importlib
import os
import sys
from pathlib import Path

gateway_root = Path(os.environ["GATEWAY_ROOT"]).resolve()
repo_root = Path(os.environ["LEADPOET_REPO_ROOT"]).resolve()
attested_root = (gateway_root / "_attested_runtime").resolve()
sys.path = [str(attested_root), str(repo_root)] + [
    path for path in sys.path if path not in {str(attested_root), str(repo_root)}
]
importlib.import_module("gateway.research_lab.config")
for module_name in (
    "leadpoet_canonical",
    "qualification",
    "validator_models",
):
    module = importlib.import_module(module_name)
    origin = Path(module.__file__).resolve()
    if not origin.is_relative_to(attested_root):
        raise RuntimeError(f"{module_name} resolved outside staged runtime: {origin}")
print("staged attested imports OK")
PREFLIGHT_ATTESTED
then
  echo "ERROR: gateway dependencies and staged attested runtime are out of sync." >&2
  exit 1
fi
echo "Building deterministic gateway role EIFs from the staged runtime"
  GATEWAY_TEE_SKIP_STAGE=1 bash "$GATEWAY_ROOT/tee/build_role_enclaves.sh"
  record_gateway_restart_timing "gateway_role_eifs_built"
  echo "Cleaning temporary role Docker images/layers before gateway relaunch"
  for role in gateway_autoresearch gateway_coordinator gateway_scoring; do
    sudo docker rmi -f "tee-enclave:${role}" 2>/dev/null || true
  done
  sudo docker builder prune -af 2>/dev/null || true
  df -h / /var/lib/docker 2>/dev/null || df -h /
  echo "Configuring Nitro allocator for the measured gateway topology"
  bash "$GATEWAY_ROOT/tee/configure_allocator.sh"
  sudo env \
    GATEWAY_ROOT="$GATEWAY_ROOT" \
    GATEWAY_TEE_EIF_ROOT="$GATEWAY_TEE_EIF_ROOT" \
    GATEWAY_ENV_FILE="$GATEWAY_ENV_FILE" \
    RESEARCH_LAB_TEE_PROTOCOL="$RESEARCH_LAB_TEE_PROTOCOL" \
    bash ./start_enclave.sh
  record_gateway_restart_timing "gateway_enclaves_started"

  echo "Starting parent-side opaque enclave egress forwarder"
  cd "$LEADPOET_REPO_ROOT"
  env -u GATEWAY_MINER_MAINTENANCE_PROOF_FD \
    -u GATEWAY_GIT_HELPER \
    -u GATEWAY_HOST_MEMORY_GUARD_PATH \
    -u GATEWAY_RESTART_AUTHORITY_ROOT \
    -u GATEWAY_RESTART_AUTHORITY_COMMIT \
    -u GATEWAY_ACTIVE_RELEASE_RESTART_INVOCATION_ID \
    -u GATEWAY_ACTIVE_RELEASE_COMPONENT \
    PYTHONPATH="$LEADPOET_REPO_ROOT" \
    setsid "$GATEWAY_PYTHON_BIN" -u -m gateway.utils.tee_egress_forwarder \
    >> "$GATEWAY_LOG_ROOT/tee_egress_forwarder.log" 2>&1 < /dev/null \
    7>&- 8>&- 9>&- 190>&- 191>&- 192>&- 193>&- 194>&- 195>&- &
  TEE_EGRESS_FORWARDER_PID="$!"
  sleep 2
  if ! ps -p "$TEE_EGRESS_FORWARDER_PID" >/dev/null 2>&1; then
    tail -80 "$GATEWAY_LOG_ROOT/tee_egress_forwarder.log" || true
    echo "ERROR: parent-side enclave egress forwarder did not start" >&2
    exit 1
  fi

  echo "Starting opaque inter-enclave TLS relay"
  cd "$LEADPOET_REPO_ROOT"
  env -u GATEWAY_MINER_MAINTENANCE_PROOF_FD \
    -u GATEWAY_GIT_HELPER \
    -u GATEWAY_HOST_MEMORY_GUARD_PATH \
    -u GATEWAY_CONTROLLER_PROCESS_HELPER \
    -u LAB_ARENA_PROCESS_HELPER \
    -u GATEWAY_RESTART_AUTHORITY_ROOT \
    -u GATEWAY_RESTART_AUTHORITY_COMMIT \
    -u GATEWAY_ACTIVE_RELEASE_RESTART_INVOCATION_ID \
    -u GATEWAY_ACTIVE_RELEASE_COMPONENT \
    PYTHONPATH="$LEADPOET_REPO_ROOT" \
    setsid "$GATEWAY_PYTHON_BIN" -m gateway.utils.tee_inter_enclave_relay \
    >> "$GATEWAY_LOG_ROOT/inter_enclave_relay.log" 2>&1 < /dev/null \
    7>&- 8>&- 9>&- 190>&- 191>&- 192>&- 193>&- 194>&- 195>&- &
  INTER_ENCLAVE_RELAY_PID="$!"
  sleep 2
  if ! ps -p "$INTER_ENCLAVE_RELAY_PID" >/dev/null 2>&1; then
    tail -80 "$GATEWAY_LOG_ROOT/inter_enclave_relay.log" || true
    echo "ERROR: inter-enclave relay did not start" >&2
    exit 1
  fi

  echo "Bootstrapping mutually attested V2 enclave runtime"
  GATEWAY_DEPLOY_STAGE="v2_runtime_bootstrap"
  export GATEWAY_DEPLOY_STAGE
  test -s "$GATEWAY_V2_RELEASE_MANIFEST" || {
    echo "ERROR: local V2 build identity is missing" >&2
    exit 1
  }
  test -s "$GATEWAY_V2_RELEASE_LINEAGE" || {
    echo "ERROR: local V2 build lineage is missing" >&2
    exit 1
  }
  test -s "$GATEWAY_V2_ARTIFACT_POLICY" || {
    echo "ERROR: encrypted V2 artifact policy is missing" >&2
    exit 1
  }
  echo "Verifying the encrypted TLS proxy profile for the V2 scoring worker"
  PYTHONPATH="$LEADPOET_REPO_ROOT" "$GATEWAY_PYTHON_BIN" -m gateway.research_lab.provider_profiles_v2 \
    --config-dir "$GATEWAY_V2_CONFIG_DIR" \
    --require-worker-proxies
  V2_BOOTSTRAP_ARGS=()
  V2_PROVISION_ARGS=()
  for envelope in "${V2_CREDENTIAL_ENVELOPES[@]}"; do
    test -s "$envelope" || {
      echo "ERROR: encrypted V2 credential envelope is missing: $envelope" >&2
      exit 1
    }
    V2_BOOTSTRAP_ARGS+=(--credential-envelope "$envelope")
    V2_PROVISION_ARGS+=(--envelope "$envelope")
  done
  PYTHONPATH="$LEADPOET_REPO_ROOT" "$GATEWAY_PYTHON_BIN" -m gateway.utils.tee_v2_bootstrap \
    --release-manifest "$GATEWAY_V2_RELEASE_MANIFEST" \
    --gateway-release-lineage "$GATEWAY_V2_RELEASE_LINEAGE" \
    "${V2_BOOTSTRAP_ARGS[@]}" \
    --protected-workflow-manifest "$GATEWAY_ROOT/_attested_runtime/protected_workflows.json" \
    --encrypted-artifact-policy "$GATEWAY_V2_ARTIFACT_POLICY" \
    --config-dir "$GATEWAY_V2_CONFIG_DIR"
  record_gateway_restart_timing "v2_runtime_bootstrapped"

  echo "Provisioning KMS ciphertext directly to the attested coordinator"
  GATEWAY_DEPLOY_STAGE="v2_kms_provision"
  export GATEWAY_DEPLOY_STAGE
  PYTHONPATH="$LEADPOET_REPO_ROOT" "$GATEWAY_PYTHON_BIN" -m gateway.utils.tee_kms_provision_v2 \
    "${V2_PROVISION_ARGS[@]}"
  record_gateway_restart_timing "v2_kms_provisioned"

echo "Verifying V2 provider and execution-manager readiness"
GATEWAY_DEPLOY_STAGE="v2_runtime_readiness"
export GATEWAY_DEPLOY_STAGE
PYTHONPATH="$LEADPOET_REPO_ROOT" "$GATEWAY_PYTHON_BIN" -m gateway.tee.verify_v2_runtime_ready
record_gateway_restart_timing "v2_runtime_ready"

echo "Installing Python dependencies"
GATEWAY_DEPLOY_STAGE="dependency_install"
export GATEWAY_DEPLOY_STAGE
cd "$GATEWAY_ROOT"
install_gateway_python_dependencies

echo "Relaunching gateway with cloned runtime env"
GATEWAY_DEPLOY_STAGE="gateway_process_launch"
export GATEWAY_DEPLOY_STAGE
set -a
. "$ENV_CLONE"
set +a
restore_gateway_restart_path_authority
enforce_deployment_environment
validate_runtime_secret_paths
export PATH="$HOME/.local/bin:$PATH"
export AWS_REGION="${AWS_REGION:-us-east-1}"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"
export GATEWAY_ENV_FILE="${GATEWAY_ENV_FILE:-/home/ec2-user/.config/leadpoet/gateway.env}"
export LEADPOET_GATEWAY_ENV_SECRET_ID="${LEADPOET_GATEWAY_ENV_SECRET_ID:-leadpoet/prod/gateway/env}"
unset RESEARCH_LAB_EVIDENCE_PROXY_URL
unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_PROFILE AWS_SESSION_TOKEN AWS_SECURITY_TOKEN
export LEADPOET_AWS_INSTANCE_ROLE_ONLY=true

# Release the build lock after the protected gateway runtime is ready.
. "$LEADPOET_REPO_ROOT/validator_tee/scripts/docker_operation_lock_v2.sh"
leadpoet_release_docker_operation_lock_v2

cd "$LEADPOET_REPO_ROOT"
env -u GATEWAY_MINER_MAINTENANCE_PROOF_FD \
  -u GATEWAY_GIT_HELPER \
  -u GATEWAY_HOST_MEMORY_GUARD_PATH \
  -u GATEWAY_CONTROLLER_PROCESS_HELPER \
  -u LAB_ARENA_PROCESS_HELPER \
  -u GATEWAY_RESTART_AUTHORITY_ROOT \
  -u GATEWAY_RESTART_AUTHORITY_COMMIT \
  -u GATEWAY_ACTIVE_RELEASE_RESTART_INVOCATION_ID \
  -u GATEWAY_ACTIVE_RELEASE_COMPONENT \
  -u PREPARED_GATEWAY_SHA \
  -u LAB_ARENA_RESTART_GUARD_GENERATION \
  setsid "$GATEWAY_PYTHON_BIN" -u -m gateway.main \
  > "$GATEWAY_LOG_FILE" 2>&1 < /dev/null \
  9>&- 190>&- 191>&- 192>&- 193>&- 194>&- 195>&- &

GATEWAY_LAUNCHER_PID="$!"
GATEWAY_PID=""
echo "gateway launcher pid: $GATEWAY_LAUNCHER_PID"
for attempt in $(seq 1 15); do
  GATEWAY_PID="$(pgrep -f "^$GATEWAY_PYTHON_BIN -u -m gateway[.]main$" | head -1 || true)"
  if [ -n "$GATEWAY_PID" ]; then
    break
  fi
  sleep 1
done
if [ -z "$GATEWAY_PID" ]; then
  tail -160 "$GATEWAY_LOG_FILE"
  echo "ERROR: gateway process was not discoverable after launch" >&2
  exit 1
fi
echo "relaunched main pid: $GATEWAY_PID"
rm -f "$ENV_CLONE" "$ENV_SECRET"

GATEWAY_DEPLOY_STAGE="gateway_health_check"
export GATEWAY_DEPLOY_STAGE
GATEWAY_HEALTH_READY=0
for attempt in $(seq 1 120); do
  GATEWAY_PID="$(pgrep -f "^$GATEWAY_PYTHON_BIN -u -m gateway[.]main$" | head -1 || true)"
  if [ -z "$GATEWAY_PID" ]; then
    tail -160 "$GATEWAY_LOG_FILE"
    echo "ERROR: gateway exited during startup" >&2
    exit 1
  fi
  if timeout 5 curl -fsS http://localhost:8000/health >/dev/null 2>&1; then
    GATEWAY_HEALTH_READY=1
    echo "Gateway base health ready after attempt $attempt"
    break
  fi
  sleep 5
done
if [ "$GATEWAY_HEALTH_READY" != "1" ]; then
  tail -120 "$GATEWAY_LOG_FILE"
  echo "ERROR: gateway base health did not become ready within 10 minutes" >&2
  exit 1
fi
record_gateway_restart_timing "gateway_base_health_ready"
if ! wait_for_gateway_v2_authority; then
  exit 1
fi
record_gateway_restart_timing "gateway_v2_health_ready"
GATEWAY_DEPLOY_STAGE="lab_arena_service_start"
export GATEWAY_DEPLOY_STAGE
start_lab_arena_service
record_gateway_restart_timing "lab_arena_service_ready"
if ! run_lab_arena_restart_guard "$LEADPOET_REPO_ROOT" ready \
    --generation "$LAB_ARENA_RESTART_GUARD_GENERATION" \
    --phase gateway_ready >/dev/null; then
  echo "ERROR: Lab Arena gateway readiness could not be recorded" >&2
  exit 1
fi
BUILD_INFO_RESPONSE="$(timeout 15 curl -fsS http://localhost:8000/build-info)"
python3 - "$GATEWAY_DEPLOY_SHA" "$BUILD_INFO_RESPONSE" <<'VERIFY_BUILD_INFO'
import json
import sys

expected = sys.argv[1]
payload = json.loads(sys.argv[2])
actual = str(payload.get("git_commit") or "").lower()
if actual != expected:
    raise SystemExit(f"gateway /build-info commit mismatch: expected {expected}, got {actual}")
print(f"verified gateway /build-info commit: {actual}")
VERIFY_BUILD_INFO

echo "Verifying gateway attestation status"
timeout 30 curl -fsS http://localhost:8000/attest >/dev/null

GATEWAY_DEPLOY_STAGE="host_restart_script_install"
export GATEWAY_DEPLOY_STAGE
install_successful_restart_script

GATEWAY_DEPLOY_STAGE="miner_maintenance_runtime_verify"
export GATEWAY_DEPLOY_STAGE
echo "Revalidating exact-candidate miner maintenance state against the live runtime"
if ! PYTHONPATH="$LEADPOET_REPO_ROOT" "$GATEWAY_PYTHON_BIN" \
    -m gateway.tee.gateway_miner_maintenance_restart_v1 \
    --verify-runtime \
    --expected-commit "$GATEWAY_DEPLOY_SHA" \
    --repo-root "$LEADPOET_REPO_ROOT" \
    --release-manifest "$GATEWAY_V2_RELEASE_MANIFEST"; then
  stop_failed_miner_maintenance_runtime
  exit 1
fi
GATEWAY_DEPLOY_STAGE="lab_arena_claim_guard_release"
export GATEWAY_DEPLOY_STAGE
if ! run_lab_arena_restart_guard "$LEADPOET_REPO_ROOT" release \
    --generation "$LAB_ARENA_RESTART_GUARD_GENERATION" >/dev/null; then
  echo "ERROR: Lab Arena restart guard release failed after gateway readiness" >&2
  exit 1
fi
LAB_ARENA_RESTART_GUARD_GENERATION=""
GATEWAY_DEPLOY_STAGE="completed"
export GATEWAY_DEPLOY_STAGE
finalize_deployment_record succeeded "$GATEWAY_DEPLOY_STAGE" >/dev/null
if [ -n "${GATEWAY_MINER_MAINTENANCE_PROOF_FD:-}" ]; then
  exec 190>&- 191>&- 192>&- 193>&- 194>&- 195>&-
  unset GATEWAY_MINER_MAINTENANCE_PROOF_FD
  unset GATEWAY_HOST_MEMORY_GUARD_PATH
  unset GATEWAY_CONTROLLER_PROCESS_HELPER LAB_ARENA_PROCESS_HELPER
fi
GATEWAY_DEPLOY_COMPLETED=1
rm -f "$GATEWAY_DEPLOY_PLAN_FILE" || true
record_gateway_restart_timing "completed" "passed"
echo "Gateway restart command completed; tail logs with: tail -f $GATEWAY_LOG_FILE"
