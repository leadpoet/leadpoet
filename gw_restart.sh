#!/bin/bash
set -euo pipefail

GATEWAY_GIT_DEPLOY_PROTOCOL="1"
LEADPOET_REPO_ROOT="${LEADPOET_REPO_ROOT:-/home/ec2-user/leadpoet_repo}"
GATEWAY_ROOT="${GATEWAY_ROOT:-$LEADPOET_REPO_ROOT/gateway}"
GATEWAY_LOG_ROOT="${GATEWAY_LOG_ROOT:-/home/ec2-user/gateway}"
GATEWAY_LOG_FILE="${GATEWAY_LOG_FILE:-$GATEWAY_LOG_ROOT/gateway.log}"
LAB_ARENA_SERVICE_LOG_FILE="${LAB_ARENA_SERVICE_LOG_FILE:-$GATEWAY_LOG_ROOT/lab_arena_service.log}"
LAB_ARENA_SERVICE_STATE_FILE="${LAB_ARENA_SERVICE_STATE_FILE:-/home/ec2-user/.config/leadpoet/lab-arena-service-process.json}"
ARENA_RUNTIME_ROOT="/home/ec2-user/.config/leadpoet/arena-runtime"
ARENA_INGRESS_STATE_FILE="/etc/caddy/arena-ingress-state.json"
ARENA_CADDY_FILE="/etc/caddy/Caddyfile"
ARENA_CONTINUITY_ENABLED="${ARENA_CONTINUITY_ENABLED:-0}"
ARENA_CANDIDATE_SLOT="${ARENA_CANDIDATE_SLOT:-}"
ARENA_CANDIDATE_TREE="${ARENA_CANDIDATE_TREE:-}"
ARENA_PREVIOUS_MODE="${ARENA_PREVIOUS_MODE:-}"
ARENA_PREVIOUS_SLOT="${ARENA_PREVIOUS_SLOT:-}"
ARENA_PREVIOUS_SHA="${ARENA_PREVIOUS_SHA:-}"
ARENA_PREVIOUS_BACKEND="${ARENA_PREVIOUS_BACKEND:-}"
ARENA_RECOVERING=0
ARENA_REUSE_PREVIOUS=0
ARENA_INGRESS_SWITCHED=0
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
# Opt-in paired deployment retains the existing all-component guard after gateway readiness.
GATEWAY_PAIRED_RESTART="${GATEWAY_PAIRED_RESTART:-0}"
case "$GATEWAY_PAIRED_RESTART" in 0|1) ;; *) echo "ERROR: invalid paired restart mode" >&2; exit 2 ;; esac
GATEWAY_RESTART_GUARD_SCOPE="$GATEWAY_ACTIVE_RELEASE_COMPONENT"
if [ "$GATEWAY_PAIRED_RESTART" = 1 ]; then
  [ "$GATEWAY_ACTIVE_RELEASE_COMPONENT" = gateway ] || exit 2
  GATEWAY_RESTART_GUARD_SCOPE=all
fi
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
GATEWAY_ACTIVE_RELEASE_RESTART_INVOCATION_ID="${GATEWAY_ACTIVE_RELEASE_RESTART_INVOCATION_ID:-$GATEWAY_RESTART_INVOCATION_ID}"
GATEWAY_RELEASE_ATTEMPTS_USED="${GATEWAY_RELEASE_ATTEMPTS_USED:-0}"
GATEWAY_RESTART_TIMING_DIR="${GATEWAY_RESTART_TIMING_DIR:-/home/ec2-user/.config/leadpoet/restart-timings}"
GATEWAY_RESTART_TIMING_FILE="${GATEWAY_RESTART_TIMING_FILE:-$GATEWAY_RESTART_TIMING_DIR/gateway-${GATEWAY_RESTART_STARTED_EPOCH}-$$.jsonl}"
GATEWAY_RESTART_TIMING_INITIALIZED="${GATEWAY_RESTART_TIMING_INITIALIZED:-0}"
PREPARED_GATEWAY_SHA="${PREPARED_GATEWAY_SHA:-}"
LAB_ARENA_RESTART_GUARD_GENERATION="${LAB_ARENA_RESTART_GUARD_GENERATION:-}"
LAB_ARENA_RESTART_DRAIN_PID=""
LAB_ARENA_RESTART_DRAIN_REPORT=""
LAB_ARENA_RESTART_DRAIN_PRIOR_JOBS=""
if [ -n "$GATEWAY_RESTART_AUTHORITY_ROOT" ]; then
  if ! [[ "$GATEWAY_RESTART_AUTHORITY_ROOT" =~ ^/tmp/gateway-restart-controller-bootstrap\.[A-Za-z0-9]+/authority$ ]] \
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
  local attempt deadline health_url
  health_url="http://localhost:8000"
  if [ "$ARENA_CONTINUITY_ENABLED" = 1 ]; then
    health_url="http://127.0.0.1:8001"
  fi
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
    if timeout 5 curl -fsS "$health_url/health/v2-authority" >/dev/null 2>&1; then
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

arena_ingress_tool() {
  local source_root="$1"
  shift
  sudo "$GATEWAY_PYTHON_BIN" -B "$source_root/scripts/arena_ingress_config.py" \
    "$@" --state-file "$ARENA_INGRESS_STATE_FILE" --caddy-file "$ARENA_CADDY_FILE"
}

arena_ingress_transition() {
  local source_root="$1" old_mode="$2" old_slot="$3" old_sha="$4" old_backend="$5"
  local new_mode="$6" new_slot="$7" new_sha="$8" new_backend="$9"
  arena_ingress_tool "$source_root" --activate \
    --expect-mode "$old_mode" --expect-slot "$old_slot" \
    --expect-sha "$old_sha" --expect-backend "$old_backend" \
    --mode "$new_mode" --slot "$new_slot" \
    --sha "$new_sha" --backend "$new_backend"
}

arena_slot_ports() {
  case "$1" in
    0) ARENA_API_PORT=8792; ARENA_EDGE_PORT=8890 ;;
    1) ARENA_API_PORT=8794; ARENA_EDGE_PORT=8891 ;;
    *) echo "ERROR: invalid Arena release slot" >&2; return 1 ;;
  esac
}

arena_runtime_tree() {
  local slot="$1" sha="$2"
  if ! [[ "$sha" =~ ^[0-9a-f]{40}$ ]]; then
    echo "ERROR: Arena release SHA is invalid" >&2
    return 1
  fi
  printf '%s/slot%s-%s\n' "$ARENA_RUNTIME_ROOT" "$slot" "$sha"
}

recover_arena_restart_owner() {
  local previous_tree helper owner state required_phase="${1:-}"
  previous_tree="$(arena_runtime_tree "$ARENA_PREVIOUS_SLOT" "$ARENA_PREVIOUS_SHA")" || return 1
  if [ -L "$previous_tree" ] || [ ! -d "$previous_tree" ]; then
    echo "ERROR: previous Arena release archive is unavailable" >&2
    return 1
  fi
  PYTHONPATH="$GATEWAY_PREFLIGHT_TREE" "$GATEWAY_PYTHON_BIN" -B - \
      "$LEADPOET_REPO_ROOT" "$previous_tree" "$ARENA_PREVIOUS_SHA" <<'PY' || return 1
from pathlib import Path
import sys
from scripts.gateway_git_deploy import verify_materialized_tree
verify_materialized_tree(
    repo_root=Path(sys.argv[1]), materialized_root=Path(sys.argv[2]),
    target_sha=sys.argv[3], strict_extras=True,
)
PY
  helper="$GATEWAY_PREFLIGHT_TREE/scripts/manage_owned_process_group.py"
  verify_controller_process_helper "$helper" || return 1
  arena_component_command edge "$ARENA_PREVIOUS_SLOT" || return 1
  owner="$("$GATEWAY_PYTHON_BIN" "$helper" verify-invocation \
    --state-file "$ARENA_COMPONENT_STATE_FILE" --cwd "$previous_tree" \
    --uid "$(id -u)" --release-sha "$ARENA_PREVIOUS_SHA" \
    -- "${ARENA_COMPONENT_COMMAND[@]}")" || return 1
  state="$(env -u LAB_ARENA_SUPABASE_URL -u LAB_ARENA_SERVICE_KEY \
    PYTHONPATH="$GATEWAY_PREFLIGHT_TREE" "$GATEWAY_PYTHON_BIN" \
    "$GATEWAY_PREFLIGHT_TREE/scripts/lab_arena_restart_claim_guard.py" owned \
    --candidate "$ARENA_PREVIOUS_SHA" --invocation "$owner" \
    --environment-file "$GATEWAY_ENV_FILE")" || return 1
  "$GATEWAY_PYTHON_BIN" - "$ARENA_PREVIOUS_SHA" "$GATEWAY_RESTART_GUARD_SCOPE" "$state" "$required_phase" <<'PY' || return 1
import json
import sys
state = json.loads(sys.argv[3])
if (state.get("schema_version") != "leadpoet.lab_arena.restart_guard_state.v1"
        or state.get("guard_present") is not True
        or state.get("candidate_commit") != sys.argv[1]
        or state.get("restart_scope") != sys.argv[2]
        or state.get("restart_phase") not in ("gateway_destructive", "gateway_ready")
        or (sys.argv[4] and state.get("restart_phase") != sys.argv[4])
        or type(state.get("guard_generation")) is not int
        or state["guard_generation"] <= 0):
    raise SystemExit("previous Arena guard is not an active exact-owner recovery state")
PY
  GATEWAY_ACTIVE_RELEASE_RESTART_INVOCATION_ID="$owner"
  ARENA_RECOVERING=1
  echo "Verified exact-owner Arena restart recovery from serving release"
}

restore_arena_recovery_guard_before_destructive() {
  local helper state
  helper="$GATEWAY_PREFLIGHT_TREE/scripts/lab_arena_restart_claim_guard.py"
  if [ "$ARENA_RECOVERING" != 1 ] || [ "$GATEWAY_DESTRUCTIVE_PHASE_STARTED" = 1 ]; then
    return 1
  fi
  env -u LAB_ARENA_SUPABASE_URL -u LAB_ARENA_SERVICE_KEY \
    PYTHONPATH="$GATEWAY_PREFLIGHT_TREE" "$GATEWAY_PYTHON_BIN" "$helper" drain \
    --candidate "$ARENA_PREVIOUS_SHA" \
    --invocation "$GATEWAY_ACTIVE_RELEASE_RESTART_INVOCATION_ID" \
    --scope "$GATEWAY_RESTART_GUARD_SCOPE" --timeout-seconds 30 \
    --environment-file "$GATEWAY_ENV_FILE" >/dev/null || return 1
  state="$(env -u LAB_ARENA_SUPABASE_URL -u LAB_ARENA_SERVICE_KEY \
    PYTHONPATH="$GATEWAY_PREFLIGHT_TREE" "$GATEWAY_PYTHON_BIN" "$helper" owned \
    --candidate "$ARENA_PREVIOUS_SHA" \
    --invocation "$GATEWAY_ACTIVE_RELEASE_RESTART_INVOCATION_ID" \
    --environment-file "$GATEWAY_ENV_FILE")" || return 1
  "$GATEWAY_PYTHON_BIN" - "$ARENA_PREVIOUS_SHA" "$state" <<'PY'
import json
import sys
state = json.loads(sys.argv[2])
if (state.get("schema_version") != "leadpoet.lab_arena.restart_guard_state.v1"
        or state.get("guard_present") is not True
        or state.get("guard_active") is not True
        or state.get("candidate_commit") != sys.argv[1]
        or state.get("restart_phase") not in ("gateway_destructive", "gateway_ready")):
    raise SystemExit("prior serving Arena guard was not restored")
PY
}


retire_arena_runtime_tree() {
  local slot="$1" sha="$2" tree="$3" expected helper report status
  expected="$(arena_runtime_tree "$slot" "$sha")" || return 1
  if [ "$tree" != "$expected" ] || [ -L "$tree" ] || [ ! -d "$tree" ]; then
    echo "ERROR: retired Arena runtime is not the exact owned archive" >&2
    return 1
  fi
  helper="$GATEWAY_PREFLIGHT_TREE/scripts/arena_runtime_cache_retirement.py"
  # The privileged operation removes only validated source-derived caches.
  # Source ownership and file modes stay unchanged; the caller removes source.
  report="$(sudo env -i PATH=/usr/bin:/bin PYTHONPATH="$GATEWAY_PREFLIGHT_TREE" \
    "$GATEWAY_PYTHON_BIN" -B "$helper" --repo-root "$LEADPOET_REPO_ROOT" \
    --slot "$slot" --sha "$sha" --owner-uid "$(id -u)" --defer-if-busy)" || return 1
  status="$("$GATEWAY_PYTHON_BIN" - "$tree" "$report" <<'PY'
import json
import sys
report = json.loads(sys.argv[2])
if (not isinstance(report, dict) or report.get("archive") != sys.argv[1]
        or type(report.get("cache_paths")) is not int or report["cache_paths"] < 0
        or report.get("status") not in ("cache_cleanup_complete", "deferred_in_use")
        or type(report.get("applied")) is not bool
        or report["applied"] != (report["status"] == "cache_cleanup_complete")):
    raise SystemExit("invalid Arena runtime retirement result")
print(report["status"])
PY
)" || return 1
  if [ "$status" = deferred_in_use ]; then
    echo "Arena runtime retirement deferred: validated inactive archive is still in use: $tree"
    return 0
  fi
  find "$tree" -type d -exec chmod u+w {} + || return 1
  rm -rf -- "$tree"
}

prepare_arena_runtime_tree() {
  local destination temporary source_root="${1:-$GATEWAY_PREFLIGHT_TREE}"
  destination="$(arena_runtime_tree "$ARENA_CANDIDATE_SLOT" "$PREPARED_GATEWAY_SHA")" || return 1
  mkdir -p "$ARENA_RUNTIME_ROOT"
  chmod 700 "$ARENA_RUNTIME_ROOT"
  if [ -e "$destination" ] || [ -L "$destination" ]; then
    echo "ERROR: Arena candidate runtime already exists" >&2
    return 1
  fi
  temporary="$(mktemp -d "$ARENA_RUNTIME_ROOT/.candidate.XXXXXX")" || return 1
  if ! cp -a "$source_root/." "$temporary/"; then
    rm -rf -- "$temporary"
    return 1
  fi
  if ! PYTHONPATH="$GATEWAY_PREFLIGHT_TREE" "$GATEWAY_PYTHON_BIN" -B - \
      "$LEADPOET_REPO_ROOT" "$temporary" "$PREPARED_GATEWAY_SHA" <<'PY'
from pathlib import Path
import sys
from scripts.gateway_git_deploy import verify_materialized_tree
verify_materialized_tree(
    repo_root=Path(sys.argv[1]), materialized_root=Path(sys.argv[2]),
    target_sha=sys.argv[3], strict_extras=True,
)
PY
  then
    rm -rf -- "$temporary"
    echo "ERROR: retained Arena runtime differs from exact candidate Git blobs" >&2
    return 1
  fi
  chmod -R a-w "$temporary"
  mv -T -- "$temporary" "$destination"
  ARENA_CANDIDATE_TREE="$destination"
}

arena_component_command() {
  local component="$1" slot="$2"
  arena_slot_ports "$slot" || return 1
  case "$component" in
    api)
      ARENA_COMPONENT_COMMAND=("$GATEWAY_PYTHON_BIN" -B -u scripts/run_lab_arena_service.py
        --environment-file "$GATEWAY_ENV_FILE" --host 127.0.0.1
        --port "$ARENA_API_PORT" --no-driver)
      ;;
    edge)
      ARENA_COMPONENT_COMMAND=("$GATEWAY_PYTHON_BIN" -B -u scripts/run_arena_proxy_edge.py
        --environment-file "$GATEWAY_ENV_FILE" --port "$ARENA_EDGE_PORT"
        --sidecar-port "$ARENA_API_PORT")
      ;;
    driver)
      ARENA_COMPONENT_COMMAND=("$GATEWAY_PYTHON_BIN" -B -u scripts/run_lab_arena_service.py
        --environment-file "$GATEWAY_ENV_FILE" --driver-only)
      ;;
    *) echo "ERROR: invalid Arena component" >&2; return 1 ;;
  esac
  ARENA_COMPONENT_STATE_FILE="$ARENA_RUNTIME_ROOT/$component-slot$slot-process.json"
  ARENA_COMPONENT_LOG_FILE="$GATEWAY_LOG_ROOT/arena-$component-slot$slot.log"
}

start_arena_component() {
  local component="$1" slot="$2" tree="$3" sha="$4" pid process_helper controller_cwd
  arena_component_command "$component" "$slot" || return 1
  process_helper="$tree/scripts/manage_owned_process_group.py"
  if ! verify_controller_process_helper "$process_helper"; then
    echo "ERROR: verified Arena process helper is unavailable" >&2
    return 1
  fi
  mkdir -p "$GATEWAY_LOG_ROOT"
  controller_cwd="$PWD"
  cd "$tree"
  if [ "$component" = edge ]; then
    env -i PATH="$PATH" HOME="$HOME" GITHUB_SHA="$sha" \
      LAB_ARENA_RELEASE_ROOT="$tree" LAB_ARENA_RELEASE_COMMIT="$sha" \
      GATEWAY_RESTART_INVOCATION_ID="$GATEWAY_RESTART_INVOCATION_ID" \
      GATEWAY_ACTIVE_RELEASE_RESTART_INVOCATION_ID="$GATEWAY_ACTIVE_RELEASE_RESTART_INVOCATION_ID" \
      LEADPOET_RESTART_INVOCATION_ID="$GATEWAY_RESTART_INVOCATION_ID" \
      PYTHONDONTWRITEBYTECODE=1 \
      setsid "${ARENA_COMPONENT_COMMAND[@]}" \
      >> "$ARENA_COMPONENT_LOG_FILE" 2>&1 < /dev/null \
      7>&- 8>&- 9>&- 190>&- 191>&- 192>&- 193>&- 194>&- 195>&- &
  else
    env -u LEADPOET_DOCKER_OPERATION_LOCK_HELD \
      -u LEADPOET_DOCKER_OPERATION_ADMISSION_LOCK_HELD -u LEADPOET_DOCKER_OPERATION_LOCK_OWNER_PID \
      -u GATEWAY_CONTROLLER_PROCESS_HELPER -u LAB_ARENA_PROCESS_HELPER \
      -u GATEWAY_RESTART_AUTHORITY_ROOT -u GATEWAY_RESTART_AUTHORITY_COMMIT \
      -u PREPARED_GATEWAY_SHA -u LAB_ARENA_RESTART_GUARD_GENERATION \
      GITHUB_SHA="$sha" \
      GATEWAY_RESTART_INVOCATION_ID="$GATEWAY_RESTART_INVOCATION_ID" \
      LEADPOET_RESTART_INVOCATION_ID="$GATEWAY_RESTART_INVOCATION_ID" \
      LAB_ARENA_RELEASE_ROOT="$tree" LAB_ARENA_RELEASE_COMMIT="$sha" \
      AWS_REGION=us-east-1 AWS_DEFAULT_REGION=us-east-1 \
      LEADPOET_AWS_INSTANCE_ROLE_ONLY=true PYTHONDONTWRITEBYTECODE=1 \
      setsid "${ARENA_COMPONENT_COMMAND[@]}" \
      >> "$ARENA_COMPONENT_LOG_FILE" 2>&1 < /dev/null \
      7>&- 8>&- 9>&- 190>&- 191>&- 192>&- 193>&- 194>&- 195>&- &
  fi
  pid="$!"
  cd "$controller_cwd"
  if ! "$GATEWAY_PYTHON_BIN" "$process_helper" record \
      --state-file "$ARENA_COMPONENT_STATE_FILE" --cwd "$tree" \
      --uid "$(id -u)" --launch-pgid "$pid" --discover-timeout-seconds 5 \
      -- "${ARENA_COMPONENT_COMMAND[@]}"; then
    kill -TERM "$pid" 2>/dev/null || true
    echo "ERROR: exact Arena component ownership was not recorded" >&2
    return 1
  fi
}

stop_arena_component() {
  local component="$1" slot="$2" tree="$3" process_helper proof state_hash
  local -a verified_stop_args=(--no-kill --signal-leader-only)
  arena_component_command "$component" "$slot" || return 1
  # Use the newly verified controller helper even when retiring a different
  # source SHA. The old runtime tree supplies only the exact process cwd.
  if [ -n "${4:-}" ]; then
    process_helper="$GATEWAY_PREFLIGHT_TREE/scripts/manage_owned_process_group.py"
  else
    process_helper="$ARENA_CANDIDATE_TREE/scripts/manage_owned_process_group.py"
  fi
  if ! verify_controller_process_helper "$process_helper"; then
    echo "ERROR: Arena release process helper is unavailable" >&2
    return 1
  fi
  if [ -n "${4:-}" ]; then
    # Recovery runs old source but stops through the exact new controller. Bind
    # its fresh private receipt to this owner after the latest idle proof.
    if [ -n "${6:-}" ]; then
      state_hash="$6"
    else
    proof="$("$GATEWAY_PYTHON_BIN" "$process_helper" verify \
      --state-file "$ARENA_COMPONENT_STATE_FILE" --cwd "$tree" --uid "$(id -u)" \
      -- "${ARENA_COMPONENT_COMMAND[@]}")" || return 1
    state_hash="$("$GATEWAY_PYTHON_BIN" -B - "$proof" <<'PY'
import json
import re
import sys
proof = json.loads(sys.argv[1])
value = proof.get("state_sha256")
if proof.get("ok") is not True or not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
    raise SystemExit("Arena recovery process receipt proof is invalid")
print(value)
PY
    )" || return 1
    fi
    verified_stop_args+=(--verified-state-sha256 "$state_hash" --verified-invocation "$4" --release-sha "$5")
  fi
  "$GATEWAY_PYTHON_BIN" "$process_helper" stop \
    --state-file "$ARENA_COMPONENT_STATE_FILE" --cwd "$tree" \
    --uid "$(id -u)" --term-seconds 6600 \
    "${verified_stop_args[@]}" -- "${ARENA_COMPONENT_COMMAND[@]}"
}

arena_component_inherited_docker_locks() {
  local component="$1" slot="$2" tree="$3" proof after status
  arena_component_command "$component" "$slot" || return 2
  proof="$("$GATEWAY_PYTHON_BIN" "$GATEWAY_PREFLIGHT_TREE/scripts/manage_owned_process_group.py" verify \
    --state-file "$ARENA_COMPONENT_STATE_FILE" --cwd "$tree" --uid "$(id -u)" \
    -- "${ARENA_COMPONENT_COMMAND[@]}")" || return 2
  if "$GATEWAY_PYTHON_BIN" - "$proof" "$LEADPOET_DOCKER_OPERATION_LOCK_FILE" \
      "${LEADPOET_DOCKER_OPERATION_ADMISSION_LOCK_FILE:-${LEADPOET_DOCKER_OPERATION_LOCK_FILE}.admission}" <<'PY'
import json, os, stat, sys
from pathlib import Path
try:
    proof = json.loads(sys.argv[1])
    pid = proof.get("pid")
    if proof.get("ok") is not True or type(pid) is not int or pid <= 1:
        raise SystemExit(2)
    held = []
    for fd, raw in zip((7, 8), sys.argv[2:]):
        path = os.path.realpath(raw)
        saved = os.stat(path, follow_symlinks=False)
        if not stat.S_ISREG(saved.st_mode) or saved.st_uid != os.getuid() or stat.S_IMODE(saved.st_mode) != 0o600:
            raise SystemExit(2)
        descriptor = Path(f"/proc/{pid}/fd/{fd}")
        try:
            target = os.readlink(descriptor)
        except FileNotFoundError:
            held.append(False)
            continue
        if target != path:
            held.append(False)
            continue
        opened = descriptor.stat()
        with Path(f"/proc/{pid}/fdinfo/{fd}").open() as stream:
            info = stream.read(4096)
        if (opened.st_dev, opened.st_ino) != (saved.st_dev, saved.st_ino):
            raise SystemExit(2)
        held.append(any(line.startswith("lock:") and "FLOCK" in line.split() and "WRITE" in line.split()
                        for line in info.splitlines()))
    raise SystemExit(0 if any(held) else 1)
except (OSError, ValueError, TypeError, AttributeError):
    raise SystemExit(2)
PY
  then status=0; else status="$?"; fi
  [ "$status" = 0 ] || [ "$status" = 1 ] || return 2
  after="$("$GATEWAY_PYTHON_BIN" "$GATEWAY_PREFLIGHT_TREE/scripts/manage_owned_process_group.py" verify \
    --state-file "$ARENA_COMPONENT_STATE_FILE" --cwd "$tree" --uid "$(id -u)" \
    -- "${ARENA_COMPONENT_COMMAND[@]}")" || return 2
  [ "$proof" = "$after" ] || return 2
  return "$status"
}

require_arena_lock_recovery_idle() {
  local state generation
  state="$(run_lab_arena_restart_guard "$GATEWAY_PREFLIGHT_TREE" owned)" || return 1
  generation="$("$GATEWAY_PYTHON_BIN" -B - "$PREPARED_GATEWAY_SHA" "$state" "$GATEWAY_RESTART_GUARD_SCOPE" "${1:-}" <<'PY'
import json
import sys
state = json.loads(sys.argv[2])
drain = state.get("drain", {})
if (state.get("schema_version") != "leadpoet.lab_arena.restart_guard_state.v1"
        or state.get("guard_present") is not True or state.get("guard_active") is not True
        or state.get("candidate_commit") != sys.argv[1]
        or state.get("restart_scope") != sys.argv[3]
        or state.get("restart_phase") != "gateway_destructive"
        or type(state.get("guard_generation")) is not int or state["guard_generation"] <= 0
        or (sys.argv[4] and str(state["guard_generation"]) != sys.argv[4])
        or drain.get("preserved") is not True
        or any(type(drain.get(key)) is not int or drain[key] != 0 for key in
               ("current_leased_count", "still_leased_count", "lost_or_mutated_count"))):
    raise SystemExit("Arena lock recovery requires an active owned idle gateway guard")
print(state["guard_generation"])
PY
  )" || return 1
  # The exact active guard blocks new API reviews. Still reject every live or
  # ambiguous paid-review claim/lease before recovering process locks.
  PYTHONPATH="$GATEWAY_PREFLIGHT_TREE" "$GATEWAY_PYTHON_BIN" -B \
    "$GATEWAY_PREFLIGHT_TREE/scripts/arena_first_cutover_idle.py" \
    --environment-file "$GATEWAY_ENV_FILE" \
    --candidate "$PREPARED_GATEWAY_SHA" \
    --invocation "$GATEWAY_ACTIVE_RELEASE_RESTART_INVOCATION_ID" \
    --scope "$GATEWAY_RESTART_GUARD_SCOPE" --generation "$generation" >/dev/null || return 1
  # The normal canonical RPC checks the same owner, generation, active lease and
  # current drain atomically. Repeating its current destructive phase is legal.
  run_lab_arena_restart_guard "$GATEWAY_PREFLIGHT_TREE" authorize \
    --generation "$generation" --phase gateway_destructive >/dev/null
}

retire_arena_retained_overlap() {
  (
    # Keep the serving archive/API untouched. Recover only the opposite slot
    # described by exact private receipts before ordinary candidate allocation.
    PREPARED_GATEWAY_SHA="$ARENA_PREVIOUS_SHA"
    helper="$GATEWAY_PREFLIGHT_TREE/scripts/manage_owned_process_group.py"
    verify_controller_process_helper "$helper" || exit 1
    for component in api edge; do
      arena_component_command "$component" "$ARENA_PREVIOUS_SLOT"
      [ "$("$GATEWAY_PYTHON_BIN" "$helper" verify-invocation \
        --state-file "$ARENA_COMPONENT_STATE_FILE" --cwd "$(arena_runtime_tree "$ARENA_PREVIOUS_SLOT" "$ARENA_PREVIOUS_SHA")" \
        --uid "$(id -u)" --release-sha "$ARENA_PREVIOUS_SHA" -- "${ARENA_COMPONENT_COMMAND[@]}")" = \
        "$GATEWAY_ACTIVE_RELEASE_RESTART_INVOCATION_ID" ] || exit 1
    done
    old_slot=$((1 - ARENA_PREVIOUS_SLOT))
    report="$("$GATEWAY_PYTHON_BIN" "$helper" inspect-retained-slot \
      --runtime-root "$ARENA_RUNTIME_ROOT" --slot "$old_slot" --uid "$(id -u)" \
      --python-bin "$GATEWAY_PYTHON_BIN" --environment-file "$GATEWAY_ENV_FILE" \
      --owner "$GATEWAY_ACTIVE_RELEASE_RESTART_INVOCATION_ID")" || exit 1
    read -r old_tree old_sha old_owner <<< "$("$GATEWAY_PYTHON_BIN" -B - "$report" <<'PY'
import json,sys
r=json.loads(sys.argv[1])
print(r['tree'] or '-',r['sha'] or '-',r['owner'])
PY
    )"
    [ "$old_tree" != - ] || exit 0
    [ "$old_tree" = "$(arena_runtime_tree "$old_slot" "$old_sha")" ] || exit 1
    PYTHONPATH="$GATEWAY_PREFLIGHT_TREE" "$GATEWAY_PYTHON_BIN" -B - \
        "$LEADPOET_REPO_ROOT" "$old_tree" "$old_sha" <<'PY' || exit 1
from pathlib import Path
import sys
from scripts.gateway_git_deploy import verify_materialized_tree
verify_materialized_tree(repo_root=Path(sys.argv[1]),materialized_root=Path(sys.argv[2]),
                         target_sha=sys.argv[3],strict_extras=True)
PY
    state="$(run_lab_arena_restart_guard "$GATEWAY_PREFLIGHT_TREE" owned)" || exit 1
    generation="$("$GATEWAY_PYTHON_BIN" -B - "$state" <<'PY'
import json,sys
print(json.loads(sys.argv[1])['guard_generation'])
PY
    )" || exit 1
    for component in driver edge api; do
      state_hash="$("$GATEWAY_PYTHON_BIN" -B - "$report" "$component" <<'PY'
import json,sys
print(json.loads(sys.argv[1])['components'].get(sys.argv[2],''))
PY
      )" || exit 1
      if [ -n "$state_hash" ]; then
        require_arena_lock_recovery_idle "$generation" || exit 1
        stop_arena_component "$component" "$old_slot" "$old_tree" "$old_owner" "$old_sha" "$state_hash" || exit 1
      fi
    done
    require_arena_lock_recovery_idle "$generation" || exit 1
    retire_arena_runtime_tree "$old_slot" "$old_sha" "$old_tree" || exit 1
  )
}

recover_arena_inherited_docker_locks() {
  (
    # Subshell variables stay available to EXIT cleanup even when errexit
    # unwinds the function; function-local variables would already be unset.
    ingress="" current_tree="" old_tree="" old_slot="" component="" status=0 leaking=0 crossed=0 ready=0
    ARENA_PREVIOUS_MODE="" ARENA_PREVIOUS_SLOT="" ARENA_PREVIOUS_SHA="" ARENA_PREVIOUS_BACKEND=""
    ARENA_CANDIDATE_TREE="" ARENA_CANDIDATE_SLOT="" ARENA_CONTINUITY_ENABLED=1 ARENA_REUSE_PREVIOUS=0
    # Recovery has no live gateway env. Load the same hydrated configuration
    # used by the normal candidate, without changing parent controller state.
    set -a
    . "$ENV_CLONE"
    set +a
    restore_gateway_restart_path_authority
    cd "$GATEWAY_PREFLIGHT_TREE"
    mode=""
    mode="$(env -u LAB_ARENA_MODE PYTHONPATH="$GATEWAY_PREFLIGHT_TREE" \
      "$GATEWAY_PYTHON_BIN" -B - "$GATEWAY_ENV_FILE" <<'PYMODE'
import os, sys
from pathlib import Path
from scripts.run_lab_arena_service import load_scoped_environment
load_scoped_environment(Path(sys.argv[1]))
print(os.environ.get("LAB_ARENA_MODE", "off").strip().lower())
PYMODE
    )" || exit 1
    case "$mode" in
      off) exit 0 ;;
      shadow|live) ;;
      *) echo "ERROR: Arena mode is invalid for lock recovery" >&2; exit 1 ;;
    esac
    ingress="$(arena_ingress_tool "$GATEWAY_PREFLIGHT_TREE" --inspect)" || exit 1
    read -r ARENA_PREVIOUS_MODE ARENA_PREVIOUS_SLOT ARENA_PREVIOUS_SHA ARENA_PREVIOUS_BACKEND <<< "$ingress"
    if [ "$ARENA_PREVIOUS_MODE:$ARENA_PREVIOUS_BACKEND" != dual:closed ]; then exit 0; fi
    current_tree="$(arena_runtime_tree "$ARENA_PREVIOUS_SLOT" "$ARENA_PREVIOUS_SHA")" || exit 1
    PREPARED_GATEWAY_SHA="$ARENA_PREVIOUS_SHA"
    old_slot=$((1 - ARENA_PREVIOUS_SLOT))
    old_tree="$(arena_runtime_tree "$old_slot" "$ARENA_PREVIOUS_SHA")" || exit 1
    trap 'status=$?; if [ "$status" -ne 0 ] && [ "$crossed" = 0 ]; then
      rollback_arena_candidate_before_destructive || true
    fi' EXIT
    for component in api edge; do
      if arena_component_inherited_docker_locks "$component" "$ARENA_PREVIOUS_SLOT" "$current_tree"; then
        leaking=1
      else status="$?"; [ "$status" = 1 ] || exit 1; fi
    done
    if [ "$leaking" = 0 ]; then
      # A prior handoff can have switched ingress before retirement failed.
      # Finish only the opposite, exact same-source owned slot that still leaks.
      for component in api edge; do
        arena_component_command "$component" "$old_slot"
        if [ -e "$ARENA_COMPONENT_STATE_FILE" ] || [ -L "$ARENA_COMPONENT_STATE_FILE" ]; then
          if arena_component_inherited_docker_locks "$component" "$old_slot" "$old_tree"; then
            leaking=1
          else status="$?"; [ "$status" = 1 ] || exit 1; fi
        fi
      done
      [ "$leaking" = 1 ] || exit 0
      crossed=1
    fi
    # This repair never moves a paid worker or removes a live driver's cwd.
    for slot in "$ARENA_PREVIOUS_SLOT" "$old_slot"; do
      arena_component_command driver "$slot"
      if [ -e "$ARENA_COMPONENT_STATE_FILE" ] || [ -L "$ARENA_COMPONENT_STATE_FILE" ]; then
        echo "ERROR: Arena lock recovery refuses a retained driver state" >&2; exit 1
      fi
      PYTHONPATH="$GATEWAY_PREFLIGHT_TREE" "$GATEWAY_PYTHON_BIN" -B - \
          "$(arena_runtime_tree "$slot" "$PREPARED_GATEWAY_SHA")" "${ARENA_COMPONENT_COMMAND[@]}" <<'PY'
import os, sys
from scripts.manage_owned_process_group import _matching_processes
if _matching_processes(argv=sys.argv[2:], cwd=sys.argv[1], uid=os.getuid()):
    raise SystemExit("Arena lock recovery refuses a running owned driver")
PY
    done
    [ "$GATEWAY_ACTIVE_RELEASE_COMPONENT" = gateway ] || exit 1
    recover_arena_restart_owner gateway_destructive || exit 1
    for component in api edge; do
      arena_component_command "$component" "$ARENA_PREVIOUS_SLOT"
      [ "$("$GATEWAY_PYTHON_BIN" "$GATEWAY_PREFLIGHT_TREE/scripts/manage_owned_process_group.py" verify-invocation \
        --state-file "$ARENA_COMPONENT_STATE_FILE" --cwd "$current_tree" --uid "$(id -u)" \
        --release-sha "$PREPARED_GATEWAY_SHA" -- "${ARENA_COMPONENT_COMMAND[@]}")" = \
        "$GATEWAY_ACTIVE_RELEASE_RESTART_INVOCATION_ID" ] || exit 1
    done
    require_arena_lock_recovery_idle || exit 1
    if [ "$crossed" = 0 ]; then
      ARENA_CANDIDATE_SLOT="$old_slot"
      prepare_arena_runtime_tree "$current_tree"
      arena_slot_ports "$ARENA_CANDIDATE_SLOT"
      "$GATEWAY_PYTHON_BIN" - "$ARENA_API_PORT" "$ARENA_EDGE_PORT" <<'PY'
import socket, sys
for value in sys.argv[1:]:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", int(value)))
PY
      start_arena_component api "$ARENA_CANDIDATE_SLOT" "$ARENA_CANDIDATE_TREE" "$PREPARED_GATEWAY_SHA"
      start_arena_component edge "$ARENA_CANDIDATE_SLOT" "$ARENA_CANDIDATE_TREE" "$PREPARED_GATEWAY_SHA"
      for component in api edge; do
        if arena_component_inherited_docker_locks "$component" "$ARENA_CANDIDATE_SLOT" "$ARENA_CANDIDATE_TREE"; then
          echo "ERROR: replacement Arena component inherited Docker locks" >&2; exit 1
        else status="$?"; [ "$status" = 1 ] || exit 1; fi
      done
      for attempt in $(seq 1 60); do
        if timeout 5 curl -fsS "http://127.0.0.1:$ARENA_API_PORT/arena/v1/current" >/dev/null 2>&1 \
            && timeout 5 curl -fsS "http://127.0.0.1:$ARENA_EDGE_PORT/arena/v1/current" >/dev/null 2>&1; then
          ready=1; break
        fi
        sleep 2
      done
      [ "$ready" = 1 ] || { echo "ERROR: Arena lock handoff candidate was not ready" >&2; exit 1; }
      require_arena_lock_recovery_idle || exit 1
      arena_ingress_transition "$GATEWAY_PREFLIGHT_TREE" dual "$ARENA_PREVIOUS_SLOT" "$PREPARED_GATEWAY_SHA" closed \
        dual "$ARENA_CANDIDATE_SLOT" "$PREPARED_GATEWAY_SHA" closed
      crossed=1
      old_slot="$ARENA_PREVIOUS_SLOT"
      old_tree="$current_tree"
    else
      ARENA_CANDIDATE_TREE="$current_tree"
      ARENA_CANDIDATE_SLOT="$ARENA_PREVIOUS_SLOT"
      # Verify the opposite archive before completing its partial retirement.
      [ ! -L "$old_tree" ] && [ -d "$old_tree" ] || exit 1
      PYTHONPATH="$GATEWAY_PREFLIGHT_TREE" "$GATEWAY_PYTHON_BIN" -B - \
          "$LEADPOET_REPO_ROOT" "$old_tree" "$PREPARED_GATEWAY_SHA" <<'PY'
from pathlib import Path
import sys
from scripts.gateway_git_deploy import verify_materialized_tree
verify_materialized_tree(repo_root=Path(sys.argv[1]), materialized_root=Path(sys.argv[2]),
                         target_sha=sys.argv[3], strict_extras=True)
PY
    fi
    arena_slot_ports "$ARENA_CANDIDATE_SLOT"
    timeout 5 curl -fsS "http://127.0.0.1:$ARENA_API_PORT/arena/v1/current" >/dev/null
    timeout 5 curl -fsS "http://127.0.0.1:$ARENA_EDGE_PORT/arena/v1/current" >/dev/null
    for component in edge api; do
      arena_component_command "$component" "$old_slot"
      if [ -e "$ARENA_COMPONENT_STATE_FILE" ] || [ -L "$ARENA_COMPONENT_STATE_FILE" ]; then
        [ "$("$GATEWAY_PYTHON_BIN" "$GATEWAY_PREFLIGHT_TREE/scripts/manage_owned_process_group.py" verify-invocation \
          --state-file "$ARENA_COMPONENT_STATE_FILE" --cwd "$old_tree" --uid "$(id -u)" \
          --release-sha "$PREPARED_GATEWAY_SHA" -- "${ARENA_COMPONENT_COMMAND[@]}")" = \
          "$GATEWAY_ACTIVE_RELEASE_RESTART_INVOCATION_ID" ] || exit 1
        require_arena_lock_recovery_idle || exit 1
        stop_arena_component "$component" "$old_slot" "$old_tree" \
          "$GATEWAY_ACTIVE_RELEASE_RESTART_INVOCATION_ID" "$PREPARED_GATEWAY_SHA"
      fi
    done
    [ "$(arena_ingress_tool "$GATEWAY_PREFLIGHT_TREE" --inspect)" = \
      "dual $ARENA_CANDIDATE_SLOT $PREPARED_GATEWAY_SHA closed" ] || exit 1
    retire_arena_runtime_tree "$old_slot" "$PREPARED_GATEWAY_SHA" "$old_tree" || exit 1
    echo "Retired exact owned Arena processes that inherited Docker locks; ingress remains closed"
  )
}

wait_for_arena_driver_startup() {
  local log_offset="$1" attempt process_helper
  arena_component_command driver "$ARENA_CANDIDATE_SLOT" || return 1
  process_helper="$ARENA_CANDIDATE_TREE/scripts/manage_owned_process_group.py"
  for attempt in $(seq 1 60); do
    if ! "$GATEWAY_PYTHON_BIN" "$process_helper" verify \
        --state-file "$ARENA_COMPONENT_STATE_FILE" --cwd "$ARENA_CANDIDATE_TREE" \
        --uid "$(id -u)" -- "${ARENA_COMPONENT_COMMAND[@]}" >/dev/null; then
      echo "ERROR: Arena driver exited before startup completed" >&2
      return 1
    fi
    if "$GATEWAY_PYTHON_BIN" - "$ARENA_COMPONENT_LOG_FILE" "$log_offset" \
        "$GATEWAY_DEPLOY_SHA" "$GATEWAY_RESTART_INVOCATION_ID" <<'PY'
import os
from pathlib import Path
import stat
import sys
path, offset, sha, invocation = sys.argv[1:]
try:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        metadata = os.fstat(fd)
        start = int(offset)
        if (not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid()
                or start < 0 or metadata.st_size < start
                or metadata.st_size - start > 1024 * 1024):
            raise SystemExit(1)
        os.lseek(fd, start, os.SEEK_SET)
        lines = os.read(fd, metadata.st_size - start).decode("utf-8", "replace").splitlines()
    finally:
        os.close(fd)
except (OSError, ValueError):
    raise SystemExit(1)
marker = (f"ARENA_SERVICE_LIFECYCLE event=startup outcome=ok source_sha={sha} "
          f"physical_role=arena-driver invocation_id={invocation}")
raise SystemExit(0 if marker in lines else 1)
PY
    then
      return 0
    fi
    sleep 2
  done
  echo "ERROR: exact Arena driver startup was not recorded" >&2
  return 1
}

wait_for_arena_current_readiness() {
  local url="$1" attempt
  # Current-round reads include durable state and can exceed five seconds.
  # Retry only this read, with a fixed bound; never release on a failed read.
  for attempt in 1 2 3; do
    if timeout 15 curl --connect-timeout 5 --max-time 15 -fsS "$url" >/dev/null; then
      return 0
    fi
    if [ "$attempt" -lt 3 ]; then sleep 2; fi
  done
  echo "ERROR: Arena current-round readiness failed after three bounded reads" >&2
  return 1
}

verify_controller_process_helper() {
  local process_helper="${1:-}"
  if [ "$process_helper" = "/proc/self/fd/195" ]; then
    "$GATEWAY_PYTHON_BIN" - "$process_helper" <<'PY'
import fcntl
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
  env -u LEADPOET_DOCKER_OPERATION_LOCK_HELD \
    -u LEADPOET_DOCKER_OPERATION_ADMISSION_LOCK_HELD -u LEADPOET_DOCKER_OPERATION_LOCK_OWNER_PID \
    -u GATEWAY_CONTROLLER_PROCESS_HELPER \
    -u LAB_ARENA_PROCESS_HELPER \
    -u GATEWAY_RESTART_AUTHORITY_ROOT \
    -u GATEWAY_RESTART_AUTHORITY_COMMIT \
    -u PREPARED_GATEWAY_SHA \
    -u LAB_ARENA_RESTART_GUARD_GENERATION \
    setsid "$GATEWAY_PYTHON_BIN" -u scripts/run_lab_arena_service.py \
      --environment-file "$GATEWAY_ENV_FILE" \
      --host 127.0.0.1 --port 8792 \
      >> "$LAB_ARENA_SERVICE_LOG_FILE" 2>&1 < /dev/null \
      7>&- 8>&- 9>&- 190>&- 191>&- 192>&- 193>&- 194>&- 195>&- &
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
  local exec_helper=0
  if [ "${1:-}" = "--exec-drain" ]; then
    exec_helper=1
    shift
  fi
  local source_root="$1"
  shift
  local guard_args=("$@" --environment-file "$GATEWAY_ENV_FILE")
  local -a guard_environment=(
    env
    -u LAB_ARENA_SUPABASE_URL
    -u LAB_ARENA_SERVICE_KEY
  )
  if [ ! -r "$source_root/scripts/lab_arena_restart_claim_guard.py" ]; then
    echo "ERROR: exact Lab Arena restart guard helper is unavailable" >&2
    return 1
  fi
  local guard_command=(
    "${guard_environment[@]}" PYTHONPATH="$source_root" "$GATEWAY_PYTHON_BIN"
    "$source_root/scripts/lab_arena_restart_claim_guard.py" "${guard_args[@]}"
    --candidate "$PREPARED_GATEWAY_SHA"
    --invocation "$GATEWAY_ACTIVE_RELEASE_RESTART_INVOCATION_ID"
  )
  if [ "$exec_helper" = "1" ]; then
    exec "${guard_command[@]}"
  fi
  "${guard_command[@]}"
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
  local source_root generation="$LAB_ARENA_RESTART_GUARD_GENERATION"
  if [ "$GATEWAY_DESTRUCTIVE_PHASE_STARTED" = "1" ]; then
    return 0
  fi
  if [ -z "$generation" ]; then
    if [ "${GATEWAY_DEPLOY_STAGE:-}" != "lab_arena_claim_drain" ] \
        || ! [[ "$PREPARED_GATEWAY_SHA" =~ ^[0-9a-f]{40}$ ]]; then
      return 0
    fi
    # The drain may have acquired its guard before it could return a generation.
    # Generation zero makes the helper read and verify exact current ownership.
    generation=0
  fi
  source_root="${GATEWAY_PREFLIGHT_TREE:-$LEADPOET_REPO_ROOT}"
  run_lab_arena_restart_guard "$source_root" abort \
    --generation "$generation" >/dev/null 2>&1 || true
  LAB_ARENA_RESTART_GUARD_GENERATION=""
}

drain_lab_arena_for_restart() {
  local source_root="$1" report
  # Provider activity can renew a lease near the end of its runtime. Cover
  # the longest supported runtime (90 minutes), its full renewable lease
  # (105 minutes), and five minutes for natural expiry and completion polling.
  LAB_ARENA_RESTART_DRAIN_PRIOR_JOBS=" $(jobs -pr | tr '\n' ' ')"
  LAB_ARENA_RESTART_DRAIN_REPORT="$(mktemp "${GATEWAY_RESTART_TIMING_FILE%.jsonl}.drain.XXXXXX")" || return 1
  (
    run_lab_arena_restart_guard --exec-drain "$source_root" drain \
      --scope "$GATEWAY_RESTART_GUARD_SCOPE" \
      --timeout-seconds 12000
  ) >"$LAB_ARENA_RESTART_DRAIN_REPORT" &
  LAB_ARENA_RESTART_DRAIN_PID=$!
  if ! wait "$LAB_ARENA_RESTART_DRAIN_PID"; then
    LAB_ARENA_RESTART_DRAIN_PID=""
    rm -f -- "$LAB_ARENA_RESTART_DRAIN_REPORT"
    LAB_ARENA_RESTART_DRAIN_REPORT=""
    return 1
  fi
  LAB_ARENA_RESTART_DRAIN_PID=""
  report="$(cat "$LAB_ARENA_RESTART_DRAIN_REPORT")" || return 1
  rm -f -- "$LAB_ARENA_RESTART_DRAIN_REPORT"
  LAB_ARENA_RESTART_DRAIN_REPORT=""
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
GATEWAY_V2_CONFIG_DIR="${GATEWAY_V2_CONFIG_DIR:-/home/ec2-user/.config/leadpoet/v2}"
GATEWAY_V2_RELEASE_MANIFEST="${GATEWAY_V2_RELEASE_MANIFEST:-$GATEWAY_TEE_EIF_ROOT/gateway-v2-release-manifest.json}"
# Release acquisition happens while the existing gateway is still serving.
# Keep candidate evidence restart-scoped so its fail-closed verifier remains
# bound to the release that actually booted it until destructive cutover.
GATEWAY_PREPARED_V2_RELEASE_MANIFEST="${GATEWAY_PREPARED_V2_RELEASE_MANIFEST:-${GATEWAY_RESTART_TIMING_FILE%.jsonl}.candidate-release.json}"
GATEWAY_V2_RELEASE_BUCKET="${GATEWAY_V2_RELEASE_BUCKET:-leadpoet-attested-v2-artifacts-493765492819}"
GATEWAY_V2_RELEASE_PREFIX="${GATEWAY_V2_RELEASE_PREFIX:-attested-v2/releases}"
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
  GATEWAY_RESTART_GIT_SSH_COMMAND
  LEADPOET_GATEWAY_ENV_SECRET_ID
  GATEWAY_RESTART_CONTROLLER_ROOT
  GATEWAY_RESTART_AUTHORITY_ROOT
  GATEWAY_RESTART_AUTHORITY_COMMIT
  GATEWAY_CONTROLLER_PROCESS_HELPER
  GATEWAY_CONTROLLER_PROCESS_STATE_FILE
  LAB_ARENA_PROCESS_HELPER
  LAB_ARENA_SERVICE_STATE_FILE
  GATEWAY_PAIRED_RESTART
  GATEWAY_RESTART_GUARD_SCOPE
  GATEWAY_ACTIVE_RELEASE_RESTART_INVOCATION_ID
  GATEWAY_RESTART_RECOVERY_LOCK_FILE
  LEADPOET_DOCKER_OPERATION_LOCK_FILE
  GATEWAY_V2_CONFIG_DIR
  GATEWAY_V2_RELEASE_MANIFEST
  GATEWAY_PREPARED_V2_RELEASE_MANIFEST
  GATEWAY_V2_OFFLINE_ARTIFACT_ROOT
  GATEWAY_V2_RELEASE_PREFIX
  GATEWAY_V2_RELEASE_BUCKET
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
if ! [[ "$GATEWAY_RELEASE_SUPERSESSION_COUNT" =~ ^[0-9]+$ ]] \
    || ! [[ "$GATEWAY_RELEASE_SUPERSESSION_MAX" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: gateway release supersession counters are invalid" >&2
  exit 2
fi
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
    f"stage={stage} status={status} elapsed_seconds={record['elapsed_seconds']} "
    f"commit_sha={commit or 'unknown'}"
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

report_gateway_recovery_target() {
  # A failed candidate cannot be restored by bypassing the canonical controller.
  # Report only a prior SHA that the existing last-good receipt proves valid.
  [ "${GATEWAY_DESTRUCTIVE_PHASE_STARTED:-0}" = "1" ] || return 0
  "$GATEWAY_PYTHON_BIN" - "$GATEWAY_DEPLOY_PLAN_FILE" "$GATEWAY_LAST_GOOD_MANIFEST" <<'PY' || true
import json
import re
import sys

try:
    with open(sys.argv[1], encoding="utf-8") as handle:
        plan = json.load(handle)
    with open(sys.argv[2], encoding="utf-8") as handle:
        last_good = json.load(handle)
    sha = plan.get("previous_sha")
    if (
        isinstance(sha, str)
        and re.fullmatch(r"[0-9a-f]{40}", sha)
        and last_good.get("status") == "succeeded"
        and last_good.get("target_sha") == sha
        and last_good.get("repo_root") == plan.get("repo_root")
    ):
        print(f"GATEWAY_RESTART_RECOVERY canonical_last_good_sha={sha}")
    else:
        print("GATEWAY_RESTART_RECOVERY canonical_last_good_sha=unverified")
except (OSError, ValueError, TypeError, AttributeError):
    print("GATEWAY_RESTART_RECOVERY canonical_last_good_sha=unavailable")
PY
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

start_gateway_offline_artifact_prepare() {
  local prepare_script process_group_marker
  local -a prepare_command
  prepare_script="$GATEWAY_PREFLIGHT_TREE/gateway/tee/prepare_offline_enclave_dependencies.sh"
  if [ -n "$GATEWAY_OFFLINE_ARTIFACT_PREPARE_PID" ]; then
    echo "ERROR: identity-enclave dependency preparation is already running" >&2
    return 1
  fi
  if [ ! -r "$prepare_script" ]; then
    echo "ERROR: prepared identity-enclave dependency helper is unavailable" >&2
    return 1
  fi
  if ! mkdir -p "$(dirname "$GATEWAY_OFFLINE_ARTIFACT_PREPARE_LOG")" \
      || ! : > "$GATEWAY_OFFLINE_ARTIFACT_PREPARE_LOG" \
      || ! chmod 600 "$GATEWAY_OFFLINE_ARTIFACT_PREPARE_LOG"; then
    echo "ERROR: identity-enclave dependency preparation log is unavailable" >&2
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
  env -u GATEWAY_CONTROLLER_PROCESS_HELPER \
    -u LAB_ARENA_PROCESS_HELPER \
    -u GATEWAY_GIT_HELPER \
    -u GATEWAY_HOST_MEMORY_GUARD_PATH \
    -u GATEWAY_RESTART_AUTHORITY_ROOT \
    -u GATEWAY_RESTART_AUTHORITY_COMMIT \
    -u GATEWAY_ACTIVE_RELEASE_RESTART_INVOCATION_ID \
    -u GATEWAY_ACTIVE_RELEASE_COMPONENT \
    -u GATEWAY_PAIRED_RESTART \
    -u GATEWAY_RESTART_GUARD_SCOPE \
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
      "identity-enclave dependency preparation"; then
    GATEWAY_OFFLINE_ARTIFACT_PREPARE_PID=""
    return 1
  fi
  echo "Started candidate identity-enclave dependency preparation as PID $GATEWAY_OFFLINE_ARTIFACT_PREPARE_PID"
  echo "Offline enclave dependency preparation log: $GATEWAY_OFFLINE_ARTIFACT_PREPARE_LOG"
}

wait_for_gateway_offline_artifact_prepare() {
  local prepare_pid status=0
  prepare_pid="$GATEWAY_OFFLINE_ARTIFACT_PREPARE_PID"
  if [ -z "$prepare_pid" ]; then
    echo "ERROR: identity-enclave dependency preparation was not started" >&2
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
    echo "ERROR: identity-enclave dependency preparation failed before shutdown" >&2
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
    GATEWAY_PAIRED_RESTART="$GATEWAY_PAIRED_RESTART" \
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
    GATEWAY_PREPARED_V2_RELEASE_MANIFEST="$GATEWAY_PREPARED_V2_RELEASE_MANIFEST" \
    GATEWAY_V2_RELEASE_BUCKET="$GATEWAY_V2_RELEASE_BUCKET" \
    GATEWAY_V2_RELEASE_PREFIX="$GATEWAY_V2_RELEASE_PREFIX" \
    GATEWAY_V2_OFFLINE_ARTIFACT_ROOT="$GATEWAY_V2_OFFLINE_ARTIFACT_ROOT" \
    GATEWAY_DEPLOY_STAGE=bootstrap \
    exec bash "$superseding_tree/gw_restart.sh"
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
else:
    raise SystemExit("running gateway command differs")
if first["cwd"] not in allowed_cwds:
    raise SystemExit("running gateway working directory differs")

reclaimable_gateway_mib = first["rss_kib"] // 1024
reclaimable_mib = reclaimable_gateway_mib
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

rollback_arena_candidate_before_destructive() {
  local live_ingress prior_ingress candidate_ingress
  if [ "$ARENA_CONTINUITY_ENABLED" != 1 ] \
      || [ "$GATEWAY_DESTRUCTIVE_PHASE_STARTED" = 1 ] \
      || [ -z "$ARENA_CANDIDATE_TREE" ]; then
    return 0
  fi
  if [ "$ARENA_CANDIDATE_TREE" != "$(arena_runtime_tree "$ARENA_CANDIDATE_SLOT" "$PREPARED_GATEWAY_SHA")" ]; then
    echo "ERROR: refusing cleanup of an unexpected Arena runtime path" >&2
    return 1
  fi
  if [ "$ARENA_REUSE_PREVIOUS" = 1 ]; then
    # The verified prior archive and edge still serve the domain. There is
    # no new process or archive to clean on an interrupted same-SHA retry.
    return 0
  fi
  live_ingress="$(arena_ingress_tool "$GATEWAY_PREFLIGHT_TREE" --inspect)" || {
    echo "ERROR: Arena ingress state is unverifiable; retaining serving candidate" >&2
    return 1
  }
  if [ "$ARENA_PREVIOUS_MODE" = legacy ]; then
    prior_ingress="legacy 0 - legacy"
    candidate_ingress="domain $ARENA_CANDIDATE_SLOT $PREPARED_GATEWAY_SHA legacy"
  elif [ "$ARENA_PREVIOUS_MODE" = domain ]; then
    prior_ingress="domain $ARENA_PREVIOUS_SLOT $ARENA_PREVIOUS_SHA legacy"
    candidate_ingress="domain $ARENA_CANDIDATE_SLOT $PREPARED_GATEWAY_SHA legacy"
  elif [ "$ARENA_PREVIOUS_MODE" = dual ]; then
    prior_ingress="dual $ARENA_PREVIOUS_SLOT $ARENA_PREVIOUS_SHA $ARENA_PREVIOUS_BACKEND"
    candidate_ingress="dual $ARENA_CANDIDATE_SLOT $PREPARED_GATEWAY_SHA $ARENA_PREVIOUS_BACKEND"
  else
    echo "ERROR: prior Arena ingress state is invalid; retaining candidate" >&2
    return 1
  fi
  if [ "$live_ingress" = "$candidate_ingress" ]; then
    if [ "$ARENA_PREVIOUS_MODE" = legacy ]; then
      if ! arena_ingress_transition "$GATEWAY_PREFLIGHT_TREE" \
          domain "$ARENA_CANDIDATE_SLOT" "$PREPARED_GATEWAY_SHA" legacy \
          legacy 0 - legacy; then
        echo "ERROR: Arena ingress rollback failed; retaining serving candidate" >&2
        return 1
      fi
    elif [ "$ARENA_PREVIOUS_MODE" = domain ]; then
      if ! arena_ingress_transition "$GATEWAY_PREFLIGHT_TREE" \
          domain "$ARENA_CANDIDATE_SLOT" "$PREPARED_GATEWAY_SHA" legacy \
          domain "$ARENA_PREVIOUS_SLOT" "$ARENA_PREVIOUS_SHA" legacy; then
        echo "ERROR: Arena ingress rollback failed; retaining serving candidate" >&2
        return 1
      fi
    elif [ "$ARENA_PREVIOUS_MODE" = dual ]; then
      if ! arena_ingress_transition "$GATEWAY_PREFLIGHT_TREE" \
          dual "$ARENA_CANDIDATE_SLOT" "$PREPARED_GATEWAY_SHA" "$ARENA_PREVIOUS_BACKEND" \
          dual "$ARENA_PREVIOUS_SLOT" "$ARENA_PREVIOUS_SHA" "$ARENA_PREVIOUS_BACKEND"; then
        echo "ERROR: Arena ingress rollback failed; retaining serving candidate" >&2
        return 1
      fi
    fi
  elif [ "$live_ingress" != "$prior_ingress" ]; then
    echo "ERROR: Arena ingress changed during rollback; retaining serving candidate" >&2
    return 1
  fi
  if ! stop_arena_component edge "$ARENA_CANDIDATE_SLOT" "$ARENA_CANDIDATE_TREE" \
      || ! stop_arena_component api "$ARENA_CANDIDATE_SLOT" "$ARENA_CANDIDATE_TREE"; then
    echo "ERROR: candidate Arena process did not drain; retaining runtime" >&2
    return 1
  fi
  if [ -L "$ARENA_CANDIDATE_TREE" ] || [ ! -d "$ARENA_CANDIDATE_TREE" ]; then
    echo "ERROR: candidate Arena runtime is not an owned directory" >&2
    return 1
  fi
  retire_arena_runtime_tree "$ARENA_CANDIDATE_SLOT" "$PREPARED_GATEWAY_SHA" "$ARENA_CANDIDATE_TREE"
}

on_gateway_restart_exit() {
  local status="$?"
  if [ "$status" -ne 0 ]; then
    if [ "$ARENA_RECOVERING" = 1 ] \
        && [ "$GATEWAY_DESTRUCTIVE_PHASE_STARTED" != 1 ]; then
      if rollback_arena_candidate_before_destructive; then
        restore_arena_recovery_guard_before_destructive || \
          echo "ERROR: prior Arena guard could not be restored" >&2
      else
        echo "ERROR: Arena ingress could not be restored; retaining restart guard" >&2
      fi
    else
      if rollback_arena_candidate_before_destructive; then
        abort_lab_arena_restart_guard_before_destructive
      else
        echo "ERROR: Arena ingress could not be restored; retaining restart guard" >&2
      fi
    fi
    record_gateway_restart_timing "${GATEWAY_DEPLOY_STAGE:-unknown}" "failed" \
      >/dev/null 2>&1 || true
    report_gateway_recovery_target
  fi
  emit_gateway_restart_sentry_summary "$status"
  cancel_gateway_offline_artifact_prepare
  if [ -n "$LAB_ARENA_RESTART_DRAIN_REPORT" ]; then
    rm -f -- "$LAB_ARENA_RESTART_DRAIN_REPORT"
  fi
  rm -f -- "$GATEWAY_PREPARED_V2_RELEASE_MANIFEST" 2>/dev/null || true
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
trap on_gateway_restart_exit EXIT

on_gateway_restart_signal() {
  local status="$1" running_pid drain_pid="$LAB_ARENA_RESTART_DRAIN_PID"
  trap '' HUP INT TERM
  if [ -z "$drain_pid" ] && [ -n "$LAB_ARENA_RESTART_DRAIN_REPORT" ]; then
    # A signal can arrive after launch but before the parent records $!.
    while read -r running_pid; do
      case "$LAB_ARENA_RESTART_DRAIN_PRIOR_JOBS" in
        *" $running_pid "*) ;;
        *) drain_pid="$running_pid"; break ;;
      esac
    done < <(jobs -pr)
  fi
  if [ -n "$drain_pid" ]; then
    # Background commands can inherit ignored SIGINT; TERM reaches Python.
    while read -r running_pid; do
      if [ "$running_pid" = "$drain_pid" ]; then
        kill -s TERM "$drain_pid" 2>/dev/null || true
        break
      fi
    done < <(jobs -pr)
    wait "$drain_pid" 2>/dev/null || true
    LAB_ARENA_RESTART_DRAIN_PID=""
  fi
  exit "$status"
}
trap 'on_gateway_restart_signal 129' HUP
trap 'on_gateway_restart_signal 130' INT
trap 'on_gateway_restart_signal 143' TERM

bootstrap_gateway_pip() {
  # pip vendors its own CA bundle. Upgrade it before any dependency download,
  # including when the ordinary dependency fingerprint is already satisfied.
  "$GATEWAY_PYTHON_BIN" - "$@" <<'PY'
import hashlib
import os
from pathlib import Path
import re
import ssl
import subprocess
import sys
import tempfile
import urllib.request

def patched_pip_is_installed():
    try:
        import pip
        from pip._vendor import certifi

        def stable_version(value):
            if not re.fullmatch(r"[0-9]+\.[0-9]+(?:\.[0-9]+)?", value):
                return (0, 0, 0)
            return tuple(int(part) for part in value.split("."))

        return (
            stable_version(pip.__version__) >= (26, 2, 2)
            and stable_version(certifi.__version__) >= (2026, 10, 19)
            and Path(certifi.where()).resolve().is_relative_to(
                Path(pip.__file__).resolve().parent
            )
        )
    except Exception:
        return False

if patched_pip_is_installed():
    raise SystemExit(0)

# PSF emSign incident guidance, 2026-10-20. The hash also protects this first
# download while the host CA store still contains the compromised root.
url = "https://files.pythonhosted.org/packages/3f/06/3e4420302363fbb71be3c7f35ce042f54bfca330e3e66b3a2e3c4cca579f/pip-26.2.2-py3-none-any.whl"
sha256 = "fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70"
custom_ca = next(
    (os.environ[name] for name in
     ("REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "PIP_CERT") if os.environ.get(name)),
    None,
)
ca_options = (
    {"capath" if Path(custom_ca).is_dir() else "cafile": custom_ca}
    if custom_ca else {}
)
context = ssl.create_default_context(**ca_options)
with tempfile.TemporaryDirectory(prefix="gateway-pip-bootstrap-") as directory:
    wheel = Path(directory) / "pip-26.2.2-py3-none-any.whl"
    with urllib.request.urlopen(url, context=context, timeout=60) as response:
        payload = response.read()
    if hashlib.sha256(payload).hexdigest() != sha256:
        raise SystemExit("ERROR: gateway pip bootstrap SHA256 mismatch")
    wheel.write_bytes(payload)
    # Execute only the verified wheel. This works even when pip is absent;
    # no network access, get-pip.py, ensurepip, or old pip code is needed.
    subprocess.run(
        [sys.executable, "-c",
         'import runpy, sys; sys.path.insert(0, sys.argv.pop(1)); '
         'runpy.run_module("pip", run_name="__main__", alter_sys=True)',
         str(wheel), "install", "--no-index", "--no-deps",
         "--disable-pip-version-check", "--force-reinstall", *sys.argv[1:],
         str(wheel)],
        check=True,
    )
PY
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
  if ! "$GATEWAY_PYTHON_BIN" -c \
      'import sys; raise SystemExit(0 if sys.prefix != sys.base_prefix else 1)'; then
    pip_scope=(--user)
  fi
  # Explicit return is required: callers use this function in an `if`, which
  # disables bash errexit throughout the function body.
  bootstrap_gateway_pip ${pip_scope[@]+"${pip_scope[@]}"} || return 1
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
  )" || return 1
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
      --use-deprecated=legacy-certs \
      ${pip_scope[@]+"${pip_scope[@]}"} \
      --requirement "$requirements_file" \
      "${GATEWAY_HOST_EXTRA_PYTHON_PACKAGES[@]}" || return 1
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

enforce_deployment_environment() {
  unset BUILD_ID BUILD_TIME_UTC BUILD_TIMESTAMP GITHUB_TAG GIT_TAG
  export LEADPOET_REPO_ROOT GATEWAY_ROOT GATEWAY_LOG_ROOT GATEWAY_LOG_FILE
  export GATEWAY_TEE_EIF_ROOT GATEWAY_V2_RELEASE_ARCHIVE_ROOT
  export GATEWAY_RESTART_TEMP_CLEANUP_MIN_AGE_SECONDS
  export GATEWAY_RESTART_EMERGENCY_BACKUP_MIN_AGE_SECONDS
  export GATEWAY_RESTART_CLEANUP_MAX_CANDIDATES
  export GATEWAY_V2_CONFIG_DIR GATEWAY_V2_RELEASE_MANIFEST
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
  export GATEWAY_RUNTIME_SOURCE_ROOT="$LEADPOET_REPO_ROOT"
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
    if [ ! -f "$release_dir/scripts/manage_owned_process_group.py" ] \
        || [ -L "$release_dir/scripts/manage_owned_process_group.py" ] \
        || [ "$(stat -c '%u:%g:%a' "$release_dir/scripts/manage_owned_process_group.py")" != "$(id -u):$(id -g):600" ] \
        || ! cmp -s "$temporary_dir/scripts/manage_owned_process_group.py" \
            "$release_dir/scripts/manage_owned_process_group.py"; then
      rm -rf -- "$temporary_dir"
      echo "ERROR: installed gateway restart controller process helper differs from the exact candidate" >&2
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

reset_orphaned_docker_storage_if_needed() {
  local required_kb cleanup_source_root
  cleanup_source_root="${GATEWAY_RESTART_AUTHORITY_ROOT:-$LEADPOET_REPO_ROOT}"
  required_kb="$(docker_build_required_kb)" || return 1
  # Export leaks can exist with ample free space. Reuse the preserving path
  # at this existing maintenance boundary instead of waiting for capacity loss.
  VALIDATOR_DOCKER_MIN_FREE_BYTES=$((required_kb * 1024)) \
    VALIDATOR_DOCKER_LIVE_RUNTIME_MIN_FREE_BYTES="${GATEWAY_DOCKER_LIVE_RUNTIME_MIN_FREE_BYTES:-18000000000}" \
    VALIDATOR_DOCKER_ALLOW_LIVE_HOST_GATEWAY_PRUNE=1 \
    bash "$cleanup_source_root/validator_tee/scripts/reclaim_docker_storage_v2.sh"
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
    is_stale_build = "pip install --no-cache-dir -r requirements.txt" in argv
    if is_stale_build and (cwd == repo_root or repo_root in cwd.parents):
        try:
            os.kill(int(entry.name), requested_signal)
        except (ProcessLookupError, PermissionError):
            pass
PY
}

run_bounded_restart_artifact_cleanup() {
  local gateway_build_root validator_build_root cleanup_source_root
  local -a build_authority=()
  # Before checkout, use the already-verified candidate archive. An absolute
  # entrypoint also prevents the old working directory from shadowing -m.
  cleanup_source_root="${GATEWAY_PREFLIGHT_TREE:-$LEADPOET_REPO_ROOT}"
  if [ -n "${PREPARED_GATEWAY_SHA:-}" ]; then
    build_authority=(--gateway-build-commit "$PREPARED_GATEWAY_SHA")
  elif [ -n "${GATEWAY_DEPLOY_SHA:-}" ]; then
    build_authority=(--gateway-build-commit "$GATEWAY_DEPLOY_SHA")
  fi
  gateway_build_root="${GATEWAY_V2_BUILD_WORK_ROOT:-$HOME/.cache/leadpoet/gateway-release-build-v2}"
  validator_build_root="${VALIDATOR_V2_BUILD_WORK_ROOT:-$HOME/.cache/leadpoet/validator-pcr0-normalizer-v2}"
  if [ ! -r "$cleanup_source_root/validator_tee/host/restart_artifact_cleanup_v2.py" ]; then
    echo "WARNING: bounded restart artifact cleanup helper is unavailable" >&2
    return 0
  fi
  if ! sudo env PYTHONPATH="$cleanup_source_root" "$GATEWAY_PYTHON_BIN" -B \
      "$cleanup_source_root/validator_tee/host/restart_artifact_cleanup_v2.py" \
      --apply "${build_authority[@]}" \
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

preflight_gateway_restart_disk() {
  local required_kb="$1" free_kb
  echo "Checking disk capacity before gateway shutdown"
  # The bounded helper excludes open paths, mounts, and retained release
  # artifacts. Docker daemon and storage cleanup remain after shutdown.
  run_bounded_restart_artifact_cleanup
  free_kb="$(root_free_kb)" || return 1
  [[ "$free_kb" =~ ^[0-9]+$ ]] || return 1
  if [ "${free_kb:-0}" -lt "$required_kb" ]; then
    echo "ERROR: insufficient free disk after cleanup: $(df -h / | tail -1)" >&2
    echo "Need at least ${required_kb} KiB free before gateway Arena Docker workloads." >&2
    return 1
  fi
}

pending_docker_reconcile() {
  local pending
  if ! pending="$(sudo python3 -I -c '
from pathlib import Path
try:
    Path("/var/lib/docker/.leadpoet-pending-overlay-reconcile").lstat()
except FileNotFoundError:
    print("0")
else:
    print("1")
')" || { [ "$pending" != "0" ] && [ "$pending" != "1" ]; }; then
    echo "ERROR: pending Docker reconciliation state is unreadable" >&2
    exit 1
  fi
  [ "$pending" = "1" ]
}

docker_build_required_kb() {
  python3 - "${GATEWAY_DOCKER_LIVE_RUNTIME_MIN_FREE_BYTES:-18000000000}" <<'PY'
import re
import sys

value = sys.argv[1]
if not re.fullmatch(r"[1-9][0-9]*", value) or int(value) > 9223372036854775807:
    raise SystemExit("invalid Docker build free-space requirement")
print((int(value) + 1023) // 1024)
PY
}

emergency_disk_preflight() {
  local free_kb emergency_lock_helper required_kb cleanup_source_root
  cleanup_source_root="${GATEWAY_RESTART_AUTHORITY_ROOT:-$LEADPOET_REPO_ROOT}"
  required_kb="$(docker_build_required_kb)" || return 1
  free_kb="$(root_free_kb)"
  if ! [[ "$free_kb" =~ ^[0-9]+$ ]]; then
    echo "ERROR: gateway disk capacity is unreadable" >&2
    return 1
  fi
  if [ "$free_kb" -ge "$required_kb" ] && ! pending_docker_reconcile; then
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
  emergency_lock_helper="$cleanup_source_root/validator_tee/scripts/docker_operation_lock_v2.sh"
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
      VALIDATOR_DOCKER_ALLOW_LIVE_HOST_GATEWAY_PRUNE=1 \
      VALIDATOR_DOCKER_ALLOW_DATA_ROOT_RESET=1 \
      bash "$cleanup_source_root/validator_tee/scripts/reclaim_docker_storage_v2.sh"; then
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
    # Only the active controller may set its shutdown state and owned paths.
    "GATEWAY_DESTRUCTIVE_PHASE_STARTED",
    "GATEWAY_CONTROLLER_PROCESS_STATE_FILE",
    "GATEWAY_PREFLIGHT_TREE",
    "GATEWAY_DEPLOY_COMMIT",
    "GATEWAY_ENV_FILE",
    "GATEWAY_RESTART_GIT_SSH_COMMAND",
    "GATEWAY_PYTHON_BIN",
    "GATEWAY_RESTART_CONTROLLER_ROOT",
    "GATEWAY_RESTART_AUTHORITY_ROOT",
    "GATEWAY_RESTART_AUTHORITY_COMMIT",
    "GATEWAY_ACTIVE_RELEASE_RESTART_INVOCATION_ID",
    "GATEWAY_ACTIVE_RELEASE_COMPONENT",
    "GATEWAY_PAIRED_RESTART",
    "GATEWAY_RESTART_GUARD_SCOPE",
    "PREPARED_GATEWAY_SHA",
    "LAB_ARENA_RESTART_GUARD_GENERATION",
    "GATEWAY_RESTART_RECOVERY_LOCK_FILE",
    "GATEWAY_RESTART_INVOCATION_ID",
    "GATEWAY_CONTROLLER_PROCESS_HELPER",
    "LAB_ARENA_PROCESS_HELPER",
    "GATEWAY_GIT_HELPER",
    "GATEWAY_HOST_MEMORY_GUARD_PATH",
    "GATEWAY_V2_CONFIG_DIR",
    "GATEWAY_V2_DEFER_WORKER_FLEETS",
    "GATEWAY_V2_OFFLINE_ARTIFACT_ROOT",
    "GATEWAY_V2_RELEASE_BUCKET",
    "GATEWAY_V2_RELEASE_ARCHIVE_ROOT",
    "GATEWAY_V2_RELEASE_MANIFEST",
    "GATEWAY_PREPARED_V2_RELEASE_MANIFEST",
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
    # Parent-shell hydration must preserve the active controller's state.
    "GATEWAY_DESTRUCTIVE_PHASE_STARTED",
    "GATEWAY_CONTROLLER_PROCESS_STATE_FILE",
    "GATEWAY_PREFLIGHT_TREE",
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
    "GATEWAY_RESTART_GIT_SSH_COMMAND",
    "GATEWAY_PYTHON_BIN",
    "GATEWAY_RESTART_CONTROLLER_ROOT",
    "GATEWAY_RESTART_AUTHORITY_ROOT",
    "GATEWAY_RESTART_AUTHORITY_COMMIT",
    "GATEWAY_ACTIVE_RELEASE_RESTART_INVOCATION_ID",
    "GATEWAY_ACTIVE_RELEASE_COMPONENT",
    "GATEWAY_PAIRED_RESTART",
    "GATEWAY_RESTART_GUARD_SCOPE",
    "PREPARED_GATEWAY_SHA",
    "LAB_ARENA_RESTART_GUARD_GENERATION",
    "GATEWAY_RESTART_RECOVERY_LOCK_FILE",
    "GATEWAY_V2_CONFIG_DIR",
    "GATEWAY_V2_OFFLINE_ARTIFACT_ROOT",
    "GATEWAY_V2_RELEASE_BUCKET",
    "GATEWAY_V2_RELEASE_ARCHIVE_ROOT",
    "GATEWAY_V2_RELEASE_MANIFEST",
    "GATEWAY_PREPARED_V2_RELEASE_MANIFEST",
    "GATEWAY_V2_RELEASE_PREFIX",
    "GATEWAY_RESTART_TEMP_CLEANUP_MIN_AGE_SECONDS",
    "GATEWAY_RESTART_EMERGENCY_BACKUP_MIN_AGE_SECONDS",
    "GATEWAY_RESTART_CLEANUP_MAX_CANDIDATES",
    "GATEWAY_TEE_FALLBACK_LOG_DIR",
    "GATEWAY_GIT_HELPER",
    "GATEWAY_HOST_MEMORY_GUARD_PATH",
    "GATEWAY_CONTROLLER_PROCESS_HELPER",
    "LAB_ARENA_PROCESS_HELPER",
    "GATEWAY_DEPENDENCY_INSTALL_FINGERPRINT",
    "GATEWAY_RESTART_PHASE",
    "ARENA_CONTINUITY_ENABLED", "ARENA_CANDIDATE_SLOT",
    "ARENA_CANDIDATE_TREE", "ARENA_PREVIOUS_MODE",
    "ARENA_PREVIOUS_SLOT", "ARENA_PREVIOUS_SHA",
    "ARENA_PREVIOUS_BACKEND", "ARENA_RECOVERING", "ARENA_REUSE_PREVIOUS",
    "ARENA_INGRESS_SWITCHED", "ARENA_RUNTIME_ROOT",
    "ARENA_INGRESS_STATE_FILE", "ARENA_CADDY_FILE",
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
    "GATEWAY_RUNTIME_SOURCE_ROOT",
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
    # The old runtime cannot supply the new controller's shutdown state.
    "GATEWAY_DESTRUCTIVE_PHASE_STARTED",
    "GATEWAY_CONTROLLER_PROCESS_STATE_FILE",
    "GATEWAY_PREFLIGHT_TREE",
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
    "GATEWAY_RESTART_GIT_SSH_COMMAND",
    "GATEWAY_PYTHON_BIN",
    "GATEWAY_RESTART_CONTROLLER_ROOT",
    "GATEWAY_RESTART_AUTHORITY_ROOT",
    "GATEWAY_RESTART_AUTHORITY_COMMIT",
    "GATEWAY_ACTIVE_RELEASE_RESTART_INVOCATION_ID",
    "GATEWAY_ACTIVE_RELEASE_COMPONENT",
    "GATEWAY_PAIRED_RESTART",
    "GATEWAY_RESTART_GUARD_SCOPE",
    "PREPARED_GATEWAY_SHA",
    "LAB_ARENA_RESTART_GUARD_GENERATION",
    "GATEWAY_RESTART_RECOVERY_LOCK_FILE",
    "GATEWAY_V2_CONFIG_DIR",
    "GATEWAY_V2_OFFLINE_ARTIFACT_ROOT",
    "GATEWAY_V2_RELEASE_BUCKET",
    "GATEWAY_V2_RELEASE_ARCHIVE_ROOT",
    "GATEWAY_V2_RELEASE_MANIFEST",
    "GATEWAY_PREPARED_V2_RELEASE_MANIFEST",
    "GATEWAY_V2_RELEASE_PREFIX",
    "GATEWAY_RESTART_TEMP_CLEANUP_MIN_AGE_SECONDS",
    "GATEWAY_RESTART_EMERGENCY_BACKUP_MIN_AGE_SECONDS",
    "GATEWAY_RESTART_CLEANUP_MAX_CANDIDATES",
    "GATEWAY_TEE_FALLBACK_LOG_DIR",
    "GATEWAY_GIT_HELPER",
    "GATEWAY_HOST_MEMORY_GUARD_PATH",
    "GATEWAY_CONTROLLER_PROCESS_HELPER",
    "LAB_ARENA_PROCESS_HELPER",
    "GATEWAY_DEPENDENCY_INSTALL_FINGERPRINT",
    "GATEWAY_RESTART_PHASE",
    "ARENA_CONTINUITY_ENABLED", "ARENA_CANDIDATE_SLOT",
    "ARENA_CANDIDATE_TREE", "ARENA_PREVIOUS_MODE",
    "ARENA_PREVIOUS_SLOT", "ARENA_PREVIOUS_SHA",
    "ARENA_PREVIOUS_BACKEND", "ARENA_RECOVERING", "ARENA_REUSE_PREVIOUS",
    "ARENA_INGRESS_SWITCHED", "ARENA_RUNTIME_ROOT",
    "ARENA_INGRESS_STATE_FILE", "ARENA_CADDY_FILE",
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
    "GATEWAY_RUNTIME_SOURCE_ROOT",
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

echo "Validating absolute gateway secret paths for canonical Git checkout"
(
  set -a
  . "$ENV_CLONE"
  set +a
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

"$GATEWAY_PYTHON_BIN" \
  "$GATEWAY_PREFLIGHT_TREE/gateway/tee/host_memory_guard_v2.py" \
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

# Retire only proven inherited locks from an exact-owner closed recovery before
# the first artifact worker can wait for Docker. Normal release gates follow.
recover_arena_inherited_docker_locks

echo "Preparing exact hash-locked enclave dependencies during release acquisition"
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
GATEWAY_LOCAL_RELEASE_SCRIPT="$GATEWAY_PREFLIGHT_TREE/gateway/tee/build_local_release_v2.sh"
GATEWAY_LOCAL_RELEASE_MODULE="$GATEWAY_PREFLIGHT_TREE/gateway/tee/local_release_v2.py"
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
  if ! run_prepared_gateway_module gateway.tee.release_channel_v2 \
        --ensure \
        --expected-commit "$PREPARED_GATEWAY_SHA" \
        --gateway-output "$GATEWAY_PREPARED_V2_RELEASE_MANIFEST"; then
    echo "ERROR: exact local gateway release identity is invalid" >&2
    echo "Gateway remains running; production shutdown has not started." >&2
    exit 75
  fi
  record_gateway_restart_timing "local_release_ready"
else
  echo "ERROR: selected release has an incomplete or unsupported V2 release acquisition contract" >&2
  echo "Gateway remains running; production shutdown has not started." >&2
  exit 75
fi
if ! follow_superseding_gateway_release; then
  echo "Gateway remains running; production shutdown has not started." >&2
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
if ! run_prepared_gateway_module gateway.tee.restart_preflight_v2 \
    --deploy-commit "$PREPARED_GATEWAY_SHA" \
    --release-manifest "$GATEWAY_PREPARED_V2_RELEASE_MANIFEST" \
    --topology-manifest "$GATEWAY_PREFLIGHT_TREE/gateway/tee/topology.json" \
    --config-dir "$GATEWAY_V2_CONFIG_DIR" \
    --topology-mode "${GATEWAY_TEE_TOPOLOGY_MODE:-full}"; then
  rm -rf "$GATEWAY_PREFLIGHT_TREE"
  echo "ERROR: prepared V2 release failed before-shutdown validation" >&2
  exit 1
fi

# Exercise the candidate's real Arena startup contract while the old service
# still owns the port.  --check-only performs the same startup checks as launch.
GATEWAY_DEPLOY_STAGE="lab_arena_startup_preflight"
export GATEWAY_DEPLOY_STAGE
if ! (
  cd "$GATEWAY_PREFLIGHT_TREE" || exit 1
  set -a
  . "$ENV_CLONE" || exit 1
  set +a
  PYTHONPATH="$GATEWAY_PREFLIGHT_TREE" "$GATEWAY_PYTHON_BIN" \
    "$GATEWAY_PREFLIGHT_TREE/scripts/run_lab_arena_service.py" \
    --environment-file "$GATEWAY_ENV_FILE" --check-only
); then
  echo "ERROR: prepared Lab Arena startup checks failed before shutdown" >&2
  echo "Gateway remains running; production shutdown has not started." >&2
  exit 1
fi
record_gateway_restart_timing "lab_arena_startup_preflight_complete"

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
GATEWAY_DEPLOY_STAGE="pre_shutdown_disk_capacity"
export GATEWAY_DEPLOY_STAGE
# Reserve 128 MiB for activation metadata and logs before the existing 10 GiB
# post-activation floor. A capacity failure leaves the old service alive.
if ! preflight_gateway_restart_disk "$((MIN_FREE_KB + 128 * 1024))"; then
  echo "Gateway remains running; production shutdown has not started." >&2
  exit 1
fi
record_gateway_restart_timing "pre_shutdown_disk_ready"
record_gateway_restart_timing "pre_shutdown_checks_complete"

# Keep Arena's existing admission proxy and API available while the single
# attested coordinator is replaced. This runs only for an active Arena mode.
ARENA_CONFIGURED_MODE="$(
  env -u LAB_ARENA_MODE PYTHONPATH="$GATEWAY_PREFLIGHT_TREE" \
    "$GATEWAY_PYTHON_BIN" -B - "$GATEWAY_ENV_FILE" <<'PY'
import os
import sys
from pathlib import Path
from scripts.run_lab_arena_service import load_scoped_environment
load_scoped_environment(Path(sys.argv[1]))
print(os.environ.get("LAB_ARENA_MODE", "off").strip().lower())
PY
)" || exit 1
case "$ARENA_CONFIGURED_MODE" in
  off)
    ARENA_CONTINUITY_ENABLED=0
    ARENA_INGRESS_BEFORE="$(arena_ingress_tool "$GATEWAY_PREFLIGHT_TREE" --inspect)" || exit 1
    if [ "$ARENA_INGRESS_BEFORE" != "legacy 0 - legacy" ]; then
      echo "ERROR: Arena cannot disable while the stable ingress owns :8000" >&2
      exit 1
    fi
    ;;
  shadow|live) ARENA_CONTINUITY_ENABLED=1 ;;
  *) echo "ERROR: Arena mode is invalid for ingress cutover" >&2; exit 1 ;;
esac
if [ "$ARENA_CONTINUITY_ENABLED" = 1 ]; then
  GATEWAY_DEPLOY_STAGE="arena_candidate_overlap"
  export GATEWAY_DEPLOY_STAGE
  ARENA_INGRESS_BEFORE="$(arena_ingress_tool "$GATEWAY_PREFLIGHT_TREE" --inspect)" || exit 1
  read -r ARENA_PREVIOUS_MODE ARENA_PREVIOUS_SLOT ARENA_PREVIOUS_SHA ARENA_PREVIOUS_BACKEND \
    <<< "$ARENA_INGRESS_BEFORE"
  case "$ARENA_PREVIOUS_MODE:$ARENA_PREVIOUS_BACKEND" in
    legacy:legacy) ARENA_CANDIDATE_SLOT=1 ;;
    domain:legacy)
      if [ "$PREPARED_GATEWAY_SHA" != "$ARENA_PREVIOUS_SHA" ]; then
        echo "ERROR: unfinished first Arena cutover must finish its exact release before another SHA" >&2
        exit 1
      fi
      ARENA_CANDIDATE_SLOT="$ARENA_PREVIOUS_SLOT"
      recover_arena_restart_owner || exit 1
      ARENA_REUSE_PREVIOUS=1
      ARENA_CANDIDATE_TREE="$(arena_runtime_tree "$ARENA_PREVIOUS_SLOT" "$ARENA_PREVIOUS_SHA")"
      ;;
    dual:closed|dual:ready)
      ARENA_CANDIDATE_SLOT=$((1 - ARENA_PREVIOUS_SLOT))
      if [ "$ARENA_PREVIOUS_BACKEND" = closed ]; then
        recover_arena_restart_owner || exit 1
      else
        ARENA_READY_GUARD_STATE="$(run_lab_arena_restart_guard \
          "$GATEWAY_PREFLIGHT_TREE" state)" || exit 1
        if "$GATEWAY_PYTHON_BIN" - "$ARENA_READY_GUARD_STATE" <<'PY'
import json
import sys
try:
    state = json.loads(sys.argv[1])
except (TypeError, ValueError):
    raise SystemExit(2)
if (not isinstance(state, dict)
        or state.get("schema_version") != "leadpoet.lab_arena.restart_guard_state.v1"
        or type(state.get("guard_present")) is not bool):
    raise SystemExit(2)
raise SystemExit(0 if state.get("guard_present") is True else 1)
PY
        then
          recover_arena_restart_owner || exit 1
        else
          state_status="$?"
          if [ "$state_status" != 1 ]; then
            echo "ERROR: Arena restart guard state is invalid" >&2
            exit 1
          fi
        fi
      fi
      if [ "$ARENA_RECOVERING" = 1 ]; then
        retire_arena_retained_overlap || exit 1
      fi
      ;;
    *) echo "ERROR: Arena ingress is not in a safe ready state" >&2; exit 1 ;;
  esac
  if [ "$ARENA_REUSE_PREVIOUS" != 1 ]; then
    prepare_arena_runtime_tree
  fi
  if ! preflight_gateway_restart_disk "$((MIN_FREE_KB + 128 * 1024))"; then
    echo "ERROR: retained Arena archive would violate the disk guard" >&2
    exit 1
  fi
  set -a
  . "$ENV_CLONE"
  set +a
  if [ "$ARENA_REUSE_PREVIOUS" = 1 ]; then
    arena_component_command api "$ARENA_CANDIDATE_SLOT"
    "$GATEWAY_PYTHON_BIN" "$GATEWAY_PREFLIGHT_TREE/scripts/manage_owned_process_group.py" verify \
      --state-file "$ARENA_COMPONENT_STATE_FILE" --cwd "$ARENA_CANDIDATE_TREE" \
      --uid "$(id -u)" -- "${ARENA_COMPONENT_COMMAND[@]}" >/dev/null
  else
    arena_slot_ports "$ARENA_CANDIDATE_SLOT"
    "$GATEWAY_PYTHON_BIN" - "$ARENA_API_PORT" "$ARENA_EDGE_PORT" <<'PY'
import socket
import sys
for value in sys.argv[1:]:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            listener.bind(("127.0.0.1", int(value)))
        except OSError:
            raise SystemExit("candidate Arena port is already occupied")
PY
    start_arena_component api "$ARENA_CANDIDATE_SLOT" "$ARENA_CANDIDATE_TREE" "$PREPARED_GATEWAY_SHA"
    start_arena_component edge "$ARENA_CANDIDATE_SLOT" "$ARENA_CANDIDATE_TREE" "$PREPARED_GATEWAY_SHA"
  fi
  arena_slot_ports "$ARENA_CANDIDATE_SLOT"
  ARENA_CANDIDATE_READY=0
  for attempt in $(seq 1 60); do
    if timeout 5 curl -fsS "http://127.0.0.1:$ARENA_API_PORT/arena/v1/current" >/dev/null 2>&1 \
        && timeout 5 curl -fsS "http://127.0.0.1:$ARENA_EDGE_PORT/arena/v1/current" >/dev/null 2>&1; then
      ARENA_CANDIDATE_READY=1
      break
    fi
    sleep 2
  done
  if [ "$ARENA_CANDIDATE_READY" != 1 ]; then
    echo "ERROR: candidate Arena API and edge were not ready before cutover" >&2
    exit 1
  fi
  for component in api edge; do
    arena_component_command "$component" "$ARENA_CANDIDATE_SLOT"
    "$GATEWAY_PYTHON_BIN" "$GATEWAY_PREFLIGHT_TREE/scripts/manage_owned_process_group.py" verify \
      --state-file "$ARENA_COMPONENT_STATE_FILE" --cwd "$ARENA_CANDIDATE_TREE" \
      --uid "$(id -u)" -- "${ARENA_COMPONENT_COMMAND[@]}" >/dev/null
  done
fi

GATEWAY_DEPLOY_STAGE="lab_arena_claim_drain"
export GATEWAY_DEPLOY_STAGE
if ! drain_lab_arena_for_restart "$GATEWAY_PREFLIGHT_TREE"; then
  echo "ERROR: Lab Arena leases did not drain before gateway shutdown" >&2
  echo "Gateway remains running; production shutdown has not started." >&2
  exit 1
fi

GATEWAY_LAB_ARENA_STOP_PROCESS_HELPER="$GATEWAY_CONTROLLER_PROCESS_HELPER"
if ! verify_controller_process_helper "$GATEWAY_LAB_ARENA_STOP_PROCESS_HELPER"; then
  echo "ERROR: verified controller Lab Arena stop helper is unavailable" >&2
  echo "Gateway remains running; production shutdown has not started." >&2
  exit 1
fi
if [ "$ARENA_CONTINUITY_ENABLED" = 1 ] \
    && { [ "$ARENA_PREVIOUS_MODE" = legacy ] || [ "$ARENA_PREVIOUS_MODE" = domain ]; }; then
  # Prove idle while the owned guard is still draining and can be aborted.
  # A failed proof leaves the old gateway and worker untouched.
  PYTHONPATH="$GATEWAY_PREFLIGHT_TREE" "$GATEWAY_PYTHON_BIN" -B \
    "$GATEWAY_PREFLIGHT_TREE/scripts/arena_first_cutover_idle.py" \
    --environment-file "$GATEWAY_ENV_FILE"
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
if [ "$ARENA_CONTINUITY_ENABLED" = 1 ]; then
  # The provider gate is process-local. Keep the old API serving until the
  # durable claim drain and guard authorize prove no active paid lease work.
  if [ "$ARENA_PREVIOUS_MODE" = legacy ]; then
    arena_ingress_transition "$GATEWAY_PREFLIGHT_TREE" \
      legacy 0 - legacy domain "$ARENA_CANDIDATE_SLOT" "$PREPARED_GATEWAY_SHA" legacy
  elif [ "$ARENA_PREVIOUS_MODE" = domain ]; then
    if [ "$ARENA_REUSE_PREVIOUS" != 1 ]; then
      arena_ingress_transition "$GATEWAY_PREFLIGHT_TREE" \
        domain "$ARENA_PREVIOUS_SLOT" "$ARENA_PREVIOUS_SHA" legacy \
        domain "$ARENA_CANDIDATE_SLOT" "$PREPARED_GATEWAY_SHA" legacy
    fi
  else
    arena_ingress_transition "$GATEWAY_PREFLIGHT_TREE" \
      dual "$ARENA_PREVIOUS_SLOT" "$ARENA_PREVIOUS_SHA" "$ARENA_PREVIOUS_BACKEND" \
      dual "$ARENA_CANDIDATE_SLOT" "$PREPARED_GATEWAY_SHA" "$ARENA_PREVIOUS_BACKEND"
  fi
  ARENA_INGRESS_SWITCHED=1
  record_gateway_restart_timing "arena_candidate_ingress_ready"
fi
echo "Stopping existing gateway and Lab Arena processes"
GATEWAY_DESTRUCTIVE_PHASE_STARTED=1
export GATEWAY_DESTRUCTIVE_PHASE_STARTED
record_gateway_restart_timing "shutdown_started"
if [ "$ARENA_CONTINUITY_ENABLED" = 1 ]; then
  if [ "$ARENA_PREVIOUS_MODE" = legacy ] || [ "$ARENA_PREVIOUS_MODE" = domain ]; then
    # Retire the legacy combined worker immediately after the idle proof.
    # Its old binary cannot stop queued reviews after the guard arrives.
    "$GATEWAY_PYTHON_BIN" "$GATEWAY_PREFLIGHT_TREE/scripts/manage_owned_process_group.py" stop \
      --state-file "$GATEWAY_CONTROLLER_PROCESS_STATE_FILE" \
      --cwd "$LEADPOET_REPO_ROOT" --uid "$(id -u)" \
      --term-seconds 6600 --no-kill --signal-leader-only -- \
      "$GATEWAY_PYTHON_BIN" -u scripts/run_lab_arena_service.py \
      --environment-file "$GATEWAY_ENV_FILE" --host 127.0.0.1 --port 8792
    # The old gateway still proxies direct :8000 to :8792 until Caddy can
    # bind that listener. Keep that route alive with a driver-free replica.
    if [ "$ARENA_CANDIDATE_SLOT" != 0 ]; then
      arena_component_command api 0
      if [ -e "$ARENA_COMPONENT_STATE_FILE" ] || [ -L "$ARENA_COMPONENT_STATE_FILE" ]; then
        "$GATEWAY_PYTHON_BIN" "$GATEWAY_PREFLIGHT_TREE/scripts/manage_owned_process_group.py" verify \
          --state-file "$ARENA_COMPONENT_STATE_FILE" --cwd "$ARENA_CANDIDATE_TREE" \
          --uid "$(id -u)" -- "${ARENA_COMPONENT_COMMAND[@]}" >/dev/null
      else
        start_arena_component api 0 "$ARENA_CANDIDATE_TREE" "$PREPARED_GATEWAY_SHA"
      fi
    fi
    ARENA_TEMP_API_READY=0
    for attempt in $(seq 1 30); do
      if timeout 5 curl -fsS http://127.0.0.1:8792/arena/v1/current >/dev/null 2>&1; then
        ARENA_TEMP_API_READY=1
        break
      fi
      sleep 2
    done
    if [ "$ARENA_TEMP_API_READY" != 1 ]; then
      echo "ERROR: temporary direct Arena API did not become ready" >&2
      exit 1
    fi
  else
    ARENA_PREVIOUS_TREE="$(arena_runtime_tree "$ARENA_PREVIOUS_SLOT" "$ARENA_PREVIOUS_SHA")"
    stop_arena_component driver "$ARENA_PREVIOUS_SLOT" "$ARENA_PREVIOUS_TREE"
    if [ "$ARENA_PREVIOUS_BACKEND" = ready ]; then
      arena_ingress_transition "$GATEWAY_PREFLIGHT_TREE" \
        dual "$ARENA_CANDIDATE_SLOT" "$PREPARED_GATEWAY_SHA" ready \
        dual "$ARENA_CANDIDATE_SLOT" "$PREPARED_GATEWAY_SHA" closed
    fi
  fi
  "$GATEWAY_PYTHON_BIN" "$GATEWAY_PREFLIGHT_TREE/scripts/manage_owned_process_group.py" stop \
    --state-file "/home/ec2-user/.config/leadpoet/gateway-process.json" \
    --cwd "$LEADPOET_REPO_ROOT" --uid "$(id -u)" \
    --term-seconds 6600 --no-kill --signal-leader-only -- \
    "$GATEWAY_PYTHON_BIN" -u -m gateway.main
  if [ "$ARENA_PREVIOUS_MODE" = legacy ] || [ "$ARENA_PREVIOUS_MODE" = domain ]; then
    if [ "$ARENA_CANDIDATE_SLOT" != 0 ]; then
      stop_arena_component api 0 "$ARENA_CANDIDATE_TREE"
    fi
    arena_ingress_transition "$GATEWAY_PREFLIGHT_TREE" \
      domain "$ARENA_CANDIDATE_SLOT" "$PREPARED_GATEWAY_SHA" legacy \
      dual "$ARENA_CANDIDATE_SLOT" "$PREPARED_GATEWAY_SHA" closed
  fi
  if [ "$ARENA_PREVIOUS_MODE" = dual ] \
      || { [ "$ARENA_PREVIOUS_MODE" = domain ] && [ "$ARENA_REUSE_PREVIOUS" != 1 ]; }; then
    # The new edge already serves both listeners. Drain old in-flight API
    # requests before retiring its archive, including on a failed backend
    # build; a later exact-owner retry can then reuse the inactive slot.
    ARENA_PREVIOUS_TREE="$(arena_runtime_tree "$ARENA_PREVIOUS_SLOT" "$ARENA_PREVIOUS_SHA")"
    stop_arena_component edge "$ARENA_PREVIOUS_SLOT" "$ARENA_PREVIOUS_TREE"
    stop_arena_component api "$ARENA_PREVIOUS_SLOT" "$ARENA_PREVIOUS_TREE"
    if [ -L "$ARENA_PREVIOUS_TREE" ] || [ ! -d "$ARENA_PREVIOUS_TREE" ]; then
      echo "ERROR: previous Arena runtime is not an owned directory" >&2
      exit 1
    fi
    retire_arena_runtime_tree "$ARENA_PREVIOUS_SLOT" "$ARENA_PREVIOUS_SHA" "$ARENA_PREVIOUS_TREE" || exit 1
  fi
else
  "$GATEWAY_PYTHON_BIN" "$GATEWAY_PREFLIGHT_TREE/scripts/manage_owned_process_group.py" stop \
    --state-file "/home/ec2-user/.config/leadpoet/gateway-process.json" \
    --cwd "$LEADPOET_REPO_ROOT" --uid "$(id -u)" \
    --term-seconds 6600 --no-kill --signal-leader-only -- \
    "$GATEWAY_PYTHON_BIN" -u -m gateway.main
  stop_lab_arena_service "$GATEWAY_LAB_ARENA_STOP_PROCESS_HELPER"
fi
record_gateway_restart_timing "shutdown_complete"
rm -rf "$GATEWAY_PREFLIGHT_TREE"
GATEWAY_PREFLIGHT_TREE=""


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

if [ "$ARENA_CONTINUITY_ENABLED" != 1 ]; then
  echo "Waiting for :8000 to free"
  for i in $(seq 1 25); do
    if ! sudo ss -tulpn 2>/dev/null | grep -q ":8000 "; then
      echo ":8000 free after ${i}s"
      break
    fi
    sleep 1
  done
fi

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
  GATEWAY_RESTART_GIT_SSH_COMMAND="$GATEWAY_RESTART_GIT_SSH_COMMAND" \
  GATEWAY_GIT_HELPER="$GATEWAY_GIT_HELPER" \
  GATEWAY_RESTART_CONTROLLER_ROOT="$GATEWAY_RESTART_CONTROLLER_ROOT" \
  GATEWAY_RESTART_AUTHORITY_ROOT="$GATEWAY_RESTART_AUTHORITY_ROOT" \
  GATEWAY_RESTART_AUTHORITY_COMMIT="$GATEWAY_RESTART_AUTHORITY_COMMIT" \
  GATEWAY_ACTIVE_RELEASE_RESTART_INVOCATION_ID="$GATEWAY_ACTIVE_RELEASE_RESTART_INVOCATION_ID" \
  GATEWAY_ACTIVE_RELEASE_COMPONENT="$GATEWAY_ACTIVE_RELEASE_COMPONENT" \
  GATEWAY_PAIRED_RESTART="$GATEWAY_PAIRED_RESTART" \
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
  ARENA_CONTINUITY_ENABLED="$ARENA_CONTINUITY_ENABLED" \
  ARENA_CANDIDATE_SLOT="$ARENA_CANDIDATE_SLOT" \
  ARENA_CANDIDATE_TREE="$ARENA_CANDIDATE_TREE" \
  ARENA_PREVIOUS_MODE="$ARENA_PREVIOUS_MODE" \
  ARENA_PREVIOUS_SLOT="$ARENA_PREVIOUS_SLOT" \
  ARENA_PREVIOUS_SHA="$ARENA_PREVIOUS_SHA" \
  GATEWAY_RESTART_TIMING_DIR="$GATEWAY_RESTART_TIMING_DIR" \
  GATEWAY_RESTART_TIMING_FILE="$GATEWAY_RESTART_TIMING_FILE" \
  GATEWAY_RESTART_TIMING_INITIALIZED="$GATEWAY_RESTART_TIMING_INITIALIZED" \
  GATEWAY_RELEASE_FOLLOW_ROOT="$GATEWAY_RELEASE_FOLLOW_ROOT" \
  GATEWAY_RELEASE_SUPERSESSION_COUNT="$GATEWAY_RELEASE_SUPERSESSION_COUNT" \
  GATEWAY_RELEASE_SUPERSESSION_MAX="$GATEWAY_RELEASE_SUPERSESSION_MAX" \
  GATEWAY_V2_CONFIG_DIR="$GATEWAY_V2_CONFIG_DIR" \
  GATEWAY_V2_RELEASE_MANIFEST="$GATEWAY_V2_RELEASE_MANIFEST" \
  GATEWAY_PREPARED_V2_RELEASE_MANIFEST="$GATEWAY_PREPARED_V2_RELEASE_MANIFEST" \
  GATEWAY_V2_RELEASE_BUCKET="$GATEWAY_V2_RELEASE_BUCKET" \
  GATEWAY_V2_RELEASE_PREFIX="$GATEWAY_V2_RELEASE_PREFIX" \
  GATEWAY_V2_OFFLINE_ARTIFACT_ROOT="$GATEWAY_V2_OFFLINE_ARTIFACT_ROOT" \
  GATEWAY_DEPLOY_STAGE="$GATEWAY_DEPLOY_STAGE" \
  bash "$GATEWAY_POST_ACTIVATE_REEXEC_SCRIPT" "$@"
fi

# Re-exec retains the caller's cwd, which Python searches before PYTHONPATH.
# Resolve post-activation modules from the exact activated checkout.
cd "$LEADPOET_REPO_ROOT"
bind_activated_gateway_guard_candidate || exit 1
if [ "$ARENA_CONTINUITY_ENABLED" = 1 ]; then
  if [ "$ARENA_CANDIDATE_TREE" != "$(arena_runtime_tree "$ARENA_CANDIDATE_SLOT" "$GATEWAY_DEPLOY_SHA")" ]; then
    echo "ERROR: retained Arena runtime path differs from activated candidate" >&2
    exit 1
  fi
  ARENA_INGRESS_AFTER="$(arena_ingress_tool "$LEADPOET_REPO_ROOT" --inspect)" || exit 1
  if [ "$ARENA_INGRESS_AFTER" != "dual $ARENA_CANDIDATE_SLOT $GATEWAY_DEPLOY_SHA closed" ]; then
    echo "ERROR: protected ingress was not closed before coordinator replacement" >&2
    exit 1
  fi
  PYTHONPATH="$LEADPOET_REPO_ROOT" "$GATEWAY_PYTHON_BIN" -B - \
      "$LEADPOET_REPO_ROOT" "$ARENA_CANDIDATE_TREE" "$GATEWAY_DEPLOY_SHA" <<'PY'
from pathlib import Path
import sys
from scripts.gateway_git_deploy import verify_materialized_tree
verify_materialized_tree(
    repo_root=Path(sys.argv[1]), materialized_root=Path(sys.argv[2]),
    target_sha=sys.argv[3], strict_extras=True,
)
PY
fi
record_gateway_restart_timing "candidate_activated"
echo "Installing the preflighted gateway manifest"
PYTHONPATH="$LEADPOET_REPO_ROOT" "$GATEWAY_PYTHON_BIN" \
  -m gateway.tee.install_gateway_release_state_v2 \
  --prepared-manifest "$GATEWAY_PREPARED_V2_RELEASE_MANIFEST" \
  --active-manifest "$GATEWAY_V2_RELEASE_MANIFEST" \
  --expected-commit "$GATEWAY_DEPLOY_SHA"
"$GATEWAY_PYTHON_BIN" \
  "$LEADPOET_REPO_ROOT/gateway/tee/host_memory_guard_v2.py" \
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

echo "Preflight disk cleanup for Docker/PCR0/Arena builds"
GATEWAY_DEPLOY_STAGE="docker_disk_cleanup"
export GATEWAY_DEPLOY_STAGE
ensure_docker_ready
df -h / /var/lib/docker 2>/dev/null || df -h /
sudo journalctl --vacuum-size=200M 2>/dev/null || true
run_bounded_restart_artifact_cleanup

sudo docker builder prune -af 2>/dev/null || true

reset_orphaned_docker_storage_if_needed

echo "Disk after cleanup"
df -h / /var/lib/docker 2>/dev/null || df -h /
sudo docker system df 2>/dev/null || true

FREE_KB="$(root_free_kb)"
DOCKER_BUILD_REQUIRED_KB="$(docker_build_required_kb)"
if ! [[ "$FREE_KB" =~ ^[0-9]+$ ]] || [ "$FREE_KB" -lt "$DOCKER_BUILD_REQUIRED_KB" ]; then
  echo "ERROR: insufficient free disk after activation: $(df -h / | tail -1)" >&2
  echo "Need at least $DOCKER_BUILD_REQUIRED_KB KiB free before starting gateway Arena Docker workloads." >&2
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
    "leadpoet_canonical",
    "qualification",
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
for module_name in ("leadpoet_canonical",):
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
  for role in gateway_coordinator; do
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
    bash ./start_enclave.sh
  record_gateway_restart_timing "gateway_enclaves_started"

  echo "Bootstrapping measured coordinator enclave runtime"
  GATEWAY_DEPLOY_STAGE="v2_runtime_bootstrap"
  export GATEWAY_DEPLOY_STAGE
  test -s "$GATEWAY_V2_RELEASE_MANIFEST" || {
    echo "ERROR: local V2 build identity is missing" >&2
    exit 1
  }
  PYTHONPATH="$LEADPOET_REPO_ROOT" "$GATEWAY_PYTHON_BIN" -m gateway.utils.tee_v2_bootstrap \
    --release-manifest "$GATEWAY_V2_RELEASE_MANIFEST" \
    --protected-workflow-manifest "$GATEWAY_ROOT/_attested_runtime/protected_workflows.json"
  record_gateway_restart_timing "v2_runtime_bootstrapped"

echo "Verifying measured V2 runtime identity readiness"
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
export PATH="$HOME/.local/bin:$PATH"
export AWS_REGION="${AWS_REGION:-us-east-1}"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"
export GATEWAY_ENV_FILE="${GATEWAY_ENV_FILE:-/home/ec2-user/.config/leadpoet/gateway.env}"
export LEADPOET_GATEWAY_ENV_SECRET_ID="${LEADPOET_GATEWAY_ENV_SECRET_ID:-leadpoet/prod/gateway/env}"
unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_PROFILE AWS_SESSION_TOKEN AWS_SECURITY_TOKEN
export LEADPOET_AWS_INSTANCE_ROLE_ONLY=true
export GATEWAY_BIND_PORT=8000
GATEWAY_BACKEND_HEALTH_URL="http://localhost:8000"
if [ "$ARENA_CONTINUITY_ENABLED" = 1 ]; then
  export GATEWAY_BIND_PORT=8001
  arena_slot_ports "$ARENA_CANDIDATE_SLOT"
  export LAB_ARENA_PROXY_MAIN_PORT="$ARENA_API_PORT"
  GATEWAY_BACKEND_HEALTH_URL="http://127.0.0.1:8001"
fi

# Release the build lock after the protected gateway runtime is ready.
. "$LEADPOET_REPO_ROOT/validator_tee/scripts/docker_operation_lock_v2.sh"
leadpoet_release_docker_operation_lock_v2

cd "$LEADPOET_REPO_ROOT"
env -u LEADPOET_DOCKER_OPERATION_LOCK_HELD \
  -u LEADPOET_DOCKER_OPERATION_ADMISSION_LOCK_HELD -u LEADPOET_DOCKER_OPERATION_LOCK_OWNER_PID \
  -u GATEWAY_GIT_HELPER \
  -u GATEWAY_HOST_MEMORY_GUARD_PATH \
  -u GATEWAY_CONTROLLER_PROCESS_HELPER \
  -u LAB_ARENA_PROCESS_HELPER \
  -u GATEWAY_RESTART_AUTHORITY_ROOT \
  -u GATEWAY_RESTART_AUTHORITY_COMMIT \
  -u GATEWAY_ACTIVE_RELEASE_RESTART_INVOCATION_ID \
  -u GATEWAY_ACTIVE_RELEASE_COMPONENT \
  -u GATEWAY_PAIRED_RESTART \
  -u GATEWAY_RESTART_GUARD_SCOPE \
  -u PREPARED_GATEWAY_SHA \
  -u LAB_ARENA_RESTART_GUARD_GENERATION \
  setsid "$GATEWAY_PYTHON_BIN" -u -m gateway.main \
  >> "$GATEWAY_LOG_FILE" 2>&1 < /dev/null \
  7>&- 8>&- 9>&- 190>&- 191>&- 192>&- 193>&- 194>&- 195>&- &

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
  if timeout 5 curl -fsS "$GATEWAY_BACKEND_HEALTH_URL/health" >/dev/null 2>&1; then
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
if [ "$ARENA_CONTINUITY_ENABLED" = 1 ]; then
  arena_component_command driver "$ARENA_CANDIDATE_SLOT"
  ARENA_DRIVER_LOG_OFFSET=0
  if [ -f "$ARENA_COMPONENT_LOG_FILE" ]; then
    ARENA_DRIVER_LOG_OFFSET="$(wc -c < "$ARENA_COMPONENT_LOG_FILE")"
  fi
  start_arena_component driver "$ARENA_CANDIDATE_SLOT" \
    "$ARENA_CANDIDATE_TREE" "$GATEWAY_DEPLOY_SHA"
  wait_for_arena_driver_startup "$ARENA_DRIVER_LOG_OFFSET"
  arena_slot_ports "$ARENA_CANDIDATE_SLOT"
  wait_for_arena_current_readiness "http://127.0.0.1:$ARENA_API_PORT/arena/v1/current"
  wait_for_arena_current_readiness "http://127.0.0.1:$ARENA_EDGE_PORT/arena/v1/current"
else
  start_lab_arena_service
fi
record_gateway_restart_timing "lab_arena_service_ready"
if ! run_lab_arena_restart_guard "$LEADPOET_REPO_ROOT" ready \
    --generation "$LAB_ARENA_RESTART_GUARD_GENERATION" \
    --phase gateway_ready >/dev/null; then
  echo "ERROR: Lab Arena gateway readiness could not be recorded" >&2
  exit 1
fi
BUILD_INFO_RESPONSE="$(timeout 15 curl -fsS "$GATEWAY_BACKEND_HEALTH_URL/build-info")"
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
timeout 30 curl -fsS "$GATEWAY_BACKEND_HEALTH_URL/attestation/document" >/dev/null

GATEWAY_DEPLOY_STAGE="host_restart_script_install"
export GATEWAY_DEPLOY_STAGE
install_successful_restart_script

if [ "$ARENA_CONTINUITY_ENABLED" = 1 ]; then
  # Only the exact activated backend may receive protected gateway traffic.
  arena_ingress_transition "$LEADPOET_REPO_ROOT" \
    dual "$ARENA_CANDIDATE_SLOT" "$GATEWAY_DEPLOY_SHA" closed \
    dual "$ARENA_CANDIDATE_SLOT" "$GATEWAY_DEPLOY_SHA" ready
  timeout 15 curl -fsS http://127.0.0.1:8000/arena/v1/current >/dev/null
  timeout 15 curl -fsS http://127.0.0.1:8000/build-info >/dev/null
  timeout 15 curl -fsS --resolve gateway.subnet71.com:443:127.0.0.1 \
    https://gateway.subnet71.com/arena/v1/current >/dev/null
  timeout 15 curl -fsS --resolve gateway.subnet71.com:443:127.0.0.1 \
    https://gateway.subnet71.com/build-info >/dev/null
fi
GATEWAY_DEPLOY_STAGE="lab_arena_claim_guard_release"
export GATEWAY_DEPLOY_STAGE
if [ "$GATEWAY_PAIRED_RESTART" = 1 ]; then
  # Only the paired coordinator may advance the same owner/generation through
  # validator readiness. Never refill between the two canonical restarts.
  paired_state="$(run_lab_arena_restart_guard "$LEADPOET_REPO_ROOT" owned \
    --generation "$LAB_ARENA_RESTART_GUARD_GENERATION")"
  "$GATEWAY_PYTHON_BIN" - "$PREPARED_GATEWAY_SHA" \
      "$GATEWAY_ACTIVE_RELEASE_RESTART_INVOCATION_ID" "$paired_state" \
      "$LAB_ARENA_RESTART_GUARD_GENERATION" <<'PAIRED_GUARD'
import json, sys
state = json.loads(sys.argv[3])
if (state.get("candidate_commit") != sys.argv[1]
        or state.get("guard_generation") != int(sys.argv[4])
        or state.get("restart_scope") != "all"
        or state.get("restart_phase") != "gateway_ready"
        or state.get("guard_active") is not True
        or state.get("paused") is not True
        or state.get("drain", {}).get("preserved") is not True):
    raise SystemExit("paired gateway guard is not ready")
print("ARENA_PAIRED_GUARD=" + json.dumps({"invocation": sys.argv[2], "state": state}, separators=(",", ":")))
PAIRED_GUARD
else
  if ! run_lab_arena_restart_guard "$LEADPOET_REPO_ROOT" release \
      --generation "$LAB_ARENA_RESTART_GUARD_GENERATION" >/dev/null; then
    echo "ERROR: Lab Arena restart guard release failed after gateway readiness" >&2
    exit 1
  fi
  LAB_ARENA_RESTART_GUARD_GENERATION=""
fi
GATEWAY_DEPLOY_STAGE="completed"
export GATEWAY_DEPLOY_STAGE
finalize_deployment_record succeeded "$GATEWAY_DEPLOY_STAGE" >/dev/null
GATEWAY_DEPLOY_COMPLETED=1
rm -f "$GATEWAY_DEPLOY_PLAN_FILE" || true
record_gateway_restart_timing "completed" "passed"
echo "Gateway restart command completed; tail logs with: tail -f $GATEWAY_LOG_FILE"
