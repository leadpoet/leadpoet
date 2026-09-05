#!/bin/bash
set -euo pipefail

VALIDATOR_ROOT="${VALIDATOR_ROOT:-/home/ec2-user/leadpoet/leadpoet}"
VALIDATOR_ENV_FILE="${VALIDATOR_ENV_FILE:-/home/ec2-user/.config/leadpoet/validator.env}"
LEADPOET_VALIDATOR_ENV_SECRET_ID="${LEADPOET_VALIDATOR_ENV_SECRET_ID:-leadpoet/prod/validator/env}"
VALIDATOR_ENV_BACKUP_DIR="${VALIDATOR_ENV_BACKUP_DIR:-/home/ec2-user/.config/leadpoet/env-backups}"
EXPECTED_AWS_ACCOUNT="${EXPECTED_AWS_ACCOUNT:-493765492819}"
HISTORICAL_THREE_ROLE_TOPOLOGY_HASH="sha256:a13a1b16fb1501f953b2396aba88b87d7e5e0d3cfac4079b9230ea6165a88f34"
HISTORICAL_THREE_ROLE_TOPOLOGY_BLOB="f79cf108e4a98ca950a0087d786958f92c5f691f"
VALIDATOR_HISTORICAL_TOPOLOGY_HASH=""
# Interpreter for the long-lived validator process. The hydrated environment
# can select the existing production venv without changing restart behavior.
VALIDATOR_PYTHON_BIN="${VALIDATOR_PYTHON_BIN:-python3}"
VALIDATOR_TELEMETRY_PYTHON_BIN="$VALIDATOR_PYTHON_BIN"
VALIDATOR_TELEMETRY_CACHE_ROOT="${VALIDATOR_TELEMETRY_CACHE_ROOT:-$HOME/.cache/leadpoet-observability/validator-host}"
VALIDATOR_V2_GATEWAY_RELEASE_MANIFEST="${VALIDATOR_V2_GATEWAY_RELEASE_MANIFEST:-/home/ec2-user/.config/leadpoet/gateway-v2-release-manifest.json}"
VALIDATOR_V2_GATEWAY_RELEASE_LINEAGE="${VALIDATOR_V2_GATEWAY_RELEASE_LINEAGE:-/home/ec2-user/.config/leadpoet/gateway-v2-release-lineage.json}"
VALIDATOR_V2_GATEWAY_RELEASE_REQUIREMENTS="${VALIDATOR_V2_GATEWAY_RELEASE_REQUIREMENTS:-/home/ec2-user/.config/leadpoet/gateway-v2-release-requirements.json}"
VALIDATOR_V2_RELEASE_MANIFEST="${VALIDATOR_V2_RELEASE_MANIFEST:-/home/ec2-user/.config/leadpoet/validator-v2-release-manifest.json}"
VALIDATOR_V2_RELEASE_ARCHIVE_ROOT="${VALIDATOR_V2_RELEASE_ARCHIVE_ROOT:-/home/ec2-user/.config/leadpoet/validator-releases-v2}"
VALIDATOR_RESTART_TEMP_CLEANUP_MIN_AGE_SECONDS="${VALIDATOR_RESTART_TEMP_CLEANUP_MIN_AGE_SECONDS:-86400}"
VALIDATOR_RESTART_CLEANUP_MAX_CANDIDATES="${VALIDATOR_RESTART_CLEANUP_MAX_CANDIDATES:-64}"
VALIDATOR_V2_HOTKEY_CONFIG="${VALIDATOR_V2_HOTKEY_CONFIG:-/home/ec2-user/.config/leadpoet/validator-hotkey-config-v2.json}"
VALIDATOR_V2_HOTKEY_ENVELOPE="${VALIDATOR_V2_HOTKEY_ENVELOPE:-/home/ec2-user/.config/leadpoet/validator-hotkey-envelope-v2.json}"
VALIDATOR_RESTART_CONTROLLER_ROOT="${VALIDATOR_RESTART_CONTROLLER_ROOT:-/home/ec2-user/.config/leadpoet/restart-controller/validator}"
VALIDATOR_RESTART_CONTROLLER_CURRENT="$VALIDATOR_RESTART_CONTROLLER_ROOT/current"
VALIDATOR_HOST_RESTART_SCRIPT="${VALIDATOR_HOST_RESTART_SCRIPT:-/home/ec2-user/validator_restart.sh}"
VALIDATOR_EXACT_COMMIT_HELPER_SOURCE="$VALIDATOR_ROOT/Leadpoet/utils/exact_commit_restart_v2.py"
VALIDATOR_PINNED_GATEWAY_VERIFIER_SOURCE="$VALIDATOR_ROOT/validator_tee/scripts/verify_pinned_gateway_release_v2.sh"
if [ -n "${VALIDATOR_ACTIVE_RELEASE_AUTHORITY_ROOT:-}" ] \
    && [ -r "$VALIDATOR_ACTIVE_RELEASE_AUTHORITY_ROOT/Leadpoet/utils/exact_commit_restart_v2.py" ]; then
  VALIDATOR_EXACT_COMMIT_HELPER_SOURCE="$VALIDATOR_ACTIVE_RELEASE_AUTHORITY_ROOT/Leadpoet/utils/exact_commit_restart_v2.py"
  VALIDATOR_PINNED_GATEWAY_VERIFIER_SOURCE="$VALIDATOR_ACTIVE_RELEASE_AUTHORITY_ROOT/validator_tee/scripts/verify_pinned_gateway_release_v2.sh"
elif [ -r "$VALIDATOR_RESTART_CONTROLLER_CURRENT/Leadpoet/utils/exact_commit_restart_v2.py" ]; then
  VALIDATOR_EXACT_COMMIT_HELPER_SOURCE="$VALIDATOR_RESTART_CONTROLLER_CURRENT/Leadpoet/utils/exact_commit_restart_v2.py"
  VALIDATOR_PINNED_GATEWAY_VERIFIER_SOURCE="$VALIDATOR_RESTART_CONTROLLER_CURRENT/validator_tee/scripts/verify_pinned_gateway_release_v2.sh"
fi
VALIDATOR_DOCKER_OPERATION_LOCK_HELPER="$VALIDATOR_ROOT/validator_tee/scripts/docker_operation_lock_v2.sh"
VALIDATOR_PINNED_GATEWAY_VERIFIER=""
VALIDATOR_RESTART_CONTROLLER_SOURCE_DIR=""
VALIDATOR_DESTRUCTIVE_PHASE_STARTED=0
VALIDATOR_RESTART_COMPLETED=0
VALIDATOR_DOCKER_LOCK_ACQUIRED=0
VALIDATOR_DEPLOY_STAGE="${VALIDATOR_DEPLOY_STAGE:-bootstrap}"
VALIDATOR_RESTART_STARTED_EPOCH="${VALIDATOR_RESTART_STARTED_EPOCH:-$(date -u +%s)}"
VALIDATOR_RESTART_INVOCATION_ID="${VALIDATOR_RESTART_INVOCATION_ID:-validator-${VALIDATOR_RESTART_STARTED_EPOCH}-$$}"
VALIDATOR_ACTIVE_RELEASE_AUTHORITY_ROOT="${VALIDATOR_ACTIVE_RELEASE_AUTHORITY_ROOT:-}"
VALIDATOR_ACTIVE_RELEASE_AUTHORITY_COMMIT="${VALIDATOR_ACTIVE_RELEASE_AUTHORITY_COMMIT:-}"
VALIDATOR_ACTIVE_RELEASE_RESTART_INVOCATION_ID="${VALIDATOR_ACTIVE_RELEASE_RESTART_INVOCATION_ID:-$VALIDATOR_RESTART_INVOCATION_ID}"
VALIDATOR_PAIRED_ACTIVE_RELEASE_REQUIRED="${VALIDATOR_PAIRED_ACTIVE_RELEASE_REQUIRED:-0}"
VALIDATOR_RELEASE_ATTEMPTS_USED="${VALIDATOR_RELEASE_ATTEMPTS_USED:-0}"
VALIDATOR_RESTART_TIMING_DIR="${VALIDATOR_RESTART_TIMING_DIR:-/home/ec2-user/.config/leadpoet/restart-timings}"
VALIDATOR_RESTART_TIMING_FILE="${VALIDATOR_RESTART_TIMING_FILE:-$VALIDATOR_RESTART_TIMING_DIR/validator-${VALIDATOR_RESTART_STARTED_EPOCH}-$$.jsonl}"
VALIDATOR_RESTART_TIMING_INITIALIZED="${VALIDATOR_RESTART_TIMING_INITIALIZED:-0}"
VALIDATOR_RELEASE_SUPERSESSION_COUNT="${VALIDATOR_RELEASE_SUPERSESSION_COUNT:-0}"
VALIDATOR_RELEASE_SUPERSESSION_MAX="${VALIDATOR_RELEASE_SUPERSESSION_MAX:-20}"
VALIDATOR_V2_RELEASE_BUCKET="${VALIDATOR_V2_RELEASE_BUCKET:-leadpoet-attested-v2-artifacts-493765492819}"
VALIDATOR_V2_RELEASE_PREFIX="${VALIDATOR_V2_RELEASE_PREFIX:-attested-v2/releases}"
VALIDATOR_ACTIVE_RELEASE_REQUIREMENTS_OUTPUT="${VALIDATOR_ACTIVE_RELEASE_REQUIREMENTS_OUTPUT:-}"
VALIDATOR_FINAL_RELEASE_REQUIREMENTS_INPUT="${VALIDATOR_FINAL_RELEASE_REQUIREMENTS_INPUT:-}"
VALIDATOR_FINAL_RELEASE_LINEAGE_INPUT="${VALIDATOR_FINAL_RELEASE_LINEAGE_INPUT:-}"
VALIDATOR_MISSING_RUNTIME_RECOVERY_REQUIREMENTS="${VALIDATOR_MISSING_RUNTIME_RECOVERY_REQUIREMENTS:-}"
VALIDATOR_MISSING_RUNTIME_RECOVERY_LINEAGE="${VALIDATOR_MISSING_RUNTIME_RECOVERY_LINEAGE:-}"
VALIDATOR_ACTIVE_PUBLICATION_JOURNAL="$VALIDATOR_ROOT/validator_weights/authoritative_weight_publication_v2.json"
VALIDATOR_STATEFUL_CUTOVER_MANIFEST="/home/ec2-user/.config/leadpoet/stateful-epoch-cutover.json"
VALIDATOR_RESTART_START_PATH="/home/ec2-user/.config/leadpoet/restart-start-v1.json"
VALIDATOR_USE_CAPTURED_RESTART_START="${LEADPOET_USE_CAPTURED_RESTART_START:-0}"
REQUESTED_STATEFUL_CUTOVER_PREPARE_ONLY="${VALIDATOR_STATEFUL_CUTOVER_PREPARE_ONLY:-0}"
unset VALIDATOR_STATEFUL_CUTOVER_PREPARE_ONLY
export VALIDATOR_V2_OFFLINE_ARTIFACT_ROOT="${VALIDATOR_V2_OFFLINE_ARTIFACT_ROOT:-$HOME/.cache/leadpoet-v2-artifacts/validator-runtime}"
VALIDATOR_WALLET_ROOT="${VALIDATOR_WALLET_ROOT:-$HOME/.bittensor/wallets}"
VALIDATOR_WALLET_NAME="${VALIDATOR_WALLET_NAME:-validator_72}"
VALIDATOR_WALLET_HOTKEY="${VALIDATOR_WALLET_HOTKEY:-default}"
LAB_ARENA_RUNNER_LOG_FILE="${LAB_ARENA_RUNNER_LOG_FILE:-/home/ec2-user/logs/lab_arena_runner.log}"
REQUESTED_VALIDATOR_DEPLOY_COMMIT="${VALIDATOR_DEPLOY_COMMIT:-}"
unset VALIDATOR_DEPLOY_COMMIT
REQUESTED_COORDINATED_EXPECTED_COMMIT="${VALIDATOR_COORDINATED_EXPECTED_COMMIT:-}"
unset VALIDATOR_COORDINATED_EXPECTED_COMMIT
export VALIDATOR_RESTART_INVOCATION_ID
export LEADPOET_RESTART_INVOCATION_ID="$VALIDATOR_RESTART_INVOCATION_ID"

if [ -n "$VALIDATOR_ACTIVE_RELEASE_AUTHORITY_ROOT" ]; then
  if ! [[ "$VALIDATOR_ACTIVE_RELEASE_AUTHORITY_ROOT" =~ ^/tmp/validator-restart-controller-bootstrap\.[A-Za-z0-9]+/authority$ ]] \
      || ! [[ "$VALIDATOR_ACTIVE_RELEASE_AUTHORITY_COMMIT" =~ ^[0-9a-f]{40}$ ]] \
      || [ ! -r "$VALIDATOR_ACTIVE_RELEASE_AUTHORITY_ROOT/validator_restart.sh" ] \
      || [ -L "$VALIDATOR_ACTIVE_RELEASE_AUTHORITY_ROOT/validator_restart.sh" ]; then
    echo "ERROR: validator active release authority controller is invalid" >&2
    exit 2
  fi
fi
case "$VALIDATOR_PAIRED_ACTIVE_RELEASE_REQUIRED" in
  0|1) ;;
  *)
    echo "ERROR: VALIDATOR_PAIRED_ACTIVE_RELEASE_REQUIRED must be 0 or 1" >&2
    exit 2
    ;;
esac
if ! [[ "$VALIDATOR_ACTIVE_RELEASE_RESTART_INVOCATION_ID" =~ ^[a-z0-9][a-z0-9_.:-]{0,127}$ ]]; then
  echo "ERROR: validator active release restart invocation identity is invalid" >&2
  exit 2
fi

run_bounded_validator_restart_artifact_cleanup() {
  local gateway_build_root validator_build_root
  gateway_build_root="${GATEWAY_V2_BUILD_WORK_ROOT:-$HOME/.cache/leadpoet/gateway-release-build-v2}"
  validator_build_root="${VALIDATOR_V2_BUILD_WORK_ROOT:-$HOME/.cache/leadpoet/validator-pcr0-normalizer-v2}"
  if [ ! -r "$VALIDATOR_ROOT/validator_tee/host/restart_artifact_cleanup_v2.py" ]; then
    echo "WARNING: bounded validator restart artifact cleanup helper is unavailable" >&2
    return 0
  fi
  if ! sudo env PYTHONPATH="$VALIDATOR_ROOT" "$VALIDATOR_PYTHON_BIN" \
      -m validator_tee.host.restart_artifact_cleanup_v2 \
      --apply \
      --temporary-root /tmp \
      --gateway-build-root "$gateway_build_root" \
      --validator-build-root "$validator_build_root" \
      --docker-lock-file "$LEADPOET_DOCKER_OPERATION_LOCK_FILE" \
      --docker-lock-owner-pid "${LEADPOET_DOCKER_OPERATION_LOCK_OWNER_PID:-$$}" \
      --temp-min-age-seconds "$VALIDATOR_RESTART_TEMP_CLEANUP_MIN_AGE_SECONDS" \
      --max-candidates "$VALIDATOR_RESTART_CLEANUP_MAX_CANDIDATES" \
      --allowed-owner-uid "$(id -u)"; then
    echo "WARNING: bounded validator restart artifact cleanup failed closed" >&2
  fi
}

stop_lab_arena_runner() {
  sudo pkill -TERM -f "scripts/run_lab_arena_runner[.]py" 2>/dev/null || true
  sleep 1
  sudo pkill -KILL -f "scripts/run_lab_arena_runner[.]py" 2>/dev/null || true
}

start_lab_arena_runner() {
  local mode api_base runsc_name runsc_path pid
  mode="${LAB_ARENA_MODE:-off}"
  case "$mode" in
    off)
      echo "Lab Arena runner is disabled"
      return 0
      ;;
    shadow|live) ;;
    *)
      echo "ERROR: LAB_ARENA_MODE must be off, shadow, or live" >&2
      return 1
      ;;
  esac
  api_base="${LAB_ARENA_API_BASE_URL:-$VALIDATOR_V2_GATEWAY_URL}"
  if [ -z "$api_base" ]; then
    echo "ERROR: Lab Arena runner API URL is unavailable" >&2
    return 1
  fi
  if [ ! -r "$VALIDATOR_ROOT/scripts/run_lab_arena_runner.py" ]; then
    echo "ERROR: Lab Arena runner entrypoint is unavailable" >&2
    return 1
  fi
  runsc_path="${LAB_ARENA_RUNSC_PATH:-}"
  if [ -z "$runsc_path" ]; then
    runsc_name="$(
      "$VALIDATOR_PYTHON_BIN" -c \
        'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["artifact_filename"])' \
        "$VALIDATOR_ROOT/gateway/tee/runsc-runtime.lock.json"
    )"
    runsc_path="$GATEWAY_V2_OFFLINE_ARTIFACT_ROOT/$runsc_name"
  fi
  if [ ! -x "$runsc_path" ]; then
    echo "ERROR: verified Lab Arena runsc binary is unavailable" >&2
    return 1
  fi
  mkdir -p "$(dirname "$LAB_ARENA_RUNNER_LOG_FILE")"
  cd "$VALIDATOR_ROOT"
  setsid sudo env \
    PYTHONPATH="$VALIDATOR_ROOT" \
    PYTHONDONTWRITEBYTECODE=1 \
    LAB_ARENA_API_BASE_URL="$api_base" \
    LAB_ARENA_WALLET_NAME="${LAB_ARENA_WALLET_NAME:-$VALIDATOR_WALLET_NAME}" \
    LAB_ARENA_HOTKEY_NAME="${LAB_ARENA_HOTKEY_NAME:-$VALIDATOR_WALLET_HOTKEY}" \
    LAB_ARENA_WALLET_PATH="${LAB_ARENA_WALLET_PATH:-$VALIDATOR_WALLET_ROOT}" \
    LAB_ARENA_RUNNER_WORK_DIR="${LAB_ARENA_RUNNER_WORK_DIR:-/var/lib/lab-arena/runner}" \
    LAB_ARENA_RUNSC_PATH="$runsc_path" \
    LAB_ARENA_REGISTRY_REPOSITORY="${LAB_ARENA_REGISTRY_REPOSITORY:-}" \
    LAB_ARENA_MAX_PARALLEL_RUNS="${LAB_ARENA_MAX_PARALLEL_RUNS:-8}" \
    "$VALIDATOR_PYTHON_BIN" -u scripts/run_lab_arena_runner.py \
      > "$LAB_ARENA_RUNNER_LOG_FILE" 2>&1 < /dev/null &
  pid="$!"
  sleep 3
  if ! sudo kill -0 "$pid" 2>/dev/null; then
    tail -120 "$LAB_ARENA_RUNNER_LOG_FILE" >&2 || true
    echo "ERROR: Lab Arena runner exited during startup" >&2
    wait "$pid" 2>/dev/null || true
    return 1
  fi
  echo "Lab Arena runner started"
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --commit)
      if [ "$#" -lt 2 ] || [ -z "${2:-}" ]; then
        echo "ERROR: --commit requires a full 40-character SHA" >&2
        exit 2
      fi
      if [ -n "$REQUESTED_VALIDATOR_DEPLOY_COMMIT" ] \
          && [ "$REQUESTED_VALIDATOR_DEPLOY_COMMIT" != "$2" ]; then
        echo "ERROR: --commit conflicts with VALIDATOR_DEPLOY_COMMIT" >&2
        exit 2
      fi
      REQUESTED_VALIDATOR_DEPLOY_COMMIT="$2"
      shift 2
      ;;
    --commit=*)
      requested_commit="${1#--commit=}"
      if [ -z "$requested_commit" ]; then
        echo "ERROR: --commit requires a full 40-character SHA" >&2
        exit 2
      fi
      if [ -n "$REQUESTED_VALIDATOR_DEPLOY_COMMIT" ] \
          && [ "$REQUESTED_VALIDATOR_DEPLOY_COMMIT" != "$requested_commit" ]; then
        echo "ERROR: --commit conflicts with VALIDATOR_DEPLOY_COMMIT" >&2
        exit 2
      fi
      REQUESTED_VALIDATOR_DEPLOY_COMMIT="$requested_commit"
      shift
      ;;
    *)
      echo "ERROR: unsupported validator restart argument: $1" >&2
      exit 2
      ;;
  esac
done
if [ -n "$REQUESTED_VALIDATOR_DEPLOY_COMMIT" ] \
    && ! [[ "$REQUESTED_VALIDATOR_DEPLOY_COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
  echo "ERROR: --commit must be a lowercase full 40-character SHA" >&2
  exit 2
fi
if [ -n "$REQUESTED_COORDINATED_EXPECTED_COMMIT" ] \
    && ! [[ "$REQUESTED_COORDINATED_EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
  echo "ERROR: VALIDATOR_COORDINATED_EXPECTED_COMMIT must be a lowercase full 40-character SHA" >&2
  exit 2
fi
if ! [[ "$VALIDATOR_RELEASE_SUPERSESSION_COUNT" =~ ^[0-9]+$ ]] \
    || ! [[ "$VALIDATOR_RELEASE_SUPERSESSION_MAX" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: validator release supersession counters are invalid" >&2
  exit 2
fi
if [ -n "$REQUESTED_VALIDATOR_DEPLOY_COMMIT" ] \
    && [ -n "$REQUESTED_COORDINATED_EXPECTED_COMMIT" ]; then
  echo "ERROR: coordinated forward commit conflicts with exact-commit rollback" >&2
  exit 2
fi

validate_validator_active_release_handoff_path() {
  local variable_name="$1"
  local path="$2"
  if [ -z "$path" ]; then
    echo "ERROR: $variable_name is required for the paired active release handoff" >&2
    return 1
  fi
  if ! [[ "$path" =~ ^/tmp/leadpoet-[A-Za-z0-9._-]+\.json$ ]]; then
    echo "ERROR: $variable_name must be one exact controller-owned /tmp/leadpoet-*.json path" >&2
    return 1
  fi
  if [ -L "$path" ] || [ -d "$path" ]; then
    echo "ERROR: $variable_name must not resolve to a symlink or directory" >&2
    return 1
  fi
}

active_release_handoff_count=0
for handoff_value in \
  "$VALIDATOR_ACTIVE_RELEASE_REQUIREMENTS_OUTPUT" \
  "$VALIDATOR_FINAL_RELEASE_REQUIREMENTS_INPUT" \
  "$VALIDATOR_FINAL_RELEASE_LINEAGE_INPUT"; do
  [ -n "$handoff_value" ] \
    && active_release_handoff_count=$((active_release_handoff_count + 1))
done
if [ "$VALIDATOR_PAIRED_ACTIVE_RELEASE_REQUIRED" = "1" ] \
    && [ "$active_release_handoff_count" -ne 3 ]; then
  echo "ERROR: paired validator active release handoff is incomplete" >&2
  exit 2
fi
if [ "$active_release_handoff_count" -ne 0 ] \
    && [ "$active_release_handoff_count" -ne 3 ]; then
  echo "ERROR: validator active release handoff paths must be supplied together" >&2
  exit 2
fi
if [ "$active_release_handoff_count" -eq 3 ]; then
  validate_validator_active_release_handoff_path \
    VALIDATOR_ACTIVE_RELEASE_REQUIREMENTS_OUTPUT \
    "$VALIDATOR_ACTIVE_RELEASE_REQUIREMENTS_OUTPUT" || exit 2
  validate_validator_active_release_handoff_path \
    VALIDATOR_FINAL_RELEASE_REQUIREMENTS_INPUT \
    "$VALIDATOR_FINAL_RELEASE_REQUIREMENTS_INPUT" || exit 2
  validate_validator_active_release_handoff_path \
    VALIDATOR_FINAL_RELEASE_LINEAGE_INPUT \
    "$VALIDATOR_FINAL_RELEASE_LINEAGE_INPUT" || exit 2
else
  VALIDATOR_ACTIVE_RELEASE_REQUIREMENTS_OUTPUT="/tmp/leadpoet-validator-standalone-initial.$$.json"
  VALIDATOR_FINAL_RELEASE_REQUIREMENTS_INPUT="$VALIDATOR_V2_GATEWAY_RELEASE_REQUIREMENTS"
  VALIDATOR_FINAL_RELEASE_LINEAGE_INPUT="$VALIDATOR_V2_GATEWAY_RELEASE_LINEAGE"
fi
if [ "$VALIDATOR_ACTIVE_RELEASE_REQUIREMENTS_OUTPUT" = "$VALIDATOR_FINAL_RELEASE_REQUIREMENTS_INPUT" ] \
    || [ "$VALIDATOR_ACTIVE_RELEASE_REQUIREMENTS_OUTPUT" = "$VALIDATOR_FINAL_RELEASE_LINEAGE_INPUT" ] \
    || [ "$VALIDATOR_FINAL_RELEASE_REQUIREMENTS_INPUT" = "$VALIDATOR_FINAL_RELEASE_LINEAGE_INPUT" ]; then
  echo "ERROR: validator active release handoff paths must be distinct" >&2
  exit 2
fi

reset_standalone_active_release_handoff_for_reexec() {
  if [ "$active_release_handoff_count" -eq 0 ]; then
    VALIDATOR_ACTIVE_RELEASE_REQUIREMENTS_OUTPUT=""
    VALIDATOR_FINAL_RELEASE_REQUIREMENTS_INPUT=""
    VALIDATOR_FINAL_RELEASE_LINEAGE_INPUT=""
  fi
}

for stable_path in \
  "$VALIDATOR_V2_GATEWAY_RELEASE_REQUIREMENTS" \
  "$VALIDATOR_V2_GATEWAY_RELEASE_LINEAGE"; do
  if [ -z "$stable_path" ] || [[ "$stable_path" != /* ]] \
      || [ "$stable_path" = "/" ] || [[ "$stable_path" == *"/../"* ]] \
      || [[ "$stable_path" == */.. ]]; then
    echo "ERROR: validator stable active release path is unsafe" >&2
    exit 2
  fi
done
if [ "$VALIDATOR_V2_GATEWAY_RELEASE_REQUIREMENTS" = "$VALIDATOR_V2_GATEWAY_RELEASE_LINEAGE" ]; then
  echo "ERROR: validator active release requirements and lineage paths must differ" >&2
  exit 2
fi

missing_runtime_recovery_count=0
for recovery_value in \
  "$VALIDATOR_MISSING_RUNTIME_RECOVERY_REQUIREMENTS" \
  "$VALIDATOR_MISSING_RUNTIME_RECOVERY_LINEAGE"; do
  [ -n "$recovery_value" ] \
    && missing_runtime_recovery_count=$((missing_runtime_recovery_count + 1))
done
if [ "$missing_runtime_recovery_count" -ne 0 ] \
    && [ "$missing_runtime_recovery_count" -ne 2 ]; then
  echo "ERROR: validator missing-runtime recovery authority is incomplete" >&2
  exit 2
fi
if [ "$missing_runtime_recovery_count" -eq 2 ]; then
  validate_validator_active_release_handoff_path \
    VALIDATOR_MISSING_RUNTIME_RECOVERY_REQUIREMENTS \
    "$VALIDATOR_MISSING_RUNTIME_RECOVERY_REQUIREMENTS" || exit 2
  validate_validator_active_release_handoff_path \
    VALIDATOR_MISSING_RUNTIME_RECOVERY_LINEAGE \
    "$VALIDATOR_MISSING_RUNTIME_RECOVERY_LINEAGE" || exit 2
  if [ "$VALIDATOR_MISSING_RUNTIME_RECOVERY_REQUIREMENTS" = "$VALIDATOR_MISSING_RUNTIME_RECOVERY_LINEAGE" ]; then
    echo "ERROR: validator missing-runtime recovery authority paths must differ" >&2
    exit 2
  fi
fi

verify_pinned_gateway_release() {
  local max_attempts="${1:-12}"
  echo "Verifying the pinned gateway release is active on ${VALIDATOR_DEPLOY_SHA}"
  VALIDATOR_PINNED_GATEWAY_MAX_ATTEMPTS="$max_attempts" \
    bash "$VALIDATOR_PINNED_GATEWAY_VERIFIER" \
    "$VALIDATOR_V2_GATEWAY_URL" \
    "$VALIDATOR_DEPLOY_SHA"
}

verify_forward_gateway_release_before_shutdown() {
  local attempts_remaining batch_attempts max_attempts
  max_attempts="${1:-12}"

  if [ -n "$REQUESTED_VALIDATOR_DEPLOY_COMMIT" ] \
      || [ -n "$REQUESTED_COORDINATED_EXPECTED_COMMIT" ]; then
    verify_pinned_gateway_release "$max_attempts"
    return
  fi

  attempts_remaining="$max_attempts"
  while [ "$attempts_remaining" -gt 0 ]; do
    batch_attempts=4
    if [ "$attempts_remaining" -lt "$batch_attempts" ]; then
      batch_attempts="$attempts_remaining"
    fi
    if verify_pinned_gateway_release "$batch_attempts"; then
      return 0
    fi
    attempts_remaining=$((attempts_remaining - batch_attempts))
    if [ "$attempts_remaining" -le 0 ]; then
      break
    fi

    # The validator is still running here. A forward release may have moved
    # while this invocation waited for its gateway, so follow it before any
    # destructive action instead of waiting forever on an obsolete SHA.
    if ! follow_superseding_validator_release; then
      echo "Forward validator release authority remains temporarily unavailable; retaining the running validator" >&2
    fi
  done

  return 1
}

stop_pinned_validator_after_alignment_failure() {
  echo "Stopping pinned validator after persistent gateway release mismatch" >&2
  sudo pkill -TERM -f ".auto_update_wrapper.sh" 2>/dev/null || true
  sudo pkill -TERM -f "neurons/validator.py" 2>/dev/null || true
  sudo pkill -TERM -f "docker logs -f leadpoet-validator-main" 2>/dev/null || true
  sudo pkill -TERM -f "validator_tee.host.chain_relay_v2" 2>/dev/null || true
  docker ps -aq \
    --filter "name=leadpoet-validator" \
    --filter "name=leadpoet-qual-worker" \
    --filter "name=leadpoet-ff-worker" \
    | xargs -r docker stop >/dev/null 2>&1 || true
  sleep 2
  sudo pkill -KILL -f ".auto_update_wrapper.sh" 2>/dev/null || true
  sudo pkill -KILL -f "neurons/validator.py" 2>/dev/null || true
  sudo pkill -KILL -f "docker logs -f leadpoet-validator-main" 2>/dev/null || true
  sudo pkill -KILL -f "validator_tee.host.chain_relay_v2" 2>/dev/null || true
  sudo nitro-cli terminate-enclave --all >/dev/null 2>&1 || true
}

capture_validator_restart_controller() {
  local source_script
  if [ -n "$VALIDATOR_RESTART_CONTROLLER_SOURCE_DIR" ]; then
    return 0
  fi
  source_script="$(readlink -f "${BASH_SOURCE[0]}")"
  VALIDATOR_RESTART_CONTROLLER_SOURCE_DIR="$(
    mktemp -d /tmp/validator-restart-controller.XXXXXX
  )"
  mkdir -p \
    "$VALIDATOR_RESTART_CONTROLLER_SOURCE_DIR/Leadpoet/utils" \
    "$VALIDATOR_RESTART_CONTROLLER_SOURCE_DIR/validator_tee/scripts"
  install -m 700 "$source_script" \
    "$VALIDATOR_RESTART_CONTROLLER_SOURCE_DIR/validator_restart.sh"
  install -m 600 "$VALIDATOR_EXACT_COMMIT_HELPER_SOURCE" \
    "$VALIDATOR_RESTART_CONTROLLER_SOURCE_DIR/Leadpoet/utils/exact_commit_restart_v2.py"
  install -m 700 "$VALIDATOR_PINNED_GATEWAY_VERIFIER_SOURCE" \
    "$VALIDATOR_RESTART_CONTROLLER_SOURCE_DIR/validator_tee/scripts/verify_pinned_gateway_release_v2.sh"
}

install_validator_restart_controller() {
  local controller_hash release_dir temporary_dir temporary_link
  local target_dir temporary
  if [ -z "$VALIDATOR_RESTART_CONTROLLER_SOURCE_DIR" ]; then
    echo "ERROR: validator restart controller was not captured" >&2
    return 1
  fi
  controller_hash="$(
    sha256sum "$VALIDATOR_RESTART_CONTROLLER_SOURCE_DIR/validator_restart.sh" \
      | cut -d' ' -f1
  )"
  mkdir -p "$VALIDATOR_RESTART_CONTROLLER_ROOT/releases"
  temporary_dir="$(mktemp -d "$VALIDATOR_RESTART_CONTROLLER_ROOT/.release.XXXXXX")"
  cp -a "$VALIDATOR_RESTART_CONTROLLER_SOURCE_DIR/." "$temporary_dir/"
  release_dir="$VALIDATOR_RESTART_CONTROLLER_ROOT/releases/$controller_hash"
  rm -rf -- "$release_dir"
  mv -- "$temporary_dir" "$release_dir"
  temporary_link="$VALIDATOR_RESTART_CONTROLLER_ROOT/.current.$$"
  ln -s "releases/$controller_hash" "$temporary_link"
  mv -Tf "$temporary_link" "$VALIDATOR_RESTART_CONTROLLER_CURRENT"

  target_dir="$(dirname "$VALIDATOR_HOST_RESTART_SCRIPT")"
  mkdir -p "$target_dir"
  temporary="$(mktemp "$target_dir/.validator_restart.sh.XXXXXX")"
  install -m 700 "$VALIDATOR_RESTART_CONTROLLER_CURRENT/validator_restart.sh" \
    "$temporary"
  mv -f "$temporary" "$VALIDATOR_HOST_RESTART_SCRIPT"
}

VALIDATOR_ENV_EXPORT="$(mktemp /tmp/validator_env_export.XXXXXX)"
SECRET_TMP="$(mktemp /tmp/validator_secret_env.XXXXXX)"

record_validator_restart_timing() {
  local stage="$1"
  local status="${2:-reached}"
  if ! mkdir -p "$VALIDATOR_RESTART_TIMING_DIR" \
      || ! chmod 700 "$VALIDATOR_RESTART_TIMING_DIR"; then
    echo "WARNING: validator restart timing directory is unavailable" >&2
    return 0
  fi
  if ! python3 - \
    "$VALIDATOR_RESTART_TIMING_FILE" \
    "$VALIDATOR_RESTART_STARTED_EPOCH" \
    "$stage" \
    "$status" \
    "${VALIDATOR_DEPLOY_SHA:-}" <<'PY'
import datetime
import json
import os
import sys
import time

path, started, stage, status, commit = sys.argv[1:]
now = time.time()
record = {
    "schema_version": "leadpoet.validator_restart_timing.v1",
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
    "VALIDATOR_RESTART_TIMING "
    f"stage={stage} status={status} elapsed_seconds={record['elapsed_seconds']}"
)
PY
  then
    echo "WARNING: validator restart timing event could not be recorded: $stage" >&2
  fi
  return 0
}

emit_validator_restart_sentry_summary() {
  local status="$1" candidate_sha summary_status telemetry_python shutdown_flag=()
  command -v timeout >/dev/null 2>&1 || return 0
  telemetry_python="${VALIDATOR_TELEMETRY_PYTHON_BIN:-$VALIDATOR_PYTHON_BIN}"
  command -v "$telemetry_python" >/dev/null 2>&1 || return 0
  [ -r "$VALIDATOR_ROOT/leadpoet_observability/sentry_cli.py" ] || return 0
  candidate_sha="${VALIDATOR_DEPLOY_SHA:-}"
  summary_status="failed"
  [ "$status" -eq 0 ] && summary_status="passed"
  if [ "$VALIDATOR_DESTRUCTIVE_PHASE_STARTED" = "1" ]; then
    shutdown_flag=(--shutdown-started)
  fi
  PYTHONPATH="$VALIDATOR_ROOT" timeout 2 \
    "$telemetry_python" -m leadpoet_observability.sentry_cli \
    restart-summary \
    --component validator \
    --status "$summary_status" \
    --stage "${VALIDATOR_DEPLOY_STAGE:-unknown}" \
    --ledger "$VALIDATOR_RESTART_TIMING_FILE" \
    --restart-invocation-id "$VALIDATOR_RESTART_INVOCATION_ID" \
    --release-attempts "$VALIDATOR_RELEASE_ATTEMPTS_USED" \
    --candidate-sha "$candidate_sha" \
    "${shutdown_flag[@]}" >/dev/null 2>&1 || true
  return 0
}

prepare_validator_sentry_host_runtime() {
  local prepared_python
  VALIDATOR_TELEMETRY_PYTHON_BIN="$VALIDATOR_PYTHON_BIN"
  case "${LEADPOET_SENTRY_ENABLED:-}" in
    1|true|TRUE|yes|YES|on|ON) ;;
    *) return 0 ;;
  esac
  [ -n "${LEADPOET_SENTRY_DSN:-}" ] || return 0
  command -v timeout >/dev/null 2>&1 || return 0
  if ! prepared_python="$(
      PYTHONPATH="$VALIDATOR_ROOT" timeout 35 \
        "$VALIDATOR_PYTHON_BIN" \
        -m leadpoet_observability.host_runtime \
        --base-python "$(command -v "$VALIDATOR_PYTHON_BIN")" \
        --repo-root "$VALIDATOR_ROOT" \
        --requirements "$VALIDATOR_ROOT/requirements.txt" \
        --lock "$VALIDATOR_ROOT/leadpoet_observability/requirements-host.lock" \
        --cache-root "$VALIDATOR_TELEMETRY_CACHE_ROOT" \
        --timeout-seconds 30
    )"; then
    echo "WARNING: validator host Sentry runtime is unavailable; restart remains fail-open" >&2
    return 0
  fi
  if [ -x "$prepared_python" ]; then
    VALIDATOR_TELEMETRY_PYTHON_BIN="$prepared_python"
    echo "Validator host Sentry runtime ready"
  fi
  return 0
}

if [ "$VALIDATOR_RESTART_TIMING_INITIALIZED" = "1" ]; then
  record_validator_restart_timing "controller_reexec"
else
  record_validator_restart_timing "invoked"
  VALIDATOR_RESTART_TIMING_INITIALIZED=1
  export VALIDATOR_RESTART_TIMING_INITIALIZED
fi

cleanup_validator_restart_preparation() {
  if [ -n "${VALIDATOR_ENV_EXPORT:-}" ]; then
    rm -f "$VALIDATOR_ENV_EXPORT"
    VALIDATOR_ENV_EXPORT=""
  fi
  if [ -n "${SECRET_TMP:-}" ]; then
    rm -f "$SECRET_TMP"
    SECRET_TMP=""
  fi
  if [ -n "$VALIDATOR_RESTART_CONTROLLER_SOURCE_DIR" ]; then
    rm -rf "$VALIDATOR_RESTART_CONTROLLER_SOURCE_DIR"
    VALIDATOR_RESTART_CONTROLLER_SOURCE_DIR=""
  fi
  if [ -n "$VALIDATOR_PINNED_GATEWAY_VERIFIER" ]; then
    rm -f "$VALIDATOR_PINNED_GATEWAY_VERIFIER"
    VALIDATOR_PINNED_GATEWAY_VERIFIER=""
  fi
  for handoff_path in \
    "$VALIDATOR_ACTIVE_RELEASE_REQUIREMENTS_OUTPUT" \
    "$VALIDATOR_FINAL_RELEASE_REQUIREMENTS_INPUT" \
    "$VALIDATOR_FINAL_RELEASE_LINEAGE_INPUT" \
    "$VALIDATOR_MISSING_RUNTIME_RECOVERY_REQUIREMENTS" \
    "$VALIDATOR_MISSING_RUNTIME_RECOVERY_LINEAGE"; do
    if [[ "$handoff_path" =~ ^/tmp/leadpoet-[A-Za-z0-9._-]+\.json$ ]]; then
      rm -f -- "$handoff_path" || true
    fi
  done
}

cleanup() {
  local status="$?"
  set +e
  if [ "$status" -ne 0 ]; then
    record_validator_restart_timing "$VALIDATOR_DEPLOY_STAGE" "failed" \
      >/dev/null 2>&1 || true
  fi
  emit_validator_restart_sentry_summary "$status"
  if [ "$VALIDATOR_DESTRUCTIVE_PHASE_STARTED" = "1" ] \
      && [ "$VALIDATOR_RESTART_COMPLETED" != "1" ]; then
    echo "Cleaning incomplete validator activation" >&2
    stop_pinned_validator_after_alignment_failure
    if [ "$VALIDATOR_DOCKER_LOCK_ACQUIRED" = "1" ] \
        && declare -F leadpoet_release_docker_operation_lock_v2 >/dev/null 2>&1; then
      leadpoet_release_docker_operation_lock_v2 || true
    fi
    VALIDATOR_DOCKER_LOCK_ACQUIRED=0
  fi
  cleanup_validator_restart_preparation
  return "$status"
}
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

git_fetch_failure_is_transient() {
  LC_ALL=C grep -Eiq \
    "HTTP (429|500|502|503|504)|curl (18|28|35|52|55|56|92)|connection reset|could not resolve host|early EOF|expected 'acknowledgments'|failed to connect|remote end hung up unexpectedly|temporary failure in name resolution|timed out"
}

fetch_validator_origin_with_retry() {
  local attempt output
  for attempt in 1 2 3 4; do
    if output="$(git fetch origin 2>&1)"; then
      [ -z "$output" ] || printf '%s\n' "$output"
      return 0
    fi
    printf '%s\n' "$output" >&2
    if ! printf '%s\n' "$output" | git_fetch_failure_is_transient; then
      return 1
    fi
    if [ "$attempt" -eq 4 ]; then
      return 1
    fi
    echo "Transient Git fetch failure; retrying before validator shutdown ($attempt/4)" >&2
    sleep "$((1 << (attempt - 1)))"
  done
  return 1
}

cd "$VALIDATOR_ROOT"

echo "Preflight: preserving tracked local validator checkout changes if present"
if ! git diff --quiet || ! git diff --cached --quiet; then
  restart_stash_message="pre-validator-restart-local-tracked-$(date -u +%Y%m%dT%H%M%SZ)"
  git stash push -m "$restart_stash_message" -- .
  echo "Preserved tracked local changes in Git stash: $restart_stash_message"
fi

echo "Pulling latest GitHub main before stopping validator"
before_head="$(git rev-parse HEAD)"
fetch_validator_origin_with_retry
if [ -n "$REQUESTED_VALIDATOR_DEPLOY_COMMIT" ]; then
  if ! [[ "$REQUESTED_VALIDATOR_DEPLOY_COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
    echo "ERROR: VALIDATOR_DEPLOY_COMMIT must be a full 40-character SHA" >&2
    exit 1
  fi
  if ! git merge-base --is-ancestor "$REQUESTED_VALIDATOR_DEPLOY_COMMIT" origin/main; then
    echo "ERROR: VALIDATOR_DEPLOY_COMMIT is not reachable from origin/main" >&2
    exit 1
  fi
  echo "Validating exact-commit V2 rollback compatibility"
  python3 "$VALIDATOR_EXACT_COMMIT_HELPER_SOURCE" \
    --repo-root "$VALIDATOR_ROOT" \
    --selected-commit "$REQUESTED_VALIDATOR_DEPLOY_COMMIT" \
    --branch-ref origin/main
  if [ ! -r "$VALIDATOR_PINNED_GATEWAY_VERIFIER_SOURCE" ]; then
    echo "ERROR: pinned gateway release verifier is unavailable" >&2
    exit 1
  fi
  VALIDATOR_PINNED_GATEWAY_VERIFIER="$(
    mktemp /tmp/verify_pinned_gateway_release_v2.XXXXXX
  )"
  cp "$VALIDATOR_PINNED_GATEWAY_VERIFIER_SOURCE" \
    "$VALIDATOR_PINNED_GATEWAY_VERIFIER"
  chmod 700 "$VALIDATOR_PINNED_GATEWAY_VERIFIER"
  capture_validator_restart_controller
  install_validator_restart_controller
  git checkout --detach "$REQUESTED_VALIDATOR_DEPLOY_COMMIT"
  echo "Selected operator-requested validator commit: $REQUESTED_VALIDATOR_DEPLOY_COMMIT"
else
  git checkout main
  git pull --ff-only origin main
fi
after_head="$(git rev-parse HEAD)"
if [ -n "$REQUESTED_COORDINATED_EXPECTED_COMMIT" ] \
    && [ "$after_head" != "$REQUESTED_COORDINATED_EXPECTED_COMMIT" ]; then
  echo "ERROR: coordinated validator candidate moved before restart preparation" >&2
  echo "       expected=$REQUESTED_COORDINATED_EXPECTED_COMMIT observed=$after_head" >&2
  exit 1
fi
VALIDATOR_DEPLOY_TOPOLOGY_ENTRY="$(
  git -C "$VALIDATOR_ROOT" ls-tree \
    "$after_head" -- gateway/tee/topology.json
)" || {
  echo "ERROR: validator target topology Git identity is unavailable" >&2
  exit 1
}
if ! [[ "$VALIDATOR_DEPLOY_TOPOLOGY_ENTRY" =~ ^100644\ blob\ [0-9a-f]{40}$'\t'gateway/tee/topology.json$ ]]; then
  echo "ERROR: validator target topology is not one regular Git blob" >&2
  exit 1
fi
VALIDATOR_DEPLOY_TOPOLOGY_BLOB="${VALIDATOR_DEPLOY_TOPOLOGY_ENTRY#100644 blob }"
VALIDATOR_DEPLOY_TOPOLOGY_BLOB="${VALIDATOR_DEPLOY_TOPOLOGY_BLOB%%$'\t'*}"
if [ "$VALIDATOR_DEPLOY_TOPOLOGY_BLOB" = "$HISTORICAL_THREE_ROLE_TOPOLOGY_BLOB" ] \
    && ! git -C "$VALIDATOR_ROOT" cat-file -e \
      "$after_head:gateway/tee/build_local_release_v2.sh" 2>/dev/null \
    && ! git -C "$VALIDATOR_ROOT" cat-file -e \
      "$after_head:gateway/tee/local_release_v2.py" 2>/dev/null; then
  VALIDATOR_HISTORICAL_TOPOLOGY_HASH="$HISTORICAL_THREE_ROLE_TOPOLOGY_HASH"
fi
current_restart_script="$(readlink -f "${BASH_SOURCE[0]}")"
candidate_restart_script="$(readlink -f "$VALIDATOR_ROOT/validator_restart.sh")"
restart_script_differs=0
if ! cmp -s "$current_restart_script" "$candidate_restart_script"; then
  restart_script_differs=1
fi
if [ -z "$REQUESTED_VALIDATOR_DEPLOY_COMMIT" ] \
   && { [ "$before_head" != "$after_head" ] \
        || [ "$restart_script_differs" = "1" ]; } \
   && [ "${VALIDATOR_RESTART_REEXECED:-0}" != "1" ]; then
  echo "Restart wrapper updated from GitHub; re-executing latest validator_restart.sh"
  reset_standalone_active_release_handoff_for_reexec
  exec env \
    VALIDATOR_RESTART_REEXECED=1 \
    VALIDATOR_COORDINATED_EXPECTED_COMMIT="$REQUESTED_COORDINATED_EXPECTED_COMMIT" \
    VALIDATOR_RESTART_STARTED_EPOCH="$VALIDATOR_RESTART_STARTED_EPOCH" \
    VALIDATOR_RESTART_INVOCATION_ID="${VALIDATOR_RESTART_INVOCATION_ID:-validator-${VALIDATOR_RESTART_STARTED_EPOCH:-unknown}-$$}" \
    VALIDATOR_ACTIVE_RELEASE_AUTHORITY_ROOT="$VALIDATOR_ACTIVE_RELEASE_AUTHORITY_ROOT" \
    VALIDATOR_ACTIVE_RELEASE_AUTHORITY_COMMIT="$VALIDATOR_ACTIVE_RELEASE_AUTHORITY_COMMIT" \
    VALIDATOR_ACTIVE_RELEASE_RESTART_INVOCATION_ID="$VALIDATOR_ACTIVE_RELEASE_RESTART_INVOCATION_ID" \
    VALIDATOR_PAIRED_ACTIVE_RELEASE_REQUIRED="$VALIDATOR_PAIRED_ACTIVE_RELEASE_REQUIRED" \
    VALIDATOR_RELEASE_ATTEMPTS_USED="${VALIDATOR_RELEASE_ATTEMPTS_USED:-0}" \
    VALIDATOR_RESTART_TIMING_DIR="$VALIDATOR_RESTART_TIMING_DIR" \
    VALIDATOR_RESTART_TIMING_FILE="$VALIDATOR_RESTART_TIMING_FILE" \
    VALIDATOR_RESTART_TIMING_INITIALIZED="$VALIDATOR_RESTART_TIMING_INITIALIZED" \
    VALIDATOR_RELEASE_SUPERSESSION_COUNT="$VALIDATOR_RELEASE_SUPERSESSION_COUNT" \
    VALIDATOR_RELEASE_SUPERSESSION_MAX="$VALIDATOR_RELEASE_SUPERSESSION_MAX" \
    VALIDATOR_ACTIVE_RELEASE_REQUIREMENTS_OUTPUT="$VALIDATOR_ACTIVE_RELEASE_REQUIREMENTS_OUTPUT" \
    VALIDATOR_FINAL_RELEASE_REQUIREMENTS_INPUT="$VALIDATOR_FINAL_RELEASE_REQUIREMENTS_INPUT" \
    VALIDATOR_FINAL_RELEASE_LINEAGE_INPUT="$VALIDATOR_FINAL_RELEASE_LINEAGE_INPUT" \
    VALIDATOR_MISSING_RUNTIME_RECOVERY_REQUIREMENTS="$VALIDATOR_MISSING_RUNTIME_RECOVERY_REQUIREMENTS" \
    VALIDATOR_MISSING_RUNTIME_RECOVERY_LINEAGE="$VALIDATOR_MISSING_RUNTIME_RECOVERY_LINEAGE" \
    VALIDATOR_V2_GATEWAY_RELEASE_REQUIREMENTS="$VALIDATOR_V2_GATEWAY_RELEASE_REQUIREMENTS" \
    bash "$VALIDATOR_ROOT/validator_restart.sh" "$@"
fi
if [ -z "$REQUESTED_VALIDATOR_DEPLOY_COMMIT" ] \
    && [ "$restart_script_differs" = "1" ]; then
  echo "ERROR: re-executed validator restart controller differs from the selected candidate" >&2
  exit 1
fi
if [ -z "$REQUESTED_VALIDATOR_DEPLOY_COMMIT" ]; then
  VALIDATOR_EXACT_COMMIT_HELPER_SOURCE="$VALIDATOR_ROOT/Leadpoet/utils/exact_commit_restart_v2.py"
  VALIDATOR_PINNED_GATEWAY_VERIFIER_SOURCE="$VALIDATOR_ROOT/validator_tee/scripts/verify_pinned_gateway_release_v2.sh"
  if [ ! -r "$VALIDATOR_PINNED_GATEWAY_VERIFIER_SOURCE" ]; then
    echo "ERROR: pinned gateway release verifier is unavailable" >&2
    exit 1
  fi
  VALIDATOR_PINNED_GATEWAY_VERIFIER="$(
    mktemp /tmp/verify_pinned_gateway_release_v2.XXXXXX
  )"
  cp "$VALIDATOR_PINNED_GATEWAY_VERIFIER_SOURCE" \
    "$VALIDATOR_PINNED_GATEWAY_VERIFIER"
  chmod 700 "$VALIDATOR_PINNED_GATEWAY_VERIFIER"
fi
capture_validator_restart_controller

if [ -z "$VALIDATOR_ACTIVE_RELEASE_AUTHORITY_COMMIT" ]; then
  VALIDATOR_ACTIVE_RELEASE_AUTHORITY_COMMIT="$(
    git -C "$VALIDATOR_ROOT" rev-parse --verify 'origin/main^{commit}'
  )"
fi
if ! [[ "$VALIDATOR_ACTIVE_RELEASE_AUTHORITY_COMMIT" =~ ^[0-9a-f]{40}$ ]] \
    || ! git -C "$VALIDATOR_ROOT" merge-base --is-ancestor \
      "$(git -C "$VALIDATOR_ROOT" rev-parse HEAD)" \
      "$VALIDATOR_ACTIVE_RELEASE_AUTHORITY_COMMIT"; then
  echo "ERROR: validator active release authority is incompatible with the selected runtime" >&2
  exit 1
fi
VALIDATOR_ACTIVE_RELEASE_CONTROLLER_ROOT="${VALIDATOR_ACTIVE_RELEASE_AUTHORITY_ROOT:-$VALIDATOR_ROOT}"
VALIDATOR_ACTIVE_RELEASE_PREPARER="$VALIDATOR_ACTIVE_RELEASE_CONTROLLER_ROOT/gateway/tee/prepare_active_release_lineage_v2.py"
if [ ! -r "$VALIDATOR_ACTIVE_RELEASE_PREPARER" ]; then
  echo "ERROR: selected validator release lacks bounded active release lineage support" >&2
  echo "Validator remains running; production shutdown has not started." >&2
  exit 1
fi
VALIDATOR_LINEAGE_HISTORICAL_TOPOLOGY_HASH=""
if [ -n "$VALIDATOR_ACTIVE_RELEASE_AUTHORITY_ROOT" ]; then
  VALIDATOR_LINEAGE_HISTORICAL_TOPOLOGY_HASH="$VALIDATOR_HISTORICAL_TOPOLOGY_HASH"
fi
if [ "$active_release_handoff_count" -eq 0 ]; then
  if [ ! -s "$VALIDATOR_FINAL_RELEASE_REQUIREMENTS_INPUT" ] \
      || [ ! -s "$VALIDATOR_FINAL_RELEASE_LINEAGE_INPUT" ] \
      || [ -L "$VALIDATOR_FINAL_RELEASE_REQUIREMENTS_INPUT" ] \
      || [ -L "$VALIDATOR_FINAL_RELEASE_LINEAGE_INPUT" ]; then
    echo "ERROR: standalone validator compact-lineage fallback is unavailable" >&2
    echo "Validator remains running; production shutdown has not started." >&2
    exit 1
  fi
  VALIDATOR_ACTIVE_RELEASE_RESTART_INVOCATION_ID="$(
    PYTHONDONTWRITEBYTECODE=1 \
      PYTHONPATH="$VALIDATOR_ACTIVE_RELEASE_CONTROLLER_ROOT" \
      "$VALIDATOR_PYTHON_BIN" - \
        "$(git -C "$VALIDATOR_ROOT" rev-parse HEAD)" \
        "$VALIDATOR_ACTIVE_RELEASE_AUTHORITY_COMMIT" \
        "$VALIDATOR_LINEAGE_HISTORICAL_TOPOLOGY_HASH" \
        "$VALIDATOR_FINAL_RELEASE_REQUIREMENTS_INPUT" \
        "$VALIDATOR_FINAL_RELEASE_LINEAGE_INPUT" <<'PY'
import json
import os
import stat
import sys

from gateway.tee.active_release_requirements_v2 import (
    validate_active_release_requirements_v2,
)
from gateway.tee.release_lineage_v2 import validate_compact_release_lineage_v2

expected_commit, expected_authority = sys.argv[1:3]
historical_topology_hash = sys.argv[3] or None
max_document_bytes = 4 * 1024 * 1024


def load_bounded_json(path: str, label: str):
    try:
        fd = os.open(
            path,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
        )
    except OSError as exc:
        raise SystemExit(f"{label} cannot be opened securely: {exc}") from exc
    try:
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode):
            raise SystemExit(f"{label} is not a regular file")
        if metadata.st_size <= 0 or metadata.st_size > max_document_bytes:
            raise SystemExit(f"{label} exceeds the bounded document size")
        payload = os.read(fd, max_document_bytes + 1)
        if len(payload) != metadata.st_size or len(payload) > max_document_bytes:
            raise SystemExit(f"{label} changed during its bounded read")
    finally:
        os.close(fd)
    try:
        return json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"{label} is not valid UTF-8 JSON") from exc


requirements = validate_active_release_requirements_v2(
    load_bounded_json(sys.argv[4], "standalone validator requirements")
)
if historical_topology_hash is None:
    lineage = validate_compact_release_lineage_v2(
        load_bounded_json(sys.argv[5], "standalone validator lineage"),
        expected_current_commit=expected_commit,
    )
else:
    from gateway.tee.release_lineage_v2 import (
        validate_historical_compact_release_lineage_v2,
    )

    lineage = validate_historical_compact_release_lineage_v2(
        load_bounded_json(sys.argv[5], "standalone validator lineage"),
        expected_topology_hash=historical_topology_hash,
        expected_current_commit=expected_commit,
    )
if requirements["candidate_commit_sha"] != expected_commit:
    raise SystemExit("standalone validator active release candidate differs")
if requirements["authority_commit_sha"] != expected_authority:
    raise SystemExit("standalone validator active release controller differs")
if set(requirements["required_commits"]) != set(lineage["releases"]):
    raise SystemExit("standalone validator active release requirements differ")
print(requirements["restart_invocation_id"])
PY
  )"
fi

follow_superseding_validator_release() {
  local helper latest_sha next_count

  if [ -n "$REQUESTED_VALIDATOR_DEPLOY_COMMIT" ] \
      || [ -n "$REQUESTED_COORDINATED_EXPECTED_COMMIT" ]; then
    return 0
  fi
  helper="$VALIDATOR_ROOT/Leadpoet/utils/restart_release_supersession_v2.py"
  if [ ! -r "$helper" ]; then
    echo "ERROR: forward validator restart authority helper is unavailable" >&2
    return 1
  fi
  if ! latest_sha="$(
      "$VALIDATOR_PYTHON_BIN" "$helper" \
        --repo-root "$VALIDATOR_ROOT" \
        --expected-commit "$VALIDATOR_DEPLOY_SHA" \
        --branch main
    )"; then
    echo "Forward validator restart authority is temporarily unreadable; retaining the running validator" >&2
    return 1
  fi
  if [ "$latest_sha" = "$VALIDATOR_DEPLOY_SHA" ]; then
    return 0
  fi

  next_count=$((VALIDATOR_RELEASE_SUPERSESSION_COUNT + 1))
  if [ "$next_count" -gt "$VALIDATOR_RELEASE_SUPERSESSION_MAX" ]; then
    echo "ERROR: validator release changed too many times during one restart invocation" >&2
    return 1
  fi
  echo "Forward validator release moved from $VALIDATOR_DEPLOY_SHA to $latest_sha; re-executing before shutdown"
  record_validator_restart_timing "release_superseded"
  cleanup_validator_restart_preparation
  git checkout main
  git merge --ff-only "$latest_sha"
  if [ "$(git rev-parse HEAD)" != "$latest_sha" ]; then
    echo "ERROR: superseding validator checkout does not match the fetched authority" >&2
    return 1
  fi

  reset_standalone_active_release_handoff_for_reexec
  exec env \
    VALIDATOR_ROOT="$VALIDATOR_ROOT" \
    VALIDATOR_ENV_FILE="$VALIDATOR_ENV_FILE" \
    LEADPOET_VALIDATOR_ENV_SECRET_ID="$LEADPOET_VALIDATOR_ENV_SECRET_ID" \
    VALIDATOR_ENV_BACKUP_DIR="$VALIDATOR_ENV_BACKUP_DIR" \
    VALIDATOR_PYTHON_BIN="$VALIDATOR_PYTHON_BIN" \
    VALIDATOR_RESTART_CONTROLLER_ROOT="$VALIDATOR_RESTART_CONTROLLER_ROOT" \
    VALIDATOR_HOST_RESTART_SCRIPT="$VALIDATOR_HOST_RESTART_SCRIPT" \
    VALIDATOR_RESTART_STARTED_EPOCH="$VALIDATOR_RESTART_STARTED_EPOCH" \
    VALIDATOR_RESTART_INVOCATION_ID="${VALIDATOR_RESTART_INVOCATION_ID:-validator-${VALIDATOR_RESTART_STARTED_EPOCH:-unknown}-$$}" \
    VALIDATOR_ACTIVE_RELEASE_AUTHORITY_ROOT="$VALIDATOR_ACTIVE_RELEASE_AUTHORITY_ROOT" \
    VALIDATOR_ACTIVE_RELEASE_AUTHORITY_COMMIT="$VALIDATOR_ACTIVE_RELEASE_AUTHORITY_COMMIT" \
    VALIDATOR_ACTIVE_RELEASE_RESTART_INVOCATION_ID="$VALIDATOR_ACTIVE_RELEASE_RESTART_INVOCATION_ID" \
    VALIDATOR_PAIRED_ACTIVE_RELEASE_REQUIRED="$VALIDATOR_PAIRED_ACTIVE_RELEASE_REQUIRED" \
    VALIDATOR_RELEASE_ATTEMPTS_USED="${VALIDATOR_RELEASE_ATTEMPTS_USED:-0}" \
    VALIDATOR_RESTART_TIMING_DIR="$VALIDATOR_RESTART_TIMING_DIR" \
    VALIDATOR_RESTART_TIMING_FILE="$VALIDATOR_RESTART_TIMING_FILE" \
    VALIDATOR_RESTART_TIMING_INITIALIZED="$VALIDATOR_RESTART_TIMING_INITIALIZED" \
    VALIDATOR_RELEASE_SUPERSESSION_COUNT="$next_count" \
    VALIDATOR_RELEASE_SUPERSESSION_MAX="$VALIDATOR_RELEASE_SUPERSESSION_MAX" \
    VALIDATOR_V2_RELEASE_BUCKET="$VALIDATOR_V2_RELEASE_BUCKET" \
    VALIDATOR_V2_RELEASE_PREFIX="$VALIDATOR_V2_RELEASE_PREFIX" \
    VALIDATOR_V2_RELEASE_ARCHIVE_ROOT="$VALIDATOR_V2_RELEASE_ARCHIVE_ROOT" \
    VALIDATOR_V2_OFFLINE_ARTIFACT_ROOT="$VALIDATOR_V2_OFFLINE_ARTIFACT_ROOT" \
    VALIDATOR_ACTIVE_RELEASE_REQUIREMENTS_OUTPUT="$VALIDATOR_ACTIVE_RELEASE_REQUIREMENTS_OUTPUT" \
    VALIDATOR_FINAL_RELEASE_REQUIREMENTS_INPUT="$VALIDATOR_FINAL_RELEASE_REQUIREMENTS_INPUT" \
    VALIDATOR_FINAL_RELEASE_LINEAGE_INPUT="$VALIDATOR_FINAL_RELEASE_LINEAGE_INPUT" \
    VALIDATOR_MISSING_RUNTIME_RECOVERY_REQUIREMENTS="$VALIDATOR_MISSING_RUNTIME_RECOVERY_REQUIREMENTS" \
    VALIDATOR_MISSING_RUNTIME_RECOVERY_LINEAGE="$VALIDATOR_MISSING_RUNTIME_RECOVERY_LINEAGE" \
    VALIDATOR_V2_GATEWAY_RELEASE_REQUIREMENTS="$VALIDATOR_V2_GATEWAY_RELEASE_REQUIREMENTS" \
    VALIDATOR_STATEFUL_CUTOVER_PREPARE_ONLY="$REQUESTED_STATEFUL_CUTOVER_PREPARE_ONLY" \
    LEADPOET_USE_CAPTURED_RESTART_START=1 \
    VALIDATOR_RESTART_REEXECED=0 \
    bash "$VALIDATOR_ROOT/validator_restart.sh"
}

echo "Preparing validator runtime env from Secrets Manager"
mkdir -p "$(dirname "$VALIDATOR_ENV_FILE")" "$VALIDATOR_ENV_BACKUP_DIR"
chmod 700 "$(dirname "$VALIDATOR_ENV_FILE")" "$VALIDATOR_ENV_BACKUP_DIR"

export AWS_REGION="${AWS_REGION:-us-east-1}"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"

if [ -f "$VALIDATOR_ENV_FILE" ]; then
  cp -p "$VALIDATOR_ENV_FILE" \
    "$VALIDATOR_ENV_BACKUP_DIR/validator.env.before-validator-restart.$(date -u +%Y%m%dT%H%M%SZ).bak"
fi

aws secretsmanager get-secret-value \
  --secret-id "$LEADPOET_VALIDATOR_ENV_SECRET_ID" \
  --query SecretString \
  --output text > "$SECRET_TMP"

python3 - "$SECRET_TMP" "$VALIDATOR_ENV_FILE" "$VALIDATOR_ENV_EXPORT" <<'PY'
import json
import re
import shlex
import sys
from pathlib import Path

src = Path(sys.argv[1])
cache = Path(sys.argv[2])
export_file = Path(sys.argv[3])
raw = src.read_text()
cache_excluded_keys = {
    "LEADPOET_RESTART_INVOCATION_ID",
    "LEADPOET_SENTRY_API_TOKEN",
    "VALIDATOR_ACTIVE_RELEASE_REQUIREMENTS_OUTPUT",
    "VALIDATOR_ACTIVE_RELEASE_AUTHORITY_ROOT",
    "VALIDATOR_ACTIVE_RELEASE_AUTHORITY_COMMIT",
    "VALIDATOR_ACTIVE_RELEASE_RESTART_INVOCATION_ID",
    "VALIDATOR_PAIRED_ACTIVE_RELEASE_REQUIRED",
    "VALIDATOR_FINAL_RELEASE_LINEAGE_INPUT",
    "VALIDATOR_FINAL_RELEASE_REQUIREMENTS_INPUT",
    "VALIDATOR_MISSING_RUNTIME_RECOVERY_LINEAGE",
    "VALIDATOR_MISSING_RUNTIME_RECOVERY_REQUIREMENTS",
    "VALIDATOR_RESTART_INVOCATION_ID",
    "VALIDATOR_V2_GATEWAY_RELEASE_REQUIREMENTS",
}

try:
    parsed = json.loads(raw)
except Exception:
    parsed = None

if isinstance(parsed, dict):
    lines = []
    for key, value in parsed.items():
        if key in cache_excluded_keys:
            continue
        if isinstance(value, (dict, list)):
            value = json.dumps(value, separators=(",", ":"))
        elif value is None:
            value = ""
        lines.append(f"{key}={value}")
    raw = "\n".join(lines) + "\n"
else:
    lines = []
    for source_line in raw.replace("\x00", "\n").splitlines():
        line = source_line.strip()
        candidate = line[len("export "):].strip() if line.startswith("export ") else line
        try:
            parts = shlex.split(candidate, posix=True)
        except ValueError:
            parts = [candidate]
        assignment = parts[0] if len(parts) == 1 else candidate
        key = assignment.split("=", 1)[0].strip() if "=" in assignment else ""
        if key in cache_excluded_keys:
            continue
        lines.append(source_line)
    raw = "\n".join(lines)
    if lines:
        raw += "\n"

cache.parent.mkdir(parents=True, exist_ok=True)
cache.write_text(raw)

skip_keys = {
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_SECURITY_TOKEN",
    "AWS_PROFILE",
    "VALIDATOR_COORDINATED_EXPECTED_COMMIT",
    "VALIDATOR_DEPLOY_COMMIT",
    "VALIDATOR_ACTIVE_RELEASE_REQUIREMENTS_OUTPUT",
    "VALIDATOR_ACTIVE_RELEASE_AUTHORITY_ROOT",
    "VALIDATOR_ACTIVE_RELEASE_AUTHORITY_COMMIT",
    "VALIDATOR_ACTIVE_RELEASE_RESTART_INVOCATION_ID",
    "VALIDATOR_PAIRED_ACTIVE_RELEASE_REQUIRED",
    "VALIDATOR_EXACT_RELEASE_PINNED",
    "VALIDATOR_FINAL_RELEASE_LINEAGE_INPUT",
    "VALIDATOR_FINAL_RELEASE_REQUIREMENTS_INPUT",
    "VALIDATOR_MISSING_RUNTIME_RECOVERY_LINEAGE",
    "VALIDATOR_MISSING_RUNTIME_RECOVERY_REQUIREMENTS",
    "VALIDATOR_PINNED_GATEWAY_COORDINATION_FILE",
    "VALIDATOR_PINNED_GATEWAY_COORDINATION_MAX_ATTEMPTS",
    "VALIDATOR_PINNED_GATEWAY_TIMEOUT_SECONDS",
    "VALIDATOR_RESTART_TEMP_CLEANUP_MIN_AGE_SECONDS",
    "VALIDATOR_RESTART_CLEANUP_MAX_CANDIDATES",
    "VALIDATOR_RESTART_INVOCATION_ID",
    "VALIDATOR_V2_GATEWAY_RELEASE_REQUIREMENTS",
    "LEADPOET_RESTART_INVOCATION_ID",
    "LEADPOET_SENTRY_API_TOKEN",
}
exports = []
for raw_line in raw.replace("\x00", "\n").splitlines():
    line = raw_line.strip()
    if not line or line.startswith("#"):
        continue
    if line.startswith("export "):
        line = line[len("export "):].strip()
    try:
        parts = shlex.split(line, posix=True)
    except ValueError:
        parts = [line]
    if len(parts) == 1 and "=" in parts[0]:
        key, value = parts[0].split("=", 1)
    elif "=" in line:
        key, value = line.split("=", 1)
    else:
        continue
    key = key.strip()
    if key in skip_keys:
        continue
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
        continue
    exports.append(f"export {key}={shlex.quote(value)}")

export_file.write_text("\n".join(exports) + "\n")
print(f"hydrated validator env cache and prepared {len(exports)} env vars")
PY
chmod 600 "$VALIDATOR_ENV_FILE"

set -a
. "$VALIDATOR_ENV_EXPORT"
set +a
export VALIDATOR_RESTART_INVOCATION_ID
export LEADPOET_RESTART_INVOCATION_ID="$VALIDATOR_RESTART_INVOCATION_ID"
prepare_validator_sentry_host_runtime

if [ ! -s "$VALIDATOR_STATEFUL_CUTOVER_MANIFEST" ]; then
  echo "ERROR: canonical stateful epoch cutover manifest is missing" >&2
  exit 1
fi
echo "Loading the canonical stateful epoch cutover manifest"
LEADPOET_SUBNET_EPOCH_CUTOVER_JSON="$(
  python3 - "$VALIDATOR_STATEFUL_CUTOVER_MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

document = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if not isinstance(document, dict):
    raise SystemExit("stateful epoch cutover manifest must be a JSON object")
print(json.dumps(document, sort_keys=True, separators=(",", ":")))
PY
)"
export LEADPOET_SUBNET_EPOCH_CUTOVER_JSON
unset LEADPOET_SUBNET_EPOCH_CUTOVER_PATH

VALIDATOR_CHAIN_SIGNING_PROFILE="$VALIDATOR_ROOT/validator_tee/enclave/chain_signing_profile_v2.json"
VALIDATOR_CHAIN_SIGNING_PROFILE="$(
  PYTHONPATH="$VALIDATOR_ROOT" "$VALIDATOR_PYTHON_BIN" - \
    "$VALIDATOR_CHAIN_SIGNING_PROFILE" \
    "${VALIDATOR_SUBTENSOR_NETWORK:-finney}" \
    "${VALIDATOR_NETUID:-71}" <<'PY'
import os
import sys
from pathlib import Path

from leadpoet_canonical.production_parity_boundary_v2 import (
    configured_chain_signing_profile_path_v2,
    validate_production_parity_boundary_document_v2,
)

network = sys.argv[2]
netuid = int(sys.argv[3])
validate_production_parity_boundary_document_v2(
    os.environ, network=network, netuid=netuid
)
print(configured_chain_signing_profile_path_v2(Path(sys.argv[1])))
PY
)"
test -r "$VALIDATOR_CHAIN_SIGNING_PROFILE" || {
  echo "ERROR: measured validator chain signing profile is unavailable" >&2
  exit 1
}
export VALIDATOR_CHAIN_SIGNING_PROFILE

VALIDATOR_WEIGHT_PROTOCOL="${VALIDATOR_WEIGHT_PROTOCOL:-authoritative_v2}"
case "$VALIDATOR_WEIGHT_PROTOCOL" in
  authoritative_v2)
    ;;
  *)
    echo "ERROR: VALIDATOR_WEIGHT_PROTOCOL must be authoritative_v2; V1 authority is retired" >&2
    exit 1
    ;;
esac
export VALIDATOR_WEIGHT_PROTOCOL
echo "Validator weight protocol: $VALIDATOR_WEIGHT_PROTOCOL"

# This is the same canonical Finney endpoint used by both validator protocol
# implementations. Keep the env override for non-production networks, but do
# not require production operators to duplicate the canonical value in the
# validator secret.
export EXPECTED_CHAIN="${EXPECTED_CHAIN:-wss://entrypoint-finney.opentensor.ai:443}"

# Shared scoring code still reads these legacy aliases. Derive them from the
# active provider credentials so they are not separate restart requirements.
export QUALIFICATION_OPENROUTER_API_KEY="${QUALIFICATION_OPENROUTER_API_KEY:-${OPENROUTER_API_KEY:-}}"
export QUALIFICATION_SCRAPINGDOG_API_KEY="${QUALIFICATION_SCRAPINGDOG_API_KEY:-${SCRAPINGDOG_API_KEY:-}}"

required_keys=(
  ENABLE_FULFILLMENT
  LEADPOET_WRAPPER_ACTIVE
  GATEWAY_URL
  SUPABASE_URL
  SUPABASE_ANON_KEY
  SUPABASE_SERVICE_ROLE_KEY
  OPENROUTER_API_KEY
  FULFILLMENT_OPENROUTER_API_KEY
  EXA_API_KEY
  SCRAPINGDOG_API_KEY
  AWS_REGION
  AWS_DEFAULT_REGION
  RESEARCH_LAB_VALIDATOR_FETCH_ENABLED
  RESEARCH_LAB_INTERNAL_API_KEY
  RESEARCH_LAB_WEIGHT_MUTATION_ENABLED
  RESEARCH_LAB_SUBMIT_ON_CHAIN_ENABLED
  EXPECTED_CHAIN
  NO_PROXY
)

required_keys+=(VALIDATOR_V2_GATEWAY_URL)

missing=()
for key in "${required_keys[@]}"; do
  if [ -z "${!key:-}" ]; then
    missing+=("$key")
  fi
done
if [ "${#missing[@]}" -gt 0 ]; then
  echo "ERROR: validator secret env missing required keys: ${missing[*]}"
  exit 1
fi

export no_proxy="${no_proxy:-$NO_PROXY}"
unset ENABLE_TEE_SUBMISSION VALIDATOR_ATTESTED_WEIGHT_MODE
unset VALIDATOR_REQUIRE_GATEWAY_WEIGHT_SUBMISSION DISABLE_GATEWAY_WEIGHT_SUBMISSION
unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_PROFILE AWS_SESSION_TOKEN AWS_SECURITY_TOKEN

HOST_HOTKEY_DIR="$VALIDATOR_WALLET_ROOT/$VALIDATOR_WALLET_NAME/hotkeys"
if [ -L "$HOST_HOTKEY_DIR" ] || { [ -e "$HOST_HOTKEY_DIR" ] && [ ! -d "$HOST_HOTKEY_DIR" ]; }; then
  echo "ERROR: validator hotkey directory is not a plain directory: $HOST_HOTKEY_DIR" >&2
  exit 1
fi
if [ ! -r "$VALIDATOR_WALLET_ROOT/$VALIDATOR_WALLET_NAME/coldkeypub.txt" ]; then
  echo "ERROR: public validator coldkey file is unavailable" >&2
  exit 1
fi

VALIDATOR_DEPLOY_SHA="$(git rev-parse HEAD)"
export VALIDATOR_DEPLOY_SHA
export VALIDATOR_V2_DEPLOY_COMMIT="$VALIDATOR_DEPLOY_SHA"
export GITHUB_SHA="$VALIDATOR_DEPLOY_SHA"
export GIT_COMMIT="$VALIDATOR_DEPLOY_SHA"
export LEADPOET_WRAPPER_ACTIVE=1
export VALIDATOR_EXACT_RELEASE_PINNED=1

case "$VALIDATOR_USE_CAPTURED_RESTART_START" in
  0|1) ;;
  *)
    echo "ERROR: LEADPOET_USE_CAPTURED_RESTART_START must be 0 or 1" >&2
    exit 1
    ;;
esac
case "$REQUESTED_STATEFUL_CUTOVER_PREPARE_ONLY" in
  0|1) ;;
  *)
    echo "ERROR: VALIDATOR_STATEFUL_CUTOVER_PREPARE_ONLY must be 0 or 1" >&2
    exit 1
    ;;
esac
if [ "$REQUESTED_STATEFUL_CUTOVER_PREPARE_ONLY" = "1" ] \
    && [ "$VALIDATOR_USE_CAPTURED_RESTART_START" != "1" ]; then
  echo "ERROR: stateful cutover enclave preparation requires a captured restart start" >&2
  exit 1
fi
if [ "$VALIDATOR_USE_CAPTURED_RESTART_START" = "1" ]; then
  test -s "$VALIDATOR_RESTART_START_PATH" || {
    echo "ERROR: captured validator restart start is missing" >&2
    exit 1
  }
  echo "Resuming the official restart start captured at operator invocation"
else
  echo "Capturing the official subnet restart start before release acquisition"
  "$VALIDATOR_PYTHON_BIN" -m Leadpoet.utils.restart_epoch_gate \
    --network "${VALIDATOR_SUBTENSOR_NETWORK:-finney}" \
    --netuid "${VALIDATOR_NETUID:-71}" \
    --capture-output "$VALIDATOR_RESTART_START_PATH"
  VALIDATOR_USE_CAPTURED_RESTART_START=1
fi

if ! follow_superseding_validator_release; then
  echo "Validator remains running; production shutdown has not started." >&2
  exit 75
fi
VALIDATOR_LOCAL_RELEASE_SCRIPT="$VALIDATOR_ROOT/gateway/tee/build_local_release_v2.sh"
VALIDATOR_LOCAL_RELEASE_MODULE="$VALIDATOR_ROOT/gateway/tee/local_release_v2.py"
VALIDATOR_HISTORICAL_RELEASE_MODULE="$VALIDATOR_ROOT/gateway/tee/release_channel_v2.py"
if [ -f "$VALIDATOR_LOCAL_RELEASE_SCRIPT" ] \
    && [ -r "$VALIDATOR_LOCAL_RELEASE_SCRIPT" ] \
    && [ ! -L "$VALIDATOR_LOCAL_RELEASE_SCRIPT" ] \
    && [ -f "$VALIDATOR_LOCAL_RELEASE_MODULE" ] \
    && [ -r "$VALIDATOR_LOCAL_RELEASE_MODULE" ] \
    && [ ! -L "$VALIDATOR_LOCAL_RELEASE_MODULE" ]; then
  echo "Preparing exact local build inputs before production shutdown"
  VALIDATOR_DEPLOY_STAGE="local_release_inputs"
  export GATEWAY_V2_OFFLINE_ARTIFACT_ROOT="${GATEWAY_V2_OFFLINE_ARTIFACT_ROOT:-$HOME/.cache/leadpoet-v2-artifacts}"
  export VALIDATOR_V2_OFFLINE_ARTIFACT_ROOT
  if ! bash "$VALIDATOR_ROOT/gateway/tee/prepare_offline_artifacts_v2.sh"; then
    echo "ERROR: exact local build inputs are unavailable" >&2
    echo "Validator remains running; production shutdown has not started." >&2
    exit 75
  fi
  if ! follow_superseding_validator_release; then
    echo "Validator remains running; production shutdown has not started." >&2
    exit 75
  fi

  echo "Building the exact local gateway and validator runtime identities"
  VALIDATOR_DEPLOY_STAGE="local_release_build"
  if ! PYTHONPATH="$VALIDATOR_ROOT" \
      GATEWAY_V2_OFFLINE_ARTIFACT_ROOT="$GATEWAY_V2_OFFLINE_ARTIFACT_ROOT" \
      VALIDATOR_V2_OFFLINE_ARTIFACT_ROOT="$VALIDATOR_V2_OFFLINE_ARTIFACT_ROOT" \
      bash "$VALIDATOR_ROOT/gateway/tee/build_local_release_v2.sh" \
        --repository "$VALIDATOR_ROOT" \
        --revision "$VALIDATOR_DEPLOY_SHA" \
        --gateway-output "$VALIDATOR_V2_GATEWAY_RELEASE_MANIFEST" \
        --validator-output "$VALIDATOR_V2_RELEASE_MANIFEST"; then
    echo "ERROR: exact local runtime identity build failed" >&2
    echo "Validator remains running; production shutdown has not started." >&2
    exit 75
  fi
  export LEADPOET_LOCAL_RELEASE_COMMIT_SHA="$VALIDATOR_DEPLOY_SHA"
  export LEADPOET_LOCAL_GATEWAY_RELEASE="$VALIDATOR_V2_GATEWAY_RELEASE_MANIFEST"
  export LEADPOET_LOCAL_VALIDATOR_RELEASE="$VALIDATOR_V2_RELEASE_MANIFEST"
  if [ -e "$VALIDATOR_V2_GATEWAY_RELEASE_LINEAGE" ] \
      || [ -L "$VALIDATOR_V2_GATEWAY_RELEASE_LINEAGE" ]; then
    export LEADPOET_LOCAL_PRIOR_RELEASE_LINEAGE="$VALIDATOR_V2_GATEWAY_RELEASE_LINEAGE"
  else
    unset LEADPOET_LOCAL_PRIOR_RELEASE_LINEAGE
  fi
  record_validator_restart_timing "local_release_ready"
elif [ ! -e "$VALIDATOR_LOCAL_RELEASE_SCRIPT" ] \
    && [ ! -L "$VALIDATOR_LOCAL_RELEASE_SCRIPT" ] \
    && [ ! -e "$VALIDATOR_LOCAL_RELEASE_MODULE" ] \
    && [ ! -L "$VALIDATOR_LOCAL_RELEASE_MODULE" ] \
    && [ -f "$VALIDATOR_HISTORICAL_RELEASE_MODULE" ] \
    && [ -r "$VALIDATOR_HISTORICAL_RELEASE_MODULE" ] \
    && [ ! -L "$VALIDATOR_HISTORICAL_RELEASE_MODULE" ] \
    && [ "$VALIDATOR_HISTORICAL_TOPOLOGY_HASH" = "$HISTORICAL_THREE_ROLE_TOPOLOGY_HASH" ] \
    && [ -n "$REQUESTED_VALIDATOR_DEPLOY_COMMIT" ] \
    && [ "$VALIDATOR_DEPLOY_SHA" != "$VALIDATOR_ACTIVE_RELEASE_AUTHORITY_COMMIT" ]; then
  echo "Acquiring the exact historical attested V2 release channel"
  VALIDATOR_DEPLOY_STAGE="historical_release_acquisition"
  unset LEADPOET_LOCAL_RELEASE_COMMIT_SHA LEADPOET_LOCAL_GATEWAY_RELEASE
  unset LEADPOET_LOCAL_VALIDATOR_RELEASE LEADPOET_LOCAL_PRIOR_RELEASE_LINEAGE
  VALIDATOR_V2_RELEASE_READY=0
  for attempt in $(seq 1 300); do
    VALIDATOR_RELEASE_ATTEMPTS_USED="$attempt"
    if follow_superseding_validator_release \
        && PYTHONPATH="$VALIDATOR_ROOT" "$VALIDATOR_PYTHON_BIN" \
        -m gateway.tee.release_channel_v2 \
        --ensure \
        --expected-commit "$VALIDATOR_DEPLOY_SHA" \
        --bucket "$VALIDATOR_V2_RELEASE_BUCKET" \
        --prefix "$VALIDATOR_V2_RELEASE_PREFIX" \
        --gateway-output "$VALIDATOR_V2_GATEWAY_RELEASE_MANIFEST" \
        --validator-output "$VALIDATOR_V2_RELEASE_MANIFEST"; then
      VALIDATOR_V2_RELEASE_READY=1
      break
    fi
    echo "Exact historical V2 release is not published yet; waiting inside the valid validator restart invocation (${attempt}/300)"
    sleep 12
  done
  if [ "$VALIDATOR_V2_RELEASE_READY" != "1" ]; then
    echo "ERROR: exact historical attested V2 release is unavailable for $VALIDATOR_DEPLOY_SHA" >&2
    echo "Validator remains running; production shutdown has not started." >&2
    exit 75
  fi
  record_validator_restart_timing "release_ready"
  record_validator_restart_timing "historical_release_ready"
else
  echo "ERROR: selected release has an incomplete or unsupported V2 release acquisition contract" >&2
  echo "Validator remains running; production shutdown has not started." >&2
  exit 75
fi
if ! follow_superseding_validator_release; then
  echo "Validator remains running; production shutdown has not started." >&2
  exit 75
fi
VALIDATOR_V2_MISSING_INPUTS=()
for required_file in \
  "$VALIDATOR_V2_GATEWAY_RELEASE_MANIFEST" \
  "$VALIDATOR_V2_RELEASE_MANIFEST" \
  "$VALIDATOR_V2_HOTKEY_CONFIG" \
  "$VALIDATOR_V2_HOTKEY_ENVELOPE"; do
  if [ ! -r "$required_file" ]; then
    VALIDATOR_V2_MISSING_INPUTS+=("$required_file")
  fi
done
if [ "${#VALIDATOR_V2_MISSING_INPUTS[@]}" -gt 0 ]; then
  python3 - "${VALIDATOR_V2_MISSING_INPUTS[@]}" <<'PY'
import json
import sys
print(json.dumps({
    "schema_version": "leadpoet.validator_v2_first_activation.v1",
    "status": "bootstrap_pending",
    "production_shutdown_started": False,
    "missing_paths": sys.argv[1:],
    "required_external_approvals": [
        "verified_validator_hotkey_envelope_and_offline_custody",
    ],
}, sort_keys=True, indent=2))
PY
  echo "Validator remains untouched. Complete the V2 bootstrap ceremony, then rerun this restart." >&2
  exit 75
fi

VALIDATOR_ANCESTRY_LINEAGE_ID="$(
  PYTHONPATH="$VALIDATOR_ROOT" "$VALIDATOR_PYTHON_BIN" - <<'PY'
from Leadpoet.utils.subnet_epoch import load_subnet_epoch_cutover
from leadpoet_canonical.ancestry_checkpoint_v2 import (
    derive_ancestry_lineage_id_v2,
)

cutover = load_subnet_epoch_cutover()
print(
    derive_ancestry_lineage_id_v2(
        cutover_mapping_hash=str(cutover.mapping_hash),
        network_genesis_hash=str(cutover.network_genesis_hash),
        netuid=int(cutover.netuid),
    )
)
PY
)"
if ! [[ "$VALIDATOR_ANCESTRY_LINEAGE_ID" =~ ^sha256:[0-9a-f]{64}$ ]]; then
  echo "ERROR: validator ancestry lineage identity is invalid" >&2
  exit 1
fi

VALIDATOR_INITIAL_TRANSITION_ARGS=()
if VALIDATOR_RUNNING_INSPECT_OUTPUT="$(
    docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' \
      leadpoet-validator-main 2>&1
)"; then
  if [ "$missing_runtime_recovery_count" -ne 0 ]; then
    echo "ERROR: validator missing-runtime recovery was supplied while the validator container exists" >&2
    exit 1
  fi
  VALIDATOR_RUNNING_DEPLOY_SHA="$(
    printf '%s\n' "$VALIDATOR_RUNNING_INSPECT_OUTPUT" \
      | sed -n 's/^VALIDATOR_V2_DEPLOY_COMMIT=//p'
  )"
  if ! [[ "$VALIDATOR_RUNNING_DEPLOY_SHA" =~ ^[0-9a-f]{40}$ ]]; then
    echo "ERROR: running validator release identity is invalid or duplicated" >&2
    exit 1
  fi
  VALIDATOR_INITIAL_TRANSITION_ARGS=(
    --running-validator-commit "$VALIDATOR_RUNNING_DEPLOY_SHA"
  )
else
  if [ "$VALIDATOR_RUNNING_INSPECT_OUTPUT" != "Error: No such object: leadpoet-validator-main" ]; then
    printf '%s\n' "$VALIDATOR_RUNNING_INSPECT_OUTPUT" >&2
    echo "ERROR: validator runtime state could not be established safely" >&2
    exit 1
  fi
  if [ "$missing_runtime_recovery_count" -ne 2 ]; then
    echo "ERROR: running validator release identity is unavailable" >&2
    exit 1
  fi
  echo "Recovering missing validator runtime from exact gateway release authority"
  VALIDATOR_INITIAL_TRANSITION_ARGS=(
    --recovery-requirements \
      "$VALIDATOR_MISSING_RUNTIME_RECOVERY_REQUIREMENTS"
    --recovery-lineage "$VALIDATOR_MISSING_RUNTIME_RECOVERY_LINEAGE"
  )
fi
if [ -e "$VALIDATOR_ACTIVE_RELEASE_REQUIREMENTS_OUTPUT" ] \
    || [ -L "$VALIDATOR_ACTIVE_RELEASE_REQUIREMENTS_OUTPUT" ]; then
  echo "ERROR: validator initial active release output already exists" >&2
  exit 1
fi

run_validator_active_release_phase() {
  local -a topology_authority_args=()
  if [ -n "$VALIDATOR_HISTORICAL_TOPOLOGY_HASH" ] \
      && [ -n "$VALIDATOR_ACTIVE_RELEASE_AUTHORITY_ROOT" ]; then
    topology_authority_args=(
      --historical-topology-hash "$VALIDATOR_HISTORICAL_TOPOLOGY_HASH"
    )
  fi
  sudo env \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="$VALIDATOR_ACTIVE_RELEASE_CONTROLLER_ROOT" \
    AWS_REGION="$AWS_REGION" \
    AWS_DEFAULT_REGION="$AWS_DEFAULT_REGION" \
    LEADPOET_LOCAL_RELEASE_COMMIT_SHA="${LEADPOET_LOCAL_RELEASE_COMMIT_SHA:-}" \
    LEADPOET_LOCAL_GATEWAY_RELEASE="${LEADPOET_LOCAL_GATEWAY_RELEASE:-}" \
    LEADPOET_LOCAL_VALIDATOR_RELEASE="${LEADPOET_LOCAL_VALIDATOR_RELEASE:-}" \
    LEADPOET_LOCAL_PRIOR_RELEASE_LINEAGE="${LEADPOET_LOCAL_PRIOR_RELEASE_LINEAGE:-}" \
    "$VALIDATOR_PYTHON_BIN" \
    "$VALIDATOR_ACTIVE_RELEASE_PREPARER" \
    "${topology_authority_args[@]}" "$@"
}

echo "Preparing the running validator active release requirements"
VALIDATOR_DEPLOY_STAGE="active_release_initial"
run_validator_active_release_phase \
  --phase validator-initial \
  --candidate-commit "$VALIDATOR_DEPLOY_SHA" \
  --authority-commit "$VALIDATOR_ACTIVE_RELEASE_AUTHORITY_COMMIT" \
  --restart-invocation-id "$VALIDATOR_ACTIVE_RELEASE_RESTART_INVOCATION_ID" \
  "${VALIDATOR_INITIAL_TRANSITION_ARGS[@]}" \
  --journal "$VALIDATOR_ACTIVE_PUBLICATION_JOURNAL" \
  --validator-hotkey-config "$VALIDATOR_V2_HOTKEY_CONFIG" \
  --chain-signing-profile "$VALIDATOR_CHAIN_SIGNING_PROFILE" \
  --repository "$VALIDATOR_ROOT" \
  --lineage-id "$VALIDATOR_ANCESTRY_LINEAGE_ID" \
  --bucket "$VALIDATOR_V2_RELEASE_BUCKET" \
  --prefix "$VALIDATOR_V2_RELEASE_PREFIX" \
  --requirements-output "$VALIDATOR_ACTIVE_RELEASE_REQUIREMENTS_OUTPUT"
if [ ! -s "$VALIDATOR_ACTIVE_RELEASE_REQUIREMENTS_OUTPUT" ] \
    || [ ! -f "$VALIDATOR_ACTIVE_RELEASE_REQUIREMENTS_OUTPUT" ] \
    || [ -L "$VALIDATOR_ACTIVE_RELEASE_REQUIREMENTS_OUTPUT" ]; then
  echo "ERROR: validator initial active release requirements are unavailable" >&2
  exit 1
fi
sudo chown -- "$(id -u):$(id -g)" \
  "$VALIDATOR_ACTIVE_RELEASE_REQUIREMENTS_OUTPUT"
chmod 600 "$VALIDATOR_ACTIVE_RELEASE_REQUIREMENTS_OUTPUT"
record_validator_restart_timing "active_release_initial_ready"
echo "Prepared validator active release requirements sidecar"

echo "Waiting for the paired gateway active release handoff while the validator remains running"
VALIDATOR_DEPLOY_STAGE="active_release_handoff"
VALIDATOR_ACTIVE_RELEASE_HANDOFF_READY=0
VALIDATOR_ACTIVE_RELEASE_HANDOFF_TIMEOUT_SECONDS="${VALIDATOR_PINNED_GATEWAY_TIMEOUT_SECONDS:-9300}"
if ! [[ "$VALIDATOR_ACTIVE_RELEASE_HANDOFF_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]] \
    || [ "$VALIDATOR_ACTIVE_RELEASE_HANDOFF_TIMEOUT_SECONDS" -gt 10800 ]; then
  echo "ERROR: validator active release handoff timeout is invalid" >&2
  exit 2
fi
VALIDATOR_ACTIVE_RELEASE_HANDOFF_DEADLINE=$((SECONDS + VALIDATOR_ACTIVE_RELEASE_HANDOFF_TIMEOUT_SECONDS))
while [ "$SECONDS" -lt "$VALIDATOR_ACTIVE_RELEASE_HANDOFF_DEADLINE" ]; do
  if [ -L "$VALIDATOR_FINAL_RELEASE_REQUIREMENTS_INPUT" ] \
      || [ -L "$VALIDATOR_FINAL_RELEASE_LINEAGE_INPUT" ] \
      || [ -d "$VALIDATOR_FINAL_RELEASE_REQUIREMENTS_INPUT" ] \
      || [ -d "$VALIDATOR_FINAL_RELEASE_LINEAGE_INPUT" ]; then
    echo "ERROR: validator final active release handoff is not a pair of plain files" >&2
    exit 1
  fi
  if [ -s "$VALIDATOR_FINAL_RELEASE_REQUIREMENTS_INPUT" ] \
      && [ -f "$VALIDATOR_FINAL_RELEASE_REQUIREMENTS_INPUT" ] \
      && [ -r "$VALIDATOR_FINAL_RELEASE_REQUIREMENTS_INPUT" ] \
      && [ -s "$VALIDATOR_FINAL_RELEASE_LINEAGE_INPUT" ] \
      && [ -f "$VALIDATOR_FINAL_RELEASE_LINEAGE_INPUT" ] \
      && [ -r "$VALIDATOR_FINAL_RELEASE_LINEAGE_INPUT" ]; then
    VALIDATOR_ACTIVE_RELEASE_HANDOFF_READY=1
    break
  fi
  sleep 1
done
if [ "$VALIDATOR_ACTIVE_RELEASE_HANDOFF_READY" != "1" ]; then
  echo "ERROR: paired gateway active release handoff did not arrive" >&2
  echo "Validator remains running; production shutdown has not started." >&2
  exit 75
fi
record_validator_restart_timing "active_release_handoff_ready"

echo "Refreshing public validator hotkey measurements for the selected release"
python3 -m validator_tee.host.refresh_hotkey_config_v2 \
  --config "$VALIDATOR_V2_HOTKEY_CONFIG" \
  --chain-profile \
    "$VALIDATOR_CHAIN_SIGNING_PROFILE" \
  --drand-hash \
    "$VALIDATOR_ROOT/validator_tee/enclave/libbittensor_drand_v2.sha256"

HOST_HOTKEY_ENTRY=""
if [ -d "$HOST_HOTKEY_DIR" ]; then
  HOST_HOTKEY_ENTRY="$(find "$HOST_HOTKEY_DIR" -mindepth 1 -maxdepth 1 -print -quit)"
fi
if [ -n "$HOST_HOTKEY_ENTRY" ]; then
  echo "ERROR: usable validator hotkey material remains on the parent: $HOST_HOTKEY_ENTRY" >&2
  echo "Create and verify the KMS envelope, move the host hotkey to approved offline custody, then restart." >&2
  exit 1
fi

echo "Preparing exact hash-locked validator artifacts before production shutdown"
python3 -m validator_tee.scripts.stage_runtime_artifacts_v2 \
  --lock "$VALIDATOR_ROOT/validator_tee/runtime-artifacts-v2.lock.json" \
  --output-dir "$VALIDATOR_V2_OFFLINE_ARTIFACT_ROOT" \
  --allow-download >/dev/null

echo "Validating the measured chain signing profile against the live runtime"
python3 -m validator_tee.host.verify_chain_signing_profile_v2 \
  --network "${VALIDATOR_SUBTENSOR_NETWORK:-finney}" \
  --profile \
    "$VALIDATOR_CHAIN_SIGNING_PROFILE"

actual_aws_account="$(aws sts get-caller-identity --query Account --output text)"
if [ "$actual_aws_account" != "$EXPECTED_AWS_ACCOUNT" ]; then
  echo "ERROR: validator AWS account is $actual_aws_account, expected $EXPECTED_AWS_ACCOUNT"
  exit 1
fi

VALIDATOR_RESTART_GATE_ARGS=(
  --network "${VALIDATOR_SUBTENSOR_NETWORK:-finney}"
  --netuid "${VALIDATOR_NETUID:-71}"
  --captured-report "$VALIDATOR_RESTART_START_PATH"
)
echo "Validating the official restart start captured at operator invocation"
VALIDATOR_DEPLOY_STAGE="pre_shutdown_validation"
if ! "$VALIDATOR_PYTHON_BIN" -m Leadpoet.utils.restart_epoch_gate \
    "${VALIDATOR_RESTART_GATE_ARGS[@]}"; then
  echo "Validator remains running; production shutdown has not started." >&2
  exit 75
fi
record_validator_restart_timing "pre_shutdown_checks_complete"
if [ "$REQUESTED_STATEFUL_CUTOVER_PREPARE_ONLY" != "1" ]; then
  unset LEADPOET_USE_CAPTURED_RESTART_START
fi

if [ "$REQUESTED_STATEFUL_CUTOVER_PREPARE_ONLY" != "1" ]; then
  echo "Checking same-SHA gateway readiness before stopping the running validator"
  VALIDATOR_DEPLOY_STAGE="pre_shutdown_gateway_alignment"
  if ! verify_forward_gateway_release_before_shutdown \
      "${VALIDATOR_PINNED_GATEWAY_PRESTART_MAX_ATTEMPTS:-3000}"; then
    echo "Validator remains running; production shutdown has not started." >&2
    exit 1
  fi
  record_validator_restart_timing "pre_shutdown_gateway_aligned"
fi

echo "Independently verifying the paired active release lineage before shutdown"
VALIDATOR_DEPLOY_STAGE="active_release_final"
prepare_validator_final_active_release_lineage() {
  run_validator_active_release_phase \
    --phase validator-final \
    --candidate-commit "$VALIDATOR_DEPLOY_SHA" \
    --authority-commit "$VALIDATOR_ACTIVE_RELEASE_AUTHORITY_COMMIT" \
    --restart-invocation-id "$VALIDATOR_ACTIVE_RELEASE_RESTART_INVOCATION_ID" \
    --initial-requirements "$VALIDATOR_ACTIVE_RELEASE_REQUIREMENTS_OUTPUT" \
    --final-requirements-input "$VALIDATOR_FINAL_RELEASE_REQUIREMENTS_INPUT" \
    --lineage-input "$VALIDATOR_FINAL_RELEASE_LINEAGE_INPUT" \
    --journal "$VALIDATOR_ACTIVE_PUBLICATION_JOURNAL" \
    --validator-hotkey-config "$VALIDATOR_V2_HOTKEY_CONFIG" \
    --chain-signing-profile "$VALIDATOR_CHAIN_SIGNING_PROFILE" \
    --repository "$VALIDATOR_ROOT" \
    --lineage-id "$VALIDATOR_ANCESTRY_LINEAGE_ID" \
    --bucket "$VALIDATOR_V2_RELEASE_BUCKET" \
    --prefix "$VALIDATOR_V2_RELEASE_PREFIX" \
    --requirements-output "$VALIDATOR_V2_GATEWAY_RELEASE_REQUIREMENTS" \
    --lineage-output "$VALIDATOR_V2_GATEWAY_RELEASE_LINEAGE"
}
prepare_validator_final_active_release_lineage

VALIDATOR_V2_MISSING_INPUTS=()
for required_file in \
  "$VALIDATOR_V2_GATEWAY_RELEASE_REQUIREMENTS" \
  "$VALIDATOR_V2_GATEWAY_RELEASE_LINEAGE"; do
  if [ ! -s "$required_file" ] || [ ! -f "$required_file" ] \
      || [ -L "$required_file" ]; then
    VALIDATOR_V2_MISSING_INPUTS+=("$required_file")
  fi
done
if [ "${#VALIDATOR_V2_MISSING_INPUTS[@]}" -gt 0 ]; then
  echo "ERROR: verified validator active release outputs are unavailable" >&2
  echo "Validator remains running; production shutdown has not started." >&2
  exit 1
fi
sudo chown -- "$(id -u):$(id -g)" \
  "$VALIDATOR_V2_GATEWAY_RELEASE_REQUIREMENTS" \
  "$VALIDATOR_V2_GATEWAY_RELEASE_LINEAGE"
chmod 600 \
  "$VALIDATOR_V2_GATEWAY_RELEASE_REQUIREMENTS" \
  "$VALIDATOR_V2_GATEWAY_RELEASE_LINEAGE"
VALIDATOR_FINAL_ACTIVE_RELEASE_HASHES="$(
  sha256sum \
    "$VALIDATOR_V2_GATEWAY_RELEASE_REQUIREMENTS" \
    "$VALIDATOR_V2_GATEWAY_RELEASE_LINEAGE"
)"
record_validator_restart_timing "active_release_lineage_ready"

echo "Validating the exact validator V2 release before production shutdown"
python3 -m validator_tee.host.restart_preflight_v2 \
  --deploy-commit "$VALIDATOR_DEPLOY_SHA" \
  --validator-release "$VALIDATOR_V2_RELEASE_MANIFEST" \
  --gateway-release "$VALIDATOR_V2_GATEWAY_RELEASE_MANIFEST" \
  --gateway-release-lineage "$VALIDATOR_V2_GATEWAY_RELEASE_LINEAGE" \
  --hotkey-config "$VALIDATOR_V2_HOTKEY_CONFIG" \
  --hotkey-envelope "$VALIDATOR_V2_HOTKEY_ENVELOPE" \
  --runtime-artifact-lock "$VALIDATOR_ROOT/validator_tee/runtime-artifacts-v2.lock.json" \
  --host-hotkey-directory "$HOST_HOTKEY_DIR"

if [ ! -r "$VALIDATOR_DOCKER_OPERATION_LOCK_HELPER" ]; then
  echo "ERROR: validator Docker operation lock helper is unavailable" >&2
  exit 1
fi
. "$VALIDATOR_DOCKER_OPERATION_LOCK_HELPER"
leadpoet_acquire_docker_operation_lock_v2
VALIDATOR_DOCKER_LOCK_ACQUIRED=1
PYTHONPATH="$VALIDATOR_ROOT" "$VALIDATOR_PYTHON_BIN" \
  -m validator_tee.host.docker_operation_guard_v2 \
  --wait \
  --timeout-seconds 1800 \
  --interval-seconds 3
run_bounded_validator_restart_artifact_cleanup

echo "Rechecking publication journal and compact lineage immediately before validator shutdown"
VALIDATOR_DEPLOY_STAGE="active_release_final_recheck"
prepare_validator_final_active_release_lineage
sudo chown -- "$(id -u):$(id -g)" \
  "$VALIDATOR_V2_GATEWAY_RELEASE_REQUIREMENTS" \
  "$VALIDATOR_V2_GATEWAY_RELEASE_LINEAGE"
chmod 600 \
  "$VALIDATOR_V2_GATEWAY_RELEASE_REQUIREMENTS" \
  "$VALIDATOR_V2_GATEWAY_RELEASE_LINEAGE"
if [ "$(
    sha256sum \
      "$VALIDATOR_V2_GATEWAY_RELEASE_REQUIREMENTS" \
      "$VALIDATOR_V2_GATEWAY_RELEASE_LINEAGE"
  )" != "$VALIDATOR_FINAL_ACTIVE_RELEASE_HASHES" ]; then
  echo "ERROR: validator active release authority changed before shutdown" >&2
  echo "Validator remains running; production shutdown has not started." >&2
  exit 1
fi
record_validator_restart_timing "active_release_lineage_rechecked"

VALIDATOR_DESTRUCTIVE_PHASE_STARTED=1
VALIDATOR_DEPLOY_STAGE="runtime_rebuild"
record_validator_restart_timing "destructive_phase_started"
echo "Stopping validator processes and containers"
stop_lab_arena_runner
sudo pkill -TERM -f ".auto_update_wrapper.sh" 2>/dev/null || true
sudo pkill -TERM -f "neurons/validator.py" 2>/dev/null || true
sudo pkill -TERM -f "docker logs -f leadpoet-validator-main" 2>/dev/null || true
sudo pkill -TERM -f "validator_tee.host.chain_relay_v2" 2>/dev/null || true
sleep 5

sudo pkill -KILL -f ".auto_update_wrapper.sh" 2>/dev/null || true
sudo pkill -KILL -f "neurons/validator.py" 2>/dev/null || true
sudo pkill -KILL -f "docker logs -f leadpoet-validator-main" 2>/dev/null || true
sudo pkill -KILL -f "validator_tee.host.chain_relay_v2" 2>/dev/null || true
sleep 2

docker ps -aq \
  --filter "name=leadpoet-validator" \
  --filter "name=leadpoet-qual-worker" \
  --filter "name=leadpoet-ff-worker" \
  | xargs -r docker stop

  docker ps -aq \
  --filter "name=leadpoet-validator" \
  --filter "name=leadpoet-qual-worker" \
  --filter "name=leadpoet-ff-worker" \
  | xargs -r docker rm

echo "Terminating existing validator Nitro enclaves"
  sudo nitro-cli terminate-enclave --all 2>/dev/null || true
  for attempt in $(seq 1 10); do
    enclave_count="$(
      sudo nitro-cli describe-enclaves \
        | python3 -c 'import json, sys; print(len(json.load(sys.stdin)))'
    )"
    if [ "$enclave_count" -eq 0 ]; then
      echo "Validator Nitro enclave pool is empty"
      break
    fi
    if [ "$attempt" -eq 10 ]; then
      echo "ERROR: ${enclave_count} validator Nitro enclave(s) remain after termination" >&2
      sudo nitro-cli describe-enclaves >&2 || true
      exit 1
    fi
    sleep 1
  done

  echo "Trimming bounded, regenerable validator host caches and archived journals"
  VALIDATOR_JOURNAL_VACUUM_SIZE="${VALIDATOR_JOURNAL_VACUUM_SIZE:-1G}"
  if command -v journalctl >/dev/null 2>&1; then
    if ! sudo journalctl --rotate \
        || ! sudo journalctl \
          --vacuum-size="$VALIDATOR_JOURNAL_VACUUM_SIZE"; then
      echo "WARNING: validator_host_journal_cleanup_failed" >&2
    fi
  fi
  if [ -d "$HOME/.cache/pip" ]; then
    if ! rm -rf -- "$HOME/.cache/pip"; then
      echo "WARNING: validator_host_pip_cache_cleanup_failed" >&2
    fi
  fi

  echo "Reclaiming validator Docker storage before the independent rebuild"
  VALIDATOR_DOCKER_ALLOW_DATA_ROOT_RESET=1 \
    bash validator_tee/scripts/reclaim_docker_storage_v2.sh

  echo "Deleting validator-base:v1 so validator independently rebuilds it"
  docker rmi -f validator-base:v1 2>/dev/null || true

  echo "Building validator enclave"
  bash validator_tee/scripts/build_enclave.sh
  test -f validator_tee/validator-enclave.eif
  echo "Verifying local EIF against the approved six-build validator release"
  python3 -m validator_tee.host.verify_release_gate_v2 \
    --verify-manifest "$VALIDATOR_V2_RELEASE_MANIFEST" \
    --local-release "$VALIDATOR_ROOT/validator_tee/validator-v2-release.json"
  echo "Archiving the complete verified validator V2 release"
  python3 -m validator_tee.host.release_archive_v2 \
    --archive \
    --release-manifest "$VALIDATOR_V2_RELEASE_MANIFEST" \
    --validator-tee-root "$VALIDATOR_ROOT/validator_tee" \
    --archive-root "$VALIDATOR_V2_RELEASE_ARCHIVE_ROOT" \
    --retain 3
  cd validator_tee
  sudo nitro-cli run-enclave \
    --eif-path validator-enclave.eif \
    --cpu-count "${VALIDATOR_ENCLAVE_CPU_COUNT:-2}" \
    --memory "${VALIDATOR_ENCLAVE_MEMORY_MB:-1024}"
  sleep 3
  cd "$VALIDATOR_ROOT"

  echo "Starting validator-enclave opaque chain TLS relay"
  CHAIN_RELAY_LOG="${VALIDATOR_CHAIN_RELAY_LOG:-/home/ec2-user/validator-chain-relay-v2.log}"
  setsid env PYTHONPATH="$VALIDATOR_ROOT" python3 -m validator_tee.host.chain_relay_v2 \
    >> "$CHAIN_RELAY_LOG" 2>&1 < /dev/null 7>&- 8>&- &
  CHAIN_RELAY_PID=$!
  sleep 2
  if ! kill -0 "$CHAIN_RELAY_PID" 2>/dev/null; then
    echo "ERROR: validator chain relay failed to start" >&2
    tail -80 "$CHAIN_RELAY_LOG" >&2 || true
    exit 1
  fi
  echo "Validator chain relay ready (pid=$CHAIN_RELAY_PID)"

  echo "Configuring the authoritative validator V2 release"
  python3 -m validator_tee.host.runtime_v2_bootstrap \
    --validator-release "$VALIDATOR_V2_RELEASE_MANIFEST" \
    --gateway-release "$VALIDATOR_V2_GATEWAY_RELEASE_MANIFEST" \
    --gateway-release-lineage "$VALIDATOR_V2_GATEWAY_RELEASE_LINEAGE" \
    --hotkey-config "$VALIDATOR_V2_HOTKEY_CONFIG"
  record_validator_restart_timing "attested_enclave_ready"

  echo "Provisioning the validator hotkey directly into Nitro with KMS"
python3 -m validator_tee.host.hotkey_bootstrap_v2 \
  --hotkey-config "$VALIDATOR_V2_HOTKEY_CONFIG" \
  --hotkey-envelope "$VALIDATOR_V2_HOTKEY_ENVELOPE"

if [ "$REQUESTED_STATEFUL_CUTOVER_PREPARE_ONLY" = "1" ]; then
  install_validator_restart_controller
  leadpoet_release_docker_operation_lock_v2
  VALIDATOR_DOCKER_LOCK_ACQUIRED=0
  VALIDATOR_RESTART_COMPLETED=1
  VALIDATOR_DEPLOY_STAGE="completed"
  record_validator_restart_timing "cutover_prepare_complete" "passed"
  echo "SUCCESS: exact attested validator enclave is prepared for stateful cutover boundary capture"
  exit 0
fi

VALIDATOR_CONTAINER_DEPLOY_SCRIPT="$(
  printf '%s/%s' \
    "$VALIDATOR_ROOT" \
    "validator_models/containerizing/deploy_dynamic.sh"
)"
if grep -Fq \
    "VALIDATOR_GATEWAY_ACTIVATION_BARRIER_V2=1" \
    "$VALIDATOR_CONTAINER_DEPLOY_SCRIPT"; then
  echo "Deferring same-SHA gateway alignment until the exact validator application image is verified"
  export VALIDATOR_GATEWAY_ACTIVATION_VERIFIER="$VALIDATOR_PINNED_GATEWAY_VERIFIER"
else
  echo "Selected rollback deployer lacks the image-prepared gateway activation barrier"
  echo "Checking same-SHA gateway alignment before invoking the legacy deployer"
  if ! verify_pinned_gateway_release \
      "${VALIDATOR_PINNED_GATEWAY_PRESTART_MAX_ATTEMPTS:-3000}"; then
    exit 1
  fi
fi

echo "Starting validator"
VALIDATOR_DEPLOY_STAGE="validator_application_start"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="${PYTHONPATH:-$VALIDATOR_ROOT}"
export VALIDATOR_FORCE_CONTAINER_DEPLOY=1
export VALIDATOR_AUTO_CONTAINER_FOLLOW_LOGS=0

if ! "$VALIDATOR_PYTHON_BIN" neurons/validator.py \
    --netuid "${VALIDATOR_NETUID:-71}" \
    --subtensor_network "${VALIDATOR_SUBTENSOR_NETWORK:-finney}" \
    --wallet_name "$VALIDATOR_WALLET_NAME" \
    --wallet_hotkey "$VALIDATOR_WALLET_HOTKEY"; then
  echo "ERROR: validator container preparation or activation failed" >&2
  exit 1
fi

if [ "$(docker inspect -f '{{.State.Running}}' leadpoet-validator-main)" != "true" ] \
    || [ "$(docker inspect -f '{{.RestartCount}}' leadpoet-validator-main)" != "0" ]; then
  echo "ERROR: validator coordinator failed its final restart-wrapper check" >&2
  docker logs --tail 160 leadpoet-validator-main >&2 || true
  exit 1
fi
record_validator_restart_timing "validator_application_ready"
echo "Rechecking same-SHA gateway alignment after validator startup"
VALIDATOR_DEPLOY_STAGE="gateway_alignment"
if ! verify_pinned_gateway_release \
    "${VALIDATOR_PINNED_GATEWAY_POSTSTART_MAX_ATTEMPTS:-12}"; then
  stop_pinned_validator_after_alignment_failure
  exit 1
fi
VALIDATOR_DEPLOY_STAGE="lab_arena_runner_start"
start_lab_arena_runner
record_validator_restart_timing "lab_arena_runner_ready"
install_validator_restart_controller
leadpoet_release_docker_operation_lock_v2
VALIDATOR_DOCKER_LOCK_ACQUIRED=0
if [ "$VALIDATOR_USE_CAPTURED_RESTART_START" = "1" ]; then
  rm -f "$VALIDATOR_RESTART_START_PATH"
fi
VALIDATOR_RESTART_COMPLETED=1
VALIDATOR_DEPLOY_STAGE="completed"
record_validator_restart_timing "completed" "passed"
echo "SUCCESS: authoritative V2 validator restart completed and verified"
