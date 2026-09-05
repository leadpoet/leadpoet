#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PRODUCTION_GATEWAY_HOST="ec2-user@52.91.135.79"
PRODUCTION_GATEWAY_RESTART="/home/ec2-user/gw_restart.sh"
PRODUCTION_GATEWAY_REPO_ROOT="/home/ec2-user/leadpoet_repo"
PRODUCTION_GATEWAY_PYTHON_BIN="/home/ec2-user/venv311/bin/python3"
PRODUCTION_VALIDATOR_PYTHON_BIN="/home/ec2-user/venv311/bin/python3"
PRODUCTION_GATEWAY_DEPLOY_READINESS_PATH="/home/ec2-user/gateway/deploy_readiness.json"
PRODUCTION_GATEWAY_RESTART_CONTROLLER_ROOT="/home/ec2-user/.config/leadpoet/restart-controller/gateway"
PRODUCTION_GATEWAY_RESTART_CONTROLLER_CURRENT="$PRODUCTION_GATEWAY_RESTART_CONTROLLER_ROOT/current"
GATEWAY_KEY="${LEADPOET_GATEWAY_SSH_KEY:-$HOME/Downloads/leadpoet-2026-07-28.pem}"
VALIDATOR_KEY="${LEADPOET_VALIDATOR_SSH_KEY:-$HOME/Downloads/leadpoet-2026-07-28.pem}"
GATEWAY_HOST="${LEADPOET_GATEWAY_SSH_HOST:-$PRODUCTION_GATEWAY_HOST}"
VALIDATOR_HOST="${LEADPOET_VALIDATOR_SSH_HOST:-ec2-user@100.59.201.156}"
GATEWAY_RESTART="${LEADPOET_GATEWAY_RESTART_PATH:-$PRODUCTION_GATEWAY_RESTART}"
VALIDATOR_RESTART="${LEADPOET_VALIDATOR_RESTART_PATH:-/home/ec2-user/validator_restart.sh}"
GATEWAY_REPO_ROOT="${LEADPOET_GATEWAY_REPO_ROOT:-$PRODUCTION_GATEWAY_REPO_ROOT}"
GATEWAY_PYTHON_BIN="${LEADPOET_GATEWAY_PYTHON_BIN:-$PRODUCTION_GATEWAY_PYTHON_BIN}"
VALIDATOR_REPO_ROOT="${LEADPOET_VALIDATOR_REPO_ROOT:-/home/ec2-user/leadpoet/leadpoet}"
VALIDATOR_PYTHON_BIN="${LEADPOET_VALIDATOR_PYTHON_BIN:-$PRODUCTION_VALIDATOR_PYTHON_BIN}"
VALIDATOR_V2_HOTKEY_CONFIG_PATH="${LEADPOET_VALIDATOR_V2_HOTKEY_CONFIG_PATH:-/home/ec2-user/.config/leadpoet/validator-hotkey-config-v2.json}"
VALIDATOR_LOCAL_GATEWAY_RELEASE_PATH="${LEADPOET_VALIDATOR_LOCAL_GATEWAY_RELEASE_PATH:-/home/ec2-user/.config/leadpoet/gateway-v2-release-manifest.json}"
VALIDATOR_LOCAL_RELEASE_PATH="${LEADPOET_VALIDATOR_LOCAL_RELEASE_PATH:-/home/ec2-user/.config/leadpoet/validator-v2-release-manifest.json}"
VALIDATOR_LOCAL_RELEASE_LINEAGE_PATH="${LEADPOET_VALIDATOR_LOCAL_RELEASE_LINEAGE_PATH:-/home/ec2-user/.config/leadpoet/gateway-v2-release-lineage.json}"
VALIDATOR_CHAIN_SIGNING_PROFILE_PATH="${LEADPOET_VALIDATOR_CHAIN_SIGNING_PROFILE_PATH:-$VALIDATOR_REPO_ROOT/validator_tee/enclave/chain_signing_profile_v2.json}"
VALIDATOR_STATEFUL_CUTOVER_MANIFEST="/home/ec2-user/.config/leadpoet/stateful-epoch-cutover.json"
GATEWAY_DEPLOY_READINESS_PATH="${LEADPOET_GATEWAY_DEPLOY_READINESS_PATH:-$PRODUCTION_GATEWAY_DEPLOY_READINESS_PATH}"
GATEWAY_ENV_SECRET_ID="${LEADPOET_GATEWAY_ENV_SECRET_ID:-}"
VALIDATOR_ENV_SECRET_ID="${LEADPOET_VALIDATOR_ENV_SECRET_ID:-}"
RELEASE_PREFIX="${LEADPOET_RELEASE_PREFIX:-attested-v2/releases}"
HISTORICAL_THREE_ROLE_TOPOLOGY_HASH="sha256:a13a1b16fb1501f953b2396aba88b87d7e5e0d3cfac4079b9230ea6165a88f34"
HISTORICAL_THREE_ROLE_TOPOLOGY_BLOB="f79cf108e4a98ca950a0087d786958f92c5f691f"
# Three-second retries allow up to 2.5 hours for the gateway rebuild, plus a
# five-minute margin for the final bounded release probes.
VALIDATOR_COORDINATION_ATTEMPTS=3000
VALIDATOR_COORDINATION_TIMEOUT_SECONDS=9300
VALIDATOR_FAILURE_CLEANUP_ATTEMPTS=60
VALIDATOR_FAILURE_MARKER_ATTEMPTS=20

commit=""
component="all"
local_python=""
local_python_target=""
local_python_venv_root=""
local_python_site_packages=""
local_python_link_identity=""
local_python_target_identity=""
local_python_venv_config_identity=""
local_python_site_identity=""
local_readiness_candidate_bound=0
disable_miner_submissions_before_restart=0
validator_job=""
gateway_job=""
gateway_job_pgid=""
gateway_job_session_file=""
gateway_restart_command=()
failure_marker_job=""
temporary_root=""
coordination_file=""
gateway_handoff_file=""
gateway_handoff_nonce=""
paired_gateway_handoff_file=""
paired_gateway_handoff_nonce=""
active_release_restart_invocation_id=""
active_release_authority_commit=""
controller_verifier_b64=""
expected_controller_commit=""
gateway_restart_log=""
gateway_observation=""
gateway_evidence=""
validator_observation=""
validator_evidence=""
transition_manifest=""
final_manifest=""
historical_topology_hash=""
validator_initial_requirements_remote=""
gateway_validator_requirements_remote=""
validator_final_requirements_remote=""
validator_final_lineage_remote=""
validator_recovery_requirements_remote=""
validator_recovery_lineage_remote=""
validator_missing_runtime_recovery=0
validator_initial_requirements_local=""
gateway_final_requirements_local=""
gateway_final_lineage_local=""
GATEWAY_ACTIVE_RELEASE_REQUIREMENTS_PATH="${LEADPOET_GATEWAY_ACTIVE_RELEASE_REQUIREMENTS_PATH:-/home/ec2-user/tee/gateway-v2-release-requirements.json}"
GATEWAY_ACTIVE_RELEASE_LINEAGE_PATH="${LEADPOET_GATEWAY_ACTIVE_RELEASE_LINEAGE_PATH:-/home/ec2-user/tee/gateway-v2-release-lineage.json}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/restart_attested_release_local.sh \
    --commit <full-40-character-sha> \
    [--component all|gateway|validator] \
    --local-python </absolute/venv/bin/python> \
    [--disable-miner-submissions-before-restart]

The default "all" mode starts both exact-commit restarts in one invocation.
A single-component restart is accepted only when the other component is
already running the selected commit.
The miner-submission option is paired-only. It durably pauses SOURCE_ADD,
holds the canonical restart guard with one invocation-specific owner and
monotonic generation, drains every leased SOURCE_ADD
work item to an exact zero readback, proves intake closed, and prepares the
exact candidate under the canonical gateway lock before the installed N-1
wrapper hydrates. A fresh retry takes over the same guard at a new generation,
fencing the prior invocation. The exact owner/generation is renewed with a
14,400-second lease and the zero-lease state is rechecked immediately before
shutdown and after candidate startup. A drain timeout aborts while
leaving SOURCE_ADD paused and guarded; successful runtime verification releases
the guard and atomically restores the SOURCE_ADD pause state that existed before
the restart. A failed restart remains paused.
EOF
}

cleanup() {
  local status="$?"
  local failure_marker_command=""
  local gateway_cancel_job=""
  set +e
  if [ -n "$gateway_job" ]; then
    if [ -n "$gateway_job_pgid" ]; then
      kill -TERM -- "-$gateway_job_pgid" 2>/dev/null || true
    else
      kill -TERM "$gateway_job" 2>/dev/null || true
    fi
    if [ -n "$gateway_handoff_file" ] && [ -n "$gateway_handoff_nonce" ]; then
      publish_gateway_handoff_value "failed:$commit" >/dev/null 2>&1 &
      gateway_cancel_job="$!"
    fi
    if [ -n "$paired_gateway_handoff_file" ] \
        && [ -n "$paired_gateway_handoff_nonce" ]; then
      publish_paired_gateway_handoff_value "failed:$commit" \
        >/dev/null 2>&1 || true
    fi
    for _ in $(seq 1 20); do
      if [ -n "$gateway_job_pgid" ] \
          && kill -0 -- "-$gateway_job_pgid" 2>/dev/null; then
        sleep 0.1
        continue
      fi
      if ! kill -0 "$gateway_job" 2>/dev/null; then
        break
      fi
      sleep 0.1
    done
    if { [ -n "$gateway_job_pgid" ] \
          && kill -0 -- "-$gateway_job_pgid" 2>/dev/null; } \
        || kill -0 "$gateway_job" 2>/dev/null; then
      if [ -n "$gateway_job_pgid" ]; then
        kill -KILL -- "-$gateway_job_pgid" 2>/dev/null || true
      fi
      kill -KILL "$gateway_job" 2>/dev/null || true
      for _ in $(seq 1 20); do
        if ! kill -0 "$gateway_job" 2>/dev/null \
            && { [ -z "$gateway_job_pgid" ] \
              || ! kill -0 -- "-$gateway_job_pgid" 2>/dev/null; }; then
          break
        fi
        sleep 0.1
      done
    fi
    if ! kill -0 "$gateway_job" 2>/dev/null; then
      wait "$gateway_job" 2>/dev/null || true
    fi
    gateway_job=""
    gateway_job_pgid=""
    if [ -n "$gateway_cancel_job" ] && kill -0 "$gateway_cancel_job" 2>/dev/null; then
      kill -TERM "$gateway_cancel_job" 2>/dev/null || true
    fi
    if [ -n "$gateway_cancel_job" ]; then
      for _ in $(seq 1 10); do
        if ! kill -0 "$gateway_cancel_job" 2>/dev/null; then
          break
        fi
        sleep 0.1
      done
      if kill -0 "$gateway_cancel_job" 2>/dev/null; then
        kill -KILL "$gateway_cancel_job" 2>/dev/null || true
      fi
      wait "$gateway_cancel_job" 2>/dev/null || true
    fi
  fi
  if [ -n "$validator_job" ] && kill -0 "$validator_job" 2>/dev/null; then
    # Cancellation is the authority revocation. Signal the remote validator
    # before attempting the durable failure marker so a slow SSH connection
    # cannot delay cleanup past the late activation boundary.
    kill -TERM "$validator_job" 2>/dev/null || true
    if [ -n "$coordination_file" ]; then
      echo "Signaling the paired validator to clean up after restart failure" >&2
      failure_marker_command="$(
        coordination_remote_command "failed:$commit"
      )"
      ssh "${ssh_common[@]}" -i "$VALIDATOR_KEY" "$VALIDATOR_HOST" \
        "$failure_marker_command" >/dev/null 2>&1 &
      failure_marker_job="$!"
    fi
    for _ in $(seq 1 "$VALIDATOR_FAILURE_CLEANUP_ATTEMPTS"); do
      if ! kill -0 "$validator_job" 2>/dev/null; then
        break
      fi
      sleep 1
    done
    if kill -0 "$validator_job" 2>/dev/null; then
      kill -KILL "$validator_job" 2>/dev/null || true
    fi
    wait "$validator_job" 2>/dev/null || true
  fi
  if [ -n "$failure_marker_job" ]; then
    for _ in $(seq 1 "$VALIDATOR_FAILURE_MARKER_ATTEMPTS"); do
      if ! kill -0 "$failure_marker_job" 2>/dev/null; then
        break
      fi
      sleep 1
    done
    if kill -0 "$failure_marker_job" 2>/dev/null; then
      kill -TERM "$failure_marker_job" 2>/dev/null || true
      sleep 1
    fi
    if kill -0 "$failure_marker_job" 2>/dev/null; then
      kill -KILL "$failure_marker_job" 2>/dev/null || true
    fi
    wait "$failure_marker_job" 2>/dev/null || true
    failure_marker_job=""
  fi
  if [ -n "$coordination_file" ]; then
    ssh "${ssh_common[@]}" -i "$VALIDATOR_KEY" "$VALIDATOR_HOST" \
      "rm -f -- '$coordination_file'" >/dev/null 2>&1 || true
  fi
  if [ -n "$gateway_handoff_file" ]; then
    ssh "${ssh_common[@]}" -i "$GATEWAY_KEY" "$GATEWAY_HOST" \
      "rm -f -- '$gateway_handoff_file'" >/dev/null 2>&1 || true
  fi
  if [ -n "$paired_gateway_handoff_file" ]; then
    ssh "${ssh_common[@]}" -i "$GATEWAY_KEY" "$GATEWAY_HOST" \
      "rm -f -- '$paired_gateway_handoff_file'" >/dev/null 2>&1 || true
  fi
  if [ -n "${validator_initial_requirements_remote:-}" ]; then
    ssh "${ssh_common[@]}" -i "$VALIDATOR_KEY" "$VALIDATOR_HOST" \
      "rm -f -- '$validator_initial_requirements_remote' '$validator_final_requirements_remote' '$validator_final_requirements_remote.tmp' '$validator_final_lineage_remote' '$validator_final_lineage_remote.tmp' '$validator_recovery_requirements_remote' '$validator_recovery_requirements_remote.tmp' '$validator_recovery_lineage_remote' '$validator_recovery_lineage_remote.tmp'" \
      >/dev/null 2>&1 || true
  fi
  if [ -n "${gateway_validator_requirements_remote:-}" ]; then
    ssh "${ssh_common[@]}" -i "$GATEWAY_KEY" "$GATEWAY_HOST" \
      "rm -f -- '$gateway_validator_requirements_remote' '$gateway_validator_requirements_remote.tmp' '$gateway_counterpart_lineage_remote' '$gateway_counterpart_lineage_remote.tmp'" \
      >/dev/null 2>&1 || true
  fi
  if [ -n "$gateway_validator_requirements_remote" ]; then
    ssh "${ssh_common[@]}" -i "$GATEWAY_KEY" "$GATEWAY_HOST" \
      "rm -f -- '$gateway_validator_requirements_remote' '$gateway_validator_requirements_remote.tmp'" \
      >/dev/null 2>&1 || true
  fi
  if [ -n "$validator_initial_requirements_remote" ] \
      || [ -n "$validator_final_requirements_remote" ] \
      || [ -n "$validator_final_lineage_remote" ]; then
    ssh "${ssh_common[@]}" -i "$VALIDATOR_KEY" "$VALIDATOR_HOST" \
      "rm -f -- '$validator_initial_requirements_remote' '$validator_final_requirements_remote' '$validator_final_requirements_remote.tmp' '$validator_final_lineage_remote' '$validator_final_lineage_remote.tmp' '$validator_recovery_requirements_remote' '$validator_recovery_requirements_remote.tmp' '$validator_recovery_lineage_remote' '$validator_recovery_lineage_remote.tmp'" \
      >/dev/null 2>&1 || true
  fi
  if [ -n "$temporary_root" ]; then
    rm -rf -- "$temporary_root"
  fi
  return "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

while [ "$#" -gt 0 ]; do
  case "$1" in
    --commit)
      if [ "$#" -lt 2 ]; then
        echo "ERROR: --commit requires a full 40-character SHA" >&2
        exit 2
      fi
      commit="$2"
      shift 2
      ;;
    --commit=*)
      commit="${1#--commit=}"
      shift
      ;;
    --component)
      if [ "$#" -lt 2 ]; then
        echo "ERROR: --component requires all, gateway, or validator" >&2
        exit 2
      fi
      component="$2"
      shift 2
      ;;
    --component=*)
      component="${1#--component=}"
      shift
      ;;
    --local-python)
      if [ "$#" -lt 2 ]; then
        echo "ERROR: --local-python requires an absolute venv executable" >&2
        exit 2
      fi
      local_python="$2"
      shift 2
      ;;
    --local-python=*)
      local_python="${1#--local-python=}"
      shift
      ;;
    --disable-miner-submissions-before-restart)
      disable_miner_submissions_before_restart=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unsupported attested restart argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! [[ "$commit" =~ ^[0-9a-f]{40}$ ]]; then
  echo "ERROR: --commit must be a lowercase full 40-character SHA" >&2
  exit 2
fi
case "$component" in
  all|gateway|validator) ;;
  *)
    echo "ERROR: --component must be all, gateway, or validator" >&2
    exit 2
    ;;
esac
if [ -z "$local_python" ] \
    || [[ "$local_python" != /* ]] \
    || [[ "$local_python" =~ [[:cntrl:]] ]]; then
  echo "ERROR: --local-python requires one absolute venv/bin/python path" >&2
  exit 2
fi
if [ "$disable_miner_submissions_before_restart" = "1" ] \
    && [ "$component" != "all" ]; then
  echo "ERROR: --disable-miner-submissions-before-restart requires --component all" >&2
  exit 2
fi
if [ "$disable_miner_submissions_before_restart" = "1" ] \
    && [ -n "$GATEWAY_ENV_SECRET_ID" ] \
    && [ "$GATEWAY_ENV_SECRET_ID" != "leadpoet/prod/gateway/env" ]; then
  echo "ERROR: miner-maintenance bootstrap requires the fixed production gateway secret" >&2
  exit 2
fi
if [ "$disable_miner_submissions_before_restart" = "1" ]; then
  if [ "$GATEWAY_HOST" != "$PRODUCTION_GATEWAY_HOST" ] \
      || [ "$GATEWAY_RESTART" != "$PRODUCTION_GATEWAY_RESTART" ] \
      || [ "$GATEWAY_REPO_ROOT" != "$PRODUCTION_GATEWAY_REPO_ROOT" ] \
      || [ "$GATEWAY_PYTHON_BIN" != "$PRODUCTION_GATEWAY_PYTHON_BIN" ] \
      || [ "$VALIDATOR_PYTHON_BIN" != "$PRODUCTION_VALIDATOR_PYTHON_BIN" ] \
      || [ "$GATEWAY_DEPLOY_READINESS_PATH" != "$PRODUCTION_GATEWAY_DEPLOY_READINESS_PATH" ]; then
    echo "ERROR: miner-maintenance bootstrap requires the fixed production gateway topology" >&2
    exit 2
  fi
  if [ -n "${GATEWAY_RESTART_CONTROLLER_ROOT:-}" ] \
      && [ "$GATEWAY_RESTART_CONTROLLER_ROOT" != "$PRODUCTION_GATEWAY_RESTART_CONTROLLER_ROOT" ]; then
    echo "ERROR: miner-maintenance bootstrap requires the fixed production gateway topology" >&2
    exit 2
  fi
  if [ -n "${GATEWAY_RESTART_CONTROLLER_CURRENT:-}" ] \
      && [ "$GATEWAY_RESTART_CONTROLLER_CURRENT" != "$PRODUCTION_GATEWAY_RESTART_CONTROLLER_CURRENT" ]; then
    echo "ERROR: miner-maintenance bootstrap requires the fixed production gateway topology" >&2
    exit 2
  fi
fi
for key in "$GATEWAY_KEY" "$VALIDATOR_KEY"; do
  if [ ! -r "$key" ]; then
    echo "ERROR: SSH key is unavailable: $key" >&2
    exit 1
  fi
done
for secret_id in "$GATEWAY_ENV_SECRET_ID" "$VALIDATOR_ENV_SECRET_ID"; do
  if [ -n "$secret_id" ] \
      && ! [[ "$secret_id" =~ ^[A-Za-z0-9/_+=.@-]+$ ]]; then
    echo "ERROR: environment secret id contains unsupported characters" >&2
    exit 2
  fi
done
for remote_path in \
  "$GATEWAY_REPO_ROOT" \
  "$GATEWAY_PYTHON_BIN" \
  "$VALIDATOR_PYTHON_BIN" \
  "$GATEWAY_DEPLOY_READINESS_PATH" \
  "$GATEWAY_ACTIVE_RELEASE_REQUIREMENTS_PATH" \
  "$GATEWAY_ACTIVE_RELEASE_LINEAGE_PATH"; do
  if ! [[ "$remote_path" =~ ^/[A-Za-z0-9._/-]+$ ]]; then
    echo "ERROR: readiness authority path contains unsupported characters" >&2
    exit 2
  fi
done
case "$RELEASE_PREFIX" in
  attested-v2/releases|attested-v2/candidates) ;;
  *)
    echo "ERROR: release prefix is outside the reviewed channels" >&2
    exit 2
    ;;
esac
if [ "$component" != "validator" ]; then
  unsafe_git_environment=(
    GIT_ALTERNATE_OBJECT_DIRECTORIES GIT_CEILING_DIRECTORIES GIT_COMMON_DIR
    GIT_CONFIG GIT_CONFIG_COUNT GIT_CONFIG_GLOBAL GIT_CONFIG_PARAMETERS
    GIT_CONFIG_SYSTEM GIT_DIR GIT_INDEX_FILE GIT_OBJECT_DIRECTORY
    GIT_REPLACE_REF_BASE GIT_WORK_TREE
  )
  for git_environment_name in "${unsafe_git_environment[@]}"; do
    if [ -n "${!git_environment_name:-}" ]; then
      echo "ERROR: miner-maintenance Git authority contains environment overrides" >&2
      exit 1
    fi
  done
  export GIT_CONFIG_NOSYSTEM=1
  export GIT_NO_REPLACE_OBJECTS=1
  if [ -n "$(git -C "$ROOT" for-each-ref --format='%(refname)' refs/replace)" ]; then
    echo "ERROR: miner-maintenance Git authority contains replacement refs" >&2
    exit 1
  fi
  for git_authority_path in info/grafts objects/info/alternates; do
    resolved_git_authority_path="$(
      git -C "$ROOT" rev-parse --git-path "$git_authority_path"
    )"
    if [[ "$resolved_git_authority_path" != /* ]]; then
      resolved_git_authority_path="$ROOT/$resolved_git_authority_path"
    fi
    if [ -e "$resolved_git_authority_path" ] \
        && { [ ! -f "$resolved_git_authority_path" ] \
          || [ -L "$resolved_git_authority_path" ] \
          || [ -s "$resolved_git_authority_path" ]; }; then
      echo "ERROR: miner-maintenance Git authority contains graft or alternate objects" >&2
      exit 1
    fi
  done
fi

if [ ! -f "$local_python" ] || [ ! -x "$local_python" ]; then
  echo "ERROR: local readiness Python is not an executable regular target: $local_python" >&2
  exit 1
fi

path_identity() {
  local path="$1"
  if stat -f '%d:%i' -- "$path" >/dev/null 2>&1; then
    stat -f '%d:%i' -- "$path"
  else
    stat -c '%d:%i' -- "$path"
  fi
}

local_python_link_identity="$(path_identity "$local_python")"
if ! local_python_target="$(
  "$local_python" -I -S -c \
    'import os,sys; print(os.path.realpath(sys.argv[1]))' \
    "$local_python"
)"; then
  echo "ERROR: local readiness Python target resolution failed" >&2
  exit 1
fi
if [ -z "$local_python_target" ] \
    || [[ "$local_python_target" != /* ]] \
    || [[ "$local_python_target" =~ [[:cntrl:]] ]] \
    || [ ! -f "$local_python_target" ] \
    || [ ! -x "$local_python_target" ]; then
  echo "ERROR: local readiness Python resolved target is invalid" >&2
  exit 1
fi
local_python_target_identity="$(path_identity "$local_python_target")"
local_python_bin_dir="$(dirname -- "$local_python")"
local_python_name="$(basename -- "$local_python")"
if [ "$(basename -- "$local_python_bin_dir")" != "bin" ] \
    || ! [[ "$local_python_name" =~ ^python([0-9]+([.][0-9]+)*)?$ ]]; then
  echo "ERROR: local readiness Python must be a venv bin/python executable" >&2
  exit 2
fi
local_python_venv_root="$(cd "$local_python_bin_dir/.." && pwd -P)"
if [ ! -f "$local_python_venv_root/pyvenv.cfg" ] \
    || [ -L "$local_python_venv_root/pyvenv.cfg" ]; then
  echo "ERROR: local readiness Python is not bound to a regular pyvenv.cfg" >&2
  exit 1
fi
local_python_venv_config_identity="$(
  path_identity "$local_python_venv_root/pyvenv.cfg"
)"
if ! local_python_version="$(
  "$local_python" -I -S -c \
    'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'
)"; then
  echo "ERROR: local readiness Python version probe failed" >&2
  exit 1
fi
if ! [[ "$local_python_version" =~ ^[0-9]+[.][0-9]+$ ]]; then
  echo "ERROR: local readiness Python returned an invalid version" >&2
  exit 1
fi
local_python_site_packages="$local_python_venv_root/lib/python$local_python_version/site-packages"
if [ ! -d "$local_python_site_packages" ] \
    || [ -L "$local_python_site_packages" ]; then
  echo "ERROR: local readiness Python site-packages is unavailable or indirect" >&2
  exit 1
fi
local_python_site_identity="$(path_identity "$local_python_site_packages")"

verify_local_readiness_python_binding() {
  if [ "$(path_identity "$local_python")" != "$local_python_link_identity" ] \
      || [ "$(path_identity "$local_python_target")" != "$local_python_target_identity" ] \
      || [ "$(path_identity "$local_python_venv_root/pyvenv.cfg")" != "$local_python_venv_config_identity" ] \
      || [ "$(path_identity "$local_python_site_packages")" != "$local_python_site_identity" ]; then
    echo "ERROR: local readiness Python binding changed after preflight" >&2
    return 1
  fi
}

LOCAL_READINESS_CANDIDATE_PATHS=(
  scripts/restart_attested_release_local.sh
  gateway/__init__.py
  gateway/build_info.py
  gateway/deploy_readiness.py
  gateway/tee/release_channel_v2.py
  gateway/tee/active_release_requirements_v2.py
  gateway/tee/release_lineage_v2.py
  gateway/tee/release_manifest_v2.py
  gateway/tee/topology.py
  leadpoet_canonical/__init__.py
  leadpoet_canonical/attested_v2.py
  leadpoet_canonical/constants.py
  leadpoet_canonical/nitro.py
  leadpoet_observability/__init__.py
  leadpoet_observability/sentry_bootstrap.py
  leadpoet_observability/sentry_operations.py
  leadpoet_observability/sentry_scrubbing.py
  validator_tee/__init__.py
  validator_tee/host/__init__.py
  validator_tee/host/release_v2.py
  validator_tee/host/vsock_client.py
)

verify_local_readiness_candidate_sources() {
  local path=""
  local expected_blob=""
  local observed_blob=""
  if [ "$local_readiness_candidate_bound" != "1" ]; then
    return 0
  fi
  for path in "${LOCAL_READINESS_CANDIDATE_PATHS[@]}"; do
    if [ ! -f "$ROOT/$path" ] || [ -L "$ROOT/$path" ]; then
      echo "ERROR: local readiness candidate source is unavailable: $path" >&2
      return 1
    fi
    if ! expected_blob="$(git -C "$ROOT" rev-parse "$branch_commit:$path")"; then
      echo "ERROR: local readiness candidate Git blob is unavailable: $path" >&2
      return 1
    fi
    if ! observed_blob="$(
      git -C "$ROOT" hash-object --no-filters "$ROOT/$path"
    )"; then
      echo "ERROR: local readiness candidate source could not be hashed: $path" >&2
      return 1
    fi
    if [ "$observed_blob" != "$expected_blob" ]; then
      echo "ERROR: local readiness candidate source differs from exact commit: $path" >&2
      return 1
    fi
  done
}

run_local_readiness_python() {
  verify_local_readiness_python_binding || return 1
  verify_local_readiness_candidate_sources || return 1
  "$local_python_target" -I -S -B -c '
from pathlib import Path
import sys

root = Path(sys.argv.pop(1)).resolve(strict=True)
site_packages = Path(sys.argv.pop(1)).resolve(strict=True)
sys.path.insert(0, str(root))
sys.path.append(str(site_packages))
source = sys.stdin.read()
exec(
    compile(source, "<leadpoet-local-readiness>", "exec"),
    {"__name__": "__main__", "__builtins__": __builtins__},
)
' "$ROOT" "$local_python_site_packages" "$@"
}

preflight_local_readiness_python() {
  local local_python_preflight=""
  if ! local_python_preflight="$(
    run_local_readiness_python \
      "$local_python" "$local_python_venv_root" "$local_python_target" \
      "$local_python_site_packages" <<'PY'
import importlib.metadata
import json
import os
from pathlib import Path
import stat
import sys

if (
    sys.flags.isolated != 1
    or sys.flags.no_site != 1
    or sys.flags.no_user_site != 1
    or "site" in sys.modules
    or "sitecustomize" in sys.modules
):
    raise SystemExit("local readiness Python is not isolated from site configuration")

selected = Path(sys.argv[1])
venv_root = Path(sys.argv[2]).resolve(strict=True)
bound_target = Path(sys.argv[3]).resolve(strict=True)
site_packages = Path(sys.argv[4]).resolve(strict=True)
if not selected.is_absolute() or selected.parent.name != "bin":
    raise SystemExit("local readiness Python path is not an absolute venv executable")
if selected.parent.parent.resolve(strict=True) != venv_root:
    raise SystemExit("local readiness Python path differs from its venv root")
if site_packages != (
    venv_root
    / "lib"
    / ("python%d.%d" % sys.version_info[:2])
    / "site-packages"
).resolve(strict=True):
    raise SystemExit("local readiness Python site-packages identity differs")
if Path(sys.path[-1]).resolve(strict=True) != site_packages or any(
    Path(entry).resolve() == site_packages for entry in sys.path[:-1]
):
    raise SystemExit("local readiness Python site-packages precedes stdlib")

allowed_owners = {0, os.getuid()}


def require_safe_stat(path, *, follow=True, allow_symlink_mode=False):
    details = path.stat() if follow else path.lstat()
    if details.st_uid not in allowed_owners:
        raise SystemExit("local readiness Python path has an untrusted owner")
    if not (allow_symlink_mode and stat.S_ISLNK(details.st_mode)) \
            and details.st_mode & 0o022:
        raise SystemExit("local readiness Python path is group/world writable")
    return details


def require_safe_venv_origin(path):
    current = Path(path).resolve(strict=True)
    try:
        current.relative_to(venv_root)
    except ValueError as exc:
        raise SystemExit("local readiness dependency is outside the selected venv") from exc
    while True:
        require_safe_stat(current)
        if current == venv_root:
            break
        current = current.parent


selected_link = require_safe_stat(
    selected,
    follow=False,
    allow_symlink_mode=True,
)
selected_target_path = selected.resolve(strict=True)
if selected_target_path != bound_target:
    raise SystemExit("local readiness Python resolved target differs")
selected_target = require_safe_stat(selected_target_path)
if not stat.S_ISREG(selected_target.st_mode) or not os.access(selected, os.X_OK):
    raise SystemExit("local readiness Python target is not executable and regular")
require_safe_venv_origin(venv_root / "pyvenv.cfg")
require_safe_venv_origin(site_packages)

import cbor2
import cryptography
from cryptography import x509
from cryptography.hazmat.primitives.asymmetric import ec
import gateway.deploy_readiness as deploy_readiness
import gateway.tee.release_channel_v2 as release_channel_v2
import gateway.tee.release_lineage_v2 as release_lineage_v2
import gateway.tee.release_manifest_v2 as release_manifest_v2
import leadpoet_canonical.attested_v2 as attested_v2
import leadpoet_canonical.nitro as nitro
import validator_tee
import validator_tee.host.release_v2 as validator_release_v2

del x509, ec
if any(
    name == "bittensor_wallet" or name.startswith("bittensor_wallet.")
    for name in sys.modules
):
    raise SystemExit("pure readiness imports loaded the validator wallet dependency")
for module in (
    deploy_readiness,
    release_channel_v2,
    release_lineage_v2,
    release_manifest_v2,
    attested_v2,
    nitro,
    validator_tee,
    validator_release_v2,
):
    origin = Path(module.__file__).resolve(strict=True)
    try:
        origin.relative_to(Path(sys.path[0]).resolve(strict=True))
    except ValueError as exc:
        raise SystemExit("local readiness candidate module escaped the exact root") from exc
for dependency in (cbor2, cryptography):
    require_safe_venv_origin(Path(dependency.__file__))
for function in (
    deploy_readiness.build_deploy_readiness_transition_marker,
    deploy_readiness.build_gateway_v2_readiness_evidence_from_observation,
    deploy_readiness.build_validator_v2_readiness_evidence_from_observation,
    deploy_readiness.build_v2_deploy_readiness_manifest,
    deploy_readiness.validate_v2_deploy_readiness_manifest,
    release_channel_v2.validate_release_channel_v2,
    release_channel_v2.validate_historical_release_channel_v2,
    release_lineage_v2.validate_historical_compact_release_lineage_v2,
    release_manifest_v2.validate_historical_release_manifest,
    validator_release_v2.validate_validator_release_manifest,
):
    if not callable(function):
        raise SystemExit("local readiness candidate function is unavailable")
available, diagnostic = nitro.verify_nitro_attestation_full(
    attestation_b64="!",
    expected_pcr0="1" * 96,
)
if available or "Missing required library" in str(diagnostic.get("error") or ""):
    raise SystemExit("local readiness Nitro dependencies failed their execution probe")
if "site" in sys.modules or "sitecustomize" in sys.modules:
    raise SystemExit("local readiness imports activated mutable site configuration")

print(
    json.dumps(
        {
            "schema_version": "leadpoet.local_readiness_python.v1",
            "executable": str(selected),
            "executable_link_device": selected_link.st_dev,
            "executable_link_inode": selected_link.st_ino,
            "executable_target": str(selected_target_path),
            "executable_target_device": selected_target.st_dev,
            "executable_target_inode": selected_target.st_ino,
            "python_prefix": sys.prefix,
            "python_version": "%d.%d.%d" % sys.version_info[:3],
            "venv_root": str(venv_root),
            "cbor2_version": importlib.metadata.version("cbor2"),
            "cryptography_version": importlib.metadata.version("cryptography"),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
)
PY
)"; then
    echo "ERROR: local readiness Python preflight failed before production mutation" >&2
    return 1
  fi
  echo "Local readiness Python preflight: $local_python_preflight"
}

temporary_root="$(mktemp -d /tmp/leadpoet-attested-restart.XXXXXX)"
helper="$temporary_root/exact_commit_restart_v2.py"
gateway_observation="$temporary_root/gateway-readiness-observation.json"
gateway_evidence="$temporary_root/gateway-readiness-evidence.json"
validator_observation="$temporary_root/validator-readiness-observation.json"
validator_evidence="$temporary_root/validator-readiness-evidence.json"
transition_manifest="$temporary_root/deploy-readiness-transition.json"
final_manifest="$temporary_root/deploy-readiness-v2.json"
validator_initial_requirements_local="$temporary_root/validator-active-release-requirements.json"
gateway_final_requirements_local="$temporary_root/gateway-active-release-requirements.json"
gateway_final_lineage_local="$temporary_root/gateway-active-release-lineage.json"
validator_counterpart_lineage_local="$temporary_root/validator-counterpart-release-lineage.json"
restart_transfer_id="$(basename "$temporary_root")"
active_release_restart_invocation_id="${LEADPOET_ACTIVE_RELEASE_RESTART_INVOCATION_ID:-restart-$(python3 -c 'import secrets; print(secrets.token_hex(24))')}"
if ! [[ "$active_release_restart_invocation_id" =~ ^[a-z0-9][a-z0-9_.:-]{0,127}$ ]]; then
  echo "ERROR: active release restart invocation identity generation failed" >&2
  exit 1
fi

echo "Fetching current public V2 compatibility authority"
git -C "$ROOT" fetch origin main
if [ "$component" != "validator" ]; then
  if [ -n "$(git -C "$ROOT" for-each-ref --format='%(refname)' refs/replace)" ]; then
    echo "ERROR: miner-maintenance Git authority changed to include replacement refs" >&2
    exit 1
  fi
  for git_authority_path in info/grafts objects/info/alternates; do
    resolved_git_authority_path="$(
      git -C "$ROOT" rev-parse --git-path "$git_authority_path"
    )"
    if [[ "$resolved_git_authority_path" != /* ]]; then
      resolved_git_authority_path="$ROOT/$resolved_git_authority_path"
    fi
    if [ -e "$resolved_git_authority_path" ] \
        && { [ ! -f "$resolved_git_authority_path" ] \
          || [ -L "$resolved_git_authority_path" ] \
          || [ -s "$resolved_git_authority_path" ]; }; then
      echo "ERROR: miner-maintenance Git authority changed to include graft or alternate objects" >&2
      exit 1
    fi
  done
fi
branch_commit="$(git -C "$ROOT" rev-parse --verify origin/main^{commit})"
git -C "$ROOT" cat-file -e "$commit^{commit}"
git -C "$ROOT" show \
  origin/main:Leadpoet/utils/exact_commit_restart_v2.py > "$helper"
python3 "$helper" \
  --repo-root "$ROOT" \
  --selected-commit "$commit" \
  --branch-ref origin/main
candidate_topology_entry="$(
  git -C "$ROOT" ls-tree "$commit" -- gateway/tee/topology.json
)" || {
  echo "ERROR: selected release topology Git identity is unavailable" >&2
  exit 1
}
if ! [[ "$candidate_topology_entry" =~ ^100644\ blob\ [0-9a-f]{40}$'\t'gateway/tee/topology.json$ ]]; then
  echo "ERROR: selected release topology is not one regular Git blob" >&2
  exit 1
fi
candidate_topology_blob="${candidate_topology_entry#100644 blob }"
candidate_topology_blob="${candidate_topology_blob%%$'\t'*}"
if [ "$candidate_topology_blob" = "$HISTORICAL_THREE_ROLE_TOPOLOGY_BLOB" ] \
    && ! git -C "$ROOT" cat-file -e \
      "$commit:gateway/tee/build_local_release_v2.sh" 2>/dev/null \
    && ! git -C "$ROOT" cat-file -e \
      "$commit:gateway/tee/local_release_v2.py" 2>/dev/null; then
  historical_topology_hash="$HISTORICAL_THREE_ROLE_TOPOLOGY_HASH"
  echo "Selected exact historical three-role gateway topology"
fi
echo "Selected release is compatible with current public auditors: $commit"
echo "Current public V2 authority commit: $branch_commit"
local_readiness_candidate_bound=1
preflight_local_readiness_python
validator_initial_requirements_remote="/tmp/leadpoet-validator-active-release-requirements.$restart_transfer_id.json"
gateway_validator_requirements_remote="/tmp/leadpoet-validator-active-release-requirements.$restart_transfer_id.json"
validator_final_requirements_remote="/tmp/leadpoet-gateway-active-release-requirements.$restart_transfer_id.json"
validator_final_lineage_remote="/tmp/leadpoet-gateway-active-release-lineage.$restart_transfer_id.json"
validator_recovery_requirements_remote="/tmp/leadpoet-validator-recovery-requirements.$restart_transfer_id.json"
validator_recovery_lineage_remote="/tmp/leadpoet-validator-recovery-lineage.$restart_transfer_id.json"
gateway_counterpart_lineage_remote="/tmp/leadpoet-validator-counterpart-release-lineage.$restart_transfer_id.json"
ssh_common=(
  -n
  -o BatchMode=yes
  -o ConnectTimeout=15
  -o ServerAliveInterval=30
  -o ServerAliveCountMax=20
)
scp_common=(
  -q
  -o BatchMode=yes
  -o ConnectTimeout=15
  -o ServerAliveInterval=30
  -o ServerAliveCountMax=20
)
selected_operator_blob="$(
  git -C "$ROOT" rev-parse \
    "$branch_commit:scripts/restart_attested_release_local.sh"
)"
local_operator_blob="$(
  git -C "$ROOT" hash-object --no-filters \
    "$ROOT/scripts/restart_attested_release_local.sh"
)"
if [ "$local_operator_blob" != "$selected_operator_blob" ]; then
  echo "ERROR: restart operator is not the exact frozen authority Git blob" >&2
  exit 1
fi
if [ "$component" != "validator" ]; then
  selected_controller_verifier_blob="$(
    git -C "$ROOT" rev-parse \
      "$branch_commit:scripts/verify_installed_gateway_controller_v1.py"
  )"
  local_controller_verifier_blob="$(
    git -C "$ROOT" hash-object --no-filters \
      "$ROOT/scripts/verify_installed_gateway_controller_v1.py"
  )"
  if [ "$local_controller_verifier_blob" != "$selected_controller_verifier_blob" ]; then
    echo "ERROR: installed-controller verifier is not the exact candidate Git blob" >&2
    exit 1
  fi
  installed_controller_target="$(
    ssh "${ssh_common[@]}" -i "$GATEWAY_KEY" "$GATEWAY_HOST" \
      "readlink -- '$PRODUCTION_GATEWAY_RESTART_CONTROLLER_CURRENT'"
  )"
  if [[ "$installed_controller_target" =~ ^releases/([0-9a-f]{40})$ ]]; then
    expected_controller_commit="${BASH_REMATCH[1]}"
  else
    echo "ERROR: installed gateway controller target is invalid" >&2
    exit 1
  fi
  python3 "$ROOT/scripts/verify_installed_gateway_controller_v1.py" \
    --repo-root "$ROOT" \
    --expected-controller-commit "$expected_controller_commit" \
    --expected-commit "$branch_commit" \
    --verify-lineage-only
  echo "Candidate-bound installed gateway controller: $expected_controller_commit"
  controller_verifier_b64="$(
    base64 < "$ROOT/scripts/verify_installed_gateway_controller_v1.py" \
      | tr -d '\n'
  )"
  if [ -z "$controller_verifier_b64" ]; then
    echo "ERROR: installed-controller verifier could not be encoded" >&2
    exit 1
  fi
fi
if [ "$disable_miner_submissions_before_restart" = "1" ]; then
  gateway_handoff_nonce="$(python3 -c 'import secrets; print(secrets.token_hex(32))')"
  if ! [[ "$gateway_handoff_nonce" =~ ^[0-9a-f]{64}$ ]]; then
    echo "ERROR: miner-maintenance handoff nonce generation failed" >&2
    exit 1
  fi
fi
if [ "$component" = "all" ]; then
  paired_gateway_handoff_nonce="$(python3 -c 'import secrets; print(secrets.token_hex(32))')"
  if ! [[ "$paired_gateway_handoff_nonce" =~ ^[0-9a-f]{64}$ ]]; then
    echo "ERROR: paired gateway handoff nonce generation failed" >&2
    exit 1
  fi
fi

install_gateway_readiness_manifest() {
  local action="$1"
  local source="$2"
  local payload=""
  case "$action" in
    readiness-transition|readiness-final) ;;
    *)
      echo "ERROR: unsupported deploy readiness install action" >&2
      return 2
      ;;
  esac
  payload="$(base64 < "$source" | tr -d '\n')"
  ssh "${ssh_common[@]}" -i "$GATEWAY_KEY" "$GATEWAY_HOST" \
    "python3 - '$GATEWAY_DEPLOY_READINESS_PATH' '$payload' '$action' <<'PY'
import base64
import json
import os
from pathlib import Path
import sys
import tempfile

destination = Path(sys.argv[1])
document = json.loads(base64.b64decode(sys.argv[2], validate=True))
action = sys.argv[3]
if action not in {'readiness-transition', 'readiness-final'}:
    raise SystemExit('deploy readiness install action is invalid')
if not isinstance(document, dict) or document.get('enforce_resume_block') is not True:
    raise SystemExit('deploy readiness payload is not fail closed')
if action == 'readiness-transition' and document.get('ok') is not False:
    raise SystemExit('deploy readiness transition payload is invalid')
if action == 'readiness-final' and (
    document.get('schema_version') != 'leadpoet.deploy_readiness.v2'
    or document.get('ok') is not True
):
    raise SystemExit('deploy readiness final payload is invalid')
destination.parent.mkdir(parents=True, exist_ok=True)
descriptor, temporary_name = tempfile.mkstemp(
    prefix='.' + destination.name + '.',
    dir=str(destination.parent),
)
temporary = Path(temporary_name)
try:
    with os.fdopen(descriptor, 'wb') as handle:
        handle.write(
            (json.dumps(document, sort_keys=True, separators=(',', ':')) + '\n').encode('ascii')
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary, 0o600)
    os.replace(temporary, destination)
finally:
    temporary.unlink(missing_ok=True)
PY"
}

invalidate_deploy_readiness() {
  run_local_readiness_python "$commit" "$transition_manifest" <<'PY'
import sys
from gateway.deploy_readiness import (
    build_deploy_readiness_transition_marker,
    write_deploy_readiness_manifest,
)

write_deploy_readiness_manifest(
    build_deploy_readiness_transition_marker(expected_commit=sys.argv[1]),
    sys.argv[2],
)
PY
  install_gateway_readiness_manifest readiness-transition "$transition_manifest"
  echo "Installed fail-closed deploy readiness transition marker for $commit"
}

coordination_remote_command() {
  local value="$1"
  case "$value" in
    "$commit"|"failed:$commit") ;;
    *)
      echo "ERROR: invalid coordinated restart marker value" >&2
      return 2
      ;;
  esac
  printf '%s\n' \
    "set -Eeuo pipefail
     umask 077
     marker='$coordination_file.tmp'
     printf '%s\\n' '$value' > \"\$marker\"
     mv -f -- \"\$marker\" '$coordination_file'"
}

publish_coordination_value() {
  local remote_command
  remote_command="$(coordination_remote_command "$1")"
  ssh "${ssh_common[@]}" -i "$VALIDATOR_KEY" "$VALIDATOR_HOST" \
    "$remote_command"
}

validate_validator_initial_release_requirements() {
  run_local_readiness_python \
    "$commit" "$branch_commit" "$active_release_restart_invocation_id" \
    "$validator_initial_requirements_local" <<'PY'
import json
import os
import stat
import sys

from gateway.tee.active_release_requirements_v2 import (
    validate_active_release_requirements_v2,
)

expected_commit = sys.argv[1]
expected_authority = sys.argv[2]
expected_invocation = sys.argv[3]
max_document_bytes = 4 * 1024 * 1024
try:
    descriptor = os.open(
        sys.argv[4],
        os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
    )
except OSError as exc:
    raise SystemExit(
        f"validator active release requirements cannot be opened securely: {exc}"
    ) from exc
try:
    metadata = os.fstat(descriptor)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_size <= 0
        or metadata.st_size > max_document_bytes
    ):
        raise SystemExit(
            "validator active release requirements are not a bounded regular file"
        )
    raw = os.read(descriptor, max_document_bytes + 1)
    if len(raw) != metadata.st_size or len(raw) > max_document_bytes:
        raise SystemExit(
            "validator active release requirements changed during its bounded read"
        )
finally:
    os.close(descriptor)
value = validate_active_release_requirements_v2(json.loads(raw))
if value["candidate_commit_sha"] != expected_commit:
    raise SystemExit("validator active release requirements candidate differs")
if value["authority_commit_sha"] != expected_authority:
    raise SystemExit("validator active release requirements authority differs")
if value["restart_invocation_id"] != expected_invocation:
    raise SystemExit("validator active release requirements invocation differs")
if value["commits_by_root"]:
    raise SystemExit("validator initial release requirements unexpectedly contain gateway roots")
print(value["selection_hash"])
PY
}

validate_gateway_final_release_authority() {
  run_local_readiness_python \
    "$commit" "$branch_commit" "$active_release_restart_invocation_id" \
    "$historical_topology_hash" \
    "$validator_initial_requirements_local" \
    "$gateway_final_requirements_local" \
    "$gateway_final_lineage_local" <<'PY'
import json
import os
import stat
import sys

from gateway.tee.active_release_requirements_v2 import (
    validate_active_release_requirements_v2,
)
from gateway.tee.release_lineage_v2 import (
    validate_compact_release_lineage_v2,
    validate_historical_compact_release_lineage_v2,
)

expected_commit = sys.argv[1]
expected_authority = sys.argv[2]
expected_invocation = sys.argv[3]
historical_topology_hash = sys.argv[4] or None
max_document_bytes = 4 * 1024 * 1024
documents = []
for raw_path in sys.argv[5:]:
    try:
        descriptor = os.open(
            raw_path,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
        )
    except OSError as exc:
        raise SystemExit(
            f"paired active release authority cannot be opened securely: {exc}"
        ) from exc
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size <= 0
            or metadata.st_size > max_document_bytes
        ):
            raise SystemExit(
                "paired active release authority is not a bounded regular file"
            )
        raw = os.read(descriptor, max_document_bytes + 1)
        if len(raw) != metadata.st_size or len(raw) > max_document_bytes:
            raise SystemExit(
                "paired active release authority changed during its bounded read"
            )
    finally:
        os.close(descriptor)
    documents.append(json.loads(raw))
initial = validate_active_release_requirements_v2(documents[0])
final = validate_active_release_requirements_v2(documents[1])
if historical_topology_hash is None:
    lineage = validate_compact_release_lineage_v2(
        documents[2],
        expected_current_commit=expected_commit,
    )
else:
    lineage = validate_historical_compact_release_lineage_v2(
        documents[2],
        expected_topology_hash=historical_topology_hash,
        expected_current_commit=expected_commit,
    )
if initial["candidate_commit_sha"] != expected_commit \
        or final["candidate_commit_sha"] != expected_commit:
    raise SystemExit("paired active release authority candidate differs")
if initial["authority_commit_sha"] != expected_authority \
        or final["authority_commit_sha"] != expected_authority:
    raise SystemExit("paired active release authority controller differs")
if initial["restart_invocation_id"] != expected_invocation \
        or final["restart_invocation_id"] != expected_invocation:
    raise SystemExit("paired active release authority invocation differs")
if initial["ancestry_lineage_id"] != final["ancestry_lineage_id"]:
    raise SystemExit("paired active release authority lineage differs")
if not set(initial["required_commits"]).issubset(final["required_commits"]):
    raise SystemExit("gateway active release authority omits validator requirements")
if set(final["required_commits"]) != set(lineage["releases"]):
    raise SystemExit("gateway active release requirements differ from compact lineage")
print(final["selection_hash"] + " " + lineage["lineage_hash"])
PY
}

fetch_validator_initial_release_requirements() {
  rm -f -- "$validator_initial_requirements_local"
  scp "${scp_common[@]}" -i "$VALIDATOR_KEY" \
    "$VALIDATOR_HOST:$validator_initial_requirements_remote" \
    "$validator_initial_requirements_local"
  chmod 600 "$validator_initial_requirements_local"
  validate_validator_initial_release_requirements >/dev/null
  echo "Verified validator active release requirements for the paired restart"
}

prepare_running_validator_release_requirements() {
  ssh "${ssh_common[@]}" -i "$VALIDATOR_KEY" "$VALIDATOR_HOST" \
    "set -Eeuo pipefail
     umask 077
     cd '$VALIDATOR_REPO_ROOT'
     test \"\$(git rev-parse --verify HEAD)\" = '$commit'
     running_commit=\$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' leadpoet-validator-main | sed -n 's/^VALIDATOR_V2_DEPLOY_COMMIT=//p')
     test \"\$running_commit\" = '$commit'
     test -f '$VALIDATOR_STATEFUL_CUTOVER_MANIFEST'
     test ! -L '$VALIDATOR_STATEFUL_CUTOVER_MANIFEST'
     test -s '$VALIDATOR_STATEFUL_CUTOVER_MANIFEST'
     test -x '$VALIDATOR_PYTHON_BIN'
     lineage_id=\$(LEADPOET_SUBNET_EPOCH_CUTOVER_PATH='$VALIDATOR_STATEFUL_CUTOVER_MANIFEST' PYTHONPATH='$VALIDATOR_REPO_ROOT' '$VALIDATOR_PYTHON_BIN' -c 'from Leadpoet.utils.subnet_epoch import load_subnet_epoch_cutover; from leadpoet_canonical.ancestry_checkpoint_v2 import derive_ancestry_lineage_id_v2; cutover = load_subnet_epoch_cutover(); print(derive_ancestry_lineage_id_v2(cutover_mapping_hash=str(cutover.mapping_hash), network_genesis_hash=str(cutover.network_genesis_hash), netuid=int(cutover.netuid)))')
     sudo env PYTHONPATH='$VALIDATOR_REPO_ROOT' \
       AWS_REGION=us-east-1 AWS_DEFAULT_REGION=us-east-1 \
       LEADPOET_LOCAL_RELEASE_COMMIT_SHA='$commit' \
       LEADPOET_LOCAL_GATEWAY_RELEASE='$VALIDATOR_LOCAL_GATEWAY_RELEASE_PATH' \
       LEADPOET_LOCAL_VALIDATOR_RELEASE='$VALIDATOR_LOCAL_RELEASE_PATH' \
       LEADPOET_LOCAL_PRIOR_RELEASE_LINEAGE='$VALIDATOR_LOCAL_RELEASE_LINEAGE_PATH' \
       '$VALIDATOR_PYTHON_BIN' -m gateway.tee.prepare_active_release_lineage_v2 \
       --phase validator-initial \
       --candidate-commit '$commit' \
       --authority-commit '$branch_commit' \
       --restart-invocation-id '$active_release_restart_invocation_id' \
       --running-validator-commit \"\$running_commit\" \
       --journal '$VALIDATOR_REPO_ROOT/validator_weights/authoritative_weight_publication_v2.json' \
       --validator-hotkey-config '$VALIDATOR_V2_HOTKEY_CONFIG_PATH' \
       --chain-signing-profile '$VALIDATOR_CHAIN_SIGNING_PROFILE_PATH' \
       --repository '$VALIDATOR_REPO_ROOT' \
       --lineage-id \"\$lineage_id\" \
       --bucket leadpoet-attested-v2-artifacts-493765492819 \
       --prefix '$RELEASE_PREFIX' \
       --requirements-output '$validator_initial_requirements_remote'
     sudo chown \"\$(id -u):\$(id -g)\" '$validator_initial_requirements_remote'
     chmod 600 '$validator_initial_requirements_remote'"
  fetch_validator_initial_release_requirements
}

install_gateway_validator_release_requirements() {
  scp "${scp_common[@]}" -i "$GATEWAY_KEY" \
    "$validator_initial_requirements_local" \
    "$GATEWAY_HOST:$gateway_validator_requirements_remote.tmp"
  ssh "${ssh_common[@]}" -i "$GATEWAY_KEY" "$GATEWAY_HOST" \
    "set -Eeuo pipefail
     umask 077
     test -s '$gateway_validator_requirements_remote.tmp'
     chmod 600 '$gateway_validator_requirements_remote.tmp'
     mv -f -- '$gateway_validator_requirements_remote.tmp' '$gateway_validator_requirements_remote'"
}

fetch_and_install_gateway_counterpart_lineage() {
  rm -f -- "$validator_counterpart_lineage_local"
  scp "${scp_common[@]}" -i "$VALIDATOR_KEY" \
    "$VALIDATOR_HOST:/home/ec2-user/.config/leadpoet/gateway-v2-release-lineage.json" \
    "$validator_counterpart_lineage_local"
  chmod 600 "$validator_counterpart_lineage_local"
  run_local_readiness_python \
    "$commit" "$historical_topology_hash" \
    "$validator_counterpart_lineage_local" <<'PY'
import json
import os
import stat
import sys

from gateway.tee.release_lineage_v2 import (
    validate_compact_release_lineage_v2,
    validate_historical_compact_release_lineage_v2,
)

max_document_bytes = 4 * 1024 * 1024


def load_bounded_json(path: str, label: str):
    try:
        fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
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


historical_topology_hash = sys.argv[2] or None
if historical_topology_hash is None:
    validate_compact_release_lineage_v2(
        load_bounded_json(sys.argv[3], "validator counterpart lineage"),
        expected_current_commit=sys.argv[1],
    )
else:
    validate_historical_compact_release_lineage_v2(
        load_bounded_json(sys.argv[3], "validator counterpart lineage"),
        expected_topology_hash=historical_topology_hash,
        expected_current_commit=sys.argv[1],
    )
PY
  scp "${scp_common[@]}" -i "$GATEWAY_KEY" \
    "$validator_counterpart_lineage_local" \
    "$GATEWAY_HOST:$gateway_counterpart_lineage_remote.tmp"
  ssh "${ssh_common[@]}" -i "$GATEWAY_KEY" "$GATEWAY_HOST" \
    "set -Eeuo pipefail
     umask 077
     test -s '$gateway_counterpart_lineage_remote.tmp'
     chmod 600 '$gateway_counterpart_lineage_remote.tmp'
     mv -f -- '$gateway_counterpart_lineage_remote.tmp' '$gateway_counterpart_lineage_remote'"
}

fetch_gateway_final_release_authority() {
  rm -f -- "$gateway_final_requirements_local" "$gateway_final_lineage_local"
  scp "${scp_common[@]}" -i "$GATEWAY_KEY" \
    "$GATEWAY_HOST:$GATEWAY_ACTIVE_RELEASE_REQUIREMENTS_PATH" \
    "$gateway_final_requirements_local"
  scp "${scp_common[@]}" -i "$GATEWAY_KEY" \
    "$GATEWAY_HOST:$GATEWAY_ACTIVE_RELEASE_LINEAGE_PATH" \
    "$gateway_final_lineage_local"
  chmod 600 "$gateway_final_requirements_local" "$gateway_final_lineage_local"
  validate_gateway_final_release_authority >/dev/null
  echo "Verified identical bounded active release authority from the restarted gateway"
}

bind_component_validator_to_gateway_release_authority() {
  rm -f -- "$gateway_final_requirements_local" "$gateway_final_lineage_local"
  scp "${scp_common[@]}" -i "$GATEWAY_KEY" \
    "$GATEWAY_HOST:$GATEWAY_ACTIVE_RELEASE_REQUIREMENTS_PATH" \
    "$gateway_final_requirements_local"
  scp "${scp_common[@]}" -i "$GATEWAY_KEY" \
    "$GATEWAY_HOST:$GATEWAY_ACTIVE_RELEASE_LINEAGE_PATH" \
    "$gateway_final_lineage_local"
  chmod 600 "$gateway_final_requirements_local" "$gateway_final_lineage_local"
  local active_release_binding=""
  active_release_binding="$(
    run_local_readiness_python \
      "$commit" "$branch_commit" "$historical_topology_hash" \
      "$gateway_final_requirements_local" \
      "$gateway_final_lineage_local" <<'PY'
import json
import os
import stat
import sys

from gateway.tee.active_release_requirements_v2 import (
    validate_active_release_requirements_v2,
)
from gateway.tee.release_lineage_v2 import (
    validate_compact_release_lineage_v2,
    validate_historical_compact_release_lineage_v2,
)

expected_commit, expected_authority = sys.argv[1:3]
historical_topology_hash = sys.argv[3] or None
max_document_bytes = 4 * 1024 * 1024


def load_bounded_json(path: str, label: str):
    try:
        fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
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
    load_bounded_json(sys.argv[4], "gateway active release requirements")
)
if historical_topology_hash is None:
    lineage = validate_compact_release_lineage_v2(
        load_bounded_json(sys.argv[5], "gateway active release lineage"),
        expected_current_commit=expected_commit,
    )
else:
    lineage = validate_historical_compact_release_lineage_v2(
        load_bounded_json(sys.argv[5], "gateway active release lineage"),
        expected_topology_hash=historical_topology_hash,
        expected_current_commit=expected_commit,
    )
if requirements["candidate_commit_sha"] != expected_commit:
    raise SystemExit("gateway active release candidate differs")
if set(requirements["required_commits"]) != set(lineage["releases"]):
    raise SystemExit("gateway active release requirements differ from lineage")
print(requirements["authority_commit_sha"])
print(requirements["restart_invocation_id"])
PY
  )"
  active_release_authority_commit="${active_release_binding%%$'\n'*}"
  active_release_restart_invocation_id="${active_release_binding##*$'\n'}"
  if ! [[ "$active_release_authority_commit" =~ ^[0-9a-f]{40}$ ]] \
      || ! git -C "$ROOT" cat-file -e \
        "$active_release_authority_commit^{commit}" \
      || ! git -C "$ROOT" merge-base --is-ancestor \
        "$commit" "$active_release_authority_commit" \
      || ! git -C "$ROOT" merge-base --is-ancestor \
        "$active_release_authority_commit" "$branch_commit"; then
    echo "ERROR: gateway active release authority is not covered by the selected controller" >&2
    return 1
  fi
  if ! [[ "$active_release_restart_invocation_id" =~ ^[a-z0-9][a-z0-9_.:-]{0,127}$ ]]; then
    echo "ERROR: gateway active release invocation identity is invalid" >&2
    return 1
  fi
  echo "Bound validator-only restart to the running gateway active release invocation"
}

install_validator_final_release_authority() {
  scp "${scp_common[@]}" -i "$VALIDATOR_KEY" \
    "$gateway_final_requirements_local" \
    "$VALIDATOR_HOST:$validator_final_requirements_remote.tmp"
  scp "${scp_common[@]}" -i "$VALIDATOR_KEY" \
    "$gateway_final_lineage_local" \
    "$VALIDATOR_HOST:$validator_final_lineage_remote.tmp"
  ssh "${ssh_common[@]}" -i "$VALIDATOR_KEY" "$VALIDATOR_HOST" \
    "set -Eeuo pipefail
     umask 077
     test -s '$validator_final_requirements_remote.tmp'
     test -s '$validator_final_lineage_remote.tmp'
     chmod 600 '$validator_final_requirements_remote.tmp' '$validator_final_lineage_remote.tmp'
     mv -f -- '$validator_final_requirements_remote.tmp' '$validator_final_requirements_remote'
     mv -f -- '$validator_final_lineage_remote.tmp' '$validator_final_lineage_remote'"
}

install_validator_missing_runtime_recovery_authority() {
  scp "${scp_common[@]}" -i "$VALIDATOR_KEY" \
    "$gateway_final_requirements_local" \
    "$VALIDATOR_HOST:$validator_recovery_requirements_remote.tmp"
  scp "${scp_common[@]}" -i "$VALIDATOR_KEY" \
    "$gateway_final_lineage_local" \
    "$VALIDATOR_HOST:$validator_recovery_lineage_remote.tmp"
  ssh "${ssh_common[@]}" -i "$VALIDATOR_KEY" "$VALIDATOR_HOST" \
    "set -Eeuo pipefail
     umask 077
     test -s '$validator_recovery_requirements_remote.tmp'
     test -s '$validator_recovery_lineage_remote.tmp'
     chmod 600 '$validator_recovery_requirements_remote.tmp' '$validator_recovery_lineage_remote.tmp'
     mv -f -- '$validator_recovery_requirements_remote.tmp' '$validator_recovery_requirements_remote'
     mv -f -- '$validator_recovery_lineage_remote.tmp' '$validator_recovery_lineage_remote'"
  echo "Installed exact gateway authority for missing validator runtime recovery"
}

report_restart_job_early_exit() {
  local label="$1"
  local process_id="$2"
  local status=0
  wait "$process_id" || status="$?"
  echo "ERROR: $label restart exited before coordinated completion (status $status)" >&2
}

run_validator_restart_against_active_gateway() {
  local validator_log="$temporary_root/validator-restart.log"
  (
    VALIDATOR_RESTART_EXEC_SSH=1 run_validator_restart
  ) > >(tee "$validator_log") 2>&1 &
  validator_job="$!"

  local requirements_ready=0
  local requirements_deadline=$((SECONDS + VALIDATOR_COORDINATION_TIMEOUT_SECONDS))
  while [ "$SECONDS" -lt "$requirements_deadline" ]; do
    if grep -Fq \
        "Prepared validator active release requirements sidecar" \
        "$validator_log"; then
      requirements_ready=1
      break
    fi
    if ! kill -0 "$validator_job" 2>/dev/null; then
      report_restart_job_early_exit validator "$validator_job"
      return 1
    fi
    sleep 1
  done
  if [ "$requirements_ready" != "1" ]; then
    echo "ERROR: validator did not prepare its active release requirements" >&2
    return 1
  fi
  fetch_validator_initial_release_requirements
  fetch_gateway_final_release_authority
  install_validator_final_release_authority
  local completion_deadline=$((SECONDS + VALIDATOR_COORDINATION_TIMEOUT_SECONDS))
  while kill -0 "$validator_job" 2>/dev/null; do
    if [ "$SECONDS" -ge "$completion_deadline" ]; then
      echo "ERROR: validator component restart exceeded the shared deadline" >&2
      return 1
    fi
    sleep 1
  done
  wait "$validator_job"
  validator_job=""
}

gateway_handoff_remote_command() {
  local value="$1"
  case "$value" in
    "$commit"|"failed:$commit") ;;
    *)
      echo "ERROR: invalid gateway miner-maintenance handoff value" >&2
      return 2
      ;;
  esac
  printf '%s\n' \
    "set -Eeuo pipefail
     umask 077
     marker='$gateway_handoff_file.tmp'
     printf '%s %s\\n' '$value' '$gateway_handoff_nonce' > \"\$marker\"
     chmod 600 \"\$marker\"
     mv -f -- \"\$marker\" '$gateway_handoff_file'"
}

publish_gateway_handoff_value() {
  local remote_command
  remote_command="$(gateway_handoff_remote_command "$1")"
  ssh "${ssh_common[@]}" -i "$GATEWAY_KEY" "$GATEWAY_HOST" \
    "$remote_command"
}

paired_gateway_handoff_remote_command() {
  local value="$1"
  case "$value" in
    "$commit"|"failed:$commit") ;;
    *)
      echo "ERROR: invalid paired gateway handoff value" >&2
      return 2
      ;;
  esac
  printf '%s\n' \
    "set -Eeuo pipefail
     umask 077
     marker='$paired_gateway_handoff_file.tmp'
     printf '%s %s\\n' '$value' '$paired_gateway_handoff_nonce' > \"\$marker\"
     chmod 600 \"\$marker\"
     mv -f -- \"\$marker\" '$paired_gateway_handoff_file'"
}

publish_paired_gateway_handoff_value() {
  local remote_command
  remote_command="$(paired_gateway_handoff_remote_command "$1")"
  ssh "${ssh_common[@]}" -i "$GATEWAY_KEY" "$GATEWAY_HOST" \
    "$remote_command"
}

gateway_active_commit() {
  ssh "${ssh_common[@]}" -i "$GATEWAY_KEY" "$GATEWAY_HOST" \
    "curl -fsS --connect-timeout 5 --max-time 15 \
      http://127.0.0.1:8000/build-info \
      | python3 -c 'import json,sys; print(json.load(sys.stdin)[\"git_commit\"])'"
}

validator_active_commit() {
  ssh "${ssh_common[@]}" -i "$VALIDATOR_KEY" "$VALIDATOR_HOST" \
    "set -Eeuo pipefail
     docker info >/dev/null
     inspect_output=\$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' \
       leadpoet-validator-main 2>&1) || {
       while [[ \"\$inspect_output\" == [[:space:]]* ]]; do
         inspect_output=\"\${inspect_output:1}\"
       done
       while [[ \"\$inspect_output\" == *[[:space:]] ]]; do
         inspect_output=\"\${inspect_output%?}\"
       done
       if [ \"\$inspect_output\" = 'Error: No such object: leadpoet-validator-main' ]; then
         exit 44
       fi
       printf '%s\n' \"\$inspect_output\" >&2
       exit 1
     }
     printf '%s\n' \"\$inspect_output\" \
       | sed -n 's/^VALIDATOR_V2_DEPLOY_COMMIT=//p'"
}

build_gateway_restart_command() {
  local bootstrap_command=""
  local bootstrap_command_b64=""
  local bootstrap_prefix="gateway-restart-controller-bootstrap"
  local gateway_restart_entrypoint_root="\$authority_root"
  local gateway_counterpart_path=""
  local gateway_secret_id="${GATEWAY_ENV_SECRET_ID:-leadpoet/prod/gateway/env}"
  local miner_bootstrap_arguments=""
  local miner_candidate_prepare=""
  local paired_required=0
  gateway_restart_command=()
  if [ "$component" = "all" ]; then
    paired_required=1
  elif [ "$component" = "gateway" ]; then
    gateway_counterpart_path="$gateway_counterpart_lineage_remote"
  fi
  if [ "$disable_miner_submissions_before_restart" = "1" ]; then
    bootstrap_prefix="gateway-miner-maintenance-bootstrap"
    gateway_restart_entrypoint_root="\$candidate_root"
    miner_bootstrap_arguments=" \\
          --miner-maintenance-bootstrap-plan \"\$bootstrap_root/plan.json\" \\
          --miner-maintenance-bootstrap-root \"\$bootstrap_root\" \\
          --miner-maintenance-handoff-file '$gateway_handoff_file' \\
          --miner-maintenance-handoff-nonce '$gateway_handoff_nonce'"
    miner_candidate_prepare="
      candidate_root=\"\$bootstrap_root/candidate\"
      mkdir -m 700 \"\$candidate_root\"
      prepared_candidate_sha=\$(run_verified_gateway_git_helper prepare --repo-root '$GATEWAY_REPO_ROOT' --repo-url https://github.com/leadpoet/leadpoet.git --branch main --deploy-commit '$commit' --plan-file \"\$bootstrap_root/plan.json\" --manifest-file \"\$bootstrap_root/candidate-manifest.json\" --last-good-file \"\$bootstrap_root/candidate-last-good-unused.json\")
      test \"\$prepared_candidate_sha\" = '$commit'
      GIT_NO_REPLACE_OBJECTS=1 git -C '$GATEWAY_REPO_ROOT' archive \"\$prepared_candidate_sha\" | tar -xf - -C \"\$candidate_root\"
      run_verified_gateway_git_helper verify-tree --plan-file \"\$bootstrap_root/plan.json\" --materialized-root \"\$candidate_root\" --phase prepared_archive --strict-extras >/dev/null
      find \"\$candidate_root\" -type f \( -perm -100 -o -perm -010 -o -perm -001 \) -exec chmod 500 {} +
      find \"\$candidate_root\" -type f ! \( -perm -100 -o -perm -010 -o -perm -001 \) -exec chmod 400 {} +
      find \"\$candidate_root\" -type d -exec chmod 500 {} +
      run_verified_gateway_git_helper verify-tree --plan-file \"\$bootstrap_root/plan.json\" --materialized-root \"\$candidate_root\" --phase prepared_archive --strict-extras >/dev/null
      controller_root='$PRODUCTION_GATEWAY_RESTART_CONTROLLER_ROOT'
      controller_release=\"\$controller_root/releases/$branch_commit\"
      controller_stage=\"\$bootstrap_root/controller-stage\"
      mkdir -p \"\$controller_root/releases\"
      chmod 700 \"\$(dirname \"\$controller_root\")\" \"\$controller_root\" \"\$controller_root/releases\"
      mkdir -m 700 \"\$controller_stage\"
      mkdir -m 700 \"\$controller_stage/Leadpoet\" \"\$controller_stage/Leadpoet/utils\" \"\$controller_stage/gateway\" \"\$controller_stage/gateway/tee\" \"\$controller_stage/scripts\"
      install -m 700 \"\$authority_root/gw_restart.sh\" \"\$controller_stage/gw_restart.sh\"
      install -m 600 \"\$authority_root/scripts/gateway_git_deploy.py\" \"\$controller_stage/scripts/gateway_git_deploy.py\"
      install -m 600 \"\$authority_root/Leadpoet/utils/exact_commit_restart_v2.py\" \"\$controller_stage/Leadpoet/utils/exact_commit_restart_v2.py\"
      install -m 600 \"\$authority_root/gateway/tee/host_memory_guard_v2.py\" \"\$controller_stage/gateway/tee/host_memory_guard_v2.py\"
      if [ -e \"\$controller_release\" ] || [ -L \"\$controller_release\" ]; then
        test -d \"\$controller_release\" && test ! -L \"\$controller_release\"
        test \"\$(stat -c '%u:%g:%a' \"\$controller_release\")\" = \"\$(id -u):\$(id -g):700\"
        test \"\$(stat -c '%u:%g:%a' \"\$controller_release/gw_restart.sh\")\" = \"\$(id -u):\$(id -g):700\"
        test \"\$(stat -c '%u:%g:%a' \"\$controller_release/scripts/gateway_git_deploy.py\")\" = \"\$(id -u):\$(id -g):600\"
        test \"\$(stat -c '%u:%g:%a' \"\$controller_release/Leadpoet/utils/exact_commit_restart_v2.py\")\" = \"\$(id -u):\$(id -g):600\"
        test \"\$(stat -c '%u:%g:%a' \"\$controller_release/gateway/tee/host_memory_guard_v2.py\")\" = \"\$(id -u):\$(id -g):600\"
        cmp -s \"\$controller_stage/gw_restart.sh\" \"\$controller_release/gw_restart.sh\"
        cmp -s \"\$controller_stage/scripts/gateway_git_deploy.py\" \"\$controller_release/scripts/gateway_git_deploy.py\"
        cmp -s \"\$controller_stage/Leadpoet/utils/exact_commit_restart_v2.py\" \"\$controller_release/Leadpoet/utils/exact_commit_restart_v2.py\"
        cmp -s \"\$controller_stage/gateway/tee/host_memory_guard_v2.py\" \"\$controller_release/gateway/tee/host_memory_guard_v2.py\"
        rm -rf -- \"\$controller_stage\"
      else
        mv -- \"\$controller_stage\" \"\$controller_release\"
      fi
      controller_link=\"\$controller_root/.current.\$\$\"
      rm -f -- \"\$controller_link\"
      ln -s \"releases/$branch_commit\" \"\$controller_link\"
      mv -Tf -- \"\$controller_link\" \"\$controller_root/current\"
      host_wrapper_temporary=\$(mktemp \"\$(dirname '$GATEWAY_RESTART')/.gw_restart.sh.XXXXXX\")
      install -m 700 \"\$authority_root/gw_restart.sh\" \"\$host_wrapper_temporary\"
      mv -f -- \"\$host_wrapper_temporary\" '$GATEWAY_RESTART'
      host_wrapper_temporary=''
      test \"\$(readlink -- \"\$controller_root/current\")\" = 'releases/$branch_commit'
      test \"\$(stat -c '%u:%g:%a' '$GATEWAY_RESTART')\" = \"\$(id -u):\$(id -g):700\"
      cmp -s '$GATEWAY_RESTART' \"\$controller_release/gw_restart.sh\""
  fi
  bootstrap_command="
      set -Eeuo pipefail
      umask 077
      unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN
      unset AWS_SECURITY_TOKEN AWS_PROFILE AWS_DEFAULT_PROFILE
      unset AWS_SHARED_CREDENTIALS_FILE AWS_WEB_IDENTITY_TOKEN_FILE AWS_ROLE_ARN
      unset AWS_ROLE_SESSION_NAME AWS_CONTAINER_CREDENTIALS_FULL_URI
      unset AWS_CONTAINER_CREDENTIALS_RELATIVE_URI AWS_CONFIG_FILE AWS_CA_BUNDLE
      unset AWS_ENDPOINT_URL AWS_ENDPOINT_URL_S3 AWS_ENDPOINT_URL_STS
      unset AWS_ENDPOINT_URL_SECRETSMANAGER AWS_EC2_METADATA_SERVICE_ENDPOINT
      unset AWS_EC2_METADATA_SERVICE_ENDPOINT_MODE AWS_METADATA_SERVICE_TIMEOUT
      unset AWS_METADATA_SERVICE_NUM_ATTEMPTS BOTO_CONFIG HTTP_PROXY HTTPS_PROXY
      unset ALL_PROXY http_proxy https_proxy all_proxy
      export AWS_REGION=us-east-1 AWS_DEFAULT_REGION=us-east-1
      export PYTHONDONTWRITEBYTECODE=1
      host_wrapper_temporary=''
      bootstrap_root=\$(mktemp -d /tmp/$bootstrap_prefix.XXXXXX)
      trap '[ -z \"\$host_wrapper_temporary\" ] || rm -f -- \"\$host_wrapper_temporary\"; chmod -R u+w \"\$bootstrap_root\" 2>/dev/null || true; rm -rf -- \"\$bootstrap_root\"' EXIT
      chmod 700 \"\$bootstrap_root\"
      authority_root=\"\$bootstrap_root/authority\"
      mkdir -m 700 \"\$authority_root\"
      controller_current='$PRODUCTION_GATEWAY_RESTART_CONTROLLER_CURRENT'
      run_verified_gateway_git_helper() {
        printf '%s' '$controller_verifier_b64' | '$GATEWAY_PYTHON_BIN' -I -S -c 'import base64,sys; source=base64.b64decode(sys.stdin.buffer.read(), validate=True); exec(compile(source, \"<exact-installed-controller-verifier>\", \"exec\"))' --repo-root '$GATEWAY_REPO_ROOT' --controller-current \"\$controller_current\" --host-restart-path '$GATEWAY_RESTART' --expected-controller-commit '$expected_controller_commit' --expected-commit '$branch_commit' --exec-helper scripts/gateway_git_deploy.py -- \"\$@\"
      }
      prepared_authority_sha=\$(run_verified_gateway_git_helper prepare --repo-root '$GATEWAY_REPO_ROOT' --repo-url https://github.com/leadpoet/leadpoet.git --branch main --deploy-commit '$branch_commit' --plan-file \"\$bootstrap_root/authority-plan.json\" --manifest-file \"\$bootstrap_root/authority-manifest.json\" --last-good-file \"\$bootstrap_root/authority-last-good-unused.json\")
      test \"\$prepared_authority_sha\" = '$branch_commit'
      GIT_NO_REPLACE_OBJECTS=1 git -C '$GATEWAY_REPO_ROOT' archive \"\$prepared_authority_sha\" | tar -xf - -C \"\$authority_root\"
      run_verified_gateway_git_helper verify-tree --plan-file \"\$bootstrap_root/authority-plan.json\" --materialized-root \"\$authority_root\" --phase prepared_archive --strict-extras >/dev/null
      find \"\$authority_root\" -type f \( -perm -100 -o -perm -010 -o -perm -001 \) -exec chmod 500 {} +
      find \"\$authority_root\" -type f ! \( -perm -100 -o -perm -010 -o -perm -001 \) -exec chmod 400 {} +
      find \"\$authority_root\" -type d -exec chmod 500 {} +
      run_verified_gateway_git_helper verify-tree --plan-file \"\$bootstrap_root/authority-plan.json\" --materialized-root \"\$authority_root\" --phase prepared_archive --strict-extras >/dev/null
$miner_candidate_prepare
      exec env \\
        LEADPOET_REPO_ROOT='$GATEWAY_REPO_ROOT' \\
        GATEWAY_ROOT='$GATEWAY_REPO_ROOT/gateway' \\
        GATEWAY_PYTHON_BIN='$GATEWAY_PYTHON_BIN' \\
        GATEWAY_HOST_RESTART_SCRIPT='$GATEWAY_RESTART' \\
        GATEWAY_RESTART_CONTROLLER_ROOT='$PRODUCTION_GATEWAY_RESTART_CONTROLLER_ROOT' \\
        GATEWAY_RESTART_AUTHORITY_ROOT=\"\$authority_root\" \\
        GATEWAY_RESTART_AUTHORITY_COMMIT='$branch_commit' \\
        GATEWAY_ACTIVE_RELEASE_RESTART_INVOCATION_ID='$active_release_restart_invocation_id' \\
        GATEWAY_PAIRED_ACTIVE_RELEASE_REQUIRED='$paired_required' \\
        GATEWAY_PAIRED_DESTRUCTIVE_HANDOFF_FILE='$paired_gateway_handoff_file' \\
        GATEWAY_PAIRED_DESTRUCTIVE_HANDOFF_NONCE='$paired_gateway_handoff_nonce' \\
        GATEWAY_PAIRED_DESTRUCTIVE_HANDOFF_TIMEOUT_SECONDS='$VALIDATOR_COORDINATION_TIMEOUT_SECONDS' \\
        LEADPOET_GATEWAY_ENV_SECRET_ID='$gateway_secret_id' \\
        GATEWAY_V2_RELEASE_BUCKET='leadpoet-attested-v2-artifacts-493765492819' \\
        RESEARCH_LAB_ATTESTED_V2_ARTIFACT_BUCKET='leadpoet-attested-v2-artifacts-493765492819' \\
        GATEWAY_V2_RELEASE_PREFIX='$RELEASE_PREFIX' \\
        GATEWAY_VALIDATOR_RELEASE_REQUIREMENTS='$gateway_validator_requirements_remote' \\
        GATEWAY_COUNTERPART_RELEASE_LINEAGE='$gateway_counterpart_path' \\
        GATEWAY_V2_RELEASE_REQUIREMENTS='$GATEWAY_ACTIVE_RELEASE_REQUIREMENTS_PATH' \\
        GATEWAY_V2_RELEASE_LINEAGE='$GATEWAY_ACTIVE_RELEASE_LINEAGE_PATH' \\
        AWS_REGION=us-east-1 AWS_DEFAULT_REGION=us-east-1 \\
        bash \"$gateway_restart_entrypoint_root/gw_restart.sh\" \\
          --commit '$commit'$miner_bootstrap_arguments"
  bootstrap_command_b64="$(
    printf '%s' "$bootstrap_command" | base64 | tr -d '\n'
  )"
  case "$bootstrap_command_b64" in
    *[!A-Za-z0-9+/=]*|'')
      echo "ERROR: gateway authority bootstrap transport encoding failed" >&2
      return 1
      ;;
  esac
  gateway_restart_command=(
    ssh -tt "${ssh_common[@]}" -i "$GATEWAY_KEY" "$GATEWAY_HOST"
    "exec bash -c \"\$(printf '%s' '$bootstrap_command_b64' | base64 --decode)\""
  )
}

run_gateway_restart() {
  build_gateway_restart_command
  "${gateway_restart_command[@]}"
}

start_gateway_restart_job() {
  local launcher_pid=""
  local observed_pgid=""
  build_gateway_restart_command
  gateway_job_session_file="$temporary_root/gateway-job-session"
  rm -f -- "$gateway_job_session_file"
  python3 -c '
import os
import sys

marker = sys.argv[1]
command = sys.argv[2:]
os.setsid()
descriptor = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
try:
    os.write(descriptor, (str(os.getpid()) + "\n").encode("ascii"))
    os.fsync(descriptor)
finally:
    os.close(descriptor)
os.execvp(command[0], command)
' "$gateway_job_session_file" "${gateway_restart_command[@]}" &
  launcher_pid="$!"
  gateway_job="$launcher_pid"
  for _ in $(seq 1 100); do
    if [ -s "$gateway_job_session_file" ]; then
      break
    fi
    if ! kill -0 "$launcher_pid" 2>/dev/null; then
      wait "$launcher_pid"
      return 1
    fi
    sleep 0.05
  done
  if [ "$(cat "$gateway_job_session_file" 2>/dev/null || true)" != "$launcher_pid" ]; then
    echo "ERROR: gateway restart did not establish its owned process group" >&2
    kill -KILL "$launcher_pid" 2>/dev/null || true
    wait "$launcher_pid" 2>/dev/null || true
    gateway_job=""
    return 1
  fi
  observed_pgid="$(ps -o pgid= -p "$launcher_pid" 2>/dev/null | tr -d '[:space:]')"
  if [ "$observed_pgid" != "$launcher_pid" ]; then
    echo "ERROR: gateway restart process-group identity differs" >&2
    kill -KILL "$launcher_pid" 2>/dev/null || true
    wait "$launcher_pid" 2>/dev/null || true
    gateway_job=""
    return 1
  fi
  gateway_job_pgid="$observed_pgid"
}

run_validator_restart() {
  local bootstrap_command=""
  local bootstrap_command_b64=""
  local coordination_environment=""
  local recovery_requirements_environment=""
  local recovery_lineage_environment=""
  local selected_active_release_authority_commit="${active_release_authority_commit:-$branch_commit}"
  local command=()
  if [ -n "$coordination_file" ]; then
    coordination_environment="$coordination_file"
  fi
  if [ "$validator_missing_runtime_recovery" = "1" ]; then
    recovery_requirements_environment="$validator_recovery_requirements_remote"
    recovery_lineage_environment="$validator_recovery_lineage_remote"
  fi
  bootstrap_command="
    set -Eeuo pipefail
    umask 077
    unset GIT_ALTERNATE_OBJECT_DIRECTORIES GIT_CEILING_DIRECTORIES GIT_COMMON_DIR
    unset GIT_CONFIG GIT_CONFIG_COUNT GIT_CONFIG_GLOBAL GIT_CONFIG_PARAMETERS
    unset GIT_CONFIG_SYSTEM GIT_DIR GIT_INDEX_FILE GIT_OBJECT_DIRECTORY
    unset GIT_REPLACE_REF_BASE GIT_WORK_TREE
    export GIT_CONFIG_NOSYSTEM=1 GIT_NO_REPLACE_OBJECTS=1 GIT_TERMINAL_PROMPT=0
    export PYTHONDONTWRITEBYTECODE=1
    git -C '$VALIDATOR_REPO_ROOT' fetch origin main
    test \"\$(git -C '$VALIDATOR_REPO_ROOT' rev-parse --verify origin/main^{commit})\" = '$branch_commit'
    git -C '$VALIDATOR_REPO_ROOT' cat-file -e '$commit^{commit}'
    git -C '$VALIDATOR_REPO_ROOT' merge-base --is-ancestor '$commit' '$branch_commit'
    test -z \"\$(git -C '$VALIDATOR_REPO_ROOT' for-each-ref --format='%(refname)' refs/replace)\"
    for authority_path in info/grafts objects/info/alternates; do
      resolved=\$(git -C '$VALIDATOR_REPO_ROOT' rev-parse --git-path \"\$authority_path\")
      case \"\$resolved\" in /*) ;; *) resolved='$VALIDATOR_REPO_ROOT/'\"\$resolved\" ;; esac
      if [ -e \"\$resolved\" ] && { [ ! -f \"\$resolved\" ] || [ -L \"\$resolved\" ] || [ -s \"\$resolved\" ]; }; then
        echo 'ERROR: validator Git authority contains graft or alternate objects' >&2
        exit 1
      fi
    done
    bootstrap_root=\$(mktemp -d /tmp/validator-restart-controller-bootstrap.XXXXXX)
    trap 'chmod -R u+w \"\$bootstrap_root\" 2>/dev/null || true; rm -rf -- \"\$bootstrap_root\"' EXIT
    chmod 700 \"\$bootstrap_root\"
    authority_root=\"\$bootstrap_root/authority\"
    mkdir -m 700 \"\$authority_root\"
    GIT_NO_REPLACE_OBJECTS=1 git -C '$VALIDATOR_REPO_ROOT' archive '$branch_commit' | tar -xf - -C \"\$authority_root\"
    test -r \"\$authority_root/validator_restart.sh\"
    test -r \"\$authority_root/gateway/tee/prepare_active_release_lineage_v2.py\"
    test -x '$VALIDATOR_PYTHON_BIN'
    test \"\$(git -C '$VALIDATOR_REPO_ROOT' hash-object --no-filters \"\$authority_root/validator_restart.sh\")\" = \"\$(git -C '$VALIDATOR_REPO_ROOT' rev-parse '$branch_commit:validator_restart.sh')\"
    find \"\$authority_root\" -type f -exec chmod 400 {} +
    find \"\$authority_root\" -type d -exec chmod 500 {} +
    exec env \\
      VALIDATOR_ROOT='$VALIDATOR_REPO_ROOT' \\
      VALIDATOR_PYTHON_BIN='$VALIDATOR_PYTHON_BIN' \\
      VALIDATOR_HOST_RESTART_SCRIPT='$VALIDATOR_RESTART' \\
      VALIDATOR_ACTIVE_RELEASE_AUTHORITY_ROOT=\"\$authority_root\" \\
      VALIDATOR_ACTIVE_RELEASE_AUTHORITY_COMMIT='$selected_active_release_authority_commit' \\
      VALIDATOR_ACTIVE_RELEASE_RESTART_INVOCATION_ID='$active_release_restart_invocation_id' \\
      VALIDATOR_PAIRED_ACTIVE_RELEASE_REQUIRED=1 \\
      VALIDATOR_V2_HOTKEY_CONFIG='$VALIDATOR_V2_HOTKEY_CONFIG_PATH' \\
      VALIDATOR_CHAIN_SIGNING_PROFILE='$VALIDATOR_CHAIN_SIGNING_PROFILE_PATH' \\
      VALIDATOR_PINNED_GATEWAY_COORDINATION_FILE='$coordination_environment' \\
      VALIDATOR_PINNED_GATEWAY_COORDINATION_MAX_ATTEMPTS='$VALIDATOR_COORDINATION_ATTEMPTS' \\
      VALIDATOR_PINNED_GATEWAY_TIMEOUT_SECONDS='$VALIDATOR_COORDINATION_TIMEOUT_SECONDS' \\
      VALIDATOR_ACTIVE_RELEASE_REQUIREMENTS_OUTPUT='$validator_initial_requirements_remote' \\
      VALIDATOR_FINAL_RELEASE_REQUIREMENTS_INPUT='$validator_final_requirements_remote' \\
      VALIDATOR_FINAL_RELEASE_LINEAGE_INPUT='$validator_final_lineage_remote' \\
      VALIDATOR_MISSING_RUNTIME_RECOVERY_REQUIREMENTS='$recovery_requirements_environment' \\
      VALIDATOR_MISSING_RUNTIME_RECOVERY_LINEAGE='$recovery_lineage_environment' \\
      LEADPOET_VALIDATOR_ENV_SECRET_ID='$VALIDATOR_ENV_SECRET_ID' \\
      VALIDATOR_V2_RELEASE_PREFIX='$RELEASE_PREFIX' \\
      bash \"\$authority_root/validator_restart.sh\" --commit '$commit'"
  bootstrap_command_b64="$(
    printf '%s' "$bootstrap_command" | base64 | tr -d '\n'
  )"
  case "$bootstrap_command_b64" in
    *[!A-Za-z0-9+/=]*|'')
      echo "ERROR: validator authority bootstrap transport encoding failed" >&2
      return 1
      ;;
  esac
  command=(
    ssh -tt "${ssh_common[@]}" -i "$VALIDATOR_KEY" "$VALIDATOR_HOST"
    "exec bash -c \"\$(printf '%s' '$bootstrap_command_b64' | base64 --decode)\""
  )
  if [ "${VALIDATOR_RESTART_EXEC_SSH:-0}" = "1" ]; then
    exec "${command[@]}"
  fi
  "${command[@]}"
}

verify_gateway_release() {
  local output="$1"
  ssh "${ssh_common[@]}" -i "$GATEWAY_KEY" "$GATEWAY_HOST" \
    "cd '$GATEWAY_REPO_ROOT' && PYTHONPATH='$GATEWAY_REPO_ROOT' \
      '$GATEWAY_PYTHON_BIN' - '$commit' '$historical_topology_hash' <<'PY'
import asyncio
import json
import os
from pathlib import Path
import sys
import urllib.request

expected = sys.argv[1]
historical_topology_hash = sys.argv[2] or None
# gateway_exact_release_ready: retained as the operator/fault-injection probe label.

def get(path):
    with urllib.request.urlopen(
        'http://127.0.0.1:8000' + path,
        timeout=35,
    ) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise SystemExit('gateway response is not an object: ' + path)
    return value

health = get('/health/v2-authority')
build = get('/build-info')
release = get('/weights/v2/release-evidence/' + expected)
attestation = get('/attest')
if health.get('status') != 'ready':
    raise SystemExit('gateway V2 authority is not ready')
if str(health.get('commit_sha') or '').lower() != expected:
    raise SystemExit('gateway V2 authority commit differs')
if str(build.get('git_commit') or '').lower() != expected:
    raise SystemExit('gateway build-info commit differs')
if str(release.get('commit_sha') or '').lower() != expected:
    raise SystemExit('gateway release evidence commit differs')
if (
    not isinstance(health.get('enclaves'), dict)
    or health['enclaves'].get('status') != 'ready'
):
    raise SystemExit('gateway authority endpoint lacks ready enclave evidence')

from gateway.deploy_readiness import read_source_commit
from gateway.tee.provider_broker_v2 import (
    measured_retry_policy_hashes,
    provider_registry_hash,
)
from gateway.tee.topology import ROLE_SPECS
from gateway.tee.verify_v2_runtime_ready import verify_v2_runtime_ready
from leadpoet_canonical.auditor_v2 import (
    fetch_locked_release_identity_cache,
    identity_cache_from_release_channel,
)
from gateway.tee.research_lab_runtime_config_v2 import (
    build_research_lab_execution_config,
)
from gateway.utils.tee_client import TEEClient
from gateway.utils.tee_kms_provision_v2 import (
    load_provider_envelopes,
    provider_reference_hashes_from_envelopes,
)
from gateway.utils.tee_v2_bootstrap import runtime_configuration_documents

processes = []
for candidate in Path('/proc').iterdir():
    if not candidate.name.isdigit():
        continue
    try:
        command = candidate.joinpath('cmdline').read_bytes().split(b'\\0')
    except OSError:
        continue
    if b'-m' in command and b'gateway.main' in command:
        processes.append(candidate)
if len(processes) != 1:
    raise SystemExit('exactly one gateway.main process is required')
runtime_environment = {}
for item in processes[0].joinpath('environ').read_bytes().split(b'\\0'):
    if b'=' not in item:
        continue
    name, value = item.split(b'=', 1)
    runtime_environment[name.decode('utf-8')] = value.decode('utf-8')
os.environ.clear()
os.environ.update(runtime_environment)

config_dir = Path(
    runtime_environment.get('GATEWAY_V2_CONFIG_DIR')
    or '/home/ec2-user/.config/leadpoet/v2'
)
gateway_release_path = Path(
    runtime_environment.get('GATEWAY_V2_RELEASE_MANIFEST')
    or '/home/ec2-user/tee/gateway-v2-release-manifest.json'
)
validator_release_path = Path(
    runtime_environment.get('GATEWAY_V2_VALIDATOR_RELEASE_MANIFEST')
    or '/home/ec2-user/tee/validator-v2-release-manifest.json'
)
lineage_path = Path(
    runtime_environment.get('GATEWAY_V2_RELEASE_LINEAGE')
    or '/home/ec2-user/tee/gateway-v2-release-lineage.json'
)
artifact_policy_path = Path(
    runtime_environment.get('GATEWAY_V2_ARTIFACT_POLICY')
    or config_dir / 'encrypted-artifact-policy.json'
)
gateway_root = Path(
    runtime_environment.get('GATEWAY_ROOT')
    or '$GATEWAY_REPO_ROOT/gateway'
)
gateway_release = json.loads(gateway_release_path.read_text(encoding='utf-8'))
lineage = json.loads(lineage_path.read_text(encoding='utf-8'))
if historical_topology_hash is not None:
    expected_historical_roles = {
        'gateway_autoresearch',
        'gateway_coordinator',
        'gateway_scoring',
    }
    if (
        historical_topology_hash
        != 'sha256:a13a1b16fb1501f953b2396aba88b87d7e5e0d3cfac4079b9230ea6165a88f34'
        or set(ROLE_SPECS) != expected_historical_roles
        or gateway_release.get('topology_hash') != historical_topology_hash
    ):
        raise SystemExit('historical gateway topology differs')
    from gateway.tee.release_channel_v2 import fetch_release_channel_v2

    channel = fetch_release_channel_v2(
        bucket=(
            runtime_environment.get('GATEWAY_V2_RELEASE_BUCKET')
            or 'leadpoet-attested-v2-artifacts-493765492819'
        ),
        commit_sha=expected,
        prefix=(
            runtime_environment.get('GATEWAY_V2_RELEASE_PREFIX')
            or 'attested-v2/releases'
        ),
    )
    if gateway_release != channel['gateway_release_manifest']:
        raise SystemExit('active gateway release differs from immutable channel')
    validator_release = channel['validator_release_manifest']
else:
    from gateway.tee.release_channel_v2 import build_release_channel_v2

    validator_release = json.loads(
        validator_release_path.read_text(encoding='utf-8')
    )
    channel = build_release_channel_v2(
        gateway_release_manifest=gateway_release,
        validator_release_manifest=validator_release,
    )
    if gateway_release != channel['gateway_release_manifest']:
        raise SystemExit('active gateway release differs from the verified runtime identity')
if identity_cache_from_release_channel(channel) != fetch_locked_release_identity_cache(
    release
):
    raise SystemExit('active gateway release differs from auditor release evidence')
envelopes = load_provider_envelopes([
    config_dir / name
    for name in (
        'artifact_master_key.json',
        'openrouter.json',
        'exa.json',
        'scrapingdog.json',
        'deepline.json',
        'supabase_service_role.json',
        'truelist.json',
    )
])
artifact_envelopes = [
    item for item in envelopes
    if item.get('credential_slot') == 'artifact_master_key'
]
if len(artifact_envelopes) != 1:
    raise SystemExit('gateway artifact master-key envelope is not unique')
protected = json.loads(
    (gateway_root / '_attested_runtime/protected_workflows.json').read_text(
        encoding='utf-8'
    )
)
protected_hash = str(protected.get('manifest_hash') or '').lower()
if historical_topology_hash is not None:
    from gateway.research_lab.provider_profiles_v2 import (
        verify_required_worker_proxy_profiles_v2,
    )

    configured_worker_counts = verify_required_worker_proxy_profiles_v2(
        config_dir=config_dir
    )['worker_counts']
else:
    from gateway.utils.tee_v2_bootstrap import configured_scoring_worker_count

    configured_worker_counts = {
        'gateway_scoring': configured_scoring_worker_count(config_dir)
    }
execution_config = build_research_lab_execution_config(
    environment=runtime_environment
)
documents = runtime_configuration_documents(
    release_manifest=gateway_release,
    gateway_release_lineage=lineage,
    provider_ref_hashes=provider_reference_hashes_from_envelopes(envelopes),
    provider_retry_policy_hashes=measured_retry_policy_hashes(protected_hash),
    provider_registry_hash=provider_registry_hash(
        execution_config=execution_config
    ),
    protected_workflow_manifest_hash=protected_hash,
    encrypted_artifact_policy=json.loads(
        artifact_policy_path.read_text(encoding='utf-8')
    ),
    artifact_master_key_ref_hash=str(
        artifact_envelopes[0]['credential_ref_hash']
    ),
    research_lab_execution_config=execution_config,
    configured_worker_counts=configured_worker_counts,
)

async def collect_runtime():
    clients = {
        role: TEEClient(cid=int(spec['cid']))
        for role, spec in ROLE_SPECS.items()
    }
    readiness = await verify_v2_runtime_ready(clients)
    boots = {}
    for role in sorted(ROLE_SPECS):
        boots[role] = await clients[role].v2_get_boot_identity()
    return readiness, boots

runtime_readiness, boots = asyncio.run(collect_runtime())
source_commit, _ = read_source_commit()
observation = {
    'schema_version': 'leadpoet.gateway_deploy_readiness_observation.v2',
    'source_commit': source_commit,
    'build_commit': build.get('git_commit'),
    'gateway_release_manifest': gateway_release,
    'validator_release_manifest': validator_release,
    'compact_lineage': lineage,
    'boot_identities': boots,
    'expected_role_config_hashes': {
        role: documents[role]['configuration_hash']
        for role in sorted(documents)
    },
    'runtime_readiness': runtime_readiness,
    'coordinator_attestation_pcr0': attestation.get('pcr0'),
}
print(json.dumps(observation, sort_keys=True, separators=(',', ':')))
PY" > "$gateway_observation"
  run_local_readiness_python \
    "$commit" "$historical_topology_hash" \
    "$gateway_observation" "$output" <<'PY'
import json
from pathlib import Path
import sys
from gateway.deploy_readiness import (
    build_gateway_v2_readiness_evidence_from_observation,
)

observation = json.loads(Path(sys.argv[3]).read_text(encoding='utf-8'))
evidence = build_gateway_v2_readiness_evidence_from_observation(
    expected_commit=sys.argv[1],
    observation=observation,
    expected_historical_topology_hash=(sys.argv[2] or None),
)
Path(sys.argv[4]).write_text(
    json.dumps(evidence, sort_keys=True, separators=(',', ':')) + '\n',
    encoding='ascii',
)
PY
  echo "Verified fresh Nitro identity and runtime configuration for all gateway roles"
}

verify_validator_release() {
  local output="$1"
  ssh "${ssh_common[@]}" -i "$VALIDATOR_KEY" "$VALIDATOR_HOST" \
    "cd '$VALIDATOR_REPO_ROOT' && PYTHONPATH='$VALIDATOR_REPO_ROOT' \
      python3 - '$commit' <<'PY'
import json
import os
from pathlib import Path
import subprocess
import sys

expected = sys.argv[1]
# validator_exact_release_ready: retained as the operator/fault-injection probe label.
running = subprocess.check_output(
    ['docker', 'inspect', '-f', '{{.State.Running}}', 'leadpoet-validator-main'],
    text=True,
).strip()
restarts = subprocess.check_output(
    ['docker', 'inspect', '-f', '{{.RestartCount}}', 'leadpoet-validator-main'],
    text=True,
).strip()
environment = subprocess.check_output(
    [
        'docker',
        'inspect',
        '-f',
        '{{range .Config.Env}}{{println .}}{{end}}',
        'leadpoet-validator-main',
    ],
    text=True,
).splitlines()
values = dict(
    line.split('=', 1)
    for line in environment
    if '=' in line
)
if running != 'true' or restarts != '0':
    raise SystemExit('validator coordinator is not cleanly running')
if values.get('VALIDATOR_V2_DEPLOY_COMMIT') != expected:
    raise SystemExit('validator coordinator commit differs')
if values.get('VALIDATOR_WEIGHT_PROTOCOL') != 'authoritative_v2':
    raise SystemExit('validator coordinator is not authoritative V2')

from gateway.tee.release_lineage_v2 import validate_compact_release_lineage_v2
from leadpoet_canonical.attested_v2 import sha256_json
from validator_tee.host.runtime_v2_bootstrap import build_runtime_configuration
from validator_tee.host.vsock_client import ValidatorEnclaveClient

host_commit = subprocess.check_output(
    ['git', '-C', '$VALIDATOR_REPO_ROOT', 'rev-parse', 'HEAD'],
    text=True,
).strip().lower()
runtime_environment = dict(os.environ)
runtime_environment.update(values)
os.environ.clear()
os.environ.update(runtime_environment)
gateway_release = json.loads(
    Path('/home/ec2-user/.config/leadpoet/gateway-v2-release-manifest.json').read_text(
        encoding='utf-8'
    )
)
validator_release = json.loads(
    Path('/home/ec2-user/.config/leadpoet/validator-v2-release-manifest.json').read_text(
        encoding='utf-8'
    )
)
lineage = json.loads(
    Path('/home/ec2-user/.config/leadpoet/gateway-v2-release-lineage.json').read_text(
        encoding='utf-8'
    )
)
hotkey_config = json.loads(
    Path('/home/ec2-user/.config/leadpoet/validator-hotkey-config-v2.json').read_text(
        encoding='utf-8'
    )
)
configuration = build_runtime_configuration(
    validator_release=validator_release,
    gateway_release=gateway_release,
    gateway_release_lineage=lineage,
    hotkey_authority_config=hotkey_config,
)
client = ValidatorEnclaveClient()
enclave_health = client.health_check()
if (
    enclave_health.get('status') != 'ok'
    or enclave_health.get('authoritative_v2_configured') is not True
    or enclave_health.get('hotkey_authority_v2_configured') is not True
):
    raise SystemExit('validator enclave authority is not fully ready')
boot = client.get_authoritative_v2_boot_identity()
observation = {
    'schema_version': 'leadpoet.validator_deploy_readiness_observation.v2',
    'host_commit': host_commit,
    'gateway_release_manifest': gateway_release,
    'validator_release_manifest': validator_release,
    'compact_lineage': lineage,
    'boot_identity': boot,
    'expected_config_hash': sha256_json(configuration),
}
print(json.dumps(observation, sort_keys=True, separators=(',', ':')))
PY" > "$validator_observation"
  run_local_readiness_python \
    "$commit" "$historical_topology_hash" \
    "$validator_observation" "$output" <<'PY'
import json
from pathlib import Path
import sys
from gateway.deploy_readiness import (
    build_validator_v2_readiness_evidence_from_observation,
)

observation = json.loads(Path(sys.argv[3]).read_text(encoding='utf-8'))
evidence = build_validator_v2_readiness_evidence_from_observation(
    expected_commit=sys.argv[1],
    observation=observation,
    expected_historical_topology_hash=(sys.argv[2] or None),
)
Path(sys.argv[4]).write_text(
    json.dumps(evidence, sort_keys=True, separators=(',', ':')) + '\n',
    encoding='ascii',
)
PY
  echo "Verified fresh Nitro identity and runtime configuration for validator_weights"
}

finalize_deploy_readiness() {
  run_local_readiness_python \
    "$commit" "$historical_topology_hash" \
    "$gateway_evidence" "$validator_evidence" "$final_manifest" <<'PY'
import json
from pathlib import Path
import sys
from gateway.deploy_readiness import (
    build_v2_deploy_readiness_manifest,
    validate_v2_deploy_readiness_manifest,
    write_deploy_readiness_manifest,
)

commit = sys.argv[1]
historical_topology_hash = sys.argv[2] or None
gateway = json.loads(Path(sys.argv[3]).read_text(encoding='utf-8'))
validator = json.loads(Path(sys.argv[4]).read_text(encoding='utf-8'))
manifest = build_v2_deploy_readiness_manifest(
    expected_commit=commit,
    gateway_evidence=gateway,
    validator_evidence=validator,
    expected_historical_topology_hash=historical_topology_hash,
)
validate_v2_deploy_readiness_manifest(
    manifest,
    runtime_source_commit=commit,
    runtime_build_commit=commit,
    expected_historical_topology_hash=historical_topology_hash,
)
write_deploy_readiness_manifest(manifest, sys.argv[5])
PY
  install_gateway_readiness_manifest readiness-final "$final_manifest"
  echo "Installed schema-v2 deploy readiness authority for $commit"
}

case "$component" in
  gateway)
    observed_validator="$(validator_active_commit)"
    if [ "$observed_validator" != "$commit" ]; then
      echo "ERROR: validator is on $observed_validator, not $commit; use --component all" >&2
      exit 1
    fi
    verify_validator_release "$validator_evidence"
    prepare_running_validator_release_requirements
    install_gateway_validator_release_requirements
    fetch_and_install_gateway_counterpart_lineage
    invalidate_deploy_readiness
    verify_local_readiness_python_binding
    run_gateway_restart
    fetch_gateway_final_release_authority
    ;;
  validator)
    observed_gateway="$(gateway_active_commit)"
    if [ "$observed_gateway" != "$commit" ]; then
      echo "ERROR: gateway is on $observed_gateway, not $commit; use --component all" >&2
      exit 1
    fi
    verify_gateway_release "$gateway_evidence"
    bind_component_validator_to_gateway_release_authority
    validator_runtime_probe_status=0
    validator_active_commit >/dev/null || validator_runtime_probe_status="$?"
    case "$validator_runtime_probe_status" in
      0) ;;
      44)
        validator_missing_runtime_recovery=1
        install_validator_missing_runtime_recovery_authority
        ;;
      *)
        echo "ERROR: validator runtime state could not be established safely" >&2
        exit 1
        ;;
    esac
    invalidate_deploy_readiness
    verify_local_readiness_python_binding
    run_validator_restart_against_active_gateway
    ;;
  all)
    validator_log="$temporary_root/validator-restart.log"
    gateway_restart_log="$temporary_root/gateway-restart.log"
    coordination_file="/tmp/leadpoet-coordinated-restart.$(basename "$temporary_root").ready"
    paired_gateway_handoff_file="/tmp/leadpoet-gateway-paired-restart.$(basename "$temporary_root").ready"
    if [ "$disable_miner_submissions_before_restart" = "1" ]; then
      gateway_handoff_file="/tmp/leadpoet-gateway-miner-maintenance-handoff.$(basename "$temporary_root").ready"
    fi
    paired_restart_deadline=$((SECONDS + VALIDATOR_COORDINATION_TIMEOUT_SECONDS))
    ssh "${ssh_common[@]}" -i "$VALIDATOR_KEY" "$VALIDATOR_HOST" \
      "rm -f -- '$coordination_file' '$validator_initial_requirements_remote' '$validator_final_requirements_remote' '$validator_final_requirements_remote.tmp' '$validator_final_lineage_remote' '$validator_final_lineage_remote.tmp' '$validator_recovery_requirements_remote' '$validator_recovery_requirements_remote.tmp' '$validator_recovery_lineage_remote' '$validator_recovery_lineage_remote.tmp'"
    ssh "${ssh_common[@]}" -i "$GATEWAY_KEY" "$GATEWAY_HOST" \
      "rm -f -- '$gateway_validator_requirements_remote' '$gateway_validator_requirements_remote.tmp' '$paired_gateway_handoff_file' '$paired_gateway_handoff_file.tmp' '$gateway_handoff_file' '$gateway_handoff_file.tmp'"

    if [ "$disable_miner_submissions_before_restart" = "1" ]; then
      echo "Starting exact-candidate miner maintenance before readiness invalidation"
      start_gateway_restart_job > >(tee "$gateway_restart_log") 2>&1
      gateway_bootstrap_ready=0
      while [ "$SECONDS" -lt "$paired_restart_deadline" ]; do
        if grep -Fq \
            "Prepared exact-candidate miner maintenance under the canonical restart lock" \
            "$gateway_restart_log"; then
          gateway_bootstrap_ready=1
          break
        fi
        if ! kill -0 "$gateway_job" 2>/dev/null; then
          report_restart_job_early_exit gateway "$gateway_job"
          exit 1
        fi
        sleep 1
      done
      if [ "$gateway_bootstrap_ready" != "1" ]; then
        echo "ERROR: gateway miner-maintenance bootstrap did not become ready" >&2
        exit 1
      fi
      echo "Exact-candidate miner maintenance is prepared under the canonical restart lock"
    fi

    invalidate_deploy_readiness
    verify_local_readiness_python_binding
    echo "Starting exact-authority validator and gateway preparation in parallel"
    (
      VALIDATOR_RESTART_EXEC_SSH=1 run_validator_restart
    ) > >(tee "$validator_log") 2>&1 &
    validator_job="$!"
    if [ "$disable_miner_submissions_before_restart" != "1" ]; then
      start_gateway_restart_job > >(tee "$gateway_restart_log") 2>&1
    fi

    capture_ready=0
    while [ "$SECONDS" -lt "$paired_restart_deadline" ]; do
      if grep -Fq \
          "Prepared validator active release requirements sidecar" \
          "$validator_log"; then
        capture_ready=1
        break
      fi
      if ! kill -0 "$validator_job" 2>/dev/null; then
        report_restart_job_early_exit validator "$validator_job"
        exit 1
      fi
      if ! kill -0 "$gateway_job" 2>/dev/null; then
        report_restart_job_early_exit gateway "$gateway_job"
        exit 1
      fi
      sleep 1
    done
    if [ "$capture_ready" != "1" ]; then
      echo "ERROR: validator did not prepare its active release requirements" >&2
      exit 1
    fi

    fetch_validator_initial_release_requirements
    install_gateway_validator_release_requirements
    if [ "$disable_miner_submissions_before_restart" = "1" ]; then
      publish_gateway_handoff_value "$commit"
    fi

    gateway_handoff_ready=0
    while [ "$SECONDS" -lt "$paired_restart_deadline" ]; do
      if ! kill -0 "$validator_job" 2>/dev/null; then
        publish_paired_gateway_handoff_value "failed:$commit" || true
        report_restart_job_early_exit validator "$validator_job"
        exit 1
      fi
      if grep -Fq \
          "Gateway pre-shutdown checks complete; awaiting paired validator liveness handoff" \
          "$gateway_restart_log"; then
        gateway_handoff_ready=1
        break
      fi
      if ! kill -0 "$gateway_job" 2>/dev/null; then
        report_restart_job_early_exit gateway "$gateway_job"
        exit 1
      fi
      sleep 1
    done
    if [ "$gateway_handoff_ready" != "1" ]; then
      echo "ERROR: gateway did not reach its paired pre-shutdown handoff" >&2
      exit 1
    fi
    publish_paired_gateway_handoff_value "$commit"

    while kill -0 "$gateway_job" 2>/dev/null; do
      if ! kill -0 "$validator_job" 2>/dev/null; then
        publish_paired_gateway_handoff_value "failed:$commit" || true
        report_restart_job_early_exit validator "$validator_job"
        exit 1
      fi
      if [ "$SECONDS" -ge "$paired_restart_deadline" ]; then
        echo "ERROR: gateway restart exceeded the shared paired deadline" >&2
        exit 1
      fi
      sleep 1
    done
	    if wait "$gateway_job"; then
	      :
	    else
	      gateway_status="$?"
	      echo "ERROR: gateway restart exited before coordinated completion (status $gateway_status)" >&2
	      exit 1
	    fi
    gateway_job=""
    gateway_job_pgid=""

    fetch_gateway_final_release_authority
    install_validator_final_release_authority
    publish_coordination_value "$commit"
    echo "Gateway restart completed; releasing exact-SHA validator activation"

    while kill -0 "$validator_job" 2>/dev/null; do
      if [ "$SECONDS" -ge "$paired_restart_deadline" ]; then
        echo "ERROR: validator restart exceeded the shared paired deadline" >&2
        exit 1
      fi
      sleep 1
    done
	    if wait "$validator_job"; then
	      :
	    else
	      validator_status="$?"
	      echo "ERROR: validator restart exited before coordinated completion (status $validator_status)" >&2
	      exit 1
	    fi
    validator_job=""
    ;;
esac

verify_gateway_release "$gateway_evidence"
verify_validator_release "$validator_evidence"
finalize_deploy_readiness
echo "SUCCESS: gateway and validator are aligned on attested release $commit"
