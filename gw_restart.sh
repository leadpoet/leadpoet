#!/bin/bash
set -euo pipefail

GATEWAY_GIT_DEPLOY_PROTOCOL="1"
LEADPOET_REPO_ROOT="${LEADPOET_REPO_ROOT:-/home/ec2-user/leadpoet_repo}"
GATEWAY_ROOT="${GATEWAY_ROOT:-$LEADPOET_REPO_ROOT/gateway}"
GATEWAY_LOG_ROOT="${GATEWAY_LOG_ROOT:-/home/ec2-user/gateway}"
GATEWAY_LOG_FILE="${GATEWAY_LOG_FILE:-$GATEWAY_LOG_ROOT/gateway.log}"
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
GATEWAY_GIT_HELPER="${GATEWAY_GIT_HELPER:-$LEADPOET_REPO_ROOT/scripts/gateway_git_deploy.py}"
GATEWAY_RESTART_PHASE="${GATEWAY_RESTART_PHASE:-prepare}"
GATEWAY_STATEFUL_CUTOVER_CEREMONY="${GATEWAY_STATEFUL_CUTOVER_CEREMONY:-0}"
GATEWAY_STATEFUL_CUTOVER_SUPABASE_TIMEOUT_SECONDS=120
GATEWAY_STATEFUL_CUTOVER_MANIFEST="/home/ec2-user/.config/leadpoet/stateful-epoch-cutover.json"
GATEWAY_STATEFUL_CUTOVER_VALIDATOR_RELEASE_MANIFEST="${GATEWAY_STATEFUL_CUTOVER_VALIDATOR_RELEASE_MANIFEST:-/home/ec2-user/.config/leadpoet/validator-v2-release-manifest.json}"
GATEWAY_RESTART_START_PATH="/home/ec2-user/.config/leadpoet/restart-start-v1.json"
GATEWAY_RESTART_LOCK_FILE="${GATEWAY_RESTART_LOCK_FILE:-/home/ec2-user/.config/leadpoet/gateway-restart.lock}"
GATEWAY_RESTART_RECOVERY_LOCK_FILE="${GATEWAY_RESTART_RECOVERY_LOCK_FILE:-${GATEWAY_RESTART_LOCK_FILE}.recovery}"
GATEWAY_DEPLOY_PLAN_FILE="${GATEWAY_DEPLOY_PLAN_FILE:-/tmp/gateway_git_deploy.$$.json}"
GATEWAY_DEPLOYMENT_DIR="${GATEWAY_DEPLOYMENT_DIR:-/home/ec2-user/.config/leadpoet/deployments}"
GATEWAY_DEPLOYMENT_MANIFEST="${GATEWAY_DEPLOYMENT_MANIFEST:-$GATEWAY_DEPLOYMENT_DIR/gateway-current.json}"
GATEWAY_LAST_GOOD_MANIFEST="${GATEWAY_LAST_GOOD_MANIFEST:-$GATEWAY_DEPLOYMENT_DIR/gateway-last-good.json}"
GATEWAY_HOST_RESTART_SCRIPT="${GATEWAY_HOST_RESTART_SCRIPT:-/home/ec2-user/gw_restart.sh}"
GATEWAY_TEE_EIF_ROOT="${GATEWAY_TEE_EIF_ROOT:-/home/ec2-user/tee}"
RESEARCH_LAB_TEE_PROTOCOL="${RESEARCH_LAB_TEE_PROTOCOL:-}"
GATEWAY_V2_CONFIG_DIR="${GATEWAY_V2_CONFIG_DIR:-/home/ec2-user/.config/leadpoet/v2}"
GATEWAY_V2_RELEASE_MANIFEST="${GATEWAY_V2_RELEASE_MANIFEST:-$GATEWAY_TEE_EIF_ROOT/gateway-v2-release-manifest.json}"
GATEWAY_V2_ARTIFACT_POLICY="${GATEWAY_V2_ARTIFACT_POLICY:-$GATEWAY_V2_CONFIG_DIR/encrypted-artifact-policy.json}"
GATEWAY_V2_ACCEPTANCE_CORPUS_MANIFEST="${GATEWAY_V2_ACCEPTANCE_CORPUS_MANIFEST:-$GATEWAY_V2_CONFIG_DIR/acceptance-corpus-v2.json}"
GATEWAY_V2_ACCEPTANCE_CORPUS_ROOT="${GATEWAY_V2_ACCEPTANCE_CORPUS_ROOT:-$GATEWAY_V2_CONFIG_DIR/acceptance-corpus-v2}"
GATEWAY_V2_RELEASE_BUCKET="${GATEWAY_V2_RELEASE_BUCKET:-leadpoet-attested-v2-artifacts-493765492819}"
RESEARCH_LAB_ATTESTED_V2_ARTIFACT_BUCKET="${RESEARCH_LAB_ATTESTED_V2_ARTIFACT_BUCKET:-$GATEWAY_V2_RELEASE_BUCKET}"
GATEWAY_V2_RELEASE_PREFIX="${GATEWAY_V2_RELEASE_PREFIX:-attested-v2/releases}"
GATEWAY_V2_KMS_KEY_ID="${GATEWAY_V2_KMS_KEY_ID:-arn:aws:kms:us-east-1:493765492819:key/c5412928-093e-4bf5-aafc-7b27c02f1445}"
export GATEWAY_V2_OFFLINE_ARTIFACT_ROOT="${GATEWAY_V2_OFFLINE_ARTIFACT_ROOT:-$HOME/.cache/leadpoet-v2-artifacts}"
export VALIDATOR_V2_OFFLINE_ARTIFACT_ROOT="${VALIDATOR_V2_OFFLINE_ARTIFACT_ROOT:-$GATEWAY_V2_OFFLINE_ARTIFACT_ROOT/validator-runtime}"
GATEWAY_DEPLOY_STAGE="${GATEWAY_DEPLOY_STAGE:-bootstrap}"
GATEWAY_DEPLOY_COMPLETED=0
GATEWAY_PREFLIGHT_TREE=""
GATEWAY_HOST_MEMORY_GUARD_PID=""
GATEWAY_HOST_MEMORY_GUARD_PATH="${GATEWAY_HOST_MEMORY_GUARD_PATH:-$LEADPOET_REPO_ROOT/gateway/tee/host_memory_guard_v2.py}"
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

install_gateway_python_dependencies() {
  local legacy_project_metadata pip_scope=() requirements_file
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
  export PATH="$HOME/.local/bin:$PATH"
  # These legacy distributions install the same `scalecodec` import namespace
  # as Cyscale and make Bittensor 10 fail at import time. The editable project
  # metadata can also retain an obsolete Bittensor <10 requirement even though
  # gateway modules load from the exact canonical checkout via PYTHONPATH.
  "$GATEWAY_PYTHON_BIN" -m pip uninstall -y \
    leadpoet-subnet substrate-interface py-scale-codec scalecodec \
    >/dev/null 2>&1 || true
  "$GATEWAY_PYTHON_BIN" -m pip install \
    "${pip_scope[@]}" \
    --requirement "$requirements_file" \
    "${GATEWAY_HOST_EXTRA_PYTHON_PACKAGES[@]}"
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
    "$GATEWAY_V2_RELEASE_MANIFEST" \
    "$GATEWAY_V2_ARTIFACT_POLICY" \
    "$GATEWAY_V2_ACCEPTANCE_CORPUS_MANIFEST" \
    "$GATEWAY_V2_ACCEPTANCE_CORPUS_ROOT" \
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
    "required_external_approvals": [
        "offline_acceptance_corpus_signature",
        "independent_gateway_and_validator_parent_build_evidence",
    ],
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

on_gateway_restart_exit() {
  local status="$?"
  if [ -n "${GATEWAY_HOST_MEMORY_GUARD_PID:-}" ]; then
    kill "$GATEWAY_HOST_MEMORY_GUARD_PID" >/dev/null 2>&1 || true
    wait "$GATEWAY_HOST_MEMORY_GUARD_PID" >/dev/null 2>&1 || true
  fi
  if [ -n "${GATEWAY_PREFLIGHT_TREE:-}" ]; then
    rm -rf "$GATEWAY_PREFLIGHT_TREE"
  fi
  if [ "$status" -ne 0 ] \
      && [ "$GATEWAY_DEPLOY_COMPLETED" != "1" ] \
      && [ -f "$GATEWAY_DEPLOY_PLAN_FILE" ] \
      && [ -f "$GATEWAY_GIT_HELPER" ]; then
    finalize_deployment_record failed "$GATEWAY_DEPLOY_STAGE" >/dev/null 2>&1 || true
  fi
}

start_gateway_host_memory_guard() {
  local guard="$GATEWAY_HOST_MEMORY_GUARD_PATH"
  if [ ! -r "$guard" ]; then
    echo "ERROR: gateway host memory guard is unavailable: $guard" >&2
    return 1
  fi
  echo "Clearing only disposable /tmp/prtest pytest processes and checking host memory"
  python3 "$guard" \
    --cleanup-disposable-tests \
    --minimum-available-mib 16384
  python3 "$guard" \
    --cleanup-disposable-tests \
    --minimum-available-mib 4096 \
    --watch-parent "$$" \
    --interval-seconds 5 &
  GATEWAY_HOST_MEMORY_GUARD_PID="$!"
}

run_prepared_gateway_module() {
  (
    cd "$GATEWAY_PREFLIGHT_TREE"
    PYTHONPATH="$GATEWAY_PREFLIGHT_TREE" "$GATEWAY_PYTHON_BIN" -m "$@"
  )
}
trap on_gateway_restart_exit EXIT

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
  export GATEWAY_TEE_EIF_ROOT
  export GATEWAY_STATEFUL_CUTOVER_CEREMONY
  export RESEARCH_LAB_TEE_PROTOCOL
  export GATEWAY_V2_CONFIG_DIR GATEWAY_V2_RELEASE_MANIFEST GATEWAY_V2_ARTIFACT_POLICY
  export GATEWAY_V2_ACCEPTANCE_CORPUS_MANIFEST GATEWAY_V2_ACCEPTANCE_CORPUS_ROOT
  export RESEARCH_LAB_ATTESTED_V2_ARTIFACT_BUCKET
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
  export ATTESTED_RUNTIME_COMMIT_SHA="$GATEWAY_DEPLOY_SHA"
  export ATTESTED_RUNTIME_GIT_REPO_URL="$GATEWAY_DEPLOY_REMOTE"
}

install_successful_restart_script() {
  local source_script="$LEADPOET_REPO_ROOT/gw_restart.sh"
  local target_script="$GATEWAY_HOST_RESTART_SCRIPT"
  local target_dir temporary
  if [ "$(cd "$(dirname "$source_script")" && pwd)/$(basename "$source_script")" = \
      "$(cd "$(dirname "$target_script")" && pwd)/$(basename "$target_script")" ]; then
    return 0
  fi
  target_dir="$(dirname "$target_script")"
  mkdir -p "$target_dir"
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
  docker_storage_counts

  if [ "${IMAGE_COUNT:-0}" -eq 0 ] \
     && [ "${CONTAINER_COUNT:-0}" -eq 0 ] \
     && [ "${VOLUME_COUNT:-0}" -eq 0 ] \
     && { [ "${free_kb_after_prune:-0}" -lt "$MIN_FREE_KB" ] || [ "${DOCKER_ROOT_KB:-0}" -gt 1024 ] || [ "${OVERLAY_DIRS:-0}" -gt 0 ]; }; then
    echo "Detected ${reason} with no tracked Docker objects; resetting full Docker data root"
    echo "docker root usage: ${DOCKER_ROOT_KB:-0} KiB; overlay usage: ${OVERLAY_KB:-0} KiB across ${OVERLAY_DIRS:-0} dirs"
    sudo systemctl stop docker.socket docker 2>/dev/null || true
    sudo rm -rf /var/lib/docker
    sudo mkdir -p /var/lib/docker
    sudo systemctl start docker
    for _ in $(seq 1 20); do
      if sudo docker info >/dev/null 2>&1; then
        break
      fi
      sleep 1
    done
    sudo docker system df 2>/dev/null || true
  fi
}

emergency_disk_preflight() {
  local free_kb
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
  sudo rm -rf /tmp/research-lab-* /tmp/pcr0_builder /tmp/docker-build-* /tmp/buildkit-* 2>/dev/null || true
  sudo docker container prune -f 2>/dev/null || true
  sudo docker builder prune -af 2>/dev/null || true
  sudo docker system prune -af --volumes 2>/dev/null || true

  local free_kb_after_prune
  free_kb_after_prune="$(root_free_kb)"
  reset_orphaned_docker_storage_if_needed "$free_kb_after_prune" "orphaned Docker storage after emergency cleanup"

  echo "Disk after emergency cleanup"
  df -h / /var/lib/docker 2>/dev/null || df -h /
}

stop_research_lab_private_model_containers() {
  local ids
  ids="$(
    sudo docker ps --format '{{.ID}} {{.Image}}' 2>/dev/null \
      | awk '$2 ~ /(^|[./])leadpoet\/sourcing-model([:@]|$)/ {print $1}' \
      || true
  )"

  if [ -z "$ids" ]; then
    echo "No running Research Lab private-model containers found"
    return 0
  fi

  echo "Stopping running Research Lab private-model containers before Docker prune"
  echo "$ids" | xargs -r sudo docker stop -t 10
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
  mkdir -p \
    "$(dirname "$GATEWAY_RESTART_LOCK_FILE")" \
    "$(dirname "$GATEWAY_RESTART_RECOVERY_LOCK_FILE")" \
    "$GATEWAY_DEPLOYMENT_DIR"
  chmod 700 "$(dirname "$GATEWAY_RESTART_LOCK_FILE")" "$GATEWAY_DEPLOYMENT_DIR"
  command -v flock >/dev/null 2>&1 || {
    echo "ERROR: flock is required for gateway Git deployments" >&2
    exit 1
  }
  acquire_gateway_restart_lock
  export GATEWAY_RESTART_LOCK_HELD=1
elif [ "$GATEWAY_RESTART_PHASE" = "post_activate" ]; then
  if [ "${GATEWAY_RESTART_LOCK_HELD:-0}" != "1" ] || [ ! -e "/proc/$$/fd/9" ]; then
    echo "ERROR: post-activation gateway restart lost the deployment lock" >&2
    exit 1
  fi
else
  echo "ERROR: unsupported GATEWAY_RESTART_PHASE=$GATEWAY_RESTART_PHASE" >&2
  exit 1
fi

if [ "$GATEWAY_RESTART_PHASE" = "prepare" ]; then
cd "$GATEWAY_ROOT"

start_gateway_host_memory_guard

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
restart_only_keys = {"GATEWAY_DEPLOY_COMMIT"}

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
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_SECURITY_TOKEN",
    "AWS_PROFILE",
    "GATEWAY_DEPLOY_COMMIT",
    "LEADPOET_REPO_ROOT",
    "GATEWAY_ROOT",
    "GATEWAY_LOG_ROOT",
    "GATEWAY_LOG_FILE",
    "GATEWAY_TEE_EIF_ROOT",
    "GATEWAY_TEE_FALLBACK_LOG_DIR",
    "GATEWAY_GIT_HELPER",
    "GATEWAY_RESTART_PHASE",
    "GATEWAY_STATEFUL_CUTOVER_CEREMONY",
    "LEADPOET_RESTART_START_PATH",
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
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_SECURITY_TOKEN",
    "AWS_PROFILE",
    "GATEWAY_DEPLOY_COMMIT",
    "LEADPOET_REPO_ROOT",
    "GATEWAY_ROOT",
    "GATEWAY_LOG_ROOT",
    "GATEWAY_LOG_FILE",
    "GATEWAY_TEE_EIF_ROOT",
    "GATEWAY_TEE_FALLBACK_LOG_DIR",
    "GATEWAY_GIT_HELPER",
    "GATEWAY_RESTART_PHASE",
    "GATEWAY_STATEFUL_CUTOVER_CEREMONY",
    "LEADPOET_RESTART_START_PATH",
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

if [ -f "$GATEWAY_STATEFUL_CUTOVER_MANIFEST" ]; then
  echo "Loading the canonical stateful epoch cutover manifest"
  export LEADPOET_SUBNET_EPOCH_CUTOVER_PATH="$GATEWAY_STATEFUL_CUTOVER_MANIFEST"
  unset LEADPOET_SUBNET_EPOCH_CUTOVER_JSON
  python3 - "$ENV_CLONE" "$GATEWAY_STATEFUL_CUTOVER_MANIFEST" <<'PY'
import shlex
import sys
from pathlib import Path

env_path = Path(sys.argv[1])
manifest_path = sys.argv[2]
cutover_keys = {
    "LEADPOET_SUBNET_EPOCH_CUTOVER_JSON",
    "LEADPOET_SUBNET_EPOCH_CUTOVER_PATH",
}
kept = []
for raw_line in env_path.read_text(encoding="utf-8").splitlines():
    line = raw_line.strip()
    candidate = line[7:].strip() if line.startswith("export ") else line
    key = candidate.split("=", 1)[0].strip() if "=" in candidate else ""
    if key not in cutover_keys:
        kept.append(raw_line)
kept.append(
    "export LEADPOET_SUBNET_EPOCH_CUTOVER_PATH=" + shlex.quote(manifest_path)
)
env_path.write_text("\n".join(kept) + "\n", encoding="utf-8")
PY
fi
if [ "$GATEWAY_STATEFUL_CUTOVER_CEREMONY" = "1" ]; then
  test -s "$GATEWAY_RESTART_START_PATH" || {
    echo "ERROR: one-time cutover restart-start capture is missing" >&2
    exit 1
  }
  export LEADPOET_RESTART_START_PATH="$GATEWAY_RESTART_START_PATH"
  printf 'export LEADPOET_RESTART_START_PATH=%q\n' \
    "$GATEWAY_RESTART_START_PATH" >> "$ENV_CLONE"
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
case "$GATEWAY_STATEFUL_CUTOVER_CEREMONY" in
  0|1) ;;
  *)
    echo "ERROR: GATEWAY_STATEFUL_CUTOVER_CEREMONY must be 0 or 1" >&2
    exit 1
    ;;
esac
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
    --plan-file "$GATEWAY_DEPLOY_PLAN_FILE" \
    --manifest-file "$GATEWAY_DEPLOYMENT_MANIFEST" \
    --last-good-file "$GATEWAY_LAST_GOOD_MANIFEST"
)"
echo "Prepared gateway commit: $PREPARED_GATEWAY_SHA"

echo "Materializing the prepared commit for pre-shutdown V2 tooling"
GATEWAY_PREFLIGHT_TREE="$(mktemp -d /tmp/gateway-v2-preflight.XXXXXX)"
if ! git -C "$LEADPOET_REPO_ROOT" archive "$PREPARED_GATEWAY_SHA" \
    | tar -xf - -C "$GATEWAY_PREFLIGHT_TREE"; then
  echo "ERROR: unable to materialize the prepared commit for V2 preflight" >&2
  exit 1
fi

RESTART_GATE_ARGS=(
  --network "${BITTENSOR_NETWORK:-finney}"
  --netuid "${BITTENSOR_NETUID:-71}"
)
if [ "$GATEWAY_STATEFUL_CUTOVER_CEREMONY" = "1" ]; then
  echo "Validating the official restart start captured at operator invocation"
  RESTART_GATE_ARGS+=(--captured-report "$GATEWAY_RESTART_START_PATH")
else
  echo "Capturing the official subnet restart window before release acquisition"
fi
if ! run_prepared_gateway_module Leadpoet.utils.restart_epoch_gate \
    "${RESTART_GATE_ARGS[@]}"; then
  echo "Gateway remains running; production shutdown has not started." >&2
  exit 75
fi

echo "Acquiring the independently built V2 release channel"
V2_RELEASE_READY=0
for attempt in $(seq 1 300); do
  if run_prepared_gateway_module gateway.tee.release_channel_v2 \
      --ensure \
      --expected-commit "$PREPARED_GATEWAY_SHA" \
      --bucket "$GATEWAY_V2_RELEASE_BUCKET" \
      --prefix "$GATEWAY_V2_RELEASE_PREFIX" \
      --gateway-output "$GATEWAY_V2_RELEASE_MANIFEST"; then
    V2_RELEASE_READY=1
    break
  fi
  echo "Approved V2 release is not published yet; waiting inside the valid restart invocation (${attempt}/300)"
  sleep 12
done
if [ "$V2_RELEASE_READY" != "1" ]; then
  echo "ERROR: independently approved V2 release is not published for $PREPARED_GATEWAY_SHA" >&2
  echo "Gateway remains running; production shutdown has not started." >&2
  exit 75
fi

echo "Preparing commit-bound KMS credential envelopes"
if ! run_prepared_gateway_module gateway.tee.prepare_gateway_envelopes_v2 \
    --install \
    --env-file "$ENV_CLONE" \
    --kms-key-id "$GATEWAY_V2_KMS_KEY_ID" \
    --deploy-commit "$PREPARED_GATEWAY_SHA" \
    --output-dir "$GATEWAY_V2_CONFIG_DIR"; then
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

echo "Validating the prepared V2 release before production shutdown"
  GATEWAY_DEPLOY_STAGE="v2_pre_shutdown_preflight"
  export GATEWAY_DEPLOY_STAGE
  V2_PREFLIGHT_CREDENTIAL_ARGS=()
  for envelope in "${V2_CREDENTIAL_ENVELOPES[@]}"; do
    V2_PREFLIGHT_CREDENTIAL_ARGS+=(--credential-envelope "$envelope")
  done
  if ! run_prepared_gateway_module gateway.tee.restart_preflight_v2 \
      --deploy-commit "$PREPARED_GATEWAY_SHA" \
      --release-manifest "$GATEWAY_V2_RELEASE_MANIFEST" \
      --topology-manifest "$GATEWAY_PREFLIGHT_TREE/gateway/tee/topology.json" \
      --artifact-policy "$GATEWAY_V2_ARTIFACT_POLICY" \
      --config-dir "$GATEWAY_V2_CONFIG_DIR" \
      --parent-env-file "$ENV_CLONE" \
      --acceptance-corpus-manifest "$GATEWAY_V2_ACCEPTANCE_CORPUS_MANIFEST" \
      --acceptance-corpus-root "$GATEWAY_V2_ACCEPTANCE_CORPUS_ROOT" \
      --topology-mode "${GATEWAY_TEE_TOPOLOGY_MODE:-full}" \
      "${V2_PREFLIGHT_CREDENTIAL_ARGS[@]}"; then
    rm -rf "$GATEWAY_PREFLIGHT_TREE"
    echo "ERROR: prepared V2 release failed before-shutdown validation" >&2
    exit 1
  fi
  echo "Preparing exact hash-locked V2 build artifacts before production shutdown"
  GATEWAY_DEPLOY_STAGE="v2_offline_artifact_prepare"
  export GATEWAY_DEPLOY_STAGE
  if ! (
      cd "$GATEWAY_PREFLIGHT_TREE"
      GATEWAY_V2_OFFLINE_ARTIFACT_ROOT="$GATEWAY_V2_OFFLINE_ARTIFACT_ROOT" \
        bash "$GATEWAY_PREFLIGHT_TREE/gateway/tee/prepare_offline_artifacts_v2.sh"
    ); then
    echo "ERROR: V2 offline artifact preparation failed before shutdown" >&2
    exit 1
  fi

if [ "$GATEWAY_STATEFUL_CUTOVER_CEREMONY" = "1" ]; then
  echo "Validating the one-time receipt-backed cutover before production shutdown"
  GATEWAY_DEPLOY_STAGE="stateful_epoch_cutover_preflight"
  export GATEWAY_DEPLOY_STAGE
  if [ ! -s "$GATEWAY_STATEFUL_CUTOVER_VALIDATOR_RELEASE_MANIFEST" ]; then
    echo "ERROR: cutover validator V2 release manifest is unavailable" >&2
    exit 1
  fi
  PYTHONPATH="$GATEWAY_PREFLIGHT_TREE" "$GATEWAY_PYTHON_BIN" - \
    "$GATEWAY_STATEFUL_CUTOVER_VALIDATOR_RELEASE_MANIFEST" <<'PY'
import sys

from gateway.research_lab.stateful_epoch_candidate_ingest_cli_v1 import (
    load_validator_release_manifest_v2,
)

load_validator_release_manifest_v2(sys.argv[1])
print("Cutover validator V2 release manifest is valid")
PY
  CUTOVER_PREFLIGHT_REPORT="$(
    export SUPABASE_TIMEOUT_SECONDS="$GATEWAY_STATEFUL_CUTOVER_SUPABASE_TIMEOUT_SECONDS"
    run_prepared_gateway_module \
      gateway.research_lab.stateful_epoch_cutover_cli_v1 \
      --release-manifest "$GATEWAY_V2_RELEASE_MANIFEST" \
      --validator-release-manifest "$GATEWAY_STATEFUL_CUTOVER_VALIDATOR_RELEASE_MANIFEST" \
      --use-attested-historical-predecessor
  )"
  printf '%s\n' "$CUTOVER_PREFLIGHT_REPORT"
  CUTOVER_PREFLIGHT_REPORT="$CUTOVER_PREFLIGHT_REPORT" \
    "$GATEWAY_PYTHON_BIN" - <<'PY'
import json
import os

report = json.loads(os.environ["CUTOVER_PREFLIGHT_REPORT"])
status = str(report.get("status") or "")
if status not in {
    "eligible",
    "already_stateful_staged",
    "already_stateful_active",
}:
    raise SystemExit(
        "stateful epoch cutover is not eligible before production shutdown"
    )
if status == "eligible":
    if report.get("predecessor_kind") != "legacy_finalized_chain_migration_v2":
        raise SystemExit("stateful epoch cutover selected an unexpected predecessor")
    if report.get("would_write") is not False:
        raise SystemExit("stateful epoch cutover preflight was not read-only")
else:
    authority = str(report.get("cutover_authority_hash") or "")
    if not authority.startswith("sha256:") or len(authority) != 71:
        raise SystemExit("durable stateful epoch authority hash is invalid")
PY
fi

rm -rf "$GATEWAY_PREFLIGHT_TREE"
GATEWAY_PREFLIGHT_TREE=""

echo "Stopping existing gateway and Research Lab worker processes"
sudo systemctl stop leadpoet-tee-egress-forwarder.service 2>/dev/null || true
sudo systemctl reset-failed leadpoet-tee-egress-forwarder.service 2>/dev/null || true
pkill -9 -f "python3 main.py" 2>/dev/null || true
pkill -9 -f "python3 -u main.py" 2>/dev/null || true
pkill -9 -f "python3 -u -m gateway.main" 2>/dev/null || true
pkill -9 -f "uvicorn" 2>/dev/null || true
pkill -9 -f "gateway/research_lab/worker_process.py" 2>/dev/null || true
pkill -9 -f "run_research_lab_hosted_worker" 2>/dev/null || true
pkill -9 -f "run_research_lab_scoring_worker" 2>/dev/null || true
pkill -9 -f "gateway.research_lab.provider_evidence_proxy" 2>/dev/null || true
pkill -9 -f "provider_evidence_proxy" 2>/dev/null || true
pkill -9 -f "gateway.utils.tee_inter_enclave_relay" 2>/dev/null || true
pkill -9 -f "gateway.utils.tee_egress_forwarder" 2>/dev/null || true
stop_research_lab_private_model_containers

echo "Stopping stuck private-model Docker builds or pip installs"
sudo pkill -TERM -f "docker build -f .*/validator_models/containerizing/Dockerfile" 2>/dev/null || true
sudo pkill -TERM -f "pip install --no-cache-dir -r requirements.txt" 2>/dev/null || true
sleep 3
sudo pkill -KILL -f "docker build -f .*/validator_models/containerizing/Dockerfile" 2>/dev/null || true
sudo pkill -KILL -f "pip install --no-cache-dir -r requirements.txt" 2>/dev/null || true
sleep 2

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

GATEWAY_DEPLOY_STAGE="restart_reexec"
export GATEWAY_DEPLOY_STAGE
unset GATEWAY_DEPLOY_COMMIT
exec env \
  GATEWAY_RESTART_PHASE=post_activate \
  GATEWAY_STATEFUL_CUTOVER_CEREMONY="$GATEWAY_STATEFUL_CUTOVER_CEREMONY" \
  GATEWAY_RESTART_LOCK_HELD=1 \
  LEADPOET_REPO_ROOT="$LEADPOET_REPO_ROOT" \
  GATEWAY_ROOT="$GATEWAY_ROOT" \
  GATEWAY_LOG_ROOT="$GATEWAY_LOG_ROOT" \
  GATEWAY_LOG_FILE="$GATEWAY_LOG_FILE" \
  GATEWAY_ENV_FILE="$GATEWAY_ENV_FILE" \
  GATEWAY_GIT_HELPER="$GATEWAY_GIT_HELPER" \
  GATEWAY_DEPLOY_PLAN_FILE="$GATEWAY_DEPLOY_PLAN_FILE" \
  GATEWAY_DEPLOYMENT_DIR="$GATEWAY_DEPLOYMENT_DIR" \
  GATEWAY_DEPLOYMENT_MANIFEST="$GATEWAY_DEPLOYMENT_MANIFEST" \
  GATEWAY_LAST_GOOD_MANIFEST="$GATEWAY_LAST_GOOD_MANIFEST" \
  GATEWAY_HOST_RESTART_SCRIPT="$GATEWAY_HOST_RESTART_SCRIPT" \
  GATEWAY_TEE_EIF_ROOT="$GATEWAY_TEE_EIF_ROOT" \
  GATEWAY_PYTHON_BIN="$GATEWAY_PYTHON_BIN" \
  RESEARCH_LAB_TEE_PROTOCOL="$RESEARCH_LAB_TEE_PROTOCOL" \
  GATEWAY_V2_CONFIG_DIR="$GATEWAY_V2_CONFIG_DIR" \
  GATEWAY_V2_RELEASE_MANIFEST="$GATEWAY_V2_RELEASE_MANIFEST" \
  GATEWAY_V2_ARTIFACT_POLICY="$GATEWAY_V2_ARTIFACT_POLICY" \
  RESEARCH_LAB_ATTESTED_V2_ARTIFACT_BUCKET="$RESEARCH_LAB_ATTESTED_V2_ARTIFACT_BUCKET" \
  GATEWAY_V2_OFFLINE_ARTIFACT_ROOT="$GATEWAY_V2_OFFLINE_ARTIFACT_ROOT" \
  VALIDATOR_V2_OFFLINE_ARTIFACT_ROOT="$VALIDATOR_V2_OFFLINE_ARTIFACT_ROOT" \
  GATEWAY_DEPLOY_STAGE="$GATEWAY_DEPLOY_STAGE" \
  bash "$LEADPOET_REPO_ROOT/gw_restart.sh" "$@"
fi

GATEWAY_DEPLOY_SHA="$(deployment_field target_sha)"
GATEWAY_DEPLOY_BRANCH="$(deployment_field branch)"
GATEWAY_DEPLOY_REMOTE="$(deployment_field remote_url)"
if [ "$(git -C "$LEADPOET_REPO_ROOT" rev-parse HEAD)" != "$GATEWAY_DEPLOY_SHA" ]; then
  echo "ERROR: canonical gateway checkout does not match activated deployment" >&2
  exit 1
fi
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
df -h / /var/lib/docker 2>/dev/null || df -h /
sudo journalctl --vacuum-size=200M 2>/dev/null || true
sudo rm -rf /tmp/research-lab-* /tmp/pcr0_builder /tmp/docker-build-* /tmp/buildkit-* 2>/dev/null || true

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

echo "Resetting gateway PCR0 builder checkout/cache"
sudo rm -rf /tmp/pcr0_builder

echo "Deleting validator-base:v1 and Docker build cache so PCR0 builder independently rebuilds it"
sudo docker rmi -f validator-base:v1 2>/dev/null || true
sudo docker builder prune -af

echo "Loading gateway runtime env for AWS/ECR checks"
GATEWAY_DEPLOY_STAGE="runtime_env_and_ecr"
export GATEWAY_DEPLOY_STAGE
set -a
. "$ENV_CLONE"
set +a
enforce_deployment_environment
validate_runtime_secret_paths
export AWS_REGION="${AWS_REGION:-us-east-1}"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"
export RESEARCH_LAB_PRIVATE_MODEL_MANIFEST_URI="${RESEARCH_LAB_PRIVATE_MODEL_MANIFEST_URI:-s3://leadpoet-private-model-artifacts-493765492819/research-lab/sourcing-model/current.json}"
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
sudo rm -f "$GATEWAY_TEE_EIF_ROOT"/enclave-build-*.json
rm -f "$GATEWAY_ROOT/tee/tee-enclave.eif"
sudo docker rmi tee-enclave:latest 2>/dev/null || true
bash "$GATEWAY_ROOT/tee/stage_attested_runtime.sh"

# Preflight: verify the worker import graph against the freshly staged
# attested runtime BEFORE building the enclave or relaunching anything.
# A gateway/ tree that imports names the staged top-level packages do not
# export would otherwise crash-loop every worker on its next respawn
# (2026-07-09 incident: config.py imported a constant an unstaged
# _attested_runtime/leadpoet_verifier/economics.py did not have).
echo "Preflight: importing host workers from the canonical Git checkout"
GATEWAY_DEPLOY_STAGE="worker_import_preflight"
export GATEWAY_DEPLOY_STAGE
if ! LEADPOET_REPO_ROOT="$LEADPOET_REPO_ROOT" PYTHONPATH="$LEADPOET_REPO_ROOT" "$GATEWAY_PYTHON_BIN" - <<'PREFLIGHT_HOST'
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
    "gateway.research_lab.worker_process",
    "gateway.research_lab.config",
    "research_lab.code_editing",
    "leadpoet_verifier.economics",
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
  echo "ERROR: host worker import preflight FAILED against canonical Git checkout." >&2
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
importlib.import_module("gateway.research_lab.code_loop_engine")
importlib.import_module("gateway.research_lab.config")
for module_name in (
    "research_lab.code_editing",
    "leadpoet_verifier.economics",
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
  echo "Cleaning temporary role Docker images/layers before gateway relaunch"
  for role in gateway_coordinator gateway_scoring gateway_autoresearch; do
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

  echo "Starting parent-side opaque enclave egress forwarder"
  cd "$LEADPOET_REPO_ROOT"
  PYTHONPATH="$LEADPOET_REPO_ROOT" setsid "$GATEWAY_PYTHON_BIN" -u -m gateway.utils.tee_egress_forwarder \
    >> "$GATEWAY_LOG_ROOT/tee_egress_forwarder.log" 2>&1 < /dev/null 9>&- &
  TEE_EGRESS_FORWARDER_PID="$!"
  sleep 2
  if ! ps -p "$TEE_EGRESS_FORWARDER_PID" >/dev/null 2>&1; then
    tail -80 "$GATEWAY_LOG_ROOT/tee_egress_forwarder.log" || true
    echo "ERROR: parent-side enclave egress forwarder did not start" >&2
    exit 1
  fi

  echo "Starting opaque inter-enclave TLS relay"
  cd "$LEADPOET_REPO_ROOT"
  PYTHONPATH="$LEADPOET_REPO_ROOT" setsid "$GATEWAY_PYTHON_BIN" -m gateway.utils.tee_inter_enclave_relay \
    >> "$GATEWAY_LOG_ROOT/inter_enclave_relay.log" 2>&1 < /dev/null 9>&- &
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
    echo "ERROR: approved V2 release manifest is missing" >&2
    exit 1
  }
  test -s "$GATEWAY_V2_ARTIFACT_POLICY" || {
    echo "ERROR: encrypted V2 artifact policy is missing" >&2
    exit 1
  }
  echo "Verifying encrypted TLS proxy profiles for all V2 workers"
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
    "${V2_BOOTSTRAP_ARGS[@]}" \
    --protected-workflow-manifest "$GATEWAY_ROOT/_attested_runtime/protected_workflows.json" \
    --encrypted-artifact-policy "$GATEWAY_V2_ARTIFACT_POLICY" \
    --config-dir "$GATEWAY_V2_CONFIG_DIR"

  echo "Provisioning KMS ciphertext directly to the attested coordinator"
  GATEWAY_DEPLOY_STAGE="v2_kms_provision"
  export GATEWAY_DEPLOY_STAGE
  PYTHONPATH="$LEADPOET_REPO_ROOT" "$GATEWAY_PYTHON_BIN" -m gateway.utils.tee_kms_provision_v2 \
    "${V2_PROVISION_ARGS[@]}"

echo "Verifying V2 provider and execution-manager readiness"
GATEWAY_DEPLOY_STAGE="v2_runtime_readiness"
export GATEWAY_DEPLOY_STAGE
PYTHONPATH="$LEADPOET_REPO_ROOT" "$GATEWAY_PYTHON_BIN" -m gateway.tee.verify_v2_runtime_ready

if [ "$GATEWAY_STATEFUL_CUTOVER_CEREMONY" = "1" ]; then
  echo "Executing the one-time receipt-backed stateful epoch cutover"
  GATEWAY_DEPLOY_STAGE="stateful_epoch_cutover"
  export GATEWAY_DEPLOY_STAGE
  CUTOVER_MAPPING_HASH="$(
    PYTHONPATH="$LEADPOET_REPO_ROOT" "$GATEWAY_PYTHON_BIN" - <<'PY'
from Leadpoet.utils.subnet_epoch import load_subnet_epoch_cutover

print(load_subnet_epoch_cutover().mapping_hash)
PY
  )"
  CUTOVER_STAGE_REPORT="$(
    cd "$LEADPOET_REPO_ROOT"
    export SUPABASE_TIMEOUT_SECONDS="$GATEWAY_STATEFUL_CUTOVER_SUPABASE_TIMEOUT_SECONDS"
    PYTHONPATH="$LEADPOET_REPO_ROOT" "$GATEWAY_PYTHON_BIN" \
      -m gateway.research_lab.stateful_epoch_cutover_cli_v1 \
      --release-manifest "$GATEWAY_V2_RELEASE_MANIFEST" \
      --validator-release-manifest "$GATEWAY_STATEFUL_CUTOVER_VALIDATOR_RELEASE_MANIFEST" \
      --apply \
      --use-attested-historical-predecessor \
      --confirm-mapping-hash "$CUTOVER_MAPPING_HASH" \
      --confirm-all-writers-stopped
  )"
  printf '%s\n' "$CUTOVER_STAGE_REPORT"
  read -r CUTOVER_STAGE_STATUS CUTOVER_AUTHORITY_HASH < <(
    CUTOVER_STAGE_REPORT="$CUTOVER_STAGE_REPORT" \
      "$GATEWAY_PYTHON_BIN" - <<'PY'
import json
import os

report = json.loads(os.environ["CUTOVER_STAGE_REPORT"])
status = str(report.get("status") or "")
authority = str(report.get("cutover_authority_hash") or "")
if status not in {
    "stateful_staged",
    "already_stateful_staged",
    "already_stateful_active",
}:
    raise SystemExit("stateful epoch cutover staging did not reach a durable state")
if not authority.startswith("sha256:") or len(authority) != 71:
    raise SystemExit("stateful epoch cutover authority hash is invalid")
print(status, authority)
PY
  )
  if [ "$CUTOVER_STAGE_STATUS" != "already_stateful_active" ]; then
    CUTOVER_ACTIVATION_REPORT="$(
      cd "$LEADPOET_REPO_ROOT"
      export SUPABASE_TIMEOUT_SECONDS="$GATEWAY_STATEFUL_CUTOVER_SUPABASE_TIMEOUT_SECONDS"
      PYTHONPATH="$LEADPOET_REPO_ROOT" "$GATEWAY_PYTHON_BIN" \
        -m gateway.research_lab.stateful_epoch_cutover_cli_v1 \
        --release-manifest "$GATEWAY_V2_RELEASE_MANIFEST" \
        --validator-release-manifest "$GATEWAY_STATEFUL_CUTOVER_VALIDATOR_RELEASE_MANIFEST" \
        --activate-staged \
        --confirm-mapping-hash "$CUTOVER_MAPPING_HASH" \
        --confirm-cutover-authority-hash "$CUTOVER_AUTHORITY_HASH" \
        --confirm-all-writers-stopped \
        --confirm-stateful-release-prepared
    )"
    printf '%s\n' "$CUTOVER_ACTIVATION_REPORT"
    CUTOVER_ACTIVATION_REPORT="$CUTOVER_ACTIVATION_REPORT" \
      "$GATEWAY_PYTHON_BIN" - <<'PY'
import json
import os

report = json.loads(os.environ["CUTOVER_ACTIVATION_REPORT"])
if report.get("status") != "stateful_active":
    raise SystemExit("stateful epoch cutover activation did not become active")
PY
  fi
  echo "Stateful epoch cutover is active; continuing the normal V2 restart"
  unset LEADPOET_RESTART_START_PATH
  sed -i '/^export LEADPOET_RESTART_START_PATH=/d' "$ENV_CLONE"
  rm -f "$GATEWAY_RESTART_START_PATH"
fi

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
enforce_deployment_environment
validate_runtime_secret_paths
export PATH="$HOME/.local/bin:$PATH"
export AWS_REGION="${AWS_REGION:-us-east-1}"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"
export GATEWAY_ENV_FILE="${GATEWAY_ENV_FILE:-/home/ec2-user/.config/leadpoet/gateway.env}"
export LEADPOET_GATEWAY_ENV_SECRET_ID="${LEADPOET_GATEWAY_ENV_SECRET_ID:-leadpoet/prod/gateway/env}"
export RESEARCH_LAB_PRIVATE_MODEL_MANIFEST_URI="${RESEARCH_LAB_PRIVATE_MODEL_MANIFEST_URI:-s3://leadpoet-private-model-artifacts-493765492819/research-lab/sourcing-model/current.json}"
unset RESEARCH_LAB_EVIDENCE_PROXY_URL RESEARCH_LAB_PROVIDER_OUTCOME_SIDECAR_PATH
unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_PROFILE AWS_SESSION_TOKEN AWS_SECURITY_TOKEN

echo "Repairing and verifying the authoritative V2 validator weight input"
GATEWAY_DEPLOY_STAGE="validator_weight_input_repair"
export GATEWAY_DEPLOY_STAGE
(
  cd "$LEADPOET_REPO_ROOT"
  PYTHONPATH="$LEADPOET_REPO_ROOT" "$GATEWAY_PYTHON_BIN" \
    -m gateway.tee.verify_weight_submission_ready_v2 --repair
)

cd "$LEADPOET_REPO_ROOT"
setsid "$GATEWAY_PYTHON_BIN" -u -m gateway.main > "$GATEWAY_LOG_FILE" 2>&1 < /dev/null 9>&- &

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
if ! timeout 60 curl -fsS http://localhost:8000/health/v2-authority >/dev/null; then
  tail -160 "$GATEWAY_LOG_FILE"
  echo "ERROR: authoritative V2 enclave/worker readiness failed" >&2
  exit 1
fi
echo "Verifying the exact HTTP handoff consumed by automatic validator weights"
GATEWAY_DEPLOY_STAGE="validator_weight_input_http_check"
export GATEWAY_DEPLOY_STAGE
PYTHONPATH="$LEADPOET_REPO_ROOT" "$GATEWAY_PYTHON_BIN" \
  -m gateway.tee.verify_weight_submission_ready_v2 \
  --gateway-url http://localhost:8000

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

timeout 30 curl -fsS http://localhost:8000/research-lab/status >/dev/null
timeout 30 curl -fsS http://localhost:8000/attest >/dev/null

GATEWAY_DEPLOY_STAGE="host_restart_script_install"
export GATEWAY_DEPLOY_STAGE
install_successful_restart_script

GATEWAY_DEPLOY_STAGE="completed"
export GATEWAY_DEPLOY_STAGE
finalize_deployment_record succeeded "$GATEWAY_DEPLOY_STAGE" >/dev/null
GATEWAY_DEPLOY_COMPLETED=1
rm -f "$GATEWAY_DEPLOY_PLAN_FILE"
echo "Gateway restart command completed; tail logs with: tail -f $GATEWAY_LOG_FILE"
