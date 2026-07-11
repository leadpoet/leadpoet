#!/bin/bash
set -euo pipefail

GATEWAY_GIT_DEPLOY_PROTOCOL="1"
LEADPOET_REPO_ROOT="${LEADPOET_REPO_ROOT:-/home/ec2-user/leadpoet_repo}"
GATEWAY_ROOT="${GATEWAY_ROOT:-$LEADPOET_REPO_ROOT/gateway}"
GATEWAY_LOG_ROOT="${GATEWAY_LOG_ROOT:-/home/ec2-user/gateway}"
GATEWAY_LOG_FILE="${GATEWAY_LOG_FILE:-$GATEWAY_LOG_ROOT/gateway.log}"
GATEWAY_ENV_FILE="${GATEWAY_ENV_FILE:-/home/ec2-user/.config/leadpoet/gateway.env}"
LEADPOET_GATEWAY_ENV_SECRET_ID="${LEADPOET_GATEWAY_ENV_SECRET_ID:-leadpoet/prod/gateway/env}"
ENV_CLONE="/tmp/gw_env_clone.sh"
ENV_SECRET="/tmp/gw_env_secret.sh"
MIN_FREE_KB=$((10 * 1024 * 1024))
EXPECTED_AWS_ACCOUNT="493765492819"
ENV_BACKUP_DIR="/home/ec2-user/.config/leadpoet/env-backups"
GATEWAY_GIT_HELPER="${GATEWAY_GIT_HELPER:-$LEADPOET_REPO_ROOT/scripts/gateway_git_deploy.py}"
GATEWAY_RESTART_PHASE="${GATEWAY_RESTART_PHASE:-prepare}"
GATEWAY_RESTART_LOCK_FILE="${GATEWAY_RESTART_LOCK_FILE:-/home/ec2-user/.config/leadpoet/gateway-restart.lock}"
GATEWAY_DEPLOY_PLAN_FILE="${GATEWAY_DEPLOY_PLAN_FILE:-/tmp/gateway_git_deploy.$$.json}"
GATEWAY_DEPLOYMENT_DIR="${GATEWAY_DEPLOYMENT_DIR:-/home/ec2-user/.config/leadpoet/deployments}"
GATEWAY_DEPLOYMENT_MANIFEST="${GATEWAY_DEPLOYMENT_MANIFEST:-$GATEWAY_DEPLOYMENT_DIR/gateway-current.json}"
GATEWAY_LAST_GOOD_MANIFEST="${GATEWAY_LAST_GOOD_MANIFEST:-$GATEWAY_DEPLOYMENT_DIR/gateway-last-good.json}"
GATEWAY_HOST_RESTART_SCRIPT="${GATEWAY_HOST_RESTART_SCRIPT:-/home/ec2-user/gw_restart.sh}"
GATEWAY_TEE_EIF_ROOT="${GATEWAY_TEE_EIF_ROOT:-/home/ec2-user/tee}"
GATEWAY_DEPLOY_STAGE="${GATEWAY_DEPLOY_STAGE:-bootstrap}"
GATEWAY_DEPLOY_COMPLETED=0

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
  if [ "$status" -ne 0 ] \
      && [ "$GATEWAY_DEPLOY_COMPLETED" != "1" ] \
      && [ -f "$GATEWAY_DEPLOY_PLAN_FILE" ] \
      && [ -f "$GATEWAY_GIT_HELPER" ]; then
    finalize_deployment_record failed "$GATEWAY_DEPLOY_STAGE" >/dev/null 2>&1 || true
  fi
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

if [ "$GATEWAY_RESTART_PHASE" = "prepare" ]; then
  mkdir -p "$(dirname "$GATEWAY_RESTART_LOCK_FILE")" "$GATEWAY_DEPLOYMENT_DIR"
  chmod 700 "$(dirname "$GATEWAY_RESTART_LOCK_FILE")" "$GATEWAY_DEPLOYMENT_DIR"
  command -v flock >/dev/null 2>&1 || {
    echo "ERROR: flock is required for gateway Git deployments" >&2
    exit 1
  }
  exec 9>"$GATEWAY_RESTART_LOCK_FILE"
  chmod 600 "$GATEWAY_RESTART_LOCK_FILE"
  if ! flock -n 9; then
    echo "ERROR: another gateway restart is already running" >&2
    exit 1
  fi
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
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
raw = src.read_text()

try:
    parsed = json.loads(raw)
except Exception:
    parsed = None

if isinstance(parsed, dict):
    lines = []
    for key, value in parsed.items():
        if isinstance(value, (dict, list)):
            value = json.dumps(value, separators=(",", ":"))
        elif value is None:
            value = ""
        lines.append(f"{key}={value}")
    raw = "\n".join(lines) + "\n"

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
    "LEADPOET_REPO_ROOT",
    "GATEWAY_ROOT",
    "GATEWAY_LOG_ROOT",
    "GATEWAY_LOG_FILE",
    "GATEWAY_TEE_EIF_ROOT",
    "GATEWAY_TEE_FALLBACK_LOG_DIR",
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
    "LEADPOET_REPO_ROOT",
    "GATEWAY_ROOT",
    "GATEWAY_LOG_ROOT",
    "GATEWAY_LOG_FILE",
    "GATEWAY_TEE_EIF_ROOT",
    "GATEWAY_TEE_FALLBACK_LOG_DIR",
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

grep -q "SUPABASE_SERVICE_ROLE_KEY" "$ENV_CLONE" || {
  echo "ERROR: hydrated/cloned env missing SUPABASE_SERVICE_ROLE_KEY"
  exit 1
}

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

echo "Stopping existing gateway and Research Lab worker processes"
pkill -9 -f "python3 main.py" 2>/dev/null || true
pkill -9 -f "python3 -u main.py" 2>/dev/null || true
pkill -9 -f "python3 -u -m gateway.main" 2>/dev/null || true
pkill -9 -f "uvicorn" 2>/dev/null || true
pkill -9 -f "gateway/research_lab/worker_process.py" 2>/dev/null || true
pkill -9 -f "run_research_lab_hosted_worker" 2>/dev/null || true
pkill -9 -f "run_research_lab_scoring_worker" 2>/dev/null || true
pkill -9 -f "gateway.research_lab.provider_evidence_proxy" 2>/dev/null || true
pkill -9 -f "provider_evidence_proxy" 2>/dev/null || true
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
exec env \
  GATEWAY_RESTART_PHASE=post_activate \
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
if ! LEADPOET_REPO_ROOT="$LEADPOET_REPO_ROOT" PYTHONPATH="$LEADPOET_REPO_ROOT" python3 - <<'PREFLIGHT_HOST'
import importlib
import os
from pathlib import Path

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
if ! GATEWAY_ROOT="$GATEWAY_ROOT" LEADPOET_REPO_ROOT="$LEADPOET_REPO_ROOT" python3 - <<'PREFLIGHT_ATTESTED'
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
sudo docker build --no-cache -f "$GATEWAY_ROOT/tee/Dockerfile.enclave" -t tee-enclave:latest "$GATEWAY_ROOT/"
ENCLAVE_BUILD_METADATA_TMP="$(mktemp /tmp/gateway-enclave-build.XXXXXX.json)"
set +e
sudo nitro-cli build-enclave --docker-uri tee-enclave:latest --output-file "$GATEWAY_TEE_EIF_ROOT/tee-enclave.eif" > "$ENCLAVE_BUILD_METADATA_TMP" 2>/dev/null
ENCLAVE_BUILD_STATUS="$?"
set -e
echo "Cleaning temporary enclave Docker image/layers before gateway relaunch"
sudo docker rmi -f tee-enclave:latest 2>/dev/null || true
sudo docker builder prune -af 2>/dev/null || true
df -h / /var/lib/docker 2>/dev/null || df -h /
if [ "$ENCLAVE_BUILD_STATUS" -ne 0 ]; then
  rm -f "$ENCLAVE_BUILD_METADATA_TMP"
  echo "ERROR: nitro-cli build-enclave failed with status $ENCLAVE_BUILD_STATUS"
  exit "$ENCLAVE_BUILD_STATUS"
fi
sudo install -m 644 "$ENCLAVE_BUILD_METADATA_TMP" "$GATEWAY_TEE_EIF_ROOT/enclave-build-gateway.json"
rm -f "$ENCLAVE_BUILD_METADATA_TMP"
sudo env \
  GATEWAY_ROOT="$GATEWAY_ROOT" \
  GATEWAY_TEE_EIF_ROOT="$GATEWAY_TEE_EIF_ROOT" \
  GATEWAY_ENV_FILE="$GATEWAY_ENV_FILE" \
  bash ./start_enclave.sh

echo "Installing Python dependencies"
GATEWAY_DEPLOY_STAGE="dependency_install"
export GATEWAY_DEPLOY_STAGE
if ! python3 -m pip --version >/dev/null 2>&1; then
  curl -s https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
  python3 /tmp/get-pip.py --user
  rm /tmp/get-pip.py
fi
export PATH="$HOME/.local/bin:$PATH"
cd "$GATEWAY_ROOT"
python3 -m pip install --user bittensor fastapi uvicorn python-multipart httpx pydantic requests cbor2 cryptography supabase boto3 minio arweave-python-client substrate-interface jsonschema awscli

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
export RESEARCH_LAB_EVIDENCE_PROXY_URL="${RESEARCH_LAB_EVIDENCE_PROXY_URL:-http://172.17.0.1:8791}"
export RESEARCH_LAB_PROVIDER_OUTCOME_SIDECAR_PATH="${RESEARCH_LAB_PROVIDER_OUTCOME_SIDECAR_PATH:-/home/ec2-user/research_lab_evidence/provider_outcomes.json}"
unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_PROFILE AWS_SESSION_TOKEN AWS_SECURITY_TOKEN

echo "Starting Research Lab provider evidence proxy"
mkdir -p /home/ec2-user/research_lab_evidence "$GATEWAY_LOG_ROOT"
cd "$LEADPOET_REPO_ROOT"
setsid python3 -m gateway.research_lab.provider_evidence_proxy \
  --host 172.17.0.1 \
  --port 8791 \
  --day-cache /home/ec2-user/research_lab_evidence/day_cache.json \
  --outcome-sidecar "$RESEARCH_LAB_PROVIDER_OUTCOME_SIDECAR_PATH" \
  >> "$GATEWAY_LOG_ROOT/provider_evidence_proxy.log" 2>&1 < /dev/null 9>&- &
PROVIDER_PROXY_PID="$!"
echo "relaunched provider evidence proxy pid: $PROVIDER_PROXY_PID"
for i in $(seq 1 10); do
  if ss -ltn "sport = :8791" 2>/dev/null | grep -q ":8791"; then
    echo "provider evidence proxy listening on :8791 after ${i}s"
    break
  fi
  sleep 1
done
if ! ss -ltn "sport = :8791" 2>/dev/null | grep -q ":8791"; then
  echo "ERROR: provider evidence proxy did not start on :8791"
  tail -80 "$GATEWAY_LOG_ROOT/provider_evidence_proxy.log" || true
  exit 1
fi

cd "$LEADPOET_REPO_ROOT"
setsid python3 -u -m gateway.main > "$GATEWAY_LOG_FILE" 2>&1 < /dev/null 9>&- &

GATEWAY_PID="$!"
echo "relaunched main pid: $GATEWAY_PID"
rm -f "$ENV_CLONE" "$ENV_SECRET"

sleep 240
GATEWAY_DEPLOY_STAGE="gateway_health_check"
export GATEWAY_DEPLOY_STAGE
if ! ps -p "$GATEWAY_PID" >/dev/null 2>&1; then
  tail -120 "$GATEWAY_LOG_FILE"
  exit 1
fi
if ! timeout 15 curl -fsS http://localhost:8000/health >/dev/null; then
  tail -120 "$GATEWAY_LOG_FILE"
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

curl -fsS http://localhost:8000/research-lab/status || true
curl -fsS http://localhost:8000/attest || true

GATEWAY_DEPLOY_STAGE="host_restart_script_install"
export GATEWAY_DEPLOY_STAGE
install_successful_restart_script

GATEWAY_DEPLOY_STAGE="completed"
export GATEWAY_DEPLOY_STAGE
finalize_deployment_record succeeded "$GATEWAY_DEPLOY_STAGE" >/dev/null
GATEWAY_DEPLOY_COMPLETED=1
rm -f "$GATEWAY_DEPLOY_PLAN_FILE"
echo "Gateway restart command completed; tail logs with: tail -f $GATEWAY_LOG_FILE"
