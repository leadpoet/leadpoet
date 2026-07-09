#!/bin/bash
set -euo pipefail

GATEWAY_ROOT="/home/ec2-user/gateway"
GATEWAY_ENV_FILE="${GATEWAY_ENV_FILE:-/home/ec2-user/.config/leadpoet/gateway.env}"
LEADPOET_GATEWAY_ENV_SECRET_ID="${LEADPOET_GATEWAY_ENV_SECRET_ID:-leadpoet/prod/gateway/env}"
ENV_CLONE="/tmp/gw_env_clone.sh"
ENV_SECRET="/tmp/gw_env_secret.sh"
MIN_FREE_KB=$((10 * 1024 * 1024))
EXPECTED_AWS_ACCOUNT="493765492819"
ENV_BACKUP_DIR="/home/ec2-user/.config/leadpoet/env-backups"

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

cd "$GATEWAY_ROOT"

PID="$(pgrep -f "python3 -u main.py" | head -1 || true)"
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

echo "Stopping existing gateway and Research Lab worker processes"
pkill -9 -f "python3 main.py" 2>/dev/null || true
pkill -9 -f "python3 -u main.py" 2>/dev/null || true
pkill -9 -f "uvicorn" 2>/dev/null || true
pkill -9 -f "gateway/research_lab/worker_process.py" 2>/dev/null || true
pkill -9 -f "run_research_lab_hosted_worker" 2>/dev/null || true
pkill -9 -f "run_research_lab_scoring_worker" 2>/dev/null || true
pkill -9 -f "gateway.research_lab.provider_evidence_proxy" 2>/dev/null || true
pkill -9 -f "provider_evidence_proxy" 2>/dev/null || true

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

echo "Clearing Python caches"
cd "$GATEWAY_ROOT"
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -delete 2>/dev/null || true
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
rm -rf ~/.cache/Python* 2>/dev/null || true
find ~/.local/lib/python3.9/site-packages -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

echo "Preflight disk cleanup for Docker/PCR0/Research Lab builds"
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
set -a
. "$ENV_CLONE"
set +a
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
chmod +x "$GATEWAY_ROOT"/tee/*.sh || true
cd "$GATEWAY_ROOT/tee"
sudo mkdir -p /home/ec2-user/tee
rm -f "$GATEWAY_ROOT/tee/tee-enclave.eif"
sudo docker rmi tee-enclave:latest 2>/dev/null || true
bash "$GATEWAY_ROOT/tee/stage_attested_runtime.sh"

# Preflight: verify the worker import graph against the freshly staged
# attested runtime BEFORE building the enclave or relaunching anything.
# A gateway/ tree that imports names the staged top-level packages do not
# export would otherwise crash-loop every worker on its next respawn
# (2026-07-09 incident: config.py imported a constant an unstaged
# _attested_runtime/leadpoet_verifier/economics.py did not have).
echo "Preflight: importing worker module against staged attested runtime"
if ! python3 - <<'PREFLIGHT'
import sys
sys.path.insert(0, "/home/ec2-user/gateway/_attested_runtime")
sys.path.insert(1, "/home/ec2-user")
import importlib
importlib.import_module("gateway.research_lab.worker_process")
importlib.import_module("gateway.research_lab.config")
importlib.import_module("leadpoet_verifier.economics")
print("preflight imports OK")
PREFLIGHT
then
  echo "ERROR: worker import preflight FAILED against staged runtime."
  echo "The gateway/ code and staged top-level packages are out of sync;"
  echo "fix the missing sync (git pull / rsync the top-level packages) and rerun."
  exit 1
fi
sudo docker build --no-cache -f "$GATEWAY_ROOT/tee/Dockerfile.enclave" -t tee-enclave:latest "$GATEWAY_ROOT/"
set +e
sudo nitro-cli build-enclave --docker-uri tee-enclave:latest --output-file /home/ec2-user/tee/tee-enclave.eif >/dev/null 2>&1
ENCLAVE_BUILD_STATUS="$?"
set -e
echo "Cleaning temporary enclave Docker image/layers before gateway relaunch"
sudo docker rmi -f tee-enclave:latest 2>/dev/null || true
sudo docker builder prune -af 2>/dev/null || true
df -h / /var/lib/docker 2>/dev/null || df -h /
if [ "$ENCLAVE_BUILD_STATUS" -ne 0 ]; then
  echo "ERROR: nitro-cli build-enclave failed with status $ENCLAVE_BUILD_STATUS"
  exit "$ENCLAVE_BUILD_STATUS"
fi
sudo ./start_enclave.sh

echo "Installing Python dependencies"
if ! python3 -m pip --version >/dev/null 2>&1; then
  curl -s https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
  python3 /tmp/get-pip.py --user
  rm /tmp/get-pip.py
fi
export PATH="$HOME/.local/bin:$PATH"
cd "$GATEWAY_ROOT"
python3 -m pip install --user bittensor fastapi uvicorn python-multipart httpx pydantic requests cbor2 cryptography supabase boto3 minio arweave-python-client substrate-interface jsonschema awscli

echo "Relaunching gateway with cloned runtime env"
set -a
. "$ENV_CLONE"
set +a
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="/home/ec2-user"
export AWS_REGION="${AWS_REGION:-us-east-1}"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"
export GATEWAY_ENV_FILE="${GATEWAY_ENV_FILE:-/home/ec2-user/.config/leadpoet/gateway.env}"
export LEADPOET_GATEWAY_ENV_SECRET_ID="${LEADPOET_GATEWAY_ENV_SECRET_ID:-leadpoet/prod/gateway/env}"
export RESEARCH_LAB_PRIVATE_MODEL_MANIFEST_URI="${RESEARCH_LAB_PRIVATE_MODEL_MANIFEST_URI:-s3://leadpoet-private-model-artifacts-493765492819/research-lab/sourcing-model/current.json}"
export RESEARCH_LAB_EVIDENCE_PROXY_URL="${RESEARCH_LAB_EVIDENCE_PROXY_URL:-http://172.17.0.1:8791}"
unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_PROFILE AWS_SESSION_TOKEN AWS_SECURITY_TOKEN

echo "Starting Research Lab provider evidence proxy"
mkdir -p /home/ec2-user/research_lab_evidence
cd /home/ec2-user
setsid python3 -m gateway.research_lab.provider_evidence_proxy \
  --host 172.17.0.1 \
  --port 8791 \
  --day-cache /home/ec2-user/research_lab_evidence/day_cache.json \
  >> "$GATEWAY_ROOT/provider_evidence_proxy.log" 2>&1 < /dev/null &
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
  tail -80 "$GATEWAY_ROOT/provider_evidence_proxy.log" || true
  exit 1
fi

cd "$GATEWAY_ROOT"
setsid python3 -u main.py > gateway.log 2>&1 < /dev/null &

GATEWAY_PID="$!"
echo "relaunched main pid: $GATEWAY_PID"
rm -f "$ENV_CLONE" "$ENV_SECRET"

sleep 240
if ! ps -p "$GATEWAY_PID" >/dev/null 2>&1; then
  tail -120 "$GATEWAY_ROOT/gateway.log"
  exit 1
fi
if ! timeout 15 curl -fsS http://localhost:8000/health >/dev/null; then
  tail -120 "$GATEWAY_ROOT/gateway.log"
  exit 1
fi

curl -fsS http://localhost:8000/research-lab/status || true
curl -fsS http://localhost:8000/attest || true
echo "Gateway restart command completed; tail logs with: tail -f $GATEWAY_ROOT/gateway.log"
