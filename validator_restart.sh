#!/bin/bash
set -euo pipefail

VALIDATOR_ROOT="${VALIDATOR_ROOT:-/home/ec2-user/leadpoet/leadpoet}"
VALIDATOR_ENV_FILE="${VALIDATOR_ENV_FILE:-/home/ec2-user/.config/leadpoet/validator.env}"
LEADPOET_VALIDATOR_ENV_SECRET_ID="${LEADPOET_VALIDATOR_ENV_SECRET_ID:-leadpoet/prod/validator/env}"
VALIDATOR_ENV_BACKUP_DIR="${VALIDATOR_ENV_BACKUP_DIR:-/home/ec2-user/.config/leadpoet/env-backups}"
EXPECTED_AWS_ACCOUNT="${EXPECTED_AWS_ACCOUNT:-493765492819}"
VALIDATOR_ENV_EXPORT="$(mktemp /tmp/validator_env_export.XXXXXX)"
SECRET_TMP="$(mktemp /tmp/validator_secret_env.XXXXXX)"

cleanup() {
  rm -f "$VALIDATOR_ENV_EXPORT" "$SECRET_TMP"
}
trap cleanup EXIT

cd "$VALIDATOR_ROOT"

echo "Preflight: preserving local validator.py diff if present"
if ! git diff --quiet -- neurons/validator.py; then
  git stash push -m "pre-prod-validator-local-validator-py-$(date -u +%Y%m%dT%H%M%SZ)" -- neurons/validator.py
fi

echo "Pulling latest GitHub main before stopping validator"
before_head="$(git rev-parse HEAD)"
git fetch origin
git checkout main
git pull --ff-only origin main
after_head="$(git rev-parse HEAD)"
if [ "$before_head" != "$after_head" ] && [ "${VALIDATOR_RESTART_REEXECED:-0}" != "1" ]; then
  echo "Restart wrapper updated from GitHub; re-executing latest validator_restart.sh"
  exec env VALIDATOR_RESTART_REEXECED=1 bash "$VALIDATOR_ROOT/validator_restart.sh" "$@"
fi

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

cache.parent.mkdir(parents=True, exist_ok=True)
cache.write_text(raw)

skip_keys = {
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_SECURITY_TOKEN",
    "AWS_PROFILE",
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

required_keys=(
  ENABLE_TEE_SUBMISSION
  ENABLE_FULFILLMENT
  ENABLE_QUALIFICATION_EVALUATION
  LEADPOET_WRAPPER_ACTIVE
  GATEWAY_URL
  SUPABASE_URL
  SUPABASE_ANON_KEY
  SUPABASE_SERVICE_ROLE_KEY
  OPENROUTER_API_KEY
  QUALIFICATION_OPENROUTER_API_KEY
  FULFILLMENT_OPENROUTER_API_KEY
  EXA_API_KEY
  SCRAPINGDOG_API_KEY
  QUALIFICATION_SCRAPINGDOG_API_KEY
  AWS_REGION
  AWS_DEFAULT_REGION
  RESEARCH_LAB_VALIDATOR_FETCH_ENABLED
  RESEARCH_LAB_VALIDATOR_SHADOW_VERIFY_ENABLED
  RESEARCH_LAB_VALIDATOR_EVALUATION_VERIFY_ENABLED
  RESEARCH_LAB_REQUIRE_SHADOW_VERIFICATION_BEFORE_SUBMIT
  RESEARCH_LAB_REQUIRE_EVALUATION_VERIFICATION_BEFORE_SUBMIT
  RESEARCH_LAB_INTERNAL_API_KEY
  RESEARCH_LAB_SCORE_BUNDLE_KMS_KEY_ID
  RESEARCH_LAB_WEIGHT_MUTATION_ENABLED
  RESEARCH_LAB_SUBMIT_ON_CHAIN_ENABLED
  QUALIFICATION_WEBSHARE_PROXY_1
  NO_PROXY
)

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
unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_PROFILE AWS_SESSION_TOKEN AWS_SECURITY_TOKEN

actual_aws_account="$(aws sts get-caller-identity --query Account --output text)"
if [ "$actual_aws_account" != "$EXPECTED_AWS_ACCOUNT" ]; then
  echo "ERROR: validator AWS account is $actual_aws_account, expected $EXPECTED_AWS_ACCOUNT"
  exit 1
fi

echo "Stopping validator processes/containers/enclave"
sudo pkill -TERM -f ".auto_update_wrapper.sh" 2>/dev/null || true
sudo pkill -TERM -f "neurons/validator.py" 2>/dev/null || true
sudo pkill -TERM -f "docker logs -f leadpoet-validator-main" 2>/dev/null || true
sleep 5

sudo pkill -KILL -f ".auto_update_wrapper.sh" 2>/dev/null || true
sudo pkill -KILL -f "neurons/validator.py" 2>/dev/null || true
sudo pkill -KILL -f "docker logs -f leadpoet-validator-main" 2>/dev/null || true
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

nitro-cli terminate-enclave --all 2>/dev/null || true
sleep 2

docker image prune -f

echo "Deleting validator-base:v1 and Docker build cache so validator independently rebuilds it"
docker rmi -f validator-base:v1 2>/dev/null || true
docker builder prune -af

echo "Building validator enclave"
bash validator_tee/scripts/build_enclave.sh
test -f validator_tee/validator-enclave.eif
cd validator_tee
nitro-cli run-enclave \
  --eif-path validator-enclave.eif \
  --cpu-count "${VALIDATOR_ENCLAVE_CPU_COUNT:-2}" \
  --memory "${VALIDATOR_ENCLAVE_MEMORY_MB:-1024}"
sleep 3
cd "$VALIDATOR_ROOT"

echo "Starting validator"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="${PYTHONPATH:-$VALIDATOR_ROOT}"

python3 neurons/validator.py \
  --netuid "${VALIDATOR_NETUID:-71}" \
  --subtensor_network "${VALIDATOR_SUBTENSOR_NETWORK:-finney}" \
  --wallet_name "${VALIDATOR_WALLET_NAME:-validator_72}" \
  --wallet_hotkey "${VALIDATOR_WALLET_HOTKEY:-default}"
