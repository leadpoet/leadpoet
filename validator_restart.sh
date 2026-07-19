#!/bin/bash
set -euo pipefail

VALIDATOR_ROOT="${VALIDATOR_ROOT:-/home/ec2-user/leadpoet/leadpoet}"
VALIDATOR_ENV_FILE="${VALIDATOR_ENV_FILE:-/home/ec2-user/.config/leadpoet/validator.env}"
LEADPOET_VALIDATOR_ENV_SECRET_ID="${LEADPOET_VALIDATOR_ENV_SECRET_ID:-leadpoet/prod/validator/env}"
VALIDATOR_ENV_BACKUP_DIR="${VALIDATOR_ENV_BACKUP_DIR:-/home/ec2-user/.config/leadpoet/env-backups}"
EXPECTED_AWS_ACCOUNT="${EXPECTED_AWS_ACCOUNT:-493765492819}"
# Interpreter for the long-lived validator process. The hydrated environment
# can select the existing production venv without changing restart behavior.
VALIDATOR_PYTHON_BIN="${VALIDATOR_PYTHON_BIN:-python3}"
VALIDATOR_V2_GATEWAY_RELEASE_MANIFEST="${VALIDATOR_V2_GATEWAY_RELEASE_MANIFEST:-/home/ec2-user/.config/leadpoet/gateway-v2-release-manifest.json}"
VALIDATOR_V2_RELEASE_MANIFEST="${VALIDATOR_V2_RELEASE_MANIFEST:-/home/ec2-user/.config/leadpoet/validator-v2-release-manifest.json}"
VALIDATOR_V2_RELEASE_ARCHIVE_ROOT="${VALIDATOR_V2_RELEASE_ARCHIVE_ROOT:-/home/ec2-user/.config/leadpoet/validator-releases-v2}"
VALIDATOR_V2_HOTKEY_CONFIG="${VALIDATOR_V2_HOTKEY_CONFIG:-/home/ec2-user/.config/leadpoet/validator-hotkey-config-v2.json}"
VALIDATOR_V2_HOTKEY_ENVELOPE="${VALIDATOR_V2_HOTKEY_ENVELOPE:-/home/ec2-user/.config/leadpoet/validator-hotkey-envelope-v2.json}"
VALIDATOR_V2_RELEASE_BUCKET="${VALIDATOR_V2_RELEASE_BUCKET:-leadpoet-attested-v2-artifacts-493765492819}"
VALIDATOR_V2_RELEASE_PREFIX="${VALIDATOR_V2_RELEASE_PREFIX:-attested-v2/releases}"
export VALIDATOR_V2_OFFLINE_ARTIFACT_ROOT="${VALIDATOR_V2_OFFLINE_ARTIFACT_ROOT:-$HOME/.cache/leadpoet-v2-artifacts/validator-runtime}"
VALIDATOR_WALLET_ROOT="${VALIDATOR_WALLET_ROOT:-$HOME/.bittensor/wallets}"
VALIDATOR_WALLET_NAME="${VALIDATOR_WALLET_NAME:-validator_72}"
VALIDATOR_WALLET_HOTKEY="${VALIDATOR_WALLET_HOTKEY:-default}"
VALIDATOR_ENV_EXPORT="$(mktemp /tmp/validator_env_export.XXXXXX)"
SECRET_TMP="$(mktemp /tmp/validator_secret_env.XXXXXX)"

cleanup() {
  rm -f "$VALIDATOR_ENV_EXPORT" "$SECRET_TMP"
}
trap cleanup EXIT

cd "$VALIDATOR_ROOT"

echo "Preflight: preserving tracked local validator checkout changes if present"
if ! git diff --quiet || ! git diff --cached --quiet; then
  restart_stash_message="pre-validator-restart-local-tracked-$(date -u +%Y%m%dT%H%M%SZ)"
  git stash push -m "$restart_stash_message" -- .
  echo "Preserved tracked local changes in Git stash: $restart_stash_message"
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

required_keys=(
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
echo "Acquiring the independently built V2 release channel"
if ! python3 -m gateway.tee.release_channel_v2 \
    --ensure \
    --expected-commit "$VALIDATOR_DEPLOY_SHA" \
    --bucket "$VALIDATOR_V2_RELEASE_BUCKET" \
    --prefix "$VALIDATOR_V2_RELEASE_PREFIX" \
    --gateway-output "$VALIDATOR_V2_GATEWAY_RELEASE_MANIFEST" \
    --validator-output "$VALIDATOR_V2_RELEASE_MANIFEST"; then
  echo "ERROR: independently approved V2 release is not published for $VALIDATOR_DEPLOY_SHA" >&2
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
        "independent_gateway_and_validator_parent_build_evidence",
        "verified_validator_hotkey_envelope_and_offline_custody",
    ],
}, sort_keys=True, indent=2))
PY
  echo "Validator remains untouched. Complete the V2 bootstrap ceremony, then rerun this restart." >&2
  exit 75
fi

echo "Refreshing public validator hotkey measurements for the selected release"
python3 -m validator_tee.host.refresh_hotkey_config_v2 \
  --config "$VALIDATOR_V2_HOTKEY_CONFIG" \
  --chain-profile \
    "$VALIDATOR_ROOT/validator_tee/enclave/chain_signing_profile_v2.json" \
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
echo "Validating the exact validator V2 release before production shutdown"
python3 -m validator_tee.host.restart_preflight_v2 \
  --deploy-commit "$VALIDATOR_DEPLOY_SHA" \
  --validator-release "$VALIDATOR_V2_RELEASE_MANIFEST" \
  --gateway-release "$VALIDATOR_V2_GATEWAY_RELEASE_MANIFEST" \
  --hotkey-config "$VALIDATOR_V2_HOTKEY_CONFIG" \
  --hotkey-envelope "$VALIDATOR_V2_HOTKEY_ENVELOPE" \
  --runtime-artifact-lock "$VALIDATOR_ROOT/validator_tee/runtime-artifacts-v2.lock.json" \
  --host-hotkey-directory "$HOST_HOTKEY_DIR"

actual_aws_account="$(aws sts get-caller-identity --query Account --output text)"
if [ "$actual_aws_account" != "$EXPECTED_AWS_ACCOUNT" ]; then
  echo "ERROR: validator AWS account is $actual_aws_account, expected $EXPECTED_AWS_ACCOUNT"
  exit 1
fi

echo "Verifying official subnet restart window immediately before shutdown"
if ! "$VALIDATOR_PYTHON_BIN" -m Leadpoet.utils.restart_epoch_gate \
    --network "${VALIDATOR_SUBTENSOR_NETWORK:-finney}" \
    --netuid "${VALIDATOR_NETUID:-71}"; then
  echo "Validator remains running; production shutdown has not started." >&2
  exit 75
fi

echo "Stopping validator processes and containers"
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
    >> "$CHAIN_RELAY_LOG" 2>&1 < /dev/null &
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
    --hotkey-config "$VALIDATOR_V2_HOTKEY_CONFIG"

  echo "Provisioning the validator hotkey directly into Nitro with KMS"
python3 -m validator_tee.host.hotkey_bootstrap_v2 \
  --hotkey-config "$VALIDATOR_V2_HOTKEY_CONFIG" \
  --hotkey-envelope "$VALIDATOR_V2_HOTKEY_ENVELOPE"

echo "Starting validator"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="${PYTHONPATH:-$VALIDATOR_ROOT}"
export VALIDATOR_FORCE_CONTAINER_DEPLOY=1
export VALIDATOR_AUTO_CONTAINER_FOLLOW_LOGS=0

"$VALIDATOR_PYTHON_BIN" neurons/validator.py \
  --netuid "${VALIDATOR_NETUID:-71}" \
  --subtensor_network "${VALIDATOR_SUBTENSOR_NETWORK:-finney}" \
  --wallet_name "$VALIDATOR_WALLET_NAME" \
  --wallet_hotkey "$VALIDATOR_WALLET_HOTKEY"

if [ "$(docker inspect -f '{{.State.Running}}' leadpoet-validator-main)" != "true" ] \
    || [ "$(docker inspect -f '{{.RestartCount}}' leadpoet-validator-main)" != "0" ]; then
  echo "ERROR: validator coordinator failed its final restart-wrapper check" >&2
  docker logs --tail 160 leadpoet-validator-main >&2 || true
  exit 1
fi
echo "SUCCESS: authoritative V2 validator restart completed and verified"
