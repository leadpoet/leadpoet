#!/bin/bash
# Canonical normal-Arena validator restart. Run from the installed N-1 checkout.
set -euo pipefail
umask 077
export PYTHONDONTWRITEBYTECODE=1

SOURCE_ROOT="${VALIDATOR_ROOT:-/home/ec2-user/leadpoet/leadpoet}"
RELEASE_ROOT="${VALIDATOR_RELEASE_ROOT:-/home/ec2-user/leadpoet/validator-releases}"
CURRENT_LINK="${VALIDATOR_CURRENT_LINK:-/home/ec2-user/leadpoet/validator-current}"
ENV_FILE="${VALIDATOR_ENV_FILE:-/home/ec2-user/.config/leadpoet/arena-validator.env}"
RUNTIME_ENV="${VALIDATOR_RUNTIME_ENV:-/home/ec2-user/.config/leadpoet/arena-validator-runtime.env}"
SERVICE_ENV="${VALIDATOR_SERVICE_ENV:-/home/ec2-user/.config/leadpoet/arena-validator-service.env}"
MANIFEST="${VALIDATOR_ARENA_SIGNER_MANIFEST:-/home/ec2-user/.config/leadpoet/arena-signer/manifest.json}"
EIF_FILE="${VALIDATOR_ARENA_SIGNER_EIF:-/home/ec2-user/.config/leadpoet/arena-signer/validator-enclave.eif}"
POLICY_FILE="${VALIDATOR_ARENA_SIGNER_POLICY:-/home/ec2-user/.config/leadpoet/arena-signer/arena_signer_policy.json}"
LEGACY_ENVELOPE="${VALIDATOR_LEGACY_HOTKEY_ENVELOPE:-/home/ec2-user/.config/leadpoet/validator-hotkey-envelope-v2.json}"
MIGRATION_KMS_KEY_ID="${VALIDATOR_ARENA_MIGRATION_KMS_KEY_ID:-arn:aws:kms:us-east-1:493765492819:key/6822c852-4bf2-4b2a-be0b-91a61057f92d}"
SERVICE="${VALIDATOR_SERVICE_NAME:-leadpoet-arena-validator.service}"
UNIT_PATH="${VALIDATOR_SERVICE_UNIT_PATH:-/etc/systemd/system/$SERVICE}"
PYTHON="${VALIDATOR_PYTHON_BIN:-/usr/bin/python3}"
READY_TIMEOUT="${VALIDATOR_READY_TIMEOUT_SECONDS:-90}"
STOP_TIMEOUT="${VALIDATOR_STOP_TIMEOUT_SECONDS:-9300}"
LOCK_FILE="${VALIDATOR_RESTART_LOCK_FILE:-/home/ec2-user/.config/leadpoet/arena-validator-restart.lock}"
TARGET_REQUEST="${VALIDATOR_DEPLOY_COMMIT:-origin/main}"
CANDIDATE_ENCLAVE_ID=""
ACTIVATED=0
SERVICE_STARTED=0
SERVICE_ATTEMPTED=0
STAGE=""
CANDIDATE_SERVICE_ENV=""
SERVICE_ENV_BACKUP=""
SERVICE_ENV_PROMOTED=0
RUNTIME_ENV_BACKUP=""
RUNTIME_ENV_PROMOTED=0

fail() { echo "ERROR: $*" >&2; exit 1; }
cleanup() {
  status=$?
  if [ "$status" -ne 0 ] && [ "$SERVICE_ATTEMPTED" -eq 1 ] && [ "$ACTIVATED" -eq 0 ]; then
    # Cancel Restart=on-failure before removing the signer it owns. This also
    # makes a failed activation safe to retry from the same release metadata.
    sudo systemctl stop "$SERVICE" >/dev/null 2>&1 || true
    sudo systemctl reset-failed "$SERVICE" >/dev/null 2>&1 || true
  fi
  if [ "$status" -ne 0 ] && [ "$SERVICE_ENV_PROMOTED" -eq 1 ] && [ "$ACTIVATED" -eq 0 ]; then
    if [ -n "$SERVICE_ENV_BACKUP" ]; then
      sudo mv -f "$SERVICE_ENV_BACKUP" "$SERVICE_ENV" || true
      SERVICE_ENV_BACKUP=""
    else
      sudo rm -f -- "$SERVICE_ENV" || true
    fi
  fi
  if [ "$status" -ne 0 ] && [ "$RUNTIME_ENV_PROMOTED" -eq 1 ] && [ "$ACTIVATED" -eq 0 ]; then
    if [ -n "$RUNTIME_ENV_BACKUP" ]; then
      sudo mv -f "$RUNTIME_ENV_BACKUP" "$RUNTIME_ENV" || true
      RUNTIME_ENV_BACKUP=""
    else
      sudo rm -f -- "$RUNTIME_ENV" || true
    fi
  fi
  if [ "$status" -ne 0 ] && [ "$ACTIVATED" -eq 0 ] && [ -n "$CANDIDATE_ENCLAVE_ID" ]; then
    sudo nitro-cli terminate-enclave --enclave-id "$CANDIDATE_ENCLAVE_ID" >/dev/null 2>&1 || true
  fi
  [ -z "$STAGE" ] || rm -rf -- "$STAGE"
  [ -z "$CANDIDATE_SERVICE_ENV" ] || sudo rm -f -- "$CANDIDATE_SERVICE_ENV"
  exit "$status"
}
trap cleanup EXIT

mkdir -p "$(dirname "$LOCK_FILE")" "$RELEASE_ROOT"
exec 9>"$LOCK_FILE"
flock -n 9 || fail "another validator restart owns the controller lock"
cd "$SOURCE_ROOT"
[ -d .git ] || fail "installed N-1 validator checkout is unavailable"
git diff --quiet && git diff --cached --quiet || fail "installed checkout has tracked changes"
git fetch origin --prune
TARGET_SHA="$(git rev-parse --verify "$TARGET_REQUEST^{commit}")"
[[ "$TARGET_SHA" =~ ^[0-9a-f]{40}$ ]] || fail "target commit is invalid"
git merge-base --is-ancestor "$TARGET_SHA" origin/main || fail "target commit is not on origin/main"

STAGE="$(mktemp -d "$RELEASE_ROOT/.candidate.XXXXXX")"
git archive --format=tar "$TARGET_SHA" | tar -xf - -C "$STAGE"
printf '%s\n' "$TARGET_SHA" > "$STAGE/.release-commit"
test -r "$STAGE/validator_restart.sh" && test -r "$STAGE/deploy/leadpoet-arena-validator.service" || fail "candidate release is incomplete"
RELEASE="$RELEASE_ROOT/$TARGET_SHA"
if [ -d "$RELEASE" ]; then
  diff -qr "$STAGE" "$RELEASE" >/dev/null || fail "existing release directory differs from exact Git source"
else
  mv "$STAGE" "$RELEASE"; STAGE=""
fi

# Freeze a root-owned candidate configuration without replacing the active
# service snapshot. It is promoted only after the old validator has drained.
CANDIDATE_SERVICE_ENV="${SERVICE_ENV}.candidate.$$"
[ "$(stat -c %u "$ENV_FILE")" = "$(id -u)" ] || fail "operator Arena environment owner differs from restart identity"
[ "$(stat -c %a "$ENV_FILE")" = "600" ] || fail "operator Arena environment must have mode 0600"
sudo install -m 0600 -o root -g root "$ENV_FILE" "$CANDIDATE_SERVICE_ENV"
[ "$(sudo stat -c %u "$CANDIDATE_SERVICE_ENV")" = 0 ] || fail "candidate service environment is not root-owned"
[ "$(sudo stat -c %a "$CANDIDATE_SERVICE_ENV")" = 600 ] || fail "candidate service environment is not private"

# Create only the two durable work roots named by the private configuration.
# Their contents are never replaced during a release switch.
read -r STATE_PATH RUNNER_PATH < <(
  cd "$RELEASE"
  PYTHONPATH="$RELEASE" "$PYTHON" - "$ENV_FILE" <<'PY'
import os,sys
from pathlib import Path
from scripts.run_arena_validator import load_environment
load_environment(Path(sys.argv[1]))
print(os.environ.get("LAB_ARENA_VALIDATOR_STATE_DIR", ""), os.environ.get("LAB_ARENA_RUNNER_WORK_DIR", ""))
PY
)
for durable in "$STATE_PATH" "$RUNNER_PATH"; do
  [[ "$durable" = /* ]] || fail "Arena durable directory is not absolute"
  sudo install -d -m 0700 -o root -g root "$durable"
  [ -d "$durable" ] && [ ! -L "$durable" ] || fail "Arena durable directory is unsafe"
done

# Validate all durable inputs before starting or stopping anything.
( cd "$RELEASE"; sudo env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$RELEASE" "$PYTHON" - "$CANDIDATE_SERVICE_ENV" "$POLICY_FILE" "$MANIFEST" "$EIF_FILE" <<'PY'
import hashlib,json,os,stat,sys
from pathlib import Path
from scripts.run_arena_validator import load_environment
from validator_tee.enclave.arena_hotkey import validate_policy
from leadpoet_canonical.lab_arena_rewards import sha256_json
env,policy_path,manifest_path,eif_path=map(Path,sys.argv[1:])
load_environment(env)
for path in (policy_path,manifest_path,eif_path):
    if not path.is_file() or path.is_symlink(): raise SystemExit("Arena signer input must be a regular non-symlink file")
required=("LAB_ARENA_VALIDATOR_STATE_DIR","LAB_ARENA_RUNNER_WORK_DIR","LAB_ARENA_RUNSC_PATH")
for name in required:
    if not os.environ.get(name,"").strip(): raise SystemExit("missing required Arena setting: "+name)
state=Path(os.environ["LAB_ARENA_VALIDATOR_STATE_DIR"])
if not state.is_absolute() or not state.is_dir() or state.is_symlink(): raise SystemExit("Arena validator state directory must already exist")
if state.stat().st_mode & 0o077: raise SystemExit("Arena validator state directory must be private")
runsc=Path(os.environ["LAB_ARENA_RUNSC_PATH"])
if not runsc.is_absolute() or not runsc.is_file() or runsc.is_symlink() or not os.access(runsc,os.X_OK): raise SystemExit("Arena runsc must be an executable regular file")
policy=validate_policy(json.loads(policy_path.read_text()))
envelope_value=os.environ.get("LAB_ARENA_HOTKEY_ENVELOPE", "").strip()
if envelope_value:
    envelope=Path(envelope_value)
    if not envelope.is_file() or envelope.is_symlink() or envelope.stat().st_mode & 0o077: raise SystemExit("normalized Arena envelope must be a private regular file")
    envelope_doc=json.loads(envelope.read_text())
    schema=envelope_doc.get("schema_version")
    if schema == "leadpoet.arena.hotkey_envelope.v1":
        if envelope_doc.get("policy") != policy or envelope_doc.get("policy_hash") != sha256_json(policy): raise SystemExit("Arena envelope differs from measured public policy")
    elif schema != "leadpoet.validator_hotkey_envelope.v2":
        raise SystemExit("Arena hotkey envelope schema is unsupported")
manifest=json.loads(manifest_path.read_text())
if set(manifest) != {"schema_version","eif_sha256","pcr0","policy_sha256"} or manifest["schema_version"] != "leadpoet.arena.signer_manifest.v1": raise SystemExit("Arena signer manifest is invalid")
digest=hashlib.sha256(eif_path.read_bytes()).hexdigest()
policy_bytes=json.dumps(policy,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()
if manifest["eif_sha256"] != "sha256:"+digest or manifest["policy_sha256"] != "sha256:"+hashlib.sha256(policy_bytes).hexdigest(): raise SystemExit("Arena signer artifact binding differs")
print(manifest["pcr0"])
PY
)
EXPECTED_PCR0="$("$PYTHON" - "$MANIFEST" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))["pcr0"])
PY
)"
[[ "$EXPECTED_PCR0" =~ ^[0-9a-fA-F]{96}$ ]] || fail "Arena signer PCR0 is invalid"
OBSERVED_PCR0="$(sudo nitro-cli describe-eif --eif-path "$EIF_FILE" | "$PYTHON" -c 'import json,sys; print(json.load(sys.stdin)["Measurements"]["PCR0"])')"
[ "${OBSERVED_PCR0,,}" = "${EXPECTED_PCR0,,}" ] || fail "Arena signer EIF measurement mismatch"

service_active=0
sudo systemctl is-active --quiet "$SERVICE" && service_active=1 || true
service_pid="$(sudo systemctl show -p MainPID --value "$SERVICE" 2>/dev/null || true)"
[[ "$service_pid" =~ ^[0-9]+$ ]] || service_pid=0
old_cid=""
if [ "$service_active" -eq 1 ]; then
  sudo test -r "$RUNTIME_ENV" || fail "active validator lacks its enclave identity"
  [ "$(sudo stat -c %u "$RUNTIME_ENV")" = 0 ] && [ "$(sudo stat -c %a "$RUNTIME_ENV")" = 600 ] || fail "active validator runtime identity is not root-private"
  old_cid="$(sudo sed -n 's/^ENCLAVE_CID=//p' "$RUNTIME_ENV")"
  [[ "$old_cid" =~ ^[0-9]+$ ]] || fail "active validator enclave identity is invalid"
fi
read -r discovered_cid old_enclave_id < <(sudo nitro-cli describe-enclaves | "$PYTHON" -c '
import json,sys
owned=[x for x in json.load(sys.stdin) if x.get("State")=="RUNNING" and (x.get("EnclaveName")=="validator-enclave" or str(x.get("EnclaveName") or "").startswith(("arena-signer-","arena-validator-")))]
if len(owned)>1: raise SystemExit("validator enclave ownership is ambiguous")
print(owned[0].get("EnclaveCID", "") if owned else "", owned[0].get("EnclaveID", "") if owned else "")
') || fail "validator enclave ownership is ambiguous"
if [ -n "$old_cid" ] && [ "$old_cid" != "$discovered_cid" ]; then fail "service signer differs from the owned enclave"; fi
old_cid="${old_cid:-$discovered_cid}"
# Select a free deterministic CID so routine restarts can alternate signers.
CANDIDATE_CID="$(sudo nitro-cli describe-enclaves | "$PYTHON" -c '
import json,sys
used={int(x["EnclaveCID"]) for x in json.load(sys.stdin) if x.get("State")=="RUNNING"}
print(next((cid for cid in range(19,32) if cid not in used), ""))
')"
[[ "$CANDIDATE_CID" =~ ^[0-9]+$ ]] || fail "no candidate enclave CID is available"

# Adopt only the exact legacy validator from the installed checkout. A process
# from another checkout or more than one owner is ambiguous and fails closed.
read -r old_pid old_start < <(
  sudo env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$RELEASE" "$PYTHON" -m validator_tee.host.arena_restart_identity \
    "$SOURCE_ROOT" "$CURRENT_LINK" "$service_pid"
) || fail "validator process ownership is ambiguous"

run_json="$(sudo nitro-cli run-enclave --eif-path "$EIF_FILE" --cpu-count "${VALIDATOR_ENCLAVE_CPU_COUNT:-2}" --memory "${VALIDATOR_ENCLAVE_MEMORY_MIB:-1024}" --enclave-cid "$CANDIDATE_CID" --enclave-name "arena-signer-${TARGET_SHA:0:12}")"
CANDIDATE_ENCLAVE_ID="$(printf '%s' "$run_json" | "$PYTHON" -c 'import json,sys; print(json.load(sys.stdin)["EnclaveID"])')"
[ -n "$CANDIDATE_ENCLAVE_ID" ] || fail "candidate signer did not return an enclave identity"
# First transition and later cold boots use a narrow recipient-only migration;
# no raw seed or general signing surface reaches the host.
if ! PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$RELEASE" ENCLAVE_CID="$CANDIDATE_CID" "$PYTHON" -c 'from validator_tee.host.vsock_client import ValidatorEnclaveClient; raise SystemExit(0 if ValidatorEnclaveClient().get_arena_hotkey_state_v1().get("provisioned") else 1)' 2>/dev/null; then
  [ -f "$LEGACY_ENVELOPE" ] && [ ! -L "$LEGACY_ENVELOPE" ] || fail "legacy encrypted hotkey envelope is unavailable"
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$RELEASE" ENCLAVE_CID="$CANDIDATE_CID" "$PYTHON" -m validator_tee.host.arena_hotkey_bootstrap migrate-legacy \
    --legacy-envelope "$LEGACY_ENVELOPE" --policy "$POLICY_FILE" --kms-key-id "$MIGRATION_KMS_KEY_ID"
fi
sudo timeout "$READY_TIMEOUT" env PYTHONDONTWRITEBYTECODE=1 "$PYTHON" "$RELEASE/scripts/run_arena_validator.py" --environment-file "$CANDIDATE_SERVICE_ENV" --enclave-cid "$CANDIDATE_CID" --check-only

# Only now drain the exact old supervised process. No broad process kill or
# transaction submission is allowed here.
if [ "$service_active" -eq 1 ]; then
  timeout "$STOP_TIMEOUT" sudo systemctl stop "$SERVICE"
elif [ -n "$old_pid" ]; then
  observed="$(awk '{print $22}' "/proc/$old_pid/stat" 2>/dev/null || true)"
  [ "$observed" = "$old_start" ] || fail "legacy validator identity changed before drain"
  kill -TERM "$old_pid"
  deadline=$((SECONDS + STOP_TIMEOUT))
  while kill -0 "$old_pid" 2>/dev/null && [ "$SECONDS" -lt "$deadline" ]; do sleep 1; done
  ! kill -0 "$old_pid" 2>/dev/null || fail "legacy validator did not drain before timeout"
fi
if ss -H -ltn '( sport = :5002 or sport = :5003 )' | grep -q .; then fail "a validator relay remains after service stop"; fi

ln -sfn "$RELEASE" "$CURRENT_LINK.new"
mv -Tf "$CURRENT_LINK.new" "$CURRENT_LINK"
if sudo test -e "$SERVICE_ENV"; then
  SERVICE_ENV_BACKUP="${SERVICE_ENV}.previous.$$"
  sudo mv "$SERVICE_ENV" "$SERVICE_ENV_BACKUP"
fi
sudo mv -f "$CANDIDATE_SERVICE_ENV" "$SERVICE_ENV"
CANDIDATE_SERVICE_ENV=""
if sudo test -f "$SERVICE_ENV"; then :; else fail "candidate service environment promotion failed"; fi
SERVICE_ENV_PROMOTED=1
runtime_tmp="$RUNTIME_ENV.tmp.$$"
printf 'ENCLAVE_CID=%s\n' "$CANDIDATE_CID" > "$runtime_tmp"
chmod 600 "$runtime_tmp"
if sudo test -e "$RUNTIME_ENV"; then
  RUNTIME_ENV_BACKUP="${RUNTIME_ENV}.previous.$$"
  sudo mv "$RUNTIME_ENV" "$RUNTIME_ENV_BACKUP"
fi
sudo install -m 0600 -o root -g root "$runtime_tmp" "$RUNTIME_ENV"
RUNTIME_ENV_PROMOTED=1
rm -f "$runtime_tmp"
sudo install -m 0644 "$RELEASE/deploy/leadpoet-arena-validator.service" "$UNIT_PATH"
sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE" >/dev/null
SERVICE_ATTEMPTED=1
sudo systemctl start "$SERVICE"
SERVICE_STARTED=1

deadline=$((SECONDS + READY_TIMEOUT))
while [ "$SECONDS" -lt "$deadline" ]; do
  main_pid="$(sudo systemctl show -p MainPID --value "$SERVICE")"
  if sudo systemctl is-active --quiet "$SERVICE" && [[ "$main_pid" =~ ^[1-9][0-9]*$ ]] && ss -H -ltn '( sport = :5002 or sport = :5003 )' | awk 'END{exit NR==2?0:1}'; then
    ACTIVATED=1
    break
  fi
  sleep 2
done
[ "$ACTIVATED" -eq 1 ] || fail "normal Arena validator did not become ready"
[ -z "$SERVICE_ENV_BACKUP" ] || sudo rm -f -- "$SERVICE_ENV_BACKUP"
SERVICE_ENV_BACKUP=""
[ -z "$RUNTIME_ENV_BACKUP" ] || sudo rm -f -- "$RUNTIME_ENV_BACKUP"
RUNTIME_ENV_BACKUP=""
[ -z "$old_enclave_id" ] || sudo nitro-cli terminate-enclave --enclave-id "$old_enclave_id" >/dev/null
CANDIDATE_ENCLAVE_ID=""
echo "SUCCESS: normal Arena validator is supervised at exact commit $TARGET_SHA"
