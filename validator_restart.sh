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
LEGACY_EIF="${VALIDATOR_LEGACY_EIF:-$SOURCE_ROOT/validator_tee/validator-enclave.eif}"
LEGACY_HOTKEY_CONFIG="${VALIDATOR_LEGACY_HOTKEY_CONFIG:-/home/ec2-user/.config/leadpoet/validator-hotkey-config-v2.json}"
LEGACY_RELEASE_MANIFEST="${VALIDATOR_LEGACY_RELEASE_MANIFEST:-/home/ec2-user/.config/leadpoet/validator-v2-release-manifest.json}"
LEGACY_GATEWAY_MANIFEST="${VALIDATOR_LEGACY_GATEWAY_MANIFEST:-/home/ec2-user/.config/leadpoet/gateway-v2-release-manifest.json}"
LEGACY_GATEWAY_LINEAGE="${VALIDATOR_LEGACY_GATEWAY_LINEAGE:-/home/ec2-user/.config/leadpoet/gateway-v2-release-lineage.json}"
MIGRATION_KMS_KEY_ID="${VALIDATOR_ARENA_MIGRATION_KMS_KEY_ID:-arn:aws:kms:us-east-1:493765492819:key/6822c852-4bf2-4b2a-be0b-91a61057f92d}"
SERVICE="${VALIDATOR_SERVICE_NAME:-leadpoet-arena-validator.service}"
UNIT_PATH="${VALIDATOR_SERVICE_UNIT_PATH:-/etc/systemd/system/$SERVICE}"
PYTHON="${VALIDATOR_PYTHON_BIN:-/home/ec2-user/venv311/bin/python3}"
READY_TIMEOUT="${VALIDATOR_READY_TIMEOUT_SECONDS:-90}"
STOP_TIMEOUT="${VALIDATOR_STOP_TIMEOUT_SECONDS:-9300}"
LOCK_FILE="${VALIDATOR_RESTART_LOCK_FILE:-/home/ec2-user/.config/leadpoet/arena-validator-restart.lock}"
TARGET_REQUEST="${VALIDATOR_DEPLOY_COMMIT:-origin/main}"
CANDIDATE_ENCLAVE_ID=""
CANDIDATE_CREATED=0
ACTIVATED=0
SERVICE_STARTED=0
SERVICE_ATTEMPTED=0
STAGE=""
UNIT_STAGE=""
LEGACY_EIF_SNAPSHOT=""
CANDIDATE_SERVICE_ENV=""
SERVICE_ENV_BACKUP=""
SERVICE_ENV_PROMOTED=0
RUNTIME_ENV_BACKUP=""
RUNTIME_ENV_PROMOTED=0
OLD_ENCLAVE_TERMINATED=0
SIGNER_HANDOFF_COMMITTED=0
OLD_ENCLAVE_ID=""
OLD_ENCLAVE_CID=""
OLD_ENCLAVE_NAME=""
OLD_ENCLAVE_CPUS=""
OLD_ENCLAVE_MEMORY=""
OLD_PCR0=""
LEGACY_CONTAINER_STOPPED=0

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
  if [ "$status" -ne 0 ] && [ "$SIGNER_HANDOFF_COMMITTED" -eq 0 ] && [ "$CANDIDATE_CREATED" -eq 1 ] && [ -n "$CANDIDATE_ENCLAVE_ID" ]; then
    sudo nitro-cli terminate-enclave --enclave-id "$CANDIDATE_ENCLAVE_ID" >/dev/null 2>&1 || true
  fi
  if [ "$status" -ne 0 ] && [ "$OLD_ENCLAVE_TERMINATED" -eq 1 ] && [ "$SIGNER_HANDOFF_COMMITTED" -eq 0 ]; then
    recovery=""
    for _ in $(seq 1 10); do
      recovery="$(sudo nitro-cli run-enclave --eif-path "$LEGACY_EIF_SNAPSHOT" --cpu-count "$OLD_ENCLAVE_CPUS" --memory "$OLD_ENCLAVE_MEMORY" --enclave-cid "$OLD_ENCLAVE_CID" --enclave-name "$OLD_ENCLAVE_NAME" 2>/dev/null || true)"
      [ -z "$recovery" ] || break
      sleep 2
    done
    observed=""
    for _ in $(seq 1 15); do
      observed="$(sudo nitro-cli describe-enclaves 2>/dev/null | RECOVERY_CID="$OLD_ENCLAVE_CID" "$PYTHON" -c 'import json,sys,os; cid=int(os.environ["RECOVERY_CID"]); xs=[x for x in json.load(sys.stdin) if x.get("State")=="RUNNING" and x.get("EnclaveCID")==cid]; print(xs[0].get("Measurements",{}).get("PCR0","") if len(xs)==1 else "")' 2>/dev/null || true)"
      [ -z "$observed" ] || break
      sleep 2
    done
    if [ "$observed" = "$OLD_PCR0" ]; then
      legacy_rpc_ready=0
      deadline=$((SECONDS + READY_TIMEOUT))
      while [ "$SECONDS" -lt "$deadline" ]; do
        if ( cd "$SOURCE_ROOT" && ENCLAVE_CID="$OLD_ENCLAVE_CID" PYTHONPATH="$SOURCE_ROOT" "$PYTHON" -c 'from validator_tee.host.vsock_client import ValidatorEnclaveClient; ValidatorEnclaveClient().health_check()' ) >/dev/null 2>&1; then legacy_rpc_ready=1; break; fi
        sleep 1
      done
      if [ "$legacy_rpc_ready" -eq 1 ] && ( cd "$SOURCE_ROOT" && ENCLAVE_CID="$OLD_ENCLAVE_CID" PYTHONPATH="$SOURCE_ROOT" "$PYTHON" -m validator_tee.host.runtime_v2_bootstrap --validator-release "$LEGACY_RELEASE_MANIFEST" --gateway-release "$LEGACY_GATEWAY_MANIFEST" --gateway-release-lineage "$LEGACY_GATEWAY_LINEAGE" --hotkey-config "$LEGACY_HOTKEY_CONFIG" ) >/dev/null 2>&1 \
          && ( cd "$SOURCE_ROOT" && ENCLAVE_CID="$OLD_ENCLAVE_CID" PYTHONPATH="$SOURCE_ROOT" "$PYTHON" -m validator_tee.host.hotkey_bootstrap_v2 --hotkey-config "$LEGACY_HOTKEY_CONFIG" --hotkey-envelope "$LEGACY_ENVELOPE" ) >/dev/null 2>&1; then
        [ "$LEGACY_CONTAINER_STOPPED" -eq 0 ] || sudo docker start "$legacy_container_id" >/dev/null 2>&1 || echo "ERROR: legacy validator container recovery failed" >&2
      else
        echo "ERROR: exact legacy signer reprovisioning failed" >&2
      fi
    else
      echo "ERROR: exact legacy signer recovery failed" >&2
    fi
  fi
  [ -z "$STAGE" ] || rm -rf -- "$STAGE"
  [ -z "${UNIT_STAGE:-}" ] || rm -f -- "$UNIT_STAGE"
  [ -z "$LEGACY_EIF_SNAPSHOT" ] || rm -f -- "$LEGACY_EIF_SNAPSHOT"
  [ -z "$CANDIDATE_SERVICE_ENV" ] || sudo rm -f -- "$CANDIDATE_SERVICE_ENV"
  exit "$status"
}
trap cleanup EXIT

[[ "$PYTHON" =~ ^/[A-Za-z0-9_./-]+$ ]] && [ -x "$PYTHON" ] || fail "validator interpreter must be an executable absolute path"

mkdir -p "$(dirname "$LOCK_FILE")" "$RELEASE_ROOT"
exec 9>"$LOCK_FILE"
flock -n 9 || fail "another validator restart owns the controller lock"
cd "$SOURCE_ROOT"
[ -d .git ] || fail "installed N-1 validator checkout is unavailable"
DOCKER_LOCK_HELPER="$SOURCE_ROOT/validator_tee/scripts/docker_operation_lock_v2.sh"
[ -f "$DOCKER_LOCK_HELPER" ] && [ ! -L "$DOCKER_LOCK_HELPER" ] || fail "canonical validator operation lock is unavailable"
. "$DOCKER_LOCK_HELPER"
leadpoet_acquire_docker_operation_lock_v2
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
  if sudo test -e "$durable" || sudo test -L "$durable"; then
    sudo test -d "$durable" && sudo test ! -L "$durable" || fail "Arena durable directory is unsafe"
  fi
  sudo install -d -m 0700 -o root -g root "$durable"
  sudo test -d "$durable" && sudo test ! -L "$durable" || fail "Arena durable directory is unsafe"
done

# Validate all durable inputs before starting or stopping anything.
( cd "$RELEASE" && sudo env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$RELEASE" "$PYTHON" - "$CANDIDATE_SERVICE_ENV" "$POLICY_FILE" "$MANIFEST" "$EIF_FILE" <<'PY'
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

# Prove the public gateway key and finalized chain before interrupting the old
# signer. Protected identity/capability checks follow after the new EIF boots.
( cd "$RELEASE" && sudo env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$RELEASE" "$PYTHON" - "$CANDIDATE_SERVICE_ENV" "$POLICY_FILE" <<'PY'
import json,os,sys
from pathlib import Path
from scripts.run_arena_validator import load_environment
from validator_tee.enclave.arena_hotkey import validate_policy
from lab_arena.chain import ArenaChain,ArenaChainConfig,connect_substrate
from lab_arena.validator import ArenaPublicApi
load_environment(Path(sys.argv[1])); policy=validate_policy(json.loads(Path(sys.argv[2]).read_text()))
endpoint=os.environ["LAB_ARENA_CHAIN_ENDPOINT"]
timeout=int(os.environ.get("LAB_ARENA_CHAIN_TIMEOUT_SECONDS","30"))
config=ArenaChainConfig(endpoint=endpoint,netuid=int(os.environ["LAB_ARENA_NETUID"]),network_name=os.environ["LAB_ARENA_NETWORK"],request_timeout_seconds=timeout)
if policy["network"]!=config.network_name or int(policy["netuid"])!=config.netuid or policy["chain_profile"]["chain_endpoint"].rstrip("/")!=endpoint.rstrip("/"):
    raise SystemExit("public chain configuration differs from measured policy")
key=ArenaPublicApi(os.environ["LAB_ARENA_API_BASE_URL"]).signing_key()
if key.get("public_key_hash")!=policy["arena_signing_key_hash"]: raise SystemExit("public Arena key differs from measured policy")
chain=ArenaChain(config,connect_substrate(config))
try:
    head=chain.finalized_head(); genesis=str(chain.client.get_block_hash(block_id=0)).lower().removeprefix("0x")
    if genesis!=str(policy["chain_profile"]["genesis_hash"]).lower().removeprefix("0x"): raise SystemExit("finalized genesis differs from measured policy")
    if not chain.refresh_metagraph().hotkeys: raise SystemExit("finalized metagraph is empty")
    print("Arena public preflight is valid at finalized block %d" % head.number)
finally: chain.close()
PY
)

service_active=0
sudo systemctl is-active --quiet "$SERVICE" && service_active=1 || true
service_pid="$(sudo systemctl show -p MainPID --value "$SERVICE" 2>/dev/null || true)"
[[ "$service_pid" =~ ^[0-9]+$ ]] || service_pid=0
read_legacy_inventory_json() {
  local container_ids container_json
  container_ids="$(
    sudo docker container ls -a \
      --filter 'name=^/leadpoet-validator-main$' \
      --filter 'name=^/leadpoet-ff-worker-' \
      --format '{{.ID}}'
  )" || return
  if [ -z "$container_ids" ]; then
    container_json='[]'
  else
    container_json="$(sudo docker inspect $container_ids)" || return
  fi
  cd "$RELEASE" && printf '%s' "$container_json" | SOURCE_ROOT="$SOURCE_ROOT" PYTHONPATH="$RELEASE" "$PYTHON" -c '
import json,os,sys
from pathlib import Path
from validator_tee.host.arena_restart_identity import validate_legacy_container_inventory
print(json.dumps(validate_legacy_container_inventory(json.load(sys.stdin),Path(os.environ["SOURCE_ROOT"])),sort_keys=True,separators=(",",":")))
'
}
legacy_inventory_json="$(read_legacy_inventory_json)" || fail "legacy validator container inventory is invalid"
read -r legacy_container_id legacy_container_pid < <(printf '%s' "$legacy_inventory_json" | "$PYTHON" -c 'import json,sys; x=json.load(sys.stdin); print(x["main_id"],x["main_pid"])')
mapfile -t legacy_worker_container_ids < <(printf '%s' "$legacy_inventory_json" | "$PYTHON" -c 'import json,sys; [print(x["container_id"]) for x in json.load(sys.stdin)["workers"]]')
mapfile -t legacy_worker_pids < <(printf '%s' "$legacy_inventory_json" | "$PYTHON" -c 'import json,sys; [print(x["pid"]) for x in json.load(sys.stdin)["workers"]]')
legacy_worker_identity_json="$(printf '%s' "$legacy_inventory_json" | "$PYTHON" -c 'import json,sys; print(json.dumps(json.load(sys.stdin)["workers"],sort_keys=True,separators=(",",":")))')"
legacy_main_identity="$(printf '%s' "$legacy_inventory_json" | "$PYTHON" -c 'import json,sys; print(json.load(sys.stdin)["main_id"])')"
[[ -z "$legacy_container_pid" || "$legacy_container_pid" =~ ^[0-9]+$ ]] || fail "legacy validator container PID is invalid"
ignore_tree_args=()
for container_pid in "$legacy_container_pid" "${legacy_worker_pids[@]}"; do
  if [ "${container_pid:-0}" -gt 0 ]; then ignore_tree_args+=(--ignore-tree-root "$container_pid"); fi
done
old_cid=""
if [ "$service_active" -eq 1 ]; then
  sudo test -r "$RUNTIME_ENV" || fail "active validator lacks its enclave identity"
  [ "$(sudo stat -c %u "$RUNTIME_ENV")" = 0 ] && [ "$(sudo stat -c %a "$RUNTIME_ENV")" = 600 ] || fail "active validator runtime identity is not root-private"
  old_cid="$(sudo sed -n 's/^ENCLAVE_CID=//p' "$RUNTIME_ENV")"
  [[ "$old_cid" =~ ^[0-9]+$ ]] || fail "active validator enclave identity is invalid"
fi
read -r discovered_cid old_enclave_id discovered_name discovered_cpus discovered_memory discovered_pcr0 < <(sudo nitro-cli describe-enclaves | "$PYTHON" -c '
import json,sys
owned=[x for x in json.load(sys.stdin) if x.get("State")=="RUNNING" and (x.get("EnclaveName")=="validator-enclave" or str(x.get("EnclaveName") or "").startswith(("arena-signer-","arena-validator-")))]
if len(owned)>1: raise SystemExit("validator enclave ownership is ambiguous")
x=owned[0] if owned else {}
print(x.get("EnclaveCID",""),x.get("EnclaveID",""),x.get("EnclaveName",""),x.get("NumberOfCPUs",""),x.get("MemoryMiB",""),x.get("Measurements",{}).get("PCR0",""))
') || fail "validator enclave ownership is ambiguous"
if [ -n "$old_cid" ] && [ "$old_cid" != "$discovered_cid" ]; then fail "service signer differs from the owned enclave"; fi
old_cid="${old_cid:-$discovered_cid}"
CANDIDATE_CID=19
reuse_candidate=0
if [ -n "$old_enclave_id" ] && [ "${discovered_pcr0,,}" = "${EXPECTED_PCR0,,}" ]; then
  CANDIDATE_CID="$discovered_cid"
  CANDIDATE_ENCLAVE_ID="$old_enclave_id"
  old_enclave_id=""
  reuse_candidate=1
elif [ -n "$old_enclave_id" ]; then
  for legacy_input in "$LEGACY_EIF" "$LEGACY_HOTKEY_CONFIG" "$LEGACY_RELEASE_MANIFEST" "$LEGACY_GATEWAY_MANIFEST" "$LEGACY_GATEWAY_LINEAGE" "$LEGACY_ENVELOPE"; do
    [ -f "$legacy_input" ] && [ ! -L "$legacy_input" ] || fail "legacy rollback input is unavailable"
  done
  [ -s "$LEGACY_EIF" ] && [ ! -L "$LEGACY_EIF" ] || fail "exact legacy signer EIF is unavailable"
  OLD_PCR0="$(sudo nitro-cli describe-eif --eif-path "$LEGACY_EIF" | "$PYTHON" -c 'import json,sys; print(json.load(sys.stdin)["Measurements"]["PCR0"])')"
  [ "${OLD_PCR0,,}" = "${discovered_pcr0,,}" ] || fail "legacy signer EIF differs from running enclave"
  LEGACY_EIF_SNAPSHOT="$(mktemp "$RELEASE_ROOT/.legacy-signer.XXXXXX.eif")"
  cp --reflink=auto "$LEGACY_EIF" "$LEGACY_EIF_SNAPSHOT"
  chmod 0600 "$LEGACY_EIF_SNAPSHOT"
  snapshot_pcr0="$(sudo nitro-cli describe-eif --eif-path "$LEGACY_EIF_SNAPSHOT" | "$PYTHON" -c 'import json,sys; print(json.load(sys.stdin)["Measurements"]["PCR0"])')"
  [ "${snapshot_pcr0,,}" = "${OLD_PCR0,,}" ] || fail "private legacy signer snapshot differs"
  OLD_ENCLAVE_ID="$old_enclave_id"; OLD_ENCLAVE_CID="$discovered_cid"; OLD_ENCLAVE_NAME="$discovered_name"
  OLD_ENCLAVE_CPUS="$discovered_cpus"; OLD_ENCLAVE_MEMORY="$discovered_memory"
fi

# Adopt only the exact legacy validator from the installed checkout. A process
# from another checkout or more than one owner is ambiguous and fails closed.
read -r old_pid old_start < <(
  cd "$RELEASE" || exit
  sudo env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$RELEASE" "$PYTHON" -m validator_tee.host.arena_restart_identity \
    "$SOURCE_ROOT" "$CURRENT_LINK" "$service_pid" ${ignore_tree_args[@]+"${ignore_tree_args[@]}"}
) || fail "validator process ownership is ambiguous"
read -r old_runner_pgid old_runner_start < <(
  cd "$RELEASE" || exit
  sudo env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$RELEASE" "$PYTHON" -m validator_tee.host.arena_restart_identity \
    "$SOURCE_ROOT" "$CURRENT_LINK" "$service_pid" --kind runner
) || fail "standalone Arena runner ownership is ambiguous"
read -r old_relay_pid old_relay_start < <(
  cd "$RELEASE" || exit
  sudo env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$RELEASE" "$PYTHON" -m validator_tee.host.arena_restart_identity \
    "$SOURCE_ROOT" "$CURRENT_LINK" "$service_pid" --kind relay
) || fail "legacy chain relay ownership is ambiguous"

if [ "$reuse_candidate" -eq 0 ]; then
  # Stop only the exact legacy weight producer before switching signer
  # semantics. Scoring and the relay remain live until protected readiness.
  if [ -n "$legacy_container_id" ] && [ "${legacy_container_pid:-0}" -gt 0 ]; then
    sudo docker stop --time "$STOP_TIMEOUT" "$legacy_container_id" >/dev/null
    LEGACY_CONTAINER_STOPPED=1
  fi
  if [ -n "$OLD_ENCLAVE_ID" ]; then
    sudo nitro-cli terminate-enclave --enclave-id "$OLD_ENCLAVE_ID" >/dev/null
    OLD_ENCLAVE_TERMINATED=1
  fi
  run_json="$(sudo nitro-cli run-enclave --eif-path "$EIF_FILE" --cpu-count "${VALIDATOR_ENCLAVE_CPU_COUNT:-2}" --memory "${VALIDATOR_ENCLAVE_MEMORY_MIB:-1024}" --enclave-cid "$CANDIDATE_CID" --enclave-name "arena-signer-${EXPECTED_PCR0:0:12}")"
  CANDIDATE_CREATED=1
  CANDIDATE_ENCLAVE_ID="$(printf '%s' "$run_json" | "$PYTHON" -c 'import json,sys; print(json.load(sys.stdin)["EnclaveID"])')"
  [ -n "$CANDIDATE_ENCLAVE_ID" ] || fail "candidate signer did not return an enclave identity"
fi
# First transition and later cold boots use a narrow recipient-only migration;
# no raw seed or general signing surface reaches the host.
candidate_rpc_ready=0
deadline=$((SECONDS + READY_TIMEOUT))
while [ "$SECONDS" -lt "$deadline" ]; do
  if ( cd "$RELEASE" && PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$RELEASE" ENCLAVE_CID="$CANDIDATE_CID" "$PYTHON" -c 'from validator_tee.host.vsock_client import ValidatorEnclaveClient; value=ValidatorEnclaveClient().get_arena_hotkey_state_v1(); assert isinstance(value.get("provisioned"),bool)' ) >/dev/null 2>&1; then candidate_rpc_ready=1; break; fi
  sleep 1
done
[ "$candidate_rpc_ready" -eq 1 ] || fail "candidate signer RPC did not become ready"
if ! ( cd "$RELEASE" && PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$RELEASE" ENCLAVE_CID="$CANDIDATE_CID" "$PYTHON" -c 'from validator_tee.host.vsock_client import ValidatorEnclaveClient; raise SystemExit(0 if ValidatorEnclaveClient().get_arena_hotkey_state_v1().get("provisioned") else 1)' ) 2>/dev/null; then
  [ -f "$LEGACY_ENVELOPE" ] && [ ! -L "$LEGACY_ENVELOPE" ] || fail "legacy encrypted hotkey envelope is unavailable"
  ( cd "$RELEASE" && PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$RELEASE" ENCLAVE_CID="$CANDIDATE_CID" "$PYTHON" -m validator_tee.host.arena_hotkey_bootstrap migrate-legacy \
    --legacy-envelope "$LEGACY_ENVELOPE" --policy "$POLICY_FILE" --kms-key-id "$MIGRATION_KMS_KEY_ID" )
fi
( cd "$RELEASE" && sudo timeout "$READY_TIMEOUT" env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$RELEASE" "$PYTHON" "$RELEASE/scripts/run_arena_validator.py" --environment-file "$CANDIDATE_SERVICE_ENV" --enclave-cid "$CANDIDATE_CID" --check-only )
SIGNER_HANDOFF_COMMITTED=1

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
stop_owned_process() {
  local pid="$1" start="$2" label="$3" signal_target="$1"
  [ -n "$pid" ] || return 0
  [ "$(sudo awk '{print $22}' "/proc/$pid/stat" 2>/dev/null || true)" = "$start" ] || fail "$label identity changed before drain"
  [ "$label" != "standalone Arena runner" ] || signal_target="-$pid"
  sudo kill -TERM -- "$signal_target"
  local wait_seconds=300
  [ "$label" != "standalone Arena runner" ] || wait_seconds="$STOP_TIMEOUT"
  local deadline=$((SECONDS + wait_seconds))
  while sudo kill -0 "$pid" 2>/dev/null && [ "$SECONDS" -lt "$deadline" ]; do sleep 1; done
  if sudo kill -0 "$pid" 2>/dev/null; then
    [ "$label" != "standalone Arena runner" ] || fail "standalone Arena runner still owns active work after drain timeout"
    sudo kill -KILL -- "$signal_target"; sleep 1
  fi
  ! sudo kill -0 "$pid" 2>/dev/null || fail "$label did not stop"
}
stop_owned_process "$old_runner_pgid" "$old_runner_start" "standalone Arena runner"
if [ -n "$legacy_container_id" ] && [ "${legacy_container_pid:-0}" -gt 0 ] && [ "$LEGACY_CONTAINER_STOPPED" -eq 0 ]; then
  sudo docker stop --time "$STOP_TIMEOUT" "$legacy_container_id" >/dev/null
  [ "$(sudo docker inspect -f '{{.State.Running}}' "$legacy_container_id")" = false ] || fail "legacy validator container did not stop"
fi
stop_owned_process "$old_relay_pid" "$old_relay_start" "legacy chain relay"
if sudo ss --vsock -H -ln | awk '$5 ~ /:500[23]$/ {found=1} END{exit !found}'; then fail "a validator relay remains after service stop"; fi

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
UNIT_STAGE="$(mktemp "$RELEASE_ROOT/.arena-validator-unit.XXXXXX")"
sed -e "s|^ExecStartPre=/usr/bin/python3 |ExecStartPre=$PYTHON |" \
    -e "s|^ExecStart=/usr/bin/python3 |ExecStart=$PYTHON |" \
    "$RELEASE/deploy/leadpoet-arena-validator.service" > "$UNIT_STAGE"
grep -Fq "ExecStartPre=$PYTHON " "$UNIT_STAGE" || fail "validator readiness interpreter template differs"
grep -Fq "ExecStart=$PYTHON " "$UNIT_STAGE" || fail "validator service interpreter template differs"
sudo install -m 0644 "$UNIT_STAGE" "$UNIT_PATH"
rm -f -- "$UNIT_STAGE"; UNIT_STAGE=""
sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE" >/dev/null
SERVICE_ATTEMPTED=1
sudo systemctl start "$SERVICE"
SERVICE_STARTED=1

deadline=$((SECONDS + READY_TIMEOUT))
while [ "$SECONDS" -lt "$deadline" ]; do
  main_pid="$(sudo systemctl show -p MainPID --value "$SERVICE")"
  if sudo systemctl is-active --quiet "$SERVICE" && [[ "$main_pid" =~ ^[1-9][0-9]*$ ]] && sudo ss --vsock -H -ln | awk '$5 ~ /:5002$/ {chain=1} $5 ~ /:5003$/ {state=1} END{exit !(chain&&state)}'; then
    ACTIVATED=1
    break
  fi
  sleep 2
done
[ "$ACTIVATED" -eq 1 ] || fail "normal Arena validator did not become ready"

# Retire exact old fulfillment workers only after the replacement service is
# active and every old work file has a corresponding result.
if [ "${#legacy_worker_container_ids[@]}" -gt 0 ]; then
  deadline=$((SECONDS + STOP_TIMEOUT))
  while ! ( cd "$RELEASE" && PYTHONPATH="$RELEASE" "$PYTHON" -c '
from pathlib import Path
import sys
from validator_tee.host.arena_restart_identity import legacy_fulfillment_queue_is_quiescent
raise SystemExit(0 if legacy_fulfillment_queue_is_quiescent(Path(sys.argv[1])) else 1)
' "$SOURCE_ROOT/validator_weights" ); do
    [ "$SECONDS" -lt "$deadline" ] || fail "legacy fulfillment work remains active"
    sleep 5
  done
  current_legacy_inventory_json="$(read_legacy_inventory_json)" \
    || fail "legacy fulfillment worker identity changed before stop"
  current_legacy_worker_identity_json="$(printf '%s' "$current_legacy_inventory_json" | "$PYTHON" -c 'import json,sys; print(json.dumps(json.load(sys.stdin)["workers"],sort_keys=True,separators=(",",":")))')"
  current_legacy_main_identity="$(printf '%s' "$current_legacy_inventory_json" | "$PYTHON" -c 'import json,sys; print(json.load(sys.stdin)["main_id"])')"
  [ "$current_legacy_main_identity" = "$legacy_main_identity" ] \
    && [ "$current_legacy_worker_identity_json" = "$legacy_worker_identity_json" ] \
    || fail "legacy fulfillment worker identity changed before stop"
  timeout "$((STOP_TIMEOUT + 30))" sudo docker stop --time "$STOP_TIMEOUT" \
    "${legacy_worker_container_ids[@]}" >/dev/null \
    || fail "legacy fulfillment workers did not stop"
  for worker_container_id in "${legacy_worker_container_ids[@]}"; do
    [ "$(sudo docker inspect -f '{{.State.Running}}' "$worker_container_id")" = false ] \
      || fail "legacy fulfillment worker remains active"
  done
fi
[ -z "$SERVICE_ENV_BACKUP" ] || sudo rm -f -- "$SERVICE_ENV_BACKUP"
SERVICE_ENV_BACKUP=""
[ -z "$RUNTIME_ENV_BACKUP" ] || sudo rm -f -- "$RUNTIME_ENV_BACKUP"
RUNTIME_ENV_BACKUP=""
CANDIDATE_ENCLAVE_ID=""
echo "SUCCESS: normal Arena validator is supervised at exact commit $TARGET_SHA"
