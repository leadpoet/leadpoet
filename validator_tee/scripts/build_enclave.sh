#!/bin/bash
# Build the small Arena signer from an exact committed Git archive.
set -euo pipefail
umask 077
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
COMMIT_REQUEST="${VALIDATOR_ARENA_SIGNER_COMMIT:-HEAD}"
COMMIT_SHA="$(git -C "$REPO_ROOT" rev-parse --verify "$COMMIT_REQUEST^{commit}")"
[[ "$COMMIT_SHA" =~ ^[0-9a-f]{40}$ ]] || { echo "invalid signer source commit" >&2; exit 1; }
POLICY_INPUT="${VALIDATOR_ARENA_SIGNER_POLICY:-${VALIDATOR_ARENA_SIGNER_POLICY_INPUT:-}}"
[ -n "$POLICY_INPUT" ] || { echo "set VALIDATOR_ARENA_SIGNER_POLICY to the reviewed public Arena policy" >&2; exit 1; }
[ -f "$POLICY_INPUT" ] && [ ! -L "$POLICY_INPUT" ] || { echo "Arena signer policy must be a non-symlink regular file" >&2; exit 1; }
POLICY_INPUT="$(python3 -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$POLICY_INPUT")"
OUTPUT_ROOT="${VALIDATOR_ARENA_SIGNER_OUTPUT_DIR:-$HOME/.cache/leadpoet/arena-signer/$COMMIT_SHA}"
OFFLINE_ROOT="${VALIDATOR_V2_OFFLINE_ARTIFACT_ROOT:-$HOME/.cache/leadpoet-v2-artifacts/validator-runtime}"
BUILD_PARENT="${VALIDATOR_ARENA_SIGNER_BUILD_ROOT:-$HOME/.cache/leadpoet/arena-signer-build}"
mkdir -p "$OUTPUT_ROOT" "$BUILD_PARENT"
exec 9>"$OUTPUT_ROOT/.publication.lock"
flock -n 9 || { echo "another Arena signer publication owns $OUTPUT_ROOT" >&2; exit 1; }
WORK="$(mktemp -d "$BUILD_PARENT/exact-source.XXXXXX")"
trap 'rm -rf -- "$WORK"' EXIT
CONTEXT="$WORK/source"
mkdir "$CONTEXT"
# The build context excludes every dirty, ignored, and untracked caller file.
git -C "$REPO_ROOT" archive --format=tar "$COMMIT_SHA" | tar -xf - -C "$CONTEXT"
mkdir -p "$CONTEXT/.validator-tee-artifacts"
# Validate once and freeze canonical bytes. The EIF, manifest and published
# policy all consume this same immutable file.
FROZEN_POLICY="$WORK/arena_signer_policy.json"
PYTHONPATH="$CONTEXT" python3 - "$POLICY_INPUT" "$FROZEN_POLICY" <<'PY'
import json,sys
from pathlib import Path
from validator_tee.enclave.arena_hotkey import validate_policy
policy=validate_policy(json.loads(Path(sys.argv[1]).read_text()))
Path(sys.argv[2]).write_text(json.dumps(policy,sort_keys=True,separators=(",",":"),ensure_ascii=False)+"\n")
PY
install -m 0644 "$FROZEN_POLICY" "$CONTEXT/.validator-tee-artifacts/arena_signer_policy.json"
. "$CONTEXT/validator_tee/scripts/docker_operation_lock_v2.sh"
leadpoet_acquire_docker_operation_lock_v2
python3 "$CONTEXT/validator_tee/scripts/stage_runtime_artifacts_v2.py" --lock "$CONTEXT/validator_tee/runtime-artifacts-v2.lock.json" --output-dir "$CONTEXT/.validator-tee-artifacts" --offline-artifact-root "$OFFLINE_ROOT"
bash "$CONTEXT/validator_tee/scripts/build_drand_cabi_v2.sh" "$CONTEXT/.validator-tee-artifacts/bittensor_drand-2.0.0.tar.gz" "$CONTEXT/.validator-tee-artifacts/libbittensor_drand_v2.so" "$CONTEXT/validator_tee/enclave/libbittensor_drand_v2.sha256"
python3 - "$FROZEN_POLICY" "$CONTEXT/.validator-tee-artifacts/libbittensor_drand_v2.so" <<'PY'
import hashlib,json,sys
from pathlib import Path
policy=json.loads(Path(sys.argv[1]).read_text())
observed=hashlib.sha256(Path(sys.argv[2]).read_bytes()).hexdigest()
if policy.get("drand_library_sha256") != observed:
    raise SystemExit("built drand library differs from measured Arena policy")
PY
RAW_IMAGE="leadpoet-arena-signer-raw:${COMMIT_SHA}"
NORMAL_IMAGE="leadpoet-arena-signer:${COMMIT_SHA}"
docker build --no-cache -f "$CONTEXT/validator_tee/Dockerfile.base" -t validator-base:v1 "$CONTEXT"
docker build --no-cache -f "$CONTEXT/validator_tee/Dockerfile.arena-signer" -t "$RAW_IMAGE" "$CONTEXT"
PYTHONPATH="$CONTEXT" python3 -m validator_tee.host.docker_image_normalizer_v2 --source-image "$RAW_IMAGE" --normalized-image "$NORMAL_IMAGE"
EIF_TMP="$WORK/validator-enclave.eif"
MEASUREMENTS="$WORK/enclave-build.json"
nitro-cli build-enclave --docker-uri "$NORMAL_IMAGE" --output-file "$EIF_TMP" | tee "$MEASUREMENTS"
PYTHONPATH="$CONTEXT" python3 - "$EIF_TMP" "$MEASUREMENTS" "$FROZEN_POLICY" "$WORK/manifest.json" <<'PY'
import hashlib,json,re,sys
from pathlib import Path
from validator_tee.enclave.arena_hotkey import validate_policy
eif,measurements,policy_path,output=sys.argv[1:]
value=json.loads(Path(measurements).read_text())
pcr0=str(value.get("Measurements",{}).get("PCR0","")).lower()
if not re.fullmatch(r"[0-9a-f]{96}",pcr0) or pcr0 == "0"*96: raise SystemExit("Nitro did not return a production PCR0")
digest=hashlib.sha256(Path(eif).read_bytes()).hexdigest()
policy=validate_policy(json.loads(Path(policy_path).read_text()))
policy_bytes=json.dumps(policy,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()
manifest={"schema_version":"leadpoet.arena.signer_manifest.v1","eif_sha256":"sha256:"+digest,"pcr0":pcr0,"policy_sha256":"sha256:"+hashlib.sha256(policy_bytes).hexdigest()}
Path(output).write_text(json.dumps(manifest,sort_keys=True,separators=(",",":"))+"\n")
PY
install -m 0644 "$EIF_TMP" "$OUTPUT_ROOT/validator-enclave.eif"
install -m 0644 "$WORK/manifest.json" "$OUTPUT_ROOT/manifest.json"
install -m 0644 "$FROZEN_POLICY" "$OUTPUT_ROOT/arena_signer_policy.json"
echo "Arena signer artifact ready: commit=$COMMIT_SHA output=$OUTPUT_ROOT"
