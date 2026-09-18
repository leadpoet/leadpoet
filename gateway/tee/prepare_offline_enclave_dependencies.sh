#!/bin/bash
# Prepare the exact identity-enclave wheel set before the release build.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ARTIFACT_ROOT="${GATEWAY_V2_OFFLINE_ARTIFACT_ROOT:-$HOME/.cache/leadpoet-v2-artifacts}"
WHEELHOUSE="$ARTIFACT_ROOT/enclave-wheelhouse-py39"
INPUT="$SCRIPT_DIR/requirements-enclave-py39.in"
LOCK="$SCRIPT_DIR/requirements-enclave-py39.lock"

mkdir -p "$ARTIFACT_ROOT"
chmod 700 "$ARTIFACT_ROOT"
TEMP_ROOT="$(mktemp -d "$ARTIFACT_ROOT/.prepare-enclave.XXXXXX")"
trap 'rm -rf "$TEMP_ROOT"' EXIT
mkdir -p "$TEMP_ROOT/wheelhouse"

if [ -d "$WHEELHOUSE" ]     && ! find "$WHEELHOUSE" -mindepth 1 -maxdepth 1       \( ! -type f -o ! -name '*.whl' \) | grep -q .     && python3 "$SCRIPT_DIR/enclave_wheelhouse.py" verify-wheelhouse       --input "$INPUT" --lock "$LOCK" --wheelhouse "$WHEELHOUSE" >/dev/null 2>&1; then
  rsync -a --delete "$WHEELHOUSE/" "$TEMP_ROOT/wheelhouse/"
else
  python3 -m pip download     --no-deps     --require-hashes     --dest "$TEMP_ROOT/wheelhouse"     --only-binary=:all:     --platform manylinux2014_x86_64     --implementation cp     --python-version 39     --abi cp39     -r "$LOCK"
fi
python3 "$SCRIPT_DIR/enclave_wheelhouse.py" verify-wheelhouse   --input "$INPUT" --lock "$LOCK" --wheelhouse "$TEMP_ROOT/wheelhouse"
python3 "$SCRIPT_DIR/normalize_attested_runtime.py" --root "$TEMP_ROOT/wheelhouse"

. "$REPO_ROOT/validator_tee/scripts/docker_operation_lock_v2.sh"
leadpoet_acquire_docker_operation_lock_v2
rm -rf "$WHEELHOUSE"
mkdir -p "$(dirname "$WHEELHOUSE")"
mv "$TEMP_ROOT/wheelhouse" "$WHEELHOUSE"
rm -rf "$TEMP_ROOT"
python3 "$SCRIPT_DIR/enclave_wheelhouse.py" verify-wheelhouse   --input "$INPUT" --lock "$LOCK" --wheelhouse "$WHEELHOUSE"
leadpoet_release_docker_operation_lock_v2
echo "Identity-enclave dependencies are hash verified in $WHEELHOUSE"
