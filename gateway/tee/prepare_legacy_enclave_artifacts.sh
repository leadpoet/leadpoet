#!/bin/bash
# Fetch and verify the deterministic inputs shared by the single legacy EIF.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ARTIFACT_ROOT="${GATEWAY_V2_OFFLINE_ARTIFACT_ROOT:-$HOME/.cache/leadpoet-v2-artifacts}"
WHEELHOUSE="$ARTIFACT_ROOT/scoring-wheelhouse-py39"
RUNSC_LOCK="$SCRIPT_DIR/runsc-runtime.lock.json"
SCORING_INPUT="$SCRIPT_DIR/requirements-scoring-py39.in"
SCORING_LOCK="$SCRIPT_DIR/requirements-scoring-py39.lock"

mkdir -p "$ARTIFACT_ROOT"
chmod 700 "$ARTIFACT_ROOT"
TEMP_ROOT="$(mktemp -d "$ARTIFACT_ROOT/.prepare-legacy.XXXXXX")"
trap 'rm -rf "$TEMP_ROOT"' EXIT

echo "Preparing the hash-locked CPython 3.9 scoring wheelhouse"
mkdir -p "$TEMP_ROOT/wheelhouse"
if [ -d "$WHEELHOUSE" ] \
    && ! find "$WHEELHOUSE" -mindepth 1 -maxdepth 1 \
      \( ! -type f -o ! -name '*.whl' \) | grep -q . \
    && python3 "$SCRIPT_DIR/scoring_wheelhouse.py" verify-wheelhouse \
      --input "$SCORING_INPUT" \
      --lock "$SCORING_LOCK" \
      --wheelhouse "$WHEELHOUSE" >/dev/null 2>&1; then
  rsync -a --delete "$WHEELHOUSE/" "$TEMP_ROOT/wheelhouse/"
else
  python3 -m pip download \
    --no-deps \
    --require-hashes \
    --dest "$TEMP_ROOT/wheelhouse" \
    --only-binary=:all: \
    --platform manylinux2014_x86_64 \
    --implementation cp \
    --python-version 39 \
    --abi cp39 \
    -r "$SCORING_LOCK"
fi
python3 "$SCRIPT_DIR/scoring_wheelhouse.py" verify-wheelhouse \
  --input "$SCORING_INPUT" \
  --lock "$SCORING_LOCK" \
  --wheelhouse "$TEMP_ROOT/wheelhouse"
python3 "$SCRIPT_DIR/normalize_attested_runtime.py" \
  --root "$TEMP_ROOT/wheelhouse"

read -r RUNSC_NAME RUNSC_URL < <(
  python3 - "$RUNSC_LOCK" <<'PY'
import json
import sys

value = json.load(open(sys.argv[1], encoding="utf-8"))
print(value["artifact_filename"], value["source_url"])
PY
)
echo "Preparing the hash-locked model sandbox runtime"
if PYTHONPATH="$REPO_ROOT" python3 "$SCRIPT_DIR/sandbox_runtime_artifact.py" verify \
    --lock "$RUNSC_LOCK" \
    --artifact "$ARTIFACT_ROOT/$RUNSC_NAME" >/dev/null 2>&1; then
  cp "$ARTIFACT_ROOT/$RUNSC_NAME" "$TEMP_ROOT/$RUNSC_NAME"
else
  curl --fail --location --proto '=https' --tlsv1.2 \
    --output "$TEMP_ROOT/$RUNSC_NAME" "$RUNSC_URL"
fi
PYTHONPATH="$REPO_ROOT" python3 "$SCRIPT_DIR/sandbox_runtime_artifact.py" verify \
  --lock "$RUNSC_LOCK" \
  --artifact "$TEMP_ROOT/$RUNSC_NAME"
chmod 755 "$TEMP_ROOT/$RUNSC_NAME"
touch -d @0 "$TEMP_ROOT/$RUNSC_NAME"

rm -rf "$WHEELHOUSE"
mkdir -p "$(dirname "$WHEELHOUSE")"
mv "$TEMP_ROOT/wheelhouse" "$WHEELHOUSE"
install -m 755 "$TEMP_ROOT/$RUNSC_NAME" "$ARTIFACT_ROOT/$RUNSC_NAME"
touch -d @0 "$ARTIFACT_ROOT/$RUNSC_NAME"

python3 "$SCRIPT_DIR/scoring_wheelhouse.py" verify-wheelhouse \
  --input "$SCORING_INPUT" \
  --lock "$SCORING_LOCK" \
  --wheelhouse "$WHEELHOUSE"
python3 "$SCRIPT_DIR/sandbox_runtime_artifact.py" verify \
  --lock "$RUNSC_LOCK" \
  --artifact "$ARTIFACT_ROOT/$RUNSC_NAME"
echo "Legacy gateway enclave artifacts are hash verified in $ARTIFACT_ROOT"
