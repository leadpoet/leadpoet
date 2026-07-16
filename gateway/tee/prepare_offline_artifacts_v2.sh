#!/bin/bash
# Fetch and verify all network-sourced V2 build artifacts before an EIF build.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ARTIFACT_ROOT="${GATEWAY_V2_OFFLINE_ARTIFACT_ROOT:-$HOME/.cache/leadpoet-v2-artifacts}"
WHEELHOUSE="$ARTIFACT_ROOT/scoring-wheelhouse-py39"
VALIDATOR_RUNTIME="$ARTIFACT_ROOT/validator-runtime"
RUNSC_LOCK="$SCRIPT_DIR/runsc-runtime.lock.json"
SCORING_INPUT="$SCRIPT_DIR/requirements-scoring-py39.in"
SCORING_LOCK="$SCRIPT_DIR/requirements-scoring-py39.lock"

mkdir -p "$ARTIFACT_ROOT"
chmod 700 "$ARTIFACT_ROOT"
TEMP_ROOT="$(mktemp -d "$ARTIFACT_ROOT/.prepare-v2.XXXXXX")"
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
python3 "$SCRIPT_DIR/normalize_attested_runtime.py" --root "$TEMP_ROOT/wheelhouse"

read -r RUNSC_NAME RUNSC_URL < <(
  python3 - "$RUNSC_LOCK" <<'PY'
import json
import sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
print(value["artifact_filename"], value["source_url"])
PY
)
echo "Preparing the hash-locked gVisor runtime"
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

echo "Preparing hash-locked validator binary artifacts"
if ! PYTHONPATH="$REPO_ROOT" python3 -m validator_tee.scripts.stage_runtime_artifacts_v2 \
    --lock "$REPO_ROOT/validator_tee/runtime-artifacts-v2.lock.json" \
    --output-dir "$TEMP_ROOT/validator-runtime" \
    --offline-artifact-root "$VALIDATOR_RUNTIME" >/dev/null 2>&1; then
  rm -rf "$TEMP_ROOT/validator-runtime"
  PYTHONPATH="$REPO_ROOT" python3 -m validator_tee.scripts.stage_runtime_artifacts_v2 \
    --lock "$REPO_ROOT/validator_tee/runtime-artifacts-v2.lock.json" \
    --output-dir "$TEMP_ROOT/validator-runtime" \
    --allow-download >/dev/null
fi

rm -rf "$WHEELHOUSE" "$VALIDATOR_RUNTIME"
mkdir -p "$(dirname "$WHEELHOUSE")"
mv "$TEMP_ROOT/wheelhouse" "$WHEELHOUSE"
mv "$TEMP_ROOT/validator-runtime" "$VALIDATOR_RUNTIME"
install -m 755 "$TEMP_ROOT/$RUNSC_NAME" "$ARTIFACT_ROOT/$RUNSC_NAME"
touch -d @0 "$ARTIFACT_ROOT/$RUNSC_NAME"
rm -rf "$TEMP_ROOT"

python3 "$SCRIPT_DIR/scoring_wheelhouse.py" verify-wheelhouse \
  --input "$SCORING_INPUT" \
  --lock "$SCORING_LOCK" \
  --wheelhouse "$WHEELHOUSE"
PYTHONPATH="$REPO_ROOT" python3 "$SCRIPT_DIR/sandbox_runtime_artifact.py" verify \
  --lock "$RUNSC_LOCK" \
  --artifact "$ARTIFACT_ROOT/$RUNSC_NAME"
VALIDATOR_CHECK_DIR="$(mktemp -d "$ARTIFACT_ROOT/.validator-check.XXXXXX")"
PYTHONPATH="$REPO_ROOT" python3 -m validator_tee.scripts.stage_runtime_artifacts_v2 \
  --lock "$REPO_ROOT/validator_tee/runtime-artifacts-v2.lock.json" \
  --output-dir "$VALIDATOR_CHECK_DIR" \
  --offline-artifact-root "$VALIDATOR_RUNTIME" >/dev/null
rm -rf "$VALIDATOR_CHECK_DIR"
echo "All V2 release artifacts are prepared and hash verified in $ARTIFACT_ROOT"
