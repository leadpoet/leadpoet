#!/bin/bash
#
# Stage top-level Research Lab runtime packages into the gateway Docker context.
# The Nitro enclave build only sees ~/gateway, so runtime dependencies imported
# as top-level packages must be copied under gateway/_attested_runtime first.

set -euo pipefail

GATEWAY_ROOT="${GATEWAY_ROOT:-$HOME/gateway}"
DEPLOY_SOURCE_ROOT="${RESEARCH_LAB_RUNTIME_SOURCE_ROOT:-$HOME}"
CLEAN_SOURCE_ROOT="${ATTESTED_RUNTIME_GIT_SOURCE_ROOT:-/tmp/leadpoet_gateway_enclave_source}"
SOURCE_REPO_URL="${ATTESTED_RUNTIME_GIT_REPO_URL:-https://github.com/leadpoet/leadpoet.git}"
DEST_ROOT="$GATEWAY_ROOT/_attested_runtime"
TMP_ROOT="$GATEWAY_ROOT/.attested_runtime.tmp"
BUILD_CONTEXT_ROOT="$GATEWAY_ROOT/_enclave_source"
BUILD_CONTEXT_TMP="$GATEWAY_ROOT/.enclave_source.tmp"
WHEELHOUSE_ROOT="$GATEWAY_ROOT/_enclave_wheelhouse"
WHEELHOUSE_TMP="$GATEWAY_ROOT/.enclave_wheelhouse.tmp"
PACKAGES=(
  "research_lab"
  "leadpoet_verifier"
  "schemas"
  "leadpoet_canonical"
  "qualification"
  "validator_models"
)

if ! command -v rsync >/dev/null 2>&1; then
  echo "ERROR: rsync is required to stage attested runtime packages" >&2
  exit 1
fi
if ! command -v git >/dev/null 2>&1; then
  echo "ERROR: git is required to stage a clean attested runtime commit" >&2
  exit 1
fi

ATTESTED_COMMIT_SHA="$(
  python3 "$GATEWAY_ROOT/tee/build_identity.py" resolve \
    --gateway-root "$GATEWAY_ROOT" \
    --source-root "$DEPLOY_SOURCE_ROOT"
)"

if [ "${ATTESTED_RUNTIME_SOURCE_IS_CLEAN_GIT_ARCHIVE:-0}" = "1" ]; then
  SOURCE_ROOT="$DEPLOY_SOURCE_ROOT"
else
  DEPLOY_SOURCE_COMMIT="$(git -C "$DEPLOY_SOURCE_ROOT" rev-parse HEAD 2>/dev/null || true)"
  DEPLOY_SOURCE_DIRTY="$(git -C "$DEPLOY_SOURCE_ROOT" status --porcelain 2>/dev/null || true)"
  if [ "$DEPLOY_SOURCE_COMMIT" = "$ATTESTED_COMMIT_SHA" ] \
      && [ -z "$DEPLOY_SOURCE_DIRTY" ] \
      && [ -d "$DEPLOY_SOURCE_ROOT/gateway" ]; then
    SOURCE_ROOT="$DEPLOY_SOURCE_ROOT"
  else
    rm -rf "$CLEAN_SOURCE_ROOT"
    git init -q "$CLEAN_SOURCE_ROOT"
    git -C "$CLEAN_SOURCE_ROOT" remote add origin "$SOURCE_REPO_URL"
    git -C "$CLEAN_SOURCE_ROOT" fetch -q --depth=1 origin "$ATTESTED_COMMIT_SHA"
    git -C "$CLEAN_SOURCE_ROOT" checkout -q --detach FETCH_HEAD
    RESOLVED_SOURCE_COMMIT="$(git -C "$CLEAN_SOURCE_ROOT" rev-parse HEAD)"
    if [ "$RESOLVED_SOURCE_COMMIT" != "$ATTESTED_COMMIT_SHA" ]; then
      echo "ERROR: clean attested source commit mismatch" >&2
      exit 1
    fi
    SOURCE_ROOT="$CLEAN_SOURCE_ROOT"
  fi
fi

SOURCE_GATEWAY_ROOT="$SOURCE_ROOT/gateway"
SCORING_REQUIREMENTS_INPUT="$SOURCE_GATEWAY_ROOT/tee/requirements-scoring-py39.in"
SCORING_REQUIREMENTS_LOCK="$SOURCE_GATEWAY_ROOT/tee/requirements-scoring-py39.lock"

if [ ! -d "$SOURCE_GATEWAY_ROOT" ]; then
  echo "ERROR: clean gateway source missing: $SOURCE_GATEWAY_ROOT" >&2
  exit 1
fi

echo "Staging attested Research Lab runtime packages"
echo "  Gateway root:  $GATEWAY_ROOT"
echo "  Source root:   $SOURCE_ROOT"
echo "  Source commit: $ATTESTED_COMMIT_SHA"
echo "  Dest root:     $DEST_ROOT"

rm -rf "$TMP_ROOT"
mkdir -p "$TMP_ROOT"

for package in "${PACKAGES[@]}"; do
  source_dir="$SOURCE_ROOT/$package"
  dest_dir="$TMP_ROOT/$package"
  if [ ! -d "$source_dir" ]; then
    echo "ERROR: required runtime package missing: $source_dir" >&2
    exit 1
  fi
  mkdir -p "$dest_dir"
  rsync -a --delete \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='*.pyo' \
    --exclude='*.pyd' \
    --exclude='*.log' \
    --exclude='*.bak' \
    --exclude='*.bak-*' \
    --exclude='*.backup*' \
    --exclude='*.tmp' \
    --exclude='*.pem' \
    --exclude='*.key' \
    --exclude='*.jwk' \
    --exclude='.DS_Store' \
    --exclude='.env' \
    "$source_dir/" "$dest_dir/"
done

python3 "$SOURCE_GATEWAY_ROOT/tee/scoring_import_closure.py" build \
  --gateway-root "$SOURCE_GATEWAY_ROOT" \
  --source-root "$SOURCE_ROOT" \
  --output "$TMP_ROOT/scoring_import_closure.json"

python3 "$SOURCE_GATEWAY_ROOT/tee/build_identity.py" build \
  --gateway-root "$SOURCE_GATEWAY_ROOT" \
  --source-root "$SOURCE_ROOT" \
  --manifest "$TMP_ROOT/scoring_import_closure.json" \
  --output "$TMP_ROOT/gateway_enclave_build_identity.json" \
  --commit "$ATTESTED_COMMIT_SHA"

python3 "$SOURCE_GATEWAY_ROOT/tee/normalize_attested_runtime.py" --root "$TMP_ROOT"

find "$TMP_ROOT" -type d -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true
find "$TMP_ROOT" -type f \( -name '*.pyc' -o -name '*.pyo' -o -name '*.pyd' \) -delete 2>/dev/null || true

rm -rf "$DEST_ROOT"
mv "$TMP_ROOT" "$DEST_ROOT"

rm -rf "$WHEELHOUSE_TMP"
mkdir -p "$WHEELHOUSE_TMP"
python3 -m pip download \
  --no-deps \
  --require-hashes \
  --dest "$WHEELHOUSE_TMP" \
  --only-binary=:all: \
  --platform manylinux2014_x86_64 \
  --implementation cp \
  --python-version 39 \
  --abi cp39 \
  -r "$SCORING_REQUIREMENTS_LOCK"
python3 "$SOURCE_GATEWAY_ROOT/tee/scoring_wheelhouse.py" verify-wheelhouse \
  --input "$SCORING_REQUIREMENTS_INPUT" \
  --lock "$SCORING_REQUIREMENTS_LOCK" \
  --wheelhouse "$WHEELHOUSE_TMP"
python3 "$SOURCE_GATEWAY_ROOT/tee/normalize_attested_runtime.py" --root "$WHEELHOUSE_TMP"
rm -rf "$WHEELHOUSE_ROOT"
mv "$WHEELHOUSE_TMP" "$WHEELHOUSE_ROOT"

# Build Docker from one normalized source tree. The outer restart command still
# uses the gateway root as its Docker context, but Docker copies only this tree,
# so checkout/rsync timestamps and unrelated host files cannot change PCR0.
rm -rf "$BUILD_CONTEXT_TMP"
mkdir -p "$BUILD_CONTEXT_TMP"
rsync -a --delete \
  --exclude='_attested_runtime/' \
  --exclude='_enclave_source/' \
  --exclude='_enclave_wheelhouse/' \
  --exclude='.attested_runtime.tmp/' \
  --exclude='.enclave_source.tmp/' \
  --exclude='.enclave_wheelhouse.tmp/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='*.pyo' \
  --exclude='*.pyd' \
  --exclude='*.log' \
  --exclude='*.bak' \
  --exclude='*.bak-*' \
  --exclude='*.backup*' \
  --exclude='*.tmp' \
  --exclude='*.eif' \
  --exclude='*.pem' \
  --exclude='*.key' \
  --exclude='*.jwk' \
  --exclude='.DS_Store' \
  --exclude='.env' \
  --exclude='secrets/' \
  --exclude='logs/' \
  --exclude='validation_artifacts/' \
  --exclude='hotpatch-backups/' \
  --exclude='BUILD_INFO.json' \
  --exclude='.source_commit' \
  "$SOURCE_GATEWAY_ROOT/" "$BUILD_CONTEXT_TMP/"
mkdir -p "$BUILD_CONTEXT_TMP/_attested_runtime"
rsync -a --delete "$DEST_ROOT/" "$BUILD_CONTEXT_TMP/_attested_runtime/"
mkdir -p "$BUILD_CONTEXT_TMP/tee/wheelhouse"
rsync -a --delete "$WHEELHOUSE_ROOT/" "$BUILD_CONTEXT_TMP/tee/wheelhouse/"
python3 "$SOURCE_GATEWAY_ROOT/tee/normalize_attested_runtime.py" --root "$BUILD_CONTEXT_TMP"
rm -rf "$BUILD_CONTEXT_ROOT"
mv "$BUILD_CONTEXT_TMP" "$BUILD_CONTEXT_ROOT"

echo "Attested runtime staged:"
find "$DEST_ROOT" -type f | wc -l | awk '{print "  files: " $1}'
echo "Deterministic gateway enclave source staged:"
find "$BUILD_CONTEXT_ROOT" -type f | wc -l | awk '{print "  files: " $1}'
