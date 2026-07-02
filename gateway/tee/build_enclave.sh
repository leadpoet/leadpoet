#!/bin/bash
#
# Build Nitro Enclave Image
# ==========================
# This script builds the enclave Docker image and converts it to .eif format
#

set -e  # Exit on error

echo "=========================================="
echo "🔨 Building Nitro Enclave Image"
echo "=========================================="

# Step 1: Build Docker image (use gateway root as build context)
echo ""
echo "📦 Step 1: Building Docker image..."
echo "   Build context: ~/gateway/ (gateway root)"
echo "   Dockerfile: ~/tee/Dockerfile.enclave"
echo "   Attested runtime: ~/gateway/_attested_runtime"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GATEWAY_ROOT="${GATEWAY_ROOT:-$HOME/gateway}"
BUILD_INFO_PATH="$GATEWAY_ROOT/BUILD_INFO.json"
BUILD_INFO_SCRIPT="$SCRIPT_DIR/../../scripts/write_gateway_build_info.py"

if [ ! -f "$BUILD_INFO_PATH" ] && [ -f "$BUILD_INFO_SCRIPT" ]; then
  echo ""
  echo "🧾 Generating gateway build info..."
  REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
  python3 "$BUILD_INFO_SCRIPT" \
    --repo-root "$REPO_ROOT" \
    --output "$BUILD_INFO_PATH" \
    --require-git-commit
fi

if [ -f "$BUILD_INFO_PATH" ]; then
  echo ""
  echo "🧾 Gateway build info:"
  echo "   File: $BUILD_INFO_PATH"
  grep -E '"(build_id|git_commit|git_branch|git_dirty|build_time_utc)"' "$BUILD_INFO_PATH" || true
else
  echo ""
  echo "⚠️  WARNING: $BUILD_INFO_PATH is missing; live commit will be reported as unknown."
  if [ "${GATEWAY_REQUIRE_KNOWN_BUILD_INFO:-false}" = "true" ]; then
    echo "ERROR: GATEWAY_REQUIRE_KNOWN_BUILD_INFO=true and no BUILD_INFO.json was found" >&2
    exit 1
  fi
fi

# Stage top-level runtime packages into the gateway build context so both PCR0
# and the gateway TEE code_hash cover the code actually imported at runtime.
bash "$SCRIPT_DIR/stage_attested_runtime.sh"

# Force fresh build (no cache) to ensure latest code is included
# Build from ~/gateway/ so .dockerignore works properly
# On EC2: ~/tee/ and ~/gateway/ are sibling directories
docker build --no-cache -f ~/tee/Dockerfile.enclave -t tee-enclave:latest ~/gateway/

# Step 2: Build enclave image file (.eif)
echo ""
echo "🔐 Step 2: Building enclave image file (.eif)..."
nitro-cli build-enclave \
  --docker-uri tee-enclave:latest \
  --output-file tee-enclave.eif \
  | tee enclave_build_output.txt

# Step 3: Extract measurements
echo ""
echo "📊 Step 3: Extracting enclave measurements..."
echo ""
echo "✅ Enclave built successfully!"
echo ""
echo "Important values (save these):"
echo "------------------------------"
grep "PCR0" enclave_build_output.txt || echo "(PCR0 not found in output)"
grep "PCR1" enclave_build_output.txt || echo "(PCR1 not found in output)"
grep "PCR2" enclave_build_output.txt || echo "(PCR2 not found in output)"
echo ""
echo "Next steps:"
echo "1. Run enclave: bash start_enclave.sh"
echo "2. Check status: sudo nitro-cli describe-enclaves"
echo "3. View logs: sudo nitro-cli console --enclave-id <ENCLAVE_ID>"
echo ""
