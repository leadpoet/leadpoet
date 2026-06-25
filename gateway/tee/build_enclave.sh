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

# Stage top-level runtime packages into the gateway build context so both PCR0
# and the gateway TEE code_hash cover the code actually imported at runtime.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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
