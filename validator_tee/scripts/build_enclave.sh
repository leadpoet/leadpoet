#!/bin/bash
#
# Build Validator Nitro Enclave Image
# ====================================
# This script builds the validator enclave Docker image and converts it to .eif format
#

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VALIDATOR_TEE_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(dirname "$VALIDATOR_TEE_DIR")"

echo "=========================================="
echo "🔨 Building Validator Nitro Enclave Image"
echo "=========================================="
echo ""
echo "Script dir: $SCRIPT_DIR"
echo "Validator TEE dir: $VALIDATOR_TEE_DIR"
echo "Repo root: $REPO_ROOT"
echo ""

# Step 1: Build Docker image
echo "📦 Step 1: Building Docker image..."
echo "   Build context: $REPO_ROOT"
echo "   Dockerfile: $VALIDATOR_TEE_DIR/Dockerfile.enclave"

# Force fresh build (no cache) to ensure latest code
docker build --no-cache \
    -f "$VALIDATOR_TEE_DIR/Dockerfile.enclave" \
    -t validator-tee-enclave:latest \
    "$REPO_ROOT"

# Step 2: Build enclave image file (.eif)
echo ""
echo "🔐 Step 2: Building enclave image file (.eif)..."

cd "$VALIDATOR_TEE_DIR"
nitro-cli build-enclave \
    --docker-uri validator-tee-enclave:latest \
    --output-file validator-enclave.eif \
    | tee enclave_build_output.txt

# Step 3: Extract measurements
echo ""
echo "📊 Step 3: Extracting enclave measurements..."
echo ""
echo "✅ Validator enclave built successfully!"
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "IMPORTANT - SAVE THESE VALUES:"
echo "═══════════════════════════════════════════════════════════"
grep -E "PCR0|PCR1|PCR2" enclave_build_output.txt || echo "(PCR values not found)"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "The PCR0 value above must be added to:"
echo "  leadpoet_canonical/nitro.py -> ALLOWED_VALIDATOR_PCR0_VALUES"
echo ""
echo "Next steps:"
echo "  1. Run enclave: bash scripts/start_enclave.sh"
echo "  2. Check status: nitro-cli describe-enclaves"
echo "  3. View logs: nitro-cli console --enclave-id <ID>"
echo ""
