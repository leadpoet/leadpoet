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

# Step 1: Build Docker image (use files from current directory)
echo ""
echo "📦 Step 1: Building Docker image..."
# Force fresh build (no cache) to ensure latest code is included
docker build --no-cache -f Dockerfile.enclave -t tee-enclave:latest .

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
