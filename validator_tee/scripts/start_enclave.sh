#!/bin/bash
#
# Start Validator Nitro Enclave
# =============================
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VALIDATOR_TEE_DIR="$(dirname "$SCRIPT_DIR")"
EIF_FILE="$VALIDATOR_TEE_DIR/validator-enclave.eif"

echo "=========================================="
echo "🚀 Starting Validator Nitro Enclave"
echo "=========================================="

# Check if EIF exists
if [ ! -f "$EIF_FILE" ]; then
    echo "❌ Error: $EIF_FILE not found!"
    echo "   Run: bash scripts/build_enclave.sh first"
    exit 1
fi

# Check if enclave already running
RUNNING=$(nitro-cli describe-enclaves | grep -c "RUNNING" || true)
if [ "$RUNNING" -gt 0 ]; then
    echo "⚠️  An enclave is already running!"
    echo ""
    nitro-cli describe-enclaves
    echo ""
    echo "To stop it: bash scripts/stop_enclave.sh"
    exit 1
fi

# Start enclave
# Memory: 1024 MB (minimum required for validator EIF)
# CPU: 2 (configured in allocator)
echo ""
echo "📦 Starting enclave..."
echo "   EIF: $EIF_FILE"
echo "   Memory: 1024 MB"
echo "   CPUs: 2"
echo ""

nitro-cli run-enclave \
    --eif-path "$EIF_FILE" \
    --cpu-count 2 \
    --memory 1024 \
    --debug-mode

echo ""
echo "✅ Enclave started!"
echo ""
echo "Enclave details:"
nitro-cli describe-enclaves
echo ""
echo "To view logs:"
echo "  nitro-cli console --enclave-id <ENCLAVE_ID>"
echo ""
echo "To test connection from host:"
echo "  cd ~/leadpoet/leadpoet"
echo "  python3 -c \"from validator_tee.host.vsock_client import ValidatorEnclaveClient; c = ValidatorEnclaveClient(); print('Public Key:', c.get_public_key())\""
echo ""
