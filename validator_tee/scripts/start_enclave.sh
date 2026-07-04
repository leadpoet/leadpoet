#!/bin/bash
#
# Start Validator Nitro Enclave
# =============================
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VALIDATOR_TEE_DIR="$(dirname "$SCRIPT_DIR")"
EIF_FILE="$VALIDATOR_TEE_DIR/validator-enclave.eif"
ENCLAVE_NAME="${VALIDATOR_ENCLAVE_NAME:-validator-enclave}"
ENCLAVE_CPU_COUNT="${VALIDATOR_ENCLAVE_CPU_COUNT:-2}"
ENCLAVE_MEMORY_MIB="${VALIDATOR_ENCLAVE_MEMORY_MIB:-1024}"
ENCLAVE_CID="${VALIDATOR_ENCLAVE_CID:-}"
DEBUG_MODE="${VALIDATOR_ENCLAVE_DEBUG_MODE:-false}"

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

# Start enclave.
# Production default is non-debug. Debug enclaves return all-zero PCRs in
# attestation documents, which the gateway correctly rejects for weight bundles.
echo ""
echo "📦 Starting enclave..."
echo "   EIF: $EIF_FILE"
echo "   Name: $ENCLAVE_NAME"
echo "   Memory: $ENCLAVE_MEMORY_MIB MB"
echo "   CPUs: $ENCLAVE_CPU_COUNT"
if [ -n "$ENCLAVE_CID" ]; then
    echo "   CID: $ENCLAVE_CID"
fi
if [ "$DEBUG_MODE" = "true" ] || [ "$DEBUG_MODE" = "1" ] || [ "$DEBUG_MODE" = "yes" ]; then
    echo "   Mode: DEBUG (PCR0 will not be accepted by production gateway)"
else
    echo "   Mode: PRODUCTION"
fi
echo ""

RUN_ARGS=(
    run-enclave
    --eif-path "$EIF_FILE"
    --cpu-count "$ENCLAVE_CPU_COUNT"
    --memory "$ENCLAVE_MEMORY_MIB"
    --enclave-name "$ENCLAVE_NAME"
)

if [ -n "$ENCLAVE_CID" ]; then
    RUN_ARGS+=(--enclave-cid "$ENCLAVE_CID")
fi

if [ "$DEBUG_MODE" = "true" ] || [ "$DEBUG_MODE" = "1" ] || [ "$DEBUG_MODE" = "yes" ]; then
    RUN_ARGS+=(--debug-mode)
fi

nitro-cli "${RUN_ARGS[@]}"

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
