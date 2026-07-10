#!/bin/bash
################################################################################
# Start TEE Enclave - AWS Nitro Enclaves
################################################################################
# 
# This script starts the Nitro Enclave with the TEE service.
# Must be run with sudo on the parent EC2 instance.
#
# Usage: sudo bash start_enclave.sh
#

set -e  # Exit on error

echo "================================================================================"
echo "🚀 STARTING TEE ENCLAVE"
echo "================================================================================"

# Configuration
# Use absolute paths (not $HOME) since this runs with sudo. The restart wrapper
# intentionally invokes this script through sudo, so read the two non-secret
# resource values from the hydrated gateway env file instead of relying on
# sudo preserving the caller's environment.
EIF_PATH="/home/ec2-user/tee/tee-enclave.eif"
ENCLAVE_CID=16
GATEWAY_ENV_FILE="${GATEWAY_ENV_FILE:-/home/ec2-user/.config/leadpoet/gateway.env}"

read_numeric_env_setting() {
    local key="$1"
    local fallback="$2"
    local value="${!key:-}"

    if [ -z "$value" ] && [ -r "$GATEWAY_ENV_FILE" ]; then
        value="$(
            grep -E "^[[:space:]]*(export[[:space:]]+)?${key}[[:space:]]*=" "$GATEWAY_ENV_FILE" \
                | tail -n 1 \
                | cut -d= -f2- \
                | tr -d "'\"[:space:]" \
                || true
        )"
    fi
    value="${value:-$fallback}"
    if ! [[ "$value" =~ ^[0-9]+$ ]]; then
        echo "ERROR: ${key} must be a positive integer" >&2
        exit 1
    fi
    printf '%s' "$value"
}

CPU_COUNT="$(read_numeric_env_setting GATEWAY_ENCLAVE_CPU_COUNT 2)"
MEMORY_MB="$(read_numeric_env_setting GATEWAY_ENCLAVE_MEMORY_MB 8192)"

if [ "$CPU_COUNT" -lt 2 ] || [ "$MEMORY_MB" -lt 8192 ]; then
    echo "ERROR: gateway enclave requires at least 2 vCPUs and 8192 MB" >&2
    exit 1
fi

# Check if EIF exists
if [ ! -f "$EIF_PATH" ]; then
    echo "❌ ERROR: Enclave image not found at $EIF_PATH"
    echo "   Run: cd ~/tee && bash build_enclave.sh"
    exit 1
fi

echo "📦 Enclave Image: $EIF_PATH"
echo "🔢 CID: $ENCLAVE_CID"
echo "🧮 CPU: $CPU_COUNT cores"
echo "💾 Memory: ${MEMORY_MB} MB"
echo ""

# Check if enclave already running
RUNNING=$(sudo nitro-cli describe-enclaves 2>/dev/null | jq -r 'length')
if [ "$RUNNING" -gt 0 ]; then
    echo "⚠️  WARNING: Enclave already running. Stopping..."
    sudo nitro-cli terminate-enclave --all
    sleep 2
fi

# Start enclave
echo "🚀 Starting enclave..."
sudo nitro-cli run-enclave \
  --cpu-count $CPU_COUNT \
  --memory $MEMORY_MB \
  --eif-path "$EIF_PATH" \
  --enclave-cid $ENCLAVE_CID

echo ""
echo "✅ Enclave started!"
echo ""

# Show status
echo "📊 Enclave Status:"
sudo nitro-cli describe-enclaves

# Get enclave ID
ENCLAVE_ID=$(sudo nitro-cli describe-enclaves | jq -r '.[0].EnclaveID')

echo ""
echo "================================================================================"
echo "✅ ENCLAVE RUNNING"
echo "================================================================================"
echo "Enclave ID: $ENCLAVE_ID"
echo "CID: $ENCLAVE_CID"
echo ""
echo "⏳ Waiting 15 seconds for enclave service to initialize..."
sleep 15
echo "✅ Enclave service should be ready"
echo ""
echo "Next steps:"
echo "  1. Provision PCRs: python3 ~/tee/provision_pcrs.py"
echo "  2. Test enclave:   python3 ~/tee/test_enclave_rpc.py"
echo "  3. View console:   sudo nitro-cli console --enclave-id $ENCLAVE_ID"
echo "================================================================================"
