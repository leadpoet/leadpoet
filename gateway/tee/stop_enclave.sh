#!/bin/bash
################################################################################
# Stop TEE Enclave - AWS Nitro Enclaves
################################################################################
#
# This script stops all running Nitro Enclaves.
# Must be run with sudo on the parent EC2 instance.
#
# Usage: sudo bash stop_enclave.sh
#

set -e  # Exit on error

echo "================================================================================"
echo "🛑 STOPPING TEE ENCLAVE"
echo "================================================================================"

# Check if any enclaves running
RUNNING=$(sudo nitro-cli describe-enclaves 2>/dev/null | jq -r 'length')

if [ "$RUNNING" -eq 0 ]; then
    echo "⚠️  No enclaves currently running"
    exit 0
fi

echo "📊 Current enclaves:"
sudo nitro-cli describe-enclaves

echo ""
echo "🛑 Terminating all enclaves..."
sudo nitro-cli terminate-enclave --all

echo ""
echo "✅ All enclaves stopped"
echo "================================================================================"

