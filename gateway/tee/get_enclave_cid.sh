#!/bin/bash
#
# Get Enclave CID from nitro-cli
# ==============================
# Returns the CID of the first running enclave
#

# Get enclave info (with sudo)
ENCLAVE_INFO=$(sudo nitro-cli describe-enclaves 2>/dev/null)

# Check if any enclaves are running
if [ -z "$ENCLAVE_INFO" ] || [ "$ENCLAVE_INFO" = "[]" ]; then
    echo "ERROR: No enclaves running" >&2
    exit 1
fi

# Extract CID using jq (or python if jq not available)
if command -v jq &> /dev/null; then
    CID=$(echo "$ENCLAVE_INFO" | jq -r '.[0].EnclaveCID')
else
    CID=$(echo "$ENCLAVE_INFO" | python3 -c "import sys, json; print(json.load(sys.stdin)[0]['EnclaveCID'])")
fi

echo "$CID"

