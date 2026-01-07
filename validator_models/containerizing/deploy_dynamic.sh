#!/bin/bash

# ==============================================================================
# DYNAMIC LeadPoet Containerized Validator Deployment
# ==============================================================================
# This script automatically detects ALL proxies in .env.docker and spawns
# the correct number of containers with FULLY DYNAMIC lead distribution.
#
# Lead distribution is calculated at runtime based on gateway MAX_LEADS_PER_EPOCH.
# No need to specify lead counts - it adapts automatically!
#
# Usage:
#   ./deploy_dynamic.sh
#
# ==============================================================================

set -e

echo "============================================================"
echo "🐳 DYNAMIC CONTAINERIZED VALIDATOR DEPLOYMENT"
echo "============================================================"
echo "📊 Lead distribution: FULLY DYNAMIC (adapts to gateway setting)"
echo ""

# SIMPLIFIED CONFIGURATION: Read from main .env file
# Validators just add WEBSHARE_PROXY_1 and WEBSHARE_PROXY_2 to their existing .env
MAIN_ENV_PATH="../../.env"

if [ -f "$MAIN_ENV_PATH" ]; then
    echo "📋 Loading configuration from main .env file..."
    source "$MAIN_ENV_PATH"
    echo "✅ Loaded from $MAIN_ENV_PATH"
else
    echo "❌ ERROR: Main .env file not found at $MAIN_ENV_PATH"
    echo ""
    echo "Expected location: ~/leadpoet/leadpoet/.env"
    echo ""
    echo "Please ensure your .env file exists with:"
    echo "  - API keys (TRUELIST_API_KEY, SCRAPINGDOG_API_KEY, etc.)"
    echo "  - Proxy URLs (WEBSHARE_PROXY_1, WEBSHARE_PROXY_2)"
    echo ""
    exit 1
fi

# OPTIONAL: Allow .env.docker to override main .env settings
if [ -f ".env.docker" ]; then
    echo "📋 Loading overrides from .env.docker..."
    source .env.docker
    echo "✅ Overrides loaded from .env.docker"
fi

echo ""

# Auto-detect proxies from .env.docker
PROXIES=()
PROXY_COUNT=0

# Check for WEBSHARE_PROXY_1, WEBSHARE_PROXY_2, WEBSHARE_PROXY_3, etc.
for i in {1..49}; do
    PROXY_VAR="WEBSHARE_PROXY_$i"
    PROXY_VALUE="${!PROXY_VAR}"
    
    if [ -n "$PROXY_VALUE" ] && [ "$PROXY_VALUE" != "http://YOUR_USERNAME:YOUR_PASSWORD@p.webshare.io:80" ]; then
        PROXIES+=("$PROXY_VALUE")
        PROXY_COUNT=$((PROXY_COUNT + 1))
    fi
done

# Get enclave CID for TEE signing (if enclave is running)
ENCLAVE_CID=""
if command -v nitro-cli &> /dev/null; then
    ENCLAVE_CID=$(nitro-cli describe-enclaves 2>/dev/null | grep -o '"EnclaveCID": [0-9]*' | head -1 | grep -o '[0-9]*' || true)
    if [ -n "$ENCLAVE_CID" ]; then
        echo "🔐 Detected running Nitro Enclave with CID: $ENCLAVE_CID"
    fi
fi

# Calculate total containers (main + workers)
# Main container uses EC2 IP (no proxy)
# Each proxy gets 1 worker container
TOTAL_CONTAINERS=$((PROXY_COUNT + 1))

echo "🔍 Auto-detected proxies: $PROXY_COUNT"
echo "📦 Total containers to deploy: $TOTAL_CONTAINERS"
echo "   - 1x Main validator (EC2 native IP)"
echo "   - ${PROXY_COUNT}x Worker containers (proxied)"
echo ""

if [ $PROXY_COUNT -eq 0 ]; then
    echo "⚠️  WARNING: No proxies configured in .env.docker"
    echo ""
    echo "For parallel processing with different IPs, add proxies:"
    echo "  WEBSHARE_PROXY_1=http://user:pass@p.webshare.io:80"
    echo "  WEBSHARE_PROXY_2=http://user:pass@p.webshare.io:80"
    echo "  WEBSHARE_PROXY_3=http://user:pass@p.webshare.io:80"
    echo ""
    echo "Deploying with 1 container (main validator only)..."
    echo ""
fi

echo "📊 Lead distribution: DYNAMIC (each container auto-calculates based on gateway setting)"
echo ""

# Verify required API keys
# Email verification: Require EITHER MEV_API_KEY OR TRUELIST_API_KEY (not both)
if [ -z "$MEV_API_KEY" ] && [ -z "$TRUELIST_API_KEY" ]; then
    echo "❌ ERROR: No email verification API key configured in .env"
    echo "   Please set EITHER:"
    echo "   - MEV_API_KEY (MyEmailVerifier) OR"
    echo "   - TRUELIST_API_KEY (TrueList)"
    echo ""
    echo "   The validator will automatically use whichever is available."
    exit 1
fi

# Other required API keys
if [ -z "$SCRAPINGDOG_API_KEY" ] || [ -z "$OPENROUTER_KEY" ]; then
    echo "❌ ERROR: Required API keys not set in .env"
    echo "   Please set: SCRAPINGDOG_API_KEY, OPENROUTER_KEY"
    exit 1
fi

# Build Docker image (from repo root, using Dockerfile in this directory)
echo "🔨 Building Docker image..."
cd "$(dirname "$0")"  # Go to script directory
SCRIPT_DIR=$(pwd)
REPO_ROOT=$(cd ../.. && pwd)  # Go to repo root

if docker build -f "$SCRIPT_DIR/Dockerfile" -t leadpoet-validator:latest "$REPO_ROOT"; then
    echo "✅ Docker image built successfully"
else
    echo "❌ ERROR: Docker build failed"
    echo "   This usually means:"
    echo "   1. Dockerfile syntax error"
    echo "   2. Missing dependencies in requirements.txt"
    echo "   3. Network issues downloading packages"
    echo ""
    echo "   Run manually to see full error:"
    echo "   cd ~/leadpoet/leadpoet"
    echo "   docker build -f validator_models/containerizing/Dockerfile -t leadpoet-validator:latest ."
    exit 1
fi
echo ""

# Stop and remove existing containers
echo "🛑 Stopping existing containers (if any)..."
docker ps -a --filter "name=leadpoet-validator" --format "{{.Names}}" | while read container; do
    docker rm -f "$container" 2>/dev/null || true
done
echo "✅ Old containers removed"
echo ""

# Function to start a container
start_container() {
    local CONTAINER_NAME=$1
    local PROXY_URL=$2
    local CONTAINER_ID=$3
    local DISPLAY_NAME=$4
    
    echo "🚀 Starting $DISPLAY_NAME..."
    echo "   Container ID: $CONTAINER_ID / $TOTAL_CONTAINERS"
    if [ -n "$PROXY_URL" ]; then
        echo "   Proxy: ${PROXY_URL:0:30}..."
    else
        echo "   Proxy: None (EC2 native IP)"
    fi
    echo "   Lead distribution: AUTO (gateway MAX_LEADS_PER_EPOCH ÷ $TOTAL_CONTAINERS)"
    
    local PROXY_ARGS=""
    if [ -n "$PROXY_URL" ]; then
        PROXY_ARGS="-e HTTP_PROXY=$PROXY_URL -e HTTPS_PROXY=$PROXY_URL"
    fi
    
    # Determine container mode (ID 0 = coordinator, others = worker)
    local MODE_ARG=""
    local VSOCK_ARG=""
    local ENCLAVE_CID_ARG=""
    local PRIVILEGED_ARG=""
    if [ "$CONTAINER_ID" -eq 0 ]; then
        MODE_ARG="--mode coordinator"
        # Coordinator needs vsock access for Nitro Enclave TEE signing
        # Requires --privileged for vsock socket creation permissions
        if [ -e /dev/vsock ]; then
            VSOCK_ARG="--device /dev/vsock"
            PRIVILEGED_ARG="--privileged"
            echo "   🔐 Enabling vsock for TEE signing (privileged mode)"
        fi
        # Pass enclave CID if available
        if [ -n "$ENCLAVE_CID" ]; then
            ENCLAVE_CID_ARG="-e ENCLAVE_CID=$ENCLAVE_CID"
            echo "   🔐 Passing ENCLAVE_CID=$ENCLAVE_CID"
        fi
    else
        MODE_ARG="--mode worker"
    fi
    
    docker run -d \
      --name "$CONTAINER_NAME" \
      --network host \
      --restart unless-stopped \
      $PRIVILEGED_ARG \
      -v ~/.bittensor/wallets:/root/.bittensor/wallets:ro \
      -v "$REPO_ROOT/validator_weights:/app/validator_weights" \
      -e LEADPOET_CONTAINER_MODE=1 \
      -e LEADPOET_WRAPPER_ACTIVE=1 \
      -e MEV_API_KEY="$MEV_API_KEY" \
      -e TRUELIST_API_KEY="$TRUELIST_API_KEY" \
      -e SCRAPINGDOG_API_KEY="$SCRAPINGDOG_API_KEY" \
      -e OPENROUTER_KEY="$OPENROUTER_KEY" \
      -e COMPANIES_HOUSE_API_KEY="$COMPANIES_HOUSE_API_KEY" \
      -e ENABLE_TEE_SUBMISSION="${ENABLE_TEE_SUBMISSION:-false}" \
      -e GATEWAY_URL="${GATEWAY_URL:-http://54.226.209.164:8000}" \
      $ENCLAVE_CID_ARG \
      $VSOCK_ARG \
      $PROXY_ARGS \
      leadpoet-validator:latest \
      --netuid 71 \
      --subtensor_network finney \
      --wallet_name validator_72 \
      --wallet_hotkey default \
      --container-id "$CONTAINER_ID" \
      --total-containers "$TOTAL_CONTAINERS" \
      $MODE_ARG > /dev/null
    
    echo "   ✅ Started: $CONTAINER_NAME"
    echo ""
}

# Deploy containers
echo "============================================================"
echo "🚀 DEPLOYING CONTAINERS"
echo "============================================================"
echo ""

# Container 0: Coordinator (no proxy, ID=0)
start_container "leadpoet-validator-main" "" 0 "Container 0: Coordinator"

# Deploy worker containers (one per proxy, ID=1, 2, 3, ...)
for i in $(seq 1 $PROXY_COUNT); do
    PROXY_URL="${PROXIES[$((i-1))]}"
    CONTAINER_ID=$i
    start_container "leadpoet-validator-worker-$i" "$PROXY_URL" "$CONTAINER_ID" "Container $CONTAINER_ID: Worker #$i"
done

# Wait for containers to start
echo "⏳ Waiting 10 seconds for containers to initialize..."
sleep 10

# Check status
echo ""
echo "============================================================"
echo "📊 CONTAINER STATUS"
echo "============================================================"
docker ps --filter "name=leadpoet-validator" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
echo ""

# Verify proxies
echo "============================================================"
echo "🌐 VERIFYING PROXY IPS"
echo "============================================================"
echo ""
echo "⏳ Waiting 30 seconds for validators to fully initialize..."
sleep 30

ALL_IPS=()

echo "🔍 Container: leadpoet-validator-main (should show EC2 IP)"
MAIN_IP=$(docker exec leadpoet-validator-main curl -s --max-time 10 https://api.ipify.org 2>/dev/null || echo "ERROR")
echo "   IP: $MAIN_IP"
ALL_IPS+=("$MAIN_IP")
echo ""

for i in $(seq 1 $PROXY_COUNT); do
    CONTAINER_NAME="leadpoet-validator-worker-$i"
    echo "🔍 Container: $CONTAINER_NAME (should show Webshare Proxy #$i IP)"
    WORKER_IP=$(docker exec "$CONTAINER_NAME" curl -s --max-time 10 https://api.ipify.org 2>/dev/null || echo "ERROR")
    echo "   IP: $WORKER_IP"
    ALL_IPS+=("$WORKER_IP")
    echo ""
done

# Check for duplicate IPs
echo "🔍 Checking for duplicate IPs..."
UNIQUE_IPS=($(printf '%s\n' "${ALL_IPS[@]}" | sort -u))
UNIQUE_COUNT=${#UNIQUE_IPS[@]}
TOTAL_COUNT=${#ALL_IPS[@]}

if [ $UNIQUE_COUNT -eq $TOTAL_COUNT ]; then
    echo "   ✅ SUCCESS: All $TOTAL_COUNT containers have DIFFERENT IPs!"
else
    echo "   ⚠️  WARNING: Found duplicate IPs!"
    echo "   Total containers: $TOTAL_COUNT"
    echo "   Unique IPs: $UNIQUE_COUNT"
    echo ""
    echo "   This means some containers are sharing IPs, which may cause rate limiting."
    echo "   Please check your proxy configuration in .env.docker"
fi
echo ""

# Summary
echo "============================================================"
echo "✅ DEPLOYMENT COMPLETE"
echo "============================================================"
echo ""
echo "📊 Summary:"
echo "   - Total containers: $TOTAL_CONTAINERS"
echo "   - Lead distribution: FULLY DYNAMIC (adapts to gateway MAX_LEADS_PER_EPOCH)"
echo "   - Unique IPs: $UNIQUE_COUNT / $TOTAL_COUNT"
echo ""
echo "   Examples of auto-scaling:"
echo "   - Gateway @ 170 leads → Each container: ~57 leads"
echo "   - Gateway @ 900 leads → Each container: 300 leads"
echo "   - Gateway @ 1200 leads → Each container: 400 leads"
echo ""
echo "📋 Next Steps:"
echo "   1. Monitor logs: docker logs -f leadpoet-validator-main"
echo "   2. Check resource usage: docker stats"
echo "   3. Verify lead distribution in logs (each container shows its range)"
echo ""
echo "🔧 To scale up (add more containers):"
echo "   1. Get another proxy from https://www.webshare.io/"
echo "   2. Add WEBSHARE_PROXY_$((PROXY_COUNT + 1))=... to .env.docker"
echo "   3. Run: ./deploy_dynamic.sh"
echo "   Done! New container auto-joins and gets its share of leads."
echo ""

