#!/bin/bash

# ==============================================================================
# DYNAMIC LeadPoet Containerized Validator Deployment
# ==============================================================================
# This script automatically detects ALL proxies in .env.docker and spawns
# the correct number of containers with auto-calculated lead ranges.
#
# Usage:
#   ./deploy_dynamic.sh [MAX_LEADS]
#
# Example:
#   ./deploy_dynamic.sh 510        # Deploy for 510 leads
#   ./deploy_dynamic.sh 1000       # Deploy for 1000 leads
# ==============================================================================

set -e

# Default max leads (override with CLI arg)
MAX_LEADS=${1:-510}

echo "============================================================"
echo "🐳 DYNAMIC CONTAINERIZED VALIDATOR DEPLOYMENT"
echo "============================================================"
echo "📊 Max leads to process: $MAX_LEADS"
echo ""

# Check if .env.docker exists
if [ ! -f ".env.docker" ]; then
    echo "❌ ERROR: .env.docker file not found!"
    echo ""
    echo "Please create .env.docker with your configuration:"
    echo "  cp docker.env.example .env.docker"
    echo "  nano .env.docker"
    echo ""
    exit 1
fi

# Load proxy URLs from .env.docker
echo "📋 Loading proxy configuration from .env.docker..."
source .env.docker

# Load API keys from main .env file if it exists (fallback if not in .env.docker)
if [ -z "$TRUELIST_API_KEY" ] && [ -f "../../.env" ]; then
    echo "📋 Loading API keys from main .env file..."
    source ../../.env
fi

echo "✅ Environment variables loaded"
echo ""

# Auto-detect proxies from .env.docker
PROXIES=()
PROXY_COUNT=0

# Check for WEBSHARE_PROXY_1, WEBSHARE_PROXY_2, WEBSHARE_PROXY_3, etc.
for i in {1..20}; do
    PROXY_VAR="WEBSHARE_PROXY_$i"
    PROXY_VALUE="${!PROXY_VAR}"
    
    if [ -n "$PROXY_VALUE" ] && [ "$PROXY_VALUE" != "http://YOUR_USERNAME:YOUR_PASSWORD@p.webshare.io:80" ]; then
        PROXIES+=("$PROXY_VALUE")
        ((PROXY_COUNT++))
    fi
done

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

# Calculate leads per container
LEADS_PER_CONTAINER=$((MAX_LEADS / TOTAL_CONTAINERS))
echo "📊 Distribution: ~$LEADS_PER_CONTAINER leads per container"
echo ""

# Verify required API keys
if [ -z "$TRUELIST_API_KEY" ] || [ -z "$SCRAPINGDOG_API_KEY" ]; then
    echo "❌ ERROR: Required API keys not set in .env.docker"
    echo "   Please set: TRUELIST_API_KEY, SCRAPINGDOG_API_KEY, OPENROUTER_KEY"
    exit 1
fi

# Build Docker image (from repo root, using Dockerfile in this directory)
echo "🔨 Building Docker image..."
cd "$(dirname "$0")"  # Go to script directory
SCRIPT_DIR=$(pwd)
REPO_ROOT=$(cd ../.. && pwd)  # Go to repo root

docker build -f "$SCRIPT_DIR/Dockerfile" -t leadpoet-validator:latest "$REPO_ROOT" 2>&1 | grep -E "(Step|Successfully|FINISHED)" || true
echo "✅ Docker image built"
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
    local START_LEAD=$3
    local END_LEAD=$4
    local DISPLAY_NAME=$5
    
    echo "🚀 Starting $DISPLAY_NAME..."
    echo "   Range: leads $START_LEAD-$END_LEAD ($((END_LEAD - START_LEAD)) leads)"
    if [ -n "$PROXY_URL" ]; then
        echo "   Proxy: ${PROXY_URL:0:30}..."
    else
        echo "   Proxy: None (EC2 native IP)"
    fi
    
    local PROXY_ARGS=""
    if [ -n "$PROXY_URL" ]; then
        PROXY_ARGS="-e HTTP_PROXY=$PROXY_URL -e HTTPS_PROXY=$PROXY_URL"
    fi
    
    # Determine container mode (main = coordinator, workers = worker)
    local MODE_ARG=""
    if [ "$CONTAINER_NAME" = "leadpoet-validator-main" ]; then
        MODE_ARG="--mode coordinator"
    else
        MODE_ARG="--mode worker"
    fi
    
    docker run -d \
      --name "$CONTAINER_NAME" \
      --network host \
      --restart unless-stopped \
      -v ~/.bittensor/wallets:/root/.bittensor/wallets:ro \
      -v "$REPO_ROOT/validator_weights:/app/validator_weights" \
      -e TRUELIST_API_KEY="$TRUELIST_API_KEY" \
      -e SCRAPINGDOG_API_KEY="$SCRAPINGDOG_API_KEY" \
      -e OPENROUTER_KEY="$OPENROUTER_KEY" \
      -e COMPANIES_HOUSE_API_KEY="$COMPANIES_HOUSE_API_KEY" \
      $PROXY_ARGS \
      leadpoet-validator:latest \
      --netuid 71 \
      --subtensor_network finney \
      --wallet_name validator_72 \
      --wallet_hotkey default \
      --lead-range "$START_LEAD-$END_LEAD" \
      $MODE_ARG > /dev/null
    
    echo "   ✅ Started: $CONTAINER_NAME"
    echo ""
}

# Deploy containers
echo "============================================================"
echo "🚀 DEPLOYING CONTAINERS"
echo "============================================================"
echo ""

# Container 1: Main validator (no proxy)
START_LEAD=0
END_LEAD=$LEADS_PER_CONTAINER
start_container "leadpoet-validator-main" "" $START_LEAD $END_LEAD "Container 1: Main Validator"

# Deploy worker containers (one per proxy)
for i in $(seq 1 $PROXY_COUNT); do
    CONTAINER_NUM=$((i + 1))
    PROXY_URL="${PROXIES[$((i-1))]}"
    START_LEAD=$((i * LEADS_PER_CONTAINER))
    END_LEAD=$(((i + 1) * LEADS_PER_CONTAINER))
    
    # Last container gets any remaining leads
    if [ $i -eq $PROXY_COUNT ]; then
        END_LEAD=$MAX_LEADS
    fi
    
    start_container "leadpoet-validator-worker-$i" "$PROXY_URL" $START_LEAD $END_LEAD "Container $CONTAINER_NUM: Worker #$i"
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
echo "   - Leads per epoch: $MAX_LEADS"
echo "   - Leads per container: ~$LEADS_PER_CONTAINER"
echo "   - Unique IPs: $UNIQUE_COUNT / $TOTAL_COUNT"
echo ""
echo "📋 Next Steps:"
echo "   1. Monitor logs: docker logs -f leadpoet-validator-main"
echo "   2. Check resource usage: docker stats"
echo "   3. View all logs: docker-compose logs -f (if using docker-compose)"
echo ""
echo "🔧 To scale up (add more containers):"
echo "   1. Get another proxy from https://www.webshare.io/"
echo "   2. Add WEBSHARE_PROXY_$((PROXY_COUNT + 1))=... to .env.docker"
echo "   3. Run: ./deploy_dynamic.sh $MAX_LEADS"
echo "   Done! New container will auto-deploy with auto-calculated lead range."
echo ""

