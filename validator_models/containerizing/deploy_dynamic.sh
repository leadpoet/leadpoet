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

terminate_stale_validator_builds() {
    local reason="${1:-preflight}"
    echo "🧹 Cleaning stale validator Docker build processes ($reason)..."

    if command -v sudo >/dev/null 2>&1 && sudo -n true >/dev/null 2>&1; then
        sudo -n pkill -TERM -f "docker build -f .*/validator_models/containerizing/Dockerfile" 2>/dev/null || true
        sudo -n pkill -TERM -f "pip install --no-cache-dir -r requirements.txt" 2>/dev/null || true
        sleep 3
        sudo -n pkill -KILL -f "docker build -f .*/validator_models/containerizing/Dockerfile" 2>/dev/null || true
        sudo -n pkill -KILL -f "pip install --no-cache-dir -r requirements.txt" 2>/dev/null || true
    else
        pkill -TERM -f "docker build -f .*/validator_models/containerizing/Dockerfile" 2>/dev/null || true
        pkill -TERM -f "pip install --no-cache-dir -r requirements.txt" 2>/dev/null || true
        sleep 3
        pkill -KILL -f "docker build -f .*/validator_models/containerizing/Dockerfile" 2>/dev/null || true
        pkill -KILL -f "pip install --no-cache-dir -r requirements.txt" 2>/dev/null || true
    fi
    sleep 2
}

docker_build_validator_image() {
    local timeout_seconds="${VALIDATOR_DOCKER_BUILD_TIMEOUT_SECONDS:-1800}"
    local build_context build_status
    build_context="$(mktemp -d /tmp/leadpoet-validator-build.XXXXXX)"
    git -C "$REPO_ROOT" archive "$VALIDATOR_V2_DEPLOY_COMMIT" \
        | tar -x -C "$build_context"

    if command -v timeout >/dev/null 2>&1; then
        timeout "$timeout_seconds" docker build \
            --build-arg "LEADPOET_BUILD_COMMIT=$VALIDATOR_V2_DEPLOY_COMMIT" \
            -f "$build_context/validator_models/containerizing/Dockerfile" \
            -t leadpoet-validator:latest \
            "$build_context" && build_status=0 || build_status=$?
    else
        echo "⚠️  timeout command unavailable; running Docker build without timeout"
        docker build \
            --build-arg "LEADPOET_BUILD_COMMIT=$VALIDATOR_V2_DEPLOY_COMMIT" \
            -f "$build_context/validator_models/containerizing/Dockerfile" \
            -t leadpoet-validator:latest \
            "$build_context" && build_status=0 || build_status=$?
    fi
    rm -rf "$build_context"
    return "$build_status"
}

# ============================================================
# EXACT-COMMIT GATE: the restart wrapper already selected and verified HEAD.
# ============================================================
# This script must never fetch or reset after the validator EIF build. Doing so
# could launch host code from a different commit than the measured enclave.
# ============================================================
REPO_DIR="/home/ec2-user/leadpoet/leadpoet"
if ! [[ "${VALIDATOR_V2_DEPLOY_COMMIT:-}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "❌ ERROR: VALIDATOR_V2_DEPLOY_COMMIT must be the approved full Git commit" >&2
    exit 1
fi
if [ ! -d "$REPO_DIR/.git" ]; then
    echo "❌ ERROR: validator repository is not a Git checkout: $REPO_DIR" >&2
    exit 1
fi
HEAD_SHA="$(git -C "$REPO_DIR" rev-parse HEAD)"
if [ "$HEAD_SHA" != "$VALIDATOR_V2_DEPLOY_COMMIT" ]; then
    echo "❌ ERROR: validator checkout moved after EIF approval" >&2
    echo "   approved=$VALIDATOR_V2_DEPLOY_COMMIT observed=$HEAD_SHA" >&2
    exit 1
fi
echo "✅ Exact validator commit remains active: $HEAD_SHA"
echo ""

# SIMPLIFIED CONFIGURATION: Read from main .env file
# Validators just add WEBSHARE_PROXY_1 and WEBSHARE_PROXY_2 to their existing .env
MAIN_ENV_PATH="../../.env"

# validator_restart.sh loads the authoritative production environment from
# Secrets Manager before invoking this script. Local .env files remain fallback
# sources for legacy operator-only settings, but may not override any inherited
# production value.
INHERITED_ENV_FILE="$(mktemp /tmp/validator-inherited-env.XXXXXX)"
INHERITED_GATEWAY_URL_SET="${GATEWAY_URL+x}"
INHERITED_GATEWAY_URL="${GATEWAY_URL:-}"
INHERITED_VALIDATOR_V2_GATEWAY_URL_SET="${VALIDATOR_V2_GATEWAY_URL+x}"
INHERITED_VALIDATOR_V2_GATEWAY_URL="${VALIDATOR_V2_GATEWAY_URL:-}"
python3 - "$INHERITED_ENV_FILE" <<'PY'
import os
import re
import shlex
import sys
from pathlib import Path

destination = Path(sys.argv[1])
lines = []
for key, value in os.environ.items():
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
        lines.append(f"export {key}={shlex.quote(value)}")
destination.write_text("\n".join(sorted(lines)) + "\n", encoding="utf-8")
destination.chmod(0o600)
PY
cleanup_inherited_env() {
    rm -f "$INHERITED_ENV_FILE"
}
trap cleanup_inherited_env EXIT

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

# Restore every inherited exported value after local fallback files have been
# read. This covers protocol, gateway, chain, feature, and secret settings.
source "$INHERITED_ENV_FILE"
if [ "$INHERITED_GATEWAY_URL_SET" = "x" ]; then
    GATEWAY_URL="$INHERITED_GATEWAY_URL"
fi
if [ "$INHERITED_VALIDATOR_V2_GATEWAY_URL_SET" = "x" ]; then
    VALIDATOR_V2_GATEWAY_URL="$INHERITED_VALIDATOR_V2_GATEWAY_URL"
fi
unset INHERITED_GATEWAY_URL_SET INHERITED_GATEWAY_URL
unset INHERITED_VALIDATOR_V2_GATEWAY_URL_SET INHERITED_VALIDATOR_V2_GATEWAY_URL

# Bind runtime metadata to the exact checkout that this script builds. The
# Docker image intentionally excludes .git, so the coordinator must receive
# the full commit SHA explicitly for signed weight binding messages.
VALIDATOR_DEPLOY_SHA="$(git -C "$REPO_DIR" rev-parse HEAD)"
if ! [[ "$VALIDATOR_DEPLOY_SHA" =~ ^[0-9a-f]{40}$ ]]; then
    echo "❌ ERROR: validator deploy commit is unavailable" >&2
    exit 1
fi
VALIDATOR_NETUID="${VALIDATOR_NETUID:-71}"
VALIDATOR_SUBTENSOR_NETWORK="${VALIDATOR_SUBTENSOR_NETWORK:-finney}"
EXPECTED_CHAIN="${EXPECTED_CHAIN:-wss://entrypoint-finney.opentensor.ai:443}"
export VALIDATOR_DEPLOY_SHA VALIDATOR_NETUID VALIDATOR_SUBTENSOR_NETWORK EXPECTED_CHAIN

RESEARCH_LAB_INTERNAL_API_KEY="${RESEARCH_LAB_INTERNAL_API_KEY:-${LEADPOET_INTERNAL_SECRET:-}}"

is_truthy() {
    case "${1:-}" in
        true|TRUE|True|1|yes|YES|Yes|y|Y|on|ON|On) return 0 ;;
        *) return 1 ;;
    esac
}

ENABLE_LEGACY_SOURCING="${ENABLE_LEGACY_SOURCING:-false}"
ENABLE_SOURCING_WORKERS="${ENABLE_SOURCING_WORKERS:-$ENABLE_LEGACY_SOURCING}"
ENABLE_QUALIFICATION_WORKERS="${ENABLE_QUALIFICATION_WORKERS:-false}"
ENABLE_FULFILLMENT="${ENABLE_FULFILLMENT:-false}"

echo ""

# Auto-detect SOURCING proxies from .env
PROXIES=()
PROXY_COUNT=0

if is_truthy "$ENABLE_SOURCING_WORKERS" || is_truthy "$ENABLE_LEGACY_SOURCING"; then
    # Check for WEBSHARE_PROXY_1, WEBSHARE_PROXY_2, WEBSHARE_PROXY_3, etc.
    for i in {1..250}; do
        PROXY_VAR="WEBSHARE_PROXY_$i"
        PROXY_VALUE="${!PROXY_VAR}"

        if [ -n "$PROXY_VALUE" ] && [ "$PROXY_VALUE" != "http://YOUR_USERNAME:YOUR_PASSWORD@p.webshare.io:80" ]; then
            PROXIES+=("$PROXY_VALUE")
            PROXY_COUNT=$((PROXY_COUNT + 1))
        fi
    done
else
    echo "🚫 Legacy sourcing worker deployment disabled (set ENABLE_SOURCING_WORKERS=true to opt in)"
fi

# Auto-detect QUALIFICATION proxies from .env
QUAL_PROXIES=()
QUAL_PROXY_COUNT=0

if is_truthy "$ENABLE_QUALIFICATION_WORKERS"; then
    # Check for QUALIFICATION_WEBSHARE_PROXY_1, QUALIFICATION_WEBSHARE_PROXY_2, etc.
    for i in {1..10}; do
        PROXY_VAR="QUALIFICATION_WEBSHARE_PROXY_$i"
        PROXY_VALUE="${!PROXY_VAR}"

        if [ -n "$PROXY_VALUE" ] && [ "$PROXY_VALUE" != "http://YOUR_USERNAME:YOUR_PASSWORD@p.webshare.io:80" ]; then
            QUAL_PROXIES+=("$PROXY_VALUE")
            QUAL_PROXY_COUNT=$((QUAL_PROXY_COUNT + 1))
        fi
    done
else
    echo "🚫 Qualification worker deployment disabled (set ENABLE_QUALIFICATION_WORKERS=true to opt in)"
fi

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

echo "🔍 Auto-detected SOURCING proxies: $PROXY_COUNT"
echo "🔍 Auto-detected QUALIFICATION proxies: $QUAL_PROXY_COUNT"
echo "📦 Total SOURCING containers to deploy: $TOTAL_CONTAINERS"
echo "   - 1x Main validator (EC2 native IP)"
echo "   - ${PROXY_COUNT}x Worker containers (proxied)"
if [ $QUAL_PROXY_COUNT -gt 0 ]; then
    echo "📦 Total QUALIFICATION workers to spawn: $QUAL_PROXY_COUNT"
fi
echo ""

if [ $PROXY_COUNT -eq 0 ] && (is_truthy "$ENABLE_SOURCING_WORKERS" || is_truthy "$ENABLE_LEGACY_SOURCING"); then
    echo "⚠️  WARNING: No proxies configured in .env.docker"
    echo ""
    echo "For parallel processing with different IPs, add proxies:"
    echo "  WEBSHARE_PROXY_1=http://user:pass@p.webshare.io:80"
    echo "  WEBSHARE_PROXY_2=http://user:pass@p.webshare.io:80"
    echo "  WEBSHARE_PROXY_3=http://user:pass@p.webshare.io:80"
    echo ""
    echo "Deploying with 1 sourcing container (main validator only)..."
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

terminate_stale_validator_builds "before-build"

if docker_build_validator_image; then
    echo "✅ Docker image built successfully"
else
    BUILD_EXIT_CODE=$?
    terminate_stale_validator_builds "after-build-failure"
    echo "❌ ERROR: Docker build failed"
    if [ "$BUILD_EXIT_CODE" -eq 124 ]; then
        echo "   Docker build timed out after ${VALIDATOR_DOCKER_BUILD_TIMEOUT_SECONDS:-1800} seconds."
        echo "   Stale docker/pip build processes were terminated."
    fi
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
IMAGE_COMMIT="$(
    docker image inspect leadpoet-validator:latest \
        --format '{{ index .Config.Labels "org.opencontainers.image.revision" }}'
)"
if [ "$IMAGE_COMMIT" != "$VALIDATOR_V2_DEPLOY_COMMIT" ]; then
    echo "❌ ERROR: validator Docker image commit label differs from approved commit" >&2
    exit 1
fi
echo "✅ Validator Docker image commit verified: $IMAGE_COMMIT"
echo ""

# Stop and remove existing containers (sourcing + qualification + fulfillment)
echo "🛑 Stopping existing containers (if any)..."
docker ps -a --filter "name=leadpoet-validator" --format "{{.Names}}" | while read container; do
    docker rm -f "$container" 2>/dev/null || true
done
docker ps -a --filter "name=leadpoet-qual-worker" --format "{{.Names}}" | while read container; do
    docker rm -f "$container" 2>/dev/null || true
done
docker ps -a --filter "name=leadpoet-ff-worker" --format "{{.Names}}" | while read container; do
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
    local LOG_DRIVER_ARGS=""
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
        # CloudWatch Logs for coordinator container (ships logs directly to AWS, no local files)
        LOG_DRIVER_ARGS="--log-driver=awslogs --log-opt awslogs-region=us-east-1 --log-opt awslogs-group=/leadpoet/validator/coordinator --log-opt awslogs-stream=coordinator --log-opt awslogs-create-group=true"
        echo "   📊 CloudWatch Logs: /leadpoet/validator/coordinator"
    else
        MODE_ARG="--mode worker"
        # CloudWatch Logs for worker containers (each worker gets its own stream)
        LOG_DRIVER_ARGS="--log-driver=awslogs --log-opt awslogs-region=us-east-1 --log-opt awslogs-group=/leadpoet/validator/workers --log-opt awslogs-stream=worker-$CONTAINER_ID --log-opt awslogs-create-group=true"
        echo "   📊 CloudWatch Logs: /leadpoet/validator/workers (stream: worker-$CONTAINER_ID)"
    fi

    docker run -d \
      --name "$CONTAINER_NAME" \
      --network host \
      --restart unless-stopped \
      $PRIVILEGED_ARG \
      $LOG_DRIVER_ARGS \
      -v ~/.bittensor/wallets:/root/.bittensor/wallets:ro \
      -v "$REPO_ROOT/validator_weights:/app/validator_weights" \
      -e PYTHONUNBUFFERED=1 \
      -e LEADPOET_CONTAINER_MODE=1 \
      -e LEADPOET_WRAPPER_ACTIVE=1 \
      -e VALIDATOR_V2_DEPLOY_COMMIT="$VALIDATOR_V2_DEPLOY_COMMIT" \
      -e GITHUB_SHA="$VALIDATOR_V2_DEPLOY_COMMIT" \
      -e GIT_COMMIT="$VALIDATOR_V2_DEPLOY_COMMIT" \
      -e VALIDATOR_V2_GATEWAY_URL="${VALIDATOR_V2_GATEWAY_URL:-}" \
      -e EXPECTED_CHAIN="${EXPECTED_CHAIN:-}" \
      -e MEV_API_KEY="$MEV_API_KEY" \
      -e TRUELIST_API_KEY="$TRUELIST_API_KEY" \
      -e ZEROBOUNCE_API_KEY="${ZEROBOUNCE_API_KEY:-}" \
      -e SCRAPINGDOG_API_KEY="$SCRAPINGDOG_API_KEY" \
      -e OPENROUTER_KEY="$OPENROUTER_KEY" \
      -e APIFY_API_TOKEN="${APIFY_API_TOKEN:-}" \
      -e FULFILLMENT_USE_APIFY="${FULFILLMENT_USE_APIFY:-false}" \
      -e COMPANIES_HOUSE_API_KEY="$COMPANIES_HOUSE_API_KEY" \
      -e BITTENSOR_NETWORK="$VALIDATOR_SUBTENSOR_NETWORK" \
      -e SUBTENSOR_NETWORK="$VALIDATOR_SUBTENSOR_NETWORK" \
      -e BITTENSOR_NETUID="$VALIDATOR_NETUID" \
      -e NETUID="$VALIDATOR_NETUID" \
      -e GIT_COMMIT_HASH="$VALIDATOR_DEPLOY_SHA" \
      -e EXPECTED_CHAIN="$EXPECTED_CHAIN" \
      -e VALIDATOR_WEIGHT_PROTOCOL=authoritative_v2 \
      -e LEADPOET_EPOCH_MODE="${LEADPOET_EPOCH_MODE:-}" \
      -e ENABLE_QUALIFICATION_EVALUATION="${ENABLE_QUALIFICATION_EVALUATION:-false}" \
      -e ENABLE_QUALIFICATION_WORKERS="${ENABLE_QUALIFICATION_WORKERS:-false}" \
      -e ENABLE_FULFILLMENT="${ENABLE_FULFILLMENT:-false}" \
      -e FULFILLMENT_OPENROUTER_API_KEY="${FULFILLMENT_OPENROUTER_API_KEY:-}" \
      -e FULFILLMENT_WEBSHARE_PROXY_1="${FULFILLMENT_WEBSHARE_PROXY_1:-}" \
      -e FULFILLMENT_WEBSHARE_PROXY_2="${FULFILLMENT_WEBSHARE_PROXY_2:-}" \
      -e FULFILLMENT_WEBSHARE_PROXY_3="${FULFILLMENT_WEBSHARE_PROXY_3:-}" \
      -e FULFILLMENT_WEBSHARE_PROXY_4="${FULFILLMENT_WEBSHARE_PROXY_4:-}" \
      -e FULFILLMENT_WEBSHARE_PROXY_5="${FULFILLMENT_WEBSHARE_PROXY_5:-}" \
      -e FULFILLMENT_WEBSHARE_PROXY_6="${FULFILLMENT_WEBSHARE_PROXY_6:-}" \
      -e FULFILLMENT_WEBSHARE_PROXY_7="${FULFILLMENT_WEBSHARE_PROXY_7:-}" \
      -e FULFILLMENT_WEBSHARE_PROXY_8="${FULFILLMENT_WEBSHARE_PROXY_8:-}" \
      -e FULFILLMENT_WEBSHARE_PROXY_9="${FULFILLMENT_WEBSHARE_PROXY_9:-}" \
      -e FULFILLMENT_WEBSHARE_PROXY_10="${FULFILLMENT_WEBSHARE_PROXY_10:-}" \
      -e GATEWAY_URL="${GATEWAY_URL:-https://gateway.subnet71.com}" \
      -e VALIDATOR_V2_GATEWAY_URL="${VALIDATOR_V2_GATEWAY_URL:-}" \
      -e LEADPOET_INTERNAL_SECRET="${LEADPOET_INTERNAL_SECRET:-}" \
      -e RESEARCH_LAB_INTERNAL_API_KEY="${RESEARCH_LAB_INTERNAL_API_KEY:-}" \
      -e RESEARCH_LAB_VALIDATOR_FETCH_ENABLED="${RESEARCH_LAB_VALIDATOR_FETCH_ENABLED:-true}" \
      -e RESEARCH_LAB_VALIDATOR_SHADOW_VERIFY_ENABLED="${RESEARCH_LAB_VALIDATOR_SHADOW_VERIFY_ENABLED:-true}" \
      -e RESEARCH_LAB_VALIDATOR_EVALUATION_VERIFY_ENABLED="${RESEARCH_LAB_VALIDATOR_EVALUATION_VERIFY_ENABLED:-true}" \
      -e RESEARCH_LAB_REQUIRE_SHADOW_VERIFICATION_BEFORE_SUBMIT="${RESEARCH_LAB_REQUIRE_SHADOW_VERIFICATION_BEFORE_SUBMIT:-true}" \
      -e RESEARCH_LAB_REQUIRE_EVALUATION_VERIFICATION_BEFORE_SUBMIT="${RESEARCH_LAB_REQUIRE_EVALUATION_VERIFICATION_BEFORE_SUBMIT:-true}" \
      -e RESEARCH_LAB_REIMBURSEMENTS_ENABLED="${RESEARCH_LAB_REIMBURSEMENTS_ENABLED:-true}" \
      -e RESEARCH_LAB_WEIGHT_MUTATION_ENABLED="${RESEARCH_LAB_WEIGHT_MUTATION_ENABLED:-true}" \
      -e RESEARCH_LAB_SUBMIT_ON_CHAIN_ENABLED="${RESEARCH_LAB_SUBMIT_ON_CHAIN_ENABLED:-true}" \
      -e QUALIFICATION_WEBSHARE_PROXY_1="${QUALIFICATION_WEBSHARE_PROXY_1:-}" \
      -e QUALIFICATION_WEBSHARE_PROXY_2="${QUALIFICATION_WEBSHARE_PROXY_2:-}" \
      -e QUALIFICATION_WEBSHARE_PROXY_3="${QUALIFICATION_WEBSHARE_PROXY_3:-}" \
      -e QUALIFICATION_WEBSHARE_PROXY_4="${QUALIFICATION_WEBSHARE_PROXY_4:-}" \
      -e QUALIFICATION_WEBSHARE_PROXY_5="${QUALIFICATION_WEBSHARE_PROXY_5:-}" \
      -e QUALIFICATION_SCRAPINGDOG_API_KEY="${QUALIFICATION_SCRAPINGDOG_API_KEY:-}" \
      -e QUALIFICATION_OPENROUTER_API_KEY="${QUALIFICATION_OPENROUTER_API_KEY:-}" \
      -e INTENT_GATE_STRICT_JUDGE_ENABLED="${INTENT_GATE_STRICT_JUDGE_ENABLED:-true}" \
      -e INTENT_VERIFIER_THREE_STAGE="${INTENT_VERIFIER_THREE_STAGE:-}" \
      -e INTENT_THREE_STAGE_S1_MODEL="${INTENT_THREE_STAGE_S1_MODEL:-}" \
      -e INTENT_THREE_STAGE_S3_MODEL="${INTENT_THREE_STAGE_S3_MODEL:-}" \
      -e INTENT_VERIFIER_REVIEW_AS_ACCEPT="${INTENT_VERIFIER_REVIEW_AS_ACCEPT:-}" \
      -e INTENT_PRECHECK_ENABLED="${INTENT_PRECHECK_ENABLED:-false}" \
      -e INTENT_PRECHECK_MODEL="${INTENT_PRECHECK_MODEL:-}" \
      -e INTENT_PRECHECK_RETRIES="${INTENT_PRECHECK_RETRIES:-}" \
      -e INTENT_PRECHECK_TIMEOUT_S="${INTENT_PRECHECK_TIMEOUT_S:-}" \
      -e INTENT_PRECHECK_CONCURRENCY="${INTENT_PRECHECK_CONCURRENCY:-}" \
      -e INTENT_URL_PREFILTER_ENABLED="${INTENT_URL_PREFILTER_ENABLED:-false}" \
      -e EXA_API_KEY="${EXA_API_KEY:-}" \
      -e DESEARCH_API_KEY="${DESEARCH_API_KEY:-}" \
      -e BUILTWITH_API_KEY="${BUILTWITH_API_KEY:-}" \
      -e QUALIFICATION_LEADS_TABLE="${QUALIFICATION_LEADS_TABLE:-test_leads_for_miners}" \
      $ENCLAVE_CID_ARG \
      $VSOCK_ARG \
      $PROXY_ARGS \
      leadpoet-validator:latest \
      --netuid "$VALIDATOR_NETUID" \
      --subtensor_network "$VALIDATOR_SUBTENSOR_NETWORK" \
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

# ════════════════════════════════════════════════════════════════════════════════
# SPAWN QUALIFICATION WORKERS (Docker containers with --restart unless-stopped)
# ════════════════════════════════════════════════════════════════════════════════
# Qualification workers evaluate miner models in parallel to sourcing.
# They use their own proxies (QUALIFICATION_WEBSHARE_PROXY_*) for ddgs/free APIs.
# Runs as Docker containers (same image as sourcing) for automatic restart on crash.
# ════════════════════════════════════════════════════════════════════════════════

if [ $QUAL_PROXY_COUNT -gt 0 ]; then
    echo ""
    echo "============================================================"
    echo "🎯 DEPLOYING QUALIFICATION WORKER CONTAINERS"
    echo "============================================================"
    echo ""

    # Stop and remove any existing qualification containers + bare processes
    pkill -9 -f "qualification_worker" 2>/dev/null || true
    for i in $(seq 1 $QUAL_PROXY_COUNT); do
        docker rm -f "leadpoet-qual-worker-$i" 2>/dev/null || true
    done
    sleep 1

    for i in $(seq 1 $QUAL_PROXY_COUNT); do
        QUAL_PROXY_VAR="QUALIFICATION_WEBSHARE_PROXY_$i"
        QUAL_PROXY_VALUE="${!QUAL_PROXY_VAR}"

        echo "🚀 Starting Qualification Worker $i (Docker container)..."
        if [ -n "$QUAL_PROXY_VALUE" ]; then
            echo "   Proxy: ${QUAL_PROXY_VALUE:0:30}..."
        fi

        QUAL_PROXY_ARGS=""
        if [ -n "$QUAL_PROXY_VALUE" ]; then
            QUAL_PROXY_ARGS="-e HTTP_PROXY=$QUAL_PROXY_VALUE -e HTTPS_PROXY=$QUAL_PROXY_VALUE"
        fi

        docker run -d \
          --name "leadpoet-qual-worker-$i" \
          --network host \
          --restart unless-stopped \
          --log-driver=awslogs \
          --log-opt awslogs-region=us-east-1 \
          --log-opt awslogs-group=/leadpoet/validator/qualification \
          --log-opt awslogs-stream=qual-worker-$i \
          --log-opt awslogs-create-group=true \
          -v "$REPO_ROOT/validator_weights:/app/validator_weights" \
          -e PYTHONUNBUFFERED=1 \
          -e LEADPOET_CONTAINER_MODE=1 \
          -e LEADPOET_WRAPPER_ACTIVE=1 \
          -e GATEWAY_URL="${GATEWAY_URL:-https://gateway.subnet71.com}" \
      -e LEADPOET_INTERNAL_SECRET="${LEADPOET_INTERNAL_SECRET:-}" \
          -e RESEARCH_LAB_INTERNAL_API_KEY="${RESEARCH_LAB_INTERNAL_API_KEY:-}" \
          -e QUALIFICATION_WEBSHARE_PROXY_1="${QUALIFICATION_WEBSHARE_PROXY_1:-}" \
          -e QUALIFICATION_WEBSHARE_PROXY_2="${QUALIFICATION_WEBSHARE_PROXY_2:-}" \
          -e QUALIFICATION_WEBSHARE_PROXY_3="${QUALIFICATION_WEBSHARE_PROXY_3:-}" \
          -e QUALIFICATION_WEBSHARE_PROXY_4="${QUALIFICATION_WEBSHARE_PROXY_4:-}" \
          -e QUALIFICATION_WEBSHARE_PROXY_5="${QUALIFICATION_WEBSHARE_PROXY_5:-}" \
          -e QUALIFICATION_SCRAPINGDOG_API_KEY="${QUALIFICATION_SCRAPINGDOG_API_KEY:-}" \
          -e QUALIFICATION_OPENROUTER_API_KEY="${QUALIFICATION_OPENROUTER_API_KEY:-}" \
          -e EXA_API_KEY="${EXA_API_KEY:-}" \
          -e INTENT_GATE_STRICT_JUDGE_ENABLED="${INTENT_GATE_STRICT_JUDGE_ENABLED:-true}" \
          -e DESEARCH_API_KEY="${DESEARCH_API_KEY:-}" \
          -e BUILTWITH_API_KEY="${BUILTWITH_API_KEY:-}" \
          -e QUALIFICATION_LEADS_TABLE="${QUALIFICATION_LEADS_TABLE:-test_leads_for_miners}" \
          $QUAL_PROXY_ARGS \
          leadpoet-validator:latest \
          --mode qualification_worker \
          --container-id "$i" > /dev/null

        echo "   ✅ Started: leadpoet-qual-worker-$i"
        echo ""
    done

    echo "✅ All $QUAL_PROXY_COUNT qualification worker containers deployed"
    echo "   (--restart unless-stopped: auto-recovers from crashes)"
    echo ""
fi

# ════════════════════════════════════════════════════════════════════════════════
# SPAWN FULFILLMENT WORKERS (Docker containers with --restart unless-stopped)
# ════════════════════════════════════════════════════════════════════════════════
# Fulfillment workers score revealed leads through Tier 1-3 pipeline.
# They use their own proxies (FULFILLMENT_WEBSHARE_PROXY_*) for ScrapingDog/LLM.
# ════════════════════════════════════════════════════════════════════════════════

# Auto-detect FULFILLMENT proxies from .env
FF_PROXIES=()
FF_PROXY_COUNT=0

if is_truthy "$ENABLE_FULFILLMENT"; then
    for i in {1..10}; do
        PROXY_VAR="FULFILLMENT_WEBSHARE_PROXY_$i"
        PROXY_VALUE="${!PROXY_VAR}"

        if [ -n "$PROXY_VALUE" ]; then
            FF_PROXIES+=("$PROXY_VALUE")
            FF_PROXY_COUNT=$((FF_PROXY_COUNT + 1))
        else
            break
        fi
    done
else
    echo "🚫 Fulfillment worker deployment disabled (ENABLE_FULFILLMENT != true)"
fi

echo "🔍 Auto-detected FULFILLMENT proxies: $FF_PROXY_COUNT"

if [ $FF_PROXY_COUNT -gt 0 ]; then
    echo ""
    echo "============================================================"
    echo "🎯 DEPLOYING FULFILLMENT WORKER CONTAINERS"
    echo "============================================================"
    echo ""

    for i in $(seq 1 $FF_PROXY_COUNT); do
        docker rm -f "leadpoet-ff-worker-$i" 2>/dev/null || true
    done
    sleep 1

    for i in $(seq 1 $FF_PROXY_COUNT); do
        FF_PROXY_VAR="FULFILLMENT_WEBSHARE_PROXY_$i"
        FF_PROXY_VALUE="${!FF_PROXY_VAR}"

        echo "🚀 Starting Fulfillment Worker $i (Docker container)..."
        if [ -n "$FF_PROXY_VALUE" ]; then
            echo "   Proxy: ${FF_PROXY_VALUE:0:30}..."
        fi

        FF_PROXY_ARGS=""
        if [ -n "$FF_PROXY_VALUE" ]; then
            FF_PROXY_ARGS="-e HTTP_PROXY=$FF_PROXY_VALUE -e HTTPS_PROXY=$FF_PROXY_VALUE"
        fi

        docker run -d \
          --name "leadpoet-ff-worker-$i" \
          --network host \
          --restart unless-stopped \
          --log-driver=awslogs \
          --log-opt awslogs-region=us-east-1 \
          --log-opt awslogs-group=/leadpoet/validator/fulfillment \
          --log-opt awslogs-stream=ff-worker-$i \
          --log-opt awslogs-create-group=true \
          -v "$REPO_ROOT/validator_weights:/app/validator_weights" \
          -e PYTHONUNBUFFERED=1 \
          -e LEADPOET_CONTAINER_MODE=1 \
          -e LEADPOET_WRAPPER_ACTIVE=1 \
          -e ENABLE_FULFILLMENT=true \
          -e GATEWAY_URL="${GATEWAY_URL:-https://gateway.subnet71.com}" \
      -e LEADPOET_INTERNAL_SECRET="${LEADPOET_INTERNAL_SECRET:-}" \
          -e RESEARCH_LAB_INTERNAL_API_KEY="${RESEARCH_LAB_INTERNAL_API_KEY:-}" \
          -e FULFILLMENT_WEBSHARE_PROXY_1="${FULFILLMENT_WEBSHARE_PROXY_1:-}" \
          -e FULFILLMENT_WEBSHARE_PROXY_2="${FULFILLMENT_WEBSHARE_PROXY_2:-}" \
          -e FULFILLMENT_WEBSHARE_PROXY_3="${FULFILLMENT_WEBSHARE_PROXY_3:-}" \
          -e FULFILLMENT_WEBSHARE_PROXY_4="${FULFILLMENT_WEBSHARE_PROXY_4:-}" \
          -e FULFILLMENT_WEBSHARE_PROXY_5="${FULFILLMENT_WEBSHARE_PROXY_5:-}" \
          -e FULFILLMENT_WEBSHARE_PROXY_6="${FULFILLMENT_WEBSHARE_PROXY_6:-}" \
          -e FULFILLMENT_WEBSHARE_PROXY_7="${FULFILLMENT_WEBSHARE_PROXY_7:-}" \
          -e FULFILLMENT_WEBSHARE_PROXY_8="${FULFILLMENT_WEBSHARE_PROXY_8:-}" \
          -e FULFILLMENT_WEBSHARE_PROXY_9="${FULFILLMENT_WEBSHARE_PROXY_9:-}" \
          -e FULFILLMENT_WEBSHARE_PROXY_10="${FULFILLMENT_WEBSHARE_PROXY_10:-}" \
          -e FULFILLMENT_OPENROUTER_API_KEY="${FULFILLMENT_OPENROUTER_API_KEY:-}" \
          -e QUALIFICATION_SCRAPINGDOG_API_KEY="${QUALIFICATION_SCRAPINGDOG_API_KEY:-}" \
          -e QUALIFICATION_OPENROUTER_API_KEY="${QUALIFICATION_OPENROUTER_API_KEY:-}" \
          -e INTENT_GATE_STRICT_JUDGE_ENABLED="${INTENT_GATE_STRICT_JUDGE_ENABLED:-true}" \
          -e SCRAPINGDOG_API_KEY="${SCRAPINGDOG_API_KEY:-}" \
          -e OPENROUTER_KEY="${OPENROUTER_KEY:-}" \
          -e APIFY_API_TOKEN="${APIFY_API_TOKEN:-}" \
          -e FULFILLMENT_USE_APIFY="${FULFILLMENT_USE_APIFY:-false}" \
          -e TRUELIST_API_KEY="${TRUELIST_API_KEY:-}" \
          -e ZEROBOUNCE_API_KEY="${ZEROBOUNCE_API_KEY:-}" \
          -e FULFILLMENT_WEBSITE_TIMEOUT_S="${FULFILLMENT_WEBSITE_TIMEOUT_S:-30}" \
          -e INTENT_VERIFIER_THREE_STAGE="${INTENT_VERIFIER_THREE_STAGE:-}" \
          -e INTENT_THREE_STAGE_S1_MODEL="${INTENT_THREE_STAGE_S1_MODEL:-}" \
          -e INTENT_THREE_STAGE_S3_MODEL="${INTENT_THREE_STAGE_S3_MODEL:-}" \
          -e INTENT_VERIFIER_REVIEW_AS_ACCEPT="${INTENT_VERIFIER_REVIEW_AS_ACCEPT:-}" \
          -e INTENT_PRECHECK_ENABLED="${INTENT_PRECHECK_ENABLED:-false}" \
          -e INTENT_PRECHECK_MODEL="${INTENT_PRECHECK_MODEL:-}" \
          -e INTENT_PRECHECK_RETRIES="${INTENT_PRECHECK_RETRIES:-}" \
          -e INTENT_PRECHECK_TIMEOUT_S="${INTENT_PRECHECK_TIMEOUT_S:-}" \
          -e INTENT_PRECHECK_CONCURRENCY="${INTENT_PRECHECK_CONCURRENCY:-}" \
          -e INTENT_URL_PREFILTER_ENABLED="${INTENT_URL_PREFILTER_ENABLED:-false}" \
          -e EXA_API_KEY="${EXA_API_KEY:-}" \
          $FF_PROXY_ARGS \
          leadpoet-validator:latest \
          --mode fulfillment_worker \
          --container-id "$i" > /dev/null

        echo "   ✅ Started: leadpoet-ff-worker-$i"
        echo ""
    done

    echo "✅ All $FF_PROXY_COUNT fulfillment worker containers deployed"
    echo ""
fi

# Wait for containers to start
echo "⏳ Waiting 10 seconds for containers to initialize..."
sleep 10

# Check status
echo ""
echo "============================================================"
echo "📊 CONTAINER STATUS"
echo "============================================================"
docker ps --filter "name=leadpoet-validator" --filter "name=leadpoet-qual-worker" --filter "name=leadpoet-ff-worker" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
echo ""

# Verify proxies
echo "============================================================"
echo "🌐 VERIFYING PROXY IPS"
echo "============================================================"
echo ""
echo "⏳ Waiting 30 seconds for validators to fully initialize..."
sleep 30

echo "🔐 Verifying authoritative validator coordinator runtime..."
if [ "$(docker inspect -f '{{.State.Running}}' leadpoet-validator-main)" != "true" ]; then
    echo "❌ ERROR: validator coordinator is not running" >&2
    docker logs --tail 160 leadpoet-validator-main >&2 || true
    exit 1
fi
if [ "$(docker inspect -f '{{.RestartCount}}' leadpoet-validator-main)" != "0" ]; then
    echo "❌ ERROR: validator coordinator restarted during startup" >&2
    docker logs --tail 160 leadpoet-validator-main >&2 || true
    exit 1
fi
MAIN_RUNTIME_ENV="$(
    docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' \
        leadpoet-validator-main
)"
if [ "$(
    printf '%s\n' "$MAIN_RUNTIME_ENV" \
        | sed -n 's/^VALIDATOR_V2_DEPLOY_COMMIT=//p'
)" != "$VALIDATOR_V2_DEPLOY_COMMIT" ]; then
    echo "❌ ERROR: validator coordinator commit differs from approved release" >&2
    exit 1
fi
if [ "$(
    printf '%s\n' "$MAIN_RUNTIME_ENV" \
        | sed -n 's/^VALIDATOR_WEIGHT_PROTOCOL=//p'
)" != "authoritative_v2" ]; then
    echo "❌ ERROR: validator coordinator is not using authoritative V2" >&2
    exit 1
fi
if [ "$(
    printf '%s\n' "$MAIN_RUNTIME_ENV" \
        | sed -n 's/^LEADPOET_EPOCH_MODE=//p'
)" != "${LEADPOET_EPOCH_MODE:-}" ]; then
    echo "❌ ERROR: validator coordinator epoch mode differs from restart selection" >&2
    exit 1
fi
if ! docker exec leadpoet-validator-main sh -c \
    "tr '\\000' ' ' </proc/1/cmdline | grep -q 'neurons/validator.py'"; then
    echo "❌ ERROR: validator coordinator process is not validator.py" >&2
    exit 1
fi
docker exec -i leadpoet-validator-main python3 - <<'PY'
import json
import os
import urllib.request

import bittensor as bt

from Leadpoet.utils.subnet_epoch import read_subnet_epoch_snapshot
from validator_tee.host.vsock_client import ValidatorEnclaveClient

if str(bt.__version__) != "10.5.0":
    raise SystemExit(f"validator Bittensor SDK mismatch: {bt.__version__}")

client = ValidatorEnclaveClient()
health = client.health_check()
if health.get("status") != "ok":
    raise SystemExit(f"validator enclave health failed: {health}")
hotkey_state = client.get_hotkey_state_v2()
if hotkey_state.get("provisioned") is not True:
    raise SystemExit("validator hotkey is not provisioned in Nitro")

gateway_url = str(os.environ.get("VALIDATOR_V2_GATEWAY_URL") or "").rstrip("/")
gateway_health = {}
gateway_authority_status = "deferred"
gateway_authority_error_type = ""
if gateway_url:
    try:
        with urllib.request.urlopen(
            gateway_url + "/health/v2-authority",
            timeout=30,
        ) as response:
            gateway_health = json.load(response)
    except Exception as exc:
        gateway_authority_error_type = type(exc).__name__
    else:
        live_commit = str(gateway_health.get("commit_sha") or "")
        expected_commit = str(os.environ.get("VALIDATOR_V2_DEPLOY_COMMIT") or "")
        if (
            gateway_health.get("status") == "ready"
            and live_commit
            and live_commit == expected_commit
        ):
            gateway_authority_status = "ready"
        else:
            gateway_authority_status = "not_aligned"
else:
    gateway_authority_error_type = "gateway_url_not_configured"

network = os.environ.get("SUBTENSOR_NETWORK", "finney")
netuid = int(os.environ.get("NETUID", "71"))
subtensor = bt.Subtensor(network=network)
try:
    epoch = read_subnet_epoch_snapshot(subtensor, netuid=netuid)
finally:
    close = getattr(subtensor, "close", None)
    if callable(close):
        close()

print(
    json.dumps(
        {
            "bittensor_version": str(bt.__version__),
            "enclave_status": health.get("status"),
            "hotkey_provisioned": True,
            "gateway_authority_status": gateway_authority_status,
            "gateway_authority_error_type": gateway_authority_error_type,
            "gateway_commit": gateway_health.get("commit_sha"),
            "subnet_epoch_index": epoch.subnet_epoch_index,
            "subnet_epoch_block": epoch.epoch_block,
        },
        sort_keys=True,
    )
)
PY
echo "✅ Authoritative validator coordinator runtime verified"
echo ""

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
echo "   - Sourcing containers: $TOTAL_CONTAINERS (1 coordinator + $PROXY_COUNT workers)"
if [ $QUAL_PROXY_COUNT -gt 0 ]; then
    echo "   - Qualification containers: $QUAL_PROXY_COUNT (Docker, auto-restart)"
fi
echo "   - Lead distribution: FULLY DYNAMIC (adapts to gateway MAX_LEADS_PER_EPOCH)"
echo "   - Unique IPs: $UNIQUE_COUNT / $TOTAL_COUNT"
echo ""
echo "   Examples of auto-scaling:"
echo "   - Gateway @ 170 leads → Each container: ~57 leads"
echo "   - Gateway @ 900 leads → Each container: 300 leads"
echo "   - Gateway @ 1200 leads → Each container: 400 leads"
echo ""
echo "📋 Next Steps:"
echo "   1. Monitor sourcing logs: docker logs -f leadpoet-validator-main"
if [ $QUAL_PROXY_COUNT -gt 0 ]; then
    echo "   2. Monitor qualification logs: docker logs -f leadpoet-qual-worker-1"
fi
echo "   3. Check resource usage: docker stats"
echo "   4. Verify lead distribution in logs (each container shows its range)"
echo ""
echo "🔧 To scale up (add more containers):"
echo "   1. Get another proxy from https://www.webshare.io/"
echo "   2. Add WEBSHARE_PROXY_$((PROXY_COUNT + 1))=... to .env.docker"
echo "   3. Run: ./deploy_dynamic.sh"
echo "   Done! New container auto-joins and gets its share of leads."
echo ""
