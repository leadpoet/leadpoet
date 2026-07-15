#!/bin/bash
# Start the approved three-role Nitro topology, or one role for component tests.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GATEWAY_ROOT="${GATEWAY_ROOT:-/home/ec2-user/gateway}"
EIF_ROOT="${GATEWAY_TEE_EIF_ROOT:-/home/ec2-user/tee}"
TOPOLOGY_MODE="${GATEWAY_TEE_TOPOLOGY_MODE:-full}"
RELEASE_MANIFEST="${GATEWAY_V2_RELEASE_MANIFEST:-$EIF_ROOT/gateway-v2-release-manifest.json}"
TEE_PROTOCOL="$(
  printf '%s' "${RESEARCH_LAB_TEE_PROTOCOL:-v2}" \
    | tr '[:upper:]' '[:lower:]'
)"
case "$TEE_PROTOCOL" in
  v2|authoritative_v2) TEE_PROTOCOL="v2" ;;
  *)
    echo "ERROR: RESEARCH_LAB_TEE_PROTOCOL must be v2; V1 authority is retired" >&2
    exit 1
    ;;
esac
ALL_ROLES=(
  gateway_coordinator
  gateway_scoring
  gateway_autoresearch
)

python3 "$SCRIPT_DIR/topology.py" --verify "$SCRIPT_DIR/topology.json"

if [ "$TOPOLOGY_MODE" = "full" ]; then
  TOTAL_CPUS="$(getconf _NPROCESSORS_CONF)"
  TOTAL_MEMORY_MIB="$(awk '/^MemTotal:/ {print int($2 / 1024)}' /proc/meminfo)"
  if [ "$TOTAL_CPUS" -lt 16 ] || [ "$TOTAL_MEMORY_MIB" -lt 125000 ]; then
    echo "ERROR: full V2 topology requires an r7i.4xlarge parent" >&2
    echo "Observed CPUs=${TOTAL_CPUS} memory_mib=${TOTAL_MEMORY_MIB}" >&2
    exit 1
  fi
  ROLES=("${ALL_ROLES[@]}")
elif [ "$TOPOLOGY_MODE" = "component" ]; then
  COMPONENT_ROLE="${GATEWAY_TEE_COMPONENT_ROLE:-gateway_coordinator}"
  case "$COMPONENT_ROLE" in
    gateway_coordinator|gateway_scoring|gateway_autoresearch) ;;
    *) echo "ERROR: invalid GATEWAY_TEE_COMPONENT_ROLE" >&2; exit 1 ;;
  esac
  ROLES=("$COMPONENT_ROLE")
else
  echo "ERROR: GATEWAY_TEE_TOPOLOGY_MODE must be full or component" >&2
  exit 1
fi

for role in "${ROLES[@]}"; do
  test -s "$EIF_ROOT/tee-enclave-${role}.eif" || {
    echo "ERROR: missing role EIF $EIF_ROOT/tee-enclave-${role}.eif" >&2
    exit 1
  }
done

sudo nitro-cli terminate-enclave --all 2>/dev/null || true
sleep 2

start_role() {
  local role="$1"
  read -r cid vcpus memory_mib < <(
    python3 - "$SCRIPT_DIR/topology.json" "$role" <<'PY'
import json
import sys
spec = json.load(open(sys.argv[1]))["roles"][sys.argv[2]]
print(spec["cid"], spec["vcpus"], spec["memory_mib"])
PY
  )
  echo "Starting ${role} CID=${cid} vcpus=${vcpus} memory_mib=${memory_mib}"
  sudo nitro-cli run-enclave \
    --cpu-count "$vcpus" \
    --memory "$memory_mib" \
    --eif-path "$EIF_ROOT/tee-enclave-${role}.eif" \
    --enclave-cid "$cid"
}

verify_roles() {
  local verify_args=()
  if [ "$TOPOLOGY_MODE" = "full" ] || [ -s "$RELEASE_MANIFEST" ]; then
    test -s "$RELEASE_MANIFEST" || {
      echo "ERROR: approved gateway V2 release manifest is unavailable" >&2
      exit 1
    }
    verify_args+=(--release-manifest "$RELEASE_MANIFEST")
  fi
  PYTHONPATH="${GATEWAY_ROOT%/gateway}" \
    python3 "$SCRIPT_DIR/verify_topology.py" "${verify_args[@]}" "$@"
}

if [ "$TOPOLOGY_MODE" = "full" ]; then
  start_role gateway_coordinator
  sleep 15
  verify_roles gateway_coordinator
  for role in gateway_scoring gateway_autoresearch; do
    start_role "$role"
  done
  sleep 15
  verify_roles "${ROLES[@]}"
else
  start_role "${ROLES[0]}"
  sleep 15
  verify_roles "${ROLES[@]}"
fi

sudo nitro-cli describe-enclaves

echo "Gateway enclave topology is healthy"
