#!/bin/bash
# Start the approved four-role Nitro topology, or one role for component tests.

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
  legacy_v1|legacy_v1_compat) TEE_PROTOCOL="legacy_v1" ;;
  v2|authoritative_v2) TEE_PROTOCOL="v2" ;;
  *)
    echo "ERROR: RESEARCH_LAB_TEE_PROTOCOL must be legacy_v1 or v2" >&2
    exit 1
    ;;
esac
ALL_ROLES=(
  gateway_coordinator
  gateway_scoring_a
  gateway_scoring_b
  gateway_autoresearch
)

if [ "$TEE_PROTOCOL" = "legacy_v1" ]; then
  EIF_PATH="$EIF_ROOT/tee-enclave.eif"
  test -s "$EIF_PATH" || {
    echo "ERROR: legacy gateway EIF is unavailable: $EIF_PATH" >&2
    exit 1
  }
  CPU_COUNT="${GATEWAY_ENCLAVE_CPU_COUNT:-2}"
  MEMORY_MIB="${GATEWAY_ENCLAVE_MEMORY_MB:-8192}"
  if ! [[ "$CPU_COUNT" =~ ^[0-9]+$ ]] || ! [[ "$MEMORY_MIB" =~ ^[0-9]+$ ]]; then
    echo "ERROR: legacy gateway enclave CPU and memory must be integers" >&2
    exit 1
  fi
  if [ "$CPU_COUNT" -lt 2 ] || [ "$MEMORY_MIB" -lt 8192 ]; then
    echo "ERROR: legacy gateway enclave requires at least 2 vCPUs and 8192 MiB" >&2
    exit 1
  fi
  sudo nitro-cli terminate-enclave --all 2>/dev/null || true
  sleep 2
  echo "Starting legacy gateway enclave from the current deploy commit"
  sudo nitro-cli run-enclave \
    --cpu-count "$CPU_COUNT" \
    --memory "$MEMORY_MIB" \
    --eif-path "$EIF_PATH" \
    --enclave-cid 16
  sleep 15
  sudo nitro-cli describe-enclaves
  PYTHONPATH="${GATEWAY_ROOT%/gateway}" python3 - <<'PY'
import asyncio

from gateway.utils.tee_client import TEEClient


async def main() -> None:
    last_error = None
    for _attempt in range(10):
        try:
            public_key = await TEEClient(cid=16)._send_rpc("get_public_key", {})
            if not isinstance(public_key, str) or len(public_key) != 64:
                raise RuntimeError("legacy gateway enclave returned an invalid public key")
            return
        except Exception as exc:
            last_error = exc
            await asyncio.sleep(2)
    raise RuntimeError("legacy gateway enclave RPC is unavailable") from last_error


asyncio.run(main())
PY
  echo "Legacy gateway enclave RPC is healthy"
  exit 0
fi

python3 "$SCRIPT_DIR/topology.py" --verify "$SCRIPT_DIR/topology.json"

if [ "$TOPOLOGY_MODE" = "full" ]; then
  TOTAL_CPUS="$(getconf _NPROCESSORS_CONF)"
  TOTAL_MEMORY_MIB="$(awk '/^MemTotal:/ {print int($2 / 1024)}' /proc/meminfo)"
  if [ "$TOTAL_CPUS" -lt 32 ] || [ "$TOTAL_MEMORY_MIB" -lt 250000 ]; then
    echo "ERROR: full V2 topology requires an r7i.8xlarge-class parent" >&2
    echo "Observed CPUs=${TOTAL_CPUS} memory_mib=${TOTAL_MEMORY_MIB}" >&2
    exit 1
  fi
  ROLES=("${ALL_ROLES[@]}")
elif [ "$TOPOLOGY_MODE" = "component" ]; then
  COMPONENT_ROLE="${GATEWAY_TEE_COMPONENT_ROLE:-gateway_coordinator}"
  case "$COMPONENT_ROLE" in
    gateway_coordinator|gateway_scoring_a|gateway_scoring_b|gateway_autoresearch) ;;
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
  for role in gateway_scoring_a gateway_scoring_b gateway_autoresearch; do
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
