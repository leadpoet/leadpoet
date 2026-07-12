#!/bin/bash
# Configure Nitro's parent allocator for the measured four-enclave topology.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOPOLOGY_MODE="${GATEWAY_TEE_TOPOLOGY_MODE:-full}"
ALLOCATOR_CONFIG="${NITRO_ENCLAVES_ALLOCATOR_CONFIG:-/etc/nitro_enclaves/allocator.yaml}"
ALLOCATOR_SERVICE="${NITRO_ENCLAVES_ALLOCATOR_SERVICE:-nitro-enclaves-allocator.service}"

python3 "$SCRIPT_DIR/topology.py" --verify "$SCRIPT_DIR/topology.json"

if [ "$TOPOLOGY_MODE" = "component" ]; then
  echo "Component topology selected; retaining the operator's existing Nitro allocator"
  exit 0
fi
if [ "$TOPOLOGY_MODE" != "full" ]; then
  echo "ERROR: GATEWAY_TEE_TOPOLOGY_MODE must be full or component" >&2
  exit 1
fi

read -r REQUIRED_CPUS REQUIRED_MEMORY_MIB < <(
  python3 - "$SCRIPT_DIR/topology.json" <<'PY'
import json
import sys

document = json.load(open(sys.argv[1], encoding="utf-8"))
roles = document.get("roles")
if not isinstance(roles, dict) or len(roles) != 4:
    raise SystemExit("full V2 topology must define exactly four roles")
print(
    sum(int(spec["vcpus"]) for spec in roles.values()),
    sum(int(spec["memory_mib"]) for spec in roles.values()),
)
PY
)

TOTAL_CPUS="$(getconf _NPROCESSORS_CONF)"
TOTAL_MEMORY_MIB="$(awk '/^MemTotal:/ {print int($2 / 1024)}' /proc/meminfo)"
if [ "$TOTAL_CPUS" -lt 32 ] || [ "$TOTAL_MEMORY_MIB" -lt 250000 ]; then
  echo "ERROR: full V2 allocator requires an r7i.8xlarge-class parent" >&2
  echo "Observed CPUs=${TOTAL_CPUS} memory_mib=${TOTAL_MEMORY_MIB}" >&2
  exit 1
fi
if [ "$REQUIRED_CPUS" -ne 26 ] || [ "$REQUIRED_MEMORY_MIB" -ne 131072 ]; then
  echo "ERROR: measured V2 allocator totals changed without restart review" >&2
  echo "Observed enclave CPUs=${REQUIRED_CPUS} memory_mib=${REQUIRED_MEMORY_MIB}" >&2
  exit 1
fi
if [ $((TOTAL_CPUS - REQUIRED_CPUS)) -lt 6 ]; then
  echo "ERROR: V2 allocator would leave fewer than six host vCPUs" >&2
  exit 1
fi
if [ $((TOTAL_MEMORY_MIB - REQUIRED_MEMORY_MIB)) -lt 120000 ]; then
  echo "ERROR: V2 allocator would leave insufficient gateway parent memory" >&2
  exit 1
fi

TEMP_CONFIG="$(mktemp)"
trap 'rm -f "$TEMP_CONFIG"' EXIT
cat > "$TEMP_CONFIG" <<EOF
---
memory_mib: ${REQUIRED_MEMORY_MIB}
cpu_count: ${REQUIRED_CPUS}
EOF

if [ -f "$ALLOCATOR_CONFIG" ] && sudo cmp -s "$TEMP_CONFIG" "$ALLOCATOR_CONFIG"; then
  echo "Nitro allocator already matches the measured V2 topology"
else
  echo "Configuring Nitro allocator: CPUs=${REQUIRED_CPUS} memory_mib=${REQUIRED_MEMORY_MIB}"
  sudo nitro-cli terminate-enclave --all 2>/dev/null || true
  sudo install -D -m 0644 "$TEMP_CONFIG" "$ALLOCATOR_CONFIG"
  sudo systemctl restart "$ALLOCATOR_SERVICE"
fi

sudo systemctl is-active --quiet "$ALLOCATOR_SERVICE" || {
  echo "ERROR: Nitro allocator service is not active" >&2
  exit 1
}
sudo cmp -s "$TEMP_CONFIG" "$ALLOCATOR_CONFIG" || {
  echo "ERROR: active Nitro allocator configuration differs from topology" >&2
  exit 1
}

echo "Nitro allocator is ready for the full V2 topology"
