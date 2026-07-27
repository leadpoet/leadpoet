#!/usr/bin/env bash
set -euo pipefail

HOST="${1:-leadpoet-gateway}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WRAPPER_SOURCE="$ROOT/scripts/research_lab_admin_wrapper_runtime.sh"

if [[ "${LEADPOET_PROD_WRITE_APPROVED:-}" != "yes" ]]; then
  cat >&2 <<'MSG'
Refusing to mutate production.

Set LEADPOET_PROD_WRITE_APPROVED=yes only after the operator has explicitly
approved installing the Research Lab admin wrapper on the gateway host.
MSG
  exit 2
fi

if [[ ! -f "$WRAPPER_SOURCE" ]]; then
  echo "Research Lab admin wrapper source not found at $WRAPPER_SOURCE" >&2
  exit 2
fi

WRAPPER_B64="$(base64 < "$WRAPPER_SOURCE" | tr -d '\n')"
ssh -o BatchMode=yes "$HOST" 'bash -s' -- "$WRAPPER_B64" <<'REMOTE'
set -euo pipefail

mkdir -p /home/ec2-user/bin
tmp="$(mktemp /home/ec2-user/bin/research-lab-admin.XXXXXX)"
trap 'rm -f "$tmp"' EXIT

printf '%s' "$1" | base64 --decode > "$tmp"

if ! bash -n "$tmp"; then
  echo "research-lab-admin: candidate wrapper syntax check failed" >&2
  exit 2
fi
chmod 700 "$tmp"
mv "$tmp" /home/ec2-user/bin/research-lab-admin
trap - EXIT

/home/ec2-user/bin/research-lab-admin --help >/dev/null
echo "Installed /home/ec2-user/bin/research-lab-admin"
REMOTE
