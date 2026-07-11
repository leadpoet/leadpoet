#!/usr/bin/env bash
set -euo pipefail

HOST="${1:-leadpoet-gateway}"

if [[ "${LEADPOET_PROD_WRITE_APPROVED:-}" != "yes" ]]; then
  cat >&2 <<'MSG'
Refusing to mutate production.

Set LEADPOET_PROD_WRITE_APPROVED=yes only after the operator has explicitly
approved installing the Research Lab admin wrapper on the gateway host.
MSG
  exit 2
fi

ssh -o BatchMode=yes "$HOST" 'bash -s' <<'REMOTE'
set -euo pipefail

mkdir -p /home/ec2-user/bin
tmp="$(mktemp /home/ec2-user/bin/research-lab-admin.XXXXXX)"
trap 'rm -f "$tmp"' EXIT

cat > "$tmp" <<'WRAPPER'
#!/usr/bin/env bash
set -euo pipefail

REPO="${LEADPOET_REPO:-/home/ec2-user/leadpoet_repo}"
ENV_FILE="${GATEWAY_ENV_FILE:-/home/ec2-user/.config/leadpoet/gateway.env}"

if [[ ! -d "$REPO/gateway" ]]; then
  echo "research-lab-admin: repo gateway package not found at $REPO" >&2
  exit 2
fi

# Load valid KEY=VALUE lines before Python imports gateway.config. The gateway
# fallback loader prints when it loads env itself; preloading keeps JSON output clean.
if [[ -f "$ENV_FILE" ]]; then
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line#${line%%[![:space:]]*}}"
    line="${line%${line##*[![:space:]]}}"
    [[ -z "$line" || "${line:0:1}" == "#" || "$line" != *"="* ]] && continue

    key="${line%%=*}"
    value="${line#*=}"
    key="${key#${key%%[![:space:]]*}}"
    key="${key%${key##*[![:space:]]}}"
    [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue

    value="${value#${value%%[![:space:]]*}}"
    value="${value%${value##*[![:space:]]}}"
    if [[ ${#value} -ge 2 ]]; then
      first="${value:0:1}"
      last="${value: -1}"
      if { [[ "$first" == "\"" && "$last" == "\"" ]] || [[ "$first" == "'" && "$last" == "'" ]]; }; then
        value="${value:1:${#value}-2}"
      fi
    fi
    export "$key=$value"
  done < "$ENV_FILE"
  export GATEWAY_ENV_FILE=/dev/null
fi

cd "$REPO"
export PYTHONPATH="$REPO"
export GATEWAY_LOG_ROOT="${GATEWAY_LOG_ROOT:-/home/ec2-user/gateway}"
export GATEWAY_TEE_FALLBACK_LOG_DIR="$GATEWAY_LOG_ROOT/gateway/logs/tee_fallback"
exec python3 -m gateway.research_lab.admin "$@"
WRAPPER

chmod 700 "$tmp"
mv "$tmp" /home/ec2-user/bin/research-lab-admin
trap - EXIT

/home/ec2-user/bin/research-lab-admin --help >/dev/null
echo "Installed /home/ec2-user/bin/research-lab-admin"
REMOTE
