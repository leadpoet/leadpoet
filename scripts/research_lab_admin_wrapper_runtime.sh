#!/usr/bin/env bash
set -euo pipefail

REPO="${LEADPOET_REPO:-/home/ec2-user/leadpoet_repo}"
ENV_FILE="${GATEWAY_ENV_FILE:-/home/ec2-user/.config/leadpoet/gateway.env}"
PYTHON_BIN="${GATEWAY_PYTHON_BIN:-/home/ec2-user/venv311/bin/python3}"

if [[ ! -d "$REPO/gateway" ]]; then
  echo "research-lab-admin: repo gateway package not found at $REPO" >&2
  exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "research-lab-admin: gateway Python not executable at $PYTHON_BIN" >&2
  exit 2
fi

# Preload only valid KEY=VALUE records before Python imports gateway.config.
# Canonical release-bound settings below override stale cached values.
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

CANONICAL_SUBNET_EPOCH_CUTOVER_PATH="/home/ec2-user/.config/leadpoet/stateful-epoch-cutover.json"
if [[ -n "${LEADPOET_SUBNET_EPOCH_CUTOVER_PATH:-}" \
  && -n "${LEADPOET_SUBNET_EPOCH_CUTOVER_JSON:-}" ]]; then
  echo "research-lab-admin: set only one subnet epoch cutover authority form" >&2
  exit 2
fi
if [[ -z "${LEADPOET_SUBNET_EPOCH_CUTOVER_PATH:-}" \
  && -z "${LEADPOET_SUBNET_EPOCH_CUTOVER_JSON:-}" ]]; then
  if [[ ! -f "$CANONICAL_SUBNET_EPOCH_CUTOVER_PATH" \
    || ! -s "$CANONICAL_SUBNET_EPOCH_CUTOVER_PATH" ]]; then
    echo "research-lab-admin: canonical subnet epoch cutover manifest is not a regular nonempty file" >&2
    exit 2
  fi
  export LEADPOET_SUBNET_EPOCH_CUTOVER_PATH="$CANONICAL_SUBNET_EPOCH_CUTOVER_PATH"
fi

export AWS_REGION="${AWS_REGION:-us-east-1}"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-$AWS_REGION}"
unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_PROFILE AWS_SESSION_TOKEN AWS_SECURITY_TOKEN
export LEADPOET_AWS_INSTANCE_ROLE_ONLY=true

cd "$REPO"
export PYTHONPATH="$REPO"
export GATEWAY_LOG_ROOT="${GATEWAY_LOG_ROOT:-/home/ec2-user/gateway}"
export GATEWAY_TEE_FALLBACK_LOG_DIR="$GATEWAY_LOG_ROOT/gateway/logs/tee_fallback"
exec "$PYTHON_BIN" -m gateway.research_lab.admin "$@"
