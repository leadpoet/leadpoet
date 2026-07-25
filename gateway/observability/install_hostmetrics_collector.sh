#!/usr/bin/env bash
# Install a host-metrics OpenTelemetry Collector on a gateway EC2 host.
#
# This is an explicit host-provisioning action. It is deliberately not called
# by gw_restart.sh, so telemetry setup cannot delay or mutate the canonical
# gateway restart path.
#
# The installer treats gateway.env as data, verifies the downloaded archive,
# validates the collector config, and runs the service without root privileges.
#
#   sudo ./install_hostmetrics_collector.sh
#
# Supported overrides:
#   GATEWAY_ENV_FILE                (default /home/ec2-user/.config/leadpoet/gateway.env)
#   GATEWAY_OTEL_METRICS_ENDPOINT   (default: GATEWAY_OTEL_ENDPOINT with /v1/traces -> /v1/metrics)
#   OTELCOL_VERSION                 (default 0.153.0)
#   OTELCOL_SHA256                  (required for any non-default version)
#   HOSTMETRICS_PYTHON_BIN          (default python3)

set -euo pipefail

GATEWAY_ENV_FILE="${GATEWAY_ENV_FILE:-/home/ec2-user/.config/leadpoet/gateway.env}"
OTELCOL_VERSION="${OTELCOL_VERSION:-0.153.0}"
HOSTMETRICS_PYTHON_BIN="${HOSTMETRICS_PYTHON_BIN:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_SRC="$SCRIPT_DIR/otelcol-hostmetrics.yaml"
ENV_READER="$SCRIPT_DIR/read_gateway_otel_env.py"
CONFIG_DST="/etc/otelcol-hostmetrics/config.yaml"
ENV_DST="/etc/otelcol-hostmetrics/collector.env"
UNIT_DST="/etc/systemd/system/otelcol-hostmetrics.service"
INSTALL_ROOT="/usr/local/lib/leadpoet/otelcol-contrib/$OTELCOL_VERSION"
BINARY_PATH="$INSTALL_ROOT/otelcol-contrib"
SERVICE_USER="otelcol-contrib"
SERVICE_GROUP="otelcol-contrib"

if [[ $EUID -ne 0 ]]; then
  echo "ERROR: run as root (sudo)." >&2
  exit 1
fi

for required_command in curl getent install sha256sum systemctl tar timeout useradd; do
  if ! command -v "$required_command" >/dev/null 2>&1; then
    echo "ERROR: required command is unavailable: $required_command" >&2
    exit 1
  fi
done
if ! command -v "$HOSTMETRICS_PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: Python interpreter is unavailable: $HOSTMETRICS_PYTHON_BIN" >&2
  exit 1
fi

resolve_gateway_value() {
  "$HOSTMETRICS_PYTHON_BIN" "$ENV_READER" \
    --env-file "$GATEWAY_ENV_FILE" \
    --key "$1"
}

# Resolve only the three OTel values. Never source or execute gateway.env.
GATEWAY_OTEL_ENDPOINT="$(resolve_gateway_value GATEWAY_OTEL_ENDPOINT)"
GATEWAY_OTEL_TOKEN="$(resolve_gateway_value GATEWAY_OTEL_TOKEN)"
GATEWAY_OTEL_METRICS_ENDPOINT="$(resolve_gateway_value GATEWAY_OTEL_METRICS_ENDPOINT)"

if [[ -z "$GATEWAY_OTEL_ENDPOINT" || -z "$GATEWAY_OTEL_TOKEN" ]]; then
  echo "ERROR: GATEWAY_OTEL_ENDPOINT and GATEWAY_OTEL_TOKEN are required." >&2
  echo "       Expected explicit environment values or entries in $GATEWAY_ENV_FILE." >&2
  exit 1
fi

# Derive the metrics URL from the traces URL unless one was given explicitly.
if [[ -z "$GATEWAY_OTEL_METRICS_ENDPOINT" ]]; then
  if [[ "$GATEWAY_OTEL_ENDPOINT" == *"/v1/traces" ]]; then
    GATEWAY_OTEL_METRICS_ENDPOINT="${GATEWAY_OTEL_ENDPOINT%/v1/traces}/v1/metrics"
  else
    echo "ERROR: GATEWAY_OTEL_ENDPOINT ('$GATEWAY_OTEL_ENDPOINT') does not end in /v1/traces;" >&2
    echo "       set GATEWAY_OTEL_METRICS_ENDPOINT explicitly and re-run." >&2
    exit 1
  fi
fi

traces_endpoint_regex='^https://[A-Za-z0-9._~:/?&=%+-]+/v1/traces$'
metrics_endpoint_regex='^https://[A-Za-z0-9._~:/?&=%+-]+/v1/metrics$'
token_regex='^[A-Za-z0-9._~+/=-]+$'
if ! [[ "$GATEWAY_OTEL_ENDPOINT" =~ $traces_endpoint_regex ]]; then
  echo "ERROR: GATEWAY_OTEL_ENDPOINT must be an HTTPS OTLP /v1/traces URL." >&2
  exit 1
fi
if ! [[ "$GATEWAY_OTEL_METRICS_ENDPOINT" =~ $metrics_endpoint_regex ]]; then
  echo "ERROR: GATEWAY_OTEL_METRICS_ENDPOINT must be an HTTPS OTLP /v1/metrics URL." >&2
  exit 1
fi
if ! [[ "$GATEWAY_OTEL_TOKEN" =~ $token_regex ]]; then
  echo "ERROR: GATEWAY_OTEL_TOKEN contains unsupported characters." >&2
  exit 1
fi
echo "Exporting host metrics to: $GATEWAY_OTEL_METRICS_ENDPOINT"

# Resolve the exact verified archive for the supported gateway architectures.
arch="$(uname -m)"
case "$arch" in
  x86_64)
    arch="amd64"
    default_sha256="f7be4acf2c04058875073afc2e74ef885a66ef4fac8a4bfc93faa7235cb1c174"
    ;;
  aarch64|arm64)
    arch="arm64"
    default_sha256="df0f98422428ad90021faef09a07bd685102f112483278ad8555383307117020"
    ;;
  *)
    echo "ERROR: unsupported architecture: $arch" >&2
    exit 1
    ;;
esac

if [[ "$OTELCOL_VERSION" != "0.153.0" && -z "${OTELCOL_SHA256:-}" ]]; then
  echo "ERROR: OTELCOL_SHA256 is required when overriding OTELCOL_VERSION." >&2
  exit 1
fi
OTELCOL_SHA256="${OTELCOL_SHA256:-$default_sha256}"
if ! [[ "$OTELCOL_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "ERROR: OTELCOL_SHA256 must be a lowercase SHA-256 digest." >&2
  exit 1
fi

archive="otelcol-contrib_${OTELCOL_VERSION}_linux_${arch}.tar.gz"
archive_url="https://github.com/open-telemetry/opentelemetry-collector-releases/releases/download/v${OTELCOL_VERSION}/${archive}"

if ! getent passwd "$SERVICE_USER" >/dev/null; then
  useradd --system --user-group --no-create-home --shell /sbin/nologin "$SERVICE_USER"
fi

if [[ ! -x "$BINARY_PATH" ]] \
  || ! "$BINARY_PATH" --version 2>&1 | grep -Fq "version $OTELCOL_VERSION"; then
  echo "Installing verified otelcol-contrib $OTELCOL_VERSION ($arch)..."
  tmp="$(mktemp -d /var/tmp/otelcol-hostmetrics.XXXXXX)"
  trap 'rm -rf "$tmp"' EXIT
  curl \
    --fail \
    --location \
    --silent \
    --show-error \
    --connect-timeout 10 \
    --max-time 180 \
    --retry 3 \
    --retry-delay 2 \
    --retry-all-errors \
    --output "$tmp/$archive" \
    "$archive_url"
  (
    cd "$tmp"
    printf '%s  %s\n' "$OTELCOL_SHA256" "$archive" | sha256sum --check -
    mkdir extracted
    tar -xzf "$archive" -C extracted
  )
  if [[ ! -x "$tmp/extracted/otelcol-contrib" ]]; then
    echo "ERROR: verified archive did not contain otelcol-contrib." >&2
    exit 1
  fi
  install -d -o root -g root -m 0755 "$INSTALL_ROOT"
  install -o root -g root -m 0755 "$tmp/extracted/otelcol-contrib" "$BINARY_PATH"
fi

if ! "$BINARY_PATH" --version 2>&1 | grep -Fq "version $OTELCOL_VERSION"; then
  echo "ERROR: installed collector version does not match $OTELCOL_VERSION." >&2
  exit 1
fi

# Validate the exact candidate config before changing the active service.
export GATEWAY_OTEL_METRICS_ENDPOINT GATEWAY_OTEL_TOKEN
timeout --signal=TERM --kill-after=5s 30s \
  "$BINARY_PATH" validate --config "$CONFIG_SRC"

install -d -o root -g "$SERVICE_GROUP" -m 0750 /etc/otelcol-hostmetrics
install -o root -g "$SERVICE_GROUP" -m 0644 "$CONFIG_SRC" "$CONFIG_DST"

umask 077
env_tmp="$(mktemp /etc/otelcol-hostmetrics/collector.env.XXXXXX)"
cat >"$env_tmp" <<EOF
GATEWAY_OTEL_METRICS_ENDPOINT=$GATEWAY_OTEL_METRICS_ENDPOINT
GATEWAY_OTEL_TOKEN=$GATEWAY_OTEL_TOKEN
EOF
chown root:"$SERVICE_GROUP" "$env_tmp"
chmod 0640 "$env_tmp"
mv -f "$env_tmp" "$ENV_DST"

unit_tmp="$(mktemp /etc/systemd/system/otelcol-hostmetrics.service.XXXXXX)"
cat >"$unit_tmp" <<EOF
[Unit]
Description=OpenTelemetry Collector - gateway host metrics
After=network-online.target
Wants=network-online.target
StartLimitIntervalSec=300
StartLimitBurst=5

[Service]
Type=simple
User=$SERVICE_USER
Group=$SERVICE_GROUP
UMask=0077
EnvironmentFile=$ENV_DST
ExecStartPre=$BINARY_PATH validate --config $CONFIG_DST
ExecStart=$BINARY_PATH --config $CONFIG_DST
Restart=on-failure
RestartSec=10
NoNewPrivileges=true
CapabilityBoundingSet=
AmbientCapabilities=
ProtectSystem=strict
ProtectHome=true
ProtectKernelTunables=true
ProtectKernelModules=true
ProtectControlGroups=true
PrivateTmp=true
PrivateDevices=true
ReadOnlyPaths=/
RestrictSUIDSGID=true
RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6
LockPersonality=true
MemoryMax=192M
TasksMax=128

[Install]
WantedBy=multi-user.target
EOF
chown root:root "$unit_tmp"
chmod 0644 "$unit_tmp"
mv -f "$unit_tmp" "$UNIT_DST"

timeout --signal=TERM --kill-after=5s 30s systemctl daemon-reload
timeout --signal=TERM --kill-after=5s 30s systemctl enable otelcol-hostmetrics.service
timeout --signal=TERM --kill-after=5s 30s systemctl restart otelcol-hostmetrics.service
timeout --signal=TERM --kill-after=5s 30s systemctl is-active --quiet otelcol-hostmetrics.service

echo
echo "Installed verified host-metrics collector $OTELCOL_VERSION."
echo "Check status with:  systemctl status otelcol-hostmetrics"
echo "Follow logs with:  journalctl -u otelcol-hostmetrics -f"
