#!/usr/bin/env bash
# Install a host-metrics OpenTelemetry Collector on a gateway EC2 host.
#
# Ships CPU / memory / storage / network metrics over OTLP/HTTP to the SAME
# collector the gateway already sends traces to, reusing the gateway's existing
# GATEWAY_OTEL_ENDPOINT and GATEWAY_OTEL_TOKEN. Runs as a standalone systemd
# service (otelcol-hostmetrics) — completely separate from the gateway process,
# so it cannot affect the in-process span boundary or the PCR0 allowlist.
#
# Idempotent: safe to re-run to pick up a new config or a rotated token.
# Opt-in: nothing in the normal gateway restart flow calls this — an operator
# runs it once per host.
#
#   sudo ./install_hostmetrics_collector.sh
#
# Reads the gateway env file for the endpoint + token (override with env vars):
#   GATEWAY_ENV_FILE                (default /home/ec2-user/.config/leadpoet/gateway.env)
#   GATEWAY_OTEL_METRICS_ENDPOINT   (default: GATEWAY_OTEL_ENDPOINT with /v1/traces -> /v1/metrics)
#   OTELCOL_VERSION                 (default 0.109.0)

set -euo pipefail

GATEWAY_ENV_FILE="${GATEWAY_ENV_FILE:-/home/ec2-user/.config/leadpoet/gateway.env}"
OTELCOL_VERSION="${OTELCOL_VERSION:-0.109.0}"
CONFIG_SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/otelcol-hostmetrics.yaml"
CONFIG_DST="/etc/otelcol-hostmetrics/config.yaml"
ENV_DST="/etc/otelcol-hostmetrics/collector.env"
UNIT_DST="/etc/systemd/system/otelcol-hostmetrics.service"

if [[ $EUID -ne 0 ]]; then
  echo "ERROR: run as root (sudo)." >&2
  exit 1
fi

# --- 1. Resolve the OTLP endpoint + token from the gateway env ---------------
if [[ -f "$GATEWAY_ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  set -a; source "$GATEWAY_ENV_FILE"; set +a
fi

: "${GATEWAY_OTEL_ENDPOINT:?GATEWAY_OTEL_ENDPOINT not set (expected in $GATEWAY_ENV_FILE)}"
: "${GATEWAY_OTEL_TOKEN:?GATEWAY_OTEL_TOKEN not set (expected in $GATEWAY_ENV_FILE)}"

# Derive the metrics URL from the traces URL unless one was given explicitly.
if [[ -z "${GATEWAY_OTEL_METRICS_ENDPOINT:-}" ]]; then
  if [[ "$GATEWAY_OTEL_ENDPOINT" == *"/v1/traces" ]]; then
    GATEWAY_OTEL_METRICS_ENDPOINT="${GATEWAY_OTEL_ENDPOINT%/v1/traces}/v1/metrics"
  else
    echo "ERROR: GATEWAY_OTEL_ENDPOINT ('$GATEWAY_OTEL_ENDPOINT') does not end in /v1/traces;" >&2
    echo "       set GATEWAY_OTEL_METRICS_ENDPOINT explicitly and re-run." >&2
    exit 1
  fi
fi
echo "Exporting host metrics to: $GATEWAY_OTEL_METRICS_ENDPOINT"

# --- 2. Install the otelcol-contrib binary (hostmetrics is a contrib comp) ---
if ! command -v otelcol-contrib >/dev/null 2>&1; then
  arch="$(uname -m)"; case "$arch" in
    x86_64) arch=amd64 ;; aarch64|arm64) arch=arm64 ;;
    *) echo "ERROR: unsupported arch $arch" >&2; exit 1 ;;
  esac
  base="https://github.com/open-telemetry/opentelemetry-collector-releases/releases/download/v${OTELCOL_VERSION}"
  rpm="otelcol-contrib_${OTELCOL_VERSION}_linux_${arch}.rpm"
  echo "Installing otelcol-contrib ${OTELCOL_VERSION} (${arch})..."
  tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
  curl -fsSL -o "$tmp/$rpm" "$base/$rpm"
  # Amazon Linux / RHEL hosts use rpm; the systemd unit shipped by the package
  # is masked below in favor of our dedicated one.
  rpm -Uvh --replacepkgs "$tmp/$rpm"
  systemctl disable --now otelcol-contrib >/dev/null 2>&1 || true
fi

# --- 3. Drop config + env + a dedicated systemd unit -------------------------
install -d -m 0755 /etc/otelcol-hostmetrics
install -m 0644 "$CONFIG_SRC" "$CONFIG_DST"

umask 077
cat >"$ENV_DST" <<EOF
GATEWAY_OTEL_METRICS_ENDPOINT=$GATEWAY_OTEL_METRICS_ENDPOINT
GATEWAY_OTEL_TOKEN=$GATEWAY_OTEL_TOKEN
EOF
chmod 0600 "$ENV_DST"

cat >"$UNIT_DST" <<EOF
[Unit]
Description=OpenTelemetry Collector - gateway host metrics
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
EnvironmentFile=$ENV_DST
ExecStart=/usr/bin/otelcol-contrib --config $CONFIG_DST
Restart=on-failure
RestartSec=10
# Least privilege: the collector only reads host stats, it needs no write access.
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=read-only
ReadOnlyPaths=/

[Install]
WantedBy=multi-user.target
EOF

# --- 4. Enable + (re)start ---------------------------------------------------
systemctl daemon-reload
systemctl enable --now otelcol-hostmetrics.service
systemctl restart otelcol-hostmetrics.service

echo
echo "Installed. Check status with:  systemctl status otelcol-hostmetrics"
echo "Follow logs with:              journalctl -u otelcol-hostmetrics -f"
