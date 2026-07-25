# Gateway host metrics (CPU / memory / storage / network)

The gateway's in-process telemetry (`otel_bootstrap.py`) is deliberately
**infra-only spans** — one HTTP server span per request, nothing else. It does
not, and should not, emit host resource metrics.

Host CPU / memory / disk / network come from a **separate** OpenTelemetry
Collector process installed on each gateway host. It is fully independent of the
gateway:

- Different process, no shared code. It never imports `otel_bootstrap.py` and
  adds **no Python dependency** to the gateway.
- Runs on the host only, in **no enclave** — so there is **no PCR0 change** and
  the fail-closed span boundary is untouched.
- Reuses the gateway's existing OTLP destination and credential
  (`GATEWAY_OTEL_ENDPOINT` / `GATEWAY_OTEL_TOKEN`), sending to the **metrics**
  path (`/v1/metrics`) instead of `/v1/traces`.

## Files

| file | what it is |
|---|---|
| `otelcol-hostmetrics.yaml` | Collector config: `hostmetrics` receiver (cpu, load, memory, disk, filesystem, network, paging) → `otlphttp` exporter. |
| `install_hostmetrics_collector.sh` | Installs `otelcol-contrib`, writes the config + a systemd unit (`otelcol-hostmetrics`), enables and starts it. Idempotent. |

## Install (per host)

Requires `GATEWAY_OTEL_ENDPOINT` and `GATEWAY_OTEL_TOKEN` in the gateway env
file (`/home/ec2-user/.config/leadpoet/gateway.env` by default):

```bash
sudo gateway/observability/install_hostmetrics_collector.sh
```

The script derives the metrics URL from the traces endpoint
(`…/v1/traces` → `…/v1/metrics`); override with `GATEWAY_OTEL_METRICS_ENDPOINT`
if your collector routes metrics elsewhere.

Verify:

```bash
systemctl status otelcol-hostmetrics
journalctl -u otelcol-hostmetrics -f
```

## What lands in telemetry

Standard OTel `system.*` metrics under `service.name = leadpoet-gateway-host`,
tagged with `host.name` / EC2 instance id (via `resourcedetection`) so each host
is separable:

- `system.cpu.utilization`, `system.cpu.load_average.*`
- `system.memory.usage`, `system.memory.utilization`
- `system.filesystem.usage`, `system.filesystem.utilization`, `system.disk.*`
- `system.network.io`, `system.network.packets`, …
- `system.paging.*`

## Uninstall

```bash
sudo systemctl disable --now otelcol-hostmetrics
sudo rm -rf /etc/otelcol-hostmetrics /etc/systemd/system/otelcol-hostmetrics.service
sudo systemctl daemon-reload
```

Removing it stops the host metrics; the gateway's request spans are unaffected
either way.
