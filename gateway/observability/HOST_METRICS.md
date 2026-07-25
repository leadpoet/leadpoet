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
| `read_gateway_otel_env.py` | Reads only the three OTel settings from `gateway.env` without executing shell content. |
| `install_hostmetrics_collector.sh` | Installs a verified `otelcol-contrib` binary and an unprivileged systemd unit (`otelcol-hostmetrics`). |

## Install — explicit host provisioning

This installer is intentionally **not** called by `gw_restart.sh`. Downloading a
package or changing systemd state must not become a dependency of the canonical
gateway restart. Install or refresh the collector as a separate, explicitly
authorized host operation:

```bash
cd /home/ec2-user/leadpoet_repo
sudo gateway/observability/install_hostmetrics_collector.sh
```

The operation is idempotent. Re-run it after rotating `GATEWAY_OTEL_TOKEN` or
changing the collector config. It:

- parses `gateway.env` as newline- or NUL-separated data and never sources it;
- downloads a versioned archive with bounded retries and verifies its SHA-256;
- validates the exact collector config before replacing the active service;
- runs as the dedicated `otelcol-contrib` user with no Linux capabilities,
  read-only system paths, an inaccessible home tree, and memory/task limits;
- enables only a hostmetrics receiver, so it opens no application ingest port.

The default collector version is `0.153.0`. A different `OTELCOL_VERSION`
requires an explicit matching `OTELCOL_SHA256`.

The script derives the metrics URL from the traces endpoint
(`…/v1/traces` → `…/v1/metrics`); override with `GATEWAY_OTEL_METRICS_ENDPOINT`
if your collector routes metrics elsewhere.

Verify:

```bash
systemctl status otelcol-hostmetrics
journalctl -u otelcol-hostmetrics --since "10 minutes ago"
```

Then ask OnePatch to confirm recent
`service.name = leadpoet-gateway-host` metrics before considering the rollout
successful. A running service alone does not prove remote ingestion.

## What lands in telemetry

Standard OTel `system.*` metrics under `service.name = leadpoet-gateway-host`,
tagged with `host.name` so each host is separable. The EC2 resource detector is
deliberately disabled; AWS account, AMI, region, availability-zone, and instance
metadata are not exported.

- `system.cpu.utilization`, `system.cpu.load_average.*`
- `system.memory.usage`, `system.memory.utilization`
- `system.filesystem.usage`, `system.filesystem.utilization`, `system.disk.*`
- `system.network.io`, `system.network.packets`, …
- `system.paging.*`

## Uninstall

```bash
sudo systemctl disable --now otelcol-hostmetrics
sudo rm -f /etc/systemd/system/otelcol-hostmetrics.service
sudo rm -f /etc/otelcol-hostmetrics/config.yaml /etc/otelcol-hostmetrics/collector.env
sudo rmdir /etc/otelcol-hostmetrics
sudo systemctl daemon-reload
```

The versioned binary and service account are intentionally retained for a safe
reinstall. The gateway's request spans are unaffected either way.
