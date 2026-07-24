# Telemetry — leadpoet-gateway

Opt-in, **infra-only** OpenTelemetry for the gateway HOST process. Off by
default; ships nothing unless explicitly enabled via env.

## Service

- `service.name`: `leadpoet-gateway` (override with `OTEL_SERVICE_NAME`)
- Runtime: Python 3.11 / FastAPI (gateway host process, `python -m gateway.main`)
- Instrumentation: hand-written request middleware in
  `gateway/observability/otel_bootstrap.py` using the already-pinned
  `opentelemetry-sdk` / `opentelemetry-exporter-otlp-proto-http` 1.27.0. **No new
  dependencies; the attested enclaves are not instrumented and their PCR0 is
  unaffected.**
- Last regenerated: 2026-07-24

## Enabling (deploy env only — never committed)

Both must be set, or the bootstrap is a complete no-op:

```
GATEWAY_OTEL_ENABLED=true
OTEL_EXPORTER_OTLP_ENDPOINT=https://leadpoet.logger.onepatch.dev
OTEL_EXPORTER_OTLP_HEADERS=Authorization=Bearer <token>
OTEL_SERVICE_NAME=leadpoet-gateway
OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
```

The endpoint and token are set in the gateway host's runtime environment, never
baked into an image or committed to the repo.

## Spans (hand-instrumented)

| Span name | Kind | Source | When it fires | Attributes |
|---|---|---|---|---|
| `<METHOD> <route-template>` | SERVER | `gateway/observability/otel_bootstrap.py` | Once per HTTP request to the gateway host (health probes suppressed) | `http.request.method`, `http.route`, `http.response.status_code`, `duration_ms` |

`http.route` is always the **route template** (e.g.
`/research-lab/allocations/attested/{epoch}`), never the concrete path.

## What is deliberately NOT emitted

This instrumentation is scoped to protect data. It never records:

- Concrete path ids, hotkeys, epochs (only route templates)
- Query strings, request bodies, or response bodies
- Request/response headers (including auth)
- Database statements or query parameters
- Any LLM prompt/completion or model I/O — the Research Lab / scoring / model /
  Langfuse code paths are **not** instrumented
- Anything from inside the attested enclaves

The only data that leaves the host is operational metadata: which route was hit,
its HTTP status, and how long it took. This is enough to observe availability,
latency, and error-rate incidents (e.g. a gateway outage shows as request
failures / absence of traffic) without exposing business or training data.
