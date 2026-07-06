# Telemetry — leadpoet-gateway

## Service

- `service.name`: `leadpoet-gateway`
- Runtime: Python / FastAPI + uvicorn (`gateway/main.py`)
- Instrumentation: `opentelemetry-sdk` 1.27.0, `opentelemetry-exporter-otlp-proto-http` 1.27.0, `opentelemetry-instrumentation-{fastapi,aiohttp-client,httpx,requests,urllib,urllib3}` 0.48b0 (the contrib release paired with SDK 1.27.0 — kept at these pins because Langfuse 2.x requires OTel <1.33 and the validator Firestore stack requires protobuf <5)
- Bootstrap: `gateway/telemetry.py` — `init_telemetry()` runs at the top of `gateway/main.py` (after `gateway/config.py` loads `.env`); `instrument_app(app)` attaches request spans after the app is created. Both are no-ops unless `OTEL_EXPORTER_OTLP_ENDPOINT` is set.
- Scope: **gateway only.** The validator (`neurons/validator.py`), miner, and auditor neurons are not instrumented.
- Backend: OTLP/HTTP via standard `OTEL_EXPORTER_OTLP_*` env vars (see `env.example`). The ingest token is write-only — it can push telemetry and cannot read anything back.
- Last regenerated: 2026-07-06

## Spans (auto-instrumented)

No hand-written spans yet — everything below comes from auto-instrumentation.

| Span name | Kind | When it fires | Key attributes |
|---|---|---|---|
| `GET /`, `POST /presign`, `GET /epoch/*`, `POST /validate/*`, `GET /manifest/*`, `POST /attest`, `GET /attestation/*`, and every other registered FastAPI route | SERVER | One span per inbound HTTP request, named `{METHOD} {route}` (parameterized route, not the raw URL). `GET /health` is excluded to keep k8s probes out of traces. | `http.method`, `http.route`, `http.status_code` |
| `GET` / `POST` (aiohttp, httpx, requests, urllib, urllib3) | CLIENT | Outbound calls — Supabase (PostgREST via httpx), S3 presign/boto3 HTTP layer (urllib3), provider APIs | `http.method`, `http.url`, `http.status_code` |

Client spans nest under the SERVER span of the request that triggered them; W3C `traceparent` is injected on outbound HTTP automatically.

## Metrics

Metrics pipeline (OTLP/HTTP, periodic reader) is configured, but no custom meters are registered yet. (Prometheus metrics via `prometheus_client` are unchanged and separate.)

## Logs

Root-logger records at INFO+ ship as OTLP log records with `trace_id`/`span_id` correlation. Note: much of the gateway logs via bare `print()`, which does **not** flow to OTLP — only `logging` module records do.

## Configuration

```
OTEL_EXPORTER_OTLP_ENDPOINT=https://<tenant>.logger.onepatch.dev   # unset = telemetry fully off
OTEL_EXPORTER_OTLP_HEADERS=Authorization=Bearer%20<write-only-token>
OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
OTEL_SERVICE_NAME=leadpoet-gateway
```

- Local: `.env` (gitignored, file mode 400). Production: the gateway env comes from AWS Secrets Manager (`leadpoet/prod/gateway/env`) — add the same four vars there to enable telemetry in prod.
- Header value is URL-encoded (`Bearer%20…`); the SDK decodes `%20` to a space.
- Never commit the endpoint/token to source — env vars only.
