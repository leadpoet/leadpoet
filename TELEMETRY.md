# Gateway infra-only telemetry

Opt-in OpenTelemetry for the gateway HOST process. One server span per HTTP
request; nothing else. See `gateway/observability/otel_bootstrap.py` for the
full contract.

## What a span contains — and all it can ever contain

| attribute | example |
|---|---|
| `http.request.method` | `GET` |
| `http.route` | `/research-lab/allocations/attested/{epoch}` (route *template* — never the concrete path) |
| `http.response.status_code` | `200` |
| `duration_ms` | `12.4` |

No bodies, query strings, headers, DB statements, model I/O, prompts, or
completions. Health/liveness routes are suppressed entirely.

## Enabling (gateway host only)

```bash
export GATEWAY_OTEL_ENABLED=1
export GATEWAY_OTEL_ENDPOINT="https://<collector>/v1/traces"
export GATEWAY_OTEL_TOKEN="<token>"            # sent as Authorization: Bearer
export GATEWAY_OTEL_SERVICE_NAME="leadpoet-gateway"   # optional
```

Both `GATEWAY_OTEL_ENABLED` and `GATEWAY_OTEL_ENDPOINT` are required; anything
less is a complete no-op. Wiring failures are swallowed — telemetry can never
delay or break gateway startup.

## Why the boundary is a guarantee, not a promise

1. **Explicit destination, fixed resource.** The exporter is constructed
   with explicit `endpoint=`/`headers=` arguments from the private
   `GATEWAY_OTEL_*` variables, and the provider resource is a fixed
   attribute dict (the SDK's env-merging resource factory is never used).
   The standard ambient exporter/resource variables are never read and
   never set — ambient auto-instrumentation or a stray bare exporter has no
   destination and no-ops.
2. **Fail-closed span validator.** Before export every span must have the
   `gateway.http` instrumentation scope, exactly the four attributes above
   with the approved types, no events/links/status descriptions, the fixed
   resource, and a registered route template (or the literal `/_unmatched`).
   A span violating any of that is dropped entirely — never mutated, never
   partially exported — and counted in a warning that omits the rejected
   values.
3. **CI guard** (`tests/test_otel_boundary_guard.py`) fails the build if:
   auto-instrumentation packages enter the requirements, a launch path uses
   the process-wrapping launcher, anything sets the global tracer provider,
   ambient exporter/resource variables appear anywhere, an OTLP exporter is
   imported outside the bootstrap module, or the bootstrap stops building a
   fixed resource / fixed unmatched-route label.

## Isolation properties

- Dedicated (non-global) tracer provider: Langfuse and every other library
  are untouched.
- Unresolved routes export the fixed `/_unmatched` label — a
  client-controlled path segment is never exported.
- Gateway host process only — never the attested enclaves; no new
  dependencies, so no PCR0 change.

## What this does NOT replace

Code enforcement stops accidental leakage; it is not full security by
itself. Operationally the collector token should be ingest-only and
rotated, repository access stays scoped, and the telemetry vendor's
retention / no-training / deletion terms should be agreed in writing.
