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
completions.

**Suppression is three exact paths, not a family.** `_SUPPRESSED_ROUTES` is
`{"/health", "/health/live", "/health/ready"}`, matched against the concrete
request path. Every other health, liveness or readiness route is traced
normally — `/health/v2-authority`, `/attest/health`, `/attestation/health` and
the `/attestation/deploy-readiness` readiness probe all export spans.
`/attestation/deploy-readiness` is probe traffic and, whenever its caller is
running, one of the highest-volume routes on the gateway, so it dominates any
unfiltered aggregate; exclude it before reading request counts as user traffic.
Exclude it from latency aggregates too, and in both directions: it answers in
about three milliseconds, so its arrival or departure moves an all-route p95 by
more than a real slowdown does. It stopped dead at 21:24 UTC on 2026-08-24 and
has emitted nothing since; the 149% jump in the gateway's aggregate p95 that
followed was that absence, not a regression. `/health/v2-authority` is
low-volume by comparison but a recurring source of 5xx, so it is worth keeping
in error queries.

**`duration_ms` above is an attribute name, not a column name.** The middleware
sets it as a float count of milliseconds, and it does arrive. But the span's
*duration column* is `duration_ns` — Int64 nanoseconds, derived by the store
from the span's start and end timestamps. Query `duration_ns` (divide by `1e6`
for milliseconds); a query written against a `duration_ms` column matches
nothing.

**There is no environment on a span.** The fixed resource dict carries only
`service.name`, so no `deployment.environment` attribute is ever set and the
store's `env` column is `''` on every gateway span. Filtering by environment
silently matches nothing — filter on `service_name` instead.

## Enabling (gateway host only)

```bash
export GATEWAY_OTEL_ENABLED=1
export GATEWAY_OTEL_ENDPOINT="https://<collector>/v1/traces"
export GATEWAY_OTEL_TOKEN="<token>"            # REQUIRED; sent as Authorization: Bearer
```

All three variables are required; anything less is a complete no-op. The
token must be non-empty (an empty explicit headers dict would let the pinned
exporter fall back to ambient header variables). The service name is a
constant (`leadpoet-gateway`), not an environment value. Initialization is
REFUSED at runtime if any ambient standard exporter variable is present in
the process environment — a CI grep cannot see variables injected by a
restart script or the live process env, so the bootstrap checks and logs the
offending names (never values) and stays off. Wiring failures are swallowed —
telemetry can never delay or break gateway startup.

## Why the boundary is a guarantee, not a promise

1. **Explicit destination, fixed resource.** The exporter is constructed
   with explicit `endpoint=`/`headers=` arguments from the private
   `GATEWAY_OTEL_*` variables, and the provider resource is a fixed
   attribute dict (the SDK's env-merging resource factory is never used).
   The standard ambient exporter/resource variables are never read and
   never set — ambient auto-instrumentation or a stray bare exporter has no
   destination and no-ops.
2. **Fail-closed complete-envelope validator.** Before export every span
   must have the `gateway.http` instrumentation scope with no
   version/schema metadata, `SERVER` kind, a root context (no parent, empty
   trace state), exactly the four attributes above with the approved types,
   a standard HTTP method, a span name equal to exactly `<method> <route>`,
   no events/links/status descriptions, the fixed resource, and a
   registered route template (or the literal `/_unmatched`). A span
   violating any of that is dropped entirely — never mutated, never
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
- Pure ASGI middleware: no `BaseHTTPMiddleware` task-group wrapper, and
  telemetry failures cannot replace a response or mask an endpoint failure.
- Unresolved routes export the fixed `/_unmatched` label — a
  client-controlled path segment is never exported.
- Gateway host process only — never the attested enclaves; no new
  dependencies, so no PCR0 change.

## What this does NOT replace

Code enforcement stops accidental leakage; it is not full security by
itself. Operationally the collector token should be ingest-only and
rotated, repository access stays scoped, and the telemetry vendor's
retention / no-training / deletion terms should be agreed in writing.

## Related: error monitoring (Sentry)

Error capture (crashes and ERROR-level logs) is a separate, equally
fail-closed integration with the same philosophy — namespaced
`LEADPOET_SENTRY_*` variables, explicit options, host processes only, never
the enclaves, and a scrubber that keeps trajectory/training data, prompts,
benchmarks, and contact data out of every event. See
[`docs/sentry_error_monitoring.md`](docs/sentry_error_monitoring.md) and
`leadpoet_observability/sentry_bootstrap.py`;
`tests/test_sentry_boundary_guard.py` enforces the boundary in CI.
