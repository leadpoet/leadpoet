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

## Service map

One service reports telemetry: **`leadpoet-gateway`**. By design it emits a
single `SERVER` span per HTTP request and nothing else — so it makes no
instrumented outgoing (client) calls, and it reports **no environment**
(`deployment.environment.name` is unset; the resource is a fixed attribute
dict with a constant service name, never an env value).

| service | incoming (server spans, `kind = 2`) | outgoing (client spans, `kind = 3`) | environments |
|---|---|---|---|
| `leadpoet-gateway` | HTTP routes across the families below | none — server spans only | none reported |

**Incoming route families** (span name is `<method> <route-template>`; the
list is the live surface observed over the trailing 7 days, grouped by area):

- **Fulfillment** — `/fulfillment/results/{request_id}`,
  `/fulfillment/requests/active`, `/fulfillment/scoring`,
  `/fulfillment/leaderboard`, `/fulfillment/rewards/active`,
  `/fulfillment/banned-hotkeys`, `/fulfillment/excluded-now/{request_id}`.
- **Research Lab** — `/research-lab/source-adapters`, `/research-lab/status`,
  `/research-lab/public/loops`, `/research-lab/public/loops/{ticket_id}`,
  `/research-lab/public/loops/summary`,
  `/research-lab/benchmarks/public/latest`,
  `/research-lab/allocations/attested/{epoch}`,
  `/research-lab/allocations/live/{epoch}`, `/research-lab/openrouter-keys`,
  `/research-lab/openrouter-keys/credential-recipient`,
  `/research-lab/tickets`, `/research-lab/loop-start`,
  `/research-lab/reports/*`, `/research-lab/engine/issues`.
- **Weights (v2)** — `/weights/v2/published/{netuid}/{epoch_id}`,
  `/weights/v2/published-compact/{netuid}/{epoch_id}`,
  `/weights/v2/release-evidence/{commit_sha}`,
  `/weights/v2/latest/{netuid}/{epoch_id}`, `/weights/current/{netuid}`,
  `/weights/inputs/v2`, `/weights/submit/v2`, `/weights/submit/compact/v2`,
  `/weights/finalize/v2`, `/weights/finalize/compact/v2`,
  `/weights/subnet-epoch/boundary/v1`.
- **Attestation** — `/attestation/deploy-readiness`,
  `/attestation/document`, `/attestation/health`, `/attest`,
  `/attest/health`.
- **Qualification** — `/qualification/model/presign`,
  `/qualification/leaderboard`,
  `/qualification/model/rate-limit/{miner_hotkey}`.
- **Epoch** — `/epoch/state`, `/epoch/current`, `/epoch/{epoch_id}/leads`.
- **Meta / root** — `/`, `/build-info`, `/health/v2-authority`, `/metrics`.
- **Unresolved** — any path that matches no registered route exports the
  fixed `/_unmatched` template (never the concrete path).

The map has two sources that agree here: the routes the gateway registers in
code and the server spans that actually arrive. Regenerate this section from
live telemetry (server/client span names by `service_name`) whenever the doc
is touched. Note: PR #41 wires standalone-collector *host metrics* for the
gateway, but no gateway metrics are arriving at the OnePatch store yet — the
only metrics present are OnePatch's own internal ones — so metrics are
intentionally absent from this map until they land.

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
