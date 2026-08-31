# Gateway infra-only telemetry

Opt-in OpenTelemetry for the gateway HOST process. One server span per HTTP
request; nothing else from that process. See
`gateway/observability/otel_bootstrap.py` for the full contract, and
"What is NOT instrumented" below for everything this leaves dark.

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
normally — `/health/v2-authority`, `/health/routing-experiments`,
`/attest/health`, `/attestation/health` and the
`/attestation/deploy-readiness` readiness probe all export spans.
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

**Span status marks 5xx only.** `_emit` sets `Status(StatusCode.ERROR)` only
when the response code is `>= 500`, so a 4xx response leaves the status
`Unset` and counting errors from the span status drops every rejection — the
only 4xx signal is the `http.response.status_code` attribute.
`StatusCode.OK` is never set, so a `status != Ok` test matches every span:
over the trailing 7 days the gateway's spans are 439,148 `Unset` (11,692 of
them 4xx), 2,003 `Error`, and 0 `Ok`.

**A route label does not mean the gateway implements that method.** The span
name is `<METHOD> <route template>`, and the template is resolved from the
router *after* routing has run — which a method mismatch still reaches. A
request to a path that exists but not for that verb gets Starlette's 405 and
still emits a span carrying the real template, so `POST /some/route` in the
span stream is not evidence that a POST handler exists. The fail-closed
validator does not catch this either: its allowlist is built from
`route.path` for every entry in `app.routes` and carries no methods at all.
Over the trailing 7 days there are 22 such spans — `POST /` (19),
`OPTIONS /` (2) and one `GET /research-lab/loop-diagnostics` — and because
span status marks 5xx only, every one of them scores as a success in any
status-based panel. Read the `http.response.status_code` attribute before
concluding a route serves a method.

**A missing span is not proof of a missing request.** Export goes through a
fail-closed complete-envelope validator: a span that does not match the
expected scope, kind, parentage, attribute set, name, resource, or route
allowlist is dropped whole rather than mutated or partially exported. The
drop is invisible from here in both directions — the wrapper returns
`SpanExportResult.SUCCESS` to the SDK even when it accepted nothing, and the
only record is a `print("gateway_otel_span_dropped …")` line on the gateway
process's stdout, which is not itself instrumented and never reaches the
collector. There is no evidence that drops have occurred, and by construction
there could not be. The practical rule: "this route went dark" is a statement
about the export path as much as about traffic, and a silence long enough to
act on should be corroborated against a neighbouring route before it is
treated as an outage.

## What is NOT instrumented

`configure_gateway_otel` is imported in exactly one place (`gateway/main.py`),
and its middleware returns early unless `scope["type"] == "http"`. The
gateway's HTTP request path is therefore the only thing in this repository
that reaches the OTel collector. Everything else is dark to it:

| dark to OTel | what it does have |
|---|---|
| The gateway's own in-process background tasks — epoch monitor, checkpoint, daily Arweave anchor, hourly Arweave batch, ICP rotation, rate-limiter and hotkey-bucket cleanup, fulfillment lifecycle, the SOURCE_ADD dispatcher and the PCR0 builder (all `asyncio.create_task` in the `main.py` lifespan) | nothing — they never pass through the ASGI middleware |
| Research Lab worker fleets — separate OS processes (`gateway/research_lab/worker_process.py`, `scripts/run_research_lab_*worker*.py`) | Sentry only |
| Validator, miner and auditor neurons (`neurons/*.py`) | Sentry only |
| Validator TEE **host-side** tooling (`validator_tee/host/*`: runtime bootstrap, release gate, gateway PCR0 builder) | Sentry only — it runs outside the enclave, so it *could* carry spans |
| The TEE **enclave** itself (`validator_tee/enclave/`) | nothing, deliberately: any added dependency changes PCR0 |
| `gw_restart.sh` and the GitHub Actions workflows | nothing |

**Sentry is not coverage for any of these.** Sentry events go to a third-party
vendor, never to the OTLP endpoint, and they are error events rather than
traces. A background task that stalls, a worker fleet that stops consuming, or
a restart that never finishes leaves nothing at all in the OTel store — "we
have Sentry" does not close that gap.

### Host metrics: shipped, not running

Host CPU / memory / disk / network are out of scope for the in-process
exporter by design, but this repository does ship a standalone collector for
them — see
[`gateway/observability/HOST_METRICS.md`](gateway/observability/HOST_METRICS.md)
(`otelcol-hostmetrics.yaml` plus `install_hostmetrics_collector.sh`), which
exports to `GATEWAY_OTEL_METRICS_ENDPOINT`.

**As of 2026-08-29 no host metrics are arriving**: the collector's metrics and
histogram tables hold zero rows from this deployment over the trailing 7 days.
This is an ops item, not a code defect — the installer is deliberately not
called by `gw_restart.sh`, so either
`sudo gateway/observability/install_hostmetrics_collector.sh` has not been run
on the gateway host, or the unit's endpoint/token is wrong. Until it is fixed,
host resource questions cannot be answered from telemetry.

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

## Deploy signal

**What deploys this repo:** nothing in GitHub. Production rollout is
`gw_restart.sh`, run on the gateway host by a person or by something host-side
that is not in this repository. There is no deploy workflow, no GitHub
Deployments API use, and no deploy commit status.

**Consequence for telemetry:** the CI/CD event stream is empty. Zero rows have
ever been ingested with `onepatch_source = 'cicd'` — re-confirmed over a 30-day
window on 2026-08-30. Every rung of the usual deploy-signal ladder
(`deployment_status`, commit `status`, `workflow_job`, `workflow_run`) is
unreachable here, so no deploy-anchored alerting, and no post-deploy soak, can
be built.

**Best reachable proxy, and it is only a proxy.** A push to `main` in the
repo-activity log stream:

```
onepatch_source = 'github' AND attrs.github.event = 'push' AND attrs.github.push.ref = 'refs/heads/main' AND attrs.github.repo = 'leadpoet/leadpoet'
```

A gateway restart typically follows such a push by about 25 minutes. It is not
a deploy signal and should not be treated as one, in both directions:

| date | pushes to `main` | gateway restarts |
|---|---|---|
| 2026-08-26 | 8 | 3 |
| 2026-08-27 | 30 | 11 |
| 2026-08-28 | 22 | 14 |
| 2026-08-29 | 9 | 16 |

Several pushes batch into one restart, so on a busy day the proxy over-counts;
and on 2026-08-29 the gateway restarted twelve times with no push behind any of
them, so it under-counts exactly when something is wrong. Restart counts above
are taken from the 5.00-second `GET /health/v2-authority` poll that
`gw_restart.sh` runs while waiting for the process to come back — that burst,
not the push, is the only first-hand evidence of a rollout this deployment
emits.

**Confidence: low.** The proxy is good enough to ask "did a release plausibly
cause this?" and not good enough to anchor anything to.

**The one-shot escalation is spent.** PR #95 proposed a single `curl` at the end
of a successful `gw_restart.sh` run, posting a deploy event to the OnePatch
ingest endpoint — roughly one second, deletable in one line. It was closed
unmerged on 2026-08-29. The same beacon now rides along in the still-open
[#103](https://github.com/leadpoet/leadpoet/pull/103) alongside the liveness
watchdog. If #103 merges, this section should be re-derived against the real
event; until then this is the ceiling, and no further escalation will be opened.

## Related: error monitoring (Sentry)

Error capture (crashes and ERROR-level logs) is a separate, equally
fail-closed integration with the same philosophy — namespaced
`LEADPOET_SENTRY_*` variables, explicit options, host processes only, never
the enclaves, and a scrubber that keeps trajectory/training data, prompts,
benchmarks, and contact data out of every event. See
[`docs/sentry_error_monitoring.md`](docs/sentry_error_monitoring.md) and
`leadpoet_observability/sentry_bootstrap.py`;
`tests/test_sentry_boundary_guard.py` enforces the boundary in CI.
