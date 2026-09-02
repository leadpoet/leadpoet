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

**Suppression is two exact paths, not a family.** `_SUPPRESSED_ROUTES` is
`{"/health/live", "/health/ready"}`, matched against the concrete request
path. Every other health, liveness or readiness route is traced normally —
`/health` itself, `/health/v2-authority`, `/health/routing-experiments`,
`/attest/health`, `/attestation/health` and the
`/attestation/deploy-readiness` readiness probe all export spans.

`/health` is deliberately **not** suppressed. It is the one route that keeps
answering `200` while the worker-authority gate is rejecting everything else,
so its spans are the only positive evidence that separates "gateway up, gate
closed" from "gateway down" — exactly the distinction an operator needs during
an incident. The two high-frequency probes (`/health/live`, `/health/ready`)
stay suppressed because they carry no such information.

Reading spans recorded before this change: `/health` was suppressed alongside
the two probes, so it emits nothing at all up to 2026-09-02 and
`/health/v2-authority` stood in for liveness. Zero `/health` spans over an
older window is the suppression, not a quiet route.

`/attestation/deploy-readiness` is probe traffic, and its volume is set
entirely by whether its caller is running — so it moves aggregates in both
directions and should be excluded before reading request counts as user
traffic. It answers in about three milliseconds, so its arrival or departure
moves an all-route p95 by more than a real slowdown does; the 149% jump in the
gateway's aggregate p95 after 2026-08-24 was its absence, not a regression.

Its 2/min poller stopped at 21:24 UTC on 2026-08-24. It is **no longer a
high-volume route**: as of 2026-09-02, the trailing 7 days hold **4 spans**
(one on 08-27, one on 08-31, two on 09-01), ranking it **30th** by volume —
against roughly 2,400-2,900 spans/day while the poller ran. Treat any figure
describing it as a top-three route as pre-08-24 history.

**`/_unmatched` means "no such route", and nothing else.** The route label
comes from `scope["route"]`, which Starlette only populates *after* routing
resolves — so a request rejected *before* the router runs (the worker-authority
503 gate, the body-size 413, the priority-middleware load-shed 503, and the
fail-closed 503s a booting gateway returns) reaches the telemetry middleware
with no route on the scope at all. `_safe_route_label` handles that by
resolving the path against the app's own route table
(`_resolve_route_template`), replaying the router's own matching: a FULL match
wins, otherwise the first PARTIAL match (Starlette's 405 path), and no
trailing-slash equivalence, because the gateway router runs with
`redirect_slashes=False`. A pre-routing 503 aimed at the attested-allocations
endpoint therefore reads `GET /research-lab/allocations/attested/{epoch}`, not
`/_unmatched`.

`/_unmatched` stays as the fixed fallback for a path that matches **no**
registered route. Only templates registered on this app are ever returned, so
the label set stays bounded and no client-controlled path segment is ever
exported. That last part is not a style preference: the fail-closed validator
requires the attribute set to be *exactly* the four attributes above, so a span
carrying `http.target` or `url.path` would be dropped whole rather than
exported. The concrete path is unexportable by construction — the route
template is the most specific thing a span can ever carry.

**Spans recorded before this change are labelled the old way, and they dominate
the error history.** Every pre-routing rejection captured earlier is filed under
`/_unmatched` with its real path discarded. As of 2026-09-02, over the trailing
7 days, `/_unmatched` carries **4,321 of the gateway's 5,433 5xx spans — 80% of
every error in the store** — across 4,945 spans (`GET` 4,887, `POST` 56,
`PUT` 2). Any per-route error breakdown over a window reaching back before the
rollout therefore still has an 80% bucket that names nothing: report
`/_unmatched` as its own series over such a window rather than folding it into
a route breakdown. Afterwards, expect `/_unmatched` volume to fall sharply and
per-route 5xx counts to rise correspondingly — those errors were always real,
they were just filed under the wrong name.

**`/health/v2-authority`'s error rate is structural, not a fault.** Over the
trailing 7 days the route reads a **48.1% 5xx rate** (1,038 of 2,160, as of
2026-09-02). That number is an artifact of the restart script and must never be
read as an error rate: `gw_restart.sh` polls `GET /health/v2-authority` every
5.00 seconds while waiting for a gateway process it has just stopped, so each
restart contributes roughly 17 503s from a booting server. Verified hour by
hour on 2026-09-01 (UTC):

| hours | requests | 5xx |
|---|---|---|
| 01:00, 02:00, 03:00 — each contains a restart poll | 24, 25, 37 | 16, 17, 16 |
| 04:00-06:00 — sustained load, no restart | 177, 161, 162 | **0, 0, 0** |
| 00:00, 08:00, 09:00 — idle | 3, 1, 1 | 0, 0, 0 |

Three and a half hours at ~170 requests/hour produced zero 5xx. The route is
not broken; it is being polled through boot. Exclude it from any service-wide
error rate.

**Useful side effect: it is a reliable restart oracle.** Because that poll is
fixed at 5.00 seconds, **8 or more `/health/v2-authority` spans inside one
minute means `gw_restart.sh` ran.** This deployment emits no deploy event, so
this is the most direct first-hand evidence of a rollout available.

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
over the trailing 7 days (as of 2026-09-02) the gateway's spans are
401,963 `Unset` (11,689 of them 4xx), 5,433 `Error`, and 0 `Ok` out of 407,396
total. The `Error` count has more than doubled since 2026-08-29 (2,003); see
the `/_unmatched` and `/health/v2-authority` notes above before reading that as
a regression.

**A route label does not mean the gateway implements that method.** The span
name is `<METHOD> <route template>`, and the template is resolved from the
router *after* routing has run — which a method mismatch still reaches. A
request to a path that exists but not for that verb gets Starlette's 405 and
still emits a span carrying the real template, so `POST /some/route` in the
span stream is not evidence that a POST handler exists. The fail-closed
validator does not catch this either: its allowlist is built from
`route.path` for every entry in `app.routes` and carries no methods at all.
Over the trailing 7 days (as of 2026-09-02) there are 22 such spans —
`POST /` (19), `OPTIONS /` (2) and one `GET /research-lab/loop-diagnostics` —
and because span status marks 5xx only, every one of them scores as a success
in any status-based panel. Read the `http.response.status_code` attribute before
concluding a route serves a method.

**A 500 may be a client disconnect that never reached a handler.** The
middleware initialises `status_code = 500` and only overwrites it when it
observes an `http.response.start` message. A client that disconnects before
the response starts is therefore recorded as a server error that never
happened. The store is consistent with this: over the trailing 7 days (as of
2026-09-02) **5,431 of the 5,433 5xx spans are 503**, and the only two 500s are
both on `/fulfillment/results/{request_id}` with durations of 8.0s and 10.3s —
long enough to be a caller timing out rather than a handler failing. Two spans is
thin evidence on its own; the code path is what makes it worth knowing. Treat
an isolated long-duration 500 as an unproven server error until corroborated.

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

**As of 2026-09-02 no host metrics are arriving**: the collector's metrics and
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
window on 2026-09-02. Every rung of the usual deploy-signal ladder
(`deployment_status`, commit `status`, `workflow_job`, `workflow_run`) is
unreachable here, so no deploy-anchored alerting, and no post-deploy soak, can
be built.

**Where the repo-activity stream lives.** In `otel.logs`, not `otel.spans` —
`scope_name = 'onepatch.github'` (689 rows over 30 days as of 2026-09-02;
`otel.spans` holds
zero). Querying spans for it returns an empty result that looks like "no
pushes".

**The `onepatch_source` tag changed mid-history.** Repo-activity rows landed
with `onepatch_source = ''` until **2026-08-25 20:26:47 UTC** and with
`onepatch_source = 'github'` from **2026-08-26 05:40:11 UTC** onward. A query
filtering on `onepatch_source = 'github'` silently drops the 423 rows before
that switchover and returns a truncated history with no error. Filter on
`scope_name = 'onepatch.github'` instead, which is stable across the boundary.

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
