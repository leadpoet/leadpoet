# Gateway infra-only telemetry

Opt-in OpenTelemetry for the gateway HOST process. One server span per HTTP
request; nothing else. See `gateway/observability/otel_bootstrap.py` for the
full contract.

## Facts

- `service_name` — one value: `leadpoet-gateway`. Nothing else in this repo
  emits OTel. (verified 2026-08-17)
- `env` — empty on every signal. The resource is a fixed dict with no
  `deployment.environment*` key, so there is no environment to filter on.
  (verified 2026-08-17)
- `scope_name` — always `gateway.http`. (verified 2026-08-17)
- Span `name` — exactly `<METHOD> <route-template>`. Filter on `name`, not on
  the `http.route` attribute: the attribute read costs ~16× the bytes for the
  same answer. (verified 2026-08-17)
- `kind` — always `2` (SERVER), and every span is a root
  (`parent_span_id = ''`). There are no client spans. (verified 2026-08-17)
- `status_code` — `2` on 5xx only. A 4xx leaves it `0` (Unset), so count 4xx
  from the `http.response.status_code` attribute, not from span status.
  (verified 2026-08-17)
- Signals present — spans only. Zero metric points, zero histogram points and
  zero log records arrive from any service in this repo. (verified 2026-08-17)

## What reads as healthy but isn't

Because span status is set on 5xx only (see Facts), a route can be answering
almost nothing but 4xx and still read as **0% error** on any status-based
dashboard. That is not hypothetical here — it is most of the gateway's error
volume. In the trailing 7 days ending 2026-08-19 the gateway served **46,461
4xx responses**, and two routes account for 45,625 of them:

| route | 4xx | of total | note |
|---|---|---|---|
| `GET /weights/v2/published/{netuid}/{epoch_id}` | 22,728 / 22,728 | **100.0% 404** | validators polling for an epoch that is not published yet |
| `GET /weights/v2/published-compact/{netuid}/{epoch_id}` | 22,897 / 23,258 | **98.4% 404** | same callers, compact variant |

Together that is **8.4% of all gateway traffic**, and it is the expected steady
state, not an incident: the poll-until-published pattern is how those callers
are written. Do not treat a drop in these 404s as a fix, or a rise as a
regression, without checking epoch publication first.

Other routes whose real behaviour is invisible to span status:

- `POST /research-lab/openrouter-keys` — 96.6% 400 (197/204).
- `GET /research-lab/engine/issues`,
  `GET /research-lab/reports/daily-noise-budget/latest`,
  `GET /research-lab/reports/candidate-generation-failures` — 100% 401 (all
  low volume: 4, 4 and 2 requests).

So: **count 4xx from the `http.response.status_code` attribute**, and read any
"error rate" built on span status as *5xx rate*, nothing more.

## What a span contains — and all it can ever contain

| attribute | example |
|---|---|
| `http.request.method` | `GET` |
| `http.route` | `/research-lab/allocations/attested/{epoch}` (route *template* — never the concrete path) |
| `http.response.status_code` | `200` |
| `duration_ms` | `12.4` |

No bodies, query strings, headers, DB statements, model I/O, prompts, or
completions.

**`duration_ms` is the attribute name, not the column name.** The middleware
sets `duration_ms` as a float count of milliseconds
(`otel_bootstrap.py`, `span.set_attribute("duration_ms", …)`), and it does
arrive in the store as a span attribute. But the span's *duration column* is
`duration_ns` — Int64 nanoseconds, derived by the store from the span's start
and end timestamps. Neither layer renames the other; they are two different
things that happen to measure the same interval. Query `duration_ns` (and
divide by `1e6` for milliseconds): it is a real column, it is cheap, and it is
what percentile helpers read. A query written against a `duration_ms` *column*
matches nothing.

**Suppression is three exact paths, not a family.** `_SUPPRESSED_ROUTES` in
`otel_bootstrap.py` is `{"/health", "/health/live", "/health/ready"}`, matched
against the concrete request path. Every other health or liveness route is
traced normally — `/health/v2-authority`, `/attest/health` and
`/attestation/health` all export spans and all appear in the store. Do not
read "health routes are suppressed" into a query; `/health/v2-authority` is
one of the busier routes on the gateway.

## Spans

One span per HTTP request, emitted by the pure-ASGI middleware in
`gateway/observability/otel_bootstrap.py`.

| span name | kind | emitted by | when | attributes |
|---|---|---|---|---|
| `<METHOD> <route-template>` | 2 (SERVER) | `otel_bootstrap.py` middleware | once per HTTP request that resolves to a registered route, apart from the three suppressed paths above | the four in the table above |
| `<METHOD> /_unmatched` | 2 (SERVER) | same | a request whose path matches no registered route; the concrete path is never exported | same |

## Metrics

None reach the store. Gateway host CPU / memory / disk / network are available
from a **separate** OpenTelemetry Collector process — config, installer and
systemd unit all shipped, documented in
[`gateway/observability/HOST_METRICS.md`](gateway/observability/HOST_METRICS.md).
Installing it is deliberately an explicit, separately authorized host
operation and is *not* wired into `gw_restart.sh`, so a gateway restart never
depends on it (`tests/test_gateway_hostmetrics_collector.py` asserts the
restart script never references the installer — the only place outside
`gateway/observability/` that names it at all). The state as of **2026-08-19**
is therefore: shipped, deliberately not wired into the restart path, and **not
installed on any host** — zero `system.*` points have ever arrived. Expect
`hostmetrics` receiver names (`system.cpu.*`, `system.memory.*`,
`system.disk.*`, `system.network.*`, `system.paging.*`) once it is installed.

## Logs

None reach the store. The gateway ships no OTLP log exporter, and error
capture goes to Sentry instead (see below) — a different destination, so
Sentry events are not queryable alongside these spans.

## Deploy signal

**Nothing in the store marks a deploy, and as of 2026-08-18 no signal is
pinned.** This section records why, so the question does not get re-asked.

**What ships the gateway.** Production rollout is a host operation:
`gw_restart.sh` (and `validator_restart.sh` for validators), run on the box
against the exact-commit channel. No GitHub workflow performs a rollout —
`Attested V2 Release` (push to `main`) builds the parent images and publishes
the immutable exact-commit release artifacts to S3; `Production Parity
Fast`/`Full` and `Production Parity Cleanup` validate an already-attested
commit; `Deploy Checks` is a gate. A green release build means an artifact
exists, not that a host is running it — so treating it as a deploy would be
wrong in exactly the expensive direction.

**GitHub Deployments exist, but only for staging and only briefly.** 39
deployments were created against the `physical-v2-staging` environment between
2026-08-13 and 2026-08-15, all by hand, all with an empty payload, and **every
one of them went `queued` → `error`**. None have been created since 2026-08-15.
That is the API a deploy signal would normally be pinned to, and here it
describes a three-day staging experiment that never succeeded, not the gateway
in production.

**Separately, the events never arrive.** The store has received **zero** CI/CD
event records for this repo, ever. Repository activity does land — pushes and
pull requests, under `scope_name = 'onepatch.github'` — but workflow runs,
workflow jobs, deployment statuses and commit statuses do not. So even the
staging signal above could not be pinned today; the evidence is not in the
store to pin against. This is an ingest-side gap on OnePatch's end, not a
change this repo needs.

**Consequence, and the one thing that reads as a deploy.** Automated
change-verification soaks cannot start: they wait for a successful deploy that
is never observed, so a merged pull request is never confirmed against
production telemetry. The only rollout-shaped event visible in this repo's own
telemetry is the **gateway restart signature** — a release-verification poll
sweep (`GET /build-info`, `GET /weights/v2/release-evidence/{commit_sha}`,
`GET /health/v2-authority` in lockstep), then a total span gap across every
route, then a short fail-closed 503 burst on `/health/v2-authority` and
`/_unmatched` as the worker authority comes back up. That is a proxy for "the
gateway process was replaced", not for "this commit shipped" — it carries no
commit sha — and a clean restart can be short enough to miss. Useful for
reading history; too weak to gate a soak on.

**A fix is proposed.** A separate OnePatch pull request (branch
`op/deploy-beacon`, being opened alongside this one) appends a bounded,
fail-open beacon to the end of `gw_restart.sh` so the deploy path announces the
commit sha it just brought up. Until that merges, the restart signature above
remains the only telemetry-visible proxy for a rollout — and it is now well
enough characterised to use deliberately. On the way back up the gateway
briefly returns 5xx on `GET /health/v2-authority` and `GET|POST /_unmatched`
*together*, in a burst lasting a few minutes; that co-occurrence is what
separates a restart from a route-specific regression. Restarts observed in the
7 days ending 2026-08-19, all UTC: **2026-08-16 10:45, 2026-08-18 02:20,
2026-08-18 03:35, 2026-08-19 02:20** — roughly daily, with the 02:20 slot
twice. What the signature still cannot tell you is *which commit* came up,
which is exactly the gap the beacon closes.

## Service map

One service reports telemetry: **`leadpoet-gateway`**. By design it emits a
single `SERVER` span per HTTP request and nothing else — so it makes no
instrumented outgoing (client) calls, and it reports **no environment**
(the resource is a fixed attribute dict with a constant service name, never
an env value).

### leadpoet-gateway

- **Environments:** none reported (`env` is empty).
- **Outgoing (client, `kind = 3`):** none — server spans only.
- **Incoming (server, `kind = 2`):** the route families below. Span name is
  `<METHOD> <route-template>`.

- **Fulfillment** — `/fulfillment/results/{request_id}`,
  `/fulfillment/requests/active`, `/fulfillment/scoring`,
  `/fulfillment/leaderboard`, `/fulfillment/rewards/active`,
  `/fulfillment/banned-hotkeys`, `/fulfillment/excluded-now/{request_id}`.
  (`/fulfillment/scoring` is the fulfillment scorer — it is not the Research
  Lab scorer, which runs headless and emits no spans at all.)
- **Research Lab** — `/research-lab/source-adapters`, `/research-lab/status`,
  `/research-lab/public/loops/{ticket_id}`,
  `/research-lab/benchmarks/public/latest`,
  `/research-lab/benchmarks/public/{benchmark_date}`,
  `/research-lab/allocations/attested/{epoch}`,
  `/research-lab/allocations/live/{epoch}`, `/research-lab/openrouter-keys`,
  `/research-lab/openrouter-keys/credential-recipient`,
  `/research-lab/tickets`, `/research-lab/loop-start`,
  `/research-lab/loop-diagnostics` (POST only — 6 requests in the trailing 7
  days. A `GET` against the same path also appears in the store, returning
  405; that is a caller using the wrong method, not a second route),
  `/research-lab/reports/daily-noise-budget/latest`,
  `/research-lab/reports/candidate-generation-failures`,
  `/research-lab/engine/issues`.
- **Weights (v2)** — `/weights/v2/published/{netuid}/{epoch_id}`,
  `/weights/v2/published-compact/{netuid}/{epoch_id}`,
  `/weights/v2/release-evidence/{commit_sha}`,
  `/weights/v2/latest/{netuid}/{epoch_id}`, `/weights/current/{netuid}`,
  `/weights/latest/{netuid}/{epoch_id}`, `/weights/inputs/v2`,
  `/weights/submit/v2`, `/weights/submit/compact/v2`,
  `/weights/finalize/v2`, `/weights/finalize/compact/v2`,
  `/weights/subnet-epoch/boundary/v1`.
- **Attestation** — `/attestation/deploy-readiness`,
  `/attestation/document`, `/attestation/health`, `/attest`,
  `/attest/health`.
- **Qualification** — `/qualification/model/presign`,
  `/qualification/leaderboard`, `/qualification/champion`,
  `/qualification/model/rate-limit/{miner_hotkey}`.
- **Epoch** — `/epoch/state`, `/epoch/current`, `/epoch/{epoch_id}/leads`.
- **Meta / root** — `/`, `/build-info`, `/health/v2-authority`, `/metrics`.
- **Unresolved** — any path that matches no registered route exports the
  fixed `/_unmatched` template (never the concrete path). Both
  `GET /_unmatched` and `POST /_unmatched` occur routinely.

- **Registered, and effectively dead** — three public Research Lab reads are
  registered and still resolve, but stopped carrying meaningful traffic on
  **2026-08-12** and belong here rather than in the live list above:
  `GET /research-lab/public/loops` ran at ~84 req/hr through 2026-08-11
  (2515 / 2261 / 1253 requests on 08-09 / 08-10 / 08-11), fell to 7
  requests on 08-12, and has served ≤4/day since — **18 requests in the
  trailing 7 days, and zero on 08-14, 08-18 and 08-19**.
  `GET /research-lab/public/loops/summary` (13) and
  `GET /research-lab/public/topic-groups` (4) are in the same near-dead band.
  The date matters: their silence is a caller that stopped calling on
  2026-08-12, so do **not** read it as a new outage.

Every route above is registered in code. A handful are registered but carried
no traffic in the trailing 7 days — `/fulfillment/excluded-now/{request_id}`,
`/weights/submit/v2`, `/weights/finalize/v2`,
`/weights/subnet-epoch/boundary/v1`,
`/qualification/model/rate-limit/{miner_hotkey}` and `/epoch/{epoch_id}/leads`
— so absence of spans on those is quiet traffic, not missing
instrumentation. (`/metrics` is no longer one of them: it carried a single
span, on 2026-08-19.)

One more caller-side shift worth recording so it is not re-investigated:
`GET /research-lab/status` fell sharply over the same period — 189.8 req/hr in
the preceding 7 days down to **61.4 req/hr** in the trailing 7 — with latency
unchanged. Fewer callers, not a slower gateway.

Regenerate this section from live telemetry (server/client
span names by `service_name`) whenever the doc is touched.

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
benchmarks, and contact data out of every event. It has grown past the
bootstrap: `sentry_operations.py` emits the bounded restart and release
summaries (stdlib-only, and its payloads deliberately exclude `icp_score` and
every other private lead signal), `sentry_scrubbing.py` holds the redaction
rules, `sentry_cli.py` is the one-shot bridge host shell workflows call, and
`host_runtime.py` builds the small hash-locked interpreter that bridge runs in
so the validator's authoritative interpreter is never touched. See
[`docs/sentry_error_monitoring.md`](docs/sentry_error_monitoring.md);
`tests/test_sentry_boundary_guard.py` and `tests/test_sentry_operations.py`
enforce the boundary in CI. These events go to Sentry, not to the OTel
store — they are not queryable alongside the spans above.
