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

## What a span contains — and all it can ever contain

| attribute | example |
|---|---|
| `http.request.method` | `GET` |
| `http.route` | `/research-lab/allocations/attested/{epoch}` (route *template* — never the concrete path) |
| `http.response.status_code` | `200` |
| `duration_ms` | `12.4` |

No bodies, query strings, headers, DB statements, model I/O, prompts, or
completions.

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
depends on it. As of 2026-08-17 no gateway metric point has arrived, which
means the collector is not running on the hosts — expect `hostmetrics`
receiver names (`system.cpu.*`, `system.memory.*`, `system.disk.*`,
`system.network.*`, `system.paging.*`) once it is installed.

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
  `/research-lab/public/loops`, `/research-lab/public/loops/{ticket_id}`,
  `/research-lab/public/loops/summary`,
  `/research-lab/public/topic-groups`,
  `/research-lab/benchmarks/public/latest`,
  `/research-lab/benchmarks/public/{benchmark_date}`,
  `/research-lab/allocations/attested/{epoch}`,
  `/research-lab/allocations/live/{epoch}`, `/research-lab/openrouter-keys`,
  `/research-lab/openrouter-keys/credential-recipient`,
  `/research-lab/tickets`, `/research-lab/loop-start`,
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

Every route above is registered in code. A handful are registered but carried
no traffic in the trailing 7 days — `/fulfillment/excluded-now/{request_id}`,
`/weights/submit/v2`, `/weights/finalize/v2`,
`/weights/subnet-epoch/boundary/v1`,
`/qualification/model/rate-limit/{miner_hotkey}`, `/epoch/{epoch_id}/leads`
and `/metrics` — so absence of spans on those is quiet traffic, not missing
instrumentation. Regenerate this section from live telemetry (server/client
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
