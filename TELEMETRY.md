# Infra-only telemetry

Opt-in OpenTelemetry for two host processes — the gateway
(`leadpoet-gateway`) and the Lab Arena sidecar (`leadpoet-arena`). Both run
through one fail-closed exporter in
`gateway/observability/otel_bootstrap.py`, which holds the full contract.

## What a span contains — and all it can ever contain

| attribute | example |
|---|---|
| `http.request.method` | `GET` |
| `http.route` | `/arena/v1/chain-outcomes/{epoch}` (route *template* — never the concrete path) |
| `http.response.status_code` | `200` |
| `duration_ms` | `12.4` |

No bodies, query strings, headers, DB statements, model I/O, prompts, or
completions. Health/liveness routes are suppressed entirely.

## Arena sidecar (`leadpoet-arena`)

The gateway proxies every Arena call through a single catch-all route
template, `/arena/{arena_path:path}`, so from gateway telemetry alone the
operation that was invoked is not recoverable — a weight-state read, a run
claim and a scoring completion are one indistinguishable label. The sidecar
therefore emits its own spans, on the same envelope, under five scopes.

**`arena.http`** — one SERVER span per sidecar request, carrying exactly the
same four attributes as above. The difference is the route: `http.route` is
the sidecar's own template (`/arena/v1/runs/{run_id}/complete`,
`/arena/v1/submissions/{submission_id}/finalize`, …), so each Arena
operation gets its own rate, error rate and latency. Still a template — run
ids, submission ids, hotkeys and epochs never appear. It carries one extra
attribute the gateway envelope does not have:

| attribute | example |
|---|---|
| `arena.denial` | `lease_token_invalid` (`-` when the request was not refused) |

A refused Arena call is otherwise an anonymous 4xx. `arena.denial` names it
with the refusal code the sidecar returned. `ServiceError.code` can
interpolate an exception message after a colon (`"contract:%s"`), so only
the literal prefix before the first colon is ever exported, and only when it
matches a strict shape.

**`arena.task`** — one INTERNAL span per pipeline stage. The Arena does its
real work off the request path (a once-a-minute driver tick plus the code
review worker), so without this a wedged pipeline and an idle one look
identical from outside. Its attribute set is a separate, equally exact
allowlist:

| attribute | example |
|---|---|
| `arena.stage` | `advance_round` (from a frozen vocabulary; see `ARENA_TASK_STAGES`) |
| `arena.outcome` | `ok` \| `idle` \| `failed` |
| `arena.error_type` | `ArenaStoreUnavailable` (exception CLASS only, shape-checked; `-` when none) |
| `arena.count` | `3` (an operational magnitude — baselines promoted, rounds active) |
| `duration_ms` | `412.0` |

Stages: `driver_tick`, `promote_baselines`, `active_rounds`,
`advance_round`, `ensure_daily_round`, `activate_rewards`,
`reconcile_provider_costs`, `review_submissions`. A stage name or outcome
outside the vocabulary drops the span whole, so no round id, submission id,
hotkey, source path, prompt, score, or model output can ride out on a label.

**`arena.provider`** — one INTERNAL span per outbound provider call. The
Arena's cost, latency and failure all originate upstream at OpenRouter,
Deepline and Scrapingdog, and none of it was visible: a slow round and a
throttled provider looked the same.

| attribute | example |
|---|---|
| `arena.provider` | `openrouter` (`ARENA_PROVIDERS`; `unknown` when the operation is not in the table) |
| `arena.operation` | `openrouter.responses` (`<provider>.<operation>`; must agree with `arena.provider`, else `unknown`) |
| `arena.outcome` | `ok` \| `refused` \| `uncertain` \| `failed` |
| `arena.error_code` | `budget_refused` (`ARENA_PROVIDER_ERROR_CODES`; `-` when none) |
| `arena.error_type` | `HTTPError` (exception CLASS only; `-` when none) |
| `arena.http_status` | `200` (what the worker was handed, 0–999) |
| `arena.provider_status` | `429` (what the provider answered, 0–999) |
| `arena.attempts` | `2` (champion credential attempts, 1–4) |
| `arena.cost_microusd` | `1375` (settled ledger cost, 0–1e9) |
| `duration_ms` | `842.0` |

Span name is `arena.provider.<provider>`. `uncertain` means the ledger could
not be settled and the cost is reconciled later; `failed` means the broker
raised rather than returning a result. The projection is the same one the
repo already trusts for private diagnostics
(`broker._provider_attempt_summary`): vocabulary members and bounded
integers, never a call identity, response hash, credential fingerprint,
model id, prompt, URL, or body.

**`arena.run`** — one INTERNAL span per finished evaluation run, recorded
when the attempt commits.

| attribute | example |
|---|---|
| `arena.run_kind` | `execute` \| `score` |
| `arena.terminal_cause` | `budget_exhausted` (all thirteen of `contracts.TERMINAL_CAUSES`) |
| `arena.outcome` | `ok` when the cause is `accepted`, else `failed` |

Span name is `arena.run.<kind>`. There is deliberately no duration: the run
row carries no start time, so any duration here would be invented rather
than measured.

**`arena.gate`** — one INTERNAL span per *contended* wait on the shared
OpenRouter concurrency gate. A call admitted with no wait emits nothing, so
this signal scales with the problem rather than with traffic.

| attribute | example |
|---|---|
| `arena.gate_outcome` | `admitted_after_wait` \| `timed_out` \| `cancelled` \| `no_capacity` |
| `duration_ms` | `2400.0` (how long the call actually waited) |

`lab_arena/telemetry.py` is the seam the pipeline calls. It owns no exporter
and reads no destination; with no recorder installed every call is a no-op,
so the Arena behaves identically whether or not telemetry is configured.

## Enabling (gateway host only)

```bash
export GATEWAY_OTEL_ENABLED=1
export GATEWAY_OTEL_ENDPOINT="https://<collector>/v1/traces"
export GATEWAY_OTEL_TOKEN="<token>"            # REQUIRED; sent as Authorization: Bearer
```

The Arena sidecar reads the SAME three values from the same protected env
file (`scripts/run_lab_arena_service.py --environment-file`), through the
enumerating reader in `gateway/observability/read_gateway_otel_env.py` that
never executes the file. Arena telemetry therefore needs no new secret and
no new host configuration — it follows the gateway's switch. The endpoint
and token are a write-only ingest pair; they can append spans and do nothing
else.

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
   must have an ACCEPTED instrumentation scope for its process — a span
   whose scope is not listed is dropped, so one process can never emit
   through another's envelope. A gateway span must have the `gateway.http`
   scope with no
   version/schema metadata, `SERVER` kind, a root context (no parent, empty
   trace state), exactly the four attributes above with the approved types,
   a standard HTTP method, a span name equal to exactly `<method> <route>`,
   no events/links/status descriptions, the fixed resource, and a
   registered route template (or the literal `/_unmatched`). A span
   violating any of that is dropped entirely — never mutated, never
   partially exported — and counted in a warning that omits the rejected
   values. An `arena.http` span is held to the identical rule against the
   sidecar's own routes; an `arena.task` span must be INTERNAL, root, and
   carry exactly the five stage attributes with a stage name and outcome
   from their frozen vocabularies, a shape-checked exception class name,
   and a bounded count. An `arena.provider`, `arena.run` or `arena.gate`
   span is held to the same rule against its own table: every exported
   string must be a member of a frozen literal vocabulary and every integer
   inside its stated bound, or the span is dropped whole. The vocabularies
   are copies of `lab_arena` constants (the bootstrap imports no
   `lab_arena`), and CI compares the copies against the originals so a new
   provider or terminal cause cannot silently start dropping spans.
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
- Host processes only — the gateway and the Arena sidecar, never the
  attested enclaves; no new dependencies, so no PCR0 change.
- Distinct service identities (`leadpoet-gateway`, `leadpoet-arena`), each a
  fixed constant, so an Arena reading is never attributed to the gateway.
- The Arena pipeline seam observes but never swallows: a stage span records
  the failure and re-raises the original exception unchanged.

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
