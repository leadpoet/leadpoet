# Sentry error monitoring (opt-in, fail-closed)

One integration covers every Leadpoet HOST runtime. It exists to answer
"what is breaking and where" — crashes, unhandled thread/task failures, and
ERROR-level logs — while guaranteeing that trajectory/training data,
prompts, benchmarks, model internals, and contact data never leave the
process. It follows the same philosophy as the OTel bootstrap in
`TELEMETRY.md`: namespaced env gate, explicit options, swallowed wiring
failures, code-enforced boundary with a CI guard.

## Coverage map

`init_sentry(component=...)` is wired at every host entry point:

| component | entry point |
|---|---|
| `gateway` | `gateway/main.py` (host FastAPI process; also covers in-process fulfillment/qualification/TEE-host code) |
| `validator` | `neurons/validator.py` (all container modes; mode is tagged) |
| `miner` | `neurons/miner.py` |
| `auditor-validator` | `neurons/auditor_validator.py` |
| `research-lab-worker` | `gateway/research_lab/worker_process.py` (hosted + scoring kinds, kind tagged) |
| `research-lab-scoring-worker` | `scripts/run_research_lab_scoring_worker.py` |
| `research-lab-scoring-worker-fleet` | `scripts/run_research_lab_scoring_worker_fleet.py` |

Coverage inside a process is automatic, not per-module: the SDK's
excepthook (crashes), threading hook (thread crashes), and stdlib logging
integration (any `logging` / `bt.logging` / `uvicorn.error` record at
ERROR or above becomes an event; asyncio's default handler logs unretrieved
task exceptions at ERROR) reach every package in the codebase. One-off
operator scripts can opt in with the same two lines wrapped in
`try/except`.

**Never wired**: the attested enclaves (`validator_tee/enclave/*`,
`gateway/tee` enclave service). Enclave images are measured (PCR0) and
have no general egress. `tests/test_sentry_boundary_guard.py` fails CI if
Sentry references reach an enclave surface or enclave requirements file.

## Enabling

```bash
export LEADPOET_SENTRY_ENABLED=1
export LEADPOET_SENTRY_DSN="https://<key>@<org>.ingest.sentry.io/<project>"
# optional
export LEADPOET_SENTRY_ENVIRONMENT=production      # default: production
export LEADPOET_SENTRY_RELEASE=<git sha>           # default: exact deploy SHA, then git HEAD
export LEADPOET_SENTRY_EXTRA_PROTECTED_MODULES=    # comma-separated prefixes; widens redaction only
export LEADPOET_SENTRY_MESSAGE_MODE=scrub          # or redact-all
```

Both gate variables are required; anything less is a complete no-op.
Ambient `SENTRY_*` variables are never read — every option is passed to
the SDK explicitly. `sentry-sdk` itself is optional at runtime: when the
package is missing the wiring logs one line and stays off, so no runtime
grows a hard dependency.

## What an event may contain — and all it may ever contain

- exception **type**, exception **module**, stack **file/function/line**
- logger name, level, timestamp, `leadpoet.component` (+ low-cardinality
  tags such as validator mode / worker kind), environment, release SHA,
  server name
- for non-protected surfaces only: the exception/log **message** after
  regex scrubbing (emails, formatted phone numbers, secret-shaped values,
  URL query strings removed; 500-char cap). UUID/sha256 join keys survive
  verbatim so events stay correlatable with `execution_trace:` /
  `cost_ledger:` refs.

## What can never leave the process

Enforced by `leadpoet_observability/sentry_scrubbing.py` (fail closed: an
event that cannot be fully scrubbed is dropped, never sent partially):

1. **Unknown top-level fields** — dropped against a pinned, error-only event
   allowlist so future/custom SDK payload shapes cannot silently expand export.

2. **Stack local variables** — `include_local_variables=False` at init, and
   any residual `vars` deleted per frame. Locals are the main vector for
   prompt/lead/trajectory payload leakage.
3. **Source context lines** — stripped from every frame. An error inside
   LLM-generated candidate code or a private model artifact must not export
   source text.
4. **Messages from protected surfaces** — events whose logger, exception
   module, or ANY stack frame touches `research_lab`,
   `gateway.research_lab`, `gateway.fulfillment`, `gateway.qualification`,
   `qualification`, `leadpoet_verifier`, `miner_models`,
   `validator_models`, lead-processing code, host model/provider/scoring
   executors, the lead pool/queue utils, or the `langfuse` /
   `openai` / `anthropic` / `openrouter` / `firecrawl` clients keep type,
   stack, and logger but have every message replaced with
   `[leadpoet-redacted:protected-surface]`. One protected link redacts the
   whole exception chain (a wrapper message can embed the inner error).
   Breadcrumb capture is disabled outright.
5. **Request/user envelopes, cookies, argv, request bodies** — dropped;
   `send_default_pii=False`, `max_request_body_size="never"`.
6. **Performance data** — no tracing, no profiling, no sessions; errors
   only. Framework/DB/HTTP auto-instrumentation stays disabled so query
   and payload data never becomes event context.
7. **Secrets** — key-based redaction (`api_key`, `token`, `authorization`,
   `service_role`, …) plus value-shape redaction (`sk-*`, AKIA keys, JWTs,
   bearer tokens) plus the marker vocabulary shared with
   `research_lab/observability/redaction.py` (`judge_prompt`,
   `hidden_benchmark`, `icp_plaintext`, …). Dynamic mapping keys, wallet
   names/addresses, hotkeys, coldkeys, seed phrases, and mnemonics are also
   removed.

Repeated log records are coalesced before event construction, and all events
with the same scrubbed type, message shape, and stack are coalesced in each
process for five minutes and assigned the same safe Sentry fingerprint across
processes. The first event is always sent, distinct failure signatures remain
independent, and each cache is bounded to 1,024 signatures.

`LEADPOET_SENTRY_MESSAGE_MODE=redact-all` redacts every message and drops
every breadcrumb regardless of surface, for maximum privacy at the cost of
triage detail. The environment can only widen redaction, never narrow it.

## The boundary is code-enforced

`tests/test_sentry_boundary_guard.py` fails CI when:

- `sentry_sdk` is imported or initialized anywhere outside
  `leadpoet_observability/sentry_bootstrap.py`;
- an enclave requirements file (`validator_tee/enclave/requirements.txt`,
  `gateway/tee/requirements*.{txt,in,lock}`) gains `sentry-sdk`, or any
  enclave/TEE module references the bootstrap;
- a non-namespaced `SENTRY_*` variable is referenced anywhere;
- the hard-off init options (`include_local_variables=False`,
  `send_default_pii=False`, `max_request_body_size="never"`,
  `traces_sample_rate=None`, `auto_enabling_integrations=False`, the
  scrubbing hooks) drift out of the bootstrap;
- the scrubber stops stripping source context/locals or dropping the
  request envelope and argv;
- a wired entry point loses its `init_sentry(` call.

## Operational notes

- **Activation is a release decision.** `sentry-sdk` is in
  `requirements.txt`/`setup.py`, so first activation on gateway/validator
  hosts changes installed dependency sets and follows the normal V2
  release/rehearsal process. The code paths themselves are inert no-ops
  until the env gate is set, so shipping them changes no behavior.
- The tested SDK version is pinned (`sentry-sdk==2.66.1`) so a dependency
  update cannot silently change integrations or payload behavior.
- The DSN is not committed anywhere; it lives in the operator env files
  (e.g. the gateway env secret). Rotate it like any ingest token, and
  agree the vendor's retention / no-training / deletion terms in writing —
  the same caveat as `TELEMETRY.md`.
- Wiring failures never break a runtime; look for
  `leadpoet_sentry_wiring_skipped` / `leadpoet_sentry_init_skipped` /
  `leadpoet_sentry_scrub_failed` lines (exception class names only) in
  process logs.
- Public auditors and external miners simply leave the gate unset; nothing
  initializes and nothing is sent.
