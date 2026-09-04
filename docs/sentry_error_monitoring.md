# Sentry host observability (opt-in and non-authoritative)

One integration covers Leadpoet **host** runtimes. It captures terminal errors
at 100% and a conservative sample of explicitly named operational stage
transactions. It does not auto-instrument HTTP, databases, chain clients, or
frameworks, and it never runs inside Nitro enclaves.

Sentry is diagnostic only. Missing configuration, a missing SDK, collector
latency/outage, scrubber failure, or any SDK exception must not change startup,
restart, scoring, attestation, weight submission, finalization, or cleanup.
Export fails closed by dropping an event that cannot be scrubbed; application
behavior fails open by preserving the original result or exception.

The validator restart controller prepares a hash-locked, host-only telemetry
environment before production shutdown when its authoritative Python does not
already contain the SDK. The environment is cached and used only to emit the
bounded restart summary; it never replaces `VALIDATOR_PYTHON_BIN` or enters a
validator, worker, auditor, relay, or enclave authority path. Preparation is
bounded and fail-open, so telemetry installation cannot block a restart.

The restart and canonical-weight event matrix is in
[`sentry_restart_weight_instrumentation.md`](sentry_restart_weight_instrumentation.md).

## Coverage

`init_sentry(component=...)` is wired at these host entry points:

| Component | Entry point |
|---|---|
| Gateway | `gateway/main.py` |
| Primary validator | `neurons/validator.py` |
| Auditor validator | `neurons/auditor_validator.py` |
| Miner | `neurons/miner.py` |
| Release/PCR0 host tools | `validator_tee/host/gateway_pcr0_builder.py`, `validator_tee/host/runtime_v2_bootstrap.py`, `validator_tee/host/verify_release_gate_v2.py` |
| Restart controllers | `gw_restart.sh`, `validator_restart.sh` through the bounded `sentry_cli` bridge |
| Attested release | `.github/workflows/attested-v2-release.yml` through the same release-summary bridge |

The host-side vsock and chain relay boundaries report sanitized state through
the initialized parent process. `validator_tee/enclave/*` and the measured
gateway enclave packages never import Sentry and never gain network egress.
`tests/test_sentry_boundary_guard.py` enforces this boundary.

## Configuration

```bash
export LEADPOET_SENTRY_ENABLED=1
export LEADPOET_SENTRY_DSN="https://<key>@<org>.ingest.sentry.io/<project>"

# Optional
export LEADPOET_SENTRY_ENVIRONMENT=production
export LEADPOET_SENTRY_RELEASE=<exact-40-character-git-sha>
export LEADPOET_SENTRY_TRACES_SAMPLE_RATE=0.01
export LEADPOET_SENTRY_EXTRA_PROTECTED_MODULES=
export LEADPOET_SENTRY_MESSAGE_MODE=scrub  # or redact-all
export LEADPOET_SENTRY_RESTART_STAGE_DEADLINE_SECONDS=900
```

Both `LEADPOET_SENTRY_ENABLED` and `LEADPOET_SENTRY_DSN` are required.
Anything less is a complete no-op. Ambient `SENTRY_*` variables are ignored;
all SDK options are explicit. Successful manual traces default to 1% and are
clamped to 10%. Terminal errors are not sampled. The SDK shutdown flush is
bounded to one second.

GitHub release jobs use the same namespaced settings, with the DSN supplied by
the `LEADPOET_SENTRY_DSN` Actions secret. Every reporting step uses
`if: always()` and `continue-on-error: true`.

## Read-only Codex API access

`LEADPOET_SENTRY_API_TOKEN` is an operator-only read credential stored in both
`leadpoet/prod/gateway/env` and `leadpoet/prod/validator/env`. It is separate
from the ingestion DSN. Restart hydration removes it from cached environment
files and runtime exports, so it does not enter gateway, validator, auditor,
worker, relay, container, enclave, attestation, or weight paths.

The standard-library helper retrieves only the API token, DSN, and optional
project identifiers over read-only SSH; the token remains in process memory.
It sends a bounded Bearer-authenticated request to the Sentry API and emits an
allowlisted, re-scrubbed response. It never prints or persists the token and
does not support raw event bodies or a raw-token output mode.

```bash
cd /Users/pranav/Downloads/Election_Analysis/Bittensor-subnet

# Securely read and validate the gateway copy without displaying it.
python3 scripts/query_sentry_api.py auth-check --secret-source gateway

# Bounded, redacted recent incident views.
python3 scripts/query_sentry_api.py issues --secret-source gateway \
  --stats-period 24h --limit 25
python3 scripts/query_sentry_api.py events --secret-source gateway \
  --stats-period 24h --limit 25

# Read-only fallback when the gateway host or secret is unavailable.
python3 scripts/query_sentry_api.py issues --secret-source validator \
  --stats-period 24h --limit 25
```

Codex may use this workflow for deployment checks, restart monitoring,
debugging, and post-deploy validation. Sentry results must be correlated with
gateway/validator logs and durable or on-chain evidence. They are observability
only and cannot replace exact-commit, attestation, PCR0, canonical-bundle,
finalization, `LastUpdate`, or vector-readback checks.

Never display the token through `aws`, `ssh`, `env`, `printenv`, `set -x`, an
inline command argument, verbose HTTP output, chat, logs, commits, or tests.

An operator configures or rotates both secret copies with one hidden prompt:

```bash
cd /Users/pranav/Downloads/Election_Analysis/Bittensor-subnet
python3 scripts/configure_sentry_api_token.py
```

The utility preserves each document's existing format, creates a new immutable
Secrets Manager version through the host's narrowly scoped AWS role, verifies
that exact version is `AWSCURRENT`, retains the prior version for rollback, and
performs a constant-time readback comparison. It never places the token in a
command argument or local file. No gateway or validator restart is required
for `query_sentry_api.py`, because that helper reads Secrets Manager directly.

## Allowed data

Only allowlisted, bounded operational fields may be exported:

- exact public release SHA and sanitized component/role/stage/status;
- deterministic release/restart/weight correlation IDs;
- hashes for bundle, weights, snapshot, receipts, publication, extrinsic,
  vector, finalization, PCR0, boot identity, and manifests;
- netuid, epoch/block counters, UID, attempts, deadlines, durations, counts,
  byte limits, response class, HTTP/RPC/SQL error codes, and fail-closed state;
- exception type/module and stack file/function/line; infrastructure messages
  only after scrubbing.

No telemetry path performs an extra Supabase or chain read solely for Sentry.

## Data that cannot leave

`leadpoet_observability/sentry_scrubbing.py` enforces a strict top-level and
field allowlist. It removes or redacts:

1. Request/user envelopes, cookies, argv, request bodies, query strings, and
   unknown future SDK fields.
2. Stack locals and source context lines.
3. Provider/customer payloads, prompts, trajectories, benchmarks, candidate or
   private-model source, scoring inputs, and complete vectors/receipt graphs.
4. Credentials, tokens, API keys, service-role keys, AWS/SSH material, proxy
   credentials, private keys, seed phrases, nonces, full signatures, raw
   attestation documents, plaintext/ciphertext envelopes, and contact data.
5. Raw wallet/hotkey/coldkey identities; only non-reversible join hashes are
   accepted.

Messages touching protected model, provider, Research Lab, qualification,
fulfillment, lead-processing, or LLM-client surfaces are replaced wholesale by
`[leadpoet-redacted:protected-surface]`. `redact-all` applies that policy to
every message and drops breadcrumbs.

Manual transactions use static Leadpoet names and separately scrubbed span
data. Automatic trace propagation is disabled, so correlation metadata never
enters signed requests or trusted V2 artifacts.

## Event volume and grouping

- Retries are warning breadcrumbs, not error events.
- One terminal logical failure is emitted after retry exhaustion.
- A bounded process-local limiter suppresses duplicate wrapper/rethrow events.
- Semantic fingerprints use component + stage + stable failure code.
- Generic ERROR logging still captures unexpected failures, with a five-minute
  bounded coalescer for repeated message shapes.
- Up to 50 breadcrumbs are retained and unknown payload fields are dropped.

## Boundary enforcement

CI fails if:

- `sentry_sdk` is imported outside `sentry_bootstrap.py`;
- enclave code or enclave requirements reference Sentry;
- non-namespaced `SENTRY_*` variables appear;
- payload/PII/source/local capture is enabled;
- framework auto-instrumentation, trace propagation, profiling, or sessions are
  enabled;
- scrub hooks, manual trace sampling bounds, or host entry-point initialization
  disappear;
- an incident failure code is missing from the instrumentation matrix.

The tested dependency remains pinned at `sentry-sdk==2.66.1`. Public auditors
and miners can leave the gate unset, in which case all helpers are inert.
