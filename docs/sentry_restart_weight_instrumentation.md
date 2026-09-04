# Sentry restart and canonical-weight instrumentation

This is the implementation contract for the incident inventory in
`leadpoet_restart_weight_sentry_issue_inventory.md`. Sentry is a sanitized,
best-effort diagnostic projection. It is never an authority or prerequisite.
Gateway and validator restarts use the selected source plus locally built
runtime identity, PCR0, canonical bundle hashes, and finalized chain state.
They do not wait for GitHub attestation or GitHub test completion.

## Correlation contract

Every operation carries the safe identifiers available at that boundary:

- `release_correlation_id`: deterministic from the exact 40-character release
  SHA and shared by GitHub release jobs, host release verification, PCR0 build,
  gateway, validator, and auditor processes.
- `restart_invocation_id`: generated once by each restart controller and
  preserved across controller re-execution and release supersession.
- `weight_correlation_id`: deterministic from exact runtime SHA + netuid +
  epoch and therefore stable before and after bundle creation. `bundle_hash`
  remains a separate comparison field so divergent bundles still share the
  trace that exposes their mismatch.
- `validator_id_hash`: a non-reversible SHA-256 join key for primary/auditor
  identity. Raw wallet names, hotkeys, and coldkeys are not exported.

Manual stage transactions derive the same 128-bit Sentry trace ID from these
correlation values. Trace headers are not propagated over trusted APIs and are
never included in signed inputs, manifests, receipts, or chain payloads. This
keeps telemetry outside every V2 trust decision while allowing independently
started processes to be searched as one operation.

## Instrumentation matrix

| # | Incident and code path | Stable `failure_code` | Transaction / stage | Captured allowlisted fields | Alert | Regression coverage |
|---|---|---|---|---|---|---|
| 1 | Staged source/runtime mismatch: `gw_restart.sh`, worker import preflight | `release.source_tree_mismatch` | `leadpoet.release` / `source_prepare`, restart `git_prepare`, `worker_import_preflight` | candidate/runtime SHA, release/source/build hashes, stage ledger, exception class | Any terminal event | signature matrix; release summary; N-1 rehearsal import closure |
| 2 | Supabase/PostgREST schema mismatch: restart schema preflight and host persistence boundary | `release.schema_contract_mismatch` | `leadpoet.release` / `schema_preflight`, restart `validator_weight_input_storage_preflight` | candidate SHA, migration/schema version, HTTP status, SQLSTATE, error code, query fingerprint | Any terminal event | signature matrix; disposable-PostgreSQL N-1 rehearsal |
| 3 | Release absent, superseded, or misaligned: release workflow and both restart controllers | `release.channel_unavailable` | `leadpoet.release` / `evidence_upload`, `immutable_publication`; restart `v2_release_acquisition` / `release_acquisition` | requested/candidate/approved/superseded/runtime SHA, attempt budget, release status, endpoint, elapsed time, shutdown flag | Terminal exhaustion; warn on rising retry duration | retry-storm de-duplication; release/restart summary; N-1 alignment gate |
| 4 | Builder disk/stale mounts: `.github/workflows/attested-v2-release.yml` | `release.builder_resource_exhausted` | `leadpoet.release` / `host_memory_guard`, `storage_reclaim`, `gateway_validator_build` | role/host alias, candidate SHA, free/required bytes, cleanup result, total duration, downstream skipped stages | Any failed builder; latency/resource threshold | release-stage classifier and workflow static contract; N-1 build |
| 5 | Artifact authentication/capacity: host restart and artifact-bound failures | `release.artifact_auth_failed` | restart `v2_offline_artifact_prepare`, `v2_credential_envelope_preparation` | sanitized artifact/hash class, schema/manifest/boot hashes, bytes/count, error class; never documents or envelopes | Immediate terminal alert | exact inventory signature mapping; scrub/oversize tests; N-1 artifact preflight |
| 6 | Enclave listener/relay unavailable: `runtime_v2_bootstrap.py`, `chain_relay_v2.py`, `vsock_client.py` | `runtime.enclave_relay_unavailable` | `leadpoet.runtime` / `enclave_boot`, `relay_request`; restart runtime readiness stages | physical role, CID/port alias, boot identity hash, method, attempts, duration, cleanup result | Terminal relay/readiness failure | host relay/vsock tests; no-enclave-egress guard; N-1 runtime readiness |
| 7 | Chain/Supabase authority unreadable: gateway startup, weight retrieval, chain transport | `authority.dependency_unreadable` | `leadpoet.authority` / dependency-specific read stage | dependency/endpoint alias, RPC method, HTTP/SQL error, attempt/backoff, current/finalized block, retryable/fail-closed | Exhaustion only; retries are breadcrumbs | signature and retry-storm tests; focused gateway/validator authority tests |
| 8 | Excessive restart work: both restart timing ledgers | `restart.stage_deadline_exceeded` | `leadpoet.restart` / every ledger stage | invocation ID, complete bounded stage ledger, per-stage/total duration, last success, first unfinished, count/bytes/cache fields where available | Any stage above configured deadline; p95 dashboard | ledger duration/deadline tests; shell heredoc/static wiring test; N-1 duration ledger |
| 9 | Missing finalized allocation authority: gateway canonical allocation path | `weight.allocation_authority_missing` | `leadpoet.weight` / `allocation_reconstruction`, restart `validator_weight_input_repair` | netuid/epoch, authority/frontier/bundle hashes, first missing epoch, fail-closed state | Immediate terminal alert | signature mapping; gateway weight-path tests; N-1 canonical bundle stage |
| 10 | Incomplete chain-realized settlement history: ancestry/frontier restart path | `weight.settlement_history_incomplete` | `leadpoet.restart` / `ancestry_frontier_recovery`, `ancestry_postcheckpoint` | activation/requested/frontier epoch, first missing epoch, pages/counts, bundle/finalization/frontier hashes, duration | Immediate terminal alert | evidence-tail classifier and ledger test; disposable schema/N-1 restart |
| 11 | Oversized/unbounded ancestry or transport frame: gateway response and host vsock boundary | `weight.ancestry_bounds_exceeded` | `leadpoet.weight` / `input_reconstruction`, `enclave_verification` | root/frontier hashes, receipt/page/byte/depth counts, frame limit, response bytes, duration | Immediate terminal; warning metric near bound | signature/redaction tests; gateway response/vsock bounds tests; N-1 compact ancestry |
| 12 | Invalid frontier source or reopened terminal graph: restart ancestry bootstrap | `weight.frontier_source_invalid` | `leadpoet.restart` / `ancestry_frontier_recovery`, `ancestry_postcheckpoint` | RPC/SQLSTATE/HTTP code, source/frontier/allocation hashes, frontier epoch, graph counts, query fingerprint | Immediate terminal alert | exact SQLSTATE signature mapping; disposable RPC contract in N-1 rehearsal |
| 13 | Gateway weight route unavailable/502: `gateway_weight_inputs_v2.py`, primary and auditor retrieval | `weight.gateway_endpoint_unavailable` | `leadpoet.weight` / `bundle_retrieval`, restart gateway alignment | route kind, HTTP status, attempt/deadline, candidate/runtime SHA, epoch/block, last completed stage | Exhaustion; urgent near submission window | retry/terminal host tests; validator/auditor retrieval tests; N-1 handoff |
| 14 | Signed request exceeds block drift: `gateway/api/weights.py`, host retrieval retry | `weight.block_drift_exhausted` | `leadpoet.weight` / `request_authorization` | request hash, submitted/current/expected block, max drift, epoch, cache hit, attempt, blocks remaining | Any new-request rejection; one event on retry exhaustion | gateway drift/cache regression tests and signature matrix |
| 15 | Authoritative result schema/lineage invalid: host enclave result boundary and validator | `weight.authoritative_result_invalid` | `leadpoet.weight` / `enclave_verification` | schema, bundle/weights/snapshot/root/boot hashes, vector count/hash, failure class; no vector/signature | Immediate terminal alert | vsock/result validation and primary/auditor focused tests; N-1 enclave flow |
| 16 | Bittensor SDK response incompatible: primary and auditor response normalization | `weight.sdk_response_invalid` | `leadpoet.weight_submission` / `broadcast` | SDK versions/class, normalized status, error code, extrinsic hash, inclusion/finalization block, role | Immediate terminal alert | SDK 10 response-shape regression tests; N-1 signing/broadcast |
| 17 | Async substrate reconnect with open subscriptions: primary transport lifecycle | `weight.chain_transport_poisoned` | `leadpoet.weight_submission` / `broadcast`, `finalization` | transport-session hash, endpoint/RPC alias, attempt, epoch/block, extrinsic hash, cleanup result | Exhaustion; track retry rate | retry lifecycle tests and signature matrix; N-1 consecutive transport path |
| 18 | Missing finalization, `LastUpdate`, vector readback, or auditor proof: primary/auditor epoch lifecycle | `weight.finalization_missing` | `leadpoet.weight_submission` / `finalization`, `last_update_readback`, `vector_readback` | validator role/hash/UID, epoch, bundle/weights/extrinsic/finalization/vector hashes, broadcast/inclusion/finalized blocks, prior/current `LastUpdate`, missing milestones | Missing primary or required auditor by bounded epoch deadline | primary/auditor milestone regressions; canonical bundle equality and N-1 finalization/readback |
| 19 | Genuine expected/observed PCR0 mismatch: host PCR builder and release verifier | `release.pcr0_mismatch` | `leadpoet.release` / `pcr0_build`, `pcr0_verification` | exact SHA, expected/observed PCR0 hash, manifest/boot/build hashes, physical role | Immediate alert; never group with upstream release failure | PCR classifier/host verifier tests; N-1 release identity gate |
| 20 | Gateway/primary/auditor canonical bundle or vector diverges | `weight.bundle_divergence` | `leadpoet.weight` / `bundle_verification`, `vector_readback` | SHA/epoch/role/validator hash, bundle/weights/vector/publication hashes, observed count, `vector_matches` | Immediate cross-role alert | canonical flow tests and N-1 byte-identical bundle/readback stage |

Unknown terminal restart failures retain `restart.terminal_failure` rather than
being mislabeled. They still include the bounded stage ledger and correlation
identity so a new family can be promoted to a stable code after investigation.

## Event behavior and cost controls

- One semantic event is emitted for a terminal logical failure. A process-local
  limiter suppresses wrapper/rethrow duplicates for the same correlation,
  component, failure code, and validator role.
- Retries are bounded breadcrumbs and stage timings are sampled manual
  transactions plus distributions. Framework, HTTP, database, and chain
  auto-instrumentation is disabled, so payloads and query bodies are never
  collected and telemetry adds no Supabase or chain reads.
- Terminal events use the Sentry error path and are not sampled. Successful
  traces default to 1% and are capped at 10%. At most 50 breadcrumbs are kept.
- Shell/Actions summary bridges are bounded to two seconds or one minute,
  respectively, use `continue-on-error`, and can never change the underlying
  exit status. Enabled release summaries make one bounded, read-only GitHub
  Actions API call to project native step durations; disabled Sentry makes no
  call. SDK shutdown flush is capped at one second.
- Missing DSN, disabled wiring, SDK import/init/capture/scrub/transport failure,
  slow collector, or Sentry outage is a no-op for the application. Original
  exceptions and fail-closed V2 behavior are preserved.

## Configuration

| Variable | Default | Contract |
|---|---:|---|
| `LEADPOET_SENTRY_ENABLED` | unset | Must be truthy together with a DSN; otherwise complete no-op |
| `LEADPOET_SENTRY_DSN` | unset | Secret runtime/Actions setting; never committed |
| `LEADPOET_SENTRY_ENVIRONMENT` | `production` | Sanitized environment label |
| `LEADPOET_SENTRY_RELEASE` | exact deploy SHA | Must resolve to exact candidate/runtime identity |
| `LEADPOET_SENTRY_TRACES_SAMPLE_RATE` | `0.01` | Successful manual stage transactions; clamped to `[0, 0.10]` |
| `LEADPOET_SENTRY_EXTRA_PROTECTED_MODULES` | unset | Comma-separated prefixes that may only widen redaction |
| `LEADPOET_SENTRY_MESSAGE_MODE` | `scrub` | `redact-all` removes all diagnostic messages |
| `LEADPOET_SENTRY_RESTART_STAGE_DEADLINE_SECONDS` | `900` | Monitoring threshold only; clamped to `[1, 7200]`, never changes restart deadline |

`LEADPOET_RESTART_INVOCATION_ID` is generated by restart controllers and
propagated to child host/container processes; it is not operator configuration.

## Example searches

Terminal restart root cause:

```text
leadpoet.failure_code:weight.settlement_history_incomplete
leadpoet.restart_invocation_id:restart:gateway:*
```

One epoch across gateway, primary, and auditors:

```text
leadpoet.weight_correlation_id:weight:<digest>
```

One exact release across GitHub and hosts:

```text
leadpoet.release_correlation_id:release:<digest>
```

The semantic fingerprint is always:

```text
["leadpoet-semantic", <component>, <stage>, <failure_code>]
```

## Recommended alerts

Create error alerts for each stable failure code, then page immediately for:

- `release.pcr0_mismatch`, `weight.bundle_divergence`, or any terminal
  attested-release job;
- `restart.terminal_failure` or any terminal gateway/validator restart after
  production shutdown begins;
- `weight.gateway_endpoint_unavailable` near the weight window;
- `weight.block_drift_exhausted`;
- `weight.finalization_missing` for the primary or any required auditor.

Create metric alerts for restart/release stage p95, repeated authority retries,
ancestry response bytes near their structural bound, and missing completed
milestones. Alerting must consume Sentry data only; it must not feed back into
release approval, attestation, restart, scoring, or weight decisions.

## Verification and unexercised boundaries

The focused suite covers disabled/enabled SDK behavior, real SDK capture via an
in-memory transport, SDK failures, bounded trace sampling, allowlist redaction,
seeded fake secrets, oversized data, retry storms, stable fingerprints,
deterministic cross-process IDs, restart/release ledgers, shell heredoc syntax,
workflow fail-open wiring, and every inventory signature. The required N-1
`prepush` rehearsal covers release identity, PostgreSQL/PostgREST contracts,
gateway/validator startup, canonical bundle equality, primary/auditor signing,
submission, finalization, `LastUpdate`, vector readback, rollback, and cleanup.

No test sends telemetry from a Nitro enclave or performs a production write.
Local macOS measurements with Python 3.11 and pinned `sentry-sdk==2.66.1`
showed a 27 ms median one-time disabled import increment, 8 ms active
initialization with an in-memory transport, 4.9 microseconds per disabled retry
breadcrumb, 7.0 microseconds per disabled stage, 80 microseconds per enabled
retry breadcrumb, 183 microseconds per enabled unsampled stage, and 0.65 ms per
sampled stage. These are development-host measurements rather than production
SLOs; restart/release stage metrics are the production regression signal.

Collector failure is exercised with both a deliberately slow local HTTP
collector and an unreachable local endpoint, while capture latency and flush
deadlines remain bounded. No test sends data to Sentry SaaS. Production alert
routing, project retention, and no-training/deletion terms remain an operator
responsibility.
