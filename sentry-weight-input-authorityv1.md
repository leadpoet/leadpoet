# Sentry Weight Input Authority v1 Plan

## Status and scope

This plan addresses the unresolved production Sentry inventory for project
`4511844334239744` over the 14-day window inspected on 2026-08-03. The Sentry
UI returned 36 rows representing 35 unique issue IDs; issue `7648940448`
appeared twice in the captured result set. The implementation base is
`origin/main` at `02d6e6ff5096ade29714a42d45542208c2e22114`.

The only newly evidenced code defect is the recurring split between the
validator's provisional weight calculation and the gateway coordinator's
measured weight inputs. The implementation will preserve strict measurement
and receipt equality by moving equality after a narrowly bounded,
receipt-backed normalization step. It will not weaken verification, add a
fallback weight vector, perform a manual chain submission, restart production,
or resolve Sentry issues without live exact-release evidence.

## Current behavior and evidence

Sentry issue `7648740574` contains 20 observed events across epochs
24,322-24,333 and releases `d3c41ee6`, `071f4eeb`, `d19f290f`, `720cb484`,
`350ecf46`, `b576bf5d`, `987ac1d1`, and `57819588`. The error alternates among:

- `bans measured input differs from calculation`
- `fulfillment_rewards measured input differs from calculation`
- `sourcing_history measured input differs from calculation`

The validator currently reads bans and fulfillment through public gateway
endpoints and reads sourcing history from a mutable local file. Later, the
gateway coordinator independently reconstructs those values from measured
Supabase reads and durable signed sourcing-epoch records. Any difference causes
`build_gateway_weight_inputs_v2()` to fail before it can return the measured
receipts. An exact retry reuses the stale provisional calculation and cannot
converge.

Two concrete production-shaped gaps are present in the current implementation:

1. Fulfillment reward rows have no canonical database ordering on either the
   public endpoint or measured Supabase query, but receipt equality treats the
   resulting list order as significant.
2. The validator permanently rewrites each banned hotkey occurrence in local
   sourcing history to `-100_000`, while measured sourcing replay aggregates
   the unchanged positive scores from signed epoch documents. Existing tests
   cover only a single fulfillment miner and unbanned sourcing history.

The public production gateway probes `/health`, `/build-info`, and
`/health/v2-authority` all returned HTTP 502 during triage. The repository's
secure Sentry helper could not reach the authorized secret source because the
configured SSH key was unavailable on this Windows host, so the signed-in
Sentry UI was used as a read-only fallback. No production writes were made.

## Sentry disposition

### Requires this code fix

- `7648740574`: recurring gateway-measured input drift across 12 consecutive
  epochs and eight releases.

### Already addressed on current main; requires exact-release rollout proof

- `7650013795`: `runtime.enclave_relay_unavailable` on failed candidate
  `04288ea0`.
- `7649919954`: restart-time `authority.dependency_unreadable` on candidate
  `b1b6fdcc`.
- `7649624680`: restart stage deadline exceeded.
- `7649234649`: restart terminal-failure summary.

Current main contains the subsequent Nitro cgroup-v1 root-readiness and runsc
gofer-root fixes (`602c1442` and `02d6e6ff`). Those changes have not been
observed live because the gateway is down, so the groups are not confirmed
resolved.

### Operational/downstream symptoms; keep fail-closed

- `7648934938`: allocation fetch HTTP 502.
- `7648940429`: weight submission blocked because allocation was unavailable.
- `7649946194`: gateway weight endpoint unavailable.
- `7648940448`: allocation authority missing.
- `7648736163`: no durable publication before the epoch window closed.
- `7648742222`, `7648742229`, `7648742205`: finalized-chain proof missing and
  its wrapper/trace groups.
- `7649149351`, `7649149340`, `7649149328`, `7648946496`: compact publication
  503/store failure and wrapper groups on superseded releases.
- `7648740462`, `7648740394`: unsigned/prior-epoch journal quarantine; expected
  fail-closed recovery behavior.
- `7648946471`, `7648946455`, `7648929001`, `7648750617`: preparation/allocation
  failures and wrappers on superseded releases.

These groups do not justify a fallback, warning suppression, or manual weight
submission. They require exact-SHA deployment, durable publication/finalization
evidence, and chain readback.

### Superseded Research Lab execution failures; monitor exact new release

- `7649029237`, `7649232899`, `7648893626`, `7648756535`, `7648947705`,
  `7648959895`, `7648919035`, `7648852473`, `7648756315`.

These protected-surface scoring, artifact-store, and enclave RPC groups span
older releases. They are not sufficiently specific to justify another patch
without recurrence on the exact candidate after gateway recovery.

### Diagnostic probe only

- `7648746574`, `7648746529`, `7648746527`: a `<stdin>` probe assumed the
  legacy `upstream_receipt_set` field even after requesting compact ancestry.
  Production code already branches correctly between full and compact
  responses. The probe/runbook should be updated separately; the production
  protocol should not be changed to satisfy it.

## Goals

1. Allow a new validator to treat authenticated gateway measurements as the
   authority for exactly nine gateway-owned mutable snapshot fields.
2. Preserve strict canonical equality between the authoritative calculation
   and every measured receipt document.
3. Preserve all scoring, allocation, reward, burn, feature-flag, chain,
   metagraph, commit, and configuration semantics.
4. Preserve byte/schema-compatible behavior for N-1 validators that do not
   select the new signed mode.
5. Make sourcing history explicitly depend on the measured bans receipt and
   reproduce the existing per-epoch `-100_000` ban penalty without mutating the
   persisted signed epoch document.
6. Prove the authoritative host calculation and validator-enclave result are
   bit-identical before publication.

## Non-goals

- Deploying, restarting, merging, or mutating production.
- Resolving Sentry groups before exact-release live verification.
- Changing Research Lab allocation, fulfillment reward arithmetic, sourcing
  scores, promotion, settlement, emissions, burn policy, or chain submission.
- Normalizing validator-owned fields or accepting an arbitrary gateway
  calculation.
- Adding a database migration or new production dependency.
- Repairing the ad hoc `<stdin>` probe in production source.

## Architecture and data flow

### Signed mode selection

Extend the canonical weight-input request with an optional signed field:

```text
snapshot_authority_mode = "gateway-measured-v2"
```

The field participates in the canonical request hash and the validator-enclave
signature. A request without the field follows the exact legacy path. The one
known value selects normalization. Unknown values and extra fields fail closed.
Because the mode is signed, legacy and normalized singleflight/cache work can
never collide.

### Normalization boundary

Add a dependency-free canonical normalizer. It accepts a provisional
calculation and measured documents and may copy only this map:

| Measured category | Permitted calculation fields |
| --- | --- |
| `bans` | `banned_hotkeys`, `banned_lookup_ok` |
| `fulfillment_rewards` | `fulfillment_share`, `fulfillment_rows`, `fulfillment_fetch_ok` |
| `leaderboard` | `leaderboard_entries`, `leaderboard_fetch_ok` |
| `sourcing_history` | `rolling_lead_count`, `rolling_scores` |

The following remain immutable: allocation document and authority, champion,
reimbursement and source-add data, leaderboard shares, metagraph, burn
ownership, epoch, block, netuid, commit/config hashes, feature flags, constants,
parent receipt fields, and every unlisted value.

The normalizer validates each document's schema, category, netuid, epoch, and
exact value shape; copies only allowed values; runs `compute_final_weights()`;
and returns the authoritative calculation plus a sorted list of changed source
categories. After normalization, all gateway input documents are regenerated
from that calculation and compared canonically with all measured documents.
The existing equality boundary moves; it is not removed.

### Ban-aware sourcing

Gateway execution ordering becomes:

1. Allocation/config-derived categories, bans, fulfillment, and leaderboard.
2. Sourcing history, with both signed sourcing-epoch graphs and the signed bans
   execution graph as parents.
3. Canonical snapshot normalization.
4. Anomaly adjustment from the normalized calculation and measured upstream
   documents.
5. Final all-category canonical equality.

The sourcing replay validates the bans receipt role, purpose, epoch,
persistence lineage, declared-parent membership, and output root. For every
authenticated epoch document, a banned hotkey already present in that epoch is
treated as exactly `-100_000`; absent hotkeys are not introduced. Raw signed
sourcing documents are never mutated or rehashed.

### Response and validator flow

Legacy full and compact responses remain unchanged. Normalized full and compact
responses add:

```text
snapshot_authority_mode
authoritative_calculation_snapshot
authoritative_calculation_snapshot_hash
normalized_source_categories
```

`calculation_snapshot_hash` continues to identify the signed provisional
snapshot. The client validates the exact response field set, authoritative
hash, allowed diff, and independently derived changed-category list before
passing the response across vsock.

The validator host first proves the caller's UID/float vector matches canonical
computation of the provisional snapshot. It then recomputes from the
authoritative snapshot, sends that exact snapshot and measured receipts to the
validator enclave, and compares the enclave vector bit-for-bit and u16-for-u16
with the authoritative host calculation. Only the enclave result proceeds to
publication and chain submission.

## File-level implementation

- `leadpoet_canonical/hotkey_authority_v2.py`
  - Add exact legacy/new request schemas and signed mode hashing.
- `leadpoet_canonical/weight_authority_v2.py`
  - Add the canonical allowlisted normalizer and changed-category derivation.
- `leadpoet_canonical/sourcing_history_v2.py`
  - Add ban-aware rolling replay with existing ordering/numeric semantics.
- `gateway/tee/coordinator_weight_source_v2.py`
  - Allow pre-normalization measured values only for the four categories in
    normalized mode; retain strict early equality elsewhere; validate the bans
    parent and produce ban-aware sourcing history.
- `gateway/research_lab/attested_weight_inputs_v2.py`
  - Sequence bans before sourcing, normalize before anomaly, and enforce final
    equality for all categories. Preserve strict legacy behavior.
- `gateway/api/weights.py`
  - Add exact normalized response models, mode-aware singleflight/cache keys,
    and hash/category-only observability.
- `validator_tee/host/gateway_weight_inputs_v2.py`
  - Sign the mode, validate normalized full/compact responses, and repeat the
    canonical normalization boundary locally.
- `validator_tee/host/authoritative_weight_flow_v2.py`
  - Verify provisional caller vector, then authoritative host/enclave vector,
    and publish only the enclave result.
- Protected workflow declarations/manifests
  - Include the request-mode, normalization, ban-aware sourcing, builder/API,
    client, and authoritative-flow symbols in both gateway and validator
    protected closures.
- Focused unit, integration, protected-closure, and restart-rehearsal fixtures
  - Add stale provisional bans, fulfillment, leaderboard, and sourcing cases.

No SQL migration or external interface other than the backward-compatible
signed request/response variant is planned.

## Security and compatibility constraints

- No host-selected value may bypass a measured receipt.
- The gateway cannot normalize an unlisted field.
- The validator client independently repeats normalization and rejects any
  extra response field or category mismatch.
- The validator enclave continues to verify attestation, receipt signatures,
  source evidence, output roots, chain state, commit, config, and final vector.
- Legacy requests and responses remain exact and strict for N-1 compatibility.
- Unknown modes fail closed.
- Logs contain only mode, hashes, and category names; never hotkeys, scores,
  source rows, provider payloads, or credentials.
- Failures remain 503/fail-closed. No stale cache, manual vector, or burn-only
  fallback is introduced.

## Failure states and observability

The normalized request must fail closed for invalid signatures, unknown modes,
forbidden field changes, malformed documents, source/receipt mismatch, missing
or duplicated bans ancestry, invalid sourcing records, normalization failure,
host/enclave divergence, or publication/finalization failure.

Record bounded stage telemetry for provisional hash, authoritative hash,
signed mode, normalized category names, and terminal failure code. Do not emit
the input values. Existing Sentry wrapper coalescing remains unchanged.

## Acceptance criteria

1. Production-shaped stale provisional bans, fulfillment, leaderboard, and
   sourcing values succeed only in signed normalized mode.
2. The same drift still fails in legacy mode.
3. Every immutable-field or extra-field mutation is rejected.
4. Measured sourcing includes and validates the direct bans authority and
   exactly reproduces the existing per-epoch ban penalty.
5. Final receipt output roots equal documents regenerated from the authoritative
   snapshot for every gateway category.
6. The provisional caller vector matches canonical provisional computation.
7. The authoritative host vector matches the validator-enclave result at UID,
   IEEE-754 float-bit, sparse UID, and u16 levels.
8. Compact and full ancestry verification remain intact.
9. Protected workflow manifests validate from the frozen candidate tree.
10. Focused tests, syntax checks, diff checks, and the repository `prepush`
    restart rehearsal pass with every critical stage passed.
11. No database migration, dependency, production mutation, or unrelated file
    enters the diff.

## Test matrix

### Canonical and component tests

- Every allowed overlay independently and in combination.
- Immutable, nested, schema, category, epoch, netuid, hash, and extra-field
  attacks.
- Multi-miner fulfillment data in reverse source order.
- Ban-aware sourcing with present/absent hotkeys and multiple epochs.
- Missing, forged, duplicated, wrong-purpose, wrong-epoch, unpersisted, and
  output-root-mismatched bans parents.
- Legacy drift rejection and normalized drift convergence.

### API and compatibility tests

- Exact legacy and normalized signed request hashes.
- Exact legacy/new compact and full response field sets.
- Unknown mode rejection and legacy/new singleflight isolation.
- Provisional versus authoritative hash/category validation.
- Compact proof and full receipt-set verification unchanged.

### Workflow and adversarial tests

- Provisional caller-vector mismatch rejected before gateway publication.
- Authoritative host/enclave mismatch rejected before publication.
- Enclave receives only the authoritative snapshot.
- Anomaly executes after normalization and binds all measured final documents.
- Protected gateway and validator closures/manifests validate.
- Restart rehearsal injects stale provisional dynamic values and reaches the
  same compact bundle at gateway, primary validator, and audit validator.

### Required release gate

Run focused pytest files for canonical weight authority, coordinator source,
attested input builder, gateway API/client, authoritative host/enclave flow,
protected workflows, canonical computation, validator epoch flow, model
sandbox, and restart evidence identity. Then run:

```text
git diff --check
python -m py_compile <all touched Python files>
python -m gateway.tee.protected_workflows --root . --manifest gateway/tee/protected_workflows.json
python -m validator_tee.host.protected_workflows_v2 --root . --manifest validator_tee/enclave/protected_workflows_v2.json
```

Because this changes gateway, validator, receipt ancestry, and weight
publication, freeze the candidate and run the repository `prepush` profile from
the verified deployed N-1 SHA. The user did not request `unaccelerated`, so the
long profile is out of scope.

## Rollout and rollback

This task may open a reviewed pull request. It does not authorize merge,
deployment, restart, Sentry resolution, production database mutation, or chain
submission.

Before any runtime-changing push, recheck the official live epoch block and
GitHub release/attestation state. A separately authorized deployment must prove
the exact candidate through `/health`, `/build-info`, `/health/v2-authority`,
PCR0, manifests, coordination marker, identical gateway/primary/audit bundle
hash, durable publication/finalization rows, finalized extrinsic, `LastUpdate`,
and chain readback. Observe three consecutive complete epochs with no new
measured-input drift or 502/downstream alarms before resolving historical
Sentry groups.

Rollback uses all components on the previous full 40-character attested SHA.
Release `57819588` may restore availability but contains the recurring
measured-input defect, so it is only an emergency fail-closed rollback and not
a durable resolution.
