# Sentry Weight Input Authority v2 Remediation Plan

## Decision

V2 is a no-code remediation and verification phase.

The complete v1 implementation has no confirmed unresolved product, security,
compatibility, test, or rehearsal defect. Do not change production source,
tests, protected manifests, or the restart runner merely to create phase-5
churn. Add only this v2 plan, then proceed to frozen-candidate verification.

Binary recommendation: accept v1 for completed-work verification, withhold push
if the official `prepush` gate or its live deployed-N-1 prerequisites cannot be
completed.

## Scope and Sentry disposition

The task remains scoped to the 35 unique Sentry issues inspected for project
`4511844334239744` over 14 days.

Only issue `7648740574` required new code. Its 20 events across epochs
24,322-24,333 and eight releases showed a recurring disagreement between
provisional validator inputs and independently measured gateway bans,
fulfillment rewards, and sourcing history.

The v1 implementation addresses that defect with a signed optional
`snapshot_authority_mode="gateway-measured-v2"` path, exact legacy-wire
preservation, a nine-field normalization allowlist, deterministic fulfillment
ordering, durable-ban-aware sourcing replay, strict receipt equality after
normalization, independent gateway/client/enclave verification, provisional
caller-vector verification, authoritative host/enclave equality, and
enclave-only publication.

The remaining Sentry groups stay in their v1 dispositions:

- Current-main restart/runtime fixes require exact-release rollout evidence.
- Allocation, publication, finalization, and chain-readback failures are
  downstream or operational fail-closed symptoms.
- Superseded Research Lab execution failures require recurrence on the exact
  new release before another patch is justified.
- The `<stdin>` compact-ancestry probe is a bad diagnostic, not a production
  protocol defect.

No fallback vector, warning suppression, manual chain submission, production
restart, Sentry resolution, or deployment is authorized.

## V1 findings and remediation status

### Protected workflow digest drift

Confirmed finding: five gateway protected AST digests were stale after the v1
source changes.

Root cause: protected symbols changed but the canonical generated manifest had
not yet been refreshed.

Remediation completed in v1: regenerate through the repository's canonical
generator while preserving the existing protected source commitments. Both
gateway and validator protected manifests now validate.

Regression evidence: both protected-manifest validation commands passed from
the LF candidate mirror.

V2 action: none.

### Compact ancestry selected the persistence wrapper

Confirmed finding: compact input ancestry could select the terminal
artifact-persistence proof and graph rather than the direct measured execution
proof. The persistence proof does not disclose the direct measured input
receipt required by compact verification.

Root cause: `_compact_weight_ancestry()` preferred the generic terminal fields
even when execution-specific compact evidence was present.

Remediation completed in v1: prefer `execution_ancestry_compact_proof` with
`execution_receipt_graph`, retaining the terminal fields as the legacy
fallback.

Regression evidence: the focused builder test verifies every compact category
resolves to the execution receipt rather than the persistence wrapper.

V2 action: none.

### Artifact-persistence output was not available as a strict receipt-bound value

Confirmed finding: ban-aware sourcing needed to consume a canonical
artifact-persistence output and prove that the terminal receipt committed
exactly that value.

Root cause: the executor output existed, but the scoring bridge did not expose
it through an exact schema-and-output-root validation boundary.

Remediation completed in v1:

- Validate exact persistence output and artifact descriptor fields.
- Require unique, sorted artifact IDs and a canonical `artifact_set_root`.
- Bind the returned output to the terminal receipt's `output_root`.
- Pass the validated output into measured sourcing.
- Require exactly one persisted artifact whose plaintext hash binds the bans
  value.

Regression evidence covers missing and extra fields, malformed descriptors,
duplicate and unsorted IDs, wrong set roots, receipt mismatch,
missing/direct/duplicated/forged/wrong-purpose/wrong-epoch bans ancestry,
output mismatch, and extra terminal parents.

V2 action: none.

### Rehearsal adapter defects discovered during real execution

Confirmed findings in the v1 rehearsal implementation included a keyword-only
source invocation mismatch, incorrect persistence logical-operation IDs,
premature local graph validation before parent merge, a sourcing receipt
stamped with the current rather than source epoch, fabricated-lineage risk, and
a missing real hotkey authorization ID.

Root cause: the first rehearsal draft did not yet reproduce all production
coordinator and validator authority contracts.

Remediation completed in v1:

- Invoke the real source resolver with its keyword-only contract.
- Use production-shaped logical operation IDs.
- Merge declared external parents before graph validation.
- Stamp signed sourcing evidence with the source epoch.
- Derive one shared lineage with `derive_ancestry_lineage_id_v2`.
- Use the real `ValidatorHotkeyAuthorityV2` seed provisioning, result
  registration, application signing, and validator publication ancestry proof.
- Exercise real coordinator manager/source/executor, gateway client, validator
  enclave authority, compact primary verifier, publication operation, public
  authority builder, compact audit verifier, and typed immutable-field
  rejection.

All five gateway-measured diagnostic stages and the joined invariant passed.

V2 action: none.

## Missing or limited acceptance evidence

### Official frozen-candidate prepush gate

Status: missing and required.

Reason: the candidate has not yet been frozen into an exact commit. The public
gateway later recovered and a read-only `GET /health/v2-authority` returned
HTTP 200 with status `ready` and deployed commit
`56f6aa425fb08f468e2701b2001f13303419236f`; this is authoritative point-in-time
N-1 evidence, but it must be rechecked immediately before the rehearsal. At
the same checkpoint, the official Subnet 71 snapshot was epoch block 193 and
current `origin/main` was `99fb514e013f751214bb8419ff607cb104514f88` with its
unit/workflow job passed while Docker smoke and independent attested builds
were still running.

Disposition: this is a phase-6/7 verification and release-readiness gate, not a
V2 source defect.

Required action:

1. Freeze the complete reviewed candidate, including both plan files.
2. Recheck the official live epoch block and current GitHub
   release/attestation state.
3. Verify the exact deployed N-1 SHA through an approved read-only authority.
4. Run:

   ```bash
   python3 scripts/run_local_restart_rehearsal.py \
     --from-sha <verified-deployed-n-minus-one-sha> \
     --candidate-sha <frozen-candidate-sha> \
     --transition forward \
     --profile prepush
   ```

5. Require every critical stage to be `passed`; no critical stage may be
   `failed` or `unexercised`.
6. Report per-stage and total duration.

The user did not request `unaccelerated`; do not run that profile.

If the deployed N-1 identity, epoch block, or attestation state remains
unavailable, stop before push. Do not substitute the Sentry release tag, a
guessed SHA, or a working-tree snapshot.

### Windows-mounted real-vsock timeout

Status: resolved for completed-work verification; not a candidate defect.

Evidence: `test_local_vsock_runs_real_framing_and_rejects_unknown_rpc` timed out
only from the Windows-mounted LF mirror; the identical unmodified LF base
passed the exact test. The exact current-main candidate then passed the full
native-Linux integrity file, 108/108, including the real-vsock test. Native
Windows collection still has pre-existing Linux-only `fcntl` and CRLF raw
signed-contract limitations.

Disposition:

- Do not increase production or rehearsal timeouts to hide a mount-specific
  slowdown.
- Do not skip the test.
- Keep candidate and prepush verification on a native LF Linux/WSL filesystem
  with the declared-requirements environment.
- Treat any future same-environment candidate-only recurrence as a real defect;
  do not widen timeouts or skip the test.

### Post-deploy production proof

Status: unavailable and out of scope because deployment was not requested.

Required later evidence, under separate deployment authorization, remains
exact candidate identity, PCR0, protected manifests, coordination marker,
identical gateway/primary/audit bundle hash, durable publication and
finalization, finalized extrinsic, `LastUpdate`, chain readback, and three
consecutive complete epochs without measured-input drift or downstream 502
alarms.

Historical Sentry issues must not be resolved before that proof exists.

## V2 change set

Create:

- `sentry-weight-input-authorityv2.md`

Do not modify:

- Production gateway, canonical, validator, enclave, or neuron modules.
- Unit or integration tests.
- Restart rehearsal implementation or behavior contract.
- Protected workflow source declarations or generated manifests.
- Dependencies, database schema, migrations, configuration, or deployment
  files.

If phase-6 verification finds a candidate-only failure, return the exact
failure to a scoped V2 implementation pass and add the smallest regression
before repeating verification. Do not preemptively change code.

## Rehearsal-size decision

Retain the 1,407-line addition to
`tests/restart_rehearsal/production_workflow_runner.py`.

The size is substantial, but the added code is not duplicated production
policy or a synthetic success fixture. It supplies strict sanitized adapters
for privileged Supabase, object-lock, chain, attestation, and hotkey boundaries
while driving the existing production coordinator, persistence, client,
enclave, compact verification, publication, and audit paths.

Splitting it into another helper module now would move rather than remove
complexity, enlarge the diff, and add import and maintenance surface without
improving incident acceptance. A later refactor is justified only if another
scenario reuses these adapters and can preserve identical stage-ledger and
integrity coverage.

## Vector-evidence decision

Do not add duplicate primary/audit vector fields solely for display.

The measured path already provides:

- Direct host/enclave equality at UID order, IEEE-754 float bits, sparse UID
  order, and u16 weights through `_verify_host_vector` and its focused
  regressions.
- A compact primary verifier that validates the signed immutable bundle.
- A compact public audit verifier that binds the same bundle hash through the
  publication authority.
- A rehearsal invariant requiring the primary and audit bundle hashes to
  match.
- Existing independent direct primary/auditor `uids` and `weights_u16`
  equality coverage in the broader rehearsal.

Re-serializing the same measured vector into another evidence field would not
be an independent check. Add it only if a future verifier consumes vector
evidence without verifying the signed bundle.

## Completed v1 and current-main integration evidence

The exact task diff was three-way integrated onto fetched current main
`99fb514e013f751214bb8419ff607cb104514f88`. All production, test, and runner
hunks applied cleanly; only the two generated protected manifests conflicted,
and both were canonically regenerated from the combined source while
preserving the current upstream commitments. The publishing worktree and
native LF candidate had the same tracked binary diff SHA-256,
`81648e3c8ea1cb5600a6baf20cb82638de4c689aa72dad501844b1021a5e2b11`.

The fresh clean LF declared-requirements verification produced:

- 247 focused tests passed.
- 59 compact/canonical/epoch/protected tests passed.
- 89 related Sentry, recovery, coordinator, bundle, submission, and external-
  boundary tests passed.
- All 11 independent workflow stages passed, including all five
  gateway-measured stages, and `gateway_measured_authority_exact` was true.
- Both protected manifests validated.
- All touched Python files compiled.
- `git diff --check` passed.
- `tests/restart_rehearsal/test_rehearsal_integrity.py` passed 108/108 from
  native LF storage, including the real-vsock framing regression.

This evidence supports a no-code V2 decision but does not replace
frozen-candidate verification.

## Required completed-work verification

Run from a clean native LF Linux/WSL candidate environment with only declared
test requirements:

```bash
python3 -m pytest -q \
  tests/test_hotkey_authority_v2.py \
  tests/test_weight_authority_v2.py \
  tests/test_sourcing_history_v2.py \
  tests/test_coordinator_executor_v2.py \
  tests/test_attested_scoring_v2_bridge.py \
  tests/test_coordinator_weight_source_v2.py \
  tests/test_attested_weight_inputs_v2.py \
  tests/test_gateway_weights_v2.py \
  tests/test_gateway_weight_inputs_client_v2.py \
  tests/test_validator_weight_authority_v2.py \
  tests/test_authoritative_weight_flow_v2.py
```

Repeat the previously green compact/canonical/epoch/protected and
Sentry/ban/release/coordinator/bundle/submission slices, then run:

```bash
python3 -m pytest -q tests/restart_rehearsal/test_rehearsal_integrity.py
git diff --check
python3 -m py_compile <all touched Python files>
python3 -m gateway.tee.protected_workflows \
  --root . \
  --manifest gateway/tee/protected_workflows.json
python3 -m validator_tee.host.protected_workflows_v2 \
  --root . \
  --manifest validator_tee/enclave/protected_workflows_v2.json
```

Finally run the frozen-candidate `prepush` command above. Re-test every v1
finding and require the gateway-measured stage ledger to include:

- `diagnostic:gateway-measured-coordinator`
- `diagnostic:gateway-measured-client-enclave`
- `diagnostic:gateway-measured-primary-compact`
- `diagnostic:gateway-measured-audit-compact`
- `diagnostic:gateway-measured-immutable-tamper`

Require `gateway_measured_authority_exact` to be true.

## Acceptance criteria

1. No V2 production, test, rehearsal, manifest, dependency, migration,
   configuration, or deployment change is introduced without new failure
   evidence.
2. All completed-work focused and integrity checks pass from a native LF
   candidate environment.
3. Both generated protected manifests validate from the frozen candidate.
4. Every critical `prepush` stage passes from the verified deployed N-1 SHA to
   the exact candidate SHA.
5. The five gateway-measured diagnostics pass and
   `gateway_measured_authority_exact` is true.
6. The candidate remains backward-compatible for exact legacy requests and
   fail-closed for unknown modes, immutable mutations, invalid ancestry, vector
   divergence, publication failure, and finalization failure.
7. No production write, restart, deployment, Sentry resolution, or chain
   submission occurs under `$ship`.
8. If live N-1, epoch, or attestation authority is unavailable, the branch is
   not pushed and the exact blocker is reported.

## Rollout limitations and rollback

This plan authorizes neither merge nor deployment.

Before push, the repository's epoch and attestation gate must be satisfied. A
later deployment must activate gateway and validators on one exact attested SHA
and prove the full durable and chain path before any Sentry issue is resolved.

Rollback, if separately authorized after deployment, must use all components
on the previous full 40-character attested SHA. The observed `57819588...`
release retains the original measured-input defect and is therefore only an
emergency fail-closed availability rollback, not a durable resolution.
