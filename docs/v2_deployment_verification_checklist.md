# V2 Deployment Verification Checklist

Use this checklist for changes that could affect the gateway, primary validator,
audit validators, weight publication, attestation, PCR0 verification, or the V2
trust model.

## Before Pushing

### 1. Synchronize and Review

- [ ] Fetch `origin/main`.
- [ ] Fast-forward or rebase the candidate onto the latest `origin/main`.
- [ ] Commit the candidate locally.
- [ ] Review the complete diff for:
  - Secrets or credentials.
  - Unrelated changes.
  - Unintended changes to V2 behavior or verification policy.

### 2. Run the Mandatory Weight and Auditor Tests

```bash
python3 -m pytest -q \
  tests/test_weight_authority_v2.py \
  tests/test_authoritative_weight_flow_v2.py \
  tests/test_chain_source_v2.py \
  tests/test_validator_enclave_python37_compatibility.py \
  tests/test_validator_publication_recovery_v2.py \
  tests/test_weight_submission_retry.py \
  tests/test_auditor_same_epoch_mirroring.py \
  tests/test_auditor_v2_hardening.py
```

- [ ] Every test passes.
- [ ] No test is skipped unexpectedly.
- [ ] Any warning is reviewed and confirmed non-blocking.

### 3. Run the Complete Isolated Restart Rehearsal

Replace `<currently-deployed-sha>` with the exact commit currently running in
production.

```bash
python3 scripts/run_local_restart_rehearsal.py \
  --from-sha <currently-deployed-sha> \
  --candidate-sha HEAD \
  --transition forward \
  --profile prepush
```

This is the mandatory 5–10 minute developer gate. It runs in Docker without
production credentials, executes both installed N-1 launchers and the candidate
Git handoff, then runs one complete gateway/validator/auditor publication
against strict local chain and durable-database services.
Before either launcher can proceed, each replica starts disposable PostgreSQL,
applies the candidate settlement-critical migrations in production order, and
executes nonempty bundle/publication/finalization state through the production
settlement authority parser. The resulting relation/column/RPC contract binds
the strict local PostgREST service to the candidate SHA.
On ARM developer machines the outer `prepush` container uses native ARM64 to
avoid QEMU overhead; the unchanged launchers still issue and validate the exact
production `linux/amd64` Docker/Nitro contracts. The `release` profile always
runs its outer replica as pinned `linux/amd64` for the final ABI check.

Failure behavior is part of this gate. Both profiles must continue through all
independent stages after an earlier failure and write one complete
`leadpoet-restart-rehearsal-<sha>-<profile>-stages.json` ledger. A failed
gateway launcher must not suppress the validator launcher or independent
canonical workflow. A failed release fault case or epoch must not suppress
later fault cases or epochs. Dependency-blocked stages must be recorded as
`unexercised` with their exact prerequisites; they are never inferred to pass.
The controller exits nonzero after collecting the full blocker set.

The frozen candidate also generates one hash-bound behavioral contract from
its protected-workflow manifest, exact production entrypoints, effective
conditional-ICP and Git-tree policies, release profile, and fault contract.
The workflow runner and final evidence join reconstruct it independently from
the read-only candidate source. Do not add fixed source counts, ICP splits,
tree widths, fault minimums, or duplicate stage lists to either component.
Policy-sized state generators must exercise the production assignment,
replacement-lineage, settlement, bundle, primary-validator, auditor, signing,
publication, finalization, and readback code. A missing, duplicate, stale, or
undeclared stage/invariant must fail the joined manifest.

Before an attested release, run the release profile:

```bash
python3 scripts/run_local_restart_rehearsal.py \
  --from-sha <currently-deployed-sha> \
  --candidate-sha HEAD \
  --transition forward \
  --profile release
```

The release profile runs forward, rollback, and roll-forward with the unchanged
installed launchers, then the full boundary fault matrix, concurrency checks,
and 100 accelerated stateful subnet epochs. The three transitions use one
unchanged candidate SHA. They also share one candidate-migrated strict
PostgREST durable state: application processes, filesystems, and enclaves are
rebuilt for each exact target commit, while the database schema and rows
survive every activation exactly as they do in production. The final join
requires each launcher's starting durable revision and content hash to match
the prior launcher's ending identity. The rollback target must also pass
`Leadpoet/utils/exact_commit_restart_v2.py` against current `origin/main`;
the helper rejects actual public auditor protocol incompatibility but does not
reject an attested release merely because its implementation predates later
reliability fixes.
The operator-facing paired rollback must use the same selected SHA for both
hosts:

```bash
bash scripts/restart_attested_release_local.sh \
  --commit <supported-previous-release-sha>
```

Use a single-component mode only when the other runtime is already verified on
that exact SHA. The paired command must capture the validator restart start,
then run gateway restart and validator preparation concurrently. Validator Git
selection, release verification, artifact staging, old-runtime shutdown, EIF
build, Nitro launch, chain relay, runtime/hotkey bootstrap, and exact
application-image build may overlap the gateway restart. No validator
coordinator or worker may start until the application image's commit label and
immutable image ID are verified, the exact-SHA coordination marker is
published, and gateway V2 authority health, build-info, and immutable release
evidence all report that same SHA. The image ID must be unchanged after the
wait, and the wrapper must repeat the full same-SHA check after startup.

A failed gateway restart must signal the validator SSH job immediately and
publish a commit-bound failure marker concurrently, with a bounded marker
write, so slow coordination transport cannot delay signal cleanup. The
operator-owned coordination path and wait bound must survive candidate
re-execution and secret hydration. The validator must fail closed without
starting any coordinator or worker and clean every resource prepared after
old-runtime shutdown: validator containers, host validator/relay processes,
Nitro enclave, and Docker lock. A retry after an earlier N-1-to-N Git handoff
may invoke only the selected SHA's clean launcher blob; the coordinated
expected SHA must be rechecked after pull and before release preparation or
shutdown. A historical rollback deployer without the image-prepared barrier
must retain the safe fallback check immediately before that deployer is
invoked. This parallel flow intentionally does not promise that the old
validator remains running through a late gateway failure; it minimizes elapsed
restart time while preserving the authority boundary.

Both profiles must produce a joined
`leadpoet-restart-rehearsal-<sha>-<profile>.json` manifest. The manifest and
console output must prove all of the following:

- [ ] `PYTHON37_FINALIZATION_PROBE_SUCCESS`
- [ ] `REHEARSAL_SUCCESS component=gateway`
- [ ] `REHEARSAL_SUCCESS component=validator`
- [ ] `REHEARSAL_STAGE_EVIDENCE` points to the exact candidate's stage ledger.
- [ ] Every declared stage is present exactly once and has status `passed`.
- [ ] Runner and join report the same candidate behavioral-contract hash.
- [ ] Production source identities exactly cover the candidate-derived
  protected-workflow and entrypoint inventory; no fixed file count is used.
- [ ] Conditional ICP and Git-tree evidence use the candidate's effective
  policy hashes and generated sizes rather than fixed current defaults.
- [ ] Settlement evidence accepts every legal contiguous persisted prefix and
  rejects duplicate, gapped, missing-first, ahead-of-target, and excessive
  backlog states through the production validator.
- [ ] Missing and unexpected stages both fail the final evidence join.
- [ ] Zero `failed` and zero `unexercised` production-critical stages.
- [ ] The multi-failure regression proves gateway, validator, and canonical
  workflow diagnostics are aggregated in one invocation.
- [ ] The release fault matrix and all 100 epochs continue after independent
  injected failures and report every resulting blocker together.
- [ ] Exact rollback succeeds when the change affects restart or release
  selection.
- [ ] Exact roll-forward succeeds again from the rollback target.
- [ ] The paired operator coordinator overlaps gateway restart with validator
  preparation, rejects single-component SHA mismatches, and proves that no
  validator coordinator or worker starts before the exact-SHA marker plus
  gateway health/build/release evidence.
- [ ] The validator application image commit label and immutable image ID are
  verified before the activation wait, the same image ID is rechecked after
  alignment, and the full same-SHA gateway check passes again after startup.
- [ ] A gateway failure immediately signals the validator and concurrently
  publishes the selected SHA's failure marker through a bounded write; the
  validator starts no coordinator or worker and cleans containers,
  relay/validator processes, enclave, and Docker lock even if it previously
  consumed a success marker.
- [ ] Candidate re-execution and secret hydration cannot replace the paired
  operator's coordination path, coordination retry bound, or overall timeout.
- [ ] A retry whose checkout already equals the candidate runs only that
  candidate's clean launcher blob; an `origin/main` advance after operator
  selection fails before release preparation and old-runtime shutdown.
- [ ] The rollback runtime checkout resolves to the selected historical SHA,
  while both installed host restart controllers remain byte-identical to the
  newer installed launcher. A second rollback invocation must still reach the
  exact-commit compatibility gate.
- [ ] Every contract stage passes.
- [ ] `REHEARSAL_POSTGRES_CONTRACT_OK` appears for both exact launchers.
- [ ] The migration-backed contract proves the pre-128 V1 transport rejection,
  post-128 persistence, exact finalized-allocation view projection, production
  settlement-authority parsing, tampered weight-receipt rejection, and
  declaration coverage for every migration referenced by the required
  schema/RPC preflight.
- [ ] The local PostgREST service reports nonzero
  `migration_backed_relations` and rejects any selected, filtered, ordered, or
  inserted column absent from the PostgreSQL catalog.
- [ ] Candidate forward, historical rollback, and candidate roll-forward use
  one candidate-migrated durable database state; every launcher begins at the
  preceding launcher's exact revision and content hash, at least one durable
  mutation is exercised, and schema/hash tampering is rejected.
- [ ] Zero rejected contract events.
- [ ] Zero `internal_substitution` events. An adapted repository module,
  repository script, or long-lived application process invalidates the
  rehearsal even when its fabricated output has the expected shape.
- [ ] Zero synthetic external fixtures. Boundary adapters must consume
  sanitized production-shaped inputs and independently validate exact argv,
  environment names, schemas, hashes, ordering, and failure behavior.
- [ ] Release SHA and PCR0 are identical across launcher and workflow evidence.
- [ ] Bundle, publication, and finalization receipts form one verified ancestry.
- [ ] Artifact-wrapped attested operations preserve both receipt layers:
  `execution_receipt` verifies against `execution_receipt_graph`, the outer
  lineage `receipt` verifies against the outer `receipt_graph`, and a
  cross-paired receipt/graph is rejected.
- [ ] Primary and auditor canonical vectors are byte-for-byte equal.
- [ ] The signed SDK extrinsic is the one finalized by the local chain.
- [ ] `LastUpdate` equals the finalized block and reveal readback equals the
  canonical vector.
- [ ] The local chain, database, processes, injected faults, and locks are clean
  after completion.
- [ ] The `prepush` profile uses at most 4 CPUs and 7 GiB per container.
- [ ] The `release` profile completes exactly 100 epochs, every configured
  boundary fault, rollback/roll-forward, and the concurrency matrix using at
  most 6 CPUs and 7 GiB per container.

The reduced resource budget is achieved by replacing only privileged external
boundaries (Docker daemon, Nitro, AWS, chain, and Supabase) with strict local
services. Repository-owned launchers, gateway/validator/auditor behavior,
canonicalization, signing, receipt generation, SDK extrinsic construction, and
verification remain candidate production code. Any new internal substitution
must fail both profiles.

### 4. Verify V2 Integrity

Confirm that the candidate contains no unintended changes to:

- [ ] Canonical UIDs or weights.
- [ ] Bundle hashes or signatures.
- [ ] Epoch mapping or exact-block verification.
- [ ] PCR0 or attestation policy.
- [ ] Exact-commit enforcement.
- [ ] Archive endpoint authority.
- [ ] Fail-closed behavior.

Do not push until every pre-push item above passes.

## After Attestation and Restart

Run these checks before resuming production:

```bash
bash scripts/_probe_weight_submission_path_local.sh full
bash scripts/_probe_weight_submission_path_local.sh auditor-bundle
bash scripts/_probe_weight_submission_path_local.sh signing-only
bash scripts/_probe_auditor_weight_submission_ready_local.sh
```

The probes must produce all of the following:

- [ ] `PROBE_SUCCESS`
- [ ] `SIGNING_PROBE_SUCCESS`
- [ ] `AUDITOR_SUBMISSION_READY`
- [ ] `NO_CHAIN_WRITE_CONFIRMED`

Then verify:

- [ ] The current-epoch gateway bundle hash exactly matches the primary
  validator's on-chain vector.
- [ ] The same hash exactly matches every audit validator's on-chain vector.
- [ ] Each expected validator's `last_update` advanced in the current epoch.
- [ ] Gateway and validator logs contain no weight-path errors.
- [ ] No authoritative publication journal was quarantined.
- [ ] The checks remain successful across the next epoch rollover.

Do not resume production until every post-restart item above passes.
