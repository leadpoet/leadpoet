# V2 Deployment Verification Checklist

Use this checklist for gateway, validator, auditor, weight, restart, settlement,
attestation, PCR0, or V2 trust-boundary changes.

## Default Gate

The normal pre-push gate is intentionally bounded to 5-10 minutes. Run the
unaccelerated profile only when the operator's current request explicitly uses
`un-accelerated` or `unaccelerated` as the requested test/rehearsal mode.
Words such as "thorough", "full", "production-equivalent", "end-to-end",
"all tests", or "release" do not select the long profile.

### 1. Synchronize

- [ ] Fetch `origin/main`.
- [ ] Preserve unrelated local work and use a clean worktree when ownership is
  uncertain.
- [ ] Integrate upstream without force-pushing.
- [ ] Freeze one candidate SHA.
- [ ] Review the exact task diff for secrets, unrelated files, and trust-policy
  changes.

### 2. Focused Checks

Run syntax checks and only tests directly affected by the diff. For weight-path
changes, the standard focused set is:

```bash
python3 -m pytest -q \
  tests/test_weight_authority_v2.py \
  tests/test_authoritative_weight_flow_v2.py \
  tests/test_chain_source_v2.py \
  tests/test_validator_publication_recovery_v2.py \
  tests/test_weight_submission_retry.py \
  tests/test_auditor_same_epoch_mirroring.py \
  tests/test_auditor_v2_hardening.py
```

Run independent focused checks in parallel where practical. Do not silently
expand to the full repository suite.

### 2a. Signed Sourcing-Model Contract Transition

For a private sourcing-model contract upgrade, keep the branch-specific model
pointer on the currently deployed artifact until the dual-compatible Leadpoet
candidate is deployed and attested.

- [ ] Verify the immutable old and new manifests and signatures with the
  configured KMS public key.
- [ ] Verify each manifest's Git commit, image digest, source-tree hash,
  compatibility-contract hash, and paired parity-fixture hash.
- [ ] Reject unknown contracts, crossed old/new contract and parity pairs,
  pointer/source mismatches, and signature or manifest tampering.
- [ ] Confirm the candidate accepts both exact reviewed versions while the
  current production release accepts its existing version unchanged.
- [ ] Confirm provider projection keeps the model-owned binding-manifest digest
  separate from provisioning-row provenance and carries no endpoint,
  credential, or provider response into the model contract.
- [ ] Confirm no database migration is required. Contract, binding, and active
  model identities remain inside existing append-only, hash-bound JSON
  documents unless a separately reviewed physical query requires a column.
- [ ] Exercise old -> new -> old contract activation and rebenchmarking with
  the dual-compatible candidate before changing the live model pointer.

The safe forward order is Leadpoet candidate push and attestation, coordinated
gateway/validator deployment, exact old-artifact weight verification, then an
operator fast-forward of the model's protected Lab branch. After the pointer
advances, require active-model supersession, rebenchmark, conditional-ICP,
scoring, allocation, primary/auditor bundle equality, submission,
finalization, `LastUpdate`, and reveal/readback evidence.

Rollback is pointer first, code second. Create normal signed model revert
commits in newest-first order until the tree exactly matches the reviewed old
contract; never force-reset the protected model branch. Verify the
old-equivalent artifact under the dual-compatible Leadpoet release before
considering a rollback to older strict consumer code.

### 2b. Bounded Ancestry Restart Rollout

Gateway releases containing the measured ancestry-bootstrap operation require
`scripts/138-research-lab-ancestry-checkpoint-bootstrap-purpose.sql` before the
restart. The pre-shutdown Supabase contract probe fails closed if that
migration, its validated role/purpose constraint, or its service-role-only RPC
is missing.

- [ ] Keep every immutable receipt, edge, boot identity, transport root, and
  host-operation root in the append-only store. A checkpoint replaces only the
  restart transport of already-verified ancestry; it never deletes authority
  or changes scoring/allocation inputs.
- [ ] On the first rollout, expect the N-1 coordinator to report the new
  operation as explicitly `unsupported`. This is the only tolerated
  pre-checkpoint result; the candidate coordinator must then validate the full
  active legacy roots, sign one bounded recursive proof per selected root,
  persist it, and pass exact durable readback before weight preparation.
- [ ] On subsequent rollouts, require the old gateway's release manifest,
  `/build-info`, V2 health, coordinator boot identity, release/PCR0
  commitments, and ancestry lineage to agree before candidate host code may
  request pre-checkpointing. The work runs at low CPU/I/O priority during
  attestation acquisition and is joined before shutdown.
- [ ] Require the candidate-side checkpoint verification on every restart,
  even when pre-checkpointing passed. It must reselect active allocation and
  sourcing roots, prove the epoch/root set is stable, and find no remaining
  full legacy root before authoritative weight preparation.
- [ ] Treat invalid health, boot/release mismatch, changed roots, proof
  omission/reordering, signature mismatch, persistence mismatch, or missing
  readback as fatal. Do not turn these into cache misses or broad retries.
- [ ] Review the `ancestry-checkpoint.log` and restart timing events
  `ancestry_precheckpoint_*` and `ancestry_postcheckpoint_*` to distinguish
  selection, proof loading, enclave execution, persistence, and readback.
- [ ] Exact-release EIF archive restore is eligible only when commit, source,
  dependency lock, role/image identity, release manifest, and PCR0 all verify.
  It accelerates a same-SHA retry or retained rollback; a new SHA still cold
  builds, and any present-but-invalid archive fails closed rather than falling
  back silently. Retain exactly the current release plus two predecessors;
  promote an older release only after exact installed readback. Rollback copies
  must complete before replacement, and cleanup may remove only aged, unopened
  interrupted staging directories.

Offline artifact preparation may overlap the release wait, but it must own a
cancelable process group and join successfully before shutdown. Downloads and
temporary verification remain parallel; final shared-cache publication and
readback must hold the Docker operation lock used by live PCR0 consumers. Release
approval, lineage verification, credential envelopes, prelaunch validation,
largest-first Nitro startup, direct authority verification, and the independent
HTTP validator handoff remain mandatory and serialized at their trust
boundaries.

### 3. Accelerated Production Rehearsal

Replace `<deployed-sha>` with the exact production N-1 SHA:

```bash
python3 scripts/run_local_restart_rehearsal.py \
  --from-sha <deployed-sha> \
  --candidate-sha HEAD \
  --transition forward \
  --profile prepush
```

The controller reports each stage duration and enforces a 600-second target.
It runs one exact forward transition and must prove:

- [ ] Candidate Git tree, source blobs, release artifacts, PCR0, roles, and
  exact commit agree.
- [ ] Installed N-1 gateway and validator launcher paths complete.
- [ ] Candidate migrations apply to disposable PostgreSQL; strict PostgREST
  validates real relation, column, view, constraint, and RPC contracts,
  including the ancestry-bootstrap purpose and contract RPC from migration
  137.
- [ ] Credential-envelope and provider preflight paths complete without
  plaintext fallback.
- [ ] Signed private-model v7 -> v8 -> v7 transition, pointer/source alignment,
  exact contract/parity pairing, and KMS verification complete; hybrid,
  unknown, and tampered artifacts fail closed.
- [ ] Gateway builds one canonical bundle.
- [ ] Primary and audit validators receive byte-identical vectors.
- [ ] Parsing, verification, SDK signing, submission, finalization,
  `LastUpdate`, reveal/readback, and cleanup complete.
- [ ] Git-tree replacement, conditional-ICP policy, settlement authority, and
  protected workflows satisfy the candidate-derived behavioral contract.
- [ ] Every declared stage appears once as `passed`; no critical stage is
  `failed` or `unexercised`.
- [ ] Total elapsed time is at most 600 seconds.

The controller continues independent stages after a failure and writes one
candidate-bound aggregate ledger. It must never suppress validator or canonical
workflow diagnostics because gateway failed, or vice versa.

If this gate is expected to exceed ten minutes, tell the operator immediately.
Do not start the long profile as a substitute.

## Explicit Unaccelerated Gate

Run this only after an explicit operator request for the un-accelerated test:

```bash
python3 scripts/run_local_restart_rehearsal.py \
  --from-sha <deployed-sha> \
  --candidate-sha HEAD \
  --transition forward \
  --profile unaccelerated
```

This is the former long release rehearsal, unchanged in scope:

- Pinned `linux/amd64` outer replica.
- Forward, rollback, and roll-forward with installed launchers.
- One candidate-migrated durable state across all transitions.
- Complete external-boundary fault and concurrency matrix.
- Exactly 100 accelerated epochs.
- Aggregate failure ledger and final joined evidence.

The old `--profile release` spelling is deliberately rejected so the long run
cannot be selected accidentally.

## Rehearsal Integrity

- Repository production shell and Python paths are authoritative. Adapt only
  Docker, Nitro, AWS, Supabase, and chain-write boundaries.
- Generate stages, policy sizes, ICP splits, tree topology, and legal state
  spaces from the frozen candidate. Do not maintain duplicate fixed values in
  the runner or evidence join.
- Apply migration files verbatim and exercise nonempty production-shaped
  settlement, bundle, publication, finalization, and provider rows.
- Require zero internal substitutions, unknown boundary calls, synthetic
  successes, duplicate stages, receipt cross-pairing, or stale identities.
- Preserve one receipt ancestry from execution through publication and
  finalization.
- Require byte-identical primary/auditor vectors and exact finalized-chain
  readback.
- Every process, client, container, injected fault, and lock must be cleaned.

## Before Push

- [ ] `git diff --check` passes.
- [ ] Touched Python files compile.
- [ ] Focused tests pass.
- [ ] `prepush` rehearsal passes within its time budget.
- [ ] Candidate contains no unintended scoring, reward, payment, allocation,
  emission, canonical-weight, signature, epoch, PCR0, attestation,
  exact-commit, or fail-closed changes.
- [ ] Fetch again and integrate any new `origin/main`.
- [ ] Observe the deployment timing policy in `AGENTS.md`.

## After Attestation and Restart

Run read-only production probes:

```bash
bash scripts/_probe_weight_submission_path_local.sh full
bash scripts/_probe_weight_submission_path_local.sh auditor-bundle
bash scripts/_probe_weight_submission_path_local.sh signing-only
bash scripts/_probe_auditor_weight_submission_ready_local.sh
```

Require:

- [ ] `PROBE_SUCCESS`, `SIGNING_PROBE_SUCCESS`,
  `AUDITOR_SUBMISSION_READY`, and `NO_CHAIN_WRITE_CONFIRMED`.
- [ ] Gateway, primary validator, and every auditor report the same current
  epoch bundle/vector hash.
- [ ] Expected `last_update` values advance and finalized chain readback
  matches.
- [ ] No publication journal quarantine or weight-path error remains.
- [ ] Evidence stays valid across the next epoch rollover.
