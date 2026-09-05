# V2 Deployment Verification Checklist

Use this checklist for gateway, validator, auditor, weight, restart, settlement,
attestation, PCR0, or V2 trust-boundary changes.

## Default Gate

The blocking pre-push gate has a 120-second outer deadline. It contains only
deterministic syntax/format checks, directly affected regressions, and one
hermetic complete transition for each release class touched by the diff. A
green process with skipped, collect-only, unexercised, or zero materially
executed required tests is not a pass. Run broad suites and the legacy
`prepush` profile asynchronously after push. Run the unaccelerated profile only
when the operator's current request explicitly uses `un-accelerated` or
`unaccelerated` as the requested test/rehearsal mode. Words such as
"thorough", "full", "production-equivalent", "end-to-end", "all tests", or
"release" do not select the long profile.

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

### 3. Deterministic Changed-Seam Transition

- [ ] Freeze the candidate SHA and exercise deployed N-1 -> candidate for the
  exact production seam changed by the diff.
- [ ] Reach every downstream output of that seam under dependency-minimal,
  production-shaped inputs, including fail-closed negatives and cleanup.
- [ ] Report materially executed test/stage counts; skipped, collect-only, or
  unexercised required stages fail the gate.
- [ ] Finish all blocking checks within 120 seconds.

### 4. Asynchronous Accelerated Production Rehearsal

Start this broad diagnostic after push, concurrently with exact-SHA
attestation. Replace `<deployed-sha>` with the exact production N-1 SHA:

```bash
python3 scripts/run_local_restart_rehearsal.py \
  --from-sha <deployed-sha> \
  --candidate-sha HEAD \
  --transition forward \
  --profile prepush
```

The controller reports each stage duration and has a 600-second target. It
runs one exact forward transition and should prove:

- [ ] Candidate Git tree, source blobs, release artifacts, PCR0, roles, and
  exact commit agree.
- [ ] Installed N-1 gateway and validator launcher paths complete.
- [ ] Candidate migrations apply to disposable PostgreSQL; strict PostgREST
  validates real relation, column, view, constraint, and RPC contracts,
  including the ancestry-bootstrap purpose and contract RPC from migration
  137.
- [ ] Credential-envelope and provider preflight paths complete without
  plaintext fallback.
- [ ] Signed private-model oldest -> newest-reviewed -> oldest transition,
  pointer/source alignment,
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
- [ ] Total elapsed time is at most 600 seconds, or the retained artifact
  identifies the exact timed-out stage.

The controller continues independent stages after a failure and writes one
candidate-bound aggregate ledger. It must never suppress validator or canonical
workflow diagnostics because gateway failed, or vice versa.

Broad-lane status, timeout, missing summary, or a failure before candidate code
does not by itself veto the release. Only an independently actionable retained
artifact proving a candidate product/trust failure blocks. Move every required
invariant unique to a broad lane into the deterministic gate. New or materially
changed broad lanes remain commissioning-only until 20 consecutive exact-main
runs reach candidate code, meet time/cleanup contracts, and retain
machine-readable executed-stage counts and actionable injected-negative
artifacts. Keep flaky or pre-candidate lanes visible and nonblocking while they
are repaired or retired.

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
- [ ] Directly affected regressions and the hermetic changed-seam transition
  pass with nonzero materially executed counts.
- [ ] The combined blocking gate finishes within 120 seconds.
- [ ] Broad suites and `prepush` are queued to run asynchronously after push;
  their status alone is not a veto without an actionable candidate-failure
  artifact.
- [ ] Candidate contains no unintended scoring, reward, payment, allocation,
  emission, canonical-weight, signature, epoch, PCR0, attestation,
  exact-commit, or fail-closed changes.
- [ ] Fetch again and integrate any new `origin/main`.
- [ ] Observe the deployment timing policy in `AGENTS.md`.

## After Attestation and Restart

Run the tracked read-only probe from a pristine worktree whose `HEAD` is the
exact active candidate. The worktree may contain no additional files,
including ignored or untracked files, and the interpreter must provide the
candidate's declared dependencies.

Obtain `FINALIZED_EPOCH_ID` from the first candidate-associated
`GET /weights/current/71` publication after the completed official baseline,
retain that epoch, and wait for its compact authority to finalize and reveal.
Obtain `AUDITOR_HOTKEYS` only from the authoritative active public-auditor
deployment inventory joined one-to-one to post-restart tracked
`startup_ready` records. Do not infer configured auditors from IPs, UIDs,
metagraph membership, historical rows, or vector equality. Fail closed if the
inventory is absent or incomplete.

```bash
auditor_args=()
for hotkey in "${AUDITOR_HOTKEYS[@]}"; do
  auditor_args+=(--auditor-hotkey "$hotkey")
done

/Users/pranav/Downloads/Election_Analysis/Bittensor-subnet/venv/bin/python -I \
  scripts/probe_weight_submission_evidence_v2.py \
  --candidate-sha "$CANDIDATE_SHA" \
  --netuid 71 \
  --epoch-id "$FINALIZED_EPOCH_ID" \
  "${auditor_args[@]}"
```

Require exit zero, schema
`leadpoet.weight_submission_evidence_probe.v2`, exact candidate/netuid/epoch,
`auditor_count` equal to the authoritative inventory, one primary plus every
auditor, identical finalized mechanism-0 vectors, advanced `LastUpdate`
values, and a finalized head past the reveal period. The probe independently
verifies the version-pinned COMPLIANCE Object-Locked release channel, compact
publication authority, signatures, finalization, and chain readback. Confirm
that no publication quarantine or weight-path terminal error remains.
