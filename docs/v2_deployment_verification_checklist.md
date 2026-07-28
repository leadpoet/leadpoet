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

On a constrained workstation, cap the primary outer Docker replicas without
changing the enforced production capacity contract:

```bash
python3 scripts/run_local_restart_rehearsal.py \
  --from-sha <currently-deployed-sha> \
  --candidate-sha HEAD \
  --transition forward \
  --profile prepush \
  --outer-cpus 4 \
  --outer-memory 6g
```

The outer limits may only reduce the selected profile's Docker budget. The
strict adapters continue to advertise and validate the production
16-vCPU/128-GiB topology, and reject any launcher request that changes it.
When a Docker runtime such as Colima does not share the operating system's
default temporary directory, set
`LEADPOET_RESTART_REHEARSAL_TEMP_ROOT` to an existing Docker-shared host
directory. The driver validates that root and places every bind-mounted
temporary beneath it; it never falls back after an invalid explicit value.

This is the mandatory 5–10 minute developer gate. It runs in Docker without
production credentials, executes both installed N-1 launchers and the candidate
Git handoff, then runs one complete gateway/validator/auditor publication
against strict local chain and durable-database services.
On ARM developer machines the outer `prepush` container uses native ARM64 to
avoid QEMU overhead; the unchanged launchers still issue and validate the exact
production `linux/amd64` Docker/Nitro contracts. The `release` profile always
runs its outer replica as pinned `linux/amd64` for the final ABI check.

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
unchanged candidate SHA. The rollback target must also pass
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
keep the existing validator running until the gateway finishes its complete
authenticated restart, and release the validator only through the exact-SHA
coordination marker. A failed gateway restart must leave the marker absent,
terminate the waiting SSH process, and preserve the existing validator.

Both profiles must produce a joined
`leadpoet-restart-rehearsal-<sha>-<profile>.json` manifest. The manifest and
console output must prove all of the following:

- [ ] `PYTHON37_FINALIZATION_PROBE_SUCCESS`
- [ ] `REHEARSAL_SUCCESS component=gateway`
- [ ] `REHEARSAL_SUCCESS component=validator`
- [ ] Exact rollback succeeds when the change affects restart or release
  selection.
- [ ] Exact roll-forward succeeds again from the rollback target.
- [ ] The paired operator coordinator completes gateway before validator,
  rejects single-component SHA mismatches, and does not release the validator
  after a gateway failure.
- [ ] The rollback runtime checkout resolves to the selected historical SHA,
  while both installed host restart controllers remain byte-identical to the
  newer installed launcher. A second rollback invocation must still reach the
  exact-commit compatibility gate.
- [ ] Every contract stage passes.
- [ ] Zero rejected contract events.
- [ ] Zero `internal_substitution` events. An adapted repository module,
  repository script, or long-lived application process invalidates the
  rehearsal even when its fabricated output has the expected shape.
- [ ] Zero synthetic external fixtures. Boundary adapters must consume
  sanitized production-shaped inputs and independently validate exact argv,
  environment names, schemas, hashes, ordering, and failure behavior.
- [ ] Release SHA and PCR0 are identical across launcher and workflow evidence.
- [ ] Bundle, publication, and finalization receipts form one verified ancestry.
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
