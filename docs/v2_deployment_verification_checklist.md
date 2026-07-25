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
  --scope exact \
  --component all
```

The rehearsal must produce all of the following:

- [ ] `PYTHON37_FINALIZATION_PROBE_SUCCESS`
- [ ] `REHEARSAL_SUCCESS component=gateway`
- [ ] `REHEARSAL_SUCCESS component=validator`
- [ ] Every contract stage passes.
- [ ] Zero rejected contract events.
- [ ] Zero `internal_substitution` events. An adapted repository module,
  repository script, or long-lived application process invalidates the
  rehearsal even when its fabricated output has the expected shape.
- [ ] Zero synthetic external fixtures. Boundary adapters must consume
  sanitized production-shaped inputs and independently validate exact argv,
  environment names, schemas, hashes, ordering, and failure behavior.
- [ ] The isolated runtime has the production 16-vCPU/128-GiB gateway resource
  profile. A reduced developer profile is valid only for targeted regressions.

Targeted launcher regressions are useful but are not deployment evidence. For
example, the artifact-persistence restart matrix runs the exact N-1 launcher
through the real candidate readiness module while adapting unrelated stages:

```bash
python3 scripts/run_local_restart_rehearsal.py \
  --from-sha <currently-deployed-sha> \
  --candidate-sha HEAD \
  --scope weight-readiness-regression \
  --component gateway
```

This command must print `TARGETED_RESTART_REGRESSION_*`, never
`REHEARSAL_SUCCESS`, and cannot satisfy this checklist.

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
