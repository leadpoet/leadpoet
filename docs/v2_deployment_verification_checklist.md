# Arena reward and weight verification

Use this gate for changes to scoring, accepted reward state, weight construction,
local-wallet signing, retries, restart recovery, or chain readback. The Arena flow
has one normal validator implementation. Retired primary/audit validator and
Research Lab allocation rehearsals are not release requirements.

## Before push

Fetch the intended base and preserve concurrent work. Test the edited checkout.
Run syntax checks and `git diff --check`. Use Python 3.11 for validator tests.

The blocking local gate has a 120-second budget. It must execute the relevant
checks successfully; a timeout or an all-skipped run is not a pass. Keep local
provider and chain boundaries controlled. Do not wait through real epochs or
broadcast manual weights as a local release test.

```bash
python3.11 -m pytest -q \
  tests/test_arena_weights.py \
  tests/test_arena_validator.py \
  tests/test_local_weight_signer.py \
  tests/test_arena_validator_local_runtime.py \
  tests/test_arena_validator_launcher.py \
  tests/test_arena_validator_restart.py \
  tests/test_arena_reveal_chain_source.py \
  tests/test_registry_roles.py \
  tests/test_validate_miner_testnet.py \
  tests/lab_arena/test_lab_arena_chain.py \
  tests/lab_arena/test_validator_eligibility.py \
  tests/lab_arena/test_validator_stake_flow.py \
  tests/lab_arena/test_lab_arena_store.py \
  tests/lab_arena/test_scorer_image_access.py \
  tests/lab_arena/test_leased_images.py \
  tests/lab_arena/test_scorer_delivery_roundtrip.py \
  tests/lab_arena/test_lab_arena_service_rules.py \
  tests/lab_arena/test_arena_weight_state.py \
  tests/lab_arena/test_lab_arena_normal_weight_flow.py
```

The gate must prove:

- Miner models use their own brokered provider credentials, without secret
  values entering stored artifacts, results, or logs.
- An uncached private scorer image downloads under an active validator lease
  without validator AWS credentials. Exact image/blob hashes and size limits
  remain enforced, and temporary download URLs do not enter logs or disk.
- Standard scoring and promotion produce the governing Arena reward basis.
- Accepted state is immutable, signed, current, and bound to the intended
  network, genesis, subnet, and epoch. Conflicts and missing state stop signing.
- Independent validators derive the same Arena winner and burn weights using
  finalized UID ownership. Retired incentive tables and allocations are absent.
- The local signer constrains the exact transaction, nonce, runtime,
  validity window, and rewarded UID ownership.
- Persisted bytes are reused after restart and unknown submission results.
  A new attempt requires proof that the prior attempt expired without inclusion.
- Chain readback proves the revealed vector and its exact transition. A
  commitment or a missing pending commitment alone does not prove success.
- Delayed outcome reports and older epoch recovery do not change rewards or
  prevent the current epoch from progressing.
- Scoring setup/cycle failures and claim denials do not stop weights. New
  mainnet execute/score claims use the shared gateway rule: registration, a
  validator permit, and effective stake >=75,000, regardless of activity or
  runner lists. Testnet retains its active-or-permitted policy without the
  mainnet minimum. Existing leases do not gain a stake minimum at completion.
  Capacity counts eligible planned runners with the same shared rule.

For database changes, apply the exact migration to disposable PostgreSQL.
Exercise upgrades with representative old objects and dependency constraints,
repeat the migration, and verify that Arena and generic qualification data
survive. Do not apply production SQL during the local gate.

For changed generic gateway services, run their focused regression tests as
well. Keep the protected workflow manifest reproducible from the committed
source. A manifest generated from an uncommitted working tree is not sufficient.

## Deployment evidence

Code push and deployment are separate actions. Follow the authorized deployment
scope. Before replacing a working process, run the normal validator's
`--check-only` command and preserve its state directory.

The normal validator requires no enclave image or KMS recipient policy. Apply
migration 208 before deploying the new scoring authorization. Keep the existing
hotkey identity and state directory; do not export enclave key material. After
an authorized deployment, record actual
finalized chain readback for each configured normal validator. Report local
tests, provisioning, and live chain results separately. Never claim live weight
submission from readiness or an HTTP acknowledgement alone.

For the first transition, apply additive migration 202 while the gateway is
running. Invoke the exact `origin/main` transition wrapper documented in
`docs/arena_normal_validator_weights.md`. It installs the gateway-only
controller and holds the canonical restart lock. The restart stops the old
producers and then waits at its exact migration 203 barrier. Apply the exact
203 SQL. For the 2026-09-10 transition, apply exact migration 204 while that
barrier remains held; it changes only the reviewed open round's runner list and
fails if the frozen configuration differs. Apply migration 205 before releasing
the barrier because current execution leases query optional credential
availability even for submissions without that credential. Then verify the 203
capability RPC and use the exact completion helper. The
helper checks the live capability and the candidate, SQL hash, and invocation
binding against the protected persistent gateway environment before the
restart can activate the new gateway. The temporary parent environment has
already been scrubbed at this point. Do not write the completion marker by
hand.

If migration 203 succeeds but a later restart stage fails, rerun the ordinary
canonical exact-commit restart. Migration 203 is idempotent, the new schema
preflight is then fully enabled, and no legacy incentive table is required.
