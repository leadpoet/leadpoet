# Arena reward and weight verification

Use this gate for changes to scoring, accepted reward state, weight construction,
protected signing, retries, restart recovery, or chain readback. The Arena flow
has one normal validator implementation. Retired primary/audit validator and
Research Lab allocation rehearsals are not release requirements.

## Before push

Fetch the intended base and preserve concurrent work. Test the edited checkout.
Run syntax checks and `git diff --check`. Use Python 3.11 for host tests and
check the measured signer sources for Python 3.7 compatibility.

The blocking local gate has a 120-second budget. It must execute the relevant
checks successfully; a timeout or an all-skipped run is not a pass. Keep local
provider and chain boundaries controlled. Do not wait through real epochs or
broadcast manual weights as a local release test.

```bash
python3.11 -m pytest -q \
  tests/test_arena_weights.py \
  tests/test_arena_validator.py \
  tests/test_arena_protected_hotkey.py \
  tests/test_arena_hotkey_bootstrap.py \
  tests/test_arena_validator_launcher.py \
  tests/test_arena_reveal_chain_source.py \
  tests/lab_arena/test_arena_weight_state.py \
  tests/lab_arena/test_lab_arena_normal_weight_flow.py
```

The gate must prove:

- Miner models use their own brokered provider credentials, without secret
  values entering stored artifacts, results, or logs.
- Standard scoring and promotion produce the governing Arena reward basis.
- Accepted state is immutable, signed, current, and bound to the intended
  network, genesis, subnet, and epoch. Conflicts and missing state stop signing.
- Independent validators derive the same Arena winner and burn weights using
  finalized UID ownership. Retired incentive tables and allocations are absent.
- The protected signer constrains the exact transaction, nonce, runtime,
  validity window, and rewarded UID ownership.
- Persisted bytes are reused after restart and unknown submission results.
  A new attempt requires proof that the prior attempt expired without inclusion.
- Chain readback proves the revealed vector and its exact transition. A
  commitment or a missing pending commitment alone does not prove success.
- Delayed outcome reports and older epoch recovery do not change rewards or
  prevent the current epoch from progressing.

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

A protected signer image change requires the measured image policy and KMS
recipient-unsealing checks. After an authorized deployment, record actual
finalized chain readback for each configured normal validator. Report local
tests, provisioning, and live chain results separately. Never claim live weight
submission from readiness or an HTTP acknowledgement alone.

For the first transition, install and verify the gateway-only restart controller
from the exact release before invoking it. The installed legacy controller can
still require deleted auditor artifacts. Keep that mismatch fail-closed and
leave the working service running until the controller upgrade is complete.
