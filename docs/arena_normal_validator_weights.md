# Normal Arena validators

The primary, Yuma, Rizzo, and other subnet validators use the same
`neurons/validator.py` process with their own local Bittensor hotkey.
Nitro, an enclave, KMS key unwrapping, and the retired auditor client are not
needed to run this process.

## Scoring and authorization

The gateway's existing subnet role rule decides whether a signed hotkey is a
registered validator. Arena applies that shared rule to its finalized chain
snapshot; it does not depend on another gateway process's initialized client.
That check authorizes both claims and scoring results.
Round runner lists do not grant or deny access. Lease ownership, signatures,
stage rules, code review, and miner self-dealing checks remain in place.

The validator pulls assigned miner source, runs the existing Arena ICP and
scoring path, and reports results with its local hotkey. Miner provider
credentials remain in the gateway broker. Validators and submitted models do
not receive the raw keys. The existing gVisor/runsc sandbox is still needed for
scoring; removing Nitro does not remove the model sandbox.

Weights start independently, before scoring setup. An empty queue, missing
runsc, scoring setup error, or scoring-loop error causes scoring to wait/retry,
not the weight loop to stop. Scoring works again when its dependency recovers.

## Weight submission

The validator verifies the gateway signing-key pin and the signed accepted
Arena state, including the governing reward basis, chain identity, subnet,
epoch window, and burn hotkey. It derives the same canonical winner/burn vector
from finalized UID ownership and signs the time-locked commitment with its own
local hotkey. The existing runtime, nonce, mortal transaction, and exact
transaction checks still apply.

No scoring work is needed to submit weights. The gateway publishes the current
epoch's state from the governing reward basis, even when no new model has been
scored. A missing, invalid, conflicting, or expired accepted state stops that
submission. It never becomes an invented burn-only state.

Signed bytes are persisted before broadcast. Restarts and uncertain results
reuse those exact bytes. A fresh attempt requires proof of expiry, absence of
inclusion, and an unchanged nonce. Earlier reveals and report retries do not
block the current epoch. Success requires finalized commitment/reveal readback,
`LastUpdate`, the exact revealed vector, and unchanged rewarded UID ownership.
Commitment inclusion alone is not success.

The local host can now access its own hotkey. Protect that machine and wallet
file. Gateway signatures authenticate Arena decisions; local signing does not
independently prove that Arena accepted the correct scores. No receipt graphs,
release identity, model ancestry, or legacy Research Lab verification is used
to authorize these weights.

## Run a validator

Install the repository's existing Python dependencies in a Python 3.11 virtual
environment. The pinned Bittensor 10.5 runtime supplies drand 2.x; the old
Bittensor 9/drand 1 host environment cannot submit stateful commitments.
Use Linux x86_64 and
the existing runsc setup to score models. For Finney SN71, set the trusted
public configuration:

```bash
export LAB_ARENA_API_BASE_URL=https://gateway.subnet71.com
export LAB_ARENA_CHAIN_ENDPOINT=wss://entrypoint-finney.opentensor.ai:443
export LAB_ARENA_SIGNING_KEY_HASH=sha256:fb0a422d437700f468beda94b4d3e05bb22dbaa6141f0e6c5f1dac9e7257d99a
export LAB_ARENA_VALIDATOR_STATE_DIR="$PWD/validator-state"
export LAB_ARENA_RUNNER_WORK_DIR="$PWD/arena-runner"
export LAB_ARENA_RUNSC_PATH=/usr/local/bin/runsc

python neurons/validator.py \
  --netuid 71 --subtensor.network finney \
  --wallet.name YOUR_WALLET --wallet.hotkey YOUR_HOTKEY \
  --wallet.path "$HOME/.bittensor/wallets"
```

Use the current trusted signing-key pin supplied by the subnet operator if it
changes. Do not blindly accept a key fetched from an untrusted gateway.
The hotkey must already exist as a private regular wallet file (mode 0600).
The validator does not create or replace wallets.

For a service, the equivalent wallet settings are `LAB_ARENA_WALLET_NAME`,
`LAB_ARENA_HOTKEY`, and `LAB_ARENA_WALLET_PATH`.
`LAB_ARENA_EXPECTED_HOTKEY` can additionally bind deployment to an existing
public identity. Use an absolute wallet path in services.
Test networks require their matching epoch mapping and an explicit
`LAB_ARENA_BURN_HOTKEY`.

Append `--check-only` to verify local signing access, the gateway key pin,
and finalized chain identity without broadcasting or claiming work.
`--once` runs one scoring poll and one weight cycle; it is not a substitute
for a continuously supervised process or finalized reveal verification.

Run only one process per hotkey. Keep the same weight state directory across
updates and retries. Never put provider keys or wallet seed values in service
configuration, logs, or Git.

## Deployment

Apply committed migration `208-lab-arena-validator-scoring-authority.sql`
before restarting the gateway to this code. It removes the duplicate SQL
runner-list gate. SQL claims remain restricted to the gateway service role.
Migrations through 207 remain prerequisites. This is not a change to scores,
promotion, rewards, or provider accounting.

Use the canonical gateway restart. For the validator, prepare a mode-0600
environment file with the public configuration and local wallet path, then run
the exact pushed main controller:

```bash
git -C /home/ec2-user/leadpoet/leadpoet fetch --no-tags origin main
git -C /home/ec2-user/leadpoet/leadpoet show "$SHA:validator_restart.sh" \
  | VALIDATOR_DEPLOY_COMMIT="$SHA" bash
```

The controller stages exact committed source, preserves the journal/work
directories, and runs local-wallet readiness before draining the active
service. It then switches the service and retains the old release and private
configuration for rollback. It does not export an enclave key or create a new
hotkey. Use an existing owner-held wallet for the transition.

The primary controller defaults to
`/home/ec2-user/arena-validator-venv311/bin/python3`. Install the committed
`requirements.txt` in that dedicated environment before restarting, or set
`VALIDATOR_PYTHON_BIN` to an already prepared compatible environment. Keep the
old environment intact so rollback can still start the previous service.

The sample systemd service supervises the same normal validator. SIGTERM stops
new claims and drains current work. The old enclave need not be terminated to
run the local path; the controller does not manage unrelated enclave services.

## Validation

Run the focused gate in `docs/v2_deployment_verification_checklist.md`.
It covers local-wallet signing, safe restart/retry, finalized reveals, idle
weight submission, scoring with brokered miner credentials, and registered
validator authorization without runner lists.

Report controlled tests, deployment readiness, and live finalized outcomes
separately. Do not claim Yuma or Rizzo is working solely because the primary is
working; each operator must run the updated process with their own wallet.
