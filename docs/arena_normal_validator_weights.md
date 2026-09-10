# Normal Arena validators

Arena / Open Source Agent Competition is the only emission incentive. Arena
accepts competition scores and publishes the signed reward state. Every normal
validator scores miner submissions through the standard lease and completion
API, then independently derives weights from that state and finalized chain
inputs. Miner provider credentials stay in the existing credential broker.

The accepted state contains the signed Arena reward basis, chain identity,
subnet, epoch, validity window, and burn hotkey. It contains no reimbursement,
legacy champion, SOURCE_ADD, fulfillment, or leaderboard allocations. The Arena
reward share, decay, promotion threshold, and effective epoch rules are
unchanged. The eligible registered Arena winner receives its share. All
remaining weight goes to the registered burn hotkey. A missing or invalid
accepted state stops submission; it never becomes a fabricated burn-only state.

## Trust and signing

Arena is trusted to accept scores and choose the winner. Independent weight
construction makes that accepted decision reproducible; it does not independently
prove that Arena accepted the correct scores.

One immutable state is accepted per network, genesis, subnet, and epoch. A
conflicting governing reward basis cannot replace it. Validators resolve hotkeys
to UIDs from finalized chain state. The protected verifier checks the signed
state, epoch window, UID ownership, nonce, runtime, and exact transaction. It
then signs only the permitted mortal time-locked weight commitment.

The dedicated `validator_tee/Dockerfile.arena-signer` image exposes only Arena
RPCs. Its measured image contains the reviewed public policy, which binds the
Arena signing key and HTTPS host, chain and subnet, epoch mapping, burn hotkey,
transaction profile, and drand library. KMS unwraps the existing encrypted
validator seed only to an attested recipient inside that measured image. The
enclave verifies that the derived hotkey matches its policy. The host cannot
read the seed or replace the policy while retaining the key.

TLS to Arena and the chain terminates inside the protected signer. The host
relays can interrupt traffic but cannot substitute an authenticated response.
The scorer obtains narrowly scoped claim and completion signatures from the
same protected service. It does not load a plaintext validator hotkey.

No GitHub commit identity, model manifest, receipt graph, historical ancestry,
or audit-validator role is required by the Arena weight path. Existing generic
qualification and artifact verification services retain their own evidence
checks. Those services do not construct Arena weights.

## Persistence and recovery

The host persists signed transaction bytes before broadcast. Retries and restart
recovery use those exact bytes. A new attempt is permitted only after the
protected verifier proves expiry, absence of the previous transaction, and an
unchanged nonce. Attempts are bounded and the old journals remain available.

Each epoch has a separate journal. An older reveal or report retry does not
block a new epoch. Final success requires chain readback of the time-locked
reveal, the exact weight vector, and unchanged rewarded UID ownership at that
transition. Commitment inclusion alone is insufficient.

Chain outcomes are separate signed observations. They do not modify the reward
state or maintain a reimbursement ledger. Delayed report delivery can resume
after a gateway outage without signing another weight transaction.

## Installation and restart

Apply additive migration `scripts/202-arena-accepted-weight-state.sql` before
the first restart. Then run the exact pushed `main` controller transition:

```bash
git -C /home/ec2-user/leadpoet_repo fetch --no-tags origin main
test "$(git -C /home/ec2-user/leadpoet_repo rev-parse origin/main)" = "$SHA"
git -C /home/ec2-user/leadpoet_repo show \
  "$SHA:scripts/transition_gateway_controller_and_restart_v1.sh" \
  | bash -s -- --commit "$SHA"
```

The transition keeps the canonical restart lock from controller installation
through restart. After it stops all old gateway incentive producers, it writes
the fixed migration barrier and waits. Apply the exact migration 203 from the
same commit. The completion helper checks the live 203 capability and binds the
completion to the candidate, SQL hash, and restart invocation before startup
continues. Run the helper from the exact Git object and use the protected
persistent gateway environment, which remains available after the temporary
parent environment is scrubbed:

```bash
git -C /home/ec2-user/leadpoet_repo show "$SHA:scripts/203-retire-legacy-incentive-weight-bridge.sql" > "/tmp/203-$SHA.sql"
git -C /home/ec2-user/leadpoet_repo show "$SHA:scripts/complete_gateway_migration_203_barrier.py" \
  | python3 - \
      --barrier /home/ec2-user/.config/leadpoet/migration-203-barrier.json \
      --completion /home/ec2-user/.config/leadpoet/migration-203-complete.json \
      --sql "/tmp/203-$SHA.sql" --commit "$SHA" \
      --env-file /home/ec2-user/.config/leadpoet/gateway.env
```

The helper does not print credentials. Later restarts use the normal canonical
command; migration 203 is idempotent and the special barrier is not required.

`neurons/validator.py` starts the normal Arena validator. The sample
`deploy/leadpoet-arena-validator.service` supervises that same implementation.
Set its interpreter and checkout paths for the installed environment.
`scripts/run_arena_validator.py` reads a mode-0600 environment file as data.

Required settings include:

```dotenv
LAB_ARENA_API_BASE_URL=https://your-arena-host.example
LAB_ARENA_CHAIN_ENDPOINT=wss://your-finalized-chain-endpoint.example
LAB_ARENA_SIGNING_KEY_HASH=sha256:<trusted-Arena-public-key-hash>
LAB_ARENA_NETWORK=finney
LAB_ARENA_NETUID=71
LAB_ARENA_VALIDATOR_STATE_DIR=/var/lib/leadpoet/arena-validator
LAB_ARENA_RUNNER_WORK_DIR=/var/lib/lab-arena/runner
LAB_ARENA_RUNSC_PATH=/usr/local/bin/runsc
LAB_ARENA_HOTKEY_ENVELOPE=/home/ec2-user/.config/leadpoet/validator-hotkey-envelope-v2.json
```

Use the existing sandbox settings for scoring. Do not put miner provider keys
or the validator seed in this environment file.

Keep the existing KMS-encrypted validator hotkey envelope as the durable boot
input. Build the signer with the reviewed public policy set through
`VALIDATOR_ARENA_SIGNER_POLICY_INPUT`. On every fresh enclave boot, the host
sends the existing KMS ciphertext and the enclave's signed recipient document
to KMS. KMS must permit the approved signer PCR0 and require recipient-only
decryption. The returned recipient ciphertext is opened only inside the
enclave and is accepted only when its hotkey matches the measured policy. This
transition does not require the owner seed on the host or a newly sealed copy.

Before stopping a working service, run its local readiness check:

```bash
python3 scripts/run_arena_validator.py \
  --environment-file /home/ec2-user/.config/leadpoet/arena-validator.env \
  --check-only
```

Preserve the validator state directory across updates. SIGTERM stops new claims
and drains current work. The validator owns chain relay port 5002 and Arena
relay port 5003. One process must own each port. Do not run an old validator
and the Arena validator with the same hotkey at the same time.

The legacy `/weights` and `/research-lab/allocations/*` routes, old audit client,
allocation producers, automatic validator Git rebuilds, and paired weight
restart machinery are removed from this release. There is no retirement flag
or fallback. A code push does not change the currently deployed old release.

## Validation

The local end-to-end gate uses disposable PostgreSQL and controlled model,
provider, and chain boundaries. It exercises scoring, winner selection,
accepted state, weight construction, protected submission, restart recovery,
and chain readback. Production Nitro key unsealing and a live chain reveal
remain deployment checks; local success does not prove either occurred.
