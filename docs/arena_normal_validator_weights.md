# Normal Arena validators

The Arena service accepts competition results and publishes a signed reward
state. Each normal validator reads that state and finalized chain data, then
constructs its own weight vector. A protected signer checks the state, chain
scope, UID ownership, and exact transaction before it signs. Chain observations
are stored separately from the reward decision.
An observation does not settle or reduce an outstanding reward obligation.

The same validator process scores miner submissions through the existing Arena
lease and completion API. Provider calls use the miner's credentials through
the existing broker. The scoring process must not load a plaintext validator
hotkey. It uses the protected application's claim and completion signatures.

## Trust boundary

Arena remains trusted to accept scores and choose the winner. Independent
weight calculation does not make an incorrect accepted score correct. The
signed accepted state makes the allocation decision explicit and reproducible.
The protected signer limits what a compromised validator host can sign.

A state is identified by network, genesis, subnet, epoch, and its content hash.
One immutable state is accepted per epoch. A changed governing reward basis is
a conflict, not a replacement. Missing inputs stop weight construction. They
must not become an empty reward or a burn-only vector.

Recipients are hotkeys. Validators resolve current UID ownership from finalized
chain state. An unregistered fixed recipient burns its allocation. An
unregistered Arena winner follows the existing Arena rule: its share returns
to the fulfillment pool. Fulfillment requests retain their absolute shares,
are reduced proportionally only when they exceed the available pool, and burn
the unused pool. The existing u16 conversion is retained.

The first transition preserves existing fulfillment and obligation economics.
A gateway database adapter reads accepted fulfillment consensus and the
current accepted obligation allocation. Only plain recipient amounts enter
the new state. Receipt graphs, model manifests, ancestry, and audit-validator
roles are not part of the normal validator's weight input. This adapter is a
temporary migration bridge; removing the remaining obligation producer needs
an explicit decision about any unpaid obligations.
The bridge currently requires a current-epoch allocation snapshot produced by
the legacy validator allocation request. There is no background producer for
that snapshot. Thus this bridge is a cutover blocker: the last legacy producer
cannot be retired while the new state still reads it. Do not enable an all-normal
validator deployment until the legacy-obligation policy and its replacement
input source are complete. The new route does not call the old attested builder.

## Process and restart

`LEADPOET_WEIGHT_MODE=arena` selects the normal path in `neurons/validator.py`.
The default remains the legacy path during migration. All validators that
join the new mechanism use this same Arena mode.

Apply `scripts/202-arena-accepted-weight-state.sql` through the normal migration
process before enabling Arena weights. This adds the accepted-state and outcome
tables. It keeps the core Arena schema at version 197, so the previous scoring
service can still run during rollback. The separate weight-state capability is
version 202.

`scripts/run_arena_validator.py` loads a private environment file as data. It
does not execute shell assignments. It accepts `LAB_ARENA_*`, `ENCLAVE_CID`,
and the mode setting. Keep the file mode at `0600`.

Required settings include:

```dotenv
LEADPOET_WEIGHT_MODE=arena
LAB_ARENA_API_BASE_URL=https://your-arena-host.example
LAB_ARENA_CHAIN_ENDPOINT=wss://your-finalized-chain-endpoint.example
LAB_ARENA_SIGNING_KEY_HASH=sha256:<trusted-Arena-public-key-hash>
LAB_ARENA_NETWORK=finney
LAB_ARENA_NETUID=71
LAB_ARENA_VALIDATOR_STATE_DIR=/var/lib/leadpoet/arena-validator
LAB_ARENA_RUNNER_WORK_DIR=/var/lib/lab-arena/runner
LAB_ARENA_RUNSC_PATH=/usr/local/bin/runsc
LAB_ARENA_HOTKEY_ENVELOPE=/home/ec2-user/.config/leadpoet/arena-hotkey-envelope.json
```

The normal validator gets its public hotkey from the protected signer. It does
not need a wallet seed file. Use the existing sandbox configuration for the
scorer. Do not place miner provider keys in this file.

The dedicated `validator_tee/Dockerfile.arena-signer` image admits only Arena
RPCs. The seed and its allowed public policy are sealed together with KMS.
The policy binds the Arena signing key and HTTPS host, chain and subnet,
epoch mapping, burn identity, and transaction rules. A MAC from the seed also
protects the policy inside the KMS recipient envelope. A parent process cannot
replace that policy while retaining the validator's key.

Prepare the encrypted envelope once from the owner's protected seed and
reviewed public policy with
`python3 -m validator_tee.host.arena_hotkey_bootstrap seal`. The normal validator host
receives only the encrypted envelope. On a fresh enclave boot,
`LAB_ARENA_HOTKEY_ENVELOPE` permits automatic KMS recipient provisioning. KMS
must permit the approved Arena signer image and enforce recipient-only
decryption through [KMS recipient attestation](https://docs.aws.amazon.com/kms/latest/APIReference/API_Decrypt.html).
The old raw-seed envelope format is deliberately rejected by this
new mode. No GitHub commit or gateway release history is part of the policy.

The validator owns the chain relay on vsock port 5002 and the Arena relay on
port 5003. TLS terminates inside the enclave. A host relay can interrupt reads,
but it cannot impersonate the configured Arena or chain endpoint. Stop the old
host relay during the explicit migration to the new service; two processes
must not share ownership of these ports.

The sample `deploy/leadpoet-arena-validator.service` supervises one normal
validator process. Set its interpreter and checkout paths to the installed
environment. It runs readiness checks before start. Before stopping a working
service, run the same check directly:

```bash
python3 scripts/run_arena_validator.py \
  --environment-file /home/ec2-user/.config/leadpoet/arena-validator.env \
  --check-only
```

Once installed, restart the host process with the normal systemd command. A
SIGTERM stops new claims and lets work in progress drain. Preserve the state
directory across updates. Signed bytes are persisted before broadcast, and a
retry uses those exact bytes. A submitted transaction and a finalized, revealed
weight vector are different outcomes; readiness alone proves neither.
Each epoch has its own journal. An older reveal or outcome-report retry does
not prevent a new epoch from submitting. Outcome delivery can resume after a
gateway outage. It does not need a new weight transaction.

Do not run a legacy validator and an Arena validator with the same hotkey at
the same time. The host service restart does not rebuild or replace the
protected signer. A signer image change still needs its measured-image and
key-unsealing checks.

## Legacy API retirement

The legacy `/weights` API remains enabled during migration. This includes the
primary publication API used by old audit clients. New normal validators use
`/arena/v1/weight-state` and `/arena/v1/chain-outcomes`.

Retirement is a local operator control read on each request. Write `retired`
to the root-owned file
`/home/ec2-user/.config/leadpoet/legacy-audit-weights.retired` with file mode
`0644`. The gateway must be able to read this public flag; only root can write
the default file. The path can be overridden by
`LEADPOET_LEGACY_AUDIT_WEIGHT_RETIREMENT_FILE`. The next legacy request returns
HTTP 410. This does not need a gateway restart or code change. Retire the old
routes only after their clients have moved to normal Arena mode. Preserve
their existing code and records until that cutover is complete.

## Validation limits

Local tests exercise the scoring, reward-state, and submission logic using
disposable PostgreSQL and controlled provider and chain boundaries. They do not
prove a new Nitro image can unseal a production key, or that a live chain has
accepted and revealed weights. Record those as separate deployment checks.
