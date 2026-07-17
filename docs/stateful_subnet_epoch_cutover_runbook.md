# Stateful subnet epoch cutover runbook

This runbook migrates LeadPoet SN71 from the historical global
`block // 360` key to Bittensor's stateful subnet scheduler without reusing an
existing integer epoch key. The official identity is `SubnetEpochIndex`, the
epoch anchor is `LastEpochBlock`, and LeadPoet's existing integer columns become
a compatibility `settlement_epoch_id` derived from one immutable cutover
mapping.

The cutover is intentionally fail closed. Do not improvise a new first epoch,
edit the manifest, clear the fence, or skip to a later official epoch after the
namespace has been fenced.

## Runtime authority

The implementation follows the current Subtensor runtime, not the obsolete
globally aligned modulo schedule:

- `SubnetEpochIndex` identifies an official subnet epoch.
- `LastEpochBlock` anchors its start.
- `PendingEpochAt`, `BlocksSinceLastStep`, `Tempo`, and the runtime safety limit
  determine the earliest possible next transition.
- Historical boundary verification uses the official archive endpoint
  `wss://archive.chain.opentensor.ai:443`.
- Live admission and weight timing use coherent best/finalized Finney snapshots.

Primary references:

- Bittensor storage API: <https://docs.learnbittensor.org/subtensor-api/storage>
- Bittensor network concepts: <https://docs.learnbittensor.org/concepts/bittensor-networks>
- Pinned Subtensor scheduler source used for this release:
  <https://github.com/opentensor/subtensor/blob/19a6485969253ee9756bc382bac536e0fcb0b72f/pallets/subtensor/src/coinbase/run_coinbase.rs>

Every storage field used in one decision must be read at the same block hash.

## Hard prerequisites

Do not start the ceremony until all of these are true:

1. The migration release is committed and pushed in both the main repository
   and `subnet_dashboard`.
2. The gateway and validator have first been restarted on that release in
   `legacy_global_360_v1` mode.
3. The gateway checkout is clean. `gw_restart.sh` rejects a dirty checkout.
4. SQL migrations 100 and 101 have passed in production.
5. A real V2 weight bundle, publication, and finalized-chain receipt exist for
   the intended last legacy epoch and validator hotkey.
6. The independently rebuilt validator release manifest and gateway V2 release
   manifest are available and verified.
7. Operators can stop every gateway and validator writer before the global
   legacy bucket rolls over.
8. There is enough time to activate and restart before block 300 of the first
   official stateful epoch. Weight submission begins at block 345.

If any prerequisite is false, remain in legacy mode. Deploying the schema and
code does not itself activate stateful epochs.

## 1. Apply the additive Supabase migrations

Use a direct PostgreSQL connection with a database owner/migration role. Do not
put credentials in shell history, source control, or this runbook.

Migration 100 contains `CREATE INDEX CONCURRENTLY` and must run outside an
explicit transaction. Migration 101 is transactional.

```bash
cd /path/to/Bittensor-subnet

psql "$SUPABASE_DB_URL" \
  -v ON_ERROR_STOP=1 \
  -f scripts/100-stateful-subnet-epoch-high-water-indexes.concurrent.sql

psql "$SUPABASE_DB_URL" \
  -v ON_ERROR_STOP=1 \
  -f scripts/101-stateful-subnet-epoch-authority.sql
```

Verify that the lifecycle remains open and that no mapping was accidentally
activated:

```sql
SELECT lifecycle_state, mapping_hash, network_genesis_hash, netuid,
       last_legacy_epoch_id, first_settlement_epoch_id
FROM public.research_lab_stateful_subnet_epoch_cutover_state_v1;

SELECT public.research_lab_stateful_subnet_epoch_cutover_public_state_v1();
```

Expected before the ceremony: `legacy_open`, with no active mapping.

## 2. Deploy the release in legacy mode first

All gateway, validator, auditor, qualification, fulfillment, scoring, and
Research Lab processes must initially use:

```text
LEADPOET_EPOCH_MODE=legacy_global_360_v1
```

Do not set either cutover-manifest variable yet:

```text
LEADPOET_SUBNET_EPOCH_CUTOVER_JSON
LEADPOET_SUBNET_EPOCH_CUTOVER_PATH
```

Restart only in a safe epoch window, before block 300. Use the repository's
normal gateway and validator restart procedures; do not substitute an rsync
deployment. After restart, verify loaded commits, process start times, resolved
module paths, PCR0s, and new V2 receipts produced by the running release.

## 3. Select and fence the final legacy key

Choose the current legacy `block // 360` key as `LAST_LEGACY_EPOCH_ID`. Its
immediate successor is reserved as `FIRST_SETTLEMENT_EPOCH_ID`. Run the fence
early enough that no process can create the reserved key before the ceremony.
The RPC measures every physical Supabase epoch-key column under locks and fails
if the proposed high-water or vacancy is wrong.

```bash
export NETWORK_GENESIS_HASH='0x...'
export NETUID='71'
export LAST_LEGACY_EPOCH_ID='...'
export FIRST_SETTLEMENT_EPOCH_ID="$((LAST_LEGACY_EPOCH_ID + 1))"

python3 -m gateway.research_lab.stateful_epoch_cutover_cli_v1 \
  --fence-before-boundary \
  --network-genesis-hash "$NETWORK_GENESIS_HASH" \
  --netuid "$NETUID" \
  --last-legacy-epoch-id "$LAST_LEGACY_EPOCH_ID" \
  --first-settlement-epoch-id "$FIRST_SETTLEMENT_EPOCH_ID" \
  --confirm-first-settlement-epoch-id "$FIRST_SETTLEMENT_EPOCH_ID" \
  --acknowledge-reserved-first-settlement-ordinal
```

The durable state must read `cutover_fenced`. Existing writes at or below the
last legacy key remain valid; every ordinary write at or above the reserved
first settlement key is rejected.

There is no rollback command after this point.

## 4. Finalize the last legacy weight and stop writers

Allow the last legacy epoch's normal V2 weight flow to complete at block 345 or
later. Confirm the exact bundle joins to its publication and finalized-chain
receipt in `research_lab_finalized_allocation_epochs_v2`.

Before the legacy global bucket rolls over, stop every writer that can allocate
an epoch key: gateway API/workers, Research Lab workers, qualification and
fulfillment workers, primary validator, and auditors. Confirm there are no
in-flight jobs. Leave them stopped through staging and activation.

If the final V2 chain receipt is absent, stop. A log line, V1 bundle, or
allocation snapshot is not sufficient proof.

## 5. Propose the archive-derived manifest at the reserved boundary

Wait for the first exact official SN71 boundary that lies inside the reserved
global bucket. The proposal command discovers the latest finalized boundary on
the official archive, re-reads that exact block, validates the durable fence,
and prints the canonical manifest.

```bash
python3 -m gateway.research_lab.stateful_epoch_cutover_cli_v1 \
  --propose-manifest \
  --network-genesis-hash "$NETWORK_GENESIS_HASH" \
  --netuid "$NETUID" \
  --last-legacy-epoch-id "$LAST_LEGACY_EPOCH_ID" \
  > /secure/operator/stateful-epoch-manifest-report.json

jq '.manifest' \
  /secure/operator/stateful-epoch-manifest-report.json \
  > /secure/operator/stateful-epoch-cutover.json

jq -r '.manifest.mapping_hash' \
  /secure/operator/stateful-epoch-manifest-report.json
```

Treat the manifest and mapping hash as immutable. The proposal is archive-first
and does not depend on a candidate row.

## 6. Capture and ingest the validator candidate offline

On the validator host, use the approved six-build validator release manifest,
the enclave-backed hotkey, and the canonical cutover manifest to produce one
signed candidate file. Keep the gateway writers stopped.

```bash
python3 -m validator_tee.host.subnet_epoch_boundary_capture_v2 \
  --cutover-manifest /secure/operator/stateful-epoch-cutover.json \
  --validator-release-manifest /secure/operator/validator-v2-release-manifest.json \
  --settlement-epoch-id "$FIRST_SETTLEMENT_EPOCH_ID" \
  --candidate-output /secure/operator/stateful-epoch-candidate.json \
  --wallet-name "$VALIDATOR_WALLET_NAME" \
  --wallet-hotkey "$VALIDATOR_HOTKEY_NAME"
```

On the gateway host, validate without writes first. Record the returned
`candidate_payload_hash`, then repeat with its exact confirmation:

```bash
python3 -m gateway.research_lab.stateful_epoch_candidate_ingest_cli_v1 \
  --candidate /secure/operator/stateful-epoch-candidate.json \
  --validator-release-manifest /secure/operator/validator-v2-release-manifest.json

python3 -m gateway.research_lab.stateful_epoch_candidate_ingest_cli_v1 \
  --candidate /secure/operator/stateful-epoch-candidate.json \
  --validator-release-manifest /secure/operator/validator-v2-release-manifest.json \
  --apply \
  --confirm-candidate-payload-hash 'sha256:...'
```

The apply result must say `durably_staged` and its mapping, boundary, receipt,
and durable readback hashes must match the dry run.

## 7. Stage, then activate, the cutover

Run the cutover CLI once without `--apply`; inspect every reported hash. Then
stage it using the exact mapping hash and approved gateway release manifest.

```bash
python3 -m gateway.research_lab.stateful_epoch_cutover_cli_v1 \
  --manifest /secure/operator/stateful-epoch-cutover.json

python3 -m gateway.research_lab.stateful_epoch_cutover_cli_v1 \
  --manifest /secure/operator/stateful-epoch-cutover.json \
  --release-manifest /secure/operator/gateway-v2-release-manifest.json \
  --apply \
  --confirm-mapping-hash 'sha256:...' \
  --confirm-all-writers-stopped
```

The durable lifecycle must now be `stateful_staged`. Record the returned
`cutover_authority_hash`. Activation is a separate explicit operation:

```bash
python3 -m gateway.research_lab.stateful_epoch_cutover_cli_v1 \
  --manifest /secure/operator/stateful-epoch-cutover.json \
  --release-manifest /secure/operator/gateway-v2-release-manifest.json \
  --activate-staged \
  --confirm-mapping-hash 'sha256:...' \
  --confirm-cutover-authority-hash 'sha256:...' \
  --confirm-all-writers-stopped \
  --confirm-stateful-release-prepared
```

Activation is allowed only while the finalized chain is still in the exact
first official cutover epoch and `epoch_block < 300`. At block 300 or in a
later official epoch it fails closed.

## 8. Start all runtimes on the same authority

Configure every gateway, validator, and auditor process with exactly one copy
of the immutable manifest:

```text
LEADPOET_EPOCH_MODE=stateful_v1
LEADPOET_SUBNET_EPOCH_CUTOVER_PATH=/secure/operator/stateful-epoch-cutover.json
```

Do not also set `LEADPOET_SUBNET_EPOCH_CUTOVER_JSON`.

Restart gateway and validator before official epoch block 300. Start ancillary
workers only after the primary authority is healthy. Validate that every
process reports the same network genesis hash, netuid, mapping hash, official
`SubnetEpochIndex`, `epoch_ref`, and compatibility settlement ordinal.

## 9. Production proof checklist

Code-level tests and a successful restart are not production proof. Before
declaring the migration complete, verify all of the following:

1. Public cutover RPC says `stateful_active` and exposes only its documented
   sanitized fields.
2. Protected Supabase rows contain exactly one mapping/cutover lineage and no
   reserved-key collision.
3. Gateway and validator loaded the intended commits and manifest hash.
4. Gateway builder PCR0, cached validator PCR0, validator live enclave PCR0,
   and validator release manifest agree.
5. A current-official-epoch V2 bundle contains the official epoch identity and
   mapped settlement key.
6. That bundle joins to a publication and finalized-chain event.
7. Primary validator on-chain `last_update` advanced for the current epoch.
8. Each auditor produced and submitted the same current-epoch vector; historical
   transport or bundle checks alone do not prove the auditor path.
9. Champion rewards, reimbursement rewards, Research Lab allocation, score
   bundle fetch/unpack, weight mutation, and chain submission all have fresh
   post-cutover evidence.
10. The admin dashboard's official epoch card agrees with the same-hash chain
    storage snapshot. A small difference from validator logs is expected when
    the dashboard shows best head and validator authority shows finalized head.

## Failure handling

- Before fencing: remain in legacy mode and correct the prerequisite.
- After fencing but before activation: keep all writers stopped and preserve
  every manifest, candidate, receipt, and report. Do not write the reserved
  ordinal through another path.
- Missed first-epoch block-300 deadline: stop. There is no approved skip-forward
  or rollback ceremony in this release.
- Supabase outage or lifecycle mismatch: all stateful writes and weight
  submissions must remain fail closed.
- Archive disagreement or unavailable historical proof: stop; never replace the
  official archive with a live/pruned node for historical authority.
