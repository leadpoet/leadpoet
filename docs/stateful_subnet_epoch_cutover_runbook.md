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
2. The exact official-only gateway and validator release has been independently
   built and published, but has not been restarted yet. The currently running
   pre-cutover release remains responsible for finalizing the last historical
   settlement key until writers are stopped.
3. The gateway checkout is clean. `gw_restart.sh` rejects a dirty checkout.
4. SQL migrations 100, 101, and 105 have passed in production.
5. Historical allocation classification coverage is complete, and at least one
   finalized historical allocation has a durable coordinator receipt proving
   the signed validator vector, finalized chain equality, signed audit event,
   and immutable checkpoint. Its proven epoch may be below the namespace
   high-water; the two values are deliberately separate.
6. The independently rebuilt validator release manifest and gateway V2 release
   manifest are available and verified.
7. Operators can stop every gateway and validator writer before the global
   legacy bucket rolls over.
8. There is enough time to activate at or before block 300 and start each
   restart at or before block 300 of the first official stateful epoch. Weight
   submission begins at block 345.

If any prerequisite is false, keep the current pre-cutover runtimes unchanged.
Deploying the schema and pushing code does not itself activate stateful epochs.

## 1. Apply the additive Supabase migrations

Use a direct PostgreSQL connection with a database owner/migration role. Do not
put credentials in shell history, source control, or this runbook.

Migration 100 contains `CREATE INDEX CONCURRENTLY` and must run outside an
explicit transaction. Do not run it through the Supabase SQL Editor: the
Dashboard request can time out while PostgreSQL is still building, and a
cancelled concurrent build can leave an invalid same-name index that
`IF NOT EXISTS` will not repair. Use a persistent direct connection or the
Supavisor **session-mode** endpoint on port 5432, never transaction mode on
port 6543. Migration 101 is transactional.

Before retrying any timed-out build, inspect active progress. If this returns a
row, wait; do not drop or recreate that index:

```sql
SELECT progress.pid,
       table_relation.oid::pg_catalog.regclass AS table_name,
       index_relation.oid::pg_catalog.regclass AS index_name,
       progress.phase,
       progress.blocks_done,
       progress.blocks_total
FROM pg_catalog.pg_stat_progress_create_index AS progress
JOIN pg_catalog.pg_class AS table_relation
  ON table_relation.oid = progress.relid
LEFT JOIN pg_catalog.pg_class AS index_relation
  ON index_relation.oid = progress.index_relid
WHERE progress.datname = pg_catalog.current_database()
ORDER BY progress.pid;
```

When no build is active, `indisvalid = false` or `indisready = false` can be an
interrupted build remnant. `indislive = false` can instead mean that a
concurrent drop is still in progress. Before changing any such index, inspect
`pg_stat_activity` and `pg_locks` for an active `CREATE INDEX`, `REINDEX`, or
`DROP INDEX` touching the exact index/table and wait for that DDL to finish.
Only after the catalog state is stable should you drop a confirmed invalid
remnant using a standalone
`DROP INDEX CONCURRENTLY public.<exact_name>;`, then rerun its exact standalone
`CREATE INDEX CONCURRENTLY` statement from migration 100. Never wrap either
statement in `BEGIN`, a `DO` block, or a function. A missing name only needs its
exact `CREATE` statement. `research_lab_epoch_payouts` is an ordinary derived
view and is intentionally not indexed; its physical source identities are
covered by the catalog catch-all.

A same-name index can also have all three flags true while targeting the wrong
table/key or using the wrong method, ordering, expression, predicate, included
columns, or operator class. In that case `IF NOT EXISTS` will never repair it.
Use migration 100's full `index_definition`/`actual_table_name` report and exact
validation error to identify the mismatch. After the same active-DDL checks,
drop only the confirmed wrong same-name index concurrently and run its canonical
standalone `CREATE INDEX CONCURRENTLY` statement from migration 100.

```bash
cd /path/to/Bittensor-subnet

psql "$SUPABASE_DB_URL" \
  -v ON_ERROR_STOP=1 \
  -f scripts/100-stateful-subnet-epoch-high-water-indexes.concurrent.sql
```

Migration 100's final `DO` block proves the exact catalog contract: each named
index belongs to the expected table, has the expected sole key/expression,
uses the built-in default B-tree operator class and canonical descending order,
and has no unexpected predicate, `INCLUDE` column, or uniqueness property. The
`indisvalid`, `indisready`, and `indislive` flags alone are not sufficient.

Those catalog checks cannot detect damaged B-tree pages or missing heap tuples.
Before the production fence, discover whether PostgreSQL's `amcheck` extension
is available and enabled:

```sql
SELECT available.name,
       available.default_version,
       installed.extversion AS installed_version,
       extension_namespace.nspname AS installed_schema
FROM pg_catalog.pg_available_extensions AS available
LEFT JOIN pg_catalog.pg_extension AS installed
  ON installed.extname = available.name
LEFT JOIN pg_catalog.pg_namespace AS extension_namespace
  ON extension_namespace.oid = installed.extnamespace
WHERE available.name = 'amcheck';
```

If `installed_version` is null, enable `amcheck` through the Supabase Database
Extensions page or the approved migration role, then rerun the discovery query.
Run the following through the same persistent direct/session-mode connection in
an off-peak window. It checks both B-tree structure and that every visible heap
tuple has a matching index tuple. `heapallindexed => true` is intentionally the
stronger and more I/O-intensive form; it retains `AccessShareLock`-level
relation locking but can take several times longer than the structural-only
check.

```sql
BEGIN READ ONLY;
SET LOCAL lock_timeout = '2s';
SET LOCAL statement_timeout = '30min';

DO $amcheck$
DECLARE
    amcheck_schema NAME;
    target_index RECORD;
BEGIN
    SELECT extension_namespace.nspname
    INTO amcheck_schema
    FROM pg_catalog.pg_extension AS extension_meta
    JOIN pg_catalog.pg_namespace AS extension_namespace
      ON extension_namespace.oid = extension_meta.extnamespace
    WHERE extension_meta.extname = 'amcheck';

    IF amcheck_schema IS NULL THEN
        RAISE EXCEPTION 'amcheck is not installed';
    END IF;

    FOR target_index IN
        WITH RECURSIVE accepted_index_roots(index_oid) AS (
            SELECT DISTINCT index_relation.oid
            FROM pg_catalog.pg_class AS relation
            JOIN pg_catalog.pg_namespace AS relation_namespace
              ON relation_namespace.oid = relation.relnamespace
            JOIN pg_catalog.pg_attribute AS column_meta
              ON column_meta.attrelid = relation.oid
            JOIN pg_catalog.pg_index AS index_meta
              ON index_meta.indrelid = relation.oid
            JOIN pg_catalog.pg_class AS index_relation
              ON index_relation.oid = index_meta.indexrelid
            JOIN pg_catalog.pg_am AS access_method
              ON access_method.oid = index_relation.relam
            JOIN pg_catalog.pg_opclass AS operator_class
              ON operator_class.oid = index_meta.indclass[0]
            WHERE relation_namespace.nspname = 'public'
              AND relation.relkind IN ('r', 'p')
              AND column_meta.attnum > 0
              AND NOT column_meta.attisdropped
              AND column_meta.atttypid IN (20, 21, 23)
              AND column_meta.attname IN (
                  'epoch', 'epoch_id', 'evaluation_epoch'
              )
              AND index_relation.relkind IN ('i', 'I')
              AND access_method.amname = 'btree'
              AND index_meta.indisvalid
              AND index_meta.indisready
              AND index_meta.indislive
              AND index_meta.indpred IS NULL
              AND index_meta.indexprs IS NULL
              AND index_meta.indnkeyatts >= 1
              AND index_meta.indkey[0] = column_meta.attnum
              AND operator_class.opcdefault
              AND operator_class.opcmethod = index_relation.relam
              AND operator_class.opcintype = column_meta.atttypid
        ), accepted_index_tree(index_oid) AS (
            SELECT index_oid
            FROM accepted_index_roots
            UNION ALL
            SELECT inheritance.inhrelid
            FROM accepted_index_tree AS parent_index
            JOIN pg_catalog.pg_inherits AS inheritance
              ON inheritance.inhparent = parent_index.index_oid
        ), physical_index_targets(index_oid) AS (
            SELECT DISTINCT index_tree.index_oid
            FROM accepted_index_tree AS index_tree
            JOIN pg_catalog.pg_class AS index_relation
              ON index_relation.oid = index_tree.index_oid
            WHERE index_relation.relkind = 'i'
            UNION
            SELECT 'public.idx_transparency_log_payload_epoch_identity_v1'::pg_catalog.regclass::OID
        )
        SELECT physical_target.index_oid,
               physical_target.index_oid::pg_catalog.regclass::TEXT AS index_name
        FROM physical_index_targets AS physical_target
        ORDER BY index_name
    LOOP
        RAISE NOTICE 'amcheck: %', target_index.index_name;
        EXECUTE pg_catalog.format(
            'SELECT %I.bt_index_check(index => $1, heapallindexed => true)',
            amcheck_schema
        ) USING target_index.index_oid::pg_catalog.regclass;
    END LOOP;
END;
$amcheck$;

COMMIT;
```

This intentionally checks every physical B-tree accepted by the migration's
catalog catch-all, including pre-existing source-table indexes behind derived
views, plus the JSONB payload expression index. An empty successful result
(only the per-index notices) is the expected result.
Any exception, timeout, or unavailable extension is an unresolved physical-
integrity evidence gap; do not describe the indexes as physically verified or
run the production fence until it is resolved. `amcheck` greatly strengthens
the proof but cannot establish the absolute absence of every possible storage
or hardware fault. This result is point-in-time evidence. Rerun the same
`heapallindexed => true` block immediately before the step 3 fence, with
epoch-key writers quiesced if operationally possible, and record the check time
and output; writes after a check are not covered by that check.

After the exact catalog validation and physical check pass, apply migrations
101 and 105:

```bash
psql "$SUPABASE_DB_URL" \
  -v ON_ERROR_STOP=1 \
  -f scripts/101-stateful-subnet-epoch-authority.sql

psql "$SUPABASE_DB_URL" \
  -v ON_ERROR_STOP=1 \
  -f scripts/105-stateful-subnet-epoch-historical-predecessor-v2.sql
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

## 2. Build the official-only release without restarting it

The attested V2 release workflow builds and publishes the exact commit on every
push to `main`. Confirm that the exact-commit release channel exists before
fencing or stopping any writer. Do not restart the gateway, validator, or
auditors onto the official-only release before activation because that release
fails closed without the activated cutover manifest.

Do not set either cutover-manifest variable on a running pre-cutover process:

```text
LEADPOET_SUBNET_EPOCH_CUTOVER_JSON
LEADPOET_SUBNET_EPOCH_CUTOVER_PATH
```

The cutover manifest is configured only after staging and immediately before
activation. The normal restart scripts consume the same persisted environment
on every later restart.

## 3. Select and fence the final legacy key

Choose the final historical settlement key as `LAST_LEGACY_EPOCH_ID`. Its
immediate successor is reserved as `FIRST_SETTLEMENT_EPOCH_ID`. This is a
one-time historical namespace bridge, not a runtime epoch calculation or
fallback. Run the fence early enough that no process can create the reserved
key before the ceremony. The RPC measures every physical Supabase epoch-key
column under locks and fails if the proposed high-water or vacancy is wrong.

Immediately before invoking the fence, rerun the migration-100 exact catalog
validator and the full `amcheck` block from step 1. Quiesce epoch-key writers
during those checks and the fence call if operationally possible. Do not use an
older point-in-time check as proof of fence-time physical integrity.

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

## 4. Verify the historical predecessor and stop writers

Run the existing V2 readiness repair and require
`historical_classification_coverage = 1.0` with an empty
`missing_historical_classifications` list. The stage command selects the newest
durable row in
`research_lab_legacy_finalized_allocation_migrations_v2` at or below
`LAST_LEGACY_EPOCH_ID`; it does not accept a host assertion or a manual
on-chain submission as predecessor authority.

Before the legacy global bucket rolls over, stop every writer that can allocate
an epoch key: gateway API/workers, Research Lab workers, qualification and
fulfillment workers, primary validator, and auditors. Confirm there are no
in-flight jobs. Leave them stopped through staging and activation.

If classification coverage is incomplete or no attested historical predecessor
exists, stop. A log line, unclassified bundle, manual submission, or allocation
snapshot is not sufficient proof.

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

## 7. Stage and activate inside the measured gateway restart

Install the exact same manifest with mode `0600` at the canonical path on the
gateway and validator hosts before restarting:

```text
/home/ec2-user/.config/leadpoet/stateful-epoch-cutover.json
```

The canonical restart scripts load that path automatically, remove any
competing inline manifest setting, and let the runtime verify the manifest
against the durable receipt-backed authority. Future restarts therefore do not
require a Secrets Manager mutation or another cutover command.

Stop the primary validator and every auditor before starting the gateway
ceremony. Then invoke the canonical gateway restart exactly once with the
explicit ceremony flag:

```bash
cd /home/ec2-user
GATEWAY_STATEFUL_CUTOVER_CEREMONY=1 \
  bash /home/ec2-user/gw_restart.sh
```

The restart performs a read-only eligibility pass before stopping the gateway.
After the exact independently built coordinator enclave is running, it stages
the cutover with the attested historical predecessor, verifies the durable
`stateful_staged` readback, performs the separately authorized activation, and
requires `stateful_active` before continuing to weight-input repair or gateway
launch. The command does not bypass the release manifest, PCR0, Nitro,
credential, receipt-graph, mapping-hash, stopped-writer, initialization, or
Supabase lifecycle checks.

The ceremony flag is deliberately not persisted. Never supply it on a normal
restart. A replay after activation is accepted only when the exact same mapping
and authority are already `stateful_active`; a different mapping fails closed.

Activation is allowed only while the finalized chain is still in the exact
first official cutover epoch and `epoch_block <= 300`. After block 300 or in a
later official epoch it fails closed.

## 8. Start all runtimes on the same authority

Every gateway, validator, and auditor process must load the same immutable
manifest from its canonical host path. The runtime has no epoch-mode switch:

```text
LEADPOET_SUBNET_EPOCH_CUTOVER_PATH=/home/ec2-user/.config/leadpoet/stateful-epoch-cutover.json
```

Do not also set `LEADPOET_SUBNET_EPOCH_CUTOVER_JSON`.

Start the gateway and validator restarts at or before official epoch block 300.
The restarts may complete after block 300. Start ancillary workers only after
the primary authority is healthy. Validate that every
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

- Before fencing: leave the current pre-cutover runtimes unchanged and correct
  the prerequisite; do not deploy an alternate epoch mode.
- After fencing but before activation: keep all writers stopped and preserve
  every manifest, candidate, receipt, and report. Do not write the reserved
  ordinal through another path.
- Missed first-epoch block-300 deadline: stop. There is no approved skip-forward
  or rollback ceremony in this release.
- Supabase outage or lifecycle mismatch: all stateful writes and weight
  submissions must remain fail closed.
- Archive disagreement or unavailable historical proof: stop; never replace the
  official archive with a live/pruned node for historical authority.
