#!/usr/bin/env python3
"""Rehearse the stateful-epoch staging RPC against the real migrations.

Spins up a disposable PostgreSQL cluster, applies the actual cutover
migrations (101, 105, 106, 107, 109, 110, 111 and optionally 112), stubs
only the tables owned by other migrations, seeds one internally consistent
historical-predecessor fixture, and calls
``research_lab_stateful_subnet_epoch_stage_v2`` with a cutover row shaped
exactly like ``build_cutover_row_v1`` output (canonical ``...Z``
``first_observed_at``).

Usage:
    python3 scripts/rehearse_stateful_epoch_stage_v2.py            # with 112
    python3 scripts/rehearse_stateful_epoch_stage_v2.py --skip-112 # regression

Exit codes: 0 = staged (or, with --skip-112, reproduced the known
row-shape rejection); 1 = anything else. Requires initdb/pg_ctl/psql
(``--pgbin`` or PATH). The rehearsal validates SQL semantics only — it does
not attest receipts or exercise PostgREST.
"""

from __future__ import annotations

import argparse
import atexit
import json
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
MIGRATIONS = [
    "101-stateful-subnet-epoch-authority.sql",
    "105-stateful-subnet-epoch-historical-predecessor-v2.sql",
    "106-repair-stateful-epoch-fence-trigger-coverage.sql",
    "107-stateful-epoch-cutover-rpc-timeouts.sql",
    "109-scope-stateful-epoch-transparency-high-water.sql",
    "110-qualify-stateful-epoch-v2-binding.sql",
    "111-refresh-unactivated-stateful-epoch-fence.sql",
]
CANONICALIZATION = "112-canonicalize-cutover-observed-at.sql"

H = lambda tag: "sha256:" + (tag * 64)[:64]  # noqa: E731
GENESIS = "0x" + ("2f" * 32)
BLOCK_HASH = "0x" + ("dc" * 32)
OBSERVED_AT = "2026-07-21T12:46:40Z"
NETUID = 71
CUTOVER_BLOCK = 8669916
TEMPO = 360
NEXT_EPOCH_BLOCK = CUTOVER_BLOCK + TEMPO
SUBNET_EPOCH_INDEX = 24018
SETTLEMENT = 24073
LAST_LEGACY = 24072
PREDECESSOR_EPOCH = 23980
HOTKEY = "5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9"

STUBS = """
CREATE TABLE IF NOT EXISTS public.transparency_log (
    id BIGSERIAL PRIMARY KEY,
    event_type TEXT NOT NULL,
    actor_hotkey TEXT NOT NULL,
    nonce UUID NOT NULL UNIQUE,
    ts TIMESTAMPTZ NOT NULL,
    payload_hash TEXT NOT NULL,
    build_id TEXT NOT NULL,
    signature TEXT NOT NULL,
    payload JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS public.research_lab_attested_execution_receipts_v2 (
    receipt_hash TEXT PRIMARY KEY,
    role TEXT NOT NULL,
    purpose TEXT NOT NULL,
    epoch_id INTEGER NOT NULL,
    receipt_status TEXT NOT NULL,
    output_root TEXT NOT NULL,
    receipt_doc JSONB NOT NULL DEFAULT '{}'::jsonb
);
CREATE TABLE IF NOT EXISTS public.research_lab_attested_weight_bundles_v2 (
    bundle_hash TEXT PRIMARY KEY,
    netuid INTEGER,
    epoch_id BIGINT,
    validator_hotkey TEXT,
    bundle_doc JSONB NOT NULL DEFAULT '{}'::jsonb
);
CREATE TABLE IF NOT EXISTS public.research_lab_attested_weight_finalizations_v2 (
    weight_finalization_event_hash TEXT PRIMARY KEY,
    bundle_hash TEXT,
    finalization_receipt_hash TEXT,
    weight_submission_event_hash TEXT,
    finalized_block BIGINT,
    finalization_doc JSONB NOT NULL DEFAULT '{}'::jsonb
);
CREATE TABLE IF NOT EXISTS public.research_lab_attested_publication_events_v2 (
    weight_submission_event_hash TEXT PRIMARY KEY,
    bundle_hash TEXT
);
CREATE TABLE IF NOT EXISTS public.research_lab_legacy_allocation_nonfinalizations_v2 (
    netuid INTEGER NOT NULL,
    epoch_id INTEGER NOT NULL,
    nonfinalization_doc JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS public.research_lab_legacy_finalized_allocation_migrations_v2 (
    netuid INTEGER NOT NULL,
    epoch_id INTEGER NOT NULL,
    allocation_hash TEXT NOT NULL,
    settlement_hash TEXT NOT NULL,
    settlement_receipt_hash TEXT PRIMARY KEY,
    settlement_doc JSONB NOT NULL
);
-- Prod defines this in the attested-store migration; keep its append-only
-- semantics so rehearsal inserts behave like production.
CREATE OR REPLACE FUNCTION public.prevent_research_lab_attested_v2_mutation()
RETURNS trigger LANGUAGE plpgsql AS $fn$
BEGIN
    IF TG_OP IN ('UPDATE', 'DELETE') THEN
        RAISE EXCEPTION '% rows are append-only', TG_TABLE_NAME;
    END IF;
    RETURN NEW;
END;
$fn$;
DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'service_role') THEN
        CREATE ROLE service_role NOLOGIN;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'anon') THEN
        CREATE ROLE anon NOLOGIN;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'authenticated') THEN
        CREATE ROLE authenticated NOLOGIN;
    END IF;
END $$;
"""


def snapshot_doc() -> dict:
    return {
        "schema_version": "leadpoet.subnet_epoch_snapshot.v1",
        "epoch_scheme": "bittensor.subnet_epoch_index.v1",
        "network_genesis_hash": GENESIS,
        "netuid": NETUID,
        "head_kind": "finalized",
        "block_hash": BLOCK_HASH,
        "current_block": CUTOVER_BLOCK,
        "last_epoch_block": CUTOVER_BLOCK,
        "pending_epoch_at": 0,
        "subnet_epoch_index": SUBNET_EPOCH_INDEX,
        "tempo": TEMPO,
        "blocks_since_last_step": 0,
        "observed_at": OBSERVED_AT,
        "epoch_id": SUBNET_EPOCH_INDEX,
        "epoch_ref": H("e"),
        "epoch_block": 0,
        "next_epoch_block": NEXT_EPOCH_BLOCK,
        "blocks_remaining": NEXT_EPOCH_BLOCK - CUTOVER_BLOCK,
        "settlement_epoch_id": SETTLEMENT,
        "cutover_mapping_hash": H("a"),
    }


def manifest_doc() -> dict:
    return {
        "schema_version": "leadpoet.subnet_epoch_cutover.v1",
        "epoch_scheme": "bittensor.subnet_epoch_index.v1",
        "network_genesis_hash": GENESIS,
        "netuid": NETUID,
        "cutover_block": CUTOVER_BLOCK,
        "cutover_block_hash": BLOCK_HASH,
        "first_subnet_epoch_index": SUBNET_EPOCH_INDEX,
        "first_settlement_epoch_id": SETTLEMENT,
        "last_legacy_epoch_id": LAST_LEGACY,
        "mapping_hash": H("a"),
    }


def authority_doc() -> dict:
    return {
        "schema_version": "leadpoet.subnet_epoch_cutover_authority.v2",
        "mapping_hash": H("a"),
        "first_epoch_ref": H("e"),
        "first_snapshot_hash": H("5"),
        "first_snapshot_receipt_hash": H("b"),
        "predecessor_kind": "legacy_finalized_chain_migration_v2",
        "predecessor_epoch_id": PREDECESSOR_EPOCH,
        "predecessor_allocation_hash": H("1"),
        "predecessor_authority_hash": H("2"),
        "predecessor_receipt_hash": H("3"),
        "manifest": manifest_doc(),
    }


def cutover_row() -> dict:
    """Mirror build_cutover_row_v1's historical-predecessor output."""

    return {
        "cutover_authority_hash": H("c"),
        "schema_version": "leadpoet.subnet_epoch_cutover_authority.v2",
        "mapping_hash": H("a"),
        "manifest_schema_version": "leadpoet.subnet_epoch_cutover.v1",
        "epoch_scheme": "bittensor.subnet_epoch_index.v1",
        "previous_epoch_scheme": "legacy_global_360_v1",
        "network_genesis_hash": GENESIS,
        "netuid": NETUID,
        "cutover_block": CUTOVER_BLOCK,
        "cutover_block_hash": BLOCK_HASH,
        "first_subnet_epoch_index": SUBNET_EPOCH_INDEX,
        "first_epoch_ref": H("e"),
        "first_settlement_epoch_id": SETTLEMENT,
        "last_legacy_epoch_id": LAST_LEGACY,
        "first_tempo": TEMPO,
        "first_pending_epoch_at": 0,
        "first_blocks_since_last_step": 0,
        "first_next_epoch_block": NEXT_EPOCH_BLOCK,
        "first_observed_at": OBSERVED_AT,
        "first_snapshot_hash": H("5"),
        "first_snapshot_receipt_hash": H("b"),
        "predecessor_kind": "legacy_finalized_chain_migration_v2",
        "predecessor_epoch_id": PREDECESSOR_EPOCH,
        "predecessor_allocation_hash": H("1"),
        "predecessor_authority_hash": H("2"),
        "predecessor_receipt_hash": H("3"),
        "last_legacy_bundle_hash": None,
        "last_legacy_weight_finalization_event_hash": None,
        "last_legacy_finalization_receipt_hash": None,
        "cutover_receipt_hash": H("d"),
        "manifest_doc": manifest_doc(),
        "first_snapshot_doc": snapshot_doc(),
        "authority_doc": authority_doc(),
    }


def initialization_event() -> dict:
    return {
        "event_type": "EPOCH_INITIALIZATION",
        "actor_hotkey": "system",
        "nonce": str(uuid.uuid4()),
        "ts": "2026-07-21T12:52:40+00:00",
        "payload_hash": "f" * 64,
        "build_id": "rehearsal",
        "signature": "system",
        "payload": {
            "epoch_id": SETTLEMENT,
            "epoch_key_semantics": "settlement_ordinal",
            "epoch_authority": snapshot_doc(),
            "epoch_boundaries": {
                "start_block": CUTOVER_BLOCK,
                "end_block": NEXT_EPOCH_BLOCK,
                "expected_end_block": NEXT_EPOCH_BLOCK,
                "pending_epoch_at": 0,
                "tempo": TEMPO,
            },
        },
    }


def fixture_sql() -> str:
    doc = json.dumps(snapshot_doc())
    return f"""
UPDATE public.research_lab_stateful_subnet_epoch_cutover_state_v1
SET lifecycle_state = 'cutover_fenced',
    mapping_hash = '{H("a")}',
    network_genesis_hash = '{GENESIS}',
    netuid = {NETUID},
    last_legacy_epoch_id = {LAST_LEGACY},
    first_settlement_epoch_id = {SETTLEMENT},
    candidate_snapshot_hash = '{H("5")}',
    candidate_receipt_hash = '{H("b")}',
    last_legacy_finalization_receipt_hash = '{H("3")}',
    cutover_authority_hash = '{H("c")}',
    fenced_at = NOW(),
    updated_at = NOW()
WHERE singleton;

INSERT INTO public.research_lab_attested_execution_receipts_v2
    (receipt_hash, role, purpose, epoch_id, receipt_status, output_root, receipt_doc)
VALUES
    ('{H("b")}', 'validator_weights', 'validator.subnet_epoch_snapshot.v2',
     {SETTLEMENT}, 'succeeded', '{H("5")}', '{{}}'),
    ('{H("3")}', 'gateway_coordinator',
     'research_lab.legacy_finalized_allocation.v2',
     {PREDECESSOR_EPOCH}, 'succeeded', '{H("2")}', '{{}}'),
    ('{H("d")}', 'gateway_coordinator', 'research_lab.subnet_epoch_cutover.v2',
     {SETTLEMENT}, 'succeeded', '{H("c")}',
     '{{"parent_receipt_hashes": ["{H("b")}", "{H("3")}"]}}');

INSERT INTO public.research_lab_legacy_finalized_allocation_migrations_v2
    (netuid, epoch_id, allocation_hash, settlement_hash,
     settlement_receipt_hash, settlement_doc)
VALUES ({NETUID}, {PREDECESSOR_EPOCH}, '{H("1")}', '{H("4")}', '{H("3")}',
        '{{"chain_target_block": 8600000}}');

INSERT INTO public.research_lab_stateful_subnet_epoch_candidates_v1
    (snapshot_hash, schema_version, mapping_hash, epoch_scheme,
     network_genesis_hash, netuid, head_kind, block_hash, current_block,
     last_epoch_block, pending_epoch_at, subnet_epoch_index, epoch_ref,
     proposed_settlement_epoch_id, validator_hotkey, candidate_payload_hash,
     validator_hotkey_signature, candidate_authorization_hash, tempo,
     blocks_since_last_step, next_epoch_block, blocks_remaining, observed_at,
     chain_state_receipt_hash, snapshot_doc)
VALUES
    ('{H("5")}', 'leadpoet.subnet_epoch_snapshot.v1', '{H("a")}',
     'bittensor.subnet_epoch_index.v1', '{GENESIS}', {NETUID}, 'finalized',
     '{BLOCK_HASH}', {CUTOVER_BLOCK}, {CUTOVER_BLOCK}, 0,
     {SUBNET_EPOCH_INDEX}, '{H("e")}', {SETTLEMENT}, '{HOTKEY}', '{H("7")}',
     '0x{"ab" * 64}', '{H("8")}', {TEMPO}, 0, {NEXT_EPOCH_BLOCK},
     {NEXT_EPOCH_BLOCK - CUTOVER_BLOCK}, '{OBSERVED_AT}', '{H("b")}',
     '{doc}');
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pgbin", default="", help="directory holding initdb/pg_ctl/psql")
    parser.add_argument("--skip-112", action="store_true",
                        help="rehearse WITHOUT the canonicalization migration "
                             "(expects the row-shape rejection)")
    parser.add_argument("--keep", action="store_true", help="keep the scratch cluster")
    args = parser.parse_args()

    pgbin = Path(args.pgbin) if args.pgbin else None
    for candidate in (
        pgbin,
        Path("/opt/homebrew/opt/postgresql@17/bin"),
        Path("/usr/lib/postgresql/17/bin"),
        Path("/usr/lib/postgresql/16/bin"),
    ):
        if candidate and (candidate / "initdb").exists():
            pgbin = candidate
            break
    else:
        if shutil.which("initdb"):
            pgbin = Path(shutil.which("initdb")).parent
        else:
            print("SKIP: no PostgreSQL server binaries found (initdb)")
            return 1

    work = Path(tempfile.mkdtemp(prefix="lp-stage-rehearsal-"))
    sock = Path(tempfile.mkdtemp(prefix="lp-pgs-", dir="/tmp"))
    data = work / "data"
    env = {"LC_ALL": "C", "PATH": "/usr/bin:/bin"}

    def pg(*cmd: str, sql: str = "", check: bool = True):
        base = [str(pgbin / cmd[0]), *cmd[1:]]
        return subprocess.run(base, input=sql or None, capture_output=True,
                              text=True, env=env, check=check)

    pg("initdb", "-D", str(data), "-U", "scratch", "-E", "UTF8")
    pg("pg_ctl", "-D", str(data), "-o",
       f"-p 55433 -k {sock} -c listen_addresses='' -c timezone=UTC",
       "-l", str(work / "pg.log"), "start")

    def stop():
        pg("pg_ctl", "-D", str(data), "stop", check=False)
        if not args.keep:
            shutil.rmtree(work, ignore_errors=True)
            shutil.rmtree(sock, ignore_errors=True)

    atexit.register(stop)

    def psql(sql: str, check: bool = True):
        return subprocess.run(
            [str(pgbin / "psql"), "-h", str(sock), "-p", "55433",
             "-U", "scratch", "-d", "postgres", "-v", "ON_ERROR_STOP=1",
             "-X", "-q", "-P", "expanded=on", "-f", "-"],
            input=sql, capture_output=True, text=True, env=env, check=check,
        )

    print(f"cluster: {work} (socket {sock})")
    psql(STUBS)
    applied = MIGRATIONS + ([] if args.skip_112 else [CANONICALIZATION])
    for name in applied:
        result = psql((SCRIPTS / name).read_text(encoding="utf-8"), check=False)
        if result.returncode != 0:
            print(f"MIGRATION FAILED: {name}\n{result.stderr[-2000:]}")
            return 1
        print(f"applied {name}")
    seeded = psql(fixture_sql(), check=False)
    if seeded.returncode != 0:
        print(f"FIXTURE FAILED:\n{seeded.stderr[-2000:]}")
        return 1
    print("fixture seeded (candidate, receipts, predecessor, fenced state)")

    stage = psql(
        "SELECT * FROM public.research_lab_stateful_subnet_epoch_stage_v2("
        f"$lp_row${json.dumps(cutover_row())}$lp_row$::jsonb, "
        f"$lp_event${json.dumps(initialization_event())}$lp_event$::jsonb);",
        check=False,
    )
    print(stage.stdout.strip() or stage.stderr.strip())

    if args.skip_112:
        if "cutover row shape is invalid" in stage.stderr:
            print("REHEARSAL OK: reproduced the pre-112 row-shape rejection")
            return 0
        print("UNEXPECTED: staging did not fail with the known rejection")
        return 1

    if stage.returncode == 0 and "stateful_staged" in stage.stdout:
        readback = psql(
            "SELECT lifecycle_state, staged_at IS NOT NULL AS staged,"
            " initialization_payload_hash"
            " FROM public.research_lab_stateful_subnet_epoch_cutover_state_v1;"
        )
        print(readback.stdout.strip())
        print("REHEARSAL OK: stage_v2 accepted the canonical cutover row")
        return 0
    print(f"REHEARSAL FAILED:\n{stage.stderr[-2000:]}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
