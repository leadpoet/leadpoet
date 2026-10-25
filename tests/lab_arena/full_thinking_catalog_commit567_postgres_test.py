"""The catalog commit accepts $6 only for full-thinking rounds."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from lab_arena.store import ArenaStore, ArenaStoreError, PsycopgTransport
from tests.lab_arena import restore_recorded_results538_postgres_test as prior
from tests.lab_arena.test_lab_arena_migration_postgres import (
    hotkey,
    round_config,
)


SCRIPTS = Path(__file__).parents[2] / "scripts"
MIGRATION = SCRIPTS / "567-lab-arena-full-thinking-catalog-commit.sql"
CATALOG = {
    "schema_version": "leadpoet.lab_arena.deepline_catalog.v1",
    "policy_version": "leadpoet.lab_arena.deepline_company_research.v1",
    "allow_people": False,
    "tools": [{"tool_id": "exa_search"}],
    "catalog_hash": "a" * 64,
}


@pytest.fixture(scope="module")
def database():
    fixture = prior.database.__wrapped__()
    db = next(fixture)
    try:
        with db[0].connect(**db[1]) as connection, connection.cursor() as cursor:
            for name in (
                "539-lab-arena-held-recovery-provider-admission.sql",
                "562-lab-arena-review-context-integrity-reason.sql",
                "563-lab-arena-review-source-digest-reason.sql",
                "564-lab-arena-full-thinking-six-dollar-cap.sql",
                "565-lab-arena-research-review-result.sql",
                "566-lab-arena-full-thinking-champion-baseline.sql",
            ):
                cursor.execute((SCRIPTS / name).read_text())
        yield db
    finally:
        fixture.close()


def _open_catalog_round(store, database, *, suffix: int, cap: int, full: bool, v4: bool):
    round_id = "arena-2099-10-%02d" % suffix
    day = datetime.now(timezone.utc).date()
    bank = day - timedelta(days=1)
    config = round_config(round_id, [hotkey("catalog-runner")],
                          cost_per_company_microusd=cap)
    config.update(
        sourcing_cost_eligibility_policy="successful_calls_flat_per_icp_v1",
        execution_icp_cap_microusd=cap,
    )
    if full:
        config["reasoning_capture_policy"] = "full_generated_thinking_v1"
    if v4:
        config["integrity_policy"] = "arena_integrity_v1"
        config["company_judgment_policy"] = "company_judgment_v1"
        config["scorer_policy"] = {
            "scoring_adapter_version": "qualification_integrity_v2",
            "company_judgment_policy": "company_judgment_v1",
        }
    config["schedule"]["submission_open"] = bank.isoformat() + "T00:00:00Z"
    assert store.create_round(round_id, config)["status"] == "created"
    # The commit authority reads an already frozen participant. Seed only that
    # state; owner admission and review are exercised in their own tests.
    submission_id = "catalog-562-%02d-sub" % suffix
    miner = hotkey("catalog-562-%02d" % suffix)
    with database[0].connect(**database[1]) as connection, connection.cursor() as cursor:
        cursor.execute("SET LOCAL session_replication_role = replica")
        cursor.execute(
            "INSERT INTO public.lab_arena_submissions "
            "(submission_id, round_id, miner_hotkey, status, is_king) "
            "VALUES (%s, %s, %s, 'frozen', TRUE)",
            (submission_id, round_id, miner),
        )
    participants = [{"submission_id": submission_id, "miner_hotkey": miner, "is_king": True}]
    args = dict(
        participants=participants,
        benchmark_ref="arena/%s/benchmark.json" % round_id,
        evaluation_date=day.isoformat(),
        icp_set_date=bank.isoformat(),
        scorer_image_digest=config["scorer_image_digest"],
        scorer_image_reference=config["scorer_image_reference"],
        deepline_catalog=CATALOG,
        **({"company_judgment_policy": "company_judgment_v1"} if v4 else {}),
    )
    return round_id, args


def test_catalog_commit_admits_only_matching_full_thinking_cap(database):
    psycopg, dsn = database
    store = ArenaStore(PsycopgTransport(lambda: psycopg.connect(**dsn)))

    # Reproduce the paid E2E failure against the real pre-migration RPC.
    blocked_id, blocked_args = _open_catalog_round(
        store, database, suffix=1, cap=6_000_000, full=True, v4=True)
    with pytest.raises(ArenaStoreError, match="lab_arena_deepline_catalog_commit_invalid"):
        store.commit_round_v2(blocked_id, **blocked_args)
    assert store.get_round(blocked_id)["status"] == "open"

    with psycopg.connect(**dsn) as connection, connection.cursor() as cursor:
        cursor.execute(MIGRATION.read_text())
        cursor.execute(MIGRATION.read_text())

    assert store.commit_round_v2(blocked_id, **blocked_args)["status"] == "ok"
    assert store.get_round(blocked_id)["status"] == "committed"
    assert store.get_round(blocked_id)["configuration_doc"]["deepline_catalog"] == CATALOG

    for suffix, cap, full, v4, allowed in (
        (2, 6_000_000, True, False, True),
        (3, 4_000_000, False, False, True),
        (4, 4_000_000, False, True, True),
        (5, 6_000_000, False, False, False),
        (6, 4_000_000, True, True, False),
    ):
        round_id, args = _open_catalog_round(
            store, database, suffix=suffix, cap=cap, full=full, v4=v4)
        if allowed:
            assert store.commit_round_v2(round_id, **args)["status"] == "ok"
            assert store.get_round(round_id)["status"] == "committed"
        else:
            with pytest.raises(ArenaStoreError, match="lab_arena_deepline_catalog_commit_invalid"):
                store.commit_round_v2(round_id, **args)
            assert store.get_round(round_id)["status"] == "open"
