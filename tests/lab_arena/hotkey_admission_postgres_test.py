"""Current-schema PostgreSQL coverage for hotkey-based Arena admission."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from lab_arena.owner_admission import OwnerAdmission
from lab_arena.store import ArenaStore, ArenaStoreError, PsycopgTransport
from tests.lab_arena.lab_arena_pg_harness import (
    POSTGREST_MIGRATIONS,
    database_with_lab_arena_migration,
)
from tests.lab_arena.test_lab_arena_migration_postgres import (
    encrypted_runtime_credentials,
    hotkey,
    round_config,
    source_submission_doc,
)


MIGRATION = "228-lab-arena-hotkey-admission.sql"
PRE_MIGRATIONS = tuple(
    migration for migration in POSTGREST_MIGRATIONS if migration != MIGRATION
)
BLOCK_HASH = "0x" + "a" * 64


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(POSTGREST_MIGRATIONS)


@pytest.fixture(scope="module")
def database_before_228():
    yield from database_with_lab_arena_migration(PRE_MIGRATIONS)


def _store(database):
    psycopg2, dsn = database
    return ArenaStore(PsycopgTransport(lambda: psycopg2.connect(**dsn)))


def _config(round_id: str, *, cap: int = 20, **updates):
    config = round_config(round_id, [hotkey(round_id + "-runner")])
    config.update(
        {
            "integrity_policy": "arena_integrity_v1",
            "max_challengers": cap,
            "cost_per_company_microusd": 1_000_000,
            "scorer_policy": {
                "scoring_adapter_version": "qualification_integrity_v2"
            },
            "custom_marker": {"unchanged": True},
        }
    )
    config.update(updates)
    return config


def _admission(owner: str, *, number: int = 123):
    return OwnerAdmission(owner, number, BLOCK_HASH)


def _register(
    store,
    round_id: str,
    submission_id: str,
    miner: str,
    owner: str,
    *,
    source_size_bytes: int = 4096,
):
    document = source_submission_doc(round_id, submission_id)
    document["source_size_bytes"] = source_size_bytes
    return store.register_submission(
        round_id,
        submission_id,
        miner,
        document,
        owner_admission=_admission(owner),
    )


def _accept(store, round_id: str, submission_id: str, miner: str):
    return store.accept_submission_with_credentials(
        round_id,
        submission_id,
        miner,
        encrypted_runtime_credentials(submission_id),
    )


def test_same_coldkey_different_hotkeys_are_admitted_concurrently(database):
    psycopg2, dsn = database
    setup = _store(database)
    round_id = "arena-2098-04-01-siblings"
    owner = hotkey("hotkey-admission-shared-owner")
    assert setup.create_round(round_id, _config(round_id))["status"] == "created"

    def submit(index: int):
        contender = ArenaStore(
            PsycopgTransport(lambda: psycopg2.connect(**dsn))
        )
        submission_id = f"sub-shared-owner-{index}"
        miner = hotkey(f"hotkey-admission-sibling-{index}")
        try:
            registered = _register(
                contender, round_id, submission_id, miner, owner
            )
            accepted = _accept(contender, round_id, submission_id, miner)
            return registered, accepted
        finally:
            contender._transport.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(submit, range(2)))
    assert [row[0]["status"] for row in results] == ["registered", "registered"]
    assert [row[1]["status"] for row in results] == ["ok", "ok"]
    accepted_hotkeys = {
        row["miner_hotkey"]
        for row in setup.list_submissions(round_id, status="accepted")
    }
    assert len(accepted_hotkeys) == 2


def test_same_hotkey_cannot_replace_an_accepted_model(database):
    psycopg2, dsn = database
    store = _store(database)
    round_id = "arena-2098-04-02-hotkey"
    miner = hotkey("hotkey-admission-one-miner")
    owner = hotkey("hotkey-admission-one-owner")
    assert store.create_round(round_id, _config(round_id))["status"] == "created"
    assert _register(store, round_id, "sub-hotkey-first", miner, owner)[
        "status"
    ] == "registered"
    assert _accept(store, round_id, "sub-hotkey-first", miner)["status"] == "ok"

    with pytest.raises(ArenaStoreError, match="lab_arena_submission_owner_changed"):
        _register(
            store,
            round_id,
            "sub-hotkey-wrong-owner",
            miner,
            hotkey("hotkey-admission-wrong-owner"),
            source_size_bytes=4097,
        )
    with pytest.raises(ArenaStoreError, match="lab_arena_submission_conflict"):
        _register(
            store,
            round_id,
            "sub-hotkey-second",
            miner,
            owner,
            source_size_bytes=4097,
        )
    with psycopg2.connect(**dsn) as connection:
        connection.autocommit = True
        with connection.cursor() as cursor:
            with pytest.raises(psycopg2.Error, match="submission_owner_immutable"):
                cursor.execute(
                    "UPDATE public.lab_arena_submissions SET owner_coldkey=%s "
                    "WHERE submission_id='sub-hotkey-first'",
                    (hotkey("hotkey-admission-mutated-owner"),),
                )
    accepted = store.list_submissions(round_id, status="accepted")
    assert [(row["submission_id"], row["source_ref"]) for row in accepted] == [
        (
            "sub-hotkey-first",
            f"arena/{round_id}/sources/sub-hotkey-first.tar.gz",
        )
    ]


def test_baseline_is_outside_twenty_challenger_cap_and_boundary_is_atomic(
    database,
):
    psycopg2, dsn = database
    store = _store(database)
    round_id = "arena-2098-04-03-cap"
    config = _config(round_id)
    assert store.create_round(round_id, config)["status"] == "created"

    baseline_id = "baseline-2098-04-03-cap"
    baseline = config["baseline_hotkey"]
    baseline_doc = source_submission_doc(round_id, baseline_id, is_king=True)
    assert store.register_submission(
        round_id, baseline_id, baseline, baseline_doc
    )["status"] == "registered"
    assert store.update_submission(
        round_id, baseline_id, "uploading", "accepted"
    )["status"] == "ok"

    entries = []
    for index in range(21):
        submission_id = f"sub-cap-{index + 1}"
        miner = hotkey(f"hotkey-admission-cap-miner-{index + 1}")
        owner = hotkey(f"hotkey-admission-cap-owner-{index + 1}")
        assert _register(store, round_id, submission_id, miner, owner)[
            "status"
        ] == "registered"
        entries.append((submission_id, miner))

    for submission_id, miner in entries[:19]:
        assert _accept(store, round_id, submission_id, miner)["status"] == "ok"

    def accept_boundary(entry):
        contender = ArenaStore(
            PsycopgTransport(lambda: psycopg2.connect(**dsn))
        )
        try:
            return _accept(contender, round_id, entry[0], entry[1])
        except ArenaStoreError as exc:
            return {"error": str(exc)}
        finally:
            contender._transport.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        boundary = list(pool.map(accept_boundary, entries[19:]))
    assert sum(row.get("status") == "ok" for row in boundary) == 1
    assert sum(
        "lab_arena_round_full" in row.get("error", "")
        for row in boundary
    ) == 1
    assert len(store.list_submissions(round_id, status="accepted")) == 21
    assert sum(
        not row["is_king"]
        for row in store.list_submissions(round_id, status="accepted")
    ) == 20


def test_migration_replay_preserves_rows_credentials_and_repairs_only_open_rounds(
    database_before_228,
):
    psycopg2, dsn = database_before_228
    store = _store(database_before_228)
    owner = hotkey("hotkey-admission-migration-owner")
    eligible = "arena-2098-05-01"
    eligible_16 = "arena-2098-05-02"
    custom_cap = "arena-2098-05-03"
    committed = "arena-2098-05-04"
    has_participants = "arena-2098-05-05"
    has_benchmark = "arena-2098-05-06"
    has_run = "arena-2098-05-07"
    historical = "arena-2001-05-08"
    shadow = "arena-2098-05-09"
    other_network = "arena-2098-05-10"
    suffixed = "arena-2098-05-11-custom"

    configs = {
        eligible: _config(eligible, cap=8),
        eligible_16: _config(eligible_16, cap=16),
        custom_cap: _config(custom_cap, cap=12),
        committed: _config(committed, cap=8),
        has_participants: _config(has_participants, cap=8),
        has_benchmark: _config(has_benchmark, cap=8),
        has_run: _config(has_run, cap=8),
        historical: _config(
            historical,
            cap=8,
            schedule={
                "submission_open": "2000-01-01T00:00:00Z",
                "submission_cutoff": "2001-06-01T00:00:00Z",
            },
        ),
        shadow: _config(shadow, cap=8, mode="shadow"),
        other_network: _config(
            other_network, cap=8, network_name="test", netuid=401
        ),
        suffixed: _config(suffixed, cap=8),
    }
    for round_id, config in configs.items():
        assert store.create_round(round_id, config)["status"] == "created"

    first_miner = hotkey("hotkey-admission-migration-first")
    assert _register(
        store, eligible, "sub-migration-first", first_miner, owner
    )["status"] == "registered"
    assert _accept(store, eligible, "sub-migration-first", first_miner)[
        "status"
    ] == "ok"
    credential_before = store.get_submission_credential(
        "sub-migration-first", first_miner, "openrouter"
    )

    run_miner = hotkey("hotkey-admission-run-miner")
    assert _register(
        store,
        has_run,
        "sub-has-run",
        run_miner,
        hotkey("hotkey-admission-run-owner"),
    )["status"] == "registered"
    assert _accept(store, has_run, "sub-has-run", run_miner)["status"] == "ok"

    with psycopg2.connect(**dsn) as connection, connection.cursor() as cursor:
        cursor.execute(
            "ALTER TABLE public.lab_arena_rounds DISABLE TRIGGER USER"
        )
        cursor.execute(
            "UPDATE public.lab_arena_rounds SET status='committed' "
            "WHERE round_id=%s",
            (committed,),
        )
        cursor.execute(
            "UPDATE public.lab_arena_rounds SET participants='[]'::jsonb || "
            "jsonb_build_array(jsonb_build_object('submission_id','held')) "
            "WHERE round_id=%s",
            (has_participants,),
        )
        cursor.execute(
            "UPDATE public.lab_arena_rounds SET benchmark_ref='held.json' "
            "WHERE round_id=%s",
            (has_benchmark,),
        )
        cursor.execute(
            "ALTER TABLE public.lab_arena_rounds ENABLE TRIGGER USER"
        )
        cursor.execute(
            "INSERT INTO public.lab_arena_runs (run_id, assignment_id, round_id, "
            "submission_id, miner_hotkey, stage, icp_position, attempt) "
            "VALUES ('run-cap-held','assignment-cap-held',%s,'sub-has-run',%s,1,0,1)",
            (has_run, run_miner),
        )

    migration_sql = (
        Path(__file__).resolve().parents[2] / "scripts" / MIGRATION
    ).read_text(encoding="utf-8")
    with psycopg2.connect(**dsn) as connection:
        connection.autocommit = True
        with connection.cursor() as cursor:
            cursor.execute(migration_sql)
            cursor.execute(migration_sql)

            cursor.execute(
                "SELECT round_id, configuration_doc ->> 'max_challengers' "
                "FROM public.lab_arena_rounds ORDER BY round_id"
            )
            caps = dict(cursor.fetchall())
            assert caps[eligible] == "20"
            assert caps[eligible_16] == "20"
            assert caps[custom_cap] == "12"
            for untouched in (
                committed,
                has_participants,
                has_benchmark,
                has_run,
                historical,
                shadow,
                other_network,
                suffixed,
            ):
                assert caps[untouched] == "8"

            cursor.execute(
                "SELECT configuration_doc FROM public.lab_arena_rounds "
                "WHERE round_id=%s",
                (eligible,),
            )
            expected = dict(configs[eligible])
            expected["max_challengers"] = 20
            assert cursor.fetchone()[0] == expected
            cursor.execute(
                "SELECT owner_coldkey, owner_block_number, owner_block_hash, "
                "source_ref, status FROM public.lab_arena_submissions "
                "WHERE submission_id='sub-migration-first'"
            )
            assert cursor.fetchone() == (
                owner,
                123,
                BLOCK_HASH,
                f"arena/{eligible}/sources/sub-migration-first.tar.gz",
                "accepted",
            )
            cursor.execute(
                "SELECT "
                "to_regclass('public.lab_arena_submissions_one_active_owner_uq'), "
                "to_regclass("
                "'public.lab_arena_submissions_one_active_per_miner_uq'), "
                "strpos(pg_get_functiondef("
                "'public.lab_arena__register_submission_v2("
                "text,text,text,jsonb,text,bigint,text)'::regprocedure), "
                "'lab_arena_owner_active_submission')"
            )
            assert cursor.fetchone() == (
                None,
                "lab_arena_submissions_one_active_per_miner_uq",
                0,
            )
            cursor.execute(
                "SELECT tgenabled FROM pg_trigger "
                "WHERE tgrelid='public.lab_arena_rounds'::regclass "
                "AND tgname='lab_arena_rounds_write_once'"
            )
            assert cursor.fetchone() == ("O",)
            wrapper = (
                "public.lab_arena_register_submission_v2"
                "(text,text,text,jsonb,text,bigint,text)"
            )
            core = (
                "public.lab_arena__register_submission_v2"
                "(text,text,text,jsonb,text,bigint,text)"
            )
            cursor.execute(
                "SELECT "
                "has_function_privilege('lab_arena_service', %s, 'EXECUTE'), "
                "has_function_privilege('anon', %s, 'EXECUTE'), "
                "has_function_privilege('authenticated', %s, 'EXECUTE'), "
                "has_function_privilege('service_role', %s, 'EXECUTE'), "
                "has_function_privilege('lab_arena_service', %s, 'EXECUTE')",
                (wrapper, wrapper, wrapper, wrapper, core),
            )
            assert cursor.fetchone() == (True, False, False, False, False)

    assert store.get_submission_credential(
        "sub-migration-first", first_miner, "openrouter"
    ) == credential_before
    second_miner = hotkey("hotkey-admission-migration-second")
    assert _register(
        store, eligible, "sub-migration-second", second_miner, owner
    )["status"] == "registered"
    assert _accept(store, eligible, "sub-migration-second", second_miner)[
        "status"
    ] == "ok"
