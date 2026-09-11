"""Disposable-PostgreSQL behavior of the current Lab Arena migrations through
``lab_arena.store`` (labarena.md sections 18.1, 18.2, 18.3).

Every write goes through the SECURITY DEFINER functions as ``lab_arena_service``
via ``PsycopgTransport``; superuser access is used only to simulate time
(moving ``lease_expires_at`` into the past) and to prove trigger guards.
"""

from __future__ import annotations

import base64
import json
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import pytest

from bittensor_wallet import Keypair

from lab_arena import contracts, rewards
from lab_arena.store import (
    ArenaRoleError,
    ArenaStore,
    ArenaStoreError,
    PsycopgTransport,
    hash_lease_token,
    new_lease_token,
)
from tests.lab_arena.lab_arena_pg_harness import (
    DEFAULT_MIGRATIONS,
    LAB_ARENA_COMBINED_PROVIDER_BUDGET_MIGRATION,
    LAB_ARENA_NEXT_DAY_ICP_MIGRATION,
    LAB_ARENA_SOURCE_DISCLOSURE_MIGRATION,
    database_with_lab_arena_migration,
)
from tests.postgres_migration_harness import SCRIPTS

_KEYS: Dict[str, str] = {}


def hotkey(label: str) -> str:
    if label not in _KEYS:
        _KEYS[label] = Keypair.create_from_uri("//" + label).ss58_address
    return _KEYS[label]


def sha(seed: str) -> str:
    return contracts.document_hash({"seed": seed})


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration()


@pytest.fixture(scope="module")
def connect(database):
    psycopg2, dsn = database

    def _connect():
        return psycopg2.connect(**dsn)

    return _connect


@pytest.fixture()
def store(connect):
    transport = PsycopgTransport(connect)
    yield ArenaStore(transport, lease_ttl_seconds=120)
    transport.close()


@pytest.fixture()
def superuser(connect):
    connection = connect()
    connection.autocommit = True
    yield connection
    connection.close()


@pytest.fixture()
def fresh_superuser():
    """Isolate historical migration tests from the current-schema fixtures."""
    database = database_with_lab_arena_migration(())
    connection = None
    try:
        psycopg2, dsn = next(database)
        connection = psycopg2.connect(**dsn)
        connection.autocommit = True
        yield connection
    finally:
        if connection is not None:
            connection.close()
        database.close()


def round_config(
    round_id: str,
    runners: List[str],
    *,
    quotas=None,
    stage_1_icps=contracts.STAGE_1_ICP_COUNT,
    max_attempts=2,
    execution_cap_microusd=5_000_000,
    scoring_cap_microusd=50_000_000,
    mode="live",
    rewards_enabled=False,
    cost_per_company_microusd=None,
) -> Dict[str, Any]:
    """The production configuration fields read by the SQL claim and quota gates."""

    body = {
        "round_id": round_id,
        "mode": mode,
        "rewards_enabled": rewards_enabled,
        "schedule": {
            "submission_open": "2000-01-01T00:00:00Z",
            "submission_cutoff": "2100-01-01T00:00:00Z",
        },
        "runner_hotkeys": runners,
        "call_quotas": dict(quotas or contracts.CALL_QUOTAS_PER_ICP),
        "scoring_call_quotas": dict(contracts.SCORING_CALL_QUOTAS_PER_WORK_ITEM),
        "stage_1_icp_count": stage_1_icps,
        "stage_2_icp_count": contracts.STAGE_2_ICP_COUNT,
        "max_attempts_per_assignment": max_attempts,
        "execution_cap_microusd": execution_cap_microusd,
        "scoring_cap_microusd": scoring_cap_microusd,
        "scorer_image_digest": "sha256:" + "a" * 64,
        "scorer_image_reference": "registry.example/lab/scorer@sha256:" + "a" * 64,
        "baseline_hotkey": hotkey("baseline"),
        "reward_constants": rewards.reward_constants_document(),
    }
    if cost_per_company_microusd is not None:
        body["cost_per_company_microusd"] = cost_per_company_microusd
    return body


def source_submission_doc(
    round_id: str,
    submission_id: str,
    *,
    is_king: bool = False,
) -> Dict[str, Any]:
    document: Dict[str, Any] = {
        "source_ref": "arena/%s/sources/%s.tar.gz" % (round_id, submission_id),
        "source_size_bytes": 4096,
        "consent": {"public_rerun": True},
    }
    if is_king:
        document["is_king"] = True
    return document


def encrypted_runtime_credentials(submission_id: str) -> Dict[str, str]:
    return {
        provider: base64.b64encode(
            ("kms-ciphertext-%s-%s" % (provider, submission_id)).encode()
        ).decode()
        for provider in ("openrouter", "deepline")
    }


def pass_code_review(store: ArenaStore, submission_id: str, miner: str) -> None:
    """Record the real pre-scoring review state required by the current schema."""

    token = new_lease_token()
    claimed = store.begin_submission_review(
        submission_id,
        miner,
        token,
        1_000,
        "anthropic/claude-sonnet-5",
        1,
        4096,
    )
    assert claimed["status"] == "claimed", claimed
    finished = store.finish_submission_review(
        submission_id,
        miner,
        token,
        "passed",
        {
            "passed": True,
            "verdict": "pass",
            "model": "anthropic/claude-sonnet-5",
            "summary": "Reviewed the complete fixture source.",
            "reviewed_files": ["harness.py"],
            "reviewed_file_count": 1,
            "reviewed_file_bytes": 4096,
            "file_count": 1,
            "source_bytes": 4096,
            "findings": [],
        },
        750,
    )
    assert finished["status"] == "passed", finished


def frozen_participants(store: ArenaStore, round_id: str, count: int, *, prefix: str, king_index=None) -> List[Dict[str, Any]]:
    participants = []
    for index in range(count):
        miner = hotkey("%s-miner-%d" % (prefix, index))
        submission_id = "%s-sub-%d" % (prefix, index)
        result = store.register_submission(
            round_id,
            submission_id,
            miner,
            source_submission_doc(round_id, submission_id),
        )
        assert result["status"] == "registered"
        result = store.accept_submission_with_credentials(
            round_id,
            submission_id,
            miner,
            encrypted_runtime_credentials(submission_id),
        )
        assert result["status"] == "ok", result
        pass_code_review(store, submission_id, miner)
        result = store.update_submission(round_id, submission_id, "accepted", "frozen", {"is_king": king_index == index})
        assert result["status"] == "ok"
        participants.append({"submission_id": submission_id, "miner_hotkey": miner, "is_king": king_index == index})
    return participants


def commit_round(store: ArenaStore, round_id: str, participants) -> None:
    result = store.transition_round(round_id, "open", "committed", {
        "participants": participants,
        "benchmark_ref": "arena/%s/benchmark.json" % round_id,
        "evaluation_date": "2026-09-02",
    })
    assert result["status"] == "ok", result


def stage_positions(stage: int):
    return list(contracts.stage_positions(stage))


def open_round(store: ArenaStore, round_id: str, *, participants=3, runners=1, prefix: str, quotas=None, stage_1_icps=contracts.STAGE_1_ICP_COUNT, max_attempts=2, king_index=None, execution_cap_microusd=5_000_000, scoring_cap_microusd=50_000_000, cost_per_company_microusd=None):
    runner_keys = [hotkey("%s-runner-%d" % (prefix, i)) for i in range(runners)]
    config = round_config(round_id, runner_keys, quotas=quotas, stage_1_icps=stage_1_icps, max_attempts=max_attempts, execution_cap_microusd=execution_cap_microusd, scoring_cap_microusd=scoring_cap_microusd, cost_per_company_microusd=cost_per_company_microusd)
    assert store.create_round(round_id, config)["status"] == "created"
    parts = frozen_participants(store, round_id, participants, prefix=prefix, king_index=king_index)
    commit_round(store, round_id, parts)
    positions = stage_positions(1)
    result = store.open_stage(round_id, 1, parts, positions)
    assert result["status"] == "ok", result
    return runner_keys, parts


def claim(store: ArenaStore, round_id: str, runner: str, *, parallelism=1, ceiling=8, excluded=(), request_id=None, token=None):
    token = token or new_lease_token()
    request_id = request_id or contracts.new_request_id()
    request_hash = sha(request_id + "bytes")
    response = store.claim_assignment(
        round_id=round_id, runner_hotkey=runner, declared_parallelism=parallelism, slot_ceiling=ceiling,
        excluded_miner_hotkeys=list(excluded), request_id=request_id, request_hash=request_hash,
        lease_token_hash=hash_lease_token(token),
    )
    return response, token, request_id, request_hash


def complete(store: ArenaStore, run_id: str, lease_token_hash: str, terminal_cause: str, *, output_ref: str = ""):
    return store.complete_attempt(
        run_id=run_id,
        lease_token_hash=lease_token_hash,
        result={"terminal_status": terminal_cause},
        terminal_cause=terminal_cause,
        output_ref=output_ref,
    )


def expire_now(superuser, run_id: str) -> None:
    with superuser.cursor() as cursor:
        cursor.execute("UPDATE public.lab_arena_runs SET lease_expires_at = clock_timestamp() - interval '1 second' WHERE run_id = %s", (run_id,))


# ---------------------------------------------------------------------------
# 18.1 schema and role
# ---------------------------------------------------------------------------


def test_migration_applies_twice_and_roles_have_exact_attributes(fresh_superuser):
    with fresh_superuser.cursor() as cursor:
        # Repeat each migration at its own schema version. Replaying an old
        # CREATE OR REPLACE VIEW over later added columns is a downgrade.
        for migration in DEFAULT_MIGRATIONS:
            sql = (SCRIPTS / migration).read_text(encoding="utf-8")
            cursor.execute(sql)
            cursor.execute(sql)
        cursor.execute("SELECT rolname, rolsuper, rolbypassrls, rolcanlogin, rolinherit, rolcreaterole, rolcreatedb, rolreplication FROM pg_roles WHERE rolname IN ('lab_arena_owner', 'lab_arena_service') ORDER BY rolname")
        rows = cursor.fetchall()
        assert rows == [
            ("lab_arena_owner", False, False, False, False, False, False, False),
            ("lab_arena_service", False, False, False, False, False, False, False),
        ]
        cursor.execute("SELECT granted.rolname FROM pg_auth_members m JOIN pg_roles granted ON granted.oid = m.roleid JOIN pg_roles r ON r.oid = m.member WHERE r.rolname = 'lab_arena_service'")
        assert cursor.fetchall() == []  # authenticator absent in the harness: no memberships at all
        cursor.execute("SELECT count(*) FROM pg_default_acl d JOIN pg_roles r ON r.oid = d.defaclrole WHERE r.rolname IN ('lab_arena_owner', 'lab_arena_service')")
        assert cursor.fetchone()[0] == 0
        cursor.execute("SELECT relname, relrowsecurity FROM pg_class WHERE relname LIKE 'lab_arena_%' AND relkind = 'r' ORDER BY relname")
        assert all(row[1] for row in cursor.fetchall())
        cursor.execute("SELECT tablename, policyname FROM pg_policies WHERE tablename LIKE 'lab_arena_%' ORDER BY 1")
        policies = cursor.fetchall()
        assert len(policies) == 6 and all(name.endswith("_service_read") for _, name in policies)


def test_next_day_icp_migration_is_repeatable(superuser):
    migration = (SCRIPTS / LAB_ARENA_NEXT_DAY_ICP_MIGRATION).read_text(
        encoding="utf-8"
    )
    with superuser.cursor() as cursor:
        cursor.execute(migration)
        cursor.execute(migration)
        cursor.execute(
            "SELECT data_type FROM information_schema.columns "
            "WHERE table_schema = 'public' "
            "AND table_name = 'lab_arena_rounds' "
            "AND column_name = 'icp_set_date'"
        )
        assert cursor.fetchone() == ("date",)
        # Restore the current function definitions after this historical
        # migration replay so later behavior tests use the current schema.
        cursor.execute(
            (SCRIPTS / LAB_ARENA_COMBINED_PROVIDER_BUDGET_MIGRATION).read_text(
                encoding="utf-8"
            )
        )


def test_next_day_icp_migration_requires_source_disclosure_migration():
    prerequisite_index = DEFAULT_MIGRATIONS.index(
        LAB_ARENA_SOURCE_DISCLOSURE_MIGRATION
    )
    database = database_with_lab_arena_migration(
        DEFAULT_MIGRATIONS[:prerequisite_index]
    )
    connection = None
    try:
        psycopg2, dsn = next(database)
        connection = psycopg2.connect(**dsn)
        connection.autocommit = True
        migration = (SCRIPTS / LAB_ARENA_NEXT_DAY_ICP_MIGRATION).read_text(
            encoding="utf-8"
        )
        with connection.cursor() as cursor:
            with pytest.raises(
                psycopg2.Error,
                match="apply 199-lab-arena-source-disclosure-time.sql first",
            ):
                cursor.execute(migration)
    finally:
        if connection is not None:
            connection.close()
        database.close()


def test_open_commit_refreshes_only_the_scorer_pin_and_legacy_callers_preserve_it(store):
    new_digest = "sha256:" + "b" * 64
    new_reference = "registry.example/lab/scorer@" + new_digest
    round_id = "arena-2026-09-21-refresh"
    original = round_config(round_id, [hotkey("refresh-runner")])
    assert store.create_round(round_id, original)["status"] == "created"
    assert store.list_runs(round_id) == []
    assert store.transition_round(
        round_id,
        "open",
        "committed",
        {
            "participants": [],
            "benchmark_ref": "arena/%s/benchmark.json" % round_id,
            "evaluation_date": "2026-09-21",
            "scorer_image_digest": new_digest,
            "scorer_image_reference": new_reference,
        },
    )["status"] == "ok"
    refreshed = store.get_round(round_id)
    assert refreshed["configuration_doc"] == {
        **original,
        "scorer_image_digest": new_digest,
        "scorer_image_reference": new_reference,
    }
    assert store.list_runs(round_id) == []

    legacy_id = "arena-2026-09-22-legacy"
    legacy = round_config(legacy_id, [hotkey("legacy-runner")])
    assert store.create_round(legacy_id, legacy)["status"] == "created"
    commit_round(store, legacy_id, [])
    assert store.get_round(legacy_id)["configuration_doc"] == legacy


@pytest.mark.parametrize(
    ("suffix", "digest", "reference"),
    [
        ("shape", "sha256:not-a-digest", "registry.example/scorer@sha256:not-a-digest"),
        ("pair", "sha256:" + "b" * 64, "registry.example/scorer@sha256:" + "c" * 64),
    ],
)
def test_open_commit_rejects_invalid_scorer_pin(store, suffix, digest, reference):
    round_id = "arena-2026-09-23-bad" + suffix
    original = round_config(round_id, [hotkey("bad-scorer-" + suffix)])
    assert store.create_round(round_id, original)["status"] == "created"
    with pytest.raises(ArenaStoreError, match="lab_arena_scorer_image_invalid"):
        store.transition_round(
            round_id,
            "open",
            "committed",
            {
                "participants": [],
                "benchmark_ref": "arena/%s/benchmark.json" % round_id,
                "evaluation_date": "2026-09-23",
                "scorer_image_digest": digest,
                "scorer_image_reference": reference,
            },
        )
    row = store.get_round(round_id)
    assert row["status"] == "open" and row["configuration_doc"] == original
    assert store.list_runs(round_id) == []


@pytest.mark.parametrize("status", ["committed", "stage1", "published", "cancelled"])
def test_scorer_pin_cannot_change_after_commit(database, superuser, status):
    round_id = "arena-2026-09-24-i" + status
    configuration = round_config(round_id, [hotkey("immutable-" + status)])
    with superuser.cursor() as cursor:
        cursor.execute(
            "INSERT INTO public.lab_arena_rounds (round_id, status, configuration_doc, rewards_enabled) "
            "VALUES (%s, %s, %s::jsonb, FALSE)",
            (round_id, status, json.dumps(configuration)),
        )
        with pytest.raises(database[0].Error):
            cursor.execute(
                "UPDATE public.lab_arena_rounds SET configuration_doc = "
                "configuration_doc || %s::jsonb WHERE round_id = %s",
                (
                    json.dumps({
                        "scorer_image_digest": "sha256:" + "b" * 64,
                        "scorer_image_reference": "registry.example/scorer@sha256:" + "b" * 64,
                    }),
                    round_id,
                ),
            )


def test_open_commit_cannot_change_any_other_configuration_field(database, superuser):
    round_id = "arena-2026-09-24-otherconfig"
    configuration = round_config(round_id, [hotkey("other-config")])
    new_digest = "sha256:" + "b" * 64
    with superuser.cursor() as cursor:
        cursor.execute(
            "INSERT INTO public.lab_arena_rounds (round_id, status, configuration_doc, rewards_enabled) "
            "VALUES (%s, 'open', %s::jsonb, FALSE)",
            (round_id, json.dumps(configuration)),
        )
        with pytest.raises(database[0].Error, match="round configuration is write-once"):
            cursor.execute(
                "UPDATE public.lab_arena_rounds SET status = 'committed', "
                "status_generation = status_generation + 1, configuration_doc = "
                "configuration_doc || %s::jsonb WHERE round_id = %s",
                (
                    json.dumps({
                        "mode": "shadow",
                        "scorer_image_digest": new_digest,
                        "scorer_image_reference": "registry.example/scorer@" + new_digest,
                    }),
                    round_id,
                ),
            )
        cursor.execute(
            "SELECT status, configuration_doc FROM public.lab_arena_rounds WHERE round_id = %s",
            (round_id,),
        )
        assert cursor.fetchone() == ("open", configuration)


def test_daily_icp_function_reads_exact_historical_date_and_source_table_is_private(database):
    psycopg2, dsn = database
    now = datetime.now(timezone.utc)
    set_id = int(now.strftime("%Y%m%d"))
    prior = now.date() - timedelta(days=1)
    prior_set_id = int(prior.strftime("%Y%m%d"))
    future_set_id = int((now.date() + timedelta(days=1)).strftime("%Y%m%d"))
    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    icps = [{"icp_id": "today-%d" % index} for index in range(20)]
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO public.qualification_private_icp_sets (
                  set_id, icps, active_from, active_until, is_active
                ) VALUES (
                  %s, %s::jsonb,
                  pg_catalog.statement_timestamp() - interval '1 hour',
                  pg_catalog.statement_timestamp() + interval '23 hours',
                  TRUE
                )
                ON CONFLICT (set_id) DO UPDATE
                SET icps = EXCLUDED.icps,
                    active_from = EXCLUDED.active_from,
                    active_until = EXCLUDED.active_until,
                    is_active = TRUE
                """,
                (
                    set_id,
                    json.dumps(icps),
                ),
            )
            cursor.execute(
                """
                INSERT INTO public.qualification_private_icp_sets (
                  set_id, icps, active_from, active_until, is_active
                ) VALUES (%s, %s::jsonb, %s, %s, FALSE)
                ON CONFLICT (set_id) DO UPDATE
                SET icps = EXCLUDED.icps,
                    active_from = EXCLUDED.active_from,
                    active_until = EXCLUDED.active_until,
                    is_active = FALSE
                """,
                (
                    prior_set_id,
                    json.dumps(icps),
                    datetime.combine(prior, datetime.min.time(), tzinfo=timezone.utc),
                    datetime.combine(now.date(), datetime.min.time(), tzinfo=timezone.utc),
                ),
            )
    finally:
        connection.close()

    transport = PsycopgTransport(lambda: psycopg2.connect(**dsn))
    try:
        daily_store = ArenaStore(transport)
        assert daily_store.current_daily_icp_set(set_id) == {
            "status": "ready",
            "set_id": set_id,
            "icps": icps,
        }
        # Historical rows are inactive after rotation, but the service can
        # retrieve the exact bank that was active on the requested UTC date.
        assert daily_store.current_daily_icp_set(prior_set_id) == {
            "status": "ready",
            "set_id": prior_set_id,
            "icps": icps,
        }
        assert daily_store.current_daily_icp_set(future_set_id) == {
            "status": "unavailable",
            "set_id": future_set_id,
        }
        control = psycopg2.connect(**dsn)
        control.autocommit = True
        try:
            with control.cursor() as cursor:
                cursor.execute(
                    "UPDATE public.qualification_private_icp_sets "
                    "SET is_active = FALSE WHERE set_id = %s",
                    (set_id,),
                )
            assert daily_store.current_daily_icp_set(set_id)["status"] == "unavailable"
            with control.cursor() as cursor:
                cursor.execute(
                    "UPDATE public.qualification_private_icp_sets "
                    "SET is_active = TRUE, icps = %s::jsonb WHERE set_id = %s",
                    (json.dumps(icps[:-1]), set_id),
                )
            assert daily_store.current_daily_icp_set(set_id)["status"] == "unavailable"
        finally:
            control.close()
    finally:
        transport.close()

    direct = psycopg2.connect(**dsn)
    direct.autocommit = True
    try:
        with direct.cursor() as cursor:
            cursor.execute("SET ROLE lab_arena_service")
            with pytest.raises(psycopg2.errors.InsufficientPrivilege):
                cursor.execute(
                    "SELECT icps FROM public.qualification_private_icp_sets"
                )
            direct.rollback()
    finally:
        direct.close()


def test_commit_persists_day_zero_bank_date_and_it_is_write_once(database, store, superuser):
    round_id = "arena-2026-09-25-bankdate"
    bank_date = datetime.now(timezone.utc).date()
    evaluation_date = bank_date + timedelta(days=1)
    configuration = round_config(round_id, [hotkey("bank-date-runner")])
    configuration["schedule"] = {
        "submission_open": datetime.combine(
            bank_date, datetime.min.time(), tzinfo=timezone.utc
        ).isoformat().replace("+00:00", "Z"),
        "submission_cutoff": datetime.combine(
            evaluation_date, datetime.min.time(), tzinfo=timezone.utc
        ).isoformat().replace("+00:00", "Z"),
    }
    assert store.create_round(round_id, configuration)["status"] == "created"
    participants = frozen_participants(
        store, round_id, 1, prefix="bank-date", king_index=0
    )
    assert store.commit_round_v2(
        round_id,
        participants=participants,
        benchmark_ref="arena/%s/benchmark.json" % round_id,
        evaluation_date=evaluation_date.isoformat(),
        icp_set_date=bank_date.isoformat(),
        scorer_image_digest=configuration["scorer_image_digest"],
        scorer_image_reference=configuration["scorer_image_reference"],
    )["status"] == "ok"
    assert str(store.get_round(round_id)["icp_set_date"]) == bank_date.isoformat()
    with superuser.cursor() as cursor:
        with pytest.raises(database[0].Error, match="icp_set_date_write_once"):
            cursor.execute(
                "UPDATE public.lab_arena_rounds SET icp_set_date = %s WHERE round_id = %s",
                ((bank_date - timedelta(days=1)).isoformat(), round_id),
            )


def test_previous_service_commit_remains_legacy_after_migration(store):
    round_id = "arena-2026-09-26-nminus1"
    configuration = round_config(round_id, [hotkey("n-minus-one-runner")])
    assert store.create_round(round_id, configuration)["status"] == "created"
    commit_round(store, round_id, [])
    row = store.get_round(round_id)
    assert row["status"] == "committed"
    assert row["icp_set_date"] is None


@pytest.mark.parametrize(
    ("mode", "initial_cap", "initial_cpl", "expected_cap", "expected_cpl"),
    [
        ("live", 5_000_000, None, 50_000_000, 500_000),
        ("shadow", 1_234_567, None, 1_234_567, None),
        ("live", 7_654_321, 123_456, 7_654_321, 123_456),
    ],
)
def test_commit_atomically_freezes_cost_policy_only_for_legacy_live_rounds(
    store, mode, initial_cap, initial_cpl, expected_cap, expected_cpl
):
    suffix = {"live": "live", "shadow": "shadow"}[mode]
    if initial_cpl is not None:
        suffix = "typed"
    round_id = "arena-2026-09-10-freeze" + suffix
    bank_date = datetime.now(timezone.utc).date()
    evaluation_date = bank_date + timedelta(days=1)
    configuration = round_config(
        round_id, [hotkey("freeze-" + suffix)], mode=mode,
        execution_cap_microusd=initial_cap,
        cost_per_company_microusd=initial_cpl,
    )
    configuration["schedule"] = {
        "submission_open": datetime.combine(
            bank_date, datetime.min.time(), tzinfo=timezone.utc
        ).isoformat().replace("+00:00", "Z"),
        "submission_cutoff": datetime.combine(
            evaluation_date, datetime.min.time(), tzinfo=timezone.utc
        ).isoformat().replace("+00:00", "Z"),
    }
    assert store.create_round(round_id, configuration)["status"] == "created"
    participants = frozen_participants(
        store, round_id, 1, prefix="freeze-" + suffix, king_index=0
    )
    result = store.commit_round_v2(
        round_id,
        participants=participants,
        benchmark_ref="arena/%s/benchmark.json" % round_id,
        evaluation_date=evaluation_date.isoformat(),
        icp_set_date=bank_date.isoformat(),
        scorer_image_digest=configuration["scorer_image_digest"],
        scorer_image_reference=configuration["scorer_image_reference"],
    )
    assert result["status"] == "ok"
    frozen = store.get_round(round_id)["configuration_doc"]
    assert frozen["execution_cap_microusd"] == expected_cap
    assert frozen.get("cost_per_company_microusd") == expected_cpl
    # A replay cannot rewrite the now-committed configuration.
    assert store.commit_round_v2(
        round_id,
        participants=participants,
        benchmark_ref="arena/%s/other.json" % round_id,
        evaluation_date=evaluation_date.isoformat(),
        icp_set_date=bank_date.isoformat(),
        scorer_image_digest=configuration["scorer_image_digest"],
        scorer_image_reference=configuration["scorer_image_reference"],
    )["status"] == "stale"
    assert store.get_round(round_id)["configuration_doc"] == frozen


def test_new_policy_stage_two_assigns_all_twenty_to_more_than_ten_challengers(
    store, superuser
):
    round_id = "arena-2026-09-27-all20"
    bank_date = datetime.now(timezone.utc).date()
    evaluation_date = bank_date + timedelta(days=1)
    configuration = round_config(round_id, [hotkey("all20-runner")])
    configuration["schedule"] = {
        "submission_open": datetime.combine(
            bank_date, datetime.min.time(), tzinfo=timezone.utc
        ).isoformat().replace("+00:00", "Z"),
        "submission_cutoff": datetime.combine(
            evaluation_date, datetime.min.time(), tzinfo=timezone.utc
        ).isoformat().replace("+00:00", "Z"),
    }
    assert store.create_round(round_id, configuration)["status"] == "created"
    participants = frozen_participants(
        store, round_id, 13, prefix="all20-stage2", king_index=0
    )
    assert store.commit_round_v2(
        round_id,
        participants=participants,
        benchmark_ref="arena/%s/benchmark.json" % round_id,
        evaluation_date=evaluation_date.isoformat(),
        icp_set_date=bank_date.isoformat(),
        scorer_image_digest=configuration["scorer_image_digest"],
        scorer_image_reference=configuration["scorer_image_reference"],
    )["status"] == "ok"
    finalists = [
        item["submission_id"] for item in participants if not item["is_king"]
    ][:10]
    with superuser.cursor() as cursor:
        cursor.execute(
            "UPDATE public.lab_arena_rounds "
            "SET status = 'stage1_scored', finalists = %s::jsonb "
            "WHERE round_id = %s",
            (json.dumps(finalists), round_id),
        )
    opened = store.open_stage(round_id, 2, participants, stage_positions(2))
    assert opened["status"] == "ok"
    assert opened["assignments"] == 13 * contracts.STAGE_2_ICP_COUNT


def test_commit_waits_for_inflight_acceptance_and_retries_without_orphan(
    connect, store
):
    round_id = "arena-2026-09-28-cutoffrace"
    bank_date = datetime.now(timezone.utc).date()
    evaluation_date = bank_date + timedelta(days=1)
    configuration = round_config(round_id, [hotkey("cutoff-race-runner")])
    configuration["schedule"] = {
        "submission_open": datetime.combine(
            bank_date, datetime.min.time(), tzinfo=timezone.utc
        ).isoformat().replace("+00:00", "Z"),
        "submission_cutoff": datetime.combine(
            evaluation_date, datetime.min.time(), tzinfo=timezone.utc
        ).isoformat().replace("+00:00", "Z"),
    }
    assert store.create_round(round_id, configuration)["status"] == "created"
    participants = frozen_participants(
        store, round_id, 1, prefix="cutoff-race-base", king_index=0
    )
    miner_hotkey = hotkey("cutoff-race-miner")
    miner_submission = "cutoff-race-miner-sub"
    assert store.register_submission(
        round_id,
        miner_submission,
        miner_hotkey,
        source_submission_doc(round_id, miner_submission),
    )["status"] == "registered"

    accepting = connect()
    accepting.autocommit = False
    try:
        with accepting.cursor() as cursor:
            cursor.execute("SET ROLE lab_arena_service")
            cursor.execute(
                "SELECT public.lab_arena_accept_submission_with_credentials("
                "%s, %s, %s, %s::jsonb)",
                (
                    round_id,
                    miner_submission,
                    miner_hotkey,
                    json.dumps(encrypted_runtime_credentials(miner_submission)),
                ),
            )
            assert cursor.fetchone()[0]["status"] == "ok"
        with ThreadPoolExecutor(max_workers=1) as executor:
            pending = executor.submit(
                store.commit_round_v2,
                round_id,
                participants=participants,
                benchmark_ref="arena/%s/benchmark.json" % round_id,
                evaluation_date=evaluation_date.isoformat(),
                icp_set_date=bank_date.isoformat(),
                scorer_image_digest=configuration["scorer_image_digest"],
                scorer_image_reference=configuration["scorer_image_reference"],
            )
            with pytest.raises(FutureTimeout):
                pending.result(timeout=0.1)
            accepting.commit()
            assert pending.result(timeout=5) == {
                "status": "retry",
                "round_status": "open",
                "remaining_admissions": 1,
            }
    finally:
        accepting.rollback()
        accepting.close()

    assert store.get_round(round_id)["status"] == "open"
    pass_code_review(store, miner_submission, miner_hotkey)
    assert store.update_submission(
        round_id, miner_submission, "accepted", "frozen", {}
    )["status"] == "ok"
    participants.append(
        {
            "submission_id": miner_submission,
            "miner_hotkey": miner_hotkey,
            "is_king": False,
        }
    )
    assert store.commit_round_v2(
        round_id,
        participants=participants,
        benchmark_ref="arena/%s/benchmark.json" % round_id,
        evaluation_date=evaluation_date.isoformat(),
        icp_set_date=bank_date.isoformat(),
        scorer_image_digest=configuration["scorer_image_digest"],
        scorer_image_reference=configuration["scorer_image_reference"],
    )["status"] == "ok"
    assert store.list_submissions(round_id, status="accepted") == []


def test_migration_removes_the_draft_receipt_and_hash_chain_state(fresh_superuser):
    draft_round_columns = (
        "generation_attempts",
        "configuration_hash",
        "journal",
        "journal_head_hash",
        "commitment_hash",
        "commitment_doc",
        "participant_set_hash",
        "stage1_scoring_plan_hash",
        "final_score_bundle_hash",
        "result_bundle_hash",
    )
    draft_run_columns = (
        "event_cursor",
        "event_head_hash",
        "receipt_doc",
        "receipt_hash",
        "provider_call_root",
        "private_event_root",
        "cost_root",
        "icp_hash",
        "work_item_id",
        "output_hash",
        "score_ref",
        "score_doc",
    )
    draft_submission_columns = (
        "submitted_digest",
        "entry_command",
        "image_environment",
        "working_dir",
    )
    with fresh_superuser.cursor() as cursor:
        cursor.execute((SCRIPTS / DEFAULT_MIGRATIONS[0]).read_text(encoding="utf-8"))
        for column in draft_round_columns:
            cursor.execute("ALTER TABLE public.lab_arena_rounds ADD COLUMN %s JSONB" % column)
        for column in draft_run_columns:
            cursor.execute("ALTER TABLE public.lab_arena_runs ADD COLUMN %s JSONB" % column)
        for column in draft_submission_columns:
            cursor.execute("ALTER TABLE public.lab_arena_submissions ADD COLUMN %s JSONB" % column)
        cursor.execute("CREATE TABLE public.lab_arena_events (event_id BIGINT)")
        cursor.execute("CREATE TABLE public.lab_arena_accounts (miner_hotkey TEXT PRIMARY KEY)")
        cursor.execute(
            "CREATE FUNCTION public.lab_arena_append_journal_entry(TEXT, JSONB) "
            "RETURNS JSONB LANGUAGE sql AS $$ SELECT '{}'::JSONB $$"
        )
        cursor.execute(
            "CREATE FUNCTION public.lab_arena_append_events(TEXT, TEXT, JSONB, INTEGER) "
            "RETURNS JSONB LANGUAGE sql AS $$ SELECT '{}'::JSONB $$"
        )
        for migration in DEFAULT_MIGRATIONS:
            cursor.execute((SCRIPTS / migration).read_text(encoding="utf-8"))
        cursor.execute(
            "SELECT table_name, column_name FROM information_schema.columns "
            "WHERE table_schema = 'public' AND ((table_name = 'lab_arena_rounds' AND column_name = ANY(%s)) "
            "OR (table_name = 'lab_arena_runs' AND column_name = ANY(%s)) "
            "OR (table_name = 'lab_arena_submissions' AND column_name = ANY(%s)))",
            (list(draft_round_columns), list(draft_run_columns), list(draft_submission_columns)),
        )
        assert cursor.fetchall() == []
        cursor.execute(
            "SELECT to_regclass('public.lab_arena_events'), to_regclass('public.lab_arena_accounts'), "
            "to_regprocedure('public.lab_arena_append_journal_entry(text,jsonb)'), "
            "to_regprocedure('public.lab_arena_append_events(text,text,jsonb,integer)'), "
            "to_regprocedure('public.lab_arena_append_generation_attempt(text,jsonb)')"
        )
        assert cursor.fetchone() == (None, None, None, None, None)


def test_service_role_function_grants_and_non_arena_denial(superuser, store):
    with superuser.cursor() as cursor:
        cursor.execute("CREATE TABLE IF NOT EXISTS public.lab_arena_test_other_table (id INT)")
        cursor.execute("CREATE OR REPLACE FUNCTION public.lab_arena_test_other_function() RETURNS INT LANGUAGE sql AS $$ SELECT 1 $$")
        cursor.execute("REVOKE ALL ON TABLE public.lab_arena_test_other_table FROM PUBLIC")
        cursor.execute("REVOKE ALL ON FUNCTION public.lab_arena_test_other_function() FROM PUBLIC")
        for role in ("lab_arena_service", "anon", "authenticated"):
            cursor.execute("SELECT has_table_privilege(%s, 'public.lab_arena_test_other_table', 'SELECT')", (role,))
            assert cursor.fetchone()[0] is False
            cursor.execute("SELECT has_function_privilege(%s, 'public.lab_arena_test_other_function()', 'EXECUTE')", (role,))
            assert cursor.fetchone()[0] is False
        for role in ("anon", "authenticated", "service_role"):
            for table in ("lab_arena_rounds", "lab_arena_submissions", "lab_arena_runs", "lab_arena_ledger"):
                cursor.execute("SELECT has_table_privilege(%s, %s, 'SELECT')", (role, "public." + table))
                assert cursor.fetchone()[0] is False, (role, table)
            cursor.execute("SELECT has_function_privilege(%s, 'public.lab_arena_create_round(text, jsonb)', 'EXECUTE')", (role,))
            assert cursor.fetchone()[0] is False
            cursor.execute(
                "SELECT has_function_privilege(%s, "
                "'public.lab_arena_current_daily_icp_set(bigint)', 'EXECUTE')",
                (role,),
            )
            assert cursor.fetchone()[0] is False
            cursor.execute(
                "SELECT has_function_privilege(%s, "
                "'public.lab_arena_commit_round_v2(text,jsonb,text,text,date,text,text)', "
                "'EXECUTE')",
                (role,),
            )
            assert cursor.fetchone()[0] is False
        cursor.execute("SELECT has_function_privilege('lab_arena_service', 'public.lab_arena_create_round(text, jsonb)', 'EXECUTE')")
        assert cursor.fetchone()[0] is True
        cursor.execute(
            "SELECT has_function_privilege('lab_arena_service', "
            "'public.lab_arena_current_daily_icp_set(bigint)', 'EXECUTE'), "
            "has_function_privilege('lab_arena_service', "
            "'public.lab_arena_commit_round_v2(text,jsonb,text,text,date,text,text)', "
            "'EXECUTE')"
        )
        assert cursor.fetchone() == (True, True)
        cursor.execute("SELECT has_function_privilege('lab_arena_service', 'public.lab_arena__terminate_open_calls(text, text)', 'EXECUTE')")
        assert cursor.fetchone()[0] is False
        for table in ("lab_arena_ledger", "lab_arena_rounds"):
            for privilege in ("INSERT", "UPDATE", "DELETE", "TRUNCATE"):
                cursor.execute("SELECT has_table_privilege('lab_arena_service', %s, %s)", ("public." + table, privilege))
                assert cursor.fetchone()[0] is False, (table, privilege)
    identity = store.require_service_role()
    assert identity["current_user"] == "lab_arena_service" and identity["rolsuper"] is False


def test_service_refuses_to_start_as_superuser(connect):
    transport = PsycopgTransport(connect, role=None)
    try:
        with pytest.raises(ArenaRoleError):
            ArenaStore(transport).require_service_role()
    finally:
        transport.close()


def test_source_admission_has_no_cross_miner_digest_identity(store):
    round_id = "arena-2026-09-02-shared"
    runner = hotkey("shared-runner")
    assert store.create_round(round_id, round_config(round_id, [runner]))["status"] == "created"
    for index in range(2):
        miner = hotkey("shared-miner-%d" % index)
        submission_id = "shared-sub-%d" % index
        assert store.register_submission(
            round_id,
            submission_id,
            miner,
            source_submission_doc(round_id, submission_id),
        )["status"] == "registered"
        with pytest.raises(ArenaStoreError, match="lab_arena_submission_credentials_required"):
            store.update_submission(
                round_id,
                submission_id,
                "uploading",
                "accepted",
                {},
            )
        accepted = store.accept_submission_with_credentials(
            round_id,
            submission_id,
            miner,
            encrypted_runtime_credentials(submission_id),
        )
        assert accepted["status"] == "ok"


def test_credential_admission_is_owner_bound_and_retries_cannot_replace_rows(store):
    round_id = "arena-2026-09-02-credentials"
    submission_id = "credential-submission"
    miner = hotkey("credential-owner")
    wrong_miner = hotkey("credential-other")
    assert store.create_round(
        round_id, round_config(round_id, [hotkey("credential-runner")])
    )["status"] == "created"
    assert store.register_submission(
        round_id,
        submission_id,
        miner,
        source_submission_doc(round_id, submission_id),
    )["status"] == "registered"
    with pytest.raises(ArenaStoreError, match="lab_arena_submission_missing"):
        store.accept_submission_with_credentials(
            round_id,
            submission_id,
            wrong_miner,
            encrypted_runtime_credentials(submission_id),
        )
    original = encrypted_runtime_credentials(submission_id)
    original["openrouter"] = base64.b64encode(b"o" * 512).decode()
    assert store.accept_submission_with_credentials(
        round_id, submission_id, miner, original
    )["status"] == "ok"
    assert store.get_submission_credential(
        submission_id, wrong_miner, "openrouter"
    ) is None
    first = store.get_submission_credential(submission_id, miner, "openrouter")
    assert first is not None and first["ciphertext_b64"] == original["openrouter"]
    replacement = encrypted_runtime_credentials(submission_id + "-replacement")
    assert store.accept_submission_with_credentials(
        round_id, submission_id, miner, replacement
    )["status"] == "existing"
    assert store.get_submission_credential(
        submission_id, miner, "openrouter"
    )["ciphertext_b64"] == original["openrouter"]
    with pytest.raises(ArenaStoreError, match="lab_arena_credential_provider_invalid"):
        store.get_submission_credential(submission_id, miner, "management")


def test_credential_admission_rechecks_the_submission_cutoff(store, superuser):
    round_id = "arena-2026-09-02-credcutoff"
    submission_id = "cutoff-submission"
    miner = hotkey("cutoff-owner")
    assert store.create_round(
        round_id, round_config(round_id, [hotkey("cutoff-runner")])
    )["status"] == "created"
    assert store.register_submission(
        round_id,
        submission_id,
        miner,
        source_submission_doc(round_id, submission_id),
    )["status"] == "registered"
    with superuser.cursor() as cursor:
        cursor.execute("ALTER TABLE public.lab_arena_rounds DISABLE TRIGGER lab_arena_rounds_write_once")
        cursor.execute(
            "UPDATE public.lab_arena_rounds SET configuration_doc = "
            "jsonb_set(configuration_doc, '{schedule,submission_cutoff}', "
            "to_jsonb('2000-01-02T00:00:00Z'::text)) WHERE round_id = %s",
            (round_id,),
        )
        cursor.execute("ALTER TABLE public.lab_arena_rounds ENABLE TRIGGER lab_arena_rounds_write_once")
    result = store.accept_submission_with_credentials(
        round_id,
        submission_id,
        miner,
        encrypted_runtime_credentials(submission_id),
    )
    assert result["status"] == "window_closed"
    assert store.get_submission(submission_id)["status"] == "uploading"
    assert store.get_submission_credential(submission_id, miner, "openrouter") is None


def test_credential_error_is_terminal_without_a_confirmation_attempt(store):
    round_id = "arena-2026-09-02-cred"
    runners, _parts = open_round(
        store, round_id, participants=1, prefix="cred"
    )
    leased, token, _, _ = claim(store, round_id, runners[0])
    result = complete(
        store,
        leased["run_id"],
        hash_lease_token(token),
        "credential_error",
    )
    assert result["status"] == "failed"
    assert "confirmation_attempt" not in result
    assignment_runs = [
        row
        for row in store.list_runs(round_id, stage=1, kind="execute")
        if row["assignment_id"] == leased["assignment_id"]
    ]
    assert len(assignment_runs) == 1
    assert assignment_runs[0]["terminal_cause"] == "credential_error"


@pytest.mark.parametrize(
    "round_id,schedule",
    [
        ("arena-2026-09-02-notopen", {"submission_open": "2100-01-01T00:00:00Z", "submission_cutoff": "2100-01-02T00:00:00Z"}),
        ("arena-2026-09-02-cutoff", {"submission_open": "2000-01-01T00:00:00Z", "submission_cutoff": "2000-01-02T00:00:00Z"}),
    ],
)
def test_database_refuses_submissions_outside_the_half_open_window(store, round_id, schedule):
    config = round_config(round_id, [hotkey(round_id + "-runner")])
    config["schedule"] = schedule
    assert store.create_round(round_id, config)["status"] == "created"
    result = store.register_submission(
        round_id,
        round_id + "-submission",
        hotkey(round_id + "-miner"),
        source_submission_doc(round_id, round_id + "-submission"),
    )
    assert result["status"] == "window_closed"


def test_service_can_add_only_the_round_baseline_after_miner_cutoff(store):
    round_id = "arena-2026-09-02-baseline"
    config = round_config(round_id, [hotkey("baseline-runner")])
    config["schedule"] = {
        "submission_open": "2000-01-01T00:00:00Z",
        "submission_cutoff": "2000-01-02T00:00:00Z",
    }
    assert store.create_round(round_id, config)["status"] == "created"
    ordinary = store.register_submission(
        round_id,
        "late-miner",
        hotkey("late-miner"),
        source_submission_doc(round_id, "late-miner"),
    )
    assert ordinary["status"] == "window_closed"
    baseline = store.register_submission(
        round_id,
        "baseline-2026-09-02-baseline",
        hotkey("public-baseline"),
        source_submission_doc(
            round_id, "baseline-2026-09-02-baseline", is_king=True
        ),
    )
    assert baseline == {
        "status": "registered",
        "submission_status": "uploading",
        "submission_id": "baseline-2026-09-02-baseline",
        "source_ref": "arena/%s/sources/baseline-2026-09-02-baseline.tar.gz"
        % round_id,
    }


def test_round_id_is_the_plain_configuration_idempotency_key(store):
    round_id = "arena-2026-09-02-id"
    original = round_config(round_id, [hotkey("id-runner")])
    assert "configuration_hash" not in original
    assert store.create_round(round_id, original)["status"] == "created"
    retry = round_config(round_id, [hotkey("different-runner")])
    assert store.create_round(round_id, retry)["status"] == "existing"
    assert store.get_round(round_id)["configuration_doc"] == original


def test_append_only_ledger_and_write_once_rounds_resist_owner_level_mutation(store, superuser):
    round_id = "arena-2026-09-02-ao"
    runners, parts = open_round(store, round_id, participants=1, prefix="ao")
    response, token, _, _ = claim(store, round_id, runners[0])
    run_id = response["run_id"]
    lease_hash = hash_lease_token(token)
    identity = contracts.provider_call_identity(attempt=1, assignment_id=response["assignment_id"], icp_position=0, action_sequence=0, operation_id="deepline.execute", request_hash=sha("q"))
    assert store.reserve_call(run_id=run_id, lease_token_hash=lease_hash, call_identity=identity, operation_id="deepline.execute", provider="deepline", funding_source="miner_key", amount_microusd=0, call_doc={"q": 1})["status"] == "reserved"
    with superuser.cursor() as cursor:
        for statement in (
            "UPDATE public.lab_arena_ledger SET amount_microusd = 0",
            "DELETE FROM public.lab_arena_ledger",
            "UPDATE public.lab_arena_rounds SET configuration_doc = '{}'::JSONB WHERE round_id = %s" % ("'" + round_id + "'"),
            "DELETE FROM public.lab_arena_rounds WHERE round_id = '%s'" % round_id,
            "DELETE FROM public.lab_arena_runs WHERE run_id = '%s'" % run_id,
        ):
            with pytest.raises(Exception):
                cursor.execute(statement)
            superuser.rollback() if not superuser.autocommit else None


def _compact_publication(
    round_id: str, published_at: str, baseline_key: str, winner_key: str
) -> Dict[str, Any]:
    baseline_id = round_id + "-baseline"
    winner_id = round_id + "-winner"
    return {
        "schema_version": contracts.PUBLICATION_SCHEMA_VERSION,
        "round_id": round_id,
        "participants": [
            {
                "submission_id": baseline_id,
                "miner_hotkey": baseline_key,
                "is_baseline": True,
            },
            {
                "submission_id": winner_id,
                "miner_hotkey": winner_key,
                "is_baseline": False,
            },
        ],
        "stage1_ranking": [],
        "finalists": [],
        "final_ranking": [
            {
                "rank": 1,
                "submission_id": winner_id,
                "final_score": 60,
                "is_baseline": False,
            },
            {
                "rank": 2,
                "submission_id": baseline_id,
                "final_score": 50,
                "is_baseline": True,
            },
        ],
        "king_decision": {
            "outcome": "crowned",
            "king_submission_id": winner_id,
            "king_hotkey": winner_key,
            "winner_submission_id": winner_id,
        },
        "published_at": published_at,
    }


def test_cost_guard_uses_highest_eligible_challenger_and_keeps_baseline_threshold(
    store, superuser
):
    round_id = "arena-2026-09-10-costguard"
    runners, parts = open_round(
        store, round_id, participants=3, runners=1, prefix="cost-guard",
        king_index=0, execution_cap_microusd=50_000_000,
        cost_per_company_microusd=500_000,
    )
    baseline, high, low = parts
    spends = {
        baseline["submission_id"]: 600_000,
        high["submission_id"]: 500_000,
        low["submission_id"]: 400_000,
    }
    with superuser.cursor() as cursor:
        for submission_id, amount in spends.items():
            cursor.execute(
                "SELECT run_id, round_id, miner_hotkey, stage FROM public.lab_arena_runs "
                "WHERE submission_id = %s AND kind = 'execute' ORDER BY run_id LIMIT 1",
                (submission_id,),
            )
            run_id, stored_round, miner_hotkey, stage = cursor.fetchone()
            cursor.execute(
                "INSERT INTO public.lab_arena_ledger "
                "(entry_kind, miner_hotkey, round_id, submission_id, run_id, stage, "
                "call_identity, provider, operation_id, funding_source, amount_microusd, entry_doc) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, 'deepline', "
                "'deepline.execute', 'miner_key', %s, '{}'::jsonb)",
                (
                    "uncertain" if submission_id == high["submission_id"] else "settlement",
                    miner_hotkey, stored_round, submission_id, run_id, stage,
                    sha("cost-guard-" + submission_id), amount,
                ),
            )
        cursor.execute(
            "SELECT run_id FROM public.lab_arena_runs WHERE submission_id = %s "
            "AND kind = 'execute' ORDER BY run_id LIMIT 1",
            (baseline["submission_id"],),
        )
        baseline_execute_run = cursor.fetchone()[0]
        score_run_id = round_id + ":baseline-score:1"
        cursor.execute(
            "INSERT INTO public.lab_arena_runs "
            "(run_id, assignment_id, round_id, submission_id, miner_hotkey, stage, "
            "icp_position, attempt, status, stage_generation, kind, scored_run_id) "
            "VALUES (%s, %s, %s, %s, %s, 1, 0, 1, 'pending', 1, 'score', %s)",
            (score_run_id, round_id + ":baseline-score", round_id,
             baseline["submission_id"], baseline["miner_hotkey"], baseline_execute_run),
        )
        cursor.execute(
            "INSERT INTO public.lab_arena_ledger "
            "(entry_kind, miner_hotkey, round_id, submission_id, run_id, stage, "
            "call_identity, provider, operation_id, funding_source, amount_microusd, entry_doc) "
            "VALUES ('reservation', %s, %s, %s, %s, 1, %s, 'openrouter', "
            "'openrouter.chat', 'miner_key', 100, '{}'::jsonb)",
            (baseline["miner_hotkey"], round_id, baseline["submission_id"],
             score_run_id, sha("cost-guard-score-inflight")),
        )
        cursor.execute(
            "UPDATE public.lab_arena_rounds SET status = 'scored' WHERE round_id = %s",
            (round_id,),
        )

    def ranking(participant, score):
        submission_id = participant["submission_id"]
        amount = spends[submission_id]
        uncertain = submission_id == high["submission_id"]
        score_inflight = 1 if submission_id == baseline["submission_id"] else 0
        eligible = amount <= 500_000 and score_inflight == 0 and not uncertain
        reason = (
            "provider_calls_inflight" if score_inflight
            else "provider_cost_uncertain" if uncertain
            else ("eligible" if eligible else "cost_per_company_exceeded")
        )
        return {
            "rank": 1,
            "submission_id": submission_id,
            "final_score": score,
            "is_baseline": bool(participant["is_king"]),
            "eligible": eligible,
            "eligibility_reason": reason,
            "cost_summary": {
                "returned_company_count": 1,
                "execution_cap_microusd": 50_000_000,
                "cost_per_company_cap_microusd": 500_000,
                "eligibility_cap_microusd": 500_000,
                "execution": {
                    "settled_microusd": 0 if uncertain else amount,
                    "reserved_or_uncertain_microusd": amount if uncertain else 0,
                    "conservative_microusd": amount,
                    "inflight_calls": 0,
                },
                "judge": {"inflight_calls": score_inflight},
            },
        }

    rows = [ranking(high, 90), ranking(low, 60), ranking(baseline, 50)]
    public_parts = [
        {
            "submission_id": part["submission_id"],
            "miner_hotkey": part["miner_hotkey"],
            "is_baseline": bool(part["is_king"]),
        }
        for part in parts
    ]

    def publication(winner):
        decision = (
            {
                "outcome": "crowned",
                "king_submission_id": winner["submission_id"],
                "winner_submission_id": winner["submission_id"],
                "king_hotkey": winner["miner_hotkey"],
            }
            if winner is not None
            else {
                "outcome": "no_king", "king_submission_id": "",
                "winner_submission_id": "", "king_hotkey": "",
            }
        )
        return {
            "schema_version": contracts.PUBLICATION_SCHEMA_VERSION,
            "round_id": round_id,
            "participants": public_parts,
            "stage1_ranking": [],
            "finalists": [],
            "final_ranking": rows,
            "king_decision": decision,
            "published_at": "2026-09-10T00:00:00Z",
        }

    # The raw top score is cost-ineligible and cannot be crowned.
    rows[0]["eligible"] = True
    rows[0]["eligibility_reason"] = "eligible"
    with superuser.cursor() as cursor:
        with pytest.raises(Exception, match="publication_cost_report_mismatch"):
            cursor.execute(
                "UPDATE public.lab_arena_rounds SET status = 'published', "
                "publication_doc = %s::jsonb, published_at = %s, "
                "king_outcome = 'crowned', king_hotkey = %s WHERE round_id = %s",
                (json.dumps(publication(high)), "2026-09-10T00:00:00Z", high["miner_hotkey"], round_id),
            )
    rows[0]["eligible"] = False
    rows[0]["eligibility_reason"] = "provider_cost_uncertain"
    with superuser.cursor() as cursor:
        with pytest.raises(Exception, match="publication_winner_invalid"):
            cursor.execute(
                "UPDATE public.lab_arena_rounds SET status = 'published', "
                "publication_doc = %s::jsonb, published_at = %s, "
                "king_outcome = 'crowned', king_hotkey = %s WHERE round_id = %s",
                (json.dumps(publication(high)), "2026-09-10T00:00:00Z", high["miner_hotkey"], round_id),
            )
    # A stored object failure is authoritative only as a fail-closed result.
    # It leaves the raw score visible but cannot make the top scorer king.
    rows[0]["eligible"] = False
    rows[0]["eligibility_reason"] = "stored_output_invalid"
    rows[0]["cost_summary"] = None
    # The next-highest eligible challenger can win. Baseline cost ineligibility
    # does not remove its quality score as the +1 promotion threshold.
    with superuser.cursor() as cursor:
        cursor.execute(
            "UPDATE public.lab_arena_rounds SET status = 'published', "
            "publication_doc = %s::jsonb, published_at = %s, "
            "king_outcome = 'crowned', king_hotkey = %s WHERE round_id = %s",
            (json.dumps(publication(low)), "2026-09-10T00:00:00Z", low["miner_hotkey"], round_id),
        )
    published = store.get_round(round_id)
    assert published["status"] == "published"
    assert published["publication_doc"]["king_decision"]["winner_submission_id"] == low["submission_id"]
    # Do not leave this fixture as an older pending promotion for later tests.
    promotion = {
        "commit": "1" * 40,
        "main_before": "2" * 40,
        "lab_before": "3" * 40,
        "timestamp": "2026-09-10T00:00:00Z",
    }
    assert store.prepare_promotion(round_id, promotion)["status"] == "prepared"
    assert store.complete_promotion(round_id, promotion)["status"] == "promoted"


def _publish_compact(store: ArenaStore, superuser, round_id: str, *, rewards_enabled: bool, mode: str = "live") -> str:
    runner = hotkey(round_id + "-runner")
    config = round_config(round_id, [runner], mode=mode, rewards_enabled=rewards_enabled)
    assert store.create_round(round_id, config)["status"] == "created"
    baseline_key = config["baseline_hotkey"]
    winner_key = hotkey(round_id + "-king")
    participants = [
        {
            "submission_id": round_id + "-baseline",
            "miner_hotkey": baseline_key,
            "is_king": True,
        },
        {
            "submission_id": round_id + "-winner",
            "miner_hotkey": winner_key,
            "is_king": False,
        },
    ]
    with superuser.cursor() as cursor:
        cursor.execute(
            "UPDATE public.lab_arena_rounds SET status = 'scored', "
            "participants = %s::jsonb, finalists = '[]'::jsonb WHERE round_id = %s",
            (json.dumps(participants), round_id),
        )
    published_at = "2026-09-02T00:00:00Z"
    result = store.transition_round(round_id, "scored", "published", {
        "publication_doc": _compact_publication(
            round_id, published_at, baseline_key, winner_key
        ),
        "published_at": published_at,
    })
    assert result["status"] == "ok"
    return published_at


def _reward_docs(round_id: str, published_at: str, epoch: int, king_key: str, *, marker: str = "") -> tuple:
    key_hash = sha("key" + marker)
    basis = {
        "schema_version": contracts.REWARD_BASIS_SCHEMA_VERSION,
        "round_id": round_id,
        "published_at": published_at,
        "effective_reward_epoch": epoch,
        "king_hotkey": king_key,
        "king_outcome": "crowned",
        "king_start_epoch": epoch,
        "reward_constants": rewards.reward_constants_document(),
        "reward_basis_hash": sha("basis" + marker),
        "signature": {"public_key_hash": key_hash},
    }
    return basis, {"public_key_hash": key_hash}


def _complete_compact_promotion(
    store: ArenaStore,
    superuser,
    round_id: str,
    published_at: str,
) -> str:
    king = store.get_round(round_id)["king_hotkey"]
    with superuser.cursor() as cursor:
        cursor.execute(
            "INSERT INTO public.lab_arena_submissions "
            "(submission_id, round_id, miner_hotkey, status, is_king) "
            "VALUES (%s, %s, %s, 'frozen', FALSE)",
            (round_id + "-winner", round_id, king),
        )
    plan = {
        "commit": "1" * 40,
        "main_before": "2" * 40,
        "lab_before": "3" * 40,
        "timestamp": published_at,
    }
    assert store.prepare_promotion(round_id, plan)["status"] == "prepared"
    assert store.complete_promotion(round_id, plan)["status"] == "promoted"
    return king


def test_compact_publication_is_independent_from_reward_activation(store, superuser):
    round_id = "arena-2026-09-02-pub"
    published_at = _publish_compact(
        store,
        superuser,
        round_id,
        rewards_enabled=False,
    )
    row = store.get_round(round_id)
    assert row["status"] == "published"
    assert row["publication_doc"]["round_id"] == round_id
    assert row["reward_activated_at"] is None
    assert row["reward_basis_doc"] is None
    with superuser.cursor() as cursor:
        cursor.execute("SELECT count(*) FROM public.lab_arena_reward_basis_v1 WHERE round_id = %s", (round_id,))
        assert cursor.fetchone()[0] == 0
    _complete_compact_promotion(store, superuser, round_id, published_at)
    promoted = store.get_round(round_id)
    assert promoted["reward_activated_at"] is None
    assert promoted["reward_basis_doc"] is None


def test_reward_activation_is_oldest_first_retry_idempotent_and_mismatch_safe(store, superuser):
    first = "arena-2026-09-03-rewarda"
    second = "arena-2026-09-03-rewardb"
    first_at = _publish_compact(store, superuser, first, rewards_enabled=True)
    second_at = _publish_compact(store, superuser, second, rewards_enabled=True)
    # New live winners must finish baseline promotion before rewards activate.
    first_king = _complete_compact_promotion(
        store,
        superuser,
        first,
        first_at,
    )
    second_king = _complete_compact_promotion(
        store,
        superuser,
        second,
        second_at,
    )
    first_basis, first_key = _reward_docs(first, first_at, 100, first_king, marker="a")
    second_basis, second_key = _reward_docs(second, second_at, 101, second_king, marker="b")
    assert store.activate_reward(second, second_basis, second_key)["status"] == "waiting_for_older_round"
    activated = store.activate_reward(first, first_basis, first_key)
    assert activated == {"status": "activated", "effective_reward_epoch": 100}
    assert store.activate_reward(first, first_basis, first_key) == {"status": "existing", "effective_reward_epoch": 100}
    with pytest.raises(ArenaStoreError, match="activation_mismatch"):
        store.activate_reward(first, dict(first_basis, reward_basis_hash=sha("different")), first_key)
    assert store.activate_reward(second, second_basis, second_key) == {"status": "activated", "effective_reward_epoch": 101}
    with superuser.cursor() as cursor:
        cursor.execute("SELECT round_id FROM public.lab_arena_reward_basis_v1 WHERE round_id = ANY(%s) ORDER BY effective_reward_epoch", ([first, second],))
        assert [row[0] for row in cursor.fetchall()] == [first, second]


def test_shadow_and_reward_disabled_rounds_never_activate(store, superuser):
    shadow = "arena-2026-09-04-shadow"
    shadow_at = _publish_compact(store, superuser, shadow, rewards_enabled=False, mode="shadow")
    shadow_basis, shadow_key = _reward_docs(shadow, shadow_at, 200, store.get_round(shadow)["king_hotkey"], marker="shadow")
    assert store.activate_reward(shadow, shadow_basis, shadow_key)["status"] == "disabled"
    disabled = "arena-2026-09-04-disabled"
    disabled_at = _publish_compact(store, superuser, disabled, rewards_enabled=False)
    disabled_basis, disabled_key = _reward_docs(disabled, disabled_at, 200, store.get_round(disabled)["king_hotkey"], marker="disabled")
    assert store.activate_reward(disabled, disabled_basis, disabled_key)["status"] == "disabled"
    with superuser.cursor() as cursor:
        cursor.execute("SELECT count(*) FROM public.lab_arena_reward_basis_v1 WHERE round_id = ANY(%s)", ([shadow, disabled],))
        assert cursor.fetchone()[0] == 0


# ---------------------------------------------------------------------------
# 18.2 assignments, leases, and stage close
# ---------------------------------------------------------------------------


def test_hundred_concurrent_claims_through_two_instances_never_duplicate(connect):
    round_id = "arena-2026-09-02-cc"
    setup = ArenaStore(PsycopgTransport(connect))
    runner_keys, parts = open_round(setup, round_id, participants=8, runners=100, prefix="cc")
    instances = [ArenaStore(PsycopgTransport(connect)), ArenaStore(PsycopgTransport(connect))]
    responses: List[Dict[str, Any]] = []
    lock = threading.Lock()

    def worker(index: int):
        response, _, _, _ = claim(instances[index % 2], round_id, runner_keys[index], parallelism=1)
        with lock:
            responses.append(response)

    with ThreadPoolExecutor(max_workers=32) as pool:
        list(pool.map(worker, range(100)))
    leased = [r for r in responses if r["status"] == "leased"]
    assert len(leased) == 80
    assert len([r for r in responses if r["status"] == "no_pending"]) == 20
    assert len({r["run_id"] for r in leased}) == 80
    assert len({r["assignment_id"] for r in leased}) == 80
    runs = setup.list_runs(round_id, stage=1, status="leased")
    assert len(runs) == 80
    # ICP-major order: all ten stage-one positions are leased once for each
    # of the eight participants, even under one hundred concurrent requests.
    positions = sorted(r["icp_position"] for r in leased)
    assert positions == sorted([p for p in range(10) for _ in range(8)])
    for instance in instances:
        instance.close()
    setup.close()


def test_concurrent_claims_for_one_runner_respect_one_slot(connect):
    round_id = "arena-2026-09-02-rcap"
    setup = ArenaStore(PsycopgTransport(connect))
    runner_keys, _ = open_round(
        setup,
        round_id,
        participants=2,
        runners=1,
        prefix="runner-capacity",
    )
    workers = [ArenaStore(PsycopgTransport(connect)) for _ in range(16)]
    start = threading.Barrier(len(workers))

    def worker(index: int):
        start.wait(timeout=30)
        return claim(
            workers[index],
            round_id,
            runner_keys[0],
            parallelism=1,
            ceiling=1,
        )[0]

    try:
        with ThreadPoolExecutor(max_workers=len(workers)) as pool:
            responses = list(pool.map(worker, range(len(workers))))
        assert [item["status"] for item in responses].count("leased") == 1
        refused = [item for item in responses if item["status"] == "no_free_slot"]
        assert len(refused) == len(workers) - 1
        assert all(item["active_leases"] == 1 and item["slot_limit"] == 1 for item in refused)
        assert len(setup.list_runs(round_id, stage=1, status="leased")) == 1
    finally:
        for worker_store in workers:
            worker_store.close()
        setup.close()


def test_claim_replay_reuse_ceiling_and_self_exclusion(store, connect):
    round_id = "arena-2026-09-02-cl"
    runners, parts = open_round(store, round_id, participants=2, runners=2, prefix="cl")
    first, token, request_id, request_hash = claim(store, round_id, runners[0], parallelism=2)
    assert first["status"] == "leased" and first["icp_position"] == 0
    # Replay with the same bytes returns the stored response, also through a fresh instance.
    replay = store.claim_assignment(round_id=round_id, runner_hotkey=runners[0], declared_parallelism=2, slot_ceiling=8, excluded_miner_hotkeys=[], request_id=request_id, request_hash=request_hash, lease_token_hash=hash_lease_token(new_lease_token()))
    assert replay == first
    fresh = ArenaStore(PsycopgTransport(connect))
    assert fresh.claim_assignment(round_id=round_id, runner_hotkey=runners[0], declared_parallelism=2, slot_ceiling=8, excluded_miner_hotkeys=[], request_id=request_id, request_hash=request_hash, lease_token_hash=hash_lease_token(new_lease_token())) == first
    fresh.close()
    reused = store.claim_assignment(round_id=round_id, runner_hotkey=runners[0], declared_parallelism=2, slot_ceiling=8, excluded_miner_hotkeys=[], request_id=request_id, request_hash=sha("different"), lease_token_hash=hash_lease_token(new_lease_token()))
    assert reused["status"] == "request_id_reused"
    # Slot ceiling: declared 5 but ceiling 2 -> second lease ok, third refused.
    second, *_ = claim(store, round_id, runners[0], parallelism=5, ceiling=2)
    assert second["status"] == "leased"
    third, *_ = claim(store, round_id, runners[0], parallelism=5, ceiling=2)
    assert third["status"] == "no_free_slot" and third["slot_limit"] == 2
    # Declared parallelism 1 with a large ceiling also refuses.
    assert claim(store, round_id, runners[1], parallelism=1, ceiling=8)[0]["status"] == "leased"
    assert claim(store, round_id, runners[1], parallelism=1, ceiling=8)[0]["status"] == "no_free_slot"
    # Self-execution exclusion: excluding both miners leaves nothing.
    excluded, *_ = claim(store, round_id, hotkey("cl-runner-0"), parallelism=8, ceiling=8, excluded=[p["miner_hotkey"] for p in parts])
    assert excluded["status"] == "no_pending"
    only_one, *_ = claim(store, round_id, hotkey("cl-runner-0"), parallelism=8, ceiling=8, excluded=[parts[0]["miner_hotkey"]])
    assert only_one["status"] == "leased" and only_one["miner_hotkey"] == parts[1]["miner_hotkey"]


def test_claim_rpc_accepts_gateway_authorized_hotkey_absent_from_round_list(
    store, superuser
):
    assert store.validator_scoring_authority_schema() == {
        "schema_version": "leadpoet.lab_arena.validator_scoring_authority.v1",
        "version": 208,
        "authority": "gateway_subnet_validator_role",
    }
    with superuser.cursor() as cursor:
        for role, expected in (
            ("lab_arena_service", True),
            ("service_role", False),
            ("anon", False),
            ("authenticated", False),
        ):
            cursor.execute(
                "SELECT has_function_privilege(%s, "
                "'public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)', "
                "'EXECUTE')",
                (role,),
            )
            assert cursor.fetchone() == (expected,)
    round_id = "arena-2026-09-02-va"
    configured, _parts = open_round(
        store, round_id, participants=1, runners=1, prefix="va"
    )
    stranger = hotkey("va-stranger")
    assert stranger not in configured

    response, *_ = claim(store, round_id, stranger)
    assert response["status"] == "leased"
    assert response["run_id"]


def test_stale_lease_fails_provider_event_and_completion_and_expiry_retries_once(store, superuser):
    round_id = "arena-2026-09-02-lx"
    # Two Deepline calls per ICP attempt and a single attempt in the stage quota: the retry after
    # expiry inherits the lost attempt's uncertain call and is refused by the stage quota once
    # the per-ICP quota still has room.
    runners, parts = open_round(store, round_id, participants=1, prefix="lx", quotas={"deepline": 2, "scrapingdog": 30, "openrouter": 60}, stage_1_icps=1, max_attempts=1)
    response, token, _, _ = claim(store, round_id, runners[0], parallelism=8)
    run_id, lease_hash = response["run_id"], hash_lease_token(token)
    identity = contracts.provider_call_identity(attempt=1, assignment_id=response["assignment_id"], icp_position=0, action_sequence=0, operation_id="deepline.execute", request_hash=sha("q1"))
    dispatched_identity = contracts.provider_call_identity(attempt=1, assignment_id=response["assignment_id"], icp_position=0, action_sequence=1, operation_id="deepline.execute", request_hash=sha("q2"))
    assert store.reserve_call(run_id=run_id, lease_token_hash=lease_hash, call_identity=identity, operation_id="deepline.execute", provider="deepline", funding_source="miner_key", amount_microusd=0, call_doc={})["status"] == "reserved"
    assert store.reserve_call(run_id=run_id, lease_token_hash=lease_hash, call_identity=dispatched_identity, operation_id="deepline.execute", provider="deepline", funding_source="miner_key", amount_microusd=0, call_doc={})["status"] == "reserved"
    assert store.mark_dispatched(run_id=run_id, lease_token_hash=lease_hash, call_identity=dispatched_identity)["status"] == "dispatched"
    # Wrong token is stale everywhere.
    wrong = hash_lease_token("other")
    assert store.mark_dispatched(run_id=run_id, lease_token_hash=wrong, call_identity=identity)["status"] == "stale"
    # Expiry: recover undispatched once, uncertain for dispatched, second attempt with fresh cap.
    expire_now(superuser, run_id)
    assert store.expire_leases(round_id) == {"status": "ok", "expired": 1, "retried": 1}
    assert store.expire_leases(round_id) == {"status": "ok", "expired": 0, "retried": 0}
    heads = {row["call_identity"]: row["entry_kind"] for row in store.list_ledger(run_id=run_id)}
    assert heads[identity] == "recovery" and heads[dispatched_identity] == "uncertain"
    for call in (store.reserve_call(run_id=run_id, lease_token_hash=lease_hash, call_identity=sha("x"), operation_id="deepline.execute", provider="deepline", funding_source="miner_key", amount_microusd=0, call_doc={}),
                 store.settle_call(run_id=run_id, lease_token_hash=lease_hash, call_identity=dispatched_identity, actual_microusd=0, terminal_response={})):
        assert call["status"] == "stale"
    terminal_replay = complete(
        store,
        run_id,
        lease_hash,
        "accepted",
        output_ref="ref",
    )
    assert terminal_replay["status"] == "failed" and terminal_replay["idempotent"] is True
    old = store.get_run(run_id)
    assert old["status"] == "failed" and old["terminal_cause"] == "lease_expired"
    second, token2, _, _ = claim(store, round_id, runners[0], parallelism=8)
    assert second["assignment_id"] == response["assignment_id"] and second["attempt"] == 2
    assert "per_icp_cap_microusd" not in second and second["lease_generation"] == 2
    # The stage quota still counts the lost attempt's uncertain call (the recovered one does not).
    lease2 = hash_lease_token(token2)
    heavy = contracts.provider_call_identity(attempt=1, assignment_id=second["assignment_id"], icp_position=0, action_sequence=0, operation_id="deepline.execute", request_hash=sha("heavy"))
    first_ok = contracts.provider_call_identity(attempt=1, assignment_id=second["assignment_id"], icp_position=0, action_sequence=1, operation_id="deepline.execute", request_hash=sha("first-ok"))
    assert store.reserve_call(run_id=second["run_id"], lease_token_hash=lease2, call_identity=first_ok, operation_id="deepline.execute", provider="deepline", funding_source="miner_key", amount_microusd=0, call_doc={})["status"] == "reserved"
    refused = store.reserve_call(run_id=second["run_id"], lease_token_hash=lease2, call_identity=heavy, operation_id="deepline.execute", provider="deepline", funding_source="miner_key", amount_microusd=0, call_doc={})
    assert refused["status"] == "refused" and refused["reason"] == "stage_quota"
    # Another provider's quota is independent, and the refusal is recorded and replayed.
    other = contracts.provider_call_identity(attempt=1, assignment_id=second["assignment_id"], icp_position=0, action_sequence=2, operation_id="scrapingdog.google", request_hash=sha("other"))
    assert store.reserve_call(run_id=second["run_id"], lease_token_hash=lease2, call_identity=other, operation_id="scrapingdog.google", provider="scrapingdog", funding_source="miner_key", amount_microusd=0, call_doc={})["status"] == "reserved"
    assert store.reserve_call(run_id=second["run_id"], lease_token_hash=lease2, call_identity=heavy, operation_id="deepline.execute", provider="deepline", funding_source="miner_key", amount_microusd=0, call_doc={})["status"] == "refused"
    # No third attempt after the second expires.
    expire_now(superuser, second["run_id"])
    assert store.expire_leases(round_id) == {"status": "ok", "expired": 1, "retried": 0}
    assert claim(store, round_id, runners[0], parallelism=8)[0]["status"] == "leased"  # other positions remain
    attempts = [r for r in store.list_runs(round_id, stage=1) if r["assignment_id"] == response["assignment_id"]]
    assert sorted(r["attempt"] for r in attempts) == [1, 2]


def test_model_caused_failure_gets_one_confirmation_attempt_and_completion_requires_closed_accounting(store):
    round_id = "arena-2026-09-02-mc"
    runners, parts = open_round(store, round_id, participants=1, prefix="mc")
    response, token, _, _ = claim(store, round_id, runners[0], parallelism=8)
    run_id, lease_hash = response["run_id"], hash_lease_token(token)
    identity = contracts.provider_call_identity(attempt=1, assignment_id=response["assignment_id"], icp_position=0, action_sequence=0, operation_id="openrouter.chat", request_hash=sha("q"))
    assert store.reserve_call(run_id=run_id, lease_token_hash=lease_hash, call_identity=identity, operation_id="openrouter.chat", provider="openrouter", funding_source="miner_key", amount_microusd=9000, call_doc={})["status"] == "reserved"
    blocked = complete(store, run_id, lease_hash, "model_timeout")
    assert blocked == {"status": "accounting_open", "open_calls": 1}
    assert store.mark_dispatched(run_id=run_id, lease_token_hash=lease_hash, call_identity=identity)["status"] == "dispatched"
    settled = store.settle_call(run_id=run_id, lease_token_hash=lease_hash, call_identity=identity, actual_microusd=4000, terminal_response={"status": 200})
    assert settled["status"] == "settled" and settled["released_microusd"] == 5000
    again = store.settle_call(run_id=run_id, lease_token_hash=lease_hash, call_identity=identity, actual_microusd=1, terminal_response={})
    assert again["status"] == "settled" and again["idempotent"] is True and again["terminal_response"] == {"status": 200}
    done = complete(store, run_id, lease_hash, "model_timeout")
    assert done["status"] == "failed"
    stored = store.get_run(run_id)
    assert stored["result_doc"] == {"terminal_status": "model_timeout"}
    assert stored["output_ref"] is None and "output_hash" not in stored
    assert not {
        "receipt_doc",
        "receipt_hash",
        "provider_call_root",
        "private_event_root",
        "cost_root",
    }.intersection(stored)
    replay = complete(store, run_id, lease_hash, "model_timeout")
    assert replay["status"] == "failed" and replay["idempotent"] is True
    assert store.expire_leases(round_id)["expired"] == 0
    # One confirmation attempt exists for the model-caused failure; with a single validator it
    # claims that attempt itself, first in ICP order.
    assert done["confirmation_attempt"] == 2
    attempts = [r for r in store.list_runs(round_id, stage=1) if r["assignment_id"] == response["assignment_id"]]
    assert [(r["attempt"], r["status"]) for r in attempts] == [(1, "failed"), (2, "pending")]
    assert attempts[1]["previous_runner_hotkey"] == runners[0]
    nxt, token2, _, _ = claim(store, round_id, runners[0], parallelism=8)
    assert nxt["status"] == "leased" and nxt["icp_position"] == 0 and nxt["attempt"] == 2
    # The second failure stands: no third attempt, and a quota exhaustion never gets a confirmation.
    second = complete(store, nxt["run_id"], hash_lease_token(token2), "model_error")
    assert second["status"] == "failed" and "confirmation_attempt" not in second
    assert [r["attempt"] for r in store.list_runs(round_id, stage=1) if r["assignment_id"] == response["assignment_id"]] == [1, 2]
    quota, token3, _, _ = claim(store, round_id, runners[0], parallelism=8)
    assert quota["icp_position"] == 1
    exhausted = complete(store, quota["run_id"], hash_lease_token(token3), "budget_exhausted")
    assert exhausted["status"] == "failed" and "confirmation_attempt" not in exhausted
    assert [r["attempt"] for r in store.list_runs(round_id, stage=1) if r["assignment_id"] == quota["assignment_id"]] == [1]


def test_attempt_completion_validates_result_and_output_pair(store):
    round_id = "arena-2026-09-02-result"
    runners, _ = open_round(store, round_id, participants=1, prefix="result-contract")
    response, token, _, _ = claim(store, round_id, runners[0])
    lease_hash = hash_lease_token(token)
    with pytest.raises(ArenaStoreError, match="lab_arena_complete_input_invalid"):
        store.complete_attempt(
            run_id=response["run_id"],
            lease_token_hash=lease_hash,
            result={"terminal_status": "model_error"},
            terminal_cause="accepted",
            output_ref="arena/output.json",
        )
    with pytest.raises(ArenaStoreError, match="lab_arena_complete_input_invalid"):
        complete(
            store,
            response["run_id"],
            lease_hash,
            "model_error",
            output_ref="arena/output.json",
        )
    accepted = complete(
        store,
        response["run_id"],
        lease_hash,
        "accepted",
        output_ref="arena/output.json",
    )
    assert accepted["status"] == "accepted"


def test_close_stage_races_hundred_operations_without_deadlock(connect):
    round_id = "arena-2026-09-02-cs"
    setup = ArenaStore(PsycopgTransport(connect))
    runners, parts = open_round(setup, round_id, participants=10, runners=1, prefix="cs")
    leases = []
    for _ in range(100):
        response, token, _, _ = claim(setup, round_id, runners[0], parallelism=100, ceiling=100)
        assert response["status"] == "leased"
        leases.append((response, hash_lease_token(token)))
    # Prepare one reserved and one dispatched call on every lease before the race.
    for index, (response, lease_hash) in enumerate(leases):
        reserved = contracts.provider_call_identity(attempt=1, assignment_id=response["assignment_id"], icp_position=response["icp_position"], action_sequence=0, operation_id="deepline.execute", request_hash=sha("r%d" % index))
        dispatched = contracts.provider_call_identity(attempt=1, assignment_id=response["assignment_id"], icp_position=response["icp_position"], action_sequence=1, operation_id="deepline.execute", request_hash=sha("d%d" % index))
        assert setup.reserve_call(run_id=response["run_id"], lease_token_hash=lease_hash, call_identity=reserved, operation_id="deepline.execute", provider="deepline", funding_source="miner_key", amount_microusd=0, call_doc={})["status"] == "reserved"
        assert setup.reserve_call(run_id=response["run_id"], lease_token_hash=lease_hash, call_identity=dispatched, operation_id="deepline.execute", provider="deepline", funding_source="miner_key", amount_microusd=0, call_doc={})["status"] == "reserved"
        assert setup.mark_dispatched(run_id=response["run_id"], lease_token_hash=lease_hash, call_identity=dispatched)["status"] == "dispatched"
    workers = [ArenaStore(PsycopgTransport(connect)) for _ in range(2)]
    closer = ArenaStore(PsycopgTransport(connect))
    outcomes: List[str] = []
    lock = threading.Lock()
    start = threading.Barrier(9)

    def late_work(index: int):
        response, lease_hash = leases[index]
        worker = workers[index % 2]
        kind = index % 5
        start.wait(timeout=30)
        identity_r = contracts.provider_call_identity(attempt=1, assignment_id=response["assignment_id"], icp_position=response["icp_position"], action_sequence=0, operation_id="deepline.execute", request_hash=sha("r%d" % index))
        identity_d = contracts.provider_call_identity(attempt=1, assignment_id=response["assignment_id"], icp_position=response["icp_position"], action_sequence=1, operation_id="deepline.execute", request_hash=sha("d%d" % index))
        if kind == 0:
            result = claim(worker, round_id, runners[0], parallelism=100, ceiling=100)[0]
        elif kind == 1:
            result = worker.reserve_call(run_id=response["run_id"], lease_token_hash=lease_hash, call_identity=sha("new%d" % index), operation_id="deepline.execute", provider="deepline", funding_source="miner_key", amount_microusd=0, call_doc={})
        elif kind == 2:
            result = worker.mark_dispatched(run_id=response["run_id"], lease_token_hash=lease_hash, call_identity=identity_r)
        elif kind == 3:
            result = worker.settle_call(run_id=response["run_id"], lease_token_hash=lease_hash, call_identity=identity_d, actual_microusd=100, terminal_response={"ok": True})
        else:
            result = complete(worker, response["run_id"], lease_hash, "accepted", output_ref="ref")
        with lock:
            outcomes.append("%d:%s" % (kind, result["status"]))

    def close():
        start.wait(timeout=30)
        result = closer.close_stage(round_id, 1)
        with lock:
            outcomes.append("close:%s" % result["status"])

    threads = [threading.Thread(target=late_work, args=(i,)) for i in range(8)]
    threads.append(threading.Thread(target=close))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)
    assert not any(thread.is_alive() for thread in threads), "deadlock or hang"
    # Remaining 92 leases are handled after the close: every late op is stale.
    for index in range(8, 100):
        response, lease_hash = leases[index]
        identity_d = contracts.provider_call_identity(attempt=1, assignment_id=response["assignment_id"], icp_position=response["icp_position"], action_sequence=1, operation_id="deepline.execute", request_hash=sha("d%d" % index))
        assert workers[index % 2].mark_dispatched(run_id=response["run_id"], lease_token_hash=lease_hash, call_identity=identity_d)["status"] == "stale"
    assert "close:cancelled" in outcomes or "close:closed" in outcomes
    final = setup.get_round(round_id)
    assert final["status"] == "cancelled"  # 100 in-flight leases lacked accepted results for an infrastructure reason
    heads: Dict[str, str] = {}
    for row in setup.list_ledger():
        if row["round_id"] == round_id and row["call_identity"]:
            heads[row["call_identity"]] = row["entry_kind"]
    assert not any(kind in ("reservation", "dispatch") for kind in heads.values()), "an open call survived the close"
    assert all(r["status"] == "failed" for r in setup.list_runs(round_id, stage=1) if r["attempt"] >= 1)
    for instance in workers + [closer, setup]:
        instance.close()


def test_close_stage_with_only_model_failures_freezes_results(store):
    round_id = "arena-2026-09-02-cf"
    runners, parts = open_round(store, round_id, participants=1, prefix="cf")
    # Even positions are accepted; odd positions fail as invalid output, and each such failure
    # gets one confirmation attempt (claimed next, in ICP order) that fails the same way.
    seen = []
    while True:
        response, token, _, _ = claim(store, round_id, runners[0], parallelism=8)
        if response["status"] != "leased":
            break
        position = response["icp_position"]
        cause = "accepted" if position % 2 == 0 else "invalid_output"
        result = complete(
            store,
            response["run_id"],
            hash_lease_token(token),
            cause,
            output_ref="ref" if cause == "accepted" else "",
        )
        assert result["status"] == ("accepted" if cause == "accepted" else "failed")
        seen.append((position, response["attempt"]))
    assert sorted(seen) == sorted([(p, 1) for p in range(0, 10, 2)] + [(p, a) for p in range(1, 10, 2) for a in (1, 2)])
    closed = store.close_stage(round_id, 1)
    assert closed["status"] == "closed" and closed["incomplete_assignments"] == 0
    assert store.close_stage(round_id, 1)["status"] == "existing"
    assert store.get_round(round_id)["status"] == "stage1_closed"
    # Scores are write-once per attempt.
    runs = store.list_runs(round_id, stage=1)
    scores = [{"run_id": r["run_id"], "per_icp_score": 12.5 if r["status"] == "accepted" else 0.0} for r in runs]
    assert len(runs) == 15  # ten first attempts and five confirmation attempts
    assert store.record_run_scores(round_id, 1, scores)["recorded"] == 15
    assert store.record_run_scores(round_id, 1, scores)["existing"] == 15
    with pytest.raises(ArenaStoreError):
        store.record_run_scores(round_id, 1, [dict(scores[0], per_icp_score=1.0)])


def test_stage_two_opens_only_for_the_finalist_and_incumbent(store):
    round_id = "arena-2026-09-02-kg"
    runner_keys = [hotkey("kg-runner")]
    assert store.create_round(round_id, round_config(round_id, runner_keys))["status"] == "created"
    parts = frozen_participants(store, round_id, 2, prefix="kg", king_index=1)
    commit_round(store, round_id, parts)
    assert store.open_stage(round_id, 1, parts, stage_positions(1))["status"] == "ok"
    king_runs = store.list_runs(round_id, stage=1, submission_id=parts[1]["submission_id"])
    assert len(king_runs) == 10 and all(r["attempt"] == 1 and r["status"] == "pending" for r in king_runs)
    executed = _execute_everything(store, round_id, runner_keys[0])
    assert len(executed) == 20
    assert store.close_stage(round_id, 1)["status"] == "closed"
    _commit_plan(store, round_id, 1)
    assert store.open_scoring(round_id, 1, _scoring_items(executed))["assignments"] == 20
    while True:
        response, token, _, _ = claim(store, round_id, runner_keys[0], parallelism=100, ceiling=100)
        if response["status"] != "leased":
            break
        assert response["kind"] == "score"
        assert _complete_accepted(store, response["run_id"], hash_lease_token(token), response["run_id"])["status"] == "accepted"
    assert store.close_scoring(round_id, 1)["round_status"] == "stage1_judged"
    execute_runs = store.list_runs(round_id, stage=1, kind="execute")
    scores = [{"run_id": run["run_id"], "per_icp_score": 50.0} for run in execute_runs]
    assert store.record_run_scores(round_id, 1, scores)["recorded"] == 20
    finalist = parts[0]["submission_id"]
    assert store.transition_round(round_id, "stage1_judged", "stage1_scored", {
        "finalists": [finalist],
    })["round_status"] == "stage1_scored"
    # Stage 2 cannot omit the incumbent.
    with pytest.raises(ArenaStoreError):
        store.open_stage(round_id, 2, [parts[0]], stage_positions(2))
    opened = store.open_stage(round_id, 2, parts, stage_positions(2))
    assert opened["status"] == "ok" and opened["assignments"] == 20
    king_stage_2 = store.list_runs(round_id, stage=2, submission_id=parts[1]["submission_id"])
    assert len(king_stage_2) == 10 and {run["icp_position"] for run in king_stage_2} == set(range(10, 20))
    stage_2_executed = _execute_everything(store, round_id, runner_keys[0])
    assert len(stage_2_executed) == 20
    assert store.close_stage(round_id, 2)["round_status"] == "stage2_closed"
    _commit_plan(store, round_id, 2)
    assert store.open_scoring(round_id, 2, _scoring_items(stage_2_executed))["assignments"] == 20
    while True:
        response, token, _, _ = claim(store, round_id, runner_keys[0], parallelism=100, ceiling=100)
        if response["status"] != "leased":
            break
        assert response["kind"] == "score" and response["stage"] == 2
        assert _complete_accepted(store, response["run_id"], hash_lease_token(token), response["run_id"])["status"] == "accepted"
    assert store.close_scoring(round_id, 2)["round_status"] == "stage2_judged"
    stage_2_runs = store.list_runs(round_id, stage=2, kind="execute")
    stage_2_scores = [{"run_id": run["run_id"], "per_icp_score": 60.0} for run in stage_2_runs]
    assert store.record_run_scores(round_id, 2, stage_2_scores)["recorded"] == 20
    assert store.transition_round(
        round_id,
        "stage2_judged",
        "scored",
        {},
    )["round_status"] == "scored"


# ---------------------------------------------------------------------------
# 18.3 accounting and dispatch
# ---------------------------------------------------------------------------


def test_concurrent_reservations_never_exceed_the_call_quota(connect):
    round_id = "arena-2026-09-02-ov"
    setup = ArenaStore(PsycopgTransport(connect))
    runners, parts = open_round(setup, round_id, participants=1, prefix="ov", quotas={"deepline": 12, "scrapingdog": 30, "openrouter": 60})
    response, token, _, _ = claim(setup, round_id, runners[0], parallelism=8)
    lease_hash = hash_lease_token(token)
    instances = [ArenaStore(PsycopgTransport(connect)), ArenaStore(PsycopgTransport(connect))]
    results = []
    lock = threading.Lock()

    def reserve(index: int):
        identity = contracts.provider_call_identity(attempt=1, assignment_id=response["assignment_id"], icp_position=0, action_sequence=index, operation_id="deepline.execute", request_hash=sha("c%d" % index))
        result = instances[index % 2].reserve_call(run_id=response["run_id"], lease_token_hash=lease_hash, call_identity=identity, operation_id="deepline.execute", provider="deepline", funding_source="miner_key", amount_microusd=0, call_doc={})
        with lock:
            results.append((result["status"], result.get("reason")))

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(reserve, range(50)))
    statuses = [status for status, _ in results]
    assert statuses.count("reserved") == 12  # the per-ICP Deepline quota, under sixteen concurrent workers
    assert statuses.count("refused") == 38 and {reason for status, reason in results if status == "refused"} == {"per_icp_quota"}
    # Quotas are per provider: Scrapingdog calls on the same attempt are untouched by the Deepline quota.
    other = contracts.provider_call_identity(attempt=1, assignment_id=response["assignment_id"], icp_position=0, action_sequence=99, operation_id="scrapingdog.google", request_hash=sha("g"))
    assert setup.reserve_call(run_id=response["run_id"], lease_token_hash=lease_hash, call_identity=other, operation_id="scrapingdog.google", provider="scrapingdog", funding_source="miner_key", amount_microusd=0, call_doc={})["status"] == "reserved"
    # A miner run cannot fall back to the organizer's host key.
    blocked = contracts.provider_call_identity(attempt=1, assignment_id=response["assignment_id"], icp_position=0, action_sequence=100, operation_id="scrapingdog.google", request_hash=sha("blocked"))
    with pytest.raises(ArenaStoreError, match="lab_arena_funding_source_mismatch"):
        setup.reserve_call(run_id=response["run_id"], lease_token_hash=lease_hash, call_identity=blocked, operation_id="scrapingdog.google", provider="scrapingdog", funding_source="host", amount_microusd=0, call_doc={})
    for instance in instances + [setup]:
        instance.close()


class _noop:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_openrouter_host_quota_dispatch_uniqueness_and_settlement_rules(connect):
    round_id = "arena-2026-09-02-or"
    setup = ArenaStore(PsycopgTransport(connect))
    runners, _parts = open_round(
        setup,
        round_id,
        participants=1,
        prefix="or",
        quotas={"deepline": 30, "scrapingdog": 30, "openrouter": 10},
        execution_cap_microusd=10_000_000,
    )
    response, token, _, _ = claim(setup, round_id, runners[0], parallelism=8)
    lease_hash = hash_lease_token(token)
    # The fixed host quota admits ten calls and refuses the remaining five.
    results = []
    lock = threading.Lock()
    instances = [ArenaStore(PsycopgTransport(connect)), ArenaStore(PsycopgTransport(connect))]

    def reserve(index: int):
        identity = contracts.provider_call_identity(attempt=1, assignment_id=response["assignment_id"], icp_position=0, action_sequence=index, operation_id="openrouter.chat", request_hash=sha("o%d" % index))
        result = instances[index % 2].reserve_call(run_id=response["run_id"], lease_token_hash=lease_hash, call_identity=identity, operation_id="openrouter.chat", provider="openrouter", funding_source="miner_key", amount_microusd=1_000_000, call_doc={})
        with lock:
            results.append((index, result["status"]))

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(reserve, range(15)))
    statuses = [status for _, status in results]
    assert statuses.count("reserved") == 10 and statuses.count("refused") == 5
    reserved_index = next(index for index, status in results if status == "reserved")
    identity = contracts.provider_call_identity(attempt=1, assignment_id=response["assignment_id"], icp_position=0, action_sequence=reserved_index, operation_id="openrouter.chat", request_hash=sha("o%d" % reserved_index))
    # Two instances mark the same dispatch: exactly one non-idempotent dispatch.
    dispatch_results = []

    def dispatch(index: int):
        result = instances[index % 2].mark_dispatched(run_id=response["run_id"], lease_token_hash=lease_hash, call_identity=identity)
        with lock:
            dispatch_results.append(result["idempotent"])

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(dispatch, range(4)))
    assert dispatch_results.count(False) == 1 and dispatch_results.count(True) == 3
    assert len([row for row in setup.list_ledger(call_identity=identity) if row["entry_kind"] == "dispatch"]) == 1
    # Settlement records a genuine charge even when it exceeds the estimate.
    settled = setup.settle_call(run_id=response["run_id"], lease_token_hash=lease_hash, call_identity=identity, actual_microusd=1_000_001, terminal_response={"usage": {"total_tokens": 10}})
    assert settled["status"] == "settled" and settled["released_microusd"] == -1
    replayed_settlement = setup.settle_call(run_id=response["run_id"], lease_token_hash=lease_hash, call_identity=identity, actual_microusd=250_000, terminal_response={})
    assert replayed_settlement["status"] == "settled" and replayed_settlement["idempotent"] is True
    # A worker-reported uncertain call consumes its full reservation and a late settle cannot release it.
    uncertain_index = next(index for index, status in results if status == "reserved" and index != reserved_index)
    uid = contracts.provider_call_identity(attempt=1, assignment_id=response["assignment_id"], icp_position=0, action_sequence=uncertain_index, operation_id="openrouter.chat", request_hash=sha("o%d" % uncertain_index))
    assert setup.mark_dispatched(run_id=response["run_id"], lease_token_hash=lease_hash, call_identity=uid)["status"] == "dispatched"
    assert setup.mark_uncertain(run_id=response["run_id"], lease_token_hash=lease_hash, call_identity=uid, call_doc={"why": "timeout"})["status"] == "uncertain"
    late = setup.settle_call(run_id=response["run_id"], lease_token_hash=lease_hash, call_identity=uid, actual_microusd=0, terminal_response={})
    assert late["status"] == "uncertain" and late["amount_microusd"] == 1_000_000
    # Settling a merely reserved (never dispatched) call is refused.
    rid = next(index for index, status in results if status == "reserved" and index not in (reserved_index, uncertain_index))
    r_identity = contracts.provider_call_identity(attempt=1, assignment_id=response["assignment_id"], icp_position=0, action_sequence=rid, operation_id="openrouter.chat", request_hash=sha("o%d" % rid))
    assert setup.settle_call(run_id=response["run_id"], lease_token_hash=lease_hash, call_identity=r_identity, actual_microusd=1, terminal_response={})["status"] == "reserved"
    for instance in instances + [setup]:
        instance.close()


def test_openrouter_execution_money_cap_is_atomic_and_counts_each_call_once(connect):
    round_id = "arena-2026-09-02-cap"
    setup = ArenaStore(PsycopgTransport(connect))
    runners, _parts = open_round(
        setup,
        round_id,
        participants=1,
        prefix="money",
        quotas={"deepline": 30, "scrapingdog": 30, "openrouter": 60},
        execution_cap_microusd=3_000_000,
    )
    leased = [claim(setup, round_id, runners[0], parallelism=100, ceiling=100)[:2] for _ in range(10)]
    instances = [ArenaStore(PsycopgTransport(connect)), ArenaStore(PsycopgTransport(connect))]
    results = []
    lock = threading.Lock()

    def reserve(index: int):
        response, token = leased[index]
        identity = contracts.provider_call_identity(
            attempt=1,
            assignment_id=response["assignment_id"],
            icp_position=response["icp_position"],
            action_sequence=0,
            operation_id="openrouter.chat",
            request_hash=sha("money-%d" % index),
        )
        result = instances[index % 2].reserve_call(
            run_id=response["run_id"],
            lease_token_hash=hash_lease_token(token),
            call_identity=identity,
            operation_id="openrouter.chat",
            provider="openrouter",
            funding_source="miner_key",
            amount_microusd=1_000_000,
            call_doc={},
        )
        with lock:
            results.append((index, identity, result))

    with ThreadPoolExecutor(max_workers=10) as pool:
        list(pool.map(reserve, range(10)))
    reserved = [(index, identity) for index, identity, result in results if result["status"] == "reserved"]
    busy = [result for _index, _identity, result in results if result["status"] == "budget_busy"]
    assert len(reserved) == 3
    assert len(busy) == 7

    # The same identity is idempotent and does not consume the cap twice.
    first_index, first_identity = reserved[0]
    first_response, first_token = leased[first_index]
    retry = setup.reserve_call(
        run_id=first_response["run_id"],
        lease_token_hash=hash_lease_token(first_token),
        call_identity=first_identity,
        operation_id="openrouter.chat",
        provider="openrouter",
        funding_source="miner_key",
        amount_microusd=1_000_000,
        call_doc={},
    )
    assert retry["status"] == "reserved" and retry["idempotent"] is True

    # Latest states are authoritative: settlement uses actual cost, uncertain
    # keeps the full reservation, and an open reservation keeps its maximum.
    assert setup.mark_dispatched(
        run_id=first_response["run_id"],
        lease_token_hash=hash_lease_token(first_token),
        call_identity=first_identity,
    )["status"] == "dispatched"
    assert setup.settle_call(
        run_id=first_response["run_id"],
        lease_token_hash=hash_lease_token(first_token),
        call_identity=first_identity,
        actual_microusd=250_000,
        terminal_response={},
    )["status"] == "settled"
    second_index, second_identity = reserved[1]
    second_response, second_token = leased[second_index]
    assert setup.mark_dispatched(
        run_id=second_response["run_id"],
        lease_token_hash=hash_lease_token(second_token),
        call_identity=second_identity,
    )["status"] == "dispatched"
    assert setup.mark_uncertain(
        run_id=second_response["run_id"],
        lease_token_hash=hash_lease_token(second_token),
        call_identity=second_identity,
        call_doc={},
    )["status"] == "uncertain"

    third_index, _third_identity = reserved[2]
    third_response, third_token = leased[third_index]
    exact_identity = contracts.provider_call_identity(
        attempt=1,
        assignment_id=third_response["assignment_id"],
        icp_position=third_response["icp_position"],
        action_sequence=1,
        operation_id="openrouter.chat",
        request_hash=sha("money-exact"),
    )
    assert setup.reserve_call(
        run_id=third_response["run_id"],
        lease_token_hash=hash_lease_token(third_token),
        call_identity=exact_identity,
        operation_id="openrouter.chat",
        provider="openrouter",
        funding_source="miner_key",
        amount_microusd=750_000,
        call_doc={},
    )["status"] == "reserved"
    over_identity = contracts.provider_call_identity(
        attempt=1,
        assignment_id=third_response["assignment_id"],
        icp_position=third_response["icp_position"],
        action_sequence=2,
        operation_id="openrouter.chat",
        request_hash=sha("money-over"),
    )
    over = setup.reserve_call(
        run_id=third_response["run_id"],
        lease_token_hash=hash_lease_token(third_token),
        call_identity=over_identity,
        operation_id="openrouter.chat",
        provider="openrouter",
        funding_source="miner_key",
        amount_microusd=1,
        call_doc={},
    )
    assert over["status"] == "budget_busy"
    assert setup.list_ledger(call_identity=over_identity) == []
    for instance in instances + [setup]:
        instance.close()


def test_combined_provider_cap_is_atomic_and_actual_overage_is_preserved(connect):
    round_id = "arena-2026-09-10-combinedcap"
    setup = ArenaStore(PsycopgTransport(connect))
    runners, _parts = open_round(
        setup,
        round_id,
        participants=1,
        runners=1,
        prefix="combined-cap",
        quotas={"deepline": 30, "scrapingdog": 30, "openrouter": 30},
        execution_cap_microusd=3_000_000,
    )
    leased = [
        claim(setup, round_id, runners[0], parallelism=8, ceiling=8)[:2]
        for _ in range(4)
    ]
    providers = ("openrouter", "deepline", "scrapingdog", "deepline")
    operations = ("openrouter.chat", "deepline.execute", "scrapingdog.google", "deepline.execute")
    instances = [ArenaStore(PsycopgTransport(connect)) for _ in range(2)]

    def reserve(index: int):
        response, token = leased[index]
        identity = contracts.provider_call_identity(
            attempt=1,
            assignment_id=response["assignment_id"],
            icp_position=response["icp_position"],
            action_sequence=0,
            operation_id=operations[index],
            request_hash=sha("combined-%d" % index),
        )
        result = instances[index % 2].reserve_call(
            run_id=response["run_id"],
            lease_token_hash=hash_lease_token(token),
            call_identity=identity,
            operation_id=operations[index],
            provider=providers[index],
            funding_source="miner_key",
            amount_microusd=1_000_000,
            call_doc={},
        )
        return index, identity, result

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(reserve, range(4)))
    admitted = [(index, identity) for index, identity, result in results if result["status"] == "reserved"]
    busy = [result for _index, _identity, result in results if result["status"] == "budget_busy"]
    assert len(admitted) == 3
    assert len(busy) == 1

    # Exact reserve replay has no second liability.
    index, identity = admitted[0]
    response, token = leased[index]
    replay = setup.reserve_call(
        run_id=response["run_id"],
        lease_token_hash=hash_lease_token(token),
        call_identity=identity,
        operation_id=operations[index],
        provider=providers[index],
        funding_source="miner_key",
        amount_microusd=1_000_000,
        call_doc={},
    )
    assert replay["status"] == "reserved" and replay["idempotent"] is True

    # A genuine charge above its estimate is recorded in full. It is never
    # rejected or clamped, and the over-cap submission cannot reserve more.
    assert setup.mark_dispatched(
        run_id=response["run_id"],
        lease_token_hash=hash_lease_token(token),
        call_identity=identity,
    )["status"] == "dispatched"
    settled = setup.settle_call(
        run_id=response["run_id"],
        lease_token_hash=hash_lease_token(token),
        call_identity=identity,
        actual_microusd=3_500_000,
        terminal_response={"billing": {"cost_usd": 3.5}},
    )
    assert settled["status"] == "settled"
    assert settled["actual_microusd"] == 3_500_000
    assert settled["released_microusd"] == -2_500_000
    settlement = setup.list_ledger(call_identity=identity)[-1]
    assert settlement["amount_microusd"] == 3_500_000

    blocked_index = next(i for i in range(4) if i not in {item[0] for item in admitted})
    blocked_response, blocked_token = leased[blocked_index]
    new_identity = contracts.provider_call_identity(
        attempt=1,
        assignment_id=blocked_response["assignment_id"],
        icp_position=blocked_response["icp_position"],
        action_sequence=1,
        operation_id="scrapingdog.google",
        request_hash=sha("after-actual-overage"),
    )
    blocked = setup.reserve_call(
        run_id=blocked_response["run_id"],
        lease_token_hash=hash_lease_token(blocked_token),
        call_identity=new_identity,
        operation_id="scrapingdog.google",
        provider="scrapingdog",
        funding_source="miner_key",
        amount_microusd=1,
        call_doc={},
    )
    assert blocked["status"] == "refused" and blocked["reason"] == "money_cap"
    summary_connection = connect()
    try:
        with summary_connection.cursor() as cursor:
            cursor.execute("SET ROLE lab_arena_service")
            cursor.execute(
                "SELECT public.lab_arena_submission_costs(%s)",
                (blocked_response["submission_id"],),
            )
            costs = cursor.fetchone()[0]
        assert costs["schema_version"] == "leadpoet.lab_arena.submission_costs.v1"
        assert costs["submission_id"] == blocked_response["submission_id"]
        by_provider = {row["provider"]: row for row in costs["providers"]}
        assert set(by_provider) <= {"openrouter", "deepline", "scrapingdog"}
        assert len(by_provider) >= 2
        assert sum(row["settled_microusd"] for row in costs["providers"]) == 3_500_000
        assert sum(row["reserved_or_uncertain_microusd"] for row in costs["providers"]) == 2_000_000
        assert sum(row["inflight_calls"] for row in costs["providers"]) == 2
        assert sum(row["uncertain_calls"] for row in costs["providers"]) == 0
        assert sum(row["refused_calls"] for row in costs["providers"]) == 1
        assert sum(row["call_count"] for row in costs["providers"]) == 4
    finally:
        summary_connection.close()
    for instance in instances + [setup]:
        instance.close()


def test_combined_provider_latest_states_release_and_retain_budget(connect, superuser):
    round_id = "arena-2026-09-10-combinedstates"
    setup = ArenaStore(PsycopgTransport(connect))
    runners, _parts = open_round(
        setup,
        round_id,
        participants=1,
        runners=1,
        prefix="combined-states",
        quotas={"deepline": 30, "scrapingdog": 30, "openrouter": 30},
        execution_cap_microusd=2_000_000,
    )
    response, token, _, _ = claim(setup, round_id, runners[0], parallelism=2, ceiling=2)
    lease_hash = hash_lease_token(token)
    recovered_identity = contracts.provider_call_identity(
        attempt=1, assignment_id=response["assignment_id"],
        icp_position=response["icp_position"], action_sequence=0,
        operation_id="deepline.execute", request_hash=sha("recover"),
    )
    uncertain_identity = contracts.provider_call_identity(
        attempt=1, assignment_id=response["assignment_id"],
        icp_position=response["icp_position"], action_sequence=1,
        operation_id="scrapingdog.google", request_hash=sha("uncertain"),
    )
    for identity, operation, provider in (
        (recovered_identity, "deepline.execute", "deepline"),
        (uncertain_identity, "scrapingdog.google", "scrapingdog"),
    ):
        assert setup.reserve_call(
            run_id=response["run_id"], lease_token_hash=lease_hash,
            call_identity=identity, operation_id=operation, provider=provider,
            funding_source="miner_key", amount_microusd=1_000_000,
            call_doc={},
        )["status"] == "reserved"
    assert setup.mark_dispatched(
        run_id=response["run_id"], lease_token_hash=lease_hash,
        call_identity=uncertain_identity,
    )["status"] == "dispatched"
    # The existing close helper recovers the undispatched reservation and keeps
    # the dispatched provider charge uncertain at its conservative maximum.
    with superuser.cursor() as cursor:
        cursor.execute(
            "SELECT public.lab_arena__terminate_open_calls(%s, %s)",
            (response["run_id"], "test_close"),
        )
        assert cursor.fetchone()[0] == 2
    assert setup.reserve_call(
        run_id=response["run_id"], lease_token_hash=lease_hash,
        call_identity=recovered_identity, operation_id="deepline.execute",
        provider="deepline", funding_source="miner_key",
        amount_microusd=1_000_000, call_doc={},
    )["status"] == "recovered"
    replacement = contracts.provider_call_identity(
        attempt=1, assignment_id=response["assignment_id"],
        icp_position=response["icp_position"], action_sequence=2,
        operation_id="openrouter.chat", request_hash=sha("replacement"),
    )
    assert setup.reserve_call(
        run_id=response["run_id"], lease_token_hash=lease_hash,
        call_identity=replacement, operation_id="openrouter.chat",
        provider="openrouter", funding_source="miner_key",
        amount_microusd=1_000_000, call_doc={},
    )["status"] == "reserved"
    over = contracts.provider_call_identity(
        attempt=1, assignment_id=response["assignment_id"],
        icp_position=response["icp_position"], action_sequence=3,
        operation_id="deepline.execute", request_hash=sha("over"),
    )
    temporary = setup.reserve_call(
        run_id=response["run_id"], lease_token_hash=lease_hash,
        call_identity=over, operation_id="deepline.execute",
        provider="deepline", funding_source="miner_key",
        amount_microusd=1, call_doc={},
    )
    assert temporary["status"] == "budget_busy"
    assert setup.list_ledger(call_identity=over) == []
    with superuser.cursor() as cursor:
        cursor.execute(
            "SELECT public.lab_arena_submission_costs(%s)",
            (response["submission_id"],),
        )
        providers = cursor.fetchone()[0]["providers"]
    assert sum(row["uncertain_calls"] for row in providers) == 1
    assert sum(row["inflight_calls"] for row in providers) == 1
    assert sum(row["reserved_or_uncertain_microusd"] for row in providers) == 2_000_000
    setup.close()


def test_dynamic_deepline_reserves_remaining_balance_without_parallel_dispatch(connect):
    round_id = "arena-2026-09-10-dynamicdd"
    setup = ArenaStore(PsycopgTransport(connect))
    runners, _parts = open_round(
        setup,
        round_id,
        participants=1,
        runners=1,
        prefix="dynamic-dd",
        quotas={"deepline": 30, "scrapingdog": 30, "openrouter": 30},
        execution_cap_microusd=50_000_000,
    )
    leases = [
        claim(setup, round_id, runners[0], parallelism=4, ceiling=4)[:2]
        for _ in range(4)
    ]

    priced_run, priced_token = leases[0]
    priced_identity = contracts.provider_call_identity(
        attempt=1, assignment_id=priced_run["assignment_id"],
        icp_position=priced_run["icp_position"], action_sequence=0,
        operation_id="openrouter.chat", request_hash=sha("priced-ten"),
    )
    assert setup.reserve_call(
        run_id=priced_run["run_id"], lease_token_hash=hash_lease_token(priced_token),
        call_identity=priced_identity, operation_id="openrouter.chat",
        provider="openrouter", funding_source="miner_key",
        amount_microusd=10_000_000, call_doc={},
    )["status"] == "reserved"
    assert setup.mark_dispatched(
        run_id=priced_run["run_id"], lease_token_hash=hash_lease_token(priced_token),
        call_identity=priced_identity,
    )["status"] == "dispatched"
    assert setup.settle_call(
        run_id=priced_run["run_id"], lease_token_hash=hash_lease_token(priced_token),
        call_identity=priced_identity, actual_microusd=4_000_000,
        terminal_response={},
    )["status"] == "settled"

    dynamic_run, dynamic_token = leases[1]
    dynamic_identity = contracts.provider_call_identity(
        attempt=1, assignment_id=dynamic_run["assignment_id"],
        icp_position=dynamic_run["icp_position"], action_sequence=0,
        operation_id="deepline.execute", request_hash=sha("dynamic-first"),
    )
    dynamic_doc = {"reserve_remaining_budget": True}
    for provider, amount, document in (
        ("openrouter", 0, dynamic_doc),
        ("deepline", 1, dynamic_doc),
        ("deepline", 0, {"reserve_remaining_budget": "true"}),
        ("deepline", 0, {"reserve_remaining_budget": False}),
    ):
        with pytest.raises(ArenaStoreError):
            setup.reserve_call(
                run_id=dynamic_run["run_id"],
                lease_token_hash=hash_lease_token(dynamic_token),
                call_identity=sha("invalid-dynamic-%s-%s-%s" % (provider, amount, document)),
                operation_id=("openrouter.chat" if provider == "openrouter" else "deepline.execute"),
                provider=provider, funding_source="miner_key",
                amount_microusd=amount, call_doc=document,
            )
    reserved = setup.reserve_call(
        run_id=dynamic_run["run_id"], lease_token_hash=hash_lease_token(dynamic_token),
        call_identity=dynamic_identity, operation_id="deepline.execute",
        provider="deepline", funding_source="miner_key",
        amount_microusd=0, call_doc=dynamic_doc,
    )
    assert reserved["status"] == "reserved"
    assert reserved["amount_microusd"] == 46_000_000
    replay = setup.reserve_call(
        run_id=dynamic_run["run_id"], lease_token_hash=hash_lease_token(dynamic_token),
        call_identity=dynamic_identity, operation_id="deepline.execute",
        provider="deepline", funding_source="miner_key",
        amount_microusd=0, call_doc=dynamic_doc,
    )
    assert replay["status"] == "reserved" and replay["idempotent"] is True

    waiting_run, waiting_token = leases[2]
    waiting_identity = contracts.provider_call_identity(
        attempt=1, assignment_id=waiting_run["assignment_id"],
        icp_position=waiting_run["icp_position"], action_sequence=0,
        operation_id="deepline.execute", request_hash=sha("dynamic-waiter"),
    )
    busy = setup.reserve_call(
        run_id=waiting_run["run_id"], lease_token_hash=hash_lease_token(waiting_token),
        call_identity=waiting_identity, operation_id="deepline.execute",
        provider="deepline", funding_source="miner_key",
        amount_microusd=0, call_doc=dynamic_doc,
    )
    assert busy["status"] == "budget_busy"
    assert setup.list_ledger(call_identity=waiting_identity) == []

    # A priced request that fits after the in-flight dynamic reservation settles
    # is temporary contention, not a permanent money-cap refusal.
    known_identity = contracts.provider_call_identity(
        attempt=1, assignment_id=waiting_run["assignment_id"],
        icp_position=waiting_run["icp_position"], action_sequence=1,
        operation_id="scrapingdog.google", request_hash=sha("known-waiter"),
    )
    known_busy = setup.reserve_call(
        run_id=waiting_run["run_id"], lease_token_hash=hash_lease_token(waiting_token),
        call_identity=known_identity, operation_id="scrapingdog.google",
        provider="scrapingdog", funding_source="miner_key",
        amount_microusd=1_000_000, call_doc={},
    )
    assert known_busy["status"] == "budget_busy"
    assert setup.list_ledger(call_identity=known_identity) == []

    assert setup.mark_dispatched(
        run_id=dynamic_run["run_id"], lease_token_hash=hash_lease_token(dynamic_token),
        call_identity=dynamic_identity,
    )["status"] == "dispatched"
    assert setup.settle_call(
        run_id=dynamic_run["run_id"], lease_token_hash=hash_lease_token(dynamic_token),
        call_identity=dynamic_identity, actual_microusd=6_000_000,
        terminal_response={},
    )["status"] == "settled"
    released = setup.reserve_call(
        run_id=waiting_run["run_id"], lease_token_hash=hash_lease_token(waiting_token),
        call_identity=waiting_identity, operation_id="deepline.execute",
        provider="deepline", funding_source="miner_key",
        amount_microusd=0, call_doc=dynamic_doc,
    )
    assert released["status"] == "reserved"
    assert released["amount_microusd"] == 40_000_000

    # Missing billing keeps the full dynamic balance uncertain. At exactly the
    # cap, another zero-input dynamic reservation is a recorded refusal.
    assert setup.mark_dispatched(
        run_id=waiting_run["run_id"], lease_token_hash=hash_lease_token(waiting_token),
        call_identity=waiting_identity,
    )["status"] == "dispatched"
    assert setup.mark_uncertain(
        run_id=waiting_run["run_id"], lease_token_hash=hash_lease_token(waiting_token),
        call_identity=waiting_identity, call_doc={"reason": "billing_missing"},
    )["status"] == "uncertain"
    final_run, final_token = leases[3]
    final_identity = contracts.provider_call_identity(
        attempt=1, assignment_id=final_run["assignment_id"],
        icp_position=final_run["icp_position"], action_sequence=0,
        operation_id="deepline.execute", request_hash=sha("dynamic-at-cap"),
    )
    refused = setup.reserve_call(
        run_id=final_run["run_id"], lease_token_hash=hash_lease_token(final_token),
        call_identity=final_identity, operation_id="deepline.execute",
        provider="deepline", funding_source="miner_key",
        amount_microusd=0, call_doc=dynamic_doc,
    )
    assert refused["status"] == "refused" and refused["reason"] == "money_cap"
    assert setup.reserve_call(
        run_id=final_run["run_id"], lease_token_hash=hash_lease_token(final_token),
        call_identity=final_identity, operation_id="deepline.execute",
        provider="deepline", funding_source="miner_key",
        amount_microusd=0, call_doc=dynamic_doc,
    )["idempotent"] is True
    setup.close()


# ---------------------------------------------------------------------------
# Validator scoring: assignments claimed by any validator, judge causes, and
# the scoring window's close and cancel behavior.
# ---------------------------------------------------------------------------


def _complete_accepted(store, run_id, lease_hash, label):
    return complete(
        store,
        run_id,
        lease_hash,
        "accepted",
        output_ref="arena/x/outputs/%s.json" % run_id,
    )


def _execute_everything(store, round_id, runner, *, parallelism=100):
    """One runner executes every pending assignment of the current stage; returns run ids by (submission, icp)."""

    executed = {}
    while True:
        response, token, _, _ = claim(store, round_id, runner, parallelism=parallelism, ceiling=parallelism)
        if response["status"] != "leased":
            break
        assert response["kind"] == "execute"
        assert _complete_accepted(store, response["run_id"], hash_lease_token(token), response["run_id"])["status"] == "accepted"
        executed[(response["submission_id"], response["icp_position"])] = response["run_id"]
    return executed


def _commit_plan(store, round_id, stage):
    plan = {"round_id": round_id, "stage": stage, "work_items": []}
    result = store.transition_round(
        round_id,
        "stage%d_closed" % stage,
        "stage%d_closed" % stage,
        {"stage%d_scoring_plan_doc" % stage: plan},
    )
    assert result["status"] == "ok", result


def _scoring_items(executed):
    items = []
    for (submission_id, position), run_id in sorted(executed.items()):
        items.append(
            {
                "scored_run_id": run_id,
                "submission_id": submission_id,
                "icp_position": position,
                "output_ref": "arena/x/outputs/%s.json" % run_id,
            }
        )
    return items


def test_scoring_assignments_are_claimed_by_any_validator_and_close_to_judged(store, superuser):
    round_id = "arena-2026-09-02-sc"
    runners, parts = open_round(
        store,
        round_id,
        participants=2,
        runners=2,
        prefix="sc",
        scoring_cap_microusd=1_000,
    )
    executor, scorer = runners
    executed = _execute_everything(store, round_id, executor)
    assert len(executed) == 20 and store.close_stage(round_id, 1)["status"] == "closed"
    _commit_plan(store, round_id, 1)
    items = _scoring_items(executed)
    opened = store.open_scoring(round_id, 1, items)
    assert opened["status"] == "ok" and opened["round_status"] == "stage1_scoring" and opened["assignments"] == 20
    assert store.open_scoring(round_id, 1, items)["status"] == "existing"
    # Any validator scores any item, including the executor of that output.
    own, own_token, _, _ = claim(store, round_id, executor, parallelism=100, ceiling=100)
    assert own["status"] == "leased" and own["kind"] == "score" and own["scored_run_id"] in executed.values()
    assert _complete_accepted(store, own["run_id"], hash_lease_token(own_token), own["run_id"])["status"] == "accepted"
    response, token, _, _ = claim(store, round_id, scorer, parallelism=100, ceiling=100)
    assert response["status"] == "leased" and response["kind"] == "score" and response["scored_run_id"] in executed.values()
    lease_hash = hash_lease_token(token)
    # An execute-only cause on a score run is refused; a judge cause is accepted.
    with pytest.raises(ArenaStoreError, match="lab_arena_complete_cause_kind_mismatch"):
        complete(store, response["run_id"], lease_hash, "model_error")
    failed = complete(store, response["run_id"], lease_hash, "judge_error")
    assert failed["status"] == "failed" and failed["confirmation_attempt"] == 2
    confirmation, confirmation_token, _, _ = claim(store, round_id, executor, parallelism=100, ceiling=100)
    assert confirmation["assignment_id"] == response["assignment_id"] and confirmation["attempt"] == 2
    assert _complete_accepted(
        store,
        confirmation["run_id"],
        hash_lease_token(confirmation_token),
        confirmation["run_id"],
    )["status"] == "accepted"
    # Score-run provider calls draw on the host scoring quota for that submission.
    response2, token2, _, _ = claim(store, round_id, scorer, parallelism=100, ceiling=100)
    lease2 = hash_lease_token(token2)
    identity = contracts.provider_call_identity(attempt=1, assignment_id=response2["assignment_id"], icp_position=response2["icp_position"], action_sequence=0, operation_id="openrouter.chat", request_hash=sha("judge"))
    reserved = store.reserve_call(run_id=response2["run_id"], lease_token_hash=lease2, call_identity=identity, operation_id="openrouter.chat", provider="openrouter", funding_source="miner_key", amount_microusd=1000, call_doc={})
    assert reserved["status"] == "reserved"
    over_identity = contracts.provider_call_identity(attempt=1, assignment_id=response2["assignment_id"], icp_position=response2["icp_position"], action_sequence=1, operation_id="openrouter.chat", request_hash=sha("judge-over"))
    over = store.reserve_call(run_id=response2["run_id"], lease_token_hash=lease2, call_identity=over_identity, operation_id="openrouter.chat", provider="openrouter", funding_source="miner_key", amount_microusd=1, call_doc={})
    assert over["status"] == "budget_busy"
    assert store.list_ledger(call_identity=over_identity) == []
    assert store.mark_dispatched(run_id=response2["run_id"], lease_token_hash=lease2, call_identity=identity)["status"] == "dispatched"
    assert store.settle_call(run_id=response2["run_id"], lease_token_hash=lease2, call_identity=identity, actual_microusd=400, terminal_response={"status": 200})["status"] == "settled"
    assert _complete_accepted(store, response2["run_id"], lease2, response2["run_id"])["status"] == "accepted"
    # Everything else is scored; the window closes to judged, not cancelled.
    while True:
        r, t, _, _ = claim(store, round_id, scorer, parallelism=100, ceiling=100)
        if r["status"] != "leased":
            break
        assert _complete_accepted(store, r["run_id"], hash_lease_token(t), r["run_id"])["status"] == "accepted"
    closed = store.close_scoring(round_id, 1)
    assert closed["status"] == "closed" and closed["round_status"] == "stage1_judged" and closed["incomplete_assignments"] == 0
    assert store.close_scoring(round_id, 1)["status"] == "existing"
    score_runs = store.list_runs(round_id, stage=1, kind="score")
    assert len(score_runs) == 21 and {run["runner_hotkey"] for run in score_runs} == {executor, scorer}
    # A judged round is terminal for its scorings: an accepted scoring is never rewritten.
    with superuser.cursor() as cursor:
        with pytest.raises(Exception, match="immutable"):
            cursor.execute("UPDATE public.lab_arena_runs SET status = 'failed', terminal_cause = 'judge_error' WHERE run_id = %s", (score_runs[0]["run_id"],))
    superuser.rollback()


def test_scoring_window_with_an_unjudged_item_cancels_and_expiry_retries_score_runs(store, superuser):
    round_id = "arena-2026-09-02-sx"
    runners, parts = open_round(store, round_id, participants=1, runners=2, prefix="sx")
    executor, scorer = runners
    executed = _execute_everything(store, round_id, executor)
    assert store.close_stage(round_id, 1)["status"] == "closed"
    _commit_plan(store, round_id, 1)
    items = _scoring_items(executed)
    assert store.open_scoring(round_id, 1, items)["assignments"] == 10
    response, token, _, _ = claim(store, round_id, scorer, parallelism=1, ceiling=1)
    assert response["kind"] == "score"
    # An expired scoring lease retries once and keeps its scoring identity.
    expire_now(superuser, response["run_id"])
    assert store.expire_leases(round_id) == {"status": "ok", "expired": 1, "retried": 1}
    retry = [run for run in store.list_runs(round_id, stage=1, kind="score") if run["assignment_id"] == response["assignment_id"] and run["attempt"] == 2][0]
    assert retry["status"] == "pending" and retry["scored_run_id"] == response["scored_run_id"]
    # Closing with pending scoring work is an infrastructure gap: the round cancels, no miner gets a zero.
    closed = store.close_scoring(round_id, 1)
    assert closed["status"] == "cancelled" and closed["incomplete_assignments"] == 10
    assert store.get_round(round_id)["cancel_reason"] == "scoring_incomplete:stage1:10"


def test_service_role_statements_locks_and_idle_transactions_are_bounded(superuser, store):
    """The service role carries per-request limits so an Arena burst cannot hold a shared instance.

    The heaviest single statement, cancelling a round at the challenger cap with
    every stage-one run open, must finish well inside the statement bound.
    """

    import time

    with superuser.cursor() as cursor:
        cursor.execute("SELECT rolconfig FROM pg_roles WHERE rolname = 'lab_arena_service'")
        settings = set(cursor.fetchone()[0] or [])
    assert {"statement_timeout=30s", "lock_timeout=5s", "idle_in_transaction_session_timeout=60s"} <= settings, settings
    round_id = "arena-2026-11-30"
    open_round(store, round_id, participants=contracts.MAX_CHALLENGERS, prefix="cap")
    assert len(store.list_runs(round_id, stage=1, status="pending")) == contracts.STAGE_1_ICP_COUNT * contracts.MAX_CHALLENGERS
    started = time.monotonic()
    cancelled = store.cancel_round(round_id, "capacity:test")
    seconds = time.monotonic() - started
    assert cancelled["status"] == "cancelled", cancelled
    assert seconds < 10.0, "cancel at the challenger cap took %.1fs" % seconds
    assert all(run["status"] == "failed" and run["terminal_cause"] == "stage_closed" for run in store.list_runs(round_id, stage=1))


def test_a_confirmation_attempt_goes_to_another_validator_and_an_unreached_one_leaves_the_zero(store):
    """With two validators the failing one cannot confirm its own verdict; a window that closes first keeps the first failure."""

    round_id = "arena-2026-09-02-cf2"
    runners, parts = open_round(store, round_id, participants=1, runners=2, prefix="cf2")
    # Both validators are active: the second one holds position 0 while the first fails position 1.
    held, held_token, _, _ = claim(store, round_id, runners[1], parallelism=8)
    assert held["icp_position"] == 0
    first, token, _, _ = claim(store, round_id, runners[0], parallelism=8)
    assert first["icp_position"] == 1
    failed = complete(store, first["run_id"], hash_lease_token(token), "model_error")
    assert failed["confirmation_attempt"] == 2
    # The failing validator skips the confirmation and takes the next position; the other validator gets it.
    same, same_token, _, _ = claim(store, round_id, runners[0], parallelism=8)
    assert same["icp_position"] == 2 and same["attempt"] == 1
    other, *_ = claim(store, round_id, runners[1], parallelism=8)
    assert other["icp_position"] == 1 and other["attempt"] == 2
    # Every other position completes normally, one lease at a time (a runner holds at most eight);
    # the confirmation attempt stays leased, unreached.
    def accept(response, token):
        result = complete(store, response["run_id"], hash_lease_token(token), "accepted", output_ref="arena/x/outputs/" + response["run_id"])
        assert result["status"] == "accepted", result

    accept(held, held_token)
    accept(same, same_token)
    accepted = 2
    while True:
        response, token, _, _ = claim(store, round_id, runners[0], parallelism=8)
        if response["status"] != "leased":
            break
        accept(response, token)
        accepted += 1
    assert accepted == 9
    # Closing the stage with the confirmation still open leaves the first failure as the miner's zero.
    closed = store.close_stage(round_id, 1)
    assert closed["status"] == "closed" and closed["incomplete_assignments"] == 0, closed
    runs = [r for r in store.list_runs(round_id, stage=1) if r["assignment_id"] == first["assignment_id"]]
    assert {(r["attempt"], r["terminal_cause"]) for r in runs} == {(1, "model_error"), (2, "stage_closed")}


def test_a_historical_inactive_validator_does_not_block_self_confirmation(store):
    round_id = "arena-2026-09-02-confirmactive"
    runners, _parts = open_round(store, round_id, participants=1, runners=2, prefix="confirm-active")
    historical, historical_token, _, _ = claim(store, round_id, runners[1], parallelism=8)
    assert complete(
        store,
        historical["run_id"],
        hash_lease_token(historical_token),
        "accepted",
        output_ref="arena/x/outputs/" + historical["run_id"],
    )["status"] == "accepted"
    first, first_token, _, _ = claim(store, round_id, runners[0], parallelism=8)
    failed = complete(store, first["run_id"], hash_lease_token(first_token), "model_error")
    assert failed["confirmation_attempt"] == 2
    confirmation, *_ = claim(store, round_id, runners[0], parallelism=8)
    assert confirmation["assignment_id"] == first["assignment_id"]
    assert confirmation["attempt"] == 2
