"""Exact champion source and capture lineage are frozen together."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pytest

from lab_arena import service as arena_service
from lab_arena.promotion import GitPromoter
from lab_arena.store import ArenaStore, PsycopgTransport
from tests.lab_arena import full_thinking_champion_baseline567_postgres_test as prior
from tests.lab_arena.full_thinking_round_test import opted_in
from tests.lab_arena.test_lab_arena_promotion_git import git, repository, source
from tests.lab_arena.test_lab_arena_reward_migration_postgres import BASELINE, MINER_A

MIGRATION = Path(__file__).parents[2] / "scripts/585-lab-arena-champion-baseline-reasoning.sql"


@pytest.fixture(scope="module")
def database():
    fixture = prior.database.__wrapped__()
    db = next(fixture)
    try:
        with db[0].connect(**db[1]) as connection, connection.cursor() as cursor:
            cursor.execute(MIGRATION.read_text())
            cursor.execute(MIGRATION.read_text())
        yield db
    finally:
        fixture.close()


def test_freeze_rpc_retains_service_only_acl(database):
    prior.test_baseline_freeze_rpc_is_service_only(database)


def test_no_champion_keeps_bootstrap_source_and_current_capture_contract(database, tmp_path):
    psycopg, dsn = database
    store = ArenaStore(PsycopgTransport(lambda: psycopg.connect(**dsn)))
    try:
        bootstrap = source("bootstrap-only")
        service = opted_in()
        service._config.defaults.baseline_hotkey = BASELINE
        service._config.defaults.reasoning_baseline_source_sha256 = "sha256:" + hashlib.sha256(bootstrap).hexdigest()
        service._store = store
        service._objects = arena_service.LocalObjectStore(tmp_path / "bootstrap-objects")
        service._config.baseline_source_fetcher = lambda *_args: bootstrap
        config = service.create_round(datetime(2026, 10, 20, tzinfo=timezone.utc))
        service.freeze_participants(config["round_id"])
        frozen = store.get_round(config["round_id"])
        assert frozen["configuration_doc"]["baseline_source_url"] == config["baseline_source_url"]
        contract = frozen["configuration_doc"]["baseline_reasoning_configuration"]
        assert contract["reasoning_capture_policy"] == config["reasoning_capture_policy"]
        assert contract["reasoning_route_registry"] == config["reasoning_route_registry"]
    finally:
        store.close()


def test_full_thinking_promotion_still_freezes_exact_source(database, tmp_path):
    prior.test_created_before_promotion_freezes_promoted_source_and_matching_payer(database, tmp_path)
    with database[0].connect(**database[1]) as connection, connection.cursor() as cursor:
        cursor.execute("SELECT configuration_doc FROM public.lab_arena_rounds WHERE round_id='arena-2026-10-24'")
        configuration = cursor.fetchone()[0]
        original = configuration["baseline_reasoning_configuration"]["reasoning_route_registry"]
        assert original["approved_routes"][0]["evidence"]["probe_id"] == "original-champion-probe"
        assert configuration["reasoning_route_registry"]["approved_routes"][0]["evidence"]["probe_id"] != "original-champion-probe"


def test_migration_preserves_rows_schema_acl_and_rejects_unknown_guard(database):
    with database[0].connect(**database[1]) as connection:
        connection.autocommit = True
        with connection.cursor() as cursor:
            cursor.execute("SELECT jsonb_agg(to_jsonb(r) ORDER BY round_id) FROM public.lab_arena_rounds r")
            rows = cursor.fetchone()[0]
            cursor.execute("SELECT nspacl::text FROM pg_namespace WHERE nspname='public'")
            acl = cursor.fetchone()[0]
            cursor.execute(MIGRATION.read_text())
            cursor.execute("SELECT jsonb_agg(to_jsonb(r) ORDER BY round_id) FROM public.lab_arena_rounds r")
            assert cursor.fetchone()[0] == rows
            cursor.execute("SELECT nspacl::text FROM pg_namespace WHERE nspname='public'")
            assert cursor.fetchone()[0] == acl
            cursor.execute("SELECT pg_get_functiondef('public.lab_arena_rounds_write_once_v1()'::regprocedure)")
            original = cursor.fetchone()[0]
            try:
                cursor.execute(original.replace("BEGIN\n", "BEGIN\n  NULL;\n", 1))
                with pytest.raises(database[0].Error, match="guard preimage differs"):
                    cursor.execute(MIGRATION.read_text())
                cursor.execute("ROLLBACK")
            finally:
                cursor.execute(original)


def test_legacy_champion_source_and_capture_contract_are_frozen(
    database, tmp_path
):
    psycopg, dsn = database
    store = ArenaStore(PsycopgTransport(lambda: psycopg.connect(**dsn)))
    control = psycopg.connect(**dsn)
    control.autocommit = True
    try:
        remote, _, bootstrap_commit = repository(tmp_path)
        bootstrap, legacy = source("bootstrap"), source("legacy")
        bootstrap_url = (
            "https://github.com/leadpoet/leadpoet-sales-agent/archive/"
            + bootstrap_commit + ".tar.gz"
        )
        legacy_round_id = "arena-2026-10-21-legacy"
        legacy_winner_id = legacy_round_id + "-miner"
        promoter = GitPromoter(str(remote), tmp_path / "promotion-cache")
        plan = promoter.prepare(legacy, round_id=legacy_round_id,
                                submission_id=legacy_winner_id,
                                timestamp="2026-10-21T01:00:00Z")
        assert promoter.publish(legacy, plan=plan, round_id=legacy_round_id,
                                submission_id=legacy_winner_id) == plan["commit"]
        legacy_url = (
            "https://github.com/leadpoet/leadpoet-sales-agent/archive/"
            + plan["commit"] + ".tar.gz"
        )
        publication = {"king_decision": {"outcome": "crowned",
                                         "winner_submission_id": legacy_winner_id,
                                         "king_submission_id": legacy_winner_id,
                                         "king_hotkey": MINER_A}}
        with control.cursor() as cursor:
            cursor.execute("SET session_replication_role = replica")
            cursor.execute(
                "INSERT INTO public.lab_arena_rounds "
                "(round_id,status,configuration_doc,rewards_enabled,promotion_required,"
                "publication_doc,promotion_doc,king_outcome,published_at,"
                "baseline_promoted_at,evaluation_date) "
                "VALUES (%s,'published',%s::jsonb,TRUE,TRUE,%s::jsonb,%s::jsonb,"
                "'crowned','2026-10-21T01:00:00Z',clock_timestamp(),'2026-10-21')",
                (legacy_round_id, json.dumps({"mode": "live", "network_name": "finney",
                                             "netuid": 71, "rewards_enabled": True}),
                 json.dumps(publication), json.dumps(plan)),
            )
            cursor.execute(
                "INSERT INTO public.lab_arena_submissions "
                "(submission_id,round_id,miner_hotkey,status,is_king) "
                "VALUES (%s,%s,%s,'frozen',FALSE)",
                (legacy_winner_id, legacy_round_id, MINER_A),
            )
            cursor.execute("SET session_replication_role = origin")

        service = opted_in()
        service._config.defaults.reasoning_baseline_source_url = bootstrap_url
        service._config.defaults.reasoning_baseline_source_sha256 = (
            "sha256:" + hashlib.sha256(bootstrap).hexdigest()
        )
        service._config.defaults.baseline_hotkey = BASELINE
        service._store = store
        service._objects = arena_service.LocalObjectStore(tmp_path / "objects")
        fetched = []
        service._config.baseline_source_fetcher = lambda url, _limit: (
            fetched.append(url) or (legacy if url == legacy_url else bootstrap)
        )
        first_round = service.create_round(datetime(2026, 10, 22, tzinfo=timezone.utc))
        selected = store.latest_promoted_champion(network_name="finney", netuid=71)
        assert selected["reasoning_capture_policy"] is None
        # N-1 runtime selects the configured bootstrap for a legacy winner.
        # New SQL refuses that selection before freezing any saved state.
        with pytest.raises(Exception, match="champion_baseline_source_invalid"):
            store.freeze_full_thinking_baseline(
                first_round["round_id"], expected_promotion_round_id=legacy_round_id,
                expected_winner_submission_id=legacy_winner_id, expected_commit=plan["commit"],
                expected_source_url=bootstrap_url,
                expected_source_sha256=first_round["reasoning_baseline_source_sha256"],
                source_url=bootstrap_url,
                source_sha256=first_round["reasoning_baseline_source_sha256"],
            )
        assert store.get_round(first_round["round_id"])["champion_funding_frozen"] is False
        participants = service.freeze_participants(first_round["round_id"])
        stored = store.get_round(first_round["round_id"])
        assert len(participants) == 1 and participants[0]["is_king"]
        assert stored["configuration_doc"]["baseline_source_url"] == legacy_url
        assert stored["configuration_doc"]["reasoning_baseline_source_sha256"] == (
            "sha256:" + hashlib.sha256(legacy).hexdigest()
        )
        assert stored["configuration_doc"]["baseline_reasoning_configuration"] == {}
        assert stored["champion_submission_id"] == legacy_winner_id
        assert stored["champion_hotkey"] == MINER_A
        assert service._objects.get(participants[0]["source_ref"]) == legacy
        assert fetched == [legacy_url, legacy_url]
        assert service.freeze_participants(first_round["round_id"]) == participants
        assert fetched == [legacy_url, legacy_url]
        with control.cursor() as cursor:
            with pytest.raises(psycopg.Error, match="configuration"):
                cursor.execute(
                    "UPDATE public.lab_arena_rounds SET configuration_doc = "
                    "jsonb_set(configuration_doc,'{baseline_reasoning_configuration}',%s::jsonb) WHERE round_id=%s",
                    (json.dumps({"reasoning_capture_policy": "full_generated_thinking_v1"}), first_round["round_id"]),
                )
    finally:
        control.close()
        store.close()
