"""A next round adopts a promoted source when its baseline is frozen."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from lab_arena import service as arena_service
from lab_arena.promotion import GitPromoter
from lab_arena.store import ArenaStore, PsycopgTransport
from tests.lab_arena import model_policy_supersession517_postgres_test as prior
from tests.lab_arena.full_thinking_round_test import opted_in
from tests.lab_arena.test_lab_arena_promotion_git import git, repository, source
from tests.lab_arena.test_lab_arena_reward_migration_postgres import BASELINE, MINER_A


MIGRATION = Path(__file__).parents[2] / "scripts/566-lab-arena-full-thinking-champion-baseline.sql"
SCRIPTS = MIGRATION.parent


@pytest.fixture(scope="module")
def database():
    fixture = prior.database.__wrapped__()
    db = next(fixture)
    try:
        with db[0].connect(**db[1]) as connection, connection.cursor() as cursor:
            for name in (
                "539-lab-arena-held-recovery-provider-admission.sql",
                "564-lab-arena-full-thinking-six-dollar-cap.sql",
            ):
                cursor.execute((SCRIPTS / name).read_text())
            cursor.execute(MIGRATION.read_text())
            cursor.execute(MIGRATION.read_text())
        yield db
    finally:
        fixture.close()


def test_baseline_freeze_rpc_is_service_only(database):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as connection, connection.cursor() as cursor:
        for name, arguments in (
            ("lab_arena_latest_promoted_champion_v1", "text,integer"),
            ("lab_arena_freeze_full_thinking_baseline_v1",
             "text,text,text,text,text,text,text,text"),
        ):
            signature = "public.%s(%s)" % (name, arguments)
            cursor.execute(
                "SELECT has_function_privilege('lab_arena_service',%s,'EXECUTE'),"
                "has_function_privilege('anon',%s,'EXECUTE'),"
                "has_function_privilege('authenticated',%s,'EXECUTE'),"
                "has_function_privilege('service_role',%s,'EXECUTE')",
                (signature, signature, signature, signature),
            )
            assert cursor.fetchone() == (True, False, False, False)


def test_first_full_thinking_round_keeps_tested_bootstrap_after_legacy_promotion(
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
        stale = store.freeze_full_thinking_baseline(
            first_round["round_id"],
            expected_promotion_round_id=legacy_round_id,
            expected_winner_submission_id=legacy_winner_id,
            expected_commit=plan["commit"],
            expected_source_url=bootstrap_url,
            expected_source_sha256=first_round["reasoning_baseline_source_sha256"],
            source_url=legacy_url,
            source_sha256="sha256:" + hashlib.sha256(legacy).hexdigest(),
        )
        assert stale["status"] == "stale"
        assert store.get_round(first_round["round_id"])["champion_funding_frozen"] is False
        participants = service.freeze_participants(first_round["round_id"])
        stored = store.get_round(first_round["round_id"])
        assert len(participants) == 1 and participants[0]["is_king"]
        assert stored["configuration_doc"]["baseline_source_url"] == bootstrap_url
        assert stored["configuration_doc"]["reasoning_baseline_source_sha256"] == (
            "sha256:" + hashlib.sha256(bootstrap).hexdigest()
        )
        assert stored["champion_submission_id"] == legacy_winner_id
        assert stored["champion_hotkey"] == MINER_A
        assert service._objects.get(participants[0]["source_ref"]) == bootstrap
        assert fetched == [bootstrap_url]
    finally:
        control.close()
        store.close()


def test_created_before_promotion_freezes_promoted_source_and_matching_payer(
    database, tmp_path
):
    psycopg, dsn = database
    store = ArenaStore(PsycopgTransport(lambda: psycopg.connect(**dsn)))
    control = psycopg.connect(**dsn)
    control.autocommit = True
    try:
        remote, _, old_commit = repository(tmp_path)
        objects = arena_service.LocalObjectStore(tmp_path / "objects")
        original, champion = source("original"), source("champion")
        old_url = (
            "https://github.com/leadpoet/leadpoet-sales-agent/archive/"
            + old_commit + ".tar.gz"
        )
        next_service = opted_in()
        next_service._config.defaults.reasoning_baseline_source_url = old_url
        next_service._config.defaults.reasoning_baseline_source_sha256 = (
            "sha256:" + hashlib.sha256(original).hexdigest()
        )
        next_service._config.defaults.baseline_hotkey = BASELINE
        next_service._store = store
        next_service._objects = objects
        next_round = next_service.create_round(datetime(2026, 10, 24, tzinfo=timezone.utc))
        assert next_round["baseline_source_url"] == old_url

        prior_round_id = "arena-2026-10-23-fullsource"
        winner_id = prior_round_id + "-miner"
        prior_configuration = {"mode": "live", "network_name": "finney", "netuid": 71,
                               "rewards_enabled": True,
                               "reasoning_capture_policy": "full_generated_thinking_v1"}
        publication = {"king_decision": {"outcome": "crowned",
                                         "winner_submission_id": winner_id,
                                         "king_submission_id": winner_id,
                                         "king_hotkey": MINER_A}}
        # This published result stands in for completed scoring. Git promotion
        # runs below; its DB marker is seeded because no ICPs ran here.
        with control.cursor() as cursor:
            cursor.execute("SET session_replication_role = replica")
            cursor.execute(
                "INSERT INTO public.lab_arena_rounds "
                "(round_id,status,configuration_doc,rewards_enabled,promotion_required,"
                "publication_doc,king_outcome,published_at,evaluation_date) "
                "VALUES (%s,'published',%s::jsonb,TRUE,TRUE,%s::jsonb,'crowned',"
                "'2026-10-23T01:00:00Z','2026-10-23')",
                (prior_round_id, json.dumps(prior_configuration), json.dumps(publication)),
            )
            cursor.execute(
                "INSERT INTO public.lab_arena_submissions "
                "(submission_id,round_id,miner_hotkey,status,is_king) "
                "VALUES (%s,%s,%s,'frozen',FALSE)",
                (winner_id, prior_round_id, MINER_A),
            )
            cursor.execute("SET session_replication_role = origin")
        with pytest.raises(arena_service.ServiceError, match="baseline_promotion_pending"):
            next_service.freeze_participants(next_round["round_id"])
        assert store.get_round(next_round["round_id"])["champion_funding_frozen"] is False

        promoter = GitPromoter(str(remote), tmp_path / "promotion-cache")
        plan = promoter.prepare(champion, round_id=prior_round_id,
                                submission_id=winner_id,
                                timestamp="2026-10-23T01:00:00Z")
        assert promoter.publish(champion, plan=plan, round_id=prior_round_id,
                                submission_id=winner_id) == plan["commit"]
        with control.cursor() as cursor:
            cursor.execute("SET session_replication_role = replica")
            cursor.execute(
                "UPDATE public.lab_arena_rounds SET promotion_doc=%s::jsonb, "
                "baseline_promoted_at=clock_timestamp() WHERE round_id=%s",
                (json.dumps(plan), prior_round_id),
            )
            cursor.execute("SET session_replication_role = origin")
        promoted_commit = git(remote, "rev-parse", "lab")
        expected_url = (
            "https://github.com/leadpoet/leadpoet-sales-agent/archive/"
            + promoted_commit + ".tar.gz"
        )
        stale = store.freeze_full_thinking_baseline(
            next_round["round_id"],
            expected_promotion_round_id=prior_round_id,
            expected_winner_submission_id=winner_id,
            expected_commit="0" * 40,
            expected_source_url=old_url,
            expected_source_sha256=next_round["reasoning_baseline_source_sha256"],
            source_url=expected_url,
            source_sha256="sha256:" + hashlib.sha256(champion).hexdigest(),
        )
        assert stale["status"] == "stale"
        assert store.get_round(next_round["round_id"])["champion_funding_frozen"] is False
        challenger_id = "challenger-2026-10-24"
        with control.cursor() as cursor:
            cursor.execute("SET session_replication_role = replica")
            cursor.execute(
                "INSERT INTO public.lab_arena_submissions "
                "(submission_id,round_id,miner_hotkey,status,is_king) "
                "VALUES (%s,%s,%s,'accepted',FALSE)",
                (challenger_id, next_round["round_id"], MINER_A),
            )
            cursor.execute("SET session_replication_role = origin")
        requested = []
        next_service._config.baseline_source_fetcher = lambda url, _limit: (
            requested.append(url) or (champion if url == expected_url else original)
        )
        assert next_service._freeze_full_thinking_baseline(
            next_round["round_id"], store.get_round(next_round["round_id"])
        )["status"] == "frozen"
        assert store.get_submission(challenger_id)["status"] == "accepted"
        # The test challenger has no review evidence; remove only this fixture
        # before the normal participant admission path runs.
        with control.cursor() as cursor:
            cursor.execute("SET session_replication_role = replica")
            cursor.execute("DELETE FROM public.lab_arena_submissions WHERE submission_id=%s",
                           (challenger_id,))
            cursor.execute("SET session_replication_role = origin")
        participants = next_service.freeze_participants(next_round["round_id"])
        row = store.get_round(next_round["round_id"])
        assert len(participants) == 1 and participants[0]["is_king"]
        assert row["configuration_doc"]["baseline_source_url"] == expected_url
        assert row["configuration_doc"]["reasoning_baseline_source_sha256"] == (
            "sha256:" + hashlib.sha256(champion).hexdigest()
        )
        assert row["champion_submission_id"] == prior_round_id + "-miner"
        assert row["champion_hotkey"] == MINER_A
        assert objects.get(participants[0]["source_ref"]) == champion
        assert requested == [expected_url, expected_url]

        # Restart and repeat use the stored snapshot and funding owner.
        restarted = opted_in()
        restarted._config.defaults.reasoning_baseline_source_url = old_url
        restarted._store = store
        restarted._objects = objects
        restarted._config.baseline_source_fetcher = lambda *_args: pytest.fail("frozen source refetched")
        assert restarted.freeze_participants(next_round["round_id"]) == participants
        assert store.get_round(next_round["round_id"])["champion_submission_id"] == (
            prior_round_id + "-miner"
        )
        restarted._objects = SimpleNamespace(get_bounded=lambda *_args: original)
        with pytest.raises(arena_service.ServiceError, match="reasoning_baseline_source_mismatch"):
            restarted.freeze_participants(next_round["round_id"])
    finally:
        control.close()
        store.close()
