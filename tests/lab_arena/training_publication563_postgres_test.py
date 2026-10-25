"""The real publication SQL admits coherent training failures without losing scores."""

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

import pytest

from tests.lab_arena.complete_baseline_final_authority_postgres_test import (
    test_normal_complete_baseline_publishes_and_activates_with_all_guards as _prove_complete,
)
from tests.lab_arena.flat_per_icp_cost500_postgres_test import (
    database_with_flat_cost_migration,
)
from scripts.verify_arena_full_thinking_e2e import full_thinking_database

MIGRATION = (Path(__file__).resolve().parents[2] /
             "scripts/563-lab-arena-training-publication-eligibility.sql")
GUARD = "public.lab_arena__per_icp_publication_valid(%s,%s::jsonb)"


@pytest.fixture()
def database():
    yield from database_with_flat_cost_migration()


@pytest.fixture()
def current_database():
    yield from full_thinking_database()


def _verify(cursor, round_id, ranking):
    if cursor.connection.get_transaction_status() == 0:
        cursor.execute("BEGIN")
    cursor.execute("SAVEPOINT publication_attempt")
    try:
        cursor.execute("SELECT " + GUARD, (round_id, json.dumps(ranking)))
        assert cursor.fetchone()[0] is True
        cursor.execute("RELEASE SAVEPOINT publication_attempt")
        return True
    except Exception as exc:
        cursor.execute("ROLLBACK TO SAVEPOINT publication_attempt")
        cursor.execute("RELEASE SAVEPOINT publication_attempt")
        assert exc.pgcode == "22023"
        return False


def _training(eligible):
    return {"eligible": eligible, "reasons": [] if eligible else ["selected_run_export_missing"],
            "source_review": {"eligible": True},
            "capture": {"eligible": eligible}}


def test_real_sql_training_gap_and_legacy_authority(database, tmp_path, monkeypatch):
    # The ordinary service produces real measured runs and cost summaries.
    _prove_complete(
        database, tmp_path, monkeypatch, False, True, 13,
    )
    psycopg2, dsn = database
    with psycopg2.connect(**dsn) as connection, connection.cursor() as cursor:
        cursor.execute("SELECT round_id,publication_doc FROM public.lab_arena_rounds "
                       "WHERE status='published' ORDER BY created_at DESC LIMIT 1")
        round_id, publication = cursor.fetchone()
        baseline = next(row for row in publication["final_ranking"] if row["is_baseline"])
        contender = next(row for row in publication["final_ranking"] if not row["is_baseline"])
        assert baseline["final_score"] is not None
        assert contender["final_score"] is not None
        assert _verify(cursor, round_id, baseline)
        assert _verify(cursor, round_id, contender)

        # Freeze the policy marker without changing any scored or cost rows.
        cursor.execute("SET LOCAL session_replication_role=replica")
        cursor.execute("UPDATE public.lab_arena_rounds SET configuration_doc="
                       "jsonb_set(configuration_doc,'{reasoning_capture_policy}',"
                       "'\"full_generated_thinking_v1\"'::jsonb) WHERE round_id=%s",
                       (round_id,))
        cursor.execute("SET LOCAL session_replication_role=origin")
        gap = deepcopy(contender)
        gap.update(eligible=False, eligibility_reason="training_data_incomplete",
                   training_data_eligibility=_training(False))
        assert not _verify(cursor, round_id, gap)  # Red against migration 500.
        before_identity = _identity(cursor)
        cursor.execute(MIGRATION.read_text())
        after_identity = _identity(cursor)
        assert after_identity == before_identity
        assert _verify(cursor, round_id, gap)
        assert gap["final_score"] == contender["final_score"]

        valid = deepcopy(contender)
        valid["training_data_eligibility"] = _training(True)
        assert _verify(cursor, round_id, valid)
        baseline_gap = deepcopy(baseline)
        baseline_gap.update(eligible=False, eligibility_reason="training_data_incomplete",
                            training_data_eligibility=_training(False))
        assert _verify(cursor, round_id, baseline_gap)
        assert baseline_gap["final_score"] == baseline["final_score"]
        forged = deepcopy(gap)
        forged["cost_summary"]["per_icp"][0]["competition_sourcing_microusd"] += 1
        assert not _verify(cursor, round_id, forged)
        forged = deepcopy(valid)
        forged["training_data_eligibility"]["source_review"]["eligible"] = False
        assert not _verify(cursor, round_id, forged)
        cursor.execute(MIGRATION.read_text())
        assert _identity(cursor) == after_identity

        cursor.execute("BEGIN")
        cursor.execute("SET LOCAL session_replication_role=replica")
        cursor.execute("UPDATE public.lab_arena_rounds SET configuration_doc="
                       "configuration_doc - 'reasoning_capture_policy' WHERE round_id=%s",
                       (round_id,))
        cursor.execute("SET LOCAL session_replication_role=origin")
        assert _verify(cursor, round_id, contender)
        assert not _verify(cursor, round_id, gap)


def _identity(cursor):
    cursor.execute("SELECT pg_catalog.pg_get_userbyid(proowner),proacl,prosecdef,"
                   "provolatile,proconfig FROM pg_catalog.pg_proc WHERE oid="
                   "'public.lab_arena__per_icp_publication_valid(text,jsonb)'::regprocedure")
    return cursor.fetchone()


@pytest.mark.parametrize("stopped", [False, True])
def test_service_publishes_incomplete_contender_and_crowns_eligible_peer(
    current_database, tmp_path, monkeypatch, stopped,
):
    """Exercise the ordinary service and database publication transition."""
    from tests.lab_arena import test_lab_arena_service_round as fixtures
    from tests.lab_arena.baseline_scored_first_postgres_test import (
        _baseline_first_judge, _enable_verified_proxy_runtime,
    )
    from tests.lab_arena.test_integrity_round import IntegrityHarness

    psycopg2, dsn = current_database
    harness = IntegrityHarness(lambda: psycopg2.connect(**dsn), tmp_path,
                               challengers=["Miner0", "Miner1"], runners=["alpha"])
    harness.service.config.defaults = replace(
        harness.service.config.defaults,
        execution_sequence_from="2000-01-01T00:00:00Z",
        per_icp_cost_policy=False,
        flat_per_icp_cost_policy=True,
        rewards_enabled=True,
    )
    _enable_verified_proxy_runtime(harness)

    def judge(companies, icp, reference):
        rows = _baseline_first_judge(companies, icp, reference)
        for row in rows:
            if row["final_score"] == 80.0:
                row.update(final_score=0.0, company_qualified=False,
                           verifier_gate_receipts=[{
                               "gate": "company_fit", "decision": "mismatch",
                           }])
            else:
                row["final_score"] = 80.0
        return rows

    monkeypatch.setattr(fixtures, "deterministic_scorer", judge)
    assert fixtures._start_round(harness, day=16, epoch=62416) == 3
    failed_id = next(submission_id for submission_id, flavor in harness.flavors.items()
                     if flavor == "Miner0")
    if stopped:
        # The real runner records failed execution results with no output.
        # Scoring and publication decide their consequence from those rows.
        harness.broken.add(failed_id)
    harness.clock.advance_to(harness.schedule()["stage_1_start"])
    harness.advance_until("scored", runners=1)
    assert harness.status() == "scored"
    row = harness.service.store.get_round(harness.round_id)
    assert {harness.flavors[p["submission_id"]] for p in row["participants"]} == {
        "PublicBaseline", "Miner0", "Miner1",
    }

    # The service's raw export reader is tested separately. Supply controlled
    # evidence outcomes here and exercise its decision plus actual SQL write.
    def training(_round, submission_id, _runs):
        return _training(harness.flavors[submission_id] == "Miner1")

    monkeypatch.setattr(harness.service, "_submission_training_eligibility", training)
    with psycopg2.connect(**dsn) as connection, connection.cursor() as cursor:
        cursor.execute("SET LOCAL session_replication_role=replica")
        cursor.execute("UPDATE public.lab_arena_rounds SET configuration_doc="
                       "configuration_doc || %s::jsonb WHERE round_id=%s",
                       (json.dumps({
                           "reasoning_capture_policy": "full_generated_thinking_v1",
                           "execution_icp_cap_microusd": 6_000_000,
                           "cost_per_company_microusd": 6_000_000,
                           "execution_cap_microusd": 120_000_000,
                       }), harness.round_id))
        cursor.execute("SET LOCAL session_replication_role=origin")

    outcome = harness.service.publish(harness.round_id)
    assert outcome["status"] == "ok"
    assert outcome["king_outcome"] == "crowned"
    saved = harness.service.store.get_round(harness.round_id)
    assert saved["status"] == "published"
    ranking = {harness.flavors[item["submission_id"]]: item
               for item in saved["publication_doc"]["final_ranking"]}
    assert all(item["final_score"] is not None for flavor, item in ranking.items()
               if flavor != "Miner0")
    assert ranking["Miner0"]["eligible"] is False
    assert ranking["Miner0"]["eligibility_reason"] == "training_data_incomplete"
    if stopped:
        failed_runs = harness.service.store.list_runs(
            harness.round_id, submission_id=failed_id, kind="execute",
        )
        assert failed_runs and all(run["terminal_cause"] == "model_error"
                                   and run["output_ref"] is None for run in failed_runs)
        # The failed execution has no measured output or score. Publication
        # keeps the unmeasured result, rather than inventing a zero.
        assert ranking["Miner0"]["final_score"] is None
    else:
        assert ranking["Miner0"]["final_score"] is not None
    assert ranking["Miner1"]["eligible"] is True
    assert saved["publication_doc"]["king_decision"]["winner_submission_id"] == next(
        submission_id for submission_id, flavor in harness.flavors.items()
        if flavor == "Miner1"
    )
    assert ranking["PublicBaseline"]["final_score"] is not None
    if not stopped:
        return

    # Follow this same published winner through the real Git and database
    # promotion path, then freeze the next round from that repository head.
    from lab_arena.promotion import GitPromoter
    from tests.lab_arena.full_thinking_round_test import approved_registry

    repository_root = tmp_path / "promotion-repository"
    repository_root.mkdir()
    remote = fixtures.promotion_repository(repository_root)
    old_head = subprocess.run(
        ("git", "--git-dir", str(remote), "rev-parse", "lab"),
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    harness.service._config.baseline_promoter_factory = lambda: GitPromoter(
        str(remote), tmp_path / "promotion-objects",
    )

    def promoted_source(url, _limit):
        assert url.endswith(".tar.gz")
        return subprocess.run(
            ("git", "--git-dir", str(remote), "archive", "--format=tar.gz", "lab"),
            check=True, capture_output=True,
        ).stdout

    first_publication = deepcopy(saved["publication_doc"])
    first_configuration = deepcopy(saved["configuration_doc"])
    harness.clock.now = datetime.fromisoformat(
        str(saved["published_at"]).replace("Z", "+00:00")
    )
    assert harness.service.promote_pending_baselines() == {"status": "ok", "promoted": 1}
    promoted = harness.service.store.get_round(harness.round_id)
    commit = promoted["promotion_doc"]["commit"]
    assert promoted["baseline_promoted_at"] is not None
    assert commit != old_head
    for branch in ("main", "lab"):
        assert subprocess.run(
            ("git", "--git-dir", str(remote), "rev-parse", branch),
            check=True, capture_output=True, text=True,
        ).stdout.strip() == commit
    with tarfile.open(fileobj=io.BytesIO(promoted_source("winner.tar.gz", 0)), mode="r:gz") as archive:
        assert archive.extractfile("flavor.txt").read() == b"Miner1"

    bootstrap_url = (
        "https://github.com/leadpoet/leadpoet-sales-agent/archive/"
        + old_head + ".tar.gz"
    )
    harness.service.config.defaults = replace(
        harness.service.config.defaults,
        reasoning_capture_from="2026-01-01T00:00:00Z",
        reasoning_baseline_source_url=bootstrap_url,
        reasoning_baseline_source_sha256=(
            "sha256:" + hashlib.sha256(harness.baseline_source).hexdigest()
        ),
    )
    harness.service._config.reasoning_registry_source = approved_registry
    harness.service._config.baseline_source_fetcher = promoted_source
    harness.clock.now = datetime.now(timezone.utc)
    next_cutoff = harness.clock.now + timedelta(days=1)
    next_config = harness.service.create_round(
        next_cutoff, round_id="arena-2099-10-17-promoted",
    )
    assert next_config["reasoning_capture_policy"] == "full_generated_thinking_v1"
    harness.round_id = next_config["round_id"]
    harness.clock.advance_to(harness.schedule()["submission_cutoff"])
    assert harness.service.advance_round(harness.round_id)["status"] == "ok"
    next_row = harness.service.store.get_round(harness.round_id)
    baseline = next(participant for participant in next_row["participants"]
                    if participant["is_king"])
    assert len(next_row["participants"]) == 1
    assert baseline["miner_hotkey"] == harness.baseline_hotkey
    assert next_row["champion_submission_id"] == saved["publication_doc"]["king_decision"]["winner_submission_id"]
    assert next_row["champion_hotkey"] == saved["king_hotkey"]
    assert next_row["configuration_doc"]["baseline_source_url"].endswith(commit + ".tar.gz")
    baseline_row = harness.service.store.get_submission(baseline["submission_id"])
    assert harness.objects.get(baseline_row["source_ref"]) == promoted_source("winner.tar.gz", 0)
    original = harness.service.store.get_round(promoted["round_id"])
    assert original["publication_doc"] == first_publication
    assert original["configuration_doc"] == first_configuration
