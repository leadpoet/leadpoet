"""The frozen 80% capture floor and real publication RPC on local PostgreSQL."""

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import subprocess
import tarfile

import pytest

from lab_arena import code_review, contracts, training_reader
from tests.lab_arena.lab_arena_pg_harness import current_service_database
from tests.lab_arena import test_lab_arena_service_round as fixtures
from tests.lab_arena.baseline_scored_first_postgres_test import (
    _baseline_first_judge, _enable_verified_proxy_runtime,
)
from tests.lab_arena.test_integrity_round import IntegrityHarness
from tests.lab_arena.training_eligibility_test import captured_example, _unsettled_selected_call, refresh_response_links
from tests.lab_arena.training_steps_test import bind_selected_output, seal
from tests.lab_arena.test_lab_arena_service_round import daily_icps


@pytest.fixture()
def database():
    yield from current_service_database()


def _rename_run(value, run_id):
    if isinstance(value, dict):
        return {run_id if key == "run" else key: _rename_run(item, run_id)
                for key, item in value.items()}
    if isinstance(value, list):
        return [_rename_run(item, run_id) for item in value]
    return run_id if value == "run" else value


@pytest.mark.parametrize("missing,expected_eligible,gap", [
    (2, True, "missing"), (3, False, "missing"),
    (2, True, "unsettled"), (3, True, "unsettled"),
    (2, True, "integrity"),
    (2, True, "mixed_thinking"), (3, False, "mixed_thinking"),
])
def test_frozen_eighty_percent_publishes_scores_and_keeps_peer_eligible(
    database, tmp_path, monkeypatch, missing, expected_eligible, gap,
):
    psycopg2, dsn = database
    # Each parameter owns a fresh database, so it must own its object store too.
    # The next round within this case still shares the same stored evidence.
    monkeypatch.setattr(fixtures, "shared_objects_root", lambda *_: tmp_path / "objects")
    harness = IntegrityHarness(lambda: psycopg2.connect(**dsn), tmp_path,
                               challengers=["Miner0", "Miner1"], runners=["alpha"])
    harness.service.config.defaults = replace(
        harness.service.config.defaults,
        benchmark_icp_count=10,
        execution_sequence_from="2000-01-01T00:00:00Z",
        per_icp_cost_policy=False,
        flat_per_icp_cost_policy=True,
        rewards_enabled=True,
    )
    harness.service.config.daily_icp_source = lambda **kwargs: {
        "status": "ready", "set_id": kwargs["set_id"], "icps": daily_icps()[:10],
    }
    _enable_verified_proxy_runtime(harness)

    def judge(companies, icp, reference):
        rows = _baseline_first_judge(companies, icp, reference)
        for row in rows:
            if row["final_score"] == 80.0:
                row.update(final_score=0.0, company_qualified=False,
                           verifier_gate_receipts=[{"gate": "company_fit", "decision": "mismatch"}])
            else:
                row["final_score"] = 80.0
        return rows

    monkeypatch.setattr(fixtures, "deterministic_scorer", judge)
    harness.chain.epoch = 62500 + missing
    harness.clock.now = datetime.now(timezone.utc)
    cutoff = datetime.now(timezone.utc) + timedelta(minutes=30)
    config = harness.service.create_round(cutoff, round_id="arena-2026-10-%02d" % (18 + missing))
    harness.round_id = config["round_id"]
    for flavor in harness.challengers:
        harness.submit(flavor, harness.round_id)
    harness.clock.advance_to(harness.schedule()["submission_cutoff"])
    advance = harness.service.advance_round(harness.round_id)
    assert advance["status"] == "ok", (advance, harness.service.store.get_round(harness.round_id).get("cancel_reason"))
    for participant in harness.service.store.get_round(harness.round_id)["participants"]:
        harness.flavors.setdefault(participant["submission_id"], "PublicBaseline")
    harness.clock.advance_to(harness.schedule()["stage_1_start"])
    harness.advance_until("scored", runners=1)
    assert harness.status() == "scored"

    # Test raw records are sealed copies of the strict capture fixture. The
    # actual service selects accepted execution IDs and performs the gate.
    selected = {}
    for submission in harness.service.store.list_submissions(harness.round_id):
        runs = harness.service.store.list_runs(
            harness.round_id, submission_id=submission["submission_id"], kind="execute")
        accepted = sorted((run for run in runs if run["status"] == "accepted"),
                          key=lambda run: run["icp_position"])
        assert len(accepted) == 10
        selected[submission["submission_id"]] = accepted
    bad_id = next(submission_id for submission_id, flavor in harness.flavors.items()
                  if flavor == "Miner0")
    omitted = {run["run_id"] for run in selected[bad_id][-missing:]}

    def export(_store, _objects, run_id):
        if run_id in omitted and gap == "missing":
            raise FileNotFoundError("synthetic missing training export")
        if run_id in omitted and gap == "integrity":
            raise training_reader.ExportError("execution_output_hash_mismatch")
        submission_id = next(sub for sub, runs in selected.items()
                             if any(run["run_id"] == run_id for run in runs))
        submission = harness.service.store.get_submission(submission_id)
        example = bind_selected_output(_rename_run(deepcopy(captured_example()), run_id))
        example["submission_id"] = submission_id
        baseline = submission.get("is_king") is True
        example["role"] = "baseline" if baseline else "miner"
        source_hash = submission.get("source_archive_sha256")
        example["source"] = {
            "sha256": source_hash if source_hash else contracts.hash_bytes(b"frozen baseline"),
            "hash_origin": "recorded_archive_hash" if source_hash else "computed_from_frozen_object",
        }
        if run_id in omitted and gap == "mixed_thinking":
            example["provider_calls"][0]["response"]["choices"][0]["finish_reason"] = "length"
            refresh_response_links(example, 0)
            for index in (1, 2):
                message = example["provider_calls"][index]["response"]["choices"][0]["message"]
                for field in ("reasoning", "reasoning_content", "reasoning_details"):
                    message.pop(field, None)
                refresh_response_links(example, index)
        seal(example)
        return _unsettled_selected_call(example) if run_id in omitted and gap == "unsettled" else example

    monkeypatch.setattr(training_reader, "export_run", export)
    with psycopg2.connect(**dsn) as connection, connection.cursor() as cursor:
        cursor.execute("SET LOCAL session_replication_role=replica")
        # Isolate capture coverage from source review: the fixture's fake
        # source is not model-directed, so install a coherent reviewed source
        # document without changing any execution or score row.
        for submission_id in selected:
            submission = harness.service.store.get_submission(submission_id)
            if submission.get("is_king"):
                continue
            digest = submission["source_archive_sha256"]
            review = {"passed": True, "research_control": {
                "policy_id": code_review.RESEARCH_POLICY_ID,
                "policy_hash": code_review.RESEARCH_POLICY_HASH,
                "reviewer_version": code_review.RESEARCH_REVIEWER_VERSION,
                "source_archive_sha256": digest,
                "classification": "model_directed",
                "evidence_claims": ["observation_to_action", "finish_continue"],
            }}
            cursor.execute("UPDATE public.lab_arena_submissions SET code_review_status='passed',"
                           "code_review_doc=coalesce(code_review_doc,'{}'::jsonb) || %s::jsonb "
                           "WHERE submission_id=%s", (json.dumps(review), submission_id))
        cursor.execute("UPDATE public.lab_arena_rounds SET configuration_doc="
                       "configuration_doc || %s::jsonb WHERE round_id=%s",
                       (json.dumps({
                           "reasoning_capture_policy": "full_generated_thinking_v1",
                           "training_minimum_icp_coverage_percent": 80,
                           "execution_icp_cap_microusd": 6_000_000,
                           "cost_per_company_microusd": 6_000_000,
                           "execution_cap_microusd": 120_000_000,
                       }), harness.round_id))
        if gap == "unsettled":
            for run_id in sorted(omitted):
                cursor.execute(
                    "INSERT INTO public.lab_arena_ledger"
                    "(entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,"
                    "call_identity,provider,operation_id,funding_source,amount_microusd,entry_doc) "
                    "SELECT 'uncertain',miner_hotkey,round_id,submission_id,run_id,stage,"
                    "%s,'openrouter','openrouter.chat','host',7000000,%s::jsonb "
                    "FROM public.lab_arena_runs WHERE run_id=%s",
                    (contracts.hash_bytes((run_id + ':unsettled').encode()),
                     json.dumps({"call": {"call_succeeded": True}}), run_id),
                )
                assert cursor.rowcount == 1
                cursor.execute(
                    "SELECT public.lab_arena_icp_cost_eligibility(%s,%s,%s,0)",
                    (harness.round_id, bad_id,
                     next(run["icp_position"] for run in selected[bad_id]
                          if run["run_id"] == run_id)),
                )
                current = cursor.fetchone()[0]
                assert current["execution"]["success_unresolved_calls"] >= 1
                assert current["execution"]["success_unresolved_microusd"] >= 7_000_000
                assert current["competition_sourcing_microusd"] < 6_000_000
                assert current["eligibility_reason"] == "eligible", current
            baseline_id = next(submission_id for submission_id, flavor in harness.flavors.items()
                               if flavor == "PublicBaseline")
            baseline_run = selected[baseline_id][0]
            cursor.execute(
                "INSERT INTO public.lab_arena_ledger"
                "(entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,"
                "call_identity,provider,operation_id,funding_source,amount_microusd,entry_doc) "
                "SELECT 'uncertain',miner_hotkey,round_id,submission_id,run_id,stage,"
                "%s,'openrouter','openrouter.chat','host',7000000,%s::jsonb "
                "FROM public.lab_arena_runs WHERE run_id=%s",
                (contracts.hash_bytes((baseline_run["run_id"] + ':unsettled').encode()),
                 json.dumps({"call": {"call_succeeded": True}}), baseline_run["run_id"]),
            )
            assert cursor.rowcount == 1
            cursor.execute("SELECT public.lab_arena_icp_cost_eligibility(%s,%s,%s,0)",
                           (harness.round_id, baseline_id, baseline_run["icp_position"]))
            assert cursor.fetchone()[0]["eligibility_reason"] == "eligible"
            cursor.execute("SELECT public.lab_arena__baseline_final_ready_v1(%s)",
                           (harness.round_id,))
            # Continuation publication materializes verified numeric projections
            # in its scored phase. Accepted judgments alone are not finality.
            assert cursor.fetchone()[0] is False
            assert all(run["per_icp_score"] is None for run in selected[baseline_id])
            baseline_judges = harness.service.store.list_runs(
                harness.round_id, submission_id=baseline_id, kind="score", status="accepted")
            assert len(baseline_judges) == 10
            assert {run["scored_run_id"] for run in baseline_judges} == {
                run["run_id"] for run in selected[baseline_id]}
            # Historical frozen full-thinking rounds without this setting keep
            # the old uncertainty decision; a newly frozen 90% or 100% floor
            # keeps confirmed-cost accounting.
            first = selected[bad_id][-1]
            cost_policy_probes = ((None, "provider_cost_uncertain"),
                                  (0, "provider_cost_uncertain"),
                                  (101, "provider_cost_uncertain"),
                                  ("80", "provider_cost_uncertain"),
                                  (90, "eligible"), (100, "eligible"))
            for floor, reason in cost_policy_probes:
                cursor.execute("SAVEPOINT cost_policy_probe")
                if floor is None:
                    cursor.execute("UPDATE public.lab_arena_rounds SET configuration_doc="
                                   "configuration_doc-'training_minimum_icp_coverage_percent' "
                                   "WHERE round_id=%s", (harness.round_id,))
                else:
                    cursor.execute("UPDATE public.lab_arena_rounds SET configuration_doc="
                                   "jsonb_set(configuration_doc,'{training_minimum_icp_coverage_percent}',%s::jsonb) "
                                   "WHERE round_id=%s", (json.dumps(floor), harness.round_id))
                cursor.execute("SELECT public.lab_arena_icp_cost_eligibility(%s,%s,%s,0)",
                               (harness.round_id, bad_id, first["icp_position"]))
                assert cursor.fetchone()[0]["eligibility_reason"] == reason
                cursor.execute("ROLLBACK TO SAVEPOINT cost_policy_probe")
            for outcome in (True, None):
                cursor.execute("SAVEPOINT confirmed_over_cap_probe")
                cursor.execute(
                    "INSERT INTO public.lab_arena_ledger"
                    "(entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,"
                    "call_identity,provider,operation_id,funding_source,amount_microusd,"
                    "entry_doc,terminal_response) "
                    "SELECT 'settlement',miner_hotkey,round_id,submission_id,run_id,stage,"
                    "%s,'openrouter','openrouter.chat','host',6000001,'{}'::jsonb,%s::jsonb "
                    "FROM public.lab_arena_runs WHERE run_id=%s",
                    (contracts.hash_bytes((first["run_id"] + ':over-cap').encode()),
                     json.dumps({"call_succeeded": outcome}), first["run_id"]),
                )
                cursor.execute("SELECT public.lab_arena_icp_cost_eligibility(%s,%s,%s,0)",
                               (harness.round_id, bad_id, first["icp_position"]))
                assert cursor.fetchone()[0]["eligibility_reason"] == "execution_cap_exceeded"
                cursor.execute("ROLLBACK TO SAVEPOINT confirmed_over_cap_probe")
            cursor.execute("SAVEPOINT active_call_probe")
            cursor.execute(
                "INSERT INTO public.lab_arena_ledger"
                "(entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,"
                "call_identity,provider,operation_id,funding_source,amount_microusd,entry_doc) "
                "SELECT 'dispatch',miner_hotkey,round_id,submission_id,run_id,stage,"
                "%s,'openrouter','openrouter.chat','host',0,'{}'::jsonb "
                "FROM public.lab_arena_runs WHERE run_id=%s",
                (contracts.hash_bytes((first["run_id"] + ':active').encode()), first["run_id"]),
            )
            cursor.execute("SELECT public.lab_arena_icp_cost_eligibility(%s,%s,%s,0)",
                           (harness.round_id, bad_id, first["icp_position"]))
            assert cursor.fetchone()[0]["eligibility_reason"] == "provider_calls_inflight"
            cursor.execute("ROLLBACK TO SAVEPOINT active_call_probe")
        cursor.execute("SET LOCAL session_replication_role=origin")

    outcome = harness.service.publish(harness.round_id)
    assert outcome["status"] == "ok"
    saved = harness.service.store.get_round(harness.round_id)
    assert saved["status"] == "published"
    if gap == "unsettled":
        # The real publisher must produce complete baseline authority while
        # retaining uncertainty. Probe legacy/frozen floor rules only after
        # that guarded writer, and roll every disposable probe back exactly.
        with psycopg2.connect(**dsn) as connection, connection.cursor() as cursor:
            def authority_snapshot():
                snapshot = {}
                for table in ("lab_arena_rounds", "lab_arena_runs", "lab_arena_ledger"):
                    cursor.execute("SELECT to_jsonb(t) FROM public." + table
                                   + " t WHERE round_id=%s ORDER BY to_jsonb(t)::text",
                                   (harness.round_id,))
                    snapshot[table] = cursor.fetchall()
                return snapshot

            before = authority_snapshot()
            cursor.execute("SELECT public.lab_arena__baseline_final_ready_v1(%s)",
                           (harness.round_id,))
            assert cursor.fetchone()[0] is True
            baseline_runs = harness.service.store.list_runs(
                harness.round_id, submission_id=baseline_id, kind="execute", status="accepted")
            assert len(baseline_runs) == 10
            assert all(run["per_icp_score"] is not None for run in baseline_runs)
            cursor.execute("SET LOCAL session_replication_role=replica")
            for floor, _reason in cost_policy_probes:
                cursor.execute("SAVEPOINT baseline_cost_policy_probe")
                if floor is None:
                    cursor.execute("UPDATE public.lab_arena_rounds SET configuration_doc="
                                   "configuration_doc-'training_minimum_icp_coverage_percent' "
                                   "WHERE round_id=%s", (harness.round_id,))
                else:
                    cursor.execute("UPDATE public.lab_arena_rounds SET configuration_doc="
                                   "jsonb_set(configuration_doc,'{training_minimum_icp_coverage_percent}',%s::jsonb) "
                                   "WHERE round_id=%s", (json.dumps(floor), harness.round_id))
                cursor.execute("SELECT public.lab_arena__baseline_final_ready_v1(%s)",
                               (harness.round_id,))
                assert cursor.fetchone()[0] is (floor in (90, 100))
                cursor.execute("ROLLBACK TO SAVEPOINT baseline_cost_policy_probe")
                assert authority_snapshot() == before
            cursor.execute("SET LOCAL session_replication_role=origin")
        assert harness.service.store.get_round(harness.round_id) == saved
    ranking = {harness.flavors[item["submission_id"]]: item
               for item in saved["publication_doc"]["final_ranking"]}
    assert all(item["final_score"] is not None for item in ranking.values())
    capture = ranking["Miner0"]["training_data_eligibility"]["capture"]
    assert capture["counts"]["complete_runs"] == (10 if gap == "unsettled" else 10 - missing)
    if gap in {"integrity", "missing"}:
        assert len(capture["counts"]["rejected_runs"]) == missing
        assert all(row["reasons"] == ["selected_run_export_invalid"]
                   for row in capture["counts"]["rejected_runs"])
    assert ranking["Miner0"]["eligible"] is expected_eligible, ranking["Miner0"]["training_data_eligibility"]
    assert ranking["Miner1"]["eligible"] is True
    if not expected_eligible:
        assert ranking["Miner0"]["eligibility_reason"] == "training_data_incomplete"
        assert saved["publication_doc"]["king_decision"]["winner_submission_id"] == next(
            submission_id for submission_id, flavor in harness.flavors.items()
            if flavor == "Miner1")

        # The same eligible winner is promoted through the real local Git
        # path, then adopted as the next round's baseline.
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

        original_publication = deepcopy(saved["publication_doc"])
        original_configuration = deepcopy(saved["configuration_doc"])
        harness.clock.now = datetime.fromisoformat(
            str(saved["published_at"]).replace("Z", "+00:00")
        )
        assert harness.service.promote_pending_baselines() == {"status": "ok", "promoted": 1}
        promoted = harness.service.store.get_round(harness.round_id)
        commit = promoted["promotion_doc"]["commit"]
        assert commit != old_head
        promoted_candidate = harness.service.store.latest_promoted_champion(
            network_name=promoted["arena_network_name"], netuid=promoted["arena_netuid"])
        assert promoted_candidate is not None
        assert promoted_candidate["reasoning_capture_policy"] == "full_generated_thinking_v1"
        assert promoted_candidate["commit"] == commit
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
        next_config = harness.service.create_round(
            harness.clock.now + timedelta(days=1),
            round_id="arena-2099-10-23-cov%d%s" % (
                missing, gap[0]),
        )
        harness.round_id = next_config["round_id"]
        harness.clock.advance_to(harness.schedule()["submission_cutoff"])
        assert harness.service.advance_round(harness.round_id)["status"] == "ok"
        next_row = harness.service.store.get_round(harness.round_id)
        baseline = next(participant for participant in next_row["participants"]
                        if participant["is_king"])
        assert next_row["champion_submission_id"] == saved["publication_doc"]["king_decision"]["winner_submission_id"]
        assert next_row["configuration_doc"]["baseline_source_url"].endswith(commit + ".tar.gz")
        baseline_row = harness.service.store.get_submission(baseline["submission_id"])
        assert harness.objects.get(baseline_row["source_ref"]) == promoted_source("winner.tar.gz", 0)
        original = harness.service.store.get_round(promoted["round_id"])
        assert original["publication_doc"] == original_publication
        assert original["configuration_doc"] == original_configuration
