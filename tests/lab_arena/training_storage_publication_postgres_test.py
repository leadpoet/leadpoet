"""A later output read fault uses bytes verified by the same publication."""

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json

import pytest

from lab_arena import code_review, contracts, training_reader
from lab_arena.output import MAX_OUTPUT_BYTES
from tests.lab_arena.lab_arena_pg_harness import current_service_database
from tests.lab_arena import test_lab_arena_service_round as fixtures
from tests.lab_arena.baseline_scored_first_postgres_test import (
    _baseline_first_judge, _enable_verified_proxy_runtime,
)
from tests.lab_arena.test_integrity_round import IntegrityHarness
from tests.lab_arena.test_lab_arena_service_round import daily_icps
from tests.lab_arena.training_coverage_publication_postgres_test import _rename_run
from tests.lab_arena.training_eligibility_test import captured_example
from tests.lab_arena.training_steps_test import bind_selected_output, seal


@pytest.fixture()
def database():
    yield from current_service_database()


def test_later_missing_output_keeps_measured_publication(database, tmp_path, monkeypatch):
    psycopg2, dsn = database
    monkeypatch.setattr(fixtures, "shared_objects_root", lambda *_: tmp_path / "objects")
    harness = IntegrityHarness(lambda: psycopg2.connect(**dsn), tmp_path,
                               challengers=["Miner0", "Miner1"], runners=["alpha"])
    harness.service.config.defaults = replace(
        harness.service.config.defaults,
        benchmark_icp_count=10,
        execution_sequence_from="2000-01-01T00:00:00Z",
        per_icp_cost_policy=False, flat_per_icp_cost_policy=True,
        rewards_enabled=True,
    )
    harness.service.config.daily_icp_source = lambda **kwargs: {
        "status": "ready", "set_id": kwargs["set_id"], "icps": daily_icps()[:10]}
    _enable_verified_proxy_runtime(harness)

    def judge(companies, icp, reference):
        rows = _baseline_first_judge(companies, icp, reference)
        for row in rows:
            if row["final_score"] == 80.0:
                row.update(final_score=0.0, company_qualified=False,
                           verifier_gate_receipts=[{
                               "gate": "company_fit", "decision": "mismatch"}])
            else:
                row["final_score"] = 80.0
        return rows

    monkeypatch.setattr(fixtures, "deterministic_scorer", judge)
    harness.chain.epoch = 62597
    harness.clock.now = datetime.now(timezone.utc)
    config = harness.service.create_round(
        harness.clock.now + timedelta(minutes=30), round_id="arena-2026-10-29")
    harness.round_id = config["round_id"]
    for flavor in harness.challengers:
        harness.submit(flavor, harness.round_id)
    harness.clock.advance_to(harness.schedule()["submission_cutoff"])
    assert harness.service.advance_round(harness.round_id)["status"] == "ok"
    for participant in harness.service.store.get_round(harness.round_id)["participants"]:
        harness.flavors.setdefault(participant["submission_id"], "PublicBaseline")
    harness.clock.advance_to(harness.schedule()["stage_1_start"])
    harness.advance_until("scored", runners=1)
    assert harness.status() == "scored"

    selected = {}
    for submission in harness.service.store.list_submissions(harness.round_id):
        runs = harness.service.store.list_runs(
            harness.round_id, submission_id=submission["submission_id"], kind="execute")
        selected[submission["submission_id"]] = sorted(
            (run for run in runs if run["status"] == "accepted"),
            key=lambda run: run["icp_position"])
        assert len(selected[submission["submission_id"]]) == 10
    affected_id = next(identity for identity, flavor in harness.flavors.items()
                       if flavor == "Miner0")
    target = selected[affected_id][-1]
    target_ref = target["output_ref"]
    assert target_ref == "arena/%s/outputs/%s.json" % (harness.round_id, target["run_id"])

    class LaterMissingStore:
        def __init__(self, source):
            self.source = source
            self.first_reads = 0
            self.later_missing = 0

        def get_bounded(self, ref, limit):
            if ref == target_ref:
                if self.first_reads:
                    self.later_missing += 1
                    raise FileNotFoundError("selected output unavailable after publication read")
                self.first_reads += 1
            return self.source.get_bounded(ref, limit)

        def get_optional_bounded(self, ref, limit):
            if ref == target_ref and self.first_reads:
                self.later_missing += 1
                return None
            optional = getattr(self.source, "get_optional_bounded", None)
            return optional(ref, limit) if callable(optional) else self.get_bounded(ref, limit)

        def __getattr__(self, name):
            return getattr(self.source, name)

    fault_store = LaterMissingStore(harness.service._objects)
    harness.service._objects = fault_store

    # Capture proof is supplied by the existing full-thinking fixture. The
    # object read itself uses the real publication cache and bounded read view.
    def export(_store, read_view, run_id):
        execution = next(run for runs in selected.values() for run in runs
                         if run["run_id"] == run_id)
        if run_id == target["run_id"]:
            assert read_view.get_bounded(target_ref, MAX_OUTPUT_BYTES) == (
                fault_store.source.get_bounded(target_ref, MAX_OUTPUT_BYTES))
        example = bind_selected_output(_rename_run(deepcopy(captured_example()), run_id))
        example["round_id"] = harness.round_id
        example["submission_id"] = execution["submission_id"]
        submission = harness.service.store.get_submission(execution["submission_id"])
        baseline = submission.get("is_king") is True
        example["role"] = "baseline" if baseline else "miner"
        source_hash = submission.get("source_archive_sha256")
        example["source"] = {
            "sha256": source_hash if source_hash else contracts.hash_bytes(b"frozen baseline"),
            "hash_origin": "recorded_archive_hash" if source_hash else "computed_from_frozen_object"}
        if run_id == target["run_id"]:
            example["output_identity"] = {
                "ref": target_ref, "document_hash": contracts.document_hash(example["output"])}
        return seal(example)

    monkeypatch.setattr(training_reader, "export_run", export)
    with psycopg2.connect(**dsn) as connection, connection.cursor() as cursor:
        cursor.execute("SET LOCAL session_replication_role=replica")
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
                "evidence_claims": ["observation_to_action", "finish_continue"]}}
            cursor.execute(
                "UPDATE public.lab_arena_submissions SET code_review_status='passed',"
                "code_review_doc=coalesce(code_review_doc,'{}'::jsonb) || %s::jsonb "
                "WHERE submission_id=%s", (json.dumps(review), submission_id))
        cursor.execute(
            "UPDATE public.lab_arena_rounds SET configuration_doc=configuration_doc || %s::jsonb "
            "WHERE round_id=%s", (json.dumps({
                "reasoning_capture_policy": "full_generated_thinking_v1",
                "training_minimum_icp_coverage_percent": 80,
                "execution_icp_cap_microusd": 6_000_000,
                "cost_per_company_microusd": 6_000_000,
                "execution_cap_microusd": 120_000_000,
            }), harness.round_id))
        cursor.execute("SET LOCAL session_replication_role=origin")

    outcome = harness.service.publish(harness.round_id)
    assert outcome["status"] == "ok"
    saved = harness.service.store.get_round(harness.round_id)
    assert saved["status"] == "published"
    ranking = {harness.flavors[row["submission_id"]]: row
               for row in saved["publication_doc"]["final_ranking"]}
    assert all(row["final_score"] is not None for row in ranking.values())
    assert ranking["Miner0"]["training_data_eligibility"]["capture"]["counts"]["complete_runs"] == 9
    assert ranking["Miner0"]["eligible"] is True, ranking["Miner0"]["training_data_eligibility"]
    assert ranking["Miner1"]["eligible"] is True
    assert fault_store.first_reads == 1
    assert fault_store.later_missing >= 1
