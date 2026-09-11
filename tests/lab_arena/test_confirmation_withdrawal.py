"""Regression: a finalist cannot veto its peers by withdrawing judge credentials."""
import copy
from datetime import datetime, timedelta, timezone

import pytest
from lab_arena import verify
from lab_arena.store import ArenaStoreError
from qualification.scoring.arena_integrity import canonical_company_identity
from tests.lab_arena.test_integrity_round import IntegrityHarness, database  # noqa: F401
from tests.lab_arena import test_lab_arena_service_round as fixtures


@pytest.mark.parametrize("failure", ["credential_error", "budget_exhausted", "judge_error", "baseline_judge_error"])
def test_confirmation_failure_is_isolated_only_for_miner_account_failures(database, tmp_path, monkeypatch, failure):
    psycopg2, dsn = database
    connect = lambda: psycopg2.connect(**dsn)
    harness = IntegrityHarness(connect, tmp_path, challengers=["Bad", "Good"], runners=["alpha"])

    def judge(companies, icp, reference):
        indexes, _ = verify.bucket_skip(icp, companies)
        rows = []
        for index in indexes:
            name = companies[index]["company_name"]
            value = 40.0 if name.startswith("PublicBaseline") else 80.0 if name.startswith("Bad") else 70.0
            rows.append({
                "final_score": value, "company_index": index,
                "company_identity_key": canonical_company_identity(companies[index]).key,
                "company_qualified": True, "duplicate_company": False,
                "verifier_gate_receipts": [{"gate": "company_fit", "decision": "match"}],
                "intent_signals_detail": [], "failure_reason": "",
            })
        return rows

    monkeypatch.setattr(fixtures, "deterministic_scorer", judge)
    harness.clock.now = datetime.now(timezone.utc)
    round_id = "arena-2026-11-02-withdraw"
    harness.service.create_round(harness.clock.now + timedelta(hours=12), round_id=round_id)
    harness.round_id = round_id
    bad = harness.submit("Bad", round_id)
    good = harness.submit("Good", round_id)
    harness.clock.advance_to(harness.schedule()["submission_cutoff"])
    harness.advance_until("stage3_scoring")
    cohort = harness.service.store.get_round(round_id)["confirmation_cohort"]
    assert bad in cohort["submission_ids"] and good in cohort["submission_ids"]

    if failure == "credential_error":
        credentials = harness.service.config.credential_manager
        original = credentials.runtime_key
        monkeypatch.setattr(credentials, "runtime_key", lambda row, provider:
            "miner-refused" if row["submission_id"] == bad else original(row, provider))
    elif failure == "budget_exhausted":
        original = harness.service.store.reserve_call
        def refuse(**kwargs):
            run = harness.service.store.get_run(kwargs["run_id"])
            if run["submission_id"] == bad and run["icp_position"] == 24:
                return {"status": "refused", "reason": "budget_exhausted"}
            return original(**kwargs)
        monkeypatch.setattr(harness.service.store, "reserve_call", refuse)
    else:
        harness.sandbox.judge_failures.add(("PublicBaseline" if failure == "baseline_judge_error" else "Bad", 24))

    if failure in ("judge_error", "baseline_judge_error"):
        for _ in range(20):
            harness.run_stage_with_runners(1)
            result = harness.service.advance_round(round_id)
            if result.get("status") == "cancelled":
                break
        current = harness.service.store.get_round(round_id)
        assert current["status"] == "cancelled"
        assert not current.get("publication_doc")
        return

    harness.advance_until("confirmed")
    harness.service = harness.build_service()  # Disqualification must survive restart.
    assert harness.service._confirmation_account_failures(harness.service.store.get_round(round_id)) == {bad}
    transition = harness.service.store.transition_round
    captured = {}
    def capture(*args):
        captured["args"] = copy.deepcopy(args)
        return {"status": "captured"}
    monkeypatch.setattr(harness.service.store, "transition_round", capture)
    assert harness.service.publish(round_id)["status"] == "captured"
    args = captured["args"]
    proposal = args[3]["publication_doc"]
    ranking = {row["submission_id"]: row for row in proposal["final_ranking"]}
    assert proposal["king_decision"]["winner_submission_id"] == good
    assert ranking[bad]["eligible"] is False
    assert ranking[bad]["eligibility_reason"] == "confirmation_account_failure"
    assert ranking[bad]["final_score"] is None
    assert ranking[bad]["main_score"] == 80.0
    assert ranking[bad]["cost_summary"]["qualified_company_count"] == 100
    assert ranking[good]["final_score"] == 70.0
    assert harness.service.store.get_round(round_id)["confirmation_cohort"] == cohort

    forged = copy.deepcopy(args[3])
    next(row for row in forged["publication_doc"]["final_ranking"] if row["submission_id"] == bad)["final_score"] = 99
    with pytest.raises(ArenaStoreError, match="withdrawn_confirmation_score_invalid"):
        transition(*args[:3], forged)
    forged = copy.deepcopy(args[3])
    next(row for row in forged["publication_doc"]["final_ranking"] if row["submission_id"] == bad)["eligible"] = True
    with pytest.raises(ArenaStoreError, match="cost_report_mismatch"):
        transition(*args[:3], forged)

    monkeypatch.setattr(harness.service.store, "transition_round", transition)
    assert harness.service.publish(round_id)["king_outcome"] == "crowned"
    current = harness.service.store.get_round(round_id)
    assert current["publication_doc"]["king_decision"]["winner_submission_id"] == good
    score_runs = harness.service.store.list_runs(round_id, stage=3, kind="score")
    assert sum(row["submission_id"] == good and row["status"] == "accepted" for row in score_runs) == 5
    # The existing runner classifies a miner-funded judge budget refusal as
    # credential_error. Both are terminal account failures, not infrastructure.
    assert any(row["submission_id"] == bad and row["terminal_cause"] == "credential_error" for row in score_runs)
    fixtures.assert_canary_absent(harness, connect)
