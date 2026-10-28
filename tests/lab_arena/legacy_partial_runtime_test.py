"""Legacy grouped reviews checkpoint terminal rows without accepting an ICP."""
from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest

from lab_arena import contact_policy, contracts, runner, runtime, scorer_entrypoint, scoring
from qualification.scoring import contact_verification, lead_scorer
from tests.test_arena_score_integrity import _icp, _positive_breakdown, _public_company
from tests.lab_arena.test_lab_arena_runner import BridgingRuntime, scoring_lease
from tests.lab_arena.trajectory_runner_test import LoggingApi, run_model
from tests.lab_arena.scorer_environment_fixture import isolated_scorer_environment


def _input(companies=None, *, contacts=False):
    policy = scoring.build_scorer_policy(scoring_adapter_version=(
        "qualification_contacts_v3" if contacts else "qualification_integrity_v2"
    ))
    companies = copy.deepcopy(companies or [_public_company("A", domain="a.example"), _public_company("B", domain="b.example")])
    icp = _icp()
    if contacts:
        icp.update(contact_policy="contacts_v1", target_roles=["VP Engineering"],
                   target_seniority="VP+", contact_geography={"countries": ["US"], "regions": [], "cities": []})
        for index, company in enumerate(companies):
            company["contact"] = {
                "full_name": "Ada Lovelace", "role": "VP Engineering",
                "linkedin_url": f"https://www.linkedin.com/in/contact-{index}/",
                "location": {"country": "US"}, "email": f"ada@company-{index}.example",
                "email_source": {"provider": "harvestapi", "tool": "harvestapi_get_profile",
                                 "record_id": f"profile-{index}"},
            }
    return scoring.build_scoring_input(
        scored_run_id="r1", icp=icp, companies=companies,
        policy=policy, evaluation_date="2026-10-12",
    )


def _install_verified_contacts(monkeypatch):
    attempted = []

    async def verified(company, *_args, **_kwargs):
        attempted.append(company["company_name"])
        return {
            "contact_qualified": True,
            "contact_identity_key": contact_verification.contact_identity_key(company["contact"]),
            "email_status": "valid",
            "contact_verification": {"decision": "verified", "subchecks": {
                key: {"status": "pass"} for key in
                ("claim", "identity", "source", "company", "role", "location", "email_attribution", "email_verification")
            }},
            "verifier_gate_receipts": [{"gate": "contact", "decision": "verified"}],
        }

    monkeypatch.setattr(contact_verification, "verify_contact", verified)
    return attempted


def _row(name="A", index=0, *, negative=False):
    row = _positive_breakdown(name, name.lower() + ".example")
    row.update(company_index=index, company_identity_key="domain:" + name.lower() + ".example",
               company_identity_alias_keys=["domain:" + name.lower() + ".example"],
               company_qualified=not negative, duplicate_company=False)
    if negative:
        row.update(final_score=0.0, intent_signals_detail=[], failure_reason="No verified primary intent")
    return row


@pytest.mark.parametrize("negative", [False, True])
def test_first_batch_checkpoints_before_sibling_exception_and_resume_skips_paid_work(monkeypatch, negative):
    document = _input()
    progress, attempted = [], []

    async def judge(**kwargs):
        name = kwargs["company"].company_name
        attempted.append(name)
        if name == "B":
            assert progress and [row["company_index"] for row in progress[-1]] == [0]
            raise RuntimeError("provider timeout")
        row = _positive_breakdown(name, "a.example")
        if negative:
            row.update(final_score=0.0, intent_signals_detail=[], failure_reason="No verified primary intent")
        return row

    monkeypatch.setattr(lead_scorer, "score_company_competition_intent", judge)
    failed = scorer_entrypoint.score_input(document, on_terminal_breakdowns=lambda rows: progress.append(copy.deepcopy(rows)))
    assert failed["failure"] == "judge_error"
    assert "breakdowns" not in failed
    assert [row["company_index"] for row in failed["terminal_breakdowns"]] == [0]
    assert attempted == ["A", "B", "B", "B"]
    attempted.clear()

    async def recovered(**kwargs):
        name = kwargs["company"].company_name
        attempted.append(name)
        return _positive_breakdown(name, "b.example")

    monkeypatch.setattr(lead_scorer, "score_company_competition_intent", recovered)
    document["prior_terminal_breakdowns"] = failed["terminal_breakdowns"]
    completed = scorer_entrypoint.score_input(document)
    assert attempted == ["B"]
    assert [row["company_index"] for row in completed["breakdowns"]] == [0, 1]
    assert completed["breakdowns"][0] == failed["terminal_breakdowns"][0]
    scoring.validate_breakdowns_for_item(completed["breakdowns"], icp=document["icp"], companies=document["companies"], integrity_policy=True)


def test_atomic_checkpoint_survives_process_interruption_during_first_batch(tmp_path, monkeypatch):
    document = _input()
    input_path, output_path = tmp_path / "input.json", tmp_path / "output.json"
    input_path.write_text(json.dumps(document))
    monkeypatch.setenv("LAB_ARENA_INPUT_PATH", str(input_path))
    monkeypatch.setenv("LAB_ARENA_OUTPUT_PATH", str(output_path))

    class Interrupted(BaseException):
        pass

    async def judge(**kwargs):
        name = kwargs["company"].company_name
        if name == "B":
            assert json.loads(output_path.read_text())["terminal_breakdowns"][0]["company_index"] == 0
            raise Interrupted()
        return _positive_breakdown(name, "a.example")

    monkeypatch.setattr(lead_scorer, "score_company_competition_intent", judge)
    with pytest.raises(Interrupted):
        scorer_entrypoint.main()
    failed = scoring.scoring_output_from_bytes(output_path.read_bytes())
    assert failed["failure"] == "judge_error" and len(failed["terminal_breakdowns"]) == 1
    assert not Path(str(output_path) + ".tmp").exists()


def test_sparse_original_index_and_second_outer_retry_accumulate(monkeypatch):
    companies = [_public_company("Skipped", domain="skip.example"), _public_company("A", domain="a.example"), _public_company("B", domain="b.example"), _public_company("C", domain="c.example")]
    companies[0]["employee_count"] = "1001-5000"
    document = _input(companies)
    attempted = []
    available = {"A"}

    async def judge(**kwargs):
        name = kwargs["company"].company_name
        attempted.append(name)
        if name not in available:
            raise RuntimeError("provider error")
        return _positive_breakdown(name, name.lower() + ".example")

    monkeypatch.setattr(lead_scorer, "score_company_competition_intent", judge)
    first = scorer_entrypoint.score_input(document)
    assert [row["company_index"] for row in first["terminal_breakdowns"]] == [1]
    document["prior_terminal_breakdowns"] = first["terminal_breakdowns"]
    available.add("B")
    attempted.clear()
    second = scorer_entrypoint.score_input(document)
    assert "A" not in attempted and "Skipped" not in attempted
    assert [row["company_index"] for row in second["terminal_breakdowns"]] == [1, 2]
    document["prior_terminal_breakdowns"] = second["terminal_breakdowns"]
    available.add("C")
    attempted.clear()
    final = scorer_entrypoint.score_input(document)
    assert attempted == ["C"]
    assert [row["company_index"] for row in final["breakdowns"]] == [1, 2, 3]


@pytest.mark.parametrize("negative", [False, True])
def test_resumed_alias_context_matches_uninterrupted_full_group(monkeypatch, negative):
    companies = [_public_company("A", domain="same.example"), _public_company("Alias A", domain="same.example")]
    for company in companies:
        company["company_linkedin"] = "https://linkedin.com/company/same"
    document = _input(companies)
    attempted = []

    async def judge(**kwargs):
        name = kwargs["company"].company_name
        attempted.append(name)
        row = _positive_breakdown(name, "same.example", "same")
        if negative:
            row.update(final_score=0.0, intent_signals_detail=[], failure_reason="No verified primary intent")
        return row

    monkeypatch.setattr(lead_scorer, "score_company_competition_intent", judge)
    full = scorer_entrypoint.score_input(document)
    assert attempted == ["A", "Alias A"]
    assert full["breakdowns"][1]["duplicate_company"] is True
    document["prior_terminal_breakdowns"] = [full["breakdowns"][0]]
    attempted.clear()
    resumed = scorer_entrypoint.score_input(document)
    assert attempted == ["Alias A"]
    assert resumed == full


@pytest.mark.parametrize("change", ["index", "duplicate_index", "retryable", "unproven", "quality", "bucket", "duplicate_without_ancestor", "positive_alias_collision", "unproven_fit_negative"])
def test_shared_sparse_validator_rejects_untrusted_or_unresolved_rows(change):
    document = _input()
    rows = [_row()]
    if change == "index":
        rows[0]["company_index"] = 4
    elif change == "duplicate_index":
        rows.append(copy.deepcopy(rows[0]))
    elif change == "retryable":
        rows[0]["verifier_gate_receipts"] = [{"gate": "company_fit", "decision": "unavailable"}]
    elif change == "unproven":
        rows[0]["intent_signals_detail"][0]["judge_verdict"]["decision"] = "unproven"
    elif change == "unproven_fit_negative":
        rows[0].update(final_score=0.0, company_qualified=False,
                       verifier_gate_receipts=[{"gate": "company_fit", "decision": "UNPROVEN"}])
    elif change == "quality":
        document["scorer_policy"]["company_quality_policy"] = "company_quality_v1"
    elif change == "bucket":
        document["companies"][0]["employee_count"] = "1001-5000"
    elif change == "duplicate_without_ancestor":
        rows[0].update(duplicate_company=True, company_qualified=False, final_score=0.0)
    else:
        rows.append(dict(copy.deepcopy(rows[0]), company_index=1))
    with pytest.raises(scoring.ScoringError):
        scoring.validate_terminal_breakdowns_for_work_item(rows, icp=document["icp"], companies=document["companies"], policy=document["scorer_policy"])


def _lease(document):
    leased = scoring_lease()
    leased.update(scorer_policy=document["scorer_policy"], integrity_policy="arena_integrity_v1",
                  icp=document["icp"], scored_output={"companies": document["companies"]})
    return leased


@pytest.mark.parametrize("change", ["valid", "execute", "wrong_run", "accepted", "retryable", "mixed", "oversized", "nested"])
def test_runner_accepts_only_bounded_failed_trusted_lease_checkpoints(change):
    document = _input()
    leased = _lease(document)
    output = scoring.build_scoring_failure("r1", "judge_error", terminal_breakdowns=[_row()])
    if change == "execute":
        leased["kind"] = "execute"
    elif change == "wrong_run":
        output["scored_run_id"] = "other"
    elif change == "accepted":
        output = scoring.build_scoring_output("r1", [_row()])
        output["terminal_breakdowns"] = [_row()]
    elif change == "retryable":
        output["terminal_breakdowns"][0]["failure_reason"] = "provider timeout"
    elif change == "mixed":
        output["company_judgments"] = []
    elif change == "oversized":
        output["terminal_breakdowns"] *= 6
    elif change == "nested":
        output["terminal_breakdowns"][0]["nested"] = [[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[0]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
    assert (runner._partial_company_judgment_output(output, leased) is not None) == (change == "valid")


def test_runner_timeout_keeps_terminal_rows_and_trusted_resume_input(tmp_path):
    document = _input()
    leased = _lease(document)
    leased["prior_terminal_breakdowns"] = [_row()]
    failed = scoring.build_scoring_failure("r1", "judge_error", terminal_breakdowns=[_row()])

    class TimedOut(BridgingRuntime):
        def run_icp(self, spec, **kwargs):
            staged = json.loads((spec.input_dir / runtime.INPUT_FILE_NAME).read_text())
            assert staged["prior_terminal_breakdowns"] == [_row()]
            assert staged["companies"] == document["companies"]
            return replace(super().run_icp(spec, **kwargs), timed_out=True)

    api = LoggingApi()
    api.leases = [leased]
    run_model(tmp_path, api, TimedOut(output=failed))
    body = api.completions[0]["body"]
    assert body["result"]["terminal_status"] == "judge_timeout"
    assert body["output"]["failure"] == "judge_timeout"
    assert body["output"]["terminal_breakdowns"] == [_row()]
    assert "breakdowns" not in body["output"]


@pytest.mark.parametrize("contacts", [False, True])
def test_earlier_recovered_alias_reprojects_retained_later_positive_without_paid_retry(monkeypatch, contacts):
    document = _input([_public_company(name, domain=name.lower() + ".example") for name in ("A", "B", "C")], contacts=contacts)
    contact_attempted = _install_verified_contacts(monkeypatch) if contacts else []
    phase = {"value": 0}
    attempted = []

    async def judge(**kwargs):
        name = kwargs["company"].company_name
        attempted.append(name)
        if name == "C" and phase["value"] < 2:
            raise RuntimeError("provider timeout")
        row = _positive_breakdown("A" if name in {"A", "B"} else "C",
                                  "same.example" if name in {"A", "B"} else "c.example",
                                  "same" if name in {"A", "B"} else "c")
        if name == "A" and phase["value"] == 0:
            row.update(final_score=0.0, intent_signals_detail=[], failure_reason="provider error",
                       verifier_gate_receipts=[{"gate": "company_fit", "decision": "unavailable"}])
        return row

    monkeypatch.setattr(lead_scorer, "score_company_competition_intent", judge)
    first = scorer_entrypoint.score_input(document)
    assert [row["company_index"] for row in first["terminal_breakdowns"]] == [1]
    document["prior_terminal_breakdowns"] = first["terminal_breakdowns"]
    phase["value"] = 1
    attempted.clear()
    contact_attempted.clear()
    second = scorer_entrypoint.score_input(document)
    assert "B" not in attempted
    assert [row["company_index"] for row in second["terminal_breakdowns"]] == [0, 1]
    assert second["terminal_breakdowns"][1]["duplicate_company"] is True
    assert second["terminal_breakdowns"][1]["final_score"] == 0.0
    if contacts:
        assert contact_attempted == ["A"]
        duplicate = second["terminal_breakdowns"][1]
        assert duplicate["contact_qualified"] is False
        assert duplicate["contact_verification"]["decision"] == "not_evaluated"
        contact_policy.validate_contact_breakdown(duplicate)
    document["prior_terminal_breakdowns"] = second["terminal_breakdowns"]
    phase["value"] = 2
    attempted.clear()
    resumed = scorer_entrypoint.score_input(document)
    assert attempted == ["C"]
    del document["prior_terminal_breakdowns"]
    full = scorer_entrypoint.score_input(document)
    fields = ("final_score", "company_qualified", "duplicate_company", "company_identity_key")
    assert [[row[field] for field in fields] for row in resumed["breakdowns"]] == [[row[field] for field in fields] for row in full["breakdowns"]]
    scoring.validate_breakdowns_for_item(
        resumed["breakdowns"], icp=document["icp"], companies=document["companies"],
        integrity_policy=True, contacts_required=contacts,
    )


def test_resumed_nonduplicate_keeps_verified_contact_without_rechecking(monkeypatch):
    document = _input(contacts=True)
    attempted = []
    contact_attempted = _install_verified_contacts(monkeypatch)
    ready = {"A"}

    async def judge(**kwargs):
        name = kwargs["company"].company_name
        attempted.append(name)
        if name not in ready:
            raise RuntimeError("provider timeout")
        return _positive_breakdown(name, name.lower() + ".example")

    monkeypatch.setattr(lead_scorer, "score_company_competition_intent", judge)
    failed = scorer_entrypoint.score_input(document)
    prior = copy.deepcopy(failed["terminal_breakdowns"])
    assert prior[0]["contact_qualified"] is True
    document["prior_terminal_breakdowns"] = prior
    ready.add("B")
    attempted.clear()
    contact_attempted.clear()
    resumed = scorer_entrypoint.score_input(document)
    assert attempted == ["B"]
    assert contact_attempted == ["B"]
    assert resumed["breakdowns"][0] == prior[0]
    scoring.validate_breakdowns_for_item(
        resumed["breakdowns"], icp=document["icp"], companies=document["companies"],
        integrity_policy=True, contacts_required=True,
    )


@pytest.mark.parametrize("change", ["valid", "stale", "cleanup", "worker_cleanup"])
def test_signaled_legacy_checkpoint_preserves_cleanup_and_completion_boundaries(tmp_path, monkeypatch, change):
    import signal
    leased = _lease(_input())
    failed = scoring.build_scoring_failure("r1", "judge_error", terminal_breakdowns=[_row()])

    class Interrupted:
        def run_icp(self, spec, **kwargs):
            error = runtime.RuntimeHostError(reason="sandbox_launcher_signaled", launch_exit_code=-signal.SIGINT,
                                             checkpoint_bytes=json.dumps(failed).encode())
            if change == "cleanup":
                raise runtime.SandboxCleanupError("cleanup incomplete", result=None) from error
            raise error

    class Api(LoggingApi):
        def complete(self, envelope):
            result = super().complete(envelope)
            return {"status": "stale"} if change == "stale" else result

    if change == "worker_cleanup":
        original_stop = runner.WorkerSocketServer.stop

        def fail_stop(server):
            original_stop(server)
            raise RuntimeError("worker cleanup incomplete")

        monkeypatch.setattr(runner.WorkerSocketServer, "stop", fail_stop)
    api = Api()
    api.leases = [leased]
    worker = run_model(tmp_path, api, Interrupted())
    if change in {"valid", "stale"}:
        assert worker.abandoned == 0 and len(api.completions) == 1
        assert api.completions[0]["body"]["output"]["terminal_breakdowns"] == [_row()]
        if change == "stale":
            assert worker.completed[0]["result"]["status"] == "stale"
    else:
        assert worker.abandoned == 1 and api.completions == []
