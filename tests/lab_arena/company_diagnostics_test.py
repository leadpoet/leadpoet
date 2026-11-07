"""Public company outcomes retain skipped checks and the disclosure boundary."""

import copy
import json
import threading
from types import SimpleNamespace

import pytest

from lab_arena import scoring
from lab_arena.public_dashboard import company_diagnostic
from lab_arena.service import ArenaService, ServiceError


def _breakdown():
    return {
        "company_index": 0, "company_qualified": False,
        "verifier_gate_receipts": [{
            "gate": "company_fit", "company_fit_stage_required": True,
            "company_fit_dimensions": dict.fromkeys(
                ("identity", "industry", "employee_size", "geography", "stage"), "match"
            ), "required_attribute_decision": "match",
        }],
        "contact_verification": {
            "decision": "not_evaluated", "reason": "company_not_qualified", "subchecks": {},
        },
    }


def test_missing_contact_does_not_claim_email_or_contact_check_ran():
    breakdown = _breakdown()
    breakdown["verifier_gate_receipts"][0]["company_fit_dimensions"]["stage"] = "unavailable"
    result = company_diagnostic(breakdown, {"company_name": "Example", "contact": None}, icp_position=2)
    assert result["missing_contact"] is True
    assert result["qualified"] is False
    assert result["checks"]["stage"] == "unavailable"
    assert result["checks"]["contact"] == result["checks"]["email"] == "not_evaluated"


def test_role_failure_skips_email_and_redacts_all_payloads():
    breakdown = _breakdown()
    breakdown["failure_reason"] = "private verifier prose"
    breakdown["contact_verification"] = {
        "decision": "mismatch", "reason": "private provider prose",
        "subchecks": {"role": {"status": "fail", "reason": "private title"}},
        "email": "private@example.test", "provider_response": {"secret": "private key"},
    }
    result = company_diagnostic(breakdown, {"company_name": "Example", "contact": {"email": "private@example.test"}}, icp_position=2)
    assert result["contact_failure"] == "role"
    assert result["checks"]["contact"] == "failed"
    assert result["checks"]["email"] == "not_evaluated"
    assert "private" not in json.dumps(result)


@pytest.mark.parametrize("status, expected", [("pass", "passed"), ("fail", "failed"), ("unavailable", "unavailable")])
def test_email_status_is_derived_from_actual_email_subcheck(status, expected):
    breakdown = _breakdown()
    breakdown["email_status"] = "valid"  # A summary alone is not proof that a check ran.
    breakdown["contact_verification"]["subchecks"] = {"email_verification": {"status": status}}
    result = company_diagnostic(breakdown, {"company_name": "Example", "contact": {}}, icp_position=1)
    assert result["checks"]["email"] == expected


def test_unresolved_intent_is_not_reported_as_factual_mismatch():
    breakdown = _breakdown()
    breakdown["intent_signals_detail"] = [{
        "matched_icp_signal": 0, "after_decay": 0,
        "judge_verdict": {"decision": "rejected_three_stage", "pipeline_decision": "review"},
    }]
    result = company_diagnostic(breakdown, {"company_name": "Example", "contact": None}, icp_position=0)
    assert result["checks"]["intent"] == "unavailable"


def _install_completed_scores_cache(service):
    """Give a bare ``ArenaService`` the completed-scores cache its reads need.

    These stubs are built with ``object.__new__``, so nothing ``__init__``
    installs is present. The published-results path reads the cache, and
    without it the test fails on a missing attribute rather than on the
    disclosure behaviour it is actually asserting.
    """
    service._completed_scores_lock = threading.Lock()
    service._completed_scores_cache = {}
    service._completed_scores_refresh_slots = threading.BoundedSemaphore(2)


def _service(monkeypatch, *, disclosed=True):
    import lab_arena.service as module
    row = {"round_id": "round", "status": "published", "configuration_doc": {"contact_policy": "contacts_v1", "scorer_policy": {}},
           "publication_doc": {"participants": [{"submission_id": "miner"}]}}
    runs = [
        {"run_id": f"run-{i}", "icp_position": i, "stage": 1,
         "round_id": "round", "kind": "execute", "submission_id": "miner",
         "output_ref": f"output-{i}", "status": "accepted"}
        for i in (0, 1)
    ]
    service = object.__new__(ArenaService)
    _install_completed_scores_cache(service)
    service._round = lambda _: row
    service._public_icp_disclosure = lambda _: {"public_positions": [0]} if disclosed else None
    judgments = [{**run, "run_id": run["run_id"] + ":score", "kind": "score",
                  "scored_run_id": run["run_id"]} for run in runs]
    service._store = SimpleNamespace(list_runs=lambda *a, **k: copy.deepcopy(
        judgments if k.get("kind") == "score" else runs))
    reads = []
    def get_output(ref, _limit):
        reads.append(ref)
        assert ref == "output-0", "private ICP output must not be fetched"
        return b'{"companies":[{"company_name":"Disclosed company","contact":null}]}'
    service._objects = SimpleNamespace(get_bounded=get_output)
    monkeypatch.setattr(module, "validate_output_document", lambda value: value)
    service._public_scoring_attribution = lambda *a, **k: {}
    service._benchmark_icps_from_row = lambda *a, **k: [{}]
    service._verified_breakdowns = lambda *a, **k: [_breakdown()]
    return service, reads, row


def test_public_api_projects_only_disclosed_accepted_company_checks(monkeypatch):
    service, reads, _ = _service(monkeypatch)
    result = service.public_results("round", "miner")
    assert reads == ["output-0"]
    assert len(result["company_diagnostics"]) == 1
    assert result["company_diagnostics"][0]["company_name"] == "Disclosed company"
    assert result["company_diagnostics"][0]["icp_position"] == 0


def test_pending_disclosure_never_fetches_company_or_verifier_evidence(monkeypatch):
    service, reads, _ = _service(monkeypatch, disclosed=False)
    service._benchmark_icps_from_row = lambda *a, **k: pytest.fail("private ICPs accessed")
    service._verified_breakdowns = lambda *a, **k: pytest.fail("private judgments accessed")
    assert service.public_results("round", "miner")["company_diagnostics"] == []
    assert reads == []


def test_invalid_stored_judgment_cannot_produce_public_diagnostics(monkeypatch):
    service, _, _ = _service(monkeypatch)
    def invalid(*a, **k):
        raise scoring.ScoringError("private evidence mismatch")
    service._verified_breakdowns = invalid
    with pytest.raises(ServiceError, match="public_contact_verification_unavailable"):
        service.public_results("round", "miner")


def test_unpublished_results_remain_private(monkeypatch):
    service, reads, row = _service(monkeypatch)
    row["status"] = "stage1_scoring"
    with pytest.raises(ServiceError, match="results_not_public"):
        service.public_results("round", "miner")
    assert reads == []


def test_company_only_diagnostics_have_no_contact_or_email_checks(monkeypatch):
    service, reads, row = _service(monkeypatch)
    row["configuration_doc"].pop("contact_policy")
    row["configuration_doc"]["integrity_policy"] = "arena_integrity_v1"
    result = service.public_results("round", "miner")
    assert reads == ["output-0"]
    assert "contact_verifications" not in result
    diagnostic = result["company_diagnostics"][0]
    assert "missing_contact" not in diagnostic
    assert "contact_failure" not in diagnostic
    assert "contact" not in diagnostic["checks"]
    assert "email" not in diagnostic["checks"]
    assert diagnostic["checks"]["identity"] == "passed"
    assert diagnostic["checks"]["intent"] == "not_evaluated"


def _diagnose(breakdown):
    return company_diagnostic(
        breakdown, {"company_name": "Example"}, icp_position=0,
        contacts_required=False,
    )


def _reason_codes(result, check):
    return [item["code"] for item in result["check_reasons"].get(check, [])]


def test_saved_fit_evidence_explains_claim_and_observed_mismatches():
    breakdown = _breakdown()
    fit = breakdown["verifier_gate_receipts"][0]
    fit["company_fit_dimensions"]["employee_size"] = "mismatch"
    fit["company_fit_dimensions"]["geography"] = "mismatch"
    fit["dimension_evidence"] = {
        "employee_size": {"submitted_decision": "mismatch", "observed_decision": "match"},
        "geography": {"submitted_decision": "match", "observed_decision": "mismatch"},
    }
    result = _diagnose(breakdown)
    assert _reason_codes(result, "employee_size") == ["submitted_value_mismatch"]
    assert _reason_codes(result, "geography") == ["verified_value_mismatch"]
    assert "submitted" in result["check_reasons"]["employee_size"][0]["message"]


def test_unavailable_check_does_not_invent_provider_or_factual_failure():
    breakdown = _breakdown()
    breakdown["verifier_gate_receipts"][0]["company_fit_dimensions"]["industry"] = "unavailable"
    result = _diagnose(breakdown)
    assert _reason_codes(result, "industry") == ["verification_unavailable"]
    assert "provider" not in result["check_reasons"]["industry"][0]["message"].lower()
    assert "identity" not in result["check_reasons"]  # Passed checks need no repair.


def test_paragraph_reports_all_failed_native_checks_without_private_text():
    breakdown = _breakdown()
    breakdown["verifier_gate_receipts"].append({
        "gate": "intent_details", "decision": "mismatch",
        "checks": {"facts_supported": False, "connects_icp": False, "natural_paragraph": True,
                   "unknown_private_check": False},
        "failed_factual_units": [{"text": "private paragraph", "quote": "private source"}],
    })
    result = _diagnose(breakdown)
    assert _reason_codes(result, "intent_details") == ["paragraph_facts_unsupported", "paragraph_icp_connection_missing"]
    assert "private" not in json.dumps(result)


def test_bonus_verifier_error_does_not_mask_terminal_primary_rejection():
    breakdown = _breakdown()
    breakdown["intent_signals_detail"] = [
        {"matched_icp_signal": 0, "after_decay": 0,
         "judge_verdict": {"decision": "rejected_freshness"}},
        {"matched_icp_signal": 1, "after_decay": 0,
         "judge_verdict": {"decision": "rejected_verifier_error", "error_class": "private provider error"}},
    ]
    result = _diagnose(breakdown)
    assert result["checks"]["intent"] == "failed"
    assert _reason_codes(result, "intent") == ["intent_outside_recency"]
    assert "private" not in json.dumps(result)


def test_bonus_only_cannot_claim_primary_was_verified_or_had_a_provider_error():
    breakdown = _breakdown()
    breakdown["intent_signals_detail"] = [{
        "matched_icp_signal": 1, "after_decay": 0,
        "judge_verdict": {"decision": "rejected_verifier_error", "error_class": "TimeoutError"},
    }]
    result = _diagnose(breakdown)
    assert result["checks"]["intent"] == "failed"
    assert _reason_codes(result, "intent") == ["primary_intent_missing"]


@pytest.mark.parametrize("verdict,code", [
    ({"decision": "rejected_verifier_error"}, "verifier_error"),
    ({"decision": "rejected_three_stage", "pipeline_decision": "review"}, "intent_evidence_unresolved"),
    ({"decision": "rejected_three_stage", "pipeline_decision": "reject"}, "primary_intent_not_verified"),
])
def test_primary_intent_reason_uses_saved_verdict_only(verdict, code):
    breakdown = _breakdown()
    breakdown["intent_signals_detail"] = [{
        "matched_icp_signal": 0, "after_decay": 0, "judge_verdict": verdict,
    }]
    assert _reason_codes(_diagnose(breakdown), "intent") == [code]


def test_diagnostic_reasons_do_not_mutate_signed_or_private_evidence():
    breakdown = _breakdown()
    breakdown["failure_reason"] = "private provider prose"
    fit = breakdown["verifier_gate_receipts"][0]
    fit["company_fit_dimensions"]["stage"] = "mismatch"
    fit["dimension_evidence"] = {"stage": {"reason": "private value", "url": "https://private.example"}}
    original = copy.deepcopy(breakdown)
    result = _diagnose(breakdown)
    assert breakdown == original
    assert "private" not in json.dumps(result)
    assert _reason_codes(result, "stage") == ["requirement_mismatch"]


def test_unknown_historical_intent_binding_does_not_claim_primary_was_missing():
    breakdown = _breakdown()
    breakdown["intent_signals_detail"] = [{"after_decay": 0, "judge_verdict": {"decision": "rejected_three_stage"}}]
    assert _reason_codes(_diagnose(breakdown), "intent") == ["primary_intent_not_verified"]


def test_unavailable_intent_gate_without_signals_does_not_blame_miner():
    breakdown = _breakdown()
    breakdown["verifier_gate_receipts"].append({"gate": "intent_verification", "decision": "unavailable"})
    result = _diagnose(breakdown)
    assert result["checks"]["intent"] == "unavailable"
    assert _reason_codes(result, "intent") == ["verification_unavailable"]


def test_multiple_recorded_fit_causes_remain_visible():
    breakdown = _breakdown()
    fit = breakdown["verifier_gate_receipts"][0]
    fit["company_fit_dimensions"]["geography"] = "mismatch"
    fit["dimension_evidence"] = {"geography": {"submitted_decision": "mismatch", "observed_decision": "mismatch"}}
    assert _reason_codes(_diagnose(breakdown), "geography") == ["submitted_value_mismatch", "verified_value_mismatch"]


def test_verified_primary_ignores_bonus_failure_and_needs_no_repair():
    breakdown = _breakdown()
    breakdown["intent_signals_detail"] = [
        {"matched_icp_signal": 0, "after_decay": 60, "judge_verdict": {"decision": "verified", "pipeline_decision": "approve"}},
        {"matched_icp_signal": 1, "after_decay": 0, "judge_verdict": {"decision": "rejected_verifier_error"}},
    ]
    result = _diagnose(breakdown)
    assert result["checks"]["intent"] == "passed"
    assert _reason_codes(result, "intent") == []


@pytest.mark.parametrize("index", [None, -1, "0", False])
@pytest.mark.parametrize("bonus", [False, True])
@pytest.mark.parametrize("verdict", [
    {"decision": "rejected_three_stage", "pipeline_decision": "review"},
    {"decision": "rejected_verifier_error", "error_class": "TimeoutError"},
])
def test_unknown_intent_binding_retains_unresolved_status(index, bonus, verdict):
    breakdown = _breakdown()
    breakdown["intent_signals_detail"] = [{
        "matched_icp_signal": index, "after_decay": 0, "judge_verdict": verdict,
    }]
    if bonus:
        breakdown["intent_signals_detail"].append({
            "matched_icp_signal": 1, "after_decay": 0,
            "judge_verdict": {"decision": "rejected_three_stage", "pipeline_decision": "reject"},
        })
    result = _diagnose(breakdown)
    assert result["checks"]["intent"] == "unavailable"
    assert _reason_codes(result, "intent") == ["verification_unavailable"]


def test_bound_primary_rejection_does_not_hide_unbound_error():
    breakdown = _breakdown()
    breakdown["intent_signals_detail"] = [
        {"matched_icp_signal": 0, "after_decay": 0,
         "judge_verdict": {"decision": "rejected_three_stage", "pipeline_decision": "reject"}},
        {"after_decay": 0, "judge_verdict": {"decision": "rejected_verifier_error"}},
    ]
    result = _diagnose(breakdown)
    assert result["checks"]["intent"] == "unavailable"
    assert _reason_codes(result, "intent") == ["primary_intent_not_verified", "verification_unavailable"]
