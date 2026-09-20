"""Public company outcomes retain skipped checks and the disclosure boundary."""

import copy
import json
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


def _service(monkeypatch, *, disclosed=True):
    import lab_arena.service as module
    row = {"status": "published", "configuration_doc": {"contact_policy": "contacts_v1", "scorer_policy": {}},
           "publication_doc": {"participants": [{"submission_id": "miner"}]}}
    runs = [
        {"run_id": f"run-{i}", "icp_position": i, "stage": 1,
         "submission_id": "miner", "output_ref": f"output-{i}", "status": "accepted"}
        for i in (0, 1)
    ]
    service = object.__new__(ArenaService)
    service._round = lambda _: row
    service._public_icp_disclosure = lambda _: {"public_positions": [0]} if disclosed else None
    service._store = SimpleNamespace(list_runs=lambda *a, **k: copy.deepcopy(runs))
    reads = []
    def get_output(ref, _limit):
        reads.append(ref)
        assert ref == "output-0", "private ICP output must not be fetched"
        return b'{"companies":[{"company_name":"Disclosed company","contact":null}]}'
    service._objects = SimpleNamespace(get_bounded=get_output)
    monkeypatch.setattr(module, "validate_output_document", lambda value: value)
    service._public_scoring_attribution = lambda *a: {}
    service._scoring_outputs = lambda *a: {"run-0": {"status": "accepted"}}
    service.evaluation_icps = lambda _: [{}]
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
    service.evaluation_icps = lambda _: pytest.fail("private ICPs accessed")
    service._scoring_outputs = lambda *a: pytest.fail("private judgments accessed")
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
