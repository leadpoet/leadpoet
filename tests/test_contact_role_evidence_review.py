"""Bounded evidence review of saved September 22 contact-role disputes."""

import asyncio
import json

import pytest

from qualification.scoring import role_batch_check
from qualification.scoring.competition import _classify_contact_role
from qualification.scoring.contact_verification import _role_matches


# Exact role/duties from the saved Bayesian Health contact-verification receipt.
BAYESIAN_DUTIES = (
    "Bayesian develops clinical AI agents for inpatient care. Our platform detects "
    "signs of sepsis and other needs using multimodal models, then enables clinicians "
    "to act quickly within the EHR. In April 2026, we became the first continuous AI "
    "sepsis monitor to gain FDA clearance. I own all new module development, and "
    "I've led deployments at our largest customers, including Cleveland Clinic "
    "and Mayo Clinic."
)
PRODUCT_QUOTE = "I own all new module development"
TARGETS = ["Clinical Operations Manager", "Healthcare IT Manager", "Product Manager"]


def _transport(monkeypatch, finding, *, initial_match=False, status_code=200):
    calls = []

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def post(self, url, **kwargs):
            calls.append(kwargs["json"])
            payload = ([{"id": 1, "match": initial_match, "reason": "Initial judgment"}]
                       if len(calls) == 1 else finding)

            class Response:
                def json(self):
                    return {"choices": [{"message": {"content": json.dumps(payload)}}]}

            response = Response()
            response.status_code = status_code if len(calls) == 2 else 200
            return response

    monkeypatch.setenv("OPENROUTER_KEY", "test-broker-key")
    monkeypatch.setattr(role_batch_check.httpx, "AsyncClient", Client)
    return calls


def _finding(**changes):
    return {"status": "VERIFIED", "target_role": "Product Manager",
            "evidence_quote": PRODUCT_QUOTE,
            "reason": "Owns product modules and deployments at the requested level.", **changes}


def test_saved_bayesian_role_can_recover_only_with_bound_current_duties(monkeypatch):
    calls = _transport(monkeypatch, _finding())
    diagnostics = {}
    matched = asyncio.run(_role_matches(
        "Product Lead", TARGETS, "", _classify_contact_role,
        BAYESIAN_DUTIES, diagnostics,
    ))
    assert matched is True
    assert len(calls) == 2
    assert diagnostics["evidence_review"]["status"] == "VERIFIED"
    assert diagnostics["evidence_review"]["evidence_quote"] == PRODUCT_QUOTE
    prompt = calls[1]["messages"][1]["content"]
    assert BAYESIAN_DUTIES in prompt
    assert "Do not infer a management level from the word Lead alone" in prompt


@pytest.mark.parametrize("change", [
    {"evidence_quote": "Owns a billion dollar product portfolio"},
    {"evidence_quote": ""},
    {"target_role": "Chief Product Officer"},
    {"status": "UNPROVEN"},
    {"status": "CONTRADICTED", "target_role": ""},
])
def test_unbound_or_negative_finding_cannot_rescue_a_role(monkeypatch, change):
    calls = _transport(monkeypatch, _finding(**change))
    result = asyncio.run(_classify_contact_role("Product Lead", TARGETS, "", BAYESIAN_DUTIES))
    assert result["match"] is False
    assert len(calls) == 2


@pytest.mark.parametrize("status_code,finding", [
    (503, _finding()), (200, []), (200, {"status": "accepted"}),
])
def test_review_failure_is_unavailable_not_fabricated_negative(monkeypatch, status_code, finding):
    _transport(monkeypatch, finding, status_code=status_code)
    with pytest.raises(RuntimeError, match="evidence review unavailable"):
        asyncio.run(_classify_contact_role("Product Lead", TARGETS, "", BAYESIAN_DUTIES))


def test_saved_buywander_without_duties_has_no_speculative_second_look(monkeypatch):
    calls = _transport(monkeypatch, _finding())
    result = asyncio.run(_classify_contact_role(
        "Front of House Manager (Operations & Growth)",
        ["Ecommerce Manager", "Merchandising Manager", "Growth Operations Manager"], "", "",
    ))
    assert result is False
    assert len(calls) == 1


@pytest.mark.parametrize("role,seniority,expected", [
    ("Security Engineer", "", True),
    ("Product Lead", "Director", False),
    ("Executive Assistant to Product Manager", "", False),
])
def test_deterministic_role_controls_never_invoke_review(monkeypatch, role, seniority, expected):
    calls = _transport(monkeypatch, _finding())
    result = asyncio.run(_role_matches(
        role, ["Security Engineer", "Product Manager"], seniority,
        _classify_contact_role, BAYESIAN_DUTIES,
    ))
    assert result is expected
    assert calls == []


def test_initial_semantic_match_does_not_spend_another_call(monkeypatch):
    calls = _transport(monkeypatch, _finding(), initial_match=True)
    assert asyncio.run(_classify_contact_role("Product Lead", TARGETS, "", BAYESIAN_DUTIES)) is True
    assert len(calls) == 1
