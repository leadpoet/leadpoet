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
                    content = payload if isinstance(payload, str) else json.dumps(payload)
                    return {"choices": [{"message": {"content": content}}]}

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


# Live round arena-2026-09-22-verifierf2626, score ICP 6, attempt 1,
# settlement 851810: a valid source-bound finding arrived inside this wrapper.
LIVE_ROLE_FINDING = {
    "status": "VERIFIED",
    "target_role": "Product Manager",
    "evidence_quote": (
        "I own all new module development, and I've led deployments at our largest "
        "customers, including Cleveland Clinic and Mayo Clinic."
    ),
    "reason": (
        "The duties explicitly state ownership of 'all new module development', "
        "which is a core responsibility of a Product Manager. The title 'Product "
        "Lead' is consistent with a Product Manager function, and the duties "
        "demonstrate product ownership and leadership, aligning with the 'Product "
        "Manager' target role."
    ),
}


@pytest.mark.parametrize("language", ["json", "JSON", ""])
def test_live_fenced_role_finding_keeps_bound_evidence(monkeypatch, language):
    content = f"```{language}\n{json.dumps(LIVE_ROLE_FINDING, indent=2)}\n```"
    calls = _transport(monkeypatch, content)
    result = asyncio.run(_classify_contact_role("Product Lead", TARGETS, "", BAYESIAN_DUTIES))
    assert result["match"] is True
    assert result["evidence_review"] == LIVE_ROLE_FINDING
    assert len(calls) == 2


@pytest.mark.parametrize("change", [
    {"evidence_quote": "Invented ownership evidence"},
    {"target_role": "Chief Product Officer"},
    {"status": "CONTRADICTED", "target_role": ""},
    {"status": "UNPROVEN"},
])
def test_fence_does_not_weaken_evidence_binding(monkeypatch, change):
    content = "```json\n" + json.dumps({**LIVE_ROLE_FINDING, **change}) + "\n```"
    _transport(monkeypatch, content)
    result = asyncio.run(_classify_contact_role("Product Lead", TARGETS, "", BAYESIAN_DUTIES))
    assert result["match"] is False


@pytest.mark.parametrize("content", [
    "```json\n{}\n``` trailing prose",
    '```json\n{"status": "VERIFIED"}\n{}\n```',
    '```json\n{"status": "VERIFIED"\n```',
    "```json\n[]\n```",
])
def test_malformed_fenced_review_remains_unavailable(monkeypatch, content):
    _transport(monkeypatch, content)
    with pytest.raises(RuntimeError, match="evidence review unavailable"):
        asyncio.run(_classify_contact_role("Product Lead", TARGETS, "", BAYESIAN_DUTIES))
