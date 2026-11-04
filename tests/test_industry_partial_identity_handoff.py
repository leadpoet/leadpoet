"""Keep already verified own-domain citations without inventing a LinkedIn ID."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from qualification.scoring import intent_details
from tests.test_intent_details_industry_support_handoff import _inputs, _industry_sources


CAPTURE = json.loads((Path(__file__).parent / "fixtures" /
    "intent_details_partial_identity_industry.json").read_text())


def test_real_partial_identity_keeps_every_verified_first_party_citation(monkeypatch):
    monkeypatch.setenv("ARENA_INDUSTRY_EVIDENCE_MODEL", "openai/gpt-6.1-sol")
    inputs = _inputs(CAPTURE)
    before = deepcopy(inputs[3])
    assert before["dimension_evidence"]["identity"]["web_identity_receipt"]["observed_linkedin_slug"] == ""
    sources = _industry_sources(intent_details.review_evidence(*inputs))
    for pair in [CAPTURE["industry_primary"], *CAPTURE["industry_support"]]:
        assert any(s["source_url"] == pair["url"] and pair["quote"] in s["admitted_text"]
                   for s in sources)
    assert inputs[3] == before


@pytest.mark.parametrize("fault", ["foreign_domain", "foreign_primary", "domain_conflict", "no_submitted_domain", "no_observed_name",
                                  "no_observed_domain", "unmatched_identity", "old_policy"])
def test_missing_linkedin_never_grants_new_source_authority(monkeypatch, fault):
    monkeypatch.setenv("ARENA_INDUSTRY_EVIDENCE_MODEL", "openai/gpt-6.1-sol")
    inputs = _inputs(CAPTURE)
    receipt = inputs[3]
    identity = receipt["dimension_evidence"]["identity"]["web_identity_receipt"]
    support = receipt["dimension_evidence"]["industry"]["web_evidence"]["supporting_evidence"]
    if fault == "foreign_domain":
        support[0]["url"] = "https://ada.cx.attacker.example/capability"
    elif fault == "foreign_primary":
        receipt["dimension_evidence"]["industry"]["web_evidence"]["url"] = "https://other.example/capability"
    elif fault == "domain_conflict":
        identity["submitted_domain"] = "unrelated.example"
    elif fault == "no_submitted_domain":
        identity["submitted_domain"] = ""
    elif fault == "no_observed_name":
        identity["observed_name"] = ""
    elif fault == "no_observed_domain":
        identity["observed_domain"] = ""
    elif fault == "unmatched_identity":
        identity["decision"] = "unavailable"
    else:
        monkeypatch.delenv("ARENA_INDUSTRY_EVIDENCE_MODEL")
    sources = _industry_sources(intent_details.review_evidence(*inputs))
    assert not any(s["source_url"] == support[0]["url"] and
                   support[0]["quote"] in s["admitted_text"] for s in sources)
    if fault == "foreign_domain":
        for pair in support[1:]:
            assert any(s["source_url"] == pair["url"] and pair["quote"] in s["admitted_text"]
                       for s in sources)
