"""Real Wiz evidence regression; all other named transaction examples are synthetic."""

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from qualification.scoring import company_evidence_investigator as investigator
from qualification.scoring import lead_scorer
from lab_arena.operations import validate_operation_request


# Google Cloud, March 11, 2026. Source and pre-prediction truth packet:
# /Users/pranav/Downloads/Arena_Verifier_Accuracy_20261103/
# stage-threshold-independent-review/frozen-source-packet.json
WIZ_URL = "https://cloud.google.com/blog/products/identity-security/google-completes-acquisition-of-wiz"
WIZ_QUOTE = (
    "I am proud to announce today that Google has completed its acquisition of "
    "Wiz, a leading cloud and AI security platform."
)


def test_real_wiz_completed_acquisition_binds_short_target():
    assert lead_scorer._acquired_stage_quote_supports_names(("Wiz",), WIZ_QUOTE)
    assert not lead_scorer._acquired_stage_quote_supports_names(("Google",), WIZ_QUOTE)


@pytest.mark.parametrize("name", ["Box", "Arm", "ADT", "Acme Analytics"])
@pytest.mark.parametrize("template", [
    "Parent completed its acquisition of {name}.",
    "{name} was acquired by Parent.",
    "{name} is a wholly owned subsidiary of Parent.",
])
def test_synthetic_exact_target_and_current_parent_names(name, template):
    quote = template.format(name=name)
    assert lead_scorer._acquired_stage_quote_supports_names((name,), quote)
    assert not lead_scorer._acquired_stage_quote_supports_names(("Parent",), quote)


@pytest.mark.parametrize("name", ["X", "AI"])
def test_synthetic_short_name_has_explicit_legal_name_grounding(name):
    quote = f"Parent completed its acquisition of {name} Corporation."
    assert lead_scorer._acquired_stage_quote_supports_names((name,), quote)
    quote = f'Parent completed its acquisition of Example Corporation ("{name}").'
    assert lead_scorer._acquired_stage_quote_supports_names((name,), quote)


@pytest.mark.parametrize("quote", [
    "Parent plans to complete its acquisition of Wiz.",
    "Parent has not completed its acquisition of Wiz.",
    "Parent expects to have completed its acquisition of Wiz in June.",
    "If Parent completed its acquisition of Wiz, ownership would change.",
    "Parent completed its acquisition of Wiz, subject to regulatory approval.",
    "Parent completed its acquisition of Wiz, but the deal was cancelled.",
    "Parent completed its acquisition of Wiz. Wiz later became independent.",
    "Parent completed its acquisition of Wiz assets.",
    "Parent completed its acquisition of assets of Wiz.",
    "Parent completed its acquisition of a minority stake in Wiz.",
    "Parent completed its acquisition of Wiz minority shares.",
    "Parent completed its acquisition of Wiz Systems.",
    "Parent completed its acquisition of Other (a partner of Wiz).",
    "Wiz completed its acquisition of Parent.",
    "Wiz, in an article reporting that Other was acquired by Parent.",
    "It was acquired by Parent.",
    "It is a wholly owned subsidiary of Parent.",
    'Parent completed its acquisition of Other ("us").',
])
def test_synthetic_short_name_preserves_adverse_controls(quote):
    assert not lead_scorer._acquired_stage_quote_supports_names(("Wiz",), quote)


def test_real_wiz_investigator_does_not_enter_false_citation_correction(monkeypatch):
    """The real investigator protocol with supplied model output, not model accuracy."""
    finding = dict(
        target="stage", status="CONTRADICTED", observed_value="Acquired",
        evidence_url=WIZ_URL, evidence_quote=WIZ_QUOTE,
        reason="Google completed its acquisition of Wiz after its venture rounds.",
    )
    requests = []

    async def fake_post(_session, _url, *, headers, payload):
        validate_operation_request("openrouter.chat", payload)
        requests.append(payload)
        return 200, {"choices": [{"message": {"tool_calls": [{
            "id": "wiz-stage", "type": "function",
            "function": {"name": "submit_findings", "arguments": json.dumps({
                "findings": [finding],
            })},
        }]}}]}

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("EXA_API_KEY", "test-key")
    monkeypatch.setattr(investigator, "_post_json", fake_post)
    monkeypatch.setattr(investigator, "_search_web", AsyncMock(return_value={"results": []}))
    monkeypatch.setattr(investigator, "_fetch_page", AsyncMock(side_effect=AssertionError("No live fetch")))
    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Wiz", "website": "https://www.wiz.io"},
        targets=("stage",), requested_stage="Series C+",
        positive_semantic_review=True,
        prior_observations={"submitted_source_urls": [WIZ_URL]},
        prefetched_pages={WIZ_URL: {"final_url": WIZ_URL, "text": WIZ_QUOTE}},
        verified_homepage_identity={"normalized_name": "wiz", "registrable_dns_domain": "wiz.io"},
    ))
    assert len(requests) == 1
    validated = result["_validated_stage_finding"]
    assert validated["status"] == "CONTRADICTED"
    assert validated["evidence_quote"] == WIZ_QUOTE
    projected = lead_scorer._project_investigator_stage({}, validated, icp_stage="series c+")
    retained = json.loads(json.dumps(projected))
    from tests.test_company_evidence_investigator import _company
    company = _company(name="Wiz", website="https://www.wiz.io")
    assert lead_scorer._decision_from_observed_stage(
        retained, "series c+", company=company, evidence_attributed=True,
    ) == "mismatch"


@pytest.mark.parametrize("name", ["wiz", "box", "arm", "adt"])
def test_synthetic_short_targets_do_not_depend_on_source_capitalization(name):
    assert lead_scorer._acquired_stage_quote_supports_names(
        (name,), f"Parent completed its acquisition of {name}.",
    )


def test_untrusted_short_pronoun_alias_cannot_replace_verified_company():
    """Synthetic exploit: prior observed 'it' is not verified company identity."""
    quote = "It was acquired by Parent."
    url = "https://parent.example/news/acquisition"
    finding = dict(target="stage", status="CONTRADICTED", observed_value="Acquired",
        evidence_url=url, evidence_quote=quote, reason="Synthetic adverse prior alias.")
    actual = investigator._validated_findings(
        {"findings": [finding]}, targets=("stage",), fetched_pages={url: quote},
        fetched_final_urls={url: url}, first_party_domains={"wiz.io"},
        identity_names={"wiz", "it"}, identity_anchor={
            "submitted_name": "Wiz", "verified_name": "Wiz",
            "submitted_domain": "wiz.io", "verified_domain": "wiz.io",
        }, requested_stage="series c+",
    )
    assert actual["stage"]["status"] == "UNPROVEN"


@pytest.mark.parametrize("reference", ["company", "we", "us", "it"])
@pytest.mark.parametrize("case", [str.lower, str.upper, str.title])
def test_anonymous_references_cannot_be_company_names_or_target_aliases(reference, case):
    token = case(reference)
    for quote in (
        f"{token} was acquired by Parent.",
        f"{token} is a wholly owned subsidiary of Parent.",
        f"Parent completed its acquisition of {token}.",
        f'Parent completed its acquisition of Example Corporation ("{token}").',
    ):
        assert not lead_scorer._acquired_stage_quote_supports_names(("Wiz", token), quote)


@pytest.mark.parametrize("quote", [
    "Parent completed its acquisition of Boxwood.",
    "Parent completed its acquisition of Sandbox.",
    "Parent completed its acquisition of Box assets.",
    "Parent completed its acquisition of Box minority shares.",
    "Box completed its acquisition of Parent.",
    "Box was not acquired by Parent.",
    "Box is not a subsidiary of Parent.",
    "Box was acquired by Parent. Box later relisted on Nasdaq.",
])
def test_synthetic_box_target_is_exact_and_current(quote):
    assert not lead_scorer._acquired_stage_quote_supports_names(("Box",), quote)


@pytest.mark.parametrize("source_name", ["Wiz", "wiz", "WIZ", "wIz", "Box", "box", "bOX", "Arm", "arm", "ARM", "ADT", "adt", "aDt"])
def test_synthetic_short_name_case_does_not_change_exact_binding(source_name):
    quote = f"Parent completed its acquisition of {source_name}."
    assert lead_scorer._acquired_stage_quote_supports_names((source_name.swapcase(),), quote)


def test_synthetic_short_current_parent_survives_saved_stage_json():
    from tests.test_company_evidence_investigator import _company

    url = "https://www.box.com/company"
    quote = "Box is a wholly owned subsidiary of Parent."
    raw = dict(target="stage", status="CONTRADICTED", observed_value="Acquired",
        evidence_url=url, evidence_quote=quote, reason="Synthetic current-parent control.")
    finding = investigator._validated_findings(
        {"findings": [raw]}, targets=("stage",), fetched_pages={url: quote},
        fetched_final_urls={url: url}, first_party_domains={"box.com"},
        identity_names={"box"}, requested_stage="series c+",
    )["stage"]
    assert finding["status"] == "CONTRADICTED"
    projected = lead_scorer._project_investigator_stage({}, finding, icp_stage="series c+")
    retained = json.loads(json.dumps(projected))
    company = _company(name="Box", website="https://www.box.com")
    assert lead_scorer._decision_from_observed_stage(
        retained, "series c+", company=company, evidence_attributed=True,
    ) == "mismatch"
    assert lead_scorer._first_party_acquisition_conflicts_with_venture_stage(
        company, {"observed_company_name": "Box", "dimension_evidence": {
            "required_attribute": {"url": url, "quote": quote},
        }}, "series c+",
    )


def test_untrusted_pronoun_alias_does_not_survive_saved_stage_decision():
    from tests.test_company_evidence_investigator import _company

    verdict = {
        "observed_company_name": "it", "observed_company_stage": "Acquired",
        "stage_matches": False,
        "dimension_evidence": {"stage": {
            "url": "https://parent.example/news/acquisition",
            "quote": "It was acquired by Parent.",
        }},
    }
    retained = json.loads(json.dumps(verdict))
    assert lead_scorer._decision_from_observed_stage(
        retained, "series c+", company=_company(name="Wiz", website="https://www.wiz.io"),
        evidence_attributed=True,
    ) == "unavailable"
