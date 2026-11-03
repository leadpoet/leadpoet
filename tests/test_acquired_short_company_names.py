"""Real Wiz evidence regression; all other named transaction examples are synthetic."""

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from qualification.scoring import company_evidence_investigator as investigator
from qualification.scoring import lead_scorer
from lab_arena.operations import validate_operation_request


# Google Cloud, March 11, 2026. Exact primary-source acquisition statement.
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


@pytest.mark.parametrize(("requested_stage", "expected_status", "expected_decision"), [
    ("Acquired", "VERIFIED", "match"),
    ("Series C+", "CONTRADICTED", "mismatch"),
])
def test_real_wiz_investigator_does_not_enter_false_citation_correction(
    monkeypatch, requested_stage, expected_status, expected_decision,
):
    """The real investigator protocol with supplied model output, not model accuracy."""
    finding = dict(
        target="stage", status=expected_status, observed_value="Acquired",
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
        targets=("stage",), requested_stage=requested_stage,
        positive_semantic_review=True,
        prior_observations={"submitted_source_urls": [WIZ_URL]},
        prefetched_pages={WIZ_URL: {"final_url": WIZ_URL, "text": WIZ_QUOTE}},
        verified_homepage_identity={"normalized_name": "wiz", "registrable_dns_domain": "wiz.io"},
    ))
    assert len(requests) == 1
    validated = result["_validated_stage_finding"]
    assert validated["status"] == expected_status
    assert validated["evidence_quote"] == WIZ_QUOTE
    normalized_stage = lead_scorer._normalize_company_stage(requested_stage)
    projected = lead_scorer._project_investigator_stage({}, validated, icp_stage=normalized_stage)
    retained = json.loads(json.dumps(projected))
    from tests.test_company_evidence_investigator import _company
    company = _company(name="Wiz", website="https://www.wiz.io")
    assert lead_scorer._decision_from_observed_stage(
        retained, normalized_stage, company=company, evidence_attributed=True,
    ) == expected_decision


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
        identity_names={"box"}, identity_anchor={"submitted_name": "Box"},
        requested_stage="series c+",
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


@pytest.mark.parametrize("alias", ["he", "she", "you", "our", "its", "Arm"])
def test_untrusted_short_observed_alias_is_not_acquisition_identity(alias):
    quote = f"{alias} was acquired by Parent."
    url = "https://parent.example/news/acquisition"
    raw = dict(target="stage", status="VERIFIED", observed_value="Acquired",
        evidence_url=url, evidence_quote=quote, reason="Synthetic unrelated observed alias.")
    findings = investigator._validated_findings(
        {"findings": [raw]}, targets=("stage",), fetched_pages={url: quote},
        fetched_final_urls={url: url}, first_party_domains={"wiz.io"},
        identity_names={"wiz", alias.casefold()}, identity_anchor={
            "submitted_name": "Wiz", "verified_name": "Wiz", "observed_name": alias,
        }, requested_stage="acquired",
    )
    assert findings["stage"]["status"] == "UNPROVEN"


@pytest.mark.parametrize("alias", ["Arm", "he", "she", "you", "our", "its", "it"])
@pytest.mark.parametrize("wrong_buyer", [False, True])
def test_forged_stage_finding_cannot_authorize_short_observed_identity(alias, wrong_buyer):
    from tests.test_company_evidence_investigator import _company

    quote = f"Wiz completed its acquisition of {alias}." if wrong_buyer else f"{alias} was acquired by Parent."
    url = "https://parent.example/news/acquisition"
    finding = dict(target="stage", status="VERIFIED", observed_value="Acquired",
        evidence_url=url, evidence_quote=quote, reason="Synthetic forged validated finding.")
    verdict = lead_scorer._project_investigator_stage(
        {"observed_company_name": alias}, finding, icp_stage="acquired",
    )
    assert lead_scorer._decision_from_observed_stage(
        json.loads(json.dumps(verdict)), "acquired", validated_stage_finding=finding,
        company=_company(name="Wiz", website="https://www.wiz.io"),
        evidence_attributed=True,
    ) == "unavailable"


@pytest.mark.parametrize("name", ["Wiz", "Box", "Arm", "ADT"])
@pytest.mark.parametrize("authority_key", ["submitted_name", "verified_name"])
def test_short_acquisition_names_require_submitted_or_verified_authority(name, authority_key):
    quote = f"Parent completed its acquisition of {name}."
    url = "https://parent.example/news/acquisition"
    finding = dict(target="stage", status="VERIFIED", observed_value="Acquired",
        evidence_url=url, evidence_quote=quote, reason="Synthetic explicit identity authority.")
    accepted = investigator._validated_findings(
        {"findings": [finding]}, targets=("stage",), fetched_pages={url: quote},
        fetched_final_urls={url: url}, first_party_domains={"wiz.io"},
        identity_names={name.casefold()}, identity_anchor={authority_key: name},
        requested_stage="acquired",
    )
    assert accepted["stage"]["status"] == "VERIFIED"
    rejected = investigator._validated_findings(
        {"findings": [finding]}, targets=("stage",), fetched_pages={url: quote},
        fetched_final_urls={url: url}, first_party_domains={"wiz.io"},
        identity_names={name.casefold()}, identity_anchor={"observed_name": name},
        requested_stage="acquired",
    )
    assert rejected["stage"]["status"] == "UNPROVEN"


def test_synthetic_validated_rebrand_authorizes_short_stage_name_without_saved_bypass():
    from tests.test_company_evidence_investigator import _company

    rebrand_url = "https://www.wiz.io/rebrand"
    stage_url = "https://parent.example/news/acquisition"
    rebrand_quote = "Arm is now Wiz following our rebrand. The domain changed from arm.com to wiz.io."
    stage_quote = "Arm was acquired by Parent."
    rebrand = dict(target="rebrand", status="VERIFIED", observed_value="Wiz",
        evidence_url=rebrand_url, evidence_quote=rebrand_quote,
        old_name="Arm", new_name="Wiz", old_domain="arm.com", new_domain="wiz.io",
        shared_linkedin_slug="arm", reason="Synthetic completed rebrand.")
    stage = dict(target="stage", status="VERIFIED", observed_value="Acquired",
        evidence_url=stage_url, evidence_quote=stage_quote, reason="Synthetic acquired alias.")
    findings = investigator._validated_findings(
        {"findings": [stage, rebrand]}, targets=("stage", "rebrand"),
        fetched_pages={rebrand_url: rebrand_quote, stage_url: stage_quote},
        fetched_final_urls={rebrand_url: rebrand_url, stage_url: stage_url},
        first_party_domains={"arm.com", "wiz.io"}, identity_names={"wiz"},
        identity_anchor={"submitted_name": "Wiz", "verified_name": "Wiz"},
        requested_stage="acquired",
    )
    assert findings["rebrand"]["status"] == "VERIFIED"
    assert findings["stage"]["status"] == "VERIFIED"
    projected = lead_scorer._project_investigator_stage(
        {"observed_company_name": "Arm"}, findings["stage"], icp_stage="acquired",
    )
    # No new saved receipt is introduced. The short observed alias alone does
    # not transfer identity authority through an arbitrary finding dictionary.
    assert lead_scorer._decision_from_observed_stage(
        json.loads(json.dumps(projected)), "acquired",
        validated_stage_finding=json.loads(json.dumps(findings["stage"])),
        company=_company(name="Wiz", website="https://www.wiz.io"),
        evidence_attributed=True,
    ) == "unavailable"
