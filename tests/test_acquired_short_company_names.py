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


WIZ_ANNOUNCEMENT_URL = "https://blog.google/innovation-and-ai/infrastructure-and-cloud/google-cloud/wiz-acquisition/"
# Exact Google source paragraph returned in the independent provider controls.
WIZ_ANNOUNCED_COMPLETION_QUOTE = (
    "MOUNTAIN VIEW, Calif. & NEW YORK, N.Y. — March 11, 2026 — Google LLC today "
    "announced the completion of its acquisition of Wiz, a leading cloud and "
    "AI security platform headquartered in New York."
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


@pytest.mark.parametrize(("evidence_url", "evidence_quote"), [
    (WIZ_URL, WIZ_QUOTE),
    (WIZ_ANNOUNCEMENT_URL, WIZ_ANNOUNCED_COMPLETION_QUOTE),
])
@pytest.mark.parametrize(("requested_stage", "expected_status", "expected_decision"), [
    ("Acquired", "VERIFIED", "match"),
    ("Series C+", "CONTRADICTED", "mismatch"),
])
def test_real_wiz_investigator_does_not_enter_false_citation_correction(
    monkeypatch, requested_stage, expected_status, expected_decision,
    evidence_url, evidence_quote,
):
    """The real investigator protocol with supplied model output, not model accuracy."""
    finding = dict(
        target="stage", status=expected_status, observed_value="Acquired",
        evidence_url=evidence_url, evidence_quote=evidence_quote,
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
        prior_observations={"submitted_source_urls": [evidence_url]},
        prefetched_pages={evidence_url: {"final_url": evidence_url, "text": evidence_quote}},
        verified_homepage_identity={"normalized_name": "wiz", "registrable_dns_domain": "wiz.io"},
    ))
    assert len(requests) == 1
    validated = result["_validated_stage_finding"]
    assert validated["status"] == expected_status
    assert validated["evidence_quote"] == evidence_quote
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


@pytest.mark.parametrize(("identity_names", "quote", "expected_status"), [
    ({"Wiz"}, "Wiz was acquired by Parent.", "VERIFIED"),
    ({"wiz", "A B C"}, "ABC was acquired by Parent.", "UNPROVEN"),
])
def test_synthetic_acquisition_authority_uses_normalized_candidate_names(
    monkeypatch, identity_names, quote, expected_status,
):
    matcher_names = []
    original_matcher = lead_scorer._acquired_stage_quote_supports_names

    def capture_matcher(names, source_quote):
        matcher_names.append(names)
        return original_matcher(names, source_quote)

    monkeypatch.setattr(lead_scorer, "_acquired_stage_quote_supports_names", capture_matcher)
    url = "https://www.wiz.io/news/acquisition"
    finding = dict(target="stage", status="VERIFIED", observed_value="Acquired",
        evidence_url=url, evidence_quote=quote, reason="Synthetic raw identity normalization.")
    result = investigator._validated_findings(
        {"findings": [finding]}, targets=("stage",), fetched_pages={url: quote},
        fetched_final_urls={url: url}, first_party_domains={"wiz.io"},
        identity_names=identity_names, identity_anchor={
            "submitted_name": "Wiz", "observed_name": "Wiz", "verified_name": "Wiz",
            "submitted_domain": "wiz.io", "observed_domain": "wiz.io", "verified_domain": "wiz.io",
        }, requested_stage="acquired",
    )
    assert result["stage"]["status"] == expected_status
    assert set(matcher_names[0]) == identity_names & {"Wiz", "wiz"}




@pytest.mark.parametrize("quote", [
    WIZ_ANNOUNCED_COMPLETION_QUOTE,
    "Parent announced the completion of its acquisition of Wiz.",
    "Parent announced today the completion of its acquisition of Wiz.",
])
def test_asserted_completion_announcement_binds_exact_target(quote):
    assert lead_scorer._acquired_stage_quote_supports_names(("Wiz",), quote)
    assert not lead_scorer._acquired_stage_quote_supports_names(("Google", "Parent"), quote)


@pytest.mark.parametrize("quote", [
    "Parent scheduled completion of its acquisition of Wiz.",
    "Parent expects completion of its acquisition of Wiz.",
    "Parent will announce the completion of its acquisition of Wiz.",
    "Parent announced the planned completion of its acquisition of Wiz.",
    "Parent announced the expected completion of its acquisition of Wiz.",
    "Parent has not announced the completion of its acquisition of Wiz.",
    "If Parent announced the completion of its acquisition of Wiz, ownership would change.",
    "Parent announced the completion of its acquisition of Wiz, subject to regulatory approval.",
    "Parent announced the completion of its acquisition of Wiz, but the deal was cancelled.",
    "Parent announced the completion of its acquisition of Wiz. Wiz later relisted on Nasdaq.",
    "Parent announced the completion of its acquisition of Wiz assets.",
    "Parent announced the completion of its acquisition of assets of Wiz.",
    "Parent announced the completion of its acquisition of a minority stake in Wiz.",
    "Parent announced the completion of its acquisition of Wiz minority shares.",
    "Wiz announced the completion of its acquisition of Parent.",
    "Parent announced the completion of its acquisition of Wiz Systems.",
])
def test_asserted_completion_announcement_preserves_adverse_controls(quote):
    assert not lead_scorer._acquired_stage_quote_supports_names(("Wiz",), quote)


@pytest.mark.parametrize(("requested_stage", "excluded", "duplicate"), [
    ("Acquired", False, True),
    ("Series C+", False, False),
    ("Acquired", True, False),
])
def test_native_acquisition_evidence_reaches_final_arena_score(
    monkeypatch, requested_stage, excluded, duplicate,
):
    """Real scoring/investigator; unrelated fit, intent and transports are mocked.

    This is an integration contract, not whole-company truth or billing proof.
    """
    from copy import deepcopy
    from lab_arena import scoring as arena_scoring
    from tests.test_company_evidence_investigator import (
        _complete_verdict, _competition_company, _finding, _icp,
    )

    calls = dict(broad=0, investigator=0, model=0, fetch=0, intent=0, search=0)
    runtimes = []
    original_prechecks = lead_scorer.run_company_zero_checks
    real_investigate = investigator.investigate_company_evidence
    about_url = "https://www.wiz.io/about"
    unrelated_text = (
        "Wiz has 11-50 employees. Wiz supplies SaaS software. "
        "Wiz is headquartered in California, United States."
    )

    async def prechecks(company, runtime, seen, **kwargs):
        runtimes.append(runtime)
        return await original_prechecks(company, runtime, seen, **kwargs)

    async def homepage(*_args, **_kwargs):
        return lead_scorer.company_fit_match("Mocked unrelated homepage identity", details={
            "identity": {"decision": "match", "evidence_source": "company_homepage",
                "observed_name": "wiz", "observed_domain": "wiz.io", "observed_linkedin_slug": "wiz"},
            "verified_homepage_transport_domain": "wiz.io",
        })

    async def broad(**_kwargs):
        calls["broad"] += 1
        return _complete_verdict(
            observed_company_name="Wiz", observed_company_website="https://www.wiz.io",
            observed_company_linkedin="https://www.linkedin.com/company/wiz",
            employee_size_evidence_url=about_url, employee_size_evidence_quote="Wiz has 11-50 employees.",
            industry_evidence_url=about_url, industry_evidence_quote="Wiz supplies SaaS software.",
            geography_evidence_url=about_url,
            geography_evidence_quote="Wiz is headquartered in California, United States.",
            observed_company_stage=requested_stage, stage_matches=True,
            stage_evidence_url=WIZ_ANNOUNCEMENT_URL,
            stage_evidence_quote="Wiz profile lists its funding stage.",
        ), ""

    async def preserve_size(verdict, *_args, **_kwargs):
        return verdict

    async def no_profile(*_args, **_kwargs):
        return None

    async def grounding(_session, url, **_kwargs):
        assert url in {about_url, WIZ_ANNOUNCEMENT_URL}
        return 200, url, unrelated_text if url == about_url else WIZ_ANNOUNCED_COMPLETION_QUOTE

    async def investigate(**kwargs):
        calls["investigator"] += 1
        assert kwargs["targets"] == ("stage",)
        return await real_investigate(**kwargs)

    async def fetch(_session, url, **_kwargs):
        assert url == WIZ_ANNOUNCEMENT_URL
        calls["fetch"] += 1
        return {"ok": True, "url": url, "final_url": url, "text": WIZ_ANNOUNCED_COMPLETION_QUOTE}

    async def search(*_args, **_kwargs):
        calls["search"] += 1
        return {"results": []}

    async def model(_session, _url, *, headers, payload):
        validate_operation_request("openrouter.chat", payload)
        calls["model"] += 1
        if calls["model"] == 1:
            name, arguments = "fetch_page", {"url": WIZ_ANNOUNCEMENT_URL}
        else:
            name, arguments = "submit_findings", {"findings": [_finding(
                "stage", status="VERIFIED" if requested_stage == "Acquired" else "CONTRADICTED",
                observed_value="Acquired", evidence_url=WIZ_ANNOUNCEMENT_URL,
                evidence_quote=WIZ_ANNOUNCED_COMPLETION_QUOTE,
            )]}
        return 200, {"choices": [{"message": {"tool_calls": [{
            "id": f"score-stage-{calls['model']}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }]}}]}

    async def intent(*_args, **_kwargs):
        calls["intent"] += 1
        return 54, 54, 1.0, 1.0, False, [{
            "raw": 54, "after_decay": 54, "matched_icp_signal": 0,
            "judge_verdict": {"decision": "verified", "pipeline_decision": "accept",
                "verification_trace": {"intent_verdict": {
                    "signal_evaluations": [{"signal_status": "supported"}],
                }}},
        }]

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("EXA_API_KEY", "test-key")
    monkeypatch.delenv("ARENA_INDUSTRY_EVIDENCE_MODEL", raising=False)
    monkeypatch.setattr(lead_scorer, "run_company_zero_checks", prechecks)
    monkeypatch.setattr(lead_scorer, "verify_company_exists", homepage)
    monkeypatch.setattr(lead_scorer, "_request_company_reverify_json", broad)
    monkeypatch.setattr(lead_scorer, "_refresh_linkedin_employee_size_observation", preserve_size)
    monkeypatch.setattr(lead_scorer, "_fetch_structured_linkedin_profile_once", no_profile)
    monkeypatch.setattr(lead_scorer, "_fetch_bounded_html", grounding)
    monkeypatch.setattr(lead_scorer, "investigate_company_evidence", investigate)
    monkeypatch.setattr(lead_scorer, "score_company_competition_intent_signal", intent)
    monkeypatch.setattr(investigator, "_post_json", model)
    monkeypatch.setattr(investigator, "_fetch_page", fetch)
    monkeypatch.setattr(investigator, "_search_web", search)
    company = {**_competition_company(), "company_name": "Wiz",
        "company_website": "https://www.wiz.io", "company_linkedin": "https://www.linkedin.com/company/wiz",
        "company_stage": requested_stage}
    rows = arena_scoring.score_work_item(
        {"scored_run_id": f"native-acquisition-{requested_stage}-{excluded}"},
        icp=_icp(company_stage=requested_stage, excluded_companies=["Wiz"] if excluded else [],
            max_companies=2, intent_signals=["Announced a completed funding event"]).model_dump(mode="json"),
        companies=[company, deepcopy(company)] if duplicate else [company],
        scorer=arena_scoring.lab_scorer(arena_scoring.build_scorer_policy(
            scoring_adapter_version="qualification_integrity_v2")), max_retries=1,
    )
    matched = requested_stage == "Acquired" and not excluded
    assert rows[0]["final_score"] == (54 if matched else 0)
    assert rows[0]["company_qualified"] is matched
    assert all(row["cost_penalty"] == row["time_penalty"] == 0 for row in rows)
    assert runtimes == [0.0]
    if excluded:
        assert calls == dict(broad=0, investigator=0, model=0, fetch=0, intent=0, search=0)
    else:
        assert calls == dict(broad=1, investigator=1, model=2, fetch=1,
            intent=int(matched), search=int(requested_stage == "Series C+"))
        receipt = rows[0]["verifier_gate_receipts"][0]
        assert receipt["company_fit_dimensions"]["stage"] == ("match" if matched else "mismatch")
        assert receipt["dimension_evidence"]["stage"]["web_evidence"] == {
            "url": WIZ_ANNOUNCEMENT_URL, "quote": WIZ_ANNOUNCED_COMPLETION_QUOTE,
        }
        assert all(receipt["company_fit_dimensions"][key] == "match"
            for key in ("identity", "employee_size", "industry", "geography"))
    if duplicate:
        assert len(rows) == 2
        assert rows[1]["duplicate_company"] is True
        assert rows[1]["final_score"] == 0
