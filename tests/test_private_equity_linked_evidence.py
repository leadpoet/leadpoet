"""Offline contracts for separate same-sponsor ownership and classification proof."""

import asyncio
from copy import deepcopy
import json
from urllib.parse import urlsplit

import pytest

from gateway.qualification.models import CompanyOutput, ICPPrompt
from qualification.scoring import company_evidence_investigator as investigator
from qualification.scoring import lead_scorer
from qualification.scoring.competition import CompetitionCompanyScorer
from lab_arena import operations as arena_operations
from qualification.scoring.company_fit_decision import (
    COMPANY_FIT_MATCH,
    COMPANY_FIT_UNAVAILABLE,
)


# Exact short source spans supplied by the read-only ownership audit. The page
# bodies below are local fixtures; these tests make no network or model calls.
AVANTUS_URL = "https://avantus.com/about-us"
AVANTUS_QUOTE = (
    "Avantus is majority-owned by global investment firm KKR, with global "
    "energy investor EIG as the sole minority equity partner."
)
KKR_URL = (
    "https://avantus.com/news/avantus-announces-completion-of-acquisition-by-kkr-"
    "and-closing-of-usd522-million-development-facility"
)
KKR_QUOTE = (
    "KKR sponsors investment funds that invest in private equity, credit and "
    "real assets and has strategic partners that manage hedge funds."
)
AVANTUS_COMPLETED_QUOTE = (
    "Today, Avantus, a premier U.S. developer of utility-scale solar and "
    "solar-plus-storage projects, and KKR, a leading global investment firm, "
    "announced the completion of the acquisition of a majority equity interest "
    "in Avantus by investment funds and accounts managed by KKR."
)
ORIGIS_URL = (
    "https://origisenergy.com/insights/origis-energy-secures-1-billion-strategic-"
    "investment-from-brookfield-and-antin/"
)
ORIGIS_QUOTE = (
    "Origis Energy welcomes Brookfield Asset Management into its investor "
    "group, alongside majority owner, Antin Infrastructure Partners, to "
    "support U.S. growth ambitions."
)
ANTIN_QUOTE = (
    "Antin Infrastructure Partners is a leading private equity firm focused "
    "on infrastructure."
)
ACME_URL = "https://acme.example/about"
ATLAS_URL = "https://atlas.example/about"
ATLAS_QUOTE = "Atlas Capital is a private equity firm."


def _validated(
    primary="Acme is majority-owned by Atlas Capital.",
    support=ATLAS_QUOTE,
    *,
    company="Acme",
    primary_url=ACME_URL,
    support_url=ATLAS_URL,
    pages=None,
    status="VERIFIED",
    extra=None,
):
    raw = {
        "target": "stage", "status": status, "observed_value": "Private Equity",
        "evidence_url": primary_url, "evidence_quote": primary,
        "supporting_evidence_url_1": support_url,
        "supporting_evidence_quote_1": support,
        "reason": "The named majority owner is classified by the linked source.",
        **(extra or {}),
    }
    if pages is None:
        pages = {primary_url: primary, support_url: support}
        if primary_url == support_url:
            pages = {primary_url: primary + " " + support}
    return investigator._validated_findings(
        {"findings": [raw]}, targets=("stage",), fetched_pages=pages,
        first_party_domains={"acme.example", "avantus.com", "origisenergy.com"},
        identity_names={lead_scorer._compact_company_name(company)},
    )["stage"]


@pytest.mark.parametrize("company,primary,url,support,support_url", [
    ("Avantus", AVANTUS_QUOTE, AVANTUS_URL, KKR_QUOTE, KKR_URL),
    ("Avantus", AVANTUS_COMPLETED_QUOTE, KKR_URL, KKR_QUOTE, KKR_URL),
    ("Origis Energy", ORIGIS_QUOTE, ORIGIS_URL, ANTIN_QUOTE, ORIGIS_URL),
])
def test_real_ownership_and_same_sponsor_classification_compose(
    company, primary, url, support, support_url,
):
    assert not lead_scorer._stage_quote_supports_observation("private equity", primary)
    finding = _validated(
        primary, support, company=company, primary_url=url, support_url=support_url,
    )
    assert finding["status"] == "VERIFIED"
    assert finding["supporting_evidence"] == [{"url": support_url, "quote": support}]
    assert json.loads(json.dumps(finding)) == finding


@pytest.mark.parametrize("primary", [
    "Acme is currently controlled by Atlas Capital.",
    "Acme's majority owner is Atlas Capital.",
    "Atlas Capital remains the controlling owner of Acme.",
    "Atlas Capital holds a majority stake in Acme.",
    "Acme announced the completion of its majority acquisition by Atlas Capital in 2019.",
    "Atlas Capital completed its majority acquisition of Acme in 2019.",
])
def test_explicit_current_control_and_old_completed_control_are_supported(primary):
    assert _validated(primary)["status"] == "VERIFIED"


@pytest.mark.parametrize("primary", [
    "Acme lists Atlas Capital as an investor.",
    "Acme received funding from Atlas Capital.",
    "Acme received a minority investment from Atlas Capital.",
    "Acme announced an agreement for a majority acquisition by Atlas Capital.",
    "Atlas Capital plans to acquire a majority stake in Acme.",
    "Acme announced the expected completion of its majority acquisition by Atlas Capital.",
    "Acme announced the completion of its majority acquisition by Atlas Capital, subject to regulatory approval.",
    "Acme announced the completion of its majority acquisition by Atlas Capital, but the transaction was cancelled.",
    "OtherCo is majority-owned by Atlas Capital. Acme is a customer of OtherCo.",
    "Acme reported that OtherCo is majority-owned by Atlas Capital.",
    "Acme is majority-owned by Other Capital. Atlas Capital advises Acme.",
    "Acme is majority-owned by a parent of Atlas Capital.",
    "Acme is majority-owned by Atlas Capital and Partners.",
    "Acme is majority-owned by Atlas Capital for Growth.",
    "Atlas Capital is the majority owner of Acme and Sons.",
    "Acme and Sons is majority-owned by Atlas Capital.",
    "Acme was previously majority-owned by Atlas Capital.",
    "Acme is majority-owned by Atlas Capital. Atlas Capital later sold its majority stake.",
    "Acme is majority-owned by Atlas Capital. Acme later relisted on Nasdaq.",
    "Acme is majority-owned by Atlas Capital. Acme was subsequently sold to OtherCo.",
    "Acme is majority-owned by Atlas Capital. Acme is now independent.",
])
def test_insufficient_pending_wrong_entity_or_superseded_ownership_is_unproven(primary):
    finding = _validated(primary)
    assert finding["status"] == "UNPROVEN"
    assert finding["supporting_evidence"] == []


@pytest.mark.parametrize("support", [
    "Other Capital is a private equity firm.",
    "Atlas is a private equity firm.",
    "Atlas Capital is an investor.",
    "Atlas Capital is an adviser to a private equity firm.",
    "Atlas Capital is a portfolio company of a private equity firm.",
    "Atlas Capital is a private equity firm's customer.",
    "Atlas Capital is a private equity firm’s customer.",
    "Atlas Capital is a private equity investor's subsidiary.",
    "Atlas Capital is a private equity investor’s subsidiary.",
    "Atlas Capital is a private markets investment manager's client.",
    "Atlas Capital is a private equity firm-backed provider.",
    "Atlas Capital is a private equity investor-owned subsidiary.",
    "Atlas Capital is a private equity firm consultant.",
    "Atlas Capital is a private equity firm service provider.",
    "Atlas Capital is a private equity firm customer.",
    "Atlas Capital is a private equity investment manager subsidiary.",
    "Atlas Capital is a private equity firm/customer.",
    "Atlas Capital is a private equity firm—customer.",
    "Atlas Capital is a private equity firm – customer.",
    "Atlas Capital is a private equity firm.customer.",
    "Atlas Capital sponsors investment funds that invest in private equity-backed debt.",
    "Atlas Capital is not a private equity firm.",
    "Atlas Capital was previously a private equity firm.",
    "Other Capital is a private equity firm. Its investor is Atlas Capital.",
])
def test_classification_requires_same_explicit_sponsor(support):
    assert not lead_scorer._private_equity_linked_support_proves_ownership(
        ["Acme"], "Acme is majority-owned by Atlas Capital.", [support],
    )
    assert _validated(support=support)["status"] == "UNPROVEN"


@pytest.mark.parametrize("ending", [
    "", ".", ", with offices worldwide.", "; it operates worldwide.",
    " focused on infrastructure.", " specializing in infrastructure.",
    " based in London.", " headquartered in London.",
    " that invests in infrastructure.", " which invests in infrastructure.",
    " with infrastructure funds.", " and manages infrastructure funds.",
    " whose funds invest in infrastructure.",
])
def test_complete_classification_and_grammatical_continuations(ending):
    assert _validated(support="Atlas Capital is a private equity firm" + ending)["status"] == "VERIFIED"


@pytest.mark.parametrize("primary", [
    AVANTUS_COMPLETED_QUOTE.replace("in Avantus by", "in OtherCo by"),
    AVANTUS_COMPLETED_QUOTE.replace("managed by KKR", "managed by Other Capital"),
    AVANTUS_COMPLETED_QUOTE.replace("the completion", "the expected completion"),
    AVANTUS_COMPLETED_QUOTE.replace("majority equity", "minority equity"),
    AVANTUS_COMPLETED_QUOTE + " The acquisition was cancelled.",
    AVANTUS_COMPLETED_QUOTE + " The acquisition remains subject to regulatory approval.",
])
def test_completed_fund_acquisition_needs_same_target_manager_and_finality(primary):
    assert _validated(
        primary, KKR_QUOTE, company="Avantus", primary_url=KKR_URL, support_url=KKR_URL,
    )["status"] == "UNPROVEN"


@pytest.mark.parametrize("pages", [
    {ACME_URL: "Acme is majority-owned by Atlas Capital."},
    {ACME_URL: "Acme is majority-owned by Atlas Capital.", ATLAS_URL: "About Atlas Capital"},
    {ATLAS_URL: ATLAS_QUOTE},
    {ACME_URL: "About Acme Ownership Investors", ATLAS_URL: ATLAS_QUOTE},
])
def test_missing_unfetched_or_navigation_only_quotes_are_unproven(pages):
    assert _validated(pages=pages)["status"] == "UNPROVEN"


def test_every_supplied_support_pair_must_be_loaded():
    finding = _validated(extra={
        "supporting_evidence_url_2": "https://atlas.example/unfetched",
        "supporting_evidence_quote_2": "Atlas Capital is a private equity firm.",
    })
    assert finding["status"] == "UNPROVEN"


def test_model_unproven_is_never_forced_to_verified():
    finding = _validated(status="UNPROVEN")
    assert finding["status"] == "UNPROVEN"
    assert finding["evidence_quote"] == ""
    prior = {"stage_matches": None}
    assert lead_scorer._project_investigator_stage(
        prior, finding, icp_stage="private equity",
    ) == prior


def test_canonical_single_quote_path_is_unchanged():
    finding = _validated(
        "Acme is majority-owned by a private equity firm.", support="", support_url="",
    )
    assert finding["status"] == "VERIFIED"
    assert finding["supporting_evidence"] == []


@pytest.mark.parametrize("name,primary,primary_url,support_quote,support_url", [
    ("Acme", "Acme is majority-owned by Atlas Capital.", ACME_URL, ATLAS_QUOTE, ATLAS_URL),
    ("Avantus", AVANTUS_QUOTE, AVANTUS_URL, KKR_QUOTE, KKR_URL),
    ("Origis Energy", ORIGIS_QUOTE, ORIGIS_URL, ANTIN_QUOTE, ORIGIS_URL),
])
def test_investigator_projection_fit_recalculation_preserves_linked_receipts(
    name, primary, primary_url, support_quote, support_url,
):
    finding = _validated(
        primary, support_quote, company=name, primary_url=primary_url, support_url=support_url,
    )
    website = "https://" + urlsplit(primary_url).hostname
    slug = lead_scorer._compact_company_name(name)
    company = CompanyOutput(
        company_name=name, company_website=website,
        company_linkedin=f"https://www.linkedin.com/company/{slug}",
        industry="Software", employee_count="11-50", country="United States", state="California",
        intent_signals=[{"description": "ownership", "source": "company_website",
                         "url": ACME_URL, "date": "2026-10-10",
                         "snippet": "Acme is majority-owned by Atlas Capital."}],
    )
    icp = ICPPrompt(
        icp_id="proof", prompt="proof", industry="Software", sub_industry="SaaS",
        employee_count="11-50", company_stage="Private Equity",
        geography="United States", product_service="software",
    )
    prior = {
        "observed_company_name": name, "observed_company_website": company.company_website,
        "observed_company_linkedin": company.company_linkedin,
        "observed_employee_count": "11-50", "employee_size_matches": True,
        "employee_size_evidence_url": primary_url,
        "employee_size_evidence_quote": f"{name} has 11-50 employees.",
        "observed_industry": "Software", "observed_subindustry": "SaaS",
        "industry_matches": True, "industry_activity_role": "supplier_operator",
        "industry_evidence_url": primary_url,
        "industry_evidence_quote": f"{name} supplies SaaS software.",
        "observed_hq_country": "United States", "observed_hq_state": "California", "geography_matches": True,
        "geography_evidence_url": primary_url,
        "geography_evidence_quote": f"{name} is headquartered in California, United States.",
    }
    projected = lead_scorer._project_investigator_stage(prior, finding, icp_stage="private equity")
    support = [{"url": support_url, "quote": support_quote}]
    assert projected["dimension_evidence"]["stage"]["supporting_evidence"] == support
    result = lead_scorer._reverify_decision(
        projected, "", "private equity", icp=icp, company=company,
        validated_stage_finding=finding, company_quality=True,
    )
    assert result.decision == COMPANY_FIT_MATCH
    assert result.details["dimension_evidence"]["stage"]["supporting_evidence"] == support
    assert lead_scorer._decision_from_observed_stage(projected, "private equity", company=company) == COMPANY_FIT_UNAVAILABLE
    for mutation in ("drop", "wrong_quote", "wrong_url"):
        changed = deepcopy(projected)
        receipt = changed["dimension_evidence"]["stage"]
        if mutation == "drop":
            receipt.pop("supporting_evidence")
        elif mutation == "wrong_quote":
            receipt["supporting_evidence"][0]["quote"] = "Other Capital is a private equity firm."
        else:
            receipt["supporting_evidence"][0]["url"] = "https://other.example/about"
        assert lead_scorer._decision_from_observed_stage(
            changed, "private equity", company=company, validated_stage_finding=finding,
        ) == COMPANY_FIT_UNAVAILABLE


def test_private_equity_prompt_keeps_existing_research_and_tool_limits():
    assert "does not expire merely because its announcement" in investigator._SYSTEM_PROMPT
    assert "SAME sponsor" in investigator._SYSTEM_PROMPT
    assert investigator.MAX_SEARCH_CALLS == 2
    assert investigator.MAX_FETCH_CALLS == 3


@pytest.mark.parametrize("name,primary,primary_url,support,support_url,classifier_valid", [
    ("Avantus", AVANTUS_QUOTE, AVANTUS_URL, KKR_QUOTE, KKR_URL, True),
    ("Origis Energy", ORIGIS_QUOTE, ORIGIS_URL, ANTIN_QUOTE, ORIGIS_URL, True),
    ("Avantus", AVANTUS_QUOTE, AVANTUS_URL,
     "KKR is a private equity firm's customer.", KKR_URL, False),
    ("Origis Energy", ORIGIS_QUOTE, ORIGIS_URL,
     "Antin Infrastructure Partners is a private equity investor’s subsidiary.",
     ORIGIS_URL, False),
    ("Avantus", AVANTUS_QUOTE, AVANTUS_URL,
     "KKR is a private equity firm-backed provider.", KKR_URL, False),
    ("Origis Energy", ORIGIS_QUOTE, ORIGIS_URL,
     "Antin Infrastructure Partners is a private equity investor-owned subsidiary.",
     ORIGIS_URL, False),
    ("Avantus", AVANTUS_QUOTE, AVANTUS_URL,
     "KKR sponsors investment funds that invest in private equity-backed debt.",
     KKR_URL, False),
    ("Avantus", AVANTUS_QUOTE, AVANTUS_URL,
     "KKR is a private equity firm consultant.", KKR_URL, False),
    ("Avantus", AVANTUS_QUOTE, AVANTUS_URL,
     "KKR is a private equity firm service provider.", KKR_URL, False),
    ("Avantus", AVANTUS_QUOTE, AVANTUS_URL,
     "KKR is a private equity firm customer.", KKR_URL, False),
    ("Origis Energy", ORIGIS_QUOTE, ORIGIS_URL,
     "Antin Infrastructure Partners is a private equity investment manager subsidiary.",
     ORIGIS_URL, False),
    ("Avantus", AVANTUS_QUOTE, AVANTUS_URL,
     "KKR is a private equity firm/customer.", KKR_URL, False),
    ("Origis Energy", ORIGIS_QUOTE, ORIGIS_URL,
     "Antin Infrastructure Partners is a private equity firm—customer.",
     ORIGIS_URL, False),
    ("Avantus", AVANTUS_QUOTE, AVANTUS_URL,
     "Other Capital is a private equity firm. KKR advises Other Capital.", KKR_URL, False),
    ("Origis Energy", ORIGIS_QUOTE, ORIGIS_URL,
     "Other Capital is a private equity firm. Antin Infrastructure Partners is an investor.",
     ORIGIS_URL, False),
])
@pytest.mark.parametrize("model_status", ["VERIFIED", "UNPROVEN"])
def test_normal_company_scoring_uses_real_investigator_and_preserves_support(
    monkeypatch, name, primary, primary_url, support, support_url, classifier_valid,
    model_status,
):
    # Positive ownership/classification spans are real source fixtures. Invalid
    # classifier spans and other qualification dimensions are synthetic controls.
    website = "https://" + urlsplit(primary_url).hostname
    domain = urlsplit(primary_url).hostname
    slug = lead_scorer._compact_company_name(name)
    pages = {primary_url: primary, support_url: support}
    if primary_url == support_url:
        pages[primary_url] = primary + " " + support
    urls = list(pages)
    calls = {"model": 0, "intent": 0}
    verdict = {
        "observed_company_name": name, "observed_company_website": website,
        "observed_company_linkedin": f"https://www.linkedin.com/company/{slug}",
        "observed_employee_count": "11-50", "employee_size_matches": True,
        "employee_size_evidence_url": primary_url,
        "employee_size_evidence_quote": f"{name} has 11-50 employees.",
        "observed_industry": "Software", "observed_subindustry": "SaaS",
        "industry_matches": True, "industry_activity_role": "supplier_operator",
        "industry_evidence_url": primary_url,
        "industry_evidence_quote": f"{name} supplies SaaS software.",
        "observed_hq_country": "United States", "observed_hq_state": "California",
        "geography_matches": True, "geography_evidence_url": primary_url,
        "geography_evidence_quote": f"{name} is headquartered in California, United States.",
        "observed_company_stage": None, "stage_matches": None,
        "stage_evidence_url": "", "stage_evidence_quote": "",
    }

    async def prechecks(*_args, **_kwargs):
        return lead_scorer.company_fit_match("synthetic other dimensions passed")

    async def homepage(*_args, **_kwargs):
        return lead_scorer.company_fit_match("homepage identity verified", details={
            "identity": {"decision": COMPANY_FIT_MATCH,
                         "evidence_source": "company_homepage",
                         "observed_name": slug, "observed_domain": domain,
                         "observed_linkedin_slug": slug},
            "verified_homepage_transport_domain": domain,
        })

    async def broad_provider(**_kwargs):
        return verdict, ""

    async def keep_observation(observation, *_args, **_kwargs):
        return observation

    async def model(_session, _url, *, headers, payload):
        del headers
        arena_operations.validate_operation_request("openrouter.chat", payload)
        calls["model"] += 1
        turn = calls["model"]
        if turn <= len(urls):
            tool, arguments = "fetch_page", {"url": urls[turn - 1]}
        else:
            # Populate the exact existing strict tool fields, including both
            # support pairs. No new model output field or protocol is needed.
            schema = investigator._tools(("stage",))[-1]["parameters"]
            fields = schema["properties"]["findings"]["items"]["properties"]
            raw = dict.fromkeys(fields, "")
            raw.update(
                target="stage", status=model_status, observed_value="Private Equity",
                activity_role="unresolved", evidence_url=primary_url,
                evidence_quote=primary, supporting_evidence_url_1=support_url,
                supporting_evidence_quote_1=support,
                reason="Exact majority-owner quote and same-sponsor classification.",
            )
            if model_status == "UNPROVEN":
                raw.update(
                    observed_value=None, evidence_url="", evidence_quote="",
                    supporting_evidence_url_1="", supporting_evidence_quote_1="",
                    reason="Current ownership remains unproven after source review.",
                )
            tool, arguments = "submit_findings", {"findings": [raw]}
        return 200, {"choices": [{"message": {"tool_calls": [{
            "id": f"pe-proof-{turn}", "type": "function",
            "function": {"name": tool, "arguments": json.dumps(arguments)},
        }]}}]}

    async def fetch_page(_session, url):
        assert url in pages
        return {"ok": True, "url": url, "text": pages[url]}

    async def bounded_html(_session, url):
        assert url in pages
        return 200, url, pages[url]

    async def search(_session, _query, **_kwargs):
        return {"results": [{"url": primary_url}]}

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
    monkeypatch.setattr(lead_scorer, "run_company_zero_checks", prechecks)
    monkeypatch.setattr(lead_scorer, "verify_company_exists", homepage)
    monkeypatch.setattr(lead_scorer, "_request_company_reverify_json", broad_provider)
    monkeypatch.setattr(lead_scorer, "_refresh_linkedin_employee_size_observation", keep_observation)
    monkeypatch.setattr(lead_scorer, "_fetch_bounded_html", bounded_html)
    monkeypatch.setattr(lead_scorer, "score_company_competition_intent_signal", intent)
    monkeypatch.setattr(investigator, "_post_json", model)
    monkeypatch.setattr(investigator, "_fetch_page", fetch_page)
    monkeypatch.setattr(investigator, "_search_web", search)
    company = {
        "company_name": name, "company_website": website,
        "company_linkedin": verdict["observed_company_linkedin"],
        "industry": "Software", "employee_count": "11-50",
        "country": "United States", "state": "California", "company_stage": "",
        "fit_summary": f"{name} supplies SaaS software.",
        "fit_evidence_urls": [primary_url],
        "intent_signals": [{"description": "completed funding", "matched_icp_signal": 0,
            "why_now": "Completed funding supports growth.",
            "url": primary_url, "date": "2026-10-01", "snippet": primary}],
    }
    icp = ICPPrompt(
        icp_id="proof", prompt="proof", industry="Software", sub_industry="SaaS",
        employee_count="11-50", company_stage="Private Equity", geography="United States",
        product_service="software", intent_signals=["Announced a completed funding event"],
    )
    scorer = CompetitionCompanyScorer(
        integrity_policy=True, company_quality=True, evidence_investigator=True,
    )
    rows = asyncio.run(scorer.score_with_breakdowns([company], icp.model_dump(mode="json"), False))
    assert json.loads(json.dumps(rows)) == rows
    assert len(rows) == 1
    row = rows[0]
    accepted = model_status == "VERIFIED" and classifier_valid
    assert row["company_qualified"] is accepted, row
    assert row["final_score"] == (54 if accepted else 0)
    assert calls["intent"] == int(accepted)
    if classifier_valid or model_status == "UNPROVEN":
        assert calls["model"] == len(urls) + 1, row
    else:
        assert len(urls) + 1 < calls["model"] <= investigator.MAX_REASONING_TURNS + 1
    receipt = row["verifier_gate_receipts"][0]
    if accepted:
        assert receipt["dimension_evidence"]["stage"]["web_evidence"]["supporting_evidence"] == [
            {"url": support_url, "quote": support},
        ]
    else:
        assert receipt["company_fit_dimensions"]["stage"] == COMPANY_FIT_UNAVAILABLE
