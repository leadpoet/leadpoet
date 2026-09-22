"""Saved September 20-22 verifier cases at the real Arena adapter boundary.

The source investigation did not retain complete provider transcripts. Exact
submitted company/ICP pairs cross the production adapter. Recorded disputed
fields and clearly labelled curated provider fixtures exercise the real gates:
Cloudforce recovers, Samsara remains unproven, and genuine negative controls
remain unqualified. These are controlled regressions, not historical replays.
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from lab_arena import scoring
from qualification.scoring import company_evidence_investigator, lead_scorer
from qualification.scoring.company_fit_decision import (
    COMPANY_FIT_MATCH,
    COMPANY_FIT_MISMATCH,
    company_fit_match,
    company_fit_unavailable,
    evaluate_company_identity,
)


FIXTURE = Path(__file__).parent / "fixtures" / "sep22_investigated_cases.json"
CASE_NAMES = {
    "Buywander",
    "Samsara",
    "Bayesian Health",
    "MontyCloud",
    "Max Retail",
    "Dexory",
    "ABEC",
    "Flexport",
    "Phia",
    "Shipday",
    "Cimulate AI",
    "TPG",
    "HiddenLayer",
    "Kevel",
    "Kong Inc.",
    "Fnality International",
    "Cloudforce",
    "Crusoe",
    "Zenskar",
}
MISSING_CONTACT_CONTROLS = {
    "Phia", "Shipday", "Cimulate AI", "HiddenLayer",
    "Fnality International", "Zenskar",
}


def _cases() -> list[dict]:
    document = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert document["schema_version"] == (
        "leadpoet.arena.saved_case_regression.v1"
    )
    return document["cases"]


def _case(name: str) -> dict:
    return next(row for row in _cases() if row["name"] == name)


def _policy() -> dict:
    return scoring.build_scorer_policy(
        scoring_adapter_version="qualification_contacts_v3",
        intent_details=True,
        company_quality=False,
    )


def _bind_placeholder_credentials(monkeypatch) -> None:
    for name in scoring.CREDENTIAL_ENV_NAMES:
        monkeypatch.setenv(name, "saved-case-placeholder")


def _domain(url: str) -> str:
    return str(urlsplit(url).hostname or "").removeprefix("www.")


def _linkedin_slug(row: dict) -> str:
    path = str(urlsplit(row["company"].get("company_linkedin") or "").path)
    parts = [part for part in path.split("/") if part]
    if len(parts) >= 2 and parts[0].casefold() == "company":
        return parts[1]
    return row["name"].casefold().replace(" ", "-").replace(".", "")


def _homepage_identity(row: dict):
    """Return the controlled result of the real homepage identity boundary."""

    company = row["company"]
    linkedin = company.get("company_linkedin") or (
        "https://www.linkedin.com/company/" + _linkedin_slug(row)
    )
    identity = evaluate_company_identity(
        submitted_name=company["company_name"],
        submitted_website=company["company_website"],
        submitted_linkedin=company.get("company_linkedin") or "",
        observed_name=company["company_name"],
        observed_website=company["company_website"],
        observed_linkedin=linkedin,
        evidence_source="company_homepage",
    )
    assert identity["decision"] == COMPANY_FIT_MATCH
    return company_fit_match(
        "controlled homepage fetch bound the exact saved entity",
        details={
            "identity": identity,
            "verified_homepage_transport_domain": _domain(
                company["company_website"]
            ),
        },
    )


def _complete_provider_verdict(
    row: dict,
    *,
    observed_employee_count=None,
    employee_size_matches=True,
    employee_size_evidence_url="https://www.linkedin.com/company/example",
    employee_size_evidence_quote="The company has an in-range team size.",
    observed_industry=None,
    observed_subindustry=None,
    industry_matches=True,
    industry_activity_role="supplier_operator",
    industry_evidence_url=None,
    industry_evidence_quote=None,
    observed_hq_country=None,
    observed_hq_state=None,
    geography_matches=True,
    geography_evidence_url=None,
    geography_evidence_quote=None,
    observed_company_stage=None,
    stage_matches=True,
    stage_evidence_url=None,
    stage_evidence_quote=None,
) -> dict:
    """Add the strict receipt fields absent from the retained raw transcripts."""

    company = row["company"]
    attribute = company.get("required_attribute") or {}
    attribute_url = str(
        attribute.get("evidence_url")
        or company["company_website"]
    )
    attribute_quote = str(
        attribute.get("evidence_quote")
        or "The company provides the requested product or service."
    )
    linkedin = company.get("company_linkedin") or (
        "https://www.linkedin.com/company/" + _linkedin_slug(row)
    )
    return {
        "observed_company_name": company["company_name"],
        "observed_company_website": company["company_website"],
        "observed_company_linkedin": linkedin,
        "observed_employee_count": (
            company.get("employee_count")
            if observed_employee_count is None
            else observed_employee_count
        ),
        "employee_size_matches": employee_size_matches,
        "employee_size_evidence_url": employee_size_evidence_url,
        "employee_size_evidence_quote": employee_size_evidence_quote,
        "observed_industry": observed_industry or company.get("industry") or "",
        "observed_subindustry": observed_subindustry or row["icp"].get("sub_industry") or "",
        "industry_matches": industry_matches,
        "industry_activity_role": industry_activity_role,
        "industry_evidence_url": industry_evidence_url or company["company_website"],
        "industry_evidence_quote": industry_evidence_quote or (
            "The company supplies the requested product or service."
        ),
        "observed_hq_country": observed_hq_country or company.get("country") or "",
        "observed_hq_state": (
            company.get("state") or ""
            if observed_hq_state is None
            else observed_hq_state
        ),
        "geography_matches": geography_matches,
        "geography_evidence_url": geography_evidence_url or company["company_website"],
        "geography_evidence_quote": geography_evidence_quote or (
            "The company is headquartered at the submitted location."
        ),
        "observed_company_stage": (
            company.get("company_stage") or ""
            if observed_company_stage is None
            else observed_company_stage
        ),
        "stage_matches": stage_matches,
        "stage_evidence_url": stage_evidence_url or company["company_website"],
        "stage_evidence_quote": stage_evidence_quote or (
            "The company has the observed current stage."
        ),
        "attribute_satisfied": True,
        "required_attribute_evidence_url": attribute_url,
        "required_attribute_evidence_quote": attribute_quote,
        "reason": "controlled saved-case evidence envelope",
    }


def _finding(target: str, **overrides) -> dict:
    result = {
        "target": target,
        "status": "VERIFIED",
        "observed_value": "",
        "observed_country": "",
        "observed_state": "",
        "observed_industry": "",
        "observed_subindustry": "",
        "activity_role": "unresolved",
        "evidence_url": "",
        "evidence_quote": "",
        "old_name": "",
        "new_name": "",
        "old_domain": "",
        "new_domain": "",
        "shared_linkedin_slug": "",
        "reason": "bounded saved-case evidence",
    }
    result.update(overrides)
    return result


def _company_receipt(result: dict) -> dict:
    return next(
        receipt
        for receipt in result["verifier_gate_receipts"]
        if receipt.get("gate") == "company_fit"
    )


def _install_company_boundaries(monkeypatch, row: dict, verdict: dict) -> None:
    async def homepage(*_args, **_kwargs):
        return _homepage_identity(row)

    async def broad_provider(**_kwargs):
        return dict(verdict), ""

    async def grounded_attribute(_session, url, **_kwargs):
        del _session, _kwargs
        quote = verdict["required_attribute_evidence_quote"]
        return 200, url, f"<html><body>{quote}</body></html>"

    async def no_structured_profile(*_args, **_kwargs):
        return None

    monkeypatch.setattr(lead_scorer, "verify_company_exists", homepage)
    monkeypatch.setattr(
        lead_scorer, "_request_company_reverify_json", broad_provider
    )
    monkeypatch.setattr(lead_scorer, "_fetch_bounded_html", grounded_attribute)
    monkeypatch.setattr(
        lead_scorer,
        "fetch_structured_linkedin_company_size",
        no_structured_profile,
    )


def _score_saved_case(row: dict) -> dict:
    return scoring.score_work_item(
        {"scored_run_id": "saved-gate-" + row["name"].casefold().replace(" ", "-")},
        icp=row["icp"],
        companies=[row["company"]],
        scorer=scoring.lab_scorer(_policy()),
        max_retries=1,
    )[0]


def test_fixture_contains_all_exact_investigated_inputs_with_only_secrets_sanitized():
    cases = _cases()
    assert len(cases) == 19
    assert {row["name"] for row in cases} == CASE_NAMES
    assert len({
        (row["round"], row["submission"], row["position"], row["name"])
        for row in cases
    }) == 19

    serialized = json.dumps(cases, sort_keys=True)
    assert "@bayesianhealth.com" not in serialized
    assert "@konghq.com" not in serialized
    assert "ACoAAA" not in serialized
    for row in cases:
        assert row["company"]["company_name"] == row["name"]
        assert row["icp"]["icp_id"]
        contact = row["company"].get("contact")
        if isinstance(contact, dict) and contact.get("email"):
            assert contact["email"].endswith("@example.invalid")

    assert {
        row["name"] for row in cases if row["company"].get("contact") is None
    } == MISSING_CONTACT_CONTROLS


def test_all_saved_cases_cross_lab_scorer_and_score_work_item(monkeypatch):
    """Exercise exact inputs through the production adapter without fake verdicts."""

    _bind_placeholder_credentials(monkeypatch)
    reached = []

    class ReachedVerifierBoundary(RuntimeError):
        pass

    async def stop_at_verifier_boundary(*, company, icp, **_kwargs):
        reached.append((company.company_name, icp.icp_id))
        raise ReachedVerifierBoundary(company.company_name)

    # This aborts before any verdict or score is produced.  It tests adapter
    # transport only; case recovery is covered at the provider boundaries.
    monkeypatch.setattr(
        lead_scorer,
        "score_company_competition_intent",
        stop_at_verifier_boundary,
    )
    arena_scorer = scoring.lab_scorer(_policy())

    for index, row in enumerate(_cases()):
        with pytest.raises(scoring.ScoringError, match="ReachedVerifierBoundary"):
            scoring.score_work_item(
                {"scored_run_id": "saved-case-%02d" % index},
                icp=row["icp"],
                companies=[row["company"]],
                scorer=arena_scorer,
                max_retries=1,
            )

    assert reached == [
        (row["name"], row["icp"]["icp_id"]) for row in _cases()
    ]


def test_exact_dexory_label_reaches_independent_company_evidence(monkeypatch):
    """The historical Robotics Engineering label cannot be a terminal mismatch."""

    _bind_placeholder_credentials(monkeypatch)
    row = _case("Dexory")
    calls = []

    async def controlled_identity_boundary(*_args, **_kwargs):
        calls.append("identity")
        return company_fit_unavailable("controlled identity evidence gap")

    async def controlled_company_model_boundary(*_args, **_kwargs):
        calls.append("company_model")
        raise RuntimeError("controlled-company-model-boundary")

    monkeypatch.setattr(
        lead_scorer, "verify_company_exists", controlled_identity_boundary
    )
    monkeypatch.setattr(
        lead_scorer, "_llm_reverify_company", controlled_company_model_boundary
    )
    arena_scorer = scoring.lab_scorer(_policy())

    with pytest.raises(scoring.ScoringError, match="controlled-company-model-boundary"):
        scoring.score_work_item(
            {"scored_run_id": "saved-dexory"},
            icp=row["icp"],
            companies=[row["company"]],
            scorer=arena_scorer,
            max_retries=1,
        )

    assert row["company"]["industry"] == "Robotics Engineering"
    assert row["icp"]["industry"] == "Hardware"
    assert calls == ["identity", "company_model"]


def test_saved_missing_contact_controls_remain_explicit_required_field_failures():
    """No evidence investigator may invent a contact for these submissions."""

    from qualification.scoring.competition import _not_evaluated_contact

    for name in sorted(MISSING_CONTACT_CONTROLS):
        row = _case(name)
        result = _not_evaluated_contact(row["company"])
        assert result["contact_qualified"] is False
        assert result["email_status"] == "unknown"
        assert result["contact_verification"] == {
            "decision": "not_evaluated",
            "reason": "company_not_qualified",
            "subchecks": {},
            "evidence_hashes": {},
            "evidence_timestamps": {},
        }


def _install_terminal_intent_fetch_miss(monkeypatch) -> None:
    """Let the real intent verifier produce a terminal zero after company fit."""

    from qualification.scoring import intent_verification_three_stage as intent

    state = {"call": 0, "url": "", "quote": ""}

    async def provider_judge(*_args, **_kwargs):
        state["call"] += 1
        if state["call"] % 2:
            return {
                "answer": {"signal_evaluations": [{
                    "signal_status": "unable_to_verify",
                    "same_entity_check": "unclear",
                    "confidence": "medium",
                }]},
                "model": "saved-case-stage-one",
                "usage": {},
            }
        return {
            "answer": {
                "overall_verdict": "disqualified",
                "overall_confidence": "high",
                "summary": "controlled downstream intent rejection",
                "missing_or_risks": [],
                "signal_evaluations": [{
                    "signal_id": "signal-1",
                    "claim": "controlled downstream intent claim",
                    "verification_mode": "source_grounded",
                    "signal_status": "contradicted",
                    "source_urls_supplied": [state["url"]],
                    "evidence_urls_used": [state["url"]],
                    "source_accessibility": "accessible",
                    "same_entity_check": "pass",
                    "entity_match_reason": "exact company name",
                    "supporting_quotes": [],
                    "contradicting_quotes": [state["quote"]],
                    "unsupported_parts": [],
                    "source_quality": "controlled_fixture",
                    "risk_notes": [],
                    "confidence": "high",
                    "claim_matches_miner_date": "no_date_in_content",
                    "author_type": "n/a",
                    "author_employer_matches_lead": "n/a",
                    "author_role_matches_spec": "n/a",
                    "author_satisfies_role_spec": "n/a",
                }],
            },
            "model": "saved-case-stage-three",
            "usage": {},
        }

    async def controlled_provider_fetch(urls, *_args, **_kwargs):
        state["url"] = urls[0]
        state["quote"] = "Cloudforce evidence contradicts this intent claim."
        return {
            "results": [{
                "url": state["url"],
                "title": "Controlled downstream intent evidence",
                "text": state["quote"],
            }],
            "statuses": [{"source": "controlled", "stage": "ok"}],
        }

    monkeypatch.setattr(intent, "_call_openrouter", provider_judge)
    monkeypatch.setattr(intent, "_fetch_sd_then_exa", controlled_provider_fetch)


@pytest.mark.parametrize(
    ("name", "raw_fields", "finding", "recovered_dimension"),
    [
        (
            "Cloudforce",
            {
                # These five values are the last retained raw Sonar judgment.
                "observed_employee_count": "51-200",
                "observed_industry": "Technology, Information and Internet",
                "observed_subindustry": "cloud and AI consulting firm",
                "industry_matches": False,
                "industry_activity_role": "unresolved",
                "industry_evidence_url": (
                    "https://gocloudforce.com/cloudforce-secures-10m-series-a-"
                    "from-owl-ventures-and-microsoft-to-democratize-safe-"
                    "equitable-ai-in-education-healthcare-and-beyond/"
                ),
                "industry_evidence_quote": (
                    "Cloudforce brings safe and equitable AI to students, "
                    "faculty, researchers, and institutions through nebulaONE."
                ),
                "observed_hq_country": "United States",
                "observed_hq_state": "Maryland",
                "observed_company_stage": "Series A",
                "stage_evidence_quote": (
                    "Cloudforce announced the closing of a $10 million "
                    "Series A funding round."
                ),
            },
            _finding(
                "industry",
                observed_value="Education technology",
                observed_industry="Education",
                observed_subindustry="AI platform for universities",
                activity_role="supplier_operator",
                evidence_url=(
                    "https://gocloudforce.com/cloudforce-secures-10m-series-a-"
                    "from-owl-ventures-and-microsoft-to-democratize-safe-"
                    "equitable-ai-in-education-healthcare-and-beyond/"
                ),
                evidence_quote=(
                    "Cloudforce nebulaONE deployments at the University of Maryland, "
                    "Baltimore County and University of Maryland Global Campus"
                ),
            ),
            "industry",
        ),
    ],
)
def test_saved_false_negative_company_fact_is_recovered_before_later_gate(
    monkeypatch, name, raw_fields, finding, recovered_dimension
):
    """Recorded raw fields plus curated leaf evidence cross the real adapter.

    Complete historical provider transcripts were not retained.  The raw
    disputed fields above are exact.  The concise leaf snippets are synthetic,
    source-equivalent fixtures derived from the audited September evidence;
    they are not represented as verbatim historical provider output.
    """

    _bind_placeholder_credentials(monkeypatch)
    row = _case(name)
    verdict = _complete_provider_verdict(row, **raw_fields)
    _install_company_boundaries(monkeypatch, row, verdict)
    _install_terminal_intent_fetch_miss(monkeypatch)
    calls = []

    async def bounded_investigator(*, targets, **_kwargs):
        calls.append(tuple(targets))
        assert finding["target"] in targets
        raw_findings = [finding]
        if "headcount" in targets and finding["target"] != "headcount":
            raw_findings.append(_finding(
                "headcount",
                observed_value="51-200",
                evidence_url="https://www.linkedin.com/company/gocloudforce",
                evidence_quote="Cloudforce has 51-200 employees company-wide.",
            ))
        fetched_pages = {
            item["evidence_url"]: item["evidence_quote"]
            for item in raw_findings
            if item["evidence_url"]
        }
        claims = company_evidence_investigator._validated_findings(
            {"findings": raw_findings},
            targets=tuple(targets),
            fetched_pages=fetched_pages,
            first_party_domains={"gocloudforce.com"},
            identity_names={"cloudforce"},
            identity_anchor={
                "submitted_name": "Cloudforce",
                "submitted_domain": "gocloudforce.com",
                "submitted_linkedin_slug": "",
                "observed_name": "Cloudforce",
                "observed_domain": "gocloudforce.com",
                "verified_domain": "gocloudforce.com",
                "observed_linkedin_slug": "cloudforce",
            },
        )
        assert claims is not None
        assert all(item["status"] == "VERIFIED" for item in claims.values())
        return {"claims": claims, "failure_reason": ""}

    monkeypatch.setattr(
        lead_scorer, "investigate_company_evidence", bounded_investigator
    )

    result = _score_saved_case(row)
    receipt = _company_receipt(result)

    assert receipt["decision"] == COMPANY_FIT_MATCH, json.dumps(receipt, indent=2)
    assert receipt["company_fit_dimensions"][recovered_dimension] == (
        COMPANY_FIT_MATCH
    )
    assert result["final_score"] == 0.0
    assert len(calls) == 1
    assert finding["target"] in calls[0]
    # Recovery only clears the disputed company fact.  The controlled missing
    # intent page still fails later in the unchanged scoring path.
    assert result["intent_signal_raw"] == 0.0
    assert result["intent_signals_detail"][0]["claim_support_verdict"] == (
        "contradicted"
    )


def test_saved_samsara_headcount_gap_is_investigated_without_inventing_count(
    monkeypatch,
):
    """The SEC lower bound triggers research but is not converted to exact 4,100."""

    _bind_placeholder_credentials(monkeypatch)
    row = _case("Samsara")
    verdict = _complete_provider_verdict(
        row,
        # Exact retained raw Sonar value; the retained response had no bound
        # headcount source fields, so strict evidence review must reopen it.
        observed_employee_count=5790,
        employee_size_matches=True,
        employee_size_evidence_url="",
        employee_size_evidence_quote="",
        observed_industry="Transportation and logistics technology",
        observed_subindustry=(
            "Connected operations platform for fleets and passenger transit"
        ),
        observed_hq_country="United States",
        observed_hq_state="California",
        observed_company_stage="Public",
        stage_evidence_url=(
            "https://www.sec.gov/Archives/edgar/data/1642896/"
            "000162828026018167/iot-20260131.htm"
        ),
        stage_evidence_quote=(
            "Samsara Inc. Class A common stock trades on the New York Stock "
            "Exchange under the symbol IOT."
        ),
    )
    _install_company_boundaries(monkeypatch, row, verdict)
    calls = []

    async def bounded_investigator(*, targets, **_kwargs):
        calls.append(tuple(targets))
        assert "headcount" in targets
        claims = {
            target: _finding(
                target,
                status="UNPROVEN",
                reason=(
                    "The filing says more than 4,100; this lower bound must "
                    "not be rewritten as an exact count or canonical band."
                ),
            )
            for target in targets
        }
        return {"claims": claims, "failure_reason": ""}

    monkeypatch.setattr(
        lead_scorer, "investigate_company_evidence", bounded_investigator
    )
    with pytest.raises(
        scoring.ScoringError,
        match="Company fit unavailable: unproven dimensions: employee_size",
    ):
        _score_saved_case(row)
    assert calls == [("headcount",)]


NEGATIVE_COMPANY_CONTROLS = [
    pytest.param(
        "ABEC",
        "geography",
        {
            "observed_hq_country": "United States",
            "observed_hq_state": "Pennsylvania",
            "geography_matches": False,
            "geography_evidence_url": "https://www.abec.com/global-footprint/",
            "geography_evidence_quote": (
                "ABEC is headquartered in Bethlehem, Pennsylvania."
            ),
            "observed_company_stage": "Private Equity",
        },
        id="abec-pennsylvania-is-not-midwest",
    ),
    pytest.param(
        "Kong Inc.",
        "geography",
        {
            "observed_hq_country": "United States",
            "observed_hq_state": "California",
            "geography_matches": False,
            "geography_evidence_url": "https://konghq.com/company/",
            "geography_evidence_quote": (
                "Kong Inc. is headquartered in San Francisco, California."
            ),
        },
        id="kong-california-is-not-us-south",
    ),
    pytest.param(
        "Flexport",
        "stage",
        {
            "observed_company_stage": "Series E",
            "stage_matches": False,
            "stage_evidence_url": (
                "https://www.flexport.com/blog/flexport-raises-935-million-"
                "to-boost-resilience-and-visibility-in-supply-chains/"
            ),
            "stage_evidence_quote": (
                "Flexport announced a $935 million Series E investment round."
            ),
        },
        id="flexport-is-not-public",
    ),
    pytest.param(
        "TPG",
        "stage",
        {
            "observed_company_stage": "Public",
            "stage_matches": False,
            "stage_evidence_url": "https://shareholders.tpg.com/",
            "stage_evidence_quote": "TPG Inc. is listed on NASDAQ as TPG.",
        },
        id="tpg-inc-is-public-not-private-equity-stage",
    ),
    pytest.param(
        "Kevel",
        "stage",
        {
            "observed_company_stage": "Series C",
            "stage_matches": False,
            "stage_evidence_url": "https://www.kevel.com/blog/series-c-press-release",
            "stage_evidence_quote": (
                "Kevel announced $23 million in Series C funding."
            ),
        },
        id="kevel-series-c-supersedes-series-b",
    ),
    pytest.param(
        "Cimulate AI",
        "stage",
        {
            "observed_company_stage": "Acquired",
            "stage_matches": False,
            "stage_evidence_url": (
                "https://www.salesforce.com/news/stories/"
                "salesforce-signs-definitive-agreement-to-acquire-cimulate/"
            ),
            "stage_evidence_quote": (
                "Salesforce completed its acquisition of Cimulate on "
                "March 3, 2026."
            ),
        },
        id="cimulate-acquisition-supersedes-series-a",
    ),
]


@pytest.mark.parametrize("name,dimension,overrides", NEGATIVE_COMPANY_CONTROLS)
def test_saved_company_negative_control_stays_zero_through_real_adapter(
    monkeypatch, name, dimension, overrides
):
    """Agentic review cannot erase an independently proven negative fact."""

    _bind_placeholder_credentials(monkeypatch)
    row = _case(name)
    verdict = _complete_provider_verdict(row, **overrides)
    _install_company_boundaries(monkeypatch, row, verdict)

    investigation_calls = []

    async def bounded_control_investigator(*, targets, **_kwargs):
        investigation_calls.append(tuple(targets))
        raw_findings = []
        for target in targets:
            if target == "stage" and dimension == "stage":
                raw_findings.append(_finding(
                    "stage",
                    status="CONTRADICTED",
                    observed_value=overrides["observed_company_stage"],
                    evidence_url=overrides["stage_evidence_url"],
                    evidence_quote=overrides["stage_evidence_quote"],
                ))
            elif target == "geography" and dimension == "geography":
                raw_findings.append(_finding(
                    "geography",
                    status="CONTRADICTED",
                    observed_value=(
                        f'{overrides["observed_hq_state"]}, '
                        f'{overrides["observed_hq_country"]}'
                    ),
                    observed_country=overrides["observed_hq_country"],
                    observed_state=overrides["observed_hq_state"],
                    evidence_url=overrides["geography_evidence_url"],
                    evidence_quote=overrides["geography_evidence_quote"],
                ))
            elif target == "headcount":
                raw_findings.append(_finding(
                    "headcount",
                    observed_value=row["company"]["employee_count"],
                    evidence_url=(
                        row["company"].get("company_linkedin")
                        or row["company"]["company_website"]
                    ),
                    evidence_quote=(
                        f'{row["name"]} has '
                        f'{row["company"]["employee_count"]} employees company-wide.'
                    ),
                ))
            else:
                raw_findings.append(_finding(target, status="UNPROVEN"))
        fetched_pages = {
            item["evidence_url"]: item["evidence_quote"]
            for item in raw_findings
            if item["evidence_url"]
        }
        normalized_name = "".join(
            character for character in row["name"].casefold()
            if character.isalnum()
        )
        claims = company_evidence_investigator._validated_findings(
            {"findings": raw_findings},
            targets=tuple(targets),
            fetched_pages=fetched_pages,
            first_party_domains={_domain(row["company"]["company_website"])},
            identity_names={normalized_name},
            identity_anchor={
                "submitted_name": row["name"],
                "submitted_domain": _domain(row["company"]["company_website"]),
                "submitted_linkedin_slug": _linkedin_slug(row),
                "observed_name": row["name"],
                "observed_domain": _domain(row["company"]["company_website"]),
                "verified_domain": _domain(row["company"]["company_website"]),
                "observed_linkedin_slug": _linkedin_slug(row),
            },
        )
        assert claims is not None
        result = {
            "claims": claims,
            "failure_reason": "",
        }
        stage_finding = claims.get("stage")
        if (
            isinstance(stage_finding, dict)
            and stage_finding.get("status") in {"VERIFIED", "CONTRADICTED"}
        ):
            result["_validated_stage_finding"] = stage_finding
        return result

    monkeypatch.setattr(
        lead_scorer, "investigate_company_evidence", bounded_control_investigator
    )
    result = _score_saved_case(row)
    receipt = _company_receipt(result)

    assert result["final_score"] == 0.0
    expected = "unavailable" if name == "Cimulate AI" else COMPANY_FIT_MISMATCH
    assert receipt["decision"] == expected
    assert receipt["company_fit_dimensions"][dimension] == expected
    if investigation_calls:
        assert dimension in investigation_calls[0] or dimension == "geography"
    assert not any(
        gate.get("gate") == "intent_verification"
        for gate in result["verifier_gate_receipts"]
    )
