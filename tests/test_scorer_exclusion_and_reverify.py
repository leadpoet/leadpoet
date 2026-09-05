"""Scorer-side exclusion enforcement + web re-verification decision logic."""

import asyncio
import copy
import os
import pickle
from unittest import mock

import pytest

from gateway.qualification.models import CompanyOutput, ICPPrompt
from qualification.scoring.lead_scorer import (
    _llm_reverify_company,
    _matches_exclusion_list,
    _reverify_decision,
    _run_company_binary_fit_checks,
    _run_competition_binary_fit_checks,
    _verify_company_fit,
)
from qualification.scoring.company_fit_decision import (
    COMPANY_FIT_MATCH,
    COMPANY_FIT_MISMATCH,
    COMPANY_FIT_UNAVAILABLE,
    aggregate_company_fit_decisions,
    company_fit_decision_contract_identity,
    company_fit_match,
    company_fit_mismatch,
    company_fit_unavailable,
    evaluate_company_identity,
)


def _company(name="Acme", website="https://acme.com", linkedin=""):
    return CompanyOutput(
        company_name=name, company_website=website, company_linkedin=linkedin,
        industry="Software", employee_count="51-200", country="United States",
        intent_signals=[{"description": "raised", "source": "news",
                         "url": "https://n.example.com/a", "date": "2026-07-01",
                         "snippet": "Acme raised a round this month."}],
    )


def _icp(**over):
    base = dict(icp_id="t", prompt="p", industry="Software", sub_industry="SaaS",
                employee_count="11-50|51-200", company_stage="",
                geography="United States", product_service="x")
    base.update(over)
    return ICPPrompt(**base)


def test_exclusion_matcher_by_domain_linkedin_name(monkeypatch):
    monkeypatch.setattr(
        "qualification.scoring.lead_scorer._registrable_domain",
        lambda url: "acme.com" if "acme" in url.lower() else "other.com",
    )
    c = _company()
    assert _matches_exclusion_list(c, ["acme.com"])
    assert _matches_exclusion_list(c, ["https://www.ACME.com/products"])
    assert _matches_exclusion_list(c, ["Acme Inc"])
    assert _matches_exclusion_list(
        _company(linkedin="https://linkedin.com/company/acme-co"),
        ["linkedin.com/company/Acme-Co"])
    assert not _matches_exclusion_list(c, ["other.com", "Different Corp"])
    assert not _matches_exclusion_list(c, [])
    assert not _matches_exclusion_list(c, None)


def test_fit_gate_zeroes_excluded_company(monkeypatch):
    monkeypatch.setattr(
        "qualification.scoring.lead_scorer._registrable_domain",
        lambda url: "acme.com" if "acme" in url.lower() else "other.com",
    )
    ok, reason = _run_competition_binary_fit_checks(
        _company(), _icp(excluded_companies=["acme.com"]))
    assert not ok and "exclusion list" in reason
    ok2, _ = _run_competition_binary_fit_checks(
        _company(), _icp(excluded_companies=["other.com"]))
    assert ok2


def test_company_fit_contract_matches_model_owned_v1():
    identity = company_fit_decision_contract_identity()
    assert identity == {
        "contract_id": "company-fit-decision:v1",
        "outcomes": ["match", "mismatch", "unavailable"],
        "precedence": ["mismatch", "unavailable", "match"],
        "passing_outcome": "match",
        "required_dimensions": [
            "identity",
            "employee_size",
            "industry",
            "geography",
        ],
        "conditional_dimensions": ["stage"],
    }
    assert aggregate_company_fit_decisions(
        {"identity": "unavailable", "industry": "mismatch"}
    ) == COMPANY_FIT_MISMATCH


def test_company_fit_result_truthiness_is_match_only():
    assert isinstance(company_fit_match("verified"), tuple)
    assert company_fit_match("verified") == (True, "verified")
    assert bool(company_fit_match("verified")) is True
    assert bool(company_fit_mismatch("conflict")) is False
    assert bool(company_fit_unavailable("provider outage")) is False


def test_company_fit_result_copy_deepcopy_and_pickle_preserve_named_state():
    result = company_fit_match("verified", details={"nested": {"value": 1}})
    cloned = copy.copy(result)
    deep_cloned = copy.deepcopy(result)
    restored = pickle.loads(pickle.dumps(result))

    for candidate in (cloned, deep_cloned, restored):
        assert candidate == (True, "verified")
        assert candidate.decision == COMPANY_FIT_MATCH
        assert candidate.details == {"nested": {"value": 1}}
        candidate.details["nested"]["value"] = 2
        assert result.details["nested"]["value"] == 1


def test_company_identity_receipt_matches_model_contract_shape():
    receipt = evaluate_company_identity(
        submitted_name="Acme Inc.",
        submitted_website="https://acme.example",
        submitted_linkedin="https://linkedin.com/company/acme",
        observed_name="Acme",
        observed_website="https://www.acme.example",
        observed_linkedin="https://www.linkedin.com/company/acme/",
        evidence_source="company_homepage",
    )
    assert receipt == {
        "decision": "match",
        "reason_code": "verifier_accepted",
        "submitted_name": "acme",
        "submitted_domain": "acme.example",
        "submitted_linkedin_slug": "acme",
        "observed_name": "acme",
        "observed_domain": "acme.example",
        "observed_linkedin_slug": "acme",
        "evidence_source": "company_homepage",
    }


def test_public_scorer_uses_shared_employee_stage_and_exclusion_gates(monkeypatch):
    import qualification.scoring.lead_scorer as scorer

    async def prechecks(*_args, **_kwargs):
        return company_fit_match()

    async def must_not_verify(*_args, **_kwargs):
        raise AssertionError("binary mismatch must stop before identity verification")

    monkeypatch.setattr(scorer, "run_company_zero_checks", prechecks)
    monkeypatch.setattr(scorer, "verify_company_exists", must_not_verify)
    monkeypatch.setattr(
        scorer,
        "_registrable_domain",
        lambda url: "acme.com" if "acme" in url.lower() else "other.com",
    )

    cases = [
        (_company(), _icp(employee_count="201-500"), "employee_size"),
        (
            _company().model_copy(update={"company_stage": "Seed"}),
            _icp(company_stage="Series A"),
            "stage",
        ),
        (_company(), _icp(excluded_companies=["acme.com"]), "exclusion list"),
    ]
    for company, icp, expected in cases:
        result = asyncio.run(scorer.score_company(company, icp, 0.0, 1.0, set()))
        assert result.final_score == 0
        assert expected in (result.failure_reason or "")

    assert _run_company_binary_fit_checks is not None


def test_public_scorer_passes_submitted_linkedin_to_identity_verifier(monkeypatch):
    import qualification.scoring.lead_scorer as scorer

    async def prechecks(*_args, **_kwargs):
        return company_fit_match()

    async def identity(*_args, **kwargs):
        assert kwargs["company_linkedin"] == "https://linkedin.com/company/acme"
        return company_fit_unavailable("stop after caller contract check")

    monkeypatch.setattr(scorer, "run_company_zero_checks", prechecks)
    monkeypatch.setattr(scorer, "verify_company_exists", identity)
    result = asyncio.run(
        scorer.score_company(
            _company(linkedin="https://linkedin.com/company/acme"),
            _icp(),
            0.0,
            1.0,
            set(),
        )
    )
    assert result.final_score == 0
    assert result.failure_reason.startswith("Company fit unavailable:")


def test_reverify_decision_semantics():
    # An explicit contradiction is a mismatch.
    assert _reverify_decision({"attribute_satisfied": False}, "attr", "").decision == COMPANY_FIT_MISMATCH
    assert _reverify_decision({"stage_matches": False}, "", "series a").decision == COMPANY_FIT_MISMATCH
    # Every required dimension must contain an actual Boolean. Missing or junk
    # data is unavailable, never an implicit match.
    assert _reverify_decision({"attribute_satisfied": True}, "attr", "").decision == COMPANY_FIT_MATCH
    assert _reverify_decision({}, "attr", "series a").decision == COMPANY_FIT_UNAVAILABLE
    assert _reverify_decision({"attribute_satisfied": "maybe"}, "attr", "").decision == COMPANY_FIT_UNAVAILABLE
    # dimension not pinned -> its verdict ignored
    assert _reverify_decision({"attribute_satisfied": False}, "", "").decision == COMPANY_FIT_MATCH


def test_reverify_early_exits_without_network():
    async def run(**env):
        with mock.patch.dict(os.environ, env, clear=False):
            return await _llm_reverify_company(_company(), _icp())
    # no attribute and no stage pinned -> no call, pass; when either IS
    # pinned the LLM check is mandatory (no kill-switch exists).
    ok, _ = asyncio.run(run())
    assert ok
    from qualification.scoring import lead_scorer as _ls
    assert not hasattr(_ls, "_scorer_reverify_enabled")


def test_reverify_is_unavailable_without_key():
    async def run():
        env = {k: "" for k in ("OPENROUTER_API_KEY",
                               "QUALIFICATION_OPENROUTER_API_KEY", "OPENROUTER_KEY")}
        with mock.patch.dict(os.environ, env, clear=False):
            return await _llm_reverify_company(
                _company(), _icp(required_attribute="privately held"))
    result = asyncio.run(run())
    assert result.decision == COMPANY_FIT_UNAVAILABLE
    assert "no_openrouter_key" in (result.reason or "")


def test_official_research_lab_scorer_accepts_only_match(monkeypatch):
    import qualification.scoring.lead_scorer as scorer

    async def prechecks(*_args, **kwargs):
        result = company_fit_unavailable("taxonomy provider outage")
        kwargs["gate_receipts"].append(result.receipt("taxonomy_industry"))
        return result

    def must_not_continue(*_args, **_kwargs):
        raise AssertionError("unavailable company fit must stop scoring")

    monkeypatch.setattr(scorer, "run_company_zero_checks", prechecks)
    monkeypatch.setattr(scorer, "_run_company_binary_fit_checks", must_not_continue)
    result = asyncio.run(
        scorer.score_company_competition_intent(
            _company(), _icp(), 0.0, 1.0, set()
        )
    )
    assert result.final_score == 0
    assert result.failure_reason.startswith("Company fit unavailable:")
    receipt = result.verifier_gate_receipts[0]
    assert receipt["decision"] == COMPANY_FIT_UNAVAILABLE
    assert receipt["gate"] == "company_fit"
    assert receipt["supporting_receipts"][0]["decision"] == COMPANY_FIT_UNAVAILABLE


def test_psl_exclusion_matches_registrable_domain_not_neighbor(monkeypatch):
    calls = []

    def registrable(url):
        calls.append(url)
        return "acme.co.uk" if "acme.co.uk" in url else "neighbor.co.uk"

    monkeypatch.setattr(
        "qualification.scoring.lead_scorer._registrable_domain", registrable
    )
    company = _company(website="https://shop.eu.acme.co.uk/path")
    assert _matches_exclusion_list(company, ["https://acme.co.uk"])
    assert not _matches_exclusion_list(company, ["https://neighbor.co.uk"])
    assert len(calls) >= 3


def test_exclusion_name_and_linkedin_are_exact_not_substring(monkeypatch):
    monkeypatch.setattr(
        "qualification.scoring.lead_scorer._registrable_domain",
        lambda url: "acme.com" if "acme.com" in url else "other.com",
    )
    company = _company(
        name="Acme Labs",
        website="https://acme.com",
        linkedin="https://linkedin.com/company/acme-labs",
    )
    assert not _matches_exclusion_list(company, ["Acme"])
    assert not _matches_exclusion_list(
        company, ["https://linkedin.com/company/acme"]
    )
    assert _matches_exclusion_list(company, ["Acme Labs Inc."])


def test_exclusion_matcher_checks_entries_after_the_first_fifty(monkeypatch):
    monkeypatch.setattr(
        "qualification.scoring.lead_scorer._registrable_domain",
        lambda url: "acme.com" if "acme" in url.casefold() else "other.com",
    )
    exclusions = [f"other-{index}.com" for index in range(50)] + ["acme.com"]
    assert _matches_exclusion_list(_company(), exclusions)


def test_web_dimension_tri_state_requires_evidence():
    icp = _icp(company_stage="Series A")
    missing = _reverify_decision(
        {
            "employee_size_matches": True,
            "industry_matches": True,
            "geography_matches": True,
            "stage_matches": True,
        },
        "",
        "series a",
        icp=icp,
    )
    assert missing.decision == COMPANY_FIT_UNAVAILABLE
    conflict = _reverify_decision(
        {
            "observed_employee_count": "201-500",
            "employee_size_matches": False,
            "observed_industry": "Software",
            "industry_matches": True,
            "observed_hq_country": "United States",
            "geography_matches": True,
            "observed_company_stage": "Series A",
            "stage_matches": True,
        },
        "",
        "series a",
        icp=icp,
    )
    assert conflict.decision == COMPANY_FIT_MISMATCH
    assert conflict.details["dimension_decisions"]["employee_size"] == (
        COMPANY_FIT_MISMATCH
    )


@pytest.mark.parametrize(
    ("dimension", "observed_field", "matching", "contradiction", "flag_field"),
    [
        (
            "employee_size",
            "observed_employee_count",
            "51-200",
            "201-500",
            "employee_size_matches",
        ),
        (
            "industry",
            "observed_industry",
            "Software",
            "Manufacturing",
            "industry_matches",
        ),
        (
            "geography",
            "observed_hq_country",
            "United States",
            "Canada",
            "geography_matches",
        ),
        (
            "stage",
            "observed_company_stage",
            "Series A",
            "Series C",
            "stage_matches",
        ),
    ],
)
def test_web_dimension_boolean_must_agree_with_canonical_observation(
    dimension,
    observed_field,
    matching,
    contradiction,
    flag_field,
):
    icp = _icp(company_stage="Series A")
    base = {
        "observed_employee_count": "51-200",
        "employee_size_matches": True,
        "observed_industry": "Software",
        "observed_subindustry": "SaaS",
        "industry_matches": True,
        "observed_hq_country": "United States",
        "geography_matches": True,
        "observed_company_stage": "Series A",
        "stage_matches": True,
    }

    observed_match_flag_false = {
        **base,
        observed_field: matching,
        flag_field: False,
    }
    inconsistent_match = _reverify_decision(
        observed_match_flag_false,
        "",
        "series a",
        icp=icp,
    )
    assert inconsistent_match.details["dimension_decisions"][dimension] == (
        COMPANY_FIT_UNAVAILABLE
    )

    observed_conflict_flag_true = {
        **base,
        observed_field: contradiction,
        flag_field: True,
    }
    inconsistent_conflict = _reverify_decision(
        observed_conflict_flag_true,
        "",
        "series a",
        icp=icp,
    )
    assert inconsistent_conflict.details["dimension_decisions"][dimension] == (
        COMPANY_FIT_UNAVAILABLE
    )

    supported_conflict = _reverify_decision(
        {
            **base,
            observed_field: contradiction,
            flag_field: False,
        },
        "",
        "series a",
        icp=icp,
    )
    assert supported_conflict.details["dimension_decisions"][dimension] == (
        COMPANY_FIT_MISMATCH
    )


def test_web_dimension_matches_require_citations_and_bound_identity():
    company = _company(linkedin="https://linkedin.com/company/acme")
    icp = _icp()
    verdict = {
        "observed_company_name": "Acme",
        "observed_company_website": "https://acme.com/about",
        "observed_company_linkedin": "https://linkedin.com/company/acme",
        "observed_employee_count": "51-200",
        "employee_size_matches": True,
        "observed_industry": "Software",
        "industry_matches": True,
        "observed_hq_country": "United States",
        "geography_matches": True,
    }
    uncited = _reverify_decision(
        verdict,
        "",
        "",
        icp=icp,
        company=company,
    )
    assert uncited.decision == COMPANY_FIT_UNAVAILABLE

    cited = _reverify_decision(
        {
            **verdict,
            "dimension_evidence": {
                dimension: {
                    "url": f"https://evidence.example/{dimension}",
                    "quote": f"Verified {dimension}",
                }
                for dimension in ("employee_size", "industry", "geography")
            },
        },
        "",
        "",
        icp=icp,
        company=company,
    )
    assert cited.decision == COMPANY_FIT_MATCH


def test_web_geography_rejects_state_conflict_and_accepts_state_match():
    icp = _icp(country="United States", geography="California")
    base = {
        "observed_employee_count": "51-200",
        "employee_size_matches": True,
        "observed_industry": "Software",
        "industry_matches": True,
        "observed_hq_country": "United States",
        "geography_matches": True,
    }
    conflict = _reverify_decision(
        {**base, "observed_hq_state": "New York"},
        "",
        "",
        icp=icp,
    )
    assert conflict.decision == COMPANY_FIT_UNAVAILABLE
    assert conflict.details["dimension_decisions"]["geography"] == (
        COMPANY_FIT_UNAVAILABLE
    )

    match = _reverify_decision(
        {**base, "observed_hq_state": "California"},
        "",
        "",
        icp=icp,
    )
    assert match.decision == COMPANY_FIT_MATCH


def test_company_identity_does_not_remove_leading_legal_looking_name_terms():
    common = {
        "submitted_website": "https://example.com",
        "submitted_linkedin": "https://linkedin.com/company/example",
        "observed_website": "https://www.example.com/about",
        "observed_linkedin": "https://www.linkedin.com/company/example/",
        "evidence_source": "company_homepage",
    }
    assert evaluate_company_identity(
        submitted_name="Group Nine Media",
        observed_name="Nine Media",
        **common,
    )["decision"] == COMPANY_FIT_MISMATCH
    assert evaluate_company_identity(
        submitted_name="AG Grid",
        observed_name="Grid",
        **common,
    )["decision"] == COMPANY_FIT_MISMATCH


def test_shared_verifier_persists_complete_dimension_receipt(monkeypatch):
    import qualification.scoring.lead_scorer as scorer

    async def prechecks(*_args, **_kwargs):
        return company_fit_match()

    async def identity(*_args, **_kwargs):
        return company_fit_match(
            "identity verified",
            details={"identity": {"decision": "match", "observed_name": "acme"}},
        )

    async def web(*_args, **kwargs):
        assert kwargs["require_company_fit_dimensions"] is True
        return company_fit_match(
            "web verified",
            details={
                "dimension_decisions": {
                    "employee_size": "match",
                    "industry": "match",
                    "geography": "match",
                    "stage": "match",
                },
                "required_attribute_decision": "match",
                "identity_decision": "match",
                "identity_receipt": {
                    "decision": "match",
                    "reason_code": "verifier_accepted",
                    "submitted_name": "acme",
                    "submitted_domain": "acme.com",
                    "submitted_linkedin_slug": "acme",
                    "observed_name": "acme",
                    "observed_domain": "acme.com",
                    "observed_linkedin_slug": "acme",
                    "evidence_source": "company_web_reverification",
                },
                "dimension_evidence": {
                    dimension: {
                        "url": f"https://evidence.example/{dimension}",
                        "quote": f"Verified {dimension}",
                    }
                    for dimension in ("employee_size", "industry", "geography")
                },
                "provider_observations": {
                    "observed_employee_count": "51-200",
                    "observed_industry": "Software",
                    "observed_hq_country": "United States",
                },
            },
        )

    monkeypatch.setattr(scorer, "run_company_zero_checks", prechecks)
    monkeypatch.setattr(scorer, "verify_company_exists", identity)
    monkeypatch.setattr(scorer, "_llm_reverify_company", web)
    result = asyncio.run(
        _verify_company_fit(
            _company(linkedin="https://linkedin.com/company/acme"),
            _icp(company_stage=""),
            0.0,
            1.0,
            set(),
            require_https_transport=True,
        )
    )
    receipt = result.receipt("company_fit")
    assert result.decision == COMPANY_FIT_MATCH
    assert receipt["company_fit_dimensions"] == {
        "identity": "match",
        "employee_size": "match",
        "industry": "match",
        "geography": "match",
        "stage": "match",
    }
    assert receipt["company_fit_stage_required"] is False
    assert set(receipt["dimension_evidence"]) >= {
        "identity",
        "employee_size",
        "industry",
        "geography",
        "stage",
    }


def test_homepage_unavailable_can_be_rescued_by_complete_web_receipt(monkeypatch):
    import qualification.scoring.lead_scorer as scorer

    async def prechecks(*_args, **_kwargs):
        return company_fit_match()

    async def homepage(*_args, **_kwargs):
        return company_fit_unavailable("homepage lacks LinkedIn binding")

    async def web(*_args, **_kwargs):
        return company_fit_match(
            "complete independent web receipt",
            details={
                "dimension_decisions": {
                    "employee_size": "match",
                    "industry": "match",
                    "geography": "match",
                    "stage": "match",
                },
                "required_attribute_decision": "match",
                "identity_decision": "match",
                "identity_receipt": {
                    "decision": "match",
                    "submitted_name": "acme",
                    "submitted_domain": "acme.com",
                    "submitted_linkedin_slug": "acme",
                    "observed_name": "acme",
                    "observed_domain": "acme.com",
                    "observed_linkedin_slug": "acme",
                    "evidence_source": "company_web_reverification",
                },
                "dimension_evidence": {
                    dimension: {
                        "url": f"https://evidence.example/{dimension}",
                        "quote": f"Verified {dimension}",
                    }
                    for dimension in ("employee_size", "industry", "geography")
                },
            },
        )

    monkeypatch.setattr(scorer, "run_company_zero_checks", prechecks)
    monkeypatch.setattr(scorer, "verify_company_exists", homepage)
    monkeypatch.setattr(scorer, "_llm_reverify_company", web)
    result = asyncio.run(
        _verify_company_fit(
            _company(linkedin="https://linkedin.com/company/acme"),
            _icp(),
            0.0,
            1.0,
            set(),
            require_https_transport=True,
        )
    )
    assert result.decision == COMPANY_FIT_MATCH
    identity = result.details["dimension_evidence"]["identity"]
    assert identity["homepage_identity_decision"] == COMPANY_FIT_UNAVAILABLE
    assert identity["web_identity_decision"] == COMPANY_FIT_MATCH


def test_homepage_unavailable_remains_unavailable_without_complete_web_receipt(monkeypatch):
    import qualification.scoring.lead_scorer as scorer

    async def prechecks(*_args, **_kwargs):
        return company_fit_match()

    async def homepage(*_args, **_kwargs):
        return company_fit_unavailable("homepage lacks LinkedIn binding")

    async def web(*_args, **_kwargs):
        return company_fit_unavailable("web provider unavailable")

    monkeypatch.setattr(scorer, "run_company_zero_checks", prechecks)
    monkeypatch.setattr(scorer, "verify_company_exists", homepage)
    monkeypatch.setattr(scorer, "_llm_reverify_company", web)
    result = asyncio.run(
        _verify_company_fit(
            _company(linkedin="https://linkedin.com/company/acme"),
            _icp(),
            0.0,
            1.0,
            set(),
            require_https_transport=True,
        )
    )
    assert result.decision == COMPANY_FIT_UNAVAILABLE
    identity = result.details["dimension_evidence"]["identity"]
    assert identity["homepage_identity_decision"] == COMPANY_FIT_UNAVAILABLE
    assert identity["web_identity_decision"] == COMPANY_FIT_UNAVAILABLE


def test_public_and_research_lab_use_the_same_shared_verifier(monkeypatch):
    import qualification.scoring.lead_scorer as scorer

    calls = []

    async def shared(*_args, **kwargs):
        calls.append(kwargs["require_https_transport"])
        return company_fit_unavailable(
            "provider evidence missing",
            details={
                "company_fit_decision": "unavailable",
                "company_fit_dimensions": {
                    "identity": "unavailable",
                    "employee_size": "unavailable",
                    "industry": "unavailable",
                    "geography": "unavailable",
                    "stage": "match",
                },
                "company_fit_stage_required": False,
                "dimension_evidence": {},
            },
        )

    monkeypatch.setattr(scorer, "_verify_company_fit", shared)
    public = asyncio.run(
        scorer.score_company(_company(), _icp(), 0.0, 1.0, set())
    )
    research = asyncio.run(
        scorer.score_company_competition_intent(
            _company(), _icp(), 0.0, 1.0, set()
        )
    )
    assert calls == [True, True]
    assert public.final_score == research.final_score == 0
    for breakdown in (public, research):
        receipt = breakdown.verifier_gate_receipts[0]
        assert receipt["gate"] == "company_fit"
        assert set(receipt["company_fit_dimensions"]) == {
            "identity",
            "employee_size",
            "industry",
            "geography",
            "stage",
        }


def test_official_fit_rejects_proven_industry_conflict_even_in_shadow(monkeypatch):
    import qualification.scoring.lead_scorer as scorer

    monkeypatch.setenv("RESEARCH_LAB_TAXONOMY_INDUSTRY_GATE", "shadow")

    async def prechecks(*_args, **_kwargs):
        return company_fit_match()

    async def must_not_fetch(*_args, **_kwargs):
        raise AssertionError("explicit industry conflict must stop paid work")

    monkeypatch.setattr(scorer, "run_company_zero_checks", prechecks)
    monkeypatch.setattr(scorer, "verify_company_exists", must_not_fetch)
    result = asyncio.run(
        _verify_company_fit(
            _company().model_copy(update={"industry": "Manufacturing"}),
            _icp(industry="Software"),
            0.0,
            1.0,
            set(),
            require_https_transport=True,
        )
    )
    assert result.decision == COMPANY_FIT_MISMATCH
    assert result.details["company_fit_dimensions"]["industry"] == (
        COMPANY_FIT_MISMATCH
    )
