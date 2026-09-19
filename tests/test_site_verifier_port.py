"""Tests for the site-verifier improvements ported into the lab verifier.

Covers the four ported surfaces and the precision-first company-fit contract.
The official default enforces deterministic industry fit. Explicit shadow and
disabled modes remain available for controlled observation.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import os
import typing

import pytest

import leadpoet_verifier.industry_taxonomy as taxonomy
from leadpoet_verifier.industry_fit import b2b_saas_evidence, industry_fit
import qualification.scoring.pre_checks as pre_checks
from gateway.qualification.models import (
    CompanyOutput,
    ICPPrompt,
    IntentSignal,
    IntentSignalSource,
)


# ---------------------------------------------------------------------------
# Production runtime compatibility
# ---------------------------------------------------------------------------


def test_new_verifier_runtime_annotations_evaluate_on_python39() -> None:
    modules = (
        "leadpoet_verifier.industry_fit",
        "leadpoet_verifier.industry_taxonomy",
        "leadpoet_verifier.identity.normalization",
    )

    for module_name in modules:
        module = importlib.import_module(module_name)
        for value in vars(module).values():
            if (
                (inspect.isfunction(value) or inspect.isclass(value))
                and value.__module__ == module_name
            ):
                typing.get_type_hints(value)


# ---------------------------------------------------------------------------
# Canonical taxonomy matcher (ported industry_taxonomy.py)
# ---------------------------------------------------------------------------


def test_taxonomy_exact_parent_and_subindustry_accepts() -> None:
    ok, detail = taxonomy.leadpoet_taxonomy_match("Software", "Software", "SaaS")
    assert ok is True
    assert detail["candidate_parent_consistent"] is True


def test_taxonomy_conflicting_parent_rejects() -> None:
    # A candidate labeled with a different canonical parent must be rejected,
    # not fuzzy-passed (the lab's old gate would have passed this through).
    ok, _ = taxonomy.leadpoet_taxonomy_match("Software", "Manufacturing", "")
    assert ok is False


def test_taxonomy_unknown_provider_label_delegates() -> None:
    # Provider-specific labels outside the taxonomy return None so bounded
    # semantic concepts (not string luck) decide.
    ok, _ = taxonomy.leadpoet_taxonomy_match(
        "Software", "Computer Software Vendors", ""
    )
    assert ok is None


def test_fuel_cell_concepts_recognized() -> None:
    # Commit 3bcfca53: fuel-cell / electrolyzer / hydrogen are
    # energy-infrastructure, so buyer requests match provider labels.
    assert "energy_infrastructure" in taxonomy.industry_concepts("fuel cell systems")
    assert "energy_infrastructure" in taxonomy.industry_concepts(
        "hydrogen electrolyzer manufacturer"
    )


def test_broad_concept_suppressed_as_modifier() -> None:
    # "logistics software" must not match a plain software vendor via the
    # broad 'software' concept — broad concepts count only standalone.
    concepts, suppressed = taxonomy.requested_industry_concepts("logistics software")
    assert "software" not in concepts
    assert "software" in suppressed


# ---------------------------------------------------------------------------
# industry_fit + B2B SaaS evidence (ported adapter matchers)
# ---------------------------------------------------------------------------


def test_industry_fit_taxonomy_authoritative() -> None:
    passed, detail = industry_fit("Software", "Software", "SaaS")
    assert passed is True
    assert detail["match_strategy"] == "leadpoet_taxonomy"


def test_industry_fit_canonical_concept_for_provider_labels() -> None:
    passed, detail = industry_fit(
        "Cybersecurity", "Computer & Network Security", ""
    )
    assert passed is True
    assert detail["match_strategy"] in {"canonical_concept", "leadpoet_taxonomy"}


def test_industry_fit_rejects_unrelated() -> None:
    passed, detail = industry_fit("Software", "Food & Beverages", "")
    assert passed is False


def test_b2b_saas_needs_buyer_and_product_evidence() -> None:
    # Product label alone is NOT proof of B2B: no customer signal -> fail.
    result = b2b_saas_evidence("Software Development", "", "", "")
    assert result["passed"] is False
    # Grounding requires label- or quote-level evidence: the frozen evidence
    # quote showing a SaaS product sold to enterprises passes.
    result = b2b_saas_evidence(
        "Software Development",
        "",
        "A platform for enterprises to manage payroll",
        "Acme provides a SaaS platform for enterprises to manage payroll",
    )
    assert result["passed"] is True
    assert "saas" in result["product_signals"]
    assert result["source_grounded"] is True
    # A description alone corroborates but cannot ground (deliberate: the
    # description can merely repeat the buyer's request).
    result = b2b_saas_evidence(
        "Software Development", "", "A SaaS platform for enterprises", ""
    )
    assert result["passed"] is False


def test_b2b_saas_service_only_rejected_without_owned_software() -> None:
    result = b2b_saas_evidence(
        "IT Services",
        "",
        "A consulting agency serving businesses",
        "",
    )
    assert result["passed"] is False
    assert result["service_only_signals"]


def _company(industry: str = "Food & Beverages", sub: str = "") -> CompanyOutput:
    return CompanyOutput(
        company_name="Acme Corp",
        company_website="https://acmecorp.io",
        industry=industry,
        sub_industry=sub,
        employee_count="51-200",
        country="United States",
        description="",
        intent_signals=[
            IntentSignal(
                source=IntentSignalSource.NEWS,
                description="Announced expansion of data engineering team",
                url="https://technews.io/acme-hiring",
                date="2026-07-01",
                snippet="Acme Corp announced it is expanding its data engineering team",
                matched_icp_signal=0,
            )
        ],
    )


def _icp(industry: str = "Software") -> ICPPrompt:
    return ICPPrompt(
        icp_id="icp-test-1",
        industry=industry,
        sub_industry="",
        employee_count="51-200",
        company_stage="",
        country="United States",
        geography="United States",
        product_service="B2B software tools",
        intent_signals=["hiring for data engineers"],
    )


# ---------------------------------------------------------------------------
# Identity resolution (ported identity/ package) — pure policy basics
# ---------------------------------------------------------------------------


def test_identity_psl_registrable_domain() -> None:
    from leadpoet_verifier.identity.normalization import normalize_host

    parts = normalize_host("app.acme.co.uk")
    # PSL-aware: registrable domain is acme.co.uk, NOT co.uk (the naive
    # rsplit('.') parsing elsewhere would get this wrong).
    assert parts.registrable_domain == "acme.co.uk"
    assert parts.public_suffix == "co.uk"


def test_identity_linkedin_url_canonicalization() -> None:
    from leadpoet_verifier.identity.normalization import (
        normalize_linkedin_company_url,
    )

    a = normalize_linkedin_company_url(
        "https://www.linkedin.com/company/acme-corp/?utm=x"
    )
    b = normalize_linkedin_company_url("https://linkedin.com/company/acme-corp")
    assert a == b


# ---------------------------------------------------------------------------
# lead_scorer hardening ports: fail-closed buckets + provider-outage fairness
# ---------------------------------------------------------------------------


def test_icp_buckets_fail_closed_on_malformed() -> None:
    from qualification.scoring.lead_scorer import _normalize_icp_employee_buckets

    # Real bands verify; malformed requirements are UNVERIFIED (cannot match)
    # instead of silently disabling the size gate (the old fail-open hole).
    assert _normalize_icp_employee_buckets(["11-50", "51-200"]) == (
        {"11-50", "51-200"},
        True,
    )
    assert _normalize_icp_employee_buckets("about 500") == (set(), False)
    assert _normalize_icp_employee_buckets(["11-50", "garbage"]) == (set(), False)
    # Proven against live data: 560 production ICPs across 37 daily sets all
    # parse verified, so this changes nothing on real benchmarks.


@pytest.mark.asyncio
async def test_provider_outage_records_verifier_error_not_content_reject(
    monkeypatch,
) -> None:
    # A three-stage result that failed for PROVIDER reasons (llm_error /
    # stage3_llm_error) must be recorded as rejected_verifier_error so the
    # evaluator's fail-open path (infrastructure failure != falsified intent)
    # actually triggers; content rejects keep rejected_three_stage.
    import qualification.scoring.lead_scorer as ls
    import qualification.scoring.intent_verification_three_stage as t3

    def stub_result(reason, s3_status, decision="reject"):
        async def fake_verify(client, **kwargs):
            return {
                "client_ready": False,
                "decision": decision,
                "rejection_reason": reason,
                "stage1": {"status": "review"},
                "stage3": {"status": s3_status},
                "scrape": {"statuses": [], "result_count": 1},
                "verdict": {},
            }
        return fake_verify

    signal = IntentSignal(
        source=IntentSignalSource.NEWS,
        description="Announced expansion of data engineering team",
        url="https://technews.io/acme-hiring",
        date="2026-07-01",
        snippet="Acme Corp announced it is expanding its data engineering team",
        matched_icp_signal=0,
    )
    icp = _icp("Software")

    async def run_with(reason, s3_status):
        verdicts: list = []
        monkeypatch.setattr(t3, "verify_three_stage", stub_result(reason, s3_status))
        score, *_ = await ls._score_single_intent_signal(
            signal,
            icp,
            None,
            "Acme Corp",
            company_website="https://acmecorp.io",
            api_key="test-key",
            llm_only_intent_gate=True,
            verdict_out=verdicts,
        )
        assert score == 0.0
        return verdicts[-1]["decision"]

    # Provider outage -> verifier_error (evaluator fails open).
    assert await run_with("stage3_llm_error:timeout", "llm_error") == (
        "rejected_verifier_error"
    )
    # Genuine content rejection -> unchanged classification.
    assert await run_with("stage3_contradicted", "contradicted") == (
        "rejected_three_stage"
    )


# ---------------------------------------------------------------------------
# PR-28 audit fix: country alias + unknown-code handling
# ---------------------------------------------------------------------------


def test_country_the_prefixed_official_names_resolve() -> None:
    from qualification.scoring.pre_checks import check_country_match

    assert check_country_match("The Bahamas", "BS").passed is True
    assert check_country_match("BS", "The Bahamas").passed is True
    assert check_country_match("The Gambia", "GM").passed is True
    assert check_country_match("GM", "The Gambia").passed is True
    assert check_country_match("The Netherlands", "Netherlands").passed is True


def test_country_unknown_code_fails_closed_prose_still_defers() -> None:
    from qualification.scoring.pre_checks import check_country_match

    # An unknown code-shaped geography must NOT accept everything.
    assert check_country_match("Brazil", "ZZ").passed is False
    assert check_country_match("ZZ", "Brazil").passed is False
    # Multi-region prose keeps the documented deferral contract.
    assert check_country_match("Germany", "EMEA").passed is True
    assert check_country_match("Japan", "APAC").passed is True


def test_country_eu_shorthand_expands_to_europe() -> None:
    from qualification.scoring.pre_checks import check_country_match

    assert check_country_match("Germany", "EU").passed is True
    assert check_country_match("France", "EU").passed is True
    assert check_country_match("Brazil", "EU").passed is False


@pytest.mark.asyncio
async def test_gate_receipts_persist_into_breakdown_end_to_end(monkeypatch) -> None:
    # The score detail must survive in the Arena's persisted breakdown.
    from qualification.scoring.lead_scorer import (
        score_company_competition_intent,
    )

    # Canonical conflict zeroes deterministically before provider calls.
    breakdown = await score_company_competition_intent(
        _company("Manufacturing"),
        _icp("Software"),
        run_cost_usd=0.0,
        run_time_seconds=1.0,
        seen_companies=set(),
    )
    assert breakdown.final_score == 0
    assert "submitted company fit conflicts" in (breakdown.failure_reason or "")
    receipts = breakdown.verifier_gate_receipts
    assert receipts and receipts[0]["gate"] == "company_fit"
    assert receipts[0]["decision"] == "mismatch"
    assert receipts[0]["company_fit_dimensions"]["industry"] == "mismatch"
    # Round-trips through the model layer (what the evaluator serializes).
    dumped = breakdown.model_dump()
    assert dumped["verifier_gate_receipts"][0]["decision"] == "mismatch"


@pytest.mark.asyncio
async def test_clean_pass_carries_no_receipt(monkeypatch) -> None:
    # Payload discipline: a trivial deterministic pass must NOT attach an
    # audit receipt — otherwise every persisted breakdown would grow without
    # audit value.
    receipts: list = []
    result = await pre_checks.run_company_zero_checks(
        _company("Software", "SaaS"),
        run_time_seconds=1.0,
        seen_companies=set(),
        gate_receipts=receipts,
    )
    assert result.passed is True and result.reason is None
    assert receipts == []  # nothing to audit on a clean deterministic pass
