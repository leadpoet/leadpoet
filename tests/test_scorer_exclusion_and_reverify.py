"""Scorer-side exclusion enforcement + web re-verification decision logic."""

import asyncio
import os
from unittest import mock

from gateway.qualification.models import CompanyOutput, ICPPrompt
from qualification.scoring.lead_scorer import (
    _llm_reverify_company,
    _matches_exclusion_list,
    _reverify_decision,
    _run_autoresearch_binary_fit_checks,
)


def _company(name="Acme", website="https://acme.com", linkedin=""):
    return CompanyOutput(
        company_name=name, company_website=website, company_linkedin=linkedin,
        industry="Software", employee_count="51-200", country="United States",
        intent_signals=[{"description": "raised", "source": "news",
                         "url": "https://n.example.com/a", "date": "2026-07-01",
                         "snippet": "Acme raised a round this month."}])


def _icp(**over):
    base = dict(icp_id="t", prompt="p", industry="Software", sub_industry="SaaS",
                employee_count="11-50|51-200", company_stage="",
                geography="United States", product_service="x")
    base.update(over)
    return ICPPrompt(**base)


def test_exclusion_matcher_by_domain_linkedin_name():
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


def test_fit_gate_zeroes_excluded_company():
    ok, reason = _run_autoresearch_binary_fit_checks(
        _company(), _icp(excluded_companies=["acme.com"]))
    assert not ok and "exclusion list" in reason
    ok2, _ = _run_autoresearch_binary_fit_checks(
        _company(), _icp(excluded_companies=["other.com"]))
    assert ok2


def test_reverify_decision_semantics():
    # affirmative false -> zero
    assert _reverify_decision({"attribute_satisfied": False}, "attr", "")[0] is False
    assert _reverify_decision({"stage_matches": False}, "", "series a")[0] is False
    # true / missing / junk -> keep (fail-open)
    assert _reverify_decision({"attribute_satisfied": True}, "attr", "")[0] is True
    assert _reverify_decision({}, "attr", "series a")[0] is True
    assert _reverify_decision({"attribute_satisfied": "maybe"}, "attr", "")[0] is True
    # dimension not pinned -> its verdict ignored
    assert _reverify_decision({"attribute_satisfied": False}, "", "")[0] is True


def test_reverify_early_exits_without_network():
    async def run(**env):
        with mock.patch.dict(os.environ, env, clear=False):
            return await _llm_reverify_company(_company(), _icp())
    # no attribute and no stage pinned -> no call, pass
    ok, _ = asyncio.run(run())
    assert ok
    # flag off -> pass without call
    ok, _ = asyncio.run(run(RESEARCH_LAB_SCORER_LLM_REVERIFY="0"))
    assert ok


def test_reverify_fail_open_without_key():
    async def run():
        env = {k: "" for k in ("OPENROUTER_API_KEY",
                               "QUALIFICATION_OPENROUTER_API_KEY", "OPENROUTER_KEY")}
        with mock.patch.dict(os.environ, env, clear=False):
            return await _llm_reverify_company(
                _company(), _icp(required_attribute="privately held"))
    ok, _ = asyncio.run(run())
    assert ok  # no key -> fail-open, company kept
