"""Fulfillment Tier-1 industry gate: deterministic ladder + LLM deferral.

The industry check rejected any lead whose provider industry description was
not string-equal to a taxonomy label — multi-clause descriptions
("Robotics; Industrial Automation and Autonomous Systems") could never pass
and viable candidates were lost. The gate now mirrors the sub-industry
ladder: exact -> equivalence class -> containment -> Tier 1.5 semantic LLM.
"""

from types import SimpleNamespace

import pytest

from gateway.fulfillment import icp_checks
from gateway.fulfillment.icp_checks import (
    semantic_industry_match,
    sub_industry_deterministic_status,
    tier1_check,
)


def _objects(lead_industry="", lead_sub="", icp_inds=None, icp_subs=None):
    lead = SimpleNamespace(
        role="",
        role_type="",
        business="Example Company",
        industry=lead_industry,
        sub_industry=lead_sub,
        company_hq_country="",
        country="",
        employee_count="",
    )
    lead_output = SimpleNamespace(role="", business=lead.business)
    icp = SimpleNamespace(
        industry=icp_inds or [],
        sub_industry=icp_subs or [],
        excluded_companies=[],
        target_role_types=[],
        target_roles=[],
        target_seniority="",
        company_country=[],
        contact_country=[],
        employee_count=[],
        company_stage="",
    )
    return lead, lead_output, icp


# ---------------------------------------------------------------------------
# Deterministic ladder
# ---------------------------------------------------------------------------

def test_exact_industry_match_passes():
    lead, lead_output, icp = _objects("Robotics", icp_inds=["Robotics"])
    assert tier1_check(lead, lead_output, icp, set()) is None


def test_multi_clause_description_containing_label_passes_deterministically():
    # The reported false-negative class: provider descriptions embedding the
    # taxonomy label must pass without an LLM call.
    lead, lead_output, icp = _objects(
        "Robotics; Industrial Automation and Autonomous Systems",
        icp_inds=["Robotics"],
    )
    assert tier1_check(lead, lead_output, icp, set()) is None


def test_containment_is_case_insensitive():
    lead, lead_output, icp = _objects(
        "ROBOTICS and industrial automation", icp_inds=["Robotics"]
    )
    assert tier1_check(lead, lead_output, icp, set()) is None


def test_semicolon_separates_multiple_industries():
    # ";" is a list separator in provider data: each listed industry is
    # checked independently, wherever it sits in the list.
    lead, lead_output, icp = _objects(
        "Industrial Automation and Autonomous Systems; Robotics",
        icp_inds=["Robotics"],
    )
    assert tier1_check(lead, lead_output, icp, set()) is None


def test_comma_separated_industries_also_split():
    lead, lead_output, icp = _objects(
        "Consumer Electronics, Hardware", icp_inds=["Hardware"]
    )
    assert tier1_check(lead, lead_output, icp, set()) is None


def test_component_hits_equivalence_map():
    # "Software ≈ Information Technology" via the equivalence map: the whole
    # multi-value string is not a map key, but the split component is.
    lead, lead_output, icp = _objects(
        "Software; Developer Tooling Platforms",
        icp_inds=["Information Technology"],
    )
    assert tier1_check(lead, lead_output, icp, set()) is None


def test_unrelated_industry_defers_to_llm_instead_of_hard_reject():
    # Messy provider description (not a taxonomy label) -> LLM judges it.
    lead, lead_output, icp = _objects(
        "Financial technology and payments infrastructure",
        icp_inds=["Fintech"],
    )
    assert tier1_check(lead, lead_output, icp, set()) == "industry_needs_llm"


def test_clean_taxonomy_label_mismatch_still_rejects_instantly():
    # A recognized taxonomy label naming a different industry is a true
    # mismatch: no LLM call, same instant reject as the original gate.
    lead, lead_output, icp = _objects(
        "Advertising", icp_inds=["Financial Services"]
    )
    assert tier1_check(lead, lead_output, icp, set()) == "industry_mismatch"


def test_clean_label_components_all_mismatching_reject_instantly():
    lead, lead_output, icp = _objects(
        "Advertising; Design", icp_inds=["Financial Services"]
    )
    assert tier1_check(lead, lead_output, icp, set()) == "industry_mismatch"


def test_mixed_clean_and_messy_components_defer_to_llm():
    # One component is not a recognized label -> the value is messy -> LLM.
    lead, lead_output, icp = _objects(
        "Advertising; Programmatic Adtech Infrastructure",
        icp_inds=["Financial Services"],
    )
    assert tier1_check(lead, lead_output, icp, set()) == "industry_needs_llm"


def test_empty_icp_industry_skips_the_gate():
    lead, lead_output, icp = _objects("Anything At All", icp_inds=[])
    assert tier1_check(lead, lead_output, icp, set()) is None


def test_sub_industry_rung_still_runs_after_exact_industry_pass():
    lead, lead_output, icp = _objects(
        "Robotics",
        lead_sub="Warehouse Automation",
        icp_inds=["Robotics"],
        icp_subs=["Surgical Robotics"],
    )
    assert tier1_check(lead, lead_output, icp, set()) == "sub_industry_needs_llm"


def test_sub_industry_deterministic_status_matches_tier1_behavior():
    lead, _, icp = _objects(
        "Robotics",
        lead_sub="Industrial Automation Systems",
        icp_inds=["Robotics"],
        icp_subs=["Industrial Automation"],
    )
    # Containment passes the shared helper (used by the Tier 1.5 bridge after
    # an industry LLM pass, so industry deferral cannot skip this rung).
    assert sub_industry_deterministic_status(lead, icp) is None
    lead.sub_industry = "Surgical Robotics"
    assert sub_industry_deterministic_status(lead, icp) == "sub_industry_needs_llm"


# ---------------------------------------------------------------------------
# Tier 1.5 semantic matcher
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_semantic_industry_match_parses_llm_verdict(monkeypatch):
    monkeypatch.setattr(icp_checks, "_get_openrouter_key", lambda: "key")

    async def fake_llm(prompt, key):
        assert "Financial technology" in prompt
        assert '"Fintech"' in prompt
        return {"match": True, "matched_industry": "Fintech"}

    monkeypatch.setattr(icp_checks, "_llm_call", fake_llm)
    matched, label = await semantic_industry_match(
        "Financial technology and payments infrastructure", ["Fintech"]
    )
    assert matched is True
    assert label == "Fintech"


@pytest.mark.asyncio
async def test_semantic_industry_match_fails_closed(monkeypatch):
    # No key -> no match; LLM failure -> no match.
    monkeypatch.setattr(icp_checks, "_get_openrouter_key", lambda: "")
    assert await semantic_industry_match("X", ["Y"]) == (False, "")

    monkeypatch.setattr(icp_checks, "_get_openrouter_key", lambda: "key")

    async def broken_llm(prompt, key):
        return None

    monkeypatch.setattr(icp_checks, "_llm_call", broken_llm)
    assert await semantic_industry_match("X", ["Y"]) == (False, "")
