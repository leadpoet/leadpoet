"""Generated ICPs must carry fulfillment-style specific attributes and intents.

The generation prompt was changed to produce specific, descriptive
``product_service`` / ``required_attribute`` value propositions and specific
fulfillment-style ``intent_signal`` sentences (instead of the broad-category
template and the fixed generic-event vocabulary), WITHOUT changing the ICP
schema or the intent-signal count. These tests lock the pipeline guarantees
that make that safe: a specific attribute/intent flows through unchanged, a
free-text intent still resolves to a valid category, and the prompt no longer
instructs the barebone shapes.
"""

import asyncio

import pytest

from gateway.tasks import icp_generator as ig
from gateway.tasks.icp_generator import (
    canonicalize_generated_icp,
    intent_category_for_signal,
    _normalize_intent_signals,
    _required_attribute_for_icp,
)

SPECIFIC_ATTR = (
    "Sells a multi-tenant subscription software platform used by revenue or "
    "operations teams to manage pipeline, deals, or customer workflows"
)
SPECIFIC_INTENT = (
    "Launched a net-new product line or major platform module in the last 12 "
    "months, evidenced by a press release, blog post, or updated product pages"
)


def test_specific_required_attribute_is_preserved_not_templated():
    """A specific LLM-provided required_attribute must pass through unchanged;
    the 'offers or provides X' template only fires when it is empty."""
    kept = _required_attribute_for_icp(
        {"required_attribute": SPECIFIC_ATTR, "product_service": "irrelevant"},
        industry="Software", sub_industry="B2B SaaS",
    )
    assert kept == SPECIFIC_ATTR
    templated = _required_attribute_for_icp(
        {"required_attribute": "", "product_service": "widgets"},
        industry="Software", sub_industry="B2B SaaS",
    )
    assert templated == "The company offers or provides widgets"


def test_canonicalize_preserves_specific_attribute_and_intent():
    icp = {
        "icp_id": "icp_1", "prompt": "test",
        "product_service": "A subscription B2B platform for revenue teams",
        "required_attribute": SPECIFIC_ATTR,
        "intent_signal": SPECIFIC_INTENT,
        "intent_signals": [SPECIFIC_INTENT],
        "employee_count": ["51-200", "201-500"],
        "intent_category": "PRODUCT_LAUNCH",
    }
    out = canonicalize_generated_icp(icp, industry="Software", sub_industry="B2B SaaS")
    assert out["required_attribute"] == SPECIFIC_ATTR   # not overwritten
    assert out["intent_signal"] == SPECIFIC_INTENT      # specific text kept
    assert out["intent_signals"][0] == SPECIFIC_INTENT
    assert out["intent_category"] == "PRODUCT_LAUNCH"


def test_free_text_intents_resolve_to_valid_categories():
    """Fulfillment-style free-text intents still map to a valid category via
    the keyword fallback, so source routing and scoring keep working with
    specific wording. (This fallback is best-effort; the authoritative
    category is the LLM-provided one, honored by canonicalize — see
    test_canonicalize_preserves_specific_attribute_and_intent.)"""
    assert intent_category_for_signal(
        "Announced a Series A or later funding round in the last 12 months"
    ) == "FUNDING"
    assert intent_category_for_signal(
        "Actively hiring for revenue-operations and integration roles per current job postings"
    ) == "HIRING"
    assert intent_category_for_signal(
        "Uses Salesforce, HubSpot, or Segment as their primary CRM"
    ) == "TECHSTACK"
    assert intent_category_for_signal(
        "Released a new product or feature set in the last 12 months, per a press release or changelog"
    ) == "PRODUCT_LAUNCH"


def test_bonus_product_launch_semantics_outrank_generic_technology_nouns():
    """Bonus intents use the deterministic fallback, so explicit event
    semantics must outrank generic words such as ``platform``."""
    bonus_signal = (
        "Launched a major platform capability or workflow expansion in the "
        "last 12 months, per a product page, changelog, or company announcement."
    )
    icp = {
        "icp_id": "icp_1", "prompt": "test",
        "product_service": "x", "required_attribute": "x-attr",
        "intent_signal": "Announced a Series A funding round in the last 12 months",
        "intent_signals": [
            "Announced a Series A funding round in the last 12 months",
            bonus_signal,
        ],
        "employee_count": ["51-200"],
        "intent_category": "FUNDING",
    }
    out = canonicalize_generated_icp(icp, industry="Software", sub_industry="B2B SaaS")
    assert out["intent_category"] == "FUNDING"
    assert out["bonus_intents"] == [
        {
            "intent_signal": bonus_signal,
            "intent_category": "PRODUCT_LAUNCH",
            "intent_max_age_days": 365,
        }
    ]
    assert intent_category_for_signal("Implemented a new CRM platform") == "TECHSTACK"


def test_intent_signal_count_is_unchanged():
    """Specificity is a wording change, not a count change: at most 5 signals
    are kept, exactly as before."""
    many = [f"specific intent number {i} in the last 12 months" for i in range(8)]
    assert len(_normalize_intent_signals(many)) == 5


def test_prompt_no_longer_instructs_barebone_shapes():
    """Source-level guard: the generation prompt must not reinstate the
    broad-category / templated-attribute / fixed-vocabulary instructions, and
    must carry the fulfillment-style specificity + supply guidance."""
    src = ig.__file__
    with open(src, encoding="utf-8") as fh:
        text = fh.read()
    # Barebone instructions removed.
    assert "broad category — NOT a single named tool" not in text
    assert '"required_attribute": "The company offers or provides <product_service' not in text
    # Fulfillment-style + supply guidance present.
    assert "fulfillment style" in text.lower() or "fulfillment-style" in text.lower()
    assert "specific" in text.lower()
    assert "broad enough that many real companies" in text.lower()


def test_prompt_reserves_and_for_requirements_every_company_must_meet(monkeypatch):
    """On 2026-11-09 a ``product_service`` that
    listed "a branded online store and marketplaces" left most
    direct-to-consumer brands unproven, while its ``required_attribute`` asked
    for the store alone. The prompt sent to the author must say how such lists
    are checked, keep "or" alternatives specific, and must not model an
    incidental "and" list in its own examples."""
    monkeypatch.setattr(ig, "OPENROUTER_API_KEY", "")
    prepared = asyncio.run(ig.prepare_icp_probe_request(
        "icp_generation", set_id=20261109, total_icps=10,
    ))
    prompt = "\n".join(
        message["content"] for message in prepared["request"]["messages"]
        if isinstance(message.get("content"), str)
    )
    assert "manage pipeline, deals, and customer workflows" not in prompt
    assert prompt.count("manage pipeline, deals, or customer workflows") == 2
    assert 'write "must provide both A and B" or "must provide all of A, B and C"' in prompt
    assert 'no catch-all such as "or other software"' in prompt
    assert 'ordinary descriptive "and" list' in prompt
    assert 'an alternative in one must not become mandatory in the other' in prompt
    assert 'different, nonconflicting requirements' in prompt
    assert 'Preserve each intended mandatory requirement' in prompt
    assert 'do not turn it into an alternative merely to broaden the ICP' in prompt


def test_prompt_keeps_hiring_intents_provable_from_one_posting(monkeypatch):
    """The 2026-11-05 and 2026-11-11 rounds each had a hiring intent that
    asked for roles "in multiple locations" or "across multiple openings".
    The intent check passed for 23 of 89 rows on those two ICPs, against 32
    of 36 on a plain hiring ICP in the 2026-11-11 round. One posting proves
    one opening, so the prompt must tell the author not to ask for more."""
    monkeypatch.setattr(ig, "OPENROUTER_API_KEY", "")
    prepared = asyncio.run(ig.prepare_icp_probe_request(
        "icp_generation", set_id=20261111, total_icps=10,
    ))
    prompt = "\n".join(
        message["content"] for message in prepared["request"]["messages"]
        if isinstance(message.get("content"), str)
    )
    assert "ONE JOB POSTING MUST BE ENOUGH PROOF" in prompt
    assert 'Name the role families only, joined with "or"' in prompt
    assert 'Do not add\na count or spread condition such as "multiple openings"' in prompt
    assert "one posting proves one opening in one place" in prompt
    # The prompt's own hiring example must stay provable from one posting.
    assert "Actively hiring for platform, integration, or revenue-operations roles" in prompt


def test_prompt_keeps_stage_pinned_and_international_icps_country_wide(monkeypatch):
    """Across the 2026-11-01 to 2026-11-13 rounds (130 ICPs, miner rows only),
    stage-pinned ICPs with a US region, state or city drew a median 33% of
    the rows of the best-filled ICP in their round and 0.53 approved companies
    per submission. Stage-pinned country-wide ICPs drew 60% and 1.34, and
    unpinned country-wide ICPs 84% and 2.15. International ICPs limited to a
    city drew 0.46 approved companies per submission against 1.80 for whole
    countries. The prompt must tell the author not to stack these choices."""
    monkeypatch.setattr(ig, "OPENROUTER_API_KEY", "")
    prepared = asyncio.run(ig.prepare_icp_probe_request(
        "icp_generation", set_id=20261114, total_icps=10,
    ))
    prompt = "\n".join(
        message["content"] for message in prepared["request"]["messages"]
        if isinstance(message.get("content"), str)
    )
    assert "POOL SIZE — DO NOT STACK NARROW CHOICES" in prompt
    assert "{company_goal}" not in prompt
    assert (
        f"Each submission must find up to {ig.COMPANY_GOAL_AVERAGE} different "
        "qualifying companies for every ICP"
    ) in prompt
    assert "A stage-pinned ICP must use a whole-country geography" in prompt
    assert "Never pair a pinned stage with a US region, a state or a city." in prompt
    assert "An international ICP should use the whole country rather than a city" in prompt
    assert "12. Does every stage-pinned ICP use a whole-country geography" in prompt
    # The rule must not reintroduce the per-ICP research step that the
    # five-company viability heuristic replaced.
    assert "This is a heuristic, not a request to research, verify, or list five companies." in prompt


def test_generated_brief_states_structured_required_and_bonus_intents():
    primary = "Announced a strategic partnership in the last 12 months."
    bonus = "Announced a leadership change in the last 12 months."
    original = {
        "prompt": "Australian payment companies with partnerships or leadership changes.",
        "industry": "Financial Services", "sub_industry": "Payments",
        "country": "Australia", "company_stage": "Public",
        "employee_count": ["51-200", "201-500"],
        "product_service": "Payments", "required_attribute": "Provides payments",
        "intent_signal": primary, "intent_signals": [primary, bonus],
        "intent_category": "PARTNERSHIP", "intent_max_age_days": 180,
    }
    out = canonicalize_generated_icp(original, industry="Financial Services", sub_industry="Payments")
    prose = out["prompt"].split("\n\nIntent qualification:\n", 1)[0]
    assert prose.startswith("Find companies meeting every company requirement.")
    assert "partnerships or leadership changes" not in prose
    assert "Country: Australia" in prose
    assert "Company stage: Public" in prose
    assert "Allowed employee size: 51-200, 201-500" in prose
    assert "Product/service requirement: Payments" in prose
    assert "Required attribute: Provides payments" in prose
    assert f"Required primary intent (180 days): {primary}" in out["prompt"]
    assert f"Optional bonus intent (365 days): {bonus}" in out["prompt"]
    assert "These intent requirements govern qualification." in out["prompt"]
    assert "an optional bonus cannot replace it" in out["prompt"]
    for key in ("country", "company_stage", "employee_count", "product_service", "required_attribute",
                "intent_signal", "intent_signals", "intent_category", "intent_max_age_days"):
        assert out[key] == original[key]
    repeated = canonicalize_generated_icp(out, industry="Financial Services", sub_industry="Payments")
    assert repeated == out


def test_generated_brief_single_intent_preserves_alternatives_within_the_signal():
    primary = "Launched a new product or major platform capability."
    out = canonicalize_generated_icp({
        "prompt": "Software companies in Canada.", "employee_count": ["51-200"],
        "intent_signal": primary, "intent_signals": [primary],
        "intent_category": "PRODUCT_LAUNCH", "intent_max_age_days": 365,
    }, industry="Software", sub_industry="B2B SaaS")
    assert f"Required primary intent (365 days): {primary}" in out["prompt"]
    assert "Optional bonus intent (" not in out["prompt"]
    assert out["bonus_intents"] == []


def test_generated_brief_drops_conflicting_model_prose_but_keeps_internal_or():
    primary = "Announced a strategic partnership or acquired a company in the last 12 months."
    out = canonicalize_generated_icp({
        "prompt": "US transport firms buying, partnering, or opening new lanes.",
        "industry": "Transportation", "sub_industry": "Logistics",
        "country": "United States", "geography": "United States, Northeast",
        "company_stage": "Private", "employee_count": "201-500",
        "product_service": "Freight or parcel delivery across multiple states",
        "required_attribute": "Operates warehouses or cross-dock hubs",
        "intent_signal": primary, "intent_signals": [primary],
        "intent_category": "PARTNERSHIP", "intent_max_age_days": 365,
    }, industry="Transportation", sub_industry="Logistics", employee_bucket_radius=0)
    prose, appendix = out["prompt"].split("\n\nIntent qualification:\n")
    assert "buying, partnering, or opening new lanes" not in prose
    assert "Freight or parcel delivery across multiple states" in prose
    assert "Industry: Transportation" in prose
    assert "Sub-industry: Logistics" in prose
    assert "Headquarters geography: United States, Northeast" in prose
    assert "Allowed employee size: 201-500" in prose
    assert f"Required primary intent (365 days): {primary}" in appendix
    assert canonicalize_generated_icp(out, industry="Transportation", sub_industry="Logistics", employee_bucket_radius=0) == out


@pytest.mark.parametrize("freeform_prompt", [
    "Looking for professional services firms that are expanding, hiring, or making leadership moves — no fluff, just companies with real proof points.",
    "what IT companies are scaling fast in Singapore — partnerships, funding, or product launches all work if the proof is public.",
    "Descriptive brief: energy operators in the US with a new facility, acquisition, or major partnership in the last year, and enough scale that they actually show up in the news.",
    "Transport firms buying, partnering, or opening new lanes.",
    "Biotech firms with funding, new programs, or regulatory approvals.",
])
def test_actual_conflicting_brief_shapes_cannot_override_primary(freeform_prompt):
    primary = "Opened a new project site or office in the last 12 months."
    out = canonicalize_generated_icp({
        "prompt": freeform_prompt, "industry": "Energy",
        "sub_industry": "Renewable power developer", "country": "United States",
        "geography": "United States, South", "company_stage": "Public",
        "employee_count": ["1,001-5,000"],
        "product_service": "Builds or owns renewable generation assets",
        "required_attribute": "Operates energy assets with public project growth",
        "intent_signal": primary,
        "intent_signals": [primary, "Announced an acquisition in the last 12 months."],
        "intent_category": "EXPANSION", "intent_max_age_days": 365,
    }, industry="Energy", sub_industry="Renewable power developer", employee_bucket_radius=0)
    prose, appendix = out["prompt"].split("\n\nIntent qualification:\n", 1)
    assert freeform_prompt not in out["prompt"]
    assert "Industry: Energy" in prose
    assert "Required primary intent (365 days): " + primary in appendix
    assert "Optional bonus intent (365 days): Announced an acquisition" in appendix


def test_stage_widening_refreshes_brief_size_without_changing_intent():
    out = canonicalize_generated_icp({
        "prompt": "Biotech firms after funding, programs, or regulatory approvals.",
        "industry": "Biotechnology", "sub_industry": "Therapeutics",
        "country": "United States", "geography": "United States",
        "company_stage": "Series A", "employee_count": "11-50",
        "product_service": "Develops therapeutics",
        "required_attribute": "Runs clinical programs",
        "intent_signal": "Raised a Series A round in the last 12 months",
        "intent_signals": ["Raised a Series A round in the last 12 months"],
        "intent_category": "FUNDING", "intent_max_age_days": 365,
    }, industry="Biotechnology", sub_industry="Therapeutics", employee_bucket_radius=0)
    from gateway.tasks.icp_generator import apply_generated_icp_contract
    before = out["prompt"].split("\n\nIntent qualification:\n", 1)[1]
    apply_generated_icp_contract([out])
    prose, appendix = out["prompt"].split("\n\nIntent qualification:\n", 1)
    assert "funding, programs, or regulatory approvals" not in prose
    assert f"Allowed employee size: {', '.join(out['employee_count'])}" in prose
    assert appendix == before
    assert "Required primary intent (365 days): Raised a Series A round" in appendix
    prompt = out["prompt"]
    apply_generated_icp_contract([out])
    assert out["prompt"] == prompt


def test_fallback_briefs_use_the_same_required_bonus_contract(monkeypatch):
    primary = "Announced a strategic partnership"
    bonus = "Announced a leadership change"
    monkeypatch.setattr(ig.random, "sample", lambda population, count: [primary, bonus])
    # Exercise each real fallback company-prose template without changing its
    # filters or treating 'or' inside a single event as a separator.
    real_choice = ig.random.choice
    for index in range(5):
        monkeypatch.setattr(ig.random, "choice", lambda values, index=index:
            values[index] if len(values) == 5 and all(isinstance(v, str) for v in values)
            and any("We're selling" in v for v in values) else real_choice(values))
        out = ig.generate_single_icp(f"brief-{index}", industry="Software")
        prose, requirements = out["prompt"].split("\n\nIntent qualification:\n")
        assert primary.lower() not in prose.lower()
        assert bonus.lower() not in prose.lower()
        assert f"Required primary intent (365 days): {primary}" in requirements
        assert f"Optional bonus intent (365 days): {bonus}" in requirements


def test_intent_section_survives_contact_projection_and_recanonicalization():
    from lab_arena.integrity import company_only_icp

    out = canonicalize_generated_icp({
        "prompt": "Software companies in Canada.", "employee_count": ["51-200"],
        "intent_signal": SPECIFIC_INTENT, "intent_signals": [SPECIFIC_INTENT],
        "intent_category": "PRODUCT_LAUNCH", "intent_max_age_days": 365,
        "product_service": "A software platform", "target_roles": ["CEO"],
    }, industry="Software", sub_industry="B2B SaaS", contacts_required=True)
    repeated = canonicalize_generated_icp(out, industry="Software", sub_industry="B2B SaaS", contacts_required=True)
    assert repeated == out
    assert out["prompt"].count("Target contacts:") == 1
    projected = company_only_icp(out)
    assert "Target contacts:" not in projected["prompt"]
    assert f"Required primary intent (365 days): {SPECIFIC_INTENT}" in projected["prompt"]
