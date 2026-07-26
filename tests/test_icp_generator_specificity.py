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


def test_llm_provided_category_is_authoritative_over_fallback():
    """The generated ICP's intent_category comes from the LLM (the output
    schema requires it) and canonicalize honors it — even when a specific
    intent's wording contains a keyword the fallback would route elsewhere
    (e.g. 'platform' would fall back to TECHSTACK)."""
    icp = {
        "icp_id": "icp_1", "prompt": "test",
        "product_service": "x", "required_attribute": "x-attr",
        "intent_signal": "Launched a new platform module in the last 12 months",
        "intent_signals": ["Launched a new platform module in the last 12 months"],
        "employee_count": ["51-200"],
        "intent_category": "PRODUCT_LAUNCH",   # LLM-provided
    }
    out = canonicalize_generated_icp(icp, industry="Software", sub_industry="B2B SaaS")
    assert out["intent_category"] == "PRODUCT_LAUNCH"   # LLM wins, not TECHSTACK fallback
    # And the fallback alone would indeed differ, which is why the LLM value is authoritative:
    assert intent_category_for_signal(icp["intent_signal"]) == "TECHSTACK"


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
