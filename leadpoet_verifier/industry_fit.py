"""Deterministic ICP industry-fit matching (ported from the site verifier).

Faithful extraction of the pure matchers from the leadpoet-site sourcing
verifier's ``adapter.py`` (commits 69b01ae9/c95d5a89 B2B-SaaS evidence,
ac2538a6 structured industry criteria, a680e218/b7fe23d3 canonical taxonomy):

- ``industry_fit`` — conservative structured-industry match: authoritative
  Leadpoet taxonomy first, bounded canonical concepts second, and an exact
  non-generic token fallback only when the taxonomy is silent and the request
  carries no specific concept beyond the contextual broad ones.
- ``b2b_saas_evidence`` — B2B SaaS requires BOTH a buyer-audience signal and a
  software-product signal, grounded in provider labels / description / frozen
  evidence quote (never the buyer's own request text), and not service-only
  unless a strong owned-software signal corroborates.

Pure stdlib + the ported ``industry_taxonomy`` module: no network, no LLM, no
providers — safe for shadow evaluation anywhere in the scoring path.
"""

from __future__ import annotations

import re
from typing import Any

from leadpoet_verifier.industry_taxonomy import (
    CONTEXTUAL_BROAD_CONCEPTS as _CONTEXTUAL_BROAD_CONCEPTS,
    industry_concepts as _industry_concepts,
    industry_tokens as _industry_tokens,
    leadpoet_taxonomy_match as _leadpoet_taxonomy_match,
    normalized_industry_text as _normalized_text,
    requested_industry_concepts as _requested_industry_concepts,
)


_B2B_CUSTOMER_PATTERNS: tuple[tuple[str, str], ...] = (
    ("b2b", r"\b(?:b2b|business to business)\b"),
    ("enterprise", r"\benterprises?\b"),
    ("business_customer", r"\b(?:business|commercial|corporate) customers?\b"),
    (
        "business_audience",
        r"\b(?:for|serv(?:e|es|ing)|helps?|supports?|used by|built for|designed for|sold to|enables?) "
        r"(?:[a-z0-9]+ ){0,3}(?:businesses|companies|enterprises|organizations|employers|teams|"
        r"smbs|smes|clinics|practices|providers|brands|retailers|merchants)\b",
    ),
    (
        "business_workflow",
        r"\b(?:business application|sales enablement|revenue intelligence|revenue operations|revops|"
        r"go to market|gtm|"
        r"customer support|customer service|human resources|hr management|payroll|applicant tracking|"
        r"ats|recruiting|recruitment|talent acquisition|workforce management|procurement|compliance|"
        r"practice management|merchant operations|retail operations|post purchase|business operations)\b",
    ),
)

_SOFTWARE_PRODUCT_PATTERNS: tuple[tuple[str, str], ...] = (
    ("saas", r"\b(?:saas|software as a service)\b"),
    ("software", r"\bsoftware\b"),
    ("application", r"\b(?:cloud based )?applications?\b"),
    ("api", r"\bapis?\b"),
    ("operating_system", r"\boperating systems?\b"),
    ("platform", r"\bplatforms?\b"),
)

_SERVICE_ONLY_PATTERNS: tuple[tuple[str, str], ...] = (
    ("agency", r"\bagenc(?:y|ies)\b"),
    ("consulting", r"\bconsulting\b"),
    ("managed_services", r"\bmanaged services\b"),
    ("outsourcing", r"\boutsourc(?:e|ed|es|ing)\b"),
    ("professional_services", r"\bprofessional services\b"),
)

_STRONG_SOFTWARE_SIGNALS = {"api", "application", "operating_system", "saas", "software"}


def _matched_pattern_names(
    text: str, patterns: tuple[tuple[str, str], ...]
) -> set[str]:
    return {name for name, pattern in patterns if re.search(pattern, text)}


def _collect_source_signals(
    signals_by_source: dict[str, dict[str, set[str]]],
    signal_kind: str,
    *,
    include: set[str] | None = None,
    exclude: set[str] | None = None,
) -> set[str]:
    result: set[str] = set()
    for source_name, signals in signals_by_source.items():
        if include is not None and source_name not in include:
            continue
        if exclude is not None and source_name in exclude:
            continue
        result.update(signals[signal_kind])
    return result


def b2b_saas_evidence(
    candidate_industry: Any,
    candidate_subindustry: Any,
    candidate_description: Any,
    candidate_evidence_quote: Any,
) -> dict[str, Any]:
    """Require both buyer audience and software product evidence for B2B SaaS.

    A broad provider label such as ``Software Development`` is useful product
    evidence but is not proof that the company sells to businesses. Conversely,
    an industry such as ``Financial Services`` should not hide explicit evidence
    that the company sells a software product to businesses. Only the provider
    labels, company description, and source quote are considered; the requested
    attribute text and model explanation are deliberately excluded because they
    can merely repeat the buyer's request.
    """

    sources = {
        "industry": _normalized_text(candidate_industry),
        "subindustry": _normalized_text(candidate_subindustry),
        "description": _normalized_text(candidate_description),
        "required_attribute_quote": _normalized_text(candidate_evidence_quote),
    }
    signals_by_source = {
        name: {
            "customer": _matched_pattern_names(value, _B2B_CUSTOMER_PATTERNS),
            "product": _matched_pattern_names(value, _SOFTWARE_PRODUCT_PATTERNS),
            "service": _matched_pattern_names(value, _SERVICE_ONLY_PATTERNS),
        }
        for name, value in sources.items()
        if value
    }
    customer_signals = _collect_source_signals(signals_by_source, "customer")
    product_signals = _collect_source_signals(signals_by_source, "product")
    service_only_signals = _collect_source_signals(signals_by_source, "service")
    corroborating_product_signals = _collect_source_signals(
        signals_by_source, "product", exclude={"industry"}
    )
    product_owned_despite_services = bool(
        corroborating_product_signals & (_STRONG_SOFTWARE_SIGNALS - {"software"})
        or {"software", "platform"} <= corroborating_product_signals
    )
    service_only = bool(service_only_signals and not product_owned_despite_services)
    label_sources = {"industry", "subindustry"}
    label_customer_signals = _collect_source_signals(
        signals_by_source, "customer", include=label_sources
    )
    label_product_signals = _collect_source_signals(
        signals_by_source, "product", include=label_sources
    )
    quote_signals = signals_by_source.get(
        "required_attribute_quote", {"customer": set(), "product": set()}
    )
    source_grounded = bool(
        (label_customer_signals and label_product_signals)
        or quote_signals["customer"]
        or quote_signals["product"]
    )
    passed = bool(
        customer_signals and product_signals and source_grounded and not service_only
    )

    matched_sources = sorted(
        name
        for name, signals in signals_by_source.items()
        if signals["customer"] or signals["product"]
    )
    return {
        "passed": passed,
        "customer_signals": sorted(customer_signals),
        "product_signals": sorted(product_signals),
        "corroborating_product_signals": sorted(corroborating_product_signals),
        "service_only_signals": sorted(service_only_signals),
        "source_grounded": source_grounded,
        "matched_sources": matched_sources,
    }


def industry_fit(
    requested: Any,
    candidate_industry: Any,
    candidate_subindustry: Any,
    *,
    candidate_description: Any = None,
    candidate_evidence_quote: Any = None,
) -> tuple[bool, dict[str, Any]]:
    """Match provider labels to a structured buyer industry conservatively.

    Exact Leadpoet parent/subindustry labels use the repository's authoritative
    taxonomy. Provider-specific labels fall back to bounded semantic aliases
    and, only for unknown or broad-only requests, exact non-generic tokens. The
    downstream source-grounded intent verifier and required-attribute evidence
    remain mandatory publication gates.
    """

    requested_text = _normalized_text(requested)
    candidate_values = (candidate_industry, candidate_subindustry)
    candidate_labels = sorted(
        {_normalized_text(value) for value in candidate_values if _normalized_text(value)}
    )
    taxonomy_decision, taxonomy_detail = _leadpoet_taxonomy_match(
        requested,
        candidate_industry,
        candidate_subindustry,
    )
    requested_concepts, suppressed_requested_concepts = _requested_industry_concepts(requested)
    candidate_concepts = _industry_concepts(*candidate_values)
    b2b_saas: dict[str, Any] | None = None
    if "b2b_saas" in requested_concepts:
        b2b_saas = b2b_saas_evidence(
            candidate_industry,
            candidate_subindustry,
            candidate_description,
            candidate_evidence_quote,
        )
        if b2b_saas["passed"]:
            candidate_concepts.add("b2b_saas")
        else:
            candidate_concepts.discard("b2b_saas")
    semantic_matched_concepts = requested_concepts & candidate_concepts
    requested_tokens = _industry_tokens(requested)
    candidate_tokens = _industry_tokens(*candidate_values)
    token_fallback_allowed = (
        taxonomy_decision is None
        and not (requested_concepts - _CONTEXTUAL_BROAD_CONCEPTS)
    )
    semantic_matched_tokens = (
        requested_tokens & candidate_tokens if token_fallback_allowed else set()
    )
    if taxonomy_decision is None:
        matched_concepts = semantic_matched_concepts
        matched_tokens = semantic_matched_tokens
        passed = bool(
            requested_text
            and candidate_labels
            and (matched_concepts or matched_tokens)
        )
        match_strategy = (
            "canonical_concept"
            if matched_concepts
            else "exact_token"
            if matched_tokens
            else None
        )
    else:
        matched_concepts = semantic_matched_concepts if taxonomy_decision else set()
        matched_tokens = set()
        passed = bool(requested_text and candidate_labels and taxonomy_decision)
        match_strategy = "leadpoet_taxonomy" if passed else None
    detail = {
        "candidate": candidate_labels,
        "requested": requested_text or None,
        "requested_concepts": sorted(requested_concepts),
        "suppressed_requested_concepts": sorted(suppressed_requested_concepts),
        "candidate_concepts": sorted(candidate_concepts),
        "matched_concepts": sorted(matched_concepts),
        "matched_tokens": sorted(matched_tokens),
        "token_fallback_allowed": token_fallback_allowed,
        "match_strategy": match_strategy,
    }
    if taxonomy_decision is not None:
        detail["leadpoet_taxonomy"] = {
            **taxonomy_detail,
            "decision": "accepted" if taxonomy_decision else "rejected",
        }
    if b2b_saas is not None:
        detail["b2b_saas_evidence"] = b2b_saas
    return passed, detail
