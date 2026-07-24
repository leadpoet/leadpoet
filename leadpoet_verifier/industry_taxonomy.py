from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional


def normalized_industry_text(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).strip()


def _load_leadpoet_taxonomy() -> tuple[
    dict[str, str],
    dict[str, tuple[str, frozenset[str]]],
]:
    path = Path(__file__).with_name("leadpoet_industry_taxonomy.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != 1:
        raise RuntimeError("Unsupported Leadpoet industry taxonomy version")

    parents = {
        normalized_industry_text(parent): parent
        for parent in payload["parent_industries"]
    }
    subindustries = {
        normalized_industry_text(subindustry): (
            subindustry,
            frozenset(normalized_industry_text(parent) for parent in parent_industries),
        )
        for subindustry, parent_industries in payload["subindustry_parents"].items()
    }
    return parents, subindustries


LEADPOET_PARENT_INDUSTRIES, LEADPOET_SUBINDUSTRY_PARENTS = (
    _load_leadpoet_taxonomy()
)


def leadpoet_taxonomy_match(
    requested: Any,
    candidate_industry: Any,
    candidate_subindustry: Any,
) -> tuple[Optional[bool], dict[str, Any]]:
    """Return an authoritative decision when exact Leadpoet labels are present.

    The repository taxonomy is stricter than semantic aliases: adjacent parent
    categories such as Payments and Financial Services remain distinct unless
    the exact subindustry is explicitly assigned to both. ``None`` delegates
    provider-specific labels to the bounded semantic matcher below.
    """

    requested_key = normalized_industry_text(requested)
    requested_parent = LEADPOET_PARENT_INDUSTRIES.get(requested_key)
    if requested_parent is None:
        return None, {}

    candidate_parent_key = normalized_industry_text(candidate_industry)
    candidate_parent = LEADPOET_PARENT_INDUSTRIES.get(candidate_parent_key)
    candidate_subindustry_key = normalized_industry_text(candidate_subindustry)
    subindustry_entry = LEADPOET_SUBINDUSTRY_PARENTS.get(candidate_subindustry_key)
    if subindustry_entry is None and candidate_parent is None:
        # Some providers emit a Leadpoet subindustry label in their broader
        # industry field. Treat that exact label as taxonomy evidence too.
        subindustry_entry = LEADPOET_SUBINDUSTRY_PARENTS.get(candidate_parent_key)

    detail: dict[str, Any] = {
        "requested_parent": requested_parent,
        "candidate_parent": candidate_parent,
        "candidate_subindustry": subindustry_entry[0] if subindustry_entry else None,
        "candidate_subindustry_parents": (
            sorted(
                LEADPOET_PARENT_INDUSTRIES[parent]
                for parent in subindustry_entry[1]
            )
            if subindustry_entry
            else []
        ),
    }

    if subindustry_entry is not None:
        _subindustry, allowed_parents = subindustry_entry
        candidate_parent_consistent = (
            candidate_parent is None or candidate_parent_key in allowed_parents
        )
        detail["candidate_parent_consistent"] = candidate_parent_consistent
        return (
            requested_key in allowed_parents and candidate_parent_consistent,
            detail,
        )

    if candidate_parent is not None:
        detail["candidate_parent_consistent"] = True
        return requested_key == candidate_parent_key, detail

    return None, detail


# Canonical concepts deliberately map buyer language and common provider labels
# into the same bounded vocabulary. Patterns stay specific enough that a broad
# word such as "security" or "automation" cannot silently cross industries.
INDUSTRY_CONCEPT_PATTERNS: dict[str, tuple[str, ...]] = {
    "aerospace": (
        r"\baerospace\b", r"\baviation\b", r"\baircraft\b", r"\bspace systems?\b",
    ),
    "agriculture": (
        r"\bagricultur(?:e|al)\b", r"\bagritech\b", r"\bagtech\b", r"\bfarming\b",
        r"\bfarm (?:tech|technology)\b",
        r"\bcrop(?:s| production)?\b", r"\blivestock\b", r"\bhorticulture\b",
        r"\baquaculture\b",
    ),
    "artificial_intelligence": (
        r"\bai\b", r"\bartificial intelligence\b", r"\bgenerative ai\b",
        r"\bmachine learning\b", r"\bcomputer vision\b", r"\blarge language models?\b",
    ),
    "automotive_mobility": (
        r"\bautomotive\b", r"\bautomobiles?\b", r"\bmotor vehicles?\b",
        r"\belectric vehicles?\b", r"\bvehicle technology\b", r"\bmobility technology\b",
    ),
    "b2b_saas": (
        r"\bsaas\b",
        r"\bsoftware as a service\b",
        r"\bsoftware development\b",
        r"\benterprise software\b",
        r"\bbusiness (?:application|software|platform)s?\b",
        r"\bcloud(?: based)? (?:application|software|platform)s?\b",
    ),
    "biotechnology_pharmaceuticals": (
        r"\bbiotech(?:nology)?\b", r"\blife sciences?\b", r"\bpharma(?:ceutical)?s?\b",
        r"\bdrug discovery\b", r"\btherapeutics?\b", r"\bbiopharma\b",
        r"\bgenomics?\b",
    ),
    "blockchain_crypto": (
        r"\bblockchain\b", r"\bcrypto(?:currency)?\b", r"\bweb3\b",
        r"\bdigital assets?\b", r"\bdecentralized finance\b", r"\bdefi\b",
    ),
    "construction": (
        r"\bconstruction\b", r"\bcivil engineering\b", r"\bbuilding materials?\b",
        r"\bconstruction technology\b", r"\bcontech\b",
    ),
    "consumer_goods": (
        r"\bconsumer goods?\b", r"\bcpg\b", r"\bconsumer packaged goods?\b",
        r"\bhousehold products?\b", r"\bpersonal care products?\b", r"\bcosmetics\b",
        r"\bcosmetic (?:brands?|companies|manufactur(?:er|ing)|products?)\b",
    ),
    "cybersecurity": (
        r"\bcyber\s*security\b",
        r"\bcomputer and network security\b",
        r"\b(?:application|cloud|data|developer|endpoint|identity|network|saas|software supply chain) security\b",
        r"\bsecurity posture management\b",
        r"\bthreat (?:detection|intelligence|management|prevention|response)\b",
        r"\bransomware\b", r"\bvulnerability management\b", r"\bzero trust\b",
    ),
    "data_analytics": (
        r"\bdata and analytics\b", r"\bdata analytics\b", r"\banalytics\b",
        r"\bbusiness intelligence\b", r"\bdata infrastructure\b", r"\bdata platforms?\b",
        r"\bdatabase(?:s| technology)?\b",
    ),
    "education": (
        r"\beducation\b", r"\bedtech\b", r"\be learning\b", r"\bonline learning\b",
        r"\blearning management systems?\b", r"\bhigher education\b",
        r"\bprimary and secondary education\b",
    ),
    "energy_infrastructure": (
        r"\b(?:battery|batteries)(?: energy)? "
        r"(?:manufactur(?:er|ing)|materials?|recycling|storage|systems?|technology)\b",
        r"\bcharging infrastructure\b", r"\bev charging\b",
        r"\bcharging stations?\b", r"\bclimate(?: tech)?\b",
        r"\belectric power\b", r"\belectrification\b", r"\benergy\b(?! trading)", r"\bgrid\b",
        r"\bnuclear\b", r"\belectroly[sz]er(?:s)?\b", r"\bfuel cell(?:s)?\b",
        r"\bhydrogen\b", r"\bmicrogrid(?:s)?\b", r"\bpower generation\b",
        r"\brenewable\b", r"\bsolar\b", r"\butilit(?:y|ies)\b",
    ),
    "banking": (
        r"\bbank(?:ing)?\b", r"\btreasury\b",
    ),
    "defense_military": (
        r"\bdefen[cs]e\b", r"\bmilitary\b", r"\bnational security\b",
    ),
    "financial_infrastructure": (
        r"\bfintech\b", r"\bfinancial infrastructure\b", r"\bfinancial services\b",
    ),
    "lending_investments": (
        r"\bcredit\b(?! (?:course|hours?|transfer|units?))",
        r"\blending\b", r"\binvestment technology\b",
    ),
    "food_beverage": (
        r"\bfood and beverage\b", r"\bfood production\b", r"\bbeverage manufacturing\b",
        r"\bbeverage(?:s| brands?| companies| producers?)?\b",
        r"\bbrewer(?:y|ies)\b", r"\bwiner(?:y|ies)\b",
    ),
    "government_public_sector": (
        r"\bgovernment\b", r"\bgovtech\b", r"\bpublic sector\b",
        r"\bpublic administration\b", r"\bgovernment administration\b",
    ),
    "hardware_electronics": (
        r"\bhardware\b(?! stores?\b)", r"\bcomputer hardware\b",
        r"\belectronics manufacturing\b",
        r"\belectronic components?\b", r"\bconsumer electronics\b",
    ),
    "healthcare": (
        r"\bclinical\b", r"\bhealth(?:care)?\b(?! insurance)", r"\bhospital(?:s)?\b",
        r"\bmedical\b", r"\bpatient(?:s)?\b", r"\bclinics?\b", r"\bdental\b",
        r"\bcare delivery\b", r"\bdigital health\b", r"\bhealthtech\b",
    ),
    "hospitality_travel": (
        r"\bhospitality\b", r"\bhotels?\b", r"\bresorts?\b",
        r"\btravel (?:agenc(?:y|ies)|companies|services)\b",
        r"\btourism\b", r"\baccommodations?\b", r"\bairlines?\b",
    ),
    "human_resources": (
        r"\bhuman resources\b", r"\bhr\b", r"\bhr tech\b", r"\bhr software\b",
        r"\bhr management\b",
        r"\bpayroll\b", r"\bworkforce management\b", r"\btalent management\b",
        r"\bapplicant tracking\b", r"\brecruiting technology\b",
    ),
    "industrial_automation": (
        r"\bfactory automation\b", r"\bindustrial automation\b", r"\bautomation machinery\b",
        r"\bcobot(?:s)?\b", r"\bindustrial control(?:s)?\b", r"\bmachine vision\b",
        r"\brobot(?:ic|ics|s)?\b",
    ),
    "industrial_manufacturing": (
        r"\badvanced manufacturing\b", r"\bindustrial manufacturing\b",
        r"\bindustrial equipment\b", r"\bindustrial machinery\b",
        r"\bmanufacturing equipment\b",
    ),
    "information_technology": (
        r"\binformation technology\b", r"\bit services?\b", r"^it$",
        r"\btechnology information and (?:internet|media)\b",
        r"\binternet services?\b",
    ),
    "insurance": (
        r"\binsurance\b", r"\binsurtech\b", r"\bunderwriting\b", r"\breinsurance\b",
    ),
    "legal_services": (
        r"\blegal\b", r"\blegal services?\b", r"\blaw practice\b", r"\blaw firms?\b",
        r"\blegaltech\b", r"\blegal (?:tech|technology)\b", r"\blawtech\b",
    ),
    "logistics": (
        r"\bfleet\b", r"\bfreight\b", r"\blogistics\b",
        r"\bsupply chain\b(?! security)", r"\btransportation\b", r"\bwarehousing\b",
        r"\blast mile delivery\b",
    ),
    "manufacturing": (r"\bmanufacturing\b",),
    "marketing_advertising": (
        r"\bmarketing\b", r"\badvertising\b", r"\bmartech\b", r"\badtech\b",
        r"\bpublic relations\b", r"\bsearch engine marketing\b", r"\bseo\b",
    ),
    "content_publishing": (
        r"\bcontent and publishing\b", r"\bdigital publishing\b", r"\bpublishing\b",
    ),
    "gaming": (
        r"\bcomputer games\b", r"\bgaming\b", r"\bvideo games\b", r"\besports\b",
    ),
    "media_broadcast": (
        r"\bmedia and entertainment\b", r"\bbroadcast media\b", r"\bbroadcasting\b",
        r"\btelevision production\b",
    ),
    "music_audio": (
        r"\bmusic and audio\b", r"\bmusic industry\b", r"\brecord labels?\b",
        r"\baudio production\b",
    ),
    "nonprofit_social_impact": (
        r"\bnonprofits?\b", r"\bnon profit\b", r"\bsocial impact\b",
        r"\bphilanthrop(?:y|ic)\b", r"\bcharit(?:y|ies|able)\b",
        r"\bcivic and social organizations?\b",
    ),
    "professional_services": (
        r"\bprofessional services\b", r"\bbusiness consulting and services\b",
        r"\bmanagement consulting\b", r"\bconsulting\b", r"\badvisory services?\b",
        r"\baccounting\b", r"\boutsourcing\b",
    ),
    "regulatory_compliance": (
        r"\bcompliance\b", r"\bregtech\b", r"\bregulatory technology\b",
        r"\bgovernance risk and compliance\b", r"\bgrc\b",
    ),
    "real_estate": (
        r"\breal estate\b", r"\bproptech\b", r"\bproperty technology\b",
        r"\bproperty management\b", r"\bcommercial property\b",
    ),
    "restaurants_food_service": (
        r"\brestaurants?\b", r"\bfood services?\b", r"\bcatering\b", r"\bdining\b",
    ),
    "retail_ecommerce": (
        r"\bretail\b", r"\be commerce\b", r"\becommerce\b", r"\bretail technology\b",
        r"\bonline marketplace\b", r"\binternet marketplace\b", r"\bcommerce and shopping\b",
    ),
    "sales_revenue": (
        r"\bsales technology\b", r"\bsales tech\b", r"\bsales enablement\b",
        r"\brevenue intelligence\b", r"\brevenue operations\b", r"\brevops\b",
        r"\bcustomer relationship management\b", r"\bcrm\b",
    ),
    "semiconductor_equipment": (
        r"\b(?:computer|semiconductor) chips?\b", r"\bmicrochips?\b",
        r"\bchip (?:design|equipment|fabrication)\b",
        r"\bdie bond(?:er|ing)?\b", r"\bmetrology\b",
        r"\bsemiconductor(?:s)?\b", r"\bwafer(?:s)?\b",
    ),
    "software": (
        r"\bsoftware\b", r"\boperating systems?\b",
    ),
    "payments": (
        r"\bpayment(?:s)?\b", r"\bpayment processing\b", r"\bbilling platforms?\b",
    ),
    "telecommunications": (
        r"\btelecommunications?\b", r"\btelecom\b", r"\bwireless services?\b",
        r"\bwired telecommunications\b", r"\bmobile network operators?\b",
        r"\bcommunications infrastructure\b",
    ),
}


# These categories are useful only when the buyer requests the broad category
# itself. In "logistics software" or "pharmaceutical manufacturing", the broad
# word is a modifier and must not let unrelated software/manufacturing through.
CONTEXTUAL_BROAD_CONCEPTS = {
    "hardware_electronics",
    "information_technology",
    "manufacturing",
    "software",
}

_BROAD_OPTION_PATTERNS: dict[str, str] = {
    "hardware_electronics": r"(?:hardware|electronics)",
    "information_technology": r"(?:information technology|technology|tech|it services?)",
    "manufacturing": r"manufacturing",
    "software": r"software",
}

GENERIC_INDUSTRY_TOKENS = {
    "and", "b2b", "business", "businesses", "company", "companies", "development",
    "equipment", "group", "hardware", "industry", "internet", "manufacturing", "marketplace",
    "platform", "platforms", "product", "products", "provider", "providers", "services",
    "software", "solutions", "systems", "tech", "technology",
}


def industry_concepts(*values: Any) -> set[str]:
    text = " ".join(normalized_industry_text(value) for value in values if value)
    return {
        concept
        for concept, patterns in INDUSTRY_CONCEPT_PATTERNS.items()
        if any(re.search(pattern, text) for pattern in patterns)
    }


def _is_standalone_broad_option(value: Any, concept: str) -> bool:
    option = _BROAD_OPTION_PATTERNS[concept]
    raw = str(value or "").casefold()
    qualifier = r"(?:\s+(?:businesses|companies|industry|providers|vendors))?"
    return bool(re.search(
        rf"(?:^|[,;/]|\bor\b)\s*(?:b2b\s+)?{option}{qualifier}\s*(?=$|[,;/]|\bor\b)",
        raw,
    ))


def requested_industry_concepts(value: Any) -> tuple[set[str], set[str]]:
    concepts = industry_concepts(value)
    specific = concepts - CONTEXTUAL_BROAD_CONCEPTS
    if not specific:
        return concepts, set()
    suppressed = {
        concept
        for concept in concepts & CONTEXTUAL_BROAD_CONCEPTS
        if not _is_standalone_broad_option(value, concept)
    }
    return concepts - suppressed, suppressed


def industry_tokens(*values: Any) -> set[str]:
    return {
        token
        for value in values
        for token in normalized_industry_text(value).split()
        if len(token) >= 5 and token not in GENERIC_INDUSTRY_TOKENS
    }
