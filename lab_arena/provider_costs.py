"""Pure provider response accounting for Lab Arena.

Authorization remains in :mod:`lab_arena.operations`.  These helpers only
price an operation or a trusted provider response after that authorization.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_CEILING
from typing import Any, Mapping, Optional


MICROUSD_PER_USD = Decimal("1000000")
SCRAPINGDOG_USD_PER_CREDIT = Decimal("0.00005")
DEEPLINE_USD_PER_CREDIT = Decimal("0.10")

_SCRAPINGDOG_CREDITS_BY_OPERATION = {
    "scrapingdog.scrape": Decimal("5"),
    "scrapingdog.profile": Decimal("100"),
    "scrapingdog.profile_post": Decimal("5"),
    "scrapingdog.linkedinjobs": Decimal("5"),
    "scrapingdog.instagram_profile": Decimal("15"),
    "scrapingdog.tiktok_profile": Decimal("5"),
    "scrapingdog.youtube_transcripts": Decimal("5"),
}
_SCRAPINGDOG_DEFAULT_CREDITS = Decimal("5")
_DEEPLINE_CREDIT_FIELDS = (
    "credits_charged",
    "totalCredits",
    "total_credits",
    "credits_used",
    "credits",
)
# Deepline's tool descriptions, checked 2026-09-10. These are reservations,
# never final charges. Tools with dynamic prices reserve the remaining budget
# in the database instead of treating an unknown price as zero.
_DEEPLINE_FIXED_CREDITS = {
    "exa_answer": Decimal("0.07"),
    "free_simple_company_search": Decimal("0"),
    "generic_http_request": Decimal("0"),
    "harvestapi_get_job": Decimal("0.01"),
    "harvestapi_get_post": Decimal("0.03"),
    "hunter_discover": Decimal("0"),
    "predictleads_company_financing_events": Decimal("0.56"),
    "predictleads_company_job_openings": Decimal("0.56"),
    "predictleads_company_news_events": Decimal("0.56"),
}


@dataclass(frozen=True)
class ProviderCost:
    """One exact provider charge expressed in integer micro-USD."""

    microusd: int
    units: Decimal
    unit_name: str
    price_basis: str


def _decimal(value: Any) -> Optional[Decimal]:
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        return None
    try:
        amount = Decimal(str(value).strip())
    except (InvalidOperation, ValueError):
        return None
    if not amount.is_finite() or amount < 0:
        return None
    return amount


def _microusd_ceiling(usd: Decimal) -> int:
    return int((usd * MICROUSD_PER_USD).to_integral_value(rounding=ROUND_CEILING))


def scrapingdog_cost(operation_id: str, parameters: Mapping[str, Any]) -> ProviderCost:
    """Price one already-approved Scrapingdog operation.

    An approved operation absent from the legacy price map uses the documented
    five-credit fallback.  This function does not authorize operation ids.
    """

    credits = _SCRAPINGDOG_CREDITS_BY_OPERATION.get(str(operation_id), _SCRAPINGDOG_DEFAULT_CREDITS)
    basis = "scrapingdog_legacy_endpoint_map"
    if operation_id == "scrapingdog.profile" and str(parameters.get("type") or "").lower() == "company":
        credits = Decimal("10")
        basis = "scrapingdog_company_profile"
    elif operation_id not in _SCRAPINGDOG_CREDITS_BY_OPERATION:
        basis = "scrapingdog_approved_operation_fallback"
    return ProviderCost(
        microusd=_microusd_ceiling(credits * SCRAPINGDOG_USD_PER_CREDIT),
        units=credits,
        unit_name="credits",
        price_basis=basis,
    )


def deepline_cost(response_json: Any) -> Optional[ProviderCost]:
    """Read credits only from a trusted top-level Deepline billing object."""

    if not isinstance(response_json, Mapping):
        return None
    billing = response_json.get("billing")
    if not isinstance(billing, Mapping):
        return None
    for field in _DEEPLINE_CREDIT_FIELDS:
        if field not in billing:
            continue
        credits = _decimal(billing[field])
        if credits is None:
            return None
        return ProviderCost(
            microusd=_microusd_ceiling(credits * DEEPLINE_USD_PER_CREDIT),
            units=credits,
            unit_name="credits",
            price_basis="deepline_billing_%s_x_0.10_usd" % field,
        )
    return None


def deepline_reservation_cost(parameters: Mapping[str, Any]) -> Optional[ProviderCost]:
    """Return a published fixed-call estimate; None means dynamically priced."""

    credits = _DEEPLINE_FIXED_CREDITS.get(str(parameters.get("tool") or ""))
    if credits is None:
        return None
    return ProviderCost(
        microusd=_microusd_ceiling(credits * DEEPLINE_USD_PER_CREDIT),
        units=credits,
        unit_name="credits",
        price_basis="deepline_fixed_call_reservation_20260910",
    )


def openrouter_cost(response_json: Any) -> Optional[ProviderCost]:
    """Read OpenRouter's exact charge from the trusted top-level usage object."""

    if not isinstance(response_json, Mapping):
        return None
    usage = response_json.get("usage")
    if not isinstance(usage, Mapping) or "cost" not in usage:
        return None
    usd = _decimal(usage["cost"])
    if usd is None:
        return None
    return ProviderCost(
        microusd=_microusd_ceiling(usd),
        units=usd,
        unit_name="usd",
        price_basis="openrouter_usage_cost",
    )


__all__ = [
    "DEEPLINE_USD_PER_CREDIT",
    "ProviderCost",
    "SCRAPINGDOG_USD_PER_CREDIT",
    "deepline_cost",
    "deepline_reservation_cost",
    "openrouter_cost",
    "scrapingdog_cost",
]
