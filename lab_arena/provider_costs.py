"""Pure provider response accounting for Lab Arena.

Authorization remains in :mod:`lab_arena.operations`.  These helpers only
price an operation or a trusted provider response after that authorization.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_CEILING
from typing import Any, Mapping, Optional, Tuple


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


def deepline_billing_history_cost(
    response_json: Any, *, request_id: str, operation: str, current_offset: int = 0
) -> Tuple[str, Optional[ProviderCost], bool, Optional[int]]:
    """Match one exact Deepline billing-history entry without retaining it.

    ``pending`` means the requested job is absent. ``nonterminal`` means its
    charge is present but not final. ``invalid`` means the history shape or
    exact job match cannot be trusted. Only one ``posted`` or ``free`` entry
    can produce a cost.
    """

    if not isinstance(response_json, Mapping):
        return "invalid", None, False, None
    recent = response_json.get("recent")
    if not isinstance(recent, Mapping):
        return "invalid", None, False, None
    entries = recent.get("entries")
    if not isinstance(entries, list) or len(entries) > 50 or any(
        not isinstance(entry, Mapping) for entry in entries
    ):
        return "invalid", None, False, None
    matches = [entry for entry in entries if entry.get("request_id") == request_id]
    if not matches:
        has_more = recent.get("has_more")
        if not isinstance(has_more, bool):
            return "invalid", None, False, None
        if not has_more:
            return "pending", None, False, None
        next_offset = recent.get("next_offset")
        if (
            isinstance(next_offset, bool)
            or not isinstance(next_offset, int)
            or next_offset <= current_offset
            or next_offset > 1_000_000
        ):
            return "invalid", None, False, None
        return "pending", None, True, next_offset
    if len(matches) != 1:
        return "invalid", None, False, None

    entry = matches[0]
    if entry.get("operation") != operation:
        return "invalid", None, False, None
    provider = entry.get("provider")
    if not isinstance(provider, str) or not provider:
        return "invalid", None, False, None
    charge_state = entry.get("charge_state")
    if not isinstance(charge_state, str):
        return "invalid", None, False, None
    if charge_state not in ("posted", "free"):
        return "nonterminal", None, False, None
    credits_value = entry.get("credits")
    if isinstance(credits_value, bool) or not isinstance(
        credits_value, (int, float, Decimal)
    ):
        return "invalid", None, False, None
    credits = _decimal(credits_value)
    if credits is None or (charge_state == "free" and credits != 0):
        return "invalid", None, False, None
    return (
        "matched",
        ProviderCost(
            microusd=_microusd_ceiling(credits * DEEPLINE_USD_PER_CREDIT),
            units=credits,
            unit_name="credits",
            price_basis="deepline_billing_history_credits_x_0.10_usd",
        ),
        False,
        None,
    )


def deepline_free_completed_cost(
    parameters: Mapping[str, Any], response_status: Any, response_json: Any
) -> Optional[ProviderCost]:
    """Prove the one Deepline operation whose completed call costs zero.

    A present billing object remains authoritative, including when malformed:
    callers must not replace invalid provider accounting with this fixed zero.
    """

    if (
        parameters.get("tool") != "free_simple_company_search"
        or isinstance(response_status, bool)
        or response_status != 200
        or not isinstance(response_json, Mapping)
        or "billing" in response_json
        or response_json.get("status") != "completed"
        or not isinstance(response_json.get("job_id"), str)
        or not response_json["job_id"].strip()
        or not isinstance(response_json.get("result"), (Mapping, list))
    ):
        return None
    return ProviderCost(
        microusd=0,
        units=Decimal("0"),
        unit_name="credits",
        price_basis="deepline_free_simple_company_search_completed_zero",
    )


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
    "deepline_billing_history_cost",
    "deepline_cost",
    "deepline_free_completed_cost",
    "deepline_reservation_cost",
    "openrouter_cost",
    "scrapingdog_cost",
]
