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
_DEEPLINE_MAX_CHARGE_GROUP_IDS = 128
_DEEPLINE_MAX_REQUEST_ID_LENGTH = 512
# Deepline's live tool descriptions reported billingMode=no_bill on 2026-09-10.
_DEEPLINE_COMPLETED_NO_BILL_BASIS = {
    "free_simple_company_search": "deepline_free_simple_company_search_completed_zero",
    "hunter_discover": "deepline_hunter_discover_completed_zero",
}
# The authenticated live contract for hunter_discover was checked on
# 2026-09-11: billingMode=no_bill, billingSource=free, and a fixed per-call
# price of zero credits/USD.  A structured provider error therefore also has
# an exact zero Deepline charge when no billing object contradicts the
# published contract.
_DEEPLINE_ERROR_NO_BILL_BASIS = {
    "hunter_discover": "deepline_hunter_discover_error_zero",
}
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

_OPENROUTER_ERROR_TOP_LEVEL_FIELDS = frozenset(
    {"error", "openrouter_metadata", "user_id"}
)
_OPENROUTER_AUXILIARY_REQUEST_FIELDS = frozenset(
    {
        "audio",
        "files",
        "image",
        "images",
        "modalities",
        "plugins",
        "prediction",
        "web_search_options",
    }
)
_OPENROUTER_INSURED_PLAIN_MODELS = frozenset(
    {
        # This allowlist limits only the zero-cost insurance fallback.  It
        # does not constrain Arena routing or normal native billing.
        "anthropic/claude-sonnet-4.5",
        "anthropic/claude-sonnet-5",
        "google/gemini-2.5-flash",
        "google/gemini-2.5-flash-lite",
        "openai/gpt-4o-mini",
        "openai/gpt-5.5",
        "openai/gpt-5.6-sol",
    }
)


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
    exact job match cannot be trusted. One ``posted`` or ``free`` entry can
    produce a cost; a strictly zero ``failed`` error entry proves no charge.
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
    matches = []
    for entry in entries:
        charge_group_ids = None
        metadata = entry.get("metadata")
        if isinstance(metadata, Mapping) and "chargeGroupIds" in metadata:
            charge_group_ids = metadata["chargeGroupIds"]
            if (
                not isinstance(charge_group_ids, list)
                or len(charge_group_ids) > _DEEPLINE_MAX_CHARGE_GROUP_IDS
                or any(
                    not isinstance(value, str)
                    or not value.strip()
                    or len(value) > _DEEPLINE_MAX_REQUEST_ID_LENGTH
                    for value in charge_group_ids
                )
                or len(set(charge_group_ids)) != len(charge_group_ids)
            ):
                return "invalid", None, False, None
        direct_match = entry.get("request_id") == request_id
        group_match = charge_group_ids is not None and request_id in charge_group_ids
        if direct_match or group_match:
            # A multi-ID charge group is an aggregate across independent
            # Runtime requests. Its credits cannot be assigned to one call.
            if charge_group_ids is not None and charge_group_ids != [request_id]:
                return "invalid", None, False, None
            entry_request_id = entry.get("request_id")
            if (
                not isinstance(entry_request_id, str)
                or not entry_request_id.strip()
                or len(entry_request_id) > _DEEPLINE_MAX_REQUEST_ID_LENGTH
            ):
                return "invalid", None, False, None
            matches.append(entry)
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
    credits_value = entry.get("credits")
    if isinstance(credits_value, bool) or not isinstance(
        credits_value, (int, float, Decimal)
    ):
        return "invalid", None, False, None
    if charge_state == "failed":
        delta = _decimal(entry.get("delta"))
        if entry.get("status") != "error" or credits_value != 0 or delta != 0:
            return "invalid", None, False, None
        return (
            "matched",
            ProviderCost(
                microusd=0,
                units=Decimal("0"),
                unit_name="credits",
                price_basis="deepline_billing_history_failed_zero",
            ),
            False,
            None,
        )
    if charge_state not in ("posted", "free"):
        return "nonterminal", None, False, None
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
    """Prove a verified Deepline no-bill operation used zero credits.

    A present billing object remains authoritative, including when malformed:
    callers must not replace invalid provider accounting with this fixed zero.
    Only Hunter's authenticated per-call ``no_bill`` contract also proves a
    structured HTTP error costs zero.  Other tools still require a valid
    completed response or exact provider billing.
    """

    tool = parameters.get("tool")
    basis = (
        _DEEPLINE_COMPLETED_NO_BILL_BASIS.get(tool)
        if isinstance(tool, str)
        else None
    )
    if (
        basis is None
        or isinstance(response_status, bool)
        or not isinstance(response_status, int)
        or not isinstance(response_json, Mapping)
        or "billing" in response_json
    ):
        return None
    if response_status != 200:
        error_basis = _DEEPLINE_ERROR_NO_BILL_BASIS.get(tool)
        if error_basis is None or not 400 <= response_status < 600:
            return None
        basis = error_basis
    elif (
        response_json.get("status") != "completed"
        or not isinstance(response_json.get("job_id"), str)
        or not response_json["job_id"].strip()
        or len(response_json["job_id"]) > _DEEPLINE_MAX_REQUEST_ID_LENGTH
        or not isinstance(response_json.get("result"), (Mapping, list))
    ):
        return None
    return ProviderCost(
        microusd=0,
        units=Decimal("0"),
        unit_name="credits",
        price_basis=basis,
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


def openrouter_generation_cost(
    response_json: Any, *, generation_id: str
) -> Optional[ProviderCost]:
    """Read one exact OpenRouter generation charge for the requested id."""

    if not isinstance(response_json, Mapping) or set(response_json) != {"data"}:
        return None
    data = response_json.get("data")
    if (
        not isinstance(data, Mapping)
        or data.get("id") != generation_id
    ):
        return None
    present = [data[name] for name in ("total_cost", "usage") if name in data]
    if not present:
        return None
    costs = [_decimal(value) for value in present]
    if any(cost is None for cost in costs) or len(set(costs)) != 1:
        return None
    usd = costs[0]
    assert usd is not None
    return ProviderCost(
        microusd=_microusd_ceiling(usd),
        units=usd,
        unit_name="usd",
        price_basis="openrouter_generation_cost",
    )


def openrouter_insured_error_cost(
    parameters: Mapping[str, Any],
    pricing: Mapping[str, Any],
    response_status: Any,
    response_json: Any,
) -> Optional[ProviderCost]:
    """Prove one plain OpenRouter 502 has no separately billable work.

    OpenRouter's Zero Completion Insurance covers model inference on an error,
    but not BYOK fees or auxiliary services.  The metadata opt-in and closed
    request checked here rule those exceptions out.  The allowlist limits this
    fallback to known plain models; optional image or web-search catalog prices
    do not prove those features ran.  See:
    https://openrouter.ai/docs/guides/features/zero-completion-insurance and
    https://openrouter.ai/docs/guides/features/router-metadata.
    """

    if (
        isinstance(response_status, bool)
        or not isinstance(response_status, int)
        or response_status != 502
        or not isinstance(response_json, Mapping)
        or not {"error", "openrouter_metadata"}.issubset(response_json)
        or not set(response_json).issubset(_OPENROUTER_ERROR_TOP_LEVEL_FIELDS)
        or any(field in parameters for field in _OPENROUTER_AUXILIARY_REQUEST_FIELDS)
        or not isinstance(pricing, Mapping)
    ):
        return None
    request_price = _decimal(pricing.get("request"))
    if request_price is None or request_price != 0:
        return None
    model = parameters.get("model")
    if (
        not isinstance(model, str)
        or model not in _OPENROUTER_INSURED_PLAIN_MODELS
        or model.endswith(":online")
        or model.startswith("perplexity/sonar")
    ):
        return None
    messages = parameters.get("messages")
    if not isinstance(messages, list) or not messages or any(
        not isinstance(message, Mapping)
        or ("content" in message and not isinstance(message["content"], str))
        for message in messages
    ):
        return None
    error = response_json.get("error")
    if (
        not isinstance(error, Mapping)
        or error.get("code") != 502
        or not isinstance(error.get("message"), str)
        or not error["message"].strip()
        or not set(error).issubset({"code", "message", "metadata"})
    ):
        return None
    metadata = response_json.get("openrouter_metadata")
    if (
        not isinstance(metadata, Mapping)
        or metadata.get("requested") != model
        or metadata.get("is_byok") is not False
        or isinstance(metadata.get("attempt"), bool)
        or not isinstance(metadata.get("attempt"), int)
        or not 0 <= metadata["attempt"] <= 128
    ):
        return None
    pipeline = metadata.get("pipeline", [])
    if not isinstance(pipeline, list) or pipeline:
        return None
    attempts = metadata.get("attempts", [])
    if (
        not isinstance(attempts, list)
        or len(attempts) > 128
        or any(
            not isinstance(attempt, Mapping)
            or isinstance(attempt.get("status"), bool)
            or not isinstance(attempt.get("status"), int)
            or not 400 <= attempt["status"] <= 599
            for attempt in attempts
        )
    ):
        return None
    return ProviderCost(
        microusd=0,
        units=Decimal("0"),
        unit_name="usd",
        price_basis="openrouter_zero_completion_insurance_error_20260911",
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
    "openrouter_generation_cost",
    "openrouter_insured_error_cost",
    "scrapingdog_cost",
]
