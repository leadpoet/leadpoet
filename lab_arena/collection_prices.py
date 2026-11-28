"""Get the actual selected endpoint's cap inputs for each collection model."""

from datetime import datetime, timezone
from decimal import Decimal
import json
from urllib.request import Request
from urllib.error import HTTPError, URLError

from lab_arena import broker
from lab_arena.collection_broker import batch_versions


class CatalogError(ValueError):
    """A bounded operator reason with the usual descriptive validation error."""

    def __init__(self, code, message, *, retry_after=None):
        self.code = code
        self.retry_after = retry_after
        super().__init__(message)


def _catalog(url, open_url):
    try:
        with open_url(
            Request(url, headers={"Accept": "application/json"}), timeout=12
        ) as response:
            raw = response.read(4 * 1024 * 1024 + 1)
    except HTTPError as exc:
        code = (
            "provider_catalog_throttled"
            if exc.code == 429
            else "provider_catalog_http_error"
        )
        raise CatalogError(code, "catalog HTTP request failed", retry_after=(exc.headers or {}).get("Retry-After")) from exc
    except (URLError, TimeoutError, ConnectionError) as exc:
        raise CatalogError(
            "provider_catalog_unreachable", "catalog unavailable"
        ) from exc
    if len(raw) > 4 * 1024 * 1024:
        raise CatalogError("provider_catalog_invalid", "oversized endpoint catalog")
    try:
        return json.loads(raw)["data"]
    except (KeyError, ValueError, TypeError) as exc:
        raise CatalogError(
            "provider_catalog_invalid", "malformed endpoint catalog"
        ) from exc


def _true_zdr_routes(open_url):
    data = _catalog("https://openrouter.ai/api/v1/endpoints/zdr", open_url)
    # Request-level zdr also honors a payer's optional safety-retention setting.
    # Distinguish that tier using the public inventory before leasing a funded job.
    matches = {}
    for row in data:
        key = (row.get("model_id"), row.get("tag"), row.get("name"))
        strict = row.get("safety_retention") is False and row.get("status") == 0
        matches[key] = matches.get(key, True) and strict
    return matches


def endpoint_prices(routes, *, open_url=None, models=None):
    open_url = open_url or broker._DIRECT_URLOPEN
    rows = []
    providers = {}
    revisions = batch_versions(routes)
    selected = dict(routes) if models is None else {m: routes[m] for m in models}
    if not selected:
        raise ValueError("no collection endpoint selected")
    strict = _true_zdr_routes(open_url)
    for model, route in selected.items():
        url = "https://openrouter.ai/api/v1/models/" + model + "/endpoints"
        data = _catalog(url, open_url)
        if data["id"] != model:
            raise CatalogError(
                "provider_catalog_invalid", "wrong endpoint catalog model"
            )
        candidates = [
            row
            for row in data["endpoints"]
            if row["tag"] == route
            and row["status"] == 0
            and row["model_id"] == model
            and row["name"].endswith(" | " + revisions[model])
        ]
        if len(candidates) != 1:
            raise CatalogError(
                "configured_endpoint_unavailable",
                "configured endpoint unavailable for " + model,
            )
        if not strict.get((model, route, candidates[0]["name"]), False):
            raise CatalogError(
                "strict_retention_unavailable",
                "strict zero-retention endpoint unavailable for " + model,
            )
        provider = candidates[0].get("provider_name")
        if not isinstance(provider, str) or not provider.strip():
            raise CatalogError(
                "provider_identity_missing", "missing provider identity for " + model
            )
        providers[model] = provider
        pricing = candidates[0]["pricing"]
        if not {"prompt", "completion"} <= set(pricing):
            raise CatalogError(
                "provider_prices_incomplete", "incomplete endpoint prices"
            )
        tiers = [pricing, *pricing.get("overrides", [])]
        prices = {
            k: str(max(Decimal(str(tier.get(k, 0))) for tier in tiers))
            for k in broker.PRICED_COMPONENTS
        }
        rows.append({"id": model, "pricing": prices})
    table = broker.price_table_from_models_response(
        {"data": rows},
        tuple(selected),
        fetched_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    table["endpoint_providers"] = providers
    return table


def prices_for_assignment(routes, policy):
    """Fetch the exact generator and the panel that may grade its claims or rank the pack."""
    judges = {policy.version(role) for role in policy.roles}
    return lambda model: endpoint_prices(routes, models=sorted(judges | {model}))


def claim_at_current_price(claim, pricing, *, before_claim=None):
    """Read eligibility before contacting providers, then bind any actual lease to its priced model."""
    if callable(pricing):
        preview = claim(expected_model=None)
        if not preview:
            return None, None
        model = preview["model"]
        prices = pricing(model)
        if before_claim:
            before_claim()
        return claim(expected_model=model), prices
    # Injected callers and older tools can supply an already validated full price table.
    if before_claim:
        before_claim()
    return claim(), pricing
