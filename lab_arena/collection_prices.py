"""Get the actual selected endpoint's cap inputs for each collection model."""

from datetime import datetime, timezone
from decimal import Decimal
import json
from urllib.request import Request

from lab_arena import broker
from lab_arena.collection_broker import batch_versions


def _catalog(url, open_url):
    with open_url(
        Request(url, headers={"Accept": "application/json"}), timeout=12
    ) as response:
        raw = response.read(4 * 1024 * 1024 + 1)
    if len(raw) > 4 * 1024 * 1024:
        raise ValueError("oversized endpoint catalog")
    return json.loads(raw)["data"]


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
            raise ValueError("wrong endpoint catalog model")
        candidates = [
            row
            for row in data["endpoints"]
            if row["tag"] == route
            and row["status"] == 0
            and row["model_id"] == model
            and row["name"].endswith(" | " + revisions[model])
        ]
        if len(candidates) != 1:
            raise ValueError("configured endpoint unavailable for " + model)
        if not strict.get((model, route, candidates[0]["name"]), False):
            raise ValueError("strict zero-retention endpoint unavailable for " + model)
        provider = candidates[0].get("provider_name")
        if not isinstance(provider, str) or not provider.strip():
            raise ValueError("missing provider identity for " + model)
        providers[model] = provider
        pricing = candidates[0]["pricing"]
        if not {"prompt", "completion"} <= set(pricing):
            raise ValueError("incomplete endpoint prices")
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
