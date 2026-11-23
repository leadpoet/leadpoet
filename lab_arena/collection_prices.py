"""Get the actual selected endpoint's cap inputs for each collection model."""

from datetime import datetime, timezone
from decimal import Decimal
import json
from urllib.request import Request

from lab_arena import broker
from lab_arena.collection_broker import VERSIONS


def endpoint_prices(routes, *, open_url=None):
    open_url = open_url or broker._DIRECT_URLOPEN
    rows = []
    if set(routes) != set(VERSIONS):
        raise ValueError("incomplete collection route set")
    for model, route in routes.items():
        url = "https://openrouter.ai/api/v1/models/" + model + "/endpoints"
        with open_url(
            Request(url, headers={"Accept": "application/json"}), timeout=12
        ) as response:
            raw = response.read(4 * 1024 * 1024 + 1)
        if len(raw) > 4 * 1024 * 1024:
            raise ValueError("oversized endpoint catalog")
        data = json.loads(raw)["data"]
        if data["id"] != model:
            raise ValueError("wrong endpoint catalog model")
        candidates = [
            row
            for row in data["endpoints"]
            if row["tag"] == route
            and row["status"] == 0
            and row["model_id"] == model
            and row["name"].endswith(" | " + VERSIONS[model])
        ]
        if len(candidates) != 1:
            raise ValueError("configured endpoint unavailable for " + model)
        pricing = candidates[0]["pricing"]
        if not {"prompt", "completion"} <= set(pricing):
            raise ValueError("incomplete endpoint prices")
        tiers = [pricing, *pricing.get("overrides", [])]
        prices = {
            k: str(max(Decimal(str(tier.get(k, 0))) for tier in tiers))
            for k in broker.PRICED_COMPONENTS
        }
        rows.append({"id": model, "pricing": prices})
    return broker.price_table_from_models_response(
        {"data": rows},
        tuple(routes),
        fetched_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
