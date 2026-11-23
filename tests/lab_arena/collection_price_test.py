import io
import json

import pytest

from lab_arena.collection_broker import VERSIONS
from lab_arena.collection_prices import endpoint_prices


def test_prices_use_exact_provider_revision_and_highest_context_tier():
    routes = dict.fromkeys(VERSIONS, "approved")
    def quote(request, timeout):
        model = request.full_url.removeprefix("https://openrouter.ai/api/v1/models/").removesuffix("/endpoints")
        row = {"tag":"approved","name":"Provider | "+VERSIONS[model],"model_id":model,"status":0,
               "pricing":{"prompt":"0.000001","completion":"0.000002","overrides":[{"prompt":"0.000004"}]}}
        return io.BytesIO(json.dumps({"data":{"id":model,"endpoints":[row,dict(row,tag="other")]}}).encode())
    result = endpoint_prices(routes,open_url=quote)
    assert all(p["prompt"]=="0.000004" for p in result["models"].values())
    routes[next(iter(routes))]="unknown"
    with pytest.raises(ValueError,match="unavailable"):
        endpoint_prices(routes,open_url=quote)
