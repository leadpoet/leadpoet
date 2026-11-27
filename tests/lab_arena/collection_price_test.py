import io
import json

import pytest

from lab_arena.collection_broker import VERSIONS
from lab_arena.collection_prices import CatalogError, endpoint_prices


def test_prices_use_exact_provider_revision_and_highest_context_tier():
    routes = dict.fromkeys(VERSIONS, "approved")

    def quote(request, timeout):
        if request.full_url.endswith("/endpoints/zdr"):
            return io.BytesIO(
                json.dumps(
                    {
                        "data": [
                            {
                                "model_id": m,
                                "tag": "approved",
                                "name": "Provider | " + v,
                                "status": 0,
                                "safety_retention": False,
                            }
                            for m, v in VERSIONS.items()
                        ]
                    }
                ).encode()
            )
        model = request.full_url.removeprefix(
            "https://openrouter.ai/api/v1/models/"
        ).removesuffix("/endpoints")
        row = {
            "tag": "approved",
            "provider_name": "Provider",
            "name": "Provider | " + VERSIONS[model],
            "model_id": model,
            "status": 0,
            "pricing": {
                "prompt": "0.000001",
                "completion": "0.000002",
                "overrides": [{"prompt": "0.000004"}],
            },
        }
        return io.BytesIO(
            json.dumps(
                {"data": {"id": model, "endpoints": [row, dict(row, tag="other")]}}
            ).encode()
        )

    result = endpoint_prices(routes, open_url=quote)
    assert all(p["prompt"] == "0.000004" for p in result["models"].values())
    routes[next(iter(routes))] = "unknown"
    with pytest.raises(CatalogError, match="unavailable") as error:
        endpoint_prices(routes, open_url=quote)
    assert error.value.code == "configured_endpoint_unavailable"


@pytest.mark.parametrize("safety", [True, None, "false"])
def test_safety_retention_or_unknown_tier_cannot_admit_a_funding_job(safety):
    def catalog(request, timeout):
        if request.full_url.endswith("/endpoints/zdr"):
            body = {
                "data": [
                    {
                        "model_id": m,
                        "name": "Provider | " + v,
                        "tag": "p",
                        "status": 0,
                        "safety_retention": safety,
                    }
                    for m, v in VERSIONS.items()
                ]
            }
        else:
            m = request.full_url.split("/models/")[1].removesuffix("/endpoints")
            body = {
                "data": {
                    "id": m,
                    "endpoints": [
                        {
                            "model_id": m,
                            "name": "Provider | " + VERSIONS[m],
                            "tag": "p",
                            "status": 0,
                        }
                    ],
                }
            }
        return io.BytesIO(json.dumps(body).encode())

    with pytest.raises(CatalogError, match="strict zero-retention") as error:
        endpoint_prices(dict.fromkeys(VERSIONS, "p"), open_url=catalog)
    assert error.value.code == "strict_retention_unavailable"
