"""Retired gateway services and reward delivery stay absent."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_legacy_fulfillment_gateway_runtime_is_absent():
    assert not any((ROOT / "gateway/fulfillment").glob("*.py"))
    source = (ROOT / "gateway/main.py").read_text()
    assert "gateway.fulfillment" not in source
    assert "ENABLE_FULFILLMENT" not in source
    assert '"/fulfillment' not in source
    assert "/fulfillment/" not in (ROOT / "gateway/edge/nginx.conf").read_text()
    assert "/fulfillment/" not in (ROOT / "gateway/middleware/priority.py").read_text()
