from gateway.research_lab import api, maintenance
from gateway.research_lab.config import ResearchLabGatewayConfig


def test_sealed_proxy_fleet_uses_explicit_worker_capacity(monkeypatch):
    monkeypatch.delenv("RESEARCH_LAB_AUTO_RESEARCH_WEBSHARE_PROXY_1", raising=False)
    config = ResearchLabGatewayConfig(
        hosted_worker_require_proxy=True,
        hosted_worker_proxy_url="",
        hosted_worker_total_workers=7,
    )

    assert api._autoresearch_loop_capacity(config) == 7
    assert maintenance._autoresearch_loop_capacity(config) == 7


def test_sealed_proxy_fleet_without_bound_workers_fails_closed(monkeypatch):
    monkeypatch.delenv("RESEARCH_LAB_AUTO_RESEARCH_WEBSHARE_PROXY_1", raising=False)
    config = ResearchLabGatewayConfig(
        hosted_worker_require_proxy=True,
        hosted_worker_proxy_url="",
        hosted_worker_total_workers=0,
    )

    assert api._autoresearch_loop_capacity(config) == 0
    assert maintenance._autoresearch_loop_capacity(config) == 0
