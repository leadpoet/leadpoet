from types import SimpleNamespace

import pytest

from gateway.research_lab import api, maintenance
from gateway.research_lab import scoring_worker as scoring_worker_module
from gateway.research_lab import worker as hosted_worker_module
from gateway.research_lab.config import ResearchLabGatewayConfig
from gateway.research_lab.provider_profiles_v2 import ProviderProfileV2Error


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


def test_hosted_worker_accepts_only_bound_encrypted_proxy_profile(monkeypatch):
    observed = {}

    def require_profile(**kwargs):
        observed.update(kwargs)
        return {"credential_ref_hashes": {"egress_proxy": "sha256:" + "1" * 64}}

    monkeypatch.setattr(
        hosted_worker_module,
        "require_worker_proxy_profile_v2",
        require_profile,
    )
    worker = object.__new__(hosted_worker_module.ResearchLabHostedWorker)
    worker.config = SimpleNamespace(
        hosted_worker_require_proxy=True,
        hosted_worker_proxy_url="",
        hosted_worker_index=6,
    )

    worker._require_worker_proxy_for_execution()

    assert observed == {
        "execution_role": "gateway_autoresearch",
        "worker_index": 6,
    }


@pytest.mark.asyncio
async def test_scoring_worker_accepts_bound_encrypted_proxy_profile(monkeypatch):
    observed = {}

    def require_profile(**kwargs):
        observed.update(kwargs)
        return {"credential_ref_hashes": {"egress_proxy": "sha256:" + "2" * 64}}

    async def maintenance_state():
        return {"paused": True, "reason": "operator_pause"}

    monkeypatch.setattr(
        scoring_worker_module,
        "require_worker_proxy_profile_v2",
        require_profile,
    )
    monkeypatch.setattr(
        scoring_worker_module,
        "get_scoring_maintenance_state",
        maintenance_state,
    )
    worker = object.__new__(
        scoring_worker_module.ResearchLabGatewayScoringWorker
    )
    worker.config = SimpleNamespace(
        scoring_worker_enabled=True,
        production_writes_enabled=True,
        evaluation_bundles_enabled=True,
        scoring_worker_require_proxy=True,
        scoring_worker_index=12,
    )
    worker.proxy_url = ""
    worker.worker_ref = "scoring-test"

    result = await worker.run_once()

    assert result["status"] == "maintenance_paused"
    assert observed == {
        "execution_role": "gateway_scoring",
        "worker_index": 12,
    }


@pytest.mark.asyncio
async def test_scoring_worker_fails_closed_without_encrypted_proxy_profile(
    monkeypatch,
):
    def reject_profile(**_kwargs):
        raise ProviderProfileV2Error("profile missing")

    monkeypatch.setattr(
        scoring_worker_module,
        "require_worker_proxy_profile_v2",
        reject_profile,
    )
    worker = object.__new__(
        scoring_worker_module.ResearchLabGatewayScoringWorker
    )
    worker.config = SimpleNamespace(
        scoring_worker_enabled=True,
        production_writes_enabled=True,
        evaluation_bundles_enabled=True,
        scoring_worker_require_proxy=True,
        scoring_worker_index=12,
    )
    worker.proxy_url = ""
    worker.worker_ref = "scoring-test"

    result = await worker.run_once()

    assert result["status"] == "scoring_worker_proxy_required"
