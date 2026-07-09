from __future__ import annotations

from decimal import Decimal

import pytest

from gateway.research_lab import alpha_pricing
from gateway.research_lab.alpha_pricing import (
    compute_alpha_price_valuation,
    inject_alpha_price_valuation,
    resolve_epoch_alpha_price_valuation,
    static_alpha_price_fallback,
)
from leadpoet_verifier.economics import allocate_research_lab_epoch


def _policy(usd_per_0_1_percent_epoch: float) -> dict[str, object]:
    return {
        "policy_id": "test-dynamic-alpha-pricing",
        "enabled": True,
        "research_lab_emission_percent": 30.0,
        "reward_epochs": 20,
        "reimbursement_epochs": 20,
        "reimbursement_allow_overpay_without_champions": True,
        "reimbursement_max_cost_multiplier_with_champions": 1.0,
        "champion_min_alpha_percent": 7.0,
        "champion_extra_alpha_percent_per_point": 0.3,
        "champion_max_alpha_percent": 15.0,
        "champion_placeholder_alpha_percent": 0.0001,
        "champion_threshold_points": 1.0,
        "usd_per_0_1_percent_epoch": usd_per_0_1_percent_epoch,
    }


def test_alpha_price_formula_matches_live_probe_shape():
    class BalanceLike:
        tao = "0.0067026"

    valuation = compute_alpha_price_valuation(
        network="finney",
        netuid=71,
        epoch=123,
        tao_per_alpha=BalanceLike(),
        tao_usd="202.6",
        miner_alpha_per_epoch="147.6",
        fetched_at="2026-07-09T00:00:00Z",
    )

    assert valuation.alpha_usd == Decimal("1.357946760")
    assert valuation.usd_per_0_1_percent_epoch == Decimal("0.200433")
    assert valuation.policy_doc()["pricing_status"] == "live"
    assert valuation.policy_doc()["miner_alpha_per_epoch"] == 147.6


@pytest.mark.parametrize(
    ("compute_usd", "scale", "expected_percent"),
    [
        (0.50, 1.0, 0.012474),
        (1.00, 1.0, 0.024949),
        (3.00, 1.0, 0.074846),
        (10.00, 1.0, 0.249486),
        (30.00, 1.0, 0.748458),
        (100.00, 1.0, 2.494861),
        (3.00, 0.5, 0.037423),
    ],
)
def test_compute_spend_converts_to_epoch_alpha_percent(compute_usd: float, scale: float, expected_percent: float):
    valuation = compute_alpha_price_valuation(
        network="finney",
        netuid=71,
        epoch=123,
        tao_per_alpha="0.006702566",
        tao_usd="202.58",
        miner_alpha_per_epoch="147.6",
        fetched_at="2026-07-09T00:00:00Z",
    )
    usd_per_0_1 = valuation.usd_per_0_1_percent_epoch
    per_epoch_usd = Decimal(str(compute_usd)) * Decimal(str(scale)) / Decimal("20")
    alpha_percent = (per_epoch_usd / usd_per_0_1) * Decimal("0.1")

    assert float(alpha_percent.quantize(Decimal("0.000001"))) == pytest.approx(expected_percent, abs=0.000002)


def test_dynamic_valuation_drives_reimbursement_allocation_without_touching_champion_surplus():
    valuation = compute_alpha_price_valuation(
        network="finney",
        netuid=71,
        epoch=500,
        tao_per_alpha="0.006702566",
        tao_usd="202.58",
        miner_alpha_per_epoch="147.6",
        fetched_at="2026-07-09T00:00:00Z",
    )
    policy = inject_alpha_price_valuation(_policy(0.162), valuation)
    reimbursement = {
        "uid": 11,
        "miner_hotkey": "5Freimburse",
        "source_id": "reimbursement_schedule:test",
        "start_epoch": 500,
        "epoch_count": 20,
        "target_reimbursement_microusd": 3_000_000,
    }
    champion = {
        "uid": 22,
        "miner_hotkey": "5Fchampion",
        "source_id": "champion_reward:test",
        "start_epoch": 500,
        "epoch_count": 20,
        "improvement_points": 6.0,
    }

    allocation = allocate_research_lab_epoch(500, policy, [reimbursement], [champion])

    assert allocation["reimbursement_allocations"][0]["paid_alpha_percent"] == pytest.approx(0.074846, abs=0.000002)
    assert allocation["reimbursement_allocations"][0]["reason"] == "full_reimbursement"
    assert allocation["champion_alpha_percent"] > 0
    assert allocation["unallocated_percent"] == pytest.approx(0.0)
    assert allocation["input_hash"].startswith("sha256:")


@pytest.mark.asyncio
async def test_epoch_price_cache_fetches_once_per_epoch(monkeypatch):
    alpha_pricing._CACHE.clear()
    calls = {"alpha": 0, "tao": 0}

    async def fake_alpha(*, network: str, netuid: int, timeout_seconds: float):
        calls["alpha"] += 1
        return Decimal("0.006702566")

    async def fake_tao(*, timeout_seconds: float):
        calls["tao"] += 1
        return Decimal("202.58")

    monkeypatch.setattr(alpha_pricing, "_fetch_tao_per_alpha", fake_alpha)
    monkeypatch.setattr(alpha_pricing, "_fetch_tao_usd", fake_tao)

    first = await resolve_epoch_alpha_price_valuation(
        network="finney",
        netuid=71,
        epoch=900,
        enabled=True,
        require_live=True,
        miner_alpha_per_epoch=147.6,
        static_usd_per_0_1_percent_epoch=0.162,
    )
    second = await resolve_epoch_alpha_price_valuation(
        network="finney",
        netuid=71,
        epoch=900,
        enabled=True,
        require_live=True,
        miner_alpha_per_epoch=147.6,
        static_usd_per_0_1_percent_epoch=0.162,
    )

    assert calls == {"alpha": 1, "tao": 1}
    assert first.pricing_status == "live"
    assert second.pricing_status == "cache_hit"
    assert second.usd_per_0_1_percent_epoch == first.usd_per_0_1_percent_epoch


@pytest.mark.asyncio
async def test_epoch_price_fetch_failure_uses_static_fallback(monkeypatch):
    alpha_pricing._CACHE.clear()
    calls = {"alpha": 0}

    async def fail_alpha(*, network: str, netuid: int, timeout_seconds: float):
        calls["alpha"] += 1
        raise RuntimeError("subtensor unavailable")

    async def fail_tao(*, timeout_seconds: float):
        raise RuntimeError("coingecko unavailable")

    monkeypatch.setattr(alpha_pricing, "_fetch_tao_per_alpha", fail_alpha)
    monkeypatch.setattr(alpha_pricing, "_fetch_tao_usd", fail_tao)

    valuation = await resolve_epoch_alpha_price_valuation(
        network="finney",
        netuid=71,
        epoch=901,
        enabled=True,
        require_live=False,
        miner_alpha_per_epoch=147.6,
        static_usd_per_0_1_percent_epoch=0.162,
        max_attempts=1,
    )

    assert valuation.pricing_status == "static_fallback"
    assert valuation.usd_per_0_1_percent_epoch == Decimal("0.162000")
    assert "unavailable" in valuation.fallback_reason

    await resolve_epoch_alpha_price_valuation(
        network="finney",
        netuid=71,
        epoch=901,
        enabled=True,
        require_live=False,
        miner_alpha_per_epoch=147.6,
        static_usd_per_0_1_percent_epoch=0.162,
        max_attempts=1,
    )
    assert calls["alpha"] == 2


def test_static_fallback_doc_is_auditable_when_dynamic_pricing_disabled():
    valuation = static_alpha_price_fallback(
        network="finney",
        netuid=71,
        epoch=902,
        static_usd_per_0_1_percent_epoch=0.162,
        miner_alpha_per_epoch=147.6,
        reason="dynamic_alpha_price_disabled",
    )

    doc = valuation.policy_doc()
    assert doc["pricing_status"] == "static_fallback"
    assert doc["usd_per_0_1_percent_epoch"] == 0.162
    assert doc["fallback_reason"] == "dynamic_alpha_price_disabled"
