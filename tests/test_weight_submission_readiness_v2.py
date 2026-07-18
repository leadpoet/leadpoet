from __future__ import annotations

import pytest

from gateway.research_lab import api, chain, maintenance
from gateway.tee import verify_weight_submission_ready_v2 as readiness
from research_lab import validator_integration


@pytest.mark.asyncio
async def test_standalone_maintenance_epoch_uses_direct_capable_resolver(
    monkeypatch,
):
    calls = []

    async def resolve(configured_epoch=None):
        calls.append(configured_epoch)
        return 24032, 8651520, "direct_subtensor:finney"

    monkeypatch.setattr(
        chain,
        "resolve_research_lab_evaluation_epoch",
        resolve,
    )

    assert await maintenance._resolve_maintenance_epoch(None) == 24032
    assert await maintenance._resolve_maintenance_epoch(24031) == 24031
    assert calls == [None]


@pytest.mark.asyncio
async def test_weight_readiness_repairs_then_validates_exact_handoff(
    monkeypatch,
):
    calls = []

    async def resolve(epoch):
        assert epoch is None
        return 24032

    async def rewards(**kwargs):
        calls.append(("rewards", kwargs))
        return {"ok": True, "migrated_count": 23}

    async def settlements(**kwargs):
        calls.append(("settlements", kwargs))
        return {"ok": True, "classified_count": 149}

    async def report(**kwargs):
        calls.append(("readiness", kwargs))
        return {
            "ready": True,
            "receipt_coverage": 1.0,
            "historical_classification_coverage": 1.0,
        }

    async def handoff(epoch, x_leadpoet_internal_key):
        calls.append(
            (
                "handoff",
                {
                    "epoch": epoch,
                    "internal_key": x_leadpoet_internal_key,
                },
            )
        )
        return {"handoff": True}

    monkeypatch.setattr(maintenance, "_resolve_maintenance_epoch", resolve)
    monkeypatch.setattr(
        maintenance,
        "backfill_champion_reward_v2_authority",
        rewards,
    )
    monkeypatch.setattr(
        maintenance,
        "backfill_champion_settlement_v2_authority",
        settlements,
    )
    monkeypatch.setattr(
        maintenance,
        "champion_v2_cutover_readiness_report",
        report,
    )
    monkeypatch.setattr(api, "get_research_lab_attested_allocation", handoff)
    monkeypatch.setattr(
        readiness,
        "_validate_handoff",
        lambda value, **_kwargs: {
            "allocation_hash": "sha256:" + "a" * 64,
            "root_receipt_hash": "sha256:" + "b" * 64,
        },
    )

    result = await readiness.verify_weight_submission_ready_v2(repair=True)

    assert result["status"] == "ready"
    assert result["champion_reward_receipts_created"] == 23
    assert result["historical_allocations_classified"] == 149
    assert [name for name, _kwargs in calls] == [
        "rewards",
        "settlements",
        "readiness",
        "handoff",
    ]
    assert calls[0][1] == {
        "epoch": 24032,
        "limit": 10000,
        "dry_run": False,
    }
    assert calls[1][1] == {
        "epoch": 24032,
        "netuid": 71,
        "limit": 10000,
        "dry_run": False,
    }


@pytest.mark.asyncio
async def test_weight_readiness_fails_before_handoff_when_authority_incomplete(
    monkeypatch,
):
    async def resolve(epoch):
        return 24032

    async def report(**_kwargs):
        return {
            "ready": False,
            "receipt_coverage": 0.0,
            "historical_classification_coverage": 0.0,
            "missing": [{"champion_reward_id": "missing"}],
            "missing_historical_classifications": [{"epoch": 24031}],
        }

    async def unexpected_handoff(*_args, **_kwargs):
        raise AssertionError("handoff must not run")

    monkeypatch.setattr(maintenance, "_resolve_maintenance_epoch", resolve)
    monkeypatch.setattr(
        maintenance,
        "champion_v2_cutover_readiness_report",
        report,
    )
    monkeypatch.setattr(
        api,
        "get_research_lab_attested_allocation",
        unexpected_handoff,
    )

    with pytest.raises(
        readiness.WeightSubmissionReadinessV2Error,
        match="obligations=1, historical_allocations=1",
    ):
        await readiness.verify_weight_submission_ready_v2(repair=False)


@pytest.mark.asyncio
async def test_weight_readiness_http_mode_uses_validator_attested_fetch(
    monkeypatch,
):
    fetched = []

    async def resolve(epoch):
        return 24032

    async def report(**_kwargs):
        return {
            "ready": True,
            "receipt_coverage": 1.0,
            "historical_classification_coverage": 1.0,
        }

    def fetch(gateway_url, epoch):
        fetched.append((gateway_url, epoch))
        return {"handoff": True}

    monkeypatch.setattr(maintenance, "_resolve_maintenance_epoch", resolve)
    monkeypatch.setattr(
        maintenance,
        "champion_v2_cutover_readiness_report",
        report,
    )
    monkeypatch.setattr(
        validator_integration,
        "fetch_research_lab_attested_allocation_bundle",
        fetch,
    )
    monkeypatch.setattr(
        readiness,
        "_validate_handoff",
        lambda value, **_kwargs: {
            "allocation_hash": "sha256:" + "a" * 64,
            "root_receipt_hash": "sha256:" + "b" * 64,
        },
    )

    result = await readiness.verify_weight_submission_ready_v2(
        repair=False,
        gateway_url="http://localhost:8000",
    )

    assert result["status"] == "ready"
    assert fetched == [("http://localhost:8000", 24032)]
