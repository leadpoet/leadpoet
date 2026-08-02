from __future__ import annotations

import pytest
from fastapi import HTTPException

from gateway.research_lab import api, chain, maintenance
from gateway.tee import verify_weight_submission_ready_v2 as readiness
from gateway.utils.tee_artifact_store_v2 import TEEArtifactStoreV2Error
from research_lab import validator_integration


@pytest.fixture(autouse=True)
def _default_historical_compute_fallback_backfill(monkeypatch):
    async def covered(**_kwargs):
        return {"ok": True, "classified_count": 0}

    monkeypatch.setattr(
        maintenance,
        "backfill_historical_compute_fallback_v2_authority",
        covered,
    )


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
async def test_storage_read_preflight_exercises_full_authority_read_without_repair(
    monkeypatch,
):
    from gateway.research_lab import champion_settlement_v2 as settlement

    calls = []

    async def resolve(epoch):
        calls.append(("resolve", epoch))
        return 24153

    async def report(**kwargs):
        calls.append(("report", kwargs))
        return {
            "ready": False,
            "receipt_coverage": 1.0,
            "historical_classification_coverage": 0.75,
        }

    async def bootstrap(**kwargs):
        calls.append(("bootstrap", kwargs))
        return {
            "activation_epoch": 24100,
            "target_epoch": 24152,
            "settled_through_epoch": 24152,
            "backlog_epoch_count": 0,
        }

    monkeypatch.setattr(maintenance, "_resolve_maintenance_epoch", resolve)
    monkeypatch.setattr(
        maintenance,
        "champion_v2_cutover_readiness_report",
        report,
    )
    monkeypatch.setattr(
        settlement,
        "validate_chain_realized_settlement_bootstrap_v1",
        bootstrap,
    )

    result = await readiness.verify_weight_submission_storage_readable_v2(
        netuid=71,
    )

    assert result == {
        "schema_version": "leadpoet.weight_submission_storage_readiness.v2",
        "status": "readable",
        "epoch": 24153,
        "ancestry_safe_epoch": 24153,
        "netuid": 71,
        "authority_ready": False,
        "receipt_coverage": 1.0,
        "historical_classification_coverage": 0.75,
    }
    assert calls == [
        ("resolve", None),
        ("bootstrap", {"netuid": 71, "target_epoch": 24152}),
        ("report", {"epoch": 24153, "netuid": 71}),
    ]


@pytest.mark.asyncio
async def test_storage_read_preflight_accepts_only_pristine_settlement_bootstrap(
    monkeypatch,
):
    from gateway.research_lab import champion_settlement_v2 as settlement

    async def resolve(_epoch):
        return 24206

    async def report(**_kwargs):
        raise AssertionError("incomplete settlement suffix must short-circuit")

    async def bootstrap(**kwargs):
        assert kwargs == {"netuid": 71, "target_epoch": 24205}
        return {
            "schema_version": (
                "leadpoet.chain_realized_settlement_bootstrap_readiness.v1"
            ),
            "status": "pristine_bootstrap_pending",
            "netuid": 71,
            "activation_epoch": 24202,
            "target_epoch": 24205,
            "backlog_epoch_count": 4,
            "source_bundle_hash": "sha256:" + "a" * 64,
            "source_finalized_block": 8717384,
            "validated_finalized_candidate_epochs": [24202, 24205],
        }

    monkeypatch.setattr(maintenance, "_resolve_maintenance_epoch", resolve)
    monkeypatch.setattr(
        maintenance,
        "champion_v2_cutover_readiness_report",
        report,
    )
    monkeypatch.setattr(
        settlement,
        "validate_chain_realized_settlement_bootstrap_v1",
        bootstrap,
    )

    result = await readiness.verify_weight_submission_storage_readable_v2(
        netuid=71,
    )

    assert result["status"] == "readable"
    assert result["authority_ready"] is False
    assert result["chain_realized_settlement_bootstrap"][
        "status"
    ] == "pristine_bootstrap_pending"
    assert result["chain_realized_settlement_bootstrap"][
        "backlog_epoch_count"
    ] == 4
    assert result["ancestry_safe_epoch"] == 24202


@pytest.mark.asyncio
async def test_storage_read_preflight_resumes_at_first_unsettled_epoch(
    monkeypatch,
):
    from gateway.research_lab import champion_settlement_v2 as settlement
    report_calls = []

    async def resolve(_epoch):
        return 24304

    async def report(**_kwargs):
        report_calls.append(dict(_kwargs))
        return {
            "ready": True,
            "receipt_coverage": 1.0,
            "historical_classification_coverage": 1.0,
        }

    async def bootstrap(**_kwargs):
        return {
            "activation_epoch": 24202,
            "target_epoch": 24303,
            "settled_through_epoch": 24302,
            "backlog_epoch_count": 1,
        }

    monkeypatch.setattr(maintenance, "_resolve_maintenance_epoch", resolve)
    monkeypatch.setattr(
        maintenance,
        "champion_v2_cutover_readiness_report",
        report,
    )
    monkeypatch.setattr(
        settlement,
        "validate_chain_realized_settlement_bootstrap_v1",
        bootstrap,
    )

    result = await readiness.verify_weight_submission_storage_readable_v2(
        netuid=71,
    )

    assert result["epoch"] == 24304
    assert result["ancestry_safe_epoch"] == 24303
    assert result["authority_ready"] is False
    assert report_calls == []


def test_storage_read_preflight_rejects_inconsistent_safe_epoch():
    with pytest.raises(
        readiness.WeightSubmissionReadinessV2Error,
        match="bootstrap report is inconsistent",
    ):
        readiness._ancestry_safe_epoch_from_storage_readiness(
            effective_epoch=24304,
            bootstrap={
                "activation_epoch": 24202,
                "target_epoch": 24303,
                "settled_through_epoch": 24302,
                "backlog_epoch_count": 2,
            },
        )


@pytest.mark.asyncio
async def test_chain_settlement_repair_proves_durable_suffix(monkeypatch):
    from gateway.research_lab import champion_settlement_v2, v2_authority

    monkeypatch.setenv("RESEARCH_LAB_INTERNAL_API_KEY", "validator-secret")
    resolved = iter((24304, 24304, 24304))
    calls = []

    async def resolve(_epoch):
        return next(resolved)

    async def ensure(**kwargs):
        calls.append(("ensure", kwargs))
        return [{"status": "settled"}]

    async def validate(**kwargs):
        calls.append(("validate", kwargs))
        return {
            "activation_epoch": 24202,
            "target_epoch": 24303,
            "settled_through_epoch": 24303,
            "backlog_epoch_count": 0,
        }

    monkeypatch.setattr(maintenance, "_resolve_maintenance_epoch", resolve)
    monkeypatch.setattr(
        v2_authority,
        "ensure_chain_realized_settlements_v1",
        ensure,
    )
    monkeypatch.setattr(
        champion_settlement_v2,
        "validate_chain_realized_settlement_bootstrap_v1",
        validate,
    )

    result = await readiness.repair_chain_realized_settlements_v1(netuid=71)

    assert result == {
        "schema_version": "leadpoet.chain_realized_settlement_repair.v1",
        "status": "ready",
        "epoch": 24304,
        "observed_epoch": 24304,
        "netuid": 71,
        "settled_through_epoch": 24303,
        "repaired_epoch_count": 1,
    }
    assert calls == [
        ("ensure", {"epoch_id": 24304, "netuid": 71}),
        ("validate", {"netuid": 71, "target_epoch": 24303}),
    ]


@pytest.mark.asyncio
async def test_chain_settlement_repair_rejects_incomplete_readback(
    monkeypatch,
):
    from gateway.research_lab import champion_settlement_v2, v2_authority

    monkeypatch.setenv("RESEARCH_LAB_INTERNAL_API_KEY", "validator-secret")

    async def resolve(_epoch):
        return 24304

    async def ensure(**_kwargs):
        return []

    async def validate(**_kwargs):
        return {
            "activation_epoch": 24202,
            "target_epoch": 24303,
            "settled_through_epoch": 24302,
            "backlog_epoch_count": 1,
        }

    monkeypatch.setattr(maintenance, "_resolve_maintenance_epoch", resolve)
    monkeypatch.setattr(
        v2_authority,
        "ensure_chain_realized_settlements_v1",
        ensure,
    )
    monkeypatch.setattr(
        champion_settlement_v2,
        "validate_chain_realized_settlement_bootstrap_v1",
        validate,
    )

    with pytest.raises(
        readiness.WeightSubmissionReadinessV2Error,
        match="left an incomplete suffix",
    ):
        await readiness.repair_chain_realized_settlements_v1(netuid=71)


@pytest.mark.asyncio
async def test_storage_read_preflight_does_not_mask_other_settlement_failures(
    monkeypatch,
):
    from gateway.research_lab import champion_settlement_v2 as settlement

    async def resolve(_epoch):
        return 24206

    async def report(**_kwargs):
        raise settlement.ChampionSettlementV2Error(
            "chain realized settlement activation is invalid"
        )

    async def bootstrap(**_kwargs):
        return {
            "activation_epoch": 24202,
            "target_epoch": 24205,
            "settled_through_epoch": 24205,
            "backlog_epoch_count": 0,
        }

    monkeypatch.setattr(maintenance, "_resolve_maintenance_epoch", resolve)
    monkeypatch.setattr(
        maintenance,
        "champion_v2_cutover_readiness_report",
        report,
    )
    monkeypatch.setattr(
        settlement,
        "validate_chain_realized_settlement_bootstrap_v1",
        bootstrap,
    )

    with pytest.raises(
        settlement.ChampionSettlementV2Error,
        match="activation is invalid",
    ):
        await readiness.verify_weight_submission_storage_readable_v2(
            netuid=71,
        )


@pytest.mark.asyncio
async def test_weight_readiness_repair_requires_internal_key_before_writes(
    monkeypatch,
):
    monkeypatch.delenv("RESEARCH_LAB_INTERNAL_API_KEY", raising=False)

    async def resolve(_epoch):
        return 24032

    async def unexpected_write(**_kwargs):
        raise AssertionError("repair writes must not run without authentication")

    monkeypatch.setattr(maintenance, "_resolve_maintenance_epoch", resolve)
    monkeypatch.setattr(
        maintenance,
        "backfill_source_add_reward_v2_authority",
        unexpected_write,
    )

    with pytest.raises(
        readiness.WeightSubmissionReadinessV2Error,
        match="internal API key is not configured",
    ):
        await readiness.verify_weight_submission_ready_v2(repair=True)


@pytest.mark.asyncio
async def test_weight_readiness_repairs_then_validates_exact_handoff(
    monkeypatch,
):
    calls = []
    monkeypatch.setenv("RESEARCH_LAB_INTERNAL_API_KEY", "validator-secret")

    async def resolve(epoch):
        assert epoch is None
        return 24032

    async def rewards(**kwargs):
        calls.append(("rewards", kwargs))
        return {"ok": True, "migrated_count": 23}

    async def source_rewards(**kwargs):
        calls.append(("source_rewards", kwargs))
        return {"ok": True, "migrated_count": 1}

    async def settlements(**kwargs):
        calls.append(("settlements", kwargs))
        return {"ok": True, "classified_count": 149}

    async def fallback(**kwargs):
        calls.append(("fallback", kwargs))
        return {"ok": True, "classified_count": 1}

    async def unexpected_report(**_kwargs):
        raise AssertionError(
            "the authoritative allocation build owns the cutover gate"
        )

    async def handoff(*, epoch, current_epoch, internal_key):
        calls.append(
            (
                "handoff",
                {
                    "epoch": epoch,
                    "current_epoch": current_epoch,
                    "internal_key": internal_key,
                },
            )
        )
        if len([item for item in calls if item[0] == "handoff"]) == 1:
            cause = RuntimeError(
                "champion V2 cutover blocked: 1 obligations and 1 "
                "historical allocations lack authoritative classifications"
            )
            raise HTTPException(
                status_code=500,
                detail="Research Lab attested allocation unavailable",
            ) from cause
        return {"handoff": True}

    monkeypatch.setattr(maintenance, "_resolve_maintenance_epoch", resolve)
    monkeypatch.setattr(
        maintenance,
        "backfill_source_add_reward_v2_authority",
        source_rewards,
    )
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
        "backfill_historical_compute_fallback_v2_authority",
        fallback,
    )
    monkeypatch.setattr(
        maintenance,
        "champion_v2_cutover_readiness_report",
        unexpected_report,
    )
    monkeypatch.setattr(
        api,
        "_get_research_lab_attested_allocation_for_resolved_current_epoch",
        handoff,
    )
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
    assert result["source_add_reward_receipts_created"] == 1
    assert result["champion_reward_receipts_created"] == 23
    assert result["historical_allocations_classified"] == 149
    assert result["historical_compute_fallbacks_classified"] == 1
    assert [name for name, _kwargs in calls] == [
        "handoff",
        "source_rewards",
        "rewards",
        "settlements",
        "fallback",
        "handoff",
    ]
    assert calls[1][1] == {
        "epoch": 24032,
        "limit": 10000,
        "dry_run": False,
    }
    assert calls[2][1] == {
        "epoch": 24032,
        "limit": 10000,
        "dry_run": False,
    }
    assert calls[3][1] == {
        "epoch": 24032,
        "netuid": 71,
        "limit": 10000,
        "dry_run": False,
    }
    assert calls[4][1] == {
        "epoch": 24032,
        "netuid": 71,
        "dry_run": False,
    }
    assert calls[0][1] == calls[5][1] == {
        "epoch": 24032,
        "current_epoch": 24032,
        "internal_key": "validator-secret",
    }


@pytest.mark.asyncio
async def test_weight_readiness_accepts_already_covered_reward(monkeypatch):
    calls = []
    monkeypatch.setenv("RESEARCH_LAB_INTERNAL_API_KEY", "validator-secret")

    async def resolve(_epoch):
        return 24036

    async def unexpected_repair(**_kwargs):
        raise AssertionError("covered authority must not run historical repair")

    async def unexpected_report(**_kwargs):
        raise AssertionError(
            "the authoritative allocation build owns the cutover gate"
        )

    async def handoff(*, epoch, current_epoch, internal_key):
        assert epoch == current_epoch == 24036
        assert internal_key == "validator-secret"
        calls.append("handoff")
        return {"handoff": True}

    monkeypatch.setattr(maintenance, "_resolve_maintenance_epoch", resolve)
    monkeypatch.setattr(
        maintenance,
        "backfill_source_add_reward_v2_authority",
        unexpected_repair,
    )
    monkeypatch.setattr(
        maintenance,
        "backfill_champion_reward_v2_authority",
        unexpected_repair,
    )
    monkeypatch.setattr(
        maintenance,
        "backfill_champion_settlement_v2_authority",
        unexpected_repair,
    )
    monkeypatch.setattr(
        maintenance,
        "champion_v2_cutover_readiness_report",
        unexpected_report,
    )
    monkeypatch.setattr(
        api,
        "_get_research_lab_attested_allocation_for_resolved_current_epoch",
        handoff,
    )
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
    assert result["source_add_reward_receipts_created"] == 0
    assert result["champion_reward_receipts_created"] == 0
    assert result["historical_allocations_classified"] == 0
    assert calls == ["handoff"]


@pytest.mark.asyncio
async def test_weight_readiness_repair_does_not_mask_invalid_handoff(
    monkeypatch,
):
    repair_calls = []
    monkeypatch.setenv("RESEARCH_LAB_INTERNAL_API_KEY", "validator-secret")

    async def resolve(_epoch):
        return 24036

    async def unexpected_repair(**_kwargs):
        repair_calls.append(True)
        return {"ok": True, "migrated_count": 0}

    async def handoff(*, epoch, current_epoch, internal_key):
        assert epoch == current_epoch == 24036
        assert internal_key == "validator-secret"
        return {"handoff": "invalid"}

    monkeypatch.setattr(maintenance, "_resolve_maintenance_epoch", resolve)
    monkeypatch.setattr(
        maintenance,
        "backfill_source_add_reward_v2_authority",
        unexpected_repair,
    )
    monkeypatch.setattr(
        maintenance,
        "backfill_champion_reward_v2_authority",
        unexpected_repair,
    )
    monkeypatch.setattr(
        maintenance,
        "backfill_champion_settlement_v2_authority",
        unexpected_repair,
    )
    monkeypatch.setattr(
        api,
        "_get_research_lab_attested_allocation_for_resolved_current_epoch",
        handoff,
    )
    monkeypatch.setattr(
        readiness,
        "_validate_handoff",
        lambda value, **_kwargs: (_ for _ in ()).throw(
            readiness.WeightSubmissionReadinessV2Error(
                "allocation signature is invalid"
            )
        ),
    )

    with pytest.raises(
        readiness.WeightSubmissionReadinessV2Error,
        match="allocation signature is invalid",
    ):
        await readiness.verify_weight_submission_ready_v2(repair=True)

    assert repair_calls == []


@pytest.mark.asyncio
async def test_weight_readiness_repair_retries_transport_without_backfill(
    monkeypatch,
):
    calls = []
    monkeypatch.setenv("RESEARCH_LAB_INTERNAL_API_KEY", "validator-secret")

    async def resolve(_epoch):
        return 24036

    async def unexpected_repair(**_kwargs):
        raise AssertionError("transport recovery must not run authority repair")

    async def handoff(*, epoch, current_epoch, internal_key):
        assert epoch == current_epoch == 24036
        assert internal_key == "validator-secret"
        calls.append("handoff")
        if len(calls) == 1:
            raise RuntimeError(
                "enclave rejected artifact persistence: unexpected_eof"
            )
        return {"handoff": True}

    monkeypatch.setattr(maintenance, "_resolve_maintenance_epoch", resolve)
    monkeypatch.setattr(
        maintenance,
        "backfill_source_add_reward_v2_authority",
        unexpected_repair,
    )
    monkeypatch.setattr(
        maintenance,
        "backfill_champion_reward_v2_authority",
        unexpected_repair,
    )
    monkeypatch.setattr(
        maintenance,
        "backfill_champion_settlement_v2_authority",
        unexpected_repair,
    )
    monkeypatch.setattr(
        api,
        "_get_research_lab_attested_allocation_for_resolved_current_epoch",
        handoff,
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
        repair=True,
        http_attempts=2,
        http_retry_seconds=0,
    )

    assert result["status"] == "ready"
    assert result["source_add_reward_receipts_created"] == 0
    assert result["champion_reward_receipts_created"] == 0
    assert result["historical_allocations_classified"] == 0
    assert calls == ["handoff", "handoff"]


@pytest.mark.asyncio
async def test_weight_readiness_fails_closed_when_allocation_authority_incomplete(
    monkeypatch,
):
    async def resolve(epoch):
        return 24032

    async def unexpected_report(**_kwargs):
        raise AssertionError(
            "the authoritative allocation build owns the cutover gate"
        )

    async def blocked_handoff(*_args, **_kwargs):
        raise RuntimeError(
            "champion V2 cutover blocked: 1 obligations and 1 historical "
            "allocations lack authoritative classifications"
        )

    monkeypatch.setattr(maintenance, "_resolve_maintenance_epoch", resolve)
    monkeypatch.setattr(
        maintenance,
        "champion_v2_cutover_readiness_report",
        unexpected_report,
    )
    monkeypatch.setattr(
        api,
        "get_research_lab_attested_allocation",
        blocked_handoff,
    )

    with pytest.raises(RuntimeError, match="champion V2 cutover blocked"):
        await readiness.verify_weight_submission_ready_v2(repair=False)


@pytest.mark.asyncio
async def test_weight_readiness_http_mode_uses_validator_attested_fetch(
    monkeypatch,
):
    fetched = []

    async def resolve(epoch):
        return 24032

    async def unexpected_report(**_kwargs):
        raise AssertionError(
            "HTTP readiness must trust only the validated gateway handoff"
        )

    def fetch(gateway_url, epoch, *, timeout_seconds):
        fetched.append((gateway_url, epoch, timeout_seconds))
        return {"handoff": True}

    monkeypatch.setattr(maintenance, "_resolve_maintenance_epoch", resolve)
    monkeypatch.setattr(
        maintenance,
        "champion_v2_cutover_readiness_report",
        unexpected_report,
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
    assert fetched == [("http://localhost:8000", 24032, 90)]


@pytest.mark.asyncio
async def test_weight_readiness_http_mode_allows_fresh_epoch_build_deadline(
    monkeypatch,
):
    calls = []

    async def resolve(_epoch):
        # The post-launch check can cross into the epoch after pre-launch repair.
        return 24033

    async def report(**_kwargs):
        return {
            "ready": True,
            "receipt_coverage": 1.0,
            "historical_classification_coverage": 1.0,
        }

    def fetch(gateway_url, epoch, *, timeout_seconds):
        calls.append((gateway_url, epoch, timeout_seconds))
        if timeout_seconds < 284:
            raise TimeoutError("fresh epoch allocation build exceeded deadline")
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
        http_attempts=1,
        http_timeout_seconds=360,
    )

    assert result["status"] == "ready"
    assert calls == [("http://localhost:8000", 24033, 360)]


@pytest.mark.asyncio
async def test_weight_readiness_http_mode_retries_transient_failure(
    monkeypatch,
):
    calls = []

    async def resolve(_epoch):
        return 24032

    async def report(**_kwargs):
        return {
            "ready": True,
            "receipt_coverage": 1.0,
            "historical_classification_coverage": 1.0,
        }

    def fetch(gateway_url, epoch, *, timeout_seconds):
        calls.append((gateway_url, epoch, timeout_seconds))
        if len(calls) == 1:
            raise RuntimeError("HTTP Error 500: Internal Server Error")
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
        http_attempts=2,
        http_retry_seconds=0,
    )

    assert result["status"] == "ready"
    assert calls == [
        ("http://localhost:8000", 24032, 90),
        ("http://localhost:8000", 24032, 90),
    ]


@pytest.mark.asyncio
async def test_weight_readiness_http_mode_fails_closed_after_retries(
    monkeypatch,
):
    calls = []

    async def resolve(_epoch):
        return 24032

    async def report(**_kwargs):
        return {
            "ready": True,
            "receipt_coverage": 1.0,
            "historical_classification_coverage": 1.0,
        }

    def fetch(gateway_url, epoch, *, timeout_seconds):
        calls.append((gateway_url, epoch, timeout_seconds))
        raise RuntimeError("HTTP Error 500: Internal Server Error")

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

    with pytest.raises(
        readiness.WeightSubmissionReadinessV2Error,
        match="failed after 3 attempts",
    ):
        await readiness.verify_weight_submission_ready_v2(
            repair=False,
            gateway_url="http://localhost:8000",
            http_attempts=3,
            http_retry_seconds=0,
        )

    assert calls == [("http://localhost:8000", 24032, 90)] * 3


@pytest.mark.asyncio
async def test_weight_readiness_direct_mode_retries_only_transport_failures(
    monkeypatch,
):
    calls = []

    async def resolve(_epoch):
        return 24032

    async def report(**_kwargs):
        return {
            "ready": True,
            "receipt_coverage": 1.0,
            "historical_classification_coverage": 1.0,
        }

    async def handoff(epoch, x_leadpoet_internal_key):
        calls.append((epoch, x_leadpoet_internal_key))
        if len(calls) == 1:
            raise RuntimeError(
                "enclave rejected artifact persistence: unexpected_eof"
            )
        return {"handoff": True}

    monkeypatch.setattr(maintenance, "_resolve_maintenance_epoch", resolve)
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

    result = await readiness.verify_weight_submission_ready_v2(
        repair=False,
        http_attempts=2,
        http_retry_seconds=0,
    )

    assert result["status"] == "ready"
    assert calls == [(24032, None), (24032, None)]


@pytest.mark.asyncio
async def test_weight_readiness_direct_mode_retries_authenticated_s3_503(
    monkeypatch,
):
    calls = []

    async def resolve(_epoch):
        return 24032

    async def report(**_kwargs):
        return {
            "ready": True,
            "receipt_coverage": 1.0,
            "historical_classification_coverage": 1.0,
        }

    async def handoff(epoch, x_leadpoet_internal_key):
        calls.append((epoch, x_leadpoet_internal_key))
        if len(calls) == 1:
            persistence_error = TEEArtifactStoreV2Error(
                "enclave rejected artifact persistence: authenticated_http_503"
            )
            raise HTTPException(
                status_code=500,
                detail="Research Lab attested allocation failed",
            ) from persistence_error
        return {"handoff": True}

    monkeypatch.setattr(maintenance, "_resolve_maintenance_epoch", resolve)
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

    result = await readiness.verify_weight_submission_ready_v2(
        repair=False,
        http_attempts=2,
        http_retry_seconds=0,
    )

    assert result["status"] == "ready"
    assert calls == [(24032, None), (24032, None)]


@pytest.mark.asyncio
async def test_weight_readiness_direct_mode_exhausts_authenticated_s3_503(
    monkeypatch,
):
    calls = []

    async def resolve(_epoch):
        return 24032

    async def report(**_kwargs):
        return {
            "ready": True,
            "receipt_coverage": 1.0,
            "historical_classification_coverage": 1.0,
        }

    async def handoff(epoch, x_leadpoet_internal_key):
        calls.append((epoch, x_leadpoet_internal_key))
        persistence_error = TEEArtifactStoreV2Error(
            "enclave rejected artifact persistence: authenticated_http_503"
        )
        raise HTTPException(
            status_code=500,
            detail="Research Lab attested allocation failed",
        ) from persistence_error

    monkeypatch.setattr(maintenance, "_resolve_maintenance_epoch", resolve)
    monkeypatch.setattr(
        maintenance,
        "champion_v2_cutover_readiness_report",
        report,
    )
    monkeypatch.setattr(api, "get_research_lab_attested_allocation", handoff)

    with pytest.raises(HTTPException, match="attested allocation failed"):
        await readiness.verify_weight_submission_ready_v2(
            repair=False,
            http_attempts=3,
            http_retry_seconds=0,
        )

    assert calls == [(24032, None)] * 3


@pytest.mark.asyncio
@pytest.mark.parametrize("status", (400, 401, 403, 404))
async def test_weight_readiness_direct_mode_does_not_retry_authenticated_4xx(
    monkeypatch,
    status,
):
    calls = []

    async def resolve(_epoch):
        return 24032

    async def report(**_kwargs):
        return {
            "ready": True,
            "receipt_coverage": 1.0,
            "historical_classification_coverage": 1.0,
        }

    async def handoff(epoch, x_leadpoet_internal_key):
        calls.append((epoch, x_leadpoet_internal_key))
        raise TEEArtifactStoreV2Error(
            "enclave rejected artifact persistence: authenticated_http_%s"
            % status
        )

    monkeypatch.setattr(maintenance, "_resolve_maintenance_epoch", resolve)
    monkeypatch.setattr(
        maintenance,
        "champion_v2_cutover_readiness_report",
        report,
    )
    monkeypatch.setattr(api, "get_research_lab_attested_allocation", handoff)

    with pytest.raises(TEEArtifactStoreV2Error, match="authenticated_http"):
        await readiness.verify_weight_submission_ready_v2(
            repair=False,
            http_attempts=3,
            http_retry_seconds=0,
        )

    assert calls == [(24032, None)]


@pytest.mark.asyncio
async def test_weight_readiness_direct_mode_does_not_retry_semantic_failure(
    monkeypatch,
):
    calls = []

    async def resolve(_epoch):
        return 24032

    async def report(**_kwargs):
        return {
            "ready": True,
            "receipt_coverage": 1.0,
            "historical_classification_coverage": 1.0,
        }

    async def handoff(epoch, x_leadpoet_internal_key):
        calls.append((epoch, x_leadpoet_internal_key))
        raise RuntimeError("allocation receipt graph differs")

    monkeypatch.setattr(maintenance, "_resolve_maintenance_epoch", resolve)
    monkeypatch.setattr(
        maintenance,
        "champion_v2_cutover_readiness_report",
        report,
    )
    monkeypatch.setattr(api, "get_research_lab_attested_allocation", handoff)

    with pytest.raises(RuntimeError, match="receipt graph differs"):
        await readiness.verify_weight_submission_ready_v2(
            repair=False,
            http_attempts=3,
            http_retry_seconds=0,
        )

    assert calls == [(24032, None)]
