"""Prove the gateway can supply the authoritative validator weight input."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from collections.abc import Mapping
from typing import Any


class WeightSubmissionReadinessV2Error(RuntimeError):
    """The authoritative Research Lab allocation is not ready for submission."""


logger = logging.getLogger(__name__)
_RETRYABLE_ALLOCATION_FAILURE_MARKERS = (
    "connection_refused",
    "connection_reset",
    "gateway timeout",
    "read operation timed out",
    "timeout",
    "tls_failure",
    "unexpected_eof",
)
_RETRYABLE_AUTHENTICATED_HTTP_FAILURE_MARKERS = tuple(
    "authenticated_http_%s" % status
    for status in (408, 429, 500, 502, 503, 504)
)
_REPAIRABLE_AUTHORITY_FAILURE_MARKERS = (
    "champion v2 cutover blocked:",
    "historical compute fallback lacks finalized allocation authority",
)


def _exception_chain_text(exc: BaseException) -> str:
    current: BaseException | None = exc
    observed: set[int] = set()
    messages: list[str] = []
    while current is not None and id(current) not in observed:
        observed.add(id(current))
        messages.append(str(current).lower())
        current = current.__cause__ or current.__context__
    return " ".join(messages)


def _retryable_allocation_failure(exc: BaseException) -> bool:
    text = _exception_chain_text(exc)
    return any(
        marker in text
        for marker in (
            *_RETRYABLE_ALLOCATION_FAILURE_MARKERS,
            *_RETRYABLE_AUTHENTICATED_HTTP_FAILURE_MARKERS,
        )
    )


def _repairable_authority_failure(exc: BaseException) -> bool:
    text = _exception_chain_text(exc)
    return any(marker in text for marker in _REPAIRABLE_AUTHORITY_FAILURE_MARKERS)


def _ancestry_safe_epoch_from_storage_readiness(
    *,
    effective_epoch: int,
    bootstrap: Mapping[str, Any] | None,
) -> int:
    """Return the first epoch whose complete predecessor history is proven."""

    normalized_epoch = int(effective_epoch)
    if bootstrap is None:
        return normalized_epoch
    try:
        activation_epoch = int(bootstrap["activation_epoch"])
        target_epoch = int(bootstrap["target_epoch"])
        backlog_epoch_count = int(bootstrap["backlog_epoch_count"])
        raw_settled_through = bootstrap.get("settled_through_epoch")
        settled_through = (
            activation_epoch - 1
            if raw_settled_through is None
            else int(raw_settled_through)
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise WeightSubmissionReadinessV2Error(
            "chain-realized settlement bootstrap report is invalid"
        ) from exc
    safe_epoch = settled_through + 1
    if (
        normalized_epoch < 0
        or activation_epoch < 0
        or target_epoch != normalized_epoch - 1
        or settled_through < activation_epoch - 1
        or safe_epoch > normalized_epoch
        or backlog_epoch_count != target_epoch - settled_through
    ):
        raise WeightSubmissionReadinessV2Error(
            "chain-realized settlement bootstrap report is inconsistent"
        )
    return safe_epoch


def _validate_handoff(
    handoff: Mapping[str, Any],
    *,
    epoch: int,
    netuid: int,
) -> dict[str, Any]:
    from leadpoet_canonical.allocation_handoff_v2 import (
        validate_allocation_handoff_v2,
    )
    from research_lab.validator_integration import (
        ResearchLabValidatorFlags,
        build_research_lab_allocation_component,
        verify_research_lab_allocation_bundle,
    )

    normalized = validate_allocation_handoff_v2(
        handoff,
        expected_epoch_id=int(epoch),
        expected_netuid=int(netuid),
    )
    bundle = normalized["bundle"]
    flags = ResearchLabValidatorFlags.from_mapping(os.environ)
    verification = verify_research_lab_allocation_bundle(bundle, flags=flags)
    if verification.get("passed") is not True:
        raise WeightSubmissionReadinessV2Error(
            "Research Lab allocation verification failed: %s"
            % list(verification.get("errors") or ())
        )
    component = build_research_lab_allocation_component(bundle, flags=flags)
    allocation_hash = str(component.get("allocation_hash") or "")
    if not allocation_hash:
        raise WeightSubmissionReadinessV2Error(
            "Research Lab allocation hash is missing"
        )
    return {
        "allocation_hash": allocation_hash,
        "root_receipt_hash": normalized["root_receipt_hash"],
    }


async def verify_weight_submission_storage_readable_v2(
    *,
    epoch: int | None = None,
    netuid: int | None = None,
) -> dict[str, Any]:
    """Read the complete durable authority path without repairing or launching."""

    from gateway.research_lab.maintenance import (
        _resolve_maintenance_epoch,
        champion_v2_cutover_readiness_report,
    )
    from gateway.research_lab.champion_settlement_v2 import (
        ChampionSettlementV2Error,
        validate_chain_realized_settlement_bootstrap_v1,
    )

    effective_epoch = await _resolve_maintenance_epoch(epoch)
    if netuid is None:
        from gateway.config import BITTENSOR_NETUID

        effective_netuid = int(BITTENSOR_NETUID)
    else:
        effective_netuid = int(netuid)
    bootstrap = await validate_chain_realized_settlement_bootstrap_v1(
        netuid=effective_netuid,
        target_epoch=effective_epoch - 1,
    )
    backlog_epoch_count = int(bootstrap.get("backlog_epoch_count") or 0)
    if backlog_epoch_count:
        readiness = {
            "ready": False,
            "receipt_coverage": 0.0,
            "historical_classification_coverage": 0.0,
        }
    else:
        try:
            readiness = await champion_v2_cutover_readiness_report(
                epoch=effective_epoch,
                netuid=effective_netuid,
            )
        except ChampionSettlementV2Error as exc:
            if str(exc) != "chain realized settlement history is incomplete":
                raise
            raise WeightSubmissionReadinessV2Error(
                "chain-realized settlement frontier disagrees with authority readiness"
            ) from exc
    result = {
        "schema_version": "leadpoet.weight_submission_storage_readiness.v2",
        "status": "readable",
        "epoch": effective_epoch,
        "ancestry_safe_epoch": _ancestry_safe_epoch_from_storage_readiness(
            effective_epoch=effective_epoch,
            bootstrap=bootstrap,
        ),
        "netuid": effective_netuid,
        "authority_ready": (
            backlog_epoch_count == 0 and readiness.get("ready") is True
        ),
        "receipt_coverage": float(readiness.get("receipt_coverage") or 0.0),
        "historical_classification_coverage": float(
            readiness.get("historical_classification_coverage")
            or readiness.get("historical_settlement_coverage")
            or 0.0
        ),
    }
    if backlog_epoch_count:
        result["chain_realized_settlement_bootstrap"] = bootstrap
    return result


async def repair_chain_realized_settlements_v1(
    *,
    epoch: int | None = None,
    netuid: int | None = None,
) -> dict[str, Any]:
    """Fill and prove the live-metagraph settlement suffix without allocation."""

    from gateway.config import BITTENSOR_NETUID
    from gateway.research_lab.champion_settlement_v2 import (
        validate_chain_realized_settlement_bootstrap_v1,
    )
    from gateway.research_lab.maintenance import _resolve_maintenance_epoch
    from gateway.research_lab.v2_authority import (
        ensure_chain_realized_settlements_v1,
    )

    if not os.getenv("RESEARCH_LAB_INTERNAL_API_KEY", "").strip():
        raise WeightSubmissionReadinessV2Error(
            "Research Lab internal API key is not configured"
        )
    effective_epoch = int(await _resolve_maintenance_epoch(epoch))
    effective_netuid = int(netuid) if netuid is not None else int(BITTENSOR_NETUID)
    current_epoch = int(await _resolve_maintenance_epoch(None))
    if effective_epoch < 0 or effective_epoch > current_epoch:
        raise WeightSubmissionReadinessV2Error(
            "chain-realized settlement repair epoch is invalid"
        )
    repaired = await ensure_chain_realized_settlements_v1(
        epoch_id=effective_epoch,
        netuid=effective_netuid,
    )
    bootstrap = await validate_chain_realized_settlement_bootstrap_v1(
        netuid=effective_netuid,
        target_epoch=effective_epoch - 1,
    )
    if int(bootstrap.get("backlog_epoch_count") or 0) != 0:
        raise WeightSubmissionReadinessV2Error(
            "chain-realized settlement repair left an incomplete suffix"
        )
    activation_epoch = int(bootstrap["activation_epoch"])
    raw_settled_through = bootstrap.get("settled_through_epoch")
    settled_through = (
        activation_epoch - 1
        if raw_settled_through is None
        else int(raw_settled_through)
    )
    if settled_through != effective_epoch - 1:
        raise WeightSubmissionReadinessV2Error(
            "chain-realized settlement repair readback is incomplete"
        )
    observed_epoch = int(await _resolve_maintenance_epoch(None))
    return {
        "schema_version": "leadpoet.chain_realized_settlement_repair.v1",
        "status": "ready",
        "epoch": effective_epoch,
        "observed_epoch": observed_epoch,
        "netuid": effective_netuid,
        "settled_through_epoch": settled_through,
        "repaired_epoch_count": len(repaired),
    }


async def verify_weight_submission_ready_v2(
    *,
    repair: bool,
    gateway_url: str | None = None,
    epoch: int | None = None,
    netuid: int | None = None,
    http_attempts: int = 3,
    http_retry_seconds: float = 2.0,
    http_timeout_seconds: int = 90,
) -> dict[str, Any]:
    """Repair legacy authority if requested, then validate the exact V2 handoff."""

    from gateway.config import BITTENSOR_NETUID
    from gateway.research_lab.maintenance import (
        _resolve_maintenance_epoch,
        backfill_champion_reward_v2_authority,
        backfill_champion_settlement_v2_authority,
        backfill_historical_compute_fallback_v2_authority,
        backfill_source_add_reward_v2_authority,
    )

    effective_epoch = await _resolve_maintenance_epoch(epoch)
    effective_netuid = int(netuid) if netuid is not None else int(BITTENSOR_NETUID)
    direct_repair_internal_key: str | None = None
    direct_repair_current_epoch: int | None = None
    if repair and not gateway_url:
        direct_repair_internal_key = os.getenv(
            "RESEARCH_LAB_INTERNAL_API_KEY", ""
        ).strip()
        if not direct_repair_internal_key:
            raise WeightSubmissionReadinessV2Error(
                "Research Lab internal API key is not configured"
            )
        direct_repair_current_epoch = (
            int(effective_epoch)
            if epoch is None
            else int(await _resolve_maintenance_epoch(None))
        )

    repairs: dict[str, Any] = (
        {
            "source_add_reward_receipts_created": 0,
            "champion_reward_receipts_created": 0,
            "historical_allocations_classified": 0,
            "historical_compute_fallbacks_classified": 0,
        }
        if repair
        else {}
    )

    async def run_authority_repairs() -> dict[str, Any]:
        source_reward_result = await backfill_source_add_reward_v2_authority(
            epoch=effective_epoch,
            limit=10000,
            dry_run=False,
        )
        if source_reward_result.get("ok") is not True:
            raise WeightSubmissionReadinessV2Error(
                "SOURCE_ADD reward authority backfill failed"
            )
        reward_result = await backfill_champion_reward_v2_authority(
            epoch=effective_epoch,
            limit=10000,
            dry_run=False,
        )
        if reward_result.get("ok") is not True:
            raise WeightSubmissionReadinessV2Error(
                "champion reward authority backfill failed"
            )
        settlement_result = await backfill_champion_settlement_v2_authority(
            epoch=effective_epoch,
            netuid=effective_netuid,
            limit=10000,
            dry_run=False,
        )
        if settlement_result.get("ok") is not True:
            raise WeightSubmissionReadinessV2Error(
                "champion settlement classification backfill failed"
            )
        fallback_result = (
            await backfill_historical_compute_fallback_v2_authority(
                epoch=effective_epoch,
                netuid=effective_netuid,
                dry_run=False,
            )
        )
        if fallback_result.get("ok") is not True:
            raise WeightSubmissionReadinessV2Error(
                "historical compute fallback classification failed"
            )
        return {
            "source_add_reward_receipts_created": int(
                source_reward_result.get("migrated_count") or 0
            ),
            "champion_reward_receipts_created": int(
                reward_result.get("migrated_count") or 0
            ),
            "historical_allocations_classified": int(
                settlement_result.get("classified_count")
                or settlement_result.get("migrated_count")
                or 0
            ),
            "historical_compute_fallbacks_classified": int(
                fallback_result.get("classified_count") or 0
            ),
        }

    # HTTP repair remains an explicit maintenance operation against a running
    # gateway. The pre-launch direct path below first proves the exact handoff
    # and only scans/writes historical authority when the cutover gate reports
    # missing classifications.
    if repair and gateway_url:
        repairs = await run_authority_repairs()

    # The allocation builder below owns the same 100%-coverage cutover gate
    # and returns no handoff unless it passes. Running the standalone report
    # here repeated the complete growing history scan immediately before that
    # authoritative build.
    if gateway_url:
        from research_lab.validator_integration import (
            fetch_research_lab_attested_allocation_bundle,
        )

        if int(http_attempts) < 1:
            raise ValueError("http_attempts must be positive")
        if float(http_retry_seconds) < 0:
            raise ValueError("http_retry_seconds must be non-negative")
        if int(http_timeout_seconds) < 1:
            raise ValueError("http_timeout_seconds must be positive")
        for attempt in range(1, int(http_attempts) + 1):
            try:
                handoff = await asyncio.to_thread(
                    fetch_research_lab_attested_allocation_bundle,
                    gateway_url,
                    effective_epoch,
                    timeout_seconds=int(http_timeout_seconds),
                )
                break
            except Exception as exc:
                if attempt >= int(http_attempts):
                    raise WeightSubmissionReadinessV2Error(
                        "gateway allocation HTTP handoff failed after "
                        f"{attempt} attempts: {exc}"
                    ) from exc
                logger.warning(
                    "weight_readiness_http_transient_retry "
                    "epoch=%s attempt=%s/%s timeout_seconds=%s "
                    "error_type=%s",
                    effective_epoch,
                    attempt,
                    int(http_attempts),
                    int(http_timeout_seconds),
                    type(exc).__name__,
                )
                await asyncio.sleep(float(http_retry_seconds))
    else:
        from gateway.research_lab.api import (
            _get_research_lab_attested_allocation_for_resolved_current_epoch,
            get_research_lab_attested_allocation,
        )

        if int(http_attempts) < 1:
            raise ValueError("http_attempts must be positive")
        if float(http_retry_seconds) < 0:
            raise ValueError("http_retry_seconds must be non-negative")

        async def load_direct_handoff() -> Mapping[str, Any]:
            for attempt in range(1, int(http_attempts) + 1):
                try:
                    if direct_repair_internal_key is not None:
                        return await (
                            _get_research_lab_attested_allocation_for_resolved_current_epoch(
                                epoch=effective_epoch,
                                current_epoch=int(direct_repair_current_epoch),
                                internal_key=direct_repair_internal_key,
                            )
                        )
                    return await get_research_lab_attested_allocation(
                        effective_epoch,
                        x_leadpoet_internal_key=None,
                    )
                except Exception as exc:
                    if (
                        not _retryable_allocation_failure(exc)
                        or attempt >= int(http_attempts)
                    ):
                        raise
                    logger.warning(
                        "weight_readiness_allocation_transient_retry "
                        "epoch=%s attempt=%s/%s error_type=%s",
                        effective_epoch,
                        attempt,
                        int(http_attempts),
                        type(exc).__name__,
                    )
                    await asyncio.sleep(float(http_retry_seconds) * attempt)
            raise AssertionError("unreachable direct allocation retry state")

        try:
            handoff = await load_direct_handoff()
        except Exception as exc:
            if not repair or not _repairable_authority_failure(exc):
                raise
            logger.warning(
                "weight_readiness_authority_repair_required epoch=%s netuid=%s",
                effective_epoch,
                effective_netuid,
            )
            repairs = await run_authority_repairs()
            handoff = await load_direct_handoff()
    verified = _validate_handoff(
        handoff,
        epoch=effective_epoch,
        netuid=effective_netuid,
    )
    result = {
        "schema_version": "leadpoet.weight_submission_readiness.v2",
        "status": "ready",
        "epoch": effective_epoch,
        "netuid": effective_netuid,
        **repairs,
        **verified,
    }
    if repair and not gateway_url:
        result["observed_epoch"] = int(
            await _resolve_maintenance_epoch(None)
        )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--repair", action="store_true")
    mode.add_argument("--repair-chain-settlements", action="store_true")
    mode.add_argument("--storage-read-preflight", action="store_true")
    parser.add_argument("--gateway-url")
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--netuid", type=int)
    parser.add_argument("--http-timeout-seconds", type=int, default=90)
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    if args.storage_read_preflight:
        if args.gateway_url:
            parser.error("--storage-read-preflight cannot use --gateway-url")
        result = asyncio.run(
            verify_weight_submission_storage_readable_v2(
                epoch=args.epoch,
                netuid=args.netuid,
            )
        )
    elif args.repair_chain_settlements:
        if args.gateway_url:
            parser.error("--repair-chain-settlements cannot use --gateway-url")
        result = asyncio.run(
            repair_chain_realized_settlements_v1(
                epoch=args.epoch,
                netuid=args.netuid,
            )
        )
    else:
        result = asyncio.run(
            verify_weight_submission_ready_v2(
                repair=bool(args.repair),
                gateway_url=args.gateway_url,
                epoch=args.epoch,
                netuid=args.netuid,
                http_timeout_seconds=args.http_timeout_seconds,
            )
        )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
