"""Prove the gateway can supply the authoritative validator weight input."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections.abc import Mapping
from typing import Any


class WeightSubmissionReadinessV2Error(RuntimeError):
    """The authoritative Research Lab allocation is not ready for submission."""


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


async def verify_weight_submission_ready_v2(
    *,
    repair: bool,
    gateway_url: str | None = None,
    epoch: int | None = None,
    netuid: int | None = None,
) -> dict[str, Any]:
    """Repair legacy authority if requested, then validate the exact V2 handoff."""

    from gateway.config import BITTENSOR_NETUID
    from gateway.research_lab.maintenance import (
        _resolve_maintenance_epoch,
        backfill_champion_reward_v2_authority,
        backfill_champion_settlement_v2_authority,
        champion_v2_cutover_readiness_report,
    )

    effective_epoch = await _resolve_maintenance_epoch(epoch)
    effective_netuid = int(netuid) if netuid is not None else int(BITTENSOR_NETUID)
    repairs: dict[str, Any] = {}
    if repair:
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
        repairs = {
            "champion_reward_receipts_created": int(
                reward_result.get("migrated_count") or 0
            ),
            "historical_allocations_classified": int(
                settlement_result.get("classified_count")
                or settlement_result.get("migrated_count")
                or 0
            ),
        }

    readiness = await champion_v2_cutover_readiness_report(
        epoch=effective_epoch,
        netuid=effective_netuid,
    )
    missing_classifications = (
        readiness.get("missing_historical_classifications")
        or readiness.get("missing_historical_settlements")
        or ()
    )
    if (
        readiness.get("ready") is not True
        or float(readiness.get("receipt_coverage") or 0.0) != 1.0
        or float(
            readiness.get("historical_classification_coverage")
            or readiness.get("historical_settlement_coverage")
            or 0.0
        )
        != 1.0
    ):
        raise WeightSubmissionReadinessV2Error(
            "champion V2 authority remains incomplete: "
            f"obligations={len(readiness.get('missing') or ())}, "
            f"historical_allocations={len(missing_classifications)}"
        )

    if gateway_url:
        from research_lab.validator_integration import (
            fetch_research_lab_attested_allocation_bundle,
        )

        handoff = await asyncio.to_thread(
            fetch_research_lab_attested_allocation_bundle,
            gateway_url,
            effective_epoch,
        )
    else:
        from gateway.research_lab.api import (
            get_research_lab_attested_allocation,
        )

        handoff = await get_research_lab_attested_allocation(
            effective_epoch,
            x_leadpoet_internal_key=None,
        )
    verified = _validate_handoff(
        handoff,
        epoch=effective_epoch,
        netuid=effective_netuid,
    )
    return {
        "schema_version": "leadpoet.weight_submission_readiness.v2",
        "status": "ready",
        "epoch": effective_epoch,
        "netuid": effective_netuid,
        **repairs,
        **verified,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repair", action="store_true")
    parser.add_argument("--gateway-url")
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--netuid", type=int)
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = asyncio.run(
        verify_weight_submission_ready_v2(
            repair=bool(args.repair),
            gateway_url=args.gateway_url,
            epoch=args.epoch,
            netuid=args.netuid,
        )
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
