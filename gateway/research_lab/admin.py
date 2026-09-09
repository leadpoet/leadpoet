"""Operator CLI for retained Research Lab controls."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from typing import Any

from gateway.deploy_readiness import (
    build_deploy_readiness,
    write_deploy_readiness_manifest,
)
from .maintenance import (
    backfill_champion_reward_v2_authority,
    backfill_champion_settlement_v2_authority,
    champion_v2_cutover_readiness_report,
    default_actor_ref,
    dumps_status,
    reconcile_champion_reward_statuses,
)
from .store import call_rpc, select_all, select_one

logger = logging.getLogger(__name__)


def _add_deploy_readiness_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--gateway-commit", help="current gateway source commit")
    parser.add_argument("--validator-commit", help="current validator source commit")
    parser.add_argument("--gateway-pcr0", help="current gateway enclave PCR0")
    parser.add_argument("--validator-pcr0", help="current validator enclave PCR0")
    parser.add_argument("--expected-gateway-commit", help="expected gateway source commit")
    parser.add_argument("--expected-validator-commit", help="expected validator source commit")
    parser.add_argument("--expected-gateway-pcr0", help="expected gateway enclave PCR0")
    parser.add_argument("--expected-validator-pcr0", help="expected validator enclave PCR0")
    parser.add_argument(
        "--require-same-commit",
        action="store_true",
        help="fail unless gateway and validator commits match",
    )
    parser.add_argument(
        "--require-pcr0",
        action="store_true",
        help="fail unless both supplied gateway and validator PCR0s are present",
    )
    parser.add_argument(
        "--require-pcr0-commit-match",
        action="store_true",
        help="fail unless matched static PCR0 allowlist metadata points at the running commit",
    )
    parser.add_argument(
        "--include-docker-health",
        action="store_true",
        help="include Docker daemon and disk headroom health as a warning check",
    )
    parser.add_argument(
        "--require-docker-build-health",
        action="store_true",
        help="run a tiny Docker smoke build and fail readiness if Docker/build storage is unhealthy",
    )

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Leadpoet Research Lab admin controls")
    sub = parser.add_subparsers(dest="command", required=True)

    readiness = sub.add_parser(
        "check-deploy-readiness", help="Check gateway and validator release alignment"
    )
    _add_deploy_readiness_args(readiness)
    readiness.add_argument("--write-manifest", nargs="?", const="")
    readiness.add_argument("--no-enforce-resume-block", action="store_true")

    champion = sub.add_parser("reconcile-champion-reward-statuses")
    champion.add_argument("--epoch", type=int)
    champion.add_argument("--netuid", type=int)
    champion.add_argument("--limit", type=int, default=50)
    champion.add_argument("--reason", default="champion_reward_status_reconciler")
    champion.add_argument("--actor-ref", default=default_actor_ref())
    champion.add_argument("--dry-run", dest="dry_run", action="store_true", default=True)
    champion.add_argument("--write", dest="dry_run", action="store_false")

    champion_auth = sub.add_parser("backfill-champion-v2-authority")
    champion_auth.add_argument("--epoch", type=int)
    champion_auth.add_argument("--limit", type=int, default=1000)
    champion_auth.add_argument(
        "--dry-run", dest="dry_run", action="store_true", default=True
    )
    champion_auth.add_argument("--write", dest="dry_run", action="store_false")

    settlement = sub.add_parser("backfill-champion-v2-settlements")
    settlement.add_argument("--epoch", type=int)
    settlement.add_argument("--netuid", type=int)
    settlement.add_argument("--limit", type=int, default=1000)
    settlement.add_argument(
        "--dry-run", dest="dry_run", action="store_true", default=True
    )
    settlement.add_argument("--write", dest="dry_run", action="store_false")

    cutover = sub.add_parser(
        "champion-v2-cutover-readiness",
        help="Require complete V2 receipt coverage for historical positive champion balances",
    )
    cutover.add_argument("--epoch", type=int)
    cutover.add_argument("--netuid", type=int)

    recover = sub.add_parser("recover-arweave-audit-epochs")
    recover.add_argument("--epoch", action="append", dest="epochs", type=int, required=True)
    recover.add_argument("--netuid", type=int)
    recover.add_argument("--dry-run", dest="dry_run", action="store_true", default=True)
    recover.add_argument("--write", dest="dry_run", action="store_false")

    checkpoint = sub.add_parser("checkpoint-arweave-now")
    checkpoint.add_argument("--write", action="store_true")

    return parser


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "check-deploy-readiness":
        result = build_deploy_readiness(
            gateway_commit=args.gateway_commit,
            validator_commit=args.validator_commit,
            gateway_pcr0=args.gateway_pcr0,
            validator_pcr0=args.validator_pcr0,
            expected_gateway_commit=args.expected_gateway_commit,
            expected_validator_commit=args.expected_validator_commit,
            expected_gateway_pcr0=args.expected_gateway_pcr0,
            expected_validator_pcr0=args.expected_validator_pcr0,
            require_same_commit=args.require_same_commit,
            require_pcr0=args.require_pcr0,
            require_pcr0_commit_match=args.require_pcr0_commit_match,
            include_docker_health=args.include_docker_health,
            require_docker_build_health=args.require_docker_build_health,
        )
        result["action"] = "check-deploy-readiness"
        if args.write_manifest is not None:
            result["manifest_path"] = str(
                write_deploy_readiness_manifest(
                    result,
                    args.write_manifest or None,
                    enforce_resume_block=not args.no_enforce_resume_block,
                )
            )
        return result
    if args.command == "reconcile-champion-reward-statuses":
        return await reconcile_champion_reward_statuses(
            epoch=args.epoch,
            netuid=args.netuid,
            limit=args.limit,
            reason=args.reason,
            actor_ref=args.actor_ref,
            dry_run=args.dry_run,
        )
    if args.command == "backfill-champion-v2-authority":
        return await backfill_champion_reward_v2_authority(
            epoch=args.epoch, limit=args.limit, dry_run=args.dry_run
        )
    if args.command == "backfill-champion-v2-settlements":
        return await backfill_champion_settlement_v2_authority(
            epoch=args.epoch,
            netuid=args.netuid,
            limit=args.limit,
            dry_run=args.dry_run,
        )
    if args.command == "champion-v2-cutover-readiness":
        return await champion_v2_cutover_readiness_report(
            epoch=args.epoch, netuid=args.netuid
        )
    if args.command == "recover-arweave-audit-epochs":
        from gateway.config import BITTENSOR_NETUID
        from .arweave_audit import recover_research_lab_checkpointed_audit_epochs
        return await recover_research_lab_checkpointed_audit_epochs(
            epochs=args.epochs,
            netuid=int(args.netuid) if args.netuid is not None else int(BITTENSOR_NETUID),
            dry_run=args.dry_run,
        )
    if args.command == "checkpoint-arweave-now":
        if not args.write:
            return {
                "ok": True,
                "dry_run": True,
                "action": "checkpoint-arweave-now",
                "guidance": "pass --write to run one immediate checkpoint batch",
            }
        from gateway.tasks.hourly_batch import hourly_batch_task
        result = await hourly_batch_task(run_immediately=True, max_batches=1)
        if not isinstance(result, dict):
            raise RuntimeError("immediate Arweave checkpoint returned no result")
        return {**result, "dry_run": False, "action": "checkpoint-arweave-now"}
    raise ValueError(f"unknown command: {args.command}")


def main() -> int:
    args = build_parser().parse_args()
    result = asyncio.run(_run(args))
    print(dumps_status(result))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
