"""Miner access to SN71 collection and historical Arena reviews."""

from __future__ import annotations

import argparse
import json
import os
import sys

from lab_arena import collection_cli
from lab_arena.miner_submit import (
    MinerSubmissionError,
    read_submission_review,
    retry_credit_failures,
    submission_credentials_from_environment,
    submit_agent_source,
)


PUBLIC_GATEWAY = "https://gateway.subnet71.com"


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--api-base-url",
        default=os.environ.get("LAB_ARENA_API_BASE_URL")
        or os.environ.get("GATEWAY_URL", PUBLIC_GATEWAY),
    )
    parser.add_argument("--wallet-name", default=None)
    parser.add_argument("--hotkey-name", default=None)
    parser.add_argument("--wallet-path", default=None)
    parser.add_argument(
        "--hotkey-uri",
        default=None,
        help="development only: derive the hotkey from a URI",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fund SN71 sales-workflow collection and inspect your spending.")
    commands = parser.add_subparsers(dest="command", required=True)
    collection_cli.add_commands(commands, _common_arguments)
    source = commands.add_parser(
        "submit-model",
        aliases=["submit-source"],
        help="legacy source upload for separate Arena deployments",
    )
    source.add_argument("--source", required=True, help="directory with harness.py")
    _common_arguments(source)
    interactive = commands.add_parser(
        "interactive", help="show current funding terms and how to enroll"
    )
    _common_arguments(interactive)
    interactive.set_defaults(mode=None)
    interactive.add_argument(
        "--task-scope",
        choices=(collection_cli.TASK_SCOPE, collection_cli.CORRECTION_SCOPE),
        default=collection_cli.TASK_SCOPE,
    )
    retry = commands.add_parser(
        "retry-credit-failures", help="legacy credit retry for separate Arena deployments",
    )
    retry.add_argument("--round-id", required=True)
    retry.add_argument("--submission-id", required=True)
    _common_arguments(retry)
    review = commands.add_parser("review", help="read private admission findings with the submitting hotkey")
    review.add_argument("--round-id", required=True)
    review.add_argument("--submission-id", required=True)
    _common_arguments(review)
    return parser


def _keypair(args):
    if args.hotkey_uri:
        from bittensor_wallet import Keypair

        return Keypair.create_from_uri(args.hotkey_uri)
    from bittensor_wallet import Wallet

    wallet_arguments = {
        "name": args.wallet_name or "default",
        "hotkey": args.hotkey_name or "default",
    }
    if args.wallet_path:
        wallet_arguments["path"] = args.wallet_path
    return Wallet(**wallet_arguments).hotkey


def submit_source(args) -> int:
    try:
        credentials = submission_credentials_from_environment()
        result = submit_agent_source(
            source_dir=args.source,
            api_base_url=args.api_base_url,
            keypair=_keypair(args),
            credentials=credentials,
        )
    except MinerSubmissionError as exc:
        forbidden = credentials.values() if "credentials" in locals() else ()
        print(
            "submission failed: %s" % exc.format_for_cli(forbidden_values=forbidden),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(dict(result), sort_keys=True))
    return 0


def interactive(args) -> int:
    print(
        "View comparison terms with leadpoet collection-terms, or use "
        "leadpoet collection-terms --task-scope synthetic_corrections for corrections.\n"
        "To enroll, see leadpoet collection-register --help and "
        "https://github.com/leadpoet/leadpoet/blob/main/docs/locked-collection.md#enrollment",
        file=sys.stderr,
    )
    args.command = "collection-terms"
    return collection_cli.run(args, _keypair)


def retry_credits(args) -> int:
    try:
        result = retry_credit_failures(
            round_id=args.round_id, submission_id=args.submission_id,
            api_base_url=args.api_base_url, keypair=_keypair(args),
        )
    except MinerSubmissionError as exc:
        print("credit retry failed: %s" % exc.format_for_cli(), file=sys.stderr)
        return 2
    print(json.dumps(dict(result), sort_keys=True))
    return 0 if result["status"] in {"queued", "replayed"} else 2


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    args = build_parser().parse_args(argv or ["interactive"])
    public_gateway = args.api_base_url.strip().lower().rstrip("/") in {PUBLIC_GATEWAY, PUBLIC_GATEWAY + ":443"}
    if public_gateway and args.command in {"submit-model", "submit-source", "retry-credit-failures"}:
        print(
            "Public champion submission commands are retired. Run leadpoet collection-terms to join collection.",
            file=sys.stderr,
        )
        return 2
    if args.command.startswith("collection-"):
        return collection_cli.run(args, _keypair)
    if args.command in {"submit-model", "submit-source"}:
        return submit_source(args)
    if args.command == "interactive":
        return interactive(args)
    if args.command == "retry-credit-failures":
        return retry_credits(args)
    if args.command == "review":
        try:
            result = read_submission_review(
                round_id=args.round_id, submission_id=args.submission_id,
                api_base_url=args.api_base_url, keypair=_keypair(args),
            )
        except MinerSubmissionError as exc:
            print("review unavailable: %s" % exc.format_for_cli(), file=sys.stderr)
            return 2
        print(json.dumps(dict(result), sort_keys=True))
        return 0
    return 2
