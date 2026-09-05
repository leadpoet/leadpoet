#!/usr/bin/env python3
"""Submit one local Arena model source directory and its run credentials.

    OPENROUTER_API_KEY=... OPENROUTER_MANAGEMENT_KEY=... DEEPLINE_API_KEY=... \
    python3 scripts/lab_arena_miner.py submit-model --source ./my-agent \
        --wallet-name W --hotkey-name H

The helper makes one bounded source archive, uploads it to the Arena's private
target, and finalizes it with the miner hotkey. Credentials are read only from
masked prompts or environment variables, and are never command arguments or
source archive content. ``--hotkey-uri`` is for development only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lab_arena.miner_submit import (  # noqa: E402
    MinerSubmissionError,
    run_interactive_submission,
    submission_credentials_from_environment,
    submit_agent_source,
)


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--api-base-url",
        default=os.environ.get("LAB_ARENA_API_BASE_URL")
        or os.environ.get("GATEWAY_URL", "https://gateway.subnet71.com"),
    )
    parser.add_argument("--wallet-name", default=None)
    parser.add_argument("--hotkey-name", default=None)
    parser.add_argument("--wallet-path", default=None)
    parser.add_argument(
        "--hotkey-uri", default=None, help="development only: derive the hotkey from a URI"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Leadpoet Lab Arena miner helper")
    commands = parser.add_subparsers(dest="command", required=True)
    source = commands.add_parser(
        "submit-model",
        aliases=["submit-source"],
        help="archive, upload, and submit a local model",
    )
    source.add_argument("--source", required=True, help="directory with harness.py")
    _common_arguments(source)
    interactive = commands.add_parser(
        "interactive", help="prompt for a local agent fork and submit it"
    )
    _common_arguments(interactive)
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
        print("submission failed: %s" % exc.code, file=sys.stderr)
        return 2
    print(json.dumps(dict(result), sort_keys=True))
    return 0


def interactive(args) -> int:
    return 0 if run_interactive_submission(_keypair(args), args.api_base_url) else 2


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command in {"submit-model", "submit-source"}:
        return submit_source(args)
    if args.command == "interactive":
        return interactive(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
