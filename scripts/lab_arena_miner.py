#!/usr/bin/env python3
"""Lab Arena miner helper for bundle submission and request signing.

    python3 scripts/lab_arena_miner.py submission-body --image ghcr.io/you/agent:latest --out body.json
    python3 scripts/lab_arena_miner.py sign --scope submission --round-id arena-2026-09-02 \\
        --body body.json --out envelope.json [--wallet-name W --hotkey-name H | --hotkey-uri //Alice]

A submission names one container image by tag or digest in any public
registry. The Arena resolves it once and uses the host's provider keys.
Hotkey signing uses the local Bittensor wallet;
``--hotkey-uri`` exists for development only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lab_arena import contracts, images  # noqa: E402

SCOPES = {
    "submission": contracts.SCOPE_SUBMISSION,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Leadpoet Lab Arena miner helper")
    commands = parser.add_subparsers(dest="command", required=True)
    body = commands.add_parser("submission-body", help="write the signed-request body that names one public image")
    body.add_argument("--image", required=True, help="registry/repository:tag or registry/repository@sha256:<digest>")
    body.add_argument("--out", required=True)
    sign = commands.add_parser("sign", help="sign a canonical Arena request document")
    sign.add_argument("--scope", required=True, choices=sorted(SCOPES))
    sign.add_argument("--round-id", required=True)
    sign.add_argument("--body", required=True, help="path to the JSON request body")
    sign.add_argument("--out", required=True)
    sign.add_argument("--wallet-name", default=None)
    sign.add_argument("--hotkey-name", default=None)
    sign.add_argument("--hotkey-uri", default=None, help="development only: derive the hotkey from a URI")
    return parser


def submission_body_document(image: str) -> dict:
    """The body of a submission request: one image and public reuse consent."""

    reference = images.parse_reference(image)
    document = {"image_reference": str(reference), "consent": {"public_rerun": True}}
    contracts.validate_submission_body(document)
    return document


def submission_body(args) -> int:
    try:
        document = submission_body_document(args.image)
    except images.ImageError as exc:
        print("image reference refused: %s" % exc.rule_id, file=sys.stderr)
        return 2
    Path(args.out).write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"image_reference": document["image_reference"], "out": args.out}))
    return 0


def _keypair(args):
    if args.hotkey_uri:
        from bittensor_wallet import Keypair

        return Keypair.create_from_uri(args.hotkey_uri)
    from bittensor_wallet import Wallet

    return Wallet(name=args.wallet_name or "default", hotkey=args.hotkey_name or "default").hotkey


def sign(args) -> int:
    body = json.loads(Path(args.body).read_text(encoding="utf-8"))
    keypair = _keypair(args)
    envelope = contracts.build_signed_request(
        scope=SCOPES[args.scope], round_id=args.round_id, hotkey=keypair.ss58_address, body=body,
        timestamp=int(time.time()), sign_message=lambda message: keypair.sign(message.encode("utf-8")).hex(),
    )
    Path(args.out).write_text(json.dumps(envelope, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"hotkey": envelope["hotkey"], "scope": envelope["scope"], "request_id": envelope["request_id"], "out": args.out}))
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "submission-body":
        return submission_body(args)
    if args.command == "sign":
        return sign(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
