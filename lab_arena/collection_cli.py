"""Key enrollment CLI using the miner's local wallet and provider environment."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
import json
import sys
import time

import requests

from lab_arena import contracts
from lab_arena.collection_intake import SCOPE, TERMS_HASH, scope_id
from lab_arena.miner_submit import (
    MinerSubmissionError,
    _api_base_url,
    _signed_request,
    submission_credentials_from_environment,
)


def add_commands(commands, common_options):
    for action in ("terms", "register", "revoke", "statement"):
        parser = commands.add_parser("collection-" + action)
        common_options(parser)
        parser.add_argument(
            "--mode", choices=("pilot", "live"), required=action != "terms"
        )
        if action == "register":
            parser.add_argument(
                "--daily-cap-usd",
                required=True,
                help="Maximum rolling 24h spend on each provider key",
            )
            parser.add_argument(
                "--accept-terms",
                required=True,
                metavar="SHA256",
                help="Digest shown by collection-terms",
            )
        elif action == "revoke":
            parser.add_argument("--registration-id", required=True)


def daily_cap(value):
    try:
        amount = Decimal(value) * 1_000_000
        if (
            amount.is_finite()
            and amount == int(amount)
            and 1 <= amount <= 10_000_000_000
        ):
            return int(amount)
    except (InvalidOperation, ValueError, OverflowError):
        pass
    raise MinerSubmissionError("daily_cap_invalid")


def run(args, wallet, *, session=requests):
    action = args.command.removeprefix("collection-")
    try:
        base = _api_base_url(args.api_base_url) + "/arena/v1/collection/"
        if action == "terms":
            response = session.get(base + "terms", timeout=30, allow_redirects=False)
        else:
            creds, body = None, {}
            if action == "register":
                if args.accept_terms != TERMS_HASH:
                    raise MinerSubmissionError("collection_terms_changed")
                creds = submission_credentials_from_environment()
                creds.pop("scrapingdog_api_key", None)
                body = {
                    "daily_cap_microusd": daily_cap(args.daily_cap_usd),
                    "terms_sha256": TERMS_HASH,
                    "credentials_sha256": contracts.document_hash(creds),
                }
            if action == "revoke":
                body = {"registration_id": args.registration_id}
            signed = _signed_request(
                scope=SCOPE[action],
                round_id=scope_id(args.mode),
                body=body,
                keypair=wallet(args),
                now=time.time,
            )
            payload = (
                signed
                if creds is None
                else {"authorization": signed, "credentials": creds}
            )
            response = session.post(
                base + action, json=payload, timeout=65, allow_redirects=False
            )
        if not 200 <= response.status_code < 300:
            print(
                "Collection request failed (HTTP %d)." % response.status_code,
                file=sys.stderr,
            )
            return 2
        result = response.json()
        # Never log an arbitrary response from a remote credential intake endpoint.
        if action == "register" or action == "revoke":
            import uuid

            result = {
                "status": action,
                "registration_id": str(uuid.UUID(result["registration_id"])),
            }
        print(json.dumps(result, sort_keys=True))
        return 0
    except (
        requests.RequestException,
        ValueError,
        KeyError,
        TypeError,
        MinerSubmissionError,
    ) as exc:
        print(
            (
                exc.code
                if isinstance(exc, MinerSubmissionError)
                else "Collection request failed."
            ),
            file=sys.stderr,
        )
        return 2
