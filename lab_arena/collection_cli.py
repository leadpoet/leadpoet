"""Key enrollment CLI using the miner's local wallet and provider environment."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
import json
import os
import sys
import time

import requests

from lab_arena import contracts
from lab_arena.collection_intake import CORRECTION_SCOPE, MAX_DAILY_SLOTS, SCOPE, TASK_SCOPE, TERMS, TERMS_HASH, scope_id
from lab_arena.credentials import OPENROUTER_FUNDING_FIELDS
from lab_arena.miner_submit import (
    MinerSubmissionError,
    _api_base_url,
    _signed_request,
    read_masked_credential,
    REQUIRED_SUBMISSION_CREDENTIAL_ENV_VARS,
    SUBMISSION_CREDENTIAL_MAX_LENGTH,
    SUBMISSION_CREDENTIAL_MIN_LENGTH,
)


# Only display local guidance. Intake failures may contain echoed credentials.
_ERROR_GUIDANCE = {
    "hotkey_unregistered": "Use a hotkey registered on Finney subnet 71.",
    "collection_ineligible_hotkey": "This hotkey is ineligible for collection.",
    "revoke_current_registration_first": "Use collection-statement to find your active registration, then collection-revoke to change it.",
    "openrouter_api_key_invalid": "Check the dedicated OpenRouter runtime key.",
    "openrouter_api_key_no_credit": "Check your runtime key's available credit and spending limit.",
    "openrouter_management_key_invalid": "Use a read-only management key from the same OpenRouter workspace as your runtime key.",
    "registration_unknown": "Check the registration ID, wallet, hotkey and mode with collection-statement.",
    "submission_rate_limited": "Wait before retrying enrollment.",
    "credential_validation_unavailable": "Provider key verification is temporarily unavailable. Try again later.",
}


def add_commands(commands, common_options):
    for action, help_text in (
        ("terms", "read the published collection terms"),
        ("register", "enroll a funding key with a signed spending cap"),
        ("revoke", "stop new charges on an enrollment"),
        ("statement", "check your enrollments, jobs and charges"),
    ):
        parser = commands.add_parser("collection-" + action, help=help_text)
        common_options(parser)
        parser.add_argument(
            "--mode", choices=("pilot", "live"), required=action != "terms"
        )
        if action in ("terms", "register"):
            parser.add_argument("--task-scope", choices=(TASK_SCOPE, CORRECTION_SCOPE), default=TASK_SCOPE)
        if action == "register":
            parser.add_argument(
                "--max-slots-per-day", type=int, choices=range(1, MAX_DAILY_SLOTS + 1),
                help="maximum distinct funded assignments per UTC day (default 1)",
            )
            parser.add_argument(
                "--daily-cap-usd",
                required=True,
                help="Maximum rolling 24h spend on your OpenRouter key",
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
        task_scope = getattr(args, "task_scope", TASK_SCOPE)
        expected = TERMS_HASH
        text = TERMS
        if action in ("terms", "register"):
            url = base + "terms" + ("?task_scope=" + CORRECTION_SCOPE if task_scope == CORRECTION_SCOPE else "")
            response = session.get(url, timeout=30, allow_redirects=False)
            if action == "register":
                if response.status_code != 200:
                    raise MinerSubmissionError("collection_correction_terms_unavailable" if task_scope == CORRECTION_SCOPE else "collection_terms_unavailable")
                published = response.json()
                if getattr(args, "max_slots_per_day", None) is not None and published.get(
                    "max_selectable_slots_per_utc_day"
                ) != MAX_DAILY_SLOTS:
                    raise MinerSubmissionError("collection_slot_opt_in_unavailable")
                if (published.get("mode") != args.mode or published.get("task_scope") != task_scope
                        or published.get("terms_sha256") != expected or published.get("terms") != text):
                    raise MinerSubmissionError("collection_terms_changed")
        if action != "terms":
            creds, body = None, {}
            if action == "register":
                if args.accept_terms != expected:
                    raise MinerSubmissionError("collection_terms_changed")
                cap = daily_cap(args.daily_cap_usd)
                creds = {}
                for field in OPENROUTER_FUNDING_FIELDS:
                    value = os.environ.get(REQUIRED_SUBMISSION_CREDENTIAL_ENV_VARS[field], "")
                    if not value and sys.stdin.isatty():
                        prompt = "OpenRouter runtime key: " if field == "openrouter_api_key" else "Read-only OpenRouter management key: "
                        value = read_masked_credential(prompt)
                    creds[field] = value
                if any(
                    not SUBMISSION_CREDENTIAL_MIN_LENGTH
                    <= len(key)
                    <= SUBMISSION_CREDENTIAL_MAX_LENGTH
                    or key != key.strip()
                    for key in creds.values()
                ):
                    raise MinerSubmissionError(
                        "openrouter_funding_credentials_required"
                    )
                body = {
                    "daily_cap_microusd": cap,
                    "terms_sha256": expected,
                    "task_scope": task_scope,
                    "credentials_sha256": contracts.document_hash(creds),
                }
                # Omitting the option preserves compatibility with older gateways and signatures.
                slots = getattr(args, "max_slots_per_day", None)
                if slots is not None:
                    body["max_slots_per_utc_day"] = slots
            if action == "revoke":
                body = {"registration_id": args.registration_id}
            from bittensor_wallet.errors import KeyFileError, PasswordError, WalletError

            try:
                keypair = wallet(args)
            except (KeyFileError, PasswordError, WalletError, OSError):
                print("Cannot open the signing hotkey. Check --wallet-name, --hotkey-name and --wallet-path.", file=sys.stderr)
                return 2
            signed = _signed_request(
                scope=SCOPE[action],
                round_id=scope_id(args.mode),
                body=body,
                keypair=keypair,
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
            try:
                denial = response.json()
            except ValueError:
                denial = None
            code = denial.get("code") if isinstance(denial, dict) else None
            guidance = _ERROR_GUIDANCE.get(code, "") if isinstance(code, str) else ""
            print(f"Collection request failed (HTTP {response.status_code}). {guidance}".rstrip(), file=sys.stderr)
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
