"""Key enrollment CLI using the miner's local wallet and provider environment."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
import json
import os
import sys
import time

import requests

from lab_arena import contracts
from lab_arena.collection_intake import (
    CORRECTION_SCOPE, MAX_DAILY_SLOTS, MAX_MICROUSD, MODELS, PAID_OFFER_ID, PAID_WORKFLOW_TERMS,
    PAID_WORKFLOW_TERMS_HASH, SCOPE, TASK_SCOPE, TERMS, TERMS_HASH, scope_id,
)
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
    "registration_conflict": "If this key is already enrolled in the pilot, check collection-statement --mode pilot and revoke it before enrolling for paid work.",
    "collection_prices_unavailable": "Paid enrollment needs a published fixed price. Check collection-terms.",
    "collection_comparison_prices_unavailable": "The gateway has no usable fixed-price offer. Check collection-terms.",
    "collection_offer_changed": "The paid offer changed. Read collection-terms before signing.",
    "collection_terms_changed": "Your mode or terms do not match the gateway. Check collection-terms.",
    "review_collection_terms_and_supply_published_accept_flags": "Read collection-terms and supply --accept-terms and, for a paid offer, --accept-offer.",
    "collection_enrollment_cancelled": "Enrollment cancelled.",
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
                metavar="SHA256",
                help="Published terms digest for unattended enrollment (interactive enrollment shows the terms)",
            )
            parser.add_argument("--accept-offer", metavar="OFFER_ID", help="Published offer ID for unattended paid enrollment")
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


def _review_paid_offer(published, args, cap):
    offer_id = published.get("offer_id")
    quotes = published.get("fixed_quotes")
    if not isinstance(offer_id, str) or not PAID_OFFER_ID.fullmatch(offer_id) or not isinstance(quotes, list) or not quotes:
        raise MinerSubmissionError("collection_comparison_prices_unavailable")
    print("Fixed reward and maximum OpenRouter spend per assignment (USD)", file=sys.stderr)
    seen = set()
    for quote in quotes:
        model = quote["model"]
        fee, budget = quote["quote_microusd"], quote["cap_microusd"]
        if (model not in MODELS or model in seen or quote["task_class"] != "P"
                or any(type(v) is not int or not 0 < v <= MAX_MICROUSD for v in (fee, budget))):
            raise MinerSubmissionError("collection_comparison_prices_unavailable")
        print(f"{model}  reward ${Decimal(fee).scaleb(-6):f}  spend cap ${Decimal(budget).scaleb(-6):f}", file=sys.stderr)
        seen.add(model)
    print(f"Your rolling 24-hour cap is ${Decimal(cap).scaleb(-6):f}. "
          f"Maximum assignments per UTC day: {getattr(args, 'max_slots_per_day', None) or 1}.", file=sys.stderr)
    pinned_offer = getattr(args, "accept_offer", None)
    if pinned_offer is not None and pinned_offer != offer_id:
        raise MinerSubmissionError("collection_offer_changed")
    return offer_id


def run(args, wallet, *, session=requests):
    action = args.command.removeprefix("collection-")
    try:
        base = _api_base_url(args.api_base_url) + "/arena/v1/collection/"
        task_scope = getattr(args, "task_scope", TASK_SCOPE)
        expected = TERMS_HASH
        text = TERMS
        offer_id = None
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
                if args.mode == "live" and task_scope == TASK_SCOPE and "offer_id" in published:
                    expected, text = PAID_WORKFLOW_TERMS_HASH, PAID_WORKFLOW_TERMS
                if (published.get("mode") != args.mode or published.get("task_scope") != task_scope
                        or published.get("terms_sha256") != expected or published.get("terms") != text):
                    raise MinerSubmissionError("collection_terms_changed")
        if action != "terms":
            creds, body = None, {}
            if action == "register":
                cap = daily_cap(args.daily_cap_usd)
                if args.mode == "live" and not published.get("fixed_quotes"):
                    raise MinerSubmissionError("collection_prices_unavailable")
                if expected == PAID_WORKFLOW_TERMS_HASH:
                    offer_id = _review_paid_offer(published, args, cap)
                elif getattr(args, "accept_offer", None) is not None:
                    raise MinerSubmissionError("collection_offer_changed")
                accepted = getattr(args, "accept_terms", None)
                if accepted is not None and accepted != expected:
                    raise MinerSubmissionError("collection_terms_changed")
                if accepted is None or (offer_id and not getattr(args, "accept_offer", None)):
                    if not sys.stdin.isatty():
                        raise MinerSubmissionError("review_collection_terms_and_supply_published_accept_flags")
                    print(text, file=sys.stderr)
                    if input("Sign this enrollment and spending cap? Type yes to continue: ").strip().lower() != "yes":
                        raise MinerSubmissionError("collection_enrollment_cancelled")
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
                if offer_id:
                    body["offer_id"] = offer_id
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
        EOFError,
        ValueError,
        KeyError,
        TypeError,
        MinerSubmissionError,
    ) as exc:
        print(
            (
                _ERROR_GUIDANCE.get(exc.code, exc.code)
                if isinstance(exc, MinerSubmissionError)
                else "Collection request failed."
            ),
            file=sys.stderr,
        )
        return 2
