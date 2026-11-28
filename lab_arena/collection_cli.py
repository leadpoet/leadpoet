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
from lab_arena.collection_pool import POOL_SCOPE, POOL_TERMS_BY_RATE
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
    "collection_pool_options_invalid": "Use --spend-limit-usd for a lifetime limit across sandbox experiments. This enrollment has no offer or daily slot limit.",
    "collection_standing_consent_required": "Register with --spend-limit-usd to authorize the live sales sandbox.",
    "collection_pool_inactive": "The live sales sandbox is not accepting standing keys.",
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
            "--mode", choices=("pilot", "live"), default="live" if action == "register" else None, required=action in ("statement", "revoke")
        )
        if action in ("terms", "register"):
            parser.add_argument("--task-scope", choices=(TASK_SCOPE, CORRECTION_SCOPE, POOL_SCOPE))
        if action == "terms":
            parser.add_argument("--offer", help="Read prices for a specific published offer")
        if action == "register":
            parser.add_argument(
                "--max-slots-per-day", type=int, choices=range(1, MAX_DAILY_SLOTS + 1),
                help="maximum distinct funded assignments per UTC day (default 1)",
            )
            limits = parser.add_mutually_exclusive_group(required=True)
            limits.add_argument("--spend-limit-usd", help="Total USD this enrollment may spend across sandbox experiments")
            limits.add_argument("--daily-cap-usd", help="Rolling 24h cap for legacy or pilot enrollment")
            parser.add_argument("--priority", type=int, choices=(0, 1), help="Standing key priority: 1 normal (default), 0 reserve")
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
    print("Validator reward targets and maximum OpenRouter spend per assignment (USD)", file=sys.stderr)
    print("Network votes and token prices determine the actual payout.", file=sys.stderr)
    history = published.get("historical_realization")
    if isinstance(history, dict):
        quoted = history.get("quoted_microusd")
        received = history.get("received_microusd_at_signed_price")
        if type(quoted) is int and quoted > 0 and type(received) is int and received >= 0:
            ratio = Decimal(received) / Decimal(quoted)
            print(f"At quote epoch {history['quote_epoch']}, miners received {100*ratio:.1f}% of targets "
                  "at the signed token price. At that rate and the measured average cost, "
                  f"the estimated margin would be {100*(Decimal('1.25')*ratio-1):.1f}%.", file=sys.stderr)
    seen = set()
    for quote in quotes:
        model = quote["model"]
        fee, budget = quote["quote_microusd"], quote["cap_microusd"]
        if (model not in MODELS or model in seen or quote["task_class"] not in {"P", "P8"}
                or any(type(v) is not int or not 0 < v <= MAX_MICROUSD for v in (fee, budget))):
            raise MinerSubmissionError("collection_comparison_prices_unavailable")
        print(f"{model}  target ${Decimal(fee).scaleb(-6):f}  spend cap ${Decimal(budget).scaleb(-6):f}", file=sys.stderr)
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
        requested_scope = getattr(args, "task_scope", None)
        standing = getattr(args, "spend_limit_usd", None) is not None
        priority = getattr(args, "priority", None)
        task_scope = requested_scope or (POOL_SCOPE if standing else TASK_SCOPE)
        if action == "register" and (
            standing != (task_scope == POOL_SCOPE)
            or (not standing and priority is not None)
            or (standing and (args.mode != "live" or getattr(args, "accept_offer", None)
                              or getattr(args, "max_slots_per_day", None) is not None))
        ):
            raise MinerSubmissionError("collection_pool_options_invalid")
        expected = TERMS_HASH
        text = TERMS
        offer_id = None
        if action in ("terms", "register"):
            url = base + "terms" + ("?task_scope=" + task_scope if task_scope in (CORRECTION_SCOPE, POOL_SCOPE) else "")
            selected = getattr(args, "offer", None) if action == "terms" else getattr(args, "accept_offer", None)
            if selected:
                if task_scope != TASK_SCOPE or not PAID_OFFER_ID.fullmatch(selected):
                    raise MinerSubmissionError("collection_offer_changed")
                url += "?offer_id=" + selected
            response = session.get(url, timeout=30, allow_redirects=False)
            if action == "register":
                if response.status_code != 200:
                    raise MinerSubmissionError("collection_correction_terms_unavailable" if task_scope == CORRECTION_SCOPE else "collection_terms_unavailable")
                published = response.json()
                if getattr(args, "max_slots_per_day", None) is not None and published.get(
                    "max_selectable_slots_per_utc_day"
                ) != MAX_DAILY_SLOTS:
                    raise MinerSubmissionError("collection_slot_opt_in_unavailable")
                if standing:
                    for rate, (text, expected) in POOL_TERMS_BY_RATE.items():
                        if published.get("reward_multiplier") == f"{rate / 10000:.2f}":
                            break
                    else:
                        raise MinerSubmissionError("collection_terms_changed")
                elif published.get("task_scope") == POOL_SCOPE:
                    raise MinerSubmissionError("collection_standing_consent_required")
                elif args.mode == "live" and task_scope == TASK_SCOPE and "offer_id" in published:
                    expected, text = PAID_WORKFLOW_TERMS_HASH, PAID_WORKFLOW_TERMS
                if (published.get("mode") != args.mode or published.get("task_scope") != task_scope
                        or published.get("terms_sha256") != expected or published.get("terms") != text):
                    raise MinerSubmissionError("collection_terms_changed")
        if action != "terms":
            creds, body = None, {}
            if action == "register":
                cap = daily_cap(args.spend_limit_usd if standing else args.daily_cap_usd)
                if args.mode == "live" and not standing and not published.get("fixed_quotes"):
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
                    if standing:
                        print(f"Total spending limit: ${Decimal(cap).scaleb(-6):f}. "
                              f"Priority: {1 if priority is None else priority}.", file=sys.stderr)
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
                    "terms_sha256": expected,
                    "task_scope": task_scope,
                    "credentials_sha256": contracts.document_hash(creds),
                }
                if standing:
                    body.update(spend_limit_microusd=cap, priority=1 if priority is None else priority)
                else:
                    body["daily_cap_microusd"] = cap
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
