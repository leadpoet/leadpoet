"""
ZeroBounce single-email verification — used as a fulfillment-only catch-all
fallback for TrueList.

Why this module exists
----------------------
TrueList's batch validator buckets a large fraction of midmarket and
enterprise mailboxes as ``accept_all`` / ``ok_for_all`` because their SMTP
servers respond identically for any mailbox name (they don't reveal whether
the mailbox actually exists).  That blanket reject is too aggressive — many
of those addresses are real and deliverable.  ZeroBounce maintains its own
mailbox-level intelligence (hash matching, supplemental MX probing, ML
confidence) that can cut through the catch-all wall on a meaningful slice
of those domains.

Design rules (DO NOT loosen without re-reading these)
-----------------------------------------------------
1.  This module is **only** called by the fulfillment scoring pipeline.
    Sourcing and qualification scoring must continue to treat catch-all
    as a hard reject — keeping behaviour identical there avoids miner
    confusion across products and avoids charging ZeroBounce credits
    against the much larger sourcing volume.
2.  We **only** call ZeroBounce when TrueList already classified the
    email as catch-all.  ZeroBounce is a fallback, not a primary
    verifier — paying ~$0.008/email for every fulfillment lead would
    triple email-verification cost.
3.  Only ZeroBounce ``status == "valid"`` flips the verdict to passed.
    Everything else (``invalid``, ``catch-all``, ``unknown``,
    ``do_not_mail``, ``spamtrap``, ``abuse``, errors, missing key, network
    failures) preserves the original TrueList catch-all rejection.  Better
    to lose a borderline lead than to accept a fabricated one.
4.  Every failure mode is swallowed and turned into ``valid=False`` so
    the scoring path can never crash because of a flaky ZeroBounce call.

API reference: https://www.zerobounce.net/docs/email-validation-api-quickstart/
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Dict, Optional

import aiohttp


logger = logging.getLogger(__name__)

ZEROBOUNCE_API_KEY = os.getenv("ZEROBOUNCE_API_KEY", "")
ZEROBOUNCE_TIMEOUT_S = float(os.getenv("ZEROBOUNCE_TIMEOUT_S", "20"))
ZEROBOUNCE_URL = "https://api.zerobounce.net/v2/validate"

# ZeroBounce returns one of these top-level statuses.  Only ``valid`` may
# override TrueList's catch-all.  The rest are treated as "still
# unverified" and we keep TrueList's original reject.
_VALID_STATUS = "valid"


async def zerobounce_validate(
    email: str,
    api_key: Optional[str] = None,
    timeout_s: Optional[float] = None,
) -> Dict[str, object]:
    """Verify a single email with ZeroBounce.

    Args:
        email:    Address to validate.  Must already be syntactically
                  reasonable — caller is expected to have run TrueList
                  first, so this only ever sees emails TrueList accepted
                  the syntax of.
        api_key:  Override.  When ``None`` (default), reads
                  ``ZEROBOUNCE_API_KEY`` from the env that was loaded at
                  module import time.
        timeout_s: Override per-call timeout.

    Returns:
        ``{
            "valid":      bool,   # True iff ZeroBounce says ``status == "valid"``
            "status":     str,    # ZeroBounce status, or sentinel for failures:
                                  # ``"skipped"`` when no API key,
                                  # ``"error"``   on network / parse errors.
            "sub_status": str,    # ZeroBounce sub_status for diagnostics.
            "error":      Optional[str],  # populated only on failure.
          }``
    """
    key = api_key if api_key is not None else ZEROBOUNCE_API_KEY
    timeout = timeout_s if timeout_s is not None else ZEROBOUNCE_TIMEOUT_S

    if not key:
        return {
            "valid": False,
            "status": "skipped",
            "sub_status": "no_api_key",
            "error": "ZEROBOUNCE_API_KEY not configured",
        }

    if not email or "@" not in email:
        return {
            "valid": False,
            "status": "error",
            "sub_status": "bad_input",
            "error": "Invalid email passed to zerobounce_validate",
        }

    params = {"api_key": key, "email": email}
    client_timeout = aiohttp.ClientTimeout(total=timeout)

    try:
        async with aiohttp.ClientSession(timeout=client_timeout) as session:
            async with session.get(ZEROBOUNCE_URL, params=params) as resp:
                if resp.status != 200:
                    body_preview = (await resp.text())[:200]
                    logger.warning(
                        "ZeroBounce HTTP %s for %s: %s",
                        resp.status, email, body_preview,
                    )
                    return {
                        "valid": False,
                        "status": "error",
                        "sub_status": f"http_{resp.status}",
                        "error": body_preview,
                    }

                data = await resp.json(content_type=None)
                status = (data.get("status") or "").strip().lower()
                sub_status = (data.get("sub_status") or "").strip().lower()
                return {
                    "valid": status == _VALID_STATUS,
                    "status": status or "unknown",
                    "sub_status": sub_status,
                    "error": None,
                }

    except asyncio.TimeoutError:
        logger.warning("ZeroBounce timeout for %s after %.0fs", email, timeout)
        return {
            "valid": False,
            "status": "error",
            "sub_status": "timeout",
            "error": f"timeout after {timeout:.0f}s",
        }
    except aiohttp.ClientError as e:
        logger.warning("ZeroBounce network error for %s: %s", email, e)
        return {
            "valid": False,
            "status": "error",
            "sub_status": "network",
            "error": str(e),
        }
    except Exception as e:  # parse error, anything else
        logger.warning("ZeroBounce unexpected error for %s: %s", email, e)
        return {
            "valid": False,
            "status": "error",
            "sub_status": "exception",
            "error": str(e),
        }


# TrueList sub_states that mean "domain accepts everything; we couldn't
# confirm this specific mailbox".  These are the only emails we route to
# ZeroBounce — anything else stays on TrueList's verdict.
TRUELIST_CATCH_ALL_STATUSES = frozenset({"accept_all", "ok_for_all"})


def is_truelist_catch_all(status: str) -> bool:
    """Return True iff a TrueList sub_state indicates a catch-all domain."""
    if not status:
        return False
    return status.strip().lower() in TRUELIST_CATCH_ALL_STATUSES
