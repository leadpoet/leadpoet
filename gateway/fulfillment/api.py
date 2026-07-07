"""
Fulfillment API Router

7 endpoints for the lead fulfillment commit-reveal system.
"""

import asyncio
import os
import logging
import base64
import threading
import time as _time
from uuid import UUID, uuid4
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Set

from dateutil.parser import isoparse as _isoparse
from fastapi import APIRouter, Header, HTTPException

from gateway.fulfillment.config import (
    T_EPOCHS, T_SECONDS_OVERRIDE, M_MINUTES,
    FULFILLMENT_BANS_ENABLED, FULFILLMENT_MAX_PARALLEL_REQUESTS,
    FULFILLMENT_MIN_REMAINING_WINDOW_MINUTES,
    FULFILLMENT_MINER_SUBMISSION_MULTIPLIER,
    epochs_to_seconds,
)
import math
from gateway.fulfillment.hashing import HASH_SCHEMA_VERSION, hash_request, verify_commit
from gateway.fulfillment.models import (
    FulfillmentICP,
    FulfillmentLead,
    FulfillmentCommitRequest,
    FulfillmentRevealRequest,
    LeadHashEntry,
    FulfillmentScoreResult,
    scrub_company_name,
)
from gateway.models.events import EventType
from gateway.utils.bans import is_hotkey_banned, is_hotkey_banned_sync, ban_hotkey
from gateway.utils.registry import is_registered_hotkey_async

logger = logging.getLogger(__name__)

_SIG_TIMESTAMP_TOLERANCE = 300  # 5 minutes

fulfillment_router = APIRouter(prefix="/fulfillment", tags=["fulfillment"])


def _enable_fulfillment() -> bool:
    return os.getenv("ENABLE_FULFILLMENT", "false").lower() == "true"


def _get_supabase():
    from gateway.db.client import get_write_client
    return get_write_client()


def _is_uuid(value: str) -> bool:
    try:
        UUID(str(value))
    except (TypeError, ValueError):
        return False
    return True


def _get_tempo(supabase) -> int:
    """Fetch current subnet tempo from DB, default 360."""
    try:
        resp = supabase.table("subnet_state") \
            .select("tempo") \
            .limit(1) \
            .execute()
        if resp.data:
            return int(resp.data[0].get("tempo", 360))
    except Exception:
        pass
    return 360


def _miner_submission_cap(num_leads: int) -> int:
    """Max leads a single miner can commit to a request.

    Returns ceil(num_leads * FULFILLMENT_MINER_SUBMISSION_MULTIPLIER), with a
    hard floor of num_leads (so the cap can never drop below what the client
    asked for — a miner must always be able to fulfill on their own).

    Default multiplier is 1.5, meaning for a 10-lead request a miner can
    submit up to 15 leads.  The surplus buffers against transient validation
    failures (TrueList/LinkedIn pass-rate is ~70-80%).  Only the top
    num_leads by score actually win rewards, so the cap has no effect on
    reward inflation — it only reduces the probability that the whole
    batch is discarded because one or two leads hit a coin-flip failure.
    """
    if num_leads <= 0:
        return 0
    cap = math.ceil(num_leads * FULFILLMENT_MINER_SUBMISSION_MULTIPLIER)
    return max(cap, num_leads)


# Statuses where a chain is still "in flight" — leads from these rows
# might still be delivered to the client.  Used by
# ``_load_in_flight_held_companies`` to scope the cross-chain held-lead
# scan.  Excludes terminal states (``fulfilled``, ``recycled``, ``expired``).
_IN_FLIGHT_REQUEST_STATES: List[str] = [
    "open",
    "continued_open",
    "pending",
    "commit_closed",
    "scoring",
    "partially_fulfilled",
]


def _load_in_flight_held_companies(
    supabase,
    client_company: str,
    exclude_request_ids: Optional[Set[str]] = None,
) -> List[str]:
    """Companies held on other in-flight chains for the same client.

    ``exclude_request_ids`` identifies "this chain" by request_id walk
    (matches ``_load_sibling_chain_held_companies``); label match would
    conflate independent chains under the same internal_label.
    """
    if not client_company or not client_company.strip():
        return []

    req_rows: List[dict] = []
    try:
        offset = 0
        for _ in range(20):
            page = supabase.table("fulfillment_requests") \
                .select("request_id") \
                .ilike("company", client_company.strip()) \
                .in_("status", _IN_FLIGHT_REQUEST_STATES) \
                .range(offset, offset + 999) \
                .execute()
            if not page.data:
                break
            req_rows.extend(page.data)
            if len(page.data) < 1000:
                break
            offset += 1000
    except Exception as e:
        logger.warning(
            f"_load_in_flight_held_companies: failed to fetch in-flight "
            f"requests for {client_company!r}: {type(e).__name__}: {e}"
        )
        return []

    excl: Set[str] = exclude_request_ids or set()
    other_chain_ids = [
        r["request_id"] for r in req_rows
        if r["request_id"] not in excl
    ]
    if not other_chain_ids:
        return []

    held_keys: List[tuple] = []
    try:
        # Chunk the request_id filter so the ``.in_()`` URL never overflows
        # when a client has many in-flight chains.
        for ci in range(0, len(other_chain_ids), 100):
            id_chunk = other_chain_ids[ci:ci + 100]
            offset = 0
            for _ in range(20):
                page = supabase.table("fulfillment_score_consensus") \
                    .select("submission_id,lead_id") \
                    .in_("request_id", id_chunk) \
                    .eq("is_chain_held", True) \
                    .range(offset, offset + 999) \
                    .execute()
                if not page.data:
                    break
                held_keys.extend(
                    (r["submission_id"], r["lead_id"]) for r in page.data
                )
                if len(page.data) < 1000:
                    break
                offset += 1000
    except Exception as e:
        logger.warning(
            f"_load_in_flight_held_companies: failed to fetch chain-held "
            f"consensus rows: {type(e).__name__}: {e}"
        )
        return []

    if not held_keys:
        return []

    submission_ids = list({sid for sid, _ in held_keys})
    subs_rows: List[dict] = []
    try:
        for i in range(0, len(submission_ids), 100):
            chunk = submission_ids[i:i + 100]
            offset = 0
            for _ in range(20):
                page = supabase.table("fulfillment_submissions") \
                    .select("submission_id,lead_data") \
                    .in_("submission_id", chunk) \
                    .range(offset, offset + 999) \
                    .execute()
                if not page.data:
                    break
                subs_rows.extend(page.data)
                if len(page.data) < 1000:
                    break
                offset += 1000
    except Exception as e:
        logger.warning(
            f"_load_in_flight_held_companies: failed to fetch submissions "
            f"for held leads: {type(e).__name__}: {e}"
        )
        return []

    lead_company: dict = {}
    for row in subs_rows:
        sid = row.get("submission_id")
        for entry in (row.get("lead_data") or []):
            lid = entry.get("lead_id")
            biz = (entry.get("data") or {}).get("business") or ""
            if sid and lid and biz:
                lead_company[(sid, lid)] = biz.strip()

    seen: set = set()
    out: List[str] = []
    for key in held_keys:
        biz = lead_company.get(key)
        if not biz:
            continue
        norm = biz.lower()
        if norm in seen:
            continue
        seen.add(norm)
        out.append(biz)
    return out


def _load_previously_delivered_companies(supabase, client_company: str) -> List[str]:
    """Pull every lead-company name this client has already received as a
    winner in a prior FULFILLED request.

    Only ``status='fulfilled'`` rows count — recycled / expired leads
    were never delivered to the client and never paid out to miners, so
    they remain eligible for inclusion in future batches.  Case-insensitive
    client-company match so "Apple" and "apple" are the same client.

    Returns the list of distinct winner ``business`` strings (original
    casing preserved), ready to drop into FulfillmentICP.excluded_companies.
    The ICP's field validator will handle dedup and normalization; this
    helper just gathers the raw candidates.
    """
    if not client_company or not client_company.strip():
        return []

    fulfilled_ids: List[str] = []
    try:
        offset = 0
        for _ in range(20):
            page = supabase.table("fulfillment_requests") \
                .select("request_id") \
                .ilike("company", client_company.strip()) \
                .eq("status", "fulfilled") \
                .range(offset, offset + 999) \
                .execute()
            if not page.data:
                break
            fulfilled_ids.extend(r["request_id"] for r in page.data)
            if len(page.data) < 1000:
                break
            offset += 1000
    except Exception as e:
        logger.warning(
            f"_load_previously_delivered_companies: failed to fetch fulfilled "
            f"requests for {client_company!r}: {type(e).__name__}: {e}"
        )
        return []

    if not fulfilled_ids:
        return []

    winner_keys: List[tuple] = []
    try:
        offset = 0
        for _ in range(20):
            page = supabase.table("fulfillment_score_consensus") \
                .select("submission_id,lead_id") \
                .in_("request_id", fulfilled_ids) \
                .eq("is_winner", True) \
                .range(offset, offset + 999) \
                .execute()
            if not page.data:
                break
            winner_keys.extend(
                (r["submission_id"], r["lead_id"]) for r in page.data
            )
            if len(page.data) < 1000:
                break
            offset += 1000
    except Exception as e:
        logger.warning(
            f"_load_previously_delivered_companies: failed to fetch winners: "
            f"{type(e).__name__}: {e}"
        )
        return []

    if not winner_keys:
        return []

    submission_ids = list({sid for sid, _ in winner_keys})
    subs_rows: List[dict] = []
    try:
        for i in range(0, len(submission_ids), 100):
            chunk = submission_ids[i:i + 100]
            offset = 0
            for _ in range(20):
                page = supabase.table("fulfillment_submissions") \
                    .select("submission_id,lead_data") \
                    .in_("submission_id", chunk) \
                    .range(offset, offset + 999) \
                    .execute()
                if not page.data:
                    break
                subs_rows.extend(page.data)
                if len(page.data) < 1000:
                    break
                offset += 1000
    except Exception as e:
        logger.warning(
            f"_load_previously_delivered_companies: failed to fetch "
            f"submissions: {type(e).__name__}: {e}"
        )
        return []

    # Build a (submission_id, lead_id) -> business lookup.
    lead_company: dict = {}
    for row in subs_rows:
        sid = row.get("submission_id")
        for entry in (row.get("lead_data") or []):
            lid = entry.get("lead_id")
            biz = (entry.get("data") or {}).get("business") or ""
            if sid and lid and biz:
                lead_company[(sid, lid)] = biz.strip()

    seen: set = set()
    out: List[str] = []
    for key in winner_keys:
        biz = lead_company.get(key)
        if not biz:
            continue
        norm = biz.lower()
        if norm in seen:
            continue
        seen.add(norm)
        out.append(biz)
    return out


def _log_event(event_type: EventType, actor_hotkey: str, payload: dict) -> None:
    """Best-effort transparency log insert.

    The transparency_log table enforces NOT NULL on several audit columns
    (``actor_hotkey``, ``nonce``, ``ts``, ``payload_hash``, ``build_id``,
    ``signature``).  The pre-fix version only populated ``event_type``,
    ``payload``, and ``created_at``, so every FULFILLMENT_* insert failed
    the NOT NULL check and was silently dropped by the except branch —
    meaning rejection reasons and commit/reveal/score events were missing
    from the log since the fulfillment system went live.

    Non-TEE gateway events (FULFILLMENT_*) can't carry a real enclave
    signature; we write ``signature=''`` so the NOT NULL is satisfied.
    TEE-signed events continue to go through gateway/utils/logger.py which
    does populate a real signature.
    """
    from gateway.config import BITTENSOR_NETWORK, BUILD_ID

    if BITTENSOR_NETWORK == "test":
        logger.info(
            f"⚠️ TESTNET MODE: Skipping {event_type.value} log to protect production transparency_log"
        )
        return

    import hashlib, json as _json, uuid as _uuid
    now = datetime.now(timezone.utc).isoformat()
    payload_hash = hashlib.sha256(
        _json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()

    try:
        supabase = _get_supabase()
        supabase.table("transparency_log").insert({
            "event_type": event_type.value,
            "actor_hotkey": actor_hotkey,
            "nonce": str(_uuid.uuid4()),
            "ts": now,
            "payload_hash": payload_hash,
            "build_id": BUILD_ID,
            "signature": "",
            "payload": payload,
            "created_at": now,
        }).execute()
    except Exception as e:
        logger.warning(f"Failed to log {event_type.value}: {e}")


async def _verify_validator_request(
    event_type: str, validator_hotkey: str,
    signature: str, nonce: str, timestamp: int,
    request_id: str = "",
) -> None:
    """Verify a validator's signature + confirm they are a registered validator.

    Raises HTTPException(403) on signature failure, unregistered hotkey, or
    non-validator role.  For request-scoped events, ``request_id`` binds the
    signature to a specific request; for global events (e.g. /scoring polling
    all requests), pass an empty string.
    """
    _verify_fulfillment_signature(
        event_type, validator_hotkey, request_id,
        signature, nonce, timestamp,
    )

    import asyncio
    try:
        is_registered, role = await asyncio.wait_for(
            is_registered_hotkey_async(validator_hotkey),
            timeout=30.0,
        )
    except asyncio.TimeoutError:
        raise HTTPException(504, detail="Metagraph query timeout — retry")

    if not is_registered:
        raise HTTPException(403, detail="Hotkey not registered on subnet")
    if role != "validator":
        raise HTTPException(403, detail="Only validators can call this endpoint")


def _verify_fulfillment_signature(
    event_type: str, hotkey: str, request_id: str,
    signature: str, nonce: str, timestamp: int,
) -> None:
    """Verify a miner's Ed25519 signature on a fulfillment request.
    The signed message binds to the request_id to prevent replay across requests.
    Raises HTTPException(403) on failure."""
    now_ts = int(_time.time())
    if abs(now_ts - timestamp) > _SIG_TIMESTAMP_TOLERANCE:
        raise HTTPException(403, detail="Timestamp too old or too far in the future")

    msg = f"{event_type}:{hotkey}:{request_id}:{nonce}:{timestamp}"
    try:
        from bittensor import Keypair
        sig_bytes = base64.b64decode(signature)
        kp = Keypair(ss58_address=hotkey)
        if not kp.verify(msg, sig_bytes):
            raise HTTPException(403, detail="Invalid signature")
    except HTTPException:
        raise
    except Exception as e:
        logger.warning(f"Signature verification error: {e}")
        raise HTTPException(403, detail="Signature verification failed")


# ---------------------------------------------------------------
# POST /fulfillment/request  — client creates a new ICP request
# ---------------------------------------------------------------
@fulfillment_router.post("/request")
async def create_request(
    icp: FulfillmentICP,
    authorization: str = Header(default=""),
):
    if not _enable_fulfillment():
        raise HTTPException(503, detail="Fulfillment system is not enabled on this gateway")

    # ───────────────────────────────────────────────────────────────────
    # Auth gate — shared-secret bearer token.
    #
    # We use the gateway's existing SUPABASE_SERVICE_ROLE_KEY as the
    # secret because (a) it's already provisioned in the gateway env,
    # so there's no new infrastructure to manage, and (b) only the
    # subnet operators hold it (it grants write access to the entire
    # production DB, so it would never be shared with miners or
    # clients anyway).  The team-only google doc that holds this key
    # IS the source of truth for who can create requests.
    #
    # NOTE: this key is NEVER hardcoded.  It is read from the
    # gateway process env at request time.  Don't log it, don't echo
    # it in error messages, don't write it anywhere.  hmac.compare_digest
    # gives us constant-time comparison so the failure path doesn't
    # leak partial-match timing info.
    #
    # Header format: ``Authorization: Bearer <SUPABASE_SERVICE_ROLE_KEY>``.
    #
    # Pre-existing requests created before this gate landed are not
    # affected — only new POSTs are challenged.
    # ───────────────────────────────────────────────────────────────────
    import hmac as _hmac
    expected_secret = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
    if not expected_secret:
        # Fail closed if the gateway itself is misconfigured rather than
        # silently allowing all requests through.
        logger.error(
            "create_request: SUPABASE_SERVICE_ROLE_KEY not set in gateway env — "
            "rejecting all requests until configured"
        )
        raise HTTPException(503, detail="Gateway auth misconfigured")

    presented = ""
    if authorization:
        parts = authorization.split(" ", 1)
        if len(parts) == 2 and parts[0].lower() == "bearer":
            presented = parts[1].strip()
        else:
            presented = authorization.strip()

    if not presented or not _hmac.compare_digest(presented, expected_secret):
        # Generic message — don't tell the caller whether their header
        # was missing, malformed, or wrong.  All three are 401.
        raise HTTPException(
            status_code=401,
            detail="Unauthorized.  POST /fulfillment/request requires a valid "
                   "bearer token in the Authorization header.",
        )

    # Enforce `company` at the API boundary only.  The model itself defaults
    # this to "" so later re-parses of the scrubbed icp_details dict by the
    # validator don't fail with "Field required" (which wedges scoring).
    # See models.py FulfillmentICP.company for the full rationale.
    if not (icp.company or "").strip():
        raise HTTPException(
            status_code=422,
            detail=[{
                "loc": ["body", "company"],
                "msg": "company is required",
                "type": "value_error.missing",
            }],
        )

    supabase = _get_supabase()
    now = datetime.now(timezone.utc)
    request_id = str(uuid4())

    # Scrub the client's company name out of every free-text field miners
    # will see BEFORE hashing / persistence.  Each match becomes
    # "[company_name]" so grammar stays natural.  `company` itself is never
    # serialized into model_dump() (Field(exclude=True)); it only lives in
    # the dedicated DB column we insert below.
    company = icp.company
    icp.prompt = scrub_company_name(icp.prompt, company)
    icp.product_service = scrub_company_name(icp.product_service, company)
    # Each entry is now an IntentSignalSpec (text + required).
    # Scrub the company name out of the ``text`` field only — the flags
    # don't carry text and are operator-set. Mutate in place via
    # model_copy(update=...) so the validator's coercion runs once on
    # ingest (above) and we don't re-create dicts here.
    #
    # Also auto-classify ``evidence_type`` when the operator/dashboard
    # left it null.  Two-stage classifier — fast deterministic regex
    # first, strict LLM fall-back second.  Both produce values from the
    # closed enum {HIRING, FUNDING, SOCIAL_POSTING, CASE_STUDY, OTHER,
    # PODCAST_APPEARANCE, TECHSTACK}.  After this block runs every spec
    # HAS a non-null evidence_type — if classification fails after
    # retries, we raise HTTP 400 so the operator gets actionable
    # feedback instead of a silent fail-open.
    from qualification.scoring.intent_precheck import (
        _classify_target_type,
        llm_classify_evidence_type,
    )

    async def _autotag_evidence_type(spec, idx: int):
        text_preview = (spec.text or "")[:100].replace("\n", " ")
        # 1. Operator-confirmed → respect verbatim
        if spec.evidence_type is not None:
            logger.info(
                "create_request: signal#%d stage=operator result=%s text=%r",
                idx, spec.evidence_type, text_preview,
            )
            return spec
        # 2. Regex first (zero cost, deterministic)
        cls = _classify_target_type(spec.text)
        if cls is not None:
            logger.info(
                "create_request: signal#%d stage=regex result=%s text=%r",
                idx, cls, text_preview,
            )
            return spec.model_copy(update={"evidence_type": cls})
        # 3. LLM fallback (strict closed-enum prompt; 3 retries inside)
        logger.info(
            "create_request: signal#%d stage=regex result=None — falling "
            "back to LLM; text=%r", idx, text_preview,
        )
        cls = await llm_classify_evidence_type(spec.text)
        if cls is not None:
            logger.info(
                "create_request: signal#%d stage=llm result=%s text=%r",
                idx, cls, text_preview,
            )
            return spec.model_copy(update={"evidence_type": cls})
        # 4. Both failed → fail-closed.  Operator must rephrase the signal
        # so the classifier can resolve it, or set evidence_type manually
        # in the dashboard before re-submitting.
        logger.error(
            "create_request: signal#%d stage=failed — REJECTING request; "
            "both regex and LLM classifiers returned None; text=%r",
            idx, text_preview,
        )
        raise HTTPException(
            status_code=400,
            detail=(
                "evidence_type_classification_failed: could not assign an "
                "evidence_type to intent_signal {!r}.  Please rephrase the "
                "signal so it clearly indicates HIRING / FUNDING / "
                "SOCIAL_POSTING / PODCAST_APPEARANCE / TECHSTACK / "
                "CASE_STUDY / OTHER, or set evidence_type explicitly when "
                "creating the request."
            ).format(spec.text[:160]),
        )

    tagged_signals = []
    for idx, spec in enumerate(icp.intent_signals):
        scrubbed = spec.model_copy(
            update={"text": scrub_company_name(spec.text, company)}
        )
        tagged_signals.append(await _autotag_evidence_type(scrubbed, idx))
    icp.intent_signals = tagged_signals
    icp.target_roles = [scrub_company_name(s, company) for s in icp.target_roles]

    # Auto-expand target_roles into common variant spellings / near-
    # synonyms so downstream per-lead matching doesn't reject legitimate
    # leads just because their LinkedIn title uses a slightly different
    # wording (e.g. "VP Sales" vs client-submitted "VP of Sales").  One
    # LLM call per request at creation time, deterministic afterwards:
    # the expanded list is locked into icp_details before hashing, so
    # every miner + validator sees the same canonical list for the
    # life of the request.  Failure of the LLM call leaves icp.target_roles
    # unchanged — the function is best-effort.
    #
    # Operators may set ``expand_target_roles=False`` on the create payload
    # to store the seed list verbatim (Field(exclude=True) — not in icp_details).
    if icp.expand_target_roles:
        try:
            from gateway.fulfillment.role_expander import expand_target_roles
            expanded = await expand_target_roles(
                icp.target_roles,
                target_seniority=icp.target_seniority,
            )
            if expanded and len(expanded) > len(icp.target_roles):
                logger.info(
                    f"create_request: expanded target_roles "
                    f"{len(icp.target_roles)} → {len(expanded)}"
                )
                icp.target_roles = expanded
        except Exception as e:
            logger.warning(
                f"create_request: target_roles expansion failed (keeping seeds): "
                f"{type(e).__name__}: {e}"
            )
    else:
        logger.info(
            "create_request: expand_target_roles=False — storing target_roles verbatim"
        )

    # Auto-populate excluded_companies from the client's prior fulfilled
    # requests ONLY when the client didn't supply the list themselves.
    # Client-supplied lists (even empty) take precedence — if the client
    # explicitly wants "no exclusions" they can pass ``["__none__"]`` or
    # similar, but in practice an empty list means "please auto-populate".
    # Only FULFILLED predecessors count: recycled / expired requests had
    # their leads discarded (miners were never paid, client never
    # received them), so they should remain eligible for the new batch.
    if not icp.excluded_companies:
        try:
            delivered = _load_previously_delivered_companies(supabase, company)
            in_flight = _load_in_flight_held_companies(
                supabase, company,
                exclude_request_ids=None,
            )
            seen = {b.strip().lower() for b in delivered if b.strip()}
            merged = list(delivered)
            for biz in in_flight:
                if biz.strip().lower() not in seen:
                    merged.append(biz)
                    seen.add(biz.strip().lower())
            icp.excluded_companies = merged
            if icp.excluded_companies:
                logger.info(
                    f"create_request: auto-populated excluded_companies for "
                    f"company={company!r}: {len(delivered)} delivered + "
                    f"{len(in_flight)} cross-chain held = "
                    f"{len(icp.excluded_companies)} total"
                )
        except Exception as e:
            logger.warning(
                f"create_request: excluded_companies auto-populate failed "
                f"(proceeding with empty list): {type(e).__name__}: {e}"
            )
            icp.excluded_companies = []

    # model_dump() excludes `internal_label` and `company` (both Field(exclude=True))
    # so neither lands in icp_details (which is what miners see).
    icp_dict = icp.model_dump(mode="json")

    # ── TRANSITION COMPAT (2026-05-18 split-region rename) ─────────────
    # The new FulfillmentICP model emits ``company_country`` / ``company_region``
    # (no legacy ``country`` / ``geography`` keys).  A validator still running
    # pre-rename code would read this row's icp_details, find NEITHER the old
    # nor the new keys it knows about, and silently bypass the country gate
    # (Pydantic ``extra='ignore'`` swallows the unknown ``company_*`` keys).
    # To stay forward-compatible during the rolling deploy, also emit the
    # legacy keys here.  Old validators read them as before; new validators
    # have a @model_validator(mode='before') that prefers the new keys.
    #
    # Remove this block once all validators are confirmed on post-2026-05-18
    # code (after Phase 2 of scripts/23-split-region-migration.sql runs).
    if icp.company_country:
        icp_dict["country"] = list(icp.company_country)
    if icp.company_region:
        icp_dict["geography"] = icp.company_region

    req_hash = hash_request(icp_dict)

    # New requests enter the queue as `pending`: no commit timer yet.
    # The lifecycle tick promotes the oldest pending rows to `open` (and
    # stamps window_start = NOW, window_end = NOW + T_EPOCHS,
    # reveal_window_end = window_end + M_MINUTES) as soon as the open
    # pool has room (max FULFILLMENT_MAX_PARALLEL_REQUESTS at a time).
    # This guarantees a request's commit window only starts ticking once
    # miners can actually see it, so queued requests can never silently
    # expire while invisible.
    row = {
        "request_id": request_id,
        "request_hash": req_hash,
        "icp_details": icp_dict,
        "num_leads": icp.num_leads,
        "window_start": None,
        "window_end": None,
        "reveal_window_end": None,
        "status": "pending",
        "created_by": "api",
        # `company` is validated as required by Pydantic, so always present.
        "company": company,
    }
    # Only attach the label if the client actually sent one — this way the
    # insert still works against older DBs that don't yet have the
    # `internal_label` column (fall-through retry below).
    if icp.internal_label:
        row["internal_label"] = icp.internal_label
    # required_attributes lives in a dedicated DB column (not inside icp_details).
    # Persist it on the initial insert so Tier 2c attribute verification can run
    # — otherwise the column stays null and the gate is silently skipped.
    if icp.required_attributes:
        row["required_attributes"] = icp.required_attributes
    try:
        supabase.table("fulfillment_requests").insert(row).execute()
    except Exception as e:
        # If a newly-added column doesn't exist on this deployment yet,
        # retry without it so request creation never hard-blocks on schema
        # drift.  Handles `internal_label`, `company`, and `required_attributes`.
        err = str(e)
        retried = False
        for missing in ("internal_label", "company", "required_attributes"):
            if missing in err and missing in row:
                row.pop(missing, None)
                retried = True
        if retried:
            supabase.table("fulfillment_requests").insert(row).execute()
        else:
            raise

    _log_event(EventType.FULFILLMENT_REQUEST_CREATED, "gateway", {
        "request_id": request_id,
        "request_hash": req_hash,
        "status": "pending",
    })

    return {
        "request_id": request_id,
        "request_hash": req_hash,
        "status": "pending",
        "num_leads": icp.num_leads,
        "note": (
            "Request is queued.  Its commit window starts only when a slot "
            f"opens in the miner-visible pool (max {FULFILLMENT_MAX_PARALLEL_REQUESTS} "
            "concurrent).  Query /fulfillment/results/{request_id} to follow progress."
        ),
    }


# ---------------------------------------------------------------
# GET /fulfillment/requests/active  — miners poll for open ICPs
# ---------------------------------------------------------------

# The open pool is identical for every miner and only changes on lifecycle
# ticks (every 30s), so the base query is cached for a few seconds instead
# of hitting the DB once per poll.  With the whole miner population polling
# on a 30s cadence, this collapses hundreds of identical queries per minute
# into ~one per TTL; a few seconds of staleness is invisible next to the
# multi-minute commit windows the rows describe.
_ACTIVE_REQUESTS_CACHE_TTL_SECONDS = float(
    os.getenv("FULFILLMENT_ACTIVE_CACHE_TTL_SECONDS", "5")
)
_active_requests_cache = {"fetched_at_mono": 0.0, "rows": None}
_active_requests_cache_lock = threading.Lock()


def _fetch_active_request_rows(supabase, cutoff: str) -> list:
    """Fetch the miner-visible open pool, cached for a few seconds.

    The lock deliberately covers the DB fetch: when the cache expires under
    concurrent polling, the first caller refreshes while the rest briefly
    serialize behind it — one query instead of a thundering herd.

    Callers must NOT mutate the returned rows (they are shared across
    requests); the handler below builds fresh dicts for everything it
    returns.
    """
    with _active_requests_cache_lock:
        now_mono = _time.monotonic()
        if (
            _active_requests_cache["rows"] is not None
            and now_mono - _active_requests_cache["fetched_at_mono"]
            < _ACTIVE_REQUESTS_CACHE_TTL_SECONDS
        ):
            return _active_requests_cache["rows"]

        # FIFO: return up to FULFILLMENT_MAX_PARALLEL_REQUESTS oldest visible
        # requests.  Miners may work on any/all of them in parallel. Once a
        # request is fulfilled/recycled/expired/partially_fulfilled, the next
        # one in line becomes visible.
        #
        # Both 'open' (fresh) and 'continued_open' (chain continuation) are
        # surfaced — they're functionally identical for the miner's purposes
        # (commit window, reveal window, num_leads, ICP shape).  The status
        # label is preserved on each entry's "status" field so miners that
        # care can detect "this request is part of an in-flight chain;
        # rewards only flow when the chain reaches its full quota".
        resp = supabase.table("fulfillment_requests") \
            .select("*") \
            .in_("status", ["open", "continued_open"]) \
            .gt("window_end", cutoff) \
            .order("window_start", desc=False) \
            .limit(FULFILLMENT_MAX_PARALLEL_REQUESTS) \
            .execute()

        _active_requests_cache["rows"] = resp.data or []
        _active_requests_cache["fetched_at_mono"] = now_mono
        return _active_requests_cache["rows"]


@fulfillment_router.get("/requests/active")
async def get_active_requests(miner_hotkey: str = ""):
    # The gateway runs a single event loop, and this is its highest-traffic
    # endpoint.  The sync Supabase client blocks whatever thread runs it, so
    # executing the handler body inline would stall every in-flight request
    # for the duration of each DB round-trip — under normal polling load the
    # loop saturates and the whole gateway stops answering.  Running the
    # body in a worker thread keeps the loop free; HTTPExceptions raised
    # inside propagate unchanged.
    return await asyncio.to_thread(_get_active_requests_impl, miner_hotkey)


def _get_active_requests_impl(miner_hotkey: str = ""):
    if not _enable_fulfillment():
        raise HTTPException(503, detail="Fulfillment system is not enabled")

    supabase = _get_supabase()

    if miner_hotkey:
        # Sync variant: this function already runs in a worker thread.
        banned, reason = is_hotkey_banned_sync(miner_hotkey)
        if banned:
            raise HTTPException(403, detail=f"Hotkey banned: {reason}")

    now = datetime.now(timezone.utc)

    # Only surface requests with at least N minutes of commit window left,
    # so a miner isn't handed a request they cannot realistically commit
    # to before it expires. Requests that fall below this threshold are
    # held back and will either be picked up by already-sourcing miners
    # (who hold a local copy from earlier polls) or expire and recycle
    # into a fresh successor with the full window.
    min_remaining = timedelta(minutes=FULFILLMENT_MIN_REMAINING_WINDOW_MINUTES)
    cutoff = (now + min_remaining).isoformat()

    rows = _fetch_active_request_rows(supabase, cutoff)

    # Per-miner committed counts for every visible request in ONE batched
    # round-trip.  A per-request lookup here multiplies each poll into
    # 1+N sequential queries, which is what saturates the endpoint under
    # normal polling load.
    committed_by_request = {}
    if miner_hotkey and rows:
        existing = supabase.table("fulfillment_submissions") \
            .select("request_id, lead_hashes") \
            .in_("request_id", [r["request_id"] for r in rows]) \
            .eq("miner_hotkey", miner_hotkey) \
            .execute()
        for s in (existing.data or []):
            # First row per request wins, matching the previous per-request
            # lookup which read only the first returned row.
            committed_by_request.setdefault(
                s["request_id"], len(s.get("lead_hashes") or [])
            )

    requests_out = []
    for r in rows:
        per_miner_cap = _miner_submission_cap(r["num_leads"])

        # Hide requests this miner has already fully committed to.  The
        # threshold is the per-miner CAP (num_leads * multiplier), not
        # num_leads itself — a miner can commit up to the cap, so they
        # are only "done" once they've hit it.  Using num_leads here
        # would hide a request from a miner who committed exactly N
        # but still has headroom to commit up to 1.5×N.
        if miner_hotkey:
            if committed_by_request.get(r["request_id"], 0) >= per_miner_cap:
                continue  # fully committed — hide this request

        icp = dict(r.get("icp_details", {}) or {})

        # Miner compatibility: ICP's employee_count is stored as a list of
        # canonical buckets (e.g. ["201-500", "501-1,000", "1,001-5,000"])
        # but the miner-facing JSON surface has always been a single range
        # string.  We include BOTH:
        #   - ``employee_count``: collapsed range string (back-compat)
        #   - ``employee_count_buckets``: the authoritative list of allowed
        #     canonical buckets (new field).  Miners that want exact-bucket
        #     matching should prefer this — the validator's scoring is an
        #     exact set-membership check against these buckets, so anything
        #     not in the list will be rejected as employee_count_mismatch.
        ec_val = icp.get("employee_count")
        try:
            from gateway.fulfillment.models import _BUCKET_RANGES, range_string_to_buckets
        except Exception:
            _BUCKET_RANGES = {}
            range_string_to_buckets = lambda _s: []
        if isinstance(ec_val, list):
            buckets = [b for b in ec_val if b in _BUCKET_RANGES]
        elif isinstance(ec_val, str) and ec_val:
            # Legacy stored string — coerce to canonical buckets for the
            # new `employee_count_buckets` field so miners that opt into
            # the exact-bucket vocabulary see the same set the scorer uses.
            buckets = range_string_to_buckets(ec_val)
        else:
            buckets = []
        if buckets:
            los, his = zip(*(_BUCKET_RANGES[b] for b in buckets))
            icp["employee_count"] = f"{min(los)}-{max(his)}"
        else:
            icp["employee_count"] = ""
        icp["employee_count_buckets"] = buckets

        # Industry / sub-industry: same dual-shape treatment as
        # employee_count.  ``industry`` (and ``sub_industry``) are now
        # multi-valued internally but historically miners parsed them as
        # single strings.  Expose:
        #   - ``industry`` / ``sub_industry``       : comma-joined string
        #     (back-compat for any miner that already does
        #     ``r["icp"]["industry"] == "Software"``-style checks).
        #     With multi-value ICPs this WILL contain commas (e.g.
        #     ``"Food and Beverage, Health Care"``); legacy miners that
        #     do exact-match comparisons should migrate to the lists below.
        #   - ``industries`` / ``sub_industries``   : authoritative lists.
        #     These are what the validator's Tier 1 gate uses (set-
        #     membership), so any miner that wants their submitted leads
        #     to pass ICP fit MUST check the lead's industry/sub_industry
        #     against these lists.
        ind_val = icp.get("industry")
        if isinstance(ind_val, list):
            ind_list = [s for s in ind_val if isinstance(s, str) and s]
        elif isinstance(ind_val, str) and ind_val:
            ind_list = [ind_val]
        else:
            ind_list = []
        icp["industry"] = ", ".join(ind_list)
        icp["industries"] = ind_list

        sub_val = icp.get("sub_industry")
        if isinstance(sub_val, list):
            sub_list = [s for s in sub_val if isinstance(s, str) and s]
        elif isinstance(sub_val, str) and sub_val:
            sub_list = [sub_val]
        else:
            sub_list = []
        icp["sub_industry"] = ", ".join(sub_list)
        icp["sub_industries"] = sub_list

        # ── Location fields exposed to miners ──────────────────────────
        # 2026-05-18: split into company_* / contact_* per-side filters.
        # Miners need both shapes (str + list) for back-compat: legacy
        # miners read ``country`` as a comma-joined string, while modern
        # ones consume the authoritative list.  We expose:
        #
        #   - ``company_country``  : List[str]   (authoritative — Tier 1a gate)
        #   - ``company_region``   : str         (Tier 1.5a LLM gate target)
        #   - ``contact_country``  : List[str]   (Tier 1b gate; usually empty)
        #   - ``contact_region``   : str         (Tier 1.5b LLM gate; usually empty)
        #
        # PLUS legacy aliases for miners that haven't migrated:
        #   - ``country``    : comma-joined company_country (was: same)
        #   - ``countries``  : list, same as company_country (was: same)
        #   - ``geography``  : company_region (was: same)
        #
        # The legacy aliases will be removed once all miner code has
        # migrated; keeping them now means existing miners don't break.
        def _to_list(v):
            if isinstance(v, list):
                return [s for s in v if isinstance(s, str) and s]
            if isinstance(v, str) and v:
                return [v]
            return []
        company_country_list = _to_list(icp.get("company_country"))
        contact_country_list = _to_list(icp.get("contact_country"))
        company_region = icp.get("company_region") or ""
        contact_region = icp.get("contact_region") or ""
        # If neither new nor legacy is populated, fall back to the legacy
        # ``country`` / ``geography`` keys for icp_details rows that
        # haven't been touched by the model_validator yet (defensive).
        if not company_country_list:
            company_country_list = _to_list(icp.get("country") or icp.get("countries"))
        if not company_region:
            company_region = icp.get("geography") or ""
        # Set authoritative new keys
        icp["company_country"] = company_country_list
        icp["company_region"]  = company_region
        icp["contact_country"] = contact_country_list
        icp["contact_region"]  = contact_region
        # Maintain legacy aliases (company-side meaning)
        icp["country"]   = ", ".join(company_country_list)
        icp["countries"] = company_country_list
        icp["geography"] = company_region

        # Merge the top-level required_attributes column into the miner-
        # facing icp payload. required_attributes is stored as a dedicated
        # DB column (not inside icp_details JSONB), so without this merge
        # miners never see which attributes the buyer requires — yet the
        # validator's Tier 2c gate (verify_required_attributes) DOES read
        # them from this same column and rejects leads that fail with
        # ``required_attribute_failed``.  The reveals endpoint at
        # ``/fulfillment/scoring`` performs the equivalent merge (see
        # ~L1158); mirroring it here keeps the miner-visible ICP shape
        # identical to what the scorer sees, so miners can pre-filter on
        # the buyer's requirements instead of submitting blind.
        if r.get("required_attributes"):
            icp["required_attributes"] = r["required_attributes"]

        requests_out.append({
            "request_id": r["request_id"],
            "icp": icp,
            "num_leads": r["num_leads"],
            "max_submissions_per_miner": per_miner_cap,
            "window_end": r["window_end"],
            "reveal_window_end": r["reveal_window_end"],
            # Miner-facing status hint:
            #   "open"           — fresh request; rewards flow normally
            #                       at the end of THIS cycle if quota is
            #                       met.
            #   "continued_open" — chain continuation; some leads were
            #                       already accepted in earlier cycles
            #                       and are held server-side.  Rewards
            #                       only flow when the chain reaches its
            #                       full quota across all generations.
            #                       num_leads here is the REMAINING
            #                       quota (chain_target − held_count),
            #                       not the chain's full size.
            "status": r.get("status", "open"),
        })

    return {
        "requests": requests_out,
        "gateway_server_time": now.isoformat(),
    }


@fulfillment_router.get("/excluded-now/{request_id}")
async def get_excluded_now(request_id: str):
    """Live cross-chain held set for a given request.

    Miners poll this before sourcing to avoid spending money on companies
    a sibling chain has already claimed.  Returns the NORMALIZED company
    keys (lowercase, legal-suffix stripped) matching the same form the
    reveal-time check uses, so a miner who normalizes locally with the
    same rule gets a clean intersection check.
    """
    if not _enable_fulfillment():
        raise HTTPException(503, detail="Fulfillment system is not enabled")
    supabase = _get_supabase()
    from gateway.fulfillment.lifecycle import _load_sibling_chain_held_companies
    excluded = sorted(_load_sibling_chain_held_companies(supabase, request_id))
    return {
        "request_id": request_id,
        "excluded_normalized": excluded,
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------
# POST /fulfillment/commit  — miner submits lead hashes
# ---------------------------------------------------------------
@fulfillment_router.post("/commit")
async def commit_leads(commit: FulfillmentCommitRequest):
    if not _enable_fulfillment():
        raise HTTPException(503, detail="Fulfillment system is not enabled")

    _verify_fulfillment_signature(
        "FULFILLMENT_COMMIT", commit.miner_hotkey, commit.request_id,
        commit.signature, commit.nonce, commit.timestamp,
    )

    supabase = _get_supabase()

    banned, reason = await is_hotkey_banned(commit.miner_hotkey)
    if banned:
        raise HTTPException(403, detail=f"Hotkey banned: {reason}")

    if commit.schema_version != HASH_SCHEMA_VERSION:
        raise HTTPException(422, detail=(
            f"Schema version mismatch: client={commit.schema_version}, "
            f"server={HASH_SCHEMA_VERSION}. Update your miner."
        ))

    req_resp = supabase.table("fulfillment_requests") \
        .select("*") \
        .eq("request_id", commit.request_id) \
        .execute()
    if not req_resp.data:
        raise HTTPException(404, detail="Request not found")

    req = req_resp.data[0]
    # Both 'open' (fresh) and 'continued_open' (chain continuation) accept
    # commits — they have identical commit window semantics; only the
    # status label differs to flag chain continuations to miners.
    if req["status"] not in ("open", "continued_open"):
        raise HTTPException(400, detail=f"Window not open (status={req['status']})")

    now = datetime.now(timezone.utc)
    if now > _isoparse(req["window_end"]):
        raise HTTPException(400, detail="Commit window expired")

    # Per-miner cap: num_leads * FULFILLMENT_MINER_SUBMISSION_MULTIPLIER,
    # ceil'd.  Default 1.5 so a miner has headroom for transient validation
    # failures (TrueList / LinkedIn scrape flakiness).  Only the top
    # num_leads by score actually win rewards, so the surplus is pure
    # headroom for the miner; it cannot inflate emission.
    num_leads_target = req["num_leads"]
    num_leads_max = _miner_submission_cap(num_leads_target)
    cap_suffix = (
        f"request asks for {num_leads_target}; per-miner cap is "
        f"{num_leads_max} (= {FULFILLMENT_MINER_SUBMISSION_MULTIPLIER}× ceiling) "
        f"to absorb validation flakiness."
    )

    # Check for existing submission (allows appending up to num_leads_max)
    existing_sub = supabase.table("fulfillment_submissions") \
        .select("submission_id, lead_hashes") \
        .eq("request_id", commit.request_id) \
        .eq("miner_hotkey", commit.miner_hotkey) \
        .execute()

    new_entries: List[dict] = []
    for entry in commit.lead_hashes:
        new_entries.append({
            "lead_id": str(uuid4()),
            "hash": entry.hash,
        })

    if existing_sub.data:
        # Append to existing submission
        sub = existing_sub.data[0]
        submission_id = sub["submission_id"]
        existing_hashes = sub.get("lead_hashes", []) or []

        total_after = len(existing_hashes) + len(new_entries)
        if total_after > num_leads_max:
            raise HTTPException(422, detail=(
                f"Too many leads: already committed {len(existing_hashes)}, "
                f"adding {len(new_entries)} would total {total_after} "
                f"which exceeds cap {num_leads_max}.  {cap_suffix}"
            ))

        if len(existing_hashes) >= num_leads_max:
            raise HTTPException(409, detail={
                "message": f"Already at cap: {len(existing_hashes)}/{num_leads_max} leads committed",
                "submission_id": submission_id,
                "num_leads": num_leads_target,
                "max_submissions_per_miner": num_leads_max,
            })

        merged_hashes = existing_hashes + new_entries
        try:
            supabase.table("fulfillment_submissions") \
                .update({"lead_hashes": merged_hashes}) \
                .eq("submission_id", submission_id) \
                .execute()
        except Exception as e:
            raise HTTPException(500, detail=f"Append commit failed: {str(e)}")
    else:
        # First commit for this miner + request
        if len(new_entries) > num_leads_max:
            raise HTTPException(422, detail=(
                f"Too many leads: submitted {len(new_entries)}, "
                f"cap is {num_leads_max}.  {cap_suffix}"
            ))

        # Direct INSERT instead of the legacy fulfillment_accept_commit RPC.
        #
        # The RPC was hardcoded to require source ``status='open'`` and
        # rejected every commit to a ``continued_open`` request with
        # ``Window not open (status=continued_open)`` (raised as Postgres
        # P0001), surfacing to miners as HTTP 500.  Same class of latent
        # bug as the lifecycle's fulfillment_close_window RPC fixed earlier
        # today — any code path that hardcodes status='open' breaks under
        # the new partial-fulfillment chain semantics.
        #
        # The status check is already done above (line ~696) where we
        # accept ``status IN (open, continued_open)``, and the commit
        # window expiry check is right after it.  So by the time we
        # reach this branch, the status is verified-good and we can do
        # a plain INSERT.  Race safety still comes from the unique
        # constraint on (request_id, miner_hotkey) — a duplicate insert
        # raises and we surface a 409 to the miner.
        from uuid import uuid4 as _uuid4
        new_submission_id = str(_uuid4())
        try:
            supabase.table("fulfillment_submissions").insert({
                "submission_id": new_submission_id,
                "request_id": commit.request_id,
                "miner_hotkey": commit.miner_hotkey,
                "num_leads": len(new_entries),
                "lead_hashes": new_entries,
                "revealed": False,
            }).execute()
            submission_id = new_submission_id
        except Exception as e:
            err_msg = str(e)
            if "unique" in err_msg.lower() or "duplicate" in err_msg.lower():
                raise HTTPException(409, detail="Race condition — retry")
            raise HTTPException(500, detail=f"Commit failed: {err_msg}")

    lead_hash_entries = new_entries

    _log_event(EventType.FULFILLMENT_COMMIT, commit.miner_hotkey, {
        "request_id": commit.request_id,
        "submission_id": submission_id,
        "miner_hotkey": commit.miner_hotkey,
        "lead_hashes": lead_hash_entries,
        "submission_timestamp": now.isoformat(),
    })

    return {
        "submission_id": submission_id,
        "lead_ids": [e["lead_id"] for e in lead_hash_entries],
    }


# ---------------------------------------------------------------
# POST /fulfillment/reveal  — miner reveals lead data
# ---------------------------------------------------------------
@fulfillment_router.post("/reveal")
async def reveal_leads(reveal: FulfillmentRevealRequest):
    if not _enable_fulfillment():
        raise HTTPException(503, detail="Fulfillment system is not enabled")

    _verify_fulfillment_signature(
        "FULFILLMENT_REVEAL", reveal.miner_hotkey, reveal.request_id,
        reveal.signature, reveal.nonce, reveal.timestamp,
    )

    supabase = _get_supabase()

    banned, reason = await is_hotkey_banned(reveal.miner_hotkey)
    if banned:
        raise HTTPException(403, detail=f"Hotkey banned: {reason}")

    sub_resp = supabase.table("fulfillment_submissions") \
        .select("*") \
        .eq("submission_id", reveal.submission_id) \
        .eq("request_id", reveal.request_id) \
        .eq("miner_hotkey", reveal.miner_hotkey) \
        .execute()
    if not sub_resp.data:
        raise HTTPException(404, detail="Submission not found")

    submission = sub_resp.data[0]
    if submission["revealed"]:
        raise HTTPException(400, detail="Already revealed")

    req_resp = supabase.table("fulfillment_requests") \
        .select("window_end, reveal_window_end, icp_details") \
        .eq("request_id", reveal.request_id) \
        .execute()
    if not req_resp.data:
        raise HTTPException(404, detail="Request not found")

    req = req_resp.data[0]
    now = datetime.now(timezone.utc)
    window_end_dt = _isoparse(req["window_end"])
    reveal_end_dt = _isoparse(req["reveal_window_end"])

    if now < window_end_dt:
        raise HTTPException(400, detail="Commit window still open — cannot reveal yet")
    if now > reveal_end_dt:
        raise HTTPException(400, detail="Reveal window expired")

    # ------------------------------------------------------------------
    # Determine how many client intent-signal slots this request has, so
    # we can validate each miner-submitted IntentSignal.matched_icp_signal
    # below.  The miner MUST tag every intent signal with a zero-based
    # index into this list — leads with any signal failing that contract
    # are rejected at reveal time and never reach scoring.
    # ------------------------------------------------------------------
    icp_details = req.get("icp_details") or {}
    client_intent_signals = icp_details.get("intent_signals") or []
    num_client_intent_signals = len(client_intent_signals)

    committed_hashes: list = submission["lead_hashes"]
    if len(reveal.leads) != len(committed_hashes):
        raise HTTPException(422, detail=(
            f"Must reveal all committed leads: expected {len(committed_hashes)}, got {len(reveal.leads)}"
        ))

    lead_data_list = []
    mismatched = []
    matched_icp_invalid = []
    for i, lead in enumerate(reveal.leads):
        lead_dict = lead.model_dump(mode="json")
        committed_hash = committed_hashes[i]["hash"]
        if not verify_commit(committed_hash, lead_dict):
            mismatched.append({
                "index": i,
                "lead_id": committed_hashes[i]["lead_id"],
            })
            continue

        # --------------------------------------------------------------
        # GATEWAY-LEVEL MATCHED_ICP_SIGNAL ENFORCEMENT
        # Every intent signal on a lead must declare which client-listed
        # intent signal it proves.  -1 (default) or an out-of-range index
        # means the miner has not adapted to the matched_icp_signal contract
        # — drop the lead with an explicit error rather than silently
        # zero-scoring it downstream.
        # --------------------------------------------------------------
        bad_signal_indexes = []
        for sig_idx, sig in enumerate(lead.intent_signals):
            mapped = getattr(sig, "matched_icp_signal", -1)
            if (
                not isinstance(mapped, int)
                or mapped < 0
                or mapped >= num_client_intent_signals
            ):
                bad_signal_indexes.append({
                    "signal_index": sig_idx,
                    "matched_icp_signal": mapped,
                    "reason": (
                        "missing_or_negative"
                        if (not isinstance(mapped, int) or mapped < 0)
                        else "out_of_range"
                    ),
                })
        if bad_signal_indexes:
            matched_icp_invalid.append({
                "index": i,
                "lead_id": committed_hashes[i]["lead_id"],
                "num_client_intent_signals": num_client_intent_signals,
                "bad_signals": bad_signal_indexes,
            })
            continue

        lead_data_list.append({
            "lead_id": committed_hashes[i]["lead_id"],
            "data": lead_dict,
        })

    pre_rejected: List[dict] = []
    if lead_data_list:
        from gateway.fulfillment.lifecycle import _load_sibling_chain_held_companies
        from gateway.fulfillment.normalize import normalize_company
        sibling_held = _load_sibling_chain_held_companies(supabase, reveal.request_id)
        if sibling_held:
            kept: List[dict] = []
            for entry in lead_data_list:
                biz = (entry.get("data") or {}).get("business") or ""
                norm = normalize_company(biz)
                if norm and norm in sibling_held:
                    pre_rejected.append({
                        "lead_id": entry["lead_id"],
                        "business": biz,
                        "reason": "sibling_chain_held",
                    })
                else:
                    kept.append(entry)
            lead_data_list = kept

    if not lead_data_list:
        if matched_icp_invalid and not mismatched:
            if num_client_intent_signals == 0:
                valid_range_msg = (
                    "this request has no client-defined intent signals to "
                    "match against — contact the operator"
                )
            else:
                valid_range_msg = (
                    f"every intent signal must set matched_icp_signal to a "
                    f"valid index in [0, {num_client_intent_signals - 1}]"
                )
            raise HTTPException(
                400,
                detail=(
                    f"All {len(reveal.leads)} lead(s) rejected: {valid_range_msg}. "
                    f"Update your miner code to populate matched_icp_signal on every "
                    f"IntentSignal."
                ),
            )
        raise HTTPException(
            400,
            detail=(
                f"All {len(reveal.leads)} lead(s) rejected: "
                f"{len(mismatched)} failed hash verification, "
                f"{len(matched_icp_invalid)} failed matched_icp_signal validation, "
                f"{len(pre_rejected)} held by sibling chain"
            ),
        )

    supabase.table("fulfillment_submissions").update({
        "revealed": True,
        "revealed_at": now.isoformat(),
        "lead_data": lead_data_list,
    }).eq("submission_id", reveal.submission_id).execute()

    print(f"✅ REVEAL stored: request={reveal.request_id[:8]}... "
          f"sub={reveal.submission_id[:8]}... miner={reveal.miner_hotkey[:8]}... "
          f"leads={len(lead_data_list)}/{len(reveal.leads)} revealed=True"
          + (f" (dropped {len(mismatched)} mismatched)" if mismatched else "")
          + (f" (dropped {len(matched_icp_invalid)} matched_icp_invalid)" if matched_icp_invalid else "")
          + (f" (dropped {len(pre_rejected)} sibling-chain-held)" if pre_rejected else ""))

    _log_event(EventType.FULFILLMENT_REVEAL, reveal.miner_hotkey, {
        "request_id": reveal.request_id,
        "miner_hotkey": reveal.miner_hotkey,
        "reveal_timestamp": now.isoformat(),
        "mismatched_indices": [m["index"] for m in mismatched],
        "matched_icp_invalid_indices": [m["index"] for m in matched_icp_invalid],
        "pre_rejected_lead_ids": [p["lead_id"] for p in pre_rejected],
    })

    return {
        "status": "revealed",
        "num_leads": len(lead_data_list),
        "mismatched": mismatched,
        "matched_icp_invalid": matched_icp_invalid,
        "pre_rejected": pre_rejected,
    }


# ---------------------------------------------------------------
# GET /fulfillment/scoring  — validators fetch revealed leads for scoring
# ---------------------------------------------------------------
def _collect_scoring_requests_sync(validator_hotkey: str) -> dict:
    """Assemble the /fulfillment/scoring payload (the blocking DB work).

    Runs in a worker thread (via asyncio.to_thread) so the synchronous
    Supabase calls never block the gateway's single event loop.  Previously
    this ran inline in the async endpoint and froze the loop for 30s+ per
    call, which timed out concurrent validator reveal-fetches AND miners'
    submit/reveal requests.

    Two latency optimisations vs the old inline version (output is identical):
      * the "already scored" lookup is scoped to the CURRENT scoring
        request_ids instead of scanning this validator's entire
        fulfillment_scores history;
      * revealed submissions are fetched in one chunked in_(...) query grouped
        by request_id instead of one paged query per request (the N+1).
    """
    supabase = _get_supabase()

    # 1. All requests currently in scoring status (paged).
    scoring_requests: List[dict] = []
    offset = 0
    for _ in range(20):
        page = supabase.table("fulfillment_requests") \
            .select("*") \
            .eq("status", "scoring") \
            .range(offset, offset + 999) \
            .execute()
        if not page.data:
            break
        scoring_requests.extend(page.data)
        if len(page.data) < 1000:
            break
        offset += 1000

    if not scoring_requests:
        return {"requests": []}
    print(f"📋 /fulfillment/scoring: {len(scoring_requests)} request(s) in scoring status")
    scoring_ids = [r["request_id"] for r in scoring_requests]

    # 2. Which of THESE requests has this validator already scored?  Scoped to
    #    the current scoring ids (was: a full fulfillment_scores history scan).
    already_scored_requests = set()
    if validator_hotkey:
        for i in range(0, len(scoring_ids), 100):
            chunk = scoring_ids[i:i + 100]
            offset = 0
            for _ in range(20):
                page = supabase.table("fulfillment_scores") \
                    .select("request_id") \
                    .eq("validator_hotkey", validator_hotkey) \
                    .in_("request_id", chunk) \
                    .range(offset, offset + 999) \
                    .execute()
                if not page.data:
                    break
                already_scored_requests.update(r["request_id"] for r in page.data)
                if len(page.data) < 1000:
                    break
                offset += 1000
        if already_scored_requests:
            print(f"   Validator {validator_hotkey[:8]}... already scored "
                  f"{len(already_scored_requests)} of {len(scoring_ids)}")

    needed_ids = [rid for rid in scoring_ids if rid not in already_scored_requests]
    if not needed_ids:
        return {"requests": []}

    # 3. Batch-fetch all revealed submissions for the needed requests in one
    #    chunked, paged query grouped by request_id (was: one paged query per
    #    request — the N+1 that pushed the endpoint past 30s).
    subs_by_req = {}
    for i in range(0, len(needed_ids), 100):
        chunk = needed_ids[i:i + 100]
        offset = 0
        for _ in range(20):
            page = supabase.table("fulfillment_submissions") \
                .select("*") \
                .eq("revealed", True) \
                .in_("request_id", chunk) \
                .order("submission_id") \
                .range(offset, offset + 999) \
                .execute()
            if not page.data:
                break
            for s in page.data:
                subs_by_req.setdefault(s["request_id"], []).append(s)
            if len(page.data) < 1000:
                break
            offset += 1000

    # 4. Build the response — SAME shape + index-alignment invariant as before.
    req_by_id = {r["request_id"]: r for r in scoring_requests}
    out = []
    for rid in needed_ids:
        r = req_by_id[rid]
        submissions = []
        for s in subs_by_req.get(rid, []):
            # SAFETY: `leads` and `lead_ids` MUST come from the same
            # `lead_data` list so they stay index-aligned — the validator
            # zips them onto scores, and mismatched lengths silently corrupt
            # consensus / winner selection.  lead_data entries are
            # {"lead_id": ..., "data": ...} so both projections are safe.
            lead_data = s.get("lead_data") or []
            submissions.append({
                "submission_id": s["submission_id"],
                "miner_hotkey": s["miner_hotkey"],
                "leads": [entry.get("data", {}) for entry in lead_data],
                "lead_ids": [entry.get("lead_id", "") for entry in lead_data],
            })

        print(f"   Returning {rid[:8]}... with {len(submissions)} submission(s), "
              f"{sum(len(s['leads']) for s in submissions)} total leads")

        # Merge the top-level required_attributes column into icp_details so
        # the validator's `FulfillmentICP(**icp_details)` reconstruction sees
        # it. required_attributes is a dedicated column (not inside icp_details
        # JSONB), so omitting this merge meant Tier 2c never fired.
        icp_payload = dict(r.get("icp_details", {}) or {})
        if r.get("required_attributes"):
            icp_payload["required_attributes"] = r["required_attributes"]

        out.append({
            "request_id": rid,
            "icp": icp_payload,
            "status": r["status"],
            "submissions": submissions,
        })

    return {"requests": out}


@fulfillment_router.get("/scoring")
async def get_scoring_requests(
    validator_hotkey: str,
    signature: str,
    nonce: str,
    timestamp: int,
):
    if not _enable_fulfillment():
        raise HTTPException(503, detail="Fulfillment system is not enabled")

    await _verify_validator_request(
        "FULFILLMENT_SCORING", validator_hotkey,
        signature, nonce, timestamp,
        request_id="",
    )

    # The blocking Supabase work runs OFF the event loop so it can never freeze
    # the gateway (which would time out miner submit/reveal calls and other
    # validators) while this single request assembles its payload.
    import asyncio
    return await asyncio.to_thread(_collect_scoring_requests_sync, validator_hotkey)


# ---------------------------------------------------------------
# POST /fulfillment/score  — validator submits scores
# ---------------------------------------------------------------
@fulfillment_router.post("/score")
async def submit_scores(
    request_id: str,
    validator_hotkey: str,
    signature: str,
    nonce: str,
    timestamp: int,
    scores: List[dict],
):
    if not _enable_fulfillment():
        raise HTTPException(503, detail="Fulfillment system is not enabled")

    await _verify_validator_request(
        "FULFILLMENT_SCORE", validator_hotkey,
        signature, nonce, timestamp,
        request_id=request_id,
    )

    supabase = _get_supabase()

    for s in scores:
        if not s.get("request_id"):
            s["request_id"] = request_id

    try:
        supabase.rpc("fulfillment_upsert_scores", {
            "p_scores": scores,
            "p_validator_hotkey": validator_hotkey,
        }).execute()
    except Exception as e:
        raise HTTPException(500, detail=f"Score submission failed: {e}")

    # The fulfillment_upsert_scores RPC was defined before the
    # `intent_signals_detail` column existed on fulfillment_scores; it
    # silently drops that field from the payload when writing the row.
    # Empirically 0 of 345 rows had this column populated pre-fix even
    # though the validator has always been POSTing it in the payload.
    # Without this column, consensus can't build intent_signal_mapping,
    # and the client-facing intent_details narrative falls back to an
    # LLM error message saying "no intent signals were provided".
    # Patch it with a direct UPDATE keyed on (request_id, validator,
    # lead_id) which is the same natural key the RPC upserts on.  This
    # is best-effort: one failed patch doesn't fail the whole submission.
    for s in scores:
        patch_fields = {}
        if s.get("intent_signals_detail"):
            patch_fields["intent_signals_detail"] = s["intent_signals_detail"]
        if s.get("failure_detail"):
            patch_fields["failure_detail"] = s["failure_detail"]
        # Tier 2c attribute-verification blob is also dropped by the RPC for
        # the same reason intent_signals_detail is (column added after RPC
        # signature was frozen).  Re-patch it directly so per-validator
        # per-lead attribute results are persisted for consensus + audit.
        if s.get("attribute_verification") is not None:
            patch_fields["attribute_verification"] = s["attribute_verification"]
        if not patch_fields:
            continue
        lid = s.get("lead_id")
        if not lid:
            continue
        try:
            supabase.table("fulfillment_scores").update(
                patch_fields
            ).eq("request_id", s.get("request_id", request_id)) \
              .eq("validator_hotkey", validator_hotkey) \
              .eq("lead_id", lid) \
              .execute()
        except Exception as e:
            logger.warning(
                f"Failed to patch fields for lead "
                f"{str(lid)[:8]}: {e}"
            )

    _log_event(EventType.FULFILLMENT_SCORED, validator_hotkey, {
        "request_id": request_id,
        "scores": [
            {"miner_hotkey": s.get("miner_hotkey"), "lead_id": s.get("lead_id"),
             "score": s.get("final_score"), "reason": s.get("failure_reason"),
             "detail": s.get("failure_detail")}
            for s in scores
        ],
        "score_timestamp": datetime.now(timezone.utc).isoformat(),
    })

    return {"status": "scores_accepted", "count": len(scores)}


# ---------------------------------------------------------------
# GET /fulfillment/results/{request_id}  — client fetches results
# ---------------------------------------------------------------
@fulfillment_router.get("/results/{request_id}")
async def get_results(request_id: str):
    if not _enable_fulfillment():
        raise HTTPException(503, detail="Fulfillment system is not enabled")
    if not _is_uuid(request_id):
        raise HTTPException(404, detail="Request not found")

    supabase = _get_supabase()

    req_resp = supabase.table("fulfillment_requests") \
        .select("status, successor_request_id") \
        .eq("request_id", request_id) \
        .execute()
    if not req_resp.data:
        raise HTTPException(404, detail="Request not found")

    req = req_resp.data[0]

    consensus_rows: List[dict] = []
    offset = 0
    for _ in range(20):
        page = supabase.table("fulfillment_score_consensus") \
            .select("*") \
            .eq("request_id", request_id) \
            .order("consensus_final_score", desc=True) \
            .range(offset, offset + 999) \
            .execute()
        if not page.data:
            break
        consensus_rows.extend(page.data)
        if len(page.data) < 1000:
            break
        offset += 1000

    leads = []
    for row in consensus_rows:
        lead_row = {
            "lead_id": row["lead_id"],
            "miner_hotkey": row["miner_hotkey"],
            "consensus_final_score": row["consensus_final_score"],
            "consensus_intent_signal_final": row["consensus_intent_signal_final"],
            "is_winner": row["is_winner"],
            "num_validators": row["num_validators"],
            "any_fabricated": row["any_fabricated"],
            "consensus_tier2_passed": row["consensus_tier2_passed"],
            "consensus_email_verified": row.get("consensus_email_verified"),
            "consensus_person_verified": row.get("consensus_person_verified"),
            "consensus_company_verified": row.get("consensus_company_verified"),
            "consensus_rep_score": row.get("consensus_rep_score"),
            # Client-facing enrichments.  Populated only for winners; others
            # get null/empty but the keys are always present for a stable API
            # shape.
            "intent_signal_mapping": row.get("intent_signal_mapping") or [],
            "intent_details": row.get("intent_details"),
        }
        leads.append(lead_row)

    result = {
        "request_id": request_id,
        "request_status": req["status"],
        "leads": leads,
        "total_leads": len(leads),
    }
    if req.get("successor_request_id"):
        result["successor_request_id"] = req["successor_request_id"]
    return result


# ---------------------------------------------------------------
# GET /fulfillment/rewards/active  — validator fetches active rewards
# ---------------------------------------------------------------
@fulfillment_router.get("/rewards/active")
async def get_active_rewards(current_epoch: int):
    """Return active (unexpired) fulfillment rewards grouped by miner hotkey.

    Used by the validator during weight calculation to determine the
    fulfillment emission carve-out from the sourcing allocation.

    The DB pagination runs in a worker thread (asyncio.to_thread) so the
    synchronous Supabase calls never block the gateway event loop. The
    validator's client times out at 45s/attempt and treats an exhausted
    fetch as "no active rewards" — which zeroes the fulfillment emission
    share and burns it. Run inline, this query queued behind the lifecycle
    tick and exceeded 50s under PostgREST latency, so the validator was
    dropping ALL fulfillment rewards some epochs. Offloading keeps it well
    under the 45s budget even during slow spells.
    """
    import asyncio
    return await asyncio.to_thread(_collect_active_rewards_sync, current_epoch)


def _collect_active_rewards_sync(current_epoch: int) -> dict:
    """Synchronous body of GET /fulfillment/rewards/active — runs in a worker
    thread so it never blocks the event loop.  See the endpoint docstring."""
    supabase = _get_supabase()

    all_rows: List[dict] = []
    offset = 0
    for _ in range(50):
        page = supabase.table("fulfillment_score_consensus") \
            .select("miner_hotkey, reward_pct, reward_expires_epoch") \
            .not_.is_("reward_pct", "null") \
            .gt("reward_expires_epoch", current_epoch) \
            .range(offset, offset + 999) \
            .execute()
        if not page.data:
            break
        all_rows.extend(page.data)
        if len(page.data) < 1000:
            break
        offset += 1000

    per_miner: dict = {}
    for row in all_rows:
        hk = row["miner_hotkey"]
        pct = float(row["reward_pct"])
        per_miner[hk] = per_miner.get(hk, 0.0) + pct

    return {"rewards": per_miner, "total_active_rows": len(all_rows)}


# ---------------------------------------------------------------
# GET /fulfillment/leaderboard  — top fulfillment miners (rolling window)
# ---------------------------------------------------------------
def _rolling_epoch_window_start(epochs: int = 140) -> datetime:
    """Return the start of a rolling window covering the last ``epochs`` epochs.

    1 epoch = 360 blocks × 12 s/block = 4 320 s.
    Default 140 epochs = 604 800 s = exactly 7.0 days.

    Unlike the previous Monday-reset approach, this window always covers
    exactly the same duration regardless of what day of the week it is —
    a miner's wins never drop off a cliff at midnight Monday.
    """
    window_seconds = epochs * 360 * 12  # 4320 s/epoch
    return datetime.now(timezone.utc) - timedelta(seconds=window_seconds)


@fulfillment_router.get("/leaderboard")
async def get_fulfillment_leaderboard(limit: int = 3):
    """Return top fulfillment miners ranked by `is_winner` count in the last 140 epochs.

    Window: rolling 140-epoch window (~7.0 days) anchored to the current
    wall-clock time.  1 epoch = 360 blocks × 12 s/block = 4 320 s, so
    140 epochs = 604 800 s = exactly 7.0 days.

    Computed from ``fulfillment_score_consensus.computed_at`` —
    any winning row whose ``computed_at`` falls within the last 140 epochs
    is counted.

    Used by the validator each weight-set cycle to identify the top-3
    miners eligible for the leaderboard emission bonus
    (LEADERBOARD_BONUS_SHARE in neurons/validator.py — split 5 / 3 / 1.5%
    across rank 1, 2, 3).  Also intended to power a public dashboard.

    Banned hotkeys are filtered out — we don't surface or pay them.
    Ties are broken by total `reward_pct` (sums the partial-fulfillment
    weighting) so a miner who won 10 leads at full weight ranks above
    one who won 10 at half weight.

    Args:
        limit: max number of top miners to return (default 3, capped at 100).

    Returns:
        {
          "leaderboard": [
            {"rank": 1, "miner_hotkey": "5...", "wins": 87, "total_reward_pct": 4.32},
            ...
          ],
          "computed_at": "2026-05-17T03:14:00Z",
          "period_start": "2026-05-10T03:14:00+00:00",
          "period_end": "2026-05-17T03:14:00+00:00",
          "total_unique_winners": 12
        }
    """
    if limit < 1:
        limit = 1
    if limit > 100:
        limit = 100

    supabase = _get_supabase()

    # Rolling 140-epoch window: count rows whose consensus computed_at falls
    # within the last 140 epochs (604 800 s = 7.0 days) from now.
    # Schema confirmed: `computed_at` exists on fulfillment_score_consensus
    # and is written by the lifecycle when the consensus row is created.
    window_start = _rolling_epoch_window_start(epochs=140)
    now_iso = datetime.now(timezone.utc).isoformat()

    winner_rows: List[dict] = []
    offset = 0
    for _ in range(50):
        page = supabase.table("fulfillment_score_consensus") \
            .select("miner_hotkey, reward_pct") \
            .eq("is_winner", True) \
            .gte("computed_at", window_start.isoformat()) \
            .range(offset, offset + 999) \
            .execute()
        if not page.data:
            break
        winner_rows.extend(page.data)
        if len(page.data) < 1000:
            break
        offset += 1000

    banned_set: set = set()
    try:
        offset = 0
        for _ in range(20):
            page = supabase.table("banned_hotkeys") \
                .select("hotkey") \
                .range(offset, offset + 999) \
                .execute()
            if not page.data:
                break
            banned_set.update(r["hotkey"] for r in page.data)
            if len(page.data) < 1000:
                break
            offset += 1000
    except Exception:
        banned_set = set()

    per_miner: dict = {}
    for row in winner_rows:
        hk = row["miner_hotkey"]
        if hk in banned_set:
            continue
        rec = per_miner.setdefault(hk, {"wins": 0, "total_reward_pct": 0.0})
        rec["wins"] += 1
        try:
            rec["total_reward_pct"] += float(row.get("reward_pct") or 0.0)
        except (TypeError, ValueError):
            pass

    # Rank: wins desc, total_reward_pct desc as tiebreaker
    ranked = sorted(
        per_miner.items(),
        key=lambda kv: (-kv[1]["wins"], -kv[1]["total_reward_pct"]),
    )[:limit]

    leaderboard = [
        {
            "rank": i + 1,
            "miner_hotkey": hk,
            "wins": rec["wins"],
            "total_reward_pct": round(rec["total_reward_pct"], 4),
        }
        for i, (hk, rec) in enumerate(ranked)
    ]

    return {
        "leaderboard": leaderboard,
        "computed_at": now_iso,
        "period_start": window_start.isoformat(),
        "period_end": now_iso,
        "total_unique_winners": len(per_miner),
    }


# ---------------------------------------------------------------
# POST /fulfillment/ban/{hotkey}  — validator requests a ban
# ---------------------------------------------------------------
@fulfillment_router.post("/ban/{hotkey}")
async def request_ban(
    hotkey: str,
    reason: str = "",
    validator_hotkey: str = "",
    request_id: str = "",
    signature: str = "",
    nonce: str = "",
    timestamp: int = 0,
):
    if not _enable_fulfillment():
        raise HTTPException(503, detail="Fulfillment system is not enabled")

    if not validator_hotkey or not signature or not nonce or not timestamp:
        raise HTTPException(
            403,
            detail="Ban requests must be signed by a validator (validator_hotkey, signature, nonce, timestamp required)",
        )
    await _verify_validator_request(
        "FULFILLMENT_BAN", validator_hotkey,
        signature, nonce, timestamp,
        request_id=request_id,
    )

    _log_event(EventType.FULFILLMENT_BAN, validator_hotkey, {
        "hotkey": hotkey,
        "reason": reason,
        "banned_by": validator_hotkey,
        "request_id": request_id,
    })

    if not FULFILLMENT_BANS_ENABLED:
        return {
            "action": "logged_only",
            "detail": "Ban logged but not executed — FULFILLMENT_BANS_ENABLED is false",
        }

    supabase = _get_supabase()
    if request_id:
        check = supabase.table("fulfillment_scores") \
            .select("score_id") \
            .eq("request_id", request_id) \
            .eq("miner_hotkey", hotkey) \
            .eq("all_fabricated", True) \
            .limit(1) \
            .execute()
        if not check.data:
            raise HTTPException(400, detail="No fabricated leads found for this hotkey on the given request")

    success = await ban_hotkey(hotkey, reason or "Fulfillment fabrication", validator_hotkey or "system")
    if not success:
        raise HTTPException(500, detail="Ban execution failed")

    q = supabase.table("fulfillment_score_consensus").update({
        "reward_pct": None,
    }).eq("miner_hotkey", hotkey)
    if request_id:
        q = q.eq("request_id", request_id)
    q.execute()

    return {"action": "banned", "hotkey": hotkey}
