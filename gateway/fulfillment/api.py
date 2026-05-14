"""
Fulfillment API Router

7 endpoints for the lead fulfillment commit-reveal system.
"""

import os
import logging
import base64
import time as _time
from uuid import uuid4
from datetime import datetime, timezone, timedelta
from typing import List

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
from gateway.utils.bans import is_hotkey_banned, ban_hotkey
from gateway.utils.registry import is_registered_hotkey_async

logger = logging.getLogger(__name__)

_SIG_TIMESTAMP_TOLERANCE = 300  # 5 minutes

fulfillment_router = APIRouter(prefix="/fulfillment", tags=["fulfillment"])


def _enable_fulfillment() -> bool:
    return os.getenv("ENABLE_FULFILLMENT", "false").lower() == "true"


def _get_supabase():
    from gateway.db.client import get_write_client
    return get_write_client()


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

    try:
        # Step 1: find every fulfilled request for this client.
        req_resp = supabase.table("fulfillment_requests") \
            .select("request_id") \
            .ilike("company", client_company.strip()) \
            .eq("status", "fulfilled") \
            .execute()
        fulfilled_ids = [r["request_id"] for r in (req_resp.data or [])]
    except Exception as e:
        logger.warning(
            f"_load_previously_delivered_companies: failed to fetch fulfilled "
            f"requests for {client_company!r}: {type(e).__name__}: {e}"
        )
        return []

    if not fulfilled_ids:
        return []

    # Step 2: find every winning consensus row across those requests.
    try:
        cons_resp = supabase.table("fulfillment_score_consensus") \
            .select("submission_id,lead_id") \
            .in_("request_id", fulfilled_ids) \
            .eq("is_winner", True) \
            .execute()
        winner_keys = [(r["submission_id"], r["lead_id"]) for r in (cons_resp.data or [])]
    except Exception as e:
        logger.warning(
            f"_load_previously_delivered_companies: failed to fetch winners: "
            f"{type(e).__name__}: {e}"
        )
        return []

    if not winner_keys:
        return []

    # Step 3: pull lead_data for those submissions and extract each
    # winner's ``business`` field.  The consensus row identifies the
    # exact (submission_id, lead_id) pair; the lead_data column is the
    # JSON array the miner committed + revealed, keyed by lead_id.
    submission_ids = list({sid for sid, _ in winner_keys})
    try:
        subs_resp = supabase.table("fulfillment_submissions") \
            .select("submission_id,lead_data") \
            .in_("submission_id", submission_ids) \
            .execute()
    except Exception as e:
        logger.warning(
            f"_load_previously_delivered_companies: failed to fetch "
            f"submissions: {type(e).__name__}: {e}"
        )
        return []

    # Build a (submission_id, lead_id) -> business lookup.
    lead_company: dict = {}
    for row in (subs_resp.data or []):
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
    # Each entry is now an IntentSignalSpec (text + required + is_scored).
    # Scrub the company name out of the ``text`` field only — the flags
    # don't carry text and are operator-set. Mutate in place via
    # model_copy(update=...) so the validator's coercion runs once on
    # ingest (above) and we don't re-create dicts here.
    icp.intent_signals = [
        spec.model_copy(update={"text": scrub_company_name(spec.text, company)})
        for spec in icp.intent_signals
    ]
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
            icp.excluded_companies = _load_previously_delivered_companies(
                supabase, company
            )
            if icp.excluded_companies:
                logger.info(
                    f"create_request: auto-populated excluded_companies for "
                    f"company={company!r}: {len(icp.excluded_companies)} entries"
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
    try:
        supabase.table("fulfillment_requests").insert(row).execute()
    except Exception as e:
        # If a newly-added column doesn't exist on this deployment yet,
        # retry without it so request creation never hard-blocks on schema
        # drift.  Handles both `internal_label` and `company`.
        err = str(e)
        retried = False
        for missing in ("internal_label", "company"):
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
@fulfillment_router.get("/requests/active")
async def get_active_requests(miner_hotkey: str = ""):
    if not _enable_fulfillment():
        raise HTTPException(503, detail="Fulfillment system is not enabled")

    supabase = _get_supabase()

    if miner_hotkey:
        banned, reason = await is_hotkey_banned(miner_hotkey)
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

    requests_out = []
    for r in (resp.data or []):
        per_miner_cap = _miner_submission_cap(r["num_leads"])

        # Hide requests this miner has already fully committed to.  The
        # threshold is the per-miner CAP (num_leads * multiplier), not
        # num_leads itself — a miner can commit up to the cap, so they
        # are only "done" once they've hit it.  Using num_leads here
        # would hide a request from a miner who committed exactly N
        # but still has headroom to commit up to 1.5×N.
        if miner_hotkey:
            existing = supabase.table("fulfillment_submissions") \
                .select("submission_id, lead_hashes") \
                .eq("request_id", r["request_id"]) \
                .eq("miner_hotkey", miner_hotkey) \
                .execute()
            if existing.data:
                committed_count = len(existing.data[0].get("lead_hashes", []))
                if committed_count >= per_miner_cap:
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

        # Country: same dual-shape treatment as industry/sub_industry.
        # ``country`` is now a list internally but legacy miners parsed it
        # as a single string.  Expose:
        #   - ``country``    : comma-joined string (back-compat).  For
        #                      multi-country ICPs this WILL contain commas
        #                      (e.g. ``"Brazil, Argentina, Colombia"``);
        #                      legacy miners doing exact-match comparisons
        #                      should migrate to the list below.
        #   - ``countries``  : authoritative list.  This is what the
        #                      validator's Tier 1 gate uses (set-membership
        #                      with alias normalization), so any miner that
        #                      wants their leads to pass the country check
        #                      MUST verify lead.company_hq_country against
        #                      this list.  Empty list = "any country".
        country_val = icp.get("country")
        if isinstance(country_val, list):
            country_list = [s for s in country_val if isinstance(s, str) and s]
        elif isinstance(country_val, str) and country_val:
            country_list = [country_val]
        else:
            country_list = []
        icp["country"] = ", ".join(country_list)
        icp["countries"] = country_list

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
        .select("window_end, reveal_window_end") \
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

    committed_hashes: list = submission["lead_hashes"]
    if len(reveal.leads) != len(committed_hashes):
        raise HTTPException(422, detail=(
            f"Must reveal all committed leads: expected {len(committed_hashes)}, got {len(reveal.leads)}"
        ))

    lead_data_list = []
    mismatched = []
    for i, lead in enumerate(reveal.leads):
        lead_dict = lead.model_dump(mode="json")
        committed_hash = committed_hashes[i]["hash"]
        if not verify_commit(committed_hash, lead_dict):
            mismatched.append({
                "index": i,
                "lead_id": committed_hashes[i]["lead_id"],
            })
            continue
        lead_data_list.append({
            "lead_id": committed_hashes[i]["lead_id"],
            "data": lead_dict,
        })

    if not lead_data_list:
        raise HTTPException(
            400,
            detail=f"All {len(reveal.leads)} lead(s) failed hash verification",
        )

    supabase.table("fulfillment_submissions").update({
        "revealed": True,
        "revealed_at": now.isoformat(),
        "lead_data": lead_data_list,
    }).eq("submission_id", reveal.submission_id).execute()

    print(f"✅ REVEAL stored: request={reveal.request_id[:8]}... "
          f"sub={reveal.submission_id[:8]}... miner={reveal.miner_hotkey[:8]}... "
          f"leads={len(lead_data_list)}/{len(reveal.leads)} revealed=True"
          + (f" (dropped {len(mismatched)} mismatched)" if mismatched else ""))

    _log_event(EventType.FULFILLMENT_REVEAL, reveal.miner_hotkey, {
        "request_id": reveal.request_id,
        "miner_hotkey": reveal.miner_hotkey,
        "reveal_timestamp": now.isoformat(),
        "mismatched_indices": [m["index"] for m in mismatched],
    })

    return {
        "status": "revealed",
        "num_leads": len(lead_data_list),
        "mismatched": mismatched,
    }


# ---------------------------------------------------------------
# GET /fulfillment/scoring  — validators fetch revealed leads for scoring
# ---------------------------------------------------------------
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

    supabase = _get_supabase()

    resp = supabase.table("fulfillment_requests") \
        .select("*") \
        .eq("status", "scoring") \
        .execute()

    scoring_count = len(resp.data or [])
    if scoring_count > 0:
        print(f"📋 /fulfillment/scoring: {scoring_count} request(s) in scoring status")

    already_scored_requests = set()
    if validator_hotkey:
        scored_resp = supabase.table("fulfillment_scores") \
            .select("request_id") \
            .eq("validator_hotkey", validator_hotkey) \
            .execute()
        already_scored_requests = {r["request_id"] for r in (scored_resp.data or [])}
        if already_scored_requests:
            print(f"   Validator {validator_hotkey[:8]}... already scored: {len(already_scored_requests)} request(s)")

    out = []
    for r in (resp.data or []):
        if r["request_id"] in already_scored_requests:
            print(f"   Skipping {r['request_id'][:8]}... (already scored by this validator)")
            continue

        subs_resp = supabase.table("fulfillment_submissions") \
            .select("*") \
            .eq("request_id", r["request_id"]) \
            .eq("revealed", True) \
            .execute()

        submissions = []
        for s in (subs_resp.data or []):
            # SAFETY: both `leads` and `lead_ids` MUST be sourced from the
            # same list (`lead_data`) so they stay index-aligned.  Previous
            # code sourced `lead_ids` from `lead_hashes` (the full committed
            # list) while `leads` came from `lead_data` (only matched entries
            # after /reveal dropped hash-mismatched ones).  When miners had
            # any partial-reveal hash mismatch, the two arrays had different
            # lengths and the validator's zip(lead_ids, results) mapped the
            # wrong lead_id onto each score — silently corrupting downstream
            # consensus and winner selection.  `lead_data` entries are
            # {"lead_id": ..., "data": ...} so both projections are safe.
            lead_data = s.get("lead_data") or []
            submissions.append({
                "submission_id": s["submission_id"],
                "miner_hotkey": s["miner_hotkey"],
                "leads": [entry.get("data", {}) for entry in lead_data],
                "lead_ids": [entry.get("lead_id", "") for entry in lead_data],
            })

        print(f"   Returning {r['request_id'][:8]}... with {len(submissions)} submission(s), "
              f"{sum(len(s['leads']) for s in submissions)} total leads")

        out.append({
            "request_id": r["request_id"],
            "icp": r.get("icp_details", {}),
            "status": r["status"],
            "submissions": submissions,
        })

    return {"requests": out}


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

    supabase = _get_supabase()

    req_resp = supabase.table("fulfillment_requests") \
        .select("status, successor_request_id") \
        .eq("request_id", request_id) \
        .execute()
    if not req_resp.data:
        raise HTTPException(404, detail="Request not found")

    req = req_resp.data[0]

    consensus_resp = supabase.table("fulfillment_score_consensus") \
        .select("*") \
        .eq("request_id", request_id) \
        .order("consensus_final_score", desc=True) \
        .execute()

    leads = []
    for row in (consensus_resp.data or []):
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
    """
    supabase = _get_supabase()

    resp = supabase.table("fulfillment_score_consensus") \
        .select("miner_hotkey, reward_pct, reward_expires_epoch") \
        .not_.is_("reward_pct", "null") \
        .gt("reward_expires_epoch", current_epoch) \
        .execute()

    per_miner: dict = {}
    for row in (resp.data or []):
        hk = row["miner_hotkey"]
        pct = float(row["reward_pct"])
        per_miner[hk] = per_miner.get(hk, 0.0) + pct

    return {"rewards": per_miner, "total_active_rows": len(resp.data or [])}


# ---------------------------------------------------------------
# GET /fulfillment/leaderboard  — top fulfillment miners by lifetime wins
# ---------------------------------------------------------------
@fulfillment_router.get("/leaderboard")
async def get_fulfillment_leaderboard(limit: int = 3):
    """Return top fulfillment miners ranked by all-time `is_winner` count.

    Used by the validator each weight-set cycle to identify the top-3
    miners eligible for the lifetime leaderboard emission bonus
    (LEADERBOARD_BONUS_SHARE in neurons/validator.py — split 2.5/1/0.5%
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
          "computed_at": "2026-04-30T03:14:00Z",
          "total_unique_winners": 12
        }
    """
    if limit < 1:
        limit = 1
    if limit > 100:
        limit = 100

    supabase = _get_supabase()

    # Fetch all winning consensus rows (typical scale: low thousands —
    # cheap.  If this ever balloons we move to a materialized view).
    resp = supabase.table("fulfillment_score_consensus") \
        .select("miner_hotkey, reward_pct") \
        .eq("is_winner", True) \
        .execute()

    # Banned hotkeys to exclude
    try:
        banned_resp = supabase.table("banned_hotkeys") \
            .select("hotkey") \
            .execute()
        banned_set = {r["hotkey"] for r in (banned_resp.data or [])}
    except Exception:
        banned_set = set()

    # Aggregate per miner
    per_miner: dict = {}
    for row in (resp.data or []):
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
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "total_unique_winners": len(per_miner),
    }


# ---------------------------------------------------------------
# POST /fulfillment/ban/{hotkey}  — validator requests a ban
# ---------------------------------------------------------------
@fulfillment_router.post("/ban/{hotkey}")
async def request_ban(hotkey: str, reason: str = "", validator_hotkey: str = "", request_id: str = ""):
    if not _enable_fulfillment():
        raise HTTPException(503, detail="Fulfillment system is not enabled")

    _log_event(EventType.FULFILLMENT_BAN, validator_hotkey or "admin", {
        "hotkey": hotkey,
        "reason": reason,
        "banned_by": validator_hotkey or "admin",
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
