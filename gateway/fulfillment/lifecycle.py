"""
Fulfillment lifecycle background task.

Manages request state transitions, consensus aggregation, reward expiry,
and request recycling.
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from uuid import uuid4
from dateutil.parser import isoparse as _isoparse

from gateway.fulfillment.config import (
    FULFILLMENT_LIFECYCLE_INTERVAL_SECONDS,
    FULFILLMENT_MIN_VALIDATORS,
    FULFILLMENT_CONSENSUS_TIMEOUT_MINUTES,
    FULFILLMENT_MAX_PARALLEL_REQUESTS,
    FULFILLMENT_MIN_REMAINING_WINDOW_MINUTES,
    L_EPOCHS,
    M_MINUTES,
    T_EPOCHS,
    T_SECONDS_OVERRIDE,
    Z_PERCENT,
    epochs_to_seconds,
)
from gateway.fulfillment.consensus import compute_fulfillment_consensus
from gateway.models.events import EventType

logger = logging.getLogger(__name__)


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


def _log_event(event_type: EventType, actor_hotkey: str, payload: dict) -> None:
    """Best-effort transparency log insert.  Populates the full audit-log
    shape (nonce, ts, payload_hash, build_id, signature, actor_hotkey) to
    satisfy the NOT NULL constraints that previously caused every
    FULFILLMENT_* insert to be silently dropped.  See the twin function in
    gateway/fulfillment/api.py for the full rationale."""
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


_ADVISORY_LOCK_KEY = int.from_bytes(
    __import__("hashlib").sha256(b"fulfillment_lifecycle").digest()[:4],
    "big",
) % (2**31)


def _try_advisory_lock(supabase) -> bool:
    """No-op. The lifecycle tick is already idempotent and race-safe at the
    per-request level (guarded UPDATE + orphan cleanup in _recycle_request),
    so a global lock is unnecessary.  It was actively harmful because
    pg_advisory_lock is SESSION-scoped and PostgREST uses a pooled
    connection: the acquire-lock RPC and release-lock RPC land on different
    pooled sessions, so the acquire would succeed on session A but the
    release would run on session B (a no-op there).  Session A would keep
    the lock indefinitely, blocking every subsequent tick until the
    gateway restarted.  This bug silently wedged the fulfillment lifecycle
    for hours at a time (observed: 5 client requests stuck in commit_closed
    for 16+ hours, zero recycles while the lock was orphaned).

    Kept the function signature so the call sites remain compatible and
    we don't have to restructure the tick code.  Always returns True so
    the tick proceeds.
    """
    return True


def _release_advisory_lock(supabase) -> None:
    """No-op companion to _try_advisory_lock.  Historical rationale above."""
    return None


async def fulfillment_lifecycle_task() -> None:
    """Background loop managing fulfillment request state transitions."""
    print("🔄 Fulfillment lifecycle task running (every 30s)")

    while True:
        try:
            await _lifecycle_tick()
        except asyncio.CancelledError:
            print("Fulfillment lifecycle task cancelled")
            break
        except Exception as e:
            print(f"❌ Fulfillment lifecycle error: {e}")
            import traceback
            traceback.print_exc()

        await asyncio.sleep(FULFILLMENT_LIFECYCLE_INTERVAL_SECONDS)


async def _lifecycle_tick() -> None:
    supabase = _get_supabase()

    if not _try_advisory_lock(supabase):
        logger.debug("Lifecycle tick skipped — another instance holds the lock")
        return

    try:
        await _lifecycle_tick_inner(supabase)
    finally:
        _release_advisory_lock(supabase)


async def _lifecycle_tick_inner(supabase) -> None:
    now = datetime.now(timezone.utc)
    now_iso = now.isoformat()

    # ────────────────────────────────────────────────────────────────
    # STEP 0: Promote 'pending' → 'open' to fill the miner-visible pool
    # ────────────────────────────────────────────────────────────────
    # Requests are created in 'pending' with NULL window timestamps.
    # We promote the oldest pending rows to 'open' whenever the open
    # pool has room, and ONLY THEN stamp window_start / window_end /
    # reveal_window_end based on the current wall clock.  This means:
    #
    #   * No request's commit timer ever ticks down while the request
    #     is invisible to miners (the pre-migration bug that caused
    #     the 6-over-the-cap requests to silently expire).
    #   * The 5-concurrent cap on the miner-facing endpoint is
    #     enforced at the data layer, not just at the query layer.
    #   * FIFO order: oldest `created_at` in 'pending' is promoted
    #     first, so clients' requests and recycled successors are
    #     served in the order they arrived.
    #
    # The UPDATE is guarded on `status='pending'` so concurrent ticks
    # can't double-promote the same row.  If two ticks race and both
    # pick the same pending row, only one UPDATE lands; the other is
    # a no-op.  Idempotent and safe.
    try:
        # Count "miner-visible" open requests, not just raw open.  The
        # miner-facing /fulfillment/requests/active endpoint hides any
        # open request whose window_end is within
        # FULFILLMENT_MIN_REMAINING_WINDOW_MINUTES of now (a safety
        # filter that prevents a miner from being handed a request
        # they can't realistically commit to in time).  If we count
        # only raw open here, a request in its "last 15 min of soon-
        # to-expire" tail keeps the slot reserved without being
        # visible to miners, which produced Mase's observed intermittent
        # "/active sometimes returns fewer than 5 active requests"
        # behaviour.  Using the visibility-aware cutoff keeps the
        # pool of miner-visible requests at MAX_PARALLEL_REQUESTS at
        # all times (or as close as pending-queue depth allows),
        # while still letting the tail-end request naturally progress
        # to commit_closed on its normal timer.
        visibility_cutoff_iso = (
            now + timedelta(minutes=FULFILLMENT_MIN_REMAINING_WINDOW_MINUTES)
        ).isoformat()
        # Count miner-visible 'open' OR 'continued_open' requests when
        # computing slots — both are surfaced to miners by /active and
        # both consume a slot in the FULFILLMENT_MAX_PARALLEL_REQUESTS
        # cap.  A continued_open is functionally identical to open for
        # commit windowing; only the label differs.
        visible_open_resp = supabase.table("fulfillment_requests") \
            .select("request_id", count="exact") \
            .in_("status", ["open", "continued_open"]) \
            .gt("window_end", visibility_cutoff_iso) \
            .execute()
        visible_open_count = visible_open_resp.count or 0
        slots = FULFILLMENT_MAX_PARALLEL_REQUESTS - visible_open_count
        if slots > 0:
            pending = supabase.table("fulfillment_requests") \
                .select("request_id") \
                .eq("status", "pending") \
                .order("created_at", desc=False) \
                .limit(slots) \
                .execute()
            if pending.data:
                if T_SECONDS_OVERRIDE > 0:
                    commit_seconds = T_SECONDS_OVERRIDE
                else:
                    commit_seconds = epochs_to_seconds(T_EPOCHS, _get_tempo(supabase))
                w_end = now + timedelta(seconds=commit_seconds)
                r_end = w_end + timedelta(minutes=M_MINUTES)
                for p in pending.data:
                    rid = p["request_id"]
                    # Decide whether this pending row should be promoted
                    # to 'open' (fresh) or 'continued_open' (chain
                    # continuation) by looking at the predecessor's
                    # status.  Predecessors with status='partially_fulfilled'
                    # signal a chain in flight — successors get the
                    # 'continued_open' label so miners (and dashboards)
                    # know prior held leads exist and rewards won't flow
                    # for this cycle alone.
                    target_status = "open"
                    try:
                        pred = supabase.table("fulfillment_requests") \
                            .select("status") \
                            .eq("successor_request_id", rid) \
                            .limit(1) \
                            .execute()
                        if pred.data and pred.data[0].get("status") == "partially_fulfilled":
                            target_status = "continued_open"
                    except Exception:
                        # Predecessor lookup is a hint only — falling
                        # back to plain 'open' is safe (commit window
                        # behavior is identical between the two).
                        pass

                    try:
                        supabase.table("fulfillment_requests").update({
                            "status": target_status,
                            "window_start": now_iso,
                            "window_end": w_end.isoformat(),
                            "reveal_window_end": r_end.isoformat(),
                        }).eq("request_id", rid) \
                          .eq("status", "pending") \
                          .execute()
                        print(f"   ⬆️  {rid[:8]}... promoted pending → {target_status} "
                              f"(window_end {w_end.strftime('%H:%M:%S')}Z)")
                    except Exception as e:
                        print(f"   ⚠️  Promote failed for {rid[:8]}: {e}")
    except Exception as e:
        print(f"❌ Promotion step error: {e}")

    # Debug: show all non-terminal request statuses
    all_req = supabase.table("fulfillment_requests") \
        .select("request_id, status, window_end, reveal_window_end") \
        .in_("status", ["pending", "open", "continued_open", "commit_closed", "scoring"]) \
        .execute()
    if all_req.data:
        print(f"🔄 Lifecycle tick @ {now_iso[:19]}Z — {len(all_req.data)} active request(s):")
        for ar in (all_req.data or []):
            we = ar.get("window_end") or "?"
            re_ = ar.get("reveal_window_end") or "?"
            print(f"   {ar['request_id'][:8]}... status={ar['status']} "
                  f"window_end={str(we)[:19]} "
                  f"reveal_end={str(re_)[:19]}")

    # Step 1: open -> commit_closed (past window_end).  Both 'open' and
    # 'continued_open' transition the same way — only the label differs.
    open_past_window = supabase.table("fulfillment_requests") \
        .select("request_id") \
        .in_("status", ["open", "continued_open"]) \
        .lt("window_end", now_iso) \
        .execute()
    if open_past_window.data:
        print(f"Lifecycle: {len(open_past_window.data)} open/continued_open request(s) past window_end")
    for r in (open_past_window.data or []):
        try:
            # Direct UPDATE rather than the legacy fulfillment_close_window
            # RPC: that RPC was hardcoded to only accept source status
            # 'open' and silently no-op'd on 'continued_open' rows, so
            # every chain that recycled into a partially_fulfilled
            # successor would freeze at its window_end with the status
            # column still showing 'continued_open' indefinitely
            # (observed Apr 27 2026 on Browser chain head a0c71809; one
            # successor stuck for 60 minutes before detection).  The
            # equivalent UPDATE here accepts both source states and is
            # also idempotent — re-running on an already-closed row
            # affects 0 rows because of the `.in_("status", ...)`
            # source-state filter.
            supabase.table("fulfillment_requests").update({
                "status": "commit_closed",
            }).eq("request_id", r["request_id"]) \
              .in_("status", ["open", "continued_open"]) \
              .execute()
            # Verify the transition actually happened
            verify = supabase.table("fulfillment_requests") \
                .select("status") \
                .eq("request_id", r["request_id"]) \
                .execute()
            actual_status = verify.data[0]["status"] if verify.data else "?"
            print(f"   {r['request_id'][:8]}... -> commit_closed (verified: {actual_status})")
        except Exception as e:
            print(f"   Error closing {r['request_id'][:8]}...: {e}")

    # Step 2: commit_closed -> scoring or recycled (past reveal_window_end)
    closed_past_reveal = supabase.table("fulfillment_requests") \
        .select("request_id, icp_details, num_leads, reveal_window_end, internal_label, company, required_attributes") \
        .eq("status", "commit_closed") \
        .lt("reveal_window_end", now_iso) \
        .execute()

    if closed_past_reveal.data:
        print(f"Lifecycle Step 2: {len(closed_past_reveal.data)} commit_closed request(s) past reveal_window_end")

    for r in (closed_past_reveal.data or []):
        rid = r["request_id"]
        print(f"   Checking {rid[:8]}... (reveal_window_end={r.get('reveal_window_end', '?')})")

        all_subs = supabase.table("fulfillment_submissions") \
            .select("submission_id, revealed, miner_hotkey") \
            .eq("request_id", rid) \
            .execute()
        print(f"   Total submissions for {rid[:8]}: {len(all_subs.data or [])}")
        for s in (all_subs.data or []):
            print(f"     sub={s['submission_id'][:8]}... miner={s['miner_hotkey'][:8]}... revealed={s['revealed']}")

        reveals = supabase.table("fulfillment_submissions") \
            .select("submission_id") \
            .eq("request_id", rid) \
            .eq("revealed", True) \
            .execute()
        print(f"   Revealed submissions: {len(reveals.data or [])}")

        if reveals.data:
            try:
                # Direct UPDATE rather than the legacy fulfillment_close_window
                # RPC, mirroring the Step 1 fix above.  The RPC's accepted
                # source-state list is opaque (defined in Supabase, not in
                # this repo) and we already saw it silently no-op for
                # 'continued_open' source rows.  A direct UPDATE with the
                # source-state guard is both transparent and idempotent.
                supabase.table("fulfillment_requests").update({
                    "status": "scoring",
                }).eq("request_id", rid) \
                  .eq("status", "commit_closed") \
                  .execute()
                print(f"   ✅ {rid[:8]}... -> scoring ({len(reveals.data)} reveal(s))")
            except Exception as e:
                print(f"   ❌ Error transitioning {rid[:8]}... to scoring: {e}")
        else:
            print(f"   ⚠️  No reveals for {rid[:8]}... — recycling")
            chain_state = _chain_held_state_for_recycle(
                supabase, rid, r.get("num_leads") or 0,
            )
            _recycle_request(
                supabase, r, now, now_iso,
                terminal_status=chain_state["recycle_status"] or "recycled",
                reason="no_reveals",
                successor_status_target=chain_state["successor_status_target"],
                successor_num_leads=chain_state["successor_num_leads"],
                held_companies=chain_state["held_companies"],
            )

    # Step 3: consensus aggregation for scoring requests + recently-transitioned
    # partially_fulfilled requests (within the consensus timeout window).
    #
    # Why include partially_fulfilled: FULFILLMENT_MIN_VALIDATORS defaults to 1,
    # so the first validator's submission triggers compute_fulfillment_consensus()
    # below, which decides chain top-K and transitions the request to a terminal
    # status (fulfilled / partially_fulfilled / recycled).  Without including
    # partially_fulfilled here, late-arriving validator scores would land in
    # fulfillment_scores but never make it into fulfillment_score_consensus —
    # they'd be silently dropped, and any winners those validators saw would
    # not be reflected in the chain-held set.
    #
    # Safety:
    #   * fulfillment_upsert_consensus is idempotent (upsert keyed on
    #     (request_id, submission_id, lead_id)) — re-running on the same data
    #     is a no-op.
    #   * _resolve_chain_topk re-resolves chain-held flags across the entire
    #     chain on every call — if late scores produce new winners, they
    #     correctly displace lower-scoring held leads.
    #   * Time-bounded to FULFILLMENT_CONSENSUS_TIMEOUT_MINUTES past
    #     reveal_window_end so we don't re-aggregate ancient requests forever.
    #   * fulfilled and recycled are NOT included — fulfilled would risk
    #     double-paying via a second _finalize_chain_rewards; recycled is a
    #     separate (and rarer) failure mode handled in a follow-up.
    # The time-bound on `reveal_window_end` exists to prevent infinite
    # re-aggregation of `partially_fulfilled` requests when late validator
    # scores trickle in.  It must NOT apply to `scoring` — that status
    # means the request is still waiting for consensus, and excluding it
    # past the 3h cutoff strands it forever (the loop never sees it again,
    # so the timeout-based forced consensus inside the loop body never
    # fires).  Observed today on 3 stuck requests (5854e3b9, caf0923d,
    # fc1e0a3f) all status=scoring and 11-14h past reveal_end.
    rerun_window_start = (
        now - timedelta(minutes=FULFILLMENT_CONSENSUS_TIMEOUT_MINUTES)
    ).isoformat()
    scoring_requests = supabase.table("fulfillment_requests") \
        .select("request_id, reveal_window_end, icp_details, num_leads, internal_label, company, required_attributes, status, successor_request_id") \
        .in_("status", ["scoring", "partially_fulfilled"]) \
        .or_(
            f"status.eq.scoring,reveal_window_end.gte.{rerun_window_start}"
        ) \
        .execute()

    if scoring_requests.data:
        print(f"Lifecycle Step 3: {len(scoring_requests.data)} request(s) in scoring status")

    for r in (scoring_requests.data or []):
        rid = r["request_id"]
        try:
            validator_count_resp = supabase.table("fulfillment_scores") \
                .select("validator_hotkey") \
                .eq("request_id", rid) \
                .execute()
            unique_validators = {s["validator_hotkey"] for s in (validator_count_resp.data or [])}

            reveal_end = _isoparse(r["reveal_window_end"])
            timeout = reveal_end + timedelta(minutes=FULFILLMENT_CONSENSUS_TIMEOUT_MINUTES)

            if len(unique_validators) < FULFILLMENT_MIN_VALIDATORS and now < timeout:
                mins_left = (timeout - now).total_seconds() / 60
                print(f"   {rid[:8]}... waiting for validators: {len(unique_validators)}/{FULFILLMENT_MIN_VALIDATORS} ({mins_left:.1f}min until timeout)")
                continue

            # ─── Guard: don't run consensus while revealed submissions are still un-scored ─────
            # Observed 2026-06-01 on request fbeed690-edbb-...: 6 miners had
            # revealed submissions (3 with 41 leads each + 3 with 9 leads each),
            # but only 2 of 6 were scored before consensus fired. The other 4
            # miners' work was discarded when the request transitioned to
            # `recycled`. Miners (5GpzK4Rm, 5H17R1Za, 5HGP4yPa, 5DcJwmjL)
            # rightly complained that their submissions "weren't scored".
            #
            # Root cause: this loop's gate counts unique VALIDATORS (currently
            # MIN=1). The moment one validator submits a score for ONE miner's
            # submission, the gate opens and consensus runs — even if other
            # miners' revealed submissions are still queued for scoring.
            #
            # Fix: count distinct revealed submissions for this request and
            # distinct submission_ids in fulfillment_scores. If revealed >
            # scored AND we're inside the consensus grace window, defer.
            # Once we hit the timeout, fall through to consensus anyway so a
            # dead validator can't wedge the request forever.
            revealed_subs_resp = supabase.table("fulfillment_submissions") \
                .select("submission_id") \
                .eq("request_id", rid) \
                .eq("revealed", True) \
                .execute()
            revealed_subs = {s["submission_id"] for s in (revealed_subs_resp.data or [])}
            scored_subs_resp = supabase.table("fulfillment_scores") \
                .select("submission_id") \
                .eq("request_id", rid) \
                .execute()
            scored_subs = {s.get("submission_id") for s in (scored_subs_resp.data or []) if s.get("submission_id")}
            unscored_subs = revealed_subs - scored_subs
            if unscored_subs and now < timeout:
                mins_left = (timeout - now).total_seconds() / 60
                print(
                    f"   {rid[:8]}... deferring consensus: "
                    f"{len(unscored_subs)}/{len(revealed_subs)} revealed submissions "
                    f"still un-scored ({mins_left:.1f}min until forced consensus)"
                )
                continue

            if len(unique_validators) == 0 and now >= timeout:
                print(
                    f"   ⚠️  {rid[:8]}... has 0 validators after timeout — "
                    f"expiring and recycling"
                )
                chain_state = _chain_held_state_for_recycle(
                    supabase, rid, r.get("num_leads") or 0,
                )
                _recycle_request(
                    supabase, r, now, now_iso,
                    # If prior held leads exist, mark the predecessor
                    # partially_fulfilled rather than expired so miners
                    # see the chain is still in flight; otherwise keep
                    # the legacy 'expired' label.
                    terminal_status=chain_state["recycle_status"] or "expired",
                    reason="no_validators_timeout",
                    successor_status_target=chain_state["successor_status_target"],
                    successor_num_leads=chain_state["successor_num_leads"],
                    held_companies=chain_state["held_companies"],
                )
                continue

            if len(unique_validators) < FULFILLMENT_MIN_VALIDATORS:
                print(
                    f"   ⚠️  {rid[:8]}... consensus timeout: "
                    f"{len(unique_validators)}/{FULFILLMENT_MIN_VALIDATORS} validators — proceeding"
                )

            # Skip re-aggregation when no new validator scores have arrived
            # since the last consensus pass.  Heavy work (compute_consensus +
            # chain top-K + DB writes) runs synchronously and blocks the
            # asyncio event loop — leaving it out when nothing changed keeps
            # the gateway's /health endpoint responsive between ticks.
            #
            # Only fires for already-partially_fulfilled rows; for fresh
            # `scoring` rows the consensus pass MUST run (no prior pass to
            # compare against).
            if r.get("status") == "partially_fulfilled":
                latest_consensus_resp = supabase.table("fulfillment_score_consensus") \
                    .select("computed_at") \
                    .eq("request_id", rid) \
                    .order("computed_at", desc=True) \
                    .limit(1).execute()
                latest_score_resp = supabase.table("fulfillment_scores") \
                    .select("scored_at") \
                    .eq("request_id", rid) \
                    .order("scored_at", desc=True) \
                    .limit(1).execute()
                if latest_consensus_resp.data and latest_score_resp.data:
                    last_consensus_at = latest_consensus_resp.data[0].get("computed_at") or ""
                    last_scored_at = latest_score_resp.data[0].get("scored_at") or ""
                    if last_scored_at <= last_consensus_at:
                        # No new scores since the last aggregation — skip.
                        # This is the common case for re-aggregation passes
                        # on stable partially_fulfilled chains.
                        continue

            consensus_results = await compute_fulfillment_consensus(rid)
            if not consensus_results:
                print(
                    f"   ⚠️  {rid[:8]}... produced empty consensus — "
                    f"expiring and recycling"
                )
                chain_state = _chain_held_state_for_recycle(
                    supabase, rid, r.get("num_leads") or 0,
                )
                _recycle_request(
                    supabase, r, now, now_iso,
                    terminal_status=chain_state["recycle_status"] or "expired",
                    reason="empty_consensus",
                    successor_status_target=chain_state["successor_status_target"],
                    successor_num_leads=chain_state["successor_num_leads"],
                    held_companies=chain_state["held_companies"],
                )
                continue

            supabase.rpc("fulfillment_upsert_consensus", {
                "p_consensus": consensus_results,
            }).execute()

            # The upsert RPC's SQL signature doesn't know about the
            # `intent_signal_mapping` column (added in a later migration),
            # so it silently drops that field on insert/update.  Same class
            # of bug as the `intent_signals_detail` patch in api.py::submit_scores.
            # Backfill it directly here so the column always reflects the
            # most-recent validator's per-signal breakdown and the downstream
            # Perplexity intent_details step has something to ground against.
            for cr in consensus_results:
                mapping = cr.get("intent_signal_mapping") or []
                if not mapping:
                    continue
                try:
                    supabase.table("fulfillment_score_consensus").update({
                        "intent_signal_mapping": mapping,
                    }).eq("request_id", rid) \
                      .eq("submission_id", cr["submission_id"]) \
                      .eq("lead_id", cr["lead_id"]) \
                      .execute()
                except Exception as e:
                    print(f"   ⚠️  Failed to patch intent_signal_mapping for "
                          f"lead {str(cr.get('lead_id',''))[:8]}: {e}")

            num_requested = r.get("num_leads") or (r.get("icp_details", {}) or {}).get("num_leads") or 0

            # Chain-aware top-K resolution.
            #   * Combines this cycle's qualifying candidates with any leads
            #     already held in earlier generations of this chain
            #     (is_chain_held=TRUE on predecessor consensus rows).
            #   * Cross-cycle dedup on company so the same business never
            #     occupies two slots, even across recycle generations.
            #   * Updates the is_chain_held flag for the entire chain
            #     (sets TRUE on the new top-K, clears FALSE on displaced
            #     leads that lost their slot to a higher-scoring entry).
            #   * Does NOT distribute rewards yet; rewards only flow when
            #     the chain reaches its full quota (see fulfilled branch
            #     below).  Held leads earn nothing until then — by design.
            chain = await _resolve_chain_topk(rid, consensus_results, num_requested)
            chain_target = chain["chain_target"]
            topk = chain["topk"]
            topk_lead_ids = chain["topk_lead_ids"]
            topk_companies = chain["topk_companies"]
            displaced_count = chain["displaced_count"]

            held_count = len(topk)
            print(
                f"   📊 {rid[:8]}... chain top-K: held {held_count}/{chain_target} "
                f"(this cycle contributed {len(chain['this_cycle_winners'])}, "
                f"displaced {displaced_count} from prior generations)"
            )

            # Track whether this is a re-aggregation pass on an already-
            # partially_fulfilled request (Step 3 now picks those up so late
            # validator scores can be folded in).  Used below to skip the
            # recycle branch when nothing materially changed — otherwise the
            # existing recycle would insert + claim-loss + delete an orphan
            # successor every tick.
            was_partially_fulfilled = r.get("status") == "partially_fulfilled"

            if held_count >= chain_target:
                # ────────────────────────────────────────────────────────
                # FULFILLED: chain reached full quota.  Distribute rewards
                # NOW for the entire top-K (including held leads from
                # earlier generations — their request_id stays in their
                # original generation, but reward_pct/expires_epoch is
                # written via calculate_lead_rewards keyed on
                # (request_id, submission_id, lead_id)).
                #
                # Re-aggregation path: if was_partially_fulfilled, the
                # successor was already created during the original
                # partial-fulfill recycle.  We mark the predecessor as
                # fulfilled and distribute rewards normally; the existing
                # successor will naturally terminate when its own scoring
                # cycle runs (chain top-K sees the chain quota already met
                # via the predecessor's held leads).
                # ────────────────────────────────────────────────────────
                winner_lead_ids = await _finalize_chain_rewards(
                    rid, topk, chain["tied_groups"],
                )
                supabase.table("fulfillment_requests").update({
                    "status": "fulfilled",
                }).eq("request_id", rid).execute()
                print(
                    f"   ✅ {rid[:8]}... -> fulfilled "
                    f"({len(winner_lead_ids)}/{chain_target} winners; "
                    f"chain reached quota)"
                )

                # Orphan-successor cleanup. dff0407d's design documented
                # that the successor "naturally terminates when its own
                # scoring cycle runs (chain top-K sees the chain quota
                # already met)" — but that only fires if the successor
                # has received submissions. An empty successor sits in
                # continued_open until its own window expires (~70 min of
                # phantom open-chain visibility, during which miners
                # waste work submitting to it). Observed 2026-05-29: 5
                # Stan chain head 300c9867 stuck in continued_open after
                # predecessor 22e05fd0 flipped to fulfilled via late-
                # score re-aggregation 2 minutes after the original
                # partial-fulfill recycle.
                #
                # Proactively recycle the orphan successor here if it has
                # 0 submissions. Active successors (any submissions) are
                # left alone — Pranav's chain-top-K reconciliation in
                # _resolve_chain_topk handles them correctly on their
                # next scoring cycle. The status guard
                # status=continued_open ensures we never clobber a
                # successor that has advanced.
                if was_partially_fulfilled:
                    succ_id = r.get("successor_request_id")
                    if succ_id:
                        try:
                            count_resp = supabase.table(
                                "fulfillment_submissions"
                            ).select(
                                "submission_id", count="exact", head=True,
                            ).eq("request_id", succ_id).execute()
                            if (count_resp.count or 0) == 0:
                                upd = supabase.table(
                                    "fulfillment_requests"
                                ).update({
                                    "status": "recycled",
                                }).eq("request_id", succ_id) \
                                  .eq("status", "continued_open") \
                                  .execute()
                                if upd.data:
                                    print(
                                        f"   🧹 cancelled empty orphan "
                                        f"successor {str(succ_id)[:8]}... "
                                        f"(parent fulfilled via "
                                        f"re-aggregation, 0 submissions)"
                                    )
                        except Exception as e:
                            print(
                                f"   ⚠️ orphan-successor cleanup failed "
                                f"for {str(succ_id)[:8]}: {e}"
                            )

                ranked = sorted(
                    consensus_results,
                    key=lambda x: x.get("consensus_final_score", 0),
                    reverse=True,
                )
                print(f"\n{'='*60}")
                print(f"🏆 FULFILLMENT RESULTS — Request {rid[:8]}...")
                print(f"   {len(ranked)} leads scored this cycle; chain target={chain_target}")
                print(f"{'='*60}")
                for i, cr in enumerate(ranked, 1):
                    miner = cr.get("miner_hotkey", "?")[:16]
                    score = cr.get("consensus_final_score", 0)
                    t2 = "✅" if cr.get("consensus_tier2_passed") else "❌"
                    is_winner = cr.get("lead_id") in winner_lead_ids
                    winner = "👑" if is_winner else "  "
                    lid = cr.get("lead_id", "?")[:8]
                    print(f"   {winner} #{i}: score={score:.1f} tier2={t2} miner={miner}... lead={lid}...")
                print(f"\n   Winners: {len(winner_lead_ids)}/{chain_target} leads")
                print(f"{'='*60}\n")
            elif held_count > 0:
                if was_partially_fulfilled:
                    # ────────────────────────────────────────────────────
                    # RE-AGGREGATION on an already-partially_fulfilled
                    # request: the successor was created the first time we
                    # transitioned here.  Re-aggregation may have updated
                    # consensus rows and is_chain_held flags (handled by
                    # _resolve_chain_topk above), but we MUST NOT recycle
                    # again — that would orphan-insert + cleanup-delete a
                    # successor every tick.  Leave status as
                    # partially_fulfilled and let the existing successor
                    # continue its own lifecycle.
                    # ────────────────────────────────────────────────────
                    print(
                        f"   🔄 {rid[:8]}... re-aggregated "
                        f"({held_count}/{chain_target} held, "
                        f"still partially_fulfilled; successor unchanged)"
                    )
                else:
                    # ────────────────────────────────────────────────────
                    # PARTIALLY_FULFILLED: chain produced some held leads
                    # but didn't hit the quota.  Successor inherits the
                    # in-flight held set (held companies become
                    # excluded_companies) and asks miners only for the
                    # remaining quota.  No rewards distributed yet; held
                    # leads sit in DB with is_chain_held=TRUE and earn $0
                    # until the chain eventually reaches `fulfilled`.
                    # ────────────────────────────────────────────────────
                    remaining = chain_target - held_count
                    print(
                        f"   📈 {rid[:8]}... -> partially_fulfilled "
                        f"({held_count}/{chain_target} held, "
                        f"recycling for {remaining} more)"
                    )
                    _recycle_request(
                        supabase, r, now, now_iso,
                        terminal_status="partially_fulfilled",
                        successor_status_target="continued_open",
                        successor_num_leads=remaining,
                        held_companies=topk_companies,
                        reason=(
                            f"chain_partial_{held_count}_of_{chain_target}"
                        ),
                    )
            else:
                if was_partially_fulfilled:
                    # ────────────────────────────────────────────────────
                    # RE-AGGREGATION wiped out all held leads (very rare
                    # edge case — would require late scores that displace
                    # every prior winner).  Do NOT re-recycle as
                    # `recycled` — that would conflict with the existing
                    # `partially_fulfilled` successor and orphan-cycle.
                    # Leave status as-is and let the existing successor
                    # take over the chain's remaining quota.
                    # ────────────────────────────────────────────────────
                    print(
                        f"   🔄 {rid[:8]}... re-aggregated to 0 held "
                        f"(was partially_fulfilled; leaving status, "
                        f"existing successor continues)"
                    )
                else:
                    # ────────────────────────────────────────────────────
                    # RECYCLED: chain produced ZERO held leads this cycle
                    # and has no held leads from prior generations either
                    # (or they all just got knocked out).  Treat this like
                    # the legacy "no useful work" recycle — successor is a
                    # fresh `open` request asking for the full quota again.
                    # ────────────────────────────────────────────────────
                    print(
                        f"   ♻️  {rid[:8]}... -> recycled "
                        f"(0 held, target {chain_target}) — fresh start"
                    )
                    _recycle_request(
                        supabase, r, now, now_iso,
                        terminal_status="recycled",
                        successor_status_target="open",
                        successor_num_leads=chain_target,
                        held_companies=[],
                        reason=f"empty_chain_topk_target_{chain_target}",
                    )

        except Exception as e:
            print(f"   ❌ Error in consensus for {rid[:8]}...: {e}")
            import traceback
            traceback.print_exc()

    # Step 4: reward expiry
    try:
        await _expire_rewards(supabase)
    except Exception as e:
        print(f"❌ Reward expiry error: {e}")


def _chain_held_state_for_recycle(supabase, request_id: str, current_num_leads: int) -> dict:
    """Lightweight chain-state lookup for the non-consensus recycle paths
    (no_reveals, no_validators_timeout, empty_consensus).

    These paths bail out BEFORE consensus is computed, so they don't run
    through ``_resolve_chain_topk``.  But the chain may already have held
    leads from earlier generations that must propagate through this
    recycle so they're not lost.

    Returns a dict matching the keyword args ``_recycle_request`` accepts:
      * ``successor_status_target``: ``"continued_open"`` if any prior
        held leads exist, else ``"open"``.
      * ``successor_num_leads``: ``chain_target − len(held)`` if held>0,
        else the chain's full target.
      * ``held_companies``: list of (company-normalized) names of the
        prior held leads, used to seed the successor's
        excluded_companies.
      * ``recycle_status``: ``"partially_fulfilled"`` when held>0
        (preferring this label for the predecessor so miners and
        dashboards see "chain in progress, awaiting more leads"),
        else falls through to the caller's chosen terminal_status
        (``"recycled"`` / ``"expired"``).
    """
    prior_held = _load_chain_held_winners(supabase, request_id)
    if not prior_held:
        return {
            "successor_status_target": "open",
            "successor_num_leads": _chain_target_num_leads(supabase, request_id, current_num_leads),
            "held_companies": [],
            "recycle_status": None,  # caller's terminal_status stands
        }

    chain_target = _chain_target_num_leads(supabase, request_id, current_num_leads)

    # Hydrate held leads' companies from fulfillment_submissions.
    needed_subs = {r["submission_id"] for r in prior_held}
    sub_lead_data: dict = {}
    if needed_subs:
        sub_resp = supabase.table("fulfillment_submissions") \
            .select("submission_id, lead_data") \
            .in_("submission_id", list(needed_subs)) \
            .execute()
        for row in (sub_resp.data or []):
            ld_list = row.get("lead_data") or []
            lookup = {ld.get("lead_id"): ld.get("data", {}) for ld in ld_list}
            sub_lead_data[row["submission_id"]] = lookup
    held_companies: list = []
    for r in prior_held:
        info = sub_lead_data.get(r["submission_id"], {}).get(r["lead_id"])
        if info:
            company = _normalize_company(info.get("business", ""))
            if company:
                held_companies.append(company)

    return {
        "successor_status_target": "continued_open",
        "successor_num_leads": max(0, chain_target - len(prior_held)),
        "held_companies": held_companies,
        "recycle_status": "partially_fulfilled",
    }


def _walk_chain_predecessors(supabase, request_id: str) -> list:
    """Return the list of all request_ids upstream of ``request_id`` in the
    recycle chain (oldest first), EXCLUDING ``request_id`` itself.

    The chain is linked via ``successor_request_id``: each predecessor row
    has its successor_request_id field set to the next generation's id.
    Walks backwards from the given request, finding the row whose
    ``successor_request_id`` equals the current id, and repeats until the
    chain root (no predecessor pointing at us).  Bounded to 1000 generations
    as a safety against pathological loops in malformed data — the prior
    cap of 50 was too tight (Daniel iMove 10 ran 98 generations and the
    walk silently stopped 48 generations short of the true root, causing
    ``_chain_target_num_leads`` to return the wrong target → chain
    incorrectly marked ``fulfilled`` with partial delivery; also caused
    ``_load_chain_held_winners`` to lose prior-generation held leads from
    the chain's view).  1000 is well above any realistic chain length while
    still guarding against runaway loops.
    """
    chain: list = []
    cur = request_id
    for _ in range(1000):
        pred = supabase.table("fulfillment_requests") \
            .select("request_id") \
            .eq("successor_request_id", cur) \
            .limit(1) \
            .execute()
        if not pred.data:
            break
        cur = pred.data[0]["request_id"]
        chain.append(cur)
    chain.reverse()  # oldest → newest
    return chain


def _load_sibling_chain_held_companies(supabase, request_id: str) -> set:
    """Cross-chain dedup at _resolve_chain_topk time.

    Returns the set of normalized company names currently held
    (``is_chain_held=TRUE``) on any OTHER active chain for the same
    ``client_company`` as ``request_id``.  "Other chain" means any
    request not in this request's own chain (this request + predecessors
    via ``successor_request_id``).

    The original ``excluded_companies`` snapshot built at create_request /
    recycle time can't see helds that landed on sibling chains AFTER the
    snapshot was taken.  This function reads that state live at
    consensus time so the resolver can drop sibling-held companies
    from the top-K candidate pool — preventing the same company from
    being held & delivered twice to the same buyer across concurrent
    requests.
    """
    # 1) This request's client_company
    req_resp = supabase.table("fulfillment_requests") \
        .select("company") \
        .eq("request_id", request_id) \
        .limit(1) \
        .execute()
    if not req_resp.data:
        return set()
    client_company = (req_resp.data[0].get("company") or "").strip()
    if not client_company:
        return set()

    # 2) This chain's request_ids (predecessors include the chain we're
    #    in; current request_id excluded since its own helds are pulled
    #    via _load_chain_held_winners + the current consensus_results).
    own_chain_ids = set(_walk_chain_predecessors(supabase, request_id))
    own_chain_ids.add(request_id)

    # 3) All other active requests for the same client.  Active =
    #    statuses where helds could still be in flight; closed/recycled/
    #    expired requests' held flags are stale and shouldn't block.
    active_statuses = [
        "open", "continued_open", "pending", "commit_closed",
        "scoring", "partially_fulfilled",
    ]
    sibling_resp = supabase.table("fulfillment_requests") \
        .select("request_id") \
        .eq("company", client_company) \
        .in_("status", active_statuses) \
        .execute()
    sibling_ids = [
        r["request_id"] for r in (sibling_resp.data or [])
        if r["request_id"] not in own_chain_ids
    ]
    if not sibling_ids:
        return set()

    # 4) Held consensus rows on the sibling requests.
    held_resp = supabase.table("fulfillment_score_consensus") \
        .select("submission_id, lead_id") \
        .in_("request_id", sibling_ids) \
        .eq("is_chain_held", True) \
        .execute()
    held_rows = held_resp.data or []
    if not held_rows:
        return set()

    # 5) Hydrate company names from submissions.  Same lookup pattern
    #    as _resolve_chain_topk's hydration step.
    sub_ids = list({r["submission_id"] for r in held_rows})
    sub_resp = supabase.table("fulfillment_submissions") \
        .select("submission_id, lead_data") \
        .in_("submission_id", sub_ids) \
        .execute()
    sub_index: dict = {}
    for sub in (sub_resp.data or []):
        for lead in (sub.get("lead_data") or []):
            sub_index[(sub["submission_id"], lead.get("lead_id"))] = (
                lead.get("data") or {}
            ).get("business", "")

    out: set = set()
    for r in held_rows:
        biz = sub_index.get((r["submission_id"], r["lead_id"]), "")
        if biz:
            norm = _normalize_company(biz)
            if norm:
                out.add(norm)
    return out


def _load_chain_held_winners(supabase, request_id: str) -> list:
    """Load all consensus rows currently flagged is_chain_held=TRUE across
    every predecessor of ``request_id`` (not including ``request_id`` itself).

    Returned shape matches the consensus_results dicts produced by
    ``compute_fulfillment_consensus``: one entry per (submission, lead),
    plus a ``request_id`` field so the caller can write back the right row
    when updating ``is_chain_held``.

    Empty list when:
      * the request is the chain root (no predecessors), OR
      * predecessors exist but none flipped ``is_chain_held`` (because the
        predecessor was a clean ``recycled`` with no useful work, or this
        is a fresh chain pre-migration).
    """
    chain_predecessors = _walk_chain_predecessors(supabase, request_id)
    if not chain_predecessors:
        return []

    held = supabase.table("fulfillment_score_consensus") \
        .select(
            "consensus_id, request_id, submission_id, lead_id, miner_hotkey, "
            "consensus_final_score, consensus_intent_signal_final, "
            "consensus_company_verified, consensus_person_verified, "
            "consensus_email_verified, consensus_decision_maker, "
            "consensus_icp_fit, consensus_rep_score, consensus_tier2_passed, "
            "any_fabricated, intent_details, intent_signal_mapping, "
            "num_validators"
        ) \
        .in_("request_id", chain_predecessors) \
        .eq("is_chain_held", True) \
        .execute()
    return list(held.data or [])


def _chain_target_num_leads(supabase, request_id: str, current_num_leads: int) -> int:
    """The chain's full quota (the K in 'top-K').

    Walks back to the chain root (the oldest predecessor with no row
    pointing at it) and reads its ``num_leads``, which represents the
    client's original request size.  This stays invariant across all
    successors in the chain.

    Falls back to ``current_num_leads`` if the chain has no predecessors
    (i.e., this IS the root).
    """
    predecessors = _walk_chain_predecessors(supabase, request_id)
    if not predecessors:
        return current_num_leads
    root_id = predecessors[0]
    try:
        root = supabase.table("fulfillment_requests") \
            .select("num_leads") \
            .eq("request_id", root_id) \
            .limit(1) \
            .execute()
        if root.data and root.data[0].get("num_leads"):
            return int(root.data[0]["num_leads"])
    except Exception as e:
        print(f"   ⚠️  Could not read chain root num_leads from {root_id[:8]}: {e}")
    return current_num_leads


async def _resolve_chain_topk(
    request_id: str,
    consensus_results: list,
    current_request_num_leads: int,
) -> dict:
    """Chain-aware top-K resolution.

    Combines the new cycle's qualifying candidates (final_score > 0) with
    any leads previously held in earlier generations of this chain
    (is_chain_held=TRUE upstream rows).  Cross-cycle dedups by company
    so the same company never occupies two slots in the final top-K
    even if multiple miners submitted leads at it across generations.
    Picks the top K (where K = the chain's original num_leads), updates
    is_chain_held flags across the entire chain accordingly, and returns
    a structured decision dict.

    Crucially: NO REWARDS ARE WRITTEN HERE.  Reward distribution only
    happens when the chain reaches its full quota; until then, held
    leads sit in DB with is_chain_held=TRUE but is_winner=FALSE and
    reward_pct=NULL.  See ``_finalize_chain_rewards`` for the
    fulfillment-time payout.

    Returned dict:
      {
        "chain_target":       int  (K — the client's original num_leads),
        "topk":               list of consensus dicts (length ≤ K),
        "topk_lead_ids":      set  (ids of leads currently held),
        "topk_companies":     list (normalized company strings, used by
                                    the recycle path's exclusion list),
        "displaced_count":    int  (held leads that just got bumped),
        "this_cycle_winners": list (subset of topk that came from this
                                    cycle — used for tracing only),
        "tied_groups":        list of (lead_id, tied_consensus_rows) so
                                    reward distribution can compute
                                    tie_count correctly later,
      }
    """
    supabase = _get_supabase()

    # 1) Load prior held leads (across all predecessor generations).
    prior_held = _load_chain_held_winners(supabase, request_id)
    chain_target = _chain_target_num_leads(supabase, request_id, current_request_num_leads)

    # 2) Hydrate "business" / company on EVERY candidate.  Need it for
    #    cross-cycle dedup (same company across generations collapses).
    #    Prior held rows already passed validation in their cycle, so we
    #    fetch their lead_data from fulfillment_submissions; current
    #    cycle results may also need hydration.
    needed_subs = {r["submission_id"] for r in (consensus_results + prior_held)}
    sub_lead_data: dict = {}
    if needed_subs:
        sub_resp = supabase.table("fulfillment_submissions") \
            .select("submission_id, lead_data") \
            .in_("submission_id", list(needed_subs)) \
            .execute()
        for row in (sub_resp.data or []):
            ld_list = row.get("lead_data") or []
            lookup = {ld.get("lead_id"): ld.get("data", {}) for ld in ld_list}
            sub_lead_data[row["submission_id"]] = lookup

    def _company_for(r: dict) -> str:
        info = sub_lead_data.get(r["submission_id"], {}).get(r["lead_id"])
        if not info:
            return ""
        return _normalize_company(info.get("business", ""))

    # 3) Build the candidate pool: this cycle's qualifying rows ∪ prior held.
    pool: list = []
    for r in consensus_results:
        if (r.get("consensus_final_score") or 0) <= 0:
            continue
        company = _company_for(r)
        if not company:
            continue
        # Tag origin so we know which rows are "from this cycle" vs prior
        pool.append({**r, "_chain_company": company, "_chain_origin": "current"})
    for r in prior_held:
        company = _company_for(r)
        if not company:
            # A prior held row whose company we can't recover — drop it.
            # Should not happen in practice but defensive.
            continue
        pool.append({**r, "_chain_company": company, "_chain_origin": "prior"})

    # 4) Cross-cycle dedup by company: keep the highest-scoring entry per
    #    company across the whole chain.  A held lead at a company can be
    #    REPLACED by a new cycle's higher-scoring lead at the same company —
    #    the held one gets displaced (is_chain_held → FALSE on next update).
    by_company: dict = {}
    for c in pool:
        key = c["_chain_company"]
        cur = by_company.get(key)
        if cur is None:
            by_company[key] = c
            continue
        # Same-company collision: pick the higher (final_score, intent_signal_final)
        cur_score = (cur.get("consensus_final_score") or 0,
                     cur.get("consensus_intent_signal_final") or 0)
        new_score = (c.get("consensus_final_score") or 0,
                     c.get("consensus_intent_signal_final") or 0)
        if new_score > cur_score:
            by_company[key] = c

    # 4.5) Cross-chain dedup: drop companies already held on a SIBLING
    #      chain for the same client.  The icp.excluded_companies snapshot
    #      from create_request / recycle freezes the exclusion list at one
    #      moment in time; if a sibling chain holds a new company AFTER the
    #      snapshot, the snapshot can't see it and Tier 1 lets it through.
    #      Catching it here at consensus time is the authoritative dedup
    #      point — this loop processes scoring requests sequentially, so
    #      whichever sibling chain reached _resolve_chain_topk first owns
    #      the company and later siblings drop it.
    sibling_held = _load_sibling_chain_held_companies(supabase, request_id)
    if sibling_held:
        removed = [k for k in by_company if k in sibling_held]
        for k in removed:
            del by_company[k]
        if removed:
            print(f"   🚫 Cross-chain dedup: dropped {len(removed)} "
                  f"sibling-held companies for {request_id[:8]}: "
                  f"{removed[:5]}{'...' if len(removed) > 5 else ''}")

    # 5) Rank by score, take top K.
    ranked = sorted(
        by_company.values(),
        key=lambda x: (
            -(x.get("consensus_final_score") or 0),
            -(x.get("consensus_intent_signal_final") or 0),
        ),
    )
    topk = ranked[:chain_target]
    topk_lead_ids = {r["lead_id"] for r in topk}
    topk_companies = [r["_chain_company"] for r in topk]

    displaced_count = sum(1 for r in prior_held if r["lead_id"] not in topk_lead_ids)

    # 6) Persist is_chain_held flags across the whole chain.
    #    Order matters: clear FALSE first so leads moving in/out within
    #    the same cycle land cleanly, then mark TRUE for the new top-K.
    chain_request_ids = [request_id] + _walk_chain_predecessors(supabase, request_id)

    # 6a) Clear is_chain_held=FALSE on every chain row that's no longer in topk.
    if topk_lead_ids:
        try:
            supabase.table("fulfillment_score_consensus").update({
                "is_chain_held": False,
            }).in_("request_id", chain_request_ids) \
              .not_.in_("lead_id", list(topk_lead_ids)) \
              .execute()
        except Exception as e:
            print(f"   ⚠️  Failed clearing is_chain_held for displaced rows: {e}")
    else:
        # Empty top-K — clear everyone in the chain.
        try:
            supabase.table("fulfillment_score_consensus").update({
                "is_chain_held": False,
            }).in_("request_id", chain_request_ids) \
              .execute()
        except Exception as e:
            print(f"   ⚠️  Failed clearing all is_chain_held: {e}")

    # 6b) Set is_chain_held=TRUE on the new top-K.  Each (request_id,
    #     submission_id, lead_id) triple uniquely identifies a row, so
    #     update them individually — there's no cheap multi-row in_()
    #     filter that disambiguates per (request_id, lead_id) pairs.
    for r in topk:
        try:
            supabase.table("fulfillment_score_consensus").update({
                "is_chain_held": True,
            }).eq("request_id", r["request_id"]) \
              .eq("submission_id", r["submission_id"]) \
              .eq("lead_id", r["lead_id"]) \
              .execute()
        except Exception as e:
            print(f"   ⚠️  Failed setting is_chain_held for "
                  f"lead {str(r.get('lead_id',''))[:8]}: {e}")

    # 7) Build tied groups — needed by the reward distribution path.
    tied_groups: list = []
    for r in topk:
        # Within a single (request_id, lead_id) pair the validator scoring
        # may have produced multiple consensus rows from tied miners.  We
        # surface the tied set so reward_pct is split correctly.
        tied = supabase.table("fulfillment_score_consensus") \
            .select("submission_id, lead_id, miner_hotkey, request_id, intent_signal_mapping") \
            .eq("request_id", r["request_id"]) \
            .eq("lead_id", r["lead_id"]) \
            .execute()
        tied_rows = list(tied.data or [])
        # When the predecessor-scoped re-query yields no rows (race with a
        # concurrent write, stale read, or cross-generation schema skew),
        # use the topk row itself as the single tied entry so the lead is
        # still represented in reward distribution.
        if not tied_rows:
            print(
                f"   ⚠️  tied re-query returned 0 rows for "
                f"request={str(r.get('request_id',''))[:8]} "
                f"lead={str(r.get('lead_id',''))[:8]} — "
                f"using topk row as the single tied entry"
            )
            tied_rows = [r]
        tied_groups.append((r["lead_id"], tied_rows))

    return {
        "chain_target": chain_target,
        "topk": topk,
        "topk_lead_ids": topk_lead_ids,
        "topk_companies": topk_companies,
        "displaced_count": displaced_count,
        "this_cycle_winners": [r for r in topk if r.get("_chain_origin") == "current"],
        "tied_groups": tied_groups,
    }


async def _finalize_chain_rewards(
    request_id: str,
    topk: list,
    tied_groups: list,
) -> set:
    """Distribute rewards for a fulfilled chain.  Sets is_winner=TRUE and
    reward_pct on each top-K consensus row.  Called ONLY when the chain
    has reached its full quota and the current request is about to
    transition to status='fulfilled'.

    Tie handling: when multiple miners produced consensus rows for the
    same (request_id, lead_id) — i.e., they submitted identical leads
    that all passed validation — Z_PERCENT is split evenly between
    them (Z_PERCENT / tie_count).  A miner whose lead was held in an
    earlier generation gets paid by the request_id where their row
    actually lives, not the chain head.

    Returns the set of lead_ids that were rewarded.
    """
    from gateway.fulfillment.rewards import calculate_lead_rewards
    supabase = _get_supabase()

    tied_by_lead = {lead_id: tied_rows for lead_id, tied_rows in tied_groups}

    winners: list = []
    for r in topk:
        lid = r["lead_id"]
        # _resolve_chain_topk always seeds tied_groups with at least one
        # entry per topk lead; the `or [r]` is a belt-and-suspenders default
        # so any future change to that contract still distributes the
        # reward for this lead.
        tied_rows = tied_by_lead.get(lid) or [r]
        tie_count = max(1, len(tied_rows))
        for tr in tied_rows:
            winners.append({
                **tr,
                "tie_count": tie_count,
            })

    if not winners:
        return set()

    current_epoch = await _get_current_epoch()
    # ``calculate_lead_rewards`` keys on (request_id, submission_id, lead_id)
    # so cross-generation winners (whose request_id is a predecessor) get
    # their own rows updated — no need to remap to the chain head.
    calculate_lead_rewards(request_id, winners, Z_PERCENT, current_epoch, L_EPOCHS)

    # Enrich each winner with the client-ready "Intent Details" paragraph.
    # Runs ONCE per winner (not per validator) because this happens post-consensus
    # on the gateway.  A Perplexity outage must NOT block reward payout, so each
    # call is individually wrapped.
    await _attach_intent_details_for_winners(supabase, request_id, winners)

    return {w["lead_id"] for w in winners}


def _synthesize_intent_details_fallback(
    icp: dict,
    signals: list,
) -> str:
    """Deterministic, LLM-free synthesis used when the OpenRouter call fails.

    Stitches together a client-readable paragraph directly from the verified
    intent signals (``intent_signal_mapping``) so a winning lead is never
    persisted with a NULL ``intent_details``.  The output is intentionally
    plainer than the LLM version — its job is to guarantee something
    informative shows up in the dashboard even when:

      - The OpenRouter key isn't provisioned on this gateway box.
      - OpenRouter is rate-limited / down / returning refusals.
      - The schema migration adding intent_breakdown hasn't run, so the
        joined LLM persist call was failing and zeroing intent_details too
        (see ``_attach_intent_details_for_winners`` — we now write the two
        columns independently so this fallback also unblocks the schema
        case).

    Returns the empty string only when there is genuinely nothing to say
    (no credited signals at all). The lifecycle caller treats empty as
    "skip the write" so a prior successful run isn't blanked.
    """
    # Only include signals that actually earned credit (after_decay_score > 0,
    # or raw_score > 0 as a legacy fallback). Failed signals must not appear
    # in the client-facing passage.
    credited = []
    for s in signals or []:
        try:
            score = float(s.get("after_decay_score") or s.get("raw_score") or 0)
        except (TypeError, ValueError):
            score = 0.0
        if score > 0:
            credited.append(s)
    if not credited:
        return ""

    sentences: list = []
    for s in credited:
        matched = (s.get("matched_icp_signal") or "").strip()
        desc = (s.get("description") or "").strip()
        snippet = (s.get("snippet") or "").strip()
        source = (s.get("source") or "").strip()
        date = (s.get("date") or "").strip()

        # Prefer the upstream-written description (already prose-shaped by
        # the scoring pipeline). If absent fall back to a trimmed snippet,
        # then to a minimal "Matched X on source / date" stub so the lead
        # is never reduced to a literal blank.
        body = desc or snippet[:400]
        if body:
            sentence = body.rstrip(". ").rstrip()
            if matched and matched.lower() not in sentence.lower():
                sentence = f"{sentence} (matches the ICP signal '{matched}')"
            sentences.append(sentence + ".")
        elif matched:
            tag = []
            if source:
                tag.append(source)
            if date:
                tag.append(date)
            tail = f" ({', '.join(tag)})" if tag else ""
            sentences.append(f"Matched the ICP signal '{matched}'{tail}.")

    # Join with single spaces. Refusals / markdown / em-dashes can't leak in
    # because we're authoring the prose ourselves from structured fields.
    out = " ".join(sentences).strip()
    return out


async def _attach_intent_details_for_winners(
    supabase,
    request_id: str,
    winners: list,
) -> None:
    """Generate + persist the Intent Details payload for each winning lead.

    For each winner:
      1. Try the LLM (one call per winner). Returns:
         - an overall paragraph -> ``fulfillment_score_consensus.intent_details``
         - a per-signal breakdown -> ``fulfillment_score_consensus.intent_breakdown``
      2. If the LLM call yields an empty passage (no key, refusal, timeout,
         non-200, parse failure), fall back to a deterministic synthesizer
         that stitches a passage straight from ``intent_signal_mapping``.
         This guarantees that a winning lead with any credited signal at
         all gets a populated ``intent_details``.
      3. Persist ``intent_details`` and ``intent_breakdown`` in SEPARATE
         Supabase updates. The two columns are independent products of the
         same LLM call — one being unwritable (column missing on a stale
         schema, JSONB constraint, transient Postgres error) must not blank
         out the other. Previously the combined update silently zeroed
         intent_details when intent_breakdown failed.

    Either column is written only when its content is non-empty so a
    partial parse failure cannot blank out a prior successful write.
    """
    if not winners:
        return

    try:
        from gateway.fulfillment.intent_details import generate_intent_details
    except Exception as e:
        # Even with the LLM module unavailable we still want winning leads
        # to carry a populated intent_details column — fall through to the
        # synthesizer path below by stubbing the function.
        print(f"   ⚠️  intent_details module unavailable, using fallback synthesizer only: {e}")

        async def generate_intent_details(_icp, _signals):  # type: ignore
            return {"passage": "", "per_signal": []}

    # Fetch the ICP once — it's the same for every winner in this request.
    try:
        req_resp = supabase.table("fulfillment_requests") \
            .select("icp_details") \
            .eq("request_id", request_id) \
            .execute()
        icp = (req_resp.data or [{}])[0].get("icp_details") or {}
    except Exception as e:
        print(f"   ⚠️  Could not load ICP for intent_details: {e}")
        icp = {}

    for w in winners:
        lead_id = w.get("lead_id")
        submission_id = w.get("submission_id")
        signals = w.get("intent_signal_mapping") or []

        # ----------------------------------------------------------------
        # Step 1: try the LLM, with up to 3 attempts on transient errors.
        # The LLM module itself returns empty (not raises) on refusal /
        # non-200 / parse failure, so we only retry when an exception
        # actually escapes (network blip, httpx timeout).
        # ----------------------------------------------------------------
        out = {"passage": "", "per_signal": []}
        for attempt in range(3):
            try:
                out = await generate_intent_details(icp, signals)
                break
            except Exception as e:
                print(
                    f"   ⚠️  intent_details LLM call raised for "
                    f"{str(lead_id)[:8]} (attempt {attempt + 1}/3): {e}"
                )
                # Exponential backoff: 1s, 2s, 4s. Cheap relative to the
                # LLM call itself; bounded so a hard outage doesn't stall
                # consensus close.
                if attempt < 2:
                    await asyncio.sleep(1 << attempt)

        passage = out.get("passage", "") or ""
        per_signal = out.get("per_signal", []) or []

        # ----------------------------------------------------------------
        # Step 2: if the LLM didn't produce a passage, synthesize one from
        # the raw signal mapping. Per_signal is LLM-only (the deterministic
        # path doesn't try to break the synthesis into LLM-style
        # paragraphs); if the LLM gave no per_signal we simply skip that
        # column for this lead.
        # ----------------------------------------------------------------
        passage_source = "llm"
        if not passage:
            fallback = _synthesize_intent_details_fallback(icp, signals)
            if fallback:
                passage = fallback
                passage_source = "fallback"

        # ----------------------------------------------------------------
        # Step 3: persist intent_details and intent_breakdown INDEPENDENTLY.
        # A failure on the second update (e.g. column missing because the
        # operator hasn't run migration 16) must not roll back the first.
        # ----------------------------------------------------------------
        if passage:
            try:
                supabase.table("fulfillment_score_consensus").update(
                    {"intent_details": passage}
                ).eq("request_id", request_id) \
                  .eq("submission_id", submission_id) \
                  .eq("lead_id", lead_id) \
                  .execute()
                print(
                    f"   📝 intent_details stored for lead {str(lead_id)[:8]} "
                    f"(passage={len(passage)} chars, source={passage_source})"
                )
            except Exception as e:
                print(
                    f"   ⚠️  Failed to persist intent_details for "
                    f"{str(lead_id)[:8]}: {e}"
                )

        if per_signal:
            try:
                supabase.table("fulfillment_score_consensus").update(
                    {"intent_breakdown": {"per_signal": per_signal}}
                ).eq("request_id", request_id) \
                  .eq("submission_id", submission_id) \
                  .eq("lead_id", lead_id) \
                  .execute()
                print(
                    f"   📝 intent_breakdown stored for lead "
                    f"{str(lead_id)[:8]} (per_signal={len(per_signal)})"
                )
            except Exception as e:
                # Common cause: ``intent_breakdown`` column missing because
                # scripts/16-fulfillment-intent-breakdown-column.sql hasn't
                # been applied. intent_details survives because we wrote it
                # in a separate request above.
                print(
                    f"   ⚠️  Failed to persist intent_breakdown for "
                    f"{str(lead_id)[:8]} (intent_details was still saved): {e}"
                )


def _recycle_request(
    supabase,
    original_request: dict,
    now: datetime,
    now_iso: str,
    *,
    terminal_status: str,
    reason: str,
    successor_status_target: str = "open",
    successor_num_leads: int = None,
    held_companies: list = None,
) -> None:
    """Create a successor request and mark the original as terminal.

    Used in three lifecycle scenarios:
      * ``terminal_status='partially_fulfilled'``: chain produced ≥1 held
        leads but fell short of the quota.  Successor enters the queue
        intended for ``continued_open`` (held in pending until the
        promotion step picks it up), inherits the in-flight chain's
        held companies as additional excluded_companies, and asks
        miners only for ``successor_num_leads`` (the remaining quota).
      * ``terminal_status='recycled'``: chain produced 0 held leads.
        Successor is a plain ``open`` (after promotion) request asking
        for the full original quota again.  Used for both
        no-reveals-after-commit and no-useful-output-after-consensus
        edge cases.
      * ``terminal_status='expired'``: validators didn't materialize or
        consensus was empty.  Successor is also ``open``.

    The two new keyword args make the recycle behavior data-driven so
    the chain logic in ``_lifecycle_tick_inner`` controls the policy
    while this function stays focused on the database race-safe
    insert+claim protocol.

    successor_status_target:
      * ``"open"`` (default) — for legacy / no-progress recycles.  The
        ``pending → open`` promotion step in step 0 transitions it.
      * ``"continued_open"`` — for partially_fulfilled chains.  The
        promotion step looks at the predecessor's status to pick the
        right transition target.

    successor_num_leads:
      * ``None`` (default) — inherit predecessor's num_leads as before.
      * specified — use this value (typically chain_target − held_count
        for partially_fulfilled successors).  Stored in the
        ``num_leads`` column of the new row.

    held_companies:
      * ``None`` or ``[]`` — no in-flight held set; only the client's
        prior fulfilled-request winners are added to excluded_companies
        (legacy behavior).
      * non-empty list — these company names are appended to the
        successor's excluded_companies so miners don't re-do work on
        leads that are already held in the chain.  When a held lead
        gets displaced in a later cycle, its company drops from this
        list automatically because the next recycle's held_companies
        only reflects the current chain top-K.
    """
    rid = original_request["request_id"]
    new_id = str(uuid4())

    # Recycle protocol:
    #   1. INSERT the successor row first.  This satisfies the DB-level FK
    #      constraint fulfillment_requests_successor_request_id_fkey, which
    #      requires the successor to exist BEFORE any row can point at it.
    #      The successor enters the queue in 'pending' state with NULL
    #      window timestamps — its commit timer only starts when the
    #      lifecycle promotion step (Step 0) moves it to 'open'.  This
    #      guarantees a recycled request never ticks down its window
    #      while invisible to miners.
    #   2. Atomically claim the predecessor with a guarded UPDATE
    #      (WHERE successor_request_id IS NULL).  Only one concurrent tick
    #      can win; the loser sees claim.data empty and cleans up its own
    #      successor insert.
    #   3. If step 2 fails for ANY reason — race-lost, network error,
    #      exception — the try/finally below deletes the just-inserted
    #      successor so we never leak an orphan 'pending' row back into
    #      the queue.
    #
    # Failure modes and their handling:
    #   * Race with another tick         -> claim_won=False, finally deletes X
    #   * UPDATE throws (network / DB)   -> exception re-raised, finally deletes X
    #   * Hard kill between INSERT and   -> orphan X left (only residual risk;
    #     the finally block                  a periodic orphan-sweep can clean it)
    successor_inserted = False
    claim_won = False
    try:
        # internal_label and company are client-side attribution fields that
        # must survive recycle — a successor still represents the same client
        # request, so dashboards and billing keep the same owner identity.
        # Neither is ever shown to miners (company is scrubbed out of the
        # ICP by api.py::create_request before hashing), but they must match
        # the predecessor so audit trails stay clean across recycles.
        successor_icp = dict(original_request.get("icp_details") or {})

        # Refresh excluded_companies on every recycle.  The predecessor's
        # list was a snapshot at its creation time; in the meantime other
        # requests for the same client may have fulfilled, so their winner
        # companies should also be excluded going forward.  Only refresh
        # when the predecessor was itself auto-populated (detected as:
        # empty list OR populated but client_company is present) — if the
        # client explicitly seeded a non-empty list at create_request time
        # we'd need a marker to preserve it, but since we rebuild the same
        # way create_request does the computed list will be a superset of
        # the client-seeded one plus new winners.  For safety we only
        # refresh when the client_company is known (non-empty).
        client_company = (original_request.get("company") or "").strip()
        if client_company:
            try:
                from gateway.fulfillment.api import (
                    _load_previously_delivered_companies,
                    _load_in_flight_held_companies,
                )
                # 1) Fulfilled-history exclusions (existing behavior).
                refreshed = _load_previously_delivered_companies(supabase, client_company)
                # 2) Cross-chain in-flight held exclusions (new — closes the
                #    CTO-flagged gap where two concurrent chains for the same
                #    client could each hold and deliver the same company).
                #    Skip this chain itself (identified by internal_label);
                #    its own held set is merged below via the held_companies
                #    parameter.  If internal_label is empty, no chain skip
                #    is applied — the helper still excludes only OTHER
                #    requests' held rows because this chain's predecessor
                #    request_ids are already covered by the held_companies
                #    parameter that's about to merge.
                own_label = (original_request.get("internal_label") or "").strip()
                cross_chain_held = _load_in_flight_held_companies(
                    supabase, client_company,
                    exclude_internal_label=(own_label or None),
                )
                if refreshed or cross_chain_held:
                    # Merge with whatever was already there so a client-
                    # supplied exclusion is never silently dropped on recycle.
                    prior = successor_icp.get("excluded_companies") or []
                    seen = {str(x).strip().lower() for x in prior if str(x).strip()}
                    merged = list(prior)
                    for biz in (*refreshed, *cross_chain_held):
                        if biz.strip().lower() not in seen:
                            merged.append(biz)
                            seen.add(biz.strip().lower())
                    successor_icp["excluded_companies"] = merged
            except Exception as e:
                print(f"   ⚠️  recycle: excluded_companies refresh failed "
                      f"for {rid[:8]} → {new_id[:8]}: {type(e).__name__}: {e}")

        # Append in-flight held companies (chain-aware) to excluded_companies.
        # These are leads currently in is_chain_held=TRUE for this chain;
        # they're not yet "delivered" to the client (no rewards distributed
        # yet — that happens only on chain fulfillment), but miners should
        # not waste work re-submitting at the same companies.  When a held
        # lead is later displaced, its company drops off naturally because
        # the next recycle's held_companies only reflects the current top-K.
        if held_companies:
            try:
                prior = successor_icp.get("excluded_companies") or []
                seen = {str(x).strip().lower() for x in prior if str(x).strip()}
                merged = list(prior)
                for biz in held_companies:
                    biz_str = str(biz or "").strip()
                    if not biz_str:
                        continue
                    if biz_str.lower() not in seen:
                        merged.append(biz_str)
                        seen.add(biz_str.lower())
                successor_icp["excluded_companies"] = merged
            except Exception as e:
                print(f"   ⚠️  recycle: held_companies merge failed "
                      f"for {rid[:8]} → {new_id[:8]}: {type(e).__name__}: {e}")

        # Determine successor's num_leads.  For partially_fulfilled chains
        # this is the remaining quota (chain_target − held_count); for
        # plain recycles or expired requests it's the original
        # num_leads (full quota).  None falls back to the predecessor's
        # num_leads for backward compatibility with non-chain callers.
        target_num_leads = (
            successor_num_leads
            if successor_num_leads is not None
            else original_request["num_leads"]
        )

        successor_row = {
            "request_id": new_id,
            "request_hash": "",
            "icp_details": successor_icp,
            "num_leads": target_num_leads,
            "internal_label": original_request.get("internal_label"),
            "company": original_request.get("company"),
            "window_start": None,
            "window_end": None,
            "reveal_window_end": None,
            "status": "pending",
            "created_by": "recycled",
        }
        # required_attributes is a dedicated column (not inside icp_details).
        # Carry it from the predecessor so Tier 2c attribute verification
        # keeps firing across the entire chain; otherwise every recycle
        # silently drops the gate.
        if original_request.get("required_attributes"):
            successor_row["required_attributes"] = original_request["required_attributes"]
        supabase.table("fulfillment_requests").insert(successor_row).execute()
        successor_inserted = True

        claim = supabase.table("fulfillment_requests").update({
            "status": terminal_status,
            "successor_request_id": new_id,
        }).eq("request_id", rid) \
          .is_("successor_request_id", "null") \
          .execute()

        if claim.data:
            claim_won = True
            _log_event(EventType.FULFILLMENT_RECYCLED, "gateway", {
                "old_request_id": rid,
                "new_request_id": new_id,
                "reason": reason,
                "terminal_status": terminal_status,
            })
            print(f"   ♻️  {rid[:8]}... {terminal_status} -> {new_id[:8]}... (reason={reason})")
    except Exception as e:
        print(f"   ❌ Error recycling {rid[:8]}...: {e}")
    finally:
        if successor_inserted and not claim_won:
            # Either we lost the race (another tick beat us) or the UPDATE
            # raised before we could confirm the claim.  Clean up the orphan
            # successor unconditionally — the miner-facing queue should never
            # see a successor that no predecessor points to.
            try:
                supabase.table("fulfillment_requests").delete().eq("request_id", new_id).execute()
            except Exception as cleanup_err:
                print(f"   ⚠️  Orphan cleanup failed for {new_id[:8]}: {cleanup_err}")


def _normalize_company(name: str) -> str:
    """Normalize company name for dedup."""
    import re
    name = name.lower().strip()
    suffixes = (
        r"\b(inc\.?|llc|ltd\.?|corp\.?|corporation|co\.?|company|"
        r"plc|gmbh|ag|sa|sas|srl|bv|nv|pty|pvt)\b"
    )
    name = re.sub(suffixes, "", name, flags=re.IGNORECASE)
    name = re.sub(r"[,.\s]+$", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


async def _get_current_epoch() -> int:
    """Return the current Bittensor epoch ID (async-safe).

    MUST be awaited.  Called from inside the gateway's running event loop
    (fulfillment_lifecycle_task), which means the synchronous variant
    ``get_current_epoch_id()`` is unsafe: it internally runs
    ``_get_current_block()``, which explicitly raises ``RuntimeError`` when
    invoked from a thread that already has a running loop.  The previous
    implementation swallowed that error and returned 0, causing every
    newly-awarded ``reward_expires_epoch`` to be ``0 + L_EPOCHS`` (e.g. 100)
    instead of ``current_epoch + L_EPOCHS`` (e.g. 22264) — making every
    winner expired-at-birth and silently zeroing out fulfillment emission.

    Using the async helper lets the lifecycle tick call ``await`` on a
    helper that reads the cached block without going through the
    sync-wrapper guard.  A broad ``except`` still returns 0 as a last
    resort, but only for a genuine subtensor outage, not the routine
    async-context mismatch the previous bug suffered from.
    """
    try:
        from gateway.utils.epoch import get_current_epoch_id_async
        return int(await get_current_epoch_id_async())
    except Exception as e:
        logger.warning(
            f"_get_current_epoch() fell back to 0 (genuine subtensor failure): "
            f"{type(e).__name__}: {e}"
        )
        return 0


async def _expire_rewards(supabase) -> None:
    """NULL out reward_pct on expired consensus rows (async)."""
    current_epoch = await _get_current_epoch()
    if current_epoch <= 0:
        return
    try:
        supabase.table("fulfillment_score_consensus").update({
            "reward_pct": None,
        }).lte("reward_expires_epoch", current_epoch).not_is("reward_pct", None).execute()
    except Exception:
        try:
            resp = supabase.table("fulfillment_score_consensus") \
                .select("consensus_id, reward_pct") \
                .lte("reward_expires_epoch", current_epoch) \
                .execute()
            for row in (resp.data or []):
                if row.get("reward_pct") is not None:
                    supabase.table("fulfillment_score_consensus").update({
                        "reward_pct": None,
                    }).eq("consensus_id", row["consensus_id"]).execute()
        except Exception as e:
            logger.error(f"Reward expiry fallback failed: {e}")
