"""
One-shot backfill for ``fulfillment_score_consensus.intent_details``.

Why this exists:
    Before the fix in gateway/fulfillment/lifecycle.py
    (_attach_intent_details_for_winners), winning leads could end up with a
    NULL intent_details column when EITHER:
      a) the OpenRouter LLM call returned empty / refused / timed out, or
      b) the joined update of intent_details + intent_breakdown failed
         because the intent_breakdown column didn't exist yet (migration
         16 not applied at the time).
    Both paths now write intent_details independently with a deterministic
    fallback, but historical winners are still blank in the dashboard's
    "Intent Details" column.

What this script does:
    Walks every ``fulfillment_score_consensus`` row where
    ``is_winner = true`` and ``intent_details`` is NULL or empty, and:
      1. Re-runs the LLM (``generate_intent_details``) using the request's
         icp_details and the row's intent_signal_mapping.
      2. If the LLM returns empty, falls back to the same deterministic
         synthesizer the live path now uses
         (_synthesize_intent_details_fallback, imported via lifecycle).
      3. Persists intent_details only — intent_breakdown gets written too
         when the LLM produced per-signal entries AND the column exists
         (errors there are tolerated, intent_details is the priority).

Usage:
    Run on the gateway host where the OpenRouter key + Supabase service
    role key are already provisioned:

        cd /home/ec2-user/Bittensor-subnet
        source venv/bin/activate
        python -m scripts.backfill_intent_details              # dry run
        python -m scripts.backfill_intent_details --apply      # write

    Optional flags:
        --request-id UUID   Limit to one fulfillment request (and the
                            entire chain it belongs to via root).
        --limit N           Stop after N winners. Defaults to all.
        --sleep S           Sleep S seconds between LLM calls
                            (default 0.4 — keeps OpenRouter under
                            the per-key rate limit during bulk runs).

Required env:
    SUPABASE_URL                                (or PUBLIC_SUPABASE_URL)
    SUPABASE_SERVICE_ROLE_KEY                   (write access)
    FULFILLMENT_OPENROUTER_API_KEY              (LLM enrichment)
        (OPENROUTER_API_KEY / OPENROUTER_KEY also accepted as fallbacks)

This script is intentionally idempotent: rows that already have a
non-empty intent_details after a prior run are skipped on subsequent
invocations. Safe to re-run after an OpenRouter outage.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from typing import Any, Dict, List, Optional


def _require_supabase():
    """Lazy import + env validation so --help works without supabase installed."""
    try:
        from supabase import create_client
    except ImportError:
        print(
            "ERROR: supabase-py not installed. Run `pip install supabase` "
            "or activate the gateway venv before invoking this script.",
            file=sys.stderr,
        )
        sys.exit(2)

    url = os.getenv("SUPABASE_URL") or os.getenv("PUBLIC_SUPABASE_URL")
    key = (
        os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        or os.getenv("SUPABASE_SECRET_KEY")
    )
    if not url or not key:
        print(
            "ERROR: SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set.",
            file=sys.stderr,
        )
        sys.exit(2)
    return create_client(url, key)


def _fetch_target_winners(
    supabase,
    request_id: Optional[str],
    limit: Optional[int],
) -> List[Dict[str, Any]]:
    """Pull winning consensus rows that need an intent_details backfill."""
    q = supabase.table("fulfillment_score_consensus").select(
        "consensus_id, request_id, submission_id, lead_id, "
        "intent_signal_mapping, intent_details"
    ).eq("is_winner", True)

    if request_id:
        q = q.eq("request_id", request_id)

    # Supabase REST limits, so paginate. We bound at 5000 rows here just to
    # avoid a runaway memory footprint; raise via --limit if you really need
    # to backfill more in one go.
    rows = q.limit(5000).execute().data or []

    # Filter to NULL / empty intent_details client-side because the REST
    # filter for "value is null OR value = ''" is awkward to express.
    needs = [
        r for r in rows
        if not (r.get("intent_details") or "").strip()
    ]
    if limit:
        needs = needs[:limit]
    return needs


def _fetch_icp(supabase, request_id: str) -> Dict[str, Any]:
    """One ICP per request — fetch once and cache."""
    resp = supabase.table("fulfillment_requests").select("icp_details").eq(
        "request_id", request_id
    ).limit(1).execute()
    if not resp.data:
        return {}
    return resp.data[0].get("icp_details") or {}


def _synthesize_intent_details_fallback(
    icp: Dict[str, Any],
    signals: List[Dict[str, Any]],
) -> str:
    """Deterministic, LLM-free synthesizer for the backfill.

    Inlined here (rather than imported from gateway.fulfillment.lifecycle) so
    this script runs against ANY gateway box without requiring the lifecycle
    code update to be deployed first. Keep this in sync with the matching
    function in gateway/fulfillment/lifecycle.py — the contract is identical:
    given a list of signals from intent_signal_mapping, return a passage
    composed of credited (after_decay_score > 0) entries, or empty string if
    nothing was credited.
    """
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

    sentences: List[str] = []
    for s in credited:
        matched = (s.get("matched_icp_signal") or "").strip()
        desc = (s.get("description") or "").strip()
        snippet = (s.get("snippet") or "").strip()
        source = (s.get("source") or "").strip()
        date = (s.get("date") or "").strip()

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

    return " ".join(sentences).strip()


async def _run(
    apply: bool,
    request_id: Optional[str],
    limit: Optional[int],
    sleep_s: float,
) -> int:
    supabase = _require_supabase()

    # Import the LLM generator lazily so --help works without the gateway
    # env present.
    from gateway.fulfillment.intent_details import generate_intent_details

    targets = _fetch_target_winners(supabase, request_id, limit)
    print(
        f"Found {len(targets)} winning row(s) with empty intent_details "
        f"({'apply' if apply else 'dry run'})."
    )

    icp_cache: Dict[str, Dict[str, Any]] = {}
    filled_llm = 0
    filled_fallback = 0
    skipped_empty = 0
    failed = 0

    for i, row in enumerate(targets):
        rid = row["request_id"]
        sub_id = row["submission_id"]
        lead_id = row["lead_id"]
        signals = row.get("intent_signal_mapping") or []

        if rid not in icp_cache:
            try:
                icp_cache[rid] = _fetch_icp(supabase, rid)
            except Exception as e:
                print(f"  [{i + 1}/{len(targets)}] icp fetch failed for {rid[:8]}: {e}")
                icp_cache[rid] = {}
        icp = icp_cache[rid]

        # Try the LLM first.
        out: Dict[str, Any] = {"passage": "", "per_signal": []}
        try:
            out = await generate_intent_details(icp, signals)
        except Exception as e:
            print(
                f"  [{i + 1}/{len(targets)}] LLM raised for lead "
                f"{str(lead_id)[:8]}: {e}"
            )

        passage = (out.get("passage") or "").strip()
        per_signal = out.get("per_signal") or []
        source = "llm"
        if not passage:
            passage = _synthesize_intent_details_fallback(icp, signals)
            source = "fallback" if passage else "none"

        if not passage:
            skipped_empty += 1
            print(
                f"  [{i + 1}/{len(targets)}] lead {str(lead_id)[:8]} "
                f"no signals to synthesize from — skipping"
            )
            continue

        if not apply:
            print(
                f"  [{i + 1}/{len(targets)}] would write {len(passage)} chars "
                f"to lead {str(lead_id)[:8]} (source={source})"
            )
            if source == "llm":
                filled_llm += 1
            else:
                filled_fallback += 1
            await asyncio.sleep(sleep_s)
            continue

        try:
            supabase.table("fulfillment_score_consensus").update(
                {"intent_details": passage}
            ).eq("request_id", rid).eq("submission_id", sub_id).eq(
                "lead_id", lead_id
            ).execute()
            if source == "llm":
                filled_llm += 1
            else:
                filled_fallback += 1
            print(
                f"  [{i + 1}/{len(targets)}] wrote {len(passage)} chars to "
                f"lead {str(lead_id)[:8]} (source={source})"
            )
        except Exception as e:
            failed += 1
            print(
                f"  [{i + 1}/{len(targets)}] write failed for lead "
                f"{str(lead_id)[:8]}: {e}"
            )

        if per_signal:
            try:
                supabase.table("fulfillment_score_consensus").update(
                    {"intent_breakdown": {"per_signal": per_signal}}
                ).eq("request_id", rid).eq("submission_id", sub_id).eq(
                    "lead_id", lead_id
                ).execute()
            except Exception as e:
                # Most likely: intent_breakdown column missing (migration 16
                # not applied). intent_details is already saved — that's
                # the priority.
                print(
                    f"  [{i + 1}/{len(targets)}] intent_breakdown write "
                    f"skipped (column may not exist): {e}"
                )

        await asyncio.sleep(sleep_s)

    print()
    print(
        f"Done. llm={filled_llm} fallback={filled_fallback} "
        f"skipped_empty={skipped_empty} failed={failed} "
        f"total_targeted={len(targets)}"
    )
    return 0 if failed == 0 else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument(
        "--apply",
        action="store_true",
        help="Actually write to Supabase. Without this flag the script "
        "runs in dry-run mode and only prints what it would do.",
    )
    p.add_argument(
        "--request-id",
        default=None,
        help="If set, only backfill winners on this request_id.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop after this many winners.",
    )
    p.add_argument(
        "--sleep",
        type=float,
        default=0.4,
        help="Seconds to sleep between LLM calls (default 0.4).",
    )
    args = p.parse_args()
    return asyncio.run(
        _run(
            apply=args.apply,
            request_id=args.request_id,
            limit=args.limit,
            sleep_s=args.sleep,
        )
    )


if __name__ == "__main__":
    sys.exit(main())
