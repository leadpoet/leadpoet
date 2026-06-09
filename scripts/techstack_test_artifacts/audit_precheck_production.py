"""Audit the precheck classifier + URL precheck on real production data.

READ-ONLY: pulls a sample of recent intent_signals from Supabase and
runs them through:

  1. _classify_target_type (regex)
  2. _check_intent_url_evidence_quality (URL precheck)

Reports:
  - distribution of classifier outputs across types
  - cases where evidence_type IS set on the spec but disagrees with the
    regex (potential drift between operator/LLM tag and regex)
  - URL precheck reject rates per type
  - sample of borderline / unclassified cases for manual inspection

NO writes.  Use to verify the precheck is solid before deploying
TECHSTACK + the strict-enum enforcement changes.
"""
import os
import sys
import json
import collections
import textwrap
from typing import Any, Dict, List, Optional

sys.path.insert(0, "/Users/tasnimul/Desktop/leadpoet")

from dotenv import load_dotenv

load_dotenv("/Users/tasnimul/Desktop/leadpoet/.env")

from supabase import create_client  # noqa: E402
from qualification.scoring.intent_precheck import (  # noqa: E402
    _classify_target_type,
    _check_intent_url_evidence_quality,
)


SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")


def main(sample_size: int = 500) -> None:
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY")
        sys.exit(1)
    sb = create_client(SUPABASE_URL, SUPABASE_KEY)

    # Pull recent requests (one row per request) to get icp.intent_signals.
    req_rows = (
        sb.table("fulfillment_requests")
          .select("request_id, icp_details, created_at")
          .order("created_at", desc=True)
          .limit(200)
          .execute()
          .data
        or []
    )
    print(f"\n=== ICP signal classification (across {len(req_rows)} requests) ===")
    spec_dist = collections.Counter()
    regex_dist = collections.Counter()
    drift_pairs = []
    null_after_regex = []
    for row in req_rows:
        icp = row.get("icp_details") or {}
        for sig in (icp.get("intent_signals") or []):
            if not isinstance(sig, dict):
                continue
            text = sig.get("text") or ""
            spec_et = sig.get("evidence_type")
            spec_dist[spec_et or "<null>"] += 1
            regex_et = _classify_target_type(text)
            regex_dist[regex_et or "<null>"] += 1
            if spec_et and regex_et and spec_et != regex_et:
                drift_pairs.append((text[:100], spec_et, regex_et))
            if spec_et is None and regex_et is None:
                null_after_regex.append(text[:120])

    print("\nSpec evidence_type distribution (operator + Gemini tag):")
    for k, v in spec_dist.most_common():
        print(f"  {k:>25s}  {v:5d}")
    print("\nRegex-classifier distribution (precheck fallback):")
    for k, v in regex_dist.most_common():
        print(f"  {k:>25s}  {v:5d}")

    print(f"\n=== Spec vs regex DRIFT ({len(drift_pairs)} cases) ===")
    for text, spec_et, regex_et in drift_pairs[:25]:
        print(f"  spec={spec_et:>20s}  regex={regex_et:>20s}  text={text!r}")
    if len(drift_pairs) > 25:
        print(f"  ... and {len(drift_pairs) - 25} more")

    print(f"\n=== Signals where BOTH spec and regex are null ({len(null_after_regex)} cases) ===")
    for text in null_after_regex[:30]:
        print(f"  {text!r}")
    if len(null_after_regex) > 30:
        print(f"  ... and {len(null_after_regex) - 30} more")

    # Now URL precheck on recent miner-submitted signals.
    # intent_signal_mapping lives on fulfillment_score_consensus rows.
    print(f"\n=== URL precheck audit (most recent {sample_size} consensus rows) ===")
    consensus_rows = (
        sb.table("fulfillment_score_consensus")
          .select("intent_signal_mapping, computed_at")
          .order("computed_at", desc=True)
          .limit(sample_size)
          .execute()
          .data
        or []
    )
    url_dist = collections.Counter()
    reject_per_type = collections.Counter()
    reject_reason_dist = collections.Counter()
    sample_per_reason: Dict[str, List[str]] = collections.defaultdict(list)

    for row in consensus_rows:
        signals = row.get("intent_signal_mapping") or []
        for sig in signals:
            if not isinstance(sig, dict):
                continue
            url = sig.get("url") or ""
            text = (
                sig.get("description")
                or sig.get("matched_icp_signal")
                or sig.get("snippet")
                or ""
            )
            target_type = _classify_target_type(text)
            verdict, reason = _check_intent_url_evidence_quality(url, target_type)
            url_dist[verdict] += 1
            if verdict == "reject":
                reject_per_type[target_type or "<null>"] += 1
                reject_reason_dist[reason] += 1
                if len(sample_per_reason[reason]) < 3:
                    sample_per_reason[reason].append((text[:80], url[:120]))

    print("\nURL precheck verdicts:")
    for k, v in url_dist.most_common():
        print(f"  {k:>10s}  {v:5d}")

    print("\nReject reasons:")
    for k, v in reject_reason_dist.most_common():
        print(f"  {k:>50s}  {v:5d}")

    print("\nReject samples per reason:")
    for reason, samples in sample_per_reason.items():
        print(f"\n  {reason}:")
        for text, url in samples:
            print(f"    text={text!r}")
            print(f"    url={url!r}")


if __name__ == "__main__":
    main()
