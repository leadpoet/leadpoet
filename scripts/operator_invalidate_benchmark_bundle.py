"""Operator tool: supersede (tombstone) a daily private-model benchmark bundle.

Why this exists
---------------
A benchmark bundle row that reaches ``current_benchmark_status == 'completed'``
satisfies the daily-baseline gate: the baseline owner skips fresh runs for that
(day, window, manifest) and the same-day reuse path accepts it for the whole
day. When such a bundle must be withdrawn — e.g. an operator carry-forward that
should not stand as the day's measurement — it must be SUPERSEDED, never
deleted: bundles are KMS-signed, referenced by dispatch events and public
reports, and may be anchored externally, so hard-deleting breaks the audit
chain. Appending a ``tombstoned`` status event flips the bundle's current
status (the current view is latest-event-wins), which makes
``_private_benchmark_row_is_valid`` reject it, which re-opens the daily gate so
the next scoring pass runs a fresh baseline.

Safety
------
* Dry-run by default; ``--write`` is required to append the event.
* Refuses to run without an explicit ``--bundle-id`` unless the targeted
  bundle's ``score_summary_doc`` carries the ``operator_carry_forward`` marker
  (i.e. by default it can only tombstone carry-forwards, not real runs).
* Prints full provenance before and after; the appended event records the
  operator reason and prior status.

Run on the gateway host (needs the gateway env / DB credentials):

    cd /home/ec2-user
    python3 -m scripts.operator_invalidate_benchmark_bundle --benchmark-date 2026-07-07
    python3 -m scripts.operator_invalidate_benchmark_bundle --benchmark-date 2026-07-07 --write
"""

import argparse
import asyncio
import json
import sys


async def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-date", required=True, help="UTC benchmark date, e.g. 2026-07-07")
    parser.add_argument("--bundle-id", default="", help="Exact benchmark_bundle_id (required to tombstone a non-carry-forward bundle)")
    parser.add_argument("--reason", default="carry_forward_not_intended", help="Operator reason recorded on the event")
    parser.add_argument("--actor-ref", default="operator:invalidate_benchmark_bundle_script", help="Actor reference recorded on the event")
    parser.add_argument("--write", action="store_true", help="Actually append the tombstone event (default: dry run)")
    args = parser.parse_args()

    from gateway.research_lab.store import (
        create_private_model_benchmark_event,
        select_many,
    )

    filters = [("benchmark_date", args.benchmark_date)]
    if args.bundle_id:
        filters.append(("benchmark_bundle_id", args.bundle_id))
    rows = await select_many(
        "research_lab_private_model_benchmark_current",
        columns=(
            "benchmark_bundle_id,benchmark_date,benchmark_quality,benchmark_attempt,"
            "scoring_worker_ref,rolling_window_hash,evaluation_epoch,"
            "current_benchmark_status,current_event_seq,created_at,score_summary_doc"
        ),
        filters=tuple(filters),
        order_by=(("created_at", True),),
        limit=25,
    )
    if not rows:
        print(f"No benchmark bundles found for {args.benchmark_date}"
              + (f" with bundle_id {args.bundle_id}" if args.bundle_id else ""))
        return 1

    targets = []
    for row in rows:
        doc = row.get("score_summary_doc") if isinstance(row.get("score_summary_doc"), dict) else {}
        is_carry_forward = "operator_carry_forward" in doc
        print("=" * 78)
        print(f"bundle_id      : {row.get('benchmark_bundle_id')}")
        print(f"date/attempt   : {row.get('benchmark_date')} / attempt {row.get('benchmark_attempt')}")
        print(f"status/quality : {row.get('current_benchmark_status')} / {row.get('benchmark_quality')}")
        print(f"worker_ref     : {row.get('scoring_worker_ref')}")
        print(f"window/epoch   : {str(row.get('rolling_window_hash'))[:24]}... / {row.get('evaluation_epoch')}")
        print(f"carry_forward  : {is_carry_forward}")
        if str(row.get("current_benchmark_status") or "") != "completed":
            print("-> skipped: not in 'completed' status (already superseded or not gate-relevant)")
            continue
        if not args.bundle_id and not is_carry_forward:
            print("-> skipped: not a carry-forward; pass --bundle-id explicitly to target a real run")
            continue
        targets.append((row, doc))

    if not targets:
        print("\nNothing to tombstone.")
        return 1

    print("=" * 78)
    print(f"{len(targets)} bundle(s) selected for tombstoning.")
    if not args.write:
        print("DRY RUN — re-run with --write to append the tombstone event(s).")
        return 0

    for row, doc in targets:
        bundle_id = str(row["benchmark_bundle_id"])
        event = await create_private_model_benchmark_event(
            benchmark_bundle_id=bundle_id,
            event_type="tombstoned",
            benchmark_status="tombstoned",
            event_doc={
                "schema_version": "1.0",
                "source": "operator_invalidate_benchmark_bundle_script",
                "operator_action": "supersede_benchmark_bundle",
                "operator_reason": args.reason,
                "actor_ref": args.actor_ref,
                "previous_benchmark_status": str(row.get("current_benchmark_status") or ""),
                "previous_benchmark_quality": str(row.get("benchmark_quality") or ""),
                "was_operator_carry_forward": "operator_carry_forward" in doc,
            },
        )
        print(f"tombstoned {bundle_id}: event seq={event.get('seq')}")

    # Post-state verification: the current view must no longer say completed.
    for row, _doc in targets:
        after = await select_many(
            "research_lab_private_model_benchmark_current",
            columns="benchmark_bundle_id,current_benchmark_status,current_event_seq",
            filters=(("benchmark_bundle_id", str(row["benchmark_bundle_id"])),),
            limit=1,
        )
        state = after[0] if after else {}
        print(f"post-state {row['benchmark_bundle_id']}: "
              f"current_benchmark_status={state.get('current_benchmark_status')} "
              f"(seq {state.get('current_event_seq')})")
        if str(state.get("current_benchmark_status") or "") == "completed":
            print("ERROR: bundle still reads completed — investigate before rerunning")
            return 2

    print("\nDone. The next scoring pass re-opens the daily baseline for this date "
          "(scoring maintenance must be resumed for the baseline owner to run it).")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
