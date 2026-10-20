#!/usr/bin/env python3
"""Measure readable reasoning on one Arena round's agent model calls.

Read-only. It decodes the OpenRouter response receipts already stored in the
provider ledger for the round's execution runs. Judge runs are not counted.
Coverage is the share of calls whose reasoning policy requested readable
reasoning that actually returned it. It prints counts only, never response
text. Exit 0 when the round meets the target, 2 when it does not (including a
round without the policy), 1 when the measurement itself failed.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_MIN_CALL_COVERAGE = 0.8
PAGE = 1000


def _ledger(store, run_id):
    rows, cursor = [], None
    while True:
        page = store.list_ledger(run_id=run_id, provider="openrouter", after_entry_id=cursor, limit=PAGE)
        rows.extend(page)
        if len(page) < PAGE:
            return rows
        next_cursor = int(page[-1]["entry_id"])
        if cursor is not None and next_cursor <= cursor:
            raise ValueError("ledger_pagination_failed")
        cursor = next_cursor


def _response(row):
    """The decoded body of one successful call, or ``None``."""
    terminal = row.get("terminal_response") or {}
    raw = terminal.get("body_b64")
    if terminal.get("call_succeeded") is not True or not isinstance(raw, str):
        return None
    try:
        document = json.loads(base64.b64decode(raw, validate=True).decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    return document if isinstance(document, dict) else None


def measure(store, round_id, *, min_call_coverage=DEFAULT_MIN_CALL_COVERAGE):
    from lab_arena import reasoning_capture

    round_row = store.get_round(round_id)
    if not round_row:
        raise ValueError("round_missing")
    policy = reasoning_capture.round_policy(round_row.get("configuration_doc") or {})
    models = defaultdict(lambda: {"calls": 0, "readable": 0, "requested": 0, "readable_requested": 0})
    treatments = defaultdict(int)
    runs = store.list_runs(round_id, kind="execute")
    summary = {"round_id": round_id, "policy_mode": policy["mode"], "execution_runs": len(runs),
               "runs_with_model_calls": 0, "runs_with_readable_reasoning": 0,
               "calls": 0, "readable_calls": 0, "requested_calls": 0, "readable_requested_calls": 0}
    for run in runs:
        seen = readable = False
        rows = _ledger(store, run["run_id"])
        # The reservation holds the requested model and the policy's record of
        # what it asked for; a response can name a dated model variant.
        reservations = {row.get("call_identity"): row.get("entry_doc") or {}
                        for row in rows if row.get("entry_kind") == "reservation"}
        for row in rows:
            document = _response(row) if row.get("entry_kind") == "settlement" else None
            if document is None:
                continue
            reservation = reservations.get(row.get("call_identity")) or {}
            model = reservation.get("model") if isinstance(reservation.get("model"), str) else "unknown"
            annotation = reservation.get("reasoning_capture")
            annotation = annotation if isinstance(annotation, dict) else {}
            requested = annotation.get("requested") is True
            texts = bool(reasoning_capture.readable_reasoning(document))
            treatments[str(annotation.get("treatment") or "no_policy")] += 1
            entry = models[model]
            entry["calls"] += 1
            entry["readable"] += texts
            entry["requested"] += requested
            entry["readable_requested"] += texts and requested
            summary["calls"] += 1
            summary["readable_calls"] += texts
            summary["requested_calls"] += requested
            summary["readable_requested_calls"] += texts and requested
            seen = True
            readable = readable or texts
        summary["runs_with_model_calls"] += seen
        summary["runs_with_readable_reasoning"] += readable
    requested_calls = summary["requested_calls"]
    summary["call_coverage"] = (
        round(summary["readable_requested_calls"] / requested_calls, 4) if requested_calls else None
    )
    summary["min_call_coverage"] = min_call_coverage
    summary["passed"] = requested_calls > 0 and summary["call_coverage"] >= min_call_coverage
    summary["treatments"] = dict(sorted(treatments.items()))
    summary["models"] = {name: dict(value) for name, value in sorted(models.items())}
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round-id", required=True)
    parser.add_argument("--environment-file", type=Path)
    parser.add_argument("--min-call-coverage", type=float, default=DEFAULT_MIN_CALL_COVERAGE)
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9._:-]{1,200}", args.round_id):
        parser.error("invalid round ID")
    if not 0 < args.min_call_coverage <= 1:
        parser.error("--min-call-coverage must be in (0, 1]")
    store = None
    try:
        if args.environment_file:
            from scripts.run_lab_arena_service import load_scoped_environment
            load_scoped_environment(args.environment_file)
        from lab_arena.store import ArenaStore, PostgrestTransport
        store = ArenaStore(PostgrestTransport(os.environ["LAB_ARENA_SUPABASE_URL"],
                                               service_key=os.environ["LAB_ARENA_SERVICE_KEY"]))
        result = measure(store, args.round_id, min_call_coverage=args.min_call_coverage)
        print(json.dumps(result, sort_keys=True))
        return 0 if result["passed"] else 2
    except Exception as exc:
        # Database errors can contain endpoints or request data; print the type only.
        print("Arena reasoning coverage failed: " + type(exc).__name__, file=sys.stderr)
        return 1
    finally:
        if store is not None:
            store.close()


if __name__ == "__main__":
    raise SystemExit(main())
