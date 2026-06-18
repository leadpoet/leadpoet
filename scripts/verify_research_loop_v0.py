#!/usr/bin/env python3
"""Verify Research Lab P0.6 Engine v0 calibration harness."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_lab.research_loop import dev_eval_lower_confidence_bound, run_research_loop_v0
from research_lab.schema_validation import validate_schema_record


def main() -> int:
    run = run_research_loop_v0()
    rerun = run_research_loop_v0()

    _assert(not validate_schema_record("research_trajectory.schema.json", run.trajectory), "trajectory validates")
    for row in run.ledger_rows:
        _assert(not validate_schema_record("results_ledger_row.schema.json", row), "ledger row validates")

    event_types = [event["type"] for event in run.trajectory["events"]]
    _assert(event_types[0] == "PROBE", "baseline probe runs first")
    _assert(event_types[1] == "LOOP_FUNDED", "loop funding event follows baseline")
    _assert(event_types.count("NODE_DRAFTED") == 2, "two candidate nodes drafted")
    _assert(event_types.count("NODE_EVALUATED") == 2, "two candidate nodes evaluated")
    _assert(event_types.count("NODE_REFLECTED") == 2, "two candidate nodes reflected")
    _assert(event_types[-1] == "PLATEAU_STOP", "run closes with plateau stop")

    _assert(run.best_node_id == "node-source-routing-v0", "source-routing node wins")
    source = next(result for result in run.evaluations if result.node_id == "node-source-routing-v0")
    overbroad = next(result for result in run.evaluations if result.node_id == "node-overbroad-v0")
    _assert(source.status == "scored", "source-routing node scored")
    _assert((source.paired_lcb_vs_parent or 0.0) > 0.0, "source-routing LCB improves")
    _assert(overbroad.status == "guardrail_breach", "overbroad node triggers guardrail")
    _assert(run.trajectory["final"]["settlement"]["balance_returned"] > 0.0, "budget remains")
    _assert(run.trajectory["final"]["settlement"]["crown"] is None, "no crown emitted")
    _assert(all(row["status"] in {"keep", "discard"} for row in run.ledger_rows), "ledger statuses valid")
    _assert(run.trajectory == rerun.trajectory, "trajectory is byte-reproducible by default")
    _assert(run.ledger_rows == rerun.ledger_rows, "ledger rows are byte-reproducible by default")
    _assert(
        run.trajectory["final"]["settlement"]["receipt_ref"]
        == rerun.trajectory["final"]["settlement"]["receipt_ref"],
        "receipt ref is reproducible",
    )

    six_fixture_lcb = dev_eval_lower_confidence_bound([1, 2, 3, 4, 5, 6])
    _assert(six_fixture_lcb["n"] == 6, "dev-eval LCB accepts more than five fixtures")
    _assert(six_fixture_lcb["boundary"] == 2.015, "dev-eval LCB uses paired-t boundary, not probation boundary")

    print("Research Lab Engine v0 calibration harness verified.")
    return 0


def _assert(condition: bool, label: str) -> None:
    if not condition:
        raise AssertionError(label)


if __name__ == "__main__":
    raise SystemExit(main())
