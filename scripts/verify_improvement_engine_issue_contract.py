#!/usr/bin/env python3
"""Verify deterministic Improvement Engine issue contracts."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_lab.improvement_engine.clusterer import cluster_events  # noqa: E402
from research_lab.improvement_engine.fingerprints import issue_fingerprint  # noqa: E402
from research_lab.improvement_engine.fix_generator import sanitized_miner_opportunity  # noqa: E402
from research_lab.improvement_engine.models import EngineTraceEvent  # noqa: E402
from research_lab.improvement_engine.scanner import issue_from_events  # noqa: E402


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    event_a = EngineTraceEvent(
        failure_category="candidate_model_zero_companies",
        runtime_stage="private_eval_pair",
        normalized_failure_reason="sourced_zero_no_error",
        component="sourcing_model",
        run_id="00000000-0000-0000-0000-000000000001",
        ticket_id="00000000-0000-0000-0000-000000000011",
        event_at="2026-07-01T00:00:00Z",
        score_delta=-5.0,
    )
    event_b = EngineTraceEvent(
        failure_category="candidate_model_zero_companies",
        runtime_stage="private_eval_pair",
        normalized_failure_reason="sourced_zero_no_error",
        component="sourcing_model",
        run_id="00000000-0000-0000-0000-000000000002",
        ticket_id="00000000-0000-0000-0000-000000000012",
        event_at="2026-07-01T01:00:00Z",
        score_delta=-4.0,
    )
    event_c = EngineTraceEvent(
        failure_category="candidate_build_failed",
        runtime_stage="candidate_patch_test_failed",
        normalized_failure_reason="tests failed",
        component="code_build",
        run_id="00000000-0000-0000-0000-000000000003",
        ticket_id="00000000-0000-0000-0000-000000000013",
        event_at="2026-07-01T02:00:00Z",
    )
    _assert(issue_fingerprint(event_a) == issue_fingerprint(event_b), "same issue did not fingerprint together")
    _assert(issue_fingerprint(event_a) != issue_fingerprint(event_c), "different issues fingerprinted together")
    clusters = cluster_events([event_a, event_b, event_c])
    _assert(len(clusters) == 2, "cluster count mismatch")
    issue = issue_from_events(issue_fingerprint(event_a), [event_a, event_b])
    _assert(issue.category == "candidate_model_zero_companies", "issue category mismatch")
    _assert(issue.occurrence_count == 2, "issue occurrence count mismatch")
    _assert(issue.evaluator_spec_doc["privacy"]["allows_sealed_icp"] is False, "evaluator privacy must block sealed ICP")
    _assert(issue.dataset_spec_doc["privacy"]["refs_only"] is True, "dataset must be refs-only")
    opportunity = sanitized_miner_opportunity(issue)
    _assert("private_refs" not in opportunity, "miner opportunity should not expose private refs")
    print("improvement engine issue contract verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
