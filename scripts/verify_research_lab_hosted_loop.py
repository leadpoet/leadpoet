#!/usr/bin/env python3
"""Verify Research Lab Hosted Loop MVP local/staged contract."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_lab.hosted_loop import verify_research_lab_hosted_loop


def main() -> int:
    summary = verify_research_lab_hosted_loop()
    print(
        "Research Lab Hosted Loop evaluator gate verified: "
        f"fixture_candidate_metrics_absent={summary['fixture_candidate_metrics_absent']}, "
        f"real_evaluator_score_bundle_required={summary['real_evaluator_score_bundle_required']}, "
        f"production_improvement_scoring_enabled={summary['production_improvement_scoring_enabled']}, "
        f"required_evaluator={summary['required_evaluator']}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
