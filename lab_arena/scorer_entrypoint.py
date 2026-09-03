"""Entrypoint of the Arena-built judge image, run by validators for scoring assignments.

Inside the gVisor sandbox with no network, the shim routes every provider call
the Research Lab evaluator makes through the validator's worker socket to the
Arena broker, which signs them with the scored miner's own keys. This process
reads one scoring input (ICP, output, signed scorer policy), scores it with
the Lab scorer exactly as the Arena's central path did, and writes the
breakdown list. Placeholder credentials satisfy the evaluator's environment
checks and are stripped by the shim's trusted-scorer mode before matching.

Failures are reported as a failure document: a judge error or a refused key
(the miner's own key or quota) is the miner's outcome, never a crash.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

from lab_arena import contracts, scoring, shim

PLACEHOLDER_CREDENTIALS = {name: "arena-placeholder-" + name.lower() for name in scoring.CREDENTIAL_ENV_NAMES}


def score_input(document: Dict[str, Any]) -> Dict[str, Any]:
    if document.get("schema_version") != scoring.SCORING_INPUT_SCHEMA_VERSION:
        raise scoring.ScoringError("scoring input schema is invalid")
    work_item_id = contracts.require_sha256(document["work_item_id"], "work_item_id")
    policy = contracts.validate_scorer_policy(document["scorer_policy"])
    icp = dict(document["icp"])
    companies = [dict(item) for item in document["companies"]]
    os.environ[shim.TRUSTED_SCORER_ENV] = "1"
    # One process scores one work item, but a reused process keeps its cache directory.
    cache_dir = os.environ.get(scoring.CACHE_DIR_ENV) or tempfile.mkdtemp(prefix="scoring-cache-")
    scoring.apply_policy_to_environment(policy, environ=os.environ, cache_dir=cache_dir, credentials=dict(PLACEHOLDER_CREDENTIALS))
    scorer = scoring.lab_scorer(policy)
    item = {"work_item_id": work_item_id, "output_hash": contracts.document_hash({"companies": companies})}
    try:
        breakdowns = scoring.score_work_item(item, icp=icp, companies=companies, scorer=scorer, max_scored_companies=int(policy["max_scored_companies"]))
    except scoring.JudgeKeyRefused as exc:
        return scoring.build_scoring_failure(work_item_id, "judge_key_refused", detail=str(exc))
    except scoring.ScoringError as exc:
        return scoring.build_scoring_failure(work_item_id, "judge_error", detail=str(exc))
    return scoring.build_scoring_output(work_item_id, breakdowns)


def main(argv: Any = None) -> int:
    input_path = Path(os.environ.get("LAB_ARENA_INPUT_PATH") or "/input/input.json")
    output_path = Path(os.environ.get("LAB_ARENA_OUTPUT_PATH") or "/output/output.json")
    document = json.loads(input_path.read_text(encoding="utf-8"))
    result = score_input(document)
    output_path.write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover - container entrypoint
    sys.exit(main())
