"""Scorer file IO using only the supplied frozen scorer and output builders."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable


def run_scorer(*, score_input: Callable[..., Any], scoring: Any, quality_policy: Any) -> int:
    input_path = Path(os.environ.get("LAB_ARENA_INPUT_PATH") or "/input/input.json")
    output_path = Path(os.environ.get("LAB_ARENA_OUTPUT_PATH") or "/output/output.json")
    document = json.loads(input_path.read_text(encoding="utf-8"))
    def write_output(result):
        # The runtime reads this same bounded output after a hard timeout.
        # Atomic replacement never exposes a partly written JSON document.
        temporary_path = output_path.with_name(output_path.name + ".tmp")
        temporary_path.write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
        os.replace(temporary_path, output_path)

    def checkpoint(rows):
        write_output(scoring.build_scoring_failure(
            document["scored_run_id"], "judge_error", company_judgments=rows,
        ))

    def terminal_checkpoint(rows):
        write_output(scoring.build_scoring_failure(
            document["scored_run_id"], "judge_error", terminal_breakdowns=rows,
        ))

    result = score_input(
        document, **({"on_company_judgments": checkpoint}
                     if (quality_policy.scorer_enabled(document["scorer_policy"])
                         or document["scorer_policy"].get("company_judgment_policy") == "company_judgment_v1")
                     else {"on_terminal_breakdowns": terminal_checkpoint}),
    )
    write_output(result)
    return 0
