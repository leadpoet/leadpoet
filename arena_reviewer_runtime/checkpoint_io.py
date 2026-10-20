"""Scorer file IO using only the supplied frozen scorer and output builders."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Callable


_NONPRINTING_QUOTE_CONTROLS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_QUOTE_FIELDS = frozenset({"quote", "evidence_quote"})


def normalize_evidence_quotes(value: Any, *, field: str = "") -> Any:
    """Normalize retained evidence quotes for the strict output contract.

    Keys, URLs, entity identifiers and decision fields keep their strict
    validation. Raw provider evidence is not modified. Apply the same visible
    quote projection to every copy in final results and partial checkpoints.
    """

    if isinstance(value, dict):
        return {key: normalize_evidence_quotes(item, field=key)
                for key, item in value.items()}
    if isinstance(value, list):
        return [normalize_evidence_quotes(item, field=field) for item in value]
    if isinstance(value, str) and isinstance(field, str) and (
        field in _QUOTE_FIELDS or field.endswith("_evidence_quote")
    ):
        return _NONPRINTING_QUOTE_CONTROLS.sub(" ", value)
    return value


def run_scorer(*, score_input: Callable[..., Any], scoring: Any, quality_policy: Any) -> int:
    input_path = Path(os.environ.get("LAB_ARENA_INPUT_PATH") or "/input/input.json")
    output_path = Path(os.environ.get("LAB_ARENA_OUTPUT_PATH") or "/output/output.json")
    document = json.loads(input_path.read_text(encoding="utf-8"))
    def write_output(result):
        # The runtime reads this same bounded output after a hard timeout.
        # Atomic replacement never exposes a partly written JSON document.
        temporary_path = output_path.with_name(output_path.name + ".tmp")
        temporary_path.write_text(
            json.dumps(normalize_evidence_quotes(result), sort_keys=True),
            encoding="utf-8",
        )
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
