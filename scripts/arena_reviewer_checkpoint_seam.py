"""Extract only the known operational scorer main into its runtime interface.

The scorer function, imports, schemas and all other source bytes are retained.
"""

from __future__ import annotations


class CheckpointSeamError(ValueError):
    """The scorer entrypoint does not have the reviewed operational shape."""


_ORIGINAL_MAIN = """def main(argv: Any = None) -> int:
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

    from lab_arena import quality_policy
    result = score_input(
        document, **({"on_company_judgments": checkpoint}
                     if quality_policy.scorer_enabled(document["scorer_policy"])
                     else {"on_terminal_breakdowns": terminal_checkpoint}),
    )
    write_output(result)
    return 0
"""
_DELEGATING_MAIN = '''def main(argv: Any = None) -> int:
    from arena_reviewer_runtime.checkpoint_io import run_scorer
    from lab_arena import quality_policy
    return run_scorer(
        score_input=score_input, scoring=scoring, quality_policy=quality_policy,
    )
'''


def convert_checkpoint_seam(source: bytes) -> bytes:
    try:
        text = source.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CheckpointSeamError("entrypoint source must be UTF-8") from exc
    if "\r" in text:
        raise CheckpointSeamError("entrypoint source must use LF newlines")
    original_count = text.count(_ORIGINAL_MAIN)
    delegated_count = text.count(_DELEGATING_MAIN)
    if original_count == 1 and delegated_count == 0:
        text = text.replace(_ORIGINAL_MAIN, _DELEGATING_MAIN, 1)
    elif original_count != 0 or delegated_count != 1:
        raise CheckpointSeamError("unrecognized or ambiguous scorer main")
    compile(text, "scorer_entrypoint.py", "exec")
    return text.encode("utf-8")
