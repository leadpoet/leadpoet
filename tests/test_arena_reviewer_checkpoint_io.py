from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from arena_reviewer_runtime import checkpoint_io
from lab_arena import scoring
from scripts.arena_reviewer_checkpoint_seam import (
    CheckpointSeamError, _DELEGATING_MAIN, _ORIGINAL_MAIN,
    convert_checkpoint_seam,
)


def _paths(tmp_path, monkeypatch, *, prior=None):
    document = {"scored_run_id": "original-run", "scorer_policy": {"frozen": True}}
    if prior is not None:
        document["prior_terminal_breakdowns"] = prior
    input_path, output_path = tmp_path / "input.json", tmp_path / "output.json"
    input_path.write_text(json.dumps(document))
    monkeypatch.setenv("LAB_ARENA_INPUT_PATH", str(input_path))
    monkeypatch.setenv("LAB_ARENA_OUTPUT_PATH", str(output_path))
    return document, output_path


@pytest.mark.parametrize("quality", [False, True])
def test_callbacks_use_frozen_builders_and_final_output_is_atomic(tmp_path, monkeypatch, quality):
    document, output = _paths(tmp_path, monkeypatch)
    callback_name = "on_company_judgments" if quality else "on_terminal_breakdowns"
    row_key = "company_judgments" if quality else "terminal_breakdowns"
    rows = [{"original_index": 0}, {"original_index": 1}]
    final = scoring.build_scoring_output("original-run", [{"final_score": 31.0}])
    calls = []

    def frozen_scorer(received, **callbacks):
        assert received == document
        assert set(callbacks) == {callback_name}
        calls.append(received)
        for count in (1, 2):
            callbacks[callback_name](rows[:count])
            assert json.loads(output.read_text()) == scoring.build_scoring_failure(
                "original-run", "judge_error", **{row_key: rows[:count]},
            )
            assert not output.with_name(output.name + ".tmp").exists()
        return final

    assert checkpoint_io.run_scorer(
        score_input=frozen_scorer, scoring=scoring,
        quality_policy=SimpleNamespace(scorer_enabled=lambda policy: quality),
    ) == 0
    assert len(calls) == 1
    assert json.loads(output.read_text()) == final
    assert output.read_text() == json.dumps(final, sort_keys=True)


def test_interrupted_partial_checkpoint_is_available_to_explicit_continuation(tmp_path, monkeypatch):
    partial = [{"original_index": 0, "final_score": 22.0}]
    _, output = _paths(tmp_path, monkeypatch)
    policy = SimpleNamespace(scorer_enabled=lambda _: False)

    def interrupted(document, *, on_terminal_breakdowns):
        on_terminal_breakdowns(partial)
        raise KeyboardInterrupt("injected worker interruption")

    with pytest.raises(KeyboardInterrupt):
        checkpoint_io.run_scorer(score_input=interrupted, scoring=scoring, quality_policy=policy)
    saved = json.loads(output.read_text())
    assert saved == scoring.build_scoring_failure(
        "original-run", "judge_error", terminal_breakdowns=partial,
    )
    document, output = _paths(tmp_path, monkeypatch, prior=saved["terminal_breakdowns"])
    final = scoring.build_scoring_output("original-run", partial)

    def continuation(received, *, on_terminal_breakdowns):
        assert received == document
        assert received["prior_terminal_breakdowns"] == partial
        return final

    assert checkpoint_io.run_scorer(
        score_input=continuation, scoring=scoring, quality_policy=policy,
    ) == 0
    assert json.loads(output.read_text()) == final


@pytest.mark.parametrize("failure_stage", ["write", "replace"])
def test_failed_final_write_preserves_checkpoint_and_next_attempt_succeeds(
    tmp_path, monkeypatch, failure_stage,
):
    _, output = _paths(tmp_path, monkeypatch)
    partial = [{"original_index": 0, "final_score": 22.0}]
    final = scoring.build_scoring_output("original-run", partial)
    policy = SimpleNamespace(scorer_enabled=lambda _: False)
    writes = []

    def frozen_scorer(document, *, on_terminal_breakdowns):
        on_terminal_breakdowns(partial)
        return final

    original_write, original_replace = Path.write_text, checkpoint_io.os.replace

    def write(path, text, **kwargs):
        writes.append(text)
        if len(writes) == 2:
            original_write(path, '{"corrupted":', **kwargs)
            raise OSError("injected interrupted temporary write")
        return original_write(path, text, **kwargs)

    def replace(source, destination):
        writes.append(str(source))
        if len(writes) == 2:
            raise OSError("injected interrupted atomic replacement")
        return original_replace(source, destination)

    with monkeypatch.context() as failure:
        if failure_stage == "write":
            failure.setattr(Path, "write_text", write)
        else:
            failure.setattr(checkpoint_io.os, "replace", replace)
        with pytest.raises(OSError):
            checkpoint_io.run_scorer(
                score_input=frozen_scorer, scoring=scoring, quality_policy=policy,
            )
    assert json.loads(output.read_text()) == scoring.build_scoring_failure(
        "original-run", "judge_error", terminal_breakdowns=partial,
    )
    assert checkpoint_io.run_scorer(
        score_input=frozen_scorer, scoring=scoring, quality_policy=policy,
    ) == 0
    assert json.loads(output.read_text()) == final


def _source():
    return (Path(__file__).resolve().parents[1] / "lab_arena/scorer_entrypoint.py").read_bytes()


def test_conversion_preserves_all_frozen_functions_and_is_idempotent():
    hooked = _source()
    original = hooked.replace(_DELEGATING_MAIN.encode(), _ORIGINAL_MAIN.encode(), 1)
    assert convert_checkpoint_seam(original) == hooked
    assert convert_checkpoint_seam(hooked) == hooked
    before = ast.parse(original)
    after = ast.parse(hooked)
    before.body = [node for node in before.body if getattr(node, "name", None) != "main"]
    after.body = [node for node in after.body if getattr(node, "name", None) != "main"]
    assert ast.dump(before) == ast.dump(after)


@pytest.mark.parametrize("corruption", ["unknown", "duplicate", "missing"])
def test_conversion_rejects_unrecognized_main(corruption):
    source = _source()
    if corruption == "unknown":
        source = source.replace(b"quality_policy=quality_policy,", b"quality_policy=None,")
    elif corruption == "duplicate":
        source += _DELEGATING_MAIN.encode()
    else:
        source = source.replace(_DELEGATING_MAIN.encode(), b"")
    with pytest.raises(CheckpointSeamError):
        convert_checkpoint_seam(source)
