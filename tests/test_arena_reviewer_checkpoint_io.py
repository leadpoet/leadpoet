from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from arena_reviewer_runtime import checkpoint_io
from lab_arena import contracts, scoring
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


@pytest.mark.parametrize("quality", [False, True])
def test_evidence_quote_controls_are_normalized_in_checkpoints_and_final_output(
    tmp_path, monkeypatch, quality,
):
    _, output = _paths(tmp_path, monkeypatch)
    quote = "Café 東京 — quoted ‘evidence’\nnext\tline\r\nwith\x03control"
    safe_quote = quote.replace("\x03", " ")
    receipt = {"dimension_evidence": {"industry": {"web_evidence": {"quote": quote}}},
               "supporting_receipts": [{"claims": {"industry": {
                   "evidence_quote": quote,
                   "supporting_evidence": [{"quote": quote}],
               }}}]}
    rows = [{"company_qualified": True, "duplicate_company": False,
             "company_identity_key": "domain:example.com", "final_score": 40.0,
             "verifier_gate_receipts": [receipt]}]
    callback_name = "on_company_judgments" if quality else "on_terminal_breakdowns"

    def frozen_scorer(_document, **callbacks):
        callbacks[callback_name](rows)
        checkpoint = json.loads(output.read_text())
        contracts.check_strict_document(checkpoint, contracts.PUBLICATION_LIMITS)
        return scoring.build_scoring_output("original-run", rows)

    checkpoint_io.run_scorer(
        score_input=frozen_scorer, scoring=scoring,
        quality_policy=SimpleNamespace(scorer_enabled=lambda _: quality),
    )
    saved = json.loads(output.read_text())
    contracts.check_strict_document(saved, contracts.PUBLICATION_LIMITS)
    actual = saved["breakdowns"][0]
    assert actual["company_qualified"] is True and actual["final_score"] == 40.0
    assert actual["company_identity_key"] == "domain:example.com"
    assert actual["verifier_gate_receipts"][0]["dimension_evidence"]["industry"]["web_evidence"]["quote"] == safe_quote
    assert rows[0]["verifier_gate_receipts"][0]["dimension_evidence"]["industry"]["web_evidence"]["quote"] == quote


@pytest.mark.parametrize("field", ["url", "evidence_url", "company_identity_key", "decision", "reason", "detail"])
def test_protected_values_and_keys_still_fail_strict_document_checks(field):
    raw = {field: "unchanged\x03value", "bad\x03key": "ordinary"}
    normalized = checkpoint_io.normalize_evidence_quotes(raw)
    assert normalized == raw
    with pytest.raises(contracts.ArenaContractError):
        contracts.check_strict_document(normalized, contracts.PUBLICATION_LIMITS)


def test_all_forbidden_quote_controls_are_replaced_but_legal_text_is_exact():
    forbidden = "".join(chr(code) for code in (*range(9), 11, 12, *range(14, 32), 127))
    legal = "Unicode café 東京 — ‘quote’\t\n\r"
    assert checkpoint_io.normalize_evidence_quotes({"quote": legal}) == {"quote": legal}
    normalized = checkpoint_io.normalize_evidence_quotes({"industry_evidence_quote": forbidden})
    assert normalized == {"industry_evidence_quote": " " * len(forbidden)}
    contracts.check_strict_document(normalized, contracts.PUBLICATION_LIMITS)


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
