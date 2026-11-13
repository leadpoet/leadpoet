"""The existing frozen50de feedback composition retains ordered retry passes."""
import ast
import copy
import hashlib
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.usefixtures("scorer_process_environment")

from scripts import arena_reviewer_legacy_company_progress_seam as seam
from scripts import arena_reviewer_investigator_feedback_seam as feedback
from scripts import build_arena_reviewer_runtime as builder
from tests import test_arena_reviewer_investigator_feedback_seam as feedback_controls
from tests.lab_arena import ordered_company_retry_passes_test as controls
from tests.lab_arena.reviewer_legacy_company_progress_composition_test import reference_sources, install_pair
from tests.lab_arena.reviewer_runtime_composition_test import layer

FLAGS = [*feedback_controls.FLAGS, "--ordered-company-retry-passes"]


def frozen_pair():
    return seam.convert_legacy_company_progress_files(reference_sources())


def test_exact_frozen_pair_changes_only_two_existing_retry_functions():
    original = frozen_pair()
    corrected = seam.convert_ordered_company_retry_passes(original)
    assert {name: hashlib.sha256(data).hexdigest() for name, data in original.items()} == seam.ORDERED_RETRY_REFERENCE_SHA256
    assert {name: hashlib.sha256(data).hexdigest() for name, data in corrected.items()} == seam.ORDERED_RETRY_REPAIRED_SHA256
    assert corrected == {name: (builder.ROOT / name).read_bytes() for name in seam.FILES}
    assert corrected[seam.COMPETITION_FILE] == original[seam.COMPETITION_FILE]
    assert seam.convert_ordered_company_retry_passes(corrected) == corrected
    restored = corrected[seam.SCORING_FILE]
    for before, after in reversed(seam._ORDERED_RETRY_DELTAS):
        assert restored.count(after) == 1
        restored = restored.replace(after, before, 1)
    assert restored == original[seam.SCORING_FILE]
    trees = [ast.parse(data) for data in (restored, corrected[seam.SCORING_FILE])]
    for tree in trees:
        tree.body = [node for node in tree.body if getattr(node, "name", "") not in {
            "score_work_item", "_score_legacy_company_work_item",
        }]
    assert ast.dump(trees[0]) == ast.dump(trees[1])


@pytest.mark.parametrize("fault", ["missing", "extra", "drift", "old_generation", "partial", "mixed"])
def test_unrecognized_or_partial_frozen_pair_fails_closed(fault):
    sources = frozen_pair()
    if fault == "missing":
        sources.pop(seam.COMPETITION_FILE)
    elif fault == "extra":
        sources["extra.py"] = b"pass\n"
    elif fault == "drift":
        sources[seam.COMPETITION_FILE] += b"\n"
    elif fault == "old_generation":
        sources = reference_sources()
    elif fault == "partial":
        before, after = seam._ORDERED_RETRY_DELTAS[0]
        sources[seam.SCORING_FILE] = sources[seam.SCORING_FILE].replace(before, after, 1)
    else:
        sources = seam.convert_ordered_company_retry_passes(sources)
        sources[seam.COMPETITION_FILE] = reference_sources()[seam.COMPETITION_FILE]
    with pytest.raises(seam.LegacyCompanyProgressSeamError):
        seam.convert_ordered_company_retry_passes(sources)


@pytest.mark.parametrize("repaired", [False, True])
def test_exact_frozen_pair_reproduces_then_fixes_depth_first_starvation(monkeypatch, repaired):
    sources = frozen_pair()
    install_pair(monkeypatch, seam.convert_ordered_company_retry_passes(sources) if repaired else sources)
    if repaired:
        controls.test_quick_companies_checkpoint_before_earlier_company_retries(monkeypatch)
    else:
        with pytest.raises(AssertionError):
            controls.test_quick_companies_checkpoint_before_earlier_company_retries(monkeypatch)


def test_exact_frozen_pair_keeps_recovery_aliases_and_final_attempt_controls(monkeypatch):
    install_pair(monkeypatch, seam.convert_ordered_company_retry_passes(frozen_pair()))
    cases = [
        (controls.test_late_earlier_alias_keeps_original_order_and_intrinsic_authorities, ()),
        (controls.test_interruption_after_quick_company_retains_its_real_checkpoint, ()),
        (controls.test_private_retry_maps_are_per_company_and_per_work_item, ()),
        *((controls.test_each_unfinished_key_gets_three_attempts_and_no_completed_replay, (fault,))
          for fault in ("provider", "malformed")),
        *((controls.test_source_local_failure_uses_final_attempt_semantics, (recovers,))
          for recovers in (False, True)),
        *((controls.test_native_fit_and_signal_receipts_survive_ordered_paragraph_retry, (participant,))
          for participant in ("baseline", "miner")),
    ]
    for control, arguments in cases:
        with monkeypatch.context() as scope:
            control(scope, *arguments)


def test_exact_frozen_pair_keeps_output_and_checkpoint_contract_errors_fatal(monkeypatch):
    from tests.lab_arena import legacy_company_judgments_test as native
    install_pair(monkeypatch, seam.convert_ordered_company_retry_passes(frozen_pair()))
    from lab_arena import scoring
    native.test_native_scorer_count_contract_error_stops_before_later_company()
    for fatal in (scoring.ScoringError("contract failure"), RuntimeError("programming failure")):
        with monkeypatch.context() as scope:
            native.test_non_exhaustion_failures_stop_before_later_company(scope, fatal)
    for fault in ("checkpoint", "cache"):
        with monkeypatch.context() as scope:
            native.test_checkpoint_or_cache_failure_stops_walk_even_after_deferred_failure(scope, fault)


def test_ordered_passes_require_existing_feedback_composition_before_docker(monkeypatch):
    monkeypatch.setattr(builder.sys, "argv", ["compose", "--grading-image", "registry.example/reviewer@" + feedback.FROZEN_IMAGE_DIGEST,
                                            "--ordered-company-retry-passes"])
    monkeypatch.setattr(builder, "image_inspect", lambda *_: pytest.fail("Docker must not run"))
    with pytest.raises(builder.CompositionError, match="requires investigator-feedback"):
        builder.main()


def test_extraction_reads_exact_feedback_and_complete_scheduler_pair(monkeypatch, tmp_path):
    original = {builder.FINALITY_FILE: feedback_controls.FROZEN_FINALITY,
                builder.GRADER_FILE: feedback_controls.original_investigator(), **frozen_pair()}
    monkeypatch.setattr(builder, "checked_output", lambda *_: "container")
    commands = []

    def run(command, **_kwargs):
        commands.append(command)
        if command[1] == "cp":
            name = command[2].removeprefix("container:/model/")
            Path(command[3]).write_bytes(original[name])

    monkeypatch.setattr(builder.subprocess, "run", run)
    assert builder.extract_frozen_sources("image", tmp_path, submitted_company_context_only=True,
        investigator_feedback=True, ordered_company_retry_passes=True) == original
    assert {command[2] for command in commands if command[1] == "cp"} == {
        "container:/model/" + name for name in original
    }
    assert commands[-1] == ["docker", "rm", "container"]


@pytest.mark.parametrize("verify", [False, True])
def test_existing_feedback_composition_adds_only_reviewed_scoring_delta(monkeypatch, capsys, verify):
    original = {builder.FINALITY_FILE: feedback_controls.FROZEN_FINALITY,
                builder.GRADER_FILE: feedback_controls.original_investigator(), **frozen_pair()}
    expected = {builder.FINALITY_FILE: feedback_controls.CORRECTED_FINALITY,
                builder.GRADER_FILE: feedback.convert_investigator_feedback_seam(original[builder.GRADER_FILE]),
                seam.SCORING_FILE: seam.convert_ordered_company_retry_passes(frozen_pair())[seam.SCORING_FILE]}
    base_ref = "registry.example/reviewer@" + feedback.FROZEN_IMAGE_DIGEST
    runtime_ref = "registry.example/reviewer@sha256:" + "b" * 64
    argv = ["compose", "--grading-image", base_ref, *FLAGS]
    if verify:
        argv += ["--verify-image", runtime_ref]
    monkeypatch.setattr(builder.sys, "argv", argv)
    monkeypatch.setattr(builder, "checked_output", lambda *_a, **_k: "c" * 40)
    base = {"Architecture": "amd64", "Os": "linux", "Config": {"Env": ["FROZEN=1"], "User": "65534:65534"}, "RootFS": {"Layers": ["sha256:base"]}}
    candidate = copy.deepcopy(base)
    candidate["RootFS"]["Layers"].append("sha256:overlay")
    monkeypatch.setattr(builder, "image_inspect", lambda ref: base if ref == base_ref else candidate)

    def extract(ref, _destination, **flags):
        assert ref == base_ref and flags == {"submitted_company_context_only": True,
            "investigator_feedback": True, "ordered_company_retry_passes": True}
        return original

    def checked_overlay(_ref, digest, files):
        assert digest == "sha256:overlay" and files == expected
        builder.verify_overlay(layer([("model/" + path, data) for path, data in files.items()]), expected)

    monkeypatch.setattr(builder, "extract_frozen_sources", extract)
    monkeypatch.setattr(builder, "verify_saved_overlay", checked_overlay)
    commands = []
    monkeypatch.setattr(builder.subprocess, "run", lambda command, **_kwargs: commands.append(command))
    assert builder.main() == 0
    proof = json.loads(capsys.readouterr().out)
    assert proof["grading_image"] == base_ref
    assert proof["grading_corrections"][-2:] == [feedback.CORRECTION_ID, seam.ORDERED_RETRY_CORRECTION_ID]
    assert proof["original_files"] == {name: hashlib.sha256(data).hexdigest() for name, data in original.items()}
    assert proof["overlay_files"] == {name: hashlib.sha256(data).hexdigest() for name, data in expected.items()}
    assert proof["base_layers_unchanged"] is proof["process_configuration_unchanged"] is True
    assert len(commands) == (0 if verify else 1)
