"""Exact810 completed-merger composition, offline behavior and source limits."""
import ast
from pathlib import Path
import copy
import hashlib
import json
import sys
from types import ModuleType

import pytest

pytestmark = pytest.mark.usefixtures("scorer_process_environment")

from scripts import build_arena_reviewer_runtime as composer
from scripts import arena_reviewer_completed_merger_seam as seam
from tests import test_completed_merger_survivor_evidence as controls
from tests.lab_arena.reviewer_runtime_composition_test import layer
from tests.reviewer_frozen_sources import frozen_source

ORIGINAL = frozen_source("completed_merger810", seam.FINALITY_FILE)
CORRECTED = seam.convert_completed_merger_seam(ORIGINAL)




def test_exact_source_changes_only_the_reviewed_merger_insertion():
    assert hashlib.sha256(ORIGINAL).hexdigest() == seam.ORIGINAL_SHA256
    assert hashlib.sha256(CORRECTED).hexdigest() == seam.REPAIRED_SHA256
    assert seam.convert_completed_merger_seam(CORRECTED) == CORRECTED
    expected = ORIGINAL.decode()
    for index in range(4):
        old, new = getattr(seam, f"_OLD_{index}"), getattr(seam, f"_NEW_{index}")
        assert expected.count(old) == 1
        assert new in (composer.ROOT / seam.FINALITY_FILE).read_text()
        expected = expected.replace(old, new, 1)
    assert CORRECTED == expected.encode()
    before, after = ast.parse(ORIGINAL), ast.parse(CORRECTED)
    allowed = {"_COMPLETED_MERGER_SURVIVOR_RE", "_ACQUIRED_STAGE_PROOF_PATTERNS",
               "_BOUND_ACQUISITION_SUBJECT_PATTERNS", "_has_affirmed_stage_proof",
               "_acquired_stage_quote_supports_names", "_completed_merger_match_has_uncertainty"}
    def unchanged(tree):
        return [ast.dump(node) for node in tree.body
                if getattr(node, "name", "") not in allowed and not (
                    isinstance(node, ast.Assign) and any(
                        isinstance(target, ast.Name) and target.id in allowed
                        for target in node.targets))]
    assert unchanged(before) == unchanged(after)


@pytest.mark.parametrize("source", [
    b"", b"\xff", ORIGINAL + b"\n", ORIGINAL.replace(b"\n", b"\r\n"),
    ORIGINAL + ORIGINAL, CORRECTED + b"# unrelated\n",
    ORIGINAL.replace(b'"rebrand"', b'"unknown"', 1),
    ORIGINAL.replace(seam._OLD_0.encode(), seam._NEW_0.encode(), 1),
])
def test_unknown_or_partial_sources_fail_closed(source):
    with pytest.raises(seam.CompletedMergerSeamError):
        seam.convert_completed_merger_seam(source)


@pytest.mark.parametrize("verify_existing", [False, True])
def test_exclusive_composer_build_and_verification_are_one_file(monkeypatch, capsys, verify_existing):
    base_ref = "registry.example/reviewer@" + seam.FROZEN_IMAGE_DIGEST
    tag = "registry.example/reviewer:reviewer-" + seam.FROZEN_IMAGE_DIGEST[7:19] + "-" + "c" * 12 + "-completed-merger-survivor"
    candidate_ref = "registry.example/reviewer@sha256:" + "b" * 64 if verify_existing else tag
    args = ["build_arena_reviewer_runtime.py", "--grading-image", base_ref,
            "--completed-merger-survivor-only"]
    if verify_existing:
        args += ["--verify-image", candidate_ref]
    monkeypatch.setattr(composer.sys, "argv", args)
    monkeypatch.setattr(composer, "checked_output", lambda *_a, **_k: "c" * 40)
    base = {"Architecture": "amd64", "Os": "linux", "Config": {},
            "RootFS": {"Layers": ["sha256:base"]}}
    candidate = copy.deepcopy(base)
    candidate["RootFS"]["Layers"].append("sha256:overlay")
    monkeypatch.setattr(composer, "image_inspect", lambda ref: base if ref == base_ref else candidate)

    def extract(ref, _destination, **flags):
        assert ref == base_ref
        assert flags == {"completed_merger_survivor_only": True}
        return {seam.FINALITY_FILE: ORIGINAL}

    def verify(ref, digest, files):
        assert ref == candidate_ref and digest == "sha256:overlay"
        assert files == {seam.FINALITY_FILE: CORRECTED}

    monkeypatch.setattr(composer, "extract_frozen_sources", extract)
    monkeypatch.setattr(composer, "verify_saved_overlay", verify)
    builds = []
    monkeypatch.setattr(composer.subprocess, "run", lambda command, **_k: builds.append(command))
    assert composer.main() == 0
    proof = json.loads(capsys.readouterr().out)
    assert proof["grading_image"] == base_ref
    assert proof["grading_corrections"] == [seam.CORRECTION_ID]
    assert proof["original_files"] == {seam.FINALITY_FILE: seam.ORIGINAL_SHA256}
    assert proof["overlay_files"] == {seam.FINALITY_FILE: seam.REPAIRED_SHA256}
    assert proof["base_layers_unchanged"] is proof["process_configuration_unchanged"] is True
    assert len(builds) == (0 if verify_existing else 1)


@pytest.mark.parametrize("other", [
    "--shim-only", "--empty-domain-negative-finality-only", "--identity-schema-repair",
    "--verified-display-brand", "--complete-negative-finality", "--source-access-denied",
    "--intent-target-denial", "--homepage-profile-alias",
])
def test_exclusive_mode_rejects_mixed_corrections_before_docker(monkeypatch, other):
    monkeypatch.setattr(composer.sys, "argv", [
        "build_arena_reviewer_runtime.py", "--grading-image", "registry.example/base@" + seam.FROZEN_IMAGE_DIGEST,
        "--completed-merger-survivor-only", other,
    ])
    with pytest.raises(composer.CompositionError, match="cannot include other corrections"):
        composer.main()


def test_mode_rejects_other_grading_image_before_docker(monkeypatch):
    monkeypatch.setattr(composer.sys, "argv", [
        "build_arena_reviewer_runtime.py", "--grading-image", "registry.example/base@sha256:" + "a" * 64,
        "--completed-merger-survivor-only",
    ])
    with pytest.raises(composer.CompositionError, match="exact frozen810"):
        composer.main()


def test_overlay_rejects_extra_or_changed_files():
    expected = {seam.FINALITY_FILE: CORRECTED}
    composer.verify_overlay(layer([("model/" + seam.FINALITY_FILE, CORRECTED)]), expected)
    for files in (
        [("model/" + seam.FINALITY_FILE, CORRECTED), ("model/unrelated.py", b"pass\n")],
        [("model/" + seam.FINALITY_FILE, CORRECTED + b"\n")],
    ):
        with pytest.raises(composer.CompositionError):
            composer.verify_overlay(layer(files), expected)




@pytest.mark.parametrize(("entry_id", "quote", "url", "proves_acquisition"), controls.SAVED_STAGE_QUOTES)
def test_frozen810_reproduces_then_repairs_retained_merger_target(entry_id, quote, url, proves_acquisition):
    before = ModuleType("_before_completed_merger")
    after = ModuleType("_after_completed_merger")
    exec(compile(ORIGINAL, "exact_frozen810.py", "exec"), vars(before))
    exec(compile(CORRECTED, "corrected_frozen810.py", "exec"), vars(after))
    assert not before._acquired_stage_quote_supports_names(("Udemy",), quote)
    assert after._acquired_stage_quote_supports_names(("Udemy",), quote) is proves_acquisition
    if proves_acquisition:
        assert not before._stage_quote_supports_observation("acquired", quote)
        assert after._stage_quote_supports_observation("acquired", quote)
    for wrong in ("Coursera", "Merger Sub", "Udemy Systems"):
        assert not after._acquired_stage_quote_supports_names((wrong,), quote)


@pytest.mark.parametrize("control", [
    "If Merger Sub merged with and into Acme Analytics, with Acme Analytics continuing as a direct wholly owned subsidiary of Parent Holdings.",
    "Merger Sub merged with and into Acme Analytics, with Acme Analytics continuing as a direct wholly owned subsidiary of Parent Holdings, subject to regulatory approval.",
    "Merger Sub expects to have merged with and into Acme Analytics, with Acme Analytics continuing as a direct wholly owned subsidiary of Parent Holdings.",
    "Merger Sub merged with and into Other Company, with Acme Analytics continuing as a wholly owned subsidiary of Parent Holdings.",
    "Merger Sub merged with and into Acme Analytics, with Acme Analytics continuing as a wholly owned subsidiary of Parent Holdings. Acme Analytics later became independent.",
])
def test_actual_corrected_frozen810_preserves_adverse_merger_controls(control):
    module = ModuleType("_adverse_completed_merger")
    exec(compile(CORRECTED, "corrected_frozen810.py", "exec"), vars(module))
    assert not module._acquired_stage_quote_supports_names(("Acme Analytics",), control)


def test_registration_only_selects_verified_runtime_under_frozen_grading_identity():
    from scripts.register_arena_reviewer_runtime import registration_arguments
    original = "registry.example/reviewer@" + seam.FROZEN_IMAGE_DIGEST
    runtime = "registry.example/reviewer@sha256:" + "b" * 64
    args = registration_arguments(original, runtime)
    assert args["p_grading_image_digest"] == seam.FROZEN_IMAGE_DIGEST
    assert args["p_runtime_image_reference"] == runtime


def test_mode_extracts_only_the_exact_frozen_lead_scorer(monkeypatch, tmp_path):
    commands = []
    reference = "registry.example/reviewer@" + seam.FROZEN_IMAGE_DIGEST
    monkeypatch.setattr(composer, "checked_output", lambda command: "test-container")

    def docker(command, **kwargs):
        commands.append(command)
        if command[:2] == ["docker", "cp"]:
            assert command[2] == "test-container:/model/" + seam.FINALITY_FILE
            Path(command[3]).write_bytes(ORIGINAL)

    monkeypatch.setattr(composer.subprocess, "run", docker)
    assert composer.extract_frozen_sources(
        reference, tmp_path, completed_merger_survivor_only=True,
    ) == {seam.FINALITY_FILE: ORIGINAL}
    assert [command[:2] for command in commands] == [["docker", "cp"], ["docker", "rm"]]


@pytest.mark.parametrize("quote", controls.ADVERSARIAL_MERGER_QUOTES)
def test_exact_corrected_frozen810_keeps_uncertain_fetched_and_saved_stages_unavailable(monkeypatch, quote):
    import qualification.scoring

    module = ModuleType("_uncertain_completed_merger_runtime")
    exec(compile(CORRECTED, "corrected_frozen810.py", "exec"), vars(module))
    monkeypatch.setattr(controls, "lead_scorer", module)
    monkeypatch.setattr(qualification.scoring, "lead_scorer", module)
    monkeypatch.setitem(sys.modules, "qualification.scoring.lead_scorer", module)
    controls.test_uncertain_merger_cannot_create_a_fetched_or_saved_stage_mismatch(quote)
