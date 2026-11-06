"""Compose the reviewed predicate on exact79cd without current grader copies."""
import ast
import copy
import hashlib
import json

import pytest

from qualification.scoring import lead_scorer
from scripts import build_arena_reviewer_runtime as composer
from scripts.arena_reviewer_empty_domain_finality_seam import (
    CORRECTION_ID, FINALITY_FILE, ORIGINAL_SHA256, PREDICATE, REPAIRED_SHA256,
    EmptyDomainFinalitySeamError, convert_empty_domain_finality_seam,
)
from tests.lab_arena.reviewer_runtime_composition_test import layer
from tests.reviewer_frozen_sources import frozen_source
from tests.test_arena_reviewer_finality_seam import _predicate
from tests import test_completed_empty_domain_negative as controls


ORIGINAL = frozen_source("empty_domain_finality79cd", FINALITY_FILE)
CORRECTED = convert_empty_domain_finality_seam(ORIGINAL)


def test_exact_frozen_source_changes_only_the_reviewed_predicate():
    assert hashlib.sha256(ORIGINAL).hexdigest() == ORIGINAL_SHA256
    assert hashlib.sha256(CORRECTED).hexdigest() == REPAIRED_SHA256
    assert convert_empty_domain_finality_seam(CORRECTED) == CORRECTED
    before, after = ast.parse(ORIGINAL), ast.parse(CORRECTED)
    original_predicate = next(n for n in before.body if isinstance(n, ast.FunctionDef) and n.name == PREDICATE)
    corrected_predicate = next(n for n in after.body if isinstance(n, ast.FunctionDef) and n.name == PREDICATE)
    reviewed = next(n for n in ast.parse(composer.ROOT.joinpath(FINALITY_FILE).read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == PREDICATE)
    assert ast.dump(corrected_predicate) == ast.dump(reviewed)
    old_segment = ast.get_source_segment(ORIGINAL.decode(), original_predicate)
    new_segment = ast.get_source_segment(CORRECTED.decode(), corrected_predicate)
    assert ORIGINAL.count(old_segment.encode()) == 1
    assert CORRECTED == ORIGINAL.replace(old_segment.encode(), new_segment.encode(), 1)
    for tree in (before, after):
        tree.body = [n for n in tree.body if not isinstance(n, ast.FunctionDef) or n.name != PREDICATE]
    assert ast.dump(before) == ast.dump(after)


@pytest.mark.parametrize("source", [
    b"", b"\xff", b"# unrelated\n" + ORIGINAL, ORIGINAL.replace(b"\n", b"\r\n"),
    ORIGINAL + ORIGINAL, CORRECTED + b"# unrelated\n",
    ORIGINAL.replace(b'"identity_not_proven"', b'"unknown_identity"', 1),
])
def test_unknown_partial_and_modified_frozen_generations_fail_closed(source):
    with pytest.raises(EmptyDomainFinalitySeamError):
        convert_empty_domain_finality_seam(source)


@pytest.mark.parametrize("verify_existing", [False, True])
def test_exclusive_build_and_verify_use_exactly_one_frozen_file(monkeypatch, capsys, verify_existing):
    base_ref = "registry.example/reviewer@sha256:" + "a" * 64
    expected_tag = "registry.example/reviewer:reviewer-" + "a" * 12 + "-" + "c" * 12 + "-empty-domain-negative-finality"
    candidate_ref = "registry.example/reviewer@sha256:" + "b" * 64 if verify_existing else expected_tag
    args = ["build_arena_reviewer_runtime.py", "--grading-image", base_ref,
            "--empty-domain-negative-finality-only"]
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
        assert flags == {"empty_domain_negative_finality_only": True}
        return {FINALITY_FILE: ORIGINAL}
    monkeypatch.setattr(composer, "extract_frozen_sources", extract)
    def verify(ref, digest, files):
        assert ref == candidate_ref and digest == "sha256:overlay"
        assert files == {FINALITY_FILE: CORRECTED}
    monkeypatch.setattr(composer, "verify_saved_overlay", verify)
    builds = []
    monkeypatch.setattr(composer.subprocess, "run", lambda command, **_k: builds.append(command))
    assert composer.main() == 0
    proof = json.loads(capsys.readouterr().out)
    assert proof["grading_corrections"] == [CORRECTION_ID]
    assert proof["original_files"] == {FINALITY_FILE: ORIGINAL_SHA256}
    assert proof["overlay_files"] == {FINALITY_FILE: REPAIRED_SHA256}
    assert proof["base_layers_unchanged"] is proof["process_configuration_unchanged"] is True
    assert len(builds) == (0 if verify_existing else 1)


@pytest.mark.parametrize("other", [
    "--shim-only", "--verified-display-brand", "--homepage-profile-alias",
    "--complete-negative-finality", "--source-access-denied", "--intent-target-denial",
])
def test_exclusive_mode_rejects_other_corrections_before_docker(monkeypatch, other):
    monkeypatch.setattr(composer.sys, "argv", [
        "build_arena_reviewer_runtime.py", "--grading-image", "registry.example/base@sha256:" + "a" * 64,
        "--empty-domain-negative-finality-only", other,
    ])
    with pytest.raises(composer.CompositionError, match="cannot include other corrections"):
        composer.main()


def test_overlay_rejects_extra_or_changed_files():
    expected = {FINALITY_FILE: CORRECTED}
    composer.verify_overlay(layer([("model/" + FINALITY_FILE, CORRECTED)]), expected)
    with pytest.raises(composer.CompositionError):
        composer.verify_overlay(layer([("model/" + FINALITY_FILE, CORRECTED),
                                      ("model/unrelated.py", b"pass\n")]), expected)
    with pytest.raises(composer.CompositionError):
        composer.verify_overlay(layer([("model/" + FINALITY_FILE, CORRECTED[:-1] + b" ")]), expected)


def test_frozen_corrected_predicate_passes_all_finality_controls(monkeypatch):
    # The original frozen predicate is the same reviewed pre-fix AST. Its
    # complete negative fails only because no terminal class is attached.
    original = _predicate(ORIGINAL)
    assert not original(controls._verdict(), ("identity",), identity_receipt=controls._identity(),
                        investigation_receipt=controls._receipt())
    monkeypatch.setattr(lead_scorer, PREDICATE, _predicate(CORRECTED))
    controls.test_saved_empty_website_observation_finishes_as_unavailable_not_a_match()
    controls.test_wrong_entity_remains_a_mismatch()
    controls.test_valid_identity_and_complete_negative_cases_keep_existing_finality()


@pytest.mark.parametrize("semantic", [False, True])
def test_composed_predicate_survives_full_company_fit_and_arena(monkeypatch, semantic):
    monkeypatch.setattr(lead_scorer, PREDICATE, _predicate(CORRECTED))
    controls.test_empty_domain_negative_survives_the_company_fit_wrapper(monkeypatch, semantic)


@pytest.mark.parametrize("fault", controls.test_empty_website_keeps_incomplete_or_failed_research_retryable.pytestmark[0].args[1])
def test_composed_predicate_keeps_failed_or_incomplete_evidence_retryable(monkeypatch, fault):
    monkeypatch.setattr(lead_scorer, PREDICATE, _predicate(CORRECTED))
    controls.test_empty_website_keeps_incomplete_or_failed_research_retryable(fault)
