"""The approved correction composes on frozen847 without copying current grading."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path

import pytest

from qualification.scoring import lead_scorer
from scripts.arena_reviewer_finality_seam import (
    FinalitySeamError, PREDICATE, _CORRECTED_GUARD, _ORIGINAL_GUARD,
    convert_finality_seam,
)
from scripts.build_arena_reviewer_runtime import (
    CompositionError, ENTRYPOINT_FILES, FINALITY_FILE, GRADER_FILE,
    expected_overlay, verify_overlay,
)
from scripts import build_arena_reviewer_runtime as composer
from scripts.arena_reviewer_finality_seam import CORRECTION_ID
from tests.lab_arena.reviewer_runtime_composition_test import layer
from tests import test_completed_cross_domain_negative as controls
from tests.reviewer_frozen_sources import frozen_source


ROOT = Path(__file__).resolve().parents[1]
_FROZEN_ORIGINAL = frozen_source("negative_finality", FINALITY_FILE)
_FROZEN_POSTIMAGE = _FROZEN_ORIGINAL.replace(
    _ORIGINAL_GUARD.encode(), _CORRECTED_GUARD.encode(), 1)


# Exact CI-green native-budget postimage; historical overlays must not track
# the evolving current investigator or require Git history in test installs.
_FROZEN_GRADER = frozen_source(
    "native_industry_budget", "qualification/scoring/company_evidence_investigator.py",
)


def _current_source():
    # The seam targets frozen847, not the evolving current implementation.
    # Keep its original predicate golden bound to the reviewed postimage.
    return _FROZEN_POSTIMAGE


def _original_source():
    text = _current_source().decode()
    assert text.count(_CORRECTED_GUARD) == 1
    return text.replace(_CORRECTED_GUARD, _ORIGINAL_GUARD, 1).encode()


def _predicate_source(source):
    text = source.decode()
    node = next(node for node in ast.parse(text).body
                if isinstance(node, ast.FunctionDef) and node.name == PREDICATE)
    return ast.get_source_segment(text, node) + "\n"


def _predicate(source):
    scope = dict(vars(lead_scorer))
    exec(compile("from __future__ import annotations\n" + _predicate_source(source),
                 "frozen_finality_predicate.py", "exec"), scope)
    return scope[PREDICATE]


def test_exact_frozen847_predicate_gets_only_the_reviewed_pr483_delta():
    original = (ROOT / 'tests/fixtures/arena_reviewer/frozen847_functions.py.txt').read_bytes()
    # Source-segment hash from frozen847 revision77db393b, including one LF.
    assert hashlib.sha256(_predicate_source(original).encode()).hexdigest() == (
        "73a270140e1bd228d3088a15f42cfe3cf1073b3609867120063880604041dd54"
    )
    assert original.count(_ORIGINAL_GUARD.encode()) == 1
    assert convert_finality_seam(original) == original.replace(
        _ORIGINAL_GUARD.encode(), _CORRECTED_GUARD.encode(), 1,
    )
    before = ast.parse(original)
    after = ast.parse(convert_finality_seam(original))
    for tree in (before, after):
        tree.body = [node for node in tree.body
                     if not isinstance(node, ast.FunctionDef) or node.name != PREDICATE]
    assert ast.dump(before) == ast.dump(after)


def test_already_corrected_source_is_returned_byte_for_byte():
    assert convert_finality_seam(_current_source()) == _current_source()


def test_unrelated_frozen_bytes_are_preserved_instead_of_using_current_grading():
    original = b"# Frozen source-specific bytes must survive.\n" + _original_source()
    expected = b"# Frozen source-specific bytes must survive.\n" + _current_source()
    assert convert_finality_seam(original) == expected


@pytest.mark.parametrize("fault", ["unknown", "partial", "duplicate", "missing", "crlf", "utf8"])
def test_unknown_partial_and_ambiguous_shapes_fail_closed(fault):
    source = _original_source()
    if fault == "unknown":
        source = source.replace(_ORIGINAL_GUARD.encode(),
                                _ORIGINAL_GUARD.replace('"observed_linkedin_slug",',
                                                        '"different_field",').encode(), 1)
    elif fault == "partial":
        source = source.replace(_ORIGINAL_GUARD.encode(),
                                (_CORRECTED_GUARD + _ORIGINAL_GUARD).encode(), 1)
    elif fault == "duplicate":
        source += _predicate_source(source).encode()
    elif fault == "missing":
        source = source.replace(("def " + PREDICATE).encode(), b"def other_predicate", 1)
    elif fault == "crlf":
        source = source.replace(b"\n", b"\r\n")
    else:
        source = b"\xff"
    with pytest.raises(FinalitySeamError):
        convert_finality_seam(source)


def _base_sources():
    files = {name: (ROOT / ("lab_arena/scorer_entrypoint.py"
                          if name == "scorer_entrypoint.py" else name)).read_bytes()
            for name in (GRADER_FILE, *ENTRYPOINT_FILES)}
    files[GRADER_FILE] = _FROZEN_GRADER
    return files


def test_composition_requires_explicit_opt_in_and_exact_frozen_input():
    original = _base_sources()
    default = expected_overlay(original)
    assert FINALITY_FILE not in default
    original[FINALITY_FILE] = _original_source()
    with pytest.raises(CompositionError):
        expected_overlay(original)
    with pytest.raises(CompositionError):
        expected_overlay(_base_sources(), complete_negative_finality=True)
    corrected = expected_overlay(original, complete_negative_finality=True)
    assert corrected[FINALITY_FILE] == convert_finality_seam(original[FINALITY_FILE])
    assert {key: value for key, value in corrected.items() if key != FINALITY_FILE} == default


def test_actual_overlay_cannot_smuggle_another_grading_change():
    original = {**_base_sources(), FINALITY_FILE: _original_source()}
    expected = expected_overlay(original, complete_negative_finality=True)
    verify_overlay(layer([("model/" + key, value) for key, value in expected.items()]), expected)
    altered = dict(expected)
    altered[FINALITY_FILE] += b"\n# Unexpected additional grading bytes\n"
    with pytest.raises(CompositionError, match="size mismatch"):
        verify_overlay(layer([("model/" + key, value) for key, value in altered.items()]), expected)
    altered[FINALITY_FILE] = expected[FINALITY_FILE][:-1] + b" "
    with pytest.raises(CompositionError, match="differs"):
        verify_overlay(layer([("model/" + key, value) for key, value in altered.items()]), expected)
    with pytest.raises(CompositionError, match="unapproved"):
        verify_overlay(layer([("model/" + key, value) for key, value in expected.items()]),
                       expected_overlay(_base_sources()))


@pytest.mark.parametrize("correction", [False, True])
@pytest.mark.parametrize("verify_existing", [False, True])
def test_build_and_verify_use_the_same_explicit_correction_option(
    monkeypatch, capsys, correction, verify_existing,
):
    grading = "registry.example/reviewer@sha256:" + "a" * 64
    runtime = "registry.example/reviewer@sha256:" + "b" * 64
    args = ["build_arena_reviewer_runtime.py", "--grading-image", grading]
    if verify_existing:
        args.extend(["--verify-image", runtime])
    else:
        runtime = ("registry.example/reviewer:reviewer-" + "a" * 12 + "-" + "c" * 12
                   + ("-negative-finality" if correction else ""))
    if correction:
        args.append("--complete-negative-finality")
    monkeypatch.setattr(composer.sys, "argv", args)
    monkeypatch.setattr(composer, "checked_output", lambda *args, **kwargs: "c" * 40)
    base = {"Architecture": "amd64", "Os": "linux", "Config": {},
            "RootFS": {"Layers": ["sha256:base"]}}
    candidate = copy.deepcopy(base)
    candidate["RootFS"]["Layers"].append("sha256:overlay")
    monkeypatch.setattr(composer, "image_inspect", lambda image: base if image == grading else candidate)

    def extract(image, destination, *, complete_negative_finality, source_access_denied,
                failed_attribute_source_recovery):
        assert image == grading
        assert complete_negative_finality is correction
        assert source_access_denied is False
        assert failed_attribute_source_recovery is False
        original = _base_sources()
        if correction:
            original[FINALITY_FILE] = _original_source()
        return original

    def verify(image, digest, expected):
        assert image == runtime and digest == "sha256:overlay"
        assert (FINALITY_FILE in expected) is correction

    monkeypatch.setattr(composer, "extract_frozen_sources", extract)
    monkeypatch.setattr(composer, "verify_saved_overlay", verify)
    builds = []
    monkeypatch.setattr(composer.subprocess, "run", lambda command, **kwargs: builds.append(command))
    assert composer.main() == 0
    evidence = json.loads(capsys.readouterr().out)
    assert evidence["grading_corrections"] == ([CORRECTION_ID] if correction else [])
    assert evidence["grading_image"] == grading
    assert evidence["runtime_image"] == runtime
    assert len(builds) == (0 if verify_existing else 1)
    if builds:
        assert builds[0][builds[0].index("-t") + 1] == runtime


@pytest.mark.parametrize("slug", ["", "leeward-renewable-energy"])
def test_composed_predicate_completes_only_a_healthy_validated_negative(slug):
    identity = {**controls._identity(), "observed_linkedin_slug": slug}
    receipt = controls._receipt()
    before = copy.deepcopy((identity, receipt))
    original = _predicate(_original_source())
    composed = _predicate(convert_finality_seam(_original_source()))
    assert original({}, ("identity", "industry", "stage"),
                    identity_receipt=identity, investigation_receipt=receipt) is bool(slug)
    assert composed({}, ("identity", "industry", "stage"),
                    identity_receipt=identity, investigation_receipt=receipt)
    assert (identity, receipt) == before


@pytest.mark.parametrize("fault", [
    "missing_slug", "null_slug", "number_slug", "whitespace_slug", "missing_identity",
    "wrong_source", "missing_name", "missing_receipt", "incomplete", "provider_error",
    "malformed_response", "missing_rebrand", "missing_target", "positive_rebrand",
    "positive_industry", "unvalidated_quote", "profile_failure", "unbound_fault",
    "http_error", "source_blocked", "fetch_failed", "provider_diagnostic_body", "empty_page",
])
def test_composed_predicate_preserves_all_unfinished_evidence_controls(monkeypatch, fault):
    monkeypatch.setattr(lead_scorer, PREDICATE,
                        _predicate(convert_finality_seam(_original_source())))
    controls.test_cross_domain_negative_keeps_unfinished_proof_retryable(fault)


@pytest.mark.parametrize("semantic", [False, True])
@pytest.mark.parametrize("homepage_fault", [None, "provider_error", "malformed_response"])
@pytest.mark.parametrize("investigation_fault", ["", "provider_error", "malformed_response", "unbound_fault", "incomplete"])
@pytest.mark.parametrize("preserve_marker", [True, False])
def test_composed_finality_through_fit_zero_and_existing_arena_completion(
    monkeypatch, semantic, homepage_fault, investigation_fault, preserve_marker,
):
    monkeypatch.setattr(lead_scorer, PREDICATE,
                        _predicate(convert_finality_seam(_original_source())))
    controls.test_negative_completion_through_fit_and_arena(
        monkeypatch, semantic, homepage_fault, investigation_fault, preserve_marker,
    )
