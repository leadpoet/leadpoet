"""Compose the typed profile alias without changing other frozen source."""

import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
from types import ModuleType

import pytest
from tests.reviewer_frozen_sources import frozen_source

from scripts import arena_reviewer_source_blocked_seam as seam
from scripts import build_arena_reviewer_runtime as composer
from scripts.arena_reviewer_source_access_seam import INVESTIGATOR_DELTAS
from tests import test_company_evidence_source_blocked as controls


# Exact CI-green native-budget postimage; historical overlays must not track
# the evolving current investigator or require Git history in test installs.
_FROZEN_GRADER = frozen_source(
    "native_industry_budget", "qualification/scoring/company_evidence_investigator.py",
)


def _sources(corrected=None):
    if corrected is None:
        corrected = (composer.ROOT / composer.GRADER_FILE).read_bytes()
    assert corrected.count(seam.NEW.encode()) == 1
    original = corrected.replace(seam.NEW.encode(), seam.OLD.encode(), 1)
    return original, corrected


def _original_files(prior_flags=False):
    names = [composer.GRADER_FILE, *composer.ENTRYPOINT_FILES]
    if prior_flags:
        names.append(composer.FINALITY_FILE)
    files = {name: (composer.ROOT / ("lab_arena/scorer_entrypoint.py"
                                   if name == "scorer_entrypoint.py" else name)).read_bytes()
             for name in names}
    files[composer.GRADER_FILE] = _sources(_FROZEN_GRADER)[0]
    return files


def test_only_one_reviewed_block_changes_and_replay_is_identical():
    original, corrected = _sources()
    assert seam.convert_source_blocked_seam(original) == corrected
    assert seam.convert_source_blocked_seam(corrected) == corrected
    assert corrected.replace(seam.NEW.encode(), seam.OLD.encode(), 1) == original
    frozen = b"# unrelated frozen prefix\n" + original + b"\n# frozen footer\n"
    assert seam.convert_source_blocked_seam(frozen) == (
        b"# unrelated frozen prefix\n" + corrected + b"\n# frozen footer\n"
    )


@pytest.mark.parametrize("fault", ["unknown", "duplicate", "missing", "partial", "crlf", "utf8"])
def test_unknown_or_partial_normalizer_fails_closed(fault):
    original, _ = _sources()
    if fault == "unknown":
        original = original.replace(seam.OLD.encode(), seam.OLD.replace('"source_gone"', '"unknown"').encode(), 1)
    elif fault == "duplicate":
        tree = ast.parse(original)
        node = next(node for node in tree.body if getattr(node, "name", "") == "_fetch_outcome")
        original += b"\n" + ast.get_source_segment(original.decode(), node).encode()
    elif fault == "missing":
        original = original.replace(b"def _fetch_outcome(", b"def renamed_outcome(", 1)
    elif fault == "partial":
        original = original.replace(seam.OLD.encode(), (seam.OLD + seam.NEW).encode(), 1)
    elif fault == "crlf":
        original = original.replace(b"\n", b"\r\n")
    else:
        original = b"\xff"
    with pytest.raises(seam.SourceBlockedSeamError):
        seam.convert_source_blocked_seam(original)


def test_existing_source_access_rules_are_required_and_can_compose_first():
    original, corrected = _sources()
    older = original
    for _, old, new in INVESTIGATOR_DELTAS:
        assert older.count(new.encode()) == 1
        older = older.replace(new.encode(), old.encode(), 1)
    with pytest.raises(seam.SourceBlockedSeamError, match="requires source-access"):
        seam.convert_source_blocked_seam(older)
    from scripts.arena_reviewer_source_access_seam import convert_source_access_seam
    assert seam.convert_source_blocked_seam(convert_source_access_seam(older)) == corrected


@pytest.mark.parametrize("prior_flags", [False, True])
def test_overlay_option_preserves_prior_corrections_and_other_files(prior_flags):
    files = _original_files(prior_flags)
    flags = dict.fromkeys(("complete_negative_finality", "source_access_denied",
                           "pending_final_correction"), prior_flags)
    before = composer.expected_overlay(files, **flags)
    after = composer.expected_overlay(files, source_blocked_alias=True, **flags)
    assert set(before) == set(after)
    assert after[composer.GRADER_FILE] == seam.convert_source_blocked_seam(before[composer.GRADER_FILE])
    assert all(after[name] == before[name] for name in before if name != composer.GRADER_FILE)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("verify_existing", [False, True])
def test_build_and_verify_share_explicit_option_and_correction_identity(
    monkeypatch, capsys, enabled, verify_existing,
):
    grading = "registry.example/reviewer@sha256:" + "a" * 64
    runtime = "registry.example/reviewer@sha256:" + "b" * 64
    args = ["compose", "--grading-image", grading, "--source-access-denied"]
    if enabled:
        args.append("--source-blocked-alias")
    if verify_existing:
        args.extend(["--verify-image", runtime])
    monkeypatch.setattr(composer.sys, "argv", args)
    monkeypatch.setattr(composer, "checked_output", lambda *a, **k: "c" * 40)
    base = {"Architecture": "amd64", "Os": "linux", "Config": {}, "RootFS": {"Layers": ["base"]}}
    candidate = copy.deepcopy(base)
    candidate["RootFS"]["Layers"].append("overlay")
    monkeypatch.setattr(composer, "image_inspect", lambda image: base if image == grading else candidate)
    monkeypatch.setattr(composer, "extract_frozen_sources", lambda *a, **k: _original_files(True))
    expected = composer.expected_overlay(_original_files(True), source_access_denied=True,
                                         source_blocked_alias=enabled)
    verified = []
    monkeypatch.setattr(composer, "verify_saved_overlay", lambda image, digest, files: verified.append((image, digest, files)))
    builds = []
    monkeypatch.setattr(composer.subprocess, "run", lambda command, **kwargs: builds.append(command))
    assert composer.main() == 0
    evidence = json.loads(capsys.readouterr().out)
    assert evidence["grading_image"] == grading
    assert evidence["grading_corrections"] == [
        "exact_exa_source_access_denied_v1",
        *([seam.CORRECTION_ID] if enabled else []),
    ]
    assert verified == [(evidence["runtime_image"], "overlay", expected)]
    if verify_existing:
        assert builds == [] and evidence["runtime_image"] == runtime
    else:
        assert len(builds) == 1
        assert ("-source-blocked-alias" in evidence["runtime_image"]) is enabled


@pytest.fixture()
def actual_frozen_sources():
    configured = os.environ.get("ARENA_SOURCE_BLOCKED_FROZEN_ROOT")
    if not configured:
        pytest.skip("exact frozen source is supplied only for frozen release validation")
    root = Path(configured)
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["image_id"] == "sha256:8d0f613c781c2126c86effa91ea0bc7f9d28857521216e6346d522f4c0345bb1"
    files = {name: (root / name).read_bytes()
             for name in (composer.GRADER_FILE, *composer.ENTRYPOINT_FILES)}
    for name, value in files.items():
        assert hashlib.sha256(value).hexdigest() == manifest["files"][name]["sha256"]
    # The producer and finality consumer used by the hermetic loop below are
    # the exact unchanged dependencies from this frozen grader as well.
    for name in ("qualification/scoring/linkedin_company_size.py", composer.FINALITY_FILE):
        frozen = (root / name).read_bytes()
        assert hashlib.sha256(frozen).hexdigest() == manifest["files"][name]["sha256"]
        assert (composer.ROOT / name).read_bytes() == frozen
    return files


def test_actual_frozen_composition_changes_only_the_reviewed_alias(actual_frozen_sources):
    before = composer.expected_overlay(actual_frozen_sources)
    after = composer.expected_overlay(actual_frozen_sources, source_blocked_alias=True)
    assert after[composer.GRADER_FILE] == seam.convert_source_blocked_seam(before[composer.GRADER_FILE])
    assert all(before[name] == after[name] for name in before if name != composer.GRADER_FILE)
    assert after[composer.GRADER_FILE].replace(seam.NEW.encode(), seam.OLD.encode(), 1) == before[composer.GRADER_FILE]


def test_original_frozen_loop_reproduces_the_blocked_profile_failure(monkeypatch, actual_frozen_sources):
    source = composer.expected_overlay(actual_frozen_sources)[composer.GRADER_FILE]
    module = ModuleType("_original_frozen_source_blocked_investigator")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    exec(compile(source, "original_frozen_source_blocked_investigator.py", "exec"), vars(module))
    monkeypatch.setattr(controls, "investigator", module)
    result = controls._run(monkeypatch, "blocked504")
    assert result["_completed_submit"] is False
    assert result["failure_reason"] == "provider_error"
    assert result["usage"]["fetch_outcomes"][0]["error_class"] == "fetch_failed"


@pytest.mark.parametrize("case", [
    "blocked409", "blocked504", "wall", "crawl502", "crawl504", "unbound504",
    "outer504", "outer_timeout", "malformed", "invalid_evidence", "required_search_failure",
])
def test_actual_frozen_loop_preserves_positive_and_negative_controls(
    monkeypatch, actual_frozen_sources, case,
):
    source = composer.expected_overlay(actual_frozen_sources, source_blocked_alias=True)[composer.GRADER_FILE]
    module = ModuleType("_exact_frozen_source_blocked_investigator")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    exec(compile(source, "exact_frozen_source_blocked_investigator.py", "exec"), vars(module))
    monkeypatch.setattr(controls, "investigator", module)
    if case in {"blocked409", "blocked504", "wall"}:
        controls.test_typed_linkedin_block_plus_successful_sources_can_complete_unproven(monkeypatch, case)
    else:
        controls.test_genuine_provider_and_incomplete_evidence_failures_remain_retryable(monkeypatch, case)
