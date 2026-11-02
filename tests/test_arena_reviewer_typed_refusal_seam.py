"""The frozen grader overlay changes only reviewed page-refusal blocks."""

import ast
import copy
import json
from pathlib import Path

import pytest
from tests.reviewer_frozen_sources import frozen_source

from scripts import build_arena_reviewer_runtime as composer
from scripts import arena_reviewer_typed_refusal_seam as seam


ROOT = Path(__file__).resolve().parents[1]


# Exact CI-green native-budget postimage; historical overlays must not track
# the evolving current investigator or require Git history in test installs.
_FROZEN_GRADER = frozen_source(
    "native_industry_budget", "qualification/scoring/company_evidence_investigator.py",
)


def _sources(investigator=None):
    verifier = (ROOT / composer.VERIFICATION_FILE).read_bytes()
    if investigator is None:
        investigator = _FROZEN_GRADER
    old_verifier = verifier
    old_investigator = investigator
    for old, new in (
        (seam._VERIFIER_CLASS_OLD, seam._VERIFIER_CLASS_NEW),
        (seam._VERIFIER_FETCH_OLD, seam._VERIFIER_FETCH_NEW),
        (seam._VERIFIER_CATCH_OLD, seam._VERIFIER_CATCH_NEW),
    ):
        assert old_verifier.count(new.encode()) == 1
        old_verifier = old_verifier.replace(new.encode(), old.encode(), 1)
    for old, new in (
        (seam._INVESTIGATOR_PREFIX_OLD, seam._INVESTIGATOR_PREFIX_NEW),
        (seam._INVESTIGATOR_CATCH_OLD, seam._INVESTIGATOR_CATCH_NEW),
    ):
        assert old_investigator.count(new.encode()) == 1
        old_investigator = old_investigator.replace(new.encode(), old.encode(), 1)
    return (old_verifier, verifier), (old_investigator, investigator)


def test_exact_helper_bytes_only_and_idempotent():
    (old_v, new_v), (old_i, new_i) = _sources()
    # The current source may change outside these reviewed helper blocks.
    # Exact frozen-image source is checked separately during image composition.
    assert seam.convert_verification_seam(old_v) == new_v
    assert seam.convert_verification_seam(new_v) == new_v
    assert seam.convert_investigator_seam(old_i) == new_i
    assert seam.convert_investigator_seam(new_i) == new_i
    for before, after, deltas in (
        (old_v, new_v, ((seam._VERIFIER_CLASS_OLD, seam._VERIFIER_CLASS_NEW),
                        (seam._VERIFIER_FETCH_OLD, seam._VERIFIER_FETCH_NEW),
                        (seam._VERIFIER_CATCH_OLD, seam._VERIFIER_CATCH_NEW))),
        (old_i, new_i, ((seam._INVESTIGATOR_PREFIX_OLD, seam._INVESTIGATOR_PREFIX_NEW),
                        (seam._INVESTIGATOR_CATCH_OLD, seam._INVESTIGATOR_CATCH_NEW))),
    ):
        for old, new in deltas:
            before = before.replace(old.encode(), b"# APPROVED BLOCK\n", 1)
            after = after.replace(new.encode(), b"# APPROVED BLOCK\n", 1)
        assert before == after
    assert ast.parse(new_v) and ast.parse(new_i)


def test_prior_verifier_correction_only_adds_reviewed_homepage_catch():
    (_, corrected), _ = _sources()
    prior = corrected.replace(
        seam._VERIFIER_CATCH_NEW.encode(), seam._VERIFIER_CATCH_OLD.encode(), 1,
    )
    assert seam.convert_verification_seam(prior) == corrected
    assert seam.convert_verification_seam(corrected) == corrected
    for old, new in (
        (seam._VERIFIER_CLASS_OLD, seam._VERIFIER_CLASS_NEW),
        (seam._VERIFIER_FETCH_OLD, seam._VERIFIER_FETCH_NEW),
    ):
        unexpected = prior.replace(new.encode(), old.encode(), 1)
        with pytest.raises(seam.TypedRefusalSeamError, match="partial"):
            seam.convert_verification_seam(unexpected)


@pytest.mark.parametrize("file,fault", [
    ("verifier", "unknown"), ("verifier", "partial"),
    ("verifier", "crlf"), ("verifier", "utf8"),
    ("investigator", "unknown"), ("investigator", "partial"),
    ("investigator", "crlf"), ("investigator", "utf8"),
])
def test_unknown_or_partial_frozen_shape_fails_closed(file, fault):
    (old_v, _), (old_i, _) = _sources(_FROZEN_GRADER)
    original = old_v if file == "verifier" else old_i
    old = (seam._VERIFIER_CLASS_OLD if file == "verifier"
           else seam._INVESTIGATOR_PREFIX_OLD).encode()
    new = (seam._VERIFIER_CLASS_NEW if file == "verifier"
           else seam._INVESTIGATOR_PREFIX_NEW).encode()
    if fault == "unknown":
        changed = (old.replace(b"def ", b"def_ ", 1) if file == "verifier"
                   else old.replace(b"safe_url = ", b"source_url = ", 1))
        original = original.replace(old, changed, 1)
    elif fault == "partial":
        original = original.replace(old, new, 1)
    elif fault == "crlf":
        original = original.replace(b"\n", b"\r\n")
    else:
        original = b"\xff"
    with pytest.raises(seam.TypedRefusalSeamError):
        (seam.convert_verification_seam if file == "verifier"
         else seam.convert_investigator_seam)(original)


def test_matching_text_moved_to_another_function_fails_closed():
    (original, _), _ = _sources()
    old = seam._VERIFIER_FETCH_OLD.encode()
    assert original.count(old) == 1
    moved = original.replace(old, b"        pass\n", 1)
    moved += b"\nasync def _unrelated_fetch(resp):\n" + old
    with pytest.raises(seam.TypedRefusalSeamError, match="outside helper"):
        seam.convert_verification_seam(moved)


def test_homepage_catch_must_match_frozen_helper_and_full_correction():
    (original, corrected), _ = _sources()
    old_catch = seam._VERIFIER_CATCH_OLD.encode()
    new_catch = seam._VERIFIER_CATCH_NEW.encode()
    assert original.count(old_catch) == corrected.count(new_catch) == 1
    with pytest.raises(seam.TypedRefusalSeamError, match="partial"):
        seam.convert_verification_seam(original.replace(old_catch, new_catch, 1))
    with pytest.raises(seam.TypedRefusalSeamError, match="unrecognized"):
        seam.convert_verification_seam(original.replace(
            old_catch, old_catch.replace(b'"provider_error"', b'"other_error"'), 1,
        ))
    moved = original.replace(
        old_catch, b"    except _ArenaProviderFetchError:\n        raise\n", 1,
    )
    moved += b"\nasync def _foreign_helper():\n    try:\n        pass\n" + old_catch
    with pytest.raises(seam.TypedRefusalSeamError, match="outside helper"):
        seam.convert_verification_seam(moved)


def _base_sources():
    (old_v, _), (old_i, _) = _sources(_FROZEN_GRADER)
    paths = (composer.GRADER_FILE, *composer.ENTRYPOINT_FILES)
    files = {name: (ROOT / ("lab_arena/scorer_entrypoint.py"
                           if name == "scorer_entrypoint.py" else name)).read_bytes()
             for name in paths}
    files[composer.GRADER_FILE] = old_i
    files[composer.VERIFICATION_FILE] = old_v
    return files


def test_composer_opt_in_preserves_other_overlay_files():
    files = _base_sources()
    default = composer.expected_overlay({k: v for k, v in files.items()
                                         if k != composer.VERIFICATION_FILE})
    corrected = composer.expected_overlay(files, authenticated_page_refusal=True)
    assert corrected[composer.VERIFICATION_FILE] == _sources()[0][1]
    assert corrected[composer.GRADER_FILE] == seam.convert_investigator_seam(
        default[composer.GRADER_FILE]
    )
    assert {k: v for k, v in corrected.items()
            if k not in {composer.GRADER_FILE, composer.VERIFICATION_FILE}} == {
                k: v for k, v in default.items() if k != composer.GRADER_FILE
            }
    with pytest.raises(composer.CompositionError):
        composer.expected_overlay(files)


@pytest.mark.parametrize("verify_existing", [False, True])
def test_build_and_verify_record_same_explicit_option(monkeypatch, capsys, verify_existing):
    grading = "registry.example/reviewer@sha256:" + "a" * 64
    runtime = "registry.example/reviewer@sha256:" + "b" * 64
    argv = ["build_arena_reviewer_runtime.py", "--grading-image", grading,
            "--authenticated-page-refusal"]
    if verify_existing:
        argv += ["--verify-image", runtime]
    monkeypatch.setattr(composer.sys, "argv", argv)
    monkeypatch.setattr(composer, "checked_output", lambda *a, **k: "c" * 40)
    base = {"Architecture": "amd64", "Os": "linux", "Config": {},
            "RootFS": {"Layers": ["sha256:base"]}}
    candidate = copy.deepcopy(base)
    candidate["RootFS"]["Layers"].append("sha256:overlay")
    monkeypatch.setattr(composer, "image_inspect",
                        lambda image: base if image == grading else candidate)
    def extract(image, destination, **kwargs):
        assert kwargs["authenticated_page_refusal"] is True
        return _base_sources()
    monkeypatch.setattr(composer, "extract_frozen_sources", extract)
    verified = []
    monkeypatch.setattr(composer, "verify_saved_overlay",
                        lambda image, digest, files: verified.append(files))
    built = []
    monkeypatch.setattr(composer.subprocess, "run",
                        lambda command, **kw: built.append(command))
    assert composer.main() == 0
    evidence = json.loads(capsys.readouterr().out)
    assert evidence["grading_corrections"] == [seam.CORRECTION_ID]
    assert composer.VERIFICATION_FILE in verified[0]
    assert len(built) == (0 if verify_existing else 1)


def test_all_correction_flags_keep_docker_tag_within_limit(monkeypatch, capsys):
    repo = "493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model"
    grading = repo + "@sha256:" + "a" * 64
    monkeypatch.setattr(composer.sys, "argv", [
        "build_arena_reviewer_runtime.py", "--grading-image", grading,
        "--complete-negative-finality", "--source-access-denied",
        "--pending-final-correction", "--failed-attribute-source-recovery",
        "--authenticated-page-refusal",
    ])
    monkeypatch.setattr(composer, "checked_output", lambda *a, **k: "c" * 40)
    base = {"Architecture": "amd64", "Os": "linux", "Config": {},
            "RootFS": {"Layers": ["sha256:base"]}}
    candidate = copy.deepcopy(base)
    candidate["RootFS"]["Layers"].append("sha256:overlay")
    monkeypatch.setattr(composer, "image_inspect",
                        lambda image: base if image == grading else candidate)
    monkeypatch.setattr(composer, "extract_frozen_sources",
                        lambda *_a, **_k: _base_sources())
    monkeypatch.setattr(composer, "expected_overlay",
                        lambda *_a, **_k: {composer.GRADER_FILE: b"test"})
    monkeypatch.setattr(composer, "verify_saved_overlay", lambda *_a: None)
    built = []
    monkeypatch.setattr(composer.subprocess, "run",
                        lambda command, **_kw: built.append(command))
    assert composer.main() == 0
    tag = next(command[command.index("-t") + 1] for command in built
               if command[:2] == ["docker", "build"])
    assert tag.startswith(repo + ":reviewer-")
    assert len(tag.rsplit(":", 1)[1]) <= 128
    assert tag.endswith("-refusal")
