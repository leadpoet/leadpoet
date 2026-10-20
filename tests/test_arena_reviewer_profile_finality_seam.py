"""Only the reviewed optional-profile control-flow correction can be composed."""
import ast
import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts import build_arena_reviewer_runtime as composer
from scripts import arena_reviewer_profile_finality_seam as seam


ROOT = Path(__file__).resolve().parents[1]


def _sources():
    corrected = (ROOT / composer.FINALITY_FILE).read_bytes()
    original = corrected
    assert original.count(seam._HELPER.encode()) == 1
    original = original.replace(seam._HELPER.encode(), b"", 1)
    for old, new in seam.DELTAS:
        assert original.count(new.encode()) == 1
        original = original.replace(new.encode(), old.encode(), 1)
    return original, corrected


def test_exact_frozen_function_and_predicate_identity_with_only_reviewed_bytes():
    original, corrected = _sources()
    tree = ast.parse(original)
    function = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == seam.FUNCTION)
    assert hashlib.sha256(ast.get_source_segment(original.decode(), function).encode()).hexdigest() == seam.ORIGINAL_FUNCTION_SHA256
    predicate = next(n.value for n in ast.walk(function)
                     if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "malformed_profile_exhausted" for t in n.targets))
    helper = ast.parse(seam._HELPER).body[0]
    returned = next(n.value for n in helper.body if isinstance(n, ast.Return))
    assert ast.dump(predicate) == ast.dump(returned)
    assert seam.convert_profile_finality_seam(original) == corrected
    assert seam.convert_profile_finality_seam(corrected) == corrected
    outside_before, outside_after = original, corrected.replace(seam._HELPER.encode(), b"", 1)
    for old, new in seam.DELTAS:
        outside_before = outside_before.replace(old.encode(), b"# REVIEWED CORRECTION\n", 1)
        outside_after = outside_after.replace(new.encode(), b"# REVIEWED CORRECTION\n", 1)
    assert outside_before == outside_after
    wrapped = b"# Frozen header\n" + original + b"\n# Frozen footer\n"
    assert seam.convert_profile_finality_seam(wrapped) == b"# Frozen header\n" + corrected + b"\n# Frozen footer\n"


@pytest.mark.parametrize("fault", ["unknown", "missing", "duplicate", "partial", "helper", "crlf", "utf8"])
def test_unknown_ambiguous_or_partial_source_fails_closed(fault):
    original, corrected = _sources()
    if fault == "unknown":
        original = original.replace(b"positive_semantic_review\n", b"not positive_semantic_review\n", 1)
    elif fault == "missing":
        original = original.replace(("async def " + seam.FUNCTION + "(").encode(), b"async def other_function(", 1)
    elif fault == "duplicate":
        node = next(n for n in ast.parse(original).body if isinstance(n, ast.AsyncFunctionDef) and n.name == seam.FUNCTION)
        original += b"\n" + ast.get_source_segment(original.decode(), node).encode() + b"\n"
    elif fault == "partial":
        original = original.replace(seam.DELTAS[0][0].encode(), seam.DELTAS[0][1].encode(), 1)
    elif fault == "helper":
        original = corrected.replace(b'Apply the existing completed-search rule', b'Alter the existing completed-search rule', 1)
    elif fault == "crlf":
        original = original.replace(b"\n", b"\r\n")
    else:
        original = b"\xff"
    with pytest.raises(seam.ProfileFinalitySeamError):
        seam.convert_profile_finality_seam(original)


def _base_sources():
    files = {name: (ROOT / ("lab_arena/scorer_entrypoint.py" if name == "scorer_entrypoint.py" else name)).read_bytes()
             for name in (composer.GRADER_FILE, *composer.ENTRYPOINT_FILES)}
    files[composer.FINALITY_FILE] = _sources()[0]
    return files


@pytest.mark.parametrize("binding", [
    "async def _completed_malformed_profile_review(*args): return True",
    "_completed_malformed_profile_review = lambda *args: True",
    "import another_module as _completed_malformed_profile_review",
    "if True:\n    _completed_malformed_profile_review = True",
])
@pytest.mark.parametrize("already_corrected", [False, True])
def test_unknown_helper_bindings_fail_closed(binding, already_corrected):
    source = _sources()[int(already_corrected)] + b"\n" + binding.encode() + b"\n"
    with pytest.raises(seam.ProfileFinalitySeamError):
        seam.convert_profile_finality_seam(source)


def test_explicit_option_changes_only_frozen_finality_and_keeps_other_overlay_bytes():
    files = _base_sources()
    previous = composer.expected_overlay({k: v for k, v in files.items() if k != composer.FINALITY_FILE})
    corrected = composer.expected_overlay(files, completed_malformed_profile=True)
    assert corrected[composer.FINALITY_FILE] == _sources()[1]
    assert {k: v for k, v in corrected.items() if k != composer.FINALITY_FILE} == previous
    with pytest.raises(composer.CompositionError):
        composer.expected_overlay(files)


@pytest.mark.parametrize("verify_existing", [False, True])
def test_build_and_verify_require_and_record_the_same_option(monkeypatch, capsys, verify_existing):
    grading = "registry.example/reviewer@sha256:" + "a" * 64
    runtime = "registry.example/reviewer@sha256:" + "b" * 64
    argv = ["build_arena_reviewer_runtime.py", "--grading-image", grading, "--completed-malformed-profile"]
    if verify_existing:
        argv += ["--verify-image", runtime]
    monkeypatch.setattr(composer.sys, "argv", argv)
    monkeypatch.setattr(composer, "checked_output", lambda *a, **k: "c" * 40)
    base = {"Architecture": "amd64", "Os": "linux", "Config": {}, "RootFS": {"Layers": ["sha256:base"]}}
    candidate = copy.deepcopy(base)
    candidate["RootFS"]["Layers"].append("sha256:overlay")
    monkeypatch.setattr(composer, "image_inspect", lambda image: base if image == grading else candidate)
    def extract(image, destination, **kwargs):
        assert kwargs == {"complete_negative_finality": False, "source_access_denied": False,
                          "failed_attribute_source_recovery": False, "completed_malformed_profile": True}
        return _base_sources()
    monkeypatch.setattr(composer, "extract_frozen_sources", extract)
    verified = []
    monkeypatch.setattr(composer, "verify_saved_overlay", lambda image, digest, files: verified.append(files))
    built = []
    monkeypatch.setattr(composer.subprocess, "run", lambda command, **kw: built.append(command))
    assert composer.main() == 0
    evidence = json.loads(capsys.readouterr().out)
    assert evidence["grading_corrections"] == [seam.CORRECTION_ID]
    assert evidence["original_files"][composer.FINALITY_FILE] == hashlib.sha256(_sources()[0]).hexdigest()
    assert verified[0][composer.FINALITY_FILE] == _sources()[1]
    assert len(built) == (0 if verify_existing else 1)
