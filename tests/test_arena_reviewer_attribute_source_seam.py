"""The shared-review correction changes only two approved frozen blocks."""

import ast
import copy
import hashlib
import json
import gzip
from pathlib import Path

import pytest

from scripts import build_arena_reviewer_runtime as composer
from scripts.arena_reviewer_attribute_source_seam import (
    AttributeSourceSeamError, CORRECTION_ID, DELTAS, convert_attribute_source_seam,
)
from tests.reviewer_frozen_sources import frozen_source


ROOT = Path(__file__).resolve().parents[1]
_FROZEN_POSTIMAGE = frozen_source("attribute_source", composer.FINALITY_FILE)


# Exact CI-green native-budget postimage; historical overlays must not track
# the evolving current investigator or require Git history in test installs.
_FROZEN_GRADER = frozen_source(
    "native_industry_budget", "qualification/scoring/company_evidence_investigator.py",
)


def _sources():
    # Preserve the exact reviewed frozen function hashes as current grading
    # evolves. The converter must still change only its two owned blocks.
    current = _FROZEN_POSTIMAGE
    original = current
    for _, old, new in DELTAS:
        assert original.count(new.encode()) == 1
        original = original.replace(new.encode(), old.encode(), 1)
    return original, current


def test_historical_fixture_rejects_changed_source_bytes(monkeypatch, tmp_path):
    from tests import reviewer_frozen_sources
    source = _FROZEN_POSTIMAGE
    fixture = tmp_path / "source.gz"
    fixture.write_bytes(gzip.compress(source, mtime=0))
    (tmp_path / "sources.json").write_text(json.dumps({"probe": {"files": {
        composer.FINALITY_FILE: {"fixture": fixture.name,
                                "sha256": hashlib.sha256(source).hexdigest()},
    }}}))
    monkeypatch.setattr(reviewer_frozen_sources, "ROOT", tmp_path)
    assert frozen_source("probe", composer.FINALITY_FILE) == source
    fixture.write_bytes(gzip.compress(source + b"\n# unreviewed grading bytes\n", mtime=0))
    with pytest.raises(AssertionError, match=composer.FINALITY_FILE):
        frozen_source("probe", composer.FINALITY_FILE)


def test_exact_frozen847_function_hashes_and_only_approved_bytes_change():
    original = (ROOT / 'tests/fixtures/arena_reviewer/frozen847_functions.py.txt').read_bytes()
    current = original
    for _, old, new in DELTAS:
        assert current.count(old.encode()) == 1
        current = current.replace(old.encode(), new.encode(), 1)
    expected_hashes = (
        "5a209cd382326ec3179ba2ba775dedee395ac741a25c90f33726c0b6df4d2698",
        "90a3903cfa59be5792e491610ad36672b92b5c83666c8c2458b90a579263a393",
    )
    tree = ast.parse(original)
    for (name, _, _), digest in zip(DELTAS, expected_hashes):
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        assert hashlib.sha256((ast.get_source_segment(original.decode(), node) + "\n").encode()).hexdigest() == digest
    assert convert_attribute_source_seam(original) == current
    assert convert_attribute_source_seam(current) == current
    before, after = original, current
    for _, old, new in DELTAS:
        before = before.replace(old.encode(), b"# APPROVED BLOCK\n", 1)
        after = after.replace(new.encode(), b"# APPROVED BLOCK\n", 1)
    assert before == after
    frozen = b"# Historical frozen module bytes.\n" + original + b"\n# Frozen footer.\n"
    assert convert_attribute_source_seam(frozen) == b"# Historical frozen module bytes.\n" + current + b"\n# Frozen footer.\n"


@pytest.mark.parametrize("fault", ["unknown", "missing", "duplicate", "partial", "mixed", "crlf", "utf8"])
def test_unknown_ambiguous_and_partially_corrected_source_fails_closed(fault):
    original, _ = _sources()
    name, old, new = DELTAS[0]
    if fault == "unknown":
        original = original.replace(old.encode(), old.replace('"geography"', '"unknown"').encode(), 1)
    elif fault == "missing":
        original = original.replace(("def " + name + "(").encode(), b"def other_function(", 1)
    elif fault == "duplicate":
        node = next(n for n in ast.parse(original).body if isinstance(n, ast.FunctionDef) and n.name == name)
        original += b"\n" + ast.get_source_segment(original.decode(), node).encode() + b"\n"
    elif fault == "partial":
        original = original.replace(old.encode(), (old + new).encode(), 1)
    elif fault == "mixed":
        original = original.replace(old.encode(), new.encode(), 1)
    elif fault == "crlf":
        original = original.replace(b"\n", b"\r\n")
    else:
        original = b"\xff"
    with pytest.raises(AttributeSourceSeamError):
        convert_attribute_source_seam(original)


def _base_sources():
    files = {name: (ROOT / ("lab_arena/scorer_entrypoint.py" if name == "scorer_entrypoint.py" else name)).read_bytes()
             for name in (composer.GRADER_FILE, *composer.ENTRYPOINT_FILES)}
    files[composer.GRADER_FILE] = _FROZEN_GRADER
    files[composer.FINALITY_FILE] = _sources()[0]
    return files


def test_explicit_opt_in_retains_all_prior_corrections_and_every_other_frozen_byte():
    from scripts.arena_reviewer_finality_seam import _CORRECTED_GUARD, _ORIGINAL_GUARD
    from scripts.arena_reviewer_source_access_seam import FINALITY_DELTAS, INVESTIGATOR_DELTAS
    from scripts.arena_reviewer_budget_seam import _OLD, _NEW
    files = _base_sources()
    files[composer.FINALITY_FILE] = files[composer.FINALITY_FILE].replace(_CORRECTED_GUARD.encode(), _ORIGINAL_GUARD.encode(), 1)
    for path, deltas in ((composer.FINALITY_FILE, FINALITY_DELTAS), (composer.GRADER_FILE, INVESTIGATOR_DELTAS)):
        for _, old, new in deltas:
            files[path] = files[path].replace(new.encode(), old.encode(), 1)
    files[composer.GRADER_FILE] = files[composer.GRADER_FILE].replace(_NEW.encode(), _OLD.encode(), 1)
    kwargs = dict(complete_negative_finality=True, source_access_denied=True, pending_final_correction=True)
    previous = composer.expected_overlay(files, **kwargs)
    corrected = composer.expected_overlay(files, **kwargs, failed_attribute_source_recovery=True)
    assert corrected[composer.FINALITY_FILE] == convert_attribute_source_seam(previous[composer.FINALITY_FILE])
    assert {k: v for k, v in corrected.items() if k != composer.FINALITY_FILE} == {
        k: v for k, v in previous.items() if k != composer.FINALITY_FILE
    }
    assert composer.expected_overlay(_base_sources(), failed_attribute_source_recovery=True)[composer.FINALITY_FILE] == _sources()[1]
    with pytest.raises(composer.CompositionError):
        composer.expected_overlay(_base_sources())


@pytest.mark.parametrize("verify_existing", [False, True])
def test_build_and_verify_record_the_same_explicit_flag(monkeypatch, capsys, verify_existing):
    grading = "registry.example/reviewer@sha256:" + "a" * 64
    runtime = "registry.example/reviewer@sha256:" + "b" * 64
    argv = ["build_arena_reviewer_runtime.py", "--grading-image", grading, "--failed-attribute-source-recovery"]
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
                          "failed_attribute_source_recovery": True}
        return _base_sources()
    monkeypatch.setattr(composer, "extract_frozen_sources", extract)
    verified = []
    monkeypatch.setattr(composer, "verify_saved_overlay", lambda image, digest, files: verified.append(files))
    built = []
    monkeypatch.setattr(composer.subprocess, "run", lambda command, **kw: built.append(command))
    assert composer.main() == 0
    evidence = json.loads(capsys.readouterr().out)
    assert evidence["grading_corrections"] == [CORRECTION_ID]
    assert evidence["original_files"][composer.FINALITY_FILE] == hashlib.sha256(_sources()[0]).hexdigest()
    assert verified[0][composer.FINALITY_FILE] == _sources()[1]
    assert len(built) == (0 if verify_existing else 1)
