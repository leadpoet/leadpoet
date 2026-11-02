"""Source access repair preserves all other frozen grading bytes."""

import ast
from pathlib import Path

import pytest
from tests.reviewer_frozen_sources import frozen_source

from scripts.arena_reviewer_source_access_seam import (
    FINALITY_DELTAS, INVESTIGATOR_DELTAS, SourceAccessSeamError,
    convert_source_access_finality_seam, convert_source_access_seam,
)


ROOT = Path(__file__).resolve().parents[1]
CASES = [
    ("qualification/scoring/company_evidence_investigator.py", INVESTIGATOR_DELTAS,
     convert_source_access_seam),
    ("qualification/scoring/lead_scorer.py", FINALITY_DELTAS,
     convert_source_access_finality_seam),
]


def _sources(path, deltas):
    # This converter preserves historical grading bytes. Prospective source
    # timeout handling no longer contains its old exception-rethrow block.
    current = (frozen_source("native_industry_budget", path)
               if path.endswith("company_evidence_investigator.py")
               else (ROOT / path).read_bytes())
    original = current
    for _, old, new in deltas:
        assert original.count(new.encode()) == 1
        original = original.replace(new.encode(), old.encode(), 1)
    return original, current


@pytest.mark.parametrize("path,deltas,convert", CASES)
def test_exact_approved_blocks_are_the_only_byte_changes(path, deltas, convert):
    original, current = _sources(path, deltas)
    assert convert(original) == current
    assert convert(current) == current
    before, after = original, convert(original)
    for _, old, new in deltas:
        before = before.replace(old.encode(), b"# APPROVED BLOCK\n", 1)
        after = after.replace(new.encode(), b"# APPROVED BLOCK\n", 1)
    assert before == after
    # Unrelated frozen bytes stay intact; no current-module transplant.
    frozen = b"# historical frozen module\n" + original + b"\n# frozen footer\n"
    assert convert(frozen) == b"# historical frozen module\n" + current + b"\n# frozen footer\n"


@pytest.mark.parametrize("path,deltas,convert", CASES)
@pytest.mark.parametrize("fault", ["unknown", "duplicate", "missing", "partial", "crlf", "utf8"])
def test_unknown_or_ambiguous_source_fails_closed(path, deltas, convert, fault):
    original, _ = _sources(path, deltas)
    name, old, new = deltas[0]
    if fault == "unknown":
        original = original.replace(old.encode(), old.replace("return", "return  " ).encode(), 1)
        if "return" not in old:
            original = original.replace(old.encode(), old.replace('"source_gone"', '"unknown"').encode(), 1)
    elif fault == "duplicate":
        tree = ast.parse(original)
        node = next(node for node in tree.body
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name)
        original += b"\n" + ast.get_source_segment(original.decode(), node).encode() + b"\n"
    elif fault == "missing":
        original = original.replace(("def " + name + "(").encode(), b"def renamed_function(", 1)
    elif fault == "partial":
        original = original.replace(old.encode(), (old + new).encode(), 1)
    elif fault == "crlf":
        original = original.replace(b"\n", b"\r\n")
    elif fault == "utf8":
        original = b"\xff"
    with pytest.raises(SourceAccessSeamError):
        convert(original)


def test_partially_applied_investigator_correction_is_rejected():
    path, deltas, convert = CASES[0]
    original, _ = _sources(path, deltas)
    _, old, new = deltas[0]
    with pytest.raises(SourceAccessSeamError, match="partially corrected"):
        convert(original.replace(old.encode(), new.encode(), 1))


def test_source_access_and_existing_negative_finality_deltas_compose():
    from scripts.arena_reviewer_finality_seam import (
        _CORRECTED_GUARD, _ORIGINAL_GUARD, convert_finality_seam,
    )
    path, deltas, convert = CASES[1]
    original, current = _sources(path, deltas)
    frozen = original.replace(_CORRECTED_GUARD.encode(), _ORIGINAL_GUARD.encode(), 1)
    assert convert(convert_finality_seam(frozen)) == current
    assert convert_finality_seam(convert(frozen)) == current
