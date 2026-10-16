from __future__ import annotations

from pathlib import Path

import pytest

from arena_reviewer_runtime.admission import ReviewerAdmission
from scripts.arena_reviewer_admission_seam import (
    AdmissionSeamError, _REPLACEMENTS, convert_admission_seam,
)


def _admission(*, loaded=False):
    return ReviewerAdmission(
        admission_seconds=110.0, judgment_reserve_seconds=30.0,
        serialization_grace_seconds=30.0, loaded_research=loaded,
    )


def test_reserve_retains_original_tools_but_late_settlement_restricts_them():
    admission = _admission()
    ordinary = admission.next_turn(elapsed=79.9, serialization_retry_pending=False)
    reserve = admission.next_turn(elapsed=80.0, serialization_retry_pending=False)
    assert (ordinary.mode, ordinary.restrict_tools) == ("ordinary", False)
    assert (reserve.mode, reserve.restrict_tools) == ("submit_only", False)
    admission.tool_settled(name="fetch_page", has_loaded_evidence=True)
    late = admission.next_turn(elapsed=180.0, serialization_retry_pending=False)
    assert (late.mode, late.restrict_tools) == ("submit_only", True)
    assert admission.next_turn(
        elapsed=181.0, serialization_retry_pending=False,
    ).mode == "stop"


@pytest.mark.parametrize("tool", ["fetch_page", "search_web", "submit_findings"])
def test_no_late_allowance_without_loaded_evidence(tool):
    admission = _admission()
    admission.tool_settled(name=tool, has_loaded_evidence=False)
    assert admission.next_turn(
        elapsed=110.0, serialization_retry_pending=False,
    ).mode == "stop"


def test_loaded_research_does_not_authorize_a_later_unrelated_turn():
    admission = _admission(loaded=True)
    assert admission.next_turn(
        elapsed=40.0, serialization_retry_pending=False,
    ).mode == "ordinary"
    assert admission.next_turn(
        elapsed=110.0, serialization_retry_pending=False,
    ).mode == "stop"


def test_serialization_repair_keeps_its_existing_expiry():
    admission = _admission()
    allowed = admission.next_turn(elapsed=139.9, serialization_retry_pending=True)
    expired = admission.next_turn(elapsed=140.0, serialization_retry_pending=True)
    assert allowed.mode == "submit_only"
    assert not allowed.restrict_tools
    assert not allowed.serialization_expired
    assert expired.serialization_expired


def _hooked_source():
    return (Path(__file__).resolve().parents[1] / "qualification/scoring"
            / "company_evidence_investigator.py").read_bytes()


def _unhooked_source(*, repaired):
    # Construct both reviewed statement shapes from the real current module.
    # This keeps the fixture small and does not substitute any grading body.
    text = _hooked_source().decode()
    for index, (alternatives, replacement) in enumerate(_REPLACEMENTS):
        old = alternatives[int(repaired) if 0 < index < 7 else 0]
        assert text.count(replacement) == 1
        text = text.replace(replacement, old, 1)
    compile(text, "unhooked_investigator.py", "exec")
    return text.encode()


@pytest.mark.parametrize("repaired", [False, True])
def test_both_admission_shapes_convert_to_exact_same_source(repaired):
    assert convert_admission_seam(_unhooked_source(repaired=repaired)) == _hooked_source()


def test_hooked_source_is_returned_byte_for_byte():
    assert convert_admission_seam(_hooked_source()) == _hooked_source()


def test_converter_preserves_grading_changes_in_its_input():
    source = _unhooked_source(repaired=False)
    source = source.replace(b"REASONING_MAX_TOKENS = 3000", b"REASONING_MAX_TOKENS = 2999")
    source = source.replace(b"You are a bounded company evidence investigator.",
                            b"Frozen source-specific grading prompt.")
    expected = _hooked_source().replace(
        b"REASONING_MAX_TOKENS = 3000", b"REASONING_MAX_TOKENS = 2999",
    ).replace(b"You are a bounded company evidence investigator.",
              b"Frozen source-specific grading prompt.")
    assert convert_admission_seam(source) == expected


@pytest.mark.parametrize("corruption", ["unknown", "partial", "duplicate"])
def test_converter_rejects_unknown_partial_or_ambiguous_shapes(corruption):
    source = _unhooked_source(repaired=False)
    if corruption == "unknown":
        source = source.replace(b"and not late_serialization_retry_pending",
                                b"and False")
    elif corruption == "partial":
        source = source.replace(b"import aiohttp\n\n", b"import aiohttp\n\n"
            b"from arena_reviewer_runtime.admission import ReviewerAdmission\n", 1)
    else:
        source += b"\nimport aiohttp\n\n"
    with pytest.raises(AdmissionSeamError):
        convert_admission_seam(source)
