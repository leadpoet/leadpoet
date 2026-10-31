from __future__ import annotations

from pathlib import Path

import pytest
from tests.reviewer_frozen_sources import frozen_source

from arena_reviewer_runtime.admission import ReviewerAdmission
from scripts.arena_reviewer_admission_seam import (
    AdmissionSeamError, _CORRECTION_ELIGIBILITY_ARGUMENT, _REPLACEMENTS,
    convert_admission_seam,
)


# Exact CI-green native-budget postimage; historical overlays must not track
# the evolving current investigator or require Git history in test installs.
_FROZEN_GRADER = frozen_source(
    "native_industry_budget", "qualification/scoring/company_evidence_investigator.py",
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


@pytest.mark.parametrize("elapsed", [110.0, 180.0])
def test_only_existing_final_correction_can_follow_a_late_judgment(elapsed):
    admission = _admission()
    admission.tool_settled(name="submit_findings", has_loaded_evidence=True)
    final = admission.next_turn(
        elapsed=elapsed, serialization_retry_pending=False,
        final_correction_eligible=True,
    )
    assert (final.mode, final.restrict_tools) == ("submit_only", True)
    # The caller owns the one existing correction turn. This allowance does
    # not create a following ordinary turn or revive a research allowance.
    assert admission.next_turn(
        elapsed=elapsed, serialization_retry_pending=False,
    ).mode == "stop"


def test_final_correction_does_not_extend_serialization_retry_grace():
    final = _admission().next_turn(
        elapsed=140.0, serialization_retry_pending=True,
        final_correction_eligible=True,
    )
    assert final.serialization_expired


def _hooked_source():
    return _FROZEN_GRADER


def _unhooked_source(*, repaired):
    # Construct both reviewed shapes from the exact historical postimage.
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


@pytest.mark.parametrize("followup_state", ["False", "industry_gap_search_needed"])
def test_admission_conversion_does_not_own_industry_followup_state(followup_state):
    source = _unhooked_source(repaired=False).replace(
        b"industry_followup_pending = industry_gap_search_needed",
        ("industry_followup_pending = " + followup_state).encode(), 1)
    expected = _hooked_source().replace(
        b"industry_followup_pending = industry_gap_search_needed",
        ("industry_followup_pending = " + followup_state).encode(), 1)
    assert convert_admission_seam(source) == expected


def test_complete_hook_receives_only_existing_correction_eligibility():
    source = _hooked_source().replace(_CORRECTION_ELIGIBILITY_ARGUMENT.encode(), b"", 1)
    assert convert_admission_seam(source) == _hooked_source()


def test_correction_binding_does_not_permit_other_partial_hook_changes():
    source = _hooked_source().decode().replace(_CORRECTION_ELIGIBILITY_ARGUMENT, "", 1)
    alternatives, replacement = _REPLACEMENTS[1]
    source = source.replace(replacement, alternatives[0], 1)
    with pytest.raises(AdmissionSeamError):
        convert_admission_seam(source.encode())


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


def test_native_budget_fixture_is_exact_and_current_unreviewed_shape_stays_rejected():
    import hashlib
    from scripts.arena_reviewer_budget_seam import _NATIVE_NEW
    assert hashlib.sha256(_FROZEN_GRADER).hexdigest() == "fa65d603c7acc9d2363594061eb40812256c1be3c2490f16549c00b084b4cf94"
    assert _FROZEN_GRADER.count(_NATIVE_NEW.encode()) == 1
    assert convert_admission_seam(_FROZEN_GRADER) == _FROZEN_GRADER
    current = (Path(__file__).resolve().parents[1] / "qualification/scoring"
               / "company_evidence_investigator.py").read_bytes()
    assert current != _FROZEN_GRADER
    # The production converter still cannot rewrite an unreviewed source
    # shape. Pinning historical test data grants no new composition authority.
    with pytest.raises(AdmissionSeamError, match="unrecognized reviewer admission block"):
        convert_admission_seam(current)
