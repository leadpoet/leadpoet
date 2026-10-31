"""Protocol repair composition preserves every unrelated frozen byte."""
import ast
from pathlib import Path

import pytest
from tests.reviewer_frozen_sources import frozen_source

from scripts import arena_reviewer_protocol_repair_seam as seam
from scripts import build_arena_reviewer_runtime as composer

ROOT = Path(__file__).resolve().parents[1]


# Exact CI-green native-budget postimage; historical overlays must not track
# the evolving current investigator or require Git history in test installs.
_FROZEN_GRADER = frozen_source(
    "native_industry_budget", "qualification/scoring/company_evidence_investigator.py",
)


def _sources(*, reserved_industry_turns=True):
    repaired = _FROZEN_GRADER
    if not reserved_industry_turns:
        repaired = repaired.replace(seam._NATIVE_REPAIR.encode(), seam._REPAIR.encode(), 1)
        repaired = repaired.replace(
            b"for _turn in range(reasoning_turn_limit + 1):",
            b"for _turn in range(MAX_REASONING_TURNS + 1):", 1)
    repair = seam._NATIVE_REPAIR if reserved_industry_turns else seam._REPAIR
    assert repaired.count(seam._HELPER.encode()) == repaired.count(repair.encode()) == 1
    original = repaired.replace(seam._HELPER.encode(), b"", 1).replace(repair.encode(), b"", 1)
    return original, repaired


@pytest.mark.parametrize("reserved_industry_turns", [False, True])
def test_exact_two_insertions_only_and_idempotent(reserved_industry_turns):
    original, repaired = _sources(reserved_industry_turns=reserved_industry_turns)
    repair = seam._NATIVE_REPAIR if reserved_industry_turns else seam._REPAIR
    assert seam.convert_protocol_repair_seam(original) == repaired
    assert seam.convert_protocol_repair_seam(repaired) == repaired
    assert repaired.replace(seam._HELPER.encode(), b"", 1).replace(repair.encode(), b"", 1) == original
    assert ast.parse(repaired)


@pytest.mark.parametrize("fault", ["helper_only", "loop_only", "helper_drift", "loop_drift",
                                   "duplicate_helper", "duplicate_loop", "missing_loop",
                                   "renamed_investigation", "moved_loop", "wrong_repair_bound",
                                   "unknown_turn_bound", "crlf", "utf8"])
def test_unknown_or_partial_source_is_rejected(fault):
    original, repaired = _sources()
    source = original
    if fault == "helper_only":
        source = original.replace(b"def _safe_https_url", seam._HELPER.encode() + b"def _safe_https_url", 1)
    elif fault == "loop_only":
        source = original.replace(seam._LOOP_OLD.encode(), (seam._REPAIR + seam._LOOP_OLD).encode(), 1)
    elif fault == "helper_drift":
        source = repaired.replace(b'len(value) > 500', b'len(value) > 501', 1)
    elif fault == "loop_drift":
        source = repaired.replace(b'incomplete_submit_retries < MAX_INCOMPLETE_SUBMIT_RETRIES',
                                  b'incomplete_submit_retries <= MAX_INCOMPLETE_SUBMIT_RETRIES')
    elif fault == "duplicate_helper":
        source = repaired + b"\n" + seam._HELPER.encode()
    elif fault == "duplicate_loop":
        source = original + b"\nasync def another_loop():\n" + seam._LOOP_OLD.encode()
    elif fault == "missing_loop":
        source = original.replace(seam._LOOP_OLD.encode(), b"                pass\n", 1)
    elif fault == "renamed_investigation":
        source = original.replace(b'async def investigate_company_evidence(', b'async def unrelated(', 1)
    elif fault == "moved_loop":
        source = original.replace(seam._LOOP_OLD.encode(), b"                pass\n", 1)
        source += b"\nasync def another_loop():\n" + seam._LOOP_OLD.encode()
    elif fault == "crlf":
        source = original.replace(b"\n", b"\r\n")
    elif fault == "wrong_repair_bound":
        source = repaired.replace(seam._NATIVE_REPAIR.encode(), seam._REPAIR.encode(), 1)
    elif fault == "unknown_turn_bound":
        source = original.replace(b"for _turn in range(reasoning_turn_limit + 1):",
                                  b"for _turn in range(arbitrary_limit + 1):", 1)
    else:
        source = b"\xff"
    with pytest.raises(seam.ProtocolRepairSeamError):
        seam.convert_protocol_repair_seam(source)


@pytest.mark.parametrize("prior_flags", [False, True])
def test_builder_flag_is_explicit_and_preserves_existing_composition(prior_flags):
    original, _ = _sources()
    names = [composer.GRADER_FILE, *composer.ENTRYPOINT_FILES]
    if prior_flags:
        names += [composer.FINALITY_FILE, composer.VERIFICATION_FILE]
    sources = {
        name: (ROOT / ("lab_arena/scorer_entrypoint.py"
                       if name == "scorer_entrypoint.py" else name)).read_bytes()
        for name in names
    }
    sources[composer.GRADER_FILE] = original
    flags = dict.fromkeys(("complete_negative_finality", "source_access_denied",
                           "pending_final_correction", "failed_attribute_source_recovery",
                           "authenticated_page_refusal", "completed_investigation_finality"), prior_flags)
    before = composer.expected_overlay(sources, **flags)
    after = composer.expected_overlay(sources, single_tool_protocol_repair=True, **flags)
    assert set(before) == set(after)
    assert after[composer.GRADER_FILE] == seam.convert_protocol_repair_seam(before[composer.GRADER_FILE])
    assert all(after[name] == before[name] for name in before if name != composer.GRADER_FILE)


def test_current_unreviewed_tool_schema_is_not_silently_transplanted():
    current = (ROOT / composer.GRADER_FILE).read_bytes()
    assert current != _FROZEN_GRADER
    with pytest.raises(seam.ProtocolRepairSeamError):
        seam.convert_protocol_repair_seam(current)
