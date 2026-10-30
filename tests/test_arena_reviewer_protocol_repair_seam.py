"""Protocol repair composition preserves every unrelated frozen byte."""
import ast
import hashlib
from pathlib import Path

import pytest

from scripts import arena_reviewer_protocol_repair_seam as seam
from scripts import build_arena_reviewer_runtime as composer

ROOT = Path(__file__).resolve().parents[1]


def historical_protocol_source(source):
    """Use the reviewed historical helper only in frozen-source test inputs."""
    text = source.decode("utf-8")
    lines = text.splitlines(keepends=True)
    helpers = [node for node in ast.parse(text).body
               if isinstance(node, ast.FunctionDef)
               and node.name == "_valid_multi_research_reply"]
    assert len(helpers) == 1
    helper = helpers[0]
    current = "".join(lines[helper.lineno - 1:helper.end_lineno])
    # Fail when either fixture input or the historical repair changes. This
    # does not teach the frozen converter to accept the current offset helper.
    assert hashlib.sha256(current.encode()).hexdigest() == (
        "241ab4e3c5bcecf2bbf89aed49da9f49dc739225f24e8048f2b9c85e276be15b"
    )
    assert hashlib.sha256(seam._HELPER.encode()).hexdigest() == (
        "57ba91aff2d9d1a914624712965859c46c1f2f233e2ccae7d846b87734dd8de3"
    )
    lines[helper.lineno - 1:helper.end_lineno] = [seam._HELPER.rstrip("\n") + "\n"]
    return "".join(lines).encode("utf-8")


def _sources():
    repaired = historical_protocol_source((ROOT / composer.GRADER_FILE).read_bytes())
    assert repaired.count(seam._HELPER.encode()) == repaired.count(seam._REPAIR.encode()) == 1
    original = repaired.replace(seam._HELPER.encode(), b"", 1).replace(seam._REPAIR.encode(), b"", 1)
    return original, repaired


def test_current_offset_helper_still_fails_closed_in_frozen_converter():
    current = (ROOT / composer.GRADER_FILE).read_bytes()
    with pytest.raises(seam.ProtocolRepairSeamError, match="partially corrected protocol source"):
        seam.convert_protocol_repair_seam(current)
    unknown = current.replace(b"len(value) > 500", b"len(value) > 501", 1)
    with pytest.raises(seam.ProtocolRepairSeamError, match="partially corrected protocol source"):
        seam.convert_protocol_repair_seam(unknown)
    with pytest.raises(AssertionError):
        historical_protocol_source(unknown)


def test_exact_two_insertions_only_and_idempotent():
    original, repaired = _sources()
    assert seam.convert_protocol_repair_seam(original) == repaired
    assert seam.convert_protocol_repair_seam(repaired) == repaired
    assert repaired.replace(seam._HELPER.encode(), b"", 1).replace(seam._REPAIR.encode(), b"", 1) == original
    assert ast.parse(repaired)


@pytest.mark.parametrize("fault", ["helper_only", "loop_only", "helper_drift", "loop_drift",
                                   "duplicate_helper", "duplicate_loop", "missing_loop",
                                   "renamed_investigation", "moved_loop", "crlf", "utf8"])
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
