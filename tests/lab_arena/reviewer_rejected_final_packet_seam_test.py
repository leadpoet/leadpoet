"""The frozen reviewer receives only the rejected-final-packet correction."""
import ast
from copy import deepcopy
import hashlib
import json
import sys

import pytest
from tests.reviewer_frozen_sources import frozen_source

from qualification.scoring import company_evidence_investigator as investigator
from scripts import build_arena_reviewer_runtime as builder
from scripts.arena_reviewer_rejected_final_packet_seam import (
    CORRECTION_ID, GUARD, NEW, OLD, RejectedFinalPacketSeamError,
    convert_rejected_final_packet_seam as convert,
)
from tests import test_final_unproven_normalization as controls


# Exact CI-green native-budget postimage; historical overlays must not track
# the evolving current investigator or require Git history in test installs.
_FROZEN_GRADER = frozen_source(
    "native_industry_budget", "qualification/scoring/company_evidence_investigator.py",
)


def corrected_grader():
    # Reuse the exact upstream197048255 investigator already pinned for the
    # frozen brand overlay. Do not loosen the release converter's source guard.
    frozen = (builder.ROOT / 'tests/fixtures/arena_reviewer/frozen_brand_investigator.py.txt').read_bytes()
    assert hashlib.sha256(frozen).hexdigest() == '4ee4e5c966a6f1815c860a3b43e7cb151712eadf7ac7912e90e271cf06c19005'
    text = _FROZEN_GRADER.decode()
    node = next(n for n in ast.parse(text).body
                if isinstance(n, ast.AsyncFunctionDef) and n.name == 'investigate_company_evidence')
    lines = text.splitlines(keepends=True)
    return ''.join(lines[:node.lineno - 1]).encode() + frozen + ''.join(lines[node.end_lineno:]).encode()


def sources(**options):
    names = [builder.GRADER_FILE, *builder.ENTRYPOINT_FILES]
    if any(options.get(name) for name in (
        "complete_negative_finality", "source_access_denied",
        "failed_attribute_source_recovery", "completed_investigation_finality",
        "identity_schema_repair",
    )):
        names.append(builder.FINALITY_FILE)
    if options.get("authenticated_page_refusal"):
        names.append(builder.VERIFICATION_FILE)
    result = {
        name: (builder.ROOT / ("lab_arena/scorer_entrypoint.py"
                              if name == "scorer_entrypoint.py" else name)).read_bytes()
        for name in names
    }
    result[builder.GRADER_FILE] = corrected_grader()
    assert result[builder.GRADER_FILE].count(NEW.encode()) == 1
    result[builder.GRADER_FILE] = result[builder.GRADER_FILE].replace(
        NEW.encode(), OLD.encode(), 1,
    )
    return result


def test_exact_guard_is_the_only_changed_frozen_byte_and_is_idempotent():
    original = sources()[builder.GRADER_FILE]
    corrected = convert(original)
    assert corrected == corrected_grader()
    assert corrected.replace(GUARD.encode(), b"", 1) == original
    assert convert(corrected) == corrected


def test_current_industry_source_is_not_the_frozen_final_packet_preimage():
    with pytest.raises(RejectedFinalPacketSeamError):
        convert((builder.ROOT / builder.GRADER_FILE).read_bytes())


@pytest.mark.parametrize("fault", [
    "utf8", "syntax", "crlf", "missing_function", "duplicate_function",
    "shadowed_function", "partial_guard", "unknown_guard", "duplicate_guard",
    "changed_precedence", "changed_completion", "quoted_block", "quoted_guard",
])
def test_unknown_partial_or_shadowed_source_fails_closed(fault):
    original = sources()[builder.GRADER_FILE]
    text = original.decode()
    if fault == "utf8":
        bad = b"\xff"
    elif fault == "syntax":
        bad = b"async def investigate_company_evidence(:"
    elif fault == "crlf":
        bad = text.replace("\n", "\r\n").encode()
    elif fault == "missing_function":
        bad = text.replace("async def investigate_company_evidence(", "async def other(", 1).encode()
    elif fault == "duplicate_function":
        bad = (text + "\nasync def investigate_company_evidence():\n    pass\n").encode()
    elif fault == "shadowed_function":
        bad = (text + "\ninvestigate_company_evidence = None\n").encode()
    elif fault == "partial_guard":
        bad = text.replace(OLD, NEW.replace(
            "                            failure_reason = MALFORMED_RESPONSE_FAILURE_REASON\n",
            "                            pass\n", 1), 1).encode()
    elif fault == "unknown_guard":
        bad = convert(original).replace(b"if rejected and not failure_reason:",
                                        b"if rejected or not failure_reason:", 1)
    elif fault == "duplicate_guard":
        bad = convert(original).replace(GUARD.encode(), (GUARD + GUARD).encode(), 1)
    elif fault == "changed_precedence":
        bad = text.replace("                            if unresolved_source\n",
                           "                            if unresolved_source or rejected\n", 1).encode()
    elif fault == "changed_completion":
        bad = text.replace('"_completed_submit": not failure_reason,',
                           '"_completed_submit": True,', 1).encode()
    elif fault == "quoted_block":
        bad = ('async def investigate_company_evidence():\n    """\n'
               + OLD + '\n    """\n').encode()
    else:
        bad = ('async def investigate_company_evidence():\n    """\n'
               + NEW + '\n    """\n    if rejected and not failure_reason:\n'
               '        failure_reason = MALFORMED_RESPONSE_FAILURE_REASON\n').encode()
    with pytest.raises(RejectedFinalPacketSeamError):
        convert(bad)


@pytest.mark.parametrize("options", [
    {}, {"identity_schema_repair": True},
    {"complete_negative_finality": True, "source_access_denied": True,
     "pending_final_correction": True, "failed_attribute_source_recovery": True,
     "authenticated_page_refusal": True, "completed_investigation_finality": True,
     "identity_schema_repair": True, "single_tool_protocol_repair": True},
])
def test_composition_is_explicit_and_preserves_every_existing_repair(options):
    original = sources(**options)
    before = builder.expected_overlay(original, **options)
    after = builder.expected_overlay(original, **options, rejected_final_packet=True)
    assert set(after) == set(before)
    assert after[builder.GRADER_FILE] == convert(before[builder.GRADER_FILE])
    assert after[builder.GRADER_FILE].replace(GUARD.encode(), b"", 1) == before[builder.GRADER_FILE]
    for name in set(before) - {builder.GRADER_FILE}:
        assert after[name] == before[name]


def install_composed_investigator(monkeypatch):
    composed = builder.expected_overlay(
        sources(), rejected_final_packet=True,
    )[builder.GRADER_FILE]
    module = ast.parse(composed)
    node = next(node for node in module.body if isinstance(node, ast.AsyncFunctionDef)
                and node.name == "investigate_company_evidence")
    code = compile(ast.Module(body=[node], type_ignores=[]), "composed_frozen_investigator.py", "exec")

    async def investigate(**kwargs):
        # The existing offline fixtures replace external I/O before invocation.
        namespace = dict(vars(investigator))
        exec(code, namespace)
        return await namespace["investigate_company_evidence"](**kwargs)

    monkeypatch.setattr(investigator, "investigate_company_evidence", investigate)


@pytest.mark.parametrize("guard", ["negative", "positive", "contradicted", "ordinary"])
def test_composed_function_keeps_valid_final_findings(monkeypatch, guard):
    install_composed_investigator(monkeypatch)
    controls.test_existing_final_correction_and_failure_guards(monkeypatch, guard)


@pytest.mark.parametrize("guard", [
    "verified_empty", "verified_unfetched", "contradicted_empty",
    "nonstring_url", "nonstring_quote",
])
def test_composed_function_reaches_fit_arena_and_cache_incomplete(monkeypatch, guard):
    install_composed_investigator(monkeypatch)
    controls.test_rejected_final_packet_stays_incomplete_through_fit_arena_and_cache(monkeypatch, guard)


@pytest.mark.parametrize("fault", ["401", "403", "429", "502", "503"])
def test_composed_function_preserves_provider_failure_priority(monkeypatch, fault):
    install_composed_investigator(monkeypatch)
    controls.test_rejected_final_packet_preserves_provider_failure(monkeypatch, fault)


def test_verify_cli_applies_same_flag_and_reports_correction(monkeypatch, capsys):
    grading = "registry.example/lab/scorer@sha256:" + "a" * 64
    runtime = "registry.example/lab/scorer@sha256:" + "b" * 64
    original = sources(identity_schema_repair=True)
    base = {"Architecture": "amd64", "Os": "linux", "Config": {},
            "RootFS": {"Layers": ["sha256:base"]}}
    candidate = deepcopy(base)
    candidate["RootFS"]["Layers"].append("sha256:correction")
    verified = []
    monkeypatch.setattr(sys, "argv", ["build", "--grading-image", grading,
        "--verify-image", runtime, "--identity-schema-repair", "--rejected-final-packet"])
    monkeypatch.setattr(builder, "checked_output", lambda *a, **k: "c" * 40)
    monkeypatch.setattr(builder, "image_inspect", lambda ref: base if ref == grading else candidate)
    monkeypatch.setattr(builder, "extract_frozen_sources", lambda *a, **k: original)

    def verify(reference, layer, files):
        assert reference == runtime and layer == "sha256:correction"
        assert files[builder.GRADER_FILE] == convert(original[builder.GRADER_FILE])
        verified.append(True)

    monkeypatch.setattr(builder, "verify_saved_overlay", verify)
    monkeypatch.setattr(builder.subprocess, "run", lambda *a, **k: pytest.fail("unexpected build or publication"))
    monkeypatch.setattr(builder, "_managed_docker_host", lambda: False)
    assert builder.main() == 0
    receipt = json.loads(capsys.readouterr().out)
    assert verified == [True]
    assert receipt["grading_image"] == grading
    assert receipt["runtime_image"] == runtime
    assert receipt["grading_corrections"] == [
        "completed_semantic_identity_schema_repair_v1", CORRECTION_ID,
    ]
