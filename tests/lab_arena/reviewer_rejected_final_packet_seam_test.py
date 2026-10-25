"""The frozen reviewer receives only the rejected-final-packet correction."""
import ast
from copy import deepcopy
import json
import subprocess
import sys

import pytest

from qualification.scoring import company_evidence_investigator as investigator
from scripts import build_arena_reviewer_runtime as builder
from scripts.arena_reviewer_rejected_final_packet_seam import (
    CORRECTION_ID, GUARD, NEW, OLD, RejectedFinalPacketSeamError,
    convert_rejected_final_packet_seam as convert,
)
from scripts.arena_reviewer_identity_schema_seam import DELTAS as IDENTITY_DELTAS
from tests import test_final_unproven_normalization as controls


# The operational converter accepts the historical reviewer completion block,
# not the current compact investigator. Keep that exact frozen contract in the
# fixture without changing which source shapes the production converter admits.
COMPACT_RESEARCH_GUARD = '''                        # Ordinary model-selected research may be refused for
                        # time without invalidating a complete negative packet.
                        if (required_industry_research_denied and not failure_reason
                                and (claims.get("industry") or {}).get("status")
                                == "UNPROVEN"):
                            failure_reason = ADMISSION_BUDGET_INTERRUPTED_FAILURE_REASON
'''
COMPACT_NEW = NEW.replace(
    "                        # A complete validated judgment",
    COMPACT_RESEARCH_GUARD + "                        # A complete validated judgment", 1,
).replace('"claims": claims,', '"claims": _strip_untrusted_product_feedback(claims),', 1)


def frozen_reviewer_completion(source):
    """Make only the two explicit compact-to-frozen fixture substitutions."""
    assert source.count(COMPACT_NEW.encode()) == 1, "compact completion fixture drift"
    assert source.count(NEW.encode()) == 0
    return source.replace(COMPACT_NEW.encode(), NEW.encode(), 1)


def frozen_reviewer_source(source):
    source = frozen_reviewer_completion(source)
    # Current product bookkeeping sits inside three historical admission
    # anchors. Move the same statements outside those anchors in this fixture;
    # retain their assignments and the entire native research implementation.
    moves = (
        ('''            incomplete_submit_retries = 0
            incomplete_submit_retry_pending = False
            output_budget_retry_pending = False
''', '''            output_budget_retry_pending = False
            incomplete_submit_retries = 0
            incomplete_submit_retry_pending = False
'''),
        ('''                elif name == "search_web":
                    if forced_product_search_call:
                        early_product_search_pending = False
                    elapsed = time.monotonic() - started
''', '''                elif name == "search_web":
                    elapsed = time.monotonic() - started
                    if forced_product_search_call:
                        early_product_search_pending = False
'''),
        ('''                elif name == "fetch_page":
                    if forced_industry_fetch_call:
                        industry_followup_fetch_pending = False
                    elapsed = time.monotonic() - started
''', '''                elif name == "fetch_page":
                    elapsed = time.monotonic() - started
                    if forced_industry_fetch_call:
                        industry_followup_fetch_pending = False
'''),
    )
    for old, new in moves:
        assert source.count(old.encode()) == 1, "compact admission fixture drift"
        source = source.replace(old.encode(), new.encode(), 1)
    return source


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
    if options.get("authenticated_page_refusal"):
        # This historical 403 composition predates the separate exact 415
        # source-policy correction. Keep its committed transport preimage;
        # current unsupported-source behavior has its own native tests.
        for name in (builder.GRADER_FILE, builder.VERIFICATION_FILE):
            result[name] = subprocess.check_output(
                ["git", "show", "36f8e91390a14398194250c3dc898a4a98836a70:" + name],
                cwd=builder.ROOT,
            )
    result[builder.GRADER_FILE] = frozen_reviewer_source(result[builder.GRADER_FILE])
    assert result[builder.GRADER_FILE].count(NEW.encode()) == 1
    result[builder.GRADER_FILE] = result[builder.GRADER_FILE].replace(
        NEW.encode(), OLD.encode(), 1,
    )
    if options.get("identity_schema_repair"):
        # The frozen identity converter predates the compact criterion argument.
        # Composition tests exercise that exact historical input, while native
        # verifier/security tests below retain the actual product-aware gate.
        historical = IDENTITY_DELTAS[1][1]
        compact = historical.replace(
            "                else None\n",
            '                else None,\n'
            '                requested_product_service=str(icp.product_service or ""),\n', 1,
        )
        assert result[builder.FINALITY_FILE].count(compact.encode()) == 1
        result[builder.FINALITY_FILE] = result[builder.FINALITY_FILE].replace(
            compact.encode(), historical.encode(), 1,
        )
    return result


def test_exact_guard_is_the_only_changed_frozen_byte_and_is_idempotent():
    original = sources()[builder.GRADER_FILE]
    corrected = convert(original)
    native = (builder.ROOT / builder.GRADER_FILE).read_bytes()
    assert corrected == frozen_reviewer_source(native)
    assert corrected.replace(GUARD.encode(), b"", 1) == original
    assert convert(corrected) == corrected


def test_compact_native_source_is_not_admitted_as_a_frozen_reviewer():
    native = (builder.ROOT / builder.GRADER_FILE).read_bytes()
    assert COMPACT_RESEARCH_GUARD.encode() in native
    assert b'"claims": _strip_untrusted_product_feedback(claims),' in native
    with pytest.raises(RejectedFinalPacketSeamError, match="unrecognized or partial"):
        convert(native)


@pytest.mark.parametrize("fault", ["missing", "duplicate", "changed_budget", "changed_export"])
def test_compact_fixture_normalization_refuses_unreviewed_source(fault):
    native = (builder.ROOT / builder.GRADER_FILE).read_bytes()
    block = COMPACT_NEW.encode()
    if fault == "missing":
        native = native.replace(block, b"", 1)
    elif fault == "duplicate":
        native += block
    elif fault == "changed_budget":
        native = native.replace(b"if (required_industry_research_denied and not failure_reason",
                                b"if (required_industry_research_denied or not failure_reason", 1)
    else:
        native = native.replace(block, block.replace(
            b'"claims": _strip_untrusted_product_feedback(claims),',
            b'"claims": claims,', 1), 1)
    with pytest.raises(AssertionError, match="compact completion fixture drift"):
        frozen_reviewer_completion(native)


@pytest.mark.parametrize("statement", [
    pytest.param(b"            incomplete_submit_retry_pending = False\n"
                 b"            output_budget_retry_pending = False\n", id="serialization_retry"),
    pytest.param(b"                    if forced_product_search_call:\n"
                 b"                        early_product_search_pending = False\n", id="product_search"),
    pytest.param(b"                    if forced_industry_fetch_call:\n"
                 b"                        industry_followup_fetch_pending = False\n", id="industry_fetch"),
])
def test_compact_admission_fixture_refuses_unreviewed_bookkeeping(statement):
    native = (builder.ROOT / builder.GRADER_FILE).read_bytes()
    assert native.count(statement) == 1
    with pytest.raises(AssertionError, match="compact admission fixture drift"):
        frozen_reviewer_source(native.replace(statement, b"", 1))


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


@pytest.fixture(params=["frozen_reviewer_overlay", "native_compact"])
def investigation_runtime(request, monkeypatch):
    if request.param == "frozen_reviewer_overlay":
        install_composed_investigator(monkeypatch)
    # The other arm uses the actual current function, including its compact
    # completion budget and exported-feedback controls, without source rewriting.
    return request.param


@pytest.mark.parametrize("guard", ["negative", "positive", "contradicted", "ordinary"])
def test_composed_function_keeps_valid_final_findings(monkeypatch, guard, investigation_runtime):
    controls.test_existing_final_correction_and_failure_guards(monkeypatch, guard)


@pytest.mark.parametrize("guard", [
    "verified_empty", "verified_unfetched", "contradicted_empty",
    "nonstring_url", "nonstring_quote",
])
def test_composed_function_reaches_fit_arena_and_cache_incomplete(monkeypatch, guard, investigation_runtime):
    controls.test_rejected_final_packet_stays_incomplete_through_fit_arena_and_cache(monkeypatch, guard)


@pytest.mark.parametrize("fault", ["401", "403", "429", "502", "503"])
def test_composed_function_preserves_provider_failure_priority(monkeypatch, fault, investigation_runtime):
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
    assert builder.main() == 0
    receipt = json.loads(capsys.readouterr().out)
    assert verified == [True]
    assert receipt["grading_image"] == grading
    assert receipt["runtime_image"] == runtime
    assert receipt["grading_corrections"] == [
        "completed_semantic_identity_schema_repair_v1", CORRECTION_ID,
    ]
