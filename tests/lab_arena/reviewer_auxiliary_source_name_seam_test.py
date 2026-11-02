"""Repair only the optional auxiliary producer in an exact frozen grader."""
import ast
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

import pytest
from tests.reviewer_frozen_sources import frozen_source

from qualification.scoring import lead_scorer
from scripts import build_arena_reviewer_runtime as builder
from scripts.arena_reviewer_auxiliary_source_name_seam import (
    ANCHOR, CORRECTION_ID, FUNCTION, GUARD, AuxiliarySourceNameSeamError,
    REPAIRED_SHA256,
    convert_auxiliary_source_name_seam as convert,
)
from tests import test_intent_details_fetched_first_party_context as controls
from tests.lab_arena.reviewer_runtime_composition_test import layer


# Exact CI-green native-budget postimage; historical overlays must not track
# the evolving current investigator or require Git history in test installs.
_FROZEN_GRADER = frozen_source(
    "native_industry_budget", "qualification/scoring/company_evidence_investigator.py",
)


def corrected_finality(source=None):
    # Exact reviewed function, also present at upstream600106212. Today's
    # native scorer can evolve without changing a frozen image's repair.
    frozen = (builder.ROOT / 'tests/fixtures/arena_reviewer/frozen_auxiliary_source_context.py.txt').read_bytes()
    assert hashlib.sha256(frozen).hexdigest() == REPAIRED_SHA256
    text = (source if source is not None else (builder.ROOT / builder.FINALITY_FILE).read_bytes()).decode()
    node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == FUNCTION)
    lines = text.splitlines(keepends=True)
    return ''.join(lines[:node.lineno - 1]).encode() + frozen + ''.join(lines[node.end_lineno:]).encode()


def sources(**options):
    names = [builder.GRADER_FILE, builder.FINALITY_FILE, *builder.ENTRYPOINT_FILES]
    if options.get("authenticated_page_refusal"):
        names.append(builder.VERIFICATION_FILE)
    if options.get("bound_homepage_legal_alias"):
        names.extend(builder.BOUND_IDENTITY_FILES)
    result = {
        name: (builder.ROOT / ("lab_arena/scorer_entrypoint.py"
                              if name == "scorer_entrypoint.py" else name)).read_bytes()
        for name in names
    }
    result[builder.GRADER_FILE] = _FROZEN_GRADER
    if options.get("archive_snapshot_https"):
        # This seam accepts its exact historical function, not today's source.
        from tests.test_arena_archive_snapshot_https import original_source
        result[builder.ARCHIVE_FILE] = original_source()
    if options.get("rejected_final_packet"):
        from tests.lab_arena.reviewer_rejected_final_packet_seam_test import corrected_grader
        result[builder.GRADER_FILE] = corrected_grader()
    if options.get("bound_homepage_legal_alias"):
        # That existing whole-module seam accepts its historical preimage,
        # rather than today's grader with later independent default changes.
        from tests.test_arena_reviewer_bound_homepage_alias_seam import original_sources
        result.update(original_sources())
    elif options.get("structured_homepage_numeric_alias"):
        from tests.lab_arena.reviewer_runtime_composition_test import _numeric_alias_sources
        result[builder.FINALITY_FILE] = _numeric_alias_sources()[builder.FINALITY_FILE]
    result[builder.FINALITY_FILE] = corrected_finality(result[builder.FINALITY_FILE])
    assert result[builder.FINALITY_FILE].count(GUARD.encode()) == 1
    result[builder.FINALITY_FILE] = result[builder.FINALITY_FILE].replace(
        GUARD.encode(), b"", 1,
    )
    return result


def test_exact_function_guard_is_the_only_change_and_is_idempotent():
    original = sources()[builder.FINALITY_FILE]
    repaired = convert(original)
    assert repaired == corrected_finality()
    assert repaired.replace(GUARD.encode(), b"", 1) == original
    assert convert(repaired) == repaired


def test_current_industry_source_cannot_be_substituted_for_frozen_function():
    with pytest.raises(AuxiliarySourceNameSeamError, match='unrecognized'):
        convert((builder.ROOT / builder.FINALITY_FILE).read_bytes())


@pytest.mark.parametrize("fault", [
    "utf8", "syntax", "crlf", "missing", "duplicate", "shadowed",
    "decorated", "changed_original", "partial", "weakened", "duplicate_guard",
    "quoted_function",
])
def test_unknown_or_partially_corrected_function_fails_closed(fault):
    original = sources()[builder.FINALITY_FILE]
    if fault == "utf8":
        bad = b"\xff"
    elif fault == "syntax":
        bad = b"def _matched_first_party_source_context(:"
    elif fault == "crlf":
        bad = original.replace(b"\n", b"\r\n")
    elif fault == "missing":
        bad = original.replace(("def " + FUNCTION + "(").encode(), b"def other(", 1)
    elif fault == "duplicate":
        bad = original + ("\ndef " + FUNCTION + "():\n    pass\n").encode()
    elif fault == "shadowed":
        bad = original + ("\n" + FUNCTION + " = None\n").encode()
    elif fault == "decorated":
        bad = original.replace(("def " + FUNCTION + "(").encode(),
                               ("@decorator\ndef " + FUNCTION + "(").encode(), 1)
    elif fault == "changed_original":
        bad = original.replace(ANCHOR.encode(),
                               (ANCHOR + "    identity = None\n").encode(), 1)
    elif fault == "partial":
        bad = convert(original).replace(b'        != _compact_company_name(identity.get("observed_name"))\n', b"", 1)
    elif fault == "weakened":
        bad = convert(original).replace(b'        != _compact_company_name(identity.get("observed_name"))\n',
                                        b'        == _compact_company_name(identity.get("observed_name"))\n', 1)
    elif fault == "duplicate_guard":
        bad = convert(original).replace(GUARD.encode(), (GUARD + GUARD).encode(), 1)
    else:
        bad = ("quoted = " + repr(original.decode()) + "\n").encode()
    with pytest.raises(AuxiliarySourceNameSeamError):
        convert(bad)


@pytest.mark.parametrize("options", [
    {},
    {"source_blocked_alias": True},
    {"identity_schema_repair": True},
    {"complete_negative_finality": True, "source_access_denied": True,
     "source_blocked_alias": True, "pending_final_correction": True,
     "failed_attribute_source_recovery": True, "authenticated_page_refusal": True,
     "completed_investigation_finality": True, "identity_schema_repair": True,
     "single_tool_protocol_repair": True,
     "rejected_final_packet": True, "archive_snapshot_https": True},
    {"structured_homepage_numeric_alias": True},
    {"bound_homepage_legal_alias": True, "numeric_profile_legal_self_name": True},
])
def test_correction_is_explicit_and_preserves_existing_options(options):
    original = sources(**options)
    saved = deepcopy(original)
    prior_sources = dict(original)
    if not any(options.get(name) for name in (
        "complete_negative_finality", "source_access_denied",
        "failed_attribute_source_recovery", "completed_investigation_finality",
        "identity_schema_repair", "structured_homepage_numeric_alias",
        "bound_homepage_legal_alias",
    )):
        prior_sources.pop(builder.FINALITY_FILE)
    before = builder.expected_overlay(prior_sources, **options)
    after = builder.expected_overlay(
        original, **options, auxiliary_source_name_binding=True,
    )
    assert after[builder.FINALITY_FILE] == convert(
        before.get(builder.FINALITY_FILE, original[builder.FINALITY_FILE]),
    )
    assert after[builder.FINALITY_FILE].replace(GUARD.encode(), b"", 1) == (
        before.get(builder.FINALITY_FILE, original[builder.FINALITY_FILE])
    )
    assert set(after) == set(before) | {builder.FINALITY_FILE}
    for name in set(before) - {builder.FINALITY_FILE}:
        assert after[name] == before[name]
    assert original == saved
    builder.verify_overlay(layer([("model/" + k, v) for k, v in after.items()]), after)
    altered = dict(after)
    altered[builder.FINALITY_FILE] = altered[builder.FINALITY_FILE].replace(GUARD.encode(), b"", 1)
    with pytest.raises(builder.CompositionError):
        builder.verify_overlay(layer([("model/" + k, v) for k, v in altered.items()]), after)


@pytest.mark.parametrize("submitted,observed,admit", [
    ("Ryft Ltd", "Ryft", False), ("Ryft", "Ryft Ltd", False),
    ("Ryft", "Ryft", True), ("Ry-ft", "RYFT", True),
])
def test_composed_function_reaches_unchanged_real_review_document(
    monkeypatch, submitted, observed, admit,
):
    repaired = builder.expected_overlay(
        sources(), auxiliary_source_name_binding=True,
    )[builder.FINALITY_FILE]
    node = next(node for node in ast.parse(repaired).body
                if isinstance(node, ast.FunctionDef) and node.name == FUNCTION)
    scope = dict(lead_scorer.__dict__)
    exec(compile(ast.Module(body=[node], type_ignores=[]), "composed_auxiliary.py", "exec"), scope)
    monkeypatch.setattr(lead_scorer, FUNCTION, scope[FUNCTION])
    controls.test_auxiliary_producer_matches_reviewers_compact_name_requirement(
        submitted, observed, admit,
    )


def test_cli_verifies_image_with_named_correction_and_exact_overlay(monkeypatch, capsys):
    original = sources()
    base = {"Architecture": "amd64", "Os": "linux", "Config": {},
            "RootFS": {"Layers": ["sha256:base"]}}
    candidate = deepcopy(base)
    candidate["RootFS"]["Layers"].append("sha256:overlay")
    grading = "registry.example/model@sha256:" + "a" * 64
    seen = {}
    monkeypatch.setattr("sys.argv", ["build", "--grading-image", grading,
                                    "--verify-image", "reviewed:runtime",
                                    "--auxiliary-source-name-binding"])
    monkeypatch.setattr(builder, "checked_output", lambda *args, **kwargs: "b" * 40)
    monkeypatch.setattr(builder, "image_inspect", lambda ref: base if ref == grading else candidate)
    def extract(reference, destination, **kwargs):
        seen.update(kwargs)
        return original
    def verify(reference, digest, files):
        seen.update(reference=reference, digest=digest, files=files)
        builder.verify_overlay(layer([("model/" + k, v) for k, v in files.items()]), files)
    monkeypatch.setattr(builder, "extract_frozen_sources", extract)
    monkeypatch.setattr(builder, "verify_saved_overlay", verify)
    monkeypatch.setattr(builder, "_managed_docker_host", lambda: False)
    assert builder.main() == 0
    receipt = json.loads(capsys.readouterr().out)
    assert seen["auxiliary_source_name_binding"] is True
    assert receipt["grading_corrections"] == [CORRECTION_ID]
    assert receipt["grading_image"] == grading
    assert receipt["runtime_image"] == "reviewed:runtime"
    assert receipt["base_layers_unchanged"] and receipt["process_configuration_unchanged"]
    assert receipt["original_files"][builder.FINALITY_FILE] == hashlib.sha256(original[builder.FINALITY_FILE]).hexdigest()
    assert seen["files"][builder.FINALITY_FILE] == convert(original[builder.FINALITY_FILE])


def test_exact_prior_published_overlay_when_supplied():
    directory = os.environ.get("ARENA_AUXILIARY_FROZEN_DIRECTORY")
    receipt_path = os.environ.get("ARENA_AUXILIARY_PRIOR_RECEIPT")
    if not directory or not receipt_path:
        pytest.skip("exact frozen image and published receipt not supplied")
    root = Path(directory)
    receipt = json.loads(Path(receipt_path).read_text())
    original = {
        name: (root / name).read_bytes()
        for name in (builder.GRADER_FILE, *builder.ENTRYPOINT_FILES)
    }
    digest = lambda content: hashlib.sha256(content).hexdigest()
    assert {name: digest(value) for name, value in original.items()} == receipt["original_files"]
    prior = builder.expected_overlay(original, source_blocked_alias=True)
    assert {name: digest(value) for name, value in prior.items()} == receipt["overlay_files"]
    original[builder.FINALITY_FILE] = (root / builder.FINALITY_FILE).read_bytes()
    repaired = builder.expected_overlay(
        original, source_blocked_alias=True, auxiliary_source_name_binding=True,
    )
    assert set(repaired) == set(prior) | {builder.FINALITY_FILE}
    assert all(repaired[name] == value for name, value in prior.items())
    assert repaired[builder.FINALITY_FILE].replace(GUARD.encode(), b"", 1) == original[builder.FINALITY_FILE]


@pytest.mark.parametrize("variable", [
    "ARENA_AUXILIARY_FROZEN_SOURCE", "ARENA_AUXILIARY_RUNTIME_SOURCE",
    "ARENA_AUXILIARY_COMPOSED_SOURCE",
])
def test_exact_extracted_image_source_when_supplied(variable):
    path = os.environ.get(variable)
    if not path:
        pytest.skip("exact image extraction not supplied")
    original = Path(path).read_bytes()
    repaired = convert(original)
    assert convert(repaired) == repaired
    if variable == "ARENA_AUXILIARY_COMPOSED_SOURCE":
        assert repaired == original
