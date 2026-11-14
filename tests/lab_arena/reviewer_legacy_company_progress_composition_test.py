"""Exact repository-reference composition; live frozen51bc extraction is pending."""
import ast
import copy
import gzip
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest

pytestmark = pytest.mark.usefixtures("scorer_process_environment")

from lab_arena import scorer_entrypoint
from qualification.scoring import lead_scorer
from scripts import build_arena_reviewer_runtime as builder
from scripts import arena_reviewer_legacy_company_progress_seam as seam
from tests.lab_arena import legacy_company_judgments_test as controls
from tests.lab_arena.reviewer_runtime_composition_test import layer
from tests.test_arena_score_integrity import _positive_breakdown

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "reviewer_seams"


def reference_sources():
    # These are exact fa88187ce repository bytes, not claimed image observations.
    sources = {name: gzip.decompress((FIXTURES / (
        "legacy-company-progress-reference-fa88187ce-" + Path(name).name + ".gz"
    )).read_bytes()) for name in seam.FILES}
    assert {name: hashlib.sha256(data).hexdigest() for name, data in sources.items()} == seam.REFERENCE_SHA256
    return sources


def test_exact_reference_pair_changes_only_reviewed_source_delta():
    original = reference_sources()
    repaired = seam.convert_legacy_company_progress_files(original)
    assert seam.convert_legacy_company_progress_files(repaired) == repaired
    assert {name: hashlib.sha256(data).hexdigest() for name, data in repaired.items()} == seam.REPAIRED_SHA256
    allowed = {
        seam.SCORING_FILE: {"_JudgeExhausted", "score_work_item", "_score_legacy_company_work_item"},
        seam.COMPETITION_FILE: {"apply_legacy_company_judgment_context", "apply_company_judgment_context"},
    }
    for name in seam.FILES:
        if name == seam.COMPETITION_FILE:
            assert repaired[name] == (builder.ROOT / name).read_bytes()
        before, after = ast.parse(original[name]), ast.parse(repaired[name])
        for tree in (before, after):
            tree.body = [node for node in tree.body if getattr(node, "name", "") not in allowed[name]]
        assert ast.dump(before) == ast.dump(after)
    before = next(node for node in ast.parse(original[seam.SCORING_FILE]).body
                  if getattr(node, "name", "") == "score_work_item")
    after = next(node for node in ast.parse(repaired[seam.SCORING_FILE]).body
                 if getattr(node, "name", "") == "score_work_item")
    assert after.body[-1].exc.func.id == "_JudgeExhausted"
    after.body[-1].exc.func.id = "ScoringError"
    assert ast.dump(before) == ast.dump(after)


@pytest.mark.parametrize("fault", ["missing", "extra", "changed", "newline", "partial", "mixed"])
def test_unknown_partial_or_mixed_pair_fails_closed(fault):
    sources = reference_sources()
    if fault == "missing":
        sources.pop(seam.SCORING_FILE)
    elif fault == "extra":
        sources["unapproved.py"] = b"pass\n"
    elif fault == "mixed":
        sources[seam.SCORING_FILE] = seam.convert_legacy_company_progress_files(sources)[seam.SCORING_FILE]
    elif fault == "partial":
        before, after = seam._CHANGES[seam.SCORING_FILE][0]
        sources[seam.SCORING_FILE] = sources[seam.SCORING_FILE].replace(before, after, 1)
    elif fault == "newline":
        sources[seam.COMPETITION_FILE] = sources[seam.COMPETITION_FILE].replace(b"\n", b"\r\n")
    else:
        sources[seam.COMPETITION_FILE] += b"# unrelated\n"
    with pytest.raises(seam.LegacyCompanyProgressSeamError):
        seam.convert_legacy_company_progress_files(sources)


def test_extraction_reads_only_two_existing_files(monkeypatch, tmp_path):
    monkeypatch.setattr(builder, "checked_output", lambda *_: "container")
    sources, commands = reference_sources(), []

    def run(command, **_kwargs):
        commands.append(command)
        if command[1] == "cp":
            name = command[2].removeprefix("container:/model/")
            Path(command[3]).write_bytes(sources[name])

    monkeypatch.setattr(builder.subprocess, "run", run)
    assert builder.extract_frozen_sources("image", tmp_path, legacy_company_progress_only=True) == sources
    assert [command[2] for command in commands if command[1] == "cp"] == [
        "container:/model/" + name for name in seam.FILES]
    assert commands[-1] == ["docker", "rm", "container"]


@pytest.mark.parametrize("verify", [False, True])
def test_exclusive_mode_builds_and_verifies_exact_pair(monkeypatch, capsys, verify):
    base_ref = "registry.example/reviewer@" + seam.FROZEN_IMAGE_DIGEST
    runtime_ref = "registry.example/reviewer@sha256:" + "b" * 64
    args = ["compose", "--grading-image", base_ref, "--legacy-company-progress-only"]
    if verify:
        args += ["--verify-image", runtime_ref]
    monkeypatch.setattr(builder.sys, "argv", args)
    monkeypatch.setattr(builder, "checked_output", lambda *_a, **_k: "c" * 40)
    base = {"Architecture": "amd64", "Os": "linux", "Config": {}, "RootFS": {"Layers": ["base"]}}
    candidate = copy.deepcopy(base)
    candidate["RootFS"]["Layers"].append("overlay")
    monkeypatch.setattr(builder, "image_inspect", lambda name: base if name == base_ref else candidate)

    def extract(reference, _destination, **flags):
        assert reference == base_ref and flags == {"legacy_company_progress_only": True}
        return reference_sources()

    monkeypatch.setattr(builder, "extract_frozen_sources", extract)
    monkeypatch.setattr(builder, "expected_overlay", lambda *_a, **_k: pytest.fail("default overlay used"))
    verified, commands = [], []
    monkeypatch.setattr(builder, "verify_saved_overlay", lambda *args: verified.append(args))
    monkeypatch.setattr(builder.subprocess, "run", lambda command, **_kwargs: commands.append(command))
    assert builder.main() == 0
    proof = json.loads(capsys.readouterr().out)
    assert proof["grading_image"] == base_ref
    assert proof["grading_corrections"] == [seam.CORRECTION_ID]
    assert proof["original_files"] == seam.REFERENCE_SHA256
    assert proof["overlay_files"] == seam.REPAIRED_SHA256
    assert proof["base_layers_unchanged"] is proof["process_configuration_unchanged"] is True
    assert verified[0][2] == seam.convert_legacy_company_progress_files(reference_sources())
    assert len(commands) == (0 if verify else 1)
    if verify:
        assert proof["runtime_image"] == runtime_ref
    else:
        assert "-legacy-company-progress" in proof["runtime_image"]


def test_other_image_is_rejected_before_docker(monkeypatch):
    monkeypatch.setattr(builder.sys, "argv", ["compose", "--grading-image",
        "registry.example/reviewer@sha256:" + "a" * 64, "--legacy-company-progress-only"])
    monkeypatch.setattr(builder, "image_inspect", lambda *_: pytest.fail("image access"))
    with pytest.raises(builder.CompositionError, match="exact frozen51bc"):
        builder.main()


# Keep this list complete as the existing composer gains correction modes.
OTHER_FLAGS = tuple(sorted({
    "--" + name.replace("_", "-") for name in (
        "completed_merger_survivor_only", "visible_auxiliary_context", "identity_repair_finality_only",
        "identity_repair_before_investigation_only", "empty_domain_negative_finality_only", "shim_only",
        "intent_target_denial", "homepage_profile_alias", "verified_display_brand", "complete_negative_finality",
        "source_access_denied", "pending_final_correction", "failed_attribute_source_recovery",
        "authenticated_page_refusal", "completed_investigation_finality", "identity_schema_repair",
        "structured_homepage_numeric_alias", "bound_homepage_legal_alias", "archive_snapshot_https",
        "greenhouse_api_source", "numeric_profile_legal_self_name", "single_tool_protocol_repair",
        "rejected_final_packet", "source_blocked_alias", "auxiliary_source_name_binding",
        "homepage_linkedin_conflict_repair", "observed_linkedin_ampersand", "verified_homepage_brand_alias",
        "source_local_refusal_finality",
    )
}))


@pytest.mark.parametrize("other", OTHER_FLAGS)
def test_other_modes_are_rejected_before_docker(monkeypatch, other):
    monkeypatch.setattr(builder.sys, "argv", ["compose", "--grading-image",
        "registry.example/reviewer@" + seam.FROZEN_IMAGE_DIGEST, "--legacy-company-progress-only", other])
    monkeypatch.setattr(builder, "image_inspect", lambda *_: pytest.fail("image access"))
    with pytest.raises(builder.CompositionError, match="cannot include other corrections"):
        builder.main()


def test_overlay_rejects_changed_or_extra_files():
    files = seam.convert_legacy_company_progress_files(reference_sources())
    entries = [("model/" + name, data) for name, data in files.items()]
    builder.verify_overlay(layer(entries), files)
    for changed in (entries + [("model/unapproved.py", b"pass\n")],
                    [(name, data + b"\n") for name, data in entries]):
        with pytest.raises(builder.CompositionError):
            builder.verify_overlay(layer(changed), files)


def install_pair(monkeypatch, sources):
    import lab_arena
    import qualification.scoring
    for name, path, parent, attribute in (
        ("qualification.scoring.competition", seam.COMPETITION_FILE, qualification.scoring, "competition"),
        ("lab_arena.scoring", seam.SCORING_FILE, lab_arena, "scoring"),
    ):
        module = ModuleType(name)
        monkeypatch.setitem(sys.modules, name, module)
        exec(compile(sources[path], path, "exec"), vars(module))
        monkeypatch.setattr(parent, attribute, module)
        if name == "lab_arena.scoring":
            monkeypatch.setattr(scorer_entrypoint, "scoring", module)
            monkeypatch.setattr(controls, "scoring", module)


@pytest.mark.parametrize("repaired", [False, True])
def test_actual_reference_pair_reproduces_then_fixes_sibling_starvation(monkeypatch, repaired):
    sources = reference_sources()
    install_pair(monkeypatch, seam.convert_legacy_company_progress_files(sources) if repaired else sources)
    calls = []

    async def judge(**kwargs):
        name = kwargs["company"].company_name
        calls.append(name)
        if name == "A":
            raise RuntimeError("provider unavailable")
        return _positive_breakdown(name, name.lower() + ".example")

    monkeypatch.setattr(lead_scorer, "score_company_competition_intent", judge)
    output = controls._score(controls._input())
    assert output["failure"] == "judge_error" and "breakdowns" not in output
    assert calls == (["A", "A", "A", "B"] if repaired else ["A", "A", "A"])
    assert [row["company_index"] for row in output["company_judgments"]] == ([1] if repaired else [])


@pytest.mark.parametrize("name,kwargs", [
    ("test_failed_first_company_keeps_later_completed_judgments", {"later_verdict": verdict, "ordered_passes": False})
    for verdict in ("match", "mismatch", "inconclusive")
] + [
    ("test_failed_cache_key_is_not_retried_at_later_duplicate_index", {"ordered_passes": False}),
    ("test_first_native_exhaustion_is_raised_unchanged_after_independent_progress", {"ordered_passes": False}),
    ("test_deferred_failure_preserves_schema_bucket_and_max_gates", {"ordered_passes": False}),
])
def test_corrected_reference_pair_preserves_native_controls(monkeypatch, name, kwargs):
    install_pair(monkeypatch, seam.convert_legacy_company_progress_files(reference_sources()))
    getattr(controls, name)(monkeypatch, **kwargs)


def test_corrected_reference_keeps_native_contract_error_fatal(monkeypatch):
    install_pair(monkeypatch, seam.convert_legacy_company_progress_files(reference_sources()))
    controls.test_native_scorer_count_contract_error_stops_before_later_company()
