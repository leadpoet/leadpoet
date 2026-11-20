"""Exact frozen396 identity/source corrections and immutable image boundary."""
import ast
import copy
import hashlib
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from scripts import build_arena_reviewer_runtime as composer
from scripts import arena_reviewer_homepage_identity_conflict_seam as seam
from tests import test_homepage_identity_conflict_repair as controls
from tests import test_intent_nonblocking_antibot_markup as source_controls
from tests.lab_arena.reviewer_runtime_composition_test import layer
from tests.reviewer_frozen_sources import frozen_source


ORIGINAL = frozen_source("homepage_identity_conflict_396", seam.FINALITY_FILE)
CORRECTED = seam.convert_homepage_identity_conflict_seam(ORIGINAL)
ORIGINAL_FILES = {
    name: frozen_source("homepage_identity_conflict_396", name) for name in seam.FILES
}
CORRECTED_FILES = seam.convert_homepage_identity_source_files(ORIGINAL_FILES)


def _install(monkeypatch, source):
    module = ModuleType("_arena_frozen396_homepage_identity_conflict")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    exec(compile(source, "exact_frozen396_lead_scorer.py", "exec"), vars(module))
    monkeypatch.setattr(controls, "scorer", module)


def test_exact_reviewed_delta_preserves_every_other_frozen_function():
    assert hashlib.sha256(ORIGINAL).hexdigest() == seam.ORIGINAL_SHA256
    assert hashlib.sha256(CORRECTED).hexdigest() == seam.REPAIRED_SHA256
    assert seam.convert_homepage_identity_conflict_seam(CORRECTED) == CORRECTED
    assert CORRECTED == frozen_source("homepage_empty_profile_parent", seam.FINALITY_FILE)
    changed = {
        "_unresolved_homepage_web_identity_conflict", "_reverify_decision",
        "_has_explicitly_unproven_fit_dimensions",
        "_run_targeted_company_evidence_investigation", "_llm_reverify_company",
    }
    before, after = ast.parse(ORIGINAL), ast.parse(CORRECTED)
    for tree in (before, after):
        tree.body = [node for node in tree.body if getattr(node, "name", "") not in changed]
    assert ast.dump(before) == ast.dump(after)
    assert CORRECTED_FILES[seam.SOURCE_FILE] == (composer.ROOT / seam.SOURCE_FILE).read_bytes()
    assert seam.convert_homepage_identity_source_files(CORRECTED_FILES) == CORRECTED_FILES
    before, after = ast.parse(ORIGINAL_FILES[seam.SOURCE_FILE]), ast.parse(CORRECTED_FILES[seam.SOURCE_FILE])
    for tree in (before, after):
        tree.body = [node for node in tree.body if getattr(node, "name", "") != "_evaluate_sd_response"]
    assert ast.dump(before) == ast.dump(after)


@pytest.mark.parametrize("source", [
    b"", b"\xff", ORIGINAL + b"\n", ORIGINAL.replace(b"\n", b"\r\n"),
    ORIGINAL + ORIGINAL, CORRECTED + b"# unrelated\n",
    ORIGINAL.replace(seam._DELTAS[0][0].encode(), seam._DELTAS[0][1].encode(), 1),
])
def test_unknown_or_partial_sources_fail_closed(source):
    with pytest.raises(seam.HomepageIdentityConflictSeamError):
        seam.convert_homepage_identity_conflict_seam(source)


@pytest.mark.parametrize("verify_existing", [False, True])
@pytest.mark.parametrize("mode", ["identity", "empty_profile", "headquarters"])
def test_exclusive_composer_build_and_verify_exact_two_files(monkeypatch, capsys, verify_existing, mode):
    empty_profile = mode != "identity"
    headquarters = mode == "headquarters"
    original = dict(ORIGINAL_FILES)
    if empty_profile:
        original[seam.PROFILE_FILE] = frozen_source("homepage_identity_conflict_396", seam.PROFILE_FILE)
    corrected = (seam.convert_preserved_headquarters_source_files(original) if headquarters
                 else seam.convert_completed_empty_profile_files(original) if empty_profile else CORRECTED_FILES)
    base_ref = "registry.example/reviewer@" + seam.FROZEN_IMAGE_DIGEST
    tag = "registry.example/reviewer:reviewer-" + seam.FROZEN_IMAGE_DIGEST[7:19] + "-" + "c" * 12 + "-homepage-identity-source"
    candidate_ref = "registry.example/reviewer@sha256:" + "b" * 64 if verify_existing else tag
    args = ["build_arena_reviewer_runtime.py", "--grading-image", base_ref,
            "--homepage-identity-and-source-validation-only"]
    if empty_profile:
        args.append("--completed-empty-profile")
    if headquarters:
        args.append("--preserved-headquarters-source")
    if verify_existing:
        args += ["--verify-image", candidate_ref]
    monkeypatch.setattr(composer.sys, "argv", args)
    monkeypatch.setattr(composer, "checked_output", lambda *_a, **_k: "c" * 40)
    base = {"Architecture": "amd64", "Os": "linux", "Config": {
        "Entrypoint": ["python3", "/model/scorer_entrypoint.py"],
        "User": "65534:65534", "WorkingDir": "/model", "Env": ["PYTHONPATH=/model"],
    }, "RootFS": {"Layers": ["sha256:base"]}}
    candidate = copy.deepcopy(base)
    candidate["RootFS"]["Layers"].append("sha256:overlay")
    monkeypatch.setattr(composer, "image_inspect", lambda ref: base if ref == base_ref else candidate)

    def extract(ref, _destination, **flags):
        assert ref == base_ref
        assert flags == {"homepage_identity_and_source_validation_only": True,
                         **({"completed_empty_profile": True} if empty_profile else {})}
        return original

    def verify(ref, digest, files):
        assert ref == candidate_ref and digest == "sha256:overlay"
        assert files == corrected

    builds = []

    def build(command, **_kwargs):
        builds.append(command)
        root = Path(command[-1])
        assert (root / "Dockerfile").read_text() == (
            "FROM " + base_ref + "\nCOPY --chown=65534:65534 overlay/ /model/\n"
        )
        assert {p.relative_to(root / "overlay").as_posix()
                for p in (root / "overlay").rglob("*") if p.is_file()} == set(corrected)

    monkeypatch.setattr(composer, "extract_frozen_sources", extract)
    monkeypatch.setattr(composer, "verify_saved_overlay", verify)
    monkeypatch.setattr(composer.subprocess, "run", build)
    assert composer.main() == 0
    proof = json.loads(capsys.readouterr().out)
    assert proof["grading_image"] == base_ref
    assert proof["grading_corrections"] == (list(seam.CORRECTIONS)
        + ([seam.EMPTY_PROFILE_CORRECTION_ID] if empty_profile else [])
        + ([seam.HEADQUARTERS_SOURCE_CORRECTION_ID] if headquarters else []))
    assert proof["original_files"] == {name: hashlib.sha256(value).hexdigest() for name, value in original.items()}
    assert proof["overlay_files"] == {name: hashlib.sha256(value).hexdigest() for name, value in corrected.items()}
    assert proof["base_layers_unchanged"] is proof["process_configuration_unchanged"] is True
    assert len(builds) == (0 if verify_existing else 1)


@pytest.mark.parametrize("other", [
    "--shim-only", "--identity-repair-before-investigation-only", "--identity-schema-repair",
    "--verified-display-brand", "--complete-negative-finality", "--source-access-denied",
])
def test_exclusive_mode_rejects_other_corrections_before_docker(monkeypatch, other):
    monkeypatch.setattr(composer.sys, "argv", [
        "build_arena_reviewer_runtime.py", "--grading-image", "registry.example/base@" + seam.FROZEN_IMAGE_DIGEST,
        "--homepage-identity-and-source-validation-only", other,
    ])
    with pytest.raises(composer.CompositionError, match="cannot include"):
        composer.main()


def test_mode_rejects_other_grading_image_before_docker(monkeypatch):
    monkeypatch.setattr(composer.sys, "argv", [
        "build_arena_reviewer_runtime.py", "--grading-image", "registry.example/base@sha256:" + "a" * 64,
        "--homepage-identity-and-source-validation-only",
    ])
    with pytest.raises(composer.CompositionError, match="exact frozen396"):
        composer.main()


def test_overlay_rejects_extra_or_changed_files():
    expected = CORRECTED_FILES
    reviewed = [("model/" + name, value) for name, value in expected.items()]
    composer.verify_overlay(layer(reviewed), expected)
    for files in (
        [*reviewed, ("model/unrelated.py", b"pass\n")],
        [("model/" + name, value + b"\n") for name, value in expected.items()],
        reviewed[:1],
    ):
        with pytest.raises(composer.CompositionError):
            composer.verify_overlay(layer(files), expected)


def test_combined_converter_rejects_partial_or_extra_source_sets():
    for sources in (
        {seam.FINALITY_FILE: ORIGINAL},
        {**ORIGINAL_FILES, "unrelated.py": b"pass\n"},
        {**ORIGINAL_FILES, seam.SOURCE_FILE: ORIGINAL_FILES[seam.SOURCE_FILE] + b"\n"},
        {**ORIGINAL_FILES, seam.SOURCE_FILE: CORRECTED_FILES[seam.SOURCE_FILE]},
    ):
        with pytest.raises(seam.HomepageIdentityConflictSeamError):
            seam.convert_homepage_identity_source_files(sources)


def test_combined_extract_reads_only_the_two_frozen_files(monkeypatch, tmp_path):
    monkeypatch.setattr(composer, "checked_output", lambda *_a, **_k: "container")
    commands = []

    def run(command, **_kwargs):
        commands.append(command)
        if command[1] == "cp":
            name = command[2].removeprefix("container:/model/")
            Path(command[3]).write_bytes(ORIGINAL_FILES[name])

    monkeypatch.setattr(composer.subprocess, "run", run)
    assert composer.extract_frozen_sources(
        "registry/base@" + seam.FROZEN_IMAGE_DIGEST, tmp_path,
        homepage_identity_and_source_validation_only=True,
    ) == ORIGINAL_FILES
    assert len([command for command in commands if command[1] == "cp"]) == 2


@pytest.mark.parametrize("investigation", [False, True])
def test_actual_frozen396_reproduces_gap_then_exact_overlay_closes_it(monkeypatch, investigation):
    _install(monkeypatch, ORIGINAL)
    before, events, _, _ = controls._exercise(monkeypatch, investigation=investigation)
    assert "lead_scorer_reverify_schema_repair" not in events
    assert before.decision == "unavailable"
    assert before.details.get("failure_class") != "insufficient_fit_evidence"
    _install(monkeypatch, CORRECTED)
    controls.test_saved_dalfen_conflict_uses_one_identity_repair_and_finishes_unqualified(
        monkeypatch, investigation,
    )


@pytest.mark.parametrize("profile", ["canonical", "matching"])
def test_exact_overlay_preserves_positive_controls(monkeypatch, profile):
    _install(monkeypatch, CORRECTED)
    controls.test_proven_identity_has_no_extra_review_or_calls(monkeypatch, profile)


@pytest.mark.parametrize("kwargs", [
    {"repair": "provider_error"}, {"repair": "malformed_response"},
    {"repair": "null"}, {"profile": "fault"}, {"homepage_fault": True},
    {"investigation": True, "source_fault": True},
])
def test_exact_overlay_preserves_retryable_faults(monkeypatch, kwargs):
    _install(monkeypatch, CORRECTED)
    controls.test_unfinished_identity_or_source_fault_stays_retryable(monkeypatch, kwargs)


def test_exact_frozen_source_reproduces_markup_false_positive_then_uses_one_read(monkeypatch):
    module = ModuleType("_arena_frozen396_source_validation")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    exec(compile(ORIGINAL_FILES[seam.SOURCE_FILE], "exact_frozen396_intent.py", "exec"), vars(module))
    body = source_controls.document("<article><p>" + source_controls.TEXT + "</p></article>")
    assert module._evaluate_sd_response(200, body) == "anti_bot_marker"
    exec(compile(CORRECTED_FILES[seam.SOURCE_FILE], "exact_corrected396_intent.py", "exec"), vars(module))
    assert module._evaluate_sd_response(200, body) == "ok"
    monkeypatch.setattr(source_controls, "intent", module)
    source_controls.test_rendered_body_uses_one_existing_fetch_without_escalation()
    source_controls.test_challenge_title_and_non_html_markers_still_fail()


def test_empty_profile_composition_preserves_all_other_frozen_bytes(monkeypatch):
    original = {**ORIGINAL_FILES, seam.PROFILE_FILE: frozen_source('homepage_identity_conflict_396', seam.PROFILE_FILE)}
    after = seam.convert_completed_empty_profile_files(original)
    for name in (seam.FINALITY_FILE, seam.PROFILE_FILE):
        assert hashlib.sha256(after[name]).hexdigest() == seam.EMPTY_PROFILE_REPAIRED_SHA256[name]
    assert after[seam.PROFILE_FILE] == (composer.ROOT / seam.PROFILE_FILE).read_bytes()
    assert after[seam.SOURCE_FILE] == CORRECTED_FILES[seam.SOURCE_FILE]
    for changed in ({**original, seam.PROFILE_FILE: original[seam.PROFILE_FILE]+b'\n'},
                    {**original, 'extra.py': b''}, ORIGINAL_FILES):
        with pytest.raises(seam.HomepageIdentityConflictSeamError):
            seam.convert_completed_empty_profile_files(changed)
    _install(monkeypatch, after[seam.FINALITY_FILE])
    controls.test_completed_empty_profile_through_native_parser_finishes_unproven(monkeypatch, True)


@pytest.mark.parametrize("retry", [False, True])
def test_headquarters_overlay_changes_only_prefetch_and_replays_retained_evidence(monkeypatch, retry):
    from tests import test_retained_headquarters_source as retained
    original = {**ORIGINAL_FILES, seam.PROFILE_FILE: frozen_source('homepage_identity_conflict_396', seam.PROFILE_FILE)}
    before = seam.convert_completed_empty_profile_files(original)
    after = seam.convert_preserved_headquarters_source_files(original)
    assert after[seam.FINALITY_FILE] == (composer.ROOT / seam.FINALITY_FILE).read_bytes()
    for name in (seam.SOURCE_FILE, seam.PROFILE_FILE): assert before[name] == after[name]
    trees = [ast.parse(files[seam.FINALITY_FILE]) for files in (before, after)]
    for tree in trees:
        tree.body = [n for n in tree.body if getattr(n, 'name', '') != '_investigator_prefetched_pages']
    assert ast.dump(trees[0]) == ast.dump(trees[1])
    _install(monkeypatch, after[seam.FINALITY_FILE])
    monkeypatch.setattr(retained, 'scorer', controls.scorer)
    retained.test_retained_quote_survives_full_investigator_validation(monkeypatch, retry, '')


def test_headquarters_mode_requires_reviewed_parent_before_docker(monkeypatch):
    monkeypatch.setattr(composer.sys, 'argv', ['build_arena_reviewer_runtime.py',
        '--grading-image', 'registry.example/base@' + seam.FROZEN_IMAGE_DIGEST,
        '--preserved-headquarters-source'])
    with pytest.raises(composer.CompositionError, match='requires completed-empty-profile'):
        composer.main()
