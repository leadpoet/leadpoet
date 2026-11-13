"""Exact frozen target repair retains the submitted-company context seam."""

import ast
import copy
import hashlib
import json
import sys
from types import ModuleType

import pytest

from scripts import arena_reviewer_active_acquisition_seam as seam
from scripts import arena_reviewer_visible_auxiliary_context_seam as visible
from scripts import build_arena_reviewer_runtime as composer
from tests import test_active_acquisition_target as controls
from tests import test_submitted_company_context_visible_binding as context_controls
from tests.lab_arena.reviewer_runtime_composition_test import layer
from tests.reviewer_frozen_sources import frozen_source


FILE = "qualification/scoring/lead_scorer.py"
FROZEN = frozen_source("submitted_company_context_d657", FILE)
ORIGINAL = visible.convert_submitted_company_context_seam(FROZEN)
CORRECTED = seam.convert_active_acquisition_seam(ORIGINAL)


def test_exact_hashes_idempotence_and_only_bound_target_pattern_changes():
    assert hashlib.sha256(ORIGINAL).hexdigest() == seam.ORIGINAL_SHA256
    assert hashlib.sha256(CORRECTED).hexdigest() == seam.REPAIRED_SHA256
    assert seam.convert_active_acquisition_seam(CORRECTED) == CORRECTED
    assert ORIGINAL.decode().count(seam._BEFORE) == 1
    assert CORRECTED == ORIGINAL.decode().replace(
        seam._BEFORE, seam._BEFORE + seam._ADDITION, 1,
    ).encode()
    assert seam._ADDITION in (composer.ROOT / FILE).read_text()
    assert visible._SUBMITTED_AFTER in CORRECTED.decode()
    before, after = ast.parse(ORIGINAL), ast.parse(CORRECTED)
    for tree in (before, after):
        tree.body = [node for node in tree.body if not (
            isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name)
                and target.id == "_BOUND_ACQUISITION_SUBJECT_PATTERNS"
                for target in node.targets
            )
        )]
    # Generic stage proof, prompts, imports and every other function stay exact.
    assert ast.dump(before) == ast.dump(after)


@pytest.mark.parametrize("source", [
    b"", b"\xff", FROZEN, ORIGINAL + b"\n", CORRECTED + b"\n",
    ORIGINAL.replace(b"\n", b"\r\n"), ORIGINAL + ORIGINAL,
    ORIGINAL.replace(b'"rebrand"', b'"unknown"', 1),
    ORIGINAL.replace(seam._BEFORE.encode(),
                     (seam._BEFORE + seam._ADDITION[:100]).encode(), 1),
])
def test_unknown_unrepaired_or_partial_source_fails_closed(source):
    with pytest.raises(ValueError):
        seam.convert_active_acquisition_seam(source)


def _install(monkeypatch, source, label):
    module = ModuleType("_frozen_active_acquisition_" + label)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    exec(compile(source, module.__name__ + ".py", "exec"), vars(module))
    return module


def test_exact_frozen_source_reproduces_then_repairs_saved_quote(monkeypatch):
    before = _install(monkeypatch, ORIGINAL, "before")
    after = _install(monkeypatch, CORRECTED, "after")
    assert not before._acquired_stage_quote_supports_names(
        (controls.SAVED_COMPANY,), controls.SAVED_QUOTE,
    )
    monkeypatch.setattr(controls, "lead_scorer", after)
    monkeypatch.setitem(sys.modules, "qualification.scoring.lead_scorer", after)
    controls.test_saved_active_acquisition_quote_binds_exact_target()
    for target in ("Acme Analytics", "Box", "Acme Analytics Inc."):
        for verb in ("acquired", "has acquired"):
            controls.test_active_completed_acquisition_binds_target_not_buyer(target, verb)
    for requested, status, decision in (
        ("public", "CONTRADICTED", after.COMPANY_FIT_MISMATCH),
        ("acquired", "VERIFIED", after.COMPANY_FIT_MATCH),
    ):
        controls.test_saved_active_acquisition_survives_native_and_saved_stage_validation(
            requested, status, decision,
        )


@pytest.mark.parametrize("quote", [
    "Acme Analytics acquired Other Holdings.",
    "Parent acquired Other. Acme Analytics serves clients.",
    "Parent announced its acquisition of Acme Analytics.",
    "Parent plans to have acquired Acme Analytics by June.",
    "Parent has not acquired Acme Analytics.",
    "Parent acquired Acme Analytics, subject to approval.",
    "Parent acquired Acme Analytics, but the deal was cancelled.",
    "Parent acquired Acme Analytics assets.",
    "Parent acquired assets of Acme Analytics.",
    "Parent acquired Acme Analytics's product.",
    "Parent acquired Acme Analytics, a software product of Other.",
    "Parent acquired Acme Analytics as a customer.",
    "Parent acquired a minority stake in Acme Analytics.",
    "Parent acquired Acme Analytics Systems.",
    "Parent acquired Acme Analytics. Acme Analytics later became independent.",
])
def test_frozen_correction_preserves_incomplete_partial_and_other_entity_controls(
    monkeypatch, quote,
):
    before = _install(monkeypatch, ORIGINAL, "control_before")
    after = _install(monkeypatch, CORRECTED, "control_after")
    assert not before._acquired_stage_quote_supports_names(("Acme Analytics",), quote)
    assert not after._acquired_stage_quote_supports_names(("Acme Analytics",), quote)


def test_frozen_correction_retains_existing_passive_proof_and_visible_context(monkeypatch):
    after = _install(monkeypatch, CORRECTED, "preserved")
    monkeypatch.setattr(controls, "lead_scorer", after)
    for quote in (
        "Acme Analytics was acquired by Parent Holdings.",
        "Parent Holdings completed its acquisition of Acme Analytics.",
        "Acme Analytics is a wholly owned subsidiary of Parent Holdings.",
    ):
        controls.test_existing_completed_acquisition_patterns_stay_valid(quote)
    monkeypatch.setattr(context_controls, "lead_scorer", after)
    context_controls.test_metadata_only_name_or_quote_cannot_poison_paragraph_preflight(
        "Accept payments online.", context_controls._IDENTITY_LINK_CONTEXT_MARKER + " Ryft",
    )
    context_controls.test_visible_submitted_company_context_keeps_exact_existing_proof(
        context_controls._IDENTITY_LINK_CONTEXT_MARKER + " unread locator",
    )


@pytest.mark.parametrize("verify_existing", [False, True])
@pytest.mark.parametrize("base_digest", sorted(visible.SUBMITTED_FROZEN_IMAGE_DIGESTS))
def test_composer_build_and_verify_prove_exact_combined_one_file_layer(
    monkeypatch, capsys, verify_existing, base_digest,
):
    base_ref = "registry.example/reviewer@" + base_digest
    tag = composer.bounded_build_tag(
        "registry.example/reviewer:reviewer-" + base_digest[7:19] + "-" + "c" * 12
        + "-submitted-company-context-active-acquisition-target"
    )
    candidate_ref = "registry.example/reviewer@sha256:" + "b" * 64 if verify_existing else tag
    args = ["build_arena_reviewer_runtime.py", "--grading-image", base_ref,
            "--submitted-company-context-only", "--active-acquisition-target"]
    if verify_existing:
        args += ["--verify-image", candidate_ref]
    monkeypatch.setattr(composer.sys, "argv", args)
    monkeypatch.setattr(composer, "checked_output", lambda *_a, **_k: "c" * 40)
    base = {"Architecture": "amd64", "Os": "linux", "Config": {
        "Env": ["FROZEN=1"], "Entrypoint": ["python", "-m", "lab_arena.scorer_entrypoint"],
        "User": "65534:65534", "WorkingDir": "/model",
    }, "RootFS": {"Layers": ["sha256:base"]}}
    candidate = copy.deepcopy(base)
    candidate["RootFS"]["Layers"].append("sha256:overlay")
    monkeypatch.setattr(composer, "image_inspect", lambda ref: base if ref == base_ref else candidate)
    def extract(ref, _destination, **flags):
        assert ref == base_ref and flags == {"submitted_company_context_only": True}
        return {FILE: FROZEN}
    def verify(ref, digest, files):
        assert ref == candidate_ref and digest == "sha256:overlay"
        assert files == {FILE: CORRECTED}
    monkeypatch.setattr(composer, "extract_frozen_sources", extract)
    monkeypatch.setattr(composer, "verify_saved_overlay", verify)
    builds = []
    monkeypatch.setattr(composer.subprocess, "run", lambda command, **_k: builds.append(command))
    assert composer.main() == 0
    proof = json.loads(capsys.readouterr().out)
    assert proof["grading_image"] == base_ref
    assert proof["grading_corrections"] == [visible.SUBMITTED_CORRECTION_ID, seam.CORRECTION_ID]
    assert proof["original_files"] == {FILE: visible.SUBMITTED_ORIGINAL_SHA256}
    assert proof["overlay_files"] == {FILE: seam.REPAIRED_SHA256}
    assert proof["base_layers_unchanged"] is proof["process_configuration_unchanged"] is True
    assert len(builds) == (0 if verify_existing else 1)


@pytest.mark.parametrize("flags,match", [
    (["--active-acquisition-target"], "requires submitted-company-context-only"),
    (["--submitted-company-context-only", "--active-acquisition-target", "--shim-only"],
     "cannot include other corrections"),
    (["--submitted-company-context-only", "--active-acquisition-target", "--complete-negative-finality"],
     "cannot include other corrections"),
    (["--submitted-company-context-only", "--active-acquisition-target", "--completed-merger-survivor-only"],
     "cannot include other corrections"),
    (["--submitted-company-context-only", "--active-acquisition-target", "--push", "--verify-image",
      "registry.example/runtime@sha256:" + "b" * 64], "verify does not publish"),
])
def test_wrong_or_mixed_modes_fail_before_docker(monkeypatch, flags, match):
    monkeypatch.setattr(composer.sys, "argv", ["build_arena_reviewer_runtime.py", "--grading-image",
        "registry.example/reviewer@" + visible.SUBMITTED_FROZEN_IMAGE_DIGEST, *flags])
    monkeypatch.setattr(composer, "image_inspect", lambda *_: pytest.fail("Docker must not run"))
    with pytest.raises(composer.CompositionError, match=match):
        composer.main()


def test_wrong_frozen_base_fails_before_docker(monkeypatch):
    monkeypatch.setattr(composer.sys, "argv", ["build_arena_reviewer_runtime.py", "--grading-image",
        "registry.example/reviewer@sha256:" + "0" * 64,
        "--submitted-company-context-only", "--active-acquisition-target"])
    monkeypatch.setattr(composer, "image_inspect", lambda *_: pytest.fail("Docker must not run"))
    with pytest.raises(composer.CompositionError, match="exact frozen d657 or 50de"):
        composer.main()


@pytest.mark.parametrize("files", [
    [], [("model/" + FILE, CORRECTED + b"\n")],
    [("model/" + FILE, ORIGINAL)],
    [("model/" + FILE, CORRECTED), ("model/unapproved.py", b"pass\n")],
])
def test_layer_rejects_missing_changed_prior_or_extra_source(files):
    with pytest.raises(composer.CompositionError):
        composer.verify_overlay(layer(files), {FILE: CORRECTED})


def test_exact_combined_layer_is_accepted():
    composer.verify_overlay(layer([("model/" + FILE, CORRECTED)]), {FILE: CORRECTED})


@pytest.mark.parametrize("change", ["architecture", "env", "entrypoint", "user", "base", "extra"])
def test_configuration_and_parent_layers_cannot_change(change):
    base = {"Architecture": "amd64", "Os": "linux", "Config": {
        "Env": ["FROZEN=1"], "Entrypoint": ["python", "-m", "lab_arena.scorer_entrypoint"],
        "User": "65534:65534",
    }, "RootFS": {"Layers": ["sha256:base"]}}
    candidate = copy.deepcopy(base)
    candidate["RootFS"]["Layers"].append("sha256:overlay")
    if change == "architecture":
        candidate["Architecture"] = "arm64"
    elif change in {"env", "entrypoint", "user"}:
        field = {"env": "Env", "entrypoint": "Entrypoint", "user": "User"}[change]
        candidate["Config"][field] = "changed"
    elif change == "base":
        candidate["RootFS"]["Layers"][0] = "sha256:other"
    else:
        candidate["RootFS"]["Layers"].append("sha256:extra")
    with pytest.raises(composer.CompositionError):
        composer.verify_image_configuration(base, candidate)
