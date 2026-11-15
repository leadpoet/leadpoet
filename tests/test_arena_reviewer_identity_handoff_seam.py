"""Only the reviewed identity delta may extend the four-option frozen runtime."""
import ast
import copy
import hashlib
import json

import pytest

from scripts import arena_reviewer_identity_handoff_seam as seam
from scripts import arena_reviewer_investigator_feedback_seam as feedback
from scripts import arena_reviewer_legacy_company_progress_seam as retry
from scripts import build_arena_reviewer_runtime as builder
from tests import test_arena_reviewer_investigator_feedback_seam as feedback_controls
from tests.lab_arena import reviewer_ordered_company_retry_composition_test as ordered
from tests.lab_arena.reviewer_runtime_composition_test import layer

FLAGS = [*ordered.FLAGS, "--identity-handoff"]


def selected_sources():
    return {
        builder.FINALITY_FILE: feedback_controls.CORRECTED_FINALITY,
        builder.GRADER_FILE: feedback.convert_investigator_feedback_seam(feedback_controls.original_investigator()),
    }


def test_exact_pair_is_reversible_idempotent_and_preserves_other_source():
    original = selected_sources()
    corrected = seam.convert_identity_handoff_files(original)
    assert {name: hashlib.sha256(data).hexdigest() for name, data in original.items()} == seam.ORIGINAL_SHA256
    assert {name: hashlib.sha256(data).hexdigest() for name, data in corrected.items()} == seam.REPAIRED_SHA256
    assert seam.convert_identity_handoff_files(corrected) == corrected
    allowed = {
        builder.FINALITY_FILE: {
            "_complete_verified_attribute_recovery_identity", "_official_child_structured_profile_lookup",
            "_alias_unresolved_structured_profile_lookup", "_structured_profile_alias_identity_receipt",
            "_run_targeted_company_evidence_investigation", "_llm_reverify_company",
        },
        builder.GRADER_FILE: {"_reversed_parenthetical_identity_names"},
    }
    for name in seam.FILES:
        restored = corrected[name]
        for before, after in reversed(seam.DELTAS[name]):
            assert restored.count(after) == 1
            restored = restored.replace(after, before, 1)
        assert restored == original[name]
        trees = [ast.parse(data) for data in (original[name], corrected[name])]
        for tree in trees:
            tree.body = [node for node in tree.body if getattr(node, "name", "") not in allowed[name]]
            if name == builder.FINALITY_FILE:
                for node in tree.body:
                    if isinstance(node, ast.ImportFrom) and node.module == "qualification.scoring.company_fit_decision":
                        node.names = [alias for alias in node.names if alias.name != "company_quality_receipt_matches_claim"]
        assert ast.dump(trees[0]) == ast.dump(trees[1])


@pytest.mark.parametrize("fault", ["missing", "extra", "drift", "old_generation", "partial", "mixed"])
def test_unknown_or_partial_pair_fails_closed(fault):
    sources = selected_sources()
    if fault == "missing":
        sources.pop(builder.GRADER_FILE)
    elif fault == "extra":
        sources["extra.py"] = b"pass\n"
    elif fault == "drift":
        sources[builder.FINALITY_FILE] += b"\n"
    elif fault == "old_generation":
        sources[builder.GRADER_FILE] = feedback_controls.original_investigator()
    elif fault == "partial":
        before, after = seam.DELTAS[builder.FINALITY_FILE][0]
        sources[builder.FINALITY_FILE] = sources[builder.FINALITY_FILE].replace(before, after, 1)
    else:
        sources[builder.GRADER_FILE] = seam.convert_identity_handoff_files(sources)[builder.GRADER_FILE]
    with pytest.raises(seam.IdentityHandoffSeamError):
        seam.convert_identity_handoff_files(sources)


def test_unreviewed_transform_cannot_pass_postimage(monkeypatch):
    before, after = seam.DELTAS[builder.GRADER_FILE][0]
    monkeypatch.setitem(seam.DELTAS, builder.GRADER_FILE, ((before, after + b"\n"),))
    with pytest.raises(seam.IdentityHandoffSeamError, match="repaired checksum mismatch"):
        seam.convert_identity_handoff_files(selected_sources())


@pytest.mark.parametrize("missing", ordered.FLAGS)
def test_identity_handoff_requires_all_existing_options_before_docker(monkeypatch, missing):
    flags = [flag for flag in FLAGS if flag != missing]
    monkeypatch.setattr(builder.sys, "argv", ["compose", "--grading-image", "registry.example/reviewer@" + feedback.FROZEN_IMAGE_DIGEST, *flags])
    monkeypatch.setattr(builder, "image_inspect", lambda *_: pytest.fail("Docker must not run"))
    with pytest.raises(builder.CompositionError, match="requires"):
        builder.main()


@pytest.mark.parametrize("fault", ["base", "mode"])
def test_identity_handoff_rejects_other_bases_or_corrections_before_docker(monkeypatch, fault):
    digest = "sha256:" + "0" * 64 if fault == "base" else feedback.FROZEN_IMAGE_DIGEST
    flags = FLAGS + (["--complete-negative-finality"] if fault == "mode" else [])
    monkeypatch.setattr(builder.sys, "argv", ["compose", "--grading-image", "registry.example/reviewer@" + digest, *flags])
    monkeypatch.setattr(builder, "image_inspect", lambda *_: pytest.fail("Docker must not run"))
    with pytest.raises(builder.CompositionError, match="exact frozen50de|cannot include other corrections"):
        builder.main()


@pytest.mark.parametrize("verify", [False, True])
def test_build_and_verify_preserve_four_corrections_and_add_only_identity_pair(monkeypatch, capsys, verify):
    original = {builder.FINALITY_FILE: feedback_controls.FROZEN_FINALITY,
                builder.GRADER_FILE: feedback_controls.original_investigator(), **ordered.frozen_pair()}
    expected = {**seam.convert_identity_handoff_files(selected_sources()),
                retry.SCORING_FILE: retry.convert_ordered_company_retry_passes(ordered.frozen_pair())[retry.SCORING_FILE]}
    base_ref = "registry.example/reviewer@" + feedback.FROZEN_IMAGE_DIGEST
    runtime_ref = "registry.example/reviewer@sha256:" + "b" * 64
    argv = ["compose", "--grading-image", base_ref, *FLAGS]
    if verify:
        argv += ["--verify-image", runtime_ref]
    monkeypatch.setattr(builder.sys, "argv", argv)
    monkeypatch.setattr(builder, "checked_output", lambda *_a, **_k: "c" * 40)
    base = {"Architecture": "amd64", "Os": "linux", "Config": {"Env": ["FROZEN=1"], "User": "65534:65534"}, "RootFS": {"Layers": ["sha256:base"]}}
    candidate = copy.deepcopy(base)
    candidate["RootFS"]["Layers"].append("sha256:overlay")
    monkeypatch.setattr(builder, "image_inspect", lambda ref: base if ref == base_ref else candidate)

    def extract(ref, _destination, **flags):
        assert ref == base_ref and flags == {"submitted_company_context_only": True,
            "investigator_feedback": True, "ordered_company_retry_passes": True}
        return original

    def checked_overlay(_ref, digest, files):
        assert digest == "sha256:overlay" and files == expected
        builder.verify_overlay(layer([("model/" + path, data) for path, data in files.items()]), expected)

    monkeypatch.setattr(builder, "extract_frozen_sources", extract)
    monkeypatch.setattr(builder, "verify_saved_overlay", checked_overlay)
    commands = []
    monkeypatch.setattr(builder.subprocess, "run", lambda command, **_kwargs: commands.append(command))
    assert builder.main() == 0
    proof = json.loads(capsys.readouterr().out)
    assert proof["grading_image"] == base_ref
    assert proof["grading_corrections"] == [feedback_controls.visible.SUBMITTED_CORRECTION_ID,
        feedback_controls.acquisition.CORRECTION_ID, feedback.CORRECTION_ID,
        retry.ORDERED_RETRY_CORRECTION_ID, seam.CORRECTION_ID]
    assert proof["original_files"] == {name: hashlib.sha256(data).hexdigest() for name, data in original.items()}
    assert proof["overlay_files"] == {name: hashlib.sha256(data).hexdigest() for name, data in expected.items()}
    assert proof["base_layers_unchanged"] is proof["process_configuration_unchanged"] is True
    assert len(commands) == (0 if verify else 1)
