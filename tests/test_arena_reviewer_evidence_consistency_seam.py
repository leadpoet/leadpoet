"""The correction changes only audited evidence predicates on frozen2ec."""
import ast
import copy
import hashlib
import json

import pytest

from scripts import arena_reviewer_evidence_consistency_seam as seam
from scripts import build_arena_reviewer_runtime as composer
from tests.reviewer_frozen_sources import frozen_source
from tests.lab_arena.reviewer_runtime_composition_test import layer


ORIGINAL = {name: frozen_source("evidence_consistency2ec", name) for name in seam.FILES}
CORRECTED = {name: frozen_source("evidence_consistency_repaired", name) for name in seam.FILES}


def test_exact_three_file_delta_preserves_every_other_top_level_definition():
    assert seam.verified_consistency_files(ORIGINAL, CORRECTED) == CORRECTED
    allowed = {
        "intent_verification_three_stage.py": {
            "_grounded_exact_text", "_supplied_url_for_miscopied_citation",
            "_apply_missing_date_policy", "_apply_guardrails",
        },
        "company_evidence_investigator.py": {
            "BOUND_LINKEDIN_HEADQUARTERS_PROOF_REASON", "_quote_supports_headquarters",
            "_HEADQUARTERS_SUBDIVISION_COUNTRIES", "_headquarters_field_subdivision_country",
            "_validated_findings", "investigate_company_evidence",
        },
        "lead_scorer.py": {
            "_stage_quote_supports_observation", "_stage_evidence_supports_observation",
            "_project_verified_linkedin_headquarters",
        },
    }
    def retained(source, names):
        result = []
        for node in ast.parse(source).body:
            if getattr(node, "name", "") in names:
                continue
            if isinstance(node, ast.Assign) and any(getattr(x, "id", "") in names for x in node.targets):
                continue
            if isinstance(node, ast.ImportFrom):
                node.names = [x for x in node.names if x.name != "_headquarters_field_subdivision_country"]
            result.append(ast.dump(node))
        return result
    for name in seam.FILES:
        names = allowed[name.rsplit("/", 1)[-1]]
        assert retained(ORIGINAL[name], names) == retained(CORRECTED[name], names)


@pytest.mark.parametrize("side", ["original", "corrected"])
@pytest.mark.parametrize("fault", ["missing", "extra", "changed"])
def test_unknown_missing_or_extra_source_is_not_admitted(side, fault):
    original, corrected = dict(ORIGINAL), dict(CORRECTED)
    target = original if side == "original" else corrected
    if fault == "missing":
        target.pop(seam.FILES[0])
    elif fault == "extra":
        target["qualification/scoring/prompts/_common.py"] = b"pass\n"
    else:
        target[seam.FILES[0]] += b"\n"
    with pytest.raises(seam.EvidenceConsistencySeamError):
        seam.verified_consistency_files(original, corrected)


@pytest.mark.parametrize("verify_existing", [False, True])
def test_actual_composer_keeps_layers_configuration_and_exact_file_set(monkeypatch, capsys, tmp_path, verify_existing):
    # Historical composition uses its exact reviewed source, not later changes.
    for name, source in CORRECTED.items():
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source)
    monkeypatch.setattr(composer, "ROOT", tmp_path)
    base_ref = "registry.example/reviewer@" + seam.FROZEN_IMAGE_DIGEST
    tag = "registry.example/reviewer:reviewer-" + seam.FROZEN_IMAGE_DIGEST[7:19] + "-" + "c" * 12 + "-evidence-consistency"
    candidate_ref = "registry.example/reviewer@sha256:" + "b" * 64 if verify_existing else tag
    args = ["builder", "--grading-image", base_ref, "--evidence-consistency-only"]
    if verify_existing:
        args += ["--verify-image", candidate_ref]
    monkeypatch.setattr(composer.sys, "argv", args)
    monkeypatch.setattr(composer, "checked_output", lambda *_a, **_k: "c" * 40)
    base = {"Architecture": "amd64", "Os": "linux", "Config": {}, "RootFS": {"Layers": ["sha256:base"]}}
    candidate = copy.deepcopy(base)
    candidate["RootFS"]["Layers"].append("sha256:overlay")
    monkeypatch.setattr(composer, "image_inspect", lambda ref: base if ref == base_ref else candidate)
    def extract(ref, _destination, **flags):
        assert ref == base_ref and flags == {"evidence_consistency_only": True}
        return ORIGINAL
    def verify(ref, digest, files):
        assert ref == candidate_ref and digest == "sha256:overlay" and files == CORRECTED
    monkeypatch.setattr(composer, "extract_frozen_sources", extract)
    monkeypatch.setattr(composer, "verify_saved_overlay", verify)
    builds = []
    monkeypatch.setattr(composer.subprocess, "run", lambda command, **_k: builds.append(command))
    assert composer.main() == 0
    proof = json.loads(capsys.readouterr().out)
    assert proof["grading_corrections"] == seam.CORRECTIONS
    assert proof["original_files"] == seam.ORIGINAL_SHA256
    assert proof["overlay_files"] == seam.REPAIRED_SHA256
    assert proof["base_layers_unchanged"] is proof["process_configuration_unchanged"] is True
    assert len(builds) == (0 if verify_existing else 1)


@pytest.mark.parametrize("other", ["--public-issuer-buyer-only", "--shim-only", "--linkedin-headquarters-feedback"])
def test_exclusive_mode_rejects_other_corrections_before_docker(monkeypatch, other):
    monkeypatch.setattr(composer.sys, "argv", ["builder", "--grading-image",
        "registry.example/reviewer@" + seam.FROZEN_IMAGE_DIGEST, "--evidence-consistency-only", other])
    with pytest.raises(composer.CompositionError, match="cannot include other corrections"):
        composer.main()


def test_different_grading_image_rejected_before_docker(monkeypatch):
    monkeypatch.setattr(composer.sys, "argv", ["builder", "--grading-image",
        "registry.example/reviewer@sha256:" + "a" * 64, "--evidence-consistency-only"])
    with pytest.raises(composer.CompositionError, match="exact verified grading image"):
        composer.main()


def test_overlay_cannot_change_prompts_rules_or_other_packaged_files():
    files = [("model/" + name, source) for name, source in CORRECTED.items()]
    composer.verify_overlay(layer(files), CORRECTED)
    with pytest.raises(composer.CompositionError):
        composer.verify_overlay(layer(files + [("model/lab_arena/scoring.py", b"pass\n")]), CORRECTED)


def test_extract_reads_only_the_three_reviewed_files(monkeypatch, tmp_path):
    monkeypatch.setattr(composer, "checked_output", lambda _command: "owned-test-container")
    seen = []
    def docker(command, **_kwargs):
        if command[:2] == ["docker", "cp"]:
            from pathlib import Path
            name = command[2].split(":/model/", 1)[1]
            assert name in ORIGINAL
            Path(command[3]).write_bytes(ORIGINAL[name])
            seen.append(name)
    monkeypatch.setattr(composer.subprocess, "run", docker)
    assert composer.extract_frozen_sources("base", tmp_path, evidence_consistency_only=True) == ORIGINAL
    assert set(seen) == set(seam.FILES)
