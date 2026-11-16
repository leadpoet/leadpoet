"""Frozen public-buyer repair: exact failure, strict controls, image boundary."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest

from scripts import build_arena_reviewer_runtime as composer
from scripts import arena_reviewer_public_buyer_seam as seam
from tests import test_public_acquisition_buyer as controls
from tests.test_public_issuer_ticker_quotes import _validated_public
from tests.lab_arena.reviewer_runtime_composition_test import layer
from tests.reviewer_frozen_sources import frozen_source

pytestmark = pytest.mark.usefixtures("scorer_process_environment")
ORIGINAL = frozen_source("public_buyer2ec", seam.FINALITY_FILE)
CORRECTED = seam.convert_public_buyer_seam(ORIGINAL)


def test_exact_source_changes_only_the_two_reviewed_predicates():
    assert hashlib.sha256(ORIGINAL).hexdigest() == seam.ORIGINAL_SHA256
    assert hashlib.sha256(CORRECTED).hexdigest() == seam.REPAIRED_SHA256
    assert seam.convert_public_buyer_seam(CORRECTED) == CORRECTED
    allowed = {"_stage_quote_supports_observation", "_stage_evidence_supports_observation"}
    def unchanged(source):
        return [ast.dump(node) for node in ast.parse(source).body
                if getattr(node, "name", "") not in allowed]
    assert unchanged(ORIGINAL) == unchanged(CORRECTED)


def install(monkeypatch, source):
    import qualification.scoring
    module = ModuleType("_public_buyer_replay")
    exec(compile(source, "frozen_public_buyer.py", "exec"), vars(module))
    monkeypatch.setattr(controls, "scorer", module)
    monkeypatch.setattr(qualification.scoring, "lead_scorer", module)
    monkeypatch.setitem(sys.modules, "qualification.scoring.lead_scorer", module)
    return module


def test_actual_failed_quote_reproduces_then_passes_unchanged_validation(monkeypatch):
    quote = "OneSpan Inc. (NASDAQ: OSPN) today announced that it has completed the acquisition of Build38"
    install(monkeypatch, ORIGINAL)
    assert _validated_public(quote, company="OneSpan", domain="onespan.com")["status"] == "UNPROVEN"
    install(monkeypatch, CORRECTED)
    assert _validated_public(quote, company="OneSpan", domain="onespan.com")["status"] == "VERIFIED"


@pytest.mark.parametrize("company,domain,quote", controls.test_issuer_buying_another_company_keeps_its_listing.pytestmark[0].args[1])
def test_corrected_frozen_positive_controls(monkeypatch, company, domain, quote):
    install(monkeypatch, CORRECTED)
    controls.test_issuer_buying_another_company_keeps_its_listing(company, domain, quote)


@pytest.mark.parametrize("quote", controls.test_target_parent_other_issuer_and_superseded_listing_do_not_pass.pytestmark[0].args[1])
def test_corrected_frozen_negative_controls(monkeypatch, quote):
    install(monkeypatch, CORRECTED)
    controls.test_target_parent_other_issuer_and_superseded_listing_do_not_pass(quote)


def test_corrected_frozen_source_identity_and_saved_result_controls(monkeypatch):
    install(monkeypatch, CORRECTED)
    controls.test_missing_identity_does_not_resolve_ambiguous_listing_and_acquisition()
    controls.test_exact_quote_and_source_presence_remain_mandatory()
    controls.test_public_buyer_survives_saved_verdict_projection()


@pytest.mark.parametrize("source", [
    b"", b"\xff", ORIGINAL + b"\n", ORIGINAL.replace(b"\n", b"\r\n"),
    ORIGINAL + ORIGINAL, CORRECTED + b"# unrelated\n",
    ORIGINAL.replace(b'"rebrand"', b'"unknown"', 1),
    ORIGINAL.replace(seam._OLD_0.encode(), seam._NEW_0.encode(), 1),
], ids=lambda source: hashlib.sha256(source).hexdigest()[:12])
def test_unknown_or_partial_sources_fail_closed(source):
    with pytest.raises(seam.PublicBuyerSeamError):
        seam.convert_public_buyer_seam(source)


@pytest.mark.parametrize("verify_existing", [False, True])
@pytest.mark.parametrize("hq_feedback", [False, True])
def test_exclusive_composer_build_and_verification_are_exact_files(monkeypatch, capsys, verify_existing, hq_feedback):
    base_ref = "registry.example/reviewer@" + seam.FROZEN_IMAGE_DIGEST
    tag = "registry.example/reviewer:reviewer-" + seam.FROZEN_IMAGE_DIGEST[7:19] + "-" + "c" * 12 + "-public-issuer-buyer"
    if hq_feedback:
        tag += "-hq-feedback"
    candidate_ref = "registry.example/reviewer@sha256:" + "b" * 64 if verify_existing else tag
    args = ["build_arena_reviewer_runtime.py", "--grading-image", base_ref,
            "--public-issuer-buyer-only"]
    expected_original = {seam.FINALITY_FILE: ORIGINAL}
    expected_files = {seam.FINALITY_FILE: CORRECTED}
    if hq_feedback:
        args += ["--linkedin-headquarters-feedback"]
        expected_original[seam.HEADQUARTERS_FILE] = frozen_source("linkedin_hq_feedback2ec", seam.HEADQUARTERS_FILE)
        expected_files[seam.HEADQUARTERS_FILE] = (composer.ROOT / seam.HEADQUARTERS_FILE).read_bytes()
    if verify_existing:
        args += ["--verify-image", candidate_ref]
    monkeypatch.setattr(composer.sys, "argv", args)
    monkeypatch.setattr(composer, "checked_output", lambda *_a, **_k: "c" * 40)
    base = {"Architecture": "amd64", "Os": "linux", "Config": {},
            "RootFS": {"Layers": ["sha256:base"]}}
    candidate = copy.deepcopy(base)
    candidate["RootFS"]["Layers"].append("sha256:overlay")
    monkeypatch.setattr(composer, "image_inspect", lambda ref: base if ref == base_ref else candidate)

    def extract(ref, _destination, **flags):
        assert ref == base_ref
        assert flags == {"public_issuer_buyer_only": True, **({"linkedin_headquarters_feedback": True} if hq_feedback else {})}
        return expected_original

    def verify(ref, digest, files):
        assert ref == candidate_ref and digest == "sha256:overlay"
        assert files == expected_files

    monkeypatch.setattr(composer, "extract_frozen_sources", extract)
    monkeypatch.setattr(composer, "verify_saved_overlay", verify)
    builds = []
    monkeypatch.setattr(composer.subprocess, "run", lambda command, **_k: builds.append(command))
    assert composer.main() == 0
    proof = json.loads(capsys.readouterr().out)
    assert proof["grading_image"] == base_ref
    assert proof["grading_corrections"] == ([seam.HEADQUARTERS_CORRECTION_ID] if hq_feedback else []) + [seam.CORRECTION_ID]
    assert proof["original_files"] == {name: hashlib.sha256(data).hexdigest() for name, data in expected_original.items()}
    assert proof["overlay_files"] == {name: hashlib.sha256(data).hexdigest() for name, data in expected_files.items()}
    assert proof["base_layers_unchanged"] is proof["process_configuration_unchanged"] is True
    assert len(builds) == (0 if verify_existing else 1)


def test_headquarters_feedback_only_changes_reviewed_diagnostics_and_guidance():
    original = frozen_source("linkedin_hq_feedback2ec", seam.HEADQUARTERS_FILE)
    corrected = (composer.ROOT / seam.HEADQUARTERS_FILE).read_bytes()
    assert seam.verified_headquarters_feedback(original, corrected) == corrected
    def preserved(source):
        return [ast.dump(node) for node in ast.parse(source).body
                if getattr(node, 'name', '') not in {'_validated_findings', 'investigate_company_evidence'}
                and not (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'BOUND_LINKEDIN_HEADQUARTERS_PROOF_REASON' for t in node.targets))]
    assert preserved(original) == preserved(corrected)
    for before, after in ((original + b'\n', corrected), (original, corrected + b'\n'), (corrected, corrected)):
        with pytest.raises(seam.PublicBuyerSeamError):
            seam.verified_headquarters_feedback(before, after)


def test_headquarters_feedback_requires_exact_composition_before_docker(monkeypatch):
    monkeypatch.setattr(composer.sys, 'argv', ['builder', '--grading-image',
        'registry.example/base@' + seam.FROZEN_IMAGE_DIGEST, '--linkedin-headquarters-feedback'])
    with pytest.raises(composer.CompositionError, match='requires the frozen public-buyer repair'):
        composer.main()


@pytest.mark.parametrize("other", [
    "--shim-only", "--empty-domain-negative-finality-only", "--identity-schema-repair",
    "--verified-display-brand", "--complete-negative-finality", "--source-access-denied",
    "--intent-target-denial", "--homepage-profile-alias",
])
def test_exclusive_mode_rejects_mixed_corrections_before_docker(monkeypatch, other):
    monkeypatch.setattr(composer.sys, "argv", [
        "build_arena_reviewer_runtime.py", "--grading-image", "registry.example/base@" + seam.FROZEN_IMAGE_DIGEST,
        "--public-issuer-buyer-only", other,
    ])
    with pytest.raises(composer.CompositionError, match="cannot include other corrections"):
        composer.main()


def test_mode_rejects_other_grading_image_before_docker(monkeypatch):
    monkeypatch.setattr(composer.sys, "argv", [
        "build_arena_reviewer_runtime.py", "--grading-image", "registry.example/base@sha256:" + "a" * 64,
        "--public-issuer-buyer-only",
    ])
    with pytest.raises(composer.CompositionError, match="exact verified grading image"):
        composer.main()


def test_overlay_rejects_extra_or_changed_files():
    expected = {seam.FINALITY_FILE: CORRECTED}
    composer.verify_overlay(layer([("model/" + seam.FINALITY_FILE, CORRECTED)]), expected)
    for files in (
        [("model/" + seam.FINALITY_FILE, CORRECTED), ("model/unrelated.py", b"pass\n")],
        [("model/" + seam.FINALITY_FILE, CORRECTED + b"\n")],
    ):
        with pytest.raises(composer.CompositionError):
            composer.verify_overlay(layer(files), expected)


def test_registration_only_selects_verified_runtime_under_frozen_grading_identity():
    from scripts.register_arena_reviewer_runtime import registration_arguments
    original = "registry.example/reviewer@" + seam.FROZEN_IMAGE_DIGEST
    runtime = "registry.example/reviewer@sha256:" + "b" * 64
    args = registration_arguments(original, runtime)
    assert args["p_grading_image_digest"] == seam.FROZEN_IMAGE_DIGEST
    assert args["p_runtime_image_reference"] == runtime


def test_mode_extracts_only_the_exact_frozen_lead_scorer(monkeypatch, tmp_path):
    commands = []
    reference = "registry.example/reviewer@" + seam.FROZEN_IMAGE_DIGEST
    monkeypatch.setattr(composer, "checked_output", lambda command: "test-container")

    def docker(command, **kwargs):
        commands.append(command)
        if command[:2] == ["docker", "cp"]:
            assert command[2] == "test-container:/model/" + seam.FINALITY_FILE
            Path(command[3]).write_bytes(ORIGINAL)

    monkeypatch.setattr(composer.subprocess, "run", docker)
    assert composer.extract_frozen_sources(
        reference, tmp_path, public_issuer_buyer_only=True,
    ) == {seam.FINALITY_FILE: ORIGINAL}
    assert [command[:2] for command in commands] == [["docker", "cp"], ["docker", "rm"]]

