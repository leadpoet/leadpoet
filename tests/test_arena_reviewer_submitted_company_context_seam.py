"""The exact d657 overlay changes only optional submitted-source admission."""
import ast
import copy
import hashlib
import io
import json
import sys
import tarfile
from types import ModuleType

import pytest

from scripts import arena_reviewer_visible_auxiliary_context_seam as seam
from scripts import build_arena_reviewer_runtime as composer
from tests import test_submitted_company_context_visible_binding as controls
from tests.reviewer_frozen_sources import frozen_source


FILE = "qualification/scoring/lead_scorer.py"
ORIGINAL = frozen_source("submitted_company_context_d657", FILE)
CORRECTED = seam.convert_submitted_company_context_seam(ORIGINAL)


def test_exact_source_correction_preserves_all_other_frozen_code():
    assert hashlib.sha256(ORIGINAL).hexdigest() == seam.SUBMITTED_ORIGINAL_SHA256
    assert hashlib.sha256(CORRECTED).hexdigest() == seam.SUBMITTED_REPAIRED_SHA256
    assert seam.convert_submitted_company_context_seam(CORRECTED) == CORRECTED
    assert CORRECTED.decode().replace(seam._SUBMITTED_AFTER, seam._SUBMITTED_BEFORE, 1).encode() == ORIGINAL
    before, after = ast.parse(ORIGINAL), ast.parse(CORRECTED)
    for tree in (before, after):
        tree.body = [node for node in tree.body if getattr(node, "name", "") != "_matched_company_source_contexts"]
    assert ast.dump(before) == ast.dump(after)


@pytest.mark.parametrize("source", [
    b"", b"\xff", ORIGINAL + b"\n", CORRECTED + b"\n",
    ORIGINAL.replace(b"\n", b"\r\n"),
    ORIGINAL.replace(b'"first_party_company"', b'"first_party_other"', 1),
])
def test_unknown_or_partial_source_fails_closed(source):
    with pytest.raises(seam.VisibleAuxiliaryContextSeamError):
        seam.convert_submitted_company_context_seam(source)


def test_frozen_source_reproduces_then_corrects_real_preflight(monkeypatch):
    def install(source):
        module = ModuleType("_frozen_submitted_company_context")
        monkeypatch.setitem(sys.modules, module.__name__, module)
        exec(compile(source, "frozen_submitted_company_context.py", "exec"), vars(module))
        monkeypatch.setattr(controls, "lead_scorer", module)
    install(ORIGINAL)
    marker = controls._IDENTITY_LINK_CONTEXT_MARKER
    with pytest.raises(AssertionError):
        controls.test_metadata_only_name_or_quote_cannot_poison_paragraph_preflight(
            "Accept payments online.", marker + " Ryft",
        )
    install(CORRECTED)
    controls.test_metadata_only_name_or_quote_cannot_poison_paragraph_preflight(
        "Accept payments online.", marker + " Ryft",
    )
    controls.test_visible_submitted_company_context_keeps_exact_existing_proof(marker + " unread locator")
    controls.test_manual_metadata_only_company_context_still_fails_native_validator()
    for change in ("identity", "redirect"):
        controls.test_unbound_optional_page_stays_excluded(change)


@pytest.mark.parametrize("verify_existing", [False, True])
def test_composer_proves_one_file_one_layer_and_unchanged_configuration(monkeypatch, capsys, verify_existing):
    base_ref = "registry.example/reviewer@" + seam.SUBMITTED_FROZEN_IMAGE_DIGEST
    tag = composer.bounded_build_tag(
        "registry.example/reviewer:reviewer-" + seam.SUBMITTED_FROZEN_IMAGE_DIGEST[7:19]
        + "-" + "c" * 12 + "-submitted-company-context"
    )
    candidate_ref = "registry.example/reviewer@sha256:" + "b" * 64 if verify_existing else tag
    args = ["build_arena_reviewer_runtime.py", "--grading-image", base_ref, "--submitted-company-context-only"]
    if verify_existing:
        args += ["--verify-image", candidate_ref]
    monkeypatch.setattr(composer.sys, "argv", args)
    monkeypatch.setattr(composer, "checked_output", lambda *_a, **_k: "c" * 40)
    base = {"Architecture": "amd64", "Os": "linux", "Config": {"Env": ["FROZEN=1"]},
            "RootFS": {"Layers": ["sha256:base"]}}
    candidate = copy.deepcopy(base)
    candidate["RootFS"]["Layers"].append("sha256:overlay")
    monkeypatch.setattr(composer, "image_inspect", lambda ref: base if ref == base_ref else candidate)
    def extract(ref, _destination, **flags):
        assert ref == base_ref and flags == {"submitted_company_context_only": True}
        return {FILE: ORIGINAL}
    def verify(ref, digest, files):
        assert ref == candidate_ref and digest == "sha256:overlay"
        assert files == {FILE: CORRECTED}
    monkeypatch.setattr(composer, "extract_frozen_sources", extract)
    monkeypatch.setattr(composer, "verify_saved_overlay", verify)
    builds = []
    monkeypatch.setattr(composer.subprocess, "run", lambda command, **_k: builds.append(command))
    assert composer.main() == 0
    proof = json.loads(capsys.readouterr().out)
    assert proof["grading_corrections"] == [seam.SUBMITTED_CORRECTION_ID]
    assert proof["overlay_files"] == {FILE: seam.SUBMITTED_REPAIRED_SHA256}
    assert proof["base_layers_unchanged"] is proof["process_configuration_unchanged"] is True
    assert len(builds) == (0 if verify_existing else 1)


@pytest.mark.parametrize("options", [["--shim-only"], ["--visible-auxiliary-context"], ["--identity-repair-finality-only"]])
def test_exclusive_mode_rejects_other_repairs_before_docker(monkeypatch, options):
    monkeypatch.setattr(composer.sys, "argv", ["build_arena_reviewer_runtime.py", "--grading-image",
        "registry.example/reviewer@" + seam.SUBMITTED_FROZEN_IMAGE_DIGEST,
        "--submitted-company-context-only", *options])
    with pytest.raises(composer.CompositionError, match="cannot include other corrections"):
        composer.main()


def test_wrong_frozen_image_fails_before_docker(monkeypatch):
    monkeypatch.setattr(composer.sys, "argv", ["build_arena_reviewer_runtime.py", "--grading-image",
        "registry.example/reviewer@sha256:" + "0" * 64, "--submitted-company-context-only"])
    with pytest.raises(composer.CompositionError, match="exact frozen d657"):
        composer.main()


@pytest.mark.parametrize("change", [None, "bytes", "extra_file"])
def test_actual_one_file_layer_bytes_are_verified(change):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        data = CORRECTED if change != "bytes" else b"x" + CORRECTED[1:]
        member = tarfile.TarInfo("model/" + FILE)
        member.size, member.mode = len(data), 0o644
        archive.addfile(member, io.BytesIO(data))
        if change == "extra_file":
            extra = tarfile.TarInfo("model/unapproved.py")
            extra.mode = 0o644
            archive.addfile(extra, io.BytesIO(b""))
    if change is None:
        composer.verify_overlay(stream.getvalue(), {FILE: CORRECTED})
    else:
        with pytest.raises(composer.CompositionError):
            composer.verify_overlay(stream.getvalue(), {FILE: CORRECTED})


@pytest.mark.parametrize("change", ["config", "base_layer", "extra_layer"])
def test_configuration_and_existing_layers_cannot_change(change):
    base = {"Architecture": "amd64", "Os": "linux", "Config": {"Env": ["FROZEN=1"]},
            "RootFS": {"Layers": ["sha256:base"]}}
    candidate = copy.deepcopy(base)
    candidate["RootFS"]["Layers"].append("sha256:overlay")
    if change == "config":
        candidate["Config"]["Env"] = ["FROZEN=0"]
    elif change == "base_layer":
        candidate["RootFS"]["Layers"][0] = "sha256:other"
    else:
        candidate["RootFS"]["Layers"].append("sha256:extra")
    with pytest.raises(composer.CompositionError):
        composer.verify_image_configuration(base, candidate)
