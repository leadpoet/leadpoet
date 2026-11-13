"""The frozen50de feedback overlay retains exact acquisition grading bytes."""
import copy
import hashlib
import json

import pytest

from scripts import arena_reviewer_active_acquisition_seam as acquisition
from scripts import arena_reviewer_investigator_feedback_seam as seam
from scripts import arena_reviewer_visible_auxiliary_context_seam as visible
from scripts import build_arena_reviewer_runtime as composer
from tests.lab_arena.reviewer_runtime_composition_test import layer
from tests.reviewer_frozen_sources import frozen_source

FINALITY = composer.FINALITY_FILE
INVESTIGATOR = composer.GRADER_FILE
FROZEN_FINALITY = frozen_source("submitted_company_context_d657", FINALITY)
CORRECTED_FINALITY = acquisition.convert_active_acquisition_seam(
    visible.convert_submitted_company_context_seam(FROZEN_FINALITY),
)
FLAGS = ["--submitted-company-context-only", "--active-acquisition-target", "--investigator-feedback"]


def original_investigator():
    source = (composer.ROOT / INVESTIGATOR).read_bytes()
    # The source correction can be stacked before this composer. Reverse only
    # its exact reviewed delta, then require the complete historical checksum.
    if hashlib.sha256(source).hexdigest() != seam.ORIGINAL_SHA256:
        for old, new in reversed(seam.DELTAS):
            assert source.count(new) == 1
            source = source.replace(new, old, 1)
    assert hashlib.sha256(source).hexdigest() == seam.ORIGINAL_SHA256
    return source


@pytest.mark.parametrize("flags", [
    ["--investigator-feedback"],
    ["--submitted-company-context-only", "--investigator-feedback"],
    ["--active-acquisition-target", "--investigator-feedback"],
])
def test_feedback_requires_both_existing_corrections_before_docker(monkeypatch, flags):
    monkeypatch.setattr(composer.sys, "argv", ["composer", "--grading-image", "registry.example/reviewer@" + seam.FROZEN_IMAGE_DIGEST, *flags])
    monkeypatch.setattr(composer, "image_inspect", lambda *_: pytest.fail("Docker must not run"))
    with pytest.raises(composer.CompositionError, match="requires submitted-company-context-only and active-acquisition-target"):
        composer.main()


@pytest.mark.parametrize("digest", [visible.SUBMITTED_FROZEN_IMAGE_DIGEST, "sha256:" + "0" * 64])
def test_feedback_rejects_other_frozen_bases_before_docker(monkeypatch, digest):
    monkeypatch.setattr(composer.sys, "argv", ["composer", "--grading-image", "registry.example/reviewer@" + digest, *FLAGS])
    monkeypatch.setattr(composer, "image_inspect", lambda *_: pytest.fail("Docker must not run"))
    with pytest.raises(composer.CompositionError, match="exact frozen50de"):
        composer.main()


@pytest.mark.parametrize("flag", ["--shim-only", "--complete-negative-finality", "--verified-display-brand", "--source-access-denied"])
def test_feedback_cannot_import_unreviewed_corrections(monkeypatch, flag):
    monkeypatch.setattr(composer.sys, "argv", ["composer", "--grading-image", "registry.example/reviewer@" + seam.FROZEN_IMAGE_DIGEST, *FLAGS, flag])
    monkeypatch.setattr(composer, "image_inspect", lambda *_: pytest.fail("Docker must not run"))
    with pytest.raises(composer.CompositionError, match="cannot include other corrections"):
        composer.main()


@pytest.mark.parametrize("verify_existing", [False, True])
def test_composer_preserves_frozen_layers_process_and_acquisition(monkeypatch, capsys, verify_existing):
    frozen = original_investigator()
    # Isolate the builder contract from the semantic transform tests below.
    corrected = frozen + b"\n# reviewed test transform\n"
    monkeypatch.setattr(seam, "convert_investigator_feedback_seam", lambda source: corrected if source == frozen else pytest.fail("wrong frozen investigator"))
    base_ref = "registry.example/reviewer@" + seam.FROZEN_IMAGE_DIGEST
    tag = composer.bounded_build_tag("registry.example/reviewer:reviewer-" + seam.FROZEN_IMAGE_DIGEST[7:19] + "-" + "c" * 12 + "-submitted-company-context-active-acquisition-target-investigator-feedback")
    candidate_ref = "registry.example/reviewer@sha256:" + "b" * 64 if verify_existing else tag
    argv = ["composer", "--grading-image", base_ref, *FLAGS]
    if verify_existing:
        argv += ["--verify-image", candidate_ref]
    monkeypatch.setattr(composer.sys, "argv", argv)
    monkeypatch.setattr(composer, "checked_output", lambda *_a, **_k: "c" * 40)
    base = {"Architecture": "amd64", "Os": "linux", "Config": {"Env": ["FROZEN=1"], "Entrypoint": ["python", "scorer_entrypoint.py"], "User": "65534:65534"}, "RootFS": {"Layers": ["sha256:base"]}}
    candidate = copy.deepcopy(base)
    candidate["RootFS"]["Layers"].append("sha256:overlay")
    monkeypatch.setattr(composer, "image_inspect", lambda ref: base if ref == base_ref else candidate)
    def extract(ref, _destination, **flags):
        assert ref == base_ref and flags == {"submitted_company_context_only": True, "investigator_feedback": True}
        return {FINALITY: FROZEN_FINALITY, INVESTIGATOR: frozen}
    expected = {FINALITY: CORRECTED_FINALITY, INVESTIGATOR: corrected}
    def verify(ref, digest, files):
        assert ref == candidate_ref and digest == "sha256:overlay"
        assert files == expected
        composer.verify_overlay(layer([("model/" + path, data) for path, data in files.items()]), expected)
    monkeypatch.setattr(composer, "extract_frozen_sources", extract)
    monkeypatch.setattr(composer, "verify_saved_overlay", verify)
    builds = []
    monkeypatch.setattr(composer.subprocess, "run", lambda command, **_k: builds.append(command))
    assert composer.main() == 0
    proof = json.loads(capsys.readouterr().out)
    assert proof["grading_image"] == base_ref
    assert proof["grading_corrections"] == [visible.SUBMITTED_CORRECTION_ID, acquisition.CORRECTION_ID, seam.CORRECTION_ID]
    assert proof["original_files"] == {FINALITY: visible.SUBMITTED_ORIGINAL_SHA256, INVESTIGATOR: seam.ORIGINAL_SHA256}
    assert proof["overlay_files"] == {path: hashlib.sha256(data).hexdigest() for path, data in expected.items()}
    assert proof["overlay_files"][FINALITY] == "7994f7574b5e955d56c86d8a3a73a41f30d4dda302e7e0fcfa4a0e197f3d7bcd"
    assert proof["base_layers_unchanged"] is proof["process_configuration_unchanged"] is True
    assert len(builds) == (0 if verify_existing else 1)
