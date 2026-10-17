"""The deployment tool cannot replace grading bytes or change process imports."""
import io
import tarfile
from copy import deepcopy

import pytest

from scripts.build_arena_reviewer_runtime import (
    CompositionError, verify_image_configuration, verify_overlay,
)


def layer(files, *, mode=0o644, kind=tarfile.REGTYPE):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        for name, content in files:
            info = tarfile.TarInfo(name)
            info.mode = mode
            info.type = kind
            info.size = len(content) if kind == tarfile.REGTYPE else 0
            archive.addfile(info, io.BytesIO(content) if info.size else None)
    return stream.getvalue()


def test_overlay_requires_exact_reviewed_bytes():
    expected = {"arena_reviewer_runtime/admission.py": b"runtime", "qualification/core.py": b"frozen"}
    verify_overlay(layer([("model/" + k, v) for k, v in expected.items()]), expected)
    with pytest.raises(CompositionError):
        verify_overlay(layer([("model/" + k, b"modified") for k in expected]), expected)


@pytest.mark.parametrize("name", [
    "model/sitecustomize.py", "model/lab_arena/scoring.py", "etc/passwd",
    "/model/arena_reviewer_runtime/admission.py", "model/../etc/passwd",
    "model/.wh.qualification", "model/arena_reviewer_runtime/.wh.admission.py",
])
def test_unapproved_paths_fail(name):
    with pytest.raises(CompositionError):
        verify_overlay(layer([(name, b"x")]), {"arena_reviewer_runtime/admission.py": b"x"})


@pytest.mark.parametrize("kind", [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.CHRTYPE])
def test_links_and_devices_fail(kind):
    with pytest.raises(CompositionError):
        verify_overlay(layer([("model/arena_reviewer_runtime/admission.py", b"x")], kind=kind),
                       {"arena_reviewer_runtime/admission.py": b"x"})


@pytest.mark.parametrize("mode", [0o666, 0o4644, 0o2644])
def test_unsafe_modes_fail(mode):
    with pytest.raises(CompositionError):
        verify_overlay(layer([("model/arena_reviewer_runtime/admission.py", b"x")], mode=mode),
                       {"arena_reviewer_runtime/admission.py": b"x"})


def test_missing_duplicate_and_extra_files_fail():
    expected = {"arena_reviewer_runtime/admission.py": b"x"}
    for files in [[], [("model/arena_reviewer_runtime/admission.py", b"x")] * 2,
                  [("model/arena_reviewer_runtime/admission.py", b"x"), ("model/qualification/core.py", b"changed")]]:
        with pytest.raises(CompositionError):
            verify_overlay(layer(files), expected)


def test_opaque_directory_metadata_cannot_hide_frozen_grading_files():
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w", format=tarfile.PAX_FORMAT) as archive:
        directory = tarfile.TarInfo("model/qualification")
        directory.type = tarfile.DIRTYPE
        directory.mode = 0o755
        directory.pax_headers = {"SCHILY.xattr.trusted.overlay.opaque": "y"}
        archive.addfile(directory)
        item = tarfile.TarInfo("model/qualification/core.py")
        item.size = 6
        item.mode = 0o644
        archive.addfile(item, io.BytesIO(b"frozen"))
    with pytest.raises(CompositionError, match="metadata"):
        verify_overlay(stream.getvalue(), {"qualification/core.py": b"frozen"})


def test_base_and_configuration_are_immutable():
    base = {"Architecture": "amd64", "Os": "linux", "Config": {
        "Entrypoint": ["python3", "/model/scorer_entrypoint.py"], "User": "65534:65534",
        "Env": ["PYTHONPATH=/model"], "WorkingDir": "/model",
    }, "RootFS": {"Layers": ["sha256:base"]}}
    candidate = deepcopy(base)
    candidate["RootFS"]["Layers"].append("sha256:repair")
    assert verify_image_configuration(base, candidate) == ["sha256:repair"]
    for field, value in [("Entrypoint", ["python3", "/alternate.py"]), ("User", "0:0"),
                         ("Env", ["PYTHONPATH=/mutable"]), ("WorkingDir", "/tmp")]:
        bad = deepcopy(candidate)
        bad["Config"][field] = value
        with pytest.raises(CompositionError):
            verify_image_configuration(base, bad)
    for layers in [["sha256:other", "sha256:repair"], ["sha256:base"],
                   ["sha256:base", "sha256:repair", "sha256:extra"]]:
        bad = deepcopy(candidate)
        bad["RootFS"]["Layers"] = layers
        with pytest.raises(CompositionError):
            verify_image_configuration(base, bad)


@pytest.mark.parametrize("negative_finality", [False, True])
def test_source_access_correction_is_explicit_and_composes(negative_finality):
    from scripts import build_arena_reviewer_runtime as builder
    from scripts.arena_reviewer_source_access_seam import (
        INVESTIGATOR_DELTAS, FINALITY_DELTAS,
        convert_source_access_seam, convert_source_access_finality_seam,
    )

    original = {
        name: (builder.ROOT / ("lab_arena/scorer_entrypoint.py"
                              if name == "scorer_entrypoint.py" else name)).read_bytes()
        for name in (builder.GRADER_FILE, builder.FINALITY_FILE, *builder.ENTRYPOINT_FILES)
    }
    # Model a frozen pre-correction image without depending on Git history.
    for name, deltas in ((builder.GRADER_FILE, INVESTIGATOR_DELTAS),
                         (builder.FINALITY_FILE, FINALITY_DELTAS)):
        for _, old, new in deltas:
            assert original[name].count(new.encode()) == 1
            original[name] = original[name].replace(new.encode(), old.encode(), 1)
    prior_sources = dict(original)
    if not negative_finality:
        prior_sources.pop(builder.FINALITY_FILE)
    prior = builder.expected_overlay(
        prior_sources, complete_negative_finality=negative_finality,
    )
    repaired = builder.expected_overlay(
        original, complete_negative_finality=negative_finality,
        source_access_denied=True,
    )
    assert repaired[builder.GRADER_FILE] == convert_source_access_seam(prior[builder.GRADER_FILE])
    assert repaired[builder.FINALITY_FILE] == convert_source_access_finality_seam(
        prior.get(builder.FINALITY_FILE, original[builder.FINALITY_FILE]))
    for name in set(prior) - {builder.GRADER_FILE, builder.FINALITY_FILE}:
        assert repaired[name] == prior[name]
    assert set(repaired) == set(prior) | {builder.FINALITY_FILE}
