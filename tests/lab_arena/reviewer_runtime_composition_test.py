"""The deployment tool cannot replace grading bytes or change process imports."""
import ast
import hashlib
import io
import os
import tarfile
from copy import deepcopy
from pathlib import Path

import pytest
from scripts import build_arena_reviewer_runtime as composer

from scripts.build_arena_reviewer_runtime import (
    CompositionError, verify_image_configuration, verify_overlay,
)
ROOT = composer.ROOT


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


@pytest.mark.parametrize("negative_finality", [False, True])
@pytest.mark.parametrize("source_access", [False, True])
def test_pending_final_correction_is_explicit_and_changes_only_its_predicate(
    negative_finality, source_access,
):
    from scripts import build_arena_reviewer_runtime as builder
    from scripts.arena_reviewer_budget_seam import _OLD, _NEW, convert_budget_seam

    names = [builder.GRADER_FILE, *builder.ENTRYPOINT_FILES]
    if negative_finality or source_access:
        names.append(builder.FINALITY_FILE)
    original = {
        name: (builder.ROOT / ("lab_arena/scorer_entrypoint.py"
                              if name == "scorer_entrypoint.py" else name)).read_bytes()
        for name in names
    }
    # Reproduce the original frozen loop; do not import current grading behavior.
    assert original[builder.GRADER_FILE].count(_NEW.encode()) == 1
    original[builder.GRADER_FILE] = original[builder.GRADER_FILE].replace(
        _NEW.encode(), _OLD.encode(), 1)
    prior = builder.expected_overlay(
        original, complete_negative_finality=negative_finality,
        source_access_denied=source_access,
    )
    repaired = builder.expected_overlay(
        original, complete_negative_finality=negative_finality,
        source_access_denied=source_access, pending_final_correction=True,
    )
    assert prior[builder.GRADER_FILE].count(_OLD.encode()) == 1
    assert repaired[builder.GRADER_FILE] == convert_budget_seam(prior[builder.GRADER_FILE])
    assert set(repaired) == set(prior)
    for name in set(prior) - {builder.GRADER_FILE}:
        assert repaired[name] == prior[name]


def test_completed_investigation_correction_preserves_all_other_frozen_bytes():
    from scripts import build_arena_reviewer_runtime as builder
    from scripts.arena_reviewer_investigation_finality_seam import (
        DELTAS, HELPER, WRAPPER_GUARD_NEW, WRAPPER_GUARD_OLD,
        convert_investigation_finality_seam,
    )
    corrected = (builder.ROOT / builder.FINALITY_FILE).read_bytes()
    original = corrected.replace(HELPER.encode(), b'', 1).replace(
        WRAPPER_GUARD_NEW.encode(), WRAPPER_GUARD_OLD.encode(), 1,
    )
    for old, new in DELTAS:
        assert original.count(new.encode()) == 1
        original = original.replace(new.encode(), old.encode(), 1)
    assert convert_investigation_finality_seam(original) == corrected
    assert convert_investigation_finality_seam(corrected) == corrected
    sources = {
        name: (builder.ROOT / ('lab_arena/scorer_entrypoint.py'
                              if name == 'scorer_entrypoint.py' else name)).read_bytes()
        for name in (builder.GRADER_FILE, builder.FINALITY_FILE, *builder.ENTRYPOINT_FILES)
    }
    sources[builder.FINALITY_FILE] = original
    prior = builder.expected_overlay(sources, complete_negative_finality=True)
    repaired = builder.expected_overlay(
        sources, complete_negative_finality=True,
        completed_investigation_finality=True,
    )
    assert repaired[builder.FINALITY_FILE] == corrected
    for name in set(prior) - {builder.FINALITY_FILE}:
        assert repaired[name] == prior[name]
    # Without explicit correction selection the frozen grader stays byte exact.
    assert prior[builder.FINALITY_FILE] == original


def test_completed_investigation_upgrades_existing_runtime_finality_only():
    from scripts.arena_reviewer_investigation_finality_seam import (
        HELPER, LEGACY_HELPER, WRAPPER_GUARD_NEW, WRAPPER_GUARD_OLD,
        convert_investigation_finality_seam,
    )
    corrected = (ROOT / composer.FINALITY_FILE).read_bytes()
    legacy = corrected.replace(HELPER.encode(), LEGACY_HELPER.encode(), 1).replace(
        WRAPPER_GUARD_NEW.encode(), WRAPPER_GUARD_OLD.encode(), 1,
    )
    assert convert_investigation_finality_seam(legacy) == corrected
    before, after = ast.parse(legacy), ast.parse(corrected)
    for tree in (before, after):
        tree.body = [node for node in tree.body if getattr(node, 'name', '') not in {
            '_settle_completed_company_investigation', '_verify_company_fit',
        }]
    assert ast.dump(before) == ast.dump(after)


def test_actual_frozen_base_and_existing_runtime_compose_to_identical_bytes():
    from scripts.arena_reviewer_investigation_finality_seam import convert_investigation_finality_seam
    base_path = os.environ.get('ARENA_FROZEN_FINALITY_SOURCE')
    runtime_path = os.environ.get('ARENA_FROZEN_FINALITY_RUNTIME_SOURCE')
    if not base_path or not runtime_path:
        pytest.skip('actual frozen release sources are supplied by the release gate')
    base, runtime = Path(base_path).read_bytes(), Path(runtime_path).read_bytes()
    assert hashlib.sha256(base).hexdigest() == 'b573bd3fb3984cd065e48b37e19c8fdd59545319bef0f1801cdc94f1f502d3b2'
    assert hashlib.sha256(runtime).hexdigest() == 'b09dcafd2f4a83b2fcc31ed44c57a901f9e7f1a5b693d09c914def52bab0128b'
    corrected = convert_investigation_finality_seam(base)
    assert corrected == convert_investigation_finality_seam(runtime)
    assert convert_investigation_finality_seam(corrected) == corrected


@pytest.mark.parametrize('fault', ['legacy_helper_new_guard', 'new_helper_old_guard',
                                  'unknown_guard', 'shadowed_wrapper'])
def test_completed_investigation_wrapper_correction_fails_closed(fault):
    from scripts.arena_reviewer_investigation_finality_seam import (
        HELPER, LEGACY_HELPER, WRAPPER_GUARD_NEW, WRAPPER_GUARD_OLD,
        InvestigationFinalitySeamError, convert_investigation_finality_seam,
    )
    source = (ROOT / composer.FINALITY_FILE).read_bytes()
    if fault == 'legacy_helper_new_guard':
        source = source.replace(HELPER.encode(), LEGACY_HELPER.encode(), 1)
    elif fault == 'new_helper_old_guard':
        source = source.replace(WRAPPER_GUARD_NEW.encode(), WRAPPER_GUARD_OLD.encode(), 1)
    elif fault == 'unknown_guard':
        source = source.replace(WRAPPER_GUARD_NEW.encode(), b'            and False\n', 1)
    else:
        source += b'\n_verify_company_fit = lambda *args: None\n'
    with pytest.raises(InvestigationFinalitySeamError):
        convert_investigation_finality_seam(source)


@pytest.mark.parametrize('fault', ['partial', 'unknown', 'missing_helper', 'changed_helper'])
def test_completed_investigation_correction_fails_closed(fault):
    from scripts import build_arena_reviewer_runtime as builder
    from scripts.arena_reviewer_investigation_finality_seam import (
        DELTAS, HELPER, WRAPPER_GUARD_NEW, WRAPPER_GUARD_OLD,
        InvestigationFinalitySeamError,
        convert_investigation_finality_seam,
    )
    source = (builder.ROOT / builder.FINALITY_FILE).read_bytes()
    if fault == 'partial':
        old, new = DELTAS[-1]
        source = source.replace(new.encode(), old.encode(), 1)
    elif fault == 'unknown':
        source = source.replace(b'async def _llm_reverify_company(', b'async def _other_verifier(', 1)
    elif fault == 'missing_helper':
        source = source.replace(HELPER.encode(), b'', 1)
    else:
        source = source.replace(b'Coverage only selects', b'Changed only selects', 1)
    with pytest.raises(InvestigationFinalitySeamError):
        convert_investigation_finality_seam(source)


@pytest.mark.parametrize('binding', [
    '_settle_completed_company_investigation = lambda *args, **kwargs: None',
    '_settle_completed_company_investigation: object = None',
    'import another_module as _settle_completed_company_investigation',
    'from another_module import _settle_completed_company_investigation',
    'class _settle_completed_company_investigation: pass',
    'async def _settle_completed_company_investigation(*args): return None',
    'try:\n    pass\nexcept Exception as _settle_completed_company_investigation:\n    pass',
    'if True:\n    _settle_completed_company_investigation = None',
])
@pytest.mark.parametrize('already_corrected', [False, True])
def test_completed_investigation_correction_rejects_shadowed_helper_binding(
    binding, already_corrected,
):
    from scripts import build_arena_reviewer_runtime as builder
    from scripts.arena_reviewer_investigation_finality_seam import (
        DELTAS, HELPER, WRAPPER_GUARD_NEW, WRAPPER_GUARD_OLD,
        InvestigationFinalitySeamError,
        convert_investigation_finality_seam,
    )
    source = (builder.ROOT / builder.FINALITY_FILE).read_bytes()
    if not already_corrected:
        assert source.count(HELPER.encode()) == 1
        source = source.replace(HELPER.encode(), b'', 1)
        source = source.replace(WRAPPER_GUARD_NEW.encode(), WRAPPER_GUARD_OLD.encode(), 1)
        for old, new in DELTAS:
            assert source.count(new.encode()) == 1
            source = source.replace(new.encode(), old.encode(), 1)
    source += b'\n' + binding.encode() + b'\n'
    with pytest.raises(InvestigationFinalitySeamError):
        convert_investigation_finality_seam(source)
