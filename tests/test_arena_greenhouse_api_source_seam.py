"""Frozen runtime composition changes source transport, never grading rules."""
from copy import deepcopy
import hashlib
import json

import pytest

from scripts import arena_reviewer_greenhouse_api_source_seam as seam
from scripts import build_arena_reviewer_runtime as builder
from tests.reviewer_frozen_sources import frozen_source


def original_source():
    return frozen_source("archive_https", builder.ARCHIVE_FILE)


def sources():
    result = {name: (builder.ROOT / ("lab_arena/scorer_entrypoint.py"
                                 if name == "scorer_entrypoint.py" else name)).read_bytes()
            for name in (builder.GRADER_FILE, *builder.ENTRYPOINT_FILES)}
    result[builder.GRADER_FILE] = frozen_source("native_industry_budget", builder.GRADER_FILE)
    return result


def test_exact_frozen_repair_is_reversible_and_idempotent():
    original = original_source()
    repaired = seam.convert_greenhouse_api_source_seam(original)
    assert seam.convert_greenhouse_api_source_seam(repaired) == repaired
    restored = repaired
    for patch in seam.PATCHES.values():
        assert restored.count(patch["replacement"].encode()) == 1
        restored = restored.replace(patch["replacement"].encode(), patch["anchor"].encode(), 1)
    assert restored == original


@pytest.mark.parametrize("name", list(seam.PATCHES))
@pytest.mark.parametrize("fault", ["changed", "duplicate", "shadowed", "decorated"])
def test_each_complete_helper_rejects_unknown_or_ambiguous_code(name, fault):
    original = original_source()
    if fault == "changed":
        original = original.replace(("def " + name + "(").encode(),
                                    ("def " + name + "(unknown_parameter, ").encode(), 1)
    elif fault == "duplicate":
        original += ("\ndef " + name + "(*args):\n    return None\n").encode()
    elif fault == "shadowed":
        original += ("\n" + name + " = lambda *args: None\n").encode()
    else:
        needle = ("async def " if name == "_scrape_exa" else "def ") + name
        original = original.replace(needle.encode(), ("@unreviewed\n" + needle).encode(), 1)
    with pytest.raises(seam.GreenhouseApiSourceSeamError):
        seam.convert_greenhouse_api_source_seam(original)


def test_option_is_explicit_and_all_other_overlay_files_are_unchanged():
    original = sources()
    prior = builder.expected_overlay(original)
    assert builder.ARCHIVE_FILE not in prior
    with pytest.raises(builder.CompositionError):
        builder.expected_overlay(original, greenhouse_api_source=True)
    original[builder.ARCHIVE_FILE] = original_source()
    repaired = builder.expected_overlay(original, greenhouse_api_source=True)
    assert set(repaired) == set(prior) | {builder.ARCHIVE_FILE}
    assert repaired[builder.ARCHIVE_FILE] == seam.convert_greenhouse_api_source_seam(original_source())
    assert all(repaired[name] == value for name, value in prior.items())


def test_cli_option_extracts_verifies_and_reports_only_exact_overlay(monkeypatch, capsys):
    original = sources()
    original[builder.ARCHIVE_FILE] = original_source()
    base = {"Architecture": "amd64", "Os": "linux", "Config": {},
            "RootFS": {"Layers": ["sha256:base"]}}
    candidate = deepcopy(base)
    candidate["RootFS"]["Layers"].append("sha256:overlay")
    grading = "registry.example/model@sha256:" + "a" * 64
    seen = {}
    monkeypatch.setattr("sys.argv", ["build", "--grading-image", grading,
                                    "--verify-image", "reviewed:runtime", "--greenhouse-api-source"])
    monkeypatch.setattr(builder, "checked_output", lambda *a, **k: "b" * 40)
    monkeypatch.setattr(builder, "image_inspect", lambda ref: base if ref == grading else candidate)
    monkeypatch.setattr(builder, "extract_frozen_sources", lambda *a, **k: seen.update(k) or original)
    monkeypatch.setattr(builder, "verify_saved_overlay", lambda *a: seen.update(overlay=a[-1]))
    assert builder.main() == 0
    receipt = json.loads(capsys.readouterr().out)
    assert seen["greenhouse_api_source"] is True
    assert receipt["grading_corrections"] == [seam.CORRECTION_ID]
    assert receipt["base_layers_unchanged"] and receipt["process_configuration_unchanged"]
    assert receipt["original_files"][builder.ARCHIVE_FILE] == hashlib.sha256(original_source()).hexdigest()
    assert seen["overlay"][builder.ARCHIVE_FILE] == seam.convert_greenhouse_api_source_seam(original_source())
