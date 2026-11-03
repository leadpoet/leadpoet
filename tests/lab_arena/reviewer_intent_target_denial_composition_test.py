"""Compose exactly two frozen allowlist insertions, without current runtime IO."""
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest

from scripts import build_arena_reviewer_runtime as builder
from scripts import arena_reviewer_intent_target_denial_seam as seam
from tests.lab_arena.reviewer_runtime_composition_test import layer


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "reviewer_seams"


def frozen_sources():
    sources = {name: gzip.decompress((FIXTURES / (
        "intent-target-denial-" + Path(name).name + ".gz"
    )).read_bytes()) for name in seam.FILES}
    assert {name: hashlib.sha256(source).hexdigest() for name, source in sources.items()} == seam.ORIGINAL_SHA256
    return sources


def test_exact_frozen_pair_changes_only_two_tuple_lines_and_is_idempotent():
    sources = frozen_sources()
    repaired = seam.convert_intent_target_denial_files(sources)
    assert {name: hashlib.sha256(source).hexdigest() for name, source in repaired.items()} == seam.REPAIRED_SHA256
    for name in sources:
        assert repaired[name].replace(seam.INSERTION, b"", 1) == sources[name]
        assert repaired[name].count(seam.INSERTION) == 1
    assert seam.convert_intent_target_denial_files(repaired) == repaired
    assert frozen_sources() == sources


def test_composed_frozen_modules_pass_saved_native_fetch_to_terminal_finality(monkeypatch):
    import qualification.scoring
    from tests import test_intent_native_target_denial_finality as regression

    repaired = seam.convert_intent_target_denial_files(frozen_sources())
    modules = {}
    for name, source in repaired.items():
        module = ModuleType("reviewed_frozen_" + Path(name).stem)
        monkeypatch.setitem(sys.modules, module.__name__, module)
        exec(compile(source, name, "exec"), module.__dict__)
        modules[name] = module
    monkeypatch.setattr(regression, "intent", modules[seam.VERIFIER_FILE])
    monkeypatch.setattr(regression, "competition", modules[seam.COMPETITION_FILE])
    monkeypatch.setitem(sys.modules, "qualification.scoring.competition", modules[seam.COMPETITION_FILE])
    monkeypatch.setattr(qualification.scoring, "competition", modules[seam.COMPETITION_FILE])
    regression.test_saved_native403_reaches_terminal_zero_without_rejudging_completed_siblings(monkeypatch)


def test_matching_host_accepts_retained_native_trace_but_exact_old_host_rejects(monkeypatch):
    import qualification.scoring
    from lab_arena import company_judgments, scoring
    from tests import test_intent_native_target_denial_finality as regression
    from tests.test_same_event_source_local_finality import _propagate_to_breakdown

    originals = frozen_sources()
    repaired = seam.convert_intent_target_denial_files(originals)

    def load(name, source):
        module = ModuleType(name)
        monkeypatch.setitem(sys.modules, name, module)
        exec(compile(source, name, "exec"), module.__dict__)
        return module

    verifier = load("native_denial_runtime_verifier", repaired[seam.VERIFIER_FILE])
    candidate = load("native_denial_matching_host", repaired[seam.COMPETITION_FILE])
    original_host = load("native_denial_original_host", originals[seam.COMPETITION_FILE])
    monkeypatch.setattr(regression, "intent", verifier)
    result, _ = regression.run_native_verifier(monkeypatch, [(200, regression.native_denial())] * 2)
    row = _propagate_to_breakdown(monkeypatch, result)
    row["verifier_gate_receipts"] = [{"gate": "company_fit", "decision": "match"}]
    terminal = candidate.terminal_company_verification_breakdown(row)
    raw = {**candidate.raw_company_judgment(terminal), "company_judgment_policy": "company_judgment_v1"}
    assert raw["intent_signals_detail"][0]["judge_verdict"]["verification_trace"]["provider_attempts"][0][
        "exa_target_crawl_failure"
    ]["observations"] == [{"error_tag": "CRAWL_HTTP_403", "error_http_status": 403}] * 2
    miss = {"company_index": 0, "cache_key": "sha256:" + "a" * 64,
            "company_input_hash": "sha256:" + "b" * 64, "authority_slot": 0}
    new_judgment = {**miss, "raw_judgment": raw}
    output = scoring.build_scoring_output("saved-native-denial-source", [terminal],
                                         company_judgments=[new_judgment])
    output = scoring.validate_scoring_output_document(output)
    lease = {"schema_version": company_judgments.LEASE_SCHEMA_VERSION,
             "hits": [], "misses": [miss]}

    def select_host(module):
        monkeypatch.setitem(sys.modules, "qualification.scoring.competition", module)
        monkeypatch.setattr(qualification.scoring, "competition", module)

    select_host(candidate)
    assert company_judgments.raw_judgment_is_cacheable(raw)
    assert company_judgments.validate_new_company_judgments(
        output["company_judgments"], lease_context=lease,
    ) == [new_judgment]
    select_host(original_host)
    assert not company_judgments.raw_judgment_is_cacheable(raw)
    with pytest.raises(company_judgments.CompanyJudgmentError, match="cacheable miss"):
        company_judgments.validate_new_company_judgments(
            output["company_judgments"], lease_context=lease,
        )


@pytest.mark.parametrize("fault", [
    "missing", "extra", "partial", "unknown", "shadowed", "outer_error_guard",
    "newlines", "invalid", "duplicate_alias",
])
@pytest.mark.parametrize("corrected", [False, True])
def test_unknown_or_partial_source_generation_fails_closed(fault, corrected):
    sources = frozen_sources()
    if corrected:
        sources = seam.convert_intent_target_denial_files(sources)
    name = seam.VERIFIER_FILE
    if fault == "missing":
        sources.pop(name)
    elif fault == "extra":
        sources["lab_arena/scoring.py"] = b"unapproved"
    elif fault == "partial":
        sources[name] = (sources[name].replace(seam.INSERTION, b"", 1) if corrected
                         else seam.convert_intent_target_denial_files(sources)[name])
    elif fault == "unknown":
        sources[name] = sources[name].replace(b'("SOURCE_NOT_AVAILABLE", 403)', b'("SOURCE_NOT_AVAILABLE", 401)', 1)
    elif fault == "shadowed":
        sources[name] += b"\n_EXA_TARGET_CRAWL_FAILURES = set()\n"
    elif fault == "outer_error_guard":
        sources[name] = sources[name].replace(b"elif envelope_error:", b"elif False:", 1)
    elif fault == "newlines":
        sources[name] = sources[name].replace(b"\n", b"\r\n")
    elif fault == "invalid":
        sources[name] = b"\xff"
    else:
        sources[name] = sources[name].replace(seam.ANCHOR, seam.ANCHOR + seam.INSERTION, 1)
    with pytest.raises(seam.IntentTargetDenialSeamError):
        seam.convert_intent_target_denial_files(sources)


@pytest.mark.parametrize("flag", [
    "--shim-only", "--complete-negative-finality", "--source-access-denied",
    "--source-blocked-alias", "--pending-final-correction", "--failed-attribute-source-recovery",
    "--completed-investigation-finality", "--identity-schema-repair", "--authenticated-page-refusal",
    "--single-tool-protocol-repair", "--rejected-final-packet", "--structured-homepage-numeric-alias",
    "--bound-homepage-legal-alias", "--numeric-profile-legal-self-name", "--archive-snapshot-https",
    "--greenhouse-api-source", "--auxiliary-source-name-binding", "--homepage-linkedin-conflict-repair",
    "--observed-linkedin-ampersand", "--verified-homepage-brand-alias",
    "--homepage-profile-alias",
    "--verified-display-brand",
])
def test_mode_rejects_other_corrections_before_image_access(monkeypatch, flag):
    monkeypatch.setattr(builder.sys, "argv", ["compose", "--grading-image",
        "registry.example/reviewer@sha256:" + "a" * 64, "--intent-target-denial", flag])
    monkeypatch.setattr(builder, "image_inspect", lambda *_: pytest.fail("image access"))
    with pytest.raises(builder.CompositionError, match="cannot include"):
        builder.main()


def test_extraction_reads_only_two_frozen_modules(monkeypatch, tmp_path):
    monkeypatch.setattr(builder, "checked_output", lambda *_: "container")
    sources = frozen_sources()
    commands = []

    def run(command, **_kwargs):
        commands.append(command)
        if command[1] == "cp":
            name = command[2].removeprefix("container:/model/")
            Path(command[3]).write_bytes(sources[name])

    monkeypatch.setattr(builder.subprocess, "run", run)
    assert builder.extract_frozen_sources("pinned-image", tmp_path, intent_target_denial=True) == sources
    assert [command[2] for command in commands if command[1] == "cp"] == [
        "container:/model/" + name for name in seam.FILES
    ]
    assert commands[-1] == ["docker", "rm", "container"]


@pytest.mark.parametrize("verify", [False, True])
def test_cli_build_and_verification_use_only_the_frozen_pair(monkeypatch, capsys, verify):
    grading = "registry.example/reviewer@sha256:" + "a" * 64
    runtime = "registry.example/reviewer@sha256:" + "b" * 64
    argv = ["compose", "--grading-image", grading, "--intent-target-denial"]
    if verify:
        argv += ["--verify-image", runtime]
    monkeypatch.setattr(builder.sys, "argv", argv)
    monkeypatch.setattr(builder, "_managed_docker_host", lambda: False)
    monkeypatch.setattr(builder, "checked_output", lambda *a, **k: "c" * 40)
    base = {"Architecture": "amd64", "Os": "linux", "Config": {"User": "65534:65534"},
            "RootFS": {"Layers": ["base"]}}
    candidate = deepcopy(base)
    candidate["RootFS"]["Layers"].append("repair-layer")
    monkeypatch.setattr(builder, "image_inspect", lambda name: base if name == grading else candidate)
    extracted, builds, verified = [], [], []

    def extract(reference, destination, **kwargs):
        extracted.append((reference, kwargs))
        return frozen_sources()

    def check(reference, digest, files):
        verified.append((reference, digest, files))
        assert set(files) == set(seam.FILES)
        builder.verify_overlay(layer([("model/" + name, data) for name, data in files.items()]), files)
        with pytest.raises(builder.CompositionError):
            builder.verify_overlay(layer([*(('model/' + n, d) for n, d in files.items()),
                                          ("model/lab_arena/scoring.py", b"unapproved")]), files)

    monkeypatch.setattr(builder, "extract_frozen_sources", extract)
    monkeypatch.setattr(builder, "expected_overlay", lambda *a, **k: pytest.fail("current runtime/grader overlay used"))
    monkeypatch.setattr(builder, "verify_saved_overlay", check)
    monkeypatch.setattr(builder.subprocess, "run", lambda command, **kwargs: builds.append(command))
    assert builder.main() == 0
    evidence = json.loads(capsys.readouterr().out)
    assert extracted == [(grading, {"intent_target_denial": True})]
    assert len(verified) == 1
    assert evidence["grading_image"] == grading
    assert evidence["grading_corrections"] == [seam.CORRECTION_ID]
    assert evidence["original_files"] == seam.ORIGINAL_SHA256
    assert evidence["overlay_files"] == seam.REPAIRED_SHA256
    assert evidence["base_layers_unchanged"] and evidence["process_configuration_unchanged"]
    assert len(builds) == (0 if verify else 1)
    if verify:
        assert evidence["runtime_image"] == runtime
    else:
        assert "-intent-target-denial" in evidence["runtime_image"]
