"""Brand composition owns three proof spans and two private prompt copies."""

from copy import deepcopy
from functools import lru_cache
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest

from scripts import arena_reviewer_homepage_brand_alias_seam as seam
from scripts import build_arena_reviewer_runtime as builder
from tests.lab_arena.reviewer_runtime_composition_test import layer


# This exact-byte composer owns the reviewed October 28 source. Later
# production verifier changes must not silently become its frozen input.
FROZEN_COMMIT = "00725d7839a5b68b56c92774f415c173c3fd8d60"
_GIT_RUN = subprocess.run  # Builder tests replace subprocess.run for Docker only.
FROZEN_SHA256 = {
    seam.SCORER: "1daa87943bbd8aaa857cf177884899a4cad26d5c6220e7c2af1764bd23a2ea2d",
    seam.FIT: "5f8f990f7719b98f7fc84ecf7236992db46c97a099c64f2d01a13d200d6e17d0",
    seam.INVESTIGATOR: "22519946a9355e9374261d39fdc5099452d531c13e4d821f57eb349e513535c5",
    "lab_arena/scorer_entrypoint.py": "7f9f11d6cd86cdce932f06f9b32db4c64cfe95ab3fca161ea8d11c67349188f4",
}


@lru_cache(maxsize=1)
def _frozen_sources():
    corrected = {}
    for name, expected in FROZEN_SHA256.items():
        data = _GIT_RUN(
            ["git", "show", f"{FROZEN_COMMIT}:{name}"], cwd=builder.ROOT,
            check=True, stdout=subprocess.PIPE, timeout=10,
        ).stdout
        assert hashlib.sha256(data).hexdigest() == expected
        corrected[name] = data
    return corrected


def sources():
    corrected = {name: _frozen_sources()[name]
                 for name in (seam.SCORER, seam.FIT, seam.INVESTIGATOR)}
    # This overlay owns exact historical functions. Industry integration can
    # change today's investigator without changing the reviewed frozen image.
    # Exact corrected function from upstream197048255 (PR832).
    frozen = (builder.ROOT / 'tests/fixtures/arena_reviewer/frozen_brand_investigator.py.txt').read_bytes()
    assert hashlib.sha256(frozen).hexdigest() == (
        '4ee4e5c966a6f1815c860a3b43e7cb151712eadf7ac7912e90e271cf06c19005')
    text = corrected[seam.INVESTIGATOR].decode()
    node = next(n for n in ast.parse(text).body
                if isinstance(n, ast.AsyncFunctionDef) and n.name == 'investigate_company_evidence')
    lines = text.splitlines(keepends=True)
    corrected[seam.INVESTIGATOR] = (
        ''.join(lines[:node.lineno - 1]).encode() + frozen
        + ''.join(lines[node.end_lineno:]).encode())
    return reverse(corrected), corrected


def reverse(files):
    original = dict(files)
    for path, pairs in (
        (seam.SCORER, ((seam.OLD_PROJECTOR, seam.NEW_PROJECTOR), (seam.OLD_WEB, seam.NEW_WEB),
                      (seam.OLD_SCORER_PROMPT, seam.NEW_SCORER_PROMPT))),
        (seam.FIT, ((seam.OLD_HOST, seam.NEW_HOST),)),
        (seam.INVESTIGATOR, ((seam.OLD_INVESTIGATOR_PROMPT, seam.NEW_INVESTIGATOR_PROMPT),)),
    ):
        for old, new in pairs:
            assert original[path].count(new.encode()) == 1
            original[path] = original[path].replace(new.encode(), old.encode(), 1)
    return original


def originals(include_brand=True):
    names = (builder.GRADER_FILE, *builder.ENTRYPOINT_FILES)
    files = {name: _frozen_sources()["lab_arena/scorer_entrypoint.py"
                                   if name == "scorer_entrypoint.py" else name]
             for name in names}
    files[builder.GRADER_FILE] = sources()[1][seam.INVESTIGATOR]
    if include_brand:
        files.update(sources()[0])
    return files


def test_exact_five_owned_spans_only_and_idempotence():
    original, corrected = sources()
    assert seam.convert_homepage_brand_alias_files(original) == corrected
    assert seam.convert_homepage_brand_alias_files(corrected) == corrected
    wrapped = {name: b"# frozen prefix\n" + data + b"\n# frozen suffix\n"
               for name, data in original.items()}
    assert seam.convert_homepage_brand_alias_files(wrapped) == {
        name: b"# frozen prefix\n" + data + b"\n# frozen suffix\n"
        for name, data in corrected.items()
    }


def test_new_industry_investigator_is_not_mistaken_for_frozen_overlay_source():
    current = {name: (builder.ROOT / name).read_bytes()
               for name in (seam.SCORER, seam.FIT, seam.INVESTIGATOR)}
    with pytest.raises(seam.HomepageBrandAliasSeamError, match='unrecognized bytes'):
        seam.convert_homepage_brand_alias_files(current)


@pytest.mark.parametrize("function,path", [
    ("_verified_homepage_identity_anchor", seam.SCORER),
    ("_web_identity_receipt", seam.SCORER),
    ("company_quality_receipt_matches_claim", seam.FIT),
    ("_llm_reverify_company", seam.SCORER),
    ("investigate_company_evidence", seam.INVESTIGATOR),
])
@pytest.mark.parametrize("fault", ["missing", "duplicate", "shadow", "decorated", "signature"])
def test_ambiguous_owner_or_changed_signature_fails_closed(function, path, fault):
    original, _ = sources()
    declaration = ("def " + function + "(").encode()
    if fault == "missing":
        original[path] = original[path].replace(declaration, b"def unrelated(", 1)
    elif fault == "duplicate":
        original[path] += ("\ndef " + function + "():\n    pass\n").encode()
    elif fault == "shadow":
        original[path] += ("\n" + function + " = None\n").encode()
    elif fault == "decorated":
        original[path] = original[path].replace(declaration, b"@decorator\n" + declaration, 1)
    else:
        original[path] = original[path].replace(declaration, declaration + b"injected, ", 1)
    with pytest.raises(seam.HomepageBrandAliasSeamError):
        seam.convert_homepage_brand_alias_files(original)


@pytest.mark.parametrize("fault", [
    "utf8", "syntax", "crlf", "mixed_files", "mixed_projector", "mixed_web", "unknown_projector",
    "unknown_host", "rebound_dependency", "changed_helper", "shadowed_helper", "escaped_local",
    "global_local", "quoted_owners", "missing_file", "extra_file",
    "mixed_scorer_prompt", "mixed_investigator_prompt", "unknown_prompt_guard",
    "unknown_prompt_literal", "duplicate_prompt_key", "moved_prompt_copy",
])
def test_unknown_partial_or_mixed_correction_fails_closed(fault):
    original, corrected = sources()
    if fault == "utf8":
        original[seam.SCORER] = b"\xff"
    elif fault == "syntax":
        original[seam.SCORER] = b"def malformed(:"
    elif fault == "crlf":
        original[seam.SCORER] = original[seam.SCORER].replace(b"\n", b"\r\n")
    elif fault == "mixed_files":
        original[seam.FIT] = corrected[seam.FIT]
    elif fault in {"mixed_projector", "mixed_web"}:
        old, new = ((seam.OLD_PROJECTOR, seam.NEW_PROJECTOR) if fault == "mixed_projector"
                    else (seam.OLD_WEB, seam.NEW_WEB))
        original[seam.SCORER] = original[seam.SCORER].replace(old.encode(), new.encode(), 1)
    elif fault == "unknown_projector":
        original[seam.SCORER] = original[seam.SCORER].replace(
            seam.OLD_PROJECTOR.encode(), seam.OLD_PROJECTOR.replace("if brand:", "if brand is not None:").encode(), 1)
    elif fault == "unknown_host":
        original[seam.FIT] = original[seam.FIT].replace(
            seam.OLD_HOST.encode(), seam.OLD_HOST.replace("len(raw_aliases) > 3", "len(raw_aliases) > 4").encode(), 1)
    elif fault == "rebound_dependency":
        original[seam.SCORER] += b"\nverified_homepage_brand_name = None\n"
    elif fault == "changed_helper":
        original[seam.FIT] = original[seam.FIT].replace(
            seam.BRAND_HELPER.encode(), seam.BRAND_HELPER.replace("return brand.strip()", "return str(brand)").encode(), 1)
    elif fault == "shadowed_helper":
        original[seam.FIT] += b"\nverified_homepage_brand_name = None\n"
    elif fault in {"escaped_local", "global_local"}:
        insertion = ("    homepage_brand = None\n" if fault == "escaped_local"
                     else "    global homepage_brand\n")
        original[seam.SCORER] = original[seam.SCORER].replace(
            seam.OLD_WEB.encode(), (insertion + seam.OLD_WEB).encode(), 1)
    elif fault == "quoted_owners":
        original[seam.SCORER] = ("quoted = " + repr(original[seam.SCORER].decode()) + "\n").encode()
    elif fault == "missing_file":
        original.pop(seam.FIT)
    elif fault in {"mixed_scorer_prompt", "mixed_investigator_prompt"}:
        path, old, new = ((seam.SCORER, seam.OLD_SCORER_PROMPT, seam.NEW_SCORER_PROMPT)
                          if fault == "mixed_scorer_prompt" else
                          (seam.INVESTIGATOR, seam.OLD_INVESTIGATOR_PROMPT, seam.NEW_INVESTIGATOR_PROMPT))
        original[path] = original[path].replace(old.encode(), new.encode(), 1)
    elif fault == "unknown_prompt_guard":
        original[seam.SCORER] = original[seam.SCORER].replace(
            b"    if verified_identity:\n        verified_identity_context = (",
            b"    if verified_identity is not None:\n        verified_identity_context = (", 1)
    elif fault == "unknown_prompt_literal":
        original[seam.SCORER] = original[seam.SCORER].replace(
            b"Server-verified homepage identity anchor (lookup context only; ",
            b"Unreviewed identity context: ", 1)
    elif fault == "duplicate_prompt_key":
        original[seam.INVESTIGATOR] = original[seam.INVESTIGATOR].replace(
            seam.OLD_INVESTIGATOR_PROMPT.encode(), seam.OLD_INVESTIGATOR_PROMPT.encode() * 2, 1)
    elif fault == "moved_prompt_copy":
        original[seam.INVESTIGATOR] = original[seam.INVESTIGATOR].replace(
            b"    input_document = {\n", b"    unrelated_document = {\n", 1)
    else:
        original["unrelated.py"] = b"pass\n"
    with pytest.raises(seam.HomepageBrandAliasSeamError):
        seam.convert_homepage_brand_alias_files(original)


@pytest.mark.parametrize("path,needle,replacement", [
    (seam.SCORER, "if identity is None or identity.decision != COMPANY_FIT_MATCH:", "if identity is None:"),
    (seam.SCORER, "if any(not isinstance(value, str) for value in observed_values.values()):", "if False:"),
    (seam.FIT, "Validate that a quality receipt is canonical and binds this exact claim.", "Unreviewed host gate."),
])
def test_unreviewed_bytes_elsewhere_in_each_owned_function_fail_closed(path, needle, replacement):
    original, _ = sources()
    assert original[path].count(needle.encode()) == 1
    original[path] = original[path].replace(needle.encode(), replacement.encode(), 1)
    with pytest.raises(seam.HomepageBrandAliasSeamError, match="unrecognized bytes"):
        seam.convert_homepage_brand_alias_files(original)


def test_explicit_option_preserves_other_overlay_bytes_and_layer_contract():
    before = builder.expected_overlay(originals(False))
    after = builder.expected_overlay(originals(), verified_homepage_brand_alias=True)
    assert {name: after[name] for name in sources()[1]} == sources()[1]
    assert all(after[name] == data for name, data in before.items())
    assert set(after) == set(before) | set(builder.BRAND_IDENTITY_FILES)
    builder.verify_overlay(layer([("model/" + name, data) for name, data in after.items()]), after)
    altered = dict(after)
    altered[builder.GRADER_FILE] += b"\n# unrelated\n"
    with pytest.raises(builder.CompositionError):
        builder.verify_overlay(layer([("model/" + name, data) for name, data in altered.items()]), after)


@pytest.mark.parametrize("enabled", [False, True])
def test_extraction_requests_only_needed_frozen_files(monkeypatch, tmp_path, enabled):
    calls = []
    data = originals()
    monkeypatch.setattr(builder, "checked_output", lambda *args, **kwargs: "container")
    def docker(command, **kwargs):
        calls.append(command)
        if command[:2] == ["docker", "cp"]:
            Path(command[-1]).write_bytes(data[command[2].split(":/model/", 1)[1]])
    monkeypatch.setattr(builder.subprocess, "run", docker)
    result = builder.extract_frozen_sources("frozen:grader", tmp_path,
                                             verified_homepage_brand_alias=enabled)
    assert set(result) == set(originals(enabled))
    assert calls[-1] == ["docker", "rm", "container"]


@pytest.mark.parametrize("verify_existing", [False, True])
def test_builder_and_verifier_record_matching_host_requirement(monkeypatch, capsys, verify_existing):
    grading = "registry.example/model@sha256:" + "a" * 64
    runtime = "registry.example/model@sha256:" + "b" * 64
    args = ["compose", "--grading-image", grading, "--verified-homepage-brand-alias"]
    if verify_existing:
        args += ["--verify-image", runtime]
    monkeypatch.setattr(builder.sys, "argv", args)
    monkeypatch.setattr(builder, "checked_output", lambda *args, **kwargs: "c" * 40)
    base = {"Architecture": "amd64", "Os": "linux", "Config": {"User": "65534"}, "RootFS": {"Layers": ["base"]}}
    candidate = deepcopy(base)
    candidate["RootFS"]["Layers"].append("overlay")
    monkeypatch.setattr(builder, "image_inspect", lambda ref: base if ref == grading else candidate)
    extractions, verified, commands = [], [], []
    def extract(*args, **kwargs):
        extractions.append(kwargs)
        return originals()
    monkeypatch.setattr(builder, "extract_frozen_sources", extract)
    monkeypatch.setattr(builder, "verify_saved_overlay", lambda image, digest, files: verified.append(files))
    monkeypatch.setattr(builder.subprocess, "run", lambda command, **kwargs: commands.append(command))
    assert builder.main() == 0
    receipt = json.loads(capsys.readouterr().out)
    assert extractions[0]["verified_homepage_brand_alias"] is True
    assert receipt["grading_corrections"] == [seam.CORRECTION_ID]
    assert receipt["host_validator_requirement"] == seam.HOST_VALIDATOR_REQUIREMENT
    assert receipt["base_layers_unchanged"] and receipt["process_configuration_unchanged"]
    assert bool(commands) is not verify_existing
    candidate["Config"]["User"] = "0"
    with pytest.raises(builder.CompositionError):
        builder.verify_image_configuration(base, candidate)


@pytest.mark.parametrize("cohort,base_directory,receipt_name", [
    ("oct26-27", "monzo-identity/direct-image-composition-inputs/base", "final-670-verify-8d.json"),
    ("oct28", "oct28-exact-images/base", "final-670-verify-8761.json"),
])
def test_both_exact_frozen_bases_preserve_final670_and_all_other_bytes(cohort, base_directory, receipt_name):
    directory = os.environ.get("ARENA_HOMEPAGE_BRAND_ARTIFACT_ROOT")
    if not directory:
        pytest.skip("exact two-base image extraction and final670 receipts not supplied")
    root = Path(directory)
    base = root / base_directory
    manifest = json.loads((base / "manifest.json").read_text())
    receipt = json.loads((root / "final-670-release" / receipt_name).read_text())
    assert manifest["reference"] == receipt["grading_image"]
    needed = (builder.GRADER_FILE, builder.FINALITY_FILE, *builder.ENTRYPOINT_FILES, *builder.BOUND_IDENTITY_FILES)
    files = {name: (base / name).read_bytes() for name in needed}
    assert {name: hashlib.sha256(data).hexdigest() for name, data in files.items()} == receipt["original_files"]
    flags = {"source_blocked_alias": True, "completed_investigation_finality": True,
             "auxiliary_source_name_binding": True, "homepage_linkedin_conflict_repair": True,
             "observed_linkedin_ampersand": True}
    prior = builder.expected_overlay(files, **flags)
    assert {name: hashlib.sha256(data).hexdigest() for name, data in prior.items()} == receipt["overlay_files"]
    candidate = builder.expected_overlay(files, **flags, verified_homepage_brand_alias=True)
    assert set(candidate) == set(prior)
    assert {name for name in prior if prior[name] != candidate[name]} == {seam.SCORER, seam.FIT, seam.INVESTIGATOR}
    owned = {name: candidate[name] for name in (seam.SCORER, seam.FIT, seam.INVESTIGATOR)}
    assert reverse(owned) == {name: prior[name] for name in owned}
    for path, functions in (
        (seam.SCORER, ("_verified_homepage_identity_anchor", "_web_identity_receipt")),
        (seam.FIT, ("company_quality_receipt_matches_claim",)),
    ):
        current = (builder.ROOT / path).read_text()
        frozen = owned[path].decode()
        for function in functions:
            def function_source(source):
                node = next(node for node in ast.parse(source).body
                            if isinstance(node, ast.FunctionDef) and node.name == function)
                return ast.get_source_segment(source, node)
            assert function_source(frozen) == function_source(current)
    assert seam.convert_homepage_brand_alias_files(owned) == owned
    host_source = candidate[seam.FIT].decode()
    host = next(node for node in ast.parse(host_source).body
                if isinstance(node, ast.FunctionDef)
                and node.name == "company_quality_receipt_matches_claim")
    assert hashlib.sha256(ast.get_source_segment(host_source, host).encode()).hexdigest() == (
        seam.HOST_VALIDATOR_REQUIREMENT["function_source_sha256"]
    )
    builder.verify_overlay(layer([("model/" + name, data) for name, data in candidate.items()]), candidate)


@pytest.mark.parametrize("cohort,base_directory,receipt_name,receipt_hash", [
    ("8d", "monzo-identity/direct-image-composition-inputs/base", "build-8d.log",
     "8517ffa8dbccf2b26175e3a071e415a31bacf8b1c15542b7097c16f90090f516"),
    ("8761", "oct28-exact-images/base", "build-8761.log",
     "ec999fb5a88c1bdde07236d9a1f2261d05c79b9e45fcc9e1b84a326dc9f3610d"),
    ("8bd", None, "build-8bd-after-pull.log",
     "a4508762ad0f99e780df43e08239d117623e91ed4f0f8d3e1bb1d44d78432b7c"),
])
def test_all_three_published_brand_overlays_change_only_prompt_copy_filters(
    cohort, base_directory, receipt_name, receipt_hash,
):
    directory = os.environ.get("ARENA_HOMEPAGE_BRAND_ARTIFACT_ROOT")
    if not directory:
        pytest.skip("hash-bound published brand build receipts not supplied")
    root = Path(directory)
    raw = (root / "verified-homepage-brand/remote-image-proof" / receipt_name).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == receipt_hash
    receipts = [json.loads(line) for line in raw.splitlines() if line.startswith(b"{")]
    assert len(receipts) == 1
    receipt = receipts[0]
    original = {}
    for name, expected_hash in receipt["original_files"].items():
        if base_directory:
            data = (root / base_directory / name).read_bytes()
        else:
            source_name = "lab_arena/scorer_entrypoint.py" if name == "scorer_entrypoint.py" else name
            data = subprocess.run(
                ["git", "show", "c249928613cfe7df8db90f82c0b824ffacdf4dd6:" + source_name],
                cwd=builder.ROOT, check=True, stdout=subprocess.PIPE, timeout=10,
            ).stdout
        assert hashlib.sha256(data).hexdigest() == expected_hash
        original[name] = data
    flags = {"verified_homepage_brand_alias": True}
    if cohort != "8bd":
        flags.update(source_blocked_alias=True, completed_investigation_finality=True,
                     auxiliary_source_name_binding=True, homepage_linkedin_conflict_repair=True,
                     observed_linkedin_ampersand=True)
    candidate = builder.expected_overlay(original, **flags)
    inverse = dict(candidate)
    for path, old, new in (
        (seam.SCORER, seam.OLD_SCORER_PROMPT, seam.NEW_SCORER_PROMPT),
        (seam.INVESTIGATOR, seam.OLD_INVESTIGATOR_PROMPT, seam.NEW_INVESTIGATOR_PROMPT),
    ):
        assert inverse[path].count(new.encode()) == 1
        inverse[path] = inverse[path].replace(new.encode(), old.encode(), 1)
    assert {name: hashlib.sha256(data).hexdigest() for name, data in inverse.items()} == receipt["overlay_files"]
    assert {name for name in candidate if candidate[name] != inverse[name]} == {seam.SCORER, seam.INVESTIGATOR}
    assert seam.convert_homepage_brand_alias_files({
        name: candidate[name] for name in (seam.SCORER, seam.FIT, seam.INVESTIGATOR)
    }) == {name: candidate[name] for name in (seam.SCORER, seam.FIT, seam.INVESTIGATOR)}
