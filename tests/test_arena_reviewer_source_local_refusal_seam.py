"""Frozen810 source proof is an explicit bounded addition to merger-only mode."""
import ast
import copy
import hashlib
import json
from types import ModuleType

import pytest

from scripts import build_arena_reviewer_runtime as composer
from scripts import arena_reviewer_completed_merger_seam as seam
from scripts import arena_reviewer_source_local_refusal_seam as source_seam
from tests.reviewer_frozen_sources import frozen_source

ORIGINAL = {name: frozen_source("completed_merger810", name)
            for name in (seam.FINALITY_FILE, *source_seam.ORIGINAL_SHA256)}
CORRECTED = {seam.FINALITY_FILE: seam.convert_completed_merger_seam(ORIGINAL[seam.FINALITY_FILE]),
             **{name: source_seam.convert_source_local_refusal_seam(name, ORIGINAL[name])
                for name in source_seam.ORIGINAL_SHA256}}

@pytest.mark.parametrize("verify_existing", [False, True])
def test_source_local_composer_build_and_verification_change_only_three_files(monkeypatch, capsys, verify_existing):
    base_ref = "registry.example/reviewer@" + seam.FROZEN_IMAGE_DIGEST
    tag = "registry.example/reviewer:reviewer-" + seam.FROZEN_IMAGE_DIGEST[7:19] + "-" + "c" * 12 + "-completed-merger-survivor-source-local-refusal"
    candidate_ref = "registry.example/reviewer@sha256:" + "b" * 64 if verify_existing else tag
    args = ["build_arena_reviewer_runtime.py", "--grading-image", base_ref,
            "--completed-merger-survivor-only", "--source-local-refusal-finality"]
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
        assert flags == {"completed_merger_survivor_only": True, "source_local_refusal_finality": True}
        return ORIGINAL

    def verify(ref, digest, files):
        assert ref == candidate_ref and digest == "sha256:overlay"
        assert files == CORRECTED

    monkeypatch.setattr(composer, "extract_frozen_sources", extract)
    monkeypatch.setattr(composer, "verify_saved_overlay", verify)
    builds = []
    monkeypatch.setattr(composer.subprocess, "run", lambda command, **_k: builds.append(command))
    assert composer.main() == 0
    proof = json.loads(capsys.readouterr().out)
    assert proof["grading_image"] == base_ref
    assert proof["grading_corrections"] == [seam.CORRECTION_ID, source_seam.CORRECTION_ID]
    assert proof["original_files"] == {name: hashlib.sha256(value).hexdigest() for name, value in ORIGINAL.items()}
    assert proof["overlay_files"] == {name: hashlib.sha256(value).hexdigest() for name, value in CORRECTED.items()}
    assert proof["base_layers_unchanged"] is proof["process_configuration_unchanged"] is True
    assert len(builds) == (0 if verify_existing else 1)



@pytest.mark.parametrize('name', list(source_seam.ORIGINAL_SHA256))
def test_only_two_reader_functions_and_marker_constants_change(name):
    assert source_seam.convert_source_local_refusal_seam(name, CORRECTED[name]) == CORRECTED[name]
    for invalid in [b'', ORIGINAL[name]+b'\n', CORRECTED[name]+b'\n']:
        with pytest.raises(source_seam.SourceLocalRefusalSeamError):
            source_seam.convert_source_local_refusal_seam(name, invalid)
    allowed = {'_fetch_bounded_html', '_parse_worker_response_with_url',
        '_ARENA_SOURCE_UNAVAILABLE_HEADER', '_ARENA_SOURCE_UNAVAILABLE_VALUE', '_ARENA_SOURCE_UNAVAILABLE_REASON',
        'SOURCE_UNAVAILABLE_HEADER', 'SOURCE_UNAVAILABLE_VALUE'}
    def unchanged(source):
        return [ast.dump(n) for n in ast.parse(source).body if getattr(n, 'name', '') not in allowed
            and not (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in allowed for t in n.targets))]
    assert unchanged(ORIGINAL[name]) == unchanged(CORRECTED[name])


def test_new_option_requires_exclusive_mode_before_docker(monkeypatch):
    monkeypatch.setattr(composer.sys, 'argv', ['builder', '--grading-image',
        'registry.example/base@'+seam.FROZEN_IMAGE_DIGEST, '--source-local-refusal-finality'])
    with pytest.raises(composer.CompositionError, match='requires completed-merger'):
        composer.main()


def test_actual_frozen_shim_preserves_only_the_host_error_proof():
    import base64
    module = ModuleType('_actual_frozen810_source_proof_shim')
    exec(compile(CORRECTED['lab_arena/shim.py'], 'frozen810_shim.py', 'exec'), vars(module))
    payload = dict(status=403, headers={module.CALL_IDENTITY_HEADER:'sha256:'+'a'*64,
        module.SOURCE_UNAVAILABLE_HEADER:module.SOURCE_UNAVAILABLE_VALUE},
        body_b64=base64.b64encode(b'{"error":{"code":"provider_request_refused"}}').decode())
    assert module._parse_worker_response_with_url(payload)[1][module.SOURCE_UNAVAILABLE_HEADER] == module.SOURCE_UNAVAILABLE_VALUE
    payload['headers'].pop(module.CALL_IDENTITY_HEADER)
    assert module.SOURCE_UNAVAILABLE_HEADER not in module._parse_worker_response_with_url(payload)[1]
