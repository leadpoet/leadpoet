"""Observed-locator composition preserves frozen source and prior repairs."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import os
import subprocess

import pytest
from tests.reviewer_frozen_sources import frozen_source

from scripts import build_arena_reviewer_runtime as builder
from scripts import arena_reviewer_observed_linkedin_ampersand_seam as seam
from tests.lab_arena.reviewer_runtime_composition_test import layer


# Exact CI-green native-budget postimage; historical overlays must not track
# the evolving current investigator or require Git history in test installs.
_FROZEN_GRADER = frozen_source(
    "native_industry_budget", "qualification/scoring/company_evidence_investigator.py",
)


def sources():
    corrected = {path: (builder.ROOT / path).read_bytes() for path in seam.FUNCTIONS}
    original = dict(corrected)
    for path, (_name, old, new) in seam.FUNCTIONS.items():
        assert original[path].count(new.encode()) == 1
        original[path] = original[path].replace(new.encode(), old.encode(), 1)
        for _module, old_import, new_import in seam.IMPORTS[path]:
            assert original[path].count(new_import.encode()) == 1
            original[path] = original[path].replace(new_import.encode(), old_import.encode(), 1)
    original[seam.FIT] = original[seam.FIT].replace((seam.NEW_HELPER + '\n\n').encode(), b'', 1)
    return original, corrected


def reverse(files):
    reverted = dict(files)
    for path, (_name, old, new) in seam.FUNCTIONS.items():
        reverted[path] = reverted[path].replace(new.encode(), old.encode(), 1)
        for _module, old_import, new_import in seam.IMPORTS[path]:
            reverted[path] = reverted[path].replace(new_import.encode(), old_import.encode(), 1)
    reverted[seam.FIT] = reverted[seam.FIT].replace((seam.NEW_HELPER + '\n\n').encode(), b'', 1)
    return reverted


def test_only_exact_functions_helper_and_imports_change_and_are_idempotent():
    original, corrected = sources()
    saved = deepcopy(original)
    assert seam.convert_observed_linkedin_ampersand_files(original) == corrected
    assert seam.convert_observed_linkedin_ampersand_files(corrected) == corrected
    assert reverse(corrected) == original
    assert original == saved
    prefixed = {path: b'# frozen prefix\n' + source + b'\n# frozen suffix\n'
                for path, source in original.items()}
    assert seam.convert_observed_linkedin_ampersand_files(prefixed) == {
        path: b'# frozen prefix\n' + source + b'\n# frozen suffix\n'
        for path, source in corrected.items()}


@pytest.mark.parametrize('path', list(seam.FUNCTIONS))
@pytest.mark.parametrize('fault', ['utf8', 'syntax', 'crlf', 'missing', 'duplicate', 'shadow',
                                  'decorated', 'unknown_function', 'mixed_file', 'mixed_function',
                                  'changed_import', 'duplicate_import', 'dependency_shadow'])
def test_unknown_or_mixed_source_fails_closed(path, fault):
    original, corrected = sources()
    bad = dict(original)
    name, old, new = seam.FUNCTIONS[path]
    if fault == 'utf8':
        bad[path] = b'\xff'
    elif fault == 'syntax':
        bad[path] = b'def invalid(:'
    elif fault == 'crlf':
        bad[path] = bad[path].replace(b'\n', b'\r\n')
    elif fault == 'missing':
        bad[path] = bad[path].replace(('def ' + name + '(').encode(), b'def other(', 1)
    elif fault == 'duplicate':
        bad[path] += ('\ndef ' + name + '():\n    pass\n').encode()
    elif fault == 'shadow':
        bad[path] += ('\n' + name + ' = None\n').encode()
    elif fault == 'decorated':
        bad[path] = bad[path].replace(('def ' + name + '(').encode(),
                                    ('@decorator\ndef ' + name + '(').encode(), 1)
    elif fault == 'unknown_function':
        bad[path] = bad[path].replace(old.encode(), old.replace('return ""', 'return "unknown"', 1).encode(), 1)
    elif fault == 'mixed_file':
        bad[path] = corrected[path]
    elif fault == 'mixed_function':
        bad[path] = bad[path].replace(old.encode(), new.encode(), 1)
    elif fault == 'changed_import':
        _module, old_import, _new_import = seam.IMPORTS[path][0]
        bad[path] = bad[path].replace(old_import.encode(), old_import.replace('import ', 'import unknown, ', 1).encode(), 1)
    elif fault == 'duplicate_import':
        bad[path] += b'\n' + seam.IMPORTS[path][0][1].encode()
    else:
        bad[path] += ('\n' + seam.INTRODUCED_IMPORTS[path][0] + ' = None\n').encode()
    with pytest.raises(seam.ObservedLinkedinAmpersandSeamError):
        seam.convert_observed_linkedin_ampersand_files(bad)


@pytest.mark.parametrize('fault', ['missing', 'duplicate', 'decorated', 'changed', 'moved', 'partial_imports', 'unknown_file'])
def test_helper_and_file_set_are_exact(fault):
    original, corrected = sources()
    bad = dict(corrected)
    if fault == 'missing':
        bad[seam.FIT] = bad[seam.FIT].replace((seam.NEW_HELPER + '\n\n').encode(), b'', 1)
    elif fault == 'duplicate':
        bad[seam.FIT] += b'\n' + seam.NEW_HELPER.encode()
    elif fault == 'decorated':
        bad[seam.FIT] = bad[seam.FIT].replace(seam.NEW_HELPER.encode(), b'@decorator\n' + seam.NEW_HELPER.encode(), 1)
    elif fault == 'changed':
        bad[seam.FIT] = bad[seam.FIT].replace(b'return quote(slug, safe="._-")', b'return slug', 1)
    elif fault == 'moved':
        bad[seam.FIT] = bad[seam.FIT].replace((seam.NEW_HELPER + '\n\n').encode(), b'', 1) + b'\n' + seam.NEW_HELPER.encode()
    elif fault == 'partial_imports':
        bad[seam.SIZE] = bad[seam.SIZE].replace(seam.NEW_IMPORT_2.encode(), seam.OLD_IMPORT_2.encode(), 1)
    else:
        bad['unrelated.py'] = b'pass\n'
    with pytest.raises(seam.ObservedLinkedinAmpersandSeamError):
        seam.convert_observed_linkedin_ampersand_files(bad)


def originals(with_prior):
    files = {name: (builder.ROOT / ('lab_arena/scorer_entrypoint.py'
                                   if name == 'scorer_entrypoint.py' else name)).read_bytes()
             for name in (builder.GRADER_FILE, *builder.ENTRYPOINT_FILES)}
    files[builder.GRADER_FILE] = _FROZEN_GRADER
    if with_prior:
        # Prior repair flags apply to the exact historical body they reviewed.
        files[builder.FINALITY_FILE] = subprocess.check_output([
            "git", "show", "dbe8a014d2488a25f53649c4f61a48248147c54e:" + builder.FINALITY_FILE,
        ], cwd=builder.ROOT)
    return files


@pytest.mark.parametrize('prior', [False, True])
def test_explicit_option_preserves_prior_runtime_and_monzo_bytes(prior):
    flags = ({'source_blocked_alias': True, 'completed_investigation_finality': True,
              'auxiliary_source_name_binding': True, 'homepage_linkedin_conflict_repair': True}
             if prior else {})
    old, corrected = sources()
    original = originals(prior)
    before = builder.expected_overlay(original, **flags)
    after = builder.expected_overlay({**original, **old}, **flags, observed_linkedin_ampersand=True)
    assert all(after[name] == source for name, source in before.items())
    assert {name: after[name] for name in corrected} == corrected
    assert set(after) == set(before) | set(corrected)
    builder.verify_overlay(layer([('model/' + name, data) for name, data in after.items()]), after)
    altered = dict(after)
    altered[seam.SIZE] += b'\n# unrelated change\n'
    with pytest.raises(builder.CompositionError):
        builder.verify_overlay(layer([('model/' + name, data) for name, data in altered.items()]), after)


@pytest.mark.parametrize('verify_existing', [False, True])
def test_build_and_verify_use_same_explicit_option(monkeypatch, capsys, verify_existing):
    grading = 'registry.example/grader@sha256:' + 'a' * 64
    runtime = 'registry.example/grader@sha256:' + 'b' * 64
    args = ['compose', '--grading-image', grading, '--observed-linkedin-ampersand']
    if verify_existing:
        args += ['--verify-image', runtime]
    monkeypatch.setattr(builder.sys, 'argv', args)
    monkeypatch.setattr(builder, 'checked_output', lambda *args, **kwargs: 'c' * 40)
    base = {'Architecture': 'amd64', 'Os': 'linux', 'Config': {}, 'RootFS': {'Layers': ['base']}}
    candidate = deepcopy(base)
    candidate['RootFS']['Layers'].append('overlay')
    monkeypatch.setattr(builder, 'image_inspect', lambda name: base if name == grading else candidate)
    seen = []
    def extract(*args, **kwargs):
        seen.append(kwargs)
        return {**originals(False), **sources()[0]}
    monkeypatch.setattr(builder, 'extract_frozen_sources', extract)
    verified = []
    monkeypatch.setattr(builder, 'verify_saved_overlay', lambda image, digest, files: verified.append(files))
    commands = []
    monkeypatch.setattr(builder.subprocess, 'run', lambda command, **kwargs: commands.append(command))
    assert builder.main() == 0
    receipt = json.loads(capsys.readouterr().out)
    assert seen[0]['observed_linkedin_ampersand'] is True
    assert receipt['grading_corrections'] == [seam.CORRECTION_ID]
    assert receipt['base_layers_unchanged'] and receipt['process_configuration_unchanged']
    assert {name: verified[0][name] for name in seam.FUNCTIONS} == sources()[1]
    assert bool(commands) is not verify_existing


def test_actual_base_sources_and_monzo_overlay_when_supplied():
    directory = os.environ.get('ARENA_OBSERVED_LINKEDIN_IMAGE_INPUTS')
    receipt_path = os.environ.get('ARENA_OBSERVED_LINKEDIN_MONZO_RECEIPT')
    if not directory or not receipt_path:
        pytest.skip('direct frozen sources and published Monzo receipt not supplied')
    root = Path(directory)
    manifest = json.loads((root / 'base/manifest.json').read_text())
    assert manifest['reference'].endswith('@sha256:8d0f613c781c2126c86effa91ea0bc7f9d28857521216e6346d522f4c0345bb1')
    needed = (builder.GRADER_FILE, builder.FINALITY_FILE, *builder.ENTRYPOINT_FILES, *builder.BOUND_IDENTITY_FILES)
    files = {name: (root / 'base' / name).read_bytes() for name in needed}
    for name, data in files.items():
        assert hashlib.sha256(data).hexdigest() == manifest['files'][name]['sha256']
    flags = {'source_blocked_alias': True, 'completed_investigation_finality': True,
             'auxiliary_source_name_binding': True, 'homepage_linkedin_conflict_repair': True}
    prior = builder.expected_overlay({name: data for name, data in files.items()
                                      if name not in builder.BOUND_IDENTITY_FILES}, **flags)
    receipt = json.loads(Path(receipt_path).read_text())
    # The recorded Monzo image predates restoration of the original mismatch
    # guard. Reconstruct only that old block to verify its immutable receipt;
    # the candidate below must preserve the corrected prior overlay exactly.
    from scripts import arena_reviewer_homepage_linkedin_conflict_seam as identity_seam
    from tests.test_arena_reviewer_homepage_linkedin_conflict_seam import previous_broad_dispatch
    recorded = dict(prior)
    previous_dispatch = previous_broad_dispatch()
    assert previous_dispatch != identity_seam.NEW_DISPATCH
    assert recorded[builder.FINALITY_FILE].count(identity_seam.NEW_DISPATCH.encode()) == 1
    recorded[builder.FINALITY_FILE] = recorded[builder.FINALITY_FILE].replace(
        identity_seam.NEW_DISPATCH.encode(), previous_dispatch.encode(), 1,
    )
    assert {name: hashlib.sha256(data).hexdigest() for name, data in recorded.items()} == receipt['overlay_files']
    candidate = builder.expected_overlay(files, **flags, observed_linkedin_ampersand=True)
    assert all(candidate[name] == data for name, data in prior.items())
    corrected = {name: candidate[name] for name in seam.FUNCTIONS}
    assert reverse(corrected) == {name: files[name] for name in seam.FUNCTIONS}
    assert seam.convert_observed_linkedin_ampersand_files(corrected) == corrected
