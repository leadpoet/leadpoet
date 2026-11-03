"""The frozen-image option changes exactly two reviewed identity blocks."""
from copy import deepcopy
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest
from tests.reviewer_frozen_sources import frozen_source

from scripts import build_arena_reviewer_runtime as builder
from scripts import arena_reviewer_homepage_linkedin_conflict_seam as seam
from tests.lab_arena.reviewer_runtime_composition_test import layer


# Exact CI-green native-budget postimage; historical overlays must not track
# the evolving current investigator or require Git history in test installs.
_FROZEN_GRADER = frozen_source(
    "native_industry_budget", "qualification/scoring/company_evidence_investigator.py",
)


def previous_broad_dispatch():
    source = subprocess.check_output([
        "git", "show", "b1255dcc9aa7b66411ef0d76c0a8473051e67442:scripts/arena_reviewer_homepage_linkedin_conflict_seam.py",
    ], cwd=builder.ROOT)
    bindings = [node.value for node in ast.parse(source).body
                if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "NEW_DISPATCH"
                        for target in node.targets)]
    assert len(bindings) == 1
    return ast.literal_eval(bindings[0])


def source_pair(corrected=None):
    if corrected is None:
        corrected = (builder.ROOT / builder.FINALITY_FILE).read_bytes()
    original = corrected
    for old, new in ((seam.OLD_DISPATCH, seam.NEW_DISPATCH), (seam.OLD_PROMPT, seam.NEW_PROMPT)):
        assert original.count(new.encode()) == 1
        original = original.replace(new.encode(), old.encode(), 1)
    return original, corrected


def originals(include_finality=True, finality_source=None):
    names = [builder.GRADER_FILE, *builder.ENTRYPOINT_FILES]
    files = {name: (builder.ROOT / ('lab_arena/scorer_entrypoint.py'
                                   if name == 'scorer_entrypoint.py' else name)).read_bytes()
             for name in names}
    files[builder.GRADER_FILE] = _FROZEN_GRADER
    if include_finality:
        files[builder.FINALITY_FILE] = source_pair(finality_source)[0]
    return files


def test_two_exact_blocks_are_the_only_change_and_are_idempotent():
    original, corrected = source_pair()
    assert original != corrected
    assert seam.convert_homepage_linkedin_conflict_seam(original) == corrected
    assert seam.convert_homepage_linkedin_conflict_seam(corrected) == corrected
    wrapped = b'# frozen prefix\n' + original + b'\n# frozen suffix\n'
    assert seam.convert_homepage_linkedin_conflict_seam(wrapped) == (
        b'# frozen prefix\n' + corrected + b'\n# frozen suffix\n')


@pytest.mark.parametrize('fault', [
    'utf8', 'syntax', 'crlf', 'missing_web', 'missing_verifier',
    'duplicate_web', 'duplicate_verifier', 'shadow_web', 'shadow_verifier',
    'decorated_web', 'decorated_verifier', 'mixed_dispatch', 'mixed_prompt',
    'weakened_match_preservation', 'previous_broad_dispatch', 'changed_domain_binding', 'prompt_binding',
    'duplicate_prompt', 'changed_prompt_guard', 'quoted_functions',
])
def test_malformed_ambiguous_or_mixed_source_fails_closed(fault):
    original, corrected = source_pair()
    bad = original
    if fault == 'utf8':
        bad = b'\xff'
    elif fault == 'syntax':
        bad = b'def _web_identity_receipt(:'
    elif fault == 'crlf':
        bad = original.replace(b'\n', b'\r\n')
    elif fault.startswith(('missing_', 'duplicate_', 'shadow_', 'decorated_')):
        name = '_web_identity_receipt' if fault.endswith('_web') else '_llm_reverify_company'
        prefix = 'def ' if fault.endswith('_web') else 'async def '
        declaration = (prefix + name + '(').encode()
        if fault.startswith('missing_'):
            bad = original.replace(declaration, (prefix + 'other(').encode(), 1)
        elif fault.startswith('duplicate_'):
            bad = original + ('\n' + prefix + name + '():\n    pass\n').encode()
        elif fault.startswith('shadow_'):
            bad = original + ('\n' + name + ' = None\n').encode()
        else:
            bad = original.replace(declaration, b'@decorator\n' + declaration, 1)
    elif fault == 'mixed_dispatch':
        bad = original.replace(seam.OLD_DISPATCH.encode(), seam.NEW_DISPATCH.encode(), 1)
    elif fault == 'mixed_prompt':
        bad = original.replace(seam.OLD_PROMPT.encode(), seam.NEW_PROMPT.encode(), 1)
    elif fault == 'weakened_match_preservation':
        bad = corrected.replace(seam.NEW_DISPATCH.encode(),
            seam.NEW_DISPATCH.replace(
                'receipt.get("decision") == COMPANY_FIT_MISMATCH',
                'receipt.get("decision") in {COMPANY_FIT_MATCH, COMPANY_FIT_UNAVAILABLE}',
            ).encode(), 1)
        assert bad != corrected
    elif fault == 'previous_broad_dispatch':
        bad = corrected.replace(seam.NEW_DISPATCH.encode(),
                                previous_broad_dispatch().encode(), 1)
        assert bad != corrected
    elif fault == 'changed_domain_binding':
        bad = original.replace(seam.OLD_DISPATCH.encode(),
                               seam.OLD_DISPATCH.replace('and receipt.get("observed_domain")',
                                                         'and receipt.get("submitted_domain")').encode(), 1)
    elif fault == 'prompt_binding':
        bad = original.replace(seam.OLD_PROMPT.encode(),
                               seam.OLD_PROMPT.replace('identity_conflict_repair =', 'wrong =').encode(), 1)
    elif fault == 'duplicate_prompt':
        bad = original.replace(seam.OLD_PROMPT.encode(), (seam.OLD_PROMPT * 2).encode(), 1)
    elif fault == 'changed_prompt_guard':
        bad = original.replace(b'if identity_conflict_reason == "web_linkedin_conflicts_with_verified_homepage":',
                               b'if identity_conflict_reason != "web_linkedin_conflicts_with_verified_homepage":', 1)
    else:
        bad = ('quoted = ' + repr(original.decode()) + '\n').encode()
    with pytest.raises(seam.HomepageLinkedinConflictSeamError):
        seam.convert_homepage_linkedin_conflict_seam(bad)


@pytest.mark.parametrize('options', [{}, {'source_blocked_alias': True},
                                    {'identity_schema_repair': True},
                                    {'auxiliary_source_name_binding': True}])
def test_explicit_option_preserves_other_frozen_bytes_and_options(options):
    prior_needs_finality = bool(options.get('identity_schema_repair') or options.get('auxiliary_source_name_binding'))
    # The legacy auxiliary correction accepts only its reviewed historical body.
    # New industry source handling must not be retrofitted into that old image.
    frozen = (subprocess.check_output([
        "git", "show", "dbe8a014d2488a25f53649c4f61a48248147c54e:" + builder.FINALITY_FILE,
    ], cwd=builder.ROOT) if options.get('auxiliary_source_name_binding') else None)
    before = builder.expected_overlay(originals(prior_needs_finality, frozen), **options)
    after = builder.expected_overlay(originals(finality_source=frozen), **options,
                                     homepage_linkedin_conflict_repair=True)
    assert after[builder.FINALITY_FILE] == source_pair(frozen)[1]
    assert all(after[name] == data for name, data in before.items() if name != builder.FINALITY_FILE)
    assert set(after) == set(before) | {builder.FINALITY_FILE}
    builder.verify_overlay(layer([('model/' + name, data) for name, data in after.items()]), after)
    altered = dict(after)
    altered[builder.FINALITY_FILE] += b'\n# unrelated change\n'
    with pytest.raises(builder.CompositionError):
        builder.verify_overlay(layer([('model/' + name, data) for name, data in altered.items()]), after)


@pytest.mark.parametrize('verify_existing', [False, True])
def test_build_and_verify_share_exact_option_and_correction_identity(monkeypatch, capsys, verify_existing):
    grading = 'registry.example/grader@sha256:' + 'a' * 64
    runtime = 'registry.example/grader@sha256:' + 'b' * 64
    args = ['compose', '--grading-image', grading, '--homepage-linkedin-conflict-repair']
    if verify_existing:
        args += ['--verify-image', runtime]
    monkeypatch.setattr(builder.sys, 'argv', args)
    monkeypatch.setattr(builder, 'checked_output', lambda *args, **kwargs: 'c' * 40)
    base = {'Architecture': 'amd64', 'Os': 'linux', 'Config': {}, 'RootFS': {'Layers': ['base']}}
    candidate = deepcopy(base)
    candidate['RootFS']['Layers'].append('overlay')
    monkeypatch.setattr(builder, 'image_inspect', lambda name: base if name == grading else candidate)
    extractions = []
    def extract(*args, **kwargs):
        extractions.append(kwargs)
        return originals()
    monkeypatch.setattr(builder, 'extract_frozen_sources', extract)
    verified = []
    monkeypatch.setattr(builder, 'verify_saved_overlay', lambda image, digest, files: verified.append(files))
    commands = []
    monkeypatch.setattr(builder.subprocess, 'run', lambda command, **kwargs: commands.append(command))
    assert builder.main() == 0
    receipt = json.loads(capsys.readouterr().out)
    assert extractions[0]['homepage_linkedin_conflict_repair'] is True
    assert receipt['grading_corrections'] == [seam.CORRECTION_ID]
    assert receipt['base_layers_unchanged'] and receipt['process_configuration_unchanged']
    assert verified[0][builder.FINALITY_FILE] == source_pair()[1]
    assert bool(commands) is not verify_existing
    if verify_existing:
        assert receipt['runtime_image'] == runtime
    else:
        assert '-homepage-linkedin-conflict' in receipt['runtime_image']


@pytest.mark.parametrize('enabled', [False, True])
def test_frozen_extraction_adds_only_the_required_source(monkeypatch, tmp_path, enabled):
    files = originals()
    calls = []
    monkeypatch.setattr(builder, 'checked_output', lambda *args, **kwargs: 'container-id')
    def docker(command, **kwargs):
        calls.append(command)
        if command[:2] == ['docker', 'cp']:
            name = command[2].split(':/model/', 1)[1]
            Path(command[3]).write_bytes(files[name])
    monkeypatch.setattr(builder.subprocess, 'run', docker)
    extracted = builder.extract_frozen_sources(
        'registry.example/grader@sha256:' + 'a' * 64, tmp_path,
        homepage_linkedin_conflict_repair=enabled,
    )
    expected = {builder.GRADER_FILE, *builder.ENTRYPOINT_FILES}
    if enabled:
        expected.add(builder.FINALITY_FILE)
    assert set(extracted) == expected
    assert all(extracted[name] == files[name] for name in expected)
    assert calls[-1] == ['docker', 'rm', 'container-id']


def test_actual_frozen_source_when_supplied():
    directory = os.environ.get('ARENA_HOMEPAGE_LINKEDIN_FROZEN_ROOT')
    if not directory:
        pytest.skip('exact frozen image extraction not supplied')
    root = Path(directory)
    manifest = json.loads((root / 'manifest.json').read_text())
    assert manifest['image_id'] == 'sha256:8d0f613c781c2126c86effa91ea0bc7f9d28857521216e6346d522f4c0345bb1'
    source = (root / builder.FINALITY_FILE).read_bytes()
    assert hashlib.sha256(source).hexdigest() == manifest['files'][builder.FINALITY_FILE]['sha256']
    repaired = seam.convert_homepage_linkedin_conflict_seam(source)
    assert repaired != source
    assert seam.convert_homepage_linkedin_conflict_seam(repaired) == repaired
    for old, new in ((seam.OLD_DISPATCH, seam.NEW_DISPATCH), (seam.OLD_PROMPT, seam.NEW_PROMPT)):
        repaired = repaired.replace(new.encode(), old.encode(), 1)
    assert repaired == source


@pytest.mark.parametrize('name,input_sha256,output_sha256', [
    ('base', 'b573bd3fb3984cd065e48b37e19c8fdd59545319bef0f1801cdc94f1f502d3b2',
     '10638a78bbe457ec4f2ad880727dd1325a3773c543db0d4d4e8324251b8358d6'),
    ('runtime', 'b09dcafd2f4a83b2fcc31ed44c57a901f9e7f1a5b693d09c914def52bab0128b',
     'ee9fff67c5f8aa31680da8de05e9222e7e8b27dc45a5d3ad9d3790685a0781ca'),
])
def test_provenance_matched_historical_sources_when_supplied(name, input_sha256, output_sha256):
    directory = os.environ.get('ARENA_HOMEPAGE_LINKEDIN_HISTORICAL_SOURCES')
    if not directory:
        pytest.skip('historical source compatibility inputs not supplied')
    root = Path(directory)
    provenance = json.loads((root / 'provenance.json').read_text())
    assert provenance['base_git_commit'] == 'bbc374a26c8efa23b41840958b63e5d2e230e9ca'
    assert provenance['base_path'] == builder.FINALITY_FILE
    source = (root / (name + '.py')).read_bytes()
    assert hashlib.sha256(source).hexdigest() == provenance[name + '_sha256'] == input_sha256
    repaired = seam.convert_homepage_linkedin_conflict_seam(source)
    assert hashlib.sha256(repaired).hexdigest() == output_sha256
    assert seam.convert_homepage_linkedin_conflict_seam(repaired) == repaired
    compile(repaired, 'historical_' + name + '_identity_repair.py', 'exec')
    for old, new in ((seam.OLD_DISPATCH, seam.NEW_DISPATCH), (seam.OLD_PROMPT, seam.NEW_PROMPT)):
        assert repaired.count(new.encode()) == 1
        repaired = repaired.replace(new.encode(), old.encode(), 1)
    # This includes every pre-existing runtime finality repair, byte for byte.
    assert repaired == source


def test_direct_base_current_and_candidate_composition_when_supplied():
    directory = os.environ.get('ARENA_HOMEPAGE_LINKEDIN_IMAGE_INPUTS')
    if not directory:
        pytest.skip('direct base and current-runtime image extracts not supplied')
    root = Path(directory)
    manifests = {kind: json.loads((root / kind / 'manifest.json').read_text())
                 for kind in ('base', 'current')}
    repository = '493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:'
    assert manifests['base']['reference'] == repository + '8d0f613c781c2126c86effa91ea0bc7f9d28857521216e6346d522f4c0345bb1'
    assert manifests['current']['reference'] == repository + 'd2d8f1ea02eecb5dc0e8572b86868ccb84f692314f16fcf98ffda01772cdbe4b'
    for kind, manifest in manifests.items():
        assert manifest['reference'] in manifest['repo_digests']
        assert manifest['missing'] == []
        assert manifest['architecture'] == 'amd64' and manifest['os'] == 'linux'
        for name, entry in manifest['files'].items():
            data = (root / kind / name).read_bytes()
            assert hashlib.sha256(data).hexdigest() == entry['sha256']
            assert len(data) == entry['bytes']
    base, current = manifests['base'], manifests['current']
    assert base['process_configuration_sha256'] == current['process_configuration_sha256']
    assert current['layers'][:-1] == base['layers']
    names = (builder.GRADER_FILE, builder.FINALITY_FILE, *builder.ENTRYPOINT_FILES)
    originals = {name: (root / 'base' / name).read_bytes() for name in names}
    existing_flags = {'source_blocked_alias': True, 'completed_investigation_finality': True,
                      'auxiliary_source_name_binding': True}
    reproduced = builder.expected_overlay(originals, **existing_flags)
    assert len(reproduced) == 8
    assert all(data == (root / 'current' / name).read_bytes() for name, data in reproduced.items())
    candidate = builder.expected_overlay(originals, **existing_flags,
                                         homepage_linkedin_conflict_repair=True)
    assert set(candidate) == set(reproduced)
    assert [name for name in candidate if candidate[name] != reproduced[name]] == [builder.FINALITY_FILE]
    assert candidate[builder.FINALITY_FILE] == seam.convert_homepage_linkedin_conflict_seam(reproduced[builder.FINALITY_FILE])
    assert hashlib.sha256(candidate[builder.FINALITY_FILE]).hexdigest() == '60ceb054effcf4b22ccb3e3da0a4368c586310b384db7451b1e87fd19f25b166'
    # Remaining captured grader files are unchanged by the existing runtime.
    for name in set(base['files']) - set(candidate):
        assert base['files'][name] == current['files'][name]
    builder.verify_overlay(layer([('model/' + name, data) for name, data in candidate.items()]), candidate)
