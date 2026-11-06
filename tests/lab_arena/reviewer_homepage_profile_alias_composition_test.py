"""Four-file composition retains native denial and exact old-profile proof."""
import ast
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest

from scripts import build_arena_reviewer_runtime as builder
from scripts import arena_reviewer_homepage_profile_alias_seam as seam
from scripts import arena_reviewer_intent_target_denial_seam as intent
from tests.lab_arena.reviewer_runtime_composition_test import layer
from tests.lab_arena.reviewer_intent_target_denial_composition_test import frozen_sources as intent_sources


FIXTURES = Path(__file__).resolve().parents[1] / 'fixtures' / 'reviewer_seams'


def frozen_sources():
    sources = intent_sources()
    for name in seam.ALIAS_FILES:
        sources[name] = gzip.decompress((FIXTURES / (
            'homepage-profile-alias-' + Path(name).name + '.gz'
        )).read_bytes())
    assert {name: hashlib.sha256(source).hexdigest() for name, source in sources.items()} == seam.ORIGINAL_SHA256
    return sources


def test_composition_changes_only_reviewed_functions_and_retains_both_denial_lines():
    original = frozen_sources()
    repaired = seam.convert_homepage_profile_alias_files(original)
    assert seam.convert_homepage_profile_alias_files(repaired) == repaired
    already_denied = {**original, **intent.convert_intent_target_denial_files({name: original[name] for name in intent.FILES})}
    assert seam.convert_homepage_profile_alias_files(already_denied) == repaired
    owners = {
        seam.ALIAS_FILES[0]: {'_unresolved_homepage_web_identity_conflict', '_conflicting_homepage_numeric_profile_lookup'},
        seam.ALIAS_FILES[1]: {'_structured_company_elements', 'project_structured_linkedin_company_identity'},
    }
    def remove_owned(source, names):
        text = source.decode()
        lines = text.splitlines(keepends=True)
        nodes = [node for node in ast.parse(text).body if isinstance(node, ast.FunctionDef) and node.name in names]
        assert {node.name for node in nodes} == names
        for node in reversed(nodes): lines[node.lineno - 1:node.end_lineno] = []
        return ''.join(lines)
    for name, names in owners.items():
        assert remove_owned(original[name], names) == remove_owned(repaired[name], names)
    for name in intent.FILES:
        assert repaired[name].replace(intent.INSERTION, b'', 1) == original[name]


@pytest.mark.parametrize('fault', ['missing', 'extra', 'unknown', 'partial_alias', 'partial_denial', 'alias_without_denial'])
def test_unknown_or_partial_four_file_generation_fails_closed(fault):
    sources = frozen_sources()
    repaired = seam.convert_homepage_profile_alias_files(sources)
    if fault == 'missing': sources.pop(seam.ALIAS_FILES[0])
    elif fault == 'extra': sources['lab_arena/scoring.py'] = b'unapproved'
    elif fault == 'unknown': sources[seam.ALIAS_FILES[0]] += b'\nunknown = True\n'
    elif fault == 'partial_alias': sources[seam.ALIAS_FILES[0]] = repaired[seam.ALIAS_FILES[0]]
    elif fault == 'partial_denial': sources[intent.FILES[0]] = repaired[intent.FILES[0]]
    else:
        sources.update({name: repaired[name] for name in seam.ALIAS_FILES})
    with pytest.raises(ValueError): seam.convert_homepage_profile_alias_files(sources)


def test_exact_composed_alias_modules_execute_proof_and_transport_controls(monkeypatch):
    import qualification.scoring
    from tests import test_vanity_homepage_profile_identity as regression
    repaired = seam.convert_homepage_profile_alias_files(frozen_sources())
    modules = {}
    for name in reversed(seam.ALIAS_FILES):
        label = 'qualification.scoring.' + Path(name).stem
        module = ModuleType(label)
        monkeypatch.setitem(sys.modules, label, module)
        exec(compile(repaired[name], name, 'exec'), module.__dict__)
        monkeypatch.setattr(qualification.scoring, Path(name).stem, module)
        modules[name] = module
    monkeypatch.setattr(regression, 'scorer', modules[seam.ALIAS_FILES[0]])
    monkeypatch.setattr(regression, 'linkedin', modules[seam.ALIAS_FILES[1]])
    regression.test_real_web_path_replaces_current_lookup_with_one_old_vanity_request(monkeypatch, 'canonical')
    regression.test_real_web_path_replaces_current_lookup_with_one_old_vanity_request(monkeypatch, 'transport')
    regression.test_nonterminal_outer_job_cannot_prove_vanity_alias('pending')
    regression.test_job_without_status_and_competing_terminal_element_paths_fail_closed()


@pytest.mark.parametrize('flag', [
    '--shim-only', '--intent-target-denial', '--complete-negative-finality', '--source-access-denied',
    '--source-blocked-alias', '--pending-final-correction', '--failed-attribute-source-recovery',
    '--completed-investigation-finality', '--identity-schema-repair', '--authenticated-page-refusal',
    '--single-tool-protocol-repair', '--rejected-final-packet', '--structured-homepage-numeric-alias',
    '--bound-homepage-legal-alias', '--numeric-profile-legal-self-name', '--archive-snapshot-https',
    '--greenhouse-api-source', '--auxiliary-source-name-binding', '--homepage-linkedin-conflict-repair',
    '--observed-linkedin-ampersand', '--verified-homepage-brand-alias',
    '--verified-display-brand',
])
def test_four_file_mode_rejects_other_corrections_before_image_access(monkeypatch, flag):
    monkeypatch.setattr(builder.sys, 'argv', ['compose', '--grading-image',
        'registry.example/reviewer@sha256:' + 'a' * 64, '--homepage-profile-alias', flag])
    monkeypatch.setattr(builder, 'image_inspect', lambda *_: pytest.fail('image access'))
    with pytest.raises(builder.CompositionError, match='cannot include'): builder.main()


def test_extraction_reads_exactly_four_frozen_modules(monkeypatch, tmp_path):
    monkeypatch.setattr(builder, 'checked_output', lambda *_: 'container')
    sources, commands = frozen_sources(), []
    def run(command, **_kwargs):
        commands.append(command)
        if command[1] == 'cp':
            name = command[2].removeprefix('container:/model/')
            Path(command[3]).write_bytes(sources[name])
    monkeypatch.setattr(builder.subprocess, 'run', run)
    assert builder.extract_frozen_sources('pinned', tmp_path, homepage_profile_alias=True) == sources
    assert [command[2] for command in commands if command[1] == 'cp'] == ['container:/model/' + name for name in seam.FILES]


@pytest.mark.parametrize('verify', [False, True])
def test_cli_build_and_verification_keep_exact_four_files_and_both_corrections(monkeypatch, capsys, verify):
    grading = 'registry.example/reviewer@sha256:' + 'a' * 64
    runtime = 'registry.example/reviewer@sha256:' + 'b' * 64
    argv = ['compose', '--grading-image', grading, '--homepage-profile-alias']
    if verify: argv += ['--verify-image', runtime]
    monkeypatch.setattr(builder.sys, 'argv', argv)
    monkeypatch.setattr(builder, '_managed_docker_host', lambda: False)
    monkeypatch.setattr(builder, 'checked_output', lambda *a, **k: 'c' * 40)
    base = {'Architecture': 'amd64', 'Os': 'linux', 'Config': {'User': '65534:65534'}, 'RootFS': {'Layers': ['base']}}
    candidate = deepcopy(base)
    candidate['RootFS']['Layers'].append('repair-layer')
    monkeypatch.setattr(builder, 'image_inspect', lambda name: base if name == grading else candidate)
    extracted, builds, checked = [], [], []
    def extract(reference, destination, **kwargs):
        extracted.append((reference, kwargs))
        return frozen_sources()
    def check(reference, digest, files):
        checked.append(files)
        assert set(files) == set(seam.FILES)
        builder.verify_overlay(layer([('model/' + n, d) for n, d in files.items()]), files)
        with pytest.raises(builder.CompositionError):
            builder.verify_overlay(layer([*(('model/' + n, d) for n, d in files.items()),
                ('model/arena_reviewer_runtime/__init__.py', b'unapproved')]), files)
    monkeypatch.setattr(builder, 'extract_frozen_sources', extract)
    monkeypatch.setattr(builder, 'expected_overlay', lambda *a, **k: pytest.fail('current runtime/grader overlay used'))
    monkeypatch.setattr(builder, 'verify_saved_overlay', check)
    monkeypatch.setattr(builder.subprocess, 'run', lambda command, **kwargs: builds.append(command))
    assert builder.main() == 0
    evidence = json.loads(capsys.readouterr().out)
    assert extracted == [(grading, {'homepage_profile_alias': True})] and len(checked) == 1
    assert evidence['grading_image'] == grading
    assert evidence['grading_corrections'] == [intent.CORRECTION_ID, seam.CORRECTION_ID]
    assert evidence['host_validator_requirement'] == seam.HOST_VALIDATOR_REQUIREMENT
    assert evidence['original_files'] == seam.ORIGINAL_SHA256
    assert evidence['overlay_files'] == seam.REPAIRED_SHA256
    assert evidence['base_layers_unchanged'] and evidence['process_configuration_unchanged']
    assert len(builds) == (0 if verify else 1)
    if verify: assert evidence['runtime_image'] == runtime
    else: assert '-homepage-profile-alias' in evidence['runtime_image']
