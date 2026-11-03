"""Five-file exact composition retains alias/denial and adds only brand attribution."""
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
from scripts import arena_reviewer_homepage_profile_alias_seam as homepage
from scripts import arena_reviewer_verified_display_brand_seam as seam
from scripts import arena_reviewer_intent_target_denial_seam as intent
from tests.lab_arena.reviewer_runtime_composition_test import layer
from tests.lab_arena.reviewer_homepage_profile_alias_composition_test import frozen_sources as homepage_sources


FIXTURES = Path(__file__).resolve().parents[1] / 'fixtures' / 'reviewer_seams'


def frozen_sources():
    sources = homepage_sources()
    sources[seam.INVESTIGATOR_FILE] = gzip.decompress((FIXTURES /
        'verified-display-brand-company_evidence_investigator.py.gz').read_bytes())
    assert {name: hashlib.sha256(source).hexdigest() for name, source in sources.items()} == seam.ORIGINAL_SHA256
    return sources


def test_exact_composition_retains_four_repairs_and_only_reviewed_investigator_delta():
    original = frozen_sources()
    repaired = seam.convert_verified_display_brand_files(original)
    assert seam.convert_verified_display_brand_files(repaired) == repaired
    assert {n: repaired[n] for n in homepage.FILES} == homepage.convert_homepage_profile_alias_files(homepage_sources())
    assert {n: hashlib.sha256(v).hexdigest() for n,v in repaired.items()} == seam.REPAIRED_SHA256
    text = original[seam.INVESTIGATOR_FILE]
    for start, end, replacement in reversed(seam.REPLACEMENTS):
        text = text[:start] + replacement + text[end:]
    assert text == repaired[seam.INVESTIGATOR_FILE]
    # Remove exactly the helper and changed callers: all remaining source is identical.
    def outside_owned(source):
        owned = {'_quote_names_verified_display_brand', '_required_attribute_support_rejection', '_validated_findings'}
        tree = ast.parse(source)
        tree.body = [node for node in tree.body if not (
            isinstance(node, ast.FunctionDef) and node.name in owned)]
        return ast.dump(tree, include_attributes=False)
    assert outside_owned(original[seam.INVESTIGATOR_FILE]) == outside_owned(text)
    assert frozen_sources() == original


@pytest.mark.parametrize('corrected', [False, True])
@pytest.mark.parametrize('fault', ['missing', 'extra', 'unknown', 'mixed_investigator',
    'mixed_alias', 'mixed_denial', 'newline', 'invalid', 'shadowed'])
def test_unknown_partial_and_mixed_five_file_generations_fail_closed(corrected, fault):
    original = frozen_sources()
    repaired = seam.convert_verified_display_brand_files(original)
    sources = dict(repaired if corrected else original)
    opposite = original if corrected else repaired
    name = seam.INVESTIGATOR_FILE
    if fault == 'missing': sources.pop(name)
    elif fault == 'extra': sources['arena_reviewer_runtime/__init__.py'] = b'unapproved'
    elif fault == 'mixed_investigator': sources[name] = opposite[name]
    elif fault == 'mixed_alias': sources[homepage.ALIAS_FILES[0]] = opposite[homepage.ALIAS_FILES[0]]
    elif fault == 'mixed_denial': sources[intent.FILES[0]] = opposite[intent.FILES[0]]
    elif fault == 'newline': sources[name] = sources[name].replace(b'\n', b'\r\n')
    elif fault == 'invalid': sources[name] = b'\xff'
    elif fault == 'shadowed': sources[name] += b'\ndef _quote_names_verified_display_brand(*args): return True\n'
    else: sources[name] += b'\nunknown = True\n'
    with pytest.raises(seam.VerifiedDisplayBrandSeamError):
        seam.convert_verified_display_brand_files(sources)


def test_frozen_fifth_module_executes_approved_quote_and_bounded_correction(monkeypatch):
    from tests import test_verified_display_brand_quote as regression
    sources = frozen_sources()
    repaired = seam.convert_verified_display_brand_files(sources)
    def load(label, source):
        module = ModuleType(label)
        monkeypatch.setitem(sys.modules, label, module)
        exec(compile(source, seam.INVESTIGATOR_FILE, 'exec'), module.__dict__)
        return module
    original = load('brand_attribution_original', sources[seam.INVESTIGATOR_FILE])
    corrected = load('brand_attribution_corrected', repaired[seam.INVESTIGATOR_FILE])
    monkeypatch.setattr(regression, 'inv', original)
    assert regression.validate()['status'] == 'UNPROVEN'
    monkeypatch.setattr(regression, 'inv', corrected)
    for quote in (regression.QUOTE, regression.SMART):
        regression.test_exact_brand_quotes_are_not_rejected_for_the_ownership_display_suffix(quote)
    for quote in ('Captify’s parent supplies a platform.', 'Captifyx supplies a platform.'):
        regression.test_parent_related_company_and_name_prefix_are_not_brand_evidence(quote)
    for changes in ({'submitted_domain': 'other.example'}, {'observed_linkedin_slug': 'other'}):
        regression.test_missing_or_conflicting_independent_anchors_stay_unproven(changes)
    regression.test_quote_must_still_occur_in_loaded_text()
    regression.test_industry_still_requires_supplier_operator()
    regression.test_bounded_correction_completes_with_named_primary_and_named_support(monkeypatch)


@pytest.mark.parametrize('flag', [
    '--shim-only', '--intent-target-denial', '--complete-negative-finality', '--source-access-denied',
    '--source-blocked-alias', '--pending-final-correction', '--failed-attribute-source-recovery',
    '--completed-investigation-finality', '--identity-schema-repair', '--authenticated-page-refusal',
    '--single-tool-protocol-repair', '--rejected-final-packet', '--structured-homepage-numeric-alias',
    '--bound-homepage-legal-alias', '--numeric-profile-legal-self-name', '--archive-snapshot-https',
    '--greenhouse-api-source', '--auxiliary-source-name-binding', '--homepage-linkedin-conflict-repair',
    '--observed-linkedin-ampersand', '--verified-homepage-brand-alias',
    '--homepage-profile-alias',
])
def test_five_file_mode_rejects_other_corrections_before_image_access(monkeypatch, flag):
    monkeypatch.setattr(builder.sys, 'argv', ['compose', '--grading-image',
        'registry.example/reviewer@sha256:' + 'a' * 64, '--verified-display-brand', flag])
    monkeypatch.setattr(builder, 'image_inspect', lambda *_: pytest.fail('image access'))
    with pytest.raises(builder.CompositionError, match='cannot include'): builder.main()


def test_extraction_reads_exactly_five_frozen_modules(monkeypatch, tmp_path):
    monkeypatch.setattr(builder, 'checked_output', lambda *_: 'container')
    sources, commands = frozen_sources(), []
    def run(command, **_kwargs):
        commands.append(command)
        if command[1] == 'cp':
            name = command[2].removeprefix('container:/model/')
            Path(command[3]).write_bytes(sources[name])
    monkeypatch.setattr(builder.subprocess, 'run', run)
    assert builder.extract_frozen_sources('pinned', tmp_path, verified_display_brand=True) == sources
    assert [command[2] for command in commands if command[1] == 'cp'] == ['container:/model/' + name for name in seam.FILES]


@pytest.mark.parametrize('verify', [False, True])
def test_cli_build_and_verification_keep_exact_five_files_and_three_corrections(monkeypatch, capsys, verify):
    grading = 'registry.example/reviewer@sha256:' + 'a' * 64
    runtime = 'registry.example/reviewer@sha256:' + 'b' * 64
    argv = ['compose', '--grading-image', grading, '--verified-display-brand']
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
    assert extracted == [(grading, {'verified_display_brand': True})] and len(checked) == 1
    assert evidence['grading_image'] == grading
    assert evidence['grading_corrections'] == [intent.CORRECTION_ID, homepage.CORRECTION_ID, seam.CORRECTION_ID]
    assert evidence['host_validator_requirement'] == seam.HOST_VALIDATOR_REQUIREMENT
    assert evidence['original_files'] == seam.ORIGINAL_SHA256
    assert evidence['overlay_files'] == seam.REPAIRED_SHA256
    assert evidence['base_layers_unchanged'] and evidence['process_configuration_unchanged']
    assert len(builds) == (0 if verify else 1)
    if verify: assert evidence['runtime_image'] == runtime
    else: assert '-verified-display-brand' in evidence['runtime_image']
