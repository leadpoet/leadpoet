"""Exact frozen-source repair keeps identity and semantic proof independent."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path

import pytest
from tests.reviewer_frozen_sources import frozen_source

from qualification.scoring import lead_scorer as scorer
from scripts import arena_reviewer_identity_schema_seam as seam
from scripts import build_arena_reviewer_runtime as composer
from scripts.arena_reviewer_investigation_finality_seam import convert_investigation_finality_seam
from tests import test_company_identity_schema_repair as controls

ROOT = composer.ROOT


# Exact CI-green native-budget postimage; historical overlays must not track
# the evolving current investigator or require Git history in test installs.
_FROZEN_GRADER = frozen_source(
    "native_industry_budget", "qualification/scoring/company_evidence_investigator.py",
)


def _sources():
    repaired = (ROOT / composer.FINALITY_FILE).read_bytes()
    original = repaired
    for old, new in seam.DELTAS:
        assert original.count(new.encode()) == 1
        original = original.replace(new.encode(), old.encode(), 1)
    return original, repaired


def _install(monkeypatch, source):
    # Compile the exact frozen verifier, retaining the existing dependency
    # namespace so the controlled provider and investigator patches apply.
    node = next(node for node in ast.parse(source).body
                if isinstance(node, ast.AsyncFunctionDef)
                and node.name == '_llm_reverify_company')
    module = ast.Module(body=[ast.ImportFrom(module='__future__',
        names=[ast.alias(name='annotations')], level=0), node], type_ignores=[])
    ast.fix_missing_locations(module)
    monkeypatch.setattr(scorer, '_llm_reverify_company', scorer._llm_reverify_company)
    exec(compile(module, 'exact_frozen_identity_verifier.py', 'exec'), vars(scorer))


def test_only_the_three_reviewed_deltas_and_idempotent():
    original, repaired = _sources()
    assert seam.convert_identity_schema_seam(original) == repaired
    assert seam.convert_identity_schema_seam(repaired) == repaired
    before, after = ast.parse(original), ast.parse(repaired)
    for tree in (before, after):
        tree.body = [node for node in tree.body
                     if getattr(node, 'name', '') != '_llm_reverify_company']
    assert ast.dump(before) == ast.dump(after)


@pytest.mark.parametrize('fault', ['init_only', 'gate_only', 'repair_only',
    'changed_gate', 'changed_repair', 'duplicate', 'missing', 'shadowed',
    'extra_assignment', 'crlf', 'utf8'])
def test_unknown_or_partial_shapes_fail_closed(fault):
    original, repaired = _sources()
    source = original
    if fault in {'init_only', 'gate_only', 'repair_only'}:
        index = {'init_only': 0, 'gate_only': 1, 'repair_only': 2}[fault]
        old, new = seam.DELTAS[index]
        source = source.replace(old.encode(), new.encode(), 1)
    elif fault == 'changed_gate':
        source = repaired.replace(b'and "identity" in missing_dimensions',
                                  b'and "industry" in missing_dimensions', 1)
    elif fault == 'changed_repair':
        source = repaired.replace(b'repaired_verdict, active_attribute=False,',
                                  b'repaired_verdict, active_attribute=True,', 1)
    elif fault == 'duplicate':
        node = next(node for node in ast.parse(original).body
                    if getattr(node, 'name', '') == '_llm_reverify_company')
        source += b'\n' + ast.get_source_segment(original.decode(), node).encode()
    elif fault == 'missing':
        source = source.replace(b'async def _llm_reverify_company(', b'async def other(', 1)
    elif fault == 'shadowed':
        source += b'\n_llm_reverify_company = None\n'
    elif fault == 'extra_assignment':
        source = repaired.replace(b'    repair_missing_identity = False\n',
                                  b'    repair_missing_identity = False\n    repair_missing_identity = True\n', 1)
    elif fault == 'crlf':
        source = source.replace(b'\n', b'\r\n')
    else:
        source = b'\xff'
    with pytest.raises(seam.IdentitySchemaSeamError):
        seam.convert_identity_schema_seam(source)


def test_builder_opt_in_preserves_every_other_file():
    original, repaired = _sources()
    sources = {
        name: (ROOT / ('lab_arena/scorer_entrypoint.py'
                       if name == 'scorer_entrypoint.py' else name)).read_bytes()
        for name in (composer.GRADER_FILE, *composer.ENTRYPOINT_FILES)
    }
    sources[composer.GRADER_FILE] = _FROZEN_GRADER
    default = composer.expected_overlay(sources)
    assert composer.FINALITY_FILE not in default
    with pytest.raises(composer.CompositionError):
        composer.expected_overlay(sources, identity_schema_repair=True)
    sources[composer.FINALITY_FILE] = original
    prior = composer.expected_overlay(sources, completed_investigation_finality=True)
    after = composer.expected_overlay(sources, completed_investigation_finality=True,
                                     identity_schema_repair=True)
    assert prior[composer.FINALITY_FILE] == original
    assert after[composer.FINALITY_FILE] == repaired
    assert {k: v for k, v in after.items() if k != composer.FINALITY_FILE} == default


@pytest.mark.parametrize('correction', [False, True])
@pytest.mark.parametrize('verify_existing', [False, True])
def test_build_and_verify_preserve_prior_repair_and_emit_distinct_identity(
    monkeypatch, capsys, correction, verify_existing,
):
    from scripts.arena_reviewer_protocol_repair_seam import CORRECTION_ID as PROTOCOL_ID
    grading = 'registry.example/reviewer@sha256:' + 'a' * 64
    runtime = 'registry.example/reviewer@sha256:' + 'b' * 64
    args = ['build_arena_reviewer_runtime.py', '--grading-image', grading,
            '--single-tool-protocol-repair']
    if verify_existing:
        args.extend(['--verify-image', runtime])
    else:
        runtime = ('registry.example/reviewer:reviewer-' + 'a' * 12 + '-' + 'c' * 12
                   + ('-identity-schema-repair' if correction else '') + '-single-tool-repair')
    if correction:
        args.append('--identity-schema-repair')
    monkeypatch.setattr(composer.sys, 'argv', args)
    monkeypatch.setattr(composer, 'checked_output', lambda *a, **kw: 'c' * 40)
    base = {'Architecture': 'amd64', 'Os': 'linux', 'Config': {},
            'RootFS': {'Layers': ['sha256:base']}}
    candidate = copy.deepcopy(base)
    candidate['RootFS']['Layers'].append('sha256:overlay')
    monkeypatch.setattr(composer, 'image_inspect',
                        lambda image: base if image == grading else candidate)

    def extract(image, destination, **kwargs):
        assert image == grading
        assert kwargs.pop('identity_schema_repair', False) is correction
        assert not any(kwargs.values())
        sources = {name: (ROOT / ('lab_arena/scorer_entrypoint.py'
                    if name == 'scorer_entrypoint.py' else name)).read_bytes()
                   for name in (composer.GRADER_FILE, *composer.ENTRYPOINT_FILES)}
        sources[composer.GRADER_FILE] = _FROZEN_GRADER
        if correction:
            sources[composer.FINALITY_FILE] = _sources()[0]
        return sources

    def verify(image, digest, files):
        assert image == runtime and digest == 'sha256:overlay'
        assert (composer.FINALITY_FILE in files) is correction
        if correction:
            assert files[composer.FINALITY_FILE] == _sources()[1]

    monkeypatch.setattr(composer, 'extract_frozen_sources', extract)
    monkeypatch.setattr(composer, 'verify_saved_overlay', verify)
    builds = []
    monkeypatch.setattr(composer.subprocess, 'run', lambda command, **kw: builds.append(command))
    assert composer.main() == 0
    evidence = json.loads(capsys.readouterr().out)
    assert evidence['grading_corrections'] == ([seam.CORRECTION_ID] if correction else []) + [PROTOCOL_ID]
    assert evidence['grading_image'] == grading
    assert evidence['runtime_image'] == runtime
    assert len(builds) == (0 if verify_existing else 1)


def test_original_bug_then_existing_repair_without_false_positive(monkeypatch):
    original, _ = _sources()
    _install(monkeypatch, original)
    before, calls, _ = controls.exercise(monkeypatch, attribute=True)
    assert calls == ['lead_scorer_reverify']
    assert before.details['identity_decision'] != scorer.COMPANY_FIT_MATCH
    assert before.details.get('failure_class') != 'insufficient_fit_evidence'
    _install(monkeypatch, seam.convert_identity_schema_seam(original))
    after, calls, investigations = controls.exercise(monkeypatch, attribute=True)
    assert calls == ['lead_scorer_reverify', 'lead_scorer_reverify_schema_repair']
    assert len(investigations) == 1
    assert after.details['identity_decision'] == scorer.COMPANY_FIT_MATCH
    assert after.decision == scorer.COMPANY_FIT_UNAVAILABLE
    assert after.details['dimension_decisions']['industry'] == scorer.COMPANY_FIT_UNAVAILABLE
    assert after.details['failure_class'] == 'insufficient_fit_evidence'


@pytest.mark.parametrize('repair', ['missing', 'other_entity', 'conflicting_slug', 'provider_error'])
def test_frozen_repair_failure_stays_retryable(monkeypatch, repair):
    original, _ = _sources()
    _install(monkeypatch, seam.convert_identity_schema_seam(original))
    result, calls, _ = controls.exercise(monkeypatch, repaired=repair)
    assert calls == ['lead_scorer_reverify', 'lead_scorer_reverify_schema_repair']
    assert result.decision != scorer.COMPANY_FIT_MATCH
    assert result.details.get('failure_class') != 'insufficient_fit_evidence'


def test_actual_frozen_release_sources_compose_and_reproduce(monkeypatch):
    base_path = os.environ.get('ARENA_IDENTITY_FROZEN_SOURCE')
    runtime_path = os.environ.get('ARENA_IDENTITY_RUNTIME_SOURCE')
    if not base_path or not runtime_path:
        pytest.skip('actual frozen release sources are supplied by the release gate')
    runtime = Path(runtime_path).read_bytes()
    assert hashlib.sha256(runtime).hexdigest() == '65b75c2fd91f44aa7257d92c83c3381acb92e8f73c8cde638c235848f07b288e'
    base = Path(base_path).read_bytes()
    corrected = seam.convert_identity_schema_seam(runtime)
    assert corrected == seam.convert_identity_schema_seam(convert_investigation_finality_seam(base))
    assert seam.convert_identity_schema_seam(corrected) == corrected
    _install(monkeypatch, runtime)
    before, calls, _ = controls.exercise(monkeypatch, attribute=True)
    assert calls == ['lead_scorer_reverify']
    assert before.details.get('failure_class') != 'insufficient_fit_evidence'
    _install(monkeypatch, corrected)
    after, calls, _ = controls.exercise(monkeypatch, attribute=True)
    assert calls == ['lead_scorer_reverify', 'lead_scorer_reverify_schema_repair']
    assert after.details['identity_decision'] == scorer.COMPANY_FIT_MATCH
    assert after.decision == scorer.COMPANY_FIT_UNAVAILABLE
    assert after.details['failure_class'] == 'insufficient_fit_evidence'
    result, calls, _ = controls.exercise(monkeypatch, repaired='provider_error')
    assert len(calls) == 2
    assert result.details.get('failure_class') != 'insufficient_fit_evidence'


@pytest.mark.parametrize('name,kwargs', [
    ('test_missing_identity_uses_existing_one_repair_without_promoting_unproven', {'attribute': False}),
    ('test_missing_identity_uses_existing_one_repair_without_promoting_unproven', {'attribute': True}),
    ('test_investigated_unproven_without_identity_gap_does_not_add_call', {}),
    *[('test_identity_repair_failure_cannot_qualify', {'repair': repair})
      for repair in ('missing', 'other_entity', 'conflicting_slug', 'provider_error')],
    ('test_incomplete_research_does_not_get_identity_repair', {}),
    ('test_repaired_identity_cannot_clear_unresolved_source_fault', {}),
    ('test_investigated_identity_keeps_existing_target_bound', {}),
    ('test_missing_or_malformed_receipt_does_not_open_repair', {'receipt_shape': 'missing'}),
    ('test_missing_or_malformed_receipt_does_not_open_repair', {'receipt_shape': 'malformed'}),
    ('test_identity_repair_preserves_independently_proven_attribute_and_stage', {}),
    ('test_real_targeted_projection_reaches_existing_identity_repair', {}),
])
def test_actual_frozen_verifier_all_identity_controls(monkeypatch, name, kwargs):
    path = os.environ.get('ARENA_IDENTITY_COMPOSED_SOURCE') or os.environ.get('ARENA_IDENTITY_RUNTIME_SOURCE')
    if not path:
        pytest.skip('actual frozen release source is supplied by the release gate')
    source = Path(path).read_bytes()
    _install(monkeypatch, seam.convert_identity_schema_seam(source))
    getattr(controls, name)(monkeypatch, **kwargs)
