"""The explicit legal/brand correction preserves all other frozen source."""
import ast
import hashlib
from pathlib import Path
import subprocess

import pytest

from scripts import arena_reviewer_bound_homepage_alias_seam as seam
from scripts import build_arena_reviewer_runtime as builder


def repaired_sources():
    # This whole-module fixture represents one reviewed historical bundle.
    # Pin every module so later independent source changes cannot mix revisions.
    sources = {}
    for name in seam.DELTAS:
        sources[name] = subprocess.check_output(
            ["git", "show", "1b1da553508b639a8c9f6edf6940e67fb8b77233:" + name],
            cwd=builder.ROOT,
        )
        assert hashlib.sha256(sources[name]).hexdigest() in (
            seam.ORIGINAL_TO_REPAIRED_SHA256[name].values()
        )
    return sources


def original_sources():
    sources = repaired_sources()
    for name, changes in seam.DELTAS.items():
        text = sources[name].decode()
        for before, after in reversed(changes):
            assert text.count(after) == 1
            text = text.replace(after, before, 1)
        sources[name] = text.encode()
        assert hashlib.sha256(sources[name]).hexdigest() in seam.ORIGINAL_TO_REPAIRED_SHA256[name]
    return sources


def test_exact_complete_correction_and_idempotence():
    repaired = repaired_sources()
    assert seam.convert_bound_homepage_legal_alias_files(original_sources()) == repaired
    assert seam.convert_bound_homepage_legal_alias_files(repaired) == repaired


@pytest.mark.parametrize('path', list(seam.DELTAS))
def test_unrecognized_source_fails_closed(path):
    sources = original_sources()
    sources[path] += b'\n# unreviewed bytes\n'
    with pytest.raises(seam.BoundHomepageAliasSeamError, match='unrecognized'):
        seam.convert_bound_homepage_legal_alias_files(sources)


def test_partial_or_incomplete_correction_fails_closed():
    sources = original_sources()
    path = next(iter(sources))
    sources[path] = repaired_sources()[path]
    with pytest.raises(seam.BoundHomepageAliasSeamError, match='partially corrected'):
        seam.convert_bound_homepage_legal_alias_files(sources)
    sources.pop(path)
    with pytest.raises(seam.BoundHomepageAliasSeamError, match='incomplete'):
        seam.convert_bound_homepage_legal_alias_files(sources)


def _node(source, name):
    nodes = [node for node in ast.parse(source).body if getattr(node, 'name', '') == name]
    assert len(nodes) == 1
    return ast.dump(nodes[0], include_attributes=False)


def test_prompt_tools_and_headcount_evidence_rules_do_not_change():
    original, repaired = original_sources(), repaired_sources()
    investigator = builder.GRADER_FILE
    for name in ('_tools', '_quote_supports_headcount', '_complete_verified_first_party_identity'):
        assert _node(original[investigator], name) == _node(repaired[investigator], name)
    old = ast.parse(original[investigator])
    new = ast.parse(repaired[investigator])
    def prompt(module):
        return [ast.dump(node, include_attributes=False) for node in module.body
                if isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == '_SYSTEM_PROMPT' for t in node.targets)]
    assert prompt(old) == prompt(new)
    for name, changes in seam.DELTAS.items():
        before, after = original[name].decode(), repaired[name].decode()
        for old_text, new_text in changes:
            before = before.replace(old_text, '', 1)
            after = after.replace(new_text, '', 1)
        assert before == after


def test_correction_is_explicit_and_composes_only_reviewed_files():
    original = original_sources()
    for name in builder.ENTRYPOINT_FILES:
        path = builder.ROOT / ('lab_arena/scorer_entrypoint.py' if name == 'scorer_entrypoint.py' else name)
        original[name] = path.read_bytes()
    files = builder.expected_overlay(original, bound_homepage_legal_alias=True)
    for name, data in repaired_sources().items():
        assert files[name] == data
    baseline = {name: original[name] for name in (builder.GRADER_FILE, *builder.ENTRYPOINT_FILES)}
    unchanged = builder.expected_overlay(baseline)
    assert not set(builder.BOUND_IDENTITY_FILES) & set(unchanged)
    assert builder.FINALITY_FILE not in unchanged
    with pytest.raises(builder.CompositionError, match='incomplete'):
        builder.expected_overlay(original)


def test_builder_flag_records_explicit_identity_correction(monkeypatch, capsys):
    from copy import deepcopy
    import json
    sources = original_sources()
    for name in builder.ENTRYPOINT_FILES:
        path = builder.ROOT / ('lab_arena/scorer_entrypoint.py' if name == 'scorer_entrypoint.py' else name)
        sources[name] = path.read_bytes()
    base = {'Architecture': 'amd64', 'Os': 'linux', 'Config': {},
            'RootFS': {'Layers': ['sha256:base']}}
    candidate = deepcopy(base)
    candidate['RootFS']['Layers'].append('sha256:overlay')
    grading = 'registry.example/model@sha256:' + 'a' * 64
    seen = {}
    monkeypatch.setattr('sys.argv', ['build', '--grading-image', grading,
                                    '--verify-image', 'reviewed:runtime',
                                    '--bound-homepage-legal-alias'])
    monkeypatch.setattr(builder, 'checked_output', lambda *args, **kwargs: 'b' * 40)
    monkeypatch.setattr(builder, 'image_inspect', lambda ref: base if ref == grading else candidate)
    def extract(ref, destination, **kwargs):
        seen.update(kwargs)
        return sources
    monkeypatch.setattr(builder, 'extract_frozen_sources', extract)
    monkeypatch.setattr(builder, 'verify_saved_overlay', lambda ref, digest, files: seen.update(files=files))
    assert builder.main() == 0
    receipt = json.loads(capsys.readouterr().out)
    assert seen['bound_homepage_legal_alias'] is True
    assert receipt['grading_corrections'] == [seam.CORRECTION_ID]
    assert receipt['base_layers_unchanged'] is True
    assert receipt['process_configuration_unchanged'] is True
    assert set(receipt['original_files']) == set(sources)
    for name, value in repaired_sources().items():
        assert seen['files'][name] == value


def test_extraction_includes_only_required_frozen_files(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(builder, 'checked_output', lambda *args, **kwargs: 'container')
    def run(command, **kwargs):
        calls.append(command)
        if command[:2] == ['docker', 'cp']:
            Path(command[-1]).write_bytes(b'original')
    monkeypatch.setattr(builder.subprocess, 'run', run)
    result = builder.extract_frozen_sources('frozen:grader', tmp_path,
                                            bound_homepage_legal_alias=True)
    assert set(result) == set(seam.DELTAS) | set(builder.ENTRYPOINT_FILES)
    assert calls[-1] == ['docker', 'rm', 'container']
    assert len([c for c in calls if c[:2] == ['docker', 'cp']]) == len(result)


def test_known_modules_from_different_grading_revisions_cannot_mix():
    sources = original_sources()
    path = builder.GRADER_FILE
    text = sources[path].decode()
    # Reconstruct only the exact known frozen investigator; keep the default
    # lead scorer to model a cross-revision bundle, not an unknown module.
    new = '''    # Preserve the existing literal field parser's supported local languages,
    # after rejecting department/office scope. The added heading carries no fact.
    localized = extract_linkedin_company_size("## About us\\n" + quote)
'''
    assert text.count(new) == 1
    text = text.replace(new, '', 1)
    new = '''        range_pattern = re.escape(compact_token).replace(r"\\-", r"\\s*-\\s*")
        token_pattern = rf"(?<![\\d-]){range_pattern}(?![\\d+]|-\\d)"
'''
    assert text.count(new) == 1
    text = text.replace(new, '        token_pattern = re.escape(compact_token)\n', 1)
    new = '''    if localized and localized["employee_count"] == observed_value:
        return True
'''
    assert text.count(new) == 1
    sources[path] = text.replace(new, '', 1).encode()
    assert hashlib.sha256(sources[path]).hexdigest() in seam.ORIGINAL_TO_REPAIRED_SHA256[path]
    with pytest.raises(seam.BoundHomepageAliasSeamError, match='bundle'):
        seam.convert_bound_homepage_legal_alias_files(sources)
