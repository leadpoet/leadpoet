"""Compose optional visible contexts after exact2dc identity finality only."""
import ast
import copy
import hashlib
import json
import sys
from types import ModuleType

import pytest

from scripts import arena_reviewer_identity_repair_finality_seam as identity
from scripts import arena_reviewer_visible_auxiliary_context_seam as seam
from scripts import build_arena_reviewer_runtime as composer
from tests import test_intent_details_fetched_first_party_context as sources
from tests import test_visible_auxiliary_context as controls
from tests.reviewer_frozen_sources import frozen_source

ORIGINAL = frozen_source('identity_repair_finality_2dc', identity.FINALITY_FILE)
IDENTITY = identity.convert_identity_repair_finality_seam(ORIGINAL)
CORRECTED = seam.convert_visible_auxiliary_context_seam(IDENTITY)


def test_exact_two_function_correction_preserves_all_other_frozen_source():
    assert hashlib.sha256(IDENTITY).hexdigest() == seam.ORIGINAL_SHA256
    assert hashlib.sha256(CORRECTED).hexdigest() == seam.REPAIRED_SHA256
    assert seam.convert_visible_auxiliary_context_seam(CORRECTED) == CORRECTED
    restored = CORRECTED.decode()
    for before, after in reversed(seam._DELTAS):
        assert restored.count(after) == 1
        restored = restored.replace(after, before, 1)
    assert restored.encode() == IDENTITY
    before, after = ast.parse(IDENTITY), ast.parse(CORRECTED)
    for tree in (before, after):
        tree.body = [node for node in tree.body if getattr(node,'name','') not in {
            '_validated_first_party_source_context', '_retain_matched_investigator_source_contexts',
        }]
    assert ast.dump(before) == ast.dump(after)


@pytest.mark.parametrize('source', [
    b'', b'\xff', ORIGINAL, IDENTITY+b'\n', CORRECTED+b'\n',
    IDENTITY.replace(b'\n',b'\r\n'),
    IDENTITY.replace(seam._DELTAS[0][0].encode(),seam._DELTAS[0][1].encode(),1),
])
def test_unknown_partial_or_wrong_generation_source_is_rejected(source):
    with pytest.raises(seam.VisibleAuxiliaryContextSeamError):
        seam.convert_visible_auxiliary_context_seam(source)


@pytest.mark.parametrize('verify_existing', [False, True])
def test_composer_combines_both_repairs_in_one_exact_file(monkeypatch,capsys,verify_existing):
    base_ref = 'registry.example/reviewer@'+identity.FROZEN_IMAGE_DIGEST
    tag = composer.bounded_build_tag('registry.example/reviewer:reviewer-'+identity.FROZEN_IMAGE_DIGEST[7:19]
        +'-'+'c'*12+'-identity-repair-finality-visible-auxiliary-context')
    runtime = 'registry.example/reviewer@sha256:'+'b'*64 if verify_existing else tag
    args = ['build_arena_reviewer_runtime.py','--grading-image',base_ref,
            '--identity-repair-finality-only','--visible-auxiliary-context']
    if verify_existing:
        args += ['--verify-image',runtime]
    monkeypatch.setattr(composer.sys,'argv',args)
    monkeypatch.setattr(composer,'checked_output',lambda *_a,**_k:'c'*40)
    base = {'Architecture':'amd64','Os':'linux','Config':{},'RootFS':{'Layers':['sha256:base']}}
    candidate = copy.deepcopy(base)
    candidate['RootFS']['Layers'].append('sha256:overlay')
    monkeypatch.setattr(composer,'image_inspect',lambda ref:base if ref==base_ref else candidate)
    def extract(ref,_destination,**flags):
        assert ref==base_ref and flags=={'identity_repair_finality_only':True}
        return {identity.FINALITY_FILE:ORIGINAL}
    def verify(ref,digest,files):
        assert ref==runtime and digest=='sha256:overlay'
        assert files=={identity.FINALITY_FILE:CORRECTED}
    monkeypatch.setattr(composer,'extract_frozen_sources',extract)
    monkeypatch.setattr(composer,'verify_saved_overlay',verify)
    builds = []
    monkeypatch.setattr(composer.subprocess,'run',lambda command,**_k:builds.append(command))
    assert composer.main()==0
    proof = json.loads(capsys.readouterr().out)
    assert proof['grading_corrections']==[identity.CORRECTION_ID,seam.CORRECTION_ID]
    assert proof['overlay_files']=={identity.FINALITY_FILE:seam.REPAIRED_SHA256}
    assert proof['base_layers_unchanged'] is proof['process_configuration_unchanged'] is True
    assert len(builds)==(0 if verify_existing else 1)


@pytest.mark.parametrize('options', [[],['--shim-only'],['--identity-schema-repair']])
def test_suboption_requires_exclusive_exact_identity_mode_before_docker(monkeypatch,options):
    monkeypatch.setattr(composer.sys,'argv',['build_arena_reviewer_runtime.py','--grading-image',
        'registry.example/reviewer@'+identity.FROZEN_IMAGE_DIGEST,'--visible-auxiliary-context',*options])
    with pytest.raises(composer.CompositionError,match='requires exact2dc'):
        composer.main()


def test_suboption_cannot_mix_other_corrections(monkeypatch):
    monkeypatch.setattr(composer.sys,'argv',['build_arena_reviewer_runtime.py','--grading-image',
        'registry.example/reviewer@'+identity.FROZEN_IMAGE_DIGEST,'--visible-auxiliary-context',
        '--identity-repair-finality-only','--shim-only'])
    with pytest.raises(composer.CompositionError,match='cannot include other corrections'):
        composer.main()


def _install(monkeypatch,source):
    module = ModuleType('_exact_frozen_2dc_visible_auxiliary')
    monkeypatch.setitem(sys.modules,module.__name__,module)
    exec(compile(source,'exact_frozen_2dc_visible_auxiliary.py','exec'),vars(module))
    monkeypatch.setattr(controls,'lead_scorer',module)
    monkeypatch.setattr(sources,'lead_scorer',module)


def test_full_frozen_source_reproduces_then_corrects_optional_rejection(monkeypatch):
    _install(monkeypatch,IDENTITY)
    with pytest.raises(AssertionError):
        controls.test_locator_only_company_name_cannot_break_the_real_paragraph_review()
    _install(monkeypatch,CORRECTED)
    controls.test_locator_only_company_name_cannot_break_the_real_paragraph_review()
    controls.test_locator_terms_cannot_win_the_single_optional_retention_slot()


@pytest.mark.parametrize('with_metadata',[False,True])
def test_full_frozen_source_preserves_visible_and_mandatory_controls(monkeypatch,with_metadata):
    _install(monkeypatch,CORRECTED)
    controls.test_visible_optional_context_survives_without_modifying_mandatory_text(with_metadata)
