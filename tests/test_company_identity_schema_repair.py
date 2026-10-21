"""An unrelated completed semantic review cannot skip identity schema repair."""
import asyncio
import copy

import pytest

from qualification.scoring import lead_scorer as scorer
from qualification.scoring.company_fit_decision import COMPANY_FIT_MATCH, COMPANY_FIT_UNAVAILABLE
from tests.test_company_evidence_investigator import _company, _icp, _complete_verdict, _finding


def exercise(monkeypatch, *, missing=True, repaired='valid', attribute=False,
             completed=True, source_fault=False, rebrand=False, receipt_shape='valid',
             preserve_proven=False, real_projection=False):
    company = _company()
    icp = _icp(required_attribute='Supplies software.' if attribute else '',
               company_stage='Public' if preserve_proven else '')
    initial = _complete_verdict()
    if missing:
        initial.update(observed_company_website='', observed_company_linkedin='')
    repair = _complete_verdict()
    if repaired == 'missing':
        repair.update(observed_company_website='', observed_company_linkedin='')
    elif repaired == 'other_entity':
        repair.update(observed_company_name='Other', observed_company_website='https://other.example',
                      observed_company_linkedin='https://www.linkedin.com/company/other')
    elif repaired == 'conflicting_slug':
        repair['observed_company_linkedin'] = 'https://www.linkedin.com/company/other'
    if preserve_proven:
        for value in (initial, repair):
            value.update(observed_company_stage='Public', stage_matches=True,
                         stage_evidence_url='https://acme.example/investors',
                         stage_evidence_quote='Acme common stock is listed on NASDAQ under ticker ACME.')
    targets = ('industry',) + (('required_attribute',) if attribute else ())
    if preserve_proven:
        targets += ('stage',)
    if rebrand:
        targets += ('rebrand',)
    calls = []
    investigations = []
    claims = {t: _finding(t, status='UNPROVEN', observed_value=None,
                          evidence_url='', evidence_quote='') for t in targets}
    if preserve_proven:
        claims['stage'] = _finding('stage')
        claims['required_attribute'] = _finding('required_attribute',
            evidence_url='https://acme.example/platform',
            evidence_quote='Acme supplies software.')
    receipt = {
        'gate': 'company_evidence_investigation', 'targets': list(targets),
        'claims': claims, 'completed_submitted_findings': completed,
        'failure_reason': '' if completed else 'provider_error',
        'usage': {'fetch_outcomes': [{'url_sha256': 'a' * 64,
            'ok': not source_fault, 'loaded_text_length': 0 if source_fault else 1000,
            'error_class': 'http_error' if source_fault else ''}]},
    }

    async def provider(**kwargs):
        calls.append(kwargs['telemetry_purpose'])
        if len(calls) == 1:
            return copy.deepcopy(initial), ''
        if repaired == 'provider_error':
            kwargs['diagnostic']['failure_reason_code'] = 'provider_error'
            return None, 'provider_error'
        return copy.deepcopy(repair), ''

    async def refresh(verdict, *_args, **_kwargs):
        return verdict

    async def ground(verdict, **_kwargs):
        verdict = dict(verdict)
        if attribute:
            verdict[scorer._REQUIRED_ATTRIBUTE_GROUNDING] = {'status': 'grounded' if preserve_proven else 'unproven'}
        return verdict, None

    async def investigate(*, verdict, **kwargs):
        investigations.append(kwargs['investigation_targets'])
        verdict = scorer._clear_reopened_positive_semantics(verdict, active_attribute=attribute)
        if attribute:
            verdict[scorer._REQUIRED_ATTRIBUTE_GROUNDING] = {'status': 'grounded' if preserve_proven else 'unproven'}
        if preserve_proven:
            verdict = scorer._project_investigator_required_attribute(verdict, claims['required_attribute'])
            verdict[scorer._REQUIRED_ATTRIBUTE_GROUNDING] = {'status': 'grounded'}
        result = scorer._reverify_decision(verdict, icp.required_attribute, icp.company_stage,
                                          icp=icp, company=company, company_quality=True,
                                          validated_stage_finding=claims.get('stage') if preserve_proven else None)
        result.details['investigation_receipt'] = (copy.deepcopy(receipt) if receipt_shape == 'valid'
                                                  else None if receipt_shape == 'missing' else [])
        return verdict, result, claims, {}, claims.get('stage', {}) if preserve_proven else {}

    monkeypatch.setenv('OPENROUTER_API_KEY', 'test-key')
    monkeypatch.setattr(scorer, '_request_company_reverify_json', provider)
    monkeypatch.setattr(scorer, '_refresh_linkedin_employee_size_observation', refresh)
    monkeypatch.setattr(scorer, '_ground_required_attribute_evidence', ground)
    monkeypatch.setattr(scorer, '_targeted_company_investigation_dimensions', lambda *_a, **_k: targets)
    monkeypatch.setattr(scorer, '_positive_semantic_review_needed', lambda *_a, **_k: True)
    if real_projection:
        async def bounded_investigator(**kwargs):
            investigations.append(kwargs['targets'])
            return {'claims': copy.deepcopy(claims), '_completed_submit': True,
                    'failure_reason': '', 'usage': copy.deepcopy(receipt['usage'])}
        monkeypatch.setattr(scorer, 'investigate_company_evidence', bounded_investigator)
    else:
        monkeypatch.setattr(scorer, '_run_targeted_company_evidence_investigation', investigate)
    result = asyncio.run(scorer._llm_reverify_company(
        company, icp, require_company_fit_dimensions=True,
        company_quality=True, evidence_investigator=True,
    ))
    return result, calls, investigations


@pytest.mark.parametrize('attribute', [False, True])
def test_missing_identity_uses_existing_one_repair_without_promoting_unproven(monkeypatch, attribute):
    result, calls, investigations = exercise(monkeypatch, attribute=attribute)
    assert calls == ['lead_scorer_reverify', 'lead_scorer_reverify_schema_repair']
    assert len(investigations) == 1
    assert result.details['identity_decision'] == COMPANY_FIT_MATCH
    assert result.decision == COMPANY_FIT_UNAVAILABLE
    assert result.details['dimension_decisions']['industry'] == COMPANY_FIT_UNAVAILABLE
    assert result.details['failure_class'] == 'insufficient_fit_evidence'


def test_investigated_unproven_without_identity_gap_does_not_add_call(monkeypatch):
    result, calls, _ = exercise(monkeypatch, missing=False)
    assert calls == ['lead_scorer_reverify']
    assert result.decision == COMPANY_FIT_UNAVAILABLE


@pytest.mark.parametrize('repair', ['missing', 'other_entity', 'conflicting_slug', 'provider_error'])
def test_identity_repair_failure_cannot_qualify(monkeypatch, repair):
    result, calls, _ = exercise(monkeypatch, repaired=repair)
    assert calls == ['lead_scorer_reverify', 'lead_scorer_reverify_schema_repair']
    assert result.decision != COMPANY_FIT_MATCH
    assert result.details.get('failure_class') != 'insufficient_fit_evidence'


def test_incomplete_research_does_not_get_identity_repair(monkeypatch):
    result, calls, _ = exercise(monkeypatch, completed=False)
    assert calls == ['lead_scorer_reverify']
    assert result.decision == COMPANY_FIT_UNAVAILABLE


def test_repaired_identity_cannot_clear_unresolved_source_fault(monkeypatch):
    result, calls, _ = exercise(monkeypatch, source_fault=True)
    assert len(calls) <= 2
    assert result.decision != COMPANY_FIT_MATCH
    assert result.details.get('failure_class') != 'insufficient_fit_evidence'


def test_investigated_identity_keeps_existing_target_bound(monkeypatch):
    result, calls, _ = exercise(monkeypatch, rebrand=True)
    assert calls == ['lead_scorer_reverify']
    assert result.decision == COMPANY_FIT_UNAVAILABLE


@pytest.mark.parametrize('receipt_shape', ['missing', 'malformed'])
def test_missing_or_malformed_receipt_does_not_open_repair(monkeypatch, receipt_shape):
    result, calls, _ = exercise(monkeypatch, receipt_shape=receipt_shape)
    assert calls == ['lead_scorer_reverify']
    assert result.decision != COMPANY_FIT_MATCH


def test_identity_repair_preserves_independently_proven_attribute_and_stage(monkeypatch):
    result, calls, _ = exercise(monkeypatch, attribute=True, preserve_proven=True)
    assert calls == ['lead_scorer_reverify', 'lead_scorer_reverify_schema_repair']
    assert result.details['identity_decision'] == COMPANY_FIT_MATCH
    assert result.details['dimension_decisions']['industry'] == COMPANY_FIT_UNAVAILABLE
    assert result.details['dimension_decisions']['stage'] == COMPANY_FIT_MATCH
    assert result.details['required_attribute_decision'] == COMPANY_FIT_MATCH
    assert result.decision == COMPANY_FIT_UNAVAILABLE


def test_real_targeted_projection_reaches_existing_identity_repair(monkeypatch):
    result, calls, investigations = exercise(monkeypatch, real_projection=True)
    assert calls == ['lead_scorer_reverify', 'lead_scorer_reverify_schema_repair']
    assert len(investigations) == 1
    assert result.details['identity_decision'] == COMPANY_FIT_MATCH
    assert result.details['dimension_decisions']['industry'] == COMPANY_FIT_UNAVAILABLE
    assert result.details['failure_class'] == 'insufficient_fit_evidence'
