"""Future-round flat sourcing costs preserve frozen scoring and legacy rules."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from types import SimpleNamespace

import pytest

from lab_arena import contracts, scoring, verify
from lab_arena.service import ArenaService, RoundDefaults
from lab_arena.training_export import assemble_example
from tests.lab_arena.deepline_round_runtime_test import _service
from tests.lab_arena.test_deepline_catalog import frozen
from tests.lab_arena.test_lab_arena_verify import _breakdown, _icp, _junk
from tests.lab_arena.training_export_test import per_icp_cost_fixture, per_icp_cost_reader
from tests.lab_arena.scorer_environment_fixture import isolated_scorer_environment


FLAT = contracts.FLAT_PER_ICP_SUCCESSFUL_CALLS_COST_POLICY
LEGACY = contracts.PER_ICP_SUCCESSFUL_CALLS_COST_POLICY


def test_flat_round_default_is_explicit_and_freezes_fixed_limits_and_catalog():
    assert RoundDefaults().flat_per_icp_cost_policy is False
    service = _service(lambda **_: frozen())
    service.config.defaults.flat_per_icp_cost_policy = True
    service.config.defaults.per_icp_cost_policy = False
    # A caller cannot accidentally carry the old per-qualified-company value
    # into the new policy document.
    assert service.config.defaults.cost_per_company_microusd == 800_000
    config = service.create_round(datetime(2026, 10, 19, tzinfo=timezone.utc))
    assert config['sourcing_cost_eligibility_policy'] == FLAT
    assert config['cost_per_company_microusd'] == 4_000_000
    assert config['execution_icp_cap_microusd'] == 4_000_000
    assert config['infrastructure_retry_cost_policy'] == contracts.INFRASTRUCTURE_RETRY_COST_POLICY
    assert config['call_quotas']['deepline'] == 0
    assert contracts.validate_round_configuration(json.loads(json.dumps(config))) == config
    before = deepcopy(config)
    service.config.defaults.flat_per_icp_cost_policy = False
    service.config.defaults.per_icp_cost_policy = True
    service._freeze_deepline_catalog(config)
    assert config == before
    legacy = service.create_round(datetime(2026, 10, 20, tzinfo=timezone.utc))
    assert legacy['sourcing_cost_eligibility_policy'] == LEGACY
    assert legacy['cost_per_company_microusd'] == 800_000


@pytest.mark.parametrize('policy,company_cap', [(FLAT, 4_000_000), (LEGACY, 800_000)])
def test_contract_requires_exact_frozen_limit_for_each_policy(policy, company_cap):
    config = _service().create_round(datetime(2026, 10, 19, tzinfo=timezone.utc))
    config.update(sourcing_cost_eligibility_policy=policy,
                  cost_per_company_microusd=company_cap)
    assert contracts.validate_round_configuration(json.loads(json.dumps(config))) == config
    for field, invalid in [('cost_per_company_microusd', 800_000 if policy == FLAT else 4_000_000),
                           ('execution_icp_cap_microusd', 4_000_001)]:
        changed = dict(config, **{field: invalid})
        with pytest.raises(contracts.ArenaContractError, match='fixed cost limits'):
            contracts.validate_round_configuration(changed)


@pytest.mark.parametrize('policy', [FLAT, LEGACY])
def test_baseline_and_miner_use_authority_decision_without_changing_raw_scores(policy):
    service = object.__new__(ArenaService)
    participants = [dict(submission_id=sid, miner_hotkey=sid, is_king=sid == 'baseline')
                    for sid in ('baseline', 'miner')]
    # These represent 0..5 qualified companies and an overshoot. Zero qualified
    # companies already have zero quality. The authority alone decides costs.
    raw = [0, 12, 24, 36, 48, 60, 60]
    runs = [dict(submission_id=p['submission_id'], icp_position=i, attempt=1,
                 per_icp_score=score, terminal_cause='accepted')
            for p in participants for i, score in enumerate(raw)]
    saved = deepcopy(runs)
    service._store = SimpleNamespace(list_runs=lambda *_args, **_kwargs: runs)
    service._qualified_company_counts = lambda *_args, **_kwargs: {i: min(i, 5) for i in range(7)}
    observed = []
    def cost(_round, sid, position, _runs, *, qualified):
        observed.append((sid, position, qualified))
        return {'eligible': position != 6}
    service._per_icp_cost_eligibility = cost
    round_row = dict(round_id='future-round', participants=participants,
                     configuration_doc={'sourcing_cost_eligibility_policy': policy})
    scores = service._score_entries_from_runs(round_row, range(7), 'final_score')
    assert [row['final_score'] for row in scores] == [sum(raw[:-1]) / 7] * 2
    assert len(observed) == 14
    assert all(('baseline', i, min(i, 5)) in observed and ('miner', i, min(i, 5)) in observed
               for i in range(7))
    assert runs == saved


def test_flat_cost_does_not_remove_filler_penalty_or_qualify_empty_results():
    buyer = _icp()
    policy = scoring.build_scorer_policy(scoring_adapter_version='qualification_integrity_v2')
    good = [dict(_breakdown(60), company_qualified=True) for _ in range(3)]
    clean = verify.per_icp_score(buyer, good, policy)
    with_filler = verify.per_icp_score(buyer, good + [_junk()], policy)
    assert with_filler['per_icp_score'] < clean['per_icp_score']
    assert with_filler['fp_gate_count'] == 1
    assert verify.per_icp_score(buyer, [], policy)['per_icp_score'] == 0


@pytest.mark.parametrize('spend,reason,eligible', [
    (4_000_000, 'eligible', True),
    (4_000_001, 'execution_cap_exceeded', False),
])
def test_flat_training_snapshot_uses_fixed_cap_and_keeps_raw_quality(spend, reason, eligible):
    values = per_icp_cost_fixture()
    values['round_row']['configuration_doc'].update(
        sourcing_cost_eligibility_policy=FLAT, cost_per_company_microusd=4_000_000)
    base = per_icp_cost_reader(spend=spend, reason=reason)
    def reader(**identity):
        result = base(**identity)
        result['eligibility'].update(cost_per_company_cap_microusd=4_000_000,
                                     eligibility_cap_microusd=4_000_000)
        return result
    result = assemble_example(**values, cost_eligibility_reader=reader)
    cost = result['competition_cost_eligibility']
    assert cost['status'] == 'available'
    assert cost['policy'] == FLAT
    assert cost['eligible'] is eligible
    assert cost['eligibility_reason'] == reason
    assert cost['eligibility_cap_microusd'] == 4_000_000
    assert result['judgment']['per_icp_score'] == 12.75


@pytest.mark.parametrize('wrong_cap,wrong_reason', [
    (800_000, 'cost_per_company_exceeded'),
    (20_000_000, 'eligible'),
    (4_000_000, 'cost_per_company_exceeded'),
])
def test_flat_training_export_rejects_legacy_or_multiplied_cost_authority(wrong_cap, wrong_reason):
    values = per_icp_cost_fixture()
    values['round_row']['configuration_doc'].update(
        sourcing_cost_eligibility_policy=FLAT, cost_per_company_microusd=4_000_000)
    base = per_icp_cost_reader(spend=4_000_001, reason=wrong_reason)
    def reader(**identity):
        result = base(**identity)
        result['eligibility'].update(cost_per_company_cap_microusd=4_000_000,
                                     eligibility_cap_microusd=wrong_cap)
        return result
    result = assemble_example(**values, cost_eligibility_reader=reader)
    assert result['competition_cost_eligibility']['status'] == 'unavailable'


def test_frozen_judge_input_and_contract_do_not_receive_sourcing_policy(monkeypatch):
    service = _service()
    cutoff = datetime(2026, 10, 19, tzinfo=timezone.utc)
    legacy = service.create_round(cutoff)
    service.config.defaults.flat_per_icp_cost_policy = True
    flat = service.create_round(cutoff)
    def judge_input(config):
        return scoring.build_scoring_input(
            scored_run_id='same-run', icp=_icp(), companies=[],
            policy=config['scorer_policy'], evaluation_date='2026-10-19')
    assert judge_input(legacy) == judge_input(flat)
    assert 'sourcing_cost_eligibility_policy' not in json.dumps(judge_input(flat))
    # A frozen judge may have the old round validator. Its runtime uses only
    # the frozen scoring policy, so it need not accept the host's new marker.
    from lab_arena import scorer_entrypoint
    from unittest.mock import patch
    import os
    def reject_new_round(_document):
        raise AssertionError('round configuration reached frozen judge')
    monkeypatch.setattr(contracts, 'validate_round_configuration', reject_new_round)
    monkeypatch.setattr(scoring, 'lab_scorer', lambda *_args, **_kwargs: lambda *_args: [])
    with patch.dict(os.environ):
        result = scorer_entrypoint.score_input(judge_input(flat))
    assert result['breakdowns'] == []
    assert 'failure' not in result


@pytest.fixture
def current_flat_cost_database():
    from tests.lab_arena.flat_per_icp_cost500_postgres_test import database_with_flat_cost_migration
    yield from database_with_flat_cost_migration()


@pytest.mark.parametrize('flat', [False, True])
def test_current_complete_baseline_publishes_flat_and_legacy_cost_policy(current_flat_cost_database, tmp_path, monkeypatch, flat):
    from tests.lab_arena.complete_baseline_final_authority_postgres_test import (
        test_normal_complete_baseline_publishes_and_activates_with_all_guards as prove_complete_baseline,
    )
    from tests.lab_arena.test_integrity_round import IntegrityHarness
    original_build = IntegrityHarness.build_service
    def build_service(harness):
        service = original_build(harness)
        service.config.defaults.flat_per_icp_cost_policy = flat
        return service
    monkeypatch.setattr(IntegrityHarness, 'build_service', build_service)
    # Reuse the known current baseline shape, all final guards and disabled
    # reward control. Boundary spend and retry behavior have separate SQL tests.
    prove_complete_baseline(current_flat_cost_database, tmp_path, monkeypatch, False, False, 15)
    psycopg2, dsn = current_flat_cost_database
    with psycopg2.connect(**dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT configuration_doc,publication_doc FROM public.lab_arena_rounds WHERE status='published'")
        config, publication = cur.fetchone()
        policy = FLAT if flat else LEGACY
        assert config['sourcing_cost_eligibility_policy'] == policy
        result = next(row for row in publication['final_ranking'] if row['is_baseline'])
        summary = result['cost_summary']
        assert summary['sourcing_cost_eligibility_policy'] == policy
        assert summary['cost_per_company_cap_microusd'] == (4_000_000 if flat else 800_000)
        assert len(summary['per_icp']) == 10
        assert all(row['eligible'] for row in summary['per_icp'])
        forged_documents = []
        forged = deepcopy(result)
        forged['cost_summary']['sourcing_cost_eligibility_policy'] = LEGACY if flat else FLAT
        forged_documents.append(forged)
        for key in ('eligible_icp_count', 'competition_sourcing_microusd'):
            forged = deepcopy(result)
            forged['cost_summary'][key] += 1
            forged_documents.append(forged)
        forged = deepcopy(result)
        forged['cost_summary']['per_icp'][0]['eligibility_cap_microusd'] += 1
        forged_documents.append(forged)
        for forged in forged_documents:
            with pytest.raises(psycopg2.Error) as rejected:
                cur.execute("SELECT public.lab_arena__per_icp_publication_valid(%s,%s::jsonb)",
                            (config['round_id'], json.dumps(forged)))
            assert rejected.value.pgcode == '22023'
            conn.rollback()
        cur.execute("SELECT public.lab_arena__per_icp_publication_valid(%s,%s::jsonb)",
                    (config['round_id'], json.dumps(result)))
        assert cur.fetchone()[0] is True
