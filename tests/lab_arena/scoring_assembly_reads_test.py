"""Assembly reads only needed outputs and revalidates evidence each invocation."""
from copy import deepcopy
from datetime import datetime, timezone
import json

import pytest

from lab_arena import contracts, scoring, verify
from lab_arena.service import ServiceError
from tests.lab_arena.public_result_company_evidence_reads_test import _public_company_result
from tests.lab_arena.legacy_company_judgments_test import _input
from tests.lab_arena.scoring_validator_attribution_test import ROUND_ID, SUBMISSION_ID, MINER


def assembly(*, extras=0):
    service, ref, lease, cached, authority, objects, reads = _public_company_result()
    row = service._round(ROUND_ID)
    row.update(status='stage2_judged', scoring_generation_floor=0,
               participants=[{'submission_id': SUBMISSION_ID, 'miner_hotkey': MINER, 'is_king': False},
                             {'submission_id': 'baseline', 'miner_hotkey': 'baseline', 'is_king': True}])
    row['configuration_doc'].update(
        execution_sequence_policy=contracts.BASELINE_SCORED_FIRST_POLICY,
        stage_1_icp_count=1, stage_2_icp_count=1,
        scorer_image_digest=ref['scope_doc']['scorer_image_digest'],
        scorer_image_reference=ref['scope_doc']['scorer_image_reference'],
        schedule={'publication_deadline': '2099-01-01T00:00:00Z',
                  'final_scoring_close': '2098-12-31T23:59:00Z'})
    service._clock = lambda: datetime(2026, 11, 6, tzinfo=timezone.utc)
    service.evaluation_icps = lambda _round_id: [_input()['icp']] * 2
    for run in service._store.rows.values():
        run.update(stage=2, scoring_generation_floor=0, miner_hotkey=MINER)
        if run['status'] == 'accepted': run['terminal_cause'] = 'accepted'
    service._store.rows['execute-current']['per_icp_score'] = None
    service._store.rows['execute-source']['output_ref'] = 'authority-execution-output'
    objects['authority-execution-output'] = deepcopy(objects['execution-output'])
    for number in range(extras):
        source = deepcopy(service._store.rows['execute-source'])
        source.update(run_id='already-scored-' + str(number), submission_id='already-' + str(number),
                      output_ref='already-output-' + str(number))
        service._store.rows[source['run_id']] = source
        objects[source['output_ref']] = deepcopy(objects['execution-output'])
    service._objects.get = lambda name: service._objects.get_bounded(name, 1024 * 1024)
    writes = []
    service._store.record_status = 'ok'
    def record(round_id, stage, scores, *, scoring_generation_floor):
        assert round_id == ROUND_ID and stage == 2
        assert scoring_generation_floor == row['scoring_generation_floor']
        if service._store.record_status != 'ok':
            return {'status': service._store.record_status}
        writes.extend(deepcopy(scores))
        for score in scores:
            service._store.rows[score['run_id']].update(deepcopy(score))
        return {'status': 'ok'}
    service._store.record_run_scores = record
    def transition(round_id, before, after, update):
        assert round_id == ROUND_ID and row['status'] == before
        row.update(status=after, **update)
        return {'status': 'ok'}
    service._store.transition_round = transition
    def cancel(round_id, reason):
        assert round_id == ROUND_ID
        return {'status': 'cancelled', 'reason': reason}
    service._store.cancel_round = cancel
    refresh_plan(service, row)
    return service, row, objects, cached, authority, reads, writes


def refresh_plan(service, row):
    sources = [service._store.rows['execute-current']]
    if 'execute-next' in service._store.rows: sources.append(service._store.rows['execute-next'])
    row['stage2_scoring_plan_doc'] = scoring.build_scoring_plan(
        round_id=ROUND_ID, stage=2, runs=sources,
        execution_sequence_policy=contracts.BASELINE_SCORED_FIRST_POLICY,
        configuration=row['configuration_doc'])


def add_ready(service, row, objects):
    source = deepcopy(service._store.rows['execute-current'])
    source.update(run_id='execute-next', icp_position=1, output_ref='execution-next', per_icp_score=None)
    source.pop('qualification_doc', None)
    judge = deepcopy(service._store.rows['score-current'])
    judge.update(run_id='score-next', scored_run_id='execute-next', icp_position=1, output_ref='score-next-output')
    service._store.rows[source['run_id']] = source
    service._store.rows[judge['run_id']] = judge
    objects['execution-next'] = deepcopy(objects['execution-output'])
    objects['score-next-output'] = deepcopy(objects['current-output'])
    objects['score-next-output']['scored_run_id'] = source['run_id']
    refresh_plan(service, row)


def test_incomplete_stage_wait_reads_no_output_objects():
    service, _, _, _, _, reads, writes = assembly(extras=211)
    service._store.rows['score-current'].update(status='pending', terminal_cause=None)
    result = service.score_stage(ROUND_ID, 2)
    assert result['status'] == 'retry' and result['remaining'] == 1
    assert sum(reads.values()) == 0
    assert writes == []


def test_one_recovered_judgment_reads_one_of_213_execution_outputs():
    service, row, _, _, _, reads, writes = assembly(extras=211)
    service._record_recovered_judgment_scores(row)
    assert len(writes) == 1 and writes[0]['run_id'] == 'execute-current'
    assert writes[0]['qualification_doc']['companies'][0]['company_qualified'] is True
    assert reads['execution-output'] == 1
    assert reads['authority-execution-output'] == 0
    assert not any(reads['already-output-' + str(i)] for i in range(211))
    assert reads['source-output'] == 1


@pytest.mark.parametrize('entry', ['stage', 'recovered'])
def test_shared_company_authority_is_read_once_and_scores_match_stateless(entry):
    service, row, objects, _, _, reads, writes = assembly()
    add_ready(service, row, objects)
    expected = {}
    for name in ('score-current', 'score-next'):
        judge = service._store.rows[name]
        expected[judge['scored_run_id']] = service._verified_breakdowns(
            judge, icp=_input()['icp'], companies=objects['execution-output']['companies'],
            policy=row['configuration_doc']['scorer_policy'])
    expected_rows = [verify.scored_row(SUBMISSION_ID, service._store.rows[rid]['icp_position'],
        rid, _input()['icp'], objects['execution-output']['companies'], breakdowns,
        row['configuration_doc']['scorer_policy']) for rid, breakdowns in expected.items()]
    expected_records = scoring.run_scores_for_store(
        {'rows': expected_rows, 'integrity_policy': 'arena_integrity_v1'},
        [service._store.rows[rid] for rid in expected])
    assert reads['source-output'] == 4
    reads.clear()
    if entry == 'stage': assert service.score_stage(ROUND_ID, 2)['status'] == 'ok'
    else: service._record_recovered_judgment_scores(row)
    assert len(writes) == 2 and reads['source-output'] == 1 and reads['company_rows'] == 1
    assert contracts.canonical_json(writes) == contracts.canonical_json(expected_records)


@pytest.mark.parametrize('change', ['authority', 'floor'])
def test_next_invocation_uses_fresh_authority_and_scoring_floor(change):
    service, row, objects, _, authority, reads, writes = assembly()
    service._record_recovered_judgment_scores(row)
    assert len(writes) == 1
    add_ready(service, row, objects)
    reads.clear()
    if change == 'authority':
        authority['runner_hotkey'] = MINER
        with pytest.raises(scoring.ScoringError, match='authority is invalid'):
            service._record_recovered_judgment_scores(row)
        assert reads['company_rows'] == 1
    else:
        row['scoring_generation_floor'] = 1
        service._record_recovered_judgment_scores(row)
        assert sum(reads.values()) == 0
    assert len(writes) == 1 and service._store.rows['execute-next']['per_icp_score'] is None


def test_recovered_score_write_refusal_still_fails_closed():
    service, row, _, _, _, _, writes = assembly()
    service._store.record_status = 'stale'
    with pytest.raises(ServiceError, match='recovered_judgment_scores_not_recorded'):
        service._record_recovered_judgment_scores(row)
    assert writes == [] and service._store.rows['execute-current']['per_icp_score'] is None


@pytest.mark.parametrize('entry', ['stage', 'recovered'])
def test_corrupt_company_evidence_still_refuses_all_score_writes(entry):
    service, row, _, cached, _, _, writes = assembly()
    cached['evidence_hash'] = 'sha256:' + 'f' * 64
    if entry == 'stage':
        assert service.score_stage(ROUND_ID, 2)['status'] == 'cancelled'
    else:
        with pytest.raises(scoring.ScoringError, match='evidence is invalid'):
            service._record_recovered_judgment_scores(row)
    assert writes == [] and service._store.rows['execute-current']['per_icp_score'] is None
