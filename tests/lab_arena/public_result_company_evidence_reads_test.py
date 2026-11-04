"""One public response may reuse successful immutable authority validation."""
from collections import Counter
from copy import deepcopy
import json

import pytest

from lab_arena import company_judgments, contracts, integrity, scoring
from lab_arena.service import ServiceError, _PublicResultEvidenceReads
from qualification.scoring.competition import raw_company_judgment
from tests.lab_arena.company_judgments_test import _miss
from tests.lab_arena.legacy_company_judgments_test import _input
from tests.lab_arena.scoring_validator_attribution_test import (
    MINER, ROUND_ID, SUBMISSION_ID, VALIDATOR_A, _execution, _score, _service,
)
from tests.test_arena_score_integrity import _positive_breakdown


def _public_company_result():
    document = _input(('A',))
    ref = company_judgments.build_company_scopes(
        scoring_input=document, round_id=ROUND_ID, network_name='finney', netuid=71,
        scorer_image_digest='sha256:' + 'a' * 64,
        scorer_image_reference='registry/scorer@sha256:' + 'a' * 64,
        integrity_policy=integrity.POLICY,
        company_judgment_policy=company_judgments.POLICY,
    )[0]
    raw = raw_company_judgment(scoring.project_company_judgments(
        document['companies'], [_positive_breakdown('A', 'a.example', 'a')],
        company_quality=False, scored_indexes=[0],
    )[0])
    raw['company_judgment_policy'] = company_judgments.POLICY
    miss = _miss(ref)
    source_output = scoring.build_scoring_failure(
        'execute-source', 'judge_error', company_judgments=[{**miss, 'raw_judgment': raw}],
    )
    evidence = company_judgments.build_evidence_snapshot(
        new_judgment={**miss, 'raw_judgment': raw}, company_ref=ref,
        source_score_run_id='score-source', source_scored_run_id='execute-source',
        source_output_ref='source-output', source_output_hash=contracts.document_hash(source_output),
        source_runner_hotkey=VALIDATOR_A, source_claim_request_id='c' * 32,
        source_claim_request_hash='sha256:' + 'd' * 64, source_lease_generation=1,
        source_completion_request_hash='sha256:' + 'e' * 64,
        runner_authority_exclusions=[VALIDATOR_A],
    )
    source_execution = _execution('execute-source', 0, submission_id='source-submission')
    source_score = _score('score-source', source_execution, VALIDATOR_A, status='failed',
        terminal_cause='judge_error', output_ref='source-output',
        claim_request_id=evidence['source_claim_request_id'],
        claim_request_hash=evidence['source_claim_request_hash'], lease_generation=1,
        claim_response={
            'runner_authority_exclusions': [VALIDATOR_A],
            'company_judgment_cache': {'schema_version': company_judgments.LEASE_SCHEMA_VERSION,
                                     'hits': [], 'misses': [miss]},
        }, result_doc={'terminal_status': 'judge_error',
                       'company_judgment_completion_request_hash': evidence['source_completion_request_hash']})
    hit = {**miss, 'evidence_hash': contracts.document_hash(evidence), 'evidence_doc': evidence}
    lease = {'schema_version': company_judgments.LEASE_SCHEMA_VERSION, 'hits': [hit], 'misses': []}
    execution = _execution('execute-current', 0)
    execution['output_ref'] = 'execution-output'
    current_score = _score('score-current', execution, VALIDATOR_A,
        company_judgment_refs=[ref], claim_response={'company_judgment_cache': lease},
        output_ref='current-output', result_doc={'company_judgment_consumed_indexes': [0]})
    breakdowns = scoring.project_company_judgments(
        document['companies'], [raw], company_quality=False, contacts_required=False,
        scored_indexes=[0], consumed_indexes=set(),
    )
    current_output = scoring.build_scoring_output('execute-current', breakdowns, company_judgments=[])
    service = _service([execution, current_score, source_execution, source_score], public_positions={0})
    row = service._round(ROUND_ID)
    row['configuration_doc'] = {
        'integrity_policy': integrity.POLICY, 'company_judgment_policy': company_judgments.POLICY,
        'scorer_policy': document['scorer_policy'],
    }
    service._round = lambda _round_id: row
    service._benchmark_icps_from_row = lambda *_a: [document['icp']]
    cached = {'cache_key': ref['cache_key'], 'company_input_hash': ref['company_input_hash'],
        'scope_doc': ref['scope_doc'], 'evidence_hash': contracts.document_hash(evidence),
        'evidence_doc': evidence, **{key: evidence[key] for key in
            ('source_score_run_id', 'source_scored_run_id', 'source_runner_hotkey')}}
    reads = Counter()

    def cache_read(key, slot):
        reads['company_rows'] += 1
        return cached if (key, slot) == (ref['cache_key'], 0) else None

    original_run = service._store.get_run
    def run_read(run_id):
        reads['run_rows'] += 1
        return original_run(run_id)

    objects = {'source-output': source_output, 'current-output': current_output,
               'execution-output': {'schema_version': contracts.OUTPUT_DOCUMENT_SCHEMA_VERSION,
                                    'companies': document['companies']}}
    def get_bounded(name, _limit):
        reads[name] += 1
        return json.dumps(objects[name]).encode()

    service._store.get_company_judgment = cache_read
    service._store.get_run = run_read
    service._objects.get_bounded = get_bounded
    return service, ref, lease, cached, source_score, objects, reads


def _make_current_score_the_authority(service, ref, cached, objects):
    current = service._store.rows['score-current']
    prior = deepcopy(cached['evidence_doc'])
    miss = _miss(ref)
    objects['current-output']['company_judgments'] = [
        {**miss, 'raw_judgment': prior['raw_judgment']}
    ]
    evidence = company_judgments.build_evidence_snapshot(
        new_judgment=objects['current-output']['company_judgments'][0], company_ref=ref,
        source_score_run_id=current['run_id'], source_scored_run_id=current['scored_run_id'],
        source_output_ref=current['output_ref'],
        source_output_hash=contracts.document_hash(objects['current-output']),
        source_runner_hotkey=VALIDATOR_A,
        source_claim_request_id=prior['source_claim_request_id'],
        source_claim_request_hash=prior['source_claim_request_hash'], source_lease_generation=1,
        source_completion_request_hash=prior['source_completion_request_hash'],
        runner_authority_exclusions=[VALIDATOR_A],
    )
    current.update(claim_request_id=evidence['source_claim_request_id'],
                   claim_request_hash=evidence['source_claim_request_hash'], lease_generation=1,
                   claim_response={'runner_authority_exclusions': [VALIDATOR_A],
                       'company_judgment_cache': {'schema_version': company_judgments.LEASE_SCHEMA_VERSION,
                                                'hits': [], 'misses': [miss]}})
    cached.update(evidence_doc=evidence, evidence_hash=contracts.document_hash(evidence),
                  **{key: evidence[key] for key in ('source_score_run_id', 'source_scored_run_id')})


def _five_company_source_proofs(*, accepted_source=False):
    service, _, _, _, source, objects, reads = _public_company_result()
    document = _input(tuple('C' + str(i) for i in range(5)))
    refs = company_judgments.build_company_scopes(
        scoring_input=document, round_id=ROUND_ID, network_name='finney', netuid=71,
        scorer_image_digest='sha256:' + 'a' * 64,
        scorer_image_reference='registry/scorer@sha256:' + 'a' * 64,
        integrity_policy=integrity.POLICY,
        company_judgment_policy=company_judgments.POLICY,
    )
    judgments = []
    for ref, company in zip(refs, document['companies']):
        name = company['company_name']
        projected = scoring.project_company_judgments(
            [company], [_positive_breakdown(name, name.lower() + '.example', name.lower())],
            company_quality=False, scored_indexes=[0],
        )[0]
        raw = raw_company_judgment(projected)
        raw['company_judgment_policy'] = company_judgments.POLICY
        judgments.append({**_miss(ref), 'raw_judgment': raw})
    if accepted_source:
        breakdowns = scoring.project_company_judgments(
            document['companies'], [row['raw_judgment'] for row in judgments],
            company_quality=False, scored_indexes=list(range(5)),
        )
        output = scoring.build_scoring_output(
            'execute-source', breakdowns, company_judgments=judgments,
        )
        source.update(status='accepted', terminal_cause='accepted',
                      result_doc={'terminal_status': 'accepted'})
    else:
        output = scoring.build_scoring_failure(
            'execute-source', 'judge_error', company_judgments=judgments,
        )
    output_hash = contracts.document_hash(output)
    source['claim_response']['company_judgment_cache']['misses'] = [_miss(ref) for ref in refs]
    objects['source-output'] = output
    caches = {}
    for ref, judgment in zip(refs, judgments):
        evidence = company_judgments.build_evidence_snapshot(
            new_judgment=judgment, company_ref=ref,
            source_score_run_id='score-source', source_scored_run_id='execute-source',
            source_output_ref='source-output', source_output_hash=output_hash,
            source_runner_hotkey=VALIDATOR_A, source_claim_request_id='c' * 32,
            source_claim_request_hash='sha256:' + 'd' * 64,
            source_lease_generation=1, source_completion_request_hash='sha256:' + 'e' * 64,
            runner_authority_exclusions=[VALIDATOR_A],
        )
        caches[(ref['cache_key'], 0)] = {
            'cache_key': ref['cache_key'], 'company_input_hash': ref['company_input_hash'],
            'scope_doc': ref['scope_doc'], 'evidence_hash': contracts.document_hash(evidence),
            'evidence_doc': evidence, **{key: evidence[key] for key in
                ('source_score_run_id', 'source_scored_run_id', 'source_runner_hotkey')},
        }

    def cache_read(key, slot):
        reads['company_rows'] += 1
        return caches.get((key, slot))

    service._store.get_company_judgment = cache_read
    return service, refs, caches, source, objects, reads


def _five_company_public_result():
    service, refs, caches, source, objects, reads = _five_company_source_proofs()
    document = _input(tuple('C' + str(i) for i in range(5)))
    row = service._round(ROUND_ID)
    row['configuration_doc']['scorer_policy'] = document['scorer_policy']
    service._round = lambda _round_id: row
    service._benchmark_icps_from_row = lambda *_args: [document['icp']]
    objects['execution-output']['companies'] = document['companies']
    current = service._store.rows['score-current']
    current['company_judgment_refs'] = refs
    current['claim_response']['company_judgment_cache'] = {
        'schema_version': company_judgments.LEASE_SCHEMA_VERSION,
        'hits': [
            {**_miss(ref), 'evidence_hash': caches[(ref['cache_key'], 0)]['evidence_hash'],
             'evidence_doc': caches[(ref['cache_key'], 0)]['evidence_doc']}
            for ref in refs
        ],
        'misses': [],
    }
    raw = [caches[(ref['cache_key'], 0)]['evidence_doc']['raw_judgment'] for ref in refs]
    scored_indexes = scoring.verify.bucket_skip(
        document['icp'], document['companies'],
        max_scored_companies=int(document['scorer_policy']['max_scored_companies']),
    )[0]
    consumed = set()
    breakdowns = scoring.project_company_judgments(
        document['companies'], raw, company_quality=False, contacts_required=False,
        scored_indexes=scored_indexes, consumed_indexes=consumed,
    )
    current['result_doc']['company_judgment_consumed_indexes'] = sorted(consumed)
    objects['current-output'] = scoring.build_scoring_output(
        'execute-current', breakdowns, company_judgments=[],
    )
    return service, reads


@pytest.mark.parametrize('accepted_source', [False, True])
def test_distinct_company_proofs_reuse_one_validated_source_output(monkeypatch, accepted_source):
    service, refs, _, _, _, reads = _five_company_source_proofs(
        accepted_source=accepted_source,
    )
    original_hash = contracts.document_hash
    original_validate = scoring.validate_scoring_output_document
    calls = Counter()

    def count_hash(value):
        if isinstance(value, dict) and value.get('schema_version') == scoring.SCORING_OUTPUT_SCHEMA_VERSION:
            calls['hash'] += 1
        return original_hash(value)

    def count_validate(value):
        calls['validate'] += 1
        return original_validate(value)

    monkeypatch.setattr(contracts, 'document_hash', count_hash)
    monkeypatch.setattr(scoring, 'validate_scoring_output_document', count_validate)
    memo = _PublicResultEvidenceReads()
    proofs = [service._company_judgment_evidence(
        cache_key=ref['cache_key'], company_input_hash=ref['company_input_hash'],
        authority_slot=0, evidence_reads=memo,
    ) for ref in refs]
    assert [proof['cache_key'] for proof in proofs] == [ref['cache_key'] for ref in refs]
    assert reads['source-output'] == 1  # Existing request-local raw-object reuse.
    assert calls == {'hash': 1, 'validate': 1}


def test_public_result_reuses_source_across_attribution_and_diagnostics(monkeypatch):
    service, reads = _five_company_public_result()
    original_get = _PublicResultEvidenceReads.source_output
    original_remember = _PublicResultEvidenceReads.remember_source_output
    original_hash = contracts.document_hash
    calls = Counter()

    def count_hash(value):
        if isinstance(value, dict) and value.get('schema_version') == scoring.SCORING_OUTPUT_SCHEMA_VERSION:
            calls['source'] += 1
        return original_hash(value)

    monkeypatch.setattr(contracts, 'document_hash', count_hash)
    monkeypatch.setattr(_PublicResultEvidenceReads, 'source_output', lambda *_args: None)
    monkeypatch.setattr(_PublicResultEvidenceReads, 'remember_source_output', lambda *_args: None)
    expected = service.public_results(ROUND_ID, SUBMISSION_ID)
    uncached_count = calls['source']
    monkeypatch.setattr(_PublicResultEvidenceReads, 'source_output', original_get)
    monkeypatch.setattr(_PublicResultEvidenceReads, 'remember_source_output', original_remember)
    calls.clear()
    reads.clear()
    actual = service.public_results(ROUND_ID, SUBMISSION_ID)
    assert contracts.canonical_json(actual) == contracts.canonical_json(expected)
    assert len(actual['company_diagnostics']) == 5
    assert actual['scoring_attribution']['unattributed_icp_count'] == 0
    assert uncached_count == 5 and calls['source'] == 1
    assert reads['source-output'] == 1


def test_source_reuse_still_checks_each_expected_hash_and_source_authority():
    service, refs, caches, source, _, _ = _five_company_source_proofs()
    memo = _PublicResultEvidenceReads()
    first, second, third = refs[:3]
    kwargs = lambda ref: dict(cache_key=ref['cache_key'],
                              company_input_hash=ref['company_input_hash'],
                              authority_slot=0, evidence_reads=memo)
    service._company_judgment_evidence(**kwargs(first))
    assert len(memo.source_outputs) == 1

    changed_hash = caches[(second['cache_key'], 0)]
    changed_hash['evidence_doc']['source_output_hash'] = 'sha256:' + 'f' * 64
    changed_hash['evidence_hash'] = contracts.document_hash(changed_hash['evidence_doc'])
    with pytest.raises(scoring.ScoringError, match='source output hash is invalid'):
        service._company_judgment_evidence(**kwargs(second))

    changed_authority = caches[(third['cache_key'], 0)]
    changed_authority['evidence_doc']['source_runner_hotkey'] = MINER
    changed_authority['evidence_doc']['runner_authority_exclusions'] = [MINER]
    changed_authority['source_runner_hotkey'] = MINER
    changed_authority['evidence_hash'] = contracts.document_hash(changed_authority['evidence_doc'])
    with pytest.raises(scoring.ScoringError, match='authority is invalid'):
        service._company_judgment_evidence(**kwargs(third))
    assert source['runner_hotkey'] == VALIDATOR_A
    assert len(memo.source_outputs) == 1  # Neither failed proof was cached.


def test_source_reuse_is_bound_to_held_raw_bytes_and_never_caches_failed_reads():
    service, refs, _, _, objects, _ = _five_company_source_proofs()
    memo = _PublicResultEvidenceReads()
    kwargs = lambda ref: dict(cache_key=ref['cache_key'],
                              company_input_hash=ref['company_input_hash'],
                              authority_slot=0, evidence_reads=memo)
    original = deepcopy(objects['source-output'])
    objects['source-output']['detail'] = 'changed'
    with pytest.raises(scoring.ScoringError, match='source output hash is invalid'):
        service._company_judgment_evidence(**kwargs(refs[0]))
    assert memo.source_outputs == {}

    objects['source-output'] = original
    memo.objects.clear()  # Simulate a fresh raw read after the failed one.
    service._company_judgment_evidence(**kwargs(refs[0]))
    assert len(memo.source_outputs) == 1
    raw = memo.objects[('source-output', scoring.MAX_SCORING_OUTPUT_BYTES)]
    _, consumer_copy = memo.source_output('source-output', scoring.MAX_SCORING_OUTPUT_BYTES, raw)
    consumer_copy['company_judgments'][0]['raw_judgment']['final_score'] = -1
    _, fresh_copy = memo.source_output('source-output', scoring.MAX_SCORING_OUTPUT_BYTES, raw)
    assert fresh_copy['company_judgments'][0]['raw_judgment']['final_score'] != -1
    changed = deepcopy(original)
    changed['detail'] = 'changed again'
    memo.objects[('source-output', scoring.MAX_SCORING_OUTPUT_BYTES)] = json.dumps(changed).encode()
    with pytest.raises(scoring.ScoringError, match='source output hash is invalid'):
        service._company_judgment_evidence(**kwargs(refs[1]))
    assert len(memo.source_outputs) == 1


def test_failed_source_object_read_cannot_populate_a_proof(monkeypatch):
    service, refs, _, _, _, _ = _five_company_source_proofs()
    memo = _PublicResultEvidenceReads()
    monkeypatch.setattr(service._objects, 'get_bounded',
                        lambda _ref, _limit: (_ for _ in ()).throw(OSError('read failed')))
    with pytest.raises(OSError, match='read failed'):
        service._company_judgment_evidence(
            cache_key=refs[0]['cache_key'], company_input_hash=refs[0]['company_input_hash'],
            authority_slot=0, evidence_reads=memo,
        )
    assert memo.objects == memo.source_outputs == {}


def test_invalid_source_document_is_not_cached_after_hash_match():
    service, refs, caches, _, objects, _ = _five_company_source_proofs(accepted_source=True)
    first = refs[0]
    objects['source-output']['breakdowns'] = 'invalid'
    cached = caches[(first['cache_key'], 0)]
    cached['evidence_doc']['source_output_hash'] = contracts.document_hash(objects['source-output'])
    cached['evidence_hash'] = contracts.document_hash(cached['evidence_doc'])
    memo = _PublicResultEvidenceReads()
    for _ in range(2):
        with pytest.raises(scoring.ScoringError, match='source output is invalid'):
            service._company_judgment_evidence(
                cache_key=first['cache_key'], company_input_hash=first['company_input_hash'],
                authority_slot=0, evidence_reads=memo,
            )
        assert memo.source_outputs == {}


@pytest.mark.parametrize('slot', [True, 0.0, '0'])
def test_custom_slot_types_keep_stateless_proof_outcome(monkeypatch, slot):
    service, refs, _, _, _, _ = _five_company_source_proofs()
    memo = _PublicResultEvidenceReads()
    first, second = refs[:2]
    service._company_judgment_evidence(
        cache_key=first['cache_key'], company_input_hash=first['company_input_hash'],
        authority_slot=0, evidence_reads=memo,
    )
    kwargs = dict(cache_key=second['cache_key'], company_input_hash=second['company_input_hash'],
                  authority_slot=slot, evidence_reads=memo)
    with monkeypatch.context() as baseline:
        baseline.setattr(_PublicResultEvidenceReads, 'source_output', lambda *_args: None)
        baseline.setattr(_PublicResultEvidenceReads, 'remember_source_output', lambda *_args: None)
        try:
            expected = service._company_judgment_evidence(**kwargs)
        except scoring.ScoringError as exc:
            expected = (type(exc), str(exc))
    try:
        actual = service._company_judgment_evidence(**kwargs)
    except scoring.ScoringError as exc:
        actual = (type(exc), str(exc))
    assert actual == expected


def test_mutable_custom_raw_bytes_and_memory_cap_disable_source_reuse(monkeypatch):
    service, refs, _, _, _, reads = _five_company_source_proofs()
    original_get = service._objects.get_bounded
    monkeypatch.setattr(service._objects, 'get_bounded',
                        lambda ref, limit: bytearray(original_get(ref, limit)))
    memo = _PublicResultEvidenceReads()
    for ref in refs[:2]:
        service._company_judgment_evidence(cache_key=ref['cache_key'],
            company_input_hash=ref['company_input_hash'], authority_slot=0,
            evidence_reads=memo)
    assert memo.source_outputs == {}

    monkeypatch.setattr(service._objects, 'get_bounded', original_get)
    limited = _PublicResultEvidenceReads()
    limited.MAX_BYTES = 1
    for ref in refs[2:4]:
        service._company_judgment_evidence(cache_key=ref['cache_key'],
            company_input_hash=ref['company_input_hash'], authority_slot=0,
            evidence_reads=limited)
    assert limited.objects == limited.source_outputs == {}
    assert reads['source-output'] == 3


@pytest.mark.parametrize('reused_failed_source', [False, True])
def test_public_response_keeps_output_and_reduces_authority_reads(reused_failed_source):
    service, ref, _, cached, _, objects, reads = _public_company_result()
    if not reused_failed_source:
        _make_current_score_the_authority(service, ref, cached, objects)
    validate = service._company_judgment_evidence
    def uncached(**kwargs):
        kwargs.pop('evidence_reads', None)
        return validate(**kwargs)
    service._company_judgment_evidence = uncached
    expected = service.public_results(ROUND_ID, SUBMISSION_ID)
    before = dict(reads)
    reads.clear()
    service._company_judgment_evidence = validate
    actual = service.public_results(ROUND_ID, SUBMISSION_ID)
    assert contracts.canonical_json(actual) == contracts.canonical_json(expected)
    assert actual['scoring_attribution']['unattributed_icp_count'] == 0
    if reused_failed_source:
        assert before == {'run_rows': 5, 'source-output': 3, 'company_rows': 3,
                          'execution-output': 1, 'current-output': 1}
        assert dict(reads) == {'run_rows': 3, 'source-output': 1, 'company_rows': 1,
                             'execution-output': 1, 'current-output': 1}
    else:
        assert before == {'run_rows': 2, 'company_rows': 2,
                          'execution-output': 1, 'current-output': 3}
        assert dict(reads) == {'company_rows': 1,
                              'execution-output': 1, 'current-output': 1}


@pytest.mark.parametrize('proof_already_read', [False, True])
def test_consumed_miss_rejects_foreign_source_after_direct_or_cached_proof(proof_already_read):
    service, ref, _, cached, _, _, reads = _public_company_result()
    current = service._store.rows['score-current']
    current['claim_response']['company_judgment_cache'] = {
        'schema_version': company_judgments.LEASE_SCHEMA_VERSION,
        'hits': [], 'misses': [_miss(ref)],
    }
    memo = {}
    if proof_already_read:
        service._company_judgment_evidence(
            cache_key=ref['cache_key'], company_input_hash=ref['company_input_hash'],
            authority_slot=0, evidence_reads=memo,
        )
    assert cached['source_score_run_id'] != current['run_id']
    with pytest.raises(scoring.ScoringError, match='consumed company judgment authority is missing or foreign'):
        service._score_run_attribution(
            current, service._store.rows['execute-current'], round_id=ROUND_ID,
            runs_by_id=service._store.rows, evidence_reads=memo,
        )
    # A fully valid foreign proof never becomes this consumer's authority.
    # The cached-proof case must not re-read its company row or source output.
    assert reads['company_rows'] == reads['source-output'] == 1
    assert len(memo) == 1


def test_later_public_request_rechecks_changed_source():
    service, _, _, _, _, objects, reads = _public_company_result()
    service.public_results(ROUND_ID, SUBMISSION_ID)
    objects['source-output']['company_judgments'][0]['raw_judgment']['final_score'] = 1
    with pytest.raises(ServiceError, match='public_company_verification_unavailable'):
        service.public_results(ROUND_ID, SUBMISSION_ID)
    assert reads['source-output'] >= 2


def test_consumer_output_is_replayed_after_shared_authority_validation():
    service, _, _, _, _, objects, reads = _public_company_result()
    objects['current-output']['breakdowns'][0]['final_score'] = 1
    with pytest.raises(ServiceError, match='public_company_verification_unavailable'):
        service.public_results(ROUND_ID, SUBMISSION_ID)
    # Attribution already validated the authority. The later consumer still
    # compares its accepted output against the unchanged frozen replay.
    assert reads['company_rows'] == reads['source-output'] == 1


def test_consumer_lease_evidence_must_match_even_after_successful_read():
    service, ref, lease, _, _, _, reads = _public_company_result()
    memo = {}
    service._company_judgment_evidence(cache_key=ref['cache_key'],
        company_input_hash=ref['company_input_hash'], authority_slot=0, evidence_reads=memo)
    forged = deepcopy(lease)
    hit = forged['hits'][0]
    hit['evidence_doc']['raw_judgment']['final_score'] = 1
    hit['evidence_hash'] = contracts.document_hash(hit['evidence_doc'])
    with pytest.raises(scoring.ScoringError, match='lease evidence changed'):
        service._validated_company_judgment_lease(forged,
            recipient_miner_hotkey=MINER, baseline=False, evidence_reads=memo)
    assert reads['company_rows'] == reads['source-output'] == 1


def test_consumer_exclusion_still_applies_to_reused_evidence():
    service, ref, lease, _, _, _, _ = _public_company_result()
    memo = {}
    service._company_judgment_evidence(cache_key=ref['cache_key'],
        company_input_hash=ref['company_input_hash'], authority_slot=0, evidence_reads=memo)
    with pytest.raises(scoring.ScoringError, match='authority is incompatible'):
        service._validated_company_judgment_lease(lease,
            recipient_miner_hotkey=VALIDATOR_A, baseline=False, evidence_reads=memo)


def test_attribution_consumer_cannot_change_the_bound_source_runner():
    service, ref, _, cached, source, _, reads = _public_company_result()
    memo = {}
    service._company_judgment_evidence(cache_key=ref['cache_key'],
        company_input_hash=ref['company_input_hash'], authority_slot=0, evidence_reads=memo)
    source['runner_hotkey'] = MINER
    evidence = cached['evidence_doc']
    with pytest.raises(scoring.ScoringError, match='attribution source is invalid'):
        service._validated_attribution_source(
            source_score_run_id=evidence['source_score_run_id'],
            source_scored_run_id=evidence['source_scored_run_id'],
            source_runner_hotkey=evidence['source_runner_hotkey'],
            round_id=ROUND_ID, runs_by_id=service._store.rows,
            company_evidence=evidence, evidence_reads=memo,
        )
    assert reads['company_rows'] == reads['source-output'] == 1


def test_mismatched_input_never_reuses_or_stores_a_failed_validation():
    service, ref, _, _, _, _, reads = _public_company_result()
    memo = {}
    service._company_judgment_evidence(cache_key=ref['cache_key'],
        company_input_hash=ref['company_input_hash'], authority_slot=0, evidence_reads=memo)
    for _ in range(2):
        with pytest.raises(scoring.ScoringError, match='input binding is invalid'):
            service._company_judgment_evidence(cache_key=ref['cache_key'],
                company_input_hash='sha256:' + 'f' * 64, authority_slot=0, evidence_reads=memo)
    assert reads['company_rows'] == 3
    assert reads['source-output'] == 1
    assert len(memo) == 1


def test_reused_success_is_isolated_from_consumer_mutation():
    service, ref, _, _, _, _, reads = _public_company_result()
    memo = {}
    kwargs = dict(cache_key=ref['cache_key'], company_input_hash=ref['company_input_hash'],
                  authority_slot=0, evidence_reads=memo)
    first = service._company_judgment_evidence(**kwargs)
    first['raw_judgment']['final_score'] = 1
    second = service._company_judgment_evidence(**kwargs)
    assert second == memo[next(iter(memo))]
    assert second['raw_judgment']['final_score'] != 1
    second['raw_judgment']['final_score'] = 2
    assert memo[next(iter(memo))]['raw_judgment']['final_score'] != 2
    assert reads['company_rows'] == reads['source-output'] == 1


def test_source_failure_is_rechecked_and_never_added_to_response_map():
    service, ref, _, _, _, objects, reads = _public_company_result()
    memo = {}
    kwargs = dict(cache_key=ref['cache_key'], company_input_hash=ref['company_input_hash'],
                  authority_slot=0, evidence_reads=memo)
    objects['source-output']['company_judgments'][0]['raw_judgment']['final_score'] = 1
    for _ in range(2):
        with pytest.raises(scoring.ScoringError, match='source output hash is invalid'):
            service._company_judgment_evidence(**kwargs)
        assert memo == {}
    assert reads['company_rows'] == reads['source-output'] == 2


def test_different_authority_slot_cannot_reuse_a_successful_read():
    service, ref, _, _, _, _, reads = _public_company_result()
    memo = {}
    service._company_judgment_evidence(cache_key=ref['cache_key'],
        company_input_hash=ref['company_input_hash'], authority_slot=0, evidence_reads=memo)
    with pytest.raises(scoring.ScoringError, match='cache entry is missing'):
        service._company_judgment_evidence(cache_key=ref['cache_key'],
            company_input_hash=ref['company_input_hash'], authority_slot=1, evidence_reads=memo)
    assert reads['company_rows'] == 2
    assert reads['source-output'] == 1
    assert len(memo) == 1
