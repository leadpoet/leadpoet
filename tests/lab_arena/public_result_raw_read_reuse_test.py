"""Response-local raw reads preserve strict authority validation and freshness."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from lab_arena import contracts, scoring
from lab_arena.service import ServiceError, _PublicResultEvidenceReads
from tests.lab_arena.public_result_company_evidence_reads_test import _public_company_result
from tests.lab_arena.scoring_validator_attribution_test import ROUND_ID, SUBMISSION_ID


def test_raw_maps_keep_identities_separate_and_copies_isolated():
    store = SimpleNamespace(
        get_run=Mock(side_effect=lambda key: {'run_id': key}),
        get_submission=Mock(side_effect=lambda key: {'submission_id': key}),
        get_signal_judgment_uses=Mock(side_effect=lambda key: [{'run_id': key}]),
        get_signal_judgment=Mock(side_effect=lambda key, slot: {'cache_key': key, 'authority_slot': slot}),
        signal_authority_exclusions=Mock(side_effect=lambda key: [key]),
    )
    reads = _PublicResultEvidenceReads()
    for name, load in (
        ('get_run', lambda key: reads.run(store, key)),
        ('get_submission', lambda key: reads.submission(store, key)),
        ('get_signal_judgment_uses', lambda key: reads.uses(store, key)),
        ('signal_authority_exclusions', lambda key: reads.authority_exclusions(store, key)),
    ):
        original = deepcopy(load('first'))
        got = load('first')
        if isinstance(got, list): got.clear()
        else: got.clear()
        assert load('first') == original
        assert load('second') != original
        assert getattr(store, name).call_count == 2
    assert reads.signal(store, 'key', 0) == reads.signal(store, 'key', 0)
    assert reads.signal(store, 'key', 1)['authority_slot'] == 1
    assert store.get_signal_judgment.call_count == 2


def test_fresh_context_observes_changed_and_missing_rows():
    store = SimpleNamespace(get_run=Mock(return_value={'run_id': 'first'}))
    reads = _PublicResultEvidenceReads()
    assert reads.run(store, 'first')['run_id'] == 'first'
    store.get_run.return_value = None
    assert reads.run(store, 'first')['run_id'] == 'first'
    assert _PublicResultEvidenceReads().run(store, 'first') is None
    store.get_run.return_value = {'run_id': 'recovered'}
    assert _PublicResultEvidenceReads().run(store, 'first')['run_id'] == 'recovered'


def test_object_reads_bind_the_exact_byte_limit_and_do_not_cache_exceptions():
    def bounded(ref, limit):
        if type(limit) is not int or limit < 4:
            raise ValueError('object bound')
        return b'body'
    objects = SimpleNamespace(get_bounded=Mock(side_effect=bounded))
    reads = _PublicResultEvidenceReads()
    assert reads.object(objects, 'ref', 4) == reads.object(objects, 'ref', 4) == b'body'
    for limit in (3, True, 3):
        with pytest.raises(ValueError, match='object bound'):
            reads.object(objects, 'ref', limit)
    assert objects.get_bounded.call_count == 4
    assert _PublicResultEvidenceReads().object(objects, 'ref', 4) == b'body'
    assert objects.get_bounded.call_count == 5


@pytest.mark.parametrize('bound', ['bytes', 'identities'])
def test_cache_bounds_leave_reads_and_validation_available(bound):
    reads = _PublicResultEvidenceReads()
    if bound == 'bytes': reads.MAX_BYTES = 1
    else: reads.MAX_IDENTITIES = 0
    store = SimpleNamespace(get_run=Mock(return_value={'run_id': 'source'}))
    assert reads.run(store, 'source') == reads.run(store, 'source')
    assert store.get_run.call_count == 2
    objects = SimpleNamespace(get_bounded=Mock(return_value=b'body'))
    assert reads.object(objects, 'ref', 4) == reads.object(objects, 'ref', 4)
    assert objects.get_bounded.call_count == 2
    assert reads._bytes == 0


def test_non_json_read_stays_on_original_validation_path():
    value = {'malformed': b'not JSON'}
    store = SimpleNamespace(get_run=Mock(return_value=value))
    reads = _PublicResultEvidenceReads()
    assert reads.run(store, 'source') is value
    assert reads.runs == {}


def test_partial_completed_authority_response_is_exactly_unchanged():
    service, _, _, _, _, _, _ = _public_company_result()
    with patch.object(_PublicResultEvidenceReads, 'MAX_BYTES', 0):
        before = service.public_results(ROUND_ID, SUBMISSION_ID)
    after = service.public_results(ROUND_ID, SUBMISSION_ID)
    assert contracts.canonical_json(before) == contracts.canonical_json(after)


@pytest.mark.parametrize('fault', ['missing_run', 'changed_run', 'missing_company', 'bad_source_json', 'changed_source'])
def test_next_public_request_revalidates_missing_or_invalid_authority(fault):
    service, _, _, _, _, objects, _ = _public_company_result()
    service.public_results(ROUND_ID, SUBMISSION_ID)
    if fault == 'missing_run': del service._store.rows['score-source']
    elif fault == 'changed_run': service._store.rows['score-source']['claim_request_hash'] = 'sha256:' + 'f' * 64
    elif fault == 'missing_company': service._store.get_company_judgment = lambda *_: None
    elif fault == 'bad_source_json': objects['source-output'] = []
    else: objects['source-output']['company_judgments'][0]['raw_judgment']['final_score'] = 1
    with pytest.raises((ServiceError, scoring.ScoringError)):
        service.public_results(ROUND_ID, SUBMISSION_ID)
