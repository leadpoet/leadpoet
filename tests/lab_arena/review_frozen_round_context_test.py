"""Service review reuses its trusted round without a second catalog dependency."""

from datetime import datetime, timezone
import json
from types import SimpleNamespace

import pytest

from lab_arena import code_review
from lab_arena.code_review_runtime import SubmissionCodeReviewer
from lab_arena.service import ArenaService
from tests.lab_arena.research_control_code_review_test import SOURCE, _assessment
from tests.lab_arena.test_lab_arena_code_review_runtime import (
    MINER_KEY, SUBMISSION, Objects, ReviewStore, ReviewTransport,
    _archive, _price_table, _review_body, _source,
)


def _service(configuration, *, research=False):
    payload = _archive(SOURCE if research else _source())
    row = {**SUBMISSION, 'source_size_bytes': len(payload),
           'reasoning_capture_policy': None, 'configuration_doc': {}}
    round_row = {'round_id': row['round_id'], 'status': 'running',
                 'configuration_doc': configuration}
    class Store(ReviewStore):
        round_reads = 0
        def get_round(self, _round_id):
            self.round_reads += 1
            if self.round_reads > 1:
                raise OSError('second catalog read unavailable')
            return round_row
        def list_submissions(self, _round_id, *, status):
            return [row] if status == 'accepted' else []
    store = Store()
    def reply(parameters):
        result = _review_body(parameters)
        if research:
            document = json.loads(result['choices'][0]['message']['content'])
            prepared = code_review.prepare_request(payload, research_control=True)
            document['research_control'] = _assessment(prepared)
            result['choices'][0]['message']['content'] = json.dumps(document)
        return {'body': result}
    transport = ReviewTransport(reply)
    reviewer = SubmissionCodeReviewer(
        store=store, objects=Objects(payload), credential_for=lambda _: MINER_KEY,
        price_table=_price_table(), transport=transport,
    )
    service = ArenaService.__new__(ArenaService)
    service._store = store
    service._config = SimpleNamespace(code_reviewer=reviewer)
    service._clock = lambda: datetime(2026, 10, 25, tzinfo=timezone.utc)
    service._require_round_ownership = lambda _: None
    service._require_round_mode = lambda value: value
    service.active_rounds = lambda: [{'round_id': row['round_id']}]
    return service, store, transport, reviewer, row, round_row


@pytest.mark.parametrize('research', [False, True])
def test_real_review_entrypoint_reuses_frozen_round_and_ignores_submission_policy(research):
    configuration = {'reasoning_capture_policy': 'full_generated_thinking_v1'} if research else {}
    service, store, transport, _, _, _ = _service(configuration, research=research)
    assert service.review_pending_submissions() == {'reviewed': 1}
    assert store.finish_calls[0][3] == 'passed'
    assert store.round_reads == 1
    assert len(transport.sent) == 1
    assert ('research_control' in store.finish_calls[0][4]) is research


@pytest.mark.parametrize('configuration', [None, [], {'reasoning_capture_policy': 'unsupported'},
                                         {'training_maximum_missing_reasoning_percent': '1'}])
def test_invalid_frozen_configuration_fails_closed_before_paid_dispatch(configuration):
    service, store, transport, _, _, _ = _service(configuration)
    assert service.review_pending_submissions() == {'reviewed': 1}
    assert store.finish_calls[0][3] == 'error'
    assert store.finish_calls[0][4]['error_code'] == 'code_review_preparation_unavailable'
    assert transport.sent == []


def test_trusted_round_context_must_match_submission_round():
    _, store, transport, reviewer, row, round_row = _service({})
    result = reviewer.review_with_frozen_round(row, {**round_row, 'round_id': 'wrong-round'})
    assert result['status'] == 'error'
    assert store.round_reads == 0
    assert transport.sent == []
