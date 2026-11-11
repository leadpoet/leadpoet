"""Admission source evidence reaches its owner through real local SQL and API."""

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
from types import SimpleNamespace

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from fastapi.testclient import TestClient
import pytest

from lab_arena import code_review, code_review_policy, contracts, submission_similarity
from lab_arena.api import create_app
from lab_arena.code_review_runtime import SubmissionCodeReviewer
from lab_arena.service import ArenaService
from tests.lab_arena.private_owner_review687_postgres_test import database, context
from tests.lab_arena.prospective_admission_primary_test import receipt
from tests.lab_arena.test_lab_arena_code_review_postgres import _age_review_error
from tests.lab_arena.test_lab_arena_code_review_runtime import (
    MINER_KEY, Objects, ReviewTransport, _archive, _price_table, _source,
)
from tests.lab_arena.test_lab_arena_migration_postgres import (
    encrypted_runtime_credentials, hotkey, round_config, source_submission_doc,
)


def _register(store, case, source):
    now = datetime.now(timezone.utc)
    rid = 'arena-2099-03-01-' + hashlib.sha256(('flow687-' + case).encode()).hexdigest()[:12]
    sid = 'flow687-' + case
    miner = hotkey(sid)
    configuration = round_config(rid, [hotkey(sid + '-runner')])
    configuration['schedule'].update({
        'submission_open': (now - timedelta(hours=2)).isoformat(),
        'submission_cutoff': (now + timedelta(minutes=5)).isoformat(),
        'benchmark_deadline': (now + timedelta(minutes=65)).isoformat(),
    })
    assert store.create_round(rid, configuration)['status'] == 'created'
    payload = _archive(source)
    document = {**source_submission_doc(rid, sid), 'source_size_bytes': len(payload)}
    assert store.register_submission(rid, sid, miner, document)['status'] == 'registered'
    inspected = submission_similarity.inspect_archive(payload)
    assert store.accept_submission_source_with_credentials(
        rid, sid, miner, encrypted_runtime_credentials(sid),
        inspected.archive_sha256, inspected.normalized_sha256,
    )['status'] == 'ok'
    return rid, sid, miner, payload


def _finding(source):
    return {
        'category': 'fabricated_evidence', 'file': 'src/agent.py',
        'evidence': '"country": icp["country"]',
        'explanation': (
            'The emitted country is the requested ICP value, asserted as a company fact. '
            'The reachable run_icp directly returns this substitution without research. '
            'The evidence instruction exists but is bypassed by the direct return; this '
            'positive unsupported output, rather than a missing machine check, is the violation.'
        ),
        'source_trace': [
            {'role': 'claim_origin', 'file': 'src/agent.py', 'excerpt': '"country": icp["country"]'},
            {'role': 'output_path', 'file': 'src/agent.py',
             'excerpt': 'return [{"country": icp["country"]}]'},
            {'role': 'evidence_control', 'file': 'prompts/evidence.txt',
             'excerpt': source['prompts/evidence.txt'].decode().strip()},
        ],
    }


def _signed_reader(store, miner):
    """Use actual signature validation, service round gates, store and API."""
    key = Ed25519PrivateKey.generate()
    def verify(hotkey_value, signature, message):
        if hotkey_value != miner:
            return False
        try:
            key.public_key().verify(bytes.fromhex(signature.removeprefix('0x')), message.encode())
            return True
        except (InvalidSignature, ValueError):
            return False
    service = object.__new__(ArenaService)
    service._store = store
    service.now = lambda: datetime.now(timezone.utc)
    service._config = SimpleNamespace(mode='live', network_name='finney', netuid=71,
                                      pinned_round_id=None, verify_signature=verify)
    def read(client, rid, sid):
        envelope = contracts.build_signed_request(
            scope=contracts.SCOPE_SUBMISSION_REVIEW, round_id=rid, hotkey=miner,
            timestamp=int(service.now().timestamp()), body={'submission_id': sid},
            sign_message=lambda message: key.sign(message.encode()).hex(),
        )
        response = client.post('/arena/v1/submissions/' + sid + '/review', json=envelope)
        assert response.status_code == 200, response.text
        assert response.headers['cache-control'] == 'private, no-store'
        assert response.headers['x-content-type-options'] == 'nosniff'
        return response
    return service, read


@pytest.mark.parametrize('case,cost', [
    ('unsafe-settled', '0.001234'), ('unsafe-unknown', None),
    ('safe', '0.001234'), ('malformed-recovery', '0.001234'),
])
def test_runtime_sql_ledger_owner_findings_and_signed_reader(context, case, cost):
    store, connection = context
    source = _source()
    unsafe = case != 'safe'
    if unsafe:
        source['src/agent.py'] = b'def run_icp(icp, tools): return [{"country": icp["country"]}]\n'
    rid, sid, miner, payload = _register(store, case, source)
    finding = _finding(source) if unsafe else None
    responses = []
    def respond(parameters):
        body = receipt(parameters, cost=cost)
        # Full-source integrity requires measured input usage even when the
        # receipt lacks enough billing data to confirm the actual charge.
        body.setdefault('usage', {})['prompt_tokens'] = 1000
        if unsafe:
            import json
            content = json.loads(body['choices'][0]['message']['content'])
            item = deepcopy(finding)
            if case == 'malformed-recovery' and not responses:
                item.pop('source_trace')
            content.update(verdict='reject', findings=[item])
            body['choices'][0]['message']['content'] = json.dumps(content)
        responses.append(body)
        return {'body': body}
    transport, objects = ReviewTransport(respond), Objects(payload)
    prices = _price_table()
    prices['models'][code_review.PROSPECTIVE_REVIEW_MODEL] = dict(prices['models'][code_review.DEFAULT_REVIEW_MODEL])
    key_reads = []
    def credential_for(row):
        key_reads.append(row['submission_id'])
        return MINER_KEY  # Public synthetic test value; no real credential is read.
    reviewer = SubmissionCodeReviewer(store=store, objects=objects, credential_for=credential_for,
        price_table=prices, transport=transport, prospective_primary_model=code_review.PROSPECTIVE_REVIEW_MODEL)
    service, read = _signed_reader(store, miner)
    with TestClient(create_app(service)) as client:
        result = reviewer.review(store.get_submission(sid))
        if case == 'malformed-recovery':
            saved = store.get_submission(sid)
            assert result['status'] == saved['code_review_status'] == 'error'
            assert saved['code_review_doc']['error_code'] == 'review_response_invalid'
            assert saved['code_review_doc']['error_reason'] == 'finding_keys'
            assert saved['code_review_doc']['review_invalid_attempts'] == 1
            assert code_review_policy.review_retry_available(saved)
            assert not read(client, rid, sid).json()['owner_findings']['available']
            assert 'owner_findings' not in saved['code_review_doc']
            assert reviewer.review(saved)['status'] == 'backoff'
            assert len(transport.sent) == 1
            _age_review_error(connection, sid)
            result = reviewer.review(store.get_submission(sid))
        saved = store.get_submission(sid)
        expected = 'rejected' if unsafe else 'passed'
        assert result['status'] == saved['code_review_status'] == expected
        assert saved['code_review_doc']['file_count'] == len(source)
        assert saved['code_review_doc']['source_bytes'] == sum(len(x) for x in source.values())
        assert saved['source_archive_sha256'] == contracts.hash_bytes(payload)
        ledger = store.list_ledger(submission_id=sid)
        attempts = 2 if case == 'malformed-recovery' else 1
        assert saved['code_review_attempts'] == attempts
        reservations = [x for x in ledger if x['entry_kind'] == 'reservation']
        terminal_costs = [x for x in ledger if x['entry_kind'] in {'settlement', 'uncertain'}]
        assert len(reservations) == len(terminal_costs) == len(transport.sent) == attempts
        assert saved['code_review_doc']['cost_status'] == ('uncertain' if cost is None else 'settled')
        assert all(x['entry_kind'] == ('uncertain' if cost is None else 'settlement') for x in terminal_costs)
        if cost is None:
            assert terminal_costs[0]['amount_microusd'] == reservations[0]['amount_microusd']
        else:
            assert all(x['amount_microusd'] == 1234 for x in terminal_costs)
        private = read(client, rid, sid)
        if unsafe:
            excerpts = private.json()['owner_findings']
            assert excerpts['available'] is True
            assert excerpts['findings'] == [{'category': 'fabricated_evidence', 'file': 'src/agent.py',
                                             'evidence': finding['evidence']}]
            assert saved['code_review_doc']['owner_findings']['findings'] == excerpts['findings']
            assert finding['explanation'] not in private.text
            assert 'source_trace' not in private.text
            assert finding['evidence'] not in client.get('/arena/v1/submissions/' + sid).text
        else:
            assert not private.json()['owner_findings']['available']
            assert 'owner_findings' not in saved['code_review_doc']
        before = deepcopy(ledger)
        observed = (len(transport.sent), len(objects.calls), len(key_reads))
        assert reviewer.review(saved)['status'] == 'existing'
        assert store.list_ledger(submission_id=sid) == before
        assert observed == (len(transport.sent), len(objects.calls), len(key_reads))
