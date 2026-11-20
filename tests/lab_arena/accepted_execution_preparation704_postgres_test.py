"""Large frozen policies remain exact when preparing corrected execution judging."""
import json
from copy import deepcopy
from pathlib import Path

import pytest

from scripts import prepare_accepted_execution_correction as preparation
from tests.lab_arena import accepted_execution_correction703_postgres_test as prior
from tests.lab_arena.accepted_execution_correction703_postgres_test import case


@pytest.fixture(scope='module')
def database():
    gen = prior.database.__wrapped__()
    pg, dsn = next(gen)
    try:
        with pg.connect(**dsn) as c, c.cursor() as q:
            sql = (prior.SCRIPTS / '704-lab-arena-correction-preparation-size.sql').read_text()
            q.execute(sql)
            q.execute(sql)  # Applying twice preserves the exact installed function and ACL.
        yield pg, dsn
    finally:
        gen.close()


def accepted_replacement(p):
    # Today's real frozen policy is 2.26 MB before its scoring plan is added.
    # Padding represents retained policy text, not extra runtime work.
    with p.c.cursor() as q:
        q.execute('SET LOCAL session_replication_role=replica')
        q.execute("UPDATE lab_arena_rounds SET configuration_doc=configuration_doc||%s::jsonb WHERE round_id=%s",
                  (json.dumps({'retained_policy_text': 'x' * 2_260_787}), prior.ROUND))
        q.execute('SET LOCAL session_replication_role=origin')
    p.c.commit()
    child = prior.grant(p)['child_identity']['run_id']
    assert prior.claim(p)['run_id'] == child
    ref = child + '-output'
    p.objects[ref] = prior.contracts.canonical_json({
        'schema_version': prior.contracts.OUTPUT_DOCUMENT_SCHEMA_VERSION,
        'companies': p.companies}).encode()
    assert p.store.complete_attempt(run_id=child, lease_token_hash=prior.TOKEN,
        result={'terminal_status': 'accepted'}, terminal_cause='accepted',
        output_ref=ref)['status'] == 'accepted'
    return child


def test_large_policy_private_preparation_is_exact(case, tmp_path):
    child = accepted_replacement(case)
    path = tmp_path / 'private.json'
    preparation.write_preparation(case.service, child, path)
    assert json.loads(path.read_text()) == preparation.prepare(case.service, child)
    assert path.stat().st_mode & 0o777 == 0o600


def test_large_policy_native_preparation_preserves_checks(case):
    p = case
    child = accepted_replacement(p)
    before = prior.snapshot(p)
    preview, payload = prior.finalize(p, child, dry=True)
    assert prior.snapshot(p) == before
    assert len(json.dumps(payload)) > 2_000_000
    result, _ = prior.finalize(p, child, payload=payload)
    assert result['child_identity'] == preview['child_identity']
    assert prior.finalize(p, child, payload=payload)[0]['idempotent'] is True
    assert p.store.get_round(prior.ROUND)['configuration_doc'] == payload['expected_round']['configuration_doc']
    judge = result['child_identity']['run_id']
    prior.age(p, judge)
    assert prior.claim(p, prior.WRONG)['status'] == 'no_pending'
    lease = prior.claim(p)
    assert lease['run_id'] == judge and lease['scored_run_id'] == child
    prior.complete_company_judgment(p, lease)
    assert p.store.get_run(judge)['status'] == 'accepted'


def test_preparation_still_rejects_oversize_and_stale_state(case, tmp_path, monkeypatch):
    p = case
    child = accepted_replacement(p)
    payload = preparation.prepare(p.service, child)
    oversized = deepcopy(payload)
    oversized['expected_round']['configuration_doc']['retained_policy_text'] = 'x' * preparation.MAX_PREPARATION_BYTES
    with pytest.raises(Exception, match='corrected execution preparation invalid'):
        prior.finalize(p, child, payload=oversized, dry=True)
    p.c.rollback()
    monkeypatch.setattr(preparation, 'prepare', lambda *args: oversized)
    path = tmp_path / 'oversized.json'
    with pytest.raises(ValueError, match='fixed size bound'):
        preparation.write_preparation(p.service, child, path)
    assert not path.exists()
    stale = deepcopy(payload)
    stale['expected_round']['configuration_doc']['retained_policy_text'] = 'changed'
    with pytest.raises(Exception, match='round changed'):
        prior.finalize(p, child, payload=stale, dry=True)
    p.c.rollback()

