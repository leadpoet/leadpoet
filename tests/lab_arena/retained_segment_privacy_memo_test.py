"""Stored rows share only the existing bounded no-secret clean-text memo."""
import base64
from collections import Counter, OrderedDict
import copy
import gzip
import json

import pytest

from lab_arena import contracts, training_capture as privacy, training_observations as obs, trajectory
from lab_arena.service import LocalObjectStore
from tests.lab_arena.training_input_test import CLAIM_HASH, TOKEN
from tests.lab_arena.training_observations_test import observation


TEXT = 'public retained company evidence ' * 400


def payload(count=8):
    rows = [observation(ordinal=i, action=i) for i in range(count)]
    for row in rows:
        row['request']['admitted_parameters']['messages'][0]['content'] = TEXT
    return {'observation_from':0, 'observations':rows, 'receipt_from':0,
        'response_receipts':[], 'receipt_updates':[], 'agent_from':0, 'agent_observations':[],
        'observation_count':count, 'local_refusal_count':0, 'response_receipt_count':0,
        'agent_observation_count':0, 'failures':[]}


def stored(tmp_path, document=None):
    document = obs._clean_segment_payload(payload() if document is None else document, secrets=(TOKEN,))
    segment, raw = obs.build_segment('retained:1', 'score', document, lease_generation=1,
        claim_request_hash=CLAIM_HASH, number=0, previous_sha256=None,
        cumulative_bytes=len(contracts.canonical_json(document).encode()), secrets=(TOKEN,))
    objects = LocalObjectStore(tmp_path)
    objects.put(obs.segment_ref('retained:1',1,CLAIM_HASH,0),raw)
    return objects, segment


def read(objects):
    return obs.read_segment(objects,'retained:1','score',1,CLAIM_HASH,0)


def test_stored_rows_scan_repeated_long_text_once_per_read(tmp_path, monkeypatch):
    objects, expected = stored(tmp_path)
    scans = Counter()
    redact = trajectory.redact_text
    def counted(text,*args,**kwargs):
        if text == TEXT:
            scans[text] += 1
        return redact(text,*args,**kwargs)
    monkeypatch.setattr(trajectory,'redact_text',counted)
    restored = read(objects)
    assert contracts.canonical_json(restored) == contracts.canonical_json(expected)
    assert scans[TEXT] == 1
    read(objects)
    assert scans[TEXT] == 2  # Memo is private to one stored segment read.


def replace_stored(objects, segment):
    segment['payload_sha256'] = contracts.document_hash(segment['payload'])
    segment['segment_sha256'] = contracts.document_hash({k:v for k,v in segment.items() if k != 'segment_sha256'})
    next(objects._root.rglob('*.gz')).write_bytes(gzip.compress(contracts.canonical_json(segment).encode(),mtime=0))


@pytest.mark.parametrize('damage', ['run','kind','generation','claim','number','previous','schema','payload_hash','segment_hash','count','privacy_key','hidden_reasoning','row_privacy','size'])
def test_every_identity_hash_size_and_privacy_check_remains(tmp_path, damage):
    objects, segment = stored(tmp_path)
    if damage == 'run': segment['run_id'] = 'other'
    elif damage == 'kind': segment['kind'] = 'execute'
    elif damage == 'generation': segment['lease_generation'] = True
    elif damage == 'claim': segment['claim_request_hash'] = 'sha256:'+'f'*64
    elif damage == 'number': segment['number'] = True
    elif damage == 'previous': segment['previous_sha256'] = 'sha256:'+'f'*64
    elif damage == 'schema': segment['schema_version'] = 'other'
    elif damage == 'count': segment['payload']['observation_count'] = True
    elif damage == 'privacy_key': segment['payload']['observations'][1]['request']['admitted_parameters']['authorization'] = TEXT
    elif damage == 'hidden_reasoning': segment['payload']['observations'][1]['request']['admitted_parameters']['response'] = {'reasoning_content': TEXT}
    elif damage == 'row_privacy': segment['payload']['observations'][1]['privacy'] = {'redacted': True,'redactions':[]}
    elif damage == 'size': segment['cumulative_bytes'] = 0
    replace_stored(objects,segment)
    if damage in ('payload_hash','segment_hash'):
        segment[damage.replace('hash','sha256')] = 'sha256:'+'0'*64
        next(objects._root.rglob('*.gz')).write_bytes(gzip.compress(contracts.canonical_json(segment).encode(),mtime=0))
    with pytest.raises(obs.ObservationError):
        read(objects)


def test_known_secret_and_node_checks_ignore_warm_clean_digest(monkeypatch):
    memo = OrderedDict()
    assert privacy.sanitize_training_value(TEXT,clean_text_digests=memo) == (TEXT,[])
    assert memo
    row = payload(1)['observations'][0]
    secret = 'retained company'
    cleaned = obs.clean_observation(row,(secret,),clean_text_digests=memo)
    assert secret not in cleaned['request']['admitted_parameters']['messages'][0]['content']
    assert cleaned['privacy']['redactions'] == ['credential']
    sanitize = privacy.sanitize_training_value
    def bounded(value,secrets=(),**kwargs):
        return sanitize(value,secrets,max_nodes=3,**kwargs)
    monkeypatch.setattr(privacy,'sanitize_training_value',bounded)
    with pytest.raises(privacy.CaptureError,match='request_structure_limit'):
        obs.clean_observation(row,clean_text_digests=memo)


def test_incoming_build_still_has_no_digest_memo_and_scans_known_secret(monkeypatch):
    seen = []
    sanitize = privacy.sanitize_training_value
    def checked(value,secrets=(),**kwargs):
        seen.append((secrets,kwargs.get('clean_text_digests')))
        return sanitize(value,secrets,**kwargs)
    monkeypatch.setattr(privacy,'sanitize_training_value',checked)
    document = payload(2)
    segment,_ = obs.build_segment('retained:1','score',document,lease_generation=1,claim_request_hash=CLAIM_HASH,
        number=0,previous_sha256=None,cumulative_bytes=len(contracts.canonical_json(document).encode()),secrets=('retained company',))
    assert seen and all(secrets == ('retained company',) and memo is None for secrets,memo in seen)
    assert all(row['privacy']['redacted'] for row in segment['payload']['observations'])


def binary_row(index=0):
    raw = TEXT.encode('latin-1')
    sha = contracts.hash_bytes(raw)
    return {'id':index,'kind':'web_fetch_binary','observed_at':'2026-11-10T00:00:00Z',
        'after_action_sequence':None,'depends_on':[], 'privacy':{'redacted':False,'redactions':[]},
        'data':{'url':'https://example.test/source.pdf','final_url':'https://example.test/source.pdf',
            'retrieved_at':'2026-11-10T00:00:00Z','status':200,'content_type':'application/pdf',
            'body_sha256':sha,'body_bytes':len(raw),'stored_body_sha256':None,'stored_body_bytes':None,
            'body_b64':base64.b64encode(raw).decode(),'body_ref':None,'privacy_review':'unreviewed_binary',
            'capture':{'source':'runtime_http_v1','complete':True,'received_body_sha256':sha,'received_body_bytes':len(raw)}}}


def test_binary_pass_shares_memo_but_rechecks_metadata_and_body_hash(tmp_path, monkeypatch):
    from lab_arena import lab_arena_checkpoint
    document = payload(1)
    document['agent_observations'] = [binary_row(0),binary_row(1)]
    document['agent_observation_count'] = 2
    objects,expected = stored(tmp_path,document)
    scans,metadata = [],[]
    redact,validate = trajectory.redact_text,lab_arena_checkpoint.validate_binary_capture_metadata
    def counted(text,*args,**kwargs):
        if text == TEXT: scans.append(1)
        return redact(text,*args,**kwargs)
    def check(*args):
        metadata.append(1)
        return validate(*args)
    monkeypatch.setattr(trajectory,'redact_text',counted)
    monkeypatch.setattr(lab_arena_checkpoint,'validate_binary_capture_metadata',check)
    assert read(objects) == expected
    assert len(scans) == 1 and len(metadata) == 4  # Shape and actual bytes, for each row.
    changed = copy.deepcopy(expected)
    changed['payload']['agent_observations'][1]['data']['capture']['received_body_bytes'] += 1
    replace_stored(objects,changed)
    with pytest.raises(obs.ObservationError):
        read(objects)
