"""Rule/source delivery, not a synthetic claim of model semantic improvement."""
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path

import pytest

from lab_arena import operations
from qualification.scoring import intent_verification_three_stage as intent
from qualification.scoring.evaluation_clock import use_evaluation_date
from qualification.scoring.prompts import _common


FIXTURE = Path(__file__).parent / 'fixtures/intent_counterparty_role_requests.json.gz'
FIXTURE_SHA = '4859bf9ce1f270a72428f77b1342bf31e3c4ea8cb14a42f97ea06e27a2488ccf'


def cases():
    raw = FIXTURE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == FIXTURE_SHA
    return json.loads(gzip.decompress(raw))['cases']


def assert_role_rule(prompt):
    normalized = ' '.join(prompt.split())
    assert 'bind that qualifier to the participant actually filling that role' in normalized
    assert "the target's own category cannot substitute" in normalized
    assert 'sponsor name inside a team, event, product, or venue name' in normalized
    assert 'reasonable semantic role inference' in normalized
    assert 'literal category labels are not required' in normalized


@pytest.mark.parametrize('case', cases(), ids=lambda row: row['id'])
def test_exact_retained_sources_and_complete_counterparty_criterion_reach_judges(case):
    row = deepcopy(case['row']); source = case['source_text']
    assert hashlib.sha256(source.encode()).hexdigest() == case['source_text_sha256']
    with use_evaluation_date(case['evaluation_date']):
        initial = intent._build_verification_prompt(row)
        final = intent._build_final_judge_prompt(row, {'results': [{
            'url': row['claimed_source_urls'][0], 'title': 'Exact retained source', 'text': source}], 'statuses': []})
    for prompt in (initial, final):
        signal = json.loads(prompt.split('Intent signal to verify:\n', 1)[1].split('\n\nThree-part verification', 1)[0])
        assert signal['target_icp_signal'] == row['_target_signal_text']
        assert signal['miner_claim'] == row['claim']
        assert signal['buyer_original_request'] == row['_buyer_request_context']
        assert row['claimed_source_urls'][0] in prompt
        assert_role_rule(prompt)
        assert 'Do not reject it solely because the parties have customer-supplier roles.' in ' '.join(prompt.split())
        assert 'explicit counterparty qualifier' in ' '.join(prompt.split())
    assert source in final
    assert len(final) <= _common.FINAL_JUDGE_PROMPT_MAX_CHARS
    messages = intent._openrouter_user_messages(final)
    assert ''.join(m['content'] for m in messages) == final
    assert all(len(m['content']) <= 32000 for m in messages)
    operations.validate_operation_request('openrouter.chat', {'model': 'openai/gpt-6-luna',
        'messages': [{'role': 'system', 'content': intent._SYS_MESSAGE}, *messages],
        'max_tokens': 2000, 'response_format': {'type': 'json_schema', 'json_schema': {
            'name': 'verification', 'strict': True, 'schema': intent._SCHEMA}}})


@pytest.mark.parametrize('evidence_type', [None, 'PARTNERSHIP', 'HIRING', 'FUNDING',
    'SOCIAL_POSTING', 'TECHSTACK', 'PODCAST_APPEARANCE'])
@pytest.mark.parametrize('recovered', [False, True])
def test_common_rule_reaches_every_dispatch_and_same_event_recovery(evidence_type, recovered):
    row = deepcopy(cases()[0]['row']); row['_evidence_type'] = evidence_type
    if recovered:
        row.update(_same_event_resolution=True, _unavailable_source_resolution=True)
    for prompt in [intent._build_verification_prompt(row), intent._build_final_judge_prompt(row,
        {'results': [{'url': row['claimed_source_urls'][0], 'text': 'Exact original event source.'}], 'statuses': []})]:
        assert_role_rule(prompt)
        assert row['_target_signal_text'] in prompt
        assert '`OR` / `either` separates alternatives' in prompt
        assert '`AND` / `all` joins requirements' in prompt


def test_changed_prompt_does_not_relabel_the_exact_saved_model_verdict():
    # This fixture is the real semantic failure. The ordinary structural guard
    # does not independently classify counterparty types. A prompt-only repair
    # must be evaluated with the actual judge later, never a responsive stub.
    case = cases()[0]
    verdict = intent._apply_guardrails(case['row'], deepcopy(case['public_verdict']))
    assert intent._decision(verdict, company_quality=True, require_same_entity=True) == 'approve'
    assert case['public_verdict']['signal_evaluations'][0]['same_entity_check'] == 'pass'


def test_unrelated_event_alternative_keeps_original_or_semantics():
    row = deepcopy(cases()[0]['row'])
    row['_target_signal_text'] = 'Announced a partnership with a bank OR launched a new product.'
    row['_evidence_type'] = 'PRODUCT_LAUNCH'
    prompt = intent._build_verification_prompt(row)
    assert_role_rule(prompt)
    assert 'Apply this block only to a chosen partnership alternative' in prompt
    assert 'One complete alternative is\n        sufficient' in prompt
