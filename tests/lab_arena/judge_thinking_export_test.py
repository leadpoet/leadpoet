"""Judge thinking uses the existing exact-pair proof, separately from sourcing."""
from copy import deepcopy

import pytest

from lab_arena import contracts, reasoning_capture, scoring, training_steps, training_thinking
from tests.lab_arena.training_steps_test import seal
from tests.lab_arena.training_thinking_test import full_example


def judge_example():
    example = full_example()
    example['configuration']['scorer_policy'] = scoring.build_scorer_policy(
        judge_models={role: 'test/model' for role in scoring.DEFAULT_JUDGE_MODELS},
        judge_provider_routes={'test/model': 'route-a'})
    example['configuration']['scorer_image_digest'] = 'sha256:' + 'e' * 64
    call = deepcopy(example['provider_calls'][0])
    call.update(run_id=example['score_run_id'], call_identity='judge-call')
    artifact = deepcopy(example['worker_observation_artifacts'][example['run_id']])
    artifact.update(run_id=call['run_id'], kind='score')
    for row in artifact['payload']['response_receipts']:
        row['call_identity'] = call['call_identity']
    for row in artifact['payload']['agent_observations']:
        row['data']['call_identity'] = call['call_identity']
    artifact['payload_sha256'] = contracts.document_hash(artifact['payload'])
    for ref in call['response_provenance']['agent_receipt_refs'] + [call['response_provenance']['worker_receipt_ref']]:
        ref.update(run_id=call['run_id'], payload_sha256=artifact['payload_sha256'])
    example['worker_observation_artifacts'][call['run_id']] = artifact
    example['provider_calls'].append(call)
    return seal(example)


def records(example, **kwargs):
    splits, manifest = training_thinking.build([seal(example)], **kwargs)
    return splits['train'] + splits['eval'], manifest


def test_judge_targets_are_explicit_and_keep_source_icp_policy_and_outcome():
    example = judge_example()
    before = deepcopy(example)
    source, _ = records(example)
    judges, manifest = records(example, role='judge')
    assert len(source) == len(judges) == 1
    assert source[0]['source_refs'][0]['run_id'] == example['run_id']
    judge = judges[0]
    assert judge['role'] == manifest['role'] == 'judge'
    assert manifest['coverage']['generated_thinking_texts'] == 1
    assert {row['role'] for row in manifest['coverage']['by_role_model_route']} == {'judge'}
    assert judge['generated_thinking'][0]['text'] == 'Inspect the retained page, then answer.'
    ref = judge['source_refs'][0]
    assert ref['run_id'] == example['score_run_id']
    assert ref['judge_context']['scored_run_id'] == example['run_id']
    assert ref['judge_context']['icp_position'] == example['icp_position']
    assert ref['judge_context']['judgment_sha256'] == contracts.document_hash(example['judgment'])
    assert ref['judge_context']['scorer_policy_sha256'] == contracts.document_hash(example['configuration']['scorer_policy'])
    assert ref['outcome']['per_icp_score'] == 1
    assert judge['quality'] == {'status': 'unreviewed', 'correct_action': 'not_established'}
    assert example == before


@pytest.mark.parametrize('damage', ['marker', 'model_policy', 'failed_score', 'foreign_score', 'input_binding',
                                   'request_hash', 'route', 'summary', 'truncated', 'privacy'])
def test_unproved_judge_targets_are_excluded(damage):
    example = judge_example()
    call = example['provider_calls'][-1]
    if damage == 'marker':
        example['configuration']['scorer_policy']['env_bindings'].clear()
    elif damage == 'model_policy':
        example['configuration']['scorer_policy']['judge_models'] = {'industry_evidence_review': 'different/model'}
    elif damage == 'failed_score':
        example['attempts'][-1]['status'] = 'failed'
    elif damage == 'foreign_score':
        call['run_id'] = 'another-score'
    elif damage == 'input_binding':
        snapshot = example['input_snapshots'][example['score_run_id']]
        snapshot['input']['scored_run_id'] = 'another-execution'
        snapshot['input_sha256'] = contracts.document_hash(snapshot['input'])
    elif damage == 'request_hash':
        call['request_hash'] = 'sha256:' + 'f' * 64
    elif damage == 'route':
        call['reasoning_capture']['reviewed_route'] = False
    elif damage == 'summary':
        call['response']['choices'][0]['message']['reasoning_details'] = [
            {'type': 'reasoning.summary', 'summary': 'A summary only'}]
        call['response']['choices'][0]['message'].pop('reasoning_content')
    elif damage == 'truncated':
        call['response']['choices'][0]['finish_reason'] = 'length'
    else:
        call['request_capture'] = {'redacted': True}
    got, _ = records(example, role='judge')
    assert got == []


def test_judge_capture_uses_current_scorer_even_when_baseline_capture_is_legacy():
    example = judge_example()
    example['role'] = 'baseline'
    example['configuration']['baseline_reasoning_configuration'] = {}
    example['execution_reasoning_configuration'] = {}
    got, _ = records(example, role='judge')
    assert len(got) == 1


@pytest.mark.parametrize('damage', [None, 'body', 'hash', 'required', 'schema_name', 'other_order'])
def test_industry_wire_hash_survives_canonical_storage_without_rewriting_receipt(damage):
    import json
    from lab_arena import operations
    example = judge_example()
    call = example['provider_calls'][-1]
    body = call['request']['outbound']['body']
    body['response_format'] = {'type': 'json_schema', 'json_schema': {
        'name': 'industry_proposal', 'strict': True, 'schema': {
            'type': 'object', 'properties': {'z_fact': {'type': 'string'}, 'a_verdict': {'type': 'string'}},
            'required': ['z_fact', 'a_verdict'], 'additionalProperties': False}}}
    wire = operations.build_outbound_request(
        'openrouter.chat', body, allow_full_thinking=True, industry_schema_order=True,
        openrouter_provider_policy={'data_collection': 'deny', 'zdr': True, 'allow_fallbacks': False,
            'only': ['route-a'], 'require_parameters': True,
            'max_price': {'prompt': 1, 'completion': 1, 'request': 0}}).body
    # The private request object is canonically serialized on disk. The
    # original wire digest still proves the exact provider-visible JSON order.
    outbound = call['request']['outbound']
    outbound.update(body=json.loads(contracts.canonical_json(json.loads(wire))),
                    body_sha256=contracts.hash_bytes(wire))
    assert contracts.document_hash(outbound['body']) != outbound['body_sha256']
    if damage == 'body':
        outbound['body']['messages'][0]['content'] = 'Changed content'
    elif damage == 'hash':
        outbound['body_sha256'] = contracts.hash_bytes(b'wrong')
    elif damage == 'required':
        outbound['body']['response_format']['json_schema']['schema']['required'].reverse()
    elif damage == 'schema_name':
        outbound['body']['response_format']['json_schema']['name'] = 'other_schema'
    elif damage == 'other_order':
        arbitrary = dict(reversed(list(json.loads(wire).items())))
        outbound['body_sha256'] = contracts.hash_bytes(json.dumps(
            arbitrary, separators=(',', ':'), ensure_ascii=False).encode())
    seal(example)
    before = deepcopy(example)
    got, _ = records(example, role='judge')
    assert len(got) == (1 if damage is None else 0)
    assert example == before


def test_bad_role_fails_instead_of_silently_exporting_sources():
    with pytest.raises(ValueError, match='invalid_target_role'):
        records(judge_example(), role='all')


def test_judge_export_preserves_exact_tool_evidence_order_and_cli(tmp_path):
    import json
    from scripts.prepare_arena_full_thinking_dataset import main
    example = judge_example()
    call = example['provider_calls'][-1]
    body = call['request']['outbound']['body']
    body['messages'] += [
        {'role': 'assistant', 'content': None, 'tool_calls': [
            {'id': 'source-1', 'type': 'function', 'function': {'name': 'read', 'arguments': '{"url":"https://example.com"}'}}]},
        {'role': 'tool', 'tool_call_id': 'source-1', 'content': 'The factory makes steel valves.'}]
    body['tools'] = [{'type': 'function', 'function': {'name': 'read',
                      'parameters': {'type': 'object', 'properties': {'url': {'type': 'string'}}}}}]
    call['request']['outbound']['body_sha256'] = contracts.document_hash(body)
    got, _ = records(example, role='judge')
    assert len(got) == 1
    assert got[0]['input'] == body
    assert got[0]['input']['messages'][-1]['tool_call_id'] == 'source-1'
    path = tmp_path / 'example.json'
    path.write_text(json.dumps(seal(example)))
    output = tmp_path / 'judge-dataset'
    assert main(['--examples', str(path), '--output', str(output), '--role', 'judge']) == 0
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['role'] == 'judge'
    written = [json.loads(line) for name in ('train.jsonl', 'eval.jsonl')
               for line in (output / name).read_text().splitlines()]
    assert written == got


@pytest.mark.parametrize('damage', [None, 'claim', 'privacy', 'hash'])
def test_judge_journal_prefix_preserves_verified_pairs_without_claiming_completion(damage):
    from lab_arena import training_observations as obs
    example = judge_example()
    example['configuration']['training_maximum_missing_reasoning_percent'] = 1
    score_id = example['score_run_id']
    call = example['provider_calls'][-1]
    artifact = example['worker_observation_artifacts'].pop(score_id)
    claim = 'sha256:' + 'c' * 64
    artifact.update(schema_version=obs.JOURNAL_SCHEMA_VERSION, complete=False, lease_generation=1,
                    claim_request_hash=claim, segment_sha256=['sha256:' + 'd' * 64],
                    segment_refs=[obs.segment_ref(score_id, 1, claim, 0)])
    artifact['payload']['complete'] = False
    artifact['payload_sha256'] = contracts.document_hash(artifact['payload'])
    for ref in call['response_provenance']['agent_receipt_refs'] + [call['response_provenance']['worker_receipt_ref']]:
        ref['payload_sha256'] = artifact['payload_sha256']
    call['response_provenance']['worker_receipt_ref']['artifact_field'] = 'partial_worker_observation_artifacts'
    example['partial_worker_observation_artifacts'] = {score_id: artifact}
    example['input_snapshots'][score_id].update(run_id=score_id, kind='score', lease_generation=1,
                                              claim_request_hash=claim)
    example['completeness'].update(complete=False, incomplete_reasons=['worker_observations_incomplete'])
    if damage == 'claim':
        artifact['claim_request_hash'] = 'sha256:' + 'a' * 64
    elif damage == 'privacy':
        example['input_snapshots'][score_id]['privacy']['redacted'] = True
    elif damage == 'hash':
        artifact['payload_sha256'] = 'sha256:' + 'b' * 64
    got, _ = records(example, role='judge')
    assert len(got) == (1 if damage is None else 0)
    if got:
        assert got[0]['source_refs'][0]['source_quality']['complete'] is False
        assert got[0]['source_refs'][0]['worker_receipt_ref']['artifact_field'] == 'partial_worker_observation_artifacts'


def test_judge_export_uses_separate_frozen_registry():
    example = judge_example()
    configuration = example['configuration']
    configuration['judge_reasoning_route_registry'] = deepcopy(configuration['reasoning_route_registry'])
    configuration['reasoning_route_registry'] = {'unrelated': 'source registry must not replace judge authority'}
    got, manifest = records(example, role='judge')
    assert len(got) == 1
    assert manifest['coverage']['generated_thinking_texts'] == 1
