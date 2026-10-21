"""Offline observed next-response candidates. No trajectory or action-quality claim.

Only self-contained text/function chat and Responses calls are supported. A
positive run result or HTTP status never makes an observed action correct.
"""
from __future__ import annotations

import copy
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass

from lab_arena import contracts, integrity, training_capture, training_export, training_dataset

SCHEMA_VERSION = 'leadpoet.lab_arena.agent_step_dataset.v1'
RECORD_SCHEMA_VERSION = 'leadpoet.lab_arena.agent_step_candidate.v1'


@dataclass(frozen=True)
class Rules:
    eval_percent: int = 20
    seed: str = 'arena-agent-steps-v1'
    production_rounds_only: bool = True

    def __post_init__(self):
        if type(self.eval_percent) is not int or not 0 < self.eval_percent < 100:
            raise ValueError('invalid_eval_percent')
        if not isinstance(self.seed, str) or not self.seed:
            raise ValueError('invalid_seed')


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _exact(value):
    clean, removed = training_export.without_hidden_reasoning(value)
    sanitized, redactions = training_capture.sanitize_training_value(value)
    return (not removed and not redactions and clean == value and sanitized == value
            and not re.search(r'\[REDACTED\]|%5[Bb]REDACTED%5[Dd]', contracts.canonical_json(value)))


def _reject_constant(value):
    raise ValueError('nonfinite_json')


def _arguments(value):
    if not isinstance(value, str):
        return False
    try:
        return isinstance(json.loads(value, parse_constant=_reject_constant), dict)
    except (ValueError, TypeError):
        return False


def _tool_calls(calls):
    _require(isinstance(calls, list) and calls, 'tool_call_shape_unsupported')
    ids = set()
    for call in calls:
        _require(isinstance(call, dict) and set(call) <= {'id', 'type', 'function'}
                 and call.get('type') == 'function' and isinstance(call.get('id'), str)
                 and bool(call['id']) and call['id'] not in ids, 'tool_call_shape_unsupported')
        ids.add(call['id'])
        fn = call.get('function') or {}
        _require(set(fn) == {'name', 'arguments'} and isinstance(fn['name'], str)
                 and bool(fn['name']) and _arguments(fn['arguments']), 'tool_call_shape_unsupported')


def _chat_messages(messages):
    _require(isinstance(messages, list) and messages, 'input_shape_unsupported')
    pending = set()
    for msg in messages:
        _require(isinstance(msg, dict) and set(msg) <= {'role', 'content', 'name', 'tool_call_id', 'tool_calls'}
                 and msg.get('role') in {'system', 'developer', 'user', 'assistant', 'tool'}
                 and (isinstance(msg.get('content'), str) or msg.get('content') is None), 'input_shape_unsupported')
        if msg['role'] == 'tool':
            _require(msg.get('tool_call_id') in pending, 'tool_context_unresolved')
            pending.remove(msg['tool_call_id'])
        else:
            _require(not pending, 'tool_context_unresolved')
            if msg.get('tool_calls'):
                _require(msg['role'] == 'assistant', 'tool_context_unresolved')
                _tool_calls(msg['tool_calls'])
                pending = {item['id'] for item in msg['tool_calls']}
    _require(not pending, 'tool_context_unresolved')


def _responses_input(value):
    if isinstance(value, str):
        return
    _require(isinstance(value, list) and value, 'input_shape_unsupported')
    pending = set()
    for item in value:
        _require(isinstance(item, dict), 'input_shape_unsupported')
        kind = item.get('type', 'message')
        if kind == 'message':
            _require(not pending and set(item) <= {'type', 'role', 'content'}
                     and item.get('role') in {'system', 'developer', 'user', 'assistant'}, 'input_shape_unsupported')
            content = item.get('content')
            _require(isinstance(content, (str, list)), 'input_shape_unsupported')
            if isinstance(content, list):
                _require(all(isinstance(part, dict) and set(part) <= {'type', 'text'}
                             and part.get('type') in {'input_text', 'output_text'}
                             and isinstance(part.get('text'), str) for part in content), 'input_shape_unsupported')
        elif kind == 'function_call':
            _require(set(item) <= {'type', 'call_id', 'name', 'arguments', 'id', 'status'}
                     and all(isinstance(item.get(k), str) for k in ('call_id', 'name', 'arguments'))
                     and item['call_id'] and item['name'] and _arguments(item['arguments'])
                     and item['call_id'] not in pending, 'tool_context_unresolved')
            pending.add(item['call_id'])
        elif kind == 'function_call_output':
            _require(set(item) <= {'type', 'call_id', 'output'} and item.get('call_id') in pending
                     and isinstance(item.get('output'), str), 'tool_context_unresolved')
            pending.remove(item['call_id'])
        else:
            raise ValueError('input_shape_unsupported')
    _require(not pending, 'tool_context_unresolved')


def _target(body, response, operation):
    _require(isinstance(response, dict), 'response_shape_unsupported')
    reasoning = []
    if operation == 'openrouter.chat':
        _chat_messages(body.get('messages'))
        choices = response.get('choices')
        _require(isinstance(choices, list) and len(choices) == 1, 'response_choice_ambiguous')
        choice = choices[0]
        msg = copy.deepcopy(choice.get('message'))
        _require(isinstance(msg, dict) and msg.get('role') == 'assistant'
                 and choice.get('finish_reason') in {'stop', 'tool_calls'}, 'response_not_complete')
        for key in ('reasoning', 'reasoning_content'):
            if key in msg:
                _require(isinstance(msg[key], str), 'reasoning_shape_unsupported')
                reasoning.append({'field': key, 'text': msg.pop(key)})
        _require(set(msg) <= {'role', 'content', 'tool_calls', 'refusal'}
                 and not msg.get('refusal') and (isinstance(msg.get('content'), str) or msg.get('content') is None), 'assistant_shape_unsupported')
        msg.pop('refusal', None)
        if msg.get('tool_calls'):
            _require(choice['finish_reason'] == 'tool_calls', 'tool_finish_mismatch')
            _tool_calls(msg['tool_calls'])
        else:
            _require(choice['finish_reason'] == 'stop' and isinstance(msg.get('content'), str)
                     and bool(msg['content']), 'assistant_shape_unsupported')
        return msg, reasoning
    _responses_input(body.get('input'))
    _require(response.get('status') == 'completed' and not response.get('error')
             and not response.get('incomplete_details'), 'response_not_complete')
    output = response.get('output')
    _require(isinstance(output, list) and output, 'response_shape_unsupported')
    target = []
    for item in output:
        _require(isinstance(item, dict), 'response_shape_unsupported')
        kind = item.get('type')
        if kind == 'reasoning':
            _require(set(item) <= {'id', 'type', 'summary', 'status'}, 'reasoning_shape_unsupported')
            summary = item.get('summary', [])
            _require(isinstance(summary, list) and all(isinstance(part, dict)
                     and set(part) == {'type', 'text'} and part['type'] == 'summary_text'
                     and isinstance(part['text'], str) for part in summary), 'reasoning_shape_unsupported')
            reasoning.extend(summary)
        elif kind == 'message':
            _require(set(item) <= {'id', 'type', 'status', 'role', 'content'}
                     and item.get('role') == 'assistant' and item.get('status') == 'completed'
                     and isinstance(item.get('content'), list) and item['content'], 'assistant_shape_unsupported')
            _require(all(isinstance(part, dict) and set(part) <= {'type', 'text', 'annotations'}
                         and part.get('type') == 'output_text' and isinstance(part.get('text'), str)
                         and not part.get('annotations') for part in item['content']), 'assistant_shape_unsupported')
            target.append(copy.deepcopy(item))
        elif kind == 'function_call':
            _require(set(item) <= {'id', 'type', 'status', 'call_id', 'name', 'arguments'}
                     and item.get('status') == 'completed'
                     and all(isinstance(item.get(k), str) and item[k] for k in ('call_id', 'name', 'arguments'))
                     and _arguments(item['arguments']), 'tool_call_shape_unsupported')
            target.append(copy.deepcopy(item))
        else:
            raise ValueError('response_shape_unsupported')
    _require(bool(target) and (len(target) == 1 or all(item['type'] == 'function_call' for item in target)), 'response_choice_ambiguous')
    _require(len({item['call_id'] for item in target if item['type'] == 'function_call'})
             == sum(item['type'] == 'function_call' for item in target), 'tool_call_shape_unsupported')
    return target, reasoning


def _candidate(example, call):
    _require(call.get('run_id') == example.get('run_id'), 'not_main_execution_call')
    op = call.get('operation_id')
    _require(call.get('provider') == 'openrouter' and op in {'openrouter.chat', 'openrouter.responses'}, 'operation_unsupported')
    request = call.get('request') or {}
    _require(request.get('operation_id') == op and request.get('request_hash') == call.get('request_hash')
             and isinstance(request.get('hashed_parameters'), dict)
             and contracts.document_hash(request['hashed_parameters']) == call.get('request_hash'), 'request_identity_mismatch')
    _require(not (call.get('request_capture') or {}).get('redacted') and _exact(request), 'request_privacy_nonexact')
    _require(not set(call.get('evidence_gaps') or []) & {'full_request_unavailable', 'request_privacy_nonexact', 'worker_observation_request_mismatch', 'worker_observation_link_invalid'}, 'call_context_gap')
    _require(not set((example.get('completeness') or {}).get('incomplete_reasons') or []) & {'worker_response_receipt_link_invalid', 'agent_response_receipt_link_invalid', 'worker_observation_privacy_nonexact', 'agent_dependency_link_invalid', 'runtime_input_context_mismatch'}, 'worker_link_gap')
    outbound = request.get('outbound') or {}
    body = outbound.get('body')
    _require(isinstance(body, dict) and contracts.document_hash(body) == outbound.get('body_sha256'), 'outbound_body_hash_mismatch')
    _require(outbound.get('method') == 'POST' and outbound.get('host') == 'openrouter.ai'
             and outbound.get('path') == ('/api/v1/responses' if op.endswith('responses') else '/api/v1/chat/completions'), 'outbound_identity_mismatch')
    _require(isinstance(body.get('model'), str) and body['model']
             and body['model'] == (request.get('effective_parameters') or {}).get('model'), 'model_identity_mismatch')
    _require(not body.get('stream') and not body.get('previous_response_id')
             and not body.get('conversation') and not body.get('prompt')
             and not body.get('plugins') and not body.get('transforms') and body.get('n', 1) == 1, 'request_continuation_or_stream_unsupported')
    _require('instructions' not in body or isinstance(body['instructions'], str), 'input_shape_unsupported')
    tools = body.get('tools', [])
    _require(isinstance(tools, list) and all(isinstance(tool, dict) and tool.get('type') == 'function' for tool in tools), 'hosted_tool_unsupported')
    provenance = call.get('response_provenance') or {}
    _require(provenance.get('agent_receipt') == 'reported_exact_body_received'
             and provenance.get('delivery_to_worker') == 'socket_written', 'agent_read_unverified')
    snapshot = (example.get('worker_observation_artifacts') or {}).get(call['run_id']) or {}
    payload = snapshot.get('payload') or {}
    _require(contracts.document_hash(payload) == snapshot.get('payload_sha256'), 'worker_payload_hash_mismatch')
    refs = provenance.get('agent_receipt_refs') or []
    acks = [row for row in payload.get('agent_observations', []) if row.get('kind') == 'response_received'
            and any(ref.get('id') == row.get('id') and ref.get('run_id') == call['run_id']
                    and ref.get('payload_sha256') == snapshot.get('payload_sha256') for ref in refs)]
    _require(acks and all(not (row.get('privacy') or {}).get('redacted') for row in acks), 'agent_read_unverified')
    identities = {contracts.canonical_json(row.get('data')) for row in acks}
    _require(len(identities) == 1, 'agent_read_ambiguous')
    data = acks[0]['data']
    _require(data.get('call_identity') == call['call_identity'] and data.get('status') == 200, 'agent_read_identity_mismatch')
    worker_ref = provenance.get('worker_receipt_ref') or {}
    worker_receipts = [row for row in payload.get('response_receipts', [])
                       if row.get('ordinal') == worker_ref.get('ordinal')]
    _require(worker_ref.get('run_id') == call['run_id'] and worker_ref.get('payload_sha256') == snapshot.get('payload_sha256')
             and len(worker_receipts) == 1, 'worker_write_unverified')
    worker = worker_receipts[0]
    _require(worker.get('write_state') == 'socket_written' and worker.get('operation_id') == op
             and worker.get('action_sequence') == call.get('action_sequence')
             and all(worker.get(key) == data.get(key) for key in ('call_identity', 'status', 'body_sha256', 'body_bytes')), 'worker_write_identity_mismatch')
    receipts = call.get('response_receipts') or []
    matches = [r for r in receipts if r.get('entry_id') in provenance.get('worker_received_receipt_entry_ids', [])
               and r.get('wire_body_sha256') == data.get('body_sha256')
               and r.get('wire_body_bytes') == data.get('body_bytes') and (r.get('metadata') or {}).get('status') == 200]
    _require(bool(matches), 'delivered_response_unresolved')
    responses = []
    for receipt in matches:
        _require(set(receipt.get('privacy_redactions') or []) <= {'hidden_reasoning'} and not receipt.get('response_unavailable'), 'response_privacy_nonexact')
        value = receipt.get('response')
        if value is None:
            ref = receipt.get('response_ref') or {}
            if ref.get('field') == 'response' and ref.get('entry_id') == call.get('terminal_entry_id'):
                value = call.get('response')
            else:
                linked = [r for r in receipts if r.get('entry_id') == ref.get('entry_id') and 'response' in r]
                _require(len(linked) == 1, 'response_reference_unresolved')
                value = linked[0]['response']
        _require(contracts.document_hash(value) == receipt.get('response_sha256'), 'response_hash_mismatch')
        _require(_exact(value), 'response_privacy_nonexact')
        responses.append(value)
    _require(len({contracts.document_hash(value) for value in responses}) == 1, 'delivered_response_ambiguous')
    target, reasoning = _target(body, responses[0], op)
    definitions = [(tool.get('function') or {}) if op == 'openrouter.chat' else tool for tool in tools]
    _require(all(isinstance(fn, dict) and isinstance(fn.get('name'), str) and fn['name']
                 and isinstance(fn.get('parameters', {}), dict) for fn in definitions), 'tool_definition_unsupported')
    declared = {fn['name'] for fn in definitions}
    _require(len(declared) == len(definitions), 'tool_definition_unsupported')
    emitted = [item['function']['name'] for item in target.get('tool_calls', [])] if isinstance(target, dict) else [item['name'] for item in target if item['type'] == 'function_call']
    _require(all(name in declared for name in emitted), 'undeclared_tool_call')
    return {'schema_version': RECORD_SCHEMA_VERSION, 'format': op, 'input': copy.deepcopy(body),
            'target': target,
            'quality': {'status': 'unreviewed', 'correct_action': 'not_established'},
            'source_refs': [{'example_hash': example['example_hash'], 'run_id': call['run_id'],
                             'call_identity': call['call_identity'], 'teacher_reasoning_metadata': reasoning,
                             'response_projection': {'opaque_reasoning_omitted': any('hidden_reasoning' in (r.get('privacy_redactions') or []) for r in matches), 'exact_provider_replay': not any(r.get('privacy_redactions') for r in matches)}, 'attempt': example.get('attempt'),
                             'round_id': example.get('round_id'), 'model': body['model'],
                             'production': bool(training_dataset._PRODUCTION_ROUND_RE.fullmatch(str(example.get('round_id', '')))
                                                and example.get('round_status') == 'published'),
                             'outcome': {'per_icp_score': (example.get('judgment') or {}).get('per_icp_score'), 'score_run_id': example.get('score_run_id')},
                             'source_quality': copy.deepcopy(example.get('completeness') or {})}]}


def build(examples, rules=None):
    """Return deterministic candidate splits and an eligibility manifest."""
    rules = rules or Rules()
    candidates, analysis, sources = {}, [], set()
    clusters = training_dataset._Clusters()
    record_groups = {}
    counts = Counter()
    for example in examples:
        source = example.get('example_hash')
        sources.add(source if isinstance(source, str) and contracts.SHA256_RE.fullmatch(source) else 'invalid_example_hash')
        reason = None
        if example.get('schema_version') != training_export.SCHEMA_VERSION:
            reason = 'example_schema_unsupported'
        elif source != contracts.document_hash({k: v for k, v in example.items() if k != 'example_hash'}):
            reason = 'example_hash_mismatch'
        elif rules.production_rounds_only and (not training_dataset._PRODUCTION_ROUND_RE.fullmatch(str(example.get('round_id', ''))) or example.get('round_status') != 'published'):
            reason = 'not_a_published_production_round'
        if reason:
            counts[reason] += 1
            analysis.append({'source_example_hash': str(source) if isinstance(source, str) and contracts.SHA256_RE.fullmatch(source) else None, 'reason': reason})
            continue
        group_keys = ['episode:' + contracts.document_hash({k: example.get(k) for k in ('round_id', 'submission_id', 'stage', 'icp_position', 'assignment_id')})]
        if example.get('icp'):
            group_keys.append('icp:' + integrity.requirement_fingerprint(example['icp'], contacts_required=training_dataset._contacts(example.get('configuration') or {})))
        for company in example.get('companies') or []:
            if isinstance(company, dict):
                group_keys.extend('company:' + key for key in training_dataset._company_keys(company.get('company') or company, {}))
        for key in group_keys:
            clusters.find(key)
            clusters.union([group_keys[0], key])
        for call in example.get('provider_calls') or []:
            if not isinstance(call, dict):
                counts['malformed_call'] += 1
                analysis.append({'source_example_hash': source, 'call_ref_hash': None, 'reason': 'malformed_call'})
                continue
            try:
                record = _candidate(example, call)
                digest = contracts.document_hash({k: record[k] for k in ('format', 'input', 'target')})
                if digest in candidates:
                    candidates[digest]['source_refs'].extend(record['source_refs'])
                    clusters.union([record_groups[digest], group_keys[0]])
                else:
                    record['candidate_id'] = digest
                    candidates[digest] = record
                    record_groups[digest] = group_keys[0]
            except (ValueError, TypeError, KeyError, AttributeError) as exc:
                reason = str(exc) if isinstance(exc, ValueError) and re.fullmatch('[a-z_]+', str(exc)) else 'malformed_call'
                counts[reason] += 1
                analysis.append({'source_example_hash': source, 'call_ref_hash': contracts.document_hash(call.get('call_identity')), 'reason': reason})
    splits = {'train': [], 'eval': [], 'analysis': []}
    for digest, record in sorted(candidates.items()):
        record['source_refs'] = sorted({contracts.canonical_json(ref): ref for ref in record['source_refs']}.values(), key=contracts.canonical_json)
        group = clusters.find(record_groups[digest])
        split = 'eval' if int(contracts.document_hash({'seed': rules.seed, 'group': group}).split(':')[-1][:8], 16) % 100 < rules.eval_percent else 'train'
        splits[split].append(record)
    splits['analysis'] = sorted({contracts.canonical_json(row): row for row in analysis}.values(), key=contracts.canonical_json)
    manifest = {'schema_version': SCHEMA_VERSION, 'rules': asdict(rules),
                'training_permission': copy.deepcopy(training_dataset.TRAINING_PERMISSION),
                'eligibility_limits': ['observed_next_response_only', 'quality_unreviewed', 'not_a_complete_trajectory',
                                       'no_training_permission_confirmed', 'text_and_local_function_calls_only',
                                       'exact_agent_read_required', 'no_encrypted_reasoning'],
                'counts': {'examples': len(sources), 'candidate_executions': len({ref['run_id'] for r in candidates.values() for ref in r['source_refs']}),
                           'records_by_split': {k: len(v) for k, v in splits.items()}, 'exclusion_reasons': dict(sorted(counts.items()))},
                'sources': sorted(sources), 'split_hashes': {k: contracts.document_hash(v) for k, v in splits.items()}}
    manifest['dataset_version'] = contracts.document_hash(manifest)
    return splits, manifest
