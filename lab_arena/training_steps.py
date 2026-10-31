"""Offline observed next-response candidates. No trajectory or action-quality claim.

Self-contained text, local tools, and exposed hosted-search Responses are
supported. Hosted targets retain the provider-executed trajectory verbatim;
they do not imply local tool execution or invent intermediate model inputs.
A positive run result or HTTP status never makes an observed action correct.
"""
from __future__ import annotations

import copy
import base64
import json
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass

from lab_arena import contracts, operations, training_capture, training_export, training_dataset, training_coverage, reasoning_capture, training_worker_proof

SCHEMA_VERSION = 'leadpoet.lab_arena.agent_step_dataset.v1'
RECORD_SCHEMA_VERSION = 'leadpoet.lab_arena.agent_step_candidate.v1'


@dataclass(frozen=True)
class Rules:
    eval_percent: int = 20
    seed: str = training_dataset.DEFAULT_SPLIT_SEED
    production_rounds_only: bool = True
    eval_after: str | None = None
    eval_mode: str = 'company_and_icp'

    def __post_init__(self):
        if type(self.eval_percent) is not int or not 0 < self.eval_percent < 100:
            raise ValueError('invalid_eval_percent')
        if not isinstance(self.seed, str) or not self.seed:
            raise ValueError('invalid_seed')
        if self.eval_after is not None and training_dataset._date(self.eval_after) is None:
            raise ValueError('invalid_eval_after')
        if self.eval_mode not in {'company', 'company_and_icp'}:
            raise ValueError('invalid_eval_mode')


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _exact(value, *, response=False, worker_proofs=None):
    if type(worker_proofs) is training_worker_proof.WorkerProofs:
        return worker_proofs.exact_privacy(value, response=response, check=_exact_stateless)
    return _exact_stateless(value, response=response)


def _exact_stateless(value, *, response=False):
    def comparable(item):
        if isinstance(item, dict):
            return {key: comparable(child) for key, child in item.items()
                    if not (item.get('type') == 'reasoning'
                            and ((key in {'summary', 'content', 'text'}
                                  and (child is None or child == '' or child == []))
                                 or key == 'encrypted_content' and child is None))}
        if isinstance(item, list):
            return [comparable(child) for child in item]
        return item

    # Empty native optional fields contain no private bytes. Compare their
    # privacy projection without rewriting the retained request or response.
    value = comparable(value)
    if response and isinstance(value, dict):
        value = copy.deepcopy(value)
        for choice in value.get('choices', []) if isinstance(value.get('choices'), list) else []:
            msg = choice.get('message') if isinstance(choice, dict) else None
            if isinstance(msg, dict):
                for field in ('reasoning', 'reasoning_content', 'reasoning_details'):
                    if field in msg and (msg[field] is None or msg[field] == '' or field == 'reasoning_details' and msg[field] == []):
                        msg.pop(field)
    clean, removed = training_export.without_hidden_reasoning(value)
    sanitized, redactions = training_capture.sanitize_training_value(value)
    # A provider can return the marker literally. Callers verify capture and
    # receipt redaction flags plus request/response hashes separately.
    return not removed and not redactions and clean == value and sanitized == value


def _reject_constant(value):
    raise ValueError('nonfinite_json')


def _arguments(value):
    if not isinstance(value, str):
        return False
    try:
        return isinstance(json.loads(value, parse_constant=_reject_constant), dict)
    except (ValueError, TypeError, RecursionError):
        return False


def _tool_calls(calls, *, argument_validator=_arguments):
    _require(isinstance(calls, list) and calls, 'tool_call_shape_unsupported')
    ids = set()
    for call in calls:
        _require(isinstance(call, dict) and set(call) <= {'id', 'type', 'function', 'index'}
                 and call.get('type') == 'function' and isinstance(call.get('id'), str)
                 and bool(call['id']) and call['id'] not in ids, 'tool_call_shape_unsupported')
        _require('index' not in call or type(call['index']) is int and call['index'] >= 0,
                 'tool_call_shape_unsupported')
        ids.add(call['id'])
        fn = call.get('function') or {}
        _require(set(fn) == {'name', 'arguments'} and isinstance(fn['name'], str)
                 and bool(fn['name']) and argument_validator(fn['arguments']), 'tool_call_shape_unsupported')


def _chat_messages(messages, *, allow_reasoning_history=False, allow_tool_error_history=False, tools=()):
    _require(isinstance(messages, list) and messages, 'input_shape_unsupported')
    pending = set()
    seen, malformed_ids = set(), set()
    for msg in messages:
        allowed = {'role', 'content', 'name', 'tool_call_id', 'tool_calls'}
        if allow_reasoning_history:
            allowed |= {'reasoning', 'reasoning_content', 'reasoning_details', 'refusal'}
        _require(isinstance(msg, dict) and set(msg) <= allowed
                 and msg.get('role') in {'system', 'developer', 'user', 'assistant', 'tool'}
                 and (isinstance(msg.get('content'), str) or msg.get('content') is None), 'input_shape_unsupported')
        _require('refusal' not in msg or msg['role'] == 'assistant' and msg['refusal'] is None,
                 'input_shape_unsupported')
        if any(field in msg for field in ('reasoning', 'reasoning_content', 'reasoning_details')):
            _require(allow_reasoning_history and msg['role'] == 'assistant', 'reasoning_history_unsupported')
            for field in ('reasoning', 'reasoning_content'):
                _require(field not in msg or msg[field] is None or isinstance(msg[field], str),
                         'reasoning_history_unsupported')
            details = msg.get('reasoning_details')
            _require(details is None or isinstance(details, list) and all(
                training_capture.readable_reasoning_item(item) for item in details),
                'reasoning_history_unsupported')
        if msg['role'] == 'tool':
            _require(msg.get('tool_call_id') in pending, 'tool_context_unresolved')
            if allow_tool_error_history and msg['tool_call_id'] in malformed_ids:
                _require(isinstance(msg.get('content'), str), 'tool_context_unresolved')
            pending.remove(msg['tool_call_id'])
        else:
            _require(not pending, 'tool_context_unresolved')
            if msg.get('tool_calls'):
                _require(msg['role'] == 'assistant', 'tool_context_unresolved')
                # A prior failed call is inert input text. Only its paired
                # result closes the history; generated targets stay JSON-only.
                _tool_calls(msg['tool_calls'], argument_validator=(
                    lambda value: isinstance(value, str)) if allow_tool_error_history else _arguments)
                for item in msg['tool_calls']:
                    identifier, function = item['id'], item['function']
                    malformed = not _arguments(function['arguments'])
                    if allow_tool_error_history:
                        _require(identifier not in malformed_ids, 'tool_context_unresolved')
                        if malformed:
                            # Current declarations govern new targets, not
                            # closed failed calls retained from earlier turns.
                            _require(identifier not in seen, 'tool_context_unresolved')
                            malformed_ids.add(identifier)
                    seen.add(identifier)
                pending = {item['id'] for item in msg['tool_calls']}
    _require(not pending, 'tool_context_unresolved')


_HOSTED_SEARCH_TYPES = {'openrouter:web_search', 'web_search_call'}


def _hosted_search_declared(body):
    return any(isinstance(tool, dict) and tool.get('type') == 'openrouter:web_search'
               for tool in body.get('tools', []))


def _hosted_search_history(value):
    return isinstance(value, list) and any(
        isinstance(item, dict) and item.get('type') in _HOSTED_SEARCH_TYPES for item in value)


def _responses_input(value, *, tools=(), max_tool_calls=None, allow_reasoning_history=False, allow_tool_error_history=False):
    # Reuse the paid boundary's closed native schema. Keep the original values;
    # this validation must never normalize or rewrite captured request bytes.
    try:
        operations.validate_operation_request('openrouter.responses', {
            'model': 'training/input', 'input': value, 'tools': list(tools),
            **({'max_tool_calls': max_tool_calls} if max_tool_calls is not None else {}),
        }, allow_full_thinking=allow_reasoning_history)
    except operations.OperationError as exc:
        raise ValueError('input_shape_unsupported') from exc
    if isinstance(value, str):
        return
    pending = {}
    seen, malformed_ids = set(), set()
    if allow_tool_error_history:
        # Retain declaration conflict/shape checks even though historical
        # failed calls no longer need a declaration in this current turn.
        _response_tool_definitions({'input': value, 'tools': list(tools)})
    for item in value:
        kind = item.get('type', 'message')
        if kind == 'message':
            # One provider turn may expose a tool call followed by assistant
            # text. That text does not resolve the call; its matching output
            # must still arrive before any other role or the end of history.
            _require(not pending or item['role'] == 'assistant', 'tool_context_unresolved')
            content = item['content']
            if isinstance(content, list):
                _require(not any(part.get('annotations') for part in content)
                         or _hosted_search_declared({'tools': tools})
                         or _hosted_search_history(value), 'retrieval_context_unsupported')
        elif kind in {'function_call', 'custom_tool_call'}:
            _require(item['call_id'] not in pending, 'tool_context_unresolved')
            if allow_tool_error_history:
                _require(item['call_id'] not in malformed_ids, 'tool_context_unresolved')
            # Preserve prior failed-call text only inside a closed local-tool
            # exchange. This never changes the generated target validator.
            if kind == 'function_call' and not _arguments(item['arguments']):
                _require(allow_tool_error_history and item['call_id'] not in seen,
                         'tool_context_unresolved')
                # The paired call is historical input. Its tool can be absent
                # from this turn's declarations; new targets are checked below.
                malformed_ids.add(item['call_id'])
            seen.add(item['call_id'])
            pending[item['call_id']] = kind
        elif kind in {'function_call_output', 'custom_tool_call_output'}:
            expected = 'function_call' if kind == 'function_call_output' else 'custom_tool_call'
            _require(pending.get(item['call_id']) == expected, 'tool_context_unresolved')
            pending.pop(item['call_id'])
        elif kind == 'reasoning':
            _require(allow_reasoning_history and _exact(item), 'reasoning_history_unsupported')
        elif kind in _HOSTED_SEARCH_TYPES:
            # Native history is captured input, not a new execution request.
            # A later turn can remove the search declaration without removing
            # previously exposed provider events or inventing local results.
            _require(item.get('status') == 'completed', 'retrieval_context_unsupported')
        elif kind != 'additional_tools':
            raise ValueError('retrieval_context_unsupported')
    _require(not pending, 'tool_context_unresolved')


def _response_tool_definitions(body):
    """Resolve exact local declarations, including native additional tools."""
    declared = {}

    def collect(tools, namespace=None):
        for tool in tools:
            if tool['type'] == 'openrouter:web_search' and namespace is None:
                continue
            if tool['type'] == 'namespace':
                collect(tool['tools'], tool['name'])
                continue
            _require(tool['type'] in {'function', 'custom'}, 'hosted_tool_unsupported')
            key = (namespace, tool['name'])
            _require(key not in declared or declared[key] == tool, 'tool_definition_unsupported')
            declared[key] = tool

    collect(body.get('tools', []))
    if isinstance(body.get('input'), list):
        for item in body['input']:
            if item.get('type') == 'additional_tools':
                collect(item['tools'])
    return declared


def _empty_target_message(item):
    """Recognize empty scaffolding only after the ordinary item validation."""
    return item['type'] == 'message' and all(not part['text'].strip()
        and not part.get('annotations') and not part.get('logprobs') for part in item['content'])


def _target(body, response, operation, *, allow_reasoning_history=False, argument_validator=_arguments, allow_tool_error_history=False,
            allow_empty_messages=False):
    _require(isinstance(response, dict), 'response_shape_unsupported')
    _require(not any(response.get(field) for field in ('citations', 'search_results', 'annotations')),
             'retrieval_context_unsupported')
    reasoning = []
    if operation == 'openrouter.chat':
        _chat_messages(body.get('messages'), allow_reasoning_history=allow_reasoning_history,
                       allow_tool_error_history=allow_tool_error_history, tools=body.get('tools', []))
        choices = response.get('choices')
        _require(isinstance(choices, list) and len(choices) == 1, 'response_choice_ambiguous')
        choice = choices[0]
        msg = copy.deepcopy(choice.get('message'))
        _require(isinstance(msg, dict) and msg.get('role') == 'assistant'
                 and choice.get('finish_reason') in {'stop', 'tool_calls'}, 'response_not_complete')
        for key in ('reasoning', 'reasoning_content'):
            if key in msg:
                value = msg.pop(key)
                _require(value is None or isinstance(value, str), 'reasoning_shape_unsupported')
                if value:
                    reasoning.append({'field': key, 'text': value})
        details = msg.pop('reasoning_details', None)
        if details is not None:
            _require(isinstance(details, list) and all(training_capture.readable_reasoning_item(item) for item in details), 'reasoning_shape_unsupported')
            reasoning.extend(copy.deepcopy(details))
        _require(set(msg) <= {'role', 'content', 'tool_calls', 'refusal'}
                 and not msg.get('refusal') and (isinstance(msg.get('content'), str) or msg.get('content') is None), 'assistant_shape_unsupported')
        msg.pop('refusal', None)
        if msg.get('tool_calls'):
            _require(choice['finish_reason'] == 'tool_calls', 'tool_finish_mismatch')
            _tool_calls(msg['tool_calls'], argument_validator=argument_validator)
        else:
            _require(choice['finish_reason'] == 'stop' and isinstance(msg.get('content'), str)
                     and bool(msg['content']), 'assistant_shape_unsupported')
        return msg, reasoning
    _responses_input(body.get('input'), tools=body.get('tools', []),
                     max_tool_calls=body.get('max_tool_calls'),
                     allow_reasoning_history=allow_reasoning_history,
                     allow_tool_error_history=allow_tool_error_history)
    _require(response.get('status') == 'completed' and not response.get('error')
             and not response.get('incomplete_details'), 'response_not_complete')
    output = response.get('output')
    _require(isinstance(output, list) and output, 'response_shape_unsupported')
    hosted = _hosted_search_history(output)
    _require(not hosted or _hosted_search_declared(body), 'hosted_tool_undeclared')
    target = []
    for item in output:
        _require(isinstance(item, dict), 'response_shape_unsupported')
        kind = item.get('type')
        if kind == 'reasoning':
            _require(set(item) <= ({'id', 'type', 'summary', 'content', 'text', 'status'}
                                      | ({'format'} if allow_reasoning_history else set())), 'reasoning_shape_unsupported')
            _require('format' not in item or isinstance(item['format'], str)
                     and 1 <= len(item['format']) <= 128, 'reasoning_shape_unsupported')
            _require('id' not in item or isinstance(item['id'], str) and 1 <= len(item['id']) <= 256,
                     'reasoning_shape_unsupported')
            _require(item.get('status') in (None, 'completed'), 'response_not_complete')
            summary = item.get('summary', [])
            _require(isinstance(summary, list) and all(isinstance(part, dict)
                     and set(part) == {'type', 'text'} and part['type'] == 'summary_text'
                     and isinstance(part['text'], str) for part in summary), 'reasoning_shape_unsupported')
            reasoning.extend(summary)
            content = item.get('content', [])
            _require(isinstance(content, list) and all(isinstance(part, dict)
                     and set(part) == {'type', 'text'} and part['type'] == 'reasoning_text'
                     and isinstance(part['text'], str) for part in content), 'reasoning_shape_unsupported')
            reasoning.extend(content)
            if 'text' in item:
                _require(isinstance(item['text'], str), 'reasoning_shape_unsupported')
                reasoning.append({'type': 'reasoning_text', 'text': item['text']})
            if hosted:
                target.append(copy.deepcopy(item))
        elif kind == 'message':
            _require(set(item) <= {'id', 'type', 'status', 'role', 'content', 'phase'}
                     and item.get('role') == 'assistant' and item.get('status') == 'completed'
                     and isinstance(item.get('content'), list) and (item['content'] or allow_empty_messages), 'assistant_shape_unsupported')
            _require(item.get('phase') in (None, 'commentary', 'final_answer'), 'assistant_shape_unsupported')
            _require(all(isinstance(part, dict) and set(part) <= {'type', 'text', 'annotations', 'logprobs'}
                         and part.get('type') == 'output_text' and isinstance(part.get('text'), str)
                         and (not part.get('annotations') or _hosted_search_declared(body)
                              or _hosted_search_history(body.get('input')))
                         for part in item['content']), 'assistant_shape_unsupported')
            try:
                operations.validate_operation_request('openrouter.responses', {
                    'model': body['model'], 'input': [item],
                }, allow_full_thinking=allow_reasoning_history)
            except operations.OperationError as exc:
                raise ValueError('assistant_shape_unsupported') from exc
            target.append(copy.deepcopy(item))
        elif kind in _HOSTED_SEARCH_TYPES:
            try:
                operations.validate_operation_request('openrouter.responses', {
                    'model': body['model'], 'input': [item],
                }, allow_full_thinking=allow_reasoning_history)
            except operations.OperationError as exc:
                raise ValueError('hosted_tool_shape_unsupported') from exc
            _require(item.get('status') == 'completed', 'response_not_complete')
            target.append(copy.deepcopy(item))
        elif kind in {'function_call', 'custom_tool_call'}:
            try:
                # A target may have no output yet; schema validation is separate
                # from the closed input-history pairing check above.
                operations.validate_operation_request('openrouter.responses', {
                    'model': body['model'], 'input': [item],
                }, allow_full_thinking=allow_reasoning_history)
            except operations.OperationError as exc:
                raise ValueError('tool_call_shape_unsupported') from exc
            _require(item.get('status') == 'completed'
                     and (kind != 'function_call' or argument_validator(item['arguments'])),
                     'tool_call_shape_unsupported')
            target.append(copy.deepcopy(item))
        else:
            raise ValueError('response_shape_unsupported')
    call_kinds = {'function_call', 'custom_tool_call'}
    messages = [item for item in target if item['type'] == 'message'
                and (not allow_empty_messages or not _empty_target_message(item))]
    calls = [item for item in target if item['type'] in call_kinds]
    # A completed turn may state its next action and issue local tool calls.
    # Preserve their wire order. Hosted output is a provider-executed
    # trajectory and can contain intermediate messages around search events.
    # Ordinary turns still require one unambiguous next action.
    _require(bool(target) and (hosted or len(messages) <= 1)
             and not (calls and any(item.get('phase') == 'final_answer' for item in messages)),
             'response_choice_ambiguous')
    _require(len({item['call_id'] for item in target if item['type'] in call_kinds})
             == sum(item['type'] in call_kinds for item in target), 'tool_call_shape_unsupported')
    return target, reasoning


def _host_result(example, call, *, allow_privacy=False, allow_worker_error=False, worker_proofs=None):
    """Resolve saved gateway bytes against one existing host socket write.

    The caller keeps route, cost, privacy-admission and target-schema policy.
    Client read reports are optional only for the validated new frozen policy;
    an existing report must still agree. No consumption claim is synthesized.
    """
    return _host_response(example, call, write_state='socket_written',
                          allow_privacy=allow_privacy, allow_worker_error=allow_worker_error, worker_proofs=worker_proofs)


def _failed_host_result(example, call):
    """Validate a saved response whose host write failed; never a target/read."""
    return _host_response(example, call, write_state='socket_write_failed')


def _receipt_response(call, receipt):
    """Resolve one retained ledger response without changing its reference."""
    value = receipt.get('response')
    if value is None:
        target = receipt.get('response_ref') or {}
        if target.get('field') == 'response' and target.get('entry_id') == training_export.response_entry_id(call):
            value = call.get('response')
        else:
            linked = [r for r in call.get('response_receipts') or []
                      if r.get('entry_id') == target.get('entry_id') and 'response' in r]
            _require(len(linked) == 1, 'response_reference_unresolved')
            value = linked[0]['response']
    _require(contracts.document_hash(value) == receipt.get('response_sha256'), 'response_hash_mismatch')
    return value


def _host_response(example, call, *, write_state, allow_privacy=False, allow_worker_error=False,
                   require_client_ack=None, worker_proofs=None):
    provenance = call.get('response_provenance') or {}
    run_id = call.get('run_id')
    request = call.get('request') or {}
    capture = call.get('request_capture') or {}
    permitted_redactions = {'hidden_reasoning'} if call.get('operation_id') in {'openrouter.chat', 'openrouter.responses'} else set()
    if not allow_privacy:
        _require(not capture.get('redacted') and _exact(request, worker_proofs=worker_proofs), 'request_privacy_nonexact')
    _require(isinstance(request, dict) and request.get('operation_id') == call.get('operation_id')
             and request.get('request_hash') == call.get('request_hash'), 'request_identity_mismatch')
    if not (allow_privacy and capture.get('redacted') is True):
        _require(isinstance(request.get('hashed_parameters'), dict)
                 and contracts.document_hash(request['hashed_parameters']) == call.get('request_hash'),
                 'request_identity_mismatch')
    artifact = (example.get('worker_observation_artifacts') or {}).get(run_id) or {}
    from lab_arena import training_observations
    attempts = [row for row in example.get('attempts') or [] if row.get('run_id') == run_id]
    attempt = attempts[0] if len(attempts) == 1 else {}
    kind = 'execute' if run_id == example.get('run_id') else attempt.get('kind')
    _require(('run_id' not in artifact or artifact['run_id'] == run_id)
             and ('kind' not in artifact or artifact['kind'] == kind)
             and ('schema_version' not in artifact or artifact['schema_version'] in {
                 training_observations.SCHEMA_VERSION, training_observations.LEGACY_SCHEMA_VERSION})
             and ('lease_generation' not in artifact or type(artifact['lease_generation']) is int
                  and artifact['lease_generation'] > 0)
             and ('claim_request_hash' not in artifact or isinstance(artifact['claim_request_hash'], str)
                  and contracts.SHA256_RE.fullmatch(artifact['claim_request_hash']) is not None)
             and all(field not in artifact or field not in attempt or artifact[field] == attempt[field]
                     for field in ('lease_generation', 'claim_request_hash')),
             'worker_snapshot_identity_mismatch')
    payload = artifact.get('payload') or {}
    digest = training_worker_proof.payload_hash(payload, worker_proofs)
    _require(artifact.get('payload_sha256') == digest, 'worker_payload_hash_mismatch')
    ref = provenance.get('worker_receipt_ref') or {}
    writes = [row for row in payload.get('response_receipts') or [] if row.get('ordinal') == ref.get('ordinal')]
    _require(provenance.get('delivery_to_worker') == write_state
             and ref.get('run_id') == run_id and ref.get('payload_sha256') == digest
             and len(writes) == 1, 'worker_write_unverified')
    write = writes[0]
    _require(write.get('write_state') == write_state
             and (write_state != 'socket_write_failed' or write.get('socket_written_at') is None)
             and all(write.get(key) == call.get(key) for key in ('call_identity', 'operation_id', 'action_sequence'))
             and type(write.get('status')) is int and 100 <= write['status'] <= 599
             and type(write.get('body_bytes')) is int and write['body_bytes'] >= 0
             and isinstance(write.get('body_sha256'), str)
             and contracts.SHA256_RE.fullmatch(write['body_sha256']) is not None,
             'worker_write_identity_mismatch')
    receipts = call.get('response_receipts') or []
    matches = [receipt for receipt in receipts
               if receipt.get('entry_id') in (provenance.get('worker_received_receipt_entry_ids') or [])
               and receipt.get('wire_body_sha256') == write['body_sha256']
               and receipt.get('wire_body_bytes') == write['body_bytes']
               and (receipt.get('metadata') or {}).get('status') == write['status']]
    values = []
    for receipt in matches:
        _require(not receipt.get('response_unavailable'), 'delivered_response_unresolved')
        if not allow_privacy:
            _require(set(receipt.get('privacy_redactions') or []) <= permitted_redactions, 'response_privacy_nonexact')
        value = _receipt_response(call, receipt)
        if not allow_privacy:
            _require(_exact(value, response=True, worker_proofs=worker_proofs), 'response_privacy_nonexact')
        values.append(value)
    if not values and allow_worker_error:
        error_ref = provenance.get('worker_error_observation_ref') or {}
        _require(not any(row.get('wire_body_sha256') for row in receipts)
                 and error_ref.get('run_id') == run_id and error_ref.get('payload_sha256') == digest,
                 'delivered_response_unresolved')
        observations = [row for row in payload.get('observations') or [] if row.get('ordinal') == error_ref.get('ordinal')]
        _require(len(observations) == 1, 'delivered_response_unresolved')
        observed = observations[0]
        response = (observed.get('outcome') or {}).get('response') or {}
        _require((observed.get('privacy') or {}).get('redacted') is False
                 and all(observed.get(key) == call.get(key) for key in ('call_identity', 'operation_id', 'action_sequence'))
                 and response.get('status') == write['status'], 'worker_write_identity_mismatch')
        raw = base64.b64decode(response['body_b64'], validate=True)
        _require(len(raw) == write['body_bytes'] and contracts.hash_bytes(raw) == write['body_sha256'],
                 'response_hash_mismatch')
        _require((observed.get('request') or {}).get('admitted_parameters') == request.get('admitted_parameters'),
                 'request_identity_mismatch')
        values.append(json.loads(raw))
    _require(values and len({contracts.document_hash(value) for value in values}) == 1,
             'delivered_response_unresolved' if not values else 'delivered_response_ambiguous')
    # Optional cooperative telemetry cannot contradict the trusted result.
    refs = provenance.get('agent_receipt_refs') or []
    reads = [row for row in payload.get('agent_observations') or [] if row.get('kind') == 'response_received'
             and ((row.get('data') or {}).get('call_identity') == call.get('call_identity')
                  or any(ref.get('id') == row.get('id') for ref in refs))]
    if write_state == 'socket_write_failed':
        _require(not reads and not refs and provenance.get('agent_receipt') in (None, 'not_reported')
                 and provenance.get('selected_response_acknowledged') is not True
                 and provenance.get('worker_error_body_acknowledged') is not True, 'agent_read_unverified')
        return {'response': values[0], 'write': write, 'receipts': matches}
    required = (not training_coverage.uses_host_result_evidence(example.get('configuration') or {})
                if require_client_ack is None else require_client_ack)
    if required or reads or refs or provenance.get('agent_receipt') == 'reported_exact_body_received':
        _require(reads and provenance.get('agent_receipt') == 'reported_exact_body_received'
                 and len(refs) == len(reads), 'agent_read_unverified')
        for read in reads:
            _require((read.get('privacy') or {}).get('redacted') is False
                     and any(ref.get('id') == read.get('id') and ref.get('run_id') == run_id
                             and ref.get('payload_sha256') == digest for ref in refs)
                     and all((read.get('data') or {}).get(key) == write.get(key)
                             for key in ('call_identity', 'status', 'body_sha256', 'body_bytes')),
                     'agent_read_identity_mismatch')
    else:
        _require(provenance.get('agent_receipt') in (None, 'not_reported')
                 and provenance.get('selected_response_acknowledged') is not True
                 and provenance.get('worker_error_body_acknowledged') is not True, 'agent_read_unverified')
    if provenance.get('selected_response_acknowledged') is True:
        _require(training_export.response_entry_id(call) in {row['entry_id'] for row in matches}, 'agent_read_identity_mismatch')
    return {'response': values[0], 'write': write, 'receipts': matches}


def _validate_target_tools(body, target, op):
    tools = body.get('tools', [])
    if op == 'openrouter.chat':
        definitions = [(tool.get('function') or {}) for tool in tools]
        _require(all(isinstance(fn, dict) and isinstance(fn.get('name'), str) and fn['name']
                     and isinstance(fn.get('parameters', {}), dict) for fn in definitions), 'tool_definition_unsupported')
        declared = {fn['name'] for fn in definitions}
        _require(len(declared) == len(definitions), 'tool_definition_unsupported')
        _require(all(item['function']['name'] in declared for item in target.get('tool_calls', [])), 'undeclared_tool_call')
    else:
        declared = _response_tool_definitions(body)
        _require(all((item.get('namespace'), item['name']) in declared
                     and declared[(item.get('namespace'), item['name'])]['type'] == (
                         'function' if item['type'] == 'function_call' else 'custom')
                     for item in target if item['type'] in {'function_call', 'custom_tool_call'}), 'undeclared_tool_call')


def _outbound_request_exact(call):
    """Check the retained provider route and the existing outbound body hash."""
    request = call.get('request') or {}
    outbound = request.get('outbound') or {}
    operation = operations.get_operation(call.get('operation_id'))
    return (operation is not None and call.get('provider') == operation.provider
            and outbound.get('method') == operation.method
            and outbound.get('host') == (operation.outbound_host or operation.host)
            and outbound.get('path') == (operation.outbound_path or operations._render_path(
                operation, request.get('hashed_parameters') or {}))
            and outbound.get('body_sha256') == (
                contracts.hash_bytes(b'') if outbound.get('body') is None
                else contracts.document_hash(outbound['body'])))


def _model_request(example, call, *, worker_proofs=None):
    """Shared exact model request checks, independent of delivery or target."""
    _require(call.get('run_id') == example.get('run_id'), 'not_main_execution_call')
    op = call.get('operation_id')
    _require(call.get('provider') == 'openrouter' and op in {'openrouter.chat', 'openrouter.responses'}, 'operation_unsupported')
    request = call.get('request') or {}
    annotation = call.get('reasoning_capture') or {}
    if isinstance(annotation, Mapping) and annotation.get('policy') == reasoning_capture.POLICY:
        _require(reasoning_capture.explicit_provider_matches(request, annotation.get('provider_route')),
                 'request_provider_route_mismatch')
    _require(not (call.get('request_capture') or {}).get('redacted') and _exact(request, worker_proofs=worker_proofs), 'request_privacy_nonexact')
    _require(request.get('operation_id') == op and request.get('request_hash') == call.get('request_hash')
             and isinstance(request.get('hashed_parameters'), dict)
             and contracts.document_hash(request['hashed_parameters']) == call.get('request_hash'), 'request_identity_mismatch')
    _require(not set(call.get('evidence_gaps') or []) & {'full_request_unavailable', 'request_privacy_nonexact', 'worker_observation_request_mismatch', 'worker_observation_link_invalid'}, 'call_context_gap')
    if not training_coverage.uses_host_result_evidence(example.get('configuration') or {}):
        _require(not set((example.get('completeness') or {}).get('incomplete_reasons') or []) & {'worker_response_receipt_link_invalid', 'agent_response_receipt_link_invalid', 'agent_dependency_link_invalid', 'runtime_input_context_mismatch'}, 'worker_link_gap')
    outbound = request.get('outbound') or {}
    body = outbound.get('body')
    _require(isinstance(body, dict) and contracts.document_hash(body) == outbound.get('body_sha256'), 'outbound_body_hash_mismatch')
    _require(outbound.get('method') == 'POST' and outbound.get('host') == 'openrouter.ai'
             and outbound.get('path') == ('/api/v1/responses' if op.endswith('responses') else '/api/v1/chat/completions'), 'outbound_identity_mismatch')
    _require(isinstance(body.get('model'), str) and body['model']
             and body['model'] == (request.get('effective_parameters') or {}).get('model'), 'model_identity_mismatch')
    _require(not body.get('stream') and not body.get('previous_response_id')
             and not body.get('conversation') and not body.get('prompt')
             and not body.get('plugins') and not body.get('transforms')
             and body.get('web_search_options') is None and body.get('n', 1) == 1, 'request_continuation_or_stream_unsupported')
    _require('instructions' not in body or isinstance(body['instructions'], str), 'input_shape_unsupported')
    tools = body.get('tools', [])
    _require(isinstance(tools, list) and all(isinstance(tool, dict) for tool in tools), 'hosted_tool_unsupported')
    if op == 'openrouter.chat':
        _require(all(tool.get('type') == 'function' for tool in tools), 'hosted_tool_unsupported')
    return op, body, tools


def _candidate(example, call, *, host_observed=False, worker_proofs=None):
    op, body, tools = _model_request(example, call, worker_proofs=worker_proofs)
    request = call.get('request') or {}
    _require(set(call.get('response_privacy_redactions') or []) <= {'hidden_reasoning'}
             and 'response_privacy_nonexact' not in (call.get('evidence_gaps') or []),
             'response_privacy_nonexact')
    provider_visible = host_observed or training_coverage.uses_host_result_evidence(
        example.get('configuration') or {})
    if provider_visible:
        # Generic observed steps prove only the exact provider-visible pair.
        # Strict policy callers keep their existing frozen read-proof rules.
        evidence = (_host_response(example, call, write_state='socket_written', require_client_ack=False, worker_proofs=worker_proofs)
                    if host_observed else _host_result(example, call, worker_proofs=worker_proofs))
        _require(evidence['write']['status'] == 200, 'agent_read_identity_mismatch')
        responses = [evidence['response']]
        matches = evidence['receipts']
    else:
        provenance = call.get('response_provenance') or {}
        _require(provenance.get('agent_receipt') == 'reported_exact_body_received'
                 and provenance.get('delivery_to_worker') == 'socket_written', 'agent_read_unverified')
        snapshot = (example.get('worker_observation_artifacts') or {}).get(call['run_id']) or {}
        payload = snapshot.get('payload') or {}
        _require(training_worker_proof.payload_hash(payload, worker_proofs) == snapshot.get('payload_sha256'), 'worker_payload_hash_mismatch')
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
                if ref.get('field') == 'response' and ref.get('entry_id') == training_export.response_entry_id(call):
                    value = call.get('response')
                else:
                    linked = [r for r in receipts if r.get('entry_id') == ref.get('entry_id') and 'response' in r]
                    _require(len(linked) == 1, 'response_reference_unresolved')
                    value = linked[0]['response']
            _require(contracts.document_hash(value) == receipt.get('response_sha256'), 'response_hash_mismatch')
            _require(_exact(value, response=True, worker_proofs=worker_proofs), 'response_privacy_nonexact')
            responses.append(value)
        _require(len({contracts.document_hash(value) for value in responses}) == 1, 'delivered_response_ambiguous')
    annotation = call.get('reasoning_capture') or {}
    # Readable retained context is observed input, not a full-thinking claim.
    # Strict callers keep the annotation-bound parser selection unchanged.
    full_history = host_observed or (isinstance(annotation, dict)
                    and annotation.get('policy') == 'full_generated_thinking_v1'
                    and annotation.get('requested') is True
                    and annotation.get('reviewed_route') is True)
    target, reasoning = _target(body, responses[0], op,
                                allow_reasoning_history=full_history,
                                allow_tool_error_history=training_coverage.uses_host_result_evidence(
                                    example.get('configuration') or {}))
    _validate_target_tools(body, target, op)
    target_provenance = {}
    if op == 'openrouter.responses':
        hosted = any(item.get('type') in _HOSTED_SEARCH_TYPES for item in responses[0]['output'])
        target_provenance['target_source_paths'] = [
            ['output', index] for index, item in enumerate(responses[0]['output'])
            if hosted or item.get('type') != 'reasoning']
        if hosted:
            target_provenance.update(target_semantics='provider_executed_trajectory',
                                     hosted_tool_execution='provider')
    return {'schema_version': RECORD_SCHEMA_VERSION, 'format': op, 'input': copy.deepcopy(body),
            'target': target,
            'quality': {'status': 'unreviewed', 'correct_action': 'not_established'},
            'source_refs': [{'example_hash': example['example_hash'], 'run_id': call['run_id'],
                             'call_identity': call['call_identity'], 'teacher_reasoning_metadata': reasoning,
                             **({'capture_scope': 'provider_visible_request_result', 'client_consumption': 'not_proved'}
                                if provider_visible else {}),
                             **target_provenance,
                             'delivered_response_sha256': contracts.document_hash(responses[0]),
                             'response_projection': {'opaque_reasoning_omitted': any('hidden_reasoning' in (r.get('privacy_redactions') or []) for r in matches), 'exact_provider_replay': not any(r.get('privacy_redactions') for r in matches)}, 'attempt': example.get('attempt'),
                             'round_id': example.get('round_id'), 'model': body['model'],
                             'production': bool(training_dataset._PRODUCTION_ROUND_RE.fullmatch(str(example.get('round_id', '')))
                                                and example.get('round_status') == 'published'),
                             'outcome': {'per_icp_score': (example.get('judgment') or {}).get('per_icp_score'), 'score_run_id': example.get('score_run_id')},
                             'source_quality': copy.deepcopy(example.get('completeness') or {})}]}


def _build(examples, rules=None, *, host_observed=False, full_thinking_candidate=None):
    """Keep strict policy evidence separate from generic host observations."""
    rules = rules or Rules()
    candidates, analysis, sources = {}, [], set()
    source_groups = training_dataset._SourceSplitGroups()
    record_groups = {}
    counts = Counter()
    for example in examples:
        episode = source_groups.add(example, production_rounds_only=rules.production_rounds_only,
                                    eval_mode=rules.eval_mode)
        source = example.get('example_hash')
        sources.add(source if isinstance(source, str) and contracts.SHA256_RE.fullmatch(source) else 'invalid_example_hash')
        reason = None
        if example.get('schema_version') != training_export.SCHEMA_VERSION:
            reason = 'example_schema_unsupported'
        elif source != contracts.document_hash({k: v for k, v in example.items() if k != 'example_hash'}):
            reason = 'example_hash_mismatch'
        elif rules.production_rounds_only and (not training_dataset._PRODUCTION_ROUND_RE.fullmatch(str(example.get('round_id', ''))) or example.get('round_status') != 'published'):
            reason = 'not_a_published_production_round'
        elif episode is None:
            reason = 'episode_identity_unverified'
        if reason:
            counts[reason] += 1
            analysis.append({'source_example_hash': str(source) if isinstance(source, str) and contracts.SHA256_RE.fullmatch(source) else None, 'reason': reason})
            del example
            continue
        for call in example.get('provider_calls') or []:
            if not isinstance(call, dict):
                counts['malformed_call'] += 1
                analysis.append({'source_example_hash': source, 'call_ref_hash': None, 'reason': 'malformed_call'})
                continue
            try:
                record = (_candidate(example, call, host_observed=True) if host_observed
                          else _candidate(example, call))
                digest = contracts.document_hash({k: record[k] for k in ('format', 'input', 'target')})
                if digest in candidates:
                    candidates[digest]['source_refs'].extend(record['source_refs'])
                    record_groups[digest].add(episode)
                else:
                    record['candidate_id'] = digest
                    candidates[digest] = record
                    record_groups[digest] = {episode}
            except (ValueError, TypeError, KeyError, AttributeError) as exc:
                reason = str(exc) if isinstance(exc, ValueError) and re.fullmatch('[a-z_]+', str(exc)) else 'malformed_call'
                counts[reason] += 1
                analysis.append({'source_example_hash': source, 'call_ref_hash': contracts.document_hash(call.get('call_identity')), 'reason': reason})
            else:
                if full_thinking_candidate is not None:
                    full_thinking_candidate(example, record)
        # Drop the raw source and last raw call before parsing the next episode.
        call = None
        del example
    splits = {'train': [], 'eval': [], 'analysis': []}
    future_groups = source_groups.future_groups(rules.eval_after)
    for digest, record in sorted(candidates.items()):
        record['source_refs'] = sorted({contracts.canonical_json(ref): ref for ref in record['source_refs']}.values(), key=contracts.canonical_json)
        assignments = {source_groups.assignment(episode, seed=rules.seed,
                                                 eval_percent=rules.eval_percent,
                                                 eval_after=rules.eval_after,
                                                 future_groups=future_groups)
                       for episode in record_groups[digest]}
        if len(assignments) != 1:
            counts['source_split_conflict'] += 1
            analysis.append({'source_example_hash': None, 'call_ref_hash': digest,
                             'reason': 'source_split_conflict'})
            continue
        split, split_reason = next(iter(assignments))
        if split_reason:
            counts[split_reason] += 1
            analysis.append({'source_example_hash': None, 'call_ref_hash': digest,
                             'reason': split_reason})
            continue
        splits[split].append(record)
    splits['analysis'] = sorted({contracts.canonical_json(row): row for row in analysis}.values(), key=contracts.canonical_json)
    manifest_rules = asdict(rules)
    manifest_rules['split_grouping'] = training_dataset.SPLIT_GROUPING
    manifest = {'schema_version': SCHEMA_VERSION, 'rules': manifest_rules,
                'training_permission': copy.deepcopy(training_dataset.TRAINING_PERMISSION),
                'eligibility_limits': ['observed_next_response_only', 'quality_unreviewed', 'not_a_complete_trajectory',
                                       'no_training_permission_confirmed', 'text_local_tools_and_exposed_hosted_search_only',
                                       'hosted_internal_prompts_not_observed',
                                       *(['host_delivery_required', 'client_consumption_not_proved'] if host_observed else
                                         ['host_delivery_required_for_new_policy', 'agent_read_required_for_historical_policy']),
                                       'no_encrypted_reasoning'],
                'counts': {'examples': len(sources), 'candidate_executions': len({ref['run_id'] for r in candidates.values() for ref in r['source_refs']}),
                           'records_by_split': {k: len(v) for k, v in splits.items()}, 'exclusion_reasons': dict(sorted(counts.items()))},
                'sources': sorted(sources),
                'split_source_hashes': sorted(source_groups.source_hashes),
                'split_hashes': {k: contracts.document_hash(v) for k, v in splits.items()}}
    manifest['dataset_version'] = contracts.document_hash(manifest)
    return splits, manifest


def build(examples, rules=None):
    """Export exact host-observed pairs without asserting client consumption."""
    return _build(examples, rules, host_observed=True)
