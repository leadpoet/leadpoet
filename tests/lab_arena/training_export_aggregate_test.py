"""Whole examples have a bounded privacy budget separate from each request."""

import base64
import copy
import json

import pytest

from lab_arena import training_capture, training_export, training_input
from tests.lab_arena.training_export_test import add_call, assert_legacy_agent_gaps, fixture, save_input


def large_example(total_calls=221):
    values = fixture()
    # The live failure joined a 62k-node frozen catalog, repeated input
    # snapshots and 221 calls (161 execution plus 60 judgment calls).
    catalog = [{"name": str(index), "description": "public tool"} for index in range(20_800)]
    values["round_row"]["configuration_doc"]["tool_catalog"] = catalog
    for run in values["runs"]:
        document = training_input.read_input(
            values["objects"], run["run_id"], run["kind"],
            run["lease_generation"], run["claim_request_hash"],
        )["input"]
        document["tool_catalog"] = catalog
        save_input(values["objects"], run["run_id"], run["kind"], document)
    add_call(values)
    template = values["ledger_rows"]
    rows = []
    for index in range(total_calls):
        run = values["runs"][int(index >= 161)]
        for entry in copy.deepcopy(template):
            entry.update({key: run[key] for key in
                          ("run_id", "round_id", "submission_id", "miner_hotkey")})
            entry["entry_id"] += index * 3
            entry["call_identity"] = "sha256:" + format(index + 1, "064x")
            entry["entry_doc"]["action_sequence"] = index if index < 161 else index - 161
            entry["entry_doc"]["base_call_identity"] = entry["call_identity"]
            if "terminal_response" in entry:
                entry["terminal_response"]["body_b64"] = base64.b64encode(
                    json.dumps({"results": ["public page"] * 400}).encode()
                ).decode()
            rows.append(entry)
    values["ledger_rows"] = rows
    for run, count in zip(values["runs"], (161, total_calls - 161)):
        run["result_doc"]["resource_summary"]["provider_call_count"] = count
    return values


@pytest.mark.parametrize("total_calls", [221, 2636, 4000])
def test_valid_large_aggregate_exports_without_relaxing_request_limit(total_calls):
    values = large_example(total_calls)
    result = training_export.assemble_example(**values)
    assert_legacy_agent_gaps(result)
    assert len(result["provider_calls"]) == total_calls
    assert len(result["configuration"]["tool_catalog"]) == 20_800
    # The original final pass rejects this exact valid assembled structure.
    with pytest.raises(training_capture.CaptureError, match="request_structure_limit"):
        training_capture.sanitize_training_value(result)
    for snapshot in result["input_snapshots"].values():
        training_capture.sanitize_training_value(snapshot)
    assert result == training_export.assemble_example(**values)


def test_aggregate_budget_is_still_finite():
    values = fixture()
    values["round_row"]["configuration_doc"]["oversized"] = [None] * training_export.MAX_EXAMPLE_NODES
    with pytest.raises(training_capture.CaptureError, match="request_structure_limit"):
        training_export.assemble_example(**values)


def test_individual_request_structure_limit_remains_unchanged():
    value = {"items": [None] * training_capture.MAX_VALUE_NODES}
    with pytest.raises(training_capture.CaptureError, match="request_structure_limit"):
        training_capture.sanitize_training_value(value)
    assert training_capture.RequestCapture(fixture()["objects"])(value)["status"] == "unavailable"
    terminal = {"body_b64": base64.b64encode(json.dumps(value).encode()).decode()}
    with pytest.raises(training_capture.CaptureError, match="request_structure_limit"):
        training_export._response_value(terminal, "openrouter")


@pytest.mark.parametrize("invalid", [{1: "non-string key"}, {"value": object()}])
def test_aggregate_still_rejects_malformed_values(invalid):
    values = fixture()
    values["round_row"]["configuration_doc"]["invalid"] = invalid
    with pytest.raises(training_capture.CaptureError, match="request_(key|value)_invalid"):
        training_export.assemble_example(**values)


def test_aggregate_keeps_depth_limit_and_private_field_redaction():
    nested = None
    for _ in range(65):
        nested = {"child": nested}
    with pytest.raises(training_capture.CaptureError, match="request_structure_limit"):
        training_capture.sanitize_training_value(nested, max_nodes=training_export.MAX_EXAMPLE_NODES)
    values = large_example()
    values["round_row"]["configuration_doc"]["private"] = {
        "authorization": "private-credential-sentinel",
        "reasoning_content": "private-reasoning-sentinel",
    }
    result = training_export.assemble_example(**values)
    assert result["configuration"]["private"] == {"authorization": "[REDACTED]"}
    assert "private-reasoning-sentinel" not in json.dumps(result)
    assert result["privacy"]["context_redactions"] == ["credential", "hidden_reasoning"]
    assert result["privacy"]["privacy_filtered"] and not result["privacy"]["exact_replay"]


def test_joined_calls_and_repeated_snapshots_have_separate_finite_budgets():
    value = {
        'configuration': {'public': [None] * 20},
        'provider_calls': [{'public': [None] * 60}, {'public': [None] * 60}],
        'judgment': {'signal_source_provider_calls': [
            {'public': [None] * 60}, {'public': [None] * 60}]},
        'input_snapshots': {'execute': {'public': [None] * 60},
                            'score-1': {'public': [None] * 60},
                            'score-2': {'public': [None] * 60}},
    }
    original = copy.deepcopy(value)
    with pytest.raises(training_capture.CaptureError, match='request_structure_limit'):
        training_capture.sanitize_training_value(value, max_nodes=100)
    cleaned, redactions = training_capture.sanitize_training_value(
        value, max_nodes=100, aggregate_parts=True)
    assert cleaned == value == original and redactions == []


@pytest.mark.parametrize('part', ['call', 'source_call', 'snapshot', 'envelope', 'nested_call',
                                  'nested_snapshot', 'nested_source_call', 'source_call_mapping'])
def test_partitioning_cannot_escape_any_part_or_global_context_budget(part):
    oversized = {'public': [None] * 100}
    value = {
        'call': {'provider_calls': [oversized]},
        'source_call': {'judgment': {'signal_source_provider_calls': [oversized]}},
        'snapshot': {'input_snapshots': {'run': oversized}},
        'envelope': {'configuration': oversized},
        'nested_call': {'configuration': {'provider_calls': [{'public': [None] * 60}] * 2}},
        'nested_snapshot': {'provider_calls': [{'input_snapshots': {
            'first': {'public': [None] * 60}, 'second': {'public': [None] * 60}}}]},
        'nested_source_call': {'provider_calls': [{'judgment': {'signal_source_provider_calls': [
            {'public': [None] * 60}, {'public': [None] * 60}]}}]},
        'source_call_mapping': {'judgment': {'signal_source_provider_calls': {
            '0': {'public': [None] * 60}, '1': {'public': [None] * 60}}}},
    }[part]
    with pytest.raises(training_capture.CaptureError, match='request_structure_limit'):
        training_capture.sanitize_training_value(value, max_nodes=100, aggregate_parts=True)


@pytest.mark.parametrize('part', ['call', 'source_call', 'snapshot', 'envelope'])
def test_partitioning_still_checks_keys_values_collisions_and_depth(part):
    def wrap(value):
        return {'call': {'provider_calls': [value]},
                'source_call': {'judgment': {'signal_source_provider_calls': [value]}},
                'snapshot': {'input_snapshots': {'run': value}},
                'envelope': {'configuration': value}}[part]
    for invalid, reason in (({1: 'bad key'}, 'request_key_invalid'),
                            ({'public': object()}, 'request_value_invalid'),
                            ({'private-token-value': 1, '[REDACTED]': 2}, 'request_redaction_key_collision')):
        with pytest.raises(training_capture.CaptureError, match=reason):
            training_capture.sanitize_training_value(wrap(invalid), secrets=['private-token-value'],
                                                     aggregate_parts=True)
    nested = None
    for _ in range(65):
        nested = {'child': nested}
    with pytest.raises(training_capture.CaptureError, match='request_structure_limit'):
        training_capture.sanitize_training_value(wrap(nested), aggregate_parts=True)


def test_partitioned_privacy_walk_keeps_full_root_path_rules_and_redaction_classes():
    parameters = {'reasoning': {'effort': 'high'}, 'messages': [
        {'role': 'assistant', 'reasoning_content': 'Exposed assistant history.'}]}
    request = {'operation_id': 'openrouter.chat',
               **{key: copy.deepcopy(parameters) for key in
                  ('admitted_parameters', 'normalized_parameters', 'effective_parameters', 'hashed_parameters')},
               'outbound': {'body': copy.deepcopy(parameters)}}
    value = {
        'provider_calls': [{'operation_id': 'openrouter.chat', 'request': request,
            'response': {'choices': [{'message': {'role': 'assistant',
                'reasoning_content': 'Exposed provider thinking.',
                'reasoning_details': [{'type': 'reasoning.text', 'text': 'Exposed typed text.'}]}}]},
            'private': {'authorization': 'private-token-value'}}],
        'input_snapshots': {'run': {'input': {'reasoning_content': 'unapproved opaque context',
                                             'public': 'contains private-token-value'}}},
        'configuration': {'private': {'authorization': 'private-token-value'},
                          'provider_calls': [{'request': {'outbound': {'body': parameters}}}]},
    }
    original = copy.deepcopy(value)
    expected = training_capture.sanitize_training_value(value, secrets=['private-token-value'])
    actual = training_capture.sanitize_training_value(value, secrets=['private-token-value'], aggregate_parts=True)
    assert actual == expected
    cleaned, redactions = actual
    assert redactions == ['credential', 'hidden_reasoning']
    assert cleaned['provider_calls'][0]['request'] == request
    assert cleaned['provider_calls'][0]['response'] == value['provider_calls'][0]['response']
    assert cleaned['input_snapshots']['run']['input']['reasoning_content'] == '[REDACTED]'
    assert 'private-token-value' not in json.dumps(cleaned)
    assert value == original
