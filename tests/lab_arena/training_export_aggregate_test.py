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


@pytest.mark.parametrize("total_calls", [221, 2636])
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
