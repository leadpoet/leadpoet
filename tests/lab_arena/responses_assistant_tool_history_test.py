"""Exposed assistant messages can precede the outputs of pending local tools."""

from copy import deepcopy

import pytest

from lab_arena import training_steps


def history():
    return [
        {"role": "user", "content": "Find company evidence."},
        {"type": "reasoning", "id": "r1", "summary": [], "content": [
            {"type": "reasoning_text", "text": "Read the company profile first."}]},
        {"type": "function_call", "call_id": "lookup-1", "name": "lookup_evidence", "arguments": "{}"},
        {"type": "message", "role": "assistant", "status": "completed", "content": [
            {"type": "output_text", "text": "Waiting for the profile.", "annotations": [], "logprobs": []}]},
        {"type": "function_call_output", "call_id": "lookup-1", "output": "Company profile."},
    ]


def test_assistant_message_does_not_close_pending_call_or_change_history():
    value = history()
    before = deepcopy(value)
    training_steps._responses_input(value, allow_reasoning_history=True)
    assert value == before


@pytest.mark.parametrize("damage", ["unresolved", "orphan", "duplicate_output", "wrong_tool_type",
                                    "user", "system", "developer"])
def test_assistant_interleave_retains_pending_call_and_role_checks(damage):
    value = history()
    if damage == "unresolved": value.pop()
    elif damage == "orphan": value[-1]["call_id"] = "unknown"
    elif damage == "duplicate_output": value.append(deepcopy(value[-1]))
    elif damage == "wrong_tool_type": value[-1]["type"] = "custom_tool_call_output"
    else: value[3] = {"type": "message", "role": damage, "content": "Do not cross a pending call."}
    with pytest.raises(ValueError, match="tool_context_unresolved"):
        training_steps._responses_input(value, allow_reasoning_history=True)


def test_plain_assistant_tool_history_does_not_require_thinking_policy():
    value = [item for item in history() if item.get("type") != "reasoning"]
    value[2] = {"role": "assistant", "content": "Waiting for the profile."}
    before = deepcopy(value)
    training_steps._responses_input(value)
    assert value == before
