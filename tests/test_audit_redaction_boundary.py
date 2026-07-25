"""RL1: the Langfuse redaction denylist must block generic LLM-payload keys
(messages/content/completion/...), which carry raw prompt/training data, so a
future call site can't silently ship them to the external sink."""
import pytest
from research_lab.observability.redaction import redact_for_langfuse, RedactionBlocked


@pytest.mark.parametrize("payload", [
    {"messages": [{"role": "system", "content": "secret ICP prompt"}]},
    {"content": "raw completion text"},
    {"completion": "..."},
    {"system_prompt": "..."},
    {"model_output": "..."},
    {"outer": {"content": "nested leak"}},
])
def test_llm_content_keys_blocked(payload):
    with pytest.raises(RedactionBlocked):
        redact_for_langfuse(payload)


def test_safe_metadata_still_passes():
    out = redact_for_langfuse({"run_id": "r1", "stage": "planner", "candidate_count": 3})
    assert out["run_id"] == "r1" and out["stage"] == "planner"
