"""Verify the request scope; actual model accuracy needs provider tests."""
import asyncio
import json

import pytest

from qualification.scoring import intent_verification_three_stage as intent
from qualification.scoring.lead_scorer import required_intent_satisfied
from tests.test_intent_three_stage_json_retry import (
    _Client, _Response, _completion, _install_call_fakes,
)


PRIMARY = "Launched a product that customers can use now."
BONUS = "Announced a partnership with a data provider."
REQUEST = f"Required primary intent: {PRIMARY}\nOptional bonus intent: {BONUS}"


@pytest.mark.parametrize("target", [PRIMARY, BONUS])
@pytest.mark.parametrize("final", [False, True])
def test_every_verifier_request_judges_only_its_bound_target(monkeypatch, target, final):
    _install_call_fakes(monkeypatch)
    row = {"id": "signal-1", "company": "Example", "website": "https://example.test",
           "claim": "A supplier announced a partnership.",
           "claimed_source_urls": ["https://example.test/news"],
           "_target_signal_text": target, "_buyer_request_context": REQUEST}
    prompt = (intent._build_final_judge_prompt(row, {"results": [{
        "url": row["claimed_source_urls"][0], "text": "Exact source body.",
    }]}) if final else intent._build_verification_prompt(row))
    client = _Client([_Response(_completion('{"signal_evaluations": []}'))])
    asyncio.run(intent._call_openrouter(client, "openai/gpt-6-luna", prompt))
    messages = client.requests[0]["kwargs"]["json"]["messages"]
    system = messages[0]["content"]
    assert "This call judges one target criterion, including an optional bonus" in system
    assert "Do not require it to satisfy another primary or bonus criterion" in system
    assert "caller separately enforces required primary intent" in system
    data = json.loads(prompt.split("Intent signal to verify:\n", 1)[1].split(
        "\n\nThree-part verification", 1)[0])
    assert data["target_icp_signal"] == target
    assert data["buyer_original_request"] == REQUEST
    assert "another participant's attributes cannot substitute" in system


def test_positive_optional_verdict_still_cannot_replace_failed_primary():
    rows = [{"matched_icp_signal": 0, "after_decay": 0},
            {"matched_icp_signal": 1, "after_decay": 51}]
    assert not required_intent_satisfied(rows)
    rows[0]["after_decay"] = 51
    assert required_intent_satisfied(rows)
