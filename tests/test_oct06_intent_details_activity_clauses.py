"""Exercise missing-action coverage controls through the existing review gate.

The provider is controlled here. Live semantic behavior requires the saved-input
provider replay; these tests verify that its independent checks reach the gate.
"""
import asyncio
import json

import pytest

from qualification.scoring import intent_details, verification_helpers
from tests.test_arena_intent_details_grounding import inputs, _review_response


@pytest.mark.parametrize("paragraph,covered,natural", [
    (
        "Acme, a reporting software company, and Beta. This activity may "
        "support Acme's reporting software platform.",
        False, False,
    ),
    (
        "Acme and Beta announced a partnership to integrate Acme's reporting "
        "software platform, which may support customer workflows.",
        True, True,
    ),
    (
        "Acme's partnership with Beta may support customer workflows on "
        "Acme's reporting software platform.",
        True, True,
    ),
])
def test_activity_coverage_does_not_come_from_the_source(
    monkeypatch, paragraph, covered, natural,
):
    company, icp, results, fit = inputs()
    company.intent_details = paragraph
    company.intent_signals = company.intent_signals[:1]
    company.intent_signals[0].description = "Acme and Beta announced a partnership."
    icp.intent_signals = ["Announced a strategic partnership."]
    results = results[:1]
    source_quote = (
        "Acme and Beta announced a partnership to integrate Acme's reporting "
        "software platform."
    )
    evaluation = results[0]["judge_verdict"]["verification_trace"][
        "intent_verdict"
    ]["signal_evaluations"][0]
    evaluation["supporting_quotes"] = [source_quote]
    calls = []

    async def judge(prompt, **kwargs):
        document = json.loads(prompt)
        calls.append(document)
        assert " ".join(
            unit["text"] for unit in document["intent_details_units"]
        ) == paragraph
        assert source_quote in json.dumps(document["admitted_evidence"])
        assert 'Company names joined by "and"' in kwargs["system_prompt"]
        checks = {name: True for name in intent_details._CHECKS}
        checks["verified_signals_covered"] = covered
        checks["natural_paragraph"] = natural
        return json.dumps(_review_response(
            checks, [{"matched_icp_signal": 0, "covered": covered}], document,
        ))

    monkeypatch.setattr(verification_helpers, "openrouter_chat", judge)
    receipt = asyncio.run(intent_details.review_intent_details(
        company, icp, results, fit,
    ))
    assert len(calls) == 1
    assert receipt["decision"] == ("match" if covered and natural else "mismatch")
    assert receipt["checks"]["facts_supported"] is True
    assert receipt["checks"]["verified_signals_covered"] is covered
    assert receipt["checks"]["natural_paragraph"] is natural
