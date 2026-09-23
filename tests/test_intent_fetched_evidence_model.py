"""Evidence-only reviewers keep search-dependent roles and broker rules intact."""

import json
from unittest.mock import AsyncMock

import pytest

from lab_arena import broker, operations, scoring
from qualification.scoring import intent_verification_three_stage as intent
from tests.test_intent_three_stage_json_retry import _Client, _Response, _completion


@pytest.mark.asyncio
@pytest.mark.parametrize(("integrity", "evidence_type", "override", "expected"), [
    (True, "FUNDING", None, intent.ARENA_EVIDENCE_MODEL),
    (True, "HIRING", None, intent.ARENA_EVIDENCE_MODEL),
    (True, None, None, intent.ARENA_EVIDENCE_MODEL),
    (True, "SOCIAL_POSTING", None, intent.STAGE3_MODEL),
    (True, "PODCAST_APPEARANCE", None, intent.STAGE3_MODEL),
    (False, "FUNDING", None, intent.STAGE3_MODEL),
    (True, "FUNDING", intent.STAGE3_MODEL, intent.STAGE3_MODEL),
])
async def test_evidence_model_selection_preserves_search_roles_and_overrides(
    monkeypatch, integrity, evidence_type, override, expected,
):
    monkeypatch.delenv("ARENA_INTENT_EVIDENCE_MODEL", raising=False)
    url = "https://acme.example/news"
    monkeypatch.setattr(intent, "_fetch_sd_then_exa", AsyncMock(return_value={
        "results": [{"url": url, "text": "Acme announced its funding."}],
        "statuses": [{"url": url, "stage": "ok"}],
    }))
    review = AsyncMock(return_value={"_error": "provider_unavailable"})
    monkeypatch.setattr(intent, "_call_openrouter", review)
    result = await intent.verify_three_stage(
        None, company_name="Acme", company_website="https://acme.example",
        company_linkedin="https://www.linkedin.com/company/acme",
        source_url=url, miner_claim="Acme announced funding.",
        target_signal_text="Recent funding", evidence_type=evidence_type,
        integrity_policy=integrity, stage1_soft_reject=True,
        stage3_model=override,
    )
    review.assert_awaited_once()
    assert review.await_args.args[1] == expected
    assert result["decision"] == "unavailable"
    assert result["stage3"]["model"] == expected


@pytest.mark.parametrize("new_role", [True, False])
def test_frozen_policy_selects_the_reviewer_before_import(new_role):
    models = dict(scoring.DEFAULT_JUDGE_MODELS)
    if not new_role:
        del models["intent_fetched_evidence"]
    policy = scoring.build_scorer_policy(judge_models=models)
    env = {}
    credentials = {key: "fixture-key" for key in scoring.CREDENTIAL_ENV_NAMES}
    scoring.apply_policy_to_environment(policy, environ=env, credentials=credentials)
    expected = intent.ARENA_EVIDENCE_MODEL if new_role else intent.STAGE3_MODEL
    assert env["ARENA_INTENT_EVIDENCE_MODEL"] == expected
    with pytest.raises(scoring.ScorerPolicyConflict):
        scoring.apply_policy_to_environment(
            policy, environ={"ARENA_INTENT_EVIDENCE_MODEL": "unapproved/model"},
            credentials=credentials,
        )
    with pytest.raises(scoring.ScorerPolicyConflict):
        scoring.apply_policy_to_environment(
            dict(policy, env_bindings={"ARENA_INTENT_EVIDENCE_MODEL": "unapproved/model"}),
            environ={}, credentials=credentials,
        )


@pytest.mark.asyncio
async def test_luna_request_passes_real_operation_and_signed_judge_admission(monkeypatch):
    monkeypatch.setattr(intent, "_get_openrouter_key", lambda: "fixture-key")
    monkeypatch.setattr(
        "qualification.scoring.openrouter_options.include_reasoning_default",
        lambda: False,
    )
    client = _Client([_Response(_completion('{"signal_evaluations":[]}'))])
    await intent._call_openrouter(client, intent.ARENA_EVIDENCE_MODEL, "fetched evidence")
    body = client.requests[0]["kwargs"]["json"]
    assert "temperature" not in body
    assert body["reasoning"] == {"effort": "low"}
    assert body["max_tokens"] == 4096
    operation_id, parameters = operations.match_request(
        "POST", intent.OPENROUTER_BASE_URL + "/chat/completions",
        json.dumps(body).encode(), {"Content-Type": "application/json"},
    )
    assert operation_id == "openrouter.chat"
    models = scoring.build_scorer_policy()["judge_models"]
    assert models["intent_fetched_evidence"] == intent.ARENA_EVIDENCE_MODEL
    table = broker.validate_price_table({
        "schema_version": broker.PRICE_TABLE_SCHEMA_VERSION,
        "fetched_at": "2026-09-23T00:00:00Z",
        "source": broker.OPENROUTER_MODELS_URL,
        "models": {model: {
            "prompt": "0.0000001", "completion": "0.0000005",
            "request": "0", "image": "0", "web_search": "0",
            "internal_reasoning": "0",
        } for model in models.values()},
    })
    admitted = broker.Broker(
        store=None, key_for=lambda _provider: "fixture-key", price_table=table,
        judge_models=list(models.values()), transport=None,
    )._openrouter_parameters(parameters, kind="score")[0]
    assert admitted["model"] == intent.ARENA_EVIDENCE_MODEL
    assert admitted["reasoning"] == {"effort": "low"}
    # An old frozen scorer policy cannot silently use the new judge.
    old_models = [model for model in models.values() if model != intent.ARENA_EVIDENCE_MODEL]
    old = broker.Broker(store=None, key_for=lambda _: "fixture-key", price_table=table,
                        judge_models=old_models, transport=None)
    with pytest.raises(broker.BrokerError):
        old._openrouter_parameters(parameters, kind="score")


@pytest.mark.asyncio
async def test_upstream_rate_limit_in_http_200_uses_existing_bounded_retry(monkeypatch):
    monkeypatch.setattr(intent, "_get_openrouter_key", lambda: "fixture-key")
    monkeypatch.setattr(intent.asyncio, "sleep", AsyncMock())
    client = _Client([
        _Response({"error": {"code": 429, "message": "temporarily rate-limited upstream"}}),
        _Response(_completion('{"signal_evaluations":[]}')),
    ])
    result = await intent._call_openrouter(client, intent.ARENA_EVIDENCE_MODEL, "evidence")
    assert client.calls == 2
    assert result["answer"] == {"signal_evaluations": []}
    assert client.requests[0]["kwargs"]["json"] == client.requests[1]["kwargs"]["json"]


@pytest.mark.parametrize(("quote", "expected"), [
    ('"Acme and BSF (Bank) announced a partnership on March 3, 2026."', True),
    ("[Acme](https://acme.example/not-partner) and [BSF (Bank)](https://bank.example) announced a partnership on March 3, 2026.", True),
    ("Acme and BSF (Bank) announced a partnership on March 3, 2025.", False),
    ("Acme and BSF (Bank) did not announce a partnership on March 3, 2026.", False),
    ("Acme announced a partnership on March 3, 2026.", False),
    ("https://acme.example/not-partner", False),
])
def test_markdown_link_display_text_is_evidence_but_href_and_changed_words_are_not(quote, expected):
    source = (
        "[Acme](https://acme.example/not-partner) and "
        "[BSF (Bank)](https://bank.example) announced a partnership on March 3, 2026."
    )
    assert intent._grounded_exact_text(source, quote) is expected
