"""Credential-isolation regression checks; these are not live-run evidence."""

from dataclasses import replace

import pytest

from lab_arena import broker as br
from test_lab_arena_broker import CHAT, CONTEXT, FakeLedgerStore, FakeTransport, make_broker


@pytest.mark.parametrize("kind", ["execute", "score"])
def test_each_submission_pays_with_its_own_key(kind):
    keys = {"s1": "miner-one-runtime-key", "s2": "miner-two-runtime-key"}
    broker, ledger, transport = make_broker(
        credential_for=lambda context, provider: keys[context.submission_id],
        funding_source_for=lambda context: "miner_key",
        judge_models=[CHAT["model"]],
    )
    for index, submission_id in enumerate(keys):
        context = replace(CONTEXT, submission_id=submission_id, assignment_id=f"assignment-{index}", kind=kind)
        result = broker.execute(context, operation_id="openrouter.chat", parameters=CHAT, action_sequence=index, timeout_ms=5000)
        assert result.status == 200
        assert result.call["funding_source"] == "miner_key"
        assert transport.sent[-1]["headers"]["authorization"] == "Bearer " + keys[submission_id]
        assert not any(secret in repr(result.to_document()) for secret in keys.values())
    assert all(call["funding_source"] == "miner_key" for call in ledger.calls.values())


def test_missing_key_never_dispatches_or_uses_host_key():
    def unavailable(context, provider):
        raise br.BrokerError("miner_credentials_unavailable")

    broker, ledger, transport = make_broker(
        credential_for=unavailable,
        funding_source_for=lambda context: "miner_key",
    )
    result = broker.execute(CONTEXT, operation_id="openrouter.chat", parameters=CHAT, action_sequence=0, timeout_ms=5000)
    assert result.status == 402
    assert result.call["error_code"] == "miner_credentials_unavailable"
    assert result.call["funding_source"] == "miner_key"
    assert not ledger.calls and not transport.sent


@pytest.mark.parametrize("status", [401, 402, 403])
def test_miner_key_refusal_is_not_an_organizer_outage(status):
    broker, ledger, transport = make_broker(
        transport=FakeTransport([(status, {"error": "refused"})]),
        credential_for=lambda context, provider: "miner-runtime-key",
        funding_source_for=lambda context: "miner_key",
    )
    result = broker.execute(CONTEXT, operation_id="openrouter.chat", parameters=CHAT, action_sequence=0, timeout_ms=5000)
    assert result.call["error_code"] == "miner_credentials_unavailable"
    assert result.call["provider_status"] == status
    assert ledger.log == ["reserve", "dispatch", "settle"]
    replay = broker.execute(CONTEXT, operation_id="openrouter.chat", parameters=CHAT, action_sequence=0, timeout_ms=5000)
    assert replay.call["error_code"] == "miner_credentials_unavailable"
    assert replay.body == result.body and replay.status == result.status
    assert len(transport.sent) == 1


def test_provider_cannot_echo_runtime_key_into_output_or_storage():
    secret = "miner-runtime-key-never-publish"
    broker, ledger, transport = make_broker(
        transport=FakeTransport([(200, {"content": secret})]),
        credential_for=lambda context, provider: secret,
        funding_source_for=lambda context: "miner_key",
    )
    result = broker.execute(CONTEXT, operation_id="openrouter.chat", parameters=CHAT, action_sequence=0, timeout_ms=5000)
    assert result.status == 502
    assert secret not in repr(result.to_document())
    assert secret not in repr(ledger.calls)
