"""Runtime-key ownership checks, separate from live provider validation."""

from dataclasses import replace

import pytest

from lab_arena.broker import BrokerError
from lab_arena.submission_runtime import SubmissionProviderKeys
from test_lab_arena_broker import CONTEXT


class Store:
    def __init__(self, row):
        self.row = row

    def get_submission(self, submission_id):
        return self.row if submission_id == self.row["submission_id"] else None

    def get_round(self, round_id):
        return {"configuration_doc": {"baseline_hotkey": "organizer"}}

    def get_submission_credential(self, submission_id, miner_hotkey, provider):
        return {"submission_id": submission_id, "miner_hotkey": miner_hotkey, "provider": provider, "ciphertext_b64": "encrypted"}


class Credentials:
    def __init__(self):
        self.calls = []

    def runtime_key(self, row, provider):
        self.calls.append((row["submission_id"], provider))
        return "miner-runtime-key"


def resolver(*, baseline=False, is_king=False, credentials=True):
    row = {
        "submission_id": "baseline-2026-09-04" if baseline else CONTEXT.submission_id,
        "round_id": "arena-2026-09-04",
        "miner_hotkey": "organizer" if baseline else CONTEXT.miner_hotkey,
        "status": "frozen", "is_king": baseline or is_king,
    }
    context = replace(CONTEXT, submission_id=row["submission_id"], miner_hotkey=row["miner_hotkey"])
    vault = Credentials() if credentials else None
    keys = SubmissionProviderKeys(store=Store(row), credentials=vault, organizer_keys={"openrouter": "host-runtime-key"})
    return keys, context, vault


@pytest.mark.parametrize("kind", ["execute", "score"])
@pytest.mark.parametrize("is_king", [False, True])
def test_miner_uses_miner_keys_even_if_it_becomes_king(kind, is_king):
    keys, context, vault = resolver(is_king=is_king)
    context = replace(context, kind=kind)
    assert keys.funding_source_for(context) == "miner_key"
    assert keys.credential_for(context, "openrouter") == "miner-runtime-key"
    assert vault.calls == [(context.submission_id, "openrouter")]


def test_only_the_organizer_baseline_uses_host_keys():
    keys, context, vault = resolver(baseline=True)
    assert keys.funding_source_for(context) == "host"
    assert keys.credential_for(context, "openrouter") == "host-runtime-key"
    assert not vault.calls


@pytest.mark.parametrize("provider", ["scrapingdog", "openrouter_management_key", "unknown"])
def test_no_unsubmitted_provider_or_management_key_route(provider):
    keys, context, vault = resolver()
    with pytest.raises(BrokerError, match="miner_provider_not_configured"):
        keys.credential_for(context, provider)
    assert not vault.calls


def test_missing_miner_credentials_cannot_fall_back_to_host():
    keys, context, _ = resolver(credentials=False)
    with pytest.raises(BrokerError, match="miner_credentials_unavailable"):
        keys.credential_for(context, "openrouter")


def test_cross_miner_identity_is_rejected():
    keys, context, vault = resolver()
    with pytest.raises(BrokerError, match="broker_unavailable"):
        keys.credential_for(replace(context, miner_hotkey="different-miner"), "openrouter")
    assert not vault.calls
