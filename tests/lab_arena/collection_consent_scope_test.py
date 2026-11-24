"""A miner signs the exact terms for the selected task and payment mode."""
from types import SimpleNamespace

import pytest

from lab_arena import collection_intake as intake, contracts
from tests.lab_arena.collection_intake_test import manager, registration


def test_published_pilot_terms_require_separate_consent(monkeypatch):
    monkeypatch.delenv("LAB_ARENA_CORRECTION_TERMS_SHA256", raising=False)
    assert intake.terms("pilot")["terms_sha256"] == intake.TERMS_HASH
    published = intake.terms("pilot", "synthetic_corrections")
    assert published["terms_sha256"] == contracts.hash_bytes(intake.CORRECTION_TERMS.encode())
    assert published["terms_sha256"] != intake.TERMS_HASH


def test_correction_registration_requires_exact_published_pilot_text(monkeypatch):
    from lab_arena.collection_intake import CORRECTION_TERMS, CORRECTION_SCOPE

    digest = contracts.hash_bytes(CORRECTION_TERMS.encode())
    monkeypatch.setenv("LAB_ARENA_CORRECTION_TERMS_SHA256", digest)
    assert intake.terms("pilot", CORRECTION_SCOPE)["terms_sha256"] == digest
    assert intake.terms("live", CORRECTION_SCOPE)["terms_sha256"] != digest
    signed, creds = registration("A", "pilot")
    signed["body"]["task_scope"] = CORRECTION_SCOPE
    calls = []
    transport = SimpleNamespace(rpc=lambda n, p: calls.append((n, p)) or {"status": "enrolled"})
    client = intake.CollectionIntake(transport, manager(), "pilot")
    with pytest.raises(intake.IntakeError, match="consent"):
        client.register(signed, "5" + "Y" * 47, creds)
    assert not calls
    signed["body"]["terms_sha256"] = digest
    client.register(signed, "5" + "Y" * 47, creds)
    assert len(calls) == 1
    assert calls[0][1]["p_consent"]["body"]["terms_sha256"] == digest
    monkeypatch.setenv("LAB_ARENA_CORRECTION_TERMS_SHA256", intake.TERMS_HASH)
    with pytest.raises(intake.IntakeError, match="terms_unavailable"):
        client.register(signed, "5" + "Y" * 47, creds)


def test_gateway_only_publishes_the_digest_to_which_the_miner_consents(monkeypatch):
    from lab_arena.service import ArenaService, ServiceError

    service = SimpleNamespace(collection_mode=lambda: "pilot")
    monkeypatch.setenv("LAB_ARENA_CORRECTION_TERMS_SHA256", "off")
    with pytest.raises(ServiceError) as refused:
        ArenaService.collection_terms(service, intake.CORRECTION_SCOPE)
    assert refused.value.status == 503
    approved = contracts.hash_bytes(intake.CORRECTION_TERMS.encode())
    monkeypatch.setenv("LAB_ARENA_CORRECTION_TERMS_SHA256", approved)
    published = ArenaService.collection_terms(service, intake.CORRECTION_SCOPE)
    assert published["terms_sha256"] == contracts.hash_bytes(published["terms"].encode())


def test_cli_sends_no_key_to_a_gateway_that_has_different_experiment_terms(monkeypatch):
    from lab_arena import collection_cli

    args = SimpleNamespace(
        command="collection-register", task_scope=intake.CORRECTION_SCOPE,
        mode="pilot", api_base_url="https://example.org", daily_cap_usd="0.1",
        accept_terms=contracts.hash_bytes(intake.CORRECTION_TERMS.encode()),
    )
    def public_terms(url, **kw):
        assert url.endswith("/collection/terms?task_scope=synthetic_corrections")
        return SimpleNamespace(status_code=200, json=lambda: intake.terms("pilot"))
    session = SimpleNamespace(get=public_terms, post=lambda *a, **kw: pytest.fail("keys were sent"))
    assert collection_cli.run(args, lambda args: pytest.fail("wallet was opened"), session=session) == 2


def test_correction_terms_configuration_cannot_activate_a_different_text():
    from scripts import configure_lab_arena_production as config
    assert config.main(["--check", "--allowed-account", "493765492819", "--correction-terms", intake.TERMS_HASH]) == 2
