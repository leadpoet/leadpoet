"""A miner signs the exact terms for the selected task and payment mode."""
from types import SimpleNamespace

import pytest

from lab_arena import collection_intake as intake, contracts
from tests.lab_arena.collection_intake_test import manager, registration


def test_both_scopes_publish_the_same_pilot_terms(monkeypatch):
    monkeypatch.delenv("LAB_ARENA_CORRECTION_TERMS_SHA256", raising=False)
    assert intake.terms("pilot")["terms_sha256"] == intake.TERMS_HASH
    published = intake.terms("pilot", "synthetic_corrections")
    assert published["terms_sha256"] == contracts.hash_bytes(intake.CORRECTION_TERMS.encode())
    assert published["terms_sha256"] == intake.TERMS_HASH


def test_correction_registration_requires_exact_published_pilot_text(monkeypatch):
    from lab_arena.collection_intake import CORRECTION_TERMS, CORRECTION_SCOPE

    digest = contracts.hash_bytes(CORRECTION_TERMS.encode())
    monkeypatch.setenv("LAB_ARENA_CORRECTION_TERMS_SHA256", digest)
    assert intake.terms("pilot", CORRECTION_SCOPE)["terms_sha256"] == digest
    assert intake.terms("live", CORRECTION_SCOPE)["terms_sha256"] == digest
    signed, creds = registration("A", "pilot")
    signed["body"]["task_scope"] = CORRECTION_SCOPE
    signed["body"]["terms_sha256"] = intake.LEGACY_COMPARISON_TERMS_HASH
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
    monkeypatch.setenv("LAB_ARENA_CORRECTION_TERMS_SHA256", intake.LEGACY_COMPARISON_TERMS_HASH)
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
    assert config.main(["--check", "--allowed-account", "493765492819", "--correction-terms", intake.LEGACY_COMPARISON_TERMS_HASH]) == 2


@pytest.mark.parametrize('staged', [True, False])
def test_live_comparison_terms_publish_only_approved_synthetic_quotes(staged):
    from lab_arena.service import ArenaService

    fees = {model: i + 5 for i, model in enumerate(intake.MODELS)}
    caps = {model: 75_000 for model in intake.MODELS}
    schedule = {
        'first_epoch': 12, 'pending': {'miner_hotkey': 'private-funder'},
        'pilot_evidence': {'control_ids': ['private-control'], 'review_sha256': 'private-review',
                           'prices': {'P': fees, 'research': fees}, 'budgets': {'P': caps, 'research': caps}},
    }
    if staged:
        schedule['staged'] = True

    def rpc(name, args):
        if name == 'collection_pool_active_v1':
            return False
        assert (name, args) == ('collection_reward_schedule_v1', {})
        return schedule

    service = SimpleNamespace(collection_mode=lambda: 'live', _store=SimpleNamespace(_transport=SimpleNamespace(rpc=rpc)))
    terms = ArenaService.collection_terms(service)
    assert terms['fixed_quotes'] == [
        {'model': model, 'task_class': 'P', 'quote_microusd': fees[model], 'cap_microusd': caps[model]}
        for model in intake.MODELS
    ]
    assert {k: v for k, v in terms.items() if k != 'fixed_quotes'} == intake.terms('live')
    schedule.clear()
    assert ArenaService.collection_terms(service)['fixed_quotes'] == []
    schedule['pilot_evidence'] = {'prices': {'P': fees}, 'budgets': {'P': {}}}
    from lab_arena.service import ServiceError
    with pytest.raises(ServiceError) as invalid:
        ArenaService.collection_terms(service)
    assert invalid.value.status == 503


def test_unpaid_comparison_terms_do_not_read_financial_state():
    from lab_arena.service import ArenaService

    assert ArenaService.collection_terms(SimpleNamespace(collection_mode=lambda: 'pilot')) == intake.terms('pilot')


def test_empty_paid_correction_offer_is_readable_and_cannot_fund_a_job(monkeypatch):
    from fastapi.testclient import TestClient
    from lab_arena.api import create_app
    from lab_arena.service import ArenaService

    monkeypatch.delenv("LAB_ARENA_LIVE_CORRECTION_TERMS_SHA256", raising=False)
    reads = []
    def rpc(name, args):
        assert (name, args) == ("collection_open_correction_quotes_v1", {})
        reads.append(name)
        return []

    transport = SimpleNamespace(rpc=rpc)
    service = object.__new__(ArenaService)
    service.collection_mode = lambda: "live"
    service._store = SimpleNamespace(_transport=transport)
    reply = TestClient(create_app(service)).get("/arena/v1/collection/terms", params={"task_scope": intake.CORRECTION_SCOPE})
    assert reply.status_code == 200
    published = reply.json()
    assert published == {**intake.terms("live", intake.CORRECTION_SCOPE), "fixed_quotes": []}

    signed, keys = registration("R", "live")
    signed["body"]["task_scope"] = intake.CORRECTION_SCOPE
    class UnusedCredentials:
        def validate_openrouter_funding(self, *args, **kwargs):
            pytest.fail("an empty offer reached provider key validation")
    client = intake.CollectionIntake(transport, UnusedCredentials(), "live")
    for offer in (published, None):
        with pytest.raises(intake.IntakeError, match="collection_compensated_corrections_unavailable") as refusal:
            client.register(signed, "5" + "Y" * 47, keys, published=offer)
        assert refusal.value.status == 503
    assert len(reads) == 2


@pytest.mark.parametrize("response", [None, {}, intake.store.ArenaStoreError("delivery failed")])
def test_failed_correction_price_read_never_becomes_an_empty_offer(response):
    from lab_arena.service import ArenaService, ServiceError

    def rpc(name, args):
        if isinstance(response, Exception):
            raise response
        return response
    service = SimpleNamespace(collection_mode=lambda: "live", _store=SimpleNamespace(_transport=SimpleNamespace(rpc=rpc)))
    with pytest.raises(ServiceError, match="collection_compensated_corrections_unavailable") as refusal:
        ArenaService.collection_terms(service, intake.CORRECTION_SCOPE)
    assert refusal.value.status == 503


def test_disabled_paid_correction_terms_cannot_publish_even_with_open_tariff(monkeypatch):
    from lab_arena.service import ArenaService, ServiceError

    transport = SimpleNamespace(rpc=lambda *args: pytest.fail("read prices with disabled consent"))
    service = SimpleNamespace(collection_mode=lambda: "live", _store=SimpleNamespace(_transport=transport))
    monkeypatch.setenv("LAB_ARENA_LIVE_CORRECTION_TERMS_SHA256", "off")
    with pytest.raises(ServiceError, match="collection_correction_terms_unavailable") as refusal:
        ArenaService.collection_terms(service, intake.CORRECTION_SCOPE)
    assert refusal.value.status == 503
