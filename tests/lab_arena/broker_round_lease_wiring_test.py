"""Paid broker calls renew the lease frozen in their Arena round."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from lab_arena import broker, wiring
from tests.lab_arena.test_lab_arena_broker import (
    CONTEXT, FakeLedgerStore, FakeTransport, HOST_KEYS, price_table,
)


class LeaseStore(FakeLedgerStore):
    def __init__(self) -> None:
        super().__init__()
        self.renewals: list[tuple[str, int]] = []
        self.paid_at = datetime(2026, 9, 15, 8, tzinfo=timezone.utc)
        self.lease_expires_at = self.paid_at

    def _renew(self, kind: str, ttl: int) -> None:
        self.renewals.append((kind, ttl))
        # The store's reserve/settle/uncertain SQL writes now + passed TTL.
        self.lease_expires_at = self.paid_at + timedelta(seconds=ttl)

    def reserve_call(self, **kwargs):
        self._renew("reserve", kwargs["lease_ttl_seconds"])
        return super().reserve_call(**kwargs)

    def settle_call(self, **kwargs):
        self._renew("settle", kwargs["lease_ttl_seconds"])
        return super().settle_call(**kwargs)

    def mark_uncertain(self, **kwargs):
        self._renew("uncertain", kwargs["lease_ttl_seconds"])
        return super().mark_uncertain(**kwargs)


def _wired_factory(monkeypatch):
    store = LeaseStore()
    provider = FakeTransport(responses=[(200, {"data": []})])
    fake_chain = SimpleNamespace(close=lambda: None)
    fake_registry = SimpleNamespace(close=lambda: None)
    digest = "sha256:" + "a" * 64
    monkeypatch.setenv("LAB_ARENA_SERVICE_KEY", "test")

    monkeypatch.setattr(wiring, "_required", lambda name: HOST_KEYS.get(
        name.removeprefix("LAB_ARENA_").removesuffix("_API_KEY").lower(), "test"
    ))
    monkeypatch.setattr(wiring, "PostgrestTransport", lambda *a, **kw: object())
    monkeypatch.setattr(wiring, "ArenaStore", lambda _transport: store)
    monkeypatch.setattr(wiring, "S3ObjectStore", lambda *a, **kw: object())
    monkeypatch.setattr(wiring.chain_module, "ArenaChainConfig", lambda **kw: SimpleNamespace(
        network_name="test", netuid=401,
    ))
    monkeypatch.setattr(wiring.chain_module, "connect_substrate", lambda _config: object())
    monkeypatch.setattr(wiring.chain_module, "ArenaChain", lambda *_a: fake_chain)
    monkeypatch.setattr(wiring, "ChainReadsAdapter", lambda _chain: object())
    monkeypatch.setattr(wiring.atexit, "register", lambda _callback: None)
    monkeypatch.setattr(wiring, "_runner_hotkeys_from_environment", lambda: ())
    monkeypatch.setattr(wiring, "registry_client_from_environment", lambda: fake_registry)
    monkeypatch.setattr(wiring.images, "ImageRules", lambda **kw: object())
    monkeypatch.setattr(wiring.images, "parse_reference", lambda _ref: object())
    monkeypatch.setattr(wiring.images, "resolve_image", lambda *_a: SimpleNamespace(
        image_digest=digest, reference="scorer@" + digest,
    ))
    monkeypatch.setattr(wiring, "_max_image_bytes_from_environment", lambda: 1)
    monkeypatch.setattr(wiring, "_rewards_enabled_from_environment", lambda: False)
    monkeypatch.setattr(wiring, "_max_challengers_from_environment", lambda: 1)
    monkeypatch.setattr(wiring, "_stage_minutes_from_environment", lambda **kw: dict(
        wiring.DEFAULT_STAGE_MINUTES,
    ))
    monkeypatch.setattr(wiring, "_daily_cutoff_hour_from_environment", lambda: None)
    monkeypatch.setattr(wiring, "_pool_percent_from_environment", lambda: 0)
    monkeypatch.setattr(wiring, "_baseline_source_url_from_environment", lambda _mode: "https://example.com/baseline")
    monkeypatch.setattr(wiring.broker_module, "fetch_openrouter_price_table", price_table)
    monkeypatch.setattr(wiring.broker_module, "HttpxProviderTransport", lambda: provider)

    class Keys:
        def __init__(self, **_kw):
            pass

        credential_for = staticmethod(lambda _context, name: HOST_KEYS[name])
        funding_source_for = staticmethod(lambda _context: "host")
        provider_funding_source_for = staticmethod(lambda _context, _provider: "host")
        retry_miner_credential_for = staticmethod(lambda _context: False)
        mark_provider_fallback = staticmethod(lambda *_args: {"status": "none"})
        provider_restart_required_for = staticmethod(lambda *_args: False)
        code_review_key = staticmethod(lambda *_args: "test")

    monkeypatch.setattr(wiring, "SubmissionProviderKeys", Keys)
    monkeypatch.setattr(wiring, "SubmissionCodeReviewer", lambda **kw: object())
    monkeypatch.setattr(wiring, "ServiceConfig", lambda **kw: SimpleNamespace(**kw))
    monkeypatch.setattr(wiring, "ArenaService", lambda config: SimpleNamespace(
        config=config, scorer_policy={"judge_models": {}},
    ))
    monkeypatch.setattr(wiring, "create_app", lambda _service: object())
    service, _app = wiring.build_service_from_environment("shadow")
    return service, store, provider


def test_gateway_paid_calls_keep_frozen_45m_round_lease(monkeypatch):
    service, store, provider = _wired_factory(monkeypatch)
    factory = service.config.broker_factory
    current = factory(service, {"configuration_doc": {"lease_ttl_seconds": 3600}})
    legacy = factory(service, {"configuration_doc": {"lease_ttl_seconds": 1200}})
    assert isinstance(current, broker.Broker)
    assert current._lease_ttl_seconds == 3600
    assert legacy._lease_ttl_seconds == 1200

    parameters = {"url": "https://example.com/about"}
    settled = current.execute(CONTEXT, operation_id="scrapingdog.scrape",
                              parameters=parameters, action_sequence=1, timeout_ms=5000)
    assert settled.status == 200
    provider.fail = True
    uncertain = current.execute(CONTEXT, operation_id="scrapingdog.scrape",
                                parameters=parameters, action_sequence=2, timeout_ms=5000)
    assert uncertain.status == 502
    assert uncertain.call["outcome"] == "uncertain"
    assert store.renewals == [
        ("reserve", 3600), ("settle", 3600),
        ("reserve", 3600), ("uncertain", 3600),
    ]
    assert store.paid_at + timedelta(minutes=40) < store.lease_expires_at

    provider.fail = False
    provider.responses.append((200, {"data": []}))
    historical = legacy.execute(CONTEXT, operation_id="scrapingdog.scrape",
                                parameters=parameters, action_sequence=3, timeout_ms=5000)
    assert historical.status == 200
    assert store.renewals[-2:] == [("reserve", 1200), ("settle", 1200)]
    assert store.lease_expires_at == store.paid_at + timedelta(minutes=20)
