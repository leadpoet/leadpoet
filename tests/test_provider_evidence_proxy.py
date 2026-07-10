"""W3 evidence-proxy upgrade: registry, usage ledger, caller attribution, key split."""

from __future__ import annotations

import http.client
import io
import json
import urllib.request
from typing import Any

import pytest

from gateway.research_lab import provider_evidence_proxy as proxy_module
from gateway.research_lab.provider_evidence_proxy import (
    CALLER_TOKEN_HEADER,
    ProviderRegistryEntry,
    ProviderUsageLedger,
    load_provider_registry,
    provider_registry_hash,
    resolve_provider_credential,
    seed_provider_registry,
    serve_evidence_proxy,
    validate_provider_registry_entries,
)
from research_lab.eval.provider_evidence_cache import canonical_request_fingerprint


class TestRegistryValidation:
    def test_seed_registry_is_valid_and_covers_legacy_providers(self):
        entries = seed_provider_registry()
        assert validate_provider_registry_entries(entries) == []
        assert {entry.id for entry in entries} == {"exa", "sd", "or", "deepline"}
        by_id = {entry.id: entry for entry in entries}
        assert by_id["exa"].base_url == "https://api.exa.ai"
        assert by_id["sd"].auth_kind == "query"
        assert by_id["or"].auth_kind == "bearer"
        assert by_id["deepline"].base_url == "https://code.deepline.com"
        assert by_id["deepline"].auth_kind == "bearer"

    def test_duplicate_ids_rejected(self):
        entries = seed_provider_registry() + [seed_provider_registry()[0]]
        assert any("duplicate" in error for error in validate_provider_registry_entries(entries))

    def test_non_https_base_url_rejected(self):
        entry = ProviderRegistryEntry(
            id="bad", base_url="http://insecure.example", auth_kind="none"
        )
        assert any("https" in error for error in validate_provider_registry_entries([entry]))

    def test_auth_without_credential_ref_rejected(self):
        entry = ProviderRegistryEntry(
            id="x1", base_url="https://x.example", auth_kind="header", auth_name="x-api-key"
        )
        assert any("credential_ref" in error for error in validate_provider_registry_entries([entry]))

    def test_registry_hash_is_order_independent(self):
        entries = seed_provider_registry()
        assert provider_registry_hash(entries) == provider_registry_hash(list(reversed(entries)))


class TestRegistryLoading:
    def test_missing_path_falls_back_to_seed(self, monkeypatch):
        monkeypatch.delenv(proxy_module.REGISTRY_PATH_ENV, raising=False)
        entries = load_provider_registry("")
        assert {entry.id for entry in entries} == {"exa", "sd", "or", "deepline"}

    def test_file_roundtrip(self, tmp_path):
        path = tmp_path / "registry.json"
        path.write_text(
            json.dumps(
                {
                    "providers": [
                        {
                            "id": "newsapi",
                            "base_url": "https://api.news.example",
                            "auth_kind": "header",
                            "auth_name": "x-key",
                            "credential_ref": ["RESEARCH_LAB_NEWSAPI_KEY"],
                            "per_day_quota": 50,
                            "cost_model": {"currency": "usd", "est_cost_microusd_per_call": 500},
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        entries = load_provider_registry(str(path))
        assert len(entries) == 1
        assert entries[0].per_day_quota == 50
        assert entries[0].est_cost_microusd() == 500

    def test_invalid_registry_file_raises_instead_of_silently_reverting(self, tmp_path):
        path = tmp_path / "registry.json"
        path.write_text(json.dumps({"providers": [{"id": "bad", "base_url": "ftp://x"}]}), encoding="utf-8")
        with pytest.raises(ValueError):
            load_provider_registry(str(path))


class TestCredentialResolution:
    def test_key_split_on_uses_only_lab_scoped_keys(self, monkeypatch):
        monkeypatch.setenv("RESEARCH_LAB_SCRAPINGDOG_API_KEY", "lab-key")
        monkeypatch.setenv("SCRAPINGDOG_API_KEY", "shared-key")
        monkeypatch.setenv("QUALIFICATION_SCRAPINGDOG_API_KEY", "fulfillment-key")
        entry = next(e for e in seed_provider_registry() if e.id == "sd")
        value, source = resolve_provider_credential(entry, key_split=True)
        assert value == "lab-key"
        assert source == "RESEARCH_LAB_SCRAPINGDOG_API_KEY"

    def test_key_split_on_never_falls_back_to_qualification_keys(self, monkeypatch):
        monkeypatch.delenv("RESEARCH_LAB_SCRAPINGDOG_API_KEY", raising=False)
        monkeypatch.delenv("SCRAPINGDOG_API_KEY", raising=False)
        monkeypatch.setenv("QUALIFICATION_SCRAPINGDOG_API_KEY", "fulfillment-key")
        entry = next(e for e in seed_provider_registry() if e.id == "sd")
        value, source = resolve_provider_credential(entry, key_split=True)
        assert value == ""
        assert source == ""

    def test_key_split_off_preserves_legacy_fallback_chain(self, monkeypatch):
        monkeypatch.delenv("RESEARCH_LAB_SCRAPINGDOG_API_KEY", raising=False)
        monkeypatch.delenv("SCRAPINGDOG_API_KEY", raising=False)
        monkeypatch.setenv("QUALIFICATION_SCRAPINGDOG_API_KEY", "fulfillment-key")
        entry = next(e for e in seed_provider_registry() if e.id == "sd")
        value, source = resolve_provider_credential(entry, key_split=False)
        assert value == "fulfillment-key"
        assert source == "QUALIFICATION_SCRAPINGDOG_API_KEY"

    def test_key_split_default_reads_env_flag(self, monkeypatch):
        monkeypatch.setenv(proxy_module.KEY_SPLIT_ENV, "true")
        monkeypatch.delenv("RESEARCH_LAB_OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.setenv("OPENROUTER_KEY", "legacy")
        entry = next(e for e in seed_provider_registry() if e.id == "or")
        value, _ = resolve_provider_credential(entry)
        assert value == ""


class TestUsageLedger:
    def test_rows_append_as_jsonl_with_caller_context(self, tmp_path):
        path = tmp_path / "ledger.jsonl"
        ledger = ProviderUsageLedger(str(path))
        ledger.record(
            provider_id="exa",
            endpoint_class="/search",
            fingerprint="fp1",
            evidence="recorded",
            status=200,
            est_cost_microusd=5000,
            caller={"caller_kind": "benchmark", "run_id": "r1"},
        )
        ledger.record(
            provider_id="exa",
            endpoint_class="/search",
            fingerprint="fp1",
            evidence="hit",
            status=200,
            est_cost_microusd=0,
            caller={"caller_kind": "benchmark", "run_id": "r2"},
        )
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert len(rows) == 2
        assert rows[0]["evidence"] == "recorded"
        assert rows[0]["caller"]["run_id"] == "r1"
        assert rows[1]["evidence"] == "hit"
        assert rows[1]["est_cost_microusd"] == 0

    def test_live_call_counter_counts_only_recorded(self, tmp_path):
        ledger = ProviderUsageLedger(str(tmp_path / "ledger.jsonl"))
        for evidence in ("recorded", "hit", "recorded", "error"):
            ledger.record(
                provider_id="sd",
                endpoint_class="/scrape",
                fingerprint="fp",
                evidence=evidence,
                status=200,
                est_cost_microusd=0,
                caller=None,
            )
        assert ledger.live_calls_today("sd") == 2
        assert ledger.live_calls_today("exa") == 0

    def test_ledger_without_path_still_counts(self):
        ledger = ProviderUsageLedger("")
        ledger.record(
            provider_id="or",
            endpoint_class="/api/v1/chat",
            fingerprint="fp",
            evidence="recorded",
            status=200,
            est_cost_microusd=0,
            caller=None,
        )
        assert ledger.live_calls_today("or") == 1


def _request(port: int, path: str, headers: dict[str, str] | None = None) -> tuple[int, bytes, dict[str, str]]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        connection.request("GET", path, headers=headers or {})
        response = connection.getresponse()
        return response.status, response.read(), dict(response.getheaders())
    finally:
        connection.close()


@pytest.fixture()
def proxy_server(tmp_path):
    """A running proxy with a test registry, ledger file, and token map."""

    ledger_path = tmp_path / "ledger.jsonl"
    token_map_path = tmp_path / "tokens.json"
    token_map_path.write_text(
        json.dumps({"tok-123": {"caller_kind": "loop_probe", "run_id": "run-9"}}), encoding="utf-8"
    )
    registry = [
        ProviderRegistryEntry(
            id="exa",
            base_url="https://api.exa.ai",
            auth_kind="header",
            auth_name="x-api-key",
            credential_ref=("RESEARCH_LAB_EXA_API_KEY", "EXA_API_KEY"),
            per_day_quota=2,
            cost_model={"currency": "usd", "est_cost_microusd_per_call": 5000},
        ),
    ]
    server, store, thread = serve_evidence_proxy(
        host="127.0.0.1",
        port=0,
        registry=registry,
        usage_ledger_path=str(ledger_path),
        caller_context={"caller_kind": "spawn_default", "worker": "w0"},
        caller_token_map_path=str(token_map_path),
        key_split=False,
    )
    try:
        yield server, store, server.server_address[1], ledger_path
    finally:
        server.shutdown()
        server.server_close()


def _ledger_rows(path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class TestProxyEndToEnd:
    def test_unknown_provider_route_404s(self, proxy_server):
        _server, _store, port, ledger_path = proxy_server
        status, body, _ = _request(port, "/nope/anything")
        assert status == 404
        assert b"unknown provider route" in body

    def test_replay_hit_writes_attributed_ledger_row(self, proxy_server):
        _server, store, port, ledger_path = proxy_server
        fingerprint = canonical_request_fingerprint("GET", "https://api.exa.ai/search?q=x", None)
        store.record(fingerprint, 200, b'{"results":[]}')
        status, body, headers = _request(port, "/exa/search?q=x")
        assert status == 200
        assert headers.get("X-Research-Lab-Evidence") == "hit"
        rows = _ledger_rows(ledger_path)
        # store.record() above also wrote nothing (direct store call) — only
        # the HTTP request produces a row.
        assert len(rows) == 1
        assert rows[0]["evidence"] == "hit"
        assert rows[0]["endpoint_class"] == "/search"
        assert rows[0]["est_cost_microusd"] == 0
        assert rows[0]["caller"] == {"caller_kind": "spawn_default", "worker": "w0"}

    def test_worker_issued_token_overrides_spawn_context(self, proxy_server):
        _server, store, port, ledger_path = proxy_server
        fingerprint = canonical_request_fingerprint("GET", "https://api.exa.ai/search?q=t", None)
        store.record(fingerprint, 200, b"{}")
        _request(port, "/exa/search?q=t", headers={CALLER_TOKEN_HEADER: "tok-123"})
        rows = _ledger_rows(ledger_path)
        assert rows[-1]["caller"] == {"caller_kind": "loop_probe", "run_id": "run-9"}

    def test_unknown_token_never_trusts_container_identity(self, proxy_server):
        _server, store, port, ledger_path = proxy_server
        fingerprint = canonical_request_fingerprint("GET", "https://api.exa.ai/search?q=u", None)
        store.record(fingerprint, 200, b"{}")
        _request(port, "/exa/search?q=u", headers={CALLER_TOKEN_HEADER: "forged-token"})
        rows = _ledger_rows(ledger_path)
        assert rows[-1]["caller"] == {"caller_kind": "unknown_token"}

    def test_missing_credential_fails_attributed_without_upstream_call(self, proxy_server, monkeypatch):
        _server, _store, port, ledger_path = proxy_server
        monkeypatch.delenv("RESEARCH_LAB_EXA_API_KEY", raising=False)
        monkeypatch.delenv("EXA_API_KEY", raising=False)

        def _no_upstream(*args, **kwargs):  # pragma: no cover - must not be reached
            raise AssertionError("upstream must not be called without a credential")

        monkeypatch.setattr(urllib.request, "urlopen", _no_upstream)
        status, body, _ = _request(port, "/exa/search?q=miss")
        assert status == 502
        assert b"credential" in body
        assert _ledger_rows(ledger_path)[-1]["evidence"] == "credential_missing"

    def test_live_call_records_and_enforces_day_quota(self, proxy_server, monkeypatch):
        _server, _store, port, ledger_path = proxy_server
        monkeypatch.setenv("RESEARCH_LAB_EXA_API_KEY", "lab-key")
        seen_auth: list[str] = []

        class _FakeResponse:
            status = 200

            def read(self):
                # Carries an Exa cost field so the per-scope cost ledger can
                # price the call (a cost-untrackable response fail-closes the
                # scope by design — covered in test_provider_cost_tracking).
                return b'{"live":true,"costDollars":{"total":0.005}}'

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        def _fake_urlopen(request, timeout=0):
            seen_auth.append(request.headers.get("X-api-key", ""))
            return _FakeResponse()

        monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
        first = _request(port, "/exa/search?q=live1")
        second = _request(port, "/exa/search?q=live2")
        assert first[0] == 200 and second[0] == 200
        assert seen_auth == ["lab-key", "lab-key"]
        # Quota is 2/day: the third distinct live request is refused.
        third = _request(port, "/exa/search?q=live3")
        assert third[0] == 429
        rows = _ledger_rows(ledger_path)
        assert [row["evidence"] for row in rows] == ["recorded", "recorded", "quota_exhausted"]
        assert rows[0]["est_cost_microusd"] == 5000
        # Replays of already-recorded requests still succeed after quota.
        replay = _request(port, "/exa/search?q=live1")
        assert replay[0] == 200
        assert _ledger_rows(ledger_path)[-1]["evidence"] == "hit"


class TestDayCacheRecordability:
    """Nonterminal poll snapshots must never enter the day cache: a recorded
    "running" replays to every later poll of the same URL, so the poller can
    never observe the terminal state (poll-freeze)."""

    def _deepline_run_body(self, status: str) -> bytes:
        return json.dumps(
            {"schemaVersion": 1, "kind": "play_run", "run": {"id": "play/x/run/1", "status": status}}
        ).encode()

    def test_deepline_nonterminal_run_poll_not_recordable(self) -> None:
        for status in ("running", "queued", "pending", "in_progress", "started"):
            assert not proxy_module._response_is_recordable(
                "deepline",
                "https://code.deepline.com/api/v2/runs/play%2Fx%2Frun%2F1",
                200,
                self._deepline_run_body(status),
            )

    def test_deepline_terminal_run_poll_recordable(self) -> None:
        for status in ("completed", "failed", "cancelled"):
            assert proxy_module._response_is_recordable(
                "deepline",
                "https://code.deepline.com/api/v2/runs/play%2Fx%2Frun%2F1",
                200,
                self._deepline_run_body(status),
            )

    def test_deepline_non_poll_paths_recordable(self) -> None:
        assert proxy_module._response_is_recordable(
            "deepline",
            "https://code.deepline.com/api/v2/plays/run",
            200,
            self._deepline_run_body("running"),
        )

    def test_exa_agent_exemption_unchanged(self) -> None:
        running = json.dumps({"object": "agent_run", "id": "agent_run_1", "status": "running"}).encode()
        done = json.dumps({"object": "agent_run", "id": "agent_run_1", "status": "completed"}).encode()
        assert not proxy_module._response_is_recordable("exa", "https://api.exa.ai/agent/runs/1", 200, running)
        assert proxy_module._response_is_recordable("exa", "https://api.exa.ai/agent/runs/1", 200, done)

    def test_exa_no_more_credits_body_not_recordable_even_if_200(self) -> None:
        body = json.dumps(
            {
                "error": "You have exceeded your credits limit. Please top up to keep using Exa.",
                "tag": "NO_MORE_CREDITS",
            }
        ).encode()

        assert not proxy_module._response_is_recordable(
            "exa",
            "https://api.exa.ai/search",
            200,
            body,
        )

    def test_exa_successful_search_body_recordable(self) -> None:
        body = json.dumps({"results": [{"url": "https://example.com"}], "costDollars": {"total": 0.007}}).encode()

        assert proxy_module._response_is_recordable(
            "exa",
            "https://api.exa.ai/search",
            200,
            body,
        )

    def test_errors_never_recordable(self) -> None:
        assert not proxy_module._response_is_recordable("deepline", "https://code.deepline.com/api/v2/runs/x", 500, b"{}")
        assert not proxy_module._response_is_recordable("sd", "https://api.scrapingdog.com/profile", 404, b"{}")


class TestUpstreamEncodingIdentity:
    """A client's Accept-Encoding must never reach the upstream: the day cache
    stores bodies with no header metadata, so a compressed recorded body would
    replay as undecodable bytes (no Content-Encoding) to every later caller."""

    def test_upstream_request_forces_identity_encoding(self, proxy_server, monkeypatch):
        _server, _store, port, _ledger_path = proxy_server
        monkeypatch.setenv("RESEARCH_LAB_EXA_API_KEY", "lab-key")
        seen_encoding: list[str] = []

        class _FakeResponse:
            status = 200

            def read(self):
                return b'{"live":true,"costDollars":{"total":0.001}}'

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        def _fake_urlopen(request, timeout=0):
            seen_encoding.append(request.headers.get("Accept-encoding", ""))
            return _FakeResponse()

        monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
        status, _body, _headers = _request(
            port, "/exa/search?q=enc1", headers={"Accept-Encoding": "gzip, deflate, br"}
        )
        assert status == 200
        assert seen_encoding == ["identity"]
