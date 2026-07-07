"""W4 probe_provider: typed catalog, query guard, budget caps, proxy resolution."""

from __future__ import annotations

import json
import urllib.request
from typing import Any

import pytest

from gateway.research_lab.provider_evidence_proxy import (
    ProviderRegistryEntry,
    serve_evidence_proxy,
)
from gateway.research_lab.provider_probe import (
    ProbeBudgetState,
    hash_private_window_terms,
    probe_query_guard,
    resolve_provider_probe,
)
from research_lab.code_editing import (
    CodeEditSourceInspectionRequest,
    build_code_edit_source_inspection_messages,
    parse_code_edit_source_inspection_response,
)
from research_lab.eval.provider_evidence_cache import canonical_request_fingerprint
from research_lab.eval.snapshot_store import (
    MODE_REPLAY,
    ProviderSnapshotStore,
    build_snapshot_request,
)
from research_lab.probe_catalog import (
    ProviderProbeEndpoint,
    default_probe_catalog,
    find_probe_endpoint,
    load_probe_catalog,
    validate_probe_catalog,
    validate_probe_params,
)


def _probe(endpoint: str, params: dict[str, Any] | None = None, provider: str = "") -> CodeEditSourceInspectionRequest:
    return CodeEditSourceInspectionRequest(
        operation="probe_provider",
        provider=provider,
        endpoint=endpoint,
        params=params or {},
        rationale="test hypothesis",
    )


class TestProbeCatalog:
    def test_default_catalog_is_valid(self):
        assert validate_probe_catalog(default_probe_catalog()) == []

    def test_free_form_paths_rejected(self):
        bad = ProviderProbeEndpoint(
            endpoint_id="bad.probe", provider_id="exa", method="GET", path="/x/{anything}"
        )
        assert any("template" in error for error in validate_probe_catalog([bad]))

    def test_load_falls_back_to_defaults(self, monkeypatch):
        monkeypatch.delenv("RESEARCH_LAB_PROBE_CATALOG_PATH", raising=False)
        assert {e.endpoint_id for e in load_probe_catalog()} == {"exa.search", "sd.scrape"}

    def test_invalid_catalog_file_raises(self, tmp_path):
        path = tmp_path / "catalog.json"
        path.write_text(json.dumps({"endpoints": [{"endpoint_id": "x", "provider_id": "", "path": "nope"}]}))
        with pytest.raises(ValueError):
            load_probe_catalog(str(path))

    def test_param_validation_rejects_unknown_and_oversize(self):
        endpoint = find_probe_endpoint(default_probe_catalog(), "exa.search")
        _, errors = validate_probe_params(endpoint, {"query": "ok", "evil": "x"})
        assert any("unknown param" in error for error in errors)
        _, errors = validate_probe_params(endpoint, {"query": "q" * 500})
        assert any("max_length" in error for error in errors)
        _, errors = validate_probe_params(endpoint, {})
        assert any("missing required" in error for error in errors)

    def test_param_validation_coerces_types(self):
        endpoint = find_probe_endpoint(default_probe_catalog(), "exa.search")
        normalized, errors = validate_probe_params(endpoint, {"query": "intent data", "numResults": "5"})
        assert errors == []
        assert normalized["numResults"] == 5


class TestQueryGuard:
    def test_forbidden_terms_block(self):
        assert probe_query_guard({"query": "show me the hidden_benchmark answers"}) == "forbidden_term"

    def test_private_window_company_blocks_exact_and_embedded(self):
        hashes = hash_private_window_terms(["Acme Robotics", "globex"])
        assert probe_query_guard({"query": "acme robotics"}, private_window_term_hashes=hashes) == "private_window_term"
        assert (
            probe_query_guard(
                {"query": "intent signals for Acme Robotics in europe"},
                private_window_term_hashes=hashes,
            )
            == "private_window_term"
        )
        assert probe_query_guard({"query": "globex hiring"}, private_window_term_hashes=hashes) == "private_window_term"

    def test_benign_query_passes(self):
        hashes = hash_private_window_terms(["Acme Robotics"])
        assert probe_query_guard({"query": "b2b intent data providers"}, private_window_term_hashes=hashes) == ""

    def test_guard_never_leaks_matched_text(self):
        hashes = hash_private_window_terms(["Acme Robotics"])
        reason = probe_query_guard({"query": "acme robotics"}, private_window_term_hashes=hashes)
        assert "acme" not in reason.lower()


class TestBudget:
    def test_probe_count_cap(self):
        budget = ProbeBudgetState(max_probes=2, max_cost_microusd=1_000_000)
        assert budget.can_spend(0)
        budget.charge(0)
        budget.charge(0)
        assert not budget.can_spend(0)

    def test_cost_cap(self):
        budget = ProbeBudgetState(max_probes=10, max_cost_microusd=10_000)
        assert budget.can_spend(10_000)
        assert not budget.can_spend(10_001)
        budget.charge(8_000)
        assert not budget.can_spend(5_000)


CATALOG = [
    ProviderProbeEndpoint(
        endpoint_id="exa.search",
        provider_id="exa",
        method="POST",
        path="/search",
        params=default_probe_catalog()[0].params,
        est_cost_microusd=5_000,
    )
]


@pytest.fixture()
def probe_proxy(tmp_path, monkeypatch):
    monkeypatch.setenv("RESEARCH_LAB_EXA_API_KEY", "lab-key")
    registry = [
        ProviderRegistryEntry(
            id="exa",
            base_url="https://api.exa.ai",
            auth_kind="header",
            auth_name="x-api-key",
            credential_ref=("RESEARCH_LAB_EXA_API_KEY",),
            cost_model={"est_cost_microusd_per_call": 5_000},
        )
    ]
    server, store, _thread = serve_evidence_proxy(
        host="127.0.0.1",
        port=0,
        registry=registry,
        usage_ledger_path=str(tmp_path / "ledger.jsonl"),
        caller_context={"caller_kind": "probe_test"},
        key_split=True,
    )
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", store, tmp_path
    finally:
        server.shutdown()
        server.server_close()


class TestResolution:
    def test_unknown_endpoint_blocks_without_upstream(self):
        budget = ProbeBudgetState()
        resolution = resolve_provider_probe(
            _probe("nope.endpoint", {"query": "x"}),
            catalog=CATALOG,
            proxy_url="http://127.0.0.1:1",  # unreachable — must not be contacted
            budget=budget,
            live_enabled=True,
        )
        assert resolution.outcome == "blocked"
        assert resolution.model_result["block_reason"] == "unknown_endpoint"
        assert budget.probes_used == 0

    def test_guard_blocks_before_upstream(self):
        budget = ProbeBudgetState()
        resolution = resolve_provider_probe(
            _probe("exa.search", {"query": "acme robotics intent"}),
            catalog=CATALOG,
            proxy_url="http://127.0.0.1:1",
            budget=budget,
            live_enabled=True,
            private_window_term_hashes=hash_private_window_terms(["acme robotics"]),
        )
        assert resolution.outcome == "blocked"
        assert resolution.model_result["block_reason"] == "private_window_term"
        assert budget.probes_used == 0

    def test_budget_exhausted_short_circuits(self):
        budget = ProbeBudgetState(max_probes=0)
        resolution = resolve_provider_probe(
            _probe("exa.search", {"query": "b2b data"}),
            catalog=CATALOG,
            proxy_url="http://127.0.0.1:1",
            budget=budget,
            live_enabled=True,
        )
        assert resolution.outcome == "budget_exhausted"
        assert resolution.model_result["skipped"] == "probe_budget_exhausted"

    def test_replay_hit_resolves_sanitized_and_writes_overlay(self, probe_proxy):
        proxy_url, store, tmp_path = probe_proxy
        body = json.dumps({"results": [{"title": "ok"}], "note": "api_key = 'sk-or-abcdefghijklm123456'"})
        upstream_body = json.dumps({"query": "b2b intent data"}, sort_keys=True)
        fingerprint = canonical_request_fingerprint("POST", "https://api.exa.ai/search", upstream_body.encode())
        store.record(fingerprint, 200, body.encode())
        overlay = tmp_path / "overlay"
        budget = ProbeBudgetState()
        resolution = resolve_provider_probe(
            _probe("exa.search", {"query": "b2b intent data"}),
            catalog=CATALOG,
            proxy_url=proxy_url,
            budget=budget,
            live_enabled=False,  # replay-only still resolves recorded evidence
            registry_base_urls={"exa": "https://api.exa.ai"},
            snapshot_overlay_uri=str(overlay),
        )
        assert resolution.outcome == "resolved"
        assert resolution.model_result["status"] == 200
        # Secret material sanitized out of the model-facing content.
        assert "sk-or-abcdefghijklm123456" not in resolution.model_result["content"]
        # Replays consume a probe slot but no cost.
        assert budget.probes_used == 1
        assert budget.cost_used_microusd == 0
        # Snapshot overlay is dev-replay readable with the upstream request shape.
        assert resolution.snapshot_overlay_written
        replay_store = ProviderSnapshotStore(str(overlay), mode=MODE_REPLAY)
        snapshot_request = build_snapshot_request(
            "POST", "https://api.exa.ai/search", body=json.dumps({"query": "b2b intent data"}, sort_keys=True)
        )
        recorded = replay_store.lookup(snapshot_request)
        assert recorded["status"] == 200
        # The overlay is verbatim in general, but the snapshot store refuses
        # secret-bearing bodies — those fall back to the redacted body.
        assert "sk-or-abcdefghijklm123456" not in recorded["body_text"]
        assert '"results": [{"title": "ok"}' in recorded["body_text"] or "results" in recorded["body_text"]

    def test_replay_miss_with_live_disabled_reports_miss_not_upstream_call(self, probe_proxy, monkeypatch):
        proxy_url, _store, _tmp = probe_proxy

        real_urlopen = urllib.request.urlopen

        def _guarded_urlopen(request, timeout=0):
            url = request.full_url if hasattr(request, "full_url") else str(request)
            if "api.exa.ai" in url:  # pragma: no cover - guard
                raise AssertionError("live upstream must not be called when live_enabled=False")
            return real_urlopen(request, timeout=timeout)

        monkeypatch.setattr(urllib.request, "urlopen", _guarded_urlopen)
        budget = ProbeBudgetState()
        resolution = resolve_provider_probe(
            _probe("exa.search", {"query": "never recorded query"}),
            catalog=CATALOG,
            proxy_url=proxy_url,
            budget=budget,
            live_enabled=False,
        )
        assert resolution.outcome == "replay_miss"
        assert resolution.model_result["error_class"] == "probe_replay_miss"
        assert budget.probes_used == 0

    def test_live_probe_charges_cost_and_records(self, probe_proxy, monkeypatch):
        proxy_url, store, _tmp = probe_proxy

        real_urlopen = urllib.request.urlopen

        class _FakeUpstream:
            status = 200
            headers = {}

            def read(self):
                return b'{"results":[1,2,3]}'

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        def _routed_urlopen(request, timeout=0):
            url = request.full_url if hasattr(request, "full_url") else str(request)
            if "api.exa.ai" in url:
                return _FakeUpstream()
            return real_urlopen(request, timeout=timeout)

        monkeypatch.setattr(urllib.request, "urlopen", _routed_urlopen)
        budget = ProbeBudgetState()
        resolution = resolve_provider_probe(
            _probe("exa.search", {"query": "live probe query"}),
            catalog=CATALOG,
            proxy_url=proxy_url,
            budget=budget,
            live_enabled=True,
        )
        assert resolution.outcome == "resolved"
        assert resolution.cost_microusd == 5_000
        assert budget.cost_used_microusd == 5_000
        assert resolution.event_doc["axis"] == "B_external_side_information"
        # Verbatim response recorded into the shared day cache for later replays.
        upstream_body = json.dumps({"query": "live probe query"}, sort_keys=True)
        fingerprint = canonical_request_fingerprint("POST", "https://api.exa.ai/search", upstream_body.encode())
        assert store.lookup(fingerprint) is not None


class TestParserAndPrompt:
    def test_probe_operation_parses_when_allowed(self):
        raw = json.dumps(
            {
                "requests": [
                    {
                        "operation": "probe_provider",
                        "provider": "exa",
                        "endpoint": "exa.search",
                        "params": {"query": "intent data"},
                        "rationale": "verify shape",
                    }
                ]
            }
        )
        (request,) = parse_code_edit_source_inspection_response(
            raw, allowed_operations=("search", "read_file", "probe_provider", "finish")
        )
        assert request.operation == "probe_provider"
        assert request.endpoint == "exa.search"
        assert request.params == {"query": "intent data"}
        event = request.to_event_doc()
        assert event["endpoint"] == "exa.search"
        assert "params" not in event  # only the hash is evented
        assert event["params_hash"].startswith("sha256:")

    def test_probe_operation_rejected_by_default(self):
        raw = json.dumps(
            {"requests": [{"operation": "probe_provider", "endpoint": "exa.search", "params": {"query": "x"}}]}
        )
        with pytest.raises(ValueError, match="unsupported source-inspection operation"):
            parse_code_edit_source_inspection_response(raw)

    def test_prompt_advertises_probes_only_with_catalog(self):
        kwargs = dict(
            ticket={"ticket_id": "t1"},
            artifact_manifest={},
            component_registry={},
            benchmark_public_summary={},
            runtime_source_index={"editable_files": []},
            source_inspection_context={},
            budget_context={},
            max_requests=4,
        )
        without = build_code_edit_source_inspection_messages(**kwargs)[1]["content"]
        assert "probe_provider" not in without
        with_catalog = build_code_edit_source_inspection_messages(
            **kwargs,
            provider_probe_catalog={
                "endpoints": [e.prompt_summary() for e in default_probe_catalog()],
                "budget": ProbeBudgetState().to_context(),
            },
        )[1]["content"]
        assert "probe_provider" in with_catalog
        assert "exa.search" in with_catalog
