"""W2 provider-outcome digest: aggregation, sanitization, prompt injection."""

from __future__ import annotations

import json
import os
import threading
import time

from gateway.research_lab import provider_outcome_digest as digest_module
from gateway.research_lab.provider_outcome_digest import (
    MAX_PROVIDER_OUTCOME_ENDPOINTS,
    MAX_PROVIDER_OUTCOME_SIDECAR_BYTES,
    PROVIDER_OUTCOME_SIDECAR_ENV,
    ProviderOutcomeSidecarAccumulator,
    build_provider_outcome_digest,
    build_run_provider_outcome_digest,
    load_provider_outcome_sidecar,
    parse_provider_error_marker_line,
)
from research_lab.code_editing import (
    build_code_edit_auto_research_messages,
    build_code_edit_source_inspection_messages,
    build_loop_direction_planner_messages,
)


class TestMarkerParsing:
    def test_marker_line_parses_class_status_provider(self):
        line = (
            "research_lab_private_runtime_provider_error HTTPError: too many requests; "
            "status=429; reason=Too Many Requests; body={\"error\":\"rate\"}; url=https://api.exa.ai/search?q=x"
        )
        parsed = parse_provider_error_marker_line(line)
        assert parsed == {"provider": "exa", "error_class": "HTTPError", "http_status": 429}

    def test_non_marker_line_returns_none(self):
        assert parse_provider_error_marker_line("plain stderr noise") is None

    def test_parsed_marker_never_carries_body_or_query(self):
        line = (
            "research_lab_private_runtime_provider_error TimeoutError: read timed out; "
            "url=https://api.scrapingdog.com/scrape?url=https://secret-company.example/careers"
        )
        parsed = parse_provider_error_marker_line(line)
        assert parsed["provider"] == "scrapingdog"
        serialized = json.dumps(parsed)
        assert "secret-company" not in serialized
        assert "careers" not in serialized


class TestDigestBuild:
    def test_aggregates_all_sources(self):
        digest = build_provider_outcome_digest(
            provider_usage_rows=[
                {"provider": "openrouter", "call_stage": "code_edit_draft"},
                {
                    "provider": "openrouter",
                    "call_stage": "code_edit_draft",
                    "failed_request": {"error_class": "rate_limited", "http_status": 429},
                },
            ],
            usage_ledger_rows=[
                {"provider_id": "exa", "endpoint_class": "/search", "status": 200, "evidence": "recorded"},
                {"provider_id": "exa", "endpoint_class": "/search", "status": 500, "evidence": "error"},
            ],
            provider_error_marker_lines=[
                "research_lab_private_runtime_provider_error HTTPError: x; status=429; url=https://api.exa.ai/search"
            ],
            day_cache_entries={"fp1": {"status": 200, "outcome": "success"}, "fp2": {"status": 429, "outcome": "error"}},
            utc_day="2026-07-06",
        )
        assert digest["utc_day"] == "2026-07-06"
        exa = digest["providers"]["exa"]
        assert exa["call_count"] == 3
        assert exa["status_histogram"]["429"] == 1
        assert exa["status_histogram"]["500"] == 1
        assert exa["endpoint_classes"]["/search"] == 2
        openrouter = digest["providers"]["openrouter"]
        assert openrouter["error_classes"]["rate_limited"] == 1
        assert openrouter["error_rate"] == 0.5
        assert digest["day_cache_outcomes"] == {"success": 1, "error": 1}
        assert any(item["key"] == "exa:HTTPError" for item in digest["top_error_classes"])
        assert digest["digest_hash"].startswith("sha256:")

    def test_snapshot_miss_counts_carry_interpretation_note(self):
        digest = build_provider_outcome_digest(
            candidate_snapshot_miss_counts={"node-1": 7, "node-2": 0},
        )
        assert digest["candidate_snapshot_miss_counts"] == {"node-1": 7, "node-2": 0}
        assert "not-comparable" in digest["snapshot_miss_note"]

    def test_digest_never_contains_bodies(self):
        digest = build_provider_outcome_digest(
            provider_error_marker_lines=[
                "research_lab_private_runtime_provider_error HTTPError: boom; status=500; "
                'body={"secret_payload":"sk-or-abcdef1234567890"}; url=https://api.exa.ai/x'
            ],
        )
        assert "sk-or-" not in json.dumps(digest)


class TestWorkerGate:
    def test_flag_off_returns_none(self, monkeypatch):
        monkeypatch.setenv("RESEARCH_LAB_PROVIDER_OUTCOME_DIGEST", "false")
        assert build_run_provider_outcome_digest() is None

    def test_flag_on_reads_only_compact_sidecar(self, monkeypatch, tmp_path):
        sidecar_path = tmp_path / "provider_outcomes.json"
        accumulator = ProviderOutcomeSidecarAccumulator(str(sidecar_path))
        accumulator.record(
            provider_id="sd",
            endpoint_class="/scrape",
            evidence="recorded",
            status=200,
            live_call=True,
            spend_microusd=5000,
            spend_kind="estimated",
        )
        accumulator.record(
            provider_id="sd",
            endpoint_class="/scrape",
            evidence="hit",
            status=200,
            live_call=False,
            spend_microusd=0,
            spend_kind="estimated",
        )
        accumulator.close()

        large_ledger = tmp_path / "ledger.jsonl"
        large_cache = tmp_path / "day_cache.json"
        large_ledger.write_text("must-not-open", encoding="utf-8")
        large_cache.write_text("must-not-open", encoding="utf-8")
        opened_paths = []
        real_open = open

        def tracked_open(path, *args, **kwargs):
            opened_paths.append(os.fspath(path))
            return real_open(path, *args, **kwargs)

        monkeypatch.setenv("RESEARCH_LAB_PROVIDER_OUTCOME_DIGEST", "true")
        monkeypatch.setenv(PROVIDER_OUTCOME_SIDECAR_ENV, str(sidecar_path))
        monkeypatch.setenv("RESEARCH_LAB_PROVIDER_USAGE_LEDGER_PATH", str(large_ledger))
        monkeypatch.setenv("RESEARCH_LAB_PROVIDER_EVIDENCE_DAY_CACHE", str(large_cache))
        monkeypatch.setattr("builtins.open", tracked_open)
        digest = build_run_provider_outcome_digest()
        assert digest is not None
        assert digest["providers"]["sd"]["call_count"] == 2
        assert digest["providers"]["sd"]["live_call_count"] == 1
        assert digest["providers"]["sd"]["cache_hit_count"] == 1
        assert digest["aggregate_spend"] == {
            "measured_microusd": 0,
            "estimated_microusd": 5000,
        }
        assert str(sidecar_path) in opened_paths
        assert str(large_ledger) not in opened_paths
        assert str(large_cache) not in opened_paths

    def test_flag_on_with_nothing_recorded_returns_none(self, monkeypatch):
        monkeypatch.setenv("RESEARCH_LAB_PROVIDER_OUTCOME_DIGEST", "true")
        monkeypatch.setenv(PROVIDER_OUTCOME_SIDECAR_ENV, "/nonexistent/provider_outcomes.json")
        assert build_run_provider_outcome_digest() is None


class TestCompactSidecar:
    def test_permissions_hash_and_live_plus_cache_spend_once(self, tmp_path):
        path = tmp_path / "outcomes.json"
        accumulator = ProviderOutcomeSidecarAccumulator(str(path), flush_interval_seconds=60)
        accumulator.record(
            provider_id="exa",
            endpoint_class="/search",
            evidence="recorded",
            status=200,
            live_call=True,
            spend_microusd=7000,
            spend_kind="measured",
        )
        accumulator.record(
            provider_id="exa",
            endpoint_class="/search",
            evidence="hit",
            status=200,
            live_call=False,
            spend_microusd=7000,
            spend_kind="measured",
        )
        accumulator.close()

        assert os.stat(path).st_mode & 0o777 == 0o600
        doc = load_provider_outcome_sidecar(str(path), stale_seconds=3600)
        assert doc is not None
        exa = doc["providers"]["exa"]
        assert exa["call_count"] == 2
        assert exa["live_call_count"] == 1
        assert exa["cache_hit_count"] == 1
        assert exa["measured_spend_microusd"] == 7000
        assert exa["estimated_spend_microusd"] == 0
        serialized = json.dumps(doc, sort_keys=True)
        assert "request_fingerprint" not in serialized
        assert "caller" not in serialized

    def test_failed_call_is_zero_spend_even_if_caller_supplies_estimate(self, tmp_path):
        path = tmp_path / "outcomes.json"
        accumulator = ProviderOutcomeSidecarAccumulator(str(path))
        accumulator.record(
            provider_id="sd",
            endpoint_class="/unknown",
            evidence="live_unrecorded",
            status=500,
            live_call=True,
            spend_microusd=999999,
            spend_kind="estimated",
        )
        accumulator.close()
        doc = load_provider_outcome_sidecar(str(path))
        sd = doc["providers"]["sd"]
        assert sd["live_call_count"] == 1
        assert sd["error_count"] == 1
        assert sd["estimated_spend_microusd"] == 0

    def test_redirect_is_non_success_and_never_records_spend(self, tmp_path):
        path = tmp_path / "outcomes.json"
        accumulator = ProviderOutcomeSidecarAccumulator(str(path))
        accumulator.record(
            provider_id="sd",
            endpoint_class="/redirect",
            evidence="live_unrecorded",
            status=302,
            live_call=True,
            spend_microusd=5000,
            spend_kind="estimated",
        )
        accumulator.close()
        sd = load_provider_outcome_sidecar(str(path))["providers"]["sd"]
        assert sd["error_count"] == 1
        assert sd["estimated_spend_microusd"] == 0

    def test_restart_reload_and_utc_rollover(self, monkeypatch, tmp_path):
        current_day = ["2026-07-10"]
        monkeypatch.setattr(digest_module, "_utc_day", lambda: current_day[0])
        path = tmp_path / "outcomes.json"
        first = ProviderOutcomeSidecarAccumulator(str(path))
        first.record(
            provider_id="exa",
            endpoint_class="/search",
            evidence="recorded",
            status=200,
            live_call=True,
            spend_microusd=100,
            spend_kind="measured",
        )
        first.close()

        second = ProviderOutcomeSidecarAccumulator(str(path))
        second.record(
            provider_id="exa",
            endpoint_class="/search",
            evidence="hit",
            status=200,
            live_call=False,
            spend_microusd=0,
            spend_kind="estimated",
        )
        second.close()
        day_one = load_provider_outcome_sidecar(str(path), stale_seconds=0)
        assert day_one["sequence"] == 2
        assert day_one["providers"]["exa"]["call_count"] == 2

        current_day[0] = "2026-07-11"
        third = ProviderOutcomeSidecarAccumulator(str(path))
        third.record(
            provider_id="exa",
            endpoint_class="/search",
            evidence="recorded",
            status=200,
            live_call=True,
            spend_microusd=200,
            spend_kind="measured",
        )
        third.close()
        day_two = load_provider_outcome_sidecar(str(path), stale_seconds=0)
        assert day_two["utc_day"] == "2026-07-11"
        assert day_two["sequence"] == 1
        assert day_two["providers"]["exa"]["call_count"] == 1

    def test_concurrent_records_are_not_lost_and_endpoints_are_bounded(self, tmp_path):
        path = tmp_path / "outcomes.json"
        accumulator = ProviderOutcomeSidecarAccumulator(str(path), flush_interval_seconds=60)

        def write_batch(worker: int) -> None:
            for index in range(75):
                accumulator.record(
                    provider_id="exa",
                    endpoint_class=f"/endpoint/{worker}/{index}",
                    evidence="hit",
                    status=200,
                    live_call=False,
                    spend_microusd=0,
                    spend_kind="estimated",
                )

        threads = [threading.Thread(target=write_batch, args=(worker,)) for worker in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        accumulator.close()

        doc = load_provider_outcome_sidecar(str(path))
        exa = doc["providers"]["exa"]
        assert exa["call_count"] == 600
        assert exa["cache_hit_count"] == 600
        assert len(exa["endpoints"]) <= MAX_PROVIDER_OUTCOME_ENDPOINTS
        assert "other" in exa["endpoints"]
        assert len(path.read_bytes()) <= MAX_PROVIDER_OUTCOME_SIDECAR_BYTES

    def test_atomic_replacements_never_expose_partial_json(self, tmp_path):
        path = tmp_path / "outcomes.json"
        accumulator = ProviderOutcomeSidecarAccumulator(
            str(path),
            flush_interval_seconds=0.02,
        )
        accumulator.record(
            provider_id="exa",
            endpoint_class="/search",
            evidence="recorded",
            status=200,
            live_call=True,
            spend_microusd=1,
            spend_kind="measured",
        )
        time.sleep(0.04)
        observed_sequences = []

        def writer() -> None:
            for _ in range(80):
                accumulator.record(
                    provider_id="exa",
                    endpoint_class="/search",
                    evidence="hit",
                    status=200,
                    live_call=False,
                    spend_microusd=0,
                    spend_kind="estimated",
                )
                time.sleep(0.001)

        thread = threading.Thread(target=writer)
        thread.start()
        while thread.is_alive():
            doc = load_provider_outcome_sidecar(str(path), stale_seconds=3600, warn=False)
            assert doc is not None
            observed_sequences.append(doc["sequence"])
        thread.join()
        accumulator.close()
        final = load_provider_outcome_sidecar(str(path), stale_seconds=3600, warn=False)
        assert observed_sequences
        assert final["sequence"] == 81
        assert final["providers"]["exa"]["call_count"] == 81

    def test_corrupt_hash_stale_oversized_and_unsafe_mode_are_inert(self, tmp_path):
        path = tmp_path / "outcomes.json"
        accumulator = ProviderOutcomeSidecarAccumulator(str(path))
        accumulator.record(
            provider_id="exa",
            endpoint_class="/search",
            evidence="recorded",
            status=200,
            live_call=True,
            spend_microusd=1,
            spend_kind="measured",
        )
        accumulator.close()
        valid = json.loads(path.read_text(encoding="utf-8"))

        assert load_provider_outcome_sidecar(
            str(path),
            now=float(valid["generated_at_epoch"]) + 3601,
            stale_seconds=3600,
            warn=False,
        ) is None
        valid["sequence"] += 1
        path.write_text(json.dumps(valid), encoding="utf-8")
        os.chmod(path, 0o600)
        assert load_provider_outcome_sidecar(str(path), warn=False) is None

        path.write_bytes(b"x" * (MAX_PROVIDER_OUTCOME_SIDECAR_BYTES + 1))
        os.chmod(path, 0o600)
        assert load_provider_outcome_sidecar(str(path), warn=False) is None

        path.write_text("{}", encoding="utf-8")
        os.chmod(path, 0o644)
        assert load_provider_outcome_sidecar(str(path), warn=False) is None


class TestPromptInjection:
    DIGEST = {"schema_version": "1.0", "providers": {"exa": {"call_count": 3, "error_rate": 0.33}}}

    def test_planner_context_gains_digest_beside_benchmark_summary(self):
        user = build_loop_direction_planner_messages(
            ticket={"ticket_id": "t1"},
            artifact_manifest={},
            component_registry={},
            benchmark_public_summary={"score": 1},
            runtime_source_index={"editable_files": []},
            budget_context={},
            provider_outcome_digest=self.DIGEST,
        )[1]["content"]
        assert "provider_outcome_digest" in user
        assert "benchmark_public_summary" in user

    def test_inspection_and_draft_contexts_gain_digest(self):
        inspection = build_code_edit_source_inspection_messages(
            ticket={"ticket_id": "t1"},
            artifact_manifest={},
            component_registry={},
            benchmark_public_summary={},
            runtime_source_index={"editable_files": []},
            source_inspection_context={},
            budget_context={},
            provider_outcome_digest=self.DIGEST,
        )[1]["content"]
        assert "provider_outcome_digest" in inspection
        draft = build_code_edit_auto_research_messages(
            ticket={"ticket_id": "t1"},
            artifact_manifest={},
            component_registry={},
            benchmark_public_summary={},
            budget_context={},
            max_candidates=1,
            provider_outcome_digest=self.DIGEST,
        )[1]["content"]
        assert "provider_outcome_digest" in draft

    def test_absent_digest_keeps_prompts_unchanged(self):
        user = build_loop_direction_planner_messages(
            ticket={"ticket_id": "t1"},
            artifact_manifest={},
            component_registry={},
            benchmark_public_summary={},
            runtime_source_index={"editable_files": []},
            budget_context={},
        )[1]["content"]
        assert "provider_outcome_digest" not in user
