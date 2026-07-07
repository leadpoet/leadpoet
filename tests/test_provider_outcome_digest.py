"""W2 provider-outcome digest: aggregation, sanitization, prompt injection."""

from __future__ import annotations

import json
import time

from gateway.research_lab.provider_outcome_digest import (
    build_provider_outcome_digest,
    build_run_provider_outcome_digest,
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
        monkeypatch.delenv("RESEARCH_LAB_PROVIDER_OUTCOME_DIGEST", raising=False)
        assert build_run_provider_outcome_digest() is None

    def test_flag_on_reads_todays_ledger_and_day_cache(self, monkeypatch, tmp_path):
        today = time.strftime("%Y-%m-%d", time.gmtime())
        ledger = tmp_path / "ledger.jsonl"
        ledger.write_text(
            "\n".join(
                [
                    json.dumps({"utc_day": today, "provider_id": "sd", "endpoint_class": "/scrape", "status": 200, "evidence": "recorded"}),
                    json.dumps({"utc_day": "2020-01-01", "provider_id": "sd", "endpoint_class": "/scrape", "status": 200, "evidence": "recorded"}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        day_cache = tmp_path / "day.json"
        day_cache.write_text(
            json.dumps({"utc_day": today, "entries": {"fp": {"status": 200, "outcome": "success", "body_b64": "c2VjcmV0"}}}),
            encoding="utf-8",
        )
        monkeypatch.setenv("RESEARCH_LAB_PROVIDER_OUTCOME_DIGEST", "true")
        monkeypatch.setenv("RESEARCH_LAB_PROVIDER_USAGE_LEDGER_PATH", str(ledger))
        monkeypatch.setenv("RESEARCH_LAB_PROVIDER_EVIDENCE_DAY_CACHE", str(day_cache))
        digest = build_run_provider_outcome_digest()
        assert digest is not None
        assert digest["providers"]["sd"]["call_count"] == 1  # stale day filtered
        assert digest["day_cache_outcomes"] == {"success": 1}
        assert "body_b64" not in json.dumps(digest)

    def test_flag_on_with_nothing_recorded_returns_none(self, monkeypatch):
        monkeypatch.setenv("RESEARCH_LAB_PROVIDER_OUTCOME_DIGEST", "true")
        monkeypatch.setenv("RESEARCH_LAB_PROVIDER_USAGE_LEDGER_PATH", "/nonexistent/ledger.jsonl")
        monkeypatch.setenv("RESEARCH_LAB_PROVIDER_EVIDENCE_DAY_CACHE", "/nonexistent/day.json")
        assert build_run_provider_outcome_digest() is None


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
