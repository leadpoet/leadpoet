"""Event-doc error sanitization (bug #36 write-side discipline).

Raw ``str(exc)`` in an event doc either poisons the audit-bundle secret scan
or trips the scripts/34 event_doc CHECK — and because event writes are
try/except-contained, a CHECK rejection silently DROPS the event (a lost
negative training example). These tests pin the shared helpers in
``logging_utils`` and, via a source scan, that ``code_loop_engine.py`` never
regresses to writing raw exception text into event docs.
"""

from __future__ import annotations

import re
from pathlib import Path

from gateway.research_lab.logging_utils import (
    event_error_diagnostics,
    runtime_error_diagnostics,
    safe_event_error_text,
)

ENGINE_SOURCE = (
    Path(__file__).resolve().parents[1] / "gateway" / "research_lab" / "code_loop_engine.py"
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def test_safe_event_error_text_redacts_secret_markers() -> None:
    exc = RuntimeError("provider rejected sk-or-abc123 with SERVICE_ROLE=x openrouter_api_key=y")
    text = safe_event_error_text(exc)
    assert "sk-or-" not in text.lower()
    assert "service_role" not in text.lower()
    assert "openrouter_api_key" not in text.lower()
    assert text.count("[redacted]") >= 3
    assert text.startswith("RuntimeError:")


def test_safe_event_error_text_redacts_credentialed_urls() -> None:
    exc = RuntimeError("push failed for https://user:supersecret@registry.example.com/repo")
    text = safe_event_error_text(exc)
    assert "supersecret" not in text
    assert "https://[redacted]@" in text


def test_safe_event_error_text_caps_length() -> None:
    assert len(safe_event_error_text(RuntimeError("x" * 5000))) <= 500


def test_event_error_diagnostics_shape() -> None:
    doc = event_error_diagnostics(ValueError("Exa http error 429 status=429"))
    assert doc["error_class"] == "ValueError"
    assert doc["provider"] == "exa"
    assert doc["status"] == 429
    assert doc["category"] == "provider_http_4xx"
    # Never carries free-form error text.
    assert set(doc) == {"error_class", "provider", "status", "category"}


def test_runtime_error_diagnostics_5xx_category() -> None:
    doc = runtime_error_diagnostics("scrapingdog HTTP Error 503")
    assert doc["provider"] == "scrapingdog"
    assert doc["category"] == "provider_http_5xx"


def test_scoring_worker_aliases_stay_wired() -> None:
    import gateway.research_lab.scoring_worker as sw

    assert sw._safe_event_error_text is safe_event_error_text
    assert sw._event_error_diagnostics is event_error_diagnostics
    assert sw._runtime_error_diagnostics is runtime_error_diagnostics


# ---------------------------------------------------------------------------
# Source pinning: no raw str(exc) in loop-engine event docs
# ---------------------------------------------------------------------------


def test_loop_engine_event_docs_never_store_raw_exception_text() -> None:
    source = ENGINE_SOURCE.read_text()
    # Event-doc error fields must use safe_event_error_text(exc); the raw
    # sliced form is exactly what bug #36 banned. Hash fields legitimately
    # hash the raw text (sha256_json({"error": str(exc)}) leaks nothing) and
    # are excluded: the banned pattern is the *sliced raw text* form.
    banned = re.findall(r'"[a-z_]*error[a-z_]*":\s*str\(exc\)\[', source)
    assert banned == [], f"raw exception text in event fields: {banned}"
    assert "reason=str(exc)" not in source


def test_loop_engine_uses_the_shared_sanitizer() -> None:
    source = ENGINE_SOURCE.read_text()
    assert "from gateway.research_lab.logging_utils import safe_event_error_text" in source
    assert source.count("safe_event_error_text(exc)") >= 10
