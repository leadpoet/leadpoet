"""Quoted credentials retain exact redaction without retrying escape suffixes."""
import gzip
import json
import random
import re

import pytest

from lab_arena import contracts, trajectory, training_capture

# Frozen pre-fix patterns from base a95ab29d8. Never use the legacy patterns
# on unbounded input; they are reference oracles for short parity controls.
LEGACY_CONTEXT = re.compile(
    r"(?i)((?:\\*[\"'])(?:api[-_ ]?key|x[-_]api[-_]key|(?:proxy[-_])?authorization|"
    r"password|passwd|secret|client[-_]secret|access[-_ ]?token|refresh[-_ ]?token|"
    r"lease(?:[-_ ]?token)?|cookie|set[-_]cookie|credentials?|private[-_]key|"
    r"secret[-_]key|session[-_]token|x[-_]amz[-_](?:credential|signature|security[-_]token))"
    r"(?:\\*[\"'])\s*:\s*(?:\\*[\"']))"
)
LEGACY_VALUE = re.compile(LEGACY_CONTEXT.pattern + r"([^\\\"'\r\n]+)")


@pytest.mark.parametrize("pattern_name", ["_QUOTED_SENSITIVE_CONTEXT_RE", "_QUOTED_SENSITIVE_VALUE_RE"])
def test_quoted_sensitive_scan_only_starts_at_escape_run_boundary(pattern_name):
    text = "\\" * 64 + '"authorization":"synthetic-private-value"'
    pattern = getattr(trajectory, pattern_name)
    first = pattern.match(text)
    assert first is not None and first.start() == 0
    # The old regex matched every suffix of the same run. Excluding those
    # starts is the work bound; it is independent of test-host timing.
    assert all(pattern.match(text, position) is None for position in range(1, 65))


def scalar_parity(text, monkeypatch):
    expected_context = LEGACY_CONTEXT.search(text)
    actual_context = trajectory._QUOTED_SENSITIVE_CONTEXT_RE.search(text)
    assert bool(actual_context) == bool(expected_context)
    if expected_context is not None:
        assert (actual_context.span(), actual_context.groups()) == (expected_context.span(), expected_context.groups())
    actual = (trajectory.redact_auth_schemes(text), trajectory.redact_text(text),
              training_capture.sanitize_training_value(text, secrets=("exact-known-private",)))
    with monkeypatch.context() as old:
        old.setattr(trajectory, "_QUOTED_SENSITIVE_CONTEXT_RE", LEGACY_CONTEXT)
        old.setattr(trajectory, "_QUOTED_SENSITIVE_VALUE_RE", LEGACY_VALUE)
        expected = (trajectory.redact_auth_schemes(text), trajectory.redact_text(text),
                    training_capture.sanitize_training_value(text, secrets=("exact-known-private",)))
    assert actual == expected


@pytest.mark.parametrize("label", ["authorization", "proxy_authorization", "api-key", "x-api-key",
                                   "password", "client_secret", "x-amz-security-token"])
@pytest.mark.parametrize("quote", ['"', "'"])
@pytest.mark.parametrize("escapes", [0, 1, 2, 3, 8, 64])
def test_escaped_private_context_preserves_exact_redaction(monkeypatch, label, quote, escapes):
    boundary = "\\" * escapes + quote
    text = "public prefix " + boundary + label + boundary + ": " + boundary + "synthetic-private-value" + boundary
    text += " ; Basic syntheticpublicprose ; exact-known-private"
    scalar_parity(text, monkeypatch)
    assert "synthetic-private-value" not in trajectory.redact_auth_schemes(text)


@pytest.mark.parametrize("text", [
    '"authorization":"[REDACTED]"',
    r'\"authorization\":\"[REDACTED]\"',
    '"authorization":"[REDACTED]]"',
    '"authorization":"Bearer [REDACTED]private-tail"',
    '"authorization":"Basic [REDACTED]/private-tail"',
    '"authorization":',
    '"authorization":\\\\',
    '\\\\"authorization',
    '\\\\"authorization" : \\\\',
    '"authorization" : "unterminated-private',
    "'password' : 'unterminated-private",
    '"public":"basic company products"',
    '"authorization" : "" ; basic company products',
    '\\\\"authorization\" : \\\\"synthetic-private-value\\\"',
])
def test_markers_and_malformed_quotes_keep_legacy_output(monkeypatch, text):
    scalar_parity(text, monkeypatch)


@pytest.mark.parametrize("depth", [0, 1, 2, 3, 4])
def test_capture_pointer_raw_hash_and_gzip_keep_legacy_output(monkeypatch, depth):
    text = {"authorization": "synthetic-private-value", "public": "basic company products",
            "literal": "Bearer [REDACTED]", "attached": "Bearer [REDACTED]private-tail"}
    for _ in range(depth):
        text = json.dumps(text)
    source = {"operation_id": "openrouter.chat", "admitted_parameters": {
        "messages": [{"role": "user", "content": text}]} ,
        "known": "exact-known-private", "encrypted_content": "opaque-private-value"}
    current = training_capture.RequestCapture(None).prepare(source, secrets=("exact-known-private",))
    with monkeypatch.context() as old:
        old.setattr(trajectory, "_QUOTED_SENSITIVE_CONTEXT_RE", LEGACY_CONTEXT)
        old.setattr(trajectory, "_QUOTED_SENSITIVE_VALUE_RE", LEGACY_VALUE)
        legacy = training_capture.RequestCapture(None).prepare(source, secrets=("exact-known-private",))
    assert isinstance(current, training_capture.PreparedRequest) and isinstance(legacy, training_capture.PreparedRequest)
    assert current.pointer == legacy.pointer and current.compressed == legacy.compressed
    assert current.pointer["sha256"] == contracts.hash_bytes(gzip.decompress(current.compressed))
    assert current.pointer["redactions"] == ["credential", "hidden_reasoning"]


def test_bounded_generated_short_contexts_keep_legacy_output(monkeypatch):
    rng = random.Random(71103)
    parts = ['\\', '"', "'", "authorization", "api_key", "password", "private-value",
             "[REDACTED]", "Bearer", "Basic", ":", "=", " ", "\n", "]", "public"]
    for _ in range(500):
        text = "".join(rng.choice(parts) for _ in range(rng.randrange(1, 25)))
        scalar_parity(text, monkeypatch)


@pytest.mark.parametrize("tail", ["", '"public"', "not-a-quote"])
def test_bounded_long_escape_runs_remain_exact(tail):
    # At most the already-tested 8 KiB run; no large-run latency assertion.
    text = "\\" * 8192 + tail
    assert trajectory.redact_auth_schemes(text) == text
