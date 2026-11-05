"""Memoize exact long leaves within one privacy walk, retaining every boundary."""
from collections import Counter
import gzip
import json

import pytest

from lab_arena import contracts, trajectory, training_capture as capture


class CustomString(str):
    pass


class CustomSecret(str):
    pass


def count_scans(monkeypatch, texts):
    counts = Counter()
    original = trajectory.redact_text
    def counted(text, *args, **kwargs):
        if text in texts:
            counts[text] += 1
        return original(text, *args, **kwargs)
    monkeypatch.setattr(trajectory, "redact_text", counted)
    return counts


def capture_document(texts):
    parameters = {"model": "openai/gpt-4o-mini", "messages": [
        {"role": "user", "content": text} for text in texts],
        "reasoning": {"effort": "high", "exclude": True}}
    return {"operation_id": "openrouter.chat", "effective_operation_id": "openrouter.chat",
            "request_hash": contracts.document_hash(parameters), "hashed_parameters": parameters,
            "admitted_parameters": parameters, "normalized_parameters": parameters,
            "effective_parameters": parameters, "outbound": {"body": parameters}, "route": {}}


def test_standalone_broker_capture_scans_each_repeated_long_leaf_once(monkeypatch):
    texts = {"public company evidence " * 1_000, "other retained company evidence " * 700}
    counts = count_scans(monkeypatch, texts)
    document = capture_document(sorted(texts))
    prepared = capture.RequestCapture(None).prepare(document, secrets=("fixed-provider-secret",))
    assert isinstance(prepared, capture.PreparedRequest)
    # The broker retains five projections of each message. This is a work
    # bound, not a wall-clock assertion that depends on the test host's speed.
    assert counts == {text: 1 for text in texts}
    expected = dict(document, schema_version=capture.SCHEMA_VERSION,
                    privacy={"redacted": False, "redactions": []})
    raw = contracts.canonical_json(expected).encode()
    assert gzip.decompress(prepared.compressed) == raw
    assert prepared.compressed == gzip.compress(raw, compresslevel=6, mtime=0)
    assert prepared.pointer["sha256"] == contracts.hash_bytes(raw)


def test_separate_walks_recheck_known_secret_and_url_variants():
    from urllib.parse import quote, quote_plus
    secret = "private value+/="
    public = "public source evidence " * 1_000
    text = public + " " + secret + " " + quote(secret, safe="") + " " + quote_plus(secret)
    value = {"content": [text] * 5}
    hidden, reasons = capture.sanitize_training_value(value, secrets=(secret,))
    assert reasons == ["credential"]
    assert hidden == {"content": [public + " [REDACTED] [REDACTED] [REDACTED]"] * 5}
    retained, reasons = capture.sanitize_training_value(value, secrets=("another-private-value",))
    assert retained == value and reasons == []
    assert capture.sanitize_training_value(value, secrets=(secret,)) == (hidden, ["credential"])


def test_cached_long_text_keeps_path_key_and_opaque_reasoning_checks():
    text = "public assistant replay evidence " * 700
    value = {"operation_id": "openrouter.chat", "admitted_parameters": {
        "messages": [{"role": "assistant", "reasoning_content": text},
                     {"role": "user", "content": text}]},
        "response": {"reasoning_content": text}, "authorization": text,
        "reasoning_details": [{"type": "reasoning.encrypted", "data": text}]}
    cleaned, reasons = capture.sanitize_training_value(value)
    assert cleaned["admitted_parameters"] == value["admitted_parameters"]
    assert cleaned["response"]["reasoning_content"] == "[REDACTED]"
    assert cleaned["authorization"] == "[REDACTED]"
    assert cleaned["reasoning_details"] == ["[REDACTED]"]
    assert reasons == ["credential", "hidden_reasoning"]


def test_repeated_long_credentials_and_urls_keep_every_redaction_class():
    public = "public company text " * 1_000
    text = public + " Bearer synthetic-private-token https://user:pass@example.test/?token=unknown-token"
    value = {"content": [text] * 5, "encrypted_content": "opaque private value"}
    cleaned, reasons = capture.sanitize_training_value(value)
    assert cleaned == {"content": [public + " [REDACTED] https://example.test/?token=%5BREDACTED%5D"] * 5,
                       "encrypted_content": "[REDACTED]"}
    assert reasons == ["credential", "hidden_reasoning"]


def test_long_key_redaction_collision_and_cached_node_limit_still_fail():
    text = "public field name " * 1_000
    with pytest.raises(capture.CaptureError, match="request_redaction_key_collision"):
        capture.sanitize_training_value({text + "private-value": "one", text + "[REDACTED]": "two"},
                                        secrets=("private-value",))
    with pytest.raises(capture.CaptureError, match="request_structure_limit"):
        capture.sanitize_training_value({"first": text, "later": [text] * 5}, max_nodes=5)


@pytest.mark.parametrize("custom", ["text", "secret"])
def test_custom_string_or_secret_keeps_existing_uncached_behavior(monkeypatch, custom):
    plain = "public custom source text " * 1_000
    text = CustomString(plain) if custom == "text" else plain
    secret = CustomSecret("unused-synthetic-secret") if custom == "secret" else "unused-synthetic-secret"
    counts = count_scans(monkeypatch, {plain})
    value = {"content": [text] * 4}
    assert capture.sanitize_training_value(value, secrets=(secret,)) == (value, [])
    assert counts[plain] == 4


def test_wide_unicode_long_cache_evicts_at_existing_eight_mib_budget(monkeypatch):
    texts = ["\U0001f642" * 400_000 + str(index) for index in range(3)]
    # Each entry is about 3.2 MiB when both retained Unicode strings are
    # charged. Two fit; the third evicts the first before its final reuse.
    charge = 2 * str.__sizeof__(texts[0])
    assert 2 * charge <= 8 * 1024**2 < 3 * charge
    counts = count_scans(monkeypatch, set(texts))
    value = {"content": texts + [texts[0]]}
    assert capture.sanitize_training_value(value) == (value, [])
    assert counts == {texts[0]: 2, texts[1]: 1, texts[2]: 1}


@pytest.mark.parametrize("suffix", [
    " ordinary Basic company description",
    " Authorization: Basic dXNlcjpwYXNz",
    ' {"authorization":"synthetic-value"}',
    "\x00",
    " https://user:pass@example.test/?token=unknown-token",
])
def test_cached_prepared_pointer_raw_hash_and_gzip_match_uncached_reference(suffix):
    text = "public source \u0100\U0001f642 evidence " * 1_000 + suffix
    document = capture_document([text, text])
    cached = capture.RequestCapture(None).prepare(document, secrets=("known-private-value",))
    # A custom secret disables only the existing built-in long-text lane.
    # Its bytes do not occur in this document; generic privacy still runs.
    sentinel = CustomSecret("unused-reference-secret")
    assert str(sentinel) not in json.dumps(document)
    uncached = capture.RequestCapture(None).prepare(document, secrets=("known-private-value", sentinel))
    assert isinstance(cached, capture.PreparedRequest) and isinstance(uncached, capture.PreparedRequest)
    assert cached.pointer == uncached.pointer
    assert cached.compressed == uncached.compressed


def test_real_broker_capture_projections_keep_outbound_billing_and_stored_bytes(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from lab_arena import broker
    from lab_arena.service import LocalObjectStore
    from tests.lab_arena.test_lab_arena_broker import CONTEXT, FakeTransport, HOST_KEYS, make_broker
    text = "retained public company evidence " * 700
    # Compare exact transport arguments under one clock, rather than comparing
    # naturally different remaining timeout fractions after CPU work.
    monkeypatch.setattr(broker, "time", SimpleNamespace(monotonic=lambda: 100.0, sleep=lambda _: None))
    counts = count_scans(monkeypatch, {text})
    results = []
    for mode in ("cached", "reference"):
        objects = LocalObjectStore(tmp_path / mode)
        def prepare(document, *, secrets=()):
            # Match production wiring's fixed provider-key plus call-secret
            # boundary. A custom secret gives the uncached reference path.
            selected = (*HOST_KEYS.values(), *secrets)
            if mode == "reference":
                selected = tuple(CustomSecret(secret) for secret in selected)
            return capture.RequestCapture(objects).prepare(document, secrets=selected)
        provider, store, transport = make_broker(capture_request=prepare,
            transport=FakeTransport([(200, {"usage": {"cost": "0.000001"}})]))
        result = provider.execute(CONTEXT, operation_id="openrouter.chat", parameters={
            "model": "openai/gpt-4o-mini", "messages": [{"role": "user", "content": text}],
        }, action_sequence=1, timeout_ms=5000)
        assert result.status == 200 and len(transport.sent) == 1 and len(store.calls) == 1
        call = next(iter(store.calls.values()))
        pointer = call["call_doc"]["request_capture"]
        restored = capture.read_request(objects, pointer)
        assert restored["outbound"]["body"] == json.loads(transport.sent[0]["body"])
        assert restored["outbound"]["body_sha256"] == contracts.hash_bytes(transport.sent[0]["body"])
        assert contracts.document_hash(restored["hashed_parameters"]) == restored["request_hash"]
        assert call["actual"] == 1 and call["kind"] == "settlement"
        results.append((result.to_document(), store.calls, transport.sent, pointer, objects.get(pointer["ref"])))
        if mode == "cached":
            assert counts[text] == 1
            counts.clear()
        else:
            assert counts[text] == 5
    assert results[0] == results[1]
