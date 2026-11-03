"""Full private request evidence, receipt joins, and observational failures."""

from __future__ import annotations

import gzip
import base64
import copy
import hashlib
import json
from dataclasses import replace
from urllib.parse import quote

import pytest

from lab_arena import broker as br, contracts, operations, trajectory, training_capture as capture, training_export
from lab_arena.service import LocalObjectStore
from lab_arena.store import ArenaStoreUnavailable
from tests.lab_arena.test_lab_arena_broker import (
    CONTEXT, FakeLedgerStore, FakeTransport, HOST_KEYS, make_broker,
)


def _document():
    parameters = {
        "model": "openai/gpt-4o-mini",
        "messages": [{"role": "user", "content": f"message {index}: " + "é" * 2000}
                     for index in range(12)],
        "tools": [{"type": "function", "function": {
            "name": "find_company_" + str(index),
            "description": "tool description " + "z" * 1000,
            "parameters": {"type": "object", "properties": {
                "address": {"type": "string", "description": "deep schema " + "y" * 1000},
            }},
        }} for index in range(15)],
        "reasoning": {"effort": "high", "exclude": True},
        "temperature": 0.2, "seed": 17,
    }
    return {
        "operation_id": "openrouter.chat", "admitted_parameters": parameters,
        "request_hash": contracts.document_hash(parameters), "hashed_parameters": parameters,
        "normalized_parameters": parameters, "effective_parameters": parameters,
        "effective_operation_id": "openrouter.chat", "route": {},
        "outbound": {"body": parameters},
    }


def test_full_long_prompt_tools_and_safe_config_roundtrip_without_projection(tmp_path):
    objects = LocalObjectStore(tmp_path)
    source = _document()
    pointer = capture.RequestCapture(objects)(source)
    assert pointer["status"] == "stored"
    assert pointer["redacted"] is False
    restored = capture.read_request(objects, pointer)
    for field in source:
        assert restored[field] == source[field]
    assert len(restored["admitted_parameters"]["messages"]) == 12
    assert len(restored["admitted_parameters"]["tools"]) == 15
    assert "[TRUNCATED" not in json.dumps(restored)


def test_deterministic_content_address_and_gzip_ignore_invocation_identity(tmp_path):
    objects = LocalObjectStore(tmp_path)
    writer = capture.RequestCapture(objects)
    first = writer(_document())
    encoded = objects.get(first["ref"])
    second = writer(_document())
    assert first == second
    assert encoded == objects.get(second["ref"])
    assert encoded[4:8] == b"\0\0\0\0"
    assert "sha256:" + hashlib.sha256(gzip.decompress(encoded)).hexdigest() == first["sha256"]
    assert len(list(tmp_path.rglob("*.gz"))) == 1


def test_secret_redaction_is_explicit_without_changing_safe_reasoning_config(tmp_path):
    secret = "private+/=provider-key"
    document = _document()
    document["admitted_parameters"]["messages"][0]["content"] = (
        f"literal {secret} encoded {quote(secret, safe='')} "
        "https://user:unknown-password@example.test/path?token=unknown-token&ok=1 "
        "https://example.test/path?X-Amz-Signature=unknown-signature"
    )
    document["other"] = {
        "authorization": "another-secret", "api_key": "unrelated-secret",
        "reasoning_content": "private trace", "encrypted_content": "opaque-private-state",
        "reasoning": {"content": "hidden trace"},
    }
    objects = LocalObjectStore(tmp_path)
    pointer = capture.RequestCapture(objects)(document, secrets=(secret,))
    restored = capture.read_request(objects, pointer)
    encoded = json.dumps(restored)
    for private in (secret, quote(secret, safe=""), "unknown-password", "unknown-token",
                    "unknown-signature", "another-secret", "unrelated-secret",
                    "private trace", "opaque-private-state", "hidden trace"):
        assert private not in encoded
    assert pointer["redacted"] is True
    assert pointer["redactions"] == ["credential", "hidden_reasoning"]
    assert restored["admitted_parameters"]["reasoning"] == {"effort": "high", "exclude": True}


@pytest.mark.parametrize("encoding_depth", [0, 1, 2])
@pytest.mark.parametrize("value", [
    r'{\"url\":\"https://example.com/?api_key=secret,\"next\":\"data\",\"last\":\"more\"}',
    r'{\"url\":\"https://example.com/?api_key=[REDACTED],\"last\":\"more\"}',
])
def test_escaped_json_url_redaction_preserves_fields_and_is_a_fixed_point(value, encoding_depth):
    for _ in range(encoding_depth):
        value = json.dumps(value)
    cleaned, _ = capture.sanitize_training_value(value)
    assert cleaned == value.replace("=secret", "=%5BREDACTED%5D")
    assert capture.sanitize_training_value(cleaned)[0] == cleaned


def test_escaped_url_boundary_preserves_query_values_and_known_secret_redaction():
    from urllib.parse import parse_qsl, urlsplit

    ordinary = "https://example.com/?q=one,two;three&keep=visible"
    assert capture.sanitize_training_value(ordinary) == (ordinary, [])
    sensitive = ordinary + "&token=private-token"
    cleaned, reasons = capture.sanitize_training_value(sensitive)
    assert dict(parse_qsl(urlsplit(cleaned).query)) == {
        "q": "one,two;three", "keep": "visible", "token": "[REDACTED]"}
    assert reasons == ["credential"]
    assert capture.sanitize_training_value(cleaned)[0] == cleaned
    value = r'{\"url\":\"https://example.com/?api_key=secret;\"next\":\"known-private-token\"}'
    cleaned, reasons = capture.sanitize_training_value(value, secrets=("known-private-token",))
    assert cleaned == value.replace("=secret", "=%5BREDACTED%5D").replace("known-private-token", "[REDACTED]")
    assert reasons == ["credential"]
    assert capture.sanitize_training_value(cleaned)[0] == cleaned


@pytest.mark.parametrize("value,expected", [
    ("https://example.com/?api_key=value#;fragment",
     "https://example.com/?api_key=%5BREDACTED%5D#;fragment"),
    ("https://example.com/?api_key=value&keep=visible#;fragment",
     "https://example.com/?api_key=%5BREDACTED%5D&keep=visible#;fragment"),
    ("https://example.com/?api_key=[REDACTED]actualsecret&keep=visible#fragment",
     "https://example.com/?api_key=%5BREDACTED%5D&keep=visible#fragment"),
    ("https://user:password@example.com/?signature=private&keep=visible#fragment",
     "https://example.com/?signature=%5BREDACTED%5D&keep=visible#fragment"),
    ("https://example.com/?api_key=[REDACTED]&keep=visible#;fragment",
     "https://example.com/?api_key=[REDACTED]&keep=visible#;fragment"),
])
def test_url_component_redaction_preserves_delimiters_and_is_a_fixed_point(value, expected):
    assert capture.sanitize_training_value(value)[0] == expected
    assert capture.sanitize_training_value(expected)[0] == expected


PUBLIC_ARTICLE_URL = (
    "https://www.prnewswire.com/news-releases/"
    "biohaven-and-sk-biopharmaceuticals-enter-into-strategic-global-licensing-"
    "agreement-for-novel-kv7-ion-channel-platform-and-opakalim-lead-candidate-"
    "for-epilepsy-302860299.html"
)


def test_public_article_slug_survives_standalone_nested_and_aggregate_capture():
    for value in (PUBLIC_ARTICLE_URL,
                  json.dumps({"url": PUBLIC_ARTICLE_URL}),
                  {"output": {"companies": [{"intent_signals": [{"url": PUBLIC_ARTICLE_URL}]}]}},
                  {"provider_calls": [{"response": {"url": PUBLIC_ARTICLE_URL}}],
                   "output": {"company_stage_evidence": [{"url": PUBLIC_ARTICLE_URL}]}}):
        assert capture.sanitize_training_value(value) == (value, [])
        assert capture.sanitize_training_value(value, aggregate_parts=True) == (value, [])


def test_public_article_slug_survives_request_storage(tmp_path):
    document = _document()
    document["admitted_parameters"]["messages"][0]["content"] = PUBLIC_ARTICLE_URL
    objects = LocalObjectStore(tmp_path)
    pointer = capture.RequestCapture(objects)(document)
    assert pointer["redacted"] is False and pointer["redactions"] == []
    assert capture.read_request(objects, pointer)["admitted_parameters"]["messages"][0]["content"] == PUBLIC_ARTICLE_URL


def test_public_article_slug_survives_native_receipt_export():
    response = {"company_stage_evidence": [{"url": PUBLIC_ARTICLE_URL}],
                "intent_signals": [{"url": PUBLIC_ARTICLE_URL}]}
    terminal = {"body_b64": base64.b64encode(json.dumps(response).encode()).decode()}
    assert training_export._response_value(terminal, "openrouter") == (response, [])


def test_public_article_slug_does_not_hide_real_credentials():
    token = "sk-TestOnly0123456789AbCdEfGhIjKlMnOpQrStUvWxYz"
    known = "private+/=credential-value"
    cases = (
        PUBLIC_ARTICLE_URL + "?token=" + token,
        "https://user:password@example.test/path/" + token,
        PUBLIC_ARTICLE_URL + " Authorization: Bearer " + token,
        "https://example.test/path/" + token,
        PUBLIC_ARTICLE_URL + " " + known + " https://example.test/path",
        "https://example.test/" + known + "/tail " + PUBLIC_ARTICLE_URL,
        PUBLIC_ARTICLE_URL + "?next=" + quote(known, safe=""),
        "https://example.test/" + quote(known, safe="") + "/tail " + PUBLIC_ARTICLE_URL,
    )
    for value in cases:
        cleaned, reasons = capture.sanitize_training_value(value, secrets=(known,))
        assert reasons == ["credential"]
        assert token not in cleaned and known not in cleaned and quote(known, safe="") not in cleaned
        assert capture.sanitize_training_value(cleaned)[0] == cleaned


@pytest.mark.parametrize("value", [
    "https://example.test/story-and-sk-abcdefghijklmnop-for-qrstuvwxyz-uvwxzy-foobar",
    "https://example.test/story-and-sk-abcdefghijklmnop-for-qrstuvwxyz-uvwxzy-foobar.html",
    "https://example.test/story-and-sk-Abcdefghijklmnop-for-qrstuvwxyz-uvwxzy-123456789.html",
    "https://example.test/story?next=sk-abcdefghijklmnop-for-qrstuvwxyz-uvwxzy-123456789",
    "https://example.test/story-and-sk-abcdefghijklmnop-for-qrstuvwxyz-uvwxzy-foobar-bazbaz-123456789.html",
    "https://example.test/story-and-sk-or-v1-abcdefghijklmnop-for-news-report-123456.html",
])
def test_lowercase_hyphenated_secret_shape_in_url_still_redacts(value):
    cleaned, reasons = capture.sanitize_training_value(value)
    assert reasons == ["credential"]
    assert "sk-" not in cleaned


def test_exact_known_secret_inside_public_article_slug_still_redacts():
    known = PUBLIC_ARTICLE_URL.split("and-", 1)[1].removesuffix(".html")
    cleaned, reasons = capture.sanitize_training_value(PUBLIC_ARTICLE_URL, secrets=(known,))
    assert known not in cleaned and reasons == ["credential"]
    assert capture.sanitize_training_value(PUBLIC_ARTICLE_URL, secrets=(PUBLIC_ARTICLE_URL,)) == (
        "[REDACTED]", ["credential"])


@pytest.mark.parametrize("secret", ["arena", "public", "slug", "0-end", "a" * 32])
def test_known_secrets_do_not_redact_internal_public_url_placeholders(monkeypatch, secret):
    from types import SimpleNamespace

    monkeypatch.setattr(capture.uuid, "uuid4", lambda: SimpleNamespace(hex="a" * 32))
    for value in (PUBLIC_ARTICLE_URL, json.dumps({"url": PUBLIC_ARTICLE_URL}),
                  {"evidence": [PUBLIC_ARTICLE_URL, PUBLIC_ARTICLE_URL]}):
        for aggregate in (False, True):
            assert capture.sanitize_training_value(
                value, secrets=(secret,), aggregate_parts=aggregate) == (value, [])


def test_public_url_placeholder_does_not_hide_decoded_query_or_real_secret(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(capture.uuid, "uuid4", lambda: SimpleNamespace(hex="a" * 32))
    secret = "arena"
    # Mixed percent encoding is decoded by query parsing, after the first
    # whole-text exact-secret scan. That second check must remain active.
    value = PUBLIC_ARTICLE_URL + "?next=%61re%6Ea#arena"
    cleaned, reasons = capture.sanitize_training_value(value, secrets=(secret,))
    assert cleaned.startswith(PUBLIC_ARTICLE_URL + "?")
    assert "%5BREDACTED%5D" in cleaned and cleaned.endswith("#[REDACTED]")
    assert reasons == ["credential"] and "arena-public-slug" not in cleaned
    assert capture.sanitize_training_value(
        "Authorization: " + PUBLIC_ARTICLE_URL, secrets=(secret,)) == (
            "Authorization: [REDACTED]", ["credential"])


@pytest.mark.parametrize("prefix", ["Bearer sk-", "sk-abcdefghijklmnop"])
def test_public_url_placeholder_survives_credential_prefix_removing_scheme(monkeypatch, prefix):
    from types import SimpleNamespace

    monkeypatch.setattr(capture.uuid, "uuid4", lambda: SimpleNamespace(hex="a" * 32))
    value = prefix + PUBLIC_ARTICLE_URL
    expected = capture.sanitize_training_value(value)
    assert expected[1] == ["credential"]
    assert capture.sanitize_training_value(value, secrets=("arena",)) == expected
    assert "arena-public-slug" not in expected[0]


def test_verified_public_url_exception_does_not_cover_other_hosts_or_auth_contexts():
    lookalike = PUBLIC_ARTICLE_URL.replace("www.prnewswire.com", "example.test")
    assert capture.sanitize_training_value(lookalike)[1] == ["credential"]
    assert capture.sanitize_training_value("Authorization: " + PUBLIC_ARTICLE_URL) == (
        "Authorization: [REDACTED]", ["credential"])
    assert capture.sanitize_training_value("password=" + PUBLIC_ARTICLE_URL) == (
        "password=[REDACTED]", ["credential"])


def test_many_exact_public_article_urls_stay_exact_with_bounded_cache():
    repeated = " ".join([PUBLIC_ARTICLE_URL] * 12)
    value = {"input_snapshots": {"run": [repeated] * 12},
             "provider_calls": [{"response": repeated}]}
    assert capture.sanitize_training_value(value, aggregate_parts=True) == (value, [])


@pytest.mark.parametrize("value,secret", [
    ("See https://example.com/private-path now", "https://example.com/private-path"),
    (r'\"password\":\"https://example.com/private-path\"', "https://example.com/private-path"),
    ("https://private-user@example.com/", "private-user@example.com"),
    ("https://[malformed/?password=private-secret", "private-secret"),
])
def test_url_boundaries_do_not_bypass_known_secret_or_malformed_url_redaction(value, secret):
    cleaned, reasons = capture.sanitize_training_value(value, secrets=(secret,))
    assert secret not in cleaned and reasons == ["credential"]
    assert capture.sanitize_training_value(cleaned)[0] == cleaned


@pytest.mark.parametrize("value,expected", [
    ("password=https://example.com/private", "password=[REDACTED]"),
    ('"password":"https://example.com/private"', '"password":"[REDACTED]"'),
])
def test_a_url_as_a_text_password_keeps_existing_redaction(value, expected):
    assert capture.sanitize_training_value(value)[0] == expected


@pytest.mark.parametrize("encoding_depth", [0, 1, 2, 3])
def test_a_url_password_is_redacted_with_its_complete_surrounding_syntax(encoding_depth):
    from lab_arena import trajectory

    value = "password=https://example.com/?api_key=value&keep=visible"
    for _ in range(encoding_depth):
        value = json.dumps(value)
    cleaned, _ = capture.sanitize_training_value(value)
    assert cleaned == trajectory.redact_text(value)
    assert capture.sanitize_training_value(cleaned)[0] == cleaned


@pytest.mark.parametrize("label", ["Basic", "Bearer"])
def test_public_url_mask_does_not_create_a_false_bearer_credential(label):
    value = label + " https://example.com/public?keep=visible#fragment"
    assert capture.sanitize_training_value(value) == (value, [])


@pytest.mark.parametrize("sentence", [
    "Basic information about the company is public.",
    "The Basic requirements are listed in the report.",
    "A basic understanding of the market helps the reader.",
])
def test_public_basic_prose_is_not_a_credential(sentence):
    assert trajectory.redact_text(sentence) == sentence
    assert capture.sanitize_training_value(sentence) == (sentence, [])


@pytest.mark.parametrize("plain", [b"user:secret", b"user:se"])
def test_unpadded_basic_credential_in_free_text_is_redacted(tmp_path, plain):
    token = base64.b64encode(plain).decode("ascii").rstrip("=")
    assert token.isalpha() and len(token) % 4 in (2, 3)
    value = "A note contains Basic " + token + " in free text."
    assert token not in trajectory.redact_text(value)
    cleaned, reasons = capture.sanitize_training_value(value)
    assert token not in cleaned
    assert reasons == ["credential"]
    event = trajectory.event("runtime.stdout", {"text": value})
    assert token not in event["content"]["text"]
    document = _document()
    document["admitted_parameters"]["messages"][0]["content"] = value
    objects = LocalObjectStore(tmp_path)
    pointer = capture.RequestCapture(objects)(document)
    assert pointer["redacted"] is True
    assert "credential" in pointer["redactions"]
    assert token not in json.dumps(capture.read_request(objects, pointer))


def test_real_auth_and_signed_url_still_redact_around_basic_prose():
    value = ("Basic information is public. "
             "Authorization: Basic dXNlcjpwYXNz "
             "Bearer opaque-bearer-secret "
             "https://example.test/file?X-Amz-Signature=private-signature")
    cleaned, reasons = capture.sanitize_training_value(value)
    assert cleaned.startswith("Basic information is public. ")
    assert reasons == ["credential"]
    for secret in ("dXNlcjpwYXNz", "opaque-bearer-secret", "private-signature"):
        assert secret not in cleaned
    assert capture.sanitize_training_value(cleaned) == (cleaned, [])


def test_explicit_auth_header_redacts_nonstandard_basic_token():
    value = "Authorization: Basic opaquealphabetictoken"
    cleaned, reasons = capture.sanitize_training_value(value)
    assert reasons == ["credential"]
    assert "opaquealphabetictoken" not in cleaned
    assert "opaquealphabetictoken" not in trajectory.redact_text(value)


@pytest.mark.parametrize("key", ["Authorization", "authorization", "AUTHORIZATION"])
@pytest.mark.parametrize("marker", ["[REDACTED]", "\"[REDACTED]\"", "'[REDACTED]'"])
def test_already_redacted_auth_example_keeps_exact_source_text(key, marker):
    value = f"Public source example: {key}: {marker} continues."
    assert trajectory.redact_text(value) == value
    assert capture.sanitize_training_value(value) == (value, [])


def test_quoted_auth_key_with_literal_marker_is_unchanged():
    value = '{"Authorization": "[REDACTED]"}'
    assert trajectory.redact_text(value) == value
    assert capture.sanitize_training_value(value) == (value, [])


@pytest.mark.parametrize("value", [
    "Authorization: [REDACTED]suffix",
    "Authorization: [REDACTED]/real-token",
    "Authorization: actual-private-value",
    '"Authorization": "actual-private-value"',
])
def test_auth_example_with_remaining_value_still_redacts(value):
    cleaned, reasons = capture.sanitize_training_value(value)
    assert cleaned != value and reasons == ["credential"]
    assert "suffix" not in cleaned and "real-token" not in cleaned
    assert "actual-private-value" not in cleaned


@pytest.mark.parametrize("value", [
    "Bearer [REDACTED]suffix",
    "Basic [REDACTED]suffix",
    'Bearer "[REDACTED]suffix"',
    "Basic '[REDACTED]suffix'",
    "Bearer [REDACTED]/real-token",
    "Bearer [REDACTED].synthetic-tail",
    "Bearer [REDACTED]:synthetic-tail",
    "Bearer [REDACTED]%2Fsynthetic-tail",
    "Bearer %5BREDACTED%5Dsynthetic-tail",
    "Basic %5BREDACTED%5Dsynthetic-tail",
    'Bearer \\"[REDACTED]synthetic-tail\\"',
    '{\\"Authorization\\":\\"[REDACTED]suffix\\"}',
    '{\\\\"Authorization\\\\":\\\\"[REDACTED]suffix\\\\"}',
    '"Proxy-Authorization": "[REDACTED]suffix"',
])
def test_attached_auth_marker_does_not_pass_as_exact(value):
    cleaned, reasons = capture.sanitize_training_value(value)
    assert cleaned != value and reasons == ["credential"]
    assert "suffix" not in cleaned and "real-token" not in cleaned
    assert "synthetic-tail" not in cleaned


@pytest.mark.parametrize("value", [
    "Bearer [REDACTED]",
    "Basic [REDACTED]",
    "Bearer %5BREDACTED%5D",
    "Basic %5BREDACTED%5D",
    'Bearer \\"[REDACTED]\\"',
    '{\\"Authorization\\":\\"[REDACTED]\\"}',
    '{\\\\"Authorization\\\\":\\\\"[REDACTED]\\\\"}',
    '"Proxy-Authorization": "[REDACTED]"',
])
def test_standalone_auth_marker_keeps_exact_source(value):
    assert capture.sanitize_training_value(value) == (value, [])


@pytest.mark.parametrize("prefix", ["sk-01234567890", "sk-or-v1-abc12345", "Bearer abc", "Basic abc"])
@pytest.mark.parametrize("tail", [
    "example.com/path", "example.com/path?token=private-value#;fragment",
    "example.com/path#https://inner.example/?token=private-value",
    "private-user:private-password@example.com/path?token=private-value#fragment",
])
def test_url_mask_does_not_split_existing_credential_tokens(prefix, tail):
    from lab_arena import trajectory

    value = prefix + "https://" + tail
    cleaned, reasons = capture.sanitize_training_value(value)
    assert cleaned.startswith("[REDACTED]://example.com")
    assert prefix not in cleaned and "private-value" not in cleaned
    assert "private-user" not in cleaned and "private-password" not in cleaned
    assert reasons == ["credential"]
    if tail == "example.com/path":
        assert cleaned == trajectory.redact_text(value)
    assert capture.sanitize_training_value(cleaned)[0] == cleaned


def test_url_mask_avoids_original_text_collisions_and_exact_secret_matching(monkeypatch):
    from types import SimpleNamespace

    nonces = iter(("collision", "unique", "unique"))
    monkeypatch.setattr(capture.uuid, "uuid4", lambda: SimpleNamespace(hex=next(nonces)))
    value = "[arena-url-collision-0] https://example.com/public"
    assert capture.sanitize_training_value(value) == (value, [])
    # Exact known secrets were removed before masking. A secret that happens
    # to be part of an internal token must not delete an unrelated public URL.
    public = "https://example.com/public"
    assert capture.sanitize_training_value(public, secrets=("arena-url",)) == (public, [])


@pytest.mark.parametrize("value", [None, "", {}, []])
def test_empty_credential_fields_roundtrip_without_false_redactions(tmp_path, value):
    document = _document()
    document["optional"] = {key: value for key in (
        "authorization", "api_key", "credentials", "private_key", "session_token",
    )}
    objects = LocalObjectStore(tmp_path)
    pointer = capture.RequestCapture(objects)(document)
    restored = capture.read_request(objects, pointer)
    assert restored["optional"] == document["optional"]
    assert pointer["redacted"] is False and pointer["redactions"] == []
    assert restored["privacy"] == {"redacted": False, "redactions": []}


@pytest.mark.parametrize("value", [False, 0, (), {"value": None}, [""], "nonempty-secret"])
def test_nonempty_or_non_json_empty_credential_values_still_redact(value):
    cleaned, reasons = capture.sanitize_training_value({"api_key": value, "credentials": value})
    assert cleaned == {"api_key": "[REDACTED]", "credentials": "[REDACTED]"}
    assert reasons == ["credential"]


def test_private_json_nul_roundtrip_keeps_diagnostic_escaping_default(tmp_path):
    text = "before\x00after"
    assert trajectory.redact_text(text) == "before\\u0000after"
    assert trajectory.redact_text(text, escape_nul=False) == text
    document = {"content": text, "key\x00part": "value"}
    objects = LocalObjectStore(tmp_path)
    pointer = capture.RequestCapture(objects)(document)
    restored = capture.read_request(objects, pointer)
    assert all(restored[key] == value for key, value in document.items())
    assert pointer["redacted"] is False and pointer["redactions"] == []
    encoded = gzip.decompress(objects.get(pointer["ref"]))
    assert b"\x00" not in encoded and b"\\u0000" in encoded


def test_private_nul_preservation_does_not_weaken_real_credential_redaction():
    secret = "known-private-secret"
    auth = "a" * 64
    text = ("start\x00 " + secret + " Bearer abcdefghijklmnop "
            "api_key=another-private-key sk-abcdefghijklmnopq "
            "https://example.test/file?auth=" + auth)
    cleaned, reasons = capture.sanitize_training_value({"content": text}, secrets=(secret,))
    assert "\x00" in cleaned["content"]
    assert reasons == ["credential"]
    for private in (secret, "abcdefghijklmnop", "another-private-key", "sk-abcdefghijklmnopq", auth):
        assert private not in cleaned["content"]


def test_provider_returned_reasoning_text_and_summary_are_kept():
    cleaned, reasons = capture.sanitize_training_value({"input": [{
        "type": "reasoning", "content": [{"type": "reasoning_text", "text": "visible text"}],
        "summary": [{"type": "summary_text", "text": "visible summary"}],
        "encrypted_content": "encrypted-secret",
    }]})
    assert cleaned["input"][0]["content"][0]["text"] == "visible text"
    assert cleaned["input"][0]["summary"][0]["text"] == "visible summary"
    assert cleaned["input"][0]["encrypted_content"] == "[REDACTED]"
    assert reasons == ["hidden_reasoning"]


def test_chat_reasoning_details_keep_text_and_redact_opaque_items():
    cleaned, reasons = capture.sanitize_training_value({"message": {
        "reasoning": "visible", "reasoning_details": [
            {"type": "reasoning.summary", "summary": "visible", "index": 0},
            {"type": "reasoning.text", "text": "also visible", "signature": "sig-secret"},
            {"type": "reasoning.text", "signature": "gemini-secret", "format": "google"},
            {"type": "reasoning.encrypted", "data": "sealed-secret"},
        ]}})
    assert "secret" not in json.dumps(cleaned)
    assert cleaned["message"]["reasoning"] == "visible"
    assert cleaned["message"]["reasoning_details"] == [
        {"type": "reasoning.summary", "summary": "visible", "index": 0},
        {"type": "reasoning.text", "text": "also visible", "signature": "[REDACTED]"},
        "[REDACTED]", "[REDACTED]"]
    assert reasons == ["hidden_reasoning"]
    # Stored placeholders remain valid when the same artifact is read later.
    assert capture.sanitize_training_value(cleaned) == (cleaned, ["hidden_reasoning"])


def test_plaintext_reasoning_content_alias_is_response_only():
    response = {"choices": [{"message": {"role": "assistant",
                                        "reasoning_content": "Visible DeepSeek text."}},
                            {"message": {"role": "assistant",
                                         "reasoning_content": {"sealed": "opaque state"}}}]}
    cleaned, reasons = capture.sanitize_training_value(response)
    assert cleaned["choices"][0]["message"]["reasoning_content"] == "Visible DeepSeek text."
    assert cleaned["choices"][1]["message"]["reasoning_content"] == "[REDACTED]"
    assert reasons == ["hidden_reasoning"]


def test_chat_assistant_replay_alias_is_saved_without_full_thinking_approval(tmp_path):
    document = {"operation_id": "openrouter.chat", "outbound": {"body": {"messages": [
        {"role": "assistant", "content": None, "reasoning_content": "Visible prior summary."},
        {"role": "user", "content": "Prompt", "reasoning_content": "Untrusted user state"},
    ]}}}
    cleaned, redactions = capture.sanitize_training_value(document)
    assert cleaned["outbound"]["body"]["messages"][0]["reasoning_content"] == "Visible prior summary."
    assert cleaned["outbound"]["body"]["messages"][1]["reasoning_content"] == "[REDACTED]"
    assert redactions == ["hidden_reasoning"]
    objects = LocalObjectStore(tmp_path)
    pointer = capture.RequestCapture(objects)(document)
    assert pointer["status"] == "stored"
    assert capture.read_request(objects, pointer)["outbound"]["body"]["messages"][0][
        "reasoning_content"] == "Visible prior summary."
    # The same projection survives the export's second privacy pass.
    from lab_arena.training_export import without_hidden_reasoning
    projected, removed = without_hidden_reasoning({"provider_calls": [{
        "operation_id": "openrouter.chat", "request": document}]})
    messages = projected["provider_calls"][0]["request"]["outbound"]["body"]["messages"]
    assert messages[0]["reasoning_content"] == "Visible prior summary."
    assert "reasoning_content" not in messages[1]
    assert removed is True
    request = {"admitted_parameters": {"messages": [{"role": "assistant",
                                                       "reasoning_content": "prior text"}],
                                       "choices": [{"message": {"reasoning_content": "spoofed"}}]}}
    cleaned, reasons = capture.sanitize_training_value(request)
    assert cleaned["admitted_parameters"]["messages"][0]["reasoning_content"] == "[REDACTED]"
    assert cleaned["admitted_parameters"]["choices"][0]["message"]["reasoning_content"] == "[REDACTED]"
    assert reasons == ["hidden_reasoning"]


def test_reasoning_items_do_not_expose_unrecognized_or_nontext_payloads():
    cleaned, reasons = capture.sanitize_training_value({"output": [{
        "type": "reasoning", "summary": [{"type": "summary_text", "text": "visible",
                                           "vendor_state": "sealed nested state"}],
        "content": [{"type": "encrypted_part", "data": "sealed content"}],
        "vendor_state": "sealed vendor state",
    }]})
    item = cleaned["output"][0]
    assert item["summary"][0]["text"] == "visible"
    assert item["summary"][0]["vendor_state"] == "[REDACTED]"
    assert item["content"] == item["vendor_state"] == "[REDACTED]"
    assert reasons == ["hidden_reasoning"]


def test_full_export_preserves_only_explicit_request_reasoning_config_paths():
    config = {"effort": "high", "summary": "concise"}
    request = {name: {"reasoning": config} for name in (
        "admitted_parameters", "normalized_parameters", "effective_parameters", "hashed_parameters",
    )}
    request["outbound"] = {"body": {"reasoning": config}}
    cleaned, reasons = capture.sanitize_training_value({"provider_calls": [{"request": request}]})
    assert cleaned["provider_calls"][0]["request"] == request
    assert reasons == []
    # A response can contain any field names. None establish a request root.
    response = {"provider_calls": [{"request": request}], "hashed_parameters": {"reasoning": config},
                "outbound": {"body": {"reasoning": config}}, "reasoning": config}
    cleaned, reasons = capture.sanitize_training_value({"provider_calls": [{"response": response}]})
    rendered = json.dumps(cleaned)
    assert '"effort": "high"' not in rendered
    assert reasons == ["hidden_reasoning"]


@pytest.mark.parametrize("source_calls", [False, True])
def test_canonical_call_roots_keep_request_settings_history_and_response_aliases(source_calls):
    from lab_arena.training_export import without_hidden_reasoning

    parameters = {"reasoning": {"effort": "low", "summary": "concise"}, "messages": [
        {"role": "assistant", "reasoning_content": "Prior returned text."}]}
    request = {"operation_id": "openrouter.chat",
               **{key: copy.deepcopy(parameters) for key in (
                   "admitted_parameters", "normalized_parameters", "effective_parameters", "hashed_parameters")},
               "outbound": {"body": copy.deepcopy(parameters)}}
    response = {"choices": [{"message": {"role": "assistant", "reasoning_content": "Returned text."}}]}
    call = {"operation_id": "openrouter.chat", "request": request, "response": response,
            "response_receipts": [{"response": copy.deepcopy(response)}],
            "upstream_response_receipts": [{"response": copy.deepcopy(response)}]}
    value = ({"judgment": {"signal_source_provider_calls": [call]}} if source_calls
             else {"provider_calls": [call]})
    assert capture.sanitize_training_value(value) == (value, [])
    assert without_hidden_reasoning(value) == (value, False)

    # Keep the privacy boundary at the exact collection roots. Provider content
    # cannot create a second trusted call or request by copying these names.
    spoof = copy.deepcopy(value)
    spoof_call = (spoof["judgment"]["signal_source_provider_calls"][0] if source_calls
                  else spoof["provider_calls"][0])
    spoof_call["response"]["nested"] = copy.deepcopy(value)
    spoof_call["request"]["outbound"]["body"]["messages"].append(
        {"role": "user", "reasoning_content": "User state."})
    history_cleaned, _ = capture.sanitize_training_value(spoof)
    history_call = (history_cleaned["judgment"]["signal_source_provider_calls"][0] if source_calls
                    else history_cleaned["provider_calls"][0])
    messages = history_call["request"]["outbound"]["body"]["messages"]
    assert messages[0]["reasoning_content"] == "Prior returned text."
    assert messages[1]["reasoning_content"] == "[REDACTED]"
    spoof_call["request"]["operation_id"] = "openrouter.responses"
    spoof_call["response"]["choices"][0]["message"].update(
        encrypted_content="Opaque state.", authorization="private-token-value",
        reasoning_content="Returned text with private-token-value")
    cleaned, reasons = capture.sanitize_training_value(spoof, secrets=("private-token-value",))
    assert reasons == ["credential", "hidden_reasoning"]
    safe_call = (cleaned["judgment"]["signal_source_provider_calls"][0] if source_calls
                 else cleaned["provider_calls"][0])
    assert safe_call["request"]["outbound"]["body"]["messages"][0]["reasoning_content"] == "[REDACTED]"
    assert safe_call["request"]["outbound"]["body"]["messages"][1]["reasoning_content"] == "[REDACTED]"
    nested = safe_call["response"]["nested"]
    nested_call = (nested["judgment"]["signal_source_provider_calls"][0] if source_calls
                   else nested["provider_calls"][0])
    assert nested_call["request"]["outbound"]["body"]["reasoning"] == "[REDACTED]"
    assert nested_call["response"]["choices"][0]["message"]["reasoning_content"] == "[REDACTED]"
    assert "private-token-value" not in json.dumps(cleaned)
    assert "Opaque state." not in json.dumps(cleaned)
    assert capture.sanitize_training_value(cleaned)[0] == cleaned
    projected, removed = without_hidden_reasoning(cleaned)
    assert removed is True
    assert without_hidden_reasoning(projected) == (projected, False)
    projected_call = (projected["judgment"]["signal_source_provider_calls"][0] if source_calls
                      else projected["provider_calls"][0])
    assert projected_call["response"]["choices"][0]["message"]["reasoning_content"] == "Returned text with [REDACTED]"
    assert "reasoning_content" not in projected_call["request"]["outbound"]["body"]["messages"][0]


@pytest.mark.parametrize("response", [
    {"choices": [{"message": {"reasoning_content": "Actual returned text."}}]},
    {"object": "response", "output": []},
    {"object": "chat.completion"}, {"object": "chat.completion", "choices": None},
    {"object": "chat.completion.chunk"}, {"object": "chat.completion.chunk", "choices": None},
])
def test_recognizable_raw_responses_cannot_create_canonical_call_roots(response):
    from lab_arena.training_export import without_hidden_reasoning

    request = {"operation_id": "openrouter.chat", "outbound": {"body": {
        "reasoning": {"effort": "low"}, "messages": [
            {"role": "assistant", "reasoning_content": "Spoofed history."}]}}}
    call = {"operation_id": "openrouter.chat", "request": request,
            "response": {"choices": [{"message": {"reasoning_content": "Spoofed response."}}]}}
    raw = {**response, "provider_calls": [call], "judgment": {"signal_source_provider_calls": [call]}}
    cleaned, reasons = capture.sanitize_training_value(raw)
    assert reasons == ["hidden_reasoning"]
    for safe_call in (cleaned["provider_calls"][0], cleaned["judgment"]["signal_source_provider_calls"][0]):
        assert safe_call["request"]["outbound"]["body"]["reasoning"] == "[REDACTED]"
    assert "Spoofed" not in json.dumps(cleaned)
    projected, removed = without_hidden_reasoning(raw)
    assert removed and "Spoofed" not in json.dumps(projected)
    if "choices" in response:
        assert cleaned["choices"] == projected["choices"] == response["choices"]


@pytest.mark.parametrize("projection", [
    "admitted_parameters", "normalized_parameters", "effective_parameters", "hashed_parameters", "outbound",
])
@pytest.mark.parametrize("response", [
    None, {"choices": []}, {"object": "response", "output": []},
    {"object": "chat.completion"}, {"object": "chat.completion", "choices": None},
    {"object": "chat.completion.chunk"}, {"object": "chat.completion.chunk", "choices": None},
])
def test_standalone_chat_history_requires_a_request_root(projection, response):
    from lab_arena.training_export import without_hidden_reasoning

    parameters = {"messages": [{"role": "assistant", "reasoning_content": "Prior assistant text."}]}
    request = {"operation_id": "openrouter.chat",
               projection: {"body": parameters} if projection == "outbound" else parameters}
    value = {**request, **(response or {})}
    cleaned, reasons = capture.sanitize_training_value(value)
    projected, removed = without_hidden_reasoning(value)
    if response is None:
        assert cleaned == projected == value
        assert reasons == [] and removed is False
    else:
        body = cleaned[projection]["body"] if projection == "outbound" else cleaned[projection]
        assert body["messages"][0]["reasoning_content"] == "[REDACTED]"
        body = projected[projection]["body"] if projection == "outbound" else projected[projection]
        assert "reasoning_content" not in body["messages"][0]
        assert reasons == ["hidden_reasoning"] and removed is True


@pytest.mark.parametrize("mutation", [
    {"sha256": "sha256:" + "0" * 64}, {"size_bytes": 1},
    {"compressed_bytes": 1}, {"redacted": True}, {"encoding": "identity"},
    {"ref": "public/request.json.gz"}, {"status": "unavailable"},
])
def test_reader_fails_closed_on_altered_pointer(tmp_path, mutation):
    objects = LocalObjectStore(tmp_path)
    pointer = capture.RequestCapture(objects)(_document())
    with pytest.raises(capture.CaptureError):
        capture.read_request(objects, {**pointer, **mutation})


def test_reader_rejects_corrupt_content_and_oversized_gzip(tmp_path, monkeypatch):
    objects = LocalObjectStore(tmp_path)
    pointer = capture.RequestCapture(objects)(_document())
    raw = gzip.compress(b"x" * 4096, mtime=0)
    monkeypatch.setattr(capture, "MAX_OBJECT_BYTES", 1024)
    pointer.update(size_bytes=1024, compressed_bytes=len(raw))
    monkeypatch.setattr(objects, "get_bounded", lambda *_args: raw)
    with pytest.raises(capture.CaptureError, match="size_mismatch"):
        capture.read_request(objects, pointer)


def test_capture_size_or_store_failure_reports_unavailable_without_truncation(tmp_path, monkeypatch):
    objects = LocalObjectStore(tmp_path)
    monkeypatch.setattr(capture, "MAX_OBJECT_BYTES", 100)
    assert capture.RequestCapture(objects)(_document())["reason"] == "request_size_limit"
    monkeypatch.setattr(capture, "MAX_OBJECT_BYTES", 1024 * 1024)
    monkeypatch.setattr(objects, "put", lambda *_args: (_ for _ in ()).throw(RuntimeError("private")))
    assert capture.RequestCapture(objects)(_document()) == capture.unavailable()


def _execute(provider, *, action=1, context=CONTEXT):
    return provider.execute(context, operation_id="openrouter.chat", parameters={
        "model": "openai/gpt-4o-mini", "messages": [{"role": "user", "content": "hello"}],
        "reasoning": {"effort": "high"}, "temperature": 0.1,
    }, action_sequence=action, timeout_ms=5000)


def test_broker_receipt_joins_exact_outbound_and_keeps_pointer_private(tmp_path):
    objects = LocalObjectStore(tmp_path)
    provider, store, transport = make_broker(capture_request=capture.RequestCapture(objects).prepare)
    result = _execute(provider)
    call = next(iter(store.calls.values()))
    pointer = call["call_doc"]["request_capture"]
    restored = capture.read_request(objects, pointer)
    assert restored["outbound"]["body"] == json.loads(transport.sent[0]["body"])
    assert restored["request_hash"] == call["call_doc"]["request_hash"]
    assert contracts.document_hash(restored["hashed_parameters"]) == restored["request_hash"]
    assert restored["outbound"]["body_sha256"] == contracts.hash_bytes(transport.sent[0]["body"])
    assert restored["outbound"]["body"]["stream"] is False
    assert restored["outbound"]["body"]["provider"] == dict(operations.OPENROUTER_STRICT_PROVIDER_POLICY)
    assert "max_tokens" not in restored["admitted_parameters"]
    assert restored["effective_parameters"]["max_tokens"] == 4096
    assert "request_capture" not in json.dumps(result.to_document())
    assert "headers" not in restored["outbound"]
    assert not any(secret in json.dumps(restored) for secret in HOST_KEYS.values())


def test_replayed_action_and_distinct_attempt_use_same_request_object(tmp_path):
    objects = LocalObjectStore(tmp_path)
    provider, store, transport = make_broker(
        capture_request=capture.RequestCapture(objects).prepare,
        transport=FakeTransport([(200, {"usage": {"cost": "0.000001"}})] * 2),
    )
    first = _execute(provider)
    second = _execute(provider)
    third = _execute(provider, action=2, context=replace(CONTEXT, run_id="other-run"))
    assert second.call["idempotent"] is True
    assert first.call["call_identity"] != third.call["call_identity"]
    assert len(transport.sent) == 2
    pointers = [value["call_doc"]["request_capture"] for value in store.calls.values()]
    assert pointers[0] == pointers[1]
    assert len(list(tmp_path.rglob("*.gz"))) == 1


def test_callback_exception_changes_neither_worker_result_nor_provider_receipts():
    def broken(*_args, **_kwargs):
        raise RuntimeError("must not escape")
    control, control_store, control_transport = make_broker()
    observed, store, transport = make_broker(capture_request=broken)
    assert _execute(control).to_document() == _execute(observed).to_document()
    assert control_store.log == store.log
    assert control_transport.sent[0]["body"] == transport.sent[0]["body"]
    assert next(iter(store.calls.values()))["call_doc"]["request_capture"] == capture.unavailable()


def test_champion_retry_receipts_share_blob_but_retain_attempt_identity(tmp_path):
    objects = LocalObjectStore(tmp_path)
    provider, store, transport = make_broker(
        transport=FakeTransport([(401, {"error": "bad account"}), (200, {"usage": {"cost": "0.000001"}})]),
        capture_request=capture.RequestCapture(objects).prepare,
        credential_for=lambda _context, name: HOST_KEYS[name],
        provider_funding_source_for=lambda _context, _provider: "miner_key",
        retry_miner_credential_for=lambda _context: True,
    )
    result = _execute(provider)
    receipts = list(store.calls.values())
    assert result.status == 200 and len(receipts) == 2 and len(transport.sent) == 2
    assert [row["call_doc"]["provider_attempt"] for row in receipts] == [1, 2]
    assert receipts[0]["identity"] != receipts[1]["identity"]
    assert receipts[0]["call_doc"]["request_capture"] == receipts[1]["call_doc"]["request_capture"]


@pytest.mark.parametrize("denial", ["stale", "budget", "quota"])
def test_no_object_write_without_authorized_admission(tmp_path, denial):
    objects = LocalObjectStore(tmp_path)
    store = FakeLedgerStore(
        per_icp_quota=0 if denial == "quota" else 30,
        openrouter_capacity=0 if denial == "budget" else 10_000_000,
    )
    store.stale = denial == "stale"
    provider, _, transport = make_broker(store=store, capture_request=capture.RequestCapture(objects).prepare)
    _execute(provider)
    assert not transport.sent
    assert not list(tmp_path.rglob("*.gz"))



@pytest.mark.parametrize("cached_judgment", [False, True])
def test_exhausted_ambiguous_reservation_preserves_prepared_request(
    tmp_path, monkeypatch, cached_judgment,
):
    objects = LocalObjectStore(tmp_path)
    writes = []
    original_put = objects.put
    def put(ref, body):
        writes.append((ref, body))
        original_put(ref, body)
    monkeypatch.setattr(objects, "put", put)

    class CommitThenLoseBothResponses(FakeLedgerStore):
        def __init__(self):
            super().__init__()
            self.arguments = []
            self.failures = [ArenaStoreUnavailable("first lost response"),
                             ArenaStoreUnavailable("second lost response")]
            self.judgment_calls = 0

        def reserve_call(self, **kwargs):
            assert writes == []  # Capture must not move before admission.
            self.arguments.append(copy.deepcopy(kwargs))
            super().reserve_call(**kwargs)
            raise self.failures[len(self.arguments) - 1]

        def reserve_judgment_call(self, **kwargs):
            self.judgment_calls += 1
            return self.reserve_call(**kwargs)

    store = CommitThenLoseBothResponses()
    if cached_judgment:
        monkeypatch.setattr(br, "_judgment_cache_material", lambda *_a, **_kw: (
            "sha256:" + "c" * 64, "{}"))
    provider, _, transport = make_broker(
        store=store, capture_request=capture.RequestCapture(objects).prepare,
        judge_models=("openai/gpt-4o-mini",),
    )
    with pytest.raises(ArenaStoreUnavailable) as raised:
        _execute(provider, context=replace(CONTEXT, kind="score" if cached_judgment else "execute"))
    assert raised.value is store.failures[1]
    assert store.arguments[0] == store.arguments[1]
    assert store.judgment_calls == (2 if cached_judgment else 0)
    assert store.log == ["reserve", "reserve"] and transport.sent == []
    assert len(store.calls) == 1 and len(writes) == 1
    receipt = next(iter(store.calls.values()))
    assert receipt["kind"] == "reservation"
    pointer = receipt["call_doc"]["request_capture"]
    assert writes[0][0] == pointer["ref"]
    restored = capture.read_request(objects, pointer)
    assert restored["request_hash"] == receipt["call_doc"]["request_hash"]
    assert contracts.document_hash(restored["hashed_parameters"]) == restored["request_hash"]
    assert restored["outbound"]["body"]["messages"] == [{"role": "user", "content": "hello"}]
    assert not any(secret in json.dumps(restored) for secret in HOST_KEYS.values())
    assert "headers" not in restored["outbound"]


@pytest.mark.parametrize("prior_ambiguous", [False, True])
def test_rolled_back_reserve_only_persists_after_prior_ambiguous_response(
    tmp_path, monkeypatch, prior_ambiguous,
):
    objects = LocalObjectStore(tmp_path)
    writes = []
    original_put = objects.put
    monkeypatch.setattr(objects, "put", lambda ref, body: (writes.append(ref), original_put(ref, body))[1])
    terminal = ArenaStoreUnavailable("rolled back", transaction_rolled_back=True)

    class RollbackStore(FakeLedgerStore):
        def reserve_call(self, **kwargs):
            self.log.append("attempt")
            if prior_ambiguous and self.log == ["attempt"]:
                super().reserve_call(**kwargs)
                raise ArenaStoreUnavailable("ambiguous response")
            raise terminal

    provider, store, transport = make_broker(
        store=RollbackStore(), capture_request=capture.RequestCapture(objects).prepare,
    )
    with pytest.raises(ArenaStoreUnavailable) as raised:
        _execute(provider)
    assert raised.value is terminal and transport.sent == []
    assert len(writes) == int(prior_ambiguous)
    assert len(store.calls) == int(prior_ambiguous)
    assert store.log.count("attempt") == (2 if prior_ambiguous else 1)


@pytest.mark.parametrize("persistence_failure", ["false", "raises"])
def test_exhausted_reservation_capture_failure_keeps_original_exception(
    tmp_path, monkeypatch, persistence_failure,
):
    terminal = ArenaStoreUnavailable("admission failure")
    attempts = []
    persist_attempts = []
    class UnavailableStore(FakeLedgerStore):
        def reserve_call(self, **kwargs):
            attempts.append(kwargs["call_identity"])
            super().reserve_call(**kwargs)
            raise terminal
    def persist(prepared):
        persist_attempts.append(prepared.pointer)
        if persistence_failure == "raises":
            raise RuntimeError("object failure")
        return False
    monkeypatch.setattr(capture.PreparedRequest, "persist", persist)
    provider, store, transport = make_broker(
        store=UnavailableStore(), capture_request=capture.RequestCapture(LocalObjectStore(tmp_path)).prepare,
    )
    with pytest.raises(ArenaStoreUnavailable) as raised:
        _execute(provider)
    assert raised.value is terminal
    assert len(persist_attempts) == 1 and len(attempts) == 2 and attempts[0] == attempts[1]
    assert store.log == ["reserve", "reserve"] and transport.sent == []
    assert not list(tmp_path.rglob("*.gz"))


@pytest.mark.parametrize("first_response_lost", [False, True])
def test_successful_admission_persists_once_after_same_identity_readback(
    tmp_path, monkeypatch, first_response_lost,
):
    objects, writes, arguments = LocalObjectStore(tmp_path), [], []
    original_put = objects.put
    def put(ref, body):
        writes.append(ref)
        original_put(ref, body)
    monkeypatch.setattr(objects, "put", put)
    class SuccessStore(FakeLedgerStore):
        def reserve_call(self, **kwargs):
            assert writes == []
            arguments.append(copy.deepcopy(kwargs))
            result = super().reserve_call(**kwargs)
            if first_response_lost and len(arguments) == 1:
                raise ArenaStoreUnavailable("lost first response")
            return result
    provider, store, transport = make_broker(
        store=SuccessStore(), capture_request=capture.RequestCapture(objects).prepare,
        transport=FakeTransport([(200, {"usage": {"cost": "0.000001"}})]),
    )
    result = _execute(provider)
    assert result.status == 200 and len(writes) == 1 and len(transport.sent) == 1
    assert len(arguments) == (2 if first_response_lost else 1)
    assert all(argument == arguments[0] for argument in arguments)
    assert len(store.calls) == 1 and store.log.count("dispatch") == store.log.count("settle") == 1
    pointer = next(iter(store.calls.values()))["call_doc"]["request_capture"]
    assert capture.read_request(objects, pointer)["outbound"]["body"] == json.loads(transport.sent[0]["body"])


def test_idempotent_request_pointer_mismatch_does_not_write_or_rewrite_receipt(tmp_path, monkeypatch):
    objects, writes = LocalObjectStore(tmp_path), []
    original_put = objects.put
    monkeypatch.setattr(objects, "put", lambda ref, body: (writes.append(ref), original_put(ref, body))[1])
    provider, store, transport = make_broker(
        capture_request=capture.RequestCapture(objects).prepare,
        transport=FakeTransport([(200, {"usage": {"cost": "0.000001"}})] * 2),
    )
    first = _execute(provider)
    receipt = store.calls[first.call["call_identity"]]
    receipt["kind"] = "reservation"
    receipt["call_doc"]["request_capture"]["ref"] = "arena/existing-different-request.json.gz"
    saved = copy.deepcopy(receipt["call_doc"])
    replay = _execute(provider)
    assert replay.status == 200
    assert len(writes) == 1 and len(transport.sent) == 2
    assert receipt["call_doc"] == saved


@pytest.mark.parametrize("fallback", [False, True])
def test_adapted_source_request_survives_exhausted_reserve_responses(tmp_path, fallback):
    terminal = ArenaStoreUnavailable("second lost response")
    arguments = []
    class LostSourceReservation(FakeLedgerStore):
        def reserve_call(self, **kwargs):
            arguments.append(copy.deepcopy(kwargs))
            super().reserve_call(**kwargs)
            raise terminal
    objects = LocalObjectStore(tmp_path)
    provider, store, transport = make_broker(
        store=LostSourceReservation(), capture_request=capture.RequestCapture(objects).prepare,
        credential_for=lambda _context, name: HOST_KEYS[name],
        funding_source_for=lambda _context: "miner_key",
    )
    with pytest.raises(ArenaStoreUnavailable) as raised:
        provider._execute_once(
            replace(CONTEXT, kind="score", round_id="arena-2026-10-13"),
            operation_id="scrapingdog.scrape", parameters={"url": "https://example.com/about"},
            action_sequence=1, timeout_ms=30_000, provider_attempt=2 if fallback else 1,
            champion_credential_retry=False, source_content_fallback=fallback,
        )
    assert raised.value is terminal
    assert len(arguments) == 2 and arguments[0] == arguments[1]
    assert len(store.calls) == 1 and transport.sent == []
    receipt = next(iter(store.calls.values()))
    restored = capture.read_request(objects, receipt["call_doc"]["request_capture"])
    assert restored["effective_operation_id"] == "deepline.execute"
    assert restored["effective_parameters"]["tool"] == ("generic_http_request" if fallback else "firecrawl_scrape")
    assert restored["outbound"]["body"]["payload"]["url"] == "https://example.com/about"
    assert not any(secret in json.dumps(restored) for secret in HOST_KEYS.values())

def test_prepare_has_no_io_and_missing_referenced_object_fails_closed(tmp_path):
    objects = LocalObjectStore(tmp_path)
    prepared = capture.RequestCapture(objects).prepare(_document())
    assert isinstance(prepared, capture.PreparedRequest)
    assert prepared.pointer["status"] == "referenced"
    assert not list(tmp_path.rglob("*.gz"))
    with pytest.raises(capture.CaptureError, match="unavailable"):
        capture.read_request(objects, prepared.pointer)
    assert prepared.persist() is True
    assert capture.read_request(objects, prepared.pointer)["schema_version"] == capture.SCHEMA_VERSION


def test_storage_failure_after_admission_preserves_provider_result(tmp_path, monkeypatch):
    objects = LocalObjectStore(tmp_path)
    monkeypatch.setattr(objects, "put", lambda *_args: (_ for _ in ()).throw(RuntimeError("private")))
    provider, store, transport = make_broker(
        capture_request=capture.RequestCapture(objects).prepare,
        transport=FakeTransport([(200, {"usage": {"cost": "0.000001"}})]),
    )
    assert _execute(provider).status == 200
    assert len(transport.sent) == 1
    pointer = next(iter(store.calls.values()))["call_doc"]["request_capture"]
    assert pointer["status"] == "referenced"
    with pytest.raises(capture.CaptureError):
        capture.read_request(objects, pointer)


def test_existing_inflight_receipt_without_capture_replays_after_rollout(tmp_path):
    objects = LocalObjectStore(tmp_path)
    provider, store, transport = make_broker(
        transport=FakeTransport([(200, {"usage": {"cost": "0.000001"}})] * 2),
    )
    first = _execute(provider)
    # This fixture represents an existing durable reservation from the prior
    # release, before a worker has committed its dispatch marker.
    receipt = store.calls[first.call["call_identity"]]
    receipt["kind"] = "reservation"
    provider._capture_request = capture.RequestCapture(objects).prepare
    second = _execute(provider)
    assert second.status == 200
    assert "request_capture" not in receipt["call_doc"]
    assert not list(tmp_path.rglob("*.gz"))


def test_cache_hit_retains_own_request_reference_and_avoids_paid_dispatch(tmp_path, monkeypatch):
    key = "sha256:" + "c" * 64
    source_call = "sha256:" + "d" * 64

    class CacheStore(FakeLedgerStore):
        def reserve_judgment_call(self, **kwargs):
            reserved = self.reserve_call(**kwargs)
            terminal = {
                "status": 200, "headers": {}, "body_b64": base64.b64encode(b'{"answer":true}').decode(),
                "call_succeeded": True, "judgment_cache_eligible": True,
                "judgment_cache_key": key, "judgment_cache_source_call_identity": source_call,
                "judgment_cache_source_run_id": "source-run",
            }
            return {**reserved, "status": "cache_hit", "amount_microusd": 0,
                    "judgment_cache_key": key, "terminal_response": terminal,
                    "source_call_identity": source_call, "source_run_id": "source-run"}

    monkeypatch.setattr(br, "_judgment_cache_material", lambda *_a, **_kw: (key, "{}"))
    monkeypatch.setattr(br, "_complete_judgment_response", lambda *_a, **_kw: True)
    objects = LocalObjectStore(tmp_path)
    provider, store, transport = make_broker(
        store=CacheStore(), capture_request=capture.RequestCapture(objects).prepare,
        judge_models=("openai/gpt-4o-mini",),
    )
    result = _execute(provider, context=replace(CONTEXT, kind="score"))
    assert result.call["cached"] is True
    assert result.call["source_call_identity"] == source_call
    pointer = next(iter(store.calls.values()))["call_doc"]["request_capture"]
    assert capture.read_request(objects, pointer)["operation_id"] == "openrouter.chat"
    assert not transport.sent


def test_effective_deepline_wrapper_is_exact_provider_body(tmp_path):
    objects = LocalObjectStore(tmp_path)
    provider, store, transport = make_broker(capture_request=capture.RequestCapture(objects).prepare)
    result = provider.execute(CONTEXT, operation_id="deepline.execute", parameters={
        "tool": "exa_search", "payload": {"query": "company market"},
    }, action_sequence=1, timeout_ms=5000)
    pointer = next(iter(store.calls.values()))["call_doc"]["request_capture"]
    restored = capture.read_request(objects, pointer)
    assert restored["outbound"]["body"] == json.loads(transport.sent[0]["body"])
    assert restored["outbound"]["body"] == {
        "provider": "exa", "operation": "exa_search", "payload": {"query": "company market"},
    }
    assert "request_capture" not in result.call


def _rejected_model_request():
    return {"admitted_parameters": {"raw_frame": {
        "operation_id": "openrouter.responses", "parameters": {
            "model": "deepseek/deepseek-v4.1-flash-1017", "input": "synthetic",
            "reasoning": {"effort": "high"}}}}, "timeout_ms": 0}


@pytest.mark.parametrize("controls", [
    {"effort": "high"}, {"summary": "concise", "context": "all_turns"},
    {"enabled": True, "exclude": False, "max_tokens": 32768},
    {"enabled": False, "exclude": True}, {},
])
def test_rejected_frame_retains_only_validated_control_values(controls):
    value = _rejected_model_request()
    value["admitted_parameters"]["raw_frame"]["parameters"]["reasoning"] = controls
    assert capture.sanitize_training_value(value) == (value, [])


def test_rejected_frame_control_capture_does_not_require_request_admission():
    value = _rejected_model_request()
    value["admitted_parameters"]["raw_frame"].update(schema_version=[], timeout_ms=False)
    assert capture.sanitize_training_value(value) == (value, [])


@pytest.mark.parametrize("controls", [
    {"effort": "private prose"}, {"effort": 1}, {"effort": None}, {"summary": "private prose"},
    {"context": "private prose"}, {"enabled": 1}, {"exclude": "false"},
    {"max_tokens": True}, {"max_tokens": 0}, {"max_tokens": "123"},
    {"effort": {"text": "opaque state"}}, {"encrypted_content": "opaque state"},
    {"effort": "high", "hidden_reasoning": "opaque state"},
    {"effort": "high", "authorization": "secret-value"},
])
def test_rejected_frame_malformed_and_opaque_controls_fail_closed(controls):
    value = _rejected_model_request()
    value["admitted_parameters"]["raw_frame"]["parameters"]["reasoning"] = controls
    cleaned, redactions = capture.sanitize_training_value(value)
    assert redactions == ["hidden_reasoning"]
    assert cleaned["admitted_parameters"]["raw_frame"]["parameters"]["reasoning"] == "[REDACTED]"


@pytest.mark.parametrize("mutation", [
    "non_model", "unbound", "missing_model", "non_object_frame", "extra_frame_field",
    "wrong_timeout", "extra_request_field", "extra_admitted_field",
    "arbitrary_wrapper", "provider_response", "wrong_observation_kind", "wrong_protocol",
    "wrong_observation_operation", "paid_identity", "wrong_saved_wrapper",
    "missing_observation_metadata", "list_schema", "dict_schema",
    "mapping_observations", "saved_mapping_observations", "list_artifacts",
    "response_artifacts", "nested_artifacts",
])
def test_rejected_frame_cannot_create_trust_from_unrelated_wrappers(mutation):
    value = _rejected_model_request()
    frame = value["admitted_parameters"]["raw_frame"]
    if mutation == "non_model":
        frame["operation_id"] = "deepline.execute"
    elif mutation == "unbound":
        frame["operation_id"] = {"name": "openrouter.responses"}
    elif mutation == "missing_model":
        del frame["parameters"]["model"]
    elif mutation == "non_object_frame":
        value["admitted_parameters"]["raw_frame"] = [frame]
    elif mutation == "extra_frame_field":
        frame["arbitrary"] = True
    elif mutation == "wrong_timeout":
        value["timeout_ms"] = 1
    elif mutation == "extra_request_field":
        value["other"] = True
    elif mutation == "extra_admitted_field":
        value["admitted_parameters"]["other"] = True
    elif mutation == "arbitrary_wrapper":
        value = {"nested": value}
    elif mutation == "provider_response":
        value = {**value, "object": "response", "output": []}
    else:
        row = {"ordinal": 0, "privacy": {"redacted": False, "redactions": []},
               "kind": "local_parse_error", "operation_id": "unknown", "action_sequence": None,
               "call_identity": None, "contributes_to_provider_call_count": False,
               "request": value, "outcome": {"protocol": "frame"}}
        if mutation == "wrong_observation_kind":
            row["kind"] = "transport_failure"
        elif mutation == "wrong_protocol":
            row["outcome"]["protocol"] = "http"
        elif mutation == "wrong_observation_operation":
            row["operation_id"] = "openrouter.responses"
        elif mutation == "paid_identity":
            row["call_identity"] = "sha256:" + "1" * 64
        elif mutation == "missing_observation_metadata":
            del row["ordinal"]
        if mutation == "mapping_observations":
            value = {"observations": {"0": row}}
        elif mutation == "saved_mapping_observations":
            value = {"schema_version": "leadpoet.lab_arena.training_observations.v2",
                     "payload": {"observations": {"0": row}}}
        elif mutation == "list_artifacts":
            value = {"worker_observation_artifacts": [{"payload": {"observations": [row]}}]}
        elif mutation in {"response_artifacts", "nested_artifacts"}:
            value = {"worker_observation_artifacts": {"run": {"payload": {"observations": [row]}}}}
            if mutation == "response_artifacts":
                value.update(object="response", output=[])
            else:
                value = {"nested": value}
        elif mutation in {"wrong_saved_wrapper", "list_schema", "dict_schema"}:
            value = {"payload": {"observations": [row]}}
            if mutation != "wrong_saved_wrapper":
                value["schema_version"] = [] if mutation == "list_schema" else {}
        else:
            value = row
    cleaned, redactions = capture.sanitize_training_value(value)
    assert redactions == ["hidden_reasoning"]
    assert '"effort": "high"' not in json.dumps(cleaned)


def test_rejected_frame_keeps_credential_and_opaque_guards_and_redaction_fixed_point():
    value = _rejected_model_request()
    parameters = value["admitted_parameters"]["raw_frame"]["parameters"]
    parameters.update(input="synthetic known-private-value", authorization="another-private-value",
                      hidden_reasoning="opaque-private-value", encrypted_content="sealed-private-value")
    cleaned, redactions = capture.sanitize_training_value(value, secrets=("known-private-value", "high"))
    assert redactions == ["credential", "hidden_reasoning"]
    assert cleaned["admitted_parameters"]["raw_frame"]["parameters"]["reasoning"] == {"effort": "[REDACTED]"}
    assert not any(word in json.dumps(cleaned) for word in (
        "known-private-value", "another-private-value", "opaque-private-value", "sealed-private-value"))
    assert capture.sanitize_training_value(cleaned) == (cleaned, redactions)


def test_repeated_text_scans_once_but_still_checks_every_node(monkeypatch):
    public = "public catalog description " * 100
    oversized = "public response text " * 1000
    value = {"input_snapshots": {"run": [public] * 40},
             "provider_calls": [{"response": [public] * 40}],
             "large": [oversized, oversized]}
    original = trajectory.redact_text
    calls = {public: 0, oversized: 0}

    def counted(text, *args, **kwargs):
        if text in calls:
            calls[text] += 1
        return original(text, *args, **kwargs)

    monkeypatch.setattr(trajectory, "redact_text", counted)
    assert capture.sanitize_training_value(value, aggregate_parts=True) == (value, [])
    assert calls == {public: 1, oversized: 1}


def _rejected_http_model_request():
    return {"admitted_parameters": {"method": "POST",
        "url": "https://openrouter.ai/api/v1/responses", "headers": {},
        "body_text": {"model": "synthetic/model", "input": "synthetic",
                      "instructions": "Return JSON", "tools": [], "provider": {},
                      "max_output_tokens": 12000, "reasoning": {"effort": "high"}}},
        "timeout_ms": 0}


@pytest.mark.parametrize("controls", [
    {"effort": "low"}, {"effort": "high"},
    {"summary": "concise", "context": "all_turns"},
    {"enabled": True, "exclude": False, "max_tokens": 2048},
    {"enabled": False, "exclude": True}, {},
])
def test_rejected_http_preserves_only_control_values(controls):
    value = _rejected_http_model_request()
    value["admitted_parameters"]["body_text"]["reasoning"] = controls
    assert capture.sanitize_training_value(value) == (value, [])


@pytest.mark.parametrize("mutation", [
    "wrong_host", "plain_http", "query", "userinfo", "port", "wrong_path", "get",
    "missing_model", "invalid_headers", "extra_admitted", "wrong_timeout",
    "provider_response", "body_response", "arbitrary_wrapper", "wrong_protocol",
    "wrong_observation_kind", "paid_identity",
])
def test_rejected_http_cannot_create_trust_from_other_contexts(mutation):
    value = _rejected_http_model_request()
    admitted = value["admitted_parameters"]
    if mutation == "wrong_host": admitted["url"] = "https://example.com/api/v1/responses"
    elif mutation == "plain_http": admitted["url"] = "http://openrouter.ai/api/v1/responses"
    elif mutation == "query": admitted["url"] += "?source=synthetic"
    elif mutation == "userinfo": admitted["url"] = "https://user@openrouter.ai/api/v1/responses"
    elif mutation == "port": admitted["url"] = "https://openrouter.ai:443/api/v1/responses"
    elif mutation == "wrong_path": admitted["url"] = "https://openrouter.ai/other"
    elif mutation == "get": admitted["method"] = "GET"
    elif mutation == "missing_model": del admitted["body_text"]["model"]
    elif mutation == "invalid_headers": admitted["headers"] = []
    elif mutation == "extra_admitted": admitted["other"] = True
    elif mutation == "wrong_timeout": value["timeout_ms"] = 1000
    elif mutation == "provider_response": value.update(object="response", output=[])
    elif mutation == "body_response": admitted["body_text"].update(object="chat.completion", choices=[])
    elif mutation == "arbitrary_wrapper": value = {"nested": value}
    else:
        value = {"ordinal": 0, "privacy": {"redacted": False, "redactions": []},
                 "kind": "local_parse_error", "operation_id": "unknown", "action_sequence": None,
                 "call_identity": None, "contributes_to_provider_call_count": False,
                 "request": value, "outcome": {"protocol": "http"}}
        if mutation == "wrong_protocol": value["outcome"]["protocol"] = "frame"
        elif mutation == "wrong_observation_kind": value["kind"] = "transport_failure"
        else: value["call_identity"] = "sha256:" + "1" * 64
    cleaned, redactions = capture.sanitize_training_value(value)
    assert redactions == (["credential", "hidden_reasoning"] if mutation == "userinfo" else ["hidden_reasoning"])
    assert '"effort": "high"' not in json.dumps(cleaned)


@pytest.mark.parametrize("controls", [
    {"effort": "private prose"}, {"effort": {"text": "opaque state"}},
    {"effort": "high", "encrypted_content": "sealed state"},
    {"effort": "high", "authorization": "private-value"},
])
def test_rejected_http_invalid_controls_are_still_private(controls):
    value = _rejected_http_model_request()
    value["admitted_parameters"]["body_text"]["reasoning"] = controls
    cleaned, redactions = capture.sanitize_training_value(value)
    assert redactions == ["hidden_reasoning"]
    assert cleaned["admitted_parameters"]["body_text"]["reasoning"] == "[REDACTED]"


def test_rejected_http_keeps_nested_reasoning_and_authentication_private():
    value = _rejected_http_model_request()
    admitted = value["admitted_parameters"]
    admitted["headers"] = {"authorization": "Bearer synthetic-private-value"}
    admitted["body_text"].update(authorization="private-value",
        nested={"reasoning": {"effort": "high"}}, encrypted_content="sealed state")
    cleaned, redactions = capture.sanitize_training_value(value)
    assert redactions == ["credential", "hidden_reasoning"]
    body = cleaned["admitted_parameters"]["body_text"]
    assert body["reasoning"] == {"effort": "high"}
    assert body["nested"]["reasoning"] == body["encrypted_content"] == "[REDACTED]"
    assert body["authorization"] == cleaned["admitted_parameters"]["headers"]["authorization"] == "[REDACTED]"
    assert capture.sanitize_training_value(cleaned)[0] == cleaned
    exact, changes = capture.sanitize_training_value(value, secrets=("high",))
    assert "credential" in changes
    assert exact["admitted_parameters"]["body_text"]["reasoning"] == {"effort": "[REDACTED]"}


def test_aggregate_reuses_short_plain_keys_and_values_with_node_limit(monkeypatch):
    label = "public short label"
    value_text = "public short value"
    value = {"catalog": [{"label": label, "value": value_text} for _ in range(80)]}
    original = trajectory.redact_text
    calls = {"label": 0, "value": 0, label: 0, value_text: 0}

    def counted(text, *args, **kwargs):
        if text in calls:
            calls[text] += 1
        return original(text, *args, **kwargs)

    monkeypatch.setattr(trajectory, "redact_text", counted)
    assert capture.sanitize_training_value(value, aggregate_parts=True) == (value, [])
    assert calls == {key: 1 for key in calls}
    with pytest.raises(capture.CaptureError, match="request_structure_limit"):
        capture.sanitize_training_value(value, aggregate_parts=True, max_nodes=15)


def test_aggregate_short_cache_preserves_privacy_and_custom_string_behavior():
    secret = "private-token-value"
    value = {"catalog": ["private " + secret] * 20}
    cleaned, reasons = capture.sanitize_training_value(
        value, secrets=(secret,), aggregate_parts=True)
    assert cleaned == {"catalog": ["private [REDACTED]"] * 20}
    assert reasons == ["credential"]
    assert capture.sanitize_training_value(value, aggregate_parts=True) == (value, [])

    class UnhashableText(str):
        __hash__ = None

    custom = UnhashableText("public short value")
    assert capture.sanitize_training_value(
        {"catalog": [custom, custom]}, aggregate_parts=True) == (
            {"catalog": [custom, custom]}, [])


def test_aggregate_short_cache_keeps_existing_entry_bound(monkeypatch):
    texts = [f"public short item {index:03d}" for index in range(600)]
    original = trajectory.redact_text
    first_scans = 0

    def counted(text, *args, **kwargs):
        nonlocal first_scans
        if text == texts[0]:
            first_scans += 1
        return original(text, *args, **kwargs)

    monkeypatch.setattr(trajectory, "redact_text", counted)
    value = {"catalog": texts + [texts[0]]}
    assert capture.sanitize_training_value(value, aggregate_parts=True) == (value, [])
    assert first_scans == 2


def test_long_text_working_set_does_not_evict_itself(monkeypatch):
    texts = [(f"public catalog entry {index:03d} " * 50) for index in range(408)]
    original = trajectory.redact_text
    calls = {text: 0 for text in texts}

    def counted(text, *args, **kwargs):
        if text in calls:
            calls[text] += 1
        return original(text, *args, **kwargs)

    monkeypatch.setattr(trajectory, "redact_text", counted)
    value = {"catalog": texts * 3}
    assert capture.sanitize_training_value(value) == (value, [])
    assert set(calls.values()) == {1}


def test_repeated_text_cache_keeps_privacy_and_is_invocation_local():
    private = "known-private-value"
    public = "public evidence " * 80
    text = "private " + private + " " + public + "https://user:pass@example.test/?token=unknown-token"
    value = {"input_snapshots": {"run": [text] * 20},
             "authorization": "another-private-value",
             "hidden_reasoning": "opaque-private-value"}
    cleaned, reasons = capture.sanitize_training_value(value, secrets=(private,))
    assert reasons == ["credential", "hidden_reasoning"]
    assert cleaned["input_snapshots"]["run"] == [
        "private [REDACTED] " + public + "https://example.test/?token=%5BREDACTED%5D"] * 20
    assert private not in json.dumps(cleaned)
    assert "unknown-token" not in json.dumps(cleaned)
    assert cleaned["authorization"] == cleaned["hidden_reasoning"] == "[REDACTED]"
    without_secret, reasons = capture.sanitize_training_value(value)
    assert private in without_secret["input_snapshots"]["run"][0]
    assert reasons == ["credential", "hidden_reasoning"]
    with pytest.raises(capture.CaptureError, match="request_redaction_key_collision"):
        capture.sanitize_training_value({private: "safe", "[REDACTED]": "safe"}, secrets=(private,))
