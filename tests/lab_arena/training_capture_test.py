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

from lab_arena import broker as br, contracts, operations, trajectory, training_capture as capture
from lab_arena.service import LocalObjectStore
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
@pytest.mark.parametrize("response", [None, {"choices": []}, {"object": "response", "output": []}])
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
