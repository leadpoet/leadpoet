"""Keep closing markup after an already-redacted value; remove actual secrets."""
import base64
import json

import pytest

from lab_arena import training_capture, training_export, trajectory


# Exact public fragment from a captured company press-release response. The
# provider had already supplied the marker and trailing closing bracket.
PUBLIC_FRAGMENT = "Context-Aware Authorization:[REDACTED]]"


def test_public_fragment_stays_exact_through_stream_capture_and_export():
    assert trajectory.redact_auth_schemes(PUBLIC_FRAGMENT) == PUBLIC_FRAGMENT
    assert trajectory.redact_text(PUBLIC_FRAGMENT) == PUBLIC_FRAGMENT
    assert trajectory.event("runtime.stdout", {"text": PUBLIC_FRAGMENT})["content"]["text"] == PUBLIC_FRAGMENT
    source = {"result": {"data": {"markdown": PUBLIC_FRAGMENT + " [Source](https://example.com/page)"}}}
    cleaned, redactions = training_capture.sanitize_training_value(source)
    assert cleaned == source and redactions == []
    raw = json.dumps(source).encode()
    exported, redactions = training_export._response_value(
        {"body_b64": base64.b64encode(raw).decode()}, "deepline")
    assert exported == source and redactions == []


@pytest.mark.parametrize("label", ["Authorization", "Proxy-Authorization", "api_key", "password"])
@pytest.mark.parametrize("marker", ["[REDACTED]", '"[REDACTED]"', "'[REDACTED]'"])
@pytest.mark.parametrize("closing", ["]", ")", "}", "]})"])
def test_existing_redacted_values_keep_only_closing_markup(label, marker, closing):
    text = label + ": " + marker + closing
    assert trajectory.redact_text(text) == text
    assert training_capture.sanitize_training_value(text) == (text, [])


@pytest.mark.parametrize("suffix", [
    "actual-secret", "/actual-secret", ".actual-secret", ":actual-secret",
    "%2Factual-secret", "\\actual-secret", "]actual-secret", ")actual-secret", "}actual-secret",
    "]actual-secret]", ")/actual-secret", "}+actual-secret", "+/=._-",
])
@pytest.mark.parametrize("label", ["Authorization", "api_key"])
def test_attached_credential_material_is_not_treated_as_closing_markup(label, suffix):
    text = label + ": [REDACTED]" + suffix
    for cleaned in (trajectory.redact_text(text), training_capture.sanitize_training_value(text)[0]):
        assert suffix not in cleaned
        assert cleaned.endswith("[REDACTED]")
    assert training_capture.sanitize_training_value(text)[1] == ["credential"]


@pytest.mark.parametrize("text,secret", [
    ("Authorization: actual-private-value]", "actual-private-value"),
    ("api_key=actual-private-value)", "actual-private-value"),
    ("password=actual-private-value}", "actual-private-value"),
    ("Authorization: Bearer opaque-private-token]", "opaque-private-token"),
    ("Authorization: Basic dXNlcjpwYXNz]", "dXNlcjpwYXNz"),
    ("Bearer [REDACTED]actual-private-tail]", "actual-private-tail"),
    ("Basic [REDACTED]/actual-private-tail)", "actual-private-tail"),
])
def test_real_credentials_remain_redacted_with_closing_markup(text, secret):
    assert secret not in trajectory.redact_text(text)
    cleaned, redactions = training_capture.sanitize_training_value(text)
    assert secret not in cleaned and redactions == ["credential"]


def test_known_secrets_and_structured_private_fields_remain_private():
    text = "Authorization: known-private-value]"
    cleaned, redactions = training_capture.sanitize_training_value(text, ("known-private-value",))
    assert "known-private-value" not in cleaned and redactions == ["credential"]
    source = {"authorization": "[REDACTED]]", "api_key": "actual-private-value]"}
    assert training_capture.sanitize_training_value(source) == (
        {"authorization": "[REDACTED]", "api_key": "[REDACTED]"}, ["credential"])
