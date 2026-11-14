"""Non-scalar response text is analysis evidence, never an exact response."""

import base64
import copy
import json

import pytest

from lab_arena import broker, contracts, training_eligibility, training_export, training_steps, training_thinking
from tests.lab_arena.training_export_test import add_call, fixture
from tests.lab_arena.training_recovered_deepline_response_test import recovered_values
from tests.lab_arena.training_steps_test import seal
from tests.lab_arena.training_thinking_test import full_example


def actual_shape_response():
    # Public fixture only. Match the saved failing receipt's structure/size,
    # without reproducing any private provider text or field names.
    value = {}
    nested = value
    for _ in range(6):
        nested["nested"] = {}
        nested = nested["nested"]
    nested["text"] = "public " * 5097 + "\ud800"
    value.update({f"object_{i}": {"label": "public"} for i in range(36)})
    value.update({f"array_{i}": ["public"] * count
                  for i, count in enumerate([8, 8, 7, 7, 7, 7])})
    value.update({f"string_{i}": "public" for i in range(26)})
    value.update({f"number_{i}": i for i in range(20)})
    value.update(true=True, false=False)
    value["string_0"] += "x" * (43026 - len(body_bytes(value)))
    assert len(body_bytes(value)) == 43026
    return value


def body_bytes(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True,
                      separators=(",", ":")).encode()


def values_with_response(response):
    values = fixture()
    add_call(values, provider="deepline", operation="deepline.execute")
    values["ledger_rows"][-1]["terminal_response"]["body_b64"] = base64.b64encode(
        body_bytes(response)).decode()
    return values


def test_actual_shape_surrogate_retains_nonexact_analysis_and_wire_proof():
    response = actual_shape_response()
    values = values_with_response(response)
    original = copy.deepcopy(values["ledger_rows"])
    exported = training_export.assemble_example(**values)
    call = exported["provider_calls"][0]
    receipt = call["response_receipts"][0]
    rendering = receipt["nonexact_response_rendering"]
    assert rendering["exact_response"] is False
    assert rendering["invalid_unicode_codepoints"] == 1
    assert rendering["format"] == "json_escaped_invalid_unicode"
    assert json.loads(rendering["text"]) == response
    assert receipt["wire_body_sha256"] == contracts.hash_bytes(body_bytes(response))
    assert receipt["wire_body_bytes"] == 43026
    assert receipt["response_unavailable"] is True
    assert "response" not in call and "response_sha256" not in call
    assert "response" not in receipt and "response_sha256" not in receipt
    assert "full_response_unavailable" in call["evidence_gaps"]
    assert values["ledger_rows"] == original
    trace = training_thinking.readable_trace(exported)
    events = trace["calls"][0]["events"]
    event = next(row for row in events if row["kind"] == "nonexact_response_analysis")
    assert event["value"] == rendering
    assert event["source_path"] == ["provider_calls", 0, "response_receipts", 0,
                                    "nonexact_response_rendering"]
    assert event["delivery"] == "not_proved"
    contracts.canonical_json(exported).encode("utf-8")


def legacy_evidence(terminal, provider):
    value, redactions = training_export._response_value(terminal, provider)
    return value, redactions, contracts.document_hash(value), None


def assessment(values, example):
    return training_eligibility.assess_capture(
        values["submission_row"], {example["run_id"]: example}, [example["run_id"]],
        enabled=True, minimum_coverage_percent=80, maximum_missing_reasoning_percent=1)


def test_analysis_does_not_change_strict_pairs_completeness_or_eligibility(monkeypatch):
    values = values_with_response(actual_shape_response())
    current = training_export.assemble_example(**values)
    monkeypatch.setattr(training_export, "_response_evidence", legacy_evidence)
    before = training_export.assemble_example(**values)
    assert current["completeness"] == before["completeness"]
    assert current["privacy"] == before["privacy"]
    assert current["costs"] == before["costs"]
    assert assessment(values, current) == assessment(values, before)
    for builder in (training_steps.build, training_thinking.build):
        after_rows, _ = builder([current])
        before_rows, _ = builder([before])
        for split in ("train", "eval"):
            assert after_rows[split] == before_rows[split] == []


def test_clean_selected_model_pair_survives_nonexact_prior_receipt():
    receipt = training_export.assemble_example(**values_with_response({"text": "\ud800"}))[
        "provider_calls"][0]["response_receipts"][0]
    current = full_example()
    current["provider_calls"][0]["response_receipts"].insert(0, receipt)
    before = copy.deepcopy(current)
    before["provider_calls"][0]["response_receipts"][0].pop("nonexact_response_rendering")
    before["provider_calls"][0]["response_receipts"][0].pop("privacy_redactions")
    for builder in (training_steps.build, training_thinking.build):
        after_splits, _ = builder([seal(current)])
        before_splits, _ = builder([seal(before)])
        after = after_splits["train"] + after_splits["eval"]
        old = before_splits["train"] + before_splits["eval"]
        assert len(after) == len(old) == 1
        for key in ("input", "target", "generated_thinking"):
            assert after[0].get(key) == old[0].get(key)


@pytest.mark.parametrize("response", [
    {"text": "日本語 🌍", "literal": "\\ud800", "result": [1, True, None]},
    {"text": "\ud83d\ude00"},  # The existing JSON parser decodes a legal pair to one scalar.
    {"text": "x" * 35680},
])
def test_valid_unicode_exports_and_readable_projection_are_unchanged(response, monkeypatch):
    values = values_with_response(response)
    current = training_export.assemble_example(**values)
    monkeypatch.setattr(training_export, "_response_evidence", legacy_evidence)
    before = training_export.assemble_example(**values)
    assert current == before
    assert training_thinking.readable_trace(current) == training_thinking.readable_trace(before)
    assert current["provider_calls"][0]["response"] == json.loads(body_bytes(response))


@pytest.mark.parametrize("response,invalid", [
    ({"text": "\ud800"}, 1), ({"text": "\udc00"}, 1),
    ({"\ud800": "日本語 🌍"}, 1), ({"text": "\ude00\ud83d"}, 2),
])
def test_non_scalar_variants_are_visible_but_never_normalized(response, invalid):
    call = training_export.assemble_example(**values_with_response(response))["provider_calls"][0]
    rendering = call["response_receipts"][0]["nonexact_response_rendering"]
    expected = json.dumps(response, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    assert rendering["text"] == expected.encode("utf-8", "backslashreplace").decode()
    assert rendering["invalid_unicode_codepoints"] == invalid
    assert rendering["exact_response"] is False
    assert "response" not in call
    with pytest.raises(UnicodeEncodeError):
        contracts.document_hash(response)  # The canonical security boundary is unchanged.


def test_privacy_and_opaque_filters_precede_rendering():
    values = values_with_response({"text": "日本語 🌍 \ud800", "api_key": "fixture-secret",
                                   "encrypted_content": "fixture-opaque"})
    exported = training_export.assemble_example(**values)
    call = exported["provider_calls"][0]
    receipt = call["response_receipts"][0]
    text = receipt["nonexact_response_rendering"]["text"]
    assert "fixture-secret" not in text and "fixture-opaque" not in text
    assert "日本語 🌍" in text and "\\ud800" in text
    assert set(receipt["privacy_redactions"]) == {"credential", "hidden_reasoning"}
    assert "response_privacy_nonexact" in call["evidence_gaps"]
    assert exported["privacy"]["privacy_filtered"] is True
    assert exported["privacy"]["hidden_reasoning_removed"] is True
    assert exported["privacy"]["exact_replay"] is False


def test_invalid_unicode_in_removed_private_fields_needs_no_analysis_rendering():
    call = training_export.assemble_example(**values_with_response({
        "text": "valid", "api_key": "fixture-secret\ud800", "encrypted_content": "fixture-opaque\ud800"}))[
        "provider_calls"][0]
    assert call["response"] == {"text": "valid", "api_key": "[REDACTED]"}
    assert "nonexact_response_rendering" not in call["response_receipts"][0]
    assert set(call["response_privacy_redactions"]) == {"credential", "hidden_reasoning"}


def test_prior_terminal_analysis_keeps_selected_valid_response():
    values = values_with_response({"text": "\ud800"})
    later = copy.deepcopy(values["ledger_rows"][-1])
    later.update(entry_id=4, entry_kind="recovery")
    later["terminal_response"]["body_b64"] = base64.b64encode(body_bytes({"text": "valid"})).decode()
    values["ledger_rows"].append(later)
    call = training_export.assemble_example(**values)["provider_calls"][0]
    assert call["response"] == {"text": "valid"}
    assert "full_response_unavailable" not in call["evidence_gaps"]
    assert call["response_receipts"][0]["response_unavailable"] is True
    assert call["response_receipts"][0]["nonexact_response_rendering"]["exact_response"] is False
    assert call["response_receipts"][1]["response_ref"] == {"field": "response", "entry_id": 4}


def upstream_values():
    values = values_with_response({"text": "valid bridge error"})
    reserve = values["ledger_rows"][0]
    raw = body_bytes({"text": "\ud800", "api_key": "fixture-secret"})
    # Construct the already-stored receipt envelope; no provider call or broker
    # capture behavior is changed by this export-only regression.
    captured = broker._capture_upstream_response(broker.ProviderResponse(503, {}, b'{}'),
        operation_id=reserve["operation_id"], request_hash=reserve["entry_doc"]["request_hash"],
        call_identity=reserve["call_identity"])
    captured.update(body_b64=base64.b64encode(raw).decode(), body_sha256=contracts.hash_bytes(raw),
                    body_bytes=len(raw), wire_body_sha256=contracts.hash_bytes(raw), wire_body_bytes=len(raw))
    values["ledger_rows"][-1]["terminal_response"]["upstream_response_receipt"] = captured
    return values


def test_upstream_analysis_keeps_bridge_response_and_identity_checks():
    values = upstream_values()
    exported = training_export.assemble_example(**values)
    call = exported["provider_calls"][0]
    assert call["response"] == {"text": "valid bridge error"}
    receipt = call["upstream_response_receipts"][0]
    assert receipt["response_unavailable"] and "response_sha256" not in receipt
    assert receipt["privacy_redactions"] == ["credential"]
    assert "fixture-secret" not in receipt["nonexact_response_rendering"]["text"]
    assert {"upstream_response_capture_unavailable", "upstream_response_privacy_nonexact"} <= set(call["evidence_gaps"])
    assert any(row["source_path"] == ["provider_calls", 0, "upstream_response_receipts", 0,
                                      "nonexact_response_rendering"]
               for row in training_thinking.readable_trace(exported)["calls"][0]["events"])
    values["ledger_rows"][-1]["terminal_response"]["upstream_response_receipt"]["call_identity"] = "wrong"
    assert "upstream_response_receipts" not in training_export.assemble_example(**values)["provider_calls"][0]


@pytest.mark.parametrize("kind", ["execute", "score"])
def test_recovered_non_scalar_body_is_diagnostic_not_delivery_authority(kind):
    values = recovered_values(kind=kind)
    recovered = values["ledger_rows"][-1]["training_recovered_response"]
    recovered["terminal_response"]["body_b64"] = base64.b64encode(body_bytes({
        "text": "\ud800", "api_key": "fixture-secret", "encrypted_content": "fixture-opaque"})).decode()
    original = copy.deepcopy(values["ledger_rows"])
    exported = training_export.assemble_example(**values)
    call = exported["provider_calls"][0]
    receipt = call["nonexact_recovered_response"]
    assert receipt["wire_body_sha256"] == contracts.hash_bytes(base64.b64decode(recovered["terminal_response"]["body_b64"]))
    assert "fixture-secret" not in receipt["nonexact_response_rendering"]["text"]
    assert "fixture-opaque" not in receipt["nonexact_response_rendering"]["text"]
    assert set(receipt["privacy_redactions"]) == {"credential", "hidden_reasoning"}
    assert exported["privacy"]["hidden_reasoning_removed"] is True
    assert "recovered_response_ref" not in call
    assert call["response_status"] == 502  # Original ledger reply, never the recovered 200.
    assert "full_response_unavailable" in call["evidence_gaps"]
    assert "worker_recovered_response_ref" not in call["response_provenance"]
    assert values["ledger_rows"] == original
    trace = training_thinking.readable_trace(exported)
    assert trace["calls"][0]["nonexact_recovered_response"] == receipt
    assert any(row["kind"] == "nonexact_response_analysis" for row in trace["calls"][0]["events"])
    for builder in (training_steps.build, training_thinking.build):
        splits, _ = builder([exported])
        assert not splits["train"] and not splits["eval"]


@pytest.mark.parametrize("damage", ["nonfinite", "mixed_nonfinite", "base64", "size"])
def test_other_invalid_evidence_gets_no_unicode_fallback(damage, monkeypatch):
    values = values_with_response({"text": "\ud800"})
    terminal = values["ledger_rows"][-1]["terminal_response"]
    if damage in {"nonfinite", "mixed_nonfinite"}:
        response = {"text": float("nan")}
        if damage == "mixed_nonfinite":
            response["a"] = "\ud800"
        terminal["body_b64"] = base64.b64encode(body_bytes(response)).decode()
    elif damage == "base64":
        terminal["body_b64"] = "!invalid!"
    else:
        monkeypatch.setattr(training_export, "MAX_RECEIPT_BYTES", 1)
    call = training_export.assemble_example(**values)["provider_calls"][0]
    assert "response" not in call
    assert "nonexact_response_rendering" not in call["response_receipts"][0]
    assert "full_response_unavailable" in call["evidence_gaps"]


@pytest.mark.parametrize("limit,has_text", [(100, False), (180, False), (300, True)])
def test_rendered_wrapper_and_text_are_bounded_without_clipping(limit, has_text, monkeypatch):
    response = {"text": "\\" * 30 + "\ud800"}
    assert len(body_bytes(response)) < limit
    values = values_with_response(response)
    monkeypatch.setattr(training_export, "MAX_RECEIPT_BYTES", limit)
    call = training_export.assemble_example(**values)["provider_calls"][0]
    rendering = call["response_receipts"][0]["nonexact_response_rendering"]
    assert ("text" in rendering) is has_text
    assert len(contracts.canonical_json(rendering).encode()) <= limit
    if has_text:
        assert json.loads(rendering["text"]) == response
    else:
        assert rendering["unavailable_reason"] == "size_limit"
    assert "response" not in call
