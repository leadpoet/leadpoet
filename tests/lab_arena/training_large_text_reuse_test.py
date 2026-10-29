"""Bounded aggregate-only reuse must preserve the sanitizer's full walk."""

from collections import OrderedDict

import pytest

from lab_arena import training_capture as capture, trajectory


CACHE_BYTES = 8 * 1024 * 1024


def count_scans(monkeypatch):
    calls = []
    original = trajectory.redact_text

    def counted(text, *args, **kwargs):
        if len(text) > 16_384:
            calls.append(text)
        return original(text, *args, **kwargs)

    monkeypatch.setattr(trajectory, "redact_text", counted)
    return calls


@pytest.mark.parametrize("unit", ["public ASCII evidence ", "🧭", "é界🧭"])
def test_repeated_large_plain_text_scans_once_per_aggregate_invocation(monkeypatch, unit):
    text = unit * (17_000 // len(unit) + 1)
    value = {"input_snapshots": {"run": [text] * 3},
             "provider_calls": [{"response": text}], "configuration": text}
    calls = count_scans(monkeypatch)
    assert capture.sanitize_training_value(value, aggregate_parts=True) == (value, [])
    assert calls == [text]
    assert capture.sanitize_training_value(value, aggregate_parts=True) == (value, [])
    assert calls == [text, text]
    # Standalone requests retain the original uncached large-string path.
    assert capture.sanitize_training_value([text, text]) == ([text, text], [])
    assert calls == [text] * 4


def test_large_reuse_keeps_secret_auth_url_opaque_and_key_rules(monkeypatch):
    private = "known-private-value"
    text = ("public evidence " * 1_200 + private +
            " Bearer sk-abcdefghijklmnopqrstuvwxyz123456 "
            "https://user:pass@example.test/?token=unknown-token")
    value = {"input_snapshots": {"run": [text, text]},
             "authorization": text, "hidden_reasoning": text}
    expected = capture.sanitize_training_value(value, secrets=(private,))
    calls = count_scans(monkeypatch)
    assert capture.sanitize_training_value(value, secrets=(private,), aggregate_parts=True) == expected
    assert len(calls) == 1
    cleaned, reasons = expected
    assert reasons == ["credential", "hidden_reasoning"]
    output = cleaned["input_snapshots"]["run"][0]
    assert not any(secret in output for secret in (private, "abcdefghijklmnopqrstuvwxyz", "unknown-token", "user:pass"))
    assert cleaned["authorization"] == cleaned["hidden_reasoning"] == "[REDACTED]"
    assert private in capture.sanitize_training_value([text], aggregate_parts=True)[0][0]
    key = "public " * 2_500 + private
    safe_key = key.replace(private, "[REDACTED]")
    with pytest.raises(capture.CaptureError, match="request_redaction_key_collision"):
        capture.sanitize_training_value({key: 1, safe_key: 2}, secrets=(private,), aggregate_parts=True)


def test_unique_churn_evicts_before_insert_with_unicode_key_and_output_budget(monkeypatch):
    records = []

    class ObservedCache(OrderedDict):
        def __setitem__(self, key, value):
            if isinstance(value, tuple):
                # Observe retained strings at the insertion boundary, including
                # the pending pair. Count both references conservatively.
                charge = sum(str.__sizeof__(k) + str.__sizeof__(v[0]) for k, v in self.items())
                charge += str.__sizeof__(key) + str.__sizeof__(value[0])
                assert charge <= CACHE_BYTES
                assert len(self) < 512
                records.append((charge, len(self) + 1))
            super().__setitem__(key, value)

    monkeypatch.setattr(capture, "OrderedDict", ObservedCache)
    calls = count_scans(monkeypatch)
    texts = [f"{index:03d}:" + "界" * 17_000 for index in range(260)]
    value = texts + [texts[-1], texts[0], texts[-1]]
    assert capture.sanitize_training_value(value, aggregate_parts=True) == (value, [])
    assert calls.count(texts[0]) == 2
    assert calls.count(texts[-1]) == 1
    assert len(calls) == 261
    assert records and max(count for _, count in records) < 260


@pytest.mark.parametrize("text,secrets", [
    ("🧭" * 1_050_000, ()),
    # Replacing each four-character secret expands the cleaned output. The
    # input alone fits; retaining both input and output exceeds the budget.
    ("aaaa" * 650_000, ("aaaa",)),
], ids=["wide-unicode", "expanded-secret"])
def test_oversized_unicode_or_expanded_output_is_not_retained(monkeypatch, text, secrets):
    assert str.__sizeof__(text) < CACHE_BYTES
    calls = count_scans(monkeypatch)
    cleaned, _ = capture.sanitize_training_value([text, text], secrets=secrets, aggregate_parts=True)
    assert cleaned[0] == cleaned[1]
    assert len(calls) == 2


def test_unpaired_surrogates_need_no_utf8_encoding(monkeypatch):
    text = "\ud800" * 17_000
    calls = count_scans(monkeypatch)
    assert capture.sanitize_training_value([text, text], aggregate_parts=True) == ([text, text], [])
    assert calls == [text]


@pytest.mark.parametrize("aggregate", [False, True])
def test_custom_strings_keep_hash_encoding_size_and_container_behavior(monkeypatch, aggregate):
    class CustomText(str):
        def __hash__(self):
            raise AssertionError("custom hash")

        def encode(self, *args, **kwargs):
            raise AssertionError("custom encode")

        def __sizeof__(self):
            raise AssertionError("custom size")

    class CustomList(list):
        iterations = 0

        def __iter__(self):
            self.iterations += 1
            return super().__iter__()

    text = CustomText("public " * 2_500)
    value = CustomList([text, text])
    calls = count_scans(monkeypatch)
    assert capture.sanitize_training_value(value, aggregate_parts=aggregate) == ([text, text], [])
    assert value.iterations == 1 and len(calls) == 2
    # The old short-string cache still invokes custom hash and raises.
    with pytest.raises(AssertionError, match="custom hash"):
        capture.sanitize_training_value(CustomText("x" * 2_000), aggregate_parts=aggregate)


def test_custom_secret_variant_disables_only_new_large_lane(monkeypatch):
    class CustomSecret(str):
        pass

    text = "public text " * 1_500
    calls = count_scans(monkeypatch)
    assert capture.sanitize_training_value([text, text], secrets=(CustomSecret("private-value"),),
                                           aggregate_parts=True) == ([text, text], [])
    assert calls == [text, text]


@pytest.mark.parametrize("max_nodes,error", [(3, "request_structure_limit"), (4, "request_value_invalid")])
def test_reused_leaves_still_count_nodes_and_preserve_error_order(monkeypatch, max_nodes, error):
    text = "public " * 2_500
    calls = count_scans(monkeypatch)
    with pytest.raises(capture.CaptureError, match=error):
        capture.sanitize_training_value([text, text, object()], max_nodes=max_nodes, aggregate_parts=True)
    assert calls == [text]


def test_nonplain_scan_output_is_not_retained(monkeypatch):
    class CustomOutput(str):
        def __sizeof__(self):
            raise AssertionError("custom output size")

    text = "public " * 2_500
    original = trajectory.redact_text
    scans = []

    def custom_output(value, *args, **kwargs):
        cleaned = original(value, *args, **kwargs)
        if value == text:
            scans.append(1)
            return CustomOutput(cleaned)
        return cleaned

    monkeypatch.setattr(trajectory, "redact_text", custom_output)
    cleaned, reasons = capture.sanitize_training_value([text, text], aggregate_parts=True)
    assert cleaned == [text, text] and reasons == []
    assert all(type(value) is CustomOutput for value in cleaned)
    assert len(scans) == 2


@pytest.mark.parametrize("aggregate", [False, True])
def test_custom_mapping_and_length_checks_are_not_prewalked(monkeypatch, aggregate):
    class CustomMapping(dict):
        item_reads = 0

        def items(self):
            self.item_reads += 1
            return super().items()

    class CustomText(str):
        lengths = 0

        def __len__(self):
            self.lengths += 1
            return super().__len__()

    text = CustomText("public " * 2_500)
    value = CustomMapping(public=[text, text])
    assert capture.sanitize_training_value(value, aggregate_parts=aggregate) == (value, [])
    assert value.item_reads == 1
    # Two original short-cache eligibility checks per long subclass leaf.
    assert text.lengths == 4


@pytest.mark.parametrize("aggregate", [False, True])
def test_custom_large_key_keeps_scan_before_original_hash_failure(monkeypatch, aggregate):
    class CustomKey(str):
        def __hash__(self):
            raise AssertionError("custom key hash")

    key = CustomKey("public " * 2_500)

    class CustomMapping(dict):
        def items(self):
            return [(key, "safe")]

    calls = count_scans(monkeypatch)
    with pytest.raises(AssertionError, match="custom key hash"):
        capture.sanitize_training_value(CustomMapping(), aggregate_parts=aggregate)
    assert calls == [key]
