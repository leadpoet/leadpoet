"""Large retained histories must not require whole-document encoding copies."""

import hashlib
import json
import math
import random
import tracemalloc

import pytest

from lab_arena import contracts, training_export
from tests.lab_arena.training_export_test import fixture


def legacy_hash(value):
    return "sha256:" + hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False).encode("utf-8")).hexdigest()


@pytest.mark.parametrize("value", [
    None, True, 0, -0.0, 1e-300, 1e300, 10**100,
    {"unicode": "日本語 🌍", "escapes": "\n\t\"\\\x00", "tuple": (1, None)},
    {3: "integer key", 2: "other"}, {False: "boolean key"}, {None: "null key"},
    {"b": [{"z": 3, "a": 2}], "a": []},
])
def test_hash_is_identical_to_existing_canonical_bytes(value):
    assert contracts.sha256_json(value) == legacy_hash(value)


def test_random_finite_float_hashes_match_existing_encoder():
    rng = random.Random(926)
    values = [rng.uniform(-1, 1) * 10.0**rng.randrange(-300, 301)
              for _ in range(1000)]
    assert contracts.document_hash(values) == legacy_hash(values)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, object(),
                                    {"a": 1, 2: "mixed keys"}])
def test_hash_still_rejects_noncanonical_values(value):
    with pytest.raises(contracts.ArenaContractError):
        contracts.document_hash(value)


def test_hash_still_rejects_circular_values_and_invalid_utf8():
    cyclic = []
    cyclic.append(cyclic)
    with pytest.raises(contracts.ArenaContractError):
        contracts.document_hash(cyclic)
    with pytest.raises(UnicodeEncodeError):
        contracts.document_hash("\ud800")


def test_hash_memory_does_not_scale_with_repeated_history_size():
    # 32 MiB on the wire, but only one retained immutable string in memory.
    value = ["public evidence " * 65536] * 32
    expected = legacy_hash(value)
    tracemalloc.start()
    try:
        actual = contracts.document_hash(value)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert actual == expected
    assert peak < 8 * 1024**2


def test_hash_batches_small_encoder_fragments_without_changing_bytes(monkeypatch):
    value = {"catalog": [{"name": str(i), "description": "evidence " * 24}
                         for i in range(2000)]}
    expected = legacy_hash(value)
    real_sha256 = hashlib.sha256
    sizes = []

    class MeasuredHash:
        def __init__(self):
            self.inner = real_sha256()

        def update(self, data):
            sizes.append(len(data))
            self.inner.update(data)

        def hexdigest(self):
            return self.inner.hexdigest()

    monkeypatch.setattr(contracts.hashlib, "sha256", MeasuredHash)
    assert contracts.document_hash(value) == expected
    # Thousands of schema fragments must not each cross the hashlib boundary.
    assert len(sizes) < 20
    assert max(sizes) <= 64 * 1024


@pytest.mark.parametrize("length", [65533, 65534, 65535, 65536, 65537, 131072])
def test_hash_buffer_boundaries_preserve_unicode_and_surrounding_values(length):
    value = ["small", {"text": "x" * length, "unicode": "日本語 🌍" * 12000}, None]
    assert contracts.document_hash(value) == legacy_hash(value)


def test_export_preserves_json_values_hash_privacy_and_input_isolation():
    values = fixture()
    values["round_row"]["configuration_doc"]["extra"] = {
        "tuple": ("日本語", 1.25, None), "authorization": "private-sentinel"}
    example = training_export.assemble_example(**values)
    assert example == json.loads(contracts.canonical_json(example))
    assert example["configuration"]["extra"] == {
        "tuple": ["日本語", 1.25, None], "authorization": "[REDACTED]"}
    body = {k: v for k, v in example.items() if k != "example_hash"}
    assert example["example_hash"] == legacy_hash(body)
    example["configuration"]["extra"]["tuple"].append("changed")
    assert values["round_row"]["configuration_doc"]["extra"]["tuple"] == (
        "日本語", 1.25, None)


def test_export_does_not_expand_shared_history_into_duplicate_strings():
    values = fixture()
    values["round_row"]["configuration_doc"]["public_context"] = [
        "Public company information. " * 4096] * 64
    tracemalloc.start()
    try:
        example = training_export.assemble_example(**values)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert len(example["configuration"]["public_context"]) == 64
    # Assembly still validates individual documents. The whole-example copies
    # previously raised this small fixture above 14 MiB by themselves.
    assert peak < 12 * 1024**2
