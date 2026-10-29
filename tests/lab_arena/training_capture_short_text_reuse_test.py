"""Short capture text reuse preserves exact evidence and all traversal checks."""
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
import threading

import pytest

from lab_arena import training_capture as capture, trajectory
from lab_arena.service import LocalObjectStore


def _spy(monkeypatch, marker):
    calls = []
    original = trajectory.redact_text

    def counted(value, *args, **kwargs):
        if value == marker:
            calls.append(value)
        return original(value, *args, **kwargs)

    monkeypatch.setattr(trajectory, "redact_text", counted)
    return calls


@pytest.mark.parametrize("aggregate", [False, True])
def test_repeated_short_schema_text_scans_once_in_each_mode(monkeypatch, aggregate):
    marker = "ordinary short schema description"
    calls = _spy(monkeypatch, marker)
    document = {"schema": [marker] * 80}
    assert capture.sanitize_training_value(document, aggregate_parts=aggregate) == (document, [])
    assert len(calls) == 1


def test_short_entries_do_not_evict_existing_medium_lane(monkeypatch):
    marker = "ordinary medium description " * 60
    calls = _spy(monkeypatch, marker)
    document = [marker, *["short value %d" % i for i in range(600)], marker]
    assert capture.sanitize_training_value(document) == (document, [])
    assert len(calls) == 1


@pytest.mark.parametrize("length", [0, 1023, 1024, 16384, 16385])
@pytest.mark.parametrize("aggregate", [False, True])
def test_cache_size_boundaries_include_short_aggregate_strings(monkeypatch, length, aggregate):
    marker = "q" * length
    calls = _spy(monkeypatch, marker)
    value = [marker] * 4
    assert capture.sanitize_training_value(value, aggregate_parts=aggregate) == (value, [])
    cached = length <= 16384 or aggregate
    assert len(calls) == (1 if cached else 4)


def test_short_string_subclass_still_scans_each_occurrence(monkeypatch):
    class Short(str):
        pass

    marker = Short("ordinary short schema description")
    calls = _spy(monkeypatch, marker)
    assert capture.sanitize_training_value([marker] * 4) == ([marker] * 4, [])
    assert len(calls) == 4


def test_mixed_subclass_collision_does_not_enter_plain_short_cache():
    class Collision(str):
        def __len__(self):
            return 1024

        def __hash__(self):
            return hash("public synthetic text")

        def __eq__(self, other):
            return other == "public synthetic text"

    value = ["public synthetic text", Collision("password: synthetic-pass-value")]
    assert capture.sanitize_training_value(value) == (
        ["public synthetic text", "password=[REDACTED]"], ["credential"])


def test_secret_subclass_retains_uncached_short_behavior(monkeypatch):
    class Secret(str):
        pass

    marker = "ordinary short schema description"
    calls = _spy(monkeypatch, marker)
    value = [marker] * 12
    assert capture.sanitize_training_value(value, secrets=(Secret("synthetic-secret"),)) == (value, [])
    assert len(calls) == 12


def test_plain_cache_is_bounded_and_invocation_local(monkeypatch):
    original = capture.OrderedDict
    instances = []

    class Tracked(original):
        def __init__(self):
            super().__init__()
            self.peak_entries = self.peak_charge = 0
            instances.append(self)

        def __setitem__(self, key, value):
            super().__setitem__(key, value)
            self.peak_entries = max(self.peak_entries, len(self))
            charge = sum(v[1] for v in self.values() if isinstance(v, tuple))
            self.peak_charge = max(self.peak_charge, charge)

    monkeypatch.setattr(capture, "OrderedDict", Tracked)
    marker = "ordinary short schema description"
    calls = _spy(monkeypatch, marker)
    value = [marker, *[("wide-%d-" % i) + "𐀀" * 900 for i in range(600)], marker]
    assert capture.sanitize_training_value(value) == (value, [])
    assert len(calls) == 2  # First entry was evicted at the count bound.
    assert len(instances) == 2
    assert instances[1].peak_entries == 512
    assert instances[1].peak_charge <= 8 * 1024 * 1024
    assert capture.sanitize_training_value([marker]) == ([marker], [])
    assert len(calls) == 3


def test_cached_text_still_checks_nodes_paths_keys_and_secret_scope():
    value = ["ordinary short text"] * 10
    with pytest.raises(capture.CaptureError, match="request_structure_limit"):
        capture.sanitize_training_value(value, max_nodes=5)
    with pytest.raises(capture.CaptureError, match="request_redaction_key_collision"):
        capture.sanitize_training_value({"synthetic-secret": 1, "[REDACTED]": 2}, secrets=("synthetic-secret",))
    source = {"admitted_parameters": {"reasoning": {"effort": "high"}},
              "response": {"reasoning": {"effort": "high"}}}
    cleaned, reasons = capture.sanitize_training_value(source)
    assert cleaned["admitted_parameters"] == source["admitted_parameters"]
    assert cleaned["response"]["reasoning"] == "[REDACTED]"
    assert reasons == ["hidden_reasoning"]
    assert capture.sanitize_training_value(["synthetic-secret"] * 2, secrets=("synthetic-secret",)) == (["[REDACTED]"] * 2, ["credential"])
    assert capture.sanitize_training_value(["synthetic-secret"]) == (["synthetic-secret"], [])


def test_concurrent_invocations_keep_independent_scans_and_secret_reasons(monkeypatch):
    marker = "ordinary short schema description"
    calls = _spy(monkeypatch, marker)
    barrier = threading.Barrier(23)

    def work(index):
        secret = "synthetic-secret-%d-end" % index
        barrier.wait()
        return capture.sanitize_training_value([marker] * 20 + [secret] * 2, secrets=(secret,))

    with ThreadPoolExecutor(max_workers=23) as pool:
        results = list(pool.map(work, range(23)))
    assert results == [([marker] * 20 + ["[REDACTED]"] * 2, ["credential"])] * 23
    assert len(calls) == 23


def _document():
    parameters = {"messages": [{"role": "user", "content": "ordinary short schema description"}] * 20,
                  "tools": [{"type": "function", "name": "read_company", "description": "ordinary short schema description"}] * 20}
    return {"operation_id": "openrouter.responses", "admitted_parameters": parameters,
            "effective_parameters": parameters, "outbound": {"body": parameters}}


def test_complete_request_prepare_storage_readback_and_export_bytes(tmp_path):
    source = _document()
    objects = LocalObjectStore(tmp_path)
    writer = capture.RequestCapture(objects)
    prepared = writer.prepare(source)
    assert isinstance(prepared, capture.PreparedRequest)
    assert prepared.persist() is True
    pointer = dict(prepared.pointer, status="stored")
    restored = capture.read_request(objects, pointer)
    expected = dict(source, schema_version=capture.SCHEMA_VERSION,
                    privacy={"redacted": False, "redactions": []})
    assert restored == expected
    expected_bytes = json.dumps(expected, sort_keys=True, separators=(",", ":"),
                                ensure_ascii=False, allow_nan=False).encode()
    # gzip's OS header varies across Python/platforms; content addresses bind
    # the canonical uncompressed bytes, not that compression metadata.
    expected_hash = "60d48460c751028241280e0db92457a72fd06a418727ad2ff3d895c2c115d566"
    assert hashlib.sha256(gzip.decompress(prepared.compressed)).hexdigest() == expected_hash
    assert pointer["sha256"] == "sha256:" + expected_hash
    assert prepared.compressed == gzip.compress(expected_bytes, compresslevel=6, mtime=0)
    assert objects.get(pointer["ref"]) == prepared.compressed
    example = {"provider_calls": [{"request_capture": restored}] * 3}
    exported, reasons = capture.sanitize_training_value(example, aggregate_parts=True)
    assert exported == example and reasons == []
    encoded = json.dumps(exported, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    assert hashlib.sha256(encoded).hexdigest() == "bf7c06ef6bdaf8d8b8c4253ab25c8eb4bca0e4228b038a2cca534408e0a0b6b1"
    assert gzip.decompress(objects.get(pointer["ref"])) == expected_bytes


@pytest.mark.parametrize("mutation,reason", [
    ("content", "request_hash_mismatch"), ("privacy", "request_privacy_mismatch"),
])
def test_changed_canonical_capture_still_fails_hash_and_privacy_checks(tmp_path, monkeypatch, mutation, reason):
    objects = LocalObjectStore(tmp_path)
    prepared = capture.RequestCapture(objects).prepare(_document())
    assert isinstance(prepared, capture.PreparedRequest)
    document = json.loads(gzip.decompress(prepared.compressed))
    if mutation == "content":
        document["admitted_parameters"]["messages"][0]["content"] = "changed capture content"
    else:
        document["privacy"] = {"redacted": True, "redactions": ["credential"]}
    raw = json.dumps(document, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False, allow_nan=False).encode()
    changed_hash = "sha256:" + hashlib.sha256(raw).hexdigest()
    assert changed_hash != prepared.pointer["sha256"]
    compressed = gzip.compress(raw, compresslevel=6, mtime=0)
    pointer = dict(prepared.pointer, size_bytes=len(raw), compressed_bytes=len(compressed))
    if mutation == "privacy":
        # A valid new content hash must not bypass the pointer's privacy binding.
        pointer.update(sha256=changed_hash, ref=capture.OBJECT_PREFIX + changed_hash[7:] + ".json.gz")
    monkeypatch.setattr(objects, "get_bounded", lambda *_: compressed)
    with pytest.raises(capture.CaptureError, match=reason):
        capture.read_request(objects, pointer)
