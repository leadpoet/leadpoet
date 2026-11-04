"""Stored input failures keep their classification through the real exporter."""

import gzip
import json

import pytest
from botocore.exceptions import ClientError

from lab_arena import contracts, training_export, training_input, training_steps, training_thinking
from tests.lab_arena.training_retry_snapshot_test import retry_values


@pytest.mark.parametrize("damage", ["valid", "missing", "404", "transient", "gzip", "json", "schema", "hash",
                                    "privacy", "private", "run", "kind", "lease", "claim", "compressed_size",
                                    "expanded_size", "store_bound", "wrong_type"])
def test_original_input_bytes_remain_missing_or_invalid_through_thinking_export(damage):
    values, _, _ = retry_values()
    row = values["runs"][0]
    ref = training_input.object_ref(row["run_id"], row["lease_generation"])
    objects = values["objects"]
    snapshot = json.loads(gzip.decompress(objects.values[ref]))
    if damage == "missing": del objects.values[ref]
    elif damage in {"404", "transient", "store_bound"}:
        original = objects.get_bounded
        def get_bounded(key, maximum):
            if key != ref:
                return original(key, maximum)
            if damage == "404":
                raise ClientError({"Error": {"Code": "NoSuchKey"}, "ResponseMetadata": {"HTTPStatusCode": 404}}, "GetObject")
            if damage == "transient": raise TimeoutError("read timeout")
            raise contracts.ArenaContractError("object exceeds source size limit")
        objects.get_bounded = get_bounded
    elif damage == "gzip": objects.values[ref] = b"not gzip"
    elif damage == "json": objects.values[ref] = gzip.compress(b"not JSON")
    elif damage == "compressed_size": objects.values[ref] = b"x" * (training_input.MAX_COMPRESSED_BYTES + 1)
    elif damage == "expanded_size": objects.values[ref] = gzip.compress(b"x" * (training_input.MAX_OBJECT_BYTES + 1))
    elif damage == "wrong_type": objects.values[ref] = "not byte storage"
    elif damage != "valid":
        if damage == "schema": snapshot["schema_version"] = "unknown"
        elif damage == "hash": snapshot["input_sha256"] = contracts.hash_bytes(b"bad")
        elif damage == "privacy": snapshot["privacy"]["redacted"] = True
        elif damage == "private": snapshot["privacy"] = {"redacted": True, "redactions": ["credential"]}
        elif damage == "run": snapshot["run_id"] = "another-run"
        elif damage == "kind": snapshot["kind"] = "score"
        elif damage == "lease": snapshot["lease_generation"] += 1
        else: snapshot["claim_request_hash"] = contracts.hash_bytes(b"other claim")
        objects.values[ref] = gzip.compress(contracts.canonical_json(snapshot).encode())
    unavailable = damage in {"missing", "404", "transient"}
    if damage not in {"valid", "private"}:
        with pytest.raises(training_input.InputCaptureError) as caught:
            training_input.read_input(objects, row["run_id"], row["kind"], row["lease_generation"], row["claim_request_hash"])
        assert (str(caught.value) == "input_unavailable") is unavailable
    example = training_export.assemble_example(**values)
    reasons = example["completeness"]["incomplete_reasons"]
    if damage not in {"valid", "private"} and not unavailable:
        assert "runtime_input_snapshot_invalid" in reasons
    if damage == "private": assert "runtime_input_privacy_nonexact" in reasons
    splits, _ = training_thinking.build([example], training_steps.Rules(production_rounds_only=False))
    records = splits["train"] + splits["eval"]
    assert len(records) == (2 if unavailable or damage == "valid" else 0)
    if unavailable:
        assert all(record["source_refs"][0]["selected_research_evidence"]["runtime_input_availability"] == "unavailable"
                   for record in records)
    assert (training_thinking.selected_research_evidence(example) is not None) is (damage == "valid")


def test_missing_input_plus_invalid_stored_judge_never_grants_absence_exception():
    values, _, _ = retry_values()
    for row in values["runs"][:2]:
        ref = training_input.object_ref(row["run_id"], row["lease_generation"])
        if row["kind"] == "execute": del values["objects"].values[ref]
        else: values["objects"].values[ref] = b"invalid gzip"
    example = training_export.assemble_example(**values)
    assert "runtime_input_snapshot_invalid" in example["completeness"]["incomplete_reasons"]
    splits, _ = training_thinking.build([example], training_steps.Rules(production_rounds_only=False))
    assert splits["train"] + splits["eval"] == []


def test_real_bounded_s3_store_size_rejection_is_invalid_without_reading_body():
    from lab_arena.service import S3ObjectStore
    class Body:
        closed = False
        def read(self, size): raise AssertionError("oversize body must not be read")
        def close(self): self.closed = True
    body = Body()
    class Client:
        def get_object(self, **kwargs):
            return {"ContentLength": training_input.MAX_COMPRESSED_BYTES + 1, "Body": body}
    store = S3ObjectStore("test", client=Client())
    with pytest.raises(training_input.InputCaptureError, match="input_storage_invalid"):
        training_input.read_input(store, "run", "execute", 1, contracts.hash_bytes(b"claim"))
    assert body.closed


@pytest.mark.parametrize("field,value", [("run_id", "invalid/id"), ("lease_generation", 0),
                                        ("claim_request_hash", "invalid")])
def test_invalid_requested_identity_fails_before_storage(field, value):
    class NoRead:
        def get_bounded(self, *args): raise AssertionError("invalid identity must not fetch")
    args = dict(run_id="run", kind="execute", lease_generation=1,
                claim_request_hash=contracts.hash_bytes(b"claim"))
    args[field] = value
    with pytest.raises(training_input.InputCaptureError) as caught:
        training_input.read_input(NoRead(), **args)
    assert str(caught.value) != "input_unavailable"


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("damage", ["valid", "missing", "gzip", "hash", "privacy", "private", "oversize"])
def test_real_local_store_missing_is_unavailable_but_present_invalid_stays_invalid(tmp_path, wrapped, damage):
    from lab_arena import training_eligibility
    from lab_arena.service import LocalObjectStore
    from lab_arena.training_reader import AvailabilityReadView
    values, _, _ = retry_values()
    row = values["runs"][0]
    ref = training_input.object_ref(row["run_id"], row["lease_generation"])
    raw_objects = values["objects"].values
    snapshot = json.loads(gzip.decompress(raw_objects[ref]))
    if damage == "missing": del raw_objects[ref]
    elif damage == "gzip": raw_objects[ref] = b"invalid gzip"
    elif damage == "oversize": raw_objects[ref] = b"x" * (training_input.MAX_COMPRESSED_BYTES + 1)
    elif damage in {"hash", "privacy", "private"}:
        if damage == "hash": snapshot["input_sha256"] = contracts.hash_bytes(b"wrong")
        elif damage == "privacy": snapshot["privacy"]["redacted"] = True
        else: snapshot["privacy"] = {"redacted": True, "redactions": ["credential"]}
        raw_objects[ref] = gzip.compress(contracts.canonical_json(snapshot).encode())
    local = LocalObjectStore(tmp_path / "objects")
    for key, raw in raw_objects.items(): local.put(key, raw)
    values["objects"] = AvailabilityReadView(local) if wrapped else local
    if damage not in {"valid", "private"}:
        with pytest.raises(training_input.InputCaptureError) as caught:
            training_input.read_input(values["objects"], row["run_id"], row["kind"], row["lease_generation"], row["claim_request_hash"])
        assert (str(caught.value) == "input_unavailable") is (damage == "missing")
    example = training_export.assemble_example(**values)
    if damage not in {"valid", "missing", "private"}:
        assert "runtime_input_snapshot_invalid" in example["completeness"]["incomplete_reasons"]
    splits, _ = training_thinking.build([example], training_steps.Rules(production_rounds_only=False))
    records = splits["train"] + splits["eval"]
    assert len(records) == (2 if damage in {"valid", "missing"} else 0)
    assert (training_thinking.selected_research_evidence(example) is not None) is (damage == "valid")
    assessment = training_eligibility.assess_capture(values["submission_row"], {row["run_id"]: example},
        [row["run_id"]], enabled=True, minimum_coverage_percent=80, maximum_missing_reasoning_percent=0)
    assert assessment["eligible"] is (damage == "valid")
    if damage == "missing":
        assert all(record["source_refs"][0]["selected_research_evidence"]["runtime_input_availability"] == "unavailable"
                   for record in records)


@pytest.mark.parametrize("failure", ["404", "timeout", "malformed"])
def test_real_s3_optional_adapter_retains_unavailability_and_invalidity(failure):
    import io
    from lab_arena.service import S3ObjectStore
    class Client:
        gets = 0
        def get_object(self, **kwargs):
            self.gets += 1
            if failure == "404":
                raise ClientError({"Error": {"Code": "NoSuchKey"}, "ResponseMetadata": {"HTTPStatusCode": 404}}, "GetObject")
            if failure == "timeout": raise TimeoutError("storage read timed out")
            return {"ContentLength": 4, "Body": io.BytesIO(b"bad!")}
    client = Client()
    with pytest.raises(training_input.InputCaptureError) as caught:
        training_input.read_input(S3ObjectStore("test", client=client), "run", "execute", 1, contracts.hash_bytes(b"claim"))
    assert (str(caught.value) == "input_unavailable") is (failure != "malformed")
    assert client.gets == 1
