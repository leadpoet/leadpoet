"""Storage availability is distinct from training evidence integrity."""

from __future__ import annotations

import copy
import io
import json
from types import SimpleNamespace

import pytest
from botocore.exceptions import ClientError, ReadTimeoutError

from lab_arena import contracts, training_eligibility, training_reader
from lab_arena.service import ArenaService, LocalObjectStore, S3ObjectStore, _PublicationReads
from tests.lab_arena.training_eligibility_test import (
    _unsettled_selected_call, assess_ten, refresh_response_links,
    selected_privacy_cascade_example, submission, ten_selected_examples,
)
from tests.lab_arena.training_steps_test import seal


REF = "arena/round/outputs/run.json"


class S3Client:
    def __init__(self, faults=()):
        self.faults = list(faults)
        self.calls = 0

    def get_object(self, *, Bucket, Key):
        self.calls += 1
        if self.faults:
            raise self.faults.pop(0)
        data = b"saved output"
        return {"Body": io.BytesIO(data), "ContentLength": len(data)}


def s3_error(code, status):
    return ClientError({"Error": {"Code": code},
                        "ResponseMetadata": {"HTTPStatusCode": status}}, "GetObject")


def test_local_and_s3_missing_are_distinct_from_verified_fallback(tmp_path):
    local = training_reader.AvailabilityReadView(LocalObjectStore(tmp_path))
    with pytest.raises(FileNotFoundError):
        local.get_bounded(REF, 100)
    assert local.unavailable == {}
    assert local.get_optional_bounded(REF, 100) is None
    assert local.unavailable == {}

    client = S3Client([s3_error("NoSuchKey", 404), s3_error("NoSuchKey", 404)])
    remote = training_reader.AvailabilityReadView(S3ObjectStore("test", client=client))
    with pytest.raises(FileNotFoundError):
        remote.get_bounded(REF, 100)
    assert remote.unavailable == {}
    assert remote.get_optional_bounded(REF, 100) is None
    assert remote.unavailable == {}
    assert client.calls == 2


@pytest.mark.parametrize("fault", [
    ReadTimeoutError(endpoint_url="https://objects.example.test"),
    s3_error("SlowDown", 503),
])
def test_transient_read_retries_once_and_recovers(fault):
    client = S3Client([fault])
    view = training_reader.AvailabilityReadView(S3ObjectStore("test", client=client))
    assert view.get_bounded(REF, 100) == b"saved output"
    assert client.calls == 2
    assert view.unavailable == {}


def test_persistent_transient_and_untrusted_s3_errors_stay_distinct():
    client = S3Client([ReadTimeoutError(endpoint_url="https://objects.example.test")
                       for _ in range(2)])
    view = training_reader.AvailabilityReadView(S3ObjectStore("test", client=client))
    with pytest.raises(ReadTimeoutError):
        view.get_bounded(REF, 100)
    assert view.unavailable == {}
    assert client.calls == 2

    client = S3Client([s3_error("AccessDenied", 403)])
    view = training_reader.AvailabilityReadView(S3ObjectStore("test", client=client))
    with pytest.raises(ClientError):
        view.get_bounded(REF, 100)
    assert view.unavailable == {}
    assert client.calls == 1


def test_cached_exact_bytes_preserve_normal_export_proof(tmp_path):
    from lab_arena.training_export import assemble_example
    from tests.lab_arena.training_export_test import fixture

    values = fixture()
    ref = values["runs"][0]["output_ref"]
    raw_document = json.loads(values["objects"].values[ref])
    raw_document["companies"][0]["company_name"] = "  Acme  "
    values["objects"].values[ref] = contracts.canonical_json(raw_document).encode()
    values["runs"][0]["output_hash"] = contracts.document_hash(raw_document)
    objects = LocalObjectStore(tmp_path)
    for object_ref, raw in values["objects"].values.items():
        objects.put(object_ref, raw)
    publication = _PublicationReads(
        SimpleNamespace(_objects=objects), values["round_row"], values["runs"])
    publication.output(ref)
    raw = values["objects"].values[ref]
    assert publication.raw_outputs_by_ref[ref] == raw
    # The first publication read succeeded. Only the later training read
    # sees an unavailable immutable object.
    (tmp_path / ref).unlink()
    view = training_reader.AvailabilityReadView(objects, {ref: publication.raw_outputs_by_ref[ref]})
    example = assemble_example(**{**values, "objects": view})
    assert example["completeness"]["complete"] is True
    assert view.unavailable == {ref: "missing"}
    assert example["output_identity"]["document_hash"] == contracts.document_hash(json.loads(raw))
    assert example["output_identity"]["document_hash"] != contracts.document_hash(example["output"])

    bad_values = copy.deepcopy({key: value for key, value in values.items() if key != "objects"})
    bad_values["runs"][0]["output_hash"] = contracts.document_hash({"other": "output"})
    bad = assemble_example(**{**bad_values, "objects": training_reader.AvailabilityReadView(
        objects, {ref: publication.raw_outputs_by_ref[ref]})})
    assert "execution_output_hash_mismatch" in bad["completeness"]["incomplete_reasons"]


def _cached_example():
    examples = ten_selected_examples()
    example = copy.deepcopy(examples["run-9"])
    ref = "arena/%s/outputs/run-9.json" % example["round_id"]
    example["output_identity"] = {
        "ref": ref, "document_hash": contracts.document_hash(example["output"])}
    examples["run-9"] = seal(example)
    return examples, ref


def _capture(examples, ref, percent=80):
    return training_eligibility.assess_capture(
        submission(), examples, list(examples), enabled=True,
        minimum_coverage_percent=percent, unavailable_outputs={"run-9": ref})


def test_verified_output_read_gap_consumes_one_icp_after_normal_proof():
    examples, ref = _cached_example()
    assert assess_ten(examples)["eligible"] is True
    verdict = _capture(examples, ref)
    assert verdict["eligible"] is True
    assert verdict["counts"]["complete_runs"] == 9
    assert verdict["counts"]["incomplete_runs"] == 1
    assert _capture(examples, ref, 100)["eligible"] is False

    for mutate in (
        lambda row: row["source"].update(sha256="sha256:" + "f" * 64),
        lambda row: row["provider_calls"][0]["request"].update(request_hash="wrong"),
        lambda row: row["provider_calls"][0]["response_receipts"][0].update(response_sha256="wrong"),
        lambda row: row["provider_calls"][0]["response_provenance"].update(agent_receipt="not_reported"),
    ):
        bad = copy.deepcopy(examples["run-9"])
        mutate(bad)
        examples["run-9"] = seal(bad)
        assert _capture(examples, ref)["eligible"] is False
    examples["run-9"] = _cached_example()[0]["run-9"]
    assert _capture(examples, ref.replace("run-9", "wrong-run"))["eligible"] is False


@pytest.mark.parametrize("fault", ["missing", "timeout"])
def test_service_uses_only_verified_publication_bytes(monkeypatch, tmp_path, fault):
    from lab_arena.service import ArenaService

    examples, ref = _cached_example()
    raw = contracts.canonical_json(examples["run-9"]["output"]).encode()
    parsed_hash = contracts.document_hash(json.loads(raw))
    if fault == "missing":
        objects = LocalObjectStore(tmp_path)
        expected_fault = FileNotFoundError
        client = None
    else:
        client = S3Client([ReadTimeoutError(endpoint_url="https://objects.example.test")
                           for _ in range(2)])
        objects = S3ObjectStore("test", client=client)
        expected_fault = ReadTimeoutError
    selected = {position: {"run_id": run_id, "output_hash": parsed_hash}
                for position, run_id in enumerate(examples)}
    row = {"round_id": examples["run-9"]["round_id"], "configuration_doc": {
        "stage_1_icp_count": 5, "stage_2_icp_count": 5,
        "training_minimum_icp_coverage_percent": 80}}
    obj = SimpleNamespace(
        _store=SimpleNamespace(get_submission=lambda _: submission()), _objects=objects,
        _is_daily_baseline=lambda *_: False,
        _selected_accepted_execution_runs=lambda *_: selected)

    def export(_store, read_view, run_id):
        if run_id == "run-9":
            assert read_view.get_bounded(ref, 100) == raw
        return examples[run_id]

    monkeypatch.setattr(training_reader, "export_run", export)
    publication = SimpleNamespace(outputs_by_ref={ref: json.loads(raw)},
                                  raw_outputs_by_ref={ref: raw})
    def check():
        return ArenaService._submission_training_eligibility(
            obj, row, "sub", [], publication_reads=publication)

    first = check()
    assert first["eligible"] is True
    assert first["capture"]["counts"]["complete_runs"] == 9
    if client is not None:
        assert client.calls >= 2
    publication.raw_outputs_by_ref.clear()
    if client is not None:
        client.faults = [ReadTimeoutError(endpoint_url="https://objects.example.test")
                         for _ in range(2)]
    # No fallback: a real exporter cannot claim the verified output. The
    # stand-in checks the same read and lets the outer handler fail closed.
    def no_fallback(_store, read_view, run_id):
        if run_id == "run-9":
            with pytest.raises(expected_fault):
                read_view.get_bounded(ref, 100)
            raise FileNotFoundError("whole export unavailable")
        return examples[run_id]
    monkeypatch.setattr(training_reader, "export_run", no_fallback)
    assert check()["eligible"] is False


@pytest.mark.parametrize("pattern", [
    ("empty", "empty", "empty"),
    ("truncated", "truncated", "truncated"),
    ("truncated", "empty", "captured"),
    ("unproved", "captured", "captured"),
])
def test_cached_output_gap_keeps_thinking_proof(pattern):
    examples, ref = _cached_example()
    example = examples["run-9"]
    for index, state in enumerate(pattern):
        response = example["provider_calls"][index]["response"]
        message = response["choices"][0]["message"]
        if state == "empty":
            for field in ("reasoning", "reasoning_content", "reasoning_details"):
                message.pop(field, None)
        elif state == "truncated":
            response["choices"][0]["finish_reason"] = "length"
        elif state == "unproved":
            message["reasoning_content"] = "Unapproved second text."
        refresh_response_links(example, index)
    examples["run-9"] = seal(example)
    verdict = _capture(examples, ref)
    assert verdict["eligible"] is (pattern[0] != "unproved"), verdict


@pytest.mark.parametrize("gap", ["privacy", "cost", "cost_and_privacy"])
def test_cached_output_gap_composes_with_existing_soft_proof(gap):
    examples = ten_selected_examples()
    if gap == "privacy":
        examples.pop("run-9")
        example = selected_privacy_cascade_example()
        examples["run"] = example
    else:
        example = _unsettled_selected_call(
            examples["run-9"], privacy=gap == "cost_and_privacy")
    run_id = example["run_id"]
    ref = "arena/%s/outputs/%s.json" % (example["round_id"], run_id)
    example["output_identity"]["ref"] = ref
    seal(example)
    assert assess_ten(examples)["eligible"] is True
    result = training_eligibility.assess_capture(
        submission(), examples, list(examples), enabled=True,
        minimum_coverage_percent=80, unavailable_outputs={run_id: ref})
    assert result["eligible"] is True, result
    assert result["counts"]["complete_runs"] == 9
