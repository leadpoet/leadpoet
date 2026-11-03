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
    assert example["output_identity"]["normalized_document_hash"] == contracts.document_hash(example["output"])

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
        lambda row: row["provider_calls"][1]["request"].update(request_hash="wrong"),
        lambda row: row["provider_calls"][1]["response_receipts"][0].update(response_sha256="wrong"),
        lambda row: row["provider_calls"][1]["response_provenance"].update(agent_receipt="not_reported"),
    ):
        bad = copy.deepcopy(examples["run-9"])
        mutate(bad)
        examples["run-9"] = seal(bad)
        rejected = _capture(examples, ref)
        assert rejected["eligible"] is True
        assert rejected["counts"]["complete_runs"] == 9
        assert rejected["counts"]["incomplete_runs"] == 1
        assert {row["run_id"] for row in rejected["counts"]["rejected_runs"]} == {"run-9"}
    examples["run-9"] = _cached_example()[0]["run-9"]
    wrong_ref = _capture(examples, ref.replace("run-9", "wrong-run"))
    assert wrong_ref["eligible"] is True
    assert wrong_ref["counts"]["complete_runs"] == 9
    assert {row["run_id"] for row in wrong_ref["counts"]["rejected_runs"]} == {"run-9"}


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
    result = check()
    assert result["eligible"] is True
    assert result["capture"]["counts"]["complete_runs"] == 9


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
    assert verdict["eligible"] is True, verdict
    assert verdict["counts"]["complete_runs"] == 9
    assert verdict["counts"]["incomplete_runs"] == 1




@pytest.mark.parametrize("artifact", ["source", "benchmark", "judge", "input", "judge_input"])
@pytest.mark.parametrize("tamper", [False, True])
@pytest.mark.parametrize("normalized", [False, True])
def test_real_s3_export_tracks_exhausted_artifact_timeout(artifact, tamper, normalized):
    from lab_arena.training_input import object_ref
    from tests.lab_arena.training_export_test import fixture

    values = fixture()
    judge = values["runs"][1]
    old_ref = judge["output_ref"]
    judge["output_ref"] = "arena/%s/outputs/%s.json" % (judge["round_id"], judge["run_id"])
    values["objects"].values[judge["output_ref"]] = values["objects"].values.pop(old_ref)
    refs = {"source": values["submission_row"]["source_ref"],
            "benchmark": values["round_row"]["benchmark_ref"],
            "judge": values["runs"][1]["output_ref"],
            "input": object_ref(values["run_id"], 1),
            "judge_input": object_ref(judge["run_id"], 1)}
    failed_ref = refs[artifact]
    execution = values["runs"][0]
    output_ref = execution["output_ref"]
    output = json.loads(values["objects"].values[output_ref])
    if normalized:
        output["companies"][0]["company_name"] = "  Acme  "
        values["objects"].values[output_ref] = contracts.canonical_json(output).encode()
    execution["output_hash"] = contracts.document_hash(output)
    if tamper:
        if artifact == "judge":
            # Output rejection would skip the judge read. Corrupt a different
            # retained input so both failures are actually observed.
            values["objects"].values[refs["input"]] = b"invalid compressed input"
        else:
            output["companies"][0]["company_name"] = "Tampered company"
            values["objects"].values[output_ref] = contracts.canonical_json(output).encode()

    class FixtureS3Client:
        def get_object(self, *, Bucket, Key):
            if Key == failed_ref:
                raise ReadTimeoutError(endpoint_url="https://objects.example.test")
            if Key not in values["objects"].values:
                raise s3_error("NoSuchKey", 404)
            raw = values["objects"].values[Key]
            return {"Body": io.BytesIO(raw), "ContentLength": len(raw)}

    view = training_reader.AvailabilityReadView(S3ObjectStore("test", client=FixtureS3Client()))
    store = SimpleNamespace(
        get_run=lambda run_id: next(row for row in values["runs"] if row["run_id"] == run_id),
        get_round=lambda _: values["round_row"], get_submission=lambda _: values["submission_row"],
        list_runs=lambda _: values["runs"], list_ledger=lambda **_: [],
        icp_cost_eligibility=lambda **_: {"eligible": True},
    )
    example = training_reader.export_run(store, view, values["run_id"])
    assert getattr(view, "transient_failures", {}) == {failed_ref: "transient"}
    kind = {"judge": "judge_output"}.get(artifact, artifact)
    assert view.artifact_refs[kind] == failed_ref
    expected = {"source": "source_unverified", "benchmark": "frozen_icp_unavailable",
                "judge": "judge_evidence_unverified", "input": "runtime_input_snapshot_missing",
                "judge_input": "runtime_input_snapshot_missing"}
    if not tamper:
        identity = example["output_identity"]
        assert identity["document_hash"] == execution["output_hash"]
        assert identity["normalized_document_hash"] == contracts.document_hash(example["output"])
        assert (identity["document_hash"] != identity["normalized_document_hash"]) is normalized
        assert expected[artifact] in example["completeness"]["incomplete_reasons"]
    else:
        reason = ("runtime_input_snapshot_missing" if artifact == "judge"
                  else "execution_output_hash_mismatch")
        assert reason in example["completeness"]["incomplete_reasons"]


def test_real_s3_capture_retries_same_bytes_after_committed_write_timeout(monkeypatch):
    from lab_arena import training_capture

    pauses = []
    monkeypatch.setattr(training_capture, "time", SimpleNamespace(sleep=pauses.append), raising=False)

    class PutClient:
        def __init__(self):
            self.values = {}
            self.puts = []

        def put_object(self, *, Bucket, Key, Body, **kwargs):
            self.puts.append((Key, Body))
            assert kwargs["IfNoneMatch"] == "*"
            if Key in self.values:
                raise s3_error("PreconditionFailed", 412)
            self.values[Key] = Body
            raise ReadTimeoutError(endpoint_url="https://objects.example.test")

        def get_object(self, *, Bucket, Key):
            return {"Body": io.BytesIO(self.values[Key])}

    client = PutClient()
    objects = S3ObjectStore("test", client=client)
    pointer = training_capture.RequestCapture(objects)({"safe": "request", "api_key": "private-key"})
    assert pointer["status"] == "stored"
    assert len(client.puts) == 2 and client.puts[0] == client.puts[1]
    assert pauses == [0.05]
    assert pointer["redactions"] == ["credential"]
    assert "private-key" not in json.dumps(training_capture.read_request(objects, pointer))


@pytest.mark.parametrize("fault,attempts", [
    (ReadTimeoutError(endpoint_url="https://objects.example.test"), 2),
    (s3_error("AccessDenied", 403), 1),
])
def test_capture_write_exhaustion_and_denial_stay_explicit(monkeypatch, fault, attempts):
    from lab_arena import training_capture

    pauses = []
    monkeypatch.setattr(training_capture, "time", SimpleNamespace(sleep=pauses.append), raising=False)

    class PutClient:
        calls = 0

        def put_object(self, **kwargs):
            self.calls += 1
            raise fault

    client = PutClient()
    result = training_capture.RequestCapture(S3ObjectStore("test", client=client))({"safe": "request"})
    assert result == training_capture.unavailable()
    assert client.calls == attempts
    assert pauses == ([0.05] if attempts == 2 else [])


@pytest.mark.parametrize("recovers", [False, True])
def test_streaming_body_error_is_retried_and_tracked(monkeypatch, recovers):
    from botocore.exceptions import ResponseStreamingError

    pauses = []
    monkeypatch.setattr(training_reader, "time", SimpleNamespace(sleep=pauses.append))

    class StreamingClient:
        calls = 0
        closed = 0

        def get_object(self, **kwargs):
            self.calls += 1
            fail = not recovers or self.calls == 1
            client = self

            class Body:
                def read(self, _limit):
                    if fail:
                        raise ResponseStreamingError(error="connection interrupted")
                    return b"saved output"

                def close(self):
                    client.closed += 1

            return {"Body": Body(), "ContentLength": 12}

    client = StreamingClient()
    view = training_reader.AvailabilityReadView(S3ObjectStore("test", client=client))
    if recovers:
        assert view.get_bounded(REF, 100) == b"saved output"
        assert view.transient_failures == {}
    else:
        with pytest.raises(ResponseStreamingError):
            view.get_bounded(REF, 100)
        assert view.transient_failures == {REF: "transient"}
    assert client.calls == client.closed == 2
    assert pauses == [0.05]


def test_transient_marker_clears_after_a_later_successful_read():
    client = S3Client([ReadTimeoutError(endpoint_url="https://objects.example.test")
                       for _ in range(2)])
    view = training_reader.AvailabilityReadView(S3ObjectStore("test", client=client))
    with pytest.raises(ReadTimeoutError):
        view.get_bounded(REF, 100)
    assert view.transient_failures == {REF: "transient"}
    assert view.get_bounded(REF, 100) == b"saved output"
    assert view.transient_failures == {}


@pytest.mark.parametrize("fault", ["timeout", "missing", "denied"])
def test_service_passes_only_completed_export_transient_refs(monkeypatch, fault):
    from lab_arena import training_input
    from tests.lab_arena.training_export_test import fixture

    values = fixture()
    row = values["round_row"]
    row["configuration_doc"].update(stage_1_icp_count=1, stage_2_icp_count=1)
    benchmark = json.loads(values["objects"].values[row["benchmark_ref"]])
    benchmark["icps"] = benchmark["icps"][:2]
    values["objects"].values[row["benchmark_ref"]] = contracts.canonical_json(benchmark).encode()
    original_execute, original_judge = values["runs"]
    second_execute = {**copy.deepcopy(original_execute), "run_id": "execute:2",
                      "assignment_id": "execute-second", "icp_position": 1,
                      "output_ref": "arena/%s/outputs/execute:2.json" % row["round_id"]}
    second_judge = {**copy.deepcopy(original_judge), "run_id": "judge:2",
                    "assignment_id": "judge-second", "icp_position": 1,
                    "scored_run_id": second_execute["run_id"], "output_ref": "second-judge-output"}
    for original, second in ((original_execute, second_execute), (original_judge, second_judge)):
        values["objects"].values[second["output_ref"]] = values["objects"].values[original["output_ref"]]
        snapshot = training_input.read_input(values["objects"], original["run_id"], original["kind"],
                                             1, original["claim_request_hash"])
        document = snapshot["input"]
        if second["kind"] == "score":
            document["scored_run_id"] = second_execute["run_id"]
        _, raw = training_input.build_snapshot(second["run_id"], second["kind"], document,
            lease_generation=1, claim_request_hash=second["claim_request_hash"])
        values["objects"].values[training_input.object_ref(second["run_id"], 1)] = raw
    values["runs"].extend((second_execute, second_judge))
    target = training_input.object_ref(original_execute["run_id"], 1)

    class Client:
        def get_object(self, *, Bucket, Key):
            if Key == target:
                if fault == "timeout":
                    raise ReadTimeoutError(endpoint_url="https://objects.example.test")
                raise s3_error("NoSuchKey" if fault == "missing" else "AccessDenied",
                               404 if fault == "missing" else 403)
            if Key not in values["objects"].values:
                raise s3_error("NoSuchKey", 404)
            raw = values["objects"].values[Key]
            return {"Body": io.BytesIO(raw), "ContentLength": len(raw)}

    store = SimpleNamespace(
        get_run=lambda identity: next(run for run in values["runs"] if run["run_id"] == identity),
        get_round=lambda _: row, get_submission=lambda _: values["submission_row"],
        list_runs=lambda _: values["runs"], list_ledger=lambda **_: [],
        icp_cost_eligibility=lambda **_: {"eligible": True},
    )
    observed = []
    assess = training_eligibility.assess_capture

    def record(*args, **kwargs):
        observed.append(kwargs["unavailable_artifacts"])
        return assess(*args, **kwargs)

    monkeypatch.setattr(training_eligibility, "assess_capture", record)
    service = SimpleNamespace(
        _store=store, _objects=S3ObjectStore("test", client=Client()),
        _is_daily_baseline=lambda *_: False,
        _selected_accepted_execution_runs=ArenaService._selected_accepted_execution_runs,
    )
    outcome = ArenaService._submission_training_eligibility(
        service, row, values["submission_row"]["submission_id"], [original_execute, second_execute])
    expected = {original_execute["run_id"]: {"input": target}} if fault == "timeout" else {}
    assert observed == [expected]
    # This no-provider fixture proves handoff, not full-thinking admission.
    assert outcome["eligible"] is False
    assert "capture" in outcome and "training_evidence_check_failed" not in outcome["reasons"]


def test_service_eligibility_failure_logs_only_safe_diagnostics(monkeypatch, caplog):
    from lab_arena import service as service_module

    examples = ten_selected_examples()
    row = {"round_id": "private-round-identity", "configuration_doc": {
        "stage_1_icp_count": 5, "stage_2_icp_count": 5}}
    identity = "private-submission-identity"
    service = SimpleNamespace(
        _store=SimpleNamespace(get_submission=lambda _: submission()), _objects=object(),
        _is_daily_baseline=lambda *_: False,
        _selected_accepted_execution_runs=lambda *_: {
            position: {"run_id": run_id} for position, run_id in enumerate(examples)},
    )

    def fail(*_args):
        raise RuntimeError("private-provider-key https://private-endpoint.example/object")

    monkeypatch.setattr(training_reader, "export_run", fail)
    with caplog.at_level("WARNING", logger=service_module.__name__):
        outcome = ArenaService._submission_training_eligibility(service, row, identity, [])
    assert outcome["eligible"] is False
    assert "selected_run_export_invalid" in outcome["reasons"]
    message = caplog.text
    diagnostics = message + repr(outcome)
    for private in (row["round_id"], identity, "private-provider-key", "private-endpoint"):
        assert private not in diagnostics
    assert all(record.exc_info is None for record in caplog.records)
