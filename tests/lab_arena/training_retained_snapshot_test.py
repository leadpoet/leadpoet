"""Retain validated capture without granting a mismatched summary authority."""

import copy
import gzip
import json
from types import SimpleNamespace

import pytest

from lab_arena import (contracts, runner, training_capture, training_dataset, training_eligibility,
                       training_observations, training_steps, training_thinking)
from lab_arena.training_export import assemble_example
from tests.lab_arena.training_export_receipts_test import save, values_with_receipts
from tests.lab_arena.training_export_test import CLAIM_HASH


RUN_ID = "execute:1"
GAP = "worker_observation_summary_invalid"


def summary_for(snapshot):
    return {"status": "stored", "payload_sha256": snapshot["payload_sha256"],
            **{key: snapshot["payload"][key]
               for key in ("observation_count", "local_refusal_count", "complete")}}


def prepared():
    values, payload = values_with_receipts()
    snapshot = save(values, payload)
    resource = values["runs"][0]["result_doc"]["resource_summary"]
    resource["worker_observations"] = summary_for(snapshot)
    return values, snapshot, resource


def assert_unverified(exported, snapshot):
    assert RUN_ID not in exported["worker_observation_artifacts"]
    assert not exported.get("partial_worker_observation_artifacts")
    retained = exported["unverified_worker_observation_artifacts"][RUN_ID]
    assert retained == {**snapshot, "complete": False, "unverified_reasons": [GAP]}
    # The original payload can be complete. The outer audit classification is not.
    assert retained["payload"] == snapshot["payload"]
    assert retained["payload_sha256"] == contracts.document_hash(retained["payload"])
    assert GAP in exported["completeness"]["incomplete_reasons"]
    assert exported["completeness"]["complete"] is False
    call = exported["provider_calls"][0]
    assert call["response_provenance"]["delivery_to_worker"] == "not_proved"
    assert "worker_receipt_ref" not in call["response_provenance"]
    assert "agent_receipt_refs" not in call["response_provenance"]
    for builder in (training_steps.build, training_thinking.build):
        splits, _ = builder([exported])
        assert not splits["train"] and not splits["eval"]
    return retained


def test_both_final_replies_lost_keep_exact_snapshot_without_authoritative_links():
    values, snapshot, resource = prepared()
    lease = {"run_id": RUN_ID, "kind": "execute", "lease_generation": 1,
             "claim_request_hash": CLAIM_HASH}
    state = runner.RunState(lease=lease, lease_token="local-test-token")
    document = snapshot["payload"]
    state.worker_observations.response_receipts = copy.deepcopy(document["response_receipts"])
    state.worker_observations.response_receipt_count = document["response_receipt_count"]
    state.worker_observations.agent_observations = copy.deepcopy(document["agent_observations"])
    state.worker_observations.agent_observation_count = document["agent_observation_count"]
    attempts = []

    def committed_reply_lost(_run_id, _token, payload):
        attempts.append(copy.deepcopy(payload))
        save(values, payload)
        raise TimeoutError("synthetic lost reply after storage")

    resource["worker_observations"] = runner._record_training_observations(
        SimpleNamespace(api=SimpleNamespace(training_observations=committed_reply_lost)),
        lease, state.lease_token, state)
    assert resource["worker_observations"]["status"] == "unavailable"
    assert attempts == [document, document]
    exported = assemble_example(**values)
    retained = assert_unverified(exported, snapshot)
    assert retained["complete"] is False and retained["payload"]["complete"] is True


@pytest.mark.parametrize("verification", ["verified", "legacy", "wrong_digest", "timeout", "same_count_conflict"])
def test_exact_ack_recovery_and_unverified_export_retention_coexist(tmp_path, verification):
    import httpx
    from fastapi.testclient import TestClient
    from lab_arena.api import create_app
    from tests.lab_arena.training_input_test import TOKEN, service_for

    values, snapshot, resource = prepared()
    service, objects = service_for(tmp_path)
    owner = service._store.get_run("r1")
    owner["run_id"] = RUN_ID
    service._store.get_run = lambda run_id: owner if run_id == RUN_ID else None
    # Use the real write-once store for both the HTTP service and exporter.
    for ref, raw in values["objects"].values.items():
        objects.put(ref, raw)
    values["objects"] = objects
    state = runner.RunState(lease={"run_id": RUN_ID}, lease_token=TOKEN)
    document = copy.deepcopy(snapshot["payload"])
    if verification == "same_count_conflict":
        document["agent_observations"][1]["data"]["value"] = {"different": True}
        assert contracts.document_hash(document) != snapshot["payload_sha256"]
    state.worker_observations.response_receipts = document["response_receipts"]
    state.worker_observations.response_receipt_count = document["response_receipt_count"]
    state.worker_observations.agent_observations = document["agent_observations"]
    state.worker_observations.agent_observation_count = document["agent_observation_count"]
    gateway = TestClient(create_app(service))
    methods = []
    route = "/arena/v1/runs/%s/training-observations" % RUN_ID

    def transport(request):
        methods.append(request.method)
        if request.method == "POST":
            response = gateway.post(route, content=request.content, headers=dict(request.headers))
            assert response.status_code == (409 if verification == "same_count_conflict" else 200)
            if response.status_code == 200:
                raise httpx.ReadTimeout("synthetic lost ACK", request=request)
        elif verification == "timeout":
            raise httpx.ReadTimeout("synthetic HEAD timeout", request=request)
        elif verification == "legacy":
            return httpx.Response(204)
        elif verification == "wrong_digest":
            return httpx.Response(204, headers={"ETag": '"' + contracts.hash_bytes(b"different") + '"'})
        else:
            response = gateway.head(route, headers=dict(request.headers))
            assert response.status_code == (412 if verification == "same_count_conflict" else 204)
        return httpx.Response(response.status_code, content=response.content, headers=dict(response.headers))

    with httpx.Client(transport=httpx.MockTransport(transport)) as client:
        api = runner.HttpArenaApiClient("http://localhost", client=client)
        resource["worker_observations"] = runner._record_training_observations(
            SimpleNamespace(api=api), state.lease, TOKEN, state)
    assert methods == ["POST", "POST", "HEAD"]
    exported = assemble_example(**values)
    if verification == "verified":
        assert resource["worker_observations"] == summary_for(snapshot)
        assert exported["worker_observation_artifacts"][RUN_ID] == snapshot
        assert not exported.get("unverified_worker_observation_artifacts")
        assert exported["completeness"]["complete"]
        assert exported["provider_calls"][0]["response_provenance"]["delivery_to_worker"] == "socket_written"
    else:
        assert resource["worker_observations"]["status"] == "unavailable"
        assert_unverified(exported, snapshot)


@pytest.mark.parametrize("damage", ["unavailable", "non_mapping", "status", "hash", "count",
                                    "local_refusal_count", "complete", "same_count_conflict"])
def test_every_summary_mismatch_retains_only_unverified_evidence(damage):
    values, snapshot, resource = prepared()
    summary = resource["worker_observations"]
    if damage == "unavailable":
        summary.update(status="unavailable")
        summary.pop("payload_sha256")
    elif damage == "non_mapping":
        resource["worker_observations"] = "unavailable"
    elif damage == "status":
        summary["status"] = "rejected"
    elif damage == "hash":
        summary["payload_sha256"] = contracts.hash_bytes(b"different")
    elif damage == "count":
        summary["observation_count"] += 1
    elif damage == "local_refusal_count":
        summary["local_refusal_count"] += 1
    elif damage == "complete":
        summary["complete"] = False
    else:
        intended = copy.deepcopy(snapshot["payload"])
        intended["agent_observations"][1]["data"]["value"] = {"different": True}
        assert all(intended[key] == snapshot["payload"][key]
                   for key in ("observation_count", "local_refusal_count", "complete"))
        assert contracts.document_hash(intended) != snapshot["payload_sha256"]
        summary.update(status="unavailable")
        summary.pop("payload_sha256")
    assert_unverified(assemble_example(**values), snapshot)


@pytest.mark.parametrize("damage", ["payload_hash", "identity", "privacy"])
def test_invalid_stored_snapshot_is_not_retained(damage):
    values, snapshot, _ = prepared()
    altered = copy.deepcopy(snapshot)
    if damage == "payload_hash":
        altered["payload_sha256"] = contracts.hash_bytes(b"different")
    elif damage == "identity":
        altered["lease_generation"] += 1
    else:
        altered["payload"]["agent_observations"][1]["data"]["value"] = {"api_key": "synthetic-secret"}
        altered["payload_sha256"] = contracts.document_hash(altered["payload"])
    values["objects"].put(training_observations.object_ref(RUN_ID, 1, CLAIM_HASH),
        gzip.compress(contracts.canonical_json(altered).encode(), mtime=0))
    exported = assemble_example(**values)
    assert RUN_ID not in exported["worker_observation_artifacts"]
    assert RUN_ID not in exported.get("partial_worker_observation_artifacts", {})
    assert RUN_ID not in exported.get("unverified_worker_observation_artifacts", {})
    assert "worker_observations_unavailable" in exported["completeness"]["incomplete_reasons"]


@pytest.mark.parametrize("summary_present", [True, False])
def test_valid_and_historically_absent_summaries_keep_existing_authority(summary_present):
    values, snapshot, resource = prepared()
    if not summary_present:
        del resource["worker_observations"]
    exported = assemble_example(**values)
    assert exported["worker_observation_artifacts"][RUN_ID] == snapshot
    assert not exported.get("partial_worker_observation_artifacts")
    assert not exported.get("unverified_worker_observation_artifacts")
    assert exported["completeness"]["complete"]
    assert exported["provider_calls"][0]["response_provenance"]["delivery_to_worker"] == "socket_written"


def test_retained_snapshot_preserves_sanitized_privacy_flags_and_hashes():
    values, payload = values_with_receipts()
    payload["agent_observations"][1]["data"]["value"] = {"api_key": "synthetic-secret"}
    snapshot = save(values, payload)
    resource = values["runs"][0]["result_doc"]["resource_summary"]
    resource["worker_observations"] = {**summary_for(snapshot), "status": "unavailable"}
    exported = assemble_example(**values)
    retained = assert_unverified(exported, snapshot)
    row = retained["payload"]["agent_observations"][1]
    assert row["privacy"] == {"redacted": True, "redactions": ["credential"]}
    assert "synthetic-secret" not in json.dumps(exported)
    assert exported["privacy"]["privacy_filtered"] is True
    assert exported["privacy"]["exact_replay"] is False


def test_unverified_parse_evidence_keeps_existing_admitted_control_privacy_rules():
    from tests.lab_arena.training_observations_test import observation

    values, _ = values_with_receipts()
    buffer = training_observations.Buffer()
    row = observation(kind="local_parse_error", operation="unknown", action=None, contributes=False)
    row["request"]["admitted_parameters"] = {"raw_frame": {
        "operation_id": "openrouter.chat", "parameters": {
            "model": "synthetic/model", "messages": [{"role": "user", "content": "Synthetic input"}],
            "reasoning": {"effort": "high"}}}}
    buffer.append(row)
    snapshot = save(values, buffer.document())
    values["runs"][0]["result_doc"]["resource_summary"]["worker_observations"] = {
        **summary_for(snapshot), "status": "unavailable"}
    exported = assemble_example(**values)
    assert_unverified(exported, snapshot)
    sanitized, redactions = training_capture.sanitize_training_value(exported)
    assert sanitized == exported and not redactions


def test_readable_keeps_unverified_reason_and_every_row_in_original_order():
    values, snapshot, resource = prepared()
    resource["worker_observations"]["status"] = "unavailable"
    exported = assemble_example(**values)
    original_hash = exported["example_hash"]
    readable = training_thinking.readable_trace(exported)
    assert readable["worker_observation_paths"] == []
    assert readable["partial_worker_observation_paths"] == []
    assert readable["unverified_worker_observation_paths"] == [["unverified_worker_observation_artifacts", RUN_ID]]
    events = readable["observations"]
    assert events[0]["kind"] == "unverified_snapshot"
    assert events[0]["value"] == [GAP]
    expected = [(collection, index, row)
                for collection in ("observations", "agent_observations", "response_receipts")
                for index, row in enumerate(snapshot["payload"][collection])]
    assert len(events) == len(expected) + 1
    for event, (collection, index, row) in zip(events[1:], expected):
        assert event["source_path"] == ["unverified_worker_observation_artifacts", RUN_ID, "payload", collection, index]
        assert event["value"] == row
    for event in events:
        value = exported
        for key in event["source_path"]:
            value = value[key]
        assert event["value"] == value
        assert event["value_sha256"] == contracts.document_hash(value)
    markdown = training_thinking.readable_trace_markdown(readable)
    assert "unverified_snapshot" in markdown and GAP in markdown
    assert exported["example_hash"] == original_hash
    assert exported["example_hash"] == contracts.document_hash({k: v for k, v in exported.items() if k != "example_hash"})


@pytest.mark.parametrize("role", ["selected", "predecessor", "judge"])
@pytest.mark.parametrize("optional_selected_capture", [False, True])
def test_retention_preserves_admission_for_every_related_run_role(role, optional_selected_capture):
    from tests.lab_arena.training_optional_capture_test import summary
    from tests.lab_arena.training_retry_snapshot_test import retry_values, save_snapshot

    values, predecessor, _ = retry_values(paid=False)
    selected = values["runs"][0]
    judge = next(run for run in values["runs"] if run["kind"] == "score")
    if role == "judge":
        # A judge-only summary gap can coexist with admitted selected thinking.
        values["runs"].remove(predecessor)
    if optional_selected_capture:
        snapshot = training_observations.read_snapshot(values["objects"], selected["run_id"],
            "execute", selected["lease_generation"], selected["claim_request_hash"])
        document = copy.deepcopy(snapshot["payload"])
        document["agent_observations"] = [summary()]
        document["agent_observation_count"] = 1
        document.update(complete=False, failures=[training_observations.OPTIONAL_CAPTURE_INCOMPLETE])
        save_snapshot(values, selected, document)
    target = {"selected": selected, "predecessor": predecessor, "judge": judge}[role]
    if role == "judge":
        save_snapshot(values, judge, training_observations.Buffer().document())
    snapshot = training_observations.read_snapshot(values["objects"], target["run_id"],
        target["kind"], target["lease_generation"], target["claim_request_hash"])
    target.setdefault("result_doc", {})
    if target["result_doc"] is None:
        target["result_doc"] = {}
    resource = target["result_doc"].setdefault("resource_summary", {})
    resource["worker_observations"] = {**summary_for(snapshot), "status": "unavailable",
        "payload_sha256": contracts.hash_bytes(b"conflicting completion summary")}
    after = assemble_example(**values)
    assert target["run_id"] in after["unverified_worker_observation_artifacts"]
    assert target["run_id"] not in after["worker_observation_artifacts"]
    # The old exporter omitted this object entirely. All pre-existing raw gaps,
    # provider calls and authoritative artifacts are unchanged by retention.
    before = copy.deepcopy(after)
    del before["unverified_worker_observation_artifacts"]
    before["example_hash"] = contracts.document_hash({key: value for key, value in before.items()
                                                       if key != "example_hash"})
    options = {"enabled": True, "minimum_coverage_percent": 80,
               "maximum_missing_reasoning_percent": after["configuration"]["training_maximum_missing_reasoning_percent"]}
    old_gate = training_eligibility.assess_capture(values["submission_row"], {selected["run_id"]: before},
                                                  [selected["run_id"]], **options)
    new_gate = training_eligibility.assess_capture(values["submission_row"], {selected["run_id"]: after},
                                                  [selected["run_id"]], **options)
    assert new_gate == old_gate
    for builder in (training_steps.build, training_thinking.build):
        old_splits, _ = builder([before])
        new_splits, _ = builder([after])
        old_ids = {row["candidate_id"] for split in ("train", "eval") for row in old_splits[split]}
        new_ids = {row["candidate_id"] for split in ("train", "eval") for row in new_splits[split]}
        assert new_ids == old_ids
        if role == "judge" and builder is training_thinking.build:
            assert len(new_ids) == 2 and new_gate["eligible"]
    old_labels, _ = training_dataset.build([before])
    new_labels, _ = training_dataset.build([after])
    assert [(row["input"], row["label"]) for split in ("train", "eval") for row in new_labels[split]] == [
        (row["input"], row["label"]) for split in ("train", "eval") for row in old_labels[split]]
