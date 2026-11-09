"""Ambiguous auth source stays in private audit and out of strict training."""
import json

import pytest

from lab_arena import contracts, training_capture, training_steps
from lab_arena.service import LocalObjectStore
from tests.lab_arena.training_steps_test import fixture, seal

SYNTHETIC = "offline-private-fixture-never-a-valid-credential"
SOURCE = ('token = "' + SYNTHETIC + '"\n'
          'headers = {"Authorization": "Bearer " + token}\n')


def _capture(tmp_path, source, *, serialized=True, secrets=()):
    example = fixture()
    call = example["provider_calls"][0]
    request = call["request"]
    body = request["outbound"]["body"]
    body["messages"][0]["content"] = (contracts.canonical_json({
        "submission_files": [{"path": "agent.py", "content": source}],
    }) if serialized else source)
    request["outbound"]["body_sha256"] = contracts.document_hash(body)
    objects = LocalObjectStore(tmp_path)
    pointer = training_capture.RequestCapture(objects)(request, secrets=secrets)
    call["request_capture"] = pointer
    call["request"] = training_capture.read_request(objects, pointer)
    return example, call, objects


def _rows(example):
    splits, manifest = training_steps.build([seal(example)])
    return splits["train"] + splits["eval"], manifest


@pytest.mark.parametrize("serialized", [False, True])
def test_locally_bound_unknown_value_keeps_private_audit_flagged(tmp_path, serialized):
    _, call, objects = _capture(tmp_path, SOURCE, serialized=serialized)
    pointer = call["request_capture"]
    assert pointer["redacted"] is True
    assert pointer["redactions"] == ["credential"]
    # Unknown source values are retained privately, never asserted to be scrubbed.
    assert SYNTHETIC in json.dumps(training_capture.read_request(objects, pointer))


@pytest.mark.parametrize("serialized", [False, True])
def test_locally_bound_unknown_value_cannot_enter_native_training(tmp_path, serialized):
    example, _, _ = _capture(tmp_path, SOURCE, serialized=serialized)
    rows, manifest = _rows(example)
    assert rows == []
    assert manifest["counts"]["exclusion_reasons"] == {"request_privacy_nonexact": 1}


def test_preexisting_unflagged_capture_is_reassessed_without_audit_rewrite(tmp_path, monkeypatch):
    # Reproduce the retained legacy shape, then use the actual native selector.
    with monkeypatch.context() as patch:
        patch.setattr(training_capture, "sanitize_training_value", lambda value, *_a, **_kw: (value, []))
        example, call, objects = _capture(tmp_path, SOURCE)
    pointer = call["request_capture"]
    retained = training_capture.read_request(objects, pointer)
    assert pointer["redacted"] is False and SYNTHETIC in json.dumps(retained)
    rows, manifest = _rows(example)
    assert rows == []
    assert manifest["counts"]["exclusion_reasons"] == {"request_privacy_nonexact": 1}
    assert training_capture.read_request(objects, pointer) == retained


def test_public_source_without_ambiguous_auth_is_still_exact_and_usable(tmp_path):
    source = 'def company_name(row):\n    return row["company"].strip()\n'
    example, call, _ = _capture(tmp_path, source)
    assert call["request_capture"]["redacted"] is False
    rows, manifest = _rows(example)
    assert len(rows) == 1 and manifest["counts"]["exclusion_reasons"] == {}
    assert json.loads(rows[0]["input"]["messages"][0]["content"])["submission_files"][0]["content"] == source


def test_known_opaque_secret_is_scrubbed_from_private_capture(tmp_path):
    example, call, _ = _capture(tmp_path, SOURCE, secrets=(SYNTHETIC,))
    assert call["request_capture"]["redacted"] is True
    assert SYNTHETIC not in json.dumps(call["request"])
    rows, manifest = _rows(example)
    assert rows == []
    assert manifest["counts"]["exclusion_reasons"] == {"request_privacy_nonexact": 1}
