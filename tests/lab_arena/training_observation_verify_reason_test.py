"""A verify refusal names whether the store or the snapshot itself failed."""

import pytest

from gateway.observability.otel_bootstrap import ARENA_DENIAL_CODES
from lab_arena import training_observations
from lab_arena.service import ServiceError
from tests.lab_arena.training_input_test import CLAIM_HASH, TOKEN, service_for
from tests.lab_arena.training_observations_test import observation, payload


DIGEST = "c" * 64


def _stored(tmp_path):
    service, objects = service_for(tmp_path)
    receipt = service.handle_training_observations("r1", TOKEN, payload(observation()))
    return service, objects, receipt["payload_sha256"]


def test_verify_confirms_the_stored_digest(tmp_path):
    service, _objects, digest = _stored(tmp_path)
    assert service.verify_training_observations("r1", TOKEN, digest) == digest


@pytest.mark.parametrize("reason,code", [
    ("observations_unavailable", "training_observations_unavailable"),
    ("size_limit", "training_observations_snapshot_invalid"),
    ("identity_or_hash_invalid", "training_observations_snapshot_invalid"),
    ("schema_payload_mismatch", "training_observations_snapshot_invalid"),
    ("privacy_invalid", "training_observations_snapshot_invalid"),
    ("something_new", "training_observations_unavailable"),
])
def test_verify_names_the_half_that_failed(tmp_path, monkeypatch, reason, code):
    service, _objects, digest = _stored(tmp_path)

    def refuse(*_args, **_kwargs):
        raise training_observations.ObservationError(reason)

    monkeypatch.setattr(training_observations, "read_snapshot", refuse)
    with pytest.raises(ServiceError) as caught:
        service.verify_training_observations("r1", TOKEN, digest)
    assert (caught.value.code, caught.value.status) == (code, 503)
    # Every refusal the sidecar can emit must survive the export allowlist.
    assert caught.value.code in ARENA_DENIAL_CODES


def test_a_store_read_failure_still_reads_as_unavailable(tmp_path):
    service, _objects, digest = _stored(tmp_path)
    stored = tmp_path / "objects" / training_observations.object_ref("r1", 1, CLAIM_HASH)
    stored.write_bytes(b"not gzip at all")
    with pytest.raises(ServiceError) as caught:
        service.verify_training_observations("r1", TOKEN, digest)
    assert caught.value.code == "training_observations_unavailable"
