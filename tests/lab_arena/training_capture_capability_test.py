"""Rolling capture releases preserve bytes while negotiating one failure tag."""

from copy import deepcopy
from datetime import datetime, timezone
from functools import lru_cache
import base64
import gzip
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace
import types

import pytest

from lab_arena import runner, service as service_module, training_observations as observations
from tests.lab_arena.claim_recovery_test import claim_service, original_claim, recovery_claim
from tests.lab_arena.training_input_test import CLAIM_HASH, TOKEN, service_for
from tests.lab_arena.training_observation_journal_test import LEASE
from tests.lab_arena.training_observations_test import IDENTITY, observation
from tests.lab_arena.training_optional_capture_test import captured_timeout, refresh, summary
from lab_arena import training_eligibility
from tests.lab_arena.training_eligibility_test import submission


CAPABILITY = "training_observation_optional_failure_tag"
OPTIONAL = "optional_capture_incomplete"
ABSENT = object()


@lru_cache(maxsize=1)
def reader_635():
    """Exact pre-capability production reader; no Git history needed in CI."""
    path = Path(__file__).parent / "fixtures/training_observations_635ba70c2.py.gz"
    source = gzip.decompress(path.read_bytes())
    assert hashlib.sha256(source).hexdigest() == (
        "701557d0ba99ed8fcbe86c33e3847477400262717a8ad5eecea267ee24a20fd8")
    module = types.ModuleType("training_observations_reader_635_fixture")
    module.__file__ = str(path)
    sys.modules[module.__name__] = module
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


def capture_api(service, monkeypatch, *, legacy_reader, uploads, lose_first_ack=False):
    """Exercise exact 635 and current readers through real capture handlers."""
    def call(method, run_id, token, document):
        with monkeypatch.context() as scope:
            if legacy_reader:
                scope.setattr(service_module, "training_observations", reader_635())
            return method(run_id, token, document)

    def segment(run_id, token, document):
        uploads.append(deepcopy(document))
        result = call(service.handle_training_observation_segment, run_id, token, document)
        if lose_first_ack and len(uploads) == 1:
            raise TimeoutError("stored reply was lost")
        return result

    return SimpleNamespace(
        training_observation_segment=segment,
        training_observations=lambda run_id, token, document: call(
            service.handle_training_observations, run_id, token, document),
    )


def state_for(capability=ABSENT, *, old_producer=False, required_gap=False, kind="execute"):
    lease = dict(LEASE, kind=kind)
    if capability is not ABSENT:
        lease[CAPABILITY] = capability
    state = runner.RunState(lease=lease, lease_token=TOKEN)
    if old_producer:
        state.worker_observations = reader_635().Buffer()
    buffer = state.worker_observations
    row = observation(identity=IDENTITY)
    buffer.append(row)
    assert len(buffer.observations) == 1
    response = row["outcome"]["response"]
    body = base64.b64decode(response["body_b64"])
    assert buffer.append_response_receipt({
        "ordinal": 0, "action_sequence": 0, "operation_id": "openrouter.chat",
        "call_identity": IDENTITY, "protocol": "frame", "status": response["status"],
        "body_sha256": "sha256:" + hashlib.sha256(body).hexdigest(), "body_bytes": len(body),
        "write_state": "socket_written", "socket_written_at": "2026-10-27T03:40:00Z",
    })
    assert buffer.append_agent_observation(summary()) == 0
    assert buffer.failures == {"capture_incomplete" if old_producer else OPTIONAL}
    if required_gap:
        buffer.failures.add("capture_incomplete")
    return state


@pytest.mark.parametrize("old_producer", [False, True])
@pytest.mark.parametrize("legacy_reader", [False, True])
@pytest.mark.parametrize("kind", ["execute", "score"])
def test_all_producer_reader_pairs_store_chained_journal_and_final(
    tmp_path, monkeypatch, old_producer, legacy_reader, kind,
):
    service, objects = service_for(tmp_path, kind=kind)
    state = state_for(ABSENT if legacy_reader else True, old_producer=old_producer, kind=kind)
    original = deepcopy(state.worker_observations.document())
    uploads = []
    api = capture_api(service, monkeypatch, legacy_reader=legacy_reader, uploads=uploads,
                      lose_first_ack=True)
    journal = runner._ObservationJournal(api, state.lease, TOKEN, state)
    assert journal._prepare()
    first = deepcopy(journal.pending)
    assert not journal._send_pending()
    assert journal.pending == first
    assert journal._send_pending()
    assert uploads[0] == uploads[1]
    assert not journal._prepare()  # Wire projection must not create empty repeat segments.
    state.worker_observations.append(observation(ordinal=1, action=1))
    assert journal._prepare() and journal._send_pending()
    assert uploads[-1]["previous_sha256"] == first[1]
    segments, partial, failure = observations.read_journal_prefix(
        objects, "r1", kind, 1, CLAIM_HASH)
    assert failure is None and len(segments) == 2
    expected_tag = "capture_incomplete" if old_producer or legacy_reader else OPTIONAL
    assert all(segment["payload"]["failures"] == [expected_tag] for segment in segments)
    assert partial["response_receipts"] == original["response_receipts"]
    assert partial["agent_observations"] == original["agent_observations"]
    result = runner._record_training_observations(SimpleNamespace(api=api), state.lease,
                                                  TOKEN, state)
    assert result["status"] == "stored" and result["complete"] is False
    saved = observations.read_snapshot(objects, "r1", kind, 1, CLAIM_HASH)["payload"]
    assert saved["failures"] == [expected_tag]
    assert saved["observations"] == state.worker_observations.document()["observations"]
    assert saved["response_receipts"] == original["response_receipts"]
    assert saved["agent_observations"] == original["agent_observations"]
    assert state.worker_observations.failures == set(original["failures"])


@pytest.mark.parametrize("capability", [ABSENT, False, None, 0, 1, "true", [], {}, [OPTIONAL]])
@pytest.mark.parametrize("required_gap", [False, True])
def test_absent_false_or_invalid_capability_preserves_raw_capture_for_635_reader(
    tmp_path, monkeypatch, capability, required_gap,
):
    service, objects = service_for(tmp_path)
    state = state_for(capability, required_gap=required_gap)
    original = deepcopy(state.worker_observations.document())
    api = capture_api(service, monkeypatch, legacy_reader=True, uploads=[])
    journal = runner._ObservationJournal(api, state.lease, TOKEN, state)
    assert journal._prepare() and journal._send_pending()
    result = runner._record_training_observations(SimpleNamespace(api=api), state.lease,
                                                  TOKEN, state)
    assert result["status"] == "stored" and not result["complete"]
    saved = observations.read_snapshot(objects, "r1", "execute", 1, CLAIM_HASH)["payload"]
    assert saved == {**original, "failures": ["capture_incomplete"]}
    assert state.worker_observations.document() == original
    assert not observations.optional_capture_only(saved)


@pytest.mark.parametrize("kind", ["execute", "score"])
def test_gateway_advertises_capture_tag_support_in_claim_and_recovery(kind):
    now = datetime(2026, 9, 2, 1, tzinfo=timezone.utc)
    service, store = claim_service(lambda: now)
    if kind == "score":
        allocate = store.claim_assignment

        def allocate_score(**kwargs):
            allocate(**kwargs)
            store.run.update(kind="score", scored_run_id="execute1")
            store.run["claim_response"].update(kind="score", scored_run_id="execute1")
            return dict(store.run["claim_response"])

        store.claim_assignment = allocate_score
        get_run = store.get_run
        store.get_run = lambda run_id: ({"output_ref": "output1"}
                                       if run_id == "execute1" else get_run(run_id))
        service._objects = SimpleNamespace(get=lambda _ref: b'{"companies": []}')
        store.round_row["configuration_doc"]["scorer_policy"] = {}
    signed = original_claim(now)
    first = service.handle_claim(signed)
    recovered = service.handle_claim(recovery_claim(signed, now))
    assert first[CAPABILITY] is recovered[CAPABILITY] is True
    assert first["kind"] == recovered["kind"] == kind
    assert CAPABILITY not in store.run["claim_response"]  # Enrich saved legacy responses too.
    assert first["claim_request_hash"] == store.run["claim_request_hash"]


@pytest.mark.parametrize("capability", [ABSENT, True, False, "true"])
def test_negotiation_never_excuses_required_capture_failure(capability):
    example = captured_timeout()
    artifact = example["worker_observation_artifacts"]["run"]
    artifact["payload"]["failures"].append("capture_incomplete")
    lease = {} if capability is ABSENT else {CAPABILITY: capability}
    artifact["payload"]["failures"] = observations.failure_tags_for_lease(
        artifact["payload"]["failures"], lease)
    # This is a required/unknown failure, even when optional attribution exists.
    refresh(example)
    before = deepcopy(example)
    result = training_eligibility.assess_capture(
        submission(), {"run": example}, ["run"], enabled=True)
    assert result["eligible"]  # Exact provider pairs survive an independent runtime failure tag.
    assert not observations.optional_capture_only(artifact["payload"])
    assert example == before


@pytest.mark.parametrize("capability", [ABSENT, True, False, "true"])
def test_unknown_failure_is_not_reclassified_or_accepted(capability):
    state = state_for(capability)
    state.worker_observations.failures.add("unknown_failure")
    document = state.worker_observations.document()
    document["failures"] = observations.failure_tags_for_lease(document["failures"], state.lease)
    assert "unknown_failure" in document["failures"]
    assert not document["complete"]
    for reader in (reader_635(), observations):
        with pytest.raises(reader.ObservationError, match="counts_invalid"):
            reader.build_snapshot("r1", "execute", document, lease_generation=1,
                                  claim_request_hash=CLAIM_HASH)


def test_exact_635_reader_rejects_unprojected_new_tag_in_both_protocols():
    state = state_for(True)
    journal = runner._ObservationJournal(SimpleNamespace(), state.lease, TOKEN, state)
    assert journal._prepare()
    document = journal.pending[0]
    with pytest.raises(reader_635().ObservationError, match="segment_failure_invalid"):
        reader_635()._clean_segment_payload(document["payload"])
    with pytest.raises(reader_635().ObservationError, match="counts_invalid"):
        reader_635().build_snapshot("r1", "execute", state.worker_observations.document(),
                                    lease_generation=1, claim_request_hash=CLAIM_HASH)
