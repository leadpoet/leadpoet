"""Active replay copies accepted execution inputs, never scorer authority."""

import copy
import io
import json
import tarfile

import pytest

from lab_arena import contracts, scoring_provider_compat
from scripts import verify_arena_parallel_round as verification
from tests.test_verify_arena_parallel_round import (
    REPLAY_SOURCE_ROUND_ID, REPLAY_TARGET_ROUND_ID, ROUND_ID,
    _replay_source_fixture, _replay_target_service,
)


SELECTED = ["source-a:0", "source-b:1"]


def _active_source(monkeypatch):
    built, runs, documents = _replay_source_fixture(monkeypatch)
    source = built.store.get_round(REPLAY_SOURCE_ROUND_ID)
    source["status"] = "stage2_scoring"
    source["configuration_doc"]["mode"] = "live"
    for run in runs:
        if run["status"] == "accepted":
            run["output_hash"] = contracts.document_hash(json.loads(documents[run["output_ref"]]))
    return built, source, runs, documents


def _replay(built, **changes):
    options = dict(assignments=SELECTED, active_frozen=True, miner_score_route=True)
    options.update(changes)
    return verification._saved_output_replay(
        built, REPLAY_SOURCE_ROUND_ID, REPLAY_TARGET_ROUND_ID, **options,
    )


def test_active_replay_has_new_inputs_without_old_judgment_authority(monkeypatch):
    built, source, runs, documents = _active_source(monkeypatch)
    # An unrelated unfinished execution must not block the selected immutable inputs.
    runs[-1]["status"] = "pending"
    runs.append({"kind": "score", "status": "failed", "result_doc": {
        "legacy_partial_checkpoint": {"old_scorer_authority": "must not copy"},
    }})
    before = copy.deepcopy((source, runs, documents))
    replay = _replay(built)
    assert (source, runs, documents) == before
    assert replay["evidence"]["active_frozen_source"] is True
    assert replay["evidence"]["historical_source"] is False
    assert replay["evidence"]["source_unaccepted_assignment_count"] == 2
    assert "omitted_failed_assignment_count" not in replay["evidence"]
    assert replay["evidence"]["selection_scope"].startswith("selected_active_frozen_")
    assert replay["evidence"]["source_schedule"] == source["configuration_doc"]["schedule"]
    assert replay["evidence"]["score_provider_route"] == scoring_provider_compat.COMPATIBILITY_VERSION
    assert replay["evidence"]["target_benchmark_hash"] == contracts.document_hash({
        "schema_version": "leadpoet.lab_arena.benchmark.v1",
        "round_id": REPLAY_TARGET_ROUND_ID, "icps": replay["icps"],
    })
    assert all(o["source_stored_output_hash_present"] for o in replay["evidence"]["origins"])
    with tarfile.open(fileobj=io.BytesIO(replay["archive"]), mode="r:gz") as archive:
        names = {member.name.rsplit("/", 1)[-1] for member in archive.getmembers() if member.isfile()}
        assert names == {"LICENSE", "outputs.json", "harness.py"}
        saved = json.load(archive.extractfile(next(m for m in archive.getmembers() if m.name.endswith("outputs.json"))))
    assert [group[0]["company_name"] for group in saved.values()] == ["Acme", "Cedar"]
    assert b"old_scorer_authority" not in replay["archive"]


@pytest.mark.parametrize("status", ["open", "published", "cancelled", "unknown"])
def test_active_replay_rejects_unfrozen_or_wrong_source_mode(monkeypatch, status):
    built, source, _runs, _documents = _active_source(monkeypatch)
    source["status"] = status
    with pytest.raises(verification.VerificationError, match="committed active live"):
        _replay(built)


@pytest.mark.parametrize("change", [
    {"assignments": None}, {"assignments": []}, {"miner_score_route": False},
])
def test_active_replay_requires_selection_and_host_compat_route(monkeypatch, change):
    built, _source, _runs, _documents = _active_source(monkeypatch)
    with pytest.raises(verification.VerificationError, match="requires selected"):
        _replay(built, **change)


@pytest.mark.parametrize("mutation", [
    "missing_bank", "missing_date", "wrong_network", "wrong_netuid", "shadow",
    "missing_hash", "wrong_hash", "pending_selected", "failed_selected",
])
def test_active_replay_rejects_invalid_or_incomplete_selected_inputs(monkeypatch, mutation):
    built, source, runs, _documents = _active_source(monkeypatch)
    if mutation == "missing_bank": source["benchmark_ref"] = None
    elif mutation == "missing_date": source["evaluation_date"] = None
    elif mutation == "wrong_network": source["configuration_doc"]["network_name"] = "other"
    elif mutation == "wrong_netuid": source["configuration_doc"]["netuid"] = 72
    elif mutation == "shadow": source["configuration_doc"]["mode"] = "shadow"
    elif mutation == "missing_hash": runs[0]["output_hash"] = ""
    elif mutation == "wrong_hash": runs[0]["output_hash"] = "sha256:" + "0" * 64
    elif mutation == "pending_selected": runs[0]["status"] = "pending"
    elif mutation == "failed_selected": runs[0]["status"] = "failed"
    with pytest.raises(verification.VerificationError):
        _replay(built)


def test_active_replay_rejects_frozen_metadata_change_during_preparation(monkeypatch):
    built, source, _runs, _documents = _active_source(monkeypatch)
    original = built.store.get_round
    calls = []
    def changed(round_id):
        if round_id == REPLAY_SOURCE_ROUND_ID:
            calls.append(round_id)
            if len(calls) == 2:
                source["configuration_doc"]["netuid"] = 72
        return original(round_id)
    built.store.get_round = changed
    with pytest.raises(verification.VerificationError, match="frozen inputs changed"):
        _replay(built)


def test_active_replay_resume_allows_source_to_finish_without_changing_inputs(monkeypatch):
    built, source, _runs, _documents = _active_source(monkeypatch)
    original = _replay(built)
    source["status"] = "published"
    get_round = built.store.get_round
    built.store.get_round = lambda rid: (
        {"round_id": REPLAY_TARGET_ROUND_ID, "configuration_doc": {"mode": "shadow"}}
        if rid == REPLAY_TARGET_ROUND_ID else get_round(rid)
    )
    resumed = _replay(built)
    assert resumed["archive"] == original["archive"]
    assert resumed["evidence"]["input_hash"] == original["evidence"]["input_hash"]
    assert resumed["evidence"]["target_benchmark_hash"] == original["evidence"]["target_benchmark_hash"]


def test_published_default_still_rejects_active_source_and_source_modes_are_explicit(monkeypatch):
    built, _source, _runs, _documents = _active_source(monkeypatch)
    with pytest.raises(verification.VerificationError, match="requires a published"):
        _replay(built, active_frozen=False)
    with pytest.raises(verification.VerificationError, match="choose one"):
        verification._build_pinned_service(
            REPLAY_TARGET_ROUND_ID, replay_published_round=REPLAY_SOURCE_ROUND_ID,
            replay_active_frozen_round=REPLAY_SOURCE_ROUND_ID,
        )
    args = verification.build_parser().parse_args([
        "serve", "--round-id", REPLAY_TARGET_ROUND_ID,
        "--environment-file", "/tmp/gateway.env",
        "--replay-active-frozen-round", REPLAY_SOURCE_ROUND_ID,
        "--replay-assignment", "source-a:0", "--replay-assignment", "source-b:1",
        "--replay-miner-score-route",
    ])
    assert args.replay_active_frozen_round == REPLAY_SOURCE_ROUND_ID
    assert args.replay_published_round is None


def test_active_resume_rejects_changed_target_icp_with_same_company_archive():
    service = _replay_target_service()
    benchmark_ref = service.store.row["benchmark_ref"]
    benchmark = json.loads(service.object_documents[benchmark_ref])
    service._saved_output_replay.update({
        "active_frozen_source": True,
        "target_benchmark_hash": contracts.document_hash(benchmark),
    })
    verification._validate_frozen_round(service, service.store.row)
    benchmark["icps"][0]["prompt"] = "Different original ICP"
    service.object_documents[benchmark_ref] = json.dumps(benchmark).encode()
    with pytest.raises(verification.VerificationError, match="benchmark inputs differ"):
        verification._validate_frozen_round(service, service.store.row)
    service.store.row["benchmark_ref"] = None
    with pytest.raises(verification.VerificationError, match="benchmark is missing"):
        verification._validate_frozen_round(service, service.store.row)
