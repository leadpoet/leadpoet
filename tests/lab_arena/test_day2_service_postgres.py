"""Three-day journey using actual SQL, service, runner, broker and publication."""
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
import time

import pytest
from fastapi.testclient import TestClient

from lab_arena import benchmark_commitment as bc, contracts
from lab_arena.api import create_app
from lab_arena.store import ArenaStoreError
from tests.lab_arena.test_lab_arena_service_round import (
    Harness, database, connect, _run_stage_one_to_scoring,
)


def start(harness, suffix):
    service = harness.service
    cutoff = (datetime.now(timezone.utc) + timedelta(seconds=8)).replace(microsecond=0)
    service.config.defaults = replace(service.config.defaults, benchmark_commit_reveal_from=bc.iso(cutoff))
    config = service.create_round(cutoff, round_id=f"arena-2026-09-11-{suffix}")
    harness.round_id = config["round_id"]
    harness.clock.now = cutoff - timedelta(minutes=1)
    submission_id = harness.submit("Day2-" + suffix, harness.round_id)
    with TestClient(create_app(service)) as client:
        base = f"/arena/v1/rounds/{harness.round_id}"
        assert client.get(base + "/benchmark-commitment").status_code == 409
        assert client.get(base + "/benchmark").status_code == 403
    time.sleep(max(0, (cutoff - datetime.now(timezone.utc)).total_seconds()) + 0.05)
    harness.clock.now = cutoff
    return submission_id, cutoff


def test_full_three_day_http_journey_survives_restart(connect, tmp_path):
    harness = Harness(connect, tmp_path, challengers=[], runners=["alpha", "beta"])
    harness.chain.epoch = 40000
    submission_id, cutoff = start(harness, "journey")
    service = harness.service
    assert service.advance_round(harness.round_id)["status"] == "ok"
    row = service.store.get_round(harness.round_id)
    assert bc.instant(row["benchmark_reveal_at"]) == cutoff + timedelta(days=1)
    frozen = service.public_benchmark_commitment(harness.round_id)
    assert bc.instant(frozen["committed_at"]) >= cutoff
    for participant in row["participants"]:
        harness.flavors.setdefault(participant["submission_id"], "PublicBaseline")
    run_icp = harness.sandbox.run_icp
    def verified_input(spec):
        from lab_arena import runtime
        staged = json.loads((spec.input_dir / runtime.INPUT_FILE_NAME).read_text())
        assert "benchmark_proof" not in staged and "canonical_preimage" not in staged
        assert "nonce" not in staged["icp"]
        assert not any("NONCE" in key or "PROOF" in key for key in spec.extra_environment)
        return run_icp(spec)
    harness.sandbox.run_icp = verified_input
    _run_stage_one_to_scoring(harness, len(row["participants"]), runners=2)
    # Restart while scoring: the committed bank, source, scores and nonce
    # selection survive even with the creation flag now absent.
    harness.service = harness.build_service()
    service = harness.service
    assert service.public_benchmark_commitment(harness.round_id) == frozen
    assert service.commit_benchmark(harness.round_id)["status"] == "existing"
    harness.advance_until("published", runners=2)
    harness.clock.now = bc.instant(service.store.get_round(harness.round_id)["published_at"])
    assert harness.clock.now < cutoff + timedelta(days=1)
    with TestClient(create_app(service)) as client:
        base = f"/arena/v1/rounds/{harness.round_id}"
        results = client.get(base + "/results/" + submission_id)
        assert results.status_code == 200
        result = results.json()
        assert result["public_icp_status"] == "pending"
        assert result["outputs"] == {} and result["scores"] == {"stage_1": [], "stage_2": []} and result["run_results"] == []
        assert result["submission_scores"]["final"] is not None
        assert client.get("/arena/v1/submissions/" + submission_id + "/code").status_code == 200
        assert client.get(base + "/benchmark").status_code == 403
        assert client.get(base + "/benchmark-commitment").json() == frozen
        harness.clock.now = cutoff + timedelta(days=1) - timedelta(microseconds=1)
        assert client.get(base + "/benchmark").status_code == 403
        harness.clock.now = cutoff + timedelta(days=1)
        reveal = client.get(base + "/benchmark")
        assert reveal.status_code == 200
        assert bc.verify_reveal(frozen, reveal.json()) == frozen["manifest_hash"]
        ready = client.get(base + "/results/" + submission_id).json()
        assert ready["public_icp_status"] == "ready" and ready["public_icp_count"] == 20
        assert sum(len(values) for values in ready["scores"].values()) == 20 and ready["outputs"]
        assert client.get("/arena/v1/competition").status_code == 200
    after = service.store.get_round(harness.round_id)
    assert after["benchmark_ref"] == row["benchmark_ref"]
    assert after["benchmark_commitment_doc"] == row["benchmark_commitment_doc"]


def test_lost_storage_and_database_acknowledgements_reuse_stored_winner(connect, tmp_path):
    harness = Harness(connect, tmp_path, challengers=[], runners=["alpha"])
    _, _ = start(harness, "lostack")
    service = harness.service
    put = service._objects.put
    def lost_put(ref, payload):
        put(ref, payload)
        if "/benchmarks/" in ref:
            raise TimeoutError("lost private upload acknowledgement")
    service._objects.put = lost_put
    commit = service.store.commit_round_v3
    def lost_commit(*args, **kwargs):
        commit(*args, **kwargs)
        raise ArenaStoreError("lost database acknowledgement")
    service.store.commit_round_v3 = lost_commit
    assert service.commit_benchmark(harness.round_id)["status"] == "existing"
    winner = service.public_benchmark_commitment(harness.round_id)
    harness.service = harness.build_service()
    harness.service.config.daily_icp_source = lambda **kwargs: pytest.fail("regenerated committed benchmark")
    assert harness.service.commit_benchmark(harness.round_id)["status"] == "existing"
    assert harness.service.public_benchmark_commitment(harness.round_id) == winner


def test_activation_only_changes_new_rounds_and_existing_response_stays_truthful(connect, tmp_path):
    harness = Harness(connect, tmp_path, challengers=[], runners=["alpha"])
    service = harness.service
    cutoff = (datetime.now(timezone.utc) + timedelta(hours=12)).replace(microsecond=0)
    old = service.create_round(cutoff, round_id="arena-2026-09-20-legacy")
    assert "benchmark_disclosure_policy" not in old
    service.config.defaults = replace(service.config.defaults, benchmark_commit_reveal_from=bc.iso(cutoff))
    assert service.create_round(cutoff, round_id=old["round_id"]) == old
    newer = service.create_round(cutoff + timedelta(days=1), round_id="arena-2026-09-21-new")
    assert newer["benchmark_disclosure_policy"] == bc.POLICY
    row = service.store.get_round(newer["round_id"])
    assert bc.instant(row["benchmark_reveal_at"]) == cutoff + timedelta(days=2)
    restarted = harness.build_service()
    assert restarted.public_round(newer["round_id"])["benchmark_state"] == "pending_commitment"
    assert restarted.public_round(old["round_id"]).get("disclosure_policy") is None


def test_strict_invalid_bank_cancels_before_freezing_even_past_deadline(connect, tmp_path):
    from tests.lab_arena.icp_fixtures import daily_icps
    harness = Harness(connect, tmp_path, challengers=[], runners=["alpha"])
    start(harness, "invalid")
    invalid = daily_icps()
    invalid[0]["baseline_score"] = 90  # reserved public display field
    harness.service.config.daily_icp_source = lambda **kw: {"status": "ready", "set_id": kw["set_id"], "icps": invalid}
    harness.service.freeze_participants = lambda _: pytest.fail("invalid bank froze participants")
    harness.clock.advance_to(harness.schedule()["benchmark_deadline"])
    result = harness.service.commit_benchmark(harness.round_id)
    assert result == {"status": "cancelled", "reason": "benchmark_data_invalid"}
    row = harness.service.store.get_round(harness.round_id)
    assert row["status"] == "cancelled" and row["benchmark_commitment_doc"] is None
    harness.clock.now = bc.instant(row["benchmark_reveal_at"])
    from lab_arena.service import ServiceError
    with pytest.raises(ServiceError, match="benchmark_not_public"):
        harness.service.public_benchmark(harness.round_id)
