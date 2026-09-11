"""Frozen commitments, exact reveal verification, and disclosure boundaries."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from lab_arena import benchmark_commitment as bc, contracts, icp_disclosure, source_disclosure
from lab_arena.api import create_app
from lab_arena.service import ArenaService, ServiceError
from lab_arena.runner import AssignmentExecutor, RunnerError

DAY1 = datetime(2026, 9, 10, 6, tzinfo=timezone.utc)
DAY2 = DAY1 + timedelta(days=1)


def fixture(status="committed"):
    row = {
        "round_id": "arena-2026-09-10", "status": "open",
        "configuration_doc": {"network_name": "finney", "netuid": 71,
            "benchmark_disclosure_policy": bc.POLICY,
            "schedule": {"submission_open": "2026-09-09T06:00:00Z", "submission_cutoff": bc.iso(DAY1)}},
        "benchmark_reveal_at": bc.iso(DAY2),
        "participants": [{"submission_id": "base", "is_king": True}],
    }
    # Unicode and Python float forms intentionally differ from JS stringify.
    icps = [{"icp_id": f"icp-{i}", "prompt": f"private ICP {i}: café 日本", "weight": 1.0, "tiny": 1e-7} for i in range(20)]
    artifact = bc.build_artifact(row, icps)
    payload = contracts.canonical_json(artifact).encode()
    row.update(status=status, icp_set_date="2026-09-09", evaluation_date="2026-09-10",
        benchmark_commitment_doc=artifact["commitment"], benchmark_committed_at=bc.iso(DAY1),
        benchmark_ref=f'arena/{row["round_id"]}/benchmarks/{contracts.hash_bytes(payload)[7:]}.json')
    return row, artifact, icps


def service_fixture(status="published", now=DAY1):
    row, artifact, icps = fixture(status)
    service = ArenaService.__new__(ArenaService)
    service._round = lambda _: row
    service.now = lambda: now
    payload = contracts.canonical_json(artifact).encode()
    reads = []
    def get(ref, limit):
        reads.append(ref)
        assert len(payload) <= limit
        return payload
    service._objects = SimpleNamespace(get_bounded=get)
    service._store = SimpleNamespace(list_runs=lambda *a, **kw: [])
    return service, row, artifact, icps, reads


def test_day1_endpoint_hash_only_no_private_object_read_and_no_store():
    service, row, artifact, icps, reads = service_fixture()
    with TestClient(create_app(service)) as http:
        commitment = http.get(f'/arena/v1/rounds/{row["round_id"]}/benchmark-commitment')
        hidden = http.get(f'/arena/v1/rounds/{row["round_id"]}/benchmark')
    assert commitment.status_code == 200 and hidden.status_code == 403
    assert commitment.json()["manifest_hash"] == artifact["commitment"]["manifest_hash"]
    assert "private ICP" not in commitment.text and "nonce" not in commitment.text
    assert reads == []
    assert all(r.headers["cache-control"] == "no-store" for r in (commitment, hidden))


def test_day2_exact_preimages_and_decorated_inputs_verify():
    service, row, artifact, icps, reads = service_fixture(now=DAY2)
    saved = service.public_benchmark_commitment(row["round_id"])
    reveal = service.public_benchmark(row["round_id"])
    assert bc.verify_reveal(saved, reveal) == saved["manifest_hash"]
    assert len(reads) == 1
    proof = {"commitment": artifact["commitment"], "canonical_preimage": artifact["canonical_preimages"][0]}
    bc.verify_assignment(proof, round_id=row["round_id"], position=0, evaluation_date=row["evaluation_date"], icp=icps[0])
    reveal["icps"][0]["prompt"] = "swapped display"
    with pytest.raises(bc.BenchmarkCommitmentError, match="benchmark_icp_mismatch"):
        bc.verify_reveal(saved, reveal)


@pytest.mark.parametrize("status", ["committed", "stage1", "stage1_scoring", "stage2", "scored"])
def test_active_overrun_never_reveals(status):
    row, _, _ = fixture(status)
    assert icp_disclosure.baseline_disclosure(row, [], DAY2 + timedelta(days=4)) is None
    assert icp_disclosure.public_metadata(row, DAY2)["benchmark_state"] == "reveal_delayed"


@pytest.mark.parametrize("status", ["published", "cancelled"])
def test_exact_boundary_and_terminal_gate(status):
    row, _, _ = fixture(status)
    assert icp_disclosure.baseline_disclosure(row, [], DAY2 - timedelta(microseconds=1)) is None
    assert icp_disclosure.baseline_disclosure(row, [], DAY2)["public_positions"] == list(range(20))
    row["benchmark_commitment_doc"] = None
    assert icp_disclosure.baseline_disclosure(row, [], DAY2) is None


@pytest.mark.parametrize("change", ["preimage", "hash", "order", "scope", "nonce", "extra"])
def test_tampering_fails_closed(change):
    row, original, _ = fixture()
    artifact = deepcopy(original)
    if change == "preimage": artifact["canonical_preimages"][0] = artifact["canonical_preimages"][0].replace("café", "cafe")
    if change == "hash": artifact["commitment"]["manifest_hash"] = "sha256:" + "0" * 64
    if change == "order": artifact["canonical_preimages"].reverse()
    if change == "scope": artifact["canonical_preimages"][0] = artifact["canonical_preimages"][0].replace('"netuid":71', '"netuid":72')
    if change == "nonce": artifact["canonical_preimages"][0] = artifact["canonical_preimages"][0].replace('"nonce":"', '"nonce":"x')
    if change == "extra": artifact["extra"] = "uncommitted data"
    with pytest.raises(bc.BenchmarkCommitmentError):
        bc.validate_artifact(artifact, commitment=row["benchmark_commitment_doc"])


def test_source_releases_on_day1_independent_of_reveal():
    row, _, _ = fixture("published")
    row.update(published_at=bc.iso(DAY1 + timedelta(hours=1)), publication_doc={"participants": [{"submission_id": "source"}]})
    submission = {"submission_id": "source", "round_id": row["round_id"], "status": "frozen", "source_ref": "private", "consent": {"public_rerun": True}}
    assert source_disclosure.disclosure_status(submission, DAY1 + timedelta(hours=1), round_row=row)["available"] is True
    assert icp_disclosure.baseline_disclosure(row, [], DAY1 + timedelta(hours=1)) is None
    row["status"] = "cancelled"
    assert source_disclosure.disclosure_status(submission, DAY2, round_row=row)["available"] is False


@pytest.mark.parametrize("marker", [None, "unknown"])
def test_unknown_marker_never_falls_back_to_legacy(marker):
    row, _, _ = fixture("published")
    row["configuration_doc"]["benchmark_disclosure_policy"] = marker
    assert icp_disclosure.disclosure_metadata(row) is None
    assert icp_disclosure.baseline_disclosure(row, [], DAY2) is None


def test_corrupt_private_object_blocks_reveal_but_hash_endpoint_still_works():
    service, row, _, _, _ = service_fixture(now=DAY2)
    service._objects.get_bounded = lambda *a: b'{"private": "corruption"}'
    assert service.public_benchmark_commitment(row["round_id"])["manifest_hash"]
    with pytest.raises(ServiceError, match="benchmark_data_invalid"):
        service.public_benchmark(row["round_id"])


def test_runner_rejects_missing_or_wrong_proof_before_creating_sandbox():
    executor = AssignmentExecutor(SimpleNamespace())  # no filesystem/sandbox needed
    with pytest.raises(RunnerError, match="benchmark_commitment_invalid"):
        executor.execute({"benchmark_disclosure_policy": bc.POLICY}, "lease", {"icp_id": "x"})


def test_offline_verifier_shared_vector_and_tamper_exit_code(tmp_path):
    import json
    from pathlib import Path
    import subprocess
    import sys
    fixture_path = Path(__file__).parent / "fixtures" / "benchmark_commit_reveal_v1.json"
    vector = json.loads(fixture_path.read_text())
    saved, reveal = tmp_path / "saved.json", tmp_path / "reveal.json"
    saved.write_text(json.dumps(vector["commitment"]))
    reveal.write_text(json.dumps(vector["reveal"]))
    command = [sys.executable, str(Path(__file__).resolve().parents[2] / "scripts" / "verify_arena_benchmark.py"), str(saved), str(reveal)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0 and "Verified all 20" in result.stdout
    vector["reveal"]["verification"]["canonical_preimages"].reverse()
    reveal.write_text(json.dumps(vector["reveal"]))
    invalid = subprocess.run(command, capture_output=True, text=True)
    assert invalid.returncode == 1 and "private ICP" not in invalid.stderr


def test_new_salts_change_hashes_without_changing_inputs():
    row, artifact, icps = fixture()
    other = bc.build_artifact(row, icps)
    assert other["commitment"]["manifest_hash"] != artifact["commitment"]["manifest_hash"]
    assert bc.validate_artifact(other, commitment=other["commitment"]) == icps


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 2**60])
def test_unsupported_json_numbers_refused(bad):
    row, _, icps = fixture()
    icps[0]["value"] = bad
    with pytest.raises((bc.BenchmarkCommitmentError, contracts.ArenaContractError, ValueError)):
        bc.build_artifact(row, icps)


@pytest.mark.parametrize("policy", [bc.POLICY, None])
def test_persistent_artifact_failure_cancels_at_existing_phase_deadline_only(policy):
    row, _, _ = fixture("stage1_scored")
    row["configuration_doc"]["schedule"]["stage_2_close"] = bc.iso(DAY1 + timedelta(hours=17))
    if policy is None:
        del row["configuration_doc"]["benchmark_disclosure_policy"]
    service = ArenaService.__new__(ArenaService)
    service._round = lambda _: row
    service._invalidate_hot_round = lambda: None
    def fail(_):
        raise ServiceError("benchmark_data_invalid", 503)
    service._advance_round_locked = fail
    service._store = SimpleNamespace(cancel_round=lambda *args: {"status": "cancelled"})
    service.now = lambda: DAY1 + timedelta(hours=16)
    with pytest.raises(ServiceError, match="benchmark_data_invalid"):
        service.advance_round(row["round_id"])
    service.now = lambda: DAY1 + timedelta(hours=17)
    if policy:
        assert service.advance_round(row["round_id"]) == {"status": "cancelled"}
    else:
        with pytest.raises(ServiceError, match="benchmark_data_invalid"):
            service.advance_round(row["round_id"])


def test_corrupt_artifact_is_rejected_before_consuming_a_lease():
    service, row, _, _, _ = service_fixture(status="committed")
    row["configuration_doc"]["runner_slot_ceiling"] = 4
    service._request_round = lambda *a, **kw: ({"hotkey": "validator", "body": {"declared_parallelism": 1}}, row)
    service._require_validator_authority = lambda _: None
    service._config = SimpleNamespace(chain=SimpleNamespace(hotkeys_owned_by_same_coldkey=lambda _: []))
    service._objects.get_bounded = lambda *a: b'{}'
    service._store.claim_assignment = lambda **kw: pytest.fail("consumed lease before verifying benchmark")
    with pytest.raises(ServiceError, match="benchmark_data_invalid"):
        service.handle_claim({})


def test_cached_open_round_refreshes_proof_when_another_coordinator_commits():
    service, row, artifact, _, _ = service_fixture(status="committed")
    row["configuration_doc"].update(runner_slot_ceiling=4, lease_ttl_seconds=420, scorer_image_digest="sha256:" + "0"*64, scorer_image_reference="scorer")
    opened = {**row, "status": "open", "benchmark_commitment_doc": None, "evaluation_date": None, "icp_set_date": None}
    request = {"hotkey": "validator", "body": {"declared_parallelism": 1}, "request_id": "id"}
    service._request_round = lambda *a, **kw: (request, opened)
    service._require_validator_authority = lambda _: None
    service._require_code_review = lambda *a: None
    service._lease_token = lambda _: "0"*64
    service._config = SimpleNamespace(chain=SimpleNamespace(hotkeys_owned_by_same_coldkey=lambda _: []))
    service._store.claim_assignment = lambda **kw: {"status": "leased", "kind": "score", "submission_id": "base", "icp_position": 0, "scored_run_id": "run"}
    service._store.get_run = lambda _: {"output_ref": "output"}
    service._objects.get = lambda _: b'{"companies": []}'
    row["configuration_doc"]["scorer_policy"] = {}
    # Request hashing is unrelated to this deliberate stale-row fixture.
    from unittest.mock import patch
    with patch.object(contracts, "request_bytes_hash", return_value="sha256:" + "0"*64):
        lease = service.handle_claim({})
    assert lease["evaluation_date"] == row["evaluation_date"]
    assert lease["benchmark_proof"]["commitment"] == artifact["commitment"]
    bc.verify_assignment(lease["benchmark_proof"], round_id=row["round_id"], position=0, evaluation_date=lease["evaluation_date"], icp=lease["icp"])
