"""Real HTTP/SQL claims and completion across a validator stake decrease."""

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from lab_arena import contracts, runner as rn
from lab_arena.api import create_app
from lab_arena.validator import run_validator_loops
from tests.lab_arena.test_lab_arena_service_round import (
    Harness, _start_round, _run_stage_one_to_scoring, connect, database, keypair,
)
from tests.test_arena_validator_local_runtime import _ImmediateStop, _WeightRecorder


class StakeHarness(Harness):
    def objects_key(self):
        # This module has its own database, so it needs its own object bucket.
        return "stake-flow"


@pytest.mark.parametrize("kind", ["execute", "score"])
def test_unlisted_validator_lease_survives_stake_drop_and_weights_continue(connect, tmp_path, kind):
    harness = StakeHarness(connect, tmp_path, challengers=["StakeFlow"], runners=["alpha", "beta"])
    alpha, beta = harness.runner_keys
    harness.service.config.defaults.runner_hotkeys = (alpha,)
    harness.chain.stakes[beta] = 75_000
    harness.chain.active[beta] = False
    participants = _start_round(harness, day=1 if kind == "execute" else 2)
    if kind == "score":
        _run_stage_one_to_scoring(harness, participants, runners=1)
    else:
        harness.clock.advance_to(harness.schedule()["stage_1_start"])
        harness.service.advance_round(harness.round_id)
    before_config = harness.service.store.get_round(harness.round_id)["configuration_doc"]
    assert before_config["runner_hotkeys"] == [alpha]
    harness.clock.now = datetime.now(timezone.utc)
    key = keypair("svc-runner-beta")
    envelope = contracts.build_signed_request(
        scope=contracts.SCOPE_CLAIM, round_id=harness.round_id,
        hotkey=beta, body={"declared_parallelism": 1},
        timestamp=int(harness.clock().timestamp()),
        sign_message=lambda message: key.sign(message.encode()).hex(),
    )
    with TestClient(create_app(harness.service)) as http:
        api = rn.HttpArenaApiClient("http://localhost", client=http)
        harness.api_factory = lambda: api
        worker = harness.runner(1, parallel=1)
        try:
            lease = api.claim(envelope)
            assert lease["status"] == "leased" and lease["kind"] == kind
            assert api.claim(envelope)["run_id"] == lease["run_id"]
            leased = harness.service.store.get_run(lease["run_id"])
            assert leased["runner_hotkey"] == beta
            harness.chain.stakes[beta] = 74_999
            # Recover a saved lease even if its original HTTP response was lost.
            # Recovery must never call the allocating RPC or extend the lease.
            original_claim = harness.service.store.claim_assignment
            def unexpected_allocation(**kwargs):
                pytest.fail("stake-denied recovery must not allocate work")
            harness.service.store.claim_assignment = unexpected_allocation
            try:
                assert api.claim(envelope) == lease
            finally:
                harness.service.store.claim_assignment = original_claim
            new_envelope = contracts.build_signed_request(
                scope=contracts.SCOPE_CLAIM, round_id=harness.round_id,
                hotkey=beta, body={"declared_parallelism": 1},
                timestamp=int(harness.clock().timestamp()),
                sign_message=lambda message: key.sign(message.encode()).hex(),
            )
            rejected = http.post("/arena/v1/runs/claim", json=new_envelope)
            assert rejected.status_code == 403
            assert rejected.json()["code"] == "runner_stake_below_minimum"
            changed_request = contracts.build_signed_request(
                scope=contracts.SCOPE_CLAIM, round_id=harness.round_id,
                hotkey=beta, body={"declared_parallelism": 2},
                timestamp=int(harness.clock().timestamp()),
                request_id=envelope["request_id"],
                sign_message=lambda message: key.sign(message.encode()).hex(),
            )
            changed = http.post("/arena/v1/runs/claim", json=changed_request)
            assert changed.status_code == 403
            assert changed.json()["code"] == "runner_stake_below_minimum"
            assert harness.service.store.get_run(lease["run_id"]) == leased

            # A cache miss can still fetch the issued lease's pinned scorer.
            harness.service.config.scorer_image_access = lambda ref, digest: {"image_digest": digest}
            access = http.get(
                "/arena/v1/runs/%s/image-access" % lease["run_id"],
                headers={"x-lab-arena-lease": lease["lease_token"]},
            )
            assert access.status_code == 200
            assert access.json()["image_digest"] == lease["image_digest"]
            completion = worker._executor.execute(lease, lease["lease_token"], lease["icp"])
            assert completion["body"]["result"]["terminal_status"] == "accepted"
            assert api.complete(completion)["status"] == "accepted"
            assert harness.service.store.get_run(lease["run_id"])["status"] == "accepted"
            assert harness.service.store.get_round(harness.round_id)["configuration_doc"] == before_config

            # Use the real HTTP claim adapter and real runner inside normal loops.
            # No new model can run after denial; two weight cycles still occur.
            recorded = _WeightRecorder()
            stop = _ImmediateStop()
            class DeniedRunner:
                def run_once(self, *, stop_event):
                    assert worker.run_once(stop_event=stop_event) == 0
                    assert recorded.second_run.wait(2)
                    stop.set()
                    return 0

                def close(self):
                    pass  # The outer finally owns this worker and shared HTTP client.

            prior_runs = harness.service.store.list_runs(harness.round_id)
            run_validator_loops(
                orchestrator=recorded, runner_factory=DeniedRunner,
                epoch_supplier=lambda: 1, stop=stop, poll_seconds=5,
            )
            assert recorded.runs >= 2
            assert worker.completed == []
            assert worker._slots.acquire(blocking=False)
            worker._slots.release()
            assert harness.service.store.list_runs(harness.round_id) == prior_runs

            harness.chain.permits[beta] = False
            revoked = http.post("/arena/v1/runs/%s/complete" % lease["run_id"], json=completion)
            assert revoked.status_code == 403
            assert revoked.json()["code"] == "runner_validator_required"
        finally:
            worker.close()
