"""Fresh authorization resolves one ambiguous claim through its original ID."""
from datetime import timedelta
from dataclasses import replace
from types import SimpleNamespace
import threading

import pytest

from lab_arena import contracts, runner as rn
from lab_arena.service import ArenaService, ServiceError
from lab_arena.store import hash_lease_token
from tests.lab_arena.test_lab_arena_runner import (
    BridgingRuntime, FakeApi, IMAGE, IMAGE_REFERENCE, MINER, ROUND, RUNNER,
    SOURCE_BYTES, SOURCE_REF, make_config, sign, verify,
)
from tests.lab_arena.stage_cutoff_drain420_postgres_test import (
    base_database, current_database, database, _seed, _claim, _complete,
    ROUND as SQL_ROUND,
)
from tests.lab_arena.validator_logging_test import logger, queued


class ClaimStore:
    def __init__(self, round_row, now):
        self.round_row = round_row
        self.now = now
        self.run = None
        self.allocations = 0
        self.arguments = []
        self.lock = threading.Lock()

    def claim_assignment(self, **kwargs):
        with self.lock:
            self.arguments.append(kwargs)
            if self.run is None:
                self.allocations += 1
                expiry = self.now() + timedelta(seconds=kwargs["lease_ttl_seconds"])
                response = {
                    "status": "leased", "request_id": kwargs["request_id"],
                    "run_id": "r1", "assignment_id": "a1", "submission_id": "s1",
                    "miner_hotkey": MINER, "kind": "execute", "stage": 1,
                    "icp_position": 0, "attempt": 1, "lease_generation": 1,
                    "stage_generation": 1, "lease_expires_at": expiry.isoformat(),
                }
                self.run = {
                    **response, "round_id": ROUND,
                    "runner_hotkey": kwargs["runner_hotkey"],
                    "claim_request_id": kwargs["request_id"],
                    "claim_request_hash": kwargs["request_hash"],
                    "lease_token_hash": kwargs["lease_token_hash"],
                    "claim_response": response,
                }
            if self.run["claim_request_hash"] != kwargs["request_hash"]:
                return {"status": "request_id_reused"}
            return dict(self.run["claim_response"])

    def get_run(self, run_id):
        return self.run if self.run and self.run["run_id"] == run_id else None

    def get_round(self, _round_id):
        return self.round_row

    def recover_claim_response(self, *, round_id, runner_hotkey, request_id, request_hash):
        if self.run is None:
            return None
        if (self.run["round_id"] != round_id
            or self.run["runner_hotkey"] != runner_hotkey
            or self.run["claim_request_id"] != request_id
            or self.run["claim_request_hash"] != request_hash):
            return None
        return dict(self.run["claim_response"])

    def get_submission_credential(self, *_args):
        return None


def claim_service(now):
    round_row = {
        "round_id": ROUND, "status": "stage1", "stage_generation": 1,
        "evaluation_date": "2026-09-02",
        "participants": [{"submission_id": "s1", "miner_hotkey": MINER,
                          "source_ref": SOURCE_REF,
                          "source_size_bytes": len(SOURCE_BYTES)}],
        "configuration_doc": {
            "runner_slot_ceiling": 1, "lease_ttl_seconds": 4500,
            "scorer_image_digest": IMAGE,
            "scorer_image_reference": IMAGE_REFERENCE,
        },
    }
    store = ClaimStore(round_row, now)
    service = object.__new__(ArenaService)
    service._store = store
    service._clock = now
    service._config = SimpleNamespace(verify_signature=verify)

    def request_round(envelope, *, scope, hot):
        assert hot is True
        validated = contracts.validate_signed_request(
            envelope, expected_scope=scope, now=int(now().timestamp()),
            verify_signature=verify, expected_round_id=ROUND,
        )
        return validated, round_row

    service._request_round = request_round
    service._benchmark_snapshot = lambda: SimpleNamespace(
        hotkeys=(RUNNER.ss58_address,), coldkeys=("owner",)
    )
    service._benchmark_validator_uid = lambda _snapshot, _hotkey: 0
    service._require_code_review = lambda *_args: None
    service.benchmark_icps = lambda _round_id: [{"icp_id": "arena:x", "prompt": "p", "max_companies": 5}]
    return service, store


def original_claim(now):
    return contracts.build_signed_request(
        scope=contracts.SCOPE_CLAIM, round_id=ROUND,
        hotkey=RUNNER.ss58_address, timestamp=int(now.timestamp()),
        body={"declared_parallelism": 1}, sign_message=sign,
    )


def recovery_claim(original, now):
    return contracts.build_signed_request(
        scope=contracts.SCOPE_CLAIM, round_id=ROUND,
        hotkey=RUNNER.ss58_address, timestamp=int(now.timestamp()),
        body={"recover_claim": original}, sign_message=sign,
    )


@pytest.mark.parametrize("elapsed", (301, 5401))
def test_recovery_replays_original_token_after_lost_response(tmp_path, elapsed):
    current = [make_config(tmp_path, FakeApi([]), BridgingRuntime(calls=0)).clock()]
    service, store = claim_service(lambda: current[0])

    class Api(FakeApi):
        def claim(self, envelope):
            self.claims.append(envelope)
            response = service.handle_claim(envelope)
            if len(self.claims) <= 3:
                raise rn.RunnerError("response lost", http_status=503)
            return response

    api = Api([])
    sandbox = BridgingRuntime(output={"companies": []}, calls=1)
    config = make_config(tmp_path, api, sandbox)
    config.clock = lambda: current[0]
    (tmp_path / "work").mkdir()
    runner = rn.Runner(config)
    try:
        assert runner.run_once(max_claims=1) == 0
        original = runner._pending_claims[ROUND]
        old_token = service._lease_token(original)
        if elapsed > 4500:
            # The server has expired the exact prior lease. Its old response
            # remains stored for idempotency but cannot be dispatched.
            store.run["status"] = "failed"
            store.run["lease_expires_at"] = (current[0] - timedelta(seconds=1)).isoformat()
        current[0] += timedelta(seconds=elapsed)
        assert runner.run_once(max_claims=1) == (0 if elapsed > 4500 else 1)
        assert store.allocations == 1
        assert {item["request_id"] for item in store.arguments} == {original["request_id"]}
        assert {item["request_hash"] for item in store.arguments} == {
            contracts.request_bytes_hash(original)
        }
        assert {item["lease_token_hash"] for item in store.arguments} == {
            hash_lease_token(old_token)
        }
        assert api.claims[3]["body"]["recover_claim"] is original
        assert runner._pending_claims == {}
        assert len(sandbox.specs) == (0 if elapsed > 4500 else 1)
        assert len(api.completions) == (0 if elapsed > 4500 else 1)
        assert len(api.provider_frames) == (0 if elapsed > 4500 else 1)
    finally:
        runner.close()


@pytest.mark.parametrize("elapsed", (301, 5401))
def test_recovery_allocates_once_when_original_never_committed(tmp_path, elapsed):
    current = [make_config(tmp_path, FakeApi([]), BridgingRuntime(calls=0)).clock()]
    service, store = claim_service(lambda: current[0])

    class Api(FakeApi):
        def claim(self, envelope):
            self.claims.append(envelope)
            if len(self.claims) <= 3:
                raise rn.RunnerError("request never reached server", http_status=503)
            return service.handle_claim(envelope)

    api = Api([])
    config = make_config(tmp_path, api, BridgingRuntime(output={"companies": []}, calls=0))
    config.clock = lambda: current[0]
    (tmp_path / "work").mkdir()
    runner = rn.Runner(config)
    try:
        assert runner.run_once(max_claims=1) == 0
        original = runner._pending_claims[ROUND]
        current[0] += timedelta(seconds=elapsed)
        assert runner.run_once(max_claims=1) == 1
        assert store.allocations == 1
        assert store.arguments[0]["request_id"] == original["request_id"]
        assert store.arguments[0]["request_hash"] == contracts.request_bytes_hash(original)
        assert runner._pending_claims == {}
    finally:
        runner.close()


@pytest.mark.parametrize("idle_status", ("no_pending", "no_free_slot"))
def test_unresolved_recovery_keeps_original_until_work_is_available(tmp_path, idle_status):
    current = [make_config(tmp_path, FakeApi([]), BridgingRuntime(calls=0)).clock()]
    service, store = claim_service(lambda: current[0])
    original_allocator = store.claim_assignment
    first_recovery = [True]

    def temporary_idle(**kwargs):
        if first_recovery[0]:
            first_recovery[0] = False
            return {"status": idle_status}
        return original_allocator(**kwargs)

    store.claim_assignment = temporary_idle

    class Api(FakeApi):
        def claim(self, envelope):
            self.claims.append(envelope)
            if len(self.claims) <= 3:
                raise rn.RunnerError("request never reached server", http_status=503)
            return service.handle_claim(envelope)

    api = Api([])
    config = make_config(tmp_path, api, BridgingRuntime(output={"companies": []}, calls=0))
    config.clock = lambda: current[0]
    runner = rn.Runner(config)
    try:
        assert runner.run_once(max_claims=1) == 0
        original = runner._pending_claims[ROUND]
        current[0] += timedelta(seconds=301)
        assert runner.run_once(max_claims=100) == 0
        assert runner._pending_claims[ROUND] is original
        assert store.allocations == 0
        assert len(api.claims) == 4  # no immediate rescan loop
        assert runner.run_once(max_claims=1) == 1
        assert store.allocations == 1
        assert store.arguments[0]["request_id"] == original["request_id"]
        assert runner._pending_claims == {}
    finally:
        runner.close()


def test_stale_recovery_no_work_is_idle_without_claim_denial(tmp_path):
    current = [make_config(tmp_path, FakeApi([]), BridgingRuntime(calls=0)).clock()]
    service, store = claim_service(lambda: current[0])

    class Api(FakeApi):
        def claim(self, envelope):
            self.claims.append(envelope)
            response = service.handle_claim(envelope)
            if len(self.claims) <= 3:
                raise rn.RunnerError("response lost", http_status=503)
            return response

    api = Api([])
    operations = logger()
    config = replace(
        make_config(tmp_path, api, BridgingRuntime(calls=0)),
        operational_logger=operations,
    )
    config.clock = lambda: current[0]
    runner = rn.Runner(config)
    try:
        assert runner.run_once(max_claims=1) == 0
        assert runner._pending_claims[ROUND]
        store.run["status"] = "accepted"
        current[0] += timedelta(seconds=301)
        assert runner.run_once(max_claims=100) == 0
        assert runner._pending_claims == {}
        assert len(api.claims) == 4  # no immediate rescan loop
        assert operations.snapshot()["scoring_state"] == "idle"
        assert all(
            event.get("content", {}).get("reason") != "claim_denied"
            for event in queued(operations)
        )
    finally:
        runner.close()


@pytest.mark.parametrize("stale", (False, True))
def test_stake_fall_lookup_still_checks_current_lease(stale):
    from datetime import datetime, timezone
    current = [datetime(2026, 9, 2, 1, 0, tzinfo=timezone.utc)]
    service, store = claim_service(lambda: current[0])
    old = original_claim(current[0])
    first = service.handle_claim(old)
    assert first["status"] == "leased"
    service._benchmark_validator_uid = lambda *_args: (_ for _ in ()).throw(
        ServiceError("runner_stake_below_minimum", 403)
    )
    current[0] += timedelta(seconds=301)
    if stale:
        store.run["status"] = "accepted"
    result = service.handle_claim(recovery_claim(old, current[0]))
    assert result["status"] == ("no_work" if stale else "leased")
    assert result["claim_recovery"] == "resolved"
    assert store.allocations == 1


def test_late_original_rpc_and_fresh_recovery_use_one_claim_identity():
    from datetime import datetime, timezone
    current = [datetime(2026, 9, 2, 1, 0, tzinfo=timezone.utc)]
    service, store = claim_service(lambda: current[0])
    entered = threading.Event()
    resume = threading.Event()
    original_allocator = store.claim_assignment

    def delayed_allocator(**kwargs):
        if threading.current_thread().name == "original-claim":
            entered.set()
            assert resume.wait(3)
        return original_allocator(**kwargs)

    store.claim_assignment = delayed_allocator
    old = original_claim(current[0])
    completed, errors = [], []

    def old_rpc():
        try:
            completed.append(service.handle_claim(old))
        except Exception as exc:
            errors.append(exc)

    worker = threading.Thread(target=old_rpc, name="original-claim")
    worker.start()
    try:
        assert entered.wait(3)
        current[0] += timedelta(seconds=301)
        recovered = service.handle_claim(recovery_claim(old, current[0]))
    finally:
        resume.set()
        worker.join(3)
    assert not worker.is_alive() and errors == []
    assert store.allocations == 1
    assert recovered["status"] == completed[0]["status"] == "leased"
    assert recovered["run_id"] == completed[0]["run_id"]
    assert recovered["lease_token"] == completed[0]["lease_token"]
    assert recovered["claim_recovery"] == "resolved"
    assert recovered["claim_recovery_request_id"] == old["request_id"]


@pytest.mark.parametrize("stale", (
    "accepted", "expired", "revoked_token", "changed_lease_generation",
    "changed_stage_generation", "round_ended",
))
def test_recovery_returns_no_work_for_stale_original_lease(stale):
    from datetime import datetime, timezone
    now = datetime(2026, 9, 2, 1, 0, tzinfo=timezone.utc)
    current = [now]
    service, store = claim_service(lambda: current[0])
    old = original_claim(now)
    first = service.handle_claim(old)
    assert first["status"] == "leased"
    current[0] += timedelta(seconds=301)
    if stale == "accepted":
        store.run["status"] = "accepted"
    elif stale == "expired":
        store.run["lease_expires_at"] = (current[0] - timedelta(seconds=1)).isoformat()
    elif stale == "revoked_token":
        store.run["lease_token_hash"] = "sha256:" + "a" * 64
    elif stale == "changed_lease_generation":
        store.run["lease_generation"] = 2
    elif stale == "changed_stage_generation":
        store.round_row["stage_generation"] = 2
    else:
        store.round_row["status"] = "cancelled"
    result = service.handle_claim(recovery_claim(old, current[0]))
    assert result == {"status": "no_work", "claim_recovery": "resolved",
                      "claim_recovery_request_id": old["request_id"]}
    assert store.allocations == 1


def test_old_gateway_denial_cannot_clear_ambiguous_claim(tmp_path):
    current = [make_config(tmp_path, FakeApi([]), BridgingRuntime(calls=0)).clock()]

    class OldApi(FakeApi):
        def claim(self, envelope):
            self.claims.append(envelope)
            if len(self.claims) <= 3:
                raise rn.RunnerError("response lost", http_status=503)
            return {"status": "rejected", "code": "declared_parallelism_invalid"}

    api = OldApi([])
    config = make_config(tmp_path, api, BridgingRuntime(calls=0))
    config.clock = lambda: current[0]
    runner = rn.Runner(config)
    try:
        with pytest.raises(rn.RunnerError):
            runner.claim_one()
        original = runner._pending_claims[ROUND]
        current[0] += timedelta(seconds=301)
        with pytest.raises(rn.RunnerError, match="recovery unavailable"):
            runner.claim_one()
        assert runner._pending_claims[ROUND] is original
        assert all(set(item["body"]) == {"recover_claim"} for item in api.claims[3:])
    finally:
        runner.close()


@pytest.mark.parametrize("response", (
    {"status": "no_work", "claim_recovery": "unresolved"},
    {"status": "no_work", "claim_recovery": "resolved",
     "claim_recovery_request_id": "0" * 32},
    {"status": "leased", "claim_recovery": "unresolved"},
))
def test_unbound_recovery_answer_keeps_original_pending(tmp_path, response):
    current = [make_config(tmp_path, FakeApi([]), BridgingRuntime(calls=0)).clock()]

    class Api(FakeApi):
        def claim(self, envelope):
            self.claims.append(envelope)
            if len(self.claims) <= 3:
                raise rn.RunnerError("response lost", http_status=503)
            return response

    api = Api([])
    config = make_config(tmp_path, api, BridgingRuntime(calls=0))
    config.clock = lambda: current[0]
    runner = rn.Runner(config)
    try:
        with pytest.raises(rn.RunnerError):
            runner.claim_one()
        original = runner._pending_claims[ROUND]
        current[0] += timedelta(seconds=301)
        with pytest.raises(rn.RunnerError, match="recovery unavailable"):
            runner.claim_one()
        assert runner._pending_claims[ROUND] is original
    finally:
        runner.close()


@pytest.mark.parametrize("change", (
    "wrong_hotkey", "wrong_round", "wrong_scope", "bad_inner_signature", "extra_body",
    "nested", "future_inner", "same_request_id",
))
def test_recovery_rejects_changed_original_identity(change):
    from datetime import datetime, timezone
    now = datetime(2026, 9, 2, 1, 0, tzinfo=timezone.utc)
    service, store = claim_service(lambda: now + timedelta(seconds=301))
    old = original_claim(now)
    outer = recovery_claim(old, now + timedelta(seconds=301))
    if change == "wrong_hotkey":
        old["hotkey"] = "5" * 48
    elif change == "wrong_round":
        old["round_id"] = "arena-other"
    elif change == "wrong_scope":
        old["scope"] = contracts.SCOPE_COMPLETE
    elif change == "bad_inner_signature":
        old["signature"] = "0x" + "0" * 128
    elif change == "extra_body":
        outer["body"]["declared_parallelism"] = 1
    elif change == "nested":
        old["body"]["recover_claim"] = {"attempt": "nested"}
        old["signature"] = "0x" + sign(contracts.signed_request_message(old))
    elif change == "future_inner":
        old["timestamp"] = outer["timestamp"] + 1
    else:
        outer["request_id"] = old["request_id"]
    # Re-sign the outer envelope after any edit to its signed body.
    outer["signature"] = "0x" + sign(contracts.signed_request_message(outer))
    with pytest.raises((ServiceError, contracts.ArenaContractError)):
        service.handle_claim(outer)
    assert store.allocations == 0


def test_normal_claim_keeps_its_original_wire_shape():
    from datetime import datetime, timezone
    now = datetime(2026, 9, 2, 1, 0, tzinfo=timezone.utc)
    service, store = claim_service(lambda: now)
    original = original_claim(now)
    response = service.handle_claim(original)
    assert response["status"] == "leased"
    assert "claim_recovery" not in response
    assert store.arguments[0]["request_id"] == original["request_id"]
    assert store.arguments[0]["request_hash"] == contracts.request_bytes_hash(original)


def test_exact_sql_replay_after_score_failure_creates_no_second_judgment_or_cost(database):
    """The original request ID cannot claim the confirmation attempt."""
    psycopg, dsn = database
    with psycopg.connect(**dsn) as connection:
        _seed(connection, kind="score", positions=1)
        lease = _claim(connection)
        assert lease["status"] == "leased"
        assert _claim(connection) == lease
        assert _complete(connection, lease, "judge_error")["status"] == "failed"
        assert _claim(connection) == lease
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT attempt,status FROM public.lab_arena_runs "
                "WHERE round_id=%s ORDER BY attempt", (SQL_ROUND,)
            )
            assert cursor.fetchall() == [(1, "failed"), (2, "pending")]
            cursor.execute(
                "SELECT count(*) FROM public.lab_arena_ledger WHERE round_id=%s",
                (SQL_ROUND,),
            )
            assert cursor.fetchone()[0] == 0
