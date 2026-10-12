"""Durable completion delivery reuses authenticated receipts, never sandboxes."""
from __future__ import annotations

import copy
import json
import os
import stat
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from lab_arena import contracts, runner as rn
from tests.lab_arena.test_lab_arena_runner import (
    BridgingRuntime, FakeApi, ROUND, RUNNER, completion_envelope,
    lease, make_config, sign, valid_company, verify,
)


class Api(FakeApi):
    def __init__(self, responses=(), leases=()):
        super().__init__(list(leases))
        self.responses = list(responses)
        self.attempts = []

    def complete(self, envelope):
        self.attempts.append(copy.deepcopy(envelope))
        response = self.responses.pop(0) if self.responses else {"status": "accepted"}
        if isinstance(response, BaseException):
            raise response
        return response


def config(tmp_path, api):
    result = make_config(tmp_path, api, BridgingRuntime(calls=0))
    result.completion_retry_seconds = ()
    result.accounting_open_retry_seconds = ()
    return result


def envelope(cfg, request_id="a" * 32):
    original = completion_envelope(output={"companies": [valid_company(1)]}, timestamp=int(cfg.clock().timestamp()))
    return contracts.build_signed_request(scope=contracts.SCOPE_COMPLETE, round_id=ROUND,
        hotkey=RUNNER.ss58_address, body=original["body"], timestamp=original["timestamp"],
        request_id=request_id, sign_message=sign)


def publish(worker, document):
    with worker._retain_completion(document) as path:
        return path


def test_real_success_pipeline_leaves_no_pending_files(tmp_path):
    api = Api(leases=[lease()])
    sandbox = BridgingRuntime(output={"companies": [valid_company(1)]}, calls=1)
    cfg = make_config(tmp_path, api, sandbox)
    cfg.work_dir.mkdir()
    worker = rn.Runner(cfg)
    try:
        assert worker.run_once() == 1
        assert worker.abandoned == 0
        assert len(sandbox.specs) == len(api.provider_frames) == len(api.attempts) == 1
        assert list((cfg.work_dir / "pending-completions").iterdir()) == []
    finally:
        worker.close()


@pytest.mark.parametrize("response", [rn.RunnerError("offline", http_status=503), {"status": "accounting_open"}])
def test_retry_exhaustion_survives_close_and_fresh_runner(tmp_path, response):
    first_api = Api([response])
    cfg = config(tmp_path, first_api)
    original = envelope(cfg)
    first = rn.Runner(cfg)
    with pytest.raises(rn.RunnerError):
        first._complete_with_retries(original)
    first.close()
    path = cfg.work_dir / "pending-completions" / (original["request_id"] + ".json")
    assert json.loads(path.read_text()) == original
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    second_api = Api()
    fresh_cfg = config(tmp_path, second_api)
    fresh = rn.Runner(fresh_cfg)
    try:
        assert fresh.run_once(max_claims=0) == 0
        assert second_api.attempts == [original]
        assert fresh_cfg.sandbox_runtime.specs == [] and second_api.provider_frames == []
        assert not path.exists()
    finally:
        fresh.close()


def test_lost_committed_reply_replays_same_identity_and_body_after_restart(tmp_path):
    api = Api([rn.RunnerError("lost200 reply", http_status=503)])
    cfg = config(tmp_path, api)
    original = envelope(cfg)
    first = rn.Runner(cfg)
    with pytest.raises(rn.RunnerError):
        first._complete_with_retries(original)
    first.close()
    replay_api = Api([{"status": "accepted", "idempotent": True}])
    fresh_cfg = config(tmp_path, replay_api)
    fresh_cfg.clock = lambda: cfg.clock() + timedelta(hours=1)
    second = rn.Runner(fresh_cfg)
    try:
        second.run_once(max_claims=0)
        replay = replay_api.attempts[0]
        assert replay["request_id"] == original["request_id"] and replay["body"] == original["body"]
        assert replay["timestamp"] != original["timestamp"]
        assert verify(replay["hotkey"], replay["signature"], contracts.signed_request_message(replay))
        assert second.completed[0]["result"]["idempotent"] is True
        assert fresh_cfg.sandbox_runtime.specs == []
    finally:
        second.close()


@pytest.mark.parametrize("status", ["stale", "rejected"])
def test_stale_and_rejected_receipts_are_archived_privately_without_paid_replay(tmp_path, status):
    api = Api([{"status": status}])
    cfg = config(tmp_path, api)
    worker = rn.Runner(cfg)
    original = envelope(cfg)
    pending = publish(worker, original)
    try:
        worker.run_once(max_claims=0)
        archived = pending.with_suffix("." + status + ".json")
        assert not pending.exists()
        assert json.loads(archived.read_text()) == original
        assert stat.S_IMODE(archived.stat().st_mode) == 0o600
        worker._pending_completion_next_poll = 0
        worker.run_once(max_claims=0)
        assert len(api.attempts) == 1 and cfg.sandbox_runtime.specs == []
    finally:
        worker.close()


@pytest.mark.parametrize("response", [{"status": "accounting_open"}, {"status": "other"}, rn.RunnerError("offline", http_status=503)])
def test_pending_poll_attempts_once_and_retains_ambiguous_outcome(tmp_path, response):
    api = Api([response])
    cfg = config(tmp_path, api)
    worker = rn.Runner(cfg)
    original = envelope(cfg)
    pending = publish(worker, original)
    try:
        worker.run_once(max_claims=0)
        assert api.attempts == [original]
        assert json.loads(pending.read_text()) == original
    finally:
        worker.close()


@pytest.mark.parametrize("change", ["body", "signature", "validator", "round", "mode", "symlink", "oversized", "json", "nested"])
def test_recovery_refuses_untrusted_files_before_resigning_or_upload(tmp_path, change):
    api = Api()
    cfg = config(tmp_path, api)
    worker = rn.Runner(cfg)
    original = envelope(cfg)
    path = publish(worker, original)
    tampered = copy.deepcopy(original)
    if change == "body":
        tampered["body"]["output"]["companies"][0]["company_name"] = "Altered"
    elif change == "signature":
        tampered["signature"] = "0x" + "0" * 128
    elif change in {"validator", "round"}:
        from bittensor_wallet import Keypair
        keypair = Keypair.create_from_uri("//Foreign") if change == "validator" else RUNNER
        tampered = contracts.build_signed_request(scope=contracts.SCOPE_COMPLETE,
            round_id="other-round" if change == "round" else ROUND, hotkey=keypair.ss58_address,
            body=original["body"], timestamp=original["timestamp"], request_id=original["request_id"],
            sign_message=lambda message: keypair.sign(message.encode()).hex())
    path.write_text(json.dumps(tampered))
    if change == "mode":
        path.chmod(0o644)
    elif change == "symlink":
        saved = tmp_path / "outside.json"
        path.rename(saved)
        path.symlink_to(saved)
    elif change == "oversized":
        path.write_bytes(b"x" * (contracts.COMPLETION_REQUEST_LIMITS.max_total_bytes + 1))
    elif change == "json":
        path.write_text('{"unfinished":')
    elif change == "nested":
        path.write_text("[" * 2000 + "]" * 2000)
    cfg.clock = lambda: datetime(2026, 9, 2, 2, 0, tzinfo=timezone.utc)
    try:
        worker.run_once(max_claims=0)
        assert api.attempts == [] and cfg.sandbox_runtime.specs == []
        assert path.exists()
    finally:
        worker.close()


def test_insecure_pending_directory_fails_before_claim_or_sandbox(tmp_path):
    api = Api(leases=[lease()])
    cfg = config(tmp_path, api)
    directory = cfg.work_dir / "pending-completions"
    directory.mkdir(parents=True)
    directory.chmod(0o755)
    worker = rn.Runner(cfg)
    try:
        with pytest.raises(rn.RunnerError, match="directory is not private"):
            worker.run_once()
        assert api.attempts == [] and cfg.sandbox_runtime.specs == [] and api.leases
    finally:
        worker.close()


def test_pending_queue_is_fair_bounded_and_normal_claims_proceed(tmp_path, monkeypatch):
    api = Api([rn.RunnerError("offline", http_status=503), {"status": "accepted"}])
    cfg = config(tmp_path, api)
    worker = rn.Runner(cfg)
    first = publish(worker, envelope(cfg, format(1, "032x")))
    second = publish(worker, envelope(cfg, format(2, "032x")))
    for i in range(3, 8):
        publish(worker, envelope(cfg, format(i, "032x")))
    clock = [0.0]
    monkeypatch.setattr(rn.time, "monotonic", lambda: clock[0])
    claims = []
    worker.claim_one = lambda round_id: claims.append(round_id) or {"status": "no_pending"}
    try:
        worker.run_once()
        assert len(api.attempts) == 1 and claims == [ROUND]
        assert first.exists() and second.exists()
        clock[0] += cfg.claim_poll_seconds
        worker.run_once()
        assert len(api.attempts) == 2 and claims == [ROUND, ROUND]
        assert first.exists() and not second.exists()
    finally:
        worker.close()


def test_file_lock_blocks_other_process_and_runner_during_original_upload(tmp_path):
    entered, finish = threading.Event(), threading.Event()
    cfg = config(tmp_path, Api())
    original = envelope(cfg)

    class BlockedApi(Api):
        def complete(self, document):
            entered.set()
            assert finish.wait(5)
            return {"status": "accepted"}

    cfg.api = BlockedApi()
    uploader = rn.Runner(cfg)
    other_api = Api()
    other = rn.Runner(config(tmp_path, other_api))
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(uploader._complete_with_retries, original)
        assert entered.wait(5)
        pending = cfg.work_dir / "pending-completions" / (original["request_id"] + ".json")
        script = "import fcntl,os,sys; f=os.open(sys.argv[1],os.O_RDONLY)\ntry: fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)\nexcept BlockingIOError: sys.exit(0)\nsys.exit(1)"
        result = subprocess.run([sys.executable, "-c", script, str(pending)], timeout=5)
        assert result.returncode == 0
        other.run_once(max_claims=0)
        assert other_api.attempts == []
        finish.set()
        assert future.result(timeout=5)["status"] == "accepted"
    uploader.close()
    other.close()


def test_different_completion_workers_do_not_share_a_global_delivery_lock(tmp_path):
    barrier = threading.Barrier(2)

    class ParallelApi(Api):
        def complete(self, document):
            barrier.wait(timeout=5)
            return {"status": "accepted"}

    cfg = config(tmp_path, ParallelApi())
    worker = rn.Runner(cfg)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(worker._complete_with_retries, envelope(cfg, token * 32)) for token in ("a", "b")]
            assert [future.result(timeout=5)["status"] for future in futures] == ["accepted", "accepted"]
        assert list((cfg.work_dir / "pending-completions").iterdir()) == []
    finally:
        worker.close()


def test_active_run_once_replays_exhausted_completion_before_healthy_work_ends(tmp_path, monkeypatch):
    second_active, second_finish = threading.Event(), threading.Event()
    clock = [0.0]
    attempts = []
    api = Api(leases=[lease("first", 0), lease("second", 1)])
    cfg = make_config(tmp_path, api, BridgingRuntime(calls=0), parallel=2)
    cfg.completion_retry_seconds = ()
    cfg.claim_poll_seconds = 0.01
    worker = rn.Runner(cfg)
    monkeypatch.setattr(rn.time, "monotonic", lambda: clock[0])

    def execute(leased, *args):
        if leased["run_id"] == "second":
            second_active.set()
            assert second_finish.wait(5)
        original = envelope(cfg, ("a" if leased["run_id"] == "first" else "b") * 32)
        original["body"]["run_id"] = leased["run_id"]
        return contracts.build_signed_request(scope=contracts.SCOPE_COMPLETE, round_id=ROUND,
            hotkey=RUNNER.ss58_address, body=original["body"], timestamp=original["timestamp"],
            request_id=original["request_id"], sign_message=sign)

    def complete(document):
        attempts.append(copy.deepcopy(document))
        if document["body"]["run_id"] == "first":
            if sum(row["body"]["run_id"] == "first" for row in attempts) == 1:
                assert second_active.wait(5)
                clock[0] = 31.0
                raise rn.RunnerError("offline", http_status=503)
            assert second_active.is_set() and not second_finish.is_set()
            second_finish.set()
        return {"status": "accepted"}

    worker._executor.execute = execute
    api.complete = complete
    try:
        assert worker.run_once(max_claims=2) == 2
        assert [row["body"]["run_id"] for row in attempts] == ["first", "first", "second"]
        assert attempts[0]["body"] == attempts[1]["body"] and attempts[0]["request_id"] == attempts[1]["request_id"]
        assert list((cfg.work_dir / "pending-completions").iterdir()) == []
    finally:
        second_finish.set()
        worker.close()


def test_actual_new_process_replays_private_saved_receipt(tmp_path):
    cfg = config(tmp_path, Api())
    worker = rn.Runner(cfg)
    original = envelope(cfg)
    pending = publish(worker, original)
    worker.close()
    script = '''
import json,sys
from pathlib import Path
from lab_arena import contracts,runner
from tests.lab_arena.completion_durability_test import Api,config
api=Api()
cfg=config(Path(sys.argv[1]),api)
worker=runner.Runner(cfg)
try:
    assert worker.run_once(max_claims=0)==0
    assert cfg.sandbox_runtime.specs==[] and api.provider_frames==[]
    request=api.attempts[0]
    print(json.dumps({"request_id":request["request_id"],"body_hash":contracts.document_hash(request["body"]),"calls":len(api.attempts)}))
finally:
    worker.close()
'''
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path)],
                            cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    saved = json.loads(result.stdout.splitlines()[-1])
    assert saved == {"request_id": original["request_id"], "body_hash": contracts.document_hash(original["body"]), "calls": 1}
    assert not pending.exists()


def test_existing_request_cannot_overwrite_different_immutable_body(tmp_path):
    api = Api()
    cfg = config(tmp_path, api)
    worker = rn.Runner(cfg)
    original = envelope(cfg)
    path = publish(worker, original)
    changed = envelope(cfg)
    changed["body"]["output"] = None
    changed = contracts.build_signed_request(scope=contracts.SCOPE_COMPLETE, round_id=ROUND,
        hotkey=RUNNER.ss58_address, body=changed["body"], timestamp=changed["timestamp"],
        request_id=changed["request_id"], sign_message=sign)
    try:
        with pytest.raises(rn.RunnerError, match="identity conflict"):
            worker._complete_with_retries(changed)
        assert json.loads(path.read_text()) == original and api.attempts == []
    finally:
        worker.close()


def test_corrupt_first_pending_file_does_not_starve_later_valid_file(tmp_path, monkeypatch):
    api = Api()
    cfg = config(tmp_path, api)
    worker = rn.Runner(cfg)
    bad = publish(worker, envelope(cfg, format(1, "032x")))
    good = publish(worker, envelope(cfg, format(2, "032x")))
    bad.write_text("malformed")
    clock = [0.0]
    monkeypatch.setattr(rn.time, "monotonic", lambda: clock[0])
    try:
        worker.run_once(max_claims=0)
        assert api.attempts == [] and bad.exists() and good.exists()
        clock[0] += cfg.claim_poll_seconds
        worker.run_once(max_claims=0)
        assert len(api.attempts) == 1 and bad.exists() and not good.exists()
    finally:
        worker.close()
