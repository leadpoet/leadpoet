"""Old journal recovery must not take priority over the current epoch."""
import json
import threading

import pytest

from lab_arena import contracts, validator
from lab_arena.weight_signer import ArenaWeightSignerError
from test_arena_validator import _orchestrator, _protected, _Signer


def _journal(path, **body):
    body["record_hash"] = contracts.document_hash(body)
    path.write_text(json.dumps(body) + "\n")


def test_current_epoch_runs_before_slow_prior_recovery():
    stop = threading.Event()
    order = []
    epoch = [10]

    class Orchestrator:
        def poll_prior_outcomes(self, value):
            order.append(("prior", value))
            epoch[0] = 11  # A slow archive read crosses an epoch boundary.

        def run_once(self, value):
            order.append(("current", value, epoch[0]))
            return "broadcast"

    class Runner:
        def run_once(self, **_kwargs):
            return 0

        def close(self):
            pass

    validator.run_validator_loops(
        orchestrator=Orchestrator(), runner_factory=Runner,
        epoch_supplier=lambda: epoch[0], stop=stop, once=True,
    )
    assert order[0] == ("current", 10, 10)


def test_prior_recovery_is_bounded_and_rotates_past_failure(tmp_path):
    for epoch in (5, 6, 7):
        _journal(tmp_path / f"epoch-{epoch}-signed.json", epoch=epoch)
    orchestrator = _orchestrator(tmp_path, _Signer(_protected(), []), [])
    seen = []

    def recover(signed):
        seen.append(signed["epoch"])
        if signed["epoch"] == 5:
            raise RuntimeError("unavailable archive")

    orchestrator._recover_protected_state = recover
    orchestrator._confirm = lambda _signed: None
    for expected in ([5], [5, 6], [5, 6, 7], [5, 6, 7, 5]):
        orchestrator.poll_prior_outcomes(10)
        assert seen == expected


def test_failed_report_retry_does_not_starve_other_prior_work(tmp_path):
    for epoch in (5, 6):
        _journal(tmp_path / f"epoch-{epoch}-signed.json", epoch=epoch)
    _journal(tmp_path / "epoch-5-outcome.json", epoch=5, reported=False)
    orchestrator = _orchestrator(tmp_path, _Signer(_protected(), []), [])
    seen = []

    def report(record):
        seen.append(("report", record["epoch"]))
        raise RuntimeError("gateway unavailable")

    orchestrator._report_outcome = report
    orchestrator._recover_protected_state = lambda signed: seen.append(("recover", signed["epoch"]))
    orchestrator._confirm = lambda _signed: None
    orchestrator.poll_prior_outcomes(10)
    assert seen == [("report", 5)]
    orchestrator.poll_prior_outcomes(10)
    assert seen == [("report", 5), ("recover", 6)]


def test_current_failure_still_allows_prior_recovery():
    events = []

    class Orchestrator:
        def run_once(self, epoch):
            events.append(("current", epoch))
            raise RuntimeError("current state unavailable")

        def poll_prior_outcomes(self, epoch):
            events.append(("prior", epoch))

    class Runner:
        def run_once(self, **_kwargs):
            return 0

        def close(self):
            pass

    epochs = iter((10, 11))
    validator.run_validator_loops(
        orchestrator=Orchestrator(), runner_factory=Runner,
        epoch_supplier=lambda: next(epochs), stop=threading.Event(), once=True,
    )
    assert events == [("current", 10), ("prior", 11)]


def test_already_reported_outcomes_use_no_recovery_slot(tmp_path):
    for epoch in (5, 6):
        _journal(tmp_path / f"epoch-{epoch}-signed.json", epoch=epoch)
    _journal(tmp_path / "epoch-5-outcome.json", epoch=5, reported=True)
    orchestrator = _orchestrator(tmp_path, _Signer(_protected(), []), [])
    seen = []

    def unexpected_report(_record):
        raise AssertionError("already reported")

    orchestrator._report_outcome = unexpected_report
    orchestrator._recover_protected_state = lambda signed: seen.append(signed["epoch"])
    orchestrator._confirm = lambda _signed: None
    orchestrator.poll_prior_outcomes(10)
    assert seen == [6]


def test_proved_prior_expiry_is_terminal_across_polls_and_restart(tmp_path):
    proof = {
        "status": "not_included_expired", "finalized": False,
        "finalized_head": {"block": 1200}, "finalized_nonce": 7,
    }
    signer = _Signer(_protected(), [proof])
    broadcasts = []
    orchestrator = _orchestrator(tmp_path, signer, broadcasts)
    assert orchestrator.run_once(9) == "broadcast"
    signed_path = orchestrator.paths.signed(9)
    signed_bytes = signed_path.read_bytes()
    signed = validator._read_hashed_json(signed_path)

    def unexpected_sign_or_report(_document):
        raise AssertionError("expired prior outcome needs no signature or report")

    signer.sign_arena_chain_outcome_v1 = unexpected_sign_or_report
    orchestrator.api.submit_chain_outcome = unexpected_sign_or_report
    assert orchestrator.poll_prior_outcomes(10) is True
    outcome = validator._read_hashed_json(orchestrator.paths.outcome(9))
    assert outcome is not None
    assert outcome == {
        "schema_version": "leadpoet.arena.validator_chain_outcome.v1",
        "state_hash": signed["state_hash"], "epoch": 9,
        "extrinsic_hash": signed["extrinsic_hash"], "outcome": proof,
        "report_document": None, "reported": True,
        "record_hash": outcome["record_hash"],
    }
    outcome_bytes = orchestrator.paths.outcome(9).read_bytes()
    assert orchestrator.poll_prior_outcomes(10) is True
    restarted = _orchestrator(tmp_path, signer, broadcasts)
    assert restarted.poll_prior_outcomes(10) is True
    assert signed_path.read_bytes() == signed_bytes
    assert orchestrator.paths.outcome(9).read_bytes() == outcome_bytes
    assert signer.prepares == signer.confirms == signer.recoveries == 1
    assert broadcasts == ["0xdeadbeef"]


@pytest.mark.parametrize("result", [
    {"status": "pending", "finalized": False},
    {"status": "included_pending_reveal", "finalized": False},
    {"status": "unproved", "finalized": False},
    "confirm_error",
    "recovery_error",
])
def test_unproved_prior_outcomes_remain_recoverable_without_new_submission(tmp_path, result):
    outcomes = [result, result] if isinstance(result, dict) else []
    signer = _Signer(_protected(), outcomes)
    broadcasts = []
    orchestrator = _orchestrator(tmp_path, signer, broadcasts)
    assert orchestrator.run_once(9) == "broadcast"
    signed_path = orchestrator.paths.signed(9)
    signed_bytes = signed_path.read_bytes()
    errors = []

    def fail(_request):
        errors.append(result)
        raise ArenaWeightSignerError("unproved prior outcome")

    if result == "confirm_error":
        signer.confirm_arena_weight_extrinsic_v1 = fail
    elif result == "recovery_error":
        signer.recover_arena_weight_extrinsic_v1 = fail
    expected_ok = isinstance(result, dict) and result["status"] in {
        "pending", "included_pending_reveal",
    }
    for _ in range(2):
        assert orchestrator.poll_prior_outcomes(10) is expected_ok
        assert not orchestrator.paths.outcome(9).exists()
        assert signed_path.read_bytes() == signed_bytes
    if isinstance(result, dict):
        assert signer.confirms == signer.recoveries == 2
    else:
        assert errors == [result, result]
    assert signer.prepares == 1
    assert broadcasts == ["0xdeadbeef"]
