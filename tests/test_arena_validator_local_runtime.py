import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from lab_arena import validator


class _ImmediateStop:
    """Event-compatible stop that yields without imposing the native 5s floor."""

    def __init__(self):
        self._event = threading.Event()

    def is_set(self):
        return self._event.is_set()

    def set(self):
        self._event.set()

    def wait(self, _timeout=None):
        return self._event.wait(0.001)


class _WeightRecorder:
    def __init__(self):
        self._lock = threading.Lock()
        self.runs = 0
        self.first_run = threading.Event()
        self.second_run = threading.Event()

    def poll_prior_outcomes(self, _epoch):
        return None

    def run_once(self, _epoch):
        with self._lock:
            self.runs += 1
            count = self.runs
        self.first_run.set()
        if count >= 2:
            self.second_run.set()
        return "state_unavailable"


@pytest.mark.parametrize("failure", [FileNotFoundError, RuntimeError, ImportError])
def test_weight_cycle_starts_before_failing_scorer_setup(failure):
    stop = _ImmediateStop()
    orchestrator = _WeightRecorder()
    observed = {}

    def runner_factory():
        assert orchestrator.first_run.wait(1)
        observed["scorer_setup_after_weight"] = True
        raise failure("scorer setup failed")

    thread = threading.Thread(
        target=validator.run_validator_loops,
        kwargs={
            "orchestrator": orchestrator,
            "runner_factory": runner_factory,
            "epoch_supplier": lambda: 1,
            "stop": stop,
            "poll_seconds": 5,
            "once": True,
        },
    )
    thread.start()
    thread.join(2)

    assert not thread.is_alive()
    assert observed == {"scorer_setup_after_weight": True}
    assert orchestrator.runs >= 1


def test_unexpected_scoring_exception_retries_while_weights_progress():
    stop = _ImmediateStop()
    orchestrator = _WeightRecorder()
    factory_calls = []
    successful_run = threading.Event()

    class _FailingRunner:
        def run_once(self, *, stop_event):
            assert stop_event is stop
            raise ValueError("unexpected scoring runtime failure")

        def close(self):
            return None

    class _SuccessfulRunner:
        def run_once(self, *, stop_event):
            assert stop_event is stop
            assert orchestrator.second_run.wait(1)
            successful_run.set()
            stop.set()
            return 1

        def close(self):
            return None

    def runner_factory():
        factory_calls.append(len(factory_calls) + 1)
        return _FailingRunner() if len(factory_calls) == 1 else _SuccessfulRunner()

    validator.run_validator_loops(
        orchestrator=orchestrator,
        runner_factory=runner_factory,
        epoch_supplier=lambda: 1,
        stop=stop,
        poll_seconds=5,
        once=False,
    )

    assert factory_calls[:2] == [1, 2]
    assert successful_run.is_set()
    assert orchestrator.runs >= 2


def test_successful_scoring_does_not_gate_weight_cycle():
    stop = _ImmediateStop()
    orchestrator = _WeightRecorder()
    scoring_finished = threading.Event()

    class _Runner:
        def run_once(self, *, stop_event):
            assert stop_event is stop
            scoring_finished.set()
            return 1

        def close(self):
            return None

    validator.run_validator_loops(
        orchestrator=orchestrator,
        runner_factory=lambda: _Runner(),
        epoch_supplier=lambda: 1,
        stop=stop,
        poll_seconds=5,
        once=True,
    )

    assert scoring_finished.is_set()
    assert orchestrator.runs >= 1


def test_no_scoring_work_still_runs_one_weight_cycle_with_once():
    stop = _ImmediateStop()
    orchestrator = _WeightRecorder()
    scoring_calls = []

    class _Runner:
        def run_once(self, *, stop_event):
            assert stop_event is stop
            scoring_calls.append(1)
            return 0

        def close(self):
            return None

    validator.run_validator_loops(
        orchestrator=orchestrator,
        runner_factory=lambda: _Runner(),
        epoch_supplier=lambda: 1,
        stop=stop,
        poll_seconds=5,
        once=True,
    )

    assert len(scoring_calls) == 1
    assert orchestrator.runs >= 1


def test_standard_wallet_arguments_are_parsed():
    args = validator._parser().parse_args(
        [
            "--wallet.name", "arena",
            "--wallet.hotkey", "validator",
            "--wallet.path", "/var/lib/wallets",
            "--check-only",
        ]
    )

    assert args.wallet_name == "arena"
    assert args.hotkey_name == "validator"
    assert args.wallet_path == "/var/lib/wallets"
    assert args.check_only is True


def test_local_hotkey_rejects_public_only_file(tmp_path):
    wallet_path = Path(tmp_path)
    hotkey_path = wallet_path / "arena" / "hotkeys" / "validator"
    hotkey_path.parent.mkdir(parents=True)
    hotkey_path.write_bytes(b"public-only-or-unusable")
    hotkey_path.chmod(0o644)
    args = SimpleNamespace(
        wallet_name="arena", hotkey_name="validator", wallet_path=str(wallet_path)
    )

    with pytest.raises(validator.ArenaValidatorError, match="private regular file"):
        validator.load_local_hotkey(args)


def test_local_hotkey_rejects_symlink(tmp_path):
    wallet_path = Path(tmp_path)
    hotkey_path = wallet_path / "arena" / "hotkeys" / "validator"
    hotkey_path.parent.mkdir(parents=True)
    target = wallet_path / "outside"
    target.write_bytes(b"private")
    target.chmod(0o600)
    hotkey_path.symlink_to(target)
    args = SimpleNamespace(
        wallet_name="arena", hotkey_name="validator", wallet_path=str(wallet_path)
    )

    with pytest.raises(validator.ArenaValidatorError, match="private regular file"):
        validator.load_local_hotkey(args)


def test_host_failure_recovers_without_stopping_weights_or_logging_secrets(capsys):
    from lab_arena.runtime_host import RuntimeHostError

    stop = _ImmediateStop()
    orchestrator = _WeightRecorder()
    factories = []

    class Runner:
        def run_once(self, *, stop_event):
            assert orchestrator.second_run.wait(1)
            stop_event.set()
            return 0

        def close(self):
            return None

    def factory():
        factories.append(1)
        if len(factories) == 1:
            raise RuntimeHostError("https://provider.invalid?key=secret-token",
                                   reason="runsc_missing", runsc_path=Path("/usr/local/bin/runsc"))
        return Runner()

    validator.run_validator_loops(orchestrator=orchestrator, runner_factory=factory,
                                  epoch_supplier=lambda: 1, stop=stop, once=False)
    output = capsys.readouterr()
    assert "phase=setup reason=runsc_missing" in output.err
    assert "/usr/local/bin/runsc" in output.err and "weights continue" in output.err
    assert "secret-token" not in output.err + output.out
    assert "scoring loop resumed" in output.out
    assert "sandbox completion is reported separately" in output.out
    assert orchestrator.runs >= 2


def test_generic_scoring_error_still_hides_provider_details(capsys):
    def factory():
        raise RuntimeError("https://provider.invalid?key=secret-token")

    orchestrator = _WeightRecorder()
    validator.run_validator_loops(orchestrator=orchestrator, runner_factory=factory,
                                  epoch_supplier=lambda: 1, stop=_ImmediateStop(), once=True)
    output = capsys.readouterr()
    assert "type=RuntimeError; weights continue" in output.err
    assert "secret-token" not in output.err
    assert orchestrator.runs >= 1
