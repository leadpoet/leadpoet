import asyncio
from pathlib import Path

import pytest

from scripts import run_temporary_testnet401_weight_only_validator as weight_only


def test_weight_only_configuration_disables_axon_and_keeps_chain_submission():
    config = weight_only._configuration(
        wallet_name="validator",
        wallet_hotkey="default",
        wallet_path=Path("/run/leadpoet-testnet401/wallets"),
        state_path=Path("/run/leadpoet-testnet401/validator-state"),
    )

    assert config.netuid == 401
    assert config.subtensor.network == "test"
    assert config.subtensor.chain_endpoint == weight_only.CHAIN_ENDPOINT
    assert config.neuron.axon_off is True
    assert config.neuron.disable_set_weights is False


def test_weight_only_loop_calls_only_existing_automatic_weight_method():
    source = Path(weight_only.__file__).read_text(encoding="utf-8")

    assert "validator.submit_weights_at_epoch_end()" in source
    assert "validator.run()" not in source
    assert "start_epoch_monitor" not in source
    assert "serve_axon" not in source
    assert "process_gateway_validation_workflow" not in source
    assert "process_curation_requests_continuous" not in source
    assert ".set_weights(" not in source
    assert source.index("validator_module, validator = build_native_validator(") < source.index(
        "asyncio.run("
    )


class _Injection:
    def inject_async_subtensor(self, value):
        self.value = value


class _Module:
    reward_module = _Injection()

    class cloud_db_module:
        _VERIFY = _Injection()

    def __init__(self):
        self.closed = False

    def _close_subtensor_connection(self, _subtensor, *, source):
        assert source == "temporary_weight_only_shutdown"
        self.closed = True


class _Validator:
    def __init__(self, *, stop, failure=None):
        self.stop = stop
        self.failure = failure
        self.async_subtensor = object()
        self.subtensor = object()
        self.should_exit = False
        self.cleaned = False

    async def initialize_async_subtensor(self):
        return None

    async def submit_weights_at_epoch_end(self):
        if self.failure:
            raise self.failure
        self.stop.set()
        return False

    async def cleanup_async_subtensor(self):
        self.cleaned = True


def test_first_native_poll_creates_exact_private_readiness(tmp_path):
    module = _Module()
    readiness = tmp_path / "ready.json"

    async def scenario():
        stop = asyncio.Event()
        validator = _Validator(stop=stop)
        await weight_only.run_weight_only_loop(
            validator_module=module,
            validator=validator,
            readiness_path=readiness,
            stop_event=stop,
        )
        return validator

    validator = asyncio.run(scenario())

    assert readiness.read_text().endswith("\n")
    assert readiness.stat().st_mode & 0o777 == 0o600
    assert validator.cleaned is True
    assert module.closed is True


def test_first_native_poll_failure_exits_without_readiness(tmp_path):
    module = _Module()
    readiness = tmp_path / "ready.json"

    validator = None

    async def scenario():
        nonlocal validator
        stop = asyncio.Event()
        validator = _Validator(stop=stop, failure=RuntimeError("bad runtime"))
        await weight_only.run_weight_only_loop(
            validator_module=module,
            validator=validator,
            readiness_path=readiness,
            stop_event=stop,
        )

    with pytest.raises(RuntimeError, match="bad runtime"):
        asyncio.run(scenario())

    assert not readiness.exists()
    assert validator is not None
    assert validator.cleaned is True
    assert module.closed is True
