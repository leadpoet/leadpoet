from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

import neurons.auditor_validator as auditor_module
import neurons.validator as validator_module
from Leadpoet.utils.subnet_epoch import STATEFUL_EPOCH_MODE


class _FakeSubtensor:
    def __init__(self, results, *, blocks=()):
        self._results = list(results)
        self._blocks = list(blocks)
        self.calls = []
        self.substrate = object()

    def set_weights(self, **kwargs):
        self.calls.append(dict(kwargs))
        return self._results.pop(0)

    def get_current_block(self):
        return self._blocks.pop(0)


class _SdkResponse:
    def __init__(self, success: bool, message: str):
        self.success = success
        self.message = message


def _primary(results, *, blocks=()):
    validator = validator_module.Validator.__new__(validator_module.Validator)
    validator.config = SimpleNamespace(netuid=71)
    validator.wallet = object()
    validator.subtensor = _FakeSubtensor(results, blocks=blocks)

    async def get_current_block_async():
        return validator.subtensor.get_current_block()

    validator.get_current_block_async = get_current_block_async
    validator._validate_durable_epoch_runtime_lifecycle = (
        lambda _epoch_id, *, force_refresh: {"lifecycle_state": "legacy_open"}
    )
    validator._last_weight_submission_epoch = None
    return validator


def _auditor(results, *, blocks=()):
    auditor = auditor_module.AuditorValidator.__new__(auditor_module.AuditorValidator)
    auditor.config = SimpleNamespace(netuid=71)
    auditor.wallet = object()
    auditor.subtensor = _FakeSubtensor(results, blocks=blocks)
    auditor._validate_durable_epoch_runtime_lifecycle = (
        lambda _epoch_id, *, force_refresh: {"lifecycle_state": "legacy_open"}
    )
    auditor.last_submitted_epoch = None
    return auditor


class _AuthoritativeContext:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.extrinsic_signature_results = [{"receipt": "signed"}]

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None


def _authoritative_args():
    return {
        "weight_authorization_id": "sha256:" + "1" * 64,
        "weight_submission_event_hash": "sha256:" + "2" * 64,
    }


def test_primary_retries_false_tuples_until_true(monkeypatch, capsys):
    validator = _primary(
        [(False, "rejected-one"), (False, "rejected-two"), (True, "accepted")],
        blocks=[345, 346, 347],
    )
    sleeps = []

    async def no_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(validator_module.asyncio, "sleep", no_sleep)
    monkeypatch.setattr(
        validator_module, "AuthoritativeSetWeightsContextV2", _AuthoritativeContext
    )
    result = asyncio.run(
        validator._set_weights_until_epoch_end(
            epoch_id=0,
            uids=[1, 2],
            weights=[0.25, 0.75],
            **_authoritative_args(),
        )
    )

    assert result is True
    assert sleeps == [12, 12]
    assert len(validator.subtensor.calls) == 3
    assert all(call["uids"] == [1, 2] for call in validator.subtensor.calls)
    assert all(call["weights"] == [0.25, 0.75] for call in validator.subtensor.calls)
    assert all(call["netuid"] == 71 for call in validator.subtensor.calls)
    assert all(call["mechid"] == 0 for call in validator.subtensor.calls)
    assert validator._last_weight_submission_epoch is None
    assert validator._last_weight_extrinsic_receipts_v2 == [
        {"receipt": "signed"}
    ]
    output = capsys.readouterr().out
    assert "rejected-one" in output
    assert "rejected-two" in output


def test_primary_stops_before_retry_after_epoch_rollover(monkeypatch):
    validator = _primary([(False, "rejected")], blocks=[359, 360])

    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr(validator_module.asyncio, "sleep", no_sleep)
    monkeypatch.setattr(
        validator_module, "AuthoritativeSetWeightsContextV2", _AuthoritativeContext
    )
    result = asyncio.run(
        validator._set_weights_until_epoch_end(
            epoch_id=0,
            uids=[0],
            weights=[1.0],
            **_authoritative_args(),
        )
    )

    assert result is False
    assert len(validator.subtensor.calls) == 1
    assert validator._last_weight_submission_epoch is None


def test_primary_best_head_rollover_vetoes_stale_finalized_epoch_before_signing(
    monkeypatch,
):
    validator = _primary([(True, "must-not-submit")])
    validator._epoch_mode = STATEFUL_EPOCH_MODE
    finalized_state = SimpleNamespace(
        workflow_epoch_id=100,
        subnet_epoch_index=10,
        current_block=719,
        blocks_remaining=1,
    )
    best_state = SimpleNamespace(
        workflow_epoch_id=101,
        subnet_epoch_index=11,
        current_block=720,
        blocks_remaining=360,
    )

    async def read_finalized():
        return finalized_state

    async def read_best():
        return best_state

    class ForbiddenSigningContext:
        def __init__(self, **_kwargs):
            raise AssertionError("stale state must be vetoed before signing")

    validator._get_epoch_state_async = read_finalized
    validator._get_best_epoch_state_async = read_best
    monkeypatch.setattr(
        validator_module,
        "AuthoritativeSetWeightsContextV2",
        ForbiddenSigningContext,
    )

    result = asyncio.run(
        validator._set_weights_until_epoch_end(
            epoch_id=100,
            subnet_epoch_index=10,
            uids=[0],
            weights=[1.0],
            **_authoritative_args(),
        )
    )

    assert result is False
    assert validator.subtensor.calls == []


def test_primary_due_best_head_vetoes_signature_before_index_advances(monkeypatch):
    validator = _primary([(True, "must-not-submit")])
    validator._epoch_mode = STATEFUL_EPOCH_MODE
    finalized_state = SimpleNamespace(
        workflow_epoch_id=100,
        subnet_epoch_index=10,
        current_block=719,
        blocks_remaining=1,
    )
    due_best_state = SimpleNamespace(
        workflow_epoch_id=100,
        subnet_epoch_index=10,
        current_block=720,
        blocks_remaining=0,
    )

    async def read_finalized():
        return finalized_state

    async def read_best():
        return due_best_state

    class ForbiddenSigningContext:
        def __init__(self, **_kwargs):
            raise AssertionError("a due best head must be vetoed before signing")

    validator._get_epoch_state_async = read_finalized
    validator._get_best_epoch_state_async = read_best
    monkeypatch.setattr(
        validator_module,
        "AuthoritativeSetWeightsContextV2",
        ForbiddenSigningContext,
    )

    result = asyncio.run(
        validator._set_weights_until_epoch_end(
            epoch_id=100,
            subnet_epoch_index=10,
            uids=[0],
            weights=[1.0],
            **_authoritative_args(),
        )
    )

    assert result is False
    assert validator.subtensor.calls == []


def test_primary_best_head_rollover_stops_retry_while_finalized_remains_old(
    monkeypatch,
):
    validator = _primary([(False, "rejected")])
    validator._epoch_mode = STATEFUL_EPOCH_MODE
    finalized_state = SimpleNamespace(
        workflow_epoch_id=100,
        subnet_epoch_index=10,
        current_block=719,
        blocks_remaining=1,
    )
    best_states = iter(
        [
            SimpleNamespace(
                workflow_epoch_id=100,
                subnet_epoch_index=10,
                current_block=719,
                blocks_remaining=1,
            ),
            SimpleNamespace(
                workflow_epoch_id=101,
                subnet_epoch_index=11,
                current_block=720,
                blocks_remaining=360,
            ),
        ]
    )

    async def read_finalized():
        return finalized_state

    async def read_best():
        return next(best_states)

    async def no_sleep(_seconds):
        return None

    validator._get_epoch_state_async = read_finalized
    validator._get_best_epoch_state_async = read_best
    monkeypatch.setattr(validator_module.asyncio, "sleep", no_sleep)
    monkeypatch.setattr(
        validator_module, "AuthoritativeSetWeightsContextV2", _AuthoritativeContext
    )

    result = asyncio.run(
        validator._set_weights_until_epoch_end(
            epoch_id=100,
            subnet_epoch_index=10,
            uids=[0],
            weights=[1.0],
            **_authoritative_args(),
        )
    )

    assert result is False
    assert len(validator.subtensor.calls) == 1


def test_primary_accepts_v10_extrinsic_response(monkeypatch):
    validator = _primary([_SdkResponse(True, "accepted")], blocks=[1080])

    async def unexpected_sleep(_seconds):
        raise AssertionError("successful submission must not sleep")

    monkeypatch.setattr(validator_module.asyncio, "sleep", unexpected_sleep)
    monkeypatch.setattr(
        validator_module, "AuthoritativeSetWeightsContextV2", _AuthoritativeContext
    )
    result = asyncio.run(
        validator._set_weights_until_epoch_end(
            epoch_id=3,
            uids=[9],
            weights=[1.0],
            **_authoritative_args(),
        )
    )

    assert result is True
    assert len(validator.subtensor.calls) == 1


def test_primary_database_outage_fails_before_signing_or_sdk(monkeypatch):
    validator = _primary([(True, "must-not-submit")])

    def unavailable(_epoch_id, *, force_refresh):
        assert force_refresh is True
        raise RuntimeError("supabase unavailable")

    validator._validate_durable_epoch_runtime_lifecycle = unavailable

    class ForbiddenSigningContext:
        def __init__(self, **_kwargs):
            raise AssertionError("database outage must fail before signing")

    monkeypatch.setattr(
        validator_module,
        "AuthoritativeSetWeightsContextV2",
        ForbiddenSigningContext,
    )

    result = asyncio.run(
        validator._set_weights_until_epoch_end(
            epoch_id=0,
            uids=[0],
            weights=[1.0],
            **_authoritative_args(),
        )
    )

    assert result is False
    assert validator.subtensor.calls == []


def test_primary_legacy_startup_checks_computed_epoch_against_durable_fence():
    validator = validator_module.Validator.__new__(validator_module.Validator)
    validator._epoch_mode = validator_module.LEGACY_EPOCH_MODE
    validator.subtensor = SimpleNamespace(block=36_360)
    observed = []

    def fenced(epoch_id, *, force_refresh):
        observed.append((epoch_id, force_refresh))
        raise RuntimeError("reserved settlement ordinal")

    validator._validate_durable_epoch_runtime_lifecycle = fenced

    with pytest.raises(RuntimeError, match="reserved settlement ordinal"):
        validator._validate_durable_epoch_runtime_startup()

    assert observed == [(101, True)]


def test_primary_stateful_startup_requires_receipt_backed_active_mapping(
    monkeypatch,
):
    from gateway.utils import epoch as epoch_utils

    cutover = SimpleNamespace(first_settlement_epoch_id=101)
    validator = validator_module.Validator.__new__(validator_module.Validator)
    validator.config = SimpleNamespace(
        netuid=71,
        subtensor=SimpleNamespace(network="finney"),
    )
    validator._epoch_mode = STATEFUL_EPOCH_MODE
    validator._epoch_cutover = cutover
    observed = []
    validator._uses_production_epoch_cutover_authority = lambda: True

    monkeypatch.setattr(
        epoch_utils,
        "validate_stateful_cutover_authority",
        lambda value, **_kwargs: observed.append(("authority", value)),
    )
    validator._validate_durable_epoch_runtime_lifecycle = (
        lambda epoch_id, *, force_refresh: observed.append(
            ("lifecycle", epoch_id, force_refresh)
        )
    )

    validator._validate_durable_epoch_runtime_startup()

    assert observed == [
        ("authority", cutover),
        ("lifecycle", 101, True),
    ]


@pytest.mark.parametrize(
    ("network", "netuid"),
    [("test", 71), ("finney", 72)],
)
def test_primary_non_production_runtime_does_not_bind_sn71_cutover_singleton(
    monkeypatch,
    network,
    netuid,
):
    from gateway.utils import epoch as epoch_utils

    validator = validator_module.Validator.__new__(validator_module.Validator)
    validator.config = SimpleNamespace(
        netuid=netuid,
        subtensor=SimpleNamespace(network=network),
    )
    validator._epoch_mode = validator_module.LEGACY_EPOCH_MODE
    monkeypatch.setattr(
        epoch_utils,
        "validate_epoch_runtime_lifecycle",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("non-production runtime must not read production DB")
        ),
    )

    assert validator._validate_durable_epoch_runtime_lifecycle(
        101,
        force_refresh=True,
    )["lifecycle_state"] == "legacy_open"


def test_primary_finney_sn71_always_checks_durable_cutover_singleton(monkeypatch):
    from gateway.utils import epoch as epoch_utils

    validator = validator_module.Validator.__new__(validator_module.Validator)
    validator.config = SimpleNamespace(
        netuid=71,
        subtensor=SimpleNamespace(network="finney"),
    )
    validator._epoch_mode = validator_module.LEGACY_EPOCH_MODE
    observed = []

    def validate(**kwargs):
        observed.append(kwargs)
        return {"lifecycle_state": "legacy_open"}

    monkeypatch.setattr(epoch_utils, "validate_epoch_runtime_lifecycle", validate)

    validator._validate_durable_epoch_runtime_lifecycle(101, force_refresh=True)

    assert observed == [
        {
            "mode": validator_module.LEGACY_EPOCH_MODE,
            "epoch_id": 101,
            "cutover": None,
            "force_refresh": True,
            "network": "finney",
            "netuid": 71,
        }
    ]


def test_primary_lifecycle_transition_after_signing_blocks_sdk(monkeypatch):
    validator = _primary([(True, "must-not-submit")])
    authority_checks = []
    finalized_state = SimpleNamespace(workflow_epoch_id=0)

    def lifecycle(_epoch_id, *, force_refresh):
        assert force_refresh is True
        authority_checks.append("checked")
        if len(authority_checks) == 2:
            raise RuntimeError("stateful_active")
        return {"lifecycle_state": "legacy_open"}

    async def read_finalized():
        return finalized_state

    validator._validate_durable_epoch_runtime_lifecycle = lifecycle
    validator._get_epoch_state_async = read_finalized
    monkeypatch.setattr(
        validator_module,
        "AuthoritativeSetWeightsContextV2",
        _AuthoritativeContext,
    )

    result = asyncio.run(
        validator._set_weights_until_epoch_end(
            epoch_id=0,
            uids=[0],
            weights=[1.0],
            **_authoritative_args(),
        )
    )

    assert result is False
    assert authority_checks == ["checked", "checked"]
    assert validator.subtensor.calls == []


def test_auditor_retries_false_tuples_until_true(monkeypatch, capsys):
    auditor = _auditor(
        [(False, "rejected-one"), (False, "rejected-two"), (True, "accepted")],
        blocks=[1065, 1066, 1067],
    )
    sleeps = []
    monkeypatch.setattr(auditor_module.time, "sleep", sleeps.append)

    result = auditor._set_weights_until_epoch_end(
        epoch_id=2,
        uids=[3, 4],
        weights=[0.4, 0.6],
    )

    assert result is True
    assert sleeps == [12, 12]
    assert len(auditor.subtensor.calls) == 3
    assert all(call["uids"] == [3, 4] for call in auditor.subtensor.calls)
    assert all(call["weights"] == [0.4, 0.6] for call in auditor.subtensor.calls)
    assert all(call["netuid"] == 71 for call in auditor.subtensor.calls)
    assert all(call["mechid"] == 0 for call in auditor.subtensor.calls)
    assert auditor.last_submitted_epoch is None
    output = capsys.readouterr().out
    assert "rejected-one" in output
    assert "rejected-two" in output


def test_auditor_accepts_v10_extrinsic_response(monkeypatch):
    auditor = _auditor([_SdkResponse(True, "accepted")], blocks=[1065])

    def unexpected_sleep(_seconds):
        raise AssertionError("successful submission must not sleep")

    monkeypatch.setattr(auditor_module.time, "sleep", unexpected_sleep)

    result = auditor._set_weights_until_epoch_end(
        epoch_id=2,
        uids=[3],
        weights=[1.0],
    )

    assert result is True
    assert auditor.subtensor.calls == [
        {
            "netuid": 71,
            "wallet": auditor.wallet,
            "uids": [3],
            "weights": [1.0],
            "wait_for_finalization": True,
            "mechid": 0,
        }
    ]


def test_auditor_lifecycle_transition_before_sdk_fails_closed(monkeypatch):
    auditor = _auditor([(True, "must-not-submit")])
    authority_checks = []
    state = SimpleNamespace(workflow_epoch_id=2)

    def lifecycle(_epoch_id, *, force_refresh):
        assert force_refresh is True
        authority_checks.append("checked")
        if len(authority_checks) == 2:
            raise RuntimeError("stateful_active")
        return {"lifecycle_state": "legacy_open"}

    auditor._validate_durable_epoch_runtime_lifecycle = lifecycle
    auditor._read_epoch_state = lambda: state
    monkeypatch.setattr(auditor_module.time, "sleep", lambda _seconds: None)

    result = auditor._set_weights_until_epoch_end(
        epoch_id=2,
        uids=[0],
        weights=[1.0],
    )

    assert result is False
    assert authority_checks == ["checked", "checked"]
    assert auditor.subtensor.calls == []


def test_auditor_legacy_startup_checks_computed_epoch_against_durable_fence():
    auditor = auditor_module.AuditorValidator.__new__(
        auditor_module.AuditorValidator
    )
    auditor.epoch_mode = auditor_module.LEGACY_EPOCH_MODE
    auditor.subtensor = SimpleNamespace(get_current_block=lambda: 36_360)
    observed = []

    def fenced(epoch_id, *, force_refresh):
        observed.append((epoch_id, force_refresh))
        raise RuntimeError("reserved settlement ordinal")

    auditor._validate_durable_epoch_runtime_lifecycle = fenced

    with pytest.raises(RuntimeError, match="reserved settlement ordinal"):
        auditor._validate_durable_epoch_runtime_startup()

    assert observed == [(101, True)]


@pytest.mark.parametrize(
    ("network", "netuid", "expects_check"),
    [("test", 71, False), ("finney", 72, False), ("finney", 71, True)],
)
def test_auditor_cutover_singleton_is_scoped_to_finney_sn71(
    monkeypatch,
    network,
    netuid,
    expects_check,
):
    from gateway.utils import epoch as epoch_utils

    auditor = auditor_module.AuditorValidator.__new__(
        auditor_module.AuditorValidator
    )
    auditor.config = SimpleNamespace(
        netuid=netuid,
        subtensor=SimpleNamespace(network=network),
    )
    auditor.epoch_mode = auditor_module.LEGACY_EPOCH_MODE
    observed = []

    def validate(**kwargs):
        observed.append(kwargs)
        return {"lifecycle_state": "legacy_open"}

    monkeypatch.setattr(epoch_utils, "validate_epoch_runtime_lifecycle", validate)

    state = auditor._validate_durable_epoch_runtime_lifecycle(
        101,
        force_refresh=True,
    )

    assert state["lifecycle_state"] == "legacy_open"
    assert bool(observed) is expects_check


def test_auditor_stops_before_retry_after_epoch_rollover(monkeypatch):
    auditor = _auditor([(False, "rejected")], blocks=[1079, 1080])
    monkeypatch.setattr(auditor_module.time, "sleep", lambda _seconds: None)

    result = auditor._set_weights_until_epoch_end(
        epoch_id=2,
        uids=[0],
        weights=[1.0],
    )

    assert result is False
    assert len(auditor.subtensor.calls) == 1
    assert auditor.last_submitted_epoch is None


def test_auditor_best_head_rollover_stops_retry_while_finalized_remains_old(
    monkeypatch,
):
    auditor = _auditor([(False, "rejected")])
    auditor.epoch_mode = STATEFUL_EPOCH_MODE
    finalized_state = SimpleNamespace(
        workflow_epoch_id=100,
        subnet_epoch_index=10,
        current_block=719,
        blocks_remaining=1,
    )
    best_states = iter(
        [
            SimpleNamespace(
                workflow_epoch_id=100,
                subnet_epoch_index=10,
                current_block=719,
                blocks_remaining=1,
            ),
            SimpleNamespace(
                workflow_epoch_id=101,
                subnet_epoch_index=11,
                current_block=720,
                blocks_remaining=360,
            ),
        ]
    )
    auditor._read_epoch_state = lambda: finalized_state
    auditor._read_best_epoch_state = lambda: next(best_states)
    monkeypatch.setattr(auditor_module.time, "sleep", lambda _seconds: None)

    result = auditor._set_weights_until_epoch_end(
        epoch_id=100,
        subnet_epoch_index=10,
        uids=[0],
        weights=[1.0],
    )

    assert result is False
    assert len(auditor.subtensor.calls) == 1


def test_all_active_submission_paths_use_epoch_bounded_helpers():
    primary_source = validator_module.Path(validator_module.__file__).read_text(encoding="utf-8")
    auditor_source = auditor_module.Path(auditor_module.__file__).read_text(encoding="utf-8")

    assert primary_source.count("await self._publish_and_set_weights(") == 3
    assert primary_source.count("await self._authorize_and_set_weights_v2(") == 1
    assert primary_source.count("await self._set_weights_until_epoch_end(") == 2
    assert "_set_legacy_weights_until_epoch_end" not in primary_source
    assert auditor_source.count("self._set_weights_until_epoch_end(") == 1
    assert "submit_burn_weights_to_uid0" not in auditor_source
    assert primary_source.count("self.subtensor.set_weights(") == 1
    assert auditor_source.count("self.subtensor.set_weights(") == 1
    assert "_submit_weights_to_gateway" not in primary_source
    assert "_submit_weights_v2" not in primary_source
    assert "VALIDATOR_ATTESTED_WEIGHT_MODE" not in primary_source
    assert "VALIDATOR_WEIGHT_PROTOCOL" in primary_source
