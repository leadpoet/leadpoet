from __future__ import annotations

import asyncio
import errno
from types import SimpleNamespace

import pytest

import neurons.auditor_validator as auditor_module
import neurons.validator as validator_module


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

    def get_subnet_hyperparameters(self, _netuid, block=None):
        raise AssertionError(
            "the Bittensor 9 compatibility wrapper must replace this method"
        )


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
        lambda *, force_refresh: {"lifecycle_state": "stateful_active"}
    )
    validator._last_weight_submission_epoch = None
    return validator


def _auditor(results, *, blocks=()):
    auditor = auditor_module.AuditorValidator.__new__(auditor_module.AuditorValidator)
    auditor.config = SimpleNamespace(netuid=71)
    auditor.wallet = object()
    auditor.subtensor = _FakeSubtensor(results, blocks=blocks)
    auditor._validate_durable_epoch_runtime_lifecycle = (
        lambda *, force_refresh: {"lifecycle_state": "stateful_active"}
    )
    auditor.last_submitted_epoch = None
    auditor.uid = 9
    finalized_state = {
        "block_hash": "0x" + "1" * 64,
        "last_update": 100,
        "weights": [],
    }

    def read_finalized_weight_submission_state():
        return dict(finalized_state)

    original_set_weights = auditor.subtensor.set_weights

    def set_weights(**kwargs):
        response = original_set_weights(**kwargs)
        success = getattr(response, "success", None)
        if success is None and isinstance(response, (tuple, list)):
            success = response[0]
        if success is True:
            # Finney commit-reveal advances LastUpdate when the encrypted
            # commitment finalizes. The public Weights row remains the prior
            # vector until the scheduled reveal at the next epoch boundary.
            finalized_state["last_update"] += 1
        return response

    auditor.subtensor.set_weights = set_weights
    auditor._read_finalized_weight_submission_state = (
        read_finalized_weight_submission_state
    )
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


def test_primary_serializes_concurrent_weight_submission_triggers():
    validator = validator_module.Validator.__new__(validator_module.Validator)
    started = asyncio.Event()
    release = asyncio.Event()
    calls = []

    async def submit_locked():
        calls.append("started")
        started.set()
        await release.wait()
        return True

    validator._submit_weights_at_epoch_end_locked = submit_locked

    async def run():
        first = asyncio.create_task(validator.submit_weights_at_epoch_end())
        await started.wait()
        duplicate = await validator.submit_weights_at_epoch_end()
        release.set()
        return await first, duplicate

    first, duplicate = asyncio.run(run())

    assert first is True
    assert duplicate is False
    assert calls == ["started"]


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

    async def current(**_kwargs):
        return True

    validator._weight_submission_epoch_is_current = current
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
    states = iter([True, False])

    async def current(**_kwargs):
        return next(states)

    validator._weight_submission_epoch_is_current = current
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

    async def current(**_kwargs):
        return True

    validator._weight_submission_epoch_is_current = current
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


def test_primary_stateful_startup_requires_receipt_backed_active_mapping(
    monkeypatch,
):
    from gateway.utils import epoch as epoch_utils

    cutover = SimpleNamespace(
        first_settlement_epoch_id=101,
        mapping_hash="sha256:" + "1" * 64,
    )
    validator = validator_module.Validator.__new__(validator_module.Validator)
    validator.config = SimpleNamespace(
        netuid=71,
        subtensor=SimpleNamespace(network="finney"),
    )
    validator._epoch_cutover = cutover
    observed = []
    validator._uses_production_epoch_cutover_authority = lambda: True

    monkeypatch.setattr(
        epoch_utils,
        "validate_stateful_cutover_authority",
        lambda value, **_kwargs: observed.append(("authority", value)),
    )
    validator._validate_durable_epoch_runtime_lifecycle = (
        lambda *, force_refresh: observed.append(
            ("lifecycle", force_refresh)
        )
    )

    validator._validate_durable_epoch_runtime_startup()

    assert observed == [
        ("authority", cutover),
        ("lifecycle", True),
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
    validator._epoch_cutover = SimpleNamespace(mapping_hash="sha256:" + "1" * 64)
    monkeypatch.setattr(
        epoch_utils,
        "validate_epoch_runtime_lifecycle",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("non-production runtime must not read production DB")
        ),
    )

    assert validator._validate_durable_epoch_runtime_lifecycle(
        force_refresh=True,
    )["lifecycle_state"] == "stateful_manifest_only"


def test_primary_finney_sn71_always_checks_durable_cutover_singleton(monkeypatch):
    from gateway.utils import epoch as epoch_utils

    validator = validator_module.Validator.__new__(validator_module.Validator)
    validator.config = SimpleNamespace(
        netuid=71,
        subtensor=SimpleNamespace(network="finney"),
    )
    cutover = SimpleNamespace(mapping_hash="sha256:" + "1" * 64)
    validator._epoch_cutover = cutover
    observed = []

    def validate(**kwargs):
        observed.append(kwargs)
        return {"lifecycle_state": "stateful_active"}

    monkeypatch.setattr(epoch_utils, "validate_epoch_runtime_lifecycle", validate)

    validator._validate_durable_epoch_runtime_lifecycle(force_refresh=True)

    assert observed == [
        {
            "cutover": cutover,
            "force_refresh": True,
            "network": "finney",
            "netuid": 71,
        }
    ]


def test_primary_and_auditor_terminal_lifecycle_survives_100_epoch_outage(
    monkeypatch,
):
    from Leadpoet.utils.subnet_epoch import SubnetEpochCutover
    from gateway.utils import epoch as epoch_utils

    cutover = SubnetEpochCutover(
        network_genesis_hash="0x" + "1" * 64,
        netuid=71,
        cutover_block=36_396,
        cutover_block_hash="0x" + "2" * 64,
        first_subnet_epoch_index=23_928,
        first_settlement_epoch_id=101,
        last_legacy_epoch_id=100,
    )
    active = {
        "lifecycle_state": "stateful_active",
        "mapping_hash": cutover.mapping_hash,
        "last_legacy_epoch_id": cutover.last_legacy_epoch_id,
        "first_settlement_epoch_id": cutover.first_settlement_epoch_id,
    }
    public_reads = {"n": 0}

    def active_then_unavailable(*_args, **_kwargs):
        public_reads["n"] += 1
        if public_reads["n"] in {1, 2}:
            return [dict(active)]
        raise TimeoutError("PostgREST timed out")

    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
    monkeypatch.setattr(epoch_utils, "call_public_rpc", active_then_unavailable)

    validator = validator_module.Validator.__new__(validator_module.Validator)
    validator.config = SimpleNamespace(
        netuid=71,
        subtensor=SimpleNamespace(network="finney"),
    )
    validator._epoch_cutover = cutover
    monkeypatch.setattr(epoch_utils, "_cutover_state_cache", None)
    monkeypatch.setattr(epoch_utils, "_validated_terminal_cutover_state", None)
    for _ in range(100):
        assert validator._validate_durable_epoch_runtime_lifecycle(
            force_refresh=True,
        ) == active

    auditor = auditor_module.AuditorValidator.__new__(
        auditor_module.AuditorValidator
    )
    auditor.config = SimpleNamespace(
        netuid=71,
        subtensor=SimpleNamespace(network="finney"),
    )
    auditor.epoch_cutover = cutover
    monkeypatch.setattr(epoch_utils, "_cutover_state_cache", None)
    monkeypatch.setattr(epoch_utils, "_validated_terminal_cutover_state", None)
    for _ in range(100):
        assert auditor._validate_durable_epoch_runtime_lifecycle(
            force_refresh=True,
        ) == active

    assert public_reads["n"] == 2


def test_primary_lifecycle_transition_after_signing_blocks_sdk(monkeypatch):
    validator = _primary([(True, "must-not-submit")])
    authority_checks = []
    finalized_state = SimpleNamespace(workflow_epoch_id=0)

    def lifecycle(*, force_refresh):
        assert force_refresh is True
        authority_checks.append("checked")
        if len(authority_checks) == 2:
            raise RuntimeError("stateful_active")
        return {"lifecycle_state": "stateful_active"}

    async def current(**_kwargs):
        lifecycle(force_refresh=True)
        return True

    validator._validate_durable_epoch_runtime_lifecycle = lifecycle
    validator._weight_submission_epoch_is_current = current
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
    state = SimpleNamespace(
        workflow_epoch_id=2,
        subnet_epoch_index=None,
        current_block=1065,
        blocks_remaining=1,
    )
    auditor._read_epoch_state = lambda: state
    auditor._read_best_epoch_state = lambda: state

    result = auditor._set_weights_until_epoch_end(
        epoch_id=2,
        uids=[3, 4],
        weights=[0.4, 0.6],
        expected_weights_u16=[43690, 65535],
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


def test_auditor_accepts_v10_finalized_commit_before_weight_reveal(
    monkeypatch,
    capsys,
    caplog,
):
    auditor = _auditor([_SdkResponse(True, "accepted")], blocks=[1065])
    caplog.set_level("INFO", logger=auditor_module.__name__)

    def unexpected_sleep(_seconds):
        raise AssertionError("successful submission must not sleep")

    monkeypatch.setattr(auditor_module.time, "sleep", unexpected_sleep)
    state = SimpleNamespace(
        workflow_epoch_id=2,
        subnet_epoch_index=None,
        current_block=1065,
        blocks_remaining=1,
    )
    auditor._read_epoch_state = lambda: state
    auditor._read_best_epoch_state = lambda: state

    result = auditor._set_weights_until_epoch_end(
        epoch_id=2,
        uids=[3],
        weights=[1.0],
        expected_weights_u16=[65535],
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
    output = capsys.readouterr().out
    assert "visible chain vector remains sealed" in output
    assert '"confirmation_stage":"timelocked_commit_finalized"' in caplog.text
    assert '"visible_vector_state":"reveal_pending"' in caplog.text
    assert '"event":"submission_failed"' not in caplog.text


def test_auditor_reconnect_is_transactional(monkeypatch):
    closed = []

    class Source:
        def __init__(self, uid):
            self.uid = uid

        def get_uid_for_hotkey_on_subnet(self, **_kwargs):
            return self.uid

        def close(self):
            closed.append(self)

    auditor = auditor_module.AuditorValidator.__new__(
        auditor_module.AuditorValidator
    )
    auditor.config = SimpleNamespace(netuid=71)
    auditor.wallet = SimpleNamespace(
        hotkey=SimpleNamespace(ss58_address="5AuditorHotkey")
    )
    previous = Source(62)
    replacement = Source(155)
    auditor.subtensor = previous
    auditor.uid = 62
    auditor.consecutive_errors = 4
    monkeypatch.setattr(
        auditor_module.bt,
        "Subtensor",
        lambda *, config: replacement,
    )

    assert auditor._reconnect_subtensor(reason="test") is True
    assert auditor.subtensor is replacement
    assert auditor.uid == 155
    assert auditor.consecutive_errors == 0
    assert closed == [previous]


def test_auditor_failed_reconnect_preserves_previous_source(monkeypatch):
    closed = []

    class Source:
        def __init__(self, uid):
            self.uid = uid

        def get_uid_for_hotkey_on_subnet(self, **_kwargs):
            return self.uid

        def close(self):
            closed.append(self)

    auditor = auditor_module.AuditorValidator.__new__(
        auditor_module.AuditorValidator
    )
    auditor.config = SimpleNamespace(netuid=71)
    auditor.wallet = SimpleNamespace(
        hotkey=SimpleNamespace(ss58_address="5AuditorHotkey")
    )
    previous = Source(62)
    replacement = Source(None)
    auditor.subtensor = previous
    auditor.uid = 62
    auditor.consecutive_errors = 4
    monkeypatch.setattr(
        auditor_module.bt,
        "Subtensor",
        lambda *, config: replacement,
    )

    assert auditor._reconnect_subtensor(reason="test") is False
    assert auditor.subtensor is previous
    assert auditor.uid == 62
    assert closed == [replacement]


def test_auditor_reconnects_stale_set_weights_socket_and_retries_exact_vector(
    monkeypatch,
):
    auditor = _auditor([(True, "accepted")])
    healthy = auditor.subtensor
    stale = _FakeSubtensor([])

    def stale_set_weights(**kwargs):
        stale.calls.append(dict(kwargs))
        raise ConnectionError("WebSocket connection is closed")

    stale.set_weights = stale_set_weights
    auditor.subtensor = stale
    reconnect_reasons = []

    def reconnect(*, reason):
        reconnect_reasons.append(reason)
        auditor.subtensor = healthy
        return True

    auditor._reconnect_subtensor = reconnect
    state = SimpleNamespace(
        workflow_epoch_id=2,
        subnet_epoch_index=None,
        current_block=1065,
        blocks_remaining=20,
    )
    auditor._read_epoch_state = lambda: state
    auditor._read_best_epoch_state = lambda: state
    sleeps = []
    monkeypatch.setattr(auditor_module.time, "sleep", sleeps.append)

    result = auditor._set_weights_until_epoch_end(
        epoch_id=2,
        uids=[3],
        weights=[1.0],
        expected_weights_u16=[65535],
    )

    assert result is True
    assert reconnect_reasons == ["weight_submission:set_weights"]
    assert stale.calls[0]["uids"] == [3]
    assert healthy.calls[0]["uids"] == [3]
    assert stale.calls[0]["weights"] == healthy.calls[0]["weights"] == [1.0]
    assert sleeps == [1]


def test_auditor_does_not_reconnect_or_retry_semantic_sdk_exception(
    monkeypatch,
):
    auditor = _auditor([])

    def invalid_set_weights(**_kwargs):
        raise ValueError("invalid signed weight payload")

    auditor.subtensor.set_weights = invalid_set_weights
    auditor._reconnect_subtensor = lambda **_kwargs: (_ for _ in ()).throw(
        AssertionError("semantic errors must not reconnect")
    )
    state = SimpleNamespace(
        workflow_epoch_id=2,
        subnet_epoch_index=None,
        current_block=1065,
        blocks_remaining=20,
    )
    auditor._read_epoch_state = lambda: state
    auditor._read_best_epoch_state = lambda: state
    monkeypatch.setattr(auditor_module.time, "sleep", lambda _seconds: None)

    with pytest.raises(ValueError, match="invalid signed weight payload"):
        auditor._set_weights_until_epoch_end(
            epoch_id=2,
            uids=[3],
            weights=[1.0],
            expected_weights_u16=[65535],
        )


def test_auditor_retries_sdk_success_until_finalized_last_update_advances(
    monkeypatch,
):
    auditor = _auditor(
        [(True, "accepted-but-unobserved"), (True, "accepted-and-finalized")]
    )
    state = SimpleNamespace(
        workflow_epoch_id=2,
        subnet_epoch_index=None,
        current_block=1065,
        blocks_remaining=20,
    )
    auditor._read_epoch_state = lambda: state
    auditor._read_best_epoch_state = lambda: state
    expected = [(3, 65535)]
    finalized_states = iter(
        [
            {
                "block_hash": "0x" + "1" * 64,
                "last_update": 100,
                "weights": [],
            },
            *[
                {
                    "block_hash": "0x" + str(index) * 64,
                    "last_update": 100,
                    "weights": [],
                }
                for index in range(2, 7)
            ],
            {
                "block_hash": "0x" + "7" * 64,
                "last_update": 101,
                "weights": expected,
            },
        ]
    )
    auditor._read_finalized_weight_submission_state = lambda: next(
        finalized_states
    )
    sleeps = []
    monkeypatch.setattr(auditor_module.time, "sleep", sleeps.append)

    result = auditor._set_weights_until_epoch_end(
        epoch_id=2,
        uids=[3],
        weights=[1.0],
        expected_weights_u16=[65535],
    )

    assert result is True
    assert len(auditor.subtensor.calls) == 2
    assert sleeps == [3, 3, 3, 3, 12]


def test_auditor_does_not_reject_prior_visible_vector_before_reveal(
    monkeypatch,
    capsys,
    caplog,
):
    auditor = _auditor([(True, "accepted")])
    caplog.set_level("INFO", logger=auditor_module.__name__)
    state = SimpleNamespace(
        workflow_epoch_id=2,
        subnet_epoch_index=None,
        current_block=1065,
        blocks_remaining=20,
    )
    auditor._read_epoch_state = lambda: state
    auditor._read_best_epoch_state = lambda: state
    finalized_states = iter(
        [
            {
                "block_hash": "0x" + "1" * 64,
                "last_update": 100,
                "weights": [],
            },
            {
                "block_hash": "0x" + "2" * 64,
                "last_update": 101,
                "weights": [(3, 60000)],
            },
        ]
    )
    auditor._read_finalized_weight_submission_state = lambda: next(
        finalized_states
    )
    monkeypatch.setattr(
        auditor_module.time,
        "sleep",
        lambda _seconds: (_ for _ in ()).throw(
            AssertionError("a finalized commitment must not retry before reveal")
        ),
    )

    result = auditor._set_weights_until_epoch_end(
        epoch_id=2,
        uids=[3],
        weights=[1.0],
        expected_weights_u16=[65535],
    )

    assert result is True
    assert len(auditor.subtensor.calls) == 1
    assert auditor.last_submitted_epoch is None
    output = capsys.readouterr().out
    assert "visible chain vector remains sealed" in output
    assert '"confirmation_stage":"timelocked_commit_finalized"' in caplog.text
    assert '"visible_vector_state":"reveal_pending"' in caplog.text
    assert '"event":"submission_failed"' not in caplog.text


def test_auditor_submit_records_commit_finalization_not_visible_reveal(
    monkeypatch,
    caplog,
):
    auditor = _auditor([_SdkResponse(True, "accepted")])
    state = SimpleNamespace(
        workflow_epoch_id=2,
        subnet_epoch_index=None,
        current_block=1065,
        blocks_remaining=20,
    )
    auditor._read_epoch_state = lambda: state
    auditor._read_best_epoch_state = lambda: state
    auditor._subnet_index_for_workflow_epoch = lambda _epoch: None
    monkeypatch.setattr(
        auditor_module.time,
        "sleep",
        lambda _seconds: (_ for _ in ()).throw(
            AssertionError("a finalized commitment must not retry")
        ),
    )
    caplog.set_level("INFO", logger=auditor_module.__name__)

    result = auditor.submit_weights_to_chain(
        2,
        {
            "epoch_id": 2,
            "block": 1065,
            "uids": [3],
            "weights_u16": [65535],
            "weights_hash": "weights-hash",
            "bundle_hash": "bundle-hash",
            "authority_stage": "finalized",
        },
        submission_epoch_id=2,
    )

    assert result is True
    assert auditor.last_submitted_epoch == 2
    assert auditor.last_authority_epoch == 2
    assert '"event":"submission_success"' in caplog.text
    assert '"confirmation_stage":"timelocked_commit_finalized"' in caplog.text
    assert '"event":"submission_failed"' not in caplog.text


def test_auditor_reports_visible_vector_when_reveal_already_matches(
    monkeypatch,
    capsys,
):
    auditor = _auditor([(True, "accepted")])
    state = SimpleNamespace(
        workflow_epoch_id=2,
        subnet_epoch_index=None,
        current_block=1065,
        blocks_remaining=20,
    )
    auditor._read_epoch_state = lambda: state
    auditor._read_best_epoch_state = lambda: state
    finalized_states = iter(
        [
            {
                "block_hash": "0x" + "1" * 64,
                "last_update": 100,
                "weights": [(8, 65535)],
            },
            {
                "block_hash": "0x" + "2" * 64,
                "last_update": 101,
                "weights": [(3, 65535)],
            },
        ]
    )
    auditor._read_finalized_weight_submission_state = lambda: next(
        finalized_states
    )
    monkeypatch.setattr(
        auditor_module.time,
        "sleep",
        lambda _seconds: (_ for _ in ()).throw(
            AssertionError("a confirmed submission must not retry")
        ),
    )

    result = auditor._set_weights_until_epoch_end(
        epoch_id=2,
        uids=[3],
        weights=[1.0],
        expected_weights_u16=[65535],
    )

    assert result is True
    assert len(auditor.subtensor.calls) == 1
    output = capsys.readouterr().out
    assert "visible chain vector already matches" in output


def test_auditor_lifecycle_transition_before_sdk_fails_closed(monkeypatch):
    auditor = _auditor([(True, "must-not-submit")])
    authority_checks = []
    state = SimpleNamespace(
        workflow_epoch_id=2,
        subnet_epoch_index=None,
        current_block=1065,
        blocks_remaining=1,
    )

    def lifecycle(*, force_refresh):
        assert force_refresh is True
        authority_checks.append("checked")
        if len(authority_checks) == 2:
            raise RuntimeError("stateful_active")
        return {"lifecycle_state": "stateful_active"}

    auditor._validate_durable_epoch_runtime_lifecycle = lifecycle
    auditor._read_epoch_state = lambda: state
    auditor._read_best_epoch_state = lambda: state
    monkeypatch.setattr(auditor_module.time, "sleep", lambda _seconds: None)

    result = auditor._set_weights_until_epoch_end(
        epoch_id=2,
        uids=[0],
        weights=[1.0],
        expected_weights_u16=[65535],
    )

    assert result is False
    assert authority_checks == ["checked", "checked"]
    assert auditor.subtensor.calls == []


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
    cutover = SimpleNamespace(mapping_hash="sha256:" + "1" * 64)
    auditor.epoch_cutover = cutover
    observed = []

    def validate(**kwargs):
        observed.append(kwargs)
        return {"lifecycle_state": "stateful_active"}

    monkeypatch.setattr(epoch_utils, "validate_epoch_runtime_lifecycle", validate)

    state = auditor._validate_durable_epoch_runtime_lifecycle(
        force_refresh=True,
    )

    assert state["lifecycle_state"] in {
        "stateful_active",
        "stateful_manifest_only",
    }
    assert bool(observed) is expects_check


def test_auditor_stops_before_retry_after_epoch_rollover(monkeypatch):
    auditor = _auditor([(False, "rejected")], blocks=[1079, 1080])
    monkeypatch.setattr(auditor_module.time, "sleep", lambda _seconds: None)
    best_states = iter(
        [
            SimpleNamespace(
                workflow_epoch_id=2,
                subnet_epoch_index=None,
                current_block=1079,
                blocks_remaining=1,
            ),
            SimpleNamespace(
                workflow_epoch_id=3,
                subnet_epoch_index=None,
                current_block=1080,
                blocks_remaining=360,
            ),
        ]
    )
    finalized = SimpleNamespace(
        workflow_epoch_id=2,
        subnet_epoch_index=None,
        current_block=1079,
        blocks_remaining=1,
    )
    auditor._read_epoch_state = lambda: finalized
    auditor._read_best_epoch_state = lambda: next(best_states)

    result = auditor._set_weights_until_epoch_end(
        epoch_id=2,
        uids=[0],
        weights=[1.0],
        expected_weights_u16=[65535],
    )

    assert result is False
    assert len(auditor.subtensor.calls) == 1
    assert auditor.last_submitted_epoch is None


def test_auditor_best_head_rollover_stops_retry_while_finalized_remains_old(
    monkeypatch,
):
    auditor = _auditor([(False, "rejected")])
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
        expected_weights_u16=[65535],
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
    assert primary_source.count("failed_source.set_weights(") == 1
    assert primary_source.count("self.subtensor.set_weights(") == 0
    assert auditor_source.count("self.subtensor.set_weights(") == 1
    assert "_submit_weights_to_gateway" not in primary_source
    assert "_submit_weights_v2" not in primary_source
    assert "VALIDATOR_ATTESTED_WEIGHT_MODE" not in primary_source
    assert "VALIDATOR_WEIGHT_PROTOCOL" in primary_source


def test_primary_reconnects_failed_source_before_rechecking_epoch(
    monkeypatch,
    capsys,
):
    validator = _primary([])
    failed_source = validator.subtensor
    replacement = _FakeSubtensor([_SdkResponse(True, "accepted")])
    events = []
    context_substrates = []

    def fail_set_weights(**kwargs):
        failed_source.calls.append(dict(kwargs))
        events.append(("set_weights", failed_source))
        raise TimeoutError("server rejected WebSocket connection: HTTP 503")

    failed_source.set_weights = fail_set_weights

    class TrackingContext(_AuthoritativeContext):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            context_substrates.append(kwargs["substrate"])

    async def no_sleep(_seconds):
        return None

    async def current(**_kwargs):
        events.append(("epoch", validator.subtensor))
        return True

    def reconnect(*, expected_source, reason):
        events.append(("reconnect", expected_source))
        assert expected_source is failed_source
        assert reason == "weight_submission_transport"
        validator.subtensor = replacement
        return True

    monkeypatch.setattr(validator_module.asyncio, "sleep", no_sleep)
    monkeypatch.setattr(
        validator_module,
        "AuthoritativeSetWeightsContextV2",
        TrackingContext,
    )
    validator._weight_submission_epoch_is_current = current
    validator._reconnect_subtensor_sync = reconnect

    result = asyncio.run(
        validator._set_weights_until_epoch_end(
            epoch_id=0,
            uids=[0],
            weights=[1.0],
            **_authoritative_args(),
        )
    )

    assert result is True
    assert events[:4] == [
        ("epoch", failed_source),
        ("set_weights", failed_source),
        ("reconnect", failed_source),
        ("epoch", replacement),
    ]
    assert len(failed_source.calls) == 1
    assert len(replacement.calls) == 1
    assert context_substrates == [
        failed_source.substrate,
        replacement.substrate,
    ]
    assert "transport error" in capsys.readouterr().out


def test_primary_propagates_transport_error_when_reconnect_fails(monkeypatch):
    validator = _primary([])
    failed_source = validator.subtensor

    def fail_set_weights(**kwargs):
        failed_source.calls.append(dict(kwargs))
        raise TimeoutError("finney websocket timed out")

    failed_source.set_weights = fail_set_weights

    async def no_sleep(_seconds):
        return None

    async def current(**_kwargs):
        return True

    reconnects = []

    def reconnect(*, expected_source, reason):
        reconnects.append((expected_source, reason))
        return False

    monkeypatch.setattr(validator_module.asyncio, "sleep", no_sleep)
    monkeypatch.setattr(
        validator_module,
        "AuthoritativeSetWeightsContextV2",
        _AuthoritativeContext,
    )
    validator._weight_submission_epoch_is_current = current
    validator._reconnect_subtensor_sync = reconnect

    with pytest.raises(TimeoutError, match="finney websocket timed out"):
        asyncio.run(
            validator._set_weights_until_epoch_end(
                epoch_id=0,
                uids=[0],
                weights=[1.0],
                **_authoritative_args(),
            )
        )

    assert reconnects == [
        (failed_source, "weight_submission_transport")
    ]
    assert len(failed_source.calls) == 1


def test_primary_does_not_misclassify_local_journal_io_as_transport(
    monkeypatch,
):
    validator = _primary([])

    def fail_set_weights(**kwargs):
        validator.subtensor.calls.append(dict(kwargs))
        raise OSError(errno.ENOSPC, "No space left on device")

    validator.subtensor.set_weights = fail_set_weights

    async def current(**_kwargs):
        return True

    reconnects = []
    validator._weight_submission_epoch_is_current = current
    validator._reconnect_subtensor_sync = (
        lambda **kwargs: reconnects.append(kwargs) or True
    )
    monkeypatch.setattr(
        validator_module,
        "AuthoritativeSetWeightsContextV2",
        _AuthoritativeContext,
    )

    with pytest.raises(OSError) as caught:
        asyncio.run(
            validator._set_weights_until_epoch_end(
                epoch_id=0,
                uids=[0],
                weights=[1.0],
                **_authoritative_args(),
            )
        )

    assert caught.value.errno == errno.ENOSPC
    assert reconnects == []
    assert len(validator.subtensor.calls) == 1


def test_primary_stops_after_reconnect_when_epoch_authority_closed(
    monkeypatch,
):
    validator = _primary([])
    failed_source = validator.subtensor
    replacement = _FakeSubtensor([_SdkResponse(True, "must-not-submit")])

    def fail_set_weights(**kwargs):
        failed_source.calls.append(dict(kwargs))
        raise ConnectionResetError("finney reset the websocket")

    failed_source.set_weights = fail_set_weights
    epoch_checks = iter([True, False])

    async def current(**_kwargs):
        return next(epoch_checks)

    async def no_sleep(_seconds):
        return None

    def reconnect(*, expected_source, reason):
        assert expected_source is failed_source
        assert reason == "weight_submission_transport"
        validator.subtensor = replacement
        return True

    monkeypatch.setattr(validator_module.asyncio, "sleep", no_sleep)
    monkeypatch.setattr(
        validator_module,
        "AuthoritativeSetWeightsContextV2",
        _AuthoritativeContext,
    )
    validator._weight_submission_epoch_is_current = current
    validator._reconnect_subtensor_sync = reconnect

    result = asyncio.run(
        validator._set_weights_until_epoch_end(
            epoch_id=0,
            uids=[0],
            weights=[1.0],
            **_authoritative_args(),
        )
    )

    assert result is False
    assert len(failed_source.calls) == 1
    assert replacement.calls == []
