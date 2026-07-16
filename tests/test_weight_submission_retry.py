from __future__ import annotations

import asyncio
from types import SimpleNamespace

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
    validator._last_weight_submission_epoch = None
    return validator


def _auditor(results, *, blocks=()):
    auditor = auditor_module.AuditorValidator.__new__(auditor_module.AuditorValidator)
    auditor.config = SimpleNamespace(netuid=71)
    auditor.wallet = object()
    auditor.subtensor = _FakeSubtensor(results, blocks=blocks)
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
        blocks=[345, 346],
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
    validator = _primary([(False, "rejected")], blocks=[360])

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


def test_primary_accepts_v10_extrinsic_response(monkeypatch):
    validator = _primary([_SdkResponse(True, "accepted")])

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


def test_auditor_retries_false_tuples_until_true(monkeypatch, capsys):
    auditor = _auditor(
        [(False, "rejected-one"), (False, "rejected-two"), (True, "accepted")],
        blocks=[1065, 1066],
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
    auditor = _auditor([_SdkResponse(True, "accepted")])

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


def test_auditor_stops_before_retry_after_epoch_rollover(monkeypatch):
    auditor = _auditor([(False, "rejected")], blocks=[1080])
    monkeypatch.setattr(auditor_module.time, "sleep", lambda _seconds: None)

    result = auditor._set_weights_until_epoch_end(
        epoch_id=2,
        uids=[0],
        weights=[1.0],
    )

    assert result is False
    assert len(auditor.subtensor.calls) == 1
    assert auditor.last_submitted_epoch is None


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
