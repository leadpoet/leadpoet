from __future__ import annotations

from types import SimpleNamespace

import pytest

from Leadpoet.utils.subnet_epoch import SubnetEpochCutover
import neurons.validator as validator_module


EVENT = "sha256:" + "1" * 64
OLD_AUTHORIZATION = "sha256:" + "2" * 64
NEW_AUTHORIZATION = "sha256:" + "3" * 64
HOTKEY = "5FqLp5QmNRiHGyj3xbLVnDHfCx25qxJX5CUhpndF9GFfZZiK"


def _record(*, published, signatures):
    result = {
        "epoch_id": 100,
        "uids": [1, 0],
        "weights": [0.2, 0.8],
        "sparse_uids": [0, 1],
        "sparse_weights_u16": [65535, 16384],
    }
    return {
        "weight_authorization_id": OLD_AUTHORIZATION,
        "published_bundle": {"weight_result": result},
        "publication": (
            {"weight_submission_event_hash": EVENT} if published else None
        ),
        "extrinsic_signature_results": list(signatures),
    }


class _Journal:
    def __init__(self, record):
        self.record = record
        self.calls = []
        self.scan_generation = 0

    def load(self):
        self.calls.append(("load", None))
        return self.record

    def record_prepared(self, record):
        self.calls.append(("prepared", dict(record)))
        self.record = dict(record)
        return self.record

    def record_published(self, acknowledgment):
        self.calls.append(("published", dict(acknowledgment)))
        self.record = {**self.record, "publication": dict(acknowledgment)}
        return self.record

    def replace_authorization(self, authorization_id):
        self.calls.append(("authorization", authorization_id))
        self.record = {
            **self.record,
            "weight_authorization_id": authorization_id,
        }
        return self.record

    def record_signed(self, result):
        self.calls.append(("signed", dict(result)))

    def reserve_finalization_scan(self):
        self.scan_generation += 1
        scan_id = "sha256:" + format(self.scan_generation, "064x")
        self.calls.append(("finalization_scan", scan_id))
        return scan_id

    def clear(self, *, expected_event_hash):
        self.calls.append(("clear", expected_event_hash))

    def quarantine(self, *, expected_epoch, reason):
        path = f"journal.quarantined.{expected_epoch}.{reason}"
        self.calls.append(("quarantine", (expected_epoch, reason, path)))
        self.record = None
        return path


class _Client:
    def __init__(self, *, signed_extrinsics, confirm_error=False):
        self.signed_extrinsics = list(signed_extrinsics)
        self.confirm_error = confirm_error
        self.calls = []

    def recover_weight_publication_v2(self, **kwargs):
        self.calls.append(("recover", kwargs))
        return {
            "weight_authorization_id": NEW_AUTHORIZATION,
            "signed_extrinsics": list(self.signed_extrinsics),
        }

    def confirm_weight_publication_v2(
        self, authorization_id, *, finalization_scan_id
    ):
        self.calls.append(
            ("confirm", (authorization_id, finalization_scan_id))
        )
        if self.confirm_error:
            raise RuntimeError("not finalized yet")
        return {"finalized": True}


def _validator(journal, client):
    validator = validator_module.Validator.__new__(validator_module.Validator)
    validator._weight_publication_journal_v2 = journal
    validator._validator_v2_client = client
    validator._epoch_cutover = SubnetEpochCutover(
        network_genesis_hash="0x" + "1" * 64,
        netuid=71,
        cutover_block=99,
        cutover_block_hash="0x" + "2" * 64,
        first_subnet_epoch_index=50,
        first_settlement_epoch_id=100,
        last_legacy_epoch_id=99,
    )
    validator.wallet = SimpleNamespace(
        hotkey=SimpleNamespace(ss58_address=HOTKEY)
    )
    substrate_calls = []

    def rpc_request(method, params):
        substrate_calls.append((method, params))
        return {"result": "0xaccepted"}

    validator.subtensor = SimpleNamespace(
        substrate=SimpleNamespace(rpc_request=rpc_request)
    )
    validator.substrate_calls = substrate_calls

    async def epoch_is_current(**_kwargs):
        return True

    validator._weight_submission_epoch_is_current = epoch_is_current

    async def finalized_state():
        return SimpleNamespace(workflow_epoch_id=100)

    async def best_state():
        return SimpleNamespace(workflow_epoch_id=100)

    validator._get_epoch_state_async = finalized_state
    validator._get_best_epoch_state_async = best_state
    return validator


@pytest.mark.asyncio
async def test_prepared_crash_replays_gateway_then_uses_epoch_bounded_chain_call(
    monkeypatch,
):
    journal = _Journal(_record(published=False, signatures=[]))
    client = _Client(signed_extrinsics=[])
    validator = _validator(journal, client)
    calls = []

    async def resume(**kwargs):
        calls.append(("resume", kwargs))
        return {"weight_submission_event_hash": EVENT}

    async def set_weights(**kwargs):
        calls.append(("set", kwargs))
        kwargs["on_signed_extrinsic"]({"signature": "durable"})
        return True

    async def finalize(**kwargs):
        calls.append(("finalize", kwargs))
        return {"acknowledgment": {"weight_finalization_event_hash": EVENT}}

    monkeypatch.setattr(
        validator_module, "resume_prepared_weight_publication_v2", resume
    )
    monkeypatch.setattr(
        validator_module, "finalize_authoritative_weight_publication_v2", finalize
    )
    validator._set_weights_until_epoch_end = set_weights

    outcome = await validator._recover_weight_publication_journal_v2(
        gateway_url="https://gateway.example"
    )
    assert outcome.epoch_id == 100
    assert outcome.status == "finalized"
    assert [item[0] for item in calls] == ["resume", "set", "finalize"]
    set_call = next(value for name, value in calls if name == "set")
    assert set_call["uids"] == [0, 1]
    assert set_call["weights"] == [0.8, 0.2]
    assert journal.calls[-1] == ("clear", EVENT)
    assert validator.substrate_calls == []


@pytest.mark.asyncio
async def test_prepared_stateful_crash_never_signs_before_epoch_evidence_replay(
    monkeypatch,
):
    record = _record(published=False, signatures=[])
    record["epoch_evidence"] = {
        "schema_version": "leadpoet.validator_subnet_epoch_evidence.v1"
    }
    journal = _Journal(record)
    client = _Client(signed_extrinsics=[])
    validator = _validator(journal, client)
    signed = []

    async def fail_epoch_evidence_replay(**kwargs):
        assert kwargs["journal_record"]["epoch_evidence"] == record[
            "epoch_evidence"
        ]
        raise RuntimeError("stateful epoch evidence was not durable")

    async def set_weights(**kwargs):
        signed.append(kwargs)
        return True

    monkeypatch.setattr(
        validator_module,
        "resume_prepared_weight_publication_v2",
        fail_epoch_evidence_replay,
    )
    validator._set_weights_until_epoch_end = set_weights

    with pytest.raises(RuntimeError, match="epoch evidence was not durable"):
        await validator._recover_weight_publication_journal_v2(
            gateway_url="https://gateway.example"
        )

    assert signed == []
    assert client.calls == []
    assert not any(name == "published" for name, _value in journal.calls)


@pytest.mark.asyncio
async def test_signed_crash_rebroadcasts_only_exact_enclave_bytes_then_finalizes(
    monkeypatch,
):
    signed = [{"receipt": True}]
    recovered_extrinsic = {
        "authorization_hash": "sha256:" + "4" * 64,
        "extrinsic_hash": "0x" + "5" * 64,
        "extrinsic_hex": "aabbcc",
    }
    journal = _Journal(_record(published=True, signatures=signed))
    client = _Client(
        signed_extrinsics=[recovered_extrinsic], confirm_error=True
    )
    validator = _validator(journal, client)

    async def finalize(**_kwargs):
        return {"acknowledgment": {"weight_finalization_event_hash": EVENT}}

    monkeypatch.setattr(
        validator_module, "finalize_authoritative_weight_publication_v2", finalize
    )
    outcome = await validator._recover_weight_publication_journal_v2(
        gateway_url="https://gateway.example"
    )
    assert outcome.epoch_id == 100
    assert outcome.status == "finalized"
    assert validator.substrate_calls == [
        ("author_submitExtrinsic", ["0xaabbcc"])
    ]
    assert journal.calls[-1] == ("clear", EVENT)


@pytest.mark.asyncio
async def test_signed_crash_never_rebroadcasts_after_durable_lifecycle_closes():
    recovered_extrinsic = {
        "authorization_hash": "sha256:" + "4" * 64,
        "extrinsic_hash": "0x" + "5" * 64,
        "extrinsic_hex": "aabbcc",
    }
    journal = _Journal(_record(published=True, signatures=[{"receipt": True}]))
    client = _Client(
        signed_extrinsics=[recovered_extrinsic],
        confirm_error=True,
    )
    validator = _validator(journal, client)

    async def lifecycle_closed(**_kwargs):
        return False

    validator._weight_submission_epoch_is_current = lifecycle_closed

    with pytest.raises(RuntimeError, match="durable epoch lifecycle"):
        await validator._recover_weight_publication_journal_v2(
            gateway_url="https://gateway.example"
        )

    assert validator.substrate_calls == []
    assert not any(name == "clear" for name, _value in journal.calls)


@pytest.mark.asyncio
async def test_finalization_failure_keeps_durable_journal(monkeypatch):
    recovered_extrinsic = {
        "authorization_hash": "sha256:" + "4" * 64,
        "extrinsic_hash": "0x" + "5" * 64,
        "extrinsic_hex": "aabbcc",
    }
    journal = _Journal(_record(published=True, signatures=[{"receipt": True}]))
    client = _Client(
        signed_extrinsics=[recovered_extrinsic], confirm_error=True
    )
    validator = _validator(journal, client)

    async def fail_finalize(**_kwargs):
        raise RuntimeError("gateway unavailable")

    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr(
        validator_module,
        "finalize_authoritative_weight_publication_v2",
        fail_finalize,
    )
    monkeypatch.setattr(validator_module.asyncio, "sleep", no_sleep)
    with pytest.raises(RuntimeError, match="lacks finalized-chain proof"):
        await validator._recover_weight_publication_journal_v2(
            gateway_url="https://gateway.example"
        )
    assert not any(name == "clear" for name, _value in journal.calls)


@pytest.mark.asyncio
async def test_finalization_retry_refreshes_epoch_and_quarantines_after_rollover(
    monkeypatch,
):
    recovered_extrinsic = {
        "authorization_hash": "sha256:" + "4" * 64,
        "extrinsic_hash": "0x" + "5" * 64,
        "extrinsic_hex": "aabbcc",
    }
    journal = _Journal(_record(published=True, signatures=[{"receipt": True}]))
    client = _Client(
        signed_extrinsics=[recovered_extrinsic],
        confirm_error=True,
    )
    validator = _validator(journal, client)
    workflow_epoch = {"value": 100}

    async def state():
        return SimpleNamespace(workflow_epoch_id=workflow_epoch["value"])

    validator._get_epoch_state_async = state
    validator._get_best_epoch_state_async = state

    async def fail_finalize(**_kwargs):
        raise RuntimeError("finalized proof temporarily unavailable")

    async def rollover_on_sleep(_seconds):
        workflow_epoch["value"] = 101

    monkeypatch.setattr(
        validator_module,
        "finalize_authoritative_weight_publication_v2",
        fail_finalize,
    )
    monkeypatch.setattr(validator_module.asyncio, "sleep", rollover_on_sleep)

    outcome = await validator._recover_weight_publication_journal_v2(
        gateway_url="https://gateway.example"
    )

    assert outcome.epoch_id == 100
    assert outcome.status == "quarantined"
    assert journal.calls[-1][0] == "quarantine"
    assert journal.calls[-1][1][:2] == (
        100,
        "signed_finalization_unresolved",
    )


@pytest.mark.asyncio
async def test_closed_unsigned_journal_is_quarantined_without_publication_or_signing(
    monkeypatch,
):
    journal = _Journal(_record(published=False, signatures=[]))
    client = _Client(signed_extrinsics=[])
    validator = _validator(journal, client)

    async def closed_state():
        return SimpleNamespace(workflow_epoch_id=101)

    validator._get_epoch_state_async = closed_state
    validator._get_best_epoch_state_async = closed_state

    async def never_resume(**_kwargs):
        pytest.fail("closed unsigned authority must not be republished")

    monkeypatch.setattr(
        validator_module,
        "resume_prepared_weight_publication_v2",
        never_resume,
    )

    outcome = await validator._recover_weight_publication_journal_v2(
        gateway_url="https://gateway.example"
    )

    assert outcome.epoch_id == 100
    assert outcome.status == "quarantined"
    assert client.calls == []
    assert journal.calls[-1][0] == "quarantine"
    assert journal.calls[-1][1][:2] == (100, "unsigned_epoch_closed")


@pytest.mark.asyncio
async def test_current_epoch_quarantine_is_not_reported_as_submission_success(
    monkeypatch,
):
    journal = _Journal(_record(published=False, signatures=[]))
    validator = _validator(journal, _Client(signed_extrinsics=[]))

    async def closed_state():
        return SimpleNamespace(workflow_epoch_id=101)

    validator._get_epoch_state_async = closed_state
    validator._get_best_epoch_state_async = closed_state
    monkeypatch.setenv("VALIDATOR_V2_GATEWAY_URL", "https://gateway.example")

    async def never_prepare(**_kwargs):
        pytest.fail("quarantined current epoch must not be republished")

    monkeypatch.setattr(
        validator_module,
        "prepare_authoritative_weight_publication_v2",
        never_prepare,
    )

    with pytest.raises(RuntimeError, match="quarantined"):
        await validator._authorize_and_set_weights_v2(
            epoch_state=SimpleNamespace(subnet_epoch_index=50),
            snapshot={"epoch_id": 100},
            host_uids=[0, 1],
            host_weights=[0.8, 0.2],
            allocation_hash="sha256:" + "4" * 64,
            leaderboard_window_start="2026-07-22T00:00:00Z",
            leaderboard_window_end="2026-07-22T01:00:00Z",
        )


@pytest.mark.asyncio
async def test_closed_signed_journal_preserves_evidence_when_release_recovery_fails():
    journal = _Journal(_record(published=True, signatures=[{"receipt": True}]))
    client = _Client(signed_extrinsics=[])
    validator = _validator(journal, client)

    async def closed_state():
        return SimpleNamespace(workflow_epoch_id=101)

    validator._get_epoch_state_async = closed_state
    validator._get_best_epoch_state_async = closed_state

    def fail_recovery(**_kwargs):
        raise RuntimeError("recovery validator release differs")

    client.recover_weight_publication_v2 = fail_recovery

    outcome = await validator._recover_weight_publication_journal_v2(
        gateway_url="https://gateway.example"
    )

    assert outcome.epoch_id == 100
    assert outcome.status == "quarantined"
    assert journal.calls[-1][0] == "quarantine"
    assert journal.calls[-1][1][:2] == (100, "signed_recovery_unresolved")


@pytest.mark.asyncio
async def test_closed_signed_journal_does_not_block_next_epoch_preparation(
    monkeypatch,
):
    journal = _Journal(_record(published=True, signatures=[{"receipt": True}]))
    client = _Client(signed_extrinsics=[])
    validator = _validator(journal, client)

    async def closed_state():
        return SimpleNamespace(workflow_epoch_id=101)

    validator._get_epoch_state_async = closed_state
    validator._get_best_epoch_state_async = closed_state

    def fail_recovery(**_kwargs):
        raise RuntimeError("recovery validator release differs")

    client.recover_weight_publication_v2 = fail_recovery
    reached = []

    class NextEpochPreparationReached(RuntimeError):
        pass

    async def prepare(**kwargs):
        reached.append(kwargs)
        raise NextEpochPreparationReached

    monkeypatch.setattr(
        validator_module,
        "prepare_authoritative_weight_publication_v2",
        prepare,
    )
    monkeypatch.setenv("VALIDATOR_V2_GATEWAY_URL", "https://gateway.example")

    with pytest.raises(NextEpochPreparationReached):
        await validator._authorize_and_set_weights_v2(
            epoch_state=SimpleNamespace(subnet_epoch_index=51),
            snapshot={"epoch_id": 101},
            host_uids=[0, 1],
            host_weights=[0.8, 0.2],
            allocation_hash="sha256:" + "4" * 64,
            leaderboard_window_start="2026-07-22T00:00:00Z",
            leaderboard_window_end="2026-07-22T01:00:00Z",
        )

    assert journal.calls[-1][0] == "quarantine"
    assert journal.calls[-1][1][:2] == (100, "signed_recovery_unresolved")
    assert len(reached) == 1
    assert reached[0]["calculation_snapshot"]["epoch_id"] == 101
