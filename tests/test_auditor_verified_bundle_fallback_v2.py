"""Durable, independently scheduled fallback for verified auditor weights."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from types import SimpleNamespace

from bittensor_wallet import Keypair
import pytest

from leadpoet_canonical.auditor_latest_verified_bundle_v2 import (
    AuditorLatestVerifiedBundleV2Error,
    LatestVerifiedBundleStoreV2,
    build_latest_verified_bundle_record_v2,
)
import neurons.auditor_validator as auditor_module


def _hash(character: str) -> str:
    return "sha256:" + (character * 64)


def _verified(
    epoch: int,
    *,
    uid: int = 7,
    weight: int = 65535,
    authority_stage: str = "published",
):
    return {
        "epoch_id": epoch,
        "netuid": 71,
        "block": epoch * 360 + 300,
        "uids": [uid],
        "weights_u16": [weight],
        "weights_hash": _hash("a"),
        "bundle_hash": _hash("b"),
        "authority_stage": authority_stage,
        "validator_hotkey": "5" * 48,
        "receipt_graph_hash": _hash("c"),
    }


def _authority(epoch: int):
    return {
        "schema_version": "leadpoet.published_weight_authority_stage.v2",
        "authority_stage": "published",
        "epoch_id_for_test": epoch,
    }


def _identity_cache():
    return {
        "schema_version": "leadpoet.independent_pcr0_identities.v2",
        "entries": [
            {
                "physical_role": "validator_weights",
                "role": "validator_weights",
                "commit_sha": "d" * 40,
                "pcr0": "e" * 96,
                "build_manifest_hash": _hash("f"),
                "dependency_lock_hash": _hash("1"),
                "verified_build_count": 3,
            }
        ],
    }


def _record(
    keypair,
    epoch: int,
    *,
    uid: int = 7,
    authority_stage: str = "published",
):
    return build_latest_verified_bundle_record_v2(
        auditor_hotkey=keypair.ss58_address,
        authority=_authority(epoch),
        identity_cache=_identity_cache(),
        verified_bundle=_verified(
            epoch,
            uid=uid,
            authority_stage=authority_stage,
        ),
        sign=keypair.sign,
    )


def _verify_with(keypair):
    return lambda message, signature: keypair.verify(message, signature)


def test_store_survives_restart_and_retains_only_the_newest_record(tmp_path):
    keypair = Keypair.create_from_mnemonic(Keypair.generate_mnemonic())
    path = tmp_path / "state" / "latest.json"
    first = LatestVerifiedBundleStoreV2(path)
    assert first.replace_if_newer(
        _record(keypair, 100),
        expected_hotkey=keypair.ss58_address,
        expected_netuid=71,
        verify_signature=_verify_with(keypair),
    )
    first_inode = path.stat().st_ino
    assert first.replace_if_newer(
        _record(keypair, 101),
        expected_hotkey=keypair.ss58_address,
        expected_netuid=71,
        verify_signature=_verify_with(keypair),
    )
    assert path.stat().st_ino != first_inode
    assert not first.replace_if_newer(
        _record(keypair, 100),
        expected_hotkey=keypair.ss58_address,
        expected_netuid=71,
        verify_signature=_verify_with(keypair),
    )

    restarted = LatestVerifiedBundleStoreV2(path)
    loaded = restarted.load(
        expected_hotkey=keypair.ss58_address,
        expected_netuid=71,
        verify_signature=_verify_with(keypair),
    )
    assert loaded["source_epoch_id"] == 101
    assert loaded["verified_bundle"]["uids"] == [7]
    assert list(path.parent.iterdir()) == [path]
    assert path.stat().st_mode & 0o777 == 0o600


def test_store_rejects_tampering_partial_json_wrong_wallet_and_wrong_netuid(tmp_path):
    keypair = Keypair.create_from_mnemonic(Keypair.generate_mnemonic())
    other = Keypair.create_from_mnemonic(Keypair.generate_mnemonic())
    path = tmp_path / "latest.json"
    store = LatestVerifiedBundleStoreV2(path)
    store.replace_if_newer(
        _record(keypair, 100),
        expected_hotkey=keypair.ss58_address,
        expected_netuid=71,
        verify_signature=_verify_with(keypair),
    )
    with pytest.raises(AuditorLatestVerifiedBundleV2Error, match="another hotkey"):
        store.load(
            expected_hotkey=other.ss58_address,
            expected_netuid=71,
            verify_signature=_verify_with(other),
        )
    with pytest.raises(AuditorLatestVerifiedBundleV2Error, match="another subnet"):
        store.load(
            expected_hotkey=keypair.ss58_address,
            expected_netuid=72,
            verify_signature=_verify_with(keypair),
        )

    tampered = json.loads(path.read_text())
    tampered["verified_bundle"]["weights_u16"] = [1]
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(AuditorLatestVerifiedBundleV2Error):
        store.load(
            expected_hotkey=keypair.ss58_address,
            expected_netuid=71,
            verify_signature=_verify_with(keypair),
        )

    path.write_text('{"schema_version":', encoding="utf-8")
    with pytest.raises(AuditorLatestVerifiedBundleV2Error, match="malformed"):
        store.load(
            expected_hotkey=keypair.ss58_address,
            expected_netuid=71,
            verify_signature=_verify_with(keypair),
        )


def test_new_verified_authority_recovers_a_corrupt_local_record(tmp_path):
    keypair = Keypair.create_from_mnemonic(Keypair.generate_mnemonic())
    path = tmp_path / "latest.json"
    path.write_text("not-json", encoding="utf-8")
    store = LatestVerifiedBundleStoreV2(path)
    assert store.replace_if_newer(
        _record(keypair, 102),
        expected_hotkey=keypair.ss58_address,
        expected_netuid=71,
        verify_signature=_verify_with(keypair),
    )
    assert store.load(
        expected_hotkey=keypair.ss58_address,
        expected_netuid=71,
        verify_signature=_verify_with(keypair),
    )["source_epoch_id"] == 102


def test_store_upgrades_published_to_finalized_without_accepting_vector_change(
    tmp_path,
):
    keypair = Keypair.create_from_mnemonic(Keypair.generate_mnemonic())
    path = tmp_path / "latest.json"
    store = LatestVerifiedBundleStoreV2(path)
    assert store.replace_if_newer(
        _record(keypair, 100, authority_stage="published"),
        expected_hotkey=keypair.ss58_address,
        expected_netuid=71,
        verify_signature=_verify_with(keypair),
    )
    assert store.replace_if_newer(
        _record(keypair, 100, authority_stage="finalized"),
        expected_hotkey=keypair.ss58_address,
        expected_netuid=71,
        verify_signature=_verify_with(keypair),
    )
    loaded = store.load(
        expected_hotkey=keypair.ss58_address,
        expected_netuid=71,
        verify_signature=_verify_with(keypair),
    )
    assert loaded["verified_bundle"]["authority_stage"] == "finalized"
    assert not store.replace_if_newer(
        _record(keypair, 100, authority_stage="published"),
        expected_hotkey=keypair.ss58_address,
        expected_netuid=71,
        verify_signature=_verify_with(keypair),
    )
    with pytest.raises(
        AuditorLatestVerifiedBundleV2Error,
        match="conflicting verified authorities",
    ):
        store.replace_if_newer(
            _record(keypair, 100, uid=8, authority_stage="finalized"),
            expected_hotkey=keypair.ss58_address,
            expected_netuid=71,
            verify_signature=_verify_with(keypair),
        )


def test_default_state_path_is_wallet_scoped_and_outside_the_git_checkout(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: tmp_path))
    path = auditor_module._latest_verified_bundle_state_path(
        netuid=71,
        hotkey="5" * 48,
    )
    assert path == (
        tmp_path
        / ".local"
        / "state"
        / "leadpoet"
        / "auditor-validator"
        / "netuid-71"
        / ("5" * 48)
        / "latest_verified_weight_bundle_v1.json"
    )
    assert not str(path).startswith(str(Path(auditor_module._REPO_ROOT)))


def _auditor(tmp_path: Path, *, source_epoch: int = 100):
    keypair = Keypair.create_from_mnemonic(Keypair.generate_mnemonic())
    auditor = auditor_module.AuditorValidator.__new__(
        auditor_module.AuditorValidator
    )
    auditor.wallet = SimpleNamespace(hotkey=keypair)
    auditor.config = SimpleNamespace(netuid=71)
    auditor.uid = 9
    auditor.last_submitted_epoch = None
    auditor.last_authority_epoch = None
    auditor._submission_lock = None
    auditor.should_exit = False
    auditor._verified_bundle_store = LatestVerifiedBundleStoreV2(
        tmp_path / "latest.json"
    )
    record = _record(keypair, source_epoch)
    auditor._verified_bundle_store.replace_if_newer(
        record,
        expected_hotkey=keypair.ss58_address,
        expected_netuid=71,
        verify_signature=_verify_with(keypair),
    )
    auditor.verify_attested_weights_v2 = lambda *_args, **_kwargs: _verified(
        source_epoch
    )
    return auditor, record


def _epoch_state(epoch: int, block: int):
    return auditor_module._AuditorEpochState(
        current_block=epoch * 360 + block,
        workflow_epoch_id=epoch,
        epoch_block=block,
        blocks_remaining=max(0, 360 - block),
        subnet_epoch_index=epoch,
    )


def test_live_fetch_persists_only_after_release_and_full_v2_verification(monkeypatch):
    monkeypatch.delenv("AUDITOR_WEIGHT_PROTOCOL", raising=False)
    auditor = auditor_module.AuditorValidator.__new__(
        auditor_module.AuditorValidator
    )
    auditor.config = SimpleNamespace(netuid=71)
    auditor.uid = 9
    auditor.gateway_url = "https://gateway.example"
    authority = _authority(100)
    identity_cache = _identity_cache()
    verified = _verified(100)
    stages = []

    async def fetch(epoch):
        stages.append(("fetch", epoch))
        return authority

    async def release(value):
        assert value is authority
        stages.append(("release", value))
        return identity_cache

    def verify(value, *, identity_cache):
        assert value is authority
        assert identity_cache is not None
        stages.append(("verify", value))
        return verified

    def persist(*, authority, identity_cache, verified_bundle):
        assert authority is not None
        assert identity_cache is not None
        assert verified_bundle is verified
        stages.append(("persist", verified_bundle))

    auditor.fetch_attested_weights_v2 = fetch
    auditor._fetch_release_identity_cache = release
    auditor.verify_attested_weights_v2 = verify
    auditor._persist_latest_verified_bundle = persist

    result, status = asyncio.run(auditor.fetch_verified_weight_authority(100))
    assert result is verified
    assert status == "v2_verified"
    assert [stage for stage, _value in stages] == [
        "fetch",
        "release",
        "verify",
        "persist",
    ]


@pytest.mark.parametrize("failure_stage", ["release", "verify", "identity"])
def test_live_fetch_never_persists_a_partially_verified_authority(
    monkeypatch,
    failure_stage,
):
    monkeypatch.delenv("AUDITOR_WEIGHT_PROTOCOL", raising=False)
    auditor = auditor_module.AuditorValidator.__new__(
        auditor_module.AuditorValidator
    )
    auditor.config = SimpleNamespace(netuid=71)
    auditor.uid = 9
    auditor.gateway_url = "https://gateway.example"
    authority = _authority(100)
    persisted = []

    async def fetch(_epoch):
        return authority

    async def release(_value):
        if failure_stage == "release":
            raise RuntimeError("release evidence unavailable")
        return _identity_cache()

    def verify(_value, *, identity_cache):
        assert identity_cache is not None
        if failure_stage == "verify":
            return None
        value = _verified(100)
        if failure_stage == "identity":
            value["epoch_id"] = 99
        return value

    auditor.fetch_attested_weights_v2 = fetch
    auditor._fetch_release_identity_cache = release
    auditor.verify_attested_weights_v2 = verify
    auditor._persist_latest_verified_bundle = lambda **kwargs: persisted.append(
        kwargs
    )

    result, status = asyncio.run(auditor.fetch_verified_weight_authority(100))
    assert result is None
    assert status == "v2_invalid"
    assert persisted == []


def test_cache_write_failure_does_not_change_normal_verified_authority(monkeypatch):
    monkeypatch.delenv("AUDITOR_WEIGHT_PROTOCOL", raising=False)
    auditor = auditor_module.AuditorValidator.__new__(
        auditor_module.AuditorValidator
    )
    auditor.config = SimpleNamespace(netuid=71)
    auditor.uid = 9
    auditor.gateway_url = "https://gateway.example"
    authority = _authority(100)
    verified = _verified(100)

    async def fetch(_epoch):
        return authority

    async def release(_value):
        return _identity_cache()

    auditor.fetch_attested_weights_v2 = fetch
    auditor._fetch_release_identity_cache = release
    auditor.verify_attested_weights_v2 = lambda *_args, **_kwargs: verified
    auditor._persist_latest_verified_bundle = lambda **_kwargs: (_ for _ in ()).throw(
        OSError("read-only state directory")
    )

    result, status = asyncio.run(auditor.fetch_verified_weight_authority(100))
    assert result is verified
    assert status == "v2_verified"


def test_fallback_reverifies_the_signed_record_and_triggers_at_355_only(tmp_path):
    auditor, _record_value = _auditor(tmp_path, source_epoch=100)
    calls = []

    async def submit_once(**kwargs):
        calls.append(kwargs)
        return True

    auditor._submit_verified_authority_once = submit_once
    assert not asyncio.run(
        auditor._submit_latest_verified_bundle_fallback(_epoch_state(101, 354))
    )
    assert asyncio.run(
        auditor._submit_latest_verified_bundle_fallback(_epoch_state(101, 355))
    )
    assert len(calls) == 1
    assert calls[0]["source_epoch_id"] == 100
    assert calls[0]["submission_epoch_id"] == 101
    assert calls[0]["submission_mode"] == "prior_epoch_verified_carry_forward"
    assert calls[0]["allow_prior_epoch"] is True


def test_current_epoch_verified_bundle_is_retried_without_relabeling(tmp_path):
    auditor, _ = _auditor(tmp_path, source_epoch=101)
    calls = []

    async def submit_once(**kwargs):
        calls.append(kwargs)
        return True

    auditor._submit_verified_authority_once = submit_once
    assert asyncio.run(
        auditor._submit_latest_verified_bundle_fallback(_epoch_state(101, 355))
    )
    assert calls[0]["source_epoch_id"] == 101
    assert calls[0]["submission_epoch_id"] == 101
    assert calls[0]["submission_mode"] == "current_epoch_verified_retry"


def test_fallback_rejects_future_or_reverification_mismatch(tmp_path):
    future, _ = _auditor(tmp_path / "future", source_epoch=102)
    future._submit_verified_authority_once = lambda **_kwargs: pytest.fail(
        "future authority must not submit"
    )
    assert not asyncio.run(
        future._submit_latest_verified_bundle_fallback(_epoch_state(101, 355))
    )

    mismatched, _ = _auditor(tmp_path / "mismatch", source_epoch=100)
    mismatched.verify_attested_weights_v2 = lambda *_args, **_kwargs: _verified(
        100, uid=8
    )
    mismatched._submit_verified_authority_once = lambda **_kwargs: pytest.fail(
        "changed verification result must not submit"
    )
    assert not asyncio.run(
        mismatched._submit_latest_verified_bundle_fallback(_epoch_state(101, 355))
    )


def test_fallback_without_a_valid_prior_record_fails_closed(tmp_path):
    auditor, _ = _auditor(tmp_path, source_epoch=100)
    auditor._verified_bundle_store.path.unlink()
    auditor._submit_verified_authority_once = lambda **_kwargs: pytest.fail(
        "missing local authority must not submit"
    )
    assert not asyncio.run(
        auditor._submit_latest_verified_bundle_fallback(_epoch_state(101, 355))
    )


def test_gateway_stall_cannot_stall_the_independent_deadline_observer(
    tmp_path,
    monkeypatch,
):
    auditor, _ = _auditor(tmp_path, source_epoch=100)
    auditor.epoch_archive_endpoint = "wss://archive.example:443"
    auditor.epoch_cutover = object()
    observer = object()
    triggered = asyncio.Event()
    gateway_release = asyncio.Event()

    monkeypatch.setattr(
        auditor_module,
        "_connect_epoch_archive_subtensor",
        lambda **_kwargs: observer,
    )
    monkeypatch.setattr(
        auditor_module,
        "validate_subnet_epoch_cutover_anchor",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        auditor_module,
        "_close_subtensor_connection",
        lambda *_args, **_kwargs: None,
    )
    auditor._fallback_epoch_state_from_observer = lambda _observer: _epoch_state(
        101, 355
    )

    async def fallback(_state):
        triggered.set()
        auditor.should_exit = True
        return True

    async def stalled_gateway_request():
        await gateway_release.wait()

    auditor._submit_latest_verified_bundle_fallback = fallback

    async def run():
        gateway_task = asyncio.create_task(stalled_gateway_request())
        fallback_task = asyncio.create_task(
            auditor._run_latest_verified_bundle_fallback_loop()
        )
        await asyncio.wait_for(triggered.wait(), timeout=2)
        assert not gateway_task.done()
        gateway_task.cancel()
        await asyncio.gather(gateway_task, fallback_task, return_exceptions=True)

    asyncio.run(run())


def test_stalled_archive_snapshot_is_bounded_and_reaches_block_355_fallback(
    tmp_path,
    monkeypatch,
):
    """A synchronous SDK read cannot consume the fallback submission window."""

    auditor, _ = _auditor(tmp_path, source_epoch=100)
    auditor.epoch_archive_endpoint = "wss://archive.example:443"
    auditor.epoch_cutover = object()
    observer = object()
    snapshot_calls = []
    submitted = asyncio.Event()

    monkeypatch.setattr(
        auditor_module,
        "FALLBACK_OBSERVER_OPERATION_TIMEOUT_SECONDS",
        0.02,
    )
    monkeypatch.setattr(
        auditor_module,
        "_connect_epoch_archive_subtensor",
        lambda **_kwargs: observer,
    )
    monkeypatch.setattr(
        auditor_module,
        "validate_subnet_epoch_cutover_anchor",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        auditor_module,
        "_close_subtensor_connection",
        lambda *_args, **_kwargs: None,
    )

    def snapshot(_observer):
        snapshot_calls.append(len(snapshot_calls) + 1)
        if len(snapshot_calls) == 1:
            time.sleep(0.08)
        return _epoch_state(101, 355)

    auditor._fallback_epoch_state_from_observer = snapshot

    async def fallback(state):
        assert state.epoch_block == 355
        submitted.set()
        auditor.should_exit = True
        return True

    auditor._submit_latest_verified_bundle_fallback = fallback

    async def run():
        task = asyncio.create_task(
            auditor._run_latest_verified_bundle_fallback_loop()
        )
        await asyncio.wait_for(submitted.wait(), timeout=5)
        await asyncio.wait_for(task, timeout=2)

    asyncio.run(run())
    assert snapshot_calls[:2] == [1, 2]


def test_deadline_observer_uses_best_head_for_exact_block_355_timing(
    tmp_path,
    monkeypatch,
):
    auditor, _ = _auditor(tmp_path, source_epoch=100)
    auditor.epoch_cutover = object()
    calls = []
    snapshot = SimpleNamespace(
        current_block=36355,
        epoch_block=355,
        blocks_remaining=5,
        subnet_epoch_index=101,
        settlement_epoch_id=lambda _cutover: 101,
    )

    def read(observer, *, netuid, finalized):
        calls.append((observer, netuid, finalized))
        return snapshot

    monkeypatch.setattr(auditor_module, "read_subnet_epoch_snapshot", read)
    observer = object()
    state = auditor._fallback_epoch_state_from_observer(observer)
    assert calls == [(observer, 71, False)]
    assert state.workflow_epoch_id == 101
    assert state.epoch_block == 355
    assert state.blocks_remaining == 5


def test_submission_lock_suppresses_concurrent_and_restart_duplicates(tmp_path):
    auditor, _ = _auditor(tmp_path, source_epoch=100)
    state = _epoch_state(101, 355)
    auditor._read_epoch_state = lambda: state
    chain_state = {"already": False}
    auditor._submission_already_finalized_for_epoch = (
        lambda _state: chain_state["already"]
    )
    calls = []

    def submit(source_epoch, bundle, *, submission_epoch_id):
        calls.append((source_epoch, submission_epoch_id, bundle["weights_hash"]))
        auditor.last_submitted_epoch = submission_epoch_id
        return True

    auditor.submit_weights_to_chain = submit

    async def concurrent():
        kwargs = {
            "source_epoch_id": 100,
            "submission_epoch_id": 101,
            "bundle": _verified(100),
            "submission_mode": "prior_epoch_verified_carry_forward",
            "allow_prior_epoch": True,
        }
        return await asyncio.gather(
            auditor._submit_verified_authority_once(**kwargs),
            auditor._submit_verified_authority_once(**kwargs),
        )

    assert asyncio.run(concurrent()) == [True, True]
    assert len(calls) == 1

    auditor.last_submitted_epoch = None
    auditor._submission_lock = None
    chain_state["already"] = True
    assert asyncio.run(
        auditor._submit_verified_authority_once(
            source_epoch_id=100,
            submission_epoch_id=101,
            bundle=_verified(100),
            submission_mode="prior_epoch_verified_carry_forward",
            allow_prior_epoch=True,
        )
    )
    assert len(calls) == 1
    assert auditor.last_submitted_epoch == 101


def test_same_verified_record_can_cover_consecutive_missed_epochs_then_recover(tmp_path):
    auditor, _ = _auditor(tmp_path, source_epoch=100)
    submissions = []

    async def submit_once(**kwargs):
        submissions.append(
            (kwargs["source_epoch_id"], kwargs["submission_epoch_id"])
        )
        auditor.last_submitted_epoch = kwargs["submission_epoch_id"]
        return True

    auditor._submit_verified_authority_once = submit_once
    assert asyncio.run(
        auditor._submit_latest_verified_bundle_fallback(_epoch_state(101, 355))
    )
    auditor.last_submitted_epoch = None
    assert asyncio.run(
        auditor._submit_latest_verified_bundle_fallback(_epoch_state(102, 355))
    )
    assert submissions == [(100, 101), (100, 102)]

    keypair = auditor.wallet.hotkey
    auditor._verified_bundle_store.replace_if_newer(
        _record(keypair, 103),
        expected_hotkey=keypair.ss58_address,
        expected_netuid=71,
        verify_signature=_verify_with(keypair),
    )
    auditor.verify_attested_weights_v2 = lambda *_args, **_kwargs: _verified(103)
    auditor.last_submitted_epoch = None
    assert asyncio.run(
        auditor._submit_latest_verified_bundle_fallback(_epoch_state(104, 355))
    )
    assert submissions[-1] == (103, 104)


def test_normal_gateway_retry_contract_is_unchanged():
    source = Path(auditor_module.__file__).read_text(encoding="utf-8")
    assert "aiohttp.ClientTimeout(total=30)" in source
    assert 'await asyncio.sleep(\n                                5' in source
    assert "VERIFIED_BUNDLE_FALLBACK_BLOCK = 355" in source
