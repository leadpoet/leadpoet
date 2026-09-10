from pathlib import Path
from types import SimpleNamespace

import pytest

from lab_arena.validator import (
    ArenaValidatorError,
    ArenaWeightOrchestrator,
    ArenaWeightPaths,
    _read_hashed_json,
)


class _Era:
    def encode(self, value):
        self.value = value

    def birth(self, current):
        return current - 1


class _Substrate:
    runtime_config = SimpleNamespace(create_scale_object=lambda _name: _Era())

    def __init__(self, broadcasts):
        self.broadcasts = broadcasts

    def get_account_nonce(self, _hotkey):
        return 7

    def get_block_hash(self, block_id):
        return "0x" + ("1" * 64)

    def rpc_request(self, method, params):
        assert method == "author_submitExtrinsic"
        self.broadcasts.append(params[0])


class _Chain:
    def __init__(self, broadcasts):
        self.client = _Substrate(broadcasts)
        self.config = SimpleNamespace(netuid=71, network_name="finney")

    def finalized_head(self):
        return SimpleNamespace(number=1050, hash="0x" + ("2" * 64))

    def refresh_metagraph(self):
        return SimpleNamespace(hotkeys=("5" + "A" * 47, "5" + "B" * 47))


class _Signer:
    def __init__(self, protected, outcomes):
        self.protected = protected
        self.outcomes = list(outcomes)
        self.prepares = 0
        self.confirms = 0

    def prepare_arena_weight_extrinsic_v1(self, _request):
        self.prepares += 1
        return dict(self.protected)

    def confirm_arena_weight_extrinsic_v1(self, _request):
        self.confirms += 1
        return dict(self.outcomes.pop(0))

    def recover_arena_weight_extrinsic_v1(self, request):
        self.recoveries = getattr(self, "recoveries", 0) + 1
        recovery = request["recovery_record"]
        return {
            "state_hash": request["accepted_state"]["state_hash"],
            "authorization_hash": recovery["authorization_hash"],
            "extrinsic_hash": recovery["extrinsic_hash"],
            "extrinsic_hex": recovery["extrinsic_hex"],
        }

    def sign_arena_chain_outcome_v1(self, document):
        return {"signature": "ab" * 64, "request_id": document["request_id"]}


def _orchestrator(tmp_path, signer, broadcasts):
    value = ArenaWeightOrchestrator(
        api=object(), chain=_Chain(broadcasts), signer=signer,
        validator_hotkey="5" + "V" * 47,
        expected_signing_key_hash="sha256:" + "3" * 64,
        paths=ArenaWeightPaths(tmp_path),
        extrinsic_period=32,
    )
    state = {
        "state_hash": "sha256:" + "4" * 64, "epoch": 9,
        "valid_from_block": 1000, "valid_until_block": 1100,
    }
    value._verified_state = lambda _epoch, _key: state
    value.api = SimpleNamespace(signing_key=lambda: {}, submit_chain_outcome=lambda _document: {"status": "recorded"})
    value._host_derivation = lambda _state, _hotkeys: {
        "state_hash": state["state_hash"], "weights_hash": "sha256:" + "5" * 64,
        "sparse_uids": [0, 1], "sparse_weights_u16": [65535, 100],
    }
    return value


def _protected():
    value = {
        "schema_version": "leadpoet.arena.weight_extrinsic.v1",
        "state_hash": "sha256:" + "4" * 64, "epoch": 9, "netuid": 71,
        "finalized_block": 1050, "finalized_block_hash": "0x" + "2" * 64,
        "weights_hash": "sha256:" + "5" * 64,
        "sparse_uids": [0, 1], "sparse_weights_u16": [65535, 100],
        "authorization_hash": "sha256:" + "6" * 64,
        "extrinsic_hash": "0x" + "7" * 64, "extrinsic_hex": "deadbeef",
        "attempt_sequence": 1,
    }
    value["recovery_record"] = {
        "authorization_hash": value["authorization_hash"],
        "extrinsic_hash": value["extrinsic_hash"],
        "extrinsic_hex": value["extrinsic_hex"],
    }
    return value


def test_signed_bytes_are_durable_before_broadcast_and_restart_reuses_them(tmp_path):
    broadcasts = []
    signer = _Signer(_protected(), [
        {"status": "pending", "finalized": False},
        {"status": "finalized", "finalized": True,
         "finalized_block_hash": "0x" + "8" * 64, "finalized_block": 1060,
         "weights_hash": "sha256:" + "5" * 64, "validator_uid": 4,
         "last_update": 1060, "revealed_weights": [[0, 65535], [1, 100]]},
    ])
    orchestrator = _orchestrator(tmp_path, signer, broadcasts)
    original_broadcast = orchestrator._broadcast

    def assert_journal_then_broadcast(value):
        assert _read_hashed_json(tmp_path / "epoch-9-signed.json")["extrinsic_hex"] == "deadbeef"
        original_broadcast(value)

    orchestrator._broadcast = assert_journal_then_broadcast
    assert orchestrator.run_once(9) == "broadcast"
    assert signer.prepares == 1
    assert broadcasts == ["0xdeadbeef"]

    assert orchestrator.run_once(9) == "rebroadcast"
    assert signer.prepares == 1
    assert signer.recoveries == 1
    assert broadcasts == ["0xdeadbeef", "0xdeadbeef"]
    assert orchestrator.run_once(9) == "finalized"
    assert signer.prepares == 1
    assert signer.recoveries == 2
    assert _read_hashed_json(tmp_path / "epoch-9-outcome.json")["extrinsic_hash"] == "0x" + "7" * 64


def test_host_and_protected_vector_mismatch_fails_before_journal_or_broadcast(tmp_path):
    broadcasts = []
    protected = _protected()
    protected["sparse_weights_u16"] = [1, 2]
    orchestrator = _orchestrator(tmp_path, _Signer(protected, []), broadcasts)
    with pytest.raises(ArenaValidatorError, match="differs"):
        orchestrator.run_once(9)
    assert not (tmp_path / "epoch-9-signed.json").exists()
    assert broadcasts == []


def test_tampered_signed_journal_fails_closed(tmp_path):
    path = Path(tmp_path) / "signed.json"
    path.write_text('{"epoch":9,"record_hash":"sha256:bad"}\n')
    with pytest.raises(ArenaValidatorError, match="journal hash"):
        _read_hashed_json(path)


def test_protected_expiry_proof_closes_old_journal_and_allows_next_epoch(tmp_path):
    broadcasts = []
    signer = _Signer(_protected(), [
        {"status": "not_included_expired", "finalized": False,
         "finalized_head": {"block": 1200}, "finalized_nonce": 7},
    ])
    orchestrator = _orchestrator(tmp_path, signer, broadcasts)
    assert orchestrator.run_once(9) == "broadcast"
    orchestrator.chain.finalized_head = lambda: SimpleNamespace(
        number=1090, hash="0x" + ("2" * 64)
    )
    assert orchestrator.run_once(9) == "not_included_expired"
    outcome = _read_hashed_json(tmp_path / "epoch-9-outcome.json")
    assert outcome["outcome"]["status"] == "not_included_expired"
    with pytest.raises(ArenaValidatorError, match="wrong epoch"):
        orchestrator.run_once(10)
    assert signer.prepares == 2


def test_included_commit_waits_for_reveal_without_blocking_next_epoch(tmp_path):
    signer = _Signer(_protected(), [
        {"status": "included_pending_reveal", "finalized": False,
         "inclusion_block": 1055, "extrinsic_hash": "0x" + "7" * 64},
    ])
    orchestrator = _orchestrator(tmp_path, signer, [])
    assert orchestrator.run_once(9) == "broadcast"
    assert orchestrator.run_once(9) == "included_pending_reveal"
    with pytest.raises(ArenaValidatorError, match="wrong epoch"):
        orchestrator.run_once(10)
    assert signer.prepares == 2


def test_missing_finalized_nonce_fails_before_protected_signing(tmp_path):
    signer = _Signer(_protected(), [])
    orchestrator = _orchestrator(tmp_path, signer, [])
    orchestrator.chain.client.get_account_nonce = lambda _hotkey: None
    with pytest.raises(ArenaValidatorError, match="nonce is unavailable"):
        orchestrator.run_once(9)
    assert signer.prepares == 0


def test_broken_prior_epoch_does_not_block_current_epoch(tmp_path):
    (tmp_path / "epoch-8-signed.json").write_text(
        '{"epoch":8,"record_hash":"sha256:bad"}\n', encoding="utf-8"
    )
    signer = _Signer(_protected(), [])
    broadcasts = []
    orchestrator = _orchestrator(tmp_path, signer, broadcasts)
    orchestrator.poll_prior_outcomes(9)
    assert orchestrator.run_once(9) == "broadcast"
    assert broadcasts == ["0xdeadbeef"]


def test_expired_mortal_attempt_retries_fresh_era_within_same_epoch(tmp_path):
    first = _protected()
    second = dict(first)
    second.update({
        "attempt_sequence": 2,
        "authorization_hash": "sha256:" + "9" * 64,
        "extrinsic_hash": "0x" + "a" * 64,
        "extrinsic_hex": "cafebabe",
    })
    second["recovery_record"] = {
        "authorization_hash": second["authorization_hash"],
        "extrinsic_hash": second["extrinsic_hash"],
        "extrinsic_hex": second["extrinsic_hex"],
    }

    class RetrySigner(_Signer):
        def prepare_arena_weight_extrinsic_v1(self, _request):
            self.prepares += 1
            return dict(first if self.prepares == 1 else second)

    signer = RetrySigner(first, [
        {"status": "not_included_expired", "finalized": False,
         "state_hash": first["state_hash"], "extrinsic_hash": first["extrinsic_hash"],
         "finalized_head": {"block": 1050}, "finalized_nonce": 7},
        {"status": "pending", "finalized": False},
    ])
    broadcasts = []
    orchestrator = _orchestrator(tmp_path, signer, broadcasts)
    assert orchestrator.run_once(9) == "broadcast"
    assert orchestrator.run_once(9) == "broadcast"
    archived = _read_hashed_json(tmp_path / "epoch-9-attempt-1-signed.json")
    active = _read_hashed_json(tmp_path / "epoch-9-signed.json")
    assert archived["extrinsic_hex"] == "deadbeef"
    assert active["attempt_sequence"] == 2
    assert active["extrinsic_hex"] == "cafebabe"
    assert broadcasts == ["0xdeadbeef", "0xcafebabe"]

    restarted = _orchestrator(tmp_path, signer, broadcasts)
    assert restarted.run_once(9) == "rebroadcast"
    assert signer.prepares == 2
    assert broadcasts[-1] == "0xcafebabe"
