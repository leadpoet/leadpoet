import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from lab_arena import contracts
from lab_arena.validator import (
    ArenaPublicApi,
    ArenaValidatorError,
    ArenaWeightOrchestrator,
    ArenaWeightPaths,
    _read_hashed_json,
)


def test_weight_api_posts_signed_scope_bound_request(monkeypatch):
    observed = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self, _limit):
            return b'{"lookup_ok":true,"state":null}'

    def urlopen(request, timeout):
        observed["request"] = request
        observed["timeout"] = timeout
        return Response()

    keypair = SimpleNamespace(
        ss58_address="5" + "V" * 47,
        sign=lambda message: b"s" * 64,
    )
    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    api = ArenaPublicApi(
        "https://arena.example", keypair=keypair,
        network="finney", netuid=71, timeout_seconds=12,
        now=lambda: 1_800_000_000,
    )
    assert api.accepted_weight_state(32001) is None
    request = observed["request"]
    assert request.full_url == "https://arena.example/arena/v1/weight-state"
    assert request.method == "POST" and observed["timeout"] == 12
    envelope = json.loads(request.data)
    assert envelope["scope"] == contracts.SCOPE_WEIGHT_STATE
    assert envelope["round_id"] == "weight-state"
    assert envelope["hotkey"] == keypair.ss58_address
    assert envelope["timestamp"] == 1_800_000_000
    assert envelope["body"] == {
        "epoch": 32001, "network": "finney", "netuid": 71,
    }
    assert envelope["signature"] == "0x" + (b"s" * 64).hex()


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
        self.submission_ready = True
        self.submission_read_error = None

    def finalized_head(self):
        return SimpleNamespace(number=1050, hash="0x" + ("2" * 64))

    def refresh_metagraph(self):
        return SimpleNamespace(hotkeys=("5" + "A" * 47, "5" + "B" * 47))

    def finalized_weight_submission_context(self, _hotkey):
        if self.submission_read_error is not None:
            raise self.submission_read_error
        return self.finalized_head(), self.refresh_metagraph(), self.submission_ready


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
         "finalized_block_hash": "8" * 64, "finalized_block": 1060,
         "commit_included_block": 1050,
         "weights_hash": "sha256:" + "5" * 64, "validator_uid": 4,
         "last_update": 1050, "revealed_weights": [[0, 65535], [1, 100]]},
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


def test_prior_poll_skips_archived_attempts_but_warns_on_malformed_names(
    tmp_path, capsys
):
    def journal(epoch):
        value = {"epoch": epoch, "record": "canonical"}
        value["record_hash"] = contracts.document_hash(value)
        return value

    # A valid archive name is produced by ArenaWeightPaths.archived_attempt;
    # its contents must not be re-polled as an active journal.
    (tmp_path / "epoch-8-attempt-1-signed.json").write_text(
        "not-json\n", encoding="utf-8"
    )
    (tmp_path / "epoch-8-signed.json").write_text(
        json.dumps(journal(8)) + "\n", encoding="utf-8"
    )
    (tmp_path / "epoch-8-attempt-invalid-signed.json").write_text(
        "not-json\n", encoding="utf-8"
    )
    (tmp_path / "epoch-not-an-epoch-signed.json").write_text(
        "not-json\n", encoding="utf-8"
    )

    orchestrator = _orchestrator(tmp_path, _Signer(_protected(), []), [])
    recovered = []
    confirmed = []
    orchestrator._recover_protected_state = lambda signed: recovered.append(signed)
    orchestrator._confirm = lambda signed: confirmed.append(signed)

    orchestrator.poll_prior_outcomes(9)

    assert [value["epoch"] for value in recovered] == [8]
    assert [value["epoch"] for value in confirmed] == [8]
    stderr = capsys.readouterr().err
    assert stderr.count("Arena validator ignored an invalid journal filename") == 2


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


def test_late_prior_update_defers_next_epoch_without_consuming_attempt(tmp_path):
    signer = _Signer(_protected(), [])
    orchestrator = _orchestrator(tmp_path, signer, [])
    orchestrator.chain.submission_ready = False

    assert orchestrator.run_once(9) == "rate_limited"
    assert signer.prepares == 0
    assert not orchestrator.paths.signed(9).exists()

    orchestrator.chain.submission_ready = True
    assert orchestrator.run_once(9) == "broadcast"
    assert signer.prepares == 1


def test_expired_attempt_waits_across_restart_until_rate_limit_is_ready(tmp_path):
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

    expired = {
        "status": "not_included_expired", "finalized": False,
        "state_hash": first["state_hash"], "extrinsic_hash": first["extrinsic_hash"],
        "finalized_head": {"block": 1050}, "finalized_nonce": 7,
    }
    signer = RetrySigner(first, [expired, expired, expired])
    broadcasts = []
    orchestrator = _orchestrator(tmp_path, signer, broadcasts)
    assert orchestrator.run_once(9) == "broadcast"
    orchestrator.chain.submission_ready = False
    assert orchestrator.run_once(9) == "rate_limited"
    assert signer.prepares == 1
    assert not orchestrator.paths.archived_attempt(9, 1).exists()

    restarted = _orchestrator(tmp_path, signer, broadcasts)
    restarted.chain.submission_ready = False
    assert restarted.run_once(9) == "rate_limited"
    assert signer.prepares == 1
    assert not restarted.paths.archived_attempt(9, 1).exists()

    restarted.chain.submission_ready = True
    assert restarted.run_once(9) == "broadcast"
    assert signer.prepares == 2
    assert _read_hashed_json(restarted.paths.archived_attempt(9, 1))["extrinsic_hex"] == "deadbeef"
    assert _read_hashed_json(restarted.paths.signed(9))["attempt_sequence"] == 2


def test_invalid_replacement_does_not_archive_active_attempt(tmp_path):
    first = _protected()
    invalid_second = dict(first)
    invalid_second["attempt_sequence"] = 2
    invalid_second["sparse_weights_u16"] = [1, 2]

    class RetrySigner(_Signer):
        def prepare_arena_weight_extrinsic_v1(self, _request):
            self.prepares += 1
            return dict(first if self.prepares == 1 else invalid_second)

    expired = {
        "status": "not_included_expired", "finalized": False,
        "state_hash": first["state_hash"], "extrinsic_hash": first["extrinsic_hash"],
        "finalized_head": {"block": 1050}, "finalized_nonce": 7,
    }
    signer = RetrySigner(first, [expired])
    orchestrator = _orchestrator(tmp_path, signer, [])
    assert orchestrator.run_once(9) == "broadcast"
    original = orchestrator.paths.signed(9).read_bytes()

    with pytest.raises(ArenaValidatorError, match="differs"):
        orchestrator.run_once(9)
    assert orchestrator.paths.signed(9).read_bytes() == original
    assert not orchestrator.paths.archived_attempt(9, 1).exists()


def test_rate_limit_read_failure_fails_before_signing(tmp_path):
    signer = _Signer(_protected(), [])
    orchestrator = _orchestrator(tmp_path, signer, [])
    orchestrator.chain.submission_read_error = RuntimeError("chain unavailable")

    with pytest.raises(RuntimeError, match="chain unavailable"):
        orchestrator.run_once(9)
    assert signer.prepares == 0
    assert not orchestrator.paths.signed(9).exists()


def test_participation_denial_blocks_new_signing_then_recovers(tmp_path):
    from lab_arena.validator import ArenaParticipationRequired

    broadcasts = []
    signer = _Signer(_protected(), [])
    orchestrator = _orchestrator(tmp_path, signer, broadcasts)
    accepted = orchestrator._verified_state
    def deny(*_):
        raise ArenaParticipationRequired("validator_participation_required")
    orchestrator._verified_state = deny
    assert orchestrator.run_once(9) == "blocked_on_participation"
    assert signer.prepares == 0 and broadcasts == []
    assert not (tmp_path / "epoch-9-signed.json").exists()
    orchestrator._verified_state = accepted
    assert orchestrator.run_once(9) == "broadcast"
    assert signer.prepares == 1


@pytest.mark.parametrize("outcome,expected", [
    ({"status": "pending", "finalized": False}, "rebroadcast"),
    ({"status": "included_pending_reveal", "finalized": False,
      "inclusion_block": 1055, "extrinsic_hash": "0x" + "7" * 64}, "included_pending_reveal"),
])
def test_participation_denial_preserves_signed_recovery(tmp_path, outcome, expected):
    signer = _Signer(_protected(), [outcome])
    orchestrator = _orchestrator(tmp_path, signer, [])
    assert orchestrator.run_once(9) == "broadcast"
    journal = (tmp_path / "epoch-9-signed.json").read_bytes()
    def forbidden(*_):
        pytest.fail("signed recovery must not fetch new weight state")
    orchestrator._verified_state = forbidden
    orchestrator.api.signing_key = forbidden
    assert orchestrator.run_once(9) == expected
    assert signer.prepares == 1 and signer.recoveries == 1
    assert (tmp_path / "epoch-9-signed.json").read_bytes() == journal


def test_participation_denial_blocks_expired_attempt_replacement(tmp_path):
    from lab_arena.validator import ArenaParticipationRequired

    signer = _Signer(_protected(), [{
        "status": "not_included_expired", "finalized": False,
        "finalized_head": {"block": 1060}, "finalized_nonce": 7,
    }])
    orchestrator = _orchestrator(tmp_path, signer, [])
    assert orchestrator.run_once(9) == "broadcast"
    journal = (tmp_path / "epoch-9-signed.json").read_bytes()
    def deny(*_):
        raise ArenaParticipationRequired("validator_participation_required")
    orchestrator._verified_state = deny
    assert orchestrator.run_once(9) == "blocked_on_participation"
    assert signer.prepares == 1
    assert (tmp_path / "epoch-9-signed.json").read_bytes() == journal
    assert not (tmp_path / "epoch-9-attempt-1-signed.json").exists()


@pytest.mark.parametrize("status,body,participation", [
    (403, b'{"status":"rejected","code":"validator_participation_required"}', True),
    (503, b'{"status":"rejected","code":"validator_participation_required"}', False),
    (403, b'{"status":"rejected","code":"other"}', False),
    (403, b'not-json-private', False),
    (403, b'{"status":"rejected","code":"validator_participation_required","extra":1}', False),
    (403, b' ' * 4097 + b'private', False),
])
def test_weight_api_only_recognizes_bounded_participation_denial(monkeypatch, status, body, participation):
    import io
    from urllib.error import HTTPError
    from lab_arena.validator import ArenaParticipationRequired

    stream = io.BytesIO(body)
    error = HTTPError("https://arena.example", status, "error", {}, stream)
    def urlopen(*_args, **_kwargs):
        raise error
    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    api = ArenaPublicApi("https://arena.example", keypair=SimpleNamespace(
        ss58_address="5" + "V" * 47, sign=lambda _: b"s" * 64,
    ), network="finney", netuid=71)
    with pytest.raises(ArenaValidatorError) as caught:
        api.accepted_weight_state(123)
    assert isinstance(caught.value, ArenaParticipationRequired) is participation
    assert "private" not in str(caught.value)
    assert stream.closed
