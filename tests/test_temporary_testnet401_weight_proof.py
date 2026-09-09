from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

from scripts import temporary_testnet401_weight_proof as proof


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = "a" * 40
RUN_ID = "pp-123456-1"
INSTANCE_ID = "i-0123456789abcdef0"
EPOCH_ID = 22_060
LAST_UPDATE = 7_959_901
LIVE_WEIGHTS_HASH = (
    "511b5b11f38fe37c09587b584b84c4b55a9cbe8981f895ddd6fce29644592e09"
)
HASHES = {
    "authority_hash": "sha256:" + "1" * 64,
    "bundle_hash": "sha256:" + "2" * 64,
    "weights_hash": LIVE_WEIGHTS_HASH,
    "weight_submission_event_hash": "sha256:" + "4" * 64,
    "weight_finalization_event_hash": "sha256:" + "5" * 64,
}


def _automatic() -> dict:
    return {
        "status": "candidate_match",
        "epoch_id": EPOCH_ID,
        **HASHES,
        "finalized_block": 7_959_800,
        "revealed_weights": proof.EXPECTED_REVEALED_WEIGHTS,
        "validator_last_update": LAST_UPDATE,
        "independent_verification_pending": True,
    }


def _status_result() -> dict:
    return {
        "ssm_command_id": "status-command",
        "receipt": {
            "schema_version": proof.NATIVE_RECEIPT_SCHEMA_VERSION,
            "stage": "status",
            "status": "ready",
            "evidence": {"automatic_chain_proof": _automatic()},
        },
    }


def _independent() -> dict:
    return {
        "status": "passed",
        "candidate_sha": CANDIDATE,
        "netuid": 401,
        "epoch_id": EPOCH_ID,
        "selected_profile_hash": proof.EXPECTED_PROFILE_HASH,
        "selected_spec_version": proof.EXPECTED_PROFILE_SPEC_VERSION,
        **HASHES,
        "commit_inclusion_block": 7_959_800,
        "commit_inclusion_block_hash": "0x" + "6" * 64,
        "revealed_last_update_block": LAST_UPDATE,
        "finalized_readback_block": 7_959_910,
        "finalized_readback_block_hash": "0x" + "7" * 64,
        "reveal_event": "SubtensorModule.TimelockedWeightsRevealed",
        "reveal_event_block": 7_959_900,
        "reveal_event_block_hash": "0x" + "a" * 64,
        "reveal_event_record_index": 12,
        "reveal_event_subnet_epoch_index": 22_061,
        "revealed_weights": proof.EXPECTED_REVEALED_WEIGHTS,
        "champion_uid": 11,
        "champion_share_exact": "1/4",
    }


def _log_evidence() -> dict:
    return {
        "status": "matched",
        "epoch_id": EPOCH_ID,
        "process_name": "validator_application",
        "cmdline_hash": "sha256:" + "8" * 64,
        "block": 7_959_790,
        "epoch_block": 65,
        "weight_submission_event_hash_prefix": HASHES["weight_submission_event_hash"][
            :20
        ],
        "weight_finalization_event_hash_prefix": HASHES[
            "weight_finalization_event_hash"
        ][:20],
        "marker_line_numbers": [1, 2, 3, 4, 5, 6],
        "raw_log_returned": False,
    }


def test_fixed_verifier_identity_and_stdin_command_are_bounded():
    verifier = proof._load_verifier()
    assert hashlib.sha256(verifier).hexdigest() == proof.VERIFIER_SHA256

    command = proof.verifier_command(
        candidate_sha=CANDIDATE,
        epoch_id=EPOCH_ID,
        verifier=verifier,
    )

    assert command.count("/usr/bin/base64 --decode") == 1
    assert "python3 -I - --epoch-id 22060" in command
    assert "BITTENSOR_NETWORK=test BITTENSOR_NETUID=401" in command
    assert proof.SOURCE_REPOSITORY in command
    assert proof.NATIVE_CONFIG in command
    assert CANDIDATE in command
    assert "status --porcelain --untracked-files=no" in command

    reader = proof._load_log_reader()
    log_command = proof.log_reader_command(
        candidate_sha=CANDIDATE,
        epoch_id=EPOCH_ID,
        run_id=RUN_ID,
        instance_id=INSTANCE_ID,
        reader=reader,
    )
    assert hashlib.sha256(reader).hexdigest() == proof.LOG_READER_SHA256
    assert log_command.count("/usr/bin/base64 --decode") == 1
    assert "python3 -I - --run-id pp-123456-1" in log_command


@pytest.mark.parametrize(
    "value",
    (True, "022060", "22060 ", "22041", str(1 << 64)),
)
def test_epoch_id_rejects_noncanonical_or_out_of_range_values(value):
    with pytest.raises(proof.TemporaryWeightProofError):
        proof._epoch_id(value)


@pytest.mark.parametrize(
    "value",
    (True, "07431466", "7431466 ", "7431465", str(1 << 64)),
)
def test_after_last_update_rejects_noncanonical_or_out_of_range_values(value):
    with pytest.raises(proof.TemporaryWeightProofError):
        proof._after_last_update(value)


def test_join_rejects_a_status_hash_that_differs():
    automatic = _automatic()
    automatic["bundle_hash"] = "sha256:" + "9" * 64
    with pytest.raises(
        proof.TemporaryWeightProofError,
        match="native and independent proof hashes differ",
    ):
        proof._require_join(automatic, _independent())


def test_independent_proof_rejects_the_retired_live_profile_identity():
    value = _independent()
    value["selected_spec_version"] = 454
    value["selected_profile_hash"] = (
        "sha256:a2db2db86ffb10bbf41dd6923e1310726031bc4183841e07e6d2da50e6e58677"
    )

    with pytest.raises(
        proof.TemporaryWeightProofError,
        match="independent proof identity differs",
    ):
        proof._independent_proof(
            json.dumps(value, sort_keys=True) + "\n",
            candidate_sha=CANDIDATE,
            epoch_id=EPOCH_ID,
        )


def test_independent_proof_rejects_reveal_before_commit():
    value = _independent()
    value["reveal_event_block"] = value["commit_inclusion_block"]

    with pytest.raises(
        proof.TemporaryWeightProofError,
        match="independent reveal event bounds differ",
    ):
        proof._independent_proof(
            json.dumps(value, sort_keys=True) + "\n",
            candidate_sha=CANDIDATE,
            epoch_id=EPOCH_ID,
        )


@pytest.mark.parametrize(
    "invalid",
    (
        "sha256:" + LIVE_WEIGHTS_HASH,
        LIVE_WEIGHTS_HASH[:-1],
        "g" * 64,
    ),
)
def test_proof_requires_raw_canonical_live_weights_hash(invalid):
    automatic = _automatic()
    automatic["weights_hash"] = invalid
    status = _status_result()
    status["receipt"]["evidence"]["automatic_chain_proof"] = automatic
    with pytest.raises(
        proof.TemporaryWeightProofError,
        match="automatic native weights hash is invalid",
    ):
        proof._automatic_status_proof(status, epoch_id=EPOCH_ID)

    independent = _independent()
    independent["weights_hash"] = invalid
    with pytest.raises(
        proof.TemporaryWeightProofError,
        match="independent proof weights hash is invalid",
    ):
        proof._independent_proof(
            json.dumps(independent, sort_keys=True) + "\n",
            candidate_sha=CANDIDATE,
            epoch_id=EPOCH_ID,
        )


def test_proof_accepts_exact_public_live_weights_hash():
    assert (
        proof._automatic_status_proof(_status_result(), epoch_id=EPOCH_ID)[
            "weights_hash"
        ]
        == LIVE_WEIGHTS_HASH
    )
    assert (
        proof._independent_proof(
            json.dumps(_independent(), sort_keys=True) + "\n",
            candidate_sha=CANDIDATE,
            epoch_id=EPOCH_ID,
        )["weights_hash"]
        == LIVE_WEIGHTS_HASH
    )


def test_every_static_proof_failure_has_an_allowlisted_reason_code():
    source = (ROOT / "scripts/temporary_testnet401_weight_proof.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    messages = {
        node.exc.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Raise)
        and isinstance(node.exc, ast.Call)
        and isinstance(node.exc.func, ast.Name)
        and node.exc.func.id == "TemporaryWeightProofError"
        and node.exc.args
        and isinstance(node.exc.args[0], ast.Constant)
        and isinstance(node.exc.args[0].value, str)
    }
    assert messages == set(proof.PROOF_FAILURE_REASON_CODES)


def test_main_redacts_arbitrary_exception_text(monkeypatch, tmp_path, capsys):
    canary = "secret-canary-must-not-appear"

    class Session:
        def __init__(self, **_kwargs):
            raise RuntimeError(canary)

    monkeypatch.setattr(proof.boto3.session, "Session", Session)
    state = tmp_path / "proof.json"
    result = proof.main(
        [
            "--region",
            proof.REGION,
            "--run-id",
            RUN_ID,
            "--candidate-sha",
            CANDIDATE,
            "--instance-id",
            INSTANCE_ID,
            "--epoch-id",
            str(EPOCH_ID),
            "--state",
            str(state),
        ]
    )
    captured = capsys.readouterr()
    receipt = json.loads(state.read_text(encoding="utf-8"))
    assert result == 1
    assert receipt == {
        "schema_version": proof.SCHEMA_VERSION,
        "status": "failed",
        "failure_type": "UnexpectedProofError",
        "reason_code": "unexpected_proof_error",
    }
    assert canary not in captured.out
    assert canary not in captured.err
    assert canary not in state.read_text(encoding="utf-8")


def test_known_proof_failure_emits_only_its_allowlisted_reason():
    canary = "independent proof fields differ"
    assert proof._failure_receipt(proof.TemporaryWeightProofError(canary)) == {
        "schema_version": proof.SCHEMA_VERSION,
        "status": "failed",
        "failure_type": "TemporaryWeightProofError",
        "reason_code": "independent_fields_differ",
    }


def test_ssm_failure_preserves_only_bounded_redacted_identity():
    canary = "secret-canary-must-not-appear"
    safe = {
        "schema_version": "leadpoet.temporary_testnet401_ssm_failure.v1",
        "ssm_command_id": "12345678-1234-1234-1234-123456789abc",
        "ssm_status": "Failed",
        "response_code": 1,
        "error_categories": ["RuntimeError", canary],
        "source_locations": [
            {"file": "verify_temporary_testnet_weights.py", "line": 530},
            {"file": canary, "line": 1},
        ],
        "shell_locations": [],
        "raw_output": canary,
    }
    error = proof.TemporaryHostError(
        "fixed temporary SSM stage failed " + json.dumps(safe)
    )
    receipt = proof._failure_receipt(error)
    assert receipt["reason_code"] == "ssm_stage_failed"
    assert receipt["ssm_failure"] == {
        "schema_version": "leadpoet.temporary_testnet401_ssm_failure.v1",
        "ssm_command_id": "12345678-1234-1234-1234-123456789abc",
        "ssm_status": "Failed",
        "response_code": 1,
        "error_categories": ["RuntimeError"],
        "source_locations": [
            {"file": "verify_temporary_testnet_weights.py", "line": 530}
        ],
        "shell_locations": [],
    }
    assert canary not in json.dumps(receipt)


def test_run_requires_last_update_to_advance_and_returns_joined_proof(monkeypatch):
    calls = []
    monkeypatch.setattr(proof, "_require_live_host", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        proof,
        "run_native_stage",
        lambda **kwargs: calls.append(("status", kwargs)) or _status_result(),
    )
    outputs = iter(
        (
            ("proof-command", json.dumps(_independent(), sort_keys=True) + "\n"),
            ("log-command", json.dumps(_log_evidence(), sort_keys=True) + "\n"),
            ("proof-command", json.dumps(_independent(), sort_keys=True) + "\n"),
            ("log-command", json.dumps(_log_evidence(), sort_keys=True) + "\n"),
        )
    )

    def send(_ssm, **kwargs):
        calls.append(("log" if "--run-id" in kwargs["command"] else "proof", kwargs))
        return next(outputs)

    monkeypatch.setattr(proof, "_send_fixed_ssm", send)

    result = proof.run_weight_proof(
        ec2=object(),
        ssm=object(),
        account_id=proof.ACCOUNT_ID,
        region=proof.REGION,
        run_id=RUN_ID,
        candidate_sha=CANDIDATE,
        instance_id=INSTANCE_ID,
        epoch_id=EPOCH_ID,
        after_last_update=LAST_UPDATE - 1,
        now=proof.datetime.now(proof.timezone.utc),
    )

    assert [kind for kind, _kwargs in calls] == ["status", "proof", "log"]
    assert result["status"] == "passed"
    assert result["after_last_update"] == LAST_UPDATE - 1
    assert result["independent_proof"]["revealed_last_update_block"] == LAST_UPDATE
    assert result["automatic_validator_log_evidence"]["status"] == "matched"
    assert result["manual_submission_exclusion"] == (
        "excluded_by_automatic_validator_log_join"
    )

    with pytest.raises(
        proof.TemporaryWeightProofError,
        match="LastUpdate did not advance",
    ):
        proof.run_weight_proof(
            ec2=object(),
            ssm=object(),
            account_id=proof.ACCOUNT_ID,
            region=proof.REGION,
            run_id=RUN_ID,
            candidate_sha=CANDIDATE,
            instance_id=INSTANCE_ID,
            epoch_id=EPOCH_ID,
            after_last_update=LAST_UPDATE,
            now=proof.datetime.now(proof.timezone.utc),
        )


def test_workflow_proof_is_one_fixed_serialized_operation():
    workflow = (ROOT / ".github/workflows/physical-v2-staging.yml").read_text(
        encoding="utf-8"
    )
    assert "- testnet401-proof" in workflow
    assert "- testnet401-inventory" in workflow
    assert "testnet401_epoch_id:" in workflow
    assert "testnet401_after_last_update:" in workflow
    assert "testnet401-proof)" in workflow
    assert "^[0-9]{1,20}$" in workflow
    assert "inputs.operation == 'testnet401-proof'" in workflow
    assert "inputs.operation == 'testnet401-inventory'" in workflow
    assert "scripts/temporary_testnet401_weight_proof.py" in workflow
    assert '--epoch-id "$TESTNET401_EPOCH_ID"' in workflow
    assert '--after-last-update "$TESTNET401_AFTER_LAST_UPDATE"' in workflow
    assert "format('testnet401-{0}'," in workflow
    assert "cleanup-run" in workflow
    assert "temporary-testnet401-inventory.json" in workflow
    temporary_job = workflow[workflow.index("\n  temporary_testnet401:") :]
    assert temporary_job.count("role-to-assume:") == 1
