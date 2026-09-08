from __future__ import annotations

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
HASHES = {
    "authority_hash": "sha256:" + "1" * 64,
    "bundle_hash": "sha256:" + "2" * 64,
    "weights_hash": "sha256:" + "3" * 64,
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
        "selected_spec_version": 454,
        **HASHES,
        "commit_inclusion_block": 7_959_800,
        "commit_inclusion_block_hash": "0x" + "6" * 64,
        "revealed_last_update_block": LAST_UPDATE,
        "finalized_readback_block": 7_959_910,
        "finalized_readback_block_hash": "0x" + "7" * 64,
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
