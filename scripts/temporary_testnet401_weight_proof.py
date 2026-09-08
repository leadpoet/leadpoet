#!/usr/bin/env python3
"""Run one fixed independent testnet401 weight proof through the retained host."""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shlex
import stat
import sys
from typing import Any, Mapping, Sequence

import boto3
from botocore.exceptions import BotoCoreError, ClientError

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.temporary_testnet_signing_host import (
    ACCOUNT_ID,
    NATIVE_CONFIG,
    NATIVE_RECEIPT_SCHEMA_VERSION,
    REGION,
    SOURCE_REPOSITORY,
    SOURCE_VENV,
    TemporaryHostError,
    _require_live_host,
    _send_fixed_ssm,
    run_native_stage,
)


SCHEMA_VERSION = "leadpoet.temporary_testnet401_weight_proof_ssm.v1"
LOG_READER_SHA256 = "14c53ff6be8e26e617b8e6f5e6196c0055f21852981d4ca9227c745b981372cd"
VERIFIER_PATH = Path(__file__).with_name("verify_temporary_testnet_weights.py")
VERIFIER_SHA256 = (
    "05d9472549d9f5b8788a6b57f636ca05ca6e037560bc3daacc93c8586eecf6a1"
)
MAX_VERIFIER_BYTES = 32 * 1024
FIRST_SETTLEMENT_EPOCH_ID = 22_042
BASELINE_LAST_UPDATE = 7_431_466
NETUID = 401
EXPECTED_PROFILE_HASH = (
    "sha256:20407ea2590f8d93be932db8dd578d2feb1ec3f1c71331495d224ef48057824f"
)
EXPECTED_PROFILE_SPEC_VERSION = 455
EXPECTED_REVEALED_WEIGHTS = [[0, 65_535], [11, 21_845]]
HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
RESULT_FIELDS = frozenset(
    {
        "status",
        "candidate_sha",
        "netuid",
        "epoch_id",
        "selected_profile_hash",
        "selected_spec_version",
        "authority_hash",
        "bundle_hash",
        "weights_hash",
        "weight_submission_event_hash",
        "weight_finalization_event_hash",
        "commit_inclusion_block",
        "commit_inclusion_block_hash",
        "revealed_last_update_block",
        "finalized_readback_block",
        "finalized_readback_block_hash",
        "revealed_weights",
        "champion_uid",
        "champion_share_exact",
    }
)


class TemporaryWeightProofError(TemporaryHostError):
    """The fixed independent proof did not match the retained native evidence."""


def _epoch_id(value: Any) -> int:
    if isinstance(value, bool):
        raise TemporaryWeightProofError("testnet401 proof epoch is invalid")
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise TemporaryWeightProofError("testnet401 proof epoch is invalid") from exc
    if str(normalized) != str(value) or not FIRST_SETTLEMENT_EPOCH_ID <= normalized < 1 << 64:
        raise TemporaryWeightProofError("testnet401 proof epoch is invalid")
    return normalized


def _after_last_update(value: Any) -> int | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        raise TemporaryWeightProofError("prior testnet401 LastUpdate is invalid")
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise TemporaryWeightProofError(
            "prior testnet401 LastUpdate is invalid"
        ) from exc
    if (
        str(normalized) != str(value)
        or not BASELINE_LAST_UPDATE <= normalized < 1 << 64
    ):
        raise TemporaryWeightProofError("prior testnet401 LastUpdate is invalid")
    return normalized


def _load_verifier(path: Path = VERIFIER_PATH) -> bytes:
    try:
        metadata = path.lstat()
        value = path.read_bytes()
    except OSError as exc:
        raise TemporaryWeightProofError("fixed proof verifier is unavailable") from exc
    if (
        not stat.S_ISREG(metadata.st_mode)
        or path.is_symlink()
        or not value
        or len(value) > MAX_VERIFIER_BYTES
        or hashlib.sha256(value).hexdigest() != VERIFIER_SHA256
    ):
        raise TemporaryWeightProofError("fixed proof verifier identity differs")
    return value


def _load_log_reader(
    path: Path = Path(__file__).with_name(
        "extract_temporary_testnet401_validator_log.py"
    ),
) -> bytes:
    try:
        metadata = path.lstat()
        value = path.read_bytes()
    except OSError as exc:
        raise TemporaryWeightProofError("fixed log reader is unavailable") from exc
    if (
        not stat.S_ISREG(metadata.st_mode)
        or path.is_symlink()
        or not value
        or len(value) > 32 * 1024
        or hashlib.sha256(value).hexdigest() != LOG_READER_SHA256
    ):
        raise TemporaryWeightProofError("fixed log reader identity differs")
    return value


def verifier_command(*, candidate_sha: str, epoch_id: int, verifier: bytes) -> str:
    epoch = _epoch_id(epoch_id)
    if not re.fullmatch(r"[0-9a-f]{40}", candidate_sha):
        raise TemporaryWeightProofError("testnet401 proof candidate is invalid")
    if (
        not verifier
        or len(verifier) > MAX_VERIFIER_BYTES
        or hashlib.sha256(verifier).hexdigest() != VERIFIER_SHA256
    ):
        raise TemporaryWeightProofError("fixed proof verifier identity differs")
    q = shlex.quote
    encoded = base64.b64encode(verifier).decode("ascii")
    return "\n".join(
        (
            "set -Eeuo pipefail",
            f"test -f {q(NATIVE_CONFIG)}",
            f"test -x {q(SOURCE_VENV + '/bin/python3')}",
            f"test \"$(/usr/bin/git -C {q(SOURCE_REPOSITORY)} rev-parse HEAD)\" = {q(candidate_sha)}",
            f"test -z \"$(/usr/bin/git -C {q(SOURCE_REPOSITORY)} "
            "status --porcelain --untracked-files=no)\"",
            f"cd {q(SOURCE_REPOSITORY)}",
            f"printf '%s' {q(encoded)} | /usr/bin/base64 --decode | "
            "BITTENSOR_NETWORK=test BITTENSOR_NETUID=401 "
            "PYTHONDONTWRITEBYTECODE=1 "
            f"{q(SOURCE_VENV + '/bin/python3')} -I - --epoch-id {epoch}",
        )
    )


def log_reader_command(
    *, candidate_sha: str, epoch_id: int, run_id: str, instance_id: str, reader: bytes
) -> str:
    epoch = _epoch_id(epoch_id)
    if not re.fullmatch(r"[0-9a-f]{40}", candidate_sha):
        raise TemporaryWeightProofError("testnet401 proof candidate is invalid")
    q = shlex.quote
    encoded = base64.b64encode(reader).decode("ascii")
    return "\n".join(
        (
            "set -Eeuo pipefail",
            f"test -f {q(NATIVE_CONFIG)}",
            f"test -x {q(SOURCE_VENV + '/bin/python3')}",
            f"cd {q(SOURCE_REPOSITORY)}",
            f"printf '%s' {q(encoded)} | /usr/bin/base64 --decode | "
            f"PYTHONDONTWRITEBYTECODE=1 {q(SOURCE_VENV + '/bin/python3')} -I - "
            f"--run-id {q(run_id)} --candidate-sha {q(candidate_sha)} "
            f"--instance-id {q(instance_id)} --epoch-id {epoch}",
        )
    )


def _automatic_status_proof(
    status_result: Mapping[str, Any], *, epoch_id: int
) -> dict[str, Any]:
    receipt = status_result.get("receipt")
    if (
        not isinstance(receipt, Mapping)
        or receipt.get("schema_version") != NATIVE_RECEIPT_SCHEMA_VERSION
        or receipt.get("stage") != "status"
        or receipt.get("status") != "ready"
    ):
        raise TemporaryWeightProofError("retained native status is not ready")
    evidence = receipt.get("evidence")
    proof = evidence.get("automatic_chain_proof") if isinstance(evidence, Mapping) else None
    if (
        not isinstance(proof, Mapping)
        or proof.get("status") != "candidate_match"
        or proof.get("independent_verification_pending") is not True
        or _epoch_id(proof.get("epoch_id")) != epoch_id
    ):
        raise TemporaryWeightProofError("automatic native proof does not match the epoch")
    required_hashes = (
        "authority_hash",
        "bundle_hash",
        "weights_hash",
        "weight_submission_event_hash",
        "weight_finalization_event_hash",
    )
    if any(HASH_RE.fullmatch(str(proof.get(name) or "")) is None for name in required_hashes):
        raise TemporaryWeightProofError("automatic native proof hashes are invalid")
    return dict(proof)


def _independent_proof(
    stdout: str, *, candidate_sha: str, epoch_id: int
) -> dict[str, Any]:
    if len(stdout.encode("utf-8")) > 64 * 1024 or stdout.count("\n") != 1:
        raise TemporaryWeightProofError("independent proof output is invalid")
    try:
        value = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise TemporaryWeightProofError("independent proof output is invalid") from exc
    if not isinstance(value, Mapping) or set(value) != RESULT_FIELDS:
        raise TemporaryWeightProofError("independent proof fields differ")
    if (
        value.get("status") != "passed"
        or value.get("candidate_sha") != candidate_sha
        or value.get("netuid") != NETUID
        or _epoch_id(value.get("epoch_id")) != epoch_id
        or value.get("selected_profile_hash") != EXPECTED_PROFILE_HASH
        or value.get("selected_spec_version") != EXPECTED_PROFILE_SPEC_VERSION
        or value.get("revealed_weights") != EXPECTED_REVEALED_WEIGHTS
        or value.get("champion_uid") != 11
        or value.get("champion_share_exact") != "1/4"
    ):
        raise TemporaryWeightProofError("independent proof identity differs")
    for name in (
        "authority_hash",
        "bundle_hash",
        "weights_hash",
        "weight_submission_event_hash",
        "weight_finalization_event_hash",
    ):
        if HASH_RE.fullmatch(str(value.get(name) or "")) is None:
            raise TemporaryWeightProofError("independent proof hashes are invalid")
    for name in (
        "commit_inclusion_block",
        "revealed_last_update_block",
        "finalized_readback_block",
    ):
        if not isinstance(value.get(name), int) or isinstance(value.get(name), bool):
            raise TemporaryWeightProofError("independent proof blocks are invalid")
    return dict(value)


def _require_join(
    automatic: Mapping[str, Any], independent: Mapping[str, Any]
) -> None:
    for name in (
        "authority_hash",
        "bundle_hash",
        "weights_hash",
        "weight_submission_event_hash",
        "weight_finalization_event_hash",
    ):
        if automatic.get(name) != independent.get(name):
            raise TemporaryWeightProofError("native and independent proof hashes differ")
    if (
        automatic.get("revealed_weights") != independent.get("revealed_weights")
        or automatic.get("finalized_block") != independent.get("commit_inclusion_block")
        or automatic.get("validator_last_update")
        != independent.get("revealed_last_update_block")
    ):
        raise TemporaryWeightProofError("native and independent chain readback differs")


def _log_evidence(stdout: str, *, epoch_id: int) -> dict[str, Any]:
    if len(stdout.encode("utf-8")) > 8192 or stdout.count("\n") != 1:
        raise TemporaryWeightProofError("automatic validator log evidence is invalid")
    try:
        value = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise TemporaryWeightProofError(
            "automatic validator log evidence is invalid"
        ) from exc
    fields = {
        "status",
        "epoch_id",
        "process_name",
        "cmdline_hash",
        "block",
        "epoch_block",
        "weight_submission_event_hash_prefix",
        "weight_finalization_event_hash_prefix",
        "marker_line_numbers",
        "raw_log_returned",
    }
    prefix = re.compile(r"^sha256:[0-9a-f]{13}$")
    if (
        not isinstance(value, Mapping)
        or set(value) != fields
        or value.get("status") != "matched"
        or _epoch_id(value.get("epoch_id")) != epoch_id
        or value.get("process_name") != "validator_application"
        or HASH_RE.fullmatch(str(value.get("cmdline_hash") or "")) is None
        or prefix.fullmatch(str(value.get("weight_submission_event_hash_prefix") or ""))
        is None
        or prefix.fullmatch(
            str(value.get("weight_finalization_event_hash_prefix") or "")
        )
        is None
        or value.get("raw_log_returned") is not False
    ):
        raise TemporaryWeightProofError("automatic validator log evidence differs")
    lines = value.get("marker_line_numbers")
    if not isinstance(lines, list) or len(lines) != 6 or lines != sorted(set(lines)):
        raise TemporaryWeightProofError("automatic validator log marker order differs")
    return dict(value)


def run_weight_proof(
    *,
    ec2: Any,
    ssm: Any,
    account_id: str,
    region: str,
    run_id: str,
    candidate_sha: str,
    instance_id: str,
    epoch_id: int,
    after_last_update: int | None,
    now: datetime,
    verifier_path: Path = VERIFIER_PATH,
) -> dict[str, Any]:
    epoch = _epoch_id(epoch_id)
    previous_last_update = _after_last_update(after_last_update)
    if account_id != ACCOUNT_ID or region != REGION:
        raise TemporaryWeightProofError("temporary proof AWS identity differs")
    _require_live_host(
        ec2,
        instance_id=instance_id,
        run_id=run_id,
        candidate_sha=candidate_sha,
        now=now,
    )
    status_result = run_native_stage(
        ec2=ec2,
        ssm=ssm,
        account_id=account_id,
        region=region,
        run_id=run_id,
        candidate_sha=candidate_sha,
        instance_id=instance_id,
        stage="status",
        now=now,
    )
    automatic = _automatic_status_proof(status_result, epoch_id=epoch)
    command_id, stdout = _send_fixed_ssm(
        ssm,
        instance_id=instance_id,
        command=verifier_command(
            candidate_sha=candidate_sha,
            epoch_id=epoch,
            verifier=_load_verifier(verifier_path),
        ),
        timeout_seconds=900,
    )
    independent = _independent_proof(
        stdout, candidate_sha=candidate_sha, epoch_id=epoch
    )
    _require_join(automatic, independent)
    log_command_id, log_stdout = _send_fixed_ssm(
        ssm,
        instance_id=instance_id,
        command=log_reader_command(
            candidate_sha=candidate_sha,
            epoch_id=epoch,
            run_id=run_id,
            instance_id=instance_id,
            reader=_load_log_reader(),
        ),
        timeout_seconds=120,
    )
    log_evidence = _log_evidence(log_stdout, epoch_id=epoch)
    if not independent["weight_submission_event_hash"].startswith(
        log_evidence["weight_submission_event_hash_prefix"]
    ) or not independent["weight_finalization_event_hash"].startswith(
        log_evidence["weight_finalization_event_hash_prefix"]
    ):
        raise TemporaryWeightProofError("automatic validator log hashes differ")
    if (
        previous_last_update is not None
        and independent["revealed_last_update_block"] <= previous_last_update
    ):
        raise TemporaryWeightProofError(
            "finalized LastUpdate did not advance after the prior proof"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "run_id": run_id,
        "candidate_sha": candidate_sha,
        "instance_id": instance_id,
        "epoch_id": epoch,
        "after_last_update": previous_last_update,
        "status_ssm_command_id": status_result["ssm_command_id"],
        "proof_ssm_command_id": command_id,
        "actor_log_ssm_command_id": log_command_id,
        "automatic_native_proof": automatic,
        "independent_proof": independent,
        "automatic_validator_log_evidence": log_evidence,
        "manual_submission_exclusion": "excluded_by_automatic_validator_log_join",
    }


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--candidate-sha", required=True)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--epoch-id", required=True)
    parser.add_argument("--after-last-update")
    parser.add_argument("--state", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        epoch = _epoch_id(args.epoch_id)
        session = boto3.session.Session(region_name=args.region)
        result = run_weight_proof(
            ec2=session.client("ec2"),
            ssm=session.client("ssm"),
            account_id=str(session.client("sts").get_caller_identity()["Account"]),
            region=args.region,
            run_id=args.run_id,
            candidate_sha=str(args.candidate_sha).lower(),
            instance_id=args.instance_id,
            epoch_id=epoch,
            after_last_update=_after_last_update(args.after_last_update),
            now=datetime.now(timezone.utc).replace(microsecond=0),
        )
        _write(args.state, result)
    except (BotoCoreError, ClientError, OSError, TemporaryHostError) as exc:
        _write(
            args.state,
            {
                "schema_version": SCHEMA_VERSION,
                "status": "failed",
                "failure_type": type(exc).__name__,
            },
        )
        print("ERROR: fixed temporary testnet401 proof failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
