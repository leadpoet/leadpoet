"""Capture an attested stateful subnet-epoch cutover candidate.

This is an explicit shadow/operator path.  Capture never changes the validator's
active epoch mode.  The returned JSON contains the exact finalized boundary and
its complete validator-enclave receipt graph; the optional authenticated publish
step stages that capture as a non-activating gateway candidate.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Dict, Mapping, Optional, Sequence

from Leadpoet.utils.subnet_epoch import SubnetEpochCutover
from leadpoet_canonical.attested_v2 import (
    sha256_json,
    validate_receipt_graph,
    verify_boot_identity_nitro,
)
from leadpoet_canonical.hotkey_authority_v2 import (
    subnet_epoch_candidate_authorization_message_v1,
)
from validator_tee.host.vsock_client import ValidatorEnclaveClient
from validator_tee.host.authoritative_weight_flow_v2 import _post_json
from validator_tee.host.gateway_weight_inputs_v2 import _gateway_endpoint
from validator_tee.host.release_v2 import validator_release_authority


_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_SS58_RE = re.compile(r"^[1-9A-HJ-NP-Za-km-z]{40,64}$")
_SIGNATURE_RE = re.compile(r"^0x[0-9a-f]{128}$")
_PCR0_RE = re.compile(r"^[0-9a-f]{96}$")
_DEFAULT_VALIDATOR_RELEASE_MANIFEST = Path(
    "/home/ec2-user/.config/leadpoet/validator-v2-release-manifest.json"
)


class SubnetEpochBoundaryCaptureV2Error(RuntimeError):
    """A proposed cutover could not be proven by the validator enclave."""


def _candidate_payload_v1(
    *,
    cutover: SubnetEpochCutover,
    capture: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "schema_version": "leadpoet.subnet_epoch_boundary_candidate_submission.v1",
        "cutover_manifest": cutover.to_dict(),
        "capture": dict(capture),
    }


def build_subnet_epoch_candidate_authorization_message_v1(
    *,
    validator_hotkey: str,
    candidate_payload: Mapping[str, Any],
) -> str:
    """Build the domain-separated wallet message verified by the gateway."""

    normalized_hotkey = str(validator_hotkey or "")
    if not _SS58_RE.fullmatch(normalized_hotkey):
        raise SubnetEpochBoundaryCaptureV2Error(
            "candidate validator hotkey is invalid"
        )
    payload_hash = sha256_json(dict(candidate_payload))
    return subnet_epoch_candidate_authorization_message_v1(
        validator_hotkey=normalized_hotkey,
        candidate_payload_hash=payload_hash,
    )


def capture_subnet_epoch_boundary_v2(
    *,
    cutover_manifest: Mapping[str, Any],
    expected_pcr0: str,
    settlement_epoch_id: Optional[int] = None,
    client: Optional[ValidatorEnclaveClient] = None,
    boot_verifier=verify_boot_identity_nitro,
) -> Dict[str, Any]:
    cutover = SubnetEpochCutover.from_mapping(cutover_manifest)
    approved_pcr0 = str(expected_pcr0 or "").strip().lower()
    if not _PCR0_RE.fullmatch(approved_pcr0) or approved_pcr0 == "0" * 96:
        raise SubnetEpochBoundaryCaptureV2Error(
            "approved validator PCR0 is invalid"
        )
    selected_settlement = (
        cutover.first_settlement_epoch_id
        if settlement_epoch_id is None
        else settlement_epoch_id
    )
    if (
        not isinstance(selected_settlement, int)
        or isinstance(selected_settlement, bool)
        or selected_settlement != cutover.first_settlement_epoch_id
    ):
        raise SubnetEpochBoundaryCaptureV2Error(
            "candidate settlement epoch must equal the manifest's first settlement epoch"
        )
    enclave_client = client or ValidatorEnclaveClient()
    try:
        result = enclave_client.capture_subnet_epoch_boundary_v2(
            cutover_manifest=cutover.to_dict(),
            settlement_epoch_id=selected_settlement,
        )
    except Exception as exc:
        raise SubnetEpochBoundaryCaptureV2Error(
            "validator enclave boundary capture failed closed"
        ) from exc
    required_fields = {
        "schema_version",
        "epoch_authority",
        "epoch_boundary",
        "epoch_authority_receipt_hash",
        "epoch_boundary_receipt_hash",
        "receipt_graph",
        "boot_identity",
        "source_artifacts",
    }
    if not isinstance(result, Mapping) or set(result) != required_fields:
        raise SubnetEpochBoundaryCaptureV2Error(
            "validator enclave boundary capture fields are invalid"
        )
    if result.get("schema_version") != "leadpoet.subnet_epoch_boundary_capture.v1":
        raise SubnetEpochBoundaryCaptureV2Error(
            "validator enclave boundary capture schema is invalid"
        )
    current = result.get("epoch_authority")
    boundary = result.get("epoch_boundary")
    graph = result.get("receipt_graph")
    boot = result.get("boot_identity")
    if not all(isinstance(item, Mapping) for item in (current, boundary, graph, boot)):
        raise SubnetEpochBoundaryCaptureV2Error(
            "validator enclave boundary capture documents are invalid"
        )
    identity_fields = {
        "network_genesis_hash": cutover.network_genesis_hash,
        "netuid": cutover.netuid,
        "subnet_epoch_index": cutover.first_subnet_epoch_index,
        "settlement_epoch_id": cutover.first_settlement_epoch_id,
        "cutover_mapping_hash": cutover.mapping_hash,
    }
    for document in (current, boundary):
        if any(document.get(field) != expected for field, expected in identity_fields.items()):
            raise SubnetEpochBoundaryCaptureV2Error(
                "validator enclave boundary capture identity differs from the manifest"
            )
    if (
        int(boundary.get("current_block", -1)) != cutover.cutover_block
        or boundary.get("block_hash") != cutover.cutover_block_hash
        or int(boundary.get("last_epoch_block", -1)) != cutover.cutover_block
        or int(boundary.get("epoch_block", -1)) != 0
    ):
        raise SubnetEpochBoundaryCaptureV2Error(
            "validator enclave boundary does not prove the proposed cutover block"
        )
    try:
        validate_receipt_graph(
            graph,
            required_purposes={"validator.subnet_epoch_snapshot.v2"},
            boot_attestation_verifier=lambda identity: boot_verifier(
                identity,
                expected_pcr0=approved_pcr0,
            ),
            require_boot_attestation_verification=True,
        )
    except Exception as exc:
        raise SubnetEpochBoundaryCaptureV2Error(
            "validator enclave boundary receipt graph is invalid"
        ) from exc
    receipts = {
        str(receipt["receipt_hash"]): receipt for receipt in graph["receipts"]
    }
    current_receipt = receipts.get(str(result["epoch_authority_receipt_hash"]))
    boundary_receipt = receipts.get(str(result["epoch_boundary_receipt_hash"]))
    declared_receipt_hash = str(result["epoch_boundary_receipt_hash"])
    if (
        not isinstance(current_receipt, Mapping)
        or not isinstance(boundary_receipt, Mapping)
        or result["epoch_authority_receipt_hash"] != declared_receipt_hash
        or current_receipt.get("receipt_hash") != declared_receipt_hash
        or boundary_receipt.get("receipt_hash") != declared_receipt_hash
        or dict(current) != dict(boundary)
        or current_receipt.get("output_root") != sha256_json(current)
        or boundary_receipt.get("output_root") != sha256_json(boundary)
        or boundary_receipt.get("parent_receipt_hashes") != []
        or graph.get("root_receipt_hash") != declared_receipt_hash
    ):
        raise SubnetEpochBoundaryCaptureV2Error(
            "validator enclave boundary receipts do not bind the captured documents"
        )
    return dict(result)


def _validate_signed_candidate_submission_v1(
    *,
    submission: Mapping[str, Any],
) -> Dict[str, Any]:
    required_fields = {
        "schema_version",
        "validator_hotkey",
        "validator_hotkey_signature",
        "cutover_manifest",
        "capture",
    }
    if not isinstance(submission, Mapping) or set(submission) != required_fields:
        raise SubnetEpochBoundaryCaptureV2Error(
            "signed candidate submission fields are invalid"
        )
    try:
        cutover = SubnetEpochCutover.from_mapping(
            submission["cutover_manifest"]
        )
    except Exception as exc:
        raise SubnetEpochBoundaryCaptureV2Error(
            "signed candidate cutover manifest is invalid"
        ) from exc
    if submission["cutover_manifest"] != cutover.to_dict():
        raise SubnetEpochBoundaryCaptureV2Error(
            "signed candidate cutover manifest is not canonical"
        )
    capture = submission.get("capture")
    if not isinstance(capture, Mapping):
        raise SubnetEpochBoundaryCaptureV2Error(
            "candidate capture is invalid"
        )
    payload = _candidate_payload_v1(cutover=cutover, capture=capture)
    hotkey = str(submission.get("validator_hotkey") or "")
    signature = str(submission.get("validator_hotkey_signature") or "")
    if not _SS58_RE.fullmatch(hotkey) or not _SIGNATURE_RE.fullmatch(signature):
        raise SubnetEpochBoundaryCaptureV2Error(
            "signed candidate authorization is invalid"
        )
    expected = {
        "schema_version": payload["schema_version"],
        "validator_hotkey": hotkey,
        "validator_hotkey_signature": signature,
        "cutover_manifest": payload["cutover_manifest"],
        "capture": payload["capture"],
    }
    if dict(submission) != expected:
        raise SubnetEpochBoundaryCaptureV2Error(
            "signed candidate submission is not canonical"
        )
    candidate_payload_hash = sha256_json(payload)
    candidate_authorization_hash = sha256_json(
        {
            "validator_hotkey": hotkey,
            "candidate_payload_hash": candidate_payload_hash,
            "validator_hotkey_signature": signature,
        }
    )
    return {
        "submission": expected,
        "cutover": cutover,
        "capture": dict(capture),
        "validator_hotkey": hotkey,
        "candidate_payload_hash": candidate_payload_hash,
        "candidate_authorization_hash": candidate_authorization_hash,
    }


def build_signed_subnet_epoch_candidate_submission_v1(
    *,
    cutover_manifest: Mapping[str, Any],
    capture: Mapping[str, Any],
    wallet: Any,
) -> Dict[str, Any]:
    """Build the exact hotkey-signed JSON body accepted by the gateway."""

    cutover = SubnetEpochCutover.from_mapping(cutover_manifest)
    if not isinstance(capture, Mapping):
        raise SubnetEpochBoundaryCaptureV2Error("candidate capture is invalid")
    payload = _candidate_payload_v1(cutover=cutover, capture=capture)
    hotkey = getattr(getattr(wallet, "hotkey", None), "ss58_address", None)
    message = build_subnet_epoch_candidate_authorization_message_v1(
        validator_hotkey=str(hotkey or ""),
        candidate_payload=payload,
    )
    try:
        raw_signature = wallet.hotkey.sign(message.encode("utf-8"))
    except Exception as exc:
        raise SubnetEpochBoundaryCaptureV2Error(
            "candidate validator wallet signing failed"
        ) from exc
    signature_hex = (
        bytes(raw_signature).hex()
        if isinstance(raw_signature, (bytes, bytearray))
        else str(raw_signature or "").lower().removeprefix("0x")
    )
    signature = "0x" + signature_hex
    if not _SIGNATURE_RE.fullmatch(signature):
        raise SubnetEpochBoundaryCaptureV2Error(
            "candidate validator wallet signature is invalid"
        )
    body = {
        "schema_version": payload["schema_version"],
        "validator_hotkey": str(hotkey),
        "validator_hotkey_signature": signature,
        "cutover_manifest": payload["cutover_manifest"],
        "capture": payload["capture"],
    }
    return _validate_signed_candidate_submission_v1(
        submission=body
    )["submission"]


async def publish_signed_subnet_epoch_candidate_submission_v1(
    *,
    submission: Mapping[str, Any],
    gateway_url: str,
    post_json=_post_json,
    timeout_seconds: float = 300.0,
) -> Dict[str, Any]:
    """Stage one pre-signed shadow capture; never activate the cutover."""

    validated = _validate_signed_candidate_submission_v1(
        submission=submission
    )
    body = validated["submission"]
    cutover = validated["cutover"]
    capture = validated["capture"]
    try:
        acknowledgment = await post_json(
            _gateway_endpoint(gateway_url)
            + "/weights/subnet-epoch/candidate/v1",
            body,
            float(timeout_seconds),
        )
    except Exception as exc:
        raise SubnetEpochBoundaryCaptureV2Error(
            "gateway candidate publication failed closed"
        ) from exc
    expected_fields = {
        "schema_version",
        "candidate_hash",
        "validator_hotkey",
        "candidate_authorization_hash",
        "mapping_hash",
        "subnet_epoch_index",
        "settlement_epoch_id",
        "boundary_block",
        "boundary_hash",
        "boundary_receipt_hash",
        "receipt_graph_hash",
        "durable_readback_hash",
    }
    boundary = capture.get("epoch_boundary")
    if (
        not isinstance(acknowledgment, Mapping)
        or set(acknowledgment) != expected_fields
        or acknowledgment.get("schema_version")
        != "leadpoet.subnet_epoch_boundary_candidate_ack.v1"
        or acknowledgment.get("candidate_hash")
        != validated["candidate_payload_hash"]
        or acknowledgment.get("validator_hotkey")
        != validated["validator_hotkey"]
        or acknowledgment.get("candidate_authorization_hash")
        != validated["candidate_authorization_hash"]
        or not isinstance(boundary, Mapping)
        or acknowledgment.get("mapping_hash") != cutover.mapping_hash
        or int(acknowledgment.get("subnet_epoch_index", -1))
        != cutover.first_subnet_epoch_index
        or int(acknowledgment.get("settlement_epoch_id", -1))
        != cutover.first_settlement_epoch_id
        or int(acknowledgment.get("boundary_block", -1))
        != cutover.cutover_block
        or acknowledgment.get("boundary_hash")
        != sha256_json(dict(boundary))
        or acknowledgment.get("boundary_receipt_hash")
        != capture.get("epoch_boundary_receipt_hash")
        or acknowledgment.get("receipt_graph_hash")
        != sha256_json(dict(capture.get("receipt_graph") or {}))
        or any(
            not _HASH_RE.fullmatch(str(acknowledgment.get(field) or ""))
            for field in ("candidate_hash", "durable_readback_hash")
        )
    ):
        raise SubnetEpochBoundaryCaptureV2Error(
            "gateway candidate acknowledgment differs from the capture"
        )
    return dict(acknowledgment)


async def publish_subnet_epoch_boundary_candidate_v1(
    *,
    cutover_manifest: Mapping[str, Any],
    capture: Mapping[str, Any],
    gateway_url: str,
    wallet: Any,
    post_json=_post_json,
    timeout_seconds: float = 300.0,
) -> Dict[str, Any]:
    """Build, sign, and stage one exact non-activating shadow capture."""

    submission = build_signed_subnet_epoch_candidate_submission_v1(
        cutover_manifest=cutover_manifest,
        capture=capture,
        wallet=wallet,
    )
    return await publish_signed_subnet_epoch_candidate_submission_v1(
        submission=submission,
        gateway_url=gateway_url,
        post_json=post_json,
        timeout_seconds=timeout_seconds,
    )


def write_signed_subnet_epoch_candidate_submission_v1(
    *,
    submission: Mapping[str, Any],
    path: Path,
    overwrite: bool = False,
) -> Path:
    """Atomically write one canonical signed POST body with mode ``0600``."""

    body = _validate_signed_candidate_submission_v1(
        submission=submission
    )["submission"]
    target = Path(path).expanduser()
    if not target.name:
        raise SubnetEpochBoundaryCaptureV2Error(
            "candidate output path is invalid"
        )
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise SubnetEpochBoundaryCaptureV2Error(
            "candidate output directory is unavailable"
        ) from exc
    if target.exists() and not overwrite:
        raise SubnetEpochBoundaryCaptureV2Error(
            "candidate output already exists; explicit overwrite is required"
        )

    descriptor = -1
    temporary = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".%s." % target.name,
            dir=str(target.parent),
        )
        temporary = Path(temporary_name)
        os.fchmod(descriptor, 0o600)
        payload = (
            json.dumps(body, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, target)
            temporary = None
        else:
            try:
                os.link(temporary, target)
            except FileExistsError as exc:
                raise SubnetEpochBoundaryCaptureV2Error(
                    "candidate output already exists; explicit overwrite is required"
                ) from exc
            temporary.unlink()
            temporary = None
        os.chmod(target, 0o600)
        directory_descriptor = os.open(str(target.parent), os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except SubnetEpochBoundaryCaptureV2Error:
        raise
    except OSError as exc:
        raise SubnetEpochBoundaryCaptureV2Error(
            "candidate output could not be written atomically"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
    return target


def _load(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SubnetEpochBoundaryCaptureV2Error(
            "cutover manifest file is unavailable or invalid"
        ) from exc
    if not isinstance(value, dict):
        raise SubnetEpochBoundaryCaptureV2Error(
            "cutover manifest file must contain one JSON object"
        )
    return value


def _approved_validator_pcr0(path: Path) -> str:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        release = validator_release_authority(value)
    except Exception as exc:
        raise SubnetEpochBoundaryCaptureV2Error(
            "approved validator release manifest is unavailable or invalid"
        ) from exc
    pcr0 = str(release.get("pcr0") or "").strip().lower()
    if not _PCR0_RE.fullmatch(pcr0) or pcr0 == "0" * 96:
        raise SubnetEpochBoundaryCaptureV2Error(
            "approved validator release PCR0 is invalid"
        )
    return pcr0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutover-manifest", type=Path, required=True)
    parser.add_argument(
        "--validator-release-manifest",
        type=Path,
        default=Path(
            os.environ.get("VALIDATOR_V2_RELEASE_MANIFEST")
            or _DEFAULT_VALIDATOR_RELEASE_MANIFEST
        ),
    )
    parser.add_argument("--settlement-epoch-id", type=int)
    parser.add_argument("--gateway-url")
    parser.add_argument("--candidate-output", type=Path)
    parser.add_argument(
        "--overwrite-candidate-output",
        action="store_true",
    )
    parser.add_argument("--wallet-name")
    parser.add_argument("--wallet-hotkey")
    parser.add_argument("--wallet-path")
    args = parser.parse_args(argv)
    needs_wallet = bool(args.gateway_url or args.candidate_output)
    if args.overwrite_candidate_output and not args.candidate_output:
        parser.error(
            "--overwrite-candidate-output requires --candidate-output"
        )
    if needs_wallet and (not args.wallet_name or not args.wallet_hotkey):
        parser.error(
            "candidate signing requires --wallet-name and --wallet-hotkey"
        )
    manifest = _load(args.cutover_manifest)
    result = capture_subnet_epoch_boundary_v2(
        cutover_manifest=manifest,
        expected_pcr0=_approved_validator_pcr0(
            args.validator_release_manifest
        ),
        settlement_epoch_id=args.settlement_epoch_id,
    )
    output = {"capture": result}
    if needs_wallet:
        from validator_tee.host.enclave_hotkey_v2 import (
            build_enclave_backed_wallet_v2,
        )

        wallet = build_enclave_backed_wallet_v2(
            name=args.wallet_name,
            hotkey_name=args.wallet_hotkey,
            path=(
                args.wallet_path
                if args.wallet_path
                else str(Path.home() / ".bittensor" / "wallets")
            ),
        )
        signed_submission = build_signed_subnet_epoch_candidate_submission_v1(
            cutover_manifest=manifest,
            capture=result,
            wallet=wallet,
        )
        if args.candidate_output:
            write_signed_subnet_epoch_candidate_submission_v1(
                submission=signed_submission,
                path=args.candidate_output,
                overwrite=args.overwrite_candidate_output,
            )
    if args.gateway_url:
        output["candidate_acknowledgment"] = asyncio.run(
            publish_signed_subnet_epoch_candidate_submission_v1(
                submission=signed_submission,
                gateway_url=args.gateway_url,
            )
        )
    print(json.dumps(output, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
