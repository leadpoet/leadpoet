"""Validate or ingest one signed subnet-epoch boundary candidate offline.

The gateway main process and its workers can remain stopped while this command
runs.  The default is a read-only validation pass.  Durable candidate and
receipt-graph inserts require both ``--apply`` and an exact payload-hash
confirmation.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import re
import sys
from typing import Any, Awaitable, Callable, Dict, Mapping, Optional, Sequence

from leadpoet_canonical.attested_v2 import sha256_json


REPORT_SCHEMA_VERSION = "leadpoet.subnet_epoch_candidate_ingest_report.v1"
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_ACK_FIELDS = frozenset(
    {
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
)


class StatefulEpochCandidateIngestError(RuntimeError):
    """The signed candidate file or its authoritative validation is invalid."""


def load_signed_candidate_v1(path: Path) -> Dict[str, Any]:
    """Load exactly one JSON object without accepting shape coercion."""

    try:
        value = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StatefulEpochCandidateIngestError(
            "signed candidate file is unavailable or invalid JSON"
        ) from exc
    if not isinstance(value, dict):
        raise StatefulEpochCandidateIngestError(
            "signed candidate file must contain one JSON object"
        )
    return value


def load_validator_release_manifest_v2(path: Path) -> Dict[str, Any]:
    """Load and validate one explicitly selected six-build validator release."""

    try:
        value = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StatefulEpochCandidateIngestError(
            "approved validator release manifest is unavailable or invalid JSON"
        ) from exc
    if not isinstance(value, dict):
        raise StatefulEpochCandidateIngestError(
            "approved validator release manifest must contain one JSON object"
        )
    try:
        from validator_tee.host.release_v2 import (
            validate_validator_release_manifest,
        )

        return validate_validator_release_manifest(value)
    except Exception as exc:
        raise StatefulEpochCandidateIngestError(
            "approved validator release manifest is invalid"
        ) from exc


def build_validator_release_boot_verifier_v1(
    manifest: Mapping[str, Any],
    *,
    nitro_verifier: Optional[Callable[..., Mapping[str, Any]]] = None,
) -> Callable[[Mapping[str, Any]], Mapping[str, Any]]:
    """Bind a candidate boot to an operator-selected independent release."""

    try:
        from validator_tee.host.release_v2 import validator_release_authority

        release = validator_release_authority(manifest)
    except Exception as exc:
        raise StatefulEpochCandidateIngestError(
            "approved validator release manifest is invalid"
        ) from exc
    if nitro_verifier is None:
        from leadpoet_canonical.attested_v2 import verify_boot_identity_nitro

        nitro_verifier = verify_boot_identity_nitro

    expected = {
        "role": "validator_weights",
        "physical_role": "validator_weights",
        "commit_sha": release["commit_sha"],
        "pcr0": release["pcr0"],
        "build_manifest_hash": release["app_manifest_hash"],
        "dependency_lock_hash": release["dependency_lock_hash"],
    }

    def verify(identity: Mapping[str, Any]) -> Mapping[str, Any]:
        if not isinstance(identity, Mapping):
            raise ValueError("candidate validator boot identity is invalid")
        for field, expected_value in expected.items():
            if identity.get(field) != expected_value:
                raise ValueError(
                    "candidate validator boot differs from approved release at %s"
                    % field
                )
        return nitro_verifier(
            identity,
            expected_pcr0=release["pcr0"],
        )

    return verify


def candidate_payload_hash_v1(value: Mapping[str, Any]) -> str:
    """Return the exact domain payload hash signed by the validator hotkey."""

    required = {"schema_version", "cutover_manifest", "capture"}
    if not required.issubset(value):
        raise StatefulEpochCandidateIngestError(
            "signed candidate file is missing its payload fields"
        )
    return sha256_json(
        {
            "schema_version": value["schema_version"],
            "cutover_manifest": value["cutover_manifest"],
            "capture": value["capture"],
        }
    )


async def _preview_candidate_row(
    envelope: Mapping[str, Any],
    **kwargs: Any,
) -> Dict[str, Any]:
    """Build the exact SQL row while deliberately performing no persistence."""

    from gateway.research_lab.stateful_epoch_authority_v1 import (
        build_pre_cutover_candidate_row_v1,
    )

    return build_pre_cutover_candidate_row_v1(
        envelope,
        cutover=kwargs["cutover"],
        validator_hotkey=kwargs["validator_hotkey"],
        candidate_payload_hash=kwargs["candidate_payload_hash"],
        validator_hotkey_signature=kwargs["validator_hotkey_signature"],
        candidate_authorization_hash=kwargs["candidate_authorization_hash"],
    )


async def ingest_subnet_epoch_candidate_v1(
    value: Mapping[str, Any],
    *,
    apply: bool = False,
    boot_attestation_verifier: Optional[
        Callable[[Mapping[str, Any]], Mapping[str, Any]]
    ] = None,
    apply_stage: Optional[Callable[..., Awaitable[Mapping[str, Any]]]] = None,
    preview_stage: Optional[
        Callable[..., Awaitable[Mapping[str, Any]]]
    ] = None,
) -> Dict[str, Any]:
    """Run the route's full authority checks with a durable or no-write sink."""

    from fastapi import HTTPException
    from gateway.api import weights as weights_api

    if not isinstance(value, Mapping):
        raise StatefulEpochCandidateIngestError(
            "signed candidate must be one object"
        )
    if boot_attestation_verifier is None:
        raise StatefulEpochCandidateIngestError(
            "an explicitly approved validator release manifest is required"
        )
    raw = dict(value)
    try:
        submission = weights_api.SubnetEpochCandidateSubmissionV1.model_validate(
            raw
        )
    except Exception as exc:
        raise StatefulEpochCandidateIngestError(
            "signed candidate fields are invalid"
        ) from exc
    if submission.model_dump(mode="python") != raw:
        raise StatefulEpochCandidateIngestError(
            "signed candidate fields are not canonical"
        )

    payload_hash = candidate_payload_hash_v1(raw)
    try:
        if apply:
            from gateway.research_lab.stateful_epoch_authority_v1 import (
                persist_pre_cutover_candidate_v1,
            )

            stage = apply_stage or weights_api._stage_subnet_epoch_candidate_v1
            acknowledgment = await stage(
                submission,
                persist_candidate=persist_pre_cutover_candidate_v1,
                boot_attestation_verifier=boot_attestation_verifier,
            )
        else:
            stage = preview_stage or weights_api._stage_subnet_epoch_candidate_v1
            acknowledgment = await stage(
                submission,
                persist_candidate=_preview_candidate_row,
                boot_attestation_verifier=boot_attestation_verifier,
            )
    except HTTPException as exc:
        raise StatefulEpochCandidateIngestError(
            "candidate authority validation failed closed "
            f"(HTTP {exc.status_code})"
        ) from exc
    except StatefulEpochCandidateIngestError:
        raise
    except Exception as exc:
        raise StatefulEpochCandidateIngestError(
            "candidate authority validation failed closed "
            f"({type(exc).__name__})"
        ) from exc

    if not isinstance(acknowledgment, Mapping):
        raise StatefulEpochCandidateIngestError(
            "candidate authority acknowledgment is invalid"
        )
    ack = dict(acknowledgment)
    from Leadpoet.utils.subnet_epoch import SubnetEpochCutover

    cutover = SubnetEpochCutover.from_mapping(raw["cutover_manifest"])
    capture = raw["capture"]
    expected_authorization_hash = sha256_json(
        {
            "validator_hotkey": submission.validator_hotkey,
            "candidate_payload_hash": payload_hash,
            "validator_hotkey_signature": submission.validator_hotkey_signature,
        }
    )
    expected_ack = {
        "candidate_hash": payload_hash,
        "validator_hotkey": submission.validator_hotkey,
        "candidate_authorization_hash": expected_authorization_hash,
        "mapping_hash": cutover.mapping_hash,
        "subnet_epoch_index": cutover.first_subnet_epoch_index,
        "settlement_epoch_id": cutover.first_settlement_epoch_id,
        "boundary_block": cutover.cutover_block,
        "boundary_hash": sha256_json(capture["epoch_boundary"]),
        "boundary_receipt_hash": capture["epoch_boundary_receipt_hash"],
        "receipt_graph_hash": sha256_json(capture["receipt_graph"]),
    }
    hash_fields = (
        "candidate_hash",
        "candidate_authorization_hash",
        "mapping_hash",
        "boundary_hash",
        "boundary_receipt_hash",
        "receipt_graph_hash",
        "durable_readback_hash",
    )
    if (
        set(ack) != _ACK_FIELDS
        or ack.get("schema_version")
        != "leadpoet.subnet_epoch_boundary_candidate_ack.v1"
        or any(ack.get(field) != value for field, value in expected_ack.items())
        or any(not _HASH_RE.fullmatch(str(ack.get(field) or "")) for field in hash_fields)
    ):
        raise StatefulEpochCandidateIngestError(
            "candidate authority acknowledgment differs"
        )

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "mode": "apply" if apply else "dry_run",
        "status": "durably_staged" if apply else "validated_no_writes",
        "candidate_payload_hash": payload_hash,
        "candidate_authorization_hash": ack.get(
            "candidate_authorization_hash"
        ),
        "mapping_hash": ack.get("mapping_hash"),
        "boundary_hash": ack.get("boundary_hash"),
        "boundary_receipt_hash": ack.get("boundary_receipt_hash"),
        "receipt_graph_hash": ack.get("receipt_graph_hash"),
    }
    if apply:
        # Persistence is content-addressed and idempotent.  The route proves
        # exact durable state but intentionally does not claim whether this
        # invocation inserted it or replayed an already-identical row.
        report["write_mode"] = "insert_or_exact_replay"
        report["durable_readback_hash"] = ack.get("durable_readback_hash")
    else:
        report["writes_applied"] = False
        report["preview_row_hash"] = ack.get("durable_readback_hash")
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument(
        "--validator-release-manifest",
        type=Path,
        required=True,
        help="explicit six-build validator release manifest used for boot verification",
    )
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--confirm-candidate-payload-hash")
    args = parser.parse_args(argv)

    try:
        value = load_signed_candidate_v1(args.candidate)
        validator_release = load_validator_release_manifest_v2(
            args.validator_release_manifest
        )
        boot_verifier = build_validator_release_boot_verifier_v1(
            validator_release
        )
        payload_hash = candidate_payload_hash_v1(value)
    except StatefulEpochCandidateIngestError as exc:
        print(
            json.dumps(
                {
                    "schema_version": REPORT_SCHEMA_VERSION,
                    "status": "failed",
                    "error": str(exc),
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 2

    if args.apply and args.confirm_candidate_payload_hash != payload_hash:
        parser.error(
            "--apply requires --confirm-candidate-payload-hash equal to "
            "the signed candidate payload hash"
        )

    try:
        report = asyncio.run(
            ingest_subnet_epoch_candidate_v1(
                value,
                apply=bool(args.apply),
                boot_attestation_verifier=boot_verifier,
            )
        )
    except StatefulEpochCandidateIngestError as exc:
        print(
            json.dumps(
                {
                    "schema_version": REPORT_SCHEMA_VERSION,
                    "status": "failed",
                    "candidate_payload_hash": payload_hash,
                    "error": str(exc),
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 2

    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
