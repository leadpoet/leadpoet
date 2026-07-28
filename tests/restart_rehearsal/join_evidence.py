#!/usr/bin/env python3.11
"""Join launcher and V2 workflow evidence into one fail-closed manifest."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Sequence


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SystemExit(f"rehearsal evidence is unreadable: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"rehearsal evidence is not an object: {path}")
    return value


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _launcher_path(
    root: Path,
    ordinal: int,
    component: str,
    transition: str,
    candidate_sha: str,
) -> Path:
    return root / (f"{ordinal}-{component}-{transition}-{candidate_sha}.json")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--from-sha", required=True)
    parser.add_argument("--candidate-sha", required=True)
    parser.add_argument("--profile", choices=("prepush", "release"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    required = [
        _launcher_path(
            args.evidence_root,
            1,
            component,
            "forward",
            args.candidate_sha,
        )
        for component in ("gateway", "validator")
    ]
    if args.profile == "release":
        required = [
            _launcher_path(
                args.evidence_root,
                ordinal,
                component,
                transition,
                candidate,
            )
            for ordinal, transition, candidate in (
                (1, "forward", args.candidate_sha),
                (2, "rollback", args.from_sha),
                (3, "forward", args.candidate_sha),
            )
            for component in ("gateway", "validator")
        ]
    workflow_path = args.evidence_root / "workflow.json"
    required.append(workflow_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(f"rehearsal evidence is incomplete: {missing}")

    launcher_rows = []
    for path in required:
        if path == workflow_path:
            continue
        value = _load(path)
        if (
            value.get("status") != "passed"
            or value.get("scope") != "exact"
            or value.get("candidate_sha")
            not in {
                args.candidate_sha,
                args.from_sha,
            }
            or not re.fullmatch(
                r"[0-9a-f]{64}",
                str(value.get("postgres_contract_sha256") or ""),
            )
        ):
            raise SystemExit(f"launcher evidence did not pass exactly: {path}")
        launcher_rows.append(
            {
                "run_ordinal": int(path.name.split("-", 1)[0]),
                "component": value.get("component"),
                "candidate_sha": value.get("candidate_sha"),
                "from_sha": value.get("from_sha"),
                "event_count": value.get("event_count"),
                "pcr0": value.get("pcr0"),
                "postgres_contract_sha256": value.get("postgres_contract_sha256"),
                "evidence_hash": _sha256(path),
            }
        )

    postgres_contracts_by_candidate: dict[str, set[str]] = {}
    for row in launcher_rows:
        postgres_contracts_by_candidate.setdefault(
            str(row["candidate_sha"]),
            set(),
        ).add(str(row["postgres_contract_sha256"]))
    if any(
        len(contract_hashes) != 1
        for contract_hashes in postgres_contracts_by_candidate.values()
    ):
        raise SystemExit("gateway and validator migration-backed contracts differ")
    candidate_postgres_contract_sha256 = next(
        iter(postgres_contracts_by_candidate[args.candidate_sha])
    )

    workflow = _load(workflow_path)
    expected_epochs = 1 if args.profile == "prepush" else 100
    epochs = workflow.get("epochs")
    workflow_stages = workflow.get("stages")
    if (
        workflow.get("status") != "passed"
        or workflow.get("profile") != args.profile
        or workflow.get("release_sha") != args.candidate_sha
        or workflow.get("epoch_count") != expected_epochs
        or not isinstance(epochs, list)
        or len(epochs) != expected_epochs
        or not isinstance(workflow_stages, list)
        or not workflow_stages
    ):
        raise SystemExit("production workflow evidence is incomplete")
    stage_status = {
        str(item.get("stage")): item.get("status")
        for item in workflow_stages
        if isinstance(item, dict)
    }
    required_workflow_stages = {
        "input-contract",
        "diagnostic:candidate-bundle-generation",
        "diagnostic:host-bundle-composition",
        "diagnostic:primary-bundle-verification",
        "diagnostic:auditor-bundle-verification",
        "diagnostic:primary-auditor-vector-equality",
        "diagnostic:sdk-signing-bridge",
        "boundary-start",
        "boundary-cleanup",
        "workflow-evidence-validation",
        *{
            f"epoch-{30_000 + ordinal}"
            for ordinal in range(expected_epochs)
        },
    }
    source_stage_count = sum(
        stage.startswith("source-identity:") for stage in stage_status
    )
    fault_stage_count = sum(
        stage.startswith("fault:") for stage in stage_status
    )
    if (
        len(stage_status) != len(workflow_stages)
        or any(status != "passed" for status in stage_status.values())
        or not required_workflow_stages.issubset(stage_status)
        or source_stage_count != 9
        or (
            args.profile == "release"
            and (
                stage_status.get("concurrency") != "passed"
                or fault_stage_count < 15
            )
        )
    ):
        raise SystemExit("production workflow stage evidence is incomplete")
    pcr0s = {str(epoch.get("pcr0")) for epoch in epochs}
    forward_pcr0s = {
        str(row.get("pcr0"))
        for row in launcher_rows
        if row.get("candidate_sha") == args.candidate_sha
    }
    if (
        len(pcr0s) != 1
        or forward_pcr0s != pcr0s
        or any(
            not epoch.get("receipt_ancestry_verified")
            or not epoch.get("canonical_vector_equal")
            or not epoch.get("auditor_verified")
            or not epoch.get("auditor_runtime_verified")
            or not epoch.get("sdk_bridge_verified")
            or not epoch.get("signed_extrinsic_hash")
            or epoch.get("last_update") != epoch.get("finalized_block")
            or not epoch.get("reveal_vector_hash")
            for epoch in epochs
        )
    ):
        raise SystemExit("epoch evidence does not form one complete authority")
    cleanup = workflow.get("cleanup")
    if (
        not isinstance(cleanup, dict)
        or cleanup.get("pending_faults") != 0
        or cleanup.get("boundary_thread_alive_after_close") is not False
        or cleanup.get("local_chain_epochs") != expected_epochs
    ):
        raise SystemExit("rehearsal cleanup evidence is incomplete")
    if args.profile == "release" and (
        len(workflow.get("fault_matrix") or []) < 15
        or workflow.get("concurrent_write_count") != 32
    ):
        raise SystemExit("release fault or concurrency evidence is incomplete")

    last_epoch = epochs[-1]
    joined = {
        "schema_version": "leadpoet.local_restart_rehearsal_evidence.v1",
        "status": "passed",
        "profile": args.profile,
        "release_sha": args.candidate_sha,
        "from_sha": args.from_sha,
        "pcr0": next(iter(pcr0s)),
        "bundle_hash": last_epoch["bundle_hash"],
        "receipt_ancestry": {
            "root_receipt_hash": last_epoch["root_receipt_hash"],
            "publication_receipt_hash": last_epoch["publication_receipt_hash"],
            "finalization_receipt_hash": last_epoch["finalization_receipt_hash"],
            "verified": True,
        },
        "canonical_vector": {
            "hash": last_epoch["canonical_vector_hash"],
            "primary_equals_auditor": True,
        },
        "auditor": {
            "production_runtime_verified": True,
            "submission_finalized": True,
        },
        "signed_extrinsic": {
            "authorization_hash": last_epoch["extrinsic_authorization_hash"],
            "extrinsic_hash": last_epoch["signed_extrinsic_hash"],
            "sdk_commit_request_hash": last_epoch["sdk_commit_request_hash"],
            "sdk_extrinsic_request_hash": last_epoch["sdk_extrinsic_request_hash"],
        },
        "finalization": {
            "block": last_epoch["finalized_block"],
            "last_update": last_epoch["last_update"],
        },
        "reveal": {"vector_hash": last_epoch["reveal_vector_hash"]},
        "epoch_count": expected_epochs,
        "launcher_evidence": launcher_rows,
        "postgres_contract_sha256": candidate_postgres_contract_sha256,
        "workflow_evidence_hash": _sha256(workflow_path),
        "fault_matrix_count": len(workflow.get("fault_matrix") or []),
        "concurrent_write_count": workflow.get("concurrent_write_count"),
        "cleanup": cleanup,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    args.output.write_text(
        json.dumps(joined, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(
        f"RESTART_REHEARSAL_SUCCESS profile={args.profile} "
        f"sha={args.candidate_sha} epochs={expected_epochs}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
