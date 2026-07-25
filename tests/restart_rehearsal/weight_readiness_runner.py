#!/usr/bin/env python3.11
"""Run the real weight-readiness module against strict external boundaries."""

from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from fastapi import HTTPException


STATE_ROOT = Path(os.environ.get("REHEARSAL_STATE_ROOT", "/rehearsal-state"))
EVENT_PATH = STATE_ROOT / "events.jsonl"
HASH_A = "sha256:" + "a" * 64
HASH_B = "sha256:" + "b" * 64
HASH_C = "sha256:" + "c" * 64
PCR0 = "e" * 96
NOW = "2026-07-25T00:00:00Z"
EPOCH = 99999
NETUID = 71


def _event(kind: str, **details: Any) -> None:
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    if kind in {
        "weight-readiness-boundary",
        "weight-readiness-persistence",
    }:
        details.setdefault("fixture_authenticity", "synthetic")
    payload = {
        "at_ns": time.time_ns(),
        "kind": kind,
        **details,
    }
    descriptor = os.open(
        EVENT_PATH,
        os.O_WRONLY | os.O_CREAT | os.O_APPEND,
        0o600,
    )
    try:
        os.write(
            descriptor,
            (
                json.dumps(payload, sort_keys=True, separators=(",", ":"))
                + "\n"
            ).encode("utf-8"),
        )
    finally:
        os.close(descriptor)


def _candidate_sha() -> str:
    value = os.environ.get("REHEARSAL_CANDIDATE_SHA", "").strip().lower()
    if len(value) != 40:
        raise RuntimeError("rehearsal candidate SHA is invalid")
    return value


def _build_handoff() -> dict[str, Any]:
    from leadpoet_canonical.allocation_handoff_v2 import (
        build_allocation_handoff_v2,
    )
    from leadpoet_canonical.attested_v2 import (
        COORDINATOR_ROLE,
        EMPTY_ARTIFACT_ROOT,
        EMPTY_HOST_OPERATION_ROOT,
        EMPTY_TRANSPORT_ROOT,
        build_boot_identity_body,
        build_execution_receipt_body,
        build_receipt_graph,
        create_boot_identity,
        create_signed_execution_receipt,
        sha256_json,
    )
    from leadpoet_verifier.economics import allocate_research_lab_epoch

    policy = {
        "policy_id": "rehearsal-policy",
        "research_lab_emission_percent": 20.0,
        "reward_epochs": 20,
    }
    allocation = allocate_research_lab_epoch(
        EPOCH,
        policy,
        [],
        [],
        active_source_add_obligations=[],
    )
    source_state = {
        "epoch": EPOCH,
        "netuid": NETUID,
        "policy_id": policy["policy_id"],
        "policy": policy,
        "source_add_obligations": [],
        "reimbursement_obligations": [],
        "champion_obligations": [],
    }
    bundle = {
        "bundle_id": "research_lab_allocation_bundle:restart-rehearsal",
        "bundle_type": "research_lab_live_allocation_bundle",
        "epoch": EPOCH,
        "netuid": NETUID,
        "submission_allowed": True,
        "on_chain_submission_allowed": True,
        "source_state": source_state,
        "source_state_hash": sha256_json(source_state),
        "allocation_doc": allocation,
        "allocation_hash": allocation["allocation_hash"],
    }

    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    ).hex()
    boot = create_boot_identity(
        body=build_boot_identity_body(
            role=COORDINATOR_ROLE,
            physical_role="gateway_coordinator",
            commit_sha=_candidate_sha(),
            pcr0=PCR0,
            build_manifest_hash=HASH_A,
            dependency_lock_hash=HASH_B,
            config_hash=HASH_C,
            boot_nonce="1" * 32,
            signing_pubkey=public_key,
            transport_pubkey="2" * 64,
            transport_certificate_hash=HASH_A,
            attestation_user_data_hash=HASH_B,
            issued_at=NOW,
        ),
        attestation_document_b64=base64.b64encode(
            b"restart-rehearsal-attestation"
        ).decode("ascii"),
    )
    root = create_signed_execution_receipt(
        body=build_execution_receipt_body(
            role=COORDINATOR_ROLE,
            purpose="research_lab.allocation.v2",
            job_id="allocation:%s" % EPOCH,
            epoch_id=EPOCH,
            sequence=0,
            commit_sha=_candidate_sha(),
            pcr0=PCR0,
            build_manifest_hash=HASH_A,
            dependency_lock_hash=HASH_B,
            config_hash=HASH_C,
            boot_identity_hash=boot["boot_identity_hash"],
            input_root=HASH_A,
            output_root=sha256_json({"allocation": allocation}),
            transport_root_hash=EMPTY_TRANSPORT_ROOT,
            host_operation_root_hash=EMPTY_HOST_OPERATION_ROOT,
            artifact_root=EMPTY_ARTIFACT_ROOT,
            parent_receipt_hashes=(),
            status="succeeded",
            failure_code=None,
            issued_at=NOW,
        ),
        enclave_pubkey=public_key,
        sign_digest=private_key.sign,
    )
    graph = build_receipt_graph(
        root_receipt_hash=root["receipt_hash"],
        boot_identities=[boot],
        receipts=[root],
        transport_attempts=[],
        host_operations=[],
    )
    return build_allocation_handoff_v2(
        bundle=bundle,
        receipt_graph=graph,
        lineage_bindings=[],
        lineage_complete=True,
        persistence={"root_receipt_hash": root["receipt_hash"]},
    )


def _run_persistence_probe(
    *,
    get_statuses: tuple[int, ...],
    head_statuses: tuple[int, ...] = (200,),
) -> dict[str, Any]:
    from gateway.tee.artifact_persistence_v2 import (
        ARTIFACT_POLICY_SCHEMA_VERSION,
        ArtifactPersistenceVerifierV2,
    )
    from gateway.tee.artifact_vault_v2 import EncryptedArtifactVaultV2
    from leadpoet_canonical.attested_v2 import canonical_json

    policy = {
        "schema_version": ARTIFACT_POLICY_SCHEMA_VERSION,
        "bucket_host": "restart-rehearsal.s3.us-east-1.amazonaws.com",
        "key_prefix": "/attested-v2/artifacts/",
        "minimum_retention_days": 365,
    }
    query = (
        "X-Amz-Algorithm=AWS4-HMAC-SHA256&"
        "X-Amz-Credential=rehearsal&X-Amz-Date=20260725T000000Z&"
        "X-Amz-Expires=300&X-Amz-SignedHeaders=host&"
        "X-Amz-Signature=rehearsal"
    )
    url = (
        "https://restart-rehearsal.s3.us-east-1.amazonaws.com/"
        "attested-v2/artifacts/item.json?%s" % query
    )
    vault = EncryptedArtifactVaultV2(
        master_key=bytes(range(32)),
        boot_identity_hash=HASH_A,
        retention_days=365,
        clock=lambda: datetime(2026, 7, 25, tzinfo=timezone.utc),
    )
    descriptor = vault.seal(
        b"restart rehearsal artifact",
        job_id="restart-rehearsal",
        purpose="research_lab.allocation.v2",
        artifact_kind="allocation",
    )
    document = vault.export_ciphertext(descriptor["artifact_id"])[
        "storage_document"
    ]
    statuses = {
        "GET": list(get_statuses),
        "HEAD": list(head_statuses),
    }

    def transport(*, method, **_kwargs):
        values = statuses[method]
        status = values.pop(0) if len(values) > 1 else values[0]
        body = (
            canonical_json(document).encode("utf-8")
            if method == "GET" and status == 200
            else b"<Error><Code>ServiceUnavailable</Code></Error>"
        )
        return {
            "http_status": status,
            "headers": {
                "x-amz-object-lock-mode": "COMPLIANCE",
                "x-amz-object-lock-retain-until-date": (
                    "2027-07-25T00:00:00Z"
                ),
            },
            "body": body,
            "tls_peer_chain_hash": HASH_B,
            "tls_protocol": "TLSv1.3",
        }

    result = ArtifactPersistenceVerifierV2(
        vault=vault,
        policy=policy,
        transport=transport,
        clock=lambda: NOW,
    ).verify(
        artifact_id=descriptor["artifact_id"],
        attestation_job_id="restart-rehearsal-persistence",
        artifact_ref=(
            "s3://restart-rehearsal/attested-v2/artifacts/item.json"
        ),
        get_url=url,
        head_url=url,
    )
    _event(
        "weight-readiness-persistence",
        status=result["status"],
        failure_code=result.get("failure_code"),
        attempts=[
            {
                "method": row["method"],
                "attempt_number": row["attempt_number"],
                "terminal_status": row["terminal_status"],
                "http_status": row["http_status"],
                "failure_code": row["failure_code"],
            }
            for row in result["transport_attempts"]
        ],
    )
    return result


def _install_boundaries(stage: str, scenario: str) -> None:
    from gateway.research_lab import api, maintenance
    from gateway.tee import verify_weight_submission_ready_v2 as readiness
    from gateway.utils.tee_artifact_store_v2 import TEEArtifactStoreV2Error
    from research_lab import validator_integration

    direct_calls = {"count": 0}

    async def resolve(_epoch):
        _event("weight-readiness-boundary", boundary="chain_epoch", status="ok")
        return EPOCH

    async def source_rewards(**_kwargs):
        _event(
            "weight-readiness-boundary",
            boundary="source_reward_backfill",
            status="ok",
        )
        return {"ok": True, "migrated_count": 0}

    async def champion_rewards(**_kwargs):
        _event(
            "weight-readiness-boundary",
            boundary="champion_reward_backfill",
            status="ok",
        )
        return {"ok": True, "migrated_count": 0}

    async def settlements(**_kwargs):
        _event(
            "weight-readiness-boundary",
            boundary="settlement_backfill",
            status="ok",
        )
        return {"ok": True, "classified_count": 0}

    async def report(**_kwargs):
        _event(
            "weight-readiness-boundary",
            boundary="cutover_readiness",
            status="ok",
        )
        return {
            "ready": True,
            "receipt_coverage": 1.0,
            "historical_classification_coverage": 1.0,
        }

    async def direct_handoff(epoch, x_leadpoet_internal_key):
        if epoch != EPOCH or x_leadpoet_internal_key is not None:
            raise AssertionError("direct allocation invocation differs")
        direct_calls["count"] += 1
        ordinal = direct_calls["count"]
        if scenario == "transient_503_recovery" and ordinal == 1:
            result = _run_persistence_probe(get_statuses=(503, 503, 503, 503))
        elif scenario == "transient_503_recovery":
            result = _run_persistence_probe(
                get_statuses=(503, 200),
                head_statuses=(503, 200),
            )
        elif scenario == "exhausted_503":
            result = _run_persistence_probe(get_statuses=(503, 503, 503, 503))
        elif scenario == "authenticated_403":
            result = _run_persistence_probe(get_statuses=(403,))
        elif scenario == "success":
            result = _run_persistence_probe(get_statuses=(200,))
        else:
            raise AssertionError("unknown readiness scenario: %s" % scenario)
        if result["status"] != "persisted":
            code = str(result.get("failure_code") or "unknown")
            _event(
                "weight-readiness-boundary",
                boundary="direct_allocation",
                ordinal=ordinal,
                status="failed",
                failure_code=code,
            )
            cause = TEEArtifactStoreV2Error(
                "enclave rejected artifact persistence: %s" % code
            )
            raise HTTPException(
                status_code=500,
                detail="Research Lab attested allocation failed",
            ) from cause
        _event(
            "weight-readiness-boundary",
            boundary="direct_allocation",
            ordinal=ordinal,
            status="ok",
        )
        return _build_handoff()

    def http_handoff(gateway_url, epoch):
        if gateway_url != "http://localhost:8000" or epoch != EPOCH:
            raise AssertionError("HTTP allocation invocation differs")
        _event(
            "weight-readiness-boundary",
            boundary="localhost_allocation_http",
            status="ok",
        )
        return _build_handoff()

    maintenance._resolve_maintenance_epoch = resolve
    maintenance.backfill_source_add_reward_v2_authority = source_rewards
    maintenance.backfill_champion_reward_v2_authority = champion_rewards
    maintenance.backfill_champion_settlement_v2_authority = settlements
    maintenance.champion_v2_cutover_readiness_report = report
    api.get_research_lab_attested_allocation = direct_handoff
    validator_integration.fetch_research_lab_attested_allocation_bundle = (
        http_handoff
    )


def main() -> int:
    os.environ.setdefault("BITTENSOR_NETWORK", "finney")
    os.environ.setdefault("BITTENSOR_NETUID", str(NETUID))
    os.environ.setdefault("RESEARCH_LAB_EMISSION_PERCENT", "20")
    scenario = os.environ.get(
        "REHEARSAL_WEIGHT_READINESS_SCENARIO",
        "transient_503_recovery",
    )
    passthrough = sys.argv[1:]
    stage = (
        "repair"
        if "--repair" in passthrough
        else "http_handoff"
        if "--gateway-url" in passthrough
        else "unknown"
    )

    from gateway.tee import verify_weight_submission_ready_v2 as readiness

    module_path = Path(readiness.__file__).resolve()
    expected_root = Path("/home/ec2-user/leadpoet_repo").resolve()
    if expected_root not in module_path.parents:
        raise RuntimeError(
            "weight readiness did not import from the candidate checkout"
        )
    module_hash = hashlib.sha256(module_path.read_bytes()).hexdigest()
    source_identity = {
        "module_path": str(module_path),
        "module_sha256": module_hash,
        "source_path": str(module_path),
        "source_git_path": (
            "gateway/tee/verify_weight_submission_ready_v2.py"
        ),
        "source_kind": "candidate_checkout",
        "source_sha256": module_hash,
        "candidate_sha": _candidate_sha(),
    }
    _event(
        "weight-readiness",
        status="started",
        stage=stage,
        implementation="production_module",
        **source_identity,
        scenario=scenario,
        argv=passthrough,
    )
    _install_boundaries(stage, scenario)
    original_argv = sys.argv
    sys.argv = [str(module_path), *passthrough]
    try:
        result = int(readiness.main())
    except BaseException as exc:
        _event(
            "weight-readiness",
            status="failed",
            stage=stage,
            implementation="production_module",
            **source_identity,
            scenario=scenario,
            error_type=type(exc).__name__,
            error=str(exc)[:300],
        )
        raise
    finally:
        sys.argv = original_argv
    _event(
        "weight-readiness",
        status="ok",
        stage=stage,
        implementation="production_module",
        **source_identity,
        scenario=scenario,
    )
    return result


if __name__ == "__main__":
    raise SystemExit(main())
