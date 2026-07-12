"""Host bridge for authoritative measured V2 autoresearch execution.

The parent is an untrusted I/O adapter.  It executes only enclave-signed,
allowlisted operations, returns exactly one terminal for each request, verifies
the complete receipt graph, and requires durable sidecar persistence before a
result can be used by the existing hosted worker.
"""

from __future__ import annotations

import asyncio
import base64
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Awaitable, Callable, Dict, Iterable, Mapping, Optional, Sequence

from gateway.tee.autoresearch_executor_v2 import AUTORESEARCH_OPERATIONS_V2
from gateway.tee.execution_job_manager_v2 import (
    JOB_SCHEMA_VERSION,
    PARENT_RECEIPT_GRAPHS_FIELD,
)
from gateway.tee.release_manifest_v2 import (
    role_expectation,
    validate_release_manifest,
)
from gateway.utils.tee_client import autoresearch_tee_client
from leadpoet_canonical.attested_v2 import (
    EMPTY_ARTIFACT_ROOT,
    build_receipt_graph,
    canonical_json,
    merkle_root,
    sha256_bytes,
    validate_receipt_graph,
    validate_signed_execution_receipt,
    validate_signed_host_operation_request,
    verify_boot_identity_nitro,
)


DEFAULT_RELEASE_MANIFEST_PATH = Path(
    "/home/ec2-user/tee/gateway-v2-release-manifest.json"
)
DEFAULT_TIMEOUT_SECONDS = 4 * 60 * 60
DEFAULT_POLL_SECONDS = 0.1
UPLOAD_CHUNK_BYTES = 512 * 1024
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class AttestedAutoresearchV2Error(RuntimeError):
    """A measured loop result is unavailable, stale, or unverifiable."""

    def __init__(
        self,
        message: str,
        *,
        authority: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.authority = dict(authority or {})


def _canonical_bytes(value: Any) -> bytes:
    return canonical_json(value).encode("utf-8")


def _load_release(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AttestedAutoresearchV2Error(
            "V2 release manifest is unavailable"
        ) from exc
    return validate_release_manifest(value)


def _release_boot_verifier(release: Mapping[str, Any]):
    def verify(identity: Mapping[str, Any]) -> Mapping[str, Any]:
        physical_role = str(identity.get("physical_role") or "")
        expectation = role_expectation(release, physical_role)
        for field in (
            "commit_sha",
            "pcr0",
            "build_manifest_hash",
            "dependency_lock_hash",
        ):
            if identity.get(field) != expectation[field]:
                raise AttestedAutoresearchV2Error(
                    "boot %s differs from V2 release" % field
                )
        return verify_boot_identity_nitro(
            identity,
            expected_pcr0=expectation["pcr0"],
        )

    return verify


def derive_autoresearch_job_id_v2(
    *,
    operation: str,
    purpose: str,
    epoch_id: int,
    sequence: int,
    payload_sha256: str,
    parent_receipt_hashes: Sequence[str],
    input_artifact_hashes: Sequence[str],
    release_hash: str,
) -> str:
    digest = sha256_bytes(
        _canonical_bytes(
            {
                "operation": operation,
                "purpose": purpose,
                "epoch_id": epoch_id,
                "sequence": sequence,
                "payload_sha256": payload_sha256,
                "parent_receipt_hashes": sorted(parent_receipt_hashes),
                "input_artifact_hashes": sorted(input_artifact_hashes),
                "release_hash": release_hash,
                "physical_role": "gateway_autoresearch",
            }
        )
    )
    return "autoresearch-v2:%s:%s" % (
        operation.replace("_", "-")[:80],
        digest.split(":", 1)[1][:32],
    )


def _merge_receipt_graphs(
    *,
    root_receipt_hash: str,
    local_boot_identity: Mapping[str, Any],
    local_receipts: Sequence[Mapping[str, Any]],
    transport_attempts: Sequence[Mapping[str, Any]],
    host_operations: Sequence[Mapping[str, Any]],
    parent_graphs: Sequence[Mapping[str, Any]],
    allowed_failed_receipt_hashes: Iterable[str] = (),
) -> Dict[str, Any]:
    boots = {
        str(local_boot_identity["boot_identity_hash"]): dict(local_boot_identity)
    }
    receipts = {}
    attempts = {
        str(item["attempt_hash"]): dict(item) for item in transport_attempts
    }
    operations = {
        str(item["request"]["request_hash"]): dict(item)
        for item in host_operations
    }
    for receipt in local_receipts:
        validate_signed_execution_receipt(receipt)
        key = str(receipt["receipt_hash"])
        if key in receipts:
            raise AttestedAutoresearchV2Error("local receipt is duplicated")
        receipts[key] = dict(receipt)
    if root_receipt_hash not in receipts:
        raise AttestedAutoresearchV2Error("root receipt is missing from local chain")

    local_hashes = set(receipts)
    external_parent_hashes = {
        str(parent_hash)
        for receipt in receipts.values()
        for parent_hash in receipt["parent_receipt_hashes"]
        if str(parent_hash) not in local_hashes
    }
    observed_parent_roots = set()
    for graph in parent_graphs:
        validate_receipt_graph(graph)
        graph_root = str(graph["root_receipt_hash"])
        if graph_root in observed_parent_roots:
            raise AttestedAutoresearchV2Error("parent receipt graph is duplicated")
        observed_parent_roots.add(graph_root)
        for identity in graph["boot_identities"]:
            key = str(identity["boot_identity_hash"])
            if key in boots and boots[key] != dict(identity):
                raise AttestedAutoresearchV2Error(
                    "boot identity conflicts across receipt graphs"
                )
            boots[key] = dict(identity)
        for receipt in graph["receipts"]:
            key = str(receipt["receipt_hash"])
            if key in receipts and receipts[key] != dict(receipt):
                raise AttestedAutoresearchV2Error(
                    "receipt conflicts across receipt graphs"
                )
            receipts[key] = dict(receipt)
        for attempt in graph["transport_attempts"]:
            key = str(attempt["attempt_hash"])
            if key in attempts and attempts[key] != dict(attempt):
                raise AttestedAutoresearchV2Error(
                    "transport attempt conflicts across receipt graphs"
                )
            attempts[key] = dict(attempt)
        for record in graph["host_operations"]:
            key = str(record["request"]["request_hash"])
            if key in operations and operations[key] != dict(record):
                raise AttestedAutoresearchV2Error(
                    "host operation conflicts across receipt graphs"
                )
            operations[key] = dict(record)
    if observed_parent_roots != external_parent_hashes:
        raise AttestedAutoresearchV2Error(
            "parent graphs differ from direct autoresearch ancestry"
        )
    return build_receipt_graph(
        root_receipt_hash=root_receipt_hash,
        boot_identities=[boots[key] for key in sorted(boots)],
        receipts=[receipts[key] for key in sorted(receipts)],
        transport_attempts=[attempts[key] for key in sorted(attempts)],
        host_operations=[operations[key] for key in sorted(operations)],
        allowed_failed_receipt_hashes=allowed_failed_receipt_hashes,
    )


def _request_not_expired(request: Mapping[str, Any]) -> None:
    try:
        expires_at = datetime.strptime(
            str(request["expires_at"]), "%Y-%m-%dT%H:%M:%SZ"
        ).replace(tzinfo=timezone.utc)
    except (KeyError, ValueError) as exc:
        raise AttestedAutoresearchV2Error(
            "host operation expiry is invalid"
        ) from exc
    if datetime.now(timezone.utc) >= expires_at:
        raise AttestedAutoresearchV2Error("host operation request expired")


async def _dispatch_host_operation(
    *,
    command: Mapping[str, Any],
    job_id: str,
    purpose: str,
    boot_identity: Mapping[str, Any],
    handlers: Mapping[
        str,
        Callable[[Mapping[str, Any], Mapping[str, Any]], Any],
    ],
    client: Any,
) -> None:
    if not isinstance(command, Mapping) or set(command) != {"request", "payload"}:
        raise AttestedAutoresearchV2Error("host operation command is malformed")
    request = command.get("request")
    payload = command.get("payload")
    if not isinstance(request, Mapping) or not isinstance(payload, Mapping):
        raise AttestedAutoresearchV2Error("host operation command body is malformed")
    validate_signed_host_operation_request(request)
    if (
        request.get("job_id") != job_id
        or request.get("purpose") != purpose
        or request.get("boot_identity_hash")
        != boot_identity.get("boot_identity_hash")
        or request.get("enclave_pubkey") != boot_identity.get("signing_pubkey")
    ):
        raise AttestedAutoresearchV2Error("host operation scope differs from job")
    if request.get("payload_hash") != sha256_bytes(_canonical_bytes(payload)):
        raise AttestedAutoresearchV2Error("host operation payload hash differs")
    _request_not_expired(request)
    operation = str(request.get("operation") or "")
    handler = handlers.get(operation)
    if handler is None:
        await client.autoresearch_v2_complete_host_operation(
            job_id=job_id,
            request_hash=str(request["request_hash"]),
            terminal_status="failed",
            response=None,
            failure_code="operation_not_configured",
        )
        return
    try:
        response = handler(dict(payload), dict(request))
        if asyncio.iscoroutine(response) or isinstance(response, Awaitable):
            response = await response
        if not isinstance(response, Mapping):
            raise AttestedAutoresearchV2Error(
                "host operation response must be an object"
            )
        await client.autoresearch_v2_complete_host_operation(
            job_id=job_id,
            request_hash=str(request["request_hash"]),
            terminal_status="succeeded",
            response=dict(response),
            failure_code=None,
        )
    except asyncio.CancelledError:
        await client.autoresearch_v2_complete_host_operation(
            job_id=job_id,
            request_hash=str(request["request_hash"]),
            terminal_status="failed",
            response=None,
            failure_code="host_operation_cancelled",
        )
        raise
    except Exception:
        await client.autoresearch_v2_complete_host_operation(
            job_id=job_id,
            request_hash=str(request["request_hash"]),
            terminal_status="failed",
            response=None,
            failure_code="host_operation_failed",
        )


async def execute_autoresearch_v2(
    *,
    operation: str,
    purpose: str,
    epoch_id: int,
    sequence: int,
    payload: Mapping[str, Any],
    host_operation_handlers: Mapping[
        str,
        Callable[[Mapping[str, Any], Mapping[str, Any]], Any],
    ],
    parent_graphs: Sequence[Mapping[str, Any]] = (),
    input_artifact_hashes: Iterable[str] = (),
    provider_credential_profile: str = "default",
    provider_credential_ref_hashes: Optional[Mapping[str, str]] = None,
    release_manifest: Optional[Mapping[str, Any]] = None,
    release_manifest_path: Path = DEFAULT_RELEASE_MANIFEST_PATH,
    client: Any = autoresearch_tee_client,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    persist_graph: Any = None,
    boot_verifier: Any = None,
    persist_transport_artifacts: Any = None,
) -> Dict[str, Any]:
    registry = AUTORESEARCH_OPERATIONS_V2
    if operation not in registry or purpose not in registry[operation]:
        raise AttestedAutoresearchV2Error(
            "autoresearch operation purpose is unauthorized"
        )
    if not isinstance(epoch_id, int) or epoch_id < 0:
        raise AttestedAutoresearchV2Error("epoch_id is invalid")
    if not isinstance(sequence, int) or sequence < 0:
        raise AttestedAutoresearchV2Error("sequence is invalid")
    if not isinstance(host_operation_handlers, Mapping):
        raise AttestedAutoresearchV2Error("host operation handlers are invalid")
    release = (
        validate_release_manifest(release_manifest)
        if release_manifest is not None
        else _load_release(release_manifest_path)
    )
    verifier = boot_verifier or _release_boot_verifier(release)
    health = await client.autoresearch_v2_health()
    if (
        health.get("authority") != "v2_only"
        or health.get("role") != "gateway_autoresearch"
        or health.get("physical_role") != "gateway_autoresearch"
        or health.get("worker_count") != 10
        or not health.get("workers_alive")
    ):
        raise AttestedAutoresearchV2Error("autoresearch enclave health is invalid")
    boot_identity = await client.v2_get_boot_identity()
    verifier(boot_identity)
    if health.get("boot_identity_hash") != boot_identity.get("boot_identity_hash"):
        raise AttestedAutoresearchV2Error("autoresearch health boot differs")

    if PARENT_RECEIPT_GRAPHS_FIELD in payload:
        raise AttestedAutoresearchV2Error(
            "autoresearch payload uses a reserved V2 authority field"
        )
    payload_document = dict(payload)
    if parent_graphs:
        payload_document[PARENT_RECEIPT_GRAPHS_FIELD] = [
            dict(graph) for graph in parent_graphs
        ]
    payload_bytes = _canonical_bytes(payload_document)
    payload_hash = sha256_bytes(payload_bytes)
    credential_refs = {
        str(name): str(digest or "").lower()
        for name, digest in dict(provider_credential_ref_hashes or {}).items()
    }
    if any(
        not re.fullmatch(r"[a-z][a-z0-9_]{1,63}", name)
        or not _HASH_RE.fullmatch(digest)
        for name, digest in credential_refs.items()
    ):
        raise AttestedAutoresearchV2Error(
            "autoresearch provider credential profile is invalid"
        )
    normalized_profile = str(provider_credential_profile or "default")
    if normalized_profile != "default":
        raise AttestedAutoresearchV2Error(
            "autoresearch provider credential profile is unsupported"
        )
    artifact_hashes = sorted(
        {str(item).lower() for item in input_artifact_hashes}
        | set(credential_refs.values())
    )
    parent_roots = sorted(
        str(graph.get("root_receipt_hash") or "") for graph in parent_graphs
    )
    if any(not _HASH_RE.fullmatch(item) for item in artifact_hashes + parent_roots):
        raise AttestedAutoresearchV2Error("autoresearch input commitment is invalid")
    job_id = derive_autoresearch_job_id_v2(
        operation=operation,
        purpose=purpose,
        epoch_id=epoch_id,
        sequence=sequence,
        payload_sha256=payload_hash,
        parent_receipt_hashes=parent_roots,
        input_artifact_hashes=artifact_hashes,
        release_hash=release["release_hash"],
    )
    manifest = {
        "schema_version": JOB_SCHEMA_VERSION,
        "job_id": job_id,
        "operation": operation,
        "purpose": purpose,
        "epoch_id": epoch_id,
        "sequence": sequence,
        "payload_sha256": payload_hash,
        "payload_size_bytes": len(payload_bytes),
        "parent_receipt_hashes": parent_roots,
        "input_artifact_hashes": artifact_hashes,
        "provider_credential_profile": normalized_profile,
        "provider_credential_ref_hashes": dict(sorted(credential_refs.items())),
    }
    submitted = await client.autoresearch_v2_submit_job(manifest)
    if submitted.get("state") == "uploading":
        offset = int(submitted.get("uploaded_bytes") or 0)
        while offset < len(payload_bytes):
            chunk = payload_bytes[offset : offset + UPLOAD_CHUNK_BYTES]
            await client.autoresearch_v2_put_chunk(
                job_id=job_id,
                offset=offset,
                data=chunk,
            )
            offset += len(chunk)
        await client.autoresearch_v2_seal_job(job_id)

    loop = asyncio.get_running_loop()
    deadline = loop.time() + max(1.0, float(timeout_seconds))
    while True:
        command = await client.autoresearch_v2_next_host_operation(
            job_id,
            wait_ms=max(0, min(250, int(poll_seconds * 1000))),
        )
        if command is not None:
            await _dispatch_host_operation(
                command=command,
                job_id=job_id,
                purpose=purpose,
                boot_identity=boot_identity,
                handlers=host_operation_handlers,
                client=client,
            )
        status = await client.autoresearch_v2_get_status(job_id)
        state = str(status.get("state") or "")
        if state in {"succeeded", "failed", "cancelled"}:
            break
        if loop.time() >= deadline:
            await client.autoresearch_v2_cancel_job(job_id)
            raise AttestedAutoresearchV2Error("autoresearch enclave job timed out")
        if command is None:
            await asyncio.sleep(max(0.01, float(poll_seconds)))

    receipt = await client.autoresearch_v2_get_receipt(job_id)
    validate_signed_execution_receipt(receipt)
    if (
        receipt.get("input_root") != payload_hash
        or receipt.get("boot_identity_hash") != boot_identity.get("boot_identity_hash")
    ):
        raise AttestedAutoresearchV2Error("autoresearch receipt binding differs")
    succeeded = state == "succeeded" and receipt.get("status") == "succeeded"
    if succeeded:
        result_bytes = bytearray()
        offset = 0
        while True:
            part = await client.autoresearch_v2_get_result(job_id, offset=offset)
            try:
                chunk = base64.b64decode(
                    str(part.get("data_b64") or ""), validate=True
                )
            except Exception as exc:
                raise AttestedAutoresearchV2Error(
                    "autoresearch result chunk is invalid"
                ) from exc
            if sha256_bytes(chunk) != part.get("chunk_sha256"):
                raise AttestedAutoresearchV2Error(
                    "autoresearch result chunk hash differs"
                )
            result_bytes.extend(chunk)
            offset += len(chunk)
            if part.get("eof"):
                if part.get("result_sha256") != status.get("result_sha256"):
                    raise AttestedAutoresearchV2Error(
                        "autoresearch result hash differs"
                    )
                break
        try:
            result = json.loads(bytes(result_bytes).decode("utf-8"))
        except Exception as exc:
            raise AttestedAutoresearchV2Error(
                "autoresearch result is invalid JSON"
            ) from exc
        if not isinstance(result, dict) or _canonical_bytes(result) != bytes(
            result_bytes
        ):
            raise AttestedAutoresearchV2Error(
                "autoresearch result is not canonical"
            )
        if receipt.get("output_root") != sha256_bytes(bytes(result_bytes)):
            raise AttestedAutoresearchV2Error(
                "autoresearch receipt output differs"
            )
    else:
        failure_code = str(status.get("error_code") or state)
        result = {"status": "failed", "failure_code": failure_code}
        if (
            receipt.get("status") != "failed"
            or receipt.get("failure_code") != failure_code
            or receipt.get("output_root") != sha256_bytes(_canonical_bytes(result))
        ):
            raise AttestedAutoresearchV2Error(
                "autoresearch failure receipt differs from terminal status"
            )

    local_receipts = await client.autoresearch_v2_get_receipts(job_id)
    transport_attempts = await client.autoresearch_v2_get_transport_attempts(job_id)
    job_artifact_hashes = await client.autoresearch_v2_get_artifact_hashes(job_id)
    if not isinstance(job_artifact_hashes, list) or any(
        not _HASH_RE.fullmatch(str(item or "")) for item in job_artifact_hashes
    ):
        raise AttestedAutoresearchV2Error(
            "autoresearch artifact commitment set is invalid"
        )
    expected_artifact_root = (
        merkle_root(job_artifact_hashes, domain="leadpoet-artifact-v2")
        if job_artifact_hashes
        else EMPTY_ARTIFACT_ROOT
    )
    if receipt.get("artifact_root") != expected_artifact_root:
        raise AttestedAutoresearchV2Error(
            "autoresearch artifact root differs from receipt"
        )
    host_operations = await client.autoresearch_v2_get_host_operations(job_id)
    external_graphs = await client.autoresearch_v2_get_external_receipt_graphs(job_id)
    transitions = (
        await client.autoresearch_v2_get_transitions(job_id) if succeeded else []
    )
    allowed_failed = (() if succeeded else (str(receipt["receipt_hash"]),))
    graph = _merge_receipt_graphs(
        root_receipt_hash=str(receipt["receipt_hash"]),
        local_boot_identity=boot_identity,
        local_receipts=local_receipts,
        transport_attempts=transport_attempts,
        host_operations=host_operations,
        parent_graphs=tuple(parent_graphs) + tuple(external_graphs),
        allowed_failed_receipt_hashes=allowed_failed,
    )
    validate_receipt_graph(
        graph,
        required_purposes=(purpose,),
        allowed_failed_receipt_hashes=allowed_failed,
        boot_attestation_verifier=verifier,
        require_boot_attestation_verification=True,
    )
    if job_artifact_hashes:
        if persist_transport_artifacts is None:
            raise AttestedAutoresearchV2Error(
                "transport artifact persistence is required"
            )
        artifact_result = persist_transport_artifacts(
            job_id=job_id,
            purpose=purpose,
            epoch_id=epoch_id,
            sequence=sequence,
            source_receipt=receipt,
            source_graph=graph,
            transport_attempts=transport_attempts,
            execution_artifact_hashes=job_artifact_hashes,
            release_manifest=release,
        )
        if asyncio.iscoroutine(artifact_result) or isinstance(
            artifact_result, Awaitable
        ):
            artifact_result = await artifact_result
        if not isinstance(artifact_result, Mapping):
            raise AttestedAutoresearchV2Error(
                "transport artifact persistence result is invalid"
            )
        final_graph = artifact_result.get("receipt_graph")
        final_receipt = artifact_result.get("receipt")
        if not isinstance(final_graph, Mapping) or not isinstance(
            final_receipt, Mapping
        ):
            raise AttestedAutoresearchV2Error(
                "transport artifact lineage is unavailable"
            )
        validate_receipt_graph(
            final_graph,
            required_purposes=(purpose, "leadpoet.artifact_persistence.v2"),
            allowed_failed_receipt_hashes=allowed_failed,
            boot_attestation_verifier=verifier,
            require_boot_attestation_verification=True,
        )
        outcome = {
            "status": "succeeded" if succeeded else "failed",
            "result": result,
            "execution_receipt": dict(receipt),
            "receipt": dict(final_receipt),
            "receipt_graph": dict(final_graph),
            "transitions": list(transitions),
            "transport_attempts": list(transport_attempts),
            "artifact_hashes": list(job_artifact_hashes),
            "host_operations": list(host_operations),
            "artifact_persistence": dict(artifact_result),
            "release_hash": release["release_hash"],
            "physical_role": "gateway_autoresearch",
        }
        if not succeeded:
            raise AttestedAutoresearchV2Error(
                "autoresearch enclave failed closed: %s" % result["failure_code"],
                authority=outcome,
            )
        return outcome

    if persist_graph is None:
        from gateway.research_lab.attested_v2_store import persist_receipt_graph_v2

        persist_graph = persist_receipt_graph_v2
    if allowed_failed:
        persistence = await persist_graph(
            graph,
            allowed_failed_receipt_hashes=allowed_failed,
        )
    else:
        persistence = await persist_graph(graph)
    if persistence.get("root_receipt_hash") != receipt["receipt_hash"]:
        raise AttestedAutoresearchV2Error(
            "autoresearch receipt durable readback differs"
        )
    outcome = {
        "status": "succeeded" if succeeded else "failed",
        "result": result,
        "receipt": dict(receipt),
        "receipt_graph": graph,
        "transitions": list(transitions),
        "transport_attempts": list(transport_attempts),
        "artifact_hashes": list(job_artifact_hashes),
        "host_operations": list(host_operations),
        "persistence": dict(persistence),
        "release_hash": release["release_hash"],
        "physical_role": "gateway_autoresearch",
    }
    if not succeeded:
        raise AttestedAutoresearchV2Error(
            "autoresearch enclave failed closed: %s" % result["failure_code"],
            authority=outcome,
        )
    return outcome
