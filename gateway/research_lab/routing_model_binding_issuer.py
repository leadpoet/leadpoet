"""Host bridge for the protected routing model-binding observation job.

The scoring enclave owns the model execution and receipt signing.  This
module only uploads the exact public observation request through an injected
V2 scoring-job client, then verifies the returned result and receipt chain.
It has no provider client, credential path, or receipt signer of its own.
"""

from __future__ import annotations

import base64
import json
import re
import time
from typing import Any, Mapping, Protocol, Sequence

from gateway.research_lab.routing_model_binding_observation import (
    ROUTING_MODEL_BINDING_OBSERVATION_PURPOSE_V2,
    VerifiedRoutingModelBindingRequirements,
)
from gateway.tee.execution_job_manager_v2 import (
    JOB_SCHEMA_VERSION,
)
from gateway.tee.scoring_executor_v2 import OP_OBSERVE_ROUTING_MODEL_BINDINGS_V2
from leadpoet_canonical.attested_v2 import (
    canonical_json,
    sha256_bytes,
    validate_signed_execution_receipt,
)
from research_lab.canonical import sha256_json
from research_lab.routing_experiments import ProviderBindingIdentity
from gateway.research_lab.routing_experiment_artifacts import (
    VerifiedRoutingArtifactLineage,
)


class RoutingModelBindingObservationIssuerError(RuntimeError):
    """The scoring observation job did not produce an exact trusted result."""


class ScoringJobExecutorV2(Protocol):
    """Minimal coordinator-side client for ``scoring_v2_*`` job RPCs."""

    def submit_job(self, manifest: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def put_chunk(
        self,
        *,
        job_id: str,
        offset: int,
        data_b64: str,
        chunk_sha256: str,
    ) -> Mapping[str, Any]: ...

    def seal_job(self, job_id: str) -> Mapping[str, Any]: ...

    def get_status(self, job_id: str) -> Mapping[str, Any]: ...

    def get_result_chunk(
        self, *, job_id: str, offset: int, max_bytes: int
    ) -> Mapping[str, Any]: ...

    def get_receipts(self, job_id: str) -> Sequence[Mapping[str, Any]]: ...


_TERMINAL_STATES = frozenset({"succeeded", "failed", "cancelled"})
_RELEASE_FIELDS = (
    "commit_sha",
    "pcr0",
    "build_manifest_hash",
    "dependency_lock_hash",
    "config_hash",
    "boot_identity_hash",
    "enclave_pubkey",
)
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def _fail(message: str) -> RoutingModelBindingObservationIssuerError:
    return RoutingModelBindingObservationIssuerError(message)


def _canonical_payload(value: Mapping[str, Any]) -> bytes:
    try:
        return canonical_json(dict(value)).encode("utf-8")
    except Exception as exc:  # noqa: BLE001 - public input fails closed
        raise _fail("routing model observation payload is not canonical") from exc


def _normalize_hashes(values: Sequence[str], name: str) -> list[str]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise _fail(f"routing observation {name} are invalid")
    normalized = [str(value or "").strip().lower() for value in values]
    if any(not _HASH_RE.fullmatch(value) for value in normalized):
        raise _fail(f"routing observation {name} are invalid")
    return sorted(set(normalized))


def _validate_summary(
    value: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    manifest_hash: str,
    payload_size: int,
) -> tuple[str, int]:
    if not isinstance(value, Mapping):
        raise _fail("scoring observation job summary is invalid")
    if (
        value.get("job_id") != manifest["job_id"]
        or value.get("operation") != manifest["operation"]
        or value.get("purpose") != manifest["purpose"]
        or value.get("manifest_hash") != manifest_hash
        or value.get("expected_bytes") != payload_size
    ):
        raise _fail("scoring observation job summary differs from manifest")
    state = str(value.get("state") or "")
    if state not in {"uploading", "queued", "running", "succeeded", "failed", "cancelled"}:
        raise _fail("scoring observation job state is invalid")
    uploaded = value.get("uploaded_bytes")
    if isinstance(uploaded, bool) or not isinstance(uploaded, int):
        raise _fail("scoring observation uploaded byte count is invalid")
    if uploaded < 0 or uploaded > payload_size:
        raise _fail("scoring observation uploaded byte count is outside payload")
    return state, uploaded


def _decode_result(
    executor: ScoringJobExecutorV2,
    *,
    job_id: str,
    chunk_size: int,
) -> tuple[dict[str, Any], str]:
    offset = 0
    chunks: list[bytes] = []
    expected_hash = ""
    expected_total: int | None = None
    max_chunks = 256
    for _ in range(max_chunks):
        raw = executor.get_result_chunk(
            job_id=job_id,
            offset=offset,
            max_bytes=chunk_size,
        )
        if not isinstance(raw, Mapping):
            raise _fail("scoring observation result chunk is invalid")
        if raw.get("job_id") != job_id or raw.get("offset") != offset:
            raise _fail("scoring observation result chunk identity differs")
        try:
            chunk = base64.b64decode(str(raw.get("data_b64") or ""), validate=True)
        except Exception as exc:  # noqa: BLE001
            raise _fail("scoring observation result chunk encoding is invalid") from exc
        if sha256_bytes(chunk) != raw.get("chunk_sha256"):
            raise _fail("scoring observation result chunk hash differs")
        result_hash = str(raw.get("result_sha256") or "")
        if expected_hash and result_hash != expected_hash:
            raise _fail("scoring observation result hash changed")
        expected_hash = result_hash
        total = raw.get("total_size_bytes")
        if isinstance(total, bool) or not isinstance(total, int) or total < 2:
            raise _fail("scoring observation result size is invalid")
        if expected_total is not None and total != expected_total:
            raise _fail("scoring observation result size changed")
        expected_total = total
        chunks.append(chunk)
        offset += len(chunk)
        if raw.get("eof") is True:
            break
        if not chunk:
            raise _fail("scoring observation result did not advance")
    else:
        raise _fail("scoring observation result exceeded chunk limit")
    body = b"".join(chunks)
    if expected_total != len(body) or sha256_bytes(body) != expected_hash:
        raise _fail("scoring observation result hash or size differs")
    try:
        decoded = json.loads(body.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise _fail("scoring observation result is not JSON") from exc
    if not isinstance(decoded, Mapping):
        raise _fail("scoring observation result is not an object")
    try:
        if canonical_json(dict(decoded)).encode("utf-8") != body:
            raise _fail("scoring observation result is not canonical")
    except RoutingModelBindingObservationIssuerError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise _fail("scoring observation result is not canonical") from exc
    return dict(decoded), expected_hash


def _validate_receipt_chain(
    *,
    receipts: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    payload: bytes,
    result: Mapping[str, Any],
    observation: Mapping[str, Any],
) -> Mapping[str, Any]:
    if not isinstance(receipts, Sequence) or isinstance(receipts, (str, bytes)):
        raise _fail("scoring observation receipts are invalid")
    if len(receipts) != 2 or any(not isinstance(item, Mapping) for item in receipts):
        raise _fail("scoring observation receipt chain must contain stage and final receipts")
    stage, final = (dict(receipts[0]), dict(receipts[1]))
    try:
        validate_signed_execution_receipt(stage)
        validate_signed_execution_receipt(final)
    except Exception as exc:
        raise _fail("scoring observation receipt signature is invalid") from exc
    for receipt in (stage, final):
        if receipt.get("role") != "gateway_scoring":
            raise _fail("scoring observation receipt signer role is invalid")
        if receipt.get("purpose") != ROUTING_MODEL_BINDING_OBSERVATION_PURPOSE_V2:
            raise _fail("scoring observation receipt purpose is invalid")
        if receipt.get("status") != "succeeded":
            raise _fail("scoring observation receipt is not successful")
    for field in _RELEASE_FIELDS:
        if stage.get(field) != final.get(field):
            raise _fail("scoring observation receipt release identity differs")
    if final.get("job_id") != manifest["job_id"]:
        raise _fail("scoring observation final receipt job differs")
    if final.get("epoch_id") != manifest["epoch_id"] or final.get("sequence") != manifest["sequence"]:
        raise _fail("scoring observation final receipt epoch or sequence differs")
    if final.get("input_root") != sha256_bytes(payload):
        raise _fail("scoring observation final receipt input differs")
    if final.get("output_root") != sha256_json(dict(result)):
        raise _fail("scoring observation final receipt output differs")
    if final.get("parent_receipt_hashes") != [stage["receipt_hash"]]:
        raise _fail("scoring observation final receipt parent differs")
    expected_stage_job = "stage:%s:0" % manifest["payload_sha256"].split(":", 1)[1][:24]
    if stage.get("job_id") != expected_stage_job:
        raise _fail("scoring observation stage job identity differs")
    if stage.get("epoch_id") != manifest["epoch_id"] or stage.get("sequence") != 0:
        raise _fail("scoring observation stage epoch or sequence differs")
    if stage.get("input_root") != observation.get("request_root"):
        raise _fail("scoring observation stage input differs")
    if stage.get("output_root") != sha256_json(dict(observation)):
        raise _fail("scoring observation stage output differs")
    if stage.get("parent_receipt_hashes") != manifest.get("parent_receipt_hashes"):
        raise _fail("scoring observation stage parent differs")
    return stage


class RoutingModelBindingObservationIssuerV2:
    """Issue one measured model-binding observation through scoring V2."""

    def __init__(
        self,
        *,
        executor: ScoringJobExecutorV2,
        chunk_size: int = 512 * 1024,
        result_chunk_size: int = 512 * 1024,
        poll_interval_seconds: float = 0.05,
        max_wait_seconds: float = 60.0,
        clock: Any = time.monotonic,
        sleeper: Any = time.sleep,
    ) -> None:
        if executor is None:
            raise ValueError("scoring observation executor is required")
        if not isinstance(chunk_size, int) or chunk_size < 1:
            raise ValueError("scoring observation chunk size is invalid")
        if not isinstance(result_chunk_size, int) or result_chunk_size < 1:
            raise ValueError("scoring observation result chunk size is invalid")
        if max_wait_seconds <= 0 or poll_interval_seconds < 0:
            raise ValueError("scoring observation wait settings are invalid")
        self._executor = executor
        self._chunk_size = chunk_size
        self._result_chunk_size = result_chunk_size
        self._poll_interval_seconds = float(poll_interval_seconds)
        self._max_wait_seconds = float(max_wait_seconds)
        self._clock = clock
        self._sleeper = sleeper

    def issue(
        self,
        *,
        job_id: str,
        epoch_id: int,
        sequence: int,
        model_kind: str,
        artifact_lineage: VerifiedRoutingArtifactLineage,
        artifact_document: Mapping[str, Any],
        source_bundle: Mapping[str, Any],
        provider_bindings: Sequence[ProviderBindingIdentity],
        parent_receipt_hashes: Sequence[str] = (),
        input_artifact_hashes: Sequence[str] = (),
    ) -> VerifiedRoutingModelBindingRequirements:
        if not isinstance(artifact_lineage, VerifiedRoutingArtifactLineage):
            raise _fail("routing observation artifact lineage is not verified")
        if model_kind not in {"private", "candidate"}:
            raise _fail("routing observation model kind is invalid")
        if not isinstance(job_id, str) or not job_id:
            raise _fail("routing observation job id is invalid")
        if isinstance(epoch_id, bool) or not isinstance(epoch_id, int) or epoch_id < 0:
            raise _fail("routing observation epoch is invalid")
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
            raise _fail("routing observation sequence is invalid")
        if not isinstance(artifact_document, Mapping) or not isinstance(source_bundle, Mapping):
            raise _fail("routing observation model documents are invalid")
        if (
            not isinstance(provider_bindings, Sequence)
            or isinstance(provider_bindings, (str, bytes))
            or not provider_bindings
            or any(not isinstance(binding, ProviderBindingIdentity) for binding in provider_bindings)
        ):
            raise _fail("routing observation provider bindings are invalid")
        normalized_parent_receipt_hashes = _normalize_hashes(
            parent_receipt_hashes, "parent receipt hashes"
        )
        normalized_input_artifact_hashes = _normalize_hashes(
            input_artifact_hashes, "input artifact hashes"
        )
        payload_doc = {
            "schema_version": "leadpoet.routing_model_binding_request.v2",
            "model_kind": model_kind,
            "artifact_lineage": artifact_lineage.to_dict(),
            "artifact": dict(artifact_document),
            "source_bundle": dict(source_bundle),
            "provider_bindings": [binding.to_dict() for binding in provider_bindings],
        }
        payload = _canonical_payload(payload_doc)
        manifest = {
            "schema_version": JOB_SCHEMA_VERSION,
            "job_id": job_id,
            "operation": OP_OBSERVE_ROUTING_MODEL_BINDINGS_V2,
            "purpose": ROUTING_MODEL_BINDING_OBSERVATION_PURPOSE_V2,
            "epoch_id": epoch_id,
            "sequence": sequence,
            "payload_sha256": sha256_bytes(payload),
            "payload_size_bytes": len(payload),
            "parent_receipt_hashes": normalized_parent_receipt_hashes,
            "input_artifact_hashes": normalized_input_artifact_hashes,
            "provider_credential_profile": "default",
            "provider_credential_ref_hashes": {},
        }
        manifest_hash = sha256_bytes(canonical_json(manifest).encode("utf-8"))
        summary = dict(self._executor.submit_job(manifest))
        state, uploaded = _validate_summary(
            summary, manifest=manifest, manifest_hash=manifest_hash, payload_size=len(payload)
        )
        while state == "uploading":
            if uploaded == len(payload):
                summary = dict(self._executor.seal_job(job_id))
                state, uploaded = _validate_summary(
                    summary, manifest=manifest, manifest_hash=manifest_hash, payload_size=len(payload)
                )
                if state == "uploading":
                    raise _fail("scoring observation job seal did not advance")
                break
            chunk = payload[uploaded : uploaded + self._chunk_size]
            summary = dict(
                self._executor.put_chunk(
                    job_id=job_id,
                    offset=uploaded,
                    data_b64=base64.b64encode(chunk).decode("ascii"),
                    chunk_sha256=sha256_bytes(chunk),
                )
            )
            state, next_uploaded = _validate_summary(
                summary, manifest=manifest, manifest_hash=manifest_hash, payload_size=len(payload)
            )
            if state == "uploading" and next_uploaded <= uploaded:
                raise _fail("scoring observation upload did not advance")
            uploaded = next_uploaded
        deadline = self._clock() + self._max_wait_seconds
        while state not in _TERMINAL_STATES:
            if self._clock() >= deadline:
                raise _fail("scoring observation job timed out")
            self._sleeper(self._poll_interval_seconds)
            summary = dict(self._executor.get_status(job_id))
            state, uploaded = _validate_summary(
                summary, manifest=manifest, manifest_hash=manifest_hash, payload_size=len(payload)
            )
        if state != "succeeded":
            raise _fail("scoring observation job did not succeed")
        result, _result_hash = _decode_result(
            self._executor, job_id=job_id, chunk_size=self._result_chunk_size
        )
        if (
            result.get("schema_version") != "leadpoet.routing_model_binding_result.v2"
            or result.get("operation") != OP_OBSERVE_ROUTING_MODEL_BINDINGS_V2
            or result.get("artifact_lineage_hash") != artifact_lineage.identity_hash()
        ):
            raise _fail("scoring observation result identity differs")
        observation = result.get("observation")
        if not isinstance(observation, Mapping):
            raise _fail("scoring observation result is missing observation")
        stage = _validate_receipt_chain(
            receipts=self._executor.get_receipts(job_id),
            manifest=manifest,
            payload=payload,
            result=result,
            observation=observation,
        )
        try:
            return VerifiedRoutingModelBindingRequirements.from_attested(observation, stage)
        except Exception as exc:
            raise _fail("scoring observation requirement verification failed") from exc


__all__ = [
    "RoutingModelBindingObservationIssuerError",
    "RoutingModelBindingObservationIssuerV2",
    "ScoringJobExecutorV2",
]
