"""Fixed authority source for the packaged Research Lab routing release.

This module is part of the release package.  It reads only attested JSON
documents mounted at fixed release paths and uses the existing TEE and
Supabase clients.  It never imports a module named by configuration or by a
request.  The exact model verifier/evaluator/runner authority is deliberately
not fabricated here: the signed model release must publish those objects with
this package before startup can proceed.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
from pathlib import Path
import threading
from typing import Any, Mapping

from gateway.research_lab.routing_release_builder import (
    ReviewedRoutingReleaseAuthoritySources,
    RoutingReleaseDependencyError,
)
from gateway.research_lab.routing_model_binding_observation import (
    VerifiedRoutingModelBindingRequirements,
)
from gateway.utils.tee_client import TEEClient
from research_lab.eval import verify_private_artifact_manifest_signature


_BUNDLE_PATH_ENV = "RESEARCH_LAB_ROUTING_AUTHORITY_BUNDLE_PATH"
_BUNDLE_KEYS_PATH_ENV = "RESEARCH_LAB_ROUTING_AUTHORITY_KEYS_PATH"
_GOLD_LABEL_PATH_ENV = "RESEARCH_LAB_ROUTING_GOLD_LABEL_DOCUMENT_PATH"
_MODEL_OBSERVATION_PATH_ENV = "RESEARCH_LAB_ROUTING_MODEL_OBSERVATION_PATH"
_PROTECTED_RECEIPT_PATH_ENV = "RESEARCH_LAB_ROUTING_PROTECTED_RECEIPT_PATH"
_TEE_CID_ENV = "RESEARCH_LAB_ROUTING_TEE_CID"
_MAX_DOCUMENT_BYTES = 16 * 1024 * 1024


def _release_document(env_name: str, label: str) -> Mapping[str, Any]:
    raw_path = str(os.environ.get(env_name) or "").strip()
    path = Path(raw_path)
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise RoutingReleaseDependencyError(
            f"routing release {label} document is not an attested regular file"
        )
    try:
        if path.stat().st_size > _MAX_DOCUMENT_BYTES:
            raise ValueError("oversize")
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - release input is fail closed
        raise RoutingReleaseDependencyError(
            f"routing release {label} document cannot be loaded"
        ) from exc
    if not isinstance(value, Mapping):
        raise RoutingReleaseDependencyError(
            f"routing release {label} document is not an object"
        )
    return dict(value)


def _pinned_keys() -> Mapping[str, str]:
    value = _release_document(_BUNDLE_KEYS_PATH_ENV, "authority key")
    if not value or any(not isinstance(key, str) or not isinstance(item, str) for key, item in value.items()):
        raise RoutingReleaseDependencyError("routing release authority key pins are invalid")
    return dict(value)


def _run(coro: Any) -> Any:
    """Run one TEE coroutine without nesting an event loop."""

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    result: list[Any] = []
    error: list[BaseException] = []

    def worker() -> None:
        try:
            result.append(asyncio.run(coro))
        except BaseException as exc:  # noqa: BLE001 - preserve RPC failure
            error.append(exc)

    thread = threading.Thread(target=worker, name="routing-release-tee-rpc")
    thread.start()
    thread.join()
    if error:
        raise error[0]
    return result[0]


class AttestedRoutingTeeJobRpc:
    """The six-method routing job surface over the existing TEE client."""

    def __init__(self, client: TEEClient) -> None:
        if not isinstance(client, TEEClient):
            raise RoutingReleaseDependencyError("routing release TEE client is invalid")
        self._client = client

    def submit_job(self, manifest: Mapping[str, Any]) -> Mapping[str, Any]:
        return dict(_run(self._client.scoring_v2_submit_job(dict(manifest))))

    def put_chunk(
        self, *, job_id: str, offset: int, data_b64: str, chunk_sha256: str
    ) -> Mapping[str, Any]:
        try:
            data = base64.b64decode(str(data_b64), validate=True)
        except Exception as exc:
            raise RoutingReleaseDependencyError("routing release TEE chunk is invalid") from exc
        result = _run(
            self._client.scoring_v2_put_chunk(
                job_id=str(job_id), offset=int(offset), data=data
            )
        )
        if str(result.get("chunk_sha256") or chunk_sha256) != str(chunk_sha256):
            raise RoutingReleaseDependencyError("routing release TEE chunk hash differs")
        return dict(result)

    def seal(self, job_id: str) -> Mapping[str, Any]:
        return dict(_run(self._client.scoring_v2_seal_job(str(job_id))))

    def status(self, job_id: str) -> Mapping[str, Any]:
        return dict(_run(self._client.scoring_v2_get_status(str(job_id))))

    def result(
        self, job_id: str, *, offset: int = 0, max_bytes: int = 512 * 1024
    ) -> Mapping[str, Any]:
        return dict(
            _run(
                self._client.scoring_v2_get_result(
                    str(job_id), offset=int(offset), max_bytes=int(max_bytes)
                )
            )
        )

    def receipts(self, job_id: str) -> tuple[Mapping[str, Any], ...]:
        return tuple(
            dict(item)
            for item in _run(self._client.scoring_v2_get_receipts(str(job_id)))
        )


def _verify_gold_label(document: Mapping[str, Any], key_id: str) -> Mapping[str, Any]:
    """Verify labels with the same KMS-backed manifest verifier as artifacts."""

    result = verify_private_artifact_manifest_signature(document)
    if result.get("verified") is not True or str(result.get("key_id") or "") != key_id:
        raise RoutingReleaseDependencyError("routing release gold-label signature is invalid")
    return dict(result)


def load_reviewed_routing_release_authority_sources() -> ReviewedRoutingReleaseAuthoritySources:
    """Load the fixed source documents and concrete runtime authorities.

    The release-owned model verifier, evaluator, and runner factory are not
    present in this consumer repository.  Failing here is intentional until
    the signed model release publishes those exact objects; returning a test
    substitute would make the Lab appear live while bypassing model ownership.
    """

    bundle = _release_document(_BUNDLE_PATH_ENV, "authority bundle")
    keys = _pinned_keys()
    gold = _release_document(_GOLD_LABEL_PATH_ENV, "gold-label")
    observation = _release_document(_MODEL_OBSERVATION_PATH_ENV, "model observation")
    protected_receipt = _release_document(_PROTECTED_RECEIPT_PATH_ENV, "protected release")
    try:
        model_observation = VerifiedRoutingModelBindingRequirements.from_attested(
            observation["result"], observation["receipt"]
        )
    except Exception as exc:  # noqa: BLE001 - signed observation is fail closed
        raise RoutingReleaseDependencyError(
            "routing release model binding observation is invalid"
        ) from exc
    try:
        cid = int(str(os.environ.get(_TEE_CID_ENV) or ""))
    except (TypeError, ValueError) as exc:
        raise RoutingReleaseDependencyError("routing release TEE CID is invalid") from exc
    _tee_rpc = AttestedRoutingTeeJobRpc(TEEClient(cid=cid))
    # These are concrete authorities.  The remaining four fields cannot be
    # inferred from documents and therefore remain an explicit release gate.
    raise RoutingReleaseDependencyError(
        "routing release exact model verifier, evaluator, and runner factory are not published"
    )


__all__ = [
    "AttestedRoutingTeeJobRpc",
    "load_reviewed_routing_release_authority_sources",
]
