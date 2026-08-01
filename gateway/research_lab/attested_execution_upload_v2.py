"""Deterministic upload convergence for immutable V2 execution jobs."""

from __future__ import annotations

import math
from typing import Any, Awaitable, Callable, Mapping

from leadpoet_canonical.attested_v2 import canonical_json, sha256_bytes


class AttestedExecutionUploadV2Error(RuntimeError):
    """An immutable execution job could not be uploaded without ambiguity."""


_POST_UPLOAD_STATES = frozenset(
    {"queued", "running", "succeeded", "failed", "cancelled"}
)


def _validated_summary(
    value: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    manifest_hash: str,
    payload_size: int,
) -> tuple[str, int]:
    if not isinstance(value, Mapping):
        raise AttestedExecutionUploadV2Error("V2 job summary is invalid")
    if (
        value.get("job_id") != manifest.get("job_id")
        or value.get("operation") != manifest.get("operation")
        or value.get("purpose") != manifest.get("purpose")
        or value.get("manifest_hash") != manifest_hash
        or value.get("expected_bytes") != payload_size
    ):
        raise AttestedExecutionUploadV2Error(
            "V2 job summary differs from immutable manifest"
        )
    state = str(value.get("state") or "")
    if state not in {"uploading"} | _POST_UPLOAD_STATES:
        raise AttestedExecutionUploadV2Error("V2 job state is invalid")
    uploaded = value.get("uploaded_bytes")
    if isinstance(uploaded, bool) or not isinstance(uploaded, int):
        raise AttestedExecutionUploadV2Error("V2 uploaded byte count is invalid")
    if uploaded < 0 or uploaded > payload_size:
        raise AttestedExecutionUploadV2Error(
            "V2 uploaded byte count is outside payload"
        )
    return state, uploaded


async def upload_attested_execution_job_v2(
    *,
    manifest: Mapping[str, Any],
    payload: bytes,
    chunk_size: int,
    submit_job: Callable[[Mapping[str, Any]], Awaitable[Mapping[str, Any]]],
    put_chunk: Callable[..., Awaitable[Mapping[str, Any]]],
    seal_job: Callable[[str], Awaitable[Mapping[str, Any]]],
    get_status: Callable[[str], Awaitable[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Upload once or converge on an identical concurrent upload.

    A failed chunk call is reconciled only when the exact immutable job has
    made monotonic progress or has already advanced beyond uploading.
    """

    if not isinstance(payload, bytes) or not payload:
        raise AttestedExecutionUploadV2Error("V2 job payload is invalid")
    if (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, int)
        or chunk_size < 1
    ):
        raise AttestedExecutionUploadV2Error("V2 upload chunk size is invalid")
    if (
        manifest.get("payload_size_bytes") != len(payload)
        or manifest.get("payload_sha256") != sha256_bytes(payload)
    ):
        raise AttestedExecutionUploadV2Error(
            "V2 job payload differs from immutable manifest"
        )

    job_id = str(manifest.get("job_id") or "")
    manifest_hash = sha256_bytes(
        canonical_json(dict(manifest)).encode("utf-8")
    )
    summary = dict(await submit_job(dict(manifest)))
    state, uploaded = _validated_summary(
        summary,
        manifest=manifest,
        manifest_hash=manifest_hash,
        payload_size=len(payload),
    )
    progress_limit = math.ceil(len(payload) / chunk_size) + 2
    progress_steps = 0

    while state == "uploading":
        progress_steps += 1
        if progress_steps > progress_limit:
            raise AttestedExecutionUploadV2Error(
                "V2 job upload exceeded bounded progress"
            )
        if uploaded == len(payload):
            try:
                summary = dict(await seal_job(job_id))
            except Exception as exc:
                try:
                    summary = dict(await get_status(job_id))
                except Exception as status_exc:
                    raise AttestedExecutionUploadV2Error(
                        "V2 job seal reconciliation failed"
                    ) from status_exc
                state, uploaded = _validated_summary(
                    summary,
                    manifest=manifest,
                    manifest_hash=manifest_hash,
                    payload_size=len(payload),
                )
                if state == "uploading":
                    raise AttestedExecutionUploadV2Error(
                        "V2 job seal failed before state advanced"
                    ) from exc
                break
            state, uploaded = _validated_summary(
                summary,
                manifest=manifest,
                manifest_hash=manifest_hash,
                payload_size=len(payload),
            )
            if state == "uploading":
                raise AttestedExecutionUploadV2Error(
                    "V2 job seal did not advance state"
                )
            break

        offset = uploaded
        chunk = payload[offset : offset + chunk_size]
        try:
            summary = dict(
                await put_chunk(job_id=job_id, offset=offset, data=chunk)
            )
        except Exception as exc:
            try:
                summary = dict(await get_status(job_id))
            except Exception as status_exc:
                raise AttestedExecutionUploadV2Error(
                    "V2 chunk reconciliation failed"
                ) from status_exc
            state, uploaded = _validated_summary(
                summary,
                manifest=manifest,
                manifest_hash=manifest_hash,
                payload_size=len(payload),
            )
            if state == "uploading" and uploaded <= offset:
                raise AttestedExecutionUploadV2Error(
                    "V2 chunk failed without monotonic progress"
                ) from exc
            continue

        state, uploaded = _validated_summary(
            summary,
            manifest=manifest,
            manifest_hash=manifest_hash,
            payload_size=len(payload),
        )
        if state == "uploading" and uploaded <= offset:
            raise AttestedExecutionUploadV2Error(
                "V2 chunk upload made no progress"
            )

    return summary
