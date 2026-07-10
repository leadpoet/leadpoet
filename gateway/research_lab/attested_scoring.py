"""Shadow/required bridge to scoring operations in the existing gateway EIF.

The host remains authoritative while mode is ``shadow``. This module submits
immutable inputs, verifies the returned bytes and receipt, and compares the
enclave output without performing database writes or provider calls itself.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping

from gateway.build_info import get_build_info
from gateway.tee.scoring_job_manager import JOB_SCHEMA_VERSION
from gateway.utils.tee_client import tee_client
from leadpoet_canonical.attested_receipts import (
    validate_signed_receipt,
    verify_receipt_lineage,
)


logger = logging.getLogger(__name__)

MODE_ENV = "RESEARCH_LAB_ATTESTED_SCORING_MODE"
TIMEOUT_ENV = "RESEARCH_LAB_ATTESTED_SCORING_TIMEOUT_SECONDS"
POLL_SECONDS_ENV = "RESEARCH_LAB_ATTESTED_SCORING_POLL_SECONDS"
PERSIST_RECEIPTS_ENV = "RESEARCH_LAB_ATTESTED_RECEIPT_PERSIST_ENABLED"
LIVE_PROVIDER_ENV = "RESEARCH_LAB_ATTESTED_SCORING_LIVE_PROVIDER_ENABLED"
MODES = frozenset({"off", "shadow", "required"})
SCORE_RECEIPT_PURPOSES = frozenset(
    {
        "research_lab.candidate_score.v1",
        "research_lab.baseline_score.v1",
        "research_lab.benchmark.v1",
        "research_lab.rebenchmark.v1",
    }
)
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")


class AttestedScoringError(RuntimeError):
    """Required-mode attested scoring failed or diverged."""


def _log_attestation_error(message: str, *args: Any) -> None:
    """Keep attestation failures visible after Bittensor logging reconfiguration."""

    target = logger if logger.isEnabledFor(logging.ERROR) else logging.getLogger()
    target.error(message, *args)


def verify_gateway_receipt_attestation(
    *,
    receipt: Mapping[str, Any],
    expected_purpose: str,
    expected_epoch_id: int,
) -> tuple[bool, dict[str, Any]]:
    """Verify genuine Nitro and extract hardware PCRs for independent checking."""

    from leadpoet_canonical.nitro import verify_nitro_attestation_full

    return verify_nitro_attestation_full(
        attestation_b64=str(receipt.get("attestation_document_b64") or ""),
        expected_pubkey=str(receipt.get("enclave_pubkey") or ""),
        expected_purpose=expected_purpose,
        expected_epoch_id=expected_epoch_id,
        role="gateway",
        # The executing host verifies AWS authenticity here. The validator and
        # auditors must separately compare this extracted PCR0 with a clean,
        # Git-derived gateway build before v2 can become authoritative.
        skip_pcr0_verification=True,
    )


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def attested_scoring_mode() -> str:
    value = str(os.getenv(MODE_ENV, "off") or "off").strip().lower()
    return value if value in MODES else "off"


def attested_receipt_persistence_enabled() -> bool:
    return str(os.getenv(PERSIST_RECEIPTS_ENV, "false") or "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def attested_live_provider_enabled() -> bool:
    return str(os.getenv(LIVE_PROVIDER_ENV, "false") or "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


async def resolve_attested_artifact_lineage(
    *,
    artifact_kind: str,
    artifact_ref: str,
    artifact_hash: str | None = None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Load one persisted artifact receipt and its complete ancestry safely."""

    if attested_scoring_mode() == "off" or not attested_receipt_persistence_enabled():
        return None, []
    try:
        from gateway.research_lab.attested_receipt_store import (
            load_attested_receipt_lineage,
            load_receipt_for_artifact,
        )

        receipt = await load_receipt_for_artifact(
            artifact_kind=artifact_kind,
            artifact_ref=artifact_ref,
            artifact_hash=artifact_hash,
        )
        if receipt is None:
            raise AttestedScoringError(
                "attested artifact receipt is unavailable: %s/%s"
                % (artifact_kind, artifact_ref)
            )
        lineage = await load_attested_receipt_lineage(receipt)
        return dict(receipt), [*lineage, dict(receipt)]
    except Exception as exc:
        if attested_scoring_mode() == "required":
            if isinstance(exc, AttestedScoringError):
                raise
            raise AttestedScoringError("required artifact receipt lineage failed") from exc
        _log_attestation_error(
            "research_lab_attested_scoring_shadow_failed operation=artifact_lineage "
            "artifact_kind=%s artifact_ref=%s error_type=%s error=%s",
            artifact_kind,
            artifact_ref,
            type(exc).__name__,
            str(exc)[:240],
        )
        return None, []


async def persist_attested_outcome_artifact_links(
    outcome: Mapping[str, Any],
    *,
    artifact_links: list[Mapping[str, Any]],
) -> str:
    """Idempotently add links after a host artifact receives its final ID/hash."""

    if attested_scoring_mode() == "off":
        return "off"
    if outcome.get("status") not in {"succeeded", "matched"}:
        if attested_scoring_mode() == "required":
            raise AttestedScoringError("required attested outcome is unavailable for linking")
        return "skipped"
    if outcome.get("persistence_status") != "persisted":
        if attested_scoring_mode() == "required":
            raise AttestedScoringError("required attested outcome was not persisted")
        return "disabled"
    receipt = outcome.get("receipt")
    pcr0 = str(outcome.get("pcr0") or "")
    if not isinstance(receipt, Mapping):
        if attested_scoring_mode() == "required":
            raise AttestedScoringError("required attested outcome receipt is missing")
        return "invalid"
    try:
        from gateway.research_lab.attested_receipt_store import persist_attested_receipt

        await persist_attested_receipt(
            receipt=receipt,
            pcr0=pcr0,
            artifact_links=artifact_links,
        )
        return "persisted"
    except Exception as exc:
        if attested_scoring_mode() == "required":
            raise AttestedScoringError("required receipt artifact linking failed") from exc
        _log_attestation_error(
            "research_lab_attested_receipt_shadow_link_failed error_type=%s error=%s",
            type(exc).__name__,
            str(exc)[:240],
        )
        return "shadow_failed"


def _float_env(name: str, default: float, minimum: float) -> float:
    try:
        return max(minimum, float(os.getenv(name, str(default))))
    except (TypeError, ValueError) as exc:
        logger.warning(
            "research_lab_attested_scoring_invalid_float_env "
            "name=%s fallback=%s error=%s",
            name,
            default,
            str(exc)[:120],
        )
        return default


def _commit_sha() -> str:
    candidates = [
        os.getenv("GITHUB_SHA"),
        os.getenv("GIT_COMMIT_HASH"),
        os.getenv("GIT_COMMIT"),
        get_build_info().get("git_commit"),
    ]
    for candidate in candidates:
        text = str(candidate or "").strip().lower()
        if _COMMIT_RE.fullmatch(text):
            return text
    try:
        root = Path(__file__).resolve().parents[2]
        value = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout.strip().lower()
    except Exception:
        value = ""
    if _COMMIT_RE.fullmatch(value):
        return value
    raise AttestedScoringError("gateway full commit SHA is unavailable")


def _normalize_evidence_roots(value: Mapping[str, Any] | None) -> dict[str, str]:
    roots = {}
    for name, digest in sorted(dict(value or {}).items()):
        text = str(digest or "").strip().lower()
        if not _HASH_RE.fullmatch(text):
            raise AttestedScoringError("invalid evidence root: %s" % name)
        roots[str(name)] = text
    return roots


def _score_bundle_receipt_output_roots(score_bundle: Mapping[str, Any]) -> set[str]:
    """Return exact signed and pre-KMS score-bundle output commitments."""

    bundle = dict(score_bundle)
    variants = [bundle]
    if "signature_ref" in bundle and bundle.get("signature_ref") != "pending":
        variants.append({**bundle, "signature_ref": "pending"})
    return {
        sha256_bytes(canonical_json_bytes({"score_bundle": variant}))
        for variant in variants
    }


def _validate_score_receipt_for_bundle(
    receipt: Mapping[str, Any],
    score_bundle: Mapping[str, Any],
) -> None:
    validate_signed_receipt(receipt)
    if receipt.get("purpose") not in SCORE_RECEIPT_PURPOSES:
        raise AttestedScoringError("score-bundle receipt purpose is invalid")
    if receipt.get("status") != "succeeded":
        raise AttestedScoringError("score-bundle receipt is not successful")
    if receipt.get("output_root") not in _score_bundle_receipt_output_roots(score_bundle):
        raise AttestedScoringError("score-bundle receipt output does not match artifact")
    score_bundle_hash = str(score_bundle.get("score_bundle_hash") or "").lower()
    if receipt.get("evidence_roots", {}).get("score_bundle") != score_bundle_hash:
        raise AttestedScoringError("score-bundle receipt evidence does not match artifact")
    gate = score_bundle.get("private_holdout_gate")
    if isinstance(gate, Mapping):
        baseline_hash = str(gate.get("baseline_benchmark_hash") or "").lower()
        if _HASH_RE.fullmatch(baseline_hash) and (
            receipt.get("evidence_roots", {}).get("baseline_score_summary")
            != baseline_hash
        ):
            raise AttestedScoringError(
                "score-bundle receipt does not bind its baseline summary"
            )


def _validate_promotion_receipt_for_bundle(
    receipt: Mapping[str, Any],
    score_bundle: Mapping[str, Any],
) -> None:
    validate_signed_receipt(receipt)
    if receipt.get("purpose") != "research_lab.promotion_decision.v1":
        raise AttestedScoringError("promotion receipt purpose is invalid")
    if receipt.get("status") != "succeeded":
        raise AttestedScoringError("promotion receipt is not successful")
    score_bundle_hash = str(score_bundle.get("score_bundle_hash") or "").lower()
    if _HASH_RE.fullmatch(score_bundle_hash) and (
        receipt.get("evidence_roots", {}).get("score_bundle") != score_bundle_hash
    ):
        raise AttestedScoringError("promotion receipt does not bind its score bundle")
    expected_status_hash = sha256_bytes(
        canonical_json_bytes({"status": "promotion_passed"})
    )
    if (
        receipt.get("evidence_roots", {}).get("promotion_decision_status")
        != expected_status_hash
    ):
        raise AttestedScoringError("promotion receipt is not a passed gate decision")


def _validate_score_receipt_baseline_ancestry(
    receipt: Mapping[str, Any],
    ancestors: list[Mapping[str, Any]],
) -> None:
    baseline_hash = str(
        receipt.get("evidence_roots", {}).get("baseline_score_summary") or ""
    ).lower()
    if not baseline_hash:
        return
    if not _HASH_RE.fullmatch(baseline_hash):
        raise AttestedScoringError("score receipt baseline evidence is invalid")
    matching = [
        ancestor
        for ancestor in ancestors
        if ancestor.get("purpose")
        in {
            "research_lab.baseline_score.v1",
            "research_lab.benchmark.v1",
            "research_lab.rebenchmark.v1",
        }
        and ancestor.get("status") == "succeeded"
        and ancestor.get("evidence_roots", {}).get("baseline_score_summary")
        == baseline_hash
    ]
    if not matching:
        raise AttestedScoringError(
            "score receipt baseline evidence has no attested baseline ancestor"
        )


async def execute_attested_scoring_operation(
    *,
    operation: str,
    purpose: str,
    epoch_id: int,
    payload: Mapping[str, Any],
    evidence_roots: Mapping[str, Any] | None = None,
    parent_receipt_hashes: list[str] | None = None,
    parent_receipts: list[Mapping[str, Any]] | None = None,
    artifact_links: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    desired_mode = attested_scoring_mode()
    if desired_mode == "off":
        return {"status": "off"}
    if desired_mode == "required" and not attested_receipt_persistence_enabled():
        raise AttestedScoringError("required attested scoring needs receipt persistence")

    try:
        forwarder_status = None
        runtime_configuration = None
        if operation == "qualification_company_scores":
            from gateway.tee.egress_policy import destination_policy_hash
            from gateway.tee.scoring_executor import (
                configuration_hash as scoring_configuration_hash,
                runtime_environment_values,
            )
            from gateway.utils.tee_egress_forwarder import ensure_tee_egress_forwarder

            forwarder_status = await asyncio.to_thread(ensure_tee_egress_forwarder)
            if forwarder_status.get("status") not in {"running", "owned_by_peer_process"}:
                raise AttestedScoringError("gateway enclave egress forwarder is unavailable")
            if forwarder_status.get("policy_hash") != destination_policy_hash():
                raise AttestedScoringError("gateway enclave egress forwarder policy mismatch")
            environment = runtime_environment_values()
            expected_configuration_hash = scoring_configuration_hash(environment)
            runtime_configuration = await tee_client.scoring_configure_runtime(
                environment=environment,
                configuration_hash=expected_configuration_hash,
            )
            if runtime_configuration.get("configuration_hash") != expected_configuration_hash:
                raise AttestedScoringError("gateway enclave scoring runtime configuration mismatch")
        health = await tee_client.scoring_health()
        enclave_mode = str(health.get("mode") or "off")
        if enclave_mode == "off":
            raise AttestedScoringError("gateway enclave attested scoring is disabled")
        if desired_mode == "required" and enclave_mode != "required":
            raise AttestedScoringError("gateway enclave is not in required scoring mode")
        if operation == "qualification_company_scores":
            egress_health = health.get("egress_proxy")
            if not isinstance(egress_health, Mapping) or egress_health.get("status") != "running":
                raise AttestedScoringError("gateway enclave egress proxy is unavailable")
            if egress_health.get("policy_hash") != forwarder_status.get("policy_hash"):
                raise AttestedScoringError("gateway enclave egress proxy policy mismatch")
        config_hash = str(health.get("config_hash") or "")
        if not _HASH_RE.fullmatch(config_hash):
            raise AttestedScoringError("gateway enclave returned an invalid scoring config hash")
        if (
            runtime_configuration is not None
            and config_hash != runtime_configuration.get("configuration_hash")
        ):
            raise AttestedScoringError("gateway enclave scoring health configuration mismatch")

        payload_bytes = canonical_json_bytes(dict(payload))
        payload_hash = sha256_bytes(payload_bytes)
        commit_sha = _commit_sha()
        if str(health.get("commit_sha") or "").lower() != commit_sha:
            raise AttestedScoringError("gateway host commit differs from enclave build identity")
        normalized_evidence = _normalize_evidence_roots(evidence_roots)
        normalized_parents = sorted(set(parent_receipt_hashes or []))
        normalized_parent_receipts: dict[str, dict[str, Any]] = {}
        for parent in parent_receipts or []:
            if not isinstance(parent, Mapping):
                raise AttestedScoringError("parent receipt is not an object")
            validate_signed_receipt(parent)
            parent_hash = str(parent.get("receipt_hash") or "")
            if parent_hash in normalized_parent_receipts:
                raise AttestedScoringError("parent receipt is duplicated")
            normalized_parent_receipts[parent_hash] = dict(parent)
        if normalized_parents:
            for parent_hash in normalized_parents:
                parent = normalized_parent_receipts.get(parent_hash)
                if parent is None:
                    raise AttestedScoringError("direct parent receipt is missing")
                verify_receipt_lineage(parent, normalized_parent_receipts)
        elif normalized_parent_receipts:
            raise AttestedScoringError("parent receipts were supplied without direct parent hashes")
        identity_hash = sha256_bytes(
            canonical_json_bytes(
                {
                    "operation": str(operation),
                    "purpose": str(purpose),
                    "epoch_id": int(epoch_id),
                    "commit_sha": commit_sha,
                    "config_hash": config_hash,
                    "payload_sha256": payload_hash,
                    "evidence_roots": normalized_evidence,
                    "parent_receipt_hashes": normalized_parents,
                }
            )
        )
        job_id = "attested-scoring:%s:%s" % (
            str(operation).replace("_", "-")[:80],
            identity_hash.split(":", 1)[1][:32],
        )
        manifest = {
            "schema_version": JOB_SCHEMA_VERSION,
            "job_id": job_id,
            "operation": str(operation),
            "purpose": str(purpose),
            "epoch_id": int(epoch_id),
            "commit_sha": commit_sha,
            "config_hash": config_hash,
            "payload_sha256": payload_hash,
            "payload_size_bytes": len(payload_bytes),
            "evidence_roots": normalized_evidence,
            "parent_receipt_hashes": normalized_parents,
        }
        submitted = await tee_client.scoring_submit_job(manifest)
        state = str(submitted.get("state") or "")
        if state == "uploading":
            offset = int(submitted.get("uploaded_bytes") or 0)
            while offset < len(payload_bytes):
                chunk = payload_bytes[offset: offset + 512 * 1024]
                await tee_client.scoring_put_chunk(job_id=job_id, offset=offset, data=chunk)
                offset += len(chunk)
            await tee_client.scoring_seal_job(job_id)

        timeout_seconds = _float_env(TIMEOUT_ENV, 1800.0, 1.0)
        poll_seconds = _float_env(POLL_SECONDS_ENV, 0.25, 0.05)
        deadline = asyncio.get_running_loop().time() + timeout_seconds
        while True:
            status = await tee_client.scoring_get_status(job_id)
            state = str(status.get("state") or "")
            if state in {"succeeded", "failed", "cancelled"}:
                break
            if asyncio.get_running_loop().time() >= deadline:
                await tee_client.scoring_cancel_job(job_id)
                raise AttestedScoringError("gateway enclave scoring job timed out")
            await asyncio.sleep(poll_seconds)
        if state != "succeeded":
            raise AttestedScoringError(
                "gateway enclave scoring job ended in %s (%s)" % (
                    state,
                    str(status.get("error_code") or "no_error_code"),
                )
            )

        result_bytes = bytearray()
        offset = 0
        expected_result_hash = str(status.get("result_sha256") or "")
        while True:
            part = await tee_client.scoring_get_result(job_id, offset=offset)
            chunk = base64.b64decode(str(part.get("data_b64") or ""), validate=True)
            if sha256_bytes(chunk) != part.get("chunk_sha256"):
                raise AttestedScoringError("gateway enclave result chunk hash mismatch")
            result_bytes.extend(chunk)
            offset += len(chunk)
            if part.get("eof"):
                break
        if sha256_bytes(bytes(result_bytes)) != expected_result_hash:
            raise AttestedScoringError("gateway enclave result hash mismatch")
        result = json.loads(bytes(result_bytes).decode("utf-8"))
        if canonical_json_bytes(result) != bytes(result_bytes):
            raise AttestedScoringError("gateway enclave result is not canonical JSON")

        receipt = await tee_client.scoring_get_receipt(job_id)
        validate_signed_receipt(receipt)
        if receipt.get("input_root") != payload_hash:
            raise AttestedScoringError("gateway enclave receipt input root mismatch")
        if receipt.get("output_root") != expected_result_hash:
            raise AttestedScoringError("gateway enclave receipt output root mismatch")
        if receipt.get("purpose") != purpose or int(receipt.get("epoch_id", -1)) != int(epoch_id):
            raise AttestedScoringError("gateway enclave receipt purpose or epoch mismatch")
        receipt_evidence = receipt.get("evidence_roots")
        if not isinstance(receipt_evidence, Mapping):
            raise AttestedScoringError("gateway enclave receipt evidence roots are invalid")
        for name, digest in normalized_evidence.items():
            if receipt_evidence.get(name) != digest:
                raise AttestedScoringError("gateway enclave receipt evidence root mismatch")
        attestation_valid, attestation_data = await asyncio.to_thread(
            verify_gateway_receipt_attestation,
            receipt=receipt,
            expected_purpose=purpose,
            expected_epoch_id=int(epoch_id),
        )
        if not attestation_valid:
            raise AttestedScoringError(
                "gateway enclave receipt attestation failed: %s"
                % str(attestation_data.get("error") or "unknown")[:240]
            )
        pcr0 = str(attestation_data.get("pcr0") or "").lower()
        if not re.fullmatch(r"[0-9a-f]{96}", pcr0) or pcr0 == "0" * 96:
            raise AttestedScoringError("gateway enclave receipt has an invalid PCR0")
        if attestation_data.get("purpose") != purpose:
            raise AttestedScoringError("gateway enclave attestation purpose mismatch")
        if int(attestation_data.get("epoch_id", -1)) != int(epoch_id):
            raise AttestedScoringError("gateway enclave attestation epoch mismatch")
        if attestation_data.get("enclave_pubkey") != receipt.get("enclave_pubkey"):
            raise AttestedScoringError("gateway enclave attestation public key mismatch")
        persistence_status = "disabled"
        if attested_receipt_persistence_enabled():
            try:
                from gateway.research_lab.attested_receipt_store import persist_attested_receipt

                await persist_attested_receipt(
                    receipt=receipt,
                    pcr0=pcr0,
                    artifact_links=list(artifact_links or []),
                )
                persistence_status = "persisted"
            except Exception as exc:
                if desired_mode == "required":
                    raise AttestedScoringError("required receipt persistence failed") from exc
                _log_attestation_error(
                    "research_lab_attested_receipt_shadow_persist_failed job_id=%s "
                    "error_type=%s error=%s",
                    job_id,
                    type(exc).__name__,
                    str(exc)[:240],
                )
                persistence_status = "shadow_failed"
        return {
            "status": "succeeded",
            "job_id": job_id,
            "result": result,
            "receipt": receipt,
            "pcr0": pcr0,
            "persistence_status": persistence_status,
            "parent_receipts": list(normalized_parent_receipts.values()),
        }
    except Exception as exc:
        if desired_mode == "required":
            if isinstance(exc, AttestedScoringError):
                raise
            raise AttestedScoringError("required gateway enclave scoring failed") from exc
        _log_attestation_error(
            "research_lab_attested_scoring_shadow_failed operation=%s error_type=%s error=%s",
            operation,
            type(exc).__name__,
            str(exc)[:240],
        )
        return {"status": "shadow_failed", "error_type": type(exc).__name__}


async def compare_score_bundle(
    *,
    epoch_id: int,
    purpose: str,
    build_payload: Mapping[str, Any],
    expected_score_bundle: Mapping[str, Any],
    evidence_roots: Mapping[str, Any] | None = None,
    parent_receipts: list[Mapping[str, Any]] | None = None,
    direct_parent_receipt_hashes: list[str] | None = None,
) -> dict[str, Any]:
    score_bundle_hash = str(expected_score_bundle.get("score_bundle_hash") or "").lower()
    if not _HASH_RE.fullmatch(score_bundle_hash):
        if attested_scoring_mode() == "required":
            raise AttestedScoringError("score bundle hash is invalid")
        _log_attestation_error(
            "research_lab_attested_scoring_shadow_failed operation=build_score_bundle "
            "error_type=AttestedScoringError error=invalid_score_bundle_hash"
        )
        return {"status": "shadow_failed", "error_type": "AttestedScoringError"}
    score_bundle_id = "score_bundle:" + score_bundle_hash.split(":", 1)[1]
    normalized_parent_receipts = [
        dict(item)
        for item in (parent_receipts or [])
        if isinstance(item, Mapping)
    ]
    outcome = await execute_attested_scoring_operation(
        operation="build_score_bundle",
        purpose=purpose,
        epoch_id=epoch_id,
        payload=build_payload,
        evidence_roots=evidence_roots,
        parent_receipt_hashes=(
            sorted(set(direct_parent_receipt_hashes))
            if direct_parent_receipt_hashes is not None
            else sorted(
                {str(item.get("receipt_hash") or "") for item in normalized_parent_receipts}
            )
        ),
        parent_receipts=normalized_parent_receipts,
    )
    if outcome.get("status") != "succeeded":
        return outcome
    actual = (outcome.get("result") or {}).get("score_bundle")
    if canonical_json_bytes(actual) != canonical_json_bytes(dict(expected_score_bundle)):
        message = "gateway enclave score bundle differs from host output"
        if attested_scoring_mode() == "required":
            raise AttestedScoringError(message)
        _log_attestation_error(
            "research_lab_attested_scoring_shadow_mismatch operation=build_score_bundle "
            "host_hash=%s enclave_hash=%s",
            sha256_bytes(canonical_json_bytes(dict(expected_score_bundle))),
            sha256_bytes(canonical_json_bytes(actual)),
        )
        return {**outcome, "status": "shadow_mismatch"}
    if outcome.get("receipt", {}).get("evidence_roots", {}).get("score_bundle") != score_bundle_hash:
        message = "gateway enclave did not derive the score-bundle evidence root"
        if attested_scoring_mode() == "required":
            raise AttestedScoringError(message)
        _log_attestation_error(
            "research_lab_attested_scoring_shadow_failed operation=build_score_bundle "
            "error=%s",
            message,
        )
        return {**outcome, "status": "shadow_failed", "error_type": "AttestedScoringError"}
    artifact_link_status = await persist_attested_outcome_artifact_links(
        outcome,
        artifact_links=[
            {
                "artifact_kind": "score_bundle",
                "artifact_ref": score_bundle_id,
                "artifact_hash": score_bundle_hash,
            }
        ],
    )
    return {
        **outcome,
        "status": "matched",
        "artifact_link_status": artifact_link_status,
    }


async def compare_baseline_score_summary(
    *,
    epoch_id: int,
    build_payload: Mapping[str, Any],
    expected_result: Mapping[str, Any],
    parent_receipts: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compare the complete daily benchmark summary with enclave execution."""

    score_summary_doc = expected_result.get("score_summary_doc")
    if not isinstance(score_summary_doc, Mapping):
        if attested_scoring_mode() == "required":
            raise AttestedScoringError("baseline score summary is missing")
        return {"status": "shadow_failed", "error_type": "AttestedScoringError"}
    normalized_parent_receipts = [
        dict(item)
        for item in (parent_receipts or [])
        if isinstance(item, Mapping)
    ]
    summary_hash = sha256_bytes(canonical_json_bytes(dict(score_summary_doc)))
    benchmark_date = str(build_payload.get("benchmark_date") or "")
    benchmark_attempt = int(build_payload.get("benchmark_attempt") or 0)
    rolling_window_hash = str(build_payload.get("rolling_window_hash") or "")
    artifact_ref = "private_baseline:%s:%s:%s" % (
        benchmark_date,
        benchmark_attempt,
        rolling_window_hash.removeprefix("sha256:")[:24],
    )
    outcome = await execute_attested_scoring_operation(
        operation="build_baseline_score_summary",
        purpose="research_lab.rebenchmark.v1",
        epoch_id=int(epoch_id),
        payload=dict(build_payload),
        parent_receipt_hashes=sorted(
            {str(item.get("receipt_hash") or "") for item in normalized_parent_receipts}
        ),
        parent_receipts=normalized_parent_receipts,
    )
    if outcome.get("status") != "succeeded":
        return outcome
    actual = outcome.get("result")
    if canonical_json_bytes(actual) != canonical_json_bytes(dict(expected_result)):
        message = "gateway enclave baseline summary differs from host output"
        if attested_scoring_mode() == "required":
            raise AttestedScoringError(message)
        _log_attestation_error(
            "research_lab_attested_scoring_shadow_mismatch "
            "operation=build_baseline_score_summary host_hash=%s enclave_hash=%s",
            sha256_bytes(canonical_json_bytes(dict(expected_result))),
            sha256_bytes(canonical_json_bytes(actual)),
        )
        return {**outcome, "status": "shadow_mismatch"}
    if (
        outcome.get("receipt", {}).get("evidence_roots", {}).get("baseline_score_summary")
        != summary_hash
    ):
        message = "gateway enclave did not derive the baseline-summary evidence root"
        if attested_scoring_mode() == "required":
            raise AttestedScoringError(message)
        _log_attestation_error(
            "research_lab_attested_scoring_shadow_failed "
            "operation=build_baseline_score_summary error=%s",
            message,
        )
        return {**outcome, "status": "shadow_failed", "error_type": "AttestedScoringError"}
    artifact_link_status = await persist_attested_outcome_artifact_links(
        outcome,
        artifact_links=[
            {
                "artifact_kind": "benchmark_score_summary",
                "artifact_ref": artifact_ref,
                "artifact_hash": summary_hash,
            }
        ],
    )
    return {
        **outcome,
        "status": "matched",
        "score_summary_hash": summary_hash,
        "artifact_ref": artifact_ref,
        "artifact_link_status": artifact_link_status,
    }


async def compare_qualification_company_scores(
    *,
    epoch_id: int,
    purpose: str,
    companies: list[Mapping[str, Any]],
    icp: Mapping[str, Any],
    is_reference_model: bool,
    provider_tape: Mapping[str, Any],
    expected_breakdowns: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Replay one host scorer call inside the enclave without provider spend."""

    from research_lab.eval.http_tape import validate_http_tape

    normalized_tape = validate_http_tape(provider_tape)
    expected = {
        "breakdowns": [dict(item) for item in expected_breakdowns],
        "scores": [
            float(item.get("final_score", 0.0) or 0.0)
            for item in expected_breakdowns
        ],
    }
    outcome = await execute_attested_scoring_operation(
        operation="qualification_company_scores",
        purpose=purpose,
        epoch_id=epoch_id,
        payload={
            "companies": [dict(item) for item in companies],
            "icp": dict(icp),
            "is_reference_model": bool(is_reference_model),
            "provider_tape": normalized_tape,
        },
        evidence_roots={"provider_http_tape": normalized_tape["tape_hash"]},
    )
    if outcome.get("status") != "succeeded":
        return outcome
    actual = outcome.get("result")
    if canonical_json_bytes(actual) != canonical_json_bytes(expected):
        message = "gateway enclave qualification scores differ from host output"
        if attested_scoring_mode() == "required":
            raise AttestedScoringError(message)
        _log_attestation_error(
            "research_lab_attested_scoring_shadow_mismatch "
            "operation=qualification_company_scores host_hash=%s enclave_hash=%s",
            sha256_bytes(canonical_json_bytes(expected)),
            sha256_bytes(canonical_json_bytes(actual)),
        )
        return {**outcome, "status": "shadow_mismatch"}
    return {**outcome, "status": "matched"}


async def execute_required_qualification_company_scores(
    *,
    epoch_id: int,
    purpose: str,
    companies: list[Mapping[str, Any]],
    icp: Mapping[str, Any],
    is_reference_model: bool,
    attestation_out: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Run providers and company scoring only inside the measured enclave."""

    if attested_scoring_mode() != "required":
        raise AttestedScoringError("live enclave scoring requires required mode")
    if not attested_live_provider_enabled():
        raise AttestedScoringError(
            "required live enclave provider execution is not release-enabled"
        )
    outcome = await execute_attested_scoring_operation(
        operation="qualification_company_scores",
        purpose=purpose,
        epoch_id=epoch_id,
        payload={
            "companies": [dict(item) for item in companies],
            "icp": dict(icp),
            "is_reference_model": bool(is_reference_model),
            "provider_execution_mode": "live_enclave",
        },
    )
    if outcome.get("status") != "succeeded":
        raise AttestedScoringError("required live enclave scoring did not succeed")
    receipt = outcome.get("receipt")
    result = outcome.get("result")
    if not isinstance(receipt, Mapping) or not isinstance(result, Mapping):
        raise AttestedScoringError("required live enclave scoring result is incomplete")
    provider_root = (receipt.get("evidence_roots") or {}).get("provider_http_tape")
    if not _HASH_RE.fullmatch(str(provider_root or "")):
        raise AttestedScoringError("required scoring receipt lacks provider evidence")
    breakdowns = result.get("breakdowns")
    scores = result.get("scores")
    if not isinstance(breakdowns, list) or not isinstance(scores, list):
        raise AttestedScoringError("required scoring output shape is invalid")
    normalized = []
    for item in breakdowns:
        if not isinstance(item, Mapping):
            raise AttestedScoringError("required scoring breakdown is invalid")
        normalized.append(dict(item))
    expected_scores = [
        float(item.get("final_score", 0.0) or 0.0)
        for item in normalized
    ]
    if canonical_json_bytes(scores) != canonical_json_bytes(expected_scores):
        raise AttestedScoringError("required scoring score projection diverged")
    if attestation_out is not None:
        attestation_out.clear()
        attestation_out.update(outcome)
    return normalized


async def _resolve_allocation_receipt_lineage(
    payload: Mapping[str, Any],
) -> tuple[
    list[str],
    dict[str, dict[str, Any]],
    list[dict[str, str]],
    list[str],
]:
    parent_hashes: list[str] = []
    parent_receipts: dict[str, dict[str, Any]] = {}
    lineage_bindings: list[dict[str, str]] = []
    missing_lineage: list[str] = []
    obligations = payload.get("active_champion_obligations")
    if not isinstance(obligations, list):
        raise AttestedScoringError("allocation champion obligations must be a list")
    if attested_receipt_persistence_enabled():
        from gateway.research_lab.attested_receipt_store import (
            load_attested_receipt_lineage,
            load_receipt_for_artifact,
        )
        from gateway.research_lab.store import select_one

        for obligation in obligations:
            if not isinstance(obligation, Mapping):
                continue
            score_bundle_id = str(obligation.get("score_bundle_id") or "").strip()
            if not score_bundle_id.startswith("score_bundle:"):
                continue
            score_bundle_hash = "sha256:" + score_bundle_id.split(":", 1)[1]
            if not _HASH_RE.fullmatch(score_bundle_hash):
                raise AttestedScoringError("allocation score bundle hash is invalid")
            score_bundle_row = await select_one(
                "research_evaluation_score_bundles",
                filters=(("score_bundle_id", score_bundle_id),),
            )
            score_bundle = (
                score_bundle_row.get("score_bundle_doc")
                if isinstance(score_bundle_row, Mapping)
                and isinstance(score_bundle_row.get("score_bundle_doc"), Mapping)
                else None
            )
            if not isinstance(score_bundle, Mapping):
                raise AttestedScoringError("allocation score bundle artifact is unavailable")
            if str(score_bundle.get("score_bundle_hash") or "") != score_bundle_hash:
                raise AttestedScoringError("allocation score bundle artifact hash diverged")
            receipt = await load_receipt_for_artifact(
                artifact_kind="promotion_decision",
                artifact_ref=score_bundle_id,
                artifact_hash=score_bundle_hash,
            )
            if receipt is None:
                missing_lineage.append(score_bundle_id)
                receipt = await load_receipt_for_artifact(
                    artifact_kind="score_bundle",
                    artifact_ref=score_bundle_id,
                    artifact_hash=score_bundle_hash,
                )
            if receipt is None:
                continue
            lineage = await load_attested_receipt_lineage(receipt)
            if receipt.get("purpose") == "research_lab.promotion_decision.v1":
                _validate_promotion_receipt_for_bundle(receipt, score_bundle)
                score_ancestors = [
                    ancestor
                    for ancestor in lineage
                    if ancestor.get("purpose") in SCORE_RECEIPT_PURPOSES
                    and ancestor.get("evidence_roots", {}).get("score_bundle")
                    == score_bundle_hash
                ]
                if not score_ancestors:
                    raise AttestedScoringError("promotion receipt lacks score-bundle ancestor")
                for score_ancestor in score_ancestors:
                    _validate_score_receipt_for_bundle(score_ancestor, score_bundle)
                    _validate_score_receipt_baseline_ancestry(
                        score_ancestor,
                        lineage,
                    )
            else:
                _validate_score_receipt_for_bundle(receipt, score_bundle)
                _validate_score_receipt_baseline_ancestry(receipt, lineage)
            receipt_hash = str(receipt["receipt_hash"])
            if receipt_hash not in parent_hashes:
                parent_hashes.append(receipt_hash)
            parent_receipts[receipt_hash] = dict(receipt)
            for ancestor in lineage:
                parent_receipts[str(ancestor["receipt_hash"])] = dict(ancestor)
            lineage_bindings.append(
                {
                    "score_bundle_id": score_bundle_id,
                    "score_bundle_hash": score_bundle_hash,
                    "receipt_hash": receipt_hash,
                    "receipt_purpose": str(receipt.get("purpose") or ""),
                }
            )
    else:
        for obligation in obligations:
            if isinstance(obligation, Mapping):
                score_bundle_id = str(obligation.get("score_bundle_id") or "").strip()
                if score_bundle_id.startswith("score_bundle:"):
                    missing_lineage.append(score_bundle_id)
    return parent_hashes, parent_receipts, lineage_bindings, missing_lineage


async def compare_allocation(
    *,
    epoch_id: int,
    payload: Mapping[str, Any],
    expected_allocation: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        (
            parent_hashes,
            parent_receipts,
            lineage_bindings,
            missing_lineage,
        ) = await _resolve_allocation_receipt_lineage(payload)
    except Exception as exc:
        if attested_scoring_mode() == "required":
            if isinstance(exc, AttestedScoringError):
                raise
            raise AttestedScoringError("required allocation lineage resolution failed") from exc
        _log_attestation_error(
            "research_lab_attested_scoring_shadow_failed operation=allocation_lineage "
            "error_type=%s error=%s",
            type(exc).__name__,
            str(exc)[:240],
        )
        return {"status": "shadow_failed", "error_type": type(exc).__name__}
    if missing_lineage and attested_scoring_mode() == "required":
        raise AttestedScoringError("required allocation receipt lineage is incomplete")

    attested_payload = {
        **dict(payload),
        "receipt_lineage_bindings": sorted(
            lineage_bindings,
            key=lambda item: (item["score_bundle_id"], item["receipt_hash"]),
        ),
    }
    allocation_hash = str(expected_allocation.get("allocation_hash") or "").lower()
    if not _HASH_RE.fullmatch(allocation_hash):
        if attested_scoring_mode() == "required":
            raise AttestedScoringError("allocation hash is invalid")
        _log_attestation_error(
            "research_lab_attested_scoring_shadow_failed operation=allocation "
            "error_type=AttestedScoringError error=invalid_allocation_hash"
        )
        return {"status": "shadow_failed", "error_type": "AttestedScoringError"}
    outcome = await execute_attested_scoring_operation(
        operation="research_lab_allocation",
        purpose="research_lab.allocation.v1",
        epoch_id=epoch_id,
        payload=attested_payload,
        parent_receipt_hashes=parent_hashes,
        parent_receipts=list(parent_receipts.values()),
    )
    if outcome.get("status") != "succeeded":
        return outcome
    actual = (outcome.get("result") or {}).get("allocation")
    if canonical_json_bytes(actual) != canonical_json_bytes(dict(expected_allocation)):
        message = "gateway enclave allocation differs from host output"
        if attested_scoring_mode() == "required":
            raise AttestedScoringError(message)
        _log_attestation_error(
            "research_lab_attested_scoring_shadow_mismatch operation=allocation "
            "host_hash=%s enclave_hash=%s",
            sha256_bytes(canonical_json_bytes(dict(expected_allocation))),
            sha256_bytes(canonical_json_bytes(actual)),
        )
        return {**outcome, "status": "shadow_mismatch"}
    if outcome.get("receipt", {}).get("evidence_roots", {}).get("allocation") != allocation_hash:
        message = "gateway enclave did not derive the allocation evidence root"
        if attested_scoring_mode() == "required":
            raise AttestedScoringError(message)
        _log_attestation_error(
            "research_lab_attested_scoring_shadow_failed operation=allocation error=%s",
            message,
        )
        return {**outcome, "status": "shadow_failed", "error_type": "AttestedScoringError"}
    artifact_link_status = await persist_attested_outcome_artifact_links(
        outcome,
        artifact_links=[
            {
                "artifact_kind": "allocation",
                "artifact_ref": "epoch:%s" % int(epoch_id),
                "artifact_hash": allocation_hash,
            }
        ],
    )
    return {
        **outcome,
        "status": "matched",
        "lineage_bindings": attested_payload["receipt_lineage_bindings"],
        "lineage_complete": not missing_lineage,
        "missing_lineage_score_bundle_ids": sorted(set(missing_lineage)),
        "artifact_link_status": artifact_link_status,
    }


async def compare_promotion_metric(
    *,
    epoch_id: int,
    score_bundle: Mapping[str, Any],
    expected_improvement_points: float,
    expected_event_doc: Mapping[str, Any],
    parent_receipt_hashes: list[str] | None = None,
) -> dict[str, Any]:
    """Compare the enclave's pure promotion metric with the host result."""

    score_bundle_hash = str(score_bundle.get("score_bundle_hash") or "").lower()
    score_bundle_id = (
        "score_bundle:" + score_bundle_hash.split(":", 1)[1]
        if _HASH_RE.fullmatch(score_bundle_hash)
        else ""
    )
    parent_receipts: list[dict[str, Any]] = []
    normalized_parent_hashes = list(parent_receipt_hashes or [])
    try:
        if not normalized_parent_hashes and score_bundle_id and attested_receipt_persistence_enabled():
            from gateway.research_lab.attested_receipt_store import (
                load_attested_receipt_lineage,
                load_receipt_for_artifact,
            )

            score_receipt = await load_receipt_for_artifact(
                artifact_kind="score_bundle",
                artifact_ref=score_bundle_id,
                artifact_hash=score_bundle_hash,
            )
            if score_receipt is not None:
                _validate_score_receipt_for_bundle(score_receipt, score_bundle)
                normalized_parent_hashes = [str(score_receipt["receipt_hash"])]
                parent_receipts = [
                    *await load_attested_receipt_lineage(score_receipt),
                    dict(score_receipt),
                ]
    except Exception as exc:
        if attested_scoring_mode() == "required":
            raise AttestedScoringError("required promotion lineage resolution failed") from exc
        _log_attestation_error(
            "research_lab_attested_scoring_shadow_failed operation=promotion_lineage "
            "error_type=%s error=%s",
            type(exc).__name__,
            str(exc)[:240],
        )
        return {"status": "shadow_failed", "error_type": type(exc).__name__}
    if attested_scoring_mode() == "required" and score_bundle_id and not normalized_parent_hashes:
        raise AttestedScoringError("required promotion receipt lacks score-bundle ancestry")

    outcome = await execute_attested_scoring_operation(
        operation="promotion_improvement",
        purpose="research_lab.promotion_metric.v1",
        epoch_id=epoch_id,
        payload={"score_bundle": dict(score_bundle)},
        parent_receipt_hashes=normalized_parent_hashes,
        parent_receipts=parent_receipts,
    )
    if outcome.get("status") != "succeeded":
        return outcome
    result = outcome.get("result") if isinstance(outcome.get("result"), Mapping) else {}
    expected = {
        "improvement_points": float(expected_improvement_points),
        "event_doc": dict(expected_event_doc),
    }
    actual = {
        "improvement_points": result.get("improvement_points"),
        "event_doc": result.get("event_doc"),
    }
    if canonical_json_bytes(actual) != canonical_json_bytes(expected):
        message = "gateway enclave promotion metric differs from host output"
        if attested_scoring_mode() == "required":
            raise AttestedScoringError(message)
        _log_attestation_error(
            "research_lab_attested_scoring_shadow_mismatch operation=promotion_improvement "
            "host_hash=%s enclave_hash=%s",
            sha256_bytes(canonical_json_bytes(expected)),
            sha256_bytes(canonical_json_bytes(actual)),
        )
        return {**outcome, "status": "shadow_mismatch"}
    if score_bundle_id and (
        outcome.get("receipt", {}).get("evidence_roots", {}).get("score_bundle")
        != score_bundle_hash
    ):
        message = "gateway enclave did not derive promotion score-bundle evidence"
        if attested_scoring_mode() == "required":
            raise AttestedScoringError(message)
        _log_attestation_error(
            "research_lab_attested_scoring_shadow_failed "
            "operation=promotion_improvement error=%s",
            message,
        )
        return {**outcome, "status": "shadow_failed", "error_type": "AttestedScoringError"}
    artifact_link_status = await persist_attested_outcome_artifact_links(
        outcome,
        artifact_links=(
            [
                {
                    "artifact_kind": "promotion_metric",
                    "artifact_ref": score_bundle_id,
                    "artifact_hash": score_bundle_hash,
                }
            ]
            if score_bundle_id
            else []
        ),
    )
    return {
        **outcome,
        "status": "matched",
        "artifact_link_status": artifact_link_status,
    }


async def compare_promotion_gate_decision(
    *,
    epoch_id: int,
    score_bundle: Mapping[str, Any],
    decision_payload: Mapping[str, Any],
    expected_decision: Mapping[str, Any],
    metric_outcome: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare the complete pure pre-merge gate decision and preserve ancestry."""

    if attested_scoring_mode() == "off":
        return {"status": "off"}
    metric_receipt = metric_outcome.get("receipt")
    if not isinstance(metric_receipt, Mapping):
        message = "promotion decision requires an attested metric parent"
        if attested_scoring_mode() == "required":
            raise AttestedScoringError(message)
        _log_attestation_error(
            "research_lab_attested_scoring_shadow_failed "
            "operation=promotion_gate_decision error=%s",
            message,
        )
        return {"status": "shadow_failed", "error_type": "AttestedScoringError"}
    try:
        validate_signed_receipt(metric_receipt)
        if metric_receipt.get("purpose") != "research_lab.promotion_metric.v1":
            raise AttestedScoringError("promotion metric parent purpose is invalid")

        all_parent_receipts: dict[str, dict[str, Any]] = {}
        for receipt in list(metric_outcome.get("parent_receipts") or []) + [metric_receipt]:
            if not isinstance(receipt, Mapping):
                raise AttestedScoringError("promotion metric lineage receipt is invalid")
            all_parent_receipts[str(receipt.get("receipt_hash") or "")] = dict(receipt)
        verify_receipt_lineage(metric_receipt, all_parent_receipts)
    except Exception as exc:
        if attested_scoring_mode() == "required":
            if isinstance(exc, AttestedScoringError):
                raise
            raise AttestedScoringError("promotion metric ancestry is invalid") from exc
        _log_attestation_error(
            "research_lab_attested_scoring_shadow_failed "
            "operation=promotion_gate_decision error_type=%s error=%s",
            type(exc).__name__,
            str(exc)[:240],
        )
        return {"status": "shadow_failed", "error_type": type(exc).__name__}

    payload = {"score_bundle": dict(score_bundle), **dict(decision_payload)}
    outcome = await execute_attested_scoring_operation(
        operation="promotion_gate_decision",
        purpose="research_lab.promotion_decision.v1",
        epoch_id=int(epoch_id),
        payload=payload,
        parent_receipt_hashes=[str(metric_receipt["receipt_hash"])],
        parent_receipts=list(all_parent_receipts.values()),
    )
    if outcome.get("status") != "succeeded":
        return outcome
    result = outcome.get("result") if isinstance(outcome.get("result"), Mapping) else {}
    actual_decision = result.get("decision")
    if canonical_json_bytes(actual_decision) != canonical_json_bytes(dict(expected_decision)):
        message = "gateway enclave promotion gate decision differs from host output"
        if attested_scoring_mode() == "required":
            raise AttestedScoringError(message)
        _log_attestation_error(
            "research_lab_attested_scoring_shadow_mismatch "
            "operation=promotion_gate_decision host_hash=%s enclave_hash=%s",
            sha256_bytes(canonical_json_bytes(dict(expected_decision))),
            sha256_bytes(canonical_json_bytes(actual_decision)),
        )
        return {**outcome, "status": "shadow_mismatch"}

    score_bundle_hash = str(score_bundle.get("score_bundle_hash") or "").lower()
    receipt_evidence = outcome.get("receipt", {}).get("evidence_roots", {})
    if _HASH_RE.fullmatch(score_bundle_hash) and (
        receipt_evidence.get("score_bundle") != score_bundle_hash
    ):
        message = "promotion decision does not bind its score bundle"
        if attested_scoring_mode() == "required":
            raise AttestedScoringError(message)
        _log_attestation_error(
            "research_lab_attested_scoring_shadow_failed "
            "operation=promotion_gate_decision error=%s",
            message,
        )
        return {**outcome, "status": "shadow_failed", "error_type": "AttestedScoringError"}
    expected_status_hash = sha256_bytes(
        canonical_json_bytes({"status": str(expected_decision.get("status") or "")})
    )
    if receipt_evidence.get("promotion_decision_status") != expected_status_hash:
        message = "promotion decision status evidence is invalid"
        if attested_scoring_mode() == "required":
            raise AttestedScoringError(message)
        _log_attestation_error(
            "research_lab_attested_scoring_shadow_failed "
            "operation=promotion_gate_decision error=%s",
            message,
        )
        return {**outcome, "status": "shadow_failed", "error_type": "AttestedScoringError"}

    score_bundle_id = (
        "score_bundle:" + score_bundle_hash.split(":", 1)[1]
        if _HASH_RE.fullmatch(score_bundle_hash)
        else ""
    )
    artifact_link_status = await persist_attested_outcome_artifact_links(
        outcome,
        artifact_links=(
            [
                {
                    "artifact_kind": "promotion_decision",
                    "artifact_ref": score_bundle_id,
                    "artifact_hash": score_bundle_hash,
                }
            ]
            if score_bundle_id
            else []
        ),
    )
    return {
        **outcome,
        "status": "matched",
        "artifact_link_status": artifact_link_status,
    }
