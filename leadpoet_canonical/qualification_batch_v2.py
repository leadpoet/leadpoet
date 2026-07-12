"""Canonical V2 qualification batch outputs without changing validation logic."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, Mapping, Sequence, Tuple

from leadpoet_canonical.attested_v2 import sha256_json
from leadpoet_canonical.sourcing_history_v2 import (
    SourcingHistoryV2Error,
    build_sourcing_decision_v2,
    validate_sourcing_decision_v2,
)


QUALIFICATION_BATCH_SCHEMA_VERSION = "leadpoet.qualification_batch.v2"
_SALT_RE = re.compile(r"^[0-9a-f]{64}$")


class QualificationBatchV2Error(ValueError):
    """A measured qualification batch output is incomplete or non-canonical."""


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str, ensure_ascii=False))


def _lead_identity(lead: Mapping[str, Any]) -> Tuple[str, str]:
    if not isinstance(lead, Mapping):
        raise QualificationBatchV2Error("qualification lead is not an object")
    lead_id = str(lead.get("lead_id") or "").strip()
    miner_hotkey = str(lead.get("miner_hotkey") or "").strip()
    if not lead_id or len(lead_id) > 512:
        raise QualificationBatchV2Error("qualification lead_id is invalid")
    if (
        not miner_hotkey
        or len(miner_hotkey) > 128
        or any(character.isspace() for character in miner_hotkey)
    ):
        raise QualificationBatchV2Error("qualification miner hotkey is invalid")
    if not isinstance(lead.get("lead_blob"), Mapping):
        raise QualificationBatchV2Error("qualification lead_blob is invalid")
    return lead_id, miner_hotkey


def _decision_values(passed: Any, automated_checks_data: Mapping[str, Any]) -> Tuple[str, int]:
    if passed not in (True, False, None):
        raise QualificationBatchV2Error("qualification pass state is invalid")
    if not isinstance(automated_checks_data, Mapping):
        raise QualificationBatchV2Error("automated checks output is invalid")
    if passed is not True:
        return "deny", 0
    rep_score_data = automated_checks_data.get("rep_score", {})
    try:
        rep_score = int(
            rep_score_data.get("total_score", 0)
            if isinstance(rep_score_data, Mapping)
            else rep_score_data
        )
    except (TypeError, ValueError) as exc:
        raise QualificationBatchV2Error("validator rep score is invalid") from exc
    return "approve", rep_score


def build_qualification_batch_output_v2(
    *,
    epoch_id: int,
    container_id: int,
    sequence_start: int,
    leads: Sequence[Mapping[str, Any]],
    batch_results: Sequence[Tuple[Any, Mapping[str, Any]]],
    salt_hex: str,
) -> Dict[str, Any]:
    if not isinstance(epoch_id, int) or isinstance(epoch_id, bool) or epoch_id < 0:
        raise QualificationBatchV2Error("qualification epoch is invalid")
    if (
        not isinstance(container_id, int)
        or isinstance(container_id, bool)
        or container_id < 0
    ):
        raise QualificationBatchV2Error("qualification container_id is invalid")
    if (
        not isinstance(sequence_start, int)
        or isinstance(sequence_start, bool)
        or sequence_start < 0
    ):
        raise QualificationBatchV2Error("qualification sequence_start is invalid")
    normalized_salt = str(salt_hex or "").lower()
    if not _SALT_RE.fullmatch(normalized_salt):
        raise QualificationBatchV2Error("qualification salt is invalid")
    if not isinstance(leads, Sequence) or isinstance(leads, (str, bytes)):
        raise QualificationBatchV2Error("qualification leads are invalid")
    if not isinstance(batch_results, Sequence) or len(batch_results) != len(leads):
        raise QualificationBatchV2Error("qualification result count differs from leads")

    validation_results = []
    local_validation_data = []
    sourcing_decisions = []
    normalized_evidence = []
    for offset, (lead, raw_result) in enumerate(zip(leads, batch_results)):
        lead_id, miner_hotkey = _lead_identity(lead)
        if (
            not isinstance(raw_result, (tuple, list))
            or len(raw_result) != 2
            or not isinstance(raw_result[1], Mapping)
        ):
            raise QualificationBatchV2Error("qualification result row is invalid")
        passed, automated_checks_data = raw_result
        decision, rep_score = _decision_values(passed, automated_checks_data)
        checks = dict(automated_checks_data)
        if passed is None:
            rejection_reason = {
                "stage": "Batch Validation",
                "check_name": "truelist_batch_skipped",
                "message": "Lead skipped due to persistent TrueList errors",
            }
            result = {
                "is_legitimate": False,
                "reason": rejection_reason,
                "skipped": True,
            }
        else:
            rejection_reason = (
                {"message": "pass"}
                if passed is True
                else (checks.get("rejection_reason") or {})
            )
            result = {
                "is_legitimate": bool(passed),
                "enhanced_lead": checks if passed is True else {},
                "reason": rejection_reason if passed is not True else None,
            }
            if passed is True:
                result["enhanced_lead"]["rep_score"] = rep_score

        clean_result = dict(result)
        if isinstance(clean_result.get("enhanced_lead"), Mapping):
            clean_enhanced = dict(clean_result["enhanced_lead"])
            for field in (
                "company_linkedin_data",
                "company_linkedin_slug",
                "company_linkedin_from_cache",
            ):
                clean_enhanced.pop(field, None)
            clean_result["enhanced_lead"] = clean_enhanced
        evidence_blob = json.dumps(clean_result, default=str)
        decision_hash = hashlib.sha256(
            (decision + normalized_salt).encode("utf-8")
        ).hexdigest()
        rep_score_hash = hashlib.sha256(
            (str(rep_score) + normalized_salt).encode("utf-8")
        ).hexdigest()
        rejection_reason_hash = hashlib.sha256(
            (
                json.dumps(rejection_reason, default=str) + normalized_salt
            ).encode("utf-8")
        ).hexdigest()
        evidence_hash = hashlib.sha256(evidence_blob.encode("utf-8")).hexdigest()
        safe_result = _json_safe(result)
        safe_reason = _json_safe(rejection_reason)
        validation_results.append(
            {
                "lead_id": lead_id,
                "decision_hash": decision_hash,
                "rep_score_hash": rep_score_hash,
                "rejection_reason_hash": rejection_reason_hash,
                "evidence_hash": evidence_hash,
                "evidence_blob": safe_result,
                "decision": decision,
                "rep_score": rep_score,
                "rejection_reason": safe_reason,
                "salt": normalized_salt,
            }
        )
        adjustment = automated_checks_data.get("is_icp_multiplier", 0.0)
        local_validation_data.append(
            {
                "lead_id": lead_id,
                "miner_hotkey": miner_hotkey,
                "decision": decision,
                "rep_score": rep_score,
                "rejection_reason": safe_reason,
                "salt": normalized_salt,
                "is_icp_multiplier": adjustment,
            }
        )
        sourcing_decisions.append(
            build_sourcing_decision_v2(
                epoch_id=epoch_id,
                sequence=sequence_start + offset,
                lead_id_hash=sha256_json({"lead_id": lead_id}),
                miner_hotkey=miner_hotkey,
                decision=decision,
                rep_score=rep_score,
                is_icp_multiplier=adjustment,
            )
        )
        normalized_evidence.append(_json_safe(automated_checks_data))

    body = {
        "schema_version": QUALIFICATION_BATCH_SCHEMA_VERSION,
        "epoch_id": epoch_id,
        "container_id": container_id,
        "sequence_start": sequence_start,
        "sequence_end": sequence_start + len(leads),
        "lead_count": len(leads),
        "validation_results": validation_results,
        "local_validation_data": local_validation_data,
        "sourcing_decisions": sourcing_decisions,
        "automated_checks_root": sha256_json(normalized_evidence),
    }
    return {**body, "batch_hash": sha256_json(body)}


def validate_qualification_batch_output_v2(value: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise QualificationBatchV2Error("qualification batch output is invalid")
    required = {
        "schema_version",
        "epoch_id",
        "container_id",
        "sequence_start",
        "sequence_end",
        "lead_count",
        "validation_results",
        "local_validation_data",
        "sourcing_decisions",
        "automated_checks_root",
        "batch_hash",
    }
    if set(value) != required or value.get("schema_version") != QUALIFICATION_BATCH_SCHEMA_VERSION:
        raise QualificationBatchV2Error("qualification batch fields are invalid")
    decisions = value.get("sourcing_decisions")
    if not isinstance(decisions, list):
        raise QualificationBatchV2Error("qualification decisions are invalid")
    try:
        normalized = [validate_sourcing_decision_v2(item) for item in decisions]
    except SourcingHistoryV2Error as exc:
        raise QualificationBatchV2Error(
            "qualification sourcing decision is invalid"
        ) from exc
    count = value.get("lead_count")
    if (
        not isinstance(count, int)
        or isinstance(count, bool)
        or count < 0
        or len(normalized) != count
        or not isinstance(value.get("validation_results"), list)
        or len(value["validation_results"]) != count
        or not isinstance(value.get("local_validation_data"), list)
        or len(value["local_validation_data"]) != count
        or value.get("sequence_end") != value.get("sequence_start") + count
    ):
        raise QualificationBatchV2Error("qualification batch cardinality is invalid")
    body = {key: value[key] for key in required if key != "batch_hash"}
    if value.get("batch_hash") != sha256_json(body):
        raise QualificationBatchV2Error("qualification batch hash differs")
    return dict(value)
