"""Measured reconstruction of the unchanged qualification lead assignment."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Mapping, Sequence
from uuid import UUID

from gateway.tee.execution_job_manager_v2 import ExecutionContextV2
from gateway.tee.supabase_source_v2 import SupabaseSourceReaderV2
from leadpoet_canonical.attested_v2 import canonical_json


QUALIFICATION_ADMISSION_INPUT_SCHEMA_VERSION = (
    "leadpoet.qualification_admission_input.v2"
)
_SALT_RE = re.compile(r"^[0-9a-f]{64}$")
_BLOB_FIELD_MAP = {
    "first": "first_name",
    "last": "last_name",
    "email": "email",
    "role": "role",
    "business": "company_name",
    "linkedin": "linkedin",
    "website": "website",
    "company_linkedin": "company_linkedin",
    "industry": "industry",
    "sub_industry": "sub_industry",
    "city": "city",
    "state": "state",
    "country": "country",
    "hq_city": "hq_city",
    "hq_state": "hq_state",
    "hq_country": "hq_country",
    "employee_count": "employee_count",
    "description": "description",
}


class QualificationAdmissionV2Error(RuntimeError):
    """The proposed qualification slice differs from authenticated epoch data."""


def _rebuild_blob(row: Mapping[str, Any]) -> Dict[str, Any]:
    blob = row.get("lead_blob") or {}
    if isinstance(blob, str):
        try:
            blob = json.loads(blob)
        except json.JSONDecodeError as exc:
            raise QualificationAdmissionV2Error("lead blob is invalid JSON") from exc
    if not isinstance(blob, Mapping):
        raise QualificationAdmissionV2Error("lead blob is not an object")
    rebuilt = dict(blob)
    for blob_key, column_name in _BLOB_FIELD_MAP.items():
        value = row.get(column_name)
        if value is not None:
            rebuilt[blob_key] = value
    return rebuilt


def _lead_id(value: Any) -> str:
    try:
        return str(UUID(str(value)))
    except (TypeError, ValueError, AttributeError) as exc:
        raise QualificationAdmissionV2Error("qualification lead_id is invalid") from exc


def _slice_bounds(total: int, container_id: int, container_count: int) -> tuple[int, int]:
    per_container = total // container_count
    remainder = total % container_count
    if container_id < remainder:
        start = container_id * (per_container + 1)
        return start, start + per_container + 1
    start = remainder * (per_container + 1) + (container_id - remainder) * per_container
    return start, start + per_container


class CoordinatorQualificationAdmissionV2:
    def __init__(self, reader: SupabaseSourceReaderV2) -> None:
        self._reader = reader

    def _read(
        self,
        policy_id: str,
        parameters: Mapping[str, Any],
        context: ExecutionContextV2,
    ) -> list[Dict[str, Any]]:
        return self._reader.read(
            policy_id=policy_id,
            parameters=parameters,
            job_id=context.job_id,
            purpose=context.purpose,
            record_transport=context.record_transport,
            record_artifact=context.record_artifact,
        )

    def resolve(
        self,
        *,
        payload: Mapping[str, Any],
        context: ExecutionContextV2,
    ) -> Dict[str, Any]:
        required = {
            "schema_version",
            "epoch_id",
            "container_id",
            "container_count",
            "sequence_start",
            "leads",
            "salt_hex",
        }
        if not isinstance(payload, Mapping) or set(payload) != required:
            raise QualificationAdmissionV2Error(
                "qualification admission payload fields are invalid"
            )
        if payload.get("schema_version") != QUALIFICATION_ADMISSION_INPUT_SCHEMA_VERSION:
            raise QualificationAdmissionV2Error(
                "qualification admission schema is invalid"
            )
        epoch_id = payload.get("epoch_id")
        container_id = payload.get("container_id")
        container_count = payload.get("container_count")
        sequence_start = payload.get("sequence_start")
        leads = payload.get("leads")
        salt_hex = str(payload.get("salt_hex") or "").lower()
        if (
            not isinstance(epoch_id, int)
            or isinstance(epoch_id, bool)
            or epoch_id != context.epoch_id
            or not isinstance(container_id, int)
            or isinstance(container_id, bool)
            or container_id < 0
            or not isinstance(container_count, int)
            or isinstance(container_count, bool)
            or container_count <= 0
            or container_id >= container_count
            or not isinstance(sequence_start, int)
            or isinstance(sequence_start, bool)
            or sequence_start < 0
            or not isinstance(leads, list)
            or not _SALT_RE.fullmatch(salt_hex)
        ):
            raise QualificationAdmissionV2Error(
                "qualification admission scope is invalid"
            )

        assignment_rows = self._read(
            "qualification_epoch_assignment",
            {"epoch_id": epoch_id},
            context,
        )
        if len(assignment_rows) != 1:
            raise QualificationAdmissionV2Error(
                "epoch assignment is unavailable or ambiguous"
            )
        assignment_payload = assignment_rows[0].get("payload")
        assignment = (
            assignment_payload.get("assignment")
            if isinstance(assignment_payload, Mapping)
            else None
        )
        assigned = (
            assignment.get("assigned_lead_ids")
            if isinstance(assignment, Mapping)
            else None
        )
        if not isinstance(assigned, list):
            raise QualificationAdmissionV2Error(
                "epoch assignment lead IDs are missing"
            )
        assigned_ids = [_lead_id(value) for value in assigned]
        if len(assigned_ids) != len(set(assigned_ids)):
            raise QualificationAdmissionV2Error(
                "epoch assignment lead IDs are duplicated"
            )
        start, end = _slice_bounds(len(assigned_ids), container_id, container_count)
        expected_ids = assigned_ids[start:end]
        if sequence_start != start:
            raise QualificationAdmissionV2Error(
                "qualification sequence start differs from assignment"
            )

        proposed_ids = []
        for lead in leads:
            if not isinstance(lead, Mapping) or set(lead) != {
                "lead_id",
                "lead_blob",
                "lead_blob_hash",
                "miner_hotkey",
            }:
                raise QualificationAdmissionV2Error(
                    "qualification lead fields are invalid"
                )
            proposed_ids.append(_lead_id(lead["lead_id"]))
        if (
            len(proposed_ids) != len(set(proposed_ids))
            or set(proposed_ids) != set(expected_ids)
        ):
            raise QualificationAdmissionV2Error(
                "qualification lead set differs from epoch assignment"
            )

        source_rows = []
        for offset in range(0, len(expected_ids), 200):
            source_rows.extend(
                self._read(
                    "qualification_leads_by_ids",
                    {"lead_ids": expected_ids[offset : offset + 200]},
                    context,
                )
            )
        source_by_id = {}
        for row in source_rows:
            lead_id = _lead_id(row.get("lead_id"))
            if lead_id in source_by_id:
                raise QualificationAdmissionV2Error(
                    "authenticated qualification lead is duplicated"
                )
            source_by_id[lead_id] = {
                "lead_id": lead_id,
                "lead_blob": _rebuild_blob(row),
                "lead_blob_hash": str(row.get("lead_blob_hash") or ""),
                "miner_hotkey": str(row.get("miner_hotkey") or ""),
            }
        if set(source_by_id) != set(expected_ids):
            raise QualificationAdmissionV2Error(
                "authenticated qualification lead set is incomplete"
            )
        normalized_leads = []
        for raw_lead, lead_id in zip(leads, proposed_ids):
            normalized = dict(raw_lead)
            normalized["lead_id"] = lead_id
            if canonical_json(normalized) != canonical_json(source_by_id[lead_id]):
                raise QualificationAdmissionV2Error(
                    "qualification lead differs from authenticated source"
                )
            normalized_leads.append(normalized)

        return {
            "epoch_id": epoch_id,
            "container_id": container_id,
            "sequence_start": sequence_start,
            "leads": normalized_leads,
            "salt_hex": salt_hex,
        }
