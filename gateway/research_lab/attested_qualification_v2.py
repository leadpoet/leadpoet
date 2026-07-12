"""Host orchestration primitives for measured V2 qualification lineage.

These helpers do not replace the current coordinator/worker scheduling. They
submit already scheduled work to measured roles, verify complete parent graphs,
and persist the final sourcing aggregate for weight lineage.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from gateway.research_lab.attested_coordinator_v2 import execute_coordinator_v2
from gateway.research_lab.attested_scoring_v2 import execute_scoring_v2
from gateway.research_lab.attested_v2_store import persist_sourcing_epoch_v2
from gateway.tee.coordinator_executor_v2 import (
    OP_ATTEST_QUALIFICATION_ADMISSION,
)
from gateway.tee.qualification_admission_v2 import (
    QUALIFICATION_ADMISSION_INPUT_SCHEMA_VERSION,
)
from gateway.tee.qualification_executor_v2 import (
    OP_QUALIFICATION_BATCH_V2,
    OP_QUALIFICATION_EMAIL_EVIDENCE_V2,
    OP_QUALIFICATION_EPOCH_V2,
    QUALIFICATION_BATCH_INPUT_SCHEMA_VERSION,
    QUALIFICATION_EMAIL_INPUT_SCHEMA_VERSION,
    QUALIFICATION_EPOCH_INPUT_SCHEMA_VERSION,
)
from leadpoet_canonical.attested_v2 import sha256_json, validate_receipt_graph
from leadpoet_canonical.qualification_batch_v2 import (
    validate_qualification_batch_output_v2,
)
from leadpoet_canonical.sourcing_history_v2 import validate_sourcing_epoch_v2


class AttestedQualificationV2Error(RuntimeError):
    """A measured qualification stage is incomplete or has invalid ancestry."""


def _execution(value: Mapping[str, Any], *, purpose: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or value.get("status") != "succeeded":
        raise AttestedQualificationV2Error("V2 qualification execution failed")
    graph = value.get("receipt_graph")
    receipt = value.get("receipt")
    result = value.get("result")
    if (
        not isinstance(graph, Mapping)
        or not isinstance(receipt, Mapping)
        or not isinstance(result, Mapping)
    ):
        raise AttestedQualificationV2Error(
            "V2 qualification execution result is incomplete"
        )
    validate_receipt_graph(graph, required_purposes={purpose})
    if (
        graph.get("root_receipt_hash") != receipt.get("receipt_hash")
        or receipt.get("purpose") != purpose
        or receipt.get("output_root") != sha256_json(result)
    ):
        raise AttestedQualificationV2Error(
            "V2 qualification execution receipt is invalid"
        )
    return dict(value)


async def attest_qualification_admission_v2(
    *,
    epoch_id: int,
    container_id: int,
    container_count: int,
    sequence_start: int,
    leads: Sequence[Mapping[str, Any]],
    salt_hex: str,
    sequence: int,
    execute: Any = execute_coordinator_v2,
    **execution_options: Any,
) -> dict[str, Any]:
    payload = {
        "schema_version": QUALIFICATION_ADMISSION_INPUT_SCHEMA_VERSION,
        "epoch_id": epoch_id,
        "container_id": container_id,
        "container_count": container_count,
        "sequence_start": sequence_start,
        "leads": [dict(lead) for lead in leads],
        "salt_hex": str(salt_hex),
    }
    value = await execute(
        operation=OP_ATTEST_QUALIFICATION_ADMISSION,
        purpose="research_lab.admission.v2",
        epoch_id=epoch_id,
        sequence=sequence,
        payload=payload,
        **execution_options,
    )
    return _execution(value, purpose="research_lab.admission.v2")


async def attest_qualification_email_evidence_v2(
    *,
    epoch_id: int,
    leads: Sequence[Mapping[str, Any]],
    admission_graphs: Sequence[Mapping[str, Any]],
    sequence: int,
    worker_index: int,
    execute: Any = execute_scoring_v2,
    **execution_options: Any,
) -> dict[str, Any]:
    value = await execute(
        operation=OP_QUALIFICATION_EMAIL_EVIDENCE_V2,
        purpose="qualification.email_evidence.v2",
        epoch_id=epoch_id,
        sequence=sequence,
        payload={
            "schema_version": QUALIFICATION_EMAIL_INPUT_SCHEMA_VERSION,
            "epoch_id": epoch_id,
            "leads": [dict(lead) for lead in leads],
        },
        worker_index=worker_index,
        parent_graphs=tuple(admission_graphs),
        **execution_options,
    )
    return _execution(value, purpose="qualification.email_evidence.v2")


async def execute_qualification_batch_v2(
    *,
    admission: Mapping[str, Any],
    email_evidence: Mapping[str, Any],
    sequence: int,
    worker_index: int,
    execute: Any = execute_scoring_v2,
    **execution_options: Any,
) -> dict[str, Any]:
    admission_value = _execution(
        admission,
        purpose="research_lab.admission.v2",
    )
    email_value = _execution(
        email_evidence,
        purpose="qualification.email_evidence.v2",
    )
    admission_doc = admission_value["result"]
    email_doc = email_value["result"]
    if int(admission_doc["epoch_id"]) != int(email_doc["epoch_id"]):
        raise AttestedQualificationV2Error(
            "qualification admission and email epochs differ"
        )
    value = await execute(
        operation=OP_QUALIFICATION_BATCH_V2,
        purpose="qualification.lead_decision.v2",
        epoch_id=int(admission_doc["epoch_id"]),
        sequence=sequence,
        payload={
            "schema_version": QUALIFICATION_BATCH_INPUT_SCHEMA_VERSION,
            "epoch_id": admission_doc["epoch_id"],
            "container_id": admission_doc["container_id"],
            "sequence_start": admission_doc["sequence_start"],
            "leads": list(admission_doc["leads"]),
            "precomputed_email_results": dict(
                email_doc["precomputed_email_results"]
            ),
            "salt_hex": admission_doc["salt_hex"],
            "admission_receipt": dict(admission_value["receipt"]),
            "email_evidence_receipt": dict(email_value["receipt"]),
        },
        worker_index=worker_index,
        parent_graphs=(
            admission_value["receipt_graph"],
            email_value["receipt_graph"],
        ),
        **execution_options,
    )
    normalized = _execution(value, purpose="qualification.lead_decision.v2")
    validate_qualification_batch_output_v2(normalized["result"])
    return normalized


async def aggregate_qualification_epoch_v2(
    *,
    epoch_id: int,
    batches: Sequence[Mapping[str, Any]],
    sequence: int,
    worker_index: int,
    execute: Any = execute_scoring_v2,
    persist_epoch: Any = persist_sourcing_epoch_v2,
    **execution_options: Any,
) -> dict[str, Any]:
    normalized_batches = [
        _execution(batch, purpose="qualification.lead_decision.v2")
        for batch in batches
    ]
    if any(int(batch["result"]["epoch_id"]) != epoch_id for batch in normalized_batches):
        raise AttestedQualificationV2Error(
            "qualification batch epoch differs from aggregate"
        )
    value = await execute(
        operation=OP_QUALIFICATION_EPOCH_V2,
        purpose="qualification.sourcing_epoch.v2",
        epoch_id=epoch_id,
        sequence=sequence,
        payload={
            "schema_version": QUALIFICATION_EPOCH_INPUT_SCHEMA_VERSION,
            "epoch_id": epoch_id,
            "batches": [
                {
                    "receipt": dict(batch["receipt"]),
                    "output": dict(batch["result"]),
                }
                for batch in normalized_batches
            ],
        },
        worker_index=worker_index,
        parent_graphs=tuple(batch["receipt_graph"] for batch in normalized_batches),
        **execution_options,
    )
    aggregate = _execution(value, purpose="qualification.sourcing_epoch.v2")
    source_doc = validate_sourcing_epoch_v2(aggregate["result"])
    persistence = await persist_epoch(
        source_doc=source_doc,
        graph=aggregate["receipt_graph"],
    )
    if persistence.get("receipt_hash") != aggregate["receipt"]["receipt_hash"]:
        raise AttestedQualificationV2Error(
            "qualification epoch durable receipt differs"
        )
    return {**aggregate, "epoch_persistence": dict(persistence)}
