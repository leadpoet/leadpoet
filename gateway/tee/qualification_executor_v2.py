"""Measured adapter for the unchanged validator qualification batch."""

from __future__ import annotations

import importlib.machinery
import json
import os
from pathlib import Path
import sys
import tempfile
import types
from typing import Any, Awaitable, Callable, Dict, Mapping, Optional

from gateway.tee.execution_job_manager_v2 import ExecutionContextV2, ExecutionResultV2
from leadpoet_canonical.attested_v2 import (
    sha256_json,
    validate_signed_execution_receipt,
)
from leadpoet_canonical.qualification_batch_v2 import (
    build_qualification_batch_output_v2,
)
from leadpoet_canonical.sourcing_history_v2 import build_sourcing_epoch_v2


QUALIFICATION_BATCH_INPUT_SCHEMA_VERSION = "leadpoet.qualification_batch_input.v2"
QUALIFICATION_EMAIL_INPUT_SCHEMA_VERSION = "leadpoet.qualification_email_input.v2"
QUALIFICATION_EPOCH_INPUT_SCHEMA_VERSION = "leadpoet.qualification_epoch_input.v2"
OP_QUALIFICATION_BATCH_V2 = "qualification_batch_v2"
OP_QUALIFICATION_EMAIL_EVIDENCE_V2 = "qualification_email_evidence_v2"
OP_QUALIFICATION_EPOCH_V2 = "qualification_epoch_v2"


class QualificationExecutorV2Error(ValueError):
    """Qualification input or attested ancestry is invalid."""


def _leadpoet_root() -> Path:
    gateway_root = Path(__file__).resolve().parents[1]
    candidates = (
        gateway_root / "_attested_runtime" / "Leadpoet",
        gateway_root.parent / "Leadpoet",
    )
    for candidate in candidates:
        if (candidate / "utils" / "utils_lead_extraction.py").is_file():
            return candidate
    raise QualificationExecutorV2Error("measured Leadpoet utility package is missing")


def install_minimal_leadpoet_namespace() -> None:
    """Expose only the exact utility modules needed by qualification checks.

    The repository's top-level ``Leadpoet.__init__`` eagerly imports the full
    validator, wallet, and subtensor stack. Those are not qualification
    dependencies and are deliberately excluded from the scoring runner.
    """

    root = _leadpoet_root()
    for name, path in (("Leadpoet", root), ("Leadpoet.utils", root / "utils")):
        existing = sys.modules.get(name)
        if existing is not None:
            existing_paths = tuple(str(item) for item in getattr(existing, "__path__", ()))
            if str(path) not in existing_paths:
                raise QualificationExecutorV2Error(
                    "Leadpoet namespace was initialized outside measured runtime"
                )
            continue
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        module.__package__ = name
        module.__spec__ = importlib.machinery.ModuleSpec(
            name=name,
            loader=None,
            is_package=True,
        )
        sys.modules[name] = module


def _broker_sentinels() -> None:
    # These values only satisfy unchanged preflight checks. Low-level HTTP
    # interception removes them, and the coordinator injects KMS-held secrets.
    for name in (
        "OPENROUTER_KEY",
        "QUALIFICATION_OPENROUTER_API_KEY",
        "QUALIFICATION_SCRAPINGDOG_API_KEY",
        "SCRAPINGDOG_API_KEY",
        "TRUELIST_API_KEY",
    ):
        os.environ[name] = "leadpoet-v2-brokered-credential"


def _source_receipt(
    value: Any,
    *,
    role: str,
    purpose: str,
    epoch_id: int,
    output_root: str,
    context: ExecutionContextV2,
) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise QualificationExecutorV2Error("qualification source receipt is missing")
    receipt = dict(value)
    validate_signed_execution_receipt(receipt)
    if (
        receipt.get("role") != role
        or receipt.get("purpose") != purpose
        or int(receipt.get("epoch_id", -1)) != epoch_id
        or receipt.get("output_root") != output_root
        or receipt.get("receipt_hash") not in context.parent_receipt_hashes
    ):
        raise QualificationExecutorV2Error(
            "qualification source receipt does not bind a declared input"
        )
    return receipt


class QualificationExecutorV2:
    def __init__(
        self,
        run_batch: Optional[
            Callable[..., Awaitable[Any]]
        ] = None,
        run_email_batch: Optional[
            Callable[[Any], Awaitable[Any]]
        ] = None,
        epoch_checker: Optional[Callable[[int, int], bool]] = None,
    ) -> None:
        self._run_batch = run_batch
        self._run_email_batch = run_email_batch
        self._epoch_checker = epoch_checker

    def _runner(self) -> Callable[..., Awaitable[Any]]:
        if self._run_batch is not None:
            return self._run_batch
        install_minimal_leadpoet_namespace()
        _broker_sentinels()
        from validator_models import automated_checks

        if self._epoch_checker is None:
            raise QualificationExecutorV2Error(
                "authenticated qualification epoch checker is unavailable"
            )
        automated_checks._check_epoch_from_block_file = self._epoch_checker

        return automated_checks.run_batch_automated_checks

    async def execute_email_evidence(
        self,
        payload: Mapping[str, Any],
        context: ExecutionContextV2,
    ) -> ExecutionResultV2:
        required = {"schema_version", "epoch_id", "leads"}
        if not isinstance(payload, Mapping) or set(payload) != required:
            raise QualificationExecutorV2Error(
                "qualification email payload fields are invalid"
            )
        if payload.get("schema_version") != QUALIFICATION_EMAIL_INPUT_SCHEMA_VERSION:
            raise QualificationExecutorV2Error(
                "qualification email schema is invalid"
            )
        epoch_id = payload.get("epoch_id")
        leads = payload.get("leads")
        if (
            not isinstance(epoch_id, int)
            or isinstance(epoch_id, bool)
            or epoch_id != context.epoch_id
            or not isinstance(leads, list)
        ):
            raise QualificationExecutorV2Error(
                "qualification email scope is invalid"
            )
        run_email_batch = self._run_email_batch
        if run_email_batch is None:
            install_minimal_leadpoet_namespace()
            _broker_sentinels()
            from validator_models.checks_email import run_centralized_truelist_batch

            run_email_batch = run_centralized_truelist_batch

        results = await run_email_batch(
            [dict(lead) for lead in leads]
        )
        if not isinstance(results, Mapping):
            raise QualificationExecutorV2Error(
                "qualification email results are invalid"
            )
        output = {
            "epoch_id": epoch_id,
            "precomputed_email_results": dict(results),
        }
        return ExecutionResultV2(
            output=output,
            artifact_hashes=(sha256_json(leads), sha256_json(output)),
        )

    async def execute_batch(
        self,
        payload: Mapping[str, Any],
        context: ExecutionContextV2,
    ) -> ExecutionResultV2:
        required = {
            "schema_version",
            "epoch_id",
            "container_id",
            "sequence_start",
            "leads",
            "precomputed_email_results",
            "salt_hex",
            "admission_receipt",
            "email_evidence_receipt",
        }
        if not isinstance(payload, Mapping) or set(payload) != required:
            raise QualificationExecutorV2Error(
                "qualification batch payload fields are invalid"
            )
        if payload.get("schema_version") != QUALIFICATION_BATCH_INPUT_SCHEMA_VERSION:
            raise QualificationExecutorV2Error("qualification batch schema is invalid")
        epoch_id = payload.get("epoch_id")
        container_id = payload.get("container_id")
        sequence_start = payload.get("sequence_start")
        leads = payload.get("leads")
        email_results = payload.get("precomputed_email_results")
        salt_hex = str(payload.get("salt_hex") or "")
        if (
            not isinstance(epoch_id, int)
            or isinstance(epoch_id, bool)
            or epoch_id != context.epoch_id
            or not isinstance(container_id, int)
            or isinstance(container_id, bool)
            or container_id < 0
            or not isinstance(sequence_start, int)
            or isinstance(sequence_start, bool)
            or sequence_start < 0
            or not isinstance(leads, list)
            or not isinstance(email_results, Mapping)
        ):
            raise QualificationExecutorV2Error("qualification batch scope is invalid")
        admission_document = {
            "epoch_id": epoch_id,
            "container_id": container_id,
            "sequence_start": sequence_start,
            "leads": leads,
            "salt_hex": salt_hex,
        }
        email_document = {
            "epoch_id": epoch_id,
            "precomputed_email_results": dict(email_results),
        }
        admission = _source_receipt(
            payload.get("admission_receipt"),
            role="gateway_coordinator",
            purpose="research_lab.admission.v2",
            epoch_id=epoch_id,
            output_root=sha256_json(admission_document),
            context=context,
        )
        email_evidence = _source_receipt(
            payload.get("email_evidence_receipt"),
            role="gateway_scoring",
            purpose="qualification.email_evidence.v2",
            epoch_id=epoch_id,
            output_root=sha256_json(email_document),
            context=context,
        )
        if sorted(context.parent_receipt_hashes) != sorted(
            (admission["receipt_hash"], email_evidence["receipt_hash"])
        ):
            raise QualificationExecutorV2Error(
                "qualification batch parents differ from source receipts"
            )

        with tempfile.TemporaryDirectory(prefix="leadpoet-qualification-v2-") as temp:
            leads_file = Path(temp) / "leads.json"
            leads_file.write_text(
                json.dumps(
                    {"truelist_results": dict(email_results)},
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                encoding="utf-8",
            )
            batch_results = await self._runner()(
                [dict(lead["lead_blob"]) for lead in leads],
                container_id=container_id,
                precomputed_email_results=None,
                leads_file_path=str(leads_file),
                current_epoch=epoch_id,
            )
        output = build_qualification_batch_output_v2(
            epoch_id=epoch_id,
            container_id=container_id,
            sequence_start=sequence_start,
            leads=leads,
            batch_results=batch_results,
            salt_hex=salt_hex,
        )
        return ExecutionResultV2(
            output=output,
            artifact_hashes=(output["automated_checks_root"], output["batch_hash"]),
        )

    def aggregate_epoch(
        self,
        payload: Mapping[str, Any],
        context: ExecutionContextV2,
    ) -> ExecutionResultV2:
        required = {"schema_version", "epoch_id", "batches"}
        if not isinstance(payload, Mapping) or set(payload) != required:
            raise QualificationExecutorV2Error(
                "qualification epoch payload fields are invalid"
            )
        if payload.get("schema_version") != QUALIFICATION_EPOCH_INPUT_SCHEMA_VERSION:
            raise QualificationExecutorV2Error("qualification epoch schema is invalid")
        epoch_id = payload.get("epoch_id")
        batches = payload.get("batches")
        if epoch_id != context.epoch_id or not isinstance(batches, list):
            raise QualificationExecutorV2Error("qualification epoch scope is invalid")
        decisions = []
        receipt_hashes = []
        seen_containers = set()
        for row in batches:
            if not isinstance(row, Mapping) or set(row) != {"receipt", "output"}:
                raise QualificationExecutorV2Error("qualification batch row is invalid")
            from leadpoet_canonical.qualification_batch_v2 import (
                validate_qualification_batch_output_v2,
            )

            output = validate_qualification_batch_output_v2(row["output"])
            receipt = _source_receipt(
                row["receipt"],
                role="gateway_scoring",
                purpose="qualification.lead_decision.v2",
                epoch_id=epoch_id,
                output_root=sha256_json(output),
                context=context,
            )
            if output["epoch_id"] != epoch_id or output["container_id"] in seen_containers:
                raise QualificationExecutorV2Error(
                    "qualification batch epoch or container is duplicated"
                )
            seen_containers.add(output["container_id"])
            decisions.extend(output["sourcing_decisions"])
            receipt_hashes.append(receipt["receipt_hash"])
        if sorted(context.parent_receipt_hashes) != sorted(receipt_hashes):
            raise QualificationExecutorV2Error(
                "qualification epoch parents differ from batch receipts"
            )
        source_doc = build_sourcing_epoch_v2(
            epoch_id=epoch_id,
            decisions=decisions,
        )
        return ExecutionResultV2(
            output=source_doc,
            artifact_hashes=(source_doc["decision_root"], source_doc["epoch_hash"]),
        )
