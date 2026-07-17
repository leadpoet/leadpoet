"""Measured source for migrating finalized pre-V2 champion settlements."""

from __future__ import annotations

import base64
import json
import re
import time
from typing import Any, Callable, Dict, Mapping, Optional

from gateway.tee.provider_broker_v2 import PROVIDER_BROKER_SCHEMA_VERSION
from leadpoet_canonical.attested_v2 import sha256_bytes
from leadpoet_canonical.legacy_settlement_v2 import (
    LEGACY_NONFINALIZATION_SCHEMA_VERSION,
    LEGACY_SETTLEMENT_SCHEMA_VERSION,
    LEGACY_SETTLEMENT_REQUEST_SCHEMA_VERSION,
    LegacySettlementV2Error,
    legacy_chain_vector_matches_bundle_v2,
    validate_legacy_allocation_nonfinalization_v2,
    validate_legacy_finalized_settlement_v2,
)


ARWEAVE_ORIGIN = "https://arweave.net"
ARWEAVE_TIMEOUT_MS = 45_000
ARWEAVE_RETRY_BACKOFF_SECONDS = (1.0, 3.0)


class CoordinatorLegacySettlementV2Error(RuntimeError):
    """A historical settlement proof could not be authenticated."""


class CoordinatorLegacySettlementSourceV2:
    def __init__(
        self,
        *,
        reader: Any,
        chain_source: Any,
        execute_provider: Callable[[Mapping[str, Any]], Mapping[str, Any]],
        retry_policy_hash: str,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._reader = reader
        self._chain_source = chain_source
        self._execute_provider = execute_provider
        self._retry_policy_hash = str(retry_policy_hash or "")
        self._sleep = sleep
        if not self._retry_policy_hash:
            raise CoordinatorLegacySettlementV2Error(
                "Arweave retry policy is unavailable"
            )

    def resolve(
        self,
        *,
        payload: Mapping[str, Any],
        context: Any,
    ) -> Dict[str, Any]:
        """Resolve only allocations proven finalized for legacy callers."""

        result = self.resolve_classification(
            payload=payload,
            context=context,
        )
        if result.get("schema_version") != LEGACY_SETTLEMENT_SCHEMA_VERSION:
            raise CoordinatorLegacySettlementV2Error(
                "legacy allocation did not reach finalized chain state"
            )
        return result

    def resolve_classification(
        self,
        *,
        payload: Mapping[str, Any],
        context: Any,
    ) -> Dict[str, Any]:
        if not isinstance(payload, Mapping) or set(payload) != {
            "schema_version",
            "netuid",
            "epoch_id",
        }:
            raise CoordinatorLegacySettlementV2Error(
                "legacy settlement request fields are invalid"
            )
        if payload.get("schema_version") != LEGACY_SETTLEMENT_REQUEST_SCHEMA_VERSION:
            raise CoordinatorLegacySettlementV2Error(
                "legacy settlement request schema is invalid"
            )
        try:
            netuid = int(payload["netuid"])
            epoch_id = int(payload["epoch_id"])
        except (TypeError, ValueError) as exc:
            raise CoordinatorLegacySettlementV2Error(
                "legacy settlement request scope is invalid"
            ) from exc
        if (
            isinstance(payload["netuid"], bool)
            or isinstance(payload["epoch_id"], bool)
            or netuid <= 0
            or epoch_id < 0
        ):
            raise CoordinatorLegacySettlementV2Error(
                "legacy settlement request scope is invalid"
            )

        anchor_rows = self._read(
            policy_id="legacy_audit_anchor_by_epoch",
            parameters={"netuid": netuid, "epoch_id": epoch_id},
            context=context,
        )
        if len(anchor_rows) != 1:
            raise CoordinatorLegacySettlementV2Error(
                "checkpointed active audit anchor is missing or ambiguous"
            )
        anchor = anchor_rows[0]
        allocation_hash = str(anchor.get("allocation_hash") or "").lower()
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", allocation_hash):
            raise CoordinatorLegacySettlementV2Error(
                "checkpointed allocation hash is invalid"
            )
        allocation_rows = self._read(
            policy_id="legacy_allocation_by_hash",
            parameters={
                "allocation_hash": allocation_hash,
                "netuid": netuid,
                "epoch_id": epoch_id,
            },
            context=context,
        )
        if len(allocation_rows) != 1:
            raise CoordinatorLegacySettlementV2Error(
                "anchor-bound legacy allocation row is missing or ambiguous"
            )
        allocation_row = allocation_rows[0]
        allocation_doc = allocation_row.get("allocation_doc")
        if (
            not isinstance(allocation_doc, Mapping)
            or allocation_row.get("allocation_hash") != allocation_hash
            or allocation_doc.get("allocation_hash") != allocation_hash
            or int(allocation_row.get("epoch") or -1) != epoch_id
            or int(allocation_row.get("netuid") or -1) != netuid
        ):
            raise CoordinatorLegacySettlementV2Error(
                "anchor-bound legacy allocation row differs from its scope"
            )

        event_hash = str(
            anchor.get("current_transparency_event_hash")
            or anchor.get("transparency_event_hash")
            or ""
        )
        log_rows = self._read(
            policy_id="legacy_transparency_event_by_hash",
            parameters={"event_hash": event_hash},
            context=context,
        )
        if len(log_rows) != 1:
            raise CoordinatorLegacySettlementV2Error(
                "signed audit event is missing or ambiguous"
            )
        log_row = log_rows[0]
        signed_entry = log_row.get("signed_log_entry")
        signed_event = (
            signed_entry.get("signed_event")
            if isinstance(signed_entry, Mapping)
            else None
        )
        audit_payload = (
            signed_event.get("payload") if isinstance(signed_event, Mapping) else None
        )
        if not isinstance(audit_payload, Mapping):
            raise CoordinatorLegacySettlementV2Error(
                "signed audit payload is missing"
            )
        expected_validator = str(audit_payload.get("actor_hotkey") or "")
        expected_weights_hash = str(anchor.get("weights_hash") or "").removeprefix(
            "sha256:"
        )

        bundles = self._read(
            policy_id="legacy_weight_bundles_by_epoch",
            parameters={"netuid": netuid, "epoch_id": epoch_id},
            context=context,
        )
        matching_bundles = [
            row
            for row in bundles
            if str(row.get("validator_hotkey") or "") == expected_validator
            and str(row.get("weights_hash") or "").removeprefix("sha256:")
            == expected_weights_hash
        ]
        if len(matching_bundles) != 1:
            raise CoordinatorLegacySettlementV2Error(
                "audit-bound legacy weight bundle is missing or ambiguous"
            )
        bundle = matching_bundles[0]
        chain_evidence = self._chain_source.read_historical_finalized_weights(
            netuid=netuid,
            epoch_id=epoch_id,
            validator_hotkey=expected_validator,
            context=context,
        )
        try:
            chain_matches = legacy_chain_vector_matches_bundle_v2(
                weight_bundle=bundle,
                chain_evidence=chain_evidence,
                expected_netuid=netuid,
                expected_epoch_id=epoch_id,
            )
        except LegacySettlementV2Error as exc:
            raise CoordinatorLegacySettlementV2Error(
                "legacy chain evidence verification failed"
            ) from exc
        if not chain_matches:
            try:
                finding = validate_legacy_allocation_nonfinalization_v2(
                    netuid=netuid,
                    epoch_id=epoch_id,
                    allocation_doc=allocation_doc,
                    weight_bundle=bundle,
                    audit_anchor=anchor,
                    transparency_log_row=log_row,
                    chain_evidence=chain_evidence,
                )
            except LegacySettlementV2Error as exc:
                raise CoordinatorLegacySettlementV2Error(
                    "legacy nonfinalization evidence verification failed"
                ) from exc
            if (
                finding.get("schema_version")
                != LEGACY_NONFINALIZATION_SCHEMA_VERSION
            ):
                raise CoordinatorLegacySettlementV2Error(
                    "legacy nonfinalization schema differs"
                )
            return finding

        tx_id = str(anchor.get("current_arweave_tx_id") or "")
        checkpoint = self._read_arweave_checkpoint(
            tx_id=tx_id,
            context=context,
            epoch_id=epoch_id,
        )
        try:
            return validate_legacy_finalized_settlement_v2(
                netuid=netuid,
                epoch_id=epoch_id,
                allocation_doc=allocation_doc,
                weight_bundle=bundle,
                audit_anchor=anchor,
                transparency_log_row=log_row,
                chain_evidence=chain_evidence,
                arweave_checkpoint=checkpoint,
            )
        except LegacySettlementV2Error as exc:
            raise CoordinatorLegacySettlementV2Error(
                "legacy settlement evidence verification failed"
            ) from exc

    def _read(
        self,
        *,
        policy_id: str,
        parameters: Mapping[str, Any],
        context: Any,
    ) -> list[Dict[str, Any]]:
        return self._reader.read(
            policy_id=policy_id,
            parameters=parameters,
            job_id=context.job_id,
            purpose=context.purpose,
            record_transport=context.record_transport,
            record_artifact=context.record_artifact,
        )

    def _read_arweave_checkpoint(
        self,
        *,
        tx_id: str,
        context: Any,
        epoch_id: int,
    ) -> Dict[str, Any]:
        if not re.fullmatch(r"[A-Za-z0-9_-]{43}", str(tx_id or "")):
            raise CoordinatorLegacySettlementV2Error(
                "Arweave transaction id is invalid"
            )
        logical_operation_id = "%s:legacy-settlement:%d:arweave-checkpoint" % (
            context.job_id,
            int(epoch_id),
        )
        last_error: Optional[str] = None
        for attempt_number in range(len(ARWEAVE_RETRY_BACKOFF_SECONDS) + 1):
            result = dict(
                self._execute_provider(
                    {
                        "schema_version": PROVIDER_BROKER_SCHEMA_VERSION,
                        "logical_operation_id": logical_operation_id,
                        "job_id": context.job_id,
                        "purpose": context.purpose,
                        "provider_id": "arweave",
                        "attempt_number": attempt_number,
                        "method": "GET",
                        "url": "%s/tx/%s/data" % (ARWEAVE_ORIGIN, tx_id),
                        "headers": {"accept": "application/json"},
                        "body_b64": base64.b64encode(b"").decode("ascii"),
                        "timeout_ms": ARWEAVE_TIMEOUT_MS,
                        "retry_policy_hash": self._retry_policy_hash,
                    }
                )
            )
            attempt = result.get("transport_attempt")
            if not isinstance(attempt, Mapping):
                raise CoordinatorLegacySettlementV2Error(
                    "Arweave terminal attempt is missing"
                )
            context.record_transport(attempt)
            context.record_artifact(str(attempt["request_artifact_hash"]))
            if attempt.get("terminal_status") == "authenticated_response":
                context.record_artifact(str(attempt["response_artifact_hash"]))
            if (
                result.get("terminal_status") == "authenticated_response"
                and 200 <= int(result.get("http_status") or 0) < 300
            ):
                try:
                    body = base64.b64decode(
                        str(result.get("body_b64") or ""), validate=True
                    )
                    encoded_checkpoint = body.strip()
                    checkpoint_bytes = base64.b64decode(
                        encoded_checkpoint + b"=" * (-len(encoded_checkpoint) % 4),
                        altchars=b"-_",
                        validate=True,
                    )
                    checkpoint = json.loads(checkpoint_bytes.decode("utf-8"))
                except Exception:
                    last_error = "malformed_json"
                else:
                    if (
                        not isinstance(checkpoint, Mapping)
                        or sha256_bytes(body) != attempt.get("response_hash")
                    ):
                        last_error = "invalid_checkpoint_response"
                    else:
                        return dict(checkpoint)
            else:
                last_error = str(
                    result.get("failure_code")
                    or "http_%s" % result.get("http_status")
                )
            if attempt_number < len(ARWEAVE_RETRY_BACKOFF_SECONDS):
                self._sleep(ARWEAVE_RETRY_BACKOFF_SECONDS[attempt_number])
        raise CoordinatorLegacySettlementV2Error(
            "Arweave checkpoint exhausted measured retries: %s"
            % (last_error or "unavailable")
        )
