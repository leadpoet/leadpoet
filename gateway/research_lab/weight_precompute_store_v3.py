"""Durable gateway frontier for V2 weight-input precomputation.

This adapter persists only the nine gateway-owned input receipts.  The
validator enclave remains the sole creator of chain, metagraph, derived, and
five validator-owned V2 receipts.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping
from uuid import NAMESPACE_URL, uuid5

from gateway.research_lab import store
from leadpoet_canonical.attested_v2 import canonical_json, sha256_json
from leadpoet_canonical.weight_authority_v2 import (
    GATEWAY_WEIGHT_INPUT_CATEGORIES,
)


STORE_SCHEMA_VERSION = "leadpoet.gateway_weight_precompute_store.v3"
RUN_SCHEMA_VERSION = "leadpoet.gateway_weight_precompute_run.v3"
INPUT_SET_SCHEMA_VERSION = "leadpoet.gateway_weight_precompute_input_set.v3"
PROTOCOL_SCHEMA_VERSION = "leadpoet.published_weight_bundle.v2"

_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_GENESIS_RE = re.compile(r"^0x[0-9a-f]{64}$")
_UNSAFE_KEY_RE = re.compile(
    r"(?:authorization|secret|credential|password|api[_-]?key|service[_-]?role|token)",
    re.IGNORECASE,
)
_SAFE_CREDENTIAL_PROOF_KEY_RE = re.compile(
    r"^credential_ref_hash(?:es)?$",
    re.IGNORECASE,
)


class GatewayWeightPrecomputeStoreV3Error(RuntimeError):
    """The durable gateway weight frontier is invalid or differs on readback."""


def _hash(value: Any, field: str) -> str:
    normalized = str(value or "").strip().lower()
    if not _HASH_RE.fullmatch(normalized):
        raise GatewayWeightPrecomputeStoreV3Error("%s is invalid" % field)
    return normalized


def _commit(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if not _COMMIT_RE.fullmatch(normalized):
        raise GatewayWeightPrecomputeStoreV3Error("release commit is invalid")
    return normalized


def _canonical_object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise GatewayWeightPrecomputeStoreV3Error("%s is invalid" % field)
    try:
        normalized = json.loads(canonical_json(dict(value)))
    except (TypeError, ValueError) as exc:
        raise GatewayWeightPrecomputeStoreV3Error(
            "%s is not canonical JSON" % field
        ) from exc
    if not isinstance(normalized, dict):
        raise GatewayWeightPrecomputeStoreV3Error("%s is invalid" % field)
    return normalized


def _reject_unsafe_values(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            if (
                _UNSAFE_KEY_RE.search(key_text)
                and not _SAFE_CREDENTIAL_PROOF_KEY_RE.fullmatch(key_text)
            ):
                raise GatewayWeightPrecomputeStoreV3Error(
                    "gateway result contains a secret or authorization field"
                )
            _reject_unsafe_values(child)
    elif isinstance(value, list):
        for child in value:
            _reject_unsafe_values(child)


def _gateway_input_receipt_hashes(value: Mapping[str, Any]) -> dict[str, str]:
    input_hashes = value.get("input_receipt_hashes")
    if not isinstance(input_hashes, Mapping) or set(input_hashes) != set(
        GATEWAY_WEIGHT_INPUT_CATEGORIES
    ):
        raise GatewayWeightPrecomputeStoreV3Error(
            "gateway input receipt categories are incomplete"
        )
    normalized = {
        category: _hash(input_hashes[category], "%s receipt hash" % category)
        for category in sorted(GATEWAY_WEIGHT_INPUT_CATEGORIES)
    }
    if any(input_hashes[category] != normalized[category] for category in normalized):
        raise GatewayWeightPrecomputeStoreV3Error(
            "gateway input receipt hashes are not canonical"
        )
    if len(set(normalized.values())) != len(normalized):
        raise GatewayWeightPrecomputeStoreV3Error(
            "gateway input receipt hashes are not unique"
        )
    return normalized


def precompute_run_id_for_request_hash(request_hash: str) -> str:
    """Return the stable V3 run id for one immutable gateway request."""

    normalized = _hash(request_hash, "request hash")
    return str(
        uuid5(
            NAMESPACE_URL,
            "leadpoet:gateway_weight_precompute.v3:" + normalized,
        )
    )


def _single_row(value: Any, label: str) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], Mapping):
        return value[0]
    raise GatewayWeightPrecomputeStoreV3Error("%s returned an invalid row" % label)


class GatewayWeightPrecomputeStoreV3:
    """Persist and load the gateway-only V2 weight-input frontier by RPC."""

    @staticmethod
    def _request(
        *,
        request_hash: str,
        release_commit_sha: str,
        release_manifest_hash: str,
        cutover: Mapping[str, Any],
        epoch_id: int,
        epoch_ref: str,
        planned_submission_block: int,
        calculation_snapshot_hash: str,
        source_input_root: str,
        gateway_result: Mapping[str, Any],
    ) -> dict[str, Any]:
        request = _hash(request_hash, "request hash")
        commit = _commit(release_commit_sha)
        manifest = _hash(release_manifest_hash, "release manifest hash")
        if not isinstance(cutover, Mapping):
            raise GatewayWeightPrecomputeStoreV3Error("cutover is invalid")
        genesis = str(cutover.get("network_genesis_hash") or "").lower()
        netuid = cutover.get("netuid")
        if (
            not _GENESIS_RE.fullmatch(genesis)
            or not isinstance(netuid, int)
            or isinstance(netuid, bool)
            or netuid <= 0
        ):
            raise GatewayWeightPrecomputeStoreV3Error("cutover network identity is invalid")
        if not isinstance(epoch_id, int) or isinstance(epoch_id, bool) or epoch_id < 0:
            raise GatewayWeightPrecomputeStoreV3Error("epoch id is invalid")
        if (
            not isinstance(planned_submission_block, int)
            or isinstance(planned_submission_block, bool)
            or planned_submission_block < 0
        ):
            raise GatewayWeightPrecomputeStoreV3Error("planned submission block is invalid")

        result = _canonical_object(gateway_result, "gateway result")
        _reject_unsafe_values(result)
        receipt_hashes = _gateway_input_receipt_hashes(result)
        authority_hash = _hash(
            result.get("gateway_authority_event_hash"),
            "gateway authority event hash",
        )
        if result.get("gateway_authority_event_hash") != authority_hash:
            raise GatewayWeightPrecomputeStoreV3Error(
                "gateway authority event hash is not canonical"
            )
        input_set_doc = {
            "schema_version": INPUT_SET_SCHEMA_VERSION,
            "request_hash": request,
            "calculation_snapshot_hash": _hash(
                calculation_snapshot_hash, "calculation snapshot hash"
            ),
            "source_input_root": _hash(source_input_root, "source input root"),
            "gateway_result": result,
            "gateway_result_hash": sha256_json(result),
        }
        return {
            "precompute_run_id": precompute_run_id_for_request_hash(request),
            "network_genesis_hash": genesis,
            "netuid": netuid,
            "epoch_id": epoch_id,
            "epoch_ref": _hash(epoch_ref, "epoch ref"),
            "request_hash": request,
            "planned_submission_block": planned_submission_block,
            "release_commit_sha": commit,
            "release_manifest_hash": manifest,
            "run_doc": {
                "schema_version": RUN_SCHEMA_VERSION,
                "request_hash": request,
                "calculation_snapshot_hash": input_set_doc[
                    "calculation_snapshot_hash"
                ],
            },
            "input_set_hash": sha256_json(input_set_doc),
            "source_input_root": input_set_doc["source_input_root"],
            "calculation_snapshot_hash": input_set_doc[
                "calculation_snapshot_hash"
            ],
            "input_receipt_hashes": receipt_hashes,
            "input_set_doc": input_set_doc,
        }

    @staticmethod
    def _verify_readback(readback: Any, request: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(readback, Mapping):
            raise GatewayWeightPrecomputeStoreV3Error("precompute readback is invalid")
        run = readback.get("run")
        inputs = readback.get("complete_input_set")
        events = readback.get("stage_events")
        if not isinstance(run, Mapping) or not isinstance(inputs, Mapping) or events != []:
            raise GatewayWeightPrecomputeStoreV3Error("precompute readback is incomplete")
        run_fields = (
            "precompute_run_id", "network_genesis_hash", "netuid", "epoch_id",
            "epoch_ref", "request_hash", "planned_submission_block",
            "release_commit_sha", "release_manifest_hash", "run_doc",
        )
        input_fields = (
            "input_set_hash", "source_input_root", "calculation_snapshot_hash",
            "input_receipt_hashes", "input_set_doc",
        )
        if any(run.get(field) != request[field] for field in run_fields) or any(
            inputs.get(field) != request[field] for field in input_fields
        ):
            raise GatewayWeightPrecomputeStoreV3Error(
                "precompute durable readback differs"
            )
        if inputs["input_set_doc"].get("gateway_result_hash") != sha256_json(
            inputs["input_set_doc"].get("gateway_result")
        ):
            raise GatewayWeightPrecomputeStoreV3Error(
                "precompute gateway result hash differs"
            )
        return dict(readback)

    async def persist(
        self,
        **kwargs: Any,
    ) -> dict[str, Any]:
        request = self._request(**kwargs)
        started = await store.call_rpc(
            "begin_research_lab_weight_precompute_run_v3",
            {
                "p_precompute_run_id": request["precompute_run_id"],
                "p_network_genesis_hash": request["network_genesis_hash"],
                "p_netuid": request["netuid"],
                "p_epoch_id": request["epoch_id"],
                "p_epoch_ref": request["epoch_ref"],
                "p_request_hash": request["request_hash"],
                "p_planned_submission_block": request["planned_submission_block"],
                "p_release_commit_sha": request["release_commit_sha"],
                "p_release_manifest_hash": request["release_manifest_hash"],
                "p_run_doc": request["run_doc"],
            },
        )
        started_row = _single_row(started, "precompute run")
        if started_row.get("precompute_run_id") != request["precompute_run_id"]:
            raise GatewayWeightPrecomputeStoreV3Error("precompute run differs")
        recorded = await store.call_rpc(
            "record_research_lab_weight_precompute_input_set_v3",
            {
                "p_precompute_run_id": request["precompute_run_id"],
                "p_input_set_hash": request["input_set_hash"],
                "p_source_input_root": request["source_input_root"],
                "p_calculation_snapshot_hash": request["calculation_snapshot_hash"],
                "p_input_receipt_hashes": request["input_receipt_hashes"],
                "p_input_set_doc": request["input_set_doc"],
            },
        )
        recorded_row = _single_row(recorded, "precompute input set")
        if recorded_row.get("input_set_hash") != request["input_set_hash"]:
            raise GatewayWeightPrecomputeStoreV3Error("precompute input set differs")
        readback = await store.call_rpc(
            "research_lab_weight_precompute_readback_v3",
            {"p_precompute_run_id": request["precompute_run_id"]},
        )
        return self._verify_readback(readback, request)

    async def load(self, **kwargs: Any) -> dict[str, Any]:
        request = self._request(**kwargs)
        readback = await store.call_rpc(
            "research_lab_weight_precompute_readback_v3",
            {"p_precompute_run_id": request["precompute_run_id"]},
        )
        return self._verify_readback(readback, request)


async def persist_gateway_weight_precompute_v3(**kwargs: Any) -> dict[str, Any]:
    """Persist one exact gateway-owned V2 precompute frontier."""

    return await GatewayWeightPrecomputeStoreV3().persist(**kwargs)


async def load_gateway_weight_precompute_v3(**kwargs: Any) -> dict[str, Any]:
    """Load and verify one exact gateway-owned V2 precompute frontier."""

    return await GatewayWeightPrecomputeStoreV3().load(**kwargs)
