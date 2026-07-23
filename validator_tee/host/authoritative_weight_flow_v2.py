"""Fail-closed host orchestration for authoritative V2 weight publication."""

from __future__ import annotations

import asyncio
import re
import struct
from typing import Any, Awaitable, Callable, Dict, Mapping, Optional

from leadpoet_canonical.binding import create_binding_message
from leadpoet_canonical.attested_v2 import sha256_json
from validator_tee.host.gateway_weight_inputs_v2 import (
    _gateway_endpoint,
    fetch_gateway_weight_inputs_v2,
)
from validator_tee.host.vsock_client import ValidatorEnclaveClient
from validator_tee.host.weight_authority_v2 import (
    build_authoritative_weight_bundle_v2,
    build_stateful_epoch_evidence_v1,
)
from leadpoet_canonical.weight_computation import normalize_to_u16_with_uids_pure
from leadpoet_canonical.weight_authority_v2 import (
    build_weight_finalization_submission_v2,
    validate_published_weight_bundle_v2,
)


class AuthoritativeWeightFlowV2Error(RuntimeError):
    """The V2 input, computation, binding, or publication stage failed."""


_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def _float_bits(values: Any) -> list[str]:
    return [struct.pack("!d", float(value)).hex() for value in values]


def _verify_host_vector(
    *,
    host_uids: Any,
    host_weights: Any,
    enclave_result: Mapping[str, Any],
) -> None:
    """Require the unchanged host formula and enclave result to be identical."""

    normalized_uids = [int(uid) for uid in host_uids]
    normalized_weights = [float(weight) for weight in host_weights]
    if normalized_uids != list(enclave_result.get("uids") or []):
        raise AuthoritativeWeightFlowV2Error(
            "host and authoritative enclave UID order differ"
        )
    if _float_bits(normalized_weights) != list(
        enclave_result.get("weight_float_bits") or []
    ):
        raise AuthoritativeWeightFlowV2Error(
            "host and authoritative enclave float weights differ"
        )
    non_zero_pairs = sorted(
        (
            (uid, weight)
            for uid, weight in zip(normalized_uids, normalized_weights)
            if weight > 0
        ),
        key=lambda pair: pair[0],
    )
    sparse_uids, sparse_weights = normalize_to_u16_with_uids_pure(
        [pair[0] for pair in non_zero_pairs],
        [pair[1] for pair in non_zero_pairs],
    )
    if sparse_uids != list(enclave_result.get("sparse_uids") or []):
        raise AuthoritativeWeightFlowV2Error(
            "host and authoritative enclave sparse UIDs differ"
        )
    if sparse_weights != list(
        enclave_result.get("sparse_weights_u16") or []
    ):
        raise AuthoritativeWeightFlowV2Error(
            "host and authoritative enclave u16 weights differ"
        )


async def _post_json(
    url: str,
    payload: Mapping[str, Any],
    timeout_seconds: float,
) -> Mapping[str, Any]:
    import aiohttp

    timeout = aiohttp.ClientTimeout(total=float(timeout_seconds))
    async with aiohttp.ClientSession(timeout=timeout, trust_env=False) as session:
        async with session.post(url, json=dict(payload)) as response:
            text = await response.text()
            if response.status != 200:
                raise AuthoritativeWeightFlowV2Error(
                    "gateway rejected authoritative V2 weights with HTTP %d: %s"
                    % (response.status, text[:300])
                )
            try:
                value = await response.json()
            except Exception as exc:
                raise AuthoritativeWeightFlowV2Error(
                    "gateway V2 publication response is not JSON"
                ) from exc
    if not isinstance(value, Mapping):
        raise AuthoritativeWeightFlowV2Error(
            "gateway V2 publication response is not an object"
        )
    return value


async def publish_stateful_epoch_evidence_v1(
    *,
    epoch_evidence: Optional[Mapping[str, Any]],
    gateway_url: str,
    post_json: Callable[
        [str, Mapping[str, Any], float], Awaitable[Mapping[str, Any]]
    ] = _post_json,
    timeout_seconds: float = 300.0,
) -> Optional[Dict[str, Any]]:
    """Durably publish the separate epoch proof before any chain signature."""

    if epoch_evidence is None:
        return None
    if not isinstance(epoch_evidence, Mapping):
        raise AuthoritativeWeightFlowV2Error(
            "stateful epoch evidence is not an object"
        )
    endpoint = _gateway_endpoint(gateway_url) + "/weights/subnet-epoch/boundary/v1"
    acknowledgment = await post_json(
        endpoint,
        epoch_evidence,
        float(timeout_seconds),
    )
    expected_fields = {
        "schema_version",
        "bundle_hash",
        "mapping_hash",
        "subnet_epoch_index",
        "settlement_epoch_id",
        "boundary_block",
        "epoch_authority_hash",
        "epoch_authority_receipt_hash",
        "boundary_hash",
        "boundary_receipt_hash",
        "receipt_graph_hash",
        "durable_readback_hash",
    }
    boundary = epoch_evidence.get("epoch_boundary")
    if (
        not isinstance(acknowledgment, Mapping)
        or set(acknowledgment) != expected_fields
        or acknowledgment.get("schema_version")
        != "leadpoet.subnet_epoch_boundary_ack.v1"
        or not isinstance(boundary, Mapping)
        or acknowledgment.get("bundle_hash")
        != epoch_evidence.get("bundle_hash")
        or acknowledgment.get("mapping_hash")
        != epoch_evidence.get("cutover_mapping_hash")
        or int(acknowledgment.get("subnet_epoch_index", -1))
        != int(boundary.get("subnet_epoch_index", -2))
        or int(acknowledgment.get("settlement_epoch_id", -1))
        != int(boundary.get("settlement_epoch_id", -2))
        or int(acknowledgment.get("boundary_block", -1))
        != int(boundary.get("current_block", -2))
        or acknowledgment.get("epoch_authority_hash")
        != epoch_evidence.get("epoch_authority_hash")
        or acknowledgment.get("epoch_authority_receipt_hash")
        != epoch_evidence.get("epoch_authority_receipt_hash")
        or acknowledgment.get("boundary_hash")
        != epoch_evidence.get("epoch_boundary_hash")
        or acknowledgment.get("boundary_receipt_hash")
        != epoch_evidence.get("epoch_boundary_receipt_hash")
        or acknowledgment.get("receipt_graph_hash")
        != sha256_json(dict(epoch_evidence.get("receipt_graph") or {}))
        or not _HASH_RE.fullmatch(
            str(acknowledgment.get("durable_readback_hash") or "")
        )
    ):
        raise AuthoritativeWeightFlowV2Error(
            "gateway stateful epoch evidence acknowledgment differs"
        )
    return dict(acknowledgment)


async def prepare_authoritative_weight_publication_v2(
    *,
    calculation_snapshot: Mapping[str, Any],
    host_uids: Any,
    host_weights: Any,
    validator_hotkey: str,
    allocation_hash: str,
    leaderboard_window_start: str,
    leaderboard_window_end: str,
    gateway_url: str,
    expected_chain: str,
    client: Optional[ValidatorEnclaveClient] = None,
    fetch_inputs: Callable[..., Awaitable[Dict[str, Any]]] = (
        fetch_gateway_weight_inputs_v2
    ),
    post_json: Callable[
        [str, Mapping[str, Any], float], Awaitable[Mapping[str, Any]]
    ] = _post_json,
    before_publish: Optional[Callable[[Mapping[str, Any]], Any]] = None,
    input_timeout_seconds: float = 90.0,
    publication_timeout_seconds: float = 300.0,
) -> Dict[str, Any]:
    """Build, publish, and return the only vector authorized for chain use."""

    enclave_client = client or ValidatorEnclaveClient()
    calculation = dict(calculation_snapshot)
    try:
        endpoint = _gateway_endpoint(gateway_url) + "/weights/submit/v2"
    except Exception as exc:
        raise AuthoritativeWeightFlowV2Error(str(exc)) from exc
    try:
        gateway_inputs = await fetch_inputs(
            gateway_url=gateway_url,
            calculation_snapshot=calculation,
            validator_hotkey=validator_hotkey,
            allocation_hash=allocation_hash,
            leaderboard_window_start=leaderboard_window_start,
            leaderboard_window_end=leaderboard_window_end,
            client=enclave_client,
            timeout_seconds=input_timeout_seconds,
        )
        enclave_response = await asyncio.to_thread(
            enclave_client.compute_authoritative_weights_v2,
            {
                "validator_hotkey": validator_hotkey,
                "calculation_snapshot": calculation,
                "input_receipt_hashes": gateway_inputs["input_receipt_hashes"],
                "gateway_authority_event_hash": gateway_inputs[
                    "gateway_authority_event_hash"
                ],
                "upstream_receipt_set": gateway_inputs["upstream_receipt_set"],
            },
        )
        _verify_host_vector(
            host_uids=host_uids,
            host_weights=host_weights,
            enclave_result=enclave_response["weight_result"],
        )
        boot = enclave_response["boot_identity"]
        graph = enclave_response["receipt_graph"]
        binding_message = create_binding_message(
            netuid=int(enclave_response["weight_result"]["netuid"]),
            chain=str(expected_chain),
            enclave_pubkey=str(boot["signing_pubkey"]),
            validator_code_hash=str(boot["build_manifest_hash"]),
            version=str(boot["commit_sha"]),
        )
        binding_signature = await asyncio.to_thread(
            enclave_client.sign_application_message_v2,
            binding_message.encode("utf-8"),
            parent_receipt_hash=str(graph["root_receipt_hash"]),
        )
        bundle = build_authoritative_weight_bundle_v2(
            enclave_response=enclave_response,
            validator_hotkey=validator_hotkey,
            binding_message=binding_message,
            binding_signature_result=binding_signature,
        )
        epoch_evidence = build_stateful_epoch_evidence_v1(
            enclave_response=enclave_response,
            published_bundle=bundle,
        )
        if before_publish is not None:
            prepared = {
                "weight_authorization_id": str(
                    enclave_response["weight_authorization_id"]
                ),
                "published_bundle": bundle,
                "epoch_evidence": epoch_evidence,
            }
            callback_result = before_publish(prepared)
            if asyncio.iscoroutine(callback_result):
                await callback_result
        publication = await post_json(
            endpoint,
            bundle,
            float(publication_timeout_seconds),
        )
    except AuthoritativeWeightFlowV2Error:
        raise
    except Exception as exc:
        raise AuthoritativeWeightFlowV2Error(
            "authoritative V2 weight preparation failed closed"
        ) from exc

    _validate_publication_acknowledgment_v2(
        bundle=bundle,
        publication=publication,
    )
    epoch_evidence_acknowledgment = await publish_stateful_epoch_evidence_v1(
        epoch_evidence=epoch_evidence,
        gateway_url=gateway_url,
        post_json=post_json,
        timeout_seconds=publication_timeout_seconds,
    )
    result = enclave_response["weight_result"]
    return {
        "uids": list(result["uids"]),
        "weights": list(result["weights"]),
        "sparse_uids": list(result["sparse_uids"]),
        "sparse_weights_u16": list(result["sparse_weights_u16"]),
        "weights_hash": str(result["weights_hash"]),
        "weight_authorization_id": str(
            enclave_response["weight_authorization_id"]
        ),
        "weight_submission_event_hash": str(
            publication["weight_submission_event_hash"]
        ),
        "enclave_response": dict(enclave_response),
        "published_bundle": bundle,
        "epoch_evidence": epoch_evidence,
        "epoch_evidence_acknowledgment": epoch_evidence_acknowledgment,
        "publication": dict(publication),
    }


def _validate_publication_acknowledgment_v2(
    *,
    bundle: Mapping[str, Any],
    publication: Mapping[str, Any],
) -> Dict[str, Any]:
    expected_fields = {
        "success",
        "epoch_id",
        "weights_count",
        "weights_hash",
        "weight_receipt_hash",
        "weight_submission_event_hash",
        "message",
    }
    result = bundle["weight_result"]
    verified = validate_published_weight_bundle_v2(bundle)
    if (
        not isinstance(publication, Mapping)
        or set(publication) != expected_fields
        or publication.get("success") is not True
        or int(publication.get("epoch_id", -1)) != int(result["epoch_id"])
        or int(publication.get("weights_count", -1))
        != len(result["sparse_uids"])
        or publication.get("weights_hash") != result["weights_hash"]
        or publication.get("weight_receipt_hash")
        != verified["weight_receipt_hash"]
        or not str(publication.get("weight_submission_event_hash") or "").startswith(
            "sha256:"
        )
    ):
        raise AuthoritativeWeightFlowV2Error(
            "gateway V2 publication acknowledgment differs from computed weights"
        )
    return dict(publication)


async def resume_prepared_weight_publication_v2(
    *,
    journal_record: Mapping[str, Any],
    gateway_url: str,
    post_json: Callable[
        [str, Mapping[str, Any], float], Awaitable[Mapping[str, Any]]
    ] = _post_json,
    timeout_seconds: float = 300.0,
) -> Dict[str, Any]:
    """Replay the exact pre-journaled bundle and verify an idempotent ack."""

    bundle = journal_record.get("published_bundle")
    if not isinstance(bundle, Mapping):
        raise AuthoritativeWeightFlowV2Error(
            "prepared V2 publication has no canonical bundle"
        )
    try:
        endpoint = _gateway_endpoint(gateway_url) + "/weights/submit/v2"
        publication = await post_json(endpoint, bundle, float(timeout_seconds))
        validated = _validate_publication_acknowledgment_v2(
            bundle=bundle,
            publication=publication,
        )
        epoch_evidence = journal_record.get("epoch_evidence")
        await publish_stateful_epoch_evidence_v1(
            epoch_evidence=(
                epoch_evidence
                if isinstance(epoch_evidence, Mapping)
                else None
            ),
            gateway_url=gateway_url,
            post_json=post_json,
            timeout_seconds=timeout_seconds,
        )
        return validated
    except AuthoritativeWeightFlowV2Error:
        raise
    except Exception as exc:
        raise AuthoritativeWeightFlowV2Error(
            "prepared V2 publication replay failed closed"
        ) from exc


async def finalize_authoritative_weight_publication_v2(
    *,
    prepared_publication: Mapping[str, Any],
    validator_hotkey: str,
    gateway_url: str,
    client: Optional[ValidatorEnclaveClient] = None,
    post_json: Callable[
        [str, Mapping[str, Any], float], Awaitable[Mapping[str, Any]]
    ] = _post_json,
    timeout_seconds: float = 600.0,
) -> Dict[str, Any]:
    """Prove finalized chain state and durably append it to gateway V2 lineage."""

    enclave_client = client or ValidatorEnclaveClient()
    try:
        endpoint = _gateway_endpoint(gateway_url) + "/weights/finalize/v2"
        authorization_id = str(
            prepared_publication["weight_authorization_id"]
        )
        event_hash = str(
            prepared_publication["weight_submission_event_hash"]
        )
        confirmation = await asyncio.to_thread(
            enclave_client.confirm_weight_publication_v2,
            authorization_id,
        )
        submission = build_weight_finalization_submission_v2(
            validator_hotkey=validator_hotkey,
            weight_submission_event_hash=event_hash,
            finalization=confirmation["finalization"],
            receipt_graph=confirmation["receipt_graph"],
        )
        acknowledgment = await post_json(
            endpoint,
            submission,
            float(timeout_seconds),
        )
    except AuthoritativeWeightFlowV2Error:
        raise
    except Exception as exc:
        raise AuthoritativeWeightFlowV2Error(
            "authoritative V2 finalized-chain publication failed closed"
        ) from exc
    expected_fields = {
        "success",
        "epoch_id",
        "weights_hash",
        "extrinsic_hash",
        "finalized_block",
        "weight_submission_event_hash",
        "weight_finalization_event_hash",
        "message",
    }
    finalization = confirmation["finalization"]
    if (
        not isinstance(acknowledgment, Mapping)
        or set(acknowledgment) != expected_fields
        or acknowledgment.get("success") is not True
        or int(acknowledgment.get("epoch_id", -1))
        != int(finalization["epoch_id"])
        or acknowledgment.get("weights_hash") != finalization["weights_hash"]
        or acknowledgment.get("extrinsic_hash")
        != finalization["extrinsic_hash"]
        or int(acknowledgment.get("finalized_block", -1))
        != int(finalization["finalized_block"])
        or acknowledgment.get("weight_submission_event_hash") != event_hash
        or not str(
            acknowledgment.get("weight_finalization_event_hash") or ""
        ).startswith("sha256:")
    ):
        raise AuthoritativeWeightFlowV2Error(
            "gateway V2 finalization acknowledgment differs from enclave proof"
        )
    return {
        "confirmation": confirmation,
        "submission": submission,
        "acknowledgment": dict(acknowledgment),
    }
