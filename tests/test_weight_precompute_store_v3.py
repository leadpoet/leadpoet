from __future__ import annotations

from copy import deepcopy

import pytest

from gateway.research_lab import weight_precompute_store_v3 as precompute
from leadpoet_canonical.attested_v2 import build_transport_attempt
from leadpoet_canonical.weight_authority_v2 import GATEWAY_WEIGHT_INPUT_CATEGORIES


def _sha(value: int) -> str:
    return "sha256:%064x" % value


def _result(*, compact: bool) -> dict:
    result = {
        "input_receipt_hashes": {
            category: _sha(index + 1)
            for index, category in enumerate(sorted(GATEWAY_WEIGHT_INPUT_CATEGORIES))
        },
        "gateway_authority_event_hash": _sha(20),
        "upstream_receipt_set": {
            "boot_identities": [],
            "receipts": [],
            "transport_attempts": [{"credential_ref_hash": _sha(21)}],
            "host_operations": [],
        },
        "compact_ancestry": None,
    }
    if compact:
        result["compact_ancestry"] = {
            "upstream_ancestry_proofs": {},
            "upstream_transport_attempts": [],
        }
    else:
        # This is the full gateway result shape. The store must preserve it,
        # rather than dropping the forensic execution material for compact use.
        result["executions"] = {
            "gateway_only": {
                "status": "succeeded",
                "credential_ref_hashes": {"supabase": _sha(22)},
            }
        }
    return result


def _kwargs(*, compact: bool = False) -> dict:
    return {
        "request_hash": _sha(30),
        "release_commit_sha": "a" * 40,
        "release_manifest_hash": _sha(31),
        "cutover": {"network_genesis_hash": "0x" + "b" * 64, "netuid": 71},
        "epoch_id": 123,
        "epoch_ref": _sha(32),
        "planned_submission_block": 456,
        "calculation_snapshot_hash": _sha(33),
        "source_input_root": _sha(34),
        "gateway_result": _result(compact=compact),
    }


def _canonical_transport_attempt() -> dict:
    return build_transport_attempt(
        request_id="d" * 32,
        logical_operation_id="weight-precompute-provider-call",
        job_id="weight-precompute-run",
        purpose="research_lab.candidate_score.v2",
        provider_id="openrouter",
        attempt_number=0,
        method="POST",
        destination_host="openrouter.ai",
        destination_port=443,
        path_hash=_sha(40),
        nonsecret_headers_hash=_sha(41),
        body_hash=_sha(42),
        credential_ref_hash=_sha(43),
        retry_policy_hash=_sha(44),
        timeout_ms=30_000,
        started_at="2026-08-13T00:00:00Z",
        terminal_status="authenticated_response",
        http_status=200,
        response_hash=_sha(45),
        request_artifact_hash=_sha(46),
        response_artifact_hash=_sha(47),
        tls_peer_chain_hash=_sha(48),
        tls_protocol="TLSv1.3",
        failure_code=None,
        completed_at="2026-08-13T00:00:01Z",
    )


@pytest.mark.asyncio
async def test_persist_keeps_exact_full_gateway_result_and_verifies_readback(monkeypatch):
    durable: dict[str, dict] = {}
    calls: list[tuple[str, dict]] = []

    async def fake_call_rpc(name: str, params: dict):
        calls.append((name, deepcopy(params)))
        if name == "begin_research_lab_weight_precompute_run_v3":
            durable["run"] = {
                "precompute_run_id": params["p_precompute_run_id"],
                "network_genesis_hash": params["p_network_genesis_hash"],
                "netuid": params["p_netuid"],
                "epoch_id": params["p_epoch_id"],
                "epoch_ref": params["p_epoch_ref"],
                "request_hash": params["p_request_hash"],
                "planned_submission_block": params["p_planned_submission_block"],
                "release_commit_sha": params["p_release_commit_sha"],
                "release_manifest_hash": params["p_release_manifest_hash"],
                "run_doc": params["p_run_doc"],
            }
            return [durable["run"]]
        if name == "record_research_lab_weight_precompute_input_set_v3":
            durable["inputs"] = {
                "input_set_hash": params["p_input_set_hash"],
                "source_input_root": params["p_source_input_root"],
                "calculation_snapshot_hash": params[
                    "p_calculation_snapshot_hash"
                ],
                "input_receipt_hashes": params["p_input_receipt_hashes"],
                "input_set_doc": params["p_input_set_doc"],
            }
            return [durable["inputs"]]
        assert name == "research_lab_weight_precompute_readback_v3"
        return {"run": durable["run"], "complete_input_set": durable["inputs"], "stage_events": []}

    monkeypatch.setattr(precompute.store, "call_rpc", fake_call_rpc)
    arguments = _kwargs(compact=False)
    result = await precompute.GatewayWeightPrecomputeStoreV3().persist(**arguments)

    assert [name for name, _params in calls] == [
        "begin_research_lab_weight_precompute_run_v3",
        "record_research_lab_weight_precompute_input_set_v3",
        "research_lab_weight_precompute_readback_v3",
    ]
    assert result["complete_input_set"]["input_set_doc"][
        "gateway_result"
    ] == arguments["gateway_result"]
    assert calls[0][1][
        "p_precompute_run_id"
    ] == precompute.precompute_run_id_for_request_hash(
        arguments["request_hash"]
    )
    assert calls[0][1]["p_request_hash"] == arguments["request_hash"]
    assert calls[0][1]["p_network_genesis_hash"] == arguments["cutover"]["network_genesis_hash"]
    assert "authorization" not in str(calls)


@pytest.mark.asyncio
async def test_load_rejects_readback_that_changes_the_compact_gateway_result(monkeypatch):
    arguments = _kwargs(compact=True)
    request = precompute.GatewayWeightPrecomputeStoreV3._request(**arguments)
    tampered = deepcopy(request["input_set_doc"])
    tampered["gateway_result"]["compact_ancestry"] = None

    async def fake_call_rpc(name: str, _params: dict):
        assert name == "research_lab_weight_precompute_readback_v3"
        return {
            "run": {field: request[field] for field in (
                "precompute_run_id", "network_genesis_hash", "netuid", "epoch_id",
                "epoch_ref", "request_hash", "planned_submission_block",
                "release_commit_sha", "release_manifest_hash", "run_doc",
            )},
            "complete_input_set": {
                "input_set_hash": request["input_set_hash"],
                "source_input_root": request["source_input_root"],
                "calculation_snapshot_hash": request["calculation_snapshot_hash"],
                "input_receipt_hashes": request["input_receipt_hashes"],
                "input_set_doc": tampered,
            },
            "stage_events": [],
        }

    monkeypatch.setattr(precompute.store, "call_rpc", fake_call_rpc)
    with pytest.raises(
        precompute.GatewayWeightPrecomputeStoreV3Error,
        match="durable readback differs",
    ):
        await precompute.GatewayWeightPrecomputeStoreV3().load(**arguments)


def test_gateway_frontier_requires_exactly_nine_inputs_and_no_authorization():
    arguments = _kwargs()
    arguments["gateway_result"]["input_receipt_hashes"].pop("bans")
    with pytest.raises(
        precompute.GatewayWeightPrecomputeStoreV3Error,
        match="categories are incomplete",
    ):
        precompute.GatewayWeightPrecomputeStoreV3._request(**arguments)

    arguments = _kwargs()
    arguments["gateway_result"]["authorization"] = "forbidden"
    with pytest.raises(
        precompute.GatewayWeightPrecomputeStoreV3Error,
        match="secret or authorization",
    ):
        precompute.GatewayWeightPrecomputeStoreV3._request(**arguments)


def test_gateway_frontier_allows_canonical_transport_nonsecret_headers_hash():
    arguments = _kwargs()
    attempt = _canonical_transport_attempt()
    arguments["gateway_result"]["upstream_receipt_set"]["transport_attempts"] = [
        attempt
    ]

    request = precompute.GatewayWeightPrecomputeStoreV3._request(**arguments)

    assert request["input_set_doc"]["gateway_result"]["upstream_receipt_set"][
        "transport_attempts"
    ] == [attempt]


def test_gateway_frontier_rejects_raw_nonsecret_headers_hash_value():
    arguments = _kwargs()
    attempt = _canonical_transport_attempt()
    attempt["nonsecret_headers_hash"] = "raw-secret-value"
    arguments["gateway_result"]["upstream_receipt_set"]["transport_attempts"] = [
        attempt
    ]

    with pytest.raises(
        precompute.GatewayWeightPrecomputeStoreV3Error,
        match="secret or authorization",
    ):
        precompute.GatewayWeightPrecomputeStoreV3._request(**arguments)


@pytest.mark.parametrize(
    "unsafe_key",
    [
        "authorization",
        "proxy-authorization",
        "raw_secret",
        "raw_credential",
        "openrouter_api_key",
        "service_role_key",
        "access_token",
    ],
)
def test_gateway_frontier_rejects_raw_secret_and_authorization_fields(
    unsafe_key,
):
    arguments = _kwargs()
    arguments["gateway_result"]["nested"] = {unsafe_key: "forbidden"}
    with pytest.raises(
        precompute.GatewayWeightPrecomputeStoreV3Error,
        match="secret or authorization",
    ):
        precompute.GatewayWeightPrecomputeStoreV3._request(**arguments)
