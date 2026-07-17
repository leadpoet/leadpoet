from __future__ import annotations

import base64
import json

from gateway.tee import coordinator_legacy_settlement_v2 as source_module
from gateway.tee.coordinator_legacy_settlement_v2 import (
    CoordinatorLegacySettlementSourceV2,
)
from gateway.tee.execution_job_manager_v2 import ExecutionContextV2
from leadpoet_canonical.attested_v2 import (
    build_transport_attempt,
    sha256_bytes,
    sha256_json,
)
from leadpoet_canonical.legacy_settlement_v2 import (
    LEGACY_SETTLEMENT_REQUEST_SCHEMA_VERSION,
)


HASH = "sha256:" + "a" * 64


class FakeReader:
    def __init__(self):
        self.calls = []

    def read(self, *, policy_id, parameters, **_kwargs):
        self.calls.append((policy_id, dict(parameters)))
        if policy_id == "legacy_audit_anchor_by_epoch":
            return [
                {
                    "allocation_hash": "sha256:" + "1" * 64,
                    "weights_hash": "2" * 64,
                    "current_transparency_event_hash": "3" * 64,
                    "current_arweave_tx_id": "A" * 43,
                }
            ]
        if policy_id == "legacy_allocation_by_hash":
            return [
                {
                    "epoch": 100,
                    "netuid": 71,
                    "allocation_hash": "sha256:" + "1" * 64,
                    "allocation_doc": {
                        "allocation_hash": "sha256:" + "1" * 64,
                    },
                }
            ]
        if policy_id == "legacy_transparency_event_by_hash":
            return [
                {
                    "signed_log_entry": {
                        "signed_event": {
                            "payload": {"actor_hotkey": "validator-1"}
                        }
                    }
                }
            ]
        if policy_id == "legacy_weight_bundles_by_epoch":
            return [
                {
                    "validator_hotkey": "validator-1",
                    "weights_hash": "2" * 64,
                },
                {
                    "validator_hotkey": "other-validator",
                    "weights_hash": "4" * 64,
                },
            ]
        raise AssertionError(policy_id)


class FakeChainSource:
    def __init__(self):
        self.calls = []

    def read_historical_finalized_weights(self, **kwargs):
        self.calls.append(dict(kwargs))
        return {"chain": "evidence"}


class FakeArweave:
    def __init__(self, *, fail_first=False):
        self.fail_first = bool(fail_first)
        self.calls = []

    def __call__(self, request):
        self.calls.append(dict(request))
        authenticated = not (self.fail_first and len(self.calls) == 1)
        checkpoint_body = json.dumps(
            {"header": {}, "signature": "x", "events_compressed": "", "tree_levels": []},
            separators=(",", ":"),
        ).encode()
        body = base64.urlsafe_b64encode(checkpoint_body).rstrip(b"=")
        attempt = build_transport_attempt(
            request_id=("%032x" % len(self.calls)),
            logical_operation_id=request["logical_operation_id"],
            job_id=request["job_id"],
            purpose=request["purpose"],
            provider_id="arweave",
            attempt_number=request["attempt_number"],
            method="GET",
            destination_host="arweave.net",
            destination_port=443,
            path_hash=HASH,
            nonsecret_headers_hash=HASH,
            body_hash=sha256_bytes(b""),
            credential_ref_hash=HASH,
            retry_policy_hash=request["retry_policy_hash"],
            timeout_ms=request["timeout_ms"],
            started_at="2026-07-10T00:00:00Z",
            terminal_status=(
                "authenticated_response" if authenticated else "transport_failure"
            ),
            http_status=200 if authenticated else None,
            response_hash=sha256_bytes(body) if authenticated else None,
            request_artifact_hash=sha256_json({"request": len(self.calls)}),
            response_artifact_hash=sha256_bytes(body) if authenticated else None,
            tls_peer_chain_hash=HASH if authenticated else None,
            tls_protocol="TLSv1.3" if authenticated else None,
            failure_code=None if authenticated else "timeout",
            completed_at="2026-07-10T00:00:00Z",
        )
        result = {
            "terminal_status": attempt["terminal_status"],
            "transport_attempt": attempt,
        }
        if authenticated:
            result.update(
                {
                    "http_status": 200,
                    "body_b64": base64.b64encode(body).decode(),
                }
            )
        else:
            result["failure_code"] = "timeout"
        return result


def test_measured_legacy_settlement_source_binds_all_evidence(monkeypatch):
    reader = FakeReader()
    chain = FakeChainSource()
    provider = FakeArweave(fail_first=True)
    sleeps = []
    captured = {}

    def verify(**kwargs):
        captured.update(kwargs)
        return {
            "schema_version": "leadpoet.legacy_finalized_allocation.v2",
            "settlement_hash": "sha256:" + "9" * 64,
        }

    monkeypatch.setattr(
        source_module,
        "validate_legacy_finalized_settlement_v2",
        verify,
    )
    monkeypatch.setattr(
        source_module,
        "legacy_chain_vector_matches_bundle_v2",
        lambda **_kwargs: True,
    )
    source = CoordinatorLegacySettlementSourceV2(
        reader=reader,
        chain_source=chain,
        execute_provider=provider,
        retry_policy_hash="sha256:" + "8" * 64,
        sleep=sleeps.append,
    )
    context = ExecutionContextV2(
        job_id="legacy-settlement:100",
        purpose="research_lab.legacy_finalized_allocation.v2",
        epoch_id=101,
    )
    result = source.resolve(
        payload={
            "schema_version": LEGACY_SETTLEMENT_REQUEST_SCHEMA_VERSION,
            "netuid": 71,
            "epoch_id": 100,
        },
        context=context,
    )

    assert result == {
        "schema_version": "leadpoet.legacy_finalized_allocation.v2",
        "settlement_hash": "sha256:" + "9" * 64,
    }
    assert [item[0] for item in reader.calls] == [
        "legacy_audit_anchor_by_epoch",
        "legacy_allocation_by_hash",
        "legacy_transparency_event_by_hash",
        "legacy_weight_bundles_by_epoch",
    ]
    assert reader.calls[1][1]["allocation_hash"] == "sha256:" + "1" * 64
    assert captured["weight_bundle"]["validator_hotkey"] == "validator-1"
    assert captured["chain_evidence"] == {"chain": "evidence"}
    assert chain.calls[0]["epoch_id"] == 100
    assert len(provider.calls) == 2
    assert provider.calls[1]["url"] == "https://arweave.net/tx/" + "A" * 43 + "/data"
    assert sleeps == [1.0]
    assert len(context.transport_attempts) == 2


def test_nonfinalized_classification_does_not_require_arweave(monkeypatch):
    reader = FakeReader()
    chain = FakeChainSource()
    provider = FakeArweave()
    finding = {
        "schema_version": "leadpoet.legacy_allocation_nonfinalization.v2",
        "finding_hash": "sha256:" + "f" * 64,
    }
    monkeypatch.setattr(
        source_module,
        "legacy_chain_vector_matches_bundle_v2",
        lambda **_kwargs: False,
    )
    monkeypatch.setattr(
        source_module,
        "validate_legacy_allocation_nonfinalization_v2",
        lambda **_kwargs: finding,
    )
    source = CoordinatorLegacySettlementSourceV2(
        reader=reader,
        chain_source=chain,
        execute_provider=provider,
        retry_policy_hash="sha256:" + "8" * 64,
    )

    result = source.resolve_classification(
        payload={
            "schema_version": LEGACY_SETTLEMENT_REQUEST_SCHEMA_VERSION,
            "netuid": 71,
            "epoch_id": 100,
        },
        context=ExecutionContextV2(
            job_id="legacy-classification:100",
            purpose="research_lab.legacy_finalized_allocation.v2",
            epoch_id=101,
        ),
    )

    assert result == finding
    assert provider.calls == []
