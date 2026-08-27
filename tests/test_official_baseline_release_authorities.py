from __future__ import annotations

from io import BytesIO
import json

import pytest

from gateway.research_lab.official_baseline_authority import (
    PROTECTED_ACTION_AUTHORITY_SHA256,
    OfficialBaselineProtectedAuthorityError,
)
from gateway.research_lab.official_baseline_custody import (
    S3OfficialBaselineDocumentCustody,
)
from gateway.research_lab.official_baseline_model_runner import (
    OfficialBaselineAuthorityUnavailable,
)
from gateway.research_lab.official_baseline_release_authorities import (
    ArtifactPreparedActionExecutor,
    OFFICIAL_BINDING_CATALOG_SCHEMA_VERSION,
    SITE_PROTECTED_ACTION_AUTHORITY_SOURCE_CALLABLE,
    SITE_PROTECTED_ACTION_AUTHORITY_SOURCE_COMMIT,
    _GatewayEvidenceProxyClient,
    _catalog_bindings,
    _official_host_availability,
    _proxy_base_url,
    protected_action_authority_contract_identity,
)
from research_lab.canonical import sha256_bytes, sha256_json
from research_lab.model_runner_protocol import ExactModelRunnerRegistration


class NoSuchKey(Exception):
    pass


class _S3:
    def __init__(self) -> None:
        self.objects: dict[str, dict] = {}

    def get_object(self, *, Bucket, Key):
        del Bucket
        if Key not in self.objects:
            raise NoSuchKey(Key)
        item = self.objects[Key]
        return {
            "Body": BytesIO(item["Body"]),
            "Metadata": dict(item["Metadata"]),
            "ServerSideEncryption": item["ServerSideEncryption"],
        }

    def put_object(self, *, Bucket, Key, Body, **kwargs):
        del Bucket
        if kwargs.get("IfNoneMatch") == "*" and Key in self.objects:
            raise RuntimeError("precondition failed")
        self.objects[Key] = {
            "Body": bytes(Body),
            "Metadata": dict(kwargs["Metadata"]),
            "ServerSideEncryption": kwargs["ServerSideEncryption"],
        }
        return {}


def _custody() -> S3OfficialBaselineDocumentCustody:
    return S3OfficialBaselineDocumentCustody(
        client=_S3(),
        bucket="fixture-bucket",
        prefix="official-baseline",
        kms_key_id="alias/fixture",
    )


def _hash(value) -> str:
    return sha256_json(value).removeprefix("sha256:")


def _catalog_row(
    *,
    action_type: str,
    tool_id: str,
    binding: str,
    response_schema: str,
) -> dict:
    return {
        "schema_version": "host-action-binding:v1",
        "action_type": action_type,
        "tool_id": tool_id,
        "binding_contract_sha256": binding,
        "response_schema_version": response_schema,
        "idempotency": "idempotent",
        "max_response_bytes": 200_000,
    }


def _catalog(*rows: dict) -> dict:
    bindings = sorted(rows, key=lambda item: (item["action_type"], item["tool_id"]))
    body = {
        "schema_version": OFFICIAL_BINDING_CATALOG_SCHEMA_VERSION,
        "bindings": bindings,
        "binding_contracts_sha256": _hash(bindings),
    }
    return {**body, "catalog_sha256": _hash(body)}


def _inventory(*rows: dict) -> dict:
    entries = sorted(rows, key=lambda item: (item["action_type"], item["tool_id"]))
    body = {
        "schema_version": "model-runner-provider-compiler-inventory:v1",
        "dispatch_schema_version": "model-runner-provider-dispatch:v1",
        "entries": entries,
        "entries_sha256": _hash(entries),
    }
    return {**body, "inventory_sha256": _hash(body)}


class _Protocol:
    def __init__(
        self,
        dispatch: dict | None = None,
        verifier: dict | None = None,
        *,
        current: bool = False,
    ):
        self.dispatch = dispatch
        self.verifier = verifier
        self.requires_raw_provider_response_custody = current
        self.binding_inputs = []

    def prepare_provider_request(self, action):
        assert self.dispatch is not None
        assert self.dispatch["action_sha256"] == action["action_sha256"]
        return dict(self.dispatch)

    def build_provider_receipt_binding(self, action, result):
        assert action["action_sha256"]
        assert result.provider_receipt_ref
        self.binding_inputs.append(result)
        if self.requires_raw_provider_response_custody:
            assert result.model_provider_response_ingestion is not None
        binding = {
            "provider_receipt_ref": result.provider_receipt_ref,
            "provider_identity_sha256": result.provider_identity_sha256,
            "receipt_sha256": _hash(
                {
                    "action_sha256": action["action_sha256"],
                    "provider_receipt_ref": result.provider_receipt_ref,
                    "provider_identity_sha256": result.provider_identity_sha256,
                    "provider_response": result.provider_response,
                    "calls": result.calls,
                    "cost_credits": result.cost_credits,
                    "latency_ms": result.latency_ms,
                }
            ),
        }
        if result.provider_response is not None:
            binding["provider_response_sha256"] = self.ingest_provider_response(
                action,
                result.provider_response,
            )["parsed_response_sha256"]
        return binding

    def ingest_provider_response(self, action, host_response):
        assert self.dispatch is not None
        parsed = {
            "schema_version": "model-provider-response:v3",
            "records": [],
            "freshness_context": {},
            "extensions": {},
            "records_sha256": _hash([]),
        }
        body = {
            "schema_version": "model-runner-provider-response-ingestion:v1",
            "action_sha256": action["action_sha256"],
            "dispatch_sha256": self.dispatch["dispatch_sha256"],
            "compiler_id": self.dispatch["compiler_id"],
            "compiler_contract_sha256": self.dispatch[
                "compiler_contract_sha256"
            ],
            "request_sha256": self.dispatch["request_sha256"],
            "host_response_schema_version": "host-provider-response:v1",
            "host_response_sha256": _hash(host_response),
            "provider": self.dispatch["provider"],
            "parsed_response_schema_version": "model-provider-response:v3",
            "parsed_response": parsed,
            "parsed_response_sha256": _hash(parsed),
        }
        return {**body, "ingestion_sha256": _hash(body)}

    def execute_verifier_action(self, action):
        assert self.verifier is not None
        assert self.verifier["action_sha256"] == action["action_sha256"]
        return dict(self.verifier)


class _Proxy:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def request(self, **kwargs):
        self.calls.append(dict(kwargs))
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


def _registration(protocol) -> ExactModelRunnerRegistration:
    return ExactModelRunnerRegistration(
        artifact_identity={
            "repository": "leadpoet/Sourcing_model",
            "branch": "leadpoet-lab",
            "commit_sha": "a" * 40,
        },
        protocol=protocol,
        host_capability_manifest={},
    )


def _provider_fixture():
    binding = "b" * 64
    tool_id = "candidate.deepline_firmographic"
    action = {
        "sequence": 1,
        "action_type": "execute_candidate_tool",
        "tool_id": tool_id,
        "binding_contract_sha256": binding,
        "response_schema_version": "model-provider-response:v2",
        "max_response_bytes": 200_000,
        "idempotency_key": "1" * 64,
        "action_sha256": "2" * 64,
        "request_fingerprint_sha256": "3" * 64,
    }
    request = {
        "method": "POST",
        "url": "https://code.deepline.com/api/v2/plays/run",
        "static_headers": {"Content-Type": "application/json"},
        "credential_binding": {
            "location": "header",
            "name": "Authorization",
            "scheme": "Bearer",
            "source": "DEEPLINE_API_KEY",
            "persist": False,
        },
        "body": {"name": "fixture", "input": {"schema_version": 3}},
        "reconciliation": {
            "run_id_json_pointer": "/run/id",
            "primary_poll": {
                "method": "GET",
                "url_template": "https://code.deepline.com/api/v2/runs/{run_id}?full=true",
            },
            "fallback_poll": {
                "method": "GET",
                "url_template": "https://code.deepline.com/api/v2/plays/run/{run_id}?full=true",
            },
            "terminal_statuses": ["completed", "failed", "cancelled"],
            "timeout_behavior": "persist_run_id_and_resume_poll_without_repost",
        },
    }
    dispatch_body = {
        "schema_version": "model-runner-provider-dispatch:v1",
        "action_sha256": action["action_sha256"],
        "action_type": action["action_type"],
        "tool_id": action["tool_id"],
        "compiler_id": "deepline.firmographic_passthrough:v1",
        "compiler_contract_sha256": binding,
        "provider": "deepline",
        "request": request,
        "request_sha256": _hash(request),
        "response_contract": {
            "schema_version": "provider-response-contract:v1"
        },
        "budgets": {
            "call_cap": 1,
            "credit_cap": 0.02,
            "timeout_seconds": 900.0,
            "max_results": 100,
            "max_response_bytes": 200_000,
        },
        "idempotency_key": "model-action:" + action["action_sha256"],
    }
    dispatch = {**dispatch_body, "dispatch_sha256": _hash(dispatch_body)}
    catalog = _catalog(
        _catalog_row(
            action_type=action["action_type"],
            tool_id=tool_id,
            binding=binding,
            response_schema=action["response_schema_version"],
        )
    )
    inventory = _inventory(
        {
            "action_type": action["action_type"],
            "tool_id": tool_id,
            "status": "supported",
            "execution_mode": "invoke",
            "compiler_id": dispatch["compiler_id"],
            "provider": dispatch["provider"],
            "compiler_contract_sha256": binding,
            "timeout_seconds": 900.0,
        }
    )
    return action, dispatch, catalog, inventory


def test_deepline_progress_survives_interruption_and_restart_never_reposts():
    action, dispatch, catalog, inventory = _provider_fixture()
    protocol = _Protocol(dispatch=dispatch, current=True)
    custody = _custody()
    interrupted = _Proxy(
        [
            (
                200,
                {"run": {"id": "run-fixture", "status": "running"}},
                {},
            ),
            OfficialBaselineProtectedAuthorityError("poll interrupted"),
        ]
    )
    executor = ArtifactPreparedActionExecutor(
        registration=_registration(protocol),
        catalog=catalog,
        inventory=inventory,
        custody=custody,
        proxy_url="http://127.0.0.1:8765",
        proxy_client=interrupted,
        sleep=lambda _seconds: None,
    )
    run_identity = {"schema_version": "fixture-run:v1", "run": "one"}
    unit_ref = "baseline_icp:" + "4" * 64
    preparation = executor.prepare(
        run_identity=run_identity, unit_ref=unit_ref, action=action
    )

    with pytest.raises(
        OfficialBaselineProtectedAuthorityError, match="poll interrupted"
    ):
        executor.execute_prepared(preparation=preparation, action=action)

    progress = custody.load_protected_action_progress(
        preparation_sha256=preparation.preparation_sha256
    )
    assert progress["run_id"] == "run-fixture"
    assert [call["method"] for call in interrupted.calls] == ["POST", "GET"]

    restarted_proxy = _Proxy(
        [
            (
                200,
                {
                    "id": "run-fixture",
                    "status": "completed",
                    "output": {
                        "schema_version": 3,
                        "segment_id": "aggregate-fixture",
                        "rows": [],
                    },
                },
                {},
            )
        ]
    )
    restarted = ArtifactPreparedActionExecutor(
        registration=_registration(protocol),
        catalog=catalog,
        inventory=inventory,
        custody=custody,
        proxy_url="http://127.0.0.1:8765",
        proxy_client=restarted_proxy,
        sleep=lambda _seconds: None,
    )

    terminal = restarted.reconcile(preparation=preparation, action=action)

    assert terminal.state == "known"
    assert terminal.protected_action_result.host_result.outcome == "succeeded"
    assert [call["method"] for call in restarted_proxy.calls] == ["GET"]
    assert terminal.protected_action_result.provider_receipt.call_count == 1
    assert terminal.protected_action_result.replay_ref["provider_request_ref"]
    assert terminal.protected_action_result.host_result.provider_response == {
        "schema_version": "host-provider-response:v1",
        "provider": "deepline",
        "status_code": 200,
        "body": {
            "id": "run-fixture",
            "status": "completed",
            "output": {
                "schema_version": 3,
                "segment_id": "aggregate-fixture",
                "rows": [],
            },
        },
    }
    assert terminal.protected_action_result.model_provider_response_ingestion[
        "schema_version"
    ] == "model-runner-provider-response-ingestion:v1"
    assert protocol.binding_inputs[-1].model_provider_response_ingestion == (
        terminal.protected_action_result.model_provider_response_ingestion
    )
    assert (
        terminal.protected_action_result.host_result
        .model_provider_response_ingestion
        is None
    )


def test_deepline_missing_model_owned_run_id_path_fails_closed_before_poll():
    action, dispatch, catalog, inventory = _provider_fixture()
    protocol = _Protocol(dispatch=dispatch, current=True)
    custody = _custody()
    proxy = _Proxy(
        [(200, {"id": "legacy-top-level", "status": "running"}, {})]
    )
    executor = ArtifactPreparedActionExecutor(
        registration=_registration(protocol),
        catalog=catalog,
        inventory=inventory,
        custody=custody,
        proxy_url="http://127.0.0.1:8765",
        proxy_client=proxy,
    )
    preparation = executor.prepare(
        run_identity={"run": "missing-nested-id"},
        unit_ref="baseline_icp:" + "9" * 64,
        action=action,
    )

    with pytest.raises(
        OfficialBaselineProtectedAuthorityError,
        match="run id is unavailable",
    ):
        executor.execute_prepared(preparation=preparation, action=action)

    assert [call["method"] for call in proxy.calls] == ["POST"]
    assert custody.load_protected_action_progress(
        preparation_sha256=preparation.preparation_sha256
    ) is None


def test_deepline_tampered_progress_fails_closed_without_network():
    action, dispatch, catalog, inventory = _provider_fixture()
    protocol = _Protocol(dispatch=dispatch)
    custody = _custody()
    executor = ArtifactPreparedActionExecutor(
        registration=_registration(protocol),
        catalog=catalog,
        inventory=inventory,
        custody=custody,
        proxy_url="http://127.0.0.1:8765",
        proxy_client=_Proxy([]),
    )
    preparation = executor.prepare(
        run_identity={"run": "two"},
        unit_ref="baseline_icp:" + "5" * 64,
        action=action,
    )
    expected = executor._progress_document(
        preparation=preparation, dispatch=dispatch, run_id="run-fixture"
    )
    custody.append_protected_action_progress(
        preparation_sha256=preparation.preparation_sha256,
        progress={**expected, "provider_run_ref": "deepline_run:other"},
    )

    with pytest.raises(
        OfficialBaselineProtectedAuthorityError, match="progress differs"
    ):
        executor.reconcile(preparation=preparation, action=action)


def test_artifact_verifier_result_is_zero_call_known_terminal():
    binding = "c" * 64
    action = {
        "sequence": 2,
        "action_type": "verify_company",
        "tool_id": "verifier.company",
        "binding_contract_sha256": binding,
        "response_schema_version": "company-verifier-result:v2",
        "max_response_bytes": 200_000,
        "idempotency_key": "6" * 64,
        "action_sha256": "7" * 64,
        "request_fingerprint_sha256": "8" * 64,
    }
    result = {
        "schema_version": "company-verifier-result:v2",
        "accepted": False,
        "reason_code": "company_constraints_not_proven",
    }
    execution_body = {
        "schema_version": "model-runner-verifier-execution:v1",
        "action_sha256": action["action_sha256"],
        "action_type": action["action_type"],
        "calls": 0,
        "cost_credits": 0.0,
        "provider_receipt_allowed": False,
        "result": result,
        "result_sha256": _hash(result),
    }
    execution = {
        **execution_body,
        "execution_sha256": _hash(execution_body),
    }
    protocol = _Protocol(verifier=execution)
    catalog = _catalog(
        _catalog_row(
            action_type=action["action_type"],
            tool_id=action["tool_id"],
            binding=binding,
            response_schema=action["response_schema_version"],
        )
    )
    inventory = _inventory(
        {
            "action_type": action["action_type"],
            "tool_id": action["tool_id"],
            "status": "virtual",
            "compiler_contract_sha256": "0" * 64,
            "timeout_seconds": 30.0,
        }
    )
    executor = ArtifactPreparedActionExecutor(
        registration=_registration(protocol),
        catalog=catalog,
        inventory=inventory,
        custody=_custody(),
        proxy_url="http://127.0.0.1:8765",
        proxy_client=_Proxy([]),
    )
    preparation = executor.prepare(
        run_identity={"run": "three"},
        unit_ref="baseline_icp:" + "9" * 64,
        action=action,
    )

    terminal = executor.execute_prepared(preparation=preparation, action=action)

    host = terminal.protected_action_result.host_result
    assert terminal.state == "known"
    assert host.outcome == "succeeded"
    assert host.reason_code == "company_constraints_not_proven"
    assert host.provider_response == result
    assert host.calls == 0
    assert host.cost_credits == 0.0
    assert terminal.protected_action_result.provider_receipt is None
    assert terminal.provider_request_ref is None


def test_catalog_hash_and_order_tampering_fail_closed():
    first = _catalog_row(
        action_type="verify_intent",
        tool_id="verifier.intent",
        binding="d" * 64,
        response_schema="intent-verifier-response:v1",
    )
    second = _catalog_row(
        action_type="verify_company",
        tool_id="verifier.company",
        binding="e" * 64,
        response_schema="company-verifier-result:v2",
    )
    valid = _catalog(first, second)
    with pytest.raises(Exception, match="ordering differs"):
        _catalog_bindings({**valid, "bindings": list(reversed(valid["bindings"]))})
    with pytest.raises(Exception, match="hash differs"):
        _catalog_bindings({**valid, "catalog_sha256": "0" * 64})


def test_exact_site_protected_action_authority_document_is_locally_hash_bound():
    identity = protected_action_authority_contract_identity()
    body = dict(identity)
    claimed = body.pop("contract_sha256")

    assert SITE_PROTECTED_ACTION_AUTHORITY_SOURCE_COMMIT == (
        "f705fe57b61ea81188c42f3d2a0f04b310a33cd8"
    )
    assert SITE_PROTECTED_ACTION_AUTHORITY_SOURCE_CALLABLE == (
        "sourcing-worker/leadpoet_sourcing_worker/"
        "site_model_action_authority.py:"
        "protected_action_authority_contract_identity"
    )
    assert "sha256:" + claimed == PROTECTED_ACTION_AUTHORITY_SHA256
    assert sha256_json(body) == PROTECTED_ACTION_AUTHORITY_SHA256
    assert body["service_operations"] == [
        "prepare",
        "execute_prepared",
        "reconcile",
    ]
    assert body["authority_context"] == {
        "owner": "host_runtime",
        "implementation": "host_specific_durable_authority",
        "static_site_run_schema_required": False,
        "credential_free_model_action": True,
        "raw_credentials_in_durable_preparation": False,
        "durable_credential_identity": "one_way_hash_only",
    }


def test_host_availability_requires_artifact_provider_and_reviewed_proxy_route():
    binding = "f" * 64
    tool_id = "candidate.future_provider"
    catalog = _catalog(
        _catalog_row(
            action_type="execute_candidate_tool",
            tool_id=tool_id,
            binding=binding,
            response_schema="model-provider-response:v2",
        )
    )

    def inventory(provider: str) -> dict:
        return _inventory(
            {
                "action_type": "execute_candidate_tool",
                "tool_id": tool_id,
                "status": "supported",
                "execution_mode": "invoke",
                "compiler_id": "future.compiler:v1",
                "provider": provider,
                "compiler_contract_sha256": binding,
                "timeout_seconds": 30.0,
            }
        )

    assert _official_host_availability(
        catalog=catalog,
        inventory=inventory("openrouter"),
        ready_provider_ids=("or",),
    ) == {tool_id: True}
    assert _official_host_availability(
        catalog=catalog,
        inventory=inventory("unreviewed_provider"),
        ready_provider_ids=("unreviewed_provider",),
    ) == {tool_id: False}
    assert _official_host_availability(
        catalog=catalog,
        inventory=inventory("openrouter"),
        ready_provider_ids=(),
    ) == {tool_id: False}


@pytest.mark.parametrize(
    "value",
    (
        "https://127.0.0.1:8765",
        "http://localhost:8765",
        "http://10.0.0.1:8765",
        "http://127.0.0.1:8765/path",
        "http://127.0.0.1:8765?query=1",
        "http://user@127.0.0.1:8765",
    ),
)
def test_official_proxy_url_is_loopback_http_only(value):
    with pytest.raises(
        OfficialBaselineAuthorityUnavailable,
        match="evidence proxy URL is invalid",
    ):
        _proxy_base_url(value)


def test_official_proxy_url_accepts_bound_ipv4_and_ipv6_loopback():
    assert _proxy_base_url("http://127.0.0.1:8765/") == (
        "http://127.0.0.1:8765"
    )
    assert _proxy_base_url("http://[::1]:8765") == "http://[::1]:8765"


def test_custody_objects_are_sse_kms_and_content_hash_bound():
    custody = _custody()
    progress = {
        "schema_version": "fixture-progress:v1",
        "preparation_sha256": "sha256:" + "a" * 64,
    }
    assert custody.append_protected_action_progress(
        preparation_sha256=progress["preparation_sha256"], progress=progress
    )
    key, stored = next(iter(custody._client.objects.items()))
    assert key.endswith("/progress.json")
    assert stored["ServerSideEncryption"] == "aws:kms"
    assert stored["Metadata"]["content-sha256"] == sha256_bytes(
        stored["Body"]
    ).removeprefix("sha256:")
    assert json.loads(stored["Body"]) == progress


def test_exa_transport_uses_only_reviewed_evidence_proxy_route():
    captured = {}

    class _Response:
        status = 200
        headers = {}

        def read(self, _limit):
            return b'{"results":[]}'

        def close(self):
            captured["closed"] = True

    def opener(request, *, timeout):
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["body"] = bytes(request.data)
        captured["timeout"] = timeout
        return _Response()

    client = _GatewayEvidenceProxyClient(
        proxy_url="http://127.0.0.1:8765", opener=opener
    )
    status, body, _headers = client.request(
        provider="exa",
        method="POST",
        upstream_url="https://api.exa.ai/search",
        static_headers={"Content-Type": "application/json"},
        body={"query": "redacted fixture"},
        query=None,
        timeout_seconds=10,
        max_response_bytes=10_000,
        cost_scope="official-baseline-fixture",
    )

    assert status == 200
    assert body == {"results": []}
    assert captured["url"] == "http://127.0.0.1:8765/exa/search"
    assert json.loads(captured["body"]) == {"query": "redacted fixture"}
    lowered_headers = {key.casefold(): value for key, value in captured["headers"].items()}
    assert "authorization" not in lowered_headers
    assert "x-api-key" not in lowered_headers
    assert captured["closed"] is True

    with pytest.raises(
        OfficialBaselineProtectedAuthorityError,
        match="outside the evidence proxy",
    ):
        client.request(
            provider="exa",
            method="POST",
            upstream_url="https://example.com/search",
            static_headers={},
            body={},
            query=None,
            timeout_seconds=10,
            max_response_bytes=10_000,
            cost_scope="official-baseline-fixture",
        )
