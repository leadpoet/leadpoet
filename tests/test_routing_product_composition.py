from __future__ import annotations

import base64
import hashlib
import json
from types import SimpleNamespace

import pytest

from gateway.research_lab.routing_experiment_api import (
    RoutingExperimentApiService,
    install_routing_experiment_api_service,
)
from gateway.research_lab.routing_execution_consumer import (
    REVIEWED_ROUTING_FACTORY_NAME,
    RoutingExecutionConsumerError,
    install_reviewed_routing_factory_registry,
)
from gateway.research_lab.routing_authority_bundle import (
    VerifiedRoutingAuthorityBundle,
)
from gateway.research_lab.routing_product_composition import (
    AUTHORITY_BUNDLE_HASH_ENV,
    BINDING_CATALOG_MANIFEST_HASH_ENV,
    MODEL_ROUTING_CATALOG_HASH_ENV,
    PROTECTED_RELEASE_BOOT_IDENTITY_HASH_ENV,
    PROTECTED_RELEASE_BUILD_MANIFEST_HASH_ENV,
    PROTECTED_RELEASE_COMMIT_SHA_ENV,
    PROTECTED_RELEASE_CONFIG_HASH_ENV,
    PROTECTED_RELEASE_DEPENDENCY_LOCK_HASH_ENV,
    PROTECTED_RELEASE_ENCLAVE_PUBKEY_ENV,
    PROTECTED_RELEASE_PCR0_ENV,
    PROTECTED_RELEASE_RECEIPT_HASH_ENV,
    PRODUCT_COMPOSITION_ENV,
    PRODUCT_COMPOSITION_VERSION,
    RoutingProductCompositionError,
    ReviewedRoutingReleaseInputs,
    _TeeJobRpcOperationExecutor,
    RoutingProviderDispatchTeeRpc,
    ROUTING_PROVIDER_AUTHORIZATION_OPERATION_V2,
    ROUTING_PROVIDER_AUTHORIZATION_PURPOSE_V2,
    ROUTING_PROVIDER_DISPATCH_OPERATION_V2,
    ROUTING_PROVIDER_DISPATCH_PURPOSE_V2,
    build_attested_provider_broker_factory,
    build_attested_protected_authorities,
    build_reviewed_admission_authority,
    build_reviewed_routing_product,
    bootstrap_reviewed_routing_product,
)
from gateway.research_lab.routing_experiment_runtime import (
    AttestedScoringV2RoutingProviderCallAuthority,
    AttestedScoringV2RoutingProviderDispatchAuthority,
    ReviewedProviderBrokerRoutingRunner,
    RoutingExperimentRuntimeConfig,
    RoutingExperimentRuntimeError,
    _ROUTING_DISPATCH_EXECUTOR_TOKEN,
    _validate_redacted_routing_dispatch_result,
)
from gateway.research_lab.routing_execution_authorization import (
    execute_routing_provider_call_authorization_v2,
)
from gateway.research_lab.routing_execution_envelope import (
    RoutingExperimentExecutionEnvelopeV2,
)
from gateway.research_lab.routing_provider_bindings import (
    VerifiedRoutingBindingCatalog,
    VerifiedRoutingUnitDataset,
)
from gateway.research_lab.routing_provider_terminal_protected import (
    routing_provider_dispatch_receipt_output_v2,
)
from research_lab.model_runner_protocol import ExactModelRunnerRegistry
from gateway.tee.execution_job_manager_v2 import (
    JOB_SCHEMA_VERSION,
    ExecutionResultV2,
)
from tests.routing_experiment_authority_fixture import authority_fixture
from research_lab.canonical import sha256_json


COMMIT = "a" * 40
MODEL_CATALOG = "sha256:" + "b" * 64
CONTRACT = "sha256:" + "c" * 64
BINDING_CATALOG = "sha256:" + "d" * 64
BUNDLE = "sha256:" + "e" * 64
UNIT_MANIFEST = "sha256:" + "f" * 64
UNIT_SET = "sha256:" + "1" * 64
_AUTHORITY_FIXTURE = authority_fixture()
_PROTECTED_RELEASE_RECEIPT = dict(
    _AUTHORITY_FIXTURE["attempts"][0]["attempt_doc"][
        "protected_release_receipt"
    ]
)


class _Rpc:
    def __init__(self):
        self.manifest = None

    def submit_job(self, manifest):
        self.manifest = dict(manifest)
        return {}

    def put_chunk(self, **_kwargs):
        return {}

    def seal(self, _job_id):
        return {}

    def status(self, _job_id):
        return {}

    def result(self, _job_id):
        return {}

    def receipts(self, _job_id):
        return ()


class _OperationRpc:
    def __init__(self, *, summary_job_substitution=False, result_job_substitution=False):
        self.summary_job_substitution = summary_job_substitution
        self.result_job_substitution = result_job_substitution
        self.manifest = None
        self.body = b""
        self.receipt = None

    def _summary(self, state, uploaded):
        job_id = self.manifest["job_id"]
        if self.summary_job_substitution:
            job_id = "routing-job:substituted"
        return {
            "job_id": job_id,
            "operation": self.manifest["operation"],
            "purpose": self.manifest["purpose"],
            "manifest_hash": sha256_json(self.manifest),
            "payload_sha256": self.manifest["payload_sha256"],
            "expected_bytes": self.manifest["payload_size_bytes"],
            "uploaded_bytes": uploaded,
            "state": state,
        }

    def submit_job(self, manifest):
        self.manifest = dict(manifest)
        return self._summary("uploading", 0)

    def put_chunk(self, *, job_id, offset, data_b64, chunk_sha256):
        assert job_id == self.manifest["job_id"]
        assert offset == 0
        self.body = base64.b64decode(data_b64)
        assert chunk_sha256 == "sha256:" + hashlib.sha256(self.body).hexdigest()
        return self._summary("uploading", len(self.body))

    def seal(self, job_id):
        assert job_id == self.manifest["job_id"]
        return self._summary("queued", len(self.body))

    def status(self, job_id):
        assert job_id == self.manifest["job_id"]
        return self._summary("succeeded", len(self.body))

    def result(self, job_id):
        payload = json.loads(self.body)
        result_payload = {"accepted": True}
        receipt_hash = "sha256:" + "9" * 64
        self.receipt = {
            "job_id": job_id,
            "purpose": self.manifest["purpose"],
            "status": "succeeded",
            "input_root": sha256_json(payload),
            "output_root": sha256_json(result_payload),
            "parent_receipt_hashes": self.manifest["parent_receipt_hashes"],
            "receipt_hash": receipt_hash,
        }
        return {
            "job_id": (
                "routing-job:substituted"
                if self.result_job_substitution
                else job_id
            ),
            "operation": self.manifest["operation"],
            "purpose": self.manifest["purpose"],
            "state": "succeeded",
            "result": result_payload,
            "execution_receipt": dict(self.receipt),
        }

    def receipts(self, _job_id):
        return (dict(self.receipt),)


class _ExecutionManagerRpc:
    """Adapt the reviewed operation manifest to the V2 manager contract."""

    def __init__(self, manager):
        self.manager = manager
        self.manifest = None
        self.submit_calls = 0

    def _summary(self, summary):
        value = dict(summary)
        value["manifest_hash"] = sha256_json(self.manifest)
        value["payload_sha256"] = self.manifest["payload_sha256"]
        value["expected_bytes"] = self.manifest["payload_size_bytes"]
        return value

    def submit_job(self, manifest):
        self.submit_calls += 1
        self.manifest = dict(manifest)
        manager_manifest = {
            "schema_version": JOB_SCHEMA_VERSION,
            "job_id": self.manifest["job_id"],
            "operation": self.manifest["operation"],
            "purpose": self.manifest["purpose"],
            "epoch_id": 24_300,
            "sequence": 1,
            "payload_sha256": self.manifest["payload_sha256"],
            "payload_size_bytes": self.manifest["payload_size_bytes"],
            "parent_receipt_hashes": list(
                self.manifest["parent_receipt_hashes"]
            ),
            "input_artifact_hashes": [],
            "provider_credential_profile": "default",
            "provider_credential_ref_hashes": {},
        }
        return self._summary(self.manager.submit(manager_manifest))

    def put_chunk(self, **kwargs):
        return self._summary(self.manager.put_chunk(**kwargs))

    def seal(self, job_id):
        return self._summary(self.manager.seal(job_id))

    def status(self, job_id):
        return self._summary(self.manager.status(job_id))

    def result(self, job_id):
        chunk = self.manager.result_chunk(job_id=job_id)
        return {
            "job_id": job_id,
            "operation": self.manifest["operation"],
            "purpose": self.manifest["purpose"],
            "state": "succeeded",
            "result": json.loads(base64.b64decode(chunk["data_b64"])),
            "execution_receipt": self.manager.receipt(job_id),
        }

    def receipts(self, job_id):
        return self.manager.receipts(job_id)


class _ModelRunnerRegistry(ExactModelRunnerRegistry):
    def __init__(self):
        pass

    def preflight_all(self):
        return {"reviewed": {"preflight_sha256": "9" * 64}}

    def resolve(self, _artifact):
        return SimpleNamespace(
            key="reviewed",
            preflight=lambda: {"preflight_sha256": "9" * 64},
            host_capability_manifest={"bindings": []},
        )


class _ModelVerifier:
    def verify_company(self, **_kwargs):
        raise AssertionError("not called")

    def verify_intent(self, **_kwargs):
        raise AssertionError("not called")

    def verify_contact(self, **_kwargs):
        raise AssertionError("not called")


class _EvaluationAdapter:
    def build_decision_receipts(self, **_kwargs):
        return ()

    def build_evaluation(self, **_kwargs):
        raise AssertionError("not called")


class _ArtifactAuthority:
    def verify(self, **_kwargs):
        return {"verified": True}


class _DurableStore:
    def __init__(self, identity: str):
        self.identity = identity

    def durable_authority_identity(self):
        return self.identity


class _Lineage(SimpleNamespace):
    def identity_hash(self):
        return sha256_json(
            {
                "commit_sha": self.commit_sha,
                "routing_catalog_hash": self.routing_catalog_hash,
                "routing_contract_hash": self.routing_contract_hash,
            }
        )


def _inputs():
    lineage = _Lineage(
        commit_sha=COMMIT,
        routing_catalog_hash=MODEL_CATALOG,
        routing_contract_hash=CONTRACT,
    )
    binding_catalog = VerifiedRoutingBindingCatalog(
        manifest_uri="s3://reviewed/bindings.json",
        manifest_hash=BINDING_CATALOG,
        signature_ref="kms:bindings",
        signing_key_id="kms-key",
        catalog_version="test-v1",
        bindings={},
    )
    unit_dataset = VerifiedRoutingUnitDataset(
        manifest_uri="s3://reviewed/units.json",
        manifest_hash=UNIT_MANIFEST,
        signature_ref="kms:units",
        signing_key_id="kms-key",
        unit_set_hash=UNIT_SET,
        provenance_hash="sha256:" + "3" * 64,
        units={},
    )
    return ReviewedRoutingReleaseInputs(
        artifact_lineage=lineage,
        binding_catalog=binding_catalog,
        unit_dataset=unit_dataset,
        authority_bundle=VerifiedRoutingAuthorityBundle(
            artifact_lineage=lineage,
            binding_catalog=binding_catalog,
            unit_dataset=unit_dataset,
            bundle_hash=BUNDLE,
        ),
        gold_labels=SimpleNamespace(labels={}),
        model_binding_observation=SimpleNamespace(
            observation_receipt_hash="sha256:" + "2" * 64
        ),
        protected_release_receipt=dict(_PROTECTED_RELEASE_RECEIPT),
        artifact_authority=_ArtifactAuthority(),
        model_runner_registry=_ModelRunnerRegistry(),
        model_verifier=_ModelVerifier(),
        evaluation_adapter=_EvaluationAdapter(),
        scoring_job_rpc=_Rpc(),
        call_authorization_job_rpc=_Rpc(),
        dispatch_job_rpc=_Rpc(),
    )


def _env():
    return {
        "RESEARCH_LAB_ROUTING_PRODUCT_COMPOSITION": PRODUCT_COMPOSITION_VERSION,
        "RESEARCH_LAB_ROUTING_MODEL_COMMIT_SHA": COMMIT,
        MODEL_ROUTING_CATALOG_HASH_ENV: MODEL_CATALOG,
        BINDING_CATALOG_MANIFEST_HASH_ENV: BINDING_CATALOG,
        "RESEARCH_LAB_ROUTING_CONTRACT_HASH": CONTRACT,
        AUTHORITY_BUNDLE_HASH_ENV: BUNDLE,
        PROTECTED_RELEASE_RECEIPT_HASH_ENV: _PROTECTED_RELEASE_RECEIPT[
            "receipt_hash"
        ],
        PROTECTED_RELEASE_COMMIT_SHA_ENV: _PROTECTED_RELEASE_RECEIPT[
            "commit_sha"
        ],
        PROTECTED_RELEASE_PCR0_ENV: _PROTECTED_RELEASE_RECEIPT["pcr0"],
        PROTECTED_RELEASE_BUILD_MANIFEST_HASH_ENV: _PROTECTED_RELEASE_RECEIPT[
            "build_manifest_hash"
        ],
        PROTECTED_RELEASE_DEPENDENCY_LOCK_HASH_ENV: _PROTECTED_RELEASE_RECEIPT[
            "dependency_lock_hash"
        ],
        PROTECTED_RELEASE_CONFIG_HASH_ENV: _PROTECTED_RELEASE_RECEIPT[
            "config_hash"
        ],
        PROTECTED_RELEASE_BOOT_IDENTITY_HASH_ENV: _PROTECTED_RELEASE_RECEIPT[
            "boot_identity_hash"
        ],
        PROTECTED_RELEASE_ENCLAVE_PUBKEY_ENV: _PROTECTED_RELEASE_RECEIPT[
            "enclave_pubkey"
        ],
    }


def test_reviewed_gate_keeps_model_and_protected_release_identities_separate():
    assert _PROTECTED_RELEASE_RECEIPT["commit_sha"] != COMMIT
    authority = build_reviewed_admission_authority(_inputs(), environment=_env())
    assert authority.inputs.artifact_lineage.commit_sha == COMMIT


def test_reviewed_gate_rejects_missing_model_registry_before_provider_rpc():
    inputs = _inputs()
    inputs = inputs.__class__(
        **{
            **inputs.__dict__,
            "model_runner_registry": None,
        }
    )
    with pytest.raises(RoutingProductCompositionError, match="runner registry"):
        build_reviewed_admission_authority(inputs, environment=_env())


def test_reviewed_gate_rejects_model_catalog_substitution_before_provider_rpc():
    inputs = _inputs()
    with pytest.raises(RoutingProductCompositionError, match="model catalog"):
        build_reviewed_admission_authority(
            inputs,
            environment={
                **_env(),
                MODEL_ROUTING_CATALOG_HASH_ENV: "sha256:" + "2" * 64,
            },
        )


def test_reviewed_gate_rejects_binding_catalog_pin_substitution():
    with pytest.raises(RoutingProductCompositionError, match="binding catalog"):
        build_reviewed_admission_authority(
            _inputs(),
            environment={
                **_env(),
                BINDING_CATALOG_MANIFEST_HASH_ENV: "sha256:" + "2" * 64,
            },
        )


def test_reviewed_gate_rejects_authority_bundle_hash_substitution():
    with pytest.raises(RoutingProductCompositionError, match="bundle identity"):
        build_reviewed_admission_authority(
            _inputs(),
            environment={
                **_env(),
                AUTHORITY_BUNDLE_HASH_ENV: "sha256:" + "2" * 64,
            },
        )


def test_reviewed_gate_rejects_bundle_binding_catalog_substitution():
    inputs = _inputs()
    substituted_catalog = SimpleNamespace(
        manifest_hash=BINDING_CATALOG,
        substitution=True,
    )
    substituted_bundle = VerifiedRoutingAuthorityBundle(
        artifact_lineage=inputs.artifact_lineage,
        binding_catalog=substituted_catalog,
        unit_dataset=inputs.unit_dataset,
        bundle_hash=BUNDLE,
    )
    changed = inputs.__class__(
        **{**inputs.__dict__, "authority_bundle": substituted_bundle}
    )
    with pytest.raises(RoutingProductCompositionError, match="bundle binding catalog"):
        build_reviewed_admission_authority(changed, environment=_env())


def test_reviewed_gate_rejects_bundle_unit_dataset_substitution():
    inputs = _inputs()
    substituted_dataset = SimpleNamespace(
        manifest_hash=UNIT_MANIFEST,
        unit_set_hash=UNIT_SET,
        substitution=True,
    )
    substituted_bundle = VerifiedRoutingAuthorityBundle(
        artifact_lineage=inputs.artifact_lineage,
        binding_catalog=inputs.binding_catalog,
        unit_dataset=substituted_dataset,
        bundle_hash=BUNDLE,
    )
    changed = inputs.__class__(
        **{**inputs.__dict__, "authority_bundle": substituted_bundle}
    )
    with pytest.raises(RoutingProductCompositionError, match="unit dataset"):
        build_reviewed_admission_authority(changed, environment=_env())


def test_reviewed_gate_requires_all_typed_tee_job_rpc_methods():
    inputs = _inputs()
    inputs = inputs.__class__(
        **{
            **inputs.__dict__,
            "dispatch_job_rpc": object(),
        }
    )
    with pytest.raises(RoutingProductCompositionError, match="dispatch"):
        build_reviewed_admission_authority(inputs, environment=_env())


def test_bootstrap_constructs_all_protected_authorities_from_typed_rpc_clients():
    authorities = build_attested_protected_authorities(
        _inputs(), environment=_env()
    )
    assert authorities.model_binding_observation_issuer is not None
    assert authorities.call_authorization_authority is not None
    assert authorities.dispatch_authority is not None


def test_product_composition_binds_authorization_parent_manifest_before_manager_submit():
    inputs = _inputs()
    authorities = build_attested_protected_authorities(inputs, environment=_env())
    executor = authorities.call_authorization_authority._executor
    rpc = inputs.call_authorization_job_rpc
    expected_parents = [
        _PROTECTED_RELEASE_RECEIPT["receipt_hash"],
        inputs.model_binding_observation.observation_receipt_hash,
    ]

    with pytest.raises(RoutingProductCompositionError, match="parent ancestry"):
        executor(
            {
                "operation": ROUTING_PROVIDER_AUTHORIZATION_OPERATION_V2,
                "purpose": ROUTING_PROVIDER_AUTHORIZATION_PURPOSE_V2,
                "payload": {"value": "rejected"},
                "parent_receipt_hashes": list(reversed(expected_parents)),
            }
        )
    assert rpc.manifest is None

    with pytest.raises(RoutingProductCompositionError, match="job summary"):
        executor(
            {
                "operation": ROUTING_PROVIDER_AUTHORIZATION_OPERATION_V2,
                "purpose": ROUTING_PROVIDER_AUTHORIZATION_PURPOSE_V2,
                "payload": {"value": "accepted"},
                "parent_receipt_hashes": expected_parents,
            }
        )
    assert rpc.manifest["parent_receipt_hashes"] == expected_parents


def test_dispatch_executor_binds_dynamic_authorization_receipt_parent():
    receipt_hash = "sha256:" + "8" * 64
    rpc = _OperationRpc()
    executor = _TeeJobRpcOperationExecutor(
        RoutingProviderDispatchTeeRpc(rpc),
        allowed_operation=ROUTING_PROVIDER_DISPATCH_OPERATION_V2,
        allowed_purpose=ROUTING_PROVIDER_DISPATCH_PURPOSE_V2,
        required_parent_receipt_hashes_factory=(
            lambda request: (
                request["payload"]["authorization_proof"]["authorization_receipt"][
                    "receipt_hash"
                ],
            )
        ),
        dispatch_token=_ROUTING_DISPATCH_EXECUTOR_TOKEN,
    )
    response = executor(
        {
            "operation": ROUTING_PROVIDER_DISPATCH_OPERATION_V2,
            "purpose": ROUTING_PROVIDER_DISPATCH_PURPOSE_V2,
            "payload": {
                "authorization_proof": {
                    "authorization_receipt": {"receipt_hash": receipt_hash}
                }
            },
            "parent_receipt_hashes": [receipt_hash],
        }
    )
    assert response["status"] == "succeeded"
    assert rpc.manifest["parent_receipt_hashes"] == [receipt_hash]

    rpc = _OperationRpc()
    executor = _TeeJobRpcOperationExecutor(
        RoutingProviderDispatchTeeRpc(rpc),
        required_parent_receipt_hashes_factory=(
            lambda request: (
                request["payload"]["authorization_proof"]["authorization_receipt"][
                    "receipt_hash"
                ],
            )
        ),
        dispatch_token=_ROUTING_DISPATCH_EXECUTOR_TOKEN,
    )
    with pytest.raises(RoutingProductCompositionError, match="parent ancestry"):
        executor(
            {
                "operation": "routing_test_v2",
                "purpose": "research_lab.routing_test.v2",
                "payload": {
                    "authorization_proof": {
                        "authorization_receipt": {"receipt_hash": receipt_hash}
                    }
                },
                "parent_receipt_hashes": ["sha256:" + "9" * 64],
            }
        )
    assert rpc.manifest is None


def test_dispatch_executor_replays_signed_model_result_without_resubmit(
    monkeypatch,
):
    from gateway.tee import execution_job_manager_v2 as job_manager_v2
    from tests.test_execution_job_manager_v2 import _manager

    monkeypatch.setattr(
        job_manager_v2,
        "validate_receipt_graphs",
        lambda *args, **kwargs: None,
    )
    model_response = {
        "schema_version": "host-provider-response:v1",
        "provider": "deepline",
        "status_code": 200,
        "body": {
            "run": {"status": "completed"},
            "outputs": {
                "model_provider_records": [],
                "freshness_context": {},
            },
        },
    }
    output = {
        "model_completion_contract_hash": "sha256:" + "3" * 64,
        "model_provider_response_sha256": sha256_json(model_response),
        "model_provider_response": model_response,
    }

    def manager_executor(_operation, _payload, _context):
        return ExecutionResultV2(
            output=output,
            receipt_output=routing_provider_dispatch_receipt_output_v2(
                output
            ),
        )

    manager, _ = _manager(
        manager_executor,
        operations={
            ROUTING_PROVIDER_DISPATCH_OPERATION_V2: {
                ROUTING_PROVIDER_DISPATCH_PURPOSE_V2
            }
        },
    )
    rpc = _ExecutionManagerRpc(manager)
    executor = _TeeJobRpcOperationExecutor(
        RoutingProviderDispatchTeeRpc(rpc),
        allowed_operation=ROUTING_PROVIDER_DISPATCH_OPERATION_V2,
        allowed_purpose=ROUTING_PROVIDER_DISPATCH_PURPOSE_V2,
        dispatch_token=_ROUTING_DISPATCH_EXECUTOR_TOKEN,
    )
    first = executor(
        {
            "operation": ROUTING_PROVIDER_DISPATCH_OPERATION_V2,
            "purpose": ROUTING_PROVIDER_DISPATCH_PURPOSE_V2,
            "payload": {"exact": True},
            "job_id": "routing-dispatch:" + "1" * 32,
            "parent_receipt_hashes": [],
        }
    )
    replay_ref = {
        "schema_version": (
            "leadpoet.research_lab.protected_model_replay_ref.v1"
        ),
        "protected_dispatch_job_id": first["execution_receipt"]["job_id"],
        "terminal_receipt_hash": first["execution_receipt"]["receipt_hash"],
        "model_provider_response_sha256": output[
            "model_provider_response_sha256"
        ],
        "model_completion_contract_hash": output[
            "model_completion_contract_hash"
        ],
    }
    replayed = executor.replay_protected_model_result(replay_ref)

    assert replayed["result"] == output
    assert rpc.submit_calls == 1
    with pytest.raises(RoutingProductCompositionError, match="differs"):
        executor.replay_protected_model_result(
            {
                **replay_ref,
                "model_provider_response_sha256": "sha256:" + "4" * 64,
            }
        )


def test_authorize_composition_and_manager_preserve_authorization_parent_order(
    monkeypatch,
):
    from gateway.tee import execution_job_manager_v2 as job_manager_v2
    from tests.test_execution_job_manager_v2 import _manager
    from tests.test_routing_provider_authorization_context import _context

    context = _context()
    release = context["protected_receipt"]
    bundle_hash = "sha256:" + "1" * 64

    def manager_executor(_operation, payload, execution_context):
        result = execute_routing_provider_call_authorization_v2(
            payload["authorization"],
            authorization_job_id=execution_context.job_id,
        )
        receipt_output = dict(result)
        receipt_output.pop("output_root")
        return ExecutionResultV2(
            output=result,
            receipt_output=receipt_output,
        )

    monkeypatch.setattr(
        job_manager_v2,
        "validate_receipt_graphs",
        lambda *args, **kwargs: None,
    )
    manager, _ = _manager(
        manager_executor,
        operations={
            ROUTING_PROVIDER_AUTHORIZATION_OPERATION_V2: {
                ROUTING_PROVIDER_AUTHORIZATION_PURPOSE_V2
            }
        },
    )
    rpc = _ExecutionManagerRpc(manager)
    inputs = ReviewedRoutingReleaseInputs(
        artifact_lineage=context["lineage"],
        binding_catalog=context["catalog"],
        unit_dataset=context["unit_dataset"],
        authority_bundle=VerifiedRoutingAuthorityBundle(
            artifact_lineage=context["lineage"],
            binding_catalog=context["catalog"],
            unit_dataset=context["unit_dataset"],
            bundle_hash=bundle_hash,
        ),
        gold_labels=SimpleNamespace(labels={}),
        model_binding_observation=context["observation"],
        protected_release_receipt=release,
        artifact_authority=_ArtifactAuthority(),
        model_runner_registry=_ModelRunnerRegistry(),
        model_verifier=_ModelVerifier(),
        evaluation_adapter=_EvaluationAdapter(),
        scoring_job_rpc=_Rpc(),
        call_authorization_job_rpc=rpc,
        dispatch_job_rpc=_Rpc(),
    )
    environment = {
        PRODUCT_COMPOSITION_ENV: PRODUCT_COMPOSITION_VERSION,
        "RESEARCH_LAB_ROUTING_MODEL_COMMIT_SHA": context[
            "lineage"
        ].commit_sha,
        MODEL_ROUTING_CATALOG_HASH_ENV: context["lineage"].routing_catalog_hash,
        BINDING_CATALOG_MANIFEST_HASH_ENV: context["catalog"].manifest_hash,
        "RESEARCH_LAB_ROUTING_CONTRACT_HASH": context[
            "lineage"
        ].routing_contract_hash,
        AUTHORITY_BUNDLE_HASH_ENV: bundle_hash,
        PROTECTED_RELEASE_RECEIPT_HASH_ENV: release["receipt_hash"],
        PROTECTED_RELEASE_COMMIT_SHA_ENV: release["commit_sha"],
        PROTECTED_RELEASE_PCR0_ENV: release["pcr0"],
        PROTECTED_RELEASE_BUILD_MANIFEST_HASH_ENV: release[
            "build_manifest_hash"
        ],
        PROTECTED_RELEASE_DEPENDENCY_LOCK_HASH_ENV: release[
            "dependency_lock_hash"
        ],
        PROTECTED_RELEASE_CONFIG_HASH_ENV: release["config_hash"],
        PROTECTED_RELEASE_BOOT_IDENTITY_HASH_ENV: release[
            "boot_identity_hash"
        ],
        PROTECTED_RELEASE_ENCLAVE_PUBKEY_ENV: release["enclave_pubkey"],
    }
    authorities = build_attested_protected_authorities(
        inputs,
        environment=environment,
    )
    expected_parents = [
        release["receipt_hash"],
        context["observation"].observation_receipt_hash,
    ]
    proof = authorities.call_authorization_authority.authorize(
        context["grant"],
        artifact_lineage=context["lineage"],
        model_binding_observation=context["observation"],
        execution_envelope=context["envelope"],
        admission_bundle=context["admission"],
        prepared_call=context["prepared"],
        protected_release_receipt=release,
        parent_receipt_graphs=(
            {
                "root_receipt_hash": release["receipt_hash"],
                "receipts": [dict(release)],
            },
            {
                "root_receipt_hash": context["observation"].observation_receipt_hash,
                "receipts": [dict(context["observation"].signed_receipt)],
            },
        ),
    )

    assert rpc.manifest["parent_receipt_hashes"] == expected_parents
    assert proof["authorization_receipt"]["parent_receipt_hashes"] == expected_parents
    assert manager.receipt(proof["authorization_receipt"]["job_id"])[
        "parent_receipt_hashes"
    ] == expected_parents


def test_reviewed_product_composition_installs_api_and_one_factory():
    calls = []
    durable_identity = "sha256:" + "7" * 64
    durable_store = _DurableStore(durable_identity)
    protected_authorities = build_attested_protected_authorities(
        _inputs(), environment=_env()
    )

    def reviewed_runner_factory(spec):
        calls.append(spec)
        return ReviewedProviderBrokerRoutingRunner(
            config=RoutingExperimentRuntimeConfig(enabled=True),
            store=durable_store,
            artifact_lineage=object(),
            compiler=object(),
            model_binding_requirements=object(),
            authorization_authority=(
                protected_authorities.call_authorization_authority
            ),
            dispatch_authority=protected_authorities.dispatch_authority,
            authorization_parent_receipt_graphs=({"receipts": []},),
            dispatch_parent_receipt_graphs=({"receipts": []},),
        )

    reviewed_runner_factory.validate_readiness = lambda: None
    reviewed_runner_factory.durable_authority_identity = durable_identity

    envelope = RoutingExperimentExecutionEnvelopeV2.from_mapping(
        authority_fixture()["execution_envelope"]
    )
    composition = build_reviewed_routing_product(
        inputs=_inputs(),
        reviewed_runner_factory=reviewed_runner_factory,
        billing_rollup_factory=lambda _spec: lambda _store: {},
        execution_envelope_factory=lambda _spec: envelope,
        store_factory=lambda: durable_store,
        environment=_env(),
    )
    assert isinstance(composition.api_service, RoutingExperimentApiService)
    assert set(composition.factory_registry) == {REVIEWED_ROUTING_FACTORY_NAME}
    assert composition.api_service.store_factory() is durable_store
    assert composition.run_factory.name == REVIEWED_ROUTING_FACTORY_NAME
    assert calls == []


def test_reviewed_runner_requires_authorization_ancestry_before_queue_execution():
    protected_authorities = build_attested_protected_authorities(
        _inputs(), environment=_env()
    )
    runner = ReviewedProviderBrokerRoutingRunner(
        config=RoutingExperimentRuntimeConfig(enabled=True),
        store=object(),
        artifact_lineage=object(),
        compiler=object(),
        model_binding_requirements=object(),
        authorization_authority=protected_authorities.call_authorization_authority,
        dispatch_authority=protected_authorities.dispatch_authority,
        dispatch_parent_receipt_graphs=({"receipts": []},),
    )
    with pytest.raises(RoutingExperimentRuntimeError, match="authorization ancestry"):
        runner.validate_composition()


def test_bootstrap_without_exact_model_runner_fails_before_sql_or_queue():
    with pytest.raises(RoutingProductCompositionError, match="runner"):
        bootstrap_reviewed_routing_product(environment=_env())


def test_api_install_is_explicit_and_app_state_selected():
    app = SimpleNamespace(state=SimpleNamespace())
    installed = RoutingExperimentApiService()
    install_routing_experiment_api_service(installed, app=app)
    assert app.state.routing_experiment_api_service is installed
    install_routing_experiment_api_service(None, app=app)
    assert app.state.routing_experiment_api_service is None


def test_consumer_registry_accepts_exactly_one_reviewed_factory():
    factory = SimpleNamespace(name=REVIEWED_ROUTING_FACTORY_NAME)
    install_reviewed_routing_factory_registry(
        {REVIEWED_ROUTING_FACTORY_NAME: factory}
    )
    install_reviewed_routing_factory_registry(
        {REVIEWED_ROUTING_FACTORY_NAME: factory}
    )

    with pytest.raises(RoutingExecutionConsumerError, match="already frozen"):
        install_reviewed_routing_factory_registry(
            {REVIEWED_ROUTING_FACTORY_NAME: SimpleNamespace(
                name=REVIEWED_ROUTING_FACTORY_NAME
            )}
        )

    with pytest.raises(RoutingExecutionConsumerError, match="exactly one"):
        install_reviewed_routing_factory_registry(
            {"unreviewed": factory}
        )


def test_factory_builder_does_not_accept_arbitrary_runner_import_path():
    inputs = _inputs()
    with pytest.raises(RoutingProductCompositionError, match="runner factory"):
        build_attested_provider_broker_factory(
            inputs=inputs,
            reviewed_runner_factory="gateway.untrusted:factory",
            billing_rollup_factory=lambda _spec: lambda _store: {},
            execution_envelope_factory=lambda _spec: object(),
            environment=_env(),
        )


def test_dispatch_authority_rejects_arbitrary_callable_executor_at_construction():
    with pytest.raises(TypeError, match="fixed-operation executor"):
        AttestedScoringV2RoutingProviderDispatchAuthority(
            executor=lambda _request: {"status": "succeeded"},
            protected_release_receipt=None,
        )


def test_dispatch_wrapper_rejects_generic_execute_object():
    class _GenericExecuteRpc(_Rpc):
        def execute(self, _request):
            return {}

    with pytest.raises(RoutingProductCompositionError, match="generic execute"):
        RoutingProviderDispatchTeeRpc(_GenericExecuteRpc())


def test_fixed_dispatch_operation_rejects_operation_mismatch_before_submit():
    rpc = _OperationRpc()
    executor = _TeeJobRpcOperationExecutor(
        RoutingProviderDispatchTeeRpc(rpc),
        allowed_operation=ROUTING_PROVIDER_DISPATCH_OPERATION_V2,
        allowed_purpose=ROUTING_PROVIDER_DISPATCH_PURPOSE_V2,
        dispatch_token=_ROUTING_DISPATCH_EXECUTOR_TOKEN,
    )
    with pytest.raises(RoutingProductCompositionError, match="fixed dispatch operation"):
        executor(
            {
                "operation": "routing_other_v2",
                "purpose": ROUTING_PROVIDER_DISPATCH_PURPOSE_V2,
                "payload": {"value": "must-not-submit"},
            }
        )
    assert rpc.manifest is None


def test_dispatch_result_rejects_raw_provider_fields():
    with pytest.raises(Exception, match="redacted schema|raw provider"):
        _validate_redacted_routing_dispatch_result(
            {"body_b64": "provider-response"}
        )


def test_protected_operation_rpc_accepts_exact_job_result_and_receipt_identity():
    response = _TeeJobRpcOperationExecutor(_OperationRpc())(
        {
            "operation": "routing_test_v2",
            "purpose": "research_lab.routing_test.v2",
            "payload": {"value": "exact"},
            "parent_receipt_hashes": ["sha256:" + "8" * 64],
        }
    )
    assert response["status"] == "succeeded"
    assert response["result"] == {"accepted": True}


def test_protected_operation_rpc_rejects_summary_job_substitution():
    with pytest.raises(RoutingProductCompositionError, match="summary identity"):
        _TeeJobRpcOperationExecutor(
            _OperationRpc(summary_job_substitution=True)
        )(
            {
                "operation": "routing_test_v2",
                "purpose": "research_lab.routing_test.v2",
                "payload": {"value": "exact"},
            }
        )


def test_protected_operation_rpc_rejects_result_job_substitution():
    with pytest.raises(RoutingProductCompositionError, match="result job identity"):
        _TeeJobRpcOperationExecutor(
            _OperationRpc(result_job_substitution=True)
        )(
            {
                "operation": "routing_test_v2",
                "purpose": "research_lab.routing_test.v2",
                "payload": {"value": "exact"},
            }
        )
