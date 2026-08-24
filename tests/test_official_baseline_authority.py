from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from io import BytesIO

import pytest

from gateway.research_lab import (
    official_baseline_authority as official_baseline_authority_module,
)
from gateway.research_lab import (
    official_baseline_release_dependencies as release_dependencies_module,
)
from gateway.research_lab.common_model_experiment import (
    CommonModelExperimentError,
    CommonModelExperimentRecoveryError,
    ProtectedModelActionResult,
)
from gateway.research_lab.official_baseline_authority import (
    AppendOnlyOfficialBaselineAuthority,
    GatewayLocalProtectedActionBridge,
    OfficialBaselineProtectedPreparation,
    OfficialBaselineProtectedAuthorityError,
    OfficialBaselineProtectedTerminal,
    OfficialBaselineReleaseComponents,
    _protected_result_document,
    build_production_official_baseline_exact_dependencies,
)
from gateway.research_lab.official_baseline_custody import (
    OfficialBaselineCustodyError,
    S3OfficialBaselineDocumentCustody,
)
from gateway.research_lab.official_baseline_model_runner import (
    OFFICIAL_BASELINE_PROVIDER_FRONTIER_SCHEMA_VERSION,
    OfficialBaselineAuthorityUnavailable,
    OfficialBaselineDependencyContext,
)
from gateway.research_lab.official_baseline_store import (
    official_baseline_action_replay_identity,
)
from research_lab.canonical import sha256_bytes, sha256_json
from research_lab.common_model_runner_host import HostActionResult
from research_lab.routing_experiments import ProviderReceipt
from tests.test_official_baseline_model_runner import _exact_fixture, _provider_action


class _Missing(Exception):
    response = {"Error": {"Code": "NoSuchKey"}}


class _Precondition(Exception):
    response = {"Error": {"Code": "PreconditionFailed"}}


class _S3:
    def __init__(self):
        self.values = {}
        self.puts = 0

    def get_object(self, *, Bucket, Key):
        del Bucket
        if Key not in self.values:
            raise _Missing
        value = self.values[Key]
        return {
            "Body": BytesIO(value["Body"]),
            "Metadata": deepcopy(value["Metadata"]),
            "ServerSideEncryption": value["ServerSideEncryption"],
        }

    def put_object(self, *, Bucket, Key, IfNoneMatch, **value):
        del Bucket
        assert IfNoneMatch == "*"
        if Key in self.values:
            raise _Precondition
        self.puts += 1
        self.values[Key] = {
            **deepcopy(value),
            "Body": bytes(value["Body"]),
        }


def _context(runner) -> OfficialBaselineDependencyContext:
    return OfficialBaselineDependencyContext(
        artifact=runner.artifact,
        artifact_pointer_uri=(
            "s3://fixture/model/branches/leadpoet-lab/current.json"
        ),
        artifact_pointer_manifest_hash=runner.artifact.manifest_hash,
        selection=runner.selection,
        spec=runner.spec,
        benchmark_date="2026-08-23",
        rolling_window_hash="sha256:" + "f" * 64,
        benchmark_attempt=1,
        evaluation_epoch=1,
        parent_graphs=(),
        worker_index=0,
        worker_ref="official-baseline-worker-1",
        evidence_proxy_url="http://127.0.0.1:8765",
        evidence_proxy_capability_sha256="sha256:" + "a" * 64,
        evidence_proxy_ready_provider_ids=("or",),
    )


def test_fresh_daily_attempt_zero_is_a_valid_frozen_context():
    runner, _projector, _authority_value, _terminal = _exact_fixture()
    context = replace(_context(runner), benchmark_attempt=0)

    context.validate()


@pytest.mark.parametrize("branch", ("main", "leadpoet-lab"))
def test_frozen_context_derives_branch_from_verified_pointer(branch):
    runner, _projector, _authority_value, _terminal = _exact_fixture()
    context = replace(
        _context(runner),
        artifact_pointer_uri=(
            f"s3://fixture/model/branches/{branch}/current.json"
        ),
    )

    context.validate()

    assert context.source_branch == branch


@pytest.mark.parametrize(
    "pointer_uri",
    (
        "s3://fixture/model/manifest.json",
        "s3://fixture/model/branches/release-candidate/current.json",
        (
            "s3://fixture/model/branches/main/branches/"
            "leadpoet-lab/current.json"
        ),
        "s3://fixture/model/branches/main/not-current.json",
        "s3://fixture/model/branches/main/current.json?version=1",
        "https://fixture/model/branches/main/current.json",
    ),
)
def test_frozen_context_rejects_noncanonical_artifact_pointer(pointer_uri):
    runner, _projector, _authority_value, _terminal = _exact_fixture()
    context = replace(_context(runner), artifact_pointer_uri=pointer_uri)

    with pytest.raises(
        OfficialBaselineAuthorityUnavailable,
        match="artifact pointer identity is invalid",
    ):
        context.validate()


def test_archive_branch_text_cannot_override_verified_pointer_branch():
    runner, _projector, _authority_value, _terminal = _exact_fixture()
    misleading_artifact = replace(
        runner.artifact,
        manifest_uri="s3://fixture/model/branches/main/manifest.json",
    )
    context = replace(_context(runner), artifact=misleading_artifact)

    with pytest.raises(
        OfficialBaselineAuthorityUnavailable,
        match="artifact pointer identity is invalid",
    ):
        context.validate()


def test_mutable_archive_cannot_supply_exact_branch_provenance():
    runner, _projector, _authority_value, _terminal = _exact_fixture()
    mutable_artifact = replace(
        runner.artifact,
        manifest_uri="s3://fixture/model/current.json",
    )
    context = replace(_context(runner), artifact=mutable_artifact)

    with pytest.raises(
        OfficialBaselineAuthorityUnavailable,
        match="artifact pointer identity is invalid",
    ):
        context.validate()


def test_pointer_manifest_hash_must_bind_the_frozen_artifact():
    runner, _projector, _authority_value, _terminal = _exact_fixture()
    context = replace(
        _context(runner),
        artifact_pointer_manifest_hash="sha256:" + "0" * 64,
    )

    with pytest.raises(
        OfficialBaselineAuthorityUnavailable,
        match="frozen dependency context differs",
    ):
        context.validate()


def test_immutable_archive_may_use_independent_storage_topology():
    runner, _projector, _authority_value, _terminal = _exact_fixture()
    relocated = replace(
        runner.artifact,
        manifest_uri=(
            "s3://immutable-release-archive/releases/"
            + runner.artifact.git_commit_sha
            + ".json"
        ),
    )
    relocated = replace(
        relocated,
        manifest_hash=sha256_json(relocated.hash_payload()),
    )
    context = replace(
        _context(runner),
        artifact=relocated,
        artifact_pointer_manifest_hash=relocated.manifest_hash,
    )

    context.validate()

    assert context.source_branch == "leadpoet-lab"


class _AttemptStore:
    def __init__(self):
        self.registration = None
        self.authorizations = {}
        self.terminals = {}
        self.events = []
        self.foreign_inflight = False

    def register_run(self, *, registration):
        if self.registration is not None and self.registration != registration:
            raise RuntimeError("registration conflict")
        idempotent = self.registration is not None
        self.registration = deepcopy(registration)
        self.events.append("register")
        return {
            "run_sha256": registration["run_sha256"],
            "registration_sha256": sha256_json(registration),
            "idempotent": idempotent,
        }

    def reserve_action(self, *, authorization):
        self.events.append("reserve")
        if self.foreign_inflight:
            return {"disposition": "inflight"}
        key = authorization["attempt_key"]
        existing = self.authorizations.get(key)
        if existing is not None and existing != authorization:
            raise RuntimeError("authorization conflict")
        disposition = "reserved_existing" if existing is not None else "reserved_new"
        self.authorizations[key] = deepcopy(authorization)
        return {
            "disposition": disposition,
            "attempt_key": key,
            "reservation_ref": "baseline_reservation:" + key.removeprefix("sha256:"),
            "lease_generation": 1,
            "protected_job_ref": authorization["protected_job_ref"],
            "protected_request_sha256": authorization["protected_request_sha256"],
        }

    def record_terminal_known(self, *, terminal):
        self.events.append("known")
        key = terminal["attempt_key"]
        existing = self.terminals.get(key)
        if existing is not None and existing != terminal:
            raise RuntimeError("terminal conflict")
        self.terminals[key] = deepcopy(terminal)
        return {"state": "terminal_known"}

    def record_terminal_uncertain(self, *, uncertainty):
        self.events.append("uncertain")
        key = uncertainty["attempt_key"]
        self.terminals[key] = deepcopy(uncertainty)
        return {"state": "terminal_uncertain"}

    def load_replay(self, *, identity):
        key = identity["attempt_key"]
        authorization = self.authorizations.get(key)
        if authorization is None:
            return {"state": "absent", "attempt_key": key}
        terminal = self.terminals.get(key)
        base = {
            "attempt_key": key,
            "reservation_ref": "baseline_reservation:" + key.removeprefix("sha256:"),
            "lease_generation": 1,
            "protected_job_ref": authorization["protected_job_ref"],
            "protected_request_sha256": authorization["protected_request_sha256"],
        }
        if terminal is None:
            return {**base, "state": "reserved"}
        if "uncertainty_sha256" in terminal:
            return {**base, "state": "terminal_uncertain"}
        return {
            **base,
            "state": "terminal_known",
            **{
                key: value
                for key, value in terminal.items()
                if key not in {"schema_version", "attempt_key"}
            },
            "attempt_sha256": sha256_json(terminal),
        }

    def close_unit(self, *, closure):
        attempts = sorted(
            (
                value["action_sequence"],
                key,
                sha256_json(self.terminals[key]),
            )
            for key, value in self.authorizations.items()
            if value["run_sha256"] == closure["run_sha256"]
            and value["unit_ref"] == closure["unit_ref"]
        )
        frontier = {
            "schema_version": OFFICIAL_BASELINE_PROVIDER_FRONTIER_SCHEMA_VERSION,
            "ordered_attempt_keys": [value[1] for value in attempts],
            "ordered_attempt_sha256s": [value[2] for value in attempts],
        }
        return {"frontier": frontier}

    def load_frontier(self, **_values):
        raise AssertionError("not exercised")


def _known_provider_terminal(action, unit_ref):
    receipt = ProviderReceipt(
        receipt_ref="provider_receipt:" + "a" * 16,
        binding_id="fixture-binding",
        tool_id=action["tool_id"],
        binding_version="v1",
        source_lineage_id="fixture-source",
        unit_ref=unit_ref,
        request_fingerprint="sha256:" + action["request_fingerprint_sha256"],
        outcome="verified",
        evidence_hash="sha256:" + "b" * 64,
        credit_microunits=10,
        latency_ms=20,
        execution_mode="measured_lab",
        call_count=1,
    )
    result = ProtectedModelActionResult(
        host_result=HostActionResult(
            outcome="succeeded",
            reason_code="fixture",
            provider_response={"records": []},
            calls=1,
            cost_credits=0.00001,
            latency_ms=20,
            provider_receipt_ref=receipt.receipt_ref,
            provider_receipt_sha256="c" * 64,
            provider_identity_sha256="d" * 64,
        ),
        provider_receipt=receipt,
        replay_ref={"schema_version": "fixture-replay:v1"},
    )
    return OfficialBaselineProtectedTerminal(
        state="known",
        protected_action_result=result,
        protected_result_sha256=sha256_json(_protected_result_document(result)),
        protected_terminal_receipt_ref="protected_terminal:fixture",
        protected_terminal_receipt_sha256="sha256:" + "e" * 64,
        provider_request_ref="provider_request:fixture",
        model_provider_response_sha256=sha256_json(
            result.host_result.provider_response
        ),
    )


def _known_failed_verifier_terminal():
    result = ProtectedModelActionResult(
        host_result=HostActionResult(
            outcome="failed",
            reason_code="not_qualified",
            provider_response=None,
            calls=0,
            cost_credits=0,
            latency_ms=2,
        )
    )
    return OfficialBaselineProtectedTerminal(
        state="known",
        protected_action_result=result,
        protected_result_sha256=sha256_json(_protected_result_document(result)),
        protected_terminal_receipt_ref="protected_terminal:verifier",
        protected_terminal_receipt_sha256="sha256:" + "f" * 64,
        provider_request_ref=None,
        model_provider_response_sha256=sha256_json(None),
    )


class _Bridge:
    def __init__(self, authority_identity_sha256):
        self.authority_identity_sha256 = authority_identity_sha256
        self.execute_count = 0
        self.reconcile_count = 0
        self.next_reconcile = None
        self.next_execute = None

    def prepare(self, *, run_identity, unit_ref, action):
        action_type = action["action_type"]
        verifier = action_type.startswith("verify_")
        return OfficialBaselineProtectedPreparation(
            authority_identity_sha256=self.authority_identity_sha256,
            run_sha256=sha256_json(dict(run_identity)),
            unit_ref=unit_ref,
            action_idempotency_sha256="sha256:" + action["idempotency_key"],
            action_sha256="sha256:" + action["action_sha256"],
            action_sequence=action["sequence"],
            action_type=action_type,
            tool_id=action["tool_id"],
            binding_contract_sha256="sha256:"
            + action["binding_contract_sha256"],
            request_fingerprint_sha256="sha256:"
            + action["request_fingerprint_sha256"],
            request_body_sha256=sha256_json(action["arguments"]),
            call_cap=0 if verifier else 1,
            credit_cap_microunits=0 if verifier else 100,
            timeout_ms=5_000,
            protected_job_ref="protected_job:" + action["idempotency_key"][:16],
            protected_request_sha256=sha256_json(dict(action)),
        )

    def execute_prepared(self, *, preparation, action):
        self.execute_count += 1
        if isinstance(self.next_execute, BaseException):
            raise self.next_execute
        return self.next_execute or _known_provider_terminal(
            action, preparation.unit_ref
        )

    def reconcile(self, *, preparation, action):
        self.reconcile_count += 1
        return self.next_reconcile or OfficialBaselineProtectedTerminal(
            state="not_started",
            protected_action_result=None,
            protected_result_sha256=None,
            protected_terminal_receipt_ref=None,
            protected_terminal_receipt_sha256=None,
            provider_request_ref=None,
            model_provider_response_sha256=None,
        )


def _authority(*, current=False):
    runner, _projector, _old_authority, _terminal = _exact_fixture(
        current=current
    )
    context = _context(runner)
    store = _AttemptStore()
    s3 = _S3()
    custody = S3OfficialBaselineDocumentCustody(
        client=s3,
        bucket="fixture-bucket",
        prefix="official-baseline",
        kms_key_id="alias/fixture-encryption",
    )
    bridge = _Bridge(
        context.selection.selection_document["protected_action_authority_sha256"]
    )
    authority = AppendOnlyOfficialBaselineAuthority(
        context=context,
        registration=runner.dependencies.registration,
        store=store,
        bridge=bridge,
        custody=custody,
    )
    run_identity = dict(runner.run_identity)
    authority.preflight_run(
        run_identity=run_identity,
        registration=runner.dependencies.registration,
    )
    unit_ref = "baseline_icp:" + "1" * 64
    return authority, bridge, store, custody, run_identity, unit_ref


def test_new_action_reserves_before_execute_and_restart_reconciles_no_duplicate():
    authority, bridge, store, _custody, run_identity, unit_ref = _authority()
    action = _provider_action({})
    dispatcher = authority.dispatcher_for_unit(
        run_identity=run_identity, unit_ref=unit_ref
    )

    first = dispatcher.dispatch_provider_action(
        action=action, variant_id="official_baseline", unit_ref=unit_ref
    )
    bridge.next_reconcile = _known_provider_terminal(action, unit_ref)
    second = authority.dispatcher_for_unit(
        run_identity=run_identity, unit_ref=unit_ref
    ).dispatch_provider_action(
        action=action, variant_id="official_baseline", unit_ref=unit_ref
    )

    assert first == second
    assert bridge.execute_count == 1
    assert bridge.reconcile_count == 1
    assert store.events.index("reserve") < store.events.index("known")


def test_current_authority_requires_compiled_dispatch_before_spend_and_joins_custody():
    authority, bridge, store, _custody, run_identity, unit_ref = _authority(
        current=True
    )
    action = _provider_action({})
    dispatcher = authority.dispatcher_for_unit(
        run_identity=run_identity,
        unit_ref=unit_ref,
    )
    events_before = list(store.events)

    with pytest.raises(
        CommonModelExperimentError,
        match="artifact-compiled provider dispatch is unavailable",
    ):
        dispatcher.dispatch_provider_action(
            action=action,
            variant_id="official_baseline",
            unit_ref=unit_ref,
        )

    assert store.events == events_before
    assert bridge.execute_count == 0

    host_response = {
        "schema_version": "host-provider-response:v1",
        "provider": "fixture",
        "status_code": 200,
        "body": {"records": []},
    }
    ingestion = authority._registration.protocol.ingest_provider_response(
        action,
        host_response,
    )
    terminal = _known_provider_terminal(action, unit_ref)
    assert terminal.protected_action_result is not None
    protected = replace(
        terminal.protected_action_result,
        host_result=replace(
            terminal.protected_action_result.host_result,
            provider_response=host_response,
        ),
        model_provider_response_ingestion=ingestion,
    )
    bridge.next_execute = replace(
        terminal,
        protected_action_result=protected,
        protected_result_sha256=sha256_json(
            _protected_result_document(protected)
        ),
        model_provider_response_sha256=sha256_json(host_response),
    )
    compiled = authority._registration.protocol.prepare_provider_request(
        action
    )

    result = dispatcher.dispatch_provider_action(
        action=action,
        variant_id="official_baseline",
        unit_ref=unit_ref,
        compiled_dispatch=compiled,
    )

    assert result.host_result.provider_response == host_response
    assert result.host_result.model_provider_response_ingestion == ingestion
    assert result.host_result.provider_action_receipt_sha256 == "e" * 64
    assert bridge.execute_count == 1


def test_reserved_existing_requires_same_authorization_then_absent_reconcile():
    authority, bridge, store, _custody, run_identity, unit_ref = _authority()
    action = _provider_action({})
    dispatcher = authority.dispatcher_for_unit(
        run_identity=run_identity, unit_ref=unit_ref
    )
    preparation = dispatcher._preparation(action)
    identity, authorization = dispatcher._authorization(
        action=action, preparation=preparation
    )
    store.reserve_action(authorization=authorization)

    dispatcher.dispatch_provider_action(
        action=action, variant_id="official_baseline", unit_ref=unit_ref
    )

    assert store.load_replay(identity=identity)["state"] == "terminal_known"
    assert store.events.count("reserve") == 2
    assert bridge.reconcile_count == 1
    assert bridge.execute_count == 1


def test_foreign_inflight_never_reconciles_or_executes():
    authority, bridge, store, _custody, run_identity, unit_ref = _authority()
    store.foreign_inflight = True
    action = _provider_action({})

    with pytest.raises(CommonModelExperimentRecoveryError, match="foreign inflight"):
        authority.dispatcher_for_unit(
            run_identity=run_identity, unit_ref=unit_ref
        ).dispatch_provider_action(
            action=action, variant_id="official_baseline", unit_ref=unit_ref
        )

    assert bridge.reconcile_count == 0
    assert bridge.execute_count == 0


def test_unknown_call_is_terminal_uncertain_and_never_redispatched():
    authority, bridge, store, _custody, run_identity, unit_ref = _authority()
    action = _provider_action({})
    bridge.next_execute = OfficialBaselineProtectedTerminal(
        state="uncertain",
        protected_action_result=None,
        protected_result_sha256=None,
        protected_terminal_receipt_ref=None,
        protected_terminal_receipt_sha256=None,
        provider_request_ref="provider_request:unknown",
        model_provider_response_sha256=None,
        uncertainty_sha256="sha256:" + "0" * 64,
    )
    dispatcher = authority.dispatcher_for_unit(
        run_identity=run_identity, unit_ref=unit_ref
    )

    with pytest.raises(CommonModelExperimentRecoveryError, match="terminal uncertain"):
        dispatcher.dispatch_provider_action(
            action=action, variant_id="official_baseline", unit_ref=unit_ref
        )
    with pytest.raises(CommonModelExperimentRecoveryError, match="terminal uncertain"):
        dispatcher.dispatch_provider_action(
            action=action, variant_id="official_baseline", unit_ref=unit_ref
        )

    assert bridge.execute_count == 1
    assert "uncertain" in store.events


def test_gateway_bridge_claims_before_execute_and_restarts_without_redispatch():
    runner, _projector, _authority_value, _terminal = _exact_fixture()
    run_identity = dict(runner.run_identity)
    unit_ref = "baseline_icp:" + "3" * 64
    action = _provider_action({})
    s3 = _S3()
    custody = S3OfficialBaselineDocumentCustody(
        client=s3,
        bucket="fixture-bucket",
        prefix="official-baseline",
        kms_key_id="alias/fixture-encryption",
    )
    executor = _Bridge(
        run_identity["authority_identity_sha256"]
    )
    bridge = GatewayLocalProtectedActionBridge(
        custody=custody,
        executor=executor,
    )
    preparation = bridge.prepare(
        run_identity=run_identity,
        unit_ref=unit_ref,
        action=action,
    )

    first = bridge.execute_prepared(preparation=preparation, action=action)
    restarted = GatewayLocalProtectedActionBridge(
        custody=custody,
        executor=executor,
    )
    second = restarted.execute_prepared(preparation=preparation, action=action)

    assert first == second
    assert first.state == "known"
    assert executor.execute_count == 1
    assert executor.reconcile_count == 0
    assert s3.puts == 2


def test_gateway_bridge_claim_without_terminal_reconciles_and_never_executes():
    runner, _projector, _authority_value, _terminal = _exact_fixture()
    run_identity = dict(runner.run_identity)
    unit_ref = "baseline_icp:" + "4" * 64
    action = _provider_action({})
    custody = S3OfficialBaselineDocumentCustody(
        client=_S3(),
        bucket="fixture-bucket",
        prefix="official-baseline",
        kms_key_id="alias/fixture-encryption",
    )
    executor = _Bridge(
        run_identity["authority_identity_sha256"]
    )
    bridge = GatewayLocalProtectedActionBridge(
        custody=custody,
        executor=executor,
    )
    preparation = bridge.prepare(
        run_identity=run_identity,
        unit_ref=unit_ref,
        action=action,
    )
    custody.append_protected_action_claim(
        preparation_sha256=preparation.preparation_sha256,
        claim=bridge._claim_document(preparation),
    )

    terminal = bridge.execute_prepared(preparation=preparation, action=action)

    assert terminal.state == "uncertain"
    assert terminal.uncertainty_sha256
    assert executor.execute_count == 0
    assert executor.reconcile_count == 1


def test_gateway_bridge_recovers_known_terminal_after_execute_interruption():
    runner, _projector, _authority_value, _terminal = _exact_fixture()
    run_identity = dict(runner.run_identity)
    unit_ref = "baseline_icp:" + "9" * 64
    action = _provider_action({})
    custody = S3OfficialBaselineDocumentCustody(
        client=_S3(),
        bucket="fixture-bucket",
        prefix="official-baseline",
        kms_key_id="alias/fixture-encryption",
    )
    executor = _Bridge(run_identity["authority_identity_sha256"])
    executor.next_execute = RuntimeError("interrupted after dispatch")
    executor.next_reconcile = _known_provider_terminal(action, unit_ref)
    bridge = GatewayLocalProtectedActionBridge(
        custody=custody,
        executor=executor,
    )
    preparation = bridge.prepare(
        run_identity=run_identity,
        unit_ref=unit_ref,
        action=action,
    )

    terminal = bridge.execute_prepared(preparation=preparation, action=action)

    assert terminal.state == "known"
    assert executor.execute_count == 1
    assert executor.reconcile_count == 1
    assert bridge.reconcile(preparation=preparation, action=action) == terminal
    assert executor.execute_count == 1
    assert executor.reconcile_count == 1


def test_gateway_bridge_rejects_tampered_claim_and_terminal():
    runner, _projector, _authority_value, _terminal = _exact_fixture()
    run_identity = dict(runner.run_identity)
    unit_ref = "baseline_icp:" + "5" * 64
    action = _provider_action({})
    custody = S3OfficialBaselineDocumentCustody(
        client=_S3(),
        bucket="fixture-bucket",
        prefix="official-baseline",
        kms_key_id="alias/fixture-encryption",
    )
    executor = _Bridge(
        run_identity["authority_identity_sha256"]
    )
    bridge = GatewayLocalProtectedActionBridge(
        custody=custody,
        executor=executor,
    )
    preparation = bridge.prepare(
        run_identity=run_identity,
        unit_ref=unit_ref,
        action=action,
    )
    claim = bridge._claim_document(preparation)
    custody.append_protected_action_claim(
        preparation_sha256=preparation.preparation_sha256,
        claim={**claim, "claim_sha256": "sha256:" + "0" * 64},
    )
    with pytest.raises(OfficialBaselineCustodyError, match="conflict"):
        bridge.execute_prepared(preparation=preparation, action=action)
    assert executor.execute_count == 0

    other_unit = "baseline_icp:" + "6" * 64
    other = bridge.prepare(
        run_identity=run_identity,
        unit_ref=other_unit,
        action=action,
    )
    custody.persist_protected_action_terminal(
        preparation_sha256=other.preparation_sha256,
        terminal={"schema_version": "tampered"},
    )
    with pytest.raises(
        Exception, match="durable terminal|closed"
    ):
        bridge.reconcile(preparation=other, action=action)


def test_gateway_bridge_does_not_persist_malformed_executor_terminal():
    runner, _projector, _authority_value, _terminal = _exact_fixture()
    run_identity = dict(runner.run_identity)
    unit_ref = "baseline_icp:" + "a" * 64
    action = _provider_action({})
    custody = S3OfficialBaselineDocumentCustody(
        client=_S3(),
        bucket="fixture-bucket",
        prefix="official-baseline",
        kms_key_id="alias/fixture-encryption",
    )
    executor = _Bridge(run_identity["authority_identity_sha256"])
    executor.next_execute = replace(
        _known_provider_terminal(action, unit_ref),
        protected_result_sha256="sha256:" + "0" * 64,
    )
    bridge = GatewayLocalProtectedActionBridge(
        custody=custody,
        executor=executor,
    )
    preparation = bridge.prepare(
        run_identity=run_identity,
        unit_ref=unit_ref,
        action=action,
    )

    with pytest.raises(
        OfficialBaselineProtectedAuthorityError,
        match="durable terminal custody differs",
    ):
        bridge.execute_prepared(preparation=preparation, action=action)

    assert custody.load_protected_action_terminal(
        preparation_sha256=preparation.preparation_sha256
    ) is None


def test_protected_provider_progress_is_append_only_and_restart_readable():
    custody = S3OfficialBaselineDocumentCustody(
        client=_S3(),
        bucket="fixture-bucket",
        prefix="official-baseline",
        kms_key_id="alias/fixture-encryption",
    )
    preparation_sha256 = "sha256:" + "7" * 64
    progress = {
        "schema_version": "fixture-provider-progress:v1",
        "preparation_sha256": preparation_sha256,
        "provider_run_ref": "provider_run:fixture",
        "dispatch_sha256": "sha256:" + "8" * 64,
    }

    assert custody.append_protected_action_progress(
        preparation_sha256=preparation_sha256,
        progress=progress,
    ) is True
    assert custody.append_protected_action_progress(
        preparation_sha256=preparation_sha256,
        progress=progress,
    ) is False
    assert custody.load_protected_action_progress(
        preparation_sha256=preparation_sha256
    ) == progress
    with pytest.raises(OfficialBaselineCustodyError, match="conflict"):
        custody.append_protected_action_progress(
            preparation_sha256=preparation_sha256,
            progress={**progress, "provider_run_ref": "provider_run:tampered"},
        )


def test_failed_offline_verifier_records_zero_call_without_provider_custody():
    authority, bridge, store, _custody, run_identity, unit_ref = _authority()
    action = _provider_action({})
    action = {
        **action,
        "action_type": "verify_company",
        "tool_id": "verifier.company",
        "sequence": 0,
    }
    body = dict(action)
    body.pop("action_sha256")
    action["action_sha256"] = sha256_json(body).removeprefix("sha256:")
    bridge.next_execute = _known_failed_verifier_terminal()

    result = authority.dispatcher_for_unit(
        run_identity=run_identity, unit_ref=unit_ref
    ).verify_company_action(action=action, unit_ref=unit_ref)
    identity = official_baseline_action_replay_identity(
        run_sha256=sha256_json(run_identity), unit_ref=unit_ref, action=action
    )
    replay = store.load_replay(identity=identity)

    assert result.outcome == "failed"
    assert replay["outcome"] == "failed"
    assert replay["call_count"] == 0
    assert replay["provider_receipt_ref"] is None


def test_encrypted_transition_replay_repairs_frontier_and_terminal_is_append_only():
    authority, bridge, store, custody, run_identity, unit_ref = _authority()
    action = _provider_action({})
    result = authority.dispatcher_for_unit(
        run_identity=run_identity, unit_ref=unit_ref
    ).dispatch_provider_action(
        action=action, variant_id="official_baseline", unit_ref=unit_ref
    )
    repository = authority.transition_repository_for_unit(
        run_identity=run_identity, unit_ref=unit_ref
    )
    continuation = {"schema_version": "fixture-continuation:v1"}
    completion = {
        "completion_sha256": "f" * 64,
        "provider_response": result.host_result.provider_response,
    }
    repository.append_model_transition(
        experiment_hash=sha256_json(run_identity),
        variant_id="official_baseline",
        unit_ref=unit_ref,
        action=action,
        continuation=continuation,
        completion=completion,
        provider_receipt=result.provider_receipt.to_dict(),
        protocol_generation_sha256=(
            authority._registration.protocol_generation.protocol_generation_sha256
        ),
        replay_ref=result.replay_ref,
    )
    loaded = authority.transition_repository_for_unit(
        run_identity=run_identity, unit_ref=unit_ref
    ).load_model_transition(
        experiment_hash=sha256_json(run_identity),
        variant_id="official_baseline",
        unit_ref=unit_ref,
        idempotency_key=action["idempotency_key"],
    )
    assert loaded["completion"] == completion
    assert repository.expected_frontier_sha256(1)

    record_identity = "sha256:" + "2" * 64
    record = {"company_outputs": []}
    persisted = custody.persist_terminal_record(
        record_identity_sha256=record_identity, record=record
    )
    custody.persist_terminal_record(
        record_identity_sha256=record_identity, record=record
    )
    assert custody.load_terminal_record(
        terminal_record_ref=persisted["terminal_record_ref"]
    ) == record
    with pytest.raises(OfficialBaselineCustodyError, match="conflict"):
        custody.persist_terminal_record(
            record_identity_sha256=record_identity,
            record={"company_outputs": [{"tampered": True}]},
        )


def test_production_factory_is_fail_closed_without_signed_component_handoff(
    monkeypatch,
):
    runner, _projector, _authority_value, _terminal = _exact_fixture()
    custody = S3OfficialBaselineDocumentCustody(
        client=_S3(),
        bucket="fixture-bucket",
        prefix="official-baseline",
        kms_key_id="alias/fixture-encryption",
    )
    monkeypatch.setattr(
        official_baseline_authority_module,
        "_production_custody",
        lambda: custody,
    )
    monkeypatch.setattr(
        release_dependencies_module,
        "OFFICIAL_BASELINE_RELEASE_AUTHORITIES_MODULE",
        "gateway.research_lab.missing_official_baseline_release_authorities",
    )
    with pytest.raises(
        OfficialBaselineAuthorityUnavailable,
        match="signed release authority package is unavailable",
    ):
        build_production_official_baseline_exact_dependencies(
            _context(runner), _AttemptStore()
        )


def test_production_factory_reuses_one_custody_for_every_authority(monkeypatch):
    runner, projector, _authority_value, _terminal = _exact_fixture()
    context = _context(runner)
    custody = S3OfficialBaselineDocumentCustody(
        client=_S3(),
        bucket="fixture-bucket",
        prefix="official-baseline",
        kms_key_id="alias/fixture-encryption",
    )
    bridge = _Bridge(
        context.selection.selection_document[
            "protected_action_authority_sha256"
        ]
    )
    captured = {}

    def loader(*, context, custody):
        captured["context"] = context
        captured["custody"] = custody
        return OfficialBaselineReleaseComponents(
            registration=runner.dependencies.registration,
            projector=projector,
            protected_bridge=bridge,
        )

    monkeypatch.setattr(
        official_baseline_authority_module,
        "_production_custody",
        lambda: custody,
    )
    monkeypatch.setattr(
        release_dependencies_module,
        "load_official_baseline_release_components",
        loader,
    )

    dependencies = build_production_official_baseline_exact_dependencies(
        context,
        _AttemptStore(),
    )

    assert captured == {"context": context, "custody": custody}
    assert dependencies.terminal_authority is custody
    assert dependencies.protected_authority._custody is custody
