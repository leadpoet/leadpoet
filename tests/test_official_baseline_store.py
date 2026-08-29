from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from gateway.research_lab import (
    official_baseline_store as official_baseline_store_module,
)

from gateway.research_lab.official_baseline_model_runner import (
    EXACT_MODEL_RUNNER_FAMILY,
    OFFICIAL_BASELINE_ACTION_AUTHORIZATION_SCHEMA_VERSION,
    OFFICIAL_BASELINE_ACTION_REPLAY_IDENTITY_SCHEMA_VERSION,
    OFFICIAL_BASELINE_ACTION_REPLAY_RESULT_SCHEMA_VERSION,
    OFFICIAL_BASELINE_ACTION_RESERVATION_RESULT_SCHEMA_VERSION,
    OFFICIAL_BASELINE_ACTION_TERMINAL_KNOWN_SCHEMA_VERSION,
    OFFICIAL_BASELINE_ACTION_TERMINAL_RESULT_SCHEMA_VERSION,
    OFFICIAL_BASELINE_ACTION_TERMINAL_UNCERTAIN_SCHEMA_VERSION,
    OFFICIAL_BASELINE_EXECUTION_SCHEMA_VERSION,
    OFFICIAL_BASELINE_PROVIDER_FRONTIER_SCHEMA_VERSION,
    OFFICIAL_BASELINE_RUN_REGISTRATION_RESULT_SCHEMA_VERSION,
    OFFICIAL_BASELINE_RUN_REGISTRATION_SCHEMA_VERSION,
    OFFICIAL_BASELINE_UNIT_CLOSURE_SCHEMA_VERSION,
    OFFICIAL_BASELINE_UNIT_COMPLETION_SCHEMA_VERSION,
    OfficialBaselineAuthorityUnavailable,
    OfficialBaselineDependencyContext,
    OfficialBaselineExactDependencies,
    select_official_baseline_release,
)
from gateway.research_lab.official_baseline_store import (
    OFFICIAL_BASELINE_MIGRATION,
    OFFICIAL_BASELINE_RPCS,
    OfficialBaselineStoreError,
    SupabaseOfficialBaselineAttemptStore,
    official_baseline_terminal_store_outcome,
)
from gateway.research_lab.scoring_worker import ResearchLabGatewayScoringWorker
from gateway.tee.rehearsal_behavior_contract_v2 import EXACT_PRODUCTION_ENTRYPOINTS
from gateway.tee.supabase_schema_preflight_v2 import (
    REQUIRED_SUPABASE_V2_RPCS,
    REQUIRED_SUPABASE_V2_SCHEMA,
)
from research_lab.canonical import sha256_json
from research_lab.eval import DockerPrivateModelSpec, PrivateModelArtifactManifest


def _sha(character: str) -> str:
    return "sha256:" + character * 64


RUN = _sha("1")
UNIT = "baseline_icp:" + "2" * 64
ATTEMPT = _sha("3")
RESERVATION = "baseline_reservation:" + ATTEMPT.removeprefix("sha256:")


def _registration() -> dict:
    return {
        "schema_version": OFFICIAL_BASELINE_RUN_REGISTRATION_SCHEMA_VERSION,
        "run_sha256": RUN,
        "benchmark_date": "2026-08-23",
        "rolling_window_hash": _sha("4"),
        "model_artifact_hash": _sha("5"),
        "manifest_hash": _sha("6"),
        "release_selection_sha256": _sha("7"),
        "artifact_key_sha256": _sha("8"),
        "protocol_generation_sha256": _sha("9"),
        "projection_identity_sha256": _sha("a"),
        "authority_identity_sha256": _sha("b"),
    }


def _authorization() -> dict:
    return {
        "schema_version": OFFICIAL_BASELINE_ACTION_AUTHORIZATION_SCHEMA_VERSION,
        "attempt_key": ATTEMPT,
        "run_sha256": RUN,
        "unit_ref": UNIT,
        "action_idempotency_sha256": _sha("c"),
        "action_sha256": _sha("d"),
        "action_sequence": 0,
        "action_type": "execute_candidate_tool",
        "tool_id": "candidate.fixture",
        "binding_contract_sha256": _sha("e"),
        "request_fingerprint_sha256": _sha("f"),
        "request_body_sha256": _sha("0"),
        "call_cap": 1,
        "credit_cap_microunits": 100,
        "timeout_ms": 30_000,
        "protected_job_ref": "baseline_job:fixture",
        "protected_request_sha256": _sha("1"),
        "lease_holder_sha256": _sha("2"),
        "expected_frontier_sha256": _sha("3"),
    }


def _verifier_authorization() -> dict:
    return {
        **_authorization(),
        "action_type": "verify_company",
        "tool_id": "verifier.company",
        "call_cap": 0,
        "credit_cap_microunits": 0,
        "timeout_ms": 0,
    }


def _terminal_known() -> dict:
    return {
        "schema_version": OFFICIAL_BASELINE_ACTION_TERMINAL_KNOWN_SCHEMA_VERSION,
        "attempt_key": ATTEMPT,
        "reservation_ref": RESERVATION,
        "lease_generation": 1,
        "protected_job_ref": "baseline_job:fixture",
        "protected_request_sha256": _sha("1"),
        "protected_result_sha256": _sha("4"),
        "protected_terminal_receipt_ref": "protected_terminal:fixture",
        "protected_terminal_receipt_sha256": _sha("5"),
        "provider_request_ref": "provider_request:fixture",
        "provider_receipt_ref": "provider_receipt:" + "6" * 16,
        "provider_receipt_sha256": _sha("6"),
        "provider_identity_sha256": _sha("7"),
        "model_provider_response_sha256": _sha("8"),
        "outcome": "succeeded",
        "call_count": 1,
        "cost_microunits": 10,
        "latency_ms": 20,
    }


def _terminal_uncertain() -> dict:
    return {
        "schema_version": OFFICIAL_BASELINE_ACTION_TERMINAL_UNCERTAIN_SCHEMA_VERSION,
        "attempt_key": ATTEMPT,
        "reservation_ref": RESERVATION,
        "lease_generation": 1,
        "protected_job_ref": "baseline_job:fixture",
        "protected_request_sha256": _sha("1"),
        "provider_request_ref": "provider_request:fixture",
        "uncertainty_sha256": _sha("9"),
    }


def _replay_identity() -> dict:
    authorization = _authorization()
    return {
        "schema_version": OFFICIAL_BASELINE_ACTION_REPLAY_IDENTITY_SCHEMA_VERSION,
        "attempt_key": authorization["attempt_key"],
        "run_sha256": authorization["run_sha256"],
        "unit_ref": authorization["unit_ref"],
        "action_idempotency_sha256": authorization["action_idempotency_sha256"],
        "action_sha256": authorization["action_sha256"],
        "request_fingerprint_sha256": authorization["request_fingerprint_sha256"],
    }


def _completion() -> dict:
    return {
        "schema_version": OFFICIAL_BASELINE_UNIT_COMPLETION_SCHEMA_VERSION,
        "run_sha256": RUN,
        "unit_ref": UNIT,
        "protocol_generation_sha256": _sha("9"),
        "raw_input_sha256": _sha("a"),
        "start_request_sha256": _sha("b"),
        "terminal_result_sha256": _sha("c"),
        "model_receipt_sha256": _sha("d"),
        "projection_sha256": _sha("e"),
    }


def _dependency_context() -> OfficialBaselineDependencyContext:
    release_identity = {"schema_version": "model-release-identity:v3"}
    execution = {
        "schema_version": OFFICIAL_BASELINE_EXECUTION_SCHEMA_VERSION,
        "runner_family": EXACT_MODEL_RUNNER_FAMILY,
        "execution_mode": "measured_lab",
        "release_identity_sha256": sha256_json(release_identity),
        "protocol_generation_sha256": _sha("1"),
        "benchmark_projection_sha256": _sha("2"),
        "protected_action_authority_sha256": _sha("3"),
    }
    artifact = PrivateModelArtifactManifest(
        model_artifact_hash=_sha("4"),
        git_commit_sha="5" * 40,
        image_digest="example.invalid/model@sha256:" + "6" * 64,
        config_hash=_sha("7"),
        component_registry_version="components:v3",
        scoring_adapter_version="scoring:v1",
        manifest_uri="s3://fixture/model/" + "5" * 40 + ".json",
        manifest_hash=_sha("8"),
        signature_ref="kms://fixture",
        signed_extensions={
            "model_release_identity": release_identity,
            "official_baseline_execution": execution,
        },
    )
    return OfficialBaselineDependencyContext(
        artifact=artifact,
        artifact_pointer_uri=(
            "s3://fixture/model/branches/leadpoet-lab/current.json"
        ),
        artifact_pointer_manifest_hash=artifact.manifest_hash,
        selection=select_official_baseline_release(artifact),
        spec=DockerPrivateModelSpec(image_digest=artifact.image_digest),
        benchmark_date="2026-08-23",
        rolling_window_hash=_sha("9"),
        benchmark_attempt=1,
        evaluation_epoch=1,
        parent_graphs=(),
        worker_index=0,
        worker_ref="fixture-worker",
        evidence_proxy_url="http://127.0.0.1:8765",
        evidence_proxy_capability_sha256=_sha("a"),
        evidence_proxy_ready_provider_ids=("or",),
    )


def _closure() -> dict:
    completion = _completion()
    frontier = {
        "schema_version": OFFICIAL_BASELINE_PROVIDER_FRONTIER_SCHEMA_VERSION,
        "ordered_attempt_keys": [ATTEMPT],
        "ordered_attempt_sha256s": [_sha("f")],
    }
    body = {
        "schema_version": OFFICIAL_BASELINE_UNIT_CLOSURE_SCHEMA_VERSION,
        **{key: value for key, value in completion.items() if key != "schema_version"},
        "ordered_attempt_keys": frontier["ordered_attempt_keys"],
        "ordered_attempt_sha256s": frontier["ordered_attempt_sha256s"],
        "provider_frontier_sha256": sha256_json(frontier),
    }
    closure_sha256 = sha256_json(body)
    return {
        **body,
        "closure_ref": "baseline_closure:" + closure_sha256.removeprefix("sha256:"),
        "closure_sha256": closure_sha256,
        "idempotent": False,
    }


class _RpcCall:
    def __init__(self, data):
        self.data = data

    def execute(self):
        return SimpleNamespace(data=deepcopy(self.data))


class _Client:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def rpc(self, name, params):
        self.calls.append((name, deepcopy(params)))
        response = self.responses[name]
        if isinstance(response, BaseException):
            raise response
        return _RpcCall(response)


class _SequencedClient(_Client):
    def __init__(self, responses, sequence):
        super().__init__(responses)
        self.sequence = list(sequence)

    def rpc(self, name, params):
        self.calls.append((name, deepcopy(params)))
        response = self.sequence.pop(0)
        if isinstance(response, BaseException):
            raise response
        return _RpcCall(response)


def _responses() -> dict:
    registration = _registration()
    authorization = _authorization()
    known = _terminal_known()
    closure = _closure()
    return {
        OFFICIAL_BASELINE_RPCS[0]: {
            "schema_version": OFFICIAL_BASELINE_RUN_REGISTRATION_RESULT_SCHEMA_VERSION,
            "run_sha256": RUN,
            "registration_sha256": sha256_json(registration),
            "idempotent": False,
        },
        OFFICIAL_BASELINE_RPCS[1]: {
            "schema_version": OFFICIAL_BASELINE_ACTION_RESERVATION_RESULT_SCHEMA_VERSION,
            "disposition": "reserved_new",
            "attempt_key": ATTEMPT,
            "reservation_ref": RESERVATION,
            "lease_generation": 1,
            "lease_expires_at": "2026-08-23T20:00:00+00:00",
            "protected_job_ref": authorization["protected_job_ref"],
            "protected_request_sha256": authorization["protected_request_sha256"],
            "attempt_sha256": sha256_json(authorization),
        },
        OFFICIAL_BASELINE_RPCS[2]: {
            "schema_version": OFFICIAL_BASELINE_ACTION_TERMINAL_RESULT_SCHEMA_VERSION,
            "state": "terminal_known",
            "attempt_key": ATTEMPT,
            "attempt_sha256": _sha("a"),
            "idempotent": False,
        },
        OFFICIAL_BASELINE_RPCS[3]: {
            "schema_version": OFFICIAL_BASELINE_ACTION_TERMINAL_RESULT_SCHEMA_VERSION,
            "state": "terminal_uncertain",
            "attempt_key": ATTEMPT,
            "attempt_sha256": _sha("b"),
            "idempotent": False,
        },
        OFFICIAL_BASELINE_RPCS[4]: {
            "schema_version": OFFICIAL_BASELINE_ACTION_REPLAY_RESULT_SCHEMA_VERSION,
            "state": "terminal_known",
            "attempt_key": ATTEMPT,
            "reservation_ref": RESERVATION,
            "lease_generation": 1,
            "lease_expires_at": "2026-08-23T20:00:00+00:00",
            "protected_job_ref": known["protected_job_ref"],
            "protected_request_sha256": known["protected_request_sha256"],
            "protected_result_sha256": known["protected_result_sha256"],
            "protected_terminal_receipt_ref": known["protected_terminal_receipt_ref"],
            "protected_terminal_receipt_sha256": known[
                "protected_terminal_receipt_sha256"
            ],
            "provider_request_ref": known["provider_request_ref"],
            "provider_receipt_ref": known["provider_receipt_ref"],
            "provider_receipt_sha256": known["provider_receipt_sha256"],
            "provider_identity_sha256": known["provider_identity_sha256"],
            "model_provider_response_sha256": known["model_provider_response_sha256"],
            "outcome": known["outcome"],
            "call_count": known["call_count"],
            "cost_microunits": known["cost_microunits"],
            "latency_ms": known["latency_ms"],
            "attempt_sha256": _sha("c"),
        },
        OFFICIAL_BASELINE_RPCS[5]: closure,
        OFFICIAL_BASELINE_RPCS[6]: {**closure, "idempotent": True},
    }


def test_supabase_store_uses_all_seven_exact_rpc_shapes():
    client = _Client(_responses())
    store = SupabaseOfficialBaselineAttemptStore(client)

    store.register_run(registration=_registration())
    store.reserve_action(authorization=_authorization())
    store.record_terminal_known(terminal=_terminal_known())
    store.record_terminal_uncertain(uncertainty=_terminal_uncertain())
    store.load_replay(identity=_replay_identity())
    store.close_unit(closure=_completion())
    store.load_frontier(run_sha256=RUN, unit_ref=UNIT)

    assert [name for name, _params in client.calls] == list(OFFICIAL_BASELINE_RPCS)
    assert [set(params) for _name, params in client.calls] == [
        {"p_registration"},
        {"p_authorization"},
        {"p_terminal"},
        {"p_terminal"},
        {"p_identity"},
        {"p_completion"},
        {"p_run_sha256", "p_unit_ref"},
    ]


def test_supabase_store_retries_exact_rpc_after_ambiguous_timeout(monkeypatch):
    responses = _responses()
    response = responses[OFFICIAL_BASELINE_RPCS[2]]
    client = _SequencedClient(
        responses,
        [TimeoutError("fixture timeout"), response],
    )
    sleeps = []
    monkeypatch.setattr(
        official_baseline_store_module.time,
        "sleep",
        sleeps.append,
    )
    store = SupabaseOfficialBaselineAttemptStore(client)

    result = store.record_terminal_known(terminal=_terminal_known())

    assert result["state"] == "terminal_known"
    assert client.calls == [
        (OFFICIAL_BASELINE_RPCS[2], {"p_terminal": _terminal_known()}),
        (OFFICIAL_BASELINE_RPCS[2], {"p_terminal": _terminal_known()}),
    ]
    assert sleeps == [0.1]


def test_supabase_store_never_retries_logic_or_conflict_error(monkeypatch):
    responses = _responses()
    client = _SequencedClient(
        responses,
        [ValueError("fixture conflict")],
    )
    sleeps = []
    monkeypatch.setattr(
        official_baseline_store_module.time,
        "sleep",
        sleeps.append,
    )
    store = SupabaseOfficialBaselineAttemptStore(client)

    with pytest.raises(OfficialBaselineStoreError, match="ValueError"):
        store.record_terminal_known(terminal=_terminal_known())

    assert len(client.calls) == 1
    assert sleeps == []


def test_supabase_store_accepts_exact_zero_call_verifier_authorization():
    authorization = _verifier_authorization()
    responses = _responses()
    reservation = responses[OFFICIAL_BASELINE_RPCS[1]]
    reservation.update(
        protected_job_ref=authorization["protected_job_ref"],
        protected_request_sha256=authorization["protected_request_sha256"],
        attempt_sha256=sha256_json(authorization),
    )
    client = _Client(responses)
    store = SupabaseOfficialBaselineAttemptStore(client)

    result = store.reserve_action(authorization=authorization)

    assert result["disposition"] == "reserved_new"
    assert client.calls == [
        (OFFICIAL_BASELINE_RPCS[1], {"p_authorization": authorization})
    ]


@pytest.mark.parametrize(
    "authorization",
    (
        {**_authorization(), "timeout_ms": 0},
        {**_verifier_authorization(), "timeout_ms": 1},
    ),
)
def test_supabase_store_rejects_wrong_timeout_class_before_rpc(authorization):
    client = _Client(_responses())
    store = SupabaseOfficialBaselineAttemptStore(client)

    with pytest.raises(OfficialBaselineStoreError, match="accounting is invalid"):
        store.reserve_action(authorization=authorization)

    assert client.calls == []


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda responses: responses[OFFICIAL_BASELINE_RPCS[0]].update(extra=True),
            "not closed",
        ),
        (
            lambda responses: responses[OFFICIAL_BASELINE_RPCS[1]].update(
                schema_version="wrong:v1"
            ),
            "schema differs",
        ),
        (
            lambda responses: responses[OFFICIAL_BASELINE_RPCS[4]].update(
                provider_receipt_sha256=None
            ),
            "custody is incomplete",
        ),
        (
            lambda responses: responses[OFFICIAL_BASELINE_RPCS[5]].update(
                closure_sha256=_sha("0")
            ),
            "closure hash differs",
        ),
    ),
)
def test_supabase_store_rejects_nonclosed_or_tampered_results(mutation, message):
    responses = _responses()
    mutation(responses)
    client = _Client(responses)
    store = SupabaseOfficialBaselineAttemptStore(client)

    calls = (
        lambda: store.register_run(registration=_registration()),
        lambda: store.reserve_action(authorization=_authorization()),
        lambda: store.load_replay(identity=_replay_identity()),
        lambda: store.close_unit(closure=_completion()),
    )
    index = [
        OFFICIAL_BASELINE_RPCS[0],
        OFFICIAL_BASELINE_RPCS[1],
        OFFICIAL_BASELINE_RPCS[4],
        OFFICIAL_BASELINE_RPCS[5],
    ].index(
        next(
            name
            for name in OFFICIAL_BASELINE_RPCS
            if responses[name] != _responses()[name]
        )
    )
    with pytest.raises((OfficialBaselineStoreError, RuntimeError), match=message):
        calls[index]()


def test_supabase_store_rejects_unclosed_input_before_rpc():
    client = _Client(_responses())
    store = SupabaseOfficialBaselineAttemptStore(client)
    registration = {**_registration(), "raw_provider_response": "forbidden"}

    with pytest.raises(OfficialBaselineStoreError, match="not closed"):
        store.register_run(registration=registration)

    assert client.calls == []


def test_supabase_store_rpc_error_is_sanitized():
    responses = _responses()
    responses[OFFICIAL_BASELINE_RPCS[0]] = RuntimeError(
        "provider secret must never escape"
    )
    store = SupabaseOfficialBaselineAttemptStore(_Client(responses))

    with pytest.raises(OfficialBaselineStoreError) as captured:
        store.register_run(registration=_registration())

    assert "provider secret" not in str(captured.value)
    assert OFFICIAL_BASELINE_RPCS[0] in str(captured.value)


@pytest.mark.parametrize(
    ("model_outcome", "stored_outcome"),
    (
        ("succeeded", "succeeded"),
        ("empty", "empty"),
        ("failed", "failed"),
        ("unavailable", "failed"),
        ("timeout", "failed"),
    ),
)
def test_known_model_outcomes_map_to_closed_sql_accounting(
    model_outcome, stored_outcome
):
    assert official_baseline_terminal_store_outcome(model_outcome) == stored_outcome


def test_unknown_or_ambiguous_outcome_cannot_enter_terminal_known():
    for outcome in ("", "uncertain", "consumed_unknown"):
        with pytest.raises(OfficialBaselineStoreError, match="unsupported"):
            official_baseline_terminal_store_outcome(outcome)


def test_failed_zero_call_verifier_terminal_and_replay_are_valid():
    terminal = _terminal_known()
    terminal.update(
        outcome="failed",
        call_count=0,
        cost_microunits=0,
        provider_request_ref=None,
        provider_receipt_ref=None,
        provider_receipt_sha256=None,
        provider_identity_sha256=None,
    )
    responses = _responses()
    replay = responses[OFFICIAL_BASELINE_RPCS[4]]
    replay.update(
        outcome="failed",
        call_count=0,
        cost_microunits=0,
        provider_request_ref=None,
        provider_receipt_ref=None,
        provider_receipt_sha256=None,
        provider_identity_sha256=None,
    )
    store = SupabaseOfficialBaselineAttemptStore(_Client(responses))

    store.record_terminal_known(terminal=terminal)
    result = store.load_replay(identity=_replay_identity())

    assert result["outcome"] == "failed"
    assert result["call_count"] == 0
    assert result["provider_receipt_ref"] is None


def test_worker_factory_constructs_one_concrete_attempt_store():
    worker = object.__new__(ResearchLabGatewayScoringWorker)
    worker._official_baseline_exact_dependencies = None
    captured = []
    concrete = SupabaseOfficialBaselineAttemptStore(_Client(_responses()))
    worker._official_baseline_attempt_store_factory = lambda: concrete

    context = _dependency_context()

    def build(frozen_context, store):
        assert frozen_context is context
        captured.append(store)
        return OfficialBaselineExactDependencies(None, None, None, None)

    worker._official_baseline_exact_dependencies_factory = build
    result = worker._construct_official_baseline_exact_dependencies(context=context)

    assert isinstance(result, OfficialBaselineExactDependencies)
    assert captured == [concrete]


def test_worker_factory_rejects_non_migration_store():
    worker = object.__new__(ResearchLabGatewayScoringWorker)
    worker._official_baseline_exact_dependencies = None
    worker._official_baseline_exact_dependencies_factory = lambda _context, _store: None
    worker._official_baseline_attempt_store_factory = object

    with pytest.raises(OfficialBaselineAuthorityUnavailable, match="migration-163"):
        worker._construct_official_baseline_exact_dependencies(
            context=_dependency_context()
        )


def test_migration_163_tables_and_all_seven_rpcs_are_restart_gated():
    relations = {
        (migration, relation)
        for migration, relation, _columns in REQUIRED_SUPABASE_V2_SCHEMA
    }
    expected_relations = {
        "research_lab_official_baseline_runs_v1",
        "research_lab_official_baseline_action_attempts_v1",
        "research_lab_official_baseline_action_terminals_v1",
        "research_lab_official_baseline_unit_closures_v1",
    }
    assert {
        (OFFICIAL_BASELINE_MIGRATION, relation) for relation in expected_relations
    } <= relations
    reserve_migration = "scripts/166-research-lab-zero-call-verifier-timeout.sql"
    assert {
        (OFFICIAL_BASELINE_MIGRATION, rpc)
        for rpc in OFFICIAL_BASELINE_RPCS
        if rpc != OFFICIAL_BASELINE_RPCS[1]
    } <= set(REQUIRED_SUPABASE_V2_RPCS)
    assert (reserve_migration, OFFICIAL_BASELINE_RPCS[1]) in set(
        REQUIRED_SUPABASE_V2_RPCS
    )
    assert {
        "gateway/research_lab/official_baseline_model_runner.py",
        "gateway/research_lab/official_baseline_store.py",
        OFFICIAL_BASELINE_MIGRATION,
        reserve_migration,
    } <= set(EXACT_PRODUCTION_ENTRYPOINTS)
    OFFICIAL_BASELINE_EXECUTION_SCHEMA_VERSION,
