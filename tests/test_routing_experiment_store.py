from __future__ import annotations

from dataclasses import dataclass
import inspect

import pytest

from gateway.research_lab.routing_experiment_attestation import (
    execute_routing_experiment_attestation_v2,
)
from gateway.research_lab.routing_experiment_store import (
    RoutingExperimentExecutionClaim,
    RoutingExperimentStoreError,
    SupabaseRoutingExperimentStore,
    routing_claim_fence_hash_v3,
)
import gateway.research_lab.routing_experiment_store as routing_store_module
from gateway.research_lab.routing_execution_authorization import (
    RoutingProviderCallAuthorizationV2,
)
from research_lab.routing_experiments import ProviderReceipt
from research_lab.canonical import sha256_json
from tests.routing_experiment_authority_fixture import authority_fixture, _signed_receipt


def _hash(char: str) -> str:
    return "sha256:" + char * 64


def _artifact_key(commit_char: str = "1") -> str:
    return (
        commit_char * 40
        + ":sha256:"
        + "a" * 64
        + ":sha256:"
        + "b" * 64
    )


def _model_transition_marker(*, artifact_key: str | None = None) -> dict:
    marker = {
        "schema_version": "leadpoet.research_lab.routing_event.v2",
        "event_schema_version": "leadpoet.research_lab.model_transition.v2",
        "variant_id": "baseline",
        "unit_ref": "unit-1",
        "artifact_key": artifact_key or _artifact_key(),
        "idempotency_key": "c" * 64,
        "action_sha256": "d" * 64,
        "continuation_sha256": _hash("e"),
        "completion_sha256": "f" * 64,
        "provider_response_sha256": _hash("1"),
        "provider_receipt": None,
        "protected_dispatch_job_id": None,
        "terminal_receipt_hash": None,
        "model_completion_contract_hash": None,
    }
    return marker


class _Response:
    def __init__(self, data):
        self.data = data


class _Query:
    def __init__(self, rows):
        self.rows = [dict(row) for row in rows]
        self.filters = []
        self.limit_value = None
        self.order_column = None

    def select(self, _columns):
        return self

    def eq(self, key, value):
        self.filters.append((key, value))
        return self

    def order(self, column):
        self.order_column = column
        return self

    def limit(self, value):
        self.limit_value = value
        return self

    def execute(self):
        rows = [
            row
            for row in self.rows
            if all(row.get(key) == value for key, value in self.filters)
        ]
        if self.order_column:
            rows.sort(key=lambda row: str(row.get(self.order_column) or ""))
        if self.limit_value is not None:
            rows = rows[: self.limit_value]
        return _Response(rows)


class _Client:
    def __init__(self, tables, *, rpc_results=None):
        self.tables = tables
        self.rpc_results = dict(rpc_results or {})
        self.rpc_calls = []

    def table(self, name):
        return _Query(self.tables.get(name, []))

    def rpc(self, name, params):
        self.rpc_calls.append((name, dict(params)))
        if name not in self.rpc_results:
            raise AssertionError("reconcile must not write")
        return _QueryResult(self.rpc_results[name])


class _QueryResult:
    def __init__(self, data):
        self.data = data

    def execute(self):
        return self


@dataclass(frozen=True)
class _Spec:
    experiment: str = _hash("a")
    allow_live_credit_spend: bool = False

    def experiment_hash(self):
        return self.experiment

    def to_dict(self):
        return {"contract_version": "test", "experiment_hash": self.experiment}


@dataclass(frozen=True)
class _Evaluation:
    receipt_id: str = "routing_evaluation_v2:" + "b" * 16
    selected_variant_id: str = "candidate"
    decision_receipt_refs: tuple[str, ...] = ("routing_decision:" + "c" * 16,)
    provider_receipt_refs: tuple[str, ...] = ("provider_receipt:" + "d" * 16,)
    billing_rollup_hash: str = _hash("e")
    billing_rollup_total_credit_microunits: int = 10

    def to_dict(self):
        return {
            "receipt_id": self.receipt_id,
            "selected_variant_id": self.selected_variant_id,
            "decision_receipt_refs": list(self.decision_receipt_refs),
            "provider_receipt_refs": list(self.provider_receipt_refs),
            "billing_rollup_hash": self.billing_rollup_hash,
            "billing_rollup_total_credit_microunits": self.billing_rollup_total_credit_microunits,
        }


def _store(
    *,
    extra_attempt=False,
    live=False,
    budget_event_type: str | None = None,
    budget_rows: list[dict] | None = None,
):
    del live
    fixture = authority_fixture()
    spec = fixture["spec"]
    evaluation = fixture["evaluation"]
    attempt_rows = [dict(item) for item in fixture["attempts"]]
    if extra_attempt:
        extra = dict(attempt_rows[0])
        extra["attempt_key"] = _hash("7")
        extra["provider_receipt_ref"] = "provider_receipt:" + "f" * 16
        attempt_rows.append(extra)
    normalized_budgets = [dict(item) for item in fixture["budgets"]]
    if budget_event_type is not None:
        normalized_budgets = [
            ({**item, "event_type": budget_event_type} if item["event_type"] == "settle" else item)
            for item in normalized_budgets
        ]
    if budget_rows is not None:
        normalized_budgets = budget_rows
    tables = {
        "research_lab_routing_experiments_v2": [
            {
                "experiment_hash": spec.experiment_hash(),
                "spec_doc": spec.to_dict(),
                "execution_envelope_hash": sha256_json(fixture["execution_envelope"]),
                "execution_envelope_doc": fixture["execution_envelope"],
            }
        ],
        "research_lab_routing_evaluation_receipts_v2": [
            {
                "receipt_id": evaluation.receipt_id,
                "experiment_hash": spec.experiment_hash(),
                "evaluation_hash": sha256_json(evaluation.to_dict()),
                "evaluation_doc": evaluation.to_dict(),
            }
        ],
        "research_lab_routing_decision_receipts_v2": [
            dict(item) for item in fixture["decisions"]
        ],
        "research_lab_routing_provider_attempts_v2": attempt_rows,
        "research_lab_routing_budget_events_v2": normalized_budgets,
        "research_lab_attested_execution_receipts_v2": [],
    }
    store = SupabaseRoutingExperimentStore(_Client(tables))
    store._test_gold_labels = fixture["labels"]
    store._test_artifact_lineage = fixture["lineage"]
    return store, spec, evaluation


def _reconcile(store, spec, evaluation, attestor):
    if isinstance(attestor, _Attestor):
        attestor.receipt_rows = store.client.tables[
            "research_lab_attested_execution_receipts_v2"
        ]
    return store.reconcile(
        spec=spec,
        evaluation=evaluation,
        attestor=attestor,
        gold_label_authority=store._test_gold_labels,
        artifact_lineage=store._test_artifact_lineage,
    )


class _Attestor:
    receipt_rows = None

    def attest(self, payload):
        result = execute_routing_experiment_attestation_v2(payload)
        receipt = _signed_receipt(
            purpose="research_lab.routing_experiment.v2",
            input_root=result["input_root"],
            output_root=result["output_root"],
            index=101,
        )
        assert self.receipt_rows is not None
        self.receipt_rows.append(
            {"receipt_hash": receipt["receipt_hash"], "receipt_doc": receipt}
        )
        return {
            "result": result,
            "receipt": receipt,
        }


def test_model_transition_load_compares_artifact_after_logical_lookup():
    experiment_hash = _hash("a")
    marker = _model_transition_marker()
    client = _Client(
        {},
        rpc_results={
            "research_lab_routing_load_model_transition_v2": marker,
        },
    )
    store = SupabaseRoutingExperimentStore(client)
    identity = {
        "experiment_hash": experiment_hash,
        "variant_id": "baseline",
        "unit_ref": "unit-1",
        "idempotency_key": "c" * 64,
    }

    assert store.load_model_transition_marker(
        **identity, artifact_key=_artifact_key()
    ) == marker
    assert client.rpc_calls == [
        (
            "research_lab_routing_load_model_transition_v2",
            {
                "p_experiment_hash": experiment_hash,
                "p_variant_id": "baseline",
                "p_unit_ref": "unit-1",
                "p_idempotency_key": "c" * 64,
            },
        )
    ]
    with pytest.raises(
        RoutingExperimentStoreError,
        match="artifact identity differs",
    ):
        store.load_model_transition_marker(
            **identity, artifact_key=_artifact_key("9")
        )


def test_model_transition_load_rejects_legacy_identityless_v1_marker():
    experiment_hash = _hash("a")
    marker = _model_transition_marker()
    marker["event_schema_version"] = (
        "leadpoet.research_lab.model_transition.v1"
    )
    marker.pop("artifact_key")
    store = SupabaseRoutingExperimentStore(
        _Client(
            {},
            rpc_results={
                "research_lab_routing_load_model_transition_v2": marker,
            },
        )
    )

    with pytest.raises(
        RoutingExperimentStoreError,
        match="legacy or unknown",
    ):
        store.load_model_transition_marker(
            experiment_hash=experiment_hash,
            variant_id="baseline",
            unit_ref="unit-1",
            idempotency_key="c" * 64,
            artifact_key=_artifact_key(),
        )


def test_model_transition_load_accepts_exact_rpc_null_without_table_scan():
    experiment_hash = _hash("a")
    client = _Client(
        {},
        rpc_results={
            "research_lab_routing_load_model_transition_v2": None,
        },
    )
    store = SupabaseRoutingExperimentStore(client)

    assert store.load_model_transition_marker(
        experiment_hash=experiment_hash,
        variant_id="baseline",
        unit_ref="unit-1",
        idempotency_key="c" * 64,
        artifact_key=_artifact_key(),
    ) is None


def _event_claim(experiment_hash: str) -> RoutingExperimentExecutionClaim:
    claim_key = _hash("b")
    return RoutingExperimentExecutionClaim(
        experiment_hash=experiment_hash,
        claim_key=claim_key,
        claim_generation=1,
        claim_fence_hash=routing_claim_fence_hash_v3(
            experiment_hash=experiment_hash,
            claim_key=claim_key,
            claim_generation=1,
        ),
    )


@pytest.mark.parametrize(
    "result_kind",
    ("empty", "wrong_hash", "non_boolean", "extra_field"),
)
def test_append_event_rejects_unconfirmed_durable_ack(result_kind):
    experiment_hash = _hash("a")
    event_doc = {"event_schema_version": "fixture.v1"}
    document = {
        "schema_version": "leadpoet.research_lab.routing_event.v2",
        **event_doc,
    }
    expected_hash = routing_store_module._event_hash(
        "run_started", document
    )
    results = {
        "empty": {},
        "wrong_hash": {
            "event_hash": _hash("9"),
            "idempotent": False,
        },
        "non_boolean": {
            "event_hash": expected_hash,
            "idempotent": "false",
        },
        "extra_field": {
            "event_hash": expected_hash,
            "idempotent": False,
            "extra": True,
        },
    }
    store = SupabaseRoutingExperimentStore(
        _Client(
            {},
            rpc_results={
                "research_lab_routing_append_fenced_event_v3": results[
                    result_kind
                ],
            },
        )
    )

    with pytest.raises(
        RoutingExperimentStoreError,
        match="routing event result is malformed",
    ):
        store.append_event(
            experiment_hash=experiment_hash,
            event_type="run_started",
            event_doc=event_doc,
            claim=_event_claim(experiment_hash),
        )


@pytest.mark.parametrize("idempotent", (False, True))
def test_append_event_accepts_exact_durable_ack(idempotent):
    experiment_hash = _hash("a")
    event_doc = {"event_schema_version": "fixture.v1"}
    document = {
        "schema_version": "leadpoet.research_lab.routing_event.v2",
        **event_doc,
    }
    expected_hash = routing_store_module._event_hash(
        "run_started", document
    )
    store = SupabaseRoutingExperimentStore(
        _Client(
            {},
            rpc_results={
                "research_lab_routing_append_fenced_event_v3": {
                    "event_hash": expected_hash,
                    "idempotent": idempotent,
                },
            },
        )
    )

    assert store.append_event(
        experiment_hash=experiment_hash,
        event_type="run_started",
        event_doc=event_doc,
        claim=_event_claim(experiment_hash),
    ) == {"event_hash": expected_hash, "idempotent": idempotent}


def test_reconciliation_recomputes_exact_durable_roots_before_attestation():
    store, spec, evaluation = _store()
    value = _reconcile(store, spec, evaluation, _Attestor())
    assert value["reconciled"] is True
    assert value["authority_receipt_hash"].startswith("sha256:")


def test_reconciliation_rejects_extra_or_fabricated_authoritative_rows():
    store, spec, evaluation = _store(extra_attempt=True)
    with pytest.raises(RoutingExperimentStoreError, match="incomplete or non-authoritative"):
        _reconcile(store, spec, evaluation, _Attestor())

    class _ForgedAttestor:
        def attest(self, payload):
            result = execute_routing_experiment_attestation_v2(payload)
            result["budget_events_root"] = _hash("7")
            return {
                "result": result,
                "receipt": {
                    "receipt_hash": _hash("9"),
                    "role": "gateway_scoring",
                    "purpose": "research_lab.routing_experiment.v2",
                    "receipt_status": "succeeded",
                    "input_root": result["input_root"],
                    "output_root": result["output_root"],
                },
            }

    store, spec, evaluation = _store()
    with pytest.raises(RoutingExperimentStoreError, match="not authoritative"):
        _reconcile(store, spec, evaluation, _ForgedAttestor())


@pytest.mark.parametrize("budget_event_type", ["reserve", "uncertain", "recover"])
def test_reconciliation_rejects_every_unsettled_live_budget_head(budget_event_type):
    store, spec, evaluation = _store(
        live=True,
        budget_event_type=budget_event_type,
    )
    with pytest.raises((RoutingExperimentStoreError, ValueError), match="budget"):
        _reconcile(store, spec, evaluation, _Attestor())


def test_reconciliation_uses_canonical_event_identity_not_input_order():
    fixture = authority_fixture()
    store, spec, evaluation = _store(budget_rows=list(reversed(fixture["budgets"])))
    assert _reconcile(store, spec, evaluation, _Attestor())["reconciled"] is True


def test_protected_attempt_rejects_missing_standard_terminal_receipt_before_rpc():
    fixture = authority_fixture()
    attempt = fixture["attempts"][0]
    document = attempt["attempt_doc"]
    authorization = RoutingProviderCallAuthorizationV2.from_mapping(
        document["call_grant"]
    )
    receipt = ProviderReceipt.from_mapping(document["provider_receipt"])
    store, _spec, _evaluation = _store()
    with pytest.raises(
        RoutingExperimentStoreError, match="terminal execution receipt is unavailable"
    ):
        store.append_protected_provider_attempt(
            experiment_hash=attempt["experiment_hash"],
            key=attempt["attempt_key"],
            receipt=receipt,
            variant_id=attempt["variant_id"],
            reservation_id=attempt["reservation_id"],
            action_id=attempt["action_id"],
            authorization=authorization,
            authorization_proof_hash=attempt["authorization_proof_hash"],
            authorization_request_hash=document["call_grant_receipt"]["input_root"],
            authorization_receipt=document["call_grant_receipt"],
            terminal_result=document.get("terminal_result") or {},
            terminal_execution_receipt=None,
            protected_release_receipt=document["protected_release_receipt"],
            admission_bundle=document["admission_bundle"],
            claim=RoutingExperimentExecutionClaim(
                attempt["experiment_hash"],
                "sha256:" + "b" * 64,
                1,
                routing_claim_fence_hash_v3(
                    experiment_hash=attempt["experiment_hash"],
                    claim_key="sha256:" + "b" * 64,
                    claim_generation=1,
                ),
            ),
            billing_state="known",
            authoritative_billed_credit_microunits=receipt.credit_microunits,
        )


def test_both_v3_append_payloads_send_the_named_authorization_request_hash():
    legacy_source = inspect.getsource(
        routing_store_module.SupabaseRoutingExperimentStore.append_provider_attempt
    )
    protected_source = inspect.getsource(
        routing_store_module.SupabaseRoutingExperimentStore.append_protected_provider_attempt
    )
    assert '"p_authorization_request_hash": authorization_request_hash' in legacy_source
    assert '"p_authorization_request_hash": authorization_request_hash' in protected_source
    assert '"authorization_request_hash": authorization_request_hash' in legacy_source
    assert '"authorization_request_hash": authorization_request_hash' in protected_source
