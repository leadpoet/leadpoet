import asyncio
import base64
import threading
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from gateway.research_lab.attested_scoring_v2 import (
    AttestedScoringV2Error,
    _build_transport_payload_document,
    _canonical_array_document_size,
    _compact_parent_graphs_for_transport,
    execute_scoring_v2,
)
from gateway.research_lab import attested_scoring_v2
from gateway.research_lab import attested_v2_store
from gateway.tee import execution_job_manager_v2
from gateway.tee import release_lineage_v2
from gateway.research_lab.attested_v2_store import (
    AttestedV2StoreError,
    _execution_result_storage_row_v2,
)
from gateway.tee.execution_job_manager_v2 import (
    ExecutionJobManagerV2,
    ExecutionJobV2Error,
    ExecutionResultV2,
    PARENT_RECEIPT_GRAPH_SET_FIELD,
    PARENT_RECEIPT_GRAPHS_FIELD,
    unpack_parent_receipt_graph_set_v2,
)
from Leadpoet.utils.subnet_epoch import CUTOVER_PATH_ENV
from gateway.tee.coordinator_executor_v2 import (
    COORDINATOR_OPERATIONS_V2,
    CoordinatorExecutorV2,
    coordinator_failed_parent_graph_policy_v2,
    coordinator_receipt_output_v2,
)
from gateway.tee.release_manifest_v2 import (
    BUILD_EVIDENCE_SCHEMA_VERSION,
    build_release_manifest,
)
from gateway.tee.scoring_executor_v2 import SCORING_OPERATIONS_V2
from gateway.tee.topology import ROLE_SPECS, topology_hash
from leadpoet_canonical.attested_v2 import (
    CHECKPOINTED_RECEIPT_GRAPH_SCHEMA_VERSION,
    build_boot_identity_body,
    build_checkpointed_receipt_graph,
    build_receipt_graph,
    build_transport_attempt,
    create_boot_identity,
    sha256_bytes,
    sha256_json,
    transport_root,
    validate_receipt_graph,
)
from leadpoet_canonical.ancestry_checkpoint_v2 import (
    ANCESTRY_DELTA_SCHEMA_VERSION,
    ANCESTRY_CHECKPOINT_BOOTSTRAP_REQUEST_SCHEMA_VERSION,
    build_compact_ancestry_proof_from_delta_v2,
    issue_ancestry_certificate_v2,
)


def _hash(character):
    return "sha256:" + character * 64


@pytest.fixture(autouse=True)
def _checkpoint_runtime(monkeypatch):
    monkeypatch.setenv(
        CUTOVER_PATH_ENV,
        str(
            Path(__file__).resolve().parents[1]
            / "config"
            / "stateful-epoch-cutover-sn71.json"
        ),
    )

    async def load_proofs(*args, **kwargs):
        return {}

    async def persist_proof(proof, *, checkpointed_graph, **kwargs):
        assert checkpointed_graph["root_receipt_hash"] == (
            proof["certificate"]["claim"]["output_root_receipt_hash"]
        )
        return {
            "root_receipt_hash": checkpointed_graph["root_receipt_hash"],
            "proof_hash": proof["proof_hash"],
        }

    monkeypatch.setattr(
        attested_v2_store,
        "load_ancestry_checkpoint_proofs_v2",
        load_proofs,
    )
    monkeypatch.setattr(
        attested_v2_store,
        "persist_ancestry_checkpoint_v2",
        persist_proof,
    )


@pytest.mark.asyncio
async def test_receipt_graph_merge_does_not_block_event_loop(monkeypatch):
    started = threading.Event()
    release = threading.Event()
    expected = {"root_receipt_hash": _hash("a")}

    def merge(**kwargs):
        assert kwargs == {"marker": "large-graph"}
        started.set()
        assert release.wait(timeout=2)
        return expected

    monkeypatch.setattr(attested_scoring_v2, "_merge_graphs", merge)
    task = asyncio.create_task(
        attested_scoring_v2._merge_graphs_async(marker="large-graph")
    )
    assert await asyncio.to_thread(started.wait, 1)
    try:
        await asyncio.wait_for(asyncio.sleep(0), timeout=0.2)
        assert task.done() is False
    finally:
        release.set()

    assert await asyncio.wait_for(task, timeout=1) is expected


@pytest.mark.asyncio
async def test_receipt_graph_validation_does_not_block_event_loop(monkeypatch):
    started = threading.Event()
    release = threading.Event()

    def validate(graph, **kwargs):
        assert graph == {"root_receipt_hash": _hash("b")}
        assert kwargs == {"required_purposes": ("purpose",)}
        started.set()
        assert release.wait(timeout=2)

    monkeypatch.setattr(attested_scoring_v2, "validate_receipt_graph", validate)
    task = asyncio.create_task(
        attested_scoring_v2._validate_receipt_graph_async(
            {"root_receipt_hash": _hash("b")},
            required_purposes=("purpose",),
        )
    )
    assert await asyncio.to_thread(started.wait, 1)
    try:
        await asyncio.wait_for(asyncio.sleep(0), timeout=0.2)
        assert task.done() is False
    finally:
        release.set()

    assert await asyncio.wait_for(task, timeout=1) is None


def test_parent_graph_transport_compaction_preserves_all_declared_roots():
    parent = {
        "root_receipt_hash": _hash("a"),
        "receipts": [{"receipt_hash": _hash("a")}],
    }
    descendant = {
        "root_receipt_hash": _hash("b"),
        "receipts": [
            {"receipt_hash": _hash("a")},
            {"receipt_hash": _hash("b")},
        ],
    }
    independent = {
        "root_receipt_hash": _hash("c"),
        "receipts": [{"receipt_hash": _hash("c")}],
    }

    compacted = _compact_parent_graphs_for_transport(
        (parent, descendant, independent)
    )

    assert compacted == [descendant, independent]
    covered = {
        receipt["receipt_hash"]
        for graph in compacted
        for receipt in graph["receipts"]
    }
    assert {_hash("a"), _hash("b"), _hash("c")} <= covered


def test_parent_graph_transport_compacts_production_sized_declared_set():
    roots = ["sha256:" + format(index, "064x") for index in range(215)]
    parent_graphs = [
        {
            "root_receipt_hash": root,
            "receipts": [
                {"receipt_hash": ancestor}
                for ancestor in roots[: index + 1]
            ],
        }
        for index, root in enumerate(roots)
    ]

    compacted = _compact_parent_graphs_for_transport(parent_graphs)

    assert len(compacted) == 1
    assert {
        receipt["receipt_hash"] for receipt in compacted[0]["receipts"]
    } == set(roots)


def test_oversized_shared_ancestry_uses_exact_deduplicated_graph_set(
    monkeypatch,
):
    monkeypatch.setattr(attested_scoring_v2, "MAX_INPUT_BYTES", 1)
    shared = {
        "receipt_hash": _hash("a"),
        "marker": "shared" * 1024,
    }
    boot = {
        "boot_identity_hash": _hash("b"),
        "marker": "shared" * 1024,
    }
    graphs = [
        {
            "schema_version": "leadpoet.attested_receipt_graph.v2",
            "root_receipt_hash": root,
            "boot_identities": [boot],
            "receipts": [shared, {"receipt_hash": root, "marker": root}],
            "transport_attempts": [],
            "host_operations": [],
        }
        for root in (_hash("c"), _hash("d"))
    ]

    document, evidence = _build_transport_payload_document(
        payload={"epoch": 24_279},
        parent_graphs=graphs,
    )

    assert PARENT_RECEIPT_GRAPHS_FIELD not in document
    assert evidence["encoding"] == "receipt_graph_set"
    assert evidence["transport_size_bytes"] < evidence["legacy_size_bytes"]
    assert evidence["unique_receipt_count"] == 3
    assert unpack_parent_receipt_graph_set_v2(
        document[PARENT_RECEIPT_GRAPH_SET_FIELD]
    ) == graphs


def test_small_parent_graph_payload_keeps_legacy_encoding():
    graph = {
        "schema_version": "leadpoet.attested_receipt_graph.v2",
        "root_receipt_hash": _hash("a"),
        "boot_identities": [],
        "receipts": [{"receipt_hash": _hash("a")}],
        "transport_attempts": [],
        "host_operations": [],
    }

    document, evidence = _build_transport_payload_document(
        payload={"epoch": 24_279},
        parent_graphs=(graph,),
    )

    assert document[PARENT_RECEIPT_GRAPHS_FIELD] == [graph]
    assert PARENT_RECEIPT_GRAPH_SET_FIELD not in document
    assert evidence["encoding"] == "receipt_graphs"
    assert evidence["legacy_size_bytes"] == len(
        attested_scoring_v2._canonical_bytes(document)
    )


def test_transport_builder_requires_explicit_allocation_graph_bound(monkeypatch):
    monkeypatch.setattr(attested_scoring_v2, "MAX_INPUT_BYTES", 1)
    graphs = [
        {
            "schema_version": "leadpoet.attested_receipt_graph.v2",
            "root_receipt_hash": "sha256:" + format(index, "064x"),
            "boot_identities": [],
            "receipts": [
                {"receipt_hash": "sha256:" + format(index, "064x")}
            ],
            "transport_attempts": [],
            "host_operations": [],
        }
        for index in range(129)
    ]

    with pytest.raises(
        ExecutionJobV2Error,
        match="external receipt graph count exceeds limit",
    ):
        _build_transport_payload_document(
            payload={"epoch": 24_294},
            parent_graphs=graphs,
        )

    document, evidence = _build_transport_payload_document(
        payload={"epoch": 24_294},
        parent_graphs=graphs,
        max_parent_graph_count=(
            execution_job_manager_v2.MAX_ALLOCATION_ANCESTRY_AUTHORITIES
        ),
    )
    assert evidence["parent_graph_count"] == 129
    assert (
        PARENT_RECEIPT_GRAPHS_FIELD in document
        or PARENT_RECEIPT_GRAPH_SET_FIELD in document
    )


def test_parent_graph_streaming_size_matches_exact_canonical_document():
    graphs = [
        {
            "schema_version": "leadpoet.attested_receipt_graph.v2",
            "root_receipt_hash": root,
            "boot_identities": [],
            "receipts": [{"receipt_hash": root}],
            "transport_attempts": [],
            "host_operations": [],
        }
        for root in (_hash("a"), _hash("b"), _hash("c"))
    ]
    document = {
        "epoch": 24_279,
        "_v2_provider_credential_profile": "benchmark_model",
    }
    exact = {
        **document,
        PARENT_RECEIPT_GRAPHS_FIELD: graphs,
    }

    assert _canonical_array_document_size(
        document,
        field=PARENT_RECEIPT_GRAPHS_FIELD,
        values=graphs,
    ) == len(attested_scoring_v2._canonical_bytes(exact))


def test_oversized_graph_without_deduplication_benefit_keeps_ancestry(
    monkeypatch,
):
    monkeypatch.setattr(attested_scoring_v2, "MAX_INPUT_BYTES", 1)
    graph = {
        "schema_version": "leadpoet.attested_receipt_graph.v2",
        "root_receipt_hash": _hash("a"),
        "boot_identities": [],
        "receipts": [{"receipt_hash": _hash("a")}],
        "transport_attempts": [],
        "host_operations": [],
    }

    document, evidence = _build_transport_payload_document(
        payload={"epoch": 24_279},
        parent_graphs=(graph,),
    )

    assert evidence["encoding"] == "receipt_graphs"
    assert document[PARENT_RECEIPT_GRAPHS_FIELD] == [graph]
    assert PARENT_RECEIPT_GRAPH_SET_FIELD not in document


def _release(commit_character="1"):
    rows = []
    for index, (role, spec) in enumerate(sorted(ROLE_SPECS.items())):
        character = "abcdef0123456789"[index]
        values = {
            "commit_sha": commit_character * 40,
            "pcr0": character * 96,
            "normalized_image_hash": _hash(character),
            "eif_hash": _hash(character),
            "source_manifest_hash": _hash("2"),
            "build_identity_hash": _hash(character),
            "execution_manifest_hash": _hash(character),
            "dependency_lock_hash": _hash("3"),
            "dockerfile_hash": _hash("4"),
            "topology_hash": topology_hash(),
        }
        for domain in ("gateway", "validator"):
            for ordinal in (1, 2, 3):
                rows.append(
                    {
                        "schema_version": BUILD_EVIDENCE_SCHEMA_VERSION,
                        "builder_domain": domain,
                        "builder_id": domain + "-parent",
                        "build_ordinal": ordinal,
                        "physical_role": role,
                        "service_role": spec["service_role"],
                        **values,
                    }
                )
    return build_release_manifest(
        rows, acceptance_signer_pubkey_hash="sha256:" + "f" * 64
    )


class _Client:
    def __init__(self, release, *, executor=None, configured_worker_count=1):
        role = "gateway_scoring"
        summary = release["roles"][role]
        self.key = Ed25519PrivateKey.generate()
        pubkey = self.key.public_key().public_bytes(
            serialization.Encoding.Raw,
            serialization.PublicFormat.Raw,
        ).hex()
        body = build_boot_identity_body(
            role="gateway_scoring",
            physical_role=role,
            commit_sha=summary["commit_sha"],
            pcr0=summary["pcr0"],
            build_manifest_hash=summary["execution_manifest_hash"],
            dependency_lock_hash=summary["dependency_lock_hash"],
            config_hash=_hash("9"),
            boot_nonce="a" * 32,
            signing_pubkey=pubkey,
            transport_pubkey="b" * 64,
            transport_certificate_hash=_hash("c"),
            attestation_user_data_hash=_hash("d"),
            issued_at="2026-07-10T00:00:00Z",
        )
        self.boot = create_boot_identity(
            body=body,
            attestation_document_b64=base64.b64encode(b"nitro").decode(),
        )
        self.manager = ExecutionJobManagerV2(
            boot_identity_supplier=lambda: self.boot,
            sign_digest=self.key.sign,
            operations=SCORING_OPERATIONS_V2,
            executor=executor
            or (
                lambda operation, payload, context: {
                    "operation": operation,
                    "echo": payload,
                }
            ),
            worker_count=1,
            configured_worker_count=configured_worker_count,
            ancestry_lineage_id=attested_scoring_v2._gateway_ancestry_lineage_id(),
            ancestry_boot_attestation_verifier=lambda identity: identity,
            ancestry_allowed_issuer_roles=(
                "gateway_autoresearch",
                "gateway_coordinator",
                "gateway_scoring",
            ),
        )

    async def scoring_v2_health(self):
        return self.manager.health()

    async def v2_get_boot_identity(self):
        return self.boot

    async def scoring_v2_submit_job(self, manifest):
        return self.manager.submit(manifest)

    async def scoring_v2_put_chunk(self, *, job_id, offset, data):
        import hashlib

        return self.manager.put_chunk(
            job_id=job_id,
            offset=offset,
            data_b64=base64.b64encode(data).decode(),
            chunk_sha256="sha256:" + hashlib.sha256(data).hexdigest(),
        )

    async def scoring_v2_seal_job(self, job_id):
        return self.manager.seal(job_id)

    async def scoring_v2_get_status(self, job_id):
        return self.manager.status(job_id)

    async def scoring_v2_cancel_job(self, job_id):
        return self.manager.cancel(job_id)

    async def scoring_v2_get_result(self, job_id, *, offset=0):
        return self.manager.result_chunk(job_id=job_id, offset=offset)

    async def scoring_v2_get_receipt(self, job_id):
        return self.manager.receipt(job_id)

    async def scoring_v2_get_receipts(self, job_id):
        return list(self.manager.receipts(job_id))

    async def scoring_v2_get_transport_attempts(self, job_id):
        return list(self.manager.transport_attempts(job_id))

    async def scoring_v2_get_host_operations(self, job_id):
        return list(self.manager.host_operations(job_id))

    async def scoring_v2_get_ancestry_compact_proof(self, job_id):
        return self.manager.ancestry_compact_proof(job_id)

    async def scoring_v2_get_artifact_hashes(self, job_id):
        return list(self.manager.artifact_hashes(job_id))

    async def scoring_v2_get_transitions(self, job_id):
        return list(self.manager.transitions(job_id))


class _CoordinatorClient(_Client):
    def __init__(self, release, *, executor=None):
        super().__init__(release)
        role = "gateway_coordinator"
        summary = release["roles"][role]
        pubkey = self.key.public_key().public_bytes(
            serialization.Encoding.Raw,
            serialization.PublicFormat.Raw,
        ).hex()
        body = build_boot_identity_body(
            role=role,
            physical_role=role,
            commit_sha=summary["commit_sha"],
            pcr0=summary["pcr0"],
            build_manifest_hash=summary["execution_manifest_hash"],
            dependency_lock_hash=summary["dependency_lock_hash"],
            config_hash=_hash("9"),
            boot_nonce="e" * 32,
            signing_pubkey=pubkey,
            transport_pubkey="f" * 64,
            transport_certificate_hash=_hash("a"),
            attestation_user_data_hash=_hash("b"),
            issued_at="2026-07-10T00:00:00Z",
        )
        self.boot = create_boot_identity(
            body=body,
            attestation_document_b64=base64.b64encode(b"nitro").decode(),
        )
        self.manager = ExecutionJobManagerV2(
            boot_identity_supplier=lambda: self.boot,
            sign_digest=self.key.sign,
            operations=COORDINATOR_OPERATIONS_V2,
            executor=executor
            or (
                lambda operation, payload, context: {
                    "operation": operation,
                    "echo": payload,
                }
            ),
            worker_count=1,
            configured_worker_count=0,
            ancestry_lineage_id=attested_scoring_v2._gateway_ancestry_lineage_id(),
            ancestry_boot_attestation_verifier=lambda identity: identity,
            ancestry_allowed_issuer_roles=(
                "gateway_autoresearch",
                "gateway_coordinator",
                "gateway_scoring",
            ),
        )

    coordinator_v2_health = _Client.scoring_v2_health
    coordinator_v2_submit_job = _Client.scoring_v2_submit_job
    coordinator_v2_put_chunk = _Client.scoring_v2_put_chunk
    coordinator_v2_seal_job = _Client.scoring_v2_seal_job
    coordinator_v2_get_status = _Client.scoring_v2_get_status
    coordinator_v2_cancel_job = _Client.scoring_v2_cancel_job
    coordinator_v2_get_result = _Client.scoring_v2_get_result
    coordinator_v2_get_receipt = _Client.scoring_v2_get_receipt
    coordinator_v2_get_receipts = _Client.scoring_v2_get_receipts
    coordinator_v2_get_transport_attempts = _Client.scoring_v2_get_transport_attempts
    coordinator_v2_get_host_operations = _Client.scoring_v2_get_host_operations
    coordinator_v2_get_ancestry_compact_proof = (
        _Client.scoring_v2_get_ancestry_compact_proof
    )
    coordinator_v2_get_artifact_hashes = _Client.scoring_v2_get_artifact_hashes
    coordinator_v2_get_transitions = _Client.scoring_v2_get_transitions


class _ConcurrentSubmitClient(_Client):
    def __init__(self, release, *, callers, executor):
        super().__init__(release, executor=executor)
        self._callers = callers
        self._submit_count = 0
        self._all_submitted = asyncio.Event()

    async def scoring_v2_submit_job(self, manifest):
        summary = self.manager.submit(manifest)
        self._submit_count += 1
        if self._submit_count == self._callers:
            self._all_submitted.set()
        await asyncio.wait_for(self._all_submitted.wait(), timeout=1)
        return summary


@pytest.mark.asyncio
async def test_v2_bridge_returns_only_durable_release_verified_result():
    release = _release()
    client = _Client(release, configured_worker_count=13)
    persisted = []

    def load_profile(
        profile,
        *,
        execution_role,
        worker_index,
        require_egress_proxy,
    ):
        assert profile == "default"
        assert execution_role == "gateway_scoring"
        assert worker_index == 12
        assert require_egress_proxy is False
        return {
            "profile": profile,
            "credential_ref_hashes": {},
            "envelopes": [],
        }

    async def persist(graph):
        validate_receipt_graph(graph)
        persisted.append(graph)
        return {"root_receipt_hash": graph["root_receipt_hash"]}

    result = await execute_scoring_v2(
        operation="benchmark_icp_score",
        purpose="research_lab.benchmark.v2",
        epoch_id=12,
        sequence=0,
        payload={"scores": [1.0, 2.0]},
        worker_index=12,
        provider_profile_loader=load_profile,
        release_manifest=release,
        client=client,
        persist_graph=persist,
        boot_verifier=lambda identity: identity,
        poll_seconds=0.001,
    )
    assert result["result"] == {
        "operation": "benchmark_icp_score",
        "echo": {"scores": [1.0, 2.0]},
    }
    assert result["status"] == "succeeded"
    assert result["physical_role"] == "gateway_scoring"
    assert result["execution_receipt"] == result["receipt"]
    assert result["execution_receipt_graph"] == result["receipt_graph"]
    assert persisted[0]["root_receipt_hash"] == result["receipt"]["receipt_hash"]


@pytest.mark.asyncio
async def test_same_manifest_concurrent_callers_converge_on_one_execution():
    release = _release()
    execution_count = 0

    def executor(operation, payload, context):
        nonlocal execution_count
        execution_count += 1
        for digest in (_hash("8"), _hash("9"), _hash("6")):
            context.record_artifact(digest)
        return ExecutionResultV2(
            output={"operation": operation, "echo": payload},
            transport_attempts=(_authenticated_attempt(context),),
        )

    caller_count = 5
    client = _ConcurrentSubmitClient(
        release,
        callers=caller_count,
        executor=executor,
    )
    artifact_client = _ArtifactCoordinator(
        release,
        (_hash("8"), _hash("6")),
    )

    async def persist(graph, **_kwargs):
        validate_receipt_graph(graph)
        return {"root_receipt_hash": graph["root_receipt_hash"]}

    async def persist_artifact(artifact_id, **kwargs):
        return {
            "status": "persisted",
            "artifact_id": artifact_id,
            "storage_document_hash": _hash("d"),
            "artifact_kind": "provider_response",
            "artifact_hash": _hash("c"),
            "encryption_context_hash": _hash("e"),
            "object_lock_mode": "COMPLIANCE",
            "retain_until": "2027-07-10T12:00:00Z",
            "transport_root": transport_root(
                _storage_attempts_for_job(
                    artifact_id,
                    kwargs["attestation_job_id"],
                )
            ),
        }

    async def persist_sidecars(**_kwargs):
        return {"artifact_link_count": 2, "transition_count": 0}

    async def run_once():
        return await execute_scoring_v2(
            operation="benchmark_icp_score",
            purpose="research_lab.benchmark.v2",
            epoch_id=12,
            sequence=0,
            payload={"scores": [1.0, 2.0]},
            worker_index=0,
            release_manifest=release,
            client=client,
            artifact_coordinator_client=artifact_client,
            persist_artifact=persist_artifact,
            artifact_bucket="immutable-bucket",
            persist_graph=persist,
            persist_sidecars=persist_sidecars,
            boot_verifier=lambda identity: identity,
            poll_seconds=0.001,
        )

    results = await asyncio.gather(*(run_once() for _ in range(caller_count)))

    assert execution_count == 1
    assert {item["receipt"]["receipt_hash"] for item in results} == {
        results[0]["receipt"]["receipt_hash"]
    }
    assert all(
        item["receipt"]["receipt_hash"]
        != item["execution_receipt"]["receipt_hash"]
        for item in results
    )
    assert all(item["result"] == results[0]["result"] for item in results)


@pytest.mark.asyncio
async def test_v2_bridge_retains_checkpoint_parent_for_business_logic_and_extends_it():
    release = _release()
    observed_parent_graphs = []

    def executor(_operation, payload, context):
        if payload["generation"] == 1:
            assert context.external_receipt_graphs == []
            assert context.external_ancestry_proofs == []
            return {"generation": 1}

        parent_root = str(payload["parent_root"])
        matching_receipts = [
            dict(receipt)
            for graph in context.external_receipt_graphs
            for receipt in graph.get("receipts") or ()
            if receipt.get("receipt_hash") == parent_root
        ]
        assert len(context.external_receipt_graphs) == 1
        assert (
            context.external_receipt_graphs[0]["schema_version"]
            == CHECKPOINTED_RECEIPT_GRAPH_SCHEMA_VERSION
        )
        assert context.external_ancestry_proofs == []
        assert len(matching_receipts) == 1
        assert matching_receipts[0]["status"] == "succeeded"
        observed_parent_graphs.append(dict(context.external_receipt_graphs[0]))
        return {"generation": 2, "observed_parent_root": parent_root}

    client = _Client(release, executor=executor)

    async def persist(graph, **_kwargs):
        return {"root_receipt_hash": graph["root_receipt_hash"]}

    common = {
        "operation": "benchmark_icp_score",
        "purpose": "research_lab.benchmark.v2",
        "worker_index": 0,
        "release_manifest": release,
        "client": client,
        "persist_graph": persist,
        "boot_verifier": lambda identity: identity,
        "poll_seconds": 0.001,
    }
    first = await execute_scoring_v2(
        **common,
        epoch_id=12,
        sequence=0,
        payload={"generation": 1},
    )
    first_graph = first["receipt_graph"]
    first_proof = first["ancestry_compact_proof"]
    assert (
        first_graph["schema_version"]
        == CHECKPOINTED_RECEIPT_GRAPH_SCHEMA_VERSION
    )

    second = await execute_scoring_v2(
        **common,
        epoch_id=13,
        sequence=0,
        payload={
            "generation": 2,
            "parent_root": first_graph["root_receipt_hash"],
        },
        parent_graphs=(first_graph,),
        parent_ancestry_proofs=(first_proof,),
    )

    assert observed_parent_graphs == [first_graph]
    assert second["result"] == {
        "generation": 2,
        "observed_parent_root": first_graph["root_receipt_hash"],
    }
    first_claim = first_proof["certificate"]["claim"]
    second_claim = second["ancestry_compact_proof"]["certificate"]["claim"]
    assert second_claim["certificate_sequence"] == (
        first_claim["certificate_sequence"] + 1
    )
    assert second_claim["parent_authorities"] == [
        {
            "schema_version": "leadpoet.attested_ancestry_parent_authority.v2",
            "authority_kind": "certificate",
            "parent_receipt_hash": first_graph["root_receipt_hash"],
            "parent_epoch_id": first_claim["local_delta_projection"][
                "root_epoch_id"
            ],
            "parent_role": first_claim["local_delta_projection"]["root_role"],
            "parent_purpose": first_claim["local_delta_projection"][
                "root_purpose"
            ],
            "authority_hash": first_proof["certificate"]["certificate_hash"],
            "authority_policy_hash": first_claim["policy"]["policy_hash"],
            "authority_sequence": first_claim["certificate_sequence"],
            "authority_purposes": first_claim["local_delta_projection"][
                "ancestry_purposes"
            ],
        }
    ]


@pytest.mark.asyncio
async def test_v2_bridge_loads_historical_checkpoint_issuer_release(
    monkeypatch,
):
    current_release = _release("1")
    historical_release = _release("2")
    current = _Client(current_release)
    historical = _Client(historical_release)

    def verify_nitro(
        identity,
        *,
        expected_pcr0,
        certificate_validity_at_attestation_time,
    ):
        assert identity["pcr0"] == expected_pcr0
        assert certificate_validity_at_attestation_time is True
        return identity

    monkeypatch.setattr(
        attested_scoring_v2,
        "verify_boot_identity_nitro",
        verify_nitro,
    )
    monkeypatch.setattr(
        release_lineage_v2,
        "verify_boot_identity_nitro",
        verify_nitro,
    )

    async def persist(graph, **_kwargs):
        return {"root_receipt_hash": graph["root_receipt_hash"]}

    first = await execute_scoring_v2(
        operation="benchmark_icp_score",
        purpose="research_lab.benchmark.v2",
        epoch_id=12,
        sequence=0,
        payload={"generation": 1},
        worker_index=0,
        release_manifest=current_release,
        client=current,
        persist_graph=persist,
        boot_verifier=lambda identity: identity,
        poll_seconds=0.001,
    )
    first_graph = first["receipt_graph"]
    delta = {
        "schema_version": ANCESTRY_DELTA_SCHEMA_VERSION,
        "root_receipt_hash": first_graph["root_receipt_hash"],
        "boot_identities": first_graph["boot_identities"],
        "receipts": first_graph["receipts"],
        "transport_attempts": first_graph["transport_attempts"],
        "host_operations": first_graph["host_operations"],
    }
    lineage_id = attested_scoring_v2._gateway_ancestry_lineage_id()
    certificate = issue_ancestry_certificate_v2(
        local_delta=delta,
        lineage_id=lineage_id,
        certificate_sequence=0,
        issuer_boot_identity=historical.boot,
        issued_at="2026-07-10T20:00:00Z",
        sign_digest=historical.key.sign,
        boot_attestation_verifier=lambda identity: identity,
        allowed_issuer_roles=("gateway_scoring",),
        required_purposes=("research_lab.benchmark.v2",),
    )
    proof = build_compact_ancestry_proof_from_delta_v2(
        delta,
        certificate,
        expected_lineage_id=lineage_id,
        boot_attestation_verifier=lambda identity: identity,
        allowed_issuer_roles=("gateway_scoring",),
    )
    checkpointed = build_checkpointed_receipt_graph(
        root_receipt_hash=first_graph["root_receipt_hash"],
        boot_identities=first_graph["boot_identities"],
        receipts=first_graph["receipts"],
        transport_attempts=first_graph["transport_attempts"],
        host_operations=first_graph["host_operations"],
        ancestry_lineage_id=lineage_id,
        ancestry_proof=proof,
        boot_attestation_verifier=lambda identity: identity,
        require_boot_attestation_verification=True,
    )
    loaded_commits = []

    second = await execute_scoring_v2(
        operation="benchmark_icp_score",
        purpose="research_lab.benchmark.v2",
        epoch_id=13,
        sequence=0,
        payload={"generation": 2},
        worker_index=0,
        parent_graphs=(checkpointed,),
        release_manifest=current_release,
        release_channel_loader=lambda commit: loaded_commits.append(commit)
        or {"gateway_release_manifest": historical_release},
        client=current,
        persist_graph=persist,
        poll_seconds=0.001,
    )

    assert second["result"] == {
        "operation": "benchmark_icp_score",
        "echo": {"generation": 2},
    }
    assert loaded_commits == [historical_release["commit_sha"]]


@pytest.mark.asyncio
async def test_checkpoint_bootstrap_resume_proof_is_input_not_session_parent():
    release = _release()
    scoring = _Client(release)

    async def persist(graph, **_kwargs):
        return {"root_receipt_hash": graph["root_receipt_hash"]}

    common = {
        "operation": "benchmark_icp_score",
        "purpose": "research_lab.benchmark.v2",
        "worker_index": 0,
        "release_manifest": release,
        "client": scoring,
        "persist_graph": persist,
        "boot_verifier": lambda identity: identity,
        "poll_seconds": 0.001,
    }
    first = await execute_scoring_v2(
        **common,
        epoch_id=12,
        sequence=0,
        payload={"generation": 1},
    )
    second = await execute_scoring_v2(
        **common,
        epoch_id=13,
        sequence=0,
        payload={"generation": 2},
        parent_graphs=(first["receipt_graph"],),
        parent_ancestry_proofs=(first["ancestry_compact_proof"],),
    )
    legacy_graph = build_receipt_graph(
        root_receipt_hash=second["receipt"]["receipt_hash"],
        boot_identities=(scoring.boot,),
        receipts=(first["receipt"], second["receipt"]),
        transport_attempts=(),
    )
    request_schema = ANCESTRY_CHECKPOINT_BOOTSTRAP_REQUEST_SCHEMA_VERSION
    coordinator = _CoordinatorClient(
        release,
        executor=lambda _operation, payload, _context: ExecutionResultV2(
            output={
                "schema_version": request_schema,
                "selected_root_receipt_hashes": list(
                    payload["selected_root_receipt_hashes"]
                ),
            },
            ancestry_checkpoint_bootstrap=True,
        ),
    )
    async def persist_checkpoint(
        proof, *, checkpointed_graph, **_kwargs
    ):
        return {
            "root_receipt_hash": checkpointed_graph["root_receipt_hash"],
            "proof_hash": proof["proof_hash"],
        }

    bootstrap = await execute_scoring_v2(
        operation="ancestry_checkpoint_bootstrap_v2",
        purpose="research_lab.ancestry_checkpoint_bootstrap.v2",
        epoch_id=13,
        sequence=0,
        payload={
            "schema_version": request_schema,
            "selected_root_receipt_hashes": [
                legacy_graph["root_receipt_hash"]
            ],
        },
        worker_index=0,
        parent_graphs=(legacy_graph,),
        parent_ancestry_proofs=(first["ancestry_compact_proof"],),
        release_manifest=release,
        client=coordinator,
        persist_graph=persist,
        persist_ancestry_checkpoint=persist_checkpoint,
        boot_verifier=lambda identity: identity,
        operation_registry=COORDINATOR_OPERATIONS_V2,
        physical_role_override="gateway_coordinator",
        expected_service_role="gateway_coordinator",
        rpc_namespace="coordinator_v2",
        poll_seconds=0.001,
    )
    selected_root = legacy_graph["root_receipt_hash"]
    resume_root = first["receipt"]["receipt_hash"]
    assert bootstrap["receipt"]["parent_receipt_hashes"] == [
        selected_root
    ]
    assert resume_root not in bootstrap["receipt"][
        "parent_receipt_hashes"
    ]
    assert [
        item["parent_receipt_hash"]
        for item in bootstrap["ancestry_compact_proof"]["certificate"][
            "claim"
        ]["parent_authorities"]
    ] == [selected_root]


@pytest.mark.asyncio
async def test_v2_bridge_verifies_projected_receipt_output():
    release = _release()
    full_output = {
        "allocation": {"allocation_hash": _hash("7")},
        "source_state": {"epoch": 12},
    }
    receipt_output = {"allocation": full_output["allocation"]}
    client = _Client(
        release,
        executor=lambda _operation, _payload, _context: ExecutionResultV2(
            output=full_output,
            receipt_output=receipt_output,
        ),
    )

    async def persist(graph):
        return {"root_receipt_hash": graph["root_receipt_hash"]}

    result = await execute_scoring_v2(
        operation="benchmark_icp_score",
        purpose="research_lab.benchmark.v2",
        epoch_id=12,
        sequence=0,
        payload={"scores": [1.0]},
        worker_index=0,
        release_manifest=release,
        client=client,
        persist_graph=persist,
        boot_verifier=lambda identity: identity,
        poll_seconds=0.001,
        receipt_output_projector=lambda _operation, output: {
            "allocation": output["allocation"]
        },
    )

    assert result["result"] == full_output
    assert result["receipt"]["output_root"] == sha256_json(receipt_output)


@pytest.mark.asyncio
async def test_v2_bridge_accepts_measured_coordinator_internal_worker_capacity():
    release = _release()
    client = _CoordinatorClient(release)

    async def persist(graph):
        return {"root_receipt_hash": graph["root_receipt_hash"]}

    result = await execute_scoring_v2(
        operation="promotion_improvement",
        purpose="research_lab.ranking.v2",
        epoch_id=12,
        sequence=0,
        payload={"score_bundle": {}},
        worker_index=0,
        provider_profile_loader=lambda *args, **kwargs: {
            "profile": "default",
            "credential_ref_hashes": {},
            "envelopes": [],
        },
        release_manifest=release,
        client=client,
        persist_graph=persist,
        boot_verifier=lambda identity: identity,
        poll_seconds=0.001,
        operation_registry=COORDINATOR_OPERATIONS_V2,
        physical_role_override="gateway_coordinator",
        expected_service_role="gateway_coordinator",
        rpc_namespace="coordinator_v2",
    )

    assert result["status"] == "succeeded"
    assert result["physical_role"] == "gateway_coordinator"
    assert client.manager.health()["configured_worker_count"] == 0


@pytest.mark.asyncio
async def test_v2_bridge_accepts_chain_settlement_projected_receipt():
    release = _release()
    settlement_doc = {
        "schema_version": (
            "leadpoet.research_lab_chain_realized_epoch_settlement.v1"
        ),
        "netuid": 71,
        "epoch_id": 12,
        "credit_hashes": [],
        "observation_summary": {
            "schema_version": "leadpoet.chain_realized_observation_summary.v1",
            "complete": True,
        },
    }
    settlement_hash = sha256_json(settlement_doc)
    executor = CoordinatorExecutorV2(
        chain_realized_settlement_resolver=lambda _payload, _context: {
            "settlement_doc": settlement_doc,
            "settlement_hash": settlement_hash,
            "credits": [],
        }
    )
    client = _CoordinatorClient(release, executor=executor)

    async def load_missing(**_kwargs):
        return None

    async def persist(graph):
        return {"root_receipt_hash": graph["root_receipt_hash"]}

    async def persist_result(**kwargs):
        return _execution_result_storage_row_v2(**kwargs)

    result = await execute_scoring_v2(
        operation="attest_chain_realized_settlement_v1",
        purpose="research_lab.chain_realized_epoch_settlement.v1",
        epoch_id=12,
        sequence=1,
        payload={"epoch_id": 12},
        worker_index=0,
        provider_profile_loader=lambda *args, **kwargs: {
            "profile": "default",
            "credential_ref_hashes": {},
            "envelopes": [],
        },
        release_manifest=release,
        client=client,
        persist_graph=persist,
        load_replayable_result=load_missing,
        persist_replayable_result=persist_result,
        boot_verifier=lambda identity: identity,
        poll_seconds=0.001,
        operation_registry=COORDINATOR_OPERATIONS_V2,
        physical_role_override="gateway_coordinator",
        expected_service_role="gateway_coordinator",
        rpc_namespace="coordinator_v2",
        receipt_output_projector=coordinator_receipt_output_v2,
    )

    assert result["status"] == "succeeded"
    assert result["result"]["settlement_hash"] == settlement_hash
    assert result["receipt"]["output_root"] == settlement_hash


@pytest.mark.asyncio
async def test_v2_bridge_replays_exact_durable_coordinator_result_without_resubmit():
    release = _release()
    client = _CoordinatorClient(release)
    captured = {}

    async def load_missing(**_kwargs):
        return None

    async def persist(graph):
        validate_receipt_graph(graph)
        return {
            "root_receipt_hash": graph["root_receipt_hash"],
            "graph_hash": sha256_json(graph),
        }

    async def persist_result(**kwargs):
        captured.update(kwargs)
        return _execution_result_storage_row_v2(**kwargs)

    common = {
        "operation": "attest_weight_input",
        "purpose": "research_lab.champion_input.v2",
        "epoch_id": 12,
        "sequence": 1,
        "payload": {"category": "champions"},
        "worker_index": 0,
        "provider_profile_loader": lambda *args, **kwargs: {
            "profile": "default",
            "credential_ref_hashes": {},
            "envelopes": [],
        },
        "release_manifest": release,
        "persist_graph": persist,
        "boot_verifier": lambda identity: identity,
        "poll_seconds": 0.001,
        "operation_registry": COORDINATOR_OPERATIONS_V2,
        "physical_role_override": "gateway_coordinator",
        "expected_service_role": "gateway_coordinator",
        "rpc_namespace": "coordinator_v2",
        "receipt_output_projector": coordinator_receipt_output_v2,
    }
    fresh = await execute_scoring_v2(
        **common,
        client=client,
        load_replayable_result=load_missing,
        persist_replayable_result=persist_result,
    )
    assert captured["receipt"] == fresh["receipt"]
    assert captured["result"] == fresh["result"]

    async def load_replay(**kwargs):
        assert kwargs == {
            "role": "gateway_coordinator",
            "operation": "attest_weight_input",
            "purpose": "research_lab.champion_input.v2",
            "job_id": fresh["receipt"]["job_id"],
        }
        return {
            "row": {"release_hash": release["release_hash"]},
            "result": fresh["result"],
            "receipt": fresh["receipt"],
            "receipt_graph": fresh["receipt_graph"],
            "artifact_hashes": fresh["artifact_hashes"],
        }

    replay_client = _CoordinatorClient(release)

    async def reject_submit(_manifest):
        raise AssertionError("durable replay must not submit a second enclave job")

    replay_client.coordinator_v2_submit_job = reject_submit
    replayed = await execute_scoring_v2(
        **common,
        client=replay_client,
        load_replayable_result=load_replay,
        persist_replayable_result=persist_result,
    )
    assert replayed["replay_status"] == "durable_exact"
    assert replayed["result"] == fresh["result"]
    assert replayed["receipt"] == fresh["receipt"]
    assert replayed["execution_receipt"] == fresh["receipt"]
    assert replayed["execution_receipt_graph"] == fresh["receipt_graph"]

    tampered = dict(fresh["result"])
    tampered["operation"] = "tampered"
    with pytest.raises(AttestedV2StoreError, match="output differs"):
        _execution_result_storage_row_v2(
            operation="attest_weight_input",
            result=tampered,
            receipt=fresh["receipt"],
            artifact_hashes=fresh["artifact_hashes"],
            release_hash=release["release_hash"],
        )


@pytest.mark.asyncio
async def test_v2_bridge_replays_exact_active_model_authority_after_restart():
    release = _release()
    captured = {}
    artifact_identity = {
        "schema_version": "leadpoet.private_model_artifact_replay_identity.v2",
        "model_artifact_hash": _hash("5"),
        "manifest_hash": _hash("6"),
        "git_commit_sha": "7" * 40,
        "config_hash": _hash("8"),
        "component_registry_version": "components-v1",
        "scoring_adapter_version": "scoring-v1",
    }
    active_model_result = {
        "schema_version": "leadpoet.active_private_model.v2",
        "artifact": artifact_identity,
        "active_model": {
            "private_model_version_id": "private-model-v1",
        },
        "source_state_hash": _hash("9"),
    }

    async def load_missing(**_kwargs):
        return None

    async def persist(graph):
        validate_receipt_graph(graph)
        return {
            "root_receipt_hash": graph["root_receipt_hash"],
            "graph_hash": sha256_json(graph),
        }

    async def persist_result(**kwargs):
        captured.update(kwargs)
        return _execution_result_storage_row_v2(**kwargs)

    common = {
        "operation": "attest_active_private_model",
        "purpose": "research_lab.active_private_model.v2",
        "epoch_id": 24_285,
        "sequence": 0,
        "payload": {
            "artifact": {
                "model_artifact_hash": _hash("5"),
                "manifest_hash": _hash("6"),
                "git_commit_sha": "7" * 40,
                "config_hash": _hash("8"),
                "component_registry_version": "components-v1",
                "scoring_adapter_version": "scoring-v1",
                "image_digest": "sha256:" + "a" * 64,
                "image_uri": (
                    "493765492819.dkr.ecr.us-east-1.amazonaws.com/"
                    "leadpoet/sourcing-model@sha256:" + "a" * 64
                ),
            }
        },
        "input_artifact_hashes": (_hash("5"), _hash("6")),
        "worker_index": 0,
        "provider_profile_loader": lambda *args, **kwargs: {
            "profile": "default",
            "credential_ref_hashes": {},
            "envelopes": [],
        },
        "release_manifest": release,
        "persist_graph": persist,
        "boot_verifier": lambda identity: identity,
        "poll_seconds": 0.001,
        "operation_registry": COORDINATOR_OPERATIONS_V2,
        "physical_role_override": "gateway_coordinator",
        "expected_service_role": "gateway_coordinator",
        "rpc_namespace": "coordinator_v2",
        "receipt_output_projector": coordinator_receipt_output_v2,
    }
    fresh = await execute_scoring_v2(
        **common,
        client=_CoordinatorClient(
            release,
            executor=lambda _operation, _payload, _context: active_model_result,
        ),
        load_replayable_result=load_missing,
        persist_replayable_result=persist_result,
    )
    assert captured["operation"] == "attest_active_private_model"
    assert captured["result"] == active_model_result
    assert "image_digest" not in captured["result"]["artifact"]

    async def load_replay(**kwargs):
        assert kwargs == {
            "role": "gateway_coordinator",
            "operation": "attest_active_private_model",
            "purpose": "research_lab.active_private_model.v2",
            "job_id": fresh["receipt"]["job_id"],
        }
        return {
            "row": {"release_hash": release["release_hash"]},
            "result": fresh["result"],
            "receipt": fresh["receipt"],
            "receipt_graph": fresh["receipt_graph"],
            "artifact_hashes": fresh["artifact_hashes"],
        }

    restarted = _CoordinatorClient(release)

    async def reject_submit(_manifest):
        raise AssertionError(
            "same-epoch active-model replay must not submit a second job"
        )

    restarted.coordinator_v2_submit_job = reject_submit
    replayed = await execute_scoring_v2(
        **common,
        client=restarted,
        load_replayable_result=load_replay,
        persist_replayable_result=persist_result,
    )

    assert replayed["replay_status"] == "durable_exact"
    assert replayed["result"] == fresh["result"]
    assert replayed["receipt"] == fresh["receipt"]


@pytest.mark.asyncio
async def test_v2_bridge_preserves_complete_local_stage_receipt_chain():
    release = _release()

    def executor(operation, payload, context):
        context.record_stage(
            purpose="research_lab.provider_evidence_tape.v2",
            input_root=sha256_json({"trace": "baseline"}),
            output_root=sha256_json({"cache": "baseline"}),
            artifact_hashes=(_hash("a"),),
        )
        return {"operation": operation, "echo": payload}

    client = _Client(release, executor=executor)
    persisted = []

    async def persist(graph):
        validate_receipt_graph(graph)
        persisted.append(graph)
        return {"root_receipt_hash": graph["root_receipt_hash"]}

    result = await execute_scoring_v2(
        operation="run_model_sandbox_v2",
        purpose="research_lab.private_model_run.v2",
        epoch_id=12,
        sequence=0,
        payload={"model_kind": "private"},
        worker_index=0,
        release_manifest=release,
        client=client,
        persist_graph=persist,
        boot_verifier=lambda identity: identity,
        poll_seconds=0.001,
    )
    receipts = {
        item["receipt_hash"]: item for item in result["receipt_graph"]["receipts"]
    }
    root = receipts[result["receipt"]["receipt_hash"]]
    assert len(receipts) == 2
    assert len(root["parent_receipt_hashes"]) == 1
    stage = receipts[root["parent_receipt_hashes"][0]]
    assert stage["purpose"] == "research_lab.provider_evidence_tape.v2"
    assert stage["parent_receipt_hashes"] == []
    assert persisted[0] == result["receipt_graph"]


@pytest.mark.asyncio
async def test_v2_bridge_leases_and_releases_attested_benchmark_profile():
    release = _release()
    credential_hash = _hash("7")
    events = []

    def executor(operation, payload, context):
        events.append("execute")
        assert context.provider_credential_profile == "benchmark_model"
        assert context.provider_credential_ref_hashes == {"exa": credential_hash}
        assert payload.pop("_v2_provider_credential_profile") == "benchmark_model"
        assert payload.pop("_v2_provider_credential_ref_hashes") == {
            "exa": credential_hash
        }
        return {"operation": operation, "echo": payload}

    client = _Client(release, executor=executor)

    class _CredentialClient:
        async def v2_release_job_credentials(self, job_id):
            events.append("release")
            return {
                "status": "released",
                "job_id": job_id,
                "released_slot_count": 1,
            }

    def load_profile(
        profile,
        *,
        execution_role,
        worker_index,
        require_egress_proxy,
    ):
        assert profile == "benchmark_model"
        assert execution_role == "gateway_scoring"
        assert worker_index == 0
        assert require_egress_proxy is False
        return {
            "profile": profile,
            "credential_ref_hashes": {"exa": credential_hash},
            "envelopes": [{"encrypted": True}],
        }

    async def provision_profile(document, *, job_id, client):
        del client
        events.append("provision")
        return {
            "profile": document["profile"],
            "job_id": job_id,
            "credential_ref_hashes": dict(document["credential_ref_hashes"]),
            "leased_credential_count": 1,
            "results": [{"status": "ready"}],
        }

    async def persist_graph(graph):
        return {"root_receipt_hash": graph["root_receipt_hash"]}

    result = await execute_scoring_v2(
        operation="run_model_sandbox_v2",
        purpose="research_lab.private_model_run.v2",
        epoch_id=12,
        sequence=0,
        payload={"model_kind": "private"},
        worker_index=0,
        provider_credential_profile="benchmark_model",
        provider_profile_loader=load_profile,
        provider_profile_provisioner=provision_profile,
        credential_coordinator_client=_CredentialClient(),
        release_manifest=release,
        client=client,
        persist_graph=persist_graph,
        boot_verifier=lambda identity: identity,
        poll_seconds=0.001,
    )
    assert result["result"] == {
        "operation": "run_model_sandbox_v2",
        "echo": {"model_kind": "private"},
    }
    assert events == ["provision", "execute", "release"]


@pytest.mark.asyncio
async def test_v2_bridge_binds_dynamic_provider_id_to_derived_job_slot():
    release = _release()
    credential_hash = _hash("7")
    key_ref_hash = _hash("8")
    events = []

    def executor(operation, payload, context):
        events.append("execute")
        assert context.provider_credential_ref_hashes == {
            "source_one": credential_hash
        }
        assert payload.pop("_v2_provider_credential_ref_hashes") == {
            "source_one": credential_hash
        }
        return {"operation": operation, "echo": payload}

    client = _Client(release, executor=executor)

    class _CredentialClient:
        async def v2_release_job_credentials(self, job_id):
            events.append("release")
            return {
                "status": "released",
                "job_id": job_id,
                "released_slot_count": 1,
            }

    def load_profile(*_args, **_kwargs):
        return {
            "profile": "default",
            "credential_ref_hashes": {},
            "envelopes": [],
        }

    def build_envelopes(job_id):
        ciphertext = b"encrypted-source-add"
        context = {"adapter_ref": "source_add:adapter:test"}
        return [
            {
                "schema_version": "leadpoet.job_provider_credential_envelope.v2",
                "job_id": job_id,
                "credential_slot": "source_add_" + "d" * 32,
                "credential_ref_hash": credential_hash,
                "credential_value_hash": credential_hash,
                "key_ref_hash": key_ref_hash,
                "ciphertext_blob_b64": base64.b64encode(ciphertext).decode(),
                "ciphertext_blob_hash": sha256_bytes(ciphertext),
                "kms_key_id_hash": _hash("9"),
                "encryption_context": context,
                "encryption_context_hash": sha256_json(context),
            }
        ]

    async def provision_job(envelope, *, client):
        del client
        events.append("provision_dynamic")
        return {
            "status": "ready",
            "job_id": envelope["job_id"],
            "credential_slot": envelope["credential_slot"],
            "credential_ref_hash": envelope["credential_value_hash"],
        }

    async def persist_graph(graph):
        return {"root_receipt_hash": graph["root_receipt_hash"]}

    result = await execute_scoring_v2(
        operation="run_model_sandbox_v2",
        purpose="research_lab.private_model_run.v2",
        epoch_id=12,
        sequence=0,
        payload={"model_kind": "private"},
        worker_index=0,
        provider_credential_ref_hashes={"source_one": credential_hash},
        provider_profile_loader=load_profile,
        additional_job_credential_envelope_builder=build_envelopes,
        job_credential_provisioner=provision_job,
        credential_coordinator_client=_CredentialClient(),
        release_manifest=release,
        client=client,
        persist_graph=persist_graph,
        boot_verifier=lambda identity: identity,
        poll_seconds=0.001,
    )

    assert result["status"] == "succeeded"
    assert events == ["provision_dynamic", "execute", "release"]


@pytest.mark.asyncio
async def test_v2_bridge_preserves_provisioning_failure_after_idempotent_cleanup():
    release = _release()
    credential_hashes = (_hash("7"), _hash("8"))
    release_counts = []

    class _CredentialClient:
        active_slot_count = 0

        async def v2_release_job_credentials(self, job_id):
            released_slot_count = self.active_slot_count
            self.active_slot_count = 0
            release_counts.append(released_slot_count)
            return {
                "status": "released",
                "job_id": job_id,
                "released_slot_count": released_slot_count,
            }

    credential_client = _CredentialClient()

    def load_profile(*_args, **_kwargs):
        return {
            "profile": "default",
            "credential_ref_hashes": {},
            "envelopes": [],
        }

    def build_envelopes(job_id):
        envelopes = []
        for index, credential_hash in enumerate(credential_hashes):
            ciphertext = f"encrypted-source-{index}".encode()
            context = {"adapter_ref": f"source_add:adapter:{index}"}
            envelopes.append(
                {
                    "schema_version":
                        "leadpoet.job_provider_credential_envelope.v2",
                    "job_id": job_id,
                    "credential_slot":
                        "source_add_" + f"{index + 1:x}" * 32,
                    "credential_ref_hash": credential_hash,
                    "credential_value_hash": credential_hash,
                    "key_ref_hash": _hash("9"),
                    "ciphertext_blob_b64":
                        base64.b64encode(ciphertext).decode(),
                    "ciphertext_blob_hash": sha256_bytes(ciphertext),
                    "kms_key_id_hash": _hash("a"),
                    "encryption_context": context,
                    "encryption_context_hash": sha256_json(context),
                }
            )
        return envelopes

    provision_attempt = 0

    async def provision_job(envelope, *, client):
        nonlocal provision_attempt
        provision_attempt += 1
        if provision_attempt == 1:
            client.active_slot_count += 1
            return {
                "status": "ready",
                "job_id": envelope["job_id"],
                "credential_slot": envelope["credential_slot"],
                "credential_ref_hash": envelope["credential_value_hash"],
            }
        await client.v2_release_job_credentials(envelope["job_id"])
        raise RuntimeError("ORIGINAL: second credential KMS failed")

    async def persist_graph(graph):
        return {"root_receipt_hash": graph["root_receipt_hash"]}

    with pytest.raises(
        RuntimeError,
        match="ORIGINAL: second credential KMS failed",
    ):
        await execute_scoring_v2(
            operation="run_model_sandbox_v2",
            purpose="research_lab.private_model_run.v2",
            epoch_id=12,
            sequence=0,
            payload={"model_kind": "private"},
            worker_index=0,
            provider_credential_ref_hashes={
                "source_one": credential_hashes[0],
                "source_two": credential_hashes[1],
            },
            provider_profile_loader=load_profile,
            additional_job_credential_envelope_builder=build_envelopes,
            job_credential_provisioner=provision_job,
            credential_coordinator_client=credential_client,
            release_manifest=release,
            client=_Client(release),
            persist_graph=persist_graph,
            boot_verifier=lambda identity: identity,
            poll_seconds=0.001,
        )

    assert provision_attempt == 2
    assert release_counts == [1, 0]


@pytest.mark.asyncio
async def test_v2_bridge_still_fails_closed_on_successful_release_mismatch():
    release = _release()
    credential_hash = _hash("7")
    release_calls = []

    class _CredentialClient:
        async def v2_release_job_credentials(self, job_id):
            release_calls.append(job_id)
            return {
                "status": "released",
                "job_id": job_id,
                "released_slot_count": 0,
            }

    def load_profile(
        profile,
        *,
        execution_role,
        worker_index,
        require_egress_proxy,
    ):
        return {
            "profile": profile,
            "credential_ref_hashes": {"exa": credential_hash},
            "envelopes": [{"encrypted": True}],
        }

    async def provision_profile(document, *, job_id, client):
        del client
        return {
            "profile": document["profile"],
            "job_id": job_id,
            "credential_ref_hashes": dict(document["credential_ref_hashes"]),
            "leased_credential_count": 1,
            "results": [{"status": "ready"}],
        }

    async def persist_graph(graph):
        return {"root_receipt_hash": graph["root_receipt_hash"]}

    with pytest.raises(
        AttestedScoringV2Error,
        match="provider credential profile release failed",
    ):
        await execute_scoring_v2(
            operation="run_model_sandbox_v2",
            purpose="research_lab.private_model_run.v2",
            epoch_id=12,
            sequence=0,
            payload={"model_kind": "private"},
            worker_index=0,
            provider_credential_profile="benchmark_model",
            provider_profile_loader=load_profile,
            provider_profile_provisioner=provision_profile,
            credential_coordinator_client=_CredentialClient(),
            release_manifest=release,
            client=_Client(release),
            persist_graph=persist_graph,
            boot_verifier=lambda identity: identity,
            poll_seconds=0.001,
        )

    assert len(release_calls) == 1


@pytest.mark.asyncio
async def test_v2_bridge_fails_when_persistence_does_not_read_back_root():
    release = _release()
    client = _Client(release)

    async def persist(_graph):
        return {"root_receipt_hash": _hash("f")}

    with pytest.raises(AttestedScoringV2Error, match="durable readback"):
        await execute_scoring_v2(
            operation="benchmark_icp_score",
            purpose="research_lab.benchmark.v2",
            epoch_id=12,
            sequence=0,
            payload={"scores": [1.0]},
            worker_index=0,
            release_manifest=release,
            client=client,
            persist_graph=persist,
            boot_verifier=lambda identity: identity,
            poll_seconds=0.001,
        )


@pytest.mark.asyncio
async def test_v2_bridge_ignores_uncommitted_orphans_and_persists_failure(
    monkeypatch,
):
    release = _release()

    def fail_executor(_operation, _payload, _context):
        raise ValueError("measured scoring failure")

    client = _Client(release, executor=fail_executor)
    persisted = []

    async def persist(graph, *, allowed_failed_receipt_hashes=()):
        allowed = set(allowed_failed_receipt_hashes)
        validate_receipt_graph(
            graph,
            required_purposes={"research_lab.benchmark.v2"},
            allowed_failed_receipt_hashes=allowed,
        )
        assert allowed == {graph["root_receipt_hash"]}
        persisted.append(graph)
        return {"root_receipt_hash": graph["root_receipt_hash"]}

    async def unexpected_artifact_persistence(**_kwargs):
        pytest.fail("static input commitments do not require artifact persistence")

    class OrphanArtifactCoordinator:
        async def v2_list_encrypted_artifacts(self, *, job_id, purpose):
            assert job_id
            assert purpose == "research_lab.benchmark.v2"
            return {
                "artifacts": [
                    {
                        "artifact_id": _hash("2"),
                        "plaintext_hash": _hash("3"),
                        "ciphertext_hash": _hash("4"),
                        "encryption_context_hash": _hash("5"),
                        "artifact_kind": "provider_request",
                        "persisted": False,
                    }
                ]
            }

    monkeypatch.setattr(
        "gateway.research_lab.attested_artifacts_v2."
        "persist_execution_transport_artifacts_v2",
        unexpected_artifact_persistence,
    )

    with pytest.raises(AttestedScoringV2Error, match="failed closed") as captured:
        await execute_scoring_v2(
            operation="benchmark_icp_score",
            purpose="research_lab.benchmark.v2",
            epoch_id=12,
            sequence=0,
            payload={"scores": [1.0]},
            worker_index=0,
            input_artifact_hashes=(_hash("1"),),
            release_manifest=release,
            client=client,
            artifact_coordinator_client=OrphanArtifactCoordinator(),
            persist_graph=persist,
            boot_verifier=lambda identity: identity,
            poll_seconds=0.001,
        )

    authority = captured.value.authority
    assert authority is not None
    assert authority["status"] == "failed"
    assert authority["execution_receipt"]["status"] == "failed"
    assert (
        authority["execution_receipt_graph"]["root_receipt_hash"]
        == authority["execution_receipt"]["receipt_hash"]
    )
    assert authority["result"] == {
        "status": "failed",
        "failure_code": "execution_valueerror",
    }
    assert persisted[0] == authority["receipt_graph"]


@pytest.mark.asyncio
async def test_v2_bridge_persists_failed_source_artifacts_with_child_local_policy(
    monkeypatch,
):
    release = _release()

    def fail_executor(_operation, _payload, _context):
        raise ValueError("measured scoring failure")

    artifact_client = _ArtifactCoordinator(release, _hash("1"))
    original_list = artifact_client.v2_list_encrypted_artifacts

    async def list_persisted(*, job_id, purpose):
        listed = await original_list(job_id=job_id, purpose=purpose)
        return {
            "artifacts": [
                {
                    **item,
                    "artifact_kind": "provider_request",
                    "persisted": True,
                }
                for item in listed["artifacts"]
            ]
        }

    artifact_client.v2_list_encrypted_artifacts = list_persisted
    persisted_policies = []

    async def persist(graph, *, allowed_failed_receipt_hashes=()):
        allowed = set(allowed_failed_receipt_hashes)
        validate_receipt_graph(
            graph,
            allowed_failed_receipt_hashes=allowed,
        )
        persisted_policies.append((graph["root_receipt_hash"], allowed))
        return {"root_receipt_hash": graph["root_receipt_hash"]}

    async def persist_sidecars(**kwargs):
        return {"artifact_link_count": len(kwargs["artifacts"])}

    monkeypatch.setattr(
        attested_v2_store,
        "persist_execution_sidecars_v2",
        persist_sidecars,
    )

    with pytest.raises(AttestedScoringV2Error, match="failed closed") as captured:
        await execute_scoring_v2(
            operation="benchmark_icp_score",
            purpose="research_lab.benchmark.v2",
            epoch_id=12,
            sequence=0,
            payload={"scores": [1.0]},
            worker_index=0,
            input_artifact_hashes=(_hash("1"),),
            release_manifest=release,
            client=_Client(release, executor=fail_executor),
            artifact_coordinator_client=artifact_client,
            persist_graph=persist,
            boot_verifier=lambda identity: identity,
            poll_seconds=0.001,
        )

    authority = captured.value.authority
    assert authority is not None
    source_receipt = authority["execution_receipt"]
    child_receipt = authority["receipt"]
    assert source_receipt["status"] == "failed"
    assert child_receipt["status"] == "succeeded"
    assert source_receipt["receipt_hash"] in child_receipt[
        "parent_receipt_hashes"
    ]
    assert persisted_policies == [
        (
            source_receipt["receipt_hash"],
            {source_receipt["receipt_hash"]},
        ),
        (child_receipt["receipt_hash"], set()),
    ]


@pytest.mark.asyncio
async def test_v2_bridge_rejects_unauthorized_purpose_before_rpc():
    release = _release()
    with pytest.raises(AttestedScoringV2Error, match="purpose"):
        await execute_scoring_v2(
            operation="benchmark_icp_score",
            purpose="research_lab.allocation.v2",
            epoch_id=12,
            sequence=0,
            payload={"scores": [1.0]},
            worker_index=0,
            release_manifest=release,
            client=object(),
        )


def _authenticated_attempt(context):
    return build_transport_attempt(
        request_id="a" * 32,
        logical_operation_id="provider-operation",
        job_id=context.job_id,
        purpose=context.purpose,
        provider_id="openrouter",
        attempt_number=0,
        method="POST",
        destination_host="openrouter.ai",
        destination_port=443,
        path_hash=_hash("1"),
        nonsecret_headers_hash=_hash("2"),
        body_hash=_hash("3"),
        credential_ref_hash=_hash("4"),
        retry_policy_hash=_hash("5"),
        timeout_ms=30000,
        started_at="2026-07-10T00:00:00Z",
        terminal_status="authenticated_response",
        http_status=200,
        response_hash=_hash("6"),
        request_artifact_hash=_hash("8"),
        response_artifact_hash=_hash("6"),
        tls_peer_chain_hash=_hash("7"),
        tls_protocol="TLSv1.3",
        failure_code=None,
        completed_at="2026-07-10T00:00:01Z",
    )


def _storage_attempts_for_job(artifact_id, job_id):
    attempts = []
    for ordinal, method in enumerate(("GET", "HEAD")):
        attempts.append(
            build_transport_attempt(
                request_id=("c" if ordinal == 0 else "d") * 32,
                logical_operation_id="%s:%s" % (artifact_id, method.lower()),
                job_id=job_id,
                purpose="leadpoet.artifact_persistence.v2",
                provider_id="aws_s3_object_lock",
                attempt_number=ordinal,
                method=method,
                destination_host="immutable.example.s3.us-east-1.amazonaws.com",
                destination_port=443,
                path_hash=_hash("1"),
                nonsecret_headers_hash=_hash("2"),
                body_hash=_hash("3"),
                credential_ref_hash=_hash("4"),
                retry_policy_hash=_hash("5"),
                timeout_ms=30000,
                started_at="2026-07-10T00:00:00Z",
                terminal_status="authenticated_response",
                http_status=200,
                response_hash=_hash("6"),
                request_artifact_hash=_hash("8"),
                response_artifact_hash=_hash("6"),
                tls_peer_chain_hash=_hash("7"),
                tls_protocol="TLSv1.3",
                failure_code=None,
                completed_at="2026-07-10T00:00:01Z",
            )
        )
    return attempts


def _storage_attempts(artifact_id):
    return _storage_attempts_for_job(artifact_id, artifact_id)


class _ArtifactCoordinator:
    def __init__(self, release, plaintext_hashes):
        if isinstance(plaintext_hashes, str):
            plaintext_hashes = (plaintext_hashes,)
        self.plaintext_hashes = tuple(plaintext_hashes)
        role = "gateway_coordinator"
        summary = release["roles"][role]
        self.key = Ed25519PrivateKey.generate()
        pubkey = self.key.public_key().public_bytes(
            serialization.Encoding.Raw,
            serialization.PublicFormat.Raw,
        ).hex()
        body = build_boot_identity_body(
            role="gateway_coordinator",
            physical_role=role,
            commit_sha=summary["commit_sha"],
            pcr0=summary["pcr0"],
            build_manifest_hash=summary["execution_manifest_hash"],
            dependency_lock_hash=summary["dependency_lock_hash"],
            config_hash=_hash("9"),
            boot_nonce="e" * 32,
            signing_pubkey=pubkey,
            transport_pubkey="f" * 64,
            transport_certificate_hash=_hash("a"),
            attestation_user_data_hash=_hash("b"),
            issued_at="2026-07-10T00:00:00Z",
        )
        self.boot = create_boot_identity(
            body=body,
            attestation_document_b64=base64.b64encode(b"nitro").decode(),
        )
        self.artifacts = [
            {
                "artifact_id": _hash(character),
                "plaintext_hash": plaintext_hash,
            }
            for character, plaintext_hash in zip(
                "89abcdef",
                self.plaintext_hashes,
            )
        ]

        def evidence_for(artifact, context):
            attempts = _storage_attempts_for_job(
                artifact["artifact_id"],
                context.job_id,
            )
            return {
                **artifact,
                "ciphertext_hash": _hash("c"),
                "artifact_ref": "s3://immutable/%s.json"
                % artifact["artifact_id"].split(":", 1)[1][:8],
                "storage_document_hash": _hash("d"),
                "encryption_context_hash": _hash("e"),
                "object_lock_mode": "COMPLIANCE",
                "retain_until": "2027-07-10T12:00:00Z",
                "transport_root": transport_root(attempts),
                "transport_attempts": attempts,
                "persisted": True,
            }
        self.manager = ExecutionJobManagerV2(
            boot_identity_supplier=lambda: self.boot,
            sign_digest=self.key.sign,
            operations=COORDINATOR_OPERATIONS_V2,
            executor=CoordinatorExecutorV2(
                artifact_evidence_supplier=lambda ids, context: [
                    evidence_for(artifact, context)
                    for artifact in self.artifacts
                    if artifact["artifact_id"] in ids
                ]
            ),
            failed_parent_graph_policy=coordinator_failed_parent_graph_policy_v2,
            worker_count=1,
            configured_worker_count=0,
            ancestry_lineage_id=attested_scoring_v2._gateway_ancestry_lineage_id(),
            ancestry_boot_attestation_verifier=lambda identity: identity,
            ancestry_allowed_issuer_roles=(
                "gateway_autoresearch",
                "gateway_coordinator",
                "gateway_scoring",
            ),
        )

    async def v2_list_encrypted_artifacts(self, *, job_id, purpose):
        return {
            "artifacts": [
                {**artifact, "job_id": job_id, "purpose": purpose}
                for artifact in self.artifacts
            ]
        }

    async def v2_get_boot_identity(self):
        return self.boot

    async def coordinator_v2_health(self):
        return self.manager.health()

    async def coordinator_v2_submit_job(self, manifest):
        return self.manager.submit(manifest)

    async def coordinator_v2_put_chunk(self, *, job_id, offset, data):
        import hashlib

        return self.manager.put_chunk(
            job_id=job_id,
            offset=offset,
            data_b64=base64.b64encode(data).decode(),
            chunk_sha256="sha256:" + hashlib.sha256(data).hexdigest(),
        )

    async def coordinator_v2_seal_job(self, job_id):
        return self.manager.seal(job_id)

    async def coordinator_v2_get_status(self, job_id):
        return self.manager.status(job_id)

    async def coordinator_v2_cancel_job(self, job_id):
        return self.manager.cancel(job_id)

    async def coordinator_v2_get_result(self, job_id, *, offset=0):
        return self.manager.result_chunk(job_id=job_id, offset=offset)

    async def coordinator_v2_get_receipt(self, job_id):
        return self.manager.receipt(job_id)

    async def coordinator_v2_get_receipts(self, job_id):
        return list(self.manager.receipts(job_id))

    async def coordinator_v2_get_transport_attempts(self, job_id):
        return list(self.manager.transport_attempts(job_id))

    async def coordinator_v2_get_host_operations(self, job_id):
        return list(self.manager.host_operations(job_id))

    async def coordinator_v2_get_ancestry_compact_proof(self, job_id):
        return self.manager.ancestry_compact_proof(job_id)

    async def coordinator_v2_get_artifact_hashes(self, job_id):
        return list(self.manager.artifact_hashes(job_id))

    async def coordinator_v2_get_transitions(self, job_id):
        return list(self.manager.transitions(job_id))


@pytest.mark.asyncio
async def test_v2_bridge_persists_every_authenticated_provider_artifact_first(
    monkeypatch,
):
    release = _release()

    def executor(operation, payload, context):
        for digest in (_hash("8"), _hash("9"), _hash("6")):
            context.record_artifact(digest)
        return ExecutionResultV2(
            output={"operation": operation, "echo": payload},
            transport_attempts=(_authenticated_attempt(context),),
        )

    client = _Client(release, executor=executor)
    persisted_artifacts = []
    persisted_graphs = []
    persisted_sidecars = []
    durability_events = []

    async def persist_artifact(artifact_id, **kwargs):
        persisted_artifacts.append((artifact_id, kwargs))
        return {
            "status": "persisted",
            "artifact_id": artifact_id,
            "storage_document_hash": _hash("d"),
            "artifact_kind": "provider_response",
            "artifact_hash": _hash("c"),
            "encryption_context_hash": _hash("e"),
            "object_lock_mode": "COMPLIANCE",
            "retain_until": "2027-07-10T12:00:00Z",
            "transport_root": transport_root(
                _storage_attempts_for_job(
                    artifact_id,
                    kwargs["attestation_job_id"],
                )
            ),
        }

    async def persist_graph(graph):
        persisted_graphs.append(graph)
        durability_events.append(("graph", graph["root_receipt_hash"]))
        return {"root_receipt_hash": graph["root_receipt_hash"]}

    async def persist_checkpoint(proof, *, checkpointed_graph, **_kwargs):
        durability_events.append(
            ("checkpoint", checkpointed_graph["root_receipt_hash"])
        )
        return {
            "root_receipt_hash": checkpointed_graph["root_receipt_hash"],
            "proof_hash": proof["proof_hash"],
        }

    monkeypatch.setattr(
        attested_v2_store,
        "persist_ancestry_checkpoint_v2",
        persist_checkpoint,
    )

    async def persist_sidecars(**kwargs):
        persisted_sidecars.append(kwargs)
        return {"artifact_link_count": 1, "transition_count": 0}

    result = await execute_scoring_v2(
        operation="benchmark_icp_score",
        purpose="research_lab.benchmark.v2",
        epoch_id=12,
        sequence=0,
        payload={"scores": [1.0]},
        worker_index=0,
        release_manifest=release,
        client=client,
        artifact_coordinator_client=_ArtifactCoordinator(
            release,
            (_hash("8"), _hash("6")),
        ),
        persist_artifact=persist_artifact,
        artifact_bucket="immutable-bucket",
        persist_graph=persist_graph,
        persist_sidecars=persist_sidecars,
        boot_verifier=lambda identity: identity,
        poll_seconds=0.001,
    )
    assert {item[0] for item in persisted_artifacts} == {_hash("8"), _hash("9")}
    assert persisted_artifacts[0][1]["bucket"] == "immutable-bucket"
    assert persisted_artifacts[0][1]["key_prefix"] == "encrypted-artifacts"
    assert (
        persisted_artifacts[0][1]["attestation_job_id"]
        == result["receipt"]["job_id"]
    )
    assert result["artifact_persistence"][0]["status"] == "persisted"
    assert result["receipt"]["purpose"] == "leadpoet.artifact_persistence.v2"
    assert result["execution_receipt"]["purpose"] == "research_lab.benchmark.v2"
    assert (
        result["execution_receipt_graph"]["root_receipt_hash"]
        == result["execution_receipt"]["receipt_hash"]
    )
    assert (
        result["receipt_graph"]["root_receipt_hash"]
        == result["receipt"]["receipt_hash"]
    )
    assert len(persisted_graphs) == 2
    assert durability_events[:2] == [
        ("graph", result["execution_receipt"]["receipt_hash"]),
        ("checkpoint", result["execution_receipt"]["receipt_hash"]),
    ]
    assert durability_events[2:4] == [
        ("graph", result["receipt"]["receipt_hash"]),
        ("checkpoint", result["receipt"]["receipt_hash"]),
    ]
    assert persisted_sidecars[0]["artifact_receipt_hash"] == result["receipt"][
        "receipt_hash"
    ]


@pytest.mark.asyncio
async def test_v2_bridge_accepts_persistence_bound_encrypted_descriptors():
    release = _release()

    def executor(operation, payload, context):
        for digest in (_hash("8"), _hash("9"), _hash("6")):
            context.record_artifact(digest)
        return ExecutionResultV2(
            output={"operation": operation, "echo": payload},
            transport_attempts=(_authenticated_attempt(context),),
        )

    artifact_client = _ArtifactCoordinator(release, (_hash("8"), _hash("6")))
    original_list = artifact_client.v2_list_encrypted_artifacts

    async def list_with_full_descriptors(*, job_id, purpose):
        listed = await original_list(job_id=job_id, purpose=purpose)
        return {
            "artifacts": [
                {
                    **item,
                    "ciphertext_hash": _hash("c"),
                    "encryption_context_hash": _hash("e"),
                }
                for item in listed["artifacts"]
            ]
        }

    artifact_client.v2_list_encrypted_artifacts = list_with_full_descriptors

    async def persist_artifact(artifact_id, **kwargs):
        return {
            "status": "persisted",
            "artifact_id": artifact_id,
            "storage_document_hash": _hash("d"),
            "artifact_kind": "provider_response",
            "artifact_hash": _hash("c"),
            "encryption_context_hash": _hash("e"),
            "object_lock_mode": "COMPLIANCE",
            "retain_until": "2027-07-10T12:00:00Z",
            "transport_root": transport_root(
                _storage_attempts_for_job(
                    artifact_id,
                    kwargs["attestation_job_id"],
                )
            ),
        }

    async def persist_graph(graph, **_kwargs):
        return {"root_receipt_hash": graph["root_receipt_hash"]}

    async def persist_sidecars(**_kwargs):
        return {"artifact_link_count": 2}

    result = await execute_scoring_v2(
        operation="benchmark_icp_score",
        purpose="research_lab.benchmark.v2",
        epoch_id=12,
        sequence=0,
        payload={"scores": [1.0]},
        worker_index=0,
        release_manifest=release,
        client=_Client(release, executor=executor),
        artifact_coordinator_client=artifact_client,
        persist_artifact=persist_artifact,
        artifact_bucket="immutable-bucket",
        persist_graph=persist_graph,
        persist_sidecars=persist_sidecars,
        boot_verifier=lambda identity: identity,
        poll_seconds=0.001,
        allow_persistence_bound_artifact_descriptors=True,
    )

    assert result["status"] == "succeeded"
    assert len(result["artifact_persistence"]) == 2


@pytest.mark.asyncio
async def test_v2_bridge_reuses_enclave_attested_persistence_on_replay():
    release = _release()

    def executor(operation, payload, context):
        for digest in (_hash("8"), _hash("9"), _hash("6")):
            context.record_artifact(digest)
        return ExecutionResultV2(
            output={"operation": operation, "echo": payload},
            transport_attempts=(_authenticated_attempt(context),),
        )

    artifact_client = _ArtifactCoordinator(release, (_hash("8"), _hash("6")))
    original_list = artifact_client.v2_list_encrypted_artifacts

    async def list_persisted(*, job_id, purpose):
        listed = await original_list(job_id=job_id, purpose=purpose)
        return {
            "artifacts": [
                {
                    **item,
                    "artifact_kind": "provider_response",
                    "ciphertext_hash": _hash("c"),
                    "encryption_context_hash": _hash("e"),
                    "persisted": True,
                }
                for item in listed["artifacts"]
            ]
        }

    artifact_client.v2_list_encrypted_artifacts = list_persisted

    async def reject_duplicate_upload(*_args, **_kwargs):
        raise AssertionError("persisted artifact must not be uploaded again")

    async def persist_graph(graph, **_kwargs):
        return {"root_receipt_hash": graph["root_receipt_hash"]}

    persisted_sidecars = []

    async def persist_sidecars(**kwargs):
        persisted_sidecars.append(kwargs)
        return {"artifact_link_count": len(kwargs["artifacts"])}

    result = await execute_scoring_v2(
        operation="benchmark_icp_score",
        purpose="research_lab.benchmark.v2",
        epoch_id=12,
        sequence=0,
        payload={"scores": [1.0]},
        worker_index=0,
        release_manifest=release,
        client=_Client(release, executor=executor),
        artifact_coordinator_client=artifact_client,
        persist_artifact=reject_duplicate_upload,
        persist_graph=persist_graph,
        persist_sidecars=persist_sidecars,
        boot_verifier=lambda identity: identity,
        poll_seconds=0.001,
        allow_persistence_bound_artifact_descriptors=True,
    )

    assert result["status"] == "succeeded"
    assert len(result["artifact_persistence"]) == 2
    assert all(
        item["status"] == "persisted" for item in result["artifact_persistence"]
    )
    assert persisted_sidecars[0]["artifacts"] == result["artifact_persistence"]


@pytest.mark.asyncio
async def test_v2_bridge_rejects_missing_encrypted_provider_artifact():
    release = _release()

    def executor(operation, payload, context):
        for digest in (_hash("8"), _hash("9"), _hash("f")):
            context.record_artifact(digest)
        return ExecutionResultV2(
            output={"operation": operation, "echo": payload},
            transport_attempts=(_authenticated_attempt(context),),
        )

    with pytest.raises(AttestedScoringV2Error, match="execution commitments"):
        await execute_scoring_v2(
            operation="benchmark_icp_score",
            purpose="research_lab.benchmark.v2",
            epoch_id=12,
            sequence=0,
            payload={"scores": [1.0]},
            worker_index=0,
            release_manifest=release,
            client=_Client(release, executor=executor),
            artifact_coordinator_client=_ArtifactCoordinator(
                release,
                (_hash("8"), _hash("f")),
            ),
            artifact_bucket="immutable-bucket",
            persist_graph=lambda graph: graph,
            boot_verifier=lambda identity: identity,
            poll_seconds=0.001,
            allow_persistence_bound_artifact_descriptors=True,
        )
