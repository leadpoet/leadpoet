from __future__ import annotations

import base64
import json
import threading
import time

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from gateway.tee.execution_job_manager_v2 import (
    JOB_SCHEMA_VERSION,
    PARENT_RECEIPT_GRAPH_SET_FIELD,
    PARENT_RECEIPT_GRAPHS_FIELD,
    ExecutionContextV2,
    ExecutionJobManagerV2,
    ExecutionJobV2Error,
    ExecutionResultV2,
    TransitionSpecV2,
    pack_parent_receipt_graph_set_v2,
    unpack_parent_receipt_graph_set_v2,
)
from gateway.tee import execution_job_manager_v2 as job_manager_v2
from leadpoet_canonical.attested_v2 import (
    CHECKPOINTED_RECEIPT_GRAPH_SCHEMA_VERSION,
    DIRECT_EGRESS_REF_HASH,
    build_boot_identity_body,
    build_checkpointed_receipt_graph,
    build_execution_receipt_body,
    build_receipt_graph,
    build_transport_attempt,
    create_boot_identity,
    create_signed_execution_receipt,
    host_operation_root,
    merkle_root,
    sha256_bytes,
    validate_signed_execution_receipt,
    validate_signed_transition_command,
)
from leadpoet_canonical.ancestry_checkpoint_v2 import (
    ANCESTRY_CHECKPOINT_BOOTSTRAP_REQUEST_SCHEMA_VERSION,
    ANCESTRY_CHECKPOINT_BOOTSTRAP_RESULT_SCHEMA_VERSION,
)


HASH = "sha256:" + "a" * 64
HASH_B = "sha256:" + "b" * 64
NOW = "2026-07-10T20:00:00Z"


def _manager(
    executor,
    *,
    checkpoint_lineage=False,
    role="gateway_scoring",
    operations=None,
):
    key = Ed25519PrivateKey.generate()
    pubkey = key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    ).hex()
    boot = create_boot_identity(
        body=build_boot_identity_body(
            role=role,
            physical_role=role,
            commit_sha="c" * 40,
            pcr0="d" * 96,
            build_manifest_hash=HASH,
            dependency_lock_hash=HASH_B,
            config_hash=HASH,
            boot_nonce="e" * 32,
            signing_pubkey=pubkey,
            transport_pubkey="f" * 64,
            transport_certificate_hash=HASH_B,
            attestation_user_data_hash=HASH,
            issued_at=NOW,
        ),
        attestation_document_b64=base64.b64encode(b"nitro").decode("ascii"),
    )
    checkpoint_kwargs = {}
    if checkpoint_lineage:
        checkpoint_kwargs = {
            "ancestry_lineage_id": HASH,
            "ancestry_boot_attestation_verifier": lambda identity: identity,
            "ancestry_allowed_issuer_roles": (role,),
        }
    manager = ExecutionJobManagerV2(
        boot_identity_supplier=lambda: boot,
        sign_digest=key.sign,
        operations=(
            operations
            if operations is not None
            else {
                "score": {
                    "research_lab.candidate_score.v2",
                    "research_lab.baseline_score.v2",
                }
            }
        ),
        executor=executor,
        worker_count=1,
        **checkpoint_kwargs,
    )
    return manager, boot


def _payload():
    return json.dumps(
        {"input": 3},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _transport_attempt(*, request_id_character="1", attempt_number=0):
    return build_transport_attempt(
        request_id=request_id_character * 32,
        logical_operation_id="provider-op-1",
        job_id="score-job-1",
        purpose="research_lab.candidate_score.v2",
        provider_id="openrouter",
        attempt_number=attempt_number,
        method="POST",
        destination_host="openrouter.ai",
        destination_port=443,
        path_hash=HASH,
        nonsecret_headers_hash=HASH,
        body_hash=HASH_B,
        credential_ref_hash=HASH,
        retry_policy_hash=HASH_B,
        timeout_ms=30000,
        started_at=NOW,
        terminal_status="authenticated_response",
        http_status=200,
        response_hash=HASH,
        request_artifact_hash=HASH,
        response_artifact_hash=HASH_B,
        tls_peer_chain_hash=HASH,
        tls_protocol="TLSv1.3",
        failure_code=None,
        completed_at=NOW,
    )


def test_execution_context_freezes_one_terminal_transport_snapshot():
    context = ExecutionContextV2(
        job_id="score-job-1",
        purpose="research_lab.candidate_score.v2",
        epoch_id=24_000,
    )
    first = _transport_attempt()
    context.record_transport(first)

    assert context.freeze_transport_attempts() == (first,)
    assert context.freeze_transport_attempts() == (first,)
    with pytest.raises(
        ExecutionJobV2Error,
        match="transport attempt arrived after execution was finalized",
    ):
        context.record_transport(
            _transport_attempt(request_id_character="2", attempt_number=1)
        )


def test_execution_context_freezes_one_artifact_snapshot():
    context = ExecutionContextV2(
        job_id="score-job-1",
        purpose="research_lab.candidate_score.v2",
        epoch_id=24_000,
    )
    context.record_artifact(HASH)

    assert context.freeze_artifact_hashes() == (HASH,)
    assert context.freeze_artifact_hashes() == (HASH,)
    with pytest.raises(
        ExecutionJobV2Error,
        match="execution artifact arrived after execution was finalized",
    ):
        context.record_artifact(HASH_B)


def test_job_receipt_and_ancestry_use_same_frozen_transport_snapshot():
    release_late_attempt = threading.Event()
    late_attempt_finished = threading.Event()
    late_errors = []
    first = _transport_attempt()
    second = _transport_attempt(request_id_character="2", attempt_number=1)
    late_thread = None

    def _executor(_operation, payload, context):
        nonlocal late_thread
        context.record_transport(first)

        def append_late():
            release_late_attempt.wait(timeout=2)
            try:
                context.record_transport(second)
            except Exception as exc:
                late_errors.append(exc)
            finally:
                late_attempt_finished.set()

        late_thread = threading.Thread(target=append_late)
        late_thread.start()
        return {"score": payload["input"]}

    manager, boot = _manager(_executor, checkpoint_lineage=True)
    assert _run(manager, _payload())["state"] == "succeeded"
    release_late_attempt.set()
    assert late_attempt_finished.wait(timeout=1)
    late_thread.join(timeout=1)

    assert len(late_errors) == 1
    assert "after execution was finalized" in str(late_errors[0])
    assert manager.transport_attempts("score-job-1") == (first,)
    receipt = manager.receipt("score-job-1")
    assert receipt["transport_root"] == merkle_root(
        [first["attempt_hash"]], domain="leadpoet-transport-v2"
    )
    build_checkpointed_receipt_graph(
        root_receipt_hash=receipt["receipt_hash"],
        boot_identities=(boot,),
        receipts=manager.receipts("score-job-1"),
        transport_attempts=manager.transport_attempts("score-job-1"),
        host_operations=manager.host_operations("score-job-1"),
        ancestry_lineage_id=HASH,
        ancestry_proof=manager.ancestry_compact_proof("score-job-1"),
        boot_attestation_verifier=lambda identity: identity,
        require_boot_attestation_verification=True,
    )


def test_job_receipt_uses_same_frozen_artifact_snapshot():
    release_late_artifact = threading.Event()
    late_artifact_finished = threading.Event()
    late_errors = []
    late_thread = None
    late_hash = "sha256:" + "c" * 64

    def _executor(_operation, payload, context):
        nonlocal late_thread
        context.record_artifact(HASH)

        def append_late():
            release_late_artifact.wait(timeout=2)
            try:
                context.record_artifact(late_hash)
            except Exception as exc:
                late_errors.append(exc)
            finally:
                late_artifact_finished.set()

        late_thread = threading.Thread(target=append_late)
        late_thread.start()
        return {"score": payload["input"]}

    manager, _boot = _manager(_executor, checkpoint_lineage=True)
    assert _run(manager, _payload())["state"] == "succeeded"
    release_late_artifact.set()
    assert late_artifact_finished.wait(timeout=1)
    late_thread.join(timeout=1)

    assert len(late_errors) == 1
    assert "after execution was finalized" in str(late_errors[0])
    assert manager.artifact_hashes("score-job-1") == (HASH_B, HASH)
    assert manager.receipt("score-job-1")["artifact_root"] == merkle_root(
        [HASH_B, HASH], domain="leadpoet-artifact-v2"
    )


def test_external_receipt_graph_count_is_larger_only_for_allocation_ancestry(
    monkeypatch,
):
    monkeypatch.setattr(
        job_manager_v2,
        "validate_receipt_graph",
        lambda _graph, **_kwargs: (),
    )
    context = ExecutionContextV2(
        job_id="score-job",
        purpose="research_lab.candidate_score.v2",
        epoch_id=24_262,
    )

    for index in range(job_manager_v2.MAX_EXTERNAL_RECEIPT_GRAPHS):
        root = "sha256:" + format(index, "064x")
        context.record_external_receipt_graph(
            {
                "root_receipt_hash": root,
                "receipts": [{"receipt_hash": root}],
            }
        )

    allocation_context = ExecutionContextV2(
        job_id="allocation-job",
        purpose="research_lab.allocation.v2",
        epoch_id=24_262,
        max_external_ancestry_authorities=(
            job_manager_v2._job_external_authority_limit(
                operation="research_lab_allocation",
                purpose="research_lab.allocation.v2",
            )
        ),
    )
    for index in range(job_manager_v2.MAX_ALLOCATION_ANCESTRY_AUTHORITIES):
        root = "sha256:" + format(index, "064x")
        allocation_context.record_external_receipt_graph(
            {
                "root_receipt_hash": root,
                "receipts": [{"receipt_hash": root}],
            }
        )

    assert (
        len(allocation_context.external_receipt_graphs)
        == job_manager_v2.MAX_ALLOCATION_ANCESTRY_AUTHORITIES
    )
    with pytest.raises(
        ExecutionJobV2Error,
        match="external receipt graph count exceeds limit",
    ):
        allocation_context.record_external_receipt_graph(
            {
                "root_receipt_hash": "sha256:" + "e" * 64,
                "receipts": [{"receipt_hash": "sha256:" + "e" * 64}],
            }
        )

    assert len(context.external_receipt_graphs) == 128
    with pytest.raises(
        ExecutionJobV2Error,
        match="external receipt graph count exceeds limit",
    ):
        context.record_external_receipt_graph(
            {
                "root_receipt_hash": "sha256:" + "f" * 64,
                "receipts": [{"receipt_hash": "sha256:" + "f" * 64}],
            }
        )


def test_transport_profile_mismatch_identifies_provider_and_hash_prefixes():
    context = ExecutionContextV2(
        job_id="score-job-1",
        purpose="research_lab.candidate_score.v2",
        epoch_id=24_000,
        provider_credential_profile="provider_preflight",
        provider_credential_ref_hashes={
            "exa": HASH,
            "egress_proxy": HASH,
        },
    )
    attempt = build_transport_attempt(
        request_id="1" * 32,
        logical_operation_id="provider-op-1",
        job_id=context.job_id,
        purpose=context.purpose,
        provider_id="exa",
        attempt_number=0,
        method="POST",
        destination_host="api.exa.ai",
        destination_port=443,
        path_hash=HASH,
        nonsecret_headers_hash=HASH,
        body_hash=HASH_B,
        credential_ref_hash=HASH_B,
        retry_policy_hash=HASH_B,
        timeout_ms=30_000,
        started_at=NOW,
        terminal_status="authenticated_response",
        http_status=200,
        response_hash=HASH,
        request_artifact_hash=HASH,
        response_artifact_hash=HASH_B,
        tls_peer_chain_hash=HASH,
        tls_protocol="TLSv1.3",
        failure_code=None,
        completed_at=NOW,
        egress_proxy_ref_hash=HASH,
    )

    with pytest.raises(
        ExecutionJobV2Error,
        match=(
            r"provider exa "
            r"\(expected=sha256:aaaaaaaa observed=sha256:bbbbbbbb\)"
        ),
    ):
        context.record_transport(attempt)


def _profile_transport_attempt(
    context,
    *,
    provider_id,
    logical_operation_id,
    egress_proxy_ref_hash,
    credential_ref_hash=HASH,
    job_id=None,
    purpose=None,
):
    return build_transport_attempt(
        request_id="2" * 32,
        logical_operation_id=logical_operation_id,
        job_id=job_id or context.job_id,
        purpose=purpose or context.purpose,
        provider_id=provider_id,
        attempt_number=0,
        method="POST",
        destination_host="qplwoislplkcegvdmbim.supabase.co",
        destination_port=443,
        path_hash=HASH,
        nonsecret_headers_hash=HASH,
        body_hash=HASH_B,
        credential_ref_hash=credential_ref_hash,
        retry_policy_hash=HASH_B,
        timeout_ms=30_000,
        started_at=NOW,
        terminal_status="authenticated_response",
        http_status=200,
        response_hash=HASH,
        request_artifact_hash=HASH,
        response_artifact_hash=HASH_B,
        tls_peer_chain_hash=HASH,
        tls_protocol="TLSv1.3",
        failure_code=None,
        completed_at=NOW,
        egress_proxy_ref_hash=egress_proxy_ref_hash,
    )


@pytest.mark.parametrize(
    "namespace",
    ("provider-outcome", "provider-evidence-cache"),
)
def test_transport_profile_accepts_only_job_bound_direct_supabase_sidecars(
    namespace,
):
    context = ExecutionContextV2(
        job_id="score-job-sidecars",
        purpose="research_lab.provider_preflight.v2",
        epoch_id=24_000,
        provider_credential_profile="provider_preflight",
        provider_credential_ref_hashes={
            "exa": HASH,
            "scrapingdog": HASH_B,
            "egress_proxy": HASH,
        },
    )
    context.record_transport(
        _profile_transport_attempt(
            context,
            provider_id="supabase",
            logical_operation_id=(
                "%s:%s:1:append" % (context.job_id, namespace)
            ),
            egress_proxy_ref_hash=DIRECT_EGRESS_REF_HASH,
            credential_ref_hash=HASH_B,
        )
    )

    assert context.transport_attempts[0]["provider_id"] == "supabase"
    assert (
        context.transport_attempts[0]["egress_proxy_ref_hash"]
        == DIRECT_EGRESS_REF_HASH
    )


@pytest.mark.parametrize(
    ("provider_id", "logical_operation_id", "egress_proxy_ref_hash"),
    (
        ("supabase", "score-job-sidecars:request-fingerprint", DIRECT_EGRESS_REF_HASH),
        ("supabase", "other-job:provider-outcome:1:append", DIRECT_EGRESS_REF_HASH),
        ("supabase", "score-job-sidecars:provider-outcome:1:append", HASH),
        ("exa", "score-job-sidecars:provider-outcome:1:append", DIRECT_EGRESS_REF_HASH),
    ),
)
def test_transport_profile_rejects_noncanonical_direct_sidecars(
    provider_id,
    logical_operation_id,
    egress_proxy_ref_hash,
):
    context = ExecutionContextV2(
        job_id="score-job-sidecars",
        purpose="research_lab.provider_preflight.v2",
        epoch_id=24_000,
        provider_credential_profile="provider_preflight",
        provider_credential_ref_hashes={
            "exa": HASH,
            "scrapingdog": HASH_B,
            "egress_proxy": HASH,
        },
    )
    with pytest.raises(ExecutionJobV2Error, match="transport proxy differs"):
        context.record_transport(
            _profile_transport_attempt(
                context,
                provider_id=provider_id,
                logical_operation_id=logical_operation_id,
                egress_proxy_ref_hash=egress_proxy_ref_hash,
            )
        )


@pytest.mark.parametrize(
    ("job_id", "purpose"),
    (
        ("different-job", None),
        (None, "research_lab.company_score.v2"),
    ),
)
def test_direct_supabase_sidecar_cannot_cross_execution_scope(job_id, purpose):
    context = ExecutionContextV2(
        job_id="score-job-sidecars",
        purpose="research_lab.provider_preflight.v2",
        epoch_id=24_000,
        provider_credential_profile="provider_preflight",
        provider_credential_ref_hashes={
            "exa": HASH,
            "egress_proxy": HASH,
        },
    )
    with pytest.raises(ExecutionJobV2Error, match="differs from execution scope"):
        context.record_transport(
            _profile_transport_attempt(
                context,
                provider_id="supabase",
                logical_operation_id=(
                    "%s:provider-outcome:1:append" % context.job_id
                ),
                egress_proxy_ref_hash=DIRECT_EGRESS_REF_HASH,
                job_id=job_id,
                purpose=purpose,
            )
        )


def test_direct_supabase_sidecar_requires_direct_route_without_worker_proxy():
    context = ExecutionContextV2(
        job_id="score-job-sidecars",
        purpose="research_lab.provider_preflight.v2",
        epoch_id=24_000,
        provider_credential_profile="provider_preflight",
        provider_credential_ref_hashes={"exa": HASH},
    )
    with pytest.raises(ExecutionJobV2Error, match="transport proxy differs"):
        context.record_transport(
            _profile_transport_attempt(
                context,
                provider_id="supabase",
                logical_operation_id=(
                    "%s:provider-outcome:1:append" % context.job_id
                ),
                egress_proxy_ref_hash=HASH,
            )
        )


def _manifest(payload, **overrides):
    value = {
        "schema_version": JOB_SCHEMA_VERSION,
        "job_id": "score-job-1",
        "operation": "score",
        "purpose": "research_lab.candidate_score.v2",
        "epoch_id": 24000,
        "sequence": 1,
        "payload_sha256": sha256_bytes(payload),
        "payload_size_bytes": len(payload),
        "parent_receipt_hashes": [],
        "input_artifact_hashes": [HASH_B],
        "provider_credential_profile": "default",
        "provider_credential_ref_hashes": {},
    }
    value.update(overrides)
    return value


def _run(manager, payload, manifest=None):
    manifest = manifest or _manifest(payload)
    manager.submit(manifest)
    manager.put_chunk(
        job_id=manifest["job_id"],
        offset=0,
        data_b64=base64.b64encode(payload).decode("ascii"),
        chunk_sha256=sha256_bytes(payload),
    )
    manager.seal(manifest["job_id"])
    deadline = time.time() + 2
    while time.time() < deadline:
        status = manager.status(manifest["job_id"])
        if status["state"] in {"succeeded", "failed"}:
            return status
        time.sleep(0.01)
    raise AssertionError("V2 job did not terminate")


def _checkpointed_parent_graph():
    manager, boot = _manager(
        lambda _operation, payload, _context: payload,
        checkpoint_lineage=True,
    )
    payload = _payload()
    assert _run(manager, payload)["state"] == "succeeded"
    receipt = manager.receipt("score-job-1")
    return build_checkpointed_receipt_graph(
        root_receipt_hash=receipt["receipt_hash"],
        boot_identities=(boot,),
        receipts=manager.receipts("score-job-1"),
        transport_attempts=manager.transport_attempts("score-job-1"),
        host_operations=manager.host_operations("score-job-1"),
        ancestry_lineage_id=HASH,
        ancestry_proof=manager.ancestry_compact_proof("score-job-1"),
        boot_attestation_verifier=lambda identity: identity,
        require_boot_attestation_verification=True,
    )


def test_parent_receipt_graph_set_preserves_membership_and_deduplicates():
    boot = {"boot_identity_hash": HASH, "marker": "shared-boot"}
    shared_receipt = {"receipt_hash": HASH, "marker": "shared-receipt"}
    child_a = {"receipt_hash": "sha256:" + "1" * 64, "marker": "a"}
    child_b = {"receipt_hash": "sha256:" + "2" * 64, "marker": "b"}
    graphs = [
        {
            "schema_version": "leadpoet.attested_receipt_graph.v2",
            "root_receipt_hash": child_a["receipt_hash"],
            "boot_identities": [boot],
            "receipts": [shared_receipt, child_a],
            "transport_attempts": [],
            "host_operations": [],
        },
        {
            "schema_version": "leadpoet.attested_receipt_graph.v2",
            "root_receipt_hash": child_b["receipt_hash"],
            "boot_identities": [boot],
            "receipts": [shared_receipt, child_b],
            "transport_attempts": [],
            "host_operations": [],
        },
    ]

    packed = pack_parent_receipt_graph_set_v2(graphs)

    assert len(packed["boot_identities"]) == 1
    assert len(packed["receipts"]) == 3
    unpacked = unpack_parent_receipt_graph_set_v2(packed)
    assert unpacked == graphs
    assert unpacked[0]["boot_identities"][0] is unpacked[1]["boot_identities"][0]
    assert unpacked[0]["receipts"][0] is unpacked[1]["receipts"][0]


def test_parent_receipt_graph_set_preserves_checkpoint_authority_and_reads_v2():
    checkpointed = _checkpointed_parent_graph()
    ordinary = {
        "schema_version": "leadpoet.attested_receipt_graph.v2",
        "root_receipt_hash": HASH_B,
        "boot_identities": [],
        "receipts": [{"receipt_hash": HASH_B}],
        "transport_attempts": [],
        "host_operations": [],
    }

    packed = pack_parent_receipt_graph_set_v2((ordinary, checkpointed))

    assert packed["schema_version"] == "leadpoet.parent_receipt_graph_set.v3"
    assert unpack_parent_receipt_graph_set_v2(packed) == [ordinary, checkpointed]
    assert packed["graphs"][1]["ancestry_proof"] == (
        checkpointed["ancestry_proof"]
    )

    malformed = json.loads(json.dumps(packed))
    malformed["graphs"][1].pop("ancestry_proof")
    with pytest.raises(ExecutionJobV2Error, match="descriptor fields are invalid"):
        unpack_parent_receipt_graph_set_v2(malformed)

    legacy = pack_parent_receipt_graph_set_v2((ordinary,))
    legacy["schema_version"] = "leadpoet.parent_receipt_graph_set.v2"
    legacy["graphs"][0].pop("schema_version")
    assert unpack_parent_receipt_graph_set_v2(legacy) == [ordinary]


def test_parent_receipt_graph_set_count_exception_is_explicit_and_scoped():
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
        for index in range(job_manager_v2.MAX_EXTERNAL_RECEIPT_GRAPHS + 1)
    ]

    with pytest.raises(
        ExecutionJobV2Error,
        match="external receipt graph count exceeds limit",
    ):
        pack_parent_receipt_graph_set_v2(graphs)

    packed = pack_parent_receipt_graph_set_v2(
        graphs,
        max_graph_count=job_manager_v2.MAX_ALLOCATION_ANCESTRY_AUTHORITIES,
    )
    with pytest.raises(
        ExecutionJobV2Error,
        match="parent receipt graph set count is invalid",
    ):
        unpack_parent_receipt_graph_set_v2(packed)
    assert unpack_parent_receipt_graph_set_v2(
        packed,
        max_graph_count=job_manager_v2.MAX_ALLOCATION_ANCESTRY_AUTHORITIES,
    ) == graphs


def test_parent_receipt_graph_set_rejects_conflicts_and_unreferenced_evidence():
    graph = {
        "schema_version": "leadpoet.attested_receipt_graph.v2",
        "root_receipt_hash": HASH,
        "boot_identities": [],
        "receipts": [{"receipt_hash": HASH, "marker": "first"}],
        "transport_attempts": [],
        "host_operations": [],
    }
    conflicting = {
        **graph,
        "root_receipt_hash": HASH_B,
        "receipts": [{"receipt_hash": HASH, "marker": "second"}],
    }
    with pytest.raises(ExecutionJobV2Error, match="conflicts for hash"):
        pack_parent_receipt_graph_set_v2((graph, conflicting))

    packed = pack_parent_receipt_graph_set_v2((graph,))
    packed["receipts"].append(
        {"receipt_hash": HASH_B, "marker": "unreferenced"}
    )
    with pytest.raises(ExecutionJobV2Error, match="unreferenced evidence"):
        unpack_parent_receipt_graph_set_v2(packed)


def test_parent_receipt_graph_set_rejects_missing_and_unknown_evidence():
    graph = {
        "schema_version": "leadpoet.attested_receipt_graph.v2",
        "root_receipt_hash": HASH,
        "boot_identities": [],
        "receipts": [{"receipt_hash": HASH}],
        "transport_attempts": [],
        "host_operations": [],
    }
    missing = pack_parent_receipt_graph_set_v2((graph,))
    missing["graphs"][0]["receipt_hashes"] = [HASH_B]
    with pytest.raises(ExecutionJobV2Error, match="reference is missing"):
        unpack_parent_receipt_graph_set_v2(missing)

    unknown = pack_parent_receipt_graph_set_v2((graph,))
    unknown["unexpected"] = True
    with pytest.raises(ExecutionJobV2Error, match="fields are invalid"):
        unpack_parent_receipt_graph_set_v2(unknown)


def test_job_rejects_multiple_parent_graph_encodings_before_execution():
    executed = []
    manager, _ = _manager(lambda *_args: executed.append(True) or {})
    graph = {
        "schema_version": "leadpoet.attested_receipt_graph.v2",
        "root_receipt_hash": HASH,
        "boot_identities": [],
        "receipts": [{"receipt_hash": HASH}],
        "transport_attempts": [],
        "host_operations": [],
    }
    payload = json.dumps(
        {
            "input": 3,
            PARENT_RECEIPT_GRAPHS_FIELD: [graph],
            PARENT_RECEIPT_GRAPH_SET_FIELD: pack_parent_receipt_graph_set_v2(
                (graph,)
            ),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    status = _run(
        manager,
        payload,
        _manifest(payload, parent_receipt_hashes=[HASH]),
    )

    assert status["state"] == "failed"
    assert executed == []


def test_large_input_exception_is_scoped_to_allocation_ancestry_consumers():
    oversized = 64 * 1024 * 1024 + 1
    allowed_scopes = {
        "research_lab_allocation": {"research_lab.allocation.v2"},
        "attest_artifact_persistence": {
            "leadpoet.artifact_persistence.v2",
        },
        "attest_weight_input": {
            "research_lab.allocation.v2",
            "research_lab.champion_input.v2",
            "research_lab.reimbursement_input.v2",
            "research_lab.source_add_reward_input.v2",
            "research_lab.anomaly_adjustment_input.v2",
        },
        "attest_weight_publication": {"gateway.weights.publication.v2"},
    }
    for operation, purposes in allowed_scopes.items():
        for purpose in purposes:
            normalized = job_manager_v2._manifest(
                _manifest(
                    b"{}",
                    operation=operation,
                    purpose=purpose,
                    payload_size_bytes=oversized,
                ),
                role="gateway_coordinator",
                operations=allowed_scopes,
            )
            assert normalized["payload_size_bytes"] == oversized

    with pytest.raises(
        ExecutionJobV2Error,
        match="payload size is outside limit",
    ):
        job_manager_v2._manifest(
            _manifest(b"{}", payload_size_bytes=oversized),
            role="gateway_scoring",
            operations={"score": {"research_lab.candidate_score.v2"}},
        )

    with pytest.raises(
        ExecutionJobV2Error,
        match="payload size is outside limit",
    ):
        job_manager_v2._manifest(
            _manifest(
                b"{}",
                operation="unrelated_operation",
                purpose="research_lab.allocation.v2",
                payload_size_bytes=oversized,
            ),
            role="gateway_coordinator",
            operations={
                "unrelated_operation": {"research_lab.allocation.v2"},
            },
        )

    with pytest.raises(
        ExecutionJobV2Error,
        match="payload size is outside limit",
    ):
        job_manager_v2._manifest(
            {
                **_manifest(
                    b"{}",
                    operation="research_lab_allocation",
                    purpose="research_lab.allocation.v2",
                ),
                "payload_size_bytes": (
                    job_manager_v2.MAX_ALLOCATION_ANCESTRY_INPUT_BYTES + 1
                ),
            },
            role="gateway_coordinator",
            operations={
                "research_lab_allocation": {"research_lab.allocation.v2"},
            },
        )


def test_large_ancestry_scope_is_applied_to_external_parent_graphs(monkeypatch):
    monkeypatch.setattr(
        job_manager_v2,
        "validate_receipt_graph",
        lambda _graph, **_kwargs: (),
    )
    assert job_manager_v2._job_input_limit_bytes(
        operation="attest_weight_publication",
        purpose="gateway.weights.publication.v2",
    ) == job_manager_v2.MAX_ALLOCATION_ANCESTRY_INPUT_BYTES
    assert job_manager_v2._job_input_limit_bytes(
        operation="score",
        purpose="research_lab.candidate_score.v2",
    ) == 64 * 1024 * 1024
    assert job_manager_v2._job_external_authority_limit(
        operation="attest_weight_publication",
        purpose="gateway.weights.publication.v2",
    ) == job_manager_v2.MAX_ALLOCATION_ANCESTRY_AUTHORITIES
    assert job_manager_v2._job_external_authority_limit(
        operation="score",
        purpose="research_lab.candidate_score.v2",
    ) == job_manager_v2.MAX_EXTERNAL_RECEIPT_GRAPHS

    graph = {
        "root_receipt_hash": HASH,
        "receipts": [{"receipt_hash": HASH}],
    }
    constrained = ExecutionContextV2(
        job_id="score-job",
        purpose="research_lab.candidate_score.v2",
        epoch_id=24_000,
        max_external_receipt_graph_bytes=2,
    )
    with pytest.raises(
        ExecutionJobV2Error,
        match="external receipt graph exceeds size limit",
    ):
        constrained.record_external_receipt_graph(graph)

    allocation_lineage = ExecutionContextV2(
        job_id="allocation-lineage-job",
        purpose="leadpoet.artifact_persistence.v2",
        epoch_id=24_000,
        max_external_receipt_graph_bytes=(
            job_manager_v2._job_input_limit_bytes(
                operation="attest_artifact_persistence",
                purpose="leadpoet.artifact_persistence.v2",
            )
        ),
    )
    assert allocation_lineage.record_external_receipt_graph(graph) == HASH


def test_success_receipt_binds_transport_artifacts_and_signed_transition():
    attempt = build_transport_attempt(
        request_id="1" * 32,
        logical_operation_id="provider-op-1",
        job_id="score-job-1",
        purpose="research_lab.candidate_score.v2",
        provider_id="openrouter",
        attempt_number=0,
        method="POST",
        destination_host="openrouter.ai",
        destination_port=443,
        path_hash=HASH,
        nonsecret_headers_hash=HASH,
        body_hash=HASH_B,
        credential_ref_hash=HASH,
        retry_policy_hash=HASH_B,
        timeout_ms=30000,
        started_at=NOW,
        terminal_status="authenticated_response",
        http_status=200,
        response_hash=HASH,
        request_artifact_hash=HASH,
        response_artifact_hash=HASH_B,
        tls_peer_chain_hash=HASH,
        tls_protocol="TLSv1.3",
        failure_code=None,
        completed_at=NOW,
    )

    def _executor(operation, payload, context):
        assert operation == "score"
        assert context.parent_receipt_hashes == ()
        context.record_transport(attempt)
        context.record_artifact(HASH)
        return ExecutionResultV2(
            output={"score": payload["input"] * 2},
            transitions=(
                TransitionSpecV2(
                    operation="insert",
                    target="research_lab_score_bundles",
                    idempotency_key="score-job-1",
                    expected_state_hash=HASH,
                    payload_hash=HASH_B,
                ),
            ),
        )

    manager, boot = _manager(_executor)
    status = _run(manager, _payload())
    assert status["state"] == "succeeded"
    receipt = manager.receipt("score-job-1")
    validate_signed_execution_receipt(receipt)
    assert receipt["boot_identity_hash"] == boot["boot_identity_hash"]
    assert receipt["transport_root"] == merkle_root(
        [attempt["attempt_hash"]], domain="leadpoet-transport-v2"
    )
    assert receipt["artifact_root"] == merkle_root(
        [HASH, HASH_B], domain="leadpoet-artifact-v2"
    )
    transition = manager.transitions("score-job-1")[0]
    validate_signed_transition_command(transition)
    assert transition["receipt_hash"] == receipt["receipt_hash"]
    result = manager.result_chunk(job_id="score-job-1")
    assert json.loads(base64.b64decode(result["data_b64"])) == {"score": 6}


def test_capacity_evicts_only_the_oldest_terminal_job(monkeypatch):
    monkeypatch.setattr(
        "gateway.tee.execution_job_manager_v2.MAX_JOB_COUNT",
        2,
    )
    monkeypatch.setattr(
        "gateway.tee.execution_job_manager_v2.MIN_TERMINAL_EVICTION_AGE_SECONDS",
        0,
    )
    manager, _ = _manager(
        lambda _operation, payload, _context: {"value": payload["input"]}
    )
    payload = _payload()
    first = _manifest(payload, job_id="score-job-1")
    second = _manifest(payload, job_id="score-job-2")
    third = _manifest(payload, job_id="score-job-3")

    assert _run(manager, payload, first)["state"] == "succeeded"
    assert _run(manager, payload, second)["state"] == "succeeded"
    submitted = manager.submit(third)

    assert submitted["state"] == "uploading"
    with pytest.raises(ExecutionJobV2Error, match="job was not found"):
        manager.status(first["job_id"])
    assert manager.status(second["job_id"])["state"] == "succeeded"
    health = manager.health()
    assert health["terminal_eviction_count"] == 1
    assert health["job_counts"] == {
        "succeeded": 1,
        "uploading": 1,
    }


def test_capacity_never_evicts_nonterminal_jobs(monkeypatch):
    monkeypatch.setattr(
        "gateway.tee.execution_job_manager_v2.MAX_JOB_COUNT",
        2,
    )
    manager, _ = _manager(lambda _operation, _payload, _context: {})
    payload = _payload()
    manager.submit(_manifest(payload, job_id="score-job-1"))
    manager.submit(_manifest(payload, job_id="score-job-2"))

    with pytest.raises(ExecutionJobV2Error, match="capacity is full"):
        manager.submit(_manifest(payload, job_id="score-job-3"))

    assert manager.health()["terminal_eviction_count"] == 0


def test_capacity_does_not_evict_recent_terminal_jobs(monkeypatch):
    monkeypatch.setattr(
        "gateway.tee.execution_job_manager_v2.MAX_JOB_COUNT",
        1,
    )
    manager, _ = _manager(lambda _operation, _payload, _context: {})
    payload = _payload()
    assert _run(manager, payload)["state"] == "succeeded"

    with pytest.raises(ExecutionJobV2Error, match="capacity is full"):
        manager.submit(_manifest(payload, job_id="score-job-2"))

    assert manager.status("score-job-1")["state"] == "succeeded"
    assert manager.health()["terminal_eviction_count"] == 0


def test_receipt_output_projection_binds_authoritative_result_only():
    full_output = {
        "allocation": {"allocation_hash": HASH},
        "source_state": {"epoch": 24000},
    }
    receipt_output = {"allocation": full_output["allocation"]}
    manager, _boot = _manager(
        lambda _operation, _payload, _context: ExecutionResultV2(
            output=full_output,
            receipt_output=receipt_output,
        )
    )
    payload = _payload()
    status = _run(
        manager,
        payload,
        _manifest(payload, parent_receipt_hashes=[]),
    )

    assert status["state"] == "succeeded"
    receipt = manager.receipt("score-job-1")
    assert receipt["output_root"] == sha256_bytes(
        json.dumps(
            receipt_output,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    assert status["result_sha256"] == sha256_bytes(
        json.dumps(
            full_output,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    result = manager.result_chunk(job_id="score-job-1")
    assert json.loads(base64.b64decode(result["data_b64"])) == full_output


def test_executor_failure_has_signed_failure_receipt_and_canonical_terminal_result():
    def _executor(_operation, _payload, _context):
        raise RuntimeError("private detail must not leave enclave")

    manager, _ = _manager(_executor)
    status = _run(manager, _payload())
    assert status["state"] == "failed"
    assert "private detail" not in str(status)
    receipt = manager.receipt("score-job-1")
    validate_signed_execution_receipt(receipt)
    assert receipt["status"] == "failed"
    assert receipt["failure_code"] == "execution_runtimeerror"
    result = manager.result_chunk(job_id="score-job-1")
    terminal = json.loads(base64.b64decode(result["data_b64"]))
    assert terminal == {
        "status": "failed",
        "failure_code": receipt["failure_code"],
    }
    assert receipt["output_root"] == sha256_bytes(
        json.dumps(terminal, sort_keys=True, separators=(",", ":")).encode()
    )



def test_stage_receipts_form_a_measured_chain_before_root_receipt():
    def _executor(_operation, payload, context):
        context.record_stage(
            purpose="research_lab.baseline_score.v2",
            input_root=sha256_bytes(json.dumps(payload, sort_keys=True).encode()),
            output_root=HASH_B,
            artifact_hashes=(HASH,),
        )
        return {"score": payload["input"]}

    manager, _ = _manager(_executor)
    status = _run(manager, _payload())
    assert status["state"] == "succeeded"
    receipts = manager.receipts("score-job-1")
    assert len(receipts) == 2
    stage, root = receipts
    assert stage["purpose"] == "research_lab.baseline_score.v2"
    assert root["parent_receipt_hashes"] == [stage["receipt_hash"]]
    assert stage["parent_receipt_hashes"] == []


def test_nested_receipt_graph_is_bound_to_root_and_retained_for_graph_merge():
    nested_key = Ed25519PrivateKey.generate()
    nested_pubkey = nested_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    ).hex()
    nested_boot = create_boot_identity(
        body=build_boot_identity_body(
            role="gateway_scoring",
            physical_role="gateway_scoring",
            commit_sha="1" * 40,
            pcr0="2" * 96,
            build_manifest_hash=HASH,
            dependency_lock_hash=HASH_B,
            config_hash=HASH,
            boot_nonce="3" * 32,
            signing_pubkey=nested_pubkey,
            transport_pubkey="4" * 64,
            transport_certificate_hash=HASH_B,
            attestation_user_data_hash=HASH,
            issued_at=NOW,
        ),
        attestation_document_b64=base64.b64encode(b"nested-nitro").decode("ascii"),
    )
    nested_receipt = create_signed_execution_receipt(
        body=build_execution_receipt_body(
            role="gateway_scoring",
            purpose="research_lab.candidate_test.v2",
            job_id="nested-dev-score",
            epoch_id=24000,
            sequence=0,
            commit_sha="1" * 40,
            pcr0="2" * 96,
            build_manifest_hash=HASH,
            dependency_lock_hash=HASH_B,
            config_hash=HASH,
            boot_identity_hash=nested_boot["boot_identity_hash"],
            input_root=HASH,
            output_root=HASH_B,
            transport_root_hash=merkle_root((), domain="leadpoet-transport-v2"),
            host_operation_root_hash=merkle_root(
                (), domain="leadpoet-host-operation-v2"
            ),
            artifact_root=merkle_root((), domain="leadpoet-artifact-v2"),
            parent_receipt_hashes=(),
            status="succeeded",
            failure_code=None,
            issued_at=NOW,
        ),
        enclave_pubkey=nested_pubkey,
        sign_digest=nested_key.sign,
    )
    nested_graph = build_receipt_graph(
        root_receipt_hash=nested_receipt["receipt_hash"],
        boot_identities=(nested_boot,),
        receipts=(nested_receipt,),
        transport_attempts=(),
        host_operations=(),
    )

    def _executor(_operation, payload, context):
        assert context.parent_receipt_hashes == (nested_receipt["receipt_hash"],)
        assert context.external_receipt_graphs == [nested_graph]
        context.record_stage(
            purpose="research_lab.baseline_score.v2",
            input_root=sha256_bytes(json.dumps(payload, sort_keys=True).encode()),
            output_root=HASH_B,
        )
        return {"score": payload["input"]}

    manager, _ = _manager(_executor)
    payload = json.dumps(
        {
            "input": 3,
            PARENT_RECEIPT_GRAPHS_FIELD: [nested_graph],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    manifest = _manifest(
        payload,
        parent_receipt_hashes=[nested_receipt["receipt_hash"]],
    )
    assert _run(manager, payload, manifest)["state"] == "succeeded"
    receipts = manager.receipts("score-job-1")
    assert receipts[-1]["parent_receipt_hashes"] == sorted(
        [receipts[-2]["receipt_hash"], nested_receipt["receipt_hash"]]
    )
    assert manager.external_receipt_graphs("score-job-1") == (nested_graph,)
    assert manager.status("score-job-1")["external_receipt_graph_count"] == 1

    graph_set_manager, _ = _manager(_executor)
    graph_set_payload = json.dumps(
        {
            "input": 3,
            PARENT_RECEIPT_GRAPH_SET_FIELD: pack_parent_receipt_graph_set_v2(
                (nested_graph,)
            ),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    graph_set_manifest = _manifest(
        graph_set_payload,
        parent_receipt_hashes=[nested_receipt["receipt_hash"]],
    )
    assert (
        _run(graph_set_manager, graph_set_payload, graph_set_manifest)["state"]
        == "succeeded"
    )
    assert graph_set_manager.external_receipt_graphs("score-job-1") == (
        nested_graph,
    )


def test_checkpointed_parent_remains_visible_and_extends_certificate_generation():
    observed = []

    def _executor(_operation, payload, context):
        if payload["input"] == 4:
            assert len(context.external_receipt_graphs) == 1
            assert context.external_ancestry_proofs == []
            parent_graph = context.external_receipt_graphs[0]
            assert (
                parent_graph["schema_version"]
                == CHECKPOINTED_RECEIPT_GRAPH_SCHEMA_VERSION
            )
            assert parent_graph["root_receipt_hash"] in {
                receipt["receipt_hash"]
                for receipt in parent_graph["receipts"]
            }
            observed.append(parent_graph)
        return {"score": payload["input"]}

    manager, boot = _manager(_executor, checkpoint_lineage=True)
    first_payload = _payload()
    assert _run(manager, first_payload)["state"] == "succeeded"
    first_receipt = manager.receipt("score-job-1")
    first_proof = manager.ancestry_compact_proof("score-job-1")
    first_graph = build_checkpointed_receipt_graph(
        root_receipt_hash=first_receipt["receipt_hash"],
        boot_identities=(boot,),
        receipts=manager.receipts("score-job-1"),
        transport_attempts=manager.transport_attempts("score-job-1"),
        host_operations=manager.host_operations("score-job-1"),
        ancestry_lineage_id=HASH,
        ancestry_proof=first_proof,
        boot_attestation_verifier=lambda identity: identity,
        require_boot_attestation_verification=True,
    )

    second_payload = json.dumps(
        {
            "input": 4,
            PARENT_RECEIPT_GRAPHS_FIELD: [first_graph],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    second_manifest = _manifest(
        second_payload,
        job_id="score-job-2",
        parent_receipt_hashes=[first_receipt["receipt_hash"]],
    )
    assert _run(manager, second_payload, second_manifest)["state"] == "succeeded"

    assert observed == [first_graph]
    second_claim = manager.ancestry_compact_proof("score-job-2")["certificate"][
        "claim"
    ]
    assert second_claim["certificate_sequence"] == 1
    assert len(second_claim["parent_authorities"]) == 1
    assert second_claim["parent_authorities"][0]["authority_kind"] == "certificate"
    assert second_claim["parent_authorities"][0]["parent_receipt_hash"] == (
        first_receipt["receipt_hash"]
    )


def test_manager_issues_bootstrap_proofs_before_normal_session_receipt():
    source_manager, source_boot = _manager(
        lambda _operation, payload, _context: payload
    )
    source_payload = _payload()
    assert _run(source_manager, source_payload)["state"] == "succeeded"
    source_receipt = source_manager.receipt("score-job-1")
    source_graph = build_receipt_graph(
        root_receipt_hash=source_receipt["receipt_hash"],
        boot_identities=(source_boot,),
        receipts=source_manager.receipts("score-job-1"),
        transport_attempts=source_manager.transport_attempts("score-job-1"),
        host_operations=source_manager.host_operations("score-job-1"),
    )

    request_schema = ANCESTRY_CHECKPOINT_BOOTSTRAP_REQUEST_SCHEMA_VERSION

    def bootstrap_executor(_operation, payload, _context):
        return ExecutionResultV2(
            output={
                "schema_version": request_schema,
                "selected_root_receipt_hashes": list(
                    payload["selected_root_receipt_hashes"]
                ),
            },
            ancestry_checkpoint_bootstrap=True,
        )

    manager, _ = _manager(
        bootstrap_executor,
        checkpoint_lineage=True,
        role="gateway_coordinator",
        operations={
            "ancestry_checkpoint_bootstrap_v2": {
                "research_lab.ancestry_checkpoint_bootstrap.v2"
            }
        },
    )
    payload = json.dumps(
        {
            "schema_version": request_schema,
            "selected_root_receipt_hashes": [source_receipt["receipt_hash"]],
            PARENT_RECEIPT_GRAPHS_FIELD: [source_graph],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    manifest = _manifest(
        payload,
        job_id="ancestry-bootstrap-1",
        operation="ancestry_checkpoint_bootstrap_v2",
        purpose="research_lab.ancestry_checkpoint_bootstrap.v2",
        parent_receipt_hashes=[source_receipt["receipt_hash"]],
    )
    assert _run(manager, payload, manifest)["state"] == "succeeded"
    result_chunk = manager.result_chunk(job_id="ancestry-bootstrap-1")
    result = json.loads(base64.b64decode(result_chunk["data_b64"]))
    assert (
        result["schema_version"]
        == ANCESTRY_CHECKPOINT_BOOTSTRAP_RESULT_SCHEMA_VERSION
    )
    assert len(result["checkpoint_proofs"]) == 1
    assert result["checkpoint_root_receipt_hashes"] == [
        source_receipt["receipt_hash"]
    ]
    validate_signed_execution_receipt(manager.receipt("ancestry-bootstrap-1"))
    assert manager.ancestry_compact_proof("ancestry-bootstrap-1")


def test_nested_graph_covers_multiple_declared_parent_receipts():
    nested_key = Ed25519PrivateKey.generate()
    nested_pubkey = nested_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    ).hex()
    nested_boot = create_boot_identity(
        body=build_boot_identity_body(
            role="gateway_scoring",
            physical_role="gateway_scoring",
            commit_sha="1" * 40,
            pcr0="2" * 96,
            build_manifest_hash=HASH,
            dependency_lock_hash=HASH_B,
            config_hash=HASH,
            boot_nonce="3" * 32,
            signing_pubkey=nested_pubkey,
            transport_pubkey="4" * 64,
            transport_certificate_hash=HASH_B,
            attestation_user_data_hash=HASH,
            issued_at=NOW,
        ),
        attestation_document_b64=base64.b64encode(b"nested-nitro").decode("ascii"),
    )

    def receipt(*, purpose, job_id, parents):
        return create_signed_execution_receipt(
            body=build_execution_receipt_body(
                role="gateway_scoring",
                purpose=purpose,
                job_id=job_id,
                epoch_id=24000,
                sequence=0,
                commit_sha="1" * 40,
                pcr0="2" * 96,
                build_manifest_hash=HASH,
                dependency_lock_hash=HASH_B,
                config_hash=HASH,
                boot_identity_hash=nested_boot["boot_identity_hash"],
                input_root=HASH,
                output_root=HASH_B,
                transport_root_hash=merkle_root(
                    (), domain="leadpoet-transport-v2"
                ),
                host_operation_root_hash=merkle_root(
                    (), domain="leadpoet-host-operation-v2"
                ),
                artifact_root=merkle_root((), domain="leadpoet-artifact-v2"),
                parent_receipt_hashes=parents,
                status="succeeded",
                failure_code=None,
                issued_at=NOW,
            ),
            enclave_pubkey=nested_pubkey,
            sign_digest=nested_key.sign,
        )

    parent = receipt(
        purpose="research_lab.candidate_test.v2",
        job_id="nested-parent",
        parents=(),
    )
    child = receipt(
        purpose="research_lab.candidate_score.v2",
        job_id="nested-child",
        parents=(parent["receipt_hash"],),
    )
    nested_graph = build_receipt_graph(
        root_receipt_hash=child["receipt_hash"],
        boot_identities=(nested_boot,),
        receipts=(parent, child),
        transport_attempts=(),
        host_operations=(),
    )

    def _executor(_operation, _payload, context):
        assert set(context.parent_receipt_hashes) == {
            parent["receipt_hash"],
            child["receipt_hash"],
        }
        assert context.external_receipt_graphs == [nested_graph]
        return {"score": 3}

    manager, _ = _manager(_executor)
    payload = json.dumps(
        {
            "input": 3,
            PARENT_RECEIPT_GRAPHS_FIELD: [nested_graph],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    manifest = _manifest(
        payload,
        parent_receipt_hashes=[
            parent["receipt_hash"],
            child["receipt_hash"],
        ],
    )

    assert _run(manager, payload, manifest)["state"] == "succeeded"
    root = manager.receipts("score-job-1")[-1]
    assert set(root["parent_receipt_hashes"]) == {
        parent["receipt_hash"],
        child["receipt_hash"],
    }


def test_job_id_and_payload_are_immutable_and_canonical():
    manager, _ = _manager(lambda _op, value, _ctx: value)
    payload = _payload()
    manifest = _manifest(payload)
    manager.submit(manifest)
    assert manager.submit(manifest)["manifest_hash"]
    with pytest.raises(ExecutionJobV2Error, match="another manifest"):
        manager.submit({**manifest, "epoch_id": 24001})
    changed_payload = b'{"input":4}'
    manager.put_chunk(
        job_id="score-job-1",
        offset=0,
        data_b64=base64.b64encode(changed_payload).decode("ascii"),
        chunk_sha256=sha256_bytes(changed_payload),
    )
    with pytest.raises(ExecutionJobV2Error, match="payload hash"):
        manager.seal("score-job-1")


def test_exact_duplicate_chunk_is_idempotent_but_conflicting_overlap_fails():
    manager, _ = _manager(lambda _op, value, _ctx: value)
    payload = _payload()
    manifest = _manifest(payload)
    manager.submit(manifest)
    encoded = base64.b64encode(payload).decode("ascii")
    digest = sha256_bytes(payload)

    first = manager.put_chunk(
        job_id=manifest["job_id"],
        offset=0,
        data_b64=encoded,
        chunk_sha256=digest,
    )
    duplicate = manager.put_chunk(
        job_id=manifest["job_id"],
        offset=0,
        data_b64=encoded,
        chunk_sha256=digest,
    )

    assert first["uploaded_bytes"] == len(payload)
    assert duplicate["uploaded_bytes"] == len(payload)
    conflicting = b"[" + payload[1:]
    with pytest.raises(ExecutionJobV2Error, match="conflicts with uploaded"):
        manager.put_chunk(
            job_id=manifest["job_id"],
            offset=0,
            data_b64=base64.b64encode(conflicting).decode("ascii"),
            chunk_sha256=sha256_bytes(conflicting),
        )


def test_v1_purpose_and_unknown_operation_fail_closed():
    manager, _ = _manager(lambda _op, value, _ctx: value)
    payload = _payload()
    with pytest.raises(ExecutionJobV2Error, match="not authorized"):
        manager.submit(
            _manifest(payload, purpose="research_lab.candidate_score.v1")
        )
    with pytest.raises(ExecutionJobV2Error, match="not authorized"):
        manager.submit(_manifest(payload, operation="blind_sign"))


@pytest.mark.parametrize(
    ("operation", "purpose"),
    (
        (
            "run_dev_hybrid_v2",
            "research_lab.candidate_hybrid_test.v2",
        ),
        (
            "run_model_sandbox_v2",
            "research_lab.candidate_hybrid_discovery.v2",
        ),
        (
            "run_model_sandbox_v2",
            "research_lab.model_compatibility.v2",
        ),
    ),
)
def test_scoring_role_authorizes_exact_hybrid_candidate_purposes(
    operation,
    purpose,
):
    payload = _payload()
    operations = {operation: {purpose}}
    manager, _ = _manager(
        lambda _operation, value, _ctx: value,
        operations=operations,
    )
    submitted = manager.submit(
        _manifest(
            payload,
            operation=operation,
            purpose=purpose,
            job_id="hybrid-job-1",
        )
    )
    assert submitted["state"] == "uploading"

    wrong_role_manager, _ = _manager(
        lambda _operation, value, _ctx: value,
        role="gateway_autoresearch",
        operations=operations,
    )
    with pytest.raises(ExecutionJobV2Error, match="not authorized"):
        wrong_role_manager.submit(
            _manifest(
                payload,
                operation=operation,
                purpose=purpose,
                job_id="wrong-role-hybrid-job-1",
            )
        )
