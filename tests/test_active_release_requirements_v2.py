from __future__ import annotations

import base64
from copy import deepcopy

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from gateway.tee import active_release_requirements_v2 as requirements_module
from gateway.tee.active_release_requirements_v2 import (
    ActiveReleaseRequirementsV2Error,
    build_active_release_requirements_v2,
    validate_active_release_requirements_v2,
)
from leadpoet_canonical.ancestry_checkpoint_v2 import (
    ANCESTRY_DELTA_SCHEMA_VERSION,
    build_compact_ancestry_proof_from_delta_v2,
    build_full_graph_parent_v2,
    issue_ancestry_certificate_v2,
)
from leadpoet_canonical.attested_v2 import (
    EMPTY_ARTIFACT_ROOT,
    EMPTY_HOST_OPERATION_ROOT,
    EMPTY_TRANSPORT_ROOT,
    SCORING_ROLE,
    build_boot_identity_body,
    build_checkpointed_receipt_graph,
    build_execution_receipt_body,
    build_receipt_graph,
    compact_checkpointed_receipt_graph,
    create_boot_identity,
    create_signed_execution_receipt,
)


LINEAGE_ID = "sha256:" + "a" * 64
CANDIDATE_COMMIT = "4" * 40
AUTHORITY_COMMIT = "6" * 40
RESTART_INVOCATION_ID = "gateway-24700-test"
TRANSITION_COMMIT = "5" * 40
GRAPH_COMMIT = "1" * 40
ISSUER_COMMIT = "2" * 40
OMITTED_PARENT_COMMIT = "3" * 40
NOW = "2026-08-24T00:00:00Z"


def _hash(character: str) -> str:
    return "sha256:" + character * 64


def _boot(commit: str, nonce: str):
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key().public_bytes_raw().hex()
    boot = create_boot_identity(
        body=build_boot_identity_body(
            role=SCORING_ROLE,
            physical_role=SCORING_ROLE,
            commit_sha=commit,
            pcr0=commit[0] * 96,
            build_manifest_hash=_hash("b"),
            dependency_lock_hash=_hash("c"),
            config_hash=_hash("d"),
            boot_nonce=nonce * 32,
            signing_pubkey=public_key,
            transport_pubkey="e" * 64,
            transport_certificate_hash=_hash("f"),
            attestation_user_data_hash=_hash("0"),
            issued_at=NOW,
        ),
        attestation_document_b64=base64.b64encode(b"nitro-attestation").decode("ascii"),
    )
    return private_key, public_key, boot


def _receipt(
    *,
    private_key,
    public_key: str,
    boot: dict,
    purpose: str,
    job_id: str,
    parents=(),
):
    return create_signed_execution_receipt(
        body=build_execution_receipt_body(
            role=SCORING_ROLE,
            purpose=purpose,
            job_id=job_id,
            epoch_id=24_700,
            sequence=0,
            commit_sha=boot["commit_sha"],
            pcr0=boot["pcr0"],
            build_manifest_hash=boot["build_manifest_hash"],
            dependency_lock_hash=boot["dependency_lock_hash"],
            config_hash=boot["config_hash"],
            boot_identity_hash=boot["boot_identity_hash"],
            input_root=_hash("1"),
            output_root=_hash("2"),
            transport_root_hash=EMPTY_TRANSPORT_ROOT,
            host_operation_root_hash=EMPTY_HOST_OPERATION_ROOT,
            artifact_root=EMPTY_ARTIFACT_ROOT,
            parent_receipt_hashes=parents,
            status="succeeded",
            failure_code=None,
            issued_at=NOW,
        ),
        enclave_pubkey=public_key,
        sign_digest=private_key.sign,
    )


@pytest.fixture
def active_graph_pair():
    parent_key, parent_public, parent_boot = _boot(OMITTED_PARENT_COMMIT, "3")
    parent_receipt = _receipt(
        private_key=parent_key,
        public_key=parent_public,
        boot=parent_boot,
        purpose="research_lab.baseline_score.v2",
        job_id="parent-baseline",
    )
    parent_graph = build_receipt_graph(
        root_receipt_hash=parent_receipt["receipt_hash"],
        boot_identities=[parent_boot],
        receipts=[parent_receipt],
        transport_attempts=[],
        host_operations=[],
    )

    graph_key, graph_public, graph_boot = _boot(GRAPH_COMMIT, "1")
    local_receipt = _receipt(
        private_key=graph_key,
        public_key=graph_public,
        boot=graph_boot,
        purpose="research_lab.candidate_score.v2",
        job_id="active-candidate",
        parents=(parent_receipt["receipt_hash"],),
    )
    delta = {
        "schema_version": ANCESTRY_DELTA_SCHEMA_VERSION,
        "root_receipt_hash": local_receipt["receipt_hash"],
        "boot_identities": [graph_boot],
        "receipts": [local_receipt],
        "transport_attempts": [],
        "host_operations": [],
    }
    issuer_key, _issuer_public, issuer_boot = _boot(ISSUER_COMMIT, "2")

    def allow_boot(identity):
        return identity

    certificate = issue_ancestry_certificate_v2(
        local_delta=delta,
        lineage_id=LINEAGE_ID,
        certificate_sequence=0,
        issuer_boot_identity=issuer_boot,
        issued_at=NOW,
        sign_digest=issuer_key.sign,
        boot_attestation_verifier=allow_boot,
        allowed_issuer_roles=(SCORING_ROLE,),
        parent_full_graphs=(build_full_graph_parent_v2(parent_graph),),
        required_purposes=("research_lab.candidate_score.v2",),
    )
    proof = build_compact_ancestry_proof_from_delta_v2(
        delta,
        certificate,
        expected_lineage_id=LINEAGE_ID,
        boot_attestation_verifier=allow_boot,
        allowed_issuer_roles=(SCORING_ROLE,),
    )
    checkpointed = build_checkpointed_receipt_graph(
        root_receipt_hash=local_receipt["receipt_hash"],
        boot_identities=[graph_boot],
        receipts=[local_receipt],
        transport_attempts=[],
        host_operations=[],
        ancestry_lineage_id=LINEAGE_ID,
        ancestry_proof=proof,
        boot_attestation_verifier=allow_boot,
        require_boot_attestation_verification=True,
    )
    compact = compact_checkpointed_receipt_graph(
        checkpointed,
        boot_attestation_verifier=allow_boot,
        require_boot_attestation_verification=True,
    )
    return checkpointed, compact, parent_graph


@pytest.mark.parametrize("graph_index", [0, 1], ids=["v3", "v4"])
def test_build_selects_exact_disclosed_releases_and_hashes_sidecar(
    active_graph_pair, graph_index
) -> None:
    graph = active_graph_pair[graph_index]
    root = graph["root_receipt_hash"]
    verified = []

    def release_nitro_verifier(identity):
        commit = identity["commit_sha"]
        if commit not in {GRAPH_COMMIT, ISSUER_COMMIT}:
            raise RuntimeError("release is not selected")
        verified.append(commit)
        return identity

    result = build_active_release_requirements_v2(
        candidate_commit_sha=CANDIDATE_COMMIT,
        authority_commit_sha=AUTHORITY_COMMIT,
        restart_invocation_id=RESTART_INVOCATION_ID,
        transition_commit_shas=(TRANSITION_COMMIT, CANDIDATE_COMMIT),
        active_graphs={root: graph},
        expected_lineage_id=LINEAGE_ID,
        boot_verifier=release_nitro_verifier,
    )

    assert result["commits_by_root"] == {root: [GRAPH_COMMIT, ISSUER_COMMIT]}
    assert result["required_commits"] == sorted(
        {
            CANDIDATE_COMMIT,
            TRANSITION_COMMIT,
            GRAPH_COMMIT,
            ISSUER_COMMIT,
        }
    )
    assert AUTHORITY_COMMIT not in result["required_commits"]
    assert OMITTED_PARENT_COMMIT not in result["required_commits"]
    assert {GRAPH_COMMIT, ISSUER_COMMIT}.issubset(set(verified))
    assert OMITTED_PARENT_COMMIT not in verified
    assert validate_active_release_requirements_v2(result) == result


def test_build_accepts_legacy_graph_and_selects_every_disclosed_boot(
    active_graph_pair,
) -> None:
    legacy = active_graph_pair[2]
    legacy_root = legacy["root_receipt_hash"]
    verified = []
    result = build_active_release_requirements_v2(
        candidate_commit_sha=CANDIDATE_COMMIT,
        authority_commit_sha=AUTHORITY_COMMIT,
        restart_invocation_id=RESTART_INVOCATION_ID,
        transition_commit_shas=(),
        active_graphs={legacy_root: legacy},
        expected_lineage_id=LINEAGE_ID,
        boot_verifier=lambda identity: verified.append(identity["commit_sha"])
        or identity,
    )

    assert result["commits_by_root"] == {legacy_root: [OMITTED_PARENT_COMMIT]}
    assert result["required_commits"] == sorted(
        {CANDIDATE_COMMIT, OMITTED_PARENT_COMMIT}
    )
    assert verified == [OMITTED_PARENT_COMMIT]


def test_build_rejects_root_mapping_tamper(active_graph_pair) -> None:

    graph = active_graph_pair[0]
    with pytest.raises(
        ActiveReleaseRequirementsV2Error,
        match="root differs from mapping key",
    ):
        build_active_release_requirements_v2(
            candidate_commit_sha=CANDIDATE_COMMIT,
            authority_commit_sha=AUTHORITY_COMMIT,
            restart_invocation_id=RESTART_INVOCATION_ID,
            transition_commit_shas=(),
            active_graphs={_hash("9"): graph},
            expected_lineage_id=LINEAGE_ID,
            boot_verifier=lambda identity: identity,
        )


@pytest.mark.parametrize("tamper", ["boot", "proof", "issuer", "tuple"])
def test_build_rejects_graph_and_proof_tamper(active_graph_pair, tamper) -> None:
    graph = deepcopy(active_graph_pair[0])
    if tamper == "boot":
        graph["boot_identities"][0]["commit_sha"] = "9" * 40
    elif tamper == "proof":
        graph["ancestry_proof"]["proof_hash"] = _hash("9")
    elif tamper == "issuer":
        del graph["ancestry_proof"]["certificate"]["issuer_boot_identity"]
    else:
        graph["boot_identities"] = tuple(graph["boot_identities"])

    with pytest.raises(
        ActiveReleaseRequirementsV2Error,
        match="graph or ancestry proof is invalid",
    ):
        build_active_release_requirements_v2(
            candidate_commit_sha=CANDIDATE_COMMIT,
            authority_commit_sha=AUTHORITY_COMMIT,
            restart_invocation_id=RESTART_INVOCATION_ID,
            transition_commit_shas=(),
            active_graphs={graph["root_receipt_hash"]: graph},
            expected_lineage_id=LINEAGE_ID,
            boot_verifier=lambda identity: identity,
        )


def test_build_requires_release_nitro_verifier_to_accept_issuer(
    active_graph_pair,
) -> None:
    graph = active_graph_pair[1]

    def reject_issuer(identity):
        if identity["commit_sha"] == ISSUER_COMMIT:
            raise RuntimeError("issuer release is unavailable")
        return identity

    with pytest.raises(
        ActiveReleaseRequirementsV2Error,
        match="graph or ancestry proof is invalid",
    ):
        build_active_release_requirements_v2(
            candidate_commit_sha=CANDIDATE_COMMIT,
            authority_commit_sha=AUTHORITY_COMMIT,
            restart_invocation_id=RESTART_INVOCATION_ID,
            transition_commit_shas=(),
            active_graphs={graph["root_receipt_hash"]: graph},
            expected_lineage_id=LINEAGE_ID,
            boot_verifier=reject_issuer,
        )


def test_build_rejects_wrong_lineage_invalid_commit_and_more_than_512() -> None:
    with pytest.raises(ActiveReleaseRequirementsV2Error, match="lineage"):
        build_active_release_requirements_v2(
            candidate_commit_sha=CANDIDATE_COMMIT,
            authority_commit_sha=AUTHORITY_COMMIT,
            restart_invocation_id=RESTART_INVOCATION_ID,
            transition_commit_shas=(),
            active_graphs={},
            expected_lineage_id="SHA256:" + "a" * 64,
            boot_verifier=lambda identity: identity,
        )
    with pytest.raises(ActiveReleaseRequirementsV2Error, match="transition commit"):
        build_active_release_requirements_v2(
            candidate_commit_sha=CANDIDATE_COMMIT,
            authority_commit_sha=AUTHORITY_COMMIT,
            restart_invocation_id=RESTART_INVOCATION_ID,
            transition_commit_shas=("A" * 40,),
            active_graphs={},
            expected_lineage_id=LINEAGE_ID,
            boot_verifier=lambda identity: identity,
        )

    transitions = tuple("%040x" % index for index in range(512))
    with pytest.raises(ActiveReleaseRequirementsV2Error, match="exceed bound"):
        build_active_release_requirements_v2(
            candidate_commit_sha=CANDIDATE_COMMIT,
            authority_commit_sha=AUTHORITY_COMMIT,
            restart_invocation_id=RESTART_INVOCATION_ID,
            transition_commit_shas=transitions,
            active_graphs={},
            expected_lineage_id=LINEAGE_ID,
            boot_verifier=lambda identity: identity,
        )


def test_sidecar_validator_rejects_hash_and_required_set_tamper(
    active_graph_pair,
) -> None:
    graph = active_graph_pair[0]
    result = build_active_release_requirements_v2(
        candidate_commit_sha=CANDIDATE_COMMIT,
        authority_commit_sha=AUTHORITY_COMMIT,
        restart_invocation_id=RESTART_INVOCATION_ID,
        transition_commit_shas=(TRANSITION_COMMIT,),
        active_graphs={graph["root_receipt_hash"]: graph},
        expected_lineage_id=LINEAGE_ID,
        boot_verifier=lambda identity: identity,
    )

    bad_hash = {**result, "selection_hash": _hash("9")}
    with pytest.raises(ActiveReleaseRequirementsV2Error, match="selection hash"):
        validate_active_release_requirements_v2(bad_hash)

    bad_required = deepcopy(result)
    bad_required["required_commits"].remove(GRAPH_COMMIT)
    with pytest.raises(ActiveReleaseRequirementsV2Error, match="required commits"):
        validate_active_release_requirements_v2(bad_required)


def test_build_rejects_active_root_count_before_graph_validation(
    monkeypatch,
) -> None:
    verifier_calls = []
    monkeypatch.setattr(requirements_module, "MAX_ACTIVE_RELEASE_ROOTS", 1)
    with pytest.raises(
        ActiveReleaseRequirementsV2Error,
        match="active release root count exceeds bound",
    ):
        build_active_release_requirements_v2(
            candidate_commit_sha=CANDIDATE_COMMIT,
            authority_commit_sha=AUTHORITY_COMMIT,
            restart_invocation_id=RESTART_INVOCATION_ID,
            transition_commit_shas=(),
            active_graphs={_hash("1"): {}, _hash("2"): {}},
            expected_lineage_id=LINEAGE_ID,
            boot_verifier=lambda identity: verifier_calls.append(identity),
        )
    assert verifier_calls == []


def test_build_rejects_cumulative_graph_bytes_before_crypto_validation(
    active_graph_pair,
    monkeypatch,
) -> None:
    graph = active_graph_pair[0]
    verifier_calls = []
    monkeypatch.setattr(requirements_module, "MAX_ACTIVE_RELEASE_GRAPH_BYTES", 1)

    with pytest.raises(
        ActiveReleaseRequirementsV2Error,
        match="active receipt graph bytes exceed bound",
    ):
        build_active_release_requirements_v2(
            candidate_commit_sha=CANDIDATE_COMMIT,
            authority_commit_sha=AUTHORITY_COMMIT,
            restart_invocation_id=RESTART_INVOCATION_ID,
            transition_commit_shas=(),
            active_graphs={graph["root_receipt_hash"]: graph},
            expected_lineage_id=LINEAGE_ID,
            boot_verifier=lambda identity: verifier_calls.append(identity),
        )

    assert verifier_calls == []
