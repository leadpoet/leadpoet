from __future__ import annotations

import asyncio
import copy
import importlib
import json
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import pytest

from leadpoet_canonical.attested_v2 import (
    EMPTY_ARTIFACT_ROOT,
    EMPTY_HOST_OPERATION_ROOT,
    EMPTY_TRANSPORT_ROOT,
    WEIGHT_ROLE,
    build_checkpointed_receipt_graph,
    build_execution_receipt_body,
    create_signed_execution_receipt,
    sha256_json,
)
from leadpoet_canonical.ancestry_checkpoint_v2 import (
    ANCESTRY_DELTA_SCHEMA_VERSION,
    build_compact_ancestry_proof_from_delta_v2,
    issue_ancestry_certificate_v2,
)
from leadpoet_canonical.binding import create_binding_message
from leadpoet_canonical.compact_weight_authority_v2 import (
    COMPACT_WEIGHT_SUBMISSION_SCHEMA_VERSION,
    VALIDATOR_WEIGHT_RECEIPT_DELTA_SCHEMA_VERSION,
    validate_compact_weight_submission_shape_v2,
)
from leadpoet_canonical.hotkey_authority_v2 import (
    build_application_signature_request_v2,
    build_weight_extrinsic_authorization_v2,
    encode_signed_extrinsic_v2,
    signed_extrinsic_hash_v2,
)
from leadpoet_canonical.weight_authority_v2 import (
    GATEWAY_WEIGHT_INPUT_CATEGORIES,
    validate_published_weight_bundle_v2,
)
from tests.test_validator_hotkey_authority_v2 import _profile
from tests.test_validator_weight_authority_v2 import (
    GATEWAY_COMMIT,
    NOW as VALIDATOR_NOW,
    VALIDATOR_HOTKEY,
    _boot as _validator_boot,
    _fixture as _validator_fixture,
    _keypair as _validator_keypair,
)
from tests.test_weight_authority_v2 import _bundle
from validator_tee.host import publication_journal_v2 as journal_module
from validator_tee.host.publication_journal_v2 import (
    AuthoritativeWeightPublicationJournalV2,
    COMPACT_JOURNAL_SCHEMA_VERSION,
    EPOCH_EVIDENCE_JOURNAL_SCHEMA_VERSION,
    JOURNAL_SCHEMA_VERSION,
    LEGACY_JOURNAL_SCHEMA_VERSION,
    WeightPublicationJournalV2Error,
    publication_journal_release_requirements_v2,
)
from validator_tee.host.weight_authority_v2 import (
    build_authoritative_weight_bundle_v2,
)


EVENT = "sha256:" + "e" * 64
AUTHORIZATION = "sha256:" + "a" * 64
LINEAGE_ID = "sha256:" + "9" * 64


def _pending_cross_release_authority():
    fixture = _validator_fixture(historical_validator_ancestry=True)
    enclave_response = fixture["authority"].compute(fixture["request"])
    enclave_response["weight_authorization_id"] = AUTHORIZATION
    boot = fixture["validator_boot"]
    binding_message = create_binding_message(
        netuid=71,
        chain="wss://entrypoint-finney.opentensor.ai:443",
        enclave_pubkey=boot["signing_pubkey"],
        validator_code_hash=boot["build_manifest_hash"],
        version=boot["commit_sha"],
    )
    request = build_application_signature_request_v2(
        message=binding_message.encode("utf-8"),
        validator_hotkey=VALIDATOR_HOTKEY,
        boot_identity_hash=boot["boot_identity_hash"],
    )
    output = {
        "schema_version": "leadpoet.application_signature_result.v2",
        "request_hash": request["request_hash"],
        "purpose": "validator.gateway_binding.v2",
        "validator_hotkey": VALIDATOR_HOTKEY,
        "signature": "f" * 128,
    }
    graph = enclave_response["receipt_graph"]
    receipt = create_signed_execution_receipt(
        body=build_execution_receipt_body(
            role=WEIGHT_ROLE,
            purpose="validator.hotkey_signature.v2",
            job_id="application-signature:%s"
            % request["request_hash"].split(":", 1)[1][:32],
            epoch_id=100,
            sequence=0,
            commit_sha=boot["commit_sha"],
            pcr0=boot["pcr0"],
            build_manifest_hash=boot["build_manifest_hash"],
            dependency_lock_hash=boot["dependency_lock_hash"],
            config_hash=boot["config_hash"],
            boot_identity_hash=boot["boot_identity_hash"],
            input_root=request["request_hash"],
            output_root=sha256_json(output),
            transport_root_hash=EMPTY_TRANSPORT_ROOT,
            host_operation_root_hash=EMPTY_HOST_OPERATION_ROOT,
            artifact_root=EMPTY_ARTIFACT_ROOT,
            parent_receipt_hashes=(graph["root_receipt_hash"],),
            status="succeeded",
            failure_code=None,
            issued_at=VALIDATOR_NOW.strftime("%Y-%m-%dT%H:%M:%SZ"),
        ),
        enclave_pubkey=boot["signing_pubkey"],
        sign_digest=fixture["validator_key"].sign,
    )
    bundle = build_authoritative_weight_bundle_v2(
        enclave_response=enclave_response,
        validator_hotkey=VALIDATOR_HOTKEY,
        binding_message=binding_message,
        binding_signature_result={**output, "receipt": receipt},
    )
    return bundle, fixture


def _pending_cross_release_bundle():
    return _pending_cross_release_authority()[0]


def _prepared_journal(bundle):
    body = {
        "schema_version": JOURNAL_SCHEMA_VERSION,
        "state": "prepared",
        "revision": 0,
        "weight_authorization_id": AUTHORIZATION,
        "published_bundle": bundle,
        "epoch_evidence": None,
        "finalization_scan_generation": 0,
        "finalization_scan_id": None,
        "publication": None,
        "extrinsic_signature_results": [],
        "updated_at": "2026-08-23T00:00:00Z",
    }
    return {**body, "journal_hash": sha256_json(body)}


def _legacy_journal(bundle, schema_version):
    current = _prepared_journal(bundle)
    excluded = {
        "journal_hash",
        "finalization_scan_generation",
        "finalization_scan_id",
    }
    if schema_version == LEGACY_JOURNAL_SCHEMA_VERSION:
        excluded.add("epoch_evidence")
    body = {key: value for key, value in current.items() if key not in excluded}
    body["schema_version"] = schema_version
    return {**body, "journal_hash": sha256_json(body)}


def _compact_journal(compact):
    body = {
        "schema_version": COMPACT_JOURNAL_SCHEMA_VERSION,
        "state": "prepared",
        "revision": 0,
        "weight_authorization_id": AUTHORIZATION,
        "compact_submission": compact,
        "publication": None,
        "extrinsic_signature_results": [],
        "finalization_scan_generation": 0,
        "finalization_scan_id": None,
        "updated_at": "2026-08-23T00:00:00Z",
    }
    return {**body, "journal_hash": sha256_json(body)}


def _real_compact_submission(monkeypatch):
    source_root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(source_root / "tests" / "restart_rehearsal"))
    compact_fixture = importlib.import_module("compact_weight_joined_runner")
    monkeypatch.setenv("REHEARSAL_SOURCE_ROOT", str(source_root))
    monkeypatch.setenv("REHEARSAL_CANDIDATE_SHA", "f" * 40)
    asyncio.run(asyncio.sleep(0))

    captured = []
    real_record_prepared = AuthoritativeWeightPublicationJournalV2.record_prepared

    def capture_compact(self, prepared):
        if isinstance(prepared, dict) and isinstance(
            prepared.get("compact_submission"), dict
        ):
            captured.append(copy.deepcopy(prepared["compact_submission"]))
        return real_record_prepared(self, prepared)

    monkeypatch.setattr(
        AuthoritativeWeightPublicationJournalV2,
        "record_prepared",
        capture_compact,
    )
    evidence = compact_fixture.exercise_compact_weight_joined_path()
    assert evidence["production_primary_compact_lifecycle"] is True
    assert len(captured) == 1
    compact = captured[0]
    lineage_id = compact["validator_ancestry_proof"]["certificate"]["claim"][
        "lineage_id"
    ]
    return compact, lineage_id


def _publication(bundle):
    verified = validate_published_weight_bundle_v2(bundle)
    return {
        "success": True,
        "epoch_id": verified["epoch_id"],
        "weights_count": len(verified["uids"]),
        "weights_hash": verified["weights_hash"],
        "weight_receipt_hash": verified["weight_receipt_hash"],
        "weight_submission_event_hash": EVENT,
        "message": "published",
    }


def _signature_result(bundle):
    verified = validate_published_weight_bundle_v2(bundle)
    result = bundle["weight_result"]
    authorization = build_weight_extrinsic_authorization_v2(
        profile=_profile(),
        validator_hotkey=bundle["validator_hotkey"],
        hotkey_public_key_hex="11" * 32,
        epoch_id=verified["epoch_id"],
        netuid=verified["netuid"],
        subnet_epoch_index=23807,
        weight_receipt_hash=verified["weight_receipt_hash"],
        weight_submission_event_hash=EVENT,
        weights_hash=verified["weights_hash"],
        sparse_uids=result["sparse_uids"],
        sparse_weights_u16=result["sparse_weights_u16"],
        commitment=b"measured-commitment",
        reveal_round=1234,
        era_current=36099,
        nonce=7,
        block_hash="22" * 32,
    )
    signature = "33" * 64
    extrinsic = encode_signed_extrinsic_v2(
        hotkey_public_key_hex="11" * 32,
        signature_hex=signature,
        era_period=authorization["era_period"],
        era_current=authorization["era_current"],
        nonce=authorization["nonce"],
        call_data_hex=authorization["call_data_hex"],
    )
    return {
        "schema_version": "leadpoet.weight_extrinsic_signature.v2",
        "authorization_hash": authorization["authorization_hash"],
        "validator_hotkey": bundle["validator_hotkey"],
        "signature": signature,
        "extrinsic_hash": signed_extrinsic_hash_v2(extrinsic),
        "authorization": authorization,
        "receipt": {"receipt_hash": "sha256:" + "4" * 64},
    }


def test_release_requirements_are_empty_without_a_pending_journal():
    assert publication_journal_release_requirements_v2(None) == {
        "journal_hash": None,
        "required_commits": [],
    }


@pytest.mark.parametrize(
    "schema_version",
    [
        LEGACY_JOURNAL_SCHEMA_VERSION,
        EPOCH_EVIDENCE_JOURNAL_SCHEMA_VERSION,
        JOURNAL_SCHEMA_VERSION,
    ],
)
def test_release_requirements_accept_legacy_v2_through_v4_journals(
    schema_version,
):
    bundle = _bundle()
    journal = (
        _prepared_journal(bundle)
        if schema_version == JOURNAL_SCHEMA_VERSION
        else _legacy_journal(bundle, schema_version)
    )

    requirements = publication_journal_release_requirements_v2(
        journal,
        expected_lineage_id=LINEAGE_ID,
        expected_validator_hotkey=bundle["validator_hotkey"],
        chain_profile=_profile(),
    )

    assert requirements == {
        "journal_hash": journal["journal_hash"],
        "required_commits": ["3" * 40],
    }


def test_release_requirements_retain_cross_release_validator_boot_from_journal():
    journal = _prepared_journal(_pending_cross_release_bundle())
    observed = set()

    requirements = publication_journal_release_requirements_v2(
        journal,
        expected_lineage_id=LINEAGE_ID,
        expected_validator_hotkey=VALIDATOR_HOTKEY,
        boot_verifier=lambda identity: (
            observed.add(identity["commit_sha"]) or dict(identity)
        ),
        chain_profile=_profile(),
    )

    assert requirements == {
        "journal_hash": journal["journal_hash"],
        "required_commits": ["a" * 40, GATEWAY_COMMIT, "f" * 40],
    }
    assert observed == set(requirements["required_commits"])


def test_release_requirements_cover_checkpointed_graph_issuer():
    bundle, fixture = _pending_cross_release_authority()
    issuer_key, issuer_public = _validator_keypair()
    issuer = _validator_boot(
        role=WEIGHT_ROLE,
        physical_role="validator_weights",
        commit="c" * 40,
        pcr0="d" * 96,
        manifest="sha256:" + "e" * 64,
        dependency_lock="sha256:" + "1" * 64,
        config_hash="sha256:" + "2" * 64,
        private_key=issuer_key,
        public_key=issuer_public,
        nonce="9",
    )
    full_graph = bundle["receipt_graph"]
    delta = {
        "schema_version": ANCESTRY_DELTA_SCHEMA_VERSION,
        "root_receipt_hash": full_graph["root_receipt_hash"],
        "boot_identities": full_graph["boot_identities"],
        "receipts": full_graph["receipts"],
        "transport_attempts": full_graph["transport_attempts"],
        "host_operations": full_graph["host_operations"],
    }
    lineage_id = "sha256:" + "3" * 64
    structural_verifier = lambda identity: dict(identity)
    certificate = issue_ancestry_certificate_v2(
        local_delta=delta,
        lineage_id=lineage_id,
        certificate_sequence=0,
        issuer_boot_identity=issuer,
        issued_at="2026-08-23T00:00:00Z",
        sign_digest=issuer_key.sign,
        boot_attestation_verifier=structural_verifier,
        allowed_issuer_roles={WEIGHT_ROLE},
    )
    proof = build_compact_ancestry_proof_from_delta_v2(
        delta,
        certificate,
        expected_lineage_id=lineage_id,
        boot_attestation_verifier=structural_verifier,
        allowed_issuer_roles={WEIGHT_ROLE},
    )
    graph = build_checkpointed_receipt_graph(
        root_receipt_hash=delta["root_receipt_hash"],
        boot_identities=delta["boot_identities"],
        receipts=delta["receipts"],
        transport_attempts=delta["transport_attempts"],
        host_operations=delta["host_operations"],
        ancestry_lineage_id=lineage_id,
        ancestry_proof=proof,
        boot_attestation_verifier=structural_verifier,
        require_boot_attestation_verification=True,
    )
    journal = _prepared_journal({**bundle, "receipt_graph": graph})

    requirements = publication_journal_release_requirements_v2(
        journal,
        expected_lineage_id=lineage_id,
        expected_validator_hotkey=bundle["validator_hotkey"],
        chain_profile=_profile(),
    )

    assert requirements["required_commits"] == sorted(
        {
            fixture["validator_boot"]["commit_sha"],
            fixture["gateway_boot"]["commit_sha"],
            fixture["historical_boot"]["commit_sha"],
            issuer["commit_sha"],
        }
    )


def test_release_requirements_reject_a_boot_disallowed_by_release_verifier():
    journal = _prepared_journal(_pending_cross_release_bundle())

    def reject_historical(identity):
        if identity["commit_sha"] == "f" * 40:
            raise ValueError("release is not approved")
        return dict(identity)

    with pytest.raises(
        WeightPublicationJournalV2Error,
        match="invalid or unapproved",
    ):
        publication_journal_release_requirements_v2(
            journal,
            expected_lineage_id=LINEAGE_ID,
            expected_validator_hotkey=VALIDATOR_HOTKEY,
            boot_verifier=reject_historical,
            chain_profile=_profile(),
        )


def test_v5_release_requirements_include_checkpoint_issuer(monkeypatch):
    fixture = _validator_fixture()
    issuer_key, issuer_public = _validator_keypair()
    del issuer_key
    checkpoint_issuer = _validator_boot(
        role=WEIGHT_ROLE,
        physical_role="validator_weights",
        commit="c" * 40,
        pcr0="d" * 96,
        manifest="sha256:" + "e" * 64,
        dependency_lock="sha256:" + "1" * 64,
        config_hash="sha256:" + "2" * 64,
        private_key=None,
        public_key=issuer_public,
        nonce="9",
    )
    lineage_id = "sha256:" + "3" * 64

    def proof(issuer, disclosed):
        return {
            "certificate": {
                "claim": {"lineage_id": lineage_id},
                "issuer_boot_identity": issuer,
            },
            "disclosed_boot_identities": [disclosed],
        }

    upstream_proof = proof(fixture["gateway_boot"], fixture["gateway_boot"])
    validator_proof = proof(checkpoint_issuer, fixture["validator_boot"])
    delta_root = "sha256:" + "4" * 64
    compact_body = {
        "schema_version": COMPACT_WEIGHT_SUBMISSION_SCHEMA_VERSION,
        "validator_hotkey": VALIDATOR_HOTKEY,
        "binding_message": "binding",
        "validator_hotkey_signature": "5" * 128,
        "weight_snapshot": {},
        "weight_result": {},
        "weights_signature": "6" * 128,
        "ancestry_commitment": "sha256:" + "7" * 64,
        "upstream_ancestry_proofs": {
            category: copy.deepcopy(upstream_proof)
            for category in GATEWAY_WEIGHT_INPUT_CATEGORIES
        },
        "upstream_transport_attempts": [],
        "validator_receipt_delta": {
            "schema_version": VALIDATOR_WEIGHT_RECEIPT_DELTA_SCHEMA_VERSION,
            "root_receipt_hash": delta_root,
            "boot_identities": [fixture["validator_boot"]],
            "receipts": [],
            "transport_attempts": [],
            "host_operations": [],
        },
        "binding_receipt": {"parent_receipt_hashes": [delta_root]},
        "validator_ancestry_proof": validator_proof,
        "epoch_authority": None,
        "epoch_boundary": None,
    }
    compact = {
        **compact_body,
        "compact_submission_hash": sha256_json(compact_body),
    }
    validate_compact_weight_submission_shape_v2(compact)

    def verify_compact(
        value,
        *,
        expected_lineage_id,
        expected_chain,
        identity_cache,
        boot_verifier,
    ):
        assert expected_lineage_id == lineage_id
        assert expected_chain == _profile()["chain_endpoint"]
        assert identity_cache is None
        assert callable(boot_verifier)
        validate_compact_weight_submission_shape_v2(value)
        return {"validator_hotkey": VALIDATOR_HOTKEY}

    monkeypatch.setattr(
        journal_module,
        "verify_compact_weight_submission_v2",
        verify_compact,
    )
    journal = _compact_journal(compact)

    with pytest.raises(
        WeightPublicationJournalV2Error,
        match="expected validator hotkey is unavailable",
    ):
        publication_journal_release_requirements_v2(
            journal,
            expected_lineage_id=lineage_id,
            chain_profile=_profile(),
        )
    with pytest.raises(
        WeightPublicationJournalV2Error,
        match="expected lineage is unavailable or invalid",
    ):
        publication_journal_release_requirements_v2(
            journal,
            expected_lineage_id=lineage_id.upper(),
            expected_validator_hotkey=VALIDATOR_HOTKEY,
            chain_profile=_profile(),
        )
    with pytest.raises(
        WeightPublicationJournalV2Error,
        match="expected chain signing profile is unavailable",
    ):
        publication_journal_release_requirements_v2(
            journal,
            expected_lineage_id=lineage_id,
            expected_validator_hotkey=VALIDATOR_HOTKEY,
        )

    requirements = publication_journal_release_requirements_v2(
        journal,
        expected_lineage_id=lineage_id,
        expected_validator_hotkey=VALIDATOR_HOTKEY,
        chain_profile=_profile(),
    )

    assert checkpoint_issuer["commit_sha"] not in {
        fixture["validator_boot"]["commit_sha"],
        fixture["gateway_boot"]["commit_sha"],
    }
    assert requirements == {
        "journal_hash": journal["journal_hash"],
        "required_commits": sorted(
            {
                fixture["validator_boot"]["commit_sha"],
                fixture["gateway_boot"]["commit_sha"],
                checkpoint_issuer["commit_sha"],
            }
        ),
    }


def test_v5_real_canonical_verifier_rejects_semantic_tamper(monkeypatch):
    compact, lineage_id = _real_compact_submission(monkeypatch)
    profile = _profile()

    requirements = publication_journal_release_requirements_v2(
        _compact_journal(compact),
        expected_lineage_id=lineage_id,
        expected_validator_hotkey=compact["validator_hotkey"],
        boot_verifier=lambda identity: dict(identity),
        chain_profile=profile,
    )
    assert requirements["required_commits"] == ["f" * 40]

    def rehash(value):
        value["compact_submission_hash"] = sha256_json(
            {
                key: item
                for key, item in value.items()
                if key != "compact_submission_hash"
            }
        )
        return value

    mutations = {}
    mutations["weight_result"] = copy.deepcopy(compact)
    mutations["weight_result"]["weight_result"]["sparse_weights_u16"][0] ^= 1
    mutations["input_receipt"] = copy.deepcopy(compact)
    input_category = sorted(
        mutations["input_receipt"]["weight_snapshot"]["input_receipt_hashes"]
    )[0]
    mutations["input_receipt"]["weight_snapshot"]["input_receipt_hashes"][
        input_category
    ] = "sha256:" + "0" * 64
    mutations["weights_signature"] = copy.deepcopy(compact)
    mutations["weights_signature"]["weights_signature"] = "0" * 128
    mutations["hotkey_signature"] = copy.deepcopy(compact)
    mutations["hotkey_signature"]["validator_hotkey_signature"] = "0" * 128
    mutations["validator_hotkey"] = copy.deepcopy(compact)
    mutations["validator_hotkey"]["validator_hotkey"] = "5" * 48

    for label, mutated in mutations.items():
        with pytest.raises(
            WeightPublicationJournalV2Error,
            match="release requirements are invalid",
        ):
            publication_journal_release_requirements_v2(
                _compact_journal(rehash(mutated)),
                expected_lineage_id=lineage_id,
                expected_validator_hotkey=compact["validator_hotkey"],
                boot_verifier=lambda identity: dict(identity),
                chain_profile=profile,
            )

    with pytest.raises(
        WeightPublicationJournalV2Error,
        match="release requirements are invalid",
    ):
        publication_journal_release_requirements_v2(
            _compact_journal(compact),
            expected_lineage_id="sha256:" + "0" * 64,
            expected_validator_hotkey=compact["validator_hotkey"],
            boot_verifier=lambda identity: dict(identity),
            chain_profile=profile,
        )


@pytest.mark.parametrize("mutation", ["journal_hash", "embedded_boot"])
def test_release_requirements_reject_tampered_or_malformed_journal(mutation):
    journal = _prepared_journal(_bundle())
    if mutation == "journal_hash":
        journal["journal_hash"] = "sha256:" + "0" * 64
    else:
        journal["published_bundle"]["receipt_graph"]["boot_identities"][0][
            "commit_sha"
        ] = ("0" * 40)
        body = {key: value for key, value in journal.items() if key != "journal_hash"}
        journal["journal_hash"] = sha256_json(body)

    with pytest.raises(WeightPublicationJournalV2Error):
        publication_journal_release_requirements_v2(
            journal,
            expected_lineage_id=LINEAGE_ID,
            expected_validator_hotkey=journal["published_bundle"][
                "validator_hotkey"
            ],
            chain_profile=_profile(),
        )


def test_journal_fsyncs_before_publication_and_survives_restart(tmp_path):
    bundle = _bundle()
    path = tmp_path / "weight-publication.json"
    journal = AuthoritativeWeightPublicationJournalV2(
        path, chain_profile=_profile()
    )
    prepared = journal.record_prepared(
        {
            "weight_authorization_id": AUTHORIZATION,
            "published_bundle": bundle,
        }
    )
    assert prepared["state"] == "prepared"
    assert prepared["schema_version"] == JOURNAL_SCHEMA_VERSION
    assert prepared["epoch_evidence"] is None
    assert prepared["finalization_scan_generation"] == 0
    assert prepared["finalization_scan_id"] is None
    assert prepared["publication"] is None
    assert path.exists()
    assert os.stat(path).st_mode & 0o777 == 0o600

    restarted = AuthoritativeWeightPublicationJournalV2(
        path, chain_profile=_profile()
    )
    published = restarted.record_published(_publication(bundle))
    assert published["state"] == "published"
    signed = restarted.record_signed(_signature_result(bundle))
    assert signed["state"] == "signed"
    assert len(signed["extrinsic_signature_results"]) == 1
    assert restarted.record_signed(_signature_result(bundle)) == signed
    first_scan = restarted.reserve_finalization_scan()
    first_record = restarted.load()
    second_scan = restarted.reserve_finalization_scan()
    second_record = restarted.load()
    assert first_scan != second_scan
    assert first_record["finalization_scan_generation"] == 1
    assert first_record["finalization_scan_id"] == first_scan
    assert second_record["finalization_scan_generation"] == 2
    assert second_record["finalization_scan_id"] == second_scan

    restarted.clear(expected_event_hash=EVENT)
    assert not path.exists()


def test_journal_rejects_corruption_and_conflicting_clear(tmp_path):
    bundle = _bundle()
    path = tmp_path / "weight-publication.json"
    journal = AuthoritativeWeightPublicationJournalV2(
        path, chain_profile=_profile()
    )
    journal.record_prepared(
        {
            "weight_authorization_id": AUTHORIZATION,
            "published_bundle": bundle,
        }
    )
    journal.record_published(_publication(bundle))
    with pytest.raises(WeightPublicationJournalV2Error, match="another"):
        journal.clear(expected_event_hash="sha256:" + "f" * 64)

    value = json.loads(path.read_text(encoding="utf-8"))
    value["weight_authorization_id"] = "sha256:" + "0" * 64
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(WeightPublicationJournalV2Error, match="hash"):
        journal.load()


def test_journal_will_not_replace_an_unfinished_epoch(tmp_path):
    journal = AuthoritativeWeightPublicationJournalV2(
        tmp_path / "weight-publication.json", chain_profile=_profile()
    )
    bundle = _bundle()
    journal.record_prepared(
        {
            "weight_authorization_id": AUTHORIZATION,
            "published_bundle": bundle,
        }
    )
    with pytest.raises(WeightPublicationJournalV2Error, match="unfinished"):
        journal.record_prepared(
            {
                "weight_authorization_id": "sha256:" + "b" * 64,
                "published_bundle": bundle,
            }
        )


def test_journal_reads_an_unfinished_legacy_v2_record(tmp_path):
    bundle = _bundle()
    path = tmp_path / "weight-publication.json"
    journal = AuthoritativeWeightPublicationJournalV2(
        path, chain_profile=_profile()
    )
    current = journal.record_prepared(
        {
            "weight_authorization_id": AUTHORIZATION,
            "published_bundle": bundle,
        }
    )
    legacy_body = {
        key: value
        for key, value in current.items()
        if key
        not in {
            "journal_hash",
            "epoch_evidence",
            "finalization_scan_generation",
            "finalization_scan_id",
        }
    }
    legacy_body["schema_version"] = LEGACY_JOURNAL_SCHEMA_VERSION
    legacy = {**legacy_body, "journal_hash": sha256_json(legacy_body)}
    path.write_text(json.dumps(legacy), encoding="utf-8")

    loaded = journal.load()
    assert loaded["schema_version"] == LEGACY_JOURNAL_SCHEMA_VERSION
    assert "epoch_evidence" not in loaded
    assert loaded["published_bundle"] == bundle


def test_journal_upgrades_v3_before_reserving_finalization_scan(tmp_path):
    bundle = _bundle()
    path = tmp_path / "weight-publication.json"
    journal = AuthoritativeWeightPublicationJournalV2(
        path, chain_profile=_profile()
    )
    journal.record_prepared(
        {
            "weight_authorization_id": AUTHORIZATION,
            "published_bundle": bundle,
        }
    )
    journal.record_published(_publication(bundle))
    current = journal.record_signed(_signature_result(bundle))
    v3_body = {
        key: value
        for key, value in current.items()
        if key
        not in {
            "journal_hash",
            "finalization_scan_generation",
            "finalization_scan_id",
        }
    }
    v3_body["schema_version"] = EPOCH_EVIDENCE_JOURNAL_SCHEMA_VERSION
    path.write_text(
        json.dumps({**v3_body, "journal_hash": sha256_json(v3_body)}),
        encoding="utf-8",
    )

    scan_id = journal.reserve_finalization_scan()
    upgraded = journal.load()

    assert upgraded["schema_version"] == JOURNAL_SCHEMA_VERSION
    assert upgraded["finalization_scan_generation"] == 1
    assert upgraded["finalization_scan_id"] == scan_id


def test_finalization_scan_reservations_are_unique_under_concurrency(tmp_path):
    bundle = _bundle()
    journal = AuthoritativeWeightPublicationJournalV2(
        tmp_path / "weight-publication.json", chain_profile=_profile()
    )
    journal.record_prepared(
        {
            "weight_authorization_id": AUTHORIZATION,
            "published_bundle": bundle,
        }
    )
    journal.record_published(_publication(bundle))
    journal.record_signed(_signature_result(bundle))

    with ThreadPoolExecutor(max_workers=10) as executor:
        scan_ids = list(
            executor.map(
                lambda _index: journal.reserve_finalization_scan(),
                range(20),
            )
        )

    current = journal.load()
    assert len(set(scan_ids)) == 20
    assert current["finalization_scan_generation"] == 20
    assert current["finalization_scan_id"] in scan_ids


def test_journal_quarantine_preserves_exact_validated_record(tmp_path):
    bundle = _bundle()
    path = tmp_path / "weight-publication.json"
    journal = AuthoritativeWeightPublicationJournalV2(
        path, chain_profile=_profile()
    )
    prepared = journal.record_prepared(
        {
            "weight_authorization_id": AUTHORIZATION,
            "published_bundle": bundle,
        }
    )
    original = path.read_bytes()

    quarantined = journal.quarantine(
        expected_epoch=100,
        reason="unsigned_epoch_closed",
    )

    assert not path.exists()
    assert quarantined.exists()
    assert quarantined.read_bytes() == original
    assert os.stat(quarantined).st_mode & 0o777 == 0o600
    assert journal.load() is None
    assert json.loads(quarantined.read_text(encoding="utf-8")) == prepared


def test_journal_quarantine_rejects_wrong_epoch(tmp_path):
    journal = AuthoritativeWeightPublicationJournalV2(
        tmp_path / "weight-publication.json", chain_profile=_profile()
    )
    journal.record_prepared(
        {
            "weight_authorization_id": AUTHORIZATION,
            "published_bundle": _bundle(),
        }
    )

    with pytest.raises(WeightPublicationJournalV2Error, match="another"):
        journal.quarantine(expected_epoch=101, reason="unsigned_epoch_closed")
