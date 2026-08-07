#!/usr/bin/env python3.11
"""Execute the real V2 canonical, signing, SDK, receipt, and auditor path.

Input generation is test-only.  Every security-sensitive output is produced or
validated by candidate production modules.  The irreversible chain broadcast
and production database are replaced by :mod:`local_services`.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import time
import traceback
from types import SimpleNamespace
from typing import Any, Callable, Mapping

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


SOURCE_ROOT = Path(os.environ.get("REHEARSAL_SOURCE_ROOT", "/source")).resolve()
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from leadpoet_canonical.attested_v2 import (  # noqa: E402
    EMPTY_HOST_OPERATION_ROOT,
    build_execution_receipt_body,
    build_receipt_graph,
    create_signed_execution_receipt,
    merkle_root,
    sha256_json,
)
from leadpoet_canonical.auditor_v2 import (  # noqa: E402
    verify_attested_weight_authority_v2,
    verify_attested_weight_bundle_v2,
)
from leadpoet_canonical.hotkey_authority_v2 import (  # noqa: E402
    build_weight_extrinsic_authorization_v2,
    chain_signing_profiles,
    encode_signed_extrinsic_v2,
    signed_extrinsic_hash_v2,
)
from leadpoet_canonical.weight_authority_v2 import (  # noqa: E402
    validate_published_weight_bundle_v2,
    validate_weight_finalization_submission_v2,
)
from local_services import (  # noqa: E402
    LocalBoundaryServices,
    LocalEnclaveSigningBoundary,
    LocalSDKSubstrateBoundary,
    local_enclave_backed_wallet,
)
from sanitized_weight_fixture import (  # noqa: E402
    EMPTY_ARTIFACT_ROOT,
    EMPTY_TRANSPORT_ROOT,
    SanitizedWeightFixture,
    VALIDATOR_HOTKEY,
)
from validator_tee.enclave.hotkey_authority_v2 import (  # noqa: E402
    _Sr25519Backend,
)
from validator_tee.host.weight_authority_v2 import (  # noqa: E402
    build_authoritative_weight_bundle_v2,
)
from gateway.tee.rehearsal_behavior_contract_v2 import (  # noqa: E402
    build_rehearsal_behavior_contract_v2,
    validate_rehearsal_behavior_contract_v2,
)
from validator_tee.host.enclave_hotkey_v2 import (  # noqa: E402
    AuthoritativeSetWeightsContextV2,
    _weight_extrinsic_module,
)


NOW = "2026-07-25T00:00:00Z"
GENESIS_HASH = (
    "0x2f0555cc76fc2840a25a6ea3b9637146806f1f44b090c175ffde2a7e5ab36c03"
)
PRIVATE_MODEL_BRANCH_POINTER_URI = (
    "s3://leadpoet-private-model-artifacts-493765492819/"
    "research-lab/sourcing-model/branches/leadpoet-lab/current.json"
)
PRIVATE_MODEL_SIGNING_KEY_ID = (
    "alias/leadpoet-research-lab-artifact-signing"
)
SIGNED_PRIVATE_MODEL_RELEASES = {
    "leadpoet-sourcing-wrapper-contract-v7": {
        "build_id": "30643934157-1",
        "compatibility_contract": {
            "contract_id": "leadpoet-sourcing-wrapper-contract-v7",
            "path": "sourcing_model/consumer_contract.json",
            "sha256": (
                "sha256:f2fea5a16de1dd1fafb1fa5259b161cd0dd8059fddaf30d8e9982d3eec391d10"
            ),
        },
        "component_registry_version": "sourcing-model-components:v2",
        "config_hash": (
            "sha256:9ae38ed373dbb9bafb4fc360d3003ad6cad9b101741932b7b9c6bc2c4bb28211"
        ),
        "consumer_parity_fixtures": {
            "path": "sourcing_model/consumer_parity_fixtures.json",
            "sha256": (
                "sha256:c39c48335a4877c091e6ca264f3f9411dbecd4992c09e9c77bdb789479076d3a"
            ),
        },
        "git_commit_sha": "4dfd54ed1a3142dbfcdad6a3b2988c5136e4f50e",
        "image_digest": (
            "493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/"
            "sourcing-model@sha256:4947b2972548cb8382636633acbaf9ecd22148da17f7ea664f237723433805f6"
        ),
        "manifest_hash": (
            "sha256:c978a61d661f6281620ebf7c7775c52bea92254593a08eca2199a62791439092"
        ),
        "manifest_uri": (
            "s3://leadpoet-private-model-artifacts-493765492819/"
            "research-lab/sourcing-model/"
            "4dfd54ed1a3142dbfcdad6a3b2988c5136e4f50e.json"
        ),
        "model_artifact_hash": (
            "sha256:54ccdacb8200c750426d815c0c7d8e379096be5514d9aa6868550a40d05d0533"
        ),
        "scoring_adapter_version": "qualification-company-scorer:v1",
        "signature_ref": (
            "s3://leadpoet-private-model-artifacts-493765492819/"
            "research-lab/sourcing-model/"
            "4dfd54ed1a3142dbfcdad6a3b2988c5136e4f50e.sig.b64"
        ),
    },
    "leadpoet-sourcing-wrapper-contract-v8": {
        "build_id": "30693188436-1",
        "compatibility_contract": {
            "contract_id": "leadpoet-sourcing-wrapper-contract-v8",
            "path": "sourcing_model/consumer_contract.json",
            "sha256": (
                "sha256:080e7b199c3e1d27ae080e497b541b560a2e12d383a709d453e7a2dd320b8dfc"
            ),
        },
        "component_registry_version": "sourcing-model-components:v2",
        "config_hash": (
            "sha256:9ae38ed373dbb9bafb4fc360d3003ad6cad9b101741932b7b9c6bc2c4bb28211"
        ),
        "consumer_parity_fixtures": {
            "path": "sourcing_model/consumer_parity_fixtures.json",
            "sha256": (
                "sha256:5527186b45294135639619d99bfcf076ec98035670f68843244ccd18fc3f80fe"
            ),
        },
        "git_commit_sha": "2d90daa8347daec34e8e7966eb6d208f47f52df2",
        "image_digest": (
            "493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/"
            "sourcing-model@sha256:dd5710d30d589b657f4bd593d4d015bbfe47374283a862bbb2aef57455c3de4a"
        ),
        "manifest_hash": (
            "sha256:3f92e56236f4c5f583ca0b3f8cf6c2b42bcf41a7a06c3ec584a8d6b8ceee6caa"
        ),
        "manifest_uri": (
            "s3://leadpoet-private-model-artifacts-493765492819/"
            "research-lab/sourcing-model/"
            "2d90daa8347daec34e8e7966eb6d208f47f52df2.json"
        ),
        "model_artifact_hash": (
            "sha256:879ace5e05383dcfebf877d60d80f7e179017a7c487741990e896c1d63caed28"
        ),
        "scoring_adapter_version": "qualification-company-scorer:v1",
        "signature_ref": (
            "s3://leadpoet-private-model-artifacts-493765492819/"
            "research-lab/sourcing-model/"
            "2d90daa8347daec34e8e7966eb6d208f47f52df2.sig.b64"
        ),
    },
    "leadpoet-sourcing-wrapper-contract-v11": {
        "build_id": "31069901085-1",
        "compatibility_contract": {
            "contract_id": "leadpoet-sourcing-wrapper-contract-v11",
            "path": "sourcing_model/consumer_contract.json",
            "sha256": (
                "sha256:2cd4d09b99db1f0ac523c3e57f361afb7c7ff1413392bd9aa5dfcee9efb81c01"
            ),
        },
        "component_registry_version": "sourcing-model-components:v2",
        "config_hash": (
            "sha256:9ae38ed373dbb9bafb4fc360d3003ad6cad9b101741932b7b9c6bc2c4bb28211"
        ),
        "consumer_parity_fixtures": {
            "path": "sourcing_model/consumer_parity_fixtures.json",
            "sha256": (
                "sha256:8b0d23b1664b5539e790c988afcb558c2aa4cf0ff925af0f7dbe2f9bc900fce4"
            ),
        },
        "facility_evidence_contract": {
            "contract_id": "facility-evidence:v1",
            "identity_policy": {
                "policy_version": "facility-identity-proof:v2",
                "sha256": (
                    "6a96d24b31da117a4f6ab2e3fe1ed87d8173e1aefc0f49cf02e72aa82a031169"
                ),
            },
            "path": "sourcing_model/facility_evidence_contract_v1.json",
            "sha256": (
                "sha256:7239d15f5db6b45c0c4f479b934d186bb1d569896e071f246162b27b6e0b9d99"
            ),
        },
        "git_commit_sha": "74a29a984938e1a443bdd0d2eed2f41f737be1e6",
        "image_digest": (
            "493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/"
            "sourcing-model@sha256:cff5ce5f9e95e749ea242cd8377d4479f017c68e78eac49fe7bb957997f09eff"
        ),
        "intent_release_benchmark": {
            "contract": {
                "contract_id": "intent-release-contracts:v1",
                "path": "sourcing_model/intent_release_contract_v1.json",
                "sha256": (
                    "sha256:6aad3eb1f43aaeec06cda69fcf4cb5ae4ab97f8de057cf7750967cbcb94bbf43"
                ),
            },
            "policy": {
                "path": "sourcing_model/intent_release_policy_v1.json",
                "payload_sha256": (
                    "sha256:53753b35b191bd5a6a61e27c634fcfdc97076bada8be7e41bb45ab14866f7075"
                ),
                "policy_id": "intent-release-policy:v1",
                "sha256": (
                    "sha256:81e0fc457d52652cc04fc12996dae32e96ef475534649c588d8a26f6aa3a1f80"
                ),
            },
        },
        "manifest_hash": (
            "sha256:84b7f21f843c46a551e346693b2079bfee63fbc62f3a0e8db00339bb57d932e8"
        ),
        "manifest_uri": (
            "s3://leadpoet-private-model-artifacts-493765492819/"
            "research-lab/sourcing-model/"
            "74a29a984938e1a443bdd0d2eed2f41f737be1e6.json"
        ),
        "model_artifact_hash": (
            "sha256:641adf30cfb197276da018702688ab3378f69ec2e7f71b2245963f537c35aab3"
        ),
        "scoring_adapter_version": "qualification-company-scorer:v1",
        "signature_ref": (
            "s3://leadpoet-private-model-artifacts-493765492819/"
            "research-lab/sourcing-model/"
            "74a29a984938e1a443bdd0d2eed2f41f737be1e6.sig.b64"
        ),
    },
}


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _run_workflow_stage(
    *,
    stage: str,
    action: Callable[[], Any],
    stages: list[dict[str, Any]],
) -> tuple[bool, Any]:
    """Fail one stage while allowing independent downstream probes to run."""

    try:
        value = action()
    except KeyboardInterrupt:
        raise
    except BaseException as exc:
        result = {
            "error": str(exc)[:2000],
            "error_type": type(exc).__name__,
            "stage": stage,
            "status": "failed",
            "traceback": traceback.format_exc(limit=20)[-12000:],
        }
        stages.append(result)
        print(
            "PRODUCTION_WORKFLOW_STAGE_FAILED_CONTINUING "
            f"stage={stage} error_type={result['error_type']} "
            f"error={result['error']!r}",
            file=sys.stderr,
            flush=True,
        )
        return False, None
    stages.append({"stage": stage, "status": "passed"})
    print(f"PRODUCTION_WORKFLOW_STAGE_PASSED stage={stage}", flush=True)
    return True, value


def _mark_workflow_stage_unexercised(
    *,
    stage: str,
    blocked_by: list[str],
    stages: list[dict[str, Any]],
) -> None:
    stages.append(
        {
            "blocked_by": list(blocked_by),
            "stage": stage,
            "status": "unexercised",
        }
    )
    print(
        "PRODUCTION_WORKFLOW_STAGE_UNEXERCISED "
        f"stage={stage} blocked_by={','.join(blocked_by)}",
        file=sys.stderr,
        flush=True,
    )


def _require_equal(left: Any, right: Any, message: str) -> Any:
    if left != right:
        raise RuntimeError(message)
    return left


def _exercise_signed_private_model_contract_transition() -> dict[str, Any]:
    """Exercise the signed oldest -> newest -> oldest production handoff."""

    local_boundaries = sys.modules.get("sitecustomize")
    if local_boundaries is None or not hasattr(
        local_boundaries,
        "_install_private_model_s3_object",
    ):
        from tests.restart_rehearsal import (
            sitecustomize as local_boundaries,
        )

    from gateway.research_lab import promotion as promotion_module
    from gateway.research_lab.model_authority_v2 import (
        AttestedPrivateModelRunnerV2,
    )
    from gateway.research_lab.promotion import (
        RepoHeadManifestNotReadyError,
        _load_repo_head_current_manifest,
    )
    from research_lab.eval import DockerPrivateModelSpec
    from research_lab.eval.private_runtime import (
        PrivateModelRuntimeError,
        build_local_private_artifact_manifest,
        load_private_artifact_manifest,
        verify_private_artifact_manifest_signature,
    )
    from research_lab.sourcing_model_contract_check import (
        resolve_reviewed_consumer_snapshot,
        reviewed_consumer_snapshots,
    )

    def split_s3_uri(uri: str) -> tuple[str, str]:
        if not str(uri).startswith("s3://"):
            raise RuntimeError("private model rehearsal URI is not S3")
        bucket, separator, key = str(uri)[5:].partition("/")
        if not separator or not bucket or not key:
            raise RuntimeError("private model rehearsal S3 URI is malformed")
        return bucket, key

    def install(uri: str, body: bytes) -> None:
        bucket, key = split_s3_uri(uri)
        local_boundaries._install_private_model_s3_object(
            bucket=bucket,
            key=key,
            body=body,
        )

    def sign_and_install(manifest: Mapping[str, Any]) -> None:
        signature = local_boundaries._sign_private_model_manifest_hash(
            str(manifest["manifest_hash"])
        )
        install(
            str(manifest["signature_ref"]),
            base64.b64encode(signature) + b"\n",
        )

    def rehash_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
        normalized = json.loads(json.dumps(dict(manifest)))
        normalized.pop("manifest_hash", None)
        return {
            **normalized,
            "manifest_hash": sha256_json(normalized),
        }

    def expect_runtime_rejection(
        action: Callable[[], Any],
        *,
        label: str,
    ) -> str:
        try:
            action()
        except PrivateModelRuntimeError as exc:
            return str(exc)
        raise RuntimeError(f"{label} did not fail closed")

    def source_root_for_snapshot(
        temporary_root: Path,
        *,
        contract_snapshot: Mapping[str, Any],
        parity_snapshot: Mapping[str, Any],
        name: str,
    ) -> Path:
        source_root = temporary_root / name
        contract = contract_snapshot["contract"]
        contract_path = source_root / str(contract["canonical_path"])
        parity_path = source_root / str(contract["parity_fixture_path"])
        contract_path.parent.mkdir(parents=True, exist_ok=True)
        parity_path.parent.mkdir(parents=True, exist_ok=True)
        contract_path.write_bytes(Path(contract_snapshot["contract_path"]).read_bytes())
        parity_path.write_bytes(Path(parity_snapshot["parity_path"]).read_bytes())
        return source_root

    expected_ids = set(SIGNED_PRIVATE_MODEL_RELEASES)
    snapshots = reviewed_consumer_snapshots()
    if not expected_ids.issubset(snapshots):
        raise RuntimeError("reviewed consumer snapshots are incomplete")
    for contract_id, release in SIGNED_PRIVATE_MODEL_RELEASES.items():
        snapshot = snapshots[contract_id]
        _require_equal(
            snapshot["contract_sha256"],
            release["compatibility_contract"]["sha256"],
            f"{contract_id} contract snapshot hash differs",
        )
        _require_equal(
            snapshot["parity_sha256"],
            release["consumer_parity_fixtures"]["sha256"],
            f"{contract_id} parity snapshot hash differs",
        )
        payload = dict(release)
        supplied_hash = str(payload.pop("manifest_hash"))
        _require_equal(
            sha256_json(payload),
            supplied_hash,
            f"{contract_id} signed release manifest hash differs",
        )

    local_boundaries._clear_private_model_s3_objects()
    built_contracts: dict[str, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory(
        prefix="leadpoet-sourcing-contract-transition-"
    ) as temporary:
        temporary_root = Path(temporary)
        for contract_id, release in SIGNED_PRIVATE_MODEL_RELEASES.items():
            source_root = source_root_for_snapshot(
                temporary_root,
                contract_snapshot=snapshots[contract_id],
                parity_snapshot=snapshots[contract_id],
                name=contract_id,
            )
            resolved = resolve_reviewed_consumer_snapshot(source_root)
            if resolved is None:
                raise RuntimeError(f"{contract_id} exact source pair did not resolve")
            signature_ref = (
                "s3://leadpoet-private-model-artifacts-493765492819/"
                "research-lab/sourcing-model/rehearsal/"
                f"{contract_id}.sig.b64"
            )
            built = build_local_private_artifact_manifest(
                source_path=source_root,
                git_commit_sha=str(release["git_commit_sha"]),
                image_digest=str(release["image_digest"]),
                manifest_uri=(
                    "s3://leadpoet-private-model-artifacts-493765492819/"
                    "research-lab/sourcing-model/rehearsal/"
                    f"{contract_id}.json"
                ),
                signature_ref=signature_ref,
                component_registry_version=str(release["component_registry_version"]),
                scoring_adapter_version=str(release["scoring_adapter_version"]),
                build_id=f"rehearsal-{contract_id}",
                config_payload={"fixture": "sanitized-production-shaped"},
                consumer_contract_id=contract_id,
            )
            _require_equal(
                built["compatibility_contract"],
                release["compatibility_contract"],
                f"{contract_id} builder contract dispatch differs",
            )
            _require_equal(
                built["consumer_parity_fixtures"],
                release["consumer_parity_fixtures"],
                f"{contract_id} builder parity dispatch differs",
            )
            sign_and_install(built)
            verification = verify_private_artifact_manifest_signature(
                built,
                key_id=PRIVATE_MODEL_SIGNING_KEY_ID,
            )
            if verification.get("verified") is not True:
                raise RuntimeError(f"{contract_id} builder signature did not verify")
            built_contracts[contract_id] = {
                "contract_sha256": built["compatibility_contract"]["sha256"],
                "manifest_hash": built["manifest_hash"],
                "parity_sha256": built["consumer_parity_fixtures"]["sha256"],
            }

        v7_id = "leadpoet-sourcing-wrapper-contract-v7"
        v8_id = "leadpoet-sourcing-wrapper-contract-v8"
        new_id = "leadpoet-sourcing-wrapper-contract-v11"
        release_ids = tuple(SIGNED_PRIVATE_MODEL_RELEASES)
        def expect_source_rejection(
            source_root: Path,
            *,
            asserted_contract_id: str,
            label: str,
        ) -> str:
            release = SIGNED_PRIVATE_MODEL_RELEASES[new_id]
            fixture_id = hashlib.sha256(label.encode("utf-8")).hexdigest()[:16]
            prefix = (
                "s3://leadpoet-private-model-artifacts-493765492819/"
                f"research-lab/sourcing-model/rehearsal/{fixture_id}"
            )
            return expect_runtime_rejection(
                lambda: build_local_private_artifact_manifest(
                    source_path=source_root,
                    git_commit_sha=str(release["git_commit_sha"]),
                    image_digest=str(release["image_digest"]),
                    manifest_uri=prefix + ".json",
                    signature_ref=prefix + ".sig.b64",
                    component_registry_version=(
                        "sourcing-model-components:v2"
                    ),
                    scoring_adapter_version=(
                        "qualification-company-scorer:v1"
                    ),
                    consumer_contract_id=asserted_contract_id,
                ),
                label=label,
            )

        hybrid_source_rejections: dict[str, str] = {}
        for contract_id in release_ids:
            for parity_id in release_ids:
                if contract_id == parity_id:
                    continue
                label = f"{contract_id}-with-{parity_id}"
                hybrid_root = source_root_for_snapshot(
                    temporary_root,
                    contract_snapshot=snapshots[contract_id],
                    parity_snapshot=snapshots[parity_id],
                    name=label,
                )
                hybrid_source_rejections[label] = expect_source_rejection(
                    hybrid_root,
                    asserted_contract_id=contract_id,
                    label=label,
                )

        exact_new_root = temporary_root / new_id
        assertion_mismatch_rejection = expect_source_rejection(
            exact_new_root,
            asserted_contract_id=v7_id,
            label="consumer contract assertion mismatch",
        )
        unknown_source_rejection = expect_source_rejection(
            exact_new_root,
            asserted_contract_id="leadpoet-sourcing-wrapper-contract-v12",
            label="unknown consumer contract assertion",
        )

        tampered_root = source_root_for_snapshot(
            temporary_root,
            contract_snapshot=snapshots[new_id],
            parity_snapshot=snapshots[new_id],
            name="tampered-new-contract",
        )
        tampered_contract_path = tampered_root / str(
            snapshots[new_id]["contract"]["canonical_path"]
        )
        tampered_contract_path.write_bytes(tampered_contract_path.read_bytes() + b"\n")
        tampered_source_rejection = expect_source_rejection(
            tampered_root,
            asserted_contract_id=new_id,
            label="tampered source contract",
        )

    for release in SIGNED_PRIVATE_MODEL_RELEASES.values():
        install(str(release["manifest_uri"]), _canonical(release) + b"\n")
        sign_and_install(release)

    config = SimpleNamespace(
        private_model_manifest_uri=PRIVATE_MODEL_BRANCH_POINTER_URI,
        private_repo_branch="leadpoet-lab",
        private_repo_url="https://example.invalid/Sourcing_model.git",
    )
    extended_release = rehash_manifest(
        {
            **SIGNED_PRIVATE_MODEL_RELEASES[new_id],
            "git_commit_sha": "f" * 40,
            "manifest_uri": (
                "s3://leadpoet-private-model-artifacts-493765492819/"
                "research-lab/sourcing-model/rehearsal/signed-extensions.json"
            ),
            "signature_ref": (
                "s3://leadpoet-private-model-artifacts-493765492819/"
                "research-lab/sourcing-model/rehearsal/signed-extensions.sig.b64"
            ),
            "intent_release_benchmark": {
                "contract": {
                    "contract_id": "intent-release-contracts:v1",
                    "sha256": "sha256:" + "5" * 64,
                },
                "policy": {
                    "policy_id": "intent-release-policy:v1",
                    "sha256": "sha256:" + "6" * 64,
                },
            },
            "facility_evidence_contract": {
                "contract_id": "facility-evidence:v1",
                "sha256": "sha256:" + "7" * 64,
            },
        }
    )
    install(
        str(extended_release["manifest_uri"]),
        _canonical(extended_release) + b"\n",
    )
    sign_and_install(extended_release)
    install(
        PRIVATE_MODEL_BRANCH_POINTER_URI,
        _canonical(extended_release) + b"\n",
    )
    extended_artifact, extended_status = asyncio.run(
        _load_repo_head_current_manifest(
            config,
            repo_main_sha=str(extended_release["git_commit_sha"]),
            wait_for_repo_head=False,
            wait_timeout_seconds=0,
            poll_seconds=1,
        )
    )
    extended_authority = AttestedPrivateModelRunnerV2(
        artifact=extended_artifact,
        spec=DockerPrivateModelSpec(
            image_digest=extended_artifact.image_digest,
            pull_before_run=False,
        ),
        model_kind="private",
        worker_index=0,
        epoch_id=30_000,
    )
    extended_version_doc = promotion_module._private_model_version_doc(
        artifact=extended_artifact,
        activation_source="rehearsal_signed_extensions",
    )
    signed_extensions_verified = bool(
        extended_status.get("status") == "current_json_matches_repo_head"
        and extended_artifact.to_dict() == extended_release
        and extended_authority.artifact.manifest_hash
        == extended_release["manifest_hash"]
        and extended_version_doc.get("private_model_manifest_hash")
        == extended_release["manifest_hash"]
    )
    if not signed_extensions_verified:
        raise RuntimeError(
            "signed manifest extensions did not survive model authority and persistence"
        )
    transitions: list[dict[str, Any]] = []

    def activate(
        contract_id: str,
        *,
        record_transition: bool = True,
    ) -> Any:
        release = SIGNED_PRIVATE_MODEL_RELEASES[contract_id]
        install(PRIVATE_MODEL_BRANCH_POINTER_URI, _canonical(release) + b"\n")
        loaded = load_private_artifact_manifest(PRIVATE_MODEL_BRANCH_POINTER_URI)
        verification = verify_private_artifact_manifest_signature(
            loaded,
            key_id=PRIVATE_MODEL_SIGNING_KEY_ID,
        )
        artifact, status = asyncio.run(
            _load_repo_head_current_manifest(
                config,
                repo_main_sha=str(release["git_commit_sha"]),
                wait_for_repo_head=False,
                wait_timeout_seconds=0,
                poll_seconds=1,
            )
        )
        if (
            verification.get("verified") is not True
            or artifact.manifest_hash != release["manifest_hash"]
            or status.get("status") != "current_json_matches_repo_head"
        ):
            raise RuntimeError(f"{contract_id} signed pointer did not align exactly")
        model_authority = AttestedPrivateModelRunnerV2(
            artifact=artifact,
            spec=DockerPrivateModelSpec(
                image_digest=artifact.image_digest,
                pull_before_run=False,
            ),
            model_kind="private",
            worker_index=0,
            epoch_id=30_000,
        )
        if type(model_authority) is not AttestedPrivateModelRunnerV2:
            raise RuntimeError(
                f"{contract_id} did not select the V2 scoring model authority"
            )
        if model_authority.artifact.manifest_hash != release["manifest_hash"]:
            raise RuntimeError(
                f"{contract_id} scoring model authority differs"
            )
        if record_transition:
            transitions.append(
                {
                    "contract_id": contract_id,
                    "git_commit_sha": release["git_commit_sha"],
                    "image_digest": release["image_digest"],
                    "manifest_hash": release["manifest_hash"],
                    "scoring_model_authority_verified": True,
                    "signature_verified": True,
                }
            )
        return artifact

    v7_id = "leadpoet-sourcing-wrapper-contract-v7"
    v8_id = "leadpoet-sourcing-wrapper-contract-v8"
    new_id = "leadpoet-sourcing-wrapper-contract-v11"
    release_ids = tuple(SIGNED_PRIVATE_MODEL_RELEASES)
    def active_row(contract_id: str) -> dict[str, Any]:
        release = SIGNED_PRIVATE_MODEL_RELEASES[contract_id]
        return {
            "private_model_version_id": f"rehearsal-{contract_id}",
            "model_artifact_hash": release["model_artifact_hash"],
            "private_model_manifest_hash": release["manifest_hash"],
            "private_model_manifest_uri": release["manifest_uri"],
            "git_commit_sha": release["git_commit_sha"],
            "current_version_status": "active",
            "current_status_at": NOW,
        }

    lineage_state: dict[str, Any] = {
        "active": active_row(v7_id),
        "repo_head": SIGNED_PRIVATE_MODEL_RELEASES[v7_id]["git_commit_sha"],
    }

    async def active_rows() -> list[dict[str, Any]]:
        return [dict(lineage_state["active"])]

    original_head_resolver = promotion_module._resolve_private_repo_head_sha
    original_active_rows = promotion_module._active_private_model_version_rows
    original_sync_enabled = promotion_module.repo_head_sync_enabled
    original_lineage_select_all = promotion_module.select_all
    candidate_owned_head_enabled = False

    async def lineage_select_all(
        table: str,
        **_kwargs: Any,
    ) -> list[dict[str, Any]]:
        if table == "research_lab_private_repo_commit_events":
            if not candidate_owned_head_enabled:
                return []
            return [
                {
                    "commit_event_id": "rehearsal-candidate-source-push",
                    "commit_status": "pushed",
                    "git_commit_sha": lineage_state["repo_head"],
                    "candidate_id": "rehearsal-candidate",
                    "score_bundle_id": "rehearsal-score-bundle",
                    "created_at": NOW,
                }
            ]
        if table == "research_lab_candidate_promotion_events":
            return []
        raise RuntimeError(
            f"unexpected durable lineage table in rehearsal: {table}"
        )

    promotion_module._resolve_private_repo_head_sha = (
        lambda **_kwargs: str(lineage_state["repo_head"])
    )
    promotion_module._active_private_model_version_rows = active_rows
    promotion_module.repo_head_sync_enabled = lambda: True
    promotion_module.select_all = lineage_select_all
    lineage_checks: dict[str, str] = {}
    try:
        activate(v7_id)
        v7_active = asyncio.run(
            promotion_module.sync_active_model_to_repo_head(
                config,
                dry_run=True,
                wait_for_repo_head=False,
            )
        )
        lineage_checks["v7_active"] = str(v7_active.get("status") or "")

        activate(new_id)
        lineage_state["repo_head"] = SIGNED_PRIVATE_MODEL_RELEASES[new_id][
            "git_commit_sha"
        ]
        new_plan = asyncio.run(
            promotion_module.sync_active_model_to_repo_head(
                config,
                dry_run=True,
                wait_for_repo_head=False,
            )
        )
        lineage_checks["new_rebenchmark_plan"] = str(
            new_plan.get("status") or ""
        )

        lineage_state["repo_head"] = SIGNED_PRIVATE_MODEL_RELEASES[v7_id][
            "git_commit_sha"
        ]
        mismatch = asyncio.run(
            promotion_module.sync_active_model_to_repo_head(
                config,
                dry_run=True,
                wait_for_repo_head=False,
            )
        )
        lineage_checks["pointer_source_mismatch"] = str(
            mismatch.get("status") or ""
        )
        if mismatch.get("benchmark_blocked_reason") != (
            "repo_head_manifest_not_ready"
        ):
            raise RuntimeError(
                "pointer/source mismatch did not block rebenchmark"
            )
        try:
            asyncio.run(
                _load_repo_head_current_manifest(
                    config,
                    repo_main_sha=str(lineage_state["repo_head"]),
                    wait_for_repo_head=False,
                    wait_timeout_seconds=0,
                    poll_seconds=1,
                )
            )
        except RepoHeadManifestNotReadyError:
            pointer_source_mismatch_rejected = True
        else:
            raise RuntimeError(
                "new pointer with old source did not fail closed"
            )

        lineage_state["repo_head"] = SIGNED_PRIVATE_MODEL_RELEASES[new_id][
            "git_commit_sha"
        ]
        lineage_state["active"] = active_row(new_id)
        new_active = asyncio.run(
            promotion_module.sync_active_model_to_repo_head(
                config,
                dry_run=True,
                wait_for_repo_head=False,
            )
        )
        lineage_checks["new_active"] = str(new_active.get("status") or "")

        activate(v7_id)
        lineage_state["repo_head"] = SIGNED_PRIVATE_MODEL_RELEASES[v7_id][
            "git_commit_sha"
        ]
        rollback_plan = asyncio.run(
            promotion_module.sync_active_model_to_repo_head(
                config,
                dry_run=True,
                wait_for_repo_head=False,
            )
        )
        lineage_checks["old_rollback_plan"] = str(
            rollback_plan.get("status") or ""
        )
        lineage_state["active"] = active_row(v7_id)
        rollback_active = asyncio.run(
            promotion_module.sync_active_model_to_repo_head(
                config,
                dry_run=True,
                wait_for_repo_head=False,
            )
        )
        lineage_checks["old_rollback_active"] = str(
            rollback_active.get("status") or ""
        )

        # A candidate push owns its exact source SHA until the normal
        # promotion path records active_version_created. Daily baseline sync
        # must not turn that same SHA into a generic repo-head release while
        # delayed current.json reconciliation is pending.
        activate(new_id, record_transition=False)
        lineage_state["repo_head"] = SIGNED_PRIVATE_MODEL_RELEASES[new_id][
            "git_commit_sha"
        ]
        lineage_state["active"] = active_row(v7_id)
        candidate_owned_head_enabled = True
        candidate_owned_plan = asyncio.run(
            promotion_module.sync_active_model_to_repo_head(
                config,
                dry_run=True,
                wait_for_repo_head=False,
            )
        )
        lineage_checks["candidate_owned_head_pending"] = str(
            candidate_owned_plan.get("status") or ""
        )
    finally:
        promotion_module._resolve_private_repo_head_sha = (
            original_head_resolver
        )
        promotion_module._active_private_model_version_rows = (
            original_active_rows
        )
        promotion_module.repo_head_sync_enabled = original_sync_enabled
        promotion_module.select_all = original_lineage_select_all

    expected_lineage_checks = {
        "v7_active": "active_is_repo_head",
        "new_rebenchmark_plan": "would_sync_active_model_to_repo_head",
        "pointer_source_mismatch": "repo_head_manifest_not_ready",
        "new_active": "active_is_repo_head",
        "old_rollback_plan": "would_sync_active_model_to_repo_head",
        "old_rollback_active": "active_is_repo_head",
        "candidate_owned_head_pending": "candidate_source_publication_pending",
    }
    _require_equal(
        lineage_checks,
        expected_lineage_checks,
        "active lineage or rebenchmark transition differs",
    )

    pending_event = {
        "promotion_event_id": "rehearsal-manifest-pending",
        "candidate_id": "rehearsal-candidate",
        "source_score_bundle_id": "rehearsal-score-bundle",
        "event_type": "promotion_checked",
        "promotion_status": "checked",
        "event_doc": {"reason": "source_pushed_manifest_pending"},
        "created_at": "2026-07-25T00:00:00+00:00",
    }
    pending_candidate = {
        "candidate_id": "rehearsal-candidate",
        "current_score_bundle_id": "rehearsal-score-bundle",
    }
    resolved_pending_event_count = 1_001
    resolved_pending_events = [
        {
            **pending_event,
            "promotion_event_id": f"rehearsal-manifest-already-recovered-{index}",
            "candidate_id": f"rehearsal-resolved-candidate-{index}",
            "source_score_bundle_id": f"rehearsal-resolved-score-bundle-{index}",
            "created_at": "2026-07-26T00:00:00+00:00",
        }
        for index in range(resolved_pending_event_count)
    ]
    pending_bundle = {
        "score_bundle_id": "rehearsal-score-bundle",
        "score_bundle_doc": {
            "candidate_artifact_hash": "sha256:" + "c" * 64,
            "parent_artifact_hash": "sha256:" + "a" * 64,
            "icp_set_hash": "sha256:" + "3" * 64,
            "evaluation_epoch": 30_000,
            "score_bundle_hash": "sha256:" + "5" * 64,
            "aggregates": {},
        },
    }
    original_select_all = promotion_module.select_all
    original_select_many = promotion_module.select_many
    original_select_one = promotion_module.select_one
    original_process_scored = (
        promotion_module.ResearchLabPromotionController.process_scored_candidate
    )

    async def pending_select_many(
        table: str,
        *,
        filters: Any,
        **_kwargs: Any,
    ) -> list[dict[str, Any]]:
        if table != "research_lab_candidate_promotion_events":
            return []
        normalized = dict(filters)
        event_type = str(normalized.get("event_type") or "")
        if event_type:
            return []
        candidate_id = str(normalized.get("candidate_id") or "")
        if candidate_id.startswith("rehearsal-resolved-candidate-"):
            raise RuntimeError(
                "terminal manifest recovery performed an N+1 candidate read"
            )
        if normalized.get("candidate_id") == "rehearsal-candidate":
            return [dict(pending_event)]
        return []

    async def pending_select_all(
        table: str,
        *,
        filters: Any,
        **_kwargs: Any,
    ) -> list[dict[str, Any]]:
        if table != "research_lab_candidate_promotion_events":
            return []
        normalized = dict(filters)
        event_type = str(normalized.get("event_type") or "")
        if event_type == "champion_reward_created":
            return [
                {
                    "candidate_id": event["candidate_id"],
                    "source_score_bundle_id": event["source_score_bundle_id"],
                    "event_type": "champion_reward_created",
                }
                for event in resolved_pending_events
            ]
        if event_type != "promotion_checked":
            return []
        if str(normalized.get("event_doc->>reason") or "") != (
            "source_pushed_manifest_pending"
        ):
            return []
        return [
            *(dict(event) for event in resolved_pending_events),
            dict(pending_event),
        ]

    async def pending_select_one(
        table: str,
        *,
        filters: Any,
        **_kwargs: Any,
    ) -> dict[str, Any] | None:
        normalized = dict(filters)
        if (
            table == "research_lab_candidate_evaluation_current"
            and normalized.get("candidate_id") == "rehearsal-candidate"
        ):
            return dict(pending_candidate)
        if (
            table == "research_evaluation_score_bundle_current"
            and normalized.get("score_bundle_id") == "rehearsal-score-bundle"
        ):
            return dict(pending_bundle)
        return None

    async def pending_process_scored(
        _self: Any,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        return {"status": "merged"}

    try:
        promotion_module.select_all = pending_select_all
        promotion_module.select_many = pending_select_many
        promotion_module.select_one = pending_select_one
        promotion_module.ResearchLabPromotionController.process_scored_candidate = (
            pending_process_scored
        )
        pending_reconcile = asyncio.run(
            promotion_module.reconcile_failed_private_source_pushes(
                config,
                worker_ref="rehearsal-worker",
                retry_after_seconds=0,
                dry_run=False,
            )
        )
    finally:
        promotion_module.select_all = original_select_all
        promotion_module.select_many = original_select_many
        promotion_module.select_one = original_select_one
        promotion_module.ResearchLabPromotionController.process_scored_candidate = (
            original_process_scored
        )
    manifest_pending_reconciled = bool(
        pending_reconcile.get("finalized") == 1
        and pending_reconcile.get("results")
        and len(pending_reconcile["results"])
        == resolved_pending_event_count + 1
        and all(
            item.get("status") == "already_rewarded"
            for item in pending_reconcile["results"][:-1]
        )
        and pending_reconcile["results"][-1].get("status") == "merged"
        and pending_reconcile["results"][-1].get("recovery_event_type")
        == "promotion_checked"
        and pending_reconcile["results"][-1].get("recovery_reason")
        == "source_pushed_manifest_pending"
        and pending_reconcile.get("attempted_recoveries") == 1
    )
    if not manifest_pending_reconciled:
        raise RuntimeError("delayed signed manifest was not reconciled")

    from gateway.research_lab import scoring_worker as scoring_worker_module

    scoring_order: list[str] = []
    original_maintenance_state = scoring_worker_module.get_scoring_maintenance_state
    original_worker_reconcile = (
        scoring_worker_module.reconcile_failed_private_source_pushes
    )
    original_worker_preflight = (
        scoring_worker_module.ResearchLabGatewayScoringWorker._run_lease_held_recovery_and_preflight
    )
    original_worker_baseline = (
        scoring_worker_module.ResearchLabGatewayScoringWorker._run_private_baseline_contained
    )
    original_baseline_owner = (
        scoring_worker_module.ResearchLabGatewayScoringWorker._is_private_baseline_owner
    )

    async def scoring_maintenance_state() -> dict[str, Any]:
        return {"paused": False, "reason": ""}

    async def scoring_preflight(_self: Any, _state: Any) -> dict[str, Any]:
        return {"proceed": True, "verdicts": []}

    async def scoring_reconcile(
        _config: Any,
        *,
        worker_ref: str,
        dry_run: bool,
    ) -> dict[str, Any]:
        if worker_ref != "rehearsal-ordering-worker" or dry_run is not False:
            raise RuntimeError("scoring reconciliation invocation differs")
        scoring_order.append("reconcile")
        return {"ok": True, "retried": 1, "finalized": 1, "results": []}

    async def scoring_baseline(_self: Any) -> dict[str, Any]:
        scoring_order.append("baseline")
        return {"status": "baseline_checkpoint_recycle"}

    try:
        scoring_worker_module.get_scoring_maintenance_state = (
            scoring_maintenance_state
        )
        scoring_worker_module.reconcile_failed_private_source_pushes = (
            scoring_reconcile
        )
        scoring_worker_module.ResearchLabGatewayScoringWorker._run_lease_held_recovery_and_preflight = (
            scoring_preflight
        )
        scoring_worker_module.ResearchLabGatewayScoringWorker._run_private_baseline_contained = (
            scoring_baseline
        )
        scoring_worker_module.ResearchLabGatewayScoringWorker._is_private_baseline_owner = (
            lambda _self: True
        )
        ordering_worker = object.__new__(
            scoring_worker_module.ResearchLabGatewayScoringWorker
        )
        ordering_worker.config = SimpleNamespace(
            scoring_worker_enabled=True,
            production_writes_enabled=True,
            evaluation_bundles_enabled=True,
            scoring_worker_require_proxy=False,
            scoring_worker_index=0,
            auto_promotion_enabled=True,
            private_baseline_rebenchmark_enabled=True,
        )
        ordering_worker.proxy_url = ""
        ordering_worker.worker_ref = "rehearsal-ordering-worker"
        ordering_worker._last_private_source_push_reconcile_at = float("-inf")
        ordering_result = asyncio.run(ordering_worker.run_once())
    finally:
        scoring_worker_module.get_scoring_maintenance_state = (
            original_maintenance_state
        )
        scoring_worker_module.reconcile_failed_private_source_pushes = (
            original_worker_reconcile
        )
        scoring_worker_module.ResearchLabGatewayScoringWorker._run_lease_held_recovery_and_preflight = (
            original_worker_preflight
        )
        scoring_worker_module.ResearchLabGatewayScoringWorker._run_private_baseline_contained = (
            original_worker_baseline
        )
        scoring_worker_module.ResearchLabGatewayScoringWorker._is_private_baseline_owner = (
            original_baseline_owner
        )
    manifest_reconcile_precedes_baseline = bool(
        scoring_order == ["reconcile", "baseline"]
        and ordering_result.get("status") == "baseline_checkpoint_recycle"
    )
    if not manifest_reconcile_precedes_baseline:
        raise RuntimeError("delayed manifest reconciliation did not precede baseline")

    hybrid_manifest_rejections: dict[str, str] = {}
    for contract_id in release_ids:
        for parity_id in release_ids:
            if contract_id == parity_id:
                continue
            label = f"{contract_id}-with-{parity_id}"
            hybrid_manifest = json.loads(
                json.dumps(SIGNED_PRIVATE_MODEL_RELEASES[contract_id])
            )
            hybrid_manifest["consumer_parity_fixtures"] = dict(
                SIGNED_PRIVATE_MODEL_RELEASES[parity_id][
                    "consumer_parity_fixtures"
                ]
            )
            hybrid_manifest["signature_ref"] = (
                "s3://leadpoet-private-model-artifacts-493765492819/"
                "research-lab/sourcing-model/rehearsal/"
                + hashlib.sha256(label.encode("utf-8")).hexdigest()[:16]
                + ".sig.b64"
            )
            hybrid_manifest = rehash_manifest(hybrid_manifest)
            sign_and_install(hybrid_manifest)
            hybrid_manifest_rejections[label] = expect_runtime_rejection(
                lambda manifest=hybrid_manifest: (
                    verify_private_artifact_manifest_signature(
                        manifest,
                        key_id=PRIVATE_MODEL_SIGNING_KEY_ID,
                    )
                ),
                label=label,
            )

    unknown_manifest = json.loads(
        json.dumps(SIGNED_PRIVATE_MODEL_RELEASES[new_id])
    )
    unknown_manifest["compatibility_contract"]["contract_id"] = (
        "leadpoet-sourcing-wrapper-contract-v9"
    )
    unknown_manifest["signature_ref"] = (
        "s3://leadpoet-private-model-artifacts-493765492819/"
        "research-lab/sourcing-model/rehearsal/unknown-manifest.sig.b64"
    )
    unknown_manifest = rehash_manifest(unknown_manifest)
    sign_and_install(unknown_manifest)
    unknown_manifest_rejection = expect_runtime_rejection(
        lambda: verify_private_artifact_manifest_signature(
            unknown_manifest,
            key_id=PRIVATE_MODEL_SIGNING_KEY_ID,
        ),
        label="unknown signed manifest contract",
    )

    tampered_manifest = json.loads(json.dumps(SIGNED_PRIVATE_MODEL_RELEASES[new_id]))
    tampered_manifest["compatibility_contract"]["sha256"] = "sha256:" + "0" * 64
    tampered_manifest["signature_ref"] = (
        "s3://leadpoet-private-model-artifacts-493765492819/"
        "research-lab/sourcing-model/rehearsal/tampered-manifest.sig.b64"
    )
    tampered_manifest = rehash_manifest(tampered_manifest)
    sign_and_install(tampered_manifest)
    tampered_manifest_rejection = expect_runtime_rejection(
        lambda: verify_private_artifact_manifest_signature(
            tampered_manifest,
            key_id=PRIVATE_MODEL_SIGNING_KEY_ID,
        ),
        label="tampered signed manifest",
    )

    invalid_signature_manifest = json.loads(
        json.dumps(SIGNED_PRIVATE_MODEL_RELEASES[new_id])
    )
    invalid_signature_manifest["signature_ref"] = (
        "s3://leadpoet-private-model-artifacts-493765492819/"
        "research-lab/sourcing-model/rehearsal/invalid-signature.sig.b64"
    )
    invalid_signature_manifest = rehash_manifest(invalid_signature_manifest)
    valid_signature = local_boundaries._sign_private_model_manifest_hash(
        str(invalid_signature_manifest["manifest_hash"])
    )
    corrupted_signature = valid_signature[:-1] + bytes((valid_signature[-1] ^ 1,))
    install(
        str(invalid_signature_manifest["signature_ref"]),
        base64.b64encode(corrupted_signature) + b"\n",
    )
    invalid_signature_rejection = expect_runtime_rejection(
        lambda: verify_private_artifact_manifest_signature(
            invalid_signature_manifest,
            key_id=PRIVATE_MODEL_SIGNING_KEY_ID,
        ),
        label="invalid manifest signature",
    )

    expected_provider_bindings = {
        "intent.source_add.bloomberry": (
            "9c0130dd089bc5645314f087c58b2c58b064b8e56d1e6bd4039e3e1502278d7b"
        ),
        "intent.source_add.builtwith": (
            "cbe713744e649d58db69f98ebf0f2eb2b82b1a5cb1281f7e7f514abd51a1e393"
        ),
        "intent.source_add.sumble": (
            "d2f5b630956ae18863ed968d1e4f1f73dade4caaf161ade1324daa5255fe8da6"
        ),
    }
    provider_binding_values: dict[str, set[str]] = {}

    def collect_provider_bindings(value: Any) -> None:
        if isinstance(value, Mapping):
            attestations = value.get("runtime_source_add_manifest_attestations")
            if isinstance(attestations, list):
                for item in attestations:
                    if isinstance(item, Mapping):
                        provider_binding_values.setdefault(
                            str(item.get("tool_id") or ""),
                            set(),
                        ).add(str(item.get("manifest_sha256") or ""))
            for item in value.values():
                collect_provider_bindings(item)
        elif isinstance(value, list):
            for item in value:
                collect_provider_bindings(item)

    current_parity = json.loads(
        Path(snapshots[new_id]["parity_path"]).read_text(encoding="utf-8")
    )
    collect_provider_bindings(current_parity)
    observed_provider_bindings = {
        tool_id: next(iter(values))
        for tool_id, values in provider_binding_values.items()
        if len(values) == 1
    }
    _require_equal(
        {tool_id: len(values) for tool_id, values in provider_binding_values.items()},
        {tool_id: 1 for tool_id in expected_provider_bindings},
        "current provider binding manifest projection is ambiguous",
    )
    _require_equal(
        observed_provider_bindings,
        expected_provider_bindings,
        "current provider binding manifest projection differs",
    )

    return {
        "built_contracts": built_contracts,
        "contract_assertion_mismatch_rejected": bool(assertion_mismatch_rejection),
        "hybrid_manifests_rejected": (
            len(hybrid_manifest_rejections)
            == len(release_ids) * (len(release_ids) - 1)
            and all(hybrid_manifest_rejections.values())
        ),
        "hybrid_sources_rejected": (
            len(hybrid_source_rejections)
            == len(release_ids) * (len(release_ids) - 1)
            and all(hybrid_source_rejections.values())
        ),
        "invalid_signature_rejected": bool(invalid_signature_rejection),
        "lineage_rebenchmark_checks": lineage_checks,
        "lineage_rebenchmark_verified": (
            lineage_checks == expected_lineage_checks
        ),
        "manifest_pending_reconciled": manifest_pending_reconciled,
        "manifest_reconcile_crosses_terminal_events": (
            manifest_pending_reconciled
        ),
        "manifest_reconcile_precedes_baseline": (
            manifest_reconcile_precedes_baseline
        ),
        "candidate_owned_repo_head_sync_blocked": (
            lineage_checks.get("candidate_owned_head_pending")
            == "candidate_source_publication_pending"
        ),
        "pointer_source_mismatch_rejected": (pointer_source_mismatch_rejected),
        "provider_bindings": observed_provider_bindings,
        "signed_extensions_verified": signed_extensions_verified,
        "rollback_exact": (
            [item["contract_id"] for item in transitions]
            == [v7_id, new_id, v7_id]
            and all(
                item.get("scoring_model_authority_verified") is True
                for item in transitions
            )
        ),
        "tampered_manifest_rejected": bool(tampered_manifest_rejection),
        "tampered_source_rejected": bool(tampered_source_rejection),
        "transitions": transitions,
        "unknown_manifest_rejected": bool(unknown_manifest_rejection),
        "unknown_source_rejected": bool(unknown_source_rejection),
    }



class _AuditorScaleValue:
    def __init__(self, value: Any):
        self.value = value


class _AuditorLocalSubstrate:
    """Exact-hash chain boundary consumed by the production auditor."""

    def __init__(self, *, epoch_id: int, block: int):
        self.epoch_id = int(epoch_id)
        self.block = int(block)
        self.last_epoch_block = self.epoch_id * 360
        self.last_update = 0
        self.weights: list[tuple[int, int]] = []

    @staticmethod
    def _hash(block: int) -> str:
        return "0x" + hashlib.sha256(
            f"leadpoet-auditor-local-block:{block}".encode("ascii")
        ).hexdigest()

    def get_block_hash(self, block: int) -> str:
        return GENESIS_HASH if int(block) == 0 else self._hash(int(block))

    def get_chain_finalised_head(self) -> str:
        return self._hash(self.block)

    def get_chain_head(self) -> str:
        return self._hash(self.block)

    def get_block_number(self, block_hash: str) -> int:
        if block_hash == GENESIS_HASH:
            return 0
        if block_hash != self._hash(self.block):
            raise RuntimeError("auditor local chain received an unknown hash")
        return self.block

    def query(
        self,
        *,
        module: str,
        storage_function: str,
        params: list[Any],
        block_hash: str,
    ) -> _AuditorScaleValue:
        if block_hash != self._hash(self.block):
            raise RuntimeError("auditor local query is not exact-hash pinned")
        if module == "Timestamp" and storage_function == "Now" and params == []:
            return _AuditorScaleValue(
                int(datetime(2026, 7, 25, tzinfo=timezone.utc).timestamp())
                * 1000
            )
        if module != "SubtensorModule":
            raise RuntimeError("auditor local query module differs")
        if params == [71]:
            scheduler = {
                "Tempo": 360,
                "LastEpochBlock": self.last_epoch_block,
                "PendingEpochAt": self.last_epoch_block + 360,
                "SubnetEpochIndex": self.epoch_id,
                "BlocksSinceLastStep": self.block - self.last_epoch_block,
                "RevealPeriodEpochs": 1,
                "LastUpdate": [self.last_update],
            }
            if storage_function not in scheduler:
                raise RuntimeError("auditor local scheduler field differs")
            return _AuditorScaleValue(scheduler[storage_function])
        if params == [71, 0] and storage_function == "Weights":
            return _AuditorScaleValue(list(self.weights))
        raise RuntimeError("auditor local query shape differs")


class _AuditorLocalSubtensor:
    def __init__(self, substrate: _AuditorLocalSubstrate):
        self.substrate = substrate

    def get_subnet_hyperparameters(
        self, netuid: int, block: int | None = None
    ) -> Any:
        if int(netuid) != 71 or block is not None:
            raise RuntimeError("auditor local hyperparameter request differs")
        return SimpleNamespace(tempo=360, commit_reveal_period=1)

    def set_weights(
        self,
        *,
        netuid: int,
        wallet: Any,
        uids: list[int],
        weights: list[float],
        wait_for_finalization: bool,
        mechid: int,
    ) -> tuple[bool, str]:
        del wallet
        if (
            int(netuid) != 71
            or wait_for_finalization is not True
            or int(mechid) != 0
            or len(uids) != len(weights)
        ):
            raise RuntimeError("auditor local set_weights contract differs")
        from leadpoet_canonical.weights import normalize_to_u16

        self.substrate.weights = list(
            zip(
                [int(uid) for uid in uids],
                normalize_to_u16(
                    [int(uid) for uid in uids],
                    [float(weight) for weight in weights],
                ),
            )
        )
        self.substrate.last_update = self.substrate.block
        return True, "local finalized chain boundary accepted"


def _run_production_auditor(
    *,
    authority: Mapping[str, Any],
    identity_cache: Mapping[str, Any],
    epoch_id: int,
    block: int,
) -> dict[str, Any]:
    """Run the real auditor verifier, exact-block gate, and submit loop."""

    import neurons.auditor_validator as auditor_module
    from Leadpoet.utils.subnet_epoch import SubnetEpochCutover

    substrate = _AuditorLocalSubstrate(epoch_id=epoch_id, block=block)
    auditor = auditor_module.AuditorValidator.__new__(
        auditor_module.AuditorValidator
    )
    auditor.config = SimpleNamespace(
        netuid=71,
        subtensor=SimpleNamespace(network="local"),
    )
    auditor.epoch_cutover = SubnetEpochCutover(
        network_genesis_hash=GENESIS_HASH,
        netuid=71,
        cutover_block=30_000 * 360,
        cutover_block_hash=_AuditorLocalSubstrate._hash(30_000 * 360),
        first_subnet_epoch_index=30_000,
        first_settlement_epoch_id=30_000,
        last_legacy_epoch_id=29_999,
    )
    auditor.epoch_archive_endpoint = "local://archive-boundary"
    auditor.epoch_archive_subtensor = _AuditorLocalSubtensor(substrate)
    auditor.subtensor = _AuditorLocalSubtensor(substrate)
    auditor.uid = 0
    auditor.wallet = SimpleNamespace(
        hotkey=SimpleNamespace(
            ss58_address="5FqLp5QmNRiHGyj3xbLVnDHfCx25qxJX5CUhpndF9GFfZZiK"
        )
    )
    auditor.last_submitted_epoch = None
    auditor.last_authority_epoch = None

    original = auditor_module.verify_attested_weight_authority_v2

    def verify_with_local_nitro(
        value: Mapping[str, Any],
        *,
        identity_cache: Mapping[str, Any],
        chain_signing_profile: Mapping[str, Any],
    ) -> dict[str, Any]:
        return original(
            value,
            identity_cache=identity_cache,
            chain_signing_profile=chain_signing_profile,
            boot_verifier=lambda _boot, expected_pcr0=None: {
                "verified": True,
                "pcr0": expected_pcr0,
                "boundary": "local_nitro_attestation",
            },
        )

    auditor_module.verify_attested_weight_authority_v2 = (
        verify_with_local_nitro
    )
    try:
        verified = auditor.verify_attested_weights_v2(
            dict(authority),
            identity_cache=dict(identity_cache),
        )
    finally:
        auditor_module.verify_attested_weight_authority_v2 = original
    if verified is None:
        raise RuntimeError("production auditor rejected local authority")
    submitted = auditor.submit_weights_to_chain(
        epoch_id,
        verified,
        submission_epoch_id=epoch_id,
    )
    if not submitted:
        raise RuntimeError("production auditor did not finalize local weights")
    return verified


def _file_identity(path: str, candidate_sha: str) -> dict[str, str]:
    source = SOURCE_ROOT / path
    if not source.is_file():
        raise RuntimeError(f"candidate production source is absent: {path}")
    import subprocess

    expected = subprocess.run(
        ["git", "-C", str(SOURCE_ROOT), "show", f"{candidate_sha}:{path}"],
        check=True,
        capture_output=True,
    ).stdout
    observed = source.read_bytes()
    if observed != expected:
        raise RuntimeError(f"candidate production source differs: {path}")
    return {
        "path": path,
        "sha256": hashlib.sha256(observed).hexdigest(),
        "commit_sha": candidate_sha,
    }


def _receipt(
    *,
    epoch_id: int,
    candidate_sha: str,
    role: str,
    purpose: str,
    job_id: str,
    private_key: Ed25519PrivateKey,
    boot: Mapping[str, Any],
    config_hash: str,
    input_root: str,
    output_root: str,
    parents: list[str],
    sequence: int,
    transport_attempts: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    attempts = list(transport_attempts or [])
    public_key = private_key.public_key().public_bytes_raw().hex()
    artifact_hashes = [
        item[key]
        for item in attempts
        for key in ("request_artifact_hash", "response_artifact_hash")
    ]
    body = build_execution_receipt_body(
        role=role,
        purpose=purpose,
        job_id=job_id,
        epoch_id=epoch_id,
        sequence=sequence,
        commit_sha=candidate_sha,
        pcr0=str(boot["pcr0"]),
        build_manifest_hash=str(boot["build_manifest_hash"]),
        dependency_lock_hash=str(boot["dependency_lock_hash"]),
        config_hash=config_hash,
        boot_identity_hash=str(boot["boot_identity_hash"]),
        input_root=input_root,
        output_root=output_root,
        transport_root_hash=(
            merkle_root(
                [str(item["attempt_hash"]) for item in attempts],
                domain="leadpoet-transport-v2",
            )
            if attempts
            else EMPTY_TRANSPORT_ROOT
        ),
        host_operation_root_hash=EMPTY_HOST_OPERATION_ROOT,
        artifact_root=(
            merkle_root(artifact_hashes, domain="leadpoet-artifact-v2")
            if artifact_hashes
            else EMPTY_ARTIFACT_ROOT
        ),
        parent_receipt_hashes=parents,
        status="succeeded",
        failure_code=None,
        issued_at=NOW,
    )
    return create_signed_execution_receipt(
        body=body,
        enclave_pubkey=public_key,
        sign_digest=private_key.sign,
    )


def _exercise_sdk_bridge(
    *,
    epoch_id: int,
    uids: list[int],
    weights_u16: list[int],
    submission_event_hash: str,
) -> dict[str, Any]:
    """Run the production Bittensor SDK interception with strict boundaries."""

    client = LocalEnclaveSigningBoundary()
    substrate = LocalSDKSubstrateBoundary()
    wallet = local_enclave_backed_wallet(client)
    mechanism = _weight_extrinsic_module()
    with AuthoritativeSetWeightsContextV2(
        substrate=substrate,
        wallet=wallet,
        weight_authorization_id=sha256_json(
            {"epoch_id": epoch_id, "kind": "sdk-weight-authorization"}
        ),
        weight_submission_event_hash=submission_event_hash,
        expected_era_period=8,
    ) as context:
        mechanism.get_encrypted_commit_v2(
            uids=uids,
            weights=weights_u16,
            version_key=10005000,
            last_epoch_block=epoch_id * 360,
            pending_epoch_at=0,
            subnet_epoch_index=epoch_id,
            tempo=360,
            blocks_since_last_step=22,
            current_block=epoch_id * 360 + 22,
            subnet_reveal_period_epochs=1,
            block_time=12.0,
            hotkey=wallet.hotkey.public_key,
        )
        signed = substrate.create_signed_extrinsic(
            call=object(),
            keypair=wallet.hotkey,
            era={"period": 8},
            nonce=None,
        )
    commit_requests = [
        request for kind, request in client.requests if kind == "commit"
    ]
    extrinsic_requests = [
        request for kind, request in client.requests if kind == "extrinsic"
    ]
    if (
        len(commit_requests) != 1
        or len(extrinsic_requests) != 1
        or commit_requests[0]["uids"] != uids
        or commit_requests[0]["weights_u16"] != weights_u16
        or len(context.extrinsic_signature_results) != 1
    ):
        raise RuntimeError("production SDK signing bridge evidence differs")
    return {
        "verified": True,
        "commit_request_hash": sha256_json(commit_requests[0]),
        "extrinsic_request_hash": sha256_json(extrinsic_requests[0]),
        "signature_hex": bytes(signed.signature).hex(),
    }


def _recompose_candidate_bundle(
    *,
    epoch_fixture: SanitizedWeightFixture,
    bundle: Mapping[str, Any],
    epoch_id: int,
) -> dict[str, Any]:
    binding_receipt = next(
        receipt
        for receipt in bundle["receipt_graph"]["receipts"]
        if receipt["purpose"] == "validator.hotkey_signature.v2"
    )
    weight_boot_for_handoff = next(
        identity
        for identity in bundle["receipt_graph"]["boot_identities"]
        if identity["physical_role"] == "validator_weights"
    )
    enclave_graph = build_receipt_graph(
        root_receipt_hash=binding_receipt["parent_receipt_hashes"][0],
        boot_identities=bundle["receipt_graph"]["boot_identities"],
        receipts=[
            receipt
            for receipt in bundle["receipt_graph"]["receipts"]
            if receipt["receipt_hash"] != binding_receipt["receipt_hash"]
        ],
        transport_attempts=bundle["receipt_graph"]["transport_attempts"],
        host_operations=bundle["receipt_graph"]["host_operations"],
    )
    return build_authoritative_weight_bundle_v2(
        enclave_response={
            "weight_snapshot": bundle["weight_snapshot"],
            "weight_result": bundle["weight_result"],
            "weights_signature": bundle["weights_signature"],
            "receipt_graph": enclave_graph,
            "boot_identity": weight_boot_for_handoff,
            "weight_authorization_id": sha256_json(
                {"epoch_id": epoch_id, "kind": "local-authorization"}
            ),
            "source_artifacts": [],
        },
        validator_hotkey=bundle["validator_hotkey"],
        binding_message=bundle["binding_message"],
        binding_signature_result={
            "purpose": "validator.gateway_binding.v2",
            "validator_hotkey": bundle["validator_hotkey"],
            "signature": bundle["validator_hotkey_signature"],
            "receipt": binding_receipt,
        },
    )


def _run_independent_epoch_diagnostics(
    *,
    candidate_sha: str,
    epoch_id: int,
    stages: list[dict[str, Any]],
) -> None:
    """Exercise independent downstream contracts before the joined epoch."""

    epoch_fixture = SanitizedWeightFixture(
        candidate_sha=candidate_sha,
        epoch_id=epoch_id,
    )
    bundle_passed, bundle = _run_workflow_stage(
        stage="diagnostic:candidate-bundle-generation",
        action=epoch_fixture.bundle,
        stages=stages,
    )
    dependent_stages = (
        "diagnostic:host-bundle-composition",
        "diagnostic:primary-bundle-verification",
        "diagnostic:auditor-bundle-verification",
        "diagnostic:primary-auditor-vector-equality",
        "diagnostic:sdk-signing-bridge",
    )
    if not bundle_passed:
        for stage in dependent_stages:
            _mark_workflow_stage_unexercised(
                stage=stage,
                blocked_by=["diagnostic:candidate-bundle-generation"],
                stages=stages,
            )
        return

    _run_workflow_stage(
        stage="diagnostic:host-bundle-composition",
        action=lambda: _require_equal(
            _recompose_candidate_bundle(
                epoch_fixture=epoch_fixture,
                bundle=bundle,
                epoch_id=epoch_id,
            ),
            bundle,
            "production host bundle composition differs from canonical fixture",
        ),
        stages=stages,
    )
    primary_passed, primary = _run_workflow_stage(
        stage="diagnostic:primary-bundle-verification",
        action=lambda: validate_published_weight_bundle_v2(bundle),
        stages=stages,
    )
    auditor_passed, auditor = _run_workflow_stage(
        stage="diagnostic:auditor-bundle-verification",
        action=lambda: verify_attested_weight_bundle_v2(
            bundle,
            identity_cache=epoch_fixture.identity_cache(bundle),
            boot_verifier=lambda _boot, expected_pcr0=None: {
                "verified": True,
                "pcr0": expected_pcr0,
                "boundary": "local_nitro_attestation",
            },
        ),
        stages=stages,
    )
    if primary_passed and auditor_passed:
        _run_workflow_stage(
            stage="diagnostic:primary-auditor-vector-equality",
            action=lambda: _require_equal(
                {
                    "uids": list(primary["uids"]),
                    "weights_u16": list(primary["weights_u16"]),
                },
                {
                    "uids": list(auditor["uids"]),
                    "weights_u16": list(auditor["weights_u16"]),
                },
                "primary and auditor canonical vectors differ",
            ),
            stages=stages,
        )
    else:
        blocked_by = []
        if not primary_passed:
            blocked_by.append("diagnostic:primary-bundle-verification")
        if not auditor_passed:
            blocked_by.append("diagnostic:auditor-bundle-verification")
        _mark_workflow_stage_unexercised(
            stage="diagnostic:primary-auditor-vector-equality",
            blocked_by=blocked_by,
            stages=stages,
        )
    _run_workflow_stage(
        stage="diagnostic:sdk-signing-bridge",
        action=lambda: _exercise_sdk_bridge(
            epoch_id=epoch_id,
            uids=[
                int(value)
                for value in bundle["weight_result"]["sparse_uids"]
            ],
            weights_u16=[
                int(value)
                for value in bundle["weight_result"]["sparse_weights_u16"]
            ],
            submission_event_hash=sha256_json(
                {"epoch_id": epoch_id, "kind": "diagnostic-publication"}
            ),
        ),
        stages=stages,
    )


def _run_epoch(
    *,
    services: LocalBoundaryServices,
    fixture: Mapping[str, Any],
    candidate_sha: str,
    epoch_id: int,
) -> dict[str, Any]:
    epoch_fixture = SanitizedWeightFixture(
        candidate_sha=candidate_sha,
        epoch_id=epoch_id,
    )
    coordinator_key = epoch_fixture.coordinator_key
    weight_key = epoch_fixture.weight_key
    bundle = epoch_fixture.bundle()
    assembled_bundle = _recompose_candidate_bundle(
        epoch_fixture=epoch_fixture,
        bundle=bundle,
        epoch_id=epoch_id,
    )
    if assembled_bundle != bundle:
        raise RuntimeError(
            "production host bundle composition differs from canonical fixture"
        )
    verified_bundle = validate_published_weight_bundle_v2(bundle)
    identity_cache = epoch_fixture.identity_cache(bundle)
    auditor_bundle = verify_attested_weight_bundle_v2(
        bundle,
        identity_cache=identity_cache,
        boot_verifier=lambda _boot, expected_pcr0=None: {
            "verified": True,
            "pcr0": expected_pcr0,
            "boundary": "local_nitro_attestation",
        },
    )
    primary_vector = {
        "uids": list(verified_bundle["uids"]),
        "weights_u16": list(verified_bundle["weights_u16"]),
    }
    auditor_vector = {
        "uids": list(auditor_bundle["uids"]),
        "weights_u16": list(auditor_bundle["weights_u16"]),
    }
    if primary_vector != auditor_vector:
        raise RuntimeError("primary and auditor canonical vectors differ")

    persisted_bundle = services.request(
        "POST",
        "/database/insert",
        {
            "kind": "published_weight_bundle_v2",
            "epoch_id": epoch_id,
            "body": bundle,
        },
    )
    coordinator_boot = next(
        identity
        for identity in bundle["receipt_graph"]["boot_identities"]
        if identity["physical_role"] == "gateway_coordinator"
    )
    weight_boot = next(
        identity
        for identity in bundle["receipt_graph"]["boot_identities"]
        if identity["physical_role"] == "validator_weights"
    )
    publication_doc = {
        "schema_version": "leadpoet.weight_publication.v2",
        "bundle_hash": verified_bundle["bundle_hash"],
        "root_receipt_hash": verified_bundle["root_receipt_hash"],
        "durable_readback_hash": persisted_bundle["evidence_hash"],
        "transparency_event_hash": sha256_json(
            {"epoch_id": epoch_id, "kind": "transparency"}
        ),
    }
    publication_receipt = _receipt(
        epoch_id=epoch_id,
        candidate_sha=candidate_sha,
        role="gateway_coordinator",
        purpose="gateway.weights.publication.v2",
        job_id=f"weight-publication-{epoch_id}",
        private_key=coordinator_key,
        boot=coordinator_boot,
        config_hash=str(coordinator_boot["config_hash"]),
        input_root=sha256_json({"publication": "input", "epoch_id": epoch_id}),
        output_root=sha256_json(publication_doc),
        parents=[verified_bundle["root_receipt_hash"]],
        sequence=200,
    )
    publication_graph = build_receipt_graph(
        root_receipt_hash=publication_receipt["receipt_hash"],
        boot_identities=bundle["receipt_graph"]["boot_identities"],
        receipts=bundle["receipt_graph"]["receipts"] + [publication_receipt],
        transport_attempts=bundle["receipt_graph"]["transport_attempts"],
    )
    submission_event_hash = sha256_json(
        {
            "bundle_hash": verified_bundle["bundle_hash"],
            "publication_receipt_hash": publication_receipt["receipt_hash"],
            "transparency_event_hash": publication_doc["transparency_event_hash"],
            "durable_readback_hash": publication_doc["durable_readback_hash"],
        }
    )
    sdk_bridge = _exercise_sdk_bridge(
        epoch_id=epoch_id,
        uids=primary_vector["uids"],
        weights_u16=primary_vector["weights_u16"],
        submission_event_hash=submission_event_hash,
    )

    profile_manifest = json.loads(
        (
            SOURCE_ROOT
            / "validator_tee/enclave/chain_signing_profile_v2.json"
        ).read_text(encoding="utf-8")
    )
    profile = next(
        item
        for item in chain_signing_profiles(profile_manifest)
        if int(item["spec_version"])
        == int(profile_manifest["spec_version"])
    )
    seed = hashlib.sha256(
        b"hotkey-seed:" + candidate_sha.encode("ascii")
    ).digest()
    sr25519 = _Sr25519Backend()
    public_key, secret_key = sr25519.pair_from_seed(seed)
    commitment = hashlib.sha512(
        b"timelocked:" + epoch_id.to_bytes(8, "big") + _canonical(primary_vector)
    ).digest()
    block = int(verified_bundle["block"])
    authorization = build_weight_extrinsic_authorization_v2(
        profile=profile,
        validator_hotkey=verified_bundle["validator_hotkey"],
        hotkey_public_key_hex=public_key.hex(),
        epoch_id=epoch_id,
        netuid=int(verified_bundle["netuid"]),
        subnet_epoch_index=epoch_id,
        weight_receipt_hash=verified_bundle["weight_receipt_hash"],
        weight_submission_event_hash=submission_event_hash,
        weights_hash=verified_bundle["weights_hash"],
        sparse_uids=primary_vector["uids"],
        sparse_weights_u16=primary_vector["weights_u16"],
        commitment=commitment,
        reveal_round=epoch_id + 1,
        era_current=block,
        nonce=epoch_id,
        block_hash=hashlib.sha256(f"block:{block}".encode("ascii")).hexdigest(),
    )
    signature = sr25519.sign(
        (public_key, secret_key),
        bytes.fromhex(authorization["signed_message_hex"]),
    )
    signed_extrinsic = encode_signed_extrinsic_v2(
        hotkey_public_key_hex=public_key.hex(),
        signature_hex=signature.hex(),
        era_period=int(authorization["era_period"]),
        era_current=int(authorization["era_current"]),
        nonce=int(authorization["nonce"]),
        call_data_hex=str(authorization["call_data_hex"]),
    )
    extrinsic_hash = signed_extrinsic_hash_v2(signed_extrinsic)
    services.request(
        "POST",
        "/chain/submit_extrinsic",
        {
            "epoch_id": epoch_id,
            "extrinsic_hash": extrinsic_hash,
            "extrinsic_hex": signed_extrinsic.hex(),
            "bundle_hash": verified_bundle["bundle_hash"],
            "weights_hash": verified_bundle["weights_hash"],
            **primary_vector,
        },
    )
    finalized = services.request(
        "POST",
        "/chain/finalize",
        {
            "epoch_id": epoch_id,
            "extrinsic_hash": extrinsic_hash,
            "finalized_block": block + 1,
        },
    )

    extrinsic_output = {
        "schema_version": "leadpoet.weight_extrinsic_signature.v2",
        "authorization_hash": authorization["authorization_hash"],
        "validator_hotkey": verified_bundle["validator_hotkey"],
        "signature": signature.hex(),
        "extrinsic_hash": extrinsic_hash,
    }
    extrinsic_receipt = _receipt(
        epoch_id=epoch_id,
        candidate_sha=candidate_sha,
        role="validator_weights",
        purpose="validator.set_weights_extrinsic.v2",
        job_id=f"set-weights-{epoch_id}",
        private_key=weight_key,
        boot=weight_boot,
        config_hash=str(weight_boot["config_hash"]),
        input_root=authorization["authorization_hash"],
        output_root=sha256_json(extrinsic_output),
        parents=[verified_bundle["weight_receipt_hash"]],
        sequence=201,
    )
    finalization_job = f"weight-finalization-{epoch_id}"
    attempts = [
        epoch_fixture.source_attempt(
            category="weight-finalization",
            job_id=finalization_job,
            purpose="validator.weights.finalized.v2",
            sequence=300,
            provider_id="bittensor_chain",
            host="entrypoint-finney.opentensor.ai",
            method="POST",
        ),
        epoch_fixture.source_attempt(
            category="weight-finalization-archive",
            job_id=finalization_job,
            purpose="validator.weights.finalized.v2",
            sequence=301,
            provider_id="bittensor_archive",
            host="archive.chain.opentensor.ai",
            method="POST",
        ),
    ]
    finalization_doc = {
        "schema_version": "leadpoet.weight_finalization.v2",
        "validator_hotkey": verified_bundle["validator_hotkey"],
        "netuid": int(verified_bundle["netuid"]),
        "epoch_id": epoch_id,
        "weights_hash": verified_bundle["weights_hash"],
        "weight_receipt_hash": verified_bundle["weight_receipt_hash"],
        "weight_submission_event_hash": submission_event_hash,
        "extrinsic_authorization": authorization,
        "extrinsic_authorization_hash": authorization["authorization_hash"],
        "extrinsic_signature": signature.hex(),
        "extrinsic_receipt_hash": extrinsic_receipt["receipt_hash"],
        "extrinsic_hash": extrinsic_hash,
        "finalized_block": int(finalized["finalized_block"]),
        "finalized_block_hash": str(finalized["finalized_block_hash"]),
        "state_transition_hash": str(finalized["state_transition_hash"]),
    }
    final_receipt = _receipt(
        epoch_id=epoch_id,
        candidate_sha=candidate_sha,
        role="validator_weights",
        purpose="validator.weights.finalized.v2",
        job_id=finalization_job,
        private_key=weight_key,
        boot=weight_boot,
        config_hash=str(weight_boot["config_hash"]),
        input_root=sha256_json(
            {
                "weight_submission_event_hash": submission_event_hash,
                "extrinsic_receipt_hashes": [extrinsic_receipt["receipt_hash"]],
            }
        ),
        output_root=sha256_json(finalization_doc),
        parents=[extrinsic_receipt["receipt_hash"]],
        sequence=202,
        transport_attempts=attempts,
    )
    final_graph = build_receipt_graph(
        root_receipt_hash=final_receipt["receipt_hash"],
        boot_identities=bundle["receipt_graph"]["boot_identities"],
        receipts=[
            item
            for item in bundle["receipt_graph"]["receipts"]
            if item["purpose"] != "validator.hotkey_signature.v2"
        ]
        + [extrinsic_receipt, final_receipt],
        transport_attempts=bundle["receipt_graph"]["transport_attempts"]
        + attempts,
    )
    finalization_submission = {
        "schema_version": "leadpoet.weight_finalization_submission.v2",
        "validator_hotkey": verified_bundle["validator_hotkey"],
        "weight_submission_event_hash": submission_event_hash,
        "finalization": finalization_doc,
        "receipt_graph": final_graph,
    }
    verified_finalization = validate_weight_finalization_submission_v2(
        finalization_submission,
        chain_signing_profile=profile_manifest,
    )
    finalization_event_hash = sha256_json(
        {
            "weight_submission_event_hash": submission_event_hash,
            "bundle_hash": verified_bundle["bundle_hash"],
            "finalization_receipt_hash": verified_finalization[
                "finalization_receipt_hash"
            ],
            "extrinsic_authorization_hash": authorization["authorization_hash"],
            "extrinsic_hash": extrinsic_hash,
            "finalized_block": finalization_doc["finalized_block"],
            "finalized_block_hash": finalization_doc["finalized_block_hash"],
            "state_transition_hash": finalization_doc["state_transition_hash"],
        }
    )
    authority = {
        "schema_version": "leadpoet.published_weight_authority.v2",
        "bundle": bundle,
        "publication": {
            "weight_submission_event_hash": submission_event_hash,
            "publication_receipt_hash": publication_receipt["receipt_hash"],
            "publication_doc": publication_doc,
            "receipt_graph": publication_graph,
        },
        "finalization": {
            "weight_finalization_event_hash": finalization_event_hash,
            "submission": finalization_submission,
        },
    }
    auditor_authority = verify_attested_weight_authority_v2(
        authority,
        identity_cache=identity_cache,
        chain_signing_profile=profile_manifest,
        boot_verifier=lambda _boot, expected_pcr0=None: {
            "verified": True,
            "pcr0": expected_pcr0,
            "boundary": "local_nitro_attestation",
        },
    )
    if auditor_authority["extrinsic_hash"] != extrinsic_hash:
        raise RuntimeError("auditor finalization differs from local chain")
    production_auditor_authority = _run_production_auditor(
        authority=authority,
        identity_cache=identity_cache,
        epoch_id=epoch_id,
        block=int(verified_bundle["block"]),
    )
    if production_auditor_authority["extrinsic_hash"] != extrinsic_hash:
        raise RuntimeError(
            "production auditor finalization differs from local chain"
        )

    reveal = services.request(
        "POST",
        "/chain/reveal",
        {"epoch_id": epoch_id, **primary_vector},
    )
    last_update = services.request(
        "GET", f"/chain/epoch/{epoch_id}/last_update"
    )
    revealed = services.request("GET", f"/chain/epoch/{epoch_id}/reveal")
    if revealed["reveal"]["vector_hash"] != reveal["vector_hash"]:
        raise RuntimeError("revealed vector readback differs")
    return {
        "epoch_id": epoch_id,
        "pcr0": weight_boot["pcr0"],
        "bundle_hash": verified_bundle["bundle_hash"],
        "root_receipt_hash": verified_bundle["root_receipt_hash"],
        "publication_receipt_hash": publication_receipt["receipt_hash"],
        "finalization_receipt_hash": verified_finalization[
            "finalization_receipt_hash"
        ],
        "receipt_ancestry_verified": True,
        "canonical_vector_hash": sha256_json(primary_vector),
        "canonical_vector_equal": True,
        "weights_hash": verified_bundle["weights_hash"],
        "extrinsic_authorization_hash": authorization["authorization_hash"],
        "signed_extrinsic_hash": extrinsic_hash,
        "sdk_bridge_verified": sdk_bridge["verified"],
        "sdk_commit_request_hash": sdk_bridge["commit_request_hash"],
        "sdk_extrinsic_request_hash": sdk_bridge["extrinsic_request_hash"],
        "finalized_block": finalized["finalized_block"],
        "last_update": last_update["last_update"],
        "reveal_vector_hash": reveal["vector_hash"],
        "auditor_verified": True,
        "auditor_runtime_verified": True,
    }


def _exercise_fault(
    services: LocalBoundaryServices,
    *,
    fault: str,
    ordinal: int,
) -> dict[str, Any]:
    services.inject(fault)
    status = {
        "http_400": 400,
        "http_403": 403,
        "http_429": 429,
        "http_500": 500,
        "duplicate_response": 409,
        "malformed_json": 502,
        "partial_body": 502,
        "unexpected_eof": 502,
        "timeout": 504,
    }.get(fault, 503)
    response = services.request(
        "POST",
        "/database/insert",
        {
            "kind": "fault_probe",
            "epoch_id": -1,
            "body": {"fault": fault, "ordinal": ordinal},
        },
        expected_status=status,
    )
    if response.get("fault") != fault:
        raise RuntimeError(f"fault response differs for {fault}")
    return {"fault": fault, "status": "fail_closed"}


def _exercise_concurrency(services: LocalBoundaryServices) -> int:
    def insert(ordinal: int) -> str:
        response = services.request(
            "POST",
            "/database/insert",
            {
                "kind": "concurrency_probe",
                "epoch_id": -2,
                "body": {"caller": ordinal},
            },
        )
        return str(response["evidence_hash"])

    with ThreadPoolExecutor(max_workers=8) as pool:
        hashes = list(pool.map(insert, range(32)))
    if len(set(hashes)) != 32:
        raise RuntimeError("concurrent durable writes were not isolated")
    return len(hashes)


async def _exercise_chain_settlement_state_space_async() -> dict[str, Any]:
    """Exercise every prefix topology through the production bootstrap gate."""

    from gateway.research_lab import champion_settlement_v2 as settlement
    from gateway.research_lab import store

    netuid = 71
    activation_epoch = 40_000
    target_epoch = activation_epoch + 4
    source_bundle_hash = sha256_json(
        {"kind": "rehearsal-settlement-source", "epoch": activation_epoch}
    )
    activation = {
        "netuid": netuid,
        "schema_version": (
            "leadpoet.research_lab_chain_realized_settlement_activation.v1"
        ),
        "first_epoch_id": activation_epoch,
        "source_bundle_hash": source_bundle_hash,
        "source_bundle_epoch_id": activation_epoch,
        "source_finalized_block": 8_700_039,
    }
    state: dict[str, Any] = {"rows": []}
    validated_ranges: list[tuple[int, int]] = []

    async def select_many(table: str, **_kwargs: Any) -> list[dict[str, Any]]:
        if table == settlement.CHAIN_REALIZED_SETTLEMENT_ACTIVATION_TABLE_V1:
            return [dict(activation)]
        if table == settlement.FINALIZED_ALLOCATION_VIEW_V2:
            return [
                {
                    "bundle_hash": source_bundle_hash,
                    "netuid": netuid,
                    "epoch_id": activation_epoch,
                    "finalized_block": activation["source_finalized_block"],
                    "finalization_receipt_hash": sha256_json(
                        {"kind": "finalization", "epoch": activation_epoch}
                    ),
                }
            ]
        raise AssertionError(f"unexpected settlement select_many table: {table}")

    async def select_all(table: str, **_kwargs: Any) -> list[dict[str, Any]]:
        if table != settlement.CHAIN_REALIZED_EPOCH_SETTLEMENT_TABLE_V1:
            raise AssertionError(
                f"unexpected settlement select_all table: {table}"
            )
        return [dict(row) for row in state["rows"]]

    async def load_chain_history(
        *,
        netuid: int,
        start_epoch: int,
        end_epoch: int,
    ) -> list[dict[str, Any]]:
        if netuid != 71 or start_epoch != activation_epoch:
            raise AssertionError("settlement prefix validation range differs")
        validated_ranges.append((start_epoch, end_epoch))
        return [
            {"epoch": epoch}
            for epoch in range(start_epoch, end_epoch + 1)
        ]

    async def load_finalized_history(
        *,
        netuid: int,
        start_epoch: int,
        end_epoch: int,
    ) -> list[dict[str, Any]]:
        if (
            netuid != 71
            or start_epoch != activation_epoch
            or end_epoch != target_epoch
        ):
            raise AssertionError("finalized source validation range differs")
        return [
            {
                "epoch": activation_epoch,
                "finalized_bundle_hashes": [source_bundle_hash],
            }
        ]

    originals = (
        store.select_many,
        store.select_all,
        settlement.load_chain_realized_allocation_history_v1,
        settlement.load_finalized_allocation_history_v2,
    )
    store.select_many = select_many
    store.select_all = select_all
    settlement.load_chain_realized_allocation_history_v1 = load_chain_history
    settlement.load_finalized_allocation_history_v2 = load_finalized_history
    try:
        accepted: list[dict[str, Any]] = []
        total_epochs = target_epoch - activation_epoch + 1
        for prefix_length in range(total_epochs + 1):
            validated_ranges.clear()
            state["rows"] = [
                {
                    "netuid": netuid,
                    "epoch_id": epoch,
                    "settlement_hash": sha256_json(
                        {"kind": "settlement", "epoch": epoch}
                    ),
                }
                for epoch in range(
                    activation_epoch,
                    activation_epoch + prefix_length,
                )
            ]
            result = (
                await settlement.validate_chain_realized_settlement_bootstrap_v1(
                    netuid=netuid,
                    target_epoch=target_epoch,
                    maximum_backlog=total_epochs,
                )
            )
            expected_status = (
                "pristine_bootstrap_pending"
                if prefix_length == 0
                else "resumable_bootstrap_pending"
            )
            if (
                result["status"] != expected_status
                or result["backlog_epoch_count"]
                != total_epochs - prefix_length
                or result["validated_chain_realized_epochs"]
                != [
                    activation_epoch + offset
                    for offset in range(prefix_length)
                ]
                or (
                    prefix_length > 0
                    and validated_ranges
                    != [
                        (
                            activation_epoch,
                            activation_epoch + prefix_length - 1,
                        )
                    ]
                )
                or (prefix_length == 0 and validated_ranges)
            ):
                raise RuntimeError(
                    "chain settlement prefix behavior differs from contract"
                )
            accepted.append(
                {
                    "prefix_length": prefix_length,
                    "status": result["status"],
                    "backlog_epoch_count": result["backlog_epoch_count"],
                }
            )

        invalid_states = {
            "duplicate": [activation_epoch, activation_epoch],
            "gap": [activation_epoch, activation_epoch + 2],
            "missing-first": [activation_epoch + 1],
            "ahead": list(range(activation_epoch, target_epoch + 2)),
        }
        rejected: list[str] = []
        for name, epochs in invalid_states.items():
            state["rows"] = [
                {
                    "netuid": netuid,
                    "epoch_id": epoch,
                    "settlement_hash": sha256_json(
                        {"kind": "invalid-settlement", "name": name, "epoch": epoch}
                    ),
                }
                for epoch in epochs
            ]
            try:
                await settlement.validate_chain_realized_settlement_bootstrap_v1(
                    netuid=netuid,
                    target_epoch=target_epoch,
                    maximum_backlog=total_epochs,
                )
            except settlement.ChampionSettlementV2Error:
                rejected.append(name)
            else:
                raise RuntimeError(
                    f"invalid settlement topology was accepted: {name}"
                )

        state["rows"] = []
        try:
            await settlement.validate_chain_realized_settlement_bootstrap_v1(
                netuid=netuid,
                target_epoch=target_epoch,
                maximum_backlog=total_epochs - 1,
            )
        except settlement.ChampionSettlementV2Error:
            rejected.append("backlog-exceeds-policy")
        else:
            raise RuntimeError("excessive settlement backlog was accepted")
        return {
            "accepted_prefixes": accepted,
            "accepted_count": len(accepted),
            "rejected_state_classes": sorted(rejected),
        }
    finally:
        (
            store.select_many,
            store.select_all,
            settlement.load_chain_realized_allocation_history_v1,
            settlement.load_finalized_allocation_history_v2,
        ) = originals


def _exercise_chain_settlement_state_space() -> dict[str, Any]:
    return asyncio.run(_exercise_chain_settlement_state_space_async())


def _exercise_conditional_icp_policy() -> dict[str, Any]:
    """Validate configured tails and center through the production selector."""

    from gateway.research_lab.config import ResearchLabGatewayConfig
    from research_lab.eval.conditional_validation import (
        build_conditional_category_assignment,
    )

    policy = (
        ResearchLabGatewayConfig.from_env().conditional_validation_policy()
    )
    policy_doc = policy.to_dict()
    if not policy.enabled:
        try:
            build_conditional_category_assignment(
                rolling_window_hash=sha256_json({"window": "disabled"}),
                benchmark_items=[],
                per_icp_summaries=[],
                policy=policy,
                baseline_serving_model_version_hash=sha256_json(
                    {"model": "disabled"}
                ),
            )
        except ValueError:
            return {
                "mode": policy.mode,
                "policy_hash": policy_doc["policy_hash"],
                "assignment_status": "disabled_fail_closed",
            }
        raise RuntimeError("disabled conditional policy accepted an assignment")

    items = []
    summaries = []
    for index in range(policy.total_icps):
        ref = f"rehearsal-icp-{index:04d}"
        items.append(
            {
                "icp_ref": ref,
                "icp_hash": sha256_json({"icp": index}),
                "intent_signal_signature": sha256_json(
                    {"intent": index}
                ),
                "set_id": index // max(1, policy.fresh_icp_count),
                "day_index": index,
                "day_rank": index + 1,
                "cohort": (
                    "fresh"
                    if index < policy.fresh_icp_count
                    else "retained"
                ),
            }
        )
        score = (
            50.0
            if policy.total_icps == 1
            else (100.0 * index) / (policy.total_icps - 1)
        )
        summaries.append({"icp_ref": ref, "score": score})

    kwargs = {
        "rolling_window_hash": sha256_json({"window": "configured"}),
        "benchmark_items": items,
        "per_icp_summaries": summaries,
        "policy": policy,
        "baseline_serving_model_version_hash": sha256_json(
            {"model": "configured"}
        ),
    }
    assignment = build_conditional_category_assignment(**kwargs)
    replay = build_conditional_category_assignment(**kwargs)
    if assignment != replay:
        raise RuntimeError("conditional ICP assignment is not deterministic")
    rows = sorted(assignment["items"], key=lambda row: float(row["score"]))
    low_refs = {
        row["icp_ref"] for row in rows[: policy.low_tail_count]
    }
    conditional_refs = {
        row["icp_ref"]
        for row in rows[
            policy.low_tail_count : (
                policy.low_tail_count + policy.conditional_total_icps
            )
        ]
    }
    high_refs = {
        row["icp_ref"]
        for row in rows[
            policy.low_tail_count + policy.conditional_total_icps :
        ]
    }
    assigned = assignment["items"]
    actual_conditional = {
        row["icp_ref"]
        for row in assigned
        if row["category"] == "conditional"
    }
    initial_refs = {
        row["icp_ref"]
        for row in assigned
        if row["category"] in {"public", "private"}
    }
    if (
        actual_conditional != conditional_refs
        or initial_refs != low_refs | high_refs
        or assignment["category_counts"]
        != {
            "public": policy.public_total_icps,
            "private": policy.private_total_icps,
            "conditional": policy.conditional_total_icps,
        }
        or sum(
            row["category"] == "public"
            and row["strength_label"] == "weak"
            for row in assigned
        )
        != policy.public_weak_total
        or sum(
            row["category"] == "private"
            and row["strength_label"] == "weak"
            for row in assigned
        )
        != policy.private_weak_total
    ):
        raise RuntimeError(
            "conditional ICP assignment differs from configured tail policy"
        )
    return {
        "policy_hash": policy_doc["policy_hash"],
        "assignment_hash": assignment["assignment_hash"],
        "category_counts": assignment["category_counts"],
        "low_tail_count": policy.low_tail_count,
        "high_tail_count": policy.high_tail_count,
        "conditional_count": policy.conditional_total_icps,
    }


def _exercise_conditional_candidate_gate() -> dict[str, Any]:
    """Prove conditional work runs only after the configured initial gate."""

    from gateway.research_lab.config import ResearchLabGatewayConfig
    from research_lab.eval.evaluator import build_holdout_gate_result

    policy = ResearchLabGatewayConfig.from_env().conditional_validation_policy()
    if not policy.enabled:
        return {
            "mode": policy.mode,
            "policy_hash": policy.to_dict()["policy_hash"],
            "advancement_status": "disabled_fail_closed",
        }

    def rows(prefix: str, count: int, score: float) -> list[dict[str, Any]]:
        return [
            {
                "icp_ref": f"{prefix}-{index:04d}",
                "candidate_company_scores": [float(score)],
            }
            for index in range(count)
        ]

    gate = {
        "conditional_validation_required": True,
        "baseline_benchmark_bundle_id": "private_benchmark:" + "1" * 64,
        "baseline_benchmark_hash": sha256_json({"baseline": "candidate-gate"}),
        "category_assignment_hash": sha256_json({"assignment": "candidate-gate"}),
        "conditional_validation_policy_hash": policy.to_dict()["policy_hash"],
        "baseline_public_score": 0.0,
        "baseline_private_score": 0.0,
        "baseline_conditional_score": 0.0,
        "baseline_preliminary_score": 0.0,
        "baseline_aggregate_score": 0.0,
        "threshold_points": float(policy.threshold_points),
    }
    passing_score = float(policy.threshold_points)
    public = rows("public", policy.public_total_icps, passing_score)
    private = rows("private", policy.private_total_icps, passing_score)
    conditional = rows(
        "conditional",
        policy.conditional_total_icps,
        passing_score,
    )

    preliminary_rows, preliminary = build_holdout_gate_result(
        public_results=public,
        private_results=private,
        conditional_results=(),
        public_icp_count=policy.public_total_icps,
        private_icp_count=policy.private_total_icps,
        conditional_icp_count=policy.conditional_total_icps,
        gate=gate,
    )
    final_rows, final = build_holdout_gate_result(
        public_results=public,
        private_results=private,
        conditional_results=conditional,
        public_icp_count=policy.public_total_icps,
        private_icp_count=policy.private_total_icps,
        conditional_icp_count=policy.conditional_total_icps,
        gate=gate,
    )
    rejected_rows, rejected = build_holdout_gate_result(
        public_results=rows(
            "public-rejected",
            policy.public_total_icps,
            0.0,
        ),
        private_results=rows(
            "private-rejected",
            policy.private_total_icps,
            0.0,
        ),
        conditional_results=conditional,
        public_icp_count=policy.public_total_icps,
        private_icp_count=policy.private_total_icps,
        conditional_icp_count=policy.conditional_total_icps,
        gate={
            **gate,
            "baseline_preliminary_score": 50.0,
        },
    )
    initial_count = policy.public_total_icps + policy.private_total_icps
    if (
        preliminary.get("decision") != "conditional_validation_required"
        or preliminary.get("conditional_holdout_evaluated") is not False
        or len(preliminary_rows) != initial_count
        or final.get("decision") != "conditional_validation_approved"
        or final.get("conditional_holdout_evaluated") is not True
        or len(final_rows) != policy.total_icps
        or rejected.get("decision")
        != "rejected_before_conditional_validation"
        or rejected.get("conditional_holdout_evaluated") is not False
        or len(rejected_rows) != initial_count
    ):
        raise RuntimeError(
            "conditional candidate advancement differs from configured gate"
        )
    return {
        "policy_hash": policy.to_dict()["policy_hash"],
        "initial_count": initial_count,
        "conditional_count": policy.conditional_total_icps,
        "final_count": len(final_rows),
        "preliminary_decision": preliminary["decision"],
        "final_decision": final["decision"],
        "rejected_decision": rejected["decision"],
    }


def _exercise_git_tree_replacement() -> dict[str, Any]:
    """Exercise real SHA-256 Git lineage, recovery, and replacement ancestry."""

    from gateway.research_lab.git_tree_models import (
        TreePolicy,
        TreeReplacement,
        derive_child_slot,
        derive_tree_id,
    )
    from gateway.research_lab.git_tree_repository import GitTreeRepository
    from research_lab.code_editing import CodeEditDraft
    from research_lab.eval.private_runtime import compute_private_source_tree_hash

    policy = TreePolicy.from_env(os.environ)
    run_id = "00000000-0000-4000-8000-000000000001"
    roots = [
        sha256_json({"root": ordinal})
        for ordinal in range(3)
    ]
    manifests = [
        sha256_json({"manifest": ordinal})
        for ordinal in range(3)
    ]
    initial_tree_id = derive_tree_id(
        run_id=run_id,
        root_artifact_hash=roots[0],
        policy=policy,
    )
    first = TreeReplacement(
        generation=1,
        replaces_tree_id=initial_tree_id,
        cancellation_event_hash=sha256_json({"cancel": 0}),
        prior_root_artifact_hash=roots[0],
        prior_root_manifest_hash=manifests[0],
        prior_policy_hash=policy.policy_hash,
        root_artifact_hash=roots[1],
        root_manifest_hash=manifests[1],
        policy_hash=policy.policy_hash,
    )
    first_tree_id = derive_tree_id(
        run_id=run_id,
        root_artifact_hash=roots[1],
        policy=policy,
        replacement=first,
    )
    second = TreeReplacement(
        generation=2,
        replaces_tree_id=first_tree_id,
        cancellation_event_hash=sha256_json({"cancel": 1}),
        prior_root_artifact_hash=roots[1],
        prior_root_manifest_hash=manifests[1],
        prior_policy_hash=policy.policy_hash,
        root_artifact_hash=roots[2],
        root_manifest_hash=manifests[2],
        policy_hash=policy.policy_hash,
        reason="replacement_target_advanced",
    )
    second_tree_id = derive_tree_id(
        run_id=run_id,
        root_artifact_hash=roots[2],
        policy=policy,
        replacement=second,
    )
    if (
        len({initial_tree_id, first_tree_id, second_tree_id}) != 3
        or TreeReplacement.from_mapping(first.to_dict()) != first
        or TreeReplacement.from_mapping(second.to_dict()) != second
        or derive_tree_id(
            run_id=run_id,
            root_artifact_hash=roots[2],
            policy=policy,
            replacement=second,
        )
        != second_tree_id
    ):
        raise RuntimeError("Git-tree replacement identity is not deterministic")

    with tempfile.TemporaryDirectory(prefix="leadpoet-rehearsal-git-tree-") as tmp:
        root = Path(tmp)
        source = root / "source"
        (source / "gateway").mkdir(parents=True)
        (source / "gateway/runtime.py").write_text(
            "VALUE = 0\n", encoding="utf-8"
        )
        (source / "research_lab_adapter.py").write_text(
            "def run_icp(icp, context):\n    return []\n",
            encoding="utf-8",
        )
        workspace = root / "tree"
        repository = GitTreeRepository(
            workspace=workspace,
            tree_id=initial_tree_id,
        )
        root_commit = repository.initialize(
            source_root=source,
            root_artifact_hash=roots[0],
            policy_hash=policy.policy_hash,
            run_id=run_id,
            root_manifest_hash=manifests[0],
            root_image_digest="sha256:" + "a" * 64,
            evaluator_commitment_hash=sha256_json({"evaluator": "rehearsal"}),
            tree_doc={"schema_version": "research_lab.git_tree.v1"},
        )
        slot = derive_child_slot(
            tree_id=initial_tree_id,
            parent_node_id="root",
            root_branch_id="",
            depth=1,
            slot_index=0,
        )
        draft = CodeEditDraft(
            failure_mode="fixture",
            mechanism="change the deterministic fixture value",
            expected_improvement="exercise the candidate Git lineage",
            risk="low",
            lane="query_construction",
            target_files=("gateway/runtime.py",),
            unified_diff=(
                "diff --git a/gateway/runtime.py b/gateway/runtime.py\n"
                "--- a/gateway/runtime.py\n"
                "+++ b/gateway/runtime.py\n"
                "@@ -1 +1 @@\n"
                "-VALUE = 0\n"
                "+VALUE = 1\n"
            ),
            redacted_summary="exercise deterministic Git-tree mutation",
            test_plan="verify source commitment",
            rollback_plan="restore the immutable root",
        )
        child = repository.commit_child(
            slot=slot,
            draft=draft,
            expected_parent_source_tree_hash=(
                compute_private_source_tree_hash(source)
            ),
        )
        repository.verify_node(commit=child)
        operation_id = sha256_json(
            {"tree_id": initial_tree_id, "kind": "generation"}
        )
        request_hash = sha256_json({"draft": draft.to_dict()})
        repository.plan_slot(
            slot=slot,
            request_hash=request_hash,
            operation_id=operation_id,
            node_doc={"node_id": slot.node_id},
        )
        repository.settle_operation(
            operation_id=operation_id,
            operation_status="succeeded",
            request_hash=request_hash,
            result_hash=sha256_json(child.to_dict()),
            settled_cost_microusd=123,
            provider_call_count=1,
            settlement_doc={"operation_kind": "generation"},
        )
        checkpoint_doc = {
            "tree_id": initial_tree_id,
            "node_ids": [slot.node_id],
            "operation_ids": [operation_id],
        }
        checkpoint_hash = sha256_json(checkpoint_doc)
        repository.commit_checkpoint(
            checkpoint_hash=checkpoint_hash,
            checkpoint_doc=checkpoint_doc,
        )
        bundle = repository.create_bundle(root / "tree.bundle")
        recovery = repository.export_recovery_state(
            checkpoint_hash=checkpoint_hash,
            bundle_uri="s3://strict-local-boundary/tree.bundle",
            bundle_hash=str(bundle["bundle_hash"]),
            bundle_size_bytes=int(bundle["bundle_size_bytes"]),
        )
        shutil.rmtree(workspace)
        restored = GitTreeRepository(
            workspace=workspace,
            tree_id=initial_tree_id,
        )
        restored_root = restored.restore_recovery_state(
            recovery_state=recovery,
            bundle_path=Path(str(bundle["bundle_path"])),
        )
        restored.verify_node_identity(
            node_id=slot.node_id,
            git_commit=child.git_commit,
            parent_node_id="root",
        )
        restored_operation = restored.inspect_operation(
            operation_id=operation_id
        )
        if (
            restored_root != root_commit
            or restored.state_status() != "complete"
            or restored_operation.get("operation", {}).get("status")
            != "succeeded"
        ):
            raise RuntimeError("Git-tree checkpoint recovery differs")
    return {
        "policy_hash": policy.policy_hash,
        "max_nodes": policy.max_nodes,
        "tree_ids": [
            initial_tree_id,
            first_tree_id,
            second_tree_id,
        ],
        "replacement_hashes": [
            first.replacement_hash,
            second.replacement_hash,
        ],
        "root_git_commit": root_commit,
        "node_git_commit": child.git_commit,
        "node_source_tree_hash": child.source_tree_hash,
        "checkpoint_hash": checkpoint_hash,
        "recovery_bundle_hash": bundle["bundle_hash"],
        "terminal_operation_count": 1,
        "restart_resume_verified": True,
    }


def _exercise_historical_metagraph_layouts() -> dict[str, Any]:
    """Exercise every candidate-declared archive layout through production."""

    from fixture_contract import (
        load_rehearsal_metagraph_account_ids,
        load_rehearsal_metagraph_hotkeys,
    )
    from gateway.tee.coordinator_chain_source_v2 import (
        CHAIN_ARCHIVE_ENDPOINT_URL,
        CoordinatorChainSourceV2,
    )
    from gateway.tee.execution_job_manager_v2 import ExecutionContextV2
    from leadpoet_canonical.attested_v2 import (
        build_transport_attempt,
        sha256_bytes,
    )
    from leadpoet_canonical.chain_source_v2 import (
        CHAIN_ARCHIVE_ENDPOINT_HOST,
        CHAIN_RPC_METHOD,
        ChainSourceV2Error,
        chain_source_policy_document,
        chain_source_policy_hash,
        encode_selective_metagraph_params,
        weights_storage_key,
    )

    policy = chain_source_policy_document()
    layouts = tuple(
        int(value) for value in policy["selective_result_last_fields"]
    )
    if (
        not layouts
        or tuple(sorted(set(layouts))) != layouts
        or any(value <= 52 for value in layouts)
    ):
        raise RuntimeError(
            "candidate chain-source result layouts are invalid"
        )
    account_ids = load_rehearsal_metagraph_account_ids(SOURCE_ROOT)
    hotkeys = load_rehearsal_metagraph_hotkeys(SOURCE_ROOT)
    validator_hotkey = hotkeys[0]
    cutover = json.loads(
        (
            SOURCE_ROOT
            / "config"
            / "stateful-epoch-cutover-sn71.json"
        ).read_text(encoding="utf-8")
    )
    netuid = int(cutover["netuid"])
    epoch_id = int(cutover["last_legacy_epoch_id"])
    target_block = (epoch_id + 1) * 360 - 1
    retry_hashes = {
        "bittensor_chain": sha256_json({"retry": "chain"}),
        "bittensor_archive": sha256_json({"retry": "archive"}),
        "coingecko": sha256_json({"retry": "coingecko"}),
    }
    def selective_fixture(last_field: int) -> str:
        if netuid < 1 << 6:
            compact_netuid = bytes((netuid << 2,))
        elif netuid < 1 << 14:
            compact_netuid = ((netuid << 2) | 1).to_bytes(2, "little")
        else:
            compact_netuid = ((netuid << 2) | 2).to_bytes(4, "little")
        encoded = bytearray(b"\x01" + compact_netuid)
        encoded.extend(b"\x00" * 4)
        encoded.extend(b"\x01" + account_ids[0])
        encoded.extend(b"\x00")
        encoded.extend(
            b"\x01"
            + ((target_block << 2) | 2).to_bytes(4, "little")
        )
        encoded.extend(b"\x00" * 44)
        encoded.extend(
            b"\x01"
            + bytes((len(account_ids) << 2,))
            + b"".join(account_ids)
        )
        encoded.extend(b"\x00" * (int(last_field) - 52))
        return "0x" + bytes(encoded).hex()

    class StrictArchiveBoundary:
        def __init__(self, *, last_field: int) -> None:
            self.last_field = int(last_field)
            self.calls: list[dict[str, Any]] = []

        def execute(
            self,
            request: Mapping[str, Any],
        ) -> dict[str, Any]:
            if (
                request.get("provider_id") != "bittensor_archive"
                or request.get("method") != "POST"
                or request.get("url") != CHAIN_ARCHIVE_ENDPOINT_URL
                or request.get("retry_policy_hash")
                != retry_hashes["bittensor_archive"]
            ):
                raise RuntimeError(
                    "historical layout probe crossed an undeclared boundary"
                )
            request_body = base64.b64decode(
                str(request["body_b64"]),
                validate=True,
            )
            rpc = json.loads(request_body)
            if set(rpc) != {"jsonrpc", "id", "method", "params"} or (
                rpc.get("jsonrpc") != "2.0"
            ):
                raise RuntimeError(
                    "historical layout probe received malformed JSON-RPC"
                )
            method = rpc.get("method")
            call_number = len(self.calls) + 1
            self.calls.append(
                {
                    "method": method,
                    "params": rpc.get("params"),
                }
            )
            if method == "chain_getFinalizedHead":
                if rpc.get("params") != []:
                    raise RuntimeError(
                        "historical finalized-head request differs"
                    )
                value: Any = "0x" + "a" * 64
            elif method == "chain_getBlockHash":
                if rpc.get("params") != [target_block]:
                    raise RuntimeError(
                        "historical layout probe requested another block"
                    )
                value = "0x" + "b" * 64
            elif method == "chain_getHeader":
                at_hash = str((rpc.get("params") or [""])[0])
                is_target = at_hash == "0x" + "b" * 64
                if at_hash not in {
                    "0x" + "a" * 64,
                    "0x" + "b" * 64,
                }:
                    raise RuntimeError(
                        "historical layout probe requested another hash"
                    )
                value = {
                    "number": hex(
                        target_block if is_target else target_block + 20
                    ),
                    "stateRoot": "0x" + "c" * 64,
                    "parentHash": "0x" + "d" * 64,
                    "extrinsicsRoot": "0x" + "e" * 64,
                    "digest": {"logs": []},
                }
            elif method == "state_call":
                if rpc.get("params") != [
                    CHAIN_RPC_METHOD,
                    encode_selective_metagraph_params(netuid=netuid),
                    "0x" + "b" * 64,
                ]:
                    raise RuntimeError(
                        "historical selective metagraph request differs"
                    )
                value = selective_fixture(self.last_field)
            elif method == "state_getStorage":
                if rpc.get("params") != [
                    weights_storage_key(
                        netuid=netuid,
                        validator_uid=0,
                    ),
                    "0x" + "b" * 64,
                ]:
                    raise RuntimeError(
                        "historical weight-storage request differs"
                    )
                value = "0x" + (
                    b"\x08"
                    + (1).to_bytes(2, "little")
                    + (1000).to_bytes(2, "little")
                    + (4).to_bytes(2, "little")
                    + (2000).to_bytes(2, "little")
                ).hex()
            else:
                raise RuntimeError(
                    "historical layout probe received an unknown RPC"
                )
            response_body = json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": rpc.get("id"),
                    "result": value,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            artifact_hash = sha256_json(
                {
                    "call": call_number,
                    "layout": self.last_field,
                    "method": method,
                }
            )
            attempt = build_transport_attempt(
                request_id=("%032x" % call_number),
                logical_operation_id=str(
                    request["logical_operation_id"]
                ),
                job_id=str(request["job_id"]),
                purpose=str(request["purpose"]),
                provider_id="bittensor_archive",
                attempt_number=int(request["attempt_number"]),
                method="POST",
                destination_host=CHAIN_ARCHIVE_ENDPOINT_HOST,
                destination_port=443,
                path_hash=sha256_json({"path": "/"}),
                nonsecret_headers_hash=sha256_json(
                    {"headers": "application/json"}
                ),
                body_hash=sha256_bytes(request_body),
                credential_ref_hash=sha256_json(
                    {"credential": "public-archive"}
                ),
                retry_policy_hash=str(request["retry_policy_hash"]),
                timeout_ms=int(request["timeout_ms"]),
                started_at=NOW,
                terminal_status="authenticated_response",
                http_status=200,
                response_hash=sha256_bytes(response_body),
                request_artifact_hash=artifact_hash,
                response_artifact_hash=sha256_bytes(response_body),
                tls_peer_chain_hash=sha256_json(
                    {"tls": "archive-rehearsal"}
                ),
                tls_protocol="TLSv1.3",
                failure_code=None,
                completed_at=NOW,
            )
            return {
                "terminal_status": "authenticated_response",
                "http_status": 200,
                "body_b64": base64.b64encode(response_body).decode(
                    "ascii"
                ),
                "transport_attempt": attempt,
            }

    def execute_layout(last_field: int) -> tuple[dict[str, Any], int]:
        boundary = StrictArchiveBoundary(last_field=last_field)
        source = CoordinatorChainSourceV2(
            execute_provider=boundary.execute,
            retry_policy_hashes=retry_hashes,
            epoch_authority={
                "mode": "stateful_v1",
                "cutover": cutover,
            },
            sleep=lambda _seconds: None,
        )
        context = ExecutionContextV2(
            job_id=f"rehearsal:historical-layout:{last_field}",
            purpose="research_lab.legacy_finalized_allocation.v2",
            epoch_id=epoch_id,
        )
        result = source.read_historical_finalized_weights(
            netuid=netuid,
            epoch_id=epoch_id,
            validator_hotkey=validator_hotkey,
            context=context,
        )
        return result, len(boundary.calls)

    accepted: list[int] = []
    call_counts: dict[str, int] = {}
    for last_field in layouts:
        result, call_count = execute_layout(last_field)
        if (
            result["target_block"] != target_block
            or result["validator_uid"] != 0
            or result["weights"] != [[1, 1000], [4, 2000]]
            or call_count != 6
        ):
            raise RuntimeError(
                "historical archive layout produced different authority"
            )
        accepted.append(last_field)
        call_counts[str(last_field)] = call_count

    rejected_layout = next(
        (
            value
            for value in range(53, max(layouts) + 1)
            if value not in layouts
        ),
        max(layouts) + 1,
    )
    try:
        execute_layout(rejected_layout)
    except ChainSourceV2Error:
        pass
    else:
        raise RuntimeError(
            "undeclared historical archive layout did not fail closed"
        )
    return {
        "policy_hash": chain_source_policy_hash(),
        "accepted_layouts": accepted,
        "rejected_layout": rejected_layout,
        "rpc_call_counts": call_counts,
    }


def _exercise_research_lab_allocation_conservation() -> dict[str, Any]:
    """Exercise the configured no-burn and compatibility allocation modes."""

    from gateway.research_lab.config import ResearchLabGatewayConfig
    from leadpoet_verifier.economics import allocate_research_lab_epoch

    policy = ResearchLabGatewayConfig.from_env().reimbursement_policy_doc(
        enabled=True
    )
    policy_hash = sha256_json(policy)
    epoch = 30_000
    cap = Decimal(str(policy["research_lab_emission_percent"]))
    if (
        cap <= 0
        or policy.get("enable_conservative") is not False
        or policy.get("enable_champ_cap") is not False
        or Decimal(
            str(
                policy[
                    "reimbursement_max_cost_multiplier_with_champions"
                ]
            )
        )
        != Decimal("2")
    ):
        raise RuntimeError(
            "Research Lab default allocation policy differs from no-burn V2"
        )

    def reimbursement(
        uid: int,
        compute_microusd: int,
    ) -> dict[str, Any]:
        return {
            "uid": uid,
            "miner_hotkey": "reimbursement-%d" % uid,
            "source_id": "reimbursement_schedule:rehearsal-%d" % uid,
            "island": "generalist",
            "status": "active",
            "start_epoch": epoch,
            "epoch_count": int(policy["reimbursement_epochs"]),
            "target_reimbursement_microusd": compute_microusd,
            "eligible_compute_microusd": compute_microusd,
        }

    current = allocate_research_lab_epoch(
        epoch,
        policy,
        [reimbursement(1, 1_000_000), reimbursement(2, 3_000_000)],
        [],
    )
    current_paid = {
        int(row["uid"]): Decimal(str(row["paid_alpha_percent"]))
        for row in current["reimbursement_allocations"]
    }
    if (
        sum(current_paid.values()) != cap
        or current_paid[2] != current_paid[1] * Decimal("3")
        or Decimal(str(current["unallocated_percent"])) != 0
    ):
        raise RuntimeError(
            "current reimbursements did not conserve the Lab cap by compute"
        )

    source_hash = sha256_json({"fixture": "historical-compute"})

    def fallback(uid: int, compute_microusd: int) -> dict[str, Any]:
        return {
            "uid": uid,
            "miner_hotkey": "fallback-%d" % uid,
            "source_id": "historical_compute_fallback:%064d" % uid,
            "island": "historical_compute",
            "status": "active",
            "target_reimbursement_microusd": compute_microusd,
            "fallback_window_start_epoch": epoch - 20,
            "fallback_window_end_epoch": epoch - 1,
            "source_allocation_epoch": epoch - 1,
            "source_allocation_hash": source_hash,
            "contribution_count": 1,
            "contribution_hash": sha256_json(
                {"uid": uid, "compute_microusd": compute_microusd}
            ),
        }

    historical = allocate_research_lab_epoch(
        epoch,
        policy,
        [],
        [],
        fallback_reimbursement_obligations=[
            fallback(3, 1_000_000),
            fallback(4, 3_000_000),
        ],
    )
    historical_paid = {
        int(row["uid"]): Decimal(str(row["paid_alpha_percent"]))
        for row in historical["reimbursement_allocations"]
    }
    if (
        sum(historical_paid.values()) != cap
        or historical_paid[4] != historical_paid[3] * Decimal("3")
        or historical.get("historical_compute_fallback_source_epoch")
        != epoch - 1
        or Decimal(str(historical["unallocated_percent"])) != 0
    ):
        raise RuntimeError(
            "historical compute fallback did not conserve the Lab cap"
        )

    champions = [
        {
            "uid": 5,
            "miner_hotkey": "champion-5",
            "source_id": "champion_reward:rehearsal-5",
            "champion_reward_id": "champion_reward:rehearsal-5",
            "island": "generalist",
            "status": "active",
            "start_epoch": epoch,
            "epoch_count": int(policy["reward_epochs"]),
            "improvement_points": 1.0,
            "desired_alpha_percent": 7.0,
        },
        {
            "uid": 6,
            "miner_hotkey": "champion-6",
            "source_id": "champion_reward:rehearsal-6",
            "champion_reward_id": "champion_reward:rehearsal-6",
            "island": "generalist",
            "status": "active",
            "start_epoch": epoch,
            "epoch_count": int(policy["reward_epochs"]),
            "improvement_points": 2.0,
            "desired_alpha_percent": 14.0,
        },
    ]
    champion_allocation = allocate_research_lab_epoch(
        epoch,
        policy,
        [],
        champions,
    )
    champion_paid = {
        int(row["uid"]): Decimal(str(row["paid_alpha_percent"]))
        for row in [
            *champion_allocation["champion_allocations"],
            *champion_allocation["queued_champion_allocations"],
        ]
    }
    if (
        sum(champion_paid.values()) != cap
        or champion_paid[6] != champion_paid[5] * Decimal("2")
        or Decimal(str(champion_allocation["unallocated_percent"])) != 0
    ):
        raise RuntimeError(
            "champions did not split the remaining Lab cap by configured reward"
        )

    valuation_microusd = int(
        (
            Decimal(str(policy["usd_per_0_1_percent_epoch"]))
            * Decimal(1_000_000)
        ).to_integral_value()
    )
    capped = allocate_research_lab_epoch(
        epoch,
        policy,
        [
            reimbursement(
                7,
                valuation_microusd * int(policy["reimbursement_epochs"]),
            )
        ],
        [champions[0]],
    )
    capped_reimbursement = Decimal(
        str(capped["reimbursement_allocations"][0]["paid_alpha_percent"])
    )
    if (
        capped_reimbursement != Decimal("0.2")
        or Decimal(str(capped["champion_alpha_percent"]))
        != cap - capped_reimbursement
        or Decimal(str(capped["unallocated_percent"])) != 0
    ):
        raise RuntimeError(
            "active-champion reimbursement cap or remainder differs"
        )

    conservative_policy = dict(policy)
    conservative_policy["enable_conservative"] = True
    conservative = allocate_research_lab_epoch(
        epoch,
        conservative_policy,
        [],
        [],
    )
    if (
        Decimal(str(conservative["unallocated_percent"])) != cap
        or conservative["reimbursement_allocations"]
        or conservative["champion_allocations"]
    ):
        raise RuntimeError(
            "conservative compatibility mode no longer preserves burn"
        )
    return {
        "policy_hash": policy_hash,
        "lab_cap_percent": float(cap),
        "current_reimbursement_alpha_percent": float(
            current["reimbursement_alpha_percent"]
        ),
        "historical_reimbursement_alpha_percent": float(
            historical["reimbursement_alpha_percent"]
        ),
        "champion_alpha_percent": float(
            champion_allocation["champion_alpha_percent"]
        ),
        "active_champion_reimbursement_alpha_percent": float(
            capped["reimbursement_alpha_percent"]
        ),
        "conservative_unallocated_percent": float(
            conservative["unallocated_percent"]
        ),
        "conserved": True,
    }


def _exercise_settlement_frontier_terminal_retirement() -> dict[str, Any]:
    """Reproduce and close the terminal-obligation frontier transition."""

    from gateway.tee.coordinator_allocation_source_v2 import (
        CoordinatorAllocationSourceV2,
        CoordinatorAllocationSourceV2Error,
    )
    from gateway.tee.execution_job_manager_v2 import ExecutionContextV2
    from gateway.tee.reward_executor_v2 import (
        champion_reward_row_projection_v2,
        source_add_reward_row_projection_v2,
    )
    from leadpoet_canonical.allocation_settlement_frontier_v2 import (
        build_allocation_settlement_frontier_v2,
        build_reward_settlement_checkpoint_v2,
        frontier_artifact_hashes_v2,
    )
    from leadpoet_canonical.attested_v2 import (
        build_boot_identity_body,
        create_boot_identity,
    )

    champion = {
        "champion_reward_id": "champion_reward:sha256:" + "a" * 64,
        "score_bundle_id": "score-bundle-rehearsal",
        "candidate_id": "candidate-rehearsal",
        "run_id": "run-rehearsal",
        "miner_hotkey": "5ChampionRehearsal",
        "miner_uid": 10,
        "island": "generalist",
        "evaluation_epoch": 119,
        "start_epoch": 120,
        "epoch_count": 20,
        "improvement_points": 1.0,
        "threshold_points": 0.0,
        "desired_alpha_percent": 7.3,
        "input_hash": "sha256:" + "b" * 64,
        "anchored_hash": "sha256:" + "c" * 64,
        "current_reward_status": "paid",
    }
    source_add = {
        "reward_ref": "source_add_reward:" + "d" * 16,
        "adapter_id": "adapter-rehearsal",
        "miner_hotkey": "5SourceAddRehearsal",
        "leg": 1,
        "reward_kind": "source_acceptance",
        "alpha_percent": 1.0,
        "reward_epochs": 20,
        "start_epoch": 120,
        "current_reward_status": "stopped_forward",
        "trigger_evidence_doc": {
            "submission_id": "source_add_submission:abcd1234abcd1234"
        },
        "public_label": "Source acceptance",
        "desired_alpha_percent": 1.0,
        "epoch_count": 20,
    }
    champion_checkpoint = build_reward_settlement_checkpoint_v2(
        reward_kind="champion",
        source_id=champion["champion_reward_id"],
        obligation_hash=sha256_json(
            champion_reward_row_projection_v2(champion)
        ),
        start_epoch=120,
        epoch_count=20,
        desired_alpha_percent=7.3,
        applied_alpha_percent=30,
        realized_alpha_percent=30,
        excess_alpha_percent=0,
    )
    source_add_checkpoint = build_reward_settlement_checkpoint_v2(
        reward_kind="source_add",
        source_id=source_add["reward_ref"],
        obligation_hash=sha256_json(
            source_add_reward_row_projection_v2(
                "source_add_leg1",
                {**source_add, "initial_reward_status": "active"},
            )
        ),
        start_epoch=120,
        epoch_count=20,
        desired_alpha_percent=1,
        applied_alpha_percent=10,
        realized_alpha_percent=10,
        excess_alpha_percent=0,
    )
    predecessor = build_allocation_settlement_frontier_v2(
        mode="legacy_full_history_bootstrap",
        netuid=71,
        allocation_epoch=120,
        predecessor_frontier_hash=None,
        reward_checkpoints=(champion_checkpoint, source_add_checkpoint),
    )
    resolver = object.__new__(CoordinatorAllocationSourceV2)
    try:
        resolver._build_settlement_frontier(
            epoch=121,
            netuid=71,
            champion_rows=[],
            source_add_rows=[],
            history=[],
            predecessor=predecessor,
        )
    except CoordinatorAllocationSourceV2Error as exc:
        if "unsettled reward disappeared" not in str(exc):
            raise
    else:
        raise RuntimeError("terminal frontier failure was not reproduced")

    rows = {
        "champion_reward_by_id": [champion],
        "source_add_reward_by_ref": [source_add],
    }
    observed_queries: list[str] = []

    def read(policy_id, parameters, _context):
        observed_queries.append(str(policy_id))
        return [dict(row) for row in rows.get(policy_id, [])]

    resolver._read = read
    context = ExecutionContextV2(
        job_id="allocation-v2:terminal-retirement-rehearsal",
        purpose="research_lab.allocation.v2",
        epoch_id=121,
        parent_receipt_hashes=(),
    )
    retirements = resolver._resolve_settlement_frontier_retirements(
        predecessor=predecessor,
        champion_rows=[],
        source_add_rows=[],
        context=context,
    )
    successor = resolver._build_settlement_frontier(
        epoch=121,
        netuid=71,
        champion_rows=[],
        source_add_rows=[],
        history=[],
        predecessor=predecessor,
        terminal_retirements=retirements,
    )
    if (
        successor["reward_checkpoint_count"] != 0
        or observed_queries
        != ["champion_reward_by_id", "source_add_reward_by_ref"]
        or {item["terminal_status"] for item in retirements}
        != {"paid", "stopped_forward"}
    ):
        raise RuntimeError("terminal frontier retirement was not exact")

    rows["champion_reward_by_id"] = [
        {**champion, "input_hash": "sha256:" + "e" * 64}
    ]
    try:
        resolver._resolve_settlement_frontier_retirements(
            predecessor=predecessor,
            champion_rows=[],
            source_add_rows=[],
            context=context,
        )
    except CoordinatorAllocationSourceV2Error as exc:
        if "terminal reward identity changed" not in str(exc):
            raise
    else:
        raise RuntimeError("mutated terminal reward did not fail closed")

    signing_key = Ed25519PrivateKey.generate()
    signing_pubkey = signing_key.public_key().public_bytes_raw().hex()
    boot_body = build_boot_identity_body(
        role="gateway_coordinator",
        physical_role="gateway_coordinator",
        commit_sha="a" * 40,
        pcr0="b" * 96,
        build_manifest_hash="sha256:" + "c" * 64,
        dependency_lock_hash="sha256:" + "d" * 64,
        config_hash="sha256:" + "e" * 64,
        boot_nonce="f" * 32,
        signing_pubkey=signing_pubkey,
        transport_pubkey="1" * 64,
        transport_certificate_hash="sha256:" + "2" * 64,
        attestation_user_data_hash="sha256:" + "3" * 64,
        issued_at=NOW,
    )
    boot_identity = create_boot_identity(
        body=boot_body,
        attestation_document_b64=base64.b64encode(
            b"frontier-release-rehearsal"
        ).decode("ascii"),
    )
    source_state = {"settlement_frontier": predecessor}
    source_state_hash = sha256_json(source_state)
    allocation = {"allocation_hash": "sha256:" + "4" * 64}
    result = {
        "allocation": allocation,
        "source_state": source_state,
        "source_state_hash": source_state_hash,
    }
    artifact_hashes = sorted(
        set(frontier_artifact_hashes_v2(predecessor))
        | {source_state_hash}
    )
    artifact_root = merkle_root(
        artifact_hashes,
        domain="leadpoet-artifact-v2",
    )
    output_root = sha256_json({"allocation": allocation})
    receipt_body = build_execution_receipt_body(
        role="gateway_coordinator",
        purpose="research_lab.allocation.v2",
        job_id="allocation-v2:frontier-release-rehearsal:120",
        epoch_id=120,
        sequence=0,
        commit_sha=boot_identity["commit_sha"],
        pcr0=boot_identity["pcr0"],
        build_manifest_hash=boot_identity["build_manifest_hash"],
        dependency_lock_hash=boot_identity["dependency_lock_hash"],
        config_hash=boot_identity["config_hash"],
        boot_identity_hash=boot_identity["boot_identity_hash"],
        input_root="sha256:" + "5" * 64,
        output_root=output_root,
        transport_root_hash=EMPTY_TRANSPORT_ROOT,
        host_operation_root_hash=EMPTY_HOST_OPERATION_ROOT,
        artifact_root=artifact_root,
        parent_receipt_hashes=(),
        status="succeeded",
        failure_code=None,
        issued_at=NOW,
    )
    receipt = create_signed_execution_receipt(
        body=receipt_body,
        enclave_pubkey=signing_pubkey,
        sign_digest=signing_key.sign,
    )
    execution = {
        "schema_version": "leadpoet.attested_execution_result.v2",
        "receipt_hash": receipt["receipt_hash"],
        "role": "gateway_coordinator",
        "operation": "research_lab_allocation",
        "purpose": "research_lab.allocation.v2",
        "job_id": receipt["job_id"],
        "epoch_id": 120,
        "sequence": 0,
        "release_hash": "sha256:" + "6" * 64,
        "input_root": receipt["input_root"],
        "output_root": output_root,
        "artifact_root": artifact_root,
        "result_hash": sha256_json(result),
        "artifact_hashes": artifact_hashes,
        "result_doc": result,
    }
    frontier_row = {
        "schema_version": predecessor["schema_version"],
        "netuid": 71,
        "allocation_epoch": 120,
        "settled_through_epoch": 119,
        "frontier_hash": predecessor["frontier_hash"],
        "predecessor_frontier_hash": None,
        "source_receipt_hash": receipt["receipt_hash"],
        "source_state_hash": source_state_hash,
        "frontier_doc": predecessor,
    }

    def authority_read(policy_id, parameters, _context):
        if policy_id == "allocation_settlement_frontier_activation":
            return [
                {
                    "schema_version": (
                        "leadpoet.research_lab_allocation_"
                        "settlement_frontier_activation.v2"
                    ),
                    "netuid": 71,
                    "first_allocation_epoch": 120,
                    "first_frontier_hash": predecessor["frontier_hash"],
                    "source_receipt_hash": receipt["receipt_hash"],
                }
            ]
        if policy_id in {
            "allocation_settlement_frontiers",
            "allocation_settlement_frontier_by_epoch",
        }:
            return [dict(frontier_row)]
        if policy_id == "attested_execution_result_by_receipt":
            return [dict(execution)]
        if policy_id == "attested_receipt_by_hash":
            return [
                {
                    "receipt_hash": receipt["receipt_hash"],
                    "receipt_doc": dict(receipt),
                }
            ]
        return []

    resolver._read = authority_read
    authority_context = ExecutionContextV2(
        job_id="allocation-v2:frontier-release-successor:121",
        purpose="research_lab.allocation.v2",
        epoch_id=121,
        parent_receipt_hashes=(receipt["receipt_hash"],),
    )
    authority_context.external_receipt_graphs = [
        build_receipt_graph(
            root_receipt_hash=receipt["receipt_hash"],
            boot_identities=(boot_identity,),
            receipts=(receipt,),
            transport_attempts=(),
        )
    ]
    required_parents: set[str] = set()
    authority = resolver._load_prior_settlement_frontier(
        epoch=121,
        netuid=71,
        context=authority_context,
        required_parents=required_parents,
    )
    if (
        authority
        != {"frontier": predecessor, "receipt_hash": receipt["receipt_hash"]}
        or required_parents != {receipt["receipt_hash"]}
        or "release_hash" in receipt
    ):
        raise RuntimeError(
            "canonical receipt and execution release authority differed"
        )
    execution["release_hash"] = "invalid"
    try:
        resolver._load_prior_settlement_frontier(
            epoch=121,
            netuid=71,
            context=authority_context,
            required_parents=set(),
        )
    except CoordinatorAllocationSourceV2Error as exc:
        if "execution authority differs" not in str(exc):
            raise
    else:
        raise RuntimeError("malformed execution release hash did not fail closed")
    return {
        "original_failure_reproduced": True,
        "champion_terminal_retired": True,
        "source_add_terminal_retired": True,
        "tampered_identity_rejected": True,
        "successor_reward_checkpoint_count": 0,
        "canonical_receipt_without_release_hash_accepted": True,
        "execution_release_hash_validated": True,
    }


def _exercise_current_frontier_release_recovery() -> dict[str, Any]:
    """Prove a release transition reuses one immutable epoch authority."""

    from gateway.research_lab import attested_v2_store, v2_authority
    from leadpoet_canonical.allocation_settlement_frontier_v2 import (
        build_allocation_settlement_frontier_v2,
        frontier_artifact_hashes_v2,
    )
    from leadpoet_canonical.allocation_handoff_v2 import (
        build_allocation_handoff_v2,
        validate_allocation_handoff_v2,
    )
    from leadpoet_canonical.attested_v2 import (
        build_boot_identity_body,
        create_boot_identity,
    )

    epoch = 24321
    frontier = build_allocation_settlement_frontier_v2(
        mode="legacy_full_history_bootstrap",
        netuid=71,
        allocation_epoch=epoch,
        predecessor_frontier_hash=None,
        reward_checkpoints=(),
    )
    policy: dict[str, Any] = {}
    source_state = {
        "epoch": epoch,
        "netuid": 71,
        "policy": policy,
        "reimbursement_obligations": [],
        "champion_obligations": [],
        "settlement_frontier": frontier,
    }
    allocation_inputs = {
        "epoch": epoch,
        "policy": policy,
        "active_reimbursement_obligations": [],
        "active_champion_obligations": [],
    }
    allocation = {
        "epoch": epoch,
        "netuid": 71,
        "allocation_hash": sha256_json({"epoch": epoch, "netuid": 71}),
    }
    source_state_hash = sha256_json(source_state)
    result = {
        "allocation": allocation,
        "allocation_inputs": allocation_inputs,
        "source_state": source_state,
        "source_state_hash": source_state_hash,
    }
    artifact_hashes = sorted(
        set(frontier_artifact_hashes_v2(frontier)) | {source_state_hash}
    )
    signing_key = Ed25519PrivateKey.generate()
    signing_pubkey = signing_key.public_key().public_bytes_raw().hex()
    source_commit = "1" * 40
    boot_body = build_boot_identity_body(
        role="gateway_coordinator",
        physical_role="gateway_coordinator",
        commit_sha=source_commit,
        pcr0="2" * 96,
        build_manifest_hash="sha256:" + "3" * 64,
        dependency_lock_hash="sha256:" + "4" * 64,
        config_hash="sha256:" + "5" * 64,
        boot_nonce="6" * 32,
        signing_pubkey=signing_pubkey,
        transport_pubkey="7" * 64,
        transport_certificate_hash="sha256:" + "8" * 64,
        attestation_user_data_hash="sha256:" + "9" * 64,
        issued_at=NOW,
    )
    boot = create_boot_identity(
        body=boot_body,
        attestation_document_b64=base64.b64encode(
            b"current-frontier-release-recovery"
        ).decode("ascii"),
    )
    receipt_body = build_execution_receipt_body(
        role="gateway_coordinator",
        purpose="research_lab.allocation.v2",
        job_id="allocation-v2:prior-release:24321",
        epoch_id=epoch,
        sequence=0,
        commit_sha=source_commit,
        pcr0=boot["pcr0"],
        build_manifest_hash=boot["build_manifest_hash"],
        dependency_lock_hash=boot["dependency_lock_hash"],
        config_hash=boot["config_hash"],
        boot_identity_hash=boot["boot_identity_hash"],
        input_root=sha256_json({"epoch": epoch, "netuid": 71}),
        output_root=sha256_json({"allocation": allocation}),
        transport_root_hash=EMPTY_TRANSPORT_ROOT,
        host_operation_root_hash=EMPTY_HOST_OPERATION_ROOT,
        artifact_root=merkle_root(
            artifact_hashes,
            domain="leadpoet-artifact-v2",
        ),
        parent_receipt_hashes=(),
        status="succeeded",
        failure_code=None,
        issued_at=NOW,
    )
    receipt = create_signed_execution_receipt(
        body=receipt_body,
        enclave_pubkey=signing_pubkey,
        sign_digest=signing_key.sign,
    )
    graph = build_receipt_graph(
        root_receipt_hash=receipt["receipt_hash"],
        boot_identities=(boot,),
        receipts=(receipt,),
        transport_attempts=(),
    )
    context = {
        "frontier": frontier,
        "row": {
            "source_receipt_hash": receipt["receipt_hash"],
        },
        "source": {
            "row": {
                "receipt_hash": receipt["receipt_hash"],
                "operation": v2_authority.OP_RESEARCH_LAB_ALLOCATION,
                "purpose": "research_lab.allocation.v2",
                "role": "gateway_coordinator",
                "epoch_id": epoch,
                "release_hash": "sha256:" + "a" * 64,
            },
            "result": result,
            "receipt": receipt,
            "receipt_graph": graph,
            "artifact_hashes": artifact_hashes,
        },
    }
    execute_calls = 0

    async def load_context(**kwargs):
        if kwargs != {"netuid": 71, "before_epoch": epoch + 1}:
            raise RuntimeError("current frontier lookup scope changed")
        return context

    async def execute(**_kwargs):
        nonlocal execute_calls
        execute_calls += 1
        raise RuntimeError("current frontier was re-executed under a new release")

    async def persist_links(**kwargs):
        if kwargs.get("receipt_hash") != receipt["receipt_hash"]:
            raise RuntimeError("current frontier business link changed authority")
        return {"business_artifact_link_count": 1}

    original_loader = (
        attested_v2_store.load_allocation_settlement_frontier_context_v2
    )
    attested_v2_store.load_allocation_settlement_frontier_context_v2 = (
        load_context
    )
    try:
        recovered = asyncio.run(
            v2_authority.build_allocation_v2(
                epoch_id=epoch,
                netuid=71,
                policy=policy,
                execute=execute,
                persist_links=persist_links,
            )
        )
        if (
            execute_calls != 0
            or recovered.get("result") != result
            or recovered.get("receipt") != receipt
            or recovered.get("receipt_graph") != graph
            or recovered.get("replay_status")
            != "durable_current_frontier"
        ):
            raise RuntimeError("current frontier release recovery differed")
        handoff = build_allocation_handoff_v2(
            bundle={
                "epoch": epoch,
                "netuid": 71,
                "allocation_doc": allocation,
            },
            receipt_graph=recovered["receipt_graph"],
            lineage_bindings=recovered["lineage_bindings"],
            lineage_complete=recovered["lineage_complete"],
            persistence=recovered["persistence"],
        )
        validate_allocation_handoff_v2(
            handoff,
            expected_epoch_id=epoch,
            expected_netuid=71,
        )

        context["source"]["row"]["release_hash"] = "invalid"
        try:
            asyncio.run(
                v2_authority.build_allocation_v2(
                    epoch_id=epoch,
                    netuid=71,
                    policy=policy,
                    execute=execute,
                    persist_links=persist_links,
                )
            )
        except v2_authority.ResearchLabV2AuthorityError as exc:
            if "source authority differs" not in str(exc):
                raise
        else:
            raise RuntimeError("malformed current frontier release was accepted")
    finally:
        attested_v2_store.load_allocation_settlement_frontier_context_v2 = (
            original_loader
        )

    return {
        "cross_release_execution_skipped": True,
        "exact_signed_authority_reused": True,
        "immutable_frontier_preserved": True,
        "canonical_handoff_verified": True,
        "malformed_release_rejected": True,
    }


def _exercise_validator_publication_release_recovery() -> dict[str, Any]:
    """Prove an approved N-1 validator journal survives N activation."""

    import subprocess
    import neurons.validator as validator_module

    from leadpoet_canonical.attested_v2 import sha256_json
    from validator_tee.enclave.hotkey_authority_v2 import (
        ValidatorHotkeyAuthorityV2,
        ValidatorHotkeyAuthorityV2Error,
        load_chain_signing_profile,
    )

    candidate_sha = subprocess.run(
        ["git", "-C", str(SOURCE_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    from_sha = str(os.environ.get("REHEARSAL_FROM_SHA") or "").lower()
    if (
        len(from_sha) != 40
        or any(value not in "0123456789abcdef" for value in from_sha)
        or from_sha == candidate_sha
    ):
        raise RuntimeError(
            "validator publication recovery requires a distinct N-1 release"
        )
    current = SanitizedWeightFixture(candidate_sha=candidate_sha, epoch_id=30_000)
    previous = SanitizedWeightFixture(candidate_sha=from_sha, epoch_id=30_000)
    current_boot = current._boot(
        role="validator_weights",
        key=current.weight_key,
        config_hash=sha256_json({"release": candidate_sha}),
    )
    old_boot = previous._boot(
        role="validator_weights",
        key=previous.weight_key,
        config_hash=sha256_json({"release": from_sha}),
    )
    expectation_fields = (
        "commit_sha",
        "pcr0",
        "build_manifest_hash",
        "dependency_lock_hash",
    )
    lineage = {
        from_sha: {
            "roles": {
                "validator_weights": {
                    field: old_boot[field] for field in expectation_fields
                }
            }
        }
    }
    verified = []

    def verify_boot(identity, **kwargs):
        if identity.get("pcr0") != kwargs.get("expected_pcr0"):
            raise RuntimeError("recovery boot PCR0 differs")
        if kwargs.get("certificate_validity_at_attestation_time") is not True:
            raise RuntimeError("historical attestation time was not enforced")
        verified.append(str(identity["boot_identity_hash"]))
        return {"verified": True}

    class _UnusedSr25519:
        pass

    authority = ValidatorHotkeyAuthorityV2(
        boot_identity_supplier=lambda: current_boot,
        gateway_release_lineage_supplier=lambda: lineage,
        validator_hotkey=VALIDATOR_HOTKEY,
        hotkey_public_key_hex="1" * 64,
        chain_profile=load_chain_signing_profile(
            SOURCE_ROOT
            / "validator_tee/enclave/chain_signing_profile_v2.json"
        ),
        sign_receipt_digest=current.weight_key.sign,
        attestation_supplier=lambda **_kwargs: b"unused",
        drand_backend=object(),
        sr25519_backend=_UnusedSr25519(),
        boot_verifier=verify_boot,
    )
    authority._verify_recovery_validator_boot(old_boot)
    if verified != [old_boot["boot_identity_hash"]]:
        raise RuntimeError("approved N-1 validator boot was not attested")

    rejected = 0
    for field, value in (
        ("pcr0", "0" * 96),
        ("build_manifest_hash", "sha256:" + "0" * 64),
        ("dependency_lock_hash", "sha256:" + "0" * 64),
    ):
        try:
            authority._verify_recovery_validator_boot(
                {**old_boot, field: value}
            )
        except ValidatorHotkeyAuthorityV2Error:
            rejected += 1
    try:
        authority._verify_recovery_validator_boot(
            {**old_boot, "commit_sha": "0" * 40}
        )
    except ValidatorHotkeyAuthorityV2Error:
        rejected += 1
    try:
        authority._verify_recovery_validator_boot(
            {
                **current_boot,
                "config_hash": "sha256:" + "0" * 64,
            }
        )
    except ValidatorHotkeyAuthorityV2Error:
        rejected += 1
    if rejected != 5:
        raise RuntimeError("validator recovery release tampering was accepted")
    if authority._recovery_finalization_only_mode(
        old_boot=current_boot,
        extrinsic_signature_results=[],
        allow_cross_release_finalization_only=False,
    ):
        raise RuntimeError("same-release recovery became finalization-only")
    finalization_only = authority._recovery_finalization_only_mode(
        old_boot=old_boot,
        extrinsic_signature_results=[{"durable_signed_extrinsic": True}],
        allow_cross_release_finalization_only=True,
    )
    finalization_mode_rejections = 0
    for signatures, allowed in (
        ([{"durable_signed_extrinsic": True}], False),
        ([], True),
    ):
        try:
            authority._recovery_finalization_only_mode(
                old_boot=old_boot,
                extrinsic_signature_results=signatures,
                allow_cross_release_finalization_only=allowed,
            )
        except ValidatorHotkeyAuthorityV2Error:
            finalization_mode_rejections += 1
    if not finalization_only or finalization_mode_rejections != 2:
        raise RuntimeError(
            "cross-release recovery was not constrained to signed finalization"
        )

    event_hash = "sha256:" + "7" * 64
    authorization_id = "sha256:" + "8" * 64

    class _RecoveryJournal:
        def __init__(self) -> None:
            self.record = {
                "weight_authorization_id": authorization_id,
                "published_bundle": {
                    "weight_result": {"epoch_id": current.epoch_id}
                },
                "publication": {
                    "weight_submission_event_hash": event_hash
                },
                "extrinsic_signature_results": [
                    {"durable_signed_extrinsic": True}
                ],
            }
            self.scan = 0
            self.cleared = False

        def load(self):
            return self.record

        def replace_authorization(self, value):
            self.record = {**self.record, "weight_authorization_id": value}
            return self.record

        def reserve_finalization_scan(self):
            self.scan += 1
            return "sha256:" + format(self.scan, "064x")

        def clear(self, *, expected_event_hash):
            if expected_event_hash != event_hash:
                raise RuntimeError("rehearsal cleared another publication")
            self.record = None
            self.cleared = True

    class _RecoveryClient:
        def recover_weight_publication_v2(self, **_kwargs):
            return {
                "weight_authorization_id": authorization_id,
                "signed_extrinsics": [
                    {
                        "authorization_hash": "sha256:" + "9" * 64,
                        "extrinsic_hash": "0x" + "a" * 64,
                        "extrinsic_hex": "00",
                    }
                ],
                "finalization_only": True,
            }

        def confirm_weight_publication_v2(
            self, _authorization_id, *, finalization_scan_id
        ):
            if not str(finalization_scan_id).startswith("sha256:"):
                raise RuntimeError("finalization scan identity is invalid")
            return {"finalized": True}

    journal = _RecoveryJournal()
    validator = validator_module.Validator.__new__(validator_module.Validator)
    validator._weight_publication_journal_v2 = journal
    validator._validator_v2_client = _RecoveryClient()
    validator.wallet = SimpleNamespace(
        hotkey=SimpleNamespace(ss58_address=VALIDATOR_HOTKEY)
    )
    active_epoch = current.epoch_id

    async def epoch_state():
        return SimpleNamespace(workflow_epoch_id=active_epoch)

    validator._get_epoch_state_async = epoch_state
    validator._get_best_epoch_state_async = epoch_state
    original_finalize = (
        validator_module.finalize_authoritative_weight_publication_v2
    )

    async def finalize(**_kwargs):
        return {
            "acknowledgment": {
                "weight_finalization_event_hash": "sha256:" + "b" * 64
            }
        }

    validator_module.finalize_authoritative_weight_publication_v2 = finalize
    try:
        same_epoch = asyncio.run(
            validator._recover_weight_publication_before_new_authority_v2(
                epoch_id=current.epoch_id,
                gateway_url="https://gateway.rehearsal.invalid",
            )
        )
        if not same_epoch or journal.record is None or journal.cleared:
            raise RuntimeError(
                "same-epoch finalized publication did not survive restart"
            )
        active_epoch = current.epoch_id + 1
        next_epoch = asyncio.run(
            validator._recover_weight_publication_before_new_authority_v2(
                epoch_id=active_epoch,
                gateway_url="https://gateway.rehearsal.invalid",
            )
        )
        if next_epoch or journal.record is not None or not journal.cleared:
            raise RuntimeError(
                "revalidated prior publication blocked the next epoch"
            )
    finally:
        validator_module.finalize_authoritative_weight_publication_v2 = (
            original_finalize
        )
    return {
        "approved_n_minus_one_recovered": True,
        "nitro_attestation_rechecked": True,
        "release_tampering_rejected": True,
        "same_release_config_mismatch_rejected": True,
        "cross_release_finalization_only": True,
        "unsigned_cross_release_rejected": True,
        "implicit_cross_release_rejected": True,
        "same_epoch_finalized_journal_retained": True,
        "next_epoch_finalized_journal_retired": True,
    }


def _exercise_receipt_graph_aggregate_pagination() -> dict[str, Any]:
    """Exercise aggregate evidence paging through the candidate store helper."""

    from gateway.research_lab import attested_v2_store

    row_limit = int(attested_v2_store._MAX_GRAPH_ROWS)
    query_chunk = int(attested_v2_store._GRAPH_QUERY_CHUNK)
    if row_limit < 1 or query_chunk < 1 or query_chunk > row_limit:
        raise RuntimeError("candidate V2 receipt graph limits are invalid")

    row_count = row_limit + 1
    width = len(str(row_count))
    expected_rows = [
        {
            "attempt_hash": (
                f"rehearsal-aggregate-attempt-{index:0{width}d}"
            )
        }
        for index in range(row_count)
    ]
    expected_by_key = {
        str(row["attempt_hash"]): dict(row) for row in expected_rows
    }
    expected_keys = set(expected_by_key)
    observed_queries: list[dict[str, Any]] = []
    original_select_all = attested_v2_store.select_all

    async def strict_select_all(
        table: str,
        *,
        filters: tuple[tuple[str, str, Any], ...],
        order_by: tuple[tuple[str, bool], ...],
        max_rows: int,
        **_kwargs: Any,
    ) -> list[dict[str, Any]]:
        if (
            table != attested_v2_store.TRANSPORT_TABLE
            or len(filters) != 1
            or filters[0][0] != "attempt_hash"
            or filters[0][1] != "in"
            or order_by != (("attempt_hash", False),)
            or int(max_rows) != row_limit
        ):
            raise RuntimeError(
                "receipt graph rehearsal received an unknown store operation"
            )
        values = [str(value) for value in filters[0][2]]
        if not values or len(values) > query_chunk:
            raise RuntimeError(
                "receipt graph rehearsal query exceeded candidate chunk limit"
            )
        unknown = sorted(set(values) - expected_keys)
        if unknown:
            raise RuntimeError(
                "receipt graph rehearsal queried undeclared evidence"
            )
        observed_queries.append(
            {
                "count": len(values),
                "first": values[0],
                "last": values[-1],
            }
        )
        return [dict(expected_by_key[value]) for value in values]

    async def exercise() -> tuple[set[str], bool]:
        attested_v2_store.select_all = strict_select_all
        try:
            existing = await attested_v2_store._existing_exact_rows(
                attested_v2_store.TRANSPORT_TABLE,
                key_field="attempt_hash",
                expected_rows=expected_rows,
            )
            try:
                await attested_v2_store._select_by_values(
                    attested_v2_store.RECEIPT_TABLE,
                    field="receipt_hash",
                    values=(
                        f"rehearsal-receipt-{index:0{width}d}"
                        for index in range(row_count)
                    ),
                    key_fields=("receipt_hash",),
                )
            except attested_v2_store.AttestedV2StoreError as exc:
                if str(exc) != "V2 receipt graph exceeds row limit":
                    raise
                structural_limit_enforced = True
            else:
                structural_limit_enforced = False
            return existing, structural_limit_enforced
        finally:
            attested_v2_store.select_all = original_select_all

    existing, structural_limit_enforced = asyncio.run(exercise())
    if existing != expected_keys:
        raise RuntimeError("aggregate V2 receipt evidence was not exact")
    if len(observed_queries) < 2:
        raise RuntimeError("aggregate V2 receipt evidence was not paged")
    if (
        max(int(query["count"]) for query in observed_queries) > query_chunk
        or not structural_limit_enforced
    ):
        raise RuntimeError("V2 receipt graph safety bounds were weakened")
    parent_hash = "sha256:" + "1" * 64
    child_hash = "sha256:" + "2" * 64
    checkpoint_delta = {
        "receipts": [
            {
                "receipt_hash": child_hash,
                "parent_receipt_hashes": [parent_hash],
            },
            {
                "receipt_hash": parent_hash,
                "parent_receipt_hashes": [],
            },
        ]
    }
    parent_first = attested_v2_store._parent_first_receipt_hashes_v2(
        checkpoint_delta,
        validated_receipts=(child_hash, parent_hash),
    )
    if parent_first != (parent_hash, child_hash):
        raise RuntimeError(
            "checkpoint receipt membership was used as insertion order"
        )
    return {
        "aggregate_rows": row_count,
        "aggregate_evidence_paged": True,
        "checkpoint_parent_first_persistence": True,
        "per_query_row_limit": row_limit,
        "query_chunk": query_chunk,
        "query_count": len(observed_queries),
        "structural_limit_enforced": True,
    }


def _exercise_receipt_graph_transport_deduplication() -> dict[str, Any]:
    """Run shared ancestry through the exact job admission and decode path."""

    import subprocess

    from gateway.tee.execution_job_manager_v2 import (
        JOB_SCHEMA_VERSION,
        MAX_ALLOCATION_ANCESTRY_AUTHORITIES,
        MAX_ALLOCATION_ANCESTRY_INPUT_BYTES,
        MAX_EXTERNAL_RECEIPT_GRAPHS,
        MAX_INPUT_BYTES,
        PARENT_RECEIPT_GRAPH_SET_FIELD,
        ExecutionJobManagerV2,
        ExecutionJobV2Error,
        pack_parent_receipt_graph_set_v2,
        unpack_parent_receipt_graph_set_v2,
    )
    from gateway.tee.release_lineage_v2 import _required_commits
    from gateway.research_lab.attested_scoring_v2 import (
        _build_transport_payload_document,
    )
    from gateway.tee.coordinator_allocation_source_v2 import (
        _receipt_graphs_by_declared_root,
    )
    from leadpoet_canonical.attested_v2 import (
        CHECKPOINTED_RECEIPT_GRAPH_SCHEMA_VERSION,
        RECEIPT_GRAPH_SCHEMA_VERSION,
        build_checkpointed_receipt_graph,
        sha256_bytes,
    )
    from leadpoet_canonical.ancestry_checkpoint_v2 import (
        ANCESTRY_DELTA_SCHEMA_VERSION,
        build_compact_ancestry_proof_from_delta_v2,
        issue_ancestry_certificate_v2,
    )

    candidate_sha = subprocess.run(
        ["git", "-C", str(SOURCE_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    from_sha = str(os.environ.get("REHEARSAL_FROM_SHA") or "").lower()
    if (
        len(from_sha) != 40
        or any(value not in "0123456789abcdef" for value in from_sha)
        or from_sha == candidate_sha
    ):
        raise RuntimeError(
            "receipt ancestry rehearsal requires a distinct N-1 release"
        )
    fixture = SanitizedWeightFixture(
        candidate_sha=candidate_sha,
        epoch_id=30_000,
    )
    historical_fixture = SanitizedWeightFixture(
        candidate_sha=from_sha,
        epoch_id=30_000,
    )
    config_hash = sha256_json({"rehearsal": "shared-receipt-ancestry"})
    boot = fixture._boot(
        role="gateway_coordinator",
        key=fixture.coordinator_key,
        config_hash=config_hash,
    )
    historical_boot = historical_fixture._boot(
        role="gateway_coordinator",
        key=historical_fixture.coordinator_key,
        config_hash=config_hash,
    )
    sample_receipt = fixture.receipt(
        role="gateway_coordinator",
        purpose="research_lab.allocation.v2",
        job_id="rehearsal-shared-ancestry-sample",
        key=fixture.coordinator_key,
        boot=boot,
        config_hash=config_hash,
        sequence=0,
    )
    sample_receipt_bytes = len(_canonical(sample_receipt))
    graph_count = MAX_ALLOCATION_ANCESTRY_AUTHORITIES - 1
    legacy_reproduction_target = MAX_INPUT_BYTES * 2 + MAX_INPUT_BYTES // 32
    shared_receipt_count = (
        legacy_reproduction_target
        + graph_count * sample_receipt_bytes
        - 1
    ) // (graph_count * sample_receipt_bytes)
    shared_receipt_count += 8

    shared_receipts: list[dict[str, Any]] = []
    parents: list[str] = []
    for index in range(shared_receipt_count):
        receipt = fixture.receipt(
            role="gateway_coordinator",
            purpose="research_lab.allocation.v2",
            job_id=f"rehearsal-shared-ancestry-{index}",
            key=fixture.coordinator_key,
            boot=boot,
            config_hash=config_hash,
            parents=parents,
            sequence=index,
        )
        shared_receipts.append(receipt)
        parents = [str(receipt["receipt_hash"])]

    checkpoint_graph_count = 2
    graphs: list[dict[str, Any]] = []
    for index in range(graph_count - checkpoint_graph_count):
        child = fixture.receipt(
            role="gateway_coordinator",
            purpose="research_lab.allocation.v2",
            job_id=f"rehearsal-independent-root-{index}",
            key=fixture.coordinator_key,
            boot=boot,
            config_hash=config_hash,
            parents=parents,
            sequence=100 + index,
        )
        graph = {
            "schema_version": RECEIPT_GRAPH_SCHEMA_VERSION,
            "root_receipt_hash": str(child["receipt_hash"]),
            "boot_identities": [boot],
            "receipts": [*shared_receipts, child],
            "transport_attempts": [],
            "host_operations": [],
        }
        graphs.append(graph)

    lineage_id = sha256_json({"rehearsal": "mixed-allocation-frontier"})

    def verify_boot(identity):
        return identity

    for index in range(checkpoint_graph_count):
        receipt = fixture.receipt(
            role="gateway_coordinator",
            purpose="research_lab.allocation.v2",
            job_id=f"rehearsal-checkpointed-root-{index}",
            key=fixture.coordinator_key,
            boot=boot,
            config_hash=config_hash,
            parents=(),
            sequence=graph_count + index,
        )
        delta = {
            "schema_version": ANCESTRY_DELTA_SCHEMA_VERSION,
            "root_receipt_hash": receipt["receipt_hash"],
            "boot_identities": [boot],
            "receipts": [receipt],
            "transport_attempts": [],
            "host_operations": [],
        }
        issuer_boot = historical_boot if index == 0 else boot
        issuer_key = (
            historical_fixture.coordinator_key
            if index == 0
            else fixture.coordinator_key
        )
        certificate = issue_ancestry_certificate_v2(
            local_delta=delta,
            lineage_id=lineage_id,
            certificate_sequence=0,
            issuer_boot_identity=issuer_boot,
            issued_at="2026-07-10T20:00:00Z",
            sign_digest=issuer_key.sign,
            boot_attestation_verifier=verify_boot,
            allowed_issuer_roles=("gateway_coordinator",),
            required_purposes=("research_lab.allocation.v2",),
        )
        proof = build_compact_ancestry_proof_from_delta_v2(
            delta,
            certificate,
            expected_lineage_id=lineage_id,
            boot_attestation_verifier=verify_boot,
            allowed_issuer_roles=("gateway_coordinator",),
        )
        graphs.append(
            build_checkpointed_receipt_graph(
                root_receipt_hash=receipt["receipt_hash"],
                boot_identities=(boot,),
                receipts=(receipt,),
                transport_attempts=(),
                host_operations=(),
                ancestry_lineage_id=lineage_id,
                ancestry_proof=proof,
                boot_attestation_verifier=verify_boot,
                require_boot_attestation_verification=True,
            )
        )
    if sum(
        graph.get("schema_version")
        == CHECKPOINTED_RECEIPT_GRAPH_SCHEMA_VERSION
        for graph in graphs
    ) != checkpoint_graph_count:
        raise RuntimeError("mixed checkpoint ancestry fixture is incomplete")
    required_release_commits = _required_commits(tuple(graphs))
    if required_release_commits != {candidate_sha, from_sha}:
        raise RuntimeError(
            "checkpoint issuer N-1 release was omitted from ancestry lineage"
        )

    try:
        pack_parent_receipt_graph_set_v2(graphs)
    except ExecutionJobV2Error as exc:
        if "external receipt graph count exceeds limit" not in str(exc):
            raise
    else:
        raise RuntimeError("ordinary V2 ancestry accepted allocation frontier")

    transport_document, transport_metadata = _build_transport_payload_document(
        payload={"epoch": 30_000},
        parent_graphs=graphs,
        max_parent_graph_count=MAX_ALLOCATION_ANCESTRY_AUTHORITIES,
    )
    if transport_metadata.get("encoding") != "receipt_graph_set":
        raise RuntimeError("oversized shared ancestry was not deduplicated")
    legacy_size_bytes = int(transport_metadata["legacy_size_bytes"])
    if legacy_size_bytes <= MAX_INPUT_BYTES * 2:
        raise RuntimeError("legacy payload did not reproduce the old boundary")
    graph_set = transport_document[PARENT_RECEIPT_GRAPH_SET_FIELD]
    reconstructed = unpack_parent_receipt_graph_set_v2(
        graph_set,
        max_graph_count=MAX_ALLOCATION_ANCESTRY_AUTHORITIES,
    )
    if reconstructed != graphs:
        raise RuntimeError("deduplicated receipt graph membership differs")
    del reconstructed
    transport_payload = _canonical(transport_document)
    if len(transport_payload) >= legacy_size_bytes:
        raise RuntimeError("shared receipt ancestry was not deduplicated")
    projected_transport_bytes = (
        MAX_ALLOCATION_ANCESTRY_INPUT_BYTES * len(transport_payload)
        + legacy_size_bytes
        - 1
    ) // legacy_size_bytes
    if projected_transport_bytes > MAX_INPUT_BYTES:
        raise RuntimeError(
            "candidate graph-set ratio lacks ordinary-input headroom"
        )

    observed: dict[str, Any] = {}

    def executor(_operation, payload, context):
        observed["payload"] = dict(payload)
        observed["graphs"] = list(context.external_receipt_graphs)
        observed["derived_graphs"] = _receipt_graphs_by_declared_root(
            context.external_receipt_graphs,
            context.parent_receipt_hashes,
        )
        return {"status": "verified"}

    manager = ExecutionJobManagerV2(
        boot_identity_supplier=lambda: boot,
        sign_digest=fixture.coordinator_key.sign,
        operations={
            "research_lab_allocation": {"research_lab.allocation.v2"}
        },
        executor=executor,
        worker_count=1,
        configured_worker_count=0,
        ancestry_lineage_id=lineage_id,
        ancestry_boot_attestation_verifier=verify_boot,
        ancestry_allowed_issuer_roles=("gateway_coordinator",),
    )
    manifest = {
        "schema_version": JOB_SCHEMA_VERSION,
        "job_id": "rehearsal-shared-ancestry-job",
        "operation": "research_lab_allocation",
        "purpose": "research_lab.allocation.v2",
        "epoch_id": 30_000,
        "sequence": 0,
        "payload_sha256": sha256_bytes(transport_payload),
        "payload_size_bytes": len(transport_payload),
        "parent_receipt_hashes": [
            str(graph["root_receipt_hash"]) for graph in graphs
        ],
        "input_artifact_hashes": [],
        "provider_credential_profile": "default",
        "provider_credential_ref_hashes": {},
    }
    manager.submit(manifest)
    for offset in range(0, len(transport_payload), 512 * 1024):
        chunk = transport_payload[offset : offset + 512 * 1024]
        manager.put_chunk(
            job_id=manifest["job_id"],
            offset=offset,
            data_b64=base64.b64encode(chunk).decode("ascii"),
            chunk_sha256=sha256_bytes(chunk),
        )
    manager.seal(manifest["job_id"])
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        status = manager.status(manifest["job_id"])
        if status["state"] in {"succeeded", "failed", "cancelled"}:
            break
        time.sleep(0.01)
    else:
        raise RuntimeError("deduplicated receipt graph job did not terminate")
    expected_graphs_by_root = {
        str(graph["root_receipt_hash"]): graph for graph in graphs
    }
    if (
        status["state"] != "succeeded"
        or observed.get("payload") != {"epoch": 30_000}
        or observed.get("graphs") != graphs
        or observed.get("derived_graphs") != expected_graphs_by_root
    ):
        raise RuntimeError("deduplicated receipt graph job was not exact")
    observed_graphs = observed["graphs"]
    derived_graphs = observed["derived_graphs"]
    first_shared_root = str(graphs[0]["root_receipt_hash"])
    second_shared_root = str(graphs[1]["root_receipt_hash"])
    if (
        observed_graphs[0]["boot_identities"][0]
        is not observed_graphs[1]["boot_identities"][0]
        or observed_graphs[0]["receipts"][0]
        is not observed_graphs[1]["receipts"][0]
        or derived_graphs[first_shared_root]["receipts"][0]
        is not derived_graphs[second_shared_root]["receipts"][0]
    ):
        raise RuntimeError("shared receipt graph evidence was rematerialized")

    malformed = json.loads(json.dumps(graph_set))
    malformed["receipts"].append(
        {
            **dict(malformed["receipts"][0]),
            "receipt_hash": sha256_json({"unreferenced": True}),
        }
    )
    try:
        unpack_parent_receipt_graph_set_v2(
            malformed,
            max_graph_count=MAX_ALLOCATION_ANCESTRY_AUTHORITIES,
        )
    except Exception as exc:
        if "unreferenced evidence" not in str(exc):
            raise
    else:
        raise RuntimeError("unreferenced graph-set evidence did not fail closed")

    return {
        "graph_count": len(graphs),
        "shared_receipt_count": len(shared_receipts),
        "legacy_size_bytes": legacy_size_bytes,
        "transport_size_bytes": len(transport_payload),
        "projected_transport_bytes_at_scoped_limit": (
            projected_transport_bytes
        ),
        "unique_receipt_count": len(graph_set["receipts"]),
        "exact_job_path_verified": True,
        "allocation_source_path_verified": True,
        "shared_object_identity_preserved": True,
        "malformed_evidence_rejected": True,
        "ordinary_graph_bound_preserved": True,
        "checkpointed_graph_count": checkpoint_graph_count,
        "checkpoint_authority_preserved": True,
        "checkpoint_release_commits": sorted(required_release_commits),
        "historical_checkpoint_issuer_included": True,
    }


def _exercise_model_sandbox_scope_binding() -> dict[str, Any]:
    import socket
    import subprocess

    from Leadpoet.utils.subnet_epoch import ensure_cutover_manifest_configured
    from gateway.research_lab.config import ResearchLabGatewayConfig
    from gateway.research_lab.model_authority_v2 import (
        _measured_environment,
        _measured_environment_for_provider_cost_scope,
    )
    from gateway.research_lab.scoring_worker import ResearchLabGatewayScoringWorker
    from gateway.tee.research_lab_runtime_config_v2 import (
        ResearchLabRuntimeConfigV2Error,
        build_research_lab_execution_config,
        validate_model_sandbox_environment,
    )
    from gateway.tee.model_sandbox_v2 import (
        MODEL_SANDBOX_BROKER_DIRECTORY,
        MODEL_SANDBOX_CGROUP_V1_CONTROL_FILES,
        MODEL_SANDBOX_REQUIRED_CONTROLLERS,
        MODEL_SANDBOX_SOURCE_DIRECTORY,
        MODEL_SANDBOX_VISIBLE_ROOT,
        RunscSandboxConfigV2,
        RunscModelSandboxV2,
        _oci_config,
        _runsc_run_command,
        model_sandbox_job_cgroup_path,
        model_source_import_bootstrap,
        prepare_model_sandbox_cgroup_v2,
    )
    from gateway.tee.provider_client_v2 import (
        BrokeredProviderTransportV2,
        ProviderClientV2Error,
    )
    from gateway.tee.source_add_runtime_v2 import (
        build_source_add_runtime_catalog_v2,
    )
    from research_lab.eval.private_runtime import (
        DockerPrivateModelSpec,
        PROVIDER_COST_EVALUATION_SCOPE_ENV,
    )

    worker = object.__new__(ResearchLabGatewayScoringWorker)
    worker.config = ResearchLabGatewayConfig.from_env()
    worker.proxy_url = None
    worker.worker_ref = "restart-rehearsal-scoring-worker"
    preliminary = worker._with_provider_cost_evaluation_scope(
        worker._private_baseline_scoring_env(),
        run_type="private_baseline_rebenchmark",
        rolling_window_hash=sha256_json({"rehearsal": "rolling-window"}),
        artifact_hash=sha256_json({"rehearsal": "model-artifact"}),
        benchmark_date="2026-07-10",
        benchmark_attempt=1,
        evaluation_epoch=30_000,
        started_at=1.0,
    )
    spec = DockerPrivateModelSpec(
        image_digest=(
            "123456789012.dkr.ecr.us-east-1.amazonaws.com/private@sha256:"
            + "7" * 64
        ),
        env_passthrough=worker._private_model_env_passthrough(),
        extra_env=preliminary,
    )
    final_scope = sha256_json({"rehearsal": "final-provider-cost-job"})
    execution_environment = dict(os.environ)
    ensure_cutover_manifest_configured(execution_environment)
    execution_config = build_research_lab_execution_config(
        config=worker.config,
        environment=execution_environment,
    )
    try:
        validate_model_sandbox_environment(
            execution_config,
            _measured_environment(spec),
            provider_cost_scope=final_scope,
        )
    except ResearchLabRuntimeConfigV2Error:
        pass
    else:
        raise RuntimeError("preliminary provider-cost scope did not fail closed")
    measured = _measured_environment_for_provider_cost_scope(
        spec,
        provider_cost_scope=final_scope,
    )
    validated = validate_model_sandbox_environment(
        execution_config,
        measured,
        provider_cost_scope=final_scope,
    )
    if validated.get(PROVIDER_COST_EVALUATION_SCOPE_ENV) != final_scope:
        raise RuntimeError("final provider-cost scope was not bound to model sandbox")

    transport = BrokeredProviderTransportV2(lambda _request: {})
    retry_policy_hashes = {"public_web": "sha256:" + "4" * 64}
    runtime_catalog = build_source_add_runtime_catalog_v2([])
    measured_scope = RunscModelSandboxV2._create_provider_scope_v2(
        transport,
        job_id="rehearsal-model-signed-failure",
        purpose="research_lab.private_model_run.v2",
        retry_policy_hashes=retry_policy_hashes,
        terminal_sink=lambda _attempt: None,
        artifact_sink=lambda _artifact: None,
        dynamic_provider_catalog=runtime_catalog,
    )
    measured_scope.record_intent("rehearsal-model-operation", 0)
    measured_scope.record_terminal(
        "rehearsal-model-operation",
        0,
        "transport_failure",
    )
    measured_scope.assert_accepted_result_is_complete()
    incomplete_scope = RunscModelSandboxV2._create_provider_scope_v2(
        transport,
        job_id="rehearsal-model-missing-terminal",
        purpose="research_lab.private_model_run.v2",
        retry_policy_hashes=retry_policy_hashes,
        terminal_sink=lambda _attempt: None,
        artifact_sink=lambda _artifact: None,
        dynamic_provider_catalog=runtime_catalog,
    )
    incomplete_scope.record_intent("rehearsal-missing-operation", 0)
    try:
        incomplete_scope.assert_accepted_result_is_complete()
    except ProviderClientV2Error as exc:
        if "missing a signed terminal record" not in str(exc):
            raise
    else:
        raise RuntimeError("model sandbox authorized a missing provider terminal")
    transport.restore()

    client_error_probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "\n".join(
                (
                    "import urllib.error, urllib.request",
                    "import gateway.tee.sandbox_http_shim_v2 as shim",
                    "shim.execute = lambda **_kwargs: "
                    "{'terminal_status': 'transport_failure', "
                    "'failure_code': 'timeout'}",
                    "shim.install()",
                    "try:",
                    "    urllib.request.urlopen('https://example.com', timeout=1)",
                    "except urllib.error.URLError as exc:",
                    "    assert 'attested transport failure: timeout' in str(exc)",
                    "else:",
                    "    raise RuntimeError('transport failure was not client-native')",
                )
            ),
        ],
        cwd=SOURCE_ROOT,
        env=dict(os.environ),
        text=True,
        capture_output=True,
        timeout=15,
        check=False,
    )
    if client_error_probe.returncode != 0:
        raise RuntimeError(
            "model sandbox transport failure did not preserve client semantics: "
            + str(client_error_probe.stderr or "")[-500:]
        )

    with tempfile.TemporaryDirectory(
        prefix="leadpoet-rehearsal-model-import-"
    ) as raw_tmp:
        root = Path(raw_tmp)
        trusted_root = root / "trusted"
        attested_root = root / "attested"
        source_root = root / "source"
        neutral_root = root / "neutral"
        neutral_root.mkdir()
        packages = {
            trusted_root / "gateway" / "__init__.py": "ORIGIN = 'trusted'\n",
            trusted_root / "gateway" / "tee" / "__init__.py": (
                "ORIGIN = 'trusted'\n"
            ),
            attested_root / "leadpoet_canonical" / "__init__.py": (
                "ORIGIN = 'trusted'\n"
            ),
            source_root / "gateway" / "__init__.py": "ORIGIN = 'source'\n",
            source_root / "gateway" / "tasks" / "__init__.py": "",
            source_root / "gateway" / "tasks" / "fixture.py": (
                "ORIGIN = 'source'\n"
            ),
            source_root / "qualification" / "__init__.py": (
                "ORIGIN = 'source'\n"
            ),
            source_root / "validator_models" / "__init__.py": (
                "ORIGIN = 'source'\n"
            ),
        }
        for path, content in packages.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        code = model_source_import_bootstrap(str(source_root)) + """
import gateway
import gateway.tasks.fixture as model_task
import gateway.tee as trusted_tee
import leadpoet_canonical as trusted_canonical
import qualification as model_qualification
import validator_models as model_validator
print(",".join((
    gateway.ORIGIN,
    trusted_tee.ORIGIN,
    trusted_canonical.ORIGIN,
    model_task.ORIGIN,
    model_qualification.ORIGIN,
    model_validator.ORIGIN,
)))
"""
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=neutral_root,
            env={
                **os.environ,
                "PYTHONPATH": os.pathsep.join(
                    (str(trusted_root), str(attested_root), str(source_root))
                ),
            },
            text=True,
            capture_output=True,
            timeout=15,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "model sandbox import isolation failed: "
                + str(completed.stderr or "")[-500:]
            )
        if completed.stdout.strip() != (
            "trusted,trusted,trusted,source,source,source"
        ):
            raise RuntimeError("model sandbox import origins differ")

    with tempfile.TemporaryDirectory(prefix="lp-") as raw_tmp:
        root = Path(raw_tmp)
        rootfs = root / "r"
        rootfs.mkdir()
        runtime_config = RunscSandboxConfigV2(
            runsc_path=Path(sys.executable),
            runsc_sha256="sha256:" + "1" * 64,
            rootfs_path=rootfs,
            rootfs_manifest_hash="sha256:" + "2" * 64,
            uid=os.getuid() or 65534,
            gid=os.getgid() or 65534,
        )
        visible_parent = rootfs / MODEL_SANDBOX_VISIBLE_ROOT.lstrip("/")
        visible_parent.mkdir(mode=0o711)
        workspace_name = "lp-job-" + ("a" * 32)
        visible_workspace = visible_parent / workspace_name
        visible_workspace.mkdir(mode=0o711)
        source_root = visible_workspace / MODEL_SANDBOX_SOURCE_DIRECTORY
        broker_root = visible_workspace / MODEL_SANDBOX_BROKER_DIRECTORY
        source_root.mkdir()
        broker_root.mkdir()
        cgroup_root = root / "cgroup"
        proc_cgroup = root / "proc-self-cgroup"
        proc_lines = []
        for hierarchy, controller in enumerate(
            sorted(MODEL_SANDBOX_REQUIRED_CONTROLLERS),
            start=1,
        ):
            current = cgroup_root / controller
            current.mkdir(parents=True)
            (current / "tasks").write_text(
                "%s\n" % os.getpid(), encoding="ascii"
            )
            proc_lines.append(f"{hierarchy}:{controller}:/")
        proc_cgroup.write_text("\n".join(proc_lines) + "\n", encoding="ascii")

        delegated_parent = prepare_model_sandbox_cgroup_v2(
            cgroup_root=cgroup_root,
            proc_self_cgroup_path=proc_cgroup,
        )
        if delegated_parent != "leadpoet-model":
            raise RuntimeError("model sandbox cgroup delegation differs")
        if any(
            (cgroup_root / controller / filename).exists()
            for controller in MODEL_SANDBOX_REQUIRED_CONTROLLERS
            for filename in MODEL_SANDBOX_CGROUP_V1_CONTROL_FILES[controller]
        ):
            raise RuntimeError(
                "Nitro controller root unexpectedly exposes child limits"
            )
        os.chown(broker_root, runtime_config.uid, runtime_config.gid)
        exposed_socket = broker_root / "provider.sock"
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            listener.bind(str(exposed_socket))
            listener.listen(1)
            os.chown(exposed_socket, runtime_config.uid, runtime_config.gid)
            exposed_socket.chmod(0o600)
            oci_config = _oci_config(
                config=runtime_config,
                source_root=source_root,
                broker_root=broker_root,
                process_args=[sys.executable, "-c", "pass"],
                environment={},
                cgroups_path=model_sandbox_job_cgroup_path(
                    delegated_parent,
                    "lp-rehearsal-contract",
                ),
            )
            runsc_command = _runsc_run_command(
                config=runtime_config,
                runsc_root=root / "runsc",
                bundle=root / "bundle",
                sandbox_id="lp-rehearsal-contract",
                host_uds=True,
            )
            process_environment = dict(
                item.split("=", 1)
                for item in oci_config["process"]["env"]
            )
            expected_socket = (
                MODEL_SANDBOX_VISIBLE_ROOT
                + "/"
                + workspace_name
                + "/"
                + MODEL_SANDBOX_BROKER_DIRECTORY
                + "/provider.sock"
            )
            if len(os.fsencode(expected_socket)) > 107:
                raise RuntimeError(
                    "model sandbox provider socket exceeds AF_UNIX path limit"
                )
            client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                client.connect(str(exposed_socket))
                accepted, _ = listener.accept()
                accepted.close()
            finally:
                client.close()
            if (
                process_environment.get("LEADPOET_SANDBOX_PROVIDER_SOCKET")
                != expected_socket
                or not any(
                    item.get("destination") == "/run"
                    and item.get("type") == "tmpfs"
                    for item in oci_config["mounts"]
                )
                or any(
                    item.get("type") == "bind"
                    for item in oci_config["mounts"]
                )
                or not broker_root.is_relative_to(rootfs)
                or not source_root.is_relative_to(rootfs)
                or process_environment.get("LEADPOET_MODEL_SOURCE_ROOT")
                != (
                    MODEL_SANDBOX_VISIBLE_ROOT
                    + "/"
                    + workspace_name
                    + "/"
                    + MODEL_SANDBOX_SOURCE_DIRECTORY
                )
                or "/dev/log" not in oci_config["linux"]["maskedPaths"]
                or oci_config["linux"].get("cgroupsPath")
                != "leadpoet-model/lp-rehearsal-contract"
                or oci_config["linux"]["resources"]
                != {
                    "memory": {"limit": runtime_config.memory_limit_bytes},
                    "cpu": {
                        "quota": runtime_config.cpu_quota,
                        "period": runtime_config.cpu_period,
                    },
                    "pids": {"limit": runtime_config.pids_limit},
                }
                or "--rootless=false" not in runsc_command
                or "--rootless=true" in runsc_command
                or "--network=none" not in runsc_command
                or "--host-uds=open" not in runsc_command
            ):
                raise RuntimeError(
                    "model sandbox rootfs-visible input contract differs"
                )
        finally:
            listener.close()
    return {
        "final_provider_cost_scope_bound": True,
        "model_provider_broker_rootfs_path_bound": True,
        "model_sandbox_cgroup_delegated": True,
        "model_sandbox_rootful_launcher_bound": True,
        "model_signed_transport_failure_fallback_bound": True,
        "model_missing_transport_terminal_rejected": True,
        "model_transport_failure_client_semantics_bound": True,
        "model_source_import_isolated": True,
        "preliminary_scope_rejected": True,
    }


def _exercise_rebenchmark_sandbox_retry_contract() -> dict[str, Any]:
    """Keep measured sandbox failures inside the bounded baseline retry path."""

    from gateway.research_lab.attested_artifacts_v2 import (
        AttestedArtifactPersistenceV2Error,
        _validate_transport_artifact_commitments,
    )
    from gateway.research_lab.attested_scoring_v2 import AttestedScoringV2Error
    from gateway.research_lab.model_authority_v2 import (
        AttestedPrivateModelRunnerV2,
        AttestedPrivateModelRunnerV2Error,
    )
    from gateway.research_lab.scoring_worker import (
        _baseline_error_is_retryable,
        _baseline_summary_checkpointable,
    )
    from gateway.tee.model_sandbox_v2 import (
        ModelSandboxV2Error,
        _model_sandbox_process_timeout_seconds,
    )
    from research_lab.eval.private_runtime import (
        PrivateModelRuntimeError,
        context_with_runtime_options,
    )

    failures = [
        AttestedScoringV2Error(
            "V2 scoring failed closed: execution_modelsandboxv2error"
        ),
        AttestedScoringV2Error(
            "V2 scoring failed closed: execution_providerclientv2error",
            authority={
                "transport_attempts": [
                    {
                        "logical_operation_id": "measured-provider-op",
                        "attempt_number": 0,
                        "terminal_status": "transport_failure",
                    }
                ]
            },
        ),
        AttestedScoringV2Error(
            "V2 scoring failed closed: execution_providerclientv2error",
            authority={
                "transport_attempts": [
                    {
                        "logical_operation_id": "measured-scrapingdog-op",
                        "attempt_number": 0,
                        "provider_id": "scrapingdog",
                        "terminal_status": "authenticated_response",
                        "http_status": 400,
                    }
                ]
            },
        ),
    ]

    async def fail_measured_operation(**_kwargs: Any) -> Any:
        if not failures:
            raise RuntimeError("rebenchmark retry fixture was exhausted")
        error = failures.pop(0)
        raise error

    async def generic_provider_client_failure(**_kwargs: Any) -> Any:
        raise AttestedScoringV2Error(
            "V2 scoring failed closed: execution_providerclientv2error",
            authority={
                "transport_attempts": [
                    {
                        "logical_operation_id": "measured-public-web-op",
                        "attempt_number": 0,
                        "provider_id": "public_web",
                        "terminal_status": "authenticated_response",
                        "http_status": 403,
                    }
                ]
            },
        )

    runner = object.__new__(AttestedPrivateModelRunnerV2)
    runner.spec = SimpleNamespace(timeout_seconds=1800)
    runner._execute_operation = fail_measured_operation  # type: ignore[method-assign]
    for _expected_failure in range(3):
        try:
            asyncio.run(runner._invoke_operation(operation="run_icp"))
        except AttestedPrivateModelRunnerV2Error as exc:
            if not isinstance(exc, PrivateModelRuntimeError):
                raise RuntimeError("measured sandbox failure left the runner contract")
            if not isinstance(exc.__cause__, AttestedScoringV2Error):
                raise RuntimeError("measured sandbox failure lost attested ancestry")
            if not _baseline_error_is_retryable(str(exc)):
                raise RuntimeError("measured sandbox failure bypassed bounded retry")
        else:
            raise RuntimeError("measured sandbox failure did not fail closed")
    runner._execute_operation = generic_provider_client_failure  # type: ignore[method-assign]
    try:
        asyncio.run(runner._invoke_operation(operation="run_icp"))
    except AttestedPrivateModelRunnerV2Error as exc:
        if _baseline_error_is_retryable(str(exc)):
            raise RuntimeError("generic provider contract failure became retryable")
    else:
        raise RuntimeError("generic provider contract failure did not fail closed")

    sandbox_timeout = _model_sandbox_process_timeout_seconds(
        {
            "operation": "run_icp",
            "input": {
                "context": {
                    "runtime_options": {"runtime_cap_seconds": 1500.0},
                }
            },
        }
    )
    if sandbox_timeout != 1503:
        raise RuntimeError("model sandbox ignored committed runtime allocation")
    runtime_options = context_with_runtime_options(
        {},
        outer_timeout_seconds=1800,
    )["runtime_options"]
    runtime_cap = float(runtime_options["runtime_cap_seconds"])
    finalization_reserve = float(
        runtime_options["finalization_reserve_seconds"]
    )
    finalization_window = sandbox_timeout - (runtime_cap - finalization_reserve)
    if finalization_reserve != 60.0 or finalization_window < 60.0:
        raise RuntimeError(
            "model runtime lacks committed result-finalization headroom"
        )
    try:
        _model_sandbox_process_timeout_seconds(
            {
                "operation": "run_icp",
                "input": {
                    "context": {
                        "runtime_options": {"runtime_cap_seconds": 1500.1},
                    }
                },
            }
        )
    except ModelSandboxV2Error:
        pass
    else:
        raise RuntimeError("model sandbox accepted an oversized runtime allocation")

    retry_failure = {
        "icp_ref": "icp-retry",
        "_runtime_error": "execution_providerclientv2error",
        "diagnostics": {"sourcing_failed": True},
    }
    recovered_result = {
        "icp_ref": "icp-retry",
        "company_count": 1,
        "score_breakdowns": [{"final_score": 1.0}],
        "diagnostics": {"sourcing_failed": False},
    }
    if _baseline_summary_checkpointable(retry_failure):
        raise RuntimeError("retryable failure became checkpoint eligible")
    if not _baseline_summary_checkpointable(recovered_result):
        raise RuntimeError("recovered ICP result remained checkpoint ineligible")

    request_hash = "sha256:" + "1" * 64
    response_hash = "sha256:" + "2" * 64
    _validate_transport_artifact_commitments(
        expected_hashes=(request_hash, response_hash, request_hash, response_hash),
        observed_hashes=(request_hash, response_hash),
        committed_hashes=(request_hash, response_hash),
    )
    missing_hash = "sha256:" + "3" * 64
    try:
        _validate_transport_artifact_commitments(
            expected_hashes=(request_hash, response_hash, missing_hash),
            observed_hashes=(request_hash, response_hash),
            committed_hashes=(request_hash, response_hash, missing_hash),
        )
    except AttestedArtifactPersistenceV2Error:
        pass
    else:
        raise RuntimeError("missing distinct transport artifact was accepted")
    return {
        "attested_ancestry_preserved": True,
        "private_runner_contract_preserved": True,
        "bounded_retry_selected": True,
        "generic_provider_contract_failure_terminal": True,
        "configured_runtime_deadline_bound": True,
        "configured_runtime_finalization_reserve_bound": True,
        "signed_http_retry_selected": True,
        "retry_checkpoint_recovery_bound": True,
        "content_addressed_artifact_persistence_bound": True,
        "missing_distinct_artifact_rejected": True,
    }


def _exercise_fresh_weight_input_lineage() -> dict[str, Any]:
    """Exercise fresh checkpoint lineage, replay, and fail-closed mismatch."""

    import subprocess

    from gateway.research_lab.attested_weight_inputs_v2 import (
        AttestedWeightInputsV2Error,
        build_gateway_weight_inputs_v2,
    )
    from leadpoet_canonical.ancestry_checkpoint_v2 import (
        ANCESTRY_DELTA_SCHEMA_VERSION,
        build_compact_ancestry_proof_from_delta_v2,
        issue_ancestry_certificate_v2,
    )
    from leadpoet_canonical.attested_v2 import (
        build_checkpointed_receipt_graph,
        validate_receipt_graph,
    )
    from leadpoet_canonical.weight_authority_v2 import (
        GATEWAY_WEIGHT_INPUT_CATEGORIES,
        WEIGHT_INPUT_PURPOSES,
        gateway_weight_input_value_documents_v2,
    )

    candidate_sha = subprocess.run(
        ["git", "-C", str(SOURCE_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    fixture = SanitizedWeightFixture(
        candidate_sha=candidate_sha,
        epoch_id=30_000,
    )
    config_hash = sha256_json({"rehearsal": "fresh-weight-input-lineage"})
    boot = fixture._boot(
        role="gateway_coordinator",
        key=fixture.coordinator_key,
        config_hash=config_hash,
    )
    allocation_receipt = fixture.receipt(
        role="gateway_coordinator",
        purpose="research_lab.allocation.v2",
        job_id="rehearsal-weight-input-allocation",
        key=fixture.coordinator_key,
        boot=boot,
        config_hash=config_hash,
        output_root=sha256_json({"allocation": 30_000}),
        sequence=0,
    )
    allocation_graph = build_receipt_graph(
        root_receipt_hash=allocation_receipt["receipt_hash"],
        boot_identities=(boot,),
        receipts=(allocation_receipt,),
        transport_attempts=(),
    )
    snapshot = fixture.calculation_snapshot(
        [allocation_receipt["receipt_hash"]],
        allocation_receipt["receipt_hash"],
    )
    expected_documents = gateway_weight_input_value_documents_v2(
        calculation_snapshot=snapshot,
        gateway_authority_event_hash=allocation_receipt["receipt_hash"],
    )
    lineage_id = sha256_json({"lineage": "fresh-weight-input"})

    def verify_boot(identity: Mapping[str, Any]) -> Mapping[str, Any]:
        return identity

    def outcome(
        *,
        category: str,
        sequence: int,
        fresh: bool,
        mismatched_execution: bool = False,
    ) -> dict[str, Any]:
        role, purpose = WEIGHT_INPUT_PURPOSES[category]
        document = expected_documents[category]
        execution_receipt = fixture.receipt(
            role=role,
            purpose=purpose,
            job_id=f"rehearsal-weight-input-{category}",
            key=fixture.coordinator_key,
            boot=boot,
            config_hash=config_hash,
            output_root=sha256_json(document),
            sequence=100 + sequence,
        )
        execution_graph = build_receipt_graph(
            root_receipt_hash=execution_receipt["receipt_hash"],
            boot_identities=(boot,),
            receipts=(execution_receipt,),
            transport_attempts=(),
        )
        execution_delta = {
            "schema_version": ANCESTRY_DELTA_SCHEMA_VERSION,
            "root_receipt_hash": execution_receipt["receipt_hash"],
            "boot_identities": [boot],
            "receipts": [execution_receipt],
            "transport_attempts": [],
            "host_operations": [],
        }
        execution_certificate = issue_ancestry_certificate_v2(
            local_delta=execution_delta,
            lineage_id=lineage_id,
            certificate_sequence=0,
            issuer_boot_identity=boot,
            issued_at=NOW,
            sign_digest=fixture.coordinator_key.sign,
            boot_attestation_verifier=verify_boot,
            allowed_issuer_roles=("gateway_coordinator",),
            required_purposes=(purpose,),
        )
        execution_proof = build_compact_ancestry_proof_from_delta_v2(
            execution_delta,
            execution_certificate,
            expected_lineage_id=lineage_id,
            boot_attestation_verifier=verify_boot,
            allowed_issuer_roles=("gateway_coordinator",),
        )
        if not fresh:
            return {
                "status": "succeeded",
                "result": document,
                "receipt": execution_receipt,
                "execution_receipt": execution_receipt,
                "execution_receipt_graph": execution_graph,
                "receipt_graph": execution_graph,
                "execution_ancestry_compact_proof": execution_proof,
                "ancestry_compact_proof": execution_proof,
            }

        persistence_receipt = fixture.receipt(
            role="gateway_coordinator",
            purpose="leadpoet.artifact_persistence.v2",
            job_id=f"rehearsal-weight-input-persistence-{category}",
            key=fixture.coordinator_key,
            boot=boot,
            config_hash=config_hash,
            output_root=sha256_json(
                {"source_receipt_hash": execution_receipt["receipt_hash"]}
            ),
            parents=(execution_receipt["receipt_hash"],),
            sequence=1_000 + sequence,
        )
        local_delta = {
            "schema_version": ANCESTRY_DELTA_SCHEMA_VERSION,
            "root_receipt_hash": persistence_receipt["receipt_hash"],
            "boot_identities": [boot],
            "receipts": [persistence_receipt],
            "transport_attempts": [],
            "host_operations": [],
        }
        certificate = issue_ancestry_certificate_v2(
            local_delta=local_delta,
            lineage_id=lineage_id,
            certificate_sequence=1,
            issuer_boot_identity=boot,
            issued_at=NOW,
            sign_digest=fixture.coordinator_key.sign,
            boot_attestation_verifier=verify_boot,
            allowed_issuer_roles=("gateway_coordinator",),
            parent_proof_disclosures=(
                (execution_proof, execution_receipt["receipt_hash"]),
            ),
            required_purposes=("leadpoet.artifact_persistence.v2",),
        )
        proof = build_compact_ancestry_proof_from_delta_v2(
            local_delta,
            certificate,
            expected_lineage_id=lineage_id,
            boot_attestation_verifier=verify_boot,
            allowed_issuer_roles=("gateway_coordinator",),
        )
        lineage_graph = build_checkpointed_receipt_graph(
            root_receipt_hash=persistence_receipt["receipt_hash"],
            boot_identities=(boot,),
            receipts=(persistence_receipt,),
            transport_attempts=(),
            host_operations=(),
            ancestry_lineage_id=lineage_id,
            ancestry_proof=proof,
            boot_attestation_verifier=verify_boot,
            require_boot_attestation_verification=True,
        )
        validate_receipt_graph(
            lineage_graph,
            required_purposes=(purpose, "leadpoet.artifact_persistence.v2"),
        )
        exposed_execution_receipt = execution_receipt
        if mismatched_execution:
            exposed_execution_receipt = fixture.receipt(
                role=role,
                purpose=purpose,
                job_id=f"rehearsal-weight-input-mismatch-{category}",
                key=fixture.coordinator_key,
                boot=boot,
                config_hash=config_hash,
                output_root=sha256_json(document),
                sequence=2_000 + sequence,
            )
        return {
            "status": "succeeded",
            "result": document,
            "receipt": persistence_receipt,
            "execution_receipt": exposed_execution_receipt,
            "execution_receipt_graph": execution_graph,
            "execution_ancestry_compact_proof": execution_proof,
            "ancestry_compact_proof": proof,
            "receipt_graph": lineage_graph,
        }

    async def run(*, fresh: bool, mismatch_category: str | None = None):
        async def execute(**kwargs):
            category = str(kwargs["payload"]["category"])
            return outcome(
                category=category,
                sequence=int(kwargs["sequence"]),
                fresh=fresh,
                mismatched_execution=category == mismatch_category,
            )

        return await build_gateway_weight_inputs_v2(
            calculation_snapshot=snapshot,
            allocation_graph=allocation_graph,
            leaderboard_window_start="2026-07-24T00:00:00Z",
            leaderboard_window_end="2026-07-25T00:00:00Z",
            execute=execute,
            load_sourcing_graphs=lambda **_kwargs: _async_value([]),
            coordinator_client_factory=object,
        )

    async def _async_value(value):
        return value

    fresh = asyncio.run(run(fresh=True))
    replay = asyncio.run(run(fresh=False))
    if fresh["input_receipt_hashes"] != replay["input_receipt_hashes"]:
        raise RuntimeError("fresh and replay input identities differ")
    compact = fresh.get("compact_ancestry")
    if not isinstance(compact, Mapping):
        raise RuntimeError("fresh execution compact ancestry is absent")
    proof_roots = {
        category: str(
            proof["certificate"]["claim"]["output_root_receipt_hash"]
        )
        for category, proof in compact["upstream_ancestry_proofs"].items()
    }
    if proof_roots != fresh["input_receipt_hashes"]:
        raise RuntimeError("fresh compact ancestry does not bind direct inputs")
    direct_hashes = set(fresh["input_receipt_hashes"].values())
    receipt_hashes = {
        str(item["receipt_hash"])
        for item in fresh["upstream_receipt_set"]["receipts"]
    }
    if (
        set(fresh["input_receipt_hashes"])
        != set(GATEWAY_WEIGHT_INPUT_CATEGORIES)
        or not direct_hashes.issubset(receipt_hashes)
        or len(receipt_hashes) != 2 * len(GATEWAY_WEIGHT_INPUT_CATEGORIES)
    ):
        raise RuntimeError("fresh weight input receipt persistence is incomplete")
    try:
        asyncio.run(run(fresh=True, mismatch_category="fulfillment_rewards"))
    except AttestedWeightInputsV2Error as exc:
        if "measured input receipt is invalid" not in str(exc):
            raise
    else:
        raise RuntimeError("mismatched fresh execution receipt did not fail closed")
    return {
        "fresh_checkpoint_lineage_accepted": True,
        "direct_execution_proof_selected": True,
        "replay_identity_equal": True,
        "direct_receipts_persisted": True,
        "mismatched_execution_rejected": True,
    }


def _exercise_stateful_compact_graph_readback() -> dict[str, Any]:
    """Exercise V3 persistence followed by its canonical V4 readback."""

    import copy
    import subprocess

    from Leadpoet.utils.subnet_epoch import (
        SubnetEpochCutover,
        SubnetEpochSnapshot,
    )
    from gateway.research_lab.stateful_epoch_authority_v1 import (
        BOUNDARY_TABLE,
        SNAPSHOT_TABLE,
        StatefulEpochAuthorityStoreError,
        persist_post_cutover_evidence_v1,
    )
    from gateway.tee.coordinator_epoch_cutover_v2 import SNAPSHOT_PURPOSE
    from leadpoet_canonical.ancestry_checkpoint_v2 import (
        ANCESTRY_DELTA_SCHEMA_VERSION,
        build_compact_ancestry_proof_from_delta_v2,
        issue_ancestry_certificate_v2,
    )
    from leadpoet_canonical.attested_v2 import (
        WEIGHT_ROLE,
        build_checkpointed_receipt_graph,
        compact_checkpointed_receipt_graph,
    )

    candidate_sha = subprocess.run(
        ["git", "-C", str(SOURCE_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    cutover = SubnetEpochCutover(
        network_genesis_hash="0x" + "1" * 64,
        netuid=71,
        cutover_block=1_000,
        cutover_block_hash="0x" + "2" * 64,
        first_subnet_epoch_index=10,
        first_settlement_epoch_id=101,
        last_legacy_epoch_id=100,
    )
    boundary_snapshot = SubnetEpochSnapshot(
        network_genesis_hash=cutover.network_genesis_hash,
        netuid=cutover.netuid,
        head_kind="finalized",
        block_hash="0x" + "3" * 64,
        current_block=1_360,
        last_epoch_block=1_360,
        pending_epoch_at=0,
        subnet_epoch_index=11,
        tempo=360,
        blocks_since_last_step=0,
        observed_at=NOW,
    )
    current_snapshot = SubnetEpochSnapshot(
        network_genesis_hash=cutover.network_genesis_hash,
        netuid=cutover.netuid,
        head_kind="finalized",
        block_hash="0x" + "4" * 64,
        current_block=1_700,
        last_epoch_block=1_360,
        pending_epoch_at=0,
        subnet_epoch_index=11,
        tempo=360,
        blocks_since_last_step=340,
        observed_at=NOW,
    )
    boundary_doc = boundary_snapshot.to_dict(cutover=cutover)
    current_doc = current_snapshot.to_dict(cutover=cutover)
    epoch_id = int(current_doc["settlement_epoch_id"])
    fixture = SanitizedWeightFixture(
        candidate_sha=candidate_sha,
        epoch_id=epoch_id,
    )
    config_hash = sha256_json({"rehearsal": "stateful-compact-readback"})
    boot = fixture._boot(
        role=WEIGHT_ROLE,
        key=fixture.weight_key,
        config_hash=config_hash,
    )
    boundary_receipt = fixture.receipt(
        role=WEIGHT_ROLE,
        purpose=SNAPSHOT_PURPOSE,
        job_id=f"subnet-epoch-boundary:{epoch_id}",
        key=fixture.weight_key,
        boot=boot,
        config_hash=config_hash,
        output_root=sha256_json(boundary_doc),
        sequence=0,
    )
    current_receipt = fixture.receipt(
        role=WEIGHT_ROLE,
        purpose=SNAPSHOT_PURPOSE,
        job_id=f"subnet-epoch-current:{epoch_id}",
        key=fixture.weight_key,
        boot=boot,
        config_hash=config_hash,
        output_root=sha256_json(current_doc),
        parents=(boundary_receipt["receipt_hash"],),
        sequence=1,
    )
    delta = {
        "schema_version": ANCESTRY_DELTA_SCHEMA_VERSION,
        "root_receipt_hash": current_receipt["receipt_hash"],
        "boot_identities": [boot],
        "receipts": [boundary_receipt, current_receipt],
        "transport_attempts": [],
        "host_operations": [],
    }
    lineage_id = sha256_json(
        {
            "cutover_mapping_hash": cutover.mapping_hash,
            "candidate_sha": candidate_sha,
        }
    )

    def verify_boot(identity: Mapping[str, Any]) -> Mapping[str, Any]:
        if identity.get("commit_sha") != candidate_sha:
            raise RuntimeError("checkpoint boot commit differs")
        return identity

    certificate = issue_ancestry_certificate_v2(
        local_delta=delta,
        lineage_id=lineage_id,
        certificate_sequence=0,
        issuer_boot_identity=boot,
        issued_at=NOW,
        sign_digest=fixture.weight_key.sign,
        boot_attestation_verifier=verify_boot,
        allowed_issuer_roles=(WEIGHT_ROLE,),
        required_purposes=(SNAPSHOT_PURPOSE,),
    )
    proof = build_compact_ancestry_proof_from_delta_v2(
        delta,
        certificate,
        expected_lineage_id=lineage_id,
        boot_attestation_verifier=verify_boot,
        allowed_issuer_roles=(WEIGHT_ROLE,),
    )
    graph = build_checkpointed_receipt_graph(
        root_receipt_hash=current_receipt["receipt_hash"],
        boot_identities=(boot,),
        receipts=(boundary_receipt, current_receipt),
        transport_attempts=(),
        host_operations=(),
        ancestry_lineage_id=lineage_id,
        ancestry_proof=proof,
        boot_attestation_verifier=verify_boot,
        require_boot_attestation_verification=True,
    )
    compact_graph = compact_checkpointed_receipt_graph(
        graph,
        boot_attestation_verifier=verify_boot,
        require_boot_attestation_verification=True,
    )
    evidence = {
        "schema_version": "leadpoet.validator_subnet_epoch_evidence.v1",
        "validator_hotkey": VALIDATOR_HOTKEY,
        "bundle_hash": sha256_json({"bundle": epoch_id}),
        "cutover_mapping_hash": cutover.mapping_hash,
        "epoch_authority": current_doc,
        "epoch_authority_hash": sha256_json(current_doc),
        "epoch_authority_receipt_hash": current_receipt["receipt_hash"],
        "epoch_boundary": boundary_doc,
        "epoch_boundary_hash": sha256_json(boundary_doc),
        "epoch_boundary_receipt_hash": boundary_receipt["receipt_hash"],
        "receipt_graph": graph,
    }
    tables: dict[str, dict[str, dict[str, Any]]] = {
        BOUNDARY_TABLE: {},
        SNAPSHOT_TABLE: {},
    }

    async def persist_graph(value):
        return {
            "root_receipt_hash": value["root_receipt_hash"],
            "graph_hash": sha256_json(dict(value)),
        }

    async def load_graph(_root):
        return copy.deepcopy(compact_graph)

    async def insert(table, row):
        key_field = {
            BOUNDARY_TABLE: "boundary_hash",
            SNAPSHOT_TABLE: "snapshot_hash",
        }[table]
        key = str(row[key_field])
        if key in tables[table]:
            raise RuntimeError("23505 duplicate key unique constraint")
        tables[table][key] = copy.deepcopy(dict(row))
        return copy.deepcopy(dict(row))

    async def select(table, *, filters):
        field, value = filters[0]
        for row in tables[table].values():
            if row.get(field) == value:
                return copy.deepcopy(row)
        return None

    durable = asyncio.run(
        persist_post_cutover_evidence_v1(
            evidence,
            cutover=cutover.to_dict(),
            persist_graph=persist_graph,
            load_graph=load_graph,
            insert=insert,
            select=select,
        )
    )
    if (
        durable["receipt_graph_hash"] != sha256_json(graph)
        or durable["boundary"]["boundary_hash"]
        != evidence["epoch_boundary_hash"]
        or durable["snapshot"]["snapshot_hash"]
        != evidence["epoch_authority_hash"]
    ):
        raise RuntimeError("canonical compact readback changed stateful evidence")

    tampered = copy.deepcopy(compact_graph)
    tampered["receipts"] = []
    attempted_insert = False

    async def load_tampered(_root):
        return copy.deepcopy(tampered)

    async def reject_insert(_table, _row):
        nonlocal attempted_insert
        attempted_insert = True
        raise RuntimeError("tampered graph reached stateful persistence")

    try:
        asyncio.run(
            persist_post_cutover_evidence_v1(
                evidence,
                cutover=cutover.to_dict(),
                persist_graph=persist_graph,
                load_graph=load_tampered,
                insert=reject_insert,
                select=select,
            )
        )
    except StatefulEpochAuthorityStoreError as exc:
        if "receipt graph readback differs" not in str(exc):
            raise
    else:
        raise RuntimeError("tampered compact graph readback was accepted")
    if attempted_insert:
        raise RuntimeError("tampered compact graph mutated stateful evidence")
    return {
        "checkpoint_v3_persisted": True,
        "canonical_v4_readback_accepted": True,
        "boundary_persisted": True,
        "snapshot_persisted": True,
        "tampered_v4_rejected_before_write": True,
    }


def _exercise_rebenchmark_provider_transport_evidence() -> dict[str, Any]:
    """Prove repeated nonterminal polls retain unique measured evidence."""

    from gateway.tee.artifact_vault_v2 import EncryptedArtifactVaultV2
    from gateway.tee.execution_job_manager_v2 import ExecutionContextV2
    from gateway.tee.inter_enclave_tls import MAX_FRAME_BYTES
    from gateway.tee.topology import COORDINATOR_ROLE, ROLE_SPECS, topology_document
    from gateway.tee.provider_broker_v2 import (
        BUILTIN_PROVIDER_ROUTES,
        MAX_RESPONSE_BODY_BYTES,
        PROVIDER_BROKER_SCHEMA_VERSION,
        PROVIDER_RPC_RESPONSE_RESERVE_BYTES,
        ProviderBrokerV2,
        credential_reference_hash,
        credential_value_hash,
        _provider_rpc_response_body_limit,
    )
    from gateway.research_lab.scoring_worker import (
        _BASELINE_ATTEMPT_RECEIPT_HASHES_FIELD,
        BaselineMaintenancePause,
        ResearchLabGatewayScoringWorker,
        _attested_receipts_with_persisted_roots,
        _baseline_summary_checkpointable,
        _load_baseline_scoring_progress,
        _private_baseline_uses_batch_execution,
        _record_baseline_attempt_parent_receipts,
        _require_v2_baseline_receipt_capacity,
        _store_baseline_scoring_progress,
    )
    from gateway.research_lab import scoring_worker as scoring_worker_module
    from gateway.tee.provider_evidence_cache_store_v2 import (
        CACHE_TRANSPORT_ATTEMPTS,
        ProviderEvidenceCacheStoreV2,
    )
    from gateway.tee.provider_outcome_store_v2 import (
        CHECKPOINT_TRANSPORT_ATTEMPTS,
        ProviderOutcomeStoreV2,
    )
    from gateway.tee.provider_outcome_v2 import ProviderOutcomeLedgerV2
    from gateway.tee.provider_semantics_v2 import ProviderSemanticsAuthorityV2
    from leadpoet_canonical.attested_v2 import (
        DIRECT_EGRESS_REF_HASH,
        build_transport_attempt,
        canonical_json,
        sha256_bytes,
        sha256_json,
    )

    if MAX_RESPONSE_BODY_BYTES != _provider_rpc_response_body_limit(
        frame_bytes=MAX_FRAME_BYTES,
        reserve_bytes=PROVIDER_RPC_RESPONSE_RESERVE_BYTES,
    ):
        raise RuntimeError("provider response limit is not bound to the RPC frame")
    encoded_response_bytes = 4 * ((MAX_RESPONSE_BODY_BYTES + 2) // 3)
    if (
        encoded_response_bytes + PROVIDER_RPC_RESPONSE_RESERVE_BYTES
        > MAX_FRAME_BYTES
    ):
        raise RuntimeError("provider response can exceed the authenticated RPC frame")

    retry_hashes = {
        provider: sha256_json({"retry": provider})
        for provider in ("exa", "supabase")
    }
    credential_hashes = {
        provider: sha256_json({"credential": provider})
        for provider in ("exa", "supabase")
    }

    class StrictBoundary:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []
            self.records: dict[tuple[str, int], dict[str, Any]] = {}
            self.retry_policy_hashes = retry_hashes
            self.cache_authenticated_transients_remaining = (
                CACHE_TRANSPORT_ATTEMPTS - 1
            )
            self.outcome_append_failures_after_commit_remaining = (
                CHECKPOINT_TRANSPORT_ATTEMPTS - 1
            )
            self.outcome_rows: dict[tuple[str, str, int], dict[str, Any]] = {}
            self.cache_rows: dict[tuple[str, str, str], dict[str, Any]] = {}

        @contextmanager
        def transient_terminal_transaction(self):
            yield

        def health(self) -> dict[str, Any]:
            return {"status": "ready", "registry_hash": sha256_json({"registry": 1})}

        def credential_available(self, *, job_id: str, slot: str) -> bool:
            del job_id
            return slot in credential_hashes

        def transport_reference_hashes(
            self,
            request: Mapping[str, Any],
        ) -> dict[str, str]:
            provider = str(request["provider_id"])
            return {
                "credential_ref_hash": credential_hashes[provider],
                "egress_proxy_ref_hash": DIRECT_EGRESS_REF_HASH,
            }

        def execute(self, request: Mapping[str, Any]) -> dict[str, Any]:
            request = dict(request)
            key = (
                str(request["logical_operation_id"]),
                int(request["attempt_number"]),
            )
            self.calls.append(request)
            if key in self.records:
                return dict(self.records[key])
            provider = str(request["provider_id"])
            if provider == "supabase":
                request_url = str(request["url"])
                if "/rpc/append_research_lab_provider_outcome_checkpoints_v2" in request_url:
                    checkpoint_rows = json.loads(
                        base64.b64decode(str(request["body_b64"]), validate=True)
                    )["checkpoint_rows"]
                    for checkpoint_row in checkpoint_rows:
                        row_key = (
                            str(checkpoint_row["artifact_master_key_ref_hash"]),
                            str(checkpoint_row["utc_day"]),
                            int(checkpoint_row["sequence"]),
                        )
                        existing = self.outcome_rows.get(row_key)
                        if existing is not None and existing != checkpoint_row:
                            raise RuntimeError(
                                "checkpoint batch identified another row"
                            )
                        self.outcome_rows[row_key] = dict(checkpoint_row)
                    body = canonical_json(
                        {
                            "checkpoint_hash": checkpoint_rows[-1]["checkpoint_hash"],
                            "checkpoint_count": len(checkpoint_rows),
                            "status": "inserted",
                        }
                    ).encode()
                elif "/rpc/append_research_lab_provider_outcome_checkpoint_v2" in request_url:
                    checkpoint_row = json.loads(
                        base64.b64decode(str(request["body_b64"]), validate=True)
                    )["checkpoint_row"]
                    row_key = (
                        str(checkpoint_row["artifact_master_key_ref_hash"]),
                        str(checkpoint_row["utc_day"]),
                        int(checkpoint_row["sequence"]),
                    )
                    existing = self.outcome_rows.get(row_key)
                    if existing is not None and existing != checkpoint_row:
                        raise RuntimeError("checkpoint hash identified another row")
                    self.outcome_rows[row_key] = dict(checkpoint_row)
                    body = canonical_json(
                        {
                            "checkpoint_hash": checkpoint_row["checkpoint_hash"],
                            "status": "existing" if existing is not None else "inserted",
                        }
                    ).encode()
                elif "/rpc/put_research_lab_provider_evidence_cache_v2" in request_url:
                    cache_row = json.loads(
                        base64.b64decode(str(request["body_b64"]), validate=True)
                    )["cache_row"]
                    row_key = (
                        str(cache_row["artifact_master_key_ref_hash"]),
                        str(cache_row["utc_day"]),
                        str(cache_row["request_fingerprint"]),
                    )
                    existing = self.cache_rows.get(row_key)
                    if existing is not None and existing != cache_row:
                        raise RuntimeError("cache identity identified another row")
                    self.cache_rows.setdefault(row_key, dict(cache_row))
                    body = canonical_json(
                        {
                            "status": "existing" if existing is not None else "inserted",
                            "cache_entry_hash": cache_row["cache_entry_hash"],
                            "cache_row": self.cache_rows[row_key],
                        }
                    ).encode()
                elif "research_lab_provider_outcome_checkpoints_v2" in request_url:
                    from urllib.parse import parse_qs, urlsplit

                    query = parse_qs(urlsplit(request_url).query)
                    day = query["utc_day"][0].split("eq.", 1)[1]
                    key_hash = query["artifact_master_key_ref_hash"][0].split(
                        "eq.", 1
                    )[1]
                    rows = [
                        row
                        for (row_key_hash, row_day, _sequence), row in self.outcome_rows.items()
                        if row_key_hash == key_hash and row_day == day
                    ]
                    if "sequence" in query:
                        sequence = int(query["sequence"][0].split("eq.", 1)[1])
                        rows = [
                            row for row in rows if int(row["sequence"]) == sequence
                        ]
                    if query.get("order") == ["sequence.desc"]:
                        rows.sort(
                            key=lambda row: int(row["sequence"]),
                            reverse=True,
                        )
                    body = canonical_json(
                        rows[: int(query.get("limit", ["2"])[0])]
                    ).encode()
                else:
                    body = b"[]"
                host = "fixture.supabase.co"
            elif provider == "exa":
                body = (
                    b'{"costDollars":0.005,"results":[]}'
                    if str(request["url"]).endswith("/search")
                    else b'{"status":"running","object":"agent_run"}'
                )
                host = "api.exa.ai"
            else:
                raise RuntimeError("provider evidence probe crossed an undeclared route")
            ordinal = len(self.records) + 1
            request_artifact_hash = sha256_json(
                {"provider": provider, "ordinal": ordinal, "kind": "request"}
            )
            response_hash = sha256_bytes(body)
            authenticated_transient = (
                provider == "supabase"
                and "research_lab_provider_evidence_cache_v2" in str(request["url"])
                and self.cache_authenticated_transients_remaining > 0
            )
            if authenticated_transient:
                self.cache_authenticated_transients_remaining -= 1
                body = canonical_json({"code": "PGRST002"}).encode()
                response_hash = sha256_bytes(body)
            transport_failure = False
            if (
                provider == "supabase"
                and (
                    "/rpc/append_research_lab_provider_outcome_checkpoints_v2"
                    in str(request["url"])
                    or "/rpc/append_research_lab_provider_outcome_checkpoint_v2"
                    in str(request["url"])
                )
                and self.outcome_append_failures_after_commit_remaining > 0
            ):
                self.outcome_append_failures_after_commit_remaining -= 1
                transport_failure = True
            attempt = build_transport_attempt(
                request_id=("%032x" % ordinal)[-32:],
                logical_operation_id=key[0],
                job_id=str(request["job_id"]),
                purpose=str(request["purpose"]),
                provider_id=provider,
                attempt_number=key[1],
                method=str(request["method"]),
                destination_host=host,
                destination_port=443,
                path_hash=sha256_json({"path": provider}),
                nonsecret_headers_hash=sha256_json(
                    {"headers": sorted(dict(request["headers"]))}
                ),
                body_hash=sha256_bytes(
                    base64.b64decode(str(request["body_b64"]), validate=True)
                ),
                credential_ref_hash=credential_hashes[provider],
                egress_proxy_ref_hash=DIRECT_EGRESS_REF_HASH,
                retry_policy_hash=str(request["retry_policy_hash"]),
                timeout_ms=int(request["timeout_ms"]),
                started_at=NOW,
                terminal_status=(
                    "transport_failure"
                    if transport_failure
                    else "authenticated_response"
                ),
                http_status=(
                    None
                    if transport_failure
                    else (503 if authenticated_transient else 200)
                ),
                response_hash=None if transport_failure else response_hash,
                request_artifact_hash=request_artifact_hash,
                response_artifact_hash=(None if transport_failure else response_hash),
                tls_peer_chain_hash=(
                    None if transport_failure else sha256_json({"tls": provider})
                ),
                tls_protocol=None if transport_failure else "TLSv1.3",
                failure_code="unexpected_eof" if transport_failure else None,
                completed_at=NOW,
            )
            result: dict[str, Any] = {
                "terminal_status": attempt["terminal_status"],
                "encrypted_request_artifact_id": request_artifact_hash,
                "transport_attempt": attempt,
                "evidence_artifact_hashes": [request_artifact_hash],
            }
            if transport_failure:
                result.update(
                    {
                        "failure_code": "unexpected_eof",
                        "failure_stage": "provider_transport",
                    }
                )
            else:
                result.update(
                    {
                        "http_status": attempt["http_status"],
                        "headers": {"content-type": "application/json"},
                        "body_b64": base64.b64encode(body).decode("ascii"),
                        "encrypted_artifact_id": response_hash,
                        "evidence_artifact_hashes": [
                            request_artifact_hash,
                            response_hash,
                        ],
                    }
                )
            self.records[key] = dict(result)
            return result

    boundary = StrictBoundary()
    vault = EncryptedArtifactVaultV2(
        master_key=bytes(range(32)),
        boot_identity_hash=sha256_json({"boot": "provider-evidence"}),
        retention_days=30,
    )
    cache = ProviderEvidenceCacheStoreV2(
        broker=boundary,
        vault=vault,
        source_boot_verifier=lambda _value: None,
        sleeper=lambda _seconds: None,
    )
    signing_key = Ed25519PrivateKey.generate()
    boot_identity = {
        "boot_identity_hash": sha256_json({"boot": "provider-evidence"}),
        "signing_pubkey": signing_key.public_key().public_bytes_raw().hex(),
    }
    authority = ProviderSemanticsAuthorityV2(
        broker=boundary,
        cache_store=cache,
        artifact_sink=vault.seal,
        boot_identity_supplier=lambda: boot_identity,
        sign_digest=signing_key.sign,
        clock=lambda: NOW,
        sleeper=lambda _seconds: None,
    )
    context = ExecutionContextV2(
        job_id="rehearsal:rebenchmark-provider-poll",
        purpose="research_lab.private_model_run.v2",
        epoch_id=1,
        provider_credential_ref_hashes=credential_hashes,
    )
    request = {
        "schema_version": PROVIDER_BROKER_SCHEMA_VERSION,
        "logical_operation_id": "rehearsal:exa-agent-poll",
        "job_id": context.job_id,
        "purpose": context.purpose,
        "provider_id": "exa",
        "attempt_number": 0,
        "method": "GET",
        "url": "https://api.exa.ai/agent/runs/rehearsal-run",
        "headers": {},
        "body_b64": "",
        "timeout_ms": 30_000,
        "retry_policy_hash": retry_hashes["exa"],
    }
    for attempt_number in (0, 1):
        result = authority.execute(
            {**request, "attempt_number": attempt_number}
        )
        if result.get("evidence") != "live_unrecorded":
            raise RuntimeError("nonterminal provider poll entered the day cache")
        for attempt in [
            *list(result.get("additional_transport_attempts") or ()),
            result["transport_attempt"],
        ]:
            context.record_transport(attempt)
    supabase_attempts = [
        item
        for item in context.transport_attempts
        if item["provider_id"] == "supabase"
    ]
    if (
        len(supabase_attempts) != CACHE_TRANSPORT_ATTEMPTS + 1
        or [item["attempt_number"] for item in supabase_attempts]
        != list(range(CACHE_TRANSPORT_ATTEMPTS + 1))
        or [item["terminal_status"] for item in supabase_attempts]
        != ["authenticated_response"] * (CACHE_TRANSPORT_ATTEMPTS + 1)
        or [item["http_status"] for item in supabase_attempts]
        != [503] * (CACHE_TRANSPORT_ATTEMPTS - 1) + [200, 200]
        or len({item["attempt_hash"] for item in supabase_attempts})
        != CACHE_TRANSPORT_ATTEMPTS + 1
        or len({item["logical_operation_id"] for item in supabase_attempts}) != 1
    ):
        raise RuntimeError("repeated provider cache reads reused transport evidence")

    cache_call_start = len(boundary.calls)
    recorded_request = {
        **request,
        "logical_operation_id": "rehearsal:exa-search-cache-put",
        "method": "POST",
        "url": "https://api.exa.ai/search",
        "headers": {"X-Research-Lab-Cost-Scope": "rehearsal-cache-put"},
        "body_b64": base64.b64encode(b'{"query":"rehearsal"}').decode(),
    }
    recorded = authority.execute(recorded_request)
    replayed = authority.execute(
        {
            **recorded_request,
            "logical_operation_id": "rehearsal:exa-search-cache-replay",
        }
    )
    cache_calls = boundary.calls[cache_call_start:]
    cache_put_calls = [
        call
        for call in cache_calls
        if "/rpc/put_research_lab_provider_evidence_cache_v2"
        in str(call["url"])
    ]
    if (
        recorded.get("evidence") != "recorded"
        or replayed.get("evidence") != "hit"
        or len(cache_put_calls) != 1
        or len(boundary.cache_rows) != 1
        or sum(call["provider_id"] == "exa" for call in cache_calls) != 1
    ):
        raise RuntimeError("atomic provider cache put/replay differed")

    outcome_store = ProviderOutcomeStoreV2(
        broker=boundary,
        vault=vault,
        sleeper=lambda _seconds: None,
    )
    outcome_ledger = ProviderOutcomeLedgerV2(clock=lambda: NOW)
    outcome_document = outcome_ledger.record(
        provider_id="deepline",
        endpoint_class="/v1/search",
        evidence="recorded",
        status=200,
        live_call=True,
        cost_event={},
    )
    outcome_call_start = len(boundary.calls)
    outcome = outcome_store.persist(
        outcome_document,
        previous_checkpoint_hash="",
        job_id=context.job_id,
        purpose=context.purpose,
    )
    persist_outcome_calls = boundary.calls[outcome_call_start:]
    if [item["method"] for item in persist_outcome_calls] != [
        "POST"
    ] * CHECKPOINT_TRANSPORT_ATTEMPTS:
        raise RuntimeError(
            "provider outcome append issued a redundant durable readback"
        )
    restore_call_start = len(boundary.calls)
    restarted_outcome_store = ProviderOutcomeStoreV2(
        broker=boundary,
        vault=vault,
        sleeper=lambda _seconds: None,
    )
    restored_outcome = restarted_outcome_store.load_latest(
        utc_day=str(outcome_document["utc_day"]),
        job_id=context.job_id,
        purpose=context.purpose,
    )
    restore_outcome_calls = boundary.calls[restore_call_start:]
    append_attempts = [
        item
        for item in outcome["transport_attempts"]
        if str(item["logical_operation_id"]).endswith(":append")
    ]
    if (
        outcome.get("status") != "persisted"
        or [item["attempt_number"] for item in append_attempts]
        != list(range(CHECKPOINT_TRANSPORT_ATTEMPTS))
        or [item["terminal_status"] for item in append_attempts]
        != ["transport_failure"] * (CHECKPOINT_TRANSPORT_ATTEMPTS - 1)
        + ["authenticated_response"]
        or len(boundary.outcome_rows) != 1
        or [item["method"] for item in restore_outcome_calls] != ["GET"]
        or restored_outcome.get("checkpoint_hash") != outcome["checkpoint_hash"]
        or restored_outcome.get("state_document") != outcome_document
    ):
        raise RuntimeError("provider outcome transient recovery differed")

    batch_documents = []
    for provider_id in ("exa", "scrapingdog", "exa"):
        batch_documents.append(
            outcome_ledger.record(
                provider_id=provider_id,
                endpoint_class="/batch",
                evidence="recorded",
                status=200,
                live_call=True,
                cost_event={},
            )
        )
    batch_call_start = len(boundary.calls)
    batch_outcome = outcome_store.persist_batch(
        [
            {
                "document": document,
                "job_id": context.job_id,
                "purpose": context.purpose,
            }
            for document in batch_documents
        ],
        previous_checkpoint_hash=outcome["checkpoint_hash"],
        job_id=context.job_id,
        purpose=context.purpose,
    )
    batch_calls = boundary.calls[batch_call_start:]
    if (
        batch_outcome.get("status") != "persisted"
        or batch_outcome.get("checkpoint_count") != len(batch_documents)
        or len(batch_calls) != 1
        or not str(batch_calls[0]["url"]).endswith(
            "/rpc/append_research_lab_provider_outcome_checkpoints_v2"
        )
        or len(boundary.outcome_rows) != 4
    ):
        raise RuntimeError("provider outcome atomic batch append differed")

    broker_credentials = {
        "openrouter": "rehearsal-openrouter",
        "exa": "rehearsal-exa",
        "scrapingdog": "rehearsal-scrapingdog",
        "deepline": "rehearsal-deepline",
        "supabase_service_role": "rehearsal-supabase",
        "truelist": "rehearsal-truelist",
    }

    successful_transport_calls: list[dict[str, Any]] = []

    def successful_transport(**request: Any) -> dict[str, Any]:
        successful_transport_calls.append(dict(request))
        return {
            "http_status": 200,
            "headers": {"content-type": "application/json"},
            "body": b'{"costDollars":0.005,"results":[]}',
            "tls_peer_chain_hash": sha256_json({"tls": "rehearsal"}),
            "tls_protocol": "TLSv1.3",
        }

    terminal_broker = ProviderBrokerV2(
        credential_ref_hashes={
            name: credential_reference_hash(value)
            for name, value in broker_credentials.items()
        },
        retry_policy_hashes={
            name: retry_hashes.get(name, sha256_json({"retry": name}))
            for name in BUILTIN_PROVIDER_ROUTES
        },
        transport=successful_transport,
        artifact_sink=vault.seal,
        clock=lambda: NOW,
    )
    terminal_broker.provision_credentials(broker_credentials)

    cache_persist_entered = threading.Event()
    release_cache_persist = threading.Event()

    class CommitBlockingCache:
        def load(self, **_kwargs: Any) -> dict[str, Any]:
            return {
                "found": False,
                "payload": {},
                "transport_attempts": [],
                "evidence_artifact_hashes": [],
            }

        def persist_recorded(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
            cache_persist_entered.set()
            if not release_cache_persist.wait(timeout=2.0):
                raise RuntimeError("provider cache commit fixture did not release")
            return {
                "transport_attempts": [],
                "evidence_artifact_hashes": [],
            }

    shared_authority = ProviderSemanticsAuthorityV2(
        broker=terminal_broker,
        cache_store=CommitBlockingCache(),
        artifact_sink=vault.seal,
        artifact_transaction=vault.transient_artifact_transaction,
        boot_identity_supplier=lambda: boot_identity,
        sign_digest=signing_key.sign,
        clock=lambda: NOW,
        sleeper=lambda _seconds: None,
        outcome_store=outcome_store,
    )
    shared_request = {
        "schema_version": PROVIDER_BROKER_SCHEMA_VERSION,
        "logical_operation_id": "rehearsal:shared-provider-owner",
        "job_id": context.job_id,
        "purpose": context.purpose,
        "provider_id": "exa",
        "attempt_number": 0,
        "method": "POST",
        "url": "https://api.exa.ai/search",
        "headers": {"content-type": "application/json"},
        "body_b64": base64.b64encode(b'{"query":"shared"}').decode("ascii"),
        "timeout_ms": 1,
        "retry_policy_hash": terminal_broker.retry_policy_hashes["exa"],
    }
    transport_count_before_shared = len(successful_transport_calls)
    with ThreadPoolExecutor(max_workers=2) as executor:
        owner = executor.submit(shared_authority.execute, shared_request)
        if not cache_persist_entered.wait(timeout=2.0):
            raise RuntimeError("provider cache commit boundary was not reached")
        follower = executor.submit(
            shared_authority.execute,
            {
                **shared_request,
                "logical_operation_id": "rehearsal:shared-provider-follower",
            },
        )
        time.sleep(0.02)
        if owner.done() or follower.done():
            raise RuntimeError("provider follower escaped the commit boundary")
        release_cache_persist.set()
        shared_results = [owner.result(timeout=2.0), follower.result(timeout=2.0)]
    latest_shared_outcome = outcome_store.load_latest(
        utc_day=NOW[:10],
        job_id=context.job_id,
        purpose=context.purpose,
        operation_suffix="singleflight-readback",
    )
    if (
        len(successful_transport_calls) != transport_count_before_shared + 1
        or {item.get("evidence") for item in shared_results} != {"recorded", "hit"}
        or not latest_shared_outcome.get("found")
        or int(latest_shared_outcome["state_document"]["sequence"])
        != len(boundary.outcome_rows)
    ):
        raise RuntimeError(
            "provider single-flight durable completion differed: "
            + canonical_json(
                {
                    "transport_delta": (
                        len(successful_transport_calls)
                        - transport_count_before_shared
                    ),
                    "evidence": sorted(
                        str(item.get("evidence") or "") for item in shared_results
                    ),
                    "outcome_found": bool(latest_shared_outcome.get("found")),
                    "outcome_sequence": int(
                        dict(latest_shared_outcome.get("state_document") or {}).get(
                            "sequence", -1
                        )
                    ),
                }
            )
        )
    shared_release = terminal_broker.release_job_credentials(context.job_id)
    if shared_release.get("released_terminal_count") != 1:
        raise RuntimeError("provider single-flight terminal cleanup differed")
    released_terminal_count_before_cleanup = terminal_broker.health()[
        "released_terminal_count"
    ]

    def execute_terminal(*, job_id: str, operation_id: str) -> dict[str, Any]:
        return terminal_broker.execute(
            {
                "schema_version": PROVIDER_BROKER_SCHEMA_VERSION,
                "logical_operation_id": operation_id,
                "job_id": job_id,
                "purpose": "leadpoet.artifact_persistence.v2",
                "provider_id": "supabase",
                "attempt_number": 0,
                "method": "GET",
                "url": (
                    "https://qplwoislplkcegvdmbim.supabase.co/"
                    "rest/v1/research_lab_attested_artifacts_v2"
                ),
                "headers": {},
                "body_b64": "",
                "timeout_ms": 30_000,
                "retry_policy_hash": terminal_broker.retry_policy_hashes[
                    "supabase"
                ],
            }
        )

    retained_job_id = "rehearsal:active-provider-job"
    retained = execute_terminal(
        job_id=retained_job_id,
        operation_id="rehearsal:active-provider-operation",
    )
    for ordinal in range(12):
        completed_job_id = f"rehearsal:completed-provider-job:{ordinal}"
        execute_terminal(
            job_id=completed_job_id,
            operation_id=f"rehearsal:completed-provider-operation:{ordinal}",
        )
        released = terminal_broker.release_job_credentials(completed_job_id)
        if (
            released.get("job_id") != completed_job_id
            or released.get("released_slot_count") != 0
            or released.get("released_terminal_count") != 1
        ):
            raise RuntimeError("completed provider job release differed")
    terminal_health = terminal_broker.health()
    replayed = execute_terminal(
        job_id=retained_job_id,
        operation_id="rehearsal:active-provider-operation",
    )
    if (
        terminal_health["terminal_count"] != 1
        or terminal_health["released_terminal_count"]
        != released_terminal_count_before_cleanup + 12
        or replayed["transport_attempt"]["attempt_hash"]
        != retained["transport_attempt"]["attempt_hash"]
    ):
        raise RuntimeError("completed provider cleanup affected active job state")
    terminal_broker.release_job_credentials(retained_job_id)
    if terminal_broker.health()["terminal_count"] != 0:
        raise RuntimeError("provider terminal state remained after job release")

    cross_worker_authority = ProviderSemanticsAuthorityV2(
        broker=terminal_broker,
        cache_store=CommitBlockingCache(),
        artifact_sink=vault.seal,
        artifact_transaction=vault.transient_artifact_transaction,
        boot_identity_supplier=lambda: boot_identity,
        sign_digest=signing_key.sign,
        clock=lambda: NOW,
        sleeper=lambda _seconds: None,
    )
    worker_a = "rehearsal:rebenchmark-worker-a"
    worker_b = "rehearsal:rebenchmark-worker-b"
    proxy_a = "https://worker-a:secret@proxy-a.example.com:443"
    proxy_b = "https://worker-b:secret@proxy-b.example.com:443"
    proxy_a_ref = credential_value_hash(proxy_a)
    proxy_b_ref = credential_value_hash(proxy_b)
    for job_id, proxy, proxy_ref in (
        (worker_a, proxy_a, proxy_a_ref),
        (worker_b, proxy_b, proxy_b_ref),
    ):
        terminal_broker.provision_job_credential(
            job_id=job_id,
            slot="egress_proxy",
            credential=proxy,
            credential_value_hash_expected=proxy_ref,
        )
    cross_worker_request = {
        **shared_request,
        "logical_operation_id": "rehearsal:cross-worker-source",
        "job_id": worker_a,
        "body_b64": base64.b64encode(b'{"query":"cross-worker"}').decode(
            "ascii"
        ),
    }
    source_result = cross_worker_authority.execute(cross_worker_request)
    replay_result = cross_worker_authority.execute(
        {
            **cross_worker_request,
            "logical_operation_id": "rehearsal:cross-worker-replay",
            "job_id": worker_b,
        }
    )
    worker_b_context = ExecutionContextV2(
        job_id=worker_b,
        purpose=context.purpose,
        epoch_id=1,
        provider_credential_ref_hashes={
            "exa": credential_reference_hash(broker_credentials["exa"]),
            "egress_proxy": proxy_b_ref,
        },
    )
    worker_b_context.record_transport(replay_result["transport_attempt"])
    if (
        source_result.get("evidence") != "recorded"
        or replay_result.get("evidence") != "hit"
        or replay_result["transport_attempt"]["egress_proxy_ref_hash"]
        != proxy_b_ref
        or replay_result["source_record"]["transport_attempt_hash"]
        != source_result["transport_attempt"]["attempt_hash"]
    ):
        raise RuntimeError("cross-worker provider cache profile binding differed")
    terminal_broker.release_job_credentials(worker_a)
    terminal_broker.release_job_credentials(worker_b)

    retry_failure = {
        "icp_ref": "rehearsal-retry",
        "_runtime_error": "execution_providerclientv2error",
        "diagnostics": {"sourcing_failed": True},
    }
    recovered_result = {
        "icp_ref": "rehearsal-retry",
        "company_count": 1,
        "score_breakdowns": [{"final_score": 1.0}],
        "diagnostics": {"sourcing_failed": False},
    }
    if (
        _baseline_summary_checkpointable(retry_failure)
        or not _baseline_summary_checkpointable(recovered_result)
    ):
        raise RuntimeError("provider recovery checkpoint eligibility differed")

    if not _private_baseline_uses_batch_execution(
        SimpleNamespace(private_baseline_concurrency=1)
    ):
        raise RuntimeError("async concurrency-one baseline used the serial runner")
    _require_v2_baseline_receipt_capacity(40)
    try:
        _require_v2_baseline_receipt_capacity(65)
    except RuntimeError:
        pass
    else:
        raise RuntimeError("oversized V2 rebenchmark receipt frontier was accepted")

    exact_receipt_frontier: set[str] = set()
    superseded_receipt_hashes: set[str] = set()
    for item_index in range(40):
        model_receipt_hash = f"sha256:{10_000 + item_index:064x}"
        scorer_receipt_hash = f"sha256:{20_000 + item_index:064x}"
        superseded_receipt_hashes.update(
            {
                f"sha256:{30_000 + item_index:064x}",
                f"sha256:{40_000 + item_index:064x}",
            }
        )
        _record_baseline_attempt_parent_receipts(
            exact_receipt_frontier,
            {
                "icp_ref": f"rehearsal-icp-{item_index}",
                _BASELINE_ATTEMPT_RECEIPT_HASHES_FIELD: [
                    model_receipt_hash,
                    scorer_receipt_hash,
                ],
            },
        )
    if (
        len(exact_receipt_frontier) != 80
        or exact_receipt_frontier.intersection(superseded_receipt_hashes)
    ):
        raise RuntimeError("V2 rebenchmark causal receipt frontier differed")

    from types import ModuleType

    checkpoint_objects: dict[tuple[str, str], bytes] = {}

    class CheckpointBody:
        def __init__(self, body: bytes) -> None:
            self._body = body

        def read(self) -> bytes:
            return self._body

    class CheckpointS3:
        def put_object(self, **kwargs: Any) -> dict[str, Any]:
            checkpoint_objects[(str(kwargs["Bucket"]), str(kwargs["Key"]))] = bytes(
                kwargs["Body"]
            )
            return {"ETag": '"rehearsal"'}

        def get_object(self, **kwargs: Any) -> dict[str, Any]:
            body = checkpoint_objects[(str(kwargs["Bucket"]), str(kwargs["Key"]))]
            return {"Body": CheckpointBody(body)}

    checkpoint_s3 = CheckpointS3()
    boto3_stub = ModuleType("boto3")
    boto3_stub.client = lambda *_args, **_kwargs: checkpoint_s3  # type: ignore[attr-defined]
    prior_boto3 = sys.modules.get("boto3")
    sys.modules["boto3"] = boto3_stub
    checkpoint_bucket = "strict-rehearsal-checkpoints"
    checkpoint_key = "baseline/progress.json"
    checkpoint_runtime_sha = "a" * 40
    checkpoint_config_hash = "sha256:" + "b" * 64
    checkpoint_repo_sha = "c" * 40
    checkpoint_manifest_hash = "sha256:" + "d" * 64
    checkpoint_receipt_hashes = [
        "sha256:" + "e" * 64,
        "sha256:" + "f" * 64,
    ]
    checkpoint_row = {
        "icp_ref": "rehearsal-checkpoint-icp",
        "icp_hash": "sha256:" + "0" * 64,
        "score": 61.0,
        "company_count": 1,
        "diagnostics": {"sourcing_failed": False},
    }
    try:
        _store_baseline_scoring_progress(
            checkpoint_bucket,
            checkpoint_key,
            benchmark_date="2026-07-25",
            window_hash="sha256:" + "1" * 64,
            private_model_artifact_hash="sha256:" + "2" * 64,
            gateway_runtime_commit_sha=checkpoint_runtime_sha,
            scoring_configuration_hash_value=checkpoint_config_hash,
            rows=[checkpoint_row],
            attested_parent_receipt_hashes=checkpoint_receipt_hashes,
            repo_git_sha=checkpoint_repo_sha,
            manifest_hash=checkpoint_manifest_hash,
        )

        restored_receipt_hashes: set[str] = set()

        def load_checkpoint(**overrides: Any) -> list[dict[str, Any]]:
            values = {
                "gateway_runtime_commit_sha": checkpoint_runtime_sha,
                "scoring_configuration_hash_value": checkpoint_config_hash,
                "repo_git_sha": checkpoint_repo_sha,
                "manifest_hash": checkpoint_manifest_hash,
                **overrides,
            }
            return _load_baseline_scoring_progress(
                checkpoint_bucket,
                checkpoint_key,
                benchmark_date="2026-07-25",
                window_hash="sha256:" + "1" * 64,
                private_model_artifact_hash="sha256:" + "2" * 64,
                parent_receipt_hashes_out=restored_receipt_hashes,
                **values,
            )

        restored_rows = load_checkpoint()
        if restored_rows != [checkpoint_row]:
            raise RuntimeError("same-release baseline checkpoint did not resume")
        if restored_receipt_hashes != set(checkpoint_receipt_hashes):
            raise RuntimeError("baseline checkpoint receipt roots did not resume")
        if load_checkpoint(gateway_runtime_commit_sha="9" * 40):
            raise RuntimeError("cross-release baseline checkpoint was reused")
        if load_checkpoint(scoring_configuration_hash_value="sha256:" + "8" * 64):
            raise RuntimeError("cross-config baseline checkpoint was reused")
        if load_checkpoint(repo_git_sha="7" * 40):
            raise RuntimeError("cross-model-source baseline checkpoint was reused")
        if load_checkpoint(manifest_hash="sha256:" + "6" * 64):
            raise RuntimeError("cross-model-manifest baseline checkpoint was reused")

        object_ref = (checkpoint_bucket, checkpoint_key)
        valid_checkpoint_body = checkpoint_objects[object_ref]
        malformed_checkpoint = json.loads(valid_checkpoint_body.decode("utf-8"))
        malformed_checkpoint.pop("completed_icp_count")
        checkpoint_objects[object_ref] = json.dumps(
            malformed_checkpoint,
            sort_keys=True,
        ).encode("utf-8")
        if load_checkpoint():
            raise RuntimeError("malformed baseline checkpoint was reused")
        checkpoint_objects[object_ref] = valid_checkpoint_body

        class LiveReceiptSource:
            @staticmethod
            def attested_receipts() -> list[dict[str, Any]]:
                return [
                    {
                        "receipt_hash": checkpoint_receipt_hashes[0],
                        "status": "succeeded",
                    }
                ]

        merged_receipts = _attested_receipts_with_persisted_roots(
            LiveReceiptSource(),
            persisted_receipt_hashes=sorted(restored_receipt_hashes),
        )
        if merged_receipts != [
            {
                "receipt_hash": checkpoint_receipt_hashes[0],
                "status": "succeeded",
            },
            {"receipt_hash": checkpoint_receipt_hashes[1]},
        ]:
            raise RuntimeError("baseline checkpoint receipt roots merged incorrectly")

        async def exercise_pause_checkpoint_resume() -> None:
            class RehearsalScoringWorker(ResearchLabGatewayScoringWorker):
                async def _run_baseline_icp(
                    self,
                    *,
                    item: Mapping[str, Any],
                    item_index: int,
                    **_kwargs: Any,
                ) -> dict[str, Any]:
                    called_indexes.append(item_index)
                    return {
                        "icp_ref": item["icp_ref"],
                        "icp_hash": item["icp_hash"],
                        "score": float(item_index),
                        "company_count": 1,
                        "sourced_count": 1,
                        "diagnostics": {},
                        "_item_index": item_index,
                        "_retryable": False,
                        "_nonempty": True,
                        "_runtime_error": "",
                        "_retry_backoff_seconds": 0.0,
                    }

            worker = object.__new__(RehearsalScoringWorker)
            worker.worker_ref = "rehearsal-baseline-worker"
            worker.config = SimpleNamespace(
                private_baseline_concurrency=1,
                private_baseline_retry_concurrency=1,
                private_baseline_provider_retry_rounds=0,
            )
            window = SimpleNamespace(
                benchmark_items=[
                    {
                        "icp_ref": "rehearsal-checkpoint-icp-1",
                        "icp_hash": "sha256:" + "1" * 64,
                    },
                    {
                        "icp_ref": "rehearsal-checkpoint-icp-2",
                        "icp_hash": "sha256:" + "2" * 64,
                    },
                ]
            )
            called_indexes: list[int] = []
            persisted_rows: dict[str, dict[str, Any]] = {}

            async def checkpoint(row: dict[str, Any]) -> bool:
                persisted_rows[str(row["icp_ref"])] = dict(row)
                _store_baseline_scoring_progress(
                    checkpoint_bucket,
                    checkpoint_key,
                    benchmark_date="2026-07-25",
                    window_hash="sha256:" + "1" * 64,
                    private_model_artifact_hash="sha256:" + "2" * 64,
                    gateway_runtime_commit_sha=checkpoint_runtime_sha,
                    scoring_configuration_hash_value=checkpoint_config_hash,
                    rows=list(persisted_rows.values()),
                    attested_parent_receipt_hashes=checkpoint_receipt_hashes,
                    repo_git_sha=checkpoint_repo_sha,
                    manifest_hash=checkpoint_manifest_hash,
                )
                return True

            pause_states = iter(
                (
                    {"paused": False},
                    {"paused": True, "reason": "operator:rehearsal-pause"},
                )
            )

            async def paused_after_first_wave() -> dict[str, Any]:
                return next(pause_states)

            original_maintenance_reader = (
                scoring_worker_module.get_scoring_maintenance_state
            )
            scoring_worker_module.get_scoring_maintenance_state = (
                paused_after_first_wave
            )
            try:
                try:
                    await worker._run_baseline_batch_inner(
                        runner=object(),
                        retry_runner=object(),
                        scorer=object(),
                        window=window,
                        run_start=time.time(),
                        icp_checkpoint=checkpoint,
                    )
                except BaselineMaintenancePause as exc:
                    if exc.completed_icps != 1 or exc.total_icps != 2:
                        raise RuntimeError(
                            "baseline pause reported the wrong checkpoint boundary"
                        ) from exc
                else:
                    raise RuntimeError("baseline ignored the operator pause boundary")
            finally:
                scoring_worker_module.get_scoring_maintenance_state = (
                    original_maintenance_reader
                )

            if called_indexes != [1]:
                raise RuntimeError("baseline started work beyond the pause boundary")
            resumed_rows = load_checkpoint()
            if [row.get("icp_ref") for row in resumed_rows] != [
                "rehearsal-checkpoint-icp-1"
            ]:
                raise RuntimeError("paused baseline checkpoint did not persist")

            async def unpaused() -> dict[str, Any]:
                return {"paused": False}

            scoring_worker_module.get_scoring_maintenance_state = unpaused
            try:
                final_rows, retry_stats = await worker._run_baseline_batch_inner(
                    runner=object(),
                    retry_runner=object(),
                    scorer=object(),
                    window=window,
                    run_start=time.time(),
                    resume_results=resumed_rows,
                    icp_checkpoint=checkpoint,
                )
            finally:
                scoring_worker_module.get_scoring_maintenance_state = (
                    original_maintenance_reader
                )
            if called_indexes != [1, 2]:
                raise RuntimeError("baseline resume duplicated a completed ICP")
            if len(final_rows) != 2 or retry_stats["unresolved"] != 0:
                raise RuntimeError("baseline did not finish after checkpoint resume")

        asyncio.run(exercise_pause_checkpoint_resume())
    finally:
        if prior_boto3 is None:
            sys.modules.pop("boto3", None)
        else:
            sys.modules["boto3"] = prior_boto3

    starting_capacity = vault.transient_capacity_state()
    topology = topology_document()
    production_wave_jobs = int(topology["benchmark_concurrency"])
    coordinator_bytes = (
        int(ROLE_SPECS[COORDINATOR_ROLE]["memory_mib"]) * 1024 * 1024
    )
    measured_envelopes_per_job = 5000
    measured_encoded_bytes_per_envelope = 64 * 1024
    vault._record_memory_bytes = lambda _record: (
        measured_encoded_bytes_per_envelope
    )
    for job_index in range(production_wave_jobs):
        for artifact_index in range(measured_envelopes_per_job):
            vault.seal(
                f"rebenchmark-{job_index}-{artifact_index}".encode(),
                job_id=f"rehearsal:rebenchmark-wave:{job_index}",
                purpose="research_lab.private_model_run.v2",
                artifact_kind="provider_response",
            )
    capacity = vault.transient_capacity_state()
    expected_count = (
        starting_capacity["transient_artifact_count"]
        + production_wave_jobs * measured_envelopes_per_job
    )
    if (
        capacity["transient_artifact_count"] != expected_count
        or capacity["transient_artifact_count"] <= 16384
        or capacity["transient_artifact_count"]
        >= capacity["maximum_transient_artifacts"]
        or capacity["transient_artifact_bytes"] <= 1024 * 1024 * 1024
        or capacity["transient_artifact_bytes"]
        >= capacity["maximum_transient_artifact_bytes"]
        or capacity["maximum_transient_artifact_bytes"] > coordinator_bytes // 2
        or capacity["active_artifact_job_count"] < production_wave_jobs
    ):
        raise RuntimeError("measured rebenchmark wave exhausted artifact capacity")
    return {
        "nonterminal_polls_live": True,
        "request_bound_cache_attempts": True,
        "execution_receipt_transport_unique": True,
        "transient_cache_transport_recovered": True,
        "transient_outcome_checkpoint_recovered": True,
        "atomic_provider_cache_put_replayed": True,
        "atomic_provider_outcome_batch_persisted": True,
        "measured_concurrent_artifact_wave_bound": True,
        "completed_provider_job_state_released": True,
        "active_provider_job_state_isolated": True,
        "provider_recovery_checkpoint_bound": True,
        "baseline_checkpoint_runtime_identity_bound": True,
        "baseline_checkpoint_receipt_ancestry_restored": True,
        "baseline_exact_receipt_frontier_bound": True,
        "baseline_pause_checkpoint_resume_complete": True,
        "baseline_v2_async_scheduler_bound": True,
        "provider_singleflight_commit_bound": True,
        "provider_cross_worker_cache_profile_bound": True,
        "provider_rpc_frame_budget_bound": True,
    }


BEHAVIOR_ACTIONS: dict[str, Callable[[], dict[str, Any]]] = {
    "signed-private-model-contract-transition": (
        _exercise_signed_private_model_contract_transition
    ),
    "chain-settlement-state-space": _exercise_chain_settlement_state_space,
    "conditional-icp-policy": _exercise_conditional_icp_policy,
    "conditional-candidate-gate": _exercise_conditional_candidate_gate,
    "git-tree-replacement": _exercise_git_tree_replacement,
    "model-sandbox-scope-binding": _exercise_model_sandbox_scope_binding,
    "rebenchmark-sandbox-retry": _exercise_rebenchmark_sandbox_retry_contract,
    "rebenchmark-provider-transport-evidence": (
        _exercise_rebenchmark_provider_transport_evidence
    ),
    "historical-metagraph-layouts": _exercise_historical_metagraph_layouts,
    "receipt-graph-aggregate-pagination": (
        _exercise_receipt_graph_aggregate_pagination
    ),
    "receipt-graph-transport-deduplication": (
        _exercise_receipt_graph_transport_deduplication
    ),
    "fresh-weight-input-lineage": _exercise_fresh_weight_input_lineage,
    "stateful-compact-graph-readback": (
        _exercise_stateful_compact_graph_readback
    ),
    "research-lab-allocation-conservation": (
        _exercise_research_lab_allocation_conservation
    ),
    "settlement-frontier-terminal-retirement": (
        _exercise_settlement_frontier_terminal_retirement
    ),
    "current-frontier-release-recovery": (
        _exercise_current_frontier_release_recovery
    ),
    "validator-publication-release-recovery": (
        _exercise_validator_publication_release_recovery
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("prepush", "release"), required=True)
    parser.add_argument("--candidate-sha", required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--boundary-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.candidate_sha) != 40 or any(
        value not in "0123456789abcdef" for value in args.candidate_sha
    ):
        parser.error("--candidate-sha must be a full lowercase Git SHA")
    expected_epochs = 1 if args.profile == "prepush" else 100
    if args.epochs != expected_epochs:
        parser.error(f"{args.profile} requires exactly {expected_epochs} epochs")

    stages: list[dict[str, Any]] = []
    fixture: dict[str, Any] | None = None
    boundary_contract: dict[str, Any] | None = None
    behavior_contract: dict[str, Any] | None = None

    def load_inputs() -> tuple[
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
    ]:
        loaded_fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
        loaded_boundary_contract = json.loads(
            args.boundary_contract.read_text(encoding="utf-8")
        )
        if loaded_fixture["sanitization"]["contains_production_credentials"]:
            raise RuntimeError("rehearsal fixture contains production credentials")
        if set(loaded_boundary_contract["forbidden_substitutions"]) != {
            "gateway",
            "validator",
            "auditor",
            "canonical_bundle",
            "receipt_graph",
            "signature",
            "sdk_extrinsic",
            "verification",
        }:
            raise RuntimeError("rehearsal substitution policy is incomplete")
        loaded_behavior_contract = validate_rehearsal_behavior_contract_v2(
            build_rehearsal_behavior_contract_v2(
                source_root=SOURCE_ROOT,
                candidate_sha=args.candidate_sha,
                profile=args.profile,
                epoch_count=args.epochs,
            )
        )
        if args.profile == "release" and list(
            loaded_fixture.get("fault_matrix") or []
        ) != loaded_behavior_contract["fault_ids"]:
            raise RuntimeError(
                "mounted fault matrix differs from candidate contract"
            )
        return (
            loaded_fixture,
            loaded_boundary_contract,
            loaded_behavior_contract,
        )

    inputs_passed, inputs = _run_workflow_stage(
        stage="input-contract",
        action=load_inputs,
        stages=stages,
    )
    if inputs_passed:
        fixture, boundary_contract, behavior_contract = inputs

    identities: list[dict[str, str]] = []
    source_paths = (
        list(behavior_contract["production_source_paths"])
        if behavior_contract is not None
        else []
    )
    for path in source_paths:
        passed, identity = _run_workflow_stage(
            stage=f"source-identity:{path}",
            action=lambda path=path: _file_identity(path, args.candidate_sha),
            stages=stages,
        )
        if passed:
            identities.append(identity)

    behavior_evidence: dict[str, Any] = {}
    behavior_scenarios = (
        list(behavior_contract["behavior_scenarios"])
        if behavior_contract is not None
        else []
    )
    for scenario in behavior_scenarios:
        action = BEHAVIOR_ACTIONS.get(scenario)
        if action is None:
            _run_workflow_stage(
                stage=f"behavior:{scenario}",
                action=lambda scenario=scenario: (_ for _ in ()).throw(
                    RuntimeError(
                        f"candidate behavior scenario has no runner: {scenario}"
                    )
                ),
                stages=stages,
            )
            continue
        passed, result = _run_workflow_stage(
            stage=f"behavior:{scenario}",
            action=action,
            stages=stages,
        )
        if passed:
            behavior_evidence[scenario] = result

    _run_independent_epoch_diagnostics(
        candidate_sha=args.candidate_sha,
        epoch_id=30_000,
        stages=stages,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    service_root = args.output.parent / "local-services"
    faults: list[dict[str, Any]] = []
    concurrent_writes = 0
    epochs: list[dict[str, Any]] = []
    boundary_events: list[dict[str, Any]] = []
    cleanup = {
        "pending_faults": 0,
        "boundary_thread_alive_before_close": False,
        "boundary_thread_alive_after_close": False,
        "local_chain_epochs": 0,
    }

    if fixture is None:
        if args.profile == "release":
            _mark_workflow_stage_unexercised(
                stage="fault-matrix",
                blocked_by=["input-contract"],
                stages=stages,
            )
            _mark_workflow_stage_unexercised(
                stage="concurrency",
                blocked_by=["input-contract"],
                stages=stages,
            )
        for ordinal in range(args.epochs):
            _mark_workflow_stage_unexercised(
                stage=f"epoch-{30_000 + ordinal}",
                blocked_by=["input-contract"],
                stages=stages,
            )
        _mark_workflow_stage_unexercised(
            stage="boundary-cleanup",
            blocked_by=["input-contract"],
            stages=stages,
        )
    else:
        if args.profile == "release":
            for ordinal, fault in enumerate(fixture["fault_matrix"]):
                def run_fault(
                    *,
                    ordinal: int = ordinal,
                    fault: str = str(fault),
                ) -> dict[str, Any]:
                    with LocalBoundaryServices(
                        root=service_root / f"fault-{ordinal:02d}",
                        fixture=fixture,
                    ) as fault_services:
                        return _exercise_fault(
                            fault_services,
                            fault=fault,
                            ordinal=ordinal,
                        )

                passed, result = _run_workflow_stage(
                    stage=f"fault:{ordinal}:{fault}",
                    action=run_fault,
                    stages=stages,
                )
                if passed:
                    faults.append(result)

            def run_concurrency() -> int:
                with LocalBoundaryServices(
                    root=service_root / "concurrency",
                    fixture=fixture,
                ) as concurrency_services:
                    return _exercise_concurrency(concurrency_services)

            passed, result = _run_workflow_stage(
                stage="concurrency",
                action=run_concurrency,
                stages=stages,
            )
            if passed:
                concurrent_writes = result

        services = LocalBoundaryServices(
            root=service_root / "epochs",
            fixture=fixture,
        )
        services_started, _ = _run_workflow_stage(
            stage="boundary-start",
            action=services.__enter__,
            stages=stages,
        )
        if services_started:
            try:
                first_epoch = 30_000
                for ordinal in range(args.epochs):
                    epoch_id = first_epoch + ordinal
                    passed, epoch = _run_workflow_stage(
                        stage=f"epoch-{epoch_id}",
                        action=lambda epoch_id=epoch_id: _run_epoch(
                            services=services,
                            fixture=fixture,
                            candidate_sha=args.candidate_sha,
                            epoch_id=epoch_id,
                        ),
                        stages=stages,
                    )
                    if passed:
                        epochs.append(epoch)
                boundary_events = list(services.state.events)
                cleanup = {
                    "pending_faults": len(services.state.faults),
                    "boundary_thread_alive_before_close": (
                        services.thread.is_alive()
                    ),
                    "boundary_thread_alive_after_close": True,
                    "local_chain_epochs": len(services.state.chain),
                }
            finally:
                cleanup_passed, _ = _run_workflow_stage(
                    stage="boundary-cleanup",
                    action=lambda: services.__exit__(None, None, None),
                    stages=stages,
                )
                cleanup["boundary_thread_alive_after_close"] = (
                    services.thread.is_alive()
                )
                cleanup["local_chain_epochs"] = len(services.state.chain)
                if not cleanup_passed:
                    cleanup["cleanup_failed"] = True
        else:
            for ordinal in range(args.epochs):
                _mark_workflow_stage_unexercised(
                    stage=f"epoch-{30_000 + ordinal}",
                    blocked_by=["boundary-start"],
                    stages=stages,
                )
            _mark_workflow_stage_unexercised(
                stage="boundary-cleanup",
                blocked_by=["boundary-start"],
                stages=stages,
            )

    validation_dependencies = [
        item["stage"] for item in stages if item.get("status") != "passed"
    ]
    stage_status = {
        str(item.get("stage")): str(item.get("status"))
        for item in stages
        if isinstance(item, Mapping)
    }
    duplicate_stage_ids = len(stage_status) != len(stages)
    expected_before_validation = (
        set(behavior_contract["required_stage_ids"])
        - {"workflow-evidence-validation"}
        if behavior_contract is not None
        else set()
    )
    observed_before_validation = set(stage_status)

    epoch_authority_complete = (
        len(epochs) == expected_epochs
        and all(
            epoch.get("canonical_vector_equal") is True
            and epoch.get("receipt_ancestry_verified") is True
            and epoch.get("auditor_verified") is True
            and epoch.get("auditor_runtime_verified") is True
            and epoch.get("sdk_bridge_verified") is True
            and bool(epoch.get("signed_extrinsic_hash"))
            and epoch.get("last_update") == epoch.get("finalized_block")
            for epoch in epochs
        )
    )
    identity_paths = [str(item.get("path")) for item in identities]
    identity_commits = {
        str(item.get("commit_sha")) for item in identities
    }
    boundary_definitions = (
        boundary_contract.get("boundaries")
        if isinstance(boundary_contract, Mapping)
        else None
    )
    unknown_boundaries_rejected = (
        isinstance(boundary_definitions, Mapping)
        and bool(boundary_definitions)
        and all(
            isinstance(definition, Mapping)
            and definition.get("reject_unknown") is True
            for definition in boundary_definitions.values()
        )
    )
    behavioral_invariants = {
        "candidate_identity_exact": (
            behavior_contract is not None
            and behavior_contract.get("candidate_sha") == args.candidate_sha
        ),
        "protected_source_identity_exact": (
            behavior_contract is not None
            and sorted(identity_paths)
            == sorted(behavior_contract["production_source_paths"])
            and identity_commits == {args.candidate_sha}
        ),
        "signed_private_model_contract_transition_exact": (
            behavior_evidence.get(
                "signed-private-model-contract-transition",
                {},
            ).get("rollback_exact")
            is True
            and behavior_evidence.get(
                "signed-private-model-contract-transition",
                {},
            ).get("pointer_source_mismatch_rejected")
            is True
            and behavior_evidence.get(
                "signed-private-model-contract-transition",
                {},
            ).get("contract_assertion_mismatch_rejected")
            is True
            and behavior_evidence.get(
                "signed-private-model-contract-transition",
                {},
            ).get("hybrid_sources_rejected")
            is True
            and behavior_evidence.get(
                "signed-private-model-contract-transition",
                {},
            ).get("hybrid_manifests_rejected")
            is True
            and behavior_evidence.get(
                "signed-private-model-contract-transition",
                {},
            ).get("tampered_source_rejected")
            is True
            and behavior_evidence.get(
                "signed-private-model-contract-transition",
                {},
            ).get("tampered_manifest_rejected")
            is True
            and behavior_evidence.get(
                "signed-private-model-contract-transition",
                {},
            ).get("unknown_source_rejected")
            is True
            and behavior_evidence.get(
                "signed-private-model-contract-transition",
                {},
            ).get("unknown_manifest_rejected")
            is True
            and behavior_evidence.get(
                "signed-private-model-contract-transition",
                {},
            ).get("invalid_signature_rejected")
            is True
            and behavior_evidence.get(
                "signed-private-model-contract-transition",
                {},
            ).get("lineage_rebenchmark_verified")
            is True
            and behavior_evidence.get(
                "signed-private-model-contract-transition",
                {},
            ).get("manifest_pending_reconciled")
            is True
            and behavior_evidence.get(
                "signed-private-model-contract-transition",
                {},
            ).get("manifest_reconcile_crosses_terminal_events")
            is True
            and behavior_evidence.get(
                "signed-private-model-contract-transition",
                {},
            ).get("manifest_reconcile_precedes_baseline")
            is True
            and behavior_evidence.get(
                "signed-private-model-contract-transition",
                {},
            ).get("signed_extensions_verified")
            is True
        ),
        "delayed_private_source_manifest_recovery_verified": (
            behavior_evidence.get(
                "signed-private-model-contract-transition",
                {},
            ).get("manifest_pending_reconciled")
            is True
            and behavior_evidence.get(
                "signed-private-model-contract-transition",
                {},
            ).get("manifest_reconcile_crosses_terminal_events")
            is True
            and behavior_evidence.get(
                "signed-private-model-contract-transition",
                {},
            ).get("manifest_reconcile_precedes_baseline")
            is True
            and behavior_evidence.get(
                "signed-private-model-contract-transition",
                {},
            ).get("candidate_owned_repo_head_sync_blocked")
            is True
        ),
        "model_sandbox_final_provider_cost_scope_bound": (
            behavior_evidence.get(
                "model-sandbox-scope-binding",
                {},
            ).get("preliminary_scope_rejected")
            is True
            and behavior_evidence.get(
                "model-sandbox-scope-binding",
                {},
            ).get("final_provider_cost_scope_bound")
            is True
            and behavior_evidence.get(
                "model-sandbox-scope-binding",
                {},
            ).get("model_source_import_isolated")
            is True
            and behavior_evidence.get(
                "model-sandbox-scope-binding",
                {},
            ).get("model_provider_broker_rootfs_path_bound")
            is True
            and behavior_evidence.get(
                "model-sandbox-scope-binding",
                {},
            ).get("model_sandbox_cgroup_delegated")
            is True
            and behavior_evidence.get(
                "model-sandbox-scope-binding",
                {},
            ).get("model_sandbox_rootful_launcher_bound")
            is True
            and behavior_evidence.get(
                "model-sandbox-scope-binding",
                {},
            ).get("model_signed_transport_failure_fallback_bound")
            is True
            and behavior_evidence.get(
                "model-sandbox-scope-binding",
                {},
            ).get("model_missing_transport_terminal_rejected")
            is True
            and behavior_evidence.get(
                "model-sandbox-scope-binding",
                {},
            ).get("model_transport_failure_client_semantics_bound")
            is True
        ),
        "rebenchmark_sandbox_failure_bounded_retry": (
            behavior_evidence.get(
                "rebenchmark-sandbox-retry",
                {},
            ).get("attested_ancestry_preserved")
            is True
            and behavior_evidence.get(
                "rebenchmark-sandbox-retry",
                {},
            ).get("private_runner_contract_preserved")
            is True
            and behavior_evidence.get(
                "rebenchmark-sandbox-retry",
                {},
            ).get("bounded_retry_selected")
            is True
            and behavior_evidence.get(
                "rebenchmark-sandbox-retry",
                {},
            ).get("generic_provider_contract_failure_terminal")
            is True
            and behavior_evidence.get(
                "rebenchmark-sandbox-retry",
                {},
            ).get("configured_runtime_deadline_bound")
            is True
            and behavior_evidence.get(
                "rebenchmark-sandbox-retry",
                {},
            ).get("configured_runtime_finalization_reserve_bound")
            is True
            and behavior_evidence.get(
                "rebenchmark-sandbox-retry",
                {},
            ).get("signed_http_retry_selected")
            is True
            and behavior_evidence.get(
                "rebenchmark-sandbox-retry",
                {},
            ).get("retry_checkpoint_recovery_bound")
            is True
            and behavior_evidence.get(
                "rebenchmark-sandbox-retry",
                {},
            ).get("content_addressed_artifact_persistence_bound")
            is True
            and behavior_evidence.get(
                "rebenchmark-sandbox-retry",
                {},
            ).get("missing_distinct_artifact_rejected")
            is True
        ),
        "rebenchmark_provider_transport_evidence_unique": (
            behavior_evidence.get(
                "rebenchmark-provider-transport-evidence",
                {},
            ).get("nonterminal_polls_live")
            is True
            and behavior_evidence.get(
                "rebenchmark-provider-transport-evidence",
                {},
            ).get("request_bound_cache_attempts")
            is True
            and behavior_evidence.get(
                "rebenchmark-provider-transport-evidence",
                {},
            ).get("execution_receipt_transport_unique")
            is True
            and behavior_evidence.get(
                "rebenchmark-provider-transport-evidence",
                {},
            ).get("transient_cache_transport_recovered")
            is True
            and behavior_evidence.get(
                "rebenchmark-provider-transport-evidence",
                {},
            ).get("transient_outcome_checkpoint_recovered")
            is True
            and behavior_evidence.get(
                "rebenchmark-provider-transport-evidence",
                {},
            ).get("measured_concurrent_artifact_wave_bound")
            is True
            and behavior_evidence.get(
                "rebenchmark-provider-transport-evidence",
                {},
            ).get("provider_rpc_frame_budget_bound")
            is True
            and behavior_evidence.get(
                "rebenchmark-provider-transport-evidence",
                {},
            ).get("provider_singleflight_commit_bound")
            is True
            and behavior_evidence.get(
                "rebenchmark-provider-transport-evidence",
                {},
            ).get("baseline_checkpoint_runtime_identity_bound")
            is True
            and behavior_evidence.get(
                "rebenchmark-provider-transport-evidence",
                {},
            ).get("baseline_checkpoint_receipt_ancestry_restored")
            is True
            and behavior_evidence.get(
                "rebenchmark-provider-transport-evidence",
                {},
            ).get("baseline_pause_checkpoint_resume_complete")
            is True
        ),
        "chain_settlement_state_space_complete": (
            "chain-settlement-state-space" in behavior_evidence
        ),
        "conditional_icp_policy_config_bound": (
            "conditional-icp-policy" in behavior_evidence
            and behavior_contract is not None
            and behavior_evidence["conditional-icp-policy"].get(
                "policy_hash"
            )
            == behavior_contract["policy_commitments"]["conditional_icp"].get(
                "policy_hash"
            )
        ),
        "conditional_candidate_advancement_exact": (
            "conditional-candidate-gate" in behavior_evidence
            and behavior_contract is not None
            and behavior_evidence["conditional-candidate-gate"].get(
                "policy_hash"
            )
            == behavior_contract["policy_commitments"]["conditional_icp"].get(
                "policy_hash"
            )
        ),
        "git_tree_replacement_deterministic": (
            "git-tree-replacement" in behavior_evidence
            and behavior_contract is not None
            and behavior_evidence["git-tree-replacement"].get("policy_hash")
            == behavior_contract["policy_commitments"]["git_tree"].get(
                "policy_hash"
            )
            and behavior_evidence["git-tree-replacement"].get(
                "restart_resume_verified"
            )
            is True
            and len(
                str(
                    behavior_evidence["git-tree-replacement"].get(
                        "root_git_commit"
                    )
                    or ""
                )
            )
            == 64
            and len(
                str(
                    behavior_evidence["git-tree-replacement"].get(
                        "node_git_commit"
                    )
                    or ""
                )
            )
            == 64
        ),
        "historical_metagraph_layouts_policy_bound": (
            "historical-metagraph-layouts" in behavior_evidence
            and behavior_contract is not None
            and behavior_evidence["historical-metagraph-layouts"].get(
                "policy_hash"
            )
            == behavior_contract["policy_commitments"]["chain_source"].get(
                "policy_hash"
            )
            and behavior_evidence["historical-metagraph-layouts"].get(
                "accepted_layouts"
            )
            == behavior_contract["policy_commitments"]["chain_source"][
                "policy"
            ].get("selective_result_last_fields")
        ),
        "receipt_graph_aggregate_evidence_paged": (
            behavior_evidence.get(
                "receipt-graph-aggregate-pagination",
                {},
            ).get("aggregate_evidence_paged")
            is True
            and behavior_evidence.get(
                "receipt-graph-aggregate-pagination",
                {},
            ).get("structural_limit_enforced")
            is True
            and behavior_evidence.get(
                "receipt-graph-aggregate-pagination",
                {},
            ).get("checkpoint_parent_first_persistence")
            is True
        ),
        "receipt_graph_transport_deduplicated_and_verified": (
            behavior_evidence.get(
                "receipt-graph-transport-deduplication",
                {},
            ).get("exact_job_path_verified")
            is True
            and behavior_evidence.get(
                "receipt-graph-transport-deduplication",
                {},
            ).get("malformed_evidence_rejected")
            is True
            and behavior_evidence.get(
                "receipt-graph-transport-deduplication",
                {},
            ).get("ordinary_graph_bound_preserved")
            is True
            and behavior_evidence.get(
                "receipt-graph-transport-deduplication",
                {},
            ).get("historical_checkpoint_issuer_included")
            is True
            and behavior_evidence.get(
                "receipt-graph-transport-deduplication",
                {},
            ).get("transport_size_bytes", 1)
            < behavior_evidence.get(
                "receipt-graph-transport-deduplication",
                {},
            ).get("legacy_size_bytes", 0)
        ),
        "fresh_weight_input_lineage_verified": (
            behavior_evidence.get(
                "fresh-weight-input-lineage", {}
            ).get("fresh_checkpoint_lineage_accepted")
            is True
            and behavior_evidence.get(
                "fresh-weight-input-lineage", {}
            ).get("direct_execution_proof_selected")
            is True
            and behavior_evidence.get(
                "fresh-weight-input-lineage", {}
            ).get("replay_identity_equal")
            is True
            and behavior_evidence.get(
                "fresh-weight-input-lineage", {}
            ).get("direct_receipts_persisted")
            is True
            and behavior_evidence.get(
                "fresh-weight-input-lineage", {}
            ).get("mismatched_execution_rejected")
            is True
        ),
        "stateful_compact_graph_readback_verified": (
            behavior_evidence.get(
                "stateful-compact-graph-readback", {}
            ).get("checkpoint_v3_persisted")
            is True
            and behavior_evidence.get(
                "stateful-compact-graph-readback", {}
            ).get("canonical_v4_readback_accepted")
            is True
            and behavior_evidence.get(
                "stateful-compact-graph-readback", {}
            ).get("boundary_persisted")
            is True
            and behavior_evidence.get(
                "stateful-compact-graph-readback", {}
            ).get("snapshot_persisted")
            is True
            and behavior_evidence.get(
                "stateful-compact-graph-readback", {}
            ).get("tampered_v4_rejected_before_write")
            is True
        ),
        "research_lab_allocation_policy_config_bound": (
            "research-lab-allocation-conservation" in behavior_evidence
            and behavior_contract is not None
            and behavior_evidence[
                "research-lab-allocation-conservation"
            ].get("policy_hash")
            == behavior_contract["policy_commitments"][
                "research_lab_allocation"
            ].get("policy_hash")
        ),
        "research_lab_allocation_conserved": (
            behavior_evidence.get(
                "research-lab-allocation-conservation",
                {},
            ).get("conserved")
            is True
        ),
        "settlement_frontier_terminal_retirement_verified": (
            behavior_evidence.get(
                "settlement-frontier-terminal-retirement",
                {},
            ).get("original_failure_reproduced")
            is True
            and behavior_evidence.get(
                "settlement-frontier-terminal-retirement",
                {},
            ).get("champion_terminal_retired")
            is True
            and behavior_evidence.get(
                "settlement-frontier-terminal-retirement",
                {},
            ).get("source_add_terminal_retired")
            is True
            and behavior_evidence.get(
                "settlement-frontier-terminal-retirement",
                {},
            ).get("tampered_identity_rejected")
            is True
            and behavior_evidence.get(
                "settlement-frontier-terminal-retirement",
                {},
            ).get("canonical_receipt_without_release_hash_accepted")
            is True
            and behavior_evidence.get(
                "settlement-frontier-terminal-retirement",
                {},
            ).get("execution_release_hash_validated")
            is True
        ),
        "current_frontier_release_recovery_verified": (
            behavior_evidence.get(
                "current-frontier-release-recovery",
                {},
            ).get("cross_release_execution_skipped")
            is True
            and behavior_evidence.get(
                "current-frontier-release-recovery",
                {},
            ).get("exact_signed_authority_reused")
            is True
            and behavior_evidence.get(
                "current-frontier-release-recovery",
                {},
            ).get("immutable_frontier_preserved")
            is True
            and behavior_evidence.get(
                "current-frontier-release-recovery",
                {},
            ).get("malformed_release_rejected")
            is True
        ),
        "validator_publication_release_recovery_verified": (
            behavior_evidence.get(
                "validator-publication-release-recovery",
                {},
            ).get("approved_n_minus_one_recovered")
            is True
            and behavior_evidence.get(
                "validator-publication-release-recovery",
                {},
            ).get("nitro_attestation_rechecked")
            is True
            and behavior_evidence.get(
                "validator-publication-release-recovery",
                {},
            ).get("release_tampering_rejected")
            is True
            and behavior_evidence.get(
                "validator-publication-release-recovery",
                {},
            ).get("same_release_config_mismatch_rejected")
            is True
            and behavior_evidence.get(
                "validator-publication-release-recovery",
                {},
            ).get("cross_release_finalization_only")
            is True
            and behavior_evidence.get(
                "validator-publication-release-recovery",
                {},
            ).get("unsigned_cross_release_rejected")
            is True
            and behavior_evidence.get(
                "validator-publication-release-recovery",
                {},
            ).get("implicit_cross_release_rejected")
            is True
            and behavior_evidence.get(
                "validator-publication-release-recovery",
                {},
            ).get("same_epoch_finalized_journal_retained")
            is True
            and behavior_evidence.get(
                "validator-publication-release-recovery",
                {},
            ).get("next_epoch_finalized_journal_retired")
            is True
        ),
        "canonical_vector_primary_auditor_equal": (
            epoch_authority_complete
            and all(
                epoch.get("canonical_vector_equal") is True
                for epoch in epochs
            )
        ),
        "receipt_ancestry_verified": (
            epoch_authority_complete
            and all(
                epoch.get("receipt_ancestry_verified") is True
                for epoch in epochs
            )
        ),
        "sdk_signing_bridge_verified": (
            epoch_authority_complete
            and all(
                epoch.get("sdk_bridge_verified") is True
                for epoch in epochs
            )
        ),
        "submission_finalized": (
            epoch_authority_complete
            and all(bool(epoch.get("signed_extrinsic_hash")) for epoch in epochs)
        ),
        "last_update_readback_equal": (
            epoch_authority_complete
            and all(
                epoch.get("last_update") == epoch.get("finalized_block")
                for epoch in epochs
            )
        ),
        "boundary_cleanup_complete": (
            cleanup["pending_faults"] == 0
            and cleanup["boundary_thread_alive_after_close"] is False
            and cleanup["local_chain_epochs"] == expected_epochs
        ),
        "unknown_boundaries_rejected": unknown_boundaries_rejected,
    }

    def validate_workflow_evidence() -> None:
        if behavior_contract is None:
            raise RuntimeError("candidate behavior contract is unavailable")
        if duplicate_stage_ids:
            raise RuntimeError("workflow emitted duplicate stage evidence")
        if observed_before_validation != expected_before_validation:
            missing = sorted(
                expected_before_validation - observed_before_validation
            )
            unexpected = sorted(
                observed_before_validation - expected_before_validation
            )
            raise RuntimeError(
                "workflow stage contract differs "
                f"missing={missing} unexpected={unexpected}"
            )
        required_invariants = set(
            behavior_contract["required_invariant_ids"]
        )
        if set(behavioral_invariants) != required_invariants:
            raise RuntimeError("workflow invariant contract differs")
        failed_invariants = sorted(
            name
            for name, passed in behavioral_invariants.items()
            if passed is not True
        )
        if failed_invariants:
            raise RuntimeError(
                "joined V2 workflow invariants failed: "
                + ",".join(failed_invariants)
            )
        if args.profile == "release" and (
            len(faults) != len(behavior_contract["fault_ids"])
            or concurrent_writes != 32
        ):
            raise RuntimeError("release fault or concurrency evidence is incomplete")

    if validation_dependencies:
        _mark_workflow_stage_unexercised(
            stage="workflow-evidence-validation",
            blocked_by=validation_dependencies,
            stages=stages,
        )
    else:
        _run_workflow_stage(
            stage="workflow-evidence-validation",
            action=validate_workflow_evidence,
            stages=stages,
        )

    status = (
        "passed"
        if all(item.get("status") == "passed" for item in stages)
        else "failed"
    )
    manifest = {
        "schema_version": "leadpoet.local_v2_workflow_evidence.v1",
        "status": status,
        "profile": args.profile,
        "release_sha": args.candidate_sha,
        "fixture_hash": sha256_json(fixture) if fixture is not None else None,
        "boundary_contract_hash": (
            sha256_json(boundary_contract)
            if boundary_contract is not None
            else None
        ),
        "behavior_contract": behavior_contract,
        "behavior_contract_hash": (
            behavior_contract.get("contract_hash")
            if behavior_contract is not None
            else None
        ),
        "behavior_evidence": behavior_evidence,
        "behavioral_invariants": behavioral_invariants,
        "production_source_identities": identities,
        "epoch_count": len(epochs),
        "epochs": epochs,
        "fault_matrix": faults,
        "concurrent_write_count": concurrent_writes,
        "boundary_event_count": len(boundary_events),
        "cleanup": cleanup,
        "stages": stages,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    args.output.write_bytes(_canonical(manifest) + b"\n")
    if status != "passed":
        failed = sum(item.get("status") == "failed" for item in stages)
        unexercised = sum(
            item.get("status") == "unexercised" for item in stages
        )
        print(
            "PRODUCTION_WORKFLOW_REHEARSAL_FAILED "
            f"profile={args.profile} failed={failed} "
            f"unexercised={unexercised} evidence={args.output}",
            file=sys.stderr,
            flush=True,
        )
        return 1
    print(
        f"PRODUCTION_WORKFLOW_REHEARSAL_SUCCESS profile={args.profile} "
        f"epochs={len(epochs)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
