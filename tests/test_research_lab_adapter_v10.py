from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import textwrap

import pytest

import research_lab.sourcing_model_contract_check as compatibility


_MAIN_V68_CONTRACT_PATH = compatibility.CONTRACT_V68_A6F_PATH
_MAIN_V28_PARITY_PATH = compatibility.PARITY_FIXTURE_V28_A6F_PATH
_MAIN_PROFILE_ID = "typed-dispatch-v68-dddf30c2"
_MAIN_COMMIT = "a6ffd17a607e715e8f31bba771ebd8a09878e745"
_LAB_COMMIT = "fae925b6de1562456d425dc11fb2ef8e23f75073"
_LIVE_LAB_SIGNED_MANIFEST = {
    "model_artifact_hash": (
        "sha256:23950f65f06a51b14feb6848d3c75ee77ad3461587cd53cb149786851500cf86"
    ),
    "git_commit_sha": _LAB_COMMIT,
    "manifest_hash": (
        "sha256:c781fd6f8a3238b24a71dace602e1a4127a6a6abaeb1df7d7970b9a36f956a47"
    ),
    "image_digest": (
        "493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@"
        "sha256:4dd70be3cd5d74c0f809faaec7ec2ecc1cae2faaca7ce7c5274fadcde45a7e48"
    ),
    "config_hash": (
        "sha256:653bfdd96f81131d8dd8f13caae6bf5ab25550391273d01681d7729e12562dd8"
    ),
    "scoring_adapter_version": "qualification-company-scorer:v2",
    "compatibility_contract": {
        "contract_id": "leadpoet-sourcing-wrapper-contract-v68",
        "path": "sourcing_model/consumer_contract.json",
        "sha256": (
            "sha256:2b3f59c8f14a55ca4b6e7d3d7bc32d4e574be4d5de4b238676753e2a413a24b7"
        ),
    },
    "consumer_parity_fixtures": {
        "path": "sourcing_model/consumer_parity_fixtures.json",
        "sha256": (
            "sha256:b9c33341726f4d16edef5a656bf4f62461a631ab3b415aa00102598c0e56551c"
        ),
    },
    "manifest_uri": (
        "s3://leadpoet-private-model-artifacts-493765492819/research-lab/"
        "sourcing-model/fae925b6de1562456d425dc11fb2ef8e23f75073.json"
    ),
    "signature_ref": (
        "s3://leadpoet-private-model-artifacts-493765492819/research-lab/"
        "sourcing-model/fae925b6de1562456d425dc11fb2ef8e23f75073.sig.b64"
    ),
}


def _expected_v10_dispatch_custody_metadata() -> dict[str, object]:
    policy = compatibility.semantic_compatibility_policy_v1()[
        "additive_dispatch_custody_v3"
    ]["metadata_binding"]
    projection = json.loads(
        compatibility.PARITY_FIXTURE_V28_PATH.read_text(encoding="utf-8")
    )["expected_model_runner_custody_v3_projection"]
    expected = {
        key: deepcopy(projection[key])
        for key in policy["required_model_keys"]
    }
    expected.update(
        {
            "completion_included": False,
            "initial_dispatch_entrypoint": (
                "dispatch_runner_initial_custody_v3"
            ),
            "initial_dispatch_schema_version": (
                "model-runner-custody:v3-initial-dispatch"
            ),
            "start_entrypoint": "build_runner_start_custody_v3",
            "start_validation_entrypoint": (
                "validate_runner_start_custody_v3"
            ),
            "action_entrypoint": "build_runner_action_custody_v3",
            "action_validation_entrypoint": (
                "validate_runner_action_custody_v3"
            ),
            "continuation_entrypoint": (
                "build_runner_initial_continuation_custody_v3"
            ),
            "continuation_validation_entrypoint": (
                "validate_runner_initial_continuation_custody_v3"
            ),
        }
    )
    assert set(expected) == set(policy["required_dispatch_keys"])
    return expected


def test_v10_reviewed_snapshots_are_byte_exact() -> None:
    assert compatibility._snapshot_sha256(compatibility.CONTRACT_V68_PATH) == (
        compatibility.ADDITIVE_DISPATCH_CUSTODY_V3_CONTRACT_SHA256
    )
    assert compatibility._snapshot_sha256(compatibility.PARITY_FIXTURE_V28_PATH) == (
        compatibility.ADDITIVE_DISPATCH_CUSTODY_V3_PARITY_SHA256
    )
    assert compatibility._snapshot_sha256(
        _MAIN_V68_CONTRACT_PATH
    ) == (
        "sha256:dddf30c25b3d53b7c8db5f20114d80a40d62ce20598c17ede7a2458399b2a075"
    )
    assert compatibility._snapshot_sha256(
        _MAIN_V28_PARITY_PATH
    ) == (
        "sha256:674d3ad18239fe4f6be1181a9f1adc476734cea4000ad46d2358971feec5d028"
    )
    contract = compatibility._safe_json_document(
        compatibility.CONTRACT_V68_PATH,
        label="v10 contract",
        violations=[],
    )
    parity = compatibility._safe_json_document(
        compatibility.PARITY_FIXTURE_V28_PATH,
        label="v10 parity",
        violations=[],
    )
    assert contract["contract_id"] == "leadpoet-sourcing-wrapper-contract-v68"
    assert parity["fixture_set_id"] == "routerverse-cross-consumer-parity-v28"


def test_v10_reviewed_profiles_bind_exact_generations() -> None:
    policy = compatibility.semantic_compatibility_policy_v1()
    profiles = compatibility._typed_dispatch_reviewed_profiles_v1(policy)

    assert {profile["profile_id"] for profile in profiles} == {
        "typed-dispatch-v68-base",
        _MAIN_PROFILE_ID,
    }
    current = compatibility._typed_dispatch_profile_for_pair_v1(
        policy,
        contract_id="leadpoet-sourcing-wrapper-contract-v68",
        contract_sha256=(
            "sha256:dddf30c25b3d53b7c8db5f20114d80a40d62ce20598c17ede7a2458399b2a075"
        ),
        parity_sha256=(
            "sha256:674d3ad18239fe4f6be1181a9f1adc476734cea4000ad46d2358971feec5d028"
        ),
    )

    assert current is not None
    assert current["profile_id"] == _MAIN_PROFILE_ID


def test_v10_reviewed_profile_registry_rejects_duplicates() -> None:
    policy = deepcopy(compatibility.semantic_compatibility_policy_v1())
    profiles = policy["additive_dispatch_custody_v3"][
        "additional_reviewed_profiles"
    ]
    profiles.append(deepcopy(profiles[0]))

    with pytest.raises(
        ValueError,
        match="typed dispatch profiles are ambiguous",
    ):
        compatibility._typed_dispatch_reviewed_profiles_v1(policy)


def test_v10_reviewed_profile_registry_is_bounded() -> None:
    policy = deepcopy(compatibility.semantic_compatibility_policy_v1())
    profiles = policy["additive_dispatch_custody_v3"][
        "additional_reviewed_profiles"
    ]
    profiles[:] = [
        deepcopy(profiles[0])
        for _ in range(
            compatibility.MAX_ADDITIONAL_TYPED_DISPATCH_PROFILES + 1
        )
    ]

    with pytest.raises(
        ValueError,
        match="typed dispatch profiles are invalid",
    ):
        compatibility._typed_dispatch_reviewed_profiles_v1(policy)


def test_v10_reviewed_profile_rejects_unreviewed_fields() -> None:
    policy = deepcopy(compatibility.semantic_compatibility_policy_v1())
    profile = policy["additive_dispatch_custody_v3"][
        "additional_reviewed_profiles"
    ][0]
    profile["callables"] = {}
    profile["profile_sha256"] = compatibility._sha256_json(
        {
            key: value
            for key, value in profile.items()
            if key != "profile_sha256"
        }
    )

    with pytest.raises(
        ValueError,
        match="typed dispatch profile is invalid",
    ):
        compatibility._typed_dispatch_reviewed_profiles_v1(policy)


@pytest.mark.parametrize(
    ("contract_snapshot", "parity_snapshot"),
    (
        (
            compatibility.CONTRACT_V68_PATH,
            _MAIN_V28_PARITY_PATH,
        ),
        (
            _MAIN_V68_CONTRACT_PATH,
            compatibility.PARITY_FIXTURE_V28_PATH,
        ),
    ),
)
def test_v10_cross_generation_contract_parity_pair_fails_closed(
    tmp_path: Path,
    contract_snapshot: Path,
    parity_snapshot: Path,
) -> None:
    contract_path = tmp_path / "sourcing_model/consumer_contract.json"
    parity_path = tmp_path / "sourcing_model/consumer_parity_fixtures.json"
    contract_path.parent.mkdir(parents=True)
    contract_path.write_bytes(contract_snapshot.read_bytes())
    parity_path.write_bytes(parity_snapshot.read_bytes())

    violations, receipt = compatibility.verify_semantic_source_tree_compatibility_v1(
        tmp_path
    )

    assert receipt is None
    assert "typed dispatch reviewed profile differs" in violations


def test_v10_profile_hash_binds_semantic_override_hashes() -> None:
    policy = deepcopy(compatibility.semantic_compatibility_policy_v1())
    profile = policy["additive_dispatch_custody_v3"][
        "additional_reviewed_profiles"
    ][0]
    profile["critical_binding_slices"][
        "sourcing_model/model_runner.py"
    ]["sha256"] = "sha256:" + "0" * 64

    with pytest.raises(
        ValueError,
        match="typed dispatch profile identity is invalid",
    ):
        compatibility._typed_dispatch_reviewed_profiles_v1(policy)


def test_v10_receipt_binds_the_exact_reviewed_profile() -> None:
    policy, policy_hash = compatibility.semantic_compatibility_policy_identity_v1()
    profile = compatibility._typed_dispatch_profile_for_pair_v1(
        policy,
        contract_id="leadpoet-sourcing-wrapper-contract-v68",
        contract_sha256=compatibility._snapshot_sha256(
            _MAIN_V68_CONTRACT_PATH
        ),
        parity_sha256=compatibility._snapshot_sha256(_MAIN_V28_PARITY_PATH),
    )
    assert profile is not None
    contract = json.loads(_MAIN_V68_CONTRACT_PATH.read_text(encoding="utf-8"))
    receipt = compatibility._semantic_compatibility_receipt(
        mode="semantic_v1",
        consumer_api_version=policy["consumer_api_version"],
        policy_hash=policy_hash,
        source_tree_hash="",
        manifest={},
        contract=contract,
        contract_hash=profile["contract_sha256"],
        parity_hash=profile["parity_sha256"],
        bindings={"typed_dispatch_profile_id": profile["profile_id"]},
    )
    assert compatibility.validate_source_tree_compatibility_receipt_v1(
        receipt,
        manifest={},
        source_tree_hash="",
        policy=policy,
        policy_hash=policy_hash,
    )["bindings"]["typed_dispatch_profile_id"] == profile["profile_id"]

    forged = deepcopy(receipt)
    forged["bindings"]["typed_dispatch_profile_id"] = "typed-dispatch-v68-base"
    forged["receipt_hash"] = compatibility._sha256_json(
        {
            key: value
            for key, value in forged.items()
            if key != "receipt_hash"
        }
    )
    with pytest.raises(ValueError, match="differs from signed artifact"):
        compatibility.validate_source_tree_compatibility_receipt_v1(
            forged,
            manifest={},
            source_tree_hash="",
            policy=policy,
            policy_hash=policy_hash,
        )


def test_v10_current_lab_release_cannot_fill_site_champion_slot() -> None:
    from research_lab.candidate_routing_experiments import _candidate_variant
    from research_lab.routing_experiments import RoutingExperimentError
    from tests.test_candidate_routing_experiments import (
        _artifact_manifest,
        _spec,
    )

    spec = _spec()
    baseline = next(
        item for item in spec.variants if item.variant_id == "baseline"
    )
    lab_artifact = replace(
        baseline.artifact,
        branch="leadpoet-lab",
        commit_sha=_LAB_COMMIT,
    )
    lab_baseline = replace(
        baseline,
        artifact=lab_artifact,
        artifact_authority_manifest=_artifact_manifest(lab_artifact),
    )
    lab_as_champion = replace(
        spec,
        variants=tuple(
            lab_baseline if item.variant_id == "baseline" else item
            for item in spec.variants
        ),
    )

    # ABI admission deliberately does not own Git branch history. The
    # candidate-composition authority independently keeps the live Lab pointer
    # in the challenger slot until that release is promoted to main.
    with pytest.raises(
        RoutingExperimentError,
        match="candidate_artifact_branch_must_be_main",
    ):
        _candidate_variant(lab_as_champion, "baseline")


def _extract_exact_model_tree(
    *, repo: Path, commit_sha: str, destination: Path
) -> None:
    result = subprocess.run(
        ["git", "-C", str(repo), "archive", commit_sha],
        check=True,
        capture_output=True,
        timeout=30,
    )
    with tarfile.open(fileobj=BytesIO(result.stdout), mode="r:") as archive:
        members = archive.getmembers()
        assert all(
            not Path(member.name).is_absolute()
            and ".." not in Path(member.name).parts
            for member in members
        )
        archive.extractall(destination, members=members)


def test_v10_exact_published_main_tree_is_admitted_by_semantic_v1(
    tmp_path: Path,
) -> None:
    repo_value = os.environ.get("LEADPOET_TEST_SOURCING_MODEL_REPO", "")
    if not repo_value:
        pytest.skip("exact Sourcing_model Git tree was not provided")
    repo = Path(repo_value).resolve()
    root = tmp_path / "exact-main"
    root.mkdir()
    _extract_exact_model_tree(
        repo=repo,
        commit_sha=_MAIN_COMMIT,
        destination=root,
    )

    receipt = compatibility.source_tree_compatibility_admission_v1(root)

    assert receipt["decision"] == "accepted"
    assert receipt["admission_mode"] == "semantic_v1"
    assert receipt["bindings"]["typed_dispatch_profile_id"] == _MAIN_PROFILE_ID


def test_v10_semantic_v1_rejects_any_declared_protocol_v2_tree(
    tmp_path: Path,
) -> None:
    (tmp_path / "research_lab_adapter.py").write_text(
        "def run_icp_outcome(icp, context=None):\n    return {}\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=(
            "qualification protocol v2 source must use unified compatibility admission"
        ),
    ):
        compatibility.source_tree_compatibility_admission_v1(tmp_path)


def test_v10_exact_lab_protocol_v2_tree_cannot_downgrade_to_semantic_v1(
    tmp_path: Path,
) -> None:
    repo_value = os.environ.get("LEADPOET_TEST_SOURCING_MODEL_REPO", "")
    if not repo_value:
        pytest.skip("exact Sourcing_model Git tree was not provided")
    root = tmp_path / "lab-v1-downgrade"
    root.mkdir()
    _extract_exact_model_tree(
        repo=Path(repo_value).resolve(),
        commit_sha=_LAB_COMMIT,
        destination=root,
    )

    with pytest.raises(
        ValueError,
        match=(
            "qualification protocol v2 source must use unified compatibility admission"
        ),
    ):
        compatibility.source_tree_compatibility_admission_v1(root)


def test_v10_exact_live_lab_tree_routes_through_signed_v2_manifest(
    tmp_path: Path,
) -> None:
    repo_value = os.environ.get("LEADPOET_TEST_SOURCING_MODEL_REPO", "")
    if not repo_value:
        pytest.skip("exact Sourcing_model Git tree was not provided")
    root = tmp_path / "live-lab-v2"
    root.mkdir()
    _extract_exact_model_tree(
        repo=Path(repo_value).resolve(),
        commit_sha=_LAB_COMMIT,
        destination=root,
    )
    observed_hash = compatibility.compute_compatibility_source_tree_hash_v1(root)
    assert observed_hash == _LIVE_LAB_SIGNED_MANIFEST["model_artifact_hash"]

    receipt = compatibility.source_tree_compatibility_admission(
        root,
        manifest=_LIVE_LAB_SIGNED_MANIFEST,
        source_tree_hash=observed_hash,
    )
    validated = compatibility.validate_source_tree_compatibility_receipt(
        receipt,
        manifest=_LIVE_LAB_SIGNED_MANIFEST,
        source_tree_hash=observed_hash,
    )

    assert validated == receipt
    assert receipt["decision"] == "accepted"
    assert receipt["admission_mode"] == "qualification_protocol_v2"
    assert receipt["git_commit_sha"] == _LAB_COMMIT
    assert receipt["manifest_hash"] == _LIVE_LAB_SIGNED_MANIFEST["manifest_hash"]
    assert receipt["image_digest"] == _LIVE_LAB_SIGNED_MANIFEST["image_digest"]
    assert receipt["contract_hash"] == _LIVE_LAB_SIGNED_MANIFEST[
        "compatibility_contract"
    ]["sha256"]
    assert receipt["parity_hash"] == _LIVE_LAB_SIGNED_MANIFEST[
        "consumer_parity_fixtures"
    ]["sha256"]
    assert receipt["bindings"] == {
        "scoring_adapter_version": "qualification-company-scorer:v2"
    }


@pytest.mark.parametrize(
    ("manifest_section", "foreign_hash"),
    (
        (
            "compatibility_contract",
            "sha256:dddf30c25b3d53b7c8db5f20114d80a40d62ce20598c17ede7a2458399b2a075",
        ),
        (
            "consumer_parity_fixtures",
            "sha256:674d3ad18239fe4f6be1181a9f1adc476734cea4000ad46d2358971feec5d028",
        ),
    ),
)
def test_v10_exact_lab_manifest_rejects_main_contract_parity_crossover(
    tmp_path: Path,
    manifest_section: str,
    foreign_hash: str,
) -> None:
    repo_value = os.environ.get("LEADPOET_TEST_SOURCING_MODEL_REPO", "")
    if not repo_value:
        pytest.skip("exact Sourcing_model Git tree was not provided")
    root = tmp_path / manifest_section
    root.mkdir()
    _extract_exact_model_tree(
        repo=Path(repo_value).resolve(),
        commit_sha=_LAB_COMMIT,
        destination=root,
    )
    observed_hash = compatibility.compute_compatibility_source_tree_hash_v1(root)
    hybrid_manifest = deepcopy(_LIVE_LAB_SIGNED_MANIFEST)
    hybrid_manifest[manifest_section]["sha256"] = foreign_hash

    with pytest.raises(
        ValueError,
        match="signed consumer documents differ from source",
    ):
        compatibility.source_tree_compatibility_admission(
            root,
            manifest=hybrid_manifest,
            source_tree_hash=observed_hash,
        )


def test_v10_policy_freezes_complete_typed_dispatch_surface() -> None:
    policy = compatibility.semantic_compatibility_policy_v1()
    dispatch = policy["additive_dispatch_custody_v3"]
    adapter = dispatch["callables"]["research_lab_adapter.py"]
    model_runner = dispatch["callables"]["sourcing_model/model_runner.py"]
    assert {
        "adapter_metadata",
        "build_runner_start_custody_v3",
        "build_runner_action_custody_v3",
        "build_runner_initial_continuation_custody_v3",
        "dispatch_runner_initial_custody_v3",
        "project_icp_request",
        "validate_runner_start_custody_v3",
        "validate_runner_action_custody_v3",
        "validate_runner_initial_continuation_custody_v3",
    } <= set(adapter)
    adapter_roots = set(
        dispatch["critical_binding_slices"]["research_lab_adapter.py"]["roots"]
    )
    assert {"project_icp_request", "run_icp"} <= adapter_roots
    assert {
        "custody_typed_encode",
        "custody_json_loads",
        "custody_json_dumps",
        "custody_json_bytes",
        "model_runner_custody_metadata",
        "model_runner_custody_parity_vectors",
        "build_model_start_request_custody_v3",
        "validate_model_start_request_custody_v3",
    } <= set(model_runner)
    assert dispatch["exact_constants"]["sourcing_model/model_runner.py"][
        "MODEL_RUNNER_CUSTODY_INTEGER_MIN"
    ] == -9007199254740991
    assert dispatch["exact_constants"]["sourcing_model/model_runner.py"][
        "MODEL_RUNNER_CUSTODY_INTEGER_MAX"
    ] == 9007199254740991
    assert dispatch["exact_constants"]["sourcing_model/routing/compiler.py"][
        "COMPILER_VERSION"
    ] == compatibility.ADDITIVE_DISPATCH_CUSTODY_V3_ROUTING_COMPILER_VERSION
    assert set(dispatch["metadata_binding"]["required_model_keys"]) <= set(
        dispatch["metadata_binding"]["required_dispatch_keys"]
    )
    assert dispatch["metadata_binding"]["dispatch_metadata_sha256"] == (
        compatibility.ADDITIVE_DISPATCH_CUSTODY_V3_METADATA_SHA256
    )
    assert len(dispatch["metadata_binding"]["required_dispatch_keys"]) == len(
        set(dispatch["metadata_binding"]["required_dispatch_keys"])
    )


def test_v10_policy_preserves_complete_reviewed_runner_abis() -> None:
    contract = json.loads(
        compatibility.CONTRACT_V68_PATH.read_text(encoding="utf-8")
    )
    relative = "sourcing_model/model_runner.py"
    reviewed = contract["functions"][relative]
    policy = compatibility.semantic_compatibility_policy_v1()[
        "additive_dispatch_custody_v3"
    ]["callables"][relative]

    assert set(policy) == set(reviewed)
    for name, positional in reviewed.items():
        contract_key = f"{relative}:{name}"
        assert policy[name]["positional"] == positional
        assert policy[name]["full_parameters"] == contract["full_parameters"][
            contract_key
        ]
        assert policy[name]["required_keyword_only"] == contract[
            "required_keyword_only"
        ].get(contract_key, [])
        assert policy[name]["is_async"] is contract["frozen_asyncness"].get(
            contract_key,
            False,
        )

    full_policy = compatibility.semantic_compatibility_policy_v1()
    reviewed_adapter = contract["functions"]["research_lab_adapter.py"]
    adapter_policy = dict(full_policy["callables"]["research_lab_adapter.py"])
    adapter_policy.update(
        full_policy["additive_dispatch_custody_v3"]["callables"][
            "research_lab_adapter.py"
        ]
    )
    assert set(adapter_policy) == set(reviewed_adapter)
    for name, positional in reviewed_adapter.items():
        assert adapter_policy[name]["positional"] == positional


def test_v3_marker_with_mutated_snapshot_fails_closed(tmp_path: Path) -> None:
    contract_path = tmp_path / "sourcing_model/consumer_contract.json"
    parity_path = tmp_path / "sourcing_model/consumer_parity_fixtures.json"
    contract_path.parent.mkdir(parents=True)
    contract_path.write_bytes(compatibility.CONTRACT_V68_PATH.read_bytes())
    parity_path.write_bytes(compatibility.PARITY_FIXTURE_V28_PATH.read_bytes())
    contract_path.write_bytes(contract_path.read_bytes().replace(b"v68", b"v69", 1))

    violations, receipt = compatibility.verify_semantic_source_tree_compatibility_v1(
        tmp_path
    )

    assert receipt is None
    assert "typed dispatch contract snapshot differs" in violations
    assert "typed dispatch contract identity is not approved" in violations


def test_v3_malformed_exact_signatures_fail_closed(tmp_path: Path) -> None:
    contract_path = tmp_path / "sourcing_model/consumer_contract.json"
    parity_path = tmp_path / "sourcing_model/consumer_parity_fixtures.json"
    contract_path.parent.mkdir(parents=True)
    contract = json.loads(
        compatibility.CONTRACT_V68_PATH.read_text(encoding="utf-8")
    )
    contract["exact_signatures"] = 1
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    parity_path.write_bytes(compatibility.PARITY_FIXTURE_V28_PATH.read_bytes())

    violations, receipt = compatibility.verify_semantic_source_tree_compatibility_v1(
        tmp_path
    )

    assert receipt is None
    assert (
        "model compatibility exact signatures declaration is invalid"
        in violations
    )


def test_v3_attestation_pair_mutation_fails_closed(tmp_path: Path) -> None:
    contract_path = tmp_path / "sourcing_model/consumer_contract.json"
    parity_path = tmp_path / "sourcing_model/consumer_parity_fixtures.json"
    contract_path.parent.mkdir(parents=True)
    contract_path.write_bytes(compatibility.CONTRACT_V68_PATH.read_bytes())
    parity_path.write_bytes(compatibility.PARITY_FIXTURE_V28_PATH.read_bytes())
    manifest = {
        "compatibility_contract": {
            "contract_id": "leadpoet-sourcing-wrapper-contract-v68",
            "path": "sourcing_model/consumer_contract.json",
            "sha256": "sha256:" + "0" * 64,
        },
        "consumer_parity_fixtures": {
            "path": "sourcing_model/consumer_parity_fixtures.json",
            "sha256": compatibility._snapshot_sha256(parity_path),
        },
        "model_artifact_hash": "",
    }

    violations = compatibility._manifest_pair_violations(
        manifest=manifest,
        contract_id="leadpoet-sourcing-wrapper-contract-v68",
        contract_path="sourcing_model/consumer_contract.json",
        contract_hash=compatibility._snapshot_sha256(contract_path),
        parity_path="sourcing_model/consumer_parity_fixtures.json",
        parity_hash=compatibility._snapshot_sha256(parity_path),
        source_tree_hash="",
    )
    assert "signed manifest compatibility contract differs from source" in violations


def test_v10_exact_dispatch_metadata_is_accepted() -> None:
    expected = _expected_v10_dispatch_custody_metadata()

    assert compatibility._sha256_json(expected) == (
        compatibility.ADDITIVE_DISPATCH_CUSTODY_V3_METADATA_SHA256
    )
    assert (
        compatibility.validate_typed_dispatch_custody_v3_metadata_v1(expected)
        == expected
    )


def test_v10_runtime_dispatch_metadata_allows_model_release_identity() -> None:
    expected = _expected_v10_dispatch_custody_metadata()
    release_contract_sha256 = "a" * 64
    current_release = deepcopy(expected)
    current_release["legacy_v2_consumer_contract_sha256"] = (
        release_contract_sha256
    )

    assert (
        compatibility.validate_typed_dispatch_custody_v3_runtime_metadata_v1(
            current_release
        )
        == current_release
    )
    with pytest.raises(
        ValueError,
        match="typed dispatch custody metadata differs",
    ):
        compatibility.validate_typed_dispatch_custody_v3_metadata_v1(
            current_release
        )


@pytest.mark.parametrize(
    "release_contract_sha256",
    (
        "A" * 64,
        "a" * 63,
        "sha256:" + "a" * 64,
        1,
        None,
    ),
)
def test_v10_runtime_dispatch_metadata_rejects_malformed_release_identity(
    release_contract_sha256: object,
) -> None:
    metadata = _expected_v10_dispatch_custody_metadata()
    metadata["legacy_v2_consumer_contract_sha256"] = release_contract_sha256

    with pytest.raises(
        ValueError,
        match="typed dispatch custody runtime metadata differs",
    ):
        compatibility.validate_typed_dispatch_custody_v3_runtime_metadata_v1(
            metadata
        )


def test_v10_runtime_dispatch_metadata_keeps_wire_abi_exact() -> None:
    metadata = _expected_v10_dispatch_custody_metadata()
    metadata["legacy_v2_consumer_contract_sha256"] = "a" * 64
    metadata["kind_ids"]["start"] = "forged-start"

    with pytest.raises(
        ValueError,
        match="typed dispatch custody runtime metadata differs",
    ):
        compatibility.validate_typed_dispatch_custody_v3_runtime_metadata_v1(
            metadata
        )


@pytest.mark.parametrize(
    ("path", "replacement"),
    (
        (("kind_ids", "start"), "forged-start"),
        (("domain_terminator_hex",), "01"),
        (("self_hash_fields", "action"), "request_sha256"),
        (("payload_fields", "start", 0), "forged_field"),
        (("envelope_fields", "action", 0), "forged_field"),
        (("typed_tags", "integer"), "66"),
        (("raw_json", "duplicate_object_keys"), "last_wins"),
        (("persisted_json", "ascii_escaped"), False),
        (("required_typed_vector_ids", 0), "forged-vector"),
        (("required_raw_json_vector_ids", 0), "forged-vector"),
        (("required_dispatch_vector_kinds", 0), "action"),
        (("max_depth",), 129),
        (("max_component_bytes",), 16777217),
        (("max_node_count",), 65537),
        (("raw_json", "max_bytes"), 16777217),
        (("integer_min",), -9007199254740992),
        (("integer_max",), 9007199254740992),
        (("float_signed_zero",), "collapse"),
        (("completion_included",), True),
        (("initial_dispatch_schema_version",), "forged-schema"),
        (("start_validation_entrypoint",), "forged-validator"),
        (("dispatch_vector_builder_id",), "partial-company-v1"),
        (("legacy_v2_consumer_contract_sha256",), "0" * 64),
    ),
)
def test_v10_dispatch_metadata_hostile_mutations_fail_closed(
    path: tuple[str | int, ...],
    replacement: object,
) -> None:
    mutated = _expected_v10_dispatch_custody_metadata()
    cursor: object = mutated
    for part in path[:-1]:
        cursor = cursor[part]  # type: ignore[index]
    cursor[path[-1]] = replacement  # type: ignore[index]

    with pytest.raises(ValueError, match="typed dispatch custody metadata differs"):
        compatibility.validate_typed_dispatch_custody_v3_metadata_v1(mutated)


def test_v10_dispatch_metadata_numeric_type_mutations_fail_closed() -> None:
    expected = _expected_v10_dispatch_custody_metadata()
    for key, replacement in (
        ("integer_min", float(expected["integer_min"])),
        ("integer_max", float(expected["integer_max"])),
        ("max_depth", 128.0),
        ("max_component_bytes", 16777216.0),
        ("max_node_count", 65536.0),
    ):
        mutated = deepcopy(expected)
        mutated[key] = replacement
        with pytest.raises(
            ValueError,
            match="typed dispatch custody metadata differs",
        ):
            compatibility.validate_typed_dispatch_custody_v3_metadata_v1(
                mutated
            )


def test_v10_dispatch_metadata_nonfinite_values_fail_closed() -> None:
    expected = _expected_v10_dispatch_custody_metadata()
    for replacement in (float("nan"), float("inf"), float("-inf")):
        mutated = deepcopy(expected)
        mutated["max_depth"] = replacement
        with pytest.raises(
            ValueError,
            match="typed dispatch custody metadata differs",
        ):
            compatibility.validate_typed_dispatch_custody_v3_metadata_v1(
                mutated
            )


def test_v10_dispatch_metadata_field_mutations_fail_closed() -> None:
    expected = _expected_v10_dispatch_custody_metadata()
    missing = deepcopy(expected)
    missing.pop("kind_ids")
    extra = deepcopy(expected)
    extra["forged"] = True

    for mutated in (missing, extra):
        with pytest.raises(
            ValueError,
            match="typed dispatch custody metadata differs",
        ):
            compatibility.validate_typed_dispatch_custody_v3_metadata_v1(
                mutated
            )


def test_v10_private_runtime_applies_exact_dispatch_metadata_gate() -> None:
    script = textwrap.dedent(
        """
        import sys
        import types

        fcntl = types.ModuleType("fcntl")
        fcntl.LOCK_EX = 2
        fcntl.LOCK_NB = 4
        fcntl.LOCK_SH = 1
        fcntl.LOCK_UN = 8
        fcntl.flock = lambda *_args, **_kwargs: None
        sys.modules["fcntl"] = fcntl

        from research_lab.eval.private_runtime import (
            PrivateModelRuntimeError,
            validate_sourcing_adapter_metadata,
        )
        from research_lab.sourcing_model_contract_check import (
            approved_typed_dispatch_custody_v3_metadata_v1,
        )

        dispatch = approved_typed_dispatch_custody_v3_metadata_v1()
        dispatch["kind_ids"]["start"] = "forged-start"
        try:
            validate_sourcing_adapter_metadata(
                {
                    "adapter_version": "sourcing-model-research-lab-adapter:v10",
                    "dispatch_custody": dispatch,
                }
            )
        except PrivateModelRuntimeError as exc:
            assert str(exc) == (
                "private model v10 dispatch custody metadata differs"
            )
        else:
            raise AssertionError("forged v10 dispatch metadata was accepted")
        """
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr or result.stdout
