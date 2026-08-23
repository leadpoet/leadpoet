from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

import research_lab.sourcing_model_contract_check as compatibility


def _required_source_root() -> Path:
    raw_source_root = os.environ.get("SOURCING_MODEL_SOURCE_ROOT")
    if not raw_source_root:
        pytest.fail(
            "SOURCING_MODEL_SOURCE_ROOT must name the exact typed-custody "
            "source tree"
        )
    source_root = Path(raw_source_root)
    if not source_root.is_dir():
        pytest.fail(
            "SOURCING_MODEL_SOURCE_ROOT typed-custody source tree is unavailable"
        )
    return source_root


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
        "validate_runner_start_custody_v3",
        "validate_runner_action_custody_v3",
        "validate_runner_initial_continuation_custody_v3",
    } <= set(adapter)
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


def test_v10_exact_source_runtime_metadata_is_accepted() -> None:
    source_root = _required_source_root()
    script = textwrap.dedent(
        """
        import sys
        import types
        from copy import deepcopy

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

        sys.path.insert(0, sys.argv[1])
        import research_lab_adapter

        raw_metadata = research_lab_adapter.adapter_metadata()
        metadata = validate_sourcing_adapter_metadata(raw_metadata)
        assert metadata["adapter_version"] == (
            "sourcing-model-research-lab-adapter:v10"
        )

        forged = deepcopy(raw_metadata)
        forged["routing"]["compiler_version"] = "routing-compiler-v4"
        forged["runtime_routing"]["compiler_version"] = "routing-compiler-v4"
        try:
            validate_sourcing_adapter_metadata(forged)
        except PrivateModelRuntimeError as exc:
            assert str(exc) == (
                "private model routing compiler version is unsupported"
            )
        else:
            raise AssertionError("v10 routing compiler hybrid was accepted")
        """
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(source_root)],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_v10_source_tree_semantic_admission_is_accepted() -> None:
    source_root = _required_source_root()
    violations, receipt = compatibility.verify_semantic_source_tree_compatibility_v1(
        source_root
    )
    assert violations == []
    assert receipt is not None
    assert receipt["bindings"]["adapter_version"] == (
        "sourcing-model-research-lab-adapter:v10"
    )
