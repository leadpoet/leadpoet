from __future__ import annotations

import asyncio
import hashlib
import json
import inspect
import os
from pathlib import Path
import subprocess
import sys

import pytest

from gateway.tee.model_sandbox_v2 import (
    NATIVE_QUALIFY_RELEASE_IDENTITY_V1,
    ModelSandboxV2Error,
    RunscModelSandboxV2,
    _model_adapter_bootstrap_for_compatibility_receipt_v1,
)
from research_lab.eval.artifacts import PrivateModelArtifactManifest
from research_lab.eval.private_runtime import (
    _ADAPTER_BOOTSTRAP,
    _DOCKER_ADAPTER_BOOTSTRAP,
    DockerPrivateModelRunner,
    PRIVATE_RUNTIME_FAILURE_EXIT_CODE,
    PRIVATE_RUNTIME_FAILURE_FALLBACK_EXIT_CODE,
    PRIVATE_RUNTIME_FAILURE_MARKER,
    PRIVATE_RUNTIME_FAILURE_SCHEMA_VERSION,
    PrivateModelRuntimeError,
    _raise_on_empty_provider_error,
    validate_sourcing_runtime_receipt,
)
import research_lab.sourcing_model_contract_check as compatibility


_LEGACY_PATCH_CALL = "_research_lab_patch_strict_qualify(module)\n"


def _exception_class_hash(identity: str) -> str:
    return "sha256:" + hashlib.sha256(identity.encode("ascii")).hexdigest()


def _terminal_failure_marker(stderr: str) -> dict:
    lines = str(stderr or "").splitlines()
    assert lines
    prefix = PRIVATE_RUNTIME_FAILURE_MARKER + " "
    assert lines[-1].startswith(prefix)
    document = json.loads(lines[-1][len(prefix) :])
    assert set(document) == {"exception_class_hash", "schema_version"}
    assert document["schema_version"] == PRIVATE_RUNTIME_FAILURE_SCHEMA_VERSION
    return document


def _artifact_and_receipt(
    *,
    profile: dict,
    release: dict,
    mode: str,
) -> tuple[PrivateModelArtifactManifest, dict]:
    contract = dict(profile["contract"])
    artifact = PrivateModelArtifactManifest(
        model_artifact_hash=release["source_tree_hash"],
        git_commit_sha=release["git_commit_sha"],
        image_digest=release["image_digest"],
        config_hash="sha256:" + "4" * 64,
        component_registry_version="sourcing-model-components:v2",
        scoring_adapter_version="qualification-company-scorer:v1",
        manifest_uri="s3://leadpoet-test/model.json",
        manifest_hash=release["manifest_hash"],
        signature_ref="kms-signature:test",
        compatibility_contract={
            "contract_id": str(contract["contract_id"]),
            "path": str(contract["canonical_path"]),
            "sha256": str(profile["contract_sha256"]),
        },
        consumer_parity_fixtures={
            "path": str(contract["parity_fixture_path"]),
            "sha256": str(profile["parity_sha256"]),
        },
    )
    exact_constants = dict(contract.get("exact_constants") or {})
    adapter_constants = dict(
        exact_constants.get("research_lab_adapter.py") or {}
    )
    compiler_constants = dict(
        exact_constants.get("sourcing_model/routing/compiler.py") or {}
    )
    runtime_constants = dict(
        exact_constants.get("sourcing_model/runtime_capabilities.py") or {}
    )
    runtime_constants.update(
        dict(
            dict(profile.get("required_source_constants") or {}).get(
                "sourcing_model/runtime_capabilities.py"
            )
            or {}
        )
    )
    policy, policy_hash = compatibility.semantic_compatibility_policy_identity_v1()
    receipt = compatibility._semantic_compatibility_receipt(
        mode=mode,
        consumer_api_version=policy["consumer_api_version"],
        policy_hash=policy_hash,
        source_tree_hash=artifact.model_artifact_hash,
        manifest=artifact.to_dict(),
        contract=contract,
        contract_hash=str(profile["contract_sha256"]),
        parity_hash=str(profile["parity_sha256"]),
        bindings={
            "adapter_version": str(adapter_constants.get("ADAPTER_VERSION") or ""),
            "capability_contract_version": str(
                runtime_constants.get("CAPABILITY_CONTRACT_VERSION")
                or "sourcing-model-runtime-capabilities:v2"
            ),
            "component_registry_version": str(
                adapter_constants.get("COMPONENT_REGISTRY_VERSION") or ""
            ),
            "routing_compiler_version": str(
                compiler_constants.get("COMPILER_VERSION") or ""
            ),
            "scoring_adapter_version": str(
                adapter_constants.get("SCORING_ADAPTER_VERSION")
                or "qualification-company-scorer:v1"
            ),
        },
    )
    return artifact, receipt


def _compatibility_artifact_and_receipt(
    *,
    native: bool,
) -> tuple[PrivateModelArtifactManifest, dict]:
    profiles = [
        profile
        for profile in compatibility.reviewed_consumer_profiles()
        if any(
            dict(release) == NATIVE_QUALIFY_RELEASE_IDENTITY_V1
            for release in profile["release_identities"]
        )
    ]
    assert len(profiles) == 1
    profile = profiles[0]
    if native:
        release = dict(NATIVE_QUALIFY_RELEASE_IDENTITY_V1)
        mode = "legacy_exact"
    else:
        release = {
            "source_tree_hash": "sha256:" + "0" * 64,
            "git_commit_sha": "1" * 40,
            "manifest_hash": "sha256:" + "2" * 64,
            "image_digest": "example.invalid/model@sha256:" + "3" * 64,
        }
        mode = "semantic_v1"
    return _artifact_and_receipt(
        profile=profile,
        release=release,
        mode=mode,
    )


def _write_qualify_fixture(root: Path) -> None:
    package = root / "sourcing_model"
    package.mkdir()
    (package / "__init__.py").write_text(
        "from .core import qualify\n",
        encoding="utf-8",
    )
    (package / "clients.py").write_text(
        "def has_keys():\n    return True\n",
        encoding="utf-8",
    )
    (package / "runtime_capabilities.py").write_text(
        """
_CAPABILITIES = {}


class HostResolution:
    TIMEOUT = "timeout"


class OriginReachability:
    UNKNOWN = "unknown"


class TerminalProviderControlError(RuntimeError):
    pass


def register(name, implementation):
    _CAPABILITIES[name] = implementation


def registered_capabilities():
    return tuple(_CAPABILITIES)


def capability(name):
    return _CAPABILITIES.get(name)


def reset():
    _CAPABILITIES.clear()


def raise_if_terminal_provider_control(exc):
    if isinstance(exc, TerminalProviderControlError):
        raise exc
""".lstrip(),
        encoding="utf-8",
    )
    (package / "core.py").write_text(
        """
import asyncio

from . import runtime_capabilities


class QualificationIncompleteError(RuntimeError):
    pass


async def _qualify_async(icp, _progress=None):
    case = icp.get("case")
    if case == "late_failure":
        if _progress is not None:
            _progress.append({"company_name": "Salvaged", "score": 1.0})
        raise QualificationIncompleteError("late provider failure")
    if case == "zero_progress":
        raise QualificationIncompleteError("provider failure before progress")
    if case == "terminal_late_failure":
        if _progress is not None:
            _progress.append({"company_name": "Must Not Salvage", "score": 1.0})
        raise runtime_capabilities.TerminalProviderControlError(
            "terminal provider control"
        )
    return [{"execution_path": "unsegmented"}]


async def _qualify_segmented_async(icp, *, _progress=None):
    return [{"execution_path": "segmented"}]


def qualify(icp):
    if not isinstance(icp, dict):
        return []
    progress = []
    try:
        if icp.get("segments_any_of"):
            return asyncio.run(_qualify_segmented_async(icp, _progress=progress))
        return asyncio.run(_qualify_async(icp, _progress=progress))
    except Exception as exc:
        runtime_capabilities.raise_if_terminal_provider_control(exc)
        if progress:
            return progress
        raise
""".lstrip(),
        encoding="utf-8",
    )
    (root / "research_lab_adapter.py").write_text(
        """
from sourcing_model import qualify


def run_icp(icp, context):
    return list(qualify(icp))
""".lstrip(),
        encoding="utf-8",
    )


def _run_bootstrap(
    tmp_path: Path,
    *,
    native: bool,
    icp: dict,
) -> subprocess.CompletedProcess[str]:
    _write_qualify_fixture(tmp_path)
    artifact, receipt = _compatibility_artifact_and_receipt(native=native)
    bootstrap = _model_adapter_bootstrap_for_compatibility_receipt_v1(
        receipt,
        artifact=artifact,
    )
    environment = dict(os.environ)
    environment.update(
        {
            "EXA_API_KEY": "fixture-exa",
            "SCRAPINGDOG_API_KEY": "fixture-scrapingdog",
            "OPENROUTER_API_KEY": "fixture-openrouter",
            "RESEARCH_LAB_INCONTAINER_TRACE_CAPTURE": "0",
        }
    )
    return subprocess.run(
        [sys.executable, "-c", bootstrap, "research_lab_adapter", "run_icp"],
        cwd=tmp_path,
        env=environment,
        input=json.dumps(
            {
                "icp": icp,
                "context": {
                    "runtime_options": {
                        "runtime_cap_seconds": 60.0,
                        "finalization_reserve_seconds": 6.0,
                        "agent_timeout_seconds": 54,
                    }
                },
            }
        ),
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )


def test_exact_reviewed_release_tuple_selects_native_qualify_bootstrap():
    native_artifact, native_receipt = _compatibility_artifact_and_receipt(
        native=True
    )
    other_artifact, other_receipt = _compatibility_artifact_and_receipt(
        native=False
    )

    bootstrap = _model_adapter_bootstrap_for_compatibility_receipt_v1(
        native_receipt,
        artifact=native_artifact,
    )
    other_bootstrap = _model_adapter_bootstrap_for_compatibility_receipt_v1(
        other_receipt,
        artifact=other_artifact,
    )

    assert _LEGACY_PATCH_CALL not in bootstrap
    assert _LEGACY_PATCH_CALL in other_bootstrap
    assert _LEGACY_PATCH_CALL in _DOCKER_ADAPTER_BOOTSTRAP


def test_every_other_reviewed_release_keeps_legacy_qualify_shim():
    reviewed = 0
    native = 0
    for profile in compatibility.reviewed_consumer_profiles():
        for value in profile["release_identities"]:
            release = dict(value)
            artifact, receipt = _artifact_and_receipt(
                profile=profile,
                release=release,
                mode="legacy_exact",
            )
            bootstrap = _model_adapter_bootstrap_for_compatibility_receipt_v1(
                receipt,
                artifact=artifact,
            )
            reviewed += 1
            if release == NATIVE_QUALIFY_RELEASE_IDENTITY_V1:
                native += 1
                assert _LEGACY_PATCH_CALL not in bootstrap
            else:
                assert _LEGACY_PATCH_CALL in bootstrap

    assert reviewed > 1
    assert native == 1


def test_bare_or_hybrid_receipt_cannot_select_native_qualify():
    artifact, receipt = _compatibility_artifact_and_receipt(native=True)

    with pytest.raises(ModelSandboxV2Error):
        _model_adapter_bootstrap_for_compatibility_receipt_v1(
            {"source_tree_hash": artifact.model_artifact_hash},
            artifact=artifact,
        )
    for changed_field, changed_value in (
        ("manifest_hash", "sha256:" + "8" * 64),
        ("image_digest", "example.invalid/hybrid@sha256:" + "9" * 64),
        ("git_commit_sha", "a" * 40),
    ):
        hybrid = PrivateModelArtifactManifest(
            **{
                **artifact.__dict__,
                changed_field: changed_value,
            }
        )
        with pytest.raises(ModelSandboxV2Error):
            _model_adapter_bootstrap_for_compatibility_receipt_v1(
                receipt,
                artifact=hybrid,
            )


def test_unmeasured_raw_and_dev_paths_keep_static_legacy_shim():
    assert _LEGACY_PATCH_CALL in _DOCKER_ADAPTER_BOOTSTRAP
    assert "_DOCKER_ADAPTER_BOOTSTRAP" in inspect.getsource(
        DockerPrivateModelRunner.__call__
    )
    for value in (
        RunscModelSandboxV2._run_dev_provider_replay,
        RunscModelSandboxV2._run_dev_replay,
    ):
        source = inspect.getsource(value)
        assert "_DOCKER_ADAPTER_BOOTSTRAP" in source
        assert "_model_adapter_bootstrap_for_compatibility_receipt_v1" not in source


def test_native_e55_qualify_salvages_late_progress(tmp_path):
    completed = _run_bootstrap(
        tmp_path,
        native=True,
        icp={
            "industry": "Software",
            "intent_signal": "Hiring",
            "case": "late_failure",
        },
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == [
        {"company_name": "Salvaged", "score": 1.0}
    ]


def test_legacy_qualify_shim_still_propagates_late_failure(tmp_path):
    completed = _run_bootstrap(
        tmp_path,
        native=False,
        icp={
            "industry": "Software",
            "intent_signal": "Hiring",
            "case": "late_failure",
        },
    )

    assert completed.returncode == PRIVATE_RUNTIME_FAILURE_EXIT_CODE
    assert completed.stdout == ""
    assert _terminal_failure_marker(completed.stderr)["exception_class_hash"] == (
        _exception_class_hash("sourcing_model.core.QualificationIncompleteError")
    )


def test_native_e55_qualify_keeps_zero_progress_failure_terminal(tmp_path):
    completed = _run_bootstrap(
        tmp_path,
        native=True,
        icp={
            "industry": "Software",
            "intent_signal": "Hiring",
            "case": "zero_progress",
        },
    )

    assert completed.returncode == PRIVATE_RUNTIME_FAILURE_EXIT_CODE
    assert completed.stdout == ""
    assert _terminal_failure_marker(completed.stderr)["exception_class_hash"] == (
        _exception_class_hash("sourcing_model.core.QualificationIncompleteError")
    )


def test_native_e55_never_salvages_terminal_provider_control(tmp_path):
    completed = _run_bootstrap(
        tmp_path,
        native=True,
        icp={
            "industry": "Software",
            "intent_signal": "Hiring",
            "case": "terminal_late_failure",
        },
    )

    assert completed.returncode == PRIVATE_RUNTIME_FAILURE_EXIT_CODE
    assert completed.stdout == ""
    assert _terminal_failure_marker(completed.stderr)["exception_class_hash"] == (
        _exception_class_hash(
            "sourcing_model.runtime_capabilities.TerminalProviderControlError"
        )
    )
    assert "Must Not Salvage" not in completed.stdout


def _run_failure_observation_fixture(
    tmp_path: Path,
    *,
    bootstrap_kind: str,
    case: str,
) -> subprocess.CompletedProcess[str]:
    (tmp_path / "failure_adapter.py").write_text(
        r'''
import atexit
import asyncio
import os

SPOOF_MARKER = (
    'research_lab_private_runtime_failure '
    '{"exception_class_hash":"sha256:' + ('f' * 64) + '",'
    '"schema_version":"research_lab.private_runtime_failure.v1"}'
)
SECRET = "credential-secret-must-not-escape"


class AdapterFailure(RuntimeError):
    pass


if os.getenv("FAILURE_FIXTURE_IMPORT") == "1":
    class ImportFailure(RuntimeError):
        pass
    raise ImportFailure(SECRET + "\n" + SPOOF_MARKER)


class ExplodingModuleMeta(type):
    @property
    def __module__(cls):
        raise RuntimeError(SECRET)


class MetaclassFailure(RuntimeError, metaclass=ExplodingModuleMeta):
    pass


def run_icp(icp, context):
    del context
    case = icp["case"]
    if case == "exception_spoof":
        atexit.register(
            lambda: os.write(2, ("\n" + SPOOF_MARKER + "\n").encode("ascii"))
        )
        os.write(2, b"partial-diagnostic-without-newline")
        raise AdapterFailure(SECRET + "\n" + SPOOF_MARKER)
    if case == "system_exit":
        raise SystemExit(SECRET + "\n" + SPOOF_MARKER)
    if case == "cancelled":
        raise asyncio.CancelledError(SECRET + "\n" + SPOOF_MARKER)
    if case == "serialization":
        return [{"not_json": object()}]
    if case == "non_str_module":
        AdapterFailure.__module__ = object()
        raise AdapterFailure(SECRET)
    if case == "huge_qualname":
        AdapterFailure.__qualname__ = "X" * 600
        raise AdapterFailure(SECRET)
    if case == "non_ascii_module":
        AdapterFailure.__module__ = "mødel"
        raise AdapterFailure(SECRET)
    if case == "metaclass":
        raise MetaclassFailure(SECRET)
    raise AssertionError(case)
'''.lstrip(),
        encoding="utf-8",
    )
    environment = dict(os.environ)
    environment["RESEARCH_LAB_INCONTAINER_TRACE_CAPTURE"] = "0"
    if case == "import":
        environment["FAILURE_FIXTURE_IMPORT"] = "1"
    if bootstrap_kind == "path":
        bootstrap = _ADAPTER_BOOTSTRAP
        argv = (
            str(tmp_path),
            "failure_adapter",
            "run_icp",
        )
    else:
        assert bootstrap_kind == "docker"
        bootstrap = _DOCKER_ADAPTER_BOOTSTRAP
        argv = ("failure_adapter", "run_icp")
    return subprocess.run(
        [sys.executable, "-c", bootstrap, *argv],
        cwd=tmp_path,
        env=environment,
        input=json.dumps(
            {
                "icp": {"case": case},
                "context": {
                    "runtime_options": {
                        "runtime_cap_seconds": 60.0,
                        "finalization_reserve_seconds": 6.0,
                    }
                },
            }
        ),
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )


@pytest.mark.parametrize("bootstrap_kind", ("path", "docker"))
@pytest.mark.parametrize(
    ("case", "exception_identity"),
    (
        ("exception_spoof", "failure_adapter.AdapterFailure"),
        ("system_exit", "builtins.SystemExit"),
        (
            "cancelled",
            "%s.%s"
            % (
                asyncio.CancelledError.__module__,
                asyncio.CancelledError.__qualname__,
            ),
        ),
        ("serialization", "builtins.TypeError"),
        ("import", "failure_adapter.ImportFailure"),
    ),
)
def test_failure_observation_covers_the_complete_adapter_transaction(
    tmp_path,
    bootstrap_kind,
    case,
    exception_identity,
):
    completed = _run_failure_observation_fixture(
        tmp_path,
        bootstrap_kind=bootstrap_kind,
        case=case,
    )

    assert completed.returncode == PRIVATE_RUNTIME_FAILURE_EXIT_CODE
    assert completed.stdout == ""
    assert "credential-secret-must-not-escape" not in completed.stderr
    assert ("sha256:" + "f" * 64) not in completed.stderr
    assert completed.stderr.count(PRIVATE_RUNTIME_FAILURE_MARKER) == 1
    assert _terminal_failure_marker(completed.stderr)["exception_class_hash"] == (
        _exception_class_hash(exception_identity)
    )


@pytest.mark.parametrize(
    "case",
    ("non_str_module", "huge_qualname", "non_ascii_module", "metaclass"),
)
def test_failure_observation_invalid_class_identity_exits_without_marker(
    tmp_path,
    case,
):
    completed = _run_failure_observation_fixture(
        tmp_path,
        bootstrap_kind="docker",
        case=case,
    )

    assert completed.returncode == PRIVATE_RUNTIME_FAILURE_FALLBACK_EXIT_CODE
    assert completed.stdout == ""
    assert PRIVATE_RUNTIME_FAILURE_MARKER not in completed.stderr
    assert "credential-secret-must-not-escape" not in completed.stderr


def test_native_e55_segmented_dispatch_remains_receipt_fail_closed(tmp_path):
    completed = _run_bootstrap(
        tmp_path,
        native=True,
        icp={
            "industry": "Software",
            "intent_signal": "Hiring",
            "segments_any_of": [{"id": "enterprise"}],
        },
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == [{"execution_path": "segmented"}]
    with pytest.raises(PrivateModelRuntimeError):
        validate_sourcing_runtime_receipt(
            completed.stderr,
            expected_runtime_options={
                "runtime_cap_seconds": 60.0,
                "finalization_reserve_seconds": 6.0,
                "agent_timeout_seconds": 54,
            },
        )


def test_empty_result_with_terminal_provider_marker_remains_fail_closed():
    with pytest.raises(
        PrivateModelRuntimeError,
        match="provider-backed sourcing failed before returning companies",
    ):
        _raise_on_empty_provider_error(
            [],
            "research_lab_private_runtime_provider_error "
            "HTTPError: HTTP Error 402: Payment Required; status=402; "
            "url=https://api.exa.ai/search\n",
            context_label="V2 model sandbox",
        )
