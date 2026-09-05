from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

from tests.restart_rehearsal.verify_evidence import (
    selected_validator_fulfillment_worker_ids,
    verify_validator_role_release_identity,
)


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_SHA = "a" * 40
IMAGE_ID = "sha256:" + "b" * 64
RELEASE_VARIABLES = (
    "LEADPOET_SENTRY_RELEASE",
    "VALIDATOR_V2_DEPLOY_COMMIT",
    "GITHUB_SHA",
    "GIT_COMMIT",
)


def _role_state(tmp_path: Path) -> dict:
    containers = {}
    names = ["leadpoet-validator-main"] + [
        f"leadpoet-ff-worker-{worker_id}"
        for worker_id in selected_validator_fulfillment_worker_ids((ROOT,))
    ]
    for ordinal, name in enumerate(names, start=1):
        log_path = tmp_path / f"{name}.log"
        log_path.write_text("started\n", encoding="utf-8")
        if name.startswith("leadpoet-ff-worker-"):
            worker_id = name.removeprefix("leadpoet-ff-worker-")
            role = "validator.fulfillment_worker"
        else:
            worker_id = ""
            role = "validator.coordinator"
        containers[name] = {
            "running": True,
            "restart_count": 0,
            "pid": ordinal,
            "environment": [
                f"{variable}={CANDIDATE_SHA}"
                for variable in RELEASE_VARIABLES
            ],
            "image_id": IMAGE_ID,
            "image_revision": CANDIDATE_SHA,
            "role": role,
            "worker_id": worker_id,
            "log_path": str(log_path),
        }
    return {
        "images": {
            "leadpoet-validator:latest": {
                "id": IMAGE_ID,
                "commit": CANDIDATE_SHA,
            }
        },
        "containers": containers,
    }


def test_candidate_derived_validator_fleet_has_exact_release_identity(
    tmp_path: Path,
) -> None:
    verify_validator_role_release_identity(
        _role_state(tmp_path),
        candidate_sha=CANDIDATE_SHA,
        candidate_roots=(ROOT,),
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("missing_worker", "candidate-derived validator worker fleet differs"),
        ("unexpected_qualification_worker", "candidate-derived validator worker fleet differs"),
        ("wrong_image", "validator role final release state is invalid"),
        ("wrong_revision", "validator role final release state is invalid"),
        ("wrong_commit_environment", "exact release environment is invalid"),
        ("wrong_role", "validator role attribution is invalid"),
    ),
)
def test_validator_role_identity_rejects_release_drift(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    state = deepcopy(_role_state(tmp_path))
    first_worker = next(
        name
        for name in state["containers"]
        if name.startswith("leadpoet-ff-worker-")
    )
    if mutation == "missing_worker":
        state["containers"].pop(first_worker)
    elif mutation == "unexpected_qualification_worker":
        retired = deepcopy(state["containers"][first_worker])
        retired["role"] = "validator.qualification_worker"
        retired["worker_id"] = "1"
        state["containers"]["leadpoet-qual-worker-1"] = retired
    elif mutation == "wrong_image":
        state["containers"][first_worker]["image_id"] = "sha256:" + "c" * 64
    elif mutation == "wrong_revision":
        state["containers"][first_worker]["image_revision"] = "d" * 40
    elif mutation == "wrong_commit_environment":
        state["containers"][first_worker]["environment"][0] = (
            "VALIDATOR_V2_DEPLOY_COMMIT=" + "e" * 40
        )
    elif mutation == "wrong_role":
        state["containers"][first_worker]["role"] = "validator.coordinator"

    with pytest.raises(SystemExit, match=message):
        verify_validator_role_release_identity(
            state,
            candidate_sha=CANDIDATE_SHA,
            candidate_roots=(ROOT,),
        )


def test_rehearsal_secret_enables_every_candidate_selected_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("REHEARSAL_CANDIDATE_SHA", CANDIDATE_SHA)
    monkeypatch.setenv("REHEARSAL_STATE_ROOT", str(tmp_path))
    adapter_path = (
        ROOT / "tests" / "restart_rehearsal" / "contract_adapter.py"
    )
    specification = importlib.util.spec_from_file_location(
        "restart_rehearsal_validator_role_adapter",
        adapter_path,
    )
    assert specification is not None
    assert specification.loader is not None
    adapter = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(adapter)
    monkeypatch.setattr(adapter, "_candidate_root", lambda: ROOT)

    expected_fulfillment_ids = selected_validator_fulfillment_worker_ids((ROOT,))
    assert adapter._candidate_fulfillment_worker_ids() == expected_fulfillment_ids
    secret = adapter._validator_secret()
    assert secret["ENABLE_FULFILLMENT"] == "true"
    assert "ENABLE_QUALIFICATION_WORKERS" not in secret
    assert "ENABLE_QUALIFICATION_EVALUATION" not in secret
    assert not any(
        key.startswith("QUALIFICATION_WEBSHARE_PROXY_") for key in secret
    )
    restart = (ROOT / "validator_restart.sh").read_text(encoding="utf-8")
    required_keys = restart.split("required_keys=(", 1)[1].split(")", 1)[0].split()
    assert "ENABLE_QUALIFICATION_EVALUATION" not in required_keys
    assert "QUALIFICATION_WEBSHARE_PROXY_1" not in required_keys
    enabled_fulfillment_ids = tuple(
        sorted(
            int(key.rsplit("_", 1)[1])
            for key, value in secret.items()
            if key.startswith("FULFILLMENT_WEBSHARE_PROXY_") and value
        )
    )
    assert enabled_fulfillment_ids == expected_fulfillment_ids
    assert secret["LEADPOET_SENTRY_RELEASE"] != CANDIDATE_SHA


def test_rehearsal_docker_exec_rejects_stdin_script_without_interactive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("REHEARSAL_CANDIDATE_SHA", CANDIDATE_SHA)
    monkeypatch.setenv("REHEARSAL_STATE_ROOT", str(tmp_path))
    adapter_path = (
        ROOT / "tests" / "restart_rehearsal" / "contract_adapter.py"
    )
    specification = importlib.util.spec_from_file_location(
        "restart_rehearsal_validator_stdin_adapter",
        adapter_path,
    )
    assert specification is not None
    assert specification.loader is not None
    adapter = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(adapter)
    handle, state = adapter._locked_state()
    state["containers"] = {
        "leadpoet-validator-main": {"running": True},
    }
    adapter._save_state(handle, state)

    assert adapter.command_docker(
        ["exec", "leadpoet-validator-main", "python3", "-"]
    ) == 97
    assert "Docker exec stdin script requires -i" in capsys.readouterr().err


def test_deployer_pins_release_identity_in_every_validator_launch_path() -> None:
    deploy = (
        ROOT / "validator_models" / "containerizing" / "deploy_dynamic.sh"
    ).read_text(encoding="utf-8")
    common_sentry_start = deploy.index("LEADPOET_SENTRY_ENV_ARGS=(")
    common_sentry = deploy[
        common_sentry_start : deploy.index("\n)\n", common_sentry_start)
    ]
    assert "-e LEADPOET_SENTRY_RELEASE" not in common_sentry
    sections = (
        deploy[
            deploy.index("start_container() {") :
            deploy.index("# Deploy containers")
        ],
        deploy[
            deploy.index("# Auto-detect FULFILLMENT proxies") :
            deploy.index("# Wait for containers to start")
        ],
    )
    for section in sections:
        for variable in RELEASE_VARIABLES:
            assert f'-e {variable}="$VALIDATOR_V2_DEPLOY_COMMIT"' in section

    fulfillment_section = sections[-1]
    for worker_id in selected_validator_fulfillment_worker_ids((ROOT,)):
        assert f"FULFILLMENT_WEBSHARE_PROXY_{worker_id}" in fulfillment_section

    identity = deploy[
        deploy.index("validate_container_release_identity() {") :
        deploy.index("validate_container_release_identity \"leadpoet-validator-main\"")
    ]
    assert "{{.Image}}" in identity
    assert "org.opencontainers.image.revision" in identity
    for variable in RELEASE_VARIABLES:
        assert variable in identity
    assert 'docker exec -i "$container_name" python3 -' in identity
    assert "from leadpoet_observability.sentry_bootstrap import _release_identity" in identity
    assert "_release_identity() != expected" in identity
    adapter = (
        ROOT / "tests" / "restart_rehearsal" / "contract_adapter.py"
    ).read_text(encoding="utf-8")
    assert 'interactive = "-i" in argv[1:]' in adapter
    assert '"Docker exec stdin script requires -i"' in adapter
    worker_preflight = deploy[
        deploy.index("validate_worker_epoch_authority() {") :
        deploy.index("echo \"🔐 Verifying official epoch authority")
    ]
    assert 'validate_container_release_identity "$container_name"' in worker_preflight
    assert "WORKER_EPOCH_AUTHORITY_BATCH_SIZE=4" in worker_preflight
    assert 'if ! wait "${worker_epoch_authority_pids[$index]}"' in worker_preflight
    assert 'cat "${worker_epoch_authority_logs[$index]}"' in worker_preflight
    assert "one or more validator worker epoch authorities failed" in worker_preflight
