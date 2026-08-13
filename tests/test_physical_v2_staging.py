from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_physical_v2_staging.py"
WORKFLOW = ROOT / ".github" / "workflows" / "physical-v2-staging.yml"


def _module():
    spec = importlib.util.spec_from_file_location("physical_v2_staging", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _config(tmp_path: Path) -> dict:
    key = tmp_path / "staging.pem"
    key.write_text("test-only\n", encoding="utf-8")
    common = {"ssh_key": str(key)}
    return {
        "schema_version": "leadpoet.physical_v2_staging_config.v1",
        "environment": "physical-v2-staging",
        "network": "test",
        "netuid": 1,
        "gateway_public_url": "https://staging-gateway.example",
        "timeout_seconds": 300,
        "poll_seconds": 2,
        "gateway": {
            **common,
            "ssh_host": "ec2-user@192.0.2.10",
            "restart_path": "/home/ec2-user/gw_restart.sh",
            "secret_id": "leadpoet/staging/gateway/env",
        },
        "primary_validator": {
            **common,
            "ssh_host": "ec2-user@192.0.2.11",
            "restart_path": "/home/ec2-user/validator_restart.sh",
            "secret_id": "leadpoet/staging/validator/env",
            "repo_root": "/home/ec2-user/leadpoet/leadpoet",
            "container_name": "leadpoet-validator-main",
        },
        "audit_validators": [
            {
                **common,
                "ssh_host": "ec2-user@192.0.2.12",
                "repo_root": "/home/ec2-user/leadpoet/leadpoet",
                "unit_name": "leadpoet-auditor-a.service",
                "expected_hotkey": "5" * 48,
            },
            {
                **common,
                "ssh_host": "ec2-user@192.0.2.13",
                "repo_root": "/home/ec2-user/leadpoet/leadpoet",
                "unit_name": "leadpoet-auditor-b.service",
                "expected_hotkey": "6" * 48,
            },
        ],
    }


def _write_config(tmp_path: Path, value: dict) -> Path:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_physical_staging_config_requires_isolated_real_boundaries(
    tmp_path: Path,
) -> None:
    module = _module()
    config = module.load_config(_write_config(tmp_path, _config(tmp_path)))

    assert config.network == "test"
    assert config.netuid == 1
    assert len(config.auditors) == 2
    assert len(
        {
            config.gateway.ssh_host,
            config.primary_validator.ssh_host,
            *(item.ssh_host for item in config.auditors),
        }
    ) == 4


@pytest.mark.parametrize(
    "mutate, message",
    [
        (
            lambda value: value["gateway"].update(
                {"ssh_host": "ec2-user@52.91.135.79"}
            ),
            "production host",
        ),
        (
            lambda value: value["gateway"].update(
                {"secret_id": "leadpoet/prod/gateway/env"}
            ),
            "staging secret",
        ),
        (
            lambda value: value.update({"network": "finney"}),
            "testnet",
        ),
        (
            lambda value: value.update(
                {"gateway_public_url": "http://52.91.135.79:8000"}
            ),
            "isolated URL",
        ),
        (
            lambda value: value.update(
                {"audit_validators": value["audit_validators"][:1]}
            ),
            "at least two",
        ),
        (
            lambda value: value["audit_validators"][0].update(
                {"ssh_host": value["gateway"]["ssh_host"]}
            ),
            "distinct hosts",
        ),
    ],
)
def test_physical_staging_config_rejects_parity_shortcuts(
    tmp_path: Path,
    mutate,
    message: str,
) -> None:
    module = _module()
    value = _config(tmp_path)
    mutate(value)

    with pytest.raises(module.PhysicalStagingError, match=message):
        module.load_config(_write_config(tmp_path, value))


def test_physical_staging_requires_identical_finalized_authority() -> None:
    module = _module()
    bundle_hash = "sha256:" + "a" * 64
    weights_hash = "sha256:" + "b" * 64
    authority = {
        "authority_stage": "finalized",
        "bundle_hash": bundle_hash,
        "compact_submission": {
            "weight_result": {
                "epoch_id": 42,
                "weights_hash": weights_hash,
            }
        },
    }

    assert module._authority_identity(authority) == (
        42,
        bundle_hash,
        weights_hash,
    )
    assert module._matching_auditor_success(
        [
            {
                "event": "submission_success",
                "netuid": 1,
                "epoch": 42,
                "bundle_hash": bundle_hash,
                "weights_hash": weights_hash,
                "confirmation_stage": "timelocked_commit_finalized",
            }
        ],
        netuid=1,
        epoch_id=42,
        bundle_hash=bundle_hash,
        weights_hash=weights_hash,
    )
    assert not module._matching_auditor_success(
        [
            {
                "event": "submission_success",
                "netuid": 1,
                "epoch": 42,
                "bundle_hash": "sha256:" + "c" * 64,
                "weights_hash": weights_hash,
                "confirmation_stage": "timelocked_commit_finalized",
            }
        ],
        netuid=1,
        epoch_id=42,
        bundle_hash=bundle_hash,
        weights_hash=weights_hash,
    )
    assert module._matching_auditor_startup(
        [
            {
                "event": "startup_ready",
                "commit": "d" * 40,
                "netuid": 1,
                "hotkey": "5" * 48,
                "gateway_endpoint": "https://staging-gateway.example/",
                "weight_protocol": "authoritative_v2",
            }
        ],
        commit="d" * 40,
        netuid=1,
        hotkey="5" * 48,
        gateway_public_url="https://staging-gateway.example",
    )


def test_physical_staging_workflow_is_attestation_bound() -> None:
    source = WORKFLOW.read_text(encoding="utf-8")

    assert "workflow_run:" in source
    assert "Attested V2 Release" in source
    assert "github.event.workflow_run.conclusion == 'success'" in source
    assert "leadpoet-v2-physical-staging" in source
    assert "run_physical_v2_staging.py" in source
    assert "LEADPOET_V2_STAGING_CONFIG_JSON" in source
    assert "LEADPOET_V2_STAGING_SSH_KEY" in source


def test_paired_restart_passes_staging_secret_ids_without_changing_defaults() -> None:
    source = (ROOT / "scripts" / "restart_attested_release_local.sh").read_text(
        encoding="utf-8"
    )

    assert 'GATEWAY_ENV_SECRET_ID="${LEADPOET_GATEWAY_ENV_SECRET_ID:-}"' in source
    assert 'VALIDATOR_ENV_SECRET_ID="${LEADPOET_VALIDATOR_ENV_SECRET_ID:-}"' in source
    assert "LEADPOET_GATEWAY_ENV_SECRET_ID='$GATEWAY_ENV_SECRET_ID'" in source
    assert "LEADPOET_VALIDATOR_ENV_SECRET_ID='$VALIDATOR_ENV_SECRET_ID'" in source
