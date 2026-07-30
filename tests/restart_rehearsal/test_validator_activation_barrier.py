from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from gateway.tee.rehearsal_behavior_contract_v2 import (
    build_rehearsal_behavior_contract_v2,
)
from tests.restart_rehearsal.verify_evidence import (
    VALIDATOR_GATEWAY_ACTIVATION_INVARIANT,
    verify_validator_gateway_activation_barrier,
)


CANDIDATE_SHA = "1" * 40
FROM_SHA = "2" * 40


def _gateway_event(endpoint: str, commit: str, **details) -> dict:
    return {
        "kind": "curl",
        "boundary": "http_service",
        "operation": "gateway_request",
        "status": "ok",
        "url": f"http://gateway.invalid:8000{endpoint}",
        "served_commit": commit,
        **details,
    }


def _late_barrier_rows() -> list[dict]:
    return [
        {"module": "validator_tee.host.refresh_hotkey_config_v2"},
        {"module": "validator_tee.host.restart_preflight_v2"},
        {"kind": "nitro", "operation": "build_enclave"},
        {"kind": "nitro", "operation": "run_enclave"},
        {"kind": "process", "process": "validator.chain_relay"},
        {"module": "validator_tee.host.runtime_v2_bootstrap"},
        {"module": "validator_tee.host.hotkey_bootstrap_v2"},
        {
            "kind": "docker",
            "operation": "build",
            "argv": ["build", "-t", "leadpoet-validator:latest", "."],
        },
        {
            "kind": "docker",
            "operation": "inspect",
            "argv": [
                "image",
                "inspect",
                "leadpoet-validator:latest",
                "--format",
                "{{ index .Config.Labels "
                '"org.opencontainers.image.revision" }}',
            ],
        },
        {
            "kind": "docker",
            "operation": "inspect",
            "argv": [
                "image",
                "inspect",
                "leadpoet-validator:latest",
                "--format",
                "{{.Id}}",
            ],
        },
        _gateway_event(
            "/health/v2-authority",
            FROM_SHA,
            gateway_probe_attempt=1,
        ),
        _gateway_event(
            "/health/v2-authority",
            CANDIDATE_SHA,
            gateway_probe_attempt=2,
        ),
        _gateway_event("/build-info", CANDIDATE_SHA),
        _gateway_event(
            f"/weights/v2/release-evidence/{CANDIDATE_SHA}",
            CANDIDATE_SHA,
        ),
        {
            "kind": "docker",
            "operation": "inspect",
            "argv": [
                "image",
                "inspect",
                "leadpoet-validator:latest",
                "--format",
                "{{.Id}}",
            ],
        },
        {
            "kind": "validator-process",
            "process": "validator.coordinator",
            "status": "started",
        },
        _gateway_event(
            "/health/v2-authority",
            CANDIDATE_SHA,
            gateway_probe_attempt=3,
        ),
        _gateway_event("/build-info", CANDIDATE_SHA),
        _gateway_event(
            f"/weights/v2/release-evidence/{CANDIDATE_SHA}",
            CANDIDATE_SHA,
        ),
    ]


def test_gateway_boundary_exposes_stale_then_candidate_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("REHEARSAL_CANDIDATE_SHA", CANDIDATE_SHA)
    monkeypatch.setenv("REHEARSAL_FROM_SHA", FROM_SHA)
    monkeypatch.setenv("REHEARSAL_STATE_ROOT", str(tmp_path))
    adapter_path = Path(__file__).with_name("contract_adapter.py")
    specification = importlib.util.spec_from_file_location(
        "restart_rehearsal_activation_adapter",
        adapter_path,
    )
    assert specification is not None
    assert specification.loader is not None
    adapter = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(adapter)
    boundary_events: list[dict] = []
    monkeypatch.setattr(
        adapter,
        "_record_external_boundary",
        lambda **details: boundary_events.append(details),
    )

    request = [
        "-fsS",
        "http://gateway.invalid:8000/health/v2-authority",
    ]
    assert adapter.command_curl(request) == 0
    first = json.loads(capsys.readouterr().out)
    assert adapter.command_curl(request) == 0
    second = json.loads(capsys.readouterr().out)

    assert first["commit_sha"] == FROM_SHA
    assert second["commit_sha"] == CANDIDATE_SHA
    assert [event["served_commit"] for event in boundary_events] == [
        FROM_SHA,
        CANDIDATE_SHA,
    ]
    assert [event["gateway_probe_attempt"] for event in boundary_events] == [
        1,
        2,
    ]


def test_late_activation_evidence_requires_preparation_before_alignment() -> None:
    result = verify_validator_gateway_activation_barrier(
        _late_barrier_rows(),
        from_sha=FROM_SHA,
        candidate_sha=CANDIDATE_SHA,
        late_activation_supported=True,
    )
    assert result == {VALIDATOR_GATEWAY_ACTIVATION_INVARIANT: True}


def test_late_activation_evidence_rejects_early_validator_process() -> None:
    rows = _late_barrier_rows()
    process = rows.pop(15)
    rows.insert(13, process)

    with pytest.raises(
        SystemExit,
        match="candidate release evidence before activation",
    ):
        verify_validator_gateway_activation_barrier(
            rows,
            from_sha=FROM_SHA,
            candidate_sha=CANDIDATE_SHA,
            late_activation_supported=True,
        )


def test_legacy_deployer_remains_behind_exact_release_fallback() -> None:
    rows = _late_barrier_rows()
    preparation = rows[:10]
    alignment = rows[10:14]
    post_alignment = rows[14:]
    legacy_rows = [
        *alignment,
        preparation[7],
        preparation[8],
        *post_alignment[1:],
    ]

    result = verify_validator_gateway_activation_barrier(
        legacy_rows,
        from_sha=FROM_SHA,
        candidate_sha=CANDIDATE_SHA,
        late_activation_supported=False,
    )
    assert result == {VALIDATOR_GATEWAY_ACTIVATION_INVARIANT: True}


def test_legacy_deployer_rejects_application_build_before_exact_release() -> None:
    rows = _late_barrier_rows()
    preparation = rows[:10]
    alignment = rows[10:14]
    post_alignment = rows[14:]
    legacy_rows = [
        *alignment[:3],
        preparation[7],
        preparation[8],
        alignment[3],
        *post_alignment[1:],
    ]

    with pytest.raises(
        SystemExit,
        match="legacy validator deployer did not remain behind",
    ):
        verify_validator_gateway_activation_barrier(
            legacy_rows,
            from_sha=FROM_SHA,
            candidate_sha=CANDIDATE_SHA,
            late_activation_supported=False,
        )


@pytest.mark.parametrize(
    ("removed_index", "expected_error"),
    (
        (9, "candidate validator preparation did not complete"),
        (14, "unchanged validator image identity after alignment"),
    ),
)
def test_late_activation_requires_both_image_identity_checks(
    removed_index: int,
    expected_error: str,
) -> None:
    rows = _late_barrier_rows()
    rows.pop(removed_index)

    with pytest.raises(SystemExit, match=expected_error):
        verify_validator_gateway_activation_barrier(
            rows,
            from_sha=FROM_SHA,
            candidate_sha=CANDIDATE_SHA,
            late_activation_supported=True,
        )


def test_candidate_contract_declares_activation_sources_and_invariant() -> None:
    source_root = Path(__file__).resolve().parents[2]
    contract = build_rehearsal_behavior_contract_v2(
        source_root=source_root,
        candidate_sha=CANDIDATE_SHA,
        profile="prepush",
        epoch_count=1,
    )

    assert contract["required_restart_invariant_ids"] == [
        VALIDATOR_GATEWAY_ACTIVATION_INVARIANT
    ]
    assert {
        "scripts/restart_attested_release_local.sh",
        "validator_models/containerizing/deploy_dynamic.sh",
        "validator_tee/scripts/verify_pinned_gateway_release_v2.sh",
    } <= set(contract["production_source_paths"])
