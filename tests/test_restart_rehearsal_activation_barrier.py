from __future__ import annotations

import pytest

from tests.restart_rehearsal.verify_evidence import (
    VALIDATOR_GATEWAY_ACTIVATION_INVARIANT,
    verify_validator_gateway_activation_barrier,
)


FROM_SHA = "a" * 40
CANDIDATE_SHA = "b" * 40


def _gateway(endpoint: str, commit: str, **extra: object) -> dict:
    return {
        "kind": "curl",
        "boundary": "http_service",
        "operation": "gateway_request",
        "url": f"http://gateway.invalid:8000{endpoint}",
        "served_commit": commit,
        **extra,
    }


def _late_activation_rows() -> list[dict]:
    return [
        _gateway(
            "/health/v2-authority",
            FROM_SHA,
            gateway_probe_attempt=1,
        ),
        _gateway("/health/v2-authority", CANDIDATE_SHA),
        _gateway("/build-info", CANDIDATE_SHA),
        _gateway(
            f"/weights/v2/release-evidence/{CANDIDATE_SHA}",
            CANDIDATE_SHA,
        ),
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
            "argv": ["leadpoet-validator:latest"],
        },
        {
            "kind": "docker",
            "operation": "inspect",
            "argv": [
                "org.opencontainers.image.revision",
                "leadpoet-validator:latest",
            ],
        },
        {
            "kind": "docker",
            "operation": "inspect",
            "argv": ["{{.Id}}", "leadpoet-validator:latest"],
        },
        _gateway("/health/v2-authority", CANDIDATE_SHA),
        _gateway("/build-info", CANDIDATE_SHA),
        _gateway(
            f"/weights/v2/release-evidence/{CANDIDATE_SHA}",
            CANDIDATE_SHA,
        ),
        {
            "kind": "docker",
            "operation": "inspect",
            "argv": ["{{.Id}}", "leadpoet-validator:latest"],
        },
        {
            "kind": "validator-process",
            "process": "validator.coordinator",
            "status": "started",
        },
        _gateway("/health/v2-authority", CANDIDATE_SHA),
        _gateway("/build-info", CANDIDATE_SHA),
        _gateway(
            f"/weights/v2/release-evidence/{CANDIDATE_SHA}",
            CANDIDATE_SHA,
        ),
    ]


def test_late_activation_accepts_production_preflight_then_build_order() -> None:
    result = verify_validator_gateway_activation_barrier(
        _late_activation_rows(),
        from_sha=FROM_SHA,
        candidate_sha=CANDIDATE_SHA,
        late_activation_supported=True,
    )

    assert result == {VALIDATOR_GATEWAY_ACTIVATION_INVARIANT: True}


def test_late_activation_requires_second_alignment_after_image_prepare() -> None:
    rows = _late_activation_rows()
    del rows[14:17]

    with pytest.raises(SystemExit, match="at activation barrier"):
        verify_validator_gateway_activation_barrier(
            rows,
            from_sha=FROM_SHA,
            candidate_sha=CANDIDATE_SHA,
            late_activation_supported=True,
        )
