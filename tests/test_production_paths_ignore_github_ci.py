from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

GITHUB_CI_GATE_MARKERS = (
    "api.github.com",
    "/actions/runs",
    "workflow_run",
    "check-runs",
    "checks/runs",
    "gh run",
    "gh api",
    "gh attest",
    "github attestation",
    "github_attestation",
    "/attestations",
    "actions/attest",
    "leadpoet_github_job_token",
    "github_job_name",
    "attested-v2-release.yml",
)


def _assert_no_github_ci_gate(paths: tuple[str, ...]) -> None:
    for relative in paths:
        path = ROOT / relative
        candidates = sorted(path.rglob("*.py")) if path.is_dir() else [path]
        for candidate in candidates:
            text = candidate.read_text(encoding="utf-8").lower()
            found = [marker for marker in GITHUB_CI_GATE_MARKERS if marker in text]
            assert not found, (
                f"{candidate.relative_to(ROOT)} contains GitHub CI gate {found}"
            )


def test_gateway_current_release_path_does_not_consult_github_ci() -> None:
    _assert_no_github_ci_gate(
        (
            "gw_restart.sh",
            "scripts/gateway_git_deploy.py",
            "gateway/tee/build_local_release_v2.sh",
            "gateway/tee/local_release_v2.py",
            "gateway/tee/release_channel_v2.py",
            "gateway/tee/restart_preflight_v2.py",
            "gateway/tee/stage_attested_runtime.sh",
            "validator_tee/host/gateway_pcr0_builder.py",
            "Leadpoet/utils/restart_release_supersession_v2.py",
        )
    )

    script = (ROOT / "gw_restart.sh").read_text(encoding="utf-8")
    local_build = script.index("Building the exact local gateway runtime identity")
    local_ready = script.index('record_gateway_restart_timing "local_release_ready"')
    historical_fallback = script.index(
        "Acquiring the exact historical attested V2 release channel"
    )
    shutdown = script.index("Stopping existing gateway and Research Lab worker processes")
    assert local_build < local_ready < historical_fallback < shutdown

    sentry_start = script.index("emit_gateway_restart_sentry_summary() {")
    sentry_end = script.index("\n}\n", sentry_start)
    sentry_summary = script[sentry_start:sentry_end]
    assert "restart-summary" in sentry_summary
    assert "release-summary" not in sentry_summary
    assert "timeout 2" in sentry_summary
    assert '"${shutdown_flag[@]}" >/dev/null 2>&1 || true' in sentry_summary


def test_attested_release_workflow_is_manual_only_with_integrity_guards() -> None:
    workflow = (ROOT / ".github/workflows/attested-v2-release.yml").read_text(
        encoding="utf-8"
    )
    trigger = workflow[: workflow.index("\nconcurrency:")]
    assert trigger == "name: Gateway Attested Release\n\non:\n  workflow_dispatch:\n"

    assert 'test "$GITHUB_REF" = "refs/heads/main"' in workflow
    assert 'test "$(git rev-parse HEAD)" = "$GITHUB_SHA"' in workflow
    assert "leadpoet_acquire_docker_operation_lock_v2" in workflow
    assert "leadpoet-gateway-v2-builder" in workflow
    assert "leadpoet-validator-v2-builder" in workflow


def test_normal_validator_restart_does_not_consult_github_ci() -> None:
    _assert_no_github_ci_gate(
        (
            "validator_restart.sh",
            "scripts/run_arena_validator.py",
            "lab_arena/validator.py",
        )
    )


def test_rebenchmark_runtime_does_not_consult_github_ci() -> None:
    # A model source may still be fetched from an exact GitHub URL. The runtime
    # must not query Actions, checks, workflow status, or attested-release jobs.
    _assert_no_github_ci_gate(
        (
            "scripts/run_lab_arena_service.py",
            "lab_arena",
        )
    )
