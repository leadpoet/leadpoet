"""Static guards for shell, Actions, and incident-matrix wiring."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from leadpoet_observability.sentry_operations import INCIDENT_FAILURE_CODES


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(relative: str) -> str:
    return (REPO_ROOT / relative).read_text(encoding="utf-8")


def test_instrumentation_matrix_covers_every_semantic_failure_code():
    matrix = _read("docs/sentry_restart_weight_instrumentation.md")
    missing = sorted(code for code in INCIDENT_FAILURE_CODES if f"`{code}`" not in matrix)
    assert not missing, f"instrumentation matrix is missing stable codes: {missing}"
    for number in range(1, 21):
        assert f"| {number} |" in matrix


def test_restart_python_heredocs_are_syntactically_valid():
    for relative in ("gw_restart.sh", "validator_restart.sh"):
        source = _read(relative)
        blocks = re.findall(r"<<'PY'\n(.*?)\nPY(?:\n|$)", source, flags=re.DOTALL)
        assert blocks, f"no quoted Python heredocs found in {relative}"
        for index, block in enumerate(blocks):
            compile(block, f"{relative}:heredoc:{index}", "exec")


def test_restart_summary_is_bounded_best_effort_and_correlated():
    for relative, component in (
        ("gw_restart.sh", "GATEWAY"),
        ("validator_restart.sh", "VALIDATOR"),
    ):
        source = _read(relative)
        assert "LEADPOET_RESTART_INVOCATION_ID" in source
        assert f'{component}_RESTART_INVOCATION_ID' in source
        assert "-m leadpoet_observability.sentry_cli" in source
        assert "timeout 2" in source
        assert ">/dev/null 2>&1 || true" in source
        assert "--restart-invocation-id" in source
        assert "--release-attempts" in source
        assert f"{component}_RELEASE_ATTEMPTS_USED" in source
        assert "trap " in source


def test_validator_restart_uses_an_isolated_host_telemetry_runtime():
    source = _read("validator_restart.sh")

    hydrate = source.index('. "$VALIDATOR_ENV_EXPORT"')
    prepare = source.index("prepare_validator_sentry_host_runtime", hydrate)
    shutdown = source.index("VALIDATOR_DESTRUCTIVE_PHASE_STARTED=1")
    assert hydrate < prepare < shutdown
    assert 'VALIDATOR_TELEMETRY_PYTHON_BIN="$VALIDATOR_PYTHON_BIN"' in source
    assert '"$telemetry_python" -m leadpoet_observability.sentry_cli' in source
    assert '"$VALIDATOR_PYTHON_BIN" neurons/validator.py' in source
    assert "requirements-host.lock" in source
    assert "--require-hashes" in _read("leadpoet_observability/host_runtime.py")
    assert "restart remains fail-open" in source


def test_restart_reexec_preserves_correlation_identity():
    gateway = _read("gw_restart.sh")
    validator = _read("validator_restart.sh")
    for source in (gateway, validator):
        for match in re.finditer(r"\bexec env \\\n(.*?)(?:\n\s+)(?:bash|\"\$)", source, flags=re.DOTALL):
            assert "RESTART_INVOCATION_ID" in match.group(0)

    assert gateway.count(
        'GATEWAY_RELEASE_ATTEMPTS_USED="${GATEWAY_RELEASE_ATTEMPTS_USED:-0}"'
    ) >= 2
    assert validator.count(
        'VALIDATOR_RELEASE_ATTEMPTS_USED="${VALIDATOR_RELEASE_ATTEMPTS_USED:-0}"'
    ) >= 2


def test_attested_release_summary_cannot_change_job_result():
    workflow = yaml.safe_load(_read(".github/workflows/attested-v2-release.yml"))
    jobs = workflow["jobs"]
    summary_steps = []
    for job in jobs.values():
        for step in job.get("steps", []):
            if str(step.get("name", "")).lower().startswith("emit bounded") and (
                "summary" in str(step.get("name", "")).lower()
            ):
                summary_steps.append(step)
    assert len(summary_steps) >= 3
    for step in summary_steps:
        assert step.get("if") == "always()"
        assert step.get("continue-on-error") is True
        assert step.get("timeout-minutes") == 1


def test_release_workflow_uses_only_namespaced_optional_sentry_configuration():
    source = _read(".github/workflows/attested-v2-release.yml")
    assert "secrets.LEADPOET_SENTRY_DSN" in source
    assert "LEADPOET_SENTRY_TRACES_SAMPLE_RATE" in source
    assert "LEADPOET_SENTRY_RELEASE" in source
    assert "LEADPOET_SENTRY_ENABLED" in source
    assert source.count("LEADPOET_GITHUB_JOB_TOKEN: ${{ github.token }}") == 3
    assert source.count("--github-job-name") == 3
    assert not re.search(r"(?<!LEADPOET_)SENTRY_[A-Z]", source)


def test_required_runtime_boundaries_have_semantic_instrumentation():
    required = {
        "gateway/api/weights.py": (
            "weight.block_drift_exhausted",
            "weight.finalization_missing",
        ),
        "validator_tee/host/gateway_weight_inputs_v2.py": (
            "weight.gateway_endpoint_unavailable",
            "weight.bundle_divergence",
        ),
        "validator_tee/host/vsock_client.py": ("weight.authoritative_result_invalid",),
        "validator_tee/host/runtime_v2_bootstrap.py": (
            "failure_code_for_exception",
            "validator_boot_attestation_verification",
        ),
        "validator_tee/host/chain_relay_v2.py": ("runtime.enclave_relay_unavailable",),
        "neurons/validator.py": (
            "weight.sdk_response_invalid",
            "weight.chain_transport_poisoned",
            "weight.finalization_missing",
        ),
        "neurons/auditor_validator.py": (
            "weight.bundle_divergence",
            "weight.finalization_missing",
        ),
    }
    for relative, codes in required.items():
        source = _read(relative)
        for code in codes:
            assert code in source, f"{relative} lost instrumentation for {code}"


def test_auditor_wrapper_exports_only_telemetry_release_identity():
    source = _read("neurons/auditor_validator.py")
    assert 'export LEADPOET_SENTRY_RELEASE="$CURRENT_COMMIT"' in source
    assert 'export GIT_COMMIT="$CURRENT_COMMIT"' not in source
