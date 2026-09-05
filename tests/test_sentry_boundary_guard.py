"""CI guard: the Sentry observability boundary is code-enforced.

Error monitoring is a single, explicit, opt-in bootstrap in
``leadpoet_observability/sentry_bootstrap.py`` (namespaced
``LEADPOET_SENTRY_*`` variables, hard-off payload capture, fail-closed
scrubbing in ``sentry_scrubbing.py``). This guard fails the build the
moment any change crosses that boundary:

1. ``sentry_sdk`` is imported or referenced only inside the bootstrap;
2. enclave requirement files and enclave/TEE surfaces never gain Sentry
   (enclave images are measured — PCR0 — and have no general egress);
3. non-namespaced ``SENTRY_*`` variables are referenced nowhere (the SDK
   must never pick up an ambient destination);
4. the hard-off capture options and scrubbing hooks stay in the bootstrap;
5. the scrubber keeps stripping stack locals, source context, the request
   and user envelopes, and argv;
6. every wired host entry point keeps its ``init_sentry`` call, so process
   coverage cannot silently regress.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = "leadpoet_observability/sentry_bootstrap.py"
SCRUBBING = "leadpoet_observability/sentry_scrubbing.py"
THIS_GUARD = "tests/test_sentry_boundary_guard.py"

ENCLAVE_REQUIREMENT_FILES = (
    "validator_tee/enclave/requirements.txt",
    "gateway/tee/requirements.txt",
    "gateway/tee/requirements-scoring-py39.in",
    "gateway/tee/requirements-scoring-py39.lock",
)

ENCLAVE_SURFACE_PREFIXES = ("validator_tee/enclave/", "gateway/tee/")

WIRED_ENTRY_POINTS = (
    "gateway/main.py",
    "neurons/validator.py",
    "neurons/miner.py",
    "neurons/auditor_validator.py",
    "validator_tee/host/gateway_pcr0_builder.py",
    "validator_tee/host/runtime_v2_bootstrap.py",
    "validator_tee/host/verify_release_gate_v2.py",
)


def _tracked_files() -> list:
    output = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [line.strip() for line in output.splitlines() if line.strip()]


def _read(relative: str) -> str:
    try:
        return (REPO_ROOT / relative).read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def test_sentry_sdk_references_only_in_bootstrap() -> None:
    pattern = re.compile(r"\bsentry_sdk\b")
    offenders = []
    for relative in _tracked_files():
        if not relative.endswith(".py"):
            continue
        if relative == BOOTSTRAP or relative.startswith("tests/"):
            continue
        if pattern.search(_read(relative)):
            offenders.append(relative)
    assert not offenders, (
        "sentry_sdk may be imported/initialized ONLY inside the bootstrap "
        "module — a second init point could bypass the fail-closed scrubber. "
        f"Offending files: {offenders}"
    )


def test_enclave_requirement_files_never_gain_sentry() -> None:
    offenders = []
    for relative in ENCLAVE_REQUIREMENT_FILES:
        if "sentry" in _read(relative).lower():
            offenders.append(relative)
    assert not offenders, (
        "Enclave images are measured (PCR0) and have no general egress; "
        f"sentry-sdk must never enter them. Offending files: {offenders}"
    )


def test_enclave_and_tee_surfaces_never_reference_error_monitoring() -> None:
    pattern = re.compile(r"sentry_sdk|leadpoet_observability")
    offenders = []
    for relative in _tracked_files():
        if not relative.startswith(ENCLAVE_SURFACE_PREFIXES):
            continue
        if pattern.search(_read(relative)):
            offenders.append(relative)
    assert not offenders, (
        "Enclave/TEE surfaces must never wire error monitoring; the gateway "
        "host process init already covers host-side TEE modules. "
        f"Offending files: {offenders}"
    )


def test_no_ambient_sentry_variables_referenced() -> None:
    # LEADPOET_SENTRY_* is the only permitted namespace. An ambient
    # SENTRY_DSN / SENTRY_ENVIRONMENT reference would let the SDK acquire a
    # destination outside the explicit bootstrap options.
    pattern = re.compile(r"(?<![A-Z_])SENTRY_[A-Z]")
    offenders = []
    for relative in _tracked_files():
        if relative == THIS_GUARD:
            continue
        if not relative.endswith(
            (".py", ".sh", ".bash", ".yml", ".yaml", ".md", ".env", ".example", "Dockerfile")
        ) and "Dockerfile" not in relative:
            continue
        if pattern.search(_read(relative)):
            offenders.append(relative)
    assert not offenders, (
        "Only namespaced LEADPOET_SENTRY_* variables may exist. "
        f"Offending files: {offenders}"
    )


def test_bootstrap_hard_off_options_present() -> None:
    source = _read(BOOTSTRAP)
    for marker in (
        'include_local_variables=False',
        'send_default_pii=False',
        'max_request_body_size="never"',
        'traces_sample_rate=_trace_sample_rate()',
        'propagate_traces=False',
        'enable_logs=False',
        'spotlight=False',
        'keep_alive=False',
        'auto_session_tracking=False',
        'auto_enabling_integrations=False',
        'default_integrations=False',
        'max_breadcrumbs=_MAX_BREADCRUMBS',
        'LoggingIntegration(',
        'level=None',
        'event_level=logging.ERROR',
        'sentry_logs_level=None',
        'debug=False',
        'before_send=',
        'before_breadcrumb=',
        'before_send_transaction=',
        'LEADPOET_SENTRY_ENABLED',
        'LEADPOET_SENTRY_DSN',
        'LEADPOET_SENTRY_TRACES_SAMPLE_RATE',
    ):
        assert marker in source, (
            "the bootstrap's hard-off capture contract drifted: missing "
            f"{marker!r}"
        )
    assert "profiles_sample_rate" not in source, (
        "profiling must never be enabled — errors only"
    )


def test_validator_containers_receive_only_namespaced_sentry_settings() -> None:
    source = _read("validator_models/containerizing/deploy_dynamic.sh")
    assert '"${LEADPOET_SENTRY_ENV_ARGS[@]}"' in source
    assert source.count('"${LEADPOET_SENTRY_ENV_ARGS[@]}"') == source.count("docker run -d")
    for name in (
        "LEADPOET_SENTRY_ENABLED",
        "LEADPOET_SENTRY_DSN",
        "LEADPOET_SENTRY_ENVIRONMENT",
        "LEADPOET_SENTRY_EXTRA_PROTECTED_MODULES",
        "LEADPOET_SENTRY_MESSAGE_MODE",
        "LEADPOET_SENTRY_TRACES_SAMPLE_RATE",
        "LEADPOET_RESTART_INVOCATION_ID",
    ):
        assert f"-e {name}" in source
        assert f"-e {name}=" not in source
    exact_release = (
        '-e LEADPOET_SENTRY_RELEASE="$VALIDATOR_V2_DEPLOY_COMMIT"'
    )
    assert source.count(exact_release) == source.count("docker run -d")
    assert "    -e LEADPOET_SENTRY_RELEASE\n" not in source


def test_bootstrap_fails_closed_and_swallows_wiring_failures() -> None:
    source = _read(BOOTSTRAP)
    assert "leadpoet_sentry_scrub_failed" in source and "return None" in source, (
        "a scrub failure must DROP the event (fail closed), never send it"
    )
    assert "except BaseException" in source, (
        "init and the scrub hooks must swallow every failure — error "
        "monitoring can never break a runtime"
    )


def test_scrubber_envelope_hygiene_markers_present() -> None:
    source = _read(SCRUBBING)
    for marker in (
        'frame.pop("context_line", None)',
        'frame.pop("pre_context", None)',
        'frame.pop("post_context", None)',
        'frame.pop("vars", None)',
        'event.pop("request", None)',
        'event.pop("user", None)',
        'extra.pop("sys.argv", None)',
    ):
        assert marker in source, (
            "the scrubber's envelope hygiene drifted: missing "
            f"{marker!r}"
        )


def test_every_wired_entry_point_initializes_sentry() -> None:
    missing = [
        relative
        for relative in WIRED_ENTRY_POINTS
        if "init_sentry(" not in _read(relative)
    ]
    assert not missing, (
        "a wired host entry point lost its init_sentry call — process "
        f"coverage silently regressed: {missing}"
    )


def test_auditor_initializes_before_auto_update_handoff() -> None:
    source = _read("neurons/auditor_validator.py")
    assert source.index("_init_sentry(component=\"auditor-validator\")") < source.index(
        "AUTO-UPDATER: Automatically updates entire repo"
    )
