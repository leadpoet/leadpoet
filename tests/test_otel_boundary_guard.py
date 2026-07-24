"""CI guard: the OpenTelemetry boundary is code-enforced, not discipline.

The gateway's telemetry is a single, explicit, infra-only exporter in
``gateway/observability/otel_bootstrap.py`` (private ``GATEWAY_OTEL_*``
variables, explicit exporter arguments, allowlisted span attributes). This
guard fails the build the moment any change crosses that boundary, so drift
is caught at PR time instead of in production:

1. no auto-instrumentation packages (``opentelemetry-instrumentation-*`` /
   ``opentelemetry-distro``) in requirements;
2. no launch path uses the ``opentelemetry-instrument`` wrapper;
3. nothing sets the GLOBAL tracer provider;
4. the standard ``OTEL_EXPORTER_*`` / ``OTEL_SERVICE_NAME`` /
   ``OTEL_RESOURCE_ATTRIBUTES`` variables appear nowhere (the destination
   and resource must stay explicit and namespaced);
5. OTLP exporter imports exist ONLY inside the bootstrap module (so Langfuse
   or any other integration can never acquire an OTLP exporter);
6. the bootstrap builds a FIXED resource and never uses the SDK's
   env-merging resource factory.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = "gateway/observability/otel_bootstrap.py"
THIS_GUARD = "tests/test_otel_boundary_guard.py"


def _tracked_files() -> list[str]:
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


def test_no_auto_instrumentation_packages_in_requirements() -> None:
    offenders = []
    for relative in _tracked_files():
        if not relative.endswith(("requirements.txt", "requirements.lock")):
            continue
        for line in _read(relative).splitlines():
            requirement = line.split("#", 1)[0].strip().lower()
            if requirement.startswith("opentelemetry-instrumentation") or (
                requirement.startswith("opentelemetry-distro")
            ):
                offenders.append(f"{relative}: {line.strip()}")
    assert not offenders, (
        "Auto-instrumentation must never enter the dependency set — it would "
        "instrument DB/HTTP/LLM paths ambiently. Offending lines: "
        f"{offenders}"
    )


def test_no_launch_path_uses_opentelemetry_instrument() -> None:
    offenders = []
    for relative in _tracked_files():
        if relative == THIS_GUARD:
            continue
        if not relative.endswith((".sh", ".bash", "Dockerfile", ".yml", ".yaml", ".py")) and (
            "Dockerfile" not in relative
        ):
            continue
        if "opentelemetry-instrument" in _read(relative):
            offenders.append(relative)
    assert not offenders, (
        "The opentelemetry-instrument launcher wraps the whole process in "
        f"ambient auto-instrumentation. Offending files: {offenders}"
    )


def test_nothing_sets_the_global_tracer_provider() -> None:
    pattern = re.compile(r"set_tracer_provider\s*\(")
    offenders = []
    for relative in _tracked_files():
        if not relative.endswith(".py") or relative == THIS_GUARD:
            continue
        if relative.startswith("tests/"):
            continue
        if pattern.search(_read(relative)):
            offenders.append(relative)
    assert not offenders, (
        "Setting the GLOBAL tracer provider lets any library emit spans "
        "through our exporter. The gateway uses a dedicated local provider "
        f"only. Offending files: {offenders}"
    )


def test_standard_otlp_env_vars_appear_nowhere() -> None:
    pattern = re.compile(
        r"OTEL_EXPORTER_|(?<!GATEWAY_)OTEL_SERVICE_NAME|OTEL_RESOURCE_ATTRIBUTES"
    )
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
        "The standard OTLP variables must stay UNSET and unreferenced so any "
        "stray exporter or auto-instrumentation has no destination. The "
        "gateway destination is passed explicitly from GATEWAY_OTEL_*. "
        f"Offending files: {offenders}"
    )


def test_otlp_exporter_imports_only_in_bootstrap() -> None:
    pattern = re.compile(r"opentelemetry\.exporter")
    offenders = []
    for relative in _tracked_files():
        if not relative.endswith(".py"):
            continue
        if relative in (BOOTSTRAP, THIS_GUARD):
            continue
        if relative.startswith("tests/"):
            continue
        if pattern.search(_read(relative)):
            offenders.append(relative)
    assert not offenders, (
        "OTLP exporters may exist ONLY inside the gateway bootstrap module — "
        "no other integration (including Langfuse) may acquire one. "
        f"Offending files: {offenders}"
    )


def test_bootstrap_uses_only_namespaced_gateway_variables() -> None:
    source = _read(BOOTSTRAP)
    assert "GATEWAY_OTEL_ENDPOINT" in source
    assert "GATEWAY_OTEL_TOKEN" in source
    # The bootstrap must pass the destination explicitly, never ambiently.
    assert re.search(r"OTLPSpanExporter\(\s*endpoint=", source), (
        "the exporter must be constructed with an explicit endpoint argument"
    )
    assert "OTLPSpanExporter()" not in source, (
        "a bare OTLPSpanExporter() would read ambient OTEL_* variables"
    )


def test_bootstrap_requires_token_and_refuses_ambient_exporter_env() -> None:
    source = _read(BOOTSTRAP)
    assert "token_missing" in source and "if not token" in source, (
        "an empty explicit headers dict lets the pinned exporter fall back "
        "to ambient header variables — a non-empty token must be required"
    )
    assert "_ambient_exporter_env_names" in source and (
        "gateway_otel_bootstrap_refused" in source
    ), (
        "a CI grep cannot see env vars injected by a restart script or the "
        "live process environment — the bootstrap must refuse initialization "
        "at runtime when ambient exporter variables are present"
    )


def test_bootstrap_service_name_is_a_constant() -> None:
    source = _read(BOOTSTRAP)
    assert re.search(r'^SERVICE_NAME = "leadpoet-gateway"$', source, re.M), (
        "the service identity must be a fixed constant"
    )
    assert "GATEWAY_OTEL_SERVICE_NAME" not in source, (
        "the service name must not be an arbitrary environment value"
    )


def test_bootstrap_builds_a_fixed_resource() -> None:
    source = _read(BOOTSTRAP)
    assert "Resource.create(" not in source, (
        "the env-merging resource factory reads ambient resource variables; "
        "the bootstrap must build the resource from a fixed attribute dict"
    )
    assert re.search(r"Resource\(\s*expected_resource\s*\)", source), (
        "the provider resource must be the exact fixed attribute dict that "
        "the fail-closed validator checks against"
    )


def test_no_client_controlled_route_fallback() -> None:
    source = _read(BOOTSTRAP)
    assert "/_unmatched" in source, (
        "unresolved routes must map to the fixed /_unmatched label so a "
        "client-controlled path segment is never exported"
    )
    assert "url.path.split" not in source, (
        "the route label must never be derived from the raw request path"
    )
