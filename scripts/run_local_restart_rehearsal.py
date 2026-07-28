#!/usr/bin/env python3
"""Run the exact local N-1 restart and V2 publication rehearsal.

The driver has two deliberately fixed profiles:

``prepush``
    One forward N-1 -> N restart and one complete V2 publication in a
    resource-bounded Docker replica.
``release``
    Forward, rollback, roll-forward, the external-boundary fault matrix,
    concurrency checks, and 100 accelerated stateful subnet epochs.

Neither profile reads production credentials.  Repository-owned behavior is
executed from the frozen candidate checkout; only the boundaries enumerated by
the rehearsal contract may be implemented locally.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
import platform
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Iterator, Sequence
from urllib.request import urlopen


REPO_ROOT = Path(__file__).resolve().parents[1]
IMAGE_REPOSITORY = "leadpoet-local-restart-rehearsal"
REHEARSAL_TEMP_ROOT_ENV = "LEADPOET_RESTART_REHEARSAL_TEMP_ROOT"
PYTHON37_IMAGE = (
    "python@sha256:"
    "b53f496ca43e5af6994f8e316cf03af31050bf7944e0e4a308ad86c001cf028b"
)
COMMITTED_HARNESS_PATHS = (
    "tests/restart_rehearsal/Dockerfile",
    "tests/restart_rehearsal/artifact_identity.py",
    "tests/restart_rehearsal/boundary_contract.json",
    "tests/restart_rehearsal/contract_adapter.py",
    "tests/restart_rehearsal/fixtures/production_shaped_v2.json",
    "tests/restart_rehearsal/gateway_boundary_service.py",
    "tests/restart_rehearsal/gateway_enclave_service.py",
    "tests/restart_rehearsal/join_evidence.py",
    "tests/restart_rehearsal/local_services.py",
    "tests/restart_rehearsal/prepare_external_artifacts.py",
    "tests/restart_rehearsal/prepare_host_fixtures.py",
    "tests/restart_rehearsal/production_workflow_runner.py",
    "tests/restart_rehearsal/sanitized_weight_fixture.py",
    "tests/restart_rehearsal/sitecustomize.py",
    "tests/restart_rehearsal/tls_connect_proxy_service.py",
    "tests/restart_rehearsal/validator_enclave_service.py",
    "tests/restart_rehearsal/run_inside.sh",
    "tests/restart_rehearsal/verify_evidence.py",
)
SCORING_WHEELHOUSE_PATHS = (
    "gateway/tee/requirements-scoring-py39.in",
    "gateway/tee/requirements-scoring-py39.lock",
)
EXTERNAL_ARTIFACT_LOCK_PATHS = (
    "gateway/tee/runsc-runtime.lock.json",
    "validator_tee/runtime-artifacts-v2.lock.json",
)
PROFILE_LIMITS = {
    "prepush": {
        "cpus": "4",
        "memory": "7g",
        "epochs": 1,
        "fault_matrix": False,
    },
    "release": {
        "cpus": "6",
        "memory": "7g",
        "epochs": 100,
        "fault_matrix": True,
    },
}
_MEMORY_LIMIT_RE = re.compile(r"^(?P<amount>[1-9][0-9]*)(?P<unit>[kmgt]?)b?$", re.I)
_MEMORY_UNIT_BYTES = {
    "": 1,
    "k": 1024,
    "m": 1024**2,
    "g": 1024**3,
    "t": 1024**4,
}


def _cpu_limit(value: str, *, option: str) -> Decimal:
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise SystemExit(f"{option} must be a positive Docker CPU limit") from exc
    if not parsed.is_finite() or parsed <= 0:
        raise SystemExit(f"{option} must be a positive Docker CPU limit")
    return parsed


def _memory_limit_bytes(value: str, *, option: str) -> int:
    match = _MEMORY_LIMIT_RE.fullmatch(value.strip())
    if match is None:
        raise SystemExit(
            f"{option} must be a positive Docker memory limit such as 6g"
        )
    return int(match.group("amount")) * _MEMORY_UNIT_BYTES[
        match.group("unit").lower()
    ]


def _effective_profile_limits(
    profile: str,
    *,
    outer_cpus: str | None,
    outer_memory: str | None,
) -> dict[str, Any]:
    """Return bounded outer limits without changing the production contract."""

    limits = dict(PROFILE_LIMITS[profile])
    if outer_cpus is not None:
        requested_cpus = _cpu_limit(outer_cpus, option="--outer-cpus")
        profile_cpus = _cpu_limit(str(limits["cpus"]), option="profile CPU limit")
        if requested_cpus > profile_cpus:
            raise SystemExit(
                "--outer-cpus cannot exceed the selected profile CPU limit"
            )
        limits["cpus"] = str(outer_cpus).strip()
    if outer_memory is not None:
        requested_memory = _memory_limit_bytes(
            outer_memory,
            option="--outer-memory",
        )
        profile_memory = _memory_limit_bytes(
            str(limits["memory"]),
            option="profile memory limit",
        )
        if requested_memory > profile_memory:
            raise SystemExit(
                "--outer-memory cannot exceed the selected profile memory limit"
            )
        limits["memory"] = str(outer_memory).strip().lower()
    return limits


def _docker_bind_source(path: Path) -> str:
    """Return the canonical host path Docker Desktop can bind safely."""

    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise SystemExit(f"Docker bind source is unavailable: {path}") from exc
    if not resolved.is_absolute():
        raise SystemExit(f"Docker bind source is not absolute: {path}")
    return str(resolved)


def _temporary_directory(*, prefix: str) -> tempfile.TemporaryDirectory[str]:
    raw_root = os.getenv(REHEARSAL_TEMP_ROOT_ENV, "").strip()
    root: Path | None = None
    if raw_root:
        try:
            root = Path(raw_root).expanduser().resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise SystemExit(
                f"{REHEARSAL_TEMP_ROOT_ENV} is unavailable: {raw_root}"
            ) from exc
        if not root.is_dir():
            raise SystemExit(
                f"{REHEARSAL_TEMP_ROOT_ENV} is not a directory: {root}"
            )
    return tempfile.TemporaryDirectory(
        prefix=prefix,
        dir=str(root) if root is not None else None,
    )


def _run(
    args: Sequence[str],
    *,
    cwd: Path = REPO_ROOT,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(args),
        cwd=str(cwd),
        check=True,
        text=True,
        capture_output=capture,
    )


def _git_sha(value: str) -> str:
    result = _run(
        ["git", "rev-parse", "--verify", f"{value}^{{commit}}"],
        capture=True,
    )
    return result.stdout.strip()


def _git_file(commit_sha: str, path: str) -> bytes:
    result = subprocess.run(
        ["git", "show", f"{commit_sha}:{path}"],
        cwd=str(REPO_ROOT),
        check=True,
        capture_output=True,
    )
    return result.stdout


def _is_ancestor(ancestor: str, descendant: str) -> bool:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=str(REPO_ROOT),
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode not in (0, 1):
        raise SystemExit("unable to resolve restart rehearsal Git ancestry")
    return result.returncode == 0


def _resolve_transition(
    from_sha: str,
    candidate_sha: str,
    requested: str,
) -> str:
    forward = _is_ancestor(from_sha, candidate_sha)
    rollback = (
        from_sha != candidate_sha
        and _is_ancestor(candidate_sha, from_sha)
    )
    if requested == "auto":
        if forward:
            return "forward"
        if rollback:
            return "rollback"
        raise SystemExit("restart rehearsal commits are unrelated")
    if requested == "forward" and not forward:
        raise SystemExit("forward rehearsal target does not descend from --from-sha")
    if requested == "rollback" and not rollback:
        raise SystemExit("rollback rehearsal target is not an ancestor of --from-sha")
    return requested


def _docker_platform(profile: str) -> str:
    if profile == "release":
        return "linux/amd64"
    machine = platform.machine().lower()
    if machine in {"arm64", "aarch64"}:
        return "linux/arm64"
    if machine in {"amd64", "x86_64"}:
        return "linux/amd64"
    raise SystemExit(f"unsupported local Docker architecture: {machine}")


def _image_tag(harness_sha: str, *, docker_platform: str) -> str:
    digest = hashlib.sha256()
    digest.update(b"harness_sha")
    digest.update(harness_sha.encode("ascii"))
    digest.update(b"docker_platform")
    digest.update(docker_platform.encode("ascii"))
    digest.update(b"requirements.txt")
    digest.update(_git_file(harness_sha, "requirements.txt"))
    for path in SCORING_WHEELHOUSE_PATHS:
        digest.update(path.encode("utf-8"))
        digest.update(_git_file(harness_sha, path))
    for path in EXTERNAL_ARTIFACT_LOCK_PATHS:
        digest.update(path.encode("utf-8"))
        digest.update(_git_file(harness_sha, path))
    for path in COMMITTED_HARNESS_PATHS:
        digest.update(path.encode("utf-8"))
        digest.update(_git_file(harness_sha, path))
    return f"{IMAGE_REPOSITORY}:{digest.hexdigest()[:16]}"


def _build_image(
    tag: str,
    *,
    harness_sha: str,
    docker_platform: str,
) -> None:
    with _temporary_directory(prefix="leadpoet-restart-image-") as raw:
        context = Path(raw)
        (context / "requirements.txt").write_bytes(
            _git_file(harness_sha, "requirements.txt")
        )
        (context / "Dockerfile").write_bytes(
            _git_file(
                harness_sha,
                "tests/restart_rehearsal/Dockerfile",
            )
        )
        for path in SCORING_WHEELHOUSE_PATHS:
            (context / Path(path).name).write_bytes(
                _git_file(harness_sha, path)
            )
        for path in EXTERNAL_ARTIFACT_LOCK_PATHS:
            (context / Path(path).name).write_bytes(
                _git_file(harness_sha, path)
            )
        harness = context / "harness"
        harness.mkdir()
        for path in COMMITTED_HARNESS_PATHS[1:]:
            relative = Path(path).relative_to("tests/restart_rehearsal")
            destination = harness / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(_git_file(harness_sha, path))
        _run(
            [
                "docker",
                "build",
                "--platform",
                docker_platform,
                "--tag",
                tag,
                ".",
            ],
            cwd=context,
        )


def _verify_driver_identity(harness_sha: str) -> None:
    path = "scripts/run_local_restart_rehearsal.py"
    if Path(__file__).resolve().read_bytes() != _git_file(harness_sha, path):
        raise SystemExit(
            "restart rehearsal driver differs from the frozen harness SHA"
        )


@contextmanager
def _isolated_source_snapshot(
    *,
    harness_sha: str,
    required_shas: Sequence[str],
) -> Iterator[Path]:
    """Copy frozen Git objects so sequential containers cannot share mutations."""

    with _temporary_directory(prefix="leadpoet-restart-source-") as raw:
        source = Path(raw) / "source"
        _run(
            [
                "git",
                "clone",
                "--quiet",
                "--no-hardlinks",
                "--no-checkout",
                str(REPO_ROOT),
                str(source),
            ]
        )
        for commit in required_shas:
            present = subprocess.run(
                ["git", "-C", str(source), "cat-file", "-e", f"{commit}^{{commit}}"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if present.returncode != 0:
                _run(
                    [
                        "git",
                        "-C",
                        str(source),
                        "fetch",
                        "--quiet",
                        str(REPO_ROOT),
                        commit,
                    ]
                )
            _run(
                [
                    "git",
                    "-C",
                    str(source),
                    "cat-file",
                    "-e",
                    f"{commit}^{{commit}}",
                ]
            )
        _run(
            [
                "git",
                "-C",
                str(source),
                "checkout",
                "--quiet",
                "--detach",
                harness_sha,
            ]
        )
        _run(
            [
                "git",
                "-C",
                str(source),
                "fsck",
                "--strict",
                "--no-dangling",
            ]
        )
        yield source


def _image_exists(tag: str) -> bool:
    result = subprocess.run(
        ["docker", "image", "inspect", tag],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def _prepare_drand_artifact(
    *,
    source_root: Path,
    candidate_sha: str,
) -> Path:
    """Independently rebuild/cache the real measured drand C ABI artifact."""

    expected_hash = _git_file(
        candidate_sha,
        "validator_tee/enclave/libbittensor_drand_v2.sha256",
    ).decode("ascii").strip()
    if len(expected_hash) != 64 or any(
        character not in "0123456789abcdef" for character in expected_hash
    ):
        raise SystemExit("candidate drand C ABI hash is invalid")
    cache_root = (
        Path.home()
        / ".cache"
        / "leadpoet-local-restart-rehearsal"
        / "drand-cabi-v2"
        / expected_hash
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    cached = cache_root / "libbittensor_drand_v2.so"

    def valid(path: Path) -> bool:
        return (
            path.is_file()
            and hashlib.sha256(path.read_bytes()).hexdigest() == expected_hash
        )

    if valid(cached):
        return cache_root
    local = REPO_ROOT / ".validator-tee-artifacts/libbittensor_drand_v2.so"
    if valid(local):
        shutil.copy2(local, cached)
        cached.chmod(0o444)
        return cache_root

    lock = json.loads(
        _git_file(
            candidate_sha,
            "validator_tee/runtime-artifacts-v2.lock.json",
        )
    )
    source_contract = lock["artifacts"]["bittensor_drand_source"]
    artifact_root = source_root / ".validator-tee-artifacts"
    artifact_root.mkdir(exist_ok=True)
    source_archive = artifact_root / source_contract["filename"]
    if not source_archive.is_file():
        with urlopen(str(source_contract["url"]), timeout=120) as response:
            source_archive.write_bytes(response.read())
    if (
        hashlib.sha256(source_archive.read_bytes()).hexdigest()
        != source_contract["sha256"]
    ):
        raise SystemExit("pinned drand source archive hash differs")
    output = artifact_root / "libbittensor_drand_v2.so"
    _run(
        [
            "/bin/bash",
            str(source_root / "validator_tee/scripts/build_drand_cabi_v2.sh"),
            str(source_archive),
            str(output),
            str(
                source_root
                / "validator_tee/enclave/libbittensor_drand_v2.sha256"
            ),
        ],
        cwd=source_root,
    )
    if not valid(output):
        raise SystemExit("real drand C ABI rebuild differs from candidate hash")
    shutil.copy2(output, cached)
    cached.chmod(0o444)
    return cache_root


def _run_component(
    tag: str,
    *,
    source_root: Path,
    component: str,
    from_sha: str,
    candidate_sha: str,
    transition: str,
    evidence_root: Path,
    drand_artifact_root: Path,
    profile: str,
    limits: dict[str, Any],
    weight_readiness_scenario: str = "production_success",
    docker_platform: str,
    fixture_seed_root: Path,
    run_ordinal: int,
    gateway_worker_fleet_mode: str,
) -> None:
    command = [
        "docker",
        "run",
        "--rm",
        "--platform",
        docker_platform,
        "--network",
        "none",
        "--cpus",
        str(limits["cpus"]),
        "--memory",
        str(limits["memory"]),
        "--pids-limit",
        "2048",
        "--security-opt",
        "no-new-privileges",
        "--tmpfs",
        "/tmp:rw,exec,nosuid,size=2g",
        "--mount",
        f"type=bind,src={_docker_bind_source(source_root)},dst=/source,readonly",
        "--mount",
        f"type=bind,src={_docker_bind_source(evidence_root)},dst=/evidence",
        "--mount",
        (
            f"type=bind,src={_docker_bind_source(drand_artifact_root)},"
            "dst=/opt/leadpoet/drand-cabi-v2,readonly"
        ),
        "--mount",
        (
            f"type=bind,src={_docker_bind_source(fixture_seed_root)},"
            "dst=/rehearsal-fixture-seed,readonly"
        ),
        "--env",
        f"REHEARSAL_COMPONENT={component}",
        "--env",
        f"REHEARSAL_FROM_SHA={from_sha}",
        "--env",
        f"REHEARSAL_CANDIDATE_SHA={candidate_sha}",
        "--env",
        f"REHEARSAL_TRANSITION={transition}",
        "--env",
        (
            "REHEARSAL_WEIGHT_READINESS_SCENARIO="
            f"{weight_readiness_scenario}"
        ),
        "--env",
        "REHEARSAL_SCOPE=exact",
        "--env",
        f"REHEARSAL_PROFILE={profile}",
        "--env",
        f"REHEARSAL_RUN_ORDINAL={run_ordinal}",
        "--env",
        f"REHEARSAL_GATEWAY_WORKER_FLEET_MODE={gateway_worker_fleet_mode}",
        tag,
    ]
    try:
        _run(command)
    except BaseException:
        _preserve_failure_evidence(
            evidence_root=evidence_root,
            candidate_sha=candidate_sha,
            stage=f"{component}-{transition}-{run_ordinal}",
            command=command,
        )
        raise


def _preserve_failure_evidence(
    *,
    evidence_root: Path,
    candidate_sha: str,
    stage: str,
    command: Sequence[str],
) -> Path:
    """Keep exact-launcher diagnostics after the temporary run is removed."""

    safe_stage = "".join(
        character if character.isalnum() or character in {"-", "_"} else "-"
        for character in stage
    )
    durable_root = Path(
        tempfile.mkdtemp(
            prefix=(
                "leadpoet-rehearsal-failure-"
                f"{candidate_sha[:12]}-{safe_stage}-"
            )
        )
    )
    copied_evidence = durable_root / "evidence"
    shutil.copytree(evidence_root, copied_evidence)
    (durable_root / "failure.json").write_text(
        json.dumps(
            {
                "candidate_sha": candidate_sha,
                "command": list(command),
                "stage": stage,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"REHEARSAL_FAILURE_EVIDENCE {durable_root}",
        file=sys.stderr,
        flush=True,
    )
    return durable_root


def _preserve_batched_failure_evidence(
    *,
    evidence_root: Path,
    candidate_sha: str,
    failures: Sequence[dict[str, Any]],
) -> Path:
    """Preserve all independent release-stage failures after batch execution."""

    durable_root = Path(
        tempfile.mkdtemp(
            prefix=(
                "leadpoet-rehearsal-failure-"
                f"{candidate_sha[:12]}-release-batch-"
            )
        )
    )
    copied_evidence = durable_root / "evidence"
    shutil.copytree(evidence_root, copied_evidence)
    report = {
        "candidate_sha": candidate_sha,
        "failure_count": len(failures),
        "failures": list(failures),
        "status": "failed",
    }
    (durable_root / "failure-summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"REHEARSAL_BATCH_FAILURE_EVIDENCE {durable_root}",
        file=sys.stderr,
        flush=True,
    )
    return durable_root


def _stage_failure(
    *,
    stage: str,
    exc: subprocess.CalledProcessError,
) -> dict[str, Any]:
    return {
        "command": [str(item) for item in exc.cmd],
        "returncode": int(exc.returncode),
        "stage": stage,
    }


@contextmanager
def _prepared_fixture_seed(
    tag: str,
    *,
    source_root: Path,
    candidate_sha: str,
    drand_artifact_root: Path,
    docker_platform: str,
    limits: dict[str, Any],
) -> Iterator[Path]:
    """Build immutable sanitized release fixtures once per target SHA."""

    with _temporary_directory(
        prefix=f"leadpoet-rehearsal-fixture-{candidate_sha[:12]}-"
    ) as raw:
        root = Path(raw)
        generated_state = root / "generated-state"
        generated_config = root / "generated-config"
        seed = root / "seed"
        generated_state.mkdir()
        generated_config.mkdir()
        seed.mkdir()
        _run(
            [
                "docker",
                "run",
                "--rm",
                "--platform",
                docker_platform,
                "--network",
                "none",
                "--cpus",
                str(limits["cpus"]),
                "--memory",
                str(limits["memory"]),
                "--pids-limit",
                "2048",
                "--security-opt",
                "no-new-privileges",
                "--tmpfs",
                "/tmp:rw,exec,nosuid,size=2g",
                "--mount",
                (
                    "type=bind,"
                    f"src={_docker_bind_source(source_root)},"
                    "dst=/source,readonly"
                ),
                "--mount",
                (
                    f"type=bind,src={_docker_bind_source(drand_artifact_root)},"
                    "dst=/opt/leadpoet/drand-cabi-v2,readonly"
                ),
                "--mount",
                (
                    f"type=bind,src={_docker_bind_source(generated_state)},"
                    "dst=/rehearsal-state"
                ),
                "--mount",
                (
                    f"type=bind,src={_docker_bind_source(generated_config)},"
                    "dst=/fixture-config"
                ),
                "--env",
                "REHEARSAL_COMPONENT=validator",
                "--env",
                f"REHEARSAL_CANDIDATE_SHA={candidate_sha}",
                "--env",
                "REHEARSAL_SCOPE=exact",
                "--env",
                "REHEARSAL_STATE_ROOT=/rehearsal-state",
                "--env",
                "PYTHONPATH=/source:/harness",
                "--entrypoint",
                "/usr/bin/python3.11",
                tag,
                "/harness/prepare_host_fixtures.py",
                "--output-dir",
                "/fixture-config",
                "--candidate-sha",
                candidate_sha,
            ]
        )
        release_input = generated_state / "release-build-input.json"
        validator_app = generated_state / "validator-app"
        gateway_identities = (
            generated_state / "gateway-enclave-build-identities"
        )
        gateway_attested_runtime = (
            generated_state / "gateway-attested-runtime"
        )
        if (
            not release_input.is_file()
            or not validator_app.is_dir()
            or not gateway_identities.is_dir()
            or not gateway_attested_runtime.is_dir()
        ):
            raise SystemExit("sanitized fixture seed is incomplete")
        release = json.loads(release_input.read_text(encoding="utf-8"))
        if release.get("commit_sha") != candidate_sha:
            raise SystemExit("sanitized fixture seed commit differs")
        shutil.copytree(generated_config, seed / "config-v2")
        shutil.copy2(release_input, seed / release_input.name)
        shutil.copytree(validator_app, seed / validator_app.name)
        shutil.copytree(
            gateway_identities,
            seed / gateway_identities.name,
        )
        shutil.copytree(
            gateway_attested_runtime,
            seed / gateway_attested_runtime.name,
        )
        (seed / "fixture-seed.json").write_text(
            json.dumps(
                {
                    "schema_version": "leadpoet.local_fixture_seed.v1",
                    "candidate_sha": candidate_sha,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
        yield seed


def _run_workflow(
    tag: str,
    *,
    source_root: Path,
    evidence_root: Path,
    candidate_sha: str,
    profile: str,
    limits: dict[str, Any],
    docker_platform: str,
) -> None:
    command = [
        "docker",
        "run",
        "--rm",
        "--platform",
        docker_platform,
        "--network",
        "none",
        "--cpus",
        str(limits["cpus"]),
        "--memory",
        str(limits["memory"]),
        "--pids-limit",
        "2048",
        "--security-opt",
        "no-new-privileges",
        "--tmpfs",
        "/tmp:rw,exec,nosuid,size=2g",
        "--mount",
        f"type=bind,src={_docker_bind_source(source_root)},dst=/source,readonly",
        "--mount",
        f"type=bind,src={_docker_bind_source(evidence_root)},dst=/evidence",
        "--env",
        "REHEARSAL_COMPONENT=workflow",
        "--env",
        f"REHEARSAL_FROM_SHA={candidate_sha}",
        "--env",
        f"REHEARSAL_CANDIDATE_SHA={candidate_sha}",
        "--env",
        "REHEARSAL_TRANSITION=forward",
        "--env",
        f"REHEARSAL_PROFILE={profile}",
        "--env",
        f"REHEARSAL_EPOCHS={limits['epochs']}",
        "--env",
        "REHEARSAL_FAULT_MATRIX="
        + ("1" if limits["fault_matrix"] else "0"),
        tag,
    ]
    try:
        _run(command)
    except BaseException:
        _preserve_failure_evidence(
            evidence_root=evidence_root,
            candidate_sha=candidate_sha,
            stage=f"workflow-{profile}",
            command=command,
        )
        raise


def _join_evidence(
    tag: str,
    *,
    source_root: Path,
    evidence_root: Path,
    from_sha: str,
    candidate_sha: str,
    profile: str,
    docker_platform: str,
) -> Path:
    output = evidence_root / (
        f"leadpoet-restart-rehearsal-{candidate_sha}-{profile}.json"
    )
    command = [
            "docker",
            "run",
            "--rm",
            "--platform",
            docker_platform,
            "--network",
            "none",
            "--cpus",
            "1",
            "--memory",
            "1g",
            "--pids-limit",
            "128",
            "--security-opt",
            "no-new-privileges",
            "--mount",
            (
                "type=bind,"
                f"src={_docker_bind_source(source_root)},"
                "dst=/source,readonly"
            ),
            "--mount",
            f"type=bind,src={_docker_bind_source(evidence_root)},dst=/evidence",
            "--entrypoint",
            "/usr/bin/python3.11",
            tag,
            "/harness/join_evidence.py",
            "--evidence-root",
            "/evidence",
            "--from-sha",
            from_sha,
            "--candidate-sha",
            candidate_sha,
            "--profile",
            profile,
            "--output",
            f"/evidence/{output.name}",
        ]
    try:
        _run(command)
        if not output.is_file():
            raise SystemExit(
                "joined restart rehearsal evidence was not produced"
            )
    except BaseException:
        _preserve_failure_evidence(
            evidence_root=evidence_root,
            candidate_sha=candidate_sha,
            stage=f"join-{profile}",
            command=command,
        )
        raise
    return output


def _run_python37_finalization_probe(source_root: Path) -> None:
    """Exercise the measured enclave's post-broadcast path under CPython 3.7."""

    _run(
        [
            "docker",
            "run",
            "--rm",
            "--platform",
            "linux/amd64",
            "--network",
            "none",
            "--cpus",
            "1",
            "--memory",
            "512m",
            "--pids-limit",
            "128",
            "--security-opt",
            "no-new-privileges",
            "--mount",
            (
                "type=bind,"
                f"src={_docker_bind_source(source_root)},"
                "dst=/source,readonly"
            ),
            "--env",
            "PYTHONPATH=/source",
            "--workdir",
            "/source",
            PYTHON37_IMAGE,
            "python",
            "tests/validator_enclave_python37_runtime_probe.py",
        ]
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--from-sha",
        default="HEAD^",
        help="Currently deployed N-1 commit whose installed launcher starts the test.",
    )
    parser.add_argument("--candidate-sha", default="HEAD")
    parser.add_argument(
        "--transition",
        choices=("auto", "forward", "rollback"),
        default="auto",
    )
    parser.add_argument(
        "--profile",
        choices=tuple(PROFILE_LIMITS),
        default="prepush",
        help="prepush is the bounded developer gate; release runs the full matrix",
    )
    parser.add_argument(
        "--scope",
        choices=("exact",),
        default="exact",
        help="only the exact repository-owned rehearsal is supported",
    )
    parser.add_argument(
        "--component",
        choices=("all",),
        default="all",
        help="the exact rehearsal always executes gateway and validator",
    )
    parser.add_argument(
        "--outer-cpus",
        help=(
            "cap each primary outer Docker replica below the profile CPU limit; "
            "does not alter the enforced production 16-vCPU contract"
        ),
    )
    parser.add_argument(
        "--outer-memory",
        help=(
            "cap each primary outer Docker replica below the profile memory "
            "limit; does not alter the enforced production 128-GiB contract"
        ),
    )
    parser.add_argument(
        "--gateway-worker-fleet-mode",
        choices=("active", "deferred"),
        default="active",
        help=(
            "active exercises compliant TLS proxy workers; deferred exercises "
            "the explicit one-restart recovery path while retaining the same "
            "compliant TLS proxy validation"
        ),
    )
    parser.add_argument("--rebuild-image", action="store_true")
    args = parser.parse_args(argv)
    limits = _effective_profile_limits(
        args.profile,
        outer_cpus=args.outer_cpus,
        outer_memory=args.outer_memory,
    )

    from_sha = _git_sha(args.from_sha)
    candidate_sha = _git_sha(args.candidate_sha)
    transition = _resolve_transition(
        from_sha,
        candidate_sha,
        args.transition,
    )
    if transition != "forward":
        raise SystemExit(
            "profile rehearsals must start with the deployed N-1 commit and "
            "a descendant candidate; the release profile performs its own "
            "rollback and roll-forward"
        )
    harness_sha = candidate_sha
    _verify_driver_identity(harness_sha)
    print(
        "REHEARSAL_OUTER_LIMITS "
        f"cpus={limits['cpus']} memory={limits['memory']} "
        "production_capacity=16-vcpu/128-gib contract_enforced=true",
        flush=True,
    )

    docker_platform = _docker_platform(args.profile)
    tag = _image_tag(harness_sha, docker_platform=docker_platform)
    if args.rebuild_image or not _image_exists(tag):
        _build_image(
            tag,
            harness_sha=harness_sha,
            docker_platform=docker_platform,
        )

    with _isolated_source_snapshot(
        harness_sha=harness_sha,
        required_shas=(from_sha, candidate_sha),
    ) as source_root:
        with _temporary_directory(
            prefix="leadpoet-restart-evidence-"
        ) as evidence_raw:
            evidence_root = Path(evidence_raw)
            print(
                "Running validator enclave finalization proof under CPython 3.7",
                flush=True,
            )
            _run_python37_finalization_probe(source_root)
            transitions = (transition,)
            if args.profile == "release" and transition == "forward":
                transitions = ("forward", "rollback", "forward")
            target_shas = {
                candidate_sha if run_transition != "rollback" else from_sha
                for run_transition in transitions
            }
            with ExitStack() as fixture_stack:
                drand_artifacts = {
                    target: _prepare_drand_artifact(
                        source_root=source_root,
                        candidate_sha=target,
                    )
                    for target in target_shas
                }
                fixture_seeds = {
                    target: fixture_stack.enter_context(
                        _prepared_fixture_seed(
                            tag,
                            source_root=source_root,
                            candidate_sha=target,
                            drand_artifact_root=drand_artifacts[target],
                            docker_platform=docker_platform,
                            limits=limits,
                        )
                    )
                    for target in target_shas
                }
                release_failures = []  # type: list[dict[str, Any]]
                for ordinal, run_transition in enumerate(transitions):
                    run_from = from_sha
                    run_candidate = candidate_sha
                    if run_transition == "rollback":
                        run_from, run_candidate = candidate_sha, from_sha
                    elif ordinal == 2:
                        run_from, run_candidate = from_sha, candidate_sha
                    for component in ("gateway", "validator"):
                        print(
                            f"Running isolated {component} restart rehearsal "
                            f"{run_from[:12]} -> {run_candidate[:12]} "
                            f"transition={run_transition} profile={args.profile}",
                            flush=True,
                        )
                        try:
                            _run_component(
                                tag,
                                source_root=source_root,
                                component=component,
                                from_sha=run_from,
                                candidate_sha=run_candidate,
                                transition=run_transition,
                                evidence_root=evidence_root,
                                drand_artifact_root=drand_artifacts[
                                    run_candidate
                                ],
                                profile=args.profile,
                                limits=limits,
                                docker_platform=docker_platform,
                                fixture_seed_root=fixture_seeds[run_candidate],
                                run_ordinal=ordinal + 1,
                                gateway_worker_fleet_mode=(
                                    args.gateway_worker_fleet_mode
                                ),
                            )
                        except subprocess.CalledProcessError as exc:
                            if args.profile != "release":
                                raise
                            stage = (
                                f"{component}-{run_transition}-"
                                f"{ordinal + 1}"
                            )
                            release_failures.append(
                                _stage_failure(stage=stage, exc=exc)
                            )
                            print(
                                "REHEARSAL_STAGE_FAILED_CONTINUING "
                                f"stage={stage} returncode={exc.returncode}",
                                file=sys.stderr,
                                flush=True,
                            )
            print(
                "Running production V2 workflow against strict local "
                f"boundaries ({args.profile})",
                flush=True,
            )
            try:
                _run_workflow(
                    tag,
                    source_root=source_root,
                    candidate_sha=candidate_sha,
                    evidence_root=evidence_root,
                    profile=args.profile,
                    limits=limits,
                    docker_platform=docker_platform,
                )
            except subprocess.CalledProcessError as exc:
                if args.profile != "release":
                    raise
                release_failures.append(
                    _stage_failure(
                        stage=f"workflow-{args.profile}",
                        exc=exc,
                    )
                )
                print(
                    "REHEARSAL_STAGE_FAILED_CONTINUING "
                    f"stage=workflow-{args.profile} "
                    f"returncode={exc.returncode}",
                    file=sys.stderr,
                    flush=True,
                )
            if release_failures:
                durable_failure = _preserve_batched_failure_evidence(
                    evidence_root=evidence_root,
                    candidate_sha=candidate_sha,
                    failures=release_failures,
                )
                raise SystemExit(
                    "release rehearsal failed after completing independent "
                    f"stages; evidence={durable_failure}"
                )
            evidence = _join_evidence(
                tag,
                source_root=source_root,
                evidence_root=evidence_root,
                from_sha=from_sha,
                candidate_sha=candidate_sha,
                profile=args.profile,
                docker_platform=docker_platform,
            )
            durable_output = Path(tempfile.gettempdir()) / evidence.name
            durable_output.write_bytes(evidence.read_bytes())
            print(f"REHEARSAL_EVIDENCE {durable_output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
