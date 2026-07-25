#!/usr/bin/env python3
"""Run exact gateway and validator restart launchers without production access."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
IMAGE_REPOSITORY = "leadpoet-local-restart-rehearsal"
TARGET_PLATFORM = "linux/amd64"
TARGET_IMAGE_ARCHITECTURE = "amd64"
DEFAULT_OUTER_CPUS = "4"
DEFAULT_OUTER_MEMORY = "6g"
PYTHON37_IMAGE = (
    "python@sha256:"
    "b53f496ca43e5af6994f8e316cf03af31050bf7944e0e4a308ad86c001cf028b"
)
COMMITTED_HARNESS_PATHS = (
    "tests/restart_rehearsal/Dockerfile",
    "tests/restart_rehearsal/contract_adapter.py",
    "tests/restart_rehearsal/contract_policy.py",
    "tests/restart_rehearsal/weight_readiness_runner.py",
    "tests/restart_rehearsal/run_inside.sh",
    "tests/restart_rehearsal/verify_evidence.py",
)


def _run(
    args: Sequence[str],
    *,
    cwd: Path = REPO_ROOT,
    capture: bool = False,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(args),
        cwd=str(cwd),
        check=True,
        text=True,
        capture_output=capture,
        env=env,
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


def _image_tag(candidate_sha: str) -> str:
    digest = hashlib.sha256()
    digest.update(b"candidate_sha")
    digest.update(candidate_sha.encode("ascii"))
    digest.update(b"requirements.txt")
    digest.update(_git_file(candidate_sha, "requirements.txt"))
    for path in COMMITTED_HARNESS_PATHS:
        digest.update(path.encode("utf-8"))
        digest.update(_git_file(candidate_sha, path))
    return f"{IMAGE_REPOSITORY}:{digest.hexdigest()[:16]}"


def _pinned_base_image(candidate_sha: str) -> str:
    dockerfile = _git_file(
        candidate_sha,
        "tests/restart_rehearsal/Dockerfile",
    ).decode("utf-8")
    for raw_line in dockerfile.splitlines():
        tokens = raw_line.strip().split()
        if not tokens or tokens[0].upper() != "FROM":
            continue
        image = next(
            (token for token in tokens[1:] if not token.startswith("--")),
            "",
        )
        if not re.fullmatch(r"[^\s]+@sha256:[0-9a-f]{64}", image):
            raise SystemExit(
                "restart rehearsal base image must use an exact sha256 digest"
            )
        return image
    raise SystemExit("restart rehearsal Dockerfile has no pinned base image")


def _image_architecture(tag: str) -> str:
    result = _run(
        [
            "docker",
            "image",
            "inspect",
            "--format",
            "{{.Architecture}}",
            tag,
        ],
        capture=True,
    )
    return result.stdout.strip()


def _verify_image_architecture(tag: str) -> None:
    architecture = _image_architecture(tag)
    if architecture != TARGET_IMAGE_ARCHITECTURE:
        raise SystemExit(
            "restart rehearsal image architecture is "
            f"{architecture or '<unknown>'}; expected {TARGET_IMAGE_ARCHITECTURE}"
        )
    result = _run(
        [
            "docker",
            "run",
            "--rm",
            "--platform",
            TARGET_PLATFORM,
            "--entrypoint",
            "/usr/bin/uname",
            tag,
            "-m",
        ],
        capture=True,
    )
    if result.stdout.strip() != "x86_64":
        raise SystemExit(
            "restart rehearsal image did not execute with Linux AMD64 semantics"
        )


def _build_image(tag: str, *, candidate_sha: str) -> None:
    base_image = _pinned_base_image(candidate_sha)
    _run(
        [
            "docker",
            "pull",
            "--platform",
            TARGET_PLATFORM,
            base_image,
        ]
    )
    with tempfile.TemporaryDirectory(prefix="leadpoet-restart-image-") as raw:
        context = Path(raw)
        (context / "requirements.txt").write_bytes(
            _git_file(candidate_sha, "requirements.txt")
        )
        (context / "Dockerfile").write_bytes(
            _git_file(
                candidate_sha,
                "tests/restart_rehearsal/Dockerfile",
            )
        )
        harness = context / "harness"
        harness.mkdir()
        for path in COMMITTED_HARNESS_PATHS[1:]:
            (harness / Path(path).name).write_bytes(
                _git_file(candidate_sha, path)
            )
        _run(
            [
                "docker",
                "build",
                "--platform",
                TARGET_PLATFORM,
                "--tag",
                tag,
                ".",
            ],
            cwd=context,
            env={**os.environ, "DOCKER_BUILDKIT": "0"},
        )
    _verify_image_architecture(tag)


def _verify_driver_identity(candidate_sha: str) -> None:
    path = "scripts/run_local_restart_rehearsal.py"
    if Path(__file__).resolve().read_bytes() != _git_file(candidate_sha, path):
        raise SystemExit(
            "restart rehearsal driver differs from the frozen candidate SHA"
        )


def _image_exists(tag: str) -> bool:
    try:
        return _image_architecture(tag) == TARGET_IMAGE_ARCHITECTURE
    except subprocess.CalledProcessError:
        return False


def _run_component(
    tag: str,
    *,
    component: str,
    from_sha: str,
    candidate_sha: str,
    weight_readiness_scenario: str = "transient_503_recovery",
    scope: str = "exact",
    outer_cpus: str = DEFAULT_OUTER_CPUS,
    outer_memory: str = DEFAULT_OUTER_MEMORY,
) -> None:
    command = [
        "docker",
        "run",
        "--rm",
        "--platform",
        TARGET_PLATFORM,
        "--network",
        "none",
        "--cpus",
        outer_cpus,
        "--memory",
        outer_memory,
        "--pids-limit",
        "2048",
        "--security-opt",
        "no-new-privileges",
        "--tmpfs",
        "/tmp:rw,exec,nosuid,size=2g",
        "--mount",
        f"type=bind,src={REPO_ROOT},dst=/source,readonly",
        "--env",
        f"REHEARSAL_COMPONENT={component}",
        "--env",
        f"REHEARSAL_FROM_SHA={from_sha}",
        "--env",
        f"REHEARSAL_CANDIDATE_SHA={candidate_sha}",
        "--env",
        (
            "REHEARSAL_WEIGHT_READINESS_SCENARIO="
            f"{weight_readiness_scenario}"
        ),
        "--env",
        f"REHEARSAL_SCOPE={scope}",
        "--env",
        f"REHEARSAL_OUTER_CPUS={outer_cpus}",
        "--env",
        f"REHEARSAL_OUTER_MEMORY={outer_memory}",
        tag,
    ]
    _run(command)


def _run_python37_finalization_probe() -> None:
    """Exercise the measured enclave's post-broadcast path under CPython 3.7."""

    _run(
        [
            "docker",
            "run",
            "--rm",
            "--platform",
            TARGET_PLATFORM,
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
            f"type=bind,src={REPO_ROOT},dst=/source,readonly",
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
        "--component",
        choices=("all", "gateway", "validator"),
        default="all",
    )
    parser.add_argument(
        "--scope",
        choices=("exact", "weight-readiness-regression"),
        default="exact",
        help=(
            "exact rejects every repository-code substitution; "
            "weight-readiness-regression runs only the labelled gateway "
            "restart fault matrix and is not deployment evidence"
        ),
    )
    parser.add_argument("--rebuild-image", action="store_true")
    parser.add_argument(
        "--outer-cpus",
        default=DEFAULT_OUTER_CPUS,
        help=(
            "CPU limit for the outer local container. The strict capacity "
            "adapter still exposes and validates the production 16-vCPU contract."
        ),
    )
    parser.add_argument(
        "--outer-memory",
        default=DEFAULT_OUTER_MEMORY,
        help=(
            "Memory limit for the outer local container. The strict capacity "
            "adapter still exposes and validates the production 128-GiB contract."
        ),
    )
    args = parser.parse_args(argv)

    try:
        if float(args.outer_cpus) <= 0:
            raise ValueError
    except ValueError:
        parser.error("--outer-cpus must be a positive number")
    if not re.fullmatch(r"[1-9][0-9]*(?:[bkmgBKMG])?", args.outer_memory):
        parser.error(
            "--outer-memory must be a positive Docker memory value such as 6g"
        )

    from_sha = _git_sha(args.from_sha)
    candidate_sha = _git_sha(args.candidate_sha)
    _run(["git", "merge-base", "--is-ancestor", from_sha, candidate_sha])
    _verify_driver_identity(candidate_sha)

    tag = _image_tag(candidate_sha)
    if args.rebuild_image or not _image_exists(tag):
        _build_image(tag, candidate_sha=candidate_sha)
    else:
        _verify_image_architecture(tag)

    components = (
        ("gateway", "validator") if args.component == "all" else (args.component,)
    )
    print(
        "REHEARSAL_RESOURCE_PROFILE "
        f"platform={TARGET_PLATFORM} "
        f"outer_cpus={args.outer_cpus} "
        f"outer_memory={args.outer_memory} "
        "advertised_cpus=16 advertised_memory=128g "
        "physical_pressure=simulated",
        flush=True,
    )
    if "validator" in components:
        print(
            "Running validator enclave finalization proof under CPython 3.7",
            flush=True,
        )
        _run_python37_finalization_probe()
    for component in components:
        scenarios = (
            (
                "transient_503_recovery",
                "exhausted_503",
                "authenticated_403",
            )
            if (
                component == "gateway"
                and args.scope == "weight-readiness-regression"
            )
            else ("transient_503_recovery",)
        )
        for scenario in scenarios:
            print(
                f"Running isolated {component} restart rehearsal "
                f"{from_sha[:12]} -> {candidate_sha[:12]} "
                f"scenario={scenario}",
                flush=True,
            )
            _run_component(
                tag,
                component=component,
                from_sha=from_sha,
                candidate_sha=candidate_sha,
                weight_readiness_scenario=scenario,
                scope=args.scope.replace("-", "_"),
                outer_cpus=args.outer_cpus,
                outer_memory=args.outer_memory,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
