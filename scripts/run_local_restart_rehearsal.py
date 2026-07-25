#!/usr/bin/env python3
"""Run exact gateway and validator restart launchers without production access."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
IMAGE_REPOSITORY = "leadpoet-local-restart-rehearsal"
PYTHON37_IMAGE = (
    "python@sha256:"
    "b53f496ca43e5af6994f8e316cf03af31050bf7944e0e4a308ad86c001cf028b"
)
COMMITTED_HARNESS_PATHS = (
    "tests/restart_rehearsal/Dockerfile",
    "tests/restart_rehearsal/contract_adapter.py",
    "tests/restart_rehearsal/weight_readiness_runner.py",
    "tests/restart_rehearsal/run_inside.sh",
    "tests/restart_rehearsal/verify_evidence.py",
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


def _image_tag(harness_sha: str) -> str:
    digest = hashlib.sha256()
    digest.update(b"harness_sha")
    digest.update(harness_sha.encode("ascii"))
    digest.update(b"requirements.txt")
    digest.update(_git_file(harness_sha, "requirements.txt"))
    for path in COMMITTED_HARNESS_PATHS:
        digest.update(path.encode("utf-8"))
        digest.update(_git_file(harness_sha, path))
    return f"{IMAGE_REPOSITORY}:{digest.hexdigest()[:16]}"


def _build_image(tag: str, *, harness_sha: str) -> None:
    with tempfile.TemporaryDirectory(prefix="leadpoet-restart-image-") as raw:
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
        harness = context / "harness"
        harness.mkdir()
        for path in COMMITTED_HARNESS_PATHS[1:]:
            (harness / Path(path).name).write_bytes(
                _git_file(harness_sha, path)
            )
        _run(
            [
                "docker",
                "build",
                "--platform",
                "linux/amd64",
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


def _image_exists(tag: str) -> bool:
    result = subprocess.run(
        ["docker", "image", "inspect", tag],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def _run_component(
    tag: str,
    *,
    component: str,
    from_sha: str,
    candidate_sha: str,
    transition: str,
    weight_readiness_scenario: str = "transient_503_recovery",
    scope: str = "exact",
) -> None:
    exact = scope == "exact"
    command = [
        "docker",
        "run",
        "--rm",
        "--platform",
        "linux/amd64",
        "--network",
        "none",
        "--cpus",
        "16" if exact else "4",
        "--memory",
        "128g" if exact else "6g",
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
        f"REHEARSAL_TRANSITION={transition}",
        "--env",
        (
            "REHEARSAL_WEIGHT_READINESS_SCENARIO="
            f"{weight_readiness_scenario}"
        ),
        "--env",
        f"REHEARSAL_SCOPE={scope}",
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
        "--transition",
        choices=("auto", "forward", "rollback"),
        default="auto",
    )
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
    args = parser.parse_args(argv)

    from_sha = _git_sha(args.from_sha)
    candidate_sha = _git_sha(args.candidate_sha)
    transition = _resolve_transition(
        from_sha,
        candidate_sha,
        args.transition,
    )
    harness_sha = candidate_sha if transition == "forward" else from_sha
    _verify_driver_identity(harness_sha)

    tag = _image_tag(harness_sha)
    if args.rebuild_image or not _image_exists(tag):
        _build_image(tag, harness_sha=harness_sha)

    components = (
        ("gateway", "validator") if args.component == "all" else (args.component,)
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
                f"transition={transition} scenario={scenario}",
                flush=True,
            )
            _run_component(
                tag,
                component=component,
                from_sha=from_sha,
                candidate_sha=candidate_sha,
                transition=transition,
                weight_readiness_scenario=scenario,
                scope=args.scope.replace("-", "_"),
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
