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
import hashlib
import json
import platform
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Callable, Iterator, Sequence
from urllib.request import urlopen


REPO_ROOT = Path(__file__).resolve().parents[1]
IMAGE_REPOSITORY = "leadpoet-local-restart-rehearsal"
REHEARSAL_BASE_IMAGES = {
    "linux/amd64": (
        "public.ecr.aws/amazonlinux/amazonlinux@sha256:"
        "7dfb72e165c7b2f5fd2ee050c202160ee0cced24991f14736b831221f2004eee"
    ),
    "linux/arm64": (
        "public.ecr.aws/amazonlinux/amazonlinux@sha256:"
        "d23b77c815875a32165bc160248a6fcaf932dbbcdb7adc157680c39e4d254b38"
    ),
}
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
    "tests/restart_rehearsal/postgres_v2_contract_probe.py",
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


def _rehearsal_base_image(docker_platform: str) -> str:
    try:
        return REHEARSAL_BASE_IMAGES[docker_platform]
    except KeyError as exc:
        raise SystemExit(
            f"unsupported rehearsal Docker platform: {docker_platform}"
        ) from exc


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
                "--build-arg",
                "REHEARSAL_BASE_IMAGE="
                + _rehearsal_base_image(docker_platform),
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

    with tempfile.TemporaryDirectory(
        prefix="leadpoet-restart-source-"
    ) as raw:
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
    weight_readiness_scenario: str = "production_success",
    docker_platform: str,
    fixture_seed_root: Path,
    run_ordinal: int,
    gateway_worker_fleet_mode: str,
) -> None:
    limits = PROFILE_LIMITS[profile]
    launcher_log = evidence_root / (
        f"{run_ordinal}-{component}-{transition}-{candidate_sha}-launcher.log"
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
        f"type=bind,src={source_root},dst=/source,readonly",
        "--mount",
        f"type=bind,src={evidence_root},dst=/evidence",
        "--mount",
        (
            f"type=bind,src={drand_artifact_root},"
            "dst=/opt/leadpoet/drand-cabi-v2,readonly"
        ),
        "--mount",
        (
            f"type=bind,src={fixture_seed_root},"
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
        with launcher_log.open("w", encoding="utf-8") as output:
            process = subprocess.Popen(
                command,
                cwd=str(REPO_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            try:
                assert process.stdout is not None
                for line in process.stdout:
                    output.write(line)
                    output.flush()
                    print(line, end="", flush=True)
                returncode = process.wait()
            except BaseException:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise
        if returncode:
            raise subprocess.CalledProcessError(returncode, command)
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
    stages: Sequence[dict[str, Any]],
) -> Path:
    """Preserve the complete full-path stage ledger after a failed run."""

    durable_root = Path(
        tempfile.mkdtemp(
            prefix=(
                "leadpoet-rehearsal-failure-"
                f"{candidate_sha[:12]}-full-path-"
            )
        )
    )
    copied_evidence = durable_root / "evidence"
    shutil.copytree(evidence_root, copied_evidence)
    failures = [item for item in stages if item.get("status") == "failed"]
    unexercised = [
        item for item in stages if item.get("status") == "unexercised"
    ]
    report = {
        "candidate_sha": candidate_sha,
        "failure_count": len(failures),
        "failures": failures,
        "stage_count": len(stages),
        "stages": list(stages),
        "status": "failed",
        "unexercised_count": len(unexercised),
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


def _stage_result_from_exception(
    *,
    stage: str,
    exc: BaseException,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "error": str(exc)[:2000],
        "error_type": type(exc).__name__,
        "stage": stage,
        "status": "failed",
    }
    if isinstance(exc, subprocess.CalledProcessError):
        result.update(_stage_failure(stage=stage, exc=exc))
        result["status"] = "failed"
    return result


def _run_independent_stage(
    *,
    stage: str,
    action: Callable[[], Any],
    stages: list[dict[str, Any]],
) -> tuple[bool, Any]:
    """Run one stage without suppressing later independent diagnostics."""

    try:
        value = action()
    except KeyboardInterrupt:
        raise
    except BaseException as exc:
        result = _stage_result_from_exception(stage=stage, exc=exc)
        stages.append(result)
        print(
            "REHEARSAL_STAGE_FAILED_CONTINUING "
            f"stage={stage} error_type={result['error_type']} "
            f"error={result['error']!r}",
            file=sys.stderr,
            flush=True,
        )
        return False, None
    stages.append({"stage": stage, "status": "passed"})
    print(f"REHEARSAL_STAGE_PASSED stage={stage}", flush=True)
    return True, value


def _mark_stage_unexercised(
    *,
    stage: str,
    blocked_by: Sequence[str],
    stages: list[dict[str, Any]],
) -> None:
    result = {
        "blocked_by": list(blocked_by),
        "stage": stage,
        "status": "unexercised",
    }
    stages.append(result)
    print(
        "REHEARSAL_STAGE_UNEXERCISED "
        f"stage={stage} blocked_by={','.join(blocked_by)}",
        file=sys.stderr,
        flush=True,
    )


def _write_stage_summary(
    *,
    evidence_root: Path,
    candidate_sha: str,
    profile: str,
    stages: Sequence[dict[str, Any]],
) -> Path:
    failed = sum(item.get("status") == "failed" for item in stages)
    unexercised = sum(
        item.get("status") == "unexercised" for item in stages
    )
    output = evidence_root / (
        f"leadpoet-restart-rehearsal-{candidate_sha}-{profile}-stages.json"
    )
    output.write_text(
        json.dumps(
            {
                "candidate_sha": candidate_sha,
                "failure_count": failed,
                "profile": profile,
                "schema_version": (
                    "leadpoet.local_restart_rehearsal_stage_summary.v1"
                ),
                "stage_count": len(stages),
                "stages": list(stages),
                "status": (
                    "passed" if failed == 0 and unexercised == 0 else "failed"
                ),
                "unexercised_count": unexercised,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return output


@contextmanager
def _recording_fixture_stack(
    stages: list[dict[str, Any]],
) -> Iterator[ExitStack]:
    stack = ExitStack()
    try:
        yield stack
    except KeyboardInterrupt:
        raise
    except BaseException as exc:
        result = _stage_result_from_exception(
            stage="fixture-orchestration",
            exc=exc,
        )
        stages.append(result)
        print(
            "REHEARSAL_STAGE_FAILED_CONTINUING "
            "stage=fixture-orchestration "
            f"error_type={result['error_type']} error={result['error']!r}",
            file=sys.stderr,
            flush=True,
        )
    finally:
        _run_independent_stage(
            stage="fixture-cleanup",
            action=stack.close,
            stages=stages,
        )


@contextmanager
def _prepared_fixture_seed(
    tag: str,
    *,
    source_root: Path,
    candidate_sha: str,
    drand_artifact_root: Path,
    docker_platform: str,
    profile: str,
) -> Iterator[Path]:
    """Build immutable sanitized release fixtures once per target SHA."""

    limits = PROFILE_LIMITS[profile]
    with tempfile.TemporaryDirectory(
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
                f"type=bind,src={source_root},dst=/source,readonly",
                "--mount",
                (
                    f"type=bind,src={drand_artifact_root},"
                    "dst=/opt/leadpoet/drand-cabi-v2,readonly"
                ),
                "--mount",
                (
                    f"type=bind,src={generated_state},"
                    "dst=/rehearsal-state"
                ),
                "--mount",
                (
                    f"type=bind,src={generated_config},"
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
    docker_platform: str,
) -> None:
    limits = PROFILE_LIMITS[profile]
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
        f"type=bind,src={source_root},dst=/source,readonly",
        "--mount",
        f"type=bind,src={evidence_root},dst=/evidence",
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
            f"type=bind,src={source_root},dst=/source,readonly",
            "--mount",
            f"type=bind,src={evidence_root},dst=/evidence",
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
            f"type=bind,src={source_root},dst=/source,readonly",
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
        with tempfile.TemporaryDirectory(
            prefix="leadpoet-restart-evidence-"
        ) as evidence_raw:
            evidence_root = Path(evidence_raw)
            stage_results: list[dict[str, Any]] = []
            print(
                "Running validator enclave finalization proof under CPython 3.7",
                flush=True,
            )
            _run_independent_stage(
                stage="python37-finalization",
                action=lambda: _run_python37_finalization_probe(source_root),
                stages=stage_results,
            )
            transitions = (transition,)
            if args.profile == "release" and transition == "forward":
                transitions = ("forward", "rollback", "forward")
            target_shas = {
                candidate_sha if run_transition != "rollback" else from_sha
                for run_transition in transitions
            }
            with _recording_fixture_stack(stage_results) as fixture_stack:
                drand_artifacts: dict[str, Path] = {}
                fixture_seeds: dict[str, Path] = {}
                for target in sorted(target_shas):
                    stage = f"drand-artifact-{target[:12]}"
                    passed, artifact = _run_independent_stage(
                        stage=stage,
                        action=lambda target=target: _prepare_drand_artifact(
                            source_root=source_root,
                            candidate_sha=target,
                        ),
                        stages=stage_results,
                    )
                    if passed:
                        drand_artifacts[target] = artifact
                for target in sorted(target_shas):
                    stage = f"fixture-seed-{target[:12]}"
                    dependency = f"drand-artifact-{target[:12]}"
                    if target not in drand_artifacts:
                        _mark_stage_unexercised(
                            stage=stage,
                            blocked_by=[dependency],
                            stages=stage_results,
                        )
                        continue
                    passed, seed = _run_independent_stage(
                        stage=stage,
                        action=lambda target=target: fixture_stack.enter_context(
                            _prepared_fixture_seed(
                                tag,
                                source_root=source_root,
                                candidate_sha=target,
                                drand_artifact_root=drand_artifacts[target],
                                docker_platform=docker_platform,
                                profile=args.profile,
                            )
                        ),
                        stages=stage_results,
                    )
                    if passed:
                        fixture_seeds[target] = seed
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
                        stage = (
                            f"{component}-{run_transition}-{ordinal + 1}"
                        )
                        blocked_by = []
                        if run_candidate not in drand_artifacts:
                            blocked_by.append(
                                f"drand-artifact-{run_candidate[:12]}"
                            )
                        if run_candidate not in fixture_seeds:
                            blocked_by.append(
                                f"fixture-seed-{run_candidate[:12]}"
                            )
                        if blocked_by:
                            _mark_stage_unexercised(
                                stage=stage,
                                blocked_by=blocked_by,
                                stages=stage_results,
                            )
                            continue
                        _run_independent_stage(
                            stage=stage,
                            action=lambda component=component,
                            run_from=run_from,
                            run_candidate=run_candidate,
                            run_transition=run_transition,
                            ordinal=ordinal: _run_component(
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
                                docker_platform=docker_platform,
                                fixture_seed_root=fixture_seeds[run_candidate],
                                run_ordinal=ordinal + 1,
                                gateway_worker_fleet_mode=(
                                    args.gateway_worker_fleet_mode
                                ),
                            ),
                            stages=stage_results,
                        )
            print(
                "Running production V2 workflow against strict local "
                f"boundaries ({args.profile})",
                flush=True,
            )
            workflow_stage = f"workflow-{args.profile}"
            _run_independent_stage(
                stage=workflow_stage,
                action=lambda: _run_workflow(
                    tag,
                    source_root=source_root,
                    candidate_sha=candidate_sha,
                    evidence_root=evidence_root,
                    profile=args.profile,
                    docker_platform=docker_platform,
                ),
                stages=stage_results,
            )
            required_join_stages = [
                item["stage"]
                for item in stage_results
                if (
                    item["stage"].startswith(("gateway-", "validator-"))
                    or item["stage"] == workflow_stage
                )
            ]
            stage_status = {
                item["stage"]: item["status"] for item in stage_results
            }
            evidence: Path | None = None
            join_stage = f"evidence-join-{args.profile}"
            blocked_join_stages = [
                stage
                for stage in required_join_stages
                if stage_status.get(stage) != "passed"
            ]
            if blocked_join_stages:
                _mark_stage_unexercised(
                    stage=join_stage,
                    blocked_by=blocked_join_stages,
                    stages=stage_results,
                )
            else:
                _, evidence = _run_independent_stage(
                    stage=join_stage,
                    action=lambda: _join_evidence(
                        tag,
                        source_root=source_root,
                        evidence_root=evidence_root,
                        from_sha=from_sha,
                        candidate_sha=candidate_sha,
                        profile=args.profile,
                        docker_platform=docker_platform,
                    ),
                    stages=stage_results,
                )
            stage_summary = _write_stage_summary(
                evidence_root=evidence_root,
                candidate_sha=candidate_sha,
                profile=args.profile,
                stages=stage_results,
            )
            incomplete = [
                item
                for item in stage_results
                if item.get("status") != "passed"
            ]
            if incomplete:
                durable_failure = _preserve_batched_failure_evidence(
                    evidence_root=evidence_root,
                    candidate_sha=candidate_sha,
                    stages=stage_results,
                )
                raise SystemExit(
                    f"{args.profile} rehearsal failed after completing independent "
                    f"stages; evidence={durable_failure}"
                )
            if evidence is None:
                raise SystemExit("joined rehearsal evidence is unexpectedly absent")
            durable_output = Path(tempfile.gettempdir()) / evidence.name
            durable_output.write_bytes(evidence.read_bytes())
            durable_stage_output = (
                Path(tempfile.gettempdir()) / stage_summary.name
            )
            durable_stage_output.write_bytes(stage_summary.read_bytes())
            print(f"REHEARSAL_EVIDENCE {durable_output}", flush=True)
            print(
                f"REHEARSAL_STAGE_EVIDENCE {durable_stage_output}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
