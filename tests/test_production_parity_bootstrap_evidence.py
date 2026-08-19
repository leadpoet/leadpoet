from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from botocore.exceptions import ClientError
import pytest
import yaml

from scripts import poll_production_parity_ssm as poller
from scripts import production_parity_bootstrap_evidence as evidence


RUN_ID = "pp-32241272808-1"
BASE_SHA = "a" * 40
CANDIDATE_SHA = "b" * 40
BUCKET = "leadpoet-parity-493765492819-0123456789abcdef"


def _output(tmp_path: Path) -> Path:
    return tmp_path / "full-evidence.json"


@pytest.mark.parametrize(
    ("stage", "category"),
    [
        ("bootstrap-environment", "CommandFailed"),
        ("bootstrap-workspace", "CommandFailed"),
        ("candidate-bundle-download", "CommandFailed"),
        ("candidate-clone", "CommandFailed"),
        ("canonical-origin-fetch", "CommandFailed"),
        ("candidate-checkout", "CommandFailed"),
        ("host-python-import", "HostImportFailed"),
        ("host-entrypoint", "HostEntrypointFailed"),
        ("evidence-upload", "EvidenceUploadFailed"),
        ("ssm-command", "SsmFailed"),
    ],
)
def test_each_bootstrap_failure_is_one_bounded_v3_document(
    tmp_path: Path,
    stage: str,
    category: str,
) -> None:
    output = _output(tmp_path)
    secret = "provider-secret-must-never-appear"

    payload, created = evidence.retain_failure(
        output=output,
        run_id=RUN_ID,
        base_sha=BASE_SHA,
        candidate_sha=CANDIDATE_SHA,
        stage=stage,
        error_category=category,
    )

    assert created is True
    value = json.loads(payload)
    assert value == {
        "schema_version": "leadpoet.production_parity_full.v3",
        "run_id": RUN_ID,
        "candidate_sha": CANDIDATE_SHA,
        "base_sha": BASE_SHA,
        "status": "failed",
        "failure_stage": stage,
        "error_type": category,
        "cleanup": {},
    }
    assert output.read_bytes() == payload
    assert len(payload) <= evidence.MAX_EVIDENCE_BYTES
    assert secret.encode() not in payload
    assert not list(tmp_path.glob(".full-evidence.*.tmp"))


@pytest.mark.parametrize(
    "overrides",
    [
        {"run_id": "../../unsafe"},
        {"base_sha": "a" * 39},
        {"candidate_sha": "A" * 40},
        {"candidate_sha": BASE_SHA},
        {"stage": "raw-command-failed"},
        {"error_category": "Exception: secret payload"},
    ],
)
def test_bootstrap_evidence_rejects_unbounded_identity(
    tmp_path: Path,
    overrides: dict[str, str],
) -> None:
    inputs = {
        "run_id": RUN_ID,
        "base_sha": BASE_SHA,
        "candidate_sha": CANDIDATE_SHA,
        "stage": "host-entrypoint",
        "error_category": "HostEntrypointFailed",
    }
    inputs.update(overrides)
    with pytest.raises(evidence.BootstrapEvidenceError):
        evidence.retain_failure(output=_output(tmp_path), **inputs)
    assert not _output(tmp_path).exists()


def test_existing_authoritative_success_wins_without_replacement(
    tmp_path: Path,
) -> None:
    output = _output(tmp_path)
    authoritative = (
        b'{"schema_version":"leadpoet.production_parity_full.v3","status":"passed"}\n'
    )
    output.write_bytes(authoritative)

    _, created = evidence.retain_failure(
        output=output,
        run_id=RUN_ID,
        base_sha=BASE_SHA,
        candidate_sha=CANDIDATE_SHA,
        stage="host-entrypoint",
        error_category="HostEntrypointFailed",
    )

    assert created is False
    assert output.read_bytes() == authoritative
    assert not list(tmp_path.glob(".full-evidence.*.tmp"))


def test_existing_symlink_is_never_followed(tmp_path: Path) -> None:
    authoritative = tmp_path / "authoritative.json"
    authoritative.write_text('{"status":"passed"}\n', encoding="utf-8")
    output = _output(tmp_path)
    output.symlink_to(authoritative)

    _, created = evidence.retain_failure(
        output=output,
        run_id=RUN_ID,
        base_sha=BASE_SHA,
        candidate_sha=CANDIDATE_SHA,
        stage="candidate-clone",
        error_category="CommandFailed",
    )

    assert created is False
    assert output.is_symlink()
    assert authoritative.read_text(encoding="utf-8") == '{"status":"passed"}\n'


def test_parent_symlink_is_rejected(tmp_path: Path) -> None:
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(OSError):
        evidence.retain_failure(
            output=(linked_parent / "full-evidence.json").absolute(),
            run_id=RUN_ID,
            base_sha=BASE_SHA,
            candidate_sha=CANDIDATE_SHA,
            stage="bootstrap-workspace",
            error_category="CommandFailed",
        )
    assert not (real_parent / "full-evidence.json").exists()


def test_authoritative_race_wins_atomically(monkeypatch, tmp_path: Path) -> None:
    output = _output(tmp_path)
    authoritative = b'{"status":"passed","authority":"host-main"}\n'
    real_link = evidence.os.link

    def racing_link(source, destination, **kwargs):
        output.write_bytes(authoritative)
        return real_link(source, destination, **kwargs)

    monkeypatch.setattr(evidence.os, "link", racing_link)
    _, created = evidence.retain_failure(
        output=output,
        run_id=RUN_ID,
        base_sha=BASE_SHA,
        candidate_sha=CANDIDATE_SHA,
        stage="host-entrypoint",
        error_category="HostEntrypointFailed",
    )

    assert created is False
    assert output.read_bytes() == authoritative
    assert not list(tmp_path.glob(".full-evidence.*.tmp"))


def test_short_writes_are_completed_before_atomic_publication(
    monkeypatch,
    tmp_path: Path,
) -> None:
    output = _output(tmp_path)
    real_write = evidence.os.write
    writes = 0

    def short_write(descriptor: int, value: memoryview) -> int:
        nonlocal writes
        writes += 1
        return real_write(descriptor, value[:7])

    monkeypatch.setattr(evidence.os, "write", short_write)
    payload, created = evidence.retain_failure(
        output=output,
        run_id=RUN_ID,
        base_sha=BASE_SHA,
        candidate_sha=CANDIDATE_SHA,
        stage="candidate-checkout",
        error_category="CommandFailed",
    )

    assert created is True
    assert writes > 1
    assert output.read_bytes() == payload


class _S3:
    def __init__(self, error: ClientError | None = None) -> None:
        self.error = error
        self.calls: list[dict[str, object]] = []

    def put_object(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return {"ETag": '"bounded"'}


def _client_error(code: str) -> ClientError:
    return ClientError(
        {"Error": {"Code": code, "Message": "secret-bearing-service-message"}},
        "PutObject",
    )


@pytest.mark.parametrize(
    ("status", "category"),
    sorted(poller.TERMINAL_ERROR_CATEGORIES.items()),
)
def test_terminal_poller_projects_only_allowlisted_bounded_evidence(
    tmp_path: Path,
    status: str,
    category: str,
) -> None:
    client = _S3()
    output = _output(tmp_path)

    poller.retain_terminal_failure(
        client,
        output=output,
        artifact_bucket=BUCKET,
        run_id=RUN_ID,
        base_sha=BASE_SHA,
        candidate_sha=CANDIDATE_SHA,
        status=status,
    )

    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["Bucket"] == BUCKET
    assert call["Key"] == f"production-parity/runs/{RUN_ID}/full-evidence.json"
    assert call["IfNoneMatch"] == "*"
    assert call["Body"] == output.read_bytes()
    assert json.loads(output.read_text(encoding="utf-8"))["error_type"] == category
    assert b"secret-bearing-service-message" not in output.read_bytes()


def test_terminal_upload_failure_keeps_local_github_artifact_safe(
    tmp_path: Path,
) -> None:
    client = _S3(_client_error("AccessDenied"))
    output = _output(tmp_path)

    with pytest.raises(poller.PollError, match="bounded terminal evidence upload"):
        poller.retain_terminal_failure(
            client,
            output=output,
            artifact_bucket=BUCKET,
            run_id=RUN_ID,
            base_sha=BASE_SHA,
            candidate_sha=CANDIDATE_SHA,
            status="Failed",
        )

    encoded = output.read_bytes()
    assert json.loads(encoded)["error_type"] == "SsmFailed"
    assert b"secret-bearing-service-message" not in encoded


def test_terminal_poller_main_retains_s3_and_github_evidence(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    s3 = _S3()

    class _Ssm:
        def get_command_invocation(self, **_kwargs):
            return {"Status": "Failed", "StandardErrorContent": "provider-secret"}

    monkeypatch.setattr(
        poller.boto3,
        "client",
        lambda service, **_kwargs: _Ssm() if service == "ssm" else s3,
    )
    github_output = tmp_path / "github-output"
    github_output.touch()
    output = _output(tmp_path)
    result = poller.main(
        [
            "--region",
            "us-east-1",
            "--command-id",
            "d151da94-87e8-4065-bab5-14c2ba6a019f",
            "--instance-id",
            "i-0e1583b50da3b6f87",
            "--max-wait-seconds",
            "30",
            "--github-output",
            str(github_output),
            "--evidence-output",
            str(output),
            "--artifact-bucket",
            BUCKET,
            "--run-id",
            RUN_ID,
            "--base-sha",
            BASE_SHA,
            "--candidate-sha",
            CANDIDATE_SHA,
        ]
    )

    assert result == 1
    assert len(s3.calls) == 1
    assert json.loads(output.read_text(encoding="utf-8"))["error_type"] == "SsmFailed"
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.strip() == "ERROR: SSM command reached terminal status Failed"
    assert "provider-secret" not in captured.err


@pytest.mark.parametrize("code", ["PreconditionFailed", "ConditionalRequestConflict"])
def test_terminal_upload_never_replaces_existing_s3_evidence(
    tmp_path: Path,
    code: str,
) -> None:
    poller.retain_terminal_failure(
        _S3(_client_error(code)),
        output=_output(tmp_path),
        artifact_bucket=BUCKET,
        run_id=RUN_ID,
        base_sha=BASE_SHA,
        candidate_sha=CANDIDATE_SHA,
        status="Failed",
    )


def _rendered_ssm_command(tmp_path: Path) -> str:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = Path(__file__).parents[1] / ".github/workflows/physical-v2-staging.yml"
    workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["validate"]["steps"]
    execute = next(
        step for step in steps if step.get("name") == "Start candidate production paths"
    )["run"]
    controller = execute.split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    replacements = {
        "${{ steps.inputs.outputs.run_id }}": RUN_ID,
        "${{ steps.inputs.outputs.base }}": BASE_SHA,
        "${{ steps.inputs.outputs.prefix }}": f"production-parity/runs/{RUN_ID}",
        "${{ steps.stack.outputs.supabase_origin }}": (
            "https://example.cloudfront.net"
        ),
        "${{ steps.stack.outputs.artifact_bucket }}": BUCKET,
    }
    for source, replacement in replacements.items():
        controller = controller.replace(source, replacement)
    assert "${{" not in controller
    output = tmp_path / "ssm-parameters.json"
    environment = {
        **os.environ,
        "AWS_REGION": "us-east-1",
        "CANDIDATE_SHA": CANDIDATE_SHA,
        "PRODUCTION_GATEWAY_SECRET_ID": "gateway-secret-id",
        "READONLY_DSN_SECRET_ID": "readonly-secret-id",
        "MINER_INTAKE_SECRET_ID": "miner-secret-id",
        "POSTGRES_IMAGE": "postgres@sha256:" + "c" * 64,
        "POSTGREST_IMAGE": "postgrest@sha256:" + "d" * 64,
        "FULL_TIMEOUT_SECONDS": "72000",
        "SSM_TIMEOUT_SECONDS": "77400",
    }
    rendered = subprocess.run(
        [sys.executable, "-", str(output)],
        input=controller,
        text=True,
        capture_output=True,
        cwd=path.parents[2],
        env=environment,
        check=False,
    )
    assert rendered.returncode == 0, rendered.stderr
    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["executionTimeout"] == ["77400"]
    assert len(document["commands"]) == 1
    return document["commands"][0]


def test_full_workflow_traps_and_uploads_every_bootstrap_stage() -> None:
    path = Path(__file__).parents[1] / ".github/workflows/physical-v2-staging.yml"
    workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["validate"]["steps"]
    by_name = {step.get("name"): step for step in steps}
    execute = by_name["Start candidate production paths"]["run"]

    assert "scripts/production_parity_bootstrap_evidence.py" in execute
    assert "trap finalize_bootstrap EXIT" in execute
    assert "aws s3api put-object" in execute
    assert "--if-none-match '*'" in execute
    assert 'evidence_parent="$(dirname -- "$evidence")"' in execute
    assert 'install -d -m 0700 -- "$evidence_parent"' in execute
    assert (
        execute.index('install -d -m 0700 -- "$evidence_parent"')
        < execute.index('/usr/bin/python3.11 -c {q(bootstrap_writer)}')
    )
    ordered = [
        "failure_stage=bootstrap-environment",
        "failure_stage=bootstrap-workspace",
        "failure_stage=candidate-bundle-download",
        "failure_stage=candidate-clone",
        "failure_stage=canonical-origin-fetch",
        "failure_stage=candidate-checkout",
        "failure_stage=host-python-import",
        "failure_stage=host-entrypoint",
    ]
    offsets = [execute.index(marker) for marker in ordered]
    assert offsets == sorted(offsets)
    assert "evidence=/run/leadpoet-production-parity/full-evidence.json" in execute
    assert "scripts/run_production_parity_full_host.py --help" in execute
    provisioner = (
        Path(__file__).parents[1] / "scripts/provision_production_parity_staging.py"
    ).read_text(encoding="utf-8")
    assert "install -d -m 0700 /run/leadpoet-production-parity" in provisioner
    assert (
        execute.index("failure_stage=host-python-import")
        < execute.index("scripts/run_production_parity_full_host.py --help")
        < execute.index('evidence="$authoritative_evidence"')
    )
    for number in range(1, 6):
        script = by_name[f"Poll candidate window {number}"]["run"]
        for argument in (
            "--evidence-output",
            "--artifact-bucket",
            "--run-id",
            "--base-sha",
            "--candidate-sha",
        ):
            assert argument in script
        assert "StandardOutputContent" not in script
        assert "StandardErrorContent" not in script
    upload = by_name["Upload redacted evidence"]
    assert "always()" in upload["if"]
    assert upload["with"]["path"].endswith("/full-evidence.json")


def test_rendered_ssm_bootstrap_is_valid_and_bounded(tmp_path: Path) -> None:
    command = _rendered_ssm_command(tmp_path)
    assert len(command.encode("utf-8")) < 24_000
    parsed = subprocess.run(
        ["bash", "-n"],
        input=command,
        text=True,
        capture_output=True,
        check=False,
    )
    assert parsed.returncode == 0, parsed.stderr


@pytest.mark.parametrize(
    ("stage", "category"),
    [
        ("bootstrap-environment", "CommandFailed"),
        ("bootstrap-workspace", "CommandFailed"),
        ("candidate-bundle-download", "CommandFailed"),
        ("candidate-clone", "CommandFailed"),
        ("canonical-origin-fetch", "CommandFailed"),
        ("candidate-checkout", "CommandFailed"),
        ("host-python-import", "HostImportFailed"),
        ("host-entrypoint", "HostEntrypointFailed"),
    ],
)
def test_rendered_ssm_uploads_exact_stage_when_early_parent_is_absent(
    tmp_path: Path,
    stage: str,
    category: str,
) -> None:
    command = _rendered_ssm_command(tmp_path / "render")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    capture = tmp_path / "uploaded-evidence.json"
    aws_stub = fake_bin / "aws"
    aws_stub.write_text(
        """#!/usr/bin/env python3
import os
from pathlib import Path
import shutil
import sys

arguments = sys.argv[1:]
if arguments[:2] == ["s3", "cp"]:
    destination = Path(arguments[3])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(b"candidate-bundle")
elif arguments[:2] == ["s3api", "put-object"]:
    body = Path(arguments[arguments.index("--body") + 1])
    shutil.copyfile(body, os.environ["CAPTURE_EVIDENCE"])
else:
    raise SystemExit(2)
""",
        encoding="utf-8",
    )
    git_stub = fake_bin / "git"
    git_stub.write_text(
        """#!/bin/sh
set -eu
case "$1 $2" in
  "clone "*) mkdir -p "$3" ;;
  "remote get-url") printf '%s\\n' 'https://github.com/leadpoet/leadpoet.git' ;;
  "rev-parse "*) printf '%s\\n' "$CANDIDATE_SHA" ;;
esac
""",
        encoding="utf-8",
    )
    sudo_stub = fake_bin / "sudo"
    sudo_stub.write_text(
        """#!/bin/sh
set -eu
operation="$1"
shift
case "$operation" in
  mkdir) exec mkdir "$@" ;;
  chown) exit 0 ;;
  *) exit 2 ;;
esac
""",
        encoding="utf-8",
    )
    host_python = fake_bin / "host-python"
    host_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    for executable in (aws_stub, git_stub, sudo_stub, host_python):
        executable.chmod(0o700)

    early_root = tmp_path / "missing-early-parent"
    work_root = tmp_path / "work"
    command = command.replace("/usr/bin/python3.11", sys.executable)
    command = command.replace(
        "/run/leadpoet-production-parity", str(early_root)
    )
    command = command.replace(
        f"/opt/leadpoet-production-parity/{RUN_ID}", str(work_root)
    )
    command = command.replace(
        "/home/ec2-user/venv311/bin/python3", str(host_python)
    )
    if stage == "bootstrap-environment":
        marker = "trap 'exit 143' TERM"
    elif stage in {"host-python-import", "host-entrypoint"}:
        marker = f"failure_category={category}"
    else:
        marker = f"failure_stage={stage}"
    assert command.count(marker) == 1
    command = command.replace(marker, marker + "\nfalse", 1)
    assert not early_root.exists()
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "CAPTURE_EVIDENCE": str(capture),
        "CANDIDATE_SHA": CANDIDATE_SHA,
        "PROVIDER_SECRET": "must-never-enter-evidence",
    }
    result = subprocess.run(
        ["bash", "-c", command],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    retained = json.loads(capture.read_text(encoding="utf-8"))
    assert retained["failure_stage"] == stage
    assert retained["error_type"] == category
    assert "must-never-enter-evidence" not in capture.read_text(encoding="utf-8")
    assert result.stdout == ""
    assert result.stderr == ""


def test_bootstrap_evidence_sources_never_capture_raw_process_material() -> None:
    root = Path(__file__).parents[1]
    combined = "\n".join(
        (root / path).read_text(encoding="utf-8")
        for path in (
            "scripts/production_parity_bootstrap_evidence.py",
            "scripts/poll_production_parity_ssm.py",
        )
    )
    for forbidden in (
        "StandardOutputContent",
        "StandardErrorContent",
        "runner.log",
        "os.environ",
        "printenv",
    ):
        assert forbidden not in combined
