from __future__ import annotations

import hashlib
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
CANDIDATE_BUNDLE_HEADER = (
    b"# v2 git bundle\n" + CANDIDATE_SHA.encode("ascii") + b" HEAD\n\n"
)
CANDIDATE_BUNDLE_BYTES = CANDIDATE_BUNDLE_HEADER + b"PACKfixture"
EXTRA_HEAD_BUNDLE_BYTES = (
    b"# v2 git bundle\n"
    + CANDIDATE_SHA.encode("ascii")
    + b" HEAD\n"
    + BASE_SHA.encode("ascii")
    + b" refs/heads/main\n\nPACKfixture"
)
CANDIDATE_BUNDLE_SHA256 = hashlib.sha256(CANDIDATE_BUNDLE_BYTES).hexdigest()
CANDIDATE_BUNDLE_SIZE_BYTES = str(len(CANDIDATE_BUNDLE_BYTES))


def _output(tmp_path: Path) -> Path:
    return tmp_path / "full-evidence.json"


@pytest.mark.parametrize(
    ("stage", "category"),
    [
        ("bootstrap-environment", "CommandFailed"),
        ("bootstrap-workspace", "CommandFailed"),
        ("candidate-bundle-download", "CommandFailed"),
        ("candidate-bundle-metadata", "CommandFailed"),
        ("candidate-bundle-file-integrity", "CommandFailed"),
        ("candidate-bundle-head", "CommandFailed"),
        ("candidate-git-runtime", "CommandFailed"),
        ("candidate-repository-directory", "CommandFailed"),
        ("candidate-bundle-verify", "CommandFailed"),
        ("candidate-repository-init", "CommandFailed"),
        ("candidate-repository-structure", "CommandFailed"),
        ("candidate-bundle-fetch", "CommandFailed"),
        ("candidate-checkout", "CommandFailed"),
        ("candidate-remote-rebind", "CommandFailed"),
        ("canonical-origin-fetch", "CommandFailed"),
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


def test_bootstrap_ssm_failure_codes_are_unique_fixed_and_bounded() -> None:
    values = evidence.BOOTSTRAP_SSM_FAILURE_CODES
    assert values == (
        (40, "bootstrap-environment", "CommandFailed"),
        (41, "bootstrap-workspace", "CommandFailed"),
        (42, "candidate-bundle-download", "CommandFailed"),
        (43, "candidate-bundle-metadata", "CommandFailed"),
        (44, "candidate-bundle-file-integrity", "CommandFailed"),
        (45, "candidate-bundle-head", "CommandFailed"),
        (46, "candidate-bundle-verify", "CommandFailed"),
        (47, "candidate-repository-init", "CommandFailed"),
        (48, "candidate-bundle-fetch", "CommandFailed"),
        (49, "candidate-checkout", "CommandFailed"),
        (50, "candidate-remote-rebind", "CommandFailed"),
        (51, "canonical-origin-fetch", "CommandFailed"),
        (52, "host-python-import", "HostImportFailed"),
        (53, "host-entrypoint", "HostEntrypointFailed"),
        (54, "evidence-upload", "EvidenceUploadFailed"),
        (55, "candidate-git-runtime", "CommandFailed"),
        (56, "candidate-repository-directory", "CommandFailed"),
        (57, "candidate-repository-structure", "CommandFailed"),
    )
    codes = [code for code, _stage, _category in values]
    assert len(codes) == len(set(codes))
    assert all(type(code) is int and 1 <= code <= 255 for code in codes)


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
        stage="candidate-bundle-file-integrity",
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


@pytest.mark.parametrize(
    ("response_code", "stage", "category"),
    evidence.BOOTSTRAP_SSM_FAILURE_CODES,
)
def test_terminal_poller_projects_authoritative_bootstrap_response_code(
    tmp_path: Path,
    response_code: int,
    stage: str,
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
        status="Failed",
        response_code=response_code,
    )

    retained = json.loads(output.read_text(encoding="utf-8"))
    assert retained["failure_stage"] == stage
    assert retained["error_type"] == category


@pytest.mark.parametrize(
    "response_code",
    [-1, 0, 1, 39, 58, 255, None, "40", True, 40.0, {}, []],
)
def test_terminal_poller_falls_back_for_not_started_malformed_or_unknown_code(
    tmp_path: Path,
    response_code: object,
) -> None:
    output = _output(tmp_path)
    poller.retain_terminal_failure(
        _S3(),
        output=output,
        artifact_bucket=BUCKET,
        run_id=RUN_ID,
        base_sha=BASE_SHA,
        candidate_sha=CANDIDATE_SHA,
        status="Failed",
        response_code=response_code,
    )

    retained = json.loads(output.read_text(encoding="utf-8"))
    assert retained["failure_stage"] == "ssm-command"
    assert retained["error_type"] == "SsmFailed"


@pytest.mark.parametrize(
    ("raw_response_code", "retained_response_code"),
    [(40, 40), (-1, -1), (999, 999), ("40", None), (True, None), (40.0, None)],
)
def test_poll_retains_only_strict_integer_response_code(
    raw_response_code: object,
    retained_response_code: int | None,
) -> None:
    class _Ssm:
        def get_command_invocation(self, **_kwargs):
            return {"Status": "Failed", "ResponseCode": raw_response_code}

    with pytest.raises(poller.TerminalPollError) as raised:
        poller.poll(
            _Ssm(),
            command_id="d151da94-87e8-4065-bab5-14c2ba6a019f",
            instance_id="i-0e1583b50da3b6f87",
            max_wait_seconds=30,
        )
    assert raised.value.response_code == retained_response_code


@pytest.mark.parametrize(
    ("status", "category"),
    sorted(
        (status, category)
        for status, category in poller.TERMINAL_ERROR_CATEGORIES.items()
        if status != "Failed"
    ),
)
def test_mapped_code_never_overrides_other_terminal_statuses(
    tmp_path: Path,
    status: str,
    category: str,
) -> None:
    output = _output(tmp_path)
    poller.retain_terminal_failure(
        _S3(),
        output=output,
        artifact_bucket=BUCKET,
        run_id=RUN_ID,
        base_sha=BASE_SHA,
        candidate_sha=CANDIDATE_SHA,
        status=status,
        response_code=40,
    )
    retained = json.loads(output.read_text(encoding="utf-8"))
    assert retained["failure_stage"] == "ssm-command"
    assert retained["error_type"] == category


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
            return {
                "Status": "Failed",
                "ResponseCode": 42,
                "StandardOutputContent": "provider-secret-output",
                "StandardErrorContent": "provider-secret",
            }

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
    retained = json.loads(output.read_text(encoding="utf-8"))
    assert retained["failure_stage"] == "candidate-bundle-download"
    assert retained["error_type"] == "CommandFailed"
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.strip() == "ERROR: SSM command reached terminal status Failed"
    assert "provider-secret" not in captured.err
    assert "provider-secret-output" not in captured.err


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
        response_code=42,
    )


def _rendered_ssm_command(
    tmp_path: Path,
    *,
    candidate_bundle_bytes: bytes = CANDIDATE_BUNDLE_BYTES,
) -> str:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = Path(__file__).parents[1] / ".github/workflows/physical-v2-staging.yml"
    workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["validate"]["steps"]
    execute = next(
        step for step in steps if step.get("name") == "Start candidate production paths"
    )["run"]
    controller = execute.split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    candidate_bundle_sha256 = hashlib.sha256(
        candidate_bundle_bytes
    ).hexdigest()
    candidate_bundle_size_bytes = str(len(candidate_bundle_bytes))
    candidate_bundle_binding_bytes = (
        json.dumps(
            {
                "candidate-sha": CANDIDATE_SHA,
                "bundle-sha256": candidate_bundle_sha256,
                "bundle-size-bytes": candidate_bundle_size_bytes,
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    replacements = {
        "${{ steps.inputs.outputs.run_id }}": RUN_ID,
        "${{ steps.inputs.outputs.base }}": BASE_SHA,
        "${{ steps.inputs.outputs.prefix }}": f"production-parity/runs/{RUN_ID}",
        "${{ steps.stack.outputs.supabase_origin }}": (
            "https://example.cloudfront.net"
        ),
        "${{ steps.stack.outputs.artifact_bucket }}": BUCKET,
        "${{ steps.candidate_bundle.outputs.sha256 }}": (
            candidate_bundle_sha256
        ),
        "${{ steps.candidate_bundle.outputs.size_bytes }}": (
            candidate_bundle_size_bytes
        ),
        "${{ steps.candidate_bundle.outputs.binding_sha256 }}": (
            hashlib.sha256(candidate_bundle_binding_bytes).hexdigest()
        ),
        "${{ steps.candidate_bundle.outputs.binding_size_bytes }}": (
            str(len(candidate_bundle_binding_bytes))
        ),
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
    assert "BOOTSTRAP_SSM_FAILURE_CODES" in execute
    assert "failure_response_code()" in execute
    assert execute.index('rm -f "$candidate_bundle"') < execute.index(
        'exit "$final_status"'
    )
    assert "aws s3api put-object" in execute
    assert "--if-none-match '*'" in execute
    assert 'evidence_parent="$(dirname -- "$evidence")" || return 1' in execute
    assert 'mkdir -m 0700 -- "$evidence_parent" >/dev/null 2>&1' in execute
    assert 'test -d "$evidence_parent" || return 1' in execute
    assert 'test ! -L "$evidence_parent" || return 1' in execute
    assert 'install -d -m 0700 -- "$evidence_parent"' not in execute
    assert (
        execute.index('mkdir -m 0700 -- "$evidence_parent"')
        < execute.index('/usr/bin/python3.11 -c {q(bootstrap_writer)}')
    )
    ordered = [
        "failure_stage=bootstrap-environment",
        "failure_stage=bootstrap-workspace",
        "failure_stage=candidate-bundle-download",
        "failure_stage=candidate-bundle-metadata",
        "failure_stage=candidate-bundle-file-integrity",
        "failure_stage=candidate-bundle-head",
        "failure_stage=candidate-repository-directory",
        "failure_stage=candidate-git-runtime",
        "failure_stage=candidate-repository-init",
        "failure_stage=candidate-repository-structure",
        "failure_stage=candidate-bundle-verify",
        "failure_stage=candidate-bundle-fetch",
        "failure_stage=candidate-checkout",
        "failure_stage=candidate-remote-rebind",
        "failure_stage=canonical-origin-fetch",
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


def test_full_workflow_candidate_bundle_is_exact_and_metadata_bound() -> None:
    path = Path(__file__).parents[1] / ".github/workflows/physical-v2-staging.yml"
    workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["validate"]["steps"]
    transfer = next(
        step
        for step in steps
        if step.get("name")
        == "Transfer exact candidate into the retained transient bucket"
    )
    script = transfer["run"]

    assert transfer["id"] == "candidate_bundle"
    assert 'git bundle create "$PARITY_TEMP/candidate.bundle" HEAD' in script
    assert "git bundle list-heads" in script
    assert '"$CANDIDATE_SHA HEAD"' in script
    assert "git bundle verify" in script
    assert "bundle_size_bytes=" in script
    assert "bundle_sha256=" in script
    assert 'bundle_binding="$PARITY_TEMP/candidate-bundle-binding.json"' in script
    assert '"bundle-sha256": sys.argv[3]' in script
    assert '"bundle-size-bytes": sys.argv[4]' in script
    assert '"candidate-sha": sys.argv[2]' in script
    assert "binding_size_bytes=" in script
    assert "binding_sha256=" in script
    assert "candidate-bundle-binding.json" in script
    assert script.count("aws s3 cp") == 2
    assert "binding_sha256=%s\\nbinding_size_bytes=%s\\n" in script
    assert "--all" not in script

    execute = next(
        step
        for step in steps
        if step.get("name") == "Start candidate production paths"
    )["run"]
    assert "candidate_bundle_binding_sha256=" in execute
    assert "candidate_bundle_binding_size_bytes=" in execute
    metadata_stage = execute.split(
        "failure_stage=candidate-bundle-metadata", 1
    )[1].split("failure_stage=candidate-bundle-file-integrity", 1)[0]
    download_stage = execute.split(
        "failure_stage=candidate-bundle-download", 1
    )[1].split("failure_stage=candidate-bundle-metadata", 1)[0]
    assert download_stage.count("aws s3api get-object") == 1
    assert "candidate-bundle-binding.json" not in download_stage
    assert '"$candidate_bundle"' in download_stage
    assert "--query Metadata" not in download_stage
    assert "head-object" not in execute
    assert "aws s3api get-object" in metadata_stage
    assert "candidate-bundle-binding.json" in metadata_stage
    assert '"$candidate_bundle_binding" >/dev/null 2>&1' in metadata_stage
    assert "sha256sum" in metadata_stage
    assert "stat -c '%s'" in metadata_stage
    assert "/dev/fd/3" not in metadata_stage
    assert "/usr/bin/python3.11 -c" not in metadata_stage
    assert "candidate_bundle_binding=" in execute
    assert "--output text" not in metadata_stage
    assert "aws s3 cp" not in metadata_stage
    head_stage = execute.split(
        "failure_stage=candidate-bundle-head", 1
    )[1].split("failure_stage=candidate-bundle-verify", 1)[0]
    assert 'LC_ALL=C head -n 3 -- "$candidate_bundle"' in head_stage
    assert "'# v2 git bundle'" in head_stage
    assert "git bundle list-heads" not in head_stage
    directory_stage = execute.split(
        "failure_stage=candidate-repository-directory", 1
    )[1].split("if [ -x /usr/bin/git ]", 1)[0]
    assert 'mkdir -m 0700 -- "$candidate_repo" "$candidate_git_home"' in (
        directory_stage
    )
    assert 'test -d "$candidate_repo"' in directory_stage
    assert 'test ! -L "$candidate_repo"' in directory_stage
    assert 'test -d "$candidate_git_home"' in directory_stage
    assert 'test ! -L "$candidate_git_home"' in directory_stage
    git_runtime_stage = execute.split(
        "failure_stage=candidate-git-runtime", 1
    )[1].split("failure_stage=candidate-repository-init", 1)[0]
    assert 'test -x "$candidate_git_bin"' in git_runtime_stage
    assert "candidate_git --version" not in git_runtime_stage
    assert "sudo -n /usr/bin/dnf -q -y install git-core" in execute
    init_stage = execute.split(
        "failure_stage=candidate-repository-init", 1
    )[1].split("failure_stage=candidate-repository-structure", 1)[0]
    assert 'candidate_git -C "$candidate_repo" init' in init_stage
    structure_stage = execute.split(
        "failure_stage=candidate-repository-structure", 1
    )[1].split("failure_stage=candidate-bundle-verify", 1)[0]
    assert 'test -d "$candidate_repo/.git"' in structure_stage
    assert 'test ! -L "$candidate_repo/.git"' in structure_stage
    assert 'git init "$candidate_repo"' not in init_stage
    assert "candidate_git()" in execute
    assert "/usr/bin/env -i" in execute
    assert "PATH=/usr/bin:/bin" in execute
    assert "LC_ALL=C" in execute
    assert 'HOME="$candidate_git_home"' in execute
    assert "GIT_CONFIG_NOSYSTEM=1" in execute
    assert "GIT_TERMINAL_PROMPT=0" in execute
    assert '"$candidate_git_bin" -c init.templateDir=' in execute
    verify_stage = execute.split(
        "failure_stage=candidate-bundle-verify", 1
    )[1].split("failure_stage=candidate-bundle-fetch", 1)[0]
    assert 'candidate_git -C "$candidate_repo" fetch --no-tags' in verify_stage
    assert '"$candidate_bundle" HEAD' in verify_stage
    assert "bundle verify" not in verify_stage
    assert "git init --bare" not in execute


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
    for response_code, stage, _category in evidence.BOOTSTRAP_SSM_FAILURE_CODES:
        assert f"{stage}) printf '%s\\n' {response_code} ;;" in command


def _run_rendered_ssm(
    tmp_path: Path,
    *,
    stage: str | None = None,
    category: str | None = None,
    upload_fails: bool = False,
    early_parent_target: Path | None = None,
    bundle_bytes: bytes = CANDIDATE_BUNDLE_BYTES,
    bound_bundle_bytes: bytes | None = None,
    bundle_verify_fails: bool = False,
    bundle_as_symlink: bool = False,
    metadata_candidate_sha: str = CANDIDATE_SHA,
    metadata_bundle_sha256: str | None = None,
    metadata_bundle_size_bytes: str | None = None,
    metadata_raw_json: str | None = None,
) -> tuple[subprocess.CompletedProcess[str], Path]:
    controller_bundle_bytes = (
        CANDIDATE_BUNDLE_BYTES
        if bound_bundle_bytes is None
        else bound_bundle_bytes
    )
    command = _rendered_ssm_command(
        tmp_path / "render",
        candidate_bundle_bytes=controller_bundle_bytes,
    )
    if metadata_bundle_sha256 is None:
        metadata_bundle_sha256 = hashlib.sha256(
            controller_bundle_bytes
        ).hexdigest()
    if metadata_bundle_size_bytes is None:
        metadata_bundle_size_bytes = str(len(controller_bundle_bytes))
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    capture = tmp_path / "uploaded-evidence.json"
    aws_calls = tmp_path / "aws-calls.jsonl"
    git_calls = tmp_path / "git-calls.jsonl"
    aws_stub = fake_bin / "aws"
    aws_stub.write_text(
        """#!/usr/bin/env python3
import json
import os
from pathlib import Path
import shutil
import sys

arguments = sys.argv[1:]
with open(os.environ["AWS_CALLS"], "a", encoding="utf-8") as handle:
    handle.write(json.dumps(arguments, separators=(",", ":")) + "\\n")
if arguments[:2] == ["s3api", "get-object"]:
    destination = Path(arguments[-1])
    key = arguments[arguments.index("--key") + 1]
    if key.endswith("/candidate.bundle"):
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = bytes.fromhex(os.environ["BUNDLE_HEX"])
        if os.environ.get("BUNDLE_AS_SYMLINK") == "1":
            symlink_target = destination.with_name(destination.name + ".payload")
            symlink_target.write_bytes(payload)
            destination.symlink_to(symlink_target)
        else:
            destination.write_bytes(payload)
    elif key.endswith("/candidate-bundle-binding.json"):
        destination.write_text(os.environ["METADATA_RAW_JSON"], encoding="utf-8")
    else:
        raise SystemExit(2)
elif arguments[:2] == ["s3api", "put-object"]:
    if os.environ.get("FAIL_EVIDENCE_UPLOAD") == "1":
        raise SystemExit(73)
    body = Path(arguments[arguments.index("--body") + 1])
    shutil.copyfile(body, os.environ["CAPTURE_EVIDENCE"])
else:
    raise SystemExit(2)
""",
        encoding="utf-8",
    )
    git_stub = fake_bin / "git"
    git_stub.write_text(
        """#!/usr/bin/env python3
import json
import os
from pathlib import Path
import sys

arguments = sys.argv[1:]
with open(os.environ["GIT_CALLS"], "a", encoding="utf-8") as handle:
    handle.write(json.dumps(arguments, separators=(",", ":")) + "\\n")

if arguments[:2] == ["-c", "init.templateDir="]:
    arguments = arguments[2:]

if arguments[:1] == ["--version"]:
    print("git version test")
elif arguments[:1] == ["init"]:
    Path(arguments[1]).mkdir(parents=True, exist_ok=False)
elif arguments[:1] == ["-C"]:
    operation = arguments[2:]
    if operation[:1] == ["init"]:
        Path(arguments[1], ".git").mkdir(parents=False, exist_ok=False)
    elif operation[:1] == ["fetch"]:
        if (
            os.environ.get("FAIL_BUNDLE_VERIFY") == "1"
            and any(value.endswith("/candidate.bundle") for value in operation)
        ):
            raise SystemExit(74)
    elif operation[:2] == ["remote", "get-url"]:
        print("https://github.com/leadpoet/leadpoet.git")
    elif operation[:1] == ["rev-parse"]:
        print(os.environ["CANDIDATE_SHA"])
    elif operation[:1] in (["checkout"], ["remote"]):
        pass
    else:
        raise SystemExit(3)
else:
    raise SystemExit(2)
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
    stat_stub = fake_bin / "stat"
    stat_stub.write_text(
        """#!/usr/bin/env python3
from pathlib import Path
import sys

print(Path(sys.argv[-1]).stat().st_size)
""",
        encoding="utf-8",
    )
    sha256sum_stub = fake_bin / "sha256sum"
    sha256sum_stub.write_text(
        """#!/usr/bin/env python3
import hashlib
from pathlib import Path
import sys

value = Path(sys.argv[-1]).read_bytes()
print(hashlib.sha256(value).hexdigest(), sys.argv[-1])
""",
        encoding="utf-8",
    )
    host_python = fake_bin / "host-python"
    host_python.write_text(
        """#!/usr/bin/env python3
from pathlib import Path
import sys

if "--help" not in sys.argv:
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('{"status":"passed","authority":"host"}\\n')
""",
        encoding="utf-8",
    )
    for executable in (
        aws_stub,
        git_stub,
        sudo_stub,
        stat_stub,
        sha256sum_stub,
        host_python,
    ):
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
    command = command.replace("/usr/bin/git", str(git_stub))
    command = command.replace("/usr/bin/env -i", "/usr/bin/env")
    if stage is not None:
        assert category is not None
        if stage == "bootstrap-environment":
            marker = "trap 'exit 143' TERM"
        elif stage in {"host-python-import", "host-entrypoint"}:
            marker = f"failure_category={category}"
        else:
            marker = f"failure_stage={stage}"
        assert command.count(marker) == 1
        command = command.replace(marker, marker + "\nfalse", 1)
    if early_parent_target is None:
        assert not early_root.exists()
    else:
        early_root.symlink_to(early_parent_target, target_is_directory=True)
    if metadata_raw_json is None:
        metadata_raw_json = (
            json.dumps(
                {
                    "candidate-sha": metadata_candidate_sha,
                    "bundle-sha256": metadata_bundle_sha256,
                    "bundle-size-bytes": metadata_bundle_size_bytes,
                },
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        )
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "AWS_CALLS": str(aws_calls),
        "CAPTURE_EVIDENCE": str(capture),
        "CANDIDATE_SHA": CANDIDATE_SHA,
        "BUNDLE_HEX": bundle_bytes.hex(),
        "BUNDLE_AS_SYMLINK": "1" if bundle_as_symlink else "0",
        "FAIL_BUNDLE_VERIFY": "1" if bundle_verify_fails else "0",
        "FAIL_EVIDENCE_UPLOAD": "1" if upload_fails else "0",
        "GIT_CALLS": str(git_calls),
        "METADATA_RAW_JSON": metadata_raw_json,
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
    return result, capture


@pytest.mark.parametrize(
    ("response_code", "stage", "category"),
    tuple(
        item
        for item in evidence.BOOTSTRAP_SSM_FAILURE_CODES
        if item[1] != "evidence-upload"
    ),
)
def test_rendered_ssm_maps_and_uploads_each_exact_failure_stage(
    tmp_path: Path,
    response_code: int,
    stage: str,
    category: str,
) -> None:
    result, capture = _run_rendered_ssm(
        tmp_path,
        stage=stage,
        category=category,
    )

    assert result.returncode == response_code
    retained = json.loads(capture.read_text(encoding="utf-8"))
    assert retained["failure_stage"] == stage
    assert retained["error_type"] == category
    assert "must-never-enter-evidence" not in capture.read_text(encoding="utf-8")
    aws_calls = [
        json.loads(line)
        for line in (tmp_path / "aws-calls.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    uploads = [call for call in aws_calls if call[:2] == ["s3api", "put-object"]]
    assert len(uploads) == 1
    assert uploads[0][uploads[0].index("--if-none-match") + 1] == "*"
    assert result.stdout == ""
    assert result.stderr == ""


def test_rendered_ssm_preserves_success_and_uploads_host_evidence(
    tmp_path: Path,
) -> None:
    result, capture = _run_rendered_ssm(tmp_path)

    assert result.returncode == 0
    assert json.loads(capture.read_text(encoding="utf-8")) == {
        "status": "passed",
        "authority": "host",
    }
    assert not (tmp_path / "work" / "candidate.bundle").exists()
    assert not list((tmp_path / "work").glob("candidate-bundle-binding.*"))
    assert result.stdout == ""
    assert result.stderr == ""


def test_rendered_ssm_downloads_bundle_and_checks_exact_binding_bytes(
    tmp_path: Path,
) -> None:
    result, _capture = _run_rendered_ssm(tmp_path)

    assert result.returncode == 0
    calls = [
        json.loads(line)
        for line in (tmp_path / "aws-calls.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    reads = [call for call in calls if call[:2] == ["s3api", "get-object"]]
    assert len(reads) == 2
    assert reads[0][reads[0].index("--key") + 1].endswith(
        "/candidate.bundle"
    )
    assert reads[0][-1] == str(tmp_path / "work" / "candidate.bundle")
    assert reads[1][reads[1].index("--key") + 1].endswith(
        "/candidate-bundle-binding.json"
    )
    assert reads[1][-1] == str(
        tmp_path / "work" / "candidate-bundle-binding.json"
    )
    assert "--region" in reads[1]
    assert all("--query" not in call for call in reads)
    assert not any(call[:2] == ["s3api", "head-object"] for call in calls)
    assert not any(call[:2] == ["s3", "cp"] for call in calls)


def test_rendered_ssm_uses_explicit_exact_candidate_git_sequence(
    tmp_path: Path,
) -> None:
    result, _capture = _run_rendered_ssm(tmp_path)

    assert result.returncode == 0
    work = tmp_path / "work"
    bundle = str(work / "candidate.bundle")
    repo = str(work / "repo")
    calls = [
        json.loads(line)
        for line in (tmp_path / "git-calls.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert all(call[:2] == ["-c", "init.templateDir="] for call in calls)
    normalized_calls = [call[2:] for call in calls]
    assert normalized_calls == [
        ["-C", repo, "init"],
        ["-C", repo, "fetch", "--no-tags", bundle, "HEAD"],
        ["-C", repo, "rev-parse", "FETCH_HEAD"],
        ["-C", repo, "checkout", "--detach", CANDIDATE_SHA],
        ["-C", repo, "rev-parse", "HEAD"],
        [
            "-C",
            repo,
            "remote",
            "add",
            "origin",
            "https://github.com/leadpoet/leadpoet.git",
        ],
        ["-C", repo, "remote", "get-url", "origin"],
        [
            "-C",
            repo,
            "fetch",
            "--no-tags",
            "origin",
            "refs/heads/main:refs/remotes/origin/main",
        ],
        ["-C", repo, "rev-parse", "origin/main"],
    ]
    assert all("clone" not in call for call in normalized_calls)


@pytest.mark.parametrize(
    ("overrides", "response_code", "failure_stage"),
    [
        ({"metadata_candidate_sha": BASE_SHA}, 43, "candidate-bundle-metadata"),
        (
            {"metadata_bundle_sha256": "0" * 64},
            43,
            "candidate-bundle-metadata",
        ),
        (
            {"metadata_bundle_size_bytes": "999"},
            43,
            "candidate-bundle-metadata",
        ),
        ({"metadata_raw_json": "{"}, 43, "candidate-bundle-metadata"),
        (
            {
                "metadata_raw_json": json.dumps(
                    {
                        "candidate-sha": CANDIDATE_SHA,
                        "bundle-sha256": CANDIDATE_BUNDLE_SHA256,
                        "bundle-size-bytes": CANDIDATE_BUNDLE_SIZE_BYTES,
                        "extra": "rejected",
                    },
                    separators=(",", ":"),
                )
            },
            43,
            "candidate-bundle-metadata",
        ),
        (
            {
                "metadata_raw_json": (
                    '{"candidate-sha":"'
                    + CANDIDATE_SHA
                    + '","candidate-sha":"'
                    + CANDIDATE_SHA
                    + '","bundle-sha256":"'
                    + CANDIDATE_BUNDLE_SHA256
                    + '","bundle-size-bytes":"'
                    + CANDIDATE_BUNDLE_SIZE_BYTES
                    + '"}'
                )
            },
            43,
            "candidate-bundle-metadata",
        ),
        (
            {"metadata_raw_json": "{" + " " * 4096 + "}"},
            43,
            "candidate-bundle-metadata",
        ),
        ({"metadata_raw_json": "[]"}, 43, "candidate-bundle-metadata"),
        (
            {
                "bundle_bytes": (
                    CANDIDATE_BUNDLE_BYTES[:-1]
                    + bytes([CANDIDATE_BUNDLE_BYTES[-1] ^ 1])
                )
            },
            44,
            "candidate-bundle-file-integrity",
        ),
        (
            {"bundle_bytes": b"short"},
            44,
            "candidate-bundle-file-integrity",
        ),
        (
            {"bundle_as_symlink": True},
            44,
            "candidate-bundle-file-integrity",
        ),
        (
            {
                "bundle_bytes": EXTRA_HEAD_BUNDLE_BYTES,
                "bound_bundle_bytes": EXTRA_HEAD_BUNDLE_BYTES,
            },
            45,
            "candidate-bundle-head",
        ),
        ({"bundle_verify_fails": True}, 46, "candidate-bundle-verify"),
    ],
    ids=(
        "metadata-candidate-mismatch",
        "metadata-sha256-mismatch",
        "metadata-size-mismatch",
        "metadata-malformed-json",
        "metadata-extra-key",
        "metadata-duplicate-key",
        "metadata-over-bound",
        "metadata-not-object",
        "downloaded-same-size-byte-corruption",
        "downloaded-size-mismatch",
        "downloaded-symlink",
        "unexpected-extra-head",
        "bundle-verification-failure",
    ),
)
def test_rendered_ssm_rejects_each_bundle_integrity_violation(
    tmp_path: Path,
    overrides: dict[str, object],
    response_code: int,
    failure_stage: str,
) -> None:
    result, capture = _run_rendered_ssm(tmp_path, **overrides)

    assert result.returncode == response_code
    retained = json.loads(capture.read_text(encoding="utf-8"))
    assert retained["failure_stage"] == failure_stage
    assert retained["error_type"] == "CommandFailed"
    assert "must-never-enter-evidence" not in capture.read_text(encoding="utf-8")
    assert not (tmp_path / "work" / "candidate.bundle").exists()
    if overrides.get("bundle_verify_fails"):
        assert (tmp_path / "work" / "repo").is_dir()
    else:
        assert not (tmp_path / "work" / "repo").exists()
    aws_calls = [
        json.loads(line)
        for line in (tmp_path / "aws-calls.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    uploads = [call for call in aws_calls if call[:2] == ["s3api", "put-object"]]
    assert len(uploads) == 1
    assert uploads[0][uploads[0].index("--if-none-match") + 1] == "*"
    assert result.stdout == ""
    assert result.stderr == ""


def test_rendered_ssm_maps_evidence_upload_failure_without_raw_output(
    tmp_path: Path,
) -> None:
    result, capture = _run_rendered_ssm(tmp_path, upload_fails=True)

    assert result.returncode == 54
    assert not capture.exists()
    assert not (tmp_path / "work" / "candidate.bundle").exists()
    assert result.stdout == ""
    assert result.stderr == ""


def test_rendered_ssm_preserves_primary_stage_when_its_upload_also_fails(
    tmp_path: Path,
) -> None:
    result, capture = _run_rendered_ssm(
        tmp_path,
        stage="candidate-bundle-download",
        category="CommandFailed",
        upload_fails=True,
    )

    assert result.returncode == 42
    assert not capture.exists()
    assert result.stdout == ""
    assert result.stderr == ""


def test_rendered_ssm_rejects_early_parent_symlink_without_mutation_or_upload(
    tmp_path: Path,
) -> None:
    target = tmp_path / "symlink-target"
    target.mkdir(mode=0o751)
    sentinel = target / "sentinel"
    sentinel.write_text("unchanged\n", encoding="utf-8")
    original_mode = target.stat().st_mode & 0o777

    result, capture = _run_rendered_ssm(
        tmp_path,
        stage="bootstrap-environment",
        category="CommandFailed",
        early_parent_target=target,
    )

    assert result.returncode == 40
    assert not capture.exists()
    assert target.stat().st_mode & 0o777 == original_mode
    assert sentinel.read_text(encoding="utf-8") == "unchanged\n"
    assert not (target / "full-evidence.json").exists()
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
