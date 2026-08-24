from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "deploy-checks.yml"
PARITY_WORKFLOW = (
    ROOT / ".github" / "workflows" / "production-parity-fast.yml"
)


def test_deploy_checks_uses_local_consumer_gates_not_private_source_checkout() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "SOURCING_MODEL_READ_TOKEN" not in workflow
    assert "repository: leadpoet/Sourcing_model" not in workflow
    assert "SOURCING_MODEL_SOURCE_SHA" not in workflow
    assert "Run exact Sourcing_model custody and consumer tests" not in workflow
    assert "exact_sourcing_model_source" not in workflow


def test_trusted_main_push_admits_signed_artifact_without_private_checkout() -> None:
    workflow = PARITY_WORKFLOW.read_text(encoding="utf-8")
    job = workflow.split("  signed-artifact-admission:", 1)[1].split(
        "\n  validate:", 1
    )[0]
    job_environment = job.split("\n    steps:", 1)[0]
    admission_step = job.split(
        "      - name: Admit current signed artifact before provider spend",
        1,
    )[1].split("\n      - name: Remove ephemeral ECR login", 1)[0]

    assert "github.event_name == 'push'" in job
    assert "github.ref == 'refs/heads/main'" in job
    assert "github.repository_id == '1075412927'" in job
    assert "id-token: write" in job
    assert "persist-credentials: false" in job
    assert "LEADPOET_PARITY_AWS_ROLE_ARN" in job
    assert "scripts/verify_signed_sourcing_artifact_admission.py" in job
    assert (
        "LEADPOET_DOCKER_OPERATION_LOCK_FILE: "
        "${{ runner.temp }}/leadpoet-docker-operation-v2.lock"
    ) in admission_step
    assert (
        "LEADPOET_DOCKER_OPERATION_ADMISSION_LOCK_FILE: "
        "${{ runner.temp }}/leadpoet-docker-operation-v2.admission.lock"
    ) in admission_step
    assert "runner.temp" not in job_environment
    assert "branches/leadpoet-lab/current.json" in job
    assert "ecr:BatchGetImage" in job
    assert "kms:Verify" in job
    for forbidden in (
        "SOURCING_MODEL_READ_TOKEN",
        "repository: leadpoet/Sourcing_model",
        "secretsmanager:",
        "ec2:",
        "s3:Put",
        "s3:Delete",
        "ecr:PutImage",
        "kms:Sign",
        "kms:Decrypt",
    ):
        assert forbidden not in job
