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


def test_production_parity_has_no_closed_model_admission_job() -> None:
    workflow = PARITY_WORKFLOW.read_text(encoding="utf-8")
    for forbidden in (
        "signed-artifact-admission",
        "verify_signed_sourcing_artifact_admission.py",
        "branches/leadpoet-lab/current.json",
        "SOURCING_MODEL_READ_TOKEN",
        "repository: leadpoet/Sourcing_model",
    ):
        assert forbidden not in workflow
