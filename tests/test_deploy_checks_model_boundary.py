from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "deploy-checks.yml"
PARITY_WORKFLOW = (
    ROOT / ".github" / "workflows" / "production-parity-fast.yml"
)


def test_deploy_checks_has_no_closed_model_checkout() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert "closed-model" not in workflow.lower()
    assert "private model" not in workflow.lower()


def test_production_parity_has_no_closed_model_admission_job() -> None:
    workflow = PARITY_WORKFLOW.read_text(encoding="utf-8")
    for forbidden in (
        "signed-artifact-admission",
        "verify_signed_sourcing_artifact_admission.py",
        "branches/leadpoet-lab/current.json",
        "closed-model",
    ):
        assert forbidden not in workflow
