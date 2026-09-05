from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _function_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def test_retired_qualification_routes_are_not_mounted() -> None:
    gateway = (ROOT / "gateway" / "main.py").read_text(encoding="utf-8")

    assert "qualification_router" not in gateway
    assert "gateway.qualification.api.router" not in gateway
    assert not (ROOT / "qualification" / "main.py").exists()


def test_miner_has_no_retired_qualification_submission_or_payment_flow() -> None:
    miner_path = ROOT / "neurons" / "miner.py"
    miner = miner_path.read_text(encoding="utf-8")
    retired_functions = {
        "get_leadpoet_coldkey",
        "get_tao_price_sync",
        "calculate_tao_required",
        "transfer_tao",
        "create_model_tarball",
        "get_presigned_upload_url",
        "upload_to_s3_presigned",
        "submit_qualification_model",
        "run_qualification_submission_flow",
    }

    assert _function_names(miner_path).isdisjoint(retired_functions)
    assert "/qualification/model" not in miner
    assert "QUALIFICATION_SUBMISSION_COST_USD" not in miner


def test_validator_has_no_retired_qualification_worker_path() -> None:
    validator_path = ROOT / "neurons" / "validator.py"
    validator = validator_path.read_text(encoding="utf-8")
    deploy = (
        ROOT / "validator_models" / "containerizing" / "deploy_dynamic.sh"
    ).read_text(encoding="utf-8")
    restart = (ROOT / "validator_restart.sh").read_text(encoding="utf-8")
    required_start = restart.index("required_keys=(")
    restart_required = restart[
        required_start : restart.index(")", required_start)
    ]
    retired_functions = {
        "detect_qualification_proxies",
        "detect_qualification_worker_ids",
        "qualification_worker_capacity",
        "qualification_worker_id_from_work_file",
        "process_qualification_workflow",
        "_qualification_register",
        "_qualification_request_work",
        "_qualification_request_batch_work",
        "_qualification_execute_work",
        "_qualification_report_results",
        "_assign_qualification_to_dedicated_workers",
        "_collect_dedicated_qualification_results",
        "_process_dedicated_qualification_results",
        "run_dedicated_qualification_worker",
    }

    assert _function_names(validator_path).isdisjoint(retired_functions)
    assert '"qualification_worker"' not in validator
    assert "--mode qualification_worker" not in deploy
    assert "ENABLE_QUALIFICATION_WORKERS" not in deploy
    assert "ENABLE_QUALIFICATION_EVALUATION" not in deploy
    assert "QUALIFICATION_WEBSHARE_PROXY" not in deploy
    assert "ENABLE_QUALIFICATION_EVALUATION" not in restart_required
    assert "QUALIFICATION_WEBSHARE_PROXY" not in restart_required
    assert "QUALIFICATION_OPENROUTER_API_KEY" not in restart_required
    assert "QUALIFICATION_SCRAPINGDOG_API_KEY" not in restart_required
    assert (
        'export QUALIFICATION_OPENROUTER_API_KEY="${QUALIFICATION_OPENROUTER_API_KEY:-${OPENROUTER_API_KEY:-}}"'
        in restart
    )
    assert (
        'export QUALIFICATION_SCRAPINGDOG_API_KEY="${QUALIFICATION_SCRAPINGDOG_API_KEY:-${SCRAPINGDOG_API_KEY:-}}"'
        in restart
    )
    for active_key in (
        "ENABLE_FULFILLMENT",
        "FULFILLMENT_OPENROUTER_API_KEY",
        "RESEARCH_LAB_VALIDATOR_FETCH_ENABLED",
        "RESEARCH_LAB_INTERNAL_API_KEY",
        "RESEARCH_LAB_WEIGHT_MUTATION_ENABLED",
        "RESEARCH_LAB_SUBMIT_ON_CHAIN_ENABLED",
    ):
        assert active_key in restart_required
    assert "required_keys+=(VALIDATOR_V2_GATEWAY_URL)" in restart


def test_retired_qualification_runtime_is_removed() -> None:
    for path in (
        ROOT / "gateway" / "qualification" / "api",
        ROOT / "qualification" / "validator",
        ROOT / "miner_models" / "qualification_model",
        ROOT / "miner_models" / "qualification_research_arm_b",
    ):
        assert not any(path.glob("*.py"))

    for name in (
        "baseline.py",
        "baseline_arms.py",
        "champion.py",
        "emissions.py",
    ):
        assert not (ROOT / "qualification" / "scoring" / name).exists()
