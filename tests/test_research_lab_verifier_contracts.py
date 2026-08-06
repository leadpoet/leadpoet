from pathlib import Path

from leadpoet_verifier.golden_vectors import run_golden_vectors
from scripts.verify_research_lab_gateway_api_contract import main as verify_gateway_api


ROOT = Path(__file__).resolve().parents[1]


def test_open_verifier_golden_vectors_match_current_score_contract():
    assert run_golden_vectors(
        pcr0_allowlist_path=str(ROOT / "pcr0_allowlist.json")
    ) == []


def test_gateway_api_default_verifier_ignores_ambient_runtime_flags(monkeypatch):
    contaminated = {
        "BITTENSOR_NETWORK": "finney",
        "BITTENSOR_NETUID": "71",
        "SUBTENSOR_NETWORK": "finney",
        "NETUID": "71",
        "RESEARCH_LAB_GATEWAY_API_ENABLED": "true",
        "RESEARCH_LAB_PRODUCTION_WRITES_ENABLED": "true",
        "RESEARCH_LAB_EVALUATION_BUNDLES_ENABLED": "true",
        "RESEARCH_LAB_WEIGHT_MUTATION_ENABLED": "false",
        "RESEARCH_LAB_CROWNING_ENABLED": "false",
        "RESEARCH_LAB_AUTO_PROMOTION_ENABLED": "false",
        "RESEARCH_LAB_ALLOWED_ISLANDS": "healthcare",
    }
    for name, value in contaminated.items():
        monkeypatch.setenv(name, value)

    assert verify_gateway_api() == 0
