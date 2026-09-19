"""Legacy Fulfillment miner and validator entrypoints stay retired."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_legacy_fulfillment_periphery_is_absent() -> None:
    for relative_path in (
        "Leadpoet/utils/hashing.py",
        "miner_models",
        "neurons/miner.py",
        "validator_models/checks_zerobounce.py",
        "validator_models/fulfillment_attribute_verification.py",
        "validator_models/fulfillment_company_verification.py",
        "validator_models/fulfillment_person_verification.py",
    ):
        assert not (ROOT / relative_path).exists(), relative_path


def test_example_configuration_has_no_fulfillment_controls() -> None:
    example = (ROOT / "env.example").read_text()
    assert "FULFILLMENT_" not in example
