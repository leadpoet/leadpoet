"""Legacy Fulfillment miner and validator entrypoints stay retired."""

import asyncio
import inspect
from pathlib import Path
from unittest.mock import AsyncMock, patch

from validator_models import automated_checks


ROOT = Path(__file__).resolve().parents[1]


def test_legacy_fulfillment_periphery_is_absent() -> None:
    for relative_path in (
        "Leadpoet/utils/hashing.py",
        "miner_models/Main_fulfillment_model",
        "miner_models/fulfillment_sourcer.py",
        "validator_models/checks_zerobounce.py",
        "validator_models/fulfillment_attribute_verification.py",
        "validator_models/fulfillment_company_verification.py",
        "validator_models/fulfillment_person_verification.py",
    ):
        assert not (ROOT / relative_path).exists(), relative_path


def test_miner_and_cloud_client_have_no_fulfillment_entrypoints() -> None:
    miner_source = (ROOT / "neurons/miner.py").read_text()
    cloud_source = (ROOT / "Leadpoet/utils/cloud_db.py").read_text()

    assert "ENABLE_FULFILLMENT" not in miner_source
    assert '"fulfillment"' not in miner_source
    assert "gateway_poll_fulfillment_requests" not in cloud_source
    assert "gateway_submit_fulfillment_commit" not in cloud_source
    assert "gateway_reveal_fulfillment" not in cloud_source
    assert "gateway_get_fulfillment_reveals" not in cloud_source
    assert "gateway_submit_fulfillment_scores" not in cloud_source
    assert "gateway_get_banned_hotkeys_snapshot" not in cloud_source


def test_example_configuration_has_no_fulfillment_controls() -> None:
    example = (ROOT / "env.example").read_text()
    assert "FULFILLMENT_" not in example


def test_shared_qualification_checks_have_one_full_validation_path() -> None:
    assert tuple(inspect.signature(automated_checks.run_stage4_5_repscore).parameters) == (
        "lead",
        "email_result",
        "stage0_2_data",
    )

    lead = {"email": "ada@example.com", "business": "Example"}
    passed_check = AsyncMock(return_value=(True, None))
    empty_rep_check = AsyncMock(return_value=(0, {}))
    with patch.object(automated_checks, "check_linkedin_gse", passed_check), \
            patch.object(automated_checks, "check_stage5_unified", passed_check), \
            patch.object(automated_checks, "check_wayback_machine", empty_rep_check), \
            patch.object(automated_checks, "check_sec_edgar", empty_rep_check), \
            patch.object(automated_checks, "check_whois_dnsbl_reputation", empty_rep_check), \
            patch.object(automated_checks, "check_gdelt_mentions", empty_rep_check), \
            patch.object(automated_checks, "check_companies_house", empty_rep_check), \
            patch.object(automated_checks, "calculate_icp_adjustment", return_value=0):
        passed, evidence = asyncio.run(
            automated_checks.run_stage4_5_repscore(
                lead,
                {"status": "email_ok", "passed": True},
                {},
            )
        )

    assert passed is True
    assert evidence["stage_3_email"]["email_status"] == "valid"
    assert evidence["stage_4_linkedin"]["linkedin_verified"] is True
    assert evidence["passed"] is True
    assert passed_check.await_count == 2
