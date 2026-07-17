from pathlib import Path
import inspect

from research_lab import validator_integration
from validator_tee.enclave import tee_service


ROOT = Path(__file__).resolve().parents[1]


def test_attestation_shadow_module_and_mode_flags_are_removed():
    assert not (ROOT / "validator_tee" / "host" / "weight_shadow.py").exists()
    validator_source = (ROOT / "neurons" / "validator.py").read_text(
        encoding="utf-8"
    )
    assert "execute_attested_weight_mode" not in validator_source
    assert "VALIDATOR_ATTESTED_WEIGHT_MODE" not in validator_source
    assert "VALIDATOR_REQUIRE_GATEWAY_WEIGHT_SUBMISSION" not in validator_source


def test_primary_weight_path_uses_only_v2_allocation_handoff():
    validator_source = (ROOT / "neurons" / "validator.py").read_text(
        encoding="utf-8"
    )
    start = validator_source.index(
        "async def _research_lab_pre_weight_submission_guard"
    )
    end = validator_source.index(
        "async def _authorize_and_set_weights_v2",
        start,
    )
    handoff = validator_source[start:end]
    assert "fetch_research_lab_attested_allocation_bundle" in handoff
    assert "validate_allocation_handoff_v2" in handoff
    assert "VALIDATOR_V2_GATEWAY_URL" in handoff
    assert "fetch_research_lab_shadow_bundle" not in handoff
    assert "fetch_research_lab_allocation_bundle" not in handoff
    assert "fetch_research_lab_evaluation_bundle_page" not in handoff


def test_weight_input_fetch_budget_leaves_time_for_chain_submission():
    assert validator_integration.WEIGHT_INPUT_FETCH_TIMEOUT_SECONDS == 90
    for fetcher in (
        validator_integration.fetch_research_lab_allocation_bundle,
        validator_integration.fetch_research_lab_attested_allocation_bundle,
    ):
        assert (
            inspect.signature(fetcher).parameters["timeout_seconds"].default
            == 90
        )


def test_primary_validator_rejects_inherited_host_weight_authority():
    validator_source = (ROOT / "neurons" / "validator.py").read_text(
        encoding="utf-8"
    )
    class_start = validator_source.index("class Validator(BaseValidatorNeuron):")
    init_start = validator_source.index("def __init__", class_start)
    class_prefix = validator_source[class_start:init_start]
    assert "def set_weights(self):" in class_prefix
    assert "direct set_weights is disabled" in class_prefix


def test_legacy_validator_rpc_commands_are_removed_from_first_boot_instruction():
    for command, extra in (
        ("get_public_key", {}),
        ("sign_weights", {"weights_hash": "1" * 64}),
        ("get_attestation", {"epoch_id": 1}),
        ("compute_weights_v2", {"snapshot": {}}),
        ):
            response = tee_service.handle_request({"command": command, **extra})
            assert response["status"] == "error"
            assert response["error"] == (
                "Legacy validator V1 RPC is permanently removed"
            )
