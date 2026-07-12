from pathlib import Path

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


def test_legacy_validator_rpc_commands_are_removed_from_first_boot_instruction():
    for command, extra in (
        ("get_public_key", {}),
        ("sign_weights", {"weights_hash": "1" * 64}),
        ("get_attestation", {"epoch_id": 1}),
        ("compute_weights_v2", {"snapshot": {}}),
    ):
        response = tee_service.handle_request({"command": command, **extra})
        assert response["status"] == "error"
        assert "permanently removed" in response["error"]
