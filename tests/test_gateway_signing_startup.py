from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_gateway_startup_uses_tee_signing_without_legacy_pem_dependency() -> None:
    source = (REPO_ROOT / "gateway" / "main.py").read_text(encoding="utf-8")

    assert "load_gateway_keypair" not in source
    assert "GATEWAY_PRIVATE_KEY_PASSWORD" not in source
    assert "Receipt integrity ENABLED (canonical hashes + TEE-signed audit events)" in source
    assert "initialize_enclave_event_signing()" in source
    assert "from gateway.tee.enclave_signer" not in source
    assert "No transparency-signing private key exists in the parent process" in source
