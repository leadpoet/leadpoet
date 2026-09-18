from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_gateway_startup_reads_measured_identity_without_legacy_pem_dependency() -> None:
    source = (REPO_ROOT / "gateway" / "main.py").read_text(encoding="utf-8")

    assert "load_gateway_keypair" not in source
    assert "GATEWAY_PRIVATE_KEY_PASSWORD" not in source
    assert "initialize_enclave_identity()" in source
    assert "from gateway.tee.enclave_signer" not in source
    assert "Nitro attestation is available for runtime identity verification" in source
    assert "enclave hash chain" not in source
