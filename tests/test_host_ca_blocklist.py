"""Exact public identities for the vendor-supported host distrust config."""

import base64
import hashlib
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "deploy/ca-trust/blocklist"
EXPECTED = {
    "leadpoet-emsign-ecc-g3.pem": (
        "86a1ecba089c4a8d3bbe2734c612ba341d813e043cf9e8a862cd5c57a36bbe6b",
        "36e68e205b53c67c7a013894e0d5c8583063468118d1ce78ecbc2200d1dd185c",
    ),
    "leadpoet-comodo-aaa.pem": (
        "d7a7a0fb5d7e2731d771e9484ebcdef71d5f0c3e0a2948782bc83ee0ea699ef4",
        "a5ddabd1602ae1c66ce11ad078e734cc473dcb8e9f573037832d8536ae3de90b",
    ),
}


def test_blocklist_contains_only_the_two_reviewed_public_certificates():
    assert {path.name for path in DIRECTORY.iterdir()} == EXPECTED.keys()
    for name, (der_sha256, pem_sha256) in EXPECTED.items():
        path = DIRECTORY / name
        assert path.is_file() and not path.is_symlink()
        payload = path.read_bytes()
        match = re.fullmatch(
            rb"-----BEGIN CERTIFICATE-----\n([A-Za-z0-9+/=\n]+)\n-----END CERTIFICATE-----\n",
            payload,
        )
        assert match, name
        der = base64.b64decode(match[1].replace(b"\n", b""), validate=True)
        assert hashlib.sha256(der).hexdigest() == der_sha256, name
        assert hashlib.sha256(payload).hexdigest() == pem_sha256, name


def test_operator_checksum_manifest_covers_exact_config_bytes():
    lines = (ROOT / "deploy/ca-trust/SHA256SUMS").read_text().splitlines()
    assert len(lines) == len(EXPECTED)
    actual = {}
    for line in lines:
        digest, path = line.split("  ")
        assert path.startswith("deploy/ca-trust/blocklist/")
        actual[Path(path).name] = digest
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert actual == {name: hashes[1] for name, hashes in EXPECTED.items()}
