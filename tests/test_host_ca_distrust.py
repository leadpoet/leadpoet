"""Pin the public G3 identity and its explicit OpenSSL AUX rejection policy."""

import base64
import hashlib
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "deploy/ca-trust/source/leadpoet-emsign-ecc-g3.trusted.pem"
SOURCE_SHA256 = "661798181d9251cd062f8ece4712bfce980138ac3ac085b4a3c77b12065b892b"
DER_SHA256 = "86a1ecba089c4a8d3bbe2734c612ba341d813e043cf9e8a862cd5c57a36bbe6b"
# DER-encoded OID values. AUX rejects are independent of X.509 extensions.
REJECTED_OIDS = {
    "anyExtendedKeyUsage": "551d2500",       # 2.5.29.37.0
    "serverAuth": "2b06010505070301",       # 1.3.6.1.5.5.7.3.1
    "clientAuth": "2b06010505070302",
    "emailProtection": "2b06010505070304",
    "codeSigning": "2b06010505070303",
    "timeStamping": "2b06010505070308",
    "OCSPSigning": "2b06010505070309",
}


def der_item(value):
    """Read one definite-length DER item; return its content and remainder."""
    assert len(value) >= 2
    tag, size = value[:2]
    offset = 2
    if size & 0x80:
        count = size & 0x7f
        assert 0 < count <= 4 and len(value) >= offset + count
        size = int.from_bytes(value[offset:offset + count], "big")
        offset += count
    end = offset + size
    assert end <= len(value)
    return tag, value[offset:end], value[end:]


def trusted_payload():
    match = re.fullmatch(
        rb"-----BEGIN TRUSTED CERTIFICATE-----\n([A-Za-z0-9+/=\n]+)\n-----END TRUSTED CERTIFICATE-----\n",
        SOURCE.read_bytes(),
    )
    assert match
    return base64.b64decode(match[1].replace(b"\n", b""), validate=True)


def test_exact_public_source_and_unchanged_x509_identity():
    assert {p.name for p in SOURCE.parent.iterdir()} == {SOURCE.name}
    assert not (ROOT / "deploy/ca-trust/blocklist").exists()
    assert SOURCE.is_file() and not SOURCE.is_symlink()
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_SHA256
    payload = trusted_payload()
    tag, _, aux = der_item(payload)
    assert tag == 0x30 and aux
    certificate = payload[:len(payload) - len(aux)]
    assert hashlib.sha256(certificate).hexdigest() == DER_SHA256


def test_aux_has_only_explicit_rejections_and_no_trusted_uses():
    _, _, aux = der_item(trusted_payload())
    tag, content, remainder = der_item(aux)
    assert tag == 0x30 and not remainder
    # X509_CERT_AUX has an optional trust SEQUENCE followed by reject [0].
    # Requiring only [0] excludes every trusted use and any unreviewed AUX field.
    tag, rejects, remainder = der_item(content)
    assert tag == 0xA0 and not remainder
    actual = []
    while rejects:
        tag, oid, rejects = der_item(rejects)
        assert tag == 0x06
        actual.append(oid.hex())
    assert actual == list(REJECTED_OIDS.values())


def test_operator_checksum_manifest_covers_the_only_installed_source():
    relative = SOURCE.relative_to(ROOT).as_posix()
    assert (ROOT / "deploy/ca-trust/SHA256SUMS").read_text() == f"{SOURCE_SHA256}  {relative}\n"
