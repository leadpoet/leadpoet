"""KmsSigner against a fake KMS client: pinned key spec, usage, algorithm, and a
signature the public verifier accepts (labarena.md 5.4, 18.5)."""

from __future__ import annotations

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec

from lab_arena import contracts, signing


class FakeKmsClient:
    def __init__(self, *, key_spec="ECC_NIST_P256", usage="SIGN_VERIFY", algorithms=("ECDSA_SHA_256",)):
        self._key = ec.generate_private_key(ec.SECP256R1())
        self.key_spec = key_spec
        self.usage = usage
        self.algorithms = tuple(algorithms)
        self.sign_calls = []

    def get_public_key(self, KeyId):
        return {
            "KeyId": KeyId,
            "KeySpec": self.key_spec,
            "KeyUsage": self.usage,
            "SigningAlgorithms": list(self.algorithms),
            "PublicKey": self._key.public_key().public_bytes(serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo),
        }

    def sign(self, KeyId, Message, MessageType, SigningAlgorithm):
        self.sign_calls.append((KeyId, MessageType, SigningAlgorithm))
        assert MessageType == "RAW" and SigningAlgorithm == "ECDSA_SHA_256"
        return {"Signature": self._key.sign(Message, ec.ECDSA(hashes.SHA256()))}


def test_kms_signer_signs_documents_the_verifier_accepts():
    client = FakeKmsClient()
    signer = signing.KmsSigner("alias/lab-arena-signing", client=client)
    document = contracts.hashed_document({"round_id": "arena-2026-09-02", "n": 1}, "doc_hash")
    signed = signing.sign_document(signer, document, hash_field="doc_hash")
    assert signed["signature"]["public_key_hash"] == signer.public_key_hash
    assert signing.verify_document_signature(signed, hash_field="doc_hash", public_key_der=signer.public_key_der, expected_public_key_hash=signer.public_key_hash) == document["doc_hash"]
    assert client.sign_calls == [("alias/lab-arena-signing", "RAW", "ECDSA_SHA_256")]
    key_document = signing.signing_key_document(signer.public_key_der)
    assert signing.load_public_key_from_document(key_document) == signer.public_key_der
    other = signing.LocalSigner.generate()
    with pytest.raises(contracts.ArenaSignatureError):
        signing.verify_document_signature(signed, hash_field="doc_hash", public_key_der=other.public_key_der, expected_public_key_hash=other.public_key_hash)


@pytest.mark.parametrize("kwargs", [
    {"key_spec": "RSA_2048"},
    {"usage": "ENCRYPT_DECRYPT"},
    {"algorithms": ("ECDSA_SHA_384",)},
])
def test_kms_signer_refuses_keys_that_do_not_match_the_pinned_identity(kwargs):
    with pytest.raises(contracts.ArenaSignatureError):
        signing.KmsSigner("alias/wrong", client=FakeKmsClient(**kwargs))
