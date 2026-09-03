"""Miner helper: encrypted key envelopes, image submission bodies, signed requests (7.3, image-by-digest plan)."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest
from bittensor_wallet import Keypair
from cryptography.hazmat.primitives.asymmetric import rsa

from lab_arena import contracts, credentials, images

ROOT = Path(__file__).resolve().parents[2]
MINER = runpy.run_path(str(ROOT / "scripts/lab_arena_miner.py"), run_name="lab_arena_miner_module")
KEY = "sk-or-v1-" + "m" * 40
DIGEST = "sha256:" + "c" * 64


def verify(hotkey, signature, message):
    raw = bytes.fromhex(signature[2:] if signature.startswith("0x") else signature)
    return bool(Keypair(ss58_address=hotkey).verify(message.encode(), raw))


def test_encrypt_key_produces_an_envelope_the_broker_decrypts_and_never_prints_the_key(tmp_path, monkeypatch, capsys):
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    recipient = credentials.recipient_document(private.public_key().public_bytes(__import__("cryptography").hazmat.primitives.serialization.Encoding.DER, __import__("cryptography").hazmat.primitives.serialization.PublicFormat.SubjectPublicKeyInfo))
    recipient_path = tmp_path / "recipient.json"
    recipient_path.write_text(json.dumps(recipient))
    out = tmp_path / "envelope.json"
    key_env = MINER["KEY_ENV_TEMPLATE"] % "OPENROUTER"
    monkeypatch.delenv(key_env, raising=False)
    assert MINER["main"](["encrypt-key", "--provider", "openrouter", "--recipient", str(recipient_path), "--out", str(out)]) == 2
    monkeypatch.setenv(key_env, KEY)
    assert MINER["main"](["encrypt-key", "--provider", "openrouter", "--recipient", str(recipient_path), "--out", str(out)]) == 0
    printed = capsys.readouterr()
    assert KEY not in printed.out and KEY not in printed.err
    envelope = json.loads(out.read_text())
    assert KEY not in out.read_text()
    handle = credentials.decrypt_runtime_key(envelope, credentials.LocalRsaDecryptor(private))
    assert handle.bearer_header()["Authorization"] == "Bearer " + KEY and envelope["provider"] == "openrouter"
    # Every provider key goes through the same command; the envelope binds the provider.
    deepline_key = "dl_" + "d" * 40
    monkeypatch.setenv(MINER["KEY_ENV_TEMPLATE"] % "DEEPLINE", deepline_key)
    deepline_out = tmp_path / "deepline.json"
    assert MINER["main"](["encrypt-key", "--provider", "deepline", "--recipient", str(recipient_path), "--out", str(deepline_out)]) == 0
    deepline_envelope = json.loads(deepline_out.read_text())
    assert deepline_envelope["provider"] == "deepline" and deepline_key not in deepline_out.read_text()
    assert credentials.decrypt_runtime_key(deepline_envelope, credentials.LocalRsaDecryptor(private)).secret() == deepline_key
    assert deepline_key not in capsys.readouterr().out


def test_submission_body_names_one_image_by_digest_with_both_consents(tmp_path, capsys):
    out = tmp_path / "body.json"
    assert MINER["main"](["submission-body", "--image", "ghcr.io/acme/agent:v3@" + DIGEST, "--out", str(out)]) == 0
    report = json.loads(capsys.readouterr().out)
    body = json.loads(out.read_text())
    assert body == {"image_reference": "ghcr.io/acme/agent:v3@" + DIGEST, "consent": {"public_rerun": True, "image_publication": True}}
    assert report["digest"] == DIGEST and contracts.validate_submission_body(body)
    # Docker Hub's short form is normalized; a reference without a registry host or digest is refused.
    assert MINER["main"](["submission-body", "--image", "docker.io/python@" + DIGEST, "--out", str(out)]) == 0
    assert json.loads(out.read_text())["image_reference"] == "docker.io/library/python@" + DIGEST
    for bad in ("acme/agent@" + DIGEST, "ghcr.io/acme/agent:v3"):
        assert MINER["main"](["submission-body", "--image", bad, "--out", str(out)]) == 2
        assert "image reference refused" in capsys.readouterr().err
    with pytest.raises(images.ImageError):
        MINER["submission_body_document"]("ghcr.io/acme/agent")


def test_sign_produces_a_valid_scoped_request(tmp_path, capsys):
    body = tmp_path / "body.json"
    body.write_text(json.dumps({"image_reference": "ghcr.io/acme/agent@" + DIGEST, "consent": {"public_rerun": True, "image_publication": True}}))
    out = tmp_path / "envelope.json"
    assert MINER["main"](["sign", "--scope", "submission", "--round-id", "arena-2026-09-02", "--body", str(body), "--out", str(out), "--hotkey-uri", "//Alice"]) == 0
    envelope = json.loads(out.read_text())
    import time

    validated = contracts.validate_signed_request(envelope, expected_scope=contracts.SCOPE_SUBMISSION, now=int(time.time()), verify_signature=verify, expected_round_id="arena-2026-09-02")
    assert validated["hotkey"] == Keypair.create_from_uri("//Alice").ss58_address
    assert contracts.validate_submission_body(validated["body"])["image_reference"] == "ghcr.io/acme/agent@" + DIGEST
    with pytest.raises(contracts.ArenaContractError):
        contracts.validate_signed_request(envelope, expected_scope=contracts.SCOPE_CLAIM, now=int(time.time()), verify_signature=verify)
