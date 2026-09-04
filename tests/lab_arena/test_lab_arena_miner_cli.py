"""Miner helper: image submission bodies and signed requests."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

from bittensor_wallet import Keypair
import pytest

from lab_arena import contracts, images

ROOT = Path(__file__).resolve().parents[2]
MINER = runpy.run_path(str(ROOT / "scripts/lab_arena_miner.py"), run_name="lab_arena_miner_module")
DIGEST = "sha256:" + "c" * 64


def verify(hotkey, signature, message):
    raw = bytes.fromhex(signature[2:] if signature.startswith("0x") else signature)
    return bool(Keypair(ss58_address=hotkey).verify(message.encode(), raw))


def test_submission_body_accepts_a_tag_or_digest_with_one_consent(tmp_path, capsys):
    out = tmp_path / "body.json"
    assert MINER["main"](["submission-body", "--image", "ghcr.io/acme/agent:v3@" + DIGEST, "--out", str(out)]) == 0
    report = json.loads(capsys.readouterr().out)
    body = json.loads(out.read_text())
    assert body == {"image_reference": "ghcr.io/acme/agent:v3@" + DIGEST, "consent": {"public_rerun": True}}
    assert report["image_reference"] == body["image_reference"] and contracts.validate_submission_body(body)
    # Docker Hub's short form is normalized; a public tag is also accepted.
    assert MINER["main"](["submission-body", "--image", "docker.io/python@" + DIGEST, "--out", str(out)]) == 0
    assert json.loads(out.read_text())["image_reference"] == "docker.io/library/python@" + DIGEST
    assert MINER["main"](["submission-body", "--image", "ghcr.io/acme/agent:v3", "--out", str(out)]) == 0
    assert json.loads(out.read_text())["image_reference"] == "ghcr.io/acme/agent:v3"
    for bad in ("acme/agent@" + DIGEST, "ghcr.io/acme/agent"):
        assert MINER["main"](["submission-body", "--image", bad, "--out", str(out)]) == 2
        assert "image reference refused" in capsys.readouterr().err
    with pytest.raises(images.ImageError):
        MINER["submission_body_document"]("ghcr.io/acme/agent")


def test_sign_produces_a_valid_scoped_request(tmp_path, capsys):
    body = tmp_path / "body.json"
    body.write_text(json.dumps({"image_reference": "ghcr.io/acme/agent@" + DIGEST, "consent": {"public_rerun": True}}))
    out = tmp_path / "envelope.json"
    assert MINER["main"](["sign", "--scope", "submission", "--round-id", "arena-2026-09-02", "--body", str(body), "--out", str(out), "--hotkey-uri", "//Alice"]) == 0
    envelope = json.loads(out.read_text())
    import time

    validated = contracts.validate_signed_request(envelope, expected_scope=contracts.SCOPE_SUBMISSION, now=int(time.time()), verify_signature=verify, expected_round_id="arena-2026-09-02")
    assert validated["hotkey"] == Keypair.create_from_uri("//Alice").ss58_address
    assert contracts.validate_submission_body(validated["body"])["image_reference"] == "ghcr.io/acme/agent@" + DIGEST
    with pytest.raises(contracts.ArenaContractError):
        contracts.validate_signed_request(envelope, expected_scope=contracts.SCOPE_CLAIM, now=int(time.time()), verify_signature=verify)
