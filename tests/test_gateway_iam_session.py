from pathlib import Path

import pytest

from scripts import gateway_iam_session as subject


def test_reads_only_owned_private_gateway_iam_fields(tmp_path: Path, monkeypatch):
    cache = tmp_path / "gateway.env"
    cache.write_text("AWS_ACCESS_KEY_ID=AKIA" + "A" * 16 + "\nAWS_SECRET_ACCESS_KEY=" + "s" * 40 + "\nOTHER=private\n")
    cache.chmod(0o600)
    captured = {}
    monkeypatch.setattr(subject.boto3.session, "Session", lambda **kwargs: captured.update(kwargs) or "session")
    assert subject.gateway_iam_session(cache=cache) == "session"
    assert captured == {"aws_access_key_id": "AKIA" + "A" * 16,
                        "aws_secret_access_key": "s" * 40,
                        "region_name": "us-east-1"}


def test_rejects_ambient_selector_or_public_cache(tmp_path: Path, monkeypatch):
    cache = tmp_path / "gateway.env"
    cache.write_text("{}")
    cache.chmod(0o644)
    with pytest.raises(subject.GatewayIamSessionError, match="metadata"):
        subject.gateway_iam_session(cache=cache)
    cache.chmod(0o600)
    monkeypatch.setenv("AWS_PROFILE", "unsafe")
    with pytest.raises(subject.GatewayIamSessionError, match="authority"):
        subject.gateway_iam_session(cache=cache)
