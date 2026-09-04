import base64
import sys
from datetime import datetime, timezone
from types import SimpleNamespace

from lab_arena import wiring


def test_ecr_registry_uses_instance_role_credentials(monkeypatch):
    calls = []

    class Ecr:
        def get_authorization_token(self, **kwargs):
            calls.append(kwargs)
            token = base64.b64encode(b"AWS:temporary-password").decode()
            return {"authorizationData": [{"authorizationToken": token, "expiresAt": datetime.fromtimestamp(5000, timezone.utc)}]}

    monkeypatch.setitem(sys.modules, "boto3", SimpleNamespace(client=lambda service, region_name: Ecr()))
    monkeypatch.setenv(
        "LAB_ARENA_REGISTRY_REPOSITORY",
        "493765492819.dkr.ecr.us-east-1.amazonaws.com/lab-arena-scorer",
    )
    monkeypatch.delenv("LAB_ARENA_REGISTRY_USERNAME", raising=False)
    monkeypatch.delenv("LAB_ARENA_REGISTRY_PASSWORD", raising=False)

    client = wiring.registry_client_from_environment()
    credentials = client._credentials(
        "493765492819.dkr.ecr.us-east-1.amazonaws.com"
    )
    assert credentials == ("AWS", "temporary-password")
    assert calls == [{"registryIds": ["493765492819"]}]


def test_ecr_credentials_refresh_before_expiry(monkeypatch):
    tokens = iter((b"AWS:first", b"AWS:second"))
    calls = []

    class Ecr:
        def get_authorization_token(self, **kwargs):
            calls.append(kwargs)
            return {"authorizationData": [{
                "authorizationToken": base64.b64encode(next(tokens)).decode(),
                "expiresAt": datetime.fromtimestamp(2000, timezone.utc),
            }]}

    now = [1000]
    monkeypatch.setattr(wiring.time, "time", lambda: now[0])
    monkeypatch.setitem(sys.modules, "boto3", SimpleNamespace(client=lambda service, region_name: Ecr()))
    monkeypatch.setenv("LAB_ARENA_REGISTRY_REPOSITORY", "493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model")
    client = wiring.registry_client_from_environment()
    host = "493765492819.dkr.ecr.us-east-1.amazonaws.com"
    assert client._credentials(host) == ("AWS", "first")
    now[0] = 1800
    assert client._credentials(host) == ("AWS", "second")
    assert len(calls) == 2


def test_validator_restart_passes_registry_repository_to_sudo():
    restart = (wiring.Path(__file__).resolve().parents[1] / "validator_restart.sh").read_text()
    assert 'LAB_ARENA_REGISTRY_REPOSITORY="${LAB_ARENA_REGISTRY_REPOSITORY:-}"' in restart
