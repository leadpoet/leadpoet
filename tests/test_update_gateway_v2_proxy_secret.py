from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from gateway.tee.update_gateway_v2_proxy_secret import (
    GatewayV2ProxySecretUpdateError,
    update_gateway_v2_proxy_secret,
)


def _production_environment() -> str:
    lines = [
        "# production-shaped gateway secret",
        "export UNRELATED_ONE='preserve me'",
        "UNRELATED_TWO=value-two",
        "export RESEARCH_LAB_SCORING_WORKER_PROCESS_COUNT=malformed",
    ]
    lines.extend(
        "export RESEARCH_LAB_AUTO_RESEARCH_WEBSHARE_PROXY_%d="
        "'http://user:password@hosted-%d.example.com:%d'"
        % (index, index, 6100 + index)
        for index in range(1, 11)
    )
    lines.extend(
        "export RESEARCH_LAB_QUALIFICATION_WEBSHARE_PROXY_%d="
        "'http://user:password@scoring-%d.example.com:%d'"
        % (index, index, 7100 + index)
        for index in range(1, 26)
    )
    return "\n".join(lines) + "\n"


class FakeSecretsClient:
    def __init__(self, secret: str):
        self.versions = {"initial-version": secret}
        self.current = "initial-version"
        self.put_calls = []

    def get_secret_value(self, *, SecretId, VersionId=None):
        version = VersionId or self.current
        return {
            "ARN": "arn:example",
            "Name": SecretId,
            "VersionId": version,
            "SecretString": self.versions[version],
        }

    def put_secret_value(
        self,
        *,
        SecretId,
        SecretString,
        ClientRequestToken,
    ):
        self.put_calls.append(
            {
                "SecretId": SecretId,
                "SecretString": SecretString,
                "ClientRequestToken": ClientRequestToken,
            }
        )
        self.versions[ClientRequestToken] = SecretString
        self.current = ClientRequestToken
        return {"VersionId": ClientRequestToken}

    def describe_secret(self, *, SecretId):
        return {
            "ARN": "arn:example",
            "Name": SecretId,
            "VersionIdsToStages": {
                version: (
                    ["AWSCURRENT"]
                    if version == self.current
                    else ["AWSPREVIOUS"]
                )
                for version in self.versions
            },
        }


def test_updates_only_v2_proxy_and_removes_stale_capacity(tmp_path):
    original = _production_environment()
    client = FakeSecretsClient(original)
    probes = []

    result = update_gateway_v2_proxy_secret(
        secrets_client=client,
        secret_id="gateway-secret",
        backup_directory=tmp_path,
        scoring_proxy="https://score-user:score-pass@proxy.example.com:443",
        proxy_fleet_probe=lambda fleets: probes.append(fleets),
        now=datetime(2026, 7, 26, tzinfo=timezone.utc),
    )

    assert probes == [
        {
            "gateway_scoring": (
                "https://score-user:score-pass@proxy.example.com:443",
            ),
        }
    ]
    persisted = client.versions[client.current]
    assert "export UNRELATED_ONE='preserve me'\n" in persisted
    assert "UNRELATED_TWO=value-two\n" in persisted
    assert original.splitlines()[4] in persisted
    assert "RESEARCH_LAB_V2_AUTORESEARCH_HTTPS_PROXY_1=" not in persisted
    assert "RESEARCH_LAB_V2_SCORING_HTTPS_PROXY_1=" in persisted
    assert "RESEARCH_LAB_SCORING_WORKER_PROCESS_COUNT=" not in persisted
    assert result["worker_counts"] == {
        "gateway_scoring": 1,
    }
    backup_path = Path(result["backup_path"])
    assert backup_path.read_text(encoding="utf-8") == original
    assert backup_path.stat().st_mode & 0o777 == 0o600
    assert tmp_path.stat().st_mode & 0o777 == 0o700


def test_receipt_counts_existing_distinct_v2_proxy_slots(tmp_path):
    original = _production_environment() + "\n".join(
        (
            "export RESEARCH_LAB_V2_SCORING_HTTPS_PROXY_2=https://second.example.com",
            "export RESEARCH_LAB_V2_SCORING_HTTPS_PROXY_4=https://second.example.com",
            "export RESEARCH_LAB_V2_SCORING_HTTPS_PROXY_7=https://seventh.example.com",
        )
    ) + "\n"
    client = FakeSecretsClient(original)
    probes = []

    result = update_gateway_v2_proxy_secret(
        secrets_client=client,
        secret_id="gateway-secret",
        backup_directory=tmp_path,
        scoring_proxy="https://first.example.com",
        proxy_fleet_probe=lambda fleets: probes.append(fleets) or fleets,
    )

    assert probes == [
        {
            "gateway_scoring": (
                "https://first.example.com",
                "https://second.example.com",
                "https://seventh.example.com",
            )
        }
    ]
    assert result["worker_counts"] == {"gateway_scoring": 3}
    persisted = client.versions[client.current]
    assert "RESEARCH_LAB_V2_SCORING_HTTPS_PROXY_2=" in persisted
    assert "RESEARCH_LAB_V2_SCORING_HTTPS_PROXY_4=" in persisted
    assert "RESEARCH_LAB_V2_SCORING_HTTPS_PROXY_7=" in persisted
    assert "RESEARCH_LAB_SCORING_WORKER_PROCESS_COUNT=" not in persisted


def test_json_update_preserves_unrelated_values_and_removes_stale_count(tmp_path):
    original_values = {
        "UNRELATED_LIST": [1, True, None],
        "UNRELATED_TEXT": " preserve exactly ",
        "RESEARCH_LAB_SCORING_WORKER_PROCESS_COUNT": "15",
        "RESEARCH_LAB_V2_SCORING_HTTPS_PROXY_1": "https://old.example.com",
    }
    client = FakeSecretsClient(json.dumps(original_values, indent=2))

    result = update_gateway_v2_proxy_secret(
        secrets_client=client,
        secret_id="gateway-secret",
        backup_directory=tmp_path,
        scoring_proxy="https://new.example.com",
        proxy_fleet_probe=lambda fleets: fleets,
    )

    persisted = json.loads(client.versions[client.current])
    assert persisted["UNRELATED_LIST"] == original_values["UNRELATED_LIST"]
    assert persisted["UNRELATED_TEXT"] == original_values["UNRELATED_TEXT"]
    assert "RESEARCH_LAB_SCORING_WORKER_PROCESS_COUNT" not in persisted
    assert persisted["RESEARCH_LAB_V2_SCORING_HTTPS_PROXY_1"] == (
        "https://new.example.com"
    )
    assert result["worker_counts"] == {"gateway_scoring": 1}


def test_live_proxy_failure_does_not_write_or_create_backup(tmp_path):
    client = FakeSecretsClient(_production_environment())

    with pytest.raises(
        GatewayV2ProxySecretUpdateError,
        match="live CONNECT validation failed",
    ):
        update_gateway_v2_proxy_secret(
            secrets_client=client,
            secret_id="gateway-secret",
            backup_directory=tmp_path,
            scoring_proxy="https://score.example.com:443",
            proxy_fleet_probe=lambda _fleets: (_ for _ in ()).throw(
                RuntimeError("credential-bearing provider error")
            ),
        )

    assert client.put_calls == []
    assert list(tmp_path.iterdir()) == []


def test_invalid_proxy_fails_before_write(tmp_path):
    client = FakeSecretsClient(_production_environment())

    with pytest.raises(
        GatewayV2ProxySecretUpdateError,
        match="HTTP CONNECT or HTTPS",
    ):
        update_gateway_v2_proxy_secret(
            secrets_client=client,
            secret_id="gateway-secret",
            backup_directory=tmp_path,
            scoring_proxy="socks5://legacy.example.com:6100",
            proxy_fleet_probe=lambda _fleets: None,
        )

    assert client.put_calls == []
    assert list(tmp_path.iterdir()) == []


def test_concurrent_secret_change_aborts_before_write(tmp_path):
    class ConcurrentClient(FakeSecretsClient):
        def __init__(self, secret):
            super().__init__(secret)
            self.read_count = 0

        def get_secret_value(self, *, SecretId, VersionId=None):
            if VersionId is None:
                self.read_count += 1
                if self.read_count == 2:
                    self.versions["concurrent"] = (
                        self.versions[self.current]
                        + "export CONCURRENT=value\n"
                    )
                    self.current = "concurrent"
            return super().get_secret_value(
                SecretId=SecretId,
                VersionId=VersionId,
            )

    client = ConcurrentClient(_production_environment())

    with pytest.raises(
        GatewayV2ProxySecretUpdateError,
        match="changed concurrently",
    ):
        update_gateway_v2_proxy_secret(
            secrets_client=client,
            secret_id="gateway-secret",
            backup_directory=tmp_path,
            scoring_proxy="https://score.example.com:443",
            proxy_fleet_probe=lambda _fleets: None,
        )

    assert client.put_calls == []
    assert len(list(tmp_path.iterdir())) == 1


def test_failed_exact_readback_restores_prior_secret(tmp_path):
    class CorruptReadbackClient(FakeSecretsClient):
        def get_secret_value(self, *, SecretId, VersionId=None):
            response = super().get_secret_value(
                SecretId=SecretId,
                VersionId=VersionId,
            )
            if VersionId and VersionId != "initial-version":
                response["SecretString"] += "export CORRUPTED=value\n"
            return response

    original = _production_environment()
    client = CorruptReadbackClient(original)

    with pytest.raises(
        GatewayV2ProxySecretUpdateError,
        match="exact readback verification",
    ):
        update_gateway_v2_proxy_secret(
            secrets_client=client,
            secret_id="gateway-secret",
            backup_directory=tmp_path,
            scoring_proxy="https://score.example.com:443",
            proxy_fleet_probe=lambda _fleets: None,
        )

    assert len(client.put_calls) == 2
    assert client.versions[client.current] == original
