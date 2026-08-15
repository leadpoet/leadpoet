from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from gateway.tee.scoring_executor import (
    SCORING_RUNTIME_ENV_NAMES,
    configuration_hash as scoring_configuration_hash,
)
from gateway.tee.update_gateway_rebenchmark_retry_secret import (
    GatewayRebenchmarkRetryUpdateError,
    _parse_shell_environment,
    update_gateway_rebenchmark_retry_secret,
)


def _scoring_hash(*, retry_rounds: str | None) -> str:
    values = {name: None for name in SCORING_RUNTIME_ENV_NAMES}
    values["RESEARCH_LAB_BENCHMARK_PROVIDER_RETRY_ROUNDS"] = retry_rounds
    return scoring_configuration_hash(values)


class FakeSecretsClient:
    def __init__(self, secret: str):
        self.versions = {"initial-version": secret}
        self.current = "initial-version"
        self.put_calls = []

    def get_secret_value(self, *, SecretId, VersionId=None):
        version = VersionId or self.current
        return {
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


@pytest.mark.parametrize(
    ("record", "expected"),
    [
        (r"DUMMY=a\ b", "a b"),
        (r'DUMMY="a\"b"', 'a"b'),
        ("DUMMY='a''b'", "ab"),
        ("DUMMY=plain words", "plain words"),
        ("DUMMY='unmatched", "'unmatched"),
    ],
)
def test_shell_parser_matches_restart_hydration_data_semantics(record, expected):
    assert _parse_shell_environment(record + "\n")["DUMMY"] == expected


def test_updates_absent_defaults_and_preserves_unrelated_shell_values(tmp_path):
    original = (
        "# production-shaped gateway secret\n"
        "\n"
        "export UNRELATED_ONE='preserve me'\n"
        "UNRELATED_TWO=value-two\n"
        "GIT_SSH_COMMAND=ssh -i /tmp/rehearsal-key -o IdentitiesOnly=yes\n"
        "LESSOPEN=| /usr/bin/lesspipe %s\n"
        "SSH_CLIENT=192.0.2.10 41000 22\n"
        "SSH_CONNECTION=192.0.2.10 41000 192.0.2.20 22\n"
        "which_declare=declare -f\n"
        "DUPLICATE_NON_TARGET=same\n"
        "DUPLICATE_NON_TARGET=same\n"
        "UNMATCHED_QUOTE_IS_DATA='preserve literally\n"
    )
    client = FakeSecretsClient(original)

    verified = update_gateway_rebenchmark_retry_secret(
        secrets_client=client,
        expected_prior_scoring_configuration_hash=_scoring_hash(
            retry_rounds=None
        ),
        secret_id="gateway-secret",
        backup_directory=tmp_path,
        now=datetime(2026, 8, 15, tzinfo=timezone.utc),
    )

    assert verified["status"] == "verified"
    assert client.put_calls == []
    assert list(tmp_path.iterdir()) == []

    result = update_gateway_rebenchmark_retry_secret(
        secrets_client=client,
        expected_prior_scoring_configuration_hash=_scoring_hash(
            retry_rounds=None
        ),
        apply=True,
        secret_id="gateway-secret",
        backup_directory=tmp_path,
        now=datetime(2026, 8, 15, tzinfo=timezone.utc),
    )

    persisted = client.versions[client.current]
    assert persisted.startswith(original)
    assert "RESEARCH_LAB_BENCHMARK_PROVIDER_RETRY_ROUNDS=2" in persisted
    assert "RESEARCH_LAB_BENCHMARK_RETRY_CONCURRENCY=1" in persisted
    assert result["status"] == "updated"
    assert result["prior_scoring_configuration_hash"].startswith("sha256:")
    assert result["current_scoring_configuration_hash"].startswith("sha256:")
    assert (
        result["prior_scoring_configuration_hash"]
        != result["current_scoring_configuration_hash"]
    )
    backup_path = Path(result["backup_path"])
    assert backup_path.read_text(encoding="utf-8") == original
    assert backup_path.stat().st_mode & 0o777 == 0o600
    assert tmp_path.stat().st_mode & 0o777 == 0o700


def test_updates_nul_separated_shell_values_without_rewriting_them(tmp_path):
    original = (
        "GIT_SSH_COMMAND=ssh -i /tmp/rehearsal-key\x00"
        "SSH_CLIENT=192.0.2.10 41000 22\x00"
    )
    client = FakeSecretsClient(original)

    result = update_gateway_rebenchmark_retry_secret(
        secrets_client=client,
        expected_prior_scoring_configuration_hash=_scoring_hash(
            retry_rounds=None
        ),
        apply=True,
        secret_id="gateway-secret",
        backup_directory=tmp_path,
    )

    assert result["status"] == "updated"
    assert client.versions[client.current].startswith(original)
    assert Path(result["backup_path"]).read_text(encoding="utf-8") == original


def test_replaces_explicit_shell_targets_once_and_preserves_surrounding_bytes(
    tmp_path,
):
    before = "UNRELATED_BEFORE=keep before\n\n"
    after = "export UNRELATED_AFTER='keep after'\n"
    client = FakeSecretsClient(
        before
        + "RESEARCH_LAB_BENCHMARK_PROVIDER_RETRY_ROUNDS='1'\n"
        + "export RESEARCH_LAB_BENCHMARK_RETRY_CONCURRENCY=2\n"
        + after
    )

    result = update_gateway_rebenchmark_retry_secret(
        secrets_client=client,
        expected_prior_scoring_configuration_hash=_scoring_hash(
            retry_rounds="1"
        ),
        apply=True,
        backup_directory=tmp_path,
    )

    persisted = client.versions[client.current]
    assert result["status"] == "updated"
    assert persisted.startswith(before + after)
    assert persisted.count(
        "RESEARCH_LAB_BENCHMARK_PROVIDER_RETRY_ROUNDS="
    ) == 1
    assert persisted.count("RESEARCH_LAB_BENCHMARK_RETRY_CONCURRENCY=") == 1


@pytest.mark.parametrize(
    "target_name",
    [
        "RESEARCH_LAB_BENCHMARK_PROVIDER_RETRY_ROUNDS",
        "RESEARCH_LAB_BENCHMARK_RETRY_CONCURRENCY",
    ],
)
def test_duplicate_target_setting_fails_before_backup_or_write(
    tmp_path,
    target_name,
):
    client = FakeSecretsClient(
        f"{target_name}=1\n"
        f"export {target_name}=1\n"
    )

    with pytest.raises(
        GatewayRebenchmarkRetryUpdateError,
        match="duplicate retry setting",
    ):
        update_gateway_rebenchmark_retry_secret(
            secrets_client=client,
            expected_prior_scoring_configuration_hash=_scoring_hash(
                retry_rounds=None
            ),
            apply=True,
            backup_directory=tmp_path,
        )

    assert client.put_calls == []
    assert list(tmp_path.iterdir()) == []


def test_duplicate_json_name_fails_before_backup_or_write(tmp_path):
    client = FakeSecretsClient(
        '{"RESEARCH_LAB_BENCHMARK_PROVIDER_RETRY_ROUNDS":"1",'
        '"RESEARCH_LAB_BENCHMARK_PROVIDER_RETRY_ROUNDS":"1"}'
    )

    with pytest.raises(
        GatewayRebenchmarkRetryUpdateError,
        match="duplicate name",
    ):
        update_gateway_rebenchmark_retry_secret(
            secrets_client=client,
            expected_prior_scoring_configuration_hash=_scoring_hash(
                retry_rounds="1"
            ),
            apply=True,
            backup_directory=tmp_path,
        )

    assert client.put_calls == []
    assert list(tmp_path.iterdir()) == []


def test_updates_exact_explicit_old_values_in_json(tmp_path):
    original = json.dumps(
        {
            "UNRELATED": "preserved",
            "RESEARCH_LAB_BENCHMARK_PROVIDER_RETRY_ROUNDS": "1",
            "RESEARCH_LAB_BENCHMARK_RETRY_CONCURRENCY": "2",
        }
    )
    client = FakeSecretsClient(original)

    result = update_gateway_rebenchmark_retry_secret(
        secrets_client=client,
        expected_prior_scoring_configuration_hash=_scoring_hash(
            retry_rounds="1"
        ),
        apply=True,
        backup_directory=tmp_path,
    )

    persisted = json.loads(client.versions[client.current])
    assert persisted == {
        "UNRELATED": "preserved",
        "RESEARCH_LAB_BENCHMARK_PROVIDER_RETRY_ROUNDS": "2",
        "RESEARCH_LAB_BENCHMARK_RETRY_CONCURRENCY": "1",
    }
    assert result["status"] == "updated"


def test_already_applied_is_a_read_only_noop(tmp_path):
    client = FakeSecretsClient(
        "RESEARCH_LAB_BENCHMARK_PROVIDER_RETRY_ROUNDS=2\n"
        "RESEARCH_LAB_BENCHMARK_RETRY_CONCURRENCY=1\n"
    )

    result = update_gateway_rebenchmark_retry_secret(
        secrets_client=client,
        expected_prior_scoring_configuration_hash=_scoring_hash(
            retry_rounds=None
        ),
        backup_directory=tmp_path,
    )

    assert result["status"] == "already_applied"
    assert result["scoring_configuration_hash"].startswith("sha256:")
    assert client.put_calls == []
    assert list(tmp_path.iterdir()) == []


def test_already_applied_rejects_unrelated_checkpoint_identity(tmp_path):
    client = FakeSecretsClient(
        "QUAL_LEADS_PER_ICP=changed\n"
        "RESEARCH_LAB_BENCHMARK_PROVIDER_RETRY_ROUNDS=2\n"
        "RESEARCH_LAB_BENCHMARK_RETRY_CONCURRENCY=1\n"
    )

    with pytest.raises(
        GatewayRebenchmarkRetryUpdateError,
        match="does not match the expected checkpoint hash",
    ):
        update_gateway_rebenchmark_retry_secret(
            secrets_client=client,
            expected_prior_scoring_configuration_hash=_scoring_hash(
                retry_rounds=None
            ),
            backup_directory=tmp_path,
        )

    assert client.put_calls == []
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "secret",
    [
        "RESEARCH_LAB_BENCHMARK_PROVIDER_RETRY_ROUNDS=3\n",
        (
            "RESEARCH_LAB_BENCHMARK_PROVIDER_RETRY_ROUNDS=1\n"
            "RESEARCH_LAB_BENCHMARK_RETRY_CONCURRENCY=3\n"
        ),
        (
            "RESEARCH_LAB_BENCHMARK_PROVIDER_RETRY_ROUNDS=2\n"
            "RESEARCH_LAB_BENCHMARK_RETRY_CONCURRENCY=2\n"
        ),
    ],
)
def test_unexpected_or_partial_state_fails_before_write(tmp_path, secret):
    client = FakeSecretsClient(secret)

    with pytest.raises(
        GatewayRebenchmarkRetryUpdateError,
        match="expected old value|expected old value or be unset",
    ):
        update_gateway_rebenchmark_retry_secret(
            secrets_client=client,
            expected_prior_scoring_configuration_hash=_scoring_hash(
                retry_rounds=None
            ),
            apply=True,
            backup_directory=tmp_path,
        )

    assert client.put_calls == []
    assert list(tmp_path.iterdir()) == []


def test_checkpoint_hash_mismatch_fails_without_backup_or_write(tmp_path):
    client = FakeSecretsClient("UNRELATED=value\n")

    with pytest.raises(
        GatewayRebenchmarkRetryUpdateError,
        match="does not match the expected checkpoint hash",
    ):
        update_gateway_rebenchmark_retry_secret(
            secrets_client=client,
            expected_prior_scoring_configuration_hash=_scoring_hash(
                retry_rounds="1"
            ),
            apply=True,
            backup_directory=tmp_path,
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
                        self.versions[self.current] + "CONCURRENT=value\n"
                    )
                    self.current = "concurrent"
            return super().get_secret_value(
                SecretId=SecretId,
                VersionId=VersionId,
            )

    client = ConcurrentClient("UNRELATED=value\n")

    with pytest.raises(
        GatewayRebenchmarkRetryUpdateError,
        match="changed concurrently",
    ):
        update_gateway_rebenchmark_retry_secret(
            secrets_client=client,
            expected_prior_scoring_configuration_hash=_scoring_hash(
                retry_rounds=None
            ),
            apply=True,
            backup_directory=tmp_path,
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
                response["SecretString"] += "CORRUPTED=value\n"
            return response

    original = "UNRELATED=value\n"
    client = CorruptReadbackClient(original)

    with pytest.raises(
        GatewayRebenchmarkRetryUpdateError,
        match="exact readback verification",
    ):
        update_gateway_rebenchmark_retry_secret(
            secrets_client=client,
            expected_prior_scoring_configuration_hash=_scoring_hash(
                retry_rounds=None
            ),
            apply=True,
            backup_directory=tmp_path,
        )

    assert len(client.put_calls) == 2
    assert client.versions[client.current] == original
