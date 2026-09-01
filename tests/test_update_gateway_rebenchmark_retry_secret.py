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
    reconcile_gateway_rebenchmark_runtime_environment,
    reconcile_gateway_rebenchmark_runtime_environment_file,
    update_gateway_rebenchmark_concurrency_secret,
    update_gateway_rebenchmark_retry_secret,
)


def _scoring_hash(
    *,
    retry_rounds: str | None,
) -> str:
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


def test_runtime_reconciliation_removes_stale_retry_defaults():
    runtime = (
        "export UNRELATED=preserved\n"
        "export RESEARCH_LAB_BENCHMARK_PROVIDER_RETRY_ROUNDS=2\n"
        "export RESEARCH_LAB_BENCHMARK_RETRY_CONCURRENCY=1\n"
    )

    reconciled = reconcile_gateway_rebenchmark_runtime_environment(
        runtime,
        authoritative_environment="export UNRELATED=preserved\n",
    )

    assert _parse_shell_environment(reconciled) == {"UNRELATED": "preserved"}


def test_runtime_reconciliation_uses_explicit_authoritative_retry_values():
    runtime = (
        "export UNRELATED=preserved\n"
        "export RESEARCH_LAB_BENCHMARK_PROVIDER_RETRY_ROUNDS=1\n"
        "export RESEARCH_LAB_BENCHMARK_PROVIDER_RETRY_ROUNDS=2\n"
        "export RESEARCH_LAB_BENCHMARK_RETRY_CONCURRENCY=2\n"
        "export RESEARCH_LAB_BENCHMARK_RETRY_CONCURRENCY=1\n"
    )
    authority = json.dumps(
        {
            "UNRELATED": "preserved",
            "RESEARCH_LAB_BENCHMARK_PROVIDER_RETRY_ROUNDS": "2",
            "RESEARCH_LAB_BENCHMARK_RETRY_CONCURRENCY": "1",
        }
    )

    reconciled = reconcile_gateway_rebenchmark_runtime_environment(
        runtime,
        authoritative_environment=authority,
    )
    parsed = _parse_shell_environment(reconciled)

    assert parsed["UNRELATED"] == "preserved"
    assert parsed["RESEARCH_LAB_BENCHMARK_PROVIDER_RETRY_ROUNDS"] == "2"
    assert parsed["RESEARCH_LAB_BENCHMARK_RETRY_CONCURRENCY"] == "1"
    assert reconciled.count(
        "RESEARCH_LAB_BENCHMARK_PROVIDER_RETRY_ROUNDS="
    ) == 1
    assert reconciled.count("RESEARCH_LAB_BENCHMARK_RETRY_CONCURRENCY=") == 1


def test_runtime_file_reconciliation_is_atomic_and_owner_only(tmp_path):
    runtime_path = tmp_path / "runtime.env"
    authority_path = tmp_path / "authority.env"
    runtime_path.write_text(
        "export UNRELATED=preserved\n"
        "export RESEARCH_LAB_BENCHMARK_PROVIDER_RETRY_ROUNDS=2\n"
        "export RESEARCH_LAB_BENCHMARK_RETRY_CONCURRENCY=1\n",
        encoding="utf-8",
    )
    authority_path.write_text("export UNRELATED=preserved\n", encoding="utf-8")

    result = reconcile_gateway_rebenchmark_runtime_environment_file(
        runtime_environment_path=runtime_path,
        authoritative_environment_path=authority_path,
    )

    assert result == {
        "status": "reconciled",
        "managed_name_count": 2,
        "present_name_count": 0,
        "absent_name_count": 2,
    }
    assert _parse_shell_environment(runtime_path.read_text(encoding="utf-8")) == {
        "UNRELATED": "preserved"
    }
    assert runtime_path.stat().st_mode & 0o777 == 0o600


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
        match="duplicate target setting",
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


def test_concurrency_update_is_exact_and_preserves_unrelated_values(tmp_path):
    original = (
        "UNRELATED='preserve me'\n"
        "RESEARCH_LAB_BENCHMARK_CONCURRENCY=5\n"
        "RESEARCH_LAB_BENCHMARK_PROVIDER_RETRY_ROUNDS=1\n"
    )
    client = FakeSecretsClient(original)
    prior_hash = _scoring_hash(
        retry_rounds="1",
    )

    verified = update_gateway_rebenchmark_concurrency_secret(
        secrets_client=client,
        expected_prior_scoring_configuration_hash=prior_hash,
        expected_current_concurrency=5,
        target_concurrency=20,
        backup_directory=tmp_path,
    )

    assert verified["status"] == "verified"
    assert verified["prior_first_pass_concurrency"] == 5
    assert verified["current_first_pass_concurrency"] == 20
    assert client.put_calls == []
    assert list(tmp_path.iterdir()) == []

    result = update_gateway_rebenchmark_concurrency_secret(
        secrets_client=client,
        expected_prior_scoring_configuration_hash=prior_hash,
        expected_current_concurrency=5,
        target_concurrency=20,
        apply=True,
        backup_directory=tmp_path,
        now=datetime(2026, 8, 28, tzinfo=timezone.utc),
    )

    persisted = client.versions[client.current]
    assert "UNRELATED='preserve me'" in persisted
    assert "RESEARCH_LAB_BENCHMARK_PROVIDER_RETRY_ROUNDS=1" in persisted
    assert persisted.count("RESEARCH_LAB_BENCHMARK_CONCURRENCY=") == 1
    assert "RESEARCH_LAB_BENCHMARK_CONCURRENCY=20" in persisted
    assert result["status"] == "updated"
    assert Path(result["backup_path"]).read_text(encoding="utf-8") == original


def test_concurrency_update_supports_default_one_and_json(tmp_path):
    original = json.dumps({"UNRELATED": "preserved"})
    client = FakeSecretsClient(original)

    result = update_gateway_rebenchmark_concurrency_secret(
        secrets_client=client,
        expected_prior_scoring_configuration_hash=_scoring_hash(
            retry_rounds=None,
        ),
        expected_current_concurrency=1,
        target_concurrency=8,
        apply=True,
        backup_directory=tmp_path,
    )

    assert result["status"] == "updated"
    assert json.loads(client.versions[client.current]) == {
        "UNRELATED": "preserved",
        "RESEARCH_LAB_BENCHMARK_CONCURRENCY": "8",
    }


def test_concurrency_update_is_idempotent_against_prior_hash(tmp_path):
    client = FakeSecretsClient(
        "UNRELATED=value\n"
        "RESEARCH_LAB_BENCHMARK_CONCURRENCY=20\n"
    )
    result = update_gateway_rebenchmark_concurrency_secret(
        secrets_client=client,
        expected_prior_scoring_configuration_hash=_scoring_hash(
            retry_rounds=None,
        ),
        expected_current_concurrency=5,
        target_concurrency=20,
        apply=True,
        backup_directory=tmp_path,
    )

    assert result["status"] == "already_applied"
    assert result["current_first_pass_concurrency"] == 20
    assert client.put_calls == []
    assert list(tmp_path.iterdir()) == []


def test_retry_concurrency_update_is_exact_and_scoring_identity_neutral(
    tmp_path,
):
    original = (
        "UNRELATED='preserve me'\n"
        "RESEARCH_LAB_BENCHMARK_CONCURRENCY=5\n"
        "RESEARCH_LAB_BENCHMARK_PROVIDER_RETRY_ROUNDS=2\n"
        "RESEARCH_LAB_BENCHMARK_RETRY_CONCURRENCY=1\n"
    )
    client = FakeSecretsClient(original)
    prior_hash = _scoring_hash(retry_rounds="2")

    verified = update_gateway_rebenchmark_concurrency_secret(
        secrets_client=client,
        expected_prior_scoring_configuration_hash=prior_hash,
        expected_current_concurrency=1,
        target_concurrency=2,
        retry_concurrency=True,
        backup_directory=tmp_path,
    )

    assert verified == {
        "status": "verified",
        "secret_id": "leadpoet/prod/gateway/env",
        "prior_retry_concurrency": 1,
        "current_retry_concurrency": 2,
        "prior_scoring_configuration_hash": prior_hash,
        "current_scoring_configuration_hash": prior_hash,
    }
    assert client.put_calls == []

    updated = update_gateway_rebenchmark_concurrency_secret(
        secrets_client=client,
        expected_prior_scoring_configuration_hash=prior_hash,
        expected_current_concurrency=1,
        target_concurrency=2,
        retry_concurrency=True,
        apply=True,
        backup_directory=tmp_path,
        now=datetime(2026, 9, 1, tzinfo=timezone.utc),
    )

    persisted = client.versions[client.current]
    assert "UNRELATED='preserve me'" in persisted
    assert "RESEARCH_LAB_BENCHMARK_CONCURRENCY=5" in persisted
    assert "RESEARCH_LAB_BENCHMARK_PROVIDER_RETRY_ROUNDS=2" in persisted
    assert persisted.count("RESEARCH_LAB_BENCHMARK_RETRY_CONCURRENCY=") == 1
    assert "RESEARCH_LAB_BENCHMARK_RETRY_CONCURRENCY=2" in persisted
    assert updated["prior_scoring_configuration_hash"] == prior_hash
    assert updated["current_scoring_configuration_hash"] == prior_hash
    assert Path(updated["backup_path"]).read_text(encoding="utf-8") == original


@pytest.mark.parametrize(
    ("expected_current", "target"),
    [(0, 20), (5, 0), (5, 65), (True, 20)],
)
def test_concurrency_update_rejects_out_of_bounds_before_secret_read(
    tmp_path,
    expected_current,
    target,
):
    class NoReadClient(FakeSecretsClient):
        def get_secret_value(self, **_kwargs):
            raise AssertionError("secret must not be read")

    with pytest.raises(
        GatewayRebenchmarkRetryUpdateError,
        match="invalid|between 1 and 64",
    ):
        update_gateway_rebenchmark_concurrency_secret(
            secrets_client=NoReadClient(""),
            expected_prior_scoring_configuration_hash=_scoring_hash(
                retry_rounds=None,
            ),
            expected_current_concurrency=expected_current,
            target_concurrency=target,
            apply=True,
            backup_directory=tmp_path,
        )


def test_concurrency_update_rejects_wrong_old_value_without_write(tmp_path):
    client = FakeSecretsClient(
        "RESEARCH_LAB_BENCHMARK_CONCURRENCY=5\n"
    )

    with pytest.raises(
        GatewayRebenchmarkRetryUpdateError,
        match="does not match the expected old value",
    ):
        update_gateway_rebenchmark_concurrency_secret(
            secrets_client=client,
            expected_prior_scoring_configuration_hash=_scoring_hash(
                retry_rounds=None,
            ),
            expected_current_concurrency=4,
            target_concurrency=20,
            apply=True,
            backup_directory=tmp_path,
        )

    assert client.put_calls == []
    assert list(tmp_path.iterdir()) == []
