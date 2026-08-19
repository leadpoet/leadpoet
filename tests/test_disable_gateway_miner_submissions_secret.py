from __future__ import annotations

import json

import pytest

from gateway.tee import disable_gateway_miner_submissions_secret as operation


INITIAL_VERSION = "11111111-1111-4111-8111-111111111111"
CONCURRENT_VERSION = "22222222-2222-4222-8222-222222222222"
PREVIOUS_VERSION = "33333333-3333-4333-8333-333333333333"
PENDING_VERSION = "44444444-4444-4444-8444-444444444444"


class FakeSecretsClient:
    def __init__(self, secret: str, *, omit_unlabeled: bool = False):
        self.versions = {INITIAL_VERSION: secret}
        self.stages = {INITIAL_VERSION: {"AWSCURRENT"}}
        self.omit_unlabeled = omit_unlabeled
        self.put_calls: list[dict[str, object]] = []
        self.stage_calls: list[dict[str, str]] = []

    def add_version(self, version: str, labels: set[str]) -> None:
        self.versions[version] = f"OLD_VERSION={version}\n"
        self.stages[version] = set(labels)

    @property
    def current(self) -> str:
        current = [
            version for version, labels in self.stages.items() if "AWSCURRENT" in labels
        ]
        if len(current) != 1:
            raise RuntimeError("current stage is ambiguous")
        return current[0]

    def get_secret_value(
        self,
        *,
        SecretId,
        VersionId=None,
        VersionStage=None,
    ):
        assert SecretId == operation.GATEWAY_SECRET_ID
        if VersionId is not None:
            version = VersionId
        elif VersionStage == "AWSCURRENT":
            version = self.current
        else:
            raise AssertionError("test reads must be version-bound")
        return {
            "Name": SecretId,
            "VersionId": version,
            "SecretString": self.versions[version],
        }

    def describe_secret(self, *, SecretId):
        assert SecretId == operation.GATEWAY_SECRET_ID
        return {
            "Name": SecretId,
            "VersionIdsToStages": {
                version: sorted(labels)
                for version, labels in self.stages.items()
                if labels or not self.omit_unlabeled
            },
        }

    def put_secret_value(
        self,
        *,
        SecretId,
        SecretString,
        ClientRequestToken,
        VersionStages,
    ):
        assert SecretId == operation.GATEWAY_SECRET_ID
        if ClientRequestToken in self.versions:
            raise RuntimeError("version token already exists")
        for label in VersionStages:
            if any(label in labels for labels in self.stages.values()):
                raise RuntimeError("stage label already exists")
        self.put_calls.append(
            {
                "SecretId": SecretId,
                "SecretString": SecretString,
                "ClientRequestToken": ClientRequestToken,
                "VersionStages": list(VersionStages),
            }
        )
        self.versions[ClientRequestToken] = SecretString
        self.stages[ClientRequestToken] = set(VersionStages)
        return {"VersionId": ClientRequestToken}

    def update_secret_version_stage(
        self,
        *,
        SecretId,
        VersionStage,
        MoveToVersionId=None,
        RemoveFromVersionId=None,
    ):
        assert SecretId == operation.GATEWAY_SECRET_ID
        self.stage_calls.append(
            {
                "VersionStage": VersionStage,
                "MoveToVersionId": MoveToVersionId or "",
                "RemoveFromVersionId": RemoveFromVersionId or "",
            }
        )
        if RemoveFromVersionId is not None:
            labels = self.stages.get(RemoveFromVersionId, set())
            if VersionStage not in labels:
                raise RuntimeError("version-stage fence failed")
        if MoveToVersionId is not None:
            if MoveToVersionId not in self.stages:
                raise RuntimeError("move target is absent")
            holders = [
                version
                for version, labels in self.stages.items()
                if VersionStage in labels
            ]
            if holders and holders != [RemoveFromVersionId]:
                raise RuntimeError("stage move removed the wrong owner")
            if VersionStage == "AWSCURRENT" and RemoveFromVersionId is not None:
                for labels in self.stages.values():
                    labels.discard("AWSPREVIOUS")
                self.stages[RemoveFromVersionId].add("AWSPREVIOUS")
            if RemoveFromVersionId is not None:
                self.stages[RemoveFromVersionId].discard(VersionStage)
            self.stages[MoveToVersionId].add(VersionStage)
        elif RemoveFromVersionId is not None:
            self.stages[RemoveFromVersionId].remove(VersionStage)
        return {}

    def install_concurrent_current(self) -> None:
        prior = self.current
        self.versions[CONCURRENT_VERSION] = self.versions[prior] + "CONCURRENT=value\n"
        for labels in self.stages.values():
            labels.discard("AWSPREVIOUS")
        self.stages[prior].add("AWSPREVIOUS")
        self.stages[prior].discard("AWSCURRENT")
        self.stages[CONCURRENT_VERSION] = {"AWSCURRENT"}


def _apply(client: FakeSecretsClient):
    return operation.disable_gateway_miner_submissions_secret(
        secrets_client=client,
        expected_current_version_id=INITIAL_VERSION,
        apply=True,
    )


def _custom_labels(client: FakeSecretsClient) -> set[str]:
    return {
        label
        for labels in client.stages.values()
        for label in labels
        if label.startswith(operation._CUSTOM_STAGE_PREFIX)
    }


def test_verify_is_read_only_and_reports_only_commitments_and_version_ids():
    original = (
        "RESEARCH_LAB_SCORING_WORKER_ENABLED=false\n"
        "RESEARCH_LAB_AUTORESEARCH_WORKER_COUNT=0\n"
        "UNRELATED_SECRET=do-not-print\n"
    )
    client = FakeSecretsClient(original)

    result = operation.disable_gateway_miner_submissions_secret(
        secrets_client=client,
    )

    assert result == {
        "status": "verified",
        "secret_id": operation.GATEWAY_SECRET_ID,
        "current_version_id": INITIAL_VERSION,
        "backup_version_id": INITIAL_VERSION,
        "document_format": "shell",
        "prior_document_commitment": operation._document_commitment(original),
        "candidate_document_commitment": operation._document_commitment(
            original + "RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED=false\n"
        ),
        "prior_stage_topology_commitment": operation._topology_commitment(
            {INITIAL_VERSION: frozenset({"AWSCURRENT"})}
        ),
    }
    assert client.put_calls == []
    assert client.stage_calls == []
    assert "do-not-print" not in json.dumps(result)


def test_apply_stages_then_cas_promotes_and_preserves_unrelated_shell_bytes():
    original = (
        "# keep this comment\n"
        "RESEARCH_LAB_SCORING_WORKER_ENABLED=false\n"
        "export RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED='true'\n"
        "RESEARCH_LAB_AUTORESEARCH_WORKER_COUNT=0\n"
    )
    client = FakeSecretsClient(original)

    result = _apply(client)

    expected = (
        "# keep this comment\n"
        "RESEARCH_LAB_SCORING_WORKER_ENABLED=false\n"
        "RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED=false\n"
        "RESEARCH_LAB_AUTORESEARCH_WORKER_COUNT=0\n"
    )
    assert result["status"] == "updated"
    assert result["backup_version_id"] == INITIAL_VERSION
    assert result["current_version_id"] == result["candidate_version_id"]
    assert client.versions[client.current] == expected
    assert client.versions[INITIAL_VERSION] == original
    assert client.stages[INITIAL_VERSION] == {"AWSPREVIOUS"}
    assert client.stages[client.current] == {"AWSCURRENT"}
    custom_stage = str(client.put_calls[0]["VersionStages"][0])
    assert custom_stage.startswith(operation._CUSTOM_STAGE_PREFIX)
    assert custom_stage != "AWSPENDING"
    assert len(custom_stage) <= 256
    assert client.stage_calls[0] == {
        "VersionStage": "AWSCURRENT",
        "MoveToVersionId": result["candidate_version_id"],
        "RemoveFromVersionId": INITIAL_VERSION,
    }
    assert client.stage_calls[1]["VersionStage"] == custom_stage


def test_success_preserves_preexisting_pending_and_uses_standard_previous_move():
    client = FakeSecretsClient("UNCHANGED=value\n", omit_unlabeled=True)
    client.add_version(PREVIOUS_VERSION, {"AWSPREVIOUS"})
    client.add_version(PENDING_VERSION, {"AWSPENDING"})

    result = _apply(client)

    candidate_id = result["candidate_version_id"]
    topology = operation._version_stages(client)
    assert topology[PENDING_VERSION] == frozenset({"AWSPENDING"})
    assert topology[INITIAL_VERSION] == frozenset({"AWSPREVIOUS"})
    assert topology[candidate_id] == frozenset({"AWSCURRENT"})
    assert PREVIOUS_VERSION not in topology
    assert _custom_labels(client) == set()
    staged_label = str(client.put_calls[0]["VersionStages"][0])
    assert staged_label != "AWSPENDING"


def test_apply_json_changes_only_fixed_target_semantics():
    original_document = {
        "RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED": "yes",
        "RESEARCH_LAB_SCORING_WORKER_ENABLED": "false",
        "RESEARCH_LAB_AUTORESEARCH_WORKER_COUNT": "0",
        "NESTED_UNRELATED": {"keep": [1, True, None]},
    }
    client = FakeSecretsClient(json.dumps(original_document, indent=2))

    result = _apply(client)

    persisted = json.loads(client.versions[client.current])
    assert result["document_format"] == "json"
    assert persisted == {
        **original_document,
        "RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED": "false",
    }
    assert json.loads(client.versions[INITIAL_VERSION]) == original_document


def test_apply_preserves_nul_separated_unrelated_records():
    original = "UNCHANGED=one\x00RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED=1\x00"
    client = FakeSecretsClient(original)

    _apply(client)

    assert client.versions[client.current] == (
        "UNCHANGED=one\x00RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED=false\x00"
    )


def test_already_false_is_idempotent_even_with_apply():
    original = "RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED=false\nUNCHANGED=value\n"
    client = FakeSecretsClient(original)

    result = _apply(client)

    assert result["status"] == "already_disabled"
    assert result["current_version_id"] == INITIAL_VERSION
    assert client.put_calls == []
    assert client.stage_calls == []


@pytest.mark.parametrize("known_false", ["0", "no", "off"])
def test_known_false_alias_is_canonicalized(known_false):
    client = FakeSecretsClient(
        f"RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED={known_false}\nUNCHANGED=value\n"
    )

    _apply(client)

    assert (
        "RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED=false\n"
        in client.versions[client.current]
    )


@pytest.mark.parametrize(
    ("secret", "message"),
    [
        (
            "RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED=maybe\n",
            "unknown boolean value",
        ),
        (
            "RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED=true\n"
            "RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED=true\n",
            "duplicate names",
        ),
        ('{"RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED":null}', "must be text"),
        ('{"BROKEN":', "JSON is malformed"),
        ("NOT_AN_ASSIGNMENT\n", "environment is malformed"),
        ("UNCHANGED=value\x00SECOND=value\n", "format is unsupported"),
    ],
)
def test_unknown_or_ambiguous_documents_fail_without_staging(secret, message):
    client = FakeSecretsClient(secret)

    with pytest.raises(operation.GatewayMinerSubmissionsDisableError, match=message):
        _apply(client)

    assert client.put_calls == []
    assert client.stage_calls == []


def test_apply_requires_matching_prior_verification_version():
    client = FakeSecretsClient("UNCHANGED=value\n")

    with pytest.raises(
        operation.GatewayMinerSubmissionsDisableError,
        match="differs from the expected current version",
    ):
        operation.disable_gateway_miner_submissions_secret(
            secrets_client=client,
            expected_current_version_id=CONCURRENT_VERSION,
            apply=True,
        )

    assert client.put_calls == []


def test_concurrent_change_before_staging_aborts_without_candidate_version():
    class ConcurrentBeforeStageClient(FakeSecretsClient):
        current_reads = 0

        def get_secret_value(self, **kwargs):
            if kwargs.get("VersionStage") == "AWSCURRENT":
                self.current_reads += 1
                if self.current_reads == 2:
                    self.install_concurrent_current()
            return super().get_secret_value(**kwargs)

    client = ConcurrentBeforeStageClient("UNCHANGED=value\n")

    with pytest.raises(
        operation.GatewayMinerSubmissionsDisableError,
        match="changed concurrently before staging",
    ):
        _apply(client)

    assert client.current == CONCURRENT_VERSION
    assert client.put_calls == []


def test_cas_promotion_cannot_overwrite_last_moment_concurrent_current():
    class ConcurrentAtPromotionClient(FakeSecretsClient):
        injected = False

        def update_secret_version_stage(self, **kwargs):
            if (
                kwargs.get("VersionStage") == "AWSCURRENT"
                and kwargs.get("MoveToVersionId")
                and not self.injected
            ):
                self.injected = True
                self.install_concurrent_current()
            return super().update_secret_version_stage(**kwargs)

    client = ConcurrentAtPromotionClient("UNCHANGED=value\n")

    with pytest.raises(
        operation.GatewayMinerSubmissionsDisableError,
        match="candidate cleanup could not restore the original stage topology",
    ):
        _apply(client)

    assert client.current == CONCURRENT_VERSION
    candidate_id = str(client.put_calls[0]["ClientRequestToken"])
    assert _custom_labels(client) == set()
    assert "AWSCURRENT" not in client.stages[candidate_id]


def test_failed_promotion_restores_preexisting_pending_and_previous_topology():
    class FailedPromotionClient(FakeSecretsClient):
        def update_secret_version_stage(self, **kwargs):
            if kwargs.get("VersionStage") == "AWSCURRENT":
                raise RuntimeError("promotion did not run")
            return super().update_secret_version_stage(**kwargs)

    client = FailedPromotionClient("UNCHANGED=value\n", omit_unlabeled=True)
    client.add_version(PREVIOUS_VERSION, {"AWSPREVIOUS"})
    client.add_version(PENDING_VERSION, {"AWSPENDING"})
    original_topology = operation._version_stages(client)

    with pytest.raises(
        operation.GatewayMinerSubmissionsDisableError,
        match="version-fenced gateway secret promotion failed",
    ):
        _apply(client)

    assert operation._version_stages(client) == original_topology
    assert _custom_labels(client) == set()


def test_ambiguous_staging_response_cleans_custom_candidate_stage():
    class AmbiguousStagingClient(FakeSecretsClient):
        def put_secret_value(self, **kwargs):
            super().put_secret_value(**kwargs)
            raise RuntimeError("response lost after staging")

    client = AmbiguousStagingClient("UNCHANGED=value\n", omit_unlabeled=True)
    client.add_version(PREVIOUS_VERSION, {"AWSPREVIOUS"})
    client.add_version(PENDING_VERSION, {"AWSPENDING"})
    original_topology = operation._version_stages(client)

    with pytest.raises(
        operation.GatewayMinerSubmissionsDisableError,
        match="could not be staged",
    ):
        _apply(client)

    candidate_id = str(client.put_calls[0]["ClientRequestToken"])
    assert client.current == INITIAL_VERSION
    assert _custom_labels(client) == set()
    assert candidate_id not in operation._version_stages(client)
    assert operation._version_stages(client) == original_topology


def test_ambiguous_promotion_response_accepts_exact_promoted_readback():
    class AmbiguousPromotionClient(FakeSecretsClient):
        raised = False

        def update_secret_version_stage(self, **kwargs):
            result = super().update_secret_version_stage(**kwargs)
            if kwargs.get("VersionStage") == "AWSCURRENT" and not self.raised:
                self.raised = True
                raise RuntimeError("response lost after promotion")
            return result

    client = AmbiguousPromotionClient("UNCHANGED=value\n", omit_unlabeled=True)
    client.add_version(PREVIOUS_VERSION, {"AWSPREVIOUS"})
    client.add_version(PENDING_VERSION, {"AWSPENDING"})

    result = _apply(client)

    assert result["status"] == "updated"
    assert client.current == result["candidate_version_id"]
    assert client.stages[client.current] == {"AWSCURRENT"}
    assert client.stages[PENDING_VERSION] == {"AWSPENDING"}
    assert _custom_labels(client) == set()


def test_corrupt_staged_readback_cleans_custom_stage_and_keeps_prior_current():
    class CorruptPendingClient(FakeSecretsClient):
        def get_secret_value(self, **kwargs):
            response = super().get_secret_value(**kwargs)
            version = kwargs.get("VersionId")
            if version is not None and version != INITIAL_VERSION:
                response["SecretString"] += "CORRUPTED=value\n"
            return response

    client = CorruptPendingClient("UNCHANGED=value\n", omit_unlabeled=True)
    client.add_version(PREVIOUS_VERSION, {"AWSPREVIOUS"})
    client.add_version(PENDING_VERSION, {"AWSPENDING"})
    original_topology = operation._version_stages(client)

    with pytest.raises(
        operation.GatewayMinerSubmissionsDisableError,
        match="could not be verified",
    ):
        _apply(client)

    candidate_id = str(client.put_calls[0]["ClientRequestToken"])
    assert client.current == INITIAL_VERSION
    assert _custom_labels(client) == set()
    assert candidate_id not in operation._version_stages(client)
    assert operation._version_stages(client) == original_topology


def test_corrupt_promoted_readback_rolls_current_label_back_to_prior_version():
    class CorruptFinalClient(FakeSecretsClient):
        corrupt_final = False

        def update_secret_version_stage(self, **kwargs):
            result = super().update_secret_version_stage(**kwargs)
            if kwargs.get("VersionStage") == "AWSCURRENT" and (
                kwargs.get("RemoveFromVersionId") == INITIAL_VERSION
            ):
                self.corrupt_final = True
            return result

        def get_secret_value(self, **kwargs):
            response = super().get_secret_value(**kwargs)
            if self.corrupt_final and kwargs.get("VersionStage") == "AWSCURRENT":
                self.corrupt_final = False
                response["SecretString"] += "CORRUPTED=value\n"
            return response

    original = "UNCHANGED=value\n"
    client = CorruptFinalClient(original)
    client.add_version(PREVIOUS_VERSION, {"AWSPREVIOUS"})
    client.add_version(PENDING_VERSION, {"AWSPENDING"})
    original_topology = operation._version_stages(client)

    with pytest.raises(
        operation.GatewayMinerSubmissionsDisableError,
        match="promoted gateway secret failed exact readback",
    ):
        _apply(client)

    assert client.current == INITIAL_VERSION
    assert client.versions[INITIAL_VERSION] == original
    assert operation._version_stages(client) == original_topology
    assert _custom_labels(client) == set()
    assert any(
        call["VersionStage"] == "AWSCURRENT"
        and call["MoveToVersionId"] == INITIAL_VERSION
        for call in client.stage_calls
    )


def test_rollback_restores_original_absence_of_previous_label():
    class CorruptFinalClient(FakeSecretsClient):
        corrupt_final = False

        def update_secret_version_stage(self, **kwargs):
            result = super().update_secret_version_stage(**kwargs)
            if kwargs.get("VersionStage") == "AWSCURRENT" and (
                kwargs.get("RemoveFromVersionId") == INITIAL_VERSION
            ):
                self.corrupt_final = True
            return result

        def get_secret_value(self, **kwargs):
            response = super().get_secret_value(**kwargs)
            if self.corrupt_final and kwargs.get("VersionStage") == "AWSCURRENT":
                self.corrupt_final = False
                response["SecretString"] += "CORRUPTED=value\n"
            return response

    client = CorruptFinalClient("UNCHANGED=value\n", omit_unlabeled=True)
    client.add_version(PENDING_VERSION, {"AWSPENDING"})
    original_topology = operation._version_stages(client)
    assert operation._stage_holders(original_topology, "AWSPREVIOUS") == []

    with pytest.raises(
        operation.GatewayMinerSubmissionsDisableError,
        match="promoted gateway secret failed exact readback",
    ):
        _apply(client)

    restored_topology = operation._version_stages(client)
    assert restored_topology == original_topology
    assert operation._stage_holders(restored_topology, "AWSPREVIOUS") == []
    assert client.stages[PENDING_VERSION] == {"AWSPENDING"}
    assert _custom_labels(client) == set()


def test_ambiguous_rollback_response_restores_full_preexisting_topology():
    class AmbiguousRollbackClient(FakeSecretsClient):
        corrupt_final = False
        rollback_response_lost = False

        def update_secret_version_stage(self, **kwargs):
            is_promotion = (
                kwargs.get("VersionStage") == "AWSCURRENT"
                and kwargs.get("RemoveFromVersionId") == INITIAL_VERSION
            )
            is_rollback = (
                kwargs.get("VersionStage") == "AWSCURRENT"
                and kwargs.get("MoveToVersionId") == INITIAL_VERSION
                and kwargs.get("RemoveFromVersionId") != INITIAL_VERSION
            )
            result = super().update_secret_version_stage(**kwargs)
            if is_promotion:
                self.corrupt_final = True
            if is_rollback and not self.rollback_response_lost:
                self.rollback_response_lost = True
                raise RuntimeError("rollback response lost")
            return result

        def get_secret_value(self, **kwargs):
            response = super().get_secret_value(**kwargs)
            if self.corrupt_final and kwargs.get("VersionStage") == "AWSCURRENT":
                self.corrupt_final = False
                response["SecretString"] += "CORRUPTED=value\n"
            return response

    client = AmbiguousRollbackClient("UNCHANGED=value\n", omit_unlabeled=True)
    client.add_version(PREVIOUS_VERSION, {"AWSPREVIOUS"})
    client.add_version(PENDING_VERSION, {"AWSPENDING"})
    original_topology = operation._version_stages(client)

    with pytest.raises(
        operation.GatewayMinerSubmissionsDisableError,
        match="promoted gateway secret failed exact readback",
    ):
        _apply(client)

    assert client.rollback_response_lost is True
    assert operation._version_stages(client) == original_topology
    assert _custom_labels(client) == set()


def test_preexisting_custom_stage_collision_fails_without_removing_it():
    collision = operation._CUSTOM_STAGE_PREFIX + ("a" * 32)
    client = FakeSecretsClient("UNCHANGED=value\n")
    client.add_version(PENDING_VERSION, {collision})
    original_topology = operation._version_stages(client)

    with pytest.raises(
        operation.GatewayMinerSubmissionsDisableError,
        match="another fixed-purpose miner disable operation is staged",
    ):
        _apply(client)

    assert operation._version_stages(client) == original_topology
    assert client.put_calls == []
    assert client.stage_calls == []


def test_rollback_failure_is_redacted_and_removes_only_custom_stage(
    monkeypatch,
    capsys,
):
    class FailedRollbackClient(FakeSecretsClient):
        corrupt_final = False

        def update_secret_version_stage(self, **kwargs):
            is_promotion = (
                kwargs.get("VersionStage") == "AWSCURRENT"
                and kwargs.get("RemoveFromVersionId") == INITIAL_VERSION
            )
            is_rollback = (
                kwargs.get("VersionStage") == "AWSCURRENT"
                and kwargs.get("MoveToVersionId") == INITIAL_VERSION
                and kwargs.get("RemoveFromVersionId") != INITIAL_VERSION
            )
            if is_rollback:
                raise RuntimeError("SENSITIVE_ROLLBACK_BACKEND_DETAIL")
            result = super().update_secret_version_stage(**kwargs)
            if is_promotion:
                self.corrupt_final = True
            return result

        def get_secret_value(self, **kwargs):
            response = super().get_secret_value(**kwargs)
            if self.corrupt_final and kwargs.get("VersionStage") == "AWSCURRENT":
                self.corrupt_final = False
                response["SecretString"] += "CORRUPTED=value\n"
            return response

    client = FailedRollbackClient("UNCHANGED=value\n")
    client.add_version(PENDING_VERSION, {"AWSPENDING"})
    monkeypatch.setattr(operation, "_verify_protected_source", lambda: None)
    monkeypatch.setattr(operation, "_instance_role_secrets_client", lambda: client)

    assert (
        operation.main(["--apply", "--expected-current-version-id", INITIAL_VERSION])
        == 2
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "SENSITIVE_ROLLBACK_BACKEND_DETAIL" not in captured.err
    assert '"status":"failed_closed"' in captured.err
    assert _custom_labels(client) == set()
    assert client.stages[PENDING_VERSION] == {"AWSPENDING"}


class _FakeCredentials:
    def __init__(self, method):
        self.method = method


class _FakeSts:
    def get_caller_identity(self):
        return {
            "Account": operation.EXPECTED_AWS_ACCOUNT_ID,
            "Arn": (
                "arn:aws:sts::493765492819:assumed-role/"
                "leadpoet-gateway-s3-cloudwatch-role/i-0123456789abcdef0"
            ),
        }


class _FakeSession:
    def __init__(self, *, credential_method="iam-role", **_kwargs):
        self.credentials = _FakeCredentials(credential_method)
        self.secrets_client = object()

    def get_credentials(self):
        return self.credentials

    def client(self, service):
        return _FakeSts() if service == "sts" else self.secrets_client


def test_instance_role_client_rejects_static_credential_environment():
    with pytest.raises(
        operation.GatewayMinerSubmissionsDisableError,
        match="credential configuration is forbidden",
    ):
        operation._instance_role_secrets_client(
            environ={
                "LEADPOET_AWS_INSTANCE_ROLE_ONLY": "true",
                "AWS_ACCESS_KEY_ID": "must-not-be-used",
            },
            session_factory=_FakeSession,
        )


def test_instance_role_client_rejects_non_instance_credential_resolution():
    def factory(**kwargs):
        return _FakeSession(credential_method="shared-credentials-file", **kwargs)

    with pytest.raises(
        operation.GatewayMinerSubmissionsDisableError,
        match="instance-role credentials are unavailable",
    ):
        operation._instance_role_secrets_client(
            environ={"LEADPOET_AWS_INSTANCE_ROLE_ONLY": "true"},
            session_factory=factory,
        )


def test_instance_role_client_accepts_only_explicit_ec2_role_mode():
    session = _FakeSession()

    result = operation._instance_role_secrets_client(
        environ={"LEADPOET_AWS_INSTANCE_ROLE_ONLY": "true"},
        session_factory=lambda **_kwargs: session,
    )

    assert result is session.secrets_client


def test_instance_role_client_rejects_another_ec2_role_in_same_account():
    class WrongRoleSts:
        def get_caller_identity(self):
            return {
                "Account": operation.EXPECTED_AWS_ACCOUNT_ID,
                "Arn": (
                    "arn:aws:sts::493765492819:assumed-role/"
                    "unrelated-role/i-0123456789abcdef0"
                ),
            }

    class WrongRoleSession(_FakeSession):
        def client(self, service):
            return WrongRoleSts() if service == "sts" else self.secrets_client

    with pytest.raises(
        operation.GatewayMinerSubmissionsDisableError,
        match="instance-role identity differs",
    ):
        operation._instance_role_secrets_client(
            environ={"LEADPOET_AWS_INSTANCE_ROLE_ONLY": "true"},
            session_factory=WrongRoleSession,
        )


def test_fixed_operation_source_binding_matches_protected_manifest():
    operation._verify_protected_source()


def test_main_redacts_unexpected_backend_exception(monkeypatch, capsys):
    class ExplodingClient:
        def get_secret_value(self, **_kwargs):
            raise RuntimeError("UNREDACTED_SECRET_VALUE")

    monkeypatch.setattr(
        operation,
        "_instance_role_secrets_client",
        lambda: ExplodingClient(),
    )

    assert operation.main([]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "UNREDACTED_SECRET_VALUE" not in captured.err
    assert '"status":"failed_closed"' in captured.err
