import copy
import json
import subprocess

import pytest

from gateway.tee.release_channel_v2 import (
    DEFAULT_PREFIX,
    MAX_LINEAGE_RELEASES,
    ReleaseChannelV2Error,
    build_release_channel_v2,
    build_release_lineage_v2,
    cli,
    fetch_release_channel_v2,
    fetch_release_lineage_v2,
    install_release_channel_v2,
    publish_release_channel_v2,
    release_channel_key,
    validate_historical_release_channel_v2,
    validate_prior_release_channel_v2,
    validate_release_channel_v2,
)
from gateway.tee.prepare_active_release_lineage_v2 import (
    PrepareActiveReleaseLineageV2Error,
    _fetch_exact_release_lineage_v2,
)
from gateway.tee.release_manifest_v2 import (
    BUILD_EVIDENCE_SCHEMA_VERSION,
    HISTORICAL_THREE_ROLE_TOPOLOGY_HASH,
    build_release_manifest,
)
from gateway.tee.topology import ROLE_SPECS, topology_hash
from leadpoet_canonical.attested_v2 import sha256_json
from validator_tee.host.release_v2 import (
    build_validator_build_evidence,
    build_validator_release,
    build_validator_release_manifest,
)


COMMIT = "1" * 40


def _hash(character):
    return "sha256:" + character * 64


def _gateway_manifest(commit=COMMIT):
    rows = []
    for index, (role, spec) in enumerate(sorted(ROLE_SPECS.items())):
        character = "abcdef0123456789"[index]
        deterministic = {
            "commit_sha": commit,
            "pcr0": character * 96,
            "normalized_image_hash": _hash(character),
            "eif_hash": _hash(character),
            "source_manifest_hash": _hash("2"),
            "build_identity_hash": _hash(character),
            "execution_manifest_hash": _hash(character),
            "dependency_lock_hash": _hash("3"),
            "dockerfile_hash": _hash("4"),
            "topology_hash": topology_hash(),
        }
        for domain in ("gateway", "validator"):
            for ordinal in (1, 2, 3):
                rows.append(
                    {
                        "schema_version": BUILD_EVIDENCE_SCHEMA_VERSION,
                        "builder_domain": domain,
                        "builder_id": domain + "-parent",
                        "build_ordinal": ordinal,
                        "physical_role": role,
                        "service_role": spec["service_role"],
                        **deterministic,
                    }
                )
    return build_release_manifest(rows, acceptance_signer_pubkey_hash=_hash("f"))


def _validator_manifest(commit=COMMIT):
    release = build_validator_release(
        commit_sha=commit,
        pcr0="2" * 96,
        app_manifest_hash=_hash("3"),
        dependency_lock_hash=_hash("4"),
        normalized_image_hash=_hash("5"),
        eif_hash=_hash("6"),
        dockerfile_hash=_hash("7"),
        base_dockerfile_hash=_hash("8"),
    )
    evidence = [
        build_validator_build_evidence(
            release,
            builder_domain=domain,
            builder_id=domain + "-parent",
            build_ordinal=ordinal,
        )
        for domain in ("gateway", "validator")
        for ordinal in (1, 2, 3)
    ]
    return build_validator_release_manifest(evidence)


def _historical_gateway_manifest(commit):
    current = _gateway_manifest(commit)
    roles = copy.deepcopy(current["roles"])
    for summary in roles.values():
        summary["topology_hash"] = HISTORICAL_THREE_ROLE_TOPOLOGY_HASH
    autoresearch = copy.deepcopy(roles["gateway_scoring"])
    autoresearch.update(
        {
            "physical_role": "gateway_autoresearch",
            "service_role": "gateway_autoresearch",
            "topology_hash": HISTORICAL_THREE_ROLE_TOPOLOGY_HASH,
        }
    )
    roles["gateway_autoresearch"] = autoresearch
    body = {
        **{
            key: value
            for key, value in current.items()
            if key != "release_hash"
        },
        "topology_hash": HISTORICAL_THREE_ROLE_TOPOLOGY_HASH,
        "roles": roles,
        "verified_build_count": 18,
    }
    return {**body, "release_hash": sha256_json(body)}


def _historical_channel(commit):
    body = {
        "schema_version": "leadpoet.attested_release_channel.v2",
        "commit_sha": commit,
        "gateway_release_manifest": _historical_gateway_manifest(commit),
        "validator_release_manifest": _validator_manifest(commit),
    }
    return {**body, "channel_hash": sha256_json(body)}


class _Body:
    def __init__(self, value):
        self._value = value

    def read(self):
        return self._value


class _S3:
    def __init__(self):
        self.objects = {}
        self.gets = []
        self.lists = []
        self.puts = []

    def get_object(self, *, Bucket, Key):
        self.gets.append((Bucket, Key))
        if (Bucket, Key) not in self.objects:
            raise KeyError(Key)
        return {"Body": _Body(self.objects[(Bucket, Key)])}

    def put_object(self, **kwargs):
        key = (kwargs["Bucket"], kwargs["Key"])
        if key in self.objects:
            raise RuntimeError("precondition failed")
        self.objects[key] = kwargs["Body"]
        self.puts.append(kwargs)

    def list_objects_v2(self, *, Bucket, Prefix, MaxKeys, **kwargs):
        self.lists.append(
            {
                "Bucket": Bucket,
                "Prefix": Prefix,
                "MaxKeys": MaxKeys,
                **kwargs,
            }
        )
        del MaxKeys, kwargs
        keys = sorted(
            key
            for bucket, key in self.objects
            if bucket == Bucket and key.startswith(Prefix)
        )
        return {
            "Contents": [{"Key": key} for key in keys],
            "IsTruncated": False,
        }


def test_channel_binds_both_independent_release_manifests():
    value = build_release_channel_v2(
        gateway_release_manifest=_gateway_manifest(),
        validator_release_manifest=_validator_manifest(),
    )
    assert validate_release_channel_v2(value, expected_commit=COMMIT) == value
    tampered = copy.deepcopy(value)
    tampered["commit_sha"] = "2" * 40
    with pytest.raises(ReleaseChannelV2Error, match="commit"):
        validate_release_channel_v2(tampered)


def test_historical_channel_is_valid_only_for_prior_lineage():
    historical = _historical_channel("2" * 40)

    assert validate_historical_release_channel_v2(historical) == historical
    assert validate_prior_release_channel_v2(historical) == historical
    with pytest.raises(Exception):
        validate_release_channel_v2(historical)
    with pytest.raises(Exception):
        build_release_channel_v2(
            gateway_release_manifest=historical[
                "gateway_release_manifest"
            ],
            validator_release_manifest=historical[
                "validator_release_manifest"
            ],
        )


def test_required_lineage_accepts_exact_legacy_prior_but_not_as_current():
    historical_commit = "2" * 40
    current = build_release_channel_v2(
        gateway_release_manifest=_gateway_manifest(),
        validator_release_manifest=_validator_manifest(),
    )
    historical = _historical_channel(historical_commit)
    s3 = _S3()
    for channel in (current, historical):
        s3.objects[("release-bucket", release_channel_key(channel["commit_sha"]))] = (
            json.dumps(channel, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode()

    lineage = fetch_release_lineage_v2(
        bucket="release-bucket",
        current_commit=COMMIT,
        s3_client=s3,
        allowed_commits=(historical_commit, COMMIT),
        required_commits=(historical_commit, COMMIT),
    )

    assert set(lineage["releases"][COMMIT]["roles"]) == {
        *ROLE_SPECS,
        "validator_weights",
    }
    assert set(lineage["releases"][historical_commit]["roles"]) == {
        *ROLE_SPECS,
        "gateway_autoresearch",
        "validator_weights",
    }
    with pytest.raises(Exception):
        fetch_release_lineage_v2(
            bucket="release-bucket",
            current_commit=historical_commit,
            s3_client=s3,
            allowed_commits=(historical_commit,),
            required_commits=(historical_commit,),
        )


def test_channel_rejects_cross_commit_manifests():
    validator = _validator_manifest()
    validator["release"]["commit_sha"] = "2" * 40
    with pytest.raises(Exception):
        build_release_channel_v2(
            gateway_release_manifest=_gateway_manifest(),
            validator_release_manifest=validator,
        )


def test_channel_publish_is_immutable_and_fetch_installs_atomically(tmp_path):
    channel = build_release_channel_v2(
        gateway_release_manifest=_gateway_manifest(),
        validator_release_manifest=_validator_manifest(),
    )
    s3 = _S3()
    published = publish_release_channel_v2(
        channel, bucket="release-bucket", s3_client=s3
    )
    assert published["commit_sha"] == COMMIT
    assert s3.puts[0]["IfNoneMatch"] == "*"
    assert s3.puts[0]["ObjectLockMode"] == "COMPLIANCE"

    fetched = fetch_release_channel_v2(
        bucket="release-bucket", commit_sha=COMMIT, s3_client=s3
    )
    gateway_output = tmp_path / "gateway.json"
    validator_output = tmp_path / "validator.json"
    install_release_channel_v2(
        fetched,
        expected_commit=COMMIT,
        gateway_output=gateway_output,
        validator_output=validator_output,
    )
    assert json.loads(gateway_output.read_text())["commit_sha"] == COMMIT
    assert (
        json.loads(validator_output.read_text())["release"]["commit_sha"]
        == COMMIT
    )
    assert gateway_output.stat().st_mode & 0o777 == 0o600


def test_candidate_install_does_not_replace_running_release(tmp_path):
    running_commit = "2" * 40
    running_manifest = _gateway_manifest(running_commit)
    active_output = tmp_path / "active-gateway.json"
    active_output.write_text(
        json.dumps(running_manifest, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="ascii",
    )
    active_before = active_output.read_bytes()

    candidate = build_release_channel_v2(
        gateway_release_manifest=_gateway_manifest(),
        validator_release_manifest=_validator_manifest(),
    )
    candidate_output = tmp_path / "restart" / "candidate-gateway.json"
    install_release_channel_v2(
        candidate,
        expected_commit=COMMIT,
        gateway_output=candidate_output,
    )

    assert active_output.read_bytes() == active_before
    assert json.loads(active_output.read_text())["commit_sha"] == running_commit
    assert json.loads(candidate_output.read_text())["commit_sha"] == COMMIT


def test_release_channel_key_is_content_addressed_by_commit():
    assert release_channel_key(COMMIT).endswith(
        f"/{COMMIT}/release-channel-v2.json"
    )


def test_release_lineage_binds_historical_exact_role_measurements():
    historical_commit = "2" * 40
    current = build_release_channel_v2(
        gateway_release_manifest=_gateway_manifest(),
        validator_release_manifest=_validator_manifest(),
    )
    historical = build_release_channel_v2(
        gateway_release_manifest=_gateway_manifest(historical_commit),
        validator_release_manifest=_validator_manifest(historical_commit),
    )
    unrelated_commit = "3" * 40
    unrelated = build_release_channel_v2(
        gateway_release_manifest=_gateway_manifest(unrelated_commit),
        validator_release_manifest=_validator_manifest(unrelated_commit),
    )
    lineage = build_release_lineage_v2(
        [historical, current],
        current_commit=COMMIT,
    )
    assert lineage["current_gateway_release_hash"] == (
        current["gateway_release_manifest"]["release_hash"]
    )
    expected = lineage["releases"][historical_commit]["roles"][
        "gateway_coordinator"
    ]
    summary = historical["gateway_release_manifest"]["roles"][
        "gateway_coordinator"
    ]
    assert expected == {
        "commit_sha": historical_commit,
        "pcr0": summary["pcr0"],
        "build_manifest_hash": summary["execution_manifest_hash"],
        "dependency_lock_hash": summary["dependency_lock_hash"],
    }
    validator_expected = lineage["releases"][historical_commit]["roles"][
        "validator_weights"
    ]
    validator_summary = historical["validator_release_manifest"]["release"]
    assert validator_expected == {
        "commit_sha": historical_commit,
        "pcr0": validator_summary["pcr0"],
        "build_manifest_hash": validator_summary["app_manifest_hash"],
        "dependency_lock_hash": validator_summary["dependency_lock_hash"],
    }

    s3 = _S3()
    for channel in (historical, current, unrelated):
        key = release_channel_key(channel["commit_sha"])
        s3.objects[("release-bucket", key)] = (
            json.dumps(channel, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode()
    assert fetch_release_lineage_v2(
        bucket="release-bucket",
        current_commit=COMMIT,
        s3_client=s3,
        allowed_commits=(historical_commit, COMMIT),
    ) == lineage


def test_release_lineage_rejects_missing_current_channel():
    historical_commit = "2" * 40
    historical = build_release_channel_v2(
        gateway_release_manifest=_gateway_manifest(historical_commit),
        validator_release_manifest=_validator_manifest(historical_commit),
    )
    with pytest.raises(ReleaseChannelV2Error, match="current release"):
        build_release_lineage_v2([historical], current_commit=COMMIT)


def test_release_lineage_ignores_unrelated_channels_before_size_bound():
    current = build_release_channel_v2(
        gateway_release_manifest=_gateway_manifest(),
        validator_release_manifest=_validator_manifest(),
    )
    s3 = _S3()
    s3.objects[("release-bucket", release_channel_key(COMMIT))] = (
        json.dumps(current, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    for index in range(513):
        unrelated_commit = f"{index + 2:040x}"
        s3.objects[
            ("release-bucket", release_channel_key(unrelated_commit))
        ] = b"untrusted branch release"

    lineage = fetch_release_lineage_v2(
        bucket="release-bucket",
        current_commit=COMMIT,
        s3_client=s3,
        allowed_commits=(COMMIT,),
    )

    assert tuple(lineage["releases"]) == (COMMIT,)


def test_required_release_lineage_direct_gets_only_explicit_commits():
    historical_commit = "2" * 40
    current = build_release_channel_v2(
        gateway_release_manifest=_gateway_manifest(),
        validator_release_manifest=_validator_manifest(),
    )
    historical = build_release_channel_v2(
        gateway_release_manifest=_gateway_manifest(historical_commit),
        validator_release_manifest=_validator_manifest(historical_commit),
    )
    s3 = _S3()
    for channel in (current, historical):
        key = release_channel_key(channel["commit_sha"])
        s3.objects[("release-bucket", key)] = (
            json.dumps(channel, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode()
    for index in range(MAX_LINEAGE_RELEASES + 1):
        unrelated_commit = f"{index + 3:040x}"
        s3.objects[
            ("release-bucket", release_channel_key(unrelated_commit))
        ] = b"untrusted branch release"

    lineage = fetch_release_lineage_v2(
        bucket="release-bucket",
        current_commit=COMMIT,
        s3_client=s3,
        allowed_commits=(historical_commit, COMMIT),
        required_commits=(historical_commit, COMMIT),
    )

    assert lineage == build_release_lineage_v2(
        [historical, current],
        current_commit=COMMIT,
    )
    assert s3.lists == []
    assert s3.gets == [
        ("release-bucket", release_channel_key(COMMIT)),
        ("release-bucket", release_channel_key(historical_commit)),
    ]


def test_required_release_lineage_uses_local_current_release_without_approved_fetch(
    monkeypatch, tmp_path
):
    gateway_path = tmp_path / "gateway-release.json"
    validator_path = tmp_path / "validator-release.json"
    gateway_path.write_text(json.dumps(_gateway_manifest()), encoding="utf-8")
    validator_path.write_text(json.dumps(_validator_manifest()), encoding="utf-8")
    monkeypatch.setenv("LEADPOET_LOCAL_RELEASE_COMMIT_SHA", COMMIT)
    monkeypatch.setenv("LEADPOET_LOCAL_GATEWAY_RELEASE", str(gateway_path))
    monkeypatch.setenv("LEADPOET_LOCAL_VALIDATOR_RELEASE", str(validator_path))
    s3 = _S3()

    lineage = fetch_release_lineage_v2(
        bucket="release-bucket",
        current_commit=COMMIT,
        s3_client=s3,
        allowed_commits=(COMMIT,),
        required_commits=(COMMIT,),
    )

    expected = build_release_channel_v2(
        gateway_release_manifest=_gateway_manifest(),
        validator_release_manifest=_validator_manifest(),
    )
    assert lineage == build_release_lineage_v2([expected], current_commit=COMMIT)
    assert s3.gets == []
    assert s3.lists == []


def test_required_release_lineage_rejects_more_than_bound_before_io():
    required = (COMMIT,) + tuple(
        f"{index:040x}" for index in range(MAX_LINEAGE_RELEASES)
    )
    s3 = _S3()

    with pytest.raises(ReleaseChannelV2Error, match="size"):
        fetch_release_lineage_v2(
            bucket="release-bucket",
            current_commit=COMMIT,
            s3_client=s3,
            required_commits=required,
        )

    assert s3.gets == []
    assert s3.lists == []


def test_required_release_lineage_rejects_missing_current_before_io():
    s3 = _S3()

    with pytest.raises(ReleaseChannelV2Error, match="current release"):
        fetch_release_lineage_v2(
            bucket="release-bucket",
            current_commit=COMMIT,
            s3_client=s3,
            required_commits=("2" * 40,),
        )

    assert s3.gets == []
    assert s3.lists == []


def test_required_release_lineage_rejects_duplicate_before_io():
    s3 = _S3()

    with pytest.raises(ReleaseChannelV2Error, match="duplicated"):
        fetch_release_lineage_v2(
            bucket="release-bucket",
            current_commit=COMMIT,
            s3_client=s3,
            required_commits=(COMMIT, COMMIT),
        )

    assert s3.gets == []
    assert s3.lists == []


@pytest.mark.parametrize("allowed_commits", (None, (), COMMIT))
def test_required_release_lineage_requires_nonempty_allowed_ancestry_before_io(
    allowed_commits,
):
    s3 = _S3()

    with pytest.raises(ReleaseChannelV2Error, match="Git ancestry"):
        fetch_release_lineage_v2(
            bucket="release-bucket",
            current_commit=COMMIT,
            s3_client=s3,
            allowed_commits=allowed_commits,
            required_commits=(COMMIT,),
        )

    assert s3.gets == []
    assert s3.lists == []


@pytest.mark.parametrize(
    "invalid_commit",
    (
        "A" * 40,
        "a" * 39,
        " " + "a" * 40,
        None,
    ),
)
def test_required_release_lineage_rejects_noncanonical_commit_before_io(
    invalid_commit,
):
    s3 = _S3()

    with pytest.raises(ReleaseChannelV2Error, match="commits are invalid"):
        fetch_release_lineage_v2(
            bucket="release-bucket",
            current_commit=COMMIT,
            s3_client=s3,
            required_commits=(COMMIT, invalid_commit),
        )

    assert s3.gets == []
    assert s3.lists == []


def test_required_release_lineage_rejects_nonancestor_before_io():
    s3 = _S3()

    with pytest.raises(ReleaseChannelV2Error, match="Git ancestry"):
        fetch_release_lineage_v2(
            bucket="release-bucket",
            current_commit=COMMIT,
            s3_client=s3,
            allowed_commits=(COMMIT,),
            required_commits=(COMMIT, "2" * 40),
        )

    assert s3.gets == []
    assert s3.lists == []


def test_required_release_lineage_rejects_missing_channel():
    current = build_release_channel_v2(
        gateway_release_manifest=_gateway_manifest(),
        validator_release_manifest=_validator_manifest(),
    )
    s3 = _S3()
    s3.objects[("release-bucket", release_channel_key(COMMIT))] = (
        json.dumps(current, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()

    with pytest.raises(ReleaseChannelV2Error, match="channel is unavailable"):
        fetch_release_lineage_v2(
            bucket="release-bucket",
            current_commit=COMMIT,
            s3_client=s3,
            allowed_commits=(COMMIT, "2" * 40),
            required_commits=(COMMIT, "2" * 40),
        )

    assert s3.lists == []


def test_required_release_lineage_rejects_tampered_channel():
    current = build_release_channel_v2(
        gateway_release_manifest=_gateway_manifest(),
        validator_release_manifest=_validator_manifest(),
    )
    current["channel_hash"] = _hash("0")
    s3 = _S3()
    s3.objects[("release-bucket", release_channel_key(COMMIT))] = (
        json.dumps(current, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()

    with pytest.raises(ReleaseChannelV2Error, match="hash differs"):
        fetch_release_lineage_v2(
            bucket="release-bucket",
            current_commit=COMMIT,
            s3_client=s3,
            allowed_commits=(COMMIT,),
            required_commits=(COMMIT,),
        )

    assert s3.lists == []


def test_cli_forwards_explicit_required_release_commits(
    monkeypatch,
    tmp_path,
    capsys,
):
    historical_commit = "2" * 40
    observed = {}

    monkeypatch.setattr(
        "gateway.tee.release_channel_v2.local_release_inputs_match",
        lambda **_kwargs: True,
    )
    monkeypatch.setattr(
        "gateway.tee.release_channel_v2.git_ancestor_commits_v2",
        lambda **_kwargs: (COMMIT, historical_commit),
    )

    def fetch_lineage(**kwargs):
        observed.update(kwargs)
        return {
            "lineage_hash": _hash("9"),
            "releases": {COMMIT: {}, historical_commit: {}},
        }

    monkeypatch.setattr(
        "gateway.tee.release_channel_v2.fetch_release_lineage_v2",
        fetch_lineage,
    )

    result = cli(
        [
            "--ensure",
            "--expected-commit",
            COMMIT,
            "--lineage-output",
            str(tmp_path / "lineage.json"),
            "--lineage-repository",
            str(tmp_path),
            "--lineage-authority-commit",
            COMMIT,
            "--lineage-required-commit",
            COMMIT,
            "--lineage-required-commit",
            historical_commit,
        ]
    )

    assert result == 0
    assert observed["current_commit"] == COMMIT
    assert observed["allowed_commits"] == (COMMIT, historical_commit)
    assert observed["required_commits"] == [COMMIT, historical_commit]
    assert json.loads(capsys.readouterr().out)["lineage_release_count"] == 2


def test_cli_reports_unpublished_channel_without_traceback(monkeypatch, capsys):
    def _unavailable(**_kwargs):
        raise ReleaseChannelV2Error("approved release channel is unavailable")

    monkeypatch.setattr(
        "gateway.tee.release_channel_v2.fetch_release_channel_v2",
        _unavailable,
    )

    result = cli(["--ensure", "--expected-commit", COMMIT])

    captured = capsys.readouterr()
    assert result == 75
    assert captured.out == ""
    assert captured.err == (
        "Release channel unavailable: "
        "approved release channel is unavailable\n"
    )
    assert "Traceback" not in captured.err


def _real_three_commit_dag(repository):
    repository.mkdir()
    subprocess.run(
        ["git", "init", "--initial-branch=main", str(repository)],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.name", "Release Test"],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "config",
            "user.email",
            "release-test@example.invalid",
        ],
        check=True,
    )
    commits = []
    for index in range(3):
        (repository / "state.txt").write_text(str(index), encoding="ascii")
        subprocess.run(
            ["git", "-C", str(repository), "add", "state.txt"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(repository), "commit", "-m", f"state {index}"],
            check=True,
            capture_output=True,
        )
        commits.append(
            subprocess.run(
                ["git", "-C", str(repository), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    return tuple(commits)


@pytest.mark.parametrize(
    ("target_index", "required_indexes"),
    (
        (1, (0, 1)),
        (0, (0, 1)),
        (2, (0, 2)),
    ),
    ids=("forward", "rollback", "roll-forward"),
)
def test_exact_release_selection_real_dag_ignores_lifetime_catalog(
    tmp_path,
    target_index,
    required_indexes,
):
    repository = tmp_path / "release-dag"
    commits = _real_three_commit_dag(repository)
    channels = {
        commit: build_release_channel_v2(
            gateway_release_manifest=_gateway_manifest(commit),
            validator_release_manifest=_validator_manifest(commit),
        )
        for commit in commits
    }
    s3 = _S3()
    for commit, channel in channels.items():
        s3.objects[("release-bucket", release_channel_key(commit))] = (
            json.dumps(channel, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("ascii")
    for index in range(MAX_LINEAGE_RELEASES + 1):
        decoy = f"{index + 1000:040x}"
        if decoy not in channels:
            s3.objects[("release-bucket", release_channel_key(decoy))] = (
                b"untrusted lifetime catalog entry"
            )

    target = commits[target_index]
    authority = commits[2]
    required = sorted(commits[index] for index in required_indexes)
    lineage = _fetch_exact_release_lineage_v2(
        candidate_commit_sha=target,
        authority_commit_sha=authority,
        required_commits=required,
        repository=repository,
        bucket="release-bucket",
        prefix=DEFAULT_PREFIX,
        s3_client=s3,
    )

    assert lineage["current_commit_sha"] == target
    assert sorted(lineage["releases"]) == required
    assert s3.lists == []
    assert sorted(s3.gets) == sorted(
        ("release-bucket", release_channel_key(commit)) for commit in required
    )


def test_exact_release_selection_does_not_substitute_target_for_authority(
    tmp_path,
) -> None:
    repository = tmp_path / "release-dag"
    oldest, newer, _newest = _real_three_commit_dag(repository)
    s3 = _S3()

    with pytest.raises(
        PrepareActiveReleaseLineageV2Error,
        match="outside release authority Git ancestry",
    ):
        _fetch_exact_release_lineage_v2(
            candidate_commit_sha=oldest,
            authority_commit_sha=oldest,
            required_commits=sorted((oldest, newer)),
            repository=repository,
            bucket="release-bucket",
            prefix=DEFAULT_PREFIX,
            s3_client=s3,
        )

    assert s3.gets == []
    assert s3.lists == []
