from __future__ import annotations

import ast
import copy
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from urllib.request import ProxyHandler

import pytest

from scripts import probe_weight_submission_evidence_v2 as probe

# Imported only after the probe itself. Production imports these lazily after
# the stdlib-only repository preflight.
from leadpoet_canonical import chain_source_v2 as chain_api
from leadpoet_canonical.compact_auditor_authority_v2 import (
    COMPACT_PUBLISHED_WEIGHT_AUTHORITY_SCHEMA_VERSION,
)


CANDIDATE = "a" * 40
PRIMARY = "5" + "A" * 47
AUDITOR_ONE = "5" + "B" * 47
AUDITOR_TWO = "5" + "C" * 47
OTHER = "5" + "D" * 47
EPOCH_ID = 24_080
TARGET_SUBNET_EPOCH_INDEX = 24_027
EXPECTED_VECTOR = [[2, 10_000], [7, 55_535]]
RELEASE_CONTRACT = {
    "roles": dict(probe._EXPECTED_RELEASE_IDENTITY_ROLES),
    "build_counts": {role: 6 for role in probe._EXPECTED_RELEASE_IDENTITY_ROLES},
}
PROFILE = {
    "mechid": 0,
    "genesis_hash": "1" * 64,
    "chain_endpoint": "wss://entrypoint-finney.opentensor.ai:443",
    "subnet_reveal_period_epochs": 1,
}


class _Cutover:
    netuid = 71
    network_genesis_hash = "0x" + "1" * 64
    mapping_hash = "sha256:" + "2" * 64

    @staticmethod
    def settlement_epoch_id(index):
        assert index == TARGET_SUBNET_EPOCH_INDEX
        return EPOCH_ID


def _runtime():
    return SimpleNamespace(
        identity_cache_schema=probe.IDENTITY_CACHE_SCHEMA_VERSION,
        authority_schema=COMPACT_PUBLISHED_WEIGHT_AUTHORITY_SCHEMA_VERSION,
        chain=chain_api,
        derive_ancestry_lineage_id_v2=lambda **_kwargs: "sha256:" + "3" * 64,
        load_immutable_release_identity=None,
        verify_authority=None,
    )


def _release_identity_cache():
    entries = []
    for index, (physical_role, service_role) in enumerate(
        sorted(probe._EXPECTED_RELEASE_IDENTITY_ROLES.items()), start=1
    ):
        entries.append(
            {
                "physical_role": physical_role,
                "role": service_role,
                "commit_sha": CANDIDATE,
                "pcr0": "%096x" % index,
                "build_manifest_hash": "sha256:" + "%064x" % index,
                "dependency_lock_hash": "sha256:" + "%064x" % (index + 10),
                "verified_build_count": 6,
            }
        )
    return {
        "schema_version": probe.IDENTITY_CACHE_SCHEMA_VERSION,
        "entries": entries,
    }


def _authority(*, stage="finalized"):
    return {
        "schema_version": COMPACT_PUBLISHED_WEIGHT_AUTHORITY_SCHEMA_VERSION,
        "authority_stage": stage,
        "finalization": (
            {
                "compact_submission": {
                    "finalization": {
                        "extrinsic_authorization": {
                            "epoch_id": EPOCH_ID,
                            "netuid": 71,
                            "subnet_epoch_index": TARGET_SUBNET_EPOCH_INDEX,
                        }
                    }
                }
            }
            if stage == "finalized"
            else None
        ),
    }


def _verified():
    return {
        "validator_hotkey": PRIMARY,
        "netuid": 71,
        "epoch_id": EPOCH_ID,
        "block": 100,
        "uids": [2, 7],
        "weights_u16": [10_000, 55_535],
        "weights_hash": "1" * 64,
        "authority_stage": "finalized",
        "authority_hash": "sha256:" + "2" * 64,
        "bundle_hash": "sha256:" + "3" * 64,
        "weight_finalization_event_hash": "sha256:" + "4" * 64,
        "finalized_block": 110,
    }


def _chain_state():
    hotkeys = [PRIMARY, AUDITOR_ONE, OTHER, AUDITOR_TWO]
    return {
        "block_hash": "0x" + "5" * 64,
        "block": 200,
        "subnet_epoch_index": TARGET_SUBNET_EPOCH_INDEX + 1,
        "metagraph_hotkeys": hotkeys,
        "validators": [
            {
                "hotkey": PRIMARY,
                "uid": 0,
                "mechanism_id": 0,
                "last_update": 110,
                "weights": copy.deepcopy(EXPECTED_VECTOR),
            },
            {
                "hotkey": AUDITOR_ONE,
                "uid": 1,
                "mechanism_id": 0,
                "last_update": 111,
                "weights": copy.deepcopy(EXPECTED_VECTOR),
            },
            {
                "hotkey": AUDITOR_TWO,
                "uid": 3,
                "mechanism_id": 0,
                "last_update": 112,
                "weights": copy.deepcopy(EXPECTED_VECTOR),
            },
        ],
    }


class _Http:
    def __init__(self, *, authority=None, candidate=CANDIDATE):
        self.documents = {
            "/build-info": {"is_commit_known": True, "git_commit": candidate},
            "/weights/v2/release-evidence/"
            + CANDIDATE: {
                "schema_version": "leadpoet.auditor_release_evidence.v2",
                "commit_sha": CANDIDATE,
                "release_channel_version_id": "version-1",
                "release_channel_get_url": "https://fixture.invalid/get",
                "release_channel_head_url": "https://fixture.invalid/head",
            },
            "/weights/v2/published-compact/71/24080": (
                copy.deepcopy(authority) if authority is not None else _authority()
            ),
        }
        self.calls = []

    def get_json(self, path, *, maximum_bytes):
        self.calls.append((path, maximum_bytes))
        return copy.deepcopy(self.documents[path])

    def open_exact_url(self, url, *, method):
        raise AssertionError("fixture release loader owns immutable reads")


class _Chain:
    def __init__(self, state=None):
        self.state = copy.deepcopy(state if state is not None else _chain_state())
        self.calls = []

    def read_finalized_state(self, *, netuid, hotkeys):
        self.calls.append((netuid, list(hotkeys)))
        return copy.deepcopy(self.state)


def _run(*, http=None, chain=None, release_loader=None, verifier=None):
    http = http or _Http()
    chain = chain or _Chain()
    release_calls = []
    verifier_calls = []

    def load_release(evidence, *, http_open):
        release_calls.append((copy.deepcopy(evidence), http_open))
        return _release_identity_cache()

    def verify(authority, **kwargs):
        verifier_calls.append((copy.deepcopy(authority), kwargs))
        return _verified()

    result = probe._verify_weight_submission_evidence_core(
        candidate=CANDIDATE,
        netuid=71,
        epoch_id=EPOCH_ID,
        auditors=[AUDITOR_ONE, AUDITOR_TWO],
        gateway_url=probe.DEFAULT_GATEWAY_URL,
        runtime=_runtime(),
        profile=PROFILE,
        cutover=_Cutover(),
        release_contract=RELEASE_CONTRACT,
        http=http,
        chain_reader=chain,
        release_identity_loader=release_loader or load_release,
        authority_verifier=verifier or verify,
    )
    return result, http, chain, release_calls, verifier_calls


def test_probe_verifies_exact_release_authority_and_finalized_vectors():
    result, http, chain, release_calls, verifier_calls = _run()

    assert result["candidate_sha"] == CANDIDATE
    assert result["netuid"] == 71
    assert result["epoch_id"] == EPOCH_ID
    assert result["auditor_count"] == 2
    assert result["validator_count"] == 3
    assert result["destination_count"] == 2
    assert result["finalized_head_block"] == 200
    assert result["target_subnet_epoch_index"] == TARGET_SUBNET_EPOCH_INDEX
    assert result["release_channel_version_hash"] == probe._sha256_json(
        {"version_id": "version-1"}
    )
    assert [item["uid"] for item in result["validators"]] == [0, 1, 3]
    assert {item["mechanism_id"] for item in result["validators"]} == {0}
    assert len({item["vector_hash"] for item in result["validators"]}) == 1
    serialized = json.dumps(result, sort_keys=True)
    assert PRIMARY not in serialized
    assert AUDITOR_ONE not in serialized
    assert AUDITOR_TWO not in serialized
    assert "version-1" not in serialized
    assert result["evidence_hash"] == probe._sha256_json(
        {key: value for key, value in result.items() if key != "evidence_hash"}
    )
    assert [path for path, _limit in http.calls] == [
        "/build-info",
        "/weights/v2/release-evidence/" + CANDIDATE,
        "/weights/v2/published-compact/71/24080",
    ]
    assert len(release_calls) == 1
    assert len(verifier_calls) == 1
    assert chain.calls == [(71, [PRIMARY, AUDITOR_ONE, AUDITOR_TWO])]


def test_probe_rejects_gateway_and_release_sha_mismatches():
    with pytest.raises(
        probe.WeightSubmissionEvidenceProbeError, match="candidate_sha_mismatch"
    ):
        _run(http=_Http(candidate="b" * 40))

    http = _Http()
    http.documents["/weights/v2/release-evidence/" + CANDIDATE]["commit_sha"] = "b" * 40
    with pytest.raises(
        probe.WeightSubmissionEvidenceProbeError, match="release_sha_mismatch"
    ):
        _run(http=http)


@pytest.mark.parametrize(
    "mutation",
    ["missing", "extra", "duplicate", "wrong_service_role", "wrong_build_count"],
)
def test_probe_rejects_non_exact_release_role_inventory(mutation):
    identity_cache = _release_identity_cache()
    entries = identity_cache["entries"]
    if mutation == "missing":
        entries.pop()
    elif mutation == "extra":
        extra = copy.deepcopy(entries[0])
        extra["physical_role"] = "unexpected"
        extra["role"] = "unexpected"
        entries.append(extra)
    elif mutation == "duplicate":
        entries[-1] = copy.deepcopy(entries[0])
    elif mutation == "wrong_service_role":
        entries[0]["role"] = "validator_weights"
    else:
        entries[0]["verified_build_count"] = 5

    with pytest.raises(
        probe.WeightSubmissionEvidenceProbeError, match="release_identity_invalid"
    ):
        _run(
            release_loader=lambda _evidence, *, http_open: copy.deepcopy(identity_cache)
        )


def test_probe_rejects_preliminary_authority_missing_auditor_and_stale_uid():
    with pytest.raises(
        probe.WeightSubmissionEvidenceProbeError, match="authority_not_finalized"
    ):
        _run(http=_Http(authority=_authority(stage="published")))

    state = _chain_state()
    state["metagraph_hotkeys"].remove(AUDITOR_TWO)
    with pytest.raises(
        probe.WeightSubmissionEvidenceProbeError, match="auditor_hotkey_missing"
    ):
        _run(chain=_Chain(state))

    state = _chain_state()
    state["validators"][1]["uid"] = 2
    with pytest.raises(
        probe.WeightSubmissionEvidenceProbeError, match="stale_validator_uid"
    ):
        _run(chain=_Chain(state))


def test_probe_rejects_nonadvanced_pending_nonzero_and_divergent_readback():
    state = _chain_state()
    state["validators"][2]["last_update"] = 100
    with pytest.raises(
        probe.WeightSubmissionEvidenceProbeError, match="last_update_not_advanced"
    ):
        _run(chain=_Chain(state))

    state = _chain_state()
    state["subnet_epoch_index"] = TARGET_SUBNET_EPOCH_INDEX
    with pytest.raises(
        probe.WeightSubmissionEvidenceProbeError, match="reveal_pending"
    ):
        _run(chain=_Chain(state))

    state = _chain_state()
    state["validators"][1]["mechanism_id"] = 1
    with pytest.raises(
        probe.WeightSubmissionEvidenceProbeError, match="mechanism_mismatch"
    ):
        _run(chain=_Chain(state))

    state = _chain_state()
    state["validators"][1]["weights"] = [[2, 10_001], [7, 55_534]]
    with pytest.raises(
        probe.WeightSubmissionEvidenceProbeError, match="vector_divergence"
    ):
        _run(chain=_Chain(state))


@pytest.mark.parametrize(
    ("netuid", "epoch", "code"),
    [
        (True, EPOCH_ID, "netuid_invalid"),
        ("71", EPOCH_ID, "netuid_invalid"),
        (0, EPOCH_ID, "netuid_invalid"),
        (65536, EPOCH_ID, "netuid_invalid"),
        (71, False, "epoch_id_invalid"),
        (71, "24080", "epoch_id_invalid"),
        (71, -1, "epoch_id_invalid"),
        (71, 1 << 64, "epoch_id_invalid"),
    ],
)
def test_input_ids_require_exact_bounded_integer_types(netuid, epoch, code):
    with pytest.raises(probe.WeightSubmissionEvidenceProbeError, match=code):
        probe._normalize_inputs(
            candidate_sha=CANDIDATE,
            netuid=netuid,
            epoch_id=epoch,
            auditor_hotkeys=[AUDITOR_ONE],
        )


def test_candidate_sha_requires_exact_string_without_coercion():
    class SideEffectingValue:
        def __str__(self):
            raise AssertionError("candidate SHA coercion executed")

    with pytest.raises(
        probe.WeightSubmissionEvidenceProbeError, match="candidate_sha_invalid"
    ):
        probe._normalize_inputs(
            candidate_sha=SideEffectingValue(),
            netuid=71,
            epoch_id=EPOCH_ID,
            auditor_hotkeys=[AUDITOR_ONE],
        )
    with pytest.raises(
        probe.WeightSubmissionEvidenceProbeError, match="candidate_sha_invalid"
    ):
        probe._normalize_inputs(
            candidate_sha=" " + CANDIDATE,
            netuid=71,
            epoch_id=EPOCH_ID,
            auditor_hotkeys=[AUDITOR_ONE],
        )


def test_auditor_iterable_stops_after_seventeenth_read_and_rejects_nonstrings():
    class InfiniteAuditors:
        def __init__(self):
            self.reads = 0

        def __iter__(self):
            return self

        def __next__(self):
            self.reads += 1
            return "5" + ("A" * 45) + "%02d" % self.reads

    values = InfiniteAuditors()
    with pytest.raises(
        probe.WeightSubmissionEvidenceProbeError, match="auditor_hotkey_limit_exceeded"
    ):
        probe._normalize_inputs(
            candidate_sha=CANDIDATE,
            netuid=71,
            epoch_id=EPOCH_ID,
            auditor_hotkeys=values,
        )
    assert values.reads == 17

    with pytest.raises(
        probe.WeightSubmissionEvidenceProbeError, match="auditor_hotkeys_invalid"
    ):
        probe._normalize_inputs(
            candidate_sha=CANDIDATE,
            netuid=71,
            epoch_id=EPOCH_ID,
            auditor_hotkeys=[object()],
        )

    for invalid in (AUDITOR_ONE + (" " * 17), "5" + ("é" * 40)):
        with pytest.raises(
            probe.WeightSubmissionEvidenceProbeError,
            match="auditor_hotkeys_invalid",
        ):
            probe._normalize_inputs(
                candidate_sha=CANDIDATE,
                netuid=71,
                epoch_id=EPOCH_ID,
                auditor_hotkeys=[invalid],
            )


@pytest.mark.parametrize("constant", [b"NaN", b"Infinity", b"-Infinity", b"1e999"])
def test_strict_json_rejects_nonfinite_values(constant):
    with pytest.raises(
        probe.WeightSubmissionEvidenceProbeError, match="gateway_document_invalid"
    ):
        probe._strict_json(b'{"value":' + constant + b"}", maximum_bytes=1024)


def test_strict_json_rejects_deep_input_with_fixed_error():
    payload = (b'{"v":' * 100) + b"0" + (b"}" * 100)
    with pytest.raises(
        probe.WeightSubmissionEvidenceProbeError, match="gateway_document_invalid"
    ):
        probe._strict_json(payload, maximum_bytes=4096)


def test_http_disables_ambient_proxies_and_bounds_malformed_urls(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://credential.invalid:8080")
    http = probe.ReadOnlyHttp("https://gateway.subnet71.com")
    handlers = [
        item for item in http._opener.handlers if isinstance(item, ProxyHandler)
    ]
    assert not handlers or all(item.proxies == {} for item in handlers)

    for value in ("https://gateway.invalid:bad", "https://" + "a" * 9000):
        with pytest.raises(
            probe.WeightSubmissionEvidenceProbeError, match="gateway_origin_invalid"
        ):
            probe.ReadOnlyHttp(value)

    with pytest.raises(
        probe.WeightSubmissionEvidenceProbeError, match="external_url_invalid"
    ):
        http.open_exact_url("https://" + "a" * 9000, method="GET")


def test_read_only_boundaries_reject_writes_before_connect_and_disable_ws_proxy():
    connections = []

    def connector(*args, **kwargs):
        connections.append((args, kwargs))
        raise AssertionError("only the allowed read should connect")

    reader = probe.FinalizedWebSocketChainReader(
        "wss://entrypoint-finney.opentensor.ai:443",
        chain_api=chain_api,
        connector=connector,
    )
    with pytest.raises(
        probe.WeightSubmissionEvidenceProbeError,
        match="write_capable_chain_call_rejected",
    ):
        reader.call("author_submitExtrinsic", ["0x00"])
    assert connections == []

    with pytest.raises(
        probe.WeightSubmissionEvidenceProbeError, match="chain_read_unavailable"
    ):
        reader.call("chain_getFinalizedHead", [])
    assert connections[0][1]["proxy"] is None

    http = probe.ReadOnlyHttp("https://gateway.subnet71.com")
    with pytest.raises(
        probe.WeightSubmissionEvidenceProbeError,
        match="write_capable_http_call_rejected",
    ):
        http.open_exact_url("https://gateway.subnet71.com/weights", method="POST")


def _git(root, *args, environment=None):
    return subprocess.run(
        ["git", *args],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
        env=environment,
    ).stdout.strip()


def _fixture_repository(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "probe-test@example.invalid")
    _git(root, "config", "user.name", "Probe Test")
    (root / "scripts").mkdir()
    (root / "scripts/probe_weight_submission_evidence_v2.py").write_text(
        "# exact probe fixture\n", encoding="utf-8"
    )
    (root / "leadpoet_canonical").mkdir()
    (root / "leadpoet_canonical/exact.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / ".gitignore").write_text(
        "*.pyc\n*.so\nignored_*.py\nignored_package\n", encoding="utf-8"
    )
    _git(root, "add", "--", ".gitignore", "scripts", "leadpoet_canonical")
    _git(root, "commit", "-qm", "fixture")
    return root, _git(root, "rev-parse", "HEAD")


def test_repository_preflight_binds_exact_head_tree_and_clean_status(tmp_path):
    root, candidate = _fixture_repository(tmp_path)
    binding = probe._CandidateRepository.preflight(root, candidate)
    assert binding.candidate_sha == candidate
    assert binding.tree_sha == _git(root, "rev-parse", "HEAD^{tree}")

    (root / "scripts/probe_weight_submission_evidence_v2.py").write_text(
        "# dirty\n", encoding="utf-8"
    )
    with pytest.raises(
        probe.WeightSubmissionEvidenceProbeError, match="candidate_sha_mismatch"
    ):
        probe._CandidateRepository.preflight(root, candidate)


def test_repository_preflight_rejects_head_mismatch_and_untracked(tmp_path):
    root, candidate = _fixture_repository(tmp_path)
    (root / "second.txt").write_text("second\n", encoding="utf-8")
    _git(root, "add", "--", "second.txt")
    _git(root, "commit", "-qm", "second")
    with pytest.raises(
        probe.WeightSubmissionEvidenceProbeError, match="candidate_sha_mismatch"
    ):
        probe._CandidateRepository.preflight(root, candidate)

    current = _git(root, "rev-parse", "HEAD")
    (root / "untracked.txt").write_text("untracked\n", encoding="utf-8")
    with pytest.raises(
        probe.WeightSubmissionEvidenceProbeError, match="candidate_sha_mismatch"
    ):
        probe._CandidateRepository.preflight(root, current)


def test_repository_preflight_rejects_ignored_import_shadow(tmp_path):
    root, candidate = _fixture_repository(tmp_path)
    package = root / "leadpoet_canonical"
    (package / "ignored_shadow.py").write_text("raise RuntimeError\n", encoding="utf-8")
    assert _git(root, "status", "--porcelain") == ""
    with pytest.raises(
        probe.WeightSubmissionEvidenceProbeError, match="candidate_sha_mismatch"
    ):
        probe._CandidateRepository.preflight(root, candidate)


@pytest.mark.parametrize("filename", ["websockets.pyc", "cryptography.so"])
def test_repository_preflight_rejects_ignored_root_import_shadow(tmp_path, filename):
    root, candidate = _fixture_repository(tmp_path)
    (root / filename).write_bytes(b"ignored import shadow")
    assert _git(root, "status", "--porcelain") == ""
    with pytest.raises(
        probe.WeightSubmissionEvidenceProbeError,
        match="candidate_sha_mismatch",
    ):
        probe._CandidateRepository.preflight(root, candidate)


def test_repository_preflight_rejects_ignored_root_package_symlink(tmp_path):
    root, candidate = _fixture_repository(tmp_path)
    target = tmp_path / "external-package"
    target.mkdir()
    (root / "ignored_package").symlink_to(target, target_is_directory=True)
    assert _git(root, "status", "--porcelain") == ""
    with pytest.raises(
        probe.WeightSubmissionEvidenceProbeError,
        match="candidate_sha_mismatch",
    ):
        probe._CandidateRepository.preflight(root, candidate)


def test_repository_preflight_sanitizes_git_environment(tmp_path, monkeypatch):
    root, candidate = _fixture_repository(tmp_path)
    fake = tmp_path / "fake-git-dir"
    fake.mkdir()
    graft = tmp_path / "ambient-grafts"
    graft.write_text("invalid ambient graft\n", encoding="utf-8")
    trace = tmp_path / "git-trace"
    monkeypatch.setenv("GIT_DIR", str(fake))
    monkeypatch.setenv("GIT_WORK_TREE", str(tmp_path / "wrong"))
    monkeypatch.setenv("GIT_OBJECT_DIRECTORY", str(tmp_path / "objects"))
    monkeypatch.setenv("GIT_GRAFT_FILE", str(graft))
    monkeypatch.setenv("GIT_SHALLOW_FILE", str(tmp_path / "missing-shallow"))
    monkeypatch.setenv("GIT_IMPLICIT_WORK_TREE", "0")
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "core.bare")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "true")
    monkeypatch.setenv("GIT_NO_REPLACE_OBJECTS", "0")
    monkeypatch.setenv("GIT_REPLACE_REF_BASE", "refs/hidden-replacements/")
    monkeypatch.setenv("GIT_TRACE", str(trace))
    monkeypatch.setenv("LD_PRELOAD", str(tmp_path / "missing-loader.so"))
    monkeypatch.setenv("DYLD_INSERT_LIBRARIES", str(tmp_path / "missing-loader.dylib"))
    binding = probe._CandidateRepository.preflight(root, candidate)
    assert binding.candidate_sha == candidate
    assert not trace.exists()


def test_repository_preflight_uses_absolute_git_and_sanitized_child_env(
    tmp_path, monkeypatch
):
    root, candidate = _fixture_repository(tmp_path)
    marker = tmp_path / "path-wrapper-executed"
    wrapper_dir = tmp_path / "wrapper-bin"
    wrapper_dir.mkdir()
    wrapper = wrapper_dir / "git"
    wrapper.write_text(
        "#!/bin/sh\nprintf executed > " + str(marker) + "\nexit 97\n",
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    monkeypatch.setenv("PATH", str(wrapper_dir))
    original_run = probe.subprocess.run
    observed = []

    def recording_run(*args, **kwargs):
        observed.append((list(args[0]), dict(kwargs["env"])))
        return original_run(*args, **kwargs)

    monkeypatch.setattr(probe.subprocess, "run", recording_run)
    binding = probe._CandidateRepository.preflight(root, candidate)
    assert binding.candidate_sha == candidate
    assert not marker.exists()
    assert observed
    assert {command[0] for command, _env in observed} == {str(probe.TRUSTED_GIT)}
    assert all(env["PATH"] == "/usr/bin:/bin" for _command, env in observed)
    assert all(env["GIT_ATTR_NOSYSTEM"] == "1" for _command, env in observed)
    assert all("LD_PRELOAD" not in env for _command, env in observed)
    assert all("DYLD_INSERT_LIBRARIES" not in env for _command, env in observed)
    assert all(
        "status" not in command and "diff" not in command for command, _env in observed
    )
    assert all("core.hooksPath=/dev/null" in command for command, _env in observed)


def test_repository_preflight_rejects_clean_filter_without_execution(tmp_path):
    root, _candidate = _fixture_repository(tmp_path)
    (root / ".gitattributes").write_text(
        "leadpoet_canonical/exact.py filter=sentinel\n", encoding="utf-8"
    )
    _git(root, "add", "--", ".gitattributes")
    _git(root, "commit", "-qm", "tracked attributes")
    candidate = _git(root, "rev-parse", "HEAD")
    binding = probe._CandidateRepository.preflight(root, candidate)
    marker = tmp_path / "clean-filter-executed"
    sentinel = tmp_path / "clean-filter"
    sentinel.write_text(
        "#!/bin/sh\nprintf executed > " + str(marker) + "\ncat\n",
        encoding="utf-8",
    )
    sentinel.chmod(0o755)
    _git(root, "config", "filter.sentinel.clean", str(sentinel))
    _git(root, "config", "filter.sentinel.smudge", "cat")
    with pytest.raises(
        probe.WeightSubmissionEvidenceProbeError,
        match="repository_filter_configuration",
    ):
        binding.recheck()
    with pytest.raises(
        probe.WeightSubmissionEvidenceProbeError,
        match="repository_filter_configuration",
    ):
        probe._CandidateRepository.preflight(root, candidate)
    assert not marker.exists()


def test_repository_preflight_never_executes_external_diff_or_fsmonitor(
    tmp_path, monkeypatch
):
    root, candidate = _fixture_repository(tmp_path)
    external_marker = tmp_path / "external-diff-executed"
    fsmonitor_marker = tmp_path / "fsmonitor-executed"

    def sentinel(name, marker):
        path = tmp_path / name
        path.write_text(
            "#!/usr/bin/env python3\n"
            "from pathlib import Path\n"
            f"Path({str(marker)!r}).write_text('executed', encoding='utf-8')\n",
            encoding="utf-8",
        )
        path.chmod(0o755)
        return path

    external = sentinel("external-diff", external_marker)
    fsmonitor = sentinel("fsmonitor", fsmonitor_marker)
    monkeypatch.setenv("GIT_EXTERNAL_DIFF", str(external))
    _git(root, "config", "diff.external", str(external))
    _git(root, "config", "core.fsmonitor", str(fsmonitor))
    _git(root, "config", "core.untrackedCache", "true")

    binding = probe._CandidateRepository.preflight(root, candidate)
    assert binding.candidate_sha == candidate
    assert not external_marker.exists()
    assert not fsmonitor_marker.exists()


def test_repository_preflight_rejects_replace_refs_and_grafts(tmp_path):
    root, candidate = _fixture_repository(tmp_path)
    (root / "second.txt").write_text("second\n", encoding="utf-8")
    _git(root, "add", "--", "second.txt")
    _git(root, "commit", "-qm", "second")
    second = _git(root, "rev-parse", "HEAD")
    _git(root, "replace", second, candidate)
    with pytest.raises(
        probe.WeightSubmissionEvidenceProbeError,
        match="repository_history_or_attributes_override",
    ):
        probe._CandidateRepository.preflight(root, second)
    _git(root, "replace", "-d", second)

    common = Path(_git(root, "rev-parse", "--git-common-dir"))
    if not common.is_absolute():
        common = root / common
    (common / "info").mkdir(exist_ok=True)
    (common / "info/grafts").write_text("fixture\n", encoding="utf-8")
    with pytest.raises(
        probe.WeightSubmissionEvidenceProbeError,
        match="repository_history_or_attributes_override",
    ):
        probe._CandidateRepository.preflight(root, second)


@pytest.mark.parametrize(
    ("relative", "contents"),
    [
        ("info/attributes", "*.py filter=ambient\n"),
        ("objects/info/alternates", "/external/object/store\n"),
        ("objects/info/http-alternates", "https://example.invalid/objects\n"),
    ],
)
def test_repository_preflight_and_recheck_reject_attribute_and_alternate_overlays(
    tmp_path, relative, contents
):
    root, candidate = _fixture_repository(tmp_path)
    binding = probe._CandidateRepository.preflight(root, candidate)
    common = Path(_git(root, "rev-parse", "--git-common-dir"))
    if not common.is_absolute():
        common = root / common
    overlay = common / relative
    overlay.parent.mkdir(parents=True, exist_ok=True)
    overlay.write_text(contents, encoding="utf-8")
    with pytest.raises(
        probe.WeightSubmissionEvidenceProbeError,
        match="repository_history_or_attributes_override",
    ):
        binding.recheck()


def test_candidate_source_finder_refuses_tracked_symlink_module(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    binding = SimpleNamespace(
        root=root,
        inventory={"leadpoet_canonical/unsafe.py": ("120000", "1" * 40)},
    )
    finder = probe._CandidateSourceFinder(binding)
    with pytest.raises(ModuleNotFoundError):
        finder.find_spec("leadpoet_canonical.unsafe")
    with pytest.raises(ModuleNotFoundError):
        finder.find_spec("validator_tee.host.release_v2")


def test_candidate_loader_executes_captured_bytes_during_restore_race(tmp_path):
    root, candidate = _fixture_repository(tmp_path)
    binding = probe._CandidateRepository.preflight(root, candidate)
    finder = probe._CandidateSourceFinder(binding)
    spec = finder.find_spec("leadpoet_canonical.exact")
    assert spec is not None
    assert isinstance(spec.loader, probe._CapturedCandidateSourceLoader)
    assert not isinstance(spec.loader, probe.importlib.machinery.SourceFileLoader)

    source_path = root / "leadpoet_canonical/exact.py"
    exact_source = source_path.read_bytes()
    marker = tmp_path / "live-path-code-executed"
    source_path.write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('executed', encoding='utf-8')\n"
        "VALUE = 2\n",
        encoding="utf-8",
    )
    previous = sys.modules.pop("leadpoet_canonical.exact", None)
    module = probe.importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        assert module.VALUE == 1
        assert not marker.exists()
    finally:
        source_path.write_bytes(exact_source)
        sys.modules.pop(spec.name, None)
        if previous is not None:
            sys.modules[spec.name] = previous
    binding._bind_loaded_module(spec.name, module)
    binding.recheck()


def test_loaded_repository_module_bytes_are_rebound_to_candidate(tmp_path):
    root, candidate = _fixture_repository(tmp_path)
    binding = probe._CandidateRepository.preflight(root, candidate)
    module = SimpleNamespace(__file__=str(root / "leadpoet_canonical/exact.py"))
    binding._bind_loaded_module("leadpoet_canonical.exact", module)

    (root / "leadpoet_canonical/exact.py").write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(
        probe.WeightSubmissionEvidenceProbeError,
        match="candidate_source_binding_invalid",
    ):
        binding._bind_loaded_module("leadpoet_canonical.exact", module)


def test_candidate_configuration_blobs_are_bound_before_and_after_validation():
    calls = []
    documents = {
        probe.PROFILE_RELATIVE_PATH: json.dumps(PROFILE).encode("utf-8"),
        probe.CUTOVER_RELATIVE_PATH: json.dumps({"schema_version": "fixture"}).encode(
            "utf-8"
        ),
    }

    class Binding:
        @staticmethod
        def read_bound_file(path, *, maximum_bytes):
            calls.append(("read", path))
            assert len(documents[path]) <= maximum_bytes
            return documents[path]

        @staticmethod
        def recheck():
            calls.append(("recheck", ""))

        @staticmethod
        def bind_loaded_modules():
            calls.append(("modules", ""))

    runtime = SimpleNamespace(
        validate_chain_signing_profile=lambda value: value,
        SubnetEpochCutover=SimpleNamespace(from_mapping=lambda _value: _Cutover()),
    )
    profile, cutover = probe._load_candidate_configuration(Binding(), runtime)
    assert profile == PROFILE
    assert cutover.netuid == 71
    assert calls.count(("read", probe.PROFILE_RELATIVE_PATH)) == 2
    assert calls.count(("read", probe.CUTOVER_RELATIVE_PATH)) == 2
    assert ("recheck", "") in calls
    assert ("modules", "") in calls


def test_candidate_import_activation_forces_root_to_sys_path_zero(
    tmp_path, monkeypatch
):
    root = tmp_path.resolve()
    binding = SimpleNamespace(root=root, inventory={})
    monkeypatch.setattr(probe, "_REPOSITORY_IMPORT_ROOTS", frozenset({"fixturepkg"}))
    monkeypatch.setattr(probe, "_reject_forbidden_runtime_modules", lambda: None)
    old_path = list(sys.path)
    old_meta = list(sys.meta_path)
    try:
        probe._activate_candidate_imports(binding)
        assert Path(sys.path[0]).resolve() == root
        assert isinstance(sys.meta_path[0], probe._CandidateSourceFinder)
    finally:
        sys.path[:] = old_path
        sys.meta_path[:] = old_meta


def test_dependency_paths_drop_relative_root_and_pythonpath_entries(
    tmp_path, monkeypatch
):
    root = tmp_path / "root"
    dependency = tmp_path / "dependency"
    root.mkdir()
    dependency.mkdir()
    monkeypatch.setattr(
        probe.sys,
        "path",
        ["", "relative", str(root), str(dependency)],
    )
    monkeypatch.setenv("PYTHONPATH", str(dependency))
    assert probe._sanitized_dependency_paths(root.resolve()) == []
    monkeypatch.delenv("PYTHONPATH")
    assert probe._sanitized_dependency_paths(root.resolve()) == []


@pytest.mark.parametrize(
    "name",
    [
        "bittensor",
        "gateway.tee.release_channel_v2",
        "leadpoet.publication.writer",
        "leadpoet.signer",
        "leadpoet.transport",
        "leadpoet.vsock",
        "leadpoet.wallet",
        "validator_tee.host.vsock_client",
    ],
)
def test_forbidden_runtime_module_deny_list(name):
    with pytest.raises(
        probe.WeightSubmissionEvidenceProbeError,
        match="forbidden_runtime_module_loaded",
    ):
        probe._reject_forbidden_runtime_modules({name: object()})
    probe._reject_forbidden_runtime_modules(
        {probe._PRIVATE_RELEASE_MODULE_PREFIX + "validator_manifest": object()}
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.replace(
            "from bittensor_wallet import Keypair",
            "from bittensor_wallet.keypair import Keypair",
        ),
        lambda value: value.replace(
            "from bittensor_wallet import Keypair",
            "from bittensor_wallet import Keypair, Wallet",
        ),
        lambda value: value.replace(
            "keypair = Keypair(ss58_address=hotkey)",
            "keypair = Keypair(ss58_address=hotkey, private_key=signature_hex)",
        ),
        lambda value: value.replace("keypair.verify(", "keypair.sign("),
        lambda value: value.replace(
            "binding_msg.encode()", "binding_msg.encode('utf-8')"
        ),
        lambda value: value.replace(
            "from bittensor_wallet import Keypair",
            "from unittest import mock\n        from bittensor_wallet import Keypair",
        ),
        lambda value: value.replace("expected_code_hash: str,", "secret_path: str,"),
    ],
)
def test_binding_wallet_policy_allows_only_exact_public_verification(mutation):
    source = (probe.ROOT / probe.BINDING_RELATIVE_PATH).read_text(encoding="utf-8")
    probe._audit_binding_public_key_verifier(source.encode("utf-8"))
    mutated = mutation(source)
    assert mutated != source
    with pytest.raises(
        probe.WeightSubmissionEvidenceProbeError,
        match="candidate_public_key_policy_invalid",
    ):
        probe._audit_binding_public_key_verifier(mutated.encode("utf-8"))


def test_nonbinding_candidate_module_cannot_reach_wallet_api():
    with pytest.raises(
        probe.WeightSubmissionEvidenceProbeError,
        match="candidate_public_key_policy_invalid",
    ):
        probe._audit_candidate_wallet_surface(
            "leadpoet_canonical/other.py",
            b"from bittensor_wallet import Keypair\n",
            binding_policy=None,
            object_id="1" * 40,
        )


def test_public_key_dependency_is_preflighted_and_runtime_restricted(monkeypatch):
    source = (probe.ROOT / probe.BINDING_RELATIVE_PATH).read_bytes()
    calls = []

    class Keypair:
        def __init__(self, *, ss58_address):
            calls.append(("constructor", ss58_address))

        def verify(self, message, signature):
            calls.append(("verify", message, signature))
            return True

    keypair = Keypair
    public = probe.ModuleType("bittensor_wallet")
    public.__file__ = "/trusted/bittensor_wallet/__init__.py"
    public.Keypair = keypair
    extension = probe.ModuleType("bittensor_wallet.bittensor_wallet")
    extension.__file__ = "/trusted/bittensor_wallet/bittensor_wallet.fixture.so"
    extension.__loader__ = probe.importlib.machinery.ExtensionFileLoader(
        extension.__name__, extension.__file__
    )
    extension.Keypair = keypair
    binding = SimpleNamespace(
        inventory={probe.BINDING_RELATIVE_PATH: ("100644", "1" * 40)},
        public_key_verifier_policy=None,
        read_bound_file=lambda path, maximum_bytes: source,
        recheck=lambda: None,
    )
    prior = {
        name: module
        for name, module in tuple(sys.modules.items())
        if name == "bittensor_wallet" or name.startswith("bittensor_wallet.")
    }
    for name in prior:
        sys.modules.pop(name, None)

    def importer(name):
        assert name == "bittensor_wallet"
        sys.modules["bittensor_wallet"] = public
        sys.modules["bittensor_wallet.bittensor_wallet"] = extension
        return public

    monkeypatch.setattr(
        probe,
        "_trusted_dependency_origin",
        lambda module, extension: str(Path(module.__file__).resolve()),
    )
    reject_runtime = probe._reject_forbidden_runtime_modules

    def reject_wallet_runtime(modules=None, *, public_key_verifier_policy=None):
        observed = sys.modules if modules is None else modules
        wallet_modules = {
            name: module
            for name, module in observed.items()
            if name == "bittensor_wallet" or name.startswith("bittensor_wallet.")
        }
        return reject_runtime(
            wallet_modules,
            public_key_verifier_policy=public_key_verifier_policy,
        )

    monkeypatch.setattr(
        probe, "_reject_forbidden_runtime_modules", reject_wallet_runtime
    )
    try:
        policy = probe._preflight_public_key_verifier_dependency(
            binding, module_importer=importer
        )
        assert binding.public_key_verifier_policy is policy
        assert {
            name
            for name in vars(sys.modules["bittensor_wallet"])
            if not name.startswith("_")
        } == {"Keypair"}
        namespace = {}
        exec("from bittensor_wallet import Keypair", namespace)
        assert namespace["Keypair"] is keypair
        from leadpoet_canonical.binding import (
            create_binding_message,
            verify_binding_message,
        )

        binding_message = create_binding_message(
            netuid=71,
            chain=PROFILE["chain_endpoint"],
            enclave_pubkey="public-enclave-key",
            validator_code_hash="public-code-hash",
        )
        assert verify_binding_message(
            binding_message,
            "00",
            PRIMARY,
            71,
            PROFILE["chain_endpoint"],
            "public-enclave-key",
            "public-code-hash",
        )
        assert calls == [
            ("constructor", PRIMARY),
            ("verify", binding_message.encode(), b"\x00"),
        ]
        probe._reject_forbidden_runtime_modules(public_key_verifier_policy=policy)

        sys.modules["bittensor_wallet.keyfile"] = probe.ModuleType(
            "bittensor_wallet.keyfile"
        )
        with pytest.raises(
            probe.WeightSubmissionEvidenceProbeError,
            match="forbidden_runtime_module_loaded",
        ):
            probe._reject_forbidden_runtime_modules(public_key_verifier_policy=policy)
        sys.modules.pop("bittensor_wallet.keyfile")

        sys.modules["bittensor_wallet"].Wallet = object()
        with pytest.raises(
            probe.WeightSubmissionEvidenceProbeError,
            match="public_key_verifier_runtime_invalid",
        ):
            probe._reject_forbidden_runtime_modules(public_key_verifier_policy=policy)
        del sys.modules["bittensor_wallet"].Wallet

        original_verify = keypair.verify
        keypair.verify = lambda *_args: True
        with pytest.raises(
            probe.WeightSubmissionEvidenceProbeError,
            match="public_key_verifier_runtime_invalid",
        ):
            probe._reject_forbidden_runtime_modules(public_key_verifier_policy=policy)
        keypair.verify = original_verify
    finally:
        for name in tuple(sys.modules):
            if name == "bittensor_wallet" or name.startswith("bittensor_wallet."):
                sys.modules.pop(name, None)
        sys.modules.update(prior)


def test_public_key_dependency_failure_is_fixed_and_redacted(monkeypatch, capsys):
    source = (probe.ROOT / probe.BINDING_RELATIVE_PATH).read_bytes()
    binding = SimpleNamespace(
        inventory={probe.BINDING_RELATIVE_PATH: ("100644", "1" * 40)},
        public_key_verifier_policy=None,
        read_bound_file=lambda path, maximum_bytes: source,
        recheck=lambda: None,
    )
    prior = {
        name: module
        for name, module in tuple(sys.modules.items())
        if name == "bittensor_wallet" or name.startswith("bittensor_wallet.")
    }
    for name in prior:
        sys.modules.pop(name, None)
    try:
        with pytest.raises(
            probe.WeightSubmissionEvidenceProbeError,
            match="public_key_verifier_dependency_invalid",
        ):
            probe._preflight_public_key_verifier_dependency(
                binding,
                module_importer=lambda _name: (_ for _ in ()).throw(
                    ImportError("SENSITIVE_DEPENDENCY_DETAIL")
                ),
            )
        captured = capsys.readouterr()
        assert "SENSITIVE_DEPENDENCY_DETAIL" not in captured.out + captured.err
        assert binding.public_key_verifier_policy is None
    finally:
        for name in tuple(sys.modules):
            if name == "bittensor_wallet" or name.startswith("bittensor_wallet."):
                sys.modules.pop(name, None)
        sys.modules.update(prior)


def test_candidate_release_contract_uses_private_modules_without_initializers():
    before = set(sys.modules)
    validators = _private_release_validators()
    added = set(sys.modules) - before
    contract = probe._candidate_release_contract(validators)
    assert contract["roles"] == probe._EXPECTED_RELEASE_IDENTITY_ROLES
    assert contract["build_counts"] == {
        role: 6 for role in probe._EXPECTED_RELEASE_IDENTITY_ROLES
    }
    assert not any(
        name == "gateway"
        or name.startswith("gateway.")
        or name == "validator_tee"
        or name.startswith("validator_tee.")
        for name in added
    )
    assert validators.gateway.__name__.startswith(probe._PRIVATE_RELEASE_MODULE_PREFIX)
    assert validators.validator.__name__.startswith(
        probe._PRIVATE_RELEASE_MODULE_PREFIX
    )


def test_private_release_loader_and_deny_list_pass_in_fresh_isolated_process():
    script_path = str(Path(probe.__file__).resolve())
    root = str(probe.ROOT)
    program = f"""
import importlib.util
from pathlib import Path
import sys
spec = importlib.util.spec_from_file_location('fresh_probe', {script_path!r})
probe = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = probe
spec.loader.exec_module(probe)
class Binding:
    root = Path({root!r})
    def read_bound_file(self, path, *, maximum_bytes):
        value = (self.root / path).read_bytes()
        assert len(value) <= maximum_bytes
        return value
    def recheck(self):
        return None
validators = probe._load_candidate_release_validators(Binding())
probe._reject_forbidden_runtime_modules()
assert validators.gateway.__name__.startswith(probe._PRIVATE_RELEASE_MODULE_PREFIX)
assert not any(name == 'gateway' or name.startswith('gateway.') for name in sys.modules)
assert not any(name == 'validator_tee' or name.startswith('validator_tee.') for name in sys.modules)
print('ok')
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", program],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == "ok\n"
    assert completed.stderr == ""


def test_probe_has_no_repository_import_before_preflight():
    source = Path(probe.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    repository_roots = tuple(probe._REPOSITORY_IMPORT_ROOTS)
    for statement in tree.body:
        if isinstance(statement, ast.ImportFrom):
            assert not str(statement.module or "").startswith(repository_roots)
        if isinstance(statement, ast.Import):
            assert all(
                not alias.name.startswith(repository_roots) for alias in statement.names
            )
    public = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef)
        and item.name == "verify_weight_submission_evidence_v2"
    )
    calls = [
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(public)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Attribute, ast.Name))
    ]
    assert calls.index("preflight") < calls.index("_load_candidate_runtime")
    runtime_loader = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == "_load_candidate_runtime"
    )
    runtime_calls = [
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(runtime_loader)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Attribute, ast.Name))
    ]
    assert runtime_calls.index("_preflight_public_key_verifier_dependency") < (
        runtime_calls.index("import_module")
    )
    assert calls.index("_load_candidate_runtime") < calls.index(
        "_verify_weight_submission_evidence_core"
    )
    assert "fetch_locked_release_identity_cache" not in source
    assert "gateway.tee.release_channel_v2" not in probe._RUNTIME_MODULE_NAMES
    assert "validator_tee.host.release_v2" not in probe._RUNTIME_MODULE_NAMES


def test_cli_requires_isolation_and_emits_fixed_argument_errors(tmp_path):
    script = str(Path(probe.__file__).resolve())
    nonisolated = subprocess.run(
        [sys.executable, script], capture_output=True, text=True, check=False
    )
    assert nonisolated.returncode == 2
    assert nonisolated.stdout == ""
    assert nonisolated.stderr == "ERROR:isolated_python_required\n"

    for extra in ([], ["--unknown-option", "sensitive-value"]):
        isolated = subprocess.run(
            [sys.executable, "-I", script, *extra],
            capture_output=True,
            text=True,
            check=False,
        )
        assert isolated.returncode == 1
        assert isolated.stdout == ""
        assert isolated.stderr == "ERROR:arguments_invalid\n"
        assert "sensitive-value" not in isolated.stderr

    marker = tmp_path / "pythonpath-executed"
    pythonpath = tmp_path / "pythonpath"
    pythonpath.mkdir()
    (pythonpath / "sitecustomize.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('executed', encoding='utf-8')\n",
        encoding="utf-8",
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(pythonpath)
    isolated = subprocess.run(
        [sys.executable, "-I", script],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    assert isolated.stderr == "ERROR:arguments_invalid\n"
    assert not marker.exists()


def test_public_entrypoint_cannot_bypass_isolated_python(monkeypatch):
    assert not sys.flags.isolated
    monkeypatch.setattr(
        probe._CandidateRepository,
        "preflight",
        lambda *_args, **_kwargs: pytest.fail("preflight must not run"),
    )
    with pytest.raises(
        probe.WeightSubmissionEvidenceProbeError,
        match="isolated_python_required",
    ):
        probe.verify_weight_submission_evidence_v2(
            candidate_sha=CANDIDATE,
            netuid=71,
            epoch_id=EPOCH_ID,
            auditor_hotkeys=[AUDITOR_ONE],
        )


def test_cli_bounds_auditor_action_before_repository_or_output():
    common = [
        "--candidate-sha",
        CANDIDATE,
        "--netuid",
        "71",
        "--epoch-id",
        str(EPOCH_ID),
    ]
    too_many = [
        item for _index in range(17) for item in ("--auditor-hotkey", AUDITOR_ONE)
    ]
    stderr = []

    class Capture:
        @staticmethod
        def write(value):
            stderr.append(value)

    original = probe.sys.stderr
    probe.sys.stderr = Capture()
    try:
        assert probe.main([*common, *too_many]) == 1
    finally:
        probe.sys.stderr = original
    assert "".join(stderr) == "ERROR:auditor_hotkey_limit_exceeded\n"

    stderr.clear()
    probe.sys.stderr = Capture()
    try:
        assert (
            probe.main(
                [
                    *common,
                    "--auditor-hotkey",
                    AUDITOR_ONE + (" " * 17),
                ]
            )
            == 1
        )
    finally:
        probe.sys.stderr = original
    assert "".join(stderr) == "ERROR:auditor_hotkeys_invalid\n"


def test_cli_catches_final_serialization_failure(monkeypatch, capsys):
    monkeypatch.setattr(
        probe,
        "verify_weight_submission_evidence_v2",
        lambda **_kwargs: {"bounded": True},
    )
    monkeypatch.setattr(
        probe,
        "_canonical_json",
        lambda _value: (_ for _ in ()).throw(RuntimeError("SENSITIVE_DETAIL")),
    )
    result = probe.main(
        [
            "--candidate-sha",
            CANDIDATE,
            "--netuid",
            "71",
            "--epoch-id",
            str(EPOCH_ID),
            "--auditor-hotkey",
            AUDITOR_ONE,
        ]
    )
    captured = capsys.readouterr()
    assert result == 1
    assert captured.out == ""
    assert captured.err == "ERROR:probe_failed\n"
    assert "SENSITIVE_DETAIL" not in captured.err


def test_git_and_import_failures_are_fixed_and_redacted(tmp_path, monkeypatch, capsys):
    with pytest.raises(
        probe.WeightSubmissionEvidenceProbeError,
        match="repository_identity_unavailable",
    ):
        probe._CandidateRepository.preflight(tmp_path, CANDIDATE)
    assert capsys.readouterr().err == ""

    fake_binding = SimpleNamespace(
        recheck=lambda: None,
        bind_loaded_modules=lambda: None,
        public_key_verifier_policy=object(),
    )
    monkeypatch.setattr(probe, "_activate_candidate_imports", lambda _binding: None)
    monkeypatch.setattr(
        probe,
        "_load_candidate_release_validators",
        lambda _binding: SimpleNamespace(),
    )
    monkeypatch.setattr(
        probe,
        "_candidate_release_contract",
        lambda _validators: RELEASE_CONTRACT,
    )
    monkeypatch.setattr(
        probe,
        "_preflight_public_key_verifier_dependency",
        lambda binding: binding.public_key_verifier_policy,
    )
    monkeypatch.setattr(
        probe, "_reject_forbidden_runtime_modules", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        probe.importlib,
        "import_module",
        lambda _name: (_ for _ in ()).throw(RuntimeError("SENSITIVE_IMPORT_DETAIL")),
    )
    with pytest.raises(
        probe.WeightSubmissionEvidenceProbeError, match="repository_import_invalid"
    ):
        probe._load_candidate_runtime(fake_binding)
    captured = capsys.readouterr()
    assert "SENSITIVE_IMPORT_DETAIL" not in captured.out + captured.err


def _hash(character):
    return "sha256:" + character * 64


def _private_release_validators():
    topology_path = "gateway/tee/topology.py"
    gateway_path = "gateway/tee/release_manifest_v2.py"
    validator_path = "validator_tee/host/release_v2.py"
    topology_alias = probe._PRIVATE_RELEASE_MODULE_PREFIX + "topology"
    topology = probe._compile_private_module(
        name=topology_alias,
        relative_path=topology_path,
        source=(probe.ROOT / topology_path).read_bytes(),
    )
    gateway = probe._compile_private_module(
        name=probe._PRIVATE_RELEASE_MODULE_PREFIX + "gateway_manifest",
        relative_path=gateway_path,
        source=(probe.ROOT / gateway_path).read_bytes(),
        topology_alias=topology_alias,
    )
    validator = probe._compile_private_module(
        name=probe._PRIVATE_RELEASE_MODULE_PREFIX + "validator_manifest",
        relative_path=validator_path,
        source=(probe.ROOT / validator_path).read_bytes(),
    )
    return SimpleNamespace(topology=topology, gateway=gateway, validator=validator)


def _gateway_manifest(commit=CANDIDATE, *, validators=None):
    validators = validators or _private_release_validators()
    rows = []
    for index, (role, spec) in enumerate(
        sorted(validators.topology.ROLE_SPECS.items())
    ):
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
            "topology_hash": validators.topology.topology_hash(),
        }
        for domain in ("gateway", "validator"):
            for ordinal in (1, 2, 3):
                rows.append(
                    {
                        "schema_version": (
                            validators.gateway.BUILD_EVIDENCE_SCHEMA_VERSION
                        ),
                        "builder_domain": domain,
                        "builder_id": domain + "-parent",
                        "build_ordinal": ordinal,
                        "physical_role": role,
                        "service_role": spec["service_role"],
                        **deterministic,
                    }
                )
    return validators.gateway.build_release_manifest(
        rows, acceptance_signer_pubkey_hash=_hash("f")
    )


def _validator_manifest(commit=CANDIDATE, *, validators=None):
    validators = validators or _private_release_validators()
    release = validators.validator.build_validator_release(
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
        validators.validator.build_validator_build_evidence(
            release,
            builder_domain=domain,
            builder_id=domain + "-parent",
            build_ordinal=ordinal,
        )
        for domain in ("gateway", "validator")
        for ordinal in (1, 2, 3)
    ]
    return validators.validator.build_validator_release_manifest(evidence)


def _locked_fixture():
    validators = _private_release_validators()
    channel_body = {
        "schema_version": probe.RELEASE_CHANNEL_SCHEMA_VERSION,
        "commit_sha": CANDIDATE,
        "gateway_release_manifest": _gateway_manifest(validators=validators),
        "validator_release_manifest": _validator_manifest(validators=validators),
    }
    channel = {**channel_body, "channel_hash": probe._sha256_json(channel_body)}
    bucket = "leadpoet-attested-v2-artifacts-493765492819"
    path = f"/attested-v2/releases/{CANDIDATE}/release-channel-v2.json"
    query = (
        "X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=fixture&"
        "X-Amz-Date=20260819T000000Z&X-Amz-Expires=300&"
        "X-Amz-SignedHeaders=host&X-Amz-Signature=fixture&VersionId=version-1"
    )
    url = f"https://{bucket}.s3.amazonaws.com{path}?{query}"
    evidence = {
        "schema_version": "leadpoet.auditor_release_evidence.v2",
        "commit_sha": CANDIDATE,
        "release_channel_version_id": "version-1",
        "release_channel_get_url": url,
        "release_channel_head_url": url,
    }
    return evidence, json.dumps(channel, separators=(",", ":")).encode("utf-8")


class _LockedResponse:
    def __init__(
        self,
        requested_url,
        *,
        method,
        payload,
        head_version="version-1",
        get_version="version-1",
        retain_until=None,
        observed_url=None,
        lock_mode="COMPLIANCE",
    ):
        self.status = 200
        self.requested_url = requested_url
        self.observed_url = observed_url or requested_url
        self.headers = {
            "x-amz-object-lock-mode": lock_mode,
            "x-amz-object-lock-retain-until-date": (
                retain_until
                or (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
            ),
            "x-amz-version-id": head_version if method == "HEAD" else get_version,
        }
        self.payload = payload if method == "GET" else b""

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def geturl(self):
        return self.observed_url

    def read(self, limit):
        return self.payload[:limit]


def _load_locked(**overrides):
    evidence, payload = _locked_fixture()
    calls = []
    validators = _private_release_validators()

    def opener(url, *, method):
        calls.append((url, method))
        return _LockedResponse(url, method=method, payload=payload, **overrides)

    return (
        probe._load_immutable_release_identity(
            evidence,
            http_open=opener,
            validators=validators,
        ),
        calls,
    )


def test_immutable_release_loader_succeeds_with_exact_release_identities():
    cache, calls = _load_locked()
    assert calls[0][1] == "HEAD"
    assert calls[1][1] == "GET"
    assert {item["physical_role"] for item in cache["entries"]} == set(
        probe._EXPECTED_RELEASE_IDENTITY_ROLES
    )
    assert len(cache["entries"]) == len(probe._EXPECTED_RELEASE_IDENTITY_ROLES)
    assert {item["verified_build_count"] for item in cache["entries"]} == {6}
    assert {item["commit_sha"] for item in cache["entries"]} == {CANDIDATE}


@pytest.mark.parametrize(
    "overrides",
    [
        {"head_version": "wrong"},
        {"get_version": "wrong"},
        {
            "retain_until": (
                datetime.now(timezone.utc) - timedelta(seconds=1)
            ).isoformat()
        },
        {"observed_url": "https://redirect.invalid/release"},
        {"lock_mode": "GOVERNANCE"},
    ],
)
def test_immutable_release_loader_rejects_head_get_expiry_redirect_and_lock(overrides):
    with pytest.raises(
        probe.WeightSubmissionEvidenceProbeError,
        match="object_locked_release_invalid",
    ):
        _load_locked(**overrides)


@pytest.mark.parametrize("mutation", ["path", "query", "version"])
def test_immutable_release_loader_rejects_path_query_and_version_mutation(mutation):
    evidence, _payload = _locked_fixture()
    if mutation == "path":
        evidence["release_channel_get_url"] = evidence[
            "release_channel_get_url"
        ].replace("release-channel-v2.json", "other.json")
    elif mutation == "query":
        evidence["release_channel_head_url"] += "&unexpected=1"
    else:
        evidence["release_channel_get_url"] = evidence[
            "release_channel_get_url"
        ].replace("VersionId=version-1", "VersionId=wrong")
    with pytest.raises(
        probe.WeightSubmissionEvidenceProbeError,
        match="object_locked_release_invalid",
    ):
        probe._load_immutable_release_identity(
            evidence,
            http_open=lambda *_args, **_kwargs: pytest.fail(
                "invalid immutable URL must fail before HTTP"
            ),
            validators=_private_release_validators(),
        )


def _compact(value):
    if value < 1 << 6:
        return bytes((value << 2,))
    if value < 1 << 14:
        return ((value << 2) | 1).to_bytes(2, "little")
    return ((value << 2) | 2).to_bytes(4, "little")


def _selective_metagraph(accounts, *, block):
    encoded = bytearray((1, 0x1D, 0x01))
    encoded.extend(b"\x00" * 4)
    encoded.extend(b"\x01" + accounts[0])
    encoded.extend(b"\x00")
    encoded.extend(b"\x01" + _compact(block))
    encoded.extend(b"\x00" * 44)
    encoded.extend(b"\x01" + _compact(len(accounts)) + b"".join(accounts))
    encoded.extend(b"\x00" * 24)
    return "0x" + encoded.hex()


class _Connection:
    def __init__(self, responses):
        self.responses = responses
        self.requests = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def send(self, body):
        self.requests.append(json.loads(body))

    def recv(self, *, timeout):
        request = self.requests[-1]
        key = (request["method"], tuple(request["params"]))
        return json.dumps(
            {"jsonrpc": "2.0", "id": request["id"], "result": self.responses[key]},
            separators=(",", ":"),
        )


def test_websocket_reader_pins_mechanism_zero_storage_to_one_finalized_hash():
    accounts = [bytes([index]) * 32 for index in (1, 2, 3)]
    hotkeys = [chain_api.ss58_encode_account_id(account) for account in accounts]
    block_hash = "0x" + "6" * 64
    block = 8_700_000
    last_updates = (
        "0x"
        + (
            _compact(3)
            + (8_699_000).to_bytes(8, "little")
            + (8_699_001).to_bytes(8, "little")
            + (8_699_002).to_bytes(8, "little")
        ).hex()
    )
    weights = (
        "0x"
        + (
            _compact(2)
            + (2).to_bytes(2, "little")
            + (10_000).to_bytes(2, "little")
            + (7).to_bytes(2, "little")
            + (55_535).to_bytes(2, "little")
        ).hex()
    )
    responses = {
        ("chain_getFinalizedHead", ()): block_hash,
        ("chain_getHeader", (block_hash,)): {
            "number": hex(block),
            "stateRoot": "0x" + "7" * 64,
            "parentHash": "0x" + "8" * 64,
            "extrinsicsRoot": "0x" + "9" * 64,
        },
        (
            "state_call",
            (
                chain_api.CHAIN_RPC_METHOD,
                chain_api.encode_selective_metagraph_params(netuid=71, mechid=0),
                block_hash,
            ),
        ): _selective_metagraph(accounts, block=block),
        (
            "state_getStorage",
            (
                chain_api.subnet_epoch_storage_key(
                    storage_name="SubnetEpochIndex", netuid=71
                ),
                block_hash,
            ),
        ): "0x"
        + TARGET_SUBNET_EPOCH_INDEX.to_bytes(8, "little").hex(),
        (
            "state_getStorage",
            (chain_api.last_update_storage_key(netuid=71), block_hash),
        ): last_updates,
        (
            "state_getStorage",
            (chain_api.weights_storage_key(netuid=71, validator_uid=0), block_hash),
        ): weights,
        (
            "state_getStorage",
            (chain_api.weights_storage_key(netuid=71, validator_uid=2), block_hash),
        ): weights,
    }
    connection = _Connection(responses)
    reader = probe.FinalizedWebSocketChainReader(
        "wss://entrypoint-finney.opentensor.ai:443",
        chain_api=chain_api,
        connector=lambda *_args, **_kwargs: connection,
    )
    state = reader.read_finalized_state(netuid=71, hotkeys=[hotkeys[0], hotkeys[2]])
    assert state["block_hash"] == block_hash
    assert [item["uid"] for item in state["validators"]] == [0, 2]
    assert {
        request["method"] for request in connection.requests
    } <= probe._READ_ONLY_RPC_METHODS
    for request in connection.requests[2:]:
        assert request["params"][-1] == block_hash
