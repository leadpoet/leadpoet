from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from gateway.tee.active_release_requirements_v2 import (
    ActiveReleaseRequirementsV2Error,
    build_active_release_requirements_v2,
)
from gateway.tee import prepare_active_release_lineage_v2 as prepare
from gateway.tee.reward_executor_v2 import source_add_reward_row_projection_v2
from gateway.tee.topology import ROLE_SPECS
from leadpoet_canonical.attested_v2 import canonical_json, sha256_json
from tests.test_validator_hotkey_authority_v2 import _profile


CANDIDATE = "a" * 40
AUTHORITY = "f" * 40
RESTART_INVOCATION_ID = "gateway-24745-test"
RUNNING_GATEWAY = "b" * 40
RUNNING_VALIDATOR = "c" * 40
JOURNAL_COMMIT = "d" * 40
NEW_JOURNAL_COMMIT = "e" * 40
LINEAGE_ID = "sha256:" + "1" * 64
VALIDATOR_HOTKEY = "5FqLp5QmNRiHGyj3xbLVnDHfCx25qxJX5CUhpndF9GFfZZiK"
CHAIN_PROFILE = _profile()


def _hash(character: str) -> str:
    return "sha256:" + character * 64


def _requirements(
    *,
    transitions=(),
    authority=AUTHORITY,
    invocation_id=RESTART_INVOCATION_ID,
):
    return build_active_release_requirements_v2(
        candidate_commit_sha=CANDIDATE,
        authority_commit_sha=authority,
        restart_invocation_id=invocation_id,
        transition_commit_shas=transitions,
        active_graphs={},
        expected_lineage_id=LINEAGE_ID,
        boot_verifier=lambda identity: identity,
    )


def _lineage(commits, *, current_commit=CANDIDATE):
    releases = {}
    for index, commit in enumerate(sorted(commits), start=1):
        role_expectations = {}
        current_roles = (*sorted(ROLE_SPECS), "validator_weights")
        historical_roles = ("gateway_autoresearch", *current_roles)
        for role in (
            current_roles if commit == current_commit else historical_roles
        ):
            role_expectations[role] = {
                "commit_sha": commit,
                "pcr0": ("%x" % ((index % 15) + 1)) * 96,
                "build_manifest_hash": _hash("2"),
                "dependency_lock_hash": _hash("3"),
            }
        releases[commit] = {
            "channel_hash": _hash("4"),
            "gateway_release_hash": _hash("5"),
            "roles": role_expectations,
        }
    body = {
        "schema_version": "leadpoet.attested_release_lineage.v1",
        "current_commit_sha": current_commit,
        "current_gateway_release_hash": releases[current_commit][
            "gateway_release_hash"
        ],
        "releases": releases,
    }
    return {**body, "lineage_hash": sha256_json(body)}


def _patch_lineage_boundaries(monkeypatch):
    fetched = []

    def fake_fetch(**kwargs):
        required = list(kwargs["required_commits"])
        fetched.append(required)
        return _lineage(required)

    monkeypatch.setattr(prepare, "_fetch_exact_release_lineage_v2", fake_fetch)
    monkeypatch.setattr(
        prepare,
        "_compact_boot_verifier",
        lambda _lineage_value: (lambda identity: identity),
    )
    return fetched


def _source_add_graph(*, root: str, purpose: str, output_root: str) -> dict:
    return {
        "schema_version": "leadpoet.attested_receipt_graph.v2",
        "root_receipt_hash": root,
        "boot_identities": [],
        "receipts": [
            {
                "receipt_hash": root,
                "role": "gateway_coordinator",
                "purpose": purpose,
                "status": "succeeded",
                "output_root": output_root,
                "parent_receipt_hashes": [],
            }
        ],
        "transport_attempts": [],
        "host_operations": [],
    }


@pytest.mark.asyncio
async def test_source_add_active_graphs_select_pending_and_reward_authority(
    monkeypatch,
) -> None:
    submission_id = "source_add_submission:1111111111111111"
    intent_id = "source_add_reward_intent:2222222222222222"
    reward_ref = "source_add_reward:3333333333333333"
    provenance_receipt = _hash("7")
    provenance_artifact = _hash("8")
    decision_receipt = _hash("9")
    intent = {
        "intent_id": intent_id,
        "submission_id": submission_id,
        "adapter_id": "adapter-pending",
        "miner_hotkey": VALIDATOR_HOTKEY,
        "leg": 1,
        "intent_status": "retry_wait",
        "approval_kind": "provenance_precheck_passed",
        "provenance_receipt_hash": provenance_receipt,
        "provenance_artifact_hash": provenance_artifact,
    }
    authority = {
        "submission_id": submission_id,
        "adapter_id": intent["adapter_id"],
        "miner_hotkey": intent["miner_hotkey"],
        "precheck_status": "provenance_precheck_passed",
        "provenance_receipt_hash": provenance_receipt,
        "provenance_artifact_hash": provenance_artifact,
    }
    future_reward = {
        "reward_ref": reward_ref,
        "adapter_id": "adapter-future",
        "miner_hotkey": VALIDATOR_HOTKEY,
        "leg": 1,
        "reward_kind": "source_acceptance",
        "alpha_percent": 0.2,
        "reward_epochs": 20,
        "start_epoch": 24_740,
        "trigger_evidence_doc": {"provenance_precheck_passed": True},
        "public_label": "Source acceptance reward",
        "current_reward_status": "active",
    }
    expected_decision_hash = sha256_json(
        source_add_reward_row_projection_v2(
            "source_add_leg1",
            {**future_reward, "initial_reward_status": "active"},
        )
    )
    calls = []

    async def select(table, **kwargs):
        calls.append((table, kwargs))
        if table == "research_lab_source_add_reward_intents":
            return [dict(intent)]
        if table == "research_lab_source_add_reward_current":
            return [dict(future_reward)]
        if table == "research_lab_source_add_provenance_leg1_authority_v1":
            return [dict(authority)]
        raise AssertionError(table)

    async def load(artifacts):
        assert set(artifacts) == {
            ("source_add_provenance", submission_id, provenance_artifact),
            ("source_add_reward_decision", reward_ref, expected_decision_hash),
        }
        return {
            ("source_add_provenance", submission_id, provenance_artifact): (
                _source_add_graph(
                    root=provenance_receipt,
                    purpose="research_lab.source_add_provenance.v2",
                    output_root=provenance_artifact,
                )
            ),
            ("source_add_reward_decision", reward_ref, expected_decision_hash): (
                _source_add_graph(
                    root=decision_receipt,
                    purpose="research_lab.reward_decision.v2",
                    output_root=expected_decision_hash,
                )
            ),
        }

    monkeypatch.setattr(
        prepare, "validate_receipt_graph", lambda *_args, **_kwargs: None
    )
    graphs = await prepare._load_active_source_add_graphs_v2(
        current_epoch=24_745,
        select_rows=select,
        load_business_graphs=load,
    )

    assert [graph["root_receipt_hash"] for graph in graphs] == sorted(
        (provenance_receipt, decision_receipt)
    )
    intent_query = next(
        kwargs for table, kwargs in calls if table.endswith("reward_intents")
    )
    reward_query = next(
        kwargs for table, kwargs in calls if table.endswith("reward_current")
    )
    assert ("intent_status", "in", ["leased", "queued", "retry_wait"]) in intent_query[
        "filters"
    ]
    assert not any(field == "start_epoch" for field, *_rest in reward_query["filters"])


@pytest.mark.asyncio
async def test_source_add_active_graphs_fail_closed_without_exact_authority() -> None:
    intent = {
        "intent_id": "source_add_reward_intent:2222222222222222",
        "submission_id": "source_add_submission:1111111111111111",
        "adapter_id": "adapter-pending",
        "miner_hotkey": VALIDATOR_HOTKEY,
        "leg": 1,
        "intent_status": "leased",
        "approval_kind": "provenance_precheck_passed",
        "provenance_receipt_hash": _hash("7"),
        "provenance_artifact_hash": _hash("8"),
    }

    async def select(table, **_kwargs):
        if table == "research_lab_source_add_reward_intents":
            return [intent]
        return []

    async def load(_artifacts):
        raise AssertionError("unproved SOURCE_ADD authority must not load a graph")

    with pytest.raises(
        prepare.PrepareActiveReleaseLineageV2Error,
        match="missing or ambiguous",
    ):
        await prepare._load_active_source_add_graphs_v2(
            current_epoch=24_745,
            select_rows=select,
            load_business_graphs=load,
        )


@pytest.mark.asyncio
async def test_source_add_active_graphs_reject_filtered_row_drift() -> None:
    intent = {
        "intent_id": "source_add_reward_intent:2222222222222222",
        "submission_id": "source_add_submission:1111111111111111",
        "adapter_id": "adapter-pending",
        "miner_hotkey": VALIDATOR_HOTKEY,
        "leg": 1,
        "intent_status": "cancelled",
        "approval_kind": "provenance_precheck_passed",
        "provenance_receipt_hash": _hash("7"),
        "provenance_artifact_hash": _hash("8"),
    }

    async def select(table, **_kwargs):
        if table == "research_lab_source_add_reward_intents":
            return [intent]
        return []

    with pytest.raises(
        prepare.PrepareActiveReleaseLineageV2Error,
        match="differs from provenance authority",
    ):
        await prepare._load_active_source_add_graphs_v2(
            current_epoch=24_745,
            select_rows=select,
            load_business_graphs=lambda _artifacts: {},
        )


def test_validator_initial_freezes_exact_journal_commits_and_verifies_again(
    monkeypatch, tmp_path
) -> None:
    fetched = _patch_lineage_boundaries(monkeypatch)
    calls = []

    def extract(
        _journal,
        *,
        expected_lineage_id,
        expected_validator_hotkey,
        chain_profile,
        boot_verifier=None,
    ):
        assert chain_profile == CHAIN_PROFILE
        calls.append(
            (
                expected_lineage_id,
                expected_validator_hotkey,
                boot_verifier is not None,
            )
        )
        return {
            "journal_hash": _hash("6"),
            "required_commits": [JOURNAL_COMMIT],
        }

    from validator_tee.host import publication_journal_v2

    monkeypatch.setattr(
        publication_journal_v2,
        "publication_journal_release_requirements_v2",
        extract,
    )
    result = prepare.prepare_validator_initial_active_lineage_v2(
        candidate_commit_sha=CANDIDATE,
        authority_commit_sha=AUTHORITY,
        restart_invocation_id=RESTART_INVOCATION_ID,
        running_validator_commit_sha=RUNNING_VALIDATOR,
        expected_validator_hotkey=VALIDATOR_HOTKEY,
        chain_signing_profile=CHAIN_PROFILE,
        journal_loader=lambda: {"snapshot": "stable"},
        repository=tmp_path,
        expected_lineage_id=LINEAGE_ID,
    )

    expected = sorted({CANDIDATE, RUNNING_VALIDATOR, JOURNAL_COMMIT})
    assert fetched == [expected]
    assert calls == [
        (LINEAGE_ID, VALIDATOR_HOTKEY, False),
        (LINEAGE_ID, VALIDATOR_HOTKEY, True),
    ]
    assert result["requirements"]["required_commits"] == expected
    assert result["requirements"]["transition_commit_shas"] == sorted(
        {RUNNING_VALIDATOR, JOURNAL_COMMIT}
    )
    assert result["journal_hash"] == _hash("6")


def test_validator_initial_rejects_journal_release_set_drift(
    monkeypatch, tmp_path
) -> None:
    _patch_lineage_boundaries(monkeypatch)
    observations = iter(
        (
            [JOURNAL_COMMIT],
            [JOURNAL_COMMIT, NEW_JOURNAL_COMMIT],
        )
    )
    from validator_tee.host import publication_journal_v2

    monkeypatch.setattr(
        publication_journal_v2,
        "publication_journal_release_requirements_v2",
        lambda _journal, **_kwargs: {
            "journal_hash": _hash("6"),
            "required_commits": next(observations),
        },
    )

    with pytest.raises(
        prepare.PrepareActiveReleaseLineageV2Error,
        match="changed during release selection",
    ):
        prepare.prepare_validator_initial_active_lineage_v2(
            candidate_commit_sha=CANDIDATE,
            authority_commit_sha=AUTHORITY,
            restart_invocation_id=RESTART_INVOCATION_ID,
            running_validator_commit_sha=RUNNING_VALIDATOR,
            expected_validator_hotkey=VALIDATOR_HOTKEY,
            chain_signing_profile=CHAIN_PROFILE,
            journal_loader=lambda: {"snapshot": "moving"},
            repository=tmp_path,
            expected_lineage_id=LINEAGE_ID,
        )


def test_validator_initial_rejects_same_release_set_journal_hash_drift(
    monkeypatch, tmp_path
) -> None:
    _patch_lineage_boundaries(monkeypatch)
    hashes = iter((_hash("6"), _hash("7")))
    from validator_tee.host import publication_journal_v2

    monkeypatch.setattr(
        publication_journal_v2,
        "publication_journal_release_requirements_v2",
        lambda _journal, **_kwargs: {
            "journal_hash": next(hashes),
            "required_commits": [JOURNAL_COMMIT],
        },
    )

    with pytest.raises(
        prepare.PrepareActiveReleaseLineageV2Error,
        match="journal changed",
    ):
        prepare.prepare_validator_initial_active_lineage_v2(
            candidate_commit_sha=CANDIDATE,
            authority_commit_sha=AUTHORITY,
            restart_invocation_id=RESTART_INVOCATION_ID,
            running_validator_commit_sha=RUNNING_VALIDATOR,
            expected_validator_hotkey=VALIDATOR_HOTKEY,
            chain_signing_profile=CHAIN_PROFILE,
            journal_loader=lambda: {"snapshot": "moving"},
            repository=tmp_path,
            expected_lineage_id=LINEAGE_ID,
        )


def test_exact_lineage_fetch_uses_required_get_path_and_frozen_authority_ancestry(
    monkeypatch, tmp_path
) -> None:
    required = sorted({CANDIDATE, RUNNING_GATEWAY, JOURNAL_COMMIT})
    observed = {}
    ancestry_calls = []

    def ancestry(**kwargs):
        ancestry_calls.append(kwargs)
        return (AUTHORITY, CANDIDATE, RUNNING_GATEWAY, JOURNAL_COMMIT)

    monkeypatch.setattr(prepare, "git_ancestor_commits_v2", ancestry)

    def fetch(**kwargs):
        observed.update(kwargs)
        return _lineage(required)

    monkeypatch.setattr(prepare, "fetch_release_lineage_v2", fetch)
    result = prepare._fetch_exact_release_lineage_v2(
        candidate_commit_sha=CANDIDATE,
        authority_commit_sha=AUTHORITY,
        required_commits=required,
        repository=tmp_path,
        bucket="immutable-bucket",
        prefix="attested/releases",
    )

    assert result["lineage_hash"] == _lineage(required)["lineage_hash"]
    assert observed["required_commits"] == required
    assert ancestry_calls == [{"repository": tmp_path, "current_commit": AUTHORITY}]
    assert set(observed["allowed_commits"]) == {AUTHORITY, *required}
    assert observed["bucket"] == "immutable-bucket"
    assert observed["prefix"] == "attested/releases"


def test_exact_lineage_fetch_rejects_nonancestor_before_remote_read(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(
        prepare,
        "git_ancestor_commits_v2",
        lambda **_kwargs: (CANDIDATE,),
    )
    remote_called = False

    def fetch(**_kwargs):
        nonlocal remote_called
        remote_called = True
        raise AssertionError("remote fetch must not run")

    monkeypatch.setattr(prepare, "fetch_release_lineage_v2", fetch)
    with pytest.raises(
        prepare.PrepareActiveReleaseLineageV2Error,
        match="outside release authority Git ancestry",
    ):
        prepare._fetch_exact_release_lineage_v2(
            candidate_commit_sha=CANDIDATE,
            authority_commit_sha=AUTHORITY,
            required_commits=sorted({CANDIDATE, JOURNAL_COMMIT}),
            repository=tmp_path,
            bucket="immutable-bucket",
            prefix="attested/releases",
        )
    assert remote_called is False


@pytest.mark.asyncio
async def test_gateway_final_reselects_frozen_epoch_and_unions_validator_authority(
    monkeypatch, tmp_path
) -> None:
    fetched = _patch_lineage_boundaries(monkeypatch)
    monkeypatch.setattr(
        prepare,
        "validate_prior_release_manifest",
        lambda value: dict(value),
    )
    initial = _requirements(transitions=(RUNNING_VALIDATOR, JOURNAL_COMMIT))
    allocation_calls = []
    sourcing_calls = []
    source_add_calls = []

    async def allocations(**kwargs):
        allocation_calls.append(kwargs)
        return []

    async def sourcing(**kwargs):
        sourcing_calls.append(kwargs)
        return []

    async def source_add(**kwargs):
        source_add_calls.append(kwargs)
        return []

    result = await prepare.prepare_gateway_final_active_lineage_v2(
        candidate_commit_sha=CANDIDATE,
        authority_commit_sha=AUTHORITY,
        restart_invocation_id=RESTART_INVOCATION_ID,
        running_gateway_release_manifest={"commit_sha": RUNNING_GATEWAY},
        validator_requirements=initial,
        epoch_id=24_745,
        netuid=71,
        policy={"policy": "exact"},
        repository=tmp_path,
        expected_lineage_id=LINEAGE_ID,
        load_allocation_graphs=allocations,
        load_sourcing_graphs=sourcing,
        load_source_add_graphs=source_add,
    )

    expected = sorted({CANDIDATE, RUNNING_GATEWAY, RUNNING_VALIDATOR, JOURNAL_COMMIT})
    assert fetched == [expected]
    assert result["requirements"]["required_commits"] == expected
    assert len(allocation_calls) == len(sourcing_calls) == len(source_add_calls) == 2
    assert {call["epoch_id"] for call in allocation_calls} == {24_745}
    assert {call["current_epoch"] for call in sourcing_calls} == {24_745}
    assert {call["window"] for call in sourcing_calls} == {30}
    assert {call["current_epoch"] for call in source_add_calls} == {24_745}


def test_gateway_final_uses_bound_validator_authority_after_n_minus_one_unsets_env(
) -> None:
    restart = (Path(__file__).resolve().parents[1] / "gw_restart.sh").read_text(
        encoding="utf-8"
    )
    bootstrap_start = restart.index(
        '  exec env \\\n    -u GATEWAY_RESTART_AUTHORITY_ROOT'
    )
    bootstrap = restart[
        bootstrap_start : restart.index("\nfi\n", bootstrap_start)
    ]
    assert "-u GATEWAY_RESTART_AUTHORITY_COMMIT" in bootstrap

    advanced_main = "9" * 40
    validator_authority = "8" * 40
    requirements = _requirements(authority=validator_authority)

    assert prepare._gateway_final_authority_commit(
        advanced_main,
        requirements,
    ) == validator_authority
    assert prepare._gateway_final_authority_commit(
        advanced_main,
        None,
    ) == advanced_main


@pytest.mark.asyncio
async def test_gateway_final_accepts_exact_legacy_running_topology(
    monkeypatch,
    tmp_path,
) -> None:
    from tests.test_release_channel_v2 import _historical_gateway_manifest

    fetched = _patch_lineage_boundaries(monkeypatch)

    async def empty(**_kwargs):
        return []

    initial = _requirements(transitions=(RUNNING_VALIDATOR,))
    result = await prepare.prepare_gateway_final_active_lineage_v2(
        candidate_commit_sha=CANDIDATE,
        authority_commit_sha=AUTHORITY,
        restart_invocation_id=RESTART_INVOCATION_ID,
        running_gateway_release_manifest=_historical_gateway_manifest(
            RUNNING_GATEWAY
        ),
        validator_requirements=initial,
        epoch_id=24_745,
        netuid=71,
        policy={},
        repository=tmp_path,
        expected_lineage_id=LINEAGE_ID,
        load_allocation_graphs=empty,
        load_sourcing_graphs=empty,
        load_source_add_graphs=empty,
    )

    expected = sorted(
        {CANDIDATE, RUNNING_GATEWAY, RUNNING_VALIDATOR}
    )
    assert fetched == [expected]
    assert result["requirements"]["required_commits"] == expected


@pytest.mark.asyncio
async def test_gateway_final_explicit_standalone_fallback_unions_installed_lineage(
    monkeypatch, tmp_path
) -> None:
    fetched = _patch_lineage_boundaries(monkeypatch)
    monkeypatch.setattr(
        prepare,
        "validate_prior_release_manifest",
        lambda value: dict(value),
    )
    fallback = _lineage(
        (RUNNING_GATEWAY, JOURNAL_COMMIT),
        current_commit=RUNNING_GATEWAY,
    )

    async def empty(**_kwargs):
        return []

    result = await prepare.prepare_gateway_final_active_lineage_v2(
        candidate_commit_sha=CANDIDATE,
        authority_commit_sha=AUTHORITY,
        restart_invocation_id=RESTART_INVOCATION_ID,
        running_gateway_release_manifest={
            "commit_sha": RUNNING_GATEWAY,
            "release_hash": _hash("5"),
        },
        fallback_lineage=fallback,
        fallback_context="standalone",
        epoch_id=24_745,
        netuid=71,
        policy={},
        repository=tmp_path,
        expected_lineage_id=LINEAGE_ID,
        load_allocation_graphs=empty,
        load_sourcing_graphs=empty,
        load_source_add_graphs=empty,
    )

    expected = sorted({CANDIDATE, RUNNING_GATEWAY, JOURNAL_COMMIT})
    assert fetched == [expected]
    assert result["requirements"]["required_commits"] == expected
    assert result["requirements"]["authority_commit_sha"] == AUTHORITY
    assert result["requirements"]["restart_invocation_id"] == RESTART_INVOCATION_ID


@pytest.mark.asyncio
async def test_gateway_final_fallback_fails_closed_when_union_exceeds_bound(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(
        prepare,
        "validate_prior_release_manifest",
        lambda value: dict(value),
    )
    installed = {RUNNING_GATEWAY}
    installed.update("%040x" % index for index in range(511))
    fallback = _lineage(installed, current_commit=RUNNING_GATEWAY)

    async def empty(**_kwargs):
        return []

    with pytest.raises(ActiveReleaseRequirementsV2Error, match="exceed bound"):
        await prepare.prepare_gateway_final_active_lineage_v2(
            candidate_commit_sha=CANDIDATE,
            authority_commit_sha=AUTHORITY,
            restart_invocation_id=RESTART_INVOCATION_ID,
            running_gateway_release_manifest={
                "commit_sha": RUNNING_GATEWAY,
                "release_hash": _hash("5"),
            },
            fallback_lineage=fallback,
            fallback_context="full-parity",
            epoch_id=24_745,
            netuid=71,
            policy={},
            repository=tmp_path,
            expected_lineage_id=LINEAGE_ID,
            load_allocation_graphs=empty,
            load_sourcing_graphs=empty,
            load_source_add_graphs=empty,
        )


@pytest.mark.asyncio
async def test_gateway_final_rejects_implicit_or_paired_fallback_context(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(
        prepare,
        "validate_prior_release_manifest",
        lambda value: dict(value),
    )
    fallback = _lineage(
        (RUNNING_GATEWAY,),
        current_commit=RUNNING_GATEWAY,
    )

    async def empty(**_kwargs):
        return []

    with pytest.raises(
        prepare.PrepareActiveReleaseLineageV2Error,
        match="explicit safe context",
    ):
        await prepare.prepare_gateway_final_active_lineage_v2(
            candidate_commit_sha=CANDIDATE,
            authority_commit_sha=AUTHORITY,
            restart_invocation_id=RESTART_INVOCATION_ID,
            running_gateway_release_manifest={
                "commit_sha": RUNNING_GATEWAY,
                "release_hash": _hash("5"),
            },
            fallback_lineage=fallback,
            epoch_id=24_745,
            netuid=71,
            policy={},
            repository=tmp_path,
            expected_lineage_id=LINEAGE_ID,
            load_allocation_graphs=empty,
            load_sourcing_graphs=empty,
        )

    with pytest.raises(
        prepare.PrepareActiveReleaseLineageV2Error,
        match="cannot use fallback context",
    ):
        await prepare.prepare_gateway_final_active_lineage_v2(
            candidate_commit_sha=CANDIDATE,
            authority_commit_sha=AUTHORITY,
            restart_invocation_id=RESTART_INVOCATION_ID,
            running_gateway_release_manifest={"commit_sha": RUNNING_GATEWAY},
            validator_requirements=_requirements(),
            fallback_context="standalone",
            epoch_id=24_745,
            netuid=71,
            policy={},
            repository=tmp_path,
            expected_lineage_id=LINEAGE_ID,
            load_allocation_graphs=empty,
            load_sourcing_graphs=empty,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("requirements", "message"),
    (
        (_requirements(authority="9" * 40), "another release authority"),
        (
            _requirements(invocation_id="gateway-24745-other"),
            "another restart invocation",
        ),
    ),
)
async def test_gateway_final_rejects_cross_invocation_or_authority_sidecar(
    monkeypatch, tmp_path, requirements, message
) -> None:
    monkeypatch.setattr(
        prepare,
        "validate_prior_release_manifest",
        lambda value: dict(value),
    )

    with pytest.raises(
        prepare.PrepareActiveReleaseLineageV2Error,
        match=message,
    ):
        await prepare.prepare_gateway_final_active_lineage_v2(
            candidate_commit_sha=CANDIDATE,
            authority_commit_sha=AUTHORITY,
            restart_invocation_id=RESTART_INVOCATION_ID,
            running_gateway_release_manifest={"commit_sha": RUNNING_GATEWAY},
            validator_requirements=requirements,
            epoch_id=24_745,
            netuid=71,
            policy={},
            repository=tmp_path,
            expected_lineage_id=LINEAGE_ID,
        )


@pytest.mark.asyncio
async def test_gateway_final_rejects_active_root_drift(monkeypatch, tmp_path) -> None:
    _patch_lineage_boundaries(monkeypatch)
    monkeypatch.setattr(
        prepare,
        "validate_prior_release_manifest",
        lambda value: dict(value),
    )
    initial = _requirements(transitions=(RUNNING_VALIDATOR,))
    from gateway.tee import bootstrap_active_ancestry_checkpoints_v2 as bootstrap

    selections = iter(({}, {_hash("7"): {"root_receipt_hash": _hash("7")}}))

    async def select(**_kwargs):
        return next(selections)

    monkeypatch.setattr(bootstrap, "_select_active_graphs", select)

    with pytest.raises(
        prepare.PrepareActiveReleaseLineageV2Error,
        match="selection changed",
    ):
        await prepare.prepare_gateway_final_active_lineage_v2(
            candidate_commit_sha=CANDIDATE,
            authority_commit_sha=AUTHORITY,
            restart_invocation_id=RESTART_INVOCATION_ID,
            running_gateway_release_manifest={"commit_sha": RUNNING_GATEWAY},
            validator_requirements=initial,
            epoch_id=24_745,
            netuid=71,
            policy={},
            repository=tmp_path,
            expected_lineage_id=LINEAGE_ID,
            load_allocation_graphs=lambda **_kwargs: [],
            load_sourcing_graphs=lambda **_kwargs: [],
        )


def test_validator_final_independently_rebuilds_and_allows_cleared_journal(
    monkeypatch, tmp_path
) -> None:
    fetched = _patch_lineage_boundaries(monkeypatch)
    initial = _requirements(transitions=(RUNNING_VALIDATOR, JOURNAL_COMMIT))
    final = _requirements(
        transitions=tuple(
            sorted(
                {
                    *initial["required_commits"],
                    RUNNING_GATEWAY,
                }
            )
        )
    )
    handed = _lineage(final["required_commits"])
    from validator_tee.host import publication_journal_v2

    monkeypatch.setattr(
        publication_journal_v2,
        "publication_journal_release_requirements_v2",
        lambda journal, **_kwargs: {
            "journal_hash": None if journal is None else _hash("8"),
            "required_commits": [],
        },
    )

    result = prepare.prepare_validator_final_active_lineage_v2(
        candidate_commit_sha=CANDIDATE,
        authority_commit_sha=AUTHORITY,
        restart_invocation_id=RESTART_INVOCATION_ID,
        initial_requirements=initial,
        final_requirements=final,
        handed_lineage=handed,
        journal_loader=lambda: None,
        expected_validator_hotkey=VALIDATOR_HOTKEY,
        chain_signing_profile=CHAIN_PROFILE,
        repository=tmp_path,
        expected_lineage_id=LINEAGE_ID,
    )

    assert fetched == [final["required_commits"]]
    assert result["requirements"] == final
    assert result["lineage"] == handed
    assert result["journal_hash"] is None


def test_validator_final_rejects_new_uncovered_journal_release(
    monkeypatch, tmp_path
) -> None:
    _patch_lineage_boundaries(monkeypatch)
    initial = _requirements(transitions=(RUNNING_VALIDATOR,))
    final = _requirements(
        transitions=tuple(sorted({*initial["required_commits"], RUNNING_GATEWAY}))
    )
    from validator_tee.host import publication_journal_v2

    monkeypatch.setattr(
        publication_journal_v2,
        "publication_journal_release_requirements_v2",
        lambda _journal, **_kwargs: {
            "journal_hash": _hash("8"),
            "required_commits": [NEW_JOURNAL_COMMIT],
        },
    )

    with pytest.raises(
        prepare.PrepareActiveReleaseLineageV2Error,
        match="uncovered release",
    ):
        prepare.prepare_validator_final_active_lineage_v2(
            candidate_commit_sha=CANDIDATE,
            authority_commit_sha=AUTHORITY,
            restart_invocation_id=RESTART_INVOCATION_ID,
            initial_requirements=initial,
            final_requirements=final,
            handed_lineage=_lineage(final["required_commits"]),
            journal_loader=lambda: {"new": "publication"},
            expected_validator_hotkey=VALIDATOR_HOTKEY,
            chain_signing_profile=CHAIN_PROFILE,
            repository=tmp_path,
            expected_lineage_id=LINEAGE_ID,
        )


def _rehash_lineage(lineage):
    body = {key: lineage[key] for key in lineage if key != "lineage_hash"}
    lineage["lineage_hash"] = sha256_json(body)
    return lineage


def test_validator_final_accepts_prior_wrapper_drift_and_installs_handed_lineage(
    monkeypatch, tmp_path
) -> None:
    initial = _requirements(transitions=(RUNNING_VALIDATOR,))
    final = _requirements(
        transitions=tuple(sorted({*initial["required_commits"], RUNNING_GATEWAY}))
    )
    handed = _lineage(final["required_commits"])
    independent = json.loads(json.dumps(handed))
    independent["releases"][RUNNING_GATEWAY]["channel_hash"] = _hash("9")
    independent["releases"][RUNNING_GATEWAY]["gateway_release_hash"] = _hash("8")
    _rehash_lineage(independent)
    monkeypatch.setattr(
        prepare,
        "_fetch_exact_release_lineage_v2",
        lambda **_kwargs: independent,
    )
    from gateway.tee import release_lineage_v2

    verified_lineages = []
    original_compact_boot_verifier = prepare._compact_boot_verifier

    def compact_boot_verifier_spy(lineage):
        verified_lineages.append(lineage)
        return original_compact_boot_verifier(lineage)

    monkeypatch.setattr(
        prepare,
        "_compact_boot_verifier",
        compact_boot_verifier_spy,
    )
    nitro_calls = []

    def verify_nitro(identity, *, expected_pcr0, certificate_validity_at_attestation_time):
        assert expected_pcr0 == identity["pcr0"]
        assert certificate_validity_at_attestation_time is True
        nitro_calls.append((identity, expected_pcr0))
        return identity

    monkeypatch.setattr(release_lineage_v2, "verify_boot_identity_nitro", verify_nitro)

    result = prepare.prepare_validator_final_active_lineage_v2(
        candidate_commit_sha=CANDIDATE,
        authority_commit_sha=AUTHORITY,
        restart_invocation_id=RESTART_INVOCATION_ID,
        initial_requirements=initial,
        final_requirements=final,
        handed_lineage=handed,
        journal_loader=lambda: None,
        expected_validator_hotkey=VALIDATOR_HOTKEY,
        chain_signing_profile=CHAIN_PROFILE,
        repository=tmp_path,
        expected_lineage_id=LINEAGE_ID,
    )

    assert result["lineage"] == handed
    assert result["journal_hash"] is None
    assert verified_lineages == [handed]
    installed = tmp_path / "gateway-v2-release-lineage.json"
    prepare._atomic_json_documents(((installed, result["lineage"]),))
    installed_lineage = prepare._validate_selected_lineage(
        json.loads(installed.read_text(encoding="utf-8")),
        historical_topology_hash=None,
        expected_current_commit=CANDIDATE,
    )
    assert installed_lineage == handed

    role = sorted(handed["releases"][RUNNING_GATEWAY]["roles"])[0]
    prior_identity = {
        "role": role,
        "physical_role": role,
        **handed["releases"][RUNNING_GATEWAY]["roles"][role],
    }
    installed_boot_verifier = (
        release_lineage_v2.build_compact_release_lineage_boot_verifier_v2(
            installed_lineage
        )
    )
    assert installed_boot_verifier(prior_identity) == prior_identity
    assert nitro_calls == [(prior_identity, prior_identity["pcr0"])]

    altered_prior_identity = {**prior_identity, "pcr0": "9" * 96}
    with pytest.raises(
        release_lineage_v2.ReleaseLineageV2Error,
        match="boot pcr0 differs from compact release lineage",
    ):
        installed_boot_verifier(altered_prior_identity)
    assert nitro_calls == [(prior_identity, prior_identity["pcr0"])]


def test_validator_final_rejects_current_wrapper_drift(monkeypatch, tmp_path) -> None:
    initial = _requirements(transitions=(RUNNING_VALIDATOR,))
    final = _requirements(
        transitions=tuple(sorted({*initial["required_commits"], RUNNING_GATEWAY}))
    )
    handed = _lineage(final["required_commits"])
    independent = json.loads(json.dumps(handed))
    independent["releases"][CANDIDATE]["channel_hash"] = _hash("9")
    _rehash_lineage(independent)
    monkeypatch.setattr(
        prepare, "_fetch_exact_release_lineage_v2", lambda **_kwargs: independent
    )

    with pytest.raises(
        prepare.PrepareActiveReleaseLineageV2Error,
        match="current release differs",
    ):
        prepare.prepare_validator_final_active_lineage_v2(
            candidate_commit_sha=CANDIDATE,
            authority_commit_sha=AUTHORITY,
            restart_invocation_id=RESTART_INVOCATION_ID,
            initial_requirements=initial,
            final_requirements=final,
            handed_lineage=handed,
            journal_loader=lambda: None,
            expected_validator_hotkey=VALIDATOR_HOTKEY,
            chain_signing_profile=CHAIN_PROFILE,
            repository=tmp_path,
            expected_lineage_id=LINEAGE_ID,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("commit_sha", NEW_JOURNAL_COMMIT),
        ("pcr0", "9" * 96),
        ("build_manifest_hash", _hash("9")),
        ("dependency_lock_hash", _hash("9")),
    ],
)
def test_validator_final_rejects_prior_role_identity_drift(
    monkeypatch, tmp_path, field, value
) -> None:
    initial = _requirements(transitions=(RUNNING_VALIDATOR,))
    final = _requirements(
        transitions=tuple(sorted({*initial["required_commits"], RUNNING_GATEWAY}))
    )
    handed = _lineage(final["required_commits"])
    independent = json.loads(json.dumps(handed))
    role = sorted(independent["releases"][RUNNING_GATEWAY]["roles"])[0]
    independent["releases"][RUNNING_GATEWAY]["roles"][role][field] = value
    _rehash_lineage(independent)
    monkeypatch.setattr(
        prepare, "_fetch_exact_release_lineage_v2", lambda **_kwargs: independent
    )

    with pytest.raises(
        prepare.PrepareActiveReleaseLineageV2Error,
        match="prior release roles differ",
    ):
        prepare.prepare_validator_final_active_lineage_v2(
            candidate_commit_sha=CANDIDATE,
            authority_commit_sha=AUTHORITY,
            restart_invocation_id=RESTART_INVOCATION_ID,
            initial_requirements=initial,
            final_requirements=final,
            handed_lineage=handed,
            journal_loader=lambda: None,
            expected_validator_hotkey=VALIDATOR_HOTKEY,
            chain_signing_profile=CHAIN_PROFILE,
            repository=tmp_path,
            expected_lineage_id=LINEAGE_ID,
        )


def test_validator_final_rejects_independent_membership_drift(
    monkeypatch, tmp_path
) -> None:
    initial = _requirements(transitions=(RUNNING_VALIDATOR,))
    final = _requirements(
        transitions=tuple(sorted({*initial["required_commits"], RUNNING_GATEWAY}))
    )
    handed = _lineage(final["required_commits"])
    independent = json.loads(json.dumps(handed))
    independent["releases"].pop(RUNNING_GATEWAY)
    _rehash_lineage(independent)
    monkeypatch.setattr(
        prepare, "_fetch_exact_release_lineage_v2", lambda **_kwargs: independent
    )

    with pytest.raises(
        prepare.PrepareActiveReleaseLineageV2Error,
        match="membership differs",
    ):
        prepare.prepare_validator_final_active_lineage_v2(
            candidate_commit_sha=CANDIDATE,
            authority_commit_sha=AUTHORITY,
            restart_invocation_id=RESTART_INVOCATION_ID,
            initial_requirements=initial,
            final_requirements=final,
            handed_lineage=handed,
            journal_loader=lambda: None,
            expected_validator_hotkey=VALIDATOR_HOTKEY,
            chain_signing_profile=CHAIN_PROFILE,
            repository=tmp_path,
            expected_lineage_id=LINEAGE_ID,
        )


def test_atomic_outputs_are_canonical_owner_only_and_prepared_before_replace(
    tmp_path,
) -> None:
    first = tmp_path / "nested" / "requirements.json"
    second = tmp_path / "nested" / "lineage.json"
    prepare._atomic_json_documents(
        ((first, {"z": 2, "a": 1}), (second, {"value": [1, 2]}))
    )

    assert first.read_text(encoding="ascii") == '{"a":1,"z":2}\n'
    assert second.read_text(encoding="ascii") == '{"value":[1,2]}\n'
    assert os.stat(first).st_mode & 0o777 == 0o600
    assert os.stat(second).st_mode & 0o777 == 0o600
    assert not list(first.parent.glob(".*.json.*"))


def test_atomic_outputs_reject_oversize_before_replacing_any_destination(
    tmp_path,
) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text("first-original\n", encoding="ascii")
    second.write_text("second-original\n", encoding="ascii")

    with pytest.raises(
        prepare.PrepareActiveReleaseLineageV2Error,
        match="active release output exceeds sidecar byte bound",
    ):
        prepare._atomic_json_documents(
            (
                (first, {"small": True}),
                (
                    second,
                    {"padding": "x" * prepare._MAX_SIDECAR_JSON_INPUT_BYTES},
                ),
            )
        )

    assert first.read_text(encoding="ascii") == "first-original\n"
    assert second.read_text(encoding="ascii") == "second-original\n"
    assert not list(tmp_path.glob(".*.json.*"))


def test_json_loader_rejects_symlink_oversize_and_non_object(tmp_path) -> None:
    regular = tmp_path / "regular.json"
    regular.write_text('{"value":1}\n', encoding="utf-8")
    assert prepare._load_json(regular, "test input") == {"value": 1}

    linked = tmp_path / "linked.json"
    linked.symlink_to(regular)
    with pytest.raises(
        prepare.PrepareActiveReleaseLineageV2Error,
        match="unavailable or invalid",
    ):
        prepare._load_json(linked, "test input")
    with pytest.raises(
        prepare.PrepareActiveReleaseLineageV2Error,
        match="unavailable or invalid",
    ):
        prepare._load_optional_journal(linked)

    oversized = tmp_path / "oversized.json"
    oversized.write_bytes(b" " * (4 * 1024 * 1024 + 1))
    with pytest.raises(
        prepare.PrepareActiveReleaseLineageV2Error,
        match="bounded regular file",
    ):
        prepare._load_json(oversized, "test input")

    large_journal = tmp_path / "large-journal.json"
    large_journal.write_text(
        json.dumps({"padding": "x" * (4 * 1024 * 1024)}),
        encoding="utf-8",
    )
    assert prepare._load_optional_journal(large_journal) == {
        "padding": "x" * (4 * 1024 * 1024)
    }

    array = tmp_path / "array.json"
    array.write_text("[]\n", encoding="utf-8")
    with pytest.raises(
        prepare.PrepareActiveReleaseLineageV2Error,
        match="must be an object",
    ):
        prepare._load_json(array, "test input")


def test_validator_initial_cli_writes_only_the_sidecar(monkeypatch, tmp_path, capsys):
    requirements = _requirements(transitions=(RUNNING_VALIDATOR,))
    lineage = _lineage(requirements["required_commits"])
    monkeypatch.setattr(
        prepare,
        "prepare_validator_initial_active_lineage_v2",
        lambda **_kwargs: {
            "requirements": requirements,
            "lineage": lineage,
            "journal_hash": None,
        },
    )
    monkeypatch.setattr(
        prepare,
        "_load_validator_authority_context",
        lambda **_kwargs: {
            "validator_hotkey": VALIDATOR_HOTKEY,
            "chain_signing_profile": CHAIN_PROFILE,
        },
    )
    output = tmp_path / "validator-requirements.json"
    assert (
        prepare.main(
            [
                "--phase",
                "validator-initial",
                "--candidate-commit",
                CANDIDATE,
                "--authority-commit",
                AUTHORITY,
                "--restart-invocation-id",
                RESTART_INVOCATION_ID,
                "--running-validator-commit",
                RUNNING_VALIDATOR,
                "--journal",
                str(tmp_path / "journal.json"),
                "--validator-hotkey-config",
                str(tmp_path / "validator-hotkey.json"),
                "--chain-signing-profile",
                str(tmp_path / "chain-profile.json"),
                "--repository",
                str(tmp_path),
                "--lineage-id",
                LINEAGE_ID,
                "--requirements-output",
                str(output),
            ]
        )
        == 0
    )
    assert json.loads(output.read_text(encoding="utf-8")) == requirements
    marker = json.loads(capsys.readouterr().out)
    assert marker["mode"] == "validator-initial"
    assert marker["status"] == "complete"
    assert marker["selection_hash"] == requirements["selection_hash"]


@pytest.mark.parametrize(
    ("flag", "message"),
    (
        (
            "--validator-hotkey-config",
            "gateway-final cannot receive validator hotkey configuration",
        ),
        (
            "--chain-signing-profile",
            "gateway-final cannot receive validator chain signing profile",
        ),
    ),
)
def test_gateway_final_rejects_validator_authority_configuration(
    tmp_path,
    flag,
    message,
) -> None:
    with pytest.raises(
        prepare.PrepareActiveReleaseLineageV2Error,
        match=message,
    ):
        prepare.main(
            [
                "--phase",
                "gateway-final",
                "--candidate-commit",
                CANDIDATE,
                "--authority-commit",
                AUTHORITY,
                "--restart-invocation-id",
                RESTART_INVOCATION_ID,
                "--repository",
                str(tmp_path),
                "--lineage-id",
                LINEAGE_ID,
                flag,
                str(tmp_path / "validator-hotkey.json"),
            ]
        )


def test_validator_initial_requires_chain_profile_before_release_fetch(
    monkeypatch,
    tmp_path,
) -> None:
    remote_reads = []
    monkeypatch.setattr(
        prepare,
        "_fetch_exact_release_lineage_v2",
        lambda **kwargs: remote_reads.append(kwargs),
    )
    with pytest.raises(
        prepare.PrepareActiveReleaseLineageV2Error,
        match="requires validator authority configuration",
    ):
        prepare.main(
            [
                "--phase",
                "validator-initial",
                "--candidate-commit",
                CANDIDATE,
                "--authority-commit",
                AUTHORITY,
                "--restart-invocation-id",
                RESTART_INVOCATION_ID,
                "--running-validator-commit",
                RUNNING_VALIDATOR,
                "--journal",
                str(tmp_path / "journal.json"),
                "--validator-hotkey-config",
                str(tmp_path / "validator-hotkey.json"),
                "--repository",
                str(tmp_path),
                "--lineage-id",
                LINEAGE_ID,
                "--requirements-output",
                str(tmp_path / "requirements.json"),
            ]
        )
    assert remote_reads == []


def test_validator_chain_profile_must_match_hotkey_config_before_release_fetch(
    monkeypatch,
    tmp_path,
) -> None:
    from validator_tee.enclave.hotkey_authority_v2 import (
        HOTKEY_AUTHORITY_CONFIG_SCHEMA_VERSION,
        MEASURED_DRAND_LIBRARY_PATH,
    )

    profile_path = tmp_path / "chain-profile.json"
    profile_path.write_text(json.dumps(CHAIN_PROFILE), encoding="utf-8")
    config_path = tmp_path / "validator-hotkey.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": HOTKEY_AUTHORITY_CONFIG_SCHEMA_VERSION,
                "validator_hotkey": VALIDATOR_HOTKEY,
                "hotkey_public_key": "1" * 64,
                "chain_signing_profile_hash": _hash("0"),
                "drand_library_path": MEASURED_DRAND_LIBRARY_PATH,
                "drand_library_sha256": "2" * 64,
            }
        ),
        encoding="utf-8",
    )
    remote_reads = []
    monkeypatch.setattr(
        prepare,
        "_fetch_exact_release_lineage_v2",
        lambda **kwargs: remote_reads.append(kwargs),
    )

    with pytest.raises(
        prepare.PrepareActiveReleaseLineageV2Error,
        match="chain signing profile differs from hotkey configuration",
    ):
        prepare.main(
            [
                "--phase",
                "validator-initial",
                "--candidate-commit",
                CANDIDATE,
                "--authority-commit",
                AUTHORITY,
                "--restart-invocation-id",
                RESTART_INVOCATION_ID,
                "--running-validator-commit",
                RUNNING_VALIDATOR,
                "--journal",
                str(tmp_path / "journal.json"),
                "--validator-hotkey-config",
                str(config_path),
                "--chain-signing-profile",
                str(profile_path),
                "--repository",
                str(tmp_path),
                "--lineage-id",
                LINEAGE_ID,
                "--requirements-output",
                str(tmp_path / "requirements.json"),
            ]
        )
    assert remote_reads == []


def test_validator_initial_rejects_v5_hotkey_mismatch_before_release_fetch(
    monkeypatch,
    tmp_path,
) -> None:
    from validator_tee.host import publication_journal_v2 as journal_module

    remote_reads = []
    monkeypatch.setattr(
        prepare,
        "_fetch_exact_release_lineage_v2",
        lambda **kwargs: remote_reads.append(kwargs),
    )
    monkeypatch.setattr(
        journal_module,
        "validate_publication_journal_v2",
        lambda _journal, **_kwargs: {
            "schema_version": journal_module.COMPACT_JOURNAL_SCHEMA_VERSION,
            "compact_submission": {},
            "journal_hash": _hash("8"),
        },
    )

    def verify_compact(
        _compact,
        *,
        expected_lineage_id,
        expected_chain,
        identity_cache,
        boot_verifier,
    ):
        assert expected_lineage_id == LINEAGE_ID
        assert expected_chain == CHAIN_PROFILE["chain_endpoint"]
        assert identity_cache is None
        assert callable(boot_verifier)
        return {"validator_hotkey": VALIDATOR_HOTKEY}

    monkeypatch.setattr(
        journal_module,
        "verify_compact_weight_submission_v2",
        verify_compact,
    )

    with pytest.raises(
        journal_module.WeightPublicationJournalV2Error,
        match="another validator hotkey",
    ):
        prepare.prepare_validator_initial_active_lineage_v2(
            candidate_commit_sha=CANDIDATE,
            authority_commit_sha=AUTHORITY,
            restart_invocation_id=RESTART_INVOCATION_ID,
            running_validator_commit_sha=RUNNING_VALIDATOR,
            expected_validator_hotkey="5DifferentValidatorHotkey",
            chain_signing_profile=CHAIN_PROFILE,
            journal_loader=lambda: {"schema_version": "v5", "state": "prepared"},
            repository=tmp_path,
            expected_lineage_id=LINEAGE_ID,
        )

    assert remote_reads == []
