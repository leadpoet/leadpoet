from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import asyncio
import json
import threading
import time

import pytest

from Leadpoet.utils.subnet_epoch import (
    SubnetEpochCutover,
    SubnetEpochError,
    SubnetEpochSnapshot,
    validate_validator_shared_epoch_file,
)
from Leadpoet.validator import reward as reward_module
import neurons.validator as validator_module

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_SOURCE = ROOT / "neurons" / "validator.py"


def _cutover() -> SubnetEpochCutover:
    return SubnetEpochCutover(
        network_genesis_hash="0x" + "1" * 64,
        netuid=71,
        cutover_block=360,
        cutover_block_hash="0x" + "2" * 64,
        first_subnet_epoch_index=10,
        first_settlement_epoch_id=100,
        last_legacy_epoch_id=99,
    )


def _snapshot(*, block: int, last_epoch_block: int, index: int, head: str):
    return SubnetEpochSnapshot(
        network_genesis_hash="0x" + "1" * 64,
        netuid=71,
        head_kind=head,
        block_hash="0x" + ("3" if index == 10 else "4") * 64,
        current_block=block,
        last_epoch_block=last_epoch_block,
        pending_epoch_at=0,
        subnet_epoch_index=index,
        tempo=360,
        blocks_since_last_step=block - last_epoch_block,
        observed_at="2026-07-16T22:45:00Z",
    )


def _between(source: str, start: str, end: str) -> str:
    start_index = source.index(start)
    end_index = source.index(end, start_index)
    return source[start_index:end_index]


def test_research_lab_fallback_uses_shared_allocation_default():
    source = VALIDATOR_SOURCE.read_text(encoding="utf-8")
    snippet = _between(
        source,
        "RESEARCH_LAB_FALLBACK_SHARE = _env_percent_share(",
        "RESEARCH_LAB_SHARE = _doc_percent_share(",
    )

    assert "DEFAULT_RESEARCH_LAB_EMISSION_PERCENT" in snippet
    assert "20.0" not in snippet


def test_epoch_debug_line_includes_absolute_and_within_epoch_blocks():
    source = VALIDATOR_SOURCE.read_text(encoding="utf-8")
    snippet = _between(
        source,
        "# DEBUG: Always log epoch status",
        "# Check if we've already processed this epoch",
    )

    assert "Block: {current_block}" in snippet
    assert "Epoch block: {blocks_into_epoch}" in snippet
    assert "remaining: {epoch_state.blocks_remaining}" in snippet
    assert "subnet index: {epoch_state.subnet_epoch_index}" in snippet


def test_already_processed_epoch_still_checks_weight_submission():
    source = VALIDATOR_SOURCE.read_text(encoding="utf-8")
    snippet = _between(
        source,
        "if current_epoch <= self._last_processed_epoch:",
        'print(f"[DEBUG] Processing epoch {current_epoch} for the FIRST TIME")',
    )

    assert "await self._check_weight_submission_for_processed_epoch(" in snippet
    assert '"already_processed"' in snippet


def test_legacy_sourcing_disabled_path_checks_weight_submission_before_return():
    source = VALIDATOR_SOURCE.read_text(encoding="utf-8")
    snippet = _between(
        source,
        'if not _env_flag("ENABLE_LEGACY_SOURCING"):',
        "# Fetch assigned leads from gateway",
    )

    assert "await self._check_weight_submission_for_processed_epoch(" in snippet
    assert '"legacy_sourcing_disabled"' in snippet
    assert snippet.index("await self._check_weight_submission_for_processed_epoch(") < snippet.index(
        "self._last_processed_epoch = current_epoch"
    )


def test_gateway_already_submitted_path_checks_weight_submission_before_return():
    source = VALIDATOR_SOURCE.read_text(encoding="utf-8")
    snippet = _between(
        source,
        "if leads is None:",
        'print(f"[DEBUG] Received {len(leads)} leads from gateway',
    )

    assert "await self._check_weight_submission_for_processed_epoch(" in snippet
    assert '"gateway_already_submitted_or_queue_empty"' in snippet
    assert snippet.index("await self._check_weight_submission_for_processed_epoch(") < snippet.index(
        "self._last_processed_epoch = current_epoch"
    )


def test_reward_transition_waits_for_finalized_subnet_epoch(monkeypatch):
    cutover = _cutover()
    old_finalized = _snapshot(
        block=719,
        last_epoch_block=360,
        index=10,
        head="finalized",
    )
    best_after_boundary = _snapshot(
        block=720,
        last_epoch_block=720,
        index=11,
        head="best",
    )
    new_finalized = _snapshot(
        block=720,
        last_epoch_block=720,
        index=11,
        head="finalized",
    )
    observed = {"finalized": old_finalized, "calls": []}

    def read_snapshot(_source, *, netuid, finalized=False):
        observed["calls"].append(finalized)
        assert netuid == 71
        return observed["finalized"] if finalized else best_after_boundary

    monkeypatch.setattr(reward_module, "load_subnet_epoch_cutover", lambda: cutover)
    monkeypatch.setattr(
        reward_module,
        "validate_subnet_epoch_cutover_anchor",
        lambda _source, _cutover: None,
    )
    monkeypatch.setattr(reward_module, "read_subnet_epoch_snapshot", read_snapshot)
    monkeypatch.setattr(reward_module, "_current_epoch", None)
    monkeypatch.setattr(reward_module, "_current_subnet_epoch_index", None)
    monkeypatch.setattr(reward_module, "_epoch_start_block", None)
    monkeypatch.setattr(reward_module, "_validated_stateful_anchor_sources", set())
    monkeypatch.setattr(reward_module, "_stateful_archive_subtensor", object())

    source = object()
    first, _epoch = reward_module._read_stateful_epoch(source)
    assert reward_module._is_epoch_ended(
        first.current_block,
        epoch_snapshot=first,
    ) is False

    # The best head has rolled over, but the destructive clear gate reads the
    # unchanged finalized epoch and therefore stays closed.
    still_old, _epoch = reward_module._read_stateful_epoch(source)
    assert best_after_boundary.subnet_epoch_index == 11
    assert still_old.subnet_epoch_index == 10
    assert reward_module._is_epoch_ended(
        still_old.current_block,
        epoch_snapshot=still_old,
    ) is False

    observed["finalized"] = new_finalized
    committed, _epoch = reward_module._read_stateful_epoch(source)
    assert reward_module._is_epoch_ended(
        committed.current_block,
        epoch_snapshot=committed,
    ) is True
    assert observed["calls"] == [True, True, True]


def test_reward_stateful_current_block_never_uses_cached_estimation(monkeypatch):
    def unavailable():
        raise SubnetEpochError("finalized state unavailable")

    monkeypatch.setattr(reward_module, "_read_stateful_epoch", unavailable)
    with pytest.raises(SubnetEpochError, match="finalized state unavailable"):
        asyncio.run(reward_module._get_current_block_async())


def test_reward_stateful_restart_initializes_without_synthetic_transition(
    monkeypatch,
):
    cutover = _cutover()
    current = _snapshot(
        block=1_080,
        last_epoch_block=1_080,
        index=12,
        head="finalized",
    )
    monkeypatch.setattr(reward_module, "load_subnet_epoch_cutover", lambda: cutover)
    monkeypatch.setattr(reward_module, "_current_epoch", None)
    monkeypatch.setattr(reward_module, "_current_subnet_epoch_index", None)
    monkeypatch.setattr(reward_module, "_epoch_start_block", None)

    assert reward_module._is_epoch_ended(
        current.current_block,
        epoch_snapshot=current,
    ) is False
    assert reward_module._current_subnet_epoch_index == 12
    assert reward_module._current_epoch == 102


def test_reward_stateful_epoch_jump_emits_exactly_one_transition(monkeypatch):
    cutover = _cutover()
    old = _snapshot(
        block=719,
        last_epoch_block=360,
        index=10,
        head="finalized",
    )
    jumped = _snapshot(
        block=1_080,
        last_epoch_block=1_080,
        index=12,
        head="finalized",
    )
    monkeypatch.setattr(reward_module, "load_subnet_epoch_cutover", lambda: cutover)
    monkeypatch.setattr(reward_module, "_current_epoch", None)
    monkeypatch.setattr(reward_module, "_current_subnet_epoch_index", None)
    monkeypatch.setattr(reward_module, "_epoch_start_block", None)

    assert reward_module._is_epoch_ended(
        old.current_block,
        epoch_snapshot=old,
    ) is False
    assert reward_module._is_epoch_ended(
        jumped.current_block,
        epoch_snapshot=jumped,
    ) is True
    assert reward_module._is_epoch_ended(
        jumped.current_block,
        epoch_snapshot=jumped,
    ) is False


def test_reward_epoch_decision_rejects_mixed_chain_blocks(monkeypatch):
    snapshot = _snapshot(
        block=719,
        last_epoch_block=360,
        index=10,
        head="finalized",
    )
    monkeypatch.setattr(reward_module, "load_subnet_epoch_cutover", _cutover)
    with pytest.raises(RuntimeError, match="different chain blocks"):
        reward_module._is_epoch_ended(
            snapshot.current_block + 1,
            epoch_snapshot=snapshot,
        )


def test_reward_cutover_uses_archive_but_live_snapshot_uses_finney(monkeypatch):
    cutover = _cutover()
    finalized = _snapshot(
        block=719,
        last_epoch_block=360,
        index=10,
        head="finalized",
    )
    live_source = object()
    archive_source = object()
    observed = {}
    monkeypatch.setattr(reward_module, "load_subnet_epoch_cutover", lambda: cutover)
    monkeypatch.setattr(reward_module, "_stateful_archive_subtensor", archive_source)
    monkeypatch.setattr(reward_module, "_validated_stateful_anchor_sources", set())
    monkeypatch.setattr(
        reward_module,
        "validate_subnet_epoch_cutover_anchor",
        lambda source, manifest: observed.update(
            archive=(source, manifest),
        ),
    )
    monkeypatch.setattr(
        reward_module,
        "read_subnet_epoch_snapshot",
        lambda source, *, netuid, finalized: (
            observed.update(live=(source, netuid, finalized)) or finalized_snapshot
        ),
    )
    finalized_snapshot = finalized

    snapshot, settlement_epoch = reward_module._read_stateful_epoch(live_source)
    assert snapshot is finalized
    assert settlement_epoch == 100
    assert observed["archive"] == (archive_source, cutover)
    assert observed["live"] == (live_source, 71, True)


def test_reward_archive_validation_cache_is_scoped_to_mapping_hash(monkeypatch):
    cutovers = [
        _cutover(),
        SubnetEpochCutover(
            network_genesis_hash="0x" + "1" * 64,
            netuid=71,
            cutover_block=360,
            cutover_block_hash="0x" + "9" * 64,
            first_subnet_epoch_index=10,
            first_settlement_epoch_id=100,
            last_legacy_epoch_id=99,
        ),
    ]
    current = {"cutover": cutovers[0]}
    archive_source = object()
    validations = []
    snapshot = _snapshot(
        block=719,
        last_epoch_block=360,
        index=10,
        head="finalized",
    )
    monkeypatch.setattr(
        reward_module,
        "load_subnet_epoch_cutover",
        lambda: current["cutover"],
    )
    monkeypatch.setattr(reward_module, "_stateful_archive_subtensor", archive_source)
    monkeypatch.setattr(reward_module, "_validated_stateful_anchor_sources", set())
    monkeypatch.setattr(
        reward_module,
        "validate_subnet_epoch_cutover_anchor",
        lambda source, manifest: validations.append((source, manifest.mapping_hash)),
    )
    monkeypatch.setattr(
        reward_module,
        "read_subnet_epoch_snapshot",
        lambda *_args, **_kwargs: snapshot,
    )

    reward_module._read_stateful_epoch(object())
    reward_module._read_stateful_epoch(object())
    current["cutover"] = cutovers[1]
    reward_module._read_stateful_epoch(object())
    assert validations == [
        (archive_source, cutovers[0].mapping_hash),
        (archive_source, cutovers[1].mapping_hash),
    ]


def test_validator_irreversible_epoch_state_uses_finalized_head(monkeypatch):
    cutover = _cutover()
    finalized_snapshot = _snapshot(
        block=719,
        last_epoch_block=360,
        index=10,
        head="finalized",
    )
    calls = []

    def read_snapshot(source, *, netuid, finalized=False):
        calls.append((source, netuid, finalized))
        return finalized_snapshot

    monkeypatch.setattr(
        validator_module,
        "read_subnet_epoch_snapshot",
        read_snapshot,
    )
    validator = validator_module.Validator.__new__(validator_module.Validator)
    validator.config = SimpleNamespace(netuid=71)
    validator.subtensor = object()
    validator._epoch_cutover = cutover
    validator._epoch_snapshot_lock = threading.Lock()
    monkeypatch.setattr(
        validator,
        "_validate_durable_epoch_runtime_lifecycle",
        lambda *, force_refresh: {
            "lifecycle_state": "active",
            "mapping_hash": cutover.mapping_hash,
        },
    )

    state = validator._read_epoch_state_sync()
    assert state.workflow_epoch_id == 100
    assert state.subnet_epoch_index == 10
    assert calls == [(validator.subtensor, 71, True)]


def test_validator_submission_liveness_reads_best_head_without_replacing_authority(
    monkeypatch,
):
    cutover = _cutover()
    best_snapshot = _snapshot(
        block=719,
        last_epoch_block=360,
        index=10,
        head="best",
    )
    calls = []

    def read_snapshot(source, *, netuid, finalized=False):
        calls.append((source, netuid, finalized))
        return best_snapshot

    monkeypatch.setattr(
        validator_module,
        "read_subnet_epoch_snapshot",
        read_snapshot,
    )
    validator = validator_module.Validator.__new__(validator_module.Validator)
    validator.config = SimpleNamespace(netuid=71)
    validator.subtensor = object()
    validator._epoch_cutover = cutover
    validator._epoch_snapshot_lock = threading.Lock()

    state = validator._read_best_epoch_state_sync()
    assert state.workflow_epoch_id == 100
    assert state.subnet_epoch_index == 10
    assert state.snapshot.head_kind == "best"
    assert calls == [(validator.subtensor, 71, False)]


def test_shared_stateful_epoch_file_rejects_tampered_derived_authority(
    monkeypatch,
    tmp_path,
):
    cutover = _cutover()
    snapshot = _snapshot(
        block=719,
        last_epoch_block=360,
        index=10,
        head="finalized",
    )
    state = validator_module._ValidatorEpochState.from_snapshot(
        snapshot,
        cutover,
    )
    monkeypatch.setattr(
        validator_module,
        "load_subnet_epoch_cutover",
        lambda: cutover,
    )
    document = state.to_shared_document()
    document["timestamp"] = int(time.time())
    document["authority"]["blocks_remaining"] += 1
    path = tmp_path / "validator_weights" / "current_block.json"
    path.parent.mkdir()
    path.write_text(json.dumps(document), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SubnetEpochError, match="authority is not canonical"):
        validator_module._read_shared_epoch_state_file(max_age_seconds=30)


def test_shared_epoch_file_is_bound_to_runtime_generation(
    monkeypatch,
    tmp_path,
):
    cutover = _cutover()
    snapshot = _snapshot(
        block=719,
        last_epoch_block=360,
        index=10,
        head="finalized",
    )
    state = validator_module._ValidatorEpochState.from_snapshot(snapshot, cutover)
    monkeypatch.setattr(
        validator_module,
        "load_subnet_epoch_cutover",
        lambda: cutover,
    )
    monkeypatch.setenv("VALIDATOR_RUNTIME_GENERATION", "generation-current")
    document = state.to_shared_document()
    document["timestamp"] = int(time.time())
    path = tmp_path / "current_block.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(SubnetEpochError, match="runtime generation"):
        validate_validator_shared_epoch_file(
            path,
            max_age_seconds=30,
            cutover=cutover,
            expected_runtime_generation="generation-old",
        )
    assert validate_validator_shared_epoch_file(
        path,
        max_age_seconds=30,
        cutover=cutover,
        expected_runtime_generation="generation-current",
    ) == snapshot
