"""The Lab Arena reward adapter (labarena.md 13.4) end to end, without a chain.

One kernel (``leadpoet_canonical.lab_arena_rewards``) derives the king's share
from a signed reward basis. The Arena signs the basis with its own signer;
the validator proposes the champion triple from the basis the gateway serves;
the canonical weight computation refuses a triple that differs from the basis
it names; the coordinator re-derives the triple from the measured view row and
the pinned Arena key. With rewards off, every document is byte-identical.
"""

from __future__ import annotations

import copy
import os
from types import SimpleNamespace
from typing import Any, Dict

import pytest

from lab_arena import contracts, rewards
from lab_arena.signing import LocalSigner, sign_document, signing_key_document
from leadpoet_canonical import lab_arena_rewards as kernel
from lab_arena.contracts import canonical_json
from leadpoet_canonical.weight_computation import (
    WEIGHT_SNAPSHOT_SCHEMA_VERSION,
    WeightComputationError,
    compute_final_weights,
    weight_config_hash,
)
from leadpoet_canonical.weight_authority_v2 import gateway_weight_input_value_documents_v2

BURN = "5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM"
KING = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
HOTKEYS = [BURN, "fulfillment-hotkey", "lab-hotkey", KING]
EPOCH = 24810


def _sha(label: str) -> str:
    return contracts.document_hash(label)


def signed_basis(signer: LocalSigner, *, pool_percent: int = 25, effective: int = 24801, start: int = 24801, outcome: str = "crowned", hotkey: str = KING) -> Dict[str, Any]:
    basis = rewards.reward_basis_document(
        round_id="arena-2026-09-02",
        published_at="2026-09-02T10:00:00Z", finalized_epoch=effective - 1, king_outcome=outcome, king_hotkey="" if outcome == "no_king" else hotkey,
        previous_king_start_epoch=None if outcome in ("crowned", "no_king") else start, reward_constants=rewards.reward_constants_document(pool_percent),
    )
    return sign_document(signer, basis, hash_field="reward_basis_hash")


def snapshot(**overrides: Any) -> Dict[str, Any]:
    value = {
        "schema_version": WEIGHT_SNAPSHOT_SCHEMA_VERSION, "netuid": 71, "epoch_id": EPOCH, "block": 36099, "commit_sha": "a" * 40, "config_hash": "",
        "parent_receipt_hashes": [], "research_lab_allocation_receipt_hash": "", "burn_target_uid": 0, "expected_burn_target_hotkey": BURN,
        "metagraph_hotkeys": list(HOTKEYS), "banned_hotkeys": [], "banned_lookup_ok": True, "ff_enabled": True, "base_burn_share": 0.0,
        "champion_share": 0.0, "champion_uid": None, "effective_champion_share": 0.0, "research_lab_fallback_share": 0.3,
        "research_lab_allocation_doc": {"lab_cap_percent": 30.0, "unallocated_percent": 25.0, "reimbursement_allocations": [], "champion_allocations": [], "queued_champion_allocations": []},
        "leaderboard_bonus_share": 0.095, "leaderboard_rank_shares": [0.05, 0.03, 0.015], "leaderboard_entries": [], "leaderboard_fetch_ok": True,
        "fulfillment_share": 0.605, "fulfillment_rows": [{"hotkey": "fulfillment-hotkey", "share": 0.605}], "fulfillment_fetch_ok": True,
        "rolling_lead_count": 0, "rolling_scores": [], "sourcing_floor_threshold": 125_000, "min_total_rep_for_distribution": 100,
    }
    value.update(overrides)
    value["config_hash"] = weight_config_hash(value)
    return value


def proposed_snapshot(basis: Dict[str, Any], **overrides: Any) -> Dict[str, Any]:
    values = kernel.champion_values(basis, EPOCH, HOTKEYS)
    return snapshot(champion_share=values["champion_share"], effective_champion_share=values["effective_champion_share"], champion_uid=values["champion_uid"], lab_arena_reward_basis=basis, **overrides)


def weight_of(result: Dict[str, Any], uid: int) -> float:
    return result["weights"][result["uids"].index(uid)] if uid in result["uids"] else 0.0


@pytest.fixture(scope="module")
def signer() -> LocalSigner:
    return LocalSigner.generate()


# ---------------------------------------------------------------------------
# The kernel: constants from the basis, exact shares, signature
# ---------------------------------------------------------------------------


def test_pool_percent_is_the_one_adjustable_setting_and_shares_stay_exact(signer):
    for percent, week_shares in ((25, [0.25, 0.2, 0.15, 0.1, 0.05]), (5, [0.05, 0.04, 0.03, 0.02, 0.01]), (50, [0.5, 0.4, 0.3, 0.2, 0.1])):
        constants = rewards.reward_constants_document(percent)
        assert [kernel.champion_share_for_week(week, constants) for week in range(5)] == week_shares
        basis = signed_basis(signer, pool_percent=percent)
        assert kernel.champion_values(basis, EPOCH, HOTKEYS)["champion_share"] == week_shares[0]
    with pytest.raises(kernel.LabArenaRewardError):
        rewards.reward_constants_document(101)
    with pytest.raises(kernel.LabArenaRewardError):
        kernel.validate_reward_constants(dict(rewards.reward_constants_document(), pool_basis="fulfillment_residual"))
    with pytest.raises(kernel.LabArenaRewardError):
        kernel.validate_reward_constants(dict(rewards.reward_constants_document(), king_pool_share_percent_by_week=[]))


def test_arena_signature_verifies_only_against_the_pinned_key(signer):
    basis = signed_basis(signer)
    key_document = signing_key_document(signer.public_key_der)
    der = kernel.signing_key_from_document(key_document, signer.public_key_hash)
    assert kernel.verify_reward_basis_signature(basis, public_key_der=der, expected_public_key_hash=signer.public_key_hash) == basis["reward_basis_hash"]
    other = LocalSigner.generate()
    with pytest.raises(kernel.LabArenaRewardError, match="pinned key hash"):
        kernel.signing_key_from_document(key_document, other.public_key_hash)
    with pytest.raises(kernel.LabArenaRewardError):
        kernel.verify_reward_basis_signature(basis, public_key_der=other.public_key_der, expected_public_key_hash=other.public_key_hash)
    tampered = dict(basis, king_start_epoch=24800)
    with pytest.raises(kernel.LabArenaRewardError, match="does not match"):
        kernel.verify_reward_basis_signature(tampered, public_key_der=der, expected_public_key_hash=signer.public_key_hash)
    resigned = sign_document(signer, contracts.hashed_document(tampered, "reward_basis_hash"), hash_field="reward_basis_hash")
    assert kernel.verify_reward_basis_signature(resigned, public_key_der=der, expected_public_key_hash=signer.public_key_hash) != basis["reward_basis_hash"]
    unsigned = {k: v for k, v in basis.items() if k != "signature"}
    with pytest.raises(kernel.LabArenaRewardError, match="unsigned"):
        kernel.verify_reward_basis_signature(unsigned, public_key_der=der, expected_public_key_hash=signer.public_key_hash)
    with pytest.raises(kernel.LabArenaRewardError):
        kernel.signing_key_hash_from_environment({})
    assert kernel.signing_key_hash_from_environment({kernel.SIGNING_KEY_HASH_ENV: signer.public_key_hash.upper()}) == signer.public_key_hash
    assert kernel.rewards_enabled_from_environment({kernel.REWARDS_ENABLED_ENV: "true"}) and not kernel.rewards_enabled_from_environment({})


# ---------------------------------------------------------------------------
# The canonical weight computation with a basis in the snapshot
# ---------------------------------------------------------------------------


def test_weights_pay_the_king_exactly_and_refuse_a_triple_that_differs_from_its_basis(signer):
    basis = signed_basis(signer)
    baseline = compute_final_weights(snapshot())
    proposed = proposed_snapshot(basis)
    result = compute_final_weights(proposed)
    assert abs(sum(result["weights"]) - 1.0) < 1e-12
    assert abs(weight_of(result, 3) - 0.25) < 1e-9 and weight_of(baseline, 3) == 0.0
    assert weight_of(result, 0) <= weight_of(baseline, 0) + 1e-9  # the Arena amount is never burned
    # A triple that is not what the basis implies is refused on every side.
    for mutate in (
        lambda s: s.update(champion_share=0.2, effective_champion_share=0.2),
        lambda s: s.update(champion_uid=1),
        lambda s: s.update(effective_champion_share=0.0),
    ):
        broken = dict(proposed)
        mutate(broken)
        broken["config_hash"] = weight_config_hash(broken)
        with pytest.raises(WeightComputationError, match="lab_arena_reward_basis"):
            compute_final_weights(broken)
    # A basis that does not hash is refused too.
    corrupt = dict(proposed, lab_arena_reward_basis=dict(basis, king_start_epoch=24800))
    with pytest.raises(WeightComputationError, match="lab_arena_reward_basis"):
        compute_final_weights(corrupt)
    # An ineligible epoch names the basis and a zero triple; that is consistent.
    stale = signed_basis(signer, effective=24700, start=24700)
    zero = snapshot(lab_arena_reward_basis=stale)
    assert kernel.champion_values(stale, EPOCH, HOTKEYS)["eligible"] is False
    assert compute_final_weights(zero)["weights"] == baseline["weights"]


def test_rewards_off_leaves_every_document_byte_identical(signer):
    plain = snapshot()
    documents = gateway_weight_input_value_documents_v2(calculation_snapshot=plain, gateway_authority_event_hash=_sha("event"))
    assert "lab_arena_reward_basis" not in documents["champions"]["value"]
    again = gateway_weight_input_value_documents_v2(calculation_snapshot=snapshot(), gateway_authority_event_hash=_sha("event"))
    assert canonical_json(documents) == canonical_json(again)
    with_basis = gateway_weight_input_value_documents_v2(calculation_snapshot=proposed_snapshot(signed_basis(signer)), gateway_authority_event_hash=_sha("event"))
    assert with_basis["champions"]["value"]["lab_arena_reward_basis"]["reward_basis_hash"] == signed_basis(signer)["reward_basis_hash"]
    assert with_basis["champions"]["value"]["champion_share"] == 0.25 and with_basis["champions"]["value"]["champion_uid"] == 3


# ---------------------------------------------------------------------------
# The coordinator: re-derive from the measured row and the pinned key
# ---------------------------------------------------------------------------


class FakeReader:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def read(self, *, policy_id, parameters, **_kwargs):
        self.calls.append((policy_id, dict(parameters)))
        return [dict(row) for row in self.rows.get(policy_id, [])]


def _coordinator_case(signer, monkeypatch, basis, proposed_basis=None, row_overrides=None, pinned=None):
    from gateway.tee.coordinator_weight_source_v2 import CoordinatorWeightSourceV2

    monkeypatch.setenv(kernel.SIGNING_KEY_HASH_ENV, pinned or signer.public_key_hash)
    row = {"round_id": basis["round_id"], "effective_reward_epoch": basis["effective_reward_epoch"], "reward_basis_hash": basis["reward_basis_hash"], "reward_basis_doc": basis, "signing_key_doc": signing_key_document(signer.public_key_der)}
    row.update(row_overrides or {})
    calculation = proposed_snapshot(proposed_basis or basis)
    documents = gateway_weight_input_value_documents_v2(calculation_snapshot=calculation, gateway_authority_event_hash=_sha("event"))
    context = SimpleNamespace(job_id="weight-input:test", purpose="research_lab.champion_input.v2", epoch_id=EPOCH, parent_receipt_hashes=(), record_transport=lambda _x: None, record_artifact=lambda _x: None)
    reader = FakeReader({"lab_arena_reward_basis": [row]})
    return CoordinatorWeightSourceV2(reader), documents["champions"], calculation, context, reader


def test_coordinator_reconstructs_the_triple_from_the_measured_row(signer, monkeypatch):
    from gateway.tee.coordinator_weight_source_v2 import CoordinatorWeightSourceV2Error

    basis = signed_basis(signer)
    source, proposed, calculation, context, reader = _coordinator_case(signer, monkeypatch, basis)
    reconstructed = source._champion_document(proposed, calculation, context)
    assert canonical_json(reconstructed) == canonical_json(proposed)
    assert reader.calls == [("lab_arena_reward_basis", {"epoch_id": EPOCH})]
    # A proposal that names a different (older) basis than the governing row is reported as newer.
    older = signed_basis(signer, effective=24790, start=24790)
    source, proposed, calculation, context, _ = _coordinator_case(signer, monkeypatch, basis, proposed_basis=older)
    with pytest.raises(CoordinatorWeightSourceV2Error, match="newer than the calculation snapshot"):
        source._champion_document(proposed, calculation, context)
    # A row signed by another key, or a key hash pinned differently, fails closed.
    other = LocalSigner.generate()
    source, proposed, calculation, context, _ = _coordinator_case(signer, monkeypatch, basis, row_overrides={"signing_key_doc": signing_key_document(other.public_key_der)})
    with pytest.raises(CoordinatorWeightSourceV2Error, match="reward basis is invalid"):
        source._champion_document(proposed, calculation, context)
    source, proposed, calculation, context, _ = _coordinator_case(signer, monkeypatch, basis, pinned=other.public_key_hash)
    with pytest.raises(CoordinatorWeightSourceV2Error, match="reward basis is invalid"):
        source._champion_document(proposed, calculation, context)
    # No pinned key at all refuses, never pays.
    source, proposed, calculation, context, _ = _coordinator_case(signer, monkeypatch, basis)
    monkeypatch.delenv(kernel.SIGNING_KEY_HASH_ENV)
    with pytest.raises(CoordinatorWeightSourceV2Error, match="reward basis is invalid"):
        source._champion_document(proposed, calculation, context)
    # No governing row for the epoch is an unavailable source, not an empty king.
    source, proposed, calculation, context, _ = _coordinator_case(signer, monkeypatch, basis)
    source._reader.rows = {}
    with pytest.raises(CoordinatorWeightSourceV2Error, match="unavailable"):
        source._champion_document(proposed, calculation, context)


def test_coordinator_requires_an_empty_legacy_slot_without_a_basis(monkeypatch, signer):
    from gateway.tee.coordinator_weight_source_v2 import CoordinatorWeightSourceV2, CoordinatorWeightSourceV2Error

    monkeypatch.delenv(kernel.REWARDS_ENABLED_ENV, raising=False)
    reader = FakeReader({})
    source = CoordinatorWeightSourceV2(reader)
    calculation = snapshot()
    documents = gateway_weight_input_value_documents_v2(calculation_snapshot=calculation, gateway_authority_event_hash=_sha("event"))
    context = SimpleNamespace(job_id="weight-input:test", purpose="research_lab.champion_input.v2", epoch_id=EPOCH, parent_receipt_hashes=(), record_transport=lambda _x: None, record_artifact=lambda _x: None)
    assert source._champion_document(documents["champions"], calculation, context) == documents["champions"]
    assert reader.calls == []

    # Once Arena rewards are active, the coordinator reads the database even
    # when the validator omits a basis. No row is a valid empty Arena; an
    # existing governing row makes the omitted snapshot stale and unsafe.
    monkeypatch.setenv(kernel.REWARDS_ENABLED_ENV, "1")
    active_reader = FakeReader({})
    active_source = CoordinatorWeightSourceV2(active_reader)
    assert active_source._champion_document(documents["champions"], calculation, context) == documents["champions"]
    assert active_reader.calls == [("lab_arena_reward_basis", {"epoch_id": EPOCH})]

    basis = signed_basis(signer)
    authoritative_reader = FakeReader({"lab_arena_reward_basis": [{"reward_basis_doc": basis}]})
    with pytest.raises(CoordinatorWeightSourceV2Error, match="omits the governing reward basis"):
        CoordinatorWeightSourceV2(authoritative_reader)._champion_document(
            documents["champions"], calculation, context
        )

    nonzero = copy.deepcopy(documents["champions"])
    nonzero["value"]["champion_share"] = 0.25
    nonzero["value"]["effective_champion_share"] = 0.25
    nonzero["value"]["champion_uid"] = 3
    with pytest.raises(CoordinatorWeightSourceV2Error, match="names no reward basis"):
        source._champion_document(nonzero, calculation, context)


# ---------------------------------------------------------------------------
# The gateway route and the validator's read of it
# ---------------------------------------------------------------------------


class _View:
    def __init__(self, rows):
        self.rows = rows
        self.filters = []

    def select(self, columns):
        self.filters.append(("select", columns))
        return self

    def lte(self, column, value):
        self.filters.append(("lte", column, value))
        return self

    def order(self, column, desc=False):
        self.filters.append(("order", column, desc))
        return self

    def limit(self, count):
        self.filters.append(("limit", count))
        return self

    def execute(self):
        rows = [row for row in self.rows if row["effective_reward_epoch"] <= [f for f in self.filters if f[0] == "lte"][0][2]]
        rows.sort(key=lambda row: -row["effective_reward_epoch"])
        return SimpleNamespace(data=rows[:1])


class _Supabase:
    def __init__(self, rows):
        self.view = _View(rows)

    def table(self, name):
        assert name == "lab_arena_reward_basis_v1"
        return self.view


def test_gateway_route_serves_the_governing_row_or_none(signer, monkeypatch):
    from gateway.fulfillment import api as fulfillment_api

    basis = signed_basis(signer)
    key_document = signing_key_document(signer.public_key_der)
    rows = [{"round_id": basis["round_id"], "effective_reward_epoch": basis["effective_reward_epoch"], "reward_basis_hash": basis["reward_basis_hash"], "reward_basis_doc": basis, "signing_key_doc": key_document}]
    monkeypatch.setattr(fulfillment_api, "_get_supabase", lambda: _Supabase(rows))
    served = fulfillment_api._collect_lab_arena_reward_basis_sync(EPOCH)
    assert served == {"epoch": EPOCH, "round_id": basis["round_id"], "reward_basis_hash": basis["reward_basis_hash"], "reward_basis": basis, "signing_key": key_document, "lookup_ok": True}
    assert fulfillment_api._collect_lab_arena_reward_basis_sync(24800)["reward_basis"] is None  # not yet effective
    incoherent = [dict(rows[0], reward_basis_hash=_sha("other"))]
    monkeypatch.setattr(fulfillment_api, "_get_supabase", lambda: _Supabase(incoherent))
    with pytest.raises(RuntimeError, match="incoherent"):
        fulfillment_api._collect_lab_arena_reward_basis_sync(EPOCH)


def test_validator_read_requires_a_complete_response(signer, monkeypatch):
    from Leadpoet.utils import cloud_db

    basis = signed_basis(signer)
    key_document = signing_key_document(signer.public_key_der)

    def fake_get(url, params=None, timeout=None):
        assert url.endswith("/fulfillment/lab-arena-reward-basis") and params == {"epoch": EPOCH}
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: fake_get.payload)

    fake_get.payload = {"epoch": EPOCH, "round_id": basis["round_id"], "reward_basis_hash": basis["reward_basis_hash"], "reward_basis": basis, "signing_key": key_document, "lookup_ok": True}
    monkeypatch.setattr(cloud_db.requests, "get", fake_get)
    monkeypatch.setattr(cloud_db.time, "sleep", lambda _s: None)
    monkeypatch.setattr(cloud_db, "bt", SimpleNamespace(logging=SimpleNamespace(warning=lambda *_a, **_k: None), Wallet=object))
    assert cloud_db.gateway_get_lab_arena_reward_basis(None, EPOCH)["reward_basis"] == basis
    fake_get.payload = {"epoch": EPOCH, "reward_basis": None, "signing_key": None, "lookup_ok": True, "round_id": None, "reward_basis_hash": None}
    assert cloud_db.gateway_get_lab_arena_reward_basis(None, EPOCH)["reward_basis"] is None
    for broken in (
        {"epoch": EPOCH, "reward_basis": basis, "signing_key": None, "lookup_ok": True},  # a basis without its key
        {"epoch": EPOCH + 1, "reward_basis": basis, "signing_key": key_document, "lookup_ok": True},  # another epoch
        {"epoch": EPOCH, "reward_basis": basis, "signing_key": key_document},  # no lookup flag
    ):
        fake_get.payload = broken
        with pytest.raises(RuntimeError, match="failed after 3 attempts"):
            cloud_db.gateway_get_lab_arena_reward_basis(None, EPOCH)
