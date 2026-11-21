"""The lead-order lane never changes Arena weights unless it pays in live mode."""

from __future__ import annotations

from fractions import Fraction

import pytest

from leadpoet_canonical import arena_weights
from lab_arena.lead_orders import money, weights
from tests.test_arena_weights import HOTKEYS, _signed_state

CAP = Fraction(3, 10)


@pytest.mark.parametrize("factor,winner", [(1_000_000, HOTKEYS[2]), (500_000, HOTKEYS[1]), (1_000_000, "5" + "D" * 47)])
def test_no_lane_returns_the_arena_vector_unchanged(factor, winner):
    state, _, _ = _signed_state(champion_reward_factor_ppm=factor, winner=winner)
    arena = arena_weights.derive_arena_weights(state, HOTKEYS)
    for lane in (None, {}, {HOTKEYS[1]: Fraction(0)}):
        assert weights.merge_lane_weights(state, HOTKEYS, lane, lane_cap=CAP) == arena
    # The restated allocation and quantization reproduce the canonical kernel.
    by_hotkey, champion, unused = weights._allocations(state, HOTKEYS)
    assert weights._sparse(state, HOTKEYS, by_hotkey, champion, unused) == arena
    assert weights.burn_residual(state, HOTKEYS) == unused


def test_lane_takes_only_from_the_burn_residual_and_respects_the_cap():
    state, _, _ = _signed_state()  # champion HOTKEYS[2] 5%, burn HOTKEYS[0] 95%
    arena = arena_weights.derive_arena_weights(state, HOTKEYS)
    merged = weights.merge_lane_weights(state, HOTKEYS, {HOTKEYS[1]: Fraction(1, 10)}, lane_cap=CAP)
    assert merged["champion_share_ppb"] == arena["champion_share_ppb"] == 50_000_000
    assert merged["burned_residual_ppb"] == 850_000_000 and merged["lead_order_lane_ppb"] == 100_000_000
    assert merged["sparse_uids"] == [0, 1, 2]
    u16 = dict(zip(merged["sparse_uids"], merged["sparse_weights_u16"]))
    assert u16[0] == 65535 and abs(u16[1] / u16[2] - 2.0) < 0.001  # champion share untouched
    with pytest.raises(arena_weights.ArenaWeightError, match="cap"):
        weights.merge_lane_weights(state, HOTKEYS, {HOTKEYS[1]: Fraction(31, 100)}, lane_cap=CAP)
    with pytest.raises(arena_weights.ArenaWeightError, match="unregistered"):
        weights.merge_lane_weights(state, HOTKEYS, {"5" + "Z" * 47: Fraction(1, 10)}, lane_cap=CAP)
    with pytest.raises(arena_weights.ArenaWeightError, match="burn residual"):
        weights.merge_lane_weights(state, HOTKEYS, {HOTKEYS[1]: Fraction(96, 100)}, lane_cap=Fraction(1))


def test_only_registered_miners_are_paid():
    entries = [money.OwedEntry("o1", HOTKEYS[1], 1, 1, 5), money.OwedEntry("o2", "5" + "Z" * 47, 1, 2, 5)]
    assert [e.order_id for e in weights.registered_entries(entries, HOTKEYS)] == ["o1"]
