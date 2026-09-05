"""Pure reward-kernel tests (labarena.md sections 13 and 18.8)."""

from __future__ import annotations

import ast
import os

import pytest

from lab_arena import contracts, rewards
from lab_arena.contracts import ArenaContractError
from leadpoet_canonical.weight_computation import _doc_percent_share

ALICE = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
BOB = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
LAB_SHARE = 0.30
LEADERBOARD_SHARE = 0.095
EXACT_WEEKLY_SHARES = (0.25, 0.2, 0.15, 0.1, 0.05)  # 25% of total emissions, decaying by week
METAGRAPH = ["5C" + "1" * 46, ALICE, BOB]


def _basis(
    *,
    outcome: str,
    finalized_epoch: int,
    king_hotkey: str = ALICE,
    previous_start: int | None = None,
    round_id: str = "arena-2026-09-02",
) -> dict:
    return rewards.reward_basis_document(
        round_id=round_id,
        published_at="2026-09-02T10:00:00Z",
        finalized_epoch=finalized_epoch,
        king_outcome=outcome,
        king_hotkey="" if outcome == "no_king" else king_hotkey,
        previous_king_start_epoch=previous_start,
    )


def _values(basis: dict, epoch_id: int, hotkeys=METAGRAPH) -> dict:
    return rewards.champion_values(basis, epoch_id, hotkeys)


# ---------------------------------------------------------------------------
# Constants and decay
# ---------------------------------------------------------------------------


def test_public_constants_match_section_1():
    assert contracts.LAB_ARENA_POOL_PERCENT == 25
    assert contracts.LAB_ARENA_POOL_BASIS == "total_emissions" and rewards.reward_constants_document()["pool_basis"] == "total_emissions"
    assert contracts.KING_POOL_SHARE_PERCENT_BY_WEEK == (100, 80, 60, 40, 20)
    assert contracts.EPOCHS_PER_REWARD_WEEK == 140
    assert contracts.ELIGIBILITY_MAX_EPOCHS == 45
    assert rewards.MAX_REWARD_WEEK_INDEX == 4


def test_reward_week_index_floors_and_clamps():
    assert rewards.reward_week_index(1000, 1000) == 0
    assert rewards.reward_week_index(1139, 1000) == 0
    assert rewards.reward_week_index(1140, 1000) == 1
    assert rewards.reward_week_index(1000 + 140 * 4, 1000) == 4
    assert rewards.reward_week_index(1000 + 140 * 40, 1000) == 4
    with pytest.raises(ValueError):
        rewards.reward_week_index(999, 1000)
    with pytest.raises(ValueError):
        rewards.reward_week_index(True, 0)


def test_king_pool_share_percent_is_the_closed_schedule():
    assert [rewards.king_pool_share_percent(i) for i in range(5)] == [100, 80, 60, 40, 20]
    for bad in (-1, 5, "0", 1.0):
        with pytest.raises(ValueError):
            rewards.king_pool_share_percent(bad)


# ---------------------------------------------------------------------------
# Shares
# ---------------------------------------------------------------------------


def test_five_exact_weekly_shares():
    for week, expected in enumerate(EXACT_WEEKLY_SHARES):
        assert rewards.champion_share_for_week(week) == expected


def test_champion_values_yield_exact_shares_across_the_schedule():
    basis = _basis(outcome="crowned", finalized_epoch=999)
    start = basis["king_start_epoch"]
    assert start == 1000
    for week, expected in enumerate(EXACT_WEEKLY_SHARES):
        # A defended row keeps the row fresh while the decay clock advances.
        epoch = start + 140 * week
        governing = _basis(outcome="defended", finalized_epoch=epoch - 1, previous_start=start) if week else basis
        values = _values(governing, epoch)
        assert values == {
            "champion_share": expected,
            "effective_champion_share": expected,
            "champion_uid": 1,
            "reward_week_index": week,
            "eligible": True,
        }


def test_fulfillment_residual_and_lab_share_derivation_match_canonical_kernel():
    assert rewards.fulfillment_residual(LAB_SHARE, LEADERBOARD_SHARE) == 0.605
    assert rewards.fulfillment_residual(0.9, 0.2) == 0.0
    cases = [
        ({"lab_cap_percent": 30}, 0.25),
        ({"lab_cap_percent": "30"}, 0.25),
        ({"lab_cap_percent": 30.5}, 0.25),
        ({"lab_cap_percent": 250}, 0.25),
        ({"lab_cap_percent": -5}, 0.25),
        ({"lab_cap_percent": ""}, 0.25),
        ({"lab_cap_percent": None}, 0.25),
        ({"lab_cap_percent": "thirty"}, 0.25),
        ({}, 0.25),
        (None, 0.25),
        ([], 0.25),
        ("30", 0.25),
    ]
    for doc, fallback in cases:
        assert rewards.derive_research_lab_share(doc, fallback) == _doc_percent_share(doc, "lab_cap_percent", fallback)
    with pytest.raises(ValueError):
        rewards.derive_research_lab_share({}, -0.1)


def test_share_is_a_fraction_of_total_emissions_independent_of_the_other_allocations():
    """The pool basis is total emissions: no allocation document changes the king's share."""

    for week in range(5):
        share = rewards.champion_share_for_week(week)
        assert share == contracts.LAB_ARENA_POOL_PERCENT / 100 * contracts.KING_POOL_SHARE_PERCENT_BY_WEEK[week] / 100
        assert 0.0 < share <= contracts.LAB_ARENA_POOL_PERCENT / 100
    # The residual the adapter shrinks is still the canonical kernel's number, and the share fits inside it today.
    assert rewards.champion_share_for_week(0) <= rewards.fulfillment_residual(LAB_SHARE, LEADERBOARD_SHARE)
    # An ineligible epoch returns everything to fulfillment: zero champion share.
    basis = _basis(outcome="crowned", finalized_epoch=999)
    assert _values(basis, 1000 + 46)["champion_share"] == 0.0


# ---------------------------------------------------------------------------
# Eligibility
# ---------------------------------------------------------------------------


def test_eligibility_boundary_at_exactly_45_epochs():
    basis = _basis(outcome="crowned", finalized_epoch=999)
    assert basis["effective_reward_epoch"] == 1000
    assert rewards.epoch_eligible(basis, 1000) is True
    assert rewards.epoch_eligible(basis, 1045) is True
    assert rewards.epoch_eligible(basis, 1046) is False
    assert _values(basis, 1045)["champion_share"] == 0.25
    assert _values(basis, 1046) == {
        "champion_share": 0.0,
        "effective_champion_share": 0.0,
        "champion_uid": None,
        "reward_week_index": 0,
        "eligible": False,
    }
    with pytest.raises(ValueError):
        rewards.epoch_eligible(basis, 999)


def test_no_catch_up_after_an_ineligible_gap():
    crowned = _basis(outcome="crowned", finalized_epoch=999)
    # Two missed rounds: the row is 46 epochs old -> ineligible, pays zero.
    assert _values(crowned, 1046)["champion_share"] == 0.0
    # The next publication (defended) keeps the start epoch; the epoch that
    # becomes eligible again pays only its own week's share, never the gap.
    defended = _basis(outcome="defended", finalized_epoch=1059, previous_start=1000)
    rows = [crowned, defended]
    assert rewards.governing_reward_basis(rows, 1046) is not None
    assert rewards.governing_reward_basis(rows, 1046)["effective_reward_epoch"] == 1000
    assert rewards.governing_reward_basis(rows, 1060)["effective_reward_epoch"] == 1060
    assert _values(defended, 1060)["champion_share"] == 0.25
    # A later fresh defense at the week-2 boundary pays week 2 only; the
    # skipped epochs are never recovered.
    later = _basis(outcome="defended", finalized_epoch=1139, previous_start=1000)
    assert _values(later, 1140)["champion_share"] == 0.2
    assert _values(later, 1140)["reward_week_index"] == 1
    # The stale defended row itself is ineligible again 46 epochs later.
    assert _values(defended, 1106)["champion_share"] == 0.0


def test_new_king_resets_the_schedule_and_defense_keeps_it():
    first = _basis(outcome="crowned", finalized_epoch=999)
    assert first["king_start_epoch"] == 1000
    defended = _basis(outcome="defended", finalized_epoch=1019, previous_start=1000)
    assert defended["king_start_epoch"] == 1000
    assert defended["effective_reward_epoch"] == 1020
    # Week 3 for the incumbent defending again at that epoch...
    week3 = _basis(outcome="defended", finalized_epoch=1000 + 140 * 2 - 1, previous_start=1000)
    assert week3["king_start_epoch"] == 1000
    assert _values(week3, 1000 + 140 * 2)["champion_share"] == 0.15
    assert _values(week3, 1000 + 140 * 2)["reward_week_index"] == 2
    # ...but a newly crowned king at that same epoch restarts at week 1.
    crowned = _basis(outcome="crowned", finalized_epoch=1000 + 140 * 2 - 1, king_hotkey=BOB)
    assert crowned["king_start_epoch"] == 1000 + 140 * 2
    values = _values(crowned, 1000 + 140 * 2)
    assert values["champion_share"] == 0.25
    assert values["champion_uid"] == 2
    assert values["reward_week_index"] == 0
    with pytest.raises(ValueError, match="previous reward basis"):
        _basis(outcome="defended", finalized_epoch=1019)
    with pytest.raises(ValueError):
        _basis(outcome="defended", finalized_epoch=1019, previous_start=1020)


# ---------------------------------------------------------------------------
# Hotkey binding
# ---------------------------------------------------------------------------


def test_hotkey_binding_unregistered_king_pays_zero():
    basis = _basis(outcome="crowned", finalized_epoch=999)
    assert _values(basis, 1000, hotkeys=["5C" + "1" * 46, BOB]) == {
        "champion_share": 0.0,
        "effective_champion_share": 0.0,
        "champion_uid": None,
        "reward_week_index": 0,
        "eligible": True,
    }
    assert rewards.champion_uid_for_hotkey(METAGRAPH, ALICE) == 1
    assert rewards.champion_uid_for_hotkey(METAGRAPH, "5" + "z" * 47) is None
    with pytest.raises(ValueError):
        rewards.champion_uid_for_hotkey([ALICE, ALICE], ALICE)
    with pytest.raises(ValueError):
        rewards.champion_uid_for_hotkey("not-a-list", ALICE)


def test_hotkey_at_champion_uid_must_equal_the_king():
    assert rewards.champion_uid_matches(METAGRAPH, 1, ALICE) is True
    assert rewards.champion_uid_matches(METAGRAPH, 2, ALICE) is False
    assert rewards.champion_uid_matches(METAGRAPH, 7, ALICE) is False
    assert rewards.champion_uid_matches(METAGRAPH, -1, ALICE) is False
    assert rewards.champion_uid_matches(METAGRAPH, None, ALICE) is False
    assert rewards.champion_uid_matches(METAGRAPH, True, ALICE) is False
    assert rewards.champion_uid_matches(METAGRAPH, 0, "") is False
    # The kernel's own UID always satisfies the binding it will be checked against.
    values = _values(_basis(outcome="crowned", finalized_epoch=999), 1000)
    assert rewards.champion_uid_matches(METAGRAPH, values["champion_uid"], ALICE)


# ---------------------------------------------------------------------------
# Outcomes, governing rows, and fail-closed reads
# ---------------------------------------------------------------------------


def test_all_four_king_outcomes():
    crowned = _basis(outcome="crowned", finalized_epoch=999)
    defended = _basis(outcome="defended", finalized_epoch=1019, previous_start=1000)
    retained = _basis(outcome="retained_ineligible", finalized_epoch=1039, previous_start=1000)
    no_king = _basis(outcome="no_king", finalized_epoch=1059)
    for basis in (crowned, defended, retained, no_king):
        contracts.validate_reward_basis(basis)
    assert _values(crowned, 1000)["champion_share"] == 0.25
    assert _values(defended, 1020)["champion_share"] == 0.25
    assert _values(retained, 1040) == {
        "champion_share": 0.0,
        "effective_champion_share": 0.0,
        "champion_uid": None,
        "reward_week_index": 0,
        "eligible": False,
    }
    assert retained["king_start_epoch"] == 1000
    assert retained["king_hotkey"] == ALICE
    assert no_king["king_hotkey"] == ""
    assert no_king["king_start_epoch"] == 0
    assert _values(no_king, 1060) == {
        "champion_share": 0.0,
        "effective_champion_share": 0.0,
        "champion_uid": None,
        "reward_week_index": None,
        "eligible": False,
    }
    assert rewards.reward_basis_document(
        round_id="arena-2026-09-05",
        published_at="2026-09-05T10:00:00Z",
        finalized_epoch=1079,
        king_outcome="retained_ineligible",
        king_hotkey=ALICE,
    )["king_start_epoch"] == 0
    with pytest.raises(ValueError):
        rewards.reward_basis_document(
            round_id="arena-2026-09-05",
            published_at="2026-09-05T10:00:00Z",
            finalized_epoch=1,
            king_outcome="no_king",
            king_hotkey=ALICE,
        )
    with pytest.raises(ValueError):
        rewards.reward_basis_document(
            round_id="arena-2026-09-05",
            published_at="2026-09-05T10:00:00Z",
            finalized_epoch=1,
            king_outcome="crowned",
            king_hotkey="",
        )


def test_governing_row_selection_and_duplicate_effective_epoch_rejection():
    a = _basis(outcome="crowned", finalized_epoch=999)
    b = _basis(outcome="defended", finalized_epoch=1019, previous_start=1000)
    c = _basis(outcome="defended", finalized_epoch=1039, previous_start=1000, round_id="arena-2026-09-04")
    assert rewards.governing_reward_basis([], 5000) is None
    assert rewards.governing_reward_basis([c, a, b], 999) is None
    assert rewards.governing_reward_basis([c, a, b], 1000)["reward_basis_hash"] == a["reward_basis_hash"]
    assert rewards.governing_reward_basis([c, a, b], 1039)["reward_basis_hash"] == b["reward_basis_hash"]
    assert rewards.governing_reward_basis([c, a, b], 9999)["reward_basis_hash"] == c["reward_basis_hash"]
    duplicate = _basis(outcome="defended", finalized_epoch=1039, previous_start=1000, round_id="arena-2026-09-06")
    with pytest.raises(ValueError, match="duplicate effective_reward_epoch"):
        rewards.governing_reward_basis([a, c, duplicate], 1000)
    # Tampered rows fail closed through the contracts validator.
    tampered = dict(a)
    tampered["king_start_epoch"] = 5
    with pytest.raises(ArenaContractError):
        rewards.governing_reward_basis([tampered], 1000)


def test_invalid_outcome_fails_closed_everywhere():
    basis = dict(_basis(outcome="crowned", finalized_epoch=999))
    basis["king_outcome"] = "ascended"
    with pytest.raises(ValueError):
        rewards.epoch_eligible(basis, 1000)
    with pytest.raises(ValueError):
        rewards.champion_values(basis, 1000, METAGRAPH)
    with pytest.raises(ValueError):
        rewards.governing_reward_basis([basis], 1000)
    with pytest.raises(ValueError):
        rewards.king_start_epoch_for_outcome("ascended", 1000, None)
    with pytest.raises(ValueError):
        rewards.reward_basis_document(
            round_id="arena-2026-09-05",
            published_at="2026-09-05T10:00:00Z",
            finalized_epoch=1,
            king_outcome="ascended",
            king_hotkey=ALICE,
        )
    for broken in (
        {"king_outcome": "crowned", "effective_reward_epoch": 1000, "king_start_epoch": 1001, "king_hotkey": ALICE},
        {"king_outcome": "crowned", "effective_reward_epoch": 1000, "king_start_epoch": 1000, "king_hotkey": ""},
        {"king_outcome": "no_king", "effective_reward_epoch": 1000, "king_start_epoch": 0, "king_hotkey": ALICE},
        {"king_outcome": "crowned", "effective_reward_epoch": "1000", "king_start_epoch": 1000, "king_hotkey": ALICE},
        "not-a-basis",
    ):
        with pytest.raises(ValueError):
            rewards.champion_values(broken, 1000, METAGRAPH)
    for bad_hotkeys in ("not-a-list", None, [ALICE, 7]):
        with pytest.raises(ValueError):
            rewards.champion_values(_basis(outcome="crowned", finalized_epoch=999), 1000, bad_hotkeys)


def test_reward_basis_document_binds_constants_and_hashes():
    basis = _basis(outcome="crowned", finalized_epoch=999)
    assert basis["reward_constants"] == {
        "pool_percent": 25,
        "pool_basis": "total_emissions",
        "king_pool_share_percent_by_week": [100, 80, 60, 40, 20],
        "epochs_per_reward_week": 140,
        "eligibility_max_epochs": 45,
    }
    assert basis["schema_version"] == contracts.REWARD_BASIS_SCHEMA_VERSION
    contracts.verify_hashed_document(basis, "reward_basis_hash")
    assert "signature" not in basis
    for removed_field in ("configuration_hash", "commitment_hash", "result_bundle_hash"):
        with pytest.raises(ArenaContractError):
            contracts.validate_reward_basis(dict(basis, **{removed_field: "sha256:" + "0" * 64}))


# ---------------------------------------------------------------------------
# Python 3.7 syntax (the kernel relocates into leadpoet_canonical/)
# ---------------------------------------------------------------------------

_PY37_BUILTIN_GENERICS = {"list", "dict", "set", "frozenset", "tuple", "type"}
_POST_37_NAMES = {"removeprefix", "removesuffix", "cached_property", "Literal", "Final", "Protocol", "TypedDict", "prod"}
_ALLOWED_IMPORT_MODULES = {
    "__future__", "base64", "hashlib", "json", "math", "fractions", "typing",
    # Signature verification only, imported lazily inside the verify function.
    "cryptography.exceptions", "cryptography.hazmat.primitives", "cryptography.hazmat.primitives.asymmetric",
}


def _annotation_has_union(annotation: ast.AST) -> bool:
    return any(isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr) for node in ast.walk(annotation))


def test_reward_kernel_is_python_37_syntax_and_stdlib_only():
    """The kernel lives in leadpoet_canonical (the validator enclave image copies it) and must stay Python 3.7."""

    from leadpoet_canonical import lab_arena_rewards as canonical_rewards

    path = os.path.join(os.path.dirname(canonical_rewards.__file__), "lab_arena_rewards.py")
    with open(path, "r", encoding="utf-8") as handle:
        source = handle.read()
    assert ":=" not in source
    assert "\nmatch " not in source and "    match " not in source
    tree = ast.parse(source, filename=path, feature_version=(3, 7))
    match_node = getattr(ast, "Match", None)
    for node in ast.walk(tree):
        assert not isinstance(node, ast.NamedExpr), "walrus operator is not Python 3.7"
        if match_node is not None:
            assert not isinstance(node, match_node), "match statement is not Python 3.7"
        if isinstance(node, ast.arguments):
            assert not getattr(node, "posonlyargs", []), "positional-only parameters are not Python 3.7"
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            assert node.value.id not in _PY37_BUILTIN_GENERICS, "PEP 585 builtin generics are not Python 3.7"
        if isinstance(node, ast.Attribute):
            assert node.attr not in _POST_37_NAMES, "%s is not available in Python 3.7" % node.attr
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module]
            for module in modules:
                assert module in _ALLOWED_IMPORT_MODULES, "unexpected import %s" % module
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    assert alias.name not in _POST_37_NAMES
        annotations = []
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            annotations.extend(arg.annotation for arg in node.args.args + node.args.kwonlyargs if arg.annotation)
            if node.returns is not None:
                annotations.append(node.returns)
        if isinstance(node, ast.AnnAssign):
            annotations.append(node.annotation)
        for annotation in annotations:
            # Under ``from __future__ import annotations`` these are strings;
            # parse them and reject PEP 604 unions and builtin generics.
            if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
                annotation = ast.parse(annotation.value, mode="eval").body
            assert not _annotation_has_union(annotation), "PEP 604 unions are not Python 3.7"
            for inner in ast.walk(annotation):
                if isinstance(inner, ast.Subscript) and isinstance(inner.value, ast.Name):
                    assert inner.value.id not in _PY37_BUILTIN_GENERICS
    assert "from __future__ import annotations" in source
