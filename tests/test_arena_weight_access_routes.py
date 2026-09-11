"""Retired public reward delivery must not bypass signed validator requests."""

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_fulfillment_has_no_public_reward_authority_route_or_collector():
    source = (ROOT / "gateway/fulfillment/api.py").read_text()
    tree = ast.parse(source)
    function_names = {
        node.name for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "get_lab_arena_reward_basis" not in function_names
    assert "_collect_lab_arena_reward_basis_sync" not in function_names
    assert '"/lab-arena-reward-basis"' not in source


def test_retired_client_cannot_fall_back_to_unsigned_reward_delivery():
    source = (ROOT / "Leadpoet/utils/cloud_db.py").read_text()
    assert "gateway_get_lab_arena_reward_basis" not in source
    assert "/fulfillment/lab-arena-reward-basis" not in source
