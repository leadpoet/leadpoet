from __future__ import annotations

import ast
from pathlib import Path
import re


ROOT = Path(__file__).parents[1]
SQL_TEMPLATE = ROOT / "scripts/284-arena-2026-09-17-baseline-recovery.sql.template"
OPERATOR_TEMPLATE = ROOT / "scripts/arena_sep17_baseline_recovery284.py.template"

SQL_PLACEHOLDERS = {
    "__FORWARD_SCHEDULE_JSON__",
    "__OLD_SCHEDULE_JSON__",
    "__PRIOR283_AUDIT_HASH__",
    "__PRIOR283_AUTHORITY_HASH__",
    "__RECOVERY_SOURCE_COMMIT__",
    "__RECOVERY_SOURCE_SHA256__",
    "__RECOVERY_SOURCE_SIZE_BYTES__",
    "__TERMINAL_BASELINE_LEDGER_COUNT__",
    "__TERMINAL_BASELINE_LEDGER_HASH__",
    "__TERMINAL_BASELINE_LEDGER_MAX_ENTRY_ID__",
    "__TERMINAL_BASELINE_RUNS_HASH__",
    "__TERMINAL_BASELINE_RUN_COUNT__",
    "__TERMINAL_BASELINE_SETTLED_MICROUSD__",
    "__TERMINAL_BASELINE_SUBMISSION_HASH__",
    "__TERMINAL_BASELINE_UNCERTAIN_MICROUSD__",
    "__TERMINAL_CANCEL_REASON__",
    "__TERMINAL_EXECUTE_COST_JSON__",
    "__TERMINAL_NONBASELINE_LEDGER_COUNT__",
    "__TERMINAL_NONBASELINE_LEDGER_HASH__",
    "__TERMINAL_NONBASELINE_LEDGER_MAX_ENTRY_ID__",
    "__TERMINAL_NONBASELINE_SUBMISSIONS_HASH__",
    "__TERMINAL_NONBASELINE_SUBMISSION_COUNT__",
    "__TERMINAL_ROUND_HASH__",
    "__TERMINAL_SCORE_COST_JSON__",
    "__TERMINAL_STAGE_GENERATION__",
    "__TERMINAL_STATUS_GENERATION__",
    "__TERMINAL_SUBMISSIONS_HASH__",
    "__TERMINAL_SUBMISSION_COUNT__",
}
OPERATOR_PLACEHOLDERS = {
    "__RECOVERY284_SOURCE_COMMIT__",
    "__RECOVERY284_SOURCE_SHA256__",
    "__RECOVERY284_SOURCE_SIZE_BYTES__",
}


def placeholders(value: str) -> set[str]:
    return set(re.findall(r"__[A-Z0-9_]+__", value))


def test_unsealed_templates_have_exact_scope_and_preservation_guards():
    sql = SQL_TEMPLATE.read_text()
    operator = OPERATOR_TEMPLATE.read_text()
    assert placeholders(sql) == SQL_PLACEHOLDERS
    assert placeholders(operator) == OPERATOR_PLACEHOLDERS
    assert "lab_arena_sep17_recovery283_archive_valid_v1() IS TRUE" in sql
    assert "lab_arena_sep17_recovery283_nonbaseline_ledger_valid_v1() IS TRUE" in sql
    assert "arena-2026-09-17-rerun283archive" in sql
    assert "baseline-2026-09-17-rerun283archive" in sql
    assert "authorized_sep17_recovery283_baseline_archive" in sql
    assert "assignment_id LIKE '%:rerun283'" in sql
    assert "assignment_id LIKE '%:rerun284'" in sql
    assert "(v_new_configuration - 'schedule') IS DISTINCT FROM" in sql
    assert "(v_round.configuration_doc - 'schedule')" in sql
    assert "WHERE round_id = v_round_id AND submission_id = v_baseline_id" in sql
    assert "WHERE round_id = v_round_id AND submission_id <> v_baseline_id" in sql
    assert "FOR v_position IN 0 .. 19 LOOP" in sql
    assert "WHERE round_id = v_round_id AND submission_id = v_baseline_id) <> 0" in sql
    assert "recovery284.tar.gz" in operator
    assert "recovery283.tar.gz" in operator
    assert "terminal_source_size_bytes = 581173" in sql
    assert "7df255f5db5267b13a65c9c3476cb47fd84837e00ce503bd1f588585ee8a37ae" in sql
    assert "e40484afde214a417128d1eadacaff716755784f" in sql
    assert "580050" not in sql
    assert "cfa4a9bf8093dd230e8643bec5508b675e6f4b0ded75ac3d0f4c2a23616b6e41" not in sql
    assert "5a89cee202416419ade064e85ff40499f51d34b0" not in sql
    assert '"execute_namespace": "rerun284"' in operator
    assert '"benchmark_deadline": "2026-09-17T14:00:00Z"' in operator
    assert "Recovery284 reads recovery283 as immutable history" in operator


def test_operator_template_renders_to_one_exact_source_identity():
    rendered = OPERATOR_TEMPLATE.read_text()
    values = {
        "__RECOVERY284_SOURCE_COMMIT__": "a" * 40,
        "__RECOVERY284_SOURCE_SHA256__": "b" * 64,
        "__RECOVERY284_SOURCE_SIZE_BYTES__": "581234",
    }
    for token, value in values.items():
        assert rendered.count(token) == 1
        rendered = rendered.replace(token, value)
    assert not placeholders(rendered)
    compile(rendered, "<sealed-recovery284-operator>", "exec")
    tree = ast.parse(rendered)
    assignments = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id not in {
            "SOURCE_COMMIT", "SOURCE_SHA256", "SOURCE_SIZE"
        }:
            continue
        if target.id == "SOURCE_SIZE":
            assert isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
            assert node.value.func.id == "int" and len(node.value.args) == 1
            assignments[target.id] = int(ast.literal_eval(node.value.args[0]))
        else:
            assignments[target.id] = ast.literal_eval(node.value)
    assert assignments == {
        "SOURCE_COMMIT": "a" * 40,
        "SOURCE_SHA256": "b" * 64,
        "SOURCE_SIZE": 581234,
    }


def test_recovery284_sealed_files_use_exact_authorities():
    sealed_sql = (
        ROOT / "scripts/284-arena-2026-09-17-baseline-recovery.sql"
    ).read_text()
    sealed_operator = (
        ROOT / "scripts/arena_sep17_baseline_recovery284.py"
    ).read_text()
    assert not placeholders(sealed_sql)
    assert not placeholders(sealed_operator)
    assert "91e8b95637c4cbc919f3150d6643985624dbb34a" in sealed_operator
    assert "b5ca95d7c9c25650c5ecb04b05bde859cffa3fe9c6c7a6602e6a006d5e3dcd99" in sealed_operator
    assert 'SOURCE_SIZE = int("582191")' in sealed_operator
    assert "91e8b95637c4cbc919f3150d6643985624dbb34a" in sealed_sql
    assert "b5ca95d7c9c25650c5ecb04b05bde859cffa3fe9c6c7a6602e6a006d5e3dcd99" in sealed_sql
