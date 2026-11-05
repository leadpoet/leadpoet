"""The aggregate RPC and exporter select the same retained successful reply."""
from copy import deepcopy
import json
from pathlib import Path
from uuid import uuid4

import pytest

from lab_arena import contracts, training_export
from tests.lab_arena.capture_summary598_postgres_test import (
    _catalog, _seed_bodies, _security_catalog, _summary, database, migrated, upgraded,
)
from tests.lab_arena.capture_summary593_postgres_test import ROUND
from tests.lab_arena.retained_openrouter_success_test import retained_success

SQL = Path(__file__).resolve().parents[2] / "scripts/620-lab-arena-retained-success-capture.sql"


@pytest.fixture(scope="module")
def retained_upgrade(database, upgraded):
    pg, dsn = database
    control_submission, *_ = seed(database, settled=True)
    assert _summary(database, control_submission)["model_calls"] == 0
    with pg.connect(**dsn) as connection, connection.cursor() as cur:
        # Match the release principal: member of the function owner, with
        # schema CREATE itself, while the owner role has no schema CREATE.
        cur.execute("CREATE ROLE capture_migrator620 NOLOGIN INHERIT")
        cur.execute("GRANT lab_arena_owner TO capture_migrator620")
        cur.execute("GRANT USAGE,CREATE ON SCHEMA public TO capture_migrator620")
        cur.execute("SELECT has_schema_privilege('lab_arena_owner','public','CREATE')")
        assert cur.fetchone() == (False,)
        before, security = _catalog(cur), _security_catalog(cur)
        cur.execute("SET ROLE capture_migrator620")
        cur.execute(SQL.read_text())
        cur.execute("RESET ROLE")
        first = _catalog(cur), _security_catalog(cur)
        cur.execute("SET ROLE capture_migrator620")
        cur.execute(SQL.read_text())
        cur.execute("RESET ROLE")
        assert first == (_catalog(cur), _security_catalog(cur))
        assert first[1] == security
        functions, *other = first[0]
        original, *old_other = before
        assert other == old_other and set(functions) == set(original)
        changed = {name for name in functions if functions[name] != original[name]}
        assert changed == {f"lab_arena_capture_summary_v{v}(text,text)" for v in (1, 2)}
        for name in changed:
            assert {k: v for k, v in functions[name].items() if k != "prosrc"} == {
                k: v for k, v in original[name].items() if k != "prosrc"}
    assert _summary(database, control_submission)["model_calls"] == 1


def seed(database, *, settled=False, reason="missing_provider_cost", fault=None):
    pg, dsn = database
    values, body = retained_success(settled=settled, reason=reason)
    submission = _seed_bodies(database, terminals=[])
    identity = contracts.hash_bytes(uuid4().bytes)
    with pg.connect(**dsn) as connection, connection.cursor() as cur:
        cur.execute("SET session_replication_role=replica")
        cur.execute("SELECT run_id FROM public.lab_arena_runs WHERE round_id=%s AND submission_id=%s",
                    (ROUND, submission))
        run = cur.fetchone()[0]
        reservation_id = uncertainty_id = None
        for row in deepcopy(values["ledger_rows"]):
            document = row.get("entry_doc") or {}
            if row["entry_kind"] == "uncertain":
                saved = document["call"]
                terminal = saved["openrouter_terminal_response"]
                if fault == "fingerprint":
                    saved["credential_fingerprint"] = "private-secret"
                elif fault == "conflicting_fingerprint":
                    saved["credential_fingerprint"] = "sha256:" + "d" * 64
                elif fault == "status":
                    terminal["status"] = "200"
                elif fault == "body":
                    terminal["body_b64"] = "@@invalid@@"
                elif fault == "reason":
                    saved["reason"] = "transport_failure"
                elif fault == "private_only":
                    saved["upstream_response_receipt"] = saved.pop("openrouter_terminal_response")
                elif fault == "membership":
                    row["round_id"] = "wrong-round"
                elif fault in {"miner_hotkey", "stage", "funding_source"}:
                    row[fault] = {"miner_hotkey": "wrong-owner", "stage": 2,
                                  "funding_source": "miner_key"}[fault]
            if row["entry_kind"] == "reservation":
                document["credential_fingerprint"] = "sha256:" + "c" * 64
            if row["entry_kind"] == "settlement":
                document["reconciled_uncertainty_entry_id"] = (
                    reservation_id if fault == "stale" else uncertainty_id)
            cur.execute("INSERT INTO public.lab_arena_ledger (entry_kind,miner_hotkey,round_id,submission_id,"
                "run_id,stage,call_identity,provider,operation_id,funding_source,amount_microusd,"
                "entry_doc,terminal_response) VALUES (%s,%s,%s,%s,%s,%s,%s,'openrouter',"
                "'openrouter.chat',%s,%s,%s,%s) RETURNING entry_id",
                (row["entry_kind"], row.get("miner_hotkey", "hk620"),
                 row["round_id"] if fault == "membership" and row["entry_kind"] == "uncertain"
                 else ROUND, submission, run, row.get("stage", 1), identity, row.get("funding_source", "host"),
                 row["amount_microusd"], json.dumps(document),
                 json.dumps(row.get("terminal_response"))))
            entry_id = cur.fetchone()[0]
            if row["entry_kind"] == "reservation":
                reservation_id = entry_id
            elif row["entry_kind"] == "uncertain":
                uncertainty_id = entry_id
        cur.execute("SELECT to_jsonb(l) FROM public.lab_arena_ledger l WHERE call_identity=%s ORDER BY entry_id", (identity,))
        rows = [row[0] for row in cur.fetchall()]
        cur.execute("SET session_replication_role=origin")
    return submission, run, rows, values["objects"], body, uncertainty_id


@pytest.mark.parametrize("settled", [False, True])
@pytest.mark.parametrize("reason", ["missing_provider_cost", "settle_failure"])
def test_export_capture_and_accounting_parity(database, retained_upgrade, settled, reason):
    submission, run, rows, objects, body, selected = seed(database, settled=settled, reason=reason)
    summary = _summary(database, submission)
    assert summary["model_calls"] == summary["usage_calls"] == summary["token_calls"] == 1
    assert [summary[k] for k in ("input_tokens", "output_tokens", "reasoning_tokens")] == [7, 11, 3]
    assert summary["models"] == [{"model": "openai/gpt-test", "calls": 1,
        "input_tokens": 7, "output_tokens": 11, "reasoning_tokens": 3}]
    calls, _ = training_export._calls(rows, {run}, objects, {run: 0})
    call = calls[0]
    assert call["response"] == json.loads(body) and training_export.response_entry_id(call) == selected
    assert call["actual_microusd"] == (19 if settled else None)
    assert ("provider_cost_unresolved" in call["evidence_gaps"]) is not settled
    assert "retained model answer" not in json.dumps(summary)
    # Existing capture timestamps describe the accounting lifecycle, including
    # delayed settlement. They do not assert upstream transport duration.
    assert summary["last_call_at"] == rows[-1]["created_at"]


@pytest.mark.parametrize("fault", ["fingerprint", "conflicting_fingerprint", "status", "body", "reason",
                                   "membership", "stale", "private_only", "miner_hotkey", "stage", "funding_source"])
def test_capture_rejects_unbound_or_malformed_retained_success(database, retained_upgrade, fault):
    submission, run, rows, objects, _, _ = seed(database, settled=True, fault=fault)
    assert _summary(database, submission)["model_calls"] == 0
    if fault != "private_only":
        with pytest.raises(training_export.ExportError):
            training_export._calls(rows, {run}, objects, {run: 0})
    else:
        calls, _ = training_export._calls(rows, {run}, objects, {run: 0})
        assert "response" in calls[0] and calls[0]["response_status"] == 502
