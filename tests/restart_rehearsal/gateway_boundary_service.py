#!/usr/bin/env python3.11
"""Strict local PostgREST equivalent for the exact gateway launcher replay."""

from __future__ import annotations

import argparse
import ast
import csv
from datetime import date
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import threading
import time
from typing import Any
from urllib.parse import parse_qsl, urlparse

try:
    from fixture_contract import (
        load_rehearsal_current_settlement_epoch_id,
    )
except ModuleNotFoundError as exc:
    if exc.name != "fixture_contract":
        raise
    from tests.restart_rehearsal.fixture_contract import (
        load_rehearsal_current_settlement_epoch_id,
    )


RUNTIME_TABLES = frozenset(
    {
        "epoch_audit_logs",
        "leads_private",
        "merkle_checkpoints",
        "published_weight_bundles",
        "qualification_baselines",
        "qualification_private_icp_sets",
        "research_lab_champion_reward_current",
        "research_lab_candidate_evaluation_current",
        "research_lab_candidate_promotion_events",
        "research_lab_gateway_control_current",
        "research_lab_public_benchmark_report_current",
        "research_lab_source_add_reward_current",
        "research_lab_stateful_subnet_epoch_cutover_state_v1",
        "research_lab_stateful_subnet_epoch_cutovers_v1",
        "transparency_log",
        "validation_evidence_private",
    }
)
RUNTIME_RPCS = frozenset(
    {
        "research_lab_source_add_claim_work",
        "research_lab_stateful_subnet_epoch_cutover_public_state_v1",
    }
)
TABLE_RE = re.compile(r"^[a-z][a-z0-9_]{0,127}$")
JSON_FILTER_RE = re.compile(
    r"^(?P<column>[a-z][a-z0-9_]{0,127})"
    r"(?P<path>(?:(?:->>|->)[A-Za-z_][A-Za-z0-9_]{0,127})+)$"
)
JSON_FILTER_TOKEN_RE = re.compile(
    r"(?P<operator>->>|->)(?P<key>[A-Za-z_][A-Za-z0-9_]{0,127})"
)
HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
DAY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
SENSITIVE_DOCUMENT_RE = re.compile(
    r"(sk-or-|sb_secret|service_role|openrouter_api_key|raw_secret|"
    r"authorization|proxy-authorization|://[^/]+:[^/@]+@)",
    re.IGNORECASE,
)
CONTROL_QUERY_FIELDS = frozenset(
    {"columns", "limit", "offset", "on_conflict", "order", "select"}
)


def _filter_scalar(raw: str, existing: Any) -> Any:
    if existing is None:
        if raw.lower() == "null":
            return None
        return raw
    if isinstance(existing, bool):
        if raw.lower() not in {"true", "false"}:
            raise ValueError("PostgREST boolean filter is invalid")
        return raw.lower() == "true"
    if isinstance(existing, int) and not isinstance(existing, bool):
        return int(raw)
    if isinstance(existing, float):
        return float(raw)
    if isinstance(existing, (dict, list)):
        value = json.loads(raw)
        if not isinstance(value, type(existing)):
            raise ValueError("PostgREST JSON filter type differs")
        return value
    return raw


def _in_values(raw: str, existing: Any) -> list[Any]:
    if not raw.startswith("(") or not raw.endswith(")"):
        raise ValueError("PostgREST in filter is invalid")
    reader = csv.reader(
        io.StringIO(raw[1:-1]),
        skipinitialspace=False,
        strict=True,
    )
    values = next(reader)
    if next(reader, None) is not None:
        raise ValueError("PostgREST in filter has multiple records")
    return [_filter_scalar(value, existing) for value in values]


def _json_filter_parts(
    reference: str,
) -> tuple[str, list[tuple[str, str]]]:
    match = JSON_FILTER_RE.fullmatch(reference)
    if match is None:
        raise ValueError("PostgREST filter column is invalid")
    tokens = [
        (token.group("operator"), token.group("key"))
        for token in JSON_FILTER_TOKEN_RE.finditer(match.group("path"))
    ]
    if (
        not tokens
        or any(operator == "->>" for operator, _key in tokens[:-1])
    ):
        raise ValueError("PostgREST filter column is invalid")
    return match.group("column"), tokens


def _filter_value(row: dict[str, Any], reference: str) -> Any:
    if TABLE_RE.fullmatch(reference):
        return row.get(reference)
    column, tokens = _json_filter_parts(reference)
    value = row.get(column)
    for operator, key in tokens:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
        if operator == "->>" and value is not None:
            if not isinstance(value, str):
                return json.dumps(
                    value,
                    sort_keys=True,
                    separators=(",", ":"),
                )
    return value


def _filter_root(reference: str) -> str:
    if TABLE_RE.fullmatch(reference):
        return reference
    column, _tokens = _json_filter_parts(reference)
    return column


def _matches_filter(row: dict[str, Any], column: str, expression: str) -> bool:
    _filter_root(column)
    if "." not in expression:
        raise ValueError("PostgREST filter operator is missing")
    negate = expression.startswith("not.")
    if negate:
        expression = expression[4:]
    operator, raw = expression.split(".", 1)
    existing = _filter_value(row, column)
    if operator == "is":
        expected = {"null": None, "true": True, "false": False}.get(raw.lower())
        if raw.lower() not in {"null", "true", "false"}:
            raise ValueError("PostgREST is filter is invalid")
        matched = (
            existing is expected
            if expected is None
            else existing == expected
        )
    elif operator == "in":
        matched = existing in _in_values(raw, existing)
    else:
        expected = _filter_scalar(raw, existing)
        if operator == "eq":
            matched = existing == expected
        elif operator == "neq":
            matched = existing != expected
        elif operator == "lt":
            matched = existing is not None and existing < expected
        elif operator == "lte":
            matched = existing is not None and existing <= expected
        elif operator == "gt":
            matched = existing is not None and existing > expected
        elif operator == "gte":
            matched = existing is not None and existing >= expected
        else:
            raise ValueError(
                "unsupported PostgREST filter operator: %s" % operator
            )
    return not matched if negate else matched


def _apply_table_query(
    rows: list[dict[str, Any]],
    query: str,
    *,
    allowed_columns: frozenset[str] | None = None,
) -> list[dict[str, Any]]:
    pairs = parse_qsl(query, keep_blank_values=True)
    referenced_columns = {
        _filter_root(name)
        for name, _value in pairs
        if name not in CONTROL_QUERY_FIELDS
    }
    if allowed_columns is not None and not referenced_columns <= allowed_columns:
        unknown = sorted(referenced_columns - allowed_columns)
        raise ValueError(
            "PostgREST filter references unknown columns: %s"
            % ",".join(unknown)
        )
    filtered = [dict(row) for row in rows]
    for name, expression in pairs:
        if name in CONTROL_QUERY_FIELDS:
            continue
        filtered = [
            row
            for row in filtered
            if _matches_filter(row, name, expression)
        ]

    orders = [value for name, value in pairs if name == "order"]
    if len(orders) > 1:
        raise ValueError("PostgREST order is duplicated")
    if orders:
        order_terms = orders[0].split(",")
        for term in reversed(order_terms):
            parts = term.split(".")
            if (
                len(parts) not in {1, 2, 3}
                or not TABLE_RE.fullmatch(parts[0])
                or (len(parts) >= 2 and parts[1] not in {"asc", "desc"})
                or (len(parts) == 3 and parts[2] not in {"nullsfirst", "nullslast"})
            ):
                raise ValueError("PostgREST order term is invalid")
            column = parts[0]
            if (
                allowed_columns is not None
                and column not in allowed_columns
            ):
                raise ValueError(
                    "PostgREST order references unknown column: %s" % column
                )
            descending = len(parts) >= 2 and parts[1] == "desc"
            nulls_first = (
                (len(parts) == 3 and parts[2] == "nullsfirst")
                or (len(parts) < 3 and descending)
            )

            def sort_key(row: dict[str, Any]) -> tuple[bool, Any]:
                value = row.get(column)
                null_bucket = value is None
                if nulls_first:
                    null_bucket = not null_bucket
                return null_bucket, value

            filtered.sort(key=sort_key, reverse=descending)

    offsets = [value for name, value in pairs if name == "offset"]
    limits = [value for name, value in pairs if name == "limit"]
    if len(offsets) > 1 or len(limits) > 1:
        raise ValueError("PostgREST pagination control is duplicated")
    offset = int(offsets[0]) if offsets else 0
    limit = int(limits[0]) if limits else None
    if offset < 0 or (limit is not None and limit < 0):
        raise ValueError("PostgREST pagination control is invalid")
    filtered = filtered[offset : None if limit is None else offset + limit]

    selections = [value for name, value in pairs if name == "select"]
    if len(selections) > 1:
        raise ValueError("PostgREST selection is duplicated")
    if selections and selections[0] != "*":
        columns = selections[0].split(",")
        if not columns or any(not TABLE_RE.fullmatch(name) for name in columns):
            raise ValueError("PostgREST selection is outside rehearsal contract")
        if allowed_columns is not None and not set(columns) <= allowed_columns:
            unknown = sorted(set(columns) - allowed_columns)
            raise ValueError(
                "PostgREST selection references unknown columns: %s"
                % ",".join(unknown)
            )
        filtered = [
            {column: row.get(column) for column in columns}
            for row in filtered
        ]
    return filtered


def _measured_query_tables(source_root: Path) -> set[str]:
    path = source_root / "gateway/tee/supabase_source_v2.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    assignments = [
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "QUERY_POLICIES"
            for target in node.targets
        )
    ]
    if len(assignments) != 1 or not isinstance(assignments[0], ast.Dict):
        raise RuntimeError("candidate measured query policies are invalid")
    tables: set[str] = set()
    for value in assignments[0].values:
        if (
            not isinstance(value, ast.Call)
            or not isinstance(value.func, ast.Name)
            or value.func.id != "SupabaseQueryV2"
        ):
            raise RuntimeError(
                "candidate measured query policy entry is invalid"
            )
        table_values = [
            keyword.value.value
            for keyword in value.keywords
            if keyword.arg == "table"
            and isinstance(keyword.value, ast.Constant)
            and isinstance(keyword.value.value, str)
        ]
        if len(table_values) != 1 or not TABLE_RE.fullmatch(table_values[0]):
            raise RuntimeError(
                "candidate measured query policy table is invalid"
            )
        tables.add(table_values[0])
    if not tables:
        raise RuntimeError("candidate measured query policy tables are empty")
    return tables


def _attested_store_tables(source_root: Path) -> set[str]:
    path = source_root / "gateway/research_lab/attested_v2_store.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    tables: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        names = [
            target.id
            for target in node.targets
            if isinstance(target, ast.Name)
            and target.id.endswith("_TABLE")
        ]
        if not names:
            continue
        if (
            len(names) != 1
            or not isinstance(node.value, ast.Constant)
            or not isinstance(node.value.value, str)
            or not TABLE_RE.fullmatch(node.value.value)
        ):
            raise RuntimeError(
                "candidate attested store table declaration is invalid"
            )
        tables.add(node.value.value)
    if not tables:
        raise RuntimeError("candidate attested store tables are empty")
    return tables


def _direct_provider_store_tables(source_root: Path) -> set[str]:
    """Load direct PostgREST table contracts from candidate provider stores."""

    tables: set[str] = set()
    paths = sorted((source_root / "gateway/tee").glob("*store_v2.py"))
    if not paths:
        raise RuntimeError("candidate provider store modules are unavailable")
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            names = [
                target.id
                for target in node.targets
                if isinstance(target, ast.Name)
                and target.id.endswith("_TABLE")
            ]
            if not names:
                continue
            if (
                len(names) != 1
                or not isinstance(node.value, ast.Constant)
                or not isinstance(node.value.value, str)
                or not TABLE_RE.fullmatch(node.value.value)
            ):
                raise RuntimeError(
                    "candidate provider store table declaration is invalid"
                )
            tables.add(node.value.value)
    if not tables:
        raise RuntimeError("candidate provider store tables are empty")
    return tables


def _schema_contract(source_root: Path) -> tuple[set[str], set[str]]:
    path = source_root / "gateway/tee/supabase_schema_preflight_v2.py"
    spec = importlib.util.spec_from_file_location(
        "_rehearsal_supabase_schema_contract", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("candidate Supabase schema contract is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tables = {
        str(row[1]) for row in module.REQUIRED_SUPABASE_V2_SCHEMA
    }
    rpcs = {str(row[1]) for row in module.REQUIRED_SUPABASE_V2_RPCS}
    return (
        tables
        | _measured_query_tables(source_root)
        | _attested_store_tables(source_root)
        | _direct_provider_store_tables(source_root)
        | set(RUNTIME_TABLES),
        rpcs | set(RUNTIME_RPCS),
    )


def _migration_schema_contract(
    path: Path,
    *,
    candidate_sha: str,
) -> tuple[dict[str, frozenset[str]], set[str]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if (
        document.get("schema_version")
        != "leadpoet.restart_rehearsal.postgres_contract.v1"
        or document.get("candidate_sha") != candidate_sha
    ):
        raise RuntimeError(
            "migration-backed schema contract differs from candidate"
        )
    expected_final_migrations = [
        "130-research-lab-provider-outcome-append.sql",
        "131-research-lab-provider-outcome-backpressure.sql",
        "132-research-lab-champion-lifetime-credit.sql",
        "133-research-lab-provider-outcome-contention-status.sql",
        "134-research-lab-provider-outcome-head-contention.sql",
        "135-research-lab-ancestry-checkpoint-sidecars.sql",
    ]
    applied_migrations = document.get("applied_migrations")
    if (
        not isinstance(applied_migrations, list)
        or applied_migrations[-6:] != expected_final_migrations
    ):
        raise RuntimeError(
            "migration-backed final migration order differs from production"
        )
    checks = document.get("checks")
    if (
        not isinstance(checks, dict)
        or not checks
        or any(value is not True for value in checks.values())
    ):
        raise RuntimeError(
            "migration-backed schema contract checks are incomplete"
        )
    raw_relations = document.get("relations")
    if not isinstance(raw_relations, dict) or not raw_relations:
        raise RuntimeError(
            "migration-backed schema contract has no relations"
        )
    relations: dict[str, frozenset[str]] = {}
    for name, relation in raw_relations.items():
        columns = relation.get("columns") if isinstance(relation, dict) else None
        if (
            not isinstance(name, str)
            or not TABLE_RE.fullmatch(name)
            or not isinstance(columns, list)
            or not columns
            or any(
                not isinstance(column, str)
                or not TABLE_RE.fullmatch(column)
                for column in columns
            )
        ):
            raise RuntimeError(
                "migration-backed relation contract is invalid"
            )
        relations[name] = frozenset(columns)
    raw_rpcs = document.get("rpcs")
    if (
        not isinstance(raw_rpcs, list)
        or any(
            not isinstance(name, str) or not TABLE_RE.fullmatch(name)
            for name in raw_rpcs
        )
    ):
        raise RuntimeError("migration-backed RPC contract is invalid")
    required_relations = {
        "research_lab_attested_transport_attempts_v2",
        "research_lab_attested_execution_receipts_v2",
        "research_lab_attested_weight_bundles_v2",
        "research_lab_attested_publication_events_v2",
        "research_lab_attested_weight_finalizations_v2",
        "research_lab_finalized_allocation_epochs_v2",
        "research_lab_emission_allocation_current",
        "research_lab_legacy_finalized_allocation_migrations_v2",
        "research_lab_chain_realized_epoch_settlements_v1",
        "research_lab_chain_realized_settlement_activation_v1",
        "research_lab_chain_realized_obligation_credits_v1",
        "research_lab_provider_outcome_checkpoints_v2",
        "research_lab_attested_ancestry_checkpoints_v2",
        "research_lab_attested_ancestry_activations_v2",
    }
    if not required_relations <= set(relations):
        raise RuntimeError(
            "migration-backed settlement relations are incomplete: %s"
            % ",".join(sorted(required_relations - set(relations)))
        )
    required_rpcs = {
        "research_lab_attested_transport_purpose_contract_v2",
        "research_lab_attested_transport_terminal_contract_v2",
        "append_research_lab_provider_outcome_checkpoint_v2",
        "research_lab_provider_outcome_contention_contract_v2",
        "research_lab_provider_outcome_contention_contract_v3",
        "persist_research_lab_chain_realized_lifetime_settlement_v2",
        "research_lab_champion_lifetime_credit_contract_v1",
        "persist_research_lab_ancestry_checkpoint_v2",
    }
    if not required_rpcs <= set(raw_rpcs):
        raise RuntimeError(
            "migration-backed transport contract RPCs are unavailable: %s"
            % ",".join(sorted(required_rpcs - set(raw_rpcs)))
        )
    return relations, set(raw_rpcs)


def _migration_provider_outcome_contract(
    path: Path,
    *,
    candidate_sha: str,
) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    contract = document.get("provider_outcome_contention_contract")
    append_evidence = document.get("provider_outcome_append")
    expected_contract = {
        "schema_version": "leadpoet.provider_outcome_contention_contract.v3",
        "lock_contention_status": "busy",
        "stale_lineage_status": "conflict",
        "candidate_checkpoint_hash": True,
        "conflict_head_checkpoint_row": "encrypted_or_null",
    }
    if (
        document.get("candidate_sha") != candidate_sha
        or contract != expected_contract
        or not isinstance(append_evidence, dict)
        or append_evidence.get("accepted_count") != 1
        or append_evidence.get("rejected_count") != 1
        or append_evidence.get("row_count") != 3
        or append_evidence.get("contention_rollback_delta") != 0
        or append_evidence.get("durable_head_conflict_verified") is not True
        or append_evidence.get("empty_head_conflict_verified") is not True
    ):
        raise RuntimeError(
            "migration-backed provider outcome contract is incomplete"
        )
    return dict(contract)


def _migration_seed_rows(
    path: Path,
    *,
    candidate_sha: str,
    relation_columns: dict[str, frozenset[str]],
) -> dict[str, list[dict[str, Any]]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    raw = document.get("seed_rows")
    expected = {
        "research_lab_finalized_allocation_epochs_v2",
        "research_lab_emission_allocation_current",
        "research_lab_legacy_finalized_allocation_migrations_v2",
        "research_lab_attested_boot_identities_v2",
        "research_lab_attested_execution_receipts_v2",
    }
    if (
        document.get("candidate_sha") != candidate_sha
        or not isinstance(raw, dict)
        or set(raw) != expected
    ):
        raise RuntimeError(
            "migration-backed allocation authority seeds are invalid"
        )
    normalized: dict[str, list[dict[str, Any]]] = {}
    for target in sorted(expected):
        rows = raw[target]
        if (
            not isinstance(rows, list)
            or len(rows) != 1
            or not isinstance(rows[0], dict)
        ):
            raise RuntimeError(
                "migration-backed allocation authority seed is invalid: %s"
                % target
            )
        row = dict(rows[0])
        expected_columns = relation_columns.get(target)
        if expected_columns is None or set(row) != set(expected_columns):
            raise RuntimeError(
                "migration-backed allocation authority seed columns differ: %s"
                % target
            )
        normalized[target] = [row]
    return normalized


class LocalPostgRESTState:
    def __init__(
        self,
        *,
        state_root: Path,
        fixture: dict[str, Any],
        source_root: Path,
        tables: set[str],
        rpcs: set[str],
        relation_columns: dict[str, frozenset[str]] | None = None,
        seed_rows: dict[str, list[dict[str, Any]]] | None = None,
        provider_outcome_contract: dict[str, Any] | None = None,
        durable_state_path: Path | None = None,
        durable_schema_sha: str = "",
    ):
        self.state_root = state_root
        self.fixture = fixture
        self.tables = tables
        self.rpcs = rpcs
        self.relation_columns = dict(relation_columns or {})
        self.lock = threading.Lock()
        self.provider_outcome_contract = dict(
            provider_outcome_contract or {}
        )
        self.durable_state_path = durable_state_path
        self.durable_schema_sha = durable_schema_sha
        self.durable_revision = 0
        self._provider_outcome_locks: dict[
            tuple[str, str], threading.Lock
        ] = {}
        self.rows: dict[str, list[dict[str, Any]]] = {
            name: [] for name in tables
        }
        for table, rows in (seed_rows or {}).items():
            if (
                table not in self.rows
                or table not in self.relation_columns
                or any(
                    not isinstance(row, dict)
                    or set(row) != set(self.relation_columns[table])
                    for row in rows
                )
            ):
                raise ValueError(
                    "local PostgREST seed rows differ from migration schema"
                )
            self.rows[table] = [dict(row) for row in rows]
        cutover = json.loads(
            (
                source_root / "config/stateful-epoch-cutover-sn71.json"
            ).read_text(encoding="utf-8")
        )
        state_table = (
            "research_lab_stateful_subnet_epoch_cutover_state_v1"
        )
        ledger_table = "research_lab_stateful_subnet_epoch_cutovers_v1"
        chain_activation_table = (
            "research_lab_chain_realized_settlement_activation_v1"
        )
        if state_table in self.rows:
            self.rows[state_table] = [
                {
                    "singleton": True,
                    "lifecycle_state": "stateful_active",
                    "mapping_hash": cutover["mapping_hash"],
                    "last_legacy_epoch_id": cutover["last_legacy_epoch_id"],
                    "first_settlement_epoch_id": (
                        cutover["first_settlement_epoch_id"]
                    ),
                }
            ]
        if ledger_table in self.rows:
            self.rows[ledger_table] = [
                {
                    "mapping_hash": cutover["mapping_hash"],
                    "manifest_doc": cutover,
                }
            ]
        if chain_activation_table in self.rows:
            network = fixture.get("network")
            if not isinstance(network, dict):
                raise ValueError("local PostgREST network fixture is invalid")
            current_block = int(network["current_block"])
            current_settlement_epoch = (
                load_rehearsal_current_settlement_epoch_id(
                    source_root
                )
            )
            first_epoch = current_settlement_epoch - 1
            if first_epoch < int(cutover["first_settlement_epoch_id"]):
                raise ValueError(
                    "local PostgREST settlement backlog predates cutover"
                )
            self.rows[chain_activation_table] = [
                {
                    "netuid": int(cutover["netuid"]),
                    "schema_version": (
                        "leadpoet.research_lab_chain_realized_settlement_activation.v1"
                    ),
                    "first_epoch_id": first_epoch,
                    "source_bundle_hash": "sha256:" + "a" * 64,
                    "source_bundle_epoch_id": first_epoch,
                    "source_finalized_block": current_block - 1,
                }
            ]
        self._restore_durable_state()
        self.cutover_state = list(self.rows.get(state_table, []))
        self.events = state_root / "local-postgrest-events.jsonl"
        with self.lock:
            self._write_durable_state_locked()

    def _restore_durable_state(self) -> None:
        path = self.durable_state_path
        if path is None or not path.exists():
            return
        document = json.loads(path.read_text(encoding="utf-8"))
        state = {
            "schema_version": document.get("schema_version"),
            "durable_schema_sha": document.get("durable_schema_sha"),
            "revision": document.get("revision"),
            "rows": document.get("rows"),
        }
        expected_hash = "sha256:" + hashlib.sha256(
            json.dumps(
                state,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        rows = state["rows"]
        if (
            state["schema_version"]
            != "leadpoet.local_postgrest_durable_state.v1"
            or state["durable_schema_sha"] != self.durable_schema_sha
            or not isinstance(state["revision"], int)
            or state["revision"] < 0
            or not isinstance(rows, dict)
            or set(rows) != set(self.rows)
            or document.get("state_hash") != expected_hash
        ):
            raise ValueError("durable PostgREST state identity differs")
        restored: dict[str, list[dict[str, Any]]] = {}
        for table, table_rows in rows.items():
            if (
                not isinstance(table_rows, list)
                or any(not isinstance(row, dict) for row in table_rows)
            ):
                raise ValueError("durable PostgREST rows are invalid")
            allowed = self.relation_columns.get(table)
            if allowed is not None and any(
                not set(row).issubset(allowed) for row in table_rows
            ):
                raise ValueError(
                    "durable PostgREST row columns differ from schema"
                )
            restored[table] = [dict(row) for row in table_rows]
        self.rows = restored
        self.durable_revision = int(state["revision"])

    def _write_durable_state_locked(self, *, mutated: bool = False) -> None:
        path = self.durable_state_path
        if path is None:
            return
        if mutated:
            self.durable_revision += 1
        state = {
            "schema_version": "leadpoet.local_postgrest_durable_state.v1",
            "durable_schema_sha": self.durable_schema_sha,
            "revision": self.durable_revision,
            "rows": self.rows,
        }
        document = {
            **state,
            "state_hash": "sha256:"
            + hashlib.sha256(
                json.dumps(
                    state,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(
            f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        temporary.write_text(
            json.dumps(
                document,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.chmod(0o600)
        os.replace(temporary, path)

    def durable_state_identity(self) -> dict[str, Any]:
        path = self.durable_state_path
        if path is None or not path.is_file():
            raise ValueError("durable PostgREST state is unavailable")
        document = json.loads(path.read_text(encoding="utf-8"))
        if (
            document.get("schema_version")
            != "leadpoet.local_postgrest_durable_state.v1"
            or document.get("durable_schema_sha")
            != self.durable_schema_sha
            or not isinstance(document.get("revision"), int)
            or not re.fullmatch(
                r"sha256:[0-9a-f]{64}",
                str(document.get("state_hash") or ""),
            )
        ):
            raise ValueError("durable PostgREST state identity is invalid")
        return {
            "durable_schema_sha": document["durable_schema_sha"],
            "revision": document["revision"],
            "state_hash": document["state_hash"],
        }

    def _provider_outcome_lock(
        self,
        key_ref_hash: str,
        utc_day: str,
    ) -> threading.Lock:
        identity = (key_ref_hash, utc_day)
        with self.lock:
            lock = self._provider_outcome_locks.get(identity)
            if lock is None:
                lock = threading.Lock()
                self._provider_outcome_locks[identity] = lock
            return lock

    def append_provider_outcome_checkpoint(
        self,
        body: Any,
    ) -> dict[str, Any]:
        if self.provider_outcome_contract.get("schema_version") != (
            "leadpoet.provider_outcome_contention_contract.v3"
        ):
            raise ValueError(
                "provider outcome migration contract is unavailable"
            )
        if not isinstance(body, dict) or set(body) != {"checkpoint_row"}:
            raise ValueError("provider outcome checkpoint RPC body is invalid")
        row = body.get("checkpoint_row")
        table = "research_lab_provider_outcome_checkpoints_v2"
        relation_columns = self.relation_columns.get(table)
        expected_columns = (
            relation_columns - {"created_at"}
            if relation_columns is not None
            else None
        )
        if (
            not isinstance(row, dict)
            or expected_columns is None
            or set(row) != set(expected_columns)
            or row.get("schema_version")
            != "leadpoet.provider_outcome_checkpoint_row.v2"
            or not isinstance(row.get("sequence"), int)
            or isinstance(row.get("sequence"), bool)
            or int(row["sequence"]) <= 0
            or not isinstance(row.get("encrypted_checkpoint_doc"), dict)
        ):
            raise ValueError(
                "provider outcome checkpoint fields are invalid"
            )
        key_ref_hash = str(row.get("artifact_master_key_ref_hash") or "")
        utc_day = str(row.get("utc_day") or "")
        checkpoint_hash = str(row.get("checkpoint_hash") or "")
        previous_hash = str(row.get("previous_checkpoint_hash") or "")
        hash_fields = {
            "artifact_master_key_ref_hash",
            "checkpoint_hash",
            "state_document_hash",
            "checkpoint_artifact_id",
        }
        try:
            parsed_day = date.fromisoformat(utc_day)
        except ValueError as exc:
            raise ValueError(
                "provider outcome checkpoint identity is invalid"
            ) from exc
        if (
            not HASH_RE.fullmatch(key_ref_hash)
            or not DAY_RE.fullmatch(utc_day)
            or parsed_day.isoformat() != utc_day
            or not HASH_RE.fullmatch(checkpoint_hash)
            or (
                previous_hash
                and not HASH_RE.fullmatch(previous_hash)
            )
            or any(
                not HASH_RE.fullmatch(str(row.get(field) or ""))
                for field in hash_fields
            )
        ):
            raise ValueError(
                "provider outcome checkpoint identity is invalid"
            )

        lineage_lock = self._provider_outcome_lock(key_ref_hash, utc_day)
        if not lineage_lock.acquire(blocking=False):
            return {
                "status": str(
                    self.provider_outcome_contract[
                        "lock_contention_status"
                    ]
                ),
                "checkpoint_hash": checkpoint_hash,
            }
        try:
            with self.lock:
                rows = self.rows[table]
                existing = next(
                    (
                        stored
                        for stored in rows
                        if stored.get("checkpoint_hash") == checkpoint_hash
                    ),
                    None,
                )
                if existing is not None:
                    durable = {
                        field: value
                        for field, value in existing.items()
                        if field != "created_at"
                    }
                    if durable != row:
                        raise ValueError(
                            "provider outcome checkpoint hash already "
                            "identifies another row"
                        )
                    return {
                        "status": "existing",
                        "checkpoint_hash": checkpoint_hash,
                    }
                lineage = [
                    stored
                    for stored in rows
                    if (
                        stored.get("artifact_master_key_ref_hash")
                        == key_ref_hash
                        and stored.get("utc_day") == utc_day
                    )
                ]
                current = (
                    max(lineage, key=lambda stored: int(stored["sequence"]))
                    if lineage
                    else None
                )
                current_row = (
                    {
                        field: value
                        for field, value in current.items()
                        if field != "created_at"
                    }
                    if current is not None
                    else None
                )
                sequence = int(row["sequence"])
                expected_sequence = (
                    int(current["sequence"]) + 1
                    if current is not None
                    else 1
                )
                expected_previous = (
                    str(current["checkpoint_hash"])
                    if current is not None
                    else ""
                )
                if (
                    sequence != expected_sequence
                    or previous_hash != expected_previous
                ):
                    return {
                        "status": str(
                            self.provider_outcome_contract[
                                "stale_lineage_status"
                            ]
                        ),
                        "checkpoint_hash": checkpoint_hash,
                        "head_checkpoint_row": current_row,
                    }
                stored = dict(row)
                stored["created_at"] = "2026-07-25T00:00:00+00:00"
                rows.append(stored)
                self._write_durable_state_locked(mutated=True)
                durable = {
                    field: value
                    for field, value in stored.items()
                    if field != "created_at"
                }
                if durable != row:
                    raise ValueError(
                        "provider outcome checkpoint durable insert differs"
                    )
            return {
                "status": "inserted",
                "checkpoint_hash": checkpoint_hash,
            }
        finally:
            lineage_lock.release()

    def persist_ancestry_checkpoint(
        self,
        body: Any,
    ) -> dict[str, Any]:
        """Mirror migration 135's atomic checkpoint RPC at the boundary."""

        if not isinstance(body, dict) or set(body) != {"checkpoint"}:
            raise ValueError("ancestry checkpoint RPC body is invalid")
        row = body.get("checkpoint")
        checkpoint_table = (
            "research_lab_attested_ancestry_checkpoints_v2"
        )
        activation_table = (
            "research_lab_attested_ancestry_activations_v2"
        )
        checkpoint_columns = self.relation_columns.get(checkpoint_table)
        activation_columns = self.relation_columns.get(activation_table)
        expected_columns = (
            checkpoint_columns - {"created_at"}
            if checkpoint_columns is not None
            else None
        )
        if (
            not isinstance(row, dict)
            or expected_columns is None
            or activation_columns is None
            or set(row) != set(expected_columns)
            or row.get("schema_version")
            != "leadpoet.attested_ancestry_certificate.v2"
            or not isinstance(row.get("certificate_sequence"), int)
            or isinstance(row.get("certificate_sequence"), bool)
            or int(row["certificate_sequence"]) < 0
        ):
            raise ValueError("ancestry checkpoint fields are invalid")

        hash_fields = {
            "root_receipt_hash",
            "lineage_id",
            "certificate_hash",
            "issuer_boot_identity_hash",
            "proof_hash",
            "checkpoint_graph_hash",
        }
        if any(
            not HASH_RE.fullmatch(str(row.get(field) or ""))
            for field in hash_fields
        ):
            raise ValueError("ancestry checkpoint identity is invalid")

        certificate = row.get("certificate_doc")
        proof = row.get("proof_doc")
        graph = row.get("checkpoint_graph_doc")
        if not all(
            isinstance(value, dict)
            for value in (certificate, proof, graph)
        ):
            raise ValueError("ancestry checkpoint documents are invalid")
        serialized_documents = json.dumps(
            [certificate, proof, graph],
            sort_keys=True,
            separators=(",", ":"),
        )
        if SENSITIVE_DOCUMENT_RE.search(serialized_documents):
            raise ValueError("ancestry checkpoint documents contain a secret")

        claim = certificate.get("claim")
        proof_certificate = proof.get("certificate")
        graph_proof = graph.get("ancestry_proof")
        sequence = int(row["certificate_sequence"])
        root_hash = str(row["root_receipt_hash"])
        lineage = str(row["lineage_id"])
        certificate_hash = str(row["certificate_hash"])
        if (
            not isinstance(claim, dict)
            or not isinstance(claim.get("parent_authorities"), list)
            or certificate.get("schema_version") != row["schema_version"]
            or certificate.get("certificate_hash") != certificate_hash
            or claim.get("output_root_receipt_hash") != root_hash
            or claim.get("lineage_id") != lineage
            or claim.get("certificate_sequence") != sequence
            or claim.get("issuer_boot_identity_hash")
            != row["issuer_boot_identity_hash"]
            or proof.get("schema_version")
            != "leadpoet.attested_ancestry_compact_proof.v2"
            or proof.get("proof_hash") != row["proof_hash"]
            or proof_certificate != certificate
            or graph.get("schema_version")
            != "leadpoet.attested_checkpointed_receipt_graph.v3"
            or graph.get("root_receipt_hash") != root_hash
            or graph.get("ancestry_lineage_id") != lineage
            or graph_proof != proof
        ):
            raise ValueError("ancestry checkpoint document identity differs")

        with self.lock:
            receipt_rows = self.rows.get(
                "research_lab_attested_execution_receipts_v2", []
            )
            boot_rows = self.rows.get(
                "research_lab_attested_boot_identities_v2", []
            )
            if not any(
                stored.get("receipt_hash") == root_hash
                for stored in receipt_rows
            ):
                raise ValueError(
                    "ancestry checkpoint receipt is not durable"
                )
            if not any(
                stored.get("boot_identity_hash")
                == row["issuer_boot_identity_hash"]
                for stored in boot_rows
            ):
                raise ValueError(
                    "ancestry checkpoint issuer boot is not durable"
                )
            checkpoints = self.rows[checkpoint_table]
            activations = self.rows[activation_table]
            for parent in claim["parent_authorities"]:
                if not isinstance(parent, dict):
                    raise ValueError(
                        "ancestry checkpoint parent authority is invalid"
                    )
                kind = str(parent.get("authority_kind") or "")
                parent_root = str(parent.get("parent_receipt_hash") or "")
                if kind == "full_projection":
                    if any(
                        stored.get("lineage_id") == lineage
                        and stored.get("activation_root_receipt_hash")
                        == parent_root
                        for stored in activations
                    ):
                        raise ValueError(
                            "compacted ancestry root rejects full graph parent"
                        )
                elif kind == "certificate":
                    parent_sequence = parent.get("authority_sequence")
                    if (
                        not isinstance(parent_sequence, int)
                        or isinstance(parent_sequence, bool)
                        or not any(
                            stored.get("root_receipt_hash") == parent_root
                            and stored.get("lineage_id") == lineage
                            and stored.get("certificate_hash")
                            == parent.get("authority_hash")
                            and stored.get("certificate_sequence")
                            == parent_sequence
                            and int(parent_sequence) < sequence
                            for stored in checkpoints
                        )
                    ):
                        raise ValueError(
                            "checkpoint certificate parent is not durable"
                        )
                elif kind == "certificate_disclosure":
                    parent_sequence = parent.get("authority_sequence")
                    if (
                        not isinstance(parent_sequence, int)
                        or isinstance(parent_sequence, bool)
                        or not any(
                            stored.get("lineage_id") == lineage
                            and stored.get("certificate_sequence")
                            == parent_sequence
                            and int(parent_sequence) < sequence
                            and any(
                                isinstance(disclosed, dict)
                                and disclosed.get("receipt_hash") == parent_root
                                for disclosed in (
                                    stored.get("proof_doc", {}).get(
                                        "disclosed_receipts", []
                                    )
                                    if isinstance(stored.get("proof_doc"), dict)
                                    else []
                                )
                            )
                            for stored in checkpoints
                        )
                    ):
                        raise ValueError(
                            "checkpoint disclosure parent is not durable"
                        )
                else:
                    raise ValueError(
                        "checkpoint parent authority kind is invalid"
                    )

            stored = next(
                (
                    existing
                    for existing in checkpoints
                    if existing.get("root_receipt_hash") == root_hash
                ),
                None,
            )
            mutated = False
            if stored is None:
                for unique_field in (
                    "certificate_hash",
                    "proof_hash",
                    "checkpoint_graph_hash",
                ):
                    if any(
                        existing.get(unique_field) == row[unique_field]
                        for existing in checkpoints
                    ):
                        raise ValueError(
                            "ancestry checkpoint unique identity conflicts"
                        )
                stored = dict(row)
                stored["created_at"] = "2026-07-25T00:00:00+00:00"
                checkpoints.append(stored)
                mutated = True
            elif {
                field: value
                for field, value in stored.items()
                if field != "created_at"
            } != row:
                raise ValueError("checkpoint durable readback conflicts")

            activation = next(
                (
                    existing
                    for existing in activations
                    if existing.get("activation_root_receipt_hash")
                    == root_hash
                ),
                None,
            )
            expected_activation = {
                "lineage_id": lineage,
                "activation_root_receipt_hash": root_hash,
                "activation_certificate_hash": certificate_hash,
            }
            if activation is None:
                if any(
                    existing.get("activation_certificate_hash")
                    == certificate_hash
                    for existing in activations
                ):
                    raise ValueError("ancestry root activation conflicts")
                activation = {
                    **expected_activation,
                    "activated_at": "2026-07-25T00:00:00+00:00",
                }
                activations.append(activation)
                mutated = True
            elif {
                field: value
                for field, value in activation.items()
                if field != "activated_at"
            } != expected_activation:
                raise ValueError("ancestry root activation conflicts")
            self._write_durable_state_locked(mutated=mutated)

        return {
            "status": "persisted",
            "root_receipt_hash": root_hash,
            "lineage_id": lineage,
            "certificate_hash": certificate_hash,
            "certificate_sequence": sequence,
            "proof_hash": row["proof_hash"],
            "checkpoint_graph_hash": row["checkpoint_graph_hash"],
            "root_activated": True,
        }

    def persist_chain_realized_settlement(
        self,
        *,
        rpc_name: str,
        body: Any,
    ) -> dict[str, Any]:
        if not isinstance(body, dict) or set(body) != {
            "requested_settlement",
            "requested_credits",
        }:
            raise ValueError("chain settlement RPC body is invalid")
        settlement = body.get("requested_settlement")
        credits = body.get("requested_credits")
        if not isinstance(settlement, dict) or not isinstance(credits, list):
            raise ValueError("chain settlement RPC payload is invalid")
        required_settlement_fields = {
            "netuid",
            "epoch_id",
            "schema_version",
            "settlement_hash",
            "settlement_receipt_hash",
            "settlement_doc",
        }
        if set(settlement) != required_settlement_fields:
            raise ValueError("chain settlement row fields are invalid")
        netuid = int(settlement["netuid"])
        epoch_id = int(settlement["epoch_id"])
        schema_version = str(settlement["schema_version"])
        expected_schema = {
            "persist_research_lab_chain_realized_settlement_v1": (
                "leadpoet.research_lab_chain_realized_epoch_settlement.v1"
            ),
            "persist_research_lab_chain_realized_unattributed_v2": (
                "leadpoet.research_lab_chain_realized_epoch_settlement.v2"
            ),
            "persist_research_lab_chain_realized_lifetime_settlement_v2": (
                "leadpoet.research_lab_chain_realized_epoch_settlement.v3"
            ),
        }.get(rpc_name)
        if expected_schema is None:
            raise ValueError("unknown chain settlement persistence RPC")
        if schema_version != expected_schema:
            raise ValueError("chain settlement RPC schema differs")
        if expected_schema.endswith(".v2") and credits:
            raise ValueError("unattributed chain settlement contains credits")
        settlement_doc = settlement.get("settlement_doc")
        if (
            not isinstance(settlement_doc, dict)
            or settlement_doc.get("schema_version") != schema_version
            or int(settlement_doc.get("netuid", -1)) != netuid
            or int(settlement_doc.get("epoch_id", -1)) != epoch_id
        ):
            raise ValueError("chain settlement document differs")
        lifetime_policy = "accelerated_lifetime_cap_v1"
        if expected_schema.endswith(".v3"):
            if (
                settlement_doc.get("champion_credit_policy")
                != lifetime_policy
            ):
                raise ValueError(
                    "lifetime chain settlement policy differs"
                )
        elif "champion_credit_policy" in settlement_doc:
            raise ValueError(
                "legacy chain settlement contains a lifetime policy"
            )
        expected_hashes = sorted(
            str(item)
            for item in (settlement_doc.get("credit_hashes") or ())
        )
        requested_hashes = sorted(
            str(item.get("credit_hash") or "")
            for item in credits
            if isinstance(item, dict)
        )
        if len(requested_hashes) != len(credits) or (
            requested_hashes != expected_hashes
        ):
            raise ValueError("chain settlement credit set differs")

        settlement_table = (
            "research_lab_chain_realized_epoch_settlements_v1"
        )
        credit_table = (
            "research_lab_chain_realized_obligation_credits_v1"
        )
        activation_table = (
            "research_lab_chain_realized_settlement_activation_v1"
        )
        with self.lock:
            mutated = False
            activation = [
                row
                for row in self.rows[activation_table]
                if int(row["netuid"]) == netuid
            ]
            if len(activation) != 1:
                raise ValueError("chain settlement activation is ambiguous")
            first_epoch = int(activation[0]["first_epoch_id"])
            if epoch_id < first_epoch:
                raise ValueError("chain settlement predates activation")
            if epoch_id > first_epoch and not any(
                int(row["netuid"]) == netuid
                and int(row["epoch_id"]) == epoch_id - 1
                for row in self.rows[settlement_table]
            ):
                raise ValueError("chain settlement predecessor is missing")

            existing = next(
                (
                    row
                    for row in self.rows[settlement_table]
                    if int(row["netuid"]) == netuid
                    and int(row["epoch_id"]) == epoch_id
                ),
                None,
            )
            exact_settlement = dict(settlement)
            if existing is None:
                exact_settlement["created_at"] = (
                    "2026-07-25T00:00:00+00:00"
                )
                self.rows[settlement_table].append(exact_settlement)
                mutated = True
            elif any(
                existing.get(field) != settlement.get(field)
                for field in required_settlement_fields
            ):
                raise ValueError("chain settlement conflicts with durable row")

            for credit in credits:
                if not isinstance(credit, dict):
                    raise ValueError("chain settlement credit row is invalid")
                credit_doc = credit.get("credit_doc")
                if (
                    not isinstance(credit_doc, dict)
                    or credit_doc.get("schema_version")
                    != credit.get("schema_version")
                ):
                    raise ValueError(
                        "chain settlement credit document differs"
                    )
                if expected_schema.endswith(".v3"):
                    if (
                        credit.get("schema_version")
                        != (
                            "leadpoet.research_lab_chain_realized_"
                            "obligation_credit.v2"
                        )
                        or credit.get("champion_credit_policy")
                        != lifetime_policy
                        or credit_doc.get("champion_credit_policy")
                        != lifetime_policy
                        or (
                            credit.get("obligation_kind")
                            in {"champion", "queued_champion"}
                            and credit.get("credited_alpha_percent")
                            != credit.get("lab_attributed_alpha_percent")
                        )
                    ):
                        raise ValueError(
                            "lifetime chain settlement credit differs"
                        )
                elif (
                    credit.get("schema_version")
                    != (
                        "leadpoet.research_lab_chain_realized_"
                        "obligation_credit.v1"
                    )
                    or credit.get("champion_credit_policy")
                    != "scheduled_bonus_v1"
                    or "champion_credit_policy" in credit_doc
                ):
                    raise ValueError(
                        "legacy chain settlement credit differs"
                    )
                existing_credit = next(
                    (
                        row
                        for row in self.rows[credit_table]
                        if int(row["netuid"]) == netuid
                        and int(row["epoch_id"]) == epoch_id
                        and row.get("obligation_kind")
                        == credit.get("obligation_kind")
                        and row.get("obligation_source_id")
                        == credit.get("obligation_source_id")
                    ),
                    None,
                )
                if existing_credit is None:
                    stored_credit = dict(credit)
                    stored_credit["created_at"] = (
                        "2026-07-25T00:00:00+00:00"
                    )
                    self.rows[credit_table].append(stored_credit)
                    mutated = True
                elif any(
                    existing_credit.get(field) != value
                    for field, value in credit.items()
                ):
                    raise ValueError(
                        "chain settlement credit conflicts with durable row"
                    )
            durable_hashes = sorted(
                str(row.get("credit_hash") or "")
                for row in self.rows[credit_table]
                if int(row["netuid"]) == netuid
                and int(row["epoch_id"]) == epoch_id
                and row.get("settlement_hash")
                == settlement.get("settlement_hash")
            )
            if durable_hashes != expected_hashes:
                raise ValueError("durable chain settlement credit set differs")
            self._write_durable_state_locked(mutated=mutated)

        return {
            "schema_version": (
                "leadpoet.research_lab_chain_realized_"
                "settlement_persistence.v1"
            ),
            "netuid": netuid,
            "epoch_id": epoch_id,
            "settlement_hash": str(settlement["settlement_hash"]),
            "settlement_receipt_hash": str(
                settlement["settlement_receipt_hash"]
            ),
            "credit_count": len(credits),
            "credit_hashes": expected_hashes,
        }

    def record(self, **row: Any) -> None:
        payload = {
            "at_ns": time.time_ns(),
            "kind": "local-postgrest",
            "implementation": "external_boundary",
            "fixture_authenticity": "production_shaped_sanitized",
            "reject_unknown": True,
            **row,
        }
        encoded = (
            json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        descriptor = os.open(
            self.events,
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
            0o600,
        )
        try:
            os.write(descriptor, encoded)
        finally:
            os.close(descriptor)


class Handler(BaseHTTPRequestHandler):
    server: "LocalPostgRESTServer"

    def log_message(self, _format: str, *args: Any) -> None:
        del args

    def _json_response(
        self,
        status: int,
        value: Any,
        *,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        body = json.dumps(
            value, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Content-Range", "0-0/0")
        for name, value in (extra_headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self) -> bool:
        apikey = self.headers.get("apikey", "")
        authorization = self.headers.get("authorization", "")
        return apikey in {"rehearsal-public", "rehearsal-secret"} and (
            authorization == f"Bearer {apikey}"
        )

    def _body(self) -> Any:
        size = int(self.headers.get("content-length", "0") or 0)
        if size > 8 * 1024 * 1024:
            raise ValueError("local PostgREST request exceeds rehearsal bound")
        if size == 0:
            return None
        value = json.loads(self.rfile.read(size))
        if not isinstance(value, (dict, list)):
            raise ValueError("local PostgREST body must be an object or array")
        return value

    def _dispatch(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self._json_response(200, {"status": "ready"})
            return
        if not self._authorized():
            self.server.state.record(
                status="rejected",
                operation="authentication",
                method=self.command,
                path=parsed.path,
            )
            self._json_response(401, {"message": "local authentication rejected"})
            return
        if parsed.path == "/rest/v1/":
            paths = {
                f"/rpc/{name}": {} for name in sorted(self.server.state.rpcs)
            }
            self.server.state.record(
                status="ok",
                operation="rpc",
                method=self.command,
                path=parsed.path,
            )
            self._json_response(200, {"paths": paths})
            return
        prefix = "/rest/v1/"
        if not parsed.path.startswith(prefix):
            self._json_response(404, {"message": "unknown local service path"})
            return
        target = parsed.path[len(prefix) :]
        if target.startswith("rpc/"):
            name = target[4:]
            if name not in self.server.state.rpcs or self.command != "POST":
                self.server.state.record(
                    status="rejected",
                    operation="rpc",
                    method=self.command,
                    target=name,
                )
                self._json_response(404, {"message": "unknown local RPC"})
                return
            body = self._body()
            self.server.state.record(
                status="ok",
                operation="rpc",
                method=self.command,
                target=name,
            )
            response: Any = []
            if name == (
                "research_lab_stateful_subnet_epoch_cutover_public_state_v1"
            ):
                response = self.server.state.cutover_state
            elif name == "research_lab_source_add_claim_work":
                # The exact fixture intentionally has no source-add work.
                response = []
            elif name in {
                "persist_research_lab_chain_realized_settlement_v1",
                "persist_research_lab_chain_realized_unattributed_v2",
                "persist_research_lab_chain_realized_lifetime_settlement_v2",
            }:
                response = self.server.state.persist_chain_realized_settlement(
                    rpc_name=name,
                    body=body,
                )
                self.server.state.record(
                    status="ok",
                    operation="chain_settlement_persisted",
                    method=self.command,
                    target=name,
                    netuid=response["netuid"],
                    epoch_id=response["epoch_id"],
                    settlement_hash=response["settlement_hash"],
                    credit_count=response["credit_count"],
                )
            elif name == "append_research_lab_provider_outcome_checkpoint_v2":
                response = (
                    self.server.state.append_provider_outcome_checkpoint(body)
                )
                checkpoint_row = (
                    body.get("checkpoint_row")
                    if isinstance(body, dict)
                    else None
                )
                self.server.state.record(
                    status="ok",
                    operation="provider_outcome_checkpoint_appended",
                    method=self.command,
                    target=name,
                    result_status=response["status"],
                    checkpoint_hash=response["checkpoint_hash"],
                    sequence=(
                        checkpoint_row.get("sequence")
                        if isinstance(checkpoint_row, dict)
                        else None
                    ),
                )
            elif name == "persist_research_lab_ancestry_checkpoint_v2":
                response = self.server.state.persist_ancestry_checkpoint(body)
                self.server.state.record(
                    status="ok",
                    operation="ancestry_checkpoint_persisted",
                    method=self.command,
                    target=name,
                    root_receipt_hash=response["root_receipt_hash"],
                    lineage_id=response["lineage_id"],
                    certificate_hash=response["certificate_hash"],
                    certificate_sequence=response["certificate_sequence"],
                    proof_hash=response["proof_hash"],
                    checkpoint_graph_hash=response[
                        "checkpoint_graph_hash"
                    ],
                    root_activated=response["root_activated"],
                )
            else:
                raise ValueError(
                    "declared RPC lacks a strict rehearsal implementation"
                )
            self._json_response(200, response)
            return
        if (
            not TABLE_RE.fullmatch(target)
            or target not in self.server.state.tables
        ):
            self.server.state.record(
                status="rejected",
                operation="table",
                method=self.command,
                target=target,
            )
            self._json_response(404, {"message": "unknown local table"})
            return
        if self.command == "GET":
            with self.server.state.lock:
                rows = list(self.server.state.rows[target])
            operation = "select"
            status = 200
            response = _apply_table_query(
                rows,
                parsed.query,
                allowed_columns=self.server.state.relation_columns.get(
                    target
                ),
            )
            if target == "research_lab_provider_outcome_checkpoints_v2":
                self.server.state.record(
                    status="ok",
                    operation="provider_outcome_checkpoint_readback",
                    method=self.command,
                    target=target,
                    row_count=len(response),
                    checkpoint_hashes=[
                        str(row.get("checkpoint_hash") or "")
                        for row in response
                    ],
                    sequences=[
                        int(row.get("sequence") or 0)
                        for row in response
                    ],
                )
        elif self.command == "POST":
            body = self._body()
            incoming = body if isinstance(body, list) else [body]
            if any(not isinstance(row, dict) for row in incoming):
                raise ValueError("local PostgREST rows must be objects")
            allowed_columns = self.server.state.relation_columns.get(target)
            if allowed_columns is not None:
                unknown = sorted(
                    {
                        column
                        for row in incoming
                        for column in row
                        if column not in allowed_columns
                    }
                )
                if unknown:
                    raise ValueError(
                        "PostgREST insert references unknown columns: %s"
                        % ",".join(unknown)
                    )
            with self.server.state.lock:
                self.server.state.rows[target].extend(incoming)
                self.server.state._write_durable_state_locked(
                    mutated=bool(incoming)
                )
            operation = "insert"
            status = 201
            response = incoming if "return=representation" in (
                self.headers.get("prefer", "")
            ) else []
        elif self.command == "PATCH":
            body = self._body()
            if not isinstance(body, dict):
                raise ValueError("local PostgREST patch must be an object")
            allowed_columns = self.server.state.relation_columns.get(target)
            if allowed_columns is not None:
                unknown = sorted(set(body) - allowed_columns)
                if unknown:
                    raise ValueError(
                        "PostgREST patch references unknown columns: %s"
                        % ",".join(unknown)
                    )
            operation = "insert"
            status = 200
            response = []
        else:
            self._json_response(405, {"message": "method not allowed"})
            return
        evidence = {
            "status": "ok",
            "operation": operation,
            "method": self.command,
            "target": target,
            "query": parsed.query,
        }
        if operation == "select":
            evidence["row_count"] = len(response)
        self.server.state.record(
            **evidence,
        )
        self._json_response(status, response)

    def do_GET(self) -> None:
        self._handle()

    def do_POST(self) -> None:
        self._handle()

    def do_PATCH(self) -> None:
        self._handle()

    def _handle(self) -> None:
        try:
            self._dispatch()
        except (KeyError, TypeError, ValueError) as exc:
            self.server.state.record(
                status="rejected",
                operation="request_validation",
                method=self.command,
                path=self.path,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            self._json_response(
                400,
                {
                    "code": "PGRST204",
                    "message": str(exc),
                },
            )


class LocalPostgRESTServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], state: LocalPostgRESTState):
        self.state = state
        super().__init__(address, Handler)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=54321)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--schema-contract", type=Path, required=True)
    parser.add_argument("--candidate-sha", required=True)
    parser.add_argument("--durable-state", type=Path)
    args = parser.parse_args()
    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    if fixture.get("sanitization", {}).get("contains_production_credentials"):
        raise RuntimeError("local PostgREST fixture contains credentials")
    tables, rpcs = _schema_contract(args.source_root)
    relation_columns, migration_rpcs = _migration_schema_contract(
        args.schema_contract,
        candidate_sha=args.candidate_sha,
    )
    provider_outcome_contract = _migration_provider_outcome_contract(
        args.schema_contract,
        candidate_sha=args.candidate_sha,
    )
    seed_rows = _migration_seed_rows(
        args.schema_contract,
        candidate_sha=args.candidate_sha,
        relation_columns=relation_columns,
    )
    tables.update(relation_columns)
    rpcs.update(migration_rpcs)
    args.state_root.mkdir(parents=True, exist_ok=True)
    state = LocalPostgRESTState(
        state_root=args.state_root,
        fixture=fixture,
        source_root=args.source_root,
        tables=tables,
        rpcs=rpcs,
        relation_columns=relation_columns,
        seed_rows=seed_rows,
        provider_outcome_contract=provider_outcome_contract,
        durable_state_path=args.durable_state,
        durable_schema_sha=args.candidate_sha,
    )
    durable_identity = state.durable_state_identity()
    server = LocalPostgRESTServer((args.host, args.port), state)
    (args.state_root / "local-postgrest.ready").write_text(
        json.dumps(
            {
                "schema_version": "leadpoet.local_postgrest.v1",
                "host": args.host,
                "port": args.port,
                "tables": len(tables),
                "rpcs": len(rpcs),
                "migration_backed_relations": len(relation_columns),
                "durable_schema_sha": args.candidate_sha,
                "durable_revision": durable_identity["revision"],
                "durable_state_hash": durable_identity["state_hash"],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    state.record(status="ready", operation="service_start")
    try:
        server.serve_forever(poll_interval=0.1)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
