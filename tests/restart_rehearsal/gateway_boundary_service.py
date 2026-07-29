#!/usr/bin/env python3.11
"""Strict local PostgREST equivalent for the exact gateway launcher replay."""

from __future__ import annotations

import argparse
import ast
import csv
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


RUNTIME_TABLES = frozenset(
    {
        "epoch_audit_logs",
        "leads_private",
        "merkle_checkpoints",
        "published_weight_bundles",
        "qualification_baselines",
        "qualification_private_icp_sets",
        "research_lab_champion_reward_current",
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


def _matches_filter(row: dict[str, Any], column: str, expression: str) -> bool:
    if not TABLE_RE.fullmatch(column):
        raise ValueError("PostgREST filter column is invalid")
    if "." not in expression:
        raise ValueError("PostgREST filter operator is missing")
    operator, raw = expression.split(".", 1)
    existing = row.get(column)
    if operator == "is":
        expected = {"null": None, "true": True, "false": False}.get(raw.lower())
        if raw.lower() not in {"null", "true", "false"}:
            raise ValueError("PostgREST is filter is invalid")
        return existing is expected if expected is None else existing == expected
    if operator == "in":
        return existing in _in_values(raw, existing)
    expected = _filter_scalar(raw, existing)
    if operator == "eq":
        return existing == expected
    if operator == "neq":
        return existing != expected
    if operator == "lt":
        return existing is not None and existing < expected
    if operator == "lte":
        return existing is not None and existing <= expected
    if operator == "gt":
        return existing is not None and existing > expected
    if operator == "gte":
        return existing is not None and existing >= expected
    raise ValueError("unsupported PostgREST filter operator: %s" % operator)


def _apply_table_query(
    rows: list[dict[str, Any]],
    query: str,
    *,
    allowed_columns: frozenset[str] | None = None,
) -> list[dict[str, Any]]:
    pairs = parse_qsl(query, keep_blank_values=True)
    referenced_columns = {
        name for name, _value in pairs if name not in CONTROL_QUERY_FIELDS
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
        "research_lab_chain_realized_epoch_settlements_v1",
        "research_lab_chain_realized_settlement_activation_v1",
        "research_lab_chain_realized_obligation_credits_v1",
        "research_lab_provider_outcome_checkpoints_v2",
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
        "persist_research_lab_chain_realized_lifetime_settlement_v2",
        "research_lab_champion_lifetime_credit_contract_v1",
    }
    if not required_rpcs <= set(raw_rpcs):
        raise RuntimeError(
            "migration-backed transport contract RPCs are unavailable: %s"
            % ",".join(sorted(required_rpcs - set(raw_rpcs)))
        )
    return relations, set(raw_rpcs)


def _migration_seed_rows(
    path: Path,
    *,
    candidate_sha: str,
    relation_columns: dict[str, frozenset[str]],
) -> dict[str, list[dict[str, Any]]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    raw = document.get("seed_rows")
    target = "research_lab_finalized_allocation_epochs_v2"
    if (
        document.get("candidate_sha") != candidate_sha
        or not isinstance(raw, dict)
        or set(raw) != {target}
        or not isinstance(raw[target], list)
        or len(raw[target]) != 1
        or not isinstance(raw[target][0], dict)
    ):
        raise RuntimeError(
            "migration-backed finalized authority seed is invalid"
        )
    row = dict(raw[target][0])
    expected_columns = relation_columns.get(target)
    if expected_columns is None or set(row) != set(expected_columns):
        raise RuntimeError(
            "migration-backed finalized authority seed columns differ"
        )
    return {target: [row]}


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
    ):
        self.state_root = state_root
        self.fixture = fixture
        self.tables = tables
        self.rpcs = rpcs
        self.relation_columns = dict(relation_columns or {})
        self.lock = threading.Lock()
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
            subnet_epoch_index = int(network["subnet_epoch_index"])
            current_block = int(network["current_block"])
            first_subnet_epoch = int(cutover["first_subnet_epoch_index"])
            if (
                subnet_epoch_index < first_subnet_epoch
                or current_block <= int(cutover["cutover_block"])
            ):
                raise ValueError(
                    "local PostgREST chain activation fixture predates cutover"
                )
            current_settlement_epoch = int(
                cutover["first_settlement_epoch_id"]
            ) + (
                subnet_epoch_index - first_subnet_epoch
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
        self.cutover_state = list(self.rows.get(state_table, []))
        self.events = state_root / "local-postgrest-events.jsonl"

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
        self.server.state.record(
            status="ok",
            operation=operation,
            method=self.command,
            target=target,
            query=parsed.query,
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
    args = parser.parse_args()
    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    if fixture.get("sanitization", {}).get("contains_production_credentials"):
        raise RuntimeError("local PostgREST fixture contains credentials")
    tables, rpcs = _schema_contract(args.source_root)
    relation_columns, migration_rpcs = _migration_schema_contract(
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
    )
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
