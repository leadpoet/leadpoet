#!/usr/bin/env python3.11
"""Strict local PostgREST equivalent for the exact gateway launcher replay."""

from __future__ import annotations

import argparse
import ast
import csv
from datetime import date, datetime, timedelta, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import signal
import threading
import time
from typing import Any, Mapping
from urllib.parse import parse_qsl, urlparse

from leadpoet_canonical.allocation_settlement_frontier_v2 import (
    frontier_artifact_hashes_v2,
    validate_allocation_settlement_frontier_v2,
)
from leadpoet_canonical.allocation_settlement_frontier_bootstrap_v2 import (
    ALLOCATION_SETTLEMENT_FRONTIER_BOOTSTRAP_OPERATION,
    ALLOCATION_SETTLEMENT_FRONTIER_BOOTSTRAP_PURPOSE,
    frontier_bootstrap_artifact_hashes_v2,
    validate_allocation_settlement_frontier_bootstrap_v2,
)
from leadpoet_canonical.attested_v2 import sha256_json

try:
    from fixture_contract import (
        load_rehearsal_current_settlement_epoch_id,
        validate_rehearsal_finalized_authority_epochs,
    )
except ModuleNotFoundError as exc:
    if exc.name != "fixture_contract":
        raise
    from tests.restart_rehearsal.fixture_contract import (
        load_rehearsal_current_settlement_epoch_id,
        validate_rehearsal_finalized_authority_epochs,
    )

try:
    from postgres_v2_contract_probe import DisposablePostgres
except ModuleNotFoundError as exc:
    if exc.name != "postgres_v2_contract_probe":
        raise
    from tests.restart_rehearsal.postgres_v2_contract_probe import (
        DisposablePostgres,
    )


RUNTIME_TABLES = frozenset(
    {
        "epoch_audit_logs",
        "leads_private",
        "merkle_checkpoints",
        "published_weight_bundles",
        "qualification_private_icp_sets",
        "research_lab_champion_reward_current",
        "research_lab_gateway_control_current",
        "research_lab_stateful_subnet_epoch_cutover_state_v1",
        "research_lab_stateful_subnet_epoch_cutovers_v1",
        "lab_arena_reward_basis_v1",
        "transparency_log",
        "validation_evidence_private",
    }
)
RUNTIME_RPCS = frozenset(
    {
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
LAB_ARENA_RESTART_RPC_PARAMETERS = {
    "lab_arena_restart_guard_state_v1": (),
    "lab_arena_acquire_restart_guard_v1": (
        ("p_guard_id", "text"),
        ("p_owner_id", "text"),
        ("p_expected_generation", "bigint"),
        ("p_lease_seconds", "integer"),
        ("p_candidate_commit", "text"),
        ("p_restart_scope", "text"),
        ("p_actor_ref", "text"),
    ),
    "lab_arena_restart_quiescence_v1": (
        ("p_guard_id", "text"),
        ("p_owner_id", "text"),
        ("p_guard_generation", "bigint"),
    ),
    "lab_arena_retarget_restart_guard_v1": (
        ("p_current_guard_id", "text"),
        ("p_owner_id", "text"),
        ("p_expected_generation", "bigint"),
        ("p_new_guard_id", "text"),
        ("p_new_candidate_commit", "text"),
        ("p_restart_scope", "text"),
        ("p_lease_seconds", "integer"),
        ("p_actor_ref", "text"),
    ),
    "lab_arena_authorize_restart_phase_v1": (
        ("p_guard_id", "text"),
        ("p_owner_id", "text"),
        ("p_guard_generation", "bigint"),
        ("p_phase", "text"),
    ),
    "lab_arena_mark_restart_ready_v1": (
        ("p_guard_id", "text"),
        ("p_owner_id", "text"),
        ("p_guard_generation", "bigint"),
        ("p_phase", "text"),
    ),
    "lab_arena_abort_restart_guard_v1": (
        ("p_guard_id", "text"),
        ("p_owner_id", "text"),
        ("p_guard_generation", "bigint"),
        ("p_actor_ref", "text"),
    ),
    "lab_arena_release_restart_guard_v1": (
        ("p_guard_id", "text"),
        ("p_owner_id", "text"),
        ("p_guard_generation", "bigint"),
        ("p_actor_ref", "text"),
    ),
}
CONTROL_QUERY_FIELDS = frozenset(
    {"columns", "limit", "offset", "on_conflict", "order", "select"}
)


class MigrationBackedLabArenaRPC:
    """Expose migration-190 SQL functions through the local HTTP boundary."""

    def __init__(self, path: Path, *, candidate_sha: str):
        self.database: DisposablePostgres | None = None
        try:
            self.database = DisposablePostgres.attach(
                path,
                candidate_sha=candidate_sha,
            )
            function_names = ",".join(
                "'" + name + "'"
                for name in sorted(LAB_ARENA_RESTART_RPC_PARAMETERS)
            )
            observed = self.database.psql(
                f"""
                SELECT p.proname || '(' ||
                       pg_get_function_identity_arguments(p.oid) || ')'
                FROM pg_proc AS p
                JOIN pg_namespace AS n ON n.oid = p.pronamespace
                WHERE n.nspname = 'public'
                  AND p.proname IN ({function_names})
                ORDER BY p.proname;
                """,
                tuples_only=True,
            ).stdout.splitlines()
            expected = sorted(
                f"{name}({', '.join(f'{field} {kind}' for field, kind in parameters)})"
                for name, parameters in LAB_ARENA_RESTART_RPC_PARAMETERS.items()
            )
            if (
                sorted(line.strip() for line in observed if line.strip())
                != expected
            ):
                raise ValueError(
                    "migration-backed Lab Arena RPC catalog differs"
                )
        except BaseException:
            if self.database is not None:
                self.database.stop()
            raise

    def call(
        self,
        name: str,
        body: Any,
        *,
        database_role: str,
    ) -> dict[str, Any]:
        parameters = LAB_ARENA_RESTART_RPC_PARAMETERS.get(name)
        if parameters is None or not isinstance(body, Mapping):
            raise ValueError("Lab Arena restart RPC differs")
        if self.database is None:
            raise ValueError("migration-backed Lab Arena restart RPC is unavailable")
        if database_role not in {"anon", "lab_arena_service", "service_role"}:
            raise ValueError("Lab Arena restart database role differs")
        expected = {field for field, _ in parameters}
        if set(body) != expected:
            raise ValueError("Lab Arena restart RPC parameters differ")
        encoded = json.dumps(dict(body), sort_keys=True, separators=(",", ":"))
        if "$leadpoet$" in encoded:
            raise ValueError("Lab Arena restart RPC payload differs")
        arguments = ",".join(
            f"(payload->>'{field}')::{kind}" for field, kind in parameters
        )
        sql = (
            f"SET ROLE {database_role};\n"
            f"WITH input AS (SELECT $leadpoet${encoded}$leadpoet$::jsonb AS payload) "
            f"SELECT public.{name}({arguments})::text FROM input;\n"
        )
        result = self.database.psql(
            sql,
            tuples_only=True,
            quiet=True,
            check=False,
        )
        if result.returncode != 0:
            raise ValueError("migration-backed Lab Arena restart RPC rejected")
        try:
            value = json.loads(result.stdout.strip())
        except json.JSONDecodeError as exc:
            raise ValueError("migration-backed Lab Arena restart RPC differs") from exc
        if not isinstance(value, Mapping):
            raise ValueError("migration-backed Lab Arena restart RPC differs")
        return dict(value)

    def stop(self) -> None:
        if self.database is not None:
            self.database.stop()
















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
        "132-research-lab-champion-lifetime-credit.sql",
        "136-research-lab-ancestry-checkpoint-sidecars.sql",
        "137-research-lab-allocation-settlement-frontier.sql",
        "138-research-lab-ancestry-checkpoint-bootstrap-purpose.sql",
        "139-research-lab-allocation-frontier-bootstrap.sql",
        "140-research-lab-allocation-frontier-historical-source.sql",
        "141-research-lab-allocation-frontier-source-contract.sql",
        "142-research-lab-source-catalog-result-replay.sql",
        "143-research-lab-compact-ancestry-checkpoints.sql",
        "144-research-lab-provider-persistence-batches.sql",
        "145-research-lab-source-add-admission-control.sql",
        "147-research-lab-source-catalog-auth-metadata.sql",
        "148-research-lab-atomic-credit-resume.sql",
        "149-research-lab-compact-weight-settlement-authority.sql",
        "155-research-lab-ancestry-disclosure-root-fast-path.sql",
        "156-production-parity-readonly-role.sql",
        "169-research-lab-source-add-post-accept-leg1.sql",
        "170-research-lab-source-add-provider-origin-uniqueness.sql",
        "171-research-lab-source-add-duplicate-privacy.sql",
        "172-research-lab-source-add-claim-control.sql",
        "173-research-lab-source-add-leg1-release-policy.sql",
        "174-research-lab-source-add-restart-state-restore.sql",
        "175-research-lab-source-add-provenance-leg1.sql",
        "176-research-lab-source-add-provenance-origin-repair.sql",
        "177-research-lab-source-add-provenance-authority-acl.sql",
        "178-research-lab-source-add-miner-status.sql",
        "186-research-lab-source-add-provisioned-status.sql",
        "179-lab-arena-v1.sql",
        "180-lab-arena-daily-competition.sql",
        "181-lab-arena-source-submissions.sql",
        "182-lab-arena-source-execution.sql",
        "183-lab-arena-miner-reward-basis.sql",
        "184-lab-arena-scoring-failure-isolation.sql",
        "185-lab-arena-miner-credentials.sql",
        "187-lab-arena-promotion-threshold.sql",
        "188-lab-arena-baseline-promotion.sql",
        "189-lab-arena-round-network-scope.sql",
        "190-lab-arena-restart-claim-drain.sql",
        "193-lab-arena-upload-recovery.sql",
        "194-lab-arena-open-scorer-refresh.sql",
        "198-retire-research-lab-source-add-schema.sql",
    ]
    applied_migrations = document.get("applied_migrations")
    if (
        not isinstance(applied_migrations, list)
        or applied_migrations[-len(expected_final_migrations) :]
        != expected_final_migrations
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
        "research_lab_maintenance_lease",
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
        "research_lab_attested_ancestry_checkpoints_v2",
        "research_lab_attested_ancestry_activations_v2",
        "research_lab_allocation_settlement_frontiers_v2",
        "research_lab_allocation_settlement_frontier_activation_v2",
        "research_lab_compact_weight_authorities_v2",
        "lab_arena_rounds",
        "lab_arena_submissions",
        "lab_arena_runs",
        "lab_arena_ledger",
        "lab_arena_reward_basis_v1",
    }
    retired_relations = {
        name for name in relations
        if name.startswith("research_lab_source_add")
        or name == "research_lab_source_catalog"
    }
    if retired_relations:
        raise RuntimeError(
            "migration-backed SOURCE_ADD relations remain: %s"
            % ",".join(sorted(retired_relations))
        )
    if not required_relations <= set(relations):
        raise RuntimeError(
            "migration-backed settlement relations are incomplete: %s"
            % ",".join(sorted(required_relations - set(relations)))
        )
    required_rpcs = {
        "research_lab_acquire_maintenance_lease",
        "research_lab_attested_transport_purpose_contract_v2",
        "research_lab_attested_transport_terminal_contract_v2",
        "put_research_lab_provider_evidence_cache_v2",
        "research_lab_provider_persistence_batch_contract_v1",
        "persist_research_lab_chain_realized_lifetime_settlement_v2",
        "research_lab_champion_lifetime_credit_contract_v1",
        "persist_research_lab_ancestry_checkpoint_v2",
        "research_lab_ancestry_disclosure_lookup_contract_v1",
        "leadpoet_production_parity_reader_contract_v1",
        "persist_research_lab_allocation_settlement_frontier_v2",
        "persist_research_lab_allocation_frontier_bootstrap_v2",
        "research_lab_ancestry_checkpoint_bootstrap_contract_v2",
        "research_lab_allocation_frontier_bootstrap_contract_v2",
        "research_lab_compact_weight_settlement_contract_v1",
        "lab_arena_current_daily_icp_set",
        "lab_arena_register_submission",
        "lab_arena_update_submission",
        "lab_arena_claim_assignment",
        "lab_arena_activate_reward",
        "lab_arena_schema_version_v1",
    }
    retired_rpcs = {
        name for name in raw_rpcs
        if name.startswith("research_lab_source_add")
        or name.startswith("research_lab_source_catalog")
    }
    if retired_rpcs:
        raise RuntimeError(
            "migration-backed SOURCE_ADD RPCs remain: %s"
            % ",".join(sorted(retired_rpcs))
        )
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
    expected = {
        "research_lab_finalized_allocation_epochs_v2",
        "research_lab_emission_allocation_current",
        "research_lab_legacy_finalized_allocation_migrations_v2",
        "research_lab_attested_boot_identities_v2",
        "research_lab_attested_execution_receipts_v2",
        "research_lab_attested_receipt_edges_v2",
        "research_lab_attested_receipt_transport_v2",
        "research_lab_attested_transport_attempts_v2",
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
        fixed_count = {
            "research_lab_finalized_allocation_epochs_v2": 2,
            "research_lab_emission_allocation_current": 1,
            "research_lab_legacy_finalized_allocation_migrations_v2": 1,
        }.get(target)
        if (
            not isinstance(rows, list)
            or (fixed_count is not None and len(rows) != fixed_count)
            or (
                target
                in {
                    "research_lab_attested_boot_identities_v2",
                    "research_lab_attested_execution_receipts_v2",
                }
                and not rows
            )
            or any(not isinstance(row, dict) for row in rows)
        ):
            raise RuntimeError(
                "migration-backed allocation authority seed is invalid: %s"
                % target
            )
        expected_columns = relation_columns.get(target)
        if expected_columns is None or any(
            set(row) != set(expected_columns) for row in rows
        ):
            raise RuntimeError(
                "migration-backed allocation authority seed columns differ: %s"
                % target
            )
        normalized[target] = [dict(row) for row in rows]
    _validate_migration_receipt_graph_seeds(normalized)
    validate_rehearsal_finalized_authority_epochs(normalized)
    return normalized


def _validate_migration_receipt_graph_seeds(
    rows_by_table: dict[str, list[dict[str, Any]]],
) -> None:
    def unique_rows(table: str, field: str) -> dict[str, dict[str, Any]]:
        indexed: dict[str, dict[str, Any]] = {}
        for row in rows_by_table[table]:
            value = str(row.get(field) or "")
            if not value or value in indexed:
                raise RuntimeError(
                    "migration-backed allocation authority seed has an "
                    "invalid or duplicated %s: %s" % (field, table)
                )
            indexed[value] = row
        return indexed

    boots = unique_rows(
        "research_lab_attested_boot_identities_v2",
        "boot_identity_hash",
    )
    receipts = unique_rows(
        "research_lab_attested_execution_receipts_v2",
        "receipt_hash",
    )
    attempts = unique_rows(
        "research_lab_attested_transport_attempts_v2",
        "attempt_hash",
    )

    edges_by_child: dict[str, set[str]] = {}
    edge_pairs: set[tuple[str, str]] = set()
    for row in rows_by_table["research_lab_attested_receipt_edges_v2"]:
        pair = (
            str(row.get("child_receipt_hash") or ""),
            str(row.get("parent_receipt_hash") or ""),
        )
        if (
            not all(pair)
            or pair in edge_pairs
            or pair[0] not in receipts
            or pair[1] not in receipts
        ):
            raise RuntimeError(
                "migration-backed allocation authority receipt edge is invalid"
            )
        edge_pairs.add(pair)
        edges_by_child.setdefault(pair[0], set()).add(pair[1])

    receipt_scopes: dict[str, tuple[str, str]] = {}
    for receipt_hash, row in receipts.items():
        document = row.get("receipt_doc")
        if not isinstance(document, dict):
            raise RuntimeError(
                "migration-backed allocation authority receipt document is missing"
            )
        parent_hashes = document.get("parent_receipt_hashes")
        boot_hash = str(document.get("boot_identity_hash") or "")
        if (
            document.get("receipt_hash") != receipt_hash
            or row.get("boot_identity_hash") != boot_hash
            or boot_hash not in boots
            or not isinstance(parent_hashes, list)
            or any(not isinstance(value, str) for value in parent_hashes)
            or len(set(parent_hashes)) != len(parent_hashes)
            or edges_by_child.get(receipt_hash, set()) != set(parent_hashes)
        ):
            raise RuntimeError(
                "migration-backed allocation authority receipt graph is incomplete"
            )
        receipt_scopes[receipt_hash] = (
            str(document.get("job_id") or ""),
            str(document.get("purpose") or ""),
        )

    attempt_scopes: dict[str, tuple[str, str]] = {}
    for attempt_hash, row in attempts.items():
        document = row.get("attempt_doc")
        if (
            not isinstance(document, dict)
            or document.get("attempt_hash") != attempt_hash
        ):
            raise RuntimeError(
                "migration-backed allocation authority transport document is invalid"
            )
        attempt_scopes[attempt_hash] = (
            str(document.get("job_id") or ""),
            str(document.get("purpose") or ""),
        )

    link_pairs: set[tuple[str, str]] = set()
    for row in rows_by_table["research_lab_attested_receipt_transport_v2"]:
        pair = (
            str(row.get("receipt_hash") or ""),
            str(row.get("attempt_hash") or ""),
        )
        if (
            not all(pair)
            or pair in link_pairs
            or pair[0] not in receipts
            or pair[1] not in attempts
            or receipt_scopes[pair[0]] != attempt_scopes[pair[1]]
        ):
            raise RuntimeError(
                "migration-backed allocation authority receipt transport is invalid"
            )
        link_pairs.add(pair)


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
        durable_state_path: Path | None = None,
        durable_schema_sha: str = "",
    ):
        self.state_root = state_root
        self.source_root = source_root
        self.fixture = fixture
        self.tables = tables
        self.rpcs = rpcs
        self.relation_columns = dict(relation_columns or {})
        self.lock = threading.Lock()
        self.durable_state_path = durable_state_path
        self.durable_schema_sha = durable_schema_sha
        self.durable_revision = 0
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
            source_rows = [
                row
                for row in self.rows.get(
                    "research_lab_finalized_allocation_epochs_v2", []
                )
                if int(row.get("netuid", -1)) == int(cutover["netuid"])
                and int(row.get("epoch_id", -1)) == first_epoch
            ]
            if self.rows.get("research_lab_finalized_allocation_epochs_v2"):
                if len(source_rows) != 1:
                    raise ValueError(
                        "local PostgREST settlement activation source differs"
                    )
                source_bundle_hash = str(source_rows[0]["bundle_hash"])
                source_finalized_block = int(
                    source_rows[0]["finalized_block"]
                )
            else:
                source_bundle_hash = "sha256:" + "a" * 64
                source_finalized_block = current_block - 1
            self.rows[chain_activation_table] = [
                {
                    "netuid": int(cutover["netuid"]),
                    "schema_version": (
                        "leadpoet.research_lab_chain_realized_settlement_activation.v1"
                    ),
                    "first_epoch_id": first_epoch,
                    "source_bundle_hash": source_bundle_hash,
                    "source_bundle_epoch_id": first_epoch,
                    "source_finalized_block": source_finalized_block,
                }
            ]
        self._restore_durable_state()
        self.cutover_state = list(self.rows.get(state_table, []))
        self.events = state_root / "local-postgrest-events.jsonl"
        with self.lock:
            self._write_durable_state_locked(mutated=False)
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


    def acquire_maintenance_lease(
        self,
        body: Any,
        *,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """Mirror migration 118's atomic acquire-or-renew RPC."""

        if not isinstance(body, dict) or set(body) != {
            "p_lease_name",
            "p_holder_ref",
            "p_ttl_seconds",
        }:
            raise ValueError("maintenance lease RPC body is invalid")
        lease_name = body.get("p_lease_name")
        holder_ref = body.get("p_holder_ref")
        ttl_seconds = body.get("p_ttl_seconds")
        if (
            not isinstance(lease_name, str)
            or not isinstance(holder_ref, str)
            or not isinstance(ttl_seconds, int)
            or isinstance(ttl_seconds, bool)
            or ttl_seconds <= 0
            or ttl_seconds > 86400
        ):
            raise ValueError("maintenance lease arguments are invalid")
        table = "research_lab_maintenance_lease"
        expected_columns = {
            "lease_name",
            "holder_ref",
            "acquired_at",
            "expires_at",
            "updated_at",
        }
        relation_columns = self.relation_columns.get(table)
        if (
            table not in self.rows
            or relation_columns is None
            or set(relation_columns) != expected_columns
        ):
            raise ValueError("maintenance lease migration contract is unavailable")

        observed_now = now or datetime.now(timezone.utc)
        if observed_now.tzinfo is None:
            raise ValueError("maintenance lease clock must be timezone-aware")
        observed_now = observed_now.astimezone(timezone.utc)
        expires_at = observed_now + timedelta(seconds=ttl_seconds)
        encoded_now = observed_now.isoformat()
        encoded_expiry = expires_at.isoformat()

        with self.lock:
            row = next(
                (
                    candidate
                    for candidate in self.rows[table]
                    if candidate.get("lease_name") == lease_name
                ),
                None,
            )
            acquired = False
            mutated = False
            if row is None:
                row = {
                    "lease_name": lease_name,
                    "holder_ref": holder_ref,
                    "acquired_at": encoded_now,
                    "expires_at": encoded_expiry,
                    "updated_at": encoded_now,
                }
                self.rows[table].append(row)
                acquired = True
                mutated = True
            else:
                try:
                    current_expiry = datetime.fromisoformat(
                        str(row.get("expires_at") or "").replace("Z", "+00:00")
                    )
                except ValueError as exc:
                    raise ValueError(
                        "maintenance lease durable expiry is invalid"
                    ) from exc
                if current_expiry.tzinfo is None:
                    raise ValueError(
                        "maintenance lease durable expiry is not timezone-aware"
                    )
                same_holder = row.get("holder_ref") == holder_ref
                if current_expiry < observed_now or same_holder:
                    if not same_holder:
                        row["holder_ref"] = holder_ref
                        row["acquired_at"] = encoded_now
                    row["expires_at"] = encoded_expiry
                    row["updated_at"] = encoded_now
                    acquired = True
                    mutated = True
            self._write_durable_state_locked(mutated=mutated)
            return {
                "acquired": acquired,
                "holder_ref": row["holder_ref"],
                "expires_at": row["expires_at"],
            }



    def put_provider_evidence_cache(self, body: Any) -> dict[str, Any]:
        if not isinstance(body, dict) or set(body) != {"cache_row"}:
            raise ValueError("provider evidence cache put RPC body is invalid")
        row = body.get("cache_row")
        table = "research_lab_provider_evidence_cache_v2"
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
            != "leadpoet.provider_evidence_cache_row.v2"
            or not isinstance(row.get("encrypted_cache_doc"), dict)
            or not HASH_RE.fullmatch(
                str(row.get("artifact_master_key_ref_hash") or "")
            )
            or not re.fullmatch(
                r"[0-9a-f]{64}",
                str(row.get("request_fingerprint") or ""),
            )
            or any(
                not HASH_RE.fullmatch(str(row.get(field) or ""))
                for field in {
                    "cache_entry_hash",
                    "cache_artifact_id",
                    "source_record_hash",
                    "source_boot_identity_hash",
                    "response_body_hash",
                }
            )
        ):
            raise ValueError("provider evidence cache put fields are invalid")
        key = (
            str(row["artifact_master_key_ref_hash"]),
            str(row["utc_day"]),
            str(row["request_fingerprint"]),
        )
        with self.lock:
            existing = next(
                (
                    stored
                    for stored in self.rows[table]
                    if (
                        str(stored.get("artifact_master_key_ref_hash")),
                        str(stored.get("utc_day")),
                        str(stored.get("request_fingerprint")),
                    )
                    == key
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
                        "provider evidence cache identity identifies another row"
                    )
                status = "existing"
            else:
                stored = dict(row)
                stored["created_at"] = "2026-07-25T00:00:00+00:00"
                self.rows[table].append(stored)
                self._write_durable_state_locked(mutated=True)
                durable = dict(row)
                status = "inserted"
        return {
            "status": status,
            "cache_entry_hash": row["cache_entry_hash"],
            "cache_row": durable,
        }

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
            not in {
                "leadpoet.attested_checkpointed_receipt_graph.v3",
                "leadpoet.attested_checkpointed_receipt_graph.v4",
            }
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
            if (
                graph.get("schema_version")
                == "leadpoet.attested_checkpointed_receipt_graph.v4"
            ):
                projection = claim.get("local_delta_projection")
                disclosed_receipts = proof.get("disclosed_receipts")
                disclosed_boots = proof.get("disclosed_boot_identities")
                if (
                    not isinstance(projection, dict)
                    or not isinstance(disclosed_receipts, list)
                    or not isinstance(disclosed_boots, list)
                    or graph.get("transport_attempts") != []
                    or graph.get("host_operations") != []
                    or graph.get("receipts") != disclosed_receipts
                    or graph.get("boot_identities") != disclosed_boots
                ):
                    raise ValueError(
                        "compact checkpoint disclosure contract is invalid"
                    )
                disclosed_receipt_hashes = {
                    str(item.get("receipt_hash") or "")
                    for item in disclosed_receipts
                    if isinstance(item, dict)
                }
                disclosed_boot_hashes = {
                    str(item.get("boot_identity_hash") or "")
                    for item in disclosed_boots
                    if isinstance(item, dict)
                }
                durable_attempts = self.rows.get(
                    "research_lab_attested_receipt_transport_v2", []
                )
                durable_hosts = self.rows.get(
                    "research_lab_attested_host_operations_v2", []
                )
                observed_counts = {
                    "receipt_count": sum(
                        stored.get("receipt_hash")
                        in disclosed_receipt_hashes
                        for stored in receipt_rows
                    ),
                    "boot_identity_count": sum(
                        stored.get("boot_identity_hash")
                        in disclosed_boot_hashes
                        for stored in boot_rows
                    ),
                    "transport_attempt_count": sum(
                        stored.get("receipt_hash")
                        in disclosed_receipt_hashes
                        for stored in durable_attempts
                    ),
                    "host_operation_count": sum(
                        stored.get("receipt_hash")
                        in disclosed_receipt_hashes
                        for stored in durable_hosts
                    ),
                }
                if any(
                    projection.get(field) != observed
                    for field, observed in observed_counts.items()
                ):
                    raise ValueError(
                        "compact checkpoint raw sidecars are incomplete"
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

    def persist_allocation_settlement_frontier(
        self,
        body: Any,
    ) -> dict[str, Any]:
        """Mirror migration 137's exact append and activation contract."""

        if not isinstance(body, dict) or set(body) != {
            "requested_frontier",
            "requested_source_receipt_hash",
            "requested_source_state_hash",
        }:
            raise ValueError(
                "allocation settlement frontier RPC body is invalid"
            )
        frontier = validate_allocation_settlement_frontier_v2(
            body.get("requested_frontier")
        )
        source_receipt_hash = str(
            body.get("requested_source_receipt_hash") or ""
        )
        source_state_hash = str(
            body.get("requested_source_state_hash") or ""
        )
        if (
            not HASH_RE.fullmatch(source_receipt_hash)
            or not HASH_RE.fullmatch(source_state_hash)
        ):
            raise ValueError("allocation frontier source hash is invalid")

        frontier_table = (
            "research_lab_allocation_settlement_frontiers_v2"
        )
        activation_table = (
            "research_lab_allocation_settlement_frontier_activation_v2"
        )
        execution_table = "research_lab_attested_execution_results_v2"
        receipt_table = "research_lab_attested_execution_receipts_v2"
        frontier_columns = self.relation_columns.get(frontier_table)
        activation_columns = self.relation_columns.get(activation_table)
        if (
            frontier_columns is None
            or activation_columns is None
            or frontier_table not in self.rows
            or activation_table not in self.rows
        ):
            raise ValueError(
                "allocation settlement frontier migration is unavailable"
            )

        netuid = int(frontier["netuid"])
        epoch = int(frontier["allocation_epoch"])
        frontier_hash = str(frontier["frontier_hash"])
        with self.lock:
            execution = next(
                (
                    row
                    for row in self.rows.get(execution_table, [])
                    if row.get("receipt_hash") == source_receipt_hash
                ),
                None,
            )
            receipt = next(
                (
                    row
                    for row in self.rows.get(receipt_table, [])
                    if row.get("receipt_hash") == source_receipt_hash
                ),
                None,
            )
            result_doc = (
                execution.get("result_doc")
                if isinstance(execution, dict)
                else None
            )
            source_state = (
                result_doc.get("source_state")
                if isinstance(result_doc, dict)
                else None
            )
            artifacts = (
                execution.get("artifact_hashes")
                if isinstance(execution, dict)
                else None
            )
            required_artifacts = set(frontier_artifact_hashes_v2(frontier)) | {
                source_state_hash
            }
            execution_receipt_fields = (
                "role",
                "purpose",
                "job_id",
                "epoch_id",
                "sequence",
                "input_root",
                "output_root",
                "artifact_root",
            )
            if (
                not isinstance(execution, dict)
                or not isinstance(receipt, dict)
                or not isinstance(result_doc, dict)
                or not isinstance(source_state, dict)
                or not isinstance(artifacts, list)
                or execution.get("role") != "gateway_coordinator"
                or execution.get("operation") != "research_lab_allocation"
                or execution.get("purpose") != "research_lab.allocation.v2"
                or execution.get("epoch_id") != epoch
                or result_doc.get("source_state_hash") != source_state_hash
                or sha256_json(source_state) != source_state_hash
                or source_state.get("settlement_frontier") != frontier
                or source_state.get("epoch") != epoch
                or source_state.get("netuid") != netuid
                or receipt.get("receipt_status") != "succeeded"
                or any(
                    receipt.get(field) != execution.get(field)
                    for field in execution_receipt_fields
                )
                or not required_artifacts.issubset(set(artifacts))
            ):
                raise ValueError(
                    "allocation settlement frontier source is invalid"
                )

            frontiers = self.rows[frontier_table]
            activations = self.rows[activation_table]
            activation = next(
                (
                    row
                    for row in activations
                    if row.get("netuid") == netuid
                ),
                None,
            )
            if activation is not None:
                first = next(
                    (
                        row
                        for row in frontiers
                        if row.get("frontier_hash")
                        == activation.get("first_frontier_hash")
                    ),
                    None,
                )
                first_doc = (
                    first.get("frontier_doc")
                    if isinstance(first, dict)
                    else None
                )
                if (
                    not isinstance(first, dict)
                    or not isinstance(first_doc, dict)
                    or first.get("netuid") != netuid
                    or first.get("allocation_epoch")
                    != activation.get("first_allocation_epoch")
                    or first.get("frontier_hash")
                    != activation.get("first_frontier_hash")
                    or first.get("source_receipt_hash")
                    != activation.get("source_receipt_hash")
                    or first_doc.get("mode")
                    != "legacy_full_history_bootstrap"
                    or first.get("predecessor_frontier_hash") is not None
                ):
                    raise ValueError(
                        "allocation settlement frontier activation is invalid"
                    )
            existing = next(
                (
                    row
                    for row in frontiers
                    if row.get("netuid") == netuid
                    and row.get("allocation_epoch") == epoch
                ),
                None,
            )
            expected_row = {
                "netuid": netuid,
                "allocation_epoch": epoch,
                "settled_through_epoch": int(
                    frontier["settled_through_epoch"]
                ),
                "schema_version": str(frontier["schema_version"]),
                "frontier_hash": frontier_hash,
                "predecessor_frontier_hash": frontier.get(
                    "predecessor_frontier_hash"
                ),
                "source_receipt_hash": source_receipt_hash,
                "source_state_hash": source_state_hash,
                "frontier_doc": frontier,
            }
            if set(expected_row) != set(frontier_columns) - {"created_at"}:
                raise ValueError(
                    "allocation settlement frontier columns differ"
                )
            if existing is not None:
                if activation is None:
                    raise ValueError(
                        "allocation settlement frontier activation is invalid"
                    )
                durable = {
                    field: value
                    for field, value in existing.items()
                    if field != "created_at"
                }
                if durable != expected_row:
                    raise ValueError(
                        "allocation settlement frontier durable row conflicts"
                    )
                status = "already_persisted"
            else:
                lineage = [
                    row for row in frontiers if row.get("netuid") == netuid
                ]
                previous = (
                    max(lineage, key=lambda row: int(row["allocation_epoch"]))
                    if lineage
                    else None
                )
                if activation is None:
                    if (
                        previous is not None
                        or frontier.get("mode")
                        != "legacy_full_history_bootstrap"
                        or frontier.get("predecessor_frontier_hash") is not None
                    ):
                        raise ValueError(
                            "allocation settlement frontier bootstrap is invalid"
                        )
                elif (
                    previous is None
                    or frontier.get("mode") != "bounded_delta_v1"
                    or frontier.get("predecessor_frontier_hash")
                    != previous.get("frontier_hash")
                    or int(frontier["settled_through_epoch"])
                    <= int(previous["settled_through_epoch"])
                ):
                    raise ValueError(
                        "allocation settlement frontier successor is invalid"
                    )

                stored = {
                    **expected_row,
                    "created_at": "2026-07-25T00:00:00+00:00",
                }
                frontiers.append(stored)
                if activation is None:
                    activation = {
                        "netuid": netuid,
                        "schema_version": (
                            "leadpoet.research_lab_allocation_settlement_"
                            "frontier_activation.v2"
                        ),
                        "first_allocation_epoch": epoch,
                        "first_frontier_hash": frontier_hash,
                        "source_receipt_hash": source_receipt_hash,
                        "activated_at": "2026-07-25T00:00:00+00:00",
                    }
                    if set(activation) != set(activation_columns):
                        raise ValueError(
                            "allocation frontier activation columns differ"
                        )
                    activations.append(activation)
                self._write_durable_state_locked(mutated=True)
                status = "persisted"

        return {
            "status": status,
            "netuid": netuid,
            "allocation_epoch": epoch,
            "frontier_hash": frontier_hash,
            "source_receipt_hash": source_receipt_hash,
            "source_state_hash": source_state_hash,
        }

    def persist_allocation_settlement_frontier_bootstrap(
        self,
        body: Any,
    ) -> dict[str, Any]:
        """Mirror migration 139's measured first-frontier contract."""

        if not isinstance(body, dict) or set(body) != {
            "requested_frontier",
            "requested_source_receipt_hash",
            "requested_source_state_hash",
        }:
            raise ValueError("allocation frontier bootstrap RPC body is invalid")
        frontier = validate_allocation_settlement_frontier_v2(
            body.get("requested_frontier")
        )
        bootstrap_receipt_hash = str(
            body.get("requested_source_receipt_hash") or ""
        )
        source_state_hash = str(
            body.get("requested_source_state_hash") or ""
        )
        if (
            not HASH_RE.fullmatch(bootstrap_receipt_hash)
            or not HASH_RE.fullmatch(source_state_hash)
            or frontier.get("mode") != "legacy_full_history_bootstrap"
            or frontier.get("predecessor_frontier_hash") is not None
        ):
            raise ValueError("allocation frontier bootstrap request is invalid")

        frontier_table = "research_lab_allocation_settlement_frontiers_v2"
        activation_table = (
            "research_lab_allocation_settlement_frontier_activation_v2"
        )
        execution_table = "research_lab_attested_execution_results_v2"
        receipt_table = "research_lab_attested_execution_receipts_v2"
        frontier_columns = self.relation_columns.get(frontier_table)
        activation_columns = self.relation_columns.get(activation_table)
        if (
            frontier_columns is None
            or activation_columns is None
            or frontier_table not in self.rows
            or activation_table not in self.rows
        ):
            raise ValueError("allocation frontier bootstrap migration is unavailable")

        netuid = int(frontier["netuid"])
        epoch = int(frontier["allocation_epoch"])
        frontier_hash = str(frontier["frontier_hash"])
        with self.lock:
            bootstrap_execution = next(
                (
                    row
                    for row in self.rows.get(execution_table, [])
                    if row.get("receipt_hash") == bootstrap_receipt_hash
                ),
                None,
            )
            bootstrap_receipt = next(
                (
                    row
                    for row in self.rows.get(receipt_table, [])
                    if row.get("receipt_hash") == bootstrap_receipt_hash
                ),
                None,
            )
            bootstrap_doc = (
                bootstrap_execution.get("result_doc")
                if isinstance(bootstrap_execution, dict)
                else None
            )
            try:
                bootstrap = validate_allocation_settlement_frontier_bootstrap_v2(
                    bootstrap_doc
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "allocation frontier bootstrap authority is invalid"
                ) from exc
            allocation_receipt_hash = str(
                bootstrap.get("allocation_source_receipt_hash") or ""
            )
            allocation_execution = next(
                (
                    row
                    for row in self.rows.get(execution_table, [])
                    if row.get("receipt_hash") == allocation_receipt_hash
                ),
                None,
            )
            allocation_receipt = next(
                (
                    row
                    for row in self.rows.get(receipt_table, [])
                    if row.get("receipt_hash") == allocation_receipt_hash
                ),
                None,
            )
            allocation_result = (
                allocation_execution.get("result_doc")
                if isinstance(allocation_execution, dict)
                else None
            )
            allocation_state = (
                allocation_result.get("source_state")
                if isinstance(allocation_result, dict)
                else None
            )
            bootstrap_artifacts = (
                bootstrap_execution.get("artifact_hashes")
                if isinstance(bootstrap_execution, dict)
                else None
            )
            execution_receipt_fields = (
                "role",
                "purpose",
                "job_id",
                "epoch_id",
                "sequence",
                "input_root",
                "output_root",
                "artifact_root",
            )
            parent_hashes = (
                bootstrap_receipt.get("receipt_doc", {}).get(
                    "parent_receipt_hashes"
                )
                if isinstance(bootstrap_receipt, dict)
                and isinstance(bootstrap_receipt.get("receipt_doc"), dict)
                else None
            )
            if (
                not isinstance(bootstrap_execution, dict)
                or not isinstance(bootstrap_receipt, dict)
                or bootstrap_execution.get("role") != "gateway_coordinator"
                or bootstrap_execution.get("operation")
                != ALLOCATION_SETTLEMENT_FRONTIER_BOOTSTRAP_OPERATION
                or bootstrap_execution.get("purpose")
                != ALLOCATION_SETTLEMENT_FRONTIER_BOOTSTRAP_PURPOSE
                or bootstrap_receipt.get("receipt_status") != "succeeded"
                or any(
                    bootstrap_receipt.get(field)
                    != bootstrap_execution.get(field)
                    for field in execution_receipt_fields
                )
                or int(bootstrap_execution.get("epoch_id", -1))
                != int(bootstrap["bootstrap_epoch"])
                or bootstrap.get("frontier") != frontier
                or bootstrap.get("source_state_hash") != source_state_hash
                or not isinstance(bootstrap_artifacts, list)
                or not set(
                    frontier_bootstrap_artifact_hashes_v2(bootstrap)
                ).issubset(set(bootstrap_artifacts))
                or not isinstance(allocation_execution, dict)
                or not isinstance(allocation_receipt, dict)
                or not isinstance(allocation_result, dict)
                or not isinstance(allocation_state, dict)
                or allocation_execution.get("role") != "gateway_coordinator"
                or allocation_execution.get("operation")
                != "research_lab_allocation"
                or allocation_execution.get("purpose")
                != "research_lab.allocation.v2"
                or int(allocation_execution.get("epoch_id", -1)) != epoch
                or allocation_result.get("source_state_hash")
                != source_state_hash
                or allocation_state.get("epoch") != epoch
                or allocation_state.get("netuid") != netuid
                or allocation_state.get("settlement_frontier") is not None
                or allocation_receipt.get("receipt_status") != "succeeded"
                or any(
                    allocation_receipt.get(field)
                    != allocation_execution.get(field)
                    for field in execution_receipt_fields
                )
                or not isinstance(parent_hashes, list)
                or allocation_receipt_hash not in parent_hashes
            ):
                raise ValueError("allocation frontier bootstrap authority is invalid")

            frontiers = self.rows[frontier_table]
            activations = self.rows[activation_table]
            existing = next(
                (
                    row
                    for row in frontiers
                    if row.get("netuid") == netuid
                    and row.get("allocation_epoch") == epoch
                ),
                None,
            )
            activation = next(
                (row for row in activations if row.get("netuid") == netuid),
                None,
            )
            expected_row = {
                "netuid": netuid,
                "allocation_epoch": epoch,
                "settled_through_epoch": int(frontier["settled_through_epoch"]),
                "schema_version": str(frontier["schema_version"]),
                "frontier_hash": frontier_hash,
                "predecessor_frontier_hash": None,
                "source_receipt_hash": bootstrap_receipt_hash,
                "source_state_hash": source_state_hash,
                "frontier_doc": frontier,
            }
            expected_activation = {
                "netuid": netuid,
                "schema_version": (
                    "leadpoet.research_lab_allocation_settlement_"
                    "frontier_activation.v2"
                ),
                "first_allocation_epoch": epoch,
                "first_frontier_hash": frontier_hash,
                "source_receipt_hash": bootstrap_receipt_hash,
            }
            if existing is not None:
                durable_row = {
                    field: value
                    for field, value in existing.items()
                    if field != "created_at"
                }
                durable_activation = (
                    {
                        field: value
                        for field, value in activation.items()
                        if field != "activated_at"
                    }
                    if isinstance(activation, dict)
                    else None
                )
                if (
                    durable_row != expected_row
                    or durable_activation != expected_activation
                ):
                    raise ValueError("allocation frontier bootstrap conflicts")
                status = "already_persisted"
            else:
                if activation is not None or any(
                    row.get("netuid") == netuid for row in frontiers
                ):
                    raise ValueError("allocation frontier bootstrap is already initialized")
                if set(expected_row) != set(frontier_columns) - {"created_at"}:
                    raise ValueError("allocation frontier bootstrap columns differ")
                if set(expected_activation) != set(activation_columns) - {
                    "activated_at"
                }:
                    raise ValueError("allocation frontier activation columns differ")
                frontiers.append(
                    {
                        **expected_row,
                        "created_at": "2026-07-25T00:00:00+00:00",
                    }
                )
                activations.append(
                    {
                        **expected_activation,
                        "activated_at": "2026-07-25T00:00:00+00:00",
                    }
                )
                self._write_durable_state_locked(mutated=True)
                status = "persisted"

        return {
            "status": status,
            "netuid": netuid,
            "allocation_epoch": epoch,
            "frontier_hash": frontier_hash,
            "source_receipt_hash": bootstrap_receipt_hash,
            "source_state_hash": source_state_hash,
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

    def _database_role(self) -> str | None:
        apikey = self.headers.get("apikey", "")
        authorization = self.headers.get("authorization", "")
        if (apikey, authorization) == (
            "rehearsal-secret",
            "Bearer rehearsal.header.signature",
        ):
            return "lab_arena_service"
        if (apikey, authorization) == (
            "rehearsal-public",
            "Bearer rehearsal-public",
        ):
            return "anon"
        if (apikey, authorization) == (
            "rehearsal-secret",
            "Bearer rehearsal-secret",
        ):
            return "service_role"
        return None

    def _authorized(self) -> bool:
        return self._database_role() is not None

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
            if name in LAB_ARENA_RESTART_RPC_PARAMETERS:
                if self.server.lab_arena_rpc is None:
                    raise ValueError(
                        "migration-backed Lab Arena restart RPC is unavailable"
                    )
                database_role = self._database_role()
                if database_role is None:
                    raise ValueError("Lab Arena restart database role differs")
                response = self.server.lab_arena_rpc.call(
                    name,
                    body,
                    database_role=database_role,
                )
            elif name == (
                "research_lab_stateful_subnet_epoch_cutover_public_state_v1"
            ):
                response = self.server.state.cutover_state
            elif name == "research_lab_acquire_maintenance_lease":
                response = self.server.state.acquire_maintenance_lease(body)
                self.server.state.record(
                    status="ok",
                    operation="maintenance_lease_acquired",
                    method=self.command,
                    target=name,
                    lease_name=(
                        body.get("p_lease_name")
                        if isinstance(body, dict)
                        else None
                    ),
                    acquired=response["acquired"],
                    holder_ref=response["holder_ref"],
                    expires_at=response["expires_at"],
                )
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
            elif name == "put_research_lab_provider_evidence_cache_v2":
                response = self.server.state.put_provider_evidence_cache(body)
                self.server.state.record(
                    status="ok",
                    operation="provider_evidence_cache_put",
                    method=self.command,
                    target=name,
                    result_status=response["status"],
                    cache_entry_hash=response["cache_entry_hash"],
                )
            elif name == "research_lab_provider_persistence_batch_contract_v1":
                if body not in ({}, None):
                    raise ValueError(
                        "provider persistence batch contract body is invalid"
                    )
                response = {
                    "schema_version": (
                        "leadpoet.provider_persistence_batch_contract.v1"
                    ),
                    "cache_put": "atomic_exact_row",
                }
            elif name == (
                "research_lab_compact_weight_settlement_contract_v1"
            ):
                if body not in ({}, None):
                    raise ValueError(
                        "compact weight settlement contract body is invalid"
                    )
                response = {
                    "schema_version": (
                        "leadpoet.research_lab_compact_weight_settlement_contract.v1"
                    ),
                    "max_authority_bytes": 8_388_608,
                    "size_constraint_valid": True,
                    "append_only_trigger_enabled": True,
                    "identity_unique_constraint_enabled": True,
                    "row_level_security_enabled": True,
                    "finalized_stage_supported": True,
                }
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
            elif name == (
                "persist_research_lab_allocation_settlement_frontier_v2"
            ):
                response = (
                    self.server.state.persist_allocation_settlement_frontier(
                        body
                    )
                )
                self.server.state.record(
                    status="ok",
                    operation="allocation_settlement_frontier_persisted",
                    method=self.command,
                    target=name,
                    result_status=response["status"],
                    netuid=response["netuid"],
                    allocation_epoch=response["allocation_epoch"],
                    frontier_hash=response["frontier_hash"],
                    source_receipt_hash=response["source_receipt_hash"],
                )
            elif name == (
                "persist_research_lab_allocation_frontier_bootstrap_v2"
            ):
                response = self.server.state.persist_allocation_settlement_frontier_bootstrap(
                    body
                )
                self.server.state.record(
                    status="ok",
                    operation="allocation_settlement_frontier_bootstrap_persisted",
                    method=self.command,
                    target=name,
                    result_status=response["status"],
                    netuid=response["netuid"],
                    allocation_epoch=response["allocation_epoch"],
                    frontier_hash=response["frontier_hash"],
                    source_receipt_hash=response["source_receipt_hash"],
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
            expected_denial = (
                self.command == "POST"
                and urlparse(self.path).path
                == "/rest/v1/rpc/lab_arena_restart_guard_state_v1"
                and self._database_role() == "anon"
                and str(exc)
                == "migration-backed Lab Arena restart RPC rejected"
            )
            self.server.state.record(
                status="expected_denial" if expected_denial else "rejected",
                operation="authorization" if expected_denial else "request_validation",
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
    def __init__(
        self,
        address: tuple[str, int],
        state: LocalPostgRESTState,
        *,
        lab_arena_rpc: MigrationBackedLabArenaRPC | None = None,
    ):
        self.state = state
        self.lab_arena_rpc = lab_arena_rpc
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
    parser.add_argument("--postgres-connection", type=Path)
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
    lab_arena_rpc = None

    def stop_server(_signal: int, _frame: Any) -> None:
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, stop_server)
    signal.signal(signal.SIGINT, stop_server)
    try:
        if args.postgres_connection is not None:
            lab_arena_rpc = MigrationBackedLabArenaRPC(
                args.postgres_connection,
                candidate_sha=args.candidate_sha,
            )
            missing_lab_arena_rpcs = (
                set(LAB_ARENA_RESTART_RPC_PARAMETERS) - migration_rpcs
            )
            if missing_lab_arena_rpcs:
                raise RuntimeError(
                    "migration-backed Lab Arena restart RPCs are incomplete"
                )
        state = LocalPostgRESTState(
            state_root=args.state_root,
            fixture=fixture,
            source_root=args.source_root,
            tables=tables,
            rpcs=rpcs,
            relation_columns=relation_columns,
            seed_rows=seed_rows,
            durable_state_path=args.durable_state,
            durable_schema_sha=args.candidate_sha,
        )
        durable_identity = state.durable_state_identity()
        server = LocalPostgRESTServer(
            (args.host, args.port),
            state,
            lab_arena_rpc=lab_arena_rpc,
        )
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
        server.serve_forever(poll_interval=0.1)
    finally:
        if "server" in locals():
            server.server_close()
        if lab_arena_rpc is not None:
            lab_arena_rpc.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
