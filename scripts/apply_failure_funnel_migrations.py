#!/usr/bin/env python3
"""Verify or apply the fixed Research Lab failure-funnel migrations safely.

The production apply path intentionally accepts no SQL or migration path from
the caller.  It binds the two fixed files to the current ``origin/main``
commit and operator-supplied SHA-256 values, then uses one direct PostgreSQL
session in autocommit mode.  Migration 150's top-level statements are sent
separately so ``CREATE INDEX CONCURRENTLY`` never enters a transaction block;
migration 151 retains its committed explicit ``BEGIN``/``COMMIT`` transaction.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import parse_qsl, unquote, urlparse


CANONICAL_REPO = Path("/Users/pranav/Downloads/Election_Analysis/Bittensor-subnet")
PROJECT_REF = "qplwoislplkcegvdmbim"
MIGRATION_150 = "scripts/150-research-lab-failure-funnel-indexes.concurrent.sql"
MIGRATION_151 = "scripts/151-research-lab-failure-funnel-reporting.sql"
MIGRATION_PATHS = (MIGRATION_150, MIGRATION_151)
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
HASH_RE = re.compile(r"^[0-9a-f]{64}$")
DOLLAR_QUOTE_RE = re.compile(r"\$(?:[A-Za-z_][A-Za-z0-9_]*)?\$")
AUTHORIZED_REMOTE_URLS = {
    "https://github.com/leadpoet/leadpoet.git",
    "https://github.com/leadpoet/leadpoet",
    "git@github.com:leadpoet/leadpoet.git",
    "ssh://git@github.com/leadpoet/leadpoet.git",
}
ADVISORY_LOCK_NAME = "leadpoet.failure_funnel.migrations.150-151.v1"

INDEX_CONTRACT: Tuple[Tuple[str, str], ...] = (
    (
        "idx_research_eval_score_bundles_ticket_created",
        "research_evaluation_score_bundles",
    ),
    (
        "idx_research_lab_company_labels_ticket_candidate",
        "research_lab_company_label_examples",
    ),
    (
        "idx_research_lab_scoring_runs_ticket_candidate",
        "research_lab_scoring_runs",
    ),
)


class MigrationApplyError(RuntimeError):
    """A fail-closed source, connection, or migration contract failure."""


@dataclass(frozen=True)
class VerifiedMigration:
    path: str
    sha256: str
    sql: str


def _run_git(repo: Path, *args: str) -> bytes:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=repo,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
        ).stdout
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise MigrationApplyError("repository verification command failed") from exc


def _require_canonical_repo(repo: Path) -> None:
    try:
        resolved = repo.resolve(strict=True)
        canonical = CANONICAL_REPO.resolve(strict=True)
    except OSError as exc:
        raise MigrationApplyError("canonical repository is unavailable") from exc
    if resolved != canonical:
        raise MigrationApplyError(
            "production migration source must be the canonical repository"
        )


def _verify_repository_source(
    repo: Path,
    *,
    commit: str,
    expected_hashes: Sequence[str],
    enforce_canonical: bool = True,
    refresh_origin: bool = True,
) -> Tuple[VerifiedMigration, VerifiedMigration]:
    if enforce_canonical:
        _require_canonical_repo(repo)
    if not SHA_RE.fullmatch(commit):
        raise MigrationApplyError("commit must be a full lowercase Git SHA")
    if len(expected_hashes) != len(MIGRATION_PATHS) or any(
        not HASH_RE.fullmatch(value) for value in expected_hashes
    ):
        raise MigrationApplyError(
            "both migration SHA-256 values must be 64 lowercase hex characters"
        )

    repo = repo.resolve()
    top_level = Path(
        _run_git(repo, "rev-parse", "--show-toplevel").decode("utf-8").strip()
    ).resolve()
    if top_level != repo:
        raise MigrationApplyError("migration source is not the repository root")
    remote_url = _run_git(repo, "remote", "get-url", "origin").decode().strip()
    if remote_url not in AUTHORIZED_REMOTE_URLS:
        raise MigrationApplyError("repository origin is not the authorized remote")
    if refresh_origin:
        _run_git(repo, "fetch", "--quiet", "origin", "main")

    head = _run_git(repo, "rev-parse", "HEAD").decode().strip()
    origin_main = _run_git(repo, "rev-parse", "origin/main").decode().strip()
    if head != commit or origin_main != commit:
        raise MigrationApplyError(
            "canonical HEAD, current origin/main, and authorized commit differ"
        )
    if _run_git(repo, "status", "--porcelain=v1", "--untracked-files=no"):
        raise MigrationApplyError("canonical tracked repository is not clean")
    if _run_git(repo, "diff", "--name-only", "origin/main", "--"):
        raise MigrationApplyError("canonical repository differs from origin/main")
    if (repo / "AGENTS.md").read_bytes() != (repo / "CLAUDE.md").read_bytes():
        raise MigrationApplyError("AGENTS.md and CLAUDE.md differ")

    verified: List[VerifiedMigration] = []
    for relative, expected_hash in zip(MIGRATION_PATHS, expected_hashes):
        path = (repo / relative).resolve()
        if path.parent != (repo / "scripts").resolve() or path.name not in {
            Path(MIGRATION_150).name,
            Path(MIGRATION_151).name,
        }:
            raise MigrationApplyError("migration path escaped the fixed allowlist")
        committed = _run_git(repo, "show", "%s:%s" % (commit, relative))
        try:
            local = path.read_bytes()
        except OSError as exc:
            raise MigrationApplyError(
                "an authorized migration file is unavailable"
            ) from exc
        if local != committed:
            raise MigrationApplyError("local migration differs from its committed blob")
        observed_hash = hashlib.sha256(committed).hexdigest()
        if observed_hash != expected_hash:
            raise MigrationApplyError(
                "migration differs from its operator-bound SHA-256"
            )
        try:
            sql = committed.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise MigrationApplyError("migration is not valid UTF-8") from exc
        verified.append(VerifiedMigration(path=relative, sha256=observed_hash, sql=sql))
    return verified[0], verified[1]


def _split_sql_statements(sql: str) -> List[str]:
    """Split trusted, hash-bound PostgreSQL SQL at top-level semicolons."""

    statements: List[str] = []
    start = 0
    index = 0
    length = len(sql)
    state = "normal"
    dollar_tag = ""
    block_depth = 0
    while index < length:
        current = sql[index]
        following = sql[index + 1] if index + 1 < length else ""
        if state == "line_comment":
            if current == "\n":
                state = "normal"
            index += 1
            continue
        if state == "block_comment":
            if current == "/" and following == "*":
                block_depth += 1
                index += 2
                continue
            if current == "*" and following == "/":
                block_depth -= 1
                index += 2
                if block_depth == 0:
                    state = "normal"
                continue
            index += 1
            continue
        if state == "single_quote":
            if current == "'":
                if following == "'":
                    index += 2
                    continue
                state = "normal"
            index += 1
            continue
        if state == "double_quote":
            if current == '"':
                if following == '"':
                    index += 2
                    continue
                state = "normal"
            index += 1
            continue
        if state == "dollar_quote":
            if sql.startswith(dollar_tag, index):
                index += len(dollar_tag)
                state = "normal"
                continue
            index += 1
            continue

        if current == "-" and following == "-":
            state = "line_comment"
            index += 2
            continue
        if current == "/" and following == "*":
            state = "block_comment"
            block_depth = 1
            index += 2
            continue
        if current == "'":
            state = "single_quote"
            index += 1
            continue
        if current == '"':
            state = "double_quote"
            index += 1
            continue
        if current == "$":
            match = DOLLAR_QUOTE_RE.match(sql, index)
            if match:
                dollar_tag = match.group(0)
                state = "dollar_quote"
                index = match.end()
                continue
        if current == ";":
            statement = sql[start : index + 1].strip()
            if statement:
                statements.append(statement)
            start = index + 1
        index += 1

    if state in {"single_quote", "double_quote", "dollar_quote", "block_comment"}:
        raise MigrationApplyError("migration contains an unterminated SQL token")
    remainder = sql[start:].strip()
    if remainder:
        # A trailing comment is harmless; anything executable must be
        # semicolon-terminated so execution cannot depend on parser recovery.
        without_comments = re.sub(
            r"(?ms)^\s*(?:--[^\n]*(?:\n|$)|/\*.*?\*/\s*)*$", "", remainder
        )
        if without_comments.strip():
            raise MigrationApplyError("migration has an unterminated SQL statement")
    return statements


def _statement_command(statement: str) -> str:
    index = 0
    length = len(statement)
    while index < length:
        while index < length and statement[index].isspace():
            index += 1
        if statement.startswith("--", index):
            newline = statement.find("\n", index + 2)
            if newline < 0:
                return ""
            index = newline + 1
            continue
        if statement.startswith("/*", index):
            end = statement.find("*/", index + 2)
            if end < 0:
                raise MigrationApplyError("migration contains an unterminated comment")
            index = end + 2
            continue
        break
    match = re.match(r"[A-Za-z]+", statement[index:])
    return match.group(0).upper() if match else ""


def _validated_statement_sets(
    migration_150: VerifiedMigration,
    migration_151: VerifiedMigration,
) -> Tuple[List[str], List[str]]:
    statements_150 = _split_sql_statements(migration_150.sql)
    commands_150 = [_statement_command(value) for value in statements_150]
    if commands_150 != ["SET", "CREATE", "CREATE", "CREATE", "DO", "RESET"]:
        raise MigrationApplyError("migration 150 statement contract is invalid")
    normalized_150 = migration_150.sql.upper()
    if "BEGIN;" in normalized_150 or "COMMIT;" in normalized_150:
        raise MigrationApplyError("migration 150 must remain nontransactional")

    statements_151 = _split_sql_statements(migration_151.sql)
    commands_151 = [_statement_command(value) for value in statements_151]
    if not commands_151 or commands_151[0] != "BEGIN" or commands_151[-1] != "COMMIT":
        raise MigrationApplyError(
            "migration 151 must retain explicit transaction bounds"
        )
    if commands_151.count("BEGIN") != 1 or commands_151.count("COMMIT") != 1:
        raise MigrationApplyError("migration 151 transaction contract is ambiguous")
    return statements_150, statements_151


def _database_url_from_environment() -> str:
    dsn = os.environ.get("SUPABASE_DB_URL", "").strip()
    if not dsn:
        raise MigrationApplyError("SUPABASE_DB_URL is unavailable")
    try:
        parsed = urlparse(dsn)
        port = parsed.port or 5432
    except ValueError as exc:
        raise MigrationApplyError("SUPABASE_DB_URL is malformed") from exc
    hostname = (parsed.hostname or "").lower()
    username = unquote(parsed.username or "")
    database = unquote(parsed.path.lstrip("/"))
    if parsed.scheme not in {"postgres", "postgresql"}:
        raise MigrationApplyError("SUPABASE_DB_URL must use PostgreSQL")
    if port != 5432:
        raise MigrationApplyError(
            "SUPABASE_DB_URL must use direct or session-mode port 5432"
        )
    if not parsed.password:
        raise MigrationApplyError("SUPABASE_DB_URL has no database credential")
    try:
        query_items = (
            parse_qsl(parsed.query, keep_blank_values=True, strict_parsing=True)
            if parsed.query
            else []
        )
    except ValueError as exc:
        raise MigrationApplyError("SUPABASE_DB_URL options are malformed") from exc
    if len(query_items) != len({key for key, _value in query_items}):
        raise MigrationApplyError("SUPABASE_DB_URL has duplicate options")
    if any(key != "sslmode" or value != "require" for key, value in query_items):
        raise MigrationApplyError("SUPABASE_DB_URL contains unauthorized options")
    if parsed.fragment:
        raise MigrationApplyError("SUPABASE_DB_URL contains a fragment")
    direct = hostname == "db.%s.supabase.co" % PROJECT_REF and username == "postgres"
    session_pooler = (
        hostname.endswith(".pooler.supabase.com")
        and username == "postgres.%s" % PROJECT_REF
    )
    if not (direct or session_pooler):
        raise MigrationApplyError(
            "SUPABASE_DB_URL is not bound to the authorized Supabase project"
        )
    if database != "postgres":
        raise MigrationApplyError("SUPABASE_DB_URL must select the postgres database")
    return dsn


def _connect_database(dsn: str) -> Any:
    try:
        import psycopg2  # type: ignore
    except ImportError as exc:
        raise MigrationApplyError(
            "direct PostgreSQL driver is unavailable; no fallback is permitted"
        ) from exc
    try:
        connection = psycopg2.connect(
            dsn,
            connect_timeout=15,
            sslmode="require",
            application_name="leadpoet_failure_funnel_migration_v1",
            options=(
                "-c statement_timeout=1800000 "
                "-c idle_in_transaction_session_timeout=300000"
            ),
        )
        connection.autocommit = True
        return connection
    except Exception as exc:
        # Never echo the driver exception: depending on driver/version it may
        # contain connection material.
        raise MigrationApplyError("direct PostgreSQL connection failed") from exc


def _query_one(
    connection: Any, statement: str, parameters: Sequence[Any] = ()
) -> Tuple[Any, ...]:
    try:
        with connection.cursor() as cursor:
            cursor.execute(statement, tuple(parameters))
            row = cursor.fetchone()
    except Exception as exc:
        raise MigrationApplyError("database contract query failed") from exc
    if row is None:
        raise MigrationApplyError("database contract query returned no row")
    return tuple(row)


def _require_database_authority(connection: Any) -> None:
    if getattr(connection, "autocommit", None) is not True:
        raise MigrationApplyError("database session is not in autocommit mode")
    row = _query_one(
        connection,
        """
        SELECT current_user,
               current_database(),
               current_setting('transaction_read_only'),
               current_setting('server_version_num')::INTEGER
        """,
    )
    if row[0] != "postgres" or row[1] != "postgres" or row[2] != "off":
        raise MigrationApplyError(
            "database session lacks the required postgres write authority"
        )
    if int(row[3]) < 120000:
        raise MigrationApplyError("PostgreSQL server version is unsupported")


def _try_advisory_lock(connection: Any) -> None:
    locked = _query_one(
        connection,
        "SELECT pg_catalog.pg_try_advisory_lock(pg_catalog.hashtextextended(%s, 0))",
        (ADVISORY_LOCK_NAME,),
    )[0]
    if locked is not True:
        raise MigrationApplyError("another failure-funnel migration apply is active")


def _release_advisory_lock(connection: Any) -> None:
    try:
        _query_one(
            connection,
            "SELECT pg_catalog.pg_advisory_unlock(pg_catalog.hashtextextended(%s, 0))",
            (ADVISORY_LOCK_NAME,),
        )
    except Exception:
        # Closing the session releases the lock.  Never hide an earlier apply
        # error with a cleanup-only diagnostic.
        pass


def _active_target_builds(connection: Any) -> List[str]:
    index_names = [item[0] for item in INDEX_CONTRACT]
    table_names = [item[1] for item in INDEX_CONTRACT]
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
            SELECT DISTINCT pg_catalog.format(
                       '%%I.%%I[%%s]',
                       table_namespace.nspname,
                       table_relation.relname,
                       COALESCE(index_relation.relname, 'catalog-pending')
                   ) AS build_label
            FROM pg_catalog.pg_stat_progress_create_index AS progress
            JOIN pg_catalog.pg_class AS table_relation
              ON table_relation.oid = progress.relid
            JOIN pg_catalog.pg_namespace AS table_namespace
              ON table_namespace.oid = table_relation.relnamespace
            LEFT JOIN pg_catalog.pg_class AS index_relation
              ON index_relation.oid = progress.index_relid
            LEFT JOIN pg_catalog.pg_namespace AS index_namespace
              ON index_namespace.oid = index_relation.relnamespace
            WHERE progress.datname = pg_catalog.current_database()
              AND (
                    (table_namespace.nspname = 'public'
                     AND table_relation.relname = ANY(%s))
                 OR (index_namespace.nspname = 'public'
                     AND index_relation.relname = ANY(%s))
              )
            ORDER BY build_label
                """,
                (table_names, index_names),
            )
            return [str(row[0]) for row in cursor.fetchall()]
    except Exception as exc:
        raise MigrationApplyError("active index-build preflight failed") from exc


def _index_states(connection: Any) -> List[Tuple[str, str, str]]:
    """Return ``(index, table, state)`` under migration 150's own contract."""

    values = ",".join(["(%s, %s)"] * len(INDEX_CONTRACT))
    parameters: List[str] = []
    for index_name, table_name in INDEX_CONTRACT:
        parameters.extend((index_name, table_name))
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
            WITH expected(index_name, table_name) AS (VALUES %s)
            SELECT expected.index_name,
                   expected.table_name,
                   index_relation.oid IS NOT NULL AS index_exists,
                   COALESCE(
                       index_namespace.nspname = 'public'
                       AND table_namespace.nspname = 'public'
                       AND table_relation.relname = expected.table_name
                       AND index_relation.relkind = 'i'
                       AND index_meta.indisvalid
                       AND index_meta.indisready
                       AND index_meta.indislive,
                       FALSE
                   ) AS contract_valid
            FROM expected
            LEFT JOIN pg_catalog.pg_namespace AS index_namespace
              ON index_namespace.nspname = 'public'
            LEFT JOIN pg_catalog.pg_class AS index_relation
              ON index_relation.relnamespace = index_namespace.oid
             AND index_relation.relname = expected.index_name
            LEFT JOIN pg_catalog.pg_index AS index_meta
              ON index_meta.indexrelid = index_relation.oid
            LEFT JOIN pg_catalog.pg_class AS table_relation
              ON table_relation.oid = index_meta.indrelid
            LEFT JOIN pg_catalog.pg_namespace AS table_namespace
              ON table_namespace.oid = table_relation.relnamespace
            ORDER BY expected.index_name
            """
                % values,
                tuple(parameters),
            )
            rows = cursor.fetchall()
    except Exception as exc:
        raise MigrationApplyError("index catalog verification failed") from exc
    states: List[Tuple[str, str, str]] = []
    for index_name, table_name, exists, valid in rows:
        state = "ready" if valid else ("invalid" if exists else "missing")
        states.append((str(index_name), str(table_name), state))
    if len(states) != len(INDEX_CONTRACT):
        raise MigrationApplyError("index contract query returned incomplete state")
    return states


def _require_no_active_builds(connection: Any) -> None:
    builds = _active_target_builds(connection)
    if builds:
        raise MigrationApplyError(
            "failure-funnel prerequisite index build is active; wait and retry"
        )


def _preflight_indexes(connection: Any) -> List[Tuple[str, str, str]]:
    _require_no_active_builds(connection)
    states = _index_states(connection)
    invalid = [
        index_name for index_name, _table_name, state in states if state == "invalid"
    ]
    if invalid:
        raise MigrationApplyError(
            "failure-funnel prerequisite index is present but violates "
            "migration 150's exact catalog contract: %s" % ", ".join(sorted(invalid))
        )
    return states


def _postflight_indexes(connection: Any) -> None:
    _require_no_active_builds(connection)
    states = _index_states(connection)
    not_ready = [
        index_name for index_name, _table_name, state in states if state != "ready"
    ]
    if not_ready:
        raise MigrationApplyError(
            "failure-funnel prerequisite indexes are not all valid, ready, and live: %s"
            % ", ".join(sorted(not_ready))
        )


def _execute_statements(
    connection: Any,
    statements: Iterable[str],
    *,
    migration_label: str,
) -> None:
    for ordinal, statement in enumerate(statements, start=1):
        try:
            with connection.cursor() as cursor:
                cursor.execute(statement)
        except Exception as exc:
            try:
                # In psycopg autocommit mode, ``connection.rollback()`` does
                # not reliably clear a transaction opened by an explicit
                # SQL ``BEGIN``.  Send the matching SQL rollback on the same
                # persistent session so migration 151 cannot leave it aborted.
                with connection.cursor() as cursor:
                    cursor.execute("ROLLBACK")
            except Exception:
                try:
                    connection.rollback()
                except Exception:
                    pass
            raise MigrationApplyError(
                "%s failed at committed statement %d" % (migration_label, ordinal)
            ) from exc


def _require_reporting_function(connection: Any) -> None:
    row = _query_one(
        connection,
        """
        SELECT pg_catalog.to_regprocedure(
                   'public.get_research_lab_failure_funnel(uuid,text)'
               ) IS NOT NULL,
               pg_catalog.has_function_privilege(
                   'service_role',
                   'public.get_research_lab_failure_funnel(uuid,text)',
                   'EXECUTE'
               ),
               pg_catalog.has_function_privilege(
                   'anon',
                   'public.get_research_lab_failure_funnel(uuid,text)',
                   'EXECUTE'
               ),
               pg_catalog.has_function_privilege(
                   'authenticated',
                   'public.get_research_lab_failure_funnel(uuid,text)',
                   'EXECUTE'
               )
        """,
    )
    if row != (True, True, False, False):
        raise MigrationApplyError("migration 151 reporting-function ACL is invalid")


def _apply_verified_migrations(
    connection: Any,
    migration_150: VerifiedMigration,
    migration_151: VerifiedMigration,
) -> List[Tuple[str, str, str]]:
    statements_150, statements_151 = _validated_statement_sets(
        migration_150, migration_151
    )
    _require_database_authority(connection)
    _try_advisory_lock(connection)
    try:
        initial_states = _preflight_indexes(connection)
        _execute_statements(
            connection,
            statements_150,
            migration_label="migration 150",
        )
        _postflight_indexes(connection)
        _execute_statements(
            connection,
            statements_151,
            migration_label="migration 151",
        )
        _postflight_indexes(connection)
        _require_reporting_function(connection)
        return initial_states
    finally:
        _release_advisory_lock(connection)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--verify-only", action="store_true")
    mode.add_argument("--apply", action="store_true")
    parser.add_argument("--commit", required=True)
    parser.add_argument("--migration-150-sha256", required=True)
    parser.add_argument("--migration-151-sha256", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    connection: Optional[Any] = None
    try:
        migration_150, migration_151 = _verify_repository_source(
            CANONICAL_REPO,
            commit=args.commit,
            expected_hashes=(
                args.migration_150_sha256,
                args.migration_151_sha256,
            ),
        )
        _validated_statement_sets(migration_150, migration_151)
        if args.verify_only:
            print("FAILURE_FUNNEL_MIGRATIONS_VERIFIED")
            print("commit=%s" % args.commit)
            print("migration_150=%s" % MIGRATION_150)
            print("migration_150_sha256=%s" % migration_150.sha256)
            print("migration_151=%s" % MIGRATION_151)
            print("migration_151_sha256=%s" % migration_151.sha256)
            return 0
        if os.environ.get("LEADPOET_OVERNIGHT_REBENCHMARK_AUTHORIZED") != "1":
            raise MigrationApplyError(
                "explicit overnight rebenchmark authorization is absent"
            )
        dsn = _database_url_from_environment()
        connection = _connect_database(dsn)
        initial_states = _apply_verified_migrations(
            connection, migration_150, migration_151
        )
        print("FAILURE_FUNNEL_MIGRATIONS_APPLY_SUCCESS")
        print("commit=%s" % args.commit)
        print("migration_150_sha256=%s" % migration_150.sha256)
        print("migration_151_sha256=%s" % migration_151.sha256)
        print(
            "preflight_indexes=%s"
            % ",".join(
                "%s:%s" % (index_name, state)
                for index_name, _table, state in initial_states
            )
        )
        print("postflight_indexes=valid_ready_live")
        print("reporting_function=service_role_only")
        return 0
    except (MigrationApplyError, OSError) as exc:
        print("FAILURE_FUNNEL_MIGRATIONS_ERROR: %s" % exc, file=sys.stderr)
        return 1
    finally:
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
