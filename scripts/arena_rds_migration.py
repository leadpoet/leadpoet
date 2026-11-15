#!/usr/bin/env python3
"""Apply exact merged Arena SQL through the dedicated, bounded RDS login."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import psycopg2

try:
    from scripts.gateway_iam_session import AWS_CLIENT_CONFIG, gateway_iam_session
except ModuleNotFoundError:
    from gateway_iam_session import AWS_CLIENT_CONFIG, gateway_iam_session

ACCOUNT = "493765492819"
REGION = "us-east-1"
REMOTE = "https://github.com/leadpoet/leadpoet.git"
INSTANCE = "leadpoet-subnet-arena"
HOST = "leadpoet-subnet-arena.cgfmio6maf8s.us-east-1.rds.amazonaws.com"
USER = "arena_migrator"
SECRET = "leadpoet/arena/migrations/pranav"
ROLE_QUERY = """
SELECT jsonb_build_object(
 'database',current_database(),'user',current_user,'session_user',session_user,
 'ssl',(SELECT ssl FROM pg_stat_ssl WHERE pid=pg_backend_pid()),
 'version',current_setting('server_version_num')::int,'recovery',pg_is_in_recovery(),
 'flags',jsonb_build_array(r.rolcanlogin,r.rolsuper,r.rolcreatedb,r.rolcreaterole,
                         r.rolreplication,r.rolbypassrls,r.rolinherit),
 'memberships',(SELECT coalesce(jsonb_agg(jsonb_build_array(g.rolname,
      m.admin_option,m.inherit_option,m.set_option) ORDER BY g.rolname),'[]'::jsonb)
   FROM pg_auth_members m JOIN pg_roles g ON g.oid=m.roleid WHERE m.member=r.oid),
 'schema_ready',has_schema_privilege(current_user,'public','USAGE') AND
                has_schema_privilege(current_user,'public','CREATE WITH GRANT OPTION'),
 'temporary_ready',has_database_privilege(current_user,current_database(),'TEMPORARY'),
 'external_definers',EXISTS(
   SELECT 1 FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace
   WHERE p.prosecdef AND n.nspname NOT IN ('pg_catalog','information_schema')
     AND p.proowner NOT IN (r.oid,'lab_arena_owner'::regrole)
     AND has_schema_privilege(current_user,n.oid,'USAGE')
     AND has_function_privilege(current_user,p.oid,'EXECUTE')),
 'container_owner',EXISTS(SELECT 1 FROM pg_database WHERE datdba=r.oid) OR
                   EXISTS(SELECT 1 FROM pg_namespace WHERE nspowner=r.oid))
FROM pg_roles r WHERE r.rolname=current_user
"""


class MigrationError(RuntimeError):
    """Safe messages only; never include provider or database exception text."""


def require(condition, message):
    if not condition:
        raise MigrationError(message)


def git(repo, *args):
    try:
        return subprocess.run(["git", "-C", str(repo), *args], check=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        raise MigrationError("repository verification failed") from None


def verified_sql(repo, migration, commit, expected_hash):
    repo, migration = Path(repo).resolve(), Path(migration).resolve()
    require(migration.parent == repo / "scripts" and
            re.fullmatch(r"[0-9]{2,4}-[a-z0-9][a-z0-9-]*\.sql", migration.name),
            "migration must be a numbered repo/scripts SQL file")
    require(re.fullmatch(r"[0-9a-f]{40}", commit or "") and
            re.fullmatch(r"[0-9a-f]{64}", expected_hash or ""), "invalid commit or hash")
    git(repo, "fetch", "--no-tags", REMOTE, "refs/heads/main")
    main = git(repo, "rev-parse", "FETCH_HEAD").decode().strip()
    git(repo, "merge-base", "--is-ancestor", commit, main)
    git(repo, "merge-base", "--is-ancestor", main, "HEAD")
    require(not git(repo, "status", "--porcelain", "--untracked-files=all"),
            "repository must be clean and include current main")
    relative = "scripts/" + migration.name
    sql = git(repo, "show", f"{commit}:{relative}")
    require(migration.read_bytes() == sql and hashlib.sha256(sql).hexdigest() == expected_hash,
            "migration bytes or hash differ")
    return sql


def connect(local_port, ca_file):
    require(isinstance(local_port, int) and 1 <= local_port <= 65535,
            "invalid local tunnel port")
    ca_file = Path(ca_file).expanduser().resolve()
    require(ca_file.is_file(), "CA file is unavailable")
    session = gateway_iam_session(profile="leadpoet-lab")
    db = session.client("rds", config=AWS_CLIENT_CONFIG).describe_db_instances(
        DBInstanceIdentifier=INSTANCE)["DBInstances"]
    require(len(db) == 1 and db[0].get("DBInstanceArn") ==
            f"arn:aws:rds:{REGION}:{ACCOUNT}:db:{INSTANCE}" and
            db[0].get("Endpoint", {}).get("Address") == HOST and
            db[0].get("Endpoint", {}).get("Port") == 5432 and
            db[0].get("DBName") == "postgres" and db[0].get("Engine") == "postgres" and
            db[0].get("DBInstanceStatus") == "available", "RDS target identity differs")
    value = session.client("secretsmanager", config=AWS_CLIENT_CONFIG).get_secret_value(SecretId=SECRET)
    require(value.get("Name") == SECRET and re.fullmatch(
        rf"arn:aws:secretsmanager:{REGION}:{ACCOUNT}:secret:{re.escape(SECRET)}-[A-Za-z0-9]{{6}}",
        value.get("ARN", "")), "migration secret identity differs")
    secret = json.loads(value["SecretString"])
    require(isinstance(secret, dict) and set(secret) == {"username", "password", "host", "port", "dbname"}
            and secret.get("username") == USER and secret.get("host") == HOST
            and type(secret.get("port")) is int and secret["port"] == 5432
            and secret.get("dbname") == "postgres" and isinstance(secret.get("password"), str)
            and bool(secret["password"]), "migration secret contract differs")
    connection = psycopg2.connect(host=HOST, hostaddr="127.0.0.1", port=local_port,
        dbname="postgres", user=USER, password=secret["password"], sslmode="verify-full",
        sslrootcert=str(ca_file), connect_timeout=15, application_name="arena_reviewed_migration")
    try:
        connection.autocommit = True
        with connection.cursor() as cursor:
            cursor.execute(ROLE_QUERY)
            state = cursor.fetchone()[0]
        require(state.get("database") == "postgres" and state.get("user") == USER
                and state.get("session_user") == USER and state.get("ssl") is True
                and 170000 <= state.get("version", 0) < 180000 and state.get("recovery") is False
                and state.get("flags") == [True, False, False, False, False, False, False]
                and state.get("memberships") == [["lab_arena_owner", False, True, True]]
                and state.get("schema_ready") is True and state.get("temporary_ready") is True
                and state.get("container_owner") is False
                and state.get("external_definers") is False, "database authority differs")
        return connection
    except BaseException:
        connection.close()
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-access", action="store_true")
    mode.add_argument("--verify-only", type=Path)
    mode.add_argument("--apply", type=Path)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--commit")
    parser.add_argument("--expected-sha256")
    parser.add_argument("--local-port", type=int)
    parser.add_argument("--ca-file", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.check_access:
            require(not args.commit and not args.expected_sha256, "check-access accepts no migration binding")
            sql = None
        else:
            sql = verified_sql(args.repo, args.verify_only or args.apply, args.commit, args.expected_sha256)
        if not args.verify_only:
            require(args.local_port is not None and args.ca_file is not None,
                    "local-port and ca-file are required")
            connection = connect(args.local_port, args.ca_file)
            try:
                if args.apply:
                    with connection.cursor() as cursor:
                        cursor.execute(sql.decode("utf-8"))
            finally:
                connection.close()
        print("ARENA_MIGRATION_APPLIED" if args.apply else "ARENA_MIGRATION_READY")
        return 0
    except MigrationError as exc:
        print(f"ARENA_MIGRATION_ERROR: {exc}", file=sys.stderr)
    except Exception as exc:
        message = ("migration requires a separately reviewed privileged channel"
                   if getattr(exc, "pgcode", None) == "42501" else "migration operation failed")
        print(f"ARENA_MIGRATION_ERROR: {message}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
