#!/usr/bin/env python3
"""Prepare, then explicitly execute one dedicated empty-target bootstrap.

No source connection, extension installation or data restore exists here.
Default mode reads target catalogs and saves private SQL/evidence. Execute
requires the exact prepared SQL SHA and reviewed three-file code SHA. Use the
completed export manifest and its SHA. --print-code-sha256 does local I/O only.

Run under root systemd with RuntimeMaxSec=14400, TimeoutStopSec=30,
KillMode=control-group and UMask=0077. Send stdout/stderr to precreated private
0600 files in a root-only 0700 directory; never return raw command output to
SSM. A connection failure during COMMIT requires a read-only state audit, not
an automatic retry. cleanup.sql is a later reviewed privilege cleanup only.
"""

import argparse
import contextlib
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import sys

if __package__:
    from . import aws_subnet_restore as restore
    from . import aws_subnet_snapshot as snapshot
else:
    import aws_subnet_restore as restore
    import aws_subnet_snapshot as snapshot


TARGET_SECRET_ARN = "arn:aws:secretsmanager:us-east-1:493765492819:secret:rds!db-18d7b717-ab16-4254-8c1a-3ddbccf19ea5-KeHJ6K"
ROLES = tuple(n for n in restore.ROLES if n != "pg_database_owner")
OWNERS = ("postgres", "lab_arena_owner", "supabase_admin", "supabase_auth_admin")
ROLE_LITERALS = ",".join("'" + n + "'" for n in ROLES)
WRITE_SETTINGS = snapshot.TRANSACTION_SETTINGS.replace("default_transaction_read_only = on", "default_transaction_read_only = off") + "SET LOCAL standard_conforming_strings=on;\n"
NS = "n.nspname !~ '^pg_' AND n.nspname<>'information_schema'"


STATE_SQL = """SELECT json_build_object(
 'role',current_user,'session',session_user,'database',current_database(),
 'host',inet_server_addr()::text,'version',current_setting('server_version_num')::int,
 'encoding',pg_encoding_to_char(d.encoding),
 'read_only',current_setting('transaction_read_only'),'recovery',pg_is_in_recovery(),
 'owner',pg_get_userbyid(d.datdba),
 'master',(SELECT json_build_object('super',rolsuper,'createdb',rolcreatedb,
  'createrole',rolcreaterole,'login',rolcanlogin) FROM pg_roles WHERE rolname=current_user),
 'other_sessions',(SELECT count(*) FROM pg_stat_activity WHERE datname=current_database() AND pid<>pg_backend_pid()),
 'locale',json_build_object('provider',d.datlocprovider,'collate',d.datcollate,'ctype',d.datctype,
  'recorded_version',d.datcollversion,'actual_version',pg_database_collation_actual_version(d.oid)),
 'database_acl',DB_ACL,
 'public',(SELECT json_build_object('owner',pg_get_userbyid(n.nspowner),'acl',SCHEMA_ACL)
  FROM pg_namespace n WHERE n.nspname='public'),
 'bootstrap_schemas',(SELECT coalesce(json_agg(json_build_object('name',n.nspname,
  'owner',pg_get_userbyid(n.nspowner),'acl',BOOTSTRAP_ACL) ORDER BY n.nspname),'[]'::json)
  FROM pg_namespace n WHERE n.nspname IN ('auth','extensions')),
 'roles',(SELECT coalesce(json_agg(json_build_object('name',r.rolname,'inherit',r.rolinherit,
  'super',r.rolsuper,'createdb',r.rolcreatedb,'createrole',r.rolcreaterole,'login',r.rolcanlogin,
  'replication',r.rolreplication,'bypassrls',r.rolbypassrls,'config',r.rolconfig,
  'can_set',pg_has_role(current_user,r.oid,'SET'),'can_inherit',pg_has_role(current_user,r.oid,'USAGE'))
  ORDER BY r.rolname),'[]'::json) FROM pg_roles r WHERE r.rolname IN (ROLE_NAMES)),
 'memberships',(SELECT coalesce(json_agg(json_build_object('name',pg_get_userbyid(m.roleid),
  'member',pg_get_userbyid(m.member),'grantor',pg_get_userbyid(m.grantor),'admin',m.admin_option,
  'inherit',m.inherit_option,'set',m.set_option) ORDER BY m.roleid,m.member,m.grantor),'[]'::json)
  FROM pg_auth_members m WHERE pg_get_userbyid(m.roleid) IN (ROLE_NAMES) OR pg_get_userbyid(m.member) IN (ROLE_NAMES)),
 'counts',json_build_object(
  'relations',(SELECT count(*) FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace WHERE USER_NS AND NOT EXISTS(SELECT 1 FROM pg_depend p WHERE p.classid='pg_class'::regclass AND p.objid=c.oid AND p.deptype='e')),
  'routines',(SELECT count(*) FROM pg_proc c JOIN pg_namespace n ON n.oid=c.pronamespace WHERE USER_NS AND NOT EXISTS(SELECT 1 FROM pg_depend p WHERE p.classid='pg_proc'::regclass AND p.objid=c.oid AND p.deptype='e')),
  'types',(SELECT count(*) FROM pg_type c JOIN pg_namespace n ON n.oid=c.typnamespace WHERE USER_NS AND c.typtype IN ('e','d') AND NOT EXISTS(SELECT 1 FROM pg_depend p WHERE p.classid='pg_type'::regclass AND p.objid=c.oid AND p.deptype='e')),
  'schemas',(SELECT count(*) FROM pg_namespace n WHERE USER_NS AND n.nspname<>'public' AND NOT EXISTS(SELECT 1 FROM pg_depend p WHERE p.classid='pg_namespace'::regclass AND p.objid=n.oid AND p.deptype='e')),
  'default_acls',(SELECT count(*) FROM pg_default_acl),
  'event_triggers',(SELECT count(*) FROM pg_event_trigger),
  'large_objects',(SELECT count(*) FROM pg_largeobject_metadata),
  'databases',(SELECT count(*) FROM pg_database WHERE NOT datistemplate AND datname NOT IN ('postgres','rdsadmin')))
) FROM pg_database d WHERE d.datname=current_database();""".replace("ROLE_NAMES", ROLE_LITERALS).replace("USER_NS", NS).replace("DB_ACL", snapshot.acl("d.datacl")).replace("SCHEMA_ACL", snapshot.acl("n.nspacl")).replace("BOOTSTRAP_ACL", snapshot.acl("n.nspacl").replace("ORDER BY x::text", 'ORDER BY x::text COLLATE "C"'))


def literal(value):
    return "'" + value.replace("'", "''") + "'"


def json_literal(value):
    return literal(json.dumps(value, sort_keys=True, separators=(",", ":"))) + "::jsonb"


def code_sha256():
    digest = hashlib.sha256()
    for path in sorted((Path(__file__), Path(snapshot.__file__), Path(restore.__file__)), key=lambda p: p.name):
        digest.update((path.name + "\0" + snapshot.sha256_file(path) + "\n").encode())
    return digest.hexdigest()


def load_manifest(path, expected_sha):
    snapshot.require(snapshot.sha256_file(path) == expected_sha
                     and (path.parent / "manifest.sha256").read_text().strip() == expected_sha, "manifest-seal")
    value = json.loads(path.read_text())
    snapshot.require(value["format"] == 1 and value["source_host"] == snapshot.SOURCE_HOST
                     and value["source_role"] == snapshot.SOURCE_ROLE and value["database"] == "postgres", "manifest-identity")
    snapshot.validate_tables(value["tables"])
    support = restore.validate_supporting_auth_role(value["supporting_auth_role"])
    snapshot.require(support["schema"]["acl"] == snapshot.AUTH_SCHEMA_ACL
                     and support["function"]["acl"] == snapshot.AUTH_ROLE_ACL, "supporting-auth-acl")
    public = restore.source_public_schema(value["schema_metadata"])
    snapshot.require(public["owner"] == "pg_database_owner", "public-owner-review")
    restore.source_extensions_schema(value["schema_metadata"])
    return value


def read_state(env, logs):
    with snapshot.Keeper(env, logs) as keeper:
        return keeper.one_json(STATE_SQL)


def guard_empty(state):
    snapshot.require(state["role"] == state["session"] == state["owner"] == "arena_admin"
                     and state["database"] == "postgres" and 170000 <= state["version"] < 180000
                     and state["encoding"] == "UTF8" and state["read_only"] == "on"
                     and not state["recovery"], "target-identity")
    snapshot.require(state["master"] == dict(super=False, createdb=True, createrole=True, login=True), "target-master")
    snapshot.require(state["other_sessions"] == 0 and not state["roles"] and not state["memberships"]
                     and all(v == 0 for v in state["counts"].values()), "target-not-empty")
    snapshot.require(state["public"] and state["public"]["owner"] == "pg_database_owner", "target-public-owner")


def role_expectations():
    snapshot.require(len(ROLES) == 11 and set(OWNERS) == restore.OWNER_ROLES, "bootstrap-role-scope")
    return [dict(name=n, inherit=n not in restore.NOINHERIT, super=False, createdb=False, createrole=False,
                 login=False, replication=False, bypassrls=False, config=None,
                 can_set=n in OWNERS, can_inherit=n in OWNERS) for n in sorted(ROLES)]


def build_sql(before, manifest):
    guard_empty(before)
    restore.source_extensions_schema(manifest["schema_metadata"])
    support = copy.deepcopy(restore.validate_supporting_auth_role(manifest["supporting_auth_role"]))
    support["function"].pop("sha256")
    state_query = STATE_SQL.rstrip().rstrip(";")
    getter_query = snapshot.SUPPORTING_AUTH_ROLE_SQL.rstrip().rstrip(";")
    sql = "BEGIN READ WRITE;\n" + WRITE_SETTINGS
    sql += "DO $guard$ DECLARE state jsonb; BEGIN SELECT v::jsonb INTO state FROM (" + state_query + ") q(v);\n"
    sql += " IF (state-'read_only')<> (" + json_literal(before) + "-'read_only') OR state->>'read_only'<>'off' THEN RAISE EXCEPTION 'target state changed'; END IF; END $guard$;\n"
    for name in ROLES:
        sql += "CREATE ROLE " + snapshot.ident(name) + " NOLOGIN " + ("NOINHERIT" if name in restore.NOINHERIT else "INHERIT") + " NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS;\n"
    sql += "GRANT " + ",".join(snapshot.ident(n) for n in OWNERS) + " TO arena_admin WITH INHERIT TRUE, SET TRUE;\n"
    sql += "GRANT CREATE ON DATABASE postgres TO postgres,supabase_admin;\nGRANT CREATE ON SCHEMA public TO postgres,lab_arena_owner;\n"
    sql += "CREATE SCHEMA extensions AUTHORIZATION postgres;\nGRANT CREATE ON SCHEMA extensions TO supabase_admin;\n"
    sql += "CREATE SCHEMA auth AUTHORIZATION supabase_admin;\n"
    for roles, privileges in (("anon", "USAGE"), ("authenticated", "USAGE"), ("service_role", "USAGE"),
                              ("supabase_auth_admin", "USAGE,CREATE"), ("dashboard_user", "USAGE,CREATE"), ("postgres", "USAGE")):
        sql += "GRANT " + privileges + " ON SCHEMA auth TO " + roles + ";\n"
    sql += support["function"]["definition"] + ";\nALTER FUNCTION auth.role() OWNER TO supabase_auth_admin;\nGRANT EXECUTE ON FUNCTION auth.role() TO dashboard_user;\n"
    sql += "DO $proof$ DECLARE state jsonb; getter jsonb; m jsonb; BEGIN\n"
    sql += " SELECT v::jsonb INTO state FROM (" + state_query + ") q(v);\n"
    sql += " IF state->'roles'<>" + json_literal(role_expectations()) + " THEN RAISE EXCEPTION 'role proof failed'; END IF;\n"
    expected_counts = {**before["counts"], "schemas": 2, "routines": 1}
    sql += " IF state->'counts'<>" + json_literal(expected_counts) + " OR state->>'other_sessions'<>'0' OR state->'public'->>'owner'<>'pg_database_owner' THEN RAISE EXCEPTION 'bootstrap scope changed'; END IF;\n"
    # C-ordering makes this permission proof independent of libc sort versions.
    schemas = [dict(name="auth",owner="supabase_admin",acl=dict(is_null=False,items=sorted(snapshot.AUTH_SCHEMA_ACL[1:-1].split(",")))),
               dict(name="extensions",owner="postgres",acl=dict(is_null=False,items=["postgres=UC/postgres","supabase_admin=C/postgres"]))]
    sql += " IF state->'bootstrap_schemas'<>" + json_literal(schemas) + " THEN RAISE EXCEPTION 'bootstrap schema proof failed'; END IF;\n"
    sql += " FOR m IN SELECT * FROM jsonb_array_elements(state->'memberships') LOOP\n"
    sql += " IF m->>'member'<>'arena_admin' OR m->>'name' NOT IN (" + ROLE_LITERALS + ") THEN RAISE EXCEPTION 'unexpected membership'; END IF;\n"
    sql += " IF m->>'name' NOT IN (" + ",".join(literal(n) for n in OWNERS) + ") AND (m->>'admin'<>'true' OR m->>'inherit'<>'false' OR m->>'set'<>'false') THEN RAISE EXCEPTION 'non-owner membership changed'; END IF; END LOOP;\n"
    sql += " IF (SELECT count(*) FROM pg_auth_members WHERE roleid=" + literal(snapshot.SOURCE_ROLE) + "::regrole)>1 THEN RAISE EXCEPTION 'export membership changed'; END IF;\n"
    sql += " SELECT v::jsonb INTO getter FROM (" + getter_query + ") q(v);\n"
    sql += " IF getter IS DISTINCT FROM " + json_literal(support) + " OR EXISTS(SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace WHERE n.nspname='auth') OR (SELECT count(*) FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace WHERE n.nspname='auth')<>1 THEN RAISE EXCEPTION 'auth getter proof failed'; END IF; END $proof$;\nCOMMIT;\n"
    return sql


def cleanup_sql(manifest):
    public = restore.source_public_schema(manifest["schema_metadata"])
    extensions = restore.source_extensions_schema(manifest["schema_metadata"])
    revoke = []
    for schema, role, owner, record in (("public","postgres","pg_database_owner",public),
                                        ("public","lab_arena_owner","pg_database_owner",public),
                                        ("extensions","supabase_admin","postgres",extensions)):
        keep = any(a.startswith(role + "=") and "C" in a.split("=", 1)[1].split("/", 1)[0]
                   and a.rsplit("/", 1)[1] == owner for a in record["acl"]["items"])
        if not keep:
            revoke.append("REVOKE CREATE ON SCHEMA " + snapshot.ident(schema) + " FROM " + snapshot.ident(role) + ";")
    return ("-- Later reviewed cleanup. Preserve CREATE privileges present in the captured source ACL.\n"
            "BEGIN READ WRITE;\n" + WRITE_SETTINGS
            + "DO $guard$ BEGIN IF current_user<>'arena_admin' OR session_user<>'arena_admin' OR current_database()<>'postgres' OR current_setting('server_version_num')::int NOT BETWEEN 170000 AND 179999 OR pg_is_in_recovery() THEN RAISE EXCEPTION 'cleanup target identity'; END IF; END $guard$;\n"
            + "REVOKE CREATE ON DATABASE postgres FROM postgres,supabase_admin;\n" + "\n".join(revoke) + "\nCOMMIT;\n")


@contextlib.contextmanager
def write_service(config):
    snapshot.require(config["host"] == snapshot.TARGET_HOST and config["user"] == "arena_admin"
                     and config["dbname"] == "postgres" and config["port"] == "5432", "write-target-identity")
    with snapshot.service(config) as env:
        path = Path(env["PGSERVICEFILE"])
        data = path.read_text()
        snapshot.require(data.count("default_transaction_read_only=on") == 1, "write-service")
        path.write_text(data.replace("default_transaction_read_only=on", "default_transaction_read_only=off"))
        yield env


def run(args):
    snapshot.require(os.geteuid() == 0, "root-required")
    snapshot.require(re.fullmatch(r"[0-9a-f]{64}", args.code_sha256 or "")
                     and args.code_sha256 == code_sha256(), "code-binding")
    manifest_path = Path(args.manifest).absolute()
    manifest = load_manifest(manifest_path, args.manifest_sha256)
    directory = Path(args.directory).absolute()
    if args.execute:
        plan = json.loads((directory / "plan.json").read_text())
        snapshot.require(plan["code_sha256"] == args.code_sha256 and plan["manifest_sha256"] == args.manifest_sha256
                         and args.sql_sha256 == plan["sql_sha256"] == snapshot.sha256_file(directory / "bootstrap.sql"), "reviewed-sql-binding")
    else:
        directory.mkdir(mode=0o700)
        (directory / "private-logs").mkdir(mode=0o700)
    logs = directory / "private-logs"
    config = snapshot.connection_config("target", snapshot.secret(TARGET_SECRET_ARN), snapshot.TARGET_HOST)
    with snapshot.service(config) as env:
        before = read_state(env, logs)
    guard_empty(before)
    sql = build_sql(before, manifest)
    sql_sha = hashlib.sha256(sql.encode()).hexdigest()
    if not args.execute:
        snapshot.private_json(directory / "before.json", before)
        snapshot.private_write(directory / "bootstrap.sql", sql.encode())
        snapshot.private_write(directory / "cleanup.sql", cleanup_sql(manifest).encode())
        snapshot.private_json(directory / "plan.json", dict(format=1, target=snapshot.TARGET_HOST,
            target_secret_arn=TARGET_SECRET_ARN, code_sha256=args.code_sha256, manifest_sha256=args.manifest_sha256,
            sql_sha256=sql_sha, cleanup_sha256=snapshot.sha256_file(directory / "cleanup.sql"),
            supporting_record_sha256=hashlib.sha256(json.dumps(manifest["supporting_auth_role"],sort_keys=True,separators=(",",":")).encode()).hexdigest(),
            definition_sha256=snapshot.AUTH_ROLE_DEFINITION_SHA256, locale=before["locale"],
            source_collation_actual_version_reported="2.40", target_glibc_package_reported="2.26-59.amzn2",
            roles=role_expectations()))
        print("bootstrap SQL prepared privately; no target writes")
        return
    snapshot.require(sql_sha == args.sql_sha256, "target-state-binding")
    snapshot.private_json(directory / "execution-started.json", dict(sql_sha256=sql_sha, code_sha256=args.code_sha256))
    with write_service(config) as env:
        output = snapshot.command(snapshot.psql_command(), env, logs, "bootstrap-sql", data=sql.encode())
    snapshot.private_write(directory / "bootstrap.stdout", output)
    snapshot.private_json(directory / "bootstrap-committed.json", dict(sql_sha256=sql_sha, status="psql-completed"))
    with snapshot.service(config) as env:
        after = read_state(env, logs)
        with snapshot.Keeper(env, logs) as keeper:
            support = snapshot.capture_supporting_auth_role(keeper)
    snapshot.private_json(directory / "after.json", after)
    snapshot.private_json(directory / "supporting-auth-role.json", support)
    print("bootstrap committed; private evidence saved")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print-code-sha256", action="store_true")
    parser.add_argument("--directory")
    parser.add_argument("--manifest")
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--code-sha256")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--sql-sha256")
    args = parser.parse_args(argv)
    if args.print_code_sha256:
        print(code_sha256())
        return 0
    if not all((args.directory,args.manifest,args.manifest_sha256,args.code_sha256)) or (args.execute and not args.sql_sha256):
        parser.error("directory, manifest, manifest-sha256 and code-sha256 are required; execute also requires sql-sha256")
    os.umask(0o077)
    previous = signal.signal(signal.SIGTERM, snapshot.interrupted)
    try:
        run(args)
        return 0
    except snapshot.Failure as error:
        print("ERROR phase=" + str(error), file=sys.stderr)
    except (Exception, KeyboardInterrupt):
        print("ERROR phase=target-bootstrap", file=sys.stderr)
    finally:
        signal.signal(signal.SIGTERM, previous)
    return 1


if __name__ == "__main__":
    sys.exit(main())
