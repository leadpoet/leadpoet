#!/usr/bin/env python3
"""Scoped native snapshot export and read-only data proof; no restore operation.

Run on the reviewed temporary runner. All artifacts are private. Child stderr
is retained privately and is never copied to operator output. No live operation
is performed at import. Local tests exercise low-level helpers, not CLI bypasses.

Both modes MUST run under an external systemd-run unit with these properties:
RuntimeMaxSec=14400, TimeoutStopSec=30, KillMode=control-group, User=root and
UMask=0077. The coordinator must verify the effective unit properties. Unlimited
SQL statement timeouts are permitted only under that four-hour hard deadline.
Precreate root-owned mode-0600 stdout/stderr files in a root-only mode-0700
directory and direct StandardOutput/StandardError to those files. Treat every
log as untrusted private data; never return raw command output through SSM.
The systemd control group is authoritative for terminating pg_dump workers.
"""

import argparse
import contextlib
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import signal
import subprocess
import sys
import tempfile


SOURCE_HOST = "aws-0-us-east-1.pooler.supabase.com"
TARGET_HOST = "leadpoet-subnet-arena.cgfmio6maf8s.us-east-1.rds.amazonaws.com"
SOURCE_ROLE = "leadpoet_aws_export_20261028"
SOURCE_USER = SOURCE_ROLE + ".qplwoislplkcegvdmbim"
SCHEMAS = ("public", "source_add_history", "extensions")
EXTENSIONS = ("pgcrypto", "uuid-ossp", "pg_trgm", "dblink", "http", "pg_stat_statements")
SOURCE_CA = "/etc/leadpoet-migration/supabase-prod-ca-2021.crt"
TARGET_CA = "/etc/leadpoet-migration/rds-us-east-1-bundle.pem"
IDENT = re.compile(r"[a-z_][a-z0-9_]*\Z", re.ASCII)
SNAPSHOT = re.compile(r"[0-9A-Fa-f]{8}-[0-9A-Fa-f]{8}-[0-9]+\Z", re.ASCII)
ARN = re.compile(r"arn:aws:secretsmanager:us-east-1:493765492819:secret:[A-Za-z0-9/_+=.@!-]+\Z", re.ASCII)
OPTIONS = ("-c default_transaction_read_only=on -c statement_timeout=0 "
           "-c idle_in_transaction_session_timeout=0 -c lock_timeout=5000 "
           "-c row_security=off -c client_encoding=UTF8 "
           "-c DateStyle=ISO -c IntervalStyle=postgres -c extra_float_digits=3 "
           "-c bytea_output=hex -c TimeZone=UTC -c search_path=pg_catalog")
BEGIN = "BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY;"
TRANSACTION_SETTINGS = """
SET LOCAL default_transaction_read_only = on;
SET LOCAL statement_timeout = 0;
SET LOCAL idle_in_transaction_session_timeout = 0;
SET LOCAL transaction_timeout = 0;
SET LOCAL lock_timeout = '5s';
SET LOCAL row_security = off;
SET LOCAL client_encoding = 'UTF8';
SET LOCAL DateStyle = 'ISO';
SET LOCAL IntervalStyle = 'postgres';
SET LOCAL extra_float_digits = 3;
SET LOCAL bytea_output = 'hex';
SET LOCAL TimeZone = 'UTC';
SET LOCAL search_path = pg_catalog;
"""


class Failure(Exception):
    """Message contains a fixed phase label, never child or row output."""


def require(condition, phase):
    if not condition:
        raise Failure(phase)


def ident(value):
    require(isinstance(value, str) and IDENT.fullmatch(value), "identifier")
    return '"' + value + '"'


def qualified(table):
    require(table["schema"] in SCHEMAS, "schema")
    return ident(table["schema"]) + "." + ident(table["name"])


def private_write(path, data):
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as out:
        out.write(data)
        out.flush()
        os.fsync(out.fileno())


def private_json(path, obj):
    private_write(path, (json.dumps(obj, sort_keys=True, indent=2) + "\n").encode())


def pg_environment():
    return {k: v for k, v in os.environ.items() if not k.startswith("PG")}


def secret(arn):
    require(isinstance(arn, str) and ARN.fullmatch(arn), "secret-arn")
    # SecretString and child stderr remain in memory. Never print either.
    result = subprocess.run(
        ["aws", "secretsmanager", "get-secret-value", "--region", "us-east-1",
         "--secret-id", arn, "--query", "SecretString", "--output", "json", "--no-cli-pager"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        env={**os.environ, "AWS_PAGER": ""}, timeout=60,
    )
    require(result.returncode == 0, "secret-fetch")
    try:
        value = json.loads(json.loads(result.stdout))
    except (ValueError, TypeError):
        raise Failure("secret-json") from None
    require(isinstance(value, dict), "secret-json")
    return value


def connection_config(mode, value, host):
    if mode == "source":
        require(host == SOURCE_HOST, "source-host")
        require(value.get("host") == host and str(value.get("port")) == "5432"
                and value.get("dbname") == "postgres" and value.get("role") == SOURCE_ROLE
                and value.get("user") == SOURCE_USER, "source-identity")
        user, ca = SOURCE_USER, SOURCE_CA
    else:
        require(mode == "target" and host == TARGET_HOST, "target-host")
        require(value.get("username") == "arena_admin", "target-identity")
        user, ca = "arena_admin", TARGET_CA
    password = value.get("password")
    require(isinstance(password, str) and password and not any(c in password for c in "\n\r\0"), "password-format")
    return dict(host=host, port="5432", dbname="postgres", user=user, password=password, ca=ca)


@contextlib.contextmanager
def service(config):
    """Only private temporary files carry usernames/passwords to libpq."""
    with tempfile.TemporaryDirectory(prefix="leadpoet-pg-") as directory:
        root = Path(directory)
        os.chmod(root, 0o700)
        # Config is constructed by connection_config; reject service-file injection.
        for key in ("host", "port", "dbname", "user", "ca"):
            require(isinstance(config[key], str) and not any(c in config[key] for c in "\n\r\0"), "service-format")
        private_write(root / "service", (
            "[migration]\n" + "\n".join(f"{key}={config[key]}" for key in ("host", "port", "dbname", "user"))
            + f"\nsslmode=verify-full\nsslrootcert={config['ca']}\nconnect_timeout=20\n"
            + "keepalives=1\nkeepalives_idle=30\nkeepalives_interval=10\nkeepalives_count=3\n"
            + "application_name=leadpoet_aws_snapshot\noptions=" + OPTIONS + "\n"
        ).encode())
        escape = lambda s: s.replace("\\", "\\\\").replace(":", "\\:")
        private_write(root / "pass", (":".join(escape(config[k]) for k in ("host", "port", "dbname", "user", "password")) + "\n").encode())
        yield {**pg_environment(), "PGSERVICEFILE": str(root / "service"), "PGPASSFILE": str(root / "pass")}


def psql_command():
    return ["psql", "--no-psqlrc", "-X", "-qAt", "--no-password", "--set=ON_ERROR_STOP=1", "--dbname=service=migration"]


def stop_child(proc):
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


@contextlib.contextmanager
def stderr_file(directory, phase):
    path = directory / (phase + "-" + secrets.token_hex(4) + ".stderr")
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as out:
        yield out


def command(args, env, logs, phase, data=None):
    with stderr_file(logs, phase) as err:
        result = subprocess.run(args, input=data, stdout=subprocess.PIPE, stderr=err, env=env, check=False)
    require(result.returncode == 0, phase)
    return result.stdout


class Keeper:
    def __init__(self, env, logs):
        self.env, self.logs, self.proc, self.err = env, logs, None, None

    def __enter__(self):
        self.err = stderr_file(self.logs, "keeper")
        out = self.err.__enter__()
        try:
            self.proc = subprocess.Popen(psql_command(), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=out, env=self.env, text=True, encoding="utf-8")
            self.query(BEGIN)
            # Session poolers may discard startup options. Record untouched
            # serialization defaults before applying transaction-local settings.
            self.initial_settings = self.one_json("""SELECT json_build_object(
              'timezone',current_setting('TimeZone'),
              'bytea_output',current_setting('bytea_output'));
            """)
            self.query(TRANSACTION_SETTINGS)
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def query(self, sql):
        marker = "END_" + secrets.token_hex(16)
        self.proc.stdin.write(sql + "\n\\echo " + marker + "\n")
        self.proc.stdin.flush()
        lines = []
        for line in self.proc.stdout:
            if line.rstrip("\n") == marker:
                return lines
            lines.append(line.rstrip("\n"))
        raise Failure("keeper-query")

    def one_json(self, sql):
        lines = self.query(sql)
        require(bool(lines), "query-shape")
        try:
            # PostgreSQL json_agg(row) can contain formatting line breaks.
            return json.loads("\n".join(lines))
        except ValueError:
            raise Failure("query-json") from None

    def snapshot(self):
        lines = self.query("SELECT pg_catalog.pg_export_snapshot();")
        require(len(lines) == 1 and SNAPSHOT.fullmatch(lines[0]), "snapshot-id")
        return lines[0]

    def __exit__(self, *_):
        if self.proc is not None:
            try:
                if self.proc.poll() is None:
                    self.proc.stdin.write("ROLLBACK;\n\\q\n")
                    self.proc.stdin.flush()
                    self.proc.stdin.close()
                    self.proc.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                stop_child(self.proc)
            finally:
                self.proc.stdin.close()
                self.proc.stdout.close()
        if self.err is not None:
            self.err.__exit__(None, None, None)


def check_identity(keeper, mode):
    info = keeper.one_json("""SELECT json_build_object(
      'role', current_user, 'session', session_user, 'db', current_database(),
      'version', current_setting('server_version_num')::int,
      'readonly', current_setting('transaction_read_only'),
      'rls', current_setting('row_security'));
    """)
    role = SOURCE_ROLE if mode == "source" else "arena_admin"
    require(info["role"] == info["session"] == role and info["db"] == "postgres"
            and 170000 <= info["version"] < 180000 and info["readonly"] == "on"
            and info["rls"] == "off", "database-identity")
    if mode == "source":
        attrs = keeper.one_json("""SELECT json_build_object(
          'safe', NOT rolsuper AND NOT rolcreaterole AND NOT rolcreatedb
             AND NOT rolreplication AND rolcanlogin,
          'expires', rolvaliduntil IS NOT NULL AND rolvaliduntil > current_timestamp
             AND rolvaliduntil < 'infinity'::timestamptz)
          FROM pg_catalog.pg_roles WHERE rolname=current_user;""")
        require(attrs["safe"] is True and attrs["expires"] is True, "source-role-scope")


def check_dump_defaults(keeper):
    # pg_dump 17 explicitly sets dates, intervals, floats and row_security, but
    # inherits TimeZone/bytea_output. Require canonical new-connection defaults
    # instead of assuming the pooler honored libpq startup options.
    require(keeper.initial_settings["timezone"] in ("UTC", "Etc/UTC")
            and keeper.initial_settings["bytea_output"] == "hex", "dump-session-defaults")


CATALOG = """SELECT coalesce(json_agg(t ORDER BY t.schema, t.name), '[]'::json) FROM (
 SELECT n.nspname AS schema, c.relname AS name, c.relkind AS kind,
   pg_catalog.row_security_active(c.oid) AS filtered_rls,
   EXISTS(SELECT 1 FROM pg_catalog.pg_inherits i WHERE i.inhrelid=c.oid OR i.inhparent=c.oid) AS inherited,
   (SELECT json_agg(json_build_object('name', a.attname, 'type',
     pg_catalog.format_type(a.atttypid,a.atttypmod), 'generated', a.attgenerated)
     ORDER BY a.attnum) FROM pg_catalog.pg_attribute a
     WHERE a.attrelid=c.oid AND a.attnum>0 AND NOT a.attisdropped) AS columns
 FROM pg_catalog.pg_class c JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace
 WHERE n.nspname IN ('public','source_add_history','extensions')
   AND c.relkind IN ('r','p','m','f')
) t;"""


def validate_tables(tables):
    require(isinstance(tables, list) and len(tables) == 48, "table-inventory")
    keys = []
    for table in tables:
        keys.append(qualified(table))
        require(table["kind"] == "r" and not table["filtered_rls"] and not table["inherited"], "unsupported-table")
        require(isinstance(table["columns"], list) and table["columns"], "column-inventory")
        columns = []
        for col in table["columns"]:
            columns.append(ident(col["name"]))
            require(not col["generated"], "generated-column")
        require(len(set(columns)) == len(columns), "duplicate-column")
    require(len(set(keys)) == len(keys), "duplicate-table")


OBJECT_KINDS = ("schemas", "relations", "columns", "constraints", "indexes", "views",
                "routines", "policies", "triggers", "sequence_definitions", "types", "default_acls")
SCHEMA_SQL = "('public','source_add_history','extensions')"


def acl(expression):
    return (f"json_build_object('is_null', {expression} IS NULL, 'items', "
            f"ARRAY(SELECT x::text FROM unnest({expression}) x ORDER BY x::text))")


def aggregate(select, order="t.schema,t.name"):
    return f"SELECT coalesce(json_agg(t ORDER BY {order}),'[]'::json) FROM ({select}) t;"


def nonextension(alias, catalog):
    return ("NOT EXISTS(SELECT 1 FROM pg_catalog.pg_depend d "
            f"WHERE d.classid='pg_catalog.{catalog}'::regclass AND d.objid={alias}.oid "
            "AND d.deptype='e')")


REL = f"FROM pg_catalog.pg_class c JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace WHERE n.nspname IN {SCHEMA_SQL}"
PROC = f"FROM pg_catalog.pg_proc p JOIN pg_catalog.pg_namespace n ON n.oid=p.pronamespace WHERE n.nspname IN {SCHEMA_SQL} AND {nonextension('p','pg_proc')}"
METADATA_QUERIES = {
    "schemas": aggregate(f"SELECT n.nspname AS schema,n.nspname AS name,pg_get_userbyid(n.nspowner) AS owner,{acl('n.nspacl')} AS acl FROM pg_catalog.pg_namespace n WHERE n.nspname IN {SCHEMA_SQL}"),
    "relations": aggregate(f"SELECT n.nspname AS schema,c.relname AS name,c.relkind AS kind,pg_get_userbyid(c.relowner) AS owner,{acl('c.relacl')} AS acl,c.relrowsecurity AS rls,c.relforcerowsecurity AS force_rls,c.relreplident AS replica_identity,ARRAY(SELECT unnest(c.reloptions) ORDER BY 1) AS options {REL} AND c.relkind IN ('r','p','v','m','S','f') AND {nonextension('c','pg_class')}"),
    "columns": aggregate(f"SELECT n.nspname AS schema,c.relname AS name,a.attnum AS position,a.attname AS column,pg_catalog.format_type(a.atttypid,a.atttypmod) AS type,a.attnotnull AS not_null,a.attidentity AS identity,a.attgenerated AS generated,pg_catalog.pg_get_expr(ad.adbin,ad.adrelid) AS default,co.collname AS collation,{acl('a.attacl')} AS acl FROM pg_catalog.pg_class c JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace JOIN pg_catalog.pg_attribute a ON a.attrelid=c.oid LEFT JOIN pg_catalog.pg_attrdef ad ON ad.adrelid=c.oid AND ad.adnum=a.attnum LEFT JOIN pg_catalog.pg_collation co ON co.oid=a.attcollation WHERE n.nspname IN {SCHEMA_SQL} AND c.relkind IN ('r','p','v','m','f') AND a.attnum>0 AND NOT a.attisdropped AND {nonextension('c','pg_class')}", "t.schema,t.name,t.position"),
    "constraints": aggregate(f"SELECT n.nspname AS schema,co.conname AS name,c.relname AS relation,co.contype AS kind,co.convalidated AS validated,co.condeferrable AS deferrable,co.condeferred AS initially_deferred,pg_catalog.pg_get_constraintdef(co.oid,false) AS definition FROM pg_catalog.pg_constraint co JOIN pg_catalog.pg_namespace n ON n.oid=co.connamespace LEFT JOIN pg_catalog.pg_class c ON c.oid=co.conrelid WHERE n.nspname IN {SCHEMA_SQL} AND {nonextension('co','pg_constraint')}", "t.schema,t.name,t.relation"),
    "indexes": aggregate(f"SELECT n.nspname AS schema,ci.relname AS name,c.relname AS relation,pg_catalog.pg_get_indexdef(i.indexrelid) AS definition,i.indisvalid AS valid,i.indisready AS ready,i.indisprimary AS primary_key,i.indisunique AS unique_key FROM pg_catalog.pg_index i JOIN pg_catalog.pg_class c ON c.oid=i.indrelid JOIN pg_catalog.pg_class ci ON ci.oid=i.indexrelid JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace WHERE n.nspname IN {SCHEMA_SQL} AND {nonextension('c','pg_class')}"),
    "views": aggregate(f"SELECT n.nspname AS schema,c.relname AS name,pg_catalog.pg_get_viewdef(c.oid,false) AS definition {REL} AND c.relkind IN ('v','m') AND {nonextension('c','pg_class')}"),
    "routines": aggregate(f"SELECT n.nspname AS schema,p.proname AS name,pg_catalog.pg_get_function_identity_arguments(p.oid) AS arguments,p.prokind AS kind,pg_get_userbyid(p.proowner) AS owner,{acl('p.proacl')} AS acl,p.prosecdef AS security_definer,p.proleakproof AS leakproof,p.provolatile AS volatility,p.proparallel AS parallel,p.proconfig AS config,CASE WHEN p.prokind<>'a' THEN pg_catalog.pg_get_functiondef(p.oid) END AS definition {PROC}", "t.schema,t.name,t.arguments"),
    "policies": aggregate(f"SELECT n.nspname AS schema,p.polname AS name,c.relname AS relation,p.polpermissive AS permissive,p.polcmd AS command,ARRAY(SELECT CASE WHEN x=0 THEN 'PUBLIC' ELSE pg_get_userbyid(x) END FROM unnest(p.polroles) x ORDER BY 1) AS roles,pg_catalog.pg_get_expr(p.polqual,p.polrelid) AS using,pg_catalog.pg_get_expr(p.polwithcheck,p.polrelid) AS check FROM pg_catalog.pg_policy p JOIN pg_catalog.pg_class c ON c.oid=p.polrelid JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace WHERE n.nspname IN {SCHEMA_SQL}", "t.schema,t.relation,t.name"),
    "triggers": aggregate(f"SELECT n.nspname AS schema,t.tgname AS name,c.relname AS relation,t.tgenabled AS enabled,pg_catalog.pg_get_triggerdef(t.oid,false) AS definition FROM pg_catalog.pg_trigger t JOIN pg_catalog.pg_class c ON c.oid=t.tgrelid JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace WHERE n.nspname IN {SCHEMA_SQL} AND NOT t.tgisinternal", "t.schema,t.relation,t.name"),
    "sequence_definitions": aggregate(f"SELECT q.schemaname AS schema,q.sequencename AS name,q.sequenceowner AS owner,q.data_type::text AS type,q.start_value,q.min_value,q.max_value,q.increment_by,q.cycle,q.cache_size,(SELECT json_build_object('schema',tn.nspname,'table',tc.relname,'column',a.attname,'kind',d.deptype) FROM pg_catalog.pg_depend d JOIN pg_catalog.pg_class tc ON tc.oid=d.refobjid JOIN pg_catalog.pg_namespace tn ON tn.oid=tc.relnamespace JOIN pg_catalog.pg_attribute a ON a.attrelid=tc.oid AND a.attnum=d.refobjsubid WHERE d.classid='pg_catalog.pg_class'::regclass AND d.objid=s.oid AND d.refclassid='pg_catalog.pg_class'::regclass AND d.deptype IN ('a','i')) AS owned_by FROM pg_catalog.pg_sequences q JOIN pg_catalog.pg_namespace n ON n.nspname=q.schemaname JOIN pg_catalog.pg_class s ON s.relnamespace=n.oid AND s.relname=q.sequencename WHERE q.schemaname IN {SCHEMA_SQL}"),
    "types": aggregate(f"SELECT n.nspname AS schema,t.typname AS name,t.typtype AS kind,pg_get_userbyid(t.typowner) AS owner,{acl('t.typacl')} AS acl,pg_catalog.format_type(t.typbasetype,t.typtypmod) AS base,t.typnotnull AS not_null,t.typdefault AS default,ARRAY(SELECT e.enumlabel FROM pg_catalog.pg_enum e WHERE e.enumtypid=t.oid ORDER BY e.enumsortorder) AS labels FROM pg_catalog.pg_type t JOIN pg_catalog.pg_namespace n ON n.oid=t.typnamespace WHERE n.nspname IN {SCHEMA_SQL} AND t.typtype IN ('e','d') AND {nonextension('t','pg_type')}"),
    "default_acls": aggregate(f"SELECT coalesce(n.nspname,'') AS schema,pg_get_userbyid(d.defaclrole) AS name,d.defaclobjtype AS kind,{acl('d.defaclacl')} AS acl FROM pg_catalog.pg_default_acl d LEFT JOIN pg_catalog.pg_namespace n ON n.oid=d.defaclnamespace WHERE d.defaclnamespace=0 OR n.nspname IN {SCHEMA_SQL}", "t.schema,t.name,t.kind"),
}


def capture_metadata(keeper):
    """Canonical catalog evidence, without rows, passwords, or physical OIDs."""
    return {key: keeper.one_json(query) for key, query in METADATA_QUERIES.items()}


def capture_manifest(keeper, snapshot):
    tables = keeper.one_json(CATALOG)
    validate_tables(tables)
    keeper.query("LOCK TABLE " + ",".join(qualified(t) for t in tables) + " IN ACCESS SHARE MODE;")
    for table in tables:
        table["count"] = keeper.one_json("SELECT count(*) FROM ONLY " + qualified(table) + ";")
    sequences = keeper.one_json("""SELECT coalesce(json_agg(json_build_object('schema',n.nspname,'name',c.relname)
      ORDER BY n.nspname,c.relname),'[]'::json) FROM pg_catalog.pg_class c
      JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace
      WHERE n.nspname IN ('public','source_add_history','extensions') AND c.relkind='S';""")
    require(len(sequences) == 3, "sequence-inventory")
    for seq in sequences:
        seq["state"] = keeper.one_json("SELECT json_build_object('last_value',last_value::text,'is_called',is_called) FROM " + qualified(seq) + ";")
    extensions = keeper.one_json("""SELECT json_agg(json_build_object('name',e.extname,'version',e.extversion,
      'schema',n.nspname) ORDER BY e.extname) FROM pg_catalog.pg_extension e
      JOIN pg_catalog.pg_namespace n ON n.oid=e.extnamespace;""")
    require(set(EXTENSIONS) <= {e["name"] for e in extensions}, "extension-inventory")
    return dict(format=1, source_host=SOURCE_HOST, source_role=SOURCE_ROLE, database="postgres",
                snapshot=snapshot, serialization=OPTIONS, tables=tables, sequences=sequences,
                sequence_caveat="Sequence state is non-MVCC and may advance independently of the table snapshot.",
                extensions=extensions, schema_metadata=capture_metadata(keeper),
                dump_session_defaults=keeper.initial_settings)


def dump_command(snapshot, directory):
    require(SNAPSHOT.fullmatch(snapshot), "snapshot-id")
    return ["pg_dump", "--dbname=service=migration", "--no-password", "-Fd", "-j2",
            "--encoding=UTF8", "--extra-float-digits=3",
            "--snapshot=" + snapshot, "--lock-wait-timeout=5s", "--strict-names",
            "--no-publications", "--no-subscriptions", "--compress=gzip:1",
            *["--schema=" + s for s in SCHEMAS],
            *["--extension=" + e for e in EXTENSIONS], "--file=" + str(directory)]


def table_data_entries(toc):
    entries = {}
    for line in toc.decode("utf-8").splitlines():
        if line.startswith(";") or " TABLE DATA " not in line:
            continue
        match = re.fullmatch(r"([0-9]+); [0-9]+ [0-9]+ TABLE DATA ([a-z_][a-z0-9_]*) ([a-z_][a-z0-9_]*) ([^\s]+)", line, re.ASCII)
        require(match is not None, "toc-entry")
        number, schema, name, _ = match.groups()
        require(schema in SCHEMAS and (schema, name) not in entries, "toc-duplicate")
        entries[schema, name] = number + ".dat.gz"
    require(len(entries) == 48, "toc-table-count")
    return entries


def stream_digest(stream, archive=False):
    """Order-independent multiset of exact COPY rows, using bounded chunks.

    COPY text escapes embedded newlines, so LF always frames a row. Actual
    directory archives append \\. plus blank lines; psql COPY TO does not.
    Count and sum preserve multiplicity even when XOR cancels duplicates.
    """
    count = total = xor = 0
    row = hashlib.sha256()
    length, prefix, ended = 0, b"", False
    while True:
        chunk = stream.read(1024 * 1024)
        if not chunk:
            break
        parts = chunk.split(b"\n")
        for index, part in enumerate(parts):
            complete = index < len(parts) - 1
            piece = part + (b"\n" if complete else b"")
            if ended:
                require(not piece.strip(b"\n"), "archive-trailing-data")
                continue
            row.update(piece)
            prefix = (prefix + piece)[:3]
            length += len(piece)
            if complete:
                if length == 3 and prefix == b"\\.\n":
                    require(archive, "unexpected-copy-terminator")
                    ended = True
                else:
                    value = int.from_bytes(row.digest(), "big")
                    count += 1
                    total = (total + value) % (1 << 256)
                    xor ^= value
                row, length, prefix = hashlib.sha256(), 0, b""
    require(length == 0 and (ended if archive else not ended), "copy-framing")
    return dict(count=count, sum256=f"{total:064x}", xor256=f"{xor:064x}")


def sha256_file(path):
    require(path.is_file() and not path.is_symlink(), "archive-file")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_tools(env, logs):
    for tool in ("psql", "pg_dump", "pg_restore"):
        out = command([tool, "--version"], env, logs, "tool-version")
        require(re.match(rb"[a-z_]+ \(PostgreSQL\) 17\.", out), "tool-version")


def validate_archive(directory, manifest, env, logs):
    toc = command(["pg_restore", "--list", str(directory)], env, logs, "archive-list")
    entries = table_data_entries(toc)
    require(set(entries) == {(t["schema"], t["name"]) for t in manifest["tables"]}, "archive-table-set")
    sequence_entries = re.findall(
        r"^[0-9]+; [0-9]+ [0-9]+ SEQUENCE SET ([a-z_][a-z0-9_]*) ([a-z_][a-z0-9_]*) [^\s]+$",
        toc.decode("utf-8"), re.MULTILINE | re.ASCII)
    require(len(sequence_entries) == 3 and set(sequence_entries)
            == {(s["schema"], s["name"]) for s in manifest["sequences"]}, "archive-sequence-set")
    files = {}
    for path in directory.iterdir():
        require(path.name == "toc.dat" or re.fullmatch(r"[0-9]+\.dat\.gz", path.name), "archive-filename")
        files[path.name] = sha256_file(path)
    for table in manifest["tables"]:
        name = entries[table["schema"], table["name"]]
        with gzip.open(directory / name, "rb") as stream:
            digest = stream_digest(stream, archive=True)
        require(digest["count"] == table["count"], "snapshot-count")
        if "digest" in table:
            require(table["digest"] == digest and table["data_file"] == name, "archive-digest")
        else:
            table.update(digest=digest, data_file=name)
    return files


def export(args):
    destination = Path(args.directory).absolute()
    partial = destination.with_name(destination.name + ".partial")
    require(not destination.exists() and not partial.exists(), "destination-exists")
    partial.mkdir(mode=0o700)
    logs = partial / "private-logs"
    logs.mkdir(mode=0o700)
    config = connection_config("source", secret(args.source_secret_arn), args.source_host)
    with service(config) as env:
        check_tools(env, logs)
        with Keeper(env, logs) as keeper:
            check_identity(keeper, "source")
            private_json(partial / "dump-session-defaults.json", keeper.initial_settings)
            check_dump_defaults(keeper)
            snapshot = keeper.snapshot()
            manifest = capture_manifest(keeper, snapshot)
            command(dump_command(snapshot, partial / "dump"), env, logs, "pg-dump")
        # Counts and dump are complete from the same snapshot. Release the
        # source transaction before local decompression and archive hashing.
        manifest["archive_files"] = validate_archive(partial / "dump", manifest, env, logs)
        private_json(partial / "manifest.json", manifest)
        private_write(partial / "manifest.sha256", (sha256_file(partial / "manifest.json") + "\n").encode())
    require(not destination.exists(), "destination-exists")
    partial.rename(destination)
    print("export complete: 48 tables; 3 sequence observations; private manifest sealed")


def copy_digest(table, snapshot, env, logs):
    require(SNAPSHOT.fullmatch(snapshot), "snapshot-id")
    columns = ",".join(ident(c["name"]) for c in table["columns"])
    # Import the keeper snapshot before any query can acquire another snapshot.
    sql = (BEGIN + "\nSET TRANSACTION SNAPSHOT '" + snapshot + "';\n"
           + TRANSACTION_SETTINGS + "\nCOPY "
           + qualified(table) + " (" + columns + ") TO STDOUT WITH (FORMAT text);\nROLLBACK;\n")
    with stderr_file(logs, "target-copy") as err:
        proc = subprocess.Popen(psql_command(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err, env=env)
        try:
            proc.stdin.write(sql.encode())
            proc.stdin.close()
            digest = stream_digest(proc.stdout)
            require(proc.wait() == 0, "target-copy")
            return digest
        finally:
            stop_child(proc)
            proc.stdout.close()


def verify_data(args):
    root = Path(args.directory).absolute()
    require(root.is_dir() and not root.is_symlink(), "export-directory")
    require(sha256_file(root / "manifest.json") == (root / "manifest.sha256").read_text().strip(), "manifest-seal")
    manifest = json.loads((root / "manifest.json").read_text())
    require(manifest["format"] == 1 and manifest["source_host"] == SOURCE_HOST
            and manifest["source_role"] == SOURCE_ROLE and manifest["database"] == "postgres"
            and manifest["serialization"] == OPTIONS, "manifest-identity")
    validate_tables(manifest["tables"])
    require(len(manifest["sequences"]) == 3, "sequence-inventory")
    for sequence in manifest["sequences"]:
        qualified(sequence)
    config = connection_config("target", secret(args.target_secret_arn), args.target_host)
    logs = root / "private-logs"
    with service(config) as env:
        check_tools(env, logs)
        files = validate_archive(root / "dump", manifest, env, logs)
        require(files == manifest["archive_files"], "archive-seal")
        receipt = dict(format=1, target_host=TARGET_HOST, database="postgres", tables=[])
        with Keeper(env, logs) as keeper:
            check_identity(keeper, "target")
            target_tables = keeper.one_json(CATALOG)
            validate_tables(target_tables)
            require([(t["schema"], t["name"], t["columns"]) for t in target_tables]
                    == [(t["schema"], t["name"], t["columns"]) for t in manifest["tables"]], "target-columns")
            snapshot = keeper.snapshot()
            for table in manifest["tables"]:
                digest = copy_digest(table, snapshot, env, logs)
                require(digest == table["digest"], "target-data-mismatch")
                result = dict(schema=table["schema"], name=table["name"], **digest)
                receipt["tables"].append(result)
                print(json.dumps(result, sort_keys=True))
        private_json(root / ("verified-data-" + secrets.token_hex(8) + ".json"), receipt)
    print("verify-data complete: all 48 table counts and full-row multiset digests match")


def interrupted(_signum, _frame):
    raise Failure("interrupted")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    exp = sub.add_parser("export")
    exp.add_argument("--directory", required=True)
    exp.add_argument("--source-secret-arn", required=True)
    exp.add_argument("--source-host", default=SOURCE_HOST, choices=[SOURCE_HOST])
    verify = sub.add_parser("verify-data")
    verify.add_argument("--directory", required=True)
    verify.add_argument("--target-secret-arn", required=True)
    verify.add_argument("--target-host", default=TARGET_HOST, choices=[TARGET_HOST])
    args = parser.parse_args(argv)
    os.umask(0o077)
    previous_handler = signal.signal(signal.SIGTERM, interrupted)
    try:
        (export if args.mode == "export" else verify_data)(args)
        return 0
    except Failure as error:
        print("ERROR phase=" + str(error), file=sys.stderr)
    except (Exception, KeyboardInterrupt):
        # No traceback or exception text: database errors can contain private rows.
        print("ERROR phase=operator-utility", file=sys.stderr)
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
    return 1


if __name__ == "__main__":
    sys.exit(main())
