"""Only mocked I/O and an opt-in disposable local PostgreSQL 17 control."""

import copy
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import aws_subnet_target_bootstrap as bootstrap
from scripts import aws_subnet_snapshot as snapshot


DEFINITION = "CREATE OR REPLACE FUNCTION auth.role()\n RETURNS text\n LANGUAGE sql\n STABLE\nAS $function$\n  select \n  coalesce(\n    nullif(current_setting('request.jwt.claim.role', true), ''),\n    (nullif(current_setting('request.jwt.claims', true), '')::jsonb ->> 'role')\n  )::text\n$function$\n"
HELPER_DEFINITION = "CREATE OR REPLACE FUNCTION extensions.pgrst_ddl_watch()\n RETURNS event_trigger\n LANGUAGE plpgsql\n SET search_path TO ''\nAS $function$\nDECLARE\n  cmd record;\nBEGIN\n  FOR cmd IN SELECT * FROM pg_event_trigger_ddl_commands()\n  LOOP\n    IF cmd.command_tag IN (\n      'CREATE SCHEMA', 'ALTER SCHEMA'\n    , 'CREATE TABLE', 'CREATE TABLE AS', 'SELECT INTO', 'ALTER TABLE'\n    , 'CREATE FOREIGN TABLE', 'ALTER FOREIGN TABLE'\n    , 'CREATE VIEW', 'ALTER VIEW'\n    , 'CREATE MATERIALIZED VIEW', 'ALTER MATERIALIZED VIEW'\n    , 'CREATE FUNCTION', 'ALTER FUNCTION'\n    , 'CREATE TRIGGER'\n    , 'CREATE TYPE', 'ALTER TYPE'\n    , 'CREATE RULE'\n    , 'COMMENT'\n    )\n    -- don't notify in case of CREATE TEMP table or other objects created on pg_temp\n    AND cmd.schema_name is distinct from 'pg_temp'\n    THEN\n      NOTIFY pgrst, 'reload schema';\n    END IF;\n  END LOOP;\nEND; $function$\n"


def manifest_fixture():
    support = dict(schema=dict(name="auth",owner="supabase_admin",acl=snapshot.AUTH_SCHEMA_ACL),
                   function=dict(schema="auth",name="role",arguments="",result="text",owner="supabase_auth_admin",
                                 acl=snapshot.AUTH_ROLE_ACL,security_definer=False,leakproof=False,volatility="s",
                                 parallel="u",strict=False,config=None,language="sql",definition=DEFINITION,
                                 sha256=hashlib.sha256(DEFINITION.encode()).hexdigest()),
                   dependencies=[dict(identity="auth",kind="n",name="auth",schema=None,type="schema")])
    public = dict(schema="public",name="public",owner="pg_database_owner",acl=dict(is_null=False,
                  items=["=U/pg_database_owner","pg_database_owner=UC/pg_database_owner","postgres=C/pg_database_owner"]))
    extensions = dict(schema="extensions",name="extensions",owner="postgres",acl=dict(is_null=False,
                      items=["=U/postgres","postgres=UC/postgres","supabase_admin=U/postgres"]))
    tables = [dict(schema="public",name=f"t_{i}",kind="r",filtered_rls=False,inherited=False,
                   columns=[dict(name="id",type="integer",generated="")]) for i in range(48)]
    return dict(format=1,source_host=snapshot.SOURCE_HOST,source_role=snapshot.SOURCE_ROLE,database="postgres",
                tables=tables,schema_metadata=dict(schemas=[public,extensions]),supporting_auth_role=support)


def empty_state():
    return dict(role="arena_admin",session="arena_admin",owner="arena_admin",database="postgres",version=170009,
                encoding="UTF8",read_only="on",recovery=False,master=dict(super=False,createdb=True,createrole=True,login=True),
                other_sessions=0,roles=[],memberships=[],counts=dict(relations=0),public=dict(owner="pg_database_owner"))


class BootstrapTests(unittest.TestCase):
    def test_fixed_target_code_binding_and_no_network_on_mismatch(self):
        self.assertEqual(bootstrap.TARGET_SECRET_ARN, "arn:aws:secretsmanager:us-east-1:493765492819:secret:rds!db-18d7b717-ab16-4254-8c1a-3ddbccf19ea5-KeHJ6K")
        args = SimpleNamespace(code_sha256="0"*64)
        with patch.object(bootstrap.os,"geteuid",return_value=0), patch.object(snapshot,"secret") as secret:
            with self.assertRaisesRegex(snapshot.Failure,"code-binding"):
                bootstrap.run(args)
        secret.assert_not_called()

    def test_guard_rejects_wrong_target_existing_work_and_sessions(self):
        baseline = empty_state()
        bootstrap.guard_empty(baseline)
        for key,value in (("role","other"),("session","other"),("owner","other"),("database","other"),
                          ("version",160000),("encoding","LATIN1"),("read_only","off"),("recovery",True),("other_sessions",1),
                          ("roles",[dict(name="postgres")]),("memberships",[{}]),("counts",dict(relations=1))):
            state = copy.deepcopy(baseline)
            state[key] = value
            with self.assertRaises(snapshot.Failure):
                bootstrap.guard_empty(state)

    def test_manifest_hash_getter_binding_and_source_aware_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot.private_json(root/"manifest.json",manifest_fixture())
            digest = snapshot.sha256_file(root/"manifest.json")
            snapshot.private_write(root/"manifest.sha256",(digest+"\n").encode())
            manifest = bootstrap.load_manifest(root/"manifest.json",digest)
            with self.assertRaisesRegex(snapshot.Failure,"manifest-seal"):
                bootstrap.load_manifest(root/"manifest.json","0"*64)
            cleanup = bootstrap.cleanup_sql(manifest)
            self.assertIn('REVOKE CREATE ON SCHEMA "public" FROM "lab_arena_owner";',cleanup)
            self.assertNotIn('REVOKE CREATE ON SCHEMA "public" FROM "postgres";',cleanup)
            self.assertIn('REVOKE CREATE ON SCHEMA "extensions" FROM "supabase_admin";',cleanup)
            manifest["schema_metadata"]["schemas"][0]["acl"]["items"] = ["postgres=C/another_grantor"]
            self.assertIn('REVOKE CREATE ON SCHEMA "public" FROM "postgres";',bootstrap.cleanup_sql(manifest))
            manifest["schema_metadata"]["schemas"][1]["owner"] = "supabase_admin"
            with self.assertRaisesRegex(snapshot.Failure,"source-extensions-schema"):
                bootstrap.build_sql(empty_state(),manifest)
        for forbidden in ("CASCADE","DROP OWNED","REINDEX","ALTER SCHEMA public OWNER","DROP SCHEMA"):
            self.assertNotIn(forbidden,cleanup)

    def test_write_service_is_private_target_only_and_removed(self):
        config = snapshot.connection_config("target",dict(username="arena_admin",password="synthetic-secret"),snapshot.TARGET_HOST)
        with bootstrap.write_service(config) as env:
            path = Path(env["PGSERVICEFILE"])
            self.assertIn("default_transaction_read_only=off",path.read_text())
            self.assertNotIn("PGPASSWORD",env)
            self.assertEqual(path.stat().st_mode&0o777,0o600)
            self.assertEqual(path.parent.stat().st_mode&0o777,0o700)
        self.assertFalse(path.parent.exists())
        config["host"] = snapshot.SOURCE_HOST
        with self.assertRaisesRegex(snapshot.Failure,"write-target-identity"):
            with bootstrap.write_service(config):
                pass


PG17 = Path("/opt/homebrew/opt/postgresql@17/bin")


@unittest.skipUnless(os.environ.get("RUN_AWS_BOOTSTRAP_LOCAL_PG17")=="1" and PG17.is_dir(),
                     "opt-in disposable local PostgreSQL 17 control")
class NativeBootstrapTests(unittest.TestCase):
    def test_nonsuper_atomic_bootstrap_rejects_stale_state_and_preserves_exact_acl(self):
        with tempfile.TemporaryDirectory(prefix="lp-bootstrap-",dir="/tmp") as directory:
            root = Path(directory)
            logs = root/"logs"
            logs.mkdir(mode=0o700)
            base = {**snapshot.pg_environment(),"PATH":str(PG17)+os.pathsep+os.environ["PATH"]}
            def run(args,data=None,env=base):
                return subprocess.run(args,input=data,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True).stdout
            with socket.socket() as sock:
                sock.bind(("127.0.0.1",0))
                port = sock.getsockname()[1]
            run([str(PG17/"initdb"),"-D",str(root/"db"),"-U","local_bootstrap","--auth=trust","--no-locale","--encoding=UTF8"])
            run([str(PG17/"pg_ctl"),"-D",str(root/"db"),"-l",str(root/"server.log"),"-o",f"-F -h 127.0.0.1 -p {port} -k {root}","-w","start"])
            bootstrap_admin = [str(PG17/"psql"),"-XqAt","-v","ON_ERROR_STOP=1","-h","127.0.0.1","-p",str(port),"-U","local_bootstrap","-d","postgres"]
            try:
                run(bootstrap_admin,data=b"CREATE ROLE arena_admin LOGIN NOSUPERUSER CREATEROLE CREATEDB INHERIT NOBYPASSRLS; ALTER DATABASE postgres OWNER TO arena_admin;")
                admin = bootstrap_admin.copy()
                admin[admin.index("-U")+1] = "arena_admin"
                def service(user):
                    path = root/(user+".service")
                    snapshot.private_write(path,(f"[migration]\nhost=127.0.0.1\nport={port}\nuser={user}\ndbname=postgres\nsslmode=disable\n").encode())
                    return {**base,"PGSERVICEFILE":str(path)}
                env = service("arena_admin")
                wrong = bootstrap.read_state(service("local_bootstrap"),logs)
                with self.assertRaisesRegex(snapshot.Failure,"target-identity"):
                    bootstrap.guard_empty(wrong)
                before = bootstrap.read_state(env,logs)
                bootstrap.guard_empty(before)
                sql = bootstrap.build_sql(before,manifest_fixture())
                # A stale reviewed plan must fail atomically before role DDL.
                run(admin,data=b"CREATE TABLE public.keep_me(id integer); INSERT INTO public.keep_me VALUES(1);")
                with self.assertRaises(snapshot.Failure):
                    bootstrap.guard_empty(bootstrap.read_state(env,logs))
                with self.assertRaises(snapshot.Failure):
                    snapshot.command(snapshot.psql_command(),env,logs,"stale-plan",data=sql.encode())
                self.assertEqual(run(admin,data=b"SELECT count(*) FROM public.keep_me;").strip(),b"1")
                self.assertEqual(bootstrap.read_state(env,logs)["roles"],[])
                run(admin,data=b"DROP TABLE public.keep_me;")
                # Force a local failure after every DDL statement: all role,
                # schema, getter and privilege changes must roll back together.
                fault = sql.rsplit("COMMIT;",1)[0]+"DO $$ BEGIN RAISE EXCEPTION 'synthetic rollback'; END $$;\nCOMMIT;\n"
                with self.assertRaises(snapshot.Failure):
                    snapshot.command(snapshot.psql_command(),env,logs,"rollback-control",data=fault.encode())
                self.assertEqual(bootstrap.read_state(env,logs),before)
                snapshot.command(snapshot.psql_command(),env,logs,"native-bootstrap",data=sql.encode())
                after = bootstrap.read_state(env,logs)
                self.assertEqual(after["roles"],bootstrap.role_expectations())
                self.assertEqual(after["public"]["owner"],"pg_database_owner")
                self.assertEqual([s["name"] for s in after["bootstrap_schemas"]],["auth","extensions"])
                with snapshot.Keeper(env,logs) as keeper:
                    support = snapshot.capture_supporting_auth_role(keeper)
                    keeper.query("SET LOCAL request.jwt.claim.role='service_role';")
                    self.assertEqual(keeper.one_json("SELECT to_json(auth.role());"),"service_role")
                    self.assertEqual(keeper.one_json("SELECT count(*) FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace WHERE n.nspname='auth';"),0)
                self.assertEqual(support,manifest_fixture()["supporting_auth_role"])
                # Restore one real audited source helper with native owners/ACLs.
                # It returns event_trigger but is never registered or executed.
                self.assertEqual(hashlib.sha256(HELPER_DEFINITION.encode()).hexdigest(),"2b4b5d702ddf70fbd1a05c9825f9f9016d5c0ef97dd3df5f5b012ea23d9b94b1")
                run(admin,data=b"CREATE DATABASE helper_source; GRANT CREATE ON DATABASE helper_source TO postgres;")
                source_admin = admin.copy()
                source_admin[-1] = "helper_source"
                source_sql = "CREATE SCHEMA extensions AUTHORIZATION postgres; GRANT USAGE ON SCHEMA extensions TO PUBLIC,supabase_admin; GRANT CREATE ON SCHEMA extensions TO supabase_admin;\n"+HELPER_DEFINITION+";\nALTER FUNCTION extensions.pgrst_ddl_watch() OWNER TO supabase_admin; GRANT EXECUTE ON FUNCTION extensions.pgrst_ddl_watch() TO postgres WITH GRANT OPTION; REVOKE CREATE ON SCHEMA extensions FROM supabase_admin;"
                run(source_admin,data=source_sql.encode())
                source_path = root/"helper-source.service"
                snapshot.private_write(source_path,Path(env["PGSERVICEFILE"]).read_bytes().replace(b"dbname=postgres",b"dbname=helper_source"))
                source_env = {**env,"PGSERVICEFILE":str(source_path)}
                snapshot.command(["pg_dump","--dbname=service=migration","-Fd","--schema=extensions","--file="+str(root/"helper-archive")],source_env,logs,"helper-dump")
                toc = snapshot.command(["pg_restore","--list",str(root/"helper-archive")],env,logs,"helper-list")
                ext_entry = next(line for line in toc.decode().splitlines() if " SCHEMA - extensions postgres" in line)
                selected,_ = bootstrap.restore.select_toc(toc,[dict(id=ext_entry.split(";",1)[0],sha256=hashlib.sha256(ext_entry.encode()).hexdigest(),reason="reviewed precreated extensions schema")],manifest_fixture()["schema_metadata"])
                snapshot.private_write(root/"helper.list",selected)
                snapshot.command(["pg_restore","--dbname=service=migration","--single-transaction","--exit-on-error","--use-list="+str(root/"helper.list"),str(root/"helper-archive")],env,logs,"helper-restore")
                cleanup = bootstrap.cleanup_sql(manifest_fixture())
                snapshot.command(snapshot.psql_command(),env,logs,"native-cleanup",data=cleanup.encode())
                self.assertEqual(run(admin,data=b"SELECT has_database_privilege('postgres','postgres','CREATE'),has_database_privilege('supabase_admin','postgres','CREATE');").strip(),b"f|f")
                self.assertEqual(run(admin,data=b"SELECT has_schema_privilege('postgres','public','CREATE'),has_schema_privilege('lab_arena_owner','public','CREATE');").strip(),b"t|f")
                self.assertEqual(run(admin,data=b"SELECT has_schema_privilege('supabase_admin','extensions','CREATE');").strip(),b"f")
                self.assertEqual(run(admin,data=b"SELECT pg_get_userbyid(proowner),proacl::text FROM pg_proc WHERE oid='extensions.pgrst_ddl_watch()'::regprocedure;").strip(),b"supabase_admin|{=X/supabase_admin,supabase_admin=X/supabase_admin,postgres=X*/supabase_admin}")
                with snapshot.Keeper(env,logs) as keeper:
                    ext = next(x for x in snapshot.capture_metadata(keeper)["schemas"] if x["name"]=="extensions")
                    self.assertEqual(ext,manifest_fixture()["schema_metadata"]["schemas"][1])
                self.assertEqual(run(admin,data=b"SELECT count(*) FROM pg_event_trigger;").strip(),b"0")
                with snapshot.Keeper(env,logs) as keeper:
                    self.assertEqual(snapshot.capture_supporting_auth_role(keeper),support)
                self.assertEqual(run(admin,data=b"SELECT rolsuper,rolcreaterole,rolcreatedb FROM pg_roles WHERE rolname=current_user;").strip(),b"f|t|t")
            finally:
                run([str(PG17/"pg_ctl"),"-D",str(root/"db"),"-m","fast","-w","stop"])


if __name__=="__main__":
    unittest.main()
