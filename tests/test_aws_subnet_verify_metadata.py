"""Post-restore proof controls; no AWS or production database connections."""
import contextlib
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
from scripts import aws_subnet_snapshot as snapshot
from scripts import aws_subnet_restore as restore
from scripts import aws_subnet_target_bootstrap as bootstrap
from scripts import aws_subnet_verify_metadata as verify
from tests.test_aws_subnet_restore import AUTH_ROLE_SUPPORT, auth_schema_sql, auth_function_sql


def manifest_fixture():
    tables = [dict(schema="public", name=f"t_{i}", kind="r", filtered_rls=False, inherited=False,
                   columns=[dict(name="id", type="integer", generated="")]) for i in range(47)]
    tables += [dict(schema="source_add_history", name="history", kind="r", filtered_rls=False,
                    inherited=False, columns=[dict(name="id", type="integer", generated="")])]
    sequences = [dict(schema="public", name=f"seq_{i}", last_value="999", is_called=True) for i in range(3)]
    metadata = {category: [] for category in snapshot.OBJECT_KINDS}
    acl = lambda owner, grant: dict(is_null=False, items=[owner + "=arwdDxt/" + owner, snapshot.SOURCE_ROLE + "=" + grant + "/" + owner])
    metadata["schemas"] = [dict(schema=s, name=s, owner="postgres", acl=dict(is_null=False,
        items=["postgres=UC/postgres", snapshot.SOURCE_ROLE + "=U/postgres"])) for s in snapshot.SCHEMAS]
    metadata["relations"] = [dict(schema=t["schema"], name=t["name"], kind="r", owner="postgres", acl=acl("postgres", "r"), rls=True) for t in tables]
    metadata["relations"] += [dict(schema=s["schema"], name=s["name"], kind="S", owner="postgres", acl=acl("postgres", "r")) for s in sequences]
    metadata["routines"] = [dict(schema="extensions" if i<6 else "public", name=f"f_{i}", arguments="", kind="f", definition="private function definition", owner="postgres", acl=dict(is_null=True, items=[])) for i in range(383)]
    return dict(format=1, source_host=snapshot.SOURCE_HOST, source_role=snapshot.SOURCE_ROLE,
                database="postgres", tables=tables, sequences=sequences, schema_metadata=metadata,
                supporting_auth_role=copy.deepcopy(AUTH_ROLE_SUPPORT))


def archive_fixture(manifest):
    return [dict(schema=s["schema"], name=s["name"], last_value=str(i+10), is_called=bool(i%2)) for i,s in enumerate(manifest["sequences"])]


def baseline_fixture():
    return dict(database_acl=dict(is_null=False, items=["=Tc/arena_admin", "arena_admin=CTc/arena_admin"]),
                counts=dict(large_objects=0, databases=0))


def actual_fixture(manifest, archive, initial):
    metadata = verify.clean_source_metadata(manifest)
    state = dict(role="arena_admin", session="arena_admin", owner="arena_admin", database="postgres",
                 version=170009, read_only="on", recovery=False, database_acl=copy.deepcopy(initial["database_acl"]),
                 roles=[r for r in bootstrap.role_expectations() if r["name"] != snapshot.SOURCE_ROLE],
                 memberships=[dict(name=n,member="arena_admin",grantor="arena_admin",admin=True,
                                   inherit=n in bootstrap.OWNERS,set=n in bootstrap.OWNERS)
                              for n in bootstrap.ROLES if n!=snapshot.SOURCE_ROLE],
                 counts=dict(relations=len(metadata["relations"])+len(metadata["indexes"]),
                 routines=len(metadata["routines"])+1, types=len(metadata["types"]), schemas=3,
                 default_acls=len(metadata["default_acls"]), event_triggers=0, large_objects=0, databases=0))
    return dict(metadata=metadata, state=state, supporting_auth_roles=[copy.deepcopy(manifest["supporting_auth_role"])], sequences=copy.deepcopy(archive))


class MetadataTests(unittest.TestCase):
    def test_only_exact_export_grants_are_removed(self):
        manifest = manifest_fixture()
        cleaned = verify.clean_source_metadata(manifest)
        self.assertEqual(len(cleaned["routines"]), 383)
        self.assertTrue(any(t["schema"]=="source_add_history" for t in cleaned["relations"]))
        self.assertTrue(all(not grant.startswith(snapshot.SOURCE_ROLE+"=") for records in cleaned.values()
                            for item in records for grant in item.get("acl",{}).get("items",[])))
        for category, item in (("routines", manifest["schema_metadata"]["routines"][0]),
                               ("relations", manifest["schema_metadata"]["relations"][0])):
            value=copy.deepcopy(manifest)
            target=value["schema_metadata"][category][0]
            target["acl"]=dict(is_null=False,items=[snapshot.SOURCE_ROLE+"=r*/postgres"])
            with self.assertRaisesRegex(snapshot.Failure,"export-acl-scope"):
                verify.clean_source_metadata(value)
        self.assertIn(snapshot.SOURCE_ROLE, json.dumps(manifest["schema_metadata"]))

    def test_acl_order_only_normalization_keeps_null_grantor_rights_and_options(self):
        manifest=manifest_fixture();archive=archive_fixture(manifest);before=baseline_fixture()
        actual=actual_fixture(manifest,archive,before)
        for part in ("schema","function"):
            source=manifest["supporting_auth_role"][part]["acl"]
            actual["supporting_auth_roles"][0][part]["acl"]="{"+",".join(reversed(source[1:-1].split(",")))+"}"
        actual["state"]["database_acl"]["items"].reverse()
        self.assertEqual(verify.compare(manifest,archive,before,actual),[])
        for replacement in (dict(is_null=True,items=[]), dict(is_null=False,items=[]),
                            dict(is_null=False,items=["=T/arena_admin","arena_admin=CTc/arena_admin"]),
                            dict(is_null=False,items=["=Tc/postgres","arena_admin=CTc/arena_admin"]),
                            dict(is_null=False,items=["=T*c/arena_admin","arena_admin=CTc/arena_admin"])):
            changed=copy.deepcopy(actual);changed["state"]["database_acl"]=replacement
            self.assertIn("database-acl",verify.compare(manifest,archive,before,changed))
        null_before=copy.deepcopy(before);null_before["database_acl"]=dict(is_null=True,items=[])
        self.assertIn("database-acl",verify.compare(manifest,archive,null_before,actual))

    def test_every_category_and_unreviewed_support_or_sequence_delta_fails(self):
        manifest=manifest_fixture();archive=archive_fixture(manifest);before=baseline_fixture()
        actual=actual_fixture(manifest,archive,before)
        for category in snapshot.OBJECT_KINDS:
            changed=copy.deepcopy(actual)
            changed["metadata"][category].append(dict(schema="public",name="unreviewed"))
            self.assertIn(category,verify.compare(manifest,archive,before,changed))
        changed=copy.deepcopy(actual);changed["state"]["roles"]=[dict(name=snapshot.SOURCE_ROLE)]
        self.assertIn("temporary-export-role",verify.compare(manifest,archive,before,changed))
        changed=copy.deepcopy(actual);changed["supporting_auth_roles"][0]["function"]["security_definer"]=True
        self.assertIn("supporting-auth-role",verify.compare(manifest,archive,before,changed))
        changed=copy.deepcopy(actual);changed["sequences"][0]["last_value"]="999"
        self.assertIn("archive-sequences",verify.compare(manifest,archive,before,changed))
        changed=copy.deepcopy(actual);changed["state"]["counts"]["event_triggers"]=1
        self.assertIn("unexpected-objects",verify.compare(manifest,archive,before,changed))

    def test_role_contract_rejects_unsafe_attributes_and_unexpected_memberships(self):
        manifest=manifest_fixture();archive=archive_fixture(manifest);before=baseline_fixture()
        actual=actual_fixture(manifest,archive,before)
        self.assertTrue(verify.target_roles_match(actual["state"]))
        for flag in ("login","super","createdb","createrole","replication","bypassrls"):
            changed=copy.deepcopy(actual);changed["state"]["roles"][0][flag]=True
            self.assertIn("target-role-security",verify.compare(manifest,archive,before,changed))
        for flag in ("admin","inherit","set"):
            changed=copy.deepcopy(actual)
            grant=next(m for m in changed["state"]["memberships"] if m["name"] not in bootstrap.OWNERS)
            grant[flag]=not grant[flag]
            self.assertFalse(verify.target_roles_match(changed["state"]))
        changed=copy.deepcopy(actual);changed["state"]["memberships"][0]["member"]="anon"
        self.assertFalse(verify.target_roles_match(changed["state"]))
        changed=copy.deepcopy(actual)
        changed["state"]["memberships"].append(dict(name="postgres",member="arena_admin",grantor="local_bootstrap",admin=True,inherit=False,set=False))
        self.assertTrue(verify.target_roles_match(changed["state"]))
        changed["state"]["memberships"].append(copy.deepcopy(changed["state"]["memberships"][-1]))
        self.assertFalse(verify.target_roles_match(changed["state"]))

    def test_invalid_reference_stops_before_secret_access_and_missing_auth_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"manifest.json"
            snapshot.private_json(path,manifest_fixture())
            with patch.object(snapshot,"secret") as secret:
                with self.assertRaisesRegex(snapshot.Failure,"reference-seal"):
                    verify.verify(SimpleNamespace(manifest=str(path),manifest_sha256="0"*64))
                secret.assert_not_called()
        manifest=manifest_fixture();archive=archive_fixture(manifest);before=baseline_fixture()
        actual=actual_fixture(manifest,archive,before)
        actual["supporting_auth_roles"]=[]
        actual["sequences"]=[]
        self.assertIn("supporting-auth-role",verify.compare(manifest,archive,before,actual))
        self.assertIn("archive-sequences",verify.compare(manifest,archive,before,actual))

    def test_cli_preserves_private_evidence_and_never_prints_definitions(self):
        manifest=manifest_fixture();archive=archive_fixture(manifest);before=baseline_fixture()
        actual=actual_fixture(manifest,archive,before)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name,value in (("manifest.json",manifest),("sequences.json",archive),("before.json",before)):
                snapshot.private_json(root/name,value)
            mh=snapshot.sha256_file(root/"manifest.json")
            snapshot.private_write(root/"manifest.sha256",(mh+"\n").encode())
            args=SimpleNamespace(manifest=str(root/"manifest.json"),manifest_sha256=mh,
                archive_sequences=str(root/"sequences.json"),archive_sequences_sha256=snapshot.sha256_file(root/"sequences.json"),
                initial_target=str(root/"before.json"),initial_target_sha256=snapshot.sha256_file(root/"before.json"),directory=str(root/"proof"))
            output=io.StringIO()
            with patch.object(bootstrap,"guard_empty"),patch.object(snapshot,"secret",return_value={}), \
                 patch.object(snapshot,"connection_config",return_value={}),patch.object(snapshot,"service"), \
                 patch.object(snapshot,"Keeper"),patch.object(verify,"capture_target",return_value=actual),contextlib.redirect_stdout(output):
                verify.verify(args)
            self.assertNotIn("private function definition",output.getvalue())
            self.assertEqual(json.loads(output.getvalue())["status"],"match")
            for path in (root/"proof").iterdir():
                self.assertEqual(path.stat().st_mode&0o777,0o700 if path.is_dir() else 0o600)
            self.assertEqual(json.loads((root/"proof"/"actual.json").read_text()),actual)
        output=io.StringIO()
        with patch.object(verify,"verify",side_effect=RuntimeError("private function definition password")),contextlib.redirect_stderr(output):
            code=verify.main(["--manifest","x","--manifest-sha256","x","--archive-sequences","x", "--archive-sequences-sha256","x","--initial-target","x","--initial-target-sha256","x","--directory","x"])
        self.assertEqual(code,1)
        self.assertEqual(output.getvalue(),"ERROR phase=metadata-verification\n")


PG17=Path("/opt/homebrew/opt/postgresql@17/bin")


@unittest.skipUnless(os.environ.get("RUN_AWS_METADATA_LOCAL_PG17")=="1" and PG17.is_dir(),"opt-in disposable local PG17 proof")
class NativeMetadataTests(unittest.TestCase):
    def test_real_catalog_parity_cleanup_and_detected_drift(self):
        with tempfile.TemporaryDirectory(prefix="lp-meta-",dir="/tmp") as directory:
            root=Path(directory);logs=root/"logs";logs.mkdir(mode=0o700)
            env={k:v for k,v in os.environ.items() if not k.startswith("PG")}
            env["PATH"]=str(PG17)+os.pathsep+env["PATH"]
            def run(args,data=None):
                return subprocess.run(args,input=data,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env,check=True).stdout
            with socket.socket() as sock:
                sock.bind(("127.0.0.1",0));port=sock.getsockname()[1]
            run([str(PG17/"initdb"),"-D",str(root/"db"),"-U","local_bootstrap","--auth=trust","--no-locale","--encoding=UTF8"])
            run([str(PG17/"pg_ctl"),"-D",str(root/"db"),"-l",str(root/"server.log"),"-o",f"-F -h 127.0.0.1 -p {port} -k {root}","-w","start"])
            admin=[str(PG17/"psql"),"-XqAt","-v","ON_ERROR_STOP=1","-h","127.0.0.1","-p",str(port),"-U","local_bootstrap","-d","postgres"]
            try:
                run(admin,b"CREATE ROLE arena_admin LOGIN NOSUPERUSER CREATEROLE CREATEDB INHERIT NOBYPASSRLS; ALTER DATABASE postgres OWNER TO arena_admin;")
                master=admin.copy();master[master.index("-U")+1]="arena_admin"
                run(master,b"GRANT CONNECT,TEMPORARY ON DATABASE postgres TO PUBLIC;")
                def read_env(user):
                    path=root/(user+".service")
                    snapshot.private_write(path,(f"[migration]\nhost=127.0.0.1\nport={port}\nuser={user}\ndbname=postgres\nsslmode=disable\noptions={snapshot.OPTIONS}\n").encode())
                    return {**env,"PGSERVICEFILE":str(path)}
                target_env=read_env("arena_admin");source_env=read_env("local_bootstrap")
                with snapshot.Keeper(target_env,logs) as keeper:
                    before=keeper.one_json(bootstrap.STATE_SQL)
                bootstrap.guard_empty(before)
                sql="\n".join("CREATE ROLE "+snapshot.ident(n)+" NOLOGIN "+("NOINHERIT" if n in restore.NOINHERIT else "INHERIT")+" NOBYPASSRLS;" for n in bootstrap.ROLES)
                sql+="GRANT postgres,lab_arena_owner,supabase_admin,supabase_auth_admin TO arena_admin WITH INHERIT TRUE,SET TRUE; GRANT CREATE ON DATABASE postgres TO postgres,lab_arena_owner,supabase_admin,supabase_auth_admin; GRANT CREATE ON SCHEMA public TO postgres,lab_arena_owner; CREATE SCHEMA source_add_history AUTHORIZATION lab_arena_owner; CREATE SCHEMA extensions AUTHORIZATION postgres; GRANT CREATE ON SCHEMA extensions TO supabase_admin;"
                sql+=auth_schema_sql()+auth_function_sql()
                sql+="CREATE TYPE public.qualification_status AS ENUM ('new'); CREATE SEQUENCE public.seq_0; CREATE SEQUENCE public.seq_1; CREATE SEQUENCE source_add_history.seq_2;"
                for i in range(48):
                    schema,name,owner=("source_add_history","history","lab_arena_owner") if i==47 else ("public",f"t_{i}","postgres")
                    sql+=f"CREATE TABLE {schema}.{name}(id integer PRIMARY KEY,val text DEFAULT 'SOURCE_ADD' CHECK(length(val)>0)); ALTER TABLE {schema}.{name} OWNER TO {owner};"
                sql+="CREATE INDEX history_val_idx ON source_add_history.history(val); ALTER TABLE public.t_0 ENABLE ROW LEVEL SECURITY; ALTER TABLE public.t_0 FORCE ROW LEVEL SECURITY; CREATE POLICY service_read ON public.t_0 TO lab_arena_service USING(true); ALTER TABLE public.t_0 ADD COLUMN generated_id integer GENERATED ALWAYS AS (COALESCE(id,0)) STORED;"
                for i in range(3):sql+=f"CREATE VIEW public.v_{i} AS SELECT id FROM public.t_{i}; ALTER VIEW public.v_{i} OWNER TO postgres;"
                sql+="CREATE FUNCTION public.f_0() RETURNS trigger LANGUAGE plpgsql AS $$BEGIN RETURN NEW; END$$; ALTER FUNCTION public.f_0() OWNER TO postgres; CREATE TRIGGER audit_t0 BEFORE INSERT ON public.t_0 FOR EACH ROW EXECUTE FUNCTION public.f_0();"
                for i in range(1,377):sql+=f"CREATE FUNCTION public.f_{i}() RETURNS integer LANGUAGE sql AS $$SELECT {i}$$; ALTER FUNCTION public.f_{i}() OWNER TO postgres;"
                for i,name in enumerate(("grant_pg_cron_access","grant_pg_graphql_access","grant_pg_net_access","pgrst_ddl_watch","pgrst_drop_watch","set_graphql_placeholder")):
                    owner="postgres" if i<2 else "supabase_admin"
                    sql+=f"CREATE FUNCTION extensions.{name}() RETURNS event_trigger LANGUAGE plpgsql AS $$BEGIN RETURN; END$$; ALTER FUNCTION extensions.{name}() OWNER TO {owner};"
                tables=",".join([f"public.t_{i}" for i in range(47)]+["source_add_history.history"])
                sql+="GRANT USAGE ON SCHEMA public,source_add_history,extensions TO "+snapshot.SOURCE_ROLE+"; GRANT SELECT ON TABLE "+tables+" TO "+snapshot.SOURCE_ROLE+"; GRANT SELECT ON SEQUENCE public.seq_0,public.seq_1,source_add_history.seq_2 TO "+snapshot.SOURCE_ROLE+"; ALTER DEFAULT PRIVILEGES FOR ROLE lab_arena_owner IN SCHEMA source_add_history GRANT SELECT ON TABLES TO lab_arena_service; REVOKE CREATE ON SCHEMA public FROM postgres,lab_arena_owner; REVOKE CREATE ON SCHEMA extensions FROM supabase_admin; SELECT setval('public.seq_0',10,false); SELECT setval('public.seq_1',11,true); SELECT setval('source_add_history.seq_2',12,false);"
                run(master,sql.encode())
                with snapshot.Keeper(source_env,logs) as keeper:
                    metadata=snapshot.capture_metadata(keeper);tables=keeper.one_json(snapshot.CATALOG)
                seqs=[dict(schema="public",name="seq_0",last_value="999",is_called=True),dict(schema="public",name="seq_1",last_value="999",is_called=True),dict(schema="source_add_history",name="seq_2",last_value="999",is_called=True)]
                manifest=dict(tables=tables,sequences=seqs,schema_metadata=metadata,supporting_auth_role=copy.deepcopy(AUTH_ROLE_SUPPORT))
                snapshot.validate_tables(tables)
                archive=[dict(schema=s["schema"],name=s["name"],last_value=str(i+10),is_called=bool(i%2)) for i,s in enumerate(seqs)]
                with snapshot.Keeper(target_env,logs) as keeper:unclean=verify.capture_target(keeper)
                self.assertIn("temporary-export-role",verify.compare(manifest,archive,before,unclean))
                run(master,restore.export_role_cleanup(manifest).encode())
                run(master,b"REVOKE CREATE ON DATABASE postgres FROM postgres,lab_arena_owner,supabase_admin,supabase_auth_admin;")
                with snapshot.Keeper(target_env,logs) as keeper:actual=verify.capture_target(keeper)
                self.assertEqual(len(actual["metadata"]["routines"]),383)
                self.assertEqual(verify.compare(manifest,archive,before,actual),[])
                # A legitimate revoke/regrant restores identical privileges but
                # changes native ACL array order. No grantor/right/option changes.
                run(master,b"BEGIN; SET ROLE supabase_auth_admin; REVOKE EXECUTE ON FUNCTION auth.role() FROM PUBLIC; GRANT EXECUTE ON FUNCTION auth.role() TO PUBLIC; RESET ROLE; COMMIT;")
                with snapshot.Keeper(target_env,logs) as keeper:reordered=verify.capture_target(keeper)
                self.assertNotEqual(reordered["supporting_auth_roles"][0]["function"]["acl"],manifest["supporting_auth_role"]["function"]["acl"])
                self.assertEqual(verify.compare(manifest,archive,before,reordered),[])
                self.assertEqual(run(master,b"SELECT to_regclass('auth.users') IS NULL;").strip(),b"t")
                run(master,b"GRANT CREATE ON SCHEMA extensions TO supabase_admin;")
                with snapshot.Keeper(target_env,logs) as keeper:residual=verify.capture_target(keeper)
                self.assertIn("schemas",verify.compare(manifest,archive,before,residual))
                run(master,b"REVOKE CREATE ON SCHEMA extensions FROM supabase_admin;")
                run(admin,b"ALTER ROLE anon LOGIN BYPASSRLS;")
                with snapshot.Keeper(target_env,logs) as keeper:unsafe_roles=verify.capture_target(keeper)
                self.assertIn("target-role-security",verify.compare(manifest,archive,before,unsafe_roles))
                run(admin,b"ALTER ROLE anon NOLOGIN NOBYPASSRLS;")
                run(master,b"GRANT anon TO arena_admin WITH INHERIT TRUE,SET TRUE;")
                with snapshot.Keeper(target_env,logs) as keeper:unsafe_membership=verify.capture_target(keeper)
                self.assertIn("target-role-security",verify.compare(manifest,archive,before,unsafe_membership))
                run(master,b"GRANT anon TO arena_admin WITH INHERIT FALSE,SET FALSE;")
                drift="ALTER TABLE public.t_0 DISABLE ROW LEVEL SECURITY; ALTER TABLE public.t_0 ALTER COLUMN val SET DEFAULT 'changed'; DROP POLICY service_read ON public.t_0; DROP TRIGGER audit_t0 ON public.t_0; DROP INDEX source_add_history.history_val_idx; ALTER TABLE source_add_history.history DROP CONSTRAINT history_val_check; ALTER SCHEMA source_add_history OWNER TO postgres; CREATE OR REPLACE VIEW public.v_0 AS SELECT id+1 AS id FROM public.t_0; CREATE OR REPLACE FUNCTION public.f_1() RETURNS integer LANGUAGE sql AS $$SELECT 999$$; ALTER TYPE public.qualification_status ADD VALUE 'changed'; ALTER DEFAULT PRIVILEGES FOR ROLE lab_arena_owner IN SCHEMA source_add_history GRANT SELECT ON TABLES TO anon; GRANT CREATE ON DATABASE postgres TO postgres; GRANT EXECUTE ON FUNCTION auth.role() TO anon; SELECT setval('public.seq_0',999,true); ALTER SEQUENCE public.seq_1 INCREMENT BY 2;"
                run(admin,drift.encode())
                with snapshot.Keeper(target_env,logs) as keeper:changed=verify.capture_target(keeper)
                differences=verify.compare(manifest,archive,before,changed)
                for category in ("relations","columns","constraints","indexes","views","routines","policies","triggers","schemas","types","default_acls","sequence_definitions","database-acl","supporting-auth-role","archive-sequences"):
                    self.assertIn(category,differences)
            finally:
                run([str(PG17/"pg_ctl"),"-D",str(root/"db"),"-m","fast","-w","stop"])


if __name__=="__main__":unittest.main()
