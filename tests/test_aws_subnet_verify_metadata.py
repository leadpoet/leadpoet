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
    return dict(owner="arena_admin", database="postgres", database_acl=dict(is_null=False, items=["=Tc/arena_admin", "arena_admin=CTc/arena_admin"]),
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
    return dict(metadata=metadata, state=state, database_acl_default=dict(database="postgres", owner="arena_admin", acl=dict(is_null=False, items=["=Tc/arena_admin", "arena_admin=CTc/arena_admin"])), supporting_auth_roles=[copy.deepcopy(manifest["supporting_auth_role"])], sequences=copy.deepcopy(archive))


def http_proof_fixture(root, manifest, manifest_digest):
    """Synthetic sealed catalog proof; no source/API access."""
    current = dict(capture=dict(sealed_manifest_sha256=manifest_digest), evidence=dict(
        evidence_kind="supplemental-current-source-catalog-not-sealed-snapshot",
        project="qplwoislplkcegvdmbim", current_user="postgres", current_database="postgres",
        transaction_read_only="on", search_path="pg_catalog",
        extension=dict(**verify.HTTP_EXTENSION, owner="supabase_admin"),
        members=[copy.deepcopy(verify.HTTP_DOMAIN_MEMBER)],
        direct_non_extension_owned_dependents=[copy.deepcopy(verify.HTTP_DOMAIN_DEPENDENCY)]))
    current_path = root / "source-members-pg-catalog.json"
    snapshot.private_json(current_path, current)
    proof = dict(format=1, manifest_sha256=manifest_digest, extension=copy.deepcopy(verify.HTTP_EXTENSION),
        current_source_members=dict(file=current_path.name, sha256=snapshot.sha256_file(current_path)),
        constraint=copy.deepcopy(verify.HTTP_DOMAIN_CONSTRAINT))
    path = root / "http-domain-proof.json"
    snapshot.private_json(path, proof)
    return path, proof, current


class MetadataTests(unittest.TestCase):
    def test_column_ordinals_preserve_visible_order_all_other_fields_and_inputs(self):
        manifest=manifest_fixture()
        def column(table,name,position):
            return dict(schema="public",name=table,column=name,position=position,type="integer",
                not_null=False,identity="",generated="",default="7",collation=None,
                acl=dict(is_null=True,items=[]))
        columns=[column("alpha","a",1),column("alpha","b",6),column("alpha","c",10),
                 column("beta","a",2),column("beta","b",7)]
        manifest["schema_metadata"]["columns"]=copy.deepcopy(columns)
        original=copy.deepcopy(manifest)
        expected=verify.clean_source_metadata(manifest)
        self.assertEqual([c["position"] for c in expected["columns"]],[1,2,3,1,2])
        self.assertEqual(verify.canonical_columns(expected["columns"]),expected["columns"])
        self.assertEqual(manifest,original)
        for raw,canonical in zip(columns,expected["columns"]):
            self.assertEqual({k:v for k,v in raw.items() if k!="position"},
                             {k:v for k,v in canonical.items() if k!="position"})
        actual=copy.deepcopy(expected)
        actual["columns"]=copy.deepcopy(columns)
        self.assertEqual(verify.canonical_metadata(actual),expected)
        self.assertEqual(actual["columns"],columns)
        for key,value in (("column","renamed"),("type","bigint"),("not_null",True),
                          ("identity","a"),("generated","s"),("default","8"),
                          ("collation","C"),("acl",dict(is_null=False,items=["anon=r/postgres"]))):
            changed=copy.deepcopy(expected);changed["columns"][0][key]=value
            self.assertEqual(restore.compare_metadata(expected,verify.canonical_metadata(changed)),["columns"])
        # A logical reorder with valid monotonic positions must remain visible.
        changed=copy.deepcopy(expected)
        changed["columns"][0],changed["columns"][1]=changed["columns"][1],changed["columns"][0]
        for i,item in enumerate(changed["columns"][:3],1):item["position"]=i
        self.assertEqual(restore.compare_metadata(expected,verify.canonical_metadata(changed)),["columns"])
        changed=copy.deepcopy(expected);changed["columns"].pop(1)
        self.assertEqual(restore.compare_metadata(expected,verify.canonical_metadata(changed)),["columns"])
        changed=copy.deepcopy(expected);changed["columns"]=changed["columns"][3:]+changed["columns"][:3]
        self.assertEqual(restore.compare_metadata(expected,verify.canonical_metadata(changed)),["columns"])

    def test_column_ordinals_reject_malformed_or_hidden_reordering(self):
        records=[dict(schema="public",name="t",column="a",position=1),
                 dict(schema="public",name="t",column="b",position=6)]
        for position in (0,-1,True,1.0,"1",None):
            changed=copy.deepcopy(records);changed[0]["position"]=position
            with self.assertRaisesRegex(snapshot.Failure,"column-order"):
                verify.canonical_columns(changed)
        cases=[]
        changed=copy.deepcopy(records);changed[1]["position"]=1;cases.append(changed)
        cases.append(list(reversed(records)))
        changed=copy.deepcopy(records);changed[1]["column"]="a";cases.append(changed)
        changed=copy.deepcopy(records);changed[1]["column"]="";cases.append(changed)
        changed=copy.deepcopy(records);del changed[1]["position"];cases.append(changed)
        changed=copy.deepcopy(records);changed.insert(1,dict(schema="public",name="other",column="a",position=1));cases.append(changed)
        for changed in cases:
            with self.assertRaisesRegex(snapshot.Failure,"column-order"):
                verify.canonical_columns(changed)

    def test_http_domain_exception_is_optional_exact_and_preserves_application_constraints(self):
        # This literal is transcribed independently from the sealed catalog,
        # rather than deriving every positive fixture from the implementation.
        self.assertEqual(verify.HTTP_DOMAIN_CONSTRAINT,dict(schema="public",name="content_type_check",
            relation=None,kind="c",validated=True,deferrable=False,initially_deferred=False,
            definition=r"CHECK ((VALUE ~ '^\S+\/\S+'::text))"))
        manifest=manifest_fixture();manifest["extensions"]=[copy.deepcopy(verify.HTTP_EXTENSION)]
        app=dict(schema="public",name="t_0_positive",relation="t_0",kind="c",validated=True,
                 deferrable=False,initially_deferred=False,definition="CHECK ((id > 0))")
        manifest["schema_metadata"]["constraints"]=[copy.deepcopy(verify.HTTP_DOMAIN_CONSTRAINT),app]
        original=copy.deepcopy(manifest)
        archive=archive_fixture(manifest);before=baseline_fixture();actual=actual_fixture(manifest,archive,before)
        actual["metadata"]["constraints"]=[copy.deepcopy(app)]
        self.assertIn("constraints",verify.compare(manifest,archive,before,actual))
        self.assertIsNone(verify.validate_http_domain_exclusion(None,None,manifest,"a"*64))
        with tempfile.TemporaryDirectory() as directory:
            path,_,_=http_proof_fixture(Path(directory),manifest,"a"*64)
            exception=verify.validate_http_domain_exclusion(path,snapshot.sha256_file(path),manifest,"a"*64)
            self.assertEqual(verify.compare(manifest,archive,before,actual,exception),[])
            for constraints in ([],[dict(app,definition="CHECK ((id >= 0))")],
                                [copy.deepcopy(app),copy.deepcopy(verify.HTTP_DOMAIN_CONSTRAINT)]):
                changed=copy.deepcopy(actual);changed["metadata"]["constraints"]=constraints
                self.assertIn("constraints",verify.compare(manifest,archive,before,changed,exception))
            duplicate=copy.deepcopy(manifest)
            duplicate["schema_metadata"]["constraints"].append(copy.deepcopy(verify.HTTP_DOMAIN_CONSTRAINT))
            with self.assertRaisesRegex(snapshot.Failure,"http-domain-manifest"):
                verify.validate_http_domain_exclusion(path,snapshot.sha256_file(path),duplicate,"a"*64)
        self.assertEqual(manifest,original)

    def test_http_domain_proof_rejects_hash_identity_dependency_and_scope_drift(self):
        manifest=manifest_fixture();manifest["extensions"]=[copy.deepcopy(verify.HTTP_EXTENSION)]
        manifest["schema_metadata"]["constraints"]=[copy.deepcopy(verify.HTTP_DOMAIN_CONSTRAINT)]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path,proof,current=http_proof_fixture(root,manifest,"a"*64)
            digest=snapshot.sha256_file(path)
            for args in ((None,digest),(path,None),(path,"0"*64)):
                with self.assertRaises(snapshot.Failure):
                    verify.validate_http_domain_exclusion(*args,manifest,"a"*64)
            with self.assertRaisesRegex(snapshot.Failure,"http-domain-proof"):
                verify.validate_http_domain_exclusion(path,digest,manifest,"b"*64)
            for change in (lambda p:p.update(extension=dict(verify.HTTP_EXTENSION,version="1.7")),
                           lambda p:p["constraint"].update(relation="t_0"),
                           lambda p:p["constraint"].update(definition="CHECK (true)"),
                           lambda p:p["constraint"].update(validated=1),
                           lambda p:p["current_source_members"].update(file="../source-members-pg-catalog.json")):
                changed=copy.deepcopy(proof);change(changed)
                path.write_text(json.dumps(changed))
                with self.assertRaises(snapshot.Failure):
                    verify.validate_http_domain_exclusion(path,snapshot.sha256_file(path),manifest,"a"*64)
            path.write_text(json.dumps(proof))
            digest=snapshot.sha256_file(path)
            source=root/"source-members-pg-catalog.json"
            changed=copy.deepcopy(current);changed["evidence"]["extension"]["owner"]="postgres"
            source.write_text(json.dumps(changed))
            with self.assertRaisesRegex(snapshot.Failure,"reference-seal"):
                verify.validate_http_domain_exclusion(path,digest,manifest,"a"*64)
            # Resealing bytes cannot authorize a different object or dependency.
            for change in (lambda e:e["extension"].update(owner="postgres"),
                           lambda e:e.update(search_path="public"),
                           lambda e:e.update(project="another_project"),
                           lambda e:e["members"].append(copy.deepcopy(verify.HTTP_DOMAIN_MEMBER)),
                           lambda e:e["members"][0].update(identity="public.application_domain"),
                           lambda e:e["direct_non_extension_owned_dependents"][0].update(dependency_type="n"),
                           lambda e:e["direct_non_extension_owned_dependents"][0].update(referenced_identity="public.application_domain")):
                changed=copy.deepcopy(current);change(changed["evidence"])
                source.write_text(json.dumps(changed))
                changed_proof=copy.deepcopy(proof)
                changed_proof["current_source_members"]["sha256"]=snapshot.sha256_file(source)
                path.write_text(json.dumps(changed_proof))
                with self.assertRaisesRegex(snapshot.Failure,"http-domain-current-proof"):
                    verify.validate_http_domain_exclusion(path,snapshot.sha256_file(path),manifest,"a"*64)

    def test_http_domain_cli_records_private_exclusion_before_comparison(self):
        manifest=manifest_fixture();manifest["extensions"]=[copy.deepcopy(verify.HTTP_EXTENSION)]
        manifest["schema_metadata"]["constraints"]=[copy.deepcopy(verify.HTTP_DOMAIN_CONSTRAINT)]
        archive=archive_fixture(manifest);before=baseline_fixture();actual=actual_fixture(manifest,archive,before)
        actual["metadata"]["constraints"]=[]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name,value in (("manifest.json",manifest),("sequences.json",archive),("before.json",before)):
                snapshot.private_json(root/name,value)
            mh=snapshot.sha256_file(root/"manifest.json")
            snapshot.private_write(root/"manifest.sha256",(mh+"\n").encode())
            proof,_,_=http_proof_fixture(root,manifest,mh)
            args=SimpleNamespace(manifest=str(root/"manifest.json"),manifest_sha256=mh,
                archive_sequences=str(root/"sequences.json"),archive_sequences_sha256=snapshot.sha256_file(root/"sequences.json"),
                initial_target=str(root/"before.json"),initial_target_sha256=snapshot.sha256_file(root/"before.json"),
                directory=str(root/"proof"),http_domain_exclusion_proof=str(proof),
                http_domain_exclusion_proof_sha256=snapshot.sha256_file(proof))
            output=io.StringIO()
            with patch.object(bootstrap,"guard_empty"),patch.object(snapshot,"secret",return_value={}), \
                 patch.object(snapshot,"connection_config",return_value={}),patch.object(snapshot,"service"), \
                 patch.object(snapshot,"Keeper"),patch.object(verify,"capture_target",return_value=actual),contextlib.redirect_stdout(output):
                verify.verify(args)
            self.assertEqual(json.loads(output.getvalue())["status"],"match")
            self.assertNotIn("content_type_check",output.getvalue())
            report=json.loads((root/"proof"/"report.json").read_text())
            self.assertEqual(report["applied_exclusions"][0]["constraint"],verify.HTTP_DOMAIN_CONSTRAINT)
            self.assertEqual(report["applied_exclusions"][0]["proof_sha256"],args.http_domain_exclusion_proof_sha256)
            self.assertEqual(json.loads((root/"proof"/"expected-metadata.json").read_text())["constraints"],[])
            # Bad optional evidence must fail before any target secret is read.
            args.http_domain_exclusion_proof_sha256="0"*64
            with patch.object(snapshot,"secret") as secret:
                with self.assertRaisesRegex(snapshot.Failure,"reference-seal"):
                    verify.verify(args)
                secret.assert_not_called()

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
        self.assertNotIn("database-acl",verify.compare(manifest,archive,null_before,actual))
        # The exception is database-only; source schema/object NULL stays strict.
        changed=copy.deepcopy(actual)
        changed["metadata"]["routines"][0]["acl"]=dict(is_null=False,items=[])
        self.assertIn("routines",verify.compare(manifest,archive,null_before,changed))

    def test_only_initial_null_database_acl_accepts_exact_server_defaults(self):
        manifest=manifest_fixture();archive=archive_fixture(manifest);before=baseline_fixture()
        before["database_acl"]=dict(is_null=True,items=[])
        actual=actual_fixture(manifest,archive,before)
        self.assertEqual(verify.database_acl_match(before,actual),(True,None))
        actual["state"]["database_acl"]=copy.deepcopy(actual["database_acl_default"]["acl"])
        matched,equivalence=verify.database_acl_match(before,actual)
        self.assertTrue(matched)
        self.assertEqual(equivalence["rule"],"initial-null-database-acl-equals-server-default")
        self.assertEqual(equivalence["owner"],"arena_admin")
        for entries in (["=CTc/arena_admin","arena_admin=CTc/arena_admin"],
                        ["=Tc/postgres","arena_admin=CTc/arena_admin"],
                        ["=c/arena_admin","arena_admin=CTc/arena_admin"],
                        ["=T*c/arena_admin","arena_admin=CTc/arena_admin"]):
            changed=copy.deepcopy(actual);changed["state"]["database_acl"]["items"]=entries
            self.assertEqual(verify.database_acl_match(before,changed),(False,None))
        for key in ("state","database_acl_default"):
            changed=copy.deepcopy(actual);changed[key]["owner"]="postgres"
            self.assertEqual(verify.database_acl_match(before,changed),(False,None))
        changed=copy.deepcopy(actual);changed["state"]["database_acl"]=dict(is_null=True,items=[])
        changed["state"]["owner"]="postgres"
        self.assertEqual(verify.database_acl_match(before,changed),(False,None))
        explicit=copy.deepcopy(before);explicit["database_acl"]=dict(is_null=False,items=["arena_admin=CTc/arena_admin"])
        self.assertEqual(verify.database_acl_match(explicit,actual),(False,None))

    def test_every_category_and_unreviewed_support_or_sequence_delta_fails(self):
        manifest=manifest_fixture();archive=archive_fixture(manifest);before=baseline_fixture()
        actual=actual_fixture(manifest,archive,before)
        for category in snapshot.OBJECT_KINDS:
            changed=copy.deepcopy(actual)
            changed["metadata"][category].append(dict(schema="public",name="unreviewed"))
            if category=="columns":
                with self.assertRaisesRegex(snapshot.Failure,"column-order"):
                    verify.compare(manifest,archive,before,changed)
            else:
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

    def test_nonowner_zero_or_one_safe_creator_grant_and_owner_authority(self):
        manifest=manifest_fixture();before=baseline_fixture()
        state=actual_fixture(manifest,archive_fixture(manifest),before)["state"]
        state["memberships"]=[m for m in state["memberships"] if m["name"] in bootstrap.OWNERS]
        self.assertTrue(verify.target_roles_match(state))
        grant=dict(name="anon",member="arena_admin",grantor="arena_admin",admin=True,inherit=False,set=False)
        state["memberships"].append(grant)
        self.assertTrue(verify.target_roles_match(state))
        duplicate=copy.deepcopy(state);duplicate["memberships"].append(copy.deepcopy(grant))
        self.assertFalse(verify.target_roles_match(duplicate))
        for field,bad in (("member","authenticated"),("admin",False),("inherit",True),("set",True)):
            changed=copy.deepcopy(state);changed["memberships"][-1][field]=bad
            self.assertFalse(verify.target_roles_match(changed))
        for owner in bootstrap.OWNERS:
            changed=copy.deepcopy(state)
            changed["memberships"]=[m for m in changed["memberships"] if m["name"]!=owner]
            self.assertFalse(verify.target_roles_match(changed))

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
        before["database_acl"]=dict(is_null=True,items=[])
        actual=actual_fixture(manifest,archive,before)
        actual["state"]["database_acl"]=copy.deepcopy(actual["database_acl_default"]["acl"])
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
            report=json.loads((root/"proof"/"report.json").read_text())
            self.assertEqual(report["applied_equivalences"][0]["rule"],"initial-null-database-acl-equals-server-default")
            self.assertEqual(report["canonicalizations"],["column-position-logical-ordinal-per-relation"])
            self.assertNotIn("applied_equivalences",json.loads(output.getvalue()))
        output=io.StringIO()
        with patch.object(verify,"verify",side_effect=RuntimeError("private function definition password")),contextlib.redirect_stderr(output):
            code=verify.main(["--manifest","x","--manifest-sha256","x","--archive-sequences","x", "--archive-sequences-sha256","x","--initial-target","x","--initial-target-sha256","x","--directory","x"])
        self.assertEqual(code,1)
        self.assertEqual(output.getvalue(),"ERROR phase=metadata-verification\n")


PG17=Path("/opt/homebrew/opt/postgresql@17/bin")


@unittest.skipUnless(os.environ.get("RUN_AWS_METADATA_LOCAL_PG17")=="1" and PG17.is_dir(),"opt-in disposable local PG17 proof")
class NativeMetadataTests(unittest.TestCase):
    def test_native_dump_restore_compacts_dropped_column_holes_only(self):
        with tempfile.TemporaryDirectory(prefix="lp-column-order-",dir="/tmp") as directory:
            root=Path(directory)
            env={k:v for k,v in os.environ.items() if not k.startswith("PG")}
            def run(args,data=None):
                return subprocess.run(args,input=data,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                    env=env,check=True,timeout=30).stdout
            with socket.socket() as sock:
                sock.bind(("127.0.0.1",0));port=sock.getsockname()[1]
            run([str(PG17/"initdb"),"-D",str(root/"db"),"-U","local_bootstrap","--auth=trust","--no-locale","--encoding=UTF8"])
            run([str(PG17/"pg_ctl"),"-D",str(root/"db"),"-l",str(root/"server.log"),"-o",f"-F -h 127.0.0.1 -p {port} -k {root}","-w","start"])
            connection=["-h","127.0.0.1","-p",str(port),"-U","local_bootstrap"]
            def sql(database,statement):
                return run([str(PG17/"psql"),"-XqAt","-v","ON_ERROR_STOP=1",*connection,"-d",database],statement.encode())
            try:
                sql("postgres","CREATE DATABASE column_source; CREATE DATABASE column_target; CREATE ROLE column_reader NOLOGIN;")
                sql("column_source","""CREATE TABLE public.ordinal_four(id integer PRIMARY KEY,
                    kept_first text DEFAULT 'source',d1 text,d2 text,d3 text,d4 text,
                    kept_last integer NOT NULL DEFAULT 7,generated_id integer GENERATED ALWAYS AS (id+1) STORED);
                    ALTER TABLE public.ordinal_four DROP COLUMN d1,DROP COLUMN d2,DROP COLUMN d3,DROP COLUMN d4;
                    GRANT SELECT (kept_last) ON public.ordinal_four TO column_reader;
                    CREATE TABLE public.ordinal_multiple(a integer,d1 text,b text,d2 text,d3 text,c integer,d4 text);
                    ALTER TABLE public.ordinal_multiple DROP COLUMN d1,DROP COLUMN d2,DROP COLUMN d3,DROP COLUMN d4;""")
                archive=root/"schema.dump"
                run([str(PG17/"pg_dump"),*connection,"--no-password","--schema-only","--format=custom","--file="+str(archive),"column_source"])
                run([str(PG17/"pg_restore"),*connection,"--no-password","--exit-on-error","--dbname=column_target",str(archive)])
                def capture(database):
                    return json.loads(sql(database,snapshot.METADATA_QUERIES["columns"]))
                source=capture("column_source");target=capture("column_target")
                self.assertEqual([c["position"] for c in source],[1,2,7,8,1,3,6])
                self.assertEqual([c["position"] for c in target],[1,2,3,4,1,2,3])
                original=copy.deepcopy(source)
                self.assertEqual(verify.canonical_columns(source),verify.canonical_columns(target))
                self.assertEqual(source,original)
                for change,undo in (
                    ("ALTER TABLE public.ordinal_four ALTER COLUMN kept_first SET DEFAULT 'changed';",
                     "ALTER TABLE public.ordinal_four ALTER COLUMN kept_first SET DEFAULT 'source';"),
                    ("ALTER TABLE public.ordinal_four ALTER COLUMN kept_first TYPE varchar(40);",
                     "ALTER TABLE public.ordinal_four ALTER COLUMN kept_first TYPE text;"),
                    ("REVOKE SELECT (kept_last) ON public.ordinal_four FROM column_reader;",
                     "GRANT SELECT (kept_last) ON public.ordinal_four TO column_reader;")):
                    sql("column_target",change)
                    self.assertNotEqual(verify.canonical_columns(source),verify.canonical_columns(capture("column_target")))
                    sql("column_target",undo)
                    self.assertEqual(verify.canonical_columns(source),verify.canonical_columns(capture("column_target")))
            finally:
                run([str(PG17/"pg_ctl"),"-D",str(root/"db"),"-m","fast","-w","stop"])

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
                def read_env(user):
                    path=root/(user+".service")
                    snapshot.private_write(path,(f"[migration]\nhost=127.0.0.1\nport={port}\nuser={user}\ndbname=postgres\nsslmode=disable\noptions={snapshot.OPTIONS}\n").encode())
                    return {**env,"PGSERVICEFILE":str(path)}
                target_env=read_env("arena_admin");source_env=read_env("local_bootstrap")
                with snapshot.Keeper(target_env,logs) as keeper:
                    before=keeper.one_json(bootstrap.STATE_SQL)
                bootstrap.guard_empty(before)
                self.assertEqual(before["database_acl"],dict(is_null=True,items=[]))
                bootstrap_manifest=manifest_fixture()
                bootstrap_manifest["schema_metadata"]["schemas"]=[dict(schema="public",name="public",owner="pg_database_owner",acl=dict(is_null=False,items=["=U/pg_database_owner","pg_database_owner=UC/pg_database_owner"])),dict(schema="extensions",name="extensions",owner="postgres",acl=dict(is_null=True,items=[]))]
                # Execute the unchanged, reviewed bootstrap and its real cleanup.
                run(master,bootstrap.build_sql(before,bootstrap_manifest).encode())
                sql="GRANT CREATE ON DATABASE postgres TO lab_arena_owner; CREATE SCHEMA source_add_history AUTHORIZATION lab_arena_owner; REVOKE CREATE ON DATABASE postgres FROM lab_arena_owner;"
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
                run(master,bootstrap.cleanup_sql(manifest).encode())
                with snapshot.Keeper(target_env,logs) as keeper:actual=verify.capture_target(keeper)
                self.assertEqual(len(actual["metadata"]["routines"]),383)
                self.assertEqual(verify.compare(manifest,archive,before,actual),[])
                self.assertFalse(actual["state"]["database_acl"]["is_null"])
                self.assertEqual(actual["state"]["database_acl"],actual["database_acl_default"]["acl"])
                self.assertEqual(verify.database_acl_match(before,actual)[1]["rule"],"initial-null-database-acl-equals-server-default")
                # Reproduce the catalog gap with the bundled plpgsql extension:
                # its member domain is excluded, but the auto-dependent CHECK
                # is not itself an extension member and remains in capture.
                # No HTTP installation or application function call is needed.
                run(admin,br"CREATE DOMAIN public.content_type AS text CHECK (VALUE ~ '^\S+\/\S+'::text); ALTER EXTENSION plpgsql ADD DOMAIN public.content_type;")
                with snapshot.Keeper(source_env,logs) as keeper:
                    domain_constraints=keeper.one_json(snapshot.METADATA_QUERIES["constraints"])
                    captured_types=keeper.one_json(snapshot.METADATA_QUERIES["types"])
                    domain_dependency=keeper.one_json("SELECT json_build_object('dependency',d.deptype,'extension',e.extname) FROM pg_catalog.pg_constraint c JOIN pg_catalog.pg_depend d ON d.classid='pg_catalog.pg_constraint'::regclass AND d.objid=c.oid JOIN pg_catalog.pg_depend t ON t.classid=d.refclassid AND t.objid=d.refobjid AND t.deptype='e' AND t.refclassid='pg_catalog.pg_extension'::regclass JOIN pg_catalog.pg_extension e ON e.oid=t.refobjid WHERE c.conname='content_type_check';")
                self.assertEqual([c for c in domain_constraints if c["name"]=="content_type_check"],
                                 [verify.HTTP_DOMAIN_CONSTRAINT])
                self.assertFalse(any(t["name"]=="content_type" for t in captured_types))
                self.assertEqual(domain_dependency,dict(dependency="a",extension="plpgsql"))
                run(admin,b"ALTER EXTENSION plpgsql DROP DOMAIN public.content_type; DROP DOMAIN public.content_type;")
                http_manifest=copy.deepcopy(manifest)
                http_manifest["extensions"]=[copy.deepcopy(verify.HTTP_EXTENSION)]
                http_manifest["schema_metadata"]["constraints"]=domain_constraints
                self.assertIn("constraints",verify.compare(http_manifest,archive,before,actual))
                proof_dir=root/"http-proof";proof_dir.mkdir(mode=0o700)
                proof,_,_=http_proof_fixture(proof_dir,http_manifest,"a"*64)
                exclusion=verify.validate_http_domain_exclusion(proof,snapshot.sha256_file(proof),http_manifest,"a"*64)
                self.assertEqual(verify.compare(http_manifest,archive,before,actual,exclusion),[])
                app_constraint_drift=copy.deepcopy(actual)
                app_constraint_drift["metadata"]["constraints"].pop()
                self.assertIn("constraints",verify.compare(http_manifest,archive,before,app_constraint_drift,exclusion))
                # Native non-default ACLs cannot use the NULL-baseline exception.
                for grant,revoke in ((b"GRANT CREATE ON DATABASE postgres TO PUBLIC;",b"REVOKE CREATE ON DATABASE postgres FROM PUBLIC;"),
                                     (b"REVOKE TEMPORARY ON DATABASE postgres FROM PUBLIC;",b"GRANT TEMPORARY ON DATABASE postgres TO PUBLIC;"),
                                     (b"GRANT CONNECT ON DATABASE postgres TO anon WITH GRANT OPTION;",b"REVOKE CONNECT ON DATABASE postgres FROM anon;")):
                    run(master,grant)
                    with snapshot.Keeper(target_env,logs) as keeper:acl_drift=verify.capture_target(keeper)
                    self.assertIn("database-acl",verify.compare(manifest,archive,before,acl_drift))
                    run(master,revoke)
                run(admin,b"ALTER DATABASE postgres OWNER TO postgres;")
                with snapshot.Keeper(target_env,logs) as keeper:owner_drift=verify.capture_target(keeper)
                self.assertIn("database-acl",verify.compare(manifest,archive,before,owner_drift))
                run(admin,b"ALTER DATABASE postgres OWNER TO arena_admin;")
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
