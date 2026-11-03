#!/usr/bin/env python3
"""Read-only restore inventory and local plan preparation. No execution mode.

Run inventory under the snapshot utility's mandatory external systemd deadline
and private root-only log controls. Plan consumes private local evidence only.
Native restore and target role/ACL writes require separate review and execution.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import sys

if __package__:
    from . import aws_subnet_snapshot as snapshot
else:
    import aws_subnet_snapshot as snapshot


ROLES = ("anon", "authenticated", "dashboard_user", "lab_arena_owner", "lab_arena_service",
         "leadpoet_parity_reader", "pg_database_owner", "postgres", "service_role",
         "supabase_admin", "supabase_auth_admin", snapshot.SOURCE_ROLE)
NOINHERIT = {"lab_arena_owner", "lab_arena_service", "leadpoet_parity_reader", snapshot.SOURCE_ROLE}
OWNER_ROLES = {"postgres", "lab_arena_owner", "supabase_admin", "supabase_auth_admin"}
OBJECT_KINDS = snapshot.OBJECT_KINDS
AUTH_ROLE_SHA256 = "c96a9358b02089b03f3962fd85c5f83c9d64f32240481915c0efe74398dc761b"
# One reviewed immutable archive and supplemental CURRENT catalog proof. These
# exceptions do not replace the sealed snapshot's ordinary-object metadata.
HTTP_ACL_REVIEW_SHA256 = {
    "toc": "de56bb235302015478bf7bd1b9c28f79af7f8b8f77b19b81d023ecb3c737d8c9",
    "source_proof": "da10e68a4ca65caf03810c50e4d97ae2cfb9373dd45784109c75cedc1dbebd32",
    "acl_sql": "aaec2549c6ae2165873a840aaf096c99f1ff8a0e7acf42da38f49aad8cbbc4e2",
    "manifest": "d93ddef1740ecfa1bb74a186f36c918ca731e4973cfd756fa69f9a62d8138f56",
}


def target_evidence(keeper):
    snapshot.check_identity(keeper, "target")
    metadata = snapshot.capture_metadata(keeper)
    roles = keeper.one_json(snapshot.aggregate("""SELECT r.rolname AS name,r.rolsuper AS super,r.rolinherit AS inherit,
      r.rolcreaterole AS createrole,r.rolcreatedb AS createdb,r.rolcanlogin AS login,
      r.rolreplication AS replication,r.rolbypassrls AS bypassrls,r.rolconfig AS config,
      pg_has_role(current_user,r.oid,'SET') AS can_set,
      pg_has_role(current_user,r.oid,'USAGE') AS can_inherit
      FROM pg_catalog.pg_roles r WHERE r.rolname IN (""" + ",".join("'" + r + "'" for r in ROLES) + ")", "t.name"))
    memberships = keeper.one_json(snapshot.aggregate("""SELECT pg_get_userbyid(roleid) AS name,
      pg_get_userbyid(member) AS member,pg_get_userbyid(grantor) AS grantor,
      admin_option AS admin,inherit_option AS inherit,set_option AS set
      FROM pg_catalog.pg_auth_members""", "t.name,t.member,t.grantor"))
    available = keeper.one_json(snapshot.aggregate("SELECT name,default_version,installed_version FROM pg_catalog.pg_available_extensions", "t.name"))
    installed = keeper.one_json(snapshot.aggregate("""SELECT e.extname AS name,e.extversion AS version,
      n.nspname AS schema,pg_get_userbyid(e.extowner) AS owner FROM pg_catalog.pg_extension e
      JOIN pg_catalog.pg_namespace n ON n.oid=e.extnamespace""", "t.name"))
    control = keeper.one_json("""SELECT json_build_object('host',inet_server_addr()::text,
      'role',current_user,'database',current_database(),'version',current_setting('server_version_num')::int,
      'recovery',pg_is_in_recovery(),'other_sessions',(SELECT count(*) FROM pg_catalog.pg_stat_activity
       WHERE datname=current_database() AND pid<>pg_backend_pid()),
      'preload',(SELECT setting FROM pg_catalog.pg_settings WHERE name='shared_preload_libraries'),
      'unexpected_schemas',ARRAY(SELECT n.nspname FROM pg_catalog.pg_namespace n
       WHERE n.nspname NOT IN ('public','source_add_history','extensions','information_schema')
       AND n.nspname !~ '^pg_' AND NOT EXISTS(SELECT 1 FROM pg_catalog.pg_depend d
        WHERE d.classid='pg_catalog.pg_namespace'::regclass AND d.objid=n.oid AND d.deptype='e')
       ORDER BY n.nspname));""")
    sequences = []
    for item in metadata["sequence_definitions"]:
        state = keeper.one_json("SELECT json_build_object('last_value',last_value::text,'is_called',is_called) FROM " + snapshot.qualified(item) + ";")
        sequences.append(dict(schema=item["schema"], name=item["name"], **state))
    return dict(format=1, host=snapshot.TARGET_HOST, captured_at=datetime.now(timezone.utc).isoformat(),
                control=control, metadata=metadata, roles=roles, memberships=memberships,
                available_extensions=available, installed_extensions=installed, sequences=sequences)


def target_blockers(evidence):
    snapshot.require(evidence["format"] == 1 and evidence["host"] == snapshot.TARGET_HOST, "target-evidence")
    control = evidence["control"]
    snapshot.require(control["role"] == "arena_admin" and control["database"] == "postgres"
                     and 170000 <= control["version"] < 180000 and not control["recovery"], "target-identity")
    blockers = []
    if control["other_sessions"] != 0:
        blockers.append("target-has-other-sessions")
    if control["unexpected_schemas"]:
        blockers.append("target-has-unexpected-user-schemas")
    metadata = evidence["metadata"]
    if any(metadata[key] for key in OBJECT_KINDS if key not in ("schemas", "default_acls")) or metadata["default_acls"]:
        blockers.append("target-has-user-objects-or-default-acls")
    roles = {r["name"]: r for r in evidence["roles"]}
    for name in ROLES:
        role = roles.get(name)
        if role is None:
            blockers.append("missing-role:" + name)
            continue
        if (any(role[k] for k in ("super", "createrole", "createdb", "login", "replication", "bypassrls"))
                or role["inherit"] != (name not in NOINHERIT) or role["config"]):
            blockers.append("unsafe-or-unreviewed-role:" + name)
        if name in OWNER_ROLES and (not role["can_set"] or not role["can_inherit"]):
            blockers.append("restore-owner-authority:" + name)
    creator = lambda m: (m["name"] == snapshot.SOURCE_ROLE and m["member"] == "arena_admin"
                         and m.get("admin") is True and m.get("inherit") is False and m.get("set") is False)
    temporary_memberships = [m for m in evidence["memberships"] if snapshot.SOURCE_ROLE in (m["name"], m["member"])]
    if len(temporary_memberships) > 1 or any(not creator(m) for m in temporary_memberships):
        blockers.append("temporary-role-membership")
    available = {e["name"] for e in evidence["available_extensions"]}
    blockers.extend("unavailable-extension:" + e for e in snapshot.EXTENSIONS if e not in available)
    return blockers


def toc_entries(toc):
    result = {}
    for line in toc.decode("utf-8").splitlines():
        if not line or line.startswith(";"):
            continue
        match = re.fullmatch(r"([0-9]+); [0-9]+ [0-9]+ (.+)", line)
        snapshot.require(match is not None and match[1] not in result, "restore-toc")
        snapshot.require(not re.match(r"(?:EVENT TRIGGER |(?:COMMENT|SECURITY LABEL) - EVENT TRIGGER )", match[2]),
                         "event-trigger-review-required")
        result[match[1]] = line
    return result


def validate_toc_inventory(entries, metadata):
    """Compare every selected routine/view name and kind with sealed metadata."""
    expected = Counter()
    for category in ("routines", "views"):
        records = metadata.get(category)
        snapshot.require(isinstance(records, list), "manifest-object-inventory")
        identities = set()
        for item in records:
            snapshot.require(isinstance(item, dict) and item.get("schema") in snapshot.SCHEMAS,
                             "manifest-object-inventory")
            snapshot.ident(item["schema"])
            snapshot.ident(item["name"])
            if category == "routines":
                snapshot.require(item.get("kind") in ("f", "p")
                                 and isinstance(item.get("arguments"), str), "manifest-routine-kind")
                kind = "FUNCTION" if item["kind"] == "f" else "PROCEDURE"
                identity = (item["schema"], item["name"], item["kind"], item["arguments"])
            else:
                kind = "VIEW"
                identity = (item["schema"], item["name"])
            snapshot.require(identity not in identities, "duplicate-manifest-object")
            identities.add(identity)
            expected[kind, item["schema"], item["name"]] += 1
    actual = Counter()
    counts = Counter()
    for line in entries.values():
        tail = re.sub(r"^[0-9]+; [0-9]+ [0-9]+ ", "", line)
        match = re.match(r"(TABLE DATA|SEQUENCE SET|VIEW|FUNCTION|PROCEDURE) ", tail)
        if not match:
            continue
        kind = match[1]
        counts[kind] += 1
        if kind in ("VIEW", "FUNCTION", "PROCEDURE"):
            obj = re.match(r"(?:VIEW|FUNCTION|PROCEDURE) ([a-z_][a-z0-9_]*) ([a-z_][a-z0-9_]*)(?:\(| )", tail)
            snapshot.require(obj is not None, "archive-object-identity")
            actual[kind, obj[1], obj[2]] += 1
    snapshot.require(counts["TABLE DATA"] == 48 and counts["SEQUENCE SET"] == 3
                     and actual == expected, "archive-object-inventory")
    return {kind: counts[kind] for kind in ("TABLE DATA", "SEQUENCE SET", "VIEW", "FUNCTION", "PROCEDURE")}


def validate_supporting_auth_role(value):
    """Require the exact audited claim getter, never a replacement Auth API."""
    snapshot.require(isinstance(value, dict) and set(value) == {"schema", "function", "dependencies"},
                     "supporting-auth-role")
    schema, function = value["schema"], value["function"]
    snapshot.require(isinstance(schema, dict) and set(schema) == {"name", "owner", "acl"}
                     and schema.get("name") == "auth" and schema.get("owner") == "supabase_admin"
                     and isinstance(schema.get("acl"), str), "supporting-auth-schema")
    required = dict(schema="auth", name="role", arguments="", result="text", owner="supabase_auth_admin",
                    language="sql", volatility="s", parallel="u", config=None)
    snapshot.require(isinstance(function, dict) and set(function) == set(required) | {
        "acl", "security_definer", "leakproof", "strict", "definition", "sha256"}, "supporting-auth-role")
    snapshot.require(all(function.get(key) == expected for key, expected in required.items())
                     and all(function.get(key) is False for key in ("security_definer", "leakproof", "strict"))
                     and isinstance(function.get("acl"), str), "supporting-auth-role-identity")
    definition = function.get("definition")
    snapshot.require(isinstance(definition, str) and function.get("sha256") == AUTH_ROLE_SHA256
                     and hashlib.sha256(definition.encode()).hexdigest() == AUTH_ROLE_SHA256,
                     "supporting-auth-role-definition")
    snapshot.require(value["dependencies"] == [dict(identity="auth", kind="n", name="auth", schema=None, type="schema")],
                     "supporting-auth-role-dependencies")
    return value


def source_public_schema(schema_metadata):
    """Read the sealed source owner; only the two reviewed owners are valid."""
    snapshot.require(isinstance(schema_metadata, dict)
                     and isinstance(schema_metadata.get("schemas"), list), "source-public-schema")
    public = [item for item in schema_metadata["schemas"] if isinstance(item, dict)
              and (item.get("schema") == "public" or item.get("name") == "public")]
    snapshot.require(len(public) == 1 and public[0].get("schema") == "public"
                     and public[0].get("name") == "public"
                     and public[0].get("owner") in ("postgres", "pg_database_owner"), "source-public-schema")
    return public[0]


def source_extensions_schema(schema_metadata):
    """Bind the precreated schema needed for native helper-owner restoration."""
    snapshot.require(isinstance(schema_metadata, dict)
                     and isinstance(schema_metadata.get("schemas"), list), "source-extensions-schema")
    records = [item for item in schema_metadata["schemas"] if isinstance(item, dict)
               and (item.get("schema") == "extensions" or item.get("name") == "extensions")]
    snapshot.require(len(records) == 1 and records[0].get("schema") == "extensions"
                     and records[0].get("name") == "extensions"
                     and records[0].get("owner") == "postgres", "source-extensions-schema")
    return records[0]


def reviewed_http_acls(toc, schema_metadata, review):
    """Join exact archive ACL fragments to the pinned HTTP-only catalog proof."""
    snapshot.require(isinstance(review, dict) and set(review) == {
        "source_proof", "acl_sql", "manifest_sha256"}, "http-acl-review")
    for key, value in (("toc", toc), ("source_proof", review["source_proof"]),
                       ("acl_sql", review["acl_sql"])):
        snapshot.require(isinstance(value, bytes)
                         and hashlib.sha256(value).hexdigest() == HTTP_ACL_REVIEW_SHA256[key],
                         "http-acl-review-seal")
    snapshot.require(review["manifest_sha256"] == HTTP_ACL_REVIEW_SHA256["manifest"], "http-acl-manifest")
    proof = json.loads(review["source_proof"])
    evidence = proof["evidence"]
    snapshot.require(proof["capture"]["sealed_manifest_sha256"] == review["manifest_sha256"]
                     and proof["capture"]["sealed_manifest_unchanged"] is True
                     and evidence["evidence_kind"] == "supplemental-current-source-catalog-not-sealed-snapshot"
                     and evidence["transaction_read_only"] == "on" and evidence["search_path"] == "pg_catalog"
                     and evidence["project"] == "qplwoislplkcegvdmbim"
                     and evidence["extension"] == dict(name="http", schema="public", owner="supabase_admin", version="1.6"),
                     "http-acl-membership-proof")
    members = [m for m in evidence["members"] if m["class"] == "pg_proc"]
    snapshot.require(len(members) == 19, "http-acl-member-count")
    snapshot.require(isinstance(schema_metadata, dict) and isinstance(schema_metadata.get("routines"), list),
                     "http-acl-ordinary-inventory")
    entries, records = toc_entries(toc), {}
    sql = review["acl_sql"].decode("utf-8")
    for member in members:
        snapshot.require(member["kind"] == "function" and member["schema"] == "public"
                         and member["owner"] == "supabase_admin", "http-acl-member-identity")
        snapshot.ident(member["name"])
        # Reject even a same-name ordinary overload: it is not an extension ACL.
        snapshot.require(not any(r["schema"] == "public" and r["name"] == member["name"]
                                 for r in schema_metadata["routines"]), "http-acl-ordinary-collision")
        identity = member["name"] + "(" + member["identity_arguments"] + ")"
        suffix = "ACL public FUNCTION " + identity + " supabase_admin"
        matches = [(key, line) for key, line in entries.items()
                   if re.sub(r"^[0-9]+; [0-9]+ [0-9]+ ", "", line) == suffix]
        snapshot.require(len(matches) == 1, "http-acl-archive-identity")
        key, line = matches[0]
        snapshot.require(key not in records, "http-acl-duplicate-member")
        marker = "-- Name: FUNCTION " + identity + "; Type: ACL; Schema: public; Owner: supabase_admin\n--\n\n"
        snapshot.require(sql.count(marker) == 1, "http-acl-fragment")
        fragment = sql.split(marker, 1)[1].split("\n\n", 1)[0] + "\n"
        expected = "".join("GRANT ALL ON FUNCTION public." + identity + " TO " + role + ";\n"
                           for role in ("postgres", "anon", "authenticated", "service_role"))
        snapshot.require(fragment == expected, "http-acl-mixed-or-changed-fragment")
        records[key] = dict(id=key, sha256=hashlib.sha256(line.encode()).hexdigest(),
                            acl_sql_sha256=hashlib.sha256(fragment.encode()).hexdigest(),
                            identity="public." + identity)
    return records


def select_toc(toc, exclusions, schema_metadata=None, http_acl_review=None):
    """Omit reviewed platform entries; all ordinary object ACLs stay intact."""
    entries = toc_entries(toc)
    http_acls = reviewed_http_acls(toc, schema_metadata, http_acl_review) if http_acl_review is not None else {}
    excluded = {}
    for item in exclusions:
        key = str(item["id"])
        snapshot.require(key in entries and key not in excluded and isinstance(item["reason"], str)
                         and item["reason"].strip(), "toc-exclusion")
        line = entries[key]
        snapshot.require(hashlib.sha256(line.encode()).hexdigest() == item["sha256"], "toc-exclusion-seal")
        tail = re.sub(r"^[0-9]+; [0-9]+ [0-9]+ ", "", line)
        public = re.fullmatch(r"SCHEMA - public (postgres|pg_database_owner)\s*", tail)
        if public:
            snapshot.require(public[1] == source_public_schema(schema_metadata)["owner"], "public-schema-owner-binding")
        elif re.fullmatch(r"SCHEMA - extensions postgres\s*", tail):
            source_extensions_schema(schema_metadata)
        elif key in http_acls:
            snapshot.require(item["sha256"] == http_acls[key]["sha256"], "http-acl-exclusion-seal")
        else:
            snapshot.require(re.fullmatch(r"(?:EXTENSION - http|COMMENT - EXTENSION http)\s*", tail), "protected-toc-exclusion")
        excluded[key] = item
    if http_acls:
        snapshot.require(set(http_acls) <= set(excluded) and any(
            re.fullmatch(r"[0-9]+; [0-9]+ [0-9]+ EXTENSION - http\s*", entries[key])
            for key in excluded), "http-acl-complete-exclusions")
    selected = "\n".join(("; excluded " + line if line.split(";", 1)[0] in excluded else line)
                         for line in toc.decode("utf-8").splitlines()) + "\n"
    return selected.encode(), entries


def sequence_states(toc, archive, env, logs, output):
    lines = [line for line in toc.decode().splitlines() if re.match(r"^[0-9]+; [0-9]+ [0-9]+ SEQUENCE SET ", line)]
    snapshot.require(len(lines) == 3, "sequence-toc")
    listing = output / "sequence-only.list"
    snapshot.private_write(listing, ("\n".join(lines) + "\n").encode())
    sql = snapshot.command(["pg_restore", "--data-only", "--use-list=" + str(listing), "--file=-", str(archive)], env, logs, "sequence-sql")
    snapshot.private_write(output / "archive-sequence.sql", sql)
    states = []
    pattern = r"SELECT pg_catalog\.setval\('([a-z_][a-z0-9_]*)\.([a-z_][a-z0-9_]*)', (-?[0-9]+), (true|false)\);"
    for match in re.finditer(pattern, sql.decode("utf-8"), re.ASCII):
        schema, name, value, called = match.groups()
        snapshot.qualified(dict(schema=schema, name=name))
        states.append(dict(schema=schema, name=name, last_value=value, is_called=called == "true"))
    snapshot.require(len(states) == 3 and len({(s["schema"], s["name"]) for s in states}) == 3, "archive-sequence-state")
    return sorted(states, key=lambda s: (s["schema"], s["name"]))


def export_role_cleanup(manifest):
    snapshot.validate_tables(manifest["tables"])
    snapshot.require(len(manifest["sequences"]) == 3, "sequence-inventory")
    role = snapshot.ident(snapshot.SOURCE_ROLE)
    return ("-- Review first: role is inert, owns nothing, and has at most its\n"
            "-- arena_admin creator-ADMIN membership. Its only object privileges\n"
            "-- are the exact source-export grants below.\n"
            "-- Never use DROP OWNED or discard mixed ACL TOC entries.\nBEGIN;\n"
            "DO $guard$ BEGIN\n"
            " IF NOT EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname='" + snapshot.SOURCE_ROLE
            + "' AND NOT rolcanlogin AND NOT rolsuper AND NOT rolbypassrls AND NOT rolcreaterole"
            " AND NOT rolcreatedb AND NOT rolreplication AND NOT rolinherit) THEN\n"
            "  RAISE EXCEPTION 'temporary role is not inert'; END IF;\n"
            " IF EXISTS (SELECT 1 FROM pg_catalog.pg_auth_members WHERE (roleid='" + snapshot.SOURCE_ROLE
            + "'::regrole OR member='" + snapshot.SOURCE_ROLE + "'::regrole) AND NOT (roleid='"
            + snapshot.SOURCE_ROLE + "'::regrole AND member='arena_admin'::regrole AND admin_option"
            " AND NOT inherit_option AND NOT set_option)) THEN\n"
            "  RAISE EXCEPTION 'temporary role has memberships'; END IF;\n"
            " IF (SELECT count(*) FROM pg_catalog.pg_auth_members WHERE roleid='" + snapshot.SOURCE_ROLE
            + "'::regrole OR member='" + snapshot.SOURCE_ROLE + "'::regrole)>1 THEN\n"
            "  RAISE EXCEPTION 'temporary role has multiple memberships'; END IF;\n"
            " IF EXISTS (SELECT 1 FROM pg_catalog.pg_shdepend WHERE refclassid='pg_catalog.pg_authid'::regclass"
            " AND refobjid='" + snapshot.SOURCE_ROLE + "'::regrole AND deptype='o') THEN\n"
            "  RAISE EXCEPTION 'temporary role owns objects'; END IF;\nEND $guard$;\n"
            "REVOKE SELECT ON TABLE " + ",".join(snapshot.qualified(t) for t in manifest["tables"]) + " FROM " + role + ";\n"
            "REVOKE SELECT ON SEQUENCE " + ",".join(snapshot.qualified(s) for s in manifest["sequences"]) + " FROM " + role + ";\n"
            "REVOKE USAGE ON SCHEMA " + ",".join(snapshot.ident(s) for s in snapshot.SCHEMAS) + " FROM " + role + ";\n"
            "DROP ROLE " + role + ";\nCOMMIT;\n")


def native_restore_command(listing, archive):
    # Candidate only. The read-only service used by this tool cannot run it.
    return ["pg_restore", "--dbname=service=migration", "--no-password", "--exit-on-error",
            "--single-transaction", "--use-list=" + str(listing), str(archive)]


def compare_metadata(expected, actual):
    """Return category names only; definitions, ACLs and SQL stay private."""
    return [key for key in OBJECT_KINDS if expected[key] != actual[key]]


def inventory(args):
    output = Path(args.output).absolute()
    output.mkdir(mode=0o700)
    logs = output / "private-logs"
    logs.mkdir(mode=0o700)
    config = snapshot.connection_config("target", snapshot.secret(args.target_secret_arn), snapshot.TARGET_HOST)
    with snapshot.service(config) as env, snapshot.Keeper(env, logs) as keeper:
        evidence = target_evidence(keeper)
    snapshot.private_json(output / "target.json", evidence)
    snapshot.private_write(output / "target.sha256", (snapshot.sha256_file(output / "target.json") + "\n").encode())
    print("target inventory saved privately; no target writes")


def plan(args):
    root = Path(args.export_directory).absolute()
    output = Path(args.output).absolute()
    output.mkdir(mode=0o700)
    logs = output / "private-logs"
    logs.mkdir(mode=0o700)
    snapshot.require(snapshot.sha256_file(root / "manifest.json") == (root / "manifest.sha256").read_text().strip(), "manifest-seal")
    manifest = json.loads((root / "manifest.json").read_text())
    snapshot.require(manifest["format"] == 1 and manifest["source_host"] == snapshot.SOURCE_HOST
                     and manifest["source_role"] == snapshot.SOURCE_ROLE and manifest["database"] == "postgres", "manifest-identity")
    snapshot.validate_tables(manifest["tables"])
    snapshot.require(isinstance(manifest.get("schema_metadata"), dict)
                     and set(manifest["schema_metadata"]) == set(OBJECT_KINDS), "source-metadata-reference")
    snapshot.require(any(t["schema"] == "source_add_history" for t in manifest["tables"]), "protected-history-missing")
    supporting_auth = validate_supporting_auth_role(manifest.get("supporting_auth_role"))
    env = snapshot.pg_environment()
    files = snapshot.validate_archive(root / "dump", manifest, env, logs)
    snapshot.require(files == manifest["archive_files"], "archive-seal")
    toc = snapshot.command(["pg_restore", "--list", str(root / "dump")], env, logs, "restore-toc")
    exclusions = json.loads(Path(args.exclusions).read_text()) if args.exclusions else []
    http_review = None
    proof_path, sql_path = getattr(args, "http_acl_source_proof", None), getattr(args, "http_acl_sql", None)
    snapshot.require(bool(proof_path) == bool(sql_path), "http-acl-review-files")
    if proof_path:
        http_review = dict(source_proof=Path(proof_path).read_bytes(), acl_sql=Path(sql_path).read_bytes(),
                           manifest_sha256=snapshot.sha256_file(root / "manifest.json"))
    listing, entries = select_toc(toc, exclusions, manifest["schema_metadata"], http_review)
    counts = validate_toc_inventory(entries, manifest["schema_metadata"])
    inventory_path = Path(args.target_inventory)
    snapshot.require(snapshot.sha256_file(inventory_path / "target.json") == (inventory_path / "target.sha256").read_text().strip(), "target-inventory-seal")
    evidence = json.loads((inventory_path / "target.json").read_text())
    blockers = target_blockers(evidence)
    # Even an explicit http exclusion cannot prove callers do not need its types.
    blockers.extend(["extension-type-and-runtime-dependencies-require-review",
                     "approved-platform-metadata-deltas-required",
                     "exact-auth-role-schema-owner-acl-prerequisites-require-review",
                     "public-schema-owner-and-initial-acl-preservation-require-review",
                     "separately-reviewed-write-enabled-target-service-and-final-preflight-required"])
    if any(re.match(r"^[0-9]+; [0-9]+ [0-9]+ SCHEMA - public(?: |$)", line) for line in entries.values()):
        blockers.append("existing-public-schema-requires-reviewed-empty-target-handling")
    snapshot.private_write(output / "restore.list", listing)
    snapshot.private_json(output / "source-metadata.json", manifest["schema_metadata"])
    snapshot.private_json(output / "supporting-auth-role.json", supporting_auth)
    snapshot.private_write(output / "source-schema.sql", snapshot.command(
        ["pg_restore", "--schema-only", "--file=-", str(root / "dump")], env, logs, "archive-schema"))
    states = sequence_states(toc, root / "dump", env, logs, output)
    snapshot.require({(s["schema"], s["name"]) for s in states}
                     == {(s["schema"], s["name"]) for s in manifest["sequences"]}, "archive-sequence-set")
    snapshot.private_json(output / "archive-sequences.json", states)
    snapshot.private_write(output / "revoke-export-role.sql", export_role_cleanup(manifest).encode())
    result = dict(format=1, execution_supported=False, status="review-required", blockers=blockers,
                  target=snapshot.TARGET_HOST, target_evidence_sha256=snapshot.sha256_file(inventory_path / "target.json"),
                  manifest_sha256=snapshot.sha256_file(root / "manifest.json"), toc_sha256=hashlib.sha256(toc).hexdigest(),
                  toc_counts=counts, exclusions=exclusions, archive_sequences=states,
                  http_acl_review=(dict(seals=HTTP_ACL_REVIEW_SHA256,
                                       entries=list(reviewed_http_acls(toc, manifest["schema_metadata"], http_review).values()))
                                   if http_review is not None else None),
                  source_public_schema_owner=source_public_schema(manifest["schema_metadata"])["owner"],
                  supporting_auth_role_sha256=AUTH_ROLE_SHA256,
                  native_restore_command=native_restore_command(output / "restore.list", root / "dump"),
                  unsupported_runtime=["5 retired postgres-owned functions using auth.jwt/auth.uid/auth.users",
                                       "2 retired postgres-owned functions using net.*"],
                  retained_platform_helpers="6 non-extension functions in extensions; do not register source event triggers")
    snapshot.private_json(output / "plan.json", result)
    print("restore plan saved privately; review required; no target writes")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    inv = modes.add_parser("inventory")
    inv.add_argument("--target-secret-arn", required=True)
    inv.add_argument("--output", required=True)
    prep = modes.add_parser("plan")
    prep.add_argument("--export-directory", required=True)
    prep.add_argument("--target-inventory", required=True)
    prep.add_argument("--exclusions")
    prep.add_argument("--http-acl-source-proof", help="Exact reviewed supplemental CURRENT HTTP catalog JSON")
    prep.add_argument("--http-acl-sql", help="Exact reviewed original archive ACL-only SQL")
    prep.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    os.umask(0o077)
    previous = signal.signal(signal.SIGTERM, snapshot.interrupted)
    try:
        (inventory if args.mode == "inventory" else plan)(args)
        return 0
    except snapshot.Failure as error:
        print("ERROR phase=" + str(error), file=sys.stderr)
    except (Exception, KeyboardInterrupt):
        print("ERROR phase=restore-plan", file=sys.stderr)
    finally:
        signal.signal(signal.SIGTERM, previous)
    return 1


if __name__ == "__main__":
    sys.exit(main())
