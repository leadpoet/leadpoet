#!/usr/bin/env python3
"""Read-only post-restore metadata proof for the dedicated rehearsal target.

Run under the reviewed root systemd deadline and private log controls used by
aws_subnet_snapshot.py. This never restores, changes ACLs or cleans up roles.
The separate verify-data operation must still prove captured row checksums.
"""
import argparse
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
    from . import aws_subnet_target_bootstrap as bootstrap
else:
    import aws_subnet_restore as restore
    import aws_subnet_snapshot as snapshot
    import aws_subnet_target_bootstrap as bootstrap


HASH = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
DATABASE_ACL_DEFAULT_SQL = ("SELECT json_build_object('database',d.datname,"
    "'owner',pg_get_userbyid(d.datdba),'acl'," + snapshot.acl("pg_catalog.acldefault('d',d.datdba)")
    + ") FROM pg_catalog.pg_database d WHERE d.datname=current_database();")
HTTP_EXTENSION = dict(name="http", schema="public", version="1.6")
HTTP_DOMAIN_CONSTRAINT = dict(schema="public", name="content_type_check", relation=None,
    kind="c", validated=True, deferrable=False, initially_deferred=False,
    definition=r"CHECK ((VALUE ~ '^\S+\/\S+'::text))")
HTTP_DOMAIN_MEMBER = dict(kind="type", **{"class": "pg_type"}, schema="public", name="content_type",
    owner="supabase_admin", identity="public.content_type", identity_arguments=None,
    result=None, acl=None, acl_entries=[])
HTTP_DOMAIN_DEPENDENCY = dict(dependent_class="pg_constraint", dependent_type="domain constraint",
    dependent_schema="public", dependent_name=None,
    dependent_identity="content_type_check on public.content_type", dependency_type="a",
    referenced_kind="type", referenced_identity="public.content_type")
COLUMN_CANONICALIZATION = "column-position-logical-ordinal-per-relation"
DBLINK_EXTENSION = dict(name="dblink", schema="public", version="1.2")
DBLINK_COMPOSITE = dict(schema="public", name="dblink_pkey_results", kind="c",
    type="dblink_pkey_results", type_kind="c", dependency="e", extension="dblink",
    extension_schema="public", extension_version="1.2", columns=[
        dict(name="position", position=1, type="integer", not_null=False),
        dict(name="colname", position=2, type="text", not_null=False)])
COMPOSITE_RELATIONS_SQL = snapshot.aggregate("""SELECT n.nspname AS schema,c.relname AS name,
 c.relkind::text AS kind,t.typname AS type,t.typtype::text AS type_kind,
 p.deptype::text AS dependency,e.extname AS extension,en.nspname AS extension_schema,
 e.extversion AS extension_version,
 (SELECT coalesce(json_agg(json_build_object('name',a.attname,'position',a.attnum,
   'type',pg_catalog.format_type(a.atttypid,a.atttypmod),'not_null',a.attnotnull)
   ORDER BY a.attnum),'[]'::json) FROM pg_catalog.pg_attribute a
   WHERE a.attrelid=c.oid AND a.attnum>0) AS columns
 FROM pg_catalog.pg_class c JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace
 JOIN pg_catalog.pg_type t ON c.reltype=t.oid AND t.typrelid=c.oid
 LEFT JOIN pg_catalog.pg_depend p ON p.classid='pg_catalog.pg_type'::regclass
   AND p.objid=t.oid AND p.objsubid=0 AND p.deptype='e'
   AND p.refclassid='pg_catalog.pg_extension'::regclass AND p.refobjsubid=0
 LEFT JOIN pg_catalog.pg_extension e ON e.oid=p.refobjid
 LEFT JOIN pg_catalog.pg_namespace en ON en.oid=e.extnamespace
 WHERE c.relkind='c' AND """ + bootstrap.NS + """
 AND NOT EXISTS(SELECT 1 FROM pg_catalog.pg_depend d
   WHERE d.classid='pg_class'::regclass AND d.objid=c.oid AND d.deptype='e')""", "t.schema,t.name")


def sealed_json(path, digest):
    path = Path(path)
    snapshot.require(isinstance(digest, str) and HASH.fullmatch(digest)
                     and snapshot.sha256_file(path) == digest, "reference-seal")
    return json.loads(path.read_text())


def exact_http_constraint(record):
    return (isinstance(record, dict) and record == HTTP_DOMAIN_CONSTRAINT
            and all(type(record[key]) is type(value) for key, value in HTTP_DOMAIN_CONSTRAINT.items()))


def validate_http_domain_exclusion(path, digest, manifest, manifest_digest):
    """Optional reviewed exception for one auto-dependent HTTP domain CHECK."""
    if path is None and digest is None:
        return None
    snapshot.require(path is not None and digest is not None, "http-domain-proof")
    proof = sealed_json(path, digest)
    snapshot.require(isinstance(proof, dict) and set(proof) == {
        "format", "manifest_sha256", "extension", "current_source_members", "constraint"}
        and type(proof["format"]) is int and proof["format"] == 1
        and proof["manifest_sha256"] == manifest_digest
        and proof["extension"] == HTTP_EXTENSION
        and exact_http_constraint(proof["constraint"]), "http-domain-proof")
    snapshot.require([e for e in manifest["extensions"] if e.get("name") == "http"] == [HTTP_EXTENSION]
        and sum(exact_http_constraint(c) for c in manifest["schema_metadata"]["constraints"]) == 1,
        "http-domain-manifest")
    reference = proof["current_source_members"]
    snapshot.require(isinstance(reference, dict) and set(reference) == {"file", "sha256"}
        and reference["file"] == "source-members-pg-catalog.json", "http-domain-current-proof")
    current = sealed_json(Path(path).parent / reference["file"], reference["sha256"])
    evidence = current["evidence"]
    snapshot.require(current["capture"]["sealed_manifest_sha256"] == manifest_digest
        and evidence["evidence_kind"] == "supplemental-current-source-catalog-not-sealed-snapshot"
        and evidence["project"] == "qplwoislplkcegvdmbim"
        and evidence["current_user"] == evidence["current_database"] == "postgres"
        and evidence["transaction_read_only"] == "on" and evidence["search_path"] == "pg_catalog"
        and evidence["extension"] == dict(**HTTP_EXTENSION, owner="supabase_admin")
        and [m for m in evidence["members"] if m.get("identity") == "public.content_type"] == [HTTP_DOMAIN_MEMBER]
        and [d for d in evidence["direct_non_extension_owned_dependents"]
             if d.get("dependent_identity") == HTTP_DOMAIN_DEPENDENCY["dependent_identity"]] == [HTTP_DOMAIN_DEPENDENCY],
        "http-domain-current-proof")
    return dict(rule="excluded-http-extension-domain-constraint", constraint=proof["constraint"],
                extension=proof["extension"], proof_sha256=digest, manifest_sha256=manifest_digest,
                current_source_members_sha256=reference["sha256"])


def acl_items(value):
    """Order only: preserve each entry's grantor, rights and grant options."""
    snapshot.require(isinstance(value, str) and value.startswith("{") and value.endswith("}")
                     and not any(c in value for c in ('"', '\\', '\n', '\r')), "acl-text")
    items = value[1:-1].split(",") if value != "{}" else []
    snapshot.require(all(re.fullmatch(r"[a-z_0-9]*=[a-zA-Z*]*/[a-z_][a-z_0-9]*", item, re.ASCII)
                         for item in items) and len(set(items)) == len(items), "acl-text")
    return sorted(items)


def canonical_acl(value):
    snapshot.require(isinstance(value, dict) and set(value) == {"is_null", "items"}
                     and type(value["is_null"]) is bool and isinstance(value["items"], list)
                     and all(isinstance(item, str) for item in value["items"]), "acl-shape")
    snapshot.require(not value["is_null"] or not value["items"], "acl-shape")
    return dict(is_null=value["is_null"], items=sorted(value["items"]))


def clean_source_metadata(manifest, http_domain_exclusion=None):
    """Canonical source metadata after the exact approved cleanup exceptions."""
    metadata = copy.deepcopy(manifest["schema_metadata"])
    snapshot.require(set(metadata) == set(snapshot.OBJECT_KINDS)
                     and all(isinstance(items, list) for items in metadata.values()), "source-metadata")
    if http_domain_exclusion is not None:
        snapshot.require(exact_http_constraint(http_domain_exclusion["constraint"])
            and sum(exact_http_constraint(c) for c in metadata["constraints"]) == 1, "http-domain-manifest")
        metadata["constraints"] = [c for c in metadata["constraints"] if not exact_http_constraint(c)]
    tables = {(t["schema"], t["name"]) for t in manifest["tables"]}
    sequences = {(s["schema"], s["name"]) for s in manifest["sequences"]}
    for category, records in metadata.items():
        for item in records:
            snapshot.require(isinstance(item, dict), "source-metadata")
            if "acl" not in item:
                continue
            acl = canonical_acl(item["acl"])
            kept = []
            for entry in acl["items"]:
                grantee, sep, rest = entry.partition("=")
                rights, slash, grantor = rest.partition("/")
                snapshot.require(sep and slash and grantor != snapshot.SOURCE_ROLE, "export-acl-scope")
                if grantee != snapshot.SOURCE_ROLE:
                    kept.append(entry)
                    continue
                key = item.get("schema"), item.get("name")
                allowed = (category == "schemas" and key[0] == key[1] and key[0] in snapshot.SCHEMAS and rights == "U")
                allowed |= (category == "relations" and item.get("kind") == "r" and key in tables and rights == "r")
                allowed |= (category == "relations" and item.get("kind") == "S" and key in sequences and rights == "r")
                snapshot.require(allowed, "export-acl-scope")
            item["acl"] = dict(is_null=acl["is_null"], items=kept)
    return canonical_metadata(metadata)


def canonical_columns(records):
    """Rank visible physical positions without sorting or changing columns."""
    result = copy.deepcopy(records)
    snapshot.require(isinstance(result, list), "column-order")
    seen_relations = set()
    relation = None
    for item in result:
        snapshot.require(isinstance(item, dict) and all(isinstance(item.get(key), str) and item[key]
                         for key in ("schema", "name", "column")), "column-order")
        key = item["schema"], item["name"]
        if key != relation:
            snapshot.require(key not in seen_relations, "column-order")
            seen_relations.add(key)
            relation, previous, ordinal, columns = key, 0, 0, set()
        position = item.get("position")
        snapshot.require(type(position) is int and position > previous
                         and item["column"] not in columns, "column-order")
        previous = position
        columns.add(item["column"])
        ordinal += 1
        item["position"] = ordinal
    return result


def canonical_metadata(metadata):
    """Normalize ACL entry order and logical column ordinals on a copy."""
    value = copy.deepcopy(metadata)
    snapshot.require(set(value) == set(snapshot.OBJECT_KINDS), "metadata-categories")
    for records in value.values():
        snapshot.require(isinstance(records, list), "metadata-shape")
        for item in records:
            if "acl" in item:
                item["acl"] = canonical_acl(item["acl"])
    value["columns"] = canonical_columns(value["columns"])
    return value


def canonical_support(value):
    result = copy.deepcopy(restore.validate_supporting_auth_role(value))
    result["schema"]["acl"] = acl_items(result["schema"]["acl"])
    result["function"]["acl"] = acl_items(result["function"]["acl"])
    return result


def validate_sequences(states, manifest):
    snapshot.require(isinstance(states, list) and len(states) == 3, "archive-sequences")
    normalized = []
    for item in states:
        snapshot.require(isinstance(item, dict) and set(item) == {"schema", "name", "last_value", "is_called"}, "archive-sequences")
        snapshot.qualified(item)
        snapshot.require(isinstance(item["last_value"], str) and re.fullmatch(r"-?[0-9]+", item["last_value"])
                         and type(item["is_called"]) is bool, "archive-sequences")
        normalized.append(dict(item))
    keys = {(item["schema"], item["name"]) for item in normalized}
    snapshot.require(len(keys) == 3 and keys == {(item["schema"], item["name"]) for item in manifest["sequences"]}, "archive-sequences")
    return sorted(normalized, key=lambda item: (item["schema"], item["name"]))


def capture_target(keeper):
    """The caller supplies the existing read-only Keeper and target service."""
    snapshot.check_identity(keeper, "target")
    metadata = snapshot.capture_metadata(keeper)
    state = keeper.one_json(bootstrap.STATE_SQL)
    composite_relations = keeper.one_json(COMPOSITE_RELATIONS_SQL)
    database_acl_default = keeper.one_json(DATABASE_ACL_DEFAULT_SQL)
    # Unlike source capture, target capture must retain ACL-order differences
    # as evidence rather than reject before semantic entry-order comparison.
    support_sql = snapshot.SUPPORTING_AUTH_ROLE_SQL.strip().rstrip(";")
    supports = keeper.one_json("SELECT coalesce(json_agg(v),'[]'::json) FROM (" + support_sql + ") q(v);")
    for support in supports:
        definition = support.get("function", {}).get("definition")
        if isinstance(definition, str):
            support["function"]["sha256"] = hashlib.sha256(definition.encode()).hexdigest()
    sequences = []
    for item in metadata["sequence_definitions"]:
        value = keeper.one_json("SELECT json_build_object('last_value',last_value::text,'is_called',is_called) FROM " + snapshot.qualified(item) + ";")
        sequences.append(dict(schema=item["schema"], name=item["name"], **value))
    return dict(metadata=metadata, state=state, database_acl_default=database_acl_default,
                supporting_auth_roles=supports, sequences=sequences, composite_relations=composite_relations)


def approved_platform_objects(manifest, composite_relations):
    """Count only the exact dblink 1.2 type-owned composite omitted by capture."""
    dblink = [e for e in manifest.get("extensions", []) if e.get("name") == "dblink"]
    snapshot.require(dblink == [] or dblink == [DBLINK_EXTENSION], "platform-composites")
    expected = [DBLINK_COMPOSITE] if dblink else []
    # JSON equality keeps booleans distinct from integer ordinals, ignores only
    # object-key order, and rejects extra fields, rows and changed attributes.
    snapshot.require(isinstance(composite_relations, list)
        and json.dumps(composite_relations, sort_keys=True) == json.dumps(expected, sort_keys=True),
        "platform-composites")
    return copy.deepcopy(composite_relations)


def database_acl_match(initial_target, actual):
    """The sole NULL equivalence: this database's unchanged owner's defaults."""
    initial_acl = canonical_acl(initial_target["database_acl"])
    state = actual["state"]
    actual_acl = canonical_acl(state["database_acl"])
    if not initial_acl["is_null"]:
        return actual_acl == initial_acl, None
    default = actual["database_acl_default"]
    if not (initial_target.get("owner") == state.get("owner") == default.get("owner") == "arena_admin"
            and initial_target.get("database") == state.get("database") == default.get("database") == "postgres"):
        return False, None
    if actual_acl["is_null"]:
        return True, None
    default_acl = canonical_acl(default["acl"])
    if default_acl["is_null"] or actual_acl != default_acl:
        return False, None
    return True, dict(rule="initial-null-database-acl-equals-server-default", scope="database-acl",
                      owner="arena_admin", initial_raw_acl_is_null=True, actual_raw_acl_is_null=False,
                      server_default_acl=default_acl)


def target_roles_match(state):
    expected = [r for r in bootstrap.role_expectations() if r["name"] != snapshot.SOURCE_ROLE]
    if state["roles"] != expected:
        return False
    names = {r["name"] for r in expected}
    rows = {name: [] for name in names}
    for membership in state["memberships"]:
        name = membership["name"]
        if name not in names or membership["member"] != "arena_admin":
            return False
        creator = (membership.get("admin") is True and membership.get("inherit") is False
                   and membership.get("set") is False)
        owner = (name in bootstrap.OWNERS and type(membership.get("admin")) is bool
                 and membership.get("inherit") is True and membership.get("set") is True)
        if not (creator or owner):
            return False
        rows[name].append((creator, owner))
    for name, grants in rows.items():
        if name in bootstrap.OWNERS:
            # PG17 may keep its automatic admin-only grant separately from the
            # explicit owner SET/INHERIT grant, or update that same grant row.
            if not 1 <= len(grants) <= 2 or sum(owner for _, owner in grants) != 1:
                return False
        # RDS17 may omit stock PG17's automatic creator membership.
        # A nonowner needs no access; permit at most one admin-only grant.
        elif len(grants) > 1 or any(grant != (True, False) for grant in grants):
            return False
    return True


def compare(manifest, archive_sequences, initial_target, actual, http_domain_exclusion=None):
    expected = clean_source_metadata(manifest, http_domain_exclusion)
    metadata = canonical_metadata(actual["metadata"])
    differences = restore.compare_metadata(expected, metadata)
    try:
        platform_objects = approved_platform_objects(manifest, actual["composite_relations"])
    except (snapshot.Failure, KeyError, TypeError, ValueError):
        differences.append("platform-composites")
        platform_objects = []
    state = actual["state"]
    if not (state["role"] == state["session"] == state["owner"] == "arena_admin"
            and state["database"] == "postgres" and 170000 <= state["version"] < 180000
            and state["read_only"] == "on" and state["recovery"] is False):
        differences.append("target-identity")
    if not target_roles_match(state):
        differences.append("target-role-security")
    if not database_acl_match(initial_target, actual)[0]:
        differences.append("database-acl")
    if any(r["name"] == snapshot.SOURCE_ROLE for r in state["roles"]) or any(
            snapshot.SOURCE_ROLE in (m["name"], m["member"]) for m in state["memberships"]):
        differences.append("temporary-export-role")
    try:
        snapshot.require(len(actual["supporting_auth_roles"]) == 1, "supporting-auth-role")
        if canonical_support(actual["supporting_auth_roles"][0]) != canonical_support(manifest["supporting_auth_role"]):
            differences.append("supporting-auth-role")
    except (snapshot.Failure, KeyError, TypeError, ValueError):
        differences.append("supporting-auth-role")
    expected_sequences = validate_sequences(archive_sequences, manifest)
    try:
        if validate_sequences(actual["sequences"], manifest) != expected_sequences:
            differences.append("archive-sequences")
    except snapshot.Failure:
        differences.append("archive-sequences")
    # These totals include the three selected schemas plus only the one auth
    # schema/getter. They detect stray objects outside the captured categories.
    expected_counts = dict(relations=len(expected["relations"]) + len(expected["indexes"]) + len(platform_objects),
                           routines=len(expected["routines"]) + 1, types=len(expected["types"]),
                           schemas=sum(s["name"] != "public" for s in expected["schemas"]) + 1,
                           default_acls=len(expected["default_acls"]), event_triggers=0,
                           large_objects=initial_target["counts"]["large_objects"],
                           databases=initial_target["counts"]["databases"])
    if state["counts"] != expected_counts:
        differences.append("unexpected-objects")
    return differences


def verify(args):
    manifest = sealed_json(args.manifest, args.manifest_sha256)
    snapshot.require((Path(args.manifest).parent / "manifest.sha256").read_text().strip() == args.manifest_sha256,
                     "manifest-seal")
    snapshot.require(manifest["format"] == 1 and manifest["source_host"] == snapshot.SOURCE_HOST
                     and manifest["source_role"] == snapshot.SOURCE_ROLE and manifest["database"] == "postgres", "manifest-identity")
    snapshot.validate_tables(manifest["tables"])
    snapshot.require(any(t["schema"] == "source_add_history" for t in manifest["tables"]), "protected-history-missing")
    http_domain_exclusion = validate_http_domain_exclusion(getattr(args, "http_domain_exclusion_proof", None),
        getattr(args, "http_domain_exclusion_proof_sha256", None), manifest, args.manifest_sha256)
    expected = clean_source_metadata(manifest, http_domain_exclusion)
    canonical_support(manifest["supporting_auth_role"])
    sequences = validate_sequences(sealed_json(args.archive_sequences, args.archive_sequences_sha256), manifest)
    initial = sealed_json(args.initial_target, args.initial_target_sha256)
    bootstrap.guard_empty(initial)
    directory = Path(args.directory).absolute()
    directory.mkdir(mode=0o700)
    logs = directory / "private-logs"
    logs.mkdir(mode=0o700)
    config = snapshot.connection_config("target", snapshot.secret(bootstrap.TARGET_SECRET_ARN), snapshot.TARGET_HOST)
    with snapshot.service(config) as env, snapshot.Keeper(env, logs) as keeper:
        actual = capture_target(keeper)
    snapshot.private_json(directory / "actual.json", actual)
    snapshot.private_json(directory / "expected-metadata.json", expected)
    differences = compare(manifest, sequences, initial, actual, http_domain_exclusion)
    counts = {category: len(actual["metadata"][category]) for category in snapshot.OBJECT_KINDS}
    equivalence = database_acl_match(initial, actual)[1]
    platform_objects = (approved_platform_objects(manifest, actual["composite_relations"])
                        if "platform-composites" not in differences else [])
    report = dict(format=1, status="mismatch" if differences else "match", differences=differences, counts=counts,
                  canonicalizations=[COLUMN_CANONICALIZATION],
                  applied_equivalences=[equivalence] if equivalence else [],
                  applied_exclusions=[http_domain_exclusion] if http_domain_exclusion else [],
                  applied_platform_objects=platform_objects,
                  manifest_sha256=args.manifest_sha256, archive_sequences_sha256=args.archive_sequences_sha256,
                  initial_target_sha256=args.initial_target_sha256,
                  data_verification="separate verify-data proof remains required")
    snapshot.private_json(directory / "report.json", report)
    print(json.dumps(dict(status=report["status"], categories=differences, counts=counts), sort_keys=True))
    snapshot.require(not differences, "metadata-mismatch")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "archive-sequences", "initial-target"):
        parser.add_argument("--" + name, required=True)
        parser.add_argument("--" + name + "-sha256", required=True)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--http-domain-exclusion-proof")
    parser.add_argument("--http-domain-exclusion-proof-sha256")
    args = parser.parse_args(argv)
    os.umask(0o077)
    previous = signal.signal(signal.SIGTERM, snapshot.interrupted)
    try:
        verify(args)
        return 0
    except snapshot.Failure as error:
        print("ERROR phase=" + str(error), file=sys.stderr)
    except (Exception, KeyboardInterrupt):
        print("ERROR phase=metadata-verification", file=sys.stderr)
    finally:
        signal.signal(signal.SIGTERM, previous)
    return 1


if __name__ == "__main__":
    sys.exit(main())
