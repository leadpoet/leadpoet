#!/usr/bin/env python3
"""Prepare, create or disable the one temporary October 28 source export role.

No I/O occurs at import. Run preflight first, then `create --plan PATH` to save
non-secret review input. After review, use `create --plan PATH --receipt PATH
--execute`. All modes require --expected-head. The code must match that commit.
Disable uses the saved receipt and --execute. Preflight with --receipt inspects
an interrupted operation without retrying DDL. This is not a migration runner.

PUBLIC privileges are unchanged. The export job MUST use READ ONLY transactions;
read-only defaults can be changed by a login and are not a security boundary.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import json
import os
from pathlib import Path
import platform
import re
import secrets
import subprocess
import sys
import tempfile
from urllib.request import Request, urlopen
import uuid

PROJECT = "qplwoislplkcegvdmbim"
ROLE = "leadpoet_aws_export_20261028"
HOST = "aws-0-us-east-1.pooler.supabase.com"
SCHEMAS = ("public", "source_add_history", "extensions")
AWS = "/Users/pranav/.local/bin/aws"
PROFILE = "leadpoet-admin"
REGION = "us-east-1"
ACCOUNT = "493765492819"
AWS_AUTHORITY_OVERRIDES = frozenset({
    "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN",
    "AWS_SECURITY_TOKEN", "AWS_PROFILE", "AWS_DEFAULT_PROFILE",
    "AWS_SHARED_CREDENTIALS_FILE", "AWS_CONFIG_FILE",
    "AWS_WEB_IDENTITY_TOKEN_FILE", "AWS_ROLE_ARN", "AWS_ROLE_SESSION_NAME",
    "AWS_CONTAINER_CREDENTIALS_FULL_URI", "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
    "AWS_EC2_METADATA_SERVICE_ENDPOINT", "AWS_EC2_METADATA_SERVICE_ENDPOINT_MODE",
    "BOTO_CONFIG",
})
SECRET_NAME = "leadpoet/migration/subnet/source-export-20261028"
COMMENT_PREFIX = "leadpoet-aws-source-export:"
CREATOR_MEMBERSHIP = [{"role": ROLE, "member": "postgres", "admin": True, "inherit": False, "set": False}]
IDENT = re.compile(r"[a-z_][a-z0-9_]{0,62}\Z", re.ASCII)
SHA = re.compile(r"[0-9a-f]{40}\Z", re.ASCII)
ARN = re.compile(r"arn:aws:secretsmanager:us-east-1:493765492819:secret:"
                 + re.escape(SECRET_NAME) + r"-[A-Za-z0-9]{6}\Z", re.ASCII)
VERIFIER = re.compile(r"SCRAM-SHA-256\$4096:[A-Za-z0-9+/]{22}==\$[A-Za-z0-9+/]{43}=:[A-Za-z0-9+/]{43}=\Z", re.ASCII)


class Failure(Exception):
    """Only a fixed phase label may reach the operator."""


def require(ok, phase):
    if not ok:
        raise Failure(phase)


def ident(value):
    require(isinstance(value, str) and IDENT.fullmatch(value), "identifier")
    return '"' + value + '"'


def literal(value):
    # All dynamic values are separately validated. Escape backslashes too, so
    # this remains safe even if a server changes standard_conforming_strings.
    require(isinstance(value, str) and "\0" not in value, "literal")
    return "E'" + value.replace("\\", "\\\\").replace("'", "''") + "'"


def operation(value):
    try:
        require(str(uuid.UUID(value)) == value, "operation-id")
    except (ValueError, TypeError, AttributeError):
        raise Failure("operation-id") from None
    return value


def run(args, data=None):
    try:
        return subprocess.run(args, input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              check=False, timeout=120,
                              env={**os.environ, "AWS_PAGER": "", "AWS_CLI_AUTO_PROMPT": "off",
                                   "AWS_IGNORE_CONFIGURED_ENDPOINT_URLS": "true"})
    except (OSError, subprocess.SubprocessError):
        raise Failure("command") from None


def access_token():
    # Same CLI Keychain record and encoding as the protected migration helper.
    # No environment fallback: this operator tool must use the local CLI record.
    require(platform.system() == "Darwin", "keychain-platform")
    result = run(["security", "find-generic-password", "-s", "Supabase CLI", "-w"])
    require(result.returncode == 0, "keychain-read")
    try:
        value = result.stdout.decode("utf-8").strip()
        prefix = "go-keyring-base64:"
        if value.startswith(prefix):
            value = base64.b64decode(value[len(prefix):], validate=True).decode("utf-8")
        require(bool(value) and not any(c in value for c in "\r\n\0"), "keychain-format")
        return value
    except (ValueError, UnicodeDecodeError):
        raise Failure("keychain-format") from None


def query(sql, *, read_only=True):
    request = Request(f"https://api.supabase.com/v1/projects/{PROJECT}/database/query",
                      data=json.dumps({"query": sql, "read_only": read_only}).encode(),
                      headers={"Authorization": "Bearer " + access_token(),
                               "Content-Type": "application/json", "Accept": "application/json",
                               "User-Agent": "leadpoet-aws-export-role/1"}, method="POST")
    try:
        with urlopen(request, timeout=120) as response:
            value = json.loads(response.read())
        require(isinstance(value, list) and all(isinstance(r, dict) for r in value), "source-response")
        return value
    except Failure:
        raise
    except Exception:
        # Never show an HTTP body, SQL text, verifier, header, or raw exception.
        raise Failure("source-request") from None


def aws(*args, payload=None):
    require(not any(os.environ.get(name) for name in AWS_AUTHORITY_OVERRIDES), "aws-authority")
    command = [AWS, *args, "--profile", PROFILE, "--region", REGION,
               "--output", "json", "--no-cli-pager"]
    if payload is None:
        return run(command)
    # AWS CLI v2 does not reliably read JSON from /dev/stdin. Only the private
    # pathname enters argv; the context removes the payload on success or error.
    try:
        with tempfile.TemporaryDirectory(prefix="leadpoet-aws-payload-") as directory:
            root = Path(directory)
            os.chmod(root, 0o700)
            path = root / "input.json"
            with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as out:
                json.dump(payload, out)
            return run(command + ["--cli-input-json", path.as_uri()])
    except OSError:
        raise Failure("aws-payload-file") from None


def aws_json(*args, payload=None):
    result = aws(*args, payload=payload)
    require(result.returncode == 0, "aws-request")
    try:
        value = json.loads(result.stdout)
        require(isinstance(value, dict), "aws-response")
        return value
    except ValueError:
        raise Failure("aws-response") from None


def check_aws():
    value = aws_json("sts", "get-caller-identity")
    require(value.get("Account") == ACCOUNT, "aws-account")
    require(re.fullmatch(rf"arn:aws:sts::{ACCOUNT}:assumed-role/[^/]+/[^/]+",
                         str(value.get("Arn") or "")) is not None, "aws-identity")


def secret_absent():
    result = aws("secretsmanager", "describe-secret", "--secret-id", SECRET_NAME)
    require(result.returncode != 0, "secret-already-exists")
    # AWS CLI has no structured error output. Recognize only the exact service
    # error code; all permission, transport and unexpected failures fail closed.
    require(b"(ResourceNotFoundException)" in result.stderr, "secret-preflight")


INVENTORY_SQL = """SELECT jsonb_agg(jsonb_build_object('schema', n.nspname,
 'name', c.relname, 'kind', c.relkind) ORDER BY n.nspname, c.relname)
 FROM pg_catalog.pg_class c JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace
 WHERE n.nspname IN ('public','source_add_history','extensions')
 AND c.relkind IN ('r','p','f','S')"""
MEMBERSHIPS_SQL = """SELECT coalesce(jsonb_agg(jsonb_build_object(
 'role', granted.rolname, 'member', recipient.rolname, 'admin', m.admin_option,
 'inherit', m.inherit_option, 'set', m.set_option) ORDER BY granted.rolname, recipient.rolname), '[]'::jsonb)
 FROM pg_catalog.pg_auth_members m
 JOIN pg_catalog.pg_roles granted ON granted.oid=m.roleid
 JOIN pg_catalog.pg_roles recipient ON recipient.oid=m.member
 WHERE m.member=r.oid OR m.roleid=r.oid"""
STATE_SQL = """SELECT jsonb_build_object(
 'current_user', current_user, 'database', current_database(),
 'schemas', (SELECT jsonb_agg(nspname ORDER BY nspname) FROM pg_catalog.pg_namespace
             WHERE nspname IN ('public','source_add_history','extensions')),
 'objects', (""" + INVENTORY_SQL + """),
 'role', (SELECT jsonb_build_object('comment', pg_catalog.shobj_description(oid,'pg_authid'),
   'login', rolcanlogin, 'bypassrls', rolbypassrls, 'super', rolsuper,
   'createdb', rolcreatedb, 'createrole', rolcreaterole, 'replication', rolreplication,
   'connections', rolconnlimit, 'config', rolconfig,
   'valid_until', rolvaliduntil,
   'grants', (SELECT coalesce(jsonb_agg(g ORDER BY g.schema, g.name, g.kind, g.privilege), '[]'::jsonb) FROM (
      SELECT n.nspname AS schema, c.relname AS name, c.relkind::text AS kind,
        a.privilege_type AS privilege, a.is_grantable AS grantable
      FROM pg_catalog.pg_class c JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace
      CROSS JOIN LATERAL pg_catalog.aclexplode(c.relacl) a WHERE a.grantee=r.oid
      UNION ALL SELECT n.nspname, n.nspname, 'n', a.privilege_type, a.is_grantable
      FROM pg_catalog.pg_namespace n CROSS JOIN LATERAL pg_catalog.aclexplode(n.nspacl) a
      WHERE a.grantee=r.oid) g),
   'memberships', (""" + MEMBERSHIPS_SQL + """))
   FROM pg_catalog.pg_roles r WHERE rolname='""" + ROLE + """'),
 'sessions', (SELECT count(*) FROM pg_catalog.pg_stat_activity WHERE usename='""" + ROLE + """')
 ) AS state;"""


def inventory(objects):
    require(isinstance(objects, list), "source-inventory")
    normalized = []
    for item in objects:
        require(isinstance(item, dict) and set(item) == {"schema", "name", "kind"}, "source-inventory")
        require(item["schema"] in SCHEMAS and item["kind"] in ("r", "S"), "source-inventory")
        ident(item["schema"])
        ident(item["name"])
        normalized.append(dict(item))
    normalized.sort(key=lambda item: (item["schema"], item["name"]))
    require(len({(i["schema"], i["name"]) for i in normalized}) == len(normalized), "source-inventory")
    require(sum(i["kind"] == "r" for i in normalized) == 48
            and sum(i["kind"] == "S" for i in normalized) == 3, "source-inventory")
    return normalized


def state():
    # The API's read_only route changes current_user to its restricted reader.
    # Use the admin route for role metadata, with SQL enforcing read-only work.
    rows = query("BEGIN READ ONLY;\n" + STATE_SQL + "\nROLLBACK;", read_only=False)
    require(len(rows) == 1 and isinstance(rows[0].get("state"), dict), "source-state")
    value = rows[0]["state"]
    require(value.get("current_user") == "postgres" and value.get("database") == "postgres", "source-identity")
    require(value.get("schemas") == sorted(SCHEMAS), "source-schemas")
    value["objects"] = inventory(value.get("objects"))
    require(type(value.get("sessions")) is int and value["sessions"] >= 0, "source-sessions")
    return value


def verified_head(repo, expected):
    require(isinstance(expected, str) and SHA.fullmatch(expected), "repo-head")
    repo = Path(repo).resolve()
    current = run(["git", "-C", str(repo), "rev-parse", "HEAD"])
    require(current.returncode == 0 and current.stdout.decode().strip() == expected, "repo-head")
    committed = run(["git", "-C", str(repo), "show", expected + ":scripts/aws_subnet_export_role.py"])
    require(committed.returncode == 0 and committed.stdout == Path(__file__).read_bytes(), "repo-tool-bytes")
    return expected


def scram_verifier(password, salt=None):
    require(isinstance(password, str) and password.isascii() and bool(password), "password-format")
    salt = secrets.token_bytes(16) if salt is None else salt
    require(isinstance(salt, bytes) and len(salt) == 16, "salt-format")
    salted = hashlib.pbkdf2_hmac("sha256", password.encode("ascii"), salt, 4096)
    client = hmac.new(salted, b"Client Key", hashlib.sha256).digest()
    stored = hashlib.sha256(client).digest()
    server = hmac.new(salted, b"Server Key", hashlib.sha256).digest()
    b64 = lambda value: base64.b64encode(value).decode("ascii")
    return "SCRAM-SHA-256$4096:" + b64(salt) + "$" + b64(stored) + ":" + b64(server)


def read_json(path):
    try:
        with Path(path).open() as source:
            value = json.load(source)
        require(isinstance(value, dict), "metadata-format")
        return value
    except (OSError, ValueError):
        raise Failure("metadata-read") from None


def write_json(path, value, *, replace=False):
    data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    path = Path(path)
    temporary = path.with_name(path.name + "." + secrets.token_hex(8)) if replace else path
    try:
        with os.fdopen(os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600), "wb") as out:
            out.write(data)
            out.flush()
            os.fsync(out.fileno())
        if replace:
            os.replace(temporary, path)
    except OSError:
        raise Failure("metadata-write") from None


def validate_metadata(value, head):
    require(value.get("format") == 1 and value.get("project") == PROJECT
            and value.get("role") == ROLE and value.get("secret_name") == SECRET_NAME
            and value.get("head") == head
            and value.get("creator_membership") == CREATOR_MEMBERSHIP, "metadata-binding")
    operation(value.get("operation_id"))
    require(value.get("objects") == inventory(value.get("objects")), "metadata-inventory")
    return value


def validate_receipt(value, head):
    validate_metadata(value, head)
    require(isinstance(value.get("secret_arn"), str) and ARN.fullmatch(value["secret_arn"]), "receipt-secret")
    return value


def guard_sql(objects, op=None, *, absent=False):
    expected = literal(json.dumps(inventory(objects), separators=(",", ":"))) + "::jsonb"
    checks = ["current_user <> 'postgres'", "current_database() <> 'postgres'",
              "(" + INVENTORY_SQL + ") IS DISTINCT FROM " + expected]
    if absent:
        checks.append("EXISTS(SELECT 1 FROM pg_catalog.pg_roles WHERE rolname=" + literal(ROLE) + ")")
    else:
        checks.append("NOT EXISTS(SELECT 1 FROM pg_catalog.pg_roles WHERE rolname=" + literal(ROLE)
                      + " AND pg_catalog.shobj_description(oid,'pg_authid')=" + literal(COMMENT_PREFIX + operation(op)) + ")")
        checks.append("EXISTS(SELECT 1 FROM pg_catalog.pg_stat_activity WHERE usename=" + literal(ROLE) + ")")
        checks.append("(SELECT (" + MEMBERSHIPS_SQL + ") FROM pg_catalog.pg_roles r WHERE r.rolname="
                      + literal(ROLE) + ") IS DISTINCT FROM "
                      + literal(json.dumps(CREATOR_MEMBERSHIP)) + "::jsonb")
    return "DO $guard$ BEGIN IF " + " OR ".join(checks) + " THEN RAISE EXCEPTION 'export role guard'; END IF; END $guard$;"


def object_list(objects, kind):
    return ", ".join(ident(o["schema"]) + "." + ident(o["name"]) for o in inventory(objects) if o["kind"] == kind)


def create_sql(plan, verifier, expires):
    require(VERIFIER.fullmatch(verifier), "verifier-format")
    op = operation(plan["operation_id"])
    role = ident(ROLE)
    return "\n".join([
        "BEGIN;", "SET LOCAL lock_timeout='5s';", "SET LOCAL statement_timeout='60s';",
        guard_sql(plan["objects"], absent=True),
        f"CREATE ROLE {role} LOGIN NOINHERIT NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION BYPASSRLS CONNECTION LIMIT 5 PASSWORD {literal(verifier)} VALID UNTIL {literal(expires)};",
        f"ALTER ROLE {role} SET default_transaction_read_only=on;",
        f"COMMENT ON ROLE {role} IS {literal(COMMENT_PREFIX + op)};",
        "GRANT USAGE ON SCHEMA " + ", ".join(ident(s) for s in SCHEMAS) + f" TO {role};",
        "GRANT SELECT ON TABLE " + object_list(plan["objects"], "r") + f" TO {role};",
        "GRANT SELECT ON SEQUENCE " + object_list(plan["objects"], "S") + f" TO {role};",
        guard_sql(plan["objects"], op),
        "COMMIT;",
    ])


def disable_sql(receipt):
    role = ident(ROLE)
    return "\n".join([
        "BEGIN;", "SET LOCAL lock_timeout='5s';", "SET LOCAL statement_timeout='60s';",
        guard_sql(receipt["objects"], receipt["operation_id"]),
        f"ALTER ROLE {role} NOLOGIN NOBYPASSRLS;",
        "REVOKE SELECT ON TABLE " + object_list(receipt["objects"], "r") + f" FROM {role};",
        "REVOKE SELECT ON SEQUENCE " + object_list(receipt["objects"], "S") + f" FROM {role};",
        "REVOKE USAGE ON SCHEMA " + ", ".join(ident(s) for s in SCHEMAS) + f" FROM {role};",
        # Catch any connection that arrived while permission changes were pending.
        guard_sql(receipt["objects"], receipt["operation_id"]), "COMMIT;",
    ])


def expected_grants(objects):
    grants = [dict(schema=o["schema"], name=o["name"], kind=o["kind"],
                   privilege="SELECT", grantable=False) for o in inventory(objects)]
    grants += [dict(schema=s, name=s, kind="n", privilege="USAGE", grantable=False) for s in SCHEMAS]
    return sorted(grants, key=lambda g: (g["schema"], g["name"], g["kind"], g["privilege"]))


def bound_role(value, receipt):
    role = value.get("role")
    require(isinstance(role, dict) and role.get("comment") == COMMENT_PREFIX + receipt["operation_id"], "role-operation-binding")
    require(role.get("memberships") == CREATOR_MEMBERSHIP, "role-memberships")
    return role


def preflight(receipt=None):
    check_aws()
    value = state()
    if receipt:
        if value.get("role") is None:
            return {"status": "inspected-role-absent", "operation_id": receipt["operation_id"]}
        role = bound_role(value, receipt)
        return {"status": "inspected", "login": role.get("login"), "bypassrls": role.get("bypassrls"),
                "sessions": value["sessions"], "operation_id": receipt["operation_id"]}
    require(value.get("role") is None, "role-already-exists")
    secret_absent()
    return value


def plan_create(path, head):
    value = preflight()
    plan = dict(format=1, project=PROJECT, role=ROLE, secret_name=SECRET_NAME, head=head,
                operation_id=str(uuid.uuid4()), objects=value["objects"], creator_membership=CREATOR_MEMBERSHIP,
                created_at=datetime.now(timezone.utc).isoformat(),
                residual_public_privileges="unchanged; export must use READ ONLY transactions")
    write_json(path, plan)
    return {"status": "planned", "operation_id": plan["operation_id"], "tables": 48, "sequences": 3}


def execute_create(plan_path, receipt_path, head):
    plan = validate_metadata(read_json(plan_path), head)
    require(receipt_path is not None, "receipt-required")
    if Path(receipt_path).exists():
        receipt = validate_receipt(read_json(receipt_path), head)
        require(receipt["operation_id"] == plan["operation_id"], "receipt-plan-binding")
        # This deliberate stop is safe even after a dropped COMMIT response.
        preflight(receipt)
        raise Failure("existing-receipt-inspected-review-required")
    value = preflight()
    require(value["objects"] == plan["objects"], "source-inventory-changed")
    password = secrets.token_urlsafe(32)
    verifier = scram_verifier(password)
    expires = (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat()
    credential = dict(host=HOST, port=5432, dbname="postgres", role=ROLE,
                      user=ROLE + "." + PROJECT, password=password, operation_id=plan["operation_id"])
    result = aws_json("secretsmanager", "create-secret", payload={
        "Name": SECRET_NAME, "ClientRequestToken": plan["operation_id"],
        "Description": "Temporary October 28 read-only source export; remove after verified transfer",
        "SecretString": json.dumps(credential, separators=(",", ":")),
        "Tags": [{"Key": "OperationId", "Value": plan["operation_id"]}],
    })
    arn = result.get("ARN")
    require(isinstance(arn, str) and ARN.fullmatch(arn) and result.get("Name") == SECRET_NAME, "secret-created-response")
    receipt = {**plan, "secret_arn": arn, "valid_until": expires, "status": "secret-created-source-not-confirmed"}
    # Preserve metadata before source DDL. No credential/verifier enters this file.
    write_json(receipt_path, receipt)
    try:
        query(create_sql(plan, verifier, expires), read_only=False)
        observed = state()
        role = bound_role(observed, receipt)
        require(role.get("login") is True and role.get("bypassrls") is True
                and all(role.get(k) is False for k in ("super", "createdb", "createrole", "replication"))
                and role.get("connections") == 5
                and "default_transaction_read_only=on" in (role.get("config") or [])
                and role.get("grants") == expected_grants(plan["objects"]), "created-role-verification")
        expiry = datetime.fromisoformat(role.get("valid_until").replace("Z", "+00:00"))
        require(datetime.now(timezone.utc) < expiry <= datetime.fromisoformat(expires), "created-role-expiry")
    except Exception:
        receipt["status"] = "source-outcome-unknown-retain-secret-inspect-before-resume"
        write_json(receipt_path, receipt, replace=True)
        raise Failure("source-outcome-unknown-receipt-retained") from None
    receipt["status"] = "created"
    write_json(receipt_path, receipt, replace=True)
    return {"status": "created", "secret_arn": arn, "operation_id": plan["operation_id"], "tables": 48, "sequences": 3}


def disable(receipt_path, head, execute=False):
    receipt = validate_receipt(read_json(receipt_path), head)
    check_aws()
    observed = state()
    bound_role(observed, receipt)
    require(observed["objects"] == receipt["objects"], "source-inventory-changed")
    require(observed["sessions"] == 0, "role-has-active-sessions")
    if not execute:
        return {"status": "disable-ready", "operation_id": receipt["operation_id"]}
    try:
        query(disable_sql(receipt), read_only=False)
        role = bound_role(state(), receipt)
        require(role.get("login") is False and role.get("bypassrls") is False
                and role.get("grants") == [], "disabled-role-verification")
    except Exception:
        receipt["status"] = "disable-outcome-unknown-retain-secret-inspect-before-resume"
        write_json(receipt_path, receipt, replace=True)
        raise Failure("disable-outcome-unknown-receipt-retained") from None
    receipt["status"] = "disabled-secret-cleanup-remains"
    write_json(receipt_path, receipt, replace=True)
    return {"status": receipt["status"], "operation_id": receipt["operation_id"], "secret_arn": receipt["secret_arn"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "create", "disable"))
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        head = verified_head(args.repo, args.expected_head)
        if args.mode == "preflight":
            require(not args.execute and args.plan is None, "mode-arguments")
            receipt = validate_receipt(read_json(args.receipt), head) if args.receipt else None
            value = preflight(receipt)
            output = value if receipt else {"status": "preflight-ready", "tables": 48, "sequences": 3}
        elif args.mode == "create":
            require(args.plan is not None, "plan-required")
            require(args.execute or args.receipt is None, "mode-arguments")
            output = execute_create(args.plan, args.receipt, head) if args.execute else plan_create(args.plan, head)
        else:
            require(args.receipt is not None and args.plan is None, "mode-arguments")
            output = disable(args.receipt, head, args.execute)
        print(json.dumps(output, sort_keys=True))
        return 0
    except Failure as exc:
        print("ERROR phase=" + str(exc), file=sys.stderr)
        return 1
    except Exception:
        print("ERROR phase=operator-utility", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
