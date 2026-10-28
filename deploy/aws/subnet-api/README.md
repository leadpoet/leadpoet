# Subnet database HTTP compatibility pair

This is packaging for the existing gateway host. It does not provision AWS,
create database roles, change Caddy, switch clients or restart application
services. The migration owner must review and perform those separate steps.

`compose.yaml` pins official PostgREST 14.5 and nginx 1.28.3 images by immutable
multiarchitecture digest. Both run as UID/GID 1000 with no capabilities, a read-only
root and no privilege escalation. Host networking is intentional: the generated
listeners bind only `127.0.0.1:3000` and `127.0.0.1:8081`. The owner must first
verify that those ports are free and that existing TLS routing can reach 8081.
Do not publish 3000 or 8081. This Linux host-network packaging is not a portable
Docker Desktop deployment recipe.

## Private configuration

The renderer takes a mode 0600 JSON regular file with exactly these names:

- `arena_key`: existing `sb_secret_` key for `lab_arena_service`.
- `gateway_key`: separate `sb_secret_` key for `service_role` SDK calls.
- `public_key`: existing `sb_publishable_` key for `anon` public RPCs.
- `jwt_secret`: dedicated high-entropy signing secret, at least 32 URL-safe
  ASCII characters. Retain the same dedicated AWS secret on ordinary app
  deploys. Do not regenerate it as part of app startup.
- `db_uri`: percent-encoded libpq URI for the dedicated least-privilege
  authenticator. Production must use the exact RDS host/database and
  `sslmode=verify-full&sslrootcert=/run/subnet-api/rds-ca.pem`. The CLI
  requires these TLS settings. Set `SUBNET_API_CA_FILE` to the reviewed AWS
  CA file for compose's read-only mount; neither tool retrieves it.

The container filename `rds-ca.pem` is the fixed interface path, not a claim
about the upstream. For an owner-approved rollback, keep the public gateway
origin and request identities unchanged. After writer drain and verified
reverse synchronization, switch the private URI to the exact source database
hostname and mount its verified source CA at this same path. Use the source
hostname rather than its IP so verify-full checks the certificate name. Switch
the URI and CA together, and prove the dedicated source authenticator's exact
SET memberships before relying on this path. Do not disable TLS verification.

Hydrate this input through the existing protected secret workflow. Keep the
input and output outside Git. No secret belongs in argv, compose environment
variables, terminal output or Docker logs. Render as root into a private parent:

```sh
python3 scripts/aws_subnet_api.py \
  --secrets-file /run/leadpoet-subnet-api-input.json \
  --output-dir /etc/leadpoet-subnet-api \
  --owner-uid 1000 --owner-gid 1000
```

The output directory is mode 0700; both configuration files are mode 0600 and
owned by the runtime UID/GID. The renderer emits only a fixed success/error
line. Delete the temporary input after the protected workflow verifies the
output. Never use `nginx -T`, PostgREST dump-config or verbose Docker inspection
to print these files. Container log drivers are disabled; nginx access/error
logs are disabled. Use status, exit codes and bounded authenticated capability
checks for diagnosis. Inspect a failed private configuration only through the
same redacted operator workflow.

The owner can review compose interpolation without secrets because it contains
only a private directory name:

```sh
SUBNET_API_CONFIG_DIR=/etc/leadpoet-subnet-api \
  SUBNET_API_CA_FILE=/etc/leadpoet/rds-ca.pem \
  docker compose -f deploy/aws/subnet-api/compose.yaml config --quiet
```

Starting the pair, routing TLS and changing endpoint/key documents remain
separate owner actions. An explicit API key/signing-secret rotation requires
stopping admission, rendering both private configs and restarting the pair as
one operation. The renderer's two file replacements do not claim atomic live
rotation. Keep the previous private pair for a reviewed rollback.

## Authentication and HTTP behavior

nginx matches opaque keys with anchored, case-sensitive regular expressions.
Literal nginx map keys are case-insensitive and must not replace these regexes.
An absent bearer is allowed only with a valid apikey. Any nonblank bearer must
carry that same apikey. Unknown, changed-case, duplicate and mismatched
credentials fail closed. Incoming role/JWT headers cannot select a role.

nginx replaces Authorization with an internally signed role JWT and removes
apikey before proxying. These internal tokens have no expiry, are scoped to
the dedicated signing key/audience/issuer and never leave the private
nginx-to-PostgREST hop. PostgREST verifies signature/audience and switches only
through the authenticator's reviewed SET memberships. PostgREST does not
independently validate the issuer string; the dedicated signer and private
hop provide that issuer boundary. It has no anonymous fallback. It exposes
only public schema REST/RPC; Auth, Storage and Realtime servers are not added.

`/rest/v1/` is removed exactly once at nginx. Other paths return 404. Requests
keep HTTP1, JSON, query filters, Prefer and SQLSTATE/status behavior. nginx
does not retry upstream requests, follow redirects or intercept SQL errors.
The 610s upstream wait preserves existing 605s publication and 65s bulk client
deadlines. There is no new nginx body cap: existing application/SQL contracts
own payload limits. Request and response buffering are disabled.

## Read-only database review before service start

The owner must create/review the authenticator separately. It must be LOGIN,
NOINHERIT, NOSUPERUSER, NOCREATEDB, NOCREATEROLE, NOREPLICATION and NOBYPASSRLS.
On PostgreSQL 17 its only SET memberships must be `anon`, `service_role` and
`lab_arena_service`, with INHERIT and ADMIN false. It must never SET
`lab_arena_owner`, `postgres`, `arena_admin` or a platform owner role.

These review queries contain no passwords and perform no writes:

```sql
SELECT rolname,rolcanlogin,rolinherit,rolsuper,rolcreatedb,rolcreaterole,
       rolreplication,rolbypassrls,rolconfig
FROM pg_roles
WHERE rolname IN ('anon','service_role','lab_arena_service',
                  'lab_arena_owner','leadpoet_subnet_authenticator');

SELECT granted.rolname AS granted_role,recipient.rolname AS recipient,
       m.admin_option,m.inherit_option,m.set_option
FROM pg_auth_members m
JOIN pg_roles granted ON granted.oid=m.roleid
JOIN pg_roles recipient ON recipient.oid=m.member
WHERE recipient.rolname='leadpoet_subnet_authenticator';

SELECT coalesce(r.rolname,'ALL') AS role,d.datname,s.setconfig
FROM pg_db_role_setting s
LEFT JOIN pg_roles r ON r.oid=s.setrole
LEFT JOIN pg_database d ON d.oid=s.setdatabase
WHERE r.rolname IN ('anon','service_role','lab_arena_service',
                   'leadpoet_subnet_authenticator') OR s.setrole=0;
```

Replace the example authenticator *name* above with the exact reviewed role
name. Preserve source ACLs/RLS and the exact `auth.role()` policy helper.
Review active role/database settings separately from schema restore metadata.
In particular, keep the Arena statement, lock and idle-transaction limits;
`SET ROLE` alone does not apply PostgreSQL login defaults, so prove the actual
PostgREST request behavior. After startup use `lab_arena_whoami` to confirm
current_user/JWT role, session_user and the NOLOGIN/non-BYPASSRLS request role.
Prove all negative ACL cases before production endpoint changes.

## Local verification

The existing integration fixtures run a disposable PostgreSQL 17.9 database,
current full-thinking SQL plus 575–579/581/583/584, PostgREST 14.5 and the actual
nginx adapter. Date-bound 580/582 production data repairs are excluded from
fresh fixtures. Positive requests use unchanged production clients with
opaque keys and `/rest/v1`; no Python JWT/path substitution is used.

```sh
python3 -m pytest -q tests/test_aws_subnet_api.py \
  tests/lab_arena/test_lab_arena_postgrest_route.py \
  tests/lab_arena/test_lab_arena_postgrest_round.py
```

Local tests do not prove RDS TLS, restored-source data/ACL parity, final
synchronization, public TLS routing, deployed releases or real production
workflow completion. Those remain owner verification gates.
