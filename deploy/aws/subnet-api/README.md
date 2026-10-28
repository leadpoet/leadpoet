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

- `arena_key`: existing Arena `sb_secret_` key; maps to `lab_arena_service` on RDS.
- `gateway_key`: separate existing gateway `sb_secret_` key; maps to `service_role` on RDS.
- `public_key`: existing `sb_publishable_` key for public RPCs; maps to `anon` on RDS.
- `jwt_secret`: dedicated high-entropy signing secret, at least 32 URL-safe
  ASCII characters. Retain the same dedicated AWS secret on ordinary app
  deploys. Do not regenerate it as part of app startup.
- `db_uri`: percent-encoded libpq URI for the dedicated least-privilege
  authenticator. Production must use the exact RDS host/database and
  `sslmode=verify-full&sslrootcert=/run/subnet-api/rds-ca.pem`. The CLI
  requires these TLS settings. Set `SUBNET_API_CA_FILE` to the reviewed AWS
  CA file for compose's read-only mount; neither tool retrieves it.

Prepare both modes from **one pinned target secret version**, with the exact
same five-field document and reviewed RDS CA. The default `--rest-backend target`
uses nginx and the dedicated RDS PostgREST connection. `--rest-backend source`
forwards nginx requests to the existing managed REST endpoint at
`https://qplwoislplkcegvdmbim.supabase.co/rest/v1/`. Source mode leaves the
PostgREST configuration bound to the same RDS URI and CA; that connection is
unused by nginx in source mode. It does not use a source database login, source
CA or separate source secret version. Managed Supabase reserves the required
role memberships, so do not attempt to provision a source authenticator.

Keep the original source public API and existing client keys available. Before
source-first activation or rollback, the owner must prove the three original
apikey-only source identities, public epoch RPC, required private reads and
negative ACL cases. The original source public RPC and anon read permissions
must remain available to retained validators/auditors that still call it.
A rollback still requires writer drain, verified reverse synchronization and
proof that the source is authoritative before selecting its REST mode.

Hydrate the retained target input through the existing protected secret
workflow. Keep input and output outside Git. No secret belongs in argv, compose
environment variables, terminal output or Docker logs. The reviewed host
preparation can fetch the pinned target version into memory. For a protected
file-based render, use the same private target input for both directories:

```sh
python3 scripts/aws_subnet_api.py \
  --secrets-file /run/leadpoet-subnet-api-target-input.json \
  --output-dir /etc/leadpoet-subnet-api/target \
  --owner-uid 1000 --owner-gid 1000 --rest-backend target
python3 scripts/aws_subnet_api.py \
  --secrets-file /run/leadpoet-subnet-api-target-input.json \
  --output-dir /etc/leadpoet-subnet-api/source \
  --owner-uid 1000 --owner-gid 1000 --rest-backend source
```

Create the private parent first. Prepare both directories, but run only one
compose pair at a time. Select the reviewed directory through
`SUBNET_API_CONFIG_DIR`; both selections use the same reviewed RDS CA through
`SUBNET_API_CA_FILE`. Place the same reviewed RDS CA in each directory as
`db-ca.pem`, owned by UID/GID 1000 with mode 0600. Keep `db_uri` bound to RDS.

The output directory is mode 0700; both configuration files are mode 0600 and
owned by the runtime UID/GID. The renderer emits only a fixed success/error
line. Delete the temporary input after the protected workflow verifies the
output. Never use `nginx -T`, PostgREST dump-config or verbose Docker inspection
to print these files. Container log drivers are disabled; nginx access/error
logs are disabled. Use status, exit codes and bounded authenticated capability
checks for diagnosis. Inspect a failed private configuration only through the
same redacted operator workflow.

Where a Compose plugin is already available, the owner can review interpolation
without secrets because it contains only private file paths:

```sh
SUBNET_API_CONFIG_DIR=/etc/leadpoet-subnet-api/source \
  SUBNET_API_CA_FILE=/etc/leadpoet-subnet-api/source/db-ca.pem \
  docker compose -f deploy/aws/subnet-api/compose.yaml config --quiet
```

The gateway has Docker CLI without a Compose plugin. Do not install a plugin
for this migration. `compose.yaml` remains the reviewed container contract;
local checks compare the native Docker argument lists with every service field.
The owner can start its equivalent pair with the reviewed commands below after
host preparation and RDS identity checks. These commands select source REST;
select the prepared `target` directory to use RDS REST instead. All credential
values remain in the private files.

```sh
SUBNET_API_CONFIG_DIR=/etc/leadpoet-subnet-api/source
SUBNET_API_CA_FILE="$SUBNET_API_CONFIG_DIR/db-ca.pem"
docker run --detach --name leadpoet-subnet-api-postgrest-1 \
  --user 1000:1000 --network host --read-only --cap-drop ALL \
  --security-opt no-new-privileges:true --log-driver none \
  --restart unless-stopped --stop-timeout 620 \
  --volume "$SUBNET_API_CONFIG_DIR/postgrest.conf:/run/subnet-api/postgrest.conf:ro" \
  --volume "$SUBNET_API_CA_FILE:/run/subnet-api/rds-ca.pem:ro" \
  postgrest/postgrest:v14.5@sha256:b574528fe109c8343c1247155734d03df8c34b462f342dca0ccc20244fc36ef9 \
  postgrest /run/subnet-api/postgrest.conf
docker run --detach --name leadpoet-subnet-api-nginx-1 \
  --user 1000:1000 --network host --read-only --cap-drop ALL \
  --security-opt no-new-privileges:true --log-driver none \
  --restart unless-stopped --stop-timeout 620 \
  --volume "$SUBNET_API_CONFIG_DIR/nginx.conf:/run/subnet-api/nginx.conf:ro" \
  --tmpfs /tmp:uid=1000,gid=1000,mode=0700 --entrypoint nginx \
  nginx:1.28.3-alpine@sha256:a8b39bd9cf0f83869a2162827a0caf6137ddf759d50a171451b335cecc87d236 \
  -c /run/subnet-api/nginx.conf -g 'daemon off;'
```

Keep the fixed container names. The owner must drain and stop the existing pair
before selecting another mode. Prove the running image digests, host network,
UID/GID, read-only mounts and selected private configuration hashes. Complete
cold private HTTP identity/ACL checks before public routing or client changes.
Rendering or a local syntax check does not start either service.

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

In target mode, nginx replaces Authorization with an internally signed role
JWT and removes apikey before proxying. These internal tokens have no expiry,
are scoped to the dedicated signing key/audience/issuer and never leave the private
nginx-to-PostgREST hop. PostgREST verifies signature/audience and switches only
through the authenticator's reviewed SET memberships. PostgREST does not
independently validate the issuer string; the dedicated signer and private
hop provide that issuer boundary. It has no anonymous fallback. It exposes
only public schema REST/RPC; Auth, Storage and Realtime servers are not added.

In source mode, nginx forwards the exact original apikey after the same key gate.
It strips Authorization, Cookie and spoofable JWT headers. No private target JWT
reaches the source. The HTTPS upstream has the fixed source Host and SNI name,
with certificate-chain and hostname verification against the container's system
CA bundle. The RDS CA is used only by PostgREST; it is not the HTTPS trust store.

Target mode removes `/rest/v1/` exactly once. Source mode preserves the original
`/rest/v1/` path and query. `/rest/v1` without its trailing slash has nginx's
local 301 redirect to `/rest/v1/`; other routes return 404. Requests keep HTTP1, JSON, query filters, Prefer and SQLSTATE/status behavior. nginx
does not retry upstream requests, follow redirects or intercept SQL errors.
The 610s upstream wait preserves existing 605s publication and 65s bulk client
deadlines. There is no new nginx body cap: existing application/SQL contracts
own payload limits. Request and response buffering are disabled.

## Read-only database review before service start

The owner must create/review the authenticator on RDS separately. It must be LOGIN,
NOINHERIT, NOSUPERUSER, NOCREATEDB, NOCREATEROLE, NOREPLICATION and NOBYPASSRLS.
On PostgreSQL 17 its only SET memberships must be `anon`, `service_role` and
`lab_arena_service`, with INHERIT and ADMIN false. It must never SET
`lab_arena_owner`, `postgres`, `arena_admin` or a platform owner role.

Run these password-free, read-only review queries on the RDS target:

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
name on RDS. Preserve the restored ACLs/RLS and exact `auth.role()` policy helper.
Source REST keeps its existing platform-managed authentication and ACLs.
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
  tests/test_aws_subnet_api_source.py \
  tests/lab_arena/test_lab_arena_postgrest_route.py \
  tests/lab_arena/test_lab_arena_postgrest_round.py
```

The source-fallback tests use pinned nginx and a private HTTPS wire stub. They
prove key gating, header removal, request/response bytes, SNI and certificate
verification. The stub does not establish actual managed-source key roles or ACLs;
the owner must verify those with cold source reads before activation.

Local tests do not prove RDS TLS, restored-source data/ACL parity, final
synchronization, public TLS routing, deployed releases or real production
workflow completion. Those remain owner verification gates.
