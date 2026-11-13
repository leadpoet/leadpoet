# Routine Arena RDS migrations

Use `scripts/arena_rds_migration.py` for reviewed, merged migrations with the
normal `leadpoet-lab` profile. It does not read the RDS master secret and has no
master-login fallback. It requires an existing localhost tunnel to the fixed
Arena RDS endpoint and the verified regional RDS CA file. Tunnel setup is a
separate authorized operation. The helper does not start or stop shared tunnels.

## One-time access setup

A separately authorized administrator creates `arena_migrator` with a generated
password stored only in Secrets Manager. Do not put passwords in SQL files,
command arguments, logs or Git. First apply the reviewed
[`692-arena-migration-definer-boundary.sql`](../scripts/692-arena-migration-definer-boundary.sql)
and verify that `lab_arena_owner` cannot execute a non-system security
definer owned by another role. The November 13 audit found 17 legacy functions
executable by PUBLIC, all owned by `postgres`. Granting schema CREATE before
closing that access would not establish the intended authority boundary. Keep
existing explicit `anon`, `authenticated`, and `service_role` grants unchanged.
The helper refuses access if this boundary later changes. These are the required
non-secret grants after that check passes:

```sql
CREATE ROLE arena_migrator LOGIN NOINHERIT NOSUPERUSER NOCREATEDB
  NOCREATEROLE NOREPLICATION NOBYPASSRLS;
GRANT lab_arena_owner TO arena_migrator
  WITH ADMIN FALSE, INHERIT TRUE, SET TRUE;
GRANT CONNECT, TEMPORARY ON DATABASE postgres TO arena_migrator;
GRANT USAGE ON SCHEMA public TO arena_migrator;
GRANT CREATE ON SCHEMA public TO arena_migrator WITH GRANT OPTION;
```

Set the generated password through a private parameterized setup operation.
Do not grant `postgres`, `arena_admin`, `rds_superuser`, replication, platform
owner roles or role-membership administration. Do not change application owners,
request roles, the authenticator, current table ACLs or RLS.

The dedicated secret name is `leadpoet/arena/migrations/pranav`. Its JSON has
exactly `username`, `password`, `host`, `port`, `dbname`: username
`arena_migrator`, fixed RDS hostname
`leadpoet-subnet-arena.cgfmio6maf8s.us-east-1.rds.amazonaws.com`, numeric port
`5432`, database `postgres`, and the generated password. Restrict normal Pranav
secret-read access to its exact returned ARN and the verified normal principal.
Preserve scoped RDS metadata reads and existing authorized transport access.
After this channel passes live verification, remove its routine master-secret
read permission. Do not remove that permission before the replacement works.
Password authentication avoids an RDS configuration change to enable IAM DB
authentication. A dedicated IAM login can be evaluated separately if needed.

## Verify and apply

Start from a clean checkout that includes current main. Supply a full merged
commit SHA, the exact numbered `scripts/*.sql` path and its reviewed SHA-256.
The helper fetches main from the fixed repository URL, checks commit ancestry,
and checks local bytes. Verification needs Git/network access but no database
credential:

```bash
python3 scripts/arena_rds_migration.py --verify-only scripts/NNN-name.sql \
  --commit FULL_SHA --expected-sha256 REVIEWED_HASH
python3 scripts/arena_rds_migration.py --check-access \
  --local-port LOCAL_TUNNEL_PORT --ca-file /path/to/rds-us-east-1-bundle.pem
python3 scripts/arena_rds_migration.py --apply scripts/NNN-name.sql \
  --commit FULL_SHA --expected-sha256 REVIEWED_HASH \
  --local-port LOCAL_TUNNEL_PORT --ca-file /path/to/rds-us-east-1-bundle.pem
```

Before apply, test the exact SQL in disposable PostgreSQL, verify the live schema
gap, and complete the normal review and release checks. After apply, verify the
live schema and affected behavior. A success response alone does not prove the
migration's postconditions. The helper sends unchanged SQL on one connection
with driver autocommit enabled. Explicit `BEGIN`/`COMMIT` remains authoritative;
standalone `CREATE INDEX CONCURRENTLY` is not wrapped in an extra transaction.
An error can leave partial effects for nontransactional SQL. Inspect its live
state before retrying. Do not split or rewrite reviewed SQL during apply.

## Authority and exclusions

This is full ownership authority over `lab_arena_owner` objects and their data.
It includes function, ACL, trigger and RLS changes. The schema CREATE grant also
permits creating public-schema objects and delegating CREATE. Existing migrations
need its grant option to temporarily grant/revoke CREATE for `lab_arena_owner`.
Run SQL as the migration login with inherited owner rights; `SET ROLE
lab_arena_owner` would lose the login's schema grant option. New persistent
objects must end owned by the reviewed application owner. Check owners and ACLs
afterward. The migration login must remain unable to execute other owners'
non-system security definers. This catalog check supplements privilege review;
it does not make arbitrary application-owner SQL safe.

The Git/path/hash gates prevent routine mistakes. They are not a security barrier
against a person who controls the login or helper. This design does not claim to
permit arbitrary DDL while preventing data access, or enforce reviewed SQL only.

Changes to other owners' objects, schemas, database settings, extensions, roles,
memberships or authentication require a separately reviewed privileged operation.
Do not add owner memberships when a migration fails. In the inspected migration
range 575–688, examples that need this separate authority include:

- `593-lab-arena-capture-summary-reader.sql`: creates/alters a role and can grant
  its membership. Re-running it still performs `ALTER ROLE`.
- `667-lab-arena-atomic-icp-bank-publication.sql`: changes the shared qualification
  table/index and publication function. Do not grant the broad `postgres` owner
  role to make it work.

PostgreSQL privileges enforce this boundary for every migration; there is no
migration-number allowlist or denylist. The helper reports insufficient SQL
privilege without printing SQL, passwords or provider errors. It never retries
or replays SQL automatically. A transaction already committed by a script is not
rolled back by a later failure or connection close. Historical migrations outside
that range have not been classified. Use the existing separately authorized
channel for an exception.
