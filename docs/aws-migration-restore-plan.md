# RDS restore plan: review required

The target is only `leadpoet-subnet-arena.cgfmio6maf8s.us-east-1.rds.amazonaws.com`,
database `postgres`. Preserve the source, protected `SOURCE_ADD` history and
production endpoints. No application uses this target yet. This plan does not
authorize a production cutover or service restart.

`scripts/aws_subnet_restore.py` has two operations: `inventory` reads the target;
`plan` reads local evidence and prepares private review files. It has **no
execute, restore, role creation, schema change or ACL-write operation**. The
native restore command in its plan is a candidate for later reviewed execution.
Do not run that command until the gates below pass.

## Evidence and remaining blockers

After runner bootstrap, install the official Supabase source CA separately at
`/etc/leadpoet-migration/supabase-prod-ca-2021.crt`, owned by root, mode 0644.
The system CA bundle alone failed certificate verification for this source
session pooler. Keep `sslmode=verify-full`; do not use an encryption-only fallback.
The coordinator verified the CA URL from Supabase Studio's custom-content
configuration:
[official Supabase CA](https://supabase-downloads.s3-ap-southeast-1.amazonaws.com/prod/ssl/prod-ca-2021.crt).
Download it over verified HTTPS into a temporary file, check its SHA-256 before
installation, then check the certificate fingerprint and expiry:

- File SHA-256: `700723581420dd1ac98fd7e9ac529f0ef210eadcaf87fc868a3ad7d114c2f3b7`.
- Certificate SHA-256: `807025AD50D4ED219D2C9C7D299C004F824EB00CF7F65AFEF607D07B72E6CAFA`.
- Expiry year: 2031; confirm the actual certificate validity at installation.

The deployed infrastructure template is unchanged. The target continues to use
the official regional RDS CA bundle. Prove actual hostname-validated source and
target connections before export or target writes.

The source session pooler discards startup `options`. The utility therefore
sets and checks its read-only transaction settings with explicit SQL, including
row security off and canonical COPY serialization. It imports a verification
snapshot before applying transaction settings and reading rows. Keep the strict
identity/read-only checks. PostgreSQL 17 pg_dump explicitly sets ISO dates,
postgres intervals, float precision and row-security behavior, but inherits
timezone and bytea output. Export checks the untouched connection defaults are
UTC and hex before application reads, records them privately, and uses explicit
UTF8/float dump options. An unexpected default blocks export; do not change the
source role or weaken verification as an automatic workaround.

Source `pg_stat_ssl` describes the pooler-to-PostgreSQL connection. It does not
measure the separately certificate-verified client-to-pooler TLS connection.
Do not infer client TLS failure from that backend-hop field or change source
SSL enforcement. No source role or server setting change is part of this fix.

The source archive must contain 48 table-data entries, 3 sequence-state entries,
3 views and 377 nonextension function/procedure entries. The sealed snapshot
manifest and all archive-file hashes must match. At least one table must be in
`source_add_history`. Never remove application/history tables or their ACLs
from a restore list to make a failed restore pass.

Capture fresh target evidence with `inventory --target-secret-arn ARN --output
NEW_PRIVATE_DIRECTORY`. It uses the snapshot utility's exact endpoint, master
identity, `verify-full`, RDS CA, temporary service/password files and read-only
transactions. Only a fixed success/failure message may reach SSM. The catalog
evidence includes roles and memberships, extension availability and installed
versions, sessions, schemas, owners, ACLs, columns/defaults, indexes,
constraints, view/routine definitions, RLS policies and flags, triggers, sequence
definitions/states, enum/domain types and default privileges. Definitions and
logs stay in root-owned mode-0600 files under a mode-0700 directory. Inventory
contains no application rows or password verifiers.

Prepare local review files with `plan --export-directory EXPORT
--target-inventory INVENTORY --output NEW_PRIVATE_DIRECTORY`. An optional
`--exclusions FILE` contains exact TOC IDs, SHA-256 of each full TOC line and a
nonempty review reason. Only the unavailable `http` extension entry, its
comment, and the exact `SCHEMA - public postgres` or
`SCHEMA - public pg_database_owner` creation/ownership entry may be proposed
for exclusion. The public entry's owner must match the single `public` record
in sealed `manifest.schema_metadata.schemas`. Other owners and missing,
duplicate or mismatched source records are rejected. The public-schema ACL
entry must remain. The tool cannot omit a table, sequence,
view, routine, index, constraint, policy, trigger or mixed ACL entry. It records
the exact manifest, TOC, inventory and exclusion identities.

The output includes `source-schema.sql`, an unchanged private schema script
from the archive; `restore.list`; the archive's actual `setval` statements and
parsed sequence states; an exact temporary-role cleanup SQL candidate; and
`plan.json`. All native command diagnostics stay private. This is preparation,
not proof that the target can run the application.

The plan remains blocked until these facts are established:

- A fresh target catalog proves the required extensions and versions are
  available, the target contains no application objects/default privileges,
  and no other session can write. The October 28 target inventory reports
  PostgreSQL `170009`, with `dblink` 1.2, `pg_stat_statements` 1.11, `pg_trgm`
  1.6, `pgcrypto` 1.3 and `uuid-ossp` 1.1 available; `http` is unavailable.
  Its preload list is `rdsutils,pg_tle,pg_stat_statements,rds_casts`. Recheck
  this evidence immediately before any target write. Verify
  `shared_preload_libraries` and actual `pg_stat_statements` use. An installed
  extension alone does not prove its functions work.
- Exact archived signatures, types and routine bodies have been checked for
  extension dependencies. Excluding `CREATE EXTENSION http` cannot supply its
  types or functions. Do not invent placeholder types or replacement network
  behavior. Preserve retained definitions and report unsupported functionality.
- The role bootstrap, membership authority and existing empty `public` schema
  have an exact reviewed treatment. Preserve the initialized empty schema and
  its initial ACLs. Derive the source owner from the sealed
  `manifest.schema_metadata.schemas` public record and bind it to the exact
  native creation/ownership TOC entry. The October 28 source observation and
  fresh RDS inventory both use `pg_database_owner`; the initialized target ACL
  is `pg_database_owner=UC/pg_database_owner` and `=U/pg_database_owner`.
  When the target owner and initial ACL baseline already match the source,
  retain that schema without an owner change. If the sealed source instead
  uses the historical `postgres` owner, review only that exact owner change
  before restore. Do not unconditionally change the owner to `postgres`.
  Exclude only the reviewed creation/ownership entry and retain its native ACL
  entry. The local PostgreSQL 17 controls cover both owner variants and compare
  restored ownership and ACLs exactly. Dropping and recreating `public`
  loses the default PUBLIC USAGE grant because native pg_dump assumes that
  initialized baseline. The helper performs neither change. Do not add
  `--clean` as a workaround.
- The sealed export includes `manifest.schema_metadata`. The snapshot utility
  captures all 12 canonical catalog categories through the same keeper and
  transaction snapshot as its table counts and native dump. Restore planning
  rejects an old manifest without this reference and saves a private copy as
  `source-metadata.json`. Use `snapshot.capture_metadata` on the restored target
  and `compare_metadata` for exact category comparison. Review every intentional
  platform difference and removal of only the temporary export-role ACLs.
  Never substitute a later moving source catalog for this captured reference.
- The coordinator supplies a separately reviewed write-enabled target service
  wrapper and checks all target identity/empty-state gates immediately before
  its first write. The existing snapshot service is deliberately read-only.

## Role and ownership treatment

Keep owner names `postgres` and `lab_arena_owner`. Keep ACL grantee names `anon`,
`authenticated`, `lab_arena_owner`, `lab_arena_service`, `leadpoet_parity_reader`,
`pg_database_owner`, `postgres`, `service_role` and `supabase_admin`.
Do not copy source passwords, logins, superuser, replication, role/database
creation or bypass-RLS privileges into this rehearsal target.

Create missing platform owner/ACL names as inert `NOLOGIN NOBYPASSRLS` roles,
without administrative flags. Preserve the two active lab roles as `NOLOGIN
NOINHERIT NOBYPASSRLS`. Preserve `NOINHERIT` for the parity name. Other inert
platform names use `INHERIT`. The built-in `pg_database_owner` already exists;
verify it rather than trying to create or modify a reserved role. These are
explicit rehearsal security changes from the hosted platform role attributes.
They do not prove future client authentication or service-role behavior.

The restoring operator must have reviewed `SET` and inheritance authority for
both owner roles. Do not assume RDS `rds_superuser` is PostgreSQL superuser.
The native non-superuser control also needs temporary database CREATE grants
to the two owner roles for schema ownership changes. Review those prerequisites,
then revoke only temporary database CREATE grants after restore and verify
the database ACL against its saved target baseline. When `public` remains owned
by `pg_database_owner`, native ownership restoration can also require temporary
CREATE on that schema for inert object-owner roles such as `postgres` and
`lab_arena_owner`. Schema ownership alone does not give those roles CREATE.
Review the exact prerequisite grants against the source manifest and saved
target baseline. After restore, remove only added privileges absent from the
intended source ACL; preserve every native source grant. Do not use a blanket
REVOKE that removes source permissions. The native control compares the full
owner and ACL metadata after this narrow prerequisite cleanup.
Verify actual owner changes, grantors, security-definer execution and privileges
on a native control and then on this target. Missing grantor/owner authority is
a blocker, not a reason to discard ownership or ACLs.

The temporary `leadpoet_aws_export_20261028` name may occur in ACL records that
also contain legitimate grants. Precreate only an inert `NOLOGIN NOINHERIT
NOBYPASSRLS` role, with no inherited memberships or owned objects, so native ACL restoration
can succeed. It must never become a target login or bypass identity. After the
restore, use the reviewed `revoke-export-role.sql` candidate to revoke only its
SELECT grants on the exact 48 tables/3 sequences and USAGE on the 3 schemas,
then drop that inert role in the same transaction. Its guards reject unsafe
attributes and ownership. The sole allowed membership is PostgreSQL 17's
automatic creator grant to `arena_admin`: ADMIN true, INHERIT false, SET false.
Keep that authority until `DROP ROLE` removes the inert role and its creator
membership. Do not revoke it first or grant the operator inherited/SET access
to the export role. Any other membership is rejected. Unexpected residual dependencies make
`DROP ROLE` fail and roll back the cleanup. Do not use `DROP OWNED`, `CASCADE`,
blanket `--no-acl` or exclusion of mixed ACL TOC entries.

## Native restore phases

1. Bind the reviewed code, export hashes, target inventory, role SQL and exact
   TOC selection. Obtain independent review of the remaining platform changes.
2. Recheck the dedicated target and empty state. Apply only the reviewed role,
   extension and empty-schema prerequisites. Do not modify source roles here.
3. Run the exact native `pg_restore` candidate with `--exit-on-error`,
   `--single-transaction` and the reviewed `--use-list`. Keep native owner and
   ACL restoration enabled. Serial restore gives one atomic database restore;
   parallel restore is not combined with `--single-transaction`. Save private
   stderr and the exit status. On failure, inspect the evidence and target state;
   do not retry against a partially changed target or broaden exclusions.
4. Audit and remove only the inert export-role grants. Compare ownership, ACLs,
   RLS, routines, views, indexes and constraints against the archived/reference
   evidence. Record each approved platform difference explicitly.
5. Run `aws_subnet_snapshot.py verify-data` to compare all 48 table counts and
   full-row multiset checksums against the captured archive. This includes
   protected `SOURCE_ADD` data. Compare the 3 target `last_value`/`is_called`
   pairs to `archive-sequences.json`, **not** to the earlier snapshot manifest
   sequence observation or a later moving source. Sequence counters are not
   transactional source snapshots.

Every live inventory/restore/verification command must run under a root systemd
unit with `RuntimeMaxSec=14400`, `TimeoutStopSec=30`, `KillMode=control-group`
and `UMask=0077`. Verify effective properties. Precreate root-only mode-0600
output/error files under a mode-0700 directory. Return only sanitized status to
SSM. Unlimited SQL statement timeouts require this external four-hour deadline.
Keep passwords in temporary private libpq files; never arguments, environment
dumps, SSM payloads or logs. A timeout requires a fresh target-state audit.

## Compatibility limits

The current source audit found 7 retired `postgres` functions that use
`auth.jwt`, `auth.uid` or `auth.users`, and 2 retained definitions that use
`net.*`. Keep their definitions and report their unsupported functionality.
An `auth.*` alias in an active function is not itself a dependency. Inspect
actual definitions, bound object dependencies and reachable calls; a text match
alone does not prove an active dependency or successful operation.

Do not declare active client/API compatibility, external HTTP behavior,
authentication, production cutover or recovery complete from this restore.
Those remain separate stages. Preserve the encrypted dump and evidence, verify
an independent encrypted backup, and remove temporary source export access only
after the coordinator completes the transfer verification.
