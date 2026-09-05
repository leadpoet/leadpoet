# Production-parity validation

This system validates the workflows that need production-scale evidence: the
complete daily rebenchmark, canonical primary/audit weight handling, and the
two bounded miner-intake admissions required for release confidence. It does
not create persistent staging miners, wallets, authorities, dashboards, or a
permanent staging fleet.

## What remains identical

Both lanes are driven by one exact Git commit. Product behavior is imported
from that commit rather than copied into a staging controller:

- SQL migrations and PostgREST/RPC contracts;
- real production-shaped database rows and large historical responses;
- provider credentials, immutable model artifacts, and scoring code;
- gateway restart, Nitro enclave, attestation, PCR0, and exact-commit checks;
- all configured ICP scoring, retry, aggregate, assignment, and persistence;
- canonical allocation and bundle construction;
- primary and audit parsing, verification, signing, SDK encoding,
  finalization, fallback, and readback behavior.

The only substitutions are explicit external safety boundaries. Production
Supabase is read through a dedicated read-only PostgreSQL role and restored to
disposable PostgreSQL. The candidate gateway writes only to that clone.
Production Finney reads remain real. The final chain broadcast is stopped at
the existing strict non-forwarding adapter after the exact signed submission
request has been built; primary and audit evidence must use the same canonical
bundle.

This proves the application path up to the irreversible network boundary. It
does not claim that an external chain included a transaction. A public testnet
would be needed only when external inclusion/finality itself must be tested.

## Fast lane

After every push to main, `Production Parity Fast` runs in parallel with
attestation and targets 5-10 minutes. It:

1. resolves live N-1 from the production gateway;
2. takes an ephemeral schema-only snapshot through the dedicated read-only
   production role and records the live relation/response scale;
3. restores that exact schema to disposable PostgreSQL and applies candidate
   migrations in production order;
4. exercises candidate-generated measured-source reads against the real
   production PostgREST data through a strict GET-only, no-body, no-redirect
   adapter, including the complete historical weight-input range;
5. exercises the candidate's real PostgREST/RPC contracts on the disposable
   database;
6. runs the candidate-derived N-1 gateway/validator/auditor rehearsal; and
7. validates canonical-bundle equality, signing, finalization, readback,
   Git-tree, ICP, settlement, retry, and cleanup contracts.

The candidate contract's independent source commitments include the exact
miner signing helpers, intake models and routes, and SOURCE_ADD miner helper.
Both lanes verify those commitments
against the candidate Git blobs and checkout. This prevents stale evidence
without changing measured runtime identity or PCR0. The full lane exercises
SOURCE_ADD admission through the candidate's measured provider path.

No production rows are copied in the fast lane, and no database dump is
uploaded as an artifact. The schema archive is destroyed on every exit path;
only redacted hashes, counts, sizes, and stage evidence are retained. The
strict adapter cannot issue POST, PATCH, PUT, DELETE, follow redirects, send a
request body, or target any host other than the pinned production Supabase
origin.

## Full lane

`Production Parity Full` starts after the exact commit's normal attested
release succeeds. It dynamically:

1. requires the exact commissioned gateway instance, AMI, subnet, VPC, and
   instance type and fails if live readback differs;
2. creates one encrypted, IMDSv2-required, Nitro-enabled EC2 instance;
3. creates one temporary CloudFront HTTPS origin for cloned PostgREST;
4. creates one run-scoped, encrypted, Object-Locked bucket and transfers the
   exact Git bundle through that bucket;
5. captures production through the dedicated read-only DSN and restores it
   only on the encrypted transient volume;
6. creates one run-scoped gateway secret that retains real provider and model
   reads but redirects every mutable Research Lab write to the clone;
7. runs the candidate's exact `gw_restart.sh --commit` and requires matching
   build, attestation, PCR0, and V2 readiness;
8. keeps production writes and fulfillment disabled while running the public
   PydanticAI baseline in a clone-local Arena shadow round;
9. executes and scores all twenty daily ICPs, restarts the Arena service and
   runner between stages, and checks recovery, persistence, and publication;
10. fetches and verifies the real gateway allocation handoff with production
    validator integration; and
11. hash-binds that verified allocation document into the exact candidate
    primary/audit signing and submission path, then requires both validators
    to consume the same canonical vector through the strict non-forwarding
    chain boundary; and
12. creates one in-memory ephemeral miner identity and exercises the exact
    candidate SOURCE_ADD HTTP request models, signatures,
    routes, measured credential verification, PostgREST/RPC calls, and durable
    writes against the clone.

The Arena uses the organizer's runtime API keys for billable provider calls.
The clone secret excludes OpenRouter management credentials. OpenRouter
requests require `data_collection=deny` and `zdr=true`; the lane does not
change workspace logging or create, rotate, or delete provider keys.
Production Supabase and the chain remain read-only.

SOURCE_ADD intentionally has a different production contract: miners submit
credential-free source proposals, while an operator adds any provider
credential later through the measured administration path. The lane first
makes one bounded read-only request to BuiltWith's official Domain API, using
its documented `Authorization: API ...` header so the key never enters a URL,
to prove the configured credential. It then submits the BuiltWith metadata
through the exact credential-free miner route. It requires `provenance_queued` plus one
unclaimed queue item, proves no downstream SOURCE_ADD work ran, and verifies
that both the retired public credential-recipient route and direct credential
injection still fail closed. Accepting the BuiltWith key in a miner request
would be staging-only behavior and is deliberately forbidden.

The ephemeral hotkey has no chain identity. The only isolated intake adapter
therefore replaces the external registration lookup for that one exact
in-memory hotkey; all signatures, ban checks, request validation, enclave
attestation, provider authentication, persistence, and fail-closed behavior
remain production code. It rejects every other hotkey and never reaches a
chain write.

The workflow immediately deletes its instance, volume, security group,
CloudFront distribution, run secret, and local database dump. The transient
bucket uses one-day S3 COMPLIANCE retention, so its candidate bundle and
redacted evidence cannot be deleted early; the scheduled cleanup job deletes
their versions and the bucket after retention expires. The same cleanup job
removes only stale resources bearing the exact run, candidate, and ephemeral
ownership tags after hard cancellation.

## One-time prerequisites

There is no GitHub Environment and no testnet setup. Commissioning is one
exact-commit operation and remains disabled until every prerequisite has been
read back. It creates only:

- one Secrets Manager record containing that DSN;
- one separate read-only-to-the-runner secret containing the BuiltWith key;
- a GitHub OIDC controller role;
- one restricted EC2 runner role and instance profile; and
- the exact repository-variable inventory for fixed production resources and
  freshly resolved official PostgreSQL/PostgREST image digests.

The standing controller trust accepts only the three named parity workflows
on `leadpoet/leadpoet` main. Its permissions are four fixed, parity-owned
customer-managed policies (EC2 launch, lifecycle/SSM, CloudFront, and data),
each below the IAM managed-policy quota. Commissioning makes the role inert,
revokes older sessions, detaches all permissions, replaces and reads back the
sole policy versions, reattaches the complete set, and runs IAM's principal
policy simulator against positive and adversarial resource/tag contexts before
restoring trust. The runner follows the same inert/session-revocation boundary.
That simulator proves the identity-policy lattice; it does not claim that a
compound AWS API propagated every context key, so the real provisioner still
fails closed on every API error and exact resource/tag readback.
RunInstances is pinned to the exact AMI, subnet, VPC, runner profile, instance
type, IMDSv2 settings, and encrypted 512-GiB gp3 shape. SSM, EC2 lifecycle, and
CloudFront mutation are limited to exact account resources carrying all parity
run, candidate, ephemeral, and Name ownership tags.

### IAM change boundary and recovery

A public baseline or Research Lab source release is not by itself an IAM
change. Ordinary commits and bundle versions that retain the
declared AWS actions, resources, account, region, roles, attachments,
permissions boundary, prefixes, and tag conditions reuse the existing
authority without policy mutation. A release needs a new IAM plan only when it
intentionally changes one of those declared AWS capabilities or its ECR, KMS,
S3, Secrets Manager, or role layout. Exact artifact and commit admission stays
fail closed without coupling policy changes to every source release.

The August 2026 rebenchmark recovery exposed mechanism defects, not missing
operator access and not a baseline commit failure. First, AWS can return
`MissingContextValues` at both aggregate and resource-specific levels,
including keys from statements that do not apply to the simulated action. For
managed parity-controller changes, the commissioner evaluates those keys
against the complete live principal-policy inventory: unknown or
action-applicable missing context fails closed, while a known key belonging
only to action-inapplicable statements does not create a false denial.

Second, the local bridge once sent its validated internal request projection,
including derived precondition fields, where the remote trust boundary accepts
only the public request contract. After caller-identity verification, the
remote correctly rejected that shape before any target-policy read or write,
but the rejection surfaced as a generic SSH failure. The bridge now has one
explicit lossless serializer from the internal projection back to the public
wire contract, reparses that wire form locally, and then lets the remote
validate the same strict contract again. A regression test asserts the exact
field inventory and round trip. Internal convenience fields must never expand
or loosen the remote request surface.

If exact `origin/main` advances while an IAM intent remains active, the typed
ledger appends the fresh exact-main authority route without rewriting the
intent or its plan. The current commissioner may reconcile the plan only
against its exact retained route. A commit-only advance with the same exact
operator/setup closure hash is IAM-authority-equivalent: reconciliation may
verify exact-applied state with simulations and emit a current-route receipt,
but it performs no new policy write. If that closure hash changed, remote
historical mode is before-only: it reads stable live inventory, returns
`before` only for the exact plan base, classifies every desired, staged,
unstable, or third state as ambiguous, and performs no simulation, cleanup,
policy write, or historical apply. This prevents an unrelated release from
orphaning an in-flight intent without letting changed code reinterpret an old
write authority.

Each durable IAM operation ID permits at most one remote apply dispatch for its
entire lifetime, including across authority refreshes and later source reverts.
Any pending or reconciliation record permanently bars another remote apply;
an exact validated policy outcome is monotonic and may only be replayed
locally, byte-for-byte, against its retained route. That zero-remote replay is
not a historical apply and must never be overwritten by weaker inventory
evidence. A newer ledger generation may supersede other stale operation
evidence only with another read-only reconciliation. Changed-authority remote
reconciliation accepts only its dedicated before/ambiguous receipt schema,
never an applied-policy receipt.

Authority and reconciliation inventory reads use bounded retries, and all
failures cross the bridge only as fixed typed diagnostics; they never trigger a
request for operator credentials or manual IAM work. For managed
parity-controller changes, the complete principal inventory is revalidated
immediately before each forward managed-policy write and after activation.
Drift before activation removes only the exact plan-bound staged version under
guarded readback. If an apply outcome is unknown, the same ledger-bound intent
must be reconciled to exact-before, exact-applied, or ambiguous state; it must
never be blindly reapplied.

Migration `156-production-parity-readonly-role.sql` is never pasted into an SQL
editor and the repository bootstrap does not apply arbitrary SQL. During an
explicitly authorized overnight rebenchmark run, the Keychain-backed skill
helper applies the exact numbered migration only after its path, current
`origin/main` commit, and SHA-256 are verified:

```bash
LEADPOET_OVERNIGHT_REBENCHMARK_AUTHORIZED=1 \
python3 ~/.codex/skills/overnight-rebenchmark-validation/scripts/apply_supabase_migration.py \
  --apply <canonical-repo>/scripts/156-production-parity-readonly-role.sql \
  --commit <current-origin-main-full-sha> \
  --expected-sha256 <verified-migration-sha256>
```

The agent then runs the exact committed commissioning orchestrator from the
synchronized canonical checkout:

```bash
python3 scripts/bootstrap_production_parity_staging.py \
  --commit <current-origin-main-full-sha> \
  --migration-sha256 <verified-migration-sha256>
```

Before any write, the orchestrator fetches `origin/main`, requires
`HEAD == origin/main == --commit`, requires a clean tracked tree, and compares
its own bytes plus the migration, IAM setup, and static installer bytes with
that commit. It sets `LEADPOET_PARITY_ENABLED=false` first. IAM-capable AWS
credentials are selected by exact name from the pinned gateway cache and stay
in gateway process memory; the CloudWatch instance role is not widened. The
validator's BuiltWith value is selected by exact name from the running
container. The read-only DSN and all request/response payloads containing it
move only through inherited pipes or encrypted in-memory SSH transport; secret
values never enter argv, environment variables, regular files, command output,
or GitHub.

Migration 156 creates the NOLOGIN reader and a fixed postgres-only password
binder. The orchestrator verifies the live migration contract, stages the
immutable static secret, invokes only the fixed parameterized binder, and then
proves a direct login starts with transaction read-only enabled and has no
effective table, sequence, schema-create, or membership write path. The
temporary validator bootstrap role is trusted for about 15 minutes, has the
same absolute cutoff in its identity policy, and is deleted and read back in
an outer `finally`. GitHub is enabled only after that cleanup and complete
variable readback; any failure re-reads `ENABLED=false`.

The Full controller runs on the existing self-hosted gateway builder, while
the measured workload remains on one disposable Nitro instance with an exact
512-GiB encrypted gp3 volume. The host workload is capped at 20 hours inside a
21.5-hour SSM/controller envelope, with fresh bounded OIDC sessions for each
poll window and cleanup. The production OpenRouter pair is read directly from
the already-authorized production gateway secret only inside the transient
runner; no additional copy is created.

## Safety contract

- Production Supabase must never be a runtime write target.
- The source credential must be non-superuser, non-replication, transaction
  read-only, and have no table write capability. BYPASSRLS is permitted solely
  so the read-only snapshot includes the complete production data shape.
- The candidate date is a future unconsumed UTC date inside the clone, avoiding
  deletion or rewriting of copied production daily state.
- The externally reachable candidate gateway keeps miner submissions disabled.
  Only the bounded in-process intake phase enables the production routes, only
  after rebenchmark and weight evidence is complete, and only against the
  disposable clone. SOURCE_ADD dispatch, paid loops, Git/model mutation,
  promotion, fulfillment, and telemetry remain disabled.
- Production Finney is read-only. The adapter cannot forward the final chain
  RPC and cannot fabricate receipts or success.
- Every identity, source archive, release artifact, allocation, and evidence
  document is bound to the exact candidate SHA.
- A failed or unexercised critical stage fails the lane.
