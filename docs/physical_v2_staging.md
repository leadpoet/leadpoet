# Production-parity validation

This system validates only the two workflows that need production-scale
evidence: the complete daily rebenchmark and canonical primary/audit weight
handling. It does not create staging miners, wallets, authorities, dashboards,
or a permanent staging fleet.

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

No production rows are copied in the fast lane, and no database dump is
uploaded as an artifact. The schema archive is destroyed on every exit path;
only redacted hashes, counts, sizes, and stage evidence are retained. The
strict adapter cannot issue POST, PATCH, PUT, DELETE, follow redirects, send a
request body, or target any host other than the pinned production Supabase
origin.

## Full lane

`Production Parity Full` starts after the exact commit's normal attested
release succeeds. It dynamically:

1. derives the live gateway AMI, subnet, VPC, and default instance type;
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
8. keeps miner submissions, autoresearch claims, promotion, fulfillment, and
   model/Git mutation disabled while enabling the real daily baseline;
9. waits for every configured ICP, retries, aggregate, configurable
   public/private/conditional assignment, persistence, and publication;
10. fetches and verifies the real gateway allocation handoff with production
    validator integration; and
11. hash-binds that verified allocation document into the exact candidate
    primary/audit signing and submission path, then requires both validators
    to consume the same canonical vector through the strict non-forwarding
    chain boundary.

The workflow immediately deletes its instance, volume, security group,
CloudFront distribution, run secret, and local database dump. The transient
bucket uses one-day S3 COMPLIANCE retention, so its candidate bundle and
redacted evidence cannot be deleted early; the scheduled cleanup job deletes
their versions and the bucket after retention expires. The same cleanup job
removes only stale resources bearing the exact run, candidate, and ephemeral
ownership tags after hard cancellation.

## One-time prerequisites

There is no GitHub Environment and no testnet setup. The operator creates one
dedicated read-only PostgreSQL role in Supabase. One idempotent helper then
verifies that role and creates only:

- one Secrets Manager record containing that DSN;
- a GitHub OIDC controller role;
- one restricted EC2 runner role and instance profile; and
- repository variables for existing production resources and immutable
  PostgreSQL/PostgREST image digests.

Run:

```bash
python3 scripts/setup_production_parity_staging.py apply \
  --production-gateway-url https://subnet71.com \
  --enable
```

The helper prompts for the DSN without echoing it, verifies the role and live
gateway, resolves immutable container digests, creates or updates the IAM
objects idempotently, and configures repository variables with `gh`. Secret
values are never printed or stored in GitHub.

## Safety contract

- Production Supabase must never be a runtime write target.
- The source credential must be non-superuser, non-replication, transaction
  read-only, and have no table write capability. BYPASSRLS is permitted solely
  so the read-only snapshot includes the complete production data shape.
- The candidate date is a future unconsumed UTC date inside the clone, avoiding
  deletion or rewriting of copied production daily state.
- Candidate claims, miner submissions, Git/model mutation, promotion,
  fulfillment, telemetry, and management credentials remain disabled.
- Production Finney is read-only. The adapter cannot forward the final chain
  RPC and cannot fabricate receipts or success.
- Every identity, source archive, release artifact, allocation, and evidence
  document is bound to the exact candidate SHA.
- A failed or unexercised critical stage fails the lane.
