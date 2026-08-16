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
miner signing helpers, intake models and routes, OpenRouter recipient/privacy
verifier, and SOURCE_ADD miner helper. Both lanes verify those commitments
against the candidate Git blobs and checkout. This prevents stale evidence
without changing measured runtime identity or PCR0. Real credential admission
remains in the authoritative full lane because it requires the candidate's
attested Nitro recipient and measured provider path.

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
    chain boundary; and
12. creates one in-memory ephemeral miner identity and exercises the exact
    candidate OpenRouter and SOURCE_ADD HTTP request models, signatures,
    routes, measured credential verification, PostgREST/RPC calls, and durable
    writes against the clone.

The OpenRouter intake uses the authorized production runtime and management
credentials in memory. It verifies the exact coordinator release evidence,
encrypts both credentials with the same miner-side implementation, submits
the sealed pair to the real route, and requires the key reference plus both
encrypted envelopes to exist in the clone with no plaintext in responses,
rows, logs, or retained evidence.

The production verifier deliberately performs one idempotent management API
write that forces OpenRouter workspace logging off and then reads it back. The
parity lane preserves that exact security behavior; it does not create, rotate,
or delete provider keys. Production Supabase and the chain remain read-only.

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

There is no GitHub Environment and no testnet setup. The operator creates one
dedicated read-only PostgreSQL role in Supabase. One idempotent helper then
verifies that role and creates only:

- one Secrets Manager record containing that DSN;
- one separate read-only-to-the-runner secret containing the BuiltWith key;
- a GitHub OIDC controller role;
- one restricted EC2 runner role and instance profile; and
- repository variables for existing production resources and immutable
  PostgreSQL/PostgREST image digests.

Generate a unique hexadecimal password locally, replace the placeholder below,
and run the SQL once in the production project's Supabase SQL editor:

```sql
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_roles WHERE rolname = 'leadpoet_parity_reader'
  ) THEN
    CREATE ROLE leadpoet_parity_reader LOGIN;
  END IF;
END
$$;

ALTER ROLE leadpoet_parity_reader WITH
  LOGIN
  PASSWORD 'REPLACE_WITH_UNIQUE_HEX_PASSWORD'
  BYPASSRLS
  NOSUPERUSER
  NOCREATEDB
  NOCREATEROLE
  NOREPLICATION
  CONNECTION LIMIT 2;
ALTER ROLE leadpoet_parity_reader SET default_transaction_read_only = on;
ALTER ROLE leadpoet_parity_reader SET idle_in_transaction_session_timeout = '5min';

GRANT CONNECT ON DATABASE postgres TO leadpoet_parity_reader;
GRANT USAGE ON SCHEMA public TO leadpoet_parity_reader;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO leadpoet_parity_reader;
GRANT SELECT ON ALL SEQUENCES IN SCHEMA public TO leadpoet_parity_reader;
ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public
  GRANT SELECT ON TABLES TO leadpoet_parity_reader;
ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public
  GRANT SELECT ON SEQUENCES TO leadpoet_parity_reader;
```

Use the Supabase session-pooler URI for that role with `sslmode=require`. Do
not paste the URI into a terminal command: the helper reads it through a
hidden prompt and stores it directly in Secrets Manager.

Run:

```bash
python3 scripts/setup_production_parity_staging.py apply \
  --production-gateway-url https://gateway.subnet71.com \
  --enable
```

The helper prompts first for the DSN and then for the BuiltWith key without
echoing either value. It verifies the role and live gateway, resolves immutable
container digests, creates or updates the IAM objects idempotently, and
configures repository variables with `gh`. Secret values are never printed or
stored in GitHub. The production OpenRouter pair is read directly from the
already-authorized production gateway secret only inside the transient runner;
no additional copy is created.

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
