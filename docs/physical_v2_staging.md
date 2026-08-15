# Production-parity staging

Production-parity staging validates the exact candidate commit against the
real production data shape without writing to production or using either
production host as a test runner. It has two lanes:

- `Production Parity Fast` is the mandatory bounded post-push lane. It starts
  in parallel with attestation and targets 5-10 minutes.
- `Physical V2 Staging Acceptance` is the authoritative full lane. It starts
  after the exact candidate attests, creates six disposable hosts, executes a
  fresh daily rebenchmark, and proves primary plus two audit-validator chain
  finalizations on Bittensor testnet.

Neither lane contains a second implementation of scoring, settlement,
allocation, restart, or weight logic. The candidate commit supplies those
paths and configuration. Stable behavioral evidence, rather than copied test
logic, determines acceptance.

## Candidate binding

Every run is bound to one full Git SHA and commits:

- every changed file plus the production restart, V2, parity, scoring,
  rebenchmark, settlement, validator, auditor, and dashboard-facing sources;
- the candidate migration inventory and exact migration bytes;
- the candidate protected-workflow and restart-behavior contracts;
- the secret-free production execution configuration resolved by candidate
  code; and
- the August 9 known-good behavioral oracle. The oracle restores required
  behavior only; it never restores historical source or configuration.

The full lane fetches `origin/main` before and after execution. A superseded
candidate cannot approve or promote a release.

## Production data without production writes

`Production Parity Snapshot` opens a PostgreSQL session with
`default_transaction_read_only=on`, verifies that state, and takes a
serializable custom-format dump from the real production database. The dump
therefore includes the actual historical volume, relation sizes, pagination
shape, and large responses that a synthetic database would miss.

The encrypted dump and manifest are KMS-protected, versioned, Object-Locked,
create-only, checksum-bound, short-lived S3 objects. The snapshot pointer
records each exact S3 version ID. Fast and physical consumers fetch those
versions explicitly, then independently verify KMS identity, retention,
metadata hash, and downloaded bytes before use; an overwritten current key or
newer object version cannot change an in-flight candidate's inputs. The
manifest records the production host identity,
the exact deployed gateway source SHA, that source tree's migration frontier,
the snapshot-capture source and contract, UTC daily-baseline frontier, relation
counts, and database sizes. Candidate code refuses an expired, rewritten,
non-ancestor, wrong-day, or already-contaminated snapshot. Candidate-only
migrations are applied after restore; they are never mislabeled as already
present in production.

The dump is restored only to disposable PostgreSQL. Candidate migrations are
then applied in production order and candidate PostgREST/RPC contracts are
executed there. Snapshot credentials are removed from the runner and mutable
runtime secrets never point back to production Supabase.

The measured candidate has one explicit parity boundary selector rather than a
second staging implementation. It defaults unconditionally to the pinned
production Supabase origin. A different origin is accepted only when the
attested execution configuration contains the complete run ID, run-scoped
`database-<run-id>` TLS origin, `test` network, and non-production netuid. The
same normalized origin changes the measured provider-registry hash and is used
for source reads, outcome checkpoints, and evidence-cache persistence. Partial
configuration, an arbitrary hostname, production network/netuid, or a mismatch
between execution config and provider registry fails enclave boot closed.

The restored data plane is never public. An internal Network Load Balancer
terminates TLS with the configured ACM certificate; its security group accepts
443 only from the run-scoped gateway and dashboard groups, and the database
host accepts PostgREST port 3000 only from that load balancer group. Bootstrap
reads both policies back from EC2 and proves that DNS resolves only to private,
non-loopback addresses with a system-trusted certificate before either
consumer starts.

## Fast lane

On every push to `main`, `.github/workflows/production-parity-fast.yml`:

1. checks out the exact push SHA and builds its source/runtime contract;
2. downloads the current immutable production snapshot;
3. restores it to disposable local PostgreSQL and applies candidate migrations;
4. starts pinned PostgREST and validates production schema, functions, RPCs,
   pagination, and production-scale reads, including the candidate's exact
   finalized-allocation authority policy across the complete cloned epoch
   range with every real page byte count and hash recorded;
5. concurrently runs the candidate-derived N-1 `prepush` rehearsal through
   the repository-owned gateway, validator, auditor, signing, finalization,
   readback, Git-tree, ICP, settlement, and cleanup paths; and
6. publishes KMS-bound, Object-Locked exact-SHA contract, ledger, and success
   commitments only when every critical stage passes and all containers and
   networks are gone. The success commitment pins the exact object versions of
   its contract, ledger, snapshot manifest, and snapshot archive.

The fast lane never broadcasts to a chain and never writes production. Strict
adapters terminate only privileged external writes. Its purpose is rapid
regression feedback while attestation builds; it does not replace the physical
lane.

The snapshot manifest commits to a normalized hash of the configured
production database hostname in addition to its deployed source SHA, schema
frontier, archive bytes, read-only capture evidence, and UTC benchmark
frontier. The fast lane independently recomputes that host commitment, so a
valid archive from another database cannot satisfy production parity.

## Full physical lane

When `LEADPOET_PARITY_ENFORCEMENT_ENABLED=true`, attestation publishes to an
immutable candidate release prefix. The full workflow then:

1. requires the exact candidate's passing fast commitment and pinned snapshot;
2. reads the live production gateway and validator EC2 descriptions and derives
   their current AMIs and instance types;
3. creates a candidate-tagged CloudFormation stack containing an ephemeral
   gateway, primary validator, two auditors, database, and dashboard;
4. verifies all six instance IDs, roles, candidate tags, running state,
   IMDSv2 requirement, and metadata-tag support before SSH;
5. materializes six run-scoped staging secrets from classified production
   boundary values plus staging-only database, wallet, network, and endpoint
   overlays;
6. restores the production snapshot, applies candidate migrations, and overlays
   immutable testnet epoch authority;
7. installs immutable testnet wallets and runs the exact N-1 gateway and
   validator launchers forward to the exact attested candidate using
   `scripts/restart_attested_release_local.sh`;
8. starts the candidate's real auditors and the latest frozen dashboard source;
9. keeps autoresearch paused, enables scoring, and waits for a fresh candidate
   daily rebenchmark;
10. requires all configured ICP results, a bounded aggregate, the complete
    candidate-derived public/private/conditional assignment, durable hashes,
    publication, candidate readiness, and exact dashboard readback; and
11. in parallel, requires one canonical gateway bundle to be retrieved and
    submitted byte-identically by the primary and both auditors for the
    configured number of consecutive epochs.

Primary acceptance includes enclave-proven finalized extrinsic inclusion and
gateway finalization persistence. Each auditor independently requires finalized
`LastUpdate` advancement and reports the exact bundle and weights hashes. The
controller rejects a wrong validator hotkey, epoch, netuid, bundle hash, weights
hash, missing signature, malformed extrinsic hash, non-finalized authority, or
non-consecutive result.

The controller then runs a separate read-only chain process from an auditor
host. That process accepts only the official Bittensor testnet endpoint and the
pinned genesis hash, resolves the primary and both auditor hotkeys to their
registered UIDs, and reads `LastUpdate` plus `Weights` at one finalized block.
Acceptance requires all three finalized updates to cover their accepted
submissions and every visible weight vector to match an accepted canonical
gateway vector. This probe cannot sign, submit, calculate, or repair weights,
so validator-reported success alone is insufficient.

The public/private/conditional sizes and tail split come from the candidate's
resolved policy. The controller contains no hard-coded 10/10/20 assumption, so
compatible policy changes are exercised without staging rewrites.

## Testnet authority and wallets

Mainnet database history cannot authorize testnet submissions. The full lane
uses a one-time real testnet ceremony, not fabricated authority:

1. establish the stateful cutover and finalized settlement/weight lineage on
   the configured testnet subnet;
2. expose that ceremony database through a read-only DSN in
   `LEADPOET_PARITY_TESTNET_AUTHORITY_DSN`;
3. build the immutable artifact:

   ```bash
   python3 scripts/build_production_parity_epoch_authority.py \
     --cutover config/production-parity-testnet-cutover.json \
     --output /secure/production-parity-testnet-authority.tar
   ```

4. publish it with `scripts/publish_production_parity_evidence.py` using KMS,
   versioning, and Object Lock; and
5. record the object version, artifact hash, KMS key, mapping hash, and genesis
   hash in the infrastructure configuration.

The builder derives the full foreign-key dependency closure from the ceremony
schema, freezes row counts and state before and after capture, and embeds a
canonical ceremony commitment. Candidate-added dependent tables are allowed
only when exact migration application creates them and they remain empty after
the authority overlay.

Provision three funded and registered testnet hotkeys for the primary and two
auditors. Package each through the repository wallet artifact format, publish
with KMS, versioning, and Object Lock, and pin its object version, hash, wallet
identity, and expected SS58 hotkey. Wallets are installed into run-scoped paths
and never logged or uploaded as workflow artifacts.

## One-time configuration

Create the GitHub environment `production-parity-staging` and configure:

- `LEADPOET_PARITY_AWS_ROLE_ARN`
- `LEADPOET_PARITY_AWS_REGION`
- `LEADPOET_PARITY_SNAPSHOT_BUCKET`
- `LEADPOET_PARITY_SNAPSHOT_KMS_KEY_ARN`
- `LEADPOET_PARITY_APPROVAL_KMS_KEY_ARN`
- `LEADPOET_PARITY_PRODUCTION_DB_HOST`
- `LEADPOET_PARITY_PRODUCTION_GATEWAY_URL`
- `LEADPOET_PARITY_PRODUCTION_GATEWAY_SECRET_ID`
- `LEADPOET_PARITY_POSTGRES_IMAGE`
- `LEADPOET_PARITY_POSTGREST_IMAGE`
- `LEADPOET_PARITY_INFRA_CONFIG_JSON`
- secret `LEADPOET_PARITY_PRODUCTION_READONLY_DSN`

Create the repository variables `LEADPOET_PARITY_INFRA_READY=false` and
`LEADPOET_PARITY_ENFORCEMENT_ENABLED=false`. They are repository variables
because GitHub evaluates job-level commissioning guards before entering the
protected environment. Until infrastructure readiness is enabled, scheduled
snapshot/cleanup jobs and both parity lanes remain cleanly skipped; the
existing attested production release channel is unchanged.

Start from `infra/production-parity-config.example.json`. Configure VPC,
subnet, Route53, an ACM wildcard certificate covering the run-scoped database
names, instance profiles, pinned container images, production EC2 references,
testnet wallets, testnet authority, and staging overlays. Database and
dashboard base images are configured; gateway, validator, and auditor host
images and sizes are deliberately derived from live production references on
every run.

The gateway overlay must provide a staging-specific, read-only `GITHUB_TOKEN`
and a staging-specific `RESEARCH_LAB_OPENROUTER_MANAGEMENT_KEY` whenever
production has an OpenRouter management credential. Runtime provider
credentials may exercise the real provider path, but source-control and
account-level provider control credentials are never inherited from
production, even if they are listed as an allowed shared boundary. Internal
gateway auth values are generated per run, while production Sentry and OTLP
credentials are cleared and both exporters remain disabled.

The AWS role must be least-privilege for the declared CloudFormation stack,
run-scoped secrets, KMS decrypt/encrypt, versioned Object-Locked immutable
evidence objects, release
candidate reads, and DNS records. It needs read-only access to the two
production instance descriptions and classified production secret documents.

After every prerequisite above is installed, set
`LEADPOET_PARITY_INFRA_READY=true`. Run the snapshot workflow once and verify a
current pointer exists. Run the fast and full workflows manually while
enforcement is disabled. Enable `LEADPOET_PARITY_ENFORCEMENT_ENABLED` only
after both lanes pass and cleanup evidence proves all run-scoped resources
were removed. Never enable enforcement before infrastructure readiness.

## Promotion and cleanup

With enforcement enabled, production restart scripts can read only the
promoted `attested-v2/releases` channel. The attestation job writes the
candidate channel; full parity alone publishes immutable approval evidence and
promotes those exact bytes. Missing, failed, cancelled, stale, superseded, or
unexercised evidence cannot promote a release.

Before its first Secrets Manager write, the controller persists the complete
deterministic cleanup scope. Every exit path then performs bounded, idempotent
deletion of run-scoped Secrets Manager documents, CloudFormation resources,
EC2 key pairs, local private keys, snapshot material, and controller state.
Promotion additionally requires exact committed cleanup evidence. There is no permanent
staging EC2 fleet and no staging workload runs on the production gateway or
validator. `Production Parity Cleanup` also runs hourly and deletes only
resources older than 12 hours whose names and both run/candidate tags satisfy
the parity contract; this covers a hard-cancelled GitHub runner without giving
the janitor a pattern that can match production resources.

## What this proves

The full lane executes the same candidate source, migration bytes, launchers,
Nitro/AF_VSOCK boundaries, artifacts, PCR0 and attestation checks, provider
transport, database behavior, rebenchmark logic, dashboard API, canonical
bundle path, signing, primary submission, auditor submission, finalization,
and readback used by production.

Bittensor testnet is the only intentional external substitution: it proves real
chain encoding, signing, broadcast, inclusion, finalization, `LastUpdate`, and
readback without risking mainnet state. Production monitoring must still prove
mainnet availability and exact-SHA activation, but production restart is no
longer the first execution of application behavior or data-scale compatibility.
