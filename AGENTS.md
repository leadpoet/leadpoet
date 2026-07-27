# Leadpoet Research Lab agent guide

Applies to the entire repository. A deeper `AGENTS.md` overrides it within its
directory.

`AGENTS.md` and `CLAUDE.md` are one document kept in two files. They must
remain word-for-word identical. Every change to either file must update the
other in the same pull request, and CI must fail if they diverge.

## Highest priority: deployment timing

General authorization to make and push fixes does **not** waive this timing
rule.

Before every push that can change a gateway, validator, auditor, release,
manifest, rehearsal, or attestation identity:

1. Fetch `origin/main` and inspect local/upstream divergence.
2. Read the official live epoch block through the bounded read-only path.
3. Inspect the release/attestation state for the current `origin/main` SHA.
4. State the candidate SHA, exact block, attestation state, and whether the
   push would supersede an active release.

Prefer one verified push before block 180, or after block 300. At or after block
180, push only when evidence confirms that code currently on GitHub cannot
complete a production-critical restart or weight workflow and waiting is more
likely to cause another production or weight-submission failure. Routine,
precautionary, documentation-only, logging-only, and non-critical fixes must
wait for the normal safe window.

If the live block or attestation state is unavailable or ambiguous, do not
push. An active candidate is frozen by default. A late critical push must be
narrow, must disclose that it resets/supersedes attestation, and must not bundle
unrelated work. A broad instruction such as "push necessary fixes" is not
permission to treat a non-critical change as a late exception.

## Repository role

This repository owns Research Lab consumption, benchmarking, fulfillment
infrastructure, and the gateway and validator runtimes. It does not own the
sourcing model's semantics.

Preserve unrelated local changes. Never reset, discard, overwrite, or include
another engineer's work. Never commit credentials, private model artifacts,
sealed benchmarks, customer data, provider payloads, or unredacted contact
data.

## Production and Git safety

- Before editing, run `git fetch origin`, inspect status and both directions of
  divergence, and fast-forward a clean checkout. Re-read every touched file.
- Preserve all unrelated modified, staged, untracked, and unpushed work. Use a
  clean worktree when ownership is uncertain.
- Never force-push, rewrite shared history, commit secrets, or push a duplicate
  fix that already exists upstream.
- Never perform production writes, restarts, process kills, Supabase mutations,
  or chain submissions. Read-only SSH diagnosis is allowed; give the operator
  exact commands for mutating actions.
- A healthy process or one HTTP 200 is not workflow proof. Require joined logs,
  durable evidence, bundle hashes, and chain readback as applicable.

The gateway and validator are separate runtimes and must activate one exact
release SHA. Gateway code and imported top-level packages come from the
canonical gateway checkout; validator code runs in its containerized runtime.
Never restore a gateway-only rsync deployment or mix files from different
commits.

## Mandatory V2 release gate

Production must never be the first full execution of a restart candidate.
Follow
[`docs/v2_deployment_verification_checklist.md`](docs/v2_deployment_verification_checklist.md)
before every V2 push/deployment:

1. Freeze one candidate SHA and run the focused weight/auditor suite.
2. Run the exact installed N-1 gateway and validator launchers against that SHA
   with the bounded `prepush` rehearsal.
3. Before an attested release, run the `release` rehearsal. It must complete
   forward, rollback, roll-forward, the fault/concurrency matrix, and 100
   accelerated epochs.
4. Require the documented success markers, zero rejected/internal-substitution
   events, byte-identical primary/auditor canonical vectors, signed SDK
   extrinsic/finalization evidence, and clean final state.
5. After attestation and both restarts, run every documented read-only
   production probe and verify the same evidence across an epoch rollover.

The rehearsal must execute repository-owned production shell and Python paths.
Only privileged external boundaries such as Docker, Nitro, AWS, Supabase, and
chain writes may use strict contract adapters. Never call a helper, mock an
internal success, or use a live production restart as a substitute for the
rehearsal. Any candidate change to restart logic, imported restart behavior,
environment contracts, dependencies, manifests, release identities, or the
rehearsal itself invalidates prior evidence and requires both launcher
rehearsals again.

The production restart and rehearsal must change together. Every affected
command, environment source, filesystem path, lock, cleanup, build, retry,
failure, and success branch needs parity coverage in the same change. Mark
each production stage `passed`, `failed`, or `unexercised`; an unexercised
critical stage blocks a safe-to-restart claim.

## Full-path incident standard

Never stop at the first observed blocker. For every restart, weight, auditor,
attestation, cutover, or Research Lab incident:

- Map the complete production sequence through verification, persistence,
  handoff, signing, submission, finalization, and readback.
- Inspect prior logs and durable failures, then statically trace downstream
  stages the current error prevented from running.
- Exercise independent later stages with authentic sanitized production-shaped
  inputs rather than assuming they pass.
- Test retries, exhaustion, cleanup, cancellation, concurrency, restart,
  rollover, and long-running state, not just one successful call or epoch.
- Report every known/likely blocker and every unexercised stage before another
  push or restart recommendation.

Permanent regression families include: dirty/detached checkouts and stale
launchers; restart-window and release-channel races; env hydration and proxy
normalization; Docker daemon/lock/tmpfs/disk/memory cleanup; artifact, PCR0,
lineage, and role identity drift; runtime `specVersion`, archive, exact-block,
and SDK compatibility; SQL/RPC/schema/pagination/uniqueness changes; transient
HTTP/TLS/EOF and oversized persistence documents; shared-client concurrency,
singleflight cancellation, and retry amplification; stale publication journals
and authorization retention; external auditor startup/update/reconnect and
irrelevant private configuration; same-epoch canonical bundle publication and
submission; and behavior across at least 100 accelerated epochs.

Every live diagnostic must have an outer timeout, close clients in `finally`,
terminate child processes, and leave no orphaned work.

## Exact-commit rollback

Use the coordinated operator command when gateway and validator are not already
verified on the same selected SHA:

```bash
bash scripts/restart_attested_release_local.sh \
  --commit <full-lowercase-40-character-sha> \
  --component all
```

Single-component mode is allowed only when the other runtime is already healthy
on that exact SHA. A rollback target must have a completed immutable attested
release with matching manifests, artifacts, PCR0, dependencies, and release
evidence, and must remain compatible with the current public auditor weight
protocol. Do not reject it merely because it predates later reliability fixes.
Every mismatch fails closed; there is no automatic fallback.

Auditor identity must be resolved by hotkey, never by IP or stale UID. Public
auditors must work from a clean external environment without private cutover
files, Nitro tooling, Supabase service credentials, or AWS credentials.

## Three-repository sourcing runtime contract

The sourcing runtime spans three repositories with independent activation
boundaries:

- `leadpoet/Sourcing_model` owns model behavior, the canonical industry
  taxonomy, runtime-capability semantics, and the `main` and `leadpoet-lab`
  artifact lineages.
- `gzaentz/leadpoet-site` owns the production wrapper, queues, verifier,
  persistence, release registry, and worker deployment.
- This repository owns Research Lab consumption and must resolve the
  branch-specific `leadpoet-lab` artifact, not the shared `main` pointer.

### Semantic ownership and consumer boundaries

A change is model-owned if it can alter candidate discovery, query construction,
ICP interpretation, branch enumeration, route eligibility or ordering, evidence
semantics, scoring, acceptance, rejection, resolution, or deduplication. This
remains true when the behavior is implemented around a host-bound provider such
as Deepline. Consumers may bind credentials and execute a model-owned plan, but
must not independently compile or reinterpret it.

`leadpoet-site` and `leadpoet` may translate a model-owned serialized contract
into host types, but the translation must be lossless and must not add, remove,
broaden, narrow, or reorder semantic constraints. Credentials, provider
transport, queues, leases, retries, cost controls, persistence, deployment,
verification, benchmarking, and publication remain host-owned operational
concerns.

`main` in `leadpoet/Sourcing_model` is the canonical source of shared model
behavior. `leadpoet-lab` periodically incorporates reviewed `main` changes and
publishes its own branch-specific artifact. Research Lab must consume that
artifact and must never reimplement missing model behavior locally.

A compatibility shim that reproduces model behavior outside `Sourcing_model`
requires an upstream tracking reference, parity fixtures, and an explicit
expiration or removal condition. It must be labeled temporary and must not be
treated as the permanent source of truth.

### Cross-repository invariants

- Drift baselines may shrink when duplicate consumer behavior is removed; they
  must never grow to make a check pass.
- Industry taxonomy is model-owned. Consumers may generate and verify
  byte-identical snapshots, but must never hand-edit or independently extend
  taxonomy values.
- Every model symbol a consumer imports, patches, or calls is a versioned
  contract term. Contract discovery and patch application fail closed when a
  required target is absent or ambiguous.
- `probe_origin: UNKNOWN` means "not yet proven" and must proceed with the full
  attempt budget; it never means dead and consumes no attempt by itself. Only
  an explicit dead result may stop that path.
- `DEFERRED_TRANSIENT` is nonterminal and must never be written to a permanent
  exclusions ledger.
- Never merge or advance a live branch pointer while a required check is
  failed, pending, or canceled.

Shipping code is not activation. A Sourcing_model `main` merge does not deploy
the site; a site repin does not serve until append-only registry promotion and
the hardened worker deployment both succeed; and Research Lab does not consume
a revision until `leadpoet-lab` advances and its branch-specific artifact
pointer is verified. Capability, resilience, or taxonomy modules remain inert
until their owning consumer explicitly wires and tests them.

### Model-quality feedback loop

Use this same workflow whether the issue is first observed in Research Lab or
on the production site:

1. A site or Lab run reveals a model-quality issue.
2. Reproduce it as a redacted deterministic fixture in `Sourcing_model`.
3. Implement the model semantics in `Sourcing_model`, not either consumer.
4. Test the exact immutable artifact through Research Lab.
5. Promote the reviewed model commit to `Sourcing_model/main`.
6. Repin the site to that exact source SHA and compatibility-contract hash.
7. Run the site's compatibility, release, registry, and deployment gates.

Lab and site do not share one operational production flow. They share the
exact model artifact, semantic contract, capability declarations, and parity
fixtures while retaining separate queues, credentials, persistence,
benchmarking, verification, and publication controls.

## Research Lab artifact consumption

- Resolve the signed branch-specific
  `research-lab/sourcing-model/branches/leadpoet-lab/current.json` pointer.
- Verify the manifest signature, immutable commit identity, image digest, and
  expected repository before using an artifact.
- Never silently fall back to the shared `main` pointer.
- Never reconstruct missing model behavior in this repository. Fix it in
  `leadpoet/Sourcing_model`, review it on `main`, incorporate it into
  `leadpoet-lab`, publish the branch artifact, and then consume that artifact.
- Treat a missing, stale, unsigned, ambiguous, or mismatched artifact as a
  fail-closed condition.

## Required verification

Before handoff:

```bash
git diff --check
python -m compileall -q Leadpoet gateway leadpoet_audit leadpoet_canonical \
  leadpoet_verifier miner_models qualification research_lab validator_models
python -m pytest -q
```

Also verify that the instruction files are identical, required CI checks are
green, and any referenced Sourcing_model artifact points to the reviewed
`leadpoet-lab` commit. Do not claim artifact activation from a green unit test
alone.
