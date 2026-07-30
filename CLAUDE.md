# Leadpoet Research Lab agent guide

Applies to the entire repository. A deeper `AGENTS.md` overrides it within its
directory.

`AGENTS.md` and `CLAUDE.md` are one document kept in two files. They must
remain word-for-word identical. Every change to either file must update the
other in the same pull request, and CI must fail if they diverge.

## Highest priority override: explicit push-now authorization

Any unambiguous current-turn instruction that both directs Codex to push the
requested fix immediately and grants authorization to push activates an
emergency push-first workflow. Natural-language variants count; no exact
phrase is required. Examples include **"I'm authorizing you to push now"**,
**"I authorize you to push now"**, **"push the fix; you have authorization"**,
and **"push immediately, then test"**. This narrow, explicit exception takes
precedence over the deployment-timing and mandatory V2 pre-push
test/rehearsal rules below.

When that authorization is present:

1. Treat the next two minutes as a hard push deadline. Stop lower-priority
   investigation, reporting, polishing, and test work. Dedicate available
   execution capacity to implementing the smallest correct fix, synchronizing
   it safely, committing it, and pushing it to `origin/main`.
2. Run **no tests, rehearsals, CI jobs, live block checks, attestation checks,
   or broad validation before the first push**. Do not question, reject,
   reinterpret, or delay the authorization because of block timing,
   attestation state, or an otherwise mandatory pre-push gate.
3. Still fetch and preserve concurrent work, refuse force-pushes, and inspect
   the exact diff for credentials or unrelated files. These are repository
   integrity operations, not test gates, and must not consume the two-minute
   window unnecessarily. If a real upstream conflict makes a non-destructive
   push impossible, report only that concrete conflict immediately.
4. Immediately after the push, run every focused, regression,
   production-equivalent, exact-launcher, `prepush`, and other verification
   gate that would normally have run before it. Report the pushed SHA first,
   then continue testing without waiting for another user response.
5. If post-push testing finds a confirmed production-critical defect, make the
   narrow correction and push it under the same emergency authorization, then
   rerun all affected tests. Do not make precautionary, cosmetic, or unrelated
   follow-up pushes.

Interpret the user's ordinary meaning instead of requiring a magic phrase. A
general implementation request, standalone standing authorization, "push when
ready", or a future conditional instruction without a current immediate-push
directive does not trigger the override. Without an unambiguous current-turn
instruction to push immediately, every normal rigorous pre-push test,
rehearsal, deployment-timing, block, and attestation rule in this document
remains fully mandatory and unchanged.

## Highest priority: deployment timing

General authorization to make and push fixes does **not** waive this timing
rule. The sole exception is the exact push-now authorization defined above.

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
- For read-only SSH to both production hosts, use the current July 28 identity:
  `/Users/pranav/Downloads/leadpoet-2026-07-28.pem`. Gateway:
  `ec2-user@52.91.135.79`; validator: `ec2-user@100.59.201.156`. Do not use the
  older `leadpoet-gateway-tee-main.pem` or `leadpoet-validator.pem` identities.
- A healthy process or one HTTP 200 is not workflow proof. Require joined logs,
  durable evidence, bundle hashes, and chain readback as applicable.

### Mandatory local/main synchronization

The tracked local checkout must use the latest merged `origin/main` as its
source of truth. Never reconstruct `main` from an unmerged pull-request branch,
an old worktree, copied files, or local-only artifacts.

The canonical checkout at
`/Users/pranav/Downloads/Election_Analysis/Bittensor-subnet` must itself be
synchronized; using or pushing from a secondary worktree does not satisfy this
rule. After any push from any checkout, return to the canonical checkout, fetch
`origin/main`, fast-forward it without disturbing preserved local-only files,
and verify `HEAD`, `origin/main`, and every Git-tracked worktree file agree.
Before telling the operator to run any migration, script, or other repository
path, additionally prove that the path is tracked by the latest
`origin/main`, exists in the canonical checkout, and has the same Git blob.
Open pull-request heads and unmerged branch files are never authoritative for
this check.

Before every edit and again immediately before every commit or push:

1. Run `git fetch origin`, `git status --short --untracked-files=all`,
   `git log origin/main..HEAD --oneline`, and
   `git log HEAD..origin/main --oneline`.
2. If there is no local-only tracked work, fast-forward local `main` to
   `origin/main` and verify that `git diff --name-status origin/main --` is
   empty before starting the task. Confirm that every path tracked by
   `origin/main` exists locally.
3. If local tracked, staged, untracked, or unpushed work may belong to another
   engineer or Codex chat, preserve it exactly and use a clean synchronized
   worktree. Never overwrite, reset, stash, rebase, delete, or silently include
   that work.
4. Treat untracked and ignored files as local-only by default. Do not add or
   push one merely because it exists locally. A local-only file may become
   tracked only when it is explicitly in scope, reviewed as required for the
   production workflow, and intentionally approved for the task.
5. After creating the task commit, the only permitted difference from the
   freshly fetched `origin/main` is the reviewed task commit itself. List every
   path in `origin/main..HEAD`, verify that no unrelated file is present, and
   verify that the worktree and index contain no additional tracked changes.
6. If `origin/main` advances before the push, integrate the new commits safely,
   re-read overlapping files, and rerun affected checks. Never force-push or
   replace a remote file with a stale local copy.
7. After a successful push, fetch again and require local `HEAD` and
   `origin/main` to resolve to the same SHA, with no tracked worktree diff and
   no local-only commit left on `main`. Preserve non-overlapping local-only
   files without committing them.

This is compatible with an intentional task change: the checkout must match
`origin/main` exactly before work begins; immediately before push, it may differ
only by the reviewed commit being pushed; immediately after push, it must match
`origin/main` exactly again.

The gateway and validator are separate runtimes and must activate one exact
release SHA. Gateway code and imported top-level packages come from the
canonical gateway checkout; validator code runs in its containerized runtime.
Never restore a gateway-only rsync deployment or mix files from different
commits.

Coordinated restart should overlap gateway work with validator preparation,
but it must preserve one hard activation boundary. Git/release verification,
artifact staging, old-runtime shutdown, EIF/Nitro/relay/runtime/hotkey setup,
and exact validator application-image build may run while the gateway
restarts. No validator coordinator or worker may start until the image commit
label and immutable image ID are verified, the selected SHA's coordination
marker is present, and gateway V2 authority health, build-info, and immutable
release evidence all report that exact SHA. Recheck the image ID after the
wait and repeat the full gateway check after startup. A gateway failure must
publish a commit-bound failure marker, and every failed or interrupted late
activation must clean validator containers, host validator/relay processes,
Nitro enclave, and Docker lock. Historical deployers without this late barrier
must keep the fail-closed check immediately before their deployer is invoked.
A retry after an N-1-to-N Git handoff may use only the selected SHA's clean
candidate launcher blob, must reject a later branch advance before restart
preparation or shutdown, and must signal the remote validator immediately on
coordinated cancellation rather than treating a replaced marker as sufficient
revocation. The operator-owned coordination path and bounded wait policy must
survive candidate re-execution and may not be replaced by hydrated secrets.

## Mandatory V2 release gate

Production must never be the first full execution of a restart candidate.
Except during the exact push-now authorization defined above, when this same
complete gate runs immediately after the first push, follow
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

When an attested operation is wrapped by artifact persistence, the exact
gateway readiness path must exercise both layers. It must pair
`execution_receipt` only with `execution_receipt_graph`, pair the outer
lineage `receipt` only with the outer `receipt_graph`, and reject either
cross-pairing. A generic receipt-ancestry fixture that collapses both layers
does not cover this production contract.

### Mandatory migration-backed durable-state rehearsal

The exact-launcher rehearsal must not infer the durable database contract from
Python constants or accept arbitrary JSON in an in-memory substitute. Before
either launcher proceeds, the pinned Amazon Linux replica must start a
disposable PostgreSQL instance and apply the candidate's settlement-critical
SQL migrations verbatim in production order.

That stage must:

- Exercise real PostgreSQL constraints, foreign keys, functions, RPC return
  documents, and view projections using nonempty sanitized production-shaped
  bundle, publication, finalization, and chain-settlement rows.
- Run the production settlement authority parser against the exact row
  returned by `research_lab_finalized_allocation_epochs_v2`; adapters may not
  invent a missing selected column as JSON `null`.
- Prove the pre-migration failure and post-migration success for every
  constraint-widening regression. In particular, V1 chain-observation and
  realized-settlement transport evidence must fail before migration 128 and
  persist after it.
- Export a candidate-SHA-bound relation/column/RPC contract. The local
  PostgREST adapter must consume that contract and reject unknown selected,
  filtered, ordered, or inserted columns instead of silently accepting them.
- Fail before any simulated shutdown with a structured diagnostic containing
  the candidate SHA, stage, migration, relation/RPC, constraint or projection,
  and underlying PostgreSQL error.

The independent canonical workflow runner remains mandatory, but it cannot
substitute for this gate. A primary/auditor bundle test beside the restart does
not prove that the gateway can reconstruct the same authority from its durable
production-shaped state. Any change to a V2 migration, required schema/RPC
preflight, durable row shape, settlement parser, bundle/publication/finalization
store, or PostgREST query must update this database rehearsal and its
regression evidence in the same change.

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

### Mandatory aggregate rehearsal result

The `prepush` and `release` rehearsal controllers must finish every independent
stage in one invocation even after an earlier stage fails. A stage itself
should fail immediately with its root-cause diagnostic; that failure must not
terminate the controller or suppress another independently executable stage.
At minimum, the same run must attempt the CPython 3.7 proof, candidate artifact
and fixture preparation, the exact installed N-1 gateway launcher, the exact
installed N-1 validator launcher, the independent canonical
gateway/primary/auditor workflow, signing, publication, finalization,
`LastUpdate`/reveal readback, cleanup, and final evidence join.

Every run, including a failed run, must write one candidate-SHA-bound stage
ledger. Each declared stage is exactly one of `passed`, `failed`, or
`unexercised`. `failed` entries include the command or production exception and
root-cause evidence; `unexercised` entries name the exact failed prerequisite.
Only a genuine data or process dependency may make a stage unexercised.
Gateway failure never skips the validator or independent canonical workflow,
and validator failure never skips the gateway or independent canonical
workflow. Release fault cases and accelerated epochs are isolated sufficiently
that one failed case cannot poison or hide the remaining cases.

Any failed or unexercised production-critical stage fails the complete gate.
The successful joined manifest is valid only when the stage ledger is complete,
contains no duplicate stage names, and every entry passed. Regressions for the
rehearsal controller must inject at least two independent simultaneous failures
and prove that both are reported while later stages still execute. A test that
only proves the first exception is not rehearsal coverage.

### Candidate-owned behavioral contract

The frozen candidate is the rehearsal source of truth. Build one versioned,
hash-bound behavior contract from the candidate's protected-workflow manifest,
exact production entrypoints, effective Research Lab/ICP/Git-tree policies,
fault contract, and release profile. The workflow runner and final evidence
join must independently reconstruct that same contract from the read-only
candidate source. They may not maintain separate fixed source counts, stage
lists, ICP counts, tree widths, fault minimums, or other implementation-shaped
expectations.

The contract declares stable observable invariants, not private implementation
phases. Generate test sizes and legal/illegal state spaces from the candidate's
validated production configuration, then exercise the production selector,
lineage, settlement, canonical bundle, primary validator, auditor, signing,
publication, finalization, and readback functions. A changed ICP split, tree
topology, settlement prefix, protected source inventory, or profile therefore
changes the contract and its generated evidence automatically. Missing,
duplicate, stale, or undeclared stages and invariants fail the final join.

The exact installed N-1 launchers remain authoritative for process sequencing,
environment hydration, filesystem state, activation, and restart behavior.
Repository-owned production logic may never be replaced by a fabricated
success. Strict adapters are permitted only at privileged external boundaries
and must reject every unknown or unconsumed operation. Adding a genuinely new
external boundary or top-level production behavior requires registering its
stable invariant and strict adapter in the same candidate; until then the
candidate fails closed. This explicit declaration is required because no test
harness can safely infer the intended semantics of arbitrary new production
features.

When production or rehearsal code adds, removes, reorders, or changes a
restart/downstream behavior, update the candidate-owned contract, generated
state-space regression, `AGENTS.md`/`CLAUDE.md`, and deployment checklist in
the same change only when the existing contract does not already derive and
exercise it. Do not duplicate the resulting inventory in the runner or join.
Except for the explicit emergency push-now workflow above, do not push from a
run that stopped at its first error or omitted its complete candidate-derived
stage ledger.

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

### Approved source suggestions from miners

- A miner may suggest an already approved SOURCE_ADD provider for either
  company discovery or intent discovery by naming the provider and the
  discovery stage in `brief_public_summary`.
- The loop may translate that prose only through the runtime-ready provider
  capability catalog. It must emit a manifest-bound
  `leadpoet.routerverse_source_incorporation.v1` request; an unapproved,
  inactive, credential-unready, unattested, ambiguous, or stage-less mention
  fails closed and cannot authorize a patch.
- The code-edit plan must use the `source_routing` lane and edit
  `sourcing_model/routing/runtime.py::SOURCE_ADD_ROUTING_REGISTRATIONS` through
  `SourceAddRoutingRegistration`. Query-only changes, hard-coded provider
  branches, new endpoints, credentials, dependencies, or network clients do
  not satisfy the request.
- Generated diffs must match the approved provider ID, stage, and immutable
  manifest exactly. Lab validates those fields before building the candidate.
- A successful model candidate is still only a model proposal. The provider
  remains unavailable until the reviewed Sourcing_model change reaches the
  `leadpoet-lab` artifact and each consumer separately binds and activates the
  exact tool ID.

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
