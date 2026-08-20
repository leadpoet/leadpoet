# Leadpoet Research Lab agent guide

Applies to this repository unless a deeper `AGENTS.md` overrides it.
`AGENTS.md` and `CLAUDE.md` must remain byte-identical.

## Priorities

1. Preserve V2 trustlessness and exact-commit behavior.
2. Preserve scoring, promotion, rewards, payments, allocation, emissions, and
   weight submission unless they are explicitly in scope.
3. Prefer the smallest correct diff. Do not refactor unrelated code.
4. Communicate immediately when scope or expected duration materially grows.

## Push-now override

An unambiguous current-turn instruction to push immediately with authorization,
including natural-language equivalents of "I authorize you to push now",
activates this workflow:

1. Fetch and preserve concurrent work, inspect the diff for secrets and
   unrelated files, then make the smallest correct commit and push within two
   minutes. Never force-push.
2. Run no tests, rehearsal, block check, or attestation check before that first
   push.
3. Immediately run the normal fast gate after the push. Push a follow-up only
   for a confirmed production-critical defect.

General authorization, "push when ready", or a future conditional instruction
does not activate this override.

## Repository synchronization

`origin/main` is authoritative. Open PRs, old worktrees, and copied files are
not.

The canonical local checkout is
`/Users/pranav/Downloads/Election_Analysis/Bittensor-subnet`. A clean worktree
may be used to protect concurrent work, but synchronizing or pushing from that
worktree does **not** synchronize the canonical checkout.

Canonical-main parity is a required start and completion gate. Check it before
editing, after every fetch or push, and immediately before the final response:

```bash
git fetch origin
git status --short --untracked-files=all
git log origin/main..HEAD --oneline
git log HEAD..origin/main --oneline
test "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)"
git diff --exit-code origin/main --
git ls-tree -r --name-only origin/main | while IFS= read -r tracked_path; do
  test -e "$tracked_path" || test -L "$tracked_path" || \
    printf 'MISSING %s\n' "$tracked_path"
done
```

- Run this gate in the canonical checkout itself. The two logs and final
  path audit must be empty, both `test`/`diff` commands must succeed, and
  every tracked path newly added on `origin/main` must exist locally. This
  checks content and complete path parity, including migrations, manifests,
  tests, and newly created modules.
- Fast-forward a clean canonical checkout with `git pull --ff-only` before
  work. Never copy one missing tracked file into an older checkout: that creates
  an incoherent partial tree and does not count as synchronization.
- If modified, staged, untracked, or unpushed work may belong to another chat
  or engineer, preserve it exactly and use a clean worktree at `origin/main`.
  Never stash, reset, clean, discard, overwrite, or rebase that work.
- Treat untracked files as local-only unless explicitly reviewed and in scope.
- Before push, fetch again. Integrate upstream safely and rerun affected fast
  checks. Never replace a newer remote file with a stale local copy.
- The task commit may contain only reviewed task paths.
- After every push, return to the canonical checkout, fetch, fast-forward it,
  and rerun the complete parity gate above. Do not report success, completion,
  or "local is current" based on a publishing worktree.
- Preserve all untracked/local-only files during canonical synchronization;
  never add or push them merely because they exist locally.
- If tracked edits or local-only commits in the canonical checkout prevent a
  safe fast-forward, do not overwrite, stash, reset, clean, or rebase them.
  Report the exact blocking paths and commits immediately, before further work
  or any final response. State which `origin/main` additions are absent or only
  partially copied. The task remains incomplete until the canonical checkout
  passes the full parity gate.

## Secrets and production safety

- Never commit credentials, PEM contents, provider payloads, private model
  artifacts, sealed benchmarks, customer data, or unredacted contact data.
- By default, never perform production writes, restarts, process kills,
  Supabase mutations, or chain submissions. Give the operator exact commands.
- Read-only SSH is allowed. Use
  `/Users/pranav/Downloads/leadpoet-2026-07-28.pem` for:
  `ec2-user@52.91.135.79` and `ec2-user@100.59.201.156`.
- One HTTP 200 or a running process is not proof. Require joined logs, durable
  rows or receipts, identical bundle hashes, and chain readback as applicable.

### Explicit overnight recovery authorization

Only a newest-turn explicit instruction to run or start
`$overnight-weight-recovery` activates this exception. An implicit skill
trigger, a diagnosis request, or general authorization does not.

While that run is active and the user has not said `STOP`, Codex may:

- Push reviewed permanent fixes under the normal synchronization, testing,
  epoch-timing, and attestation rules.
- Apply an exact committed, numbered, idempotent Supabase migration after it
  passes disposable-PostgreSQL coverage and a read-only live schema check.
- Run the repository's canonical gateway and validator restart scripts over
  SSH after the exact candidate attests, then run production probes and monitor
  logs/readback until the skill's three-consecutive-epoch condition succeeds.
- Use existing configured credentials without displaying, copying, rotating,
  or persisting their values. Use repository-owned configuration tooling only
  when the tested fix requires it, with backup and non-secret readback.

This exception never permits ad hoc remote source edits, arbitrary SQL or row
deletion, direct/manual/emergency chain submissions, manual weight vectors,
secret rotation, process manipulation outside the canonical restart scripts,
or bypassing attestation, PCR0, exact-commit, archive, signature, enclave, or
fail-closed checks. `STOP` revokes the exception immediately. Every other skill
and workflow remains under the default production-write prohibition.

### Explicit overnight rebenchmark authorization

Only a newest-turn explicit instruction to run or start
`$overnight-rebenchmark-validation` activates this separate exception. An
implicit skill trigger, diagnosis request, or general authorization does not.

While that run is active and the user has not said `STOP`, Codex may:

- Push reviewed permanent fixes under the normal synchronization, testing,
  epoch-timing, attestation, and exact-release rules.
- Apply an exact committed, numbered, idempotent Supabase migration after it
  passes disposable-PostgreSQL coverage and a read-only live schema check.
- Use repository-owned admin/configuration tooling to preserve maintenance
  state and resume scoring when required for the explicitly requested official
  rebenchmark. It must not unpause autoresearch or miner submissions when they
  were paused at invocation.
- Run the canonical gateway and validator restart scripts after the exact SHA
  attests, then run production probes and monitor until one complete latest-
  model rebenchmark scores every configured ICP, derives and persists the
  candidate policy's public/private/conditional assignment, publishes the
  aggregate, exposes that score on the subnet dashboard, and leaves candidate
  evaluation ready.

This exception never permits ad hoc remote source edits, arbitrary SQL or row
deletion, fabricated scores or provider responses, manual weight vectors,
direct/manual/emergency chain submissions, secret rotation or disclosure,
process manipulation outside repository-owned admin and canonical restart
tools, or bypassing attestation, PCR0, exact-commit, archive, signature,
enclave, model-lineage, canonical-bundle, settlement, or fail-closed checks.
`STOP` revokes the exception at the next safe atomic boundary. Every other
skill and workflow remains under the default production-write prohibition.

### Explicit local autoresearch authorization

Only a newest-turn explicit instruction to run or start
`$overnight-autoresearch-testing-local` activates this separate exception. An
implicit trigger, diagnosis request, skill-creation request, or general
authorization does not.

While active and until the user says `STOP`, Codex may use the skill's fixed
in-memory bridge to read only its allowlisted gateway secret groups, make
budget-bound live OpenRouter validation and generation calls, create disposable
local databases, containers, hotkeys, gateways, workers, Git trees,
checkpoints, and candidates, and push tested permanent product or rehearsal
fixes under normal repository, test, epoch, and attestation rules.

This exception is local-only. It never permits production Supabase writes,
production restarts or process manipulation, chain writes, production wallets,
secret rotation or disclosure, candidate publication or promotion,
model-pointer or branch mutation, fabricated provider output, or bypassing
attestation, PCR0, exact-commit, encrypted-credential, model-lineage, Git-tree,
receipt, archive, settlement, or fail-closed checks. Production access remains
read-only and all test state must be disposable. Secret values may exist only
in the bridge/orchestrator process memory and encrypted production request
path, never files, arguments, logs, ledgers, or chat.

## Deployment timing

Except for the push-now override, before a runtime, restart, release, manifest,
rehearsal, or attestation-changing push:

1. Read the bounded official live epoch block.
2. Inspect GitHub release/attestation state for current `origin/main`.
3. State candidate SHA, exact block, attestation state, and whether the push
   supersedes an active candidate.

Prefer one push before block 180 or after block 300. At or after block 180,
push only a confirmed production-critical fix when waiting is more likely to
break restart or weight submission. If block or attestation state is
unavailable, do not push.

## Explicit testing skills

- Use `$bugs` only when explicitly requested for a reported Research Lab defect
  that should be reproduced, minimally fixed, regression-tested, and safely
  pushed; expect 15–60+ minutes and never invoke it automatically. Route one
  `gpt-5.6-sol` `xhigh` lead and require a fresh read-only Sol `xhigh` review;
  allow `max` only for an evidence-backed reasoning bottleneck.
- Use `$ship` only when explicitly requested for the full v1 plan/build/test,
  evidence-based v2 remediation, final verification, and pull-request workflow;
  expect 30–120+ minutes or longer for CI and never invoke it automatically.
  Route fresh read-only Sol `xhigh` planning/review phases and one sequential
  writable `gpt-5.6-terra` `max` implementation/verification agent.
- Skill invocation is not the push-now override and does not bypass any
  repository, production-safety, test-profile, epoch, or attestation rule.

## Default verification: deterministic 2-minute gate

The blocking pre-push gate has a 120-second outer deadline. It contains only
syntax/format checks, directly affected regressions, and one hermetic complete
transition for every release class touched by the diff. A green process with
skipped, collect-only, unexercised, or zero materially executed required tests
is not a pass. Run broad suites and the legacy `prepush` profile asynchronously
after push.

Run the unaccelerated profile only when the user's current request explicitly
includes `un-accelerated` or `unaccelerated` as the requested test/rehearsal
mode. Do not infer it from words such as "thorough", "full",
"production-equivalent", "end-to-end", "all tests", or "release".

For ordinary changes:

1. Run syntax/format checks and only directly affected unit/regression tests.
2. For V2, gateway, validator, auditor, restart, durable-state, settlement,
   scoring, or weight changes, freeze the candidate SHA and exercise the exact
   deployed-N-1 -> candidate transition for the changed production seam. The
   test must reach every downstream output of that seam under dependency-
   minimal, production-shaped inputs and include fail-closed negatives.
3. Require the combined blocking checks to finish within 120 seconds and
   report materially executed test/stage counts plus cleanup.

After push, start the broad legacy rehearsal concurrently with attestation:

```bash
python3 scripts/run_local_restart_rehearsal.py \
  --from-sha <deployed-n-minus-one-sha> \
  --candidate-sha HEAD \
  --transition forward \
  --profile prepush
```

This asynchronous run has a 600-second target and should exercise:

- Candidate-bound source, artifact, release, PCR0, and role identity.
- Production N-1 gateway and validator launcher paths.
- Disposable PostgreSQL with candidate migrations and strict PostgREST/RPC
  contracts.
- Credential-envelope and provider preflight boundaries.
- One canonical gateway bundle delivered byte-identically to primary and audit
  validators.
- Parsing, verification, SDK signing, submission, finalization, `LastUpdate`,
  reveal/readback, cleanup, and a complete stage ledger.
- Candidate-derived Git-tree, conditional-ICP, settlement, and protected-flow
  invariants.

Independent stages continue after a failure and end in one ledger with every
stage `passed`, `failed`, or `unexercised`. Broad-lane status, timeout, or a
missing failure summary alone has no candidate veto. An independently
actionable retained artifact proving a candidate product/trust failure blocks;
every required invariant unique to a broad lane must first move into the
deterministic gate. A bounded checkout, transfer, fixture, capacity, or
staging-bootstrap failure before candidate product execution is visible but
nonblocking after exact classification; it is not green product coverage.
Per-stage and total duration must be reported.

A new or materially changed broad lane starts commissioning-only. It earns
status-based veto eligibility only after 20 consecutive exact-main runs reach
candidate code, meet their time and cleanup contracts, and retain
machine-readable executed-stage counts plus actionable injected-negative
artifacts. Even after commissioning, a red status without an actionable
artifact has no veto. Keep ineligible lanes visible and repair or retire them.

Do not run the whole repository suite before push. Broad CI runs after push and
must not serialize exact-SHA attestation.

## Explicit unaccelerated verification

Only when the current request uses the explicit trigger above, run:

```bash
python3 scripts/run_local_restart_rehearsal.py \
  --from-sha <deployed-n-minus-one-sha> \
  --candidate-sha HEAD \
  --transition forward \
  --profile unaccelerated
```

This preserves the former long release logic exactly: pinned `linux/amd64`,
forward, rollback, roll-forward, full fault/concurrency matrix, and 100
accelerated epochs. It is supplemental evidence, never the default pre-push
gate. The old `--profile release` CLI spelling is intentionally invalid.

## Production-parity staging

`LEADPOET_PARITY_ENABLED` is the single commissioning guard. After every push
to `main`, `Production Parity Fast` is an asynchronous post-push diagnostic
that runs in parallel with attestation. It uses the exact pushed SHA,
resolves live N-1, restores the exact production schema to disposable
PostgreSQL, applies candidate migrations, and exercises candidate-generated
measured-source reads against real production data through a strict GET-only,
no-body, no-redirect adapter. It also runs the candidate-derived restart,
rebenchmark-contract, canonical-bundle, primary/audit signing, finalization,
readback, and cleanup checks. Fast validation never copies production rows.
Its independent source commitments also hash-bind the exact miner signer,
intake models and routes, OpenRouter recipient/privacy verifier, and SOURCE_ADD
miner helper against the candidate Git blobs and checkout. This prevents stale
fast-lane evidence without changing measured runtime identity or PCR0.

`Production Parity Full` provides authoritative-coverage evidence for
rebenchmark or weight changes when it reaches candidate code and retains an
actionable stage ledger. After exact-SHA attestation it creates one encrypted transient Nitro
host derived from the live gateway AMI, runs the exact candidate gateway
restart against the database clone and real provider/model reads, completes
every candidate-configured ICP and assignment, verifies the real allocation
handoff, hash-binds that allocation into one canonical candidate-derived
vector, and exercises exact primary/audit SDK submission through the strict non-forwarding chain boundary. It proves the application path but does not
claim external chain inclusion. Testnet is required only when inclusion itself
is explicitly in scope. The full lane also uses one ephemeral in-memory miner
to exercise the exact OpenRouter sealed-credential admission and credential-
free SOURCE_ADD admission against the clone. Real provider credentials remain
in memory, retained evidence is redacted, and the only intake adapter is the
external registration lookup for that one ephemeral hotkey.

Neither lane may duplicate ICP counts, scoring, settlement, allocation,
signing, or weight policy. Production Supabase is read-only and never a runtime
write target; mutable state lives only in the clone. The externally reachable
gateway keeps miner submissions disabled; the isolated intake phase runs only
after rebenchmark/weight proof and cannot dispatch SOURCE_ADD or paid loops.
Autoresearch claims, promotion, fulfillment, and Git/model mutation stay
disabled. No permanent staging fleet, persistent staging wallet,
testnet authority, or GitHub Environment is permitted. Missing cleanup or any
failed/unexercised critical stage fails the lane, but lane failure alone does
not veto the candidate. Only an independently actionable retained artifact
proving a candidate product/trust defect blocks. An unknown or no-summary
result remains unproven, never green; move any invariant it uniquely covers
into the deterministic gate. A conclusively pre-candidate infrastructure
failure is quarantined and repaired separately. See `docs/physical_v2_staging.md`.

## Rehearsal contract

- Exercise repository-owned production shell and Python paths. Strict adapters
  are allowed only at privileged external boundaries such as AWS, Nitro,
  Docker, Supabase, and chain writes.
- Derive behavior and test sizes from the frozen candidate's configuration and
  protected-workflow contract. Do not duplicate fixed ICP counts, tree widths,
  stage inventories, or settlement prefixes in the runner.
- Start disposable PostgreSQL and apply settlement-critical SQL verbatim in
  production order. Exercise real constraints, views, functions, RPCs,
  pagination, uniqueness, and nonempty production-shaped state.
- A production/restart behavior change and its rehearsal coverage ship
  together. New external boundaries require a strict declared adapter.
- Test retries, exhaustion, cancellation, cleanup, concurrency, restart, and
  rollover. Every diagnostic needs an outer timeout and must clean clients and
  child processes.

## V2 runtime invariants

- Gateway and validator activate one exact attested SHA. Never mix files or
  restore gateway-only rsync deployment.
- No validator coordinator or worker starts until its immutable image and the
  gateway coordination marker, V2 health, build info, and release evidence all
  match that SHA. Recheck after startup.
- PCR0, attestation, import closure, release manifests, encrypted credential
  envelopes, and exact-commit checks remain fail-closed.
- Gateway produces one canonical epoch weight bundle. Primary and public audit
  validators retrieve and submit that exact bundle; they never recalculate it.
- Auditor identity is resolved by hotkey, not IP or stale UID.
- Failure and interruption clean validator containers, relays, Nitro enclave,
  and Docker lock, and publish a commit-bound failure marker.
- Rollback uses:

```bash
bash scripts/restart_attested_release_local.sh \
  --commit <full-40-character-attested-sha> \
  --local-python </absolute/venv/bin/python> \
  --component all
```

Single-component rollback is allowed only when the other runtime is already
verified on the same SHA.

## Incident standard

For restart, weight, auditor, attestation, cutover, or Research Lab incidents:

1. Within 15 minutes, map every downstream stage as passed, failed, safely
   exercised, or explicitly unexercised. Record the map before the first source
   edit; an unexercised product/trust invariant remains blocking.
2. Trace verification, persistence, handoff, signing, submission,
   finalization, and readback.
3. Inspect durable failures and statically trace blocked downstream stages.
4. Exercise independent later stages with sanitized production-shaped inputs.
5. Report all evidenced blockers and unexercised stages, not only the first
   exception.
6. At T+30 minutes, freeze one forward candidate or choose a compatible exact-
   attested rollback. At T+90, if the workflow is still unavailable, execute
   the compatible canonical rollback by default unless a precise migration,
   data, model, durable-state, or trust incompatibility is recorded, or a fully
   hermetically proven forward candidate is waiting only on bounded attestation
   or restart authority.
7. Preserve the last valid published baseline through a failed refresh. Serve
   it only while the exact deployed freshness/model/window policy says it is
   eligible; never extend freshness or cross model lineage during an incident.
8. Do not alter intended scoring or trust policy merely to silence a warning.

## Sentry API access for Codex

Codex is authorized to use the read-only Sentry API for deployment checks,
restart monitoring, incident debugging, and post-deploy validation. Use Sentry
alongside gateway and validator logs and durable/on-chain evidence; Sentry is
diagnostic and is never authority for V2 state or proof that a workflow
succeeded.

The API token lives in both production Secrets Manager environment documents
under `LEADPOET_SENTRY_API_TOKEN`. It is deliberately stripped from cached
environment files and runtime exports during restart. Do not retrieve it with a
raw `aws`, `ssh`, `env`, or `printenv` command. The authorized secure read is
encapsulated by the repository helper and does not display the token:

```bash
cd /Users/pranav/Downloads/Election_Analysis/Bittensor-subnet
python3 scripts/query_sentry_api.py auth-check --secret-source gateway
```

Use the validator copy only when the gateway host or secret is unavailable:

```bash
python3 scripts/query_sentry_api.py auth-check --secret-source validator
```

Query bounded, redacted recent issues and events with:

```bash
python3 scripts/query_sentry_api.py issues --secret-source gateway \
  --stats-period 24h --limit 25
python3 scripts/query_sentry_api.py events --secret-source gateway \
  --stats-period 24h --limit 25
```

The token must never be printed into chat, command output, logs, commits, test
fixtures, shell history, or process arguments. Never use `set -x`, `curl -v`,
or an inline bearer header with it. Do not add a raw-token output mode to the
helper. If the helper is unavailable, synchronize the canonical checkout with
`origin/main`; do not ask the user to paste the credential again.

## Research Lab and sourcing boundaries

- This repository owns Research Lab consumption, benchmarking, fulfillment
  infrastructure, gateway, and validator runtimes. `leadpoet/Sourcing_model`
  owns model semantics and industry taxonomy.
- Research Lab consumes the signed branch-specific
  `research-lab/sourcing-model/branches/leadpoet-lab/current.json` artifact.
  Verify repository, commit, signature, image digest, and lineage. Never fall
  back to the shared `main` pointer.
- Model-quality fixes belong in `Sourcing_model`; consumers may bind
  credentials and transport but may not reinterpret model-owned plans.
- Preserve exactly one official candidate entering paid scoring. Keep
  development/inner-loop scores isolated from promotion, rewards, payments,
  allocation, emissions, and weights unless the production contract explicitly
  advances the selected candidate.

## Routine handoff

Run the checks appropriate to touched files:

```bash
git diff --check
python3 -m py_compile <touched-python-files>
```

For Pydantic changes, round-trip JSON. For scoring changes, scan touched paths
for silent exception sentinels. Ask before adding production dependencies.
Report concrete evidence and any test not run.
