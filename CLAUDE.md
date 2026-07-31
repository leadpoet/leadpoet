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

Before editing and again before committing or pushing:

```bash
git fetch origin
git status --short --untracked-files=all
git log origin/main..HEAD --oneline
git log HEAD..origin/main --oneline
```

- Fast-forward a clean checkout before work.
- If modified, staged, untracked, or unpushed work may belong to another chat
  or engineer, preserve it exactly and use a clean worktree at `origin/main`.
  Never stash, reset, clean, discard, overwrite, or rebase that work.
- Treat untracked files as local-only unless explicitly reviewed and in scope.
- Before push, fetch again. Integrate upstream safely and rerun affected fast
  checks. Never replace a newer remote file with a stale local copy.
- The task commit may contain only reviewed task paths.
- After push, fetch and verify `HEAD == origin/main`, no tracked diff remains,
  and the canonical checkout contains every tracked `origin/main` path.

## Secrets and production safety

- Never commit credentials, PEM contents, provider payloads, private model
  artifacts, sealed benchmarks, customer data, or unredacted contact data.
- Never perform production writes, restarts, process kills, Supabase
  mutations, or chain submissions. Give the operator exact commands.
- Read-only SSH is allowed. Use
  `/Users/pranav/Downloads/leadpoet-2026-07-28.pem` for:
  `ec2-user@52.91.135.79` and `ec2-user@100.59.201.156`.
- One HTTP 200 or a running process is not proof. Require joined logs, durable
  rows or receipts, identical bundle hashes, and chain readback as applicable.

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

## Default verification: 5-10 minutes

The default release gate is the bounded `prepush` profile. Run the long profile
only when the user's current request explicitly includes `un-accelerated` or
`unaccelerated` as the requested test/rehearsal mode. Do not infer the long
profile from words such as "thorough", "full", "production-equivalent",
"end-to-end", "all tests", or "release".

For ordinary changes:

1. Run syntax/format checks and only directly affected unit/regression tests.
2. For V2, gateway, validator, auditor, restart, durable-state, settlement, or
   weight changes, freeze the candidate SHA and run:

```bash
python3 scripts/run_local_restart_rehearsal.py \
  --from-sha <deployed-n-minus-one-sha> \
  --candidate-sha HEAD \
  --transition forward \
  --profile prepush
```

The prepush run has a 600-second target and must exercise:

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
stage `passed`, `failed`, or `unexercised`. A critical failed or unexercised
stage fails the gate. Per-stage and total duration must be reported. If the
scope or expected runtime exceeds ten minutes, tell the user before expanding
the work; do not silently start the long profile.

Do not run the whole repository suite by default when the focused tests and
prepush contract cover the change. CI may run broader checks after push.

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
  --component all
```

Single-component rollback is allowed only when the other runtime is already
verified on the same SHA.

## Incident standard

For restart, weight, auditor, attestation, cutover, or Research Lab incidents:

1. Trace verification, persistence, handoff, signing, submission,
   finalization, and readback.
2. Inspect durable failures and statically trace blocked downstream stages.
3. Exercise independent later stages with sanitized production-shaped inputs.
4. Report all evidenced blockers and unexercised stages, not only the first
   exception.
5. Do not alter intended scoring or trust policy merely to silence a warning.

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
