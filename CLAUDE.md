# Agent Instructions

Applies to this repository. Keep `AGENTS.md` and `CLAUDE.md` byte-identical.

## Work and delivery

Make the smallest durable change that completes the user's request. Resolve minor
ambiguity from code and existing conventions. Ask only when a decision materially
changes the outcome or needs authority the user has not already granted. Preserve
the agreed scope and approvals across follow-ups. Do not delete user data,
unrelated files, or requested functionality without authorization.

Implementation requests include testing, pushing, merging, deployment where
applicable, and verification of the affected result, unless the user says
otherwise. The owning task completes these steps; it does not depend on another
chat's release handoff. Production operations use the existing authorized scope,
canonical controllers and shared release lock. Wait for a held lock, recheck live
state, and continue. Do not repeat a deployment already verified live.

Independent agent review and delegation are optional. Use them when they improve
the work; unavailable agent capacity does not block implementation or release.
Required repository checks and configured release approvals still apply.

## Git and concurrent work

Fetch at task start or resumption and before integration or release. Inspect
`git status --short`, divergence counts, and a bounded commit log; expand only
when needed. Start new branches from the latest remote default branch unless the
task needs another base. Continue existing PR work from its relevant branch.
Before push or merge, verify the edited checkout includes its intended base and
any stacked parent.

Preserve dirty and concurrent work. Inspect the diff before overlapping edits;
use a clean worktree when needed. Do not stash, reset, clean, overwrite, rebase,
or discard another change. Keep tests tied to the edited checkout and sync
through commits or PRs. Never force-push or rewrite shared history.

GitHub uses the GitHub App and git credential helper. Use git for pushes; do not
gate work on `gh auth status`.

Keep user-visible tasks isolated. Report overlaps and conflicts in this chat.
Do not send or relay cross-chat messages without a direct user request naming
the message and recipient. Internal subagents report only to their owning task.
Here, “coordinator” means that task's coordinator, not another chat.

## Verification

Trace the affected behavior through callers, defaults, validators, stored state,
and failure handling. For a defect, reproduce it and add a regression that fails
before the fix. Run the smallest useful checks for the change, then expand when
risk or failures require it. Verify the actual affected flow and preserved
behavior; report substitutions and unverified paths. Repeat checks only when
changes or missing evidence invalidate their result.

Run `git diff --check`; compile touched Python files. For Pydantic changes,
round-trip JSON. For scoring changes, check for silent exception sentinels.
Preserve required security, migration, release, billing and audit checks. Ask
before adding production dependencies. Passing tests, a plan, or a healthy
process alone does not prove the requested outcome.

## Arena contracts and relevant references

Preserve frozen round source, ICPs, scorer and policies; accepted judgments;
provider costs; private provenance; published results; rewards and weights.
Distinguish substantive negative results from incomplete work and provider
failures. Before baseline execution and publication, verify the frozen source
against the intended finalized prior champion and resolve pending prior-round
corrections before freezing the next baseline.

Keep raw capture, training eligibility, dataset quality and training permissions
separate. Preserve original evidence, hashes, source/lease/receipt links, privacy,
explicit gaps and the round's frozen coverage rules. Do not invent missing
historical evidence or replay paid work merely to manufacture verification.

Read the relevant existing document for the changed surface:

| Surface | Reference |
| --- | --- |
| Scoring, costs and evidence | [Score integrity](docs/arena-score-integrity.md), [company quality](docs/arena-company-quality.md) |
| Capture and training | [Training data](docs/arena_training_data.md), [curation](docs/arena-training-curation.md), [full-thinking policy](docs/arena-full-thinking-policy.md) |
| Host and scorer releases | [Arena runbook](lab_arena/RUNBOOK.md), [gateway deployment](docs/gateway_git_deployment_runbook.md) |
| Paired restart and recovery | [Paired restart](docs/arena_paired_restart.md) |
| Validator rewards and weights | [Verification checklist](docs/v2_deployment_verification_checklist.md), [normal validator](docs/arena_normal_validator_weights.md) |
| AWS access and signing authority | [Production access](docs/production-access.md), [key boundaries](docs/arena-iam-key-boundaries.md) |
| Attested builds | [Release isolation](docs/attested-release-isolation.md) |

These documents contain the existing controls; this file adds no release gates.
Keep secrets out of source and diagnostics. Use existing authorized AWS sessions
and verify the principal, account and required permissions before requesting
new access. Use normal operator access when it is sufficient.

Keep searches bounded to the relevant repository paths. Do not recursively
search a drive root, user profile, `.codex`, or `AppData` without explicit scope.
Use OnePatch only when explicitly requested; verify fresh scoped telemetry and
direct production evidence without starting unrequested monitors or schedules.

## Historical task records

Dated approvals and incident instructions are in the
[instruction archive](docs/agent-instruction-history-20261025.md). Read only the
relevant entry when continuing that task, including its scope, supersession and
expiry. Archive entries grant no new authority; the current request and its
limits take precedence. Keep new one-task details in that task's record rather
than appending them to these standing instructions.
