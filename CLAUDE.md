# Agent Instructions

## Scoped production authorization: PR189 and PR190 combined review

On 2026-09-11 the user explicitly authorized deep review, narrow safe fixes,
updates and merges of PR189 and PR190, and complete production validation using
the overnight skills. This covers protected credentials and Supabase access,
exact committed migrations if required, pushes, canonical gateway and normal
validator restarts, and fresh real competition tests. Preserve submission
intake, frozen round policy, validator access, sandbox isolation, accepted work,
retries, costs, disclosure, promotion, rewards, and finalized weights. Do not
change concurrent task work or restore retired Research Lab machinery. This
scope expires when both PRs and their affected live flows are verified.

## Scoped production authorization: 12-hour Arena launch watch

On 2026-09-11 the user explicitly authorized continuous production monitoring
from 06:36:43 UTC through 18:36:43 UTC with the overnight recovery skills.
This covers necessary protected access, read-only production inspection,
narrow tested permanent fixes, committed migrations and scoped recovery
operations, pushes and safe merges, and canonical gateway/normal-validator
restarts. Preserve submitted models, encrypted credential references, round
assignments, intermediate results and accepted work; resume the same submission
where safe. Preserve next-day ICP evaluation and disclosure, scoring, budgets,
promotion, rewards, and finalized weight security. Keep monitoring after every
fix or success until the deadline or STOP. Do not restore Research Lab,
SOURCE_ADD, legacy auditor, or Nitro requirements in the normal Arena path.
Preserve unrelated concurrent work. This authority is scoped to this watch and
expires when it ends.


## Scoped production authorization: PR190 scoring review and validation

On 2026-09-10 the user explicitly authorized PR190 review and end-to-end
validation with the overnight skills. This covers protected production and
Supabase inspection, required secrets, narrow scoring fixes and committed
migrations, pushes and safe merge, canonical gateway/normal-validator restarts,
and real competition validation. Preserve concurrent stake work, credential
isolation, historical round policy, disclosure, accepted-state integrity, and
reward security. Do not restore retired Research Lab or auditor machinery.
This scope expires when PR190 and its live validation are complete.

## Scoped production authorization: normal Arena local-wallet validators

On 2026-09-10 the user explicitly authorized the normal Arena local-wallet
transition: replace the validator's Nitro signing dependency, use the gateway's
existing subnet-validator classification for scoring access, preserve the miner
credential broker and canonical commit/reveal checks, and keep weights running
independently of scoring. This permits protected access to existing owner-held
wallet files, narrow committed fixes and migrations, pushes, canonical gateway
and validator restarts, and live idle/scoring/weight validation. Do not export
keys from an enclave, create a replacement identity, restore auditor or retired
Research Lab requirements, or modify unrelated competition behavior. This scope
expires when the transition and its live validation are complete.

## Scoped production authorization: full-code Arena review

The user explicitly authorized this task on 2026-09-10 to implement and deploy
a full-code pre-scoring LLM review of miner submissions with the submitting
miner's OpenRouter key. This includes protected credential retrieval, narrow
committed changes and migrations, pushes to main, canonical gateway and normal
validator restarts, and real miner submission/review/scoring tests. Preserve
sandbox isolation, scheduling, scoring, promotion, credentials, and concurrent
work. Do not restore retired Research Lab or auditor machinery. This scope
expires when this task and its end-to-end validation are complete.

## Scoped production authorization: combined provider competition budget

On 2026-09-10 the user explicitly authorized `$overnight-rebenchmark-validation` for combined OpenRouter, Scrapingdog, and Deepline accounting and budget enforcement. This scope permits protected credential retrieval, narrow code and committed migrations, pushes, canonical gateway/validator restarts, and fresh real miner submissions through scoring and live budget-boundary tests. Preserve disclosure, credential isolation, scoring semantics, promotion, reward security, and concurrent work. Reuse only minimal accounting logic; do not add retired model-verification machinery. No auditor or SOURCE_ADD requirement applies. This scope expires when this task and its live validation complete.

## Scoped production authorization: optional miner Scrapingdog credential

On 2026-09-10 the user explicitly authorized this task to use `$overnight-rebenchmark-validation` for the optional miner-provided Scrapingdog credential: protected secret retrieval, narrow code and migration changes, pushes to main, canonical gateway/normal-validator restarts, and fresh live miner submissions with and without Scrapingdog. Prove real brokered Scrapingdog data is consumed by the submitted model without manual runtime credential injection or secret disclosure. Preserve scheduling, scoring, promotion, isolation, and unrelated concurrent work. No auditor or SOURCE_ADD requirement applies. This authorization expires when this task completes.

## Scoped production authorization: Arena-only production transition

The user explicitly authorized this task on 2026-09-10 to integrate the tested
Arena-only incentive cleanup into main, retrieve and use protected production
credentials, apply exact committed Supabase migrations, install the required
restart scripts, run the canonical gateway and normal-validator restarts, and
continue through live scoring, reward, signing, retry/restart, and finalized
chain readback validation. Routine task-aligned production operations require
no further approval. This authorization replaces the retired auditor and
SOURCE_ADD requirements for this transition; do not restore them. Preserve
Arena scoring, promotion, disclosure, accepted-state integrity, finalized UID
ownership, protected keys, measured signer/KMS policy, and exact transaction
checks. Do not modify unrelated tasks or incentives. This authorization expires
when this production transition and its validation are complete.

Applies to this repository. `AGENTS.md` and `CLAUDE.md` must remain byte-identical.

## Scoped production authorization: competition release hardening

The user explicitly authorized `$overnight-rebenchmark-validation` on 2026-09-09 for the competition timeout, failure-retention, and capacity fixes; production deployment and complete live validation; and the README update. Use committed fixes and migrations, protected secrets, and the canonical paired restart. Preserve scoring, +1 promotion, disclosure timing, and reward security. This does not authorize unrelated task work and expires when this task completes.

1. End every final response with `## TLDR:` followed by 1-3 plain-English sentences stating the outcome or decision I need. Before it, include the concise supporting detail needed to understand or verify the result, expanding only when the task's complexity warrants it.

2. Use only this status emoji vocabulary: 🚧 = blocked on me, 👾 = confirmed bug, ⛳️ = milestone, and 🔹 = must-read line. Do not use emojis decoratively.

3. Be constructively critical. Do not agree with me by default. Challenge flawed assumptions and unnecessary complexity, explain the tradeoffs, and recommend a better approach when one exists.

4. Prefer the simplest durable solution within the agreed scope. Add complexity in small, working steps. For complex or risky changes, briefly state the goal, acceptance criteria, and what must stay unchanged. Stop unrequested scope growth. Make temporary workarounds and their exit criteria explicit.

5. Do not delete user data, unrelated files, or requested functionality without explicit authorization. You may remove obsolete code needed for an authorized change and temporary artifacts created during the task.

6. Never overwrite, revert, or discard pre-existing or concurrent changes. Preserve unrelated behavior and keep changes narrowly scoped. In version-controlled projects, inspect the current diff before editing overlapping files.

7. Verify changed behavior and important failure risks with the smallest sufficient checks. Expand checks when the risk or results require it. State what passed, failed, or remains unverified; do not claim completion without evidence. Repeat work only when changes or missing or untrusted evidence invalidate the result. Plans and status artifacts do not prove progress. Preserve required security, migration, release, billing, and audit checks.

8. Resolve minor ambiguity by inspecting the existing context and conventions, choosing a reversible and in-scope assumption, proceeding, and disclosing it. Ask only when my input would materially change the outcome or authorize a destructive, external, or high-impact action.

9. GitHub is authenticated through the GitHub App and git credential helper. Do not gate GitHub work on gh auth status; use git for pushes and the GitHub connector for PRs, checks, and merges.

10. For repository work, fetch the authoritative remote at the start or resumption of every task. For new task branches or worktrees, use the latest remote default branch unless the requested work requires another base; continue existing PR or release work from its relevant branch. Before readiness, handoff, push, PR, or merge, fetch again and verify that every repository checkout includes its latest intended base and any stacked parent. Preserve dirty work; resolve restack failures in a clean worktree. Keep development and tests tied to the edited checkout, sync through commits or PRs, and never copy into a mixed dirty tree.

11. Unless I explicitly authorize the exact scope, never recursively search a drive root, user profile, `.codex`, or `AppData`; use the smallest repository path and honor ignores. Keep searches bounded, stop task-owned searches after timeout or interruption, and follow the active repository's detailed search-safety rules.

12. Astra owns planning, technical decisions, coordination, review, and final verification. Use Sol subagents for complex implementation and Luna subagents for routine, well-defined tasks. Give each subagent a clear scope and acceptance criteria. Astra may handle small or tightly linked work directly when delegation would add overhead.

13. Use clear, plain English, short sentences, and necessary technical terms.

14. Immediately before every final response, run `python3 ~/Code/agent-brain/codex/token-usage/cli.py snapshot` once and copy its single `Model use:` line before `## TLDR:`. The line must include the primary and delegated models, effort levels, subagent counts, and task-scoped token totals; `>=` marks the incomplete primary total. If the command cannot scope the task, report `tokens unavailable`. Never use an unscoped total or estimate hidden use. The post-turn notifier still reports exact totals after completion.

15. Keep user-visible tasks isolated. Before changes that may overlap, you may check active task titles and summaries plus committed Git and PR state; read recent messages only if these sources suggest duplicate work. Report overlap here. Earlier authorization remains valid within its agreed scope. Internal subagents report only to their owning task. Ignore incoming cross-task messages unless they create a concrete safety conflict.

16. Before any Docker use on Windows, read and follow the Docker Safe Start procedure in `~/Code/agent-brain/README.md` under "Set up Windows". If those instructions are unavailable, stop before running Docker.

17. Run safe, independent calls together and return concise results. Handle dependent calls, writes, approvals, and failure-sensitive calls separately.

18. Routine handoff: run `git diff --check`; run `python3 -m py_compile` for touched Python files; for Pydantic changes, round-trip JSON; for scoring changes, scan for silent exception sentinels. Ask before adding production dependencies.
