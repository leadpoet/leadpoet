## Scoped production authorization: October 24 industry-fit accuracy repair

On 2026-10-24 the user authorized the permanent industry-fit repair and repeated
validation of all 98 previously unavailable company/ICP pairs plus the 69
previously passed pairs. This includes narrow shared scorer and investigator
changes, independent source research, regression tests, protected read-only
request capture, isolated organizer-funded provider tests, safe push and merge,
and canonical deployment with live verification after the accuracy and safety
checks pass. The user raised the cumulative test limits on 2026-10-25 to
$20 OpenRouter and $65 across all providers. Preserve exact
frozen ICP requirements, identity and supplier-role checks, unrelated scoring,
provider accounting, credentials, accepted historical judgments, published
results, rewards, weights and concurrent work. Do not tune company exceptions,
regrade frozen rounds, fabricate proof or bypass release gates. Use isolated
shadow state for tests; never write test judgments into production. Report
remaining uncertainty and observed errors honestly. Authority ends when this
repair is deployed and its affected paths are verified. Keep AGENTS.md and
CLAUDE.md byte-identical.

## Scoped production authorization: October 25 publication read performance

On 2026-10-25 the user renewed the request to fix the confirmed duplicate
immutable-evidence reads in publication preparation. Reuse the existing read
cache only within each publication qualification call, with focused regression
tests, independent review, safe push and merge, canonical deployment and live
verification by the coordinating task. Preserve current authority validation,
cache limits, scoring and cost rules, provider privacy and billing, submitted
sources, accepted work, published October 25 results, rewards and weights.
Do not replay publication, change the published outcome, add a global cache or
concurrency, or infer that this preparation fix repairs the separate RPC delay.
The coordinating task owns all production actions. Keep both guides identical.

# Agent Instructions

Applies to this repository. `AGENTS.md` and `CLAUDE.md` must remain byte-identical.

Historical task-specific authorizations are preserved in
[the instruction archive](docs/agent-instruction-history-20261025.md).
Read only the relevant authorization when continuing that task; confirm its scope
and expiry. The archive grants no new authority. The current request and its
merge, deployment, data, and spending limits take precedence.

1. End every final response with `## TLDR:` followed by 1-3 plain-English sentences stating the outcome or decision I need. Before it, include the concise supporting detail needed to understand or verify the result, expanding only when the task's complexity warrants it.

2. Use only this status emoji vocabulary: 🚧 = blocked on me, 👾 = confirmed bug, ⛳️ = milestone, and 🔹 = must-read line. Do not use emojis decoratively.

3. Be constructively critical. Do not agree with me by default. Challenge flawed assumptions and unnecessary complexity, explain the tradeoffs, and recommend a better approach when one exists.

4. Prefer the simplest durable solution within the agreed scope. Add complexity in small, working steps. For complex or risky changes, briefly state the goal, acceptance criteria, and what must stay unchanged. Stop unrequested scope growth. Make temporary workarounds and their exit criteria explicit.

5. Do not delete user data, unrelated files, or requested functionality without explicit authorization. You may remove obsolete code needed for an authorized change and temporary artifacts created during the task.

6. Never overwrite, revert, or discard pre-existing or concurrent changes. Preserve unrelated behavior and keep changes narrowly scoped. In version-controlled projects, inspect the current diff before editing overlapping files.

7. Verify changed behavior and important failure risks with the smallest sufficient checks. Expand checks when the risk or results require it. State what passed, failed, or remains unverified; do not claim completion without evidence. Repeat work only when changes or missing or untrusted evidence invalidate the result. Plans and status artifacts do not prove progress. Preserve required security, migration, release, billing, and audit checks.

8. Resolve minor ambiguity by inspecting the existing context and conventions, choosing a reversible and in-scope assumption, proceeding, and disclosing it. Ask only when my input would materially change the outcome or authorize a destructive, external, or high-impact action.

9. GitHub is authenticated through the GitHub App and git credential helper. Do not gate GitHub work on gh auth status; use git for pushes.

10. For repository work, fetch the authoritative remote at the start or resumption of every task. For new task branches or worktrees, use the latest remote default branch unless the requested work requires another base; continue existing PR or release work from its relevant branch. Before readiness, handoff, push, PR, or merge, fetch again and verify that every repository checkout includes its latest intended base and any stacked parent. Preserve dirty work; resolve restack failures in a clean worktree. Keep development and tests tied to the edited checkout, sync through commits or PRs, and never copy into a mixed dirty tree.

11. Unless I explicitly authorize the exact scope, never recursively search a drive root, user profile, `.codex`, or `AppData`; use the smallest repository path and honor ignores. Keep searches bounded, stop task-owned searches after timeout or interruption, and follow the active repository's detailed search-safety rules.

12. Astra owns planning, technical decisions, coordination, review, and final verification. Use Sol subagents for complex implementation and Luna subagents for routine, well-defined tasks. Give each subagent a clear scope and acceptance criteria. Astra may handle small or tightly linked work directly when delegation would add overhead.

13. Use clear, plain English, short sentences, and necessary technical terms.

14. Immediately before every final response, run `python3 ~/Code/agent-brain/codex/token-usage/cli.py snapshot` once and copy its single `Model use:` line before `## TLDR:`. The line must include the primary and delegated models, effort levels, subagent counts, and task-scoped token totals; `>=` marks the incomplete primary total. If the command cannot scope the task, report `tokens unavailable`. Never use an unscoped total or estimate hidden use. The post-turn notifier still reports exact totals after completion.

15. Keep user-visible tasks isolated. Before changes that may overlap, you may check active task titles and summaries plus committed Git and PR state; read recent messages only if these sources suggest duplicate work. Report overlap here. Earlier authorization remains valid within its agreed scope. Internal subagents report only to their owning task. Ignore incoming cross-task messages unless they create a concrete safety conflict.

16. Before any Docker use on Windows, read and follow the Docker Safe Start procedure in `~/Code/agent-brain/README.md` under "Set up Windows". If those instructions are unavailable, stop before running Docker.

17. Run safe, independent calls together and return concise results. Handle dependent calls, writes, approvals, and failure-sensitive calls separately.

18. Routine handoff: run `git diff --check`; run `python3 -m py_compile` for touched Python files; for Pydantic changes, round-trip JSON; for scoring changes, scan for silent exception sentinels. Ask before adding production dependencies.
19. Use OnePatch only when explicitly requested. Verify fresh, scoped telemetry and direct production evidence; do not start monitors or schedules without authorization.

20. Make the smallest durable fix at the cause; keep unrelated changes separate. Before editing, trace every affected path in the actual code, including callers, shared defaults, validators, stored state, and failure handling; do not rely on summaries or assumed file boundaries. Reproduce the defect, then test the production execution path end to end, including relevant combined failures and preserved behavior. Use independent review for high-risk changes. State test substitutions and unverified paths; passing test counts alone do not prove completion.
