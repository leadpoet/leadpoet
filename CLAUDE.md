# Agent Instructions

Keep this file and `CLAUDE.md` byte-identical.

## Delivery

- Make the smallest durable change. Carry forward the user's scope and approvals. Use existing code to resolve routine choices. New production dependencies require approval. Preserve user data, unrelated files and requested functionality unless removal is authorized.
- The owning task tests, pushes, merges and, where applicable, deploys and verifies live unless the user directs otherwise. Use canonical controllers, shared release locks and configured approvals. Wait for a held lock, then check what is already live.
- Fetch at start, resumption and integration or release. Inspect status, divergence and recent commits. Branch from the latest remote default or continue the existing PR. Verify the intended base and stacked parent before push or merge. Use git and its credential helper for pushes.
- Inspect concurrent diffs, work in a clean worktree when needed, and test that checkout. Integrate through commits or PRs. Never discard others' edits, force-push or rewrite shared history. Report overlap to the owning task. Delegation is optional.

## Verification

Trace callers, defaults, validators, stored state and failure handling. Reproduce bugs with a failing regression. Run focused checks, verify the affected flow, and report gaps. Run `git diff --check`, compile touched Python, round-trip changed Pydantic models through JSON, and check scoring changes for silent exception sentinels.

An independent, reproducible check against authoritative evidence suffices for a deterministic fact, such as a file hash or finalized payout. Measuring agreement with people requires genuine independent blind human ratings. Preserve existing security, migration, billing, audit and release checks. [Collection judging](docs/judge-output-spec.md)

Keep secrets out of source and diagnostics. Prefer authorized normal AWS access and verify the principal, account and permissions. Bound searches to relevant paths. Recursive searches of drive roots, user profiles, `.codex` or `AppData` require explicit scope. Use OnePatch only on request, with fresh scoped telemetry and direct evidence. Start monitors or schedules only when requested.

## Arena

Preserve frozen sources, ICPs, scorers, policies, accepted judgments, provider costs, provenance, results, rewards and weights. Verify a baseline's source against its intended finalized prior champion before execution or publication. Resolve pending corrections before freezing its successor. Record incomplete work and provider failures distinctly from negative results.

Keep original evidence, hashes, source/lease/receipt links, privacy, coverage rules and gaps. Apply separate checks for training eligibility, quality and permissions. Never fabricate historical evidence or pay to replay work solely to recreate proof.

| Surface | Existing controls |
| --- | --- |
| Scoring | [Integrity and costs](docs/arena-score-integrity.md), [company quality](docs/arena-company-quality.md) |
| Training | [Capture](docs/arena_training_data.md), [curation](docs/arena-training-curation.md), [full thinking](docs/arena-full-thinking-policy.md) |
| Releases | [Arena runbook](lab_arena/RUNBOOK.md), [gateway](docs/gateway_git_deployment_runbook.md), [paired restart](docs/arena_paired_restart.md), [attested builds](docs/attested-release-isolation.md) |
| Weights | [Verification checklist](docs/v2_deployment_verification_checklist.md), [normal validator](docs/arena_normal_validator_weights.md) |
| Authority | [Production access](docs/production-access.md), [key boundaries](docs/arena-iam-key-boundaries.md) |

## Gamma lead orders

Anyone holding Subnet 71 Gamma may order sales leads, paid for by miners'
provider keys and repaid from emissions ([contract](docs/gamma-lead-orders.md)).
Boundary:

- `LEAD_ORDERS_MODE` is `off` by default. `staff` and `shadow` take
  orders only from `LEAD_ORDERS_STAFF_HOTKEYS` and charge no Gamma.
  `shadow` computes and publishes rewards and never weights them. `live`
  needs Gamma enabled on netuid 71 and a change to this section that names
  the release turning it on.
- Money comes only from the signed policy record. A policy is refused unless
  `min_charge_ratio` exceeds `miner_premium` plus the largest provider
  discount. Lane rewards come only from the unallocated burn share, at most
  `lane_emission_cap` of miner emission per epoch. Arena rounds, results,
  rewards and weights stay unchanged when the lane is off or empty.
- The broker holds miner keys. The subnet's own key is always last, and every call
  is reserved before it is sent, settled at the provider-reported cost, and
  recorded. A call whose cost is uncertain is never charged or rewarded.
- Provider calls, including those carrying a person's data, may run on
  miners' keys (Gavin, Nov 19 2026). The broker uses only keys whose miner
  accepted the current miner data terms. The suppression feed holds only
  hashed emails and LinkedIn URLs, never raw values, and is readable by
  confirmed buyers and miners with accepted terms.
- Delivery files never pass through this repository or the gateway. The
  private runner delivers them. Receipts publish totals per key and a Merkle
  root, never prompts, models, tool names, call order or lead data.
- A failed order is never retried automatically. Setting the mode to `off`
  is the rollback.

## Historical instructions

[Archived instructions](docs/agent-instruction-history-20261025.md) apply only within their task, scope and expiry. They grant no new authority. The current request governs. Keep new task-specific details in the task record.
