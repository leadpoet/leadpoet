# Admission refusal recovery: failure map

Target: complete genuine admission review after a provider refusal. One refusal
starts an episode. Its initial primary call counts as the first of at most two
primary reservations. Earlier ordinary transient attempts do not consume this
allowance. The episode then permits at most two full-source alternate reservations.
No completed pass/reject is repeated. A valid verdict with missing cost stays an
explicit terminal error with its conservative uncertain charge.

| Path | Required result | Reproducible proof |
| --- | --- | --- |
| Source/security/digest failure | No approval or provider retry | Existing source checks and terminal-chain controls |
| First primary refusal | Preserve charge; retry primary once | Typed refusal runtime and PostgreSQL sequence |
| Seven transients before first refusal | Start episode at refusal8, permit primary9 | Real PostgreSQL episode-origin regression |
| Second primary refusal/timeout | Use alternate, never restart primary | Mixed-failure sequence and route mismatch controls |
| First alternate refusal/timeout | Permit one independent alternate attempt | Runtime and PostgreSQL tests |
| Second alternate failure | Terminal incomplete result | SQL counts, shared policy and stale-worker controls |
| Crash during primary retry or either alternate | Preserve uncertain cost, origin and consumed reservation | Expiry regressions at positions2/3/4 |
| Legacy generic HTTP200 rejection | Latest row and terminal ledger authoritative refusal pair must match current deterministic call/reservation and primary model | Positive missing/untyped optional diagnostics; negative403, wrong model, old proof and finalized-round tests |
| Worker/scheduler/public reads | Same legacy eligibility without stored rewrites | Central Store get/list hydration controls |
| Unknown cost after valid verdict | Clear reviewing lease; preserve uncertainty; no paid replay | Real runtime+PostgreSQL at primary1, primary2, alternate1 |
| Valid pass/reject | Stop permanently | Completed-decision and idempotent ledger checks |
| Restart guard/deadline/cutoff | Existing admission gates remain authoritative | Preserved gate tests and unchanged SQL seams |

The alternate preserves every source file, research and similarity context,
source binding, strict response validation, private credential boundary and ZDR
policy. It requires fresh direct routing evidence, exact model and Azure identity,
no hidden attempt/compression, <=922,000 input tokens and input plus capped output
within1,050,000 tokens. Conservative pricing reuses the existing Sol Azure ceiling.

The revised, undeployed migration599 changes only exact existing function bodies
and capability metadata. It adds no table, column or RPC signature. The durable
`refusal_episode_start` and reservation lineage survive errors and expiry. Legacy
HTTP200 classification uses the old runtime's authoritative error pair; optional
provider diagnostics remain observations. Finalized rounds never hydrate for
recovery and all original ledger records remain unchanged.

Validation:388 affected Python and real PostgreSQL tests passed, including
migration before/after, identical reapply, failed-upgrade rollback, non-superuser
ACL preservation and prior unrelated-row snapshots. Compilation, instruction-file
equality and `git diff --check` passed. No provider calls or production mutations
were made by this implementation task. The coordinator owns final independent
review, release, current provider evidence and deployed-output verification.
