# Leadpoet Research Lab Engine Program v0

Status: Phase 0 lab-only calibration program.

Purpose:

- Measure real loop depth before publishing a loop price or batch size.
- Exercise the minimal research-loop shape: hypothesize, patch, dev-eval,
  reflect.
- Produce schema-valid trajectory and results-ledger records from run one.

Operating constraints:

- Run only against frozen L1 fixtures.
- Run baseline-first on the reference fork before any candidate node is scored.
- Keep the qualification interface identical to the validator-facing
  `qualify(icp)` shape, but do not call production APIs, Supabase, gateway
  workflows, or live champion selection.
- Treat the fork as trusted lab code. Whole-file edits are allowed in v0, but
  this program is never miner-exposed.
- Every node must record cost, latency, cache hits, targeted metric, complexity,
  status, and a comparative reflection.
- A node is kept only if it improves paired dev-eval versus its parent without
  guardrail regressions. A score-neutral simplification can be kept only when
  complexity strictly decreases and no guardrail regresses.
- The output of this v0 program is calibration evidence, not a production
  promotion, crown, or payment event.

Default Phase 0 budget:

- Loop balance: 10.00 USD.
- Node draft cost: measured placeholder in fixture, not a charge to users.
- Node evaluation cost: measured placeholder in fixture, not a charge to users.
- Reflection cost: measured placeholder in fixture, not a charge to users.

Retirement condition:

- Engine v0 retires when the typed-patch v1 engine matches v0 yield on matched
  budgets and the component registry has been frozen from v0 diff-clustering
  evidence.
