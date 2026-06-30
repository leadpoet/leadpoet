# Research Lab Flow Reference

## Intended Mechanism

The private sourcing model is improved by miner-funded autoresearch loops:

1. The gateway reads the current private model manifest.
2. A hosted autoresearch worker extracts the current model runtime from the image.
3. The LLM inspects model code and generates a code or prompt diff.
4. The gateway applies the diff, builds a candidate image, and writes a candidate artifact.
5. Scoring workers compare the candidate against the applicable current parent model.
6. Candidates that beat the threshold can become champion/promoted.
7. Miner reimbursement and lab incentives are based on valid compute, candidate outcomes, and promotion/reimbursement rules.

The goal is a compounding cycle: miners keep proposing model-code improvements, some fail, some score no gain, and the successful ones improve the production model used to find high-intent leads.

## Status Vocabulary

- `not_started`: ticket exists but no queue/run/receipt is visible.
- `queued`: run or candidate is waiting for worker capacity.
- `running`: hosted autoresearch is actively generating or building candidate work.
- `candidate_generation_complete`: autoresearch produced a candidate, but scoring has not finished.
- `waiting_for_baseline`: candidate cannot score yet because the matching private baseline is not ready.
- `needs_rescore`: candidate was built against a stale parent and must be rebased/rescored.
- `blocked_for_credit`: miner OpenRouter key cannot currently fund provider calls.
- `scoring`: candidate is actively assigned/evaluating.
- `scored_no_gain`: candidate scored, but did not improve enough.
- `scored_promising`: candidate scored a positive result but has not necessarily promoted.
- `promoted`: candidate became the current champion.
- `failed`: terminal failure before a scoreable candidate/result.

## Evidence Rules

- Concrete evidence: DB current rows, append-only event rows, hashes with retrievable artifacts, score bundles, promotion events, on-chain records.
- Weaker evidence: gateway log snippets, stdout/stderr, process status.
- Never claim a model improved unless a score bundle or promotion event supports it.
- Never claim a run is active from public dashboard text alone; verify queue/candidate current rows.
- Never recommend blindly resuming failed runs. Separate credit-blocked, baseline-not-ready, stale-parent, infra failure, and invalid-patch cases.
