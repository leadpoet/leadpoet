# Candidate waterfall experiment sidecars

PR 96 adds candidate waterfall evidence to the shared PR 93 routing
experiment lifecycle. PR 93 remains the owner of experiment specs, signed
variants, provider receipts, decisions, evaluations, budgets, and promotion.
This change does not create a second route or promotion lifecycle.

## Artifact and target authority

Every candidate experiment has two signed artifacts: the baseline uses the
reviewed `main` artifact, and the challenger uses the distinct reviewed
`leadpoet-lab` artifact. A baseline `leadpoet-lab` artifact or a challenger
`main` artifact is rejected. The artifact key, branch, commit, manifest,
release identity, binding contract, and candidate-waterfall contract must
match the stored experiment spec and decision.

The exact target is supplied by the signed experiment input and the signed
Model unit input. The worker requires those two pre-dispatch values to match
before SQL or provider work. After the unit completes, the Model terminal must
also emit the explicit `target_verified_qualified_count`; it must match the
signed target before the terminal or any sidecar is persisted. The consumer
never derives the target from counts, a stop policy, a provider receipt, or a
decision.

## Independent Model unit-terminal authority

The Model completes only after all provider calls and decisions for a unit.
Therefore a provider-attempt row cannot contain the final Model waterfall.
After each exact `run_unit` completes and its decision exists, the worker
appends one claim-fenced row to
`research_lab_candidate_model_unit_terminals`. It appends this row before any
candidate waterfall sidecar. The row is immutable, forced-RLS protected, and
has one unique `(experiment_hash, variant_id, unit_ref)` identity.

The terminal row is a safe projection only. It contains hashes, IDs, ordered
provider and verification references, redacted counts, call/cost/latency
measurements, artifact identity, decision identity, and chain data. It never
stores provider, company, contact, raw response, credential, or prompt
payloads. It includes the terminal, Model receipt, orchestration, waterfall,
start-request, and candidate-plan hashes.

The signed Model waterfall and each signed attempt must explicitly provide:

- `target_verified_qualified_count`, `disposition`, explicit
  `provider_outcome`, `stop_policy_sha256`,
  `step_order`, `attempt_sequence`, `previous_attempt_sha256`,
  `verification_receipt_sha256`, `attempt_chain_sha256`, and
  `published_count`;
- top-level `published_count`, `verification_receipt_sha256`,
  `attempt_chain_sha256`, and `publication_projection_sha256`; and
- per-attempt `publication_projection_sha256`.

The adapter only recomputes these hashes to compare them with the explicit
Model values. It does not default sequence, map outcome to disposition,
compute the authoritative verification bundle or chain, or synthesize a
published count. A missing canonical field fails closed.

The current protected Sourcing_model artifact is not compatible with this
receipt: its `target_count` is not a lossless
`target_verified_qualified_count`, and it does not emit the explicit
`provider_outcome`, `disposition`, `step_order`, `attempt_sequence`,
`stop_policy_sha256`, verification receipt, attempt-chain, prior-attempt,
per-attempt/top-level publication, or publication-projection fields (its
attempt fields are only `attempt_index` and `previous_attempt_sha256`).
Activation of this PR is therefore blocked until the upstream artifact
contract emits them. Synthetic fixtures are labelled as synthetic and are not
activation evidence.

## Skips, verification, and chains

A skipped provider has no provider receipt, zero calls, zero cost, zero
latency, zero candidate counts, and an explicit skipped decision reason. A
terminal or sidecar that claims a provider receipt for a skipped tool is
rejected. An attempted provider must have exactly one matching durable
provider receipt, and an extra or missing receipt is rejected.

The ordered verification reference list must have exactly one reference per
verified-qualified candidate. Its explicit Model bundle hash must match the
ordered list. Duplicate, missing, or reordered references fail closed.

The ordered attempt list starts at zero. Each attempt carries its explicit
prior hash and chain hash. SQL independently recomputes each prefix only to
compare with the stored Model chain values. Missing, extra, reordered, or
tampered attempts fail closed. Totals and `published_count = 0` are checked
against every terminal projection.

## Append and promotion parity

Terminal append validates the appended unit against its signed spec, decision,
provider attempts, skipped reasons, references, counts, and explicit hashes.
Sidecar append validates every sidecar field against that durable terminal and
uses the same unit authority check. Promotion runs the whole-experiment
authority check: exact terminal coverage for every signed variant and
calibration/holdout unit, no extra terminal or provider attempt, complete
chains, and exact sidecar parity. Idempotent replay accepts the same document;
conflicting replay is rejected.

Provider attempts remain valid across recovery when their immutable attempt
key, parent experiment, provider receipt, authorization chain, and original
claim generation validate. A later worker claim generation may append the
terminal or sidecar after restart; requiring the old provider attempt to use
the new claim would reject valid recovery. The current append claim still
fences every new PR 96 write, and the immutable provider attempt proof is
revalidated before use.

This PR remains disabled and does not activate a model, route, provider, or
production release. Promotion and activation remain separate reviewed gates.
