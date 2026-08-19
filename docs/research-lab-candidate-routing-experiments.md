# Candidate waterfall experiment sidecars

This change is stacked on PR 93. PR 93 is the only owner of routing
experiments, variants, provider bindings, credit budgets, provider receipts,
route decision receipts, evaluation gates, and Lab promotion references.
Company sourcing and intent experiments therefore use one lifecycle and one
set of safety rules.

This PR adds four candidate-only parts:

1. `candidate_waterfall_receipt_from_model` parses the exact
   `Sourcing_model` `CandidateStepAttemptReceipt`. It attaches attempted steps
   to the existing V2 provider and decision receipts. It attaches a skipped
   step to the existing skipped-decision reason without inventing a provider
   receipt.
2. `evaluate_candidate_waterfall_metrics` derives calibration and holdout
   metrics for raw, normalized, unique, verified-qualified, and published
   companies. These metrics are sidecars. They do not select or promote a
   route.
3. `validate_candidate_routing_model_runtime` compares the isolated Model
   runtime's candidate waterfall contract hash with the exact artifact hash
   in the V2 experiment variant. A partial, unsafe, or different runtime fails
   closed before a receipt is accepted.
4. `scripts/156-research-lab-candidate-routing-experiments.sql` stores only
   the Model receipt sidecars and candidate metric sidecars. Both tables are
   append-only, service-role-only, and protected by forced row-level security.

The adapter never compiles a route, calls a provider, selects an unrecorded
fallback, or writes a second promotion decision. Use PR 93's
`evaluate_routing_experiment_v2` for provider execution and route evaluation.
Use `promote_routing_experiment_v2_to_lab` for the immutable Lab reference.

The identity rules are explicit:

- PR 93 Lab hashes use the `sha256:` prefix.
- Model route, stop-policy, attempt, and verification hashes are exact
  64-character lowercase SHA-256 values.
- The receipt adapter converts the Model plan hash only for comparison with
  the shared decision receipt. It preserves the original Model hash in the
  candidate sidecar.
- Provider payloads, credentials, raw ICP text, scraped text, and secrets are
  not stored.

This PR does not activate a production route, change a model pointer, or add
a production provider binding. Observe, replay, shadow, canary, and production
activation remain separate reviewed steps.
