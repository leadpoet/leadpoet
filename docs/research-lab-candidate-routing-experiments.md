# Candidate waterfall experiment sidecars

This change is stacked on PR 93. PR 93 is the only owner of routing
experiments, variants, provider bindings, credit budgets, provider receipts,
route decision receipts, evaluation gates, and Lab promotion references.
Company sourcing and intent experiments therefore use one lifecycle and one
set of safety rules.

This PR adds five candidate-only parts:

1. `adapt_exact_model_candidate_receipt` accepts the terminal exact-runner
   result. It verifies the signed, model-owned candidate waterfall and every
   attempt in its hash chain. Each invoked attempt must name one independently
   persisted Lab provider receipt. Tool ID, provider outcome, call count,
   billed credit microunits, latency, and receipt coverage must agree exactly.
   A missing, extra, reused, or inconsistent provider receipt fails closed.
2. `candidate_waterfall_receipt_from_model` remains the typed adapter for the
   existing serialized candidate-waterfall contract. It gives the exact plan,
   stop policy, and full receipt prefix to the `Sourcing_model` evaluator. It
   attaches the current attempted step to the existing V2 provider and
   decision receipts. It attaches a skipped step to the existing
   skipped-decision reason without inventing a provider receipt. Prefix hashes
   prevent sidecars from different valid executions from being mixed. Provider
   call count, billed credits, and latency come from the authoritative provider
   receipt. A different Model call count, cost, or latency fails closed.
3. `evaluate_candidate_waterfall_metrics` derives calibration and holdout
   metrics for raw, normalized, unique, verified-qualified, and published
   companies. Cost efficiency is measured in billed provider credits, not a
   Model USD estimate. It requires complete sidecar coverage for every provider
   and decision receipt in the shared evaluation. It rejects a reused provider
   receipt, a different compiled target, or a broken attempt chain, so omitted
   or duplicated outcomes cannot improve the metrics. These metrics are
   sidecars. They do not select or promote a route.
4. `validate_candidate_routing_model_runtime` uses PR 93's pinned Model
   adapter to verify the exact artifact identity, signed artifact manifest,
   catalog hash, policy hash, and runtime-exported candidate waterfall
   identity. The candidate execution contract is a separate Model contract
   and is not compared with the general routing-contract hash. A partial,
   unsigned, unsafe, or different runtime fails closed before a receipt is
   accepted; replay and measured Lab runs also require PR 93's cryptographic
   artifact authority.
5. `scripts/162-research-lab-candidate-routing-experiments.sql` stores only
   the Model receipt sidecars and candidate metric sidecars. Both tables are
   append-only, service-role-only, and protected by forced row-level security.
   Foreign keys bind them to PR 93 experiments, decisions, and evaluations,
   including the shared experiment hash. A global partial unique index
   prevents provider-receipt reuse. Each stored JSON document must exactly
   match its indexed scalar columns and content hash.

The adapter never compiles a route, calls a provider, selects an unrecorded
fallback, or writes a second promotion decision. Use PR 93's
`evaluate_routing_experiment_v2` for provider execution and route evaluation.
Use `promote_routing_experiment_v2_to_lab` for the immutable Lab reference.

The identity rules are explicit:

- Both baseline and challenger artifacts are exact signed `leadpoet-lab`
  artifacts. The baseline is the admitted/current artifact and the challenger
  must have a distinct exact artifact identity; the Lab never accepts `main`.
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
