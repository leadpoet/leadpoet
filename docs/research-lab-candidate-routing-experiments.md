# Candidate-routing experiments

The candidate-routing experiment lane is replay-first and separate from the
official Research Lab candidate-evaluation and promotion rails.

`Sourcing_model` remains the only owner of route compilation, tool IDs,
profile policy, and route-plan hashes. This repository records the signed model
and catalog hashes, stores normalized route-step outcomes, and evaluates a
frozen provider snapshot. It must not compile a second waterfall or select an
unrecorded provider fallback.

Model-owned routing, catalog, profile, plan, stop-policy, attempt-receipt, and
verification-receipt hashes stay as the model's exact 64-character lowercase
SHA-256 values. Lab-owned record and snapshot hashes keep the Lab-native
`sha256:` prefix. This distinction prevents a consumer from silently changing
an identity at the repository boundary.

The stable service boundary is:

```python
from research_lab.candidate_routing_experiments import evaluate_routing_replay
```

Use `candidate_routing_attempt_from_model_receipt` to project a serialized
attempt. Its `model_runtime` argument must be the exact branch-specific model
runtime from the private-model runner or an isolated replay environment. The
Lab host does not import an unpinned model checkout.

Call `validate_candidate_routing_model_runtime` before a replay arm starts.
It requires the profile compiler, model-owned stop-policy compiler, waterfall
evaluator, catalog and policy builders, tool catalog, and signed receipt
parser. A partial or older model runtime fails closed.

`evaluate_routing_replay` accepts immutable experiment, arm, run, and attempt
contracts. An attempt contains only bounded counts, cost, latency, hashes, and
status codes. Provider payloads, credentials, raw ICP text, and scraped text
are not accepted. If a `ProviderSnapshotStore` is supplied, it must be strict
replay mode and its verified manifest hash must match the experiment.

The evaluator returns one metric and one routing-only decision per arm. The
decision states are `rejected`, `replay_only`, `eligible_for_shadow`, and
`eligible_for_canary`. These states do not activate a worker, move a model
pointer, create an official score bundle, or change rewards.

The additive migration
`scripts/156-research-lab-candidate-routing-experiments.sql` persists the
same contracts with append-only triggers, composite lineage foreign keys,
service-role-only access, and forced row-level security. It does not reference
official candidate evaluation tables.

Production activation is intentionally separate. A production release must
resolve the signed model artifact, bind every selected `candidate.*` tool to
an explicit reviewed provider adapter, apply the receipt migration, and pass
observe, shadow, and canary gates before it can change live sourcing.
