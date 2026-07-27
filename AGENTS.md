# Leadpoet Research Lab agent guide

Applies to the entire repository. A deeper `AGENTS.md` overrides it within its
directory.

`AGENTS.md` and `CLAUDE.md` are one document kept in two files. They must
remain word-for-word identical. Every change to either file must update the
other in the same pull request, and CI must fail if they diverge.

## Repository role

This repository owns Research Lab consumption, benchmarking, fulfillment
infrastructure, and the gateway and validator runtimes. It does not own the
sourcing model's semantics.

Preserve unrelated local changes. Never reset, discard, overwrite, or include
another engineer's work. Never commit credentials, private model artifacts,
sealed benchmarks, customer data, provider payloads, or unredacted contact
data.

## Three-repository sourcing runtime contract

The sourcing runtime spans three repositories with independent activation
boundaries:

- `leadpoet/Sourcing_model` owns model behavior, the canonical industry
  taxonomy, runtime-capability semantics, and the `main` and `leadpoet-lab`
  artifact lineages.
- `gzaentz/leadpoet-site` owns the production wrapper, queues, verifier,
  persistence, release registry, and worker deployment.
- This repository owns Research Lab consumption and must resolve the
  branch-specific `leadpoet-lab` artifact, not the shared `main` pointer.

### Semantic ownership and consumer boundaries

A change is model-owned if it can alter candidate discovery, query construction,
ICP interpretation, branch enumeration, route eligibility or ordering, evidence
semantics, scoring, acceptance, rejection, resolution, or deduplication. This
remains true when the behavior is implemented around a host-bound provider such
as Deepline. Consumers may bind credentials and execute a model-owned plan, but
must not independently compile or reinterpret it.

`leadpoet-site` and `leadpoet` may translate a model-owned serialized contract
into host types, but the translation must be lossless and must not add, remove,
broaden, narrow, or reorder semantic constraints. Credentials, provider
transport, queues, leases, retries, cost controls, persistence, deployment,
verification, benchmarking, and publication remain host-owned operational
concerns.

`main` in `leadpoet/Sourcing_model` is the canonical source of shared model
behavior. `leadpoet-lab` periodically incorporates reviewed `main` changes and
publishes its own branch-specific artifact. Research Lab must consume that
artifact and must never reimplement missing model behavior locally.

A compatibility shim that reproduces model behavior outside `Sourcing_model`
requires an upstream tracking reference, parity fixtures, and an explicit
expiration or removal condition. It must be labeled temporary and must not be
treated as the permanent source of truth.

### Cross-repository invariants

- Drift baselines may shrink when duplicate consumer behavior is removed; they
  must never grow to make a check pass.
- Industry taxonomy is model-owned. Consumers may generate and verify
  byte-identical snapshots, but must never hand-edit or independently extend
  taxonomy values.
- Every model symbol a consumer imports, patches, or calls is a versioned
  contract term. Contract discovery and patch application fail closed when a
  required target is absent or ambiguous.
- `probe_origin: UNKNOWN` means "not yet proven" and must proceed with the full
  attempt budget; it never means dead and consumes no attempt by itself. Only
  an explicit dead result may stop that path.
- `DEFERRED_TRANSIENT` is nonterminal and must never be written to a permanent
  exclusions ledger.
- Never merge or advance a live branch pointer while a required check is
  failed, pending, or canceled.

Shipping code is not activation. A Sourcing_model `main` merge does not deploy
the site; a site repin does not serve until append-only registry promotion and
the hardened worker deployment both succeed; and Research Lab does not consume
a revision until `leadpoet-lab` advances and its branch-specific artifact
pointer is verified. Capability, resilience, or taxonomy modules remain inert
until their owning consumer explicitly wires and tests them.

## Research Lab artifact consumption

- Resolve the signed branch-specific
  `research-lab/sourcing-model/branches/leadpoet-lab/current.json` pointer.
- Verify the manifest signature, immutable commit identity, image digest, and
  expected repository before using an artifact.
- Never silently fall back to the shared `main` pointer.
- Never reconstruct missing model behavior in this repository. Fix it in
  `leadpoet/Sourcing_model`, review it on `main`, incorporate it into
  `leadpoet-lab`, publish the branch artifact, and then consume that artifact.
- Treat a missing, stale, unsigned, ambiguous, or mismatched artifact as a
  fail-closed condition.

## Required verification

Before handoff:

```bash
git diff --check
python -m compileall -q Leadpoet gateway leadpoet_audit leadpoet_canonical \
  leadpoet_verifier miner_models qualification research_lab validator_models
python -m pytest -q
```

Also verify that the instruction files are identical, required CI checks are
green, and any referenced Sourcing_model artifact points to the reviewed
`leadpoet-lab` commit. Do not claim artifact activation from a green unit test
alone.
