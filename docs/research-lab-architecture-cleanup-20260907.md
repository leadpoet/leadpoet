# Research Lab architecture cleanup

Research Lab now runs the public `lab` baseline and miner source submissions
through Arena. The gateway accepts signed submissions, persists the round and
work, and assigns work to validators. Baseline and miner participants use the
same frozen ICPs and scorer. Arena publishes the competition result.

## Removed runtime paths

- The disabled V1 branches and their unused evaluation and axis-rollup code.
- The sealed model-artifact upload RPC and its obsolete rehearsal transport.
- New loop reimbursement and private-model champion reward issuance. This
  includes participation, payment, rebate, grant, and staged-crowning helpers
  with no current application callers.
- The SOURCE_ADD Leg 2 implementation judge, scoring operation, provider
  profile, reward producer, and configuration.
- Provider outcome digests, checkpoint chains, snapshots, sidecar files,
  restart settings, and their runtime schema requirements.
- The old loop credit-resume queue rehearsal and its release checks. Historical
  migrations remain available for settlement and existing data.
- The unused Git-tree provider-call cap. Current request and provider quotas
  remain in force.
- Private-model budget soft stops that returned synthetic successful provider
  responses. Cost exhaustion and cost-tracking failures use the existing
  HTTP 402 refusal, accounting, and request cleanup paths.
- The stale private-model soft-stop documentation and redundant private-model
  error-class branch. Active event diagnostics retain the actual exception
  class, provider, HTTP status, and failure category.
- Uncalled loop-generated SOURCE_ADD suggestions and the old local P1.5
  sandbox-review, workflow-guard, IP-assignment, and bounty simulator.
- The disabled manual SOURCE_ADD CLI, its exclusive Docker trial and persistence
  writer, and its private trial-output contract. Active SOURCE_ADD manifests,
  intake records, measured V2 functional probes, and provisioning remain.
- The model sandbox rootfs marker and gVisor installation inside the gateway
  enclave. Arena executes on validator hosts and retains its verified gVisor
  runtime.

The dashboard change is in the separate `subnet_dashboard` repository. It
replaces private-model, loop, lineage, and release views with Arena round,
baseline, and winner data. SOURCE_ADD, settlement, emissions, validator
monitoring, and weight alerts remain active.

## Preserved active boundaries

| Workflow | Current boundary |
| --- | --- |
| Daily baseline and miner competition | `lab_arena/service.py`, `store.py`, `driver.py`, and the Arena SQL RPCs |
| Submission and execution | `gateway/api/arena_proxy.py`, `lab_arena/miner_submit.py`, source bundle validation, signed requests, runner leases, and gVisor |
| ICP scoring | `lab_arena/scoring.py` and the shared company competition scorer |
| Provider use | Provider transport, evidence cache, cost ledger, usage quotas, and failure handling |
| SOURCE_ADD | Provenance, acceptance, Leg 1 rewards, provisioning, encrypted credentials, work leases, and recovery |
| Settlement | Existing obligation projections, payment credits, allocation state, and finalized chain observations |
| Weights | Canonical vector construction, primary publication and recovery, and exact audit-validator mirroring |
| Restart | Canonical controllers, durable SOURCE_ADD guard and restore, runtime readiness, and credential recovery |

Historical reward records are not new reward producers. Existing reimbursement
and champion obligations still feed settlement and allocation fallback rules.
Their readers and deterministic arithmetic remain. All 43 retained functions
in `leadpoet_verifier/economics.py` were checked against the prior syntax trees;
none changed in this removal pass.

The remaining shared attestation and receipt code authenticates current
financial state, keys, and runtime operations. Historical role and purpose
values also validate existing settlement ancestry. These are not Arena model
admission requirements. Arena has no dependency on the gateway enclave modules
or a private-model commit, release manifest, or receipt graph.

## Database cutover

Migration 186 changes the active SOURCE_ADD status to `provisioned`. It leaves
append-only events, provision references, documents, and credential envelopes
unchanged. The current view normalizes the historical status. The migration
updates only the known SOURCE_ADD function identities and fails if their
expected contracts are absent. Startup checks must use the resulting function
contracts.

Apply only the final committed migration after its real database tests and
schema probe pass. Quiesce SOURCE_ADD for the schema/code cutover through the
canonical restart guard. Verify that the canonical restart restores the prior
durable control state. Do not delete historical SQL files or database rows.

## Acceptance evidence

Local validation covers Arena rounds through PostgreSQL and PostgREST, shared
ICP scoring, runner loss and recovery, provider replay and failure handling,
SOURCE_ADD, existing-obligation allocation, settlement, primary weights, and
audit mirroring. The final release must also pass the exact candidate's schema
and protected-code checks.

Local tests do not prove production completion. Release acceptance additionally
requires the canonical restarts, fresh runtime evidence, a real signed miner
submission scored through the gateway, database, and validator, and three
consecutive automated weight epochs for the primary and every required audit
validator. Auditor identities must come from deployment inventory joined to
their startup records. A dashboard monitoring list is not that inventory.
