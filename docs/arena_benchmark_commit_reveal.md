# Benchmark commitments and Day 2 reveal

New rounds can opt into `commit_reveal_day2_v1`. Existing rounds keep their
stored policy. A missing policy retains the previous disclosure behavior;
null and unknown policies are invalid.

| Time | Public access | Assigned validator access |
| --- | --- | --- |
| Day 0, submissions open | Schedule and submission state | No benchmark assignments |
| Day 1, submission cutoff | Twenty salted ICP hashes and the manifest hash | Assigned ICP, its verification proof, and approved source for execution |
| Day 1, scoring complete | Aggregate scores, eligible source previews, promotion and normal rewards | Existing authorized work |
| Day 2, cutoff plus 24 hours | Exact ICPs, nonces and verification data; detailed results for published rounds | Normal assignment rules |

Reveal additionally requires a published or cancelled round. An active round
that overruns the reveal time stays private and reports `reveal_delayed`.
Cancellation before commitment reveals nothing. Cancellation after commitment
releases the benchmark at the scheduled time, but releases no source, scores,
or rewards. Source publication and promotion retain their completed-evaluation
gate; the benchmark delay does not postpone them.

The round ID/evaluation date remains Day 1. The stored `icp_set_date` remains
Day 0. Non-midnight cutoffs reveal at the same UTC time the following day.
The cutoff is a fixed timestamp; daylight-saving changes do not alter it.

## Verify a benchmark

Save the public commitment during Day 1, then download the revealed benchmark
on Day 2. Replace the URL and round ID with the round being inspected:

```bash
curl --fail "$GATEWAY_URL/arena/v1/rounds/ROUND_ID/benchmark-commitment" -o commitment.json
curl --fail "$GATEWAY_URL/arena/v1/rounds/ROUND_ID/benchmark" -o reveal.json
python3 scripts/verify_arena_benchmark.py commitment.json reveal.json
```

The verifier requires all twenty exact inputs, positions, network/round/date
scope, unique ICP IDs, independent nonces, and matching SHA-256 hashes. It also
checks that the displayed ICP fields match the committed inputs. Canonical
UTF-8 preimages are exported so Python and browser number formatting cannot
change the hashes. The dashboard performs the same hash verification.
A successful check proves equality to the saved commitment. The gateway's
`committed_at` timestamp does not independently prove when an observer first
saw the commitment; retain the original Day 1 download.

The hash-only endpoint never reads the private benchmark object. Before a
commitment exists it returns 409; for legacy rounds it returns 404. The
benchmark endpoint returns 403 until reveal is eligible. Successful and failed
responses carry `Cache-Control: no-store`. Published results return aggregate
`submission_scores` on Day 1 with `public_icp_status: pending` and empty detailed
arrays/outputs. Day 2 changes that status to `ready` and exposes detailed results.
The testnet public proxy keeps both benchmark routes private.

## Storage and recovery

Migration 210 adds the reveal timestamp, public commitment document and
database-generated commitment timestamp. It preserves the current global
schema marker and adds a private, scoped capability RPC. The v3 commit RPC
atomically freezes participants, bank/date metadata, source/scorer settings,
and commitment. The database prohibits early commits, commitment replacement,
policy changes, and bypass through older transition RPCs.

The private artifact is written first to
`arena/<round>/benchmarks/<sha256>.json`, then selected by the database. All
successful readers verify its bytes against its content-addressed path and
stored manifest. The validator additionally verifies its assigned preimage
before staging input. Proofs and nonces are not added to model input or its
environment. Completion and scoring signatures retain their existing format.

A lost upload acknowledgement is recovered by reading and comparing the exact
candidate bytes. A lost database acknowledgement reloads the committed winner.
Concurrent preparation may leave an unreferenced private candidate; it cannot
replace the winner. Restarts never regenerate a committed benchmark. Invalid
or missing selected artifacts fail closed with a safe error, without publishing
plaintext or converting infrastructure failure into a zero score.

## Rollout and rollback

Activation defaults off. Set `LAB_ARENA_BENCHMARK_COMMIT_REVEAL_FROM` to an aware
ISO timestamp for a future submission cutoff after deploying compatible readers.
Only rounds newly created at or after that cutoff get the marker. Existing open
rounds keep their saved configuration. Choose the cutoff before its submission
window is created, and ensure that bank was not already publicly exposed.
Removing the variable disables future opt-ins; it never changes committed rounds.

1. Install the new preserved gateway restart controller and verify its source
   identity before applying migration 210. The previously installed controller
   cannot enforce the new reader floor.
2. Apply the exact committed migration through the existing protected migration
   process. Keep existing rounds on their original policy during preparation.
3. Deploy the compatible gateway/service, validators and dashboard with activation
   unset. Use the canonical paired restart and normal release checks.
4. Verify schema capability, legacy reads, private routes, validator authorization,
   source access, scoring, and release compatibility; then configure a future cutoff.
5. Observe that round through Day 0 submission, Day 1 public hashes and assigned
   work, completed scoring/source/promotion/rewards, and the Day 2 verified reveal.

Once migration 210 is installed, the preserved controller refuses a target
without the exact disclosure capability declaration before stopping the running
service. Failed or uncertain capability probes also stop preparation. Recovery
uses a compatible forward fix; do not downgrade the schema, remove commitment
metadata, or restart an older reader after activation. See the
[gateway deployment runbook](gateway_git_deployment_runbook.md) for controller
installation and the release compatibility gate.

Validators assigned work, the coordinator, and required upstream providers see
private inputs. This change controls public application disclosure. Existing
source review, broker controls, validator authorization, budgets, final billing,
and reward security remain required. Stronger protection against deliberate
provider exfiltration is a separate change.
