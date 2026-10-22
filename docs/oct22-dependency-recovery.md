# October 22 unfinished execution recovery

This is an operator data repair after the general installer fix in PR 698.
It does not add a miner-specific runtime path. The submitted archive exceeded
the old installer workspace while meeting the unchanged final bundle limit.
An actual native SourceCache installation passed with the repaired workspace.
Nine historical attempts have retained ENOSPC evidence; eleven have no retained
exact cause. Do not rewrite those historical records or infer missing evidence.

The baseline and 20 fully judged miners are preserved. Only ten positions with
no accepted execution are eligible for this continuation. Submitted source,
ICPs, scorer, model, scoring rules, provider accounting and worker access remain
unchanged. The original final judging and publication deadlines remain in force.

## Application order

Use the authorized Keychain-backed migration helper with the exact reviewed
commit and file hash. Read back durable state after every application: a notice
or an already-applied response is not proof that a transition occurred.

1. Confirm all live workers contain PR 698, the restart control is clear at the
   expected generation, no unrelated work is active, and every preimage matches.
   Old pending rows in terminal rounds or shadow-only test rounds are preserved;
   they are outside live dispatch. Any leased/submitted work still blocks the
   repair, as does pending work in another nonterminal live or unknown-mode round.
   Apply 542 before its latest-start limit. It appends ten execution attempts and
   preserves all 665 existing runs. It extends only the execution close, leaving
   time for judging before the original final deadline.
2. Once all ten jobs are dispatched and at least one is leased, apply 543 and
   verify the exact owned transition hold. Leased execution and provider calls
   continue. A missed gate makes no change. Do not run an unattended gate poller.
3. Watch startup, provider calls, actual terminal outcomes and cost settlement.
   When all ten are terminal and all active work has drained, apply 544. It also
   supports the ordinary driver having closed execution first.
4. Inspect actual failure evidence. A `model_error` label alone does not justify
   a zero. Infrastructure failures remain incomplete. Build the 545 payload from
   the current native scoring builder and saved outputs. Preserve every other
   plan row. Review and commit the exact output-dependent SQL binding before
   application. Never substitute empty references or manufactured judgments.
5. Apply the bound 545 transaction. It appends judges only for accepted outputs,
   uses ordinary score persistence for proved genuine model failures, and
   restores the prior operator control. Missing judgments still block normal
   publication. Watch the real judges, persistence, costs and publication.
6. Stop any gate/prepare retry before an abort. Use 546 only to release this
   recovery's own hold. It preserves live leases and terminal records, works
   after the deadline and refuses a different owner. Guards in 543/544 prevent
   reacquisition after an unwind. An unwind is not a completed evaluation.

If fresh infrastructure failure prevents completion, retain NULL scores and
inspect the ordinary finalization result. Do not relabel it as miner failure,
reuse historical setup zeros, cancel healthy peers, or claim successful scoring.
Any further correction needs its own exact evidence, review and committed repair.

After the phase change, refresh the normal public submissions snapshot. A cache
rebuild must not be confused with deleted scores. Verify accepted run/output
bindings, historical ledger and judgment reuse remain intact. Continue the watch
through actual publication, promotion, incentives and all validators' weights.

## Verification evidence

Private task evidence is retained in `/private/tmp/arena-oct22-watch-evidence`.
It contains the native installation receipt, independent review, exact SQL
hashes and disposable PostgreSQL rehearsals using 38 native function definitions.
The rehearsals cover ten accepted executions and judges, eight accepted plus two
genuine model failures, partial/all infrastructure failures with NULL scores,
driver-close-first recovery, malformed payload rejection, owned-hold release,
replay and foreign-owner refusal. Existing rows, ledger and nonempty judgment
cache contents are compared before and after. These are local controls; live
execution and publication must be verified separately.
