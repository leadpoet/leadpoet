# Exact September 15 Arena baseline rerun

This is an operator procedure for `arena-2026-09-15`. It uses the original
September 15 twenty-ICP bank and the existing daily Arena driver. Do not start
it until the new Tyche baseline passes its paid twenty-ICP pilot and its exact
`main` and `lab` source commit is pushed. Keep the other rounds, miner rows,
provider calls, and accepted chain weights in place.

Migration `256` installs the guarded September 15 prepare and baseline-only
scoring RPCs. Migration `257` installs the separate open-only September 16
config adoption RPC. Applying either migration changes no round. Apply these
exact committed migrations before the canonical Store restart that uses their
RPC signatures. After the pushed lab source is staged at the write-once
`baseline-2026-09-15-rerun256.tar.gz` key and read back, create and apply a
separate exact committed release migration. It must insert one owner-only
`lab_arena_sep15_rerun_release_authority` row with the reviewed source SHA,
size, full lab commit, original bank SHA, forward schedule, and physically
verified worker slots. A service caller cannot set that authority.

Use `arena_sep15_exact_rerun.py` with the live scoped environment:

1. Run `stage-source --dry-run`, then `stage-source` with the full pushed lab
   commit, reviewed forward schedule file, and current verified worker slots.
   Compare the printed source SHA, size, commit, and bank SHA with the release
   migration before applying it. The source key is write-once.
2. Run `prepare --dry-run`, then `prepare` with the same arguments. The SQL
   preflight requires the original activated `no_king` basis, settled old
   baseline cost of 14,071,603 microusd, exact frozen work counts, the original
   bank, the owner release seal, and enough forward time for all twenty new
   executions and two scoring attempt waves. It archives only the forty old
   baseline runs and their ledger into a cancelled evidence round. It creates
   twenty new baseline execute assignments on the official September 15 round.
3. Let the normal driver and validators finish all twenty execute assignments.
   At each committed stage close, run `open-scoring --stage 1` or
   `open-scoring --stage 2`. The command derives the full normal scorer plan,
   then the exact RPC opens ten new baseline score assignments. Historical
   challenger judgments stay in place. The normal `score_stage` writes the new
   baseline scores and reuses each old challenger score only when its score and
   qualification receipt are exactly the same. The normal publisher then
   derives cost eligibility and the final ranking. September 15 publication
   requires a positive baseline final score and the original `no_king` reward
   decision. The old activated reward basis hash, document, signature key,
   epoch, and activation time remain unchanged.

If a command loses its RPC response, run `audit` before any replay. This
command reads only the current round, archive, run, and ledger metadata. It
prints no ICP or miner identity.

| Uncertain command | Audit state | Action |
| --- | --- | --- |
| `stage-source` | The source key contains the expected bytes and SHA. | Use those sealed bytes. Repeating `stage-source` with the same commit is safe. |
| `prepare` | Round is still `published`; archive has zero baseline runs; no new baseline assignments exist. | Retry `prepare` with the same proof while the forward schedule is still valid. |
| `prepare` | Round is `stage1`; archive has forty baseline runs and 14,071,603 microusd; official round has twenty new baseline execute assignments and 354 preserved challenger runs. | Prepare committed. Let the normal driver continue. Do not make another source or ledger transfer. |
| Stage 1 or 2 `open-scoring` | Round is still the matching `stageN_closed` with zero new stage score assignments. | Retry that stage command with the committed plan. |
| Stage 1 or 2 `open-scoring` | Round is `stageN_scoring` or later with exactly ten new score assignments for that stage. | Opening committed. Let the normal driver continue. The exact SQL RPC also returns `existing` on an identical replay. |
| `adopt-sep16` | September 16 still has the legacy open configuration. | Retry adoption only if open-only preflight still passes. |
| `adopt-sep16` | September 16 has the exact 45-minute marker and parallel configuration. | Adoption committed. Keep all accepted submissions. |

If the audit shows any mixed state or changed source/basis/cost, stop the
one-off command and inspect the owner-only audit in the protected read-only
operator path. Do not infer a commit from a timeout or repeat an unmatched
RPC. Each SQL RPC rejects a different replay and keeps the original ledger
prefix immutable. A later settlement for an existing uncertain challenger
call is allowed only when it names the original uncertainty entry; a new call
or unrelated ledger entry fails the scorer/publication seal.

September 16 adoption is separate. The current capacity formula uses the
configured runner ceiling; it does not prove physical worker availability.
Before adoption, confirm the three already-accepted challengers fit the
unchanged execution and scoring windows using actual worker slots. Parallel
execution closes both sets of ten ICPs before stage-one scoring starts. With
one runner hotkey and eleven available worker slots, baseline plus three
challengers therefore require eighty executions in the first phase: at least
eight full 45-minute waves, or fifteen waves (about 11.5 hours) if every
assignment uses its second attempt. The existing four-hour stage-one window
does not hold that retry reserve. Keep the original intake, submission, ICP,
and scorer policy unchanged; defer adoption until a reviewed exact timing
change makes the accepted intake feasible.
