# September 16 native TYCHE baseline rerun

This procedure reruns only `baseline-2026-09-16` on the published September 16
round. It keeps the original twenty ICPs, miner results, cost history, signed
reward basis, and promotion state. The normal Arena driver still executes,
scores, aggregates, and publishes the round.

Do not start until the paid twenty-ICP pilot passes, the tested
`leadpoet/champion_model` commit is the head of both `main` and `lab`, and the
same scorer image has passed its release checks.

1. Apply inert migration `265` with the matching service release. This changes
   no round rows.
2. Create the forward schedule. Keep the original `submission_open` and
   `submission_cutoff`; reserve two 45-minute execution waves and two scoring
   attempts per stage with ten verified runner slots.
3. Run `stage-source --dry-run`, then `stage-source`. Supply the full tested
   `champion_model` lab commit and the reviewed schedule. The helper fetches
   only the explicit public lab archive, validates the source bundle and
   commit, writes the new object key once, and reads it back.
4. Reread the protected Sep16 state after billing reconciliation. Render,
   review, commit, and apply owner migration `266`. It inserts only the exact
   release authority.
5. Run `prepare --dry-run`, then `prepare` with the same commit and schedule.
   The RPC archives only the old baseline evidence and creates twenty fresh
   baseline execution assignments.
6. Leave the normal driver and runners active. When a stage closes, the driver
   commits its normal scoring plan. Migration `265` rejects the driver's
   generic scoring INSERT for this prepared rerun, so it cannot race the
   exact route. Run `open-scoring --stage 1` or `open-scoring --stage 2`. The
   helper derives the normal plan and sends it to the baseline-only RPC. The
   driver then continues normal scoring and publication.
7. If a mutating command loses its response, run `audit` before replay. It
   returns only round status and assignment counts.

Example command shape:

```bash
python3 scripts/arena_sep16_native_rerun.py \
  --environment-file /protected/path/lab-arena.env \
  stage-source \
  --expected-lab-commit 0123456789abcdef0123456789abcdef01234567 \
  --forward-schedule-file /protected/path/sep16-forward-schedule.json \
  --dry-run
```

All commands require `LAB_ARENA_MODE=live`. The helper prints no ICP, miner,
credential, or provider-response content.
