# Sales sandbox training data

Freeze a private dataset from the existing verified export index on the operator host. The freeze includes successful runs, verified failures and released partials, along with the source and dataset hashes. A quarantine remains outside the dataset. A training manifest reports zero examples for an arm until its actual inputs exist.

```sh
python -m scripts.freeze_sales_sandbox freeze \
  --export-root "$HOME/.local/share/leadpoet/collection/watch/exports" \
  --output /private/path/sandbox-freeze
python -m scripts.freeze_sales_sandbox materialize \
  --bundle /private/path/sandbox-freeze --arm A7 --output /private/path/A7.jsonl
```

Use a new output path for each freeze or materialization. Pass `--snapshot DIGEST` to reproduce a specific immutable index. The private bundle contains each training trajectory once in `sources.zip`. Recipe files point into that ZIP. The case map counts passes, failures and incomplete runs separately by case, model and presentation. Earlier sandbox data is marked separately from v3.

| Arms | Inputs and use |
| --- | --- |
| A1, A2 | Untuned student passes. A2 uses full groups of eight with two to six verified passes. |
| A3, A4 | Pair a student's failure with its own pass or a teacher's pass on the identical initial task and tools. |
| A5, A6 | Identical close-teacher passes and action targets. A5 strips reasoning. A6 preserves complete short exposed reasoning on the same rows. The manifest records the threshold, default 256 words per turn. Use the Poet trainer’s `--train-short-reasoning 512` option for A6 to put short reasoning tokens under supervision. |
| A7 | Passing low-effort DeepSeek Pro trajectories with action targets. |
| A8, A9 | Verified pairs from the exact state before a student or expert mistake. Only the differing next action is a loss target. |
| A10 | Student passes and observed failures tagged with their first failed check. In the failure example the model predicts the check label from the trajectory. |
| A11 | Equivalent restatements from independent, authenticated untuned student draws at a verified teacher state. |
| A12, A13 | Balanced passes on three training presentations or two sandboxes. Materialize `--control` for an equal number of single-sandbox, base-presentation passes. |
| A14 | Complete groups of eight untuned student runs with native rewards. Each member retains its own original actions and run ID. |

Signed C3 and C6 inputs are optional. Supply the private producer bundles with repeated `--node-bundle` or `--restatement-bundle` arguments, the two approved source repositories with `--champion-repo` and `--poet-repo`, and the independently pinned `LAB_ARENA_COLLECTION_TRAINER_PUBLIC_KEY_FILE`. The verifier replays the original student run or compares the observed restatement with its frozen teacher state. An expert's success on a task cannot stand in for student evidence.

Use the same trained-token budget across comparisons. The pinned Poet trainer at `f4f006eb556f839b42d9f90b823654230cb627a1` reads SFT rows with `node/train_anchor.py`, preference arms with `node/train_preferences.py pairs`, and the A14 rows with `node/train_preferences.py groups`. Save the freeze's snapshot, recipe and source digests with the checkpoint and evaluation outcomes. C5 high-effort sweeps contribute to the case map only. Evaluation seeds, held-out sandboxes and the fourth tool presentation stay out of training.
