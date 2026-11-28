# Registered sales experiments

V3 runs through the existing standing miner pool. The operator selects the tasks, model routes and limits. The original enrollment and verified actual-cost reward rules apply. Frozen historical manifests retain their original producer and grader.

## Starting a program

Use the approved champion repository, the private Poet repository and the production rubric. Stage a finite one-case C1 pack with the following command. Set the caps to the approved per-run amounts in micro-USD. Each assignment reserves only its own model's ceiling.

```sh
python -m scripts.prepare_collection_experiment prepare-v3 C1 \
  --family P9 --pack-count 1 --followups \
  --directory "$OFFERS" --inputs "$INPUTS" --prompt "$PROMPT" \
  --champion-repo "$AGENT" --poet-repo "$POET" \
  --cap-microusd "$DEFAULT_CAP" --cap-for "deepseek/deepseek-v4.1-flash-1017=$FLASH_CAP"
```

Omit `--pack-count 1` to stage the rest with the same settings. Repeat `--family` for P7, P8, P9 and P10. A changed budget or model configuration cannot buy a slot already staged. C1 assigns six approved open models four times on each of 200 cases per sandbox. C2 uses the close and far teachers on 100 evenly spaced matching cases, four times each on two trainable variants. Base, brief and review use the same underlying case ordinal. The fourth variant is held out. The pinned producer controls the true task identities and tool schemas.

Only C1 packs created with `--followups` request automatic research follow-ups. The existing watch service reconciles each durable private release after its normal checks. On a selected failed case, two independent experts continue twice each from the observed state. Repeated observations of the same student decision are deduplicated. For each of 100 paired ordinals, up to three funded-expert mistakes and four independently signed node-student mistakes may supply a correction state. The two source allocations keep both comparisons available. The high-effort C5 pass-rate sweep uses up to 50 cases per sandbox after all six teachers complete four verified low-effort trials, with 25–75% passing. C5 retains its results and payments for evaluation. Its assistant turns are masked from training and never enter preference pairs.

## Local students and restatements

On the producing node, use `sign-students --runs INDEX.json --family P9 --producer NAME --private-key PEM --output BUNDLE --champion-repo AGENT --poet-repo POET` with the same Python module. The private JSON array identifies each real native folder, producer run ID, student model and weight revision, and repeat 0–7. The reader replays the original tools and final grade, checks the training partition, and preserves the model's original exposed reasoning. The untuned student is Qwen/Qwen3.5-4B at `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.

Copy the signed bundle to the authorized training location. The gateway only needs the trusted public key configured in `LAB_ARENA_COLLECTION_TRAINER_PUBLIC_KEY_FILE`. Use `prepare-v3 C4 --student-bundle BUNDLE` with the same paths and spending caps to continue signed student failures. The full signed C3 dataset includes passing and failing runs. The hosted C3 command requires the exact student on an approved distillable zero-retention route. Today the approved provider roster has no such route.

Use `sign-restatements --runs INDEX.json --producer NAME --private-key PEM --output BUNDLE` for the original C6 node records. The training reader requires eight distinct, signed attempts at the same authenticated teacher checkpoint, replays that checkpoint and checks the model's actual action before selecting equivalent restatements. All private data stays out of the repositories.

## Registering additional sandboxes

The included catalog binds each existing sandbox to the reviewed producer revision, native verifier, immutable train/evaluation partition, route safeguards, variants and case-pattern count. A new trainable sandbox requires a signed registry record and six content-addressed gate artifacts. They cover a separate solver on 200 seeds, the observed teacher panel size and at least 90% passes with every miss reviewed, four failing lazy policies, at least 24 patterns, an untuned student pass rate of 15–75%, and a training partition distinct from evaluation.

The operator can install the reviewed record with `admit-sandbox --record FILE --public-key PEM --artifacts PROOFS --directory REGISTRY --ca-file RDS_CA --local-port TUNNEL`. The gateway and exporter read signed entries from `LAB_ARENA_SANDBOX_REGISTRY`. The service role has read-only catalog access. Funding requires matching registry identity in the worker and the database. An evaluation-only entry can be frozen without authorizing a paid training run.
