# Arena training review

Each dataset preparation CLI writes the original `train.jsonl`, `eval.jsonl`,
`analysis.jsonl`, and `manifest.json`, plus a private `curation.json` sidecar.
The sidecar uses `leadpoet.lab_arena.training_curation.v1`. It binds each flag
and retained qualification verdict projection to the exact raw record hash and
source example hash. It also binds the complete source split record hashes, so
editing a record and updating only its review decision cannot reuse an old
curation manifest. It does not change raw records, their labels, the dataset
version, split assignment, scoring, or competition eligibility. The training
permission remains `unconfirmed`.

Use `--review-sample-size N` to write `review.jsonl`. A reviewer may copy a row
to a decision JSONL file and set `human_review_status` to `accepted` or
`rejected`, with a nonempty `reviewer` and `review_reason`. For accepted
qualification records, set `reviewed_label` to the exact `original_label`.
For agent-step and full-thinking records, leave `reviewed_label` null. The
record ID, record hash, split file, source dataset version, curation version,
and source provenance fields must stay unchanged. Each decision binds the
review policy, flags, and retained verdict projection used to create the
sample. Missing or stale versions, unknown statuses, duplicate decisions,
changed provenance, and label changes fail the entire preparation.

Pass `--review-decisions decisions.jsonl` with the same source exports and
rules, and a new `--output` directory, to write a separate reviewed dataset.
Only accepted candidates enter its train/eval files. Rejected and unreviewed
candidates go to analysis. An empty decision file gives empty reviewed train
and eval files and exit code 2. The reviewed manifest binds the source dataset
and curation versions, decision hashes, and reviewed record hashes. Review does
not bypass the existing source, scorer cohort, privacy, evidence, or strict
thinking checks. A changed label needs a separate evidence-backed curation
contract; this version rejects it.
