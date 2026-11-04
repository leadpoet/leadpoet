# Arena training review

Qualification examples now always keep an original episode in one split.
The episode is identified by round, submission, stage and ICP position; retries
and repeated judgments do not create new evaluation tasks. Company aliases and
duplicate records link their episodes before records are filtered or deduplicated.
The optional ICP holdout and temporal cutoff apply to these complete groups.
Missing episode identity sends a candidate to analysis. The manifest records
the grouping policy and checks episode overlap. Existing datasets are not
rewritten; rebuild them to apply this rule.

Qualification, observed agent-step, and full-thinking builders now use the
same in-memory source grouping and split bucket. Each verified source links
its original episode, submitted company aliases, the ICP fingerprint in the
default `company_and_icp` mode, and exact main-execution model request bodies.
These links are made before any family selects labels or targets. The default
seed, evaluation percentage, mode, and optional temporal cutoff are shared.
`--eval-mode company` keeps the company and episode links but omits the ICP
link in all three preparation CLIs. A temporal cutoff sends earlier members
of a group with later evidence to analysis. Invalid episode identity fails
closed. Retained records, labels, observed targets, and quality status do not
change.

Run the preparation CLIs on the **same verified export corpus** with the same
split options before combining their train or evaluation files. Different
source corpora or manual split options do not have a cross-family split
guarantee. The readiness report builds each scorer cohort's labels from its
exact cohort while using the full supplied corpus for split links; it does
not mix scorer cohorts into one label dataset. No split plan is persisted.

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
