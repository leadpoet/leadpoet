# Arena decision summaries

Models can call `lab_arena_checkpoint.log_decision()` to record concise,
model-written explanations in the existing private trajectory table. The
baseline attaches these summaries to its existing research and final-output
calls. No extra model call, database table, provider credential, or validator
environment setting is required.

Each `runtime.decision` contains:

- `objective`: the immediate research goal.
- `decision`: `investigate`, `accept`, `reject`, `defer`, or `finish`.
- `evidence`: up to five references to already observed evidence; empty is valid.
- `rationale`: a short disclosed explanation of the choice.
- `next_action`: what the model intends to do next.
- `candidate`: an optional candidate name or domain.
- `source: model_reported`, a host sequence, and the last allocated provider
  action sequence. These help order the record; they do not prove the model's
  explanation or establish causality between concurrent calls.

Text fields are limited to 500 characters each and the complete JSON control
frame to 4 KiB, including escaped characters. Each execution retains up to 63
regular summaries and one reserved final `finish` summary. The runtime applies
the existing trajectory redaction rules. Do not
send hidden reasoning, credentials, full pages, or explanations invented after
the fact. These are model reports, not verified facts or access to a model's
internal reasoning.

The validator derives run, round, ICP, submission, and validator attribution
from the existing lease. It uploads bounded batches through the gateway. The
gateway stores them with its existing credentials. The helper never dispatches
a provider call or consumes provider quota. Capture is best effort: its boolean
result is not a research result and must not trigger a research retry.

At execution completion or a caught runtime failure, `runtime.decision_capture`
reports `provided` or `not_provided`, with recorded and omitted counts. Scorer
runs do not emit model decision summaries. Existing submitted model archives
are unchanged. They report no summaries until their authors adopt the helper.
Older validators must update to capture summaries; no additional configuration
is needed. An abrupt host/process failure can still prevent a buffered runtime
upload, as with the existing runtime logs.

The September 28 baseline adds this instrumentation to the promoted model
before the normal source freeze. It is a source change, with a small additional
output-token cost, even though it adds no provider turn and leaves the final
company output schema unchanged. The stored baseline source commit identifies
this adaptation. Future promoted model archives must also emit summaries to
provide the same coverage.

## Runtime source versions

Updated validators attach `validator_source_commit`, `validator_source_origin`
and `validator_source_dirty` to each execution/scoring `runtime.started` event
and to `lab_arena_runs.result_doc.resource_summary`. The same summary is in
`runtime.finished`. Existing run and trajectory columns supply the round,
submission, ICP position, attempt, run kind, model role and validator hotkey.
This covers baseline and miner jobs through the shared runner.

The source version is captured once when the process imports the runtime.
Canonical releases reuse their existing `.release-commit` file. Validators
running a Git checkout use its full HEAD SHA and report local changes. No
extra environment setting is needed. An unavailable version is `unknown`;
it does not block claims, execution, scoring, completion or rewards. Archive
releases report dirty state as `unknown`: the marker alone does not prove
that files are unchanged. These fields are reported audit metadata, not
cryptographic attestation or an eligibility check.

`gateway_claim_source_commit` records the gateway process that served the
lease. It is not the version of a later gateway that accepts completion.
Scoring start events also contain `scorer_image_reference`, the existing
frozen image digest. The host validator SHA must not be treated as the
scorer container's source commit.

Audit these fields with the existing private trajectory reader
`ArenaStore.list_trajectory_events(run_id)` and run reader
`ArenaStore.get_run(run_id)`. An early setup failure retains its start/error
trajectory even when it has no completion result. An abrupt process failure
can still lose a best-effort trajectory upload. Older records remain unchanged,
and external validators must update the shared runner to emit the fields.
The metadata helps identify the code to replay; it does not guarantee identical
future provider responses.

A reused judgment is not a new validator execution. For an existing
`cached_run_result.v1` result, follow `source_score_run_id` to the original
run's metadata. This preserves the actual executing validator identity and
does not claim a new execution for a cache hit.
