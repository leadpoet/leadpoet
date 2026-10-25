# Reasoning and trajectory capture audit — 2026-10-23

This is a draft PR audit, not a production deployment. The user authorized up to
$8 of OpenRouter tests. All paid requests use synthetic company evidence. No
production miner input, database write, restart, or round change was used.

## Evidence and acceptance

The audit covers raw structured capture, private export, readable rendering,
worker delivery, tool continuations, and structured final output. A complete
labeled sales trajectory also needs a real task, final submission, and accepted
judgment. The diagnostic probes do not invent those records or mark themselves
as complete training examples.

Every route in the draft runtime registry passed fresh real calls through:

1. The actual Broker and HTTP provider transport, with ZDR and exact provider pin.
2. Actual ArenaStore PostgreSQL reservation, dispatch, and settlement RPCs in a
   disposable local database; the SQL ledger is read back separately.
3. Request capture in a private local object store, reopened by the exporter.
4. The real local WorkerSocketServer and SDK response-read acknowledgement.
5. Two model/tool continuations and final strict JSON for a company whose newer
   headcount contradicts its older profile. Every route returned the correct
   negative qualification and the required evidence IDs.
6. Hash-checked raw export and a readable view with source paths, returned text,
   tools, usage, costs, delivery proof, and timing.

The socket adapter delegates directly to Broker; it is not the deployed gateway
HTTP service or a Linux sandbox. No claim of production rollout, complete ICP
quality, or publication readiness is made.

## Whitelist results

| Model | Exact provider | Decision |
| --- | --- | --- |
| GPT-OSS 120B | DeepInfra BF16 | Retain; live capture and both tool continuations pass |
| Mistral Large 4 | Mistral ZDR | Retain; final thinking saved; empty tool turns remain empty |
| DeepSeek V4.1 Flash 1017 | Fireworks | Retain; live capture and both tool continuations pass |
| GPT-OSS 20B | DeepInfra BF16 | Add; positive and negative cases plus worker readback pass |
| Reflection Beam 1.0 501B A23B | Nebius FP8 | Add; positive and negative cases plus worker readback pass |
| Gemma 4 31B IT | CoreWeave FP4 | Add; positive and negative cases plus worker readback pass |

These are exact route approvals. No model family or arbitrary fallback is
approved. Generated text is the provider-exposed unsummarized token sequence;
it does not prove access to all internal computation. A model can return no
thinking on an individual call. Empty turns are counted separately.

Additional live candidates were not admitted:

| Model/route | Observed limit |
| --- | --- |
| MiniMax M3.1 Flash / InferenceNet FP4 | Tools and JSON passed twice, but no readable reasoning was returned |
| NVIDIA Nemotron 3.5 Lightning / DeepInfra BF16 | Forced-tool request rejected by provider routing |
| Cohere Command A+ / Cohere | Required parameters rejected by provider routing |
| Mistral Small 2603 / Mistral ZDR | One 429; another probe returned two tools at once, so the required sequential continuation proof is incomplete |

The inventory contains many untested candidates. A catalog capability flag is
not end-to-end evidence. No claim is made that all OpenRouter models were tested.

## Hosted model diagnostics and ZDR

- OpenAI GPT-5.6 Luna / Azure completed three turns after the continuation-field
  fix. Its second turn returned a 381-character summary plus opaque state. All
  three usage records retained reasoning-token counts (28, 81, 55). ZDR stayed
  enabled. This is a summary-only diagnostic, not a full-thinking approval.
- Gemini 3.5 Flash Lite / Google Vertex completed three turns. The final turn
  contained 407 characters of visible text alongside opaque state. Both layers
  were accounted for; this does not establish full generated thinking.
- Claude Sonnet 5.5 / Google Vertex completed both tools and strict JSON using
  automatic tool choice, but returned no thinking in these three calls. This is
  `requested_empty`, not evidence that the model is universally unsupported.

Azure's diagnostic needs `require_parameters:false` because OpenRouter's route
metadata rejects the Responses output-cap parameter when that filter is true.
The diagnostic still pins Azure, disables fallback, enforces a price ceiling,
requests `summary:auto`, keeps ZDR, and verifies actual results. This is isolated
probe behavior; no production routing filter was weakened. Claude's automatic
choice is also a diagnostic-only control. No hosted route was newly whitelisted.

## Raw storage and readable representation

| Data requested | Raw representation and view | Limit or check |
| --- | --- | --- |
| Provider-exposed thinking and summaries | Structured response, full text, source paths, alias paths, hashes, and semantic classification | Summaries and unknown text never become approved full thinking |
| Reasoning tokens | Original usage object and separate reasoning-token count | A counter is not readable text |
| Opaque continuation and signatures | Original private ledger response bytes and unchanged live conversation state | Opaque bytes are redacted from training exports and readable views; never decrypted or reconstructed |
| Prompts, tools, results, intermediate events | Captured outbound body, provider response, worker observation snapshot and receipt links | Unsupported local client state is not inferred |
| ICP, input snapshots, final output and judgments | Existing linked export records and readable context | Missing records remain explicit gaps; probes have no fabricated labels |
| Cost and timing | Original provider usage/cost, accounting lifecycle, call timestamps, initial transport duration, attempt runtime | Transport timing excludes later reconciliation and is not TTFT |
| Reconstruction | Run/call/action IDs, ordered history, tool IDs, source and payload hashes, versions and retry records | Request and receipt size limits and privacy edits remain visible |

The gateway currently requests completed provider responses (`stream:false`).
There are no upstream streaming delta events on that path to preserve. Buffered
Codex SSE is a delivery format, not an additional provider reasoning trace.
Direct native web access and unsupported clients can still lack complete local
state; encrypted network tunnels cannot supply plaintext page content.

## Small fixes from this audit

- Preserve the official `reasoning_content` alias in trusted Chat assistant
  request history even outside the new full-thinking policy. Full-thinking
  eligibility is still a separate verified claim.
- Recover upstream response receipts from uncertain accounting records whose
  evidence is stored at the entry root. Unknown billing is not called settled.
- Index readable summary fields and deduplicate identical reasoning aliases.
  Keep raw source paths and all visible text; do not shorten the projection.
- Infer requested reasoning from the saved outbound controls when there is no
  full-policy annotation. Count nested Responses reasoning usage. Report mixed
  readable/opaque replies without hiding the readable text, and label empty
  responses as empty rather than unsupported.
- Add a private renderer for existing exports, including document context,
  missing fields, completeness gaps, and derived-status markers that do not
  pretend to be raw values.
- Add bounded initial provider transport duration to settled, uncertain, and
  failure evidence without changing provider outcomes.
- Permit the returned Responses reasoning `format` tag in stateless history.
  Preserve its opaque content, item IDs, tool IDs, and ordering.

## Test corrections, kept separate from product fixes

Adding `parallel_tool_calls:false` to the test caused all six exact routes to
fail OpenRouter parameter filtering. Removing that extra parameter restored all
six worker tests. Production caller controls are unchanged. Reflection returned
valid evidence IDs in a different order; the final JSON check now accepts an
unordered unique set, as the schema allows. Claude's thinking path does not
accept forced tool choice, so its summary-only diagnostic uses auto. These
failures are retained in the private record, not erased.

Earlier probe reports use `route_authority: committed` and
`production_whitelist_approved` to mean only that the local registry contained
the route. Those names did not establish deployment. The harness now reports
`working_registry`, `registry_listed`, and `route_probe_passed`, with
`production_activation: false`. Original reports remain unchanged for audit.

## Evidence totals and operator view

The audit made **85 provider attempts across 13 models**. The independent saved
view check reopened **36 exports containing 84 calls**, compared every returned
readable reasoning string and usage object with the wire response, and checked
indexed source paths and hashes. One early Mistral Small error probe failed its
export readback and is explicitly unverified; it is not whitelisted. Every one
of the six final whitelist probes has complete three-call readback and worker
acknowledgement evidence. Missing task state and scoring labels remain visible.

Raw response usage reports total **$0.0225049626**. The shared $8 budget also
retains **$5.277604** in conservative holds for unpriced failures and early probe
exceptions. Those holds are not a claim that the provider charged that amount.
Known charges plus these holds stay below the authorized cap. No paid testing
remains running. The [machine evidence manifest](evidence/full-thinking/capture-audit-20261023.json)
contains report/export/view hashes, route outcomes, costs, and current coverage.

Render a normal exported example with:

```sh
python3 scripts/render_arena_training_trace.py \
  --example /path/to/example.json --output /new/private/view-directory
```

The command verifies the export hash before writing. It copies the exact raw
export and creates readable JSON and Markdown with relative links. The new
directory is mode 0700 and files are 0600; existing output is never overwritten.

## Release limits

The policy remains off. Active and historical rounds retain their frozen rules.
Future activation still requires a tested compatible baseline archive and digest,
the exact migration, representative full ICP scoring, and production storage and
publication proof. Capture gaps do not cancel rounds or stop scoring. There is no
model substitution, no ZDR relaxation, and no merge or deployment in this task.
