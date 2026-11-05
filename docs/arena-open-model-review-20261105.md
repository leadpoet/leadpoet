# Arena open-weight model review — 2026-11-05

This review builds on the committed implementation in PR1041 and the deployed
internal capture work in PR1044. It tests exact provider routes, not model names
alone. A catalog reasoning flag or token count is not proof of readable thinking.
“Full” here means provider-exposed readable thinking; it does not mean access to
undisclosed provider internals.

## Scope and release gate

The goal is useful sales research and judgment data for CPT/SFT, at comparable
quality and no higher total verifier cost. Existing frozen rounds, miner
eligibility, scoring criteria, source authority, prices and completed judgments
must remain unchanged. New defaults apply only when a new round is created.

This document is being completed with the full-flow and deployment evidence.
The component results below are not a claim that the new policy is deployed.

## Selected mapping under validation

| Arena work | Previous model | Candidate and exact route | Reasoning / output cap |
| --- | --- | --- | --- |
| Company reverification and industry research controller | Sonar; Sol controller | DeepSeek V4.1 Flash 1017, `nebius/fp8` | low / 8192 |
| Intent signal gate and fetched evidence | GPT-6 Luna | GPT-OSS-120B, `near-ai/fp4` | low / 4096 |
| Intent paragraph review | Sonnet 5.5 | DeepSeek V4.1 Flash 1017, `nebius/fp8` | low / 8192; latency review pending |
| Role checks and generic company investigator | Gemini 3.5 Flash Lite | GPT-OSS-120B, `near-ai/fp4` | low / 4096 |
| Positive semantic company review | GPT-6 Luna | GPT-OSS-120B, `near-ai/fp4` | low / 4096 |
| Claim equivalence for evidence reuse | GPT-6 Luna | GPT-OSS-120B, `near-ai/fp4` | low / **500, 15 seconds** |
| Search fallback, including social/podcast verification | Sonar Pro | DeepSeek V4.1 Flash 1017, `nebius/fp8` | low / 8192 |
| Final industry classification | GPT-6.1 Sol | retained after expanded quality tests | existing policy |
| Admission and security review | Sonnet 5.5, Sol fallback | retained | existing policy |
| ICP generation | Sonnet 5.5 | retained after factual evaluation | existing policy |
| ICP exclusion research | DeepSeek, from PR1044 | retained after cheaper variants failed complete gates | high / 4096 |

GPT-OSS is an open-weight model despite its `openai/` model identifier. The two
selected open models use explicit provider selection, no automatic fallback,
ZDR/data-collection denial, reviewed route ceilings and frozen request settings.
Company search uses native Responses so exposed reasoning, search events and
citations survive capture. Serial local tools remain Chat calls with their
arguments, results and reasoning history.

## Component evaluation

The frozen evaluation contains 63 cases: real retained source/production
examples, adversarial edits to those examples, and declared synthetic role
controls. Gold constraints were set before responses. Twelve paid cases for
the four smaller roles were kept out of calibration. The candidate was locked
before those requests. Cases sharing a company/event family stay together.
These small suites are regression evidence, not an estimate of population
accuracy or a substitute for full scoring.

The frozen manifest hash is
`0371a6af73a4bccdf92c970a538feac11a4819c7494d947448e75a9dfa7578e6`.
The candidate lock hash is
`863c55df7177fa4fbfa176c70fd37477c111501829c6715b09f6273fb38733ce`.
Raw requests, responses, endpoint metadata, source hashes, labels and assessments
remain in private review artifacts, not in this repository.

| Role | Calibration expected fields: current / Near AI | Holdout: current / Near AI | Holdout cost: current / Near AI |
| --- | --- | --- | --- |
| Intent | 8/9 / 8/9 | 2/3 / 3/3 | $0.00325215 / $0.00081351 |
| Paragraph | 9/9 / 8/9 | 3/3 / 3/3 | $0.046932 / $0.00051168 |
| Role | 7/9 / 8/9 | 2/3 / 3/3 | $0.0033886 / $0.00033384 |
| Claim equivalence | 9/9 / 9/9 | 3/3 / 3/3 | $0.000218 / $0.00007719 |

All 48 Near AI calibration/holdout replies passed schema/native parsing and
returned readable thinking. The held-out 12 cost $0.00173622 versus $0.05379075
for the current models. No prompt change or retry followed the holdout.
One deterministic unsafe claim comparison was rejected before any paid call.

The calibration errors remain relevant. Near AI made one paragraph
factual-support error, but the complete paragraph gate still rejected that
example. It also missed a valid role. The intent subtype disagreement was a
safe rejection shared by the baseline. The held-out baseline had one native
false acceptance of a wrong-year event and one malformed role reply. A format
failure is counted as unavailable, never as a correct negative.

The same GPT-OSS model on DeepInfra BF16 was less reliable on these requests:
two malformed outputs and one unsupported date receipt. Changing to Near AI
also changes precision to FP4; this experiment cannot isolate provider effects
from precision or generation variation. Median holdout latency in seconds was
6.35/7.07 for intent, 1.80/3.71 for paragraphs, 1.04/1.62 for roles and 2.33/1.50
for comparisons (current/Near AI). Lower token cost does not imply lower latency
for every role. The later full scoring test exposed paragraph quote-binding
failures after the existing repair attempt. These component results therefore
do not qualify GPT-OSS as the final paragraph reviewer.

Manual and independent review also found incorrect positive labels in two
later EPAM request contexts. They contained the same paragraph, whose wording
referred to the requested company filters. The existing writing rule requires
`natural_paragraph=false` for that selection commentary, even when all business
facts and the partnership event are supported. Original requests, labels and
results are preserved as disputed calibration. They are not two independent
positive examples, and their original expected company score of 51 is invalid.
New client-facing wordings have separate labels set before any model response.
The old paragraph remains a negative writing control, distinct from a false
primary-event claim.

DeepSeek with the original paragraph prompt passed all 12 frozen cases and
both new positive wordings, without repairs, invalid replies or timeouts.
The two paired new positives cost $0.057048 on Sonnet versus $0.0134982 on
DeepSeek. Latency was 2.35–2.48 seconds versus 12.24–16.64 seconds, so route
latency remains a release concern. Across all 14 DeepSeek checks, median
latency was 5.72 seconds. GLM Flash also passed these 14 checks, but failed
the original writing-negative control. A preregistered generic writing-rule
reminder still missed both negative contexts and introduced a quote-binding
failure on a new positive. That variant did not advance to its reserved
three-case check. No paragraph prompt change is selected.

Seven separately retained production requests screened GPT-OSS-120B,
GPT-OSS-20B, Ling 3.1 Flash, Gemma 4 31B and Nemotron 3.5 Lightning. GPT-OSS-120B
completed all seven with readable thinking. The smaller alternatives had
truncation, invalid output or timeouts on this packet set. These seven were
positive development examples and are not counted as independent accuracy data.

## Harder exceptions and broader search

The full 39-model whitelist and its 95 approved model/provider/API routes were
reviewed against fresh endpoint metadata and the saved PR1041 capability calls.
The fresh catalog matched 86 routes exactly; 80 had status zero, nine were
missing and six had another status. Saved simple probes returned readable
text for 26 models. Neither result qualifies every model for every Arena task.

PR1041's industry data has 137 historical slots but only 111 distinct
company/ICP clusters. Its saved Sol baseline had 122/137 correct labels,
four false positives and eleven false negatives. Kimi's full run increased
false positives to ten. Failed budget admission is separate from model failure
in all comparisons.

Fresh 35-case industry tests initially gave Sol 29/35, DeepSeek 27/35 with four
invalid replies, and GLM Flash 24/35 with four HTTP429 failures. DeepSeek's three
empty explanations were invalid under the existing native parser. A generic
nonempty explanation schema/prompt experiment then completed all 35 with the
same confusion counts as Sol: TP14, TN15, FP2, FN4. It cost $0.360286992 versus
$2.30229175 for the saved Sol calls. Across all 137 slots, this variant then
scored 120/137 with TP53, TN67, FP6, FN9 and two truncated outputs. All 137
returned readable thinking, but the extra false positives and incomplete
outputs fail the replacement gate. The run cost $1.388905020.

GLM 5.2 on `coreweave/fp4`, low reasoning and 8192 output tokens, improved the
35-case screen to 32/35 with one false positive. On all 137 fixed slots it
scored 120/137: TP55, TN65, FP9, FN7 and one truncated reply. This does not meet
Sol's quality result. All 137 exposed readable reasoning and the expected
provider route. The selected responses cost $2.34161746; median measured
latency was 8.714 seconds across 135 calls, with two conservative wall-time
upper bounds kept separate. Sol remains selected.

A local disk-full error left 83 attempts without saved responses during the
first larger GLM run.
Every complete response was kept; only lost responses were repeated with the
same request bytes. The unknown original charges retain a $6.740806 budget
reservation, separate from the selected-response cost above. No label was
changed after testing, and no unsuccessful model result was replaced by a
better repeat. New private test dispatches check free disk space first.

Admission remains security sensitive. Prior realistic complete-source tests
found unsafe acceptances by DeepSeek and Kimi. The additional GLM Flash screen
passed native schema on only three of six examples and the combined expected
labels on two. Exact source excerpts, reachable research control and all
misconduct checks remain required. Short benign snippets alone do not justify
replacing the admission reviewer.

Prior ICP candidates passed some structural checks but produced false hard
filters or exhausted reasoning without a complete bank. A fresh GLM Flash bank
passed the ten-item parser but missed the international quota and included
incorrect employee bands and event dates. Sonnet remains selected. Cheaper
DeepSeek exclusion variants each passed only three of four factual cases.
GLM Flash passed four factual cases, but one exceeded the existing 45-second
deadline. The current DeepSeek high setting remains selected. These thirteen
native calls cost $0.159101056, with no unresolved charge reservation. Passing
the bank parser is not proof that example companies satisfy all filters.

## Capture and training value

PR1044 already captures admission, ICP generation and exclusion calls through
the existing private internal recorder/exporter. PR1041 adds prospective judge
capture through the existing broker, worker, storage and training exporter.
This work integrates that foundation and fixes native final-answer selection,
route/API validation, frozen request options, controller role routing and
evidence-cache model identity. No new service, database table, production
dependency or secondary verification model is introduced.

Judge training targets retain the frozen ICP, scored miner/baseline input,
source observations, exact available request/response data, exposed thinking,
tool order and arguments/results, provider identity, token/cost/latency data,
verdicts and links to scoring outcomes. Failed attempts and corrections remain
visible. Miner research, judge work and other internal tasks remain distinct.
Cache reuse keeps the original evidence and judgment origin; it is not counted
as a new independent model decision.

The narrow proprietary exception cannot claim full thinking. Its available
output, evidence and judgment links stay in raw exports, while independently
proved open-model turns can still become SFT candidates. A mixed trajectory is
explicitly partial. Export checks reject damaged route identity, request hashes
and worker evidence. Model-generated labels remain unreviewed until separately
admitted; an accepted score is not human-verified truth.

The most useful datasets are source-grounded company-fit decisions, explicit
negative examples, dated intent verification, rejected unsupported claims,
research corrections, and paired miner output/judge feedback. Preserve the full
context, source date and provenance when extracting these records. Split by
company/event and source lineage to reduce leakage and duplicate cache labels.

Coverage is not 100%. Proprietary hidden reasoning remains unavailable. Native
hosted search can expose URLs and events without the fetched page text or every
internal provider turn. Retain that distinction; do not reconstruct missing
evidence or promote incomplete transcripts to full-trajectory targets. Existing
privacy, permission and dataset quality fields remain applicable.

## Validation and release evidence

Pending: final full scoring comparison, exact image/worker verification,
required CI, protected merge, canonical paired deployment and live readback.
No production completion is asserted by this draft section.

## All 39 whitelist entries

This table reports saved simple capability probes and the fresh exact-route
catalog check. A successful probe is not a task-specific quality approval.
Route counts distinguish APIs, so the same provider can appear twice.

| Model | Healthy exact routes / registered | Saved probe: readable text / completion |
| --- | --- | --- |
| `deepseek/deepseek-v4-pro-0813` | 2 / 2 | yes / stop |
| `deepseek/deepseek-v4.1-flash-1017` | 6 / 8 | yes / stop |
| `fireworks/ember-1` | 2 / 2 | no / no completed answer |
| `google/gemma-4-31b-it` | 2 / 2 | yes / stop |
| `inclusionai/ling-3.1-flash` | 2 / 2 | yes / stop |
| `meta/muse-spark-open` | 0 / 2 | no / no completed answer |
| `minimax/minimax-m3` | 2 / 2 | no / stop |
| `minimax/minimax-m3.1-flash` | 0 / 2 | no / stop |
| `mistralai/mistral-large-4-0` | 2 / 2 | yes / stop |
| `mistralai/mistral-medium-3-5` | 2 / 2 | yes / stop |
| `moonshotai/kimi-k2-thinking` | 1 / 1 | no / no completed answer |
| `moonshotai/kimi-k2.5` | 2 / 2 | no / no completed answer |
| `moonshotai/kimi-k2.6` | 2 / 2 | yes / stop |
| `moonshotai/kimi-k2.7-code` | 4 / 4 | yes / stop |
| `moonshotai/kimi-k3` | 2 / 2 | yes / stop |
| `moonshotai/kimi-k3.1-pro` | 4 / 4 | yes / stop |
| `nvidia/nemotron-3.5-lightning` | 2 / 2 | yes / stop |
| `nvidia/nemotron-3.5-super-vl` | 2 / 2 | yes / stop |
| `openai/gpt-oss-120b` | 5 / 5 | yes / stop |
| `openai/gpt-oss-20b` | 3 / 3 | yes / stop |
| `qwen/qwen3.8-2.4t-a95b` | 2 / 2 | yes / stop |
| `qwen/qwen3.8-27b` | 2 / 2 | yes / stop |
| `reflection/beam-1.0-501b-a23b` | 2 / 5 | no / no completed answer |
| `stepfun/step-5-preview` | 0 / 2 | no / no completed answer |
| `tencent/hy4-preview` | 2 / 2 | yes / length |
| `upstage/solar-mini4` | 2 / 2 | yes / stop |
| `upstage/solar-pro4` | 2 / 2 | yes / stop |
| `xiaomi/mimo-v2.6-flash` | 0 / 2 | yes / stop |
| `z-ai/glm-4.5v` | 2 / 2 | no / no completed answer |
| `z-ai/glm-4.6` | 2 / 2 | yes / stop |
| `z-ai/glm-4.6v` | 2 / 2 | no / no completed answer |
| `z-ai/glm-4.7` | 2 / 2 | yes / stop |
| `z-ai/glm-5` | 2 / 2 | yes / stop |
| `z-ai/glm-5.1` | 2 / 2 | no / stop |
| `z-ai/glm-5.2` | 1 / 1 | yes / stop |
| `z-ai/glm-5.3` | 2 / 4 | yes / stop |
| `z-ai/glm-5.3-flash` | 2 / 2 | no / no completed answer |
| `z-ai/glm-5.3-flashx` | 2 / 2 | no / no completed answer |
| `z-ai/glm-5v-turbo` | 2 / 2 | no / no completed answer |
