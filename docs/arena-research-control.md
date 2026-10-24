# Research control: tested proposal, not an active competition rule

## Decision

Do not ban Python or fixed workflows. They can produce useful company research
and training examples. For a dataset that teaches a model to control research,
also collect model decisions that choose the next action using observations and
choose when to finish. This is a behavior requirement, not a programming-language,
framework, JSON-format, or call-count requirement.

This PR adds an **offline, experimental source assessment** and executable
counterexamples. It does not change submission admission, the source misconduct
reviewer, the baseline, scoring, capture, model routes, rewards, or round state.
It is separate from full-thinking capture PR #732. Neither raw thinking nor a
tool-call transcript, alone, proves that the model controls research.

The user requested implementation only if investigation showed the proposed fix
was sound. Investigation supports the training-data distinction. It does **not**
establish that an agent-only admission gate improves scores or student training.
The actual full-source reviewer trial did not produce an acceptable assessment.
Accordingly, automatic rejection is deliberately not implemented or activated.

## Original evidence

The public October 23 champion is
`sub-b342d25fb1d9d63f60265c1b697c5420`. All 31 source files matched promoted
`leadpoet-sales-agent` commit `15bd359703f4ce0b6420bc4531e666fc12098622`.
The public endpoint reported `truncated=false`.

- `strategy.json:30` selects `research_mode: "scout"`.
- `harness.py:1868` selects the scout path. At line 2203, the source says this
  bundle omits `agent.py` and `context.py`; the unused generic agent entry raises.
- `scout.py:3439` controls the research stages in Python. Its ordinary JSON
  model calls do not form the generic adaptive agent loop.
- Models still generate executed search queries (`scout.py:467`), extract
  candidates from returned evidence (`:766`), and change candidate priority
  (`:3195`). These are useful query, extraction, and ranking demonstrations.
- The actual strategy disables the optional second search pass. Its existence
  in source does not establish that the champion used it.

Sources: [published round](https://gateway.subnet71.com/arena/v1/rounds/arena-2026-10-23),
[complete public source](https://gateway.subnet71.com/arena/v1/submissions/sub-b342d25fb1d9d63f60265c1b697c5420/code),
[pinned scout source](https://github.com/leadpoet/leadpoet-sales-agent/blob/15bd359703f4ce0b6420bc4531e666fc12098622/scout.py),
[public execution results](https://gateway.subnet71.com/arena/v1/rounds/arena-2026-10-23/results/sub-b342d25fb1d9d63f60265c1b697c5420).

Eight offline causal checks against that exact source passed. Changing stubbed
model queries changed real search-method arguments. Changing model ranking
changed candidate order. Changing extraction changed candidate inclusion. The
tested stage sequence stayed fixed when returned evidence count changed from
zero to five. That last check isolates the controller through triage; it does
not exercise all downstream resolution. No provider was called in those checks.

All ten public champion executions reported incomplete worker observations.
This is a capture limit, not proof of no model reasoning or poor research.
This PR does not repair those gaps or claim complete historical trajectories.

The distinction matches [Anthropic's workflow/agent definitions](https://www.anthropic.com/research/building-effective-agents).
That guide also recommends simple workflows when they fit a task. The
[ReAct paper](https://arxiv.org/abs/2210.03629) provides evidence for interleaved
actions and observations on other tasks; it does not prove gains for this Arena.

## What the assessment does

`lab_arena.research_control` implements `model_directed_research_v1`.
It prepares every validated source file for a separate model assessment and
strictly validates the saved response. Classifications are:

| Classification | Meaning |
| --- | --- |
| `model_directed` | Source describes a reachable controller that lets model output choose a meaningful next research action from observations and choose finish/continue. |
| `workflow` | Source fixes substantive research control; model calls can still perform useful components. |
| `unverified` | Missing or dynamic controller code prevents a sound classification. |

Python dispatch, JSON actions, native tools, and hybrid scaffolds are allowed.
Schema validation, transport, budget stops, safe retries, and deterministic
validation remain useful host functions. No minimum turns or repeated replanning
is required. A model can finish immediately when enough evidence is available.
No private provider thinking is extracted or reconstructed.

The parser checks source/policy/request hashes, model identity, finish status,
ordered complete file coverage, exact source excerpts, a closed evidence schema,
and context-integrity evidence when required. It rejects malformed, truncated,
unbound, or unsupported responses. It does not prove that an authentic excerpt
supports the model's interpretation. Tests explicitly demonstrate that limit.

There is no service import of this module. Its output is not a misconduct result,
submission pass, score, promotion decision, or production eligibility certificate.
There is no new dependency, table, migration, background job, or live-round flag.

## Reproducible checks

Ten small source bundles exercise adaptive JSON, native tools, a bounded hybrid,
fixed extraction, one-shot query planning, ignored-model and forced-tool decoys,
dead adaptive code, and unresolved dynamic controllers.

These are **conditional controller fixtures**. Tests inject model and tool
callables with `configure(...)`; Arena does not provide that interface to miners.
The fixtures are not deployment-ready submissions. Their labeled oracle concerns
control behavior under injection, not source-review accuracy on arbitrary miners.
The runtime tests change model actions and observations independently, inspect
tool-owned action records, and verify tool IDs, order, final JSON, and bounds.
Self-reported trace logs are not used as proof of actions.

Actual OpenRouter calls used the existing reviewer model,
`anthropic/claude-sonnet-5.5`, pinned to `google-vertex/global`, with ZDR and
fallback disabled. Ten fixture assessments returned nine valid, oracle-matching
classifications. The ignored-model decoy reply had an extra evidence field and
was correctly rejected. This is a small development check, not a blinded accuracy
estimate or production readiness claim.

A separate request supplied all 31 public champion files. It used 534,992 input
tokens. Its response reached the 2,048 output-token limit and lacked the required
router context-integrity metadata. It was rejected. No complete classification
was inferred from its partial text. Increasing the output cap may address that
truncation; it does not repair missing integrity proof or establish accuracy.

The 11 live calls reported $1.448036 in new spend. They used the original shared
$8 test ledger. Prior unresolved reservations were retained. No new allowance,
production change, or unrestricted retry was used. The committed evidence file
records exact source and response hashes, results, token usage, and limits.
The ten synthetic fixture replies are retained exactly for offline replay. The
champion reply stays private; its public projection contains only model/provider,
usage, finish status, and metadata presence. Generated reasoning and opaque
signatures are omitted. The original and projection have separate hashes; the
projection cannot establish byte-exact readback of the original.

Run local checks:

```sh
python3 -m pytest tests/lab_arena/research_control_test.py tests/lab_arena/research_control_behavior_test.py tests/lab_arena/research_control_cli_test.py tests/lab_arena/research_control_live_replay_test.py -q
```

Prepare a new private request or validate a saved reply without a network call:

```sh
python3 scripts/assess_arena_research_control.py --source submission.tar.gz --output request.json
python3 scripts/assess_arena_research_control.py --source submission.tar.gz --response provider-response.json --output assessment.json
```

Outputs are new mode-0600 files; existing files are never overwritten. Submitted
code is not executed by this CLI. A successful exit means request preparation or
response validation succeeded, not that a submission is eligible.

## What must be proved before an admission change

1. Measure classification on a separate, labeled set of real, runnable source
   archives, including hybrid agents, dynamic code, decoys, and short valid runs.
   Define acceptable false-rejection and unresolved rates before that evaluation.
2. Prove complete-source transport, adequate output capacity, and usable evidence
   for large archives. Do not silently remove files or convert errors to rejects.
3. Compare matched research tasks and actual student training with held-out
   companies. Measure factual quality and useful training yield, not trace length.
4. Apply a frozen prospective rule equally to miners and the baseline. The
   current service, key selection, and SQL review functions exempt the baseline;
   removing only one bypass would not create a fair rule. Existing reviews must
   not be reused as proof under a new policy. Plan baseline eligibility before
   activation so a new rule cannot prevent the next round from opening.

Until those checks pass, preserve component datasets and full-action datasets as
separate useful sources. A better dataset objective does not justify a language
ban or an unvalidated competition gate.
