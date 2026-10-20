# Arena agent reasoning capture

Policy `agent_reasoning_v1` makes model calls from agents reason and return
readable reasoning wherever that is safe to add. Training exports then keep
that text next to the call's request, response and outcome.

It applies only to rounds that freeze it. Judge (score) runs are never
changed.

## Why

On 2026-10-19 production held 512,426 settled OpenRouter calls, but agents
almost never returned readable reasoning:

- OpenAI models return reasoning as encrypted content unless a summary is
  requested, and no agent requested one. The Codex runtime sets only an
  effort.
- 427 of 447 miner runs from October 17 to 19 never turned reasoning on.

Agents cannot provide this themselves without changing their code. The
gateway broker already sees every agent model call, so the gateway sets it.

## What the gateway changes

Each round freezes the reasoning profile of every OpenRouter model at
creation: mandatory, on by default, default effort and supported efforts. The
profiles come from the gateway's startup catalog snapshot, so a later catalog
change cannot alter a running round. The v1 rules below are frozen once a
round has used them; a change needs a new policy version.

For each `openrouter.chat` or `openrouter.responses` call in an execute run:

| Agent request | Sent |
|---|---|
| No reasoning setting, model reasons by default | The model's default effort, made explicit (no behavior change) |
| No reasoning setting, model does not reason by default | Effort `low` |
| Reasoning turned on without an effort (`enabled: true`, `{}`, `include_reasoning: true`) | Kept at the provider's default |
| Effort `minimal` | Raised to `low` |
| Effort `none`, `enabled: false` or a zero budget | Raised to `low`, except for Anthropic and Google models, where the agent's choice stands |
| Effort `low` or higher, or a positive thinking budget | Kept |
| `exclude: true` | Changed to `false`, so reasoning text is returned |
| Legacy `reasoning_effort` / `include_reasoning` | Folded into the one `reasoning` object |
| OpenAI chat model, or any Responses call that reasons | `summary: "auto"` added when absent |

Wherever the table says `low`, optional Anthropic thinking (for example
Claude Sonnet 4.5) and Gemini models that list no efforts (Gemini 2.5) get a
1,024-token thinking budget on chat instead.

Whenever the policy raises reasoning, it adds up to 1,024 output tokens of
room for it, never past the caps of 4,096 (chat) and 32,768 (Responses). The
reservation records `headroom`, or `headroom_capped` when the cap left less
room and the new reasoning shares the agent's own limit. A chat call with no
`max_tokens` defaults to 4,096, the cap. How close to the cap the policy goes
depends on how the model's reasoning can be bounded:

- **Thinking budget** (optional Claude, Gemini 2.5): the provider applies it
  to the thinking itself, so near the cap the visible answer keeps about
  3,072 chat tokens or more.
- **Effort** (the catalog lists efforts, for example GPT-5 and Gemini 3): `low`
  has no hard token bound but is usually short. It is also raised near the
  cap.
- **Neither** (the catalog lists no efforts, for example DeepSeek V3.2, GLM
  or Grok): OpenRouter can only switch reasoning on at the model's own
  length. The policy does so only when the full 1,024 tokens fit; otherwise
  the call is unchanged (`skipped: headroom_unavailable`).

A model whose catalog efforts skip `low` is not raised to a costlier one
(`skipped: no_low_effort`).

This trade-off rests on production evidence. Over the October 17–19 rounds,
3,471 agent calls to Gemini 2.5 Flash with a 4,000-token limit (45% of its
calls) and 225 Claude Sonnet 4.5 calls at 4,096 never produced a visible
answer above 2,972 tokens, and none was cut off. Gemini 3.5 Flash Lite, which
agents already use, returned no reasoning on any of 332 agent calls at its
default `minimal` effort; 147 of those used a 4,000-token limit, and their
answers peaked at 1,965 tokens. Skipping calls near the cap would have left
almost half of the main models' calls without reasoning. Truncation is the
risk to watch, mainly for effort models: the probe's near-cap case asks for a
long answer and fails if it is cut off, and the coverage report counts
cut-off answers on calls where the policy added reasoning.

The policy never rewrites an agent's sampling settings. It leaves these calls
exactly as the agent wrote them:

- Optional Anthropic thinking on chat with a temperature other than 1,
  `top_k`, `top_p` below 0.95, forced tool choice, an assistant prefill or a
  tool-loop continuation (the chat schema cannot carry earlier thinking
  blocks back). Recorded as `anthropic_thinking_conflict`.
- Optional Anthropic thinking through the Responses API, which is never
  switched on there.
- An explicit opt-out on an Anthropic or Google model, whose thinking state
  must round-trip during tool use. Recorded as `agent_disabled`.
- Models whose catalog lists no reasoning support, such as Perplexity Sonar.

If the operation schema would refuse the policy form, for example because it
pushes a request over the size cap, the agent's own valid request is sent. The
reservation records `unapplied: operation_schema`. The policy can add
reasoning; it cannot make a valid agent request invalid.

The rewrite happens before request hashing. The request hash, call identity,
captured request and sent body all bind the parameters actually sent, and a
retried call reproduces them exactly. Each reservation's `entry_doc.reasoning_capture`
records the treatment, whether readable reasoning was requested, the changes,
the effort and any skip reason.

## Cost and competition effect

Reasoning tokens bill as output on the agent's own key and count toward its
per-ICP cost limits. Most calls change only by an explicit default or a
summary request. Calls that were not reasoning now reason at `low` effort,
which adds latency and some cost. Each round's public view
(`/arena/v1/rounds/<round_id>`) lists `reasoning_capture_policy`; the
current-round summary does not yet. Announce activation to miners before the
first affected submission window.

## What readable reasoning is

Training exports keep reasoning text that a provider deliberately returns:
chat `message.reasoning`, readable `reasoning_details` and Responses
reasoning summaries or content. For most hosted models this is a summary. For
some open-weight models (for example DeepSeek, Qwen or gpt-oss) it can be the
model's full reasoning text. Encrypted reasoning, signatures and redacted
thinking stay opaque and are removed. Adaptive models can skip reasoning on
easy turns, so not every call carries text.

This applies to every export once deployed, including historical receipts,
because exports re-read responses from the ledger.

## Activation

1. Merge and deploy through the canonical gateway restart. The broker runs
   only in the gateway, so validators need no update.
2. Preview the rewrites at no cost, then run the live probe with an operator
   key (a few cents):

   ```bash
   python3 scripts/probe_reasoning_capture.py --dry-run
   ```

   ```bash
   OPENROUTER_API_KEY=... python3 scripts/probe_reasoning_capture.py
   ```

   Exit 0 means no policy-caused failure, every successful requested call
   returned readable reasoning, and no answer was cut off on a call where the
   policy added reasoning, including a long answer near the 4,096 chat cap. A
   failure that the agent's own form shares is reported as pre-existing. The
   probe uses the production host route for the Luna Responses models. The
   default model list covers current agent models; `--model` replaces it.
3. Set the activation time. Rounds whose intake opens at or after it freeze
   the policy:

   ```bash
   python3 scripts/configure_lab_arena_production.py --reasoning-capture-from 2026-10-22T00:00:00Z --allowed-account <account-id> --apply
   ```

   This writes only `LAB_ARENA_REASONING_CAPTURE_FROM`, like the other
   activation times. Then restart the gateway. An empty value clears it.
4. After the first such round, measure coverage from the stored receipts:

   ```bash
   python3 scripts/arena_reasoning_coverage.py --round-id arena-2026-10-23 --environment-file ...
   ```

   It counts agent calls whose policy requested readable reasoning and
   reports the share that returned it per model. It exits 2 below the target
   (default 80%). It also reports the capture rate across all calls to models
   that can reason, why the rest were not requested, and how many answers were
   cut off on calls where the policy added reasoning.

Rounds created without the policy, including every existing round, are
unchanged. If the catalog snapshot is unavailable when a round is created, the
round is created without the policy and the gateway logs an error. A round is
never blocked for this.

Rolling the gateway back to a release without this policy makes rounds that
froze it invalid for that release. Roll back only before such a round opens,
or roll forward.

## Known limits

- Gemini 3 tool loops through OpenRouter expect returned reasoning details on
  the next turn. The chat schema cannot carry them today. This affects those
  loops with or without this policy.
- OpenAI may return an empty summary for short reasoning.
- A worker observation stored before this change that kept a `signature` or
  `data` field inside a typed reasoning item outside `reasoning_details` no
  longer re-cleans to itself; its export reports
  `worker_observations_unavailable`. OpenRouter responses do not carry that
  shape, so this is not expected in practice.
- OpenRouter lists `google/gemini-2.5-flash` and `google/gemini-2.5-flash-lite`
  with an expiration date of 2026-10-20.
