# Arena agent reasoning capture

Policy `agent_reasoning_v1` makes every model call from an agent reason and
return readable reasoning. Training exports then keep that text next to the
call's request, response and outcome.

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
change cannot alter a running round.

For each `openrouter.chat` or `openrouter.responses` call in an execute run:

| Agent request | Sent |
|---|---|
| No reasoning setting, model reasons by default | The model's default effort, made explicit (no behavior change) |
| No reasoning setting, model does not reason by default | Effort `low` |
| Effort `none` or `minimal`, `enabled: false`, or a zero budget | Raised to `low` (or the lowest supported effort above it) |
| Effort `low` or higher, or a positive thinking budget | Kept |
| `exclude: true` | Changed to `false`, so reasoning text is returned |
| Legacy `reasoning_effort` / `include_reasoning` | Folded into the one `reasoning` object |
| OpenAI chat model, or any Responses call | `summary: "auto"` added when absent |
| Optional Anthropic thinking (for example Claude Sonnet 4.5) | A 1,024-token thinking budget |

When the policy raises reasoning, it adds 1,024 output tokens so new
reasoning does not consume the visible answer. The chat cap stays 4,096 and
the Responses cap 32,768.

The policy never rewrites an agent's sampling settings. Optional Anthropic
thinking refuses a temperature other than 1, `top_k`, `top_p` below 0.95 and
forced tool choice, so such calls are sent unchanged and recorded as
`anthropic_sampling_conflict`. Models whose catalog lists no reasoning
support, such as Perplexity Sonar, are also sent unchanged.

If the operation schema would refuse the policy form, for example because it
pushes a request over the size cap, the agent's own valid request is sent. The
reservation records `unapplied: operation_schema`. The policy can add
reasoning; it cannot make a valid agent request invalid.

The rewrite happens before request hashing. The request hash, call identity,
captured request and sent body all bind the parameters actually sent, and a
retried call reproduces them exactly. Each reservation's `entry_doc.reasoning_capture`
records the treatment, whether readable reasoning was requested, the changes
and the effort.

## Cost and competition effect

Reasoning tokens bill as output on the agent's own key and count toward its
per-ICP cost limits. Most calls change only by an explicit default or a
summary request. Calls that were not reasoning now reason at `low` effort,
which adds latency and some cost. The public round view lists
`reasoning_capture_policy`. Announce activation to miners before the first
affected submission window.

## Readable is not hidden

Summaries and thinking text are what providers deliberately return. They are
not a model's full internal chain of thought. Encrypted reasoning, signatures
and redacted thinking stay opaque and are removed from training exports.
Adaptive models can still skip reasoning on easy turns, so not every call
will carry text.

## Activation

1. Merge and deploy through the canonical gateway restart. The broker runs
   only in the gateway, so validators need no update.
2. Run the live probe with an operator key (a few cents):

   ```bash
   OPENROUTER_API_KEY=... python3 scripts/probe_reasoning_capture.py
   ```

   Exit 0 means no policy-caused failure, and every successful requested call
   returned readable reasoning. A failure that the agent's own form shares is
   reported as pre-existing.
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

   It counts agent calls whose policy requested readable reasoning and reports
   the share that returned it, per model. It exits 2 below the target (default
   80%).

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
- OpenRouter lists `google/gemini-2.5-flash` and `google/gemini-2.5-flash-lite`
  with an expiration date of 2026-10-20.
