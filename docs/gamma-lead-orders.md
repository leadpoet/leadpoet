# Gamma lead orders

Anyone holding Leadpoet Gamma on netuid 71 can order sales leads. Miners'
OpenRouter and Deepline keys pay the provider calls and are repaid from
emissions at spend × (1 + premium). The lead-finding code is private and runs
in Leadpoet's own runner; everything that decides money is in this repository.

Status: built behind `LEADPOET_LEAD_ORDERS_MODE=off` (the default). Nothing
runs until a reviewed boundary is added to `AGENTS.md`. While off, every route
answers 404 and no store, chain, KMS or provider client is built.

Code: `lab_arena/lead_orders/` (policy, money, gamma, keys, broker, receipts,
rewards, weights, service), `gateway/api/lead_orders.py` (routes) and
`scripts/707-lead-orders.sql` (append-only tables and RPCs, RLS on).

## Modes

| Mode | Who can order | Keys used | Rewards |
| --- | --- | --- | --- |
| `off` | nobody | none | none |
| `staff` | hotkeys in `LEADPOET_LEAD_ORDERS_STAFF_HOTKEYS`, no Gamma | Leadpoet's keys only | none |
| `shadow` | staff hotkeys, no Gamma | Leadpoet's keys only | what-if only: published in the receipt, never owed or weighted |
| `live` | any hotkey with Gamma balance | ranked miner keys, then Leadpoet's | weighted, under the lane cap |

Keys follow each order's own mode, not the gateway's current mode: only a
`live` order spends miner keys, even after the gateway moves to `live`.
Miners can register keys in `shadow` and `live`. A shadow order's receipt
carries `what_if_rewards_microusd`, the reward its settled spend would have
earned on miner keys. Shadow spend never counts toward the key ranking, so
shadow cannot move the live order of keys.

## Policy (one signed, versioned record)

| Field | Start |
| --- | --- |
| `price_per_lead_usd` | 0.20 |
| `miner_premium` | 0.25 |
| `min_leads_per_order` | 10 |
| `min_charge_ratio` | 0.40 |
| `spend_ceiling_per_lead_usd` | 0.50 |
| `lane_emission_cap` | 0.30 of miner emission per epoch |
| `ranking_window_epochs` | 140 |
| `daily_leads_per_hotkey` | 500 |
| `order_deadline_minutes` | 60 |
| `max_provider_discounts` | Deepline 0.10, OpenRouter 0 |
| `deepline_reserve_per_call_usd` | 0.50 (worst-case hold per Deepline call) |
| `openrouter_max_output_tokens` | 16384 (`max_tokens` is clamped to this) |
| `miner_terms_version` | `miner-terms-2026-11-20` (the miner data terms a key's owner must have signed) |
| `miner_terms_sha256` | sha256 of the exact terms text, so a signature names the words, not just a label |

The record also carries `version` and `schema_version`, and is signed with the
Arena signing key. Each version is stored once; an order records the version
it was placed under.

Invariants, checked when a policy is loaded; a policy that breaks either is
refused:

- `min_charge_ratio > miner_premium + max_provider_discount`, so a buyer who
  is also a miner can never earn more than they are charged (no farming).
- `min_charge_ratio × spend_ceiling_per_lead_usd ≤ price_per_lead_usd`, so the
  minimum charge never exceeds the hold.

Known limit: the first invariant holds only for the assumed discounts. A miner
whose real provider discount is above `max_provider_discounts` can profit from
ordering against their own keys. `lane_emission_cap` bounds what that can earn
per epoch.

Each order records the policy version it was placed under. Its charge, rewards
and receipt always use that stored version, never the current one.

## Money

- Balance (USD) = Gamma the hotkey spent to the Leadpoet service on netuid 71
  (chain counter `GammaSpent(netuid, service, hotkey)` at a finalized block,
  10^9 units per USD) − sum of charges on settled orders − open holds.
- Order hold = N × price. Refused if balance < hold, N < minimum, or the daily
  cap would be passed.
- Spend ceiling = N × `spend_ceiling_per_lead_usd`. The broker answers 402 once
  an order's reserved spend would pass it.
- Charge = min(hold, max(delivered × price, ceil(min_charge_ratio × spend))),
  in whole micro-USD. Spend is settled spend only.
- Miner reward (USD) = floor(that miner's rewardable spend on the order × (1 +
  premium)), where each call's rewardable spend is min(cost, reserved). A cost
  above the call's reservation is still charged to the buyer, but the excess
  earns no reward; the miner's statement shows it as `excess_microusd`.
  Leadpoet's own keys earn nothing. If an order's total rewardable spend
  passes its ceiling, every reward on it is scaled by ceiling / total.
- A call whose cost the provider did not report (no cost field on a 2xx, a
  timeout, a transport error) is settled `uncertain`: its reservation keeps
  counting against the ceiling, and it is never charged to the buyer or
  rewarded. A 4xx with no cost settles at 0.
- An order the runner does not finish by its deadline (plus 300 s grace,
  longer than the broker's 240 s Deepline timeout) is finished as `failed`
  with reason `deadline_passed` at the next claim, and charged the minimum
  charge on its settled spend.
- Finish is refused with 409 `calls_open` while a call opened in the last
  300 s (the broker's longest provider wait, 240 s, plus a margin) is still
  open: the runner waits and finishes again. An older open call was orphaned
  (the broker stopped between reserve and settle, as on a deploy) and the
  finish settles it `uncertain`; with no provider request id it cannot be read
  back. The overdue sweep (deadline plus grace) settles every open call
  `uncertain`. Once finished,
  the order takes no reservations. The receipt's call rows are frozen when the
  receipt is signed, and the buyer's charge never changes after finish.
- Repayment of uncertain calls (live orders, miner keys only). The broker
  stores the provider's request id on every call it can, a timeout included:
  OpenRouter's generation id, Deepline's request id. The epoch pass (never a
  claim) runs a bounded read-back step: 10 calls and 15 s per pass, 5 s per
  lookup, calls under 7 days old, through OpenRouter's free generation lookup
  on the same miner key. Every attempt is recorded append-only. A terminal
  answer (a refused key, a malformed record) stops that call; a retry waits
  10 min × 2^(tries − 1), at most 6 tries; the least recently tried calls go
  first, so a call that never answers cannot block the others. A provider answer that arrives after finish is also kept, in a
  separate late-settlement table. When either gives a known cost, the miner is
  owed a late reward of floor(min(cost, reserved) × (1 + premium)), paid in a
  following epoch's payout like any other reward and listed in the miner's
  statement. It is never charged to the buyer and never changes the receipt.
- Late rewards keep the no-farming rule: all rewards on an order stay at or
  below charge × (1 + premium) / `min_charge_ratio`. A delivered order has
  room for them; a failed order, charged only the minimum on its settled
  spend, has almost none, so a late cost on a failed order is mostly not
  repaid (known limit).
- A finished order with no receipt (for example, signing failed) is settled
  again by an idempotent reconcile pass that runs on every claim and every
  epoch pass.
- Per epoch, rewards owed are paid oldest first up to the lane cap; the rest
  carries forward in USD and converts to alpha at the paying epoch's price.
  Late rewards join the same queue by the epoch they were recorded in. The
  emission price source is not decided yet: until it is, `live` payouts are
  refused. A `shadow` epoch publishes a payout document with no payments;
  shadow orders owe nothing (their what-if rewards are in their receipts).
- A `live` payout counts only `live` orders; recording a payment against a
  non-live order is refused. Staff and shadow orders earn nothing.
- The Gamma service id (`LEADPOET_LEAD_ORDERS_GAMMA_SERVICE`) must be the
  SS58 hotkey registered as a service on the subnet. Runtime 493 uses an
  AccountId32 for this key; numeric service ids are refused before a chain read.
  The storage item name is configurable
  (`LEADPOET_LEAD_ORDERS_GAMMA_STORAGE`, default `GammaSpent`). A chain without
  it reads as no balance, and `live` orders are refused (`gamma_unavailable`).
  A counter that goes backwards is refused with 422 `gamma_counter_backwards`.

## Key order

Per provider, miner keys are ranked by USD spend settled on `live` orders over
the last `ranking_window_epochs`, highest first; ties and new keys by
registration time. Leadpoet's key is always last. On 401, 402, 403,
429-with-no-retry or a spent limit the broker moves to the next key for the
rest of the order.

Each miner key has a daily cap (`daily_cap_usd`, required at registration,
USD 1 to 10,000, at most 6 decimals, else 422 `daily_cap_invalid`). The
reserve step locks the key and adds the key's spend over the last 24 hours
using the actual cost as soon as a settlement, late response or readback supplies it.
An open or unresolved call counts at its reservation. A new
reservation that would pass the cap is refused `key_capped` and the broker
tries the next key. So no order, however many run at once, spends more than
the cap on a key in a day, apart from a provider reporting more than was
reserved.

Miner data terms are required. A miner running an order on their own key can
see the order's lead data in their provider account, so a key is registered
only after its miner has signed the current terms text, named by
`miner_terms_sha256`, and the broker ranks only keys whose miner accepted
exactly that text. When the text changes, keys signed under the old one stay
registered but are skipped until their miner signs again, even if the version
label was not changed. The terms text is in
`docs/gamma-lead-orders-miner-terms.md`: use the data only to run orders,
keep no copies, honour the suppression feed, report problems.

- `GET /lead-orders/miner-terms` (public): `{terms_version, terms_sha256,
  text}`. Served only when the deployed text hashes to the policy's
  `miner_terms_sha256`, else 503 `terms_unavailable`.
- `POST /lead-orders/miner-terms` `{terms_version, terms_sha256}`
  (hotkey-signed): both must equal the policy's `miner_terms_version` and
  `miner_terms_sha256` (else 422 `terms_version_not_current`). The signed
  bytes must be that object as canonical JSON (sorted keys, no spaces), else
  422 `terms_body_not_canonical`, so the stored body sha256 alone proves what
  was signed. The acceptance is stored append-only with the
  hotkey, version, the request's signature, its `X-Timestamp` and the body's
  sha256, so it can be re-verified later. Answers `{status: accepted | exists,
  terms_version, accepted_at}`.
- `POST /lead-orders/keys` without a current acceptance is 403
  `miner_terms_not_accepted`.

## Runner protocol (private runner ⇄ gateway)

Bearer `LEADPOET_LEAD_ORDERS_RUNNER_SECRET` on every call. JSON bodies.

`POST /internal/lead-orders/claim` → `204` when nothing is queued, else the
order below. The token is issued and the deadline set at claim (now +
`order_deadline_minutes`); the token is shown once and only its hash is
stored.

```json
{
  "order_id": "uuid",
  "hotkey": "5F…",
  "mode": "staff | shadow | live",
  "leads_requested": 10,
  "deadline": "2026-11-19T21:00:00Z",
  "spend_ceiling_usd": "5.00",
  "seller": {"product": "…", "summary": "…"},
  "search": {
    "signal": "hiring",
    "terms": ["…"],
    "buyer_titles": ["…"],
    "market": {"countries": ["US"], "employee_ranges": ["11-50"], "industries": ["…"]},
    "excluded_companies": ["example.com"]
  },
  "broker": {
    "openrouter_base_url": "https://…/lead-orders/broker/openrouter/api/v1",
    "deepline_base_url": "https://…/lead-orders/broker/deepline",
    "token": "one-order token, used as the API key for both"
  }
}
```

`POST /internal/lead-orders/{order_id}/heartbeat` `{}` →
`{"continue": true, "spent_usd": "1.23", "deadline": "…Z"}`. `continue: false`
once the order is finished, the deadline has passed or the ceiling is reached;
the runner stops and finishes with what it has. `spent_usd` counts settled
spend plus open and uncertain reservations, the same total the ceiling checks.
USD amounts are decimal strings with at least two places.

`POST /internal/lead-orders/{order_id}/finish`:

```json
{
  "status": "delivered | failed",
  "leads_delivered": 6,
  "delivery_url": "https://leadpoet.com/api/subnet-leads/<opaque token>",
  "failure_reason": null
}
```

Lead data never passes through the gateway. The runner stores the file in
Leadpoet's private storage and returns an opaque, expiring download URL; the
gateway hands that URL to the buyer. `delivery_url` must be `https://`.
`failed` charges the minimum charge on spend so far. A failed order is never
retried automatically. Finishing twice changes nothing and returns the first
result. `leads_delivered` above `leads_requested` is refused with 422
`leads_delivered_invalid`. The response is `{status, charge_microusd, charged_usd,
receipt_hash}`.

## Broker

`/lead-orders/broker/openrouter/*` and `/lead-orders/broker/deepline/*` accept
the one-order token as the provider API key, pick the key, inject the real
credential, reserve the worst-case cost before sending, settle at the
provider-reported cost, and write one ledger row per call: order, provider,
miner hotkey (null for Leadpoet's key), key fingerprint, cost in micro-USD.

The token goes in `Authorization: Bearer <token>`, so provider SDKs work with
the base URLs unchanged. It is bound to one order and stops working once that
order finishes (409 `order_finished`) or its deadline passes (410
`order_expired`). Other answers: 402 `spend_ceiling` (the call was not sent),
502 `provider_unreachable`, 503 `provider_keys_exhausted`. Only the
provider's `content-type` header is passed back.

Anything outside the allowlists below is refused with a 400 before a
reservation is made or a key is chosen.

- OpenRouter: only `POST api/v1/chat/completions`, no streaming. `model`, or
  a `models` fallback list of up to 4 with `route: "fallback"`; every model
  must be in the cached OpenRouter price table, and the reservation is the
  dearest model's worst-case cost of the whole body plus the output cap
  (`max_tokens` or `max_completion_tokens`, clamped to
  `openrouter_max_output_tokens`; reasoning counts inside that cap, and
  `reasoning.max_tokens` is clamped to it). Where `provider.max_price` sets a
  component, that price is reserved instead of the listed one, since
  OpenRouter never routes above it. The broker always sends a full
  `provider.max_price`: a component the runner left out is filled with the
  dearest listed price across the request's models, so OpenRouter cannot
  route to an endpoint dearer than the reservation assumed. The broker always sends
  `provider.data_collection: "deny"` and `zdr: true`, so lead data never goes
  to an endpoint that keeps it; a request asking otherwise is 400
  `provider_privacy_required`. It sets `usage: {include: true}` and settles
  at the reported `usage.cost` of the model that answered.
  Accepted fields: `model`, `models`, `route`, `messages`, `max_tokens`,
  `max_completion_tokens`, `reasoning` (`enabled`, `effort`, `max_tokens`,
  `exclude`), `provider` (`data_collection`, `zdr`, `require_parameters`,
  `max_price` with `prompt`, `completion`, `internal_reasoning`, `request`),
  `user`, sampling settings, `seed`, `stop`, `response_format`, function
  `tools`, `tool_choice`, `parallel_tool_calls`, `stream` (false) and `usage`.
  Any other field, `plugins` included, is 400 `field_not_allowed:<name>`
  (OpenRouter's docs give response-healing no price and do not say it is
  free). Other provider preferences (`order`, `only`, `ignore`, `sort`) are
  400 `provider_field_not_allowed:<name>`. A model variant such as `:online`
  is 400 `model_variant_not_allowed`, a non-function tool 400
  `tool_not_allowed`, and non-text message content 400 `content_not_allowed`.
- OpenRouter Decisions: one additional route, `POST api/alpha/decisions`, only
  for Lead Search's fixed Jev market-fit question. It accepts only
  `typesafe/jev-1.13`, the market and company state, that one choice question,
  and the exact TypeSafe-only ZDR policy (`data_collection: deny`, no
  fallbacks, max prompt price $0.042/M tokens, no output or request charge).
  The body is capped at 24 KiB and the call has a 10-second timeout. Its
  reservation is 1,033 micro-USD (one input token per byte at the enforced
  price); it settles at OpenRouter's reported `usage.cost`. An answer without
  a reported cost remains uncertain under the normal broker rules. Any other
  model, question, field or provider policy is refused before key selection.
- Deepline: only `POST api/v2/integrations/{operation}/execute`, with no query
  string, for the operations the Lead Search finding core uses
  (`DEEPLINE_OPERATIONS` in `lab_arena/lead_orders/broker.py`). Another
  operation is 400 `operation_not_allowed`, a query 400 `query_not_allowed`,
  and any other path or method 404 `route_not_allowed`. Of the caller's headers only
  the three response-shape headers the site sends are passed on, each with
  its one value: `x-deepline-execute-response-contract: raw-v2`,
  `x-deepline-execute-response-intent: raw` and `x-deepline-tool-error-schema:
  1`. Another value is 400 `header_not_allowed:<name>`. They are not stored in
  the ledger or receipts. `firecrawl_scrape` is allowed (Hiring reads a fresh
  copy of an indexed job post whose own page fails); the custom-event tools
  (`serper_google_maps_search`, `socrata_query_dataset`) stay refused. Each
  call reserves the operation's known fixed price where Arena's Deepline price
  table has one (at least 1 micro-USD), else `deepline_reserve_per_call_usd`,
  and settles at the cost read
  from the response body the way the site reads it: `billing.cost_usd` when it
  is a finite number ≥ 0 (rounded up to the micro-USD), else the credit
  fields. Arena's own Deepline cost reading is unchanged. Async job billing
  that is reported only later is a known limit: such calls settle `uncertain`.

The runner also fetches public web pages (LinkedIn guest job pages,
employers' own sites) directly, not through the broker. They cost nothing and
use no key, so they have no ledger row.

## Buyer API (hotkey-signed)

Each request carries `X-Hotkey`, `X-Timestamp` (±60 s) and `X-Signature`, an
sr25519 signature over `METHOD\nHOST\nPATH\nTIMESTAMP\nsha256(body)`. HOST
is the gateway's public host, lowercased (the host of
`LEADPOET_LEAD_ORDERS_PUBLIC_BASE_URL`, with a port only if it is not a
default one: `:443` and `:80` are dropped on both sides),
so a signature made for another gateway, such as Arena's, is refused here.
Miner-terms acceptances store the signed host too.

- `POST /lead-orders/buyers` `{email, company, country, terms_version}`: sends a
  confirmation link; free-mail domains are refused. Orders need a confirmed
  buyer who accepted the current terms.
- `GET /lead-orders/balance`.
- `POST /lead-orders` `{seller, search, leads}` → `{order_id, hold_usd}`.
- `GET /lead-orders/{order_id}` → status, delivered, charged, `delivery_url`
  when delivered, and the receipt.

`POST /lead-orders` may carry its own `order_id` (UUID) to make retries safe;
the same id from the same hotkey and body returns the existing order; the
same id with a different hotkey or body is 409 `order_id_taken`. `seller` and
`search` are objects stored as given and handed to the runner; together they
may not pass 16 KB. `leads` must be 1–1,000 within the policy's limits (else
422 `leads_invalid`). Placement refuses with 422, before any hold, a body the
runner's schema (site `server/lead-search/subnet/order.ts`) would refuse:
`seller.product` 1–200 characters (`seller_invalid:product`),
`seller.summary` present and at most 1,000 (empty is fine;
`seller_invalid:summary`), `search.signal` 1–64 (`search_invalid:signal`),
`search.market` an object (`search_invalid:market`), and the lists, each
optional but a list when present: `terms` ≤ 8 × 120, `buyer_titles` ≤ 12 ×
120, `excluded_companies` ≤ 500 × 253, `market.countries` ≤ 20 × 60,
`market.employee_ranges` ≤ 20 × 60, `market.industries` ≤ 50 × 200
(`search_invalid:<list>` or `search_invalid:market.<list>`). Other text is trimmed
and must not be empty. Other fields are kept and ignored, as the runner does.
A signal the runner does not support passes placement; the runner finishes
that order `failed` with no spend. In `live` a buyer also needs a fresh Gamma reading. The
confirmation link is `GET /lead-orders/buyers/confirm?token=…`. Until a
mailer is wired the registration is stored and the call answers 503
`confirmation_email_unavailable`.

Miner key routes (hotkey-signed): `POST /lead-orders/keys` `{provider,
credentials, daily_cap_usd}` stores the key encrypted with the Arena
credential KMS key and returns `{status, key_id, public_fingerprint,
daily_cap_usd}`; one active key per miner and provider.
`POST /lead-orders/keys/{key_id}/revoke`.

`GET /lead-orders/statement?since=<call sequence>` (hotkey-signed; the
signature covers the path without the query) is a miner's own ledger. It
answers up to 500 calls on the miner's keys after `since`, in sequence order,
each with `sequence`, `call_id`, `order_id`, `key_id`, `provider`,
`reserved_microusd`, `status` (`open`, `settled`, `uncertain`),
`cost_microusd`, `excess_microusd` (cost above the reservation, not
rewarded), `settled_at`, `provider_request_id`, `late_cost_microusd`,
`late_at`, `late_reward_microusd`, `receipt_hash` and
`order_reward_microusd`. It also gives all-time totals per key (`calls`,
reserved, settled, excess, open and uncertain calls, late cost and late
reward), `rewards_owed_microusd`, `rewards_paid_microusd`, `next_since` and
`more`. A miner can match every row to their provider's own usage records by
the request id, which appears in the statement only, never in a public
receipt.

## Suppression feed

People who opted out or asked to be suppressed, as hashes only. Raw emails
and LinkedIn URLs are never accepted or stored. Hashes are sha256 hex
(64 lowercase characters) of:

- `email_sha256`: the trimmed, lowercased email.
- `linkedin_sha256`: the normalised profile URL: trimmed, `http(s)://`, a
  two-letter country subdomain and `www.` stripped, trailing slashes removed,
  lowercased (`https://de.linkedin.com/in/Ada-One/` →
  `linkedin.com/in/ada-one`). This is leadpoet-site's
  `normalizeContactLinkedin`; `lab_arena.lead_orders.suppressions` has the same
  functions in Python.

`POST /internal/lead-orders/suppressions` (runner bearer secret):

```json
{"suppressions": [
  {"email_sha256": "<64 hex>", "linkedin_sha256": "<64 hex>", "added_at": "2026-11-19T10:00:00Z"}
]}
```

1 to 500 entries. Each has `added_at` (UTC, `YYYY-MM-DDTHH:MM:SS[.ffffff]Z`,
not more than 5 minutes ahead) and one or both hashes. Any other field, a
value that is not 64 lowercase hex characters, or a bad `added_at` refuses the
whole push with 422 (`field_not_allowed`, `hash_invalid`, `hash_required`,
`added_at_invalid`, `suppressions_invalid`); error codes never echo a value.
Rows are append-only; a hash already on the list keeps its first row. Answers
`{"received": <entries>, "added": <new hash rows>}`, so a retry is safe.

`GET /lead-orders/suppressions?since=<sequence>` (hotkey-signed; the signature
covers the path without the query): for a confirmed buyer or a miner who
signed the current miner terms, else 403 `suppressions_not_allowed`. Answers
`{"suppressions": [{"sequence", "kind", "hash", "added_at"}], "next_since",
"more"}`, up to 1,000 rows in sequence order; call again with `next_since`
while `more` is true.

Known limit: the hashes are unsalted, so a reader who already has someone's
email or profile URL can tell whether that person is suppressed. That is the
feed's purpose. A reader cannot recover an address it does not already have.

## Receipts (public)

Per order: order id, hotkey, leads requested and delivered, USD charged, policy
version, and per key: provider, miner hotkey, key fingerprint, calls, USD; plus
a Merkle root over the per-call rows. Signed by the gateway. Tool names, models,
prompts, call order and lead data are never published.

- `GET /lead-orders/receipts/{order_id}` → the signed receipt and its call
  rows (call id, provider, miner hotkey, key fingerprint, reserved, status,
  cost), sorted by random call id. Anyone can recompute the Merkle root.
- `GET /lead-orders/epochs/{epoch}` → the signed payout document for an epoch
  (404 when the lane is off, for any input).
- `GET /lead-orders/policies/{version}` → the signed policy document a receipt
  names by `policy_version` (404 `policy_unknown`), so anyone can recompute
  its charge and rewards.
- A payout document lists late rewards as payments that carry the `call_id`
  they repay.

## Weights

In `live`, a payout document's paid shares become a lane fraction per miner
hotkey. `lab_arena.lead_orders.weights.merge_lane_weights` takes that weight
only from the burn residual of the Arena weight vector, never from Arena
winners, keeps the lane under `lane_emission_cap`, and refuses hotkeys that
are not registered. With no lane fractions (any mode but `live`, or an empty
epoch) it returns the Arena weights unchanged. It is not yet called by the
validator's weight signer: wiring it in is a separate reviewed change.

## Known limits

- Lead-order keys are encrypted with the Arena credential KMS key, not a
  dedicated key, so the two share access and audit scope.
- The suppression hashes are unsalted (see the suppression feed).
- Miners see what Leadpoet calls on their keys. Every call runs on miner keys
  in `live` (Gavin's choice, Nov 20 2026), so a miner can read the request
  shapes, models and Deepline operations in their own provider account. The
  lead-finding code stays private; the method behind it is visible.
- Late costs on a failed order are mostly not repaid (see Money), and a late
  Deepline cost is repaid only when Deepline's answer arrives after finish:
  there is no Deepline read-back.
- The ledger stores no model or operation per call, so the statement cannot
  show them; miners match calls by provider request id instead.
