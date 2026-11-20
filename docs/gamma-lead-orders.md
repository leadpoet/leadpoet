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
`scripts/703-lead-orders.sql` (append-only tables and RPCs, RLS on).

## Modes

| Mode | Who can order | Keys used | Rewards |
| --- | --- | --- | --- |
| `off` | nobody | none | none |
| `staff` | hotkeys in `LEADPOET_LEAD_ORDERS_STAFF_HOTKEYS`, no Gamma | Leadpoet's keys only | none |
| `shadow` | staff hotkeys, no Gamma | ranked miner keys, then Leadpoet's | computed and published, never weighted |
| `live` | any hotkey with Gamma balance | ranked miner keys, then Leadpoet's | weighted, under the lane cap |

Keys follow each order's own mode, not the gateway's current mode: an order
placed in `staff` spends only Leadpoet's keys even after the gateway moves to
`shadow` or `live`.

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
- Miner reward (USD) = floor(that miner's settled spend on the order × (1 +
  premium)). Leadpoet's own keys earn nothing. If an order's total settled
  spend passes its ceiling (a provider reported more than was reserved), every
  reward on it is scaled by ceiling / total spend.
- A call whose cost the provider did not report (no cost field on a 2xx, a
  timeout, a transport error) is settled `uncertain`: its reservation keeps
  counting against the ceiling, and it is never charged to the buyer or
  rewarded. A 4xx with no cost settles at 0.
- An order the runner does not finish by its deadline (plus 300 s grace,
  longer than the broker's 240 s Deepline timeout) is finished as `failed`
  with reason `deadline_passed` at the next claim, and charged the minimum
  charge on its settled spend.
- At finish, the order stops taking reservations and every call still open is
  settled `uncertain`. A provider answer that arrives after finish is kept in
  a separate late-settlement table and is never charged, rewarded or
  published. The receipt's call rows are frozen when the receipt is signed.
- A finished order with no receipt (for example, signing failed) is settled
  again by an idempotent reconcile pass that runs on every claim and every
  epoch pass.
- Per epoch, rewards owed are paid oldest first up to the lane cap; the rest
  carries forward in USD and converts to alpha at the paying epoch's price.
  The emission price source is not decided yet: until it is, `live` payouts are
  refused and `shadow` publishes what is owed with no payments. Shadow payouts
  are never recorded as paid, so shadow does not simulate the carry-forward.
- What is owed is counted per mode. A `live` payout counts only `live` orders,
  so it never pays a shadow backlog; recording a payment against a non-live
  order is refused. Shadow payouts publish shadow orders only and are never
  weighted. Staff orders earn nothing.
- The Gamma service id (`LEADPOET_LEAD_ORDERS_GAMMA_SERVICE`) may be an SS58
  account or a number. The storage item name is configurable
  (`LEADPOET_LEAD_ORDERS_GAMMA_STORAGE`, default `GammaSpent`). A chain without
  it reads as no balance, and `live` orders are refused (`gamma_unavailable`).
  A counter that goes backwards is refused with 422 `gamma_counter_backwards`.

## Key order

Per provider, miner keys are ranked by USD spend settled over the last
`ranking_window_epochs`, highest first; ties and new keys by registration time.
Leadpoet's key is always last. On 401, 402, 403, 429-with-no-retry or a spent
limit the broker moves to the next key for the rest of the order.

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

- OpenRouter: only `POST api/v1/chat/completions`, no streaming. The model
  must be in the cached OpenRouter price table; the reservation is the
  worst-case cost of the prompt plus `max_tokens`. The broker sets `usage:
  {include: true}` and settles at the reported `usage.cost`. Only the fields
  that reservation prices are accepted: `model`, `messages`, `max_tokens`,
  sampling settings, `seed`, `stop`, `response_format`, function `tools`,
  `tool_choice`, `parallel_tool_calls`, `stream` (false) and `usage`. Any other
  field (`models`, `plugins`, `provider`, `reasoning`, ...) is 400
  `field_not_allowed:<name>`. A model variant such as `:online` is 400
  `model_variant_not_allowed`, a non-function tool 400 `tool_not_allowed`, and
  non-text message content 400 `content_not_allowed`.
- Deepline: only `POST api/v2/integrations/{operation}/execute`, with no query
  string, for the operations the Lead Search finding core uses
  (`DEEPLINE_OPERATIONS` in `lab_arena/lead_orders/broker.py`). Another
  operation is 400 `operation_not_allowed`, a query 400 `query_not_allowed`,
  and any other path or method 404 `route_not_allowed`. Each call reserves
  `deepline_reserve_per_call_usd` and settles at the cost read from the
  response body. Async job billing that is
  reported only later is a known limit: such calls settle `uncertain`.

## Buyer API (hotkey-signed)

Each request carries `X-Hotkey`, `X-Timestamp` (±60 s) and `X-Signature`, an
sr25519 signature over `METHOD\nPATH\nTIMESTAMP\nsha256(body)`.

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
may not pass 16 KB. In `live` a buyer also needs a fresh Gamma reading. The
confirmation link is `GET /lead-orders/buyers/confirm?token=…`. Until a
mailer is wired the registration is stored and the call answers 503
`confirmation_email_unavailable`.

Miner key routes (hotkey-signed): `POST /lead-orders/keys` `{provider,
credentials}` stores the key encrypted with the Arena credential KMS key and
returns `{key_id, public_fingerprint}`; one active key per miner and
provider. `POST /lead-orders/keys/{key_id}/revoke`.

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

## Weights

In `live`, a payout document's paid shares become a lane fraction per miner
hotkey. `lab_arena.lead_orders.weights.merge_lane_weights` takes that weight
only from the burn residual of the Arena weight vector, never from Arena
winners, keeps the lane under `lane_emission_cap`, and refuses hotkeys that
are not registered. With no lane fractions (any mode but `live`, or an empty
epoch) it returns the Arena weights unchanged. It is not yet called by the
validator's weight signer: wiring it in is a separate reviewed change.
