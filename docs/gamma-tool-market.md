# Gamma tool market

Gamma's tool market sells metered access to go-to-market (GTM) data tools.
Miners supply the vendor keys and set a price per unit. Agents and buyers
pay in Gamma and call one API. Each call goes to the cheapest healthy key.

It is off by default. Every route answers 404 until
`LEADPOET_TOOL_MARKET_MODE=live`. A tool trades only once it is listed in
`LEADPOET_TOOL_MARKET_TOOLS`, which is empty by default.

## Why a tool is off until its vendor agrees

Most vendors' terms forbid sharing or reselling a key. A tool joins
`LEADPOET_TOOL_MARKET_TOOLS` only after one of these:

- the vendor agrees in writing that keys may be resold this way; or
- Leadpoet signs a reseller deal with the vendor.

Deepline and OpenRouter are left out for this reason. [Deepline's terms
§4](https://deepline.com/terms) bar sharing credentials, and Deepline
competes with this product. [OpenRouter's terms](https://openrouter.ai/terms)
bar selling or transferring access.

## Launch set

These are pure usage-based APIs that Leadpoet already uses. Each one either
reports its own usage or charges a fixed number of credits per request.

| Tool | Unit | Operations | Free key check |
| --- | --- | --- | --- |
| [ScrapeCreators](https://docs.scrapecreators.com) | per call (1 credit) | `linkedin_profile`, `linkedin_company`, `google_search` | yes: [credit balance](https://docs.scrapecreators.com/v1/account/credit-balance) |
| [DataForSEO](https://docs.dataforseo.com/v3/serp/google/organic/live/advanced/) | vendor USD (`cost` in each answer) | `google_organic_serp`, `google_ads_search_volume` | yes: [user data](https://docs.dataforseo.com/v3/appendix/user_data/) |
| [Exa](https://docs.exa.ai/reference/search) | vendor USD (`costDollars.total`) | `search`, `contents` | no |
| [HarvestAPI](https://docs.harvestapi.io/) | per call | `linkedin_profile`, `linkedin_company` | no |

The operations and their allowed parameters are in
`lab_arena/tool_market/catalog.py`. The key vault enforces the same hosts
and route shapes again (`lead_order_vault/vault.py`, `TOOLS`).

## 2.0 list

These tools are good candidates once the launch set has run:

- **Email checks:** NeverBounce, ZeroBounce, Reoon, Kickbox. NeverBounce
  takes its key in a query parameter and reports errors inside a 2xx
  answer, so the vault and catalog need a rule for that first.
- **Signals and search:** PredictLeads, Serper.
- **Email finding:** Enrow.
- **Reseller deals only:** FullEnrich, Prospeo, Findymail and Crustdata.
  Their terms rule out shared keys, so they join only through a reseller
  deal.

## How it works

1. **List a key.** A miner seals a vendor key to the vault on their own
   machine (`python -m lead_order_vault.cli seal`). They then send the
   envelope to `POST /tool-market/keys`, signed with their hotkey.
   - Leadpoet stores only the envelope. Only the vault can open it.
   - The vault checks the key with the vendor's free balance call, if the
     vendor has one. A key with no credit is refused.
   - The same vendor key can be listed only once at a time.
2. **Post an ask.** `POST /tool-market/asks` sets what the miner earns per
   unit. `ask_microusd: null` withdraws the ask.
   - For a per-call tool, the ask is micro-USD per call.
   - For a vendor-USD tool, the ask is micro-USD per vendor USD. For
     example, `900000` sells at 90% of the vendor's own price.
3. **Open a tab.** A buyer pays Gamma to the tool market's service. They
   sign `POST /tool-market/tabs` with a budget and a lifetime, and get back
   a bearer token. While the tab is open, its whole budget is held against
   the buyer's balance. Once it closes or expires, only what its calls
   spent stays held.
4. **Call.** `POST /tool-market/call` sends `{tool, operation, params}` with
   the tab token. The market tries keys cheapest first, with equal prices
   in ask order, and tries up to 3 keys. Each call reserves its maximum
   charge against the tab before it is sent, then settles to the real
   charge.

Both buyers and miners sign the configured terms (`POST /tool-market/terms`)
before they trade. This change writes no terms text.

## Order book and key health

A key is in its tool's book only while all of these hold:

- it is listed and not revoked;
- its miner accepted the current terms;
- its latest ask is a price, not a withdrawal;
- its latest health event is `valid` or `unchecked`;
- what it earned in the last 24 h is under its daily cap.

Health events come from vault checks and from calls.

- A vendor answer of 401, 402 or 403 marks the key `failed`. The call then
  moves on to the next key, and the buyer pays nothing for the failed try.
- A 429 with no `Retry-After` skips the key for this call only. A 429 with
  `Retry-After` goes back to the buyer.
- A key whose vendor has a free check is checked again before use once its
  last check is older than 6 hours. This needs no cron job.
- After topping up a failed key, the miner calls
  `POST /tool-market/keys/{id}/check` to put it back in the book.

`GET /tool-market/books/{tool}` shows the asks and what a buyer pays for
each. It shows no miner hotkeys and no key data.

## Money

All amounts are integer micro-USD.

- **What the miner is owed:** `owed = floor(units × ask / unit_scale)`.
  Units are capped at the operation's maximum.
- **What the buyer pays:** `charge = ceil(owed × 1.10)`.
- **Example:** a call asked at $0.01 charges the buyer $0.011, and the
  miner is owed $0.01. The 10% difference is Leadpoet's.
- **Unknown usage:** sometimes the usage is unknown, such as a 5xx answer,
  a timeout, or a 2xx with no cost in it. Then the buyer pays nothing and
  the miner is owed nothing. The miner bears that risk.
  - A request that may have reached the vendor is never retried on another
    key, so it cannot be used twice.
- **Late answers:** a call still open after 10 minutes holds nothing and
  can no longer settle.

## Storage

All tool-market data is in `scripts/770-tool-market.sql`.

- Every table is append-only, with row-level security on and no table
  grants. The service role reaches the data only through its functions.
- The SQL enforces the money rules on stored state: tab budgets, daily
  caps, one listing per key, one settlement per call, and no charge above
  the call's reserve.

## Configuration

| Variable | Meaning |
| --- | --- |
| `LEADPOET_TOOL_MARKET_MODE` | `off` (default) or `live` |
| `LEADPOET_TOOL_MARKET_TOOLS` | comma-separated tools that may trade; empty by default |
| `LEADPOET_TOOL_MARKET_BUYER_HOTKEYS` | optional allowlist of buyer hotkeys, for a staff rollout |
| `LEADPOET_TOOL_MARKET_GAMMA_SERVICE` | the tool market's Gamma service SS58 hotkey |
| `LEADPOET_TOOL_MARKET_TERMS_VERSION`, `LEADPOET_TOOL_MARKET_TERMS_SHA256` | the terms text both sides sign |
| `LEADPOET_TOOL_MARKET_PUBLIC_BASE_URL` | the https host every signed request names |

It reuses the Arena store and chain settings, and the key vault
deployment, exactly as lead orders do.

## Known limits

- **Miner payout is not wired yet.** Settlements record what each miner is
  owed for each epoch. Turning that into weight needs a payout ledger and
  validator wiring, which reuse `lab_arena.lead_orders.money.epoch_payout`.
  Until then, owed amounts build up but pay nothing.
- **The vendor's own report sets the usage.** For per-call tools, Leadpoet
  counts a 2xx answer as one use, whatever its body says.
- **Exa's cost is an estimate.** Exa calls `costDollars` an estimate.
- **Some keys cannot be checked first.** Exa and HarvestAPI have no free
  key check, so a dead key is found by its first failed call.
- **HarvestAPI prices are unconfirmed.** Its per-endpoint prices are not
  published, so launch keeps only its two profile reads.
- **No vault receipts for miners.** Calls do not yet keep the vault's
  signed receipts, so miners cannot check Leadpoet's reports against the
  vault. Lead orders keep them; this lane can add them the same way.
- **Lead orders still exist.** They stay as they are, and off. Retiring them
  is a separate decision.
