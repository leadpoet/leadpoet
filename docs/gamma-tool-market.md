# Gamma tool market

The Gamma tool market sells metered access to go-to-market (GTM) data tools.
Miners supply the vendor keys and set a price per unit. Agents and buyers
pay in Gamma and call one API. Each call goes to the cheapest healthy key.

It is off by default. Every route answers 404 until
`LEADPOET_TOOL_MARKET_MODE=live`. Once live, every catalog tool trades.
`LEADPOET_TOOL_MARKET_TOOLS_OFF` turns single tools off, for example
`exa,hunter`.

## Miner attestation

A miner lists a key only by signing, with their hotkey, this exact
statement inside the register request (`ATTESTATION` in
`lab_arena/tool_market/market.py`; the network's name is `NETWORK_NAME`):

> tool-market-attestation-v2: I hold the account this key belongs to, and
> its terms allow me to supply capacity through the Gamma network. I agree
> that the network may use this key to serve other buyers' requests. I alone
> am responsible for my account's compliance and hold the relationship with
> the provider, and I will revoke this key if that stops being true. The
> network, its validators and its contributors make no claim about any
> provider's terms and are not liable for any action a provider takes on my
> account.

The signed request body, its signature, host and time are stored with the
key, so anyone can verify the attestation later. The statement is
versioned: a new version is asked for at the next listing. The market's
terms (`POST /tool-market/terms`) are asked again whenever their version
changes, and a key trades only while its miner accepted the current ones.
This follows Bittensor subnet 28's miner terms.

## Launch set

These are the end vendors Leadpoet already uses, called directly with each
miner's own vendor key. Deepline is not in the market: it resells these
same vendors. Each one bills per request or reports its own cost.

| Tool | Unit | Operations | Free key check |
| --- | --- | --- | --- |
| [ZeroBounce](https://www.zerobounce.net/docs/) | per credit; an `unknown` result is free | `validate` | yes: credits |
| [Prospeo](https://prospeo.io/api-docs) | per credit; no match is free | `enrich_person` | yes: account information |
| [Findymail](https://app.findymail.com/docs) | per credit; no email found is free | `find_from_linkedin`, `find_from_name` | yes: credits |
| [Hunter](https://hunter.io/api-documentation/v2) | per search; no email found is free | `email_finder` | yes: account (key only, no balance) |
| [HarvestAPI](https://docs.harvestapi.io/) | per call | `linkedin_profile`, `linkedin_company` | no |
| [Exa](https://docs.exa.ai/reference/search) | vendor USD (`costDollars.total`) | `search`, `contents` | no |

## Adding a tool

A tool is data, not a new code path:

1. One `Operation` per endpoint in `lab_arena/tool_market/catalog.py`:
   method, URL, unit, maximum units per call, allowed parameters, and how
   to count what the vendor charged. A vendor whose status codes mean
   something else, such as an error inside a 200, names a `real_status`
   reader.
2. One `Tool` entry in `lead_order_vault/tools.py`: the host, where the key
   goes (a header, with or without `Bearer`, or a query parameter), which
   routes it may call, and its free balance check. The vault is a separate
   attested image, so it keeps this allowlist rather than trusting the
   gateway's catalog. A vault change needs a new vault release.

## 2.0 list

These tools are good candidates once the launch set has run:

- **Signals:** PredictLeads (jobs, news, technologies).
- **Email finding:** LeadMagic, Dropleads, Enrow.
- **Email checks:** NeverBounce, Reoon, Kickbox, BounceBan.
- **Search and scraping:** DataForSEO, ScrapeCreators, Serper.
- **Reseller deals:** FullEnrich and Crustdata.

## How it works

1. **List a key.** A miner seals a vendor key to the vault on their own
   machine (`python -m lead_order_vault.cli seal`). They then send the
   envelope to `POST /tool-market/keys`, signed with their hotkey.
   - Leadpoet stores only the envelope. Only the vault can open it.
   - The body carries the miner's signed attestation (above).
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
- what it earned in the last 24 h is under its daily cap. A miner picks
  the cap, at most $20 a day per key at launch.

Health events come from vault checks and from calls.

- A vendor answer of 401, 402 or 403 moves the call to the next key, and
  the buyer pays nothing for the failed try. These are read as each vendor
  means them: ZeroBounce's error inside a 200, Prospeo's
  `INSUFFICIENT_CREDITS` and `INVALID_API_KEY` inside a 400, Findymail's
  423 (paused, read as 401) and Hunter's 429 (out of searches).
- A buyer shapes the request, so that answer alone does not always take a
  key out of the book. A 401 (a bad key or a paused account) marks it
  `failed`. A 402 marks it `failed` unless the vendor's free check still sees
  a positive balance; the check runs at most once a minute per key, and
  without one due a 402 only skips the key. A 403 only skips it.
- A key goes to the back of its book, still in it, after a bare 429 in the
  last minute or 3 answers of unknown usage (5xx, timeouts) in the last hour.
- A key whose last 3 calls were all refused (403), for at least two buyers,
  leaves the book until its miner asks for a check.
- A 429 with no `Retry-After` skips the key for this call only. Hunter's
  403 is its rate limit, so it counts as a 429. A 429 with
  `Retry-After` goes back to the buyer.
- A key whose vendor has a free check is checked again before use once its
  last check is older than 6 hours. This needs no cron job.
- After topping up a failed key, the miner calls
  `POST /tool-market/keys/{id}/check` to put it back in the book.

A miner's own tab never buys from their own keys. For security, a hotkey
registers at most one key per tool every 30 days, revoked keys included. A
revoked key that never served a call does not count.

An ask may be at most $0.25 per credit, or 100% of the vendor's own price
for a vendor-USD tool. A buyer can send `max_ask_microusd` with a call, and
no key asking more serves it.

`GET /tool-market/catalog` lists every tool that trades: its operations and
their parameters, its price ceiling, and the cheapest ask with what a buyer
pays for it. `GET /tool-market/books/{tool}` shows the asks and what a buyer pays for
each. It shows no miner hotkeys and no key data.

## Money

All amounts are integer micro-USD.

- **What the miner is owed:** `owed = floor(units × ask / unit_scale)`.
  Units are capped at the operation's maximum.
- **What the buyer pays:** `charge = ceil(owed × 1.10)`.
- **Example:** a call asked at $0.01 charges the buyer $0.011, and the
  miner is owed $0.01. The 10% is the subnet's margin; Leadpoet is one
  buyer among others.
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
| `LEADPOET_TOOL_MARKET_TOOLS_OFF` | comma-separated tools turned off; empty by default, so all trade |
| `LEADPOET_TOOL_MARKET_BUYER_HOTKEYS` | optional allowlist of buyer hotkeys, for a staff rollout |
| `LEADPOET_TOOL_MARKET_GAMMA_SERVICE` | the tool market's Gamma service SS58 hotkey; must differ from lead orders' |
| `LEADPOET_TOOL_MARKET_TERMS_VERSION`, `LEADPOET_TOOL_MARKET_TERMS_SHA256` | the terms text both sides sign |
| `LEADPOET_TOOL_MARKET_PUBLIC_BASE_URL` | the https host every signed request names |

It reuses the Arena store and chain settings, and the key vault
deployment, exactly as lead orders do.

## Known limits

- **Miner payout is not wired yet.** Settlements record what each miner is
  owed for each epoch. Turning that into weight needs a payout ledger and
  validator wiring, which reuse `lab_arena.lead_orders.money.epoch_payout`.
  Until then, owed amounts build up but pay nothing. That wiring must:
  skip a hotkey that is not registered on the subnet and keep its amount
  owed, rather than fail the whole payout; give every positive share at
  least one weight unit, or record as paid only what the weights deliver;
  price each epoch from one snapshot at its closing block; and publish each
  epoch's settlements so validators can check them.
- **The vendor's answer sets the usage.** Leadpoet counts a credit when the
  answer shows one was spent: a ZeroBounce result other than `unknown`, a
  Prospeo match that is not a free re-enrichment, an email from Findymail
  or Hunter, any HarvestAPI 2xx. Prospeo's rule is read from its docs, not
  from a bill; check it against a miner's statement before scaling.
- **ZeroBounce cannot tell a dead key from an empty account.** Both mark
  the key `no_credit` until its miner asks for a check.
- **One hotkey per miner is assumed.** The own-key rule, the daily caps
  and the 30-day registration limit are per hotkey. A miner with several
  hotkeys can buy from themselves; with the 10% margin that loses money
  until payouts are wired, so payouts must value what a buyer burns and
  what a miner earns at the same per-epoch price.
- **Day 1 has no operator hold.** Ops can turn a whole tool off
  (`LEADPOET_TOOL_MARKET_TOOLS_OFF`) but cannot hold one key or miner.
- **A running vault is trusted until it restarts.** The gateway checks the
  vault's keys against the KMS on each connect, and only an approved build
  gets those keys. Removing an approval on Base does not stop a vault
  already running; its owner stops it.
- **Exa's cost is an estimate.** Exa calls `costDollars` an estimate.
- **Some keys cannot be checked first.** Exa and HarvestAPI have no free
  key check, so a dead key is found by its first failed call. Hunter's
  check proves the key but reads no balance.
- **HarvestAPI prices are unconfirmed.** Its per-endpoint prices are not
  published, so launch keeps only its two profile reads.
- **No vault receipts for miners.** Calls do not yet keep the vault's
  signed receipts, so miners cannot check Leadpoet's reports against the
  vault. Lead orders keep them; this lane can add them the same way.
- **Ask increases apply at once.** A buyer who cares sends
  `max_ask_microusd`; a delay on increases can follow.
- **No MCP server yet.** Agents call the HTTP API with a tab token. An MCP
  server that lists the catalog's operations as tools is a separate package.
- **Lead orders still exist.** They stay as they are, and off. Retiring them
  is a separate decision.
