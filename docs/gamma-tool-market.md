# Subnet 71 tool market

The Subnet 71 tool market sells metered access to go-to-market (GTM) data tools.
Miners supply the vendor keys and set a price per unit. Agents and buyers
pay in Gamma and call one API. Gamma is Subnet 71's on-chain spending: what a
buyer's hotkey spent to the market's Gamma service on netuid 71
(`GammaSpent`). Each call selects an eligible key from the price-ranked book, subject to health, privacy, price and budget limits. The attested vault and the vendor execute the request. Miners supply the account capacity.

It is off by default. Every route answers 404 until
`LEADPOET_TOOL_MARKET_MODE=live`. In live mode a tool needs an approved supplier before it can serve calls.
`LEADPOET_TOOL_MARKET_TOOLS_OFF` turns single tools off, for example
`exa,hunter`.

## How it works

1. **List a key.** A miner seals a vendor key to the vault on their own
   machine (`python -m lead_order_vault.cli seal`). They then send the
   envelope to `POST /tool-market/keys`, signed with their hotkey.
   - Leadpoet stores only the envelope. Only the vault can open it.
   - The operator first approves the provider agreement for this tool and supplier. The body also carries the miner's signed attestation (below).
   - The vault checks the key with the vendor's free balance call, if the
     vendor has one. A key with no credit is refused.
   - The same vendor key can be listed only once at a time.
   - Only a hotkey registered on the subnet lists keys (`403
     miner_not_registered`), and a miner's own tab never buys from their
     own keys.
   - For security, a hotkey registers at most one key per tool every 30
     days, revoked keys included (`429 key_registration_too_soon` with
     `next_at`). A key revoked before it took a call does not count, but a
     hotkey still registers at most 2 keys per tool in any 30 days.
   - A refused listing is not checked with the vendor again, for that
     hotkey and tool, for a minute (`429 check_too_soon`).
2. **Post an ask.** `POST /tool-market/asks` sets what the miner earns per
   unit. `ask_microusd: null` withdraws the ask.
   - For a per-call tool, the ask is micro-USD per call.
   - For a vendor-USD tool, the ask is micro-USD per vendor USD. For
     example, `900000` sells at 90% of the vendor's own price.
   - An ask is at most $0.25 per call, or 100% of the vendor's own price
     for a vendor-USD tool. Every operation's largest call must owe
     something at the ask (`400 ask_too_low`), and the key's daily cap must
     hold 4 of them (`400 ask_too_high_for_cap`).
3. **Open a tab (an API key).** A buyer pays Gamma to the tool market's
   service once; that spend is their balance and stays until used. They
   sign `POST /tool-market/tabs` with a budget and a lifetime of up to 30
   days, and get back a bearer token to use like any API key. While the
   tab is open, its whole budget is held against the buyer's balance, and
   the rest of the balance stays free for other tabs. Once it closes or
   expires, only what its calls spent stays held.
4. **Call.** `POST /tool-market/call` sends `{tool, operation, params}` with
   the tab token. The market tries keys cheapest first, with equal prices
   in ask order: it reserves on up to 10 asks and sends to up to 3 keys.
   Each call reserves its maximum charge against the tab before it is
   sent, then settles to the real charge. With `max_ask_microusd`, no key
   asking more serves the call.

Both buyers and miners sign the configured terms (`POST /tool-market/terms`,
text in `docs/tool-market-terms.md`) before they trade, and again whenever
their version changes.

## For buyers

`lab_arena/tool_market/client.py` is the whole buyer client. It runs from
this repository and needs only `bittensor_wallet` beyond it, for signing:

```bash
export SUBNET71_URL=https://<gateway>
python -m lab_arena.tool_market.client terms             # once
python -m lab_arena.tool_market.client balance
python -m lab_arena.tool_market.client key --budget 20 --days 30
export SUBNET71_API_KEY=<token it printed>
python -m lab_arena.tool_market.client call hunter email_finder domain=acme.com full_name="Ada Lee"
python -m lab_arena.tool_market.client call exa search query="crm vendors" numResults:=5
python -m lab_arena.tool_market.client key-status        # the key's spend and expiry
python -m lab_arena.tool_market.client key-close         # unspent budget back to the balance
```

`name=value` sends text and `name:=value` sends JSON. `--private` (or
`private` in `call_tool`) uses only suppliers that keep the request
private. The client follows no redirect and talks only https, or http
to this machine.

For an agent, `python -m lab_arena.tool_market.client mcp` is a stdio MCP
server with two tools: `find_tool` (the catalog) and `call_tool`. It holds
only the API key, so an agent can spend at most that key's budget.

## Routes

All under `/tool-market`.

- **Public:** `GET /catalog`, `GET /books/{tool}`.
- **Tab token** (`Authorization: Bearer`): `POST /call`, `GET /tab`,
  `POST /tab/close`.
- **Signed by the hotkey:** `POST /terms`, `GET /balance`, `POST /tabs`,
  `GET /usage`, and for miners `POST /keys`, `POST /keys/{id}/check`,
  `POST /keys/{id}/revoke`, `POST /asks` and `GET /statement` (their keys,
  health and what each earned).

`GET /catalog` lists every tool that trades: its operations, one line on
what each does, their parameters, its price ceiling, and the first ask a
call tries with what a buyer pays for it, so an agent can pick a tool by
the job it needs done. A request with a bad parameter is refused before
any charge. `GET /books/{tool}` shows the asks and what a buyer pays for
each, with no miner hotkeys and no key data.

A buyer has at most 8 calls in flight at once in a gateway process
(`429 too_many_calls_in_flight`), and calls run on their own 32 threads,
so slow free answers cannot hold up the rest of the gateway. Key
registrations and checks share 4 threads of their own, and a miner has
one in flight at a time per process (`429 check_in_flight`).

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

## Using the existing Lead Search client

For an approved ZeroBounce pilot, pass the existing site client a tab token as
`credentials.apiKey` and the gateway origin plus `/tool-market` as
`credentials.hostUrl` **for that operation only**. Its existing
`POST /api/v2/integrations/zerobounce_validate/execute` sends
`{"payload":{"email":"person@example.org"}}`. The adapter uses the same
market reservation and settlement as a native call, requires private supply,
and returns the raw vendor verdict under `toolResponse.raw` with the buyer's
actual charge in `billing.cost_usd`. Every accepted request is a fresh lookup.
A vendor error or malformed verdict is a failed lookup. Retain the client
timeout and the original privacy policy. An ambiguous timeout needs
reconciliation before sending another paid request.

This adapter covers only ZeroBounce verification. Lead Search also uses
CrustData, Harvest jobs/search/posts/reactions, email discovery, bulk tools
and multi-step plays. Even an identically named profile operation can use
different fields, enrichment options and prices. Test other operations against
the platform's existing parser, expected fields, error and billing contracts
before routing them. The platform keeps the lead-search orchestration,
ranking, verification and customer interface.

[Platform client and BYOK accounting](https://github.com/leadpoet/leadpoet-site/blob/932f97a1b38cc6999dec5dac7161facd094fbc52/server/deepline.ts)

## Miner attestation

A miner lists a key only by signing, with their hotkey, this exact
statement inside the register request (`ATTESTATION` in
`lab_arena/tool_market/market.py`; the subnet's name is `NETWORK_NAME`):

> tool-market-attestation-v3: I hold the account this key belongs to, and
> its terms allow me to supply capacity through Bittensor Subnet 71. I agree
> that Subnet 71 may use this key to serve other buyers' requests. I alone am
> responsible for my account's compliance and hold the relationship with the
> provider, and I will revoke this key if that stops being true. Subnet 71,
> its validators and its contributors make no claim about any provider's
> terms and are not liable for any action a provider takes on my account.

The signed request body, its signature, host and time are stored with the
key, so anyone can verify the attestation later. The statement is
versioned: a new version is asked for at the next listing.

## Order book and key health

A key is in its tool's book only while all of these hold:

- the operator approved this supplier for this tool, and the key is listed and not revoked;
- its miner accepted the current terms;
- its latest ask is a price, not a withdrawal;
- its latest health event is `valid` or `unchecked`;
- what it earned in the last 24 h, with unknown usage and calls not
  settled at the most they could owe, is under its daily cap. A miner picks the
  cap, at most $20 a day per key at launch. A call skips a key without
  room for it.

Health events come from vault checks and from calls.

- A vendor answer of 401, 402, 403 or 429 (with or without `Retry-After`)
  moves the call to the next key, and the buyer pays nothing for the
  failed try. These are read as each vendor means them: ZeroBounce's error
  inside a 200, Prospeo's `INSUFFICIENT_CREDITS` and `INVALID_API_KEY`
  inside a 400, Findymail's 423 (paused, read as 401), and Hunter's 429
  (out of searches, read as 402) and 403 (its rate limit, read as 429).
- A buyer shapes the request, so that answer alone does not always take a
  key out of the book. A 401 (a bad key or a paused account) marks it
  `invalid`. A 402 does so only when the vendor's free check finds no
  balance, or none to read; a tool with no check, or a key checked in the
  last minute, only skips the key. A 403 or 429 only skips it.
- A key goes to the back of its book, still in it, after a 429 in the
  last minute, or in the last hour a 402 that did not fail it, a 403 or
  unknown usage, each for 3 different buyers. A buyer can shape each of those,
  so one buyer cannot push a rival's key back.
- There is no scheduled re-check: a key that dies between checks is found
  by its next call.
- After topping up an invalid key, the miner calls
  `POST /tool-market/keys/{id}/check` to put it back in the book. Exa and
  HarvestAPI have no free check, so this is refused
  (`409 key_cannot_be_checked`): a key a call found dead there stays out,
  and the miner lists a new key (the 30-day limit applies).

## Privacy

A miner owns the vendor account behind their key, so whatever that vendor
shows its account owner, the miner can see. The vault protects the key, not
the request.

- Each tool states its `request_privacy` in the catalog and its book:
  `visible` when the vendor documents a request history, `unverified` when
  it does not (treat it as visible), and `private` only after Leadpoet checks
  a vendor account itself (`REQUEST_PRIVACY` in `catalog.py`).
- Miners who sign a private-supply agreement, promising not to read buyers'
  requests, are listed in `LEADPOET_TOOL_MARKET_PRIVATE_SUPPLIERS`. Leadpoet
  can be one of them.
- A call with `"private": true` goes only to those miners' keys, or to a tool
  marked `private`. The book marks which asks can serve one.
- This lowers the risk; it does not remove it. The vendor still holds the
  data.

## Usage history

`GET /tool-market/usage?after=<watermark>&limit=<1-500, default 100>`, signed by the
buyer, lists the buyer's own calls, oldest first: tool, operation, status,
units, charge and times. It never returns a vendor's answer. Pass the
returned `watermark` back as `after` for the next page. A call is listed once
it is 11 minutes old (the 10-minute answer window plus a minute), when it can
no longer change, so paging never skips one. Each call's own answer already
carries its charge.

## Money

All amounts are integer micro-USD.

- **What the miner is owed:** `owed = floor(units × ask / unit_scale)`.
  Units are capped at the operation's maximum.
- **What the buyer pays:** `charge = ceil(owed × 1.10)`.
- **Example:** a call asked at $0.01 charges the buyer $0.011, and the
  miner is owed $0.01. The spread is 10% of the miner ask (about 9.09% of the buyer charge before operating costs). It is a ledger difference. Gamma spend does not transfer cash to Leadpoet.
- **Unknown usage:** sometimes the usage is unknown, such as a 5xx answer,
  a timeout, or a 2xx with no cost in it. Then the buyer pays nothing and
  the miner is owed nothing. The miner bears that risk, at most the key's
  daily cap a day valued at the ask: a key asked below the vendor's price
  can lose more in vendor spend. A buyer skips a key when the next call
  would take their unknown usage and calls in flight past a quarter of its
  cap, so one buyer cannot push a key out of the book for free.
  - A call cut off before it settles, by a restart, counts as unknown
    usage, since it may have reached the vendor. Restarts that cut calls
    can shut a buyer out of a key for a day, or out of a tool for an hour.
    Give the gateway a stop grace above 45 s so a restart drains calls.
  - A request that may have reached the vendor is never retried on another
    key, so it cannot be used twice.
  - A buyer shapes some of these, such as an answer too large to return.
    So a buyer has at most 20 calls to one tool, across all their tabs, that
    are in flight or ended with unknown usage in the last hour. Past that,
    calls to that tool are refused (`429 uncertain_limit`); other tools are
    not affected.
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
| `LEADPOET_TOOL_MARKET_APPROVED_SUPPLIERS` | comma-separated `tool:miner_hotkey` pairs reviewed for third-party resale. Empty by default. Remove the pair and reload gateway configuration to stop new routing. Private supply also requires separate privacy approval. |
| `LEADPOET_TOOL_MARKET_TOOLS_OFF` | comma-separated tools turned off; empty by default, so all trade |
| `LEADPOET_TOOL_MARKET_BUYER_HOTKEYS` | optional allowlist of buyer hotkeys, for a staff rollout |
| `LEADPOET_TOOL_MARKET_PRIVATE_SUPPLIERS` | hotkeys of miners with a private-supply agreement; serve private calls |
| `LEADPOET_TOOL_MARKET_GAMMA_SERVICE` | the tool market's Gamma service SS58 hotkey; must differ from lead orders' |
| `LEADPOET_TOOL_MARKET_TERMS_VERSION`, `LEADPOET_TOOL_MARKET_TERMS_SHA256` | the terms both sides sign: `tool-market-terms-v1` and the SHA-256 of `docs/tool-market-terms.md`'s exact bytes (`PUBLISHED_TERMS` in `config.py`); live mode refuses any other pair |
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
  price each epoch from one snapshot at its closing block, taken once the
  10-minute answer window after that block has passed (a settlement carries
  the epoch its call started in); and publish each epoch's settlements so
  validators can check them.
- **The vendor's answer sets the usage.** Leadpoet counts a credit when the
  answer shows one was spent: a ZeroBounce result other than `unknown`, a
  Prospeo match that is not a free re-enrichment, an email from Findymail
  or Hunter, a successful HarvestAPI profile or company result. Prospeo's rule is read from its docs, not
  from a bill; check it against a miner's statement before scaling.
- **One hotkey per miner is assumed.** The own-key rule, the daily caps
  and the 30-day registration limit are per hotkey. A miner with several
  hotkeys can buy from themselves; with the 10% margin that loses money
  until payouts are wired, so payouts must value what a buyer burns and
  what a miner earns at the same per-epoch price.
- **Supplier approval is per provider and miner.** Reload the list to stop new calls for that pair. Existing in-flight calls can still settle.
- **A running vault is trusted until it restarts.** The gateway checks the
  vault's keys against the KMS on each connect, and only an approved build
  gets those keys. Removing an approval on Base does not stop a vault
  already running; its owner stops it.
- **Exa's cost is an estimate.** Exa calls `costDollars` an estimate.
- **Many buyers can still hold a key's cap or the call threads.** Four
  buyers causing unknown usage can keep a key out of its book for a day,
  and four buyers with slow calls can fill the call threads. Each needs its
  own funded hotkey; rate limits per coldkey would close this.
- **A buyer can use up a key's vendor rate limit.** Answers that find
  nothing, and requests the vendor refuses, are free, so one buyer with 8
  calls in flight can hold a key at a small rate limit. Other buyers' calls
  then move to the next key at no cost, and the key goes to the back only
  once 3 buyers see a 429. A minimum charge per answered call would close
  this.
- **The vault checks a key before the 30-day rule is read.** A hotkey at
  its limit still gets one vendor check per tool a minute, which tells it
  whether a key is live. The miner sealed the key, so it holds it already;
  the check adds only Leadpoet's address. A read-only 30-day check before
  the vault would close this.
- **Three reads grow with history.** A balance reads every tab the buyer
  ever opened, each call sums its key's earlier calls, and the book
  computes each key's last 24 h and last hour from the call ledger on
  every call. Fine at launch volumes; keep running totals, and cache the
  public catalog and books, before they grow.
- **A lost API key holds its budget until it expires.** Only its token
  closes it, and the CLI makes 30-day keys by default; a signed "close my
  keys" route would let a buyer free a lost one.
- **A signed balance read needs no funds.** Each costs a few chain reads,
  and every signed request adds a nonce row (a reading row only when the
  reading changed). Rate-limit the route at the edge before opening it widely.
- **Anyone can fill the vault's connections.** It serves at most 128 at
  once, and one that sends nothing holds its slot for 10 s, so a stranger
  can keep every call at `503 no_supply` (nothing is charged). Limit the
  vault port to the gateway's egress, leaving only `/v1/identity` public.
- **HarvestAPI prices are unconfirmed.** Its per-endpoint prices are not
  published, so launch keeps only its two profile reads.
- **No vault receipts for miners.** Calls do not yet keep the vault's
  signed receipts, so miners cannot check Leadpoet's reports against the
  vault. Lead orders keep them; this lane can add them the same way.
- **Ask increases apply at once.** A buyer who cares sends
  `max_ask_microusd`; a delay on increases can follow.
- **A key's budget is fixed.** Gamma spent later does not raise an open
  key's budget; create another key.
- **The terms name no governing law.** Subnet 71 has no legal entity
  yet; add one when an entity exists.
- **Lead orders still exist.** They stay as they are, and off. Retiring them
  is a separate decision.

## Adding a tool

A tool is data, not a new code path:

1. One `Operation` per endpoint in `lab_arena/tool_market/catalog.py`:
   method, URL, unit, maximum units per call, allowed parameters, how to
   count what the vendor charged, and one line on what it `does`. A vendor
   whose status codes mean something else, such as an error inside a 200,
   names a `real_status` reader.
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
