# Gamma tool market economics

Price the operation and the result the buyer needs. A cached company lookup, a fresh full profile, a short profile and a profile with email have different costs and deliver different data. Compare equivalent calls on the buyer's existing plan. When the buyer already brings a vendor key to its gateway, the relevant cost includes the vendor's debit and any actual gateway charge.

## Published price checks (November 2026)

USD per unit for one fresh request with a billable result. Recheck rates and contract terms before enabling supply.

| Operation | Direct provider | Treg managed | Deepline managed |
|---|---|---:|---:|
| ZeroBounce single email verification | $0.0129 at 10,000 prepaid, $0.00649 at 100,000, $0.004398 at 500,000 | $0.0138 | $0.014 |
| HarvestAPI LinkedIn company lookup | Account tariff; Treg reports observing a $0.004 debit on its own Harvest account | $0.004 | Current public tariff unverified |
| HarvestAPI LinkedIn full profile without email | Account tariff; Treg reports observing a $0.0064 debit on its own Harvest account | $0.0064 | Current public tariff unverified |

ZeroBounce does not debit an unknown result. The Deepline number uses the tool's dated `currentPricing.usdPerUnit` field (November 27), rather than the older comparison field on the same page. Treg also quotes a basic Harvest profile at $0.004. Harvest documents the lower-cost `main` option and higher-cost email option. Treg's Harvest prices are observations on one account and should not be treated as every supplier's invoice. [ZeroBounce unit prices](https://www.zerobounce.net/docs/frequently-asked-questions/billing-and-payments/how-is-pricing-determined-for-email-verification) · [Deepline ZeroBounce catalog row](https://deepline.com/api/public/gtm-data-pricing/actions?source=deepline&q=zerobounce_validate) · [Treg ZeroBounce](https://treg.to/tools/zerobounce) · [Treg HarvestAPI](https://treg.to/tools/harvestapi) · [Harvest profile contract](https://docs.harvestapi.io/linkedin-api-reference/profile/get)

## Break-even for a buyer and a miner

For one billable call with a miner ask of `A` integer micro-USD, the buyer pays `ceil(1.10 × A)`. The maximum ask that strictly beats an incumbent cost of `C` integer micro-USD is `floor(10 × (C − 1) / 11)`. Miner contribution equals the amount actually collected less vendor debits (including billable misses and ambiguous failures), unused prepaid capacity, and operating and conversion costs. A unit sold below that total cannot sustain supply. Each Harvest tool ask covers both of its operations, so the supplier must survive a buyer selecting only the higher-cost one. [Charge implementation](../lab_arena/tool_market/book.py)

For example, an approved ZeroBounce supplier consuming a full 500,000-credit pack could ask $0.0055 per billable check. Per 100,000 checks the buyer spends $605, the supplier accrues $550 against $439.80 of vendor usage, and the gross protocol spread is $55. A buyer already paying $649 directly saves $44 (6.8%) before routing and conversion expenses. A supplier paying the same $649 needs a higher ask than this buyer can accept. Reseller contract pricing may differ from retail.

The 10% markup on the ask is about 9.09% of buyer spend before operating costs. It does not establish cash revenue for the platform. Only vendor spend actually displaced improves Leadpoet's gross margin, assuming unchanged customer pricing and quality. For example, on a hypothetical $5,000 monthly bill, moving 20% of spend at a genuine 15% discount saves $150/month gross. Fixed subscriptions that must be kept, included credits, fallback calls and incremental operating costs can erase that saving.

## Adoption decision

1. Reconcile a billing period of buyer tool receipts to Deepline/Treg and direct vendor statements. Group by operation, requested options, funding source, cache hit, billed outcome, usable result and dollars paid. Count a prepaid provider credit once. Keep experimental-lab and customer-funded usage out of the platform's own cost baseline.
2. Choose operations for which an authorized supplier's ask passes the buyer's actual break-even cap. Replay representative saved answers and compare field coverage, cache behavior, failure rate and p95 latency. Before changing the default route, validate a bounded parallel sample against live vendor billing and latency.
3. Confirm written supplier redistribution rights and data handling for each vendor. ZeroBounce's standard terms restrict third-party access; its agency partner program offers separately arranged client services. Harvest requires key confidentiality and respect for account quotas. Confirm that a supplier's agreement covers this precise shared service. [ZeroBounce terms §2](https://www.zerobounce.net/terms-of-use) · [ZeroBounce partners](https://www.zerobounce.net/agency-partners) · [HarvestAPI terms](https://harvestapi.io/info/terms-of-service)

A funded tab and a recorded supplier liability prove separate parts of the transaction. Verify actual supplier settlement and repeat purchases before counting paid demand. Measuring a Gamma spend counter does not execute a market purchase of alpha or a buyback, and miner receivables do not fund another tool's tab. Track externally funded usage separately from platform routing and supplier self-dealing.
