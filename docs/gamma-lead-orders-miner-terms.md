# Gamma lead orders: miner data terms (PLACEHOLDER)

> **PLACEHOLDER. NOT LEGAL TEXT. Leadpoet must write and approve these terms
> before any miner key is accepted in `shadow` or `live`.** The version a
> miner signs is the policy's `miner_terms_version`, currently
> `miner-terms-placeholder-1`. Publishing real terms means a new version in a
> new signed policy record; every miner then signs again.

The terms must cover at least:

1. **Use.** Lead data seen through a provider key (requests, responses, logs
   and usage records in the miner's OpenRouter or Deepline account) is used
   only to run Leadpoet lead orders.
2. **No copies.** The miner keeps no copy of that data, and turns off or
   deletes provider-side logging and retention where the provider allows it.
3. **Suppression.** The miner honours the suppression feed
   (`GET /lead-orders/suppressions`): no contact with, and no processing of,
   a suppressed person.

Leadpoet still needs to decide: retention and deletion periods, audit and
enforcement (revoking a key, withholding rewards), the governing law, and how
the terms relate to Leadpoet's customer data agreements and sub-processor list.
