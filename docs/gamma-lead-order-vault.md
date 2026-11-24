# Lead-order key vault

A miner's OpenRouter or Deepline key is opened only inside a small public
service running on Phala Cloud (Intel TDX). Leadpoet stores an encrypted
envelope it cannot open, and sends each provider request through the vault,
which swaps the envelope for the key.

## Why it can be trusted

- The vault's private keys come from Phala's KMS, which releases them only to
  a build whose compose hash is approved on the app's contract on Base. The
  approval list is public, and its owner is a multisig.
- Each vault public key carries a KMS signature chain. `cli verify` and
  `cli seal` check it against the app id and KMS root key in
  `lead_order_vault/deployment.json`, which anyone can compare with Base.
- The code is `lead_order_vault/envelope.py` and `vault.py`, about 700 lines,
  with `cryptography` as the only dependency.

## Fixed rules in the vault

- Only the requests the lead-order broker sends: OpenRouter chat completions,
  the Decisions route and the free generation cost read-back; Deepline
  `integrations/<tool>/execute` for the listed tools. HTTPS on 443, no
  redirects. A key works only on its own provider's host. Billing, plays,
  monitors and key management are out of reach.
- A new Deepline tool or OpenRouter route needs a new vault build and its
  on-chain approval. A test keeps the vault's list equal to the broker's.
- The key goes only into `authorization: Bearer`.
- An OpenRouter management key is used once, at `/v1/check`, to prove the
  miner owns the runtime key. It can never make a call.
- A response that contains the whole key is withheld. A provider that shows a
  few characters of a key (as OpenRouter's key labels do) is not caught.
- Only Leadpoet's broker, with the caller token, may use `check` and `call`.
- No database, no files, no request logs.

## Receipts

Every forwarded call returns a receipt signed by the vault's receipts key:
boot id, global and per-key sequence numbers, key fingerprint, miner hotkey,
host, path, and SHA-256 of the full target, the forwarded headers, and the
request and response bodies. Check receipts against the receipts key from
`cli verify`, never against the `public_key` a receipt carries. `GET /v1/usage`
returns a signed per-key call count for the current boot; pass a random
`nonce` to prove the answer is fresh. A miner compares it
with the calls Leadpoet reports; a gap means a call Leadpoet did not report.

## Who you are trusting

The vault's keys belong to the app, not to one build. Any build the app's
owner approves later can open every key sealed before it. So the owner of the
approval contract is the real trust point. If Leadpoet alone controls it,
Leadpoet could approve a build that reads keys; the approval would be public
on Base, but there is no timelock to warn miners first. Watch the contract's
approval events, and seal again only to builds you have checked.

## Miner steps

```bash
python -m lead_order_vault.cli verify
python -m lead_order_vault.cli seal --provider openrouter --hotkey <your hotkey>
```

`seal` reads the key from the terminal and prints the envelope to submit.
Both commands refuse until the vault is deployed.

## Known limits

- Counters reset on restart. Calls made just before a restart cannot be
  compared, though the new boot id shows a restart happened.
- If Phala rotates its KMS root, the vault's keys change and miners must seal
  their keys again.
- One instance, no high availability. When it is down, miner-key calls stop.
- No timelock on build approvals; the multisig is the control.
- The vault proves which calls it made, not that a provider call outside the
  vault never happened with a key leaked elsewhere.
