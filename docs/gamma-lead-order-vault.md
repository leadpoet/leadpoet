# Lead-order key vault

A miner's OpenRouter or Deepline key is opened only inside a small public
service running on Phala Cloud (Intel TDX). Leadpoet stores an encrypted
envelope it cannot open, and sends each provider request through the vault,
which swaps the envelope for the key and returns a signed receipt.

## One split

The vault holds keys and proves how they were used. The gateway decides
what to buy: which models, which Deepline tools, prices and limits. So the
vault checks request shapes, not names, and almost never changes.

| Change | Vault update? |
| --- | --- |
| New OpenRouter model, new Deepline tool, new prices | No |
| New provider, or a provider moves its API | Yes |
| Security fix in the vault | Yes |

Each update is a new build plus one public approval on Base.

## Why it can be trusted

- Phala's KMS releases the vault's private keys only to builds approved on
  the app's contract on Base.
- Each vault public key carries a KMS signature chain. `cli verify` and
  `cli seal` check it against the app id and KMS root key in
  `lead_order_vault/deployment.json`, which anyone can compare with Base.
- The code is `lead_order_vault/envelope.py` and `vault.py`, about 700 lines,
  with `cryptography` as the only dependency.

## Rules in the vault

- OpenRouter: chat completions, Decisions, and the free generation cost
  read-back. Deepline: `POST /api/v2/integrations/<tool>/execute`, any tool.
  Billing, plays, monitors and key management are out of reach.
- HTTPS on 443, no redirects, and a key works only on its own provider.
- The key goes only into `authorization: Bearer`. A response that contains
  the whole key is withheld.
- An OpenRouter management key is used once, at `/v1/check`, to prove the
  miner owns the runtime key. It never makes a call.
- Only Leadpoet's broker, with the caller token, may use `check` and `call`.
- No database, no files, no request logs.

## Receipts

Each forwarded call returns a receipt signed by the vault's receipts key. It
carries the boot id, the key fingerprint, the miner hotkey, host and path,
SHA-256 of the target, headers and bodies, and a per-key sequence number.
A miner checks the sequence numbers in the receipts Leadpoet reports: a gap
is a call Leadpoet did not report. Check receipts against the receipts key
from `cli verify`, never against the `public_key` a receipt carries.

## Who you are trusting

The vault's keys belong to the app, not to one build, so any build the
contract owner approves can open every sealed key. The owner is the real
trust point. Approvals are public on Base, but Phala has no timelock.

## Miner steps

```bash
python -m lead_order_vault.cli verify
python -m lead_order_vault.cli seal --provider openrouter --hotkey <your hotkey>
python -m lead_order_vault.cli receipts page1.json page2.json
```

`seal` reads the key from the terminal and prints the envelope to submit as
`credentials` to `POST /lead-orders/keys`. `receipts` takes every page of
`GET /lead-orders/vault-receipts` from `since=0`. It checks each signature
against the vault's receipts key, lists any gap in each key's sequence, and
names any key to seal again because the vault's key changed. It exits 2 when
something is wrong. All commands refuse until the vault is deployed.

## Releasing

1. Merge the change to `main`, then run **Lead-order vault release** from
   Actions on `main`. It builds the image twice, stops unless both digests
   match, pushes `ghcr.io/leadpoet/lead-order-vault@sha256:…`, and uploads
   `docker-compose.yml` pinned to that digest with `release.json` (commit,
   image, commit time used as the build time, BuildKit image).
2. Update the Phala Cloud app with that compose file. Phala shows its
   compose hash.
3. The app owner approves the hash with `addComposeHash` on the app's
   contract on Base. This is the one step that grants a build the keys, so
   it stays a signed, manual step.
4. Run `cli verify`. Its `compose_hash` must be the approved one.

First deploy only: make the GHCR package public so Phala can pull it, and
fill `deployment.json` (url, app id, KMS root key, contract, chain). The
**Lead-order vault uptime** check then runs `cli verify` every 30 minutes;
a failed run is the alert.

To check a release yourself, check out the commit in `release.json` and run
`bash .github/scripts/lead_order_vault_build.sh` from the repository root
(Docker with buildx). It prints the digest, which must match the image in
the compose file. Then compare the compose file's hash with the one
approved on Base and the one `cli verify` prints.

## Known limits

- Sequence numbers restart with the vault. A hidden call just before a
  restart, or after a key's last reported call, leaves no gap.
- If Phala rotates its KMS root, miners must seal their keys again, with a
  new provider key: a key is registered once, and a revoked key cannot
  return.
- A second vault instance of the same app numbers calls under its own boot
  id. Hidden calls made only there leave no gap. Only the app's owner can
  start one.
- One instance, no high availability.
- The vault does not judge which Deepline tools are worth paying for; the
  gateway does, and the receipts show every call.
- Partial key echoes, such as a few characters in a label, are not caught.
