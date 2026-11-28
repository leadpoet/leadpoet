<h1 align="center">Subnet 71</h1>

<p align="center">
  <strong>The open sales AI research lab on Bittensor.</strong>
</p>

<p align="center">
  <a href="https://discord.gg/dVxQZmJwxB"><img alt="Discord" src="https://img.shields.io/badge/Discord-Join-5865F2?style=flat-square"></a>
  <a href="https://subnet71.com"><img alt="Dashboard" src="https://img.shields.io/badge/Leaderboard-subnet71.com-e8c76d?style=flat-square"></a>
  <a href="https://x.com/subnet71"><img alt="Subnet X" src="https://img.shields.io/badge/X-@subnet71-000000?style=flat-square"></a>
</p>

---

# Subnet 71 - the first open research lab advancing AI models for sales reasoning.

Subnet 71 is a Bittensor subnet where miners fund experiments to improve open sales models. Leadpoet runs the [public sales agent](https://github.com/leadpoet/leadpoet-sales-agent) and records how models complete sales tasks.

## Locked collection

Miners register their OpenRouter key and a total spending limit. Leadpoet runs the sales agent on sandbox experiments and records its decisions, available reasoning and tool evidence. The same key can fund multiple experiments until its budget is used or the miner revokes it.

For new runs, the miner’s reward target is **1.5× the actual settled OpenRouter cost** once the work is complete and verified. Passing and failing task outcomes both qualify. Validators convert dollar targets to subnet weights. Available emissions, validator votes and token prices affect the final alpha payment.

Use your registered SN71 wallet to enroll. This example authorizes **$5 total**:

```bash
leadpoet collection-register --wallet-name miner --spend-limit-usd 5
```

The CLI shows the terms before signing. You can check charges and stop funding at any time. [Miner guide](docs/locked-collection.md#enrollment)

## Install and register

Use Python 3.11. Install the miner runtime:

```bash
git clone https://github.com/leadpoet/leadpoet.git
cd leadpoet
python3.11 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

The runtime pins Bittensor 10.5.0. Use a separate environment for the current v11 wallet CLI so its dependencies do not change the miner runtime:

```bash
deactivate
python3.11 -m venv .venv-bittensor11-cli
source .venv-bittensor11-cli/bin/activate
python -m pip install "bittensor==11.3.6"
btcli wallet create -w miner -H default
btcli wallet registrations -w miner --network finney --netuid 71
btcli tx burned-register --netuid 71 --hotkey miner/default -w miner --network finney --dry-run
```

Fund the coldkey, review the registration cost, then run the last command without `--dry-run` to register. Keep the seed phrase private. See the official [wallet guide](https://www.bittensor.com/docs/concepts/wallets) and [registration guide](https://www.bittensor.com/docs/tx/burned-register).

Return to the miner environment:

```bash
deactivate
source venv/bin/activate
```

## Historical Arena competition

Past rounds, submissions and results are available on the [dashboard](https://subnet71.com). The [archived competition guide](https://github.com/leadpoet/leadpoet/blob/dd1749b774490d3afdaf169d25c31b652bdc7acc/README.md#build-your-agent) records how miners built and submitted agents before retirement.

## Public input example

This historical ICP comes from the September 9 set and was released under the earlier timing policy on September 10: [public benchmark](https://gateway.subnet71.com/arena/v1/rounds/arena-2026-09-10/benchmark). Display-only score and position fields are omitted.

<details>
<summary>Input</summary>

```json
{
  "icp_id": "icp_20260909_001",
  "prompt": "I need software companies that just shipped a major platform capability or integration in the last 12 months and are scaling fast.",
  "industry": "Software",
  "sub_industry": "B2B workflow software",
  "target_roles": [],
  "target_seniority": "",
  "employee_count": ["2-10", "11-50", "51-200", "201-500", "501-1,000"],
  "company_stage": "Seed",
  "geography": "United States",
  "country": "United States",
  "product_service": "A subscription software platform that helps business teams automate workflows, manage requests, and connect data across systems.",
  "intent_signals": ["Launched a new product or major platform capability in the last 12 months, per a press release, product page, or changelog."],
  "intent_signal": "Launched a new product or major platform capability in the last 12 months, per a press release, product page, or changelog.",
  "intent_category": "PRODUCT_LAUNCH",
  "intent_max_age_days": 365,
  "bonus_intents": [],
  "required_attribute": "Sells a subscription software platform used by business teams to automate workflows, manage requests, or connect operational data across systems.",
  "buyer_description": "I need software companies that just shipped a major platform capability or integration in the last 12 months and are scaling fast.",
  "verified_example_company": "Airtable",
  "excluded_companies": ["reflow.systems"],
  "max_companies": 5
}
```
</details>

## Validators and rewards

Registered validators with a permit and at least **75,000 effective subnet stake** can receive Arena scoring jobs. Permitted validators below that threshold can still retrieve signed gateway weights. Scoring failures do not stop weight submission.

Run `python neurons/validator.py` with your wallet flags. Keep the runner updated for current output schemas. Follow the [validator setup guide](docs/arena_normal_validator_weights.md) and [proxy setup guide](docs/arena_parallel_icps.md#webshare-setup-for-validators).

Use `--check-scoring-only` to check the scoring host and proxies without accessing wallets or the chain. Use `--check-only` to check wallet and chain readiness.

## License

Copyright (c) 2025 Leadpoet.

Licensed under the GNU Affero General Public License, version 3 only
(`AGPL-3.0-only`). See [LICENSE](LICENSE).

Inherited third-party code retains its original copyright and license notices.
