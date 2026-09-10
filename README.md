<h1 align="center">Leadpoet</h1>

<p align="center">
  <strong>AI sales intelligence, built on Bittensor.</strong>
</p>

<p align="center">
  <a href="https://discord.gg/tMcmbPKvz"><img alt="Discord" src="https://img.shields.io/badge/Discord-Join-5865F2?style=flat-square"></a>
  <a href="https://subnet71.com"><img alt="Dashboard" src="https://img.shields.io/badge/Leaderboard-subnet71.com-e8c76d?style=flat-square"></a>
  <a href="https://leadpoet.com"><img alt="Website" src="https://img.shields.io/badge/Website-leadpoet.com-f3f4f6?style=flat-square"></a>
  <a href="https://x.com/subnet71"><img alt="Subnet X" src="https://img.shields.io/badge/X-@subnet71-000000?style=flat-square"></a>
  <a href="https://x.com/LeadpoetAI"><img alt="Leadpoet X" src="https://img.shields.io/badge/X-@LeadpoetAI-000000?style=flat-square"></a>
</p>


---

# Leadpoet Open Source Agent Competition

Leadpoet is Bittensor subnet 71. Miners improve an open sales-research agent. The public baseline lives in the promoted `lab` branch of [pydantic-harness](https://github.com/leadpoet/pydantic-harness).

## Daily schedule

Day 0: submit models while that day's 20 ICPs stay hidden. At about **00:00 UTC on Day 1**, those ICPs become public and evaluation starts for the baseline and prior-day models. All 20 are processed without fixed pauses between batches. Code, final and per-ICP scores publish when evaluation finishes. Day 1's new ICP set stays hidden until Day 2.

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
python -m pip install "bittensor==11.1.0"
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

## Build your agent

Fork the public baseline, or clone its promoted version as a starting point:

```bash
git clone --branch lab https://github.com/leadpoet/pydantic-harness.git ../my-agent
```

In `harness.py`, define or re-export this synchronous function with exactly one positional parameter:

```python
def run_icp(icp: dict) -> list[dict]:
    """Return at most five companies."""
```

Change the harness, model, prompts, and approved API routing. List Python dependencies in `requirements.txt`: package names and version constraints only, with binary wheels available. URLs, VCS dependencies, local paths, nested requirements, and source builds are not supported. Return at most five companies as a JSON list. Use `[]` if there are no valid matches.

The output below shows the exact supported fields. It is a format example, not a real company claim. `company_linkedin`, `company_stage`, and `state` may be empty; `required_attribute` may be `null` when it is not required. `matched_icp_signal` is the zero-based position in the input's `intent_signals` list.

<details>
<summary>Output</summary>

```json
[
  {
    "company_name": "Example",
    "company_website": "https://example.com/",
    "company_linkedin": "",
    "industry": "Software",
    "employee_count": "51-200",
    "company_stage": "Series A",
    "country": "United States",
    "state": "",
    "fit_summary": "Why it fits.",
    "fit_evidence_urls": ["https://example.com/about"],
    "intent_signals": [
      {
        "matched_icp_signal": 0,
        "description": "Recent required event.",
        "date": "2026-08-20",
        "why_now": "Why now.",
        "url": "https://example.com/news",
        "snippet": "Supporting source text."
      }
    ],
    "required_attribute": {
      "text": "Required characteristic.",
      "passed": true,
      "evidence_url": "https://example.com/about",
      "evidence_quote": "Supporting quote.",
      "explanation": "Why it passes."
    }
  }
]
```
</details>

Scoring checks company fit, intent, and supporting evidence across all 20 ICPs. A winning model must beat the daily baseline mean by at least 1.0 point on the 0–100 scale. Ties and smaller gains do not promote. The gateway promotes winning code to `main` and `lab` for the next baseline. Rewards activate separately through settlement. The current champion pool starts at 25% of the subnet's emissions and decays weekly, subject to registration and continued eligibility.

## Submit a model

Use the interactive CLI:

```bash
python scripts/lab_arena_miner.py interactive \
  --wallet-name miner --hotkey-name default \
  --api-base-url https://gateway.subnet71.com
```

Enter `../my-agent` as the source directory and confirm submission, then enter your OpenRouter API key, OpenRouter management key, and Deepline key in the masked prompts. Runtime keys are stored encrypted and are never published or shared with other miners; the management key is checked and not stored.

For automation, set `OPENROUTER_API_KEY`, `OPENROUTER_MANAGEMENT_KEY`, and `DEEPLINE_API_KEY`, then replace `interactive` with `submit-model --source ../my-agent` in the command above.

Do not include keys or `.env` files in the source directory. Placeholder-only `.env.example`, `.env.sample`, and `.env.template` files are allowed. Source limits are 10 MiB compressed, 50 MiB unpacked, and 1,000 files. Track admission, scoring, per-ICP results, and champion status on the [dashboard](https://subnet71.com).

## Public input example

This real ICP comes from the September 9 set, released for evaluation on September 10: [public benchmark](https://gateway.subnet71.com/arena/v1/rounds/arena-2026-09-10/benchmark). Display-only score and position fields are omitted.

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

<details>
<summary>Auditor archive configuration</summary>

Auditors use `wss://archive.chain.opentensor.ai:443` by default. To use an
archive node you operate and trust, export the optional setting before starting
the auditor with your usual arguments:

```bash
export BITTENSOR_ARCHIVE_ENDPOINT="wss://your-archive.example:443"
python neurons/auditor_validator.py --netuid 71 --wallet.name my_wallet --wallet.hotkey default
```

`AUDITOR_BITTENSOR_ARCHIVE_ENDPOINT` is also supported. Set only one variable;
conflicting values stop startup. If both are unset or blank, the official
default remains in use. Public endpoints require `wss://`; `ws://` is accepted
only for private or loopback addresses. The node must provide historical archive
state for the configured chain.

This setting applies only to the auditor. Gateway, primary-validator, and
restart checks keep their official endpoint. Chain identity, cutover, and weight
verification remain required. No source-code patch or Git autostash is needed
for this setting. Restart the auditor after changing its environment.

</details>

## License

MIT. See [LICENSE](LICENSE).
