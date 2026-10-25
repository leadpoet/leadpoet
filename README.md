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

# Leadpoet - the first open research lab advancing AI models for sales reasoning.

Leadpoet is Bittensor subnet 71. Miners improve an open sales-research agent. The public baseline lives in the promoted `lab` branch of [leadpoet-sales-agent](https://github.com/leadpoet/leadpoet-sales-agent).

## Daily schedule

Submit during the day before **00:00 UTC**, when submissions close and evaluation starts. The benchmark stays private during evaluation. Results and source publish according to the round's disclosure schedule. Check the [dashboard](https://subnet71.com) for current deadlines and results.

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
git clone --branch lab https://github.com/leadpoet/leadpoet-sales-agent.git ../my-agent
```

In `harness.py`, define or re-export this synchronous function with exactly one positional parameter:

```python
def run_icp(icp: dict) -> list[dict]:
    """Return at most five companies."""
```

Keep the complete [Tyche AGPL-3.0 license](https://github.com/gzaentz/tyche/blob/main/LICENSE) beside `harness.py` as `LICENSE`, `LICENSE.txt`, or `LICENSE.md`.

You can change the harness, model, prompts, and tools. Put package names and version constraints in `requirements.txt`; dependencies must have binary wheels. URLs, VCS dependencies, local paths, nested requirements, and source builds are not supported.

The sandbox blocks direct network access. Keep the baseline's broker transport for approved OpenRouter, Scrapingdog, and Deepline calls. See the [miner guide](docs/miner-model-submissions.md) for API access and limits.

**OpenRouter must allow [Zero Data Retention (ZDR)](https://openrouter.ai/docs/guides/features/zdr) endpoints for your chosen model.** Calls fail if your account blocks the required provider. Keep your API keys funded.

### Upcoming thinking-model whitelist

**Draft policy, not active in current rounds.** When enabled, agents and subagents must use one of these routes:

| OpenRouter model | Provider | API |
| --- | --- | --- |
| `openai/gpt-oss-120b` | `deepinfra/bf16` | Chat Completions |
| `openai/gpt-oss-20b` | `deepinfra/bf16` | Chat Completions |
| `mistralai/mistral-large-4-0` | `mistral/zdr` | Chat Completions |
| `deepseek/deepseek-v4.1-flash-1017` | `fireworks` | Chat Completions or Responses, including Codex |
| `reflection/beam-1.0-501b-a23b` | `nebius/fp8` | Chat Completions |
| `google/gemma-4-31b-it` | `coreweave/fp4` | Chat Completions |

Allow the listed provider in your [OpenRouter settings](https://openrouter.ai/settings/privacy). Arena uses that route with ZDR and no provider fallback. For local tests, enable thinking with:

- Chat Completions: `"reasoning": {"effort": "high", "exclude": false}`.
- Gemma: `"reasoning": {"enabled": true, "exclude": false}` instead.
- DeepSeek Responses: `"reasoning": {"effort": "high"}`.

Keep normal conversation history and tool-call IDs intact. **Keep your harness and custom tools. No custom logging or receipts are required.** These rounds require model-directed research and full provider-exposed thinking; summaries alone do not qualify. The sourcing cap increases to **$6 per ICP**. Champion eligibility requires a passing review and usable training data for at least **80% of ICPs**.

Return a JSON list of up to five companies, or `[]` if none qualify. Use the `output_schema_version` and policy settings provided in the ICP input. Current rounds source companies only, with no contact details required. See [output requirements](docs/arena-company-quality.md) for required fields.

This example shows the output format. `matched_icp_signal` is the zero-based index in the input's `intent_signals` list. Example URLs and claims must be replaced with real evidence.

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
    "intent_details": "Example launched its workflow platform on August 20, 2026. The launch could create more implementation work as customers adopt the platform. This activity makes Example relevant to the ICP for business software companies expanding their workflow offering.",
    "intent_signals": [
      {
        "matched_icp_signal": 0,
        "description": "Example launched its workflow platform.",
        "date": "2026-08-20",
        "url": "https://example.com/news"
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

Scoring checks company fit, intent, and supporting evidence. The highest eligible score must beat the daily baseline by **at least 8 points** to become champion in new rounds. Winning code becomes the next baseline.

Current reward tiers are cumulative:

| Improvement over baseline | Total emissions before decay |
| --- | --- |
| +8 points | 15% |
| +15 points | 30% |
| +20 points | 50% |

Slots remain with their holder until replaced. Each slot halves every 140 epochs, reaching a 1/16 floor after four periods. A new win resets only the slots it replaces. See [reward rules](docs/arena_reward_slots.md) for activation and historical policies.

**Champion miners must keep their submitted API credentials funded. If Leadpoet funds a champion rebenchmark, that period's champion incentive is reduced by 50%.**

Current rounds have a **$4 sourcing cap per ICP**, across OpenRouter, Scrapingdog, and Deepline, including retries. Only confirmed charges count toward admission. An ICP above the successful-sourcing cost limit contributes zero; other ICPs still count. Judging has a separate default **$50 allowance per submitted model**, and code review is billed separately. Historical rounds retain their original rules. See [cost details](docs/arena-score-integrity.md).

## Submit a model

Use the interactive CLI:

```bash
python scripts/lab_arena_miner.py interactive \
  --wallet-name miner --hotkey-name default \
  --api-base-url https://gateway.subnet71.com
```

Enter `../my-agent` as the source directory and confirm submission, then enter your OpenRouter API key, OpenRouter management key, and Deepline key in the masked prompts. Provide the optional Scrapingdog key only if your model uses it; otherwise press Enter. Runtime keys are stored encrypted and are never published or shared with other miners; the management key is checked and not stored.

For automation, set `OPENROUTER_API_KEY`, `OPENROUTER_MANAGEMENT_KEY`, and `DEEPLINE_API_KEY`, plus `SCRAPINGDOG_API_KEY` if used. Replace `interactive` with `submit-model --source ../my-agent` in the command above.

Every submission must pass code review before evaluation. Review uses the submitting miner's OpenRouter key and is charged separately. Keep source and bundled prompts readable. Track review status and cost on the dashboard.

Do not include keys or `.env` files in the source directory. Placeholder-only `.env.example`, `.env.sample`, and `.env.template` files are allowed. Source limits are **10 MiB compressed, 50 MiB unpacked, and 1,000 files**.

Each registered hotkey can submit one model per daily round. First submissions close at **00:00 UTC**. You have **one replacement attempt before 23:00 UTC** using the same command and hotkey. Upload and validation must finish before the cutoff. An abandoned or failed replacement still uses that attempt, but the last accepted model remains selected.

Track submissions, scores, and champion status on the [dashboard](https://subnet71.com). See the [miner guide](docs/miner-model-submissions.md) for detailed submission rules.

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
