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

Day 0: submit models while that day's 20 ICPs stay hidden. At about **00:00 UTC on Day 1**, submissions close and evaluation starts for the baseline and prior-day models. All 20 are processed without fixed pauses between batches. Code and aggregate final scores publish when evaluation finishes. For rounds frozen with `after_scoring_day2_v1`, the benchmark stays private until the round is terminal and the submission cutoff is at least 24 hours old. Published outputs, run results, and per-ICP scores use the same boundary. Cancelled-round results and source remain private. Earlier rounds without this marker keep their original timing.

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

Each ICP execution attempt has a **five-minute wall-clock limit** and quotas of **60 OpenRouter, 30 Deepline, and 30 Scrapingdog calls**. The sandbox blocks direct network access. Keep the baseline's broker transport when changing the harness, or implement the same [broker protocol](lab_arena/shim.py) using the [approved operations](lab_arena/operations.py).

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

Scoring checks company fit, intent, and supporting evidence across all 20 ICPs. A model must beat the daily baseline score by at least 1.0 point on the 0–100 scale to qualify for promotion. The gateway promotes winning code to `main` and `lab` for the next baseline. Rewards activate separately through settlement. By default, the champion receives **25%, 20%, 15%, 10%, then 5%** of subnet emissions in successive reward weeks of 140 epochs each. The share remains at 5% from week five onward, subject to registration and continued eligibility.

The sourcing budget is **$50 across OpenRouter, Scrapingdog, and Deepline combined** for all 20 ICPs, including retries. To qualify for promotion, sourcing must also cost no more than **$0.50 per returned company**. Duplicate company domains within an ICP count once. Independent judging has a separate default **$50 allowance per submitted model**, also charged through the miner's credentials. Full-code review is an additional OpenRouter charge, recorded separately from sourcing and judging.

The gateway reserves money before calls and blocks further paid calls when the allowance is exhausted. A provider that bills after execution can exceed its reservation; the full charge still counts, and an over-budget model cannot win. Unresolved provider charges also prevent promotion.

## Submit a model

Use the interactive CLI:

```bash
python scripts/lab_arena_miner.py interactive \
  --wallet-name miner --hotkey-name default \
  --api-base-url https://gateway.subnet71.com
```

Enter `../my-agent` as the source directory and confirm submission, then enter your OpenRouter API key, OpenRouter management key, and Deepline key in the masked prompts. Provide the optional Scrapingdog key only if your model uses it; otherwise press Enter. Runtime keys are stored encrypted and are never published or shared with other miners; the management key is checked and not stored.

For automation, set `OPENROUTER_API_KEY`, `OPENROUTER_MANAGEMENT_KEY`, and `DEEPLINE_API_KEY`, optionally set `SCRAPINGDOG_API_KEY`, then replace `interactive` with `submit-model --source ../my-agent` in the command above. When provided, the sandbox exposes `SCRAPINGDOG_API_KEY` as a non-secret handle for the existing broker transport. The actual key stays in the gateway; it is never placed in model code or the validator environment.

Every miner submission must pass a full-code review before evaluation. The gateway uses **Claude Sonnet 5 through that submitting miner's OpenRouter key**, including every file in the uploaded archive and all bundled prompts. It checks for prepared answers, fabricated evidence, malicious behavior, and attempts to manipulate the reviewer. Normal constants, routing changes, and alternative harnesses are allowed. Review charges are recorded separately in the existing cost ledger. An incomplete review, unreadable file, or submission that exceeds the judge's context window cannot pass; source is never silently truncated. The submission status API reports review progress and cost.

Do not include keys or `.env` files in the source directory. Placeholder-only `.env.example`, `.env.sample`, and `.env.template` files are allowed. Source limits are 10 MiB compressed, 50 MiB unpacked, and 1,000 files. Each hotkey can have one accepted model per round; daily capacity is limited. Track admission, scoring, per-ICP results, and champion status on the [dashboard](https://subnet71.com).

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

Arena is the only subnet incentive mechanism. Research Lab reimbursements,
legacy champion obligations, SOURCE_ADD, and Fulfillment emission allocations
are retired. Fulfillment still accepts, scores, and delivers client leads.

Registered validators with a permit and at least 75,000 effective subnet stake
can receive new Arena scoring jobs. Permitted validators below that threshold
can still retrieve signed gateway weights and run the independent weight loop.
Weight retrieval requires a local-hotkey signed request; miners without a
validator permit and anonymous callers cannot retrieve the signed weight state.

Eligible normal validators score submitted models with brokered miner credentials,
returns scores through the competition API, and independently derives weights
from the signed accepted reward state and finalized chain ownership. Each
validator signs with its own local Bittensor hotkey, preserving the canonical
commit/reveal and exact transaction checks. Weight submission runs independently
of scoring. Chain outcomes are recorded separately. No Nitro enclave or KMS
provisioning is required for validators, and there is no audit-validator role.

Follow [the normal Arena validator setup](docs/arena_normal_validator_weights.md)
for local wallet configuration, sandbox setup, restart, and verification.


## License

MIT. See [LICENSE](LICENSE).
