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

Leadpoet is Bittensor subnet 71. Miners improve an open sales-research agent. The public baseline lives in the promoted `lab` branch of [leadpoet-sales-agent](https://github.com/leadpoet/leadpoet-sales-agent).

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
git clone --branch lab https://github.com/leadpoet/leadpoet-sales-agent.git ../my-agent
```

In `harness.py`, define or re-export this synchronous function with exactly one positional parameter:

```python
def run_icp(icp: dict) -> list[dict]:
    """Return at most five companies."""
```

Keep `LICENSE` beside `harness.py`. It must contain the complete
[Tyche AGPL-3.0 license](https://github.com/gzaentz/tyche/blob/main/LICENSE).
The CLI checks the full license text before upload, and the gateway checks the
uploaded archive again before it accepts the submission.

Change the harness, model, prompts, and approved API routing. List Python dependencies in `requirements.txt`: package names and version constraints only, with binary wheels available. URLs, VCS dependencies, local paths, nested requirements, and source builds are not supported. Return at most five companies as a JSON list. Use `[]` if there are no valid matches.

Each new ICP execution attempt has a **45-minute wall-clock limit** and quotas of **200 OpenRouter, 30 Deepline, and 30 Scrapingdog calls**. Historical rounds keep the exact quotas frozen in their round configuration. The sandbox blocks direct network access. Keep the baseline's broker transport when changing the harness, or implement the same [broker protocol](lab_arena/shim.py) using the [approved operations](lab_arena/operations.py).

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

Rounds that announce `intent_details_policy: "intent_details_v1"` use output
schema v6 for current company-only rounds (v5 for historical contact rounds). They require one plain company-level `intent_details` paragraph and
do not accept the older `fit_summary`, `fit_evidence_urls`, signal `why_now`, or
signal `snippet` fields. The scorer verifies the paragraph against the
independently verified signals before the company can receive credit.

The current competition discovers companies/accounts only. Contacts, roles, emails, and contact verification are not required and do not affect scoring. Optional legacy contact data is ignored. Historical contact rounds retain their frozen output and verification contract; see the [historical contact contract](docs/arena-contacts.md).

Rounds that announce `company_quality_policy: "company_quality_v1"` require a matching company LinkedIn URL and the headquarters state for U.S. companies. They share individual verification judgments and account for the fraction of requested companies that qualify within each buyer request. Missing new details zero only the affected company. See [company quality, scoring and activation](docs/arena-company-quality.md).

Before submission, `/arena/v1/current` (`open_round`) and
`/arena/v1/rounds/{round_id}` expose the round's pinned
`output_schema_version` and its enabled policy markers. An absent marker means
that policy is disabled for the round. The version matches the schema used to
validate model output; historical rounds continue to report their frozen
version.

Your `harness.run_icp(icp)` input also includes `output_schema_version`; use this round-specific version and its policy markers when constructing company output.

Validators must update their runner to support company-only output v6. Older runners receive `validator_output_schema_upgrade_required` before a lease is allocated; this does not consume an ICP attempt or affect weight submission.

Scoring checks company fit, intent, and supporting evidence across all configured ICPs (currently 10). A model must beat the daily baseline score by at least 0.5 points on the 0–100 scale to qualify for promotion. The gateway promotes winning code to `main` and `lab` for the next baseline. Rewards activate separately through settlement. By default, the champion receives **30%, 24%, 18%, 12%, then 6%** of subnet emissions in successive reward weeks of 140 epochs each. The share remains at 6% from week five onward, subject to registration and continued eligibility. Existing signed reward bases keep their recorded pool percentage.

**Champion miners must keep their submitted API credentials funded; if Leadpoet must fund a champion rebenchmark via fallback credentials, that period’s champion incentive is reduced by 50%.**

Each ICP has a **$4 sourcing limit across OpenRouter, Scrapingdog, and Deepline combined**, including retries. An admitted call can finish above the limit; later paid sourcing calls then stop for that ICP. Only confirmed charges consume admission budget. Pending bills and temporary monetary holds do not block research.

After company verification, each ICP can spend up to **$0.80 per verified, qualified company** on successful sourcing calls. Duplicate or unqualified companies add no allowance. An ICP over that cost allowance contributes zero; other ICPs still count. Independent judging has a separate default **$50 allowance per submitted model**, using the same credential selection but excluded from sourcing spend and eligibility. Full-code review is also recorded separately. Historical rounds retain their frozen cost rules. See the [cost accounting details](docs/arena-score-integrity.md).

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

Temporary provider or transport failures may retry with increasing delays, up to six review attempts before the round's existing benchmark deadline; credential, credit, and invalid-source failures do not retry. Other incomplete responses retain the three-attempt limit. If review cannot pass, the admitted submission stays visible on the dashboard as not evaluated, with no score or published source.

Do not include keys or `.env` files in the source directory. Placeholder-only `.env.example`, `.env.sample`, and `.env.template` files are allowed. `LICENSE.txt` or `LICENSE.md` can replace `LICENSE`, but it must stay beside `harness.py` and contain the same complete AGPL-3.0 text. Source limits are 10 MiB compressed, 50 MiB unpacked, and 1,000 files.

Each hotkey can have one accepted model per daily round. To replace your queued
model, run the same submission command once more with the same hotkey before the public
`submission_replacement_cutoff`: **23:00 UTC**, one hour before the next 00:00 UTC
round boundary. Upload and final source, license, and credential validation must
finish before that cutoff. Replacement is refused at or after 23:00 UTC, or
once evaluation starts. The previous accepted model remains selected if these
checks fail or finalization does not finish in time. Each revision has its own
source archive; prior review charges and replacement links remain in the audit
history. New code reviews start only after replacement closes. The selected
revision must still pass review before evaluation.

Only one replacement attempt is allowed per hotkey per daily round: reserving its upload uses the allowance even if the upload is abandoned or validation fails, and the last accepted model remains selected if the replacement fails.

For example, a queued model submitted at 09:00 UTC on December 9 can be replaced using the same submission command and hotkey with updated source, as long as upload and validation finish before 23:00 UTC that day.

First submissions retain the normal 00:00 UTC submission deadline. Hotkeys
under the same coldkey may each submit. Each round admits up to 20 challengers,
plus the baseline. Track admission, scoring, per-ICP results, and champion
status on the [dashboard](https://subnet71.com).

Once the round's ICPs are public, open a submission and expand an ICP to see each company's recorded checks, including company-fit, intent, failed checks, and checks that were not evaluated.

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

Arena is the only subnet incentive and work-allocation mechanism. Research Lab
reimbursements, legacy champion obligations, and SOURCE_ADD are retired.

Registered validators with a permit and at least 75,000 effective subnet stake
can receive new Arena scoring jobs. Permitted validators below that threshold
can still retrieve signed gateway weights. Recent Arena work is not required
at any stake level. Validators can retrieve weights when no jobs are available
or their scoring process is unavailable.
Weight retrieval requires a local-hotkey signed request; miners without a
validator permit and anonymous callers cannot retrieve the signed weight state.

Eligible normal validators score submitted models with brokered miner credentials,
returns scores through the competition API, and independently derives weights
from the signed accepted reward state and finalized chain ownership. Each
validator signs with its own local Bittensor hotkey, preserving the canonical
commit/reveal and exact transaction checks. Scoring and weights use separate
loops; scoring failures do not stop weight submission.
Already-signed transaction recovery and pending reveals continue. Chain outcomes are recorded separately. No Nitro enclave or KMS
provisioning is required for validators, and there is no audit-validator role.

Run `python neurons/validator.py` with your existing wallet flags. The same
validator handles scoring startup through existing non-interactive sudo
permission while preserving your wallet and state paths. It does not grant
host permissions or stop weights when scoring cannot start.
Follow [the normal Arena validator setup](docs/arena_normal_validator_weights.md)
for local wallet configuration, sandbox setup, restart, and verification.
Configure your own proxies using the [Webshare setup guide](docs/arena_parallel_icps.md#webshare-setup-for-validators).
Use `--check-scoring-only` to verify host setup, proxy connections, distinct
exit IPs and memory capacity without accessing wallets or the chain;
the separate installed-runtime probe verifies sandbox
execution. The normal `--check-only` command verifies wallet and chain readiness.


## License

Copyright (c) 2025 Leadpoet.

Licensed under the GNU Affero General Public License, version 3 only
(`AGPL-3.0-only`). See [LICENSE](LICENSE).

Inherited third-party code retains its original copyright and license notices.
