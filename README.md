<h1 align="center">Leadpoet</h1>

<p align="center">
  <strong>AI sales research and fulfillment, evaluated on Bittensor.</strong>
</p>

<p align="center">
  <a href="https://discord.gg/tMcmbPKvz"><img alt="Discord" src="https://img.shields.io/badge/Discord-Join-5865F2?style=flat-square"></a>
  <a href="https://subnet71.com"><img alt="Leaderboard" src="https://img.shields.io/badge/Leaderboard-subnet71.com-d6ad5f?style=flat-square"></a>
  <a href="https://leadpoet.com"><img alt="Website" src="https://img.shields.io/badge/Website-leadpoet.com-111827?style=flat-square"></a>
  <a href="https://x.com/subnet71"><img alt="Subnet X" src="https://img.shields.io/badge/X-@subnet71-000000?style=flat-square"></a>
  <a href="https://x.com/LeadpoetAI"><img alt="Leadpoet X" src="https://img.shields.io/badge/X-@LeadpoetAI-000000?style=flat-square"></a>
  <a href="https://www.linkedin.com/company/leadpoet/"><img alt="LinkedIn" src="https://img.shields.io/badge/LinkedIn-Leadpoet-0A66C2?style=flat-square"></a>
</p>

---

Leadpoet is Bittensor Subnet 71. The subnet rewards miners for improving and operating AI systems that find high-quality sales leads.

The network currently has two production tracks:

- **Research Lab** - miners fund hosted auto-research loops that try to improve Leadpoet's private sourcing model.
- **Fulfillment** - miners compete on real client requests by submitting enriched leads that match a specific ICP.

## Dashboard

Use the public dashboard to track:

- Current private-model benchmark score.
- Public ICP benchmark examples and scores.
- Recent Research Lab activity.
- Fulfillment activity and leaderboard.

Dashboard: [subnet71.com](https://subnet71.com)

## Installation

```bash
git clone https://github.com/leadpoet/leadpoet.git
cd leadpoet

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -e .
```

Requirements:

- Python 3.9 to 3.12
- Bittensor wallet
- Bittensor CLI

```bash
pip install "bittensor>=9.10"
btcli wallet create
```

## Miners

Register on subnet 71:

```bash
btcli subnet register \
  --netuid 71 \
  --subtensor.network finney \
  --wallet.name miner \
  --wallet.hotkey default
```

Run the miner:

```bash
python neurons/miner.py \
  --wallet_name miner \
  --wallet_hotkey default \
  --netuid 71 \
  --subtensor_network finney
```

The miner will ask which mode to run:

- **Auto Research** - default Research Lab mode.
- **Fulfillment** - client-request lead fulfillment mode.

### Research Lab

Research Lab lets miners contribute compute toward improving Leadpoet's private sourcing model.

Current flow:

1. The miner provides an OpenRouter key.
2. The miner enters an optional research focus.
3. The miner pays the loop-start fee in TAO.
4. The gateway runs the hosted auto-research loop.
5. Candidate improvements are scored.
6. Rewards are assigned to miners.

### Fulfillment

Fulfillment miners compete on real client requests. A client publishes an ICP, miners submit enriched leads, and validators score each lead for fit, accuracy, and intent evidence.

High-level flow:

1. Client request is published.
2. Miners commit hashed leads during the commit window.
3. Miners reveal full lead data during the reveal window.
4. Validators score revealed leads.
5. Winning leads earn emissions over the reward runway.

Fulfillment leads should include:

- Contact name, email, LinkedIn, title, role type, seniority, and location.
- Company name, website, LinkedIn, industry, sub-industry, size, and HQ location.
- A clear company description.
- Intent evidence with source, URL, date, snippet, and matched ICP signal.
- Optional attribute evidence for required client constraints.

Common rejection causes:

- Role, seniority, industry, geography, or employee-count mismatch.
- Invalid or unverifiable email.
- Weak company description.
- Missing required intent signal.
- Intent snippet not present on the cited page.
- Wrong `source` for an intent URL.

Use the correct intent source:

| URL type | `source` |
| --- | --- |
| Company website or blog | `company_website` |
| Lever, Greenhouse, Indeed, careers pages | `job_board` |
| Press releases and news articles | `news` |
| LinkedIn pages, posts, jobs | `linkedin` |
| X, Threads, Instagram, Facebook, TikTok | `social_media` |
| GitHub repositories or organizations | `github` |
| G2, Capterra, TrustRadius, Glassdoor, Trustpilot | `review_site` |
| Wikipedia | `wikipedia` |
| Other verifiable dated events | `other` |

Reference fulfillment code lives in `miner_models/Main_fulfillment_model/`. It is a starting point, not a guaranteed competitive miner.

## Validators

Register and run a validator on subnet 71:

```bash
btcli subnet register \
  --netuid 71 \
  --subtensor.network finney \
  --wallet.name validator \
  --wallet.hotkey default
```

```bash
python neurons/validator.py \
  --wallet_name validator \
  --wallet_hotkey default \
  --netuid 71 \
  --subtensor_network finney
```

Validators verify Research Lab receipts, evaluation bundles, fulfillment scoring, and final weight allocation.

Useful validator environment variables:

```bash
export TRUELIST_API_KEY="your_truelist_key"
export SCRAPINGDOG_API_KEY="your_scrapingdog_key"
export OPENROUTER_KEY="your_openrouter_key"
export COMPANIES_HOUSE_API_KEY="your_companies_house_key"
```

See [`env.example`](env.example) for a fuller configuration template.

## Rewards

The active production split is designed around both Research Lab and Fulfillment:

- Research Lab has its own allocation for compute reimbursement and successful model improvements.
- Fulfillment rewards winning leads from client requests.
- The weekly leaderboard rewards top fulfillment performance.

Exact weights are computed by the validator from signed gateway bundles and current subnet policy.

## Transparency

Leadpoet uses signed gateway artifacts and validator-side verification for Research Lab and Fulfillment outputs.

Useful tools:

```bash
python scripts/verify_attestation.py
python scripts/decompress_arweave_checkpoint.py
```

For more detail, see [`scripts/VERIFICATION_GUIDE.md`](scripts/VERIFICATION_GUIDE.md).

## Support

- Email: hello@leadpoet.com

## License

MIT. See [`LICENSE`](LICENSE).
