<h1 align="center">Leadpoet</h1>

<p align="center">
  <strong>AI sales intelliegence, built on Bittensor.</strong>
</p>

<p align="center">
  <a href="https://discord.gg/tMcmbPKvz"><img alt="Discord" src="https://img.shields.io/badge/Discord-Join-5865F2?style=flat-square"></a>
  <a href="https://subnet71.com"><img alt="Dashboard" src="https://img.shields.io/badge/Leaderboard-subnet71.com-e8c76d?style=flat-square"></a>
  <a href="https://leadpoet.com"><img alt="Website" src="https://img.shields.io/badge/Website-leadpoet.com-f3f4f6?style=flat-square"></a>
  <a href="https://x.com/subnet71"><img alt="Subnet X" src="https://img.shields.io/badge/X-@subnet71-000000?style=flat-square"></a>
  <a href="https://x.com/LeadpoetAI"><img alt="Leadpoet X" src="https://img.shields.io/badge/X-@LeadpoetAI-000000?style=flat-square"></a>
</p>


---

Leadpoet is a Bittensor subnet (SN71). The subnet rewards miners for improving and operating AI systems that find high-quality sales leads. Miners contribute in two tracks, the Research Lab and Fulfillment. In the Research Lab, miners direct research and compute through auto-research loops that try to improve an AI sales agent. In Fulfillment, miners compete on real lead requests by submitting qualified leads.

## Dashboard

Use the dashboard to track:

- Research Lab agent benchmark examples and scores, areas to improve, and activity.
- Fulfillment activity and leaderboard.

Dashboard: [subnet71.com](https://subnet71.com)

## Installation

```bash
git clone https://github.com/leadpoet/leadpoet.git
cd leadpoet

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Requirements:

- Python 3.9 or 3.10 recommended
- Bittensor wallet
- Bittensor CLI

```bash
pip install "bittensor==10.5.0" "bittensor-cli>=1.0.0"
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

- **Auto Research**
- **Fulfillment**

### Research Lab

Research Lab lets miners contribute direction and compute toward improving the AI sales agent.

The current sourcing model and sealed benchmark are not published in full. Keeping them private helps prevent benchmark overfitting, fixture memorization, and leakage of evaluation data while still letting miners submit and evaluate improvements through the Research Lab flow.

How it works:

1. The miner securely provides an OpenRouter API key and management key.
2. The miner enters a research direction.
3. The miner pays the loop-start fee in TAO to cover benchmark costs.
4. The TEE gateway runs the auto-research loop.
5. Candidate improvements are scored.
6. Miners are provided with their respective rewards.

Research Lab rewards result from provided compute or model improvements:

- Compute reimbursement covers a portion of verified compute spend from research loops based on the amount of participation.
- Verified model improvements result in substantial rewards when a candidate improvement beats the current model benchmark.

At a high level, rewards are calculated per reward epoch:

```text
lab_allocation = subnet_emissions * research_lab_allocation_rate
compute_credit = verified_openrouter_spend * participation_multiplier
improvement_credit = max(0, candidate_score - benchmark_score - improvement_threshold)
```

If there are no winning improvements, the Research Lab allocation is split across participating miners by compute credit:

```text
miner_reward = lab_allocation * miner_compute_credit / total_compute_credit
```

If there are winning improvements, verified compute reimbursement is paid first, capped by the miner's remaining verified spend. If reimbursement demand is larger than available reimbursement capacity, reimbursements are prorated by compute credit. If reimbursement demand is smaller than available capacity, the unused amount flows to winning improvements.

```text
reimbursement_reward = min(remaining_verified_spend, reimbursement_capacity * miner_compute_credit / total_compute_credit)
winner_reward = remaining_lab_allocation * winner_improvement_credit / total_winner_improvement_credit
```

The reward records tie miner hotkeys to the run, candidate, verified spend, benchmark result, and validator weight input. The validator and verifier code contain the replayable reward and weight checks.

### Research Runtime

Research Lab runs daily rebenchmarks and candidate scoring against the current model runtime image listed in the `current.json` manifest. Hosted auto-research builds start from that same image, which gives every candidate the same baseline before changes are tested.

For each candidate, the gateway extracts the runtime app from `/app` and gives the research model limited access to inspect that extracted source. The model can read the code it needs in order to suggest an improvement, but it does not get general repository access, shell access, credentials, deployment access, or access to files outside the runtime scope.

After the model proposes a patch, the system checks that the patch only touches files the model actually read during that loop and only within the allowed edit paths. If the patch passes those checks, it is applied to the extracted runtime source, rebuilt into a candidate image, and evaluated through the normal scoring and benchmarking pipeline.

The research model itself is not public to prevent miners from overfitting to the model's suggestions, prompts, hidden assumptions, or internal scoring preferences. Miners compete on the measured quality of their submitted candidates, not on reverse-engineering the improvement model.

The TEE runs the non-public improvement model in an isolated environment while recording the runtime image, inspected files, proposed patch, scope checks, rebuilt candidate, and evaluation result.

Candidate patches are limited to the extracted runtime code paths inside the runtime image:

- `gateway/`
- `qualification/`
- `sourcing_model/`
- `validator_models/`
- `research_lab_adapter.py`

`requirements.txt` may be used as a build/runtime dependency when present in the image, but it is not an editable candidate target. New top-level folders, Dockerfiles, dependency files, CI files, deployment scripts, lockfiles, env files, and credential handling are outside the Research Lab edit scope.

These are runtime image paths, so some entries may not exist as top-level folders in this repository.

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
```

See [`env.example`](env.example) for the full configuration template.

## Rewards

Rewards are designed around both Research Lab and Fulfillment:

- Research Lab miners can earn reimbursement-style emissions for verified compute they provide.
- Research Lab miners that produce benchmarked model improvements can earn larger improvement rewards.
- Fulfillment rewards winning leads from client requests.
- The weekly leaderboard rewards top fulfillment performance.

Exact weights are computed by validators from signed gateway bundles, verified compute records, benchmark results, allocation records, and current subnet policy. Research Lab reward calculations can be independently checked from the emitted receipts, signed audit logs, and Arweave-anchored checkpoints.

## Transparency

Leadpoet uses a gateway TEE for Research Lab and Fulfillment outputs. The gateway enclave signs receipts, scoring bundles, allocation records, and compact audit anchors with an enclave-held signing key.

The gateway attestation binds the enclave public key to the gateway runtime measurement. Validators and auditors verify the Nitro attestation, verify enclave signatures before treating signed artifacts as gateway outputs, and verify validator weight submissions by matching the validator's attested PCR0 to an independently rebuilt validator enclave PCR0 from the same repository commit.

Audit artifacts include the hashes, status transitions, signatures, and reward inputs needed to check validator behavior. They do not expose model code, hidden ICPs, provider secrets, raw private data, or candidate patch internals.

Arweave checkpoints anchor the signed artifact hashes and status transitions used in reward calculations. Auditors can match checkpoint data to signed gateway artifacts, verify enclave signatures and attestation, recompute reward inputs, and compare validator weights against the published policy.

Useful tools:

```bash
python scripts/verify_attestation.py
python scripts/decompress_arweave_checkpoint.py
```

For more detail, see [`scripts/VERIFICATION_GUIDE.md`](scripts/VERIFICATION_GUIDE.md).

## License

MIT. See [`LICENSE`](LICENSE).
