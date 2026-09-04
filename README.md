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

Leadpoet is a Bittensor subnet (SN71). The subnet rewards miners for improving and operating AI systems that find high-quality sales leads. Miners contribute in two tracks, the Research Lab and Fulfillment. The Research Lab includes an open agent-bundle Arena beside the existing research reward path. In Fulfillment, miners compete on real lead requests by submitting qualified leads.

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

- **Agent Competition**
- **Fulfillment**
- **Submit API Source**
- **Check API Source Submissions**

### Research Lab

Research Lab now operates the agent-bundle Arena. It creates no model changes
and runs no autoresearch or code-edit loop. Existing reward settlement remains
downstream of the Arena result.

#### Agent Bundle Arena

The Arena rebenchmarks the public baseline on the daily ICPs and evaluates
miner-submitted forks on the same ICPs. A miner can change the harness, model,
prompts, dependencies, provider use, and internal logic. The benchmark score is
the quality authority.

The public starting point is
[`leadpoet/pydantic-harness`](https://github.com/leadpoet/pydantic-harness).
Fork it or replace any part of it. The one source-code contract is:

```python
def run_icp(icp: dict) -> list[dict]:
    """Return at most five companies for one ICP."""
```

`harness.py` must expose this function. It can define the function or re-export
it from vendored code. The function must be synchronous and must have
exactly one positional parameter. It cannot have keyword-only parameters,
`*args`, or `**kwargs`. It must return JSON-ready company objects in the schema
below. The ICP dictionary contains the public business criteria, including
industry, geography, employee-count ranges, product/service, required
attributes, required intent signals, bonus intent signals, and descriptive
prompt fields. Agents must ignore fields they do not use so the host can add
descriptive fields without changing the function signature.

Each returned company must have this shape:

```json
{
  "company_name": "Example",
  "company_website": "https://example.com/",
  "company_linkedin": "https://www.linkedin.com/company/example/",
  "industry": "Software",
  "employee_count": "51-200",
  "company_stage": "Series A",
  "country": "United States",
  "state": "California",
  "fit_summary": "Why this company fits the ICP.",
  "fit_evidence_urls": ["https://example.com/about"],
  "intent_signals": [{
    "matched_icp_signal": 0,
    "description": "The required recent event.",
    "date": "2026-08-20",
    "why_now": "Why a sales representative should contact the company now.",
    "url": "https://example.com/news/event",
    "snippet": "Source text that supports the claim."
  }],
  "required_attribute": {
    "text": "The required company characteristic.",
    "passed": true,
    "evidence_url": "https://example.com/about",
    "evidence_quote": "Source text that proves the characteristic.",
    "explanation": "Why the evidence satisfies the requirement."
  }
}
```

`company_linkedin`, `company_stage`, `state`, and `required_attribute` can be
empty or omitted only where the public contract permits it. All other shown
fields are required, and `intent_signals` must contain at least one item.
Provider credentials come from the Arena host and never from a miner
submission. The public harness README contains a complete ICP example.

An agent can vendor its Python code and can include an optional
`requirements.txt`. The runner accepts normal package names and version
constraints and installs binary wheels only. URLs, local paths, nested
requirements files, VCS dependencies, and source builds are not accepted.
Miners can replace PydanticAI with any design that exposes the one Python
`run_icp` adapter. The adapter is the stable competition boundary; the
framework, model, prompts, routing, and internal logic are not admission
identities or promotion gates.

Submit the local source directory. No Dockerfile, public registry, image tag,
commit identity, receipt, or release manifest is part of miner admission:

```bash
python3 scripts/lab_arena_miner.py submit-source --source ./my-agent \
  --wallet-name default --hotkey-name default
```

Operator and bundle details: [Arena operator guide](lab_arena/RUNBOOK.md),
[input contract](lab_arena/runner.py), [output contract](lab_arena/output.py),
[provider adapter and socket protocol](lab_arena/shim.py), and
[submission helper](scripts/lab_arena_miner.py). Examples are documentation,
not admission or scoring requirements.

At the first cutoff, the Arena automatically enters the configured public
PydanticAI source archive through the same source checks, runner, provider
limits, and scorer as miner bundles. The downloaded bytes are frozen for that
round. Every new daily round downloads and scores the public baseline again;
the previous miner winner remains in reward history but is not the next day's
threshold.

#### Submit API Source

Choose **Submit API Source** in the miner menu. You only need to provide the API's public integration details:

- The source/API name and source kind.
- Its HTTPS API base URL and public documentation URL.
- The authentication type (`none`, API key, or bearer token) and rate-limit notes.
- At least one working endpoint example: `GET` or `POST`, a relative path, what it does, and an example query or JSON body.

Provenance notes and third-party reference URLs are optional. Do not submit API keys or other secrets; an operator adds any required credential after submission.

The gateway checks that the source manifest is credible, novel, and not already
in the active model. Duplicate, already-integrated, and invalid submissions
receive the same generic failure response and do not earn rewards. A submission
that passes the measured provenance precheck automatically receives **0.2% of
emissions per epoch for 20 epochs**. Leg 1 is processed FIFO and currently
allows up to 50 approvals per UTC day; separate per-hotkey anti-spam limits also
apply. Operator testing and catalog provisioning happen later and do not gate
Leg 1; only provisioned catalog sources can be used by approved product
integrations.

Choose **Check API Source Submissions** to view your own submission decisions
and Leg 1 reward state. The miner signs this read request with the same hotkey
that made the submission. The gateway returns only that hotkey's records and
safe, public reason categories. It does not show another miner's submissions,
raw validation evidence, duplicate matches, catalog contents, or sources used
by the current model. An **approved** result means that the automated Leg 1
approval boundary passed; it does not mean that the source is already in the
catalog or model.

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
| Government or education sources that do not fit another category | `other` |

Fulfillment validates the declared source against the evidence URL before
applying its score multiplier. First-party `job_board` evidence must be on the
lead company's own careers/jobs property; third-party job, news, social,
review, and reference sources must use a recognized platform. Arbitrary
third-party or self-published domains are not accepted as `other`.

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

Validators verify the canonical weight allocation and submit the resulting subnet weights.

Useful validator environment variables:

```bash
export TRUELIST_API_KEY="your_truelist_key"
export SCRAPINGDOG_API_KEY="your_scrapingdog_key"
export OPENROUTER_KEY="your_openrouter_key"
```

See [`env.example`](env.example) for the full configuration template.

## Rewards

Rewards are designed around both the Research Lab competition and Fulfillment:

- Research Lab miners submit agent bundles that are scored against the daily public baseline.
- Existing reimbursement and settlement records remain available to the weight-allocation path.
- Fulfillment rewards winning leads from client requests.
- The weekly leaderboard rewards top fulfillment performance.

Exact weights are computed from the gateway's canonical allocation bundle and current subnet policy. The validator and auditor verify and submit the same bundle.

## Transparency

The validator weight path remains attested. Validators and auditors verify the gateway bundle and use validator enclave attestation for weight submission. These controls protect the subnet weight path; they are not admission or scoring requirements for Research Lab agent bundles.

Agent submissions use the documented competition input and output contract. The competition does not require a Git identity, release manifest, receipt chain, or repository attestation from miners.

Useful tools:

```bash
python scripts/verify_attestation.py
```

For more detail, see [`scripts/VERIFICATION_GUIDE.md`](scripts/VERIFICATION_GUIDE.md).

## License

MIT. See [`LICENSE`](LICENSE).
