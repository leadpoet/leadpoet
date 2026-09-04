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
documented by the public harness. Provider credentials come from the Arena
host and never from a miner submission.

An agent can vendor its Python code and can include an optional
`requirements.txt`. The runner accepts normal package names and version
constraints and installs binary wheels only. URLs, local paths, nested
requirements files, VCS dependencies, and source builds are not accepted.
This lets miners use PydanticAI, Pi, Codex, or another Python agent design
without submitting a container image.

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
Leg 1; only provisioned catalog sources can be used by improvement loops.

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
