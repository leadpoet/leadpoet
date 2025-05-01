# LeadPoet | Premium Sales Leads Powered by Bittensor.

Welcome to LeadPoet, a decentralized lead generation subnet built on Bittensor, designed for SaaS, finance, healthcare, e-commerce, and B2B agencies seeking high-quality, conversion-ready sales leads.

## Overview

LeadPoet leverages Bittensor's decentralized architecture to create a scalable marketplace for lead generation. Miners generate lead batches, validators ensure quality through rigorous auditing, and buyers access curated, real-time leads optimized for conversion. This eliminates reliance on static databases, ensuring fresher, more relevant leads at lower costs.

- **Miners**: Generate lead batches in response to buyer queries using their lead generation model (example model provided in miner_models/get_leads.py).
- **Validators**: Audit lead quality using their own model or validator_models/os_validator_model.py.
- **Buyers**: Request leads via the Leadpoet/api/leadpoet_api.py CLI or the leadpoet.com website, receiving validated leads with fields like Email and LinkedIn Pages.

**Data Flow**: Buyers submit a query → Miners generate leads → Validators audit and score (≥90% to pass) → Approved leads undergo automated checks (≥90% to pass) → Highest-scoring leads are delivered to the buyer.

**Token**: TAO is used for staking, rewards, and (future) lead purchases.

## Getting Started

### Prerequisites

- **Hardware**: 16GB RAM, 4-core CPU, 100GB SSD.
- **Software**: Python 3.8+, Bittensor CLI (pip install bittensor>=6.9.3).
- **TAO Wallet**: Required for staking and rewards. Create with btcli wallet create.
- **API Keys (Optional for Miners)**: For real lead generation using Hunter.io and Clearbit, set:
  
```bash
export HUNTER_API_KEY=your_hunter_api_key
export CLEARBIT_API_KEY=your_clearbit_api_key
```

### Installation

**Clone the Repository**:
```bash
git clone https://github.com/Pranav-create/Leadpoet.git
cd Leadpoet
```

**Install the LeadPoet Package**:
```bash
pip install .
```

This installs dependencies listed in setup.py, including bittensor, requests, numpy, and others.

**Configure Your Wallet**:
1. Create a wallet if you haven't: `btcli wallet create`.
2. Stake TAO as needed (see participation requirements below).
3. Verify wallet setup: `btcli wallet overview --wallet.name <your_wallet> --wallet.hotkey <your_hotkey>`.

**Set API Keys (Optional for Miners)**:
To enable real lead generation in miner_models/get_leads.py:
```bash
export HUNTER_API_KEY=your_hunter_api_key
export CLEARBIT_API_KEY=your_clearbit_api_key
```

## For Miners

### Participation Requirements
- **Stake**: Minimum 2 TAO.
- **Register**: Register on the subnet (netuid 343):
```bash
btcli stake --amount 2 --wallet.name <your_wallet> --wallet.hotkey <your_hotkey>
btcli subnet register --netuid 343 --subtensor.network test --wallet.name <your_wallet> --wallet.hotkey <your_hotkey>
```

### Running a Miner
Run your miner to generate leads:
```bash
python neurons/miner.py --netuid 343 --subtensor.network test --wallet.name miner --wallet.hotkey default --logging.debug
```

- Add `--use_open_source_lead_model` to use get_leads.py with Hunter.io/Clearbit (requires API keys).
- Add `--mock` for local testing with dummy data.
- Add `--logging_trace` for detailed logs.

**Behavior**:
- Responds to validator/buyer queries within 2 minutes + 2 seconds per lead.
- Generates leads in JSON format (see below).
- In mock mode, uses MockDendrite and MockSubtensor to simulate responses.

### Lead Format
Leads follow this JSON structure:
```json
{
    "Business": "Octiv",
    "Owner Full name": "Jeff Romero",
    "First": "Jeff",
    "Last": "Romero",
    "Owner(s) Email": "jeff@octivdigital.com",
    "LinkedIn": "https://www.linkedin.com/in/jeffromero/",
    "Website": "https://www.octivdigital.com/",
    "Industry": "Tech & AI",
    "Region": "US"
}
```

### Miner Incentives
Miners are rewarded based on lead quality and reliability:

**Accuracy Score (A_m)**:
- Calculated over time with weighted periods:
  - Recent (previous 14 days): 55% weight
  - Mid-term (previous 30 days): 25% weight
  - Long-term (previous 90 days): 20% weight

**Dependability (D_m)**:
- Based on uptime (queries responded to with approved leads / total queries).
- Weighted using the same time periods as accuracy.

**Weighted Score (W_m)**: W_m = A_m × D_m.

**Rewards (M_m)**:
- Total emissions M split proportionally.
- Formula: M_m = M × (W_m / Σ W_m) for miners with A_m > 0.5.

**Example**:
- Miner A: 450 good leads, 90% uptime (D_A = 1.9), W_A = 450 × 1.9 = 855.
- Miner B: 450 good leads, 50% uptime (D_B = 1.5), W_B = 450 × 1.5 = 675.
- Total W = 1530. Rewards: A gets ~559 TAO, B gets ~441 TAO.

**Best Practices**:
- Ensure accurate contact data (emails, LinkedIn URLs).
- Align leads with requested industry/region.
- Maintain high uptime to boost D_m.

## For Validators

### Participation Requirements
- **Stake**: Minimum 20 TAO.
- **Register**:
```bash
btcli stake --amount 20 --wallet.name <your_wallet> --wallet.hotkey <your_hotkey>
btcli subnet register --netuid 343 --subtensor.network test --wallet.name <your_wallet> --wallet.hotkey <your_hotkey>
```

### Running a Validator
Run your validator to audit leads:
```bash
python neurons/validator.py --netuid 343 --subtensor.network test --wallet.name validator --wallet.hotkey default --logging.debug
```

- Add `--use_open_source_validator_model` to use os_validator_model.py.
- Add `--mock` for local testing.
- Add `--logging_trace` for detailed logs.

**Behavior**:
- Queries miners for lead batches (100 leads by default).
- Validates ~20% of each batch within 2 minutes using os_validator_model.py.
- Runs automated checks with automated_checks.py.
- Assigns scores (0–100%) and updates miner weights.

### Validation Process
**Audit**:
- Sample size: 10–50% of leads, based on miner accuracy (e.g., 20/100 for mid-range miners).
- Checks:
  - Format: Email regex (^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$), no duplicates.
  - Accuracy: No disposable domains, basic domain/website reachability.
  - Relevance: Matches industry/region filters.
- Outcome: Approves if ≥90% of sampled leads are valid; rejects otherwise.

**Automated Checks**:
- Post-validation using automated_checks.py.
- Verifies email existence (Hunter.io in non-mock mode) and company website accessibility.
- Approves if ≥90% pass; otherwise, validator reputation decreases (-20 points).

### Validator Incentives
Validators are rewarded based on accuracy and consistency:

**Precision (P_v)**:
- Reflects validator accuracy over time.
- Adjustments:
  - Correct validation: +10 (O_v within 10% of final score)
  - Incorrect validation: -15 (if invalid leads slip through)
  - Buyer feedback: up to +15 or -25

**Consistency (C_v)**:
- Reflects how often validator score O_v is close to final F.
- Weighted with:
  - Recent (previous 14 days): 55% weight
  - Mid-term (previous 30 days): 25% weight
  - Long-term (previous 90 days): 20% weight

**Weighted Reputation (R_v)**: R_v = P_v × C_v × F_v.
- Where F_v = 1 unless the validator is flagged for collusion (then F_v = 0)

**Final score (F)**:
- Each validator's score O_v is weighted by their reputation R_v.
- F = ∑[O_v × (R_v / Rs_total)] where Rs_total = ∑R_v across all validators v with R_v > 15.
- The lead list with the highest final score is sent to the buyer.

**Rewards (V_v)**:
- Total emissions V split proportionally.
- Formula: V_v = V × (R_v / ∑R_v) for validators with R_v > 15.

**Trusted Validators**:
- R_v ≥ 100 and ≥1 month of operation.
- If >67% reject a batch, it's denied (even if >50% approve).

**Best Practices**:
- Maintain high scoring precision to avoid reputation penalties.
- Ensure consistency with the final score F.
- Monitor logs for validation failures.
  
## For Buyers

### Participation Requirements
- **Stake**: Minimum 50 TAO.
```bash
btcli stake --amount 50 --wallet.name <your_wallet> --wallet.hotkey <your_hotkey>
```

### Accessing Leads
Buyers request leads via the CLI API, receiving validated batches:
```bash
python leadpoet-api --netuid 343  --subtensor.network test --wallet.name buyer --wallet.hotkey default --logging.debug
```

- Add `--mock` for local testing.
- Add `--logging_trace` for detailed logs.

**Interactive CLI**:
1. Input number of leads (1–100).
2. Select industry (Tech & AI, Finance & Fintech, Health & Wellness, Media & Education, Energy & Industry) or skip.
3. Specify region (e.g., US) or skip.
4. Receive leads with Owner(s) Email, LinkedIn, etc.

**Example Output**:
```json
[
    {
        "Business": "Octiv",
        "Owner Full name": "Jeff Romero",
        "First": "Jeff",
        "Last": "Romero",
        "Owner(s) Email": "jeff@octivdigital.com",
        "LinkedIn": "https://www.linkedin.com/in/jeffromero/",
        "Website": "https://www.octivdigital.com/",
        "Industry": "Tech & AI",
        "Region": "US"
    }
]
```

Note: The HTTP API endpoint (POST /generate_leads) is planned for future updates. Currently, use the CLI.

## Automated Subnet Checks
Post-validation checks ensure lead quality:

1. **Invalid Lead Check**: Detects duplicates, invalid emails, or incorrect formats. If failed, batch score resets to 0, and validator reputation decreases (-20 points).

2. **Collusion Check**: Analyzes buyer feedback and validator scoring patterns using PyGOD and DBScan.
   - Collusion Score (V_c) ≥ 0.7 flags validators.
   - Penalty: If V_c ≥ 0.7, F_v is set to 0 for 90 days, disabling emissions. Affected buyers are also temporarily restricted from submitting queries.

## Technical Details

### Architecture
- **Miners**: neurons/miner.py uses get_leads.py to generate leads.
- **Validators**: neurons/validator.py uses os_validator_model.py and automated_checks.py for scoring.
- **Buyers**: Leadpoet/api/leadpoet_api.py queries miners and filters validated leads.
- **Mock Mode**: Uses Leadpoet/mock.py (MockDendrite, MockSubtensor) for local testing.

### Lead Workflow
1. Buyer requests leads (1–100, optional industry/region) via leadpoet-api.
2. Miners generate leads using get_leads.py.
3. Validators score leads using os_validator_model.py (≥90% to pass).
4. Approved leads undergo automated_checks.py (≥90% to pass).
5. Leads are added to a pool (Leadpoet/base/utils/pool.py) and filtered for delivery.
6. Up to three retry attempts if validation or checks fail.

### API Endpoints
- **POST /generate_leads**: Request leads with num_leads (1-100), optional industry and region.

### Open-Source Frameworks
- **Lead Generation**: miner_models/get_leads.py uses Hunter.io/Clearbit (non-mock mode).
- **Validation**: validator_models/os_validator_model.py checks email format, domain/website reachability.
- **Automated Checks**: validator_models/automated_checks.py verifies email existence and company websites.

### Running in Mock Mode
Test locally without a network connection:

**Miner**:
```bash
python neurons/miner.py --netuid 343 --subtensor.network test --wallet.name miner --wallet.hotkey default --logging.debug
```

**Validator**:
```bash
python neurons/validator.py --netuid 343 --subtensor.network test --wallet.name validator --wallet.hotkey default --logging.debug
```

**API**:
```bash
python leadpoet-api --netuid 343  --subtensor.network test --wallet.name buyer --wallet.hotkey default --logging.debug
```

Note: Mock mode uses dummy data and does not currently enforce active miner/validator processes.

### Scalability
- **Current**: Handles 100-lead batches, ~36,000 leads/day with 10 validators (12 batches/hour/validator).
- **Future**: Scales with miner-validator ratio (target 1:20).

## Roadmap

1. **MVP (Testnet)**: Core lead generation, validation, and CLI API (current).

2. **Next**:
   - Enforce active miner/validator process checks.
   - HTTP API endpoint (POST /generate_leads).
   - Governance voting and compliance auditing.

3. **Future**:
   - Trusted validator thresholds (3-month operation).
   - Sharding for scalability.
   - Mainnet launch.
   - UI at leadpoet.com for querying, purchasing, and feedback.

## Support
- Email: [hello@leadpoet.com](mailto:hello@leadpoet.com)
- Website: https://leadpoet.com

## License
MIT License - see LICENSE for details.
