# LeadPoet | Premium Sales Leads Powered by Bittensor.

Welcome to LeadPoet, a decentralized prospect generation subnet built on Bittensor, with an initial focus on SMB sales agencies seeking high-quality, conversion-ready sales leads.

## Overview

LeadPoet leverages Bittensor's decentralized architecture to create a scalable marketplace for prospect generation. Miners generate prospect batches, validators ensure quality through rigorous auditing, and buyers access curated, real-time prospects optimized for conversion. This eliminates reliance on static databases, ensuring fresher, more relevant prospects at lower costs.

- **Miners**: Generate prospect batches in response to buyer queries using their prospect generation model (example model provided in miner_models/get_leads.py).
- **Validators**: Audit prospect quality using their own model or validator_models/os_validator_model.py.
- **Buyers**: Request prospects via the Leadpoet/api/leadpoet_api.py CLI or the leadpoet.com website, receiving validated prospects with fields like Email and LinkedIn Pages.

**Data Flow**: Buyers submit a query → Miners generate prospects → Validators audit and score (≥90% to pass) → Approved prospects undergo automated checks (≥90% to pass) → Highest-scoring prospects are delivered to the buyer.

**Token**: TAO is used for staking, rewards, and (future) prospect purchases.

## Getting Started

### Prerequisites

- **Hardware**: 16GB RAM, 4-core CPU, 100GB SSD.
- **Software**: Python 3.8+, Bittensor CLI (pip install bittensor>=6.9.3).
- **TAO Wallet**: Required for staking and rewards. Create with btcli wallet create.
- **API Keys (Optional for Miners)**: For real prospect generation using Hunter.io and Clearbit, set:
  
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
To enable real prospect generation in miner_models/get_leads.py:
```bash
export HUNTER_API_KEY=your_hunter_api_key
export CLEARBIT_API_KEY=your_clearbit_api_key
```

## For Miners

### Participation Requirements
- **Register**: Burn a variable amount of TAO to register on the subnet (netuid 343):
```bash
btcli subnet register --netuid 343 --subtensor.network test --wallet.name <your_wallet> --wallet.hotkey <your_hotkey>
```

### Running a Miner
Run your miner to generate prospects:
```bash
python neurons/miner.py --netuid 343 --subtensor.network test --wallet.name miner --wallet.hotkey default --logging.debug
```

- Add `--use_open_source_lead_model` to use get_leads.py with Hunter.io/Clearbit (requires API keys).
- Add `--mock` for local testing with dummy data.
- Add `--logging_trace` for detailed logs.

**Behavior**:
- Responds to validator/buyer queries within 2 minutes + 2 seconds per prospect.
- Generates prospects in JSON format (see below).
- In mock mode, uses MockDendrite and MockSubtensor to simulate responses.

### Prospect Format
Prospects follow this JSON structure:
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
Miners are rewarded based on prospect quality and reliability:

**Best Practices**:
- Ensure accurate contact data (emails, LinkedIn URLs).
- Align prospects with requested industry/region.
- Maintain high uptime to boost rewards.

## For Validators

### Participation Requirements
- **Stake**: Minimum 20 TAO:
```bash
btcli stake --amount 20 --subtensor.network test --wallet.name <your_wallet> --wallet.hotkey <your_hotkey>
```
- **Register**: Registers on the subnet (netuid 343):
```bash
btcli subnet register --netuid 343 --subtensor.network test --wallet.name <your_wallet> --wallet.hotkey <your_hotkey>
```

### Running a Validator
Run your validator to audit prospects:
```bash
python neurons/validator.py --netuid 343 --subtensor.network test --wallet.name validator --wallet.hotkey default --logging.debug
```

- Add `--use_open_source_validator_model` to use os_validator_model.py.
- Add `--mock` for local testing.
- Add `--logging_trace` for detailed logs.

**Behavior**:
- Queries miners for prospect batches (100 prospects by default).
- Validates ~20% of each batch within 2 minutes using os_validator_model.py.
- Runs automated checks with automated_checks.py.
- Assigns scores (0–100%) and updates miner weights.

### Validation Process
**Audit**:
- Checks:
  - Format: Email regex (^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$), no duplicates.
  - Accuracy: No disposable domains, basic domain/website reachability.
  - Relevance: Matches industry/region filters.
- Outcome: Approves if ≥90% of sampled prospects are valid; rejects otherwise.

**Automated Checks**:
- Post-validation using automated_checks.py.
- Verifies email existence (Hunter.io in non-mock mode) and company website accessibility.
- Approves if ≥90% pass; otherwise, validator reputation decreases (-20 points).

### Validator Incentives
Validators are rewarded based on accuracy and consistency.

**Best Practices**:
- Maintain high scoring precision to avoid reputation penalties.
- Ensure consistency with the final score.
- Monitor logs for validation failures.
  
## For Buyers

### Participation Requirements
- **Stake**: Minimum 50 TAO.
```bash
btcli stake --amount 50 --subtensor.network test --wallet.name <your_wallet> --wallet.hotkey <your_hotkey>
```

### Accessing Prospects
Buyers request prospects via the CLI API, receiving validated batches:
```bash
python leadpoet-api --netuid 343  --subtensor.network test --wallet.name buyer --wallet.hotkey default --logging.debug
```

- Add `--mock` for local testing.
- Add `--logging_trace` for detailed logs.

**Interactive CLI**:
1. Input number of prospects (1–100).
2. Select industry (Tech & AI, Finance & Fintech, Health & Wellness, Media & Education, Energy & Industry) or skip.
3. Specify region (e.g., US) or skip.
4. Receive prospects with Owner(s) Email, LinkedIn, etc.

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
Post-validation checks ensure prospect quality:

1. **Invalid Prospect Check**: Detects duplicates, invalid emails, or incorrect formats. If failed, batch score resets to 0, and validator reputation decreases (-20 points).

2. **Collusion Check**: Analyzes buyer feedback and validator scoring patterns using PyGOD and DBScan.
   - Collusion Score (V_c) ≥ 0.7 flags validators.
   - Penalty: If V_c ≥ 0.7, F_v is set to 0 for 90 days, disabling emissions. Affected buyers are also temporarily restricted from submitting queries.

## Technical Details

### Architecture
- **Miners**: neurons/miner.py uses get_leads.py to generate prospects.
- **Validators**: neurons/validator.py uses os_validator_model.py and automated_checks.py for scoring.
- **Buyers**: Leadpoet/api/leadpoet_api.py queries miners and filters validated prospects.
- **Mock Mode**: Uses Leadpoet/mock.py (MockDendrite, MockSubtensor) for local testing.

### Prospect Workflow
1. Buyer requests prospects (1–100, optional industry/region) via leadpoet-api.
2. Miners generate prospects using get_leads.py.
3. Validators score prospects using os_validator_model.py (≥90% to pass).
4. Approved prospects undergo automated_checks.py (≥90% to pass).
5. Prospects are added to a pool (Leadpoet/base/utils/pool.py) and filtered for delivery.
6. Up to three retry attempts if validation or checks fail.

### API Endpoints
- **POST /generate_leads**: Request prospects with num_leads (1-100), optional industry and region.

### Open-Source Frameworks
- **Prospect Generation**: miner_models/get_leads.py uses Hunter.io/Clearbit (non-mock mode).
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

## Roadmap

1. **MVP (Current)**:
   - Live on testnet.
   - Core prospect generation, validation, and CLI API functinality.

2. **Next**:
   - Mainnet launch.
   - Early access for CRM and prospect generation at leadpoet.com.

3. **Future**:
   - Enforce compliance auditing.
   - Full launch at leadpoet.com, including pricing tiers.

## Support
- Email: [hello@leadpoet.com](mailto:hello@leadpoet.com)
- Website: https://leadpoet.com

## License
MIT License - see LICENSE for details.
