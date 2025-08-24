# LeadPoet | Premium Sales Leads Powered by Bittensor

Welcome to LeadPoet, a decentralized prospect generation subnet built on Bittensor.

## Overview

LeadPoet leverages Bittensor's decentralized architecture to create a scalable marketplace for prospect generation. Miners source high-quality prospects, validators ensure quality through rigorous auditing, and buyers access curated, real-time prospects optimized for conversion. This eliminates reliance on static databases, ensuring fresher, more relevant prospects at lower costs.

**Workflow**:
1. Miners source high-quality prospects using Firecrawl web scraping and LLM classification
2. Validators check legitimacy of sourced prospects; legitimate prospects enter the Prospect Pool
3. Buyers submit requests to validators who send it to all available miners for prospect lists based on their Ideal Customer Profile (ICP)
4. Miners curate prospect lists from the Prospect Pool and submit them to validators
5. Validators score each prospect with an inference-based conversion model
6. Buyers receive the Final Curated List from the validator—the highest-scoring ICP prospects—and manage them in their CRM systems

**Data Flow**: Buyers submit a query → Miners generate prospects → Validators audit and score (≥90% to pass) → Approved prospects undergo automated checks (≥90% to pass) → Highest-scoring prospects are delivered to the buyer.

**Token**: TAO is used for staking, rewards, and (future) prospect purchases.

## Getting Started

### Prerequisites

- **Hardware**: 16GB RAM, 4-core CPU, 100GB SSD
- **Software**: Python 3.8+, Bittensor CLI (`pip install bittensor>=6.9.3`)
- **TAO Wallet**: Required for staking and rewards. Create with `btcli wallet create`
- **API Keys**: Required for real prospect generation and validation:

```bash
# Required for miners using Firecrawl sourcing
export FIRECRAWL_API_KEY=your_firecrawl_api_key

# Required for reading leads from cloud DB
export  LEAD_API="https://leadpoet-api-511161415764.us-central1.run.app"

# Required for miners using OpenRouter LLM classification
export OPENROUTER_API_KEY=your_openrouter_api_key

# Required for validators using Hunter.io email verification
export HUNTER_API_KEY=your_hunter_api_key

# Required for validators using Mailgun email validation
export MAILGUN_API_KEY=your_mailgun_api_key
export MAILGUN_DOMAIN=your_mailgun_domain
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

This installs dependencies listed in `setup.py`, including bittensor, requests, numpy, and others.

**Configure Your Wallet**:
1. Create a wallet if you haven't: `btcli wallet create`
2. Stake TAO as needed (see participation requirements below)
3. Verify wallet setup: `btcli wallet overview --wallet.name <your_wallet> --wallet.hotkey <your_hotkey>`

## For Miners

### Participation Requirements
- **Register**: Burn a variable amount of TAO to register on the subnet (netuid 401):
```bash
btcli subnet register --netuid 401 --subtensor.network test --wallet.name <your_wallet> --wallet.hotkey <your_hotkey>
```

### One-time: publish your miner’s public address
Forward a TCP port on your router / VPS (e.g. `18091`) **once**, then run:

```bash
python scripts/post_ip.py \
    --netuid 401 \
    --subtensor_network test \
    --wallet_name <your_wallet> \
    --wallet_hotkey <your_hotkey> \
    --external_ip <your_public_ip> \
    --external_port <forwarded_port>
```

This writes the `(ip,port)` to the subnet metagraph so validators can reach
your axon. Re-run it only if your IP or port changes.

### Running a Miner
Run your miner to generate prospects:
```bash
python neurons/miner.py \
    --wallet_name miner \
    --wallet_hotkey default \
    --netuid 401 \
    --subtensor_network test \
    --use_open_source_lead_model
```

**Behavior**:
- Responds to validator/buyer queries within 2 minutes + 2 seconds per prospect
- Generates prospects in JSON format (see below)
- Uses Firecrawl for web scraping and OpenRouter for LLM classification
- Continuously sources new leads from domains in `data/domains.csv`

### Prospect Format
Prospects follow this JSON structure:
```json
{
    "business": "A Dozen Cousins",
    "owner_full_name": "Ibraheem Basir",
    "first": "Ibraheem",
    "last": "Basir",
    "owner_email": "ib@adozencousins.com",
    "linkedin": "https://www.linkedin.com/in/ibraheembasir/",
    "website": "https://adozencousins.com/",
    "industry": "Tech & AI",
    "sub_industry": "",
    "region": "US",
    "source": "5FEtvBzsh5Zc8nDyq4Jb2nZ7o6ZD2homYsKjbZtFj5tybqth"
}
```

### Miner Incentives
Miners are rewarded based on prospect quality and reliability:

**Best Practices**:
- Ensure accurate contact data (emails, LinkedIn URLs)
- Align prospects with requested industry/region
- Maintain high uptime to boost rewards

## For Validators

### Participation Requirements
- **Stake**: Minimum 20 TAO:
```bash
btcli stake --amount 20 --subtensor.network test --wallet.name <your_wallet> --wallet.hotkey <your_hotkey>
```
- **Register**: Register on the subnet (netuid 401):
```bash
btcli subnet register --netuid 401 --subtensor.network test --wallet.name <your_wallet> --wallet.hotkey <your_hotkey>
```
- **Add Validator Permissions**: Add validator permissions to your wallet:
```bash
btcli subnet add --netuid 401 --subtensor.network test --wallet.name <your_wallet> --wallet.hotkey <your_hotkey>
```

### Running a Validator
Run your validator to audit prospects:
```bash
python neurons/validator.py \
    --wallet_name validator \
    --wallet_hotkey default \
    --netuid 401 \
    --subtensor_network test
```

**Behavior**:
- Queries miners for prospect batches (100 prospects by default)
- Validates ~20% of each batch within 2 minutes using `os_validator_model.py`
- Runs automated checks with `automated_checks.py`
- Assigns scores (0–100%) and updates miner weights
- Processes sourced leads continuously and adds them to the prospect pool

### Validation Process
**Audit**:
- Checks:
  - Format: Email regex (`^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$`), no duplicates
  - Accuracy: No disposable domains, basic domain/website reachability
  - Relevance: Matches industry/region filters
- Outcome: Approves if ≥90% of sampled prospects are valid; rejects otherwise

**Automated Checks**:
- Post-validation using `automated_checks.py`
- Verifies email existence (Hunter.io) and company website accessibility
- Approves if ≥90% pass; otherwise, validator reputation decreases (-20 points)

### Validator Incentives
Validators are rewarded based on accuracy and consistency.

**Best Practices**:
- Maintain high scoring precision to avoid reputation penalties
- Ensure consistency with the final score
- Monitor logs for validation failures

## For Buyers

### Participation Requirements
- **Stake**: Minimum 50 TAO
```bash
btcli stake --amount 50 --subtensor.network test --wallet.name <your_wallet> --wallet.hotkey <your_hotkey>
```

### Accessing Prospects
Buyers request prospects via the CLI API, receiving validated batches:
```bash
python Leadpoet/api/leadpoet_api.py \
    --wallet_name owner \
    --wallet_hotkey default \
    --netuid 401 \
    --subtensor_network test
```

**Interactive CLI**:
1. Input number of prospects (1–100)
2. Describe your business & ideal customer
3. Receive prospects with owner_email, linkedin, etc.

**Example Output**:
```json
[
    {
        "business": "A Dozen Cousins",
        "owner_full_name": "Ibraheem Basir",
        "first": "Ibraheem",
        "last": "Basir",
        "owner_email": "ib@adozencousins.com",
        "linkedin": "https://www.linkedin.com/in/ibraheembasir/",
        "website": "https://adozencousins.com/",
        "industry": "Tech & AI",
        "sub_industry": "",
        "region": "US",
        "source": "5FEtvBzsh5Zc8nDyq4Jb2nZ7o6ZD2homYsKjbZtFj5tybqth"
    }
]
```

## Automated Subnet Checks
Post-validation checks ensure prospect quality:

1. **Invalid Prospect Check**: Detects duplicates, invalid emails, or incorrect formats. If failed, batch score resets to 0, and validator reputation decreases (-20 points).



## Technical Details

### Architecture
- **Miners**: `neurons/miner.py` uses `get_leads.py` and `firecrawl_sourcing.py` to generate prospects
- **Validators**: `neurons/validator.py` uses `os_validator_model.py` and `automated_checks.py` for scoring
- **Buyers**: `Leadpoet/api/leadpoet_api.py` queries miners and filters validated prospects

### Prospect Workflow
1. Buyer requests prospects (1–100, business description) via `leadpoet_api.py`
2. Miners generate prospects using `get_leads.py` and `firecrawl_sourcing.py`
3. Validators score prospects using `os_validator_model.py` (≥90% to pass)
4. Approved prospects undergo `automated_checks.py` (≥90% to pass)
5. Prospects are added to a pool (`Leadpoet/base/utils/pool.py`) and filtered for delivery
6. Up to three retry attempts if validation or checks fail

### API Endpoints
- **CLI Interface**: Interactive command-line interface for requesting prospects

### Open-Source Frameworks
- **Prospect Generation**: `miner_models/get_leads.py` uses Hunter.io APIs
- **Web Scraping**: `miner_models/firecrawl_sourcing.py` uses Firecrawl API for contact extraction
- **LLM Classification**: Uses OpenRouter API for industry classification
- **Validation**: `validator_models/os_validator_model.py` checks email format, domain/website reachability
- **Automated Checks**: `validator_models/automated_checks.py` verifies email existence and company websites

## API Key Setup

### Required API Keys

**For Miners**:
- **Firecrawl API Key**: Required for web scraping and contact extraction
  - Get from: https://firecrawl.dev/
  - Set: `export FIRECRAWL_API_KEY=your_key`

- **Leadpoet API**: Required for reading leads from cloud DB
  - Set: `export  LEAD_API="https://leadpoet-api-511161415764.us-central1.run.app"`

- **OpenRouter API Key**: Required for LLM-based industry classification
  - Get from: https://openrouter.ai/
  - Set: `export OPENROUTER_API_KEY=your_key`

**For Validators**:

- **Leadpoet API**: Required for writing leads to cloud DB
  - Set: `export  LEAD_API="https://leadpoet-api-511161415764.us-central1.run.app"`

- **Hunter.io API Key**: Required for email verification
  - Get from: https://hunter.io/
  - Set: `export HUNTER_API_KEY=your_key`

- **Mailgun API Key**: Required for email validation
  - Get from: https://mailgun.com/
  - Set: `export MAILGUN_API_KEY=your_key`
  - Set: `export MAILGUN_DOMAIN=your_domain`

### Setting Up API Keys
```bash
# Add to your ~/.bashrc or ~/.zshrc for persistence
export FIRECRAWL_API_KEY=your_firecrawl_api_key
export OPENROUTER_API_KEY=your_openrouter_api_key
export HUNTER_API_KEY=your_hunter_api_key
export MAILGUN_API_KEY=your_mailgun_api_key
export MAILGUN_DOMAIN=your_mailgun_domain

# Reload shell configuration
source ~/.bashrc  # or source ~/.zshrc
```

## Support
- Email: [hello@leadpoet.com](mailto:hello@leadpoet.com)
- Website: https://leadpoet.com

## License
MIT License - see LICENSE for details.
