# Leadpoet | Premium Lead Generation Powered by Bittensor

Leadpoet is a decentralized lead generation subnet built on Bittensor that delivers fresh, validated prospects through a consensus-driven marketplace. Starting with high-quality lead sourcing today, our vision is to evolve Leadpoet into a fully automated sales engine where qualified meetings with prospects seamlessly appear on your calendar.

## Overview

Leadpoet transforms lead generation by creating a decentralized marketplace where:
- **Miners** source high-quality prospects using web scraping and AI
- **Validators** ensure quality through consensus-based validation (2-3 validators per lead)
- **Buyers** access curated, real-time prospects optimized for their Ideal Customer Profile (ICP)

### Consensus Validation

Unlike traditional lead databases, Leadpoet requires **consensus from multiple validators** before a lead is accepted:
- Each prospect must be validated by 2-3 independent validators
- Validators must agree (valid/invalid) for consensus
- Prevents gaming and ensures higher quality leads

## Prerequisites

### Hardware Requirements
- **Miners/Validators**: 16GB RAM, 8-core CPU, 100GB SSD
- **Network**: Stable internet connection with open ports for axon communication

### Software Requirements
- Python 3.8+
- Bittensor CLI: `pip install bittensor>=6.9.3`
- TAO Wallet: Create with `btcli wallet create`

## Required Credentials

### For Miners

```bash
# Required for basic mining
export FIRECRAWL_API_KEY="your_firecrawl_key"        # Web scraping
export OPENROUTER_KEY="your_openrouter_key"          # AI classification

# Required for Lead Sorcerer (advanced lead generation)
export GSE_API_KEY="your_google_api_key"             # Google Search
export GSE_CX="your_search_engine_id"                # Custom Search ID

# Optional Enrichment APIs (for enhanced lead quality)
export CORESIGNAL_API_TOKEN="your_coresignal_token"      # Company data enrichment
export COMPANY_ENRICH_API_KEY="your_company_enrich_key"  # Additional company enrichment
export ANYMAIL_FINDER_API_KEY="your_anymail_key"         # Email discovery
export SNOVIO_CLIENT_ID="your_snovio_id"                 # Snov.io email finder
export SNOVIO_CLIENT_SECRET="your_snovio_secret"         # Snov.io secret
export MAILGUN_SMTP_LOGIN="your_mailgun_login"           # Email validation
export MAILGUN_SMTP_PW="your_mailgun_password"           # Mailgun password
```

### For Validators

```bash
# LLM for lead ranking and validation
export OPENROUTER_KEY="your_openrouter_key"          # Required for intent scoring

# Email validation services (Optional - falls back to basic checks)
export ZEROBOUNCE_API_KEY="your_zerobounce_key"      # Advanced email validation
export HUNTER_API_KEY="your_hunter_key"              # Email verification

# Google Search for validation (Optional)
export GSE_API_KEY="your_google_api_key"             # For company verification
export GSE_CX="your_search_engine_id"                # Custom search engine ID
```

## Installation

```bash
# Clone the repository
git clone https://github.com/leadpoet/Leadpoet.git
cd Leadpoet

# Install dependencies
pip install -e .

# Verify installation
python -c "import Leadpoet; print('Leadpoet installed successfully')"
```

## For Miners

### Getting Started

1. **Register on subnet** (netuid 71):
```bash
btcli subnet register \
    --netuid 71 \
    --subtensor.network finney \
    --wallet.name miner \
    --wallet.hotkey default
```

2. **Publish your IP** (one-time setup):
```bash
python scripts/post_ip.py \
    --netuid 71 \
    --subtensor_network finney \
    --wallet_name miner \
    --wallet_hotkey default \
    --external_ip YOUR_PUBLIC_IP \
    --external_port 18091
```

3. **Run the miner**:
```bash
python neurons/miner.py \
    --wallet_name miner \
    --wallet_hotkey default \
    --netuid 71 \
    --subtensor_network finney
```

### How Miners Work

1. **Continuous Sourcing**: Actively search for new prospects
2. **Smart Submission**: Send prospects to Supabase `prospect_queue`
3. **Consensus Validation**: Prospects validated by 2-3 validators
4. **Accepted Leads**: Only consensus-approved leads enter the main database

### Reward System

Miners earn rewards based on the leads they source that get accepted through consensus:
- If you source 60% of accepted leads in an epoch, you receive 60% of emissions
- Simple, transparent, and directly tied to value creation
- Epoch duration: 72 minutes (360 blocks)

## For Validators

### Getting Started

1. **Stake TAO** (meet base Bittensor validator requirements):
```bash
btcli stake add \
    --amount <amount> \
    --subtensor.network finney \
    --wallet.name validator \
    --wallet.hotkey default
```

2. **Register on subnet**:
```bash
btcli subnet register \
    --netuid 71 \
    --subtensor.network finney \
    --wallet.name validator \
    --wallet.hotkey default
```

3. **Run the validator** (includes automatic code updates):
```bash
python neurons/validator.py \
    --wallet_name validator \
    --wallet_hotkey default \
    --netuid 71 \
    --subtensor_network finney
```

Note: Validators automatically update from GitHub every 5 minutes to ensure they're running the latest code.

### Consensus Validation System

Validators pull prospects from the queue (first-come, first-served) and have a 15-second window to validate. With 2-3 validators participating, agreement is required for consensus. Accepted leads move to the main database, rejected leads are discarded.

**Eligibility for Rewards:**
- Must participate in at least 10% of consensus decisions per epoch
- Verified server-side via Edge Function 
- If eligible, validators receive miner weights to set on-chain

### Validation Checks

Validators perform multi-stage validation:
1. **Email validation**: Format, domain, disposable check
2. **DNS/SPF/DMARC**: Email deliverability verification  
3. **Company verification**: Website, LinkedIn, Google search
4. **LLM validation**: AI-powered legitimacy scoring

## Reward Distribution

### Epoch-Based Distribution

Every 72 minutes (1 epoch):
1. Validators check eligibility (10% consensus participation requirement)
2. Weights calculated based on which miners sourced accepted leads
3. Weights set on-chain proportional to leads sourced

### Security Features

- **Edge Function enforcement**: Eligibility checked server-side, cannot be bypassed
- **No local calculations**: Validators can't manipulate weights
- **JWT-based auth**: Validators only have limited database access
- **Consensus requirement**: No single validator can accept/reject leads

## Architecture

### Data Flow

```
Miner sources lead → Prospect Queue → 2-3 Validators validate → 
Consensus (validators agree) → Lead Database → Available to Buyers
```

### Database Structure (Supabase)

- **`prospect_queue`**: Incoming prospects from miners
- **`validation_tracking`**: Individual validator assessments
- **`consensus_results`**: Aggregated consensus decisions
- **`leads`**: Final accepted leads with consensus scores
- **`members`**: Registered miners and validators

### Security Model

```
Miners:      Write to prospect_queue only
Validators:  Read prospect_queue, Write validation_tracking
Edge Func:   Full access (calculates weights server-side)
Buyers:      Read leads only
```

## Roadmap

### Month 1: Launch & Foundation (Current)
- Codebase goes live on SN71
- Launch and refine the sourcing mechanism, ensuring only high-quality leads enter the lead DB
- Establish stable miner and validator operations
- Monitor and optimize consensus validation system

### Month 2: Curation & Beta Users
- Implement curation rewards into the incentive mechanism
- Miners begin curating leads for beta users based on Ideal Customer Profiles (ICPs)
- Refine LLM-based lead ranking and scoring
- Onboard initial beta customers for feedback

### Month 3: Product Launch & Growth
- Full product launch with marketing and sales campaigns
- Open Leadpoet platform to paying customers
- Scale miner curation and sourcing capabilities
- Introduce weekly ICP themes to incentivize sourcing leads in specific industries
- Optimize end-to-end lead generation pipeline

Note: Implementation details may evolve based on network performance and community feedback.

## Troubleshooting

### Common Issues

**"No JWT token available"**
- Validators need to wait for token generation on first run
- Token auto-refreshes every hour

**"Not eligible - less than 10% consensus"**
- Validator needs to validate more prospects to meet eligibility threshold

**"Prospect already in queue"**
- Duplicate detection is working correctly
- Miner should source new, unique leads

**Consensus not reached after 15 seconds**
- Need more validators online
- With only 2 validators: both must agree
- With 1 validator: prospect gets reset to queue

## Support

For support and discussion:
- **Bittensor Discord**: Join the Leadpoet SN71 channel and message us!
- **Email**: hello@leadpoet.com

## License

MIT License - See LICENSE file for details

---

**Leadpoet** - Decentralized lead generation powered by Bittensor
