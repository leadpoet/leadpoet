# Leadpoet | AI Sales Agents Powered by Bittensor

Leadpoet is Subnet 71, the decentralized AI sales agent subnet built on Bittensor. Leadpoet's vision is streamlining the top of sales funnel, starting with high-quality lead generation today and evolving into a fully automated sales engine where meetings with your ideal customers seamlessly appear on your calendar.

## Overview

Leadpoet transforms lead generation by creating a decentralized marketplace where:
- **Miners** source high-quality prospects using web scraping and AI
- **Validators** ensure quality through consensus-based validation
- **Buyers** access curated prospects optimized for their Ideal Customer Profile (ICP)

Unlike traditional lead databases, Leadpoet requires **consensus from multiple validators** before a lead is approved:
- Each prospect is validated by three independent validators
- Prevents gaming and ensures the lead pool limited to **verified, highest quality** leads

## Prerequisites

### Hardware Requirements
- **Miners/Validators**: 16GB RAM, 8-core CPU, 100GB SSD
- **Network**: Stable internet connection with open ports for axon communication

### Software Requirements
- Python 3.8+
- Bittensor CLI: `pip install bittensor>=6.9.3`
- Bittensor Wallet: `btcli wallet create`

## Required Credentials

### For Miners

```bash
# Required for Dynamic Lead Generation
export FIRECRAWL_API_KEY="your_firecrawl_key"        # Web scraping
export OPENROUTER_KEY="your_openrouter_key"          # AI classification
export GSE_API_KEY="your_google_api_key"             # Google Search
export GSE_CX="your_search_engine_id"                # Custom Search ID
export ANYMAIL_FINDER_API_KEY="your_anymail_key"     # Email discovery

# Optional Enrichment APIs For Enhanced Lead Quality
export CORESIGNAL_API_TOKEN="your_coresignal_token"      # Company data enrichment
export COMPANY_ENRICH_API_KEY="your_company_enrich_key"  # Additional company enrichment
export SNOVIO_CLIENT_ID="your_snovio_id"                 # Snov.io email finder
export SNOVIO_CLIENT_SECRET="your_snovio_secret"         # Snov.io secret
export MAILGUN_SMTP_LOGIN="your_mailgun_login"           # Email validation
export MAILGUN_SMTP_PW="your_mailgun_password"           # Mailgun password
```

### For Validators

```bash
# Email, LLM, and Google Search Validation Tools
export OPENROUTER_KEY="your_openrouter_key"          # Required for lead validation
export ZEROBOUNCE_API_KEY="your_zerobounce_key"      # Primary email validation
export HUNTER_API_KEY="your_hunter_key"              # Fallback email verification
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
3. **Consensus Validation**: Prospects validated by validators
4. **Approved Leads**: Only consensus-approved leads enter the main lead pool

### Lead JSON Structure

Miners must submit prospects with the following structure:

```json
{
  "business": "SpaceX",
  "owner_full_name": "Elon Musk",
  "owner_email": "elon@spacex.com",
  "website": "https://spacex.com",
  "phone": "+1-310-363-6000",
  "linkedin": "https://linkedin.com/in/elonmusk",
  "industry": "Aerospace Manufacturing",
  "location": "Hawthorne, CA",
  "employee_count": "10000+",
  "founded_year": 2002,
  "description": "Aerospace manufacturer and space transportation company focused on reducing space transportation costs"
}
```

### Reward System

Miners earn rewards **proportional to approved leads** they source:
- If a miner sources 60% of approved leads in an epoch, they receive 60% of miner emissions for the following epoch
- Simple, transparent, and directly tied to value creation

## For Validators

### Getting Started

1. **Stake Alpha / TAO** (meet base Bittensor validator requirements):
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

3. **Run the validator**:
```bash
python neurons/validator.py \
    --wallet_name validator \
    --wallet_hotkey default \
    --netuid 71 \
    --subtensor_network finney
```

Note: Validators are configured to auto-update from GitHub on a 5-minute interval.

### Consensus Validation System

Validators pull prospects from the queue (first-come, first-served) and have a 15-second window to validate. With three validators participating, majority agreement is required for consensus. Approved leads move to the main database, rejected leads are discarded.

**Eligibility for Rewards:**
- Must participate in at least 10% of consensus decisions per epoch
- Verified server-side via Edge Function 
- If eligible, validators receive miner weights at end of epoch to commit on-chain

**Validators perform multi-stage quality checks:**
1. **Email validation**: Format, domain, disposable check, deliverability check
2. **Company & Contact verification**: Website, LinkedIn, Google search
3. **Reputation Score**: Coming soon

## Reward Distribution

### Epoch-Based Rewards

1. Validators check eligibility (> 10% consensus participation requirement)
2. Miner weights calculated based on sourced approved leads
3. Weights set on-chain proportional to leads sourced

### Security Features

- **Edge Function enforcement**: Eligibility checked server-side, cannot be bypassed
- **Server-side weight calculations**: Validators can't manipulate weights
- **JWT-based auth**: Validators only have limited database access
- **Consensus requirement**: No single validator can approve/reject leads

## Data Flow

```
Miner Sources Leads → Prospect Queue → Validators Run Quality Checks → 
Validator Consensus → Lead Pool → Curation for Buyer Requests (Month 2)
```

## Roadmap

### Month 1: Launch & Foundation
- Codebase goes live on SN71
- Refine sourcing; gatekeep low-quality leads from the DB
- Ensure stable miner and validator operations
- Monitor and optimize consensus validation

### Month 2: Curation & Beta Users
- Miners begin curating leads from the lead pool based on Ideal Customer Profiles (ICPs)
- Implement curation rewards into the incentive mechanism
- Onboard initial beta customers for feedback
- Refine models for lead ranking and scoring

### Month 3: Product Launch & Growth
- Product launch with marketing and sales campaigns
- Open Leadpoet platform to paying customers
- Scale miner curation and sourcing capabilities
- Introduce weekly ICP themes to incentivize sourcing leads in specific industries
- Optimize end-to-end lead generation pipeline

## Troubleshooting

Common Errors:

**"No JWT token available"**
- Validators need to wait for token generation on first run
- Token auto-refreshes every hour

**"Not eligible - less than 10% consensus"**
- Validator needs to validate more prospects to meet eligibility threshold

**"Prospect already in queue"**
- The prospect attempted to be submitted is already in the prospect queue

**Consensus not reached after 15 seconds**
- Insufficient number of validators ran quality checks on the lead to reach consensus

## Support

For support and discussion:
- **Leadpoet FAQ**: Check out our FAQ at www.leadpoet.com/faq to learn more about Leadpoet!
- **Bittensor Discord**: Join the Leadpoet SN71 channel and message us!
- **Email**: hello@leadpoet.com

## License

MIT License - See LICENSE file for details
