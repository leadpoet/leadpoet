# Leadpoet | Premium B2B Leads Powered by Bittensor

A decentralized B2B lead generation subnet built on Bittensor that delivers fresh, validated prospects through a consensus-driven marketplace.

## 🚀 Overview

Leadpoet transforms B2B lead generation by creating a decentralized marketplace where:
- **Miners** source high-quality prospects using web scraping and AI
- **Validators** ensure quality through consensus-based validation (3+ validators per lead)
- **Buyers** access curated, real-time prospects optimized for their ICP (Ideal Customer Profile)

### Key Innovation: Consensus Validation

Unlike traditional lead databases, Leadpoet requires **consensus from multiple validators** before a lead is accepted:
- Each prospect must be validated by 3 independent validators
- 2+ validators must agree (valid/invalid) for consensus
- Prevents gaming and ensures higher quality leads
- Validators must participate in ≥10% of consensus decisions to earn rewards

## 📋 Prerequisites

### Hardware Requirements
- **Miners/Validators**: 16GB RAM, 4-core CPU, 100GB SSD
- **Network**: Stable internet connection with open ports for axon communication

### Software Requirements
- Python 3.8+
- Bittensor CLI: `pip install bittensor>=6.9.3`
- TAO Wallet: Create with `btcli wallet create`

## 🔑 Required Credentials

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

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/leadpoet/Leadpoet.git
cd bittensor-subnet

# Install dependencies
pip install -e .

# Verify installation
python -c "import Leadpoet; print('Leadpoet installed successfully')"
```

## ⛏️ For Miners

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

1. **Continuous Sourcing**: Actively search for new B2B prospects
2. **Smart Submission**: Send prospects to Supabase `prospect_queue`
3. **Consensus Validation**: Prospects validated by 3+ validators
4. **Accepted Leads**: Only consensus-approved leads enter the main database

### Reward System (100% Sourcing-Based)

Miners earn rewards purely based on the leads they source that get accepted through consensus:
- If you source 60% of accepted leads in an epoch → you get 60% of emissions
- Simple, transparent, and directly tied to value creation
- Epoch duration: 72 minutes (360 blocks)

## 🛡️ For Validators

### Getting Started

1. **Stake TAO** (minimum 20 TAO):
```bash
btcli stake add \
    --amount 20 \
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

### Consensus Validation System

**How it works:**
1. Validators pull prospects from the queue (first-come, first-served)
2. Each prospect gets a 15-second window for validation
3. Need 3 validators to participate, 2+ must agree for consensus
4. Accepted leads → main database, Rejected leads → discarded

**Eligibility for Rewards:**
- Must participate in ≥10% of consensus decisions per epoch
- Checked via secure Edge Function (cannot be bypassed)
- If eligible, validator can set miner weights on-chain

### Validation Checks

Validators perform multi-stage validation:
1. **Email validation**: Format, domain, disposable check
2. **DNS/SPF/DMARC**: Email deliverability verification  
3. **Company verification**: Website, LinkedIn, Google search
4. **LLM validation**: AI-powered legitimacy scoring

## 💰 Reward Distribution

### Current System (Epoch-based)

Every 72 minutes (1 epoch):
1. **Validators check eligibility**: Via Edge Function (≥10% consensus participation)
2. **Calculate weights**: Based on which miners sourced accepted leads
3. **Set weights on-chain**: Proportional to leads sourced
4. **TAO distribution**: Automatic based on weights

### Security Features

- **Edge Function enforcement**: 10% rule checked server-side
- **No local calculations**: Validators can't manipulate weights
- **JWT-based auth**: Validators only have limited database access
- **Consensus requirement**: No single validator can accept/reject leads

## 📊 For API Clients (Buyers)

### Accessing Leads

```bash
python Leadpoet/api/leadpoet_api.py \
    --wallet_name buyer \
    --wallet_hotkey default \
    --netuid 71 \
    --subtensor_network finney
```

**Interactive Flow:**
1. Specify number of leads needed (1-100)
2. Describe your ideal customer profile
3. Receive consensus-validated, AI-ranked leads

### Lead Quality Guarantees

- ✅ Validated by 3+ independent validators
- ✅ Consensus required (2+ validators must agree)
- ✅ Fresh data (sourced in real-time, not from stale databases)
- ✅ AI-scored for relevance to your ICP

## 🏗️ Architecture

### Data Flow

```
Miner sources lead → Prospect Queue → 3+ Validators validate → 
Consensus (2+ agree) → Lead Database → Available to Buyers
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

## 🔧 Troubleshooting

### Common Issues

**"No JWT token available"**
- Validators need to wait for token generation on first run
- Token auto-refreshes every hour

**"Not eligible - less than 10% consensus"**
- Validator needs to validate more prospects
- Check participation with Edge Function logs

**"Prospect already in queue"**
- Duplicate detection is working (this is good!)
- Miner should source new, unique leads

**Consensus not reached after 15 seconds**
- Need more validators online
- With only 2 validators: both must agree
- With 1 validator: prospect gets reset to queue

## 📚 Key Concepts

### Consensus Rules
- **3 validators**: 2+ must agree (standard case)
- **2 validators**: Both must agree (degraded mode)
- **1 validator**: No consensus, prospect reset
- **15-second window**: From first pull to consensus deadline

### Epoch Timing
- **Duration**: 72 minutes (360 blocks)
- **Weight calculation**: Blocks 80-120 (testing) or 355-360 (production)
- **Automatic cleanup**: Old validation data removed each epoch

### Quality Metrics
- **Consensus score**: How many validators agreed
- **Intent score**: AI-based relevance scoring
- **Email score**: Deliverability confidence

## 📞 Support

- **Discord**: [Join our community](https://discord.gg/leadpoet)
- **Email**: hello@leadpoet.com

## 📄 License

MIT License - See LICENSE file for details

---

**Leadpoet** - Decentralized B2B lead generation powered by Bittensor 🚀
