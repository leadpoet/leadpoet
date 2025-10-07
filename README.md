# LeadPoet | Premium Sales Leads Powered by Bittensor

Welcome to Leadpoet, a decentralized prospect generation subnet built on Bittensor.

## Overview

Leadpoet leverages Bittensor's decentralized architecture to create a scalable marketplace for prospect generation. Miners source high-quality prospects, validators ensure quality through rigorous auditing, and buyers access curated, real-time prospects optimized for conversion. This eliminates reliance on static databases, ensuring fresher, more relevant prospects at lower costs.

### Workflow

1. Miners continuously source high-quality prospects using Firecrawl web scraping and LLM classification
2. Validators check legitimacy of sourced prospects; legitimate prospects enter the Prospect Pool
3. Buyers submit API requests that are broadcast to ALL validators and miners simultaneously via Firestore
4. Miners pause sourcing, curate prospect lists from the Prospect Pool, and submit them to ALL validators
5. ALL validators independently rank the prospects using LLM-based intent scoring
6. API client calculates consensus ranking from all validators (weighted by validator trust)
7. Buyers receive the Final Curated List—consensus-ranked, highest-scoring ICP prospects

**Data Flow**: Buyer submits query → Broadcast to ALL miners & validators → Miners curate prospects → ALL validators independently rank prospects → Client-side consensus calculation → Top N prospects delivered to buyer.

**Token**: TAO is used for staking, rewards, and (future) prospect purchases.

## Getting Started

### Prerequisites

- **Hardware**: 16GB RAM, 4-core CPU, 100GB SSD
- **Software**: Python 3.8+, Bittensor CLI (`pip install bittensor>=6.9.3`)
- **TAO Wallet**: Required for staking and rewards. Create with `btcli wallet create`
- **Google Cloud Credentials**: Required for Firestore database access (all nodes)

### Required API Keys & Credentials

#### For ALL Nodes (Miners, Validators, API Clients)

```bash
# Google Cloud Firestore credentials (REQUIRED)
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your-firebase-credentials.json"
```

#### For Miners

```bash
# REQUIRED: Web scraping and contact extraction
export FIRECRAWL_API_KEY=your_firecrawl_api_key

# REQUIRED: LLM-based industry classification and intent scoring
export OPENROUTER_KEY=your_openrouter_api_key
```

#### For Validators

```bash
# REQUIRED: LLM-based lead ranking
export OPENROUTER_KEY=your_openrouter_api_key

# OPTIONAL: Email verification (falls back to mock if not provided)
export HUNTER_API_KEY=your_hunter_api_key

# OPTIONAL: Advanced email validation (falls back to mock if not provided)
export ZEROBOUNCE_API_KEY=your_zerobounce_api_key

# OPTIONAL: Google Custom Search for LLM validation checks
export GSE_API_KEY=your_google_api_key
export GSE_CX=your_google_cse_id
```

### Google Cloud Firestore Setup

**For Miners/Validators** (Simple 3-step setup):

1. **Get Firebase Config from LeadPoet**:
   - Contact team or check subnet documentation for public config values

2. **Set Environment Variables**:
   ```bash
   export FIREBASE_API_KEY="AIza..."
   export FIREBASE_PROJECT_ID="leadpoet-subnet"
   export FIREBASE_AUTH_DOMAIN="leadpoet-subnet.firebaseapp.com"
   ```

3. **Add to Shell Config** (for persistence):
   ```bash
   echo 'export FIREBASE_API_KEY="AIza..."' >> ~/.bashrc
   echo 'export FIREBASE_PROJECT_ID="leadpoet-subnet"' >> ~/.bashrc
   echo 'export FIREBASE_AUTH_DOMAIN="leadpoet-subnet.firebaseapp.com"' >> ~/.bashrc
   source ~/.bashrc
   ```

**No service account keys needed!** Authentication happens automatically using Firebase anonymous auth + your Bittensor wallet signature.

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

This installs dependencies listed in `setup.py`, including bittensor, requests, numpy, google-cloud-firestore, and others.

**Configure Your Wallet**:
1. Create a wallet if you haven't: `btcli wallet create`
2. Stake TAO as needed (see participation requirements below)
3. Verify wallet setup: `btcli wallet overview --wallet.name <your_wallet> --wallet.hotkey <your_hotkey>`

## For Miners

### Participation Requirements

**Register**: Burn a variable amount of TAO to register on the subnet (netuid 401):
```bash
btcli subnet register --netuid 401 --subtensor.network test --wallet.name <your_wallet> --wallet.hotkey <your_hotkey>
```

### One-time: Publish Your Miner's Public Address

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

This writes the `(ip,port)` to the subnet metagraph so validators can reach your axon. Re-run it only if your IP or port changes.

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

#### Behavior

- Continuously sources new leads
- Monitors Firestore for broadcast API requests (polls every 1 second)
- When API request received:
  - Pauses sourcing
  - Curates prospects from pool using LLM intent scoring
  - Submits curated leads to Firestore for ALL validators to rank
  - Resumes sourcing after submission
- Uses Firecrawl for web scraping and OpenRouter for LLM classification
- Batches multiple leads into single LLM prompts to reduce API costs

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
    "source": "5FEtvBzsh5Zc8nDyq4Jb2nZ7o6ZD2homYsKjbZtFj5tybqth",
    "miner_intent_score": 0.850
}
```

### Miner Incentives

Miners are rewarded based on two mechanisms:

**V2 Reward System** (Current):
- **Sourcing Rewards (S)**: 45% of emissions are provided to miners who source the prospects which are chosen by the validators.
- **Curation Rewards (C)**: 45% of emissions are provided to miners who curate the prospects which are chosen by the validators.
- **Baseline (B)**: 10% of emissions distributed equally to all miners who sourced leads in the last epoch
- Final weight: `W = 0.45 * S + 0.45 * C + 0.10 * B`

**Best Practices**:
- Maintain diverse sourcing to maximize S rewards
- Ensure accurate contact data (emails, LinkedIn URLs)
- Curate high-quality prospects that match client descriptions
- Maintain high uptime to receive API requests

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

### Running a Validator

Run your validator to audit prospects:
```bash
python neurons/validator.py \
    --wallet_name validator \
    --wallet_hotkey default \
    --netuid 401 \
    --subtensor_network test
```

#### Behavior

- Continuously validates sourced leads from Firestore
- Monitors Firestore for broadcast API requests (polls every 1 second)
- When API request received:
  - Pauses sourced lead processing
  - Waits up to 180 seconds for miner submissions
  - Ranks ALL miner leads using LLM scoring (2 rounds per lead)
  - Publishes weights on-chain
  - Submits ranking to Firestore with validator trust score
  - Resumes sourced lead processing
- Runs HTTP server for status queries (auto-detects available port)

### Validation Process

#### Sourced Lead Validation

- Checks:
  - Format: Email regex (`^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$`), no duplicates
  - Accuracy: No disposable domains, basic domain/website reachability
  - Relevance: Matches industry/region filters
- Uses `automated_checks.py` for comprehensive validation
- Approved prospects enter the prospect pool

#### API Request Lead Ranking

- Uses OpenRouter LLM to score each lead (0-0.5 range)
- Two-round scoring with different models for robustness
- Publishes weights based on V2 reward calculation
- Submits ranking with validator trust score for consensus

### Validator Incentives

Validators earn emissions for accurate and consistent ranking.

**Best Practices**:
- Maintain high uptime to process API requests
- Ensure OpenRouter API key has sufficient credits
- Monitor logs for validation/ranking failures
- Keep Firestore credentials valid

## For Buyers (API Clients)

### Participation Requirements

**Stake**: Minimum 50 TAO
```bash
btcli stake --amount 50 --subtensor.network test --wallet.name <your_wallet> --wallet.hotkey <your_hotkey>
```

### Accessing Prospects

Buyers request prospects via the CLI API:
```bash
python Leadpoet/api/leadpoet_api.py \
    --wallet_name owner \
    --wallet_hotkey default \
    --netuid 401 \
    --subtensor_network test
```

#### Interactive CLI

1. Input number of prospects (1–100)
2. Describe your business & ideal customer
3. Request is broadcast to ALL validators and miners via Firestore
4. Multiple validators independently rank prospects
5. Client calculates consensus ranking weighted by validator trust
6. Receive top N consensus-ranked prospects

#### Consensus Calculation

For each lead, the final consensus score is:
```
S_lead = (S_1 × V_1) + (S_2 × V_2) + ... + (S_N × V_N)
```
Where:
- `S_v` = score from validator v for that lead
- `V_v` = trust value for validator v from metagraph
- `N` = total number of validators who submitted rankings

#### Example Output

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
        "consensus_score": 0.92,
        "num_validators_ranked": 2
    }
]
```

## Technical Details

### Architecture

#### Broadcast API Flow

1. **API Client** → Writes request to Firestore `api_requests` collection
2. **ALL Miners** → Poll Firestore, detect request, curate leads, write to `miner_submissions`
3. **ALL Validators** → Poll Firestore, detect request, fetch miner submissions, rank leads
4. **ALL Validators** → Publish weights on-chain, write rankings to `validator_rankings`
5. **API Client** → Poll Firestore for validator rankings, calculate consensus, return top N

#### Components

- **Miners**: `neurons/miner.py` sources leads and curates for API requests
- **Validators**: `neurons/validator.py` validates sourced leads and ranks API request leads
- **API Client**: `Leadpoet/api/leadpoet_api.py` broadcasts requests and calculates consensus
- **Consensus**: `Leadpoet/validator/consensus.py` implements weighted consensus algorithm
- **Database**: Google Cloud Firestore for real-time broadcast coordination

#### Firestore Collections

- `api_requests`: Broadcast API requests from clients
- `miner_submissions`: Curated leads from miners for specific requests
- `validator_rankings`: Independent rankings from each validator
- `prospects`: Validated sourced leads in the prospect pool
- `validator_weights`: Historical weight publications

### LLM Integration

#### Miner Intent Scoring

- Batches multiple leads into single prompt for efficiency
- Returns JSON array with individual scores per lead
- Reduces API calls from N to 1 (where N = number of leads)

#### Validator Lead Ranking

- Two-round scoring with different models
- Primary: DeepSeek Chat (free tier)
- Fallback: Mistral 7B Instruct
- Handles rate limiting with fallback models

### Open-Source Frameworks

- **Prospect Generation**: `miner_models/get_leads.py` uses Firecrawl API for contact extraction
- **LLM Classification**: Uses OpenRouter API for industry classification and intent scoring
- **Validation**: `validator_models/automated_checks.py` verifies email existence and company websites
- **Consensus**: `Leadpoet/validator/consensus.py` implements weighted consensus ranking

## API Key Setup Reference

### Complete Environment Variables

```bash
# ═══════════════════════════════════════════════════════════
# REQUIRED FOR ALL NODES
# ═══════════════════════════════════════════════════════════
export FIREBASE_API_KEY="AIza..."
export FIREBASE_PROJECT_ID="leadpoet-subnet"
export FIREBASE_AUTH_DOMAIN="leadpoet-subnet.firebaseapp.com"

# ═══════════════════════════════════════════════════════════
# REQUIRED FOR MINERS
# ═══════════════════════════════════════════════════════════
export FIRECRAWL_API_KEY=your_firecrawl_api_key
export OPENROUTER_KEY=your_openrouter_api_key

# ═══════════════════════════════════════════════════════════
# REQUIRED FOR VALIDATORS
# ═══════════════════════════════════════════════════════════
export OPENROUTER_KEY=your_openrouter_api_key

# ═══════════════════════════════════════════════════════════
# OPTIONAL (falls back to mock if not provided)
# ═══════════════════════════════════════════════════════════
# Email verification
export HUNTER_API_KEY=your_hunter_api_key
export ZEROBOUNCE_API_KEY=your_zerobounce_api_key

# Google Custom Search for LLM validation checks
export GSE_API_KEY=your_google_api_key
export GSE_CX=your_google_cse_id
```

### Getting API Keys

1. **Firebase Config** (REQUIRED):
   - Contact LeadPoet team or check subnet documentation for public config values
   - No service account keys needed

2. **Firecrawl** (REQUIRED for miners):
   - https://firecrawl.dev/
   - Sign up → Get API key from dashboard

3. **OpenRouter** (REQUIRED for miners & validators):
   - https://openrouter.ai/
   - Sign up → Get API key → Add credits for usage

4. **Hunter.io** (REQUIRED for validators):
   - https://hunter.io/
   - Sign up → Get API key from dashboard
   - Falls back to mock validation if not provided

5. **ZeroBounce** (REQUIRED for validators):
   - https://zerobounce.net/
   - Sign up → Get API key from dashboard
   - Falls back to mock validation if not provided

6. **Google Custom Search** (REQUIRED for miners & validators):
   - https://console.cloud.google.com
   - Enable Custom Search API
   - Create API key (`GSE_API_KEY`)
   - Create Custom Search Engine at https://programmablesearchengine.google.com
   - Get Search Engine ID (`GSE_CX`)
   - Falls back to skipping search if not provided

### Persistent Configuration

Add to your shell configuration file for automatic loading:

```bash
# For bash users
nano ~/.bashrc

# For zsh users
nano ~/.zshrc

# Add all export statements, then reload:
source ~/.bashrc  # or source ~/.zshrc
```

## Troubleshooting

### Common Issues

**"Firebase config not set"**:
- Verify environment variables: `echo $FIREBASE_PROJECT_ID`
- Contact LeadPoet team for correct Firebase config values

**"429 Too Many Requests" from OpenRouter**:
- Free tier models have strict rate limits (1-2 requests/minute)
- Upgrade to paid plan or add credits
- Batch scoring reduces API calls significantly

**"No validator responses" in API client**:
- Verify validators are running: check their terminal output
- Ensure Firebase config is correct on all nodes
- Check request_id in Firestore console for debugging

**Validator "ConcurrencyError" with subtensor**:
- This is normal - validators sync metagraph less frequently to avoid conflicts
- No action needed - validator continues operating

## Support

- Email: [hello@leadpoet.com](mailto:hello@leadpoet.com)
- Website: https://leadpoet.com

## License

MIT License - see LICENSE for details.
