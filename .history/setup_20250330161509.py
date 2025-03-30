Setup Process for LeadPoet Subnet
Here’s how to set up and use the LeadPoet subnet with this setup.py:

Prerequisites:
Python 3.8+ installed.
Git installed (for cloning the repository).
API keys for Hunter.io and Clearbit (set as environment variables: HUNTER_API_KEY, CLEARBIT_API_KEY).
Clone the Repository:
bash

Collapse

Wrap

Copy
git clone https://github.com/yourusername/leadpoet-subnet.git
cd leadpoet-subnet
Set Up a Virtual Environment (Recommended):
bash

Collapse

Wrap

Copy
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install the Package:
bash

Collapse

Wrap

Copy
pip install .
This installs all dependencies listed in install_requires and registers the leadpoet, leadpoet-validate, and leadpoet-api CLI commands.
Configure Environment Variables:
bash

Collapse

Wrap

Copy
export HUNTER_API_KEY="your_hunter_api_key"
export CLEARBIT_API_KEY="your_clearbit_api_key"
Add these to your .bashrc, .zshrc, or equivalent for persistence.
Run a Miner:
bash

Collapse

Wrap

Copy
leadpoet --netuid 33 --wallet.name your_miner_wallet --wallet.hotkey your_miner_hotkey
Replace your_miner_wallet and your_miner_hotkey with your Bittensor wallet credentials.
Add --use_open_source_lead_model to use get_leads.py (once moved to miner_models/).
Run a Validator:
bash

Collapse

Wrap

Copy
leadpoet-validate --netuid 33 --wallet.name your_validator_wallet --wallet.hotkey your_validator_hotkey
Use --use_open_source_validator_model to enable os_validator_model.py (once moved).
Validator audits miner batches and scores them per your documentation.
Test with Mock Environment:
bash

Collapse

Wrap

Copy
python template/mock.py
Simulates the subnet with 5 miners and 1 validator, using dummy leads or get_leads.py if available.
Use the API (Optional):
If you add a main() to leadpoet_api.py, run:
bash

Collapse

Wrap

Copy
leadpoet-api --netuid 33 --wallet.name your_wallet --wallet.hotkey your_hotkey
Alternatively, integrate programmatically:
python

Collapse

Wrap

Copy
from template.api.leadpoet_api import LeadPoetAPI
import asyncio
wallet = bt.wallet(name="your_wallet", hotkey="your_hotkey")
api = LeadPoetAPI(wallet, netuid=33)
leads = asyncio.run(api.get_leads(10, "Tech & AI", "US"))
print(leads)
Move External Models (Future Step):
Place get_leads.py in miner_models/, os_validator_model.py and automated_checks.py in validator_models/.
Update imports in miner.py, validator.py, and mock.py to from miner_models.get_leads import get_leads, etc.
Verification Against Technical Documentation
CLI Support: entry_points enable leadpoet submit-like functionality via leadpoet and leadpoet-validate.
Dependencies: Cover automated pre-checks (dnspython, requests), miner lead generation (requests), and scoring (numpy).
Scalability: Package structure supports the 1:20 miner-validator ratio and 100-lead batches via mock.py and forward.py.
Process: Installation supports the workflow: miner submission, validator review, and API access.