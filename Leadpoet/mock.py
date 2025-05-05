# Leadpoet/mock.py

import time
import asyncio
import random
import bittensor as bt
from typing import List, Dict, Optional
from Leadpoet.protocol import LeadRequest
from miner_models.get_leads import get_leads
import numpy as np
from validator_models.automated_checks import validate_lead_list as auto_check_leads

class MockWallet:
    def __init__(self, name: str = "mock_wallet", hotkey: str = "default"):
        self.name = name
        self.hotkey_str = hotkey
        self._hotkey = MockHotkey()
        self._coldkey = MockHotkey()
        self.hotkey_ss58_address = "5MockHotkeyAddress123456789"  # Direct attribute for compatibility
        self.coldkey_ss58_address = "5MockColdkeyAddress123456789"
        self.ss58_address = "5MockHotkeyAddress123456789"  # Fallback for dendrite compatibility

    @property
    def hotkey(self):
        return self._hotkey

    @property
    def coldkey(self):
        return self._coldkey

    def sign(self, data: bytes) -> bytes:
        """Mock sign method to handle unexpected dendrite calls."""
        bt.logging.debug("MockWallet.sign called, returning dummy signature")
        return b"\x00" * 64  # Dummy 64-byte signature

class MockHotkey:
    """A mock hotkey class to simulate a bittensor keypair."""
    def __init__(self):
        self.ss58_address = "5MockHotkeyAddress123456789"
        self.public_key = b"\x00" * 32  # Dummy public key

    def verify(self, *args, **kwargs):
        return True  # Always pass verification in mock mode

    def sign(self, data: bytes) -> bytes:
        """Mock sign method to handle unexpected dendrite calls."""
        bt.logging.debug("MockHotkey.sign called, returning dummy signature")
        return b"\x00" * 64  # Dummy 64-byte signature

class MockSubtensor(bt.MockSubtensor):
    """Mock Bittensor subtensor for LeadPoet subnet simulation."""
    def __init__(self, netuid: int = 343, n: int = 5, wallet: Optional[bt.wallet] = None, network: str = "mock"):
        super().__init__(network=network)
        self.netuid = netuid
        self._mock_neurons = {}
        self._subnets = {}
        self._initialize_subnet(netuid)
        self._register_mock_neurons(netuid, n, wallet)
        bt.logging.info(f"MockSubtensor initialized with {n} miners and 1 validator on netuid {netuid}")
        bt.logging.debug(f"Subnet info for netuid {netuid}: {self._subnets.get(netuid, {})}")

    def _initialize_subnet(self, netuid: int):
        """Initialize subnet with default values."""
        if not self.subnet_exists(netuid):
            self.create_subnet(netuid)
        self._subnets[netuid] = {
            'owner': 'mock_owner',
            'stake': 0,
            'subnet_emission': 0,
            'tempo': 0,
            'recycle': 0,
            'created_block': 0,
            'name': b'leadpoet',
            'symbol': b'LPT',
            'network_registered_at': 0,
            'owner_hotkey': '5MockOwnerHotkey',
            'owner_coldkey': '5MockOwnerColdkey',
            'block': 0,
            'alpha_in': 0,
            'alpha_out': 0,
            'tao_in': 0,
            'tao_out': 0,
            'alpha_in_emission': 0,
            'alpha_out_emission': 0,
            'tao_in_emission': 0,
            'tao_out_emission': 0,
            'pending_alpha_emission': 0,
            'pending_root_emission': 0,
            'subnet_volume': 0,
            'moving_price': {"bits": 0},
            'rho': 0,
            'kappa': 0,
            'min_allowed_weights': 0,
            'max_weights_limit': 1000,
            'weights_version': 600,
            'weights_rate_limit': 0,
            'activity_cutoff': 5000,
            'max_validators': 100,
            'num_uids': 6,
            'max_uids': 1000,
            'burn': 0,
            'difficulty': 0,
            'registration_allowed': True,
            'pow_registration_allowed': True,
            'immunity_period': 1000,
            'min_difficulty': 0,
            'max_difficulty': 0,
            'min_burn': 0,
            'max_burn': 0,
            'adjustment_alpha': 0,
            'adjustment_interval': 0,
            'target_regs_per_interval': 0,
            'max_regs_per_block': 0,
            'serving_rate_limit': 0,
            'commit_reveal_weights_enabled': False,
            'commit_reveal_period': 0,
            'liquid_alpha_enabled': False,
            'liquid_alpha_period': 0,
            'hotkeys_permitted': [],
            'max_hotkeys': 1000,
            'min_stake': 0,
            'max_stake': 1000000,
            'emission_rate': 0,
            'consensus_period': 0,
            'incentive_period': 0,
            'trust_period': 0,
            'alpha_high': 0,
            'alpha_low': 0,
            'alpha_adjustment_rate': 0,
            'max_emission': 0,
            'min_emission': 0,
            'stake_threshold': 0,
            'max_incentive': 0,
            'min_incentive': 0,
            'validator_trust_threshold': 0,
            'pruning_enabled': False,
            'pruning_period': 0,
            'bonds_moving_avg': 0,
            'weights_moving_avg': 0,
            'dividends_moving_avg': 0,
            'emission_moving_avg': 0,
            'trust_moving_avg': 0,
            'incentive_moving_avg': 0,
            'consensus_moving_avg': 0,
            'max_bonds': 0,
            'min_bonds': 0,
            'bond_adjustment_rate': 0,
            'max_dividends': 0,
            'min_dividends': 0,
            'subnet_governance_enabled': False,
            'governance_period': 0,
            'max_governance_proposals': 0,
            'active': True,
            'registered': True,
            'locked': False,
            'total_stake': [],
            'total_emission': [],
            'stake_moving_avg': 0,
            'emission_distribution_enabled': False,
            'emission_distribution_period': 0,
            'proposal_voting_enabled': False,
            'proposal_voting_period': 0,
            'min_proposal_stake': 0,
            'max_proposal_stake': 0,
            'validator_permit': True,
            'max_permitted_validators': 100,
            'validator_stake_threshold': 0,
            'validator_registration_fee': 0,
            'min_validator_stake': 0,
            'max_validator_stake': 1000000,
            'validator_activity_cutoff': 5000,
            'consensus_threshold': 0,
            'incentive_distribution_enabled': False,
            'incentive_distribution_period': 0,
            'last_update': 0,
            'subnet_version': 0,
            'hotkey_count': 6,
            'total_neurons': 6,
            'min_validator_count': 1,
            'max_validator_count': 100,
            'subnet_state_hash': '',
            'governance_threshold': 0,
            'proposal_approval_threshold': 0,
            'subnet_metadata': {},
            'block_at_registration': 0,
            'registration_timestamp': 0,
            'subnet_creator': '5MockOwnerHotkey',
            'subnet_type': 'public',
            'emission_per_block': 0,
            'max_neurons': 1000,
            'min_neuron_stake': 0,
            'max_neuron_stake': 1000000,
            'subnet_status': 'active',
            'consensus_version': 0,
            'alpha_stake': [],
            'tao_stake': [],
            'weights': [],
            'bonds': [],
            'trust_scores': [],
            'incentive': [],
            'dividends': [],
            'ranks': [],
            'emissions': [],
            'active_neurons': 6,
            'stake_distribution': [],
            'neuron_stakes': [],
            'validator_stakes': [],
            'consensus_scores': [],
            'pruning_scores': [],
            'tao_dividends_per_hotkey': [],
            'alpha_dividends_per_hotkey': [],
            'hotkey_trust': [],
            'hotkey_incentive': [],
            'hotkey_emission': [],
            'hotkey_ranks': [],
            'hotkey_consensus': [],
            'hotkey_pruning_scores': [],
        }
        bt.logging.debug(f"Initialized subnet {netuid} with keys: {list(self._subnets[netuid].keys())}")

    def _register_mock_neurons(self, netuid: int, n: int, wallet: Optional[bt.wallet]):
        neurons = []
        wallet_hotkey = getattr(wallet, 'hotkey_ss58_address', '5MockHotkeyAddress123456789')
        wallet_coldkey = getattr(wallet, 'coldkey_ss58_address', '5MockColdkeyAddress123456789')
        if wallet is not None:
            neurons.append({
                'uid': 0,
                'netuid': netuid,
                'hotkey': wallet_hotkey,
                'coldkey': wallet_coldkey,
                'balance': 100000,
                'stake': 100000,  # Ensure stake is set
                'validator_permit': True,
                'ip': '127.0.0.1',
                'port': 8091,
                'ip_type': 4,
                'version': 600,
                'active': 1
            })
        for i in range(1, n + 1):
            neurons.append({
                'uid': i,
                'netuid': netuid,
                'hotkey': f"miner-hotkey-{i}",
                'coldkey': f"mock-coldkey-{i}",
                'balance': 100000,
                'stake': 100000,  # Ensure stake is set
                'validator_permit': False,
                'ip': '127.0.0.1',
                'port': 8091 + i,
                'ip_type': 4,
                'version': 600,
                'active': 1
            })
        self._mock_neurons[netuid] = neurons
        if not hasattr(self, '_neurons'):
            self._neurons = {}
        self._neurons[netuid] = neurons
        bt.logging.debug(f"Registered {len(neurons)} mock neurons for netuid {netuid}")

    def get_neurons(self, netuid: int) -> List[Dict]:
        return self._mock_neurons.get(netuid, [])

    def neurons_lite(self, netuid: int, block: Optional[int] = None) -> List['bt.NeuronInfo']:
        if netuid not in self._mock_neurons:
            raise Exception("Subnet does not exist")
        neurons = []
        for neuron_dict in self._mock_neurons[netuid]:
            neuron = bt.NeuronInfo(
                uid=neuron_dict['uid'],
                netuid=neuron_dict['netuid'],
                active=neuron_dict['active'],
                stake=neuron_dict['stake'],
                stake_dict={neuron_dict['hotkey']: neuron_dict['stake']},
                total_stake=neuron_dict['stake'],
                rank=0.0,
                emission=0.0,
                incentive=0.0,
                consensus=0.0,
                trust=0.0,
                validator_trust=0.0,
                dividends=0.0,
                pruning_score=0.0,
                hotkey=neuron_dict['hotkey'],
                coldkey=neuron_dict['coldkey'],
                axon_info=bt.AxonInfo(
                    version=neuron_dict['version'],
                    ip=neuron_dict['ip'],
                    port=neuron_dict['port'],
                    ip_type=neuron_dict['ip_type'],
                    hotkey=neuron_dict['hotkey'],
                    coldkey=neuron_dict['coldkey']
                ),
                weights=[],
                bonds=[],
                last_update=0,
                validator_permit=neuron_dict['validator_permit'],
                prometheus_info=None
            )
            neurons.append(neuron)
        bt.logging.debug(f"Returning {len(neurons)} neurons for netuid {netuid}")
        return neurons

    def query_runtime_api(self, *args, **kwargs) -> None:
        bt.logging.debug("MockSubtensor: query_runtime_api called, no action taken in mock mode")
        return None

    def get_metagraph_info(self, netuid: int, block: Optional[int] = None) -> 'bt.MetagraphInfo':
        """Mock implementation of get_metagraph_info."""
        if not self.subnet_exists(netuid):
            raise Exception("Subnet does not exist")
        subnet_info = self._subnets.get(netuid)
        if not subnet_info:
            bt.logging.error(f"No subnet info found for netuid {netuid}")
            self._initialize_subnet(netuid)
            subnet_info = self._subnets[netuid]
        metagraph_dict = {
            'netuid': netuid,
            'owner': subnet_info.get('owner', 'mock_owner'),
            'stake': subnet_info.get('stake', 0),
            'subnet_emission': subnet_info.get('subnet_emission', 0),
            'tempo': subnet_info.get('tempo', 0),
            'recycle': subnet_info.get('recycle', 0),
            'created_block': subnet_info.get('created_block', 0),
            'name': subnet_info.get('name', b'leadpoet'),
            'symbol': subnet_info.get('symbol', b'LPT'),
            'network_registered_at': subnet_info.get('network_registered_at', 0),
            'owner_hotkey': subnet_info.get('owner_hotkey', '5MockOwnerHotkey'),
            'owner_coldkey': subnet_info.get('owner_coldkey', '5MockOwnerColdkey'),
            'block': block if block is not None else subnet_info.get('block', 0),
            'last_step': 0,
            'blocks_since_last_step': 0,
            'alpha_in': subnet_info.get('alpha_in', 0),
            'alpha_out': subnet_info.get('alpha_out', 0),
            'tao_in': subnet_info.get('tao_in', 0),
            'tao_out': subnet_info.get('tao_out', 0),
            'alpha_in_emission': subnet_info.get('alpha_in_emission', 0),
            'alpha_out_emission': subnet_info.get('alpha_out_emission', 0),
            'tao_in_emission': subnet_info.get('tao_in_emission', 0),
            'tao_out_emission': subnet_info.get('tao_out_emission', 0),
            'pending_alpha_emission': subnet_info.get('pending_alpha_emission', 0),
            'pending_root_emission': subnet_info.get('pending_root_emission', 0),
            'subnet_volume': subnet_info.get('subnet_volume', 0),
            'moving_price': subnet_info.get('moving_price', {"bits": 0}),
            'rho': subnet_info.get('rho', 0),
            'kappa': subnet_info.get('kappa', 0),
            'min_allowed_weights': subnet_info.get('min_allowed_weights', 0),
            'max_weights_limit': subnet_info.get('max_weights_limit', 1000),
            'weights_version': subnet_info.get('weights_version', 600),
            'weights_rate_limit': subnet_info.get('weights_rate_limit', 0),
            'activity_cutoff': subnet_info.get('activity_cutoff', 5000),
            'max_validators': subnet_info.get('max_validators', 100),
            'num_uids': subnet_info.get('num_uids', 6),
            'max_uids': subnet_info.get('max_uids', 1000),
            'burn': subnet_info.get('burn', 0),
            'difficulty': subnet_info.get('difficulty', 0),
            'registration_allowed': subnet_info.get('registration_allowed', True),
            'pow_registration_allowed': subnet_info.get('pow_registration_allowed', True),
            'immunity_period': subnet_info.get('immunity_period', 1000),
            'min_difficulty': subnet_info.get('min_difficulty', 0),
            'max_difficulty': subnet_info.get('max_difficulty', 0),
            'min_burn': subnet_info.get('min_burn', 0),
            'max_burn': subnet_info.get('max_burn', 0),
            'adjustment_alpha': subnet_info.get('adjustment_alpha', 0),
            'adjustment_interval': subnet_info.get('adjustment_interval', 0),
            'target_regs_per_interval': subnet_info.get('target_regs_per_interval', 0),
            'max_regs_per_block': subnet_info.get('max_regs_per_block', 0),
            'serving_rate_limit': subnet_info.get('serving_rate_limit', 0),
            'commit_reveal_weights_enabled': subnet_info.get('commit_reveal_weights_enabled', False),
            'commit_reveal_period': subnet_info.get('commit_reveal_period', 0),
            'liquid_alpha_enabled': subnet_info.get('liquid_alpha_enabled', False),
            'liquid_alpha_period': subnet_info.get('liquid_alpha_period', 0),
            'hotkeys_permitted': subnet_info.get('hotkeys_permitted', []),
            'max_hotkeys': subnet_info.get('max_hotkeys', 1000),
            'min_stake': subnet_info.get('min_stake', 0),
            'max_stake': subnet_info.get('max_stake', 1000000),
            'emission_rate': subnet_info.get('emission_rate', 0),
            'consensus_period': subnet_info.get('consensus_period', 0),
            'incentive_period': subnet_info.get('incentive_period', 0),
            'trust_period': subnet_info.get('trust_period', 0),
            'alpha_high': subnet_info.get('alpha_high', 0),
            'alpha_low': subnet_info.get('alpha_low', 0),
            'alpha_adjustment_rate': subnet_info.get('alpha_adjustment_rate', 0),
            'max_emission': subnet_info.get('max_emission', 0),
            'min_emission': subnet_info.get('min_emission', 0),
            'stake_threshold': subnet_info.get('stake_threshold', 0),
            'max_incentive': subnet_info.get('max_incentive', 0),
            'min_incentive': subnet_info.get('min_incentive', 0),
            'validator_trust_threshold': subnet_info.get('validator_trust_threshold', 0),
            'pruning_enabled': subnet_info.get('pruning_enabled', False),
            'pruning_period': subnet_info.get('pruning_period', 0),
            'bonds_moving_avg': subnet_info.get('bonds_moving_avg', 0),
            'weights_moving_avg': subnet_info.get('weights_moving_avg', 0),
            'dividends_moving_avg': subnet_info.get('dividends_moving_avg', 0),
            'emission_moving_avg': subnet_info.get('emission_moving_avg', 0),
            'trust_moving_avg': subnet_info.get('trust_moving_avg', 0),
            'incentive_moving_avg': subnet_info.get('incentive_moving_avg', 0),
            'consensus_moving_avg': subnet_info.get('consensus_moving_avg', 0),
            'max_bonds': subnet_info.get('max_bonds', 0),
            'min_bonds': subnet_info.get('min_bonds', 0),
            'bond_adjustment_rate': subnet_info.get('bond_adjustment_rate', 0),
            'max_dividends': subnet_info.get('max_dividends', 0),
            'min_dividends': subnet_info.get('min_dividends', 0),
            'subnet_governance_enabled': subnet_info.get('subnet_governance_enabled', False),
            'governance_period': subnet_info.get('governance_period', 0),
            'max_governance_proposals': subnet_info.get('max_governance_proposals', 0),
            'active': subnet_info.get('active', True),
            'registered': subnet_info.get('registered', True),
            'locked': subnet_info.get('locked', False),
            'total_stake': subnet_info.get('total_stake', []),
            'total_emission': subnet_info.get('total_emission', []),
            'stake_moving_avg': subnet_info.get('stake_moving_avg', 0),
            'emission_distribution_enabled': subnet_info.get('emission_distribution_enabled', False),
            'emission_distribution_period': subnet_info.get('emission_distribution_period', 0),
            'proposal_voting_enabled': subnet_info.get('proposal_voting_enabled', False),
            'proposal_voting_period': subnet_info.get('proposal_voting_period', 0),
            'min_proposal_stake': subnet_info.get('min_proposal_stake', 0),
            'max_proposal_stake': subnet_info.get('max_proposal_stake', 0),
            'validator_permit': subnet_info.get('validator_permit', True),
            'max_permitted_validators': subnet_info.get('max_permitted_validators', 100),
            'validator_stake_threshold': subnet_info.get('validator_stake_threshold', 0),
            'validator_registration_fee': subnet_info.get('validator_registration_fee', 0),
            'min_validator_stake': subnet_info.get('min_validator_stake', 0),
            'max_validator_stake': subnet_info.get('max_validator_stake', 1000000),
            'validator_activity_cutoff': subnet_info.get('validator_activity_cutoff', 5000),
            'consensus_threshold': subnet_info.get('consensus_threshold', 0),
            'incentive_distribution_enabled': subnet_info.get('incentive_distribution_enabled', False),
            'incentive_distribution_period': subnet_info.get('incentive_distribution_period', 0),
            'last_update': subnet_info.get('last_update', 0),
            'subnet_version': subnet_info.get('subnet_version', 0),
            'hotkey_count': subnet_info.get('hotkey_count', 6),
            'total_neurons': subnet_info.get('total_neurons', 6),
            'min_validator_count': subnet_info.get('min_validator_count', 1),
            'max_validator_count': subnet_info.get('max_validator_count', 100),
            'subnet_state_hash': subnet_info.get('subnet_state_hash', ''),
            'governance_threshold': subnet_info.get('governance_threshold', 0),
            'proposal_approval_threshold': subnet_info.get('proposal_approval_threshold', 0),
            'subnet_metadata': subnet_info.get('subnet_metadata', {}),
            'block_at_registration': subnet_info.get('block_at_registration', 0),
            'registration_timestamp': subnet_info.get('registration_timestamp', 0),
            'subnet_creator': subnet_info.get('subnet_creator', '5MockOwnerHotkey'),
            'subnet_type': subnet_info.get('subnet_type', 'public'),
            'emission_per_block': subnet_info.get('emission_per_block', 0),
            'max_neurons': subnet_info.get('max_neurons', 1000),
            'min_neuron_stake': subnet_info.get('min_neuron_stake', 0),
            'max_neuron_stake': subnet_info.get('max_neuron_stake', 1000000),
            'subnet_status': subnet_info.get('subnet_status', 'active'),
            'consensus_version': subnet_info.get('consensus_version', 0),
            'alpha_stake': subnet_info.get('alpha_stake', []),
            'tao_stake': subnet_info.get('tao_stake', []),
            'weights': subnet_info.get('weights', []),
            'bonds': subnet_info.get('bonds', []),
            'trust_scores': subnet_info.get('trust_scores', []),
            'incentive': subnet_info.get('incentive', []),
            'dividends': subnet_info.get('dividends', []),
            'ranks': subnet_info.get('ranks', []),
            'emissions': subnet_info.get('emissions', []),
            'active_neurons': subnet_info.get('active_neurons', 6),
            'stake_distribution': subnet_info.get('stake_distribution', []),
            'neuron_stakes': subnet_info.get('neuron_stakes', []),
            'validator_stakes': subnet_info.get('validator_stakes', []),
            'consensus_scores': subnet_info.get('consensus_scores', []),
            'pruning_scores': subnet_info.get('pruning_scores', []),
            'tao_dividends_per_hotkey': subnet_info.get('tao_dividends_per_hotkey', []),
            'alpha_dividends_per_hotkey': subnet_info.get('alpha_dividends_per_hotkey', []),
            'hotkey_trust': subnet_info.get('hotkey_trust', []),
            'hotkey_incentive': subnet_info.get('hotkey_incentive', []),
            'hotkey_emission': subnet_info.get('hotkey_emission', []),
            'hotkey_ranks': subnet_info.get('hotkey_ranks', []),
            'hotkey_consensus': subnet_info.get('hotkey_consensus', []),
            'hotkey_pruning_scores': subnet_info.get('hotkey_pruning_scores', []),
        }
        bt.logging.debug(f"Returning metagraph info for netuid {netuid}: {metagraph_dict}")
        return bt.MetagraphInfo.from_dict(metagraph_dict)

class MockMetagraph(bt.metagraph):
    def __init__(self, netuid: int = 343, network: str = "mock", subtensor: Optional[bt.MockSubtensor] = None):
        super().__init__(netuid=netuid, network=network, sync=False)
        if subtensor is not None:
            self.subtensor = subtensor
        self.sync(subtensor=subtensor)
        # Initialize stake array
        self.S = np.array([neuron.stake for neuron in self.neurons], dtype=np.float32)
        for axon in self.axons:
            axon.ip = "127.0.0.1"
            axon.port = 8091 + self.axons.index(axon)
        bt.logging.info(f"MockMetagraph: {self}")
        bt.logging.debug(f"Axons: {self.axons}")
        bt.logging.debug(f"Stake array (S): {self.S}")

class MockDendrite(bt.dendrite):
    """Mock dendrite for LeadPoet subnet, simulating miner responses."""
    def __init__(self, wallet: bt.wallet, use_open_source: bool = True):
        super().__init__(wallet)
        self._wallet = wallet
        self.use_open_source = use_open_source
        self.email_counter = 0
        self._session = None
        bt.logging.debug(f"MockDendrite initialized with wallet: {self._wallet}")

    def generate_dummy_lead(self) -> Dict:
        """Generate a single dummy lead."""
        self.email_counter += 1
        lead = {
            "Business": f"Mock Business {self.email_counter}",
            "Owner Full name": f"Owner {self.email_counter}",
            "First": f"First {self.email_counter}",
            "Last": f"Last {self.email_counter}",
            "Owner(s) Email": f"owner{self.email_counter}@mockleadpoet.com",
            "LinkedIn": f"https://linkedin.com/in/owner{self.email_counter}",
            "Website": f"https://business{self.email_counter}.com",
            "Industry": random.choice(["Tech & AI", "Finance & Fintech", "Health & Wellness", "Media & Education", "Energy & Industry"]),
            "Region": random.choice(["US", "EU", "Asia", "Global"])
        }
        bt.logging.debug(f"Generated dummy lead: {lead}")
        return lead

    def preprocess_synapse_for_request(self, axon: bt.AxonInfo, synapse: bt.Synapse, timeout: float) -> bt.Synapse:
        """Override preprocessing for mock wallet."""
        try:
            # Use wallet's hotkey_ss58_address or fallback
            wallet_hotkey = getattr(self._wallet, 'hotkey_ss58_address', '5MockHotkeyAddress123456789')
            synapse.dendrite = bt.TerminalInfo(
                hotkey=wallet_hotkey,
                version=600,
                ip="0.0.0.0",
                port=0,
                process_time=0.0,
                status_code=200,
                status_message="OK"
            )
            synapse.axon = bt.TerminalInfo(
                hotkey=axon.hotkey,
                version=600,
                ip=axon.ip,
                port=axon.port,
                process_time=0.0,
                status_code=200,
                status_message="OK"
            )
            bt.logging.debug(f"Preprocessed synapse for axon {axon.hotkey}")
            return synapse
        except Exception as e:
            bt.logging.error(f"Error in preprocess_synapse_for_request: {e}")
            raise

    async def forward(
        self,
        axons: List[bt.AxonInfo],
        synapse: bt.Synapse = LeadRequest(num_leads=1),
        timeout: float = 30,
        deserialize: bool = True,
        run_async: bool = True,
        streaming: bool = False,
    ) -> List[LeadRequest]:
        """Simulate miner responses."""
        bt.logging.debug(f"MockDendrite.forward called with {len(axons)} axons, synapse={synapse}")
        if streaming:
            raise NotImplementedError("Streaming not implemented.")
        if not isinstance(synapse, LeadRequest):
            raise ValueError("Synapse must be a LeadRequest")

        async def query_axon(axon: bt.AxonInfo) -> LeadRequest:
            bt.logging.debug(f"Querying axon {axon.hotkey} at {axon.ip}:{axon.port}")
            start_time = time.time()
            response = synapse.copy()
            try:
                response = self.preprocess_synapse_for_request(axon, response, timeout)
                process_time = random.uniform(0.5, min(5, timeout))
                bt.logging.debug(f"Simulating process_time={process_time:.2f}s for axon {axon.hotkey}")
                await asyncio.sleep(process_time)
                if process_time >= timeout:
                    bt.logging.warning(f"Timeout for axon {axon.hotkey}")
                    response.leads = []
                    response.dendrite.status_code = 408
                    response.dendrite.status_message = "Timeout"
                    response.dendrite.process_time = str(timeout)
                else:
                    if self.use_open_source:
                        try:
                            bt.logging.debug(f"Calling get_leads for {synapse.num_leads} leads, industry={synapse.industry}, region={synapse.region}")
                            leads = await get_leads(synapse.num_leads, synapse.industry, synapse.region)
                            normalized_leads = [
                                {
                                    "Business": lead.get("Business", "Unknown"),
                                    "Owner Full name": lead.get("Owner Full name", "Unknown"),
                                    "First": lead.get("First", "Unknown"),
                                    "Last": lead.get("Last", "Unknown"),
                                    "Owner(s) Email": lead.get("Owner(s) Email", "unknown@mock.com"),
                                    "LinkedIn": lead.get("LinkedIn", ""),
                                    "Website": lead.get("Website", ""),
                                    "Industry": lead.get("Industry", "Unknown"),
                                    "Region": synapse.region or "Global"
                                } for lead in leads
                            ]
                            bt.logging.debug(f"Generated {len(normalized_leads)} leads for axon {axon.hotkey}")
                        except Exception as e:
                            bt.logging.warning(f"Open-source lead generation failed for axon {axon.hotkey}: {e}. Falling back to dummy leads.")
                            normalized_leads = [self.generate_dummy_lead() for _ in range(synapse.num_leads)]
                    else:
                        normalized_leads = [self.generate_dummy_lead() for _ in range(synapse.num_leads)]
                        bt.logging.debug(f"Generated {len(normalized_leads)} dummy leads for axon {axon.hotkey}")
                    response.leads = normalized_leads
                    response.dendrite.status_code = 200
                    response.dendrite.status_message = "OK"
                    response.dendrite.process_time = str(process_time)
            except Exception as e:
                bt.logging.error(f"Error processing axon {axon.hotkey}: {e}")
                response.leads = []
                response.dendrite.status_code = 500
                response.dendrite.status_message = f"Error: {e}"
                response.dendrite.process_time = str(time.time() - start_time)
            return response

        try:
            responses = await asyncio.gather(*(query_axon(axon) for axon in axons), return_exceptions=True)
            valid_responses = [resp for resp in responses if not isinstance(resp, Exception)]
            bt.logging.debug(f"Processed {len(responses)} responses, {len(valid_responses)} valid")
            for resp in responses:
                if isinstance(resp, Exception):
                    bt.logging.error(f"Exception in query_axon: {resp}")
            if not valid_responses:
                bt.logging.warning("No valid responses, generating a single dummy response")
                response = synapse.copy()
                response.leads = [self.generate_dummy_lead() for _ in range(synapse.num_leads)]
                response.dendrite.status_code = 200
                response.dendrite.status_message = "OK"
                response.dendrite.process_time = "0.5"
                valid_responses.append(response)
            return valid_responses
        except Exception as e:
            bt.logging.error(f"Error in forward: {e}")
            return []

    def __str__(self) -> str:
        hotkey_address = getattr(self._wallet, 'hotkey_ss58_address', 'unknown')
        return f"MockDendrite({hotkey_address}, use_open_source={self.use_open_source})"