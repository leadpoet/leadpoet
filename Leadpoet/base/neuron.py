import re
import random
import numpy as np
import bittensor as bt
import os
import sys
from Leadpoet.validator.forward import forward


class MockAxonInfo:
    """Simple mock AxonInfo class to mimic bittensor.AxonInfo."""
    def __init__(self):
        self.ip = "0.0.0.0"
        self.port = 0
        self.hotkey = "mock_hotkey"
        self.coldkey = "mock_coldkey"
        self.version = 1
        self.ip_type = 4  # IPv4


class MockNeuron:
    """Simple mock neuron class to mimic NeuronInfoLite."""
    def __init__(self, uid, stake, hotkey):
        self.uid = uid
        self.stake = stake
        self.hotkey = hotkey
        self.trust = 0.0
        self.rank = 0.0
        self.consensus = 0.0
        self.incentive = 0.0
        self.emission = 0.0
        self.dividends = 0.0
        self.last_update = 1
        self.active = True
        self.validator_permit = False
        self.validator_trust = 0.0
        self.axon_info = MockAxonInfo()


class MockSubtensor:
    """Minimal mock Subtensor class to bypass connection attempts."""
    def __init__(self, netuid=33):
        self.netuid = netuid
        self.chain_endpoint = None  # Mimic bt.subtensor with no real endpoint
        self.subnets = {netuid: {'stake': 0, 'block': 1}}
        self._neurons = {
            netuid: [MockNeuron(uid=i, stake=0, hotkey=f'mock_hotkey_{i}') for i in range(5)]
        }
        self.block = 1

    def neurons_lite(self, netuid=None, block=None):
        if netuid is None:
            return self._neurons[self.netuid]
        if netuid in self.subnets:
            return self._neurons.get(netuid, [])
        raise Exception("Subnet does not exist")

    def subnet_exists(self, netuid):
        return netuid in self.subnets

    def get_current_block(self):
        return self.block  # Return a static mock block number

    def get_metagraph_info(self, netuid, block=None):
        return None

    @property
    def is_mock(self):
        return True


class MockWallet:
    """Simple mock wallet class to bypass keyfile requirements in mock mode."""
    def __init__(self):
        self.hotkey = MockHotkey()
        self.coldkey = MockColdkey()

    @property
    def hotkey_file(self):
        return MockKeyfile()

    @property
    def coldkey_file(self):
        return MockKeyfile()


class MockHotkey:
    """Mock hotkey with a dummy SS58 address."""
    ss58_address = "5MockHotkeyAddress123456789"


class MockColdkey:
    """Mock coldkey with a dummy SS58 address."""
    ss58_address = "5MockColdkeyAddress123456789"


class MockKeyfile:
    """Mock keyfile to simulate existence."""
    def exists(self):
        return True


class BaseNeuron:
    """Base class for all neurons in the Leadpoet subnet."""
    def __init__(self, config=None):
        # If no config, create one and check for --mock in sys.argv
        if config is None:
            config = bt.config()
        # Force mock mode if --mock is in command-line args
        is_mock = '--mock' in sys.argv
        if is_mock:
            if not hasattr(config, 'subtensor') or config.subtensor is None:
                config.subtensor = bt.Config()
            config.mock = True
            config.subtensor._mock = True
            config.subtensor.mock = True
            config.subtensor.chain_endpoint = None  # Prevent connection attempts
            config.subtensor.network = "mock_network"
            if not hasattr(config, 'netuid') or config.netuid is None:
                config.netuid = 33  # Default netuid
            # Ensure mock subnet exists
            config.subtensor.mock_subnets = {33: {'stake': 0, 'block': 1}}
            config.subtensor.mock_neurons = {
                33: [{'uid': i, 'stake': 0, 'hotkey': f'mock_hotkey_{i}'} for i in range(5)]
            }
            # Ensure blacklist is initialized
            if not hasattr(config, 'blacklist') or config.blacklist is None:
                config.blacklist = bt.Config()
            config.blacklist.force_validator_permit = False
            # Ensure axon is initialized
            if not hasattr(config, 'axon') or config.axon is None:
                config.axon = bt.Config()
            config.axon.ip = "0.0.0.0"  # Default mock IP
            config.axon.port = 8091     # Default port (common for bittensor axons)
        self.config = config
        # Set wallet before any parent init calls
        self.wallet = MockWallet() if is_mock else bt.wallet(config=self.config)
        # Use bt.subtensor with mocked config, fallback to custom mock if needed
        try:
            self.subtensor = bt.subtensor(config=self.config)
            bt.logging.debug(f"Subtensor mock: {self.subtensor.is_mock}, endpoint: {self.subtensor.chain_endpoint}")
        except Exception as e:
            bt.logging.warning(f"bt.subtensor failed with {e}, using custom MockSubtensor")
            self.subtensor = MockSubtensor(netuid=self.config.netuid)
        self.metagraph = bt.metagraph(netuid=self.config.netuid, subtensor=self.subtensor)
        self.step = 0

    def sync(self):
        """Syncs the metagraph with the network."""
        self.metagraph.sync(subtensor=self.subtensor)


class Validator:
    """Validator neuron class for the LeadPoet subnet."""
    def __init__(self, config=None):
        from Leadpoet.base.validator import BaseValidatorNeuron
        super().__init__(BaseValidatorNeuron, self, config=config)
        bt.logging.info("load_state()")
        self.load_state()
        self.email_regex = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        self.sample_ratio = 0.2
        self.reputation_score = 0
        self.consistency_streak = 0

    def validate_email(self, email: str) -> bool:
        return bool(self.email_regex.match(email))

    def check_duplicates(self, leads: list) -> set:
        emails = [lead.get('Owner(s) Email', '') for lead in leads]
        seen = set()
        duplicates = set(email for email in emails if email in seen or seen.add(email))
        return duplicates

    def simulate_manual_review(self, leads: list, sample_size: int) -> float:
        if not leads:
            return 0.0
        valid_emails = sum(1 for lead in leads[:sample_size] if self.validate_email(lead.get('Owner(s) Email', '')))
        base_rate = random.uniform(0.8, 1.0) * (valid_emails / sample_size)
        return base_rate

    async def forward(self):
        await forward(self, post_process=self._update_reputation_and_consistency)

    def _update_reputation_and_consistency(self, rewards: np.ndarray, miner_uids: list):
        avg_reward = np.mean(rewards) if rewards.size > 0 else 0
        if avg_reward >= 0.9:
            self.reputation_score += 5
            self.consistency_streak += 1
        else:
            self.reputation_score -= 10
            self.consistency_streak = 0
        self.consistency_factor = min(1 + 0.025 * self.consistency_streak, 2.0)
        bt.logging.debug(f"Reputation: {self.reputation_score}, Consistency Factor: {self.consistency_factor}")

    def save_state(self):
        bt.logging.info("Saving validator state.")
        state_path = os.path.join(self.config.neuron.full_path, "validator_state.npz")
        np.savez(
            state_path,
            step=self.step,
            scores=self.scores,
            hotkeys=self.hotkeys,
            reputation_score=self.reputation_score,
            consistency_streak=self.consistency_streak,
            consistency_factor=self.consistency_factor
        )

    def load_state(self):
        state_path = os.path.join(self.config.neuron.full_path, "validator_state.npz")
        if os.path.exists(state_path):
            bt.logging.info("Loading validator state.")
            try:
                state = np.load(state_path)
                self.step = state["step"]
                self.scores = state["scores"]
                self.hotkeys = state["hotkeys"]
                self.reputation_score = state["reputation_score"]
                self.consistency_streak = state["consistency_streak"]
                self.consistency_factor = state["consistency_factor"]
            except Exception as e:
                bt.logging.warning(f"Failed to load state: {e}. Using defaults.")
        else:
            bt.logging.info("No state file found. Initializing with defaults.")
            self.step = 0
            self.scores = np.zeros(self.metagraph.n, dtype=np.float32)
            self.hotkeys = self.metagraph.hotkeys.copy()
            self.reputation_score = 0
            self.consistency_streak = 0
            self.consistency_factor = 1.0


if __name__ == "__main__":
    import time
    with Validator() as validator:
        while True:
            bt.logging.info(f"Validator running... {time.time()}")
            time.sleep(5)