import numpy as np
import bittensor as bt
import os
import sys
import argparse  # Added for add_args

class MockAxonInfo:
    """Simple mock AxonInfo class to mimic bittensor.AxonInfo."""
    def __init__(self):
        self.ip = "0.0.0.0"
        self.port = 0
        self.hotkey = "mock_hotkey"
        self.coldkey = "mock_coldkey"
        self.version = 1
        self.ip_type = 4
        self.is_serving = True

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
    def __init__(self, netuid=343):  # Default to command-line netuid
        self.netuid = netuid
        self.chain_endpoint = None
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
        return self.block

    def get_metagraph_info(self, netuid, block=None):
        return None

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
    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        """Basic argument parser method for inheritance."""
        pass  # Minimal implementation to satisfy super() call

    def __init__(self, config=None):
        if config is None:
            config = bt.config()
        is_mock = getattr(config, 'mock', False) or '--mock' in sys.argv
        if is_mock:
            if not hasattr(config, 'subtensor') or config.subtensor is None:
                config.subtensor = bt.Config()
            config.mock = True
            config.subtensor.mock = True
            config.subtensor.chain_endpoint = None
            config.subtensor.network = "mock_network"
            if not hasattr(config, 'netuid') or config.netuid is None:
                config.netuid = 343  # Match command-line netuid
            config.subtensor.mock_subnets = {config.netuid: {'stake': 0, 'block': 1}}
            config.subtensor.mock_neurons = {
                config.netuid: [{'uid': i, 'stake': 0, 'hotkey': f'mock_hotkey_{i}'} for i in range(5)]
            }
            if not hasattr(config, 'blacklist') or config.blacklist is None:
                config.blacklist = bt.Config()
            config.blacklist.force_validator_permit = False
            if not hasattr(config, 'axon') or config.axon is None:
                config.axon = bt.Config()
            config.axon.ip = "0.0.0.0"
            config.axon.port = 8091
        self.config = config
        self.wallet = MockWallet() if is_mock else bt.wallet(config=self.config)
        bt.logging.debug(f"Initializing subtensor, mock mode: {is_mock}")
        if is_mock:
            self.subtensor = MockSubtensor(netuid=self.config.netuid)
            bt.logging.info(f"Using MockSubtensor for netuid {self.config.netuid}")
        else:
            try:
                # Explicitly set testnet endpoint if not configured
                if not hasattr(self.config, 'subtensor') or not hasattr(self.config.subtensor, 'chain_endpoint'):
                    self.config.subtensor = bt.Config()
                    self.config.subtensor.network = "test"
                    self.config.subtensor.chain_endpoint = "ws://testnet-finch.opentensor.ai:9944"
                self.subtensor = bt.subtensor(config=self.config)
                bt.logging.info(f"Subtensor initialized, endpoint: {self.subtensor.chain_endpoint}, network: {self.config.subtensor.network}")
            except Exception as e:
                bt.logging.error(f"Failed to initialize bt.subtensor: {e}")
                raise RuntimeError(f"Subtensor initialization failed: {e}")
        self.metagraph = bt.metagraph(netuid=self.config.netuid, subtensor=self.subtensor)
        self.step = 0
        self.block = self.subtensor.get_current_block() if not is_mock else 1

    def sync(self):
        """Syncs the metagraph with the network."""
        self.metagraph.sync(subtensor=self.subtensor)