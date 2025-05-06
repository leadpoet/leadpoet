import numpy as np
import bittensor as bt
import os
import sys
import argparse

class MockAxonInfo:
    def __init__(self, uid):
        self.ip = "127.0.0.1"
        self.port = 8091 + uid
        self.hotkey = f"mock_hotkey_{uid}"
        self.coldkey = f"mock_coldkey_{uid}"
        self.version = 600
        self.ip_type = 4
        self.is_serving = True

class MockNeuron:
    def __init__(self, uid, stake, hotkey, validator_permit=False):
        self.uid = uid
        self.stake = stake
        self.hotkey = hotkey
        self.trust = 0.0
        self.validator_trust = 0.0
        self.rank = 0.0
        self.consensus = 0.0
        self.incentive = 0.0
        self.emission = 0.0
        self.dividends = 0.0
        self.last_update = 1
        self.active = True
        self.validator_permit = validator_permit
        self.axon_info = MockAxonInfo(uid)

class MockSubtensor:
    def __init__(self, netuid=343):
        self.netuid = netuid
        self.chain_endpoint = None
        self.subnets = {netuid: {'stake': 0, 'block': 1}}
        self._neurons = {
            netuid: [
                MockNeuron(uid=0, stake=20.0, hotkey="5CJyMxw6YJJvLhPf58gSpMB7mvSKSCMx9RXhXJum6cNfqMEz", validator_permit=True),
                MockNeuron(uid=1, stake=2.0, hotkey="5D73anXA8XELS2tSjnGQKMoVog1vKuTQCoJHrEGaXpZBAWpS"),
                MockNeuron(uid=2, stake=2.0, hotkey="mock_hotkey_2"),
                MockNeuron(uid=3, stake=2.0, hotkey="mock_hotkey_3"),
                MockNeuron(uid=4, stake=2.0, hotkey="mock_hotkey_4"),
            ]
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
    ss58_address = "5MockHotkeyAddress123456789"

class MockColdkey:
    ss58_address = "5MockColdkeyAddress123456789"

class MockKeyfile:
    def exists(self):
        return True

class BaseNeuron:
    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        pass

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
                config.netuid = 343
            config.subtensor.mock_subnets = {config.netuid: {'stake': 0, 'block': 1}}
            config.subtensor.mock_neurons = {
                config.netuid: [{'uid': i, 'stake': 2.0 if i > 0 else 20.0, 'hotkey': f'mock_hotkey_{i}'} for i in range(5)]
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
                if not hasattr(self.config, 'subtensor') or not hasattr(self.config.subtensor, 'chain_endpoint'):
                    self.config.subtensor = bt.Config()
                    self.config.subtensor.network = "test"
                    self.config.subtensor.chain_endpoint = "wss://test.finney.opentensor.ai:443"
                self.subtensor = bt.subtensor(config=self.config)
                bt.logging.info(f"Subtensor initialized, endpoint: {self.subtensor.chain_endpoint}, network: {self.config.subtensor.network}")
            except Exception as e:
                bt.logging.error(f"Failed to initialize bt.subtensor: {e}")
                raise RuntimeError(f"Subtensor initialization failed: {e}")
        self.metagraph = bt.metagraph(netuid=self.config.netuid, subtensor=self.subtensor)
        self.step = 0
        self.block = self.subtensor.get_current_block() if not is_mock else 1

    def sync(self):
        self.metagraph.sync(subtensor=self.subtensor)