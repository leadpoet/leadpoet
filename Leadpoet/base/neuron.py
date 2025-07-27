import bittensor as bt
import os
import sys
import argparse

class BaseNeuron:
    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        pass

    def __init__(self, config=None):
        if config is None:
            config = bt.config()
        self.config = config
        self.wallet = bt.wallet(config=self.config)
        bt.logging.debug("Initializing subtensor for real network")
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
        self.block = self.subtensor.get_current_block()

    def sync(self):
        self.metagraph.sync(subtensor=self.subtensor)