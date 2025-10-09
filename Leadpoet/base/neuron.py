import bittensor as bt
import os
import sys
import argparse
import asyncio

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
        self.should_exit = False
        self.is_running = False
        self.thread = None
        self.lock = asyncio.Lock()

    def config_neuron(self, path: str):
        if not hasattr(self.config, 'neuron') or self.config.neuron is None:
            self.config.neuron = bt.Config()
            self.config.neuron.axon_off = False
            self.config.neuron.num_concurrent_forwards = 1
            self.config.neuron.full_path = path
            self.config.neuron.moving_average_alpha = 0.1
            self.config.neuron.sample_size = 5
            bt.logging.debug("Initialized config.neuron with defaults")

    def config_axon(self, port: int):
        if not hasattr(self.config, 'axon') or self.config.axon is None:
            self.config.axon = bt.Config()
            self.config.axon.ip = "0.0.0.0"
            self.config.axon.port = port
            bt.logging.debug("Initialized config.axon with default values")

    def sync(self):
        self.metagraph.sync(subtensor=self.subtensor)