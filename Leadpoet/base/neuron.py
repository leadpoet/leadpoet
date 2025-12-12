import bittensor as bt
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
        
        # ════════════════════════════════════════════════════════════
        # PROXY BYPASS FOR BITTENSOR WEBSOCKET
        # ════════════════════════════════════════════════════════════
        # Workers use HTTP proxies for API calls (different IPs), but websocket
        # connections to Bittensor must bypass the proxy (proxies don't support websockets)
        # Temporarily unset proxy env vars for Bittensor init, then restore them
        import os
        proxy_env_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']
        saved_proxies = {}
        for var in proxy_env_vars:
            if var in os.environ:
                saved_proxies[var] = os.environ[var]
                del os.environ[var]
        
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
        finally:
            # Restore proxy environment variables for API calls
            for var, value in saved_proxies.items():
                os.environ[var] = value
        
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