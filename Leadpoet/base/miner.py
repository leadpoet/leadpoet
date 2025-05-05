import time
import asyncio
import threading
import argparse
import traceback
import bittensor as bt
from Leadpoet.base.neuron import BaseNeuron
from typing import Union

class BaseMinerNeuron(BaseNeuron):
   
    neuron_type: str = "MinerNeuron"

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
    
        parser.add_argument(
            "--netuid",
            type=int,
            help="The network UID of the subnet to connect to",
            default=343
        )
        parser.add_argument(
            "--subtensor_network",
            type=str,
            help="The network to connect to (e.g., test, main)",
            default="test"
        )
        parser.add_argument(
            "--wallet_name",
            type=str,
            help="The name of the wallet to use",
            required=True
        )
        parser.add_argument(
            "--wallet_hotkey",
            type=str,
            help="The hotkey of the wallet to use",
            required=True
        )
        parser.add_argument(
            "--use_open_source_lead_model",
            action="store_true",
            help="Use the open-source lead generation model instead of dummy leads"
        )
        parser.add_argument(
            "--blacklist_force_validator_permit",
            action="store_true",
            help="Only allow validators to query the miner",
            default=False
        )
        parser.add_argument(
            "--blacklist_allow_non_registered",
            action="store_true",
            help="Allow non-registered hotkeys to query the miner",
            default=False
        )
        parser.add_argument(
            "--neuron_epoch_length",
            type=int,
            help="Number of blocks between metagraph syncs",
            default=1000
        )
        parser.add_argument(
            "--logging_trace",
            action="store_true",
            help="Enable trace-level logging",
            default=False
        )
        parser.add_argument(
            "--mock",
            action="store_true",
            help="Run in mock mode",
            default=False
        )

    def __init__(self, config=None):
        super().__init__(config=config)
        if config.logging_trace:
            bt.logging.set_trace(True)
        is_mock = getattr(self.config, 'mock', False)

        # Ensure config.axon is initialized
        if not hasattr(self.config, 'axon') or self.config.axon is None:
            self.config.axon = bt.Config()
            self.config.axon.ip = "0.0.0.0"
            self.config.axon.port = 8091
            bt.logging.debug("Initialized config.axon with default values")

        # Set miner UID
        self.uid = None
        if is_mock:
            self.uid = 0  # Default UID for mock mode
            bt.logging.info(f"Mock mode: Set miner UID to {self.uid}")
        else:
            max_retries = 5
            retry_delay = 10
            hotkey = self.wallet.hotkey.ss58_address
            bt.logging.debug(f"Attempting to set UID for hotkey: {hotkey}")
            for attempt in range(max_retries):
                try:
                    # Try neurons_lite
                    bt.logging.debug("Querying neurons_lite")
                    neurons = self.subtensor.neurons_lite(netuid=self.config.netuid)
                    bt.logging.debug(f"Neurons retrieved via neurons_lite: {[n.hotkey for n in neurons]}")
                    for neuron in neurons:
                        if neuron.hotkey == hotkey:
                            self.uid = neuron.uid
                            bt.logging.info(f"Miner UID set to {self.uid} via neurons_lite")
                            break
                    if self.uid is not None:
                        break
                    # Fallback to neurons
                    bt.logging.debug("Falling back to neurons method")
                    neurons = self.subtensor.neurons(netuid=self.config.netuid)
                    bt.logging.debug(f"Neurons retrieved via neurons: {[n.hotkey for n in neurons]}")
                    for neuron in neurons:
                        if neuron.hotkey == hotkey:
                            self.uid = neuron.uid
                            bt.logging.info(f"Miner UID set to {self.uid} via neurons")
                            break
                    if self.uid is not None:
                        break
                    bt.logging.warning(f"Attempt {attempt + 1}/{max_retries}: Wallet {self.config.wallet_name}/{self.config.wallet_hotkey} not found in netuid {self.config.netuid}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                except Exception as e:
                    bt.logging.error(f"Attempt {attempt + 1}/{max_retries}: Failed to set UID: {str(e)}\n{traceback.format_exc()}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
            if self.uid is None:
                bt.logging.warning(f"Wallet {self.config.wallet_name}/{self.config.wallet_hotkey} not registered on netuid {self.config.netuid} after {max_retries} attempts")
              

        # Warn if allowing incoming requests from anyone
        if not self.config.blacklist_force_validator_permit:
            bt.logging.warning("You are allowing non-validators to send requests to your miner. This is a security risk.")
        if self.config.blacklist_allow_non_registered:
            bt.logging.warning("You are allowing non-registered entities to send requests to your miner. This is a security risk.")

        # Initialize axon
        self.axon = bt.axon(
            wallet=self.wallet,
            config=self.config() if callable(self.config) else self.config,
        )
        bt.logging.info(f"Attaching forward function to miner axon.")
        self.axon.attach(
            forward_fn=self.forward,
            blacklist_fn=self.blacklist,
            priority_fn=self.priority,
        )
        bt.logging.info(f"Axon created: {self.axon}")

        # Instantiate runners
        self.should_exit: bool = False
        self.is_running: bool = False
        self.thread: Union[threading.Thread, None] = None
        self.lock = asyncio.Lock()

    def run(self):
        self.sync()
        if self.uid is None:
            bt.logging.error("Cannot run miner: UID not set. Please register the wallet on the network.")
            return

        if not self.config.mock:
            bt.logging.info(f"Serving miner axon {self.axon} on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}")
            self.axon.serve(netuid=self.config.netuid, subtensor=self.subtensor)
            self.axon.start()
        else:
            bt.logging.info("Mock mode: Not serving axon.")

        bt.logging.info(f"Miner starting at block: {self.block}")
        try:
            while not self.should_exit:
                if self.config.mock:
                    # In mock mode, skip epoch length check as last_update may be None
                    time.sleep(1)
                    if self.should_exit:
                        break
                    self.sync()
                    self.step += 1
                else:
                    while (
                        self.uid is not None and
                        self.metagraph.last_update[self.uid] is not None and
                        self.block - self.metagraph.last_update[self.uid] < self.config.neuron_epoch_length
                    ):
                        time.sleep(1)
                        if self.should_exit:
                            break
                    self.sync()
                    self.step += 1
        except KeyboardInterrupt:
            if not self.config.mock:
                self.axon.stop()
            bt.logging.success("Miner killed by keyboard interrupt.")
            exit()
        except Exception as e:
            bt.logging.error(traceback.format_exc())

    def run_in_background_thread(self):
        if not self.is_running:
            bt.logging.debug("Starting miner in background thread.")
            self.should_exit = False
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()
            self.is_running = True
            bt.logging.debug("Started")

    def stop_run_thread(self):
        if self.is_running:
            bt.logging.debug("Stopping miner in background thread.")
            self.should_exit = True
            if self.thread is not None:
                self.thread.join(5)
            self.is_running = False
            bt.logging.debug("Stopped")

    def __enter__(self):
        self.run_in_background_thread()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop_run_thread()

    def resync_metagraph(self):
        bt.logging.info("resync_metagraph()")
        self.metagraph.sync(subtensor=self.subtensor)