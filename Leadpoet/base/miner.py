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
        parser.add_argument("--netuid", type=int, help="The network UID of the subnet to connect to", default=343)
        parser.add_argument("--subtensor_network", type=str, help="The network to connect to (e.g., test, main)", default="test")
        parser.add_argument("--wallet_name", type=str, help="The name of the wallet to use", required=True)
        parser.add_argument("--wallet_hotkey", type=str, help="The hotkey of the wallet to use", required=True)
        parser.add_argument("--use_open_source_lead_model", action="store_true", help="Use the open-source lead generation model instead of dummy leads")
        parser.add_argument("--blacklist_force_validator_permit", action="store_true", help="Only allow validators to query the miner", default=False)
        parser.add_argument("--blacklist_allow_non_registered", action="store_true", help="Allow non-registered hotkeys to query the miner", default=False)
        parser.add_argument("--neuron_epoch_length", type=int, help="Number of blocks between metagraph syncs", default=1000)
        parser.add_argument("--logging_trace", action="store_true", help="Enable trace-level logging", default=False)

    def __init__(self, config=None):
        super().__init__(config=config)
        if config.logging_trace:
            bt.logging.set_trace(True)

        if not hasattr(self.config, 'neuron') or self.config.neuron is None:
            self.config.neuron = bt.Config()
            self.config.neuron.axon_off = False
            self.config.neuron.num_concurrent_forwards = 1
            self.config.neuron.full_path = "./miner_state"
            self.config.neuron.moving_average_alpha = 0.1
            self.config.neuron.sample_size = 5
            bt.logging.debug("Initialized config.neuron with defaults")

        if not hasattr(self.config, 'axon') or self.config.axon is None:
            self.config.axon = bt.Config()
            self.config.axon.ip = "0.0.0.0"
            self.config.axon.port = 8091
            bt.logging.debug("Initialized config.axon with default values")

        if not hasattr(self.config, 'blacklist') or self.config.blacklist is None:
            self.config.blacklist = bt.Config()
            self.config.blacklist.force_validator_permit = False
            self.config.blacklist.allow_non_registered = False
            bt.logging.debug("Initialized config.blacklist with defaults")

        if not hasattr(self.config, 'priority') or self.config.priority is None:
            self.config.priority = bt.Config()
            self.config.priority.default_priority = 0.0
            bt.logging.debug("Initialized config.priority with defaults")

        bt.logging.info("Registering wallet on network...")
        max_retries = 3
        retry_delay = 5
        for attempt in range(max_retries):
            try:
                self.uid = self.subtensor.get_uid_for_hotkey_on_subnet(
                    hotkey_ss58=self.wallet.hotkey.ss58_address,
                    netuid=self.config.netuid,
                )
                if self.uid is not None:
                    bt.logging.success(f"Wallet registered with UID: {self.uid}")
                    break
                else:
                    bt.logging.warning(f"Attempt {attempt + 1}/{max_retries}: Wallet not registered on netuid {self.config.netuid}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
            except Exception as e:
                bt.logging.error(f"Attempt {attempt + 1}/{max_retries}: Failed to set UID: {str(e)}\n{traceback.format_exc()}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)

        if self.uid is None:
            bt.logging.warning(f"Wallet {self.config.wallet_name}/{self.config.wallet_hotkey} not registered on netuid {self.config.netuid} after {max_retries} attempts")

        # For testnet, it's normal to allow non-validators
        if not self.config.blacklist_force_validator_permit:
            bt.logging.info("Testnet mode: Allowing non-validators to send requests (normal for testnet)")
        if self.config.blacklist_allow_non_registered:
            bt.logging.info("Testnet mode: Allowing non-registered entities to send requests (normal for testnet)")

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

        self.should_exit: bool = False
        self.is_running: bool = False
        self.thread: Union[threading.Thread, None] = None
        self.lock = asyncio.Lock()

    def run(self):
        self.sync()
        if self.uid is None:
            bt.logging.error("Cannot run miner: UID not set. Please register the wallet on the network.")
            return

        print(f"   Starting axon serve...")
        bt.logging.info(f"Running miner for subnet: {self.config.netuid} on network: {self.config.subtensor.chain_endpoint} with config: {self.config}")
        self.axon.serve(netuid=self.config.netuid, subtensor=self.subtensor)
        print(f"   Axon serve completed, starting axon...")
        self.axon.start()
        print(f"   Axon started successfully!")

        bt.logging.info(f"Miner starting at block: {self.block}")
        try:
            while not self.should_exit:
                bt.logging.info(f"Miner running... {time.time()}")
                time.sleep(5)
                last_update = self.metagraph.last_update[self.uid] if self.uid is not None and self.uid < len(self.metagraph.last_update) else 0

                if last_update is None or last_update == 0:
                    bt.logging.warning(f"last_update for UID {self.uid} is invalid. Resyncing metagraph.")
                    self.resync_metagraph()
                    continue

                epoch_length = getattr(self.config.neuron, 'epoch_length', 1000)
                while self.uid is not None and last_update is not None and self.block - last_update < epoch_length:
                    time.sleep(1)
                    if self.should_exit:
                        break
                self.sync()
                self.step += 1
        except KeyboardInterrupt:
            self.axon.stop()
            bt.logging.success("Miner killed by keyboard interrupt.")
            exit()
        except Exception as e:
            print(f"   Error in miner run loop: {e}")
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
