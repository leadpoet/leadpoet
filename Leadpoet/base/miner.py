import time
import threading
import argparse
import traceback
import bittensor as bt
import os
from Leadpoet.base.neuron import BaseNeuron

class BaseMinerNeuron(BaseNeuron):
    neuron_type: str = "MinerNeuron"

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        parser.add_argument("--netuid", type=int, help="The network UID of the subnet to connect to", default=71)
        parser.add_argument("--subtensor_network", type=str, help="The network to connect to (e.g., test, main)", default="finney")
        parser.add_argument("--wallet_name", type=str, help="The name of the wallet to use", required=True)
        parser.add_argument("--wallet_hotkey", type=str, help="The hotkey of the wallet to use", required=True)
        parser.add_argument("--use_open_source_lead_model", action="store_true", help="Use the open-source lead generation model instead of dummy leads")
        parser.add_argument("--blacklist_force_validator_permit", action="store_true", help="Only allow validators to query the miner", default=False)
        parser.add_argument("--blacklist_allow_non_registered", action="store_true", help="Allow non-registered hotkeys to query the miner", default=False)
        parser.add_argument("--neuron_epoch_length", type=int, help="Number of blocks between metagraph syncs", default=1000)
        parser.add_argument("--logging_trace", action="store_true", help="Enable trace-level logging", default=False)
        parser.add_argument("--axon_ip", type=str,
                            help="Public IP address that validators should use to reach this miner",
                            default=None)
        parser.add_argument("--axon_port", type=int,
                            help="Public port that validators should use to reach this miner",
                            default=None)

    def __init__(self, config=None):
        super().__init__(config=config)
        if self.config.logging_trace:
            bt.logging.set_trace(True)

        self.config_neuron("./miner_state")
        self.config_axon(8091)

        if getattr(self.config, "axon_ip", None):
            self.config.axon.external_ip = self.config.axon_ip
        if getattr(self.config, "axon_port", None):
            self.config.axon.external_port = self.config.axon_port
            self.config.axon.port         = self.config.axon_port

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

        # Auto-adopt previously-published axon address (before we build the axon)
        if (
            not getattr(self.config.axon, "external_ip", None)
            or not getattr(self.config.axon, "external_port", None)
        ):
            try:
                published = self.metagraph.axons[self.uid]          # on-chain record
                self.config.axon.external_ip   = published.ip
                self.config.axon.external_port = int(published.port)
                self.config.axon.port          = int(published.port)  # bind locally on same port
                bt.logging.info(
                    f"Adopted on-chain axon endpoint "
                    f"{self.config.axon.external_ip}:{self.config.axon.external_port}"
                )
            except Exception as e:
                bt.logging.warning(f"Could not read on-chain axon metadata: {e}")

        # Enable low-level gRPC logs (silent - only for debugging)
        os.environ.setdefault("GRPC_VERBOSITY", "ERROR")  # Changed from DEBUG to ERROR
        os.environ.setdefault("GRPC_TRACE", "")

        # NOW build the axon with the correct port
        self.axon = bt.axon(
            wallet=self.wallet,
            ip      = "0.0.0.0",
            port    = self.config.axon.port,
            external_ip   = self.config.axon.external_ip,
            external_port = self.config.axon.external_port,
        )
        bt.logging.info("Attaching forward function to miner axon.")
        self.axon.attach(
            forward_fn=self.forward,
            blacklist_fn=self.blacklist,
            priority_fn=self.priority,
        )
        bt.logging.info(f"Axon created: {self.axon}")

        # Axon configured - logging reduced for cleaner output
        bt.logging.info(f"Axon ready: {self.config.axon.external_ip or '0.0.0.0'}:{self.config.axon.external_port or self.config.axon.port}")

    def run(self):
        self.sync()
        if self.uid is None:
            bt.logging.error("Cannot run miner: UID not set. Please register the wallet on the network.")
            return

        print("   Starting axon serve...")
        bt.logging.info(f"Running miner for subnet: {self.config.netuid} on network: {self.config.subtensor.chain_endpoint} with config: {self.config}")
        print("   [axon.serve] calling serve()")
        self.axon.serve(netuid=self.config.netuid, subtensor=self.subtensor)
        print("   Axon serve completed, starting axon...")
        print("   [axon.start] starting gRPC server …")
        self.axon.start()
        print("   Axon started successfully!")
        # Post-start visibility
        print(f"🖧  Local gRPC listener  : 0.0.0.0:{self.config.axon.port}")
        print(f"🌐  External endpoint   : {self.config.axon.external_ip}:{self.config.axon.external_port}")

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
