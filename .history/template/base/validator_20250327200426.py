# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# TODO(developer): Set your name
# Copyright © 2023 <your name>


import copy
import numpy as np
import asyncio
import argparse
import threading
import bittensor as bt

from typing import List, Union
from traceback import print_exception

from template.base.neuron import BaseNeuron
from template.base.utils.weight_utils import (
    process_weights_for_netuid,
    convert_weights_and_uids_for_emit,
)
from template.mock import MockDendrite
from template.utils.config import add_validator_args

class BaseValidatorNeuron(BaseNeuron):
    """
    Base class for Bittensor validators. Your validator should inherit from this class.
    """

    neuron_type: str = "ValidatorNeuron"

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        super().add_args(parser)
        add_validator_args(cls, parser)

    def __init__(self, config=None):
        super().__init__(config=config)

        # Save a copy of the hotkeys to local memory.
        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)

        # Dendrite lets us send messages to other nodes (axons) in the network.
        if self.config.mock:
            self.dendrite = MockDendrite(wallet=self.wallet)
        else:
            self.dendrite = bt.dendrite(wallet=self.wallet)
        bt.logging.info(f"Dendrite: {self.dendrite}")

        # Set up initial scoring weights for validation
        bt.logging.info("Building validation weights.")
        self.scores = np.zeros(self.metagraph.n, dtype=np.float32)

        # Init sync with the network. Updates the metagraph.
        self.sync()

        # Serve axon to enable external connections.
        if not self.config.neuron.axon_off:
            self.serve_axon()
        else:
            bt.logging.warning("axon off, not serving ip to chain.")

        # Create asyncio event loop to manage async tasks.
        self.loop = asyncio.get_event_loop()

        # Instantiate runners
        self.should_exit: bool = False
        self.is_running: bool = False
        self.thread: Union[threading.Thread, None] = None
        self.lock = asyncio.Lock()

    def serve_axon(self):
        """Serve axon to enable external connections."""
        bt.logging.info("serving ip to chain...")
        try:
            self.axon = bt.axon(wallet=self.wallet, config=self.config)
            try:
                self.subtensor.serve_axon(
                    netuid=self.config.netuid,
                    axon=self.axon,
                )
                bt.logging.info(
                    f"Running validator {self.axon} on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
                )
            except Exception as e:
                bt.logging.error(f"Failed to serve Axon with exception: {e}")
                pass
        except Exception as e:
            bt.logging.error(f"Failed to create Axon initialize with exception: {e}")
            pass

    async def concurrent_forward(self):
        coroutines = [
            self.forward()
            for _ in range(self.config.neuron.num_concurrent_forwards)
        ]
        await asyncio.gather(*coroutines)

    def run(self):
        """Initiates and manages the main loop for the validator on the Bittensor network."""
        self.sync()
        bt.logging.info(f"Validator starting at block: {self.block}")
        try:
            while True:
                bt.logging.info(f"step({self.step}) block({self.block})")
                self.loop.run_until_complete(self.concurrent_forward())
                if self.should_exit:
                    break
                self.sync()
                self.step += 1
        except KeyboardInterrupt:
            self.axon.stop()
            bt.logging.success("Validator killed by keyboard interrupt.")
            exit()
        except Exception as err:
            bt.logging.error(f"Error during validation: {str(err)}")
            bt.logging.debug(str(print_exception(type(err), err, err.__traceback__)))

    def run_in_background_thread(self):
        """Starts the validator's operations in a background thread."""
        if not self.is_running:
            bt.logging.debug("Starting validator in background thread.")
            self.should_exit = False
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()
            self.is_running = True
            bt.logging.debug("Started")

    def stop_run_thread(self):
        """Stops the validator's operations in the background thread."""
        if self.is_running:
            bt.logging.debug("Stopping validator in background thread.")
            self.should_exit = True
            self.thread.join(5)
            self.is_running = False
            bt.logging.debug("Stopped")

    def __enter__(self):
        self.run_in_background_thread()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Stops the validator's background operations upon exiting the context."""
        if self.is_running:
            bt.logging.debug("Stopping validator in background thread.")
            self.should_exit = True
            self.thread.join(5)
            self.is_running = False
            bt.logging.debug("Stopped")

    def set_weights(self):
        """Sets the validator weights to the metagraph hotkeys based on miner scores."""
        if np.isnan(self.scores).any():
            bt.logging.warning("Scores contain NaN values.")
        norm = np.linalg.norm(self.scores, ord=1, axis=0, keepdims=True)
        if np.any(norm == 0) or np.isnan(norm).any():
            norm = np.ones_like(norm)
        raw_weights = self.scores / norm
        bt.logging.debug("raw_weights", raw_weights)
        bt.logging.debug("raw_weight_uids", str(self.metagraph.uids.tolist()))
        processed_weight_uids, processed_weights = process_weights_for_netuid(
            uids=self.metagraph.uids,
            weights=raw_weights,
            netuid=self.config.netuid,
            subtensor=self.subtensor,
            metagraph=self.metagraph,
        )
        bt.logging.debug("processed_weights", processed_weights)
        bt.logging.debug("processed_weight_uids", processed_weight_uids)
        uint_uids, uint_weights = convert_weights_and_uids_for_emit(
            uids=processed_weight_uids, weights=processed_weights
        )
        bt.logging.debug("uint_weights", uint_weights)
        bt.logging.debug("uint_uids", uint_uids)
        result, msg = self.subtensor.set_weights(
            wallet=self.wallet,
            netuid=self.config.netuid,
            uids=uint_uids,
            weights=uint_weights,
            wait_for_finalization=False,
            wait_for_inclusion=False,
            version_key=self.spec_version,
        )
        if result is True:
            bt.logging.info("set_weights on chain successfully!")
        else:
            bt.logging.error("set_weights failed", msg)

    def resync_metagraph(self):
        """Resyncs the metagraph and updates hotkeys and scores."""
        bt.logging.info("resync_metagraph()")
        previous_metagraph = copy.deepcopy(self.metagraph)
        self.metagraph.sync(subtensor=self.subtensor)
        if previous_metagraph.axons == self.metagraph.axons:
            return
        bt.logging.info("Metagraph updated, re-syncing hotkeys, dendrite pool and moving averages")
        for uid, hotkey in enumerate(self.hotkeys):
            if hotkey != self.metagraph.hotkeys[uid]:
                self.scores[uid] = 0
        if len(self.hotkeys) < len(self.metagraph.hotkeys):
            new_moving_average = np.zeros((self.metagraph.n))
            min_len = min(len(self.hotkeys), len(self.scores))
            new_moving_average[:min_len] = self.scores[:min_len]
            self.scores = new_moving_average
        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)

    def update_scores(self, rewards: np.ndarray, uids: List[int]):
        """Performs exponential moving average on the scores based on rewards."""
        if np.isnan(rewards).any():
            bt.logging.warning(f"NaN values detected in rewards: {rewards}")
            rewards = np.nan_to_num(rewards, nan=0)
        rewards = np.asarray(rewards)
        if isinstance(uids, np.ndarray):
            uids_array = uids.copy()
        else:
            uids_array = np.array(uids)
        if rewards.size == 0 or uids_array.size == 0:
            bt.logging.info(f"rewards: {rewards}, uids_array: {uids_array}")
            bt.logging.warning("Either rewards or uids_array is empty.")
            return
        if rewards.size != uids_array.size:
            raise ValueError(
                f"Shape mismatch: rewards {rewards.shape} vs uids {uids_array.shape}"
            )
        scattered_rewards: np.ndarray = np.zeros_like(self.scores)
        scattered_rewards[uids_array] = rewards
        bt.logging.debug(f"Scattered rewards: {rewards}")
        alpha: float = self.config.neuron.moving_average_alpha
        self.scores: np.ndarray = (
            alpha * scattered_rewards + (1 - alpha) * self.scores
        )
        bt.logging.debug(f"Updated moving avg scores: {self.scores}")

    def save_state(self):
        """Saves the state of the validator to a file."""
        bt.logging.info("Saving validator state.")
        np.savez(
            self.config.neuron.full_path + "/state.npz",
            step=self.step,
            scores=self.scores,
            hotkeys=self.hotkeys,
        )

    def load_state(self):
        """Loads the state of the validator from a file."""
        bt.logging.info("Loading validator state.")
        state = np.load(self.config.neuron.full_path + "/state.npz")
        self.step = state["step"]
        self.scores = state["scores"]
        self.hotkeys = state["hotkeys"]