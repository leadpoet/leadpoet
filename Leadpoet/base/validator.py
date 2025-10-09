import copy
import numpy as np
import asyncio
import argparse
import threading
import bittensor as bt
import time
from typing import List, Union
from traceback import print_exception
from Leadpoet.base.neuron import BaseNeuron
from Leadpoet.base.utils.weight_utils import (
    process_weights_for_netuid,
    convert_weights_and_uids_for_emit,
)
from Leadpoet.utils.config import add_validator_args
from Leadpoet.validator.reward import calculate_emissions

class BaseValidatorNeuron(BaseNeuron):
    neuron_type: str = "ValidatorNeuron"

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        super().add_args(parser)
        add_validator_args(cls, parser)

    def __init__(self, config=None):
        super().__init__(config=config)
        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)

        self.config_neuron("./validator_state")
        self.config_axon(8093)

        self.dendrite = bt.dendrite(wallet=self.wallet)
        bt.logging.info(f"Dendrite: {self.dendrite}")
        bt.logging.info("Building validation weights.")
        self.scores = np.zeros(self.metagraph.n, dtype=np.float32)
        self.sync()
        if not self.config.neuron.axon_off:
            self.serve_axon()
        
        self.total_emissions = 1000.0  # Default emissions value for subnet

    def serve_axon(self):
        bt.logging.info("Serving validator axon...")
        try:
            self.axon = bt.axon(wallet=self.wallet, config=self.config)
            # expose buyer-query endpoint (LeadRequest → LeadRequest)
            self.axon.attach(self.forward)           # ← NEW
            self.subtensor.serve_axon(
                netuid=self.config.netuid,
                axon=self.axon,
            )
            bt.logging.info(
                f"Running validator for subnet: {self.config.netuid} on network: {self.config.subtensor.chain_endpoint} with config: {self.config}"
            )
        except Exception as e:
            bt.logging.error(f"Failed to serve axon: {e}")
            self.axon = None

    async def concurrent_forward(self):
        coroutines = [
            self.forward()
            for _ in range(self.config.neuron.num_concurrent_forwards)
        ]
        await asyncio.gather(*coroutines)
        emissions = calculate_emissions(self, self.total_emissions, [self])
        bt.logging.info(f"Validator emissions: {emissions}")

    def run(self):
        self.sync()
        bt.logging.info(f"Validator starting at block: {self.block}")
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            while True:
                bt.logging.info(f"step({self.step}) block({self.block})")
                loop.run_until_complete(self.concurrent_forward())
                if self.should_exit:
                    break
                self.sync()
                self.step += 1
            loop.close()
        except KeyboardInterrupt:
            self.axon.stop()
            bt.logging.success("Validator killed by keyboard interrupt.")
            exit()
        except Exception as err:
            bt.logging.error(f"Error during validation: {str(err)}")
            bt.logging.debug(str(print_exception(type(err), err, err.__traceback__)))

    def run_in_background_thread(self):
        if not self.is_running:
            bt.logging.debug("Starting validator in background thread.")
            self.should_exit = False
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()
            self.is_running = True
            bt.logging.debug("Started")

    def stop_run_thread(self):
        if self.is_running:
            bt.logging.debug("Stopping validator in background thread.")
            self.should_exit = True
            if self.thread is not None:
                self.thread.join(5)
            self.is_running = False
            bt.logging.debug("Stopped")

    def __enter__(self):
        self.run_in_background_thread()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.is_running:
            bt.logging.debug("Stopping validator in background thread.")
            self.should_exit = True
            if self.thread is not None:
                self.thread.join(5)
            self.is_running = False
            bt.logging.debug("Stopped")

    def set_weights(self):
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
        if result:
            bt.logging.info("set_weights on chain successfully!")
        else:
            bt.logging.error("set_weights failed", msg)

    def resync_metagraph(self):
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
        scattered_rewards = np.zeros_like(self.scores)
        scattered_rewards[uids_array] = rewards
        bt.logging.debug(f"Scattered rewards: {rewards}")
        alpha = self.config.neuron.moving_average_alpha
        self.scores = alpha * scattered_rewards + (1 - alpha) * self.scores
        bt.logging.debug(f"Updated moving avg scores: {self.scores}")

    def save_state(self):
        bt.logging.info("Saving validator state.")
        np.savez(
            self.config.neuron.full_path + "/state.npz",
            step=self.step,
            scores=self.scores,
            hotkeys=self.hotkeys,
        )

    def load_state(self):
        bt.logging.info("Loading validator state.")
        state = np.load(self.config.neuron.full_path + "/state.npz")
        self.step = state["step"]
        self.scores = state["scores"]
        self.hotkeys = state["hotkeys"]
