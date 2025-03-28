# The MIT License (MIT)
# Copyright © 2025 Yuma Rao
# Copyright © 2025 Opentensor Foundation

# IN PROGRESS FOR LEADPOET

import sys
import unittest

import bittensor as bt
import torch

from neurons.validator import Validator
from template.base.validator import BaseValidatorNeuron
from template.protocol import Dummy
from template.utils.uids import get_random_uids
from template.validator.reward import get_rewards


class TemplateValidatorNeuronTestCase(unittest.TestCase):
    """
    This class contains unit tests for the RewardEvent classes.

    The tests cover different scenarios where completions may or may not be successful and the reward events are checked that they don't contain missing values.
    The `reward` attribute of all RewardEvents is expected to be a float, and the `is_filter_model` attribute is expected to be a boolean.
    """

    def setUp(self):
        sys.argv = sys.argv[0] + ["--config", "tests/configs/validator.json"]

        config = BaseValidatorNeuron.config()
        config.wallet._mock = True
        config.metagraph._mock = True
        config.subtensor._mock = True
        self.neuron = Validator(config)
        self.miner_uids = get_random_uids(self, k=10)

    def test_run_single_step(self):
        # TODO: Test a single step
        pass

    def test_sync_error_if_not_registered(self):
        # TODO: Test that the validator throws an error if it is not registered on metagraph
        pass

    def test_forward(self):
        # TODO: Test that the forward function returns the correct value
        pass

    def test_dummy_responses(self):
        # TODO: Test that the dummy responses are correctly constructed

        responses = self.neuron.dendrite.query(
            # Send the query to miners in the network.
            axons=[
                self.neuron.metagraph.axons[uid] for uid in self.miner_uids
            ],
            # Construct a dummy query.
            synapse=Dummy(dummy_input=self.neuron.step),
            # All responses have the deserialize function called on them before returning.
            deserialize=True,
        )

        for i, response in enumerate(responses):
            self.assertEqual(response, self.neuron.step * 2)

    def test_reward(self):
        # TODO: Test that the reward function returns the correct value
        responses = self.dendrite.query(
            # Send the query to miners in the network.
            axons=[self.metagraph.axons[uid] for uid in self.miner_uids],
            # Construct a dummy query.
            synapse=Dummy(dummy_input=self.neuron.step),
            # All responses have the deserialize function called on them before returning.
            deserialize=True,
        )

        rewards = get_rewards(self.neuron, responses)
        expected_rewards = torch.FloatTensor([1.0] * len(responses))
        self.assertEqual(rewards, expected_rewards)

    def test_reward_with_nan(self):
        # TODO: Test that NaN rewards are correctly sanitized
        # TODO: Test that a bt.logging.warning is thrown when a NaN reward is sanitized
        responses = self.dendrite.query(
            # Send the query to miners in the network.
            axons=[self.metagraph.axons[uid] for uid in self.miner_uids],
            # Construct a dummy query.
            synapse=Dummy(dummy_input=self.neuron.step),
            # All responses have the deserialize function called on them before returning.
            deserialize=True,
        )

        rewards = get_rewards(self.neuron, responses)
        expected_rewards = rewards.clone()
        # Add NaN values to rewards
        rewards[0] = float("nan")

        with self.assertLogs(bt.logging, level="WARNING") as cm:
            self.neuron.update_scores(rewards, self.miner_uids)
