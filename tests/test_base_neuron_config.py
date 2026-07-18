from types import SimpleNamespace

import bittensor as bt

from Leadpoet.base.neuron import BaseNeuron


def test_config_neuron_fills_defaults_on_existing_config_section():
    neuron = SimpleNamespace(
        disable_set_weights=False,
        mode="coordinator",
    )
    instance = object.__new__(BaseNeuron)
    instance.config = SimpleNamespace(neuron=neuron)

    instance.config_neuron("./validator_state")

    assert neuron.disable_set_weights is False
    assert neuron.mode == "coordinator"
    assert neuron.axon_off is False
    assert neuron.num_concurrent_forwards == 1
    assert neuron.full_path == "./validator_state"
    assert neuron.moving_average_alpha == 0.1
    assert neuron.sample_size == 5


def test_config_neuron_preserves_explicit_values():
    neuron = SimpleNamespace(
        axon_off=True,
        num_concurrent_forwards=4,
        full_path="/var/lib/leadpoet/validator",
        moving_average_alpha=0.25,
        sample_size=9,
    )
    instance = object.__new__(BaseNeuron)
    instance.config = SimpleNamespace(neuron=neuron)

    instance.config_neuron("./validator_state")

    assert neuron.axon_off is True
    assert neuron.num_concurrent_forwards == 4
    assert neuron.full_path == "/var/lib/leadpoet/validator"
    assert neuron.moving_average_alpha == 0.25
    assert neuron.sample_size == 9


def test_config_neuron_fills_defaults_on_bittensor_config():
    config = bt.Config()
    config.neuron = bt.Config()
    config.neuron.disable_set_weights = False
    config.neuron.mode = "coordinator"
    instance = object.__new__(BaseNeuron)
    instance.config = config

    instance.config_neuron("./validator_state")

    assert config.neuron.full_path == "./validator_state"
    assert config.neuron.axon_off is False
    assert config.neuron.num_concurrent_forwards == 1
