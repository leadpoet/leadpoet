import argparse

def add_validator_args(cls, parser: argparse.ArgumentParser):
    parser.add_argument(
        "--use_open_source_validator_model",
        action="store_true",
        help="Use the open-source validator model instead of simulated review"
    )
    parser.add_argument(
        "--neuron.disable_set_weights",
        action="store_true",
        help="Disables setting weights.",
        default=False,
        dest="neuron_disable_set_weights",
    )
    parser.add_argument(
        "--neuron.sample_size",
        type=int,
        help="Number of miners to query per forward pass",
        default=10
    )
    parser.add_argument(
        "--neuron.moving_average_alpha",
        type=float,
        help="Moving average alpha for score updates",
        default=0.1
    )
    parser.add_argument(
        "--sourcing_interval",
        type=int,
        help="Interval (in seconds) between each sourcing cycle by the miner",
        default=60
    )
    parser.add_argument(
        "--queue_maxsize",
        type=int,
        help="Maximum size of the in-memory prospect queue",
        default=1000
    )