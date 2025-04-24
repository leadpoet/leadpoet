import argparse

def add_validator_args(cls, parser: argparse.ArgumentParser):
    parser.add_argument(
        "--use_open_source_validator_model",
        action="store_true",
        help="Use the open-source validator model instead of simulated review"
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