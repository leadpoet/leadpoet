
import os
import subprocess
import argparse
import bittensor as bt
from .logging import setup_events_logger

def is_cuda_available():
    try:
        output = subprocess.check_output(["nvidia-smi", "-L"], stderr=subprocess.STDOUT)
        if "NVIDIA" in output.decode("utf-8"):
            return "cuda"
    except Exception:
        pass
    try:
        output = subprocess.check_output(["nvcc", "--version"]).decode("utf-8")
        if "release" in output:
            return "cuda"
    except Exception:
        pass
    return "cpu"

def check_config(cls, config: "bt.Config"):
    bt.logging.check_config(config)
    full_path = os.path.expanduser(
        "{}/{}/{}/netuid{}/{}".format(
            config.logging.logging_dir,
            config.wallet.name,
            config.wallet.hotkey,
            config.netuid,
            config.neuron.name,
        )
    )
    config.neuron.full_path = os.path.expanduser(full_path)
    if not os.path.exists(config.neuron.full_path):
        os.makedirs(config.neuron.full_path, exist_ok=True)
    if not config.neuron.dont_save_events:
        events_logger = setup_events_logger(
            config.neuron.full_path, config.neuron.events_retention_size
        )
        bt.logging.register_primary_logger(events_logger.name)

def add_args(cls, parser):
    parser.add_argument("--netuid", type=int, help="Subnet netuid", default=71)
    parser.add_argument(
        "--neuron.device",
        type=str,
        help="Device to run on.",
        default=is_cuda_available(),
    )
    parser.add_argument(
        "--neuron.epoch_length",
        type=int,
        help="The default epoch length (how often we set weights, measured in 12 second blocks).",
        default=100,
    )
    parser.add_argument(
        "--neuron.events_retention_size",
        type=str,
        help="Events retention size.",
        default=2 * 1024 * 1024 * 1024,  # 2 GB
    )
    parser.add_argument(
        "--neuron.dont_save_events",
        action="store_true",
        help="If set, we dont save events to a log file.",
        default=False,
    )
    parser.add_argument(
        "--wandb.off",
        action="store_true",
        help="Turn off wandb.",
        default=False,
    )
    parser.add_argument(
        "--wandb.offline",
        action="store_true",
        help="Runs wandb in offline mode.",
        default=False,
    )
    parser.add_argument(
        "--wandb.notes",
        type=str,
        help="Notes to add to the wandb run.",
        default="",
    )

def add_miner_args(cls, parser):
    parser.add_argument(
        "--neuron.name",
        type=str,
        help="Trials for this neuron go in neuron.root / (wallet_cold - wallet_hot) / neuron.name. ",
        default="miner",
    )
    parser.add_argument(
        "--blacklist.force_validator_permit",
        action="store_true",
        help="If set, we will force incoming requests to have a permit.",
        default=False,
    )
    parser.add_argument(
        "--blacklist.allow_non_registered",
        action="store_true",
        help="If set, miners will accept queries from non registered entities. (Dangerous!)",
        default=False,
    )
    parser.add_argument(
        "--wandb.project_name",
        type=str,
        default="template-miners",
        help="Wandb project to log to.",
    )
    parser.add_argument(
        "--wandb.entity",
        type=str,
        default="opentensor-dev",
        help="Wandb entity to log to.",
    )

def add_validator_args(cls, parser):
    # List of arguments to add
    validator_args = [
        {
            "name": "--neuron.name",
            "type": str,
            "help": "Trials for this neuron go in neuron.root / (wallet_cold - wallet_hot) / neuron.name. ",
            "default": "validator",
        },
        {
            "name": "--neuron.timeout",
            "type": float,
            "help": "The timeout for each forward call in seconds.",
            "default": 10,
        },
        {
            "name": "--neuron.num_concurrent_forwards",
            "type": int,
            "help": "The number of concurrent forwards running at any time.",
            "default": 1,
        },
        {
            "name": "--neuron.sample_size",
            "type": int,
            "help": "The number of miners to query in a single step.",
            "default": 50,
        },
        {
            "name": "--neuron.disable_set_weights",
            "action": "store_true",
            "help": "Disables setting weights.",
            "default": False,
        },
        {
            "name": "--neuron.moving_average_alpha",
            "type": float,
            "help": "Moving average alpha parameter, how much to add of the new observation.",
            "default": 0.1,
        },
        {
            "name": "--neuron.axon_off",
            "action": "store_true",
            "help": "Set this flag to not attempt to serve an Axon.",
            "default": False,
            "dest": "neuron.axon_off",  # Handle alias --axon_off
        },
        {
            "name": "--neuron.vpermit_tao_limit",
            "type": int,
            "help": "The maximum number of TAO allowed to query a validator with a vpermit.",
            "default": 4096,
        },
        {
            "name": "--wandb.project_name",
            "type": str,
            "help": "The name of the project where you are sending the new run.",
            "default": "template-validators",
        },
        {
            "name": "--wandb.entity",
            "type": str,
            "help": "The name of the project where you are sending the new run.",
            "default": "opentensor-dev",
        },
        {
            "name": "--use_open_source_validator_model",
            "action": "store_true",
            "help": "Use open-source validator model for lead validation.",
            "default": True,
        },
    ]

    # Check existing arguments to avoid conflicts
    existing_args = {action.dest for action in parser._actions}
    for arg in validator_args:
        dest = arg.get("dest", arg["name"].lstrip('-').replace('-', '_'))
        if dest not in existing_args:
            parser_kwargs = {k: v for k, v in arg.items() if k not in ("name", "dest")}
            parser.add_argument(arg["name"], **parser_kwargs)
        else:
            bt.logging.debug(f"Skipping argument {arg['name']} as it already exists in parser.")

def config(cls):
    parser = argparse.ArgumentParser()
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.axon.add_args(parser)
    cls.add_args(parser)
    return bt.config(parser)