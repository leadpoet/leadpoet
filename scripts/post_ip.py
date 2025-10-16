#!/usr/bin/env python3
"""
Register a miner’s public (IP-address, port) on-chain.

Equivalent to SCORE-Vision’s `fiber-post-ip`.  
Run it once (or whenever your IP/port changes) **before** starting the miner.
"""

import argparse
import sys
import bittensor as bt


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publish your miner's public (IP, port) to the subnet metagraph"
    )
    parser.add_argument("--netuid", type=int, required=True, help="Subnet id (e.g. 71)")
    parser.add_argument(
        "--subtensor_network",
        type=str,
        default="test",
        help="Bittensor network (test / finney / main)",
    )
    parser.add_argument("--wallet_name", required=True, help="Bittensor wallet name")
    parser.add_argument("--wallet_hotkey", required=True, help="Bittensor hotkey name")
    parser.add_argument(
        "--external_ip", required=True, help="Public IP address validators will dial"
    )
    parser.add_argument(
        "--external_port",
        type=int,
        required=True,
        help="Public TCP port validators will dial",
    )
    args = parser.parse_args()

    # ───────────────────────── Bittensor config ──────────────────────────
    # Root config
    config = bt.Config()
    config.netuid = args.netuid

    # ---------------- Subtensor section ----------------
    config.subtensor = bt.Config()
    config.subtensor.network = args.subtensor_network

    # ---------------- Axon section ---------------------
    config.axon = bt.Config()
    # Advertised (public) address
    config.axon.external_ip   = args.external_ip
    config.axon.external_port = args.external_port
    # Internal bind – irrelevant here but required
    config.axon.ip   = "0.0.0.0"
    config.axon.port = args.external_port

    wallet = bt.wallet(name=args.wallet_name, hotkey=args.wallet_hotkey)
    subtensor = bt.subtensor(network=args.subtensor_network)

    # Dummy axon → publish metadata on-chain
    axon = bt.axon(wallet=wallet, config=config)
    subtensor.serve_axon(netuid=args.netuid, axon=axon)
    print(
        f"✅ Published {args.external_ip}:{args.external_port} for "
        f"{wallet.hotkey.ss58_address} on netuid {args.netuid}"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        bt.logging.error(str(err))
        sys.exit(1)