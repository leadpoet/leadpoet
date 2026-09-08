"""Run only the native automatic weight loop for temporary testnet401 proof.

This removable entrypoint constructs the real ``neurons.validator.Validator``
with its enclave-backed wallet and calls its unchanged
``submit_weights_at_epoch_end`` method.  It does not start the epoch monitor,
serve an axon, score leads, curate requests, construct weights, or call a chain
extrinsic itself.  Delete this file after the automatic finalized proof.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import signal
from typing import Optional, Sequence


NETWORK = "test"
NETUID = 401
CHAIN_ENDPOINT = "wss://test.finney.opentensor.ai:443"
POLL_SECONDS = 10.0


def _configuration(*, wallet_name: str, wallet_hotkey: str, wallet_path: Path, state_path: Path):
    import bittensor as bt

    config = bt.Config()
    config.wallet = bt.Config()
    config.wallet.name = wallet_name
    config.wallet.hotkey = wallet_hotkey
    config.wallet.path = str(wallet_path)
    config.wallet_name = wallet_name
    config.wallet_hotkey = wallet_hotkey
    config.netuid = NETUID
    config.subtensor = bt.Config()
    config.subtensor.network = NETWORK
    config.subtensor.chain_endpoint = CHAIN_ENDPOINT
    config.neuron = bt.Config()
    config.neuron.axon_off = True
    config.neuron.disable_set_weights = False
    config.neuron.container_id = 0
    config.neuron.total_containers = 1
    config.neuron.mode = "coordinator"
    config.neuron.full_path = str(state_path)
    return config


def build_native_validator(
    *,
    wallet_name: str,
    wallet_hotkey: str,
    wallet_path: Path,
    state_path: Path,
):
    if os.environ.get("BITTENSOR_NETWORK") != NETWORK or os.environ.get(
        "BITTENSOR_NETUID"
    ) != str(NETUID):
        raise RuntimeError("temporary validator environment is not testnet401")
    state_path.mkdir(parents=True, mode=0o700, exist_ok=True)
    os.chdir(state_path)

    from neurons import validator as validator_module

    validator_module.ensure_data_files()
    config = _configuration(
        wallet_name=wallet_name,
        wallet_hotkey=wallet_hotkey,
        wallet_path=wallet_path,
        state_path=state_path,
    )
    validator = validator_module.Validator(config=config)
    if config.neuron.axon_off is not True or getattr(validator, "axon", None) is not None:
        raise RuntimeError("weight-only validator unexpectedly created an axon")
    return validator_module, validator


def _write_readiness(path: Path) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "schema_version": "leadpoet.temporary_testnet401_weight_poll_ready.v1",
                "status": "ready",
            },
            handle,
            sort_keys=True,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


async def run_weight_only_loop(
    *,
    validator_module,
    validator,
    readiness_path: Path,
    stop_event: Optional[asyncio.Event] = None,
) -> None:
    stop = stop_event or asyncio.Event()
    loop = asyncio.get_running_loop()
    if stop_event is None:
        for selected_signal in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(selected_signal, stop.set)
    try:
        await validator.initialize_async_subtensor()
        validator_module.reward_module.inject_async_subtensor(
            validator.async_subtensor
        )
        validator_module.cloud_db_module._VERIFY.inject_async_subtensor(
            validator.async_subtensor
        )
        ready = False
        while not stop.is_set():
            try:
                submitted = await validator.submit_weights_at_epoch_end()
                if not ready:
                    _write_readiness(readiness_path)
                    ready = True
                print(
                    json.dumps(
                        {
                            "event": "automatic_weight_tick",
                            "submitted_or_already_complete": bool(submitted),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            except Exception as exc:
                if not ready:
                    raise
                print(
                    json.dumps(
                        {
                            "event": "automatic_weight_tick_failed",
                            "failure_type": type(exc).__name__,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            try:
                await asyncio.wait_for(stop.wait(), timeout=POLL_SECONDS)
            except asyncio.TimeoutError:
                pass
    finally:
        validator.should_exit = True
        await validator.cleanup_async_subtensor()
        validator_module._close_subtensor_connection(
            validator.subtensor,
            source="temporary_weight_only_shutdown",
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wallet-name", required=True)
    parser.add_argument("--wallet-hotkey", required=True)
    parser.add_argument("--wallet-path", type=Path, required=True)
    parser.add_argument("--state-path", type=Path, required=True)
    parser.add_argument("--readiness-file", type=Path, required=True)
    args = parser.parse_args(argv)
    validator_module, validator = build_native_validator(
        wallet_name=args.wallet_name,
        wallet_hotkey=args.wallet_hotkey,
        wallet_path=args.wallet_path,
        state_path=args.state_path,
    )
    asyncio.run(
        run_weight_only_loop(
            validator_module=validator_module,
            validator=validator,
            readiness_path=args.readiness_file,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
