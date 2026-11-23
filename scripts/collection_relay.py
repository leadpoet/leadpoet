"""Connect the isolated validator and gateway workers via the operator's verified SSH aliases."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import time
from contextlib import ExitStack

from scripts.collection_review import read_file
import json


def run_peer(host, command, stdin, stdout):
    if (
        not isinstance(command, list)
        or not command
        or not all(isinstance(s, str) and s and "\x00" not in s for s in command)
    ):
        raise ValueError("invalid worker command")
    return subprocess.Popen(
        [
            "ssh",
            "-T",
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            "ForwardAgent=no",
            "-o",
            "ConnectTimeout=12",
            host,
            shlex.join(command),
        ],
        stdin=stdin,
        stdout=stdout,
        close_fds=True,
    )


def relay(gateway_command, validator_command):
    # The operator shell never parses, logs or stores the private protocol.
    # Only the two children retain the pipe descriptors after launch.
    deadline = time.monotonic() + 2 * 3600
    with ExitStack() as workers:
        with ExitStack() as pipes:
            up, down = (os.pipe() for _ in range(2))
            for fd in (*up, *down):
                pipes.callback(os.close, fd)
            gateway = run_peer("leadpoet-gateway-ssm", gateway_command, up[0], down[1])
            workers.callback(_close, gateway)
            validator = run_peer(
                "leadpoet-validator-ssm", validator_command, down[0], up[1]
            )
            workers.callback(_close, validator)
        g = gateway.wait(timeout=max(1, deadline - time.monotonic()))
        v = validator.wait(timeout=max(1, deadline - time.monotonic()))
        if g or v:
            raise RuntimeError(f"collection relay failed (gateway={g}, validator={v})")


def _close(child):
    if child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--worker-commands",
        type=Path,
        required=True,
        help="private local JSON with gateway and validator argv lists for one research run",
    )
    c = json.loads(read_file(p.parse_args().worker_commands))
    if set(c) != {"gateway", "validator"}:
        raise ValueError("both worker commands required")
    relay(c["gateway"], c["validator"])


if __name__ == "__main__":
    main()
