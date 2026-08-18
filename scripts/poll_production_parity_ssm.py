#!/usr/bin/env python3
"""Poll one production-parity SSM command for one credential-bounded window."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import time
from typing import Any, Sequence

import boto3
from botocore.exceptions import BotoCoreError, ClientError


ID_RE = re.compile(r"^[A-Za-z0-9-]{8,128}$")
TERMINAL_FAILURES = {
    "Failed", "Cancelled", "TimedOut", "Cancelling", "DeliveryTimedOut"
}


class PollError(RuntimeError):
    pass


def poll(
    client: Any,
    *,
    command_id: str,
    instance_id: str,
    max_wait_seconds: int,
) -> str:
    if (
        ID_RE.fullmatch(command_id) is None
        or re.fullmatch(r"^i-[0-9a-f]{8,17}$", instance_id) is None
        or not 30 <= max_wait_seconds <= 17_400
    ):
        raise PollError("SSM poll inputs are invalid")
    deadline = time.monotonic() + max_wait_seconds
    while True:
        try:
            response = client.get_command_invocation(
                CommandId=command_id, InstanceId=instance_id
            )
            status = str(response.get("Status") or "")
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") == "InvocationDoesNotExist":
                status = "Pending"
            else:
                raise PollError("SSM invocation read failed") from exc
        except BotoCoreError as exc:
            raise PollError("SSM invocation read failed") from exc
        if status == "Success":
            return "success"
        if status in TERMINAL_FAILURES:
            raise PollError(f"SSM command reached terminal status {status}")
        if status not in {
            "Pending",
            "InProgress",
            "Delayed",
        }:
            raise PollError("SSM command status is invalid")
        if time.monotonic() >= deadline:
            return "pending"
        time.sleep(min(30, max(0, deadline - time.monotonic())))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", required=True)
    parser.add_argument("--command-id", required=True)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--max-wait-seconds", type=int, required=True)
    parser.add_argument("--github-output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        status = poll(
            boto3.client("ssm", region_name=args.region),
            command_id=args.command_id,
            instance_id=args.instance_id,
            max_wait_seconds=args.max_wait_seconds,
        )
        with args.github_output.open("a", encoding="utf-8") as handle:
            handle.write(f"status={status}\n")
    except (OSError, PollError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"status": status}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
