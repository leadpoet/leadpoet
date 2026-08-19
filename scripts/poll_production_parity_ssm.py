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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.production_parity_bootstrap_evidence import (
    BootstrapEvidenceError,
    bootstrap_failure_identity_from_response_code,
    retain_failure,
)


ID_RE = re.compile(r"^[A-Za-z0-9-]{8,128}$")
TERMINAL_FAILURES = {
    "Failed",
    "Cancelled",
    "TimedOut",
    "Cancelling",
    "DeliveryTimedOut",
}
TERMINAL_ERROR_CATEGORIES = {
    "Failed": "SsmFailed",
    "Cancelled": "SsmCancelled",
    "TimedOut": "SsmTimedOut",
    "Cancelling": "SsmCancelling",
    "DeliveryTimedOut": "SsmDeliveryTimedOut",
}
ARTIFACT_BUCKET_RE = re.compile(r"^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$")


class PollError(RuntimeError):
    pass


class TerminalPollError(PollError):
    def __init__(self, status: str, *, response_code: object = None) -> None:
        if status not in TERMINAL_FAILURES:
            raise ValueError("terminal SSM status is invalid")
        super().__init__(f"SSM command reached terminal status {status}")
        self.status = status
        self.response_code = response_code if type(response_code) is int else None


def terminal_failure_identity(
    *, status: str, response_code: object
) -> tuple[str, str]:
    if status == "Failed":
        exact = bootstrap_failure_identity_from_response_code(response_code)
        if exact is not None:
            return exact
    return "ssm-command", TERMINAL_ERROR_CATEGORIES[status]


def retain_terminal_failure(
    s3_client: Any,
    *,
    output: Path,
    artifact_bucket: str,
    run_id: str,
    base_sha: str,
    candidate_sha: str,
    status: str,
    response_code: object = None,
) -> None:
    if ARTIFACT_BUCKET_RE.fullmatch(artifact_bucket) is None:
        raise PollError("bounded evidence bucket identity is invalid")
    try:
        stage, error_category = terminal_failure_identity(
            status=status,
            response_code=response_code,
        )
        payload, created = retain_failure(
            output=output,
            run_id=run_id,
            base_sha=base_sha,
            candidate_sha=candidate_sha,
            stage=stage,
            error_category=error_category,
        )
    except (BootstrapEvidenceError, KeyError, OSError, ValueError) as exc:
        raise PollError("bounded terminal evidence retention failed") from exc
    if not created:
        return
    key = f"production-parity/runs/{run_id}/full-evidence.json"
    try:
        s3_client.put_object(
            Bucket=artifact_bucket,
            Key=key,
            Body=payload,
            ContentType="application/json",
            IfNoneMatch="*",
        )
    except ClientError as exc:
        if str(exc.response.get("Error", {}).get("Code") or "") in {
            "PreconditionFailed",
            "ConditionalRequestConflict",
        }:
            return
        raise PollError("bounded terminal evidence upload failed") from exc
    except BotoCoreError as exc:
        raise PollError("bounded terminal evidence upload failed") from exc


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
            raise TerminalPollError(
                status,
                response_code=response.get("ResponseCode"),
            )
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
    parser.add_argument("--evidence-output", type=Path, required=True)
    parser.add_argument("--artifact-bucket", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--base-sha", required=True)
    parser.add_argument("--candidate-sha", required=True)
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
    except TerminalPollError as exc:
        try:
            retain_terminal_failure(
                boto3.client("s3", region_name=args.region),
                output=args.evidence_output,
                artifact_bucket=args.artifact_bucket,
                run_id=args.run_id,
                base_sha=args.base_sha,
                candidate_sha=args.candidate_sha,
                status=exc.status,
                response_code=exc.response_code,
            )
        except PollError:
            print(
                "ERROR: SSM terminal failure evidence retention failed",
                file=sys.stderr,
            )
            return 1
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except (OSError, PollError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"status": status}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
