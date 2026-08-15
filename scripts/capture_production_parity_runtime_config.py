#!/usr/bin/env python3
"""Capture the canonical secret-free production execution configuration."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterator, Mapping, Sequence

import boto3
from botocore.exceptions import BotoCoreError, ClientError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.materialize_production_parity_secrets import (
    SecretMaterializationError,
    _parse_environment_document,
)
from gateway.tee.research_lab_runtime_config_v2 import (
    ResearchLabRuntimeConfigV2Error,
    build_research_lab_execution_config,
)


RUNTIME_PREFIXES = ("RESEARCH_LAB_", "BITTENSOR_", "LEADPOET_")
RUNTIME_EXACT_NAMES = {"NETUID", "SUBTENSOR_NETWORK"}


class RuntimeConfigCaptureError(RuntimeError):
    pass


def _secret_string(client: Any, secret_id: str) -> str:
    try:
        value = client.get_secret_value(SecretId=secret_id).get("SecretString")
    except (BotoCoreError, ClientError) as exc:
        raise RuntimeConfigCaptureError(
            "production runtime environment is unavailable"
        ) from exc
    if not isinstance(value, str) or not value:
        raise RuntimeConfigCaptureError(
            "production runtime environment is invalid"
        )
    return value


@contextmanager
def _isolated_runtime_environment(values: Mapping[str, str]) -> Iterator[None]:
    """Make candidate config resolution independent of runner configuration."""

    removed = {
        key: value
        for key, value in os.environ.items()
        if key.startswith(RUNTIME_PREFIXES) or key in RUNTIME_EXACT_NAMES
    }
    for key in removed:
        os.environ.pop(key, None)
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update({str(key): str(value) for key, value in values.items()})
    try:
        yield
    finally:
        for key in values:
            old = previous[key]
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old
        os.environ.update(removed)


def canonical_runtime_config(values: Mapping[str, str]) -> dict[str, Any]:
    environment = {str(key): str(value) for key, value in values.items()}
    if not (
        environment.get("LEADPOET_SUBNET_EPOCH_CUTOVER_JSON")
        or environment.get("LEADPOET_SUBNET_EPOCH_CUTOVER_PATH")
    ):
        cutover_path = ROOT / "config/stateful-epoch-cutover-sn71.json"
        try:
            cutover = json.loads(cutover_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RuntimeConfigCaptureError(
                "candidate epoch cutover configuration is unavailable"
            ) from exc
        environment["LEADPOET_SUBNET_EPOCH_CUTOVER_JSON"] = json.dumps(
            cutover, sort_keys=True, separators=(",", ":")
        )
    try:
        with _isolated_runtime_environment(environment):
            execution_config = build_research_lab_execution_config(
                environment=environment
            )
    except ResearchLabRuntimeConfigV2Error as exc:
        raise RuntimeConfigCaptureError(
            "production runtime configuration is not V2-classified"
        ) from exc
    return {"execution_config": execution_config}


def capture(*, client: Any, secret_id: str, output: Path) -> dict[str, Any]:
    try:
        parsed = _parse_environment_document(
            _secret_string(client, secret_id),
            field="production gateway environment",
        )
    except SecretMaterializationError as exc:
        raise RuntimeConfigCaptureError(
            "production runtime environment could not be parsed"
        ) from exc
    document = canonical_runtime_config(parsed)
    payload = (
        json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("ascii")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    output.chmod(0o600)
    return {
        "behavior_key_count": len(
            document["execution_config"]["behavior_environment"]
        ),
        "field_count": len(document["execution_config"]["fields"]),
        "sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--secret-id", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = capture(
            client=boto3.client("secretsmanager", region_name=args.region),
            secret_id=str(args.secret_id),
            output=args.output,
        )
    except (OSError, ValueError, BotoCoreError, ClientError, RuntimeConfigCaptureError):
        print("ERROR: production behavior configuration capture failed", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
