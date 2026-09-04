"""Host-owned entrypoint for one untrusted Arena source bundle."""

from __future__ import annotations

import importlib.util
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any

SOURCE_DIR = Path("/agent/source")
DEPENDENCY_DIR = Path("/agent/deps")
INPUT_PATH = Path("/input/icp.json")
OUTPUT_PATH = Path("/output/companies.json")


class AgentContractError(RuntimeError):
    """The submitted harness does not implement the public boundary."""


def _load_run_icp(source_dir: Path) -> Any:
    harness_path = source_dir / "harness.py"
    if not harness_path.is_file():
        raise AgentContractError("harness.py is missing")
    sys.path.insert(0, str(DEPENDENCY_DIR))
    sys.path.insert(0, str(source_dir))
    specification = importlib.util.spec_from_file_location(
        "_lab_arena_submission_harness", harness_path
    )
    if specification is None or specification.loader is None:
        raise AgentContractError("harness.py cannot be loaded")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    function = getattr(module, "run_icp", None)
    if not callable(function) or inspect.iscoroutinefunction(function):
        raise AgentContractError("harness.run_icp must be synchronous")
    try:
        parameters = list(inspect.signature(function).parameters.values())
    except (TypeError, ValueError) as exc:
        raise AgentContractError("harness.run_icp signature is unavailable") from exc
    positional = [
        parameter
        for parameter in parameters
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if (
        len(positional) != 1
        or any(parameter.kind == inspect.Parameter.KEYWORD_ONLY for parameter in parameters)
        or any(
            parameter.kind
            in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
            for parameter in parameters
        )
    ):
        raise AgentContractError("harness.run_icp must take exactly one positional input")
    return function


def run(
    *,
    source_dir: Path = SOURCE_DIR,
    input_path: Path = INPUT_PATH,
    output_path: Path = OUTPUT_PATH,
) -> None:
    """Call ``harness.run_icp(icp)`` and write the standard output document."""

    document = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or not isinstance(document.get("icp"), dict):
        raise AgentContractError("ICP input is invalid")
    company_limit = document.get("company_limit", 5)
    if (
        isinstance(company_limit, bool)
        or not isinstance(company_limit, int)
        or not 1 <= company_limit <= 5
    ):
        raise AgentContractError("company_limit must be an integer from 1 through 5")
    previous_limit = os.environ.get("LAB_ARENA_COMPANY_LIMIT")
    os.environ["LAB_ARENA_COMPANY_LIMIT"] = str(company_limit)
    try:
        result = _load_run_icp(source_dir)(dict(document["icp"]))
    finally:
        if previous_limit is None:
            os.environ.pop("LAB_ARENA_COMPANY_LIMIT", None)
        else:
            os.environ["LAB_ARENA_COMPANY_LIMIT"] = previous_limit
    if inspect.isawaitable(result):
        raise AgentContractError("harness.run_icp must be synchronous")
    if not isinstance(result, list) or any(not isinstance(item, dict) for item in result):
        raise AgentContractError("harness.run_icp must return a list of company objects")
    output_path.write_text(
        json.dumps({"companies": result}, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )


def main() -> int:
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
