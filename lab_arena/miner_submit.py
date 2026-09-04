"""Build and submit a miner's local agent source to the Arena.

The public callable is ``harness.run_icp(icp)``.  The miner helper only checks
that the callable is present.  It does not import or run miner code on the
host.  OCI is the transport format: the helper builds and pushes the miner's
fork, then sends the existing signed Arena submission request.
"""

from __future__ import annotations

import ast
import ipaddress
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Optional
from urllib.parse import urlsplit

import requests

from lab_arena import contracts, images


class MinerSubmissionError(RuntimeError):
    """A local build or Arena submission failed with one stable reason."""

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        super().__init__(detail or code)


def validate_agent_source(source_dir: str | Path) -> Path:
    """Validate the small source boundary without executing miner code."""

    source = Path(source_dir).expanduser().resolve()
    if not source.is_dir():
        raise MinerSubmissionError("source_directory_missing")

    harness_path = source / "harness.py"
    if not harness_path.is_file():
        raise MinerSubmissionError("harness_file_missing")
    if not (source / "Dockerfile").is_file():
        raise MinerSubmissionError("dockerfile_missing")

    try:
        tree = ast.parse(harness_path.read_text(encoding="utf-8"), filename=str(harness_path))
    except (OSError, UnicodeError, SyntaxError) as exc:
        raise MinerSubmissionError("harness_invalid", type(exc).__name__) from exc

    definitions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "run_icp"
    ]
    if not definitions:
        raise MinerSubmissionError("run_icp_missing")
    definition = definitions[-1]
    if isinstance(definition, ast.AsyncFunctionDef):
        raise MinerSubmissionError("run_icp_must_be_sync")

    positional = list(definition.args.posonlyargs) + list(definition.args.args)
    if not positional:
        raise MinerSubmissionError("run_icp_input_missing")
    required_positional = len(positional) - len(definition.args.defaults)
    required_keyword_only = sum(default is None for default in definition.args.kw_defaults)
    if required_positional > 1 or required_keyword_only:
        raise MinerSubmissionError("run_icp_has_required_extra_inputs")
    return source


def _api_base_url(value: str) -> str:
    base = str(value or "").strip().rstrip("/")
    parsed = urlsplit(base)
    if not parsed.hostname or parsed.query or parsed.fragment or parsed.username or parsed.password:
        raise MinerSubmissionError("arena_url_invalid")
    if parsed.scheme == "https":
        return base
    if parsed.scheme != "http":
        raise MinerSubmissionError("arena_url_must_use_https")
    try:
        loopback = ipaddress.ip_address(parsed.hostname).is_loopback
    except ValueError:
        loopback = parsed.hostname == "localhost"
    if not loopback:
        raise MinerSubmissionError("arena_url_must_use_https")
    return base


def _json_response(response: Any, operation: str) -> Mapping[str, Any]:
    try:
        document = response.json()
    except (TypeError, ValueError) as exc:
        raise MinerSubmissionError("arena_response_invalid", operation) from exc
    if not isinstance(document, Mapping):
        raise MinerSubmissionError("arena_response_invalid", operation)
    if not 200 <= int(response.status_code) < 300:
        reason = str(document.get("code") or document.get("detail") or "request_failed")[:120]
        raise MinerSubmissionError("arena_request_failed", reason)
    return document


def find_open_round(api_base_url: str, *, session: Any = requests) -> str:
    """Return the one round that currently accepts submissions."""

    base = _api_base_url(api_base_url)
    try:
        response = session.get(base + "/arena/v1/current", timeout=30)
    except requests.RequestException as exc:
        raise MinerSubmissionError("arena_unreachable", type(exc).__name__) from exc
    document = _json_response(response, "current_round")
    open_round = document.get("open_round")
    if not isinstance(open_round, Mapping) or open_round.get("status") != "open":
        raise MinerSubmissionError("submission_window_closed")
    round_id = open_round.get("round_id")
    if not isinstance(round_id, str) or not contracts.ROUND_ID_RE.match(round_id):
        raise MinerSubmissionError("arena_response_invalid", "open_round")
    return round_id


def build_and_push_agent(
    source_dir: Path,
    image_reference: str,
    *,
    command_runner: Optional[Callable[..., Any]] = None,
) -> None:
    """Build one Linux AMD64 image and push it with the user's Docker login."""

    if command_runner is None:
        if shutil.which("docker") is None:
            raise MinerSubmissionError("docker_not_found")
        command_runner = subprocess.run

    commands = (
        ["docker", "build", "--platform", "linux/amd64", "--tag", image_reference, str(source_dir)],
        ["docker", "push", image_reference],
    )
    for command in commands:
        try:
            command_runner(command, check=True, timeout=1800)
        except FileNotFoundError as exc:
            raise MinerSubmissionError("docker_not_found") from exc
        except subprocess.TimeoutExpired as exc:
            raise MinerSubmissionError("docker_command_timed_out", command[1]) from exc
        except subprocess.CalledProcessError as exc:
            raise MinerSubmissionError("docker_command_failed", command[1]) from exc


def submit_agent_source(
    *,
    source_dir: str | Path,
    image_reference: str,
    api_base_url: str,
    keypair: Any,
    session: Any = requests,
    command_runner: Optional[Callable[..., Any]] = None,
    now: Callable[[], float] = time.time,
) -> Mapping[str, Any]:
    """Build, push, sign, and submit one local agent fork."""

    source = validate_agent_source(source_dir)
    try:
        reference = images.parse_reference(image_reference)
    except images.ImageError as exc:
        raise MinerSubmissionError("image_reference_invalid", exc.rule_id) from exc
    if reference.digest is not None or reference.tag is None:
        raise MinerSubmissionError("image_build_target_must_be_tag")

    base = _api_base_url(api_base_url)
    round_id = find_open_round(base, session=session)
    build_and_push_agent(source, str(reference), command_runner=command_runner)

    body = {"image_reference": str(reference), "consent": {"public_rerun": True}}
    contracts.validate_submission_body(body)
    try:
        hotkey = keypair.ss58_address
        envelope = contracts.build_signed_request(
            scope=contracts.SCOPE_SUBMISSION,
            round_id=round_id,
            hotkey=hotkey,
            body=body,
            timestamp=int(now()),
            sign_message=lambda message: keypair.sign(message.encode("utf-8")).hex(),
        )
    except (AttributeError, TypeError, contracts.ArenaContractError) as exc:
        raise MinerSubmissionError("wallet_signing_failed", type(exc).__name__) from exc

    try:
        response = session.post(base + "/arena/v1/submissions", json=envelope, timeout=60)
    except requests.RequestException as exc:
        raise MinerSubmissionError("arena_unreachable", type(exc).__name__) from exc
    result = dict(_json_response(response, "submission"))
    if not isinstance(result.get("submission_id"), str) or not result["submission_id"]:
        raise MinerSubmissionError("arena_response_invalid", "submission")
    result["round_id"] = round_id
    return result


def run_interactive_submission(
    keypair: Any,
    api_base_url: str,
    *,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> bool:
    """Ask only for source and its public image target, then submit it."""

    output_fn("")
    output_fn("AGENT COMPETITION SUBMISSION")
    output_fn("Your source must contain harness.py with run_icp(icp) and a Dockerfile.")
    output_fn("The Arena supplies provider credentials. Do not put API keys in your source.")
    source_dir = input_fn("Agent source directory: ").strip()
    image_reference = input_fn("Public image tag, for example ghcr.io/you/agent:v1: ").strip()
    if not source_dir or not image_reference:
        output_fn("Submission cancelled: source directory and image tag are required.")
        return False
    if input_fn("Build, push, and submit this agent? [y/N]: ").strip().lower() not in {"y", "yes"}:
        output_fn("Submission cancelled.")
        return False
    try:
        result = submit_agent_source(
            source_dir=source_dir,
            image_reference=image_reference,
            api_base_url=api_base_url,
            keypair=keypair,
        )
    except MinerSubmissionError as exc:
        output_fn("Submission failed: %s" % exc.code)
        return False
    output_fn("Submission accepted: %s" % result["submission_id"])
    output_fn("Round: %s" % result["round_id"])
    return True
