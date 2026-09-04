"""Create and submit a miner's local agent source archive to the Arena."""

from __future__ import annotations

import ipaddress
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.parse import urlsplit

import requests

from lab_arena import contracts, source_bundle


class MinerSubmissionError(RuntimeError):
    """A local archive or Arena submission failed with one stable reason."""

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        super().__init__(detail or code)


def validate_agent_source(source_dir: str | Path) -> Path:
    """Validate the bounded source shape and Python syntax without importing it."""

    try:
        return source_bundle.validate_source_directory(source_dir)
    except source_bundle.SourceBundleError as exc:
        raise MinerSubmissionError(exc.code) from exc


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


def _signed_request(
    *, scope: str, round_id: str, body: Mapping[str, Any], keypair: Any, now: Callable[[], float]
) -> Mapping[str, Any]:
    try:
        return contracts.build_signed_request(
            scope=scope,
            round_id=round_id,
            hotkey=keypair.ss58_address,
            body=body,
            timestamp=int(now()),
            sign_message=lambda message: keypair.sign(message.encode("utf-8")).hex(),
        )
    except (AttributeError, TypeError, contracts.ArenaContractError) as exc:
        raise MinerSubmissionError("wallet_signing_failed", type(exc).__name__) from exc


def _upload_source(
    session: Any,
    archive_path: Path,
    upload_url: str,
    upload_headers: Mapping[str, str],
) -> None:
    try:
        with archive_path.open("rb") as handle:
            response = session.put(
                upload_url,
                data=handle,
                headers=dict(upload_headers),
                timeout=300,
            )
    except (OSError, requests.RequestException) as exc:
        raise MinerSubmissionError("source_upload_failed", type(exc).__name__) from exc
    # A retry can find that the first upload succeeded after its response was
    # lost. The signed finalize call will validate those stored bytes.
    if int(response.status_code) == 412:
        return
    if not 200 <= int(response.status_code) < 300:
        raise MinerSubmissionError("source_upload_failed", "http_%d" % int(response.status_code))


def submit_agent_source(
    *,
    source_dir: str | Path,
    api_base_url: str,
    keypair: Any,
    session: Any = requests,
    now: Callable[[], float] = time.time,
) -> Mapping[str, Any]:
    """Archive, upload, and finalize one local agent fork."""

    source = validate_agent_source(source_dir)
    base = _api_base_url(api_base_url)
    round_id = find_open_round(base, session=session)
    fd, raw_path = tempfile.mkstemp(prefix="lab-arena-source-", suffix=".tar.gz")
    os.close(fd)
    archive_path = Path(raw_path)
    try:
        try:
            descriptor = source_bundle.write_source_archive(source, archive_path)
        except source_bundle.SourceBundleError as exc:
            raise MinerSubmissionError(exc.code) from exc
        presign_body = {**descriptor, "consent": {"public_rerun": True}}
        contracts.validate_submission_presign_body(presign_body)
        presign = _signed_request(
            scope=contracts.SCOPE_SUBMISSION_PRESIGN,
            round_id=round_id,
            body=presign_body,
            keypair=keypair,
            now=now,
        )
        try:
            response = session.post(
                base + "/arena/v1/submissions/presign", json=presign, timeout=60
            )
        except requests.RequestException as exc:
            raise MinerSubmissionError("arena_unreachable", type(exc).__name__) from exc
        target = _json_response(response, "submission_presign")
        submission_id = str(target.get("submission_id") or "")
        source_ref = str(target.get("source_ref") or "")
        finalize_body = {
            "submission_id": submission_id,
            "source_ref": source_ref,
            **descriptor,
        }
        try:
            contracts.validate_submission_finalize_body(finalize_body)
        except contracts.ArenaContractError as exc:
            raise MinerSubmissionError("arena_response_invalid", "submission_presign") from exc
        headers = target.get("upload_headers")
        if not isinstance(headers, Mapping) or not isinstance(target.get("upload_url"), str):
            raise MinerSubmissionError("arena_response_invalid", "submission_presign")
        _upload_source(session, archive_path, str(target["upload_url"]), headers)
        finalize = _signed_request(
            scope=contracts.SCOPE_SUBMISSION_FINALIZE,
            round_id=round_id,
            body=finalize_body,
            keypair=keypair,
            now=now,
        )
        try:
            response = session.post(
                "%s/arena/v1/submissions/%s/finalize" % (base, submission_id),
                json=finalize,
                timeout=60,
            )
        except requests.RequestException as exc:
            raise MinerSubmissionError("arena_unreachable", type(exc).__name__) from exc
        result = dict(_json_response(response, "submission_finalize"))
        if result.get("status") != "accepted" or result.get("submission_id") != submission_id:
            raise MinerSubmissionError("arena_response_invalid", "submission_finalize")
        result["round_id"] = round_id
        return result
    finally:
        archive_path.unlink(missing_ok=True)


def run_interactive_submission(
    keypair: Any,
    api_base_url: str,
    *,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> bool:
    """Ask only for the agent source directory, then submit it."""

    output_fn("")
    output_fn("AGENT COMPETITION SUBMISSION")
    output_fn("Your source must contain harness.py with synchronous run_icp(icp).")
    output_fn("The Arena supplies provider credentials. Do not put API keys in your source.")
    source_dir = input_fn("Agent source directory: ").strip()
    if not source_dir:
        output_fn("Submission cancelled: a source directory is required.")
        return False
    if input_fn("Upload and submit this agent? [y/N]: ").strip().lower() not in {"y", "yes"}:
        output_fn("Submission cancelled.")
        return False
    try:
        result = submit_agent_source(
            source_dir=source_dir,
            api_base_url=api_base_url,
            keypair=keypair,
        )
    except MinerSubmissionError as exc:
        output_fn("Submission failed: %s" % exc.code)
        return False
    output_fn("Submission accepted: %s" % result["submission_id"])
    output_fn("Round: %s" % result["round_id"])
    return True
