"""Create and submit a miner's local agent source archive to the Arena."""

from __future__ import annotations

import base64
import getpass
import hashlib
import ipaddress
import os
import tempfile
import time
import warnings
import re
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.parse import urlsplit

import requests

from lab_arena import contracts, source_bundle


class MinerSubmissionError(RuntimeError):
    """A local archive or Arena submission failed with one stable reason."""

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        self.detail = detail
        super().__init__(detail or code)

    def format_for_cli(self, *, forbidden_values: Any = ()) -> str:
        """Return bounded diagnostic text with secrets and controls removed."""
        detail = str(self.detail or "")
        for value in forbidden_values or ():
            if isinstance(value, str) and value:
                detail = detail.replace(value, "[REDACTED]")
        detail = re.sub(
            r"[\x00-\x1f\x7f]",
            lambda match: "\\x%02x" % ord(match.group(0)),
            detail,
        )
        detail = detail[:240]
        return self.code if not detail else "%s (%s)" % (self.code, detail)


SUBMISSION_CREDENTIAL_ENV_VARS = {
    "openrouter_api_key": "OPENROUTER_API_KEY",
    "openrouter_management_key": "OPENROUTER_MANAGEMENT_KEY",
    "deepline_api_key": "DEEPLINE_API_KEY",
}
SUBMISSION_CREDENTIAL_MIN_LENGTH = 16
SUBMISSION_CREDENTIAL_MAX_LENGTH = 4096


def validate_submission_credentials(
    credentials: Mapping[str, str] | None,
) -> dict[str, str]:
    """Return the exact, non-empty credential mapping required for a model run."""

    if not isinstance(credentials, Mapping):
        raise MinerSubmissionError("submission_credentials_required")
    if set(credentials) != set(SUBMISSION_CREDENTIAL_ENV_VARS):
        raise MinerSubmissionError("submission_credentials_required")
    normalized: dict[str, str] = {}
    for name in SUBMISSION_CREDENTIAL_ENV_VARS:
        value = credentials.get(name)
        if (
            not isinstance(value, str)
            or value != value.strip()
            or not SUBMISSION_CREDENTIAL_MIN_LENGTH
            <= len(value)
            <= SUBMISSION_CREDENTIAL_MAX_LENGTH
        ):
            raise MinerSubmissionError("submission_credentials_required")
        normalized[name] = value
    return normalized


def submission_credentials_from_environment(
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Load all model credentials from environment variables or fail closed."""

    values = os.environ if environ is None else environ
    return validate_submission_credentials(
        {
            name: values.get(environment_name, "")
            for name, environment_name in SUBMISSION_CREDENTIAL_ENV_VARS.items()
        }
    )


def prompt_submission_credentials(
    *,
    environ: Mapping[str, str] | None = None,
    getpass_fn: Callable[[str], str] = getpass.getpass,
) -> dict[str, str]:
    """Read credentials from the environment or masked interactive prompts."""

    values = os.environ if environ is None else environ
    credentials: dict[str, str] = {}
    prompts = {
        "openrouter_api_key": "OpenRouter API key: ",
        "openrouter_management_key": "OpenRouter management key: ",
        "deepline_api_key": "Deepline API key: ",
    }
    for name, environment_name in SUBMISSION_CREDENTIAL_ENV_VARS.items():
        value = values.get(environment_name, "")
        if value:
            credentials[name] = value
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", getpass.GetPassWarning)
                credentials[name] = getpass_fn(prompts[name])
        except (getpass.GetPassWarning, EOFError) as exc:
            raise MinerSubmissionError("credential_prompt_unavailable") from exc
    return validate_submission_credentials(credentials)


def validate_agent_source(
    source_dir: str | Path, *, forbidden_values: Any = ()
) -> Path:
    """Validate the bounded source shape and Python syntax without importing it."""

    try:
        source = source_bundle.validate_source_directory(source_dir)
        files = source_bundle._source_files(source)
        secret_names = tuple(
            value for value in forbidden_values or () if isinstance(value, str) and value
        )
        for _candidate, relative_name, _details in files:
            if any(secret in relative_name for secret in secret_names):
                raise source_bundle.SourceBundleError(
                    "source_contains_credentials", path=relative_name
                )
        return source
    except source_bundle.SourceBundleError as exc:
        raise MinerSubmissionError(exc.code, exc.path or "") from exc


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


_SAFE_ADMISSION_ERRORS = frozenset({
    "hotkey_unregistered", "baseline_hotkey_reserved", "submission_window_closed",
    "submission_rate_limited", "submission_conflict", "submission_not_uploading",
    "source_upload_unavailable", "credential_validation_unavailable",
    "credential_kms_unavailable", "submission_credentials_missing",
    "submission_superseded", "submission_rejected:source_checksum_mismatch",
    "submission_rejected:source_contains_credentials",
    "submission_rejected:openrouter_api_key_invalid",
    "submission_rejected:openrouter_api_key_no_credit",
    "submission_rejected:openrouter_management_key_invalid",
    "submission_rejected:deepline_api_key_invalid",
    "submission_rejected:submission_credentials_invalid",
})


def _json_response(response: Any, operation: str) -> Mapping[str, Any]:
    try:
        document = response.json()
    except (TypeError, ValueError) as exc:
        raise MinerSubmissionError("arena_response_invalid", operation) from exc
    if not isinstance(document, Mapping):
        raise MinerSubmissionError("arena_response_invalid", operation)
    if not 200 <= int(response.status_code) < 300:
        # The gateway can include submitted credentials in a framework error.
        # Surface only fixed, known codes. Never echo arbitrary server text.
        code = document.get("code")
        if isinstance(code, str) and code in _SAFE_ADMISSION_ERRORS:
            raise MinerSubmissionError(code, operation)
        raise MinerSubmissionError("arena_request_failed", operation)
    return document


def find_open_round(api_base_url: str, *, session: Any = requests) -> str:
    """Return the one round that currently accepts submissions."""

    base = _api_base_url(api_base_url)
    response = None
    for attempt in range(3):
        try:
            response = session.get(base + "/arena/v1/current", timeout=30)
        except requests.RequestException as exc:
            if attempt == 2:
                raise MinerSubmissionError("arena_unreachable", type(exc).__name__) from exc
            time.sleep(1 << attempt)
            continue
        status = int(getattr(response, "status_code", 0))
        if status not in {502, 503, 504}:
            break
        if attempt == 2:
            raise MinerSubmissionError("arena_unreachable", "http_%d" % status)
        time.sleep(1 << attempt)
    assert response is not None
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
                allow_redirects=False,
            )
    except (OSError, requests.RequestException) as exc:
        raise MinerSubmissionError("source_upload_failed", type(exc).__name__) from exc
    # A retry can find that the first upload succeeded after its response was
    # lost. The signed finalize call will validate those stored bytes.
    if int(response.status_code) == 412:
        return
    if not 200 <= int(response.status_code) < 300:
        detail = "http_%d" % int(response.status_code)
        known_codes = {
            "AccessDenied", "BadDigest", "EntityTooLarge", "ExpiredToken",
            "InvalidAccessKeyId", "InvalidDigest", "InvalidRequest",
            "NoSuchBucket", "PreconditionFailed", "RequestTimeout", "SlowDown",
            "SignatureDoesNotMatch",
        }
        try:
            match = re.search(
                r"<Code>\s*([^<\s]{1,80})\s*</Code>",
                str(response.text)[:8192],
            )
        except (AttributeError, TypeError):
            match = None
        if match and match.group(1) in known_codes:
            detail += " code=%s" % match.group(1)
        request_id = ""
        value = response.headers.get("x-amz-request-id", "") if hasattr(response, "headers") else ""
        if isinstance(value, str) and re.fullmatch(r"[A-Z0-9]{16}", value):
            request_id = value
        if request_id:
            detail += " request_id=%s" % request_id
        raise MinerSubmissionError("source_upload_failed", detail)


def submit_agent_source(
    *,
    source_dir: str | Path,
    api_base_url: str,
    keypair: Any,
    credentials: Mapping[str, str] | None = None,
    session: Any = requests,
    now: Callable[[], float] = time.time,
) -> Mapping[str, Any]:
    """Archive, upload, and finalize one local agent fork."""

    submission_credentials = validate_submission_credentials(credentials)
    source = validate_agent_source(source_dir, forbidden_values=submission_credentials.values())
    base = _api_base_url(api_base_url)
    round_id = find_open_round(base, session=session)
    fd, raw_path = tempfile.mkstemp(prefix="lab-arena-source-", suffix=".tar.gz")
    os.close(fd)
    archive_path = Path(raw_path)
    try:
        try:
            descriptor = source_bundle.write_source_archive(source, archive_path)
            archive_bytes = archive_path.read_bytes()
            source_bundle.validate_source_archive(
                archive_bytes,
                forbidden_values=submission_credentials.values(),
            )
        except source_bundle.SourceBundleError as exc:
            raise MinerSubmissionError(exc.code, exc.path or "") from exc
        except OSError as exc:
            raise MinerSubmissionError("source_archive_failed") from exc
        presign_body = {
            **descriptor,
            "source_content_md5": base64.b64encode(
                hashlib.md5(archive_bytes, usedforsecurity=False).digest()
            ).decode("ascii"),
            "consent": {"public_rerun": True},
        }
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
                base + "/arena/v1/submissions/presign",
                json=presign,
                timeout=60,
                allow_redirects=False,
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
            "credentials": submission_credentials,
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
                allow_redirects=False,
            )
        except requests.RequestException as exc:
            raise MinerSubmissionError("arena_unreachable", type(exc).__name__) from exc
        result = _json_response(response, "submission_finalize")
        if result.get("status") != "accepted" or result.get("submission_id") != submission_id:
            raise MinerSubmissionError("arena_response_invalid", "submission_finalize")
        # Return a minimal submission result. Never propagate an echoed request body.
        return {
            "status": "accepted",
            "submission_id": submission_id,
            "round_id": round_id,
        }
    finally:
        archive_path.unlink(missing_ok=True)


def run_interactive_submission(
    keypair: Any,
    api_base_url: str,
    *,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
    getpass_fn: Callable[[str], str] = getpass.getpass,
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Collect local source and masked credentials, then submit the model."""

    output_fn("")
    output_fn("MODEL SUBMISSION")
    output_fn("Your source must contain harness.py with synchronous run_icp(icp).")
    output_fn("Do not put API keys in your source; credentials are sent separately.")
    output_fn("The OpenRouter API key and Deepline API key pay for model execution and scoring.")
    output_fn("The OpenRouter management key is used by the gateway only.")
    output_fn("If this agent wins, its source will be published as the public baseline.")
    source_dir = input_fn("Agent source directory: ").strip()
    if not source_dir:
        output_fn("Submission cancelled: a source directory is required.")
        return False
    if input_fn("Upload and submit this agent? [y/N]: ").strip().lower() not in {"y", "yes"}:
        output_fn("Submission cancelled.")
        return False
    try:
        credentials = prompt_submission_credentials(
            environ=environ,
            getpass_fn=getpass_fn,
        )
        result = submit_agent_source(
            source_dir=source_dir,
            api_base_url=api_base_url,
            keypair=keypair,
            credentials=credentials,
        )
    except MinerSubmissionError as exc:
        output_fn("Submission failed: %s" % exc.format_for_cli(forbidden_values=credentials.values() if 'credentials' in locals() else ()))
        return False
    output_fn("Submission accepted: %s" % result["submission_id"])
    output_fn("Round: %s" % result["round_id"])
    return True
