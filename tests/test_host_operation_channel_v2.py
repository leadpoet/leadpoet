from __future__ import annotations

import base64
from datetime import datetime, timezone
import threading

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from gateway.tee.host_operation_channel_v2 import (
    HostOperationChannelV2,
    HostOperationV2Error,
)
from leadpoet_canonical.attested_v2 import (
    build_boot_identity_body,
    create_boot_identity,
    validate_signed_host_operation_request,
    validate_signed_host_operation_terminal,
)


def _hash(character):
    return "sha256:" + character * 64


def _channel():
    key = Ed25519PrivateKey.generate()
    pubkey = key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    ).hex()
    boot = create_boot_identity(
        body=build_boot_identity_body(
            role="gateway_autoresearch",
            physical_role="gateway_autoresearch",
            commit_sha="a" * 40,
            pcr0="b" * 96,
            build_manifest_hash=_hash("c"),
            dependency_lock_hash=_hash("d"),
            config_hash=_hash("e"),
            boot_nonce="f" * 32,
            signing_pubkey=pubkey,
            transport_pubkey="1" * 64,
            transport_certificate_hash=_hash("2"),
            attestation_user_data_hash=_hash("3"),
            issued_at="2026-07-10T12:00:00Z",
        ),
        attestation_document_b64=base64.b64encode(b"nitro").decode(),
    )
    return HostOperationChannelV2(
        job_id="autoresearch-job-1",
        purpose="research_lab.candidate_build.v2",
        boot_identity=boot,
        sign_digest=key.sign,
        allowed_operations={"build_candidate_image"},
        clock=lambda: datetime(2026, 7, 10, 12, 0, 0, tzinfo=timezone.utc),
    )


def test_signed_command_blocks_until_one_valid_terminal() -> None:
    channel = _channel()
    observed = {}

    def run():
        observed["response"] = channel.execute(
            operation="build_candidate_image",
            payload={"source_diff_hash": _hash("4")},
            expected_state_hash=_hash("5"),
            timeout_seconds=5,
            response_validator=lambda value: {
                "candidate_manifest_hash": value["candidate_manifest_hash"]
            },
        )

    thread = threading.Thread(target=run)
    thread.start()
    command = channel.next_command(wait_ms=1000)
    assert command is not None
    validate_signed_host_operation_request(command["request"])
    terminal = channel.complete(
        request_hash=command["request"]["request_hash"],
        terminal_status="succeeded",
        response={"candidate_manifest_hash": _hash("6")},
    )
    validate_signed_host_operation_terminal(terminal)
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert observed["response"] == {"candidate_manifest_hash": _hash("6")}
    assert channel.terminal_hashes() == (terminal["terminal_hash"],)


def test_host_cannot_complete_unknown_or_complete_twice() -> None:
    channel = _channel()
    with pytest.raises(HostOperationV2Error, match="not found"):
        channel.complete(
            request_hash=_hash("a"),
            terminal_status="failed",
            response=None,
            failure_code="host_failed",
        )


def test_unallowlisted_host_operation_fails_before_emission() -> None:
    channel = _channel()
    with pytest.raises(HostOperationV2Error, match="allowlisted"):
        channel.execute(
            operation="arbitrary_shell",
            payload={},
            expected_state_hash=_hash("5"),
            timeout_seconds=1,
            response_validator=lambda value: value,
        )
    assert channel.ledger() == ()


def test_failed_terminal_is_signed_and_propagated() -> None:
    channel = _channel()
    observed = {}

    def run():
        try:
            channel.execute(
                operation="build_candidate_image",
                payload={"source_diff_hash": _hash("4")},
                expected_state_hash=_hash("5"),
                timeout_seconds=5,
                response_validator=lambda value: value,
            )
        except HostOperationV2Error as exc:
            observed["error"] = exc

    thread = threading.Thread(target=run)
    thread.start()
    command = channel.next_command(wait_ms=1000)
    terminal = channel.complete(
        request_hash=command["request"]["request_hash"],
        terminal_status="failed",
        response={"diagnostic_hash": _hash("7")},
        failure_code="candidate_build_failed",
    )
    thread.join(timeout=2)
    assert observed["error"].terminal == terminal
    validate_signed_host_operation_terminal(terminal)


def test_response_normalization_cannot_change_signed_hash() -> None:
    channel = _channel()
    observed = {}

    def run():
        try:
            channel.execute(
                operation="build_candidate_image",
                payload={"source_diff_hash": _hash("4")},
                expected_state_hash=_hash("5"),
                timeout_seconds=5,
                response_validator=lambda value: {
                    "candidate_manifest_hash": value["candidate_manifest_hash"]
                },
            )
        except HostOperationV2Error as exc:
            observed["error"] = exc

    thread = threading.Thread(target=run)
    thread.start()
    command = channel.next_command(wait_ms=1000)
    channel.complete(
        request_hash=command["request"]["request_hash"],
        terminal_status="succeeded",
        response={"candidate_manifest_hash": _hash("6"), "extra": "host-added"},
    )
    thread.join(timeout=2)
    assert "response_validation_failed" in str(observed["error"])
    assert observed["error"].terminal["terminal_status"] == "failed"


def test_close_signs_cancel_terminal_and_rejects_new_operations() -> None:
    channel = _channel()
    observed = {}

    def run():
        try:
            channel.execute(
                operation="build_candidate_image",
                payload={"source_diff_hash": _hash("4")},
                expected_state_hash=_hash("5"),
                timeout_seconds=5,
                response_validator=lambda value: value,
            )
        except HostOperationV2Error as exc:
            observed["error"] = exc

    thread = threading.Thread(target=run)
    thread.start()
    assert channel.next_command(wait_ms=1000) is not None
    channel.close()
    thread.join(timeout=2)
    assert observed["error"].terminal["failure_code"] == "cancelled"
    assert len(channel.complete_ledger()) == 1
    with pytest.raises(HostOperationV2Error, match="closed"):
        channel.execute(
            operation="build_candidate_image",
            payload={},
            expected_state_hash=_hash("5"),
            timeout_seconds=1,
            response_validator=lambda value: value,
        )
