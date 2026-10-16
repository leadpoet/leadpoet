"""Finite diagnostics for weight reads; no response or exception text is emitted."""
from functools import wraps
from urllib.error import HTTPError

from leadpoet_canonical.chain_source_v2 import ChainSourceV2Error


# Exact literals from trusted parsers only. An unrecognized message stays unknown.
_REASONS = {
    "chain RPC response is invalid JSON": "rpc_invalid_json",
    "chain RPC response fields are invalid": "rpc_invalid_fields",
    "chain RPC response binding is invalid": "rpc_invalid_binding",
    "chain RPC returned an authenticated error": "rpc_error_response",
    "chain RPC result is missing": "rpc_result_missing",
    "finalized header is invalid": "header_invalid",
    "finalized header number is negative": "header_number_negative",
    "finalized header number is invalid": "header_number_invalid",
    "finalized block response fields are invalid": "block_fields_invalid",
    "finalized block body is invalid": "block_body_invalid",
    "finalized block number differs from request": "block_number_mismatch",
    "finalized block extrinsic set is invalid": "block_extrinsics_invalid",
    "finalized block extrinsic is invalid hex": "block_extrinsic_invalid_hex",
    "finalized block extrinsics exceed policy": "block_extrinsics_exceed_policy",
    "runtime metadata is invalid": "metadata_invalid",
    "runtime metadata is outside measured policy": "metadata_exceeds_policy",
    "runtime metadata is invalid hex": "metadata_invalid_hex",
    "last-update storage value is invalid": "last_update_invalid",
    "last-update storage value is empty": "last_update_empty",
    "last-update storage count exceeds policy": "last_update_exceeds_policy",
    "last-update storage value is invalid hex": "last_update_invalid_hex",
    "last-update storage has truncated": "last_update_truncated",
    "last-update storage has trailing bytes": "last_update_trailing_bytes",
    "weight storage value is invalid": "weights_invalid",
    "weight storage value is empty": "weights_empty",
    "weight storage count exceeds policy": "weights_exceed_policy",
    "weight storage value is invalid hex": "weights_invalid_hex",
    "weight storage contains duplicate UIDs": "weights_duplicate_uids",
    "weight storage has truncated": "weights_truncated",
    "weight storage has trailing bytes": "weights_trailing_bytes",
    "timelocked weight storage value is invalid": "commits_invalid",
    "timelocked weight storage value is empty": "commits_empty",
    "timelocked weight commit count exceeds policy": "commits_exceed_policy",
    "timelocked weight storage has trailing bytes": "commits_trailing_bytes",
    "timelocked weight storage value is invalid hex": "commits_invalid_hex",
    "timelocked weight account is truncated": "commit_account_truncated",
    "timelocked weight block is truncated": "commit_block_truncated",
    "timelocked commitment size is invalid": "commit_size_invalid",
    "timelocked commitment is truncated": "commit_truncated",
    "timelocked reveal round is truncated": "commit_round_truncated",
    "selective metagraph result is absent": "metagraph_absent",
    "selective metagraph result is invalid": "metagraph_invalid",
    "selective metagraph result is invalid hex": "metagraph_invalid_hex",
    "selective metagraph result layout is unsupported": "metagraph_layout_unsupported",
    "selective metagraph result has trailing bytes": "metagraph_trailing_bytes",
    "owner hotkey is absent": "metagraph_owner_absent",
    "owner hotkey is truncated": "metagraph_owner_truncated",
    "metagraph block is absent": "metagraph_block_absent",
    "metagraph hotkeys are absent": "metagraph_hotkeys_absent",
    "metagraph hotkey count is outside policy": "metagraph_count_exceeds_policy",
    "metagraph hotkeys are truncated": "metagraph_hotkeys_truncated",
    "host chain RPC transport is closed": "rpc_transport_closed",
    "host chain RPC retry deadline exhausted": "rpc_retry_deadline",
    "host chain RPC returned an HTTP error": "rpc_http_error",
    "host chain RPC returned a transient HTTP error": "rpc_http_transient",
    "host chain RPC response exceeds limit": "rpc_response_exceeds_policy",
    "host chain RPC HTTP response cleanup failed": "rpc_cleanup_failed",
    "host chain RPC exhausted transient HTTP retries": "rpc_retries_exhausted",
    "host chain RPC request failed": "rpc_request_failed",
}
_OPERATIONS = frozenset({
    "finalized_head", "signing_runtime", "account_nonce", "canonical_block_hash", "snapshot",
    "confirm_inclusion", "confirm_reveal",
})
_RPC_METHODS = frozenset({
    "chain_getFinalizedHead", "chain_getBlockHash", "chain_getBlock",
    "chain_getHeader", "state_getRuntimeVersion", "state_getMetadata",
    "state_getStorage", "state_getStorageHash", "state_call",
})
_RPC_OPERATIONS = frozenset(role + "_" + method
                            for role in ("live", "archive") for method in _RPC_METHODS)


def _trusted_errors(exc):
    # Late import avoids a cycle with the source methods that add context.
    from lab_arena.chain_source import ValidatorChainSourceV2Error
    seen = set()
    for _ in range(8):
        if exc is None or id(exc) in seen:
            break
        seen.add(id(exc))
        if type(exc) in (ChainSourceV2Error, ValidatorChainSourceV2Error):
            yield exc
        cause = exc.__cause__
        exc = cause if cause is not None else (None if exc.__suppress_context__ else exc.__context__)


def _set_context(exc, field, value):
    try:
        for error in _trusted_errors(exc):
            setattr(error, field, value)
    except Exception:
        pass  # Diagnostics must not replace the original failure.


def note_rpc_failure(exc, role, method):
    operation = role + "_" + method if type(role) is str and type(method) is str else ""
    if operation in _RPC_OPERATIONS:
        _set_context(exc, "_arena_weight_rpc_operation", operation)


def chain_read(operation):
    """Add a fixed read stage, then re-raise the same exception and cause chain."""
    assert operation in _OPERATIONS

    def decorate(function):
        @wraps(function)
        def read(*args, **kwargs):
            try:
                return function(*args, **kwargs)
            except Exception as exc:
                try:
                    _set_context(exc, "_arena_weight_read_operation", operation)
                except Exception:
                    pass
                raise
        return read
    return decorate


def chain_error_fields(exc):
    """Project only known stages and exact trusted parser literals into codes."""
    try:
        return _chain_error_fields(exc)
    except Exception:
        return {}  # Formatting is best effort, just like event delivery.


def _chain_error_fields(exc):
    operation = rpc = reason = http_status = None
    for error in _trusted_errors(exc):
        candidate = getattr(error, "_arena_weight_read_operation", None)
        if type(candidate) is str and candidate in _OPERATIONS:
            operation = candidate
        candidate = getattr(error, "_arena_weight_rpc_operation", None)
        if type(candidate) is str and candidate in _RPC_OPERATIONS:
            rpc = candidate
        # Never inspect arbitrary exception types, prefixes or substrings.
        candidate = (_REASONS.get(error.args[0])
                     if len(error.args) == 1 and type(error.args[0]) is str else None)
        if candidate is not None:
            reason = candidate
        # Only a native HTTPError directly caused by our trusted wrapper qualifies.
        # JSON-RPC error codes are discarded by the canonical parser and stay unknown.
        cause = error.__cause__
        if type(cause) is HTTPError and type(cause.code) is int and 100 <= cause.code <= 599:
            http_status = "http_" + str(cause.code)
    result = {"reason": reason} if reason is not None else {}
    if operation or rpc:
        # Keep a failure code visible even when reason carries prior_epoch_N.
        result["operation"] = ":".join(
            value for value in (operation, rpc, reason, http_status) if value)
    return result
