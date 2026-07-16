"""Exact C-ABI bridge for enclave-generated Bittensor drand commitments.

The official ``bittensor-drand`` 2.0.0 source exposes
``cr_generate_commit_v2`` for the stateful epoch model.
Its published Python wheel targets CPython 3.9 and must not be imported into
the validator's pinned CPython 3.7 enclave.  Production therefore supplies a
separately rebuilt C-ABI-only shared object whose hash is part of the measured
validator release.  This module refuses unpinned libraries and has no Python
or host fallback.
"""

from __future__ import annotations

import ctypes
import hashlib
from pathlib import Path
import re
from typing import Any, Callable, Sequence, Tuple


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class DrandCommitV2Error(RuntimeError):
    """The measured drand helper is absent, corrupt, or rejected a request."""


class _CRByteBuffer(ctypes.Structure):
    _fields_ = [
        ("ptr", ctypes.POINTER(ctypes.c_uint8)),
        ("len", ctypes.c_size_t),
        ("cap", ctypes.c_size_t),
    ]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise DrandCommitV2Error("drand helper is unavailable") from exc
    return digest.hexdigest()


class CtypesDrandCommitBackendV2:
    """Call only the pinned stateful ``cr_generate_commit_v2`` C ABI."""

    def __init__(
        self,
        *,
        library_path: Path,
        expected_sha256: str,
        library_loader: Callable[[str], Any] = ctypes.CDLL,
    ) -> None:
        digest = str(expected_sha256 or "").strip().lower()
        if not _SHA256_RE.fullmatch(digest):
            raise DrandCommitV2Error("drand helper hash is invalid")
        path = Path(library_path)
        if file_sha256(path) != digest:
            raise DrandCommitV2Error("drand helper hash mismatch")
        try:
            library = library_loader(str(path))
            generate = library.cr_generate_commit_v2
            free_buffer = library.cr_free
            free_string = library.cr_free_str
        except Exception as exc:
            raise DrandCommitV2Error("drand helper C ABI is unavailable") from exc
        generate.argtypes = [
            ctypes.POINTER(ctypes.c_uint16),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint16),
            ctypes.c_size_t,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint16,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_double,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_char_p),
        ]
        generate.restype = _CRByteBuffer
        free_buffer.argtypes = [_CRByteBuffer]
        free_buffer.restype = None
        free_string.argtypes = [ctypes.c_void_p]
        free_string.restype = None
        self._generate = generate
        self._free_buffer = free_buffer
        self._free_string = free_string
        self.library_path = path
        self.library_sha256 = digest

    def generate_commit(
        self,
        *,
        uids: Sequence[int],
        weights_u16: Sequence[int],
        version_key: int,
        last_epoch_block: int,
        pending_epoch_at: int,
        subnet_epoch_index: int,
        tempo: int,
        blocks_since_last_step: int,
        current_block: int,
        subnet_reveal_period_epochs: int,
        block_time: float,
        hotkey_public_key: bytes,
    ) -> Tuple[bytes, int]:
        normalized_uids = [int(value) for value in uids]
        normalized_weights = [int(value) for value in weights_u16]
        if (
            not normalized_uids
            or len(normalized_uids) != len(normalized_weights)
            or normalized_uids != sorted(normalized_uids)
            or len(set(normalized_uids)) != len(normalized_uids)
            or any(value < 0 or value > 65535 for value in normalized_uids)
            or any(value < 1 or value > 65535 for value in normalized_weights)
        ):
            raise DrandCommitV2Error("drand sparse weights are invalid")
        numeric = (
            int(version_key),
            int(last_epoch_block),
            int(pending_epoch_at),
            int(subnet_epoch_index),
            int(tempo),
            int(blocks_since_last_step),
            int(current_block),
            int(subnet_reveal_period_epochs),
        )
        if any(value < 0 for value in numeric):
            raise DrandCommitV2Error("drand chain parameter is negative")
        if numeric[4] <= 0 or numeric[4] > 65535:
            raise DrandCommitV2Error("drand tempo is invalid")
        if numeric[1] > numeric[6]:
            raise DrandCommitV2Error("drand epoch state is invalid")
        if any(value > 2**64 - 1 for value in numeric):
            raise DrandCommitV2Error("drand chain parameter is too large")
        if not isinstance(block_time, (int, float)) or not 0 < float(block_time) <= 120:
            raise DrandCommitV2Error("drand block time is invalid")
        hotkey = bytes(hotkey_public_key)
        if len(hotkey) != 32:
            raise DrandCommitV2Error("drand hotkey public key is invalid")

        uid_array = (ctypes.c_uint16 * len(normalized_uids))(*normalized_uids)
        weight_array = (ctypes.c_uint16 * len(normalized_weights))(
            *normalized_weights
        )
        hotkey_array = (ctypes.c_uint8 * len(hotkey)).from_buffer_copy(hotkey)
        reveal_round = ctypes.c_uint64(0)
        error_pointer = ctypes.c_char_p()
        buffer = self._generate(
            uid_array,
            len(normalized_uids),
            weight_array,
            len(normalized_weights),
            numeric[0],
            numeric[1],
            numeric[2],
            numeric[3],
            numeric[4],
            numeric[5],
            numeric[6],
            numeric[7],
            float(block_time),
            hotkey_array,
            len(hotkey),
            ctypes.byref(reveal_round),
            ctypes.byref(error_pointer),
        )
        try:
            if error_pointer.value:
                try:
                    message = error_pointer.value.decode("utf-8", errors="replace")
                finally:
                    self._free_string(ctypes.cast(error_pointer, ctypes.c_void_p))
                raise DrandCommitV2Error("drand helper rejected request: %s" % message)
            if not bool(buffer.ptr) or buffer.len <= 0 or buffer.len > 1024 * 1024:
                raise DrandCommitV2Error("drand helper returned an invalid commitment")
            commitment = ctypes.string_at(buffer.ptr, buffer.len)
            if reveal_round.value <= 0:
                raise DrandCommitV2Error("drand helper returned an invalid reveal round")
            return bytes(commitment), int(reveal_round.value)
        finally:
            if bool(buffer.ptr):
                self._free_buffer(buffer)
