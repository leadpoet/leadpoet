import ctypes
import hashlib

import pytest

from validator_tee.enclave.drand_v2 import (
    CtypesDrandCommitBackendV2,
    DrandCommitV2Error,
    _CRByteBuffer,
)


class _Function:
    def __init__(self, callback):
        self.callback = callback
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        return self.callback(*args)


class _Library:
    def __init__(self, *, error=None):
        self.calls = []
        self.freed = []
        self._buffers = []
        self._error = error
        self.cr_generate_commit_v2 = _Function(self._generate)
        self.cr_free = _Function(self.freed.append)
        self.cr_free_str = _Function(lambda pointer: self.freed.append(pointer))

    def _generate(
        self,
        uids,
        uids_len,
        weights,
        weights_len,
        version_key,
        last_epoch_block,
        pending_epoch_at,
        subnet_epoch_index,
        tempo,
        blocks_since_last_step,
        current_block,
        reveal_epochs,
        block_time,
        hotkey,
        hotkey_len,
        round_out,
        error_out,
    ):
        self.calls.append(
            {
                "uids": [uids[index] for index in range(uids_len)],
                "weights": [weights[index] for index in range(weights_len)],
                "version_key": int(version_key),
                "last_epoch_block": int(last_epoch_block),
                "pending_epoch_at": int(pending_epoch_at),
                "subnet_epoch_index": int(subnet_epoch_index),
                "tempo": int(tempo),
                "blocks_since_last_step": int(blocks_since_last_step),
                "current_block": int(current_block),
                "reveal_epochs": int(reveal_epochs),
                "block_time": float(block_time),
                "hotkey": bytes(hotkey[index] for index in range(hotkey_len)),
            }
        )
        if self._error:
            value = ctypes.create_string_buffer(self._error.encode())
            self._buffers.append(value)
            error_pointer = ctypes.cast(error_out, ctypes.POINTER(ctypes.c_char_p))
            error_pointer[0] = ctypes.cast(value, ctypes.c_char_p)
            return _CRByteBuffer()
        commitment = b"measured-encrypted-commitment"
        buffer = (ctypes.c_uint8 * len(commitment))(*commitment)
        self._buffers.append(buffer)
        ctypes.cast(round_out, ctypes.POINTER(ctypes.c_uint64))[0] = 998877
        return _CRByteBuffer(
            ctypes.cast(buffer, ctypes.POINTER(ctypes.c_uint8)),
            len(commitment),
            len(commitment),
        )


def _backend(tmp_path, library):
    path = tmp_path / "libdrand.so"
    path.write_bytes(b"exact-rebuilt-helper")
    return CtypesDrandCommitBackendV2(
        library_path=path,
        expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        library_loader=lambda _path: library,
    )


def test_drand_backend_passes_exact_weight_commit_inputs_and_frees_buffer(tmp_path):
    library = _Library()
    backend = _backend(tmp_path, library)
    commitment, reveal_round = backend.generate_commit(
        uids=[0, 14],
        weights_u16=[65535, 3210],
        version_key=901,
        last_epoch_block=8596445,
        pending_epoch_at=0,
        subnet_epoch_index=23859,
        tempo=360,
        blocks_since_last_step=263,
        current_block=8596708,
        subnet_reveal_period_epochs=1,
        block_time=12.0,
        hotkey_public_key=b"h" * 32,
    )
    assert commitment == b"measured-encrypted-commitment"
    assert reveal_round == 998877
    assert library.calls == [
        {
            "uids": [0, 14],
            "weights": [65535, 3210],
            "version_key": 901,
            "last_epoch_block": 8596445,
            "pending_epoch_at": 0,
            "subnet_epoch_index": 23859,
            "tempo": 360,
            "blocks_since_last_step": 263,
            "current_block": 8596708,
            "reveal_epochs": 1,
            "block_time": 12.0,
            "hotkey": b"h" * 32,
        }
    ]
    assert len(library.freed) == 1


def test_drand_backend_rejects_hash_mismatch_before_loading(tmp_path):
    path = tmp_path / "libdrand.so"
    path.write_bytes(b"wrong")
    with pytest.raises(DrandCommitV2Error, match="hash mismatch"):
        CtypesDrandCommitBackendV2(
            library_path=path,
            expected_sha256="0" * 64,
            library_loader=lambda _path: (_ for _ in ()).throw(
                AssertionError("untrusted library loaded")
            ),
        )


@pytest.mark.parametrize(
    "uids,weights,error",
    [
        ([14, 0], [1, 2], "sparse"),
        ([0, 0], [1, 2], "sparse"),
        ([0], [0], "sparse"),
        ([0], [1, 2], "sparse"),
    ],
)
def test_drand_backend_rejects_noncanonical_vector(tmp_path, uids, weights, error):
    backend = _backend(tmp_path, _Library())
    with pytest.raises(DrandCommitV2Error, match=error):
        backend.generate_commit(
            uids=uids,
            weights_u16=weights,
            version_key=1,
            last_epoch_block=90,
            pending_epoch_at=0,
            subnet_epoch_index=4,
            tempo=360,
            blocks_since_last_step=10,
            current_block=100,
            subnet_reveal_period_epochs=1,
            block_time=12.0,
            hotkey_public_key=b"h" * 32,
        )


def test_drand_backend_propagates_helper_error_without_output(tmp_path):
    library = _Library(error="timelock failure")
    backend = _backend(tmp_path, library)
    with pytest.raises(DrandCommitV2Error, match="timelock failure"):
        backend.generate_commit(
            uids=[0],
            weights_u16=[65535],
            version_key=1,
            last_epoch_block=90,
            pending_epoch_at=0,
            subnet_epoch_index=4,
            tempo=360,
            blocks_since_last_step=10,
            current_block=100,
            subnet_reveal_period_epochs=1,
            block_time=12.0,
            hotkey_public_key=b"h" * 32,
        )
