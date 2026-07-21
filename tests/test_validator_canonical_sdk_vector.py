import pytest

from neurons.validator import _canonical_sdk_weight_vector


def test_canonical_sdk_vector_sorts_original_floats_to_match_authorization():
    assert _canonical_sdk_weight_vector(
        {
            "uids": [7, 0, 3],
            "weights": [0.2, 0.7, 0.1],
            "sparse_uids": [0, 3, 7],
            "sparse_weights_u16": [65535, 9362, 18724],
        }
    ) == ([0, 3, 7], [0.7, 0.1, 0.2])


def test_canonical_sdk_vector_rejects_any_authorization_difference():
    with pytest.raises(RuntimeError, match="differs from enclave authorization"):
        _canonical_sdk_weight_vector(
            {
                "uids": [1, 0],
                "weights": [0.2, 0.8],
                "sparse_uids": [1, 0],
                "sparse_weights_u16": [16384, 65535],
            }
        )
