"""Preserve only the reviewed worker socket return margin in frozen source."""
from __future__ import annotations

import ast
import hashlib

from scripts.arena_reviewer_source_access_seam import SourceAccessSeamError, _convert


CORRECTION_ID = "worker_socket_return_margin_v1"
SHIM_FILE = "lab_arena/shim.py"
OPERATIONS_FILE = "lab_arena/operations.py"
OPERATIONS_SHA256 = "69ec35c484ad7cd0017b5dae360bc2e58a55c2b38c490eec1bba6cf4dc1e0c7e"
SHIM_SHA256 = frozenset({
    "00399bb3a8994c45f67b83abcfde781c61fdb28fe58153816be93b17c835c486",
    "9d95c64ca570038c8bf2acfc42807552a8e92457595e3730360f163d369d13db",
})


class ShimResponseWindowSeamError(ValueError):
    """The frozen transport does not match the reviewed timeout boundary."""


OLD = '''        connection.settimeout(
            operations.BUDGET_ADMISSION_MAX_SECONDS
            + bounded_timeout_ms / 1000.0
            # The shim does not have the run context needed to know whether a
            # legacy scoring request will route to Deepline in the broker.
            + operations.PROVIDER_BILLING_RECONCILIATION_SECONDS
            + SOCKET_GRACE_SECONDS
        )
'''
NEW = OLD.replace(
    "            + SOCKET_GRACE_SECONDS\n",
    "            # The upstream API has its own transport allowance. Keep the\n"
    "            # separate socket margin for dispatch and carrying its reply back.\n"
    "            + operations.PROVIDER_API_TIMEOUT_GRACE_SECONDS\n"
    "            + SOCKET_GRACE_SECONDS\n",
)


def convert_shim_response_window(shim_source: bytes, operations_source: bytes) -> bytes:
    try:
        # Support the independently inspected frozen source generation only.
        # A literal assignment alone cannot exclude later imports or rebinding.
        if (hashlib.sha256(operations_source).hexdigest() != OPERATIONS_SHA256
                or hashlib.sha256(shim_source).hexdigest() not in SHIM_SHA256):
            raise ValueError("frozen transport source generation is not reviewed")
        tree = ast.parse(operations_source.decode("utf-8"))
        name = "PROVIDER_API_TIMEOUT_GRACE_SECONDS"
        bindings = [node for node in ast.walk(tree)
                    if isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, ast.Store)]
        constants = [node.value for node in tree.body if isinstance(node, ast.Assign)
                     and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                     and node.targets[0].id == name]
        if len(bindings) != 1 or len(constants) != 1:
            raise ValueError("API grace constant is missing or ambiguous")
        value = ast.literal_eval(constants[0])
        if type(value) is not float or value != 15.0:
            raise ValueError("API grace differs from the reviewed fifteen seconds")
        return _convert(shim_source, (("_dispatch_with_response_url", OLD, NEW),))
    except (SourceAccessSeamError, UnicodeDecodeError, SyntaxError, ValueError, TypeError) as exc:
        raise ShimResponseWindowSeamError("unrecognized frozen shim response window") from exc
