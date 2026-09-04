"""Leadpoet Lab Arena V1.

The Arena is a separate, non-enclave service. This package is never staged
into an enclave image and is never imported by a measured package. Keep this
module free of eager imports so the runtime import closure of every Arena
entrypoint stays explicit and testable.
"""

LAB_ARENA_PACKAGE_VERSION = "lab_arena.v1"

__all__ = ["LAB_ARENA_PACKAGE_VERSION"]
