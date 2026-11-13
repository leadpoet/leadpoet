"""Apply only the reviewed investigator prompt and feedback correction on50de.

The complete preimage and postimage are checksum-bound. All other frozen
source bytes, grader policy and process configuration must stay unchanged.
"""
from __future__ import annotations

CORRECTION_ID = "investigator_prompt_feedback_v1"
FROZEN_IMAGE_DIGEST = "sha256:50deca7b8bf5129816abaeb59fd5bc1aabb554c529cc4151b8175f9edd0b8144"
ORIGINAL_SHA256 = "eaa205dfe6a1707199246008f7108c5883e1d298b6c7c682d41b16c55e677483"


def convert_investigator_feedback_seam(source: bytes) -> bytes:
    """Fail closed until the exact reviewed source delta is supplied."""
    raise ValueError("investigator feedback delta is not yet reviewed")
