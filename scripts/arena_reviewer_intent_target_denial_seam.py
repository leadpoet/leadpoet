"""Add one native target-denial alias to two exact frozen source modules."""
from __future__ import annotations

import hashlib


CORRECTION_ID = "intent_exact_target_native_crawl_denial_v1"
VERIFIER_FILE = "qualification/scoring/intent_verification_three_stage.py"
COMPETITION_FILE = "qualification/scoring/competition.py"
FILES = (VERIFIER_FILE, COMPETITION_FILE)
ORIGINAL_SHA256 = {
    VERIFIER_FILE: "c11d19434f65e8de8e57b102e5a81121ab4926494fe7f7ceff730d5c4f9d79c6",
    COMPETITION_FILE: "3ce73fd0906274dfcf08f3e014f01e27fba129a3a24571e6a88381f003d22a4b",
}
REPAIRED_SHA256 = {
    VERIFIER_FILE: "723b58edf4f04d97959b3a1c458df66cba75f3cd24a8eff1984a2884ff862744",
    COMPETITION_FILE: "d855d8ef3e3c42bb1b2c0e4aa04a1779b54351b8931e4cb64006c68d91be66f8",
}
ANCHOR = b'    ("SOURCE_NOT_AVAILABLE", 403),\n'
INSERTION = b'    ("CRAWL_HTTP_403", 403),\n'


class IntentTargetDenialSeamError(ValueError):
    """The frozen pair is not one complete reviewed source generation."""


def convert_intent_target_denial_files(sources: dict[str, bytes]) -> dict[str, bytes]:
    if set(sources) != set(FILES):
        raise IntentTargetDenialSeamError("intent target denial requires exactly two frozen modules")
    hashes = {name: hashlib.sha256(source).hexdigest() for name, source in sources.items()}
    if hashes == REPAIRED_SHA256:
        return dict(sources)
    if hashes != ORIGINAL_SHA256:
        raise IntentTargetDenialSeamError("unrecognized or partial frozen intent target denial source")
    repaired = {}
    for name, source in sources.items():
        if source.count(ANCHOR) != 1 or INSERTION in source:
            raise IntentTargetDenialSeamError("intent target denial anchor differs")
        repaired[name] = source.replace(ANCHOR, ANCHOR + INSERTION, 1)
    if {name: hashlib.sha256(source).hexdigest() for name, source in repaired.items()} != REPAIRED_SHA256:
        raise IntentTargetDenialSeamError("intent target denial repair differs from reviewed source")
    return repaired
