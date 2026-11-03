"""Read checksum-bound historical reviewer sources without Git-history access."""
import gzip
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).parent / "fixtures" / "reviewer_seams"


def frozen_source(cohort: str, path: str) -> bytes:
    manifest = json.loads((ROOT / "sources.json").read_text())
    entry = manifest[cohort]["files"][path]
    source = gzip.decompress((ROOT / entry["fixture"]).read_bytes())
    assert hashlib.sha256(source).hexdigest() == entry["sha256"], path
    return source
