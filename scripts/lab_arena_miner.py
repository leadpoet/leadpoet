#!/usr/bin/env python3
"""Use the SN71 miner CLI from a source checkout.

    python3 scripts/lab_arena_miner.py collection-terms
    python3 scripts/lab_arena_miner.py collection-register --help
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lab_arena.miner_cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
