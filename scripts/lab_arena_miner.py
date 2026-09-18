#!/usr/bin/env python3
"""Submit one local Arena model source directory and its run credentials.

    OPENROUTER_API_KEY=... OPENROUTER_MANAGEMENT_KEY=... DEEPLINE_API_KEY=... \
    python3 scripts/lab_arena_miner.py submit-model --source ./my-agent \
        --wallet-name W --hotkey-name H

The helper makes one bounded source archive, uploads it to the Arena's private
target, and finalizes it with the miner hotkey. Credentials are read only from
masked prompts or environment variables, and are never command arguments or
source archive content. ``--hotkey-uri`` is for development only.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lab_arena.miner_cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
