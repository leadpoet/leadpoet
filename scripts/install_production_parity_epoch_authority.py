#!/usr/bin/env python3
"""Install one immutable testnet epoch authority on an ephemeral parity host."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from leadpoet_canonical.production_parity_epoch_authority import (  # noqa: E402
    ProductionParityEpochAuthorityError,
    install,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--region", required=True)
    args = parser.parse_args(argv)
    try:
        value = json.loads(args.spec.read_text(encoding="utf-8"))
        result = install(
            value,
            run_id=str(args.run_id),
            region=str(args.region),
        )
    except (OSError, ValueError, ProductionParityEpochAuthorityError):
        print(
            "ERROR: production parity epoch authority installation failed",
            file=sys.stderr,
        )
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
