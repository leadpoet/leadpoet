"""Generate the fixed Research Lab routing dependency release shim.

The generated module never accepts a request-selected import path.  It calls
the one allowlisted attested authority provider and pins the protected
workflow manifest identity into the release package.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from gateway.research_lab.routing_release_builder import (
    render_generated_release_module,
)
from gateway.tee.protected_workflows import DEFAULT_MANIFEST, load_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("gateway/research_lab/routing_release_dependencies.py"),
    )
    parser.add_argument("--protected-manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args(argv)
    manifest = load_manifest(args.protected_manifest)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        render_generated_release_module(
            protected_workflow_manifest_hash=str(manifest["manifest_hash"])
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
