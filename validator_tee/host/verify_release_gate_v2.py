"""Create and enforce the six-build validator V2 release gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from validator_tee.host.release_v2 import (
    DETERMINISTIC_RELEASE_FIELDS,
    ValidatorReleaseV2Error,
    build_validator_build_evidence,
    build_validator_release_manifest,
    validate_validator_release,
    validate_validator_release_manifest,
)


def _load(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidatorReleaseV2Error("validator release input is unavailable") from exc
    if not isinstance(value, Mapping):
        raise ValidatorReleaseV2Error("validator release input must be an object")
    return dict(value)


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(value), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--emit-evidence", action="store_true")
    mode.add_argument("--build-manifest", action="store_true")
    mode.add_argument("--verify-manifest", type=Path)
    parser.add_argument("--local-release", type=Path)
    parser.add_argument("--evidence", action="append", type=Path, default=[])
    parser.add_argument("--builder-domain")
    parser.add_argument("--builder-id")
    parser.add_argument("--build-ordinal", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    if args.emit_evidence:
        if not all(
            (
                args.local_release,
                args.builder_domain,
                args.builder_id,
                args.build_ordinal,
                args.output,
            )
        ):
            raise ValidatorReleaseV2Error(
                "evidence output requires local release and builder identity"
            )
        value = build_validator_build_evidence(
            _load(args.local_release),
            builder_domain=args.builder_domain,
            builder_id=args.builder_id,
            build_ordinal=args.build_ordinal,
        )
        _write(args.output, value)
        print("validator_v2_build_evidence=%s" % args.output)
        return 0

    if args.build_manifest:
        if len(args.evidence) != 6 or args.output is None:
            raise ValidatorReleaseV2Error(
                "manifest build requires six evidence files and --output"
            )
        value = build_validator_release_manifest(
            [_load(path) for path in args.evidence]
        )
        _write(args.output, value)
        print("validator_v2_release_manifest_hash=%s" % value["release_manifest_hash"])
        return 0

    manifest = validate_validator_release_manifest(_load(args.verify_manifest))
    if args.local_release is None:
        raise ValidatorReleaseV2Error(
            "manifest verification requires --local-release"
        )
    local = validate_validator_release(_load(args.local_release))
    for field in DETERMINISTIC_RELEASE_FIELDS:
        if local[field] != manifest["release"][field]:
            raise ValidatorReleaseV2Error(
                "local validator build differs from approved six-build release at %s"
                % field
            )
    print("validator_v2_release_gate=verified")
    print("validator_v2_release_manifest_hash=%s" % manifest["release_manifest_hash"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
