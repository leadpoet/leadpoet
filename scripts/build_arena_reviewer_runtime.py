#!/usr/bin/env python3
"""Compose a runtime repair on an immutable Arena grader, without rebuilding it.

The original image layers, dependencies and process configuration stay intact.
Only the dedicated runtime package and reviewed operational-interface
extractions can be added by default. An explicit option applies only the
approved terminal-negative, exact-source-access, pending-final-correction and
shared failed attribute-source review repairs to frozen source.
Neither path imports today's grading code.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import re
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
GRADER_FILE = "qualification/scoring/company_evidence_investigator.py"
FINALITY_FILE = "qualification/scoring/lead_scorer.py"
ENTRYPOINT_FILES = ("lab_arena/scorer_entrypoint.py", "scorer_entrypoint.py")
RUNTIME_DIRECTORY = "arena_reviewer_runtime"
MAX_OVERLAY_BYTES = 4 * 1024 * 1024


class CompositionError(ValueError):
    pass


def checked_output(args: list[str], **kwargs: Any) -> str:
    return subprocess.check_output(args, text=True, **kwargs).strip()


def image_inspect(reference: str) -> dict[str, Any]:
    return json.loads(checked_output(["docker", "image", "inspect", reference]))[0]


def verify_image_configuration(base: dict, candidate: dict) -> list[str]:
    """A runtime release cannot change the base, interpreter or import path."""
    for name in ("Architecture", "Os", "Config"):
        if candidate.get(name) != base.get(name):
            raise CompositionError("composed image changed " + name)
    original = base["RootFS"]["Layers"]
    composed = candidate["RootFS"]["Layers"]
    if composed[:len(original)] != original or len(composed) != len(original) + 1:
        raise CompositionError("composed image must append exactly one runtime layer")
    return composed[len(original):]


def verify_overlay(data: bytes, expected_files: dict[str, bytes]) -> None:
    """Verify actual added bytes, not a label claiming grading equivalence."""
    if len(data) > MAX_OVERLAY_BYTES:
        raise CompositionError("runtime layer exceeds its bounded size")
    expected = {"model/" + path: value for path, value in expected_files.items()}
    directories = {str(parent) for path in expected for parent in PurePosixPath(path).parents}
    found: set[str] = set()
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as archive:
        for member in archive:
            if set(member.pax_headers) - {"mtime", "atime", "ctime"} or member.sparse:
                raise CompositionError("runtime layer contains filesystem metadata overrides")
            name = member.name.rstrip("/")
            while name.startswith("./"):
                name = name[2:]
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or ".wh." in path.name:
                raise CompositionError("unsafe runtime layer path")
            if member.type == tarfile.DIRTYPE and name in directories:
                if member.mode & 0o7000 or member.mode & 0o002:
                    raise CompositionError("unsafe runtime layer directory mode")
                continue
            if member.type not in (tarfile.REGTYPE, tarfile.AREGTYPE) or name not in expected or name in found:
                raise CompositionError("runtime layer changes an unapproved file")
            if member.mode & 0o7000 or member.mode & 0o002:
                raise CompositionError("unsafe runtime layer file mode")
            if member.size != len(expected[name]):
                raise CompositionError("runtime file size mismatch")
            stream = archive.extractfile(member)
            if stream is None or stream.read() != expected[name]:
                raise CompositionError("runtime file differs from reviewed source")
            found.add(name)
    if found != set(expected):
        raise CompositionError("runtime layer is missing expected files")


def verify_saved_overlay(reference: str, digest: str, expected: dict[str, bytes]) -> None:
    """Stream Docker's archive; do not unpack the base or trust archive names."""
    process = subprocess.Popen(["docker", "image", "save", reference], stdout=subprocess.PIPE)
    matches = 0
    try:
        assert process.stdout is not None
        with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
            for member in archive:
                if not member.isfile() or member.size > MAX_OVERLAY_BYTES:
                    continue
                stream = archive.extractfile(member)
                if stream is None:
                    continue
                data = stream.read()
                # Docker's OCI store exports gzip blobs; its classic store
                # exports uncompressed layer tar files. RootFS pins the hash
                # of the uncompressed bytes in either case.
                if data.startswith(b"\x1f\x8b"):
                    with gzip.GzipFile(fileobj=io.BytesIO(data)) as compressed:
                        data = compressed.read(MAX_OVERLAY_BYTES + 1)
                    if len(data) > MAX_OVERLAY_BYTES:
                        continue
                if "sha256:" + hashlib.sha256(data).hexdigest() == digest:
                    verify_overlay(data, expected)
                    matches += 1
        if process.wait() != 0 or matches != 1:
            raise CompositionError("could not verify the exact added runtime layer")
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait()


def extract_frozen_sources(
    reference: str, destination: Path, *, complete_negative_finality: bool = False,
    source_access_denied: bool = False,
    failed_attribute_source_recovery: bool = False,
) -> dict[str, bytes]:
    container = checked_output(["docker", "create", reference])
    try:
        files = {}
        names = (GRADER_FILE, *ENTRYPOINT_FILES)
        if complete_negative_finality or source_access_denied or failed_attribute_source_recovery:
            names += (FINALITY_FILE,)
        for name in names:
            path = destination / name
            path.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["docker", "cp", container + ":/model/" + name,
                            str(path)], check=True, stdout=subprocess.DEVNULL)
            files[name] = path.read_bytes()
        return files
    finally:
        subprocess.run(["docker", "rm", container], check=True, stdout=subprocess.DEVNULL)


def expected_overlay(
    original: dict[str, bytes], *, complete_negative_finality: bool = False,
    source_access_denied: bool = False,
    pending_final_correction: bool = False,
    failed_attribute_source_recovery: bool = False,
) -> dict[str, bytes]:
    from scripts.arena_reviewer_admission_seam import convert_admission_seam
    from scripts.arena_reviewer_checkpoint_seam import convert_checkpoint_seam

    expected = {GRADER_FILE, *ENTRYPOINT_FILES}
    if complete_negative_finality or source_access_denied or failed_attribute_source_recovery:
        expected.add(FINALITY_FILE)
    if set(original) != expected:
        raise CompositionError("frozen operational sources are incomplete")
    files = {GRADER_FILE: convert_admission_seam(original[GRADER_FILE])}
    files.update({name: convert_checkpoint_seam(original[name]) for name in ENTRYPOINT_FILES})
    if complete_negative_finality or source_access_denied or failed_attribute_source_recovery:
        files[FINALITY_FILE] = original[FINALITY_FILE]
    if complete_negative_finality:
        from scripts.arena_reviewer_finality_seam import convert_finality_seam
        files[FINALITY_FILE] = convert_finality_seam(files[FINALITY_FILE])
    if source_access_denied:
        from scripts.arena_reviewer_source_access_seam import (
            convert_source_access_seam, convert_source_access_finality_seam,
        )
        files[GRADER_FILE] = convert_source_access_seam(files[GRADER_FILE])
        files[FINALITY_FILE] = convert_source_access_finality_seam(files[FINALITY_FILE])
    if pending_final_correction:
        from scripts.arena_reviewer_budget_seam import convert_budget_seam
        files[GRADER_FILE] = convert_budget_seam(files[GRADER_FILE])
    if failed_attribute_source_recovery:
        from scripts.arena_reviewer_attribute_source_seam import convert_attribute_source_seam
        files[FINALITY_FILE] = convert_attribute_source_seam(files[FINALITY_FILE])
    for path in sorted((ROOT / RUNTIME_DIRECTORY).glob("*.py")):
        if path.is_symlink():
            raise CompositionError("runtime source must not be a symlink")
        files[RUNTIME_DIRECTORY + "/" + path.name] = path.read_bytes()
    if len(files) < 3:
        raise CompositionError("runtime source is incomplete")
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grading-image", required=True)
    parser.add_argument("--verify-image", help="verify an existing composition instead of building")
    parser.add_argument("--push", action="store_true")
    parser.add_argument(
        "--complete-negative-finality", action="store_true",
        help="apply only the approved PR483 terminal-negative correction; use the same option for --verify-image",
    )
    parser.add_argument(
        "--source-access-denied", action="store_true",
        help="apply the exact-source access-denial correction; use the same option for --verify-image",
    )
    parser.add_argument(
        "--pending-final-correction", action="store_true",
        help="allow an already pending submit-only final correction; use the same option for --verify-image",
    )
    parser.add_argument(
        "--failed-attribute-source-recovery", action="store_true",
        help="include a failed attribute source in an already planned review; use the same option for --verify-image",
    )
    args = parser.parse_args()
    match = re.fullmatch(r"([a-zA-Z0-9./_:-]+)@(sha256:[0-9a-f]{64})", args.grading_image)
    if not match:
        raise CompositionError("grading image must be pinned by digest")
    if args.push and args.verify_image:
        raise CompositionError("verify does not publish")
    revision = checked_output(["git", "rev-parse", "HEAD"], cwd=ROOT)
    if args.push and checked_output(["git", "status", "--porcelain"], cwd=ROOT):
        raise CompositionError("publishing requires a clean committed checkout")
    base = image_inspect(args.grading_image)
    if base.get("Architecture") != "amd64" or base.get("Os") != "linux":
        raise CompositionError("grading image must be linux/amd64")
    with tempfile.TemporaryDirectory(prefix="arena-reviewer-") as directory:
        root = Path(directory)
        original = extract_frozen_sources(
            args.grading_image, root / "original",
            complete_negative_finality=args.complete_negative_finality,
            source_access_denied=args.source_access_denied,
            failed_attribute_source_recovery=args.failed_attribute_source_recovery,
        )
        files = expected_overlay(
            original, complete_negative_finality=args.complete_negative_finality,
            source_access_denied=args.source_access_denied,
            pending_final_correction=args.pending_final_correction,
            failed_attribute_source_recovery=args.failed_attribute_source_recovery,
        )
        tag = args.verify_image or (
            match[1] + ":reviewer-" + match[2][7:19] + "-" + revision[:12]
            + ("-negative-finality" if args.complete_negative_finality else "")
            + ("-source-access" if args.source_access_denied else "")
            + ("-pending-final-correction" if args.pending_final_correction else "")
            + ("-attribute-source-recovery" if args.failed_attribute_source_recovery else "")
        )
        if not args.verify_image:
            for name, data in files.items():
                path = root / "overlay" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
                path.chmod(0o644)
            (root / "Dockerfile").write_text(
                "FROM " + args.grading_image + "\nCOPY --chown=65534:65534 overlay/ /model/\n"
            )
            subprocess.run(["docker", "build", "--platform", "linux/amd64",
                            "--provenance=false", "--sbom=false", "-t", tag, str(root)], check=True)
        candidate = image_inspect(tag)
        added = verify_image_configuration(base, candidate)
        verify_saved_overlay(tag, added[0], files)
        reference = tag
        if args.push:
            subprocess.run(["docker", "push", tag], check=True)
            digests = image_inspect(tag).get("RepoDigests") or []
            matching = [value for value in digests if value.startswith(match[1] + "@sha256:")]
            if len(matching) != 1:
                raise CompositionError("published runtime digest is ambiguous")
            reference = matching[0]
        from scripts.arena_reviewer_finality_seam import CORRECTION_ID
        corrections = [CORRECTION_ID] if args.complete_negative_finality else []
        if args.source_access_denied:
            from scripts.arena_reviewer_source_access_seam import CORRECTION_ID as SOURCE_ACCESS_CORRECTION_ID
            corrections.append(SOURCE_ACCESS_CORRECTION_ID)
        if args.pending_final_correction:
            from scripts.arena_reviewer_budget_seam import CORRECTION_ID as BUDGET_CORRECTION_ID
            corrections.append(BUDGET_CORRECTION_ID)
        if args.failed_attribute_source_recovery:
            from scripts.arena_reviewer_attribute_source_seam import CORRECTION_ID as ATTRIBUTE_SOURCE_CORRECTION_ID
            corrections.append(ATTRIBUTE_SOURCE_CORRECTION_ID)
        print(json.dumps({
            "grading_image": args.grading_image,
            "runtime_image": reference,
            "runtime_source_commit": revision,
            "base_layers_unchanged": True,
            "process_configuration_unchanged": True,
            "grading_corrections": corrections,
            "original_files": {name: hashlib.sha256(value).hexdigest() for name, value in original.items()},
            "overlay_files": {name: hashlib.sha256(value).hexdigest() for name, value in files.items()},
        }, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (CompositionError, ValueError) as exc:
        raise SystemExit(str(exc)) from None
