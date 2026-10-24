#!/usr/bin/env python3
"""Compose a runtime repair on an immutable Arena grader, without rebuilding it.

The original image layers, dependencies and process configuration stay intact.
Only the dedicated runtime package and reviewed operational-interface
extractions can be added by default. An explicit option applies only the
approved terminal-negative, exact-source-access, pending-final-correction and
shared failed attribute-source review, identity schema-repair reachability and
rejected-final-packet completion and structured numeric-homepage alias
repairs to frozen source. A separate exact-source option repairs explicit
first-party legal/brand identity and same-origin Organization URL resolution.
No option imports today's complete grading implementation.
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
VERIFICATION_FILE = "qualification/scoring/company_verification.py"
FINALITY_FILE = "qualification/scoring/lead_scorer.py"
ARCHIVE_FILE = "qualification/scoring/intent_verification_three_stage.py"
BOUND_IDENTITY_FILES = (
    VERIFICATION_FILE,
    "qualification/scoring/company_fit_decision.py",
    "qualification/scoring/linkedin_company_size.py",
)
ENTRYPOINT_FILES = ("lab_arena/scorer_entrypoint.py", "scorer_entrypoint.py")
RUNTIME_DIRECTORY = "arena_reviewer_runtime"
MAX_OVERLAY_BYTES = 4 * 1024 * 1024


class CompositionError(ValueError):
    pass


def bounded_build_tag(reference: str) -> str:
    """Keep Docker's 128-character tag limit without dropping repair identity."""
    repository, _, tag = reference.rpartition(":")
    if len(tag) > 128:
        tag = tag[:111] + "-" + hashlib.sha256(tag.encode("utf-8")).hexdigest()[:16]
    return repository + ":" + tag


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
    authenticated_page_refusal: bool = False,
    completed_investigation_finality: bool = False,
    identity_schema_repair: bool = False,
    structured_homepage_numeric_alias: bool = False,
    bound_homepage_legal_alias: bool = False,
    archive_snapshot_https: bool = False,
) -> dict[str, bytes]:
    container = checked_output(["docker", "create", reference])
    try:
        files = {}
        names = (GRADER_FILE, *ENTRYPOINT_FILES)
        if (complete_negative_finality or source_access_denied
                or failed_attribute_source_recovery or completed_investigation_finality
                or identity_schema_repair or structured_homepage_numeric_alias
                or bound_homepage_legal_alias):
            names += (FINALITY_FILE,)
        if authenticated_page_refusal:
            names += (VERIFICATION_FILE,)
        if bound_homepage_legal_alias:
            names += BOUND_IDENTITY_FILES
        if archive_snapshot_https:
            names += (ARCHIVE_FILE,)
        for name in dict.fromkeys(names):
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
    authenticated_page_refusal: bool = False,
    completed_investigation_finality: bool = False,
    identity_schema_repair: bool = False,
    structured_homepage_numeric_alias: bool = False,
    bound_homepage_legal_alias: bool = False,
    archive_snapshot_https: bool = False,
    numeric_profile_legal_self_name: bool = False,
    single_tool_protocol_repair: bool = False,
    rejected_final_packet: bool = False,
) -> dict[str, bytes]:
    from scripts.arena_reviewer_admission_seam import convert_admission_seam
    from scripts.arena_reviewer_checkpoint_seam import convert_checkpoint_seam

    if numeric_profile_legal_self_name and not bound_homepage_legal_alias:
        raise CompositionError("numeric self-name requires the reviewed legal alias repair")
    expected = {GRADER_FILE, *ENTRYPOINT_FILES}
    if (complete_negative_finality or source_access_denied
                or failed_attribute_source_recovery or completed_investigation_finality
                or identity_schema_repair or structured_homepage_numeric_alias
                or bound_homepage_legal_alias):
        expected.add(FINALITY_FILE)
    if authenticated_page_refusal:
        expected.add(VERIFICATION_FILE)
    if bound_homepage_legal_alias:
        expected.update(BOUND_IDENTITY_FILES)
    if archive_snapshot_https:
        expected.add(ARCHIVE_FILE)
    if set(original) != expected:
        raise CompositionError("frozen operational sources are incomplete")
    files = {GRADER_FILE: convert_admission_seam(original[GRADER_FILE])}
    files.update({name: convert_checkpoint_seam(original[name]) for name in ENTRYPOINT_FILES})
    if (complete_negative_finality or source_access_denied
                or failed_attribute_source_recovery or completed_investigation_finality
                or identity_schema_repair or structured_homepage_numeric_alias
                or bound_homepage_legal_alias):
        files[FINALITY_FILE] = original[FINALITY_FILE]
    if bound_homepage_legal_alias:
        files.update({name: original[name] for name in BOUND_IDENTITY_FILES})
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
    if authenticated_page_refusal:
        from scripts.arena_reviewer_typed_refusal_seam import (
            convert_investigator_seam, convert_verification_seam,
        )
        files[GRADER_FILE] = convert_investigator_seam(files[GRADER_FILE])
        files[VERIFICATION_FILE] = convert_verification_seam(original[VERIFICATION_FILE])
    if completed_investigation_finality:
        from scripts.arena_reviewer_investigation_finality_seam import (
            convert_investigation_finality_seam,
        )
        files[FINALITY_FILE] = convert_investigation_finality_seam(files[FINALITY_FILE])
    if identity_schema_repair:
        from scripts.arena_reviewer_identity_schema_seam import convert_identity_schema_seam
        files[FINALITY_FILE] = convert_identity_schema_seam(files[FINALITY_FILE])
    if structured_homepage_numeric_alias:
        from scripts.arena_reviewer_structured_homepage_alias_seam import (
            convert_structured_homepage_alias_seam,
        )
        files[FINALITY_FILE] = convert_structured_homepage_alias_seam(files[FINALITY_FILE])
    if single_tool_protocol_repair:
        from scripts.arena_reviewer_protocol_repair_seam import convert_protocol_repair_seam
        files[GRADER_FILE] = convert_protocol_repair_seam(files[GRADER_FILE])
    if rejected_final_packet:
        from scripts.arena_reviewer_rejected_final_packet_seam import convert_rejected_final_packet_seam
        files[GRADER_FILE] = convert_rejected_final_packet_seam(files[GRADER_FILE])
    if bound_homepage_legal_alias:
        from scripts.arena_reviewer_bound_homepage_alias_seam import (
            convert_bound_homepage_legal_alias_files,
        )
        names = (GRADER_FILE, FINALITY_FILE, *BOUND_IDENTITY_FILES)
        files.update(convert_bound_homepage_legal_alias_files(
            {name: files[name] for name in names},
        ))
    if numeric_profile_legal_self_name:
        from scripts.arena_reviewer_structured_homepage_alias_seam import (
            SELF_NAME_CHANGES, convert_numeric_profile_self_name_files,
        )
        files.update(convert_numeric_profile_self_name_files(
            {name: files[name] for name in SELF_NAME_CHANGES},
        ))
    if archive_snapshot_https:
        from scripts.arena_reviewer_archive_https_seam import convert_archive_https_seam
        files[ARCHIVE_FILE] = convert_archive_https_seam(original[ARCHIVE_FILE])
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
    parser.add_argument(
        "--completed-investigation-finality", action="store_true",
        help="reuse exact completed-investigation finality before semantic exits; use the same option for --verify-image",
    )
    parser.add_argument(
        "--identity-schema-repair", action="store_true",
        help="reach existing identity schema repair after completed semantic review; use the same option for --verify-image",
    )
    parser.add_argument(
        "--authenticated-page-refusal", action="store_true",
        help="classify only a host-bound exact page refusal; use the same option for --verify-image",
    )
    parser.add_argument(
        "--single-tool-protocol-repair", action="store_true",
        help="repair one valid research batch within existing limits; use the same option for --verify-image",
    )
    parser.add_argument(
        "--rejected-final-packet", action="store_true",
        help="keep rejected final submissions incomplete; use the same option for --verify-image",
    )
    parser.add_argument(
        "--structured-homepage-numeric-alias", action="store_true",
        help="carry independently proved numeric-homepage identity into quote binding; use the same option for --verify-image",
    )
    parser.add_argument(
        "--bound-homepage-legal-alias", action="store_true",
        help="carry exact first-party legal/brand identity through frozen source; use the same option for --verify-image",
    )
    parser.add_argument(
        "--numeric-profile-legal-self-name", action="store_true",
        help="retain only an exact numeric main profile's leading legal self-name; requires the existing legal alias repair",
    )
    parser.add_argument(
        "--archive-snapshot-https", action="store_true",
        help="use HTTPS for an exact source-bound Wayback snapshot locator; use the same option for --verify-image",
    )
    args = parser.parse_args()
    if args.numeric_profile_legal_self_name and not args.bound_homepage_legal_alias:
        raise CompositionError("numeric self-name requires the reviewed legal alias repair")
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
            **({"completed_investigation_finality": True}
               if args.completed_investigation_finality else {}),
            **({"identity_schema_repair": True}
               if args.identity_schema_repair else {}),
            **({"structured_homepage_numeric_alias": True}
               if args.structured_homepage_numeric_alias else {}),
            **({"bound_homepage_legal_alias": True}
               if args.bound_homepage_legal_alias else {}),
            **({"archive_snapshot_https": True}
               if args.archive_snapshot_https else {}),
            **({"authenticated_page_refusal": True}
               if args.authenticated_page_refusal else {}),
        )
        files = expected_overlay(
            original, complete_negative_finality=args.complete_negative_finality,
            source_access_denied=args.source_access_denied,
            pending_final_correction=args.pending_final_correction,
            failed_attribute_source_recovery=args.failed_attribute_source_recovery,
            authenticated_page_refusal=args.authenticated_page_refusal,
            completed_investigation_finality=args.completed_investigation_finality,
            identity_schema_repair=args.identity_schema_repair,
            structured_homepage_numeric_alias=args.structured_homepage_numeric_alias,
            bound_homepage_legal_alias=args.bound_homepage_legal_alias,
            numeric_profile_legal_self_name=args.numeric_profile_legal_self_name,
            single_tool_protocol_repair=args.single_tool_protocol_repair,
            rejected_final_packet=args.rejected_final_packet,
            archive_snapshot_https=args.archive_snapshot_https,
        )
        tag = args.verify_image or (
            match[1] + ":reviewer-" + match[2][7:19] + "-" + revision[:12]
            + ("-negative-finality" if args.complete_negative_finality else "")
            + ("-source-access" if args.source_access_denied else "")
            + ("-pending-final-correction" if args.pending_final_correction else "")
            + ("-attribute-source-recovery" if args.failed_attribute_source_recovery else "")
            + ("-refusal" if args.authenticated_page_refusal else "")
            + ("-investigation-finality" if args.completed_investigation_finality else "")
            + ("-identity-schema-repair" if args.identity_schema_repair else "")
            + ("-structured-homepage-alias" if args.structured_homepage_numeric_alias else "")
            + ("-bound-homepage-legal-alias" if args.bound_homepage_legal_alias else "")
            + ("-numeric-legal-self-name" if args.numeric_profile_legal_self_name else "")
            + ("-archive-https" if args.archive_snapshot_https else "")
            + ("-single-tool-repair" if args.single_tool_protocol_repair else "")
            + ("-rejected-final-packet" if args.rejected_final_packet else "")
        )
        if not args.verify_image:
            tag = bounded_build_tag(tag)
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
        if args.authenticated_page_refusal:
            from scripts.arena_reviewer_typed_refusal_seam import CORRECTION_ID as PAGE_REFUSAL_CORRECTION_ID
            corrections.append(PAGE_REFUSAL_CORRECTION_ID)
        if args.completed_investigation_finality:
            from scripts.arena_reviewer_investigation_finality_seam import CORRECTION_ID as INVESTIGATION_FINALITY_CORRECTION_ID
            corrections.append(INVESTIGATION_FINALITY_CORRECTION_ID)
        if args.identity_schema_repair:
            from scripts.arena_reviewer_identity_schema_seam import CORRECTION_ID as IDENTITY_SCHEMA_CORRECTION_ID
            corrections.append(IDENTITY_SCHEMA_CORRECTION_ID)
        if args.structured_homepage_numeric_alias:
            from scripts.arena_reviewer_structured_homepage_alias_seam import CORRECTION_ID as STRUCTURED_HOMEPAGE_ALIAS_CORRECTION_ID
            corrections.append(STRUCTURED_HOMEPAGE_ALIAS_CORRECTION_ID)
        if args.single_tool_protocol_repair:
            from scripts.arena_reviewer_protocol_repair_seam import CORRECTION_ID as PROTOCOL_CORRECTION_ID
            corrections.append(PROTOCOL_CORRECTION_ID)
        if args.rejected_final_packet:
            from scripts.arena_reviewer_rejected_final_packet_seam import CORRECTION_ID as REJECTED_FINAL_PACKET_CORRECTION_ID
            corrections.append(REJECTED_FINAL_PACKET_CORRECTION_ID)
        if args.bound_homepage_legal_alias:
            from scripts.arena_reviewer_bound_homepage_alias_seam import CORRECTION_ID as BOUND_HOMEPAGE_ALIAS_CORRECTION_ID
            corrections.append(BOUND_HOMEPAGE_ALIAS_CORRECTION_ID)
        if args.numeric_profile_legal_self_name:
            from scripts.arena_reviewer_structured_homepage_alias_seam import SELF_NAME_CORRECTION_ID
            corrections.append(SELF_NAME_CORRECTION_ID)
        if args.archive_snapshot_https:
            from scripts.arena_reviewer_archive_https_seam import CORRECTION_ID as ARCHIVE_HTTPS_CORRECTION_ID
            corrections.append(ARCHIVE_HTTPS_CORRECTION_ID)
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
