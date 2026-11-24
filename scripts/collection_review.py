"""Offline reviewer utility. Publish no private artifacts; print only counts and digests."""

import argparse
import json
import os
from pathlib import Path
import stat
import sys

from lab_arena import collection_judge as judge
from lab_arena import collection_judge_review as review


def read_file(path, *, maximum=128 * 1024 * 1024, private=True):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as f:
        info = os.fstat(f.fileno())
        if (
            not stat.S_ISREG(info.st_mode)
            or private
            and (info.st_uid != os.getuid() or info.st_mode & 0o077)
        ):
            raise ValueError("input must be an owned regular private file (mode 0600)")
        if info.st_size > maximum:
            raise ValueError("input too large")
        data = f.read(maximum + 1)
        if len(data) > maximum:
            raise ValueError("input too large")
        return data


def write_new(path, value):
    encoded = (
        json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False)
        + "\n"
    ).encode()
    fd = os.open(path, os.O_WRONLY | os.O_NOFOLLOW | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(encoded)
        f.flush()
        os.fsync(f.fileno())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sample = sub.add_parser("sample")
    sign = sub.add_parser("sign")
    verify = sub.add_parser("verify")
    approve = sub.add_parser(
        "qualify",
        help="Independently approve a measured pilot panel for correction review",
    )
    approve.add_argument("--reviewer-public-key", type=Path, required=True)
    approve.add_argument("--approver", required=True)
    approve.add_argument(
        "--allow-live",
        action="store_true",
        help="Also approve this exact measured panel for compensated collection",
    )
    approve.add_argument("--approver-private-key", type=Path, required=True)
    approve.add_argument("--attestation", type=Path, required=True)
    for command in [sample, sign, verify, approve]:
        command.add_argument("--bundle", type=Path, required=True)
    sample.add_argument("--jobs", type=Path, required=True)
    sample.add_argument("--claim-jobs", type=Path)
    sample.add_argument("--marks-template", type=Path, required=True)
    sign.add_argument("--reviewer", required=True)
    sign.add_argument("--marks", type=Path, required=True)
    sign.add_argument("--private-key", type=Path, required=True)
    for command in [sign, verify, approve]:
        command.add_argument("--receipt", type=Path, required=True)
    verify.add_argument("--public-key", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        bundle = json.loads(read_file(args.bundle))
        if args.action == "sample":
            _, jobs = review.prepare_review(bundle)
            if (
                args.jobs == args.marks_template
                or args.jobs.exists()
                or args.marks_template.exists()
            ):
                raise ValueError("sample output already exists")
            extra = {}
            if args.claim_jobs:
                if args.claim_jobs in (args.jobs, args.marks_template):
                    raise ValueError("duplicate output paths")
                aliases, disputes = review.claim_review_jobs(bundle)
                write_new(args.claim_jobs, disputes)
                extra = {
                    "claim_sample_sha256": judge.digest(disputes),
                    "claim_decisions": dict.fromkeys(aliases),
                }
            write_new(args.jobs, jobs)
            write_new(
                args.marks_template,
                {
                    **extra,
                    "sample_sha256": judge.digest(jobs),
                    "marks": {alias: dict.fromkeys(judge.PARTS) for alias in jobs},
                },
            )
            print(
                json.dumps(
                    {"blind_jobs": len(jobs), "sample_sha256": judge.digest(jobs)}
                )
            )
        elif args.action == "qualify":
            approved = review.approve_qualification(
                bundle,
                json.loads(read_file(args.receipt)),
                read_file(args.reviewer_public_key, maximum=8192, private=False),
                args.approver,
                read_file(args.approver_private_key, maximum=8192),
                allow_live=args.allow_live,
            )
            write_new(args.attestation, approved)
            print(json.dumps({"attestation_sha256": judge.digest(approved)}))
        elif args.action == "sign":
            signed = review.sign_review(
                bundle,
                json.loads(read_file(args.marks)),
                args.reviewer,
                read_file(args.private_key, maximum=8192),
            )
            write_new(args.receipt, signed)
            print(
                json.dumps(
                    {
                        "receipt_sha256": judge.digest(signed),
                        "holds": signed["report"]["holds"],
                    }
                )
            )
        else:
            report = review.verify_review(
                bundle,
                json.loads(read_file(args.receipt)),
                read_file(args.public_key, maximum=8192, private=False),
            )
            print(
                json.dumps(
                    {
                        "quality_ready": True,
                        "batch_id": report["batch_id"],
                        "blind_human_marks": report["blind_human_marks"],
                    }
                )
            )
        return 0
    except (OSError, KeyError, TypeError, ValueError, RecursionError) as exc:
        print(f"review failed ({type(exc).__name__})", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
