"""Import existing private Poet claim-reader votes. No provider I/O or charge is possible."""

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
from lab_arena import collection_claims as cc, collection_judge as judge
from lab_arena.collection_workflow import workflow_sources
from scripts.collection_review import read_file, write_new


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["import"])
    for name in ("inventory", "poet-repo", "champion-repo", "private-key", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        docs = json.loads(read_file(args.inventory))
        judge._fields(docs, ("schema", "sources"))
        if (
            docs["schema"] != "collection.claim-files.v1"
            or not isinstance(docs["sources"], list)
            or not 0 < len(docs["sources"]) <= 10000
        ):
            raise ValueError("invalid source inventory")
        entries = []
        with workflow_sources(args.champion_repo, args.poet_repo) as (_, builders):
            reader = importlib.import_module("claims")
            if (
                Path(reader.__file__).resolve().parent
                != Path(builders.__file__).resolve().parent
            ):
                raise ValueError("unpinned claim reader import")
            for source in docs["sources"]:
                judge._fields(source, ("source_run_id", "run_folder", "claims_sha256"))
                folder = Path(source["run_folder"])
                blob = read_file(folder / "claims.json", private=False)
                if hashlib.sha256(blob).hexdigest() != source["claims_sha256"]:
                    raise ValueError("claim file changed")
                original = [
                    json.loads(read_file(folder / (name + ".json"), private=False))
                    for name in ("trace", "out", "meta")
                ]
                entries.extend(
                    cc.ingest_run(
                        builders,
                        reader,
                        source["source_run_id"],
                        *original,
                        json.loads(blob),
                        source["claims_sha256"],
                    )
                )
        signed = cc.sign_pack(entries, read_file(args.private_key, maximum=8192))
        write_new(args.out, signed)
        print(
            json.dumps({"imported": len(entries), "pack_sha256": judge.digest(signed)})
        )
        return 0
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        IndexError,
        RecursionError,
    ) as exc:
        print(f"claim import failed ({type(exc).__name__})", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
