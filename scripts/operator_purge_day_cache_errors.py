"""Operator tool: purge cached error responses from the evidence-proxy day cache.

The evidence proxy replays any recorded response to every identical request
for the rest of the UTC day. Before the recordability fix, error responses
(4xx/5xx captured during a transient provider outage) were recorded too, so
the failures kept replaying all day and every benchmark retry died against
the cache instead of reaching the recovered provider.

This removes ONLY entries with status >= 400 from the day-cache file,
preserving every good (2xx) entry so the warm cache stays warm. Dry-run by
default. The proxy holds the cache in memory: restart the proxy after a
--write purge so it reloads the cleaned file.

    python3 -m scripts.operator_purge_day_cache_errors --path /home/ec2-user/research_lab_evidence/day_cache.json
    python3 -m scripts.operator_purge_day_cache_errors --path ... --write
"""

import argparse
import json
import os
import sys
from collections import Counter


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", required=True, help="Path to day_cache.json")
    parser.add_argument("--write", action="store_true", help="Write the purged file (default: dry run)")
    args = parser.parse_args()

    with open(args.path, "r", encoding="utf-8") as handle:
        doc = json.load(handle)
    entries = doc.get("entries")
    if not isinstance(entries, dict):
        print("Unexpected schema: no 'entries' object — refusing to touch the file")
        return 2

    by_status = Counter()
    error_keys = []
    for key, record in entries.items():
        try:
            status = int((record or {}).get("status") or 0)
        except (TypeError, ValueError):
            status = 0
        by_status[status] += 1
        if status >= 400:
            error_keys.append(key)

    print(f"utc_day={doc.get('utc_day')} total_entries={len(entries)}")
    print(f"status distribution: {dict(sorted(by_status.items(), key=lambda x: -x[1]))}")
    print(f"error entries (status >= 400) to purge: {len(error_keys)}")
    if not error_keys:
        print("Nothing to purge.")
        return 0
    if not args.write:
        print("DRY RUN — re-run with --write to purge (then restart the proxy to reload).")
        return 0

    for key in error_keys:
        del entries[key]
    tmp = f"{args.path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(doc, handle, sort_keys=True, separators=(",", ":"))
    os.replace(tmp, args.path)

    with open(args.path, "r", encoding="utf-8") as handle:
        after = json.load(handle)
    remaining = after.get("entries") or {}
    residual_errors = sum(
        1 for r in remaining.values() if int((r or {}).get("status") or 0) >= 400
    )
    print(f"purged {len(error_keys)}; remaining entries={len(remaining)}; residual errors={residual_errors}")
    if residual_errors:
        print("ERROR: residual error entries remain — investigate")
        return 2
    print("Done. Restart the evidence proxy so the in-memory cache reloads the cleaned file.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
