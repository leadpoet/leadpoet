#!/usr/bin/env python3
"""Dry-run-first recorder for Research Lab L1 dev-eval provider snapshots.

Runs the CURRENT champion once over a chosen dev ICP set with LIVE providers,
capturing every provider response (Exa, Scrapingdog, OpenRouter, ...) into a
frozen snapshot set that `research_lab.eval.dev_eval.evaluate_dev` replays
deterministically (§6.3-1). Recording spends real provider budget, so it is
double-gated: the default invocation only prints the plan, and a live run
requires BOTH

  --record

and the environment gate

  RESEARCH_LAB_DEV_SNAPSHOT_RECORD_ENABLED=true

The dev ICP set is built with `build_dev_icp_set`, which hard-excludes any
ICP whose ref/hash/intent-signal signature appears in the supplied holdout
window hashes (leak-cluster guard) and prints the exclusion proof.

Champion runners:
  --adapter-path   private champion checkout, run in a subprocess (mirrors
                   SubprocessPrivateModelRunner with the record bootstrap
                   prepended) — the path used on a gateway box.
  --champion-image immutable ECR digest, run through docker with the snapshot
                   directory volume-mounted (mirrors DockerPrivateModelRunner).

Recording writes to a LOCAL directory (the in-process/in-container record
bootstrap persists files); sync the directory to the S3 prefix behind
RESEARCH_LAB_DEV_SNAPSHOT_URI afterwards if the fleet replays from S3.

Example (gateway box):

  RESEARCH_LAB_DEV_SNAPSHOT_RECORD_ENABLED=true \
  python3 scripts/record_research_lab_dev_snapshots.py \
      --source-icps /tmp/source_icps.json \
      --exclude-hashes /tmp/holdout_window_hashes.json \
      --size 6 --seed dev-v1 \
      --snapshot-dir /var/lib/research_lab/dev_snapshots/dev-v1 \
      --adapter-path /opt/champion \
      --record
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RECORD_ENABLED_ENV = "RESEARCH_LAB_DEV_SNAPSHOT_RECORD_ENABLED"
TRUTHY_VALUES = {"1", "true", "yes", "on"}

PROVIDER_KEY_GROUPS = (
    ("EXA_API_KEY",),
    ("SCRAPINGDOG_API_KEY", "QUALIFICATION_SCRAPINGDOG_API_KEY"),
    ("OPENROUTER_API_KEY", "QUALIFICATION_OPENROUTER_API_KEY", "OPENROUTER_KEY"),
)


def _load_json_file(path: str) -> Any:
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


def _load_source_items(path: str) -> list[dict[str, Any]]:
    decoded = _load_json_file(path)
    if isinstance(decoded, Mapping):
        for key in ("items", "benchmark_items", "icps"):
            if isinstance(decoded.get(key), list):
                decoded = decoded[key]
                break
    if not isinstance(decoded, list):
        raise ValueError(f"source ICP file must be a JSON list (or hold one): {path}")
    return [dict(item) for item in decoded if isinstance(item, Mapping)]


def _load_exclusions(args: argparse.Namespace) -> list[str]:
    exclusions: list[str] = list(args.exclude_hash or [])
    if args.exclude_hashes:
        decoded = _load_json_file(args.exclude_hashes)
        if isinstance(decoded, Mapping):
            decoded = decoded.get("hashes") or decoded.get("item_refs") or []
        if not isinstance(decoded, list):
            raise ValueError("exclude-hashes file must be a JSON list (or hold one)")
        exclusions.extend(str(item) for item in decoded)
    return exclusions


def _provider_key_presence() -> dict[str, bool]:
    return {
        "/".join(group): any(os.getenv(name) for name in group)
        for group in PROVIDER_KEY_GROUPS
    }


def _subprocess_env(snapshot_dir: str) -> dict[str, str]:
    from research_lab.eval.private_runtime import private_model_env_passthrough
    from research_lab.eval.snapshot_store import SNAPSHOT_DIR_ENV

    env = {"PATH": os.environ.get("PATH", ""), "PYTHONUNBUFFERED": "1"}
    for name in private_model_env_passthrough():
        if name in os.environ:
            env[name] = os.environ[name]
    env[SNAPSHOT_DIR_ENV] = snapshot_dir
    return env


def _record_icp_with_subprocess(
    *,
    adapter_path: str,
    module_name: str,
    callable_name: str,
    icp: Mapping[str, Any],
    snapshot_dir: str,
    timeout_seconds: int,
) -> list[Mapping[str, Any]]:
    """Run one champion ICP in a subprocess with the record bootstrap installed.

    Mirrors SubprocessPrivateModelRunner but prepends the snapshot record
    bootstrap so live provider responses are persisted per request key while
    passing through unchanged.
    """
    from research_lab.eval import private_runtime
    from research_lab.eval.snapshot_store import dev_record_bootstrap

    adapter_bootstrap = getattr(private_runtime, "_ADAPTER_BOOTSTRAP", None)
    if not adapter_bootstrap:
        raise RuntimeError("private_runtime adapter bootstrap is unavailable")
    payload = {
        "icp": private_runtime.canonicalize_private_model_icp(icp),
        "context": {"dev_snapshot_recording": True},
    }
    command = [
        sys.executable,
        "-c",
        dev_record_bootstrap() + adapter_bootstrap,
        str(Path(adapter_path).expanduser().resolve()),
        module_name,
        callable_name,
    ]
    completed = subprocess.run(
        command,
        input=json.dumps(payload, separators=(",", ":"), sort_keys=True),
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        env=_subprocess_env(snapshot_dir),
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"champion adapter failed with code {completed.returncode}: "
            f"{completed.stderr[-1200:]}"
        )
    decoded = json.loads(completed.stdout)
    if not isinstance(decoded, list):
        raise RuntimeError("champion adapter must return a JSON array")
    return decoded


def _record_icp_with_docker(
    *,
    image_digest: str,
    module_name: str,
    callable_name: str,
    icp: Mapping[str, Any],
    snapshot_dir: str,
    timeout_seconds: int,
    docker_executable: str = "docker",
) -> list[Mapping[str, Any]]:
    """Run one champion ICP through docker with the snapshot dir mounted.

    Mirrors DockerPrivateModelRunner but volume-mounts the snapshot directory
    and prepends the record bootstrap to the in-container adapter bootstrap.
    """
    from research_lab.eval import private_runtime
    from research_lab.eval.snapshot_store import (
        SNAPSHOT_DIR_ENV,
        dev_record_bootstrap,
    )

    docker_bootstrap = getattr(private_runtime, "_DOCKER_ADAPTER_BOOTSTRAP", None)
    if not docker_bootstrap:
        raise RuntimeError("private_runtime docker adapter bootstrap is unavailable")
    if "@sha256:" not in image_digest:
        raise RuntimeError("champion image must be an immutable digest")
    container_dir = "/research_lab_dev_snapshots"
    payload = {
        "icp": private_runtime.canonicalize_private_model_icp(icp),
        "context": {"dev_snapshot_recording": True},
    }
    env_args: list[str] = []
    for name in private_runtime.private_model_env_passthrough():
        if name in os.environ:
            env_args.extend(["-e", name])
    command = [
        docker_executable,
        "run",
        "--rm",
        "-i",
        "-v",
        f"{Path(snapshot_dir).expanduser().resolve()}:{container_dir}",
        "-e",
        f"{SNAPSHOT_DIR_ENV}={container_dir}",
        *env_args,
        image_digest,
        "python",
        "-c",
        dev_record_bootstrap() + docker_bootstrap,
        module_name,
        callable_name,
    ]
    completed = subprocess.run(
        command,
        input=json.dumps(payload, separators=(",", ":"), sort_keys=True),
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        env={**_subprocess_env(str(snapshot_dir)), "PATH": os.environ.get("PATH", "")},
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"docker champion adapter failed with code {completed.returncode}: "
            f"{completed.stderr[-1200:]}"
        )
    decoded = json.loads(completed.stdout)
    if not isinstance(decoded, list):
        raise RuntimeError("docker champion adapter must return a JSON array")
    return decoded


def _print_plan(
    *,
    dev_set: Any,
    snapshot_dir: str,
    runner_label: str,
    recording: bool,
) -> None:
    proof = dev_set.manifest["exclusion_proof"]
    print("Research Lab dev-snapshot recorder")
    print(f"  mode:                {'RECORD (live providers)' if recording else 'DRY RUN'}")
    print(f"  dev_set_hash:        {dev_set.dev_set_hash}")
    print(f"  selected_icps:       {len(dev_set.items)}")
    print(f"  source_icps:         {dev_set.manifest['source_icp_count']}")
    print(f"  excluded_icps:       {proof['excluded_item_count']} (leak-cluster guard)")
    print(f"  exclusion_set_hash:  {proof['exclusion_set_hash']}")
    print(f"  snapshot_dir:        {snapshot_dir}")
    print(f"  champion_runner:     {runner_label}")
    for group, present in _provider_key_presence().items():
        print(f"  provider_key[{group}]: {'present' if present else 'MISSING'}")
    for item in dev_set.items:
        print(f"    dev icp: {item['icp_ref']} {item['icp_hash']}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Record a frozen provider snapshot set for the L1 dev-eval rung"
    )
    parser.add_argument("--source-icps", required=True, help="JSON file of candidate dev ICPs (benchmark-item or raw ICP shapes)")
    parser.add_argument("--exclude-hashes", default="", help="JSON file listing holdout-window icp_refs/icp_hashes/signatures to exclude")
    parser.add_argument("--exclude-hash", action="append", default=[], help="Additional exclusion entry (repeatable)")
    parser.add_argument("--size", type=int, default=6, help="Dev ICP set size (default 6)")
    parser.add_argument("--seed", default="dev-v1", help="Deterministic selection seed (default dev-v1)")
    parser.add_argument("--snapshot-dir", default="", help="Local snapshot directory (default: RESEARCH_LAB_DEV_SNAPSHOT_URI when it is a local path)")
    parser.add_argument("--adapter-path", default="", help="Private champion checkout for subprocess execution")
    parser.add_argument("--champion-image", default="", help="Immutable champion ECR digest for docker execution")
    parser.add_argument("--module-name", default="research_lab_adapter")
    parser.add_argument("--callable-name", default="run_icp")
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--record", action="store_true", help=f"Actually run the champion with live providers (also requires {RECORD_ENABLED_ENV}=true)")
    args = parser.parse_args()

    from research_lab.canonical import utc_now_iso
    from research_lab.eval.dev_eval import build_dev_icp_set
    from research_lab.eval.snapshot_store import (
        MODE_RECORD,
        SNAPSHOT_URI_ENV,
        ProviderSnapshotStore,
    )

    try:
        source_items = _load_source_items(args.source_icps)
        exclusions = _load_exclusions(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}")
        return 1

    try:
        dev_set = build_dev_icp_set(
            source_items,
            exclude_window_hashes=exclusions,
            size=args.size,
            seed=args.seed,
        )
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"ERROR: could not build dev ICP set: {exc}")
        return 1

    snapshot_dir = str(args.snapshot_dir or os.getenv(SNAPSHOT_URI_ENV) or "").strip()
    if not snapshot_dir:
        print(f"ERROR: set --snapshot-dir or {SNAPSHOT_URI_ENV}")
        return 1
    if snapshot_dir.startswith("s3://"):
        print(
            "ERROR: recording requires a LOCAL snapshot directory (the record "
            "bootstrap writes files); record locally, then sync to S3."
        )
        return 1

    if args.adapter_path and args.champion_image:
        print("ERROR: pass exactly one of --adapter-path / --champion-image")
        return 1
    if args.adapter_path:
        runner_label = f"subprocess:{args.adapter_path}"
    elif args.champion_image:
        runner_label = f"docker:{args.champion_image}"
    else:
        runner_label = "NOT CONFIGURED (pass --adapter-path or --champion-image)"

    recording = bool(args.record)
    _print_plan(
        dev_set=dev_set,
        snapshot_dir=snapshot_dir,
        runner_label=runner_label,
        recording=recording,
    )

    if not recording:
        print("DRY RUN: no provider calls were made and nothing was written.")
        print(f"Re-run with --record and {RECORD_ENABLED_ENV}=true to record.")
        return 0

    if str(os.getenv(RECORD_ENABLED_ENV) or "").strip().lower() not in TRUTHY_VALUES:
        print(f"ERROR: --record requires {RECORD_ENABLED_ENV}=true")
        return 1
    if not args.adapter_path and not args.champion_image:
        print("ERROR: --record requires --adapter-path or --champion-image")
        return 1
    missing = [group for group, present in _provider_key_presence().items() if not present]
    if missing:
        print(f"ERROR: missing provider keys: {', '.join(missing)}")
        return 1

    store = ProviderSnapshotStore(snapshot_dir, mode=MODE_RECORD)
    failures = 0
    for item in dev_set.items:
        ref = item["icp_ref"]
        try:
            if args.adapter_path:
                companies = _record_icp_with_subprocess(
                    adapter_path=args.adapter_path,
                    module_name=args.module_name,
                    callable_name=args.callable_name,
                    icp=item["icp"],
                    snapshot_dir=snapshot_dir,
                    timeout_seconds=args.timeout_seconds,
                )
            else:
                companies = _record_icp_with_docker(
                    image_digest=args.champion_image,
                    module_name=args.module_name,
                    callable_name=args.callable_name,
                    icp=item["icp"],
                    snapshot_dir=snapshot_dir,
                    timeout_seconds=args.timeout_seconds,
                )
            print(f"recorded {ref}: {len(companies)} companies, snapshots={store.snapshot_count()}")
        except Exception as exc:  # noqa: BLE001 - keep recording the remaining ICPs
            failures += 1
            print(f"WARNING: recording failed for {ref}: {exc}")

    manifest = store.build_manifest(
        icp_set_hash=dev_set.dev_set_hash,
        dev_set_manifest=dev_set.manifest,
        recorded_at=utc_now_iso(),
    )
    store.write_manifest(manifest)
    verification = store.verify_manifest(expected_icp_set_hash=dev_set.dev_set_hash)
    print(f"snapshot_count={manifest['snapshot_count']}")
    print(f"content_hash={manifest['content_hash']}")
    print(f"manifest_hash={manifest['manifest_hash']}")
    print(f"manifest_verified={verification['passed']} errors={verification['errors']}")
    if failures:
        print(f"WARNING: {failures} ICP(s) failed to record; snapshot set may be partial.")
    return 0 if verification["passed"] and not failures else 1


if __name__ == "__main__":
    sys.exit(main())
