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
      --size 8 --seed dev-v1 \
      --snapshot-dir /var/lib/research_lab/dev_snapshots/dev-v1 \
      --adapter-path /opt/champion \
      --record
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid
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


def _subprocess_env(snapshot_dir: str, *, icp_ref: str = "") -> dict[str, str]:
    from research_lab.eval.private_runtime import private_model_env_passthrough
    from research_lab.eval.snapshot_store import SNAPSHOT_DIR_ENV

    env = {"PATH": os.environ.get("PATH", ""), "PYTHONUNBUFFERED": "1"}
    for name in private_model_env_passthrough():
        if name in os.environ:
            env[name] = os.environ[name]
    env[SNAPSHOT_DIR_ENV] = snapshot_dir
    env["RESEARCH_LAB_DEV_RECORD_ICP_REF"] = str(icp_ref)
    return env


def _record_icp_with_subprocess(
    *,
    adapter_path: str,
    module_name: str,
    callable_name: str,
    icp: Mapping[str, Any],
    icp_ref: str = "",
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
        env=_subprocess_env(snapshot_dir, icp_ref=icp_ref),
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
    icp_ref: str = "",
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
        "-e",
        f"RESEARCH_LAB_DEV_RECORD_ICP_REF={icp_ref}",
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
        env={
            **_subprocess_env(str(snapshot_dir), icp_ref=icp_ref),
            "PATH": os.environ.get("PATH", ""),
        },
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


def _replay_icp_with_docker(
    *,
    image_digest: str,
    module_name: str,
    callable_name: str,
    icp: Mapping[str, Any],
    snapshot_dir: str,
    timeout_seconds: int,
    docker_executable: str = "docker",
) -> list[Mapping[str, Any]]:
    """Replay one ICP with networking disabled to prove the set is complete."""
    from research_lab.eval import private_runtime
    from research_lab.eval.snapshot_store import (
        MISS_POLICY_STRICT,
        container_replay_env,
        dev_replay_bootstrap,
    )

    docker_bootstrap = getattr(private_runtime, "_DOCKER_ADAPTER_BOOTSTRAP", None)
    if not docker_bootstrap:
        raise RuntimeError("private_runtime docker adapter bootstrap is unavailable")
    container_dir = "/research_lab_dev_snapshots"
    payload = {
        "icp": private_runtime.canonicalize_private_model_icp(icp),
        "context": {"dev_snapshot_replay_validation": True},
    }
    env_args: list[str] = []
    for name, value in container_replay_env(
        container_dir, miss_policy=MISS_POLICY_STRICT
    ).items():
        env_args.extend(["-e", f"{name}={value}"])
    command = [
        docker_executable,
        "run",
        "--rm",
        "-i",
        "--network",
        "none",
        "-v",
        f"{Path(snapshot_dir).expanduser().resolve()}:{container_dir}:ro",
        *env_args,
        image_digest,
        "python",
        "-c",
        dev_replay_bootstrap() + docker_bootstrap,
        module_name,
        callable_name,
    ]
    completed = subprocess.run(
        command,
        input=json.dumps(payload, separators=(",", ":"), sort_keys=True),
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        env={**os.environ},
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"offline replay failed with code {completed.returncode}: "
            f"{completed.stderr[-1200:]}"
        )
    decoded = json.loads(completed.stdout)
    if not isinstance(decoded, list):
        raise RuntimeError("offline replay adapter must return a JSON array")
    return [dict(item) for item in decoded if isinstance(item, Mapping)]


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


def _recording_failure_summary(
    *,
    runner_failure_refs: Sequence[str],
    failure_file: Path,
) -> dict[str, Any]:
    """Return deduplicated event and affected-ICP counts for one recording run."""
    provider_events: set[tuple[str, str, str]] = set()
    invalid_rows = 0
    if failure_file.exists():
        for line in failure_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                invalid_rows += 1
                continue
            if not isinstance(row, Mapping):
                invalid_rows += 1
                continue
            provider_events.add(
                (
                    str(row.get("icp_ref") or ""),
                    str(row.get("request_key") or ""),
                    str(row.get("reason") or "record_failure"),
                )
            )

    runner_refs = {str(ref) for ref in runner_failure_refs if str(ref)}
    provider_refs = {event[0] for event in provider_events if event[0]}
    return {
        "runner_failure_count": len(runner_refs),
        "provider_failure_event_count": len(provider_events) + invalid_rows,
        "failed_icp_count": len(runner_refs | provider_refs),
        "unattributed_provider_failure_count": (
            sum(1 for event in provider_events if not event[0]) + invalid_rows
        ),
        "has_failures": bool(runner_refs or provider_events or invalid_rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Record a frozen provider snapshot set for the L1 dev-eval rung"
    )
    parser.add_argument("--source-icps", required=True, help="JSON file of candidate dev ICPs (benchmark-item or raw ICP shapes)")
    parser.add_argument("--exclude-hashes", default="", help="JSON file listing holdout-window icp_refs/icp_hashes/signatures to exclude")
    parser.add_argument("--exclude-hash", action="append", default=[], help="Additional exclusion entry (repeatable)")
    parser.add_argument("--size", type=int, default=8, help="Dev ICP set size (must be 8)")
    parser.add_argument("--seed", default="dev-v1", help="Deterministic selection seed (default dev-v1)")
    parser.add_argument("--snapshot-dir", default="", help="Local snapshot directory (default: RESEARCH_LAB_DEV_SNAPSHOT_URI when it is a local path)")
    parser.add_argument("--adapter-path", default="", help="Private champion checkout for subprocess execution")
    parser.add_argument("--champion-image", default="", help="Immutable champion ECR digest for docker execution")
    parser.add_argument("--source-commit", default=os.getenv("RESEARCH_LAB_PRIVATE_COMMIT_SHA", ""))
    parser.add_argument("--model-config-hash", default=os.getenv("RESEARCH_LAB_PRIVATE_MODEL_CONFIG_HASH", ""))
    parser.add_argument("--provider-model-id", action="append", default=[])
    parser.add_argument("--module-name", default="research_lab_adapter")
    parser.add_argument("--callable-name", default="run_icp")
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--record", action="store_true", help=f"Actually run the champion with live providers (also requires {RECORD_ENABLED_ENV}=true)")
    args = parser.parse_args()

    if args.size != 8:
        print("ERROR: production development snapshots require exactly 8 ICPs")
        return 1

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

    if args.adapter_path:
        print("ERROR: production snapshot recording requires --champion-image; --adapter-path is not isolated")
        return 1
    if args.champion_image:
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
    if not args.champion_image:
        print("ERROR: production recording requires --champion-image for offline replay proof")
        return 1
    if "@sha256:" not in args.champion_image:
        print("ERROR: --champion-image must be an immutable image digest")
        return 1
    if len(str(args.source_commit)) != 40:
        print("ERROR: --source-commit must be the exact 40-character champion commit")
        return 1
    if not str(args.model_config_hash).startswith("sha256:"):
        print("ERROR: --model-config-hash must be an exact sha256 commitment")
        return 1
    provider_model_ids = sorted({str(item).strip() for item in args.provider_model_id if str(item).strip()})
    if not provider_model_ids:
        print("ERROR: pass at least one --provider-model-id used by the champion")
        return 1
    missing = [group for group, present in _provider_key_presence().items() if not present]
    if missing:
        print(f"ERROR: missing provider keys: {', '.join(missing)}")
        return 1

    target = Path(snapshot_dir).expanduser().resolve()
    staging = target.with_name(f".{target.name}.recording.{os.getpid()}.{uuid.uuid4().hex}")
    if target.exists():
        print(f"ERROR: immutable snapshot destination already exists: {target}")
        return 1
    staging.mkdir(parents=True, exist_ok=False)
    store = ProviderSnapshotStore(str(staging), mode=MODE_RECORD)
    runner_failure_refs: list[str] = []
    replay_output_hashes: list[dict[str, str]] = []
    try:
        for item in dev_set.items:
            ref = item["icp_ref"]
            try:
                companies = _record_icp_with_docker(
                    image_digest=args.champion_image,
                    module_name=args.module_name,
                    callable_name=args.callable_name,
                    icp=item["icp"],
                    icp_ref=ref,
                    snapshot_dir=str(staging),
                    timeout_seconds=args.timeout_seconds,
                )
                print(f"recorded {ref}: {len(companies)} companies, snapshots={store.snapshot_count()}")
            except Exception as exc:  # noqa: BLE001 - collect every failed ICP
                runner_failure_refs.append(str(ref))
                print(f"WARNING: recording failed for {ref}: {exc}")

        failure_file = staging / "record_failures.jsonl"
        failure_summary = _recording_failure_summary(
            runner_failure_refs=runner_failure_refs,
            failure_file=failure_file,
        )
        if failure_summary["provider_failure_event_count"]:
            print(
                "WARNING: "
                f"{failure_summary['provider_failure_event_count']} distinct provider "
                "snapshot failure event(s) recorded"
            )

        store.write_dev_icp_items(dev_set.items)
        if not failure_summary["has_failures"]:
            for item in dev_set.items:
                outputs = _replay_icp_with_docker(
                    image_digest=args.champion_image,
                    module_name=args.module_name,
                    callable_name=args.callable_name,
                    icp=item["icp"],
                    snapshot_dir=str(staging),
                    timeout_seconds=args.timeout_seconds,
                )
                encoded = json.dumps(outputs, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                replay_output_hashes.append(
                    {
                        "icp_hash": str(item["icp_hash"]),
                        "output_hash": "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
                    }
                )

        manifest = store.build_manifest(
            icp_set_hash=dev_set.dev_set_hash,
            dev_set_manifest=dev_set.manifest,
            recorded_at=utc_now_iso(),
            provenance={
                "champion_image_digest": args.champion_image,
                "source_commit": str(args.source_commit),
                "model_config_hash": str(args.model_config_hash),
                "provider_model_ids": provider_model_ids,
                "replay_output_hashes": replay_output_hashes,
            },
        )
        store.write_manifest(manifest)
        verification = store.verify_manifest(expected_icp_set_hash=dev_set.dev_set_hash)
        ready = store.build_ready_document(manifest)
        if verification["passed"] and not failure_summary["has_failures"]:
            store.write_ready_document(ready)
        ready_verification = store.verify_ready_document(require_signature=False)
        print(f"snapshot_count={manifest['snapshot_count']}")
        print(f"content_hash={manifest['content_hash']}")
        print(f"manifest_hash={manifest['manifest_hash']}")
        print(f"manifest_verified={verification['passed']} errors={verification['errors']}")
        print(f"ready_verified={ready_verification['passed']} errors={ready_verification['errors']}")
        if failure_summary["has_failures"]:
            print(
                "WARNING: snapshot set was not published: "
                f"failed_icps={failure_summary['failed_icp_count']} "
                f"runner_failures={failure_summary['runner_failure_count']} "
                "provider_failure_events="
                f"{failure_summary['provider_failure_event_count']} "
                "unattributed_provider_failures="
                f"{failure_summary['unattributed_provider_failure_count']}"
            )
        if (
            not verification["passed"]
            or not ready_verification["passed"]
            or failure_summary["has_failures"]
        ):
            return 1
        os.replace(staging, target)
        print(f"snapshot_ready={target}")
        return 0
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
