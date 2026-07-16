#!/usr/bin/env python3
"""Dry-run-first recorder for current-day Research Lab dev snapshots.

Runs the CURRENT champion once over the complete scored daily benchmark bank,
capturing every provider response (Exa, Scrapingdog, OpenRouter, ...) into a
frozen snapshot set that `research_lab.eval.dev_eval.evaluate_dev` replays
deterministically (§6.3-1). Recording spends real provider budget, so it is
double-gated: the default invocation only prints the plan, and a live run
requires BOTH

  --record

and the environment gate

  RESEARCH_LAB_DEV_SNAPSHOT_RECORD_ENABLED=true

The per-tree weak/strong cohort is selected later from this immutable bank.
This recorder never chooses ICPs from retired sets and never prints hidden ICP
refs or payloads.

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
      --snapshot-dir /var/lib/research_lab/dev_snapshots/dev-v1 \
      --champion-image <immutable-ecr-digest> \
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

from gateway.research_lab.config import (  # noqa: E402
    MAX_RESEARCH_LAB_GIT_TREE_ICP_COUNT,
    RESEARCH_LAB_GIT_TREE_ENV_BY_FIELD,
    ResearchLabGitTreeConfig,
    ResearchLabGitTreeConfigError,
)

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


def _load_source_export(path: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    decoded = _load_json_file(path)
    if (
        not isinstance(decoded, Mapping)
        or decoded.get("schema_version") != "research_lab.dev_icp_export.v2"
        or not isinstance(decoded.get("items"), list)
        or not isinstance(decoded.get("daily_bank_manifest"), Mapping)
    ):
        raise ValueError(
            "source ICP file must be a current-day dev ICP export v2 document"
        )
    items = [
        dict(item) for item in decoded["items"] if isinstance(item, Mapping)
    ]
    if len(items) != len(decoded["items"]):
        raise ValueError("source ICP export contains an invalid item")
    return items, dict(decoded["daily_bank_manifest"])


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
    print("Research Lab dev-snapshot recorder")
    print(f"  mode:                {'RECORD (live providers)' if recording else 'DRY RUN'}")
    print(f"  daily_bank_hash:     {dev_set.manifest['daily_bank_hash']}")
    print(f"  benchmark_date:      {dev_set.manifest['benchmark_date']}")
    print(f"  bank_icps:           {len(dev_set.items)}")
    print(f"  snapshot_dir:        {snapshot_dir}")
    print(f"  champion_runner:     {runner_label}")
    for group, present in _provider_key_presence().items():
        print(f"  provider_key[{group}]: {'present' if present else 'MISSING'}")


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
    parser.add_argument("--source-icps", required=True, help="Current-day dev ICP export v2 JSON")
    parser.add_argument(
        "--size",
        type=int,
        default=None,
        help=(
            "Compatibility check only; when supplied it must match "
            + RESEARCH_LAB_GIT_TREE_ENV_BY_FIELD["live_max_icps_per_node"]
        ),
    )
    parser.add_argument("--snapshot-dir", default="", help="Local snapshot directory (default: RESEARCH_LAB_DEV_SNAPSHOT_URI when it is a local path)")
    parser.add_argument("--adapter-path", default="", help="Private champion checkout for subprocess execution")
    parser.add_argument("--champion-image", default="", help="Immutable champion ECR digest for docker execution")
    parser.add_argument("--source-commit", default=os.getenv("RESEARCH_LAB_PRIVATE_COMMIT_SHA", ""))
    parser.add_argument("--model-config-hash", default=os.getenv("RESEARCH_LAB_PRIVATE_MODEL_CONFIG_HASH", ""))
    parser.add_argument("--private-model-manifest-hash", required=True)
    parser.add_argument("--provider-model-id", action="append", default=[])
    parser.add_argument("--module-name", default="research_lab_adapter")
    parser.add_argument("--callable-name", default="run_icp")
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--record", action="store_true", help=f"Actually run the champion with live providers (also requires {RECORD_ENABLED_ENV}=true)")
    args = parser.parse_args()

    try:
        configured_icp_count = (
            ResearchLabGitTreeConfig.from_env().live_max_icps_per_node
        )
    except ResearchLabGitTreeConfigError as exc:
        print(f"ERROR: invalid Git-tree configuration: {exc}")
        return 1
    if not 1 <= configured_icp_count <= MAX_RESEARCH_LAB_GIT_TREE_ICP_COUNT:
        print("ERROR: configured Git-tree development ICP count is invalid")
        return 1
    if args.size is not None and args.size != configured_icp_count:
        print(
            "ERROR: --size differs from the configured Git-tree development "
            f"ICP count ({configured_icp_count})"
        )
        return 1

    from research_lab.canonical import utc_now_iso
    from research_lab.eval.dev_eval import build_current_day_dev_bank
    from research_lab.eval.snapshot_store import (
        MODE_RECORD,
        SNAPSHOT_URI_ENV,
        ProviderSnapshotStore,
    )

    try:
        source_items, source_bank_manifest = _load_source_export(
            args.source_icps
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}")
        return 1

    try:
        dev_set = build_current_day_dev_bank(
            source_items,
            benchmark_date=str(source_bank_manifest.get("benchmark_date") or ""),
            benchmark_bundle_id=str(
                source_bank_manifest.get("benchmark_bundle_id") or ""
            ),
            benchmark_bundle_hash=str(
                source_bank_manifest.get("benchmark_bundle_hash") or ""
            ),
            rolling_window_hash=str(
                source_bank_manifest.get("rolling_window_hash") or ""
            ),
            private_model_manifest_hash=str(
                source_bank_manifest.get("private_model_manifest_hash") or ""
            ),
            evaluation_epoch=int(
                source_bank_manifest.get("evaluation_epoch") or 0
            ),
        )
        if dev_set.manifest != source_bank_manifest:
            raise ValueError("daily bank manifest differs from exported ICPs")
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"ERROR: could not validate current-day dev ICP bank: {exc}")
        return 1
    if len(dev_set.items) < configured_icp_count:
        print("ERROR: daily bank is smaller than the configured per-node ICP count")
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
    if str(args.private_model_manifest_hash) != str(
        dev_set.manifest.get("private_model_manifest_hash") or ""
    ):
        print("ERROR: daily baseline model manifest differs from the active champion")
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
        for item_index, item in enumerate(dev_set.items, start=1):
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
                print(
                    f"recorded daily ICP {item_index}/{len(dev_set.items)}: "
                    f"{len(companies)} companies, snapshots={store.snapshot_count()}"
                )
            except Exception as exc:  # noqa: BLE001 - collect every failed ICP
                runner_failure_refs.append(str(ref))
                print(
                    f"WARNING: recording failed for daily ICP {item_index}: "
                    f"{type(exc).__name__}"
                )

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
                "private_model_manifest_hash": str(
                    args.private_model_manifest_hash
                ),
                "provider_model_ids": provider_model_ids,
                "replay_output_hashes": replay_output_hashes,
            },
        )
        store.write_manifest(manifest)
        verification = store.verify_manifest(expected_icp_set_hash=dev_set.dev_set_hash)
        ready = store.build_ready_document(manifest)
        if verification["passed"] and not failure_summary["has_failures"]:
            store.write_ready_document(ready)
        ready_verification = store.verify_ready_document(
            expected_dev_icp_count=configured_icp_count,
            require_signature=False,
        )
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
