"""Run one centrally pinned research assignment on the trusted Linux validator.

stdin/stdout are a private authenticated operator channel. Never print payloads
or environment values on errors. The OCI command and every mount are host-set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import uuid

from lab_arena import contracts, operations, output, runner, runtime, source_bundle
from lab_arena.collection_intake import MODELS
from lab_arena.cache_retention import workspace_cache_lock
from lab_arena.collection_research import PipeAPI, encode, receive, send
from lab_arena.collection_workflow import CHAMPION_REV
from lab_arena.web_egress import WebEgressServer


def validated_completion(result, company_count):
    """A zero exit can accompany a failed native supervisor or partial checkpoint."""
    if result.stderr_truncated:
        return False
    summaries = []
    for line in result.stderr.splitlines():
        if not line.startswith(b"{"):
            continue
        try:
            doc = json.loads(line)
        except ValueError:
            continue
        if not isinstance(doc, dict):
            continue
        if "tyche_arena_error" in doc or "tyche_arena_supervisor_exit" in doc:
            return False
        if "tyche_arena_exit" in doc:
            summaries.append(doc["tyche_arena_exit"])
    if len(summaries) != 1 or not isinstance(summaries[0], dict):
        return False
    report = summaries[0]
    return (
        report.get("final_review_status") == "validated"
        and type(report.get("published_count")) is int
        and report["published_count"] == company_count
        and report.get("held_published_count") == 0
        and report.get("unconfirmed_published_count") == 0
    )


def run_assignment(config, job, api, sandbox):
    if (
        set(job)
        != {
            "type",
            "job_id",
            "scenario_id",
            "model",
            "harness_commit",
            "evaluation_date",
            "icp",
            "deepline_catalog",
            "scorer_image_digest",
        }
        or job["type"] != "research_start"
        or job["harness_commit"] != CHAMPION_REV
        or job["model"] not in MODELS
    ):
        raise ValueError("unexpected research assignment")
    identity = str(uuid.UUID(job["job_id"]))
    rootfs = config.rootfs.resolve(strict=True)
    if (
        config.image_digest != job["scorer_image_digest"]
        or not re.fullmatch(r"sha256:[a-f0-9]{64}", config.image_digest)
        or (rootfs.parent / ".exported").read_text().strip() != config.image_digest
        or rootfs.name != "rootfs"
        or not rootfs.is_dir()
    ):
        raise ValueError("trusted scorer image unavailable")
    work = config.work_dir.resolve(strict=True)
    # The canonical execution host preinstalls the champion's pinned wheels.
    # Never install code supplied by the remote request or share writable mounts.
    deps = config.dependencies.resolve(strict=True)
    if not deps.is_dir() or deps.stat().st_mode & 0o222:
        raise ValueError("readonly dependencies required")
    contents = subprocess.check_output(
        [
            "git",
            "-C",
            str(config.champion_checkout),
            "archive",
            "--format=tar.gz",
            CHAMPION_REV,
        ]
    )
    code_root = Path(runner.__file__).parent
    with tempfile.TemporaryDirectory(prefix="research-", dir=work) as td:
        root = Path(td)
        source = root / "source"
        source.mkdir()
        source_bundle.extract_source_archive(contents, source)
        if (
            not runner.SourceCache._compatible_dependencies(deps.parent)
            or (deps.parent / "source" / "requirements.txt").read_bytes()
            != (source / "requirements.txt").read_bytes()
        ):
            raise ValueError(
                "dependency cache ABI or requirements differ from pinned champion"
            )
        inp, out, sockets = (root / name for name in ["input", "output", "sockets"])
        for d in (inp, out, sockets):
            d.mkdir(mode=0o700)
        icp = dict(
            job["icp"]
        )  # exact retained private agent-visible input; no evaluator fields supplied
        schema = job["icp"].get("output_schema_version")
        if schema != "leadpoet.lab_arena.output.v6":
            raise ValueError("research requires company-only output")
        icp["output_schema_version"] = schema
        document = {
            "schema_version": "leadpoet.lab_arena.icp_input.v1",
            "icp": icp,
            "model": job["model"],
            "evaluation_date": job["evaluation_date"],
            "output_schema_version": schema,
            "company_limit": int(icp.get("max_companies", 5)),
            "deepline_catalog": job["deepline_catalog"],
            "provider_operations": [
                "openrouter.responses",
                "openrouter.chat",
                "deepline.execute",
                *sorted(operations.CATALOG_OPERATIONS),
            ],
        }
        if not 1 <= document["company_limit"] <= 5:
            raise ValueError("invalid company limit")
        (inp / runtime.INPUT_FILE_NAME).write_bytes(encode(document))
        mounts = {
            name: runner._stage_agent_entrypoint(
                code_root / filename, root, filename=target
            )
            for name, (filename, target) in {
                "agent_entrypoint_path": ("agent_entrypoint.py", "agent-entrypoint.py"),
                "checkpoint_module_path": (
                    "lab_arena_checkpoint.py",
                    "lab_arena_checkpoint.py",
                ),
                "codex_module_path": ("lab_arena_codex.py", "lab_arena_codex.py"),
                "web_bridge_path": ("web_egress_bridge.py", "web-egress-bridge.py"),
            }.items()
        }
        path = sockets / runtime.SANDBOX_SOCKET_NAME
        state = runner.RunState(
            lease={
                "run_id": identity,
                "kind": "execute",
                "deepline_catalog": job["deepline_catalog"],
                "reasoning_capture_policy": "full_generated_thinking_v1",
            },
            lease_token="local",
        )
        spec = runtime.SandboxSpec(
            sandbox_id="collection-" + identity,
            rootfs_path=rootfs,
            input_dir=inp,
            output_dir=out,
            socket_path=path,
            entry_command=runtime.AGENT_ENTRY_COMMAND + ("--locked-research",),
            working_dir=runtime.AGENT_WORKING_DIR,
            evaluation_date=job["evaluation_date"],
            random_seed=int(hashlib.sha256(identity.encode()).hexdigest()[:8], 16),
            source_dir=source,
            dependency_dir=deps,
            wall_clock_seconds=config.seconds,
            **mounts,
        )
        server = runner.WorkerSocketServer(path, api, state)
        with WebEgressServer(
            sockets / runtime.SANDBOX_WEB_SOCKET_NAME, proxy_url=config.web_proxy
        ):
            server.start()
            try:
                result = sandbox.run_icp(spec)
            finally:
                server.stop()
        if (
            result.timed_out
            or result.exit_code != 0
            or result.output_error
            or not result.output_bytes
            or result.checkpoint_output_invalid
        ):
            raise ValueError("no complete research result")
        parsed = output.output_document_from_bytes(
            result.output_bytes,
            require_intent_dates=True,
            expected_schema_version=schema,
        )
        return {
            "type": "research_result",
            "execution_completed": validated_completion(
                result, len(parsed["companies"])
            ),
            "job_id": identity,
            "companies": parsed["companies"],
        }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--champion-checkout", type=Path, required=True)
    parser.add_argument("--rootfs", type=Path, required=True)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--dependencies", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--runsc", type=Path, default=Path("/usr/local/bin/runsc"))
    parser.add_argument("--seconds", type=int, default=contracts.ICP_WALL_CLOCK_SECONDS)
    args = parser.parse_args(argv)
    if not 60 <= args.seconds <= contracts.ICP_WALL_CLOCK_SECONDS:
        parser.error("invalid time limit")
    args.web_proxy = os.environ.get("LEADPOET_COLLECTION_WEB_PROXY") or None
    # Reserve stdout exclusively for the private protocol even if a legacy
    # component writes a progress line. Do not log traces from this process.
    wire = sys.stdout.buffer
    sys.stdout = sys.stderr
    try:
        job = receive(sys.stdin.buffer)
        if job == {"type": "idle"}:
            return 0
        sandbox_work = args.work_dir.resolve(strict=True) / "sandboxes"
        sandbox_work.mkdir(mode=0o700, exist_ok=True)
        sandbox = runtime.RunscRuntime(
            runtime.RuntimeConfig(runsc_path=args.runsc, work_dir=sandbox_work)
        )
        cache_root = args.rootfs.resolve(strict=True).parents[2]
        with workspace_cache_lock(cache_root, exclusive=False, create=False):
            result = run_assignment(args, job, PipeAPI(sys.stdin.buffer, wire), sandbox)
        send(wire, result)
        return 0
    except Exception as exc:
        send(wire, {"type": "research_error", "code": type(exc).__name__})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
