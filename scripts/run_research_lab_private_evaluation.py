#!/usr/bin/env python3
"""Run a real private-model Research Lab evaluation.

This is the worker boundary for Phase 1 Research Lab evaluation. It does not
clone private repos or create fake improvements. It runs the supplied private
adapter against a sealed benchmark payload, scores base vs candidate, writes a
score bundle, and can optionally POST that bundle to the gateway internal API.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_lab.eval import (  # noqa: E402
    CandidatePatchManifest,
    PrivateModelAdapterSpec,
    PrivateModelArtifactManifest,
    SealedBenchmarkSet,
    SubprocessPrivateModelRunner,
    evaluate_private_model_pair,
)
from research_lab.observability.langfuse_client import flush_langfuse, observation  # noqa: E402
from research_lab.observability.redaction import miner_hotkey_hash  # noqa: E402
from research_lab.observability.tracing import finish_score_bundle_observation  # noqa: E402


def main() -> int:
    args = _parse_args()
    artifact = PrivateModelArtifactManifest.from_mapping(_read_json(args.artifact_manifest))
    benchmark_payload = _read_json(args.benchmark)
    patch = CandidatePatchManifest.from_mapping(_read_json(args.patch_manifest))
    run_context = _read_json(args.run_context)

    benchmark = SealedBenchmarkSet.from_mapping(benchmark_payload)
    benchmark_items = benchmark_payload.get("items") or []
    if not benchmark_items:
        raise SystemExit("ERROR: benchmark file must include private sealed items under `items`")

    runner = SubprocessPrivateModelRunner(
        PrivateModelAdapterSpec(
            source_path=Path(args.private_model_path).expanduser().resolve(),
            module_name=args.adapter_module,
            callable_name=args.adapter_callable,
            timeout_seconds=args.timeout_seconds,
        )
    )
    with observation(
        "research_lab.private_eval_pair",
        metadata={
            "run_id": str(run_context.get("run_id") or ""),
            "ticket_id": str(run_context.get("ticket_id") or ""),
            "miner_hotkey_hash": miner_hotkey_hash(str(run_context.get("miner_hotkey") or "")),
            "parent_artifact_hash": artifact.model_artifact_hash,
            "candidate_patch_hash": patch.manifest_hash(),
            "icp_set_hash": benchmark.icp_set_hash,
            "benchmark_split_ref": benchmark.split_ref,
            "standalone_eval": True,
        },
    ) as obs:
        score_bundle = asyncio.run(
            evaluate_private_model_pair(
                artifact_manifest=artifact,
                benchmark=benchmark,
                patch_manifest=patch,
                benchmark_items=benchmark_items,
                base_runner=runner,
                candidate_runner=runner,
                run_context=run_context,
                policy=_read_json(args.policy) if args.policy else {},
            )
        )
        trace_id = finish_score_bundle_observation(obs, score_bundle)
        if trace_id:
            score_bundle = {**score_bundle, "langfuse_trace_id": trace_id}
    flush_langfuse()

    _write_json(args.output, score_bundle)
    print(f"Wrote Research Lab score bundle: {args.output}")
    print(f"score_bundle_hash={score_bundle['score_bundle_hash']}")

    if args.gateway_url:
        _post_gateway_score_bundle(args.gateway_url, args.internal_api_key, score_bundle)
        print("Posted Research Lab score bundle to gateway")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-model-path", required=True, help="Path to immutable private model checkout/artifact mount")
    parser.add_argument("--artifact-manifest", required=True, type=Path, help="JSON manifest for the private model artifact")
    parser.add_argument("--benchmark", required=True, type=Path, help="Sealed benchmark JSON; includes private `items` for evaluator only")
    parser.add_argument("--patch-manifest", required=True, type=Path, help="Candidate patch manifest JSON")
    parser.add_argument("--run-context", required=True, type=Path, help="Run context JSON")
    parser.add_argument("--policy", type=Path, help="Optional evaluation policy JSON")
    parser.add_argument("--output", required=True, type=Path, help="Output score-bundle JSON path")
    parser.add_argument("--adapter-module", default="research_lab_adapter")
    parser.add_argument("--adapter-callable", default="run_icp")
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--gateway-url", default="", help="Optional gateway URL for internal score-bundle POST")
    parser.add_argument("--internal-api-key", default="", help="Required when --gateway-url is set")
    return parser.parse_args()


def _read_json(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise SystemExit(f"ERROR: expected JSON object in {path}")
    return data


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _post_gateway_score_bundle(gateway_url: str, internal_api_key: str, score_bundle: dict) -> None:
    if not internal_api_key:
        raise SystemExit("ERROR: --internal-api-key is required with --gateway-url")
    body = json.dumps({"score_bundle": score_bundle}, separators=(",", ":")).encode("utf-8")
    req = urllib.request.Request(
        gateway_url.rstrip("/") + "/research-lab/evaluations/score-bundles",
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "x-leadpoet-internal-key": internal_api_key,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            if resp.status >= 300:
                raise SystemExit(f"ERROR: gateway returned HTTP {resp.status}")
    except urllib.error.HTTPError as exc:
        detail = exc.read(800).decode("utf-8", errors="replace")
        raise SystemExit(f"ERROR: gateway rejected score bundle: HTTP {exc.code}: {detail}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
