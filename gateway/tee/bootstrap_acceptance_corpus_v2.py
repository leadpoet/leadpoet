"""Build sanitized V2 acceptance fixtures from append-only production history.

This is an operator-side, pre-activation tool. It does not mutate production.
Rows are fetched through PostgREST, reduced to an explicit non-secret schema,
and written as canonical content-addressed fixtures. Promotion coverage is
replayed through the same pure promotion authority used by the measured
runtime; branch labels are never copied from synthetic database rows.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Dict, Mapping, Optional, Sequence
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from gateway.tee.acceptance_corpus_v2 import (
    REQUIRED_PROMOTION_BRANCHES,
    build_acceptance_corpus_from_index_v2,
    validate_acceptance_corpus_v2,
)
from research_lab.eval.promotion_metric import promotion_gate_decision


_HASH_RE = re.compile(r"^(?:sha256:)?[0-9a-f]{64}$")
_ISO_FRACTION_RE = re.compile(
    r"^(?P<head>.*[Tt ]\d{2}:\d{2}:\d{2})\."
    r"(?P<fraction>\d+)(?P<zone>Z|[+-]\d{2}:\d{2})$"
)
_FORBIDDEN = re.compile(
    r"(sk-or-|sb_secret|service_role|raw_secret|api[_-]?key|authorization|"
    r"proxy[_-]?(?:url|authorization)|request_body|response_body|provider_output|"
    r"://[^/\s]+:[^/@\s]+@)",
    re.IGNORECASE,
)
_PAGE_SIZE = 500


class AcceptanceCorpusBootstrapV2Error(RuntimeError):
    """Historical evidence cannot safely satisfy the V2 activation gate."""


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise AcceptanceCorpusBootstrapV2Error(
            "historical fixture is not canonical JSON"
        ) from exc


def _hash(value: Any) -> str:
    payload = value if isinstance(value, bytes) else _canonical(value)
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _normalized_hash(value: Any, *, fallback: Any) -> str:
    text = str(value or "").strip().lower()
    if _HASH_RE.fullmatch(text):
        return text if text.startswith("sha256:") else "sha256:" + text
    return _hash(fallback)


def _timestamp(value: Any) -> str:
    text = str(value or "")
    fraction_match = _ISO_FRACTION_RE.fullmatch(text)
    if fraction_match:
        fraction = (fraction_match.group("fraction") + "000000")[:6]
        text = (
            fraction_match.group("head")
            + "."
            + fraction
            + fraction_match.group("zone")
        )
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise AcceptanceCorpusBootstrapV2Error(
            "historical fixture timestamp is invalid"
        ) from exc
    if parsed.tzinfo is None:
        raise AcceptanceCorpusBootstrapV2Error(
            "historical fixture timestamp lacks timezone"
        )
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _reject_secret_markers(value: Any) -> None:
    if _FORBIDDEN.search(_canonical(value).decode("ascii")):
        raise AcceptanceCorpusBootstrapV2Error(
            "sanitized historical fixture contains a forbidden secret marker"
        )


class PostgrestHistoryReader:
    def __init__(self, *, url: str, service_role_key: str) -> None:
        base = str(url or "").strip().rstrip("/")
        key = str(service_role_key or "").strip()
        if not base.startswith("https://") or not key:
            raise AcceptanceCorpusBootstrapV2Error(
                "Supabase URL and service-role key are required"
            )
        self._base = base + "/rest/v1/"
        self._headers = {"apikey": key, "Authorization": "Bearer " + key}

    def rows(
        self,
        table: str,
        *,
        columns: Sequence[str],
        order: str,
        filters: Sequence[tuple[str, str, str]] = (),
        limit: Optional[int] = None,
    ) -> list[Dict[str, Any]]:
        result: list[Dict[str, Any]] = []
        offset = 0
        while limit is None or len(result) < limit:
            page_limit = min(_PAGE_SIZE, (limit or _PAGE_SIZE) - len(result))
            query: list[tuple[str, str]] = [
                ("select", ",".join(columns)),
                ("order", order),
                ("limit", str(page_limit)),
                ("offset", str(offset)),
            ]
            query.extend((field, "%s.%s" % (operator, value)) for field, operator, value in filters)
            request = Request(
                self._base + table + "?" + urlencode(query),
                headers=self._headers,
            )
            try:
                with urlopen(request, timeout=30) as response:
                    page = json.load(response)
            except Exception as exc:
                raise AcceptanceCorpusBootstrapV2Error(
                    "historical query failed for %s" % table
                ) from exc
            if not isinstance(page, list) or any(not isinstance(row, Mapping) for row in page):
                raise AcceptanceCorpusBootstrapV2Error(
                    "historical query returned invalid rows for %s" % table
                )
            result.extend(dict(row) for row in page)
            if len(page) < page_limit:
                break
            offset += len(page)
        return result


def _promotion_input(score_doc: Mapping[str, Any]) -> Dict[str, Any]:
    aggregates = score_doc.get("aggregates")
    gate = score_doc.get("private_holdout_gate")
    safe_aggregates: Dict[str, Any] = {}
    if isinstance(aggregates, Mapping):
        for name in ("mean_delta", "provider_excluded_icp_ids"):
            if name in aggregates:
                safe_aggregates[name] = aggregates[name]
    safe: Dict[str, Any] = {"aggregates": safe_aggregates}
    if isinstance(gate, Mapping):
        allowed = {
            "decision",
            "private_holdout_evaluated",
            "conditional_validation_required",
            "conditional_holdout_evaluated",
            "baseline_aggregate_score",
            "candidate_total_score",
            "candidate_delta_vs_daily_baseline",
        }
        safe["private_holdout_gate"] = {
            name: gate[name] for name in sorted(allowed) if name in gate
        }
    _reject_secret_markers(safe)
    return safe


def _write_fixture(
    *,
    root: Path,
    kind: str,
    fixture_id: str,
    captured_at: str,
    artifact: Mapping[str, Any],
    expected_output: Any,
    receipt_root: Any,
    metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    _reject_secret_markers(artifact)
    relative = Path(kind) / (hashlib.sha256(fixture_id.encode("utf-8")).hexdigest()[:24] + ".json")
    destination = root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical(artifact) + b"\n"
    destination.write_bytes(payload)
    destination.chmod(0o600)
    return {
        "kind": kind,
        "fixture_id": fixture_id,
        "captured_at": _timestamp(captured_at),
        "artifact_path": relative.as_posix(),
        "expected_output_hash": _hash(expected_output),
        "receipt_root": _normalized_hash(receipt_root, fallback=artifact),
        "metadata": dict(metadata),
    }


def _require(rows: Sequence[Mapping[str, Any]], count: int, label: str) -> None:
    if len(rows) < count:
        raise AcceptanceCorpusBootstrapV2Error(
            "production history has only %d %s rows; %d required"
            % (len(rows), label, count)
        )


def build_fixture_index(reader: PostgrestHistoryReader, *, corpus_root: Path) -> list[Dict[str, Any]]:
    root = Path(corpus_root)
    if root.exists():
        raise AcceptanceCorpusBootstrapV2Error("acceptance corpus root already exists")
    root.mkdir(parents=True, mode=0o700)
    fixtures: list[Dict[str, Any]] = []

    scores = reader.rows(
        "research_evaluation_score_bundles",
        columns=("score_bundle_id", "created_at", "score_bundle_hash", "anchored_hash", "score_bundle_doc"),
        order="created_at.asc,score_bundle_id.asc",
        limit=500,
    )
    _require(scores, 100, "score-bundle")
    for row in scores[-100:]:
        doc = _promotion_input(row.get("score_bundle_doc") or {})
        artifact = {
            "schema_version": "leadpoet.v2_acceptance_score_bundle_fixture.v1",
            "score_bundle_id": str(row.get("score_bundle_id") or ""),
            "score_bundle_hash": _normalized_hash(row.get("score_bundle_hash"), fallback=doc),
            "promotion_input": doc,
        }
        fixtures.append(_write_fixture(
            root=root, kind="score_bundle",
            fixture_id="score_bundle:%s" % artifact["score_bundle_id"],
            captured_at=row["created_at"], artifact=artifact,
            expected_output=artifact["score_bundle_hash"],
            receipt_root=row.get("anchored_hash") or row.get("score_bundle_hash"),
            metadata={},
        ))

    benchmarks = reader.rows(
        "research_lab_private_model_benchmark_bundles",
        columns=("benchmark_bundle_id", "benchmark_date", "created_at", "benchmark_bundle_hash", "anchored_hash", "aggregate_score", "benchmark_quality", "evaluation_epoch"),
        order="benchmark_date.desc,created_at.desc",
        limit=200,
    )
    by_date: Dict[str, Mapping[str, Any]] = {}
    for row in benchmarks:
        by_date.setdefault(str(row.get("benchmark_date") or ""), row)
    selected_benchmarks = [by_date[date] for date in sorted(by_date) if date][-14:]
    _require(selected_benchmarks, 14, "unique benchmark-date")
    for row in selected_benchmarks:
        artifact = {name: row.get(name) for name in (
            "benchmark_bundle_id", "benchmark_date", "benchmark_bundle_hash",
            "aggregate_score", "benchmark_quality", "evaluation_epoch",
        )}
        fixtures.append(_write_fixture(
            root=root, kind="daily_benchmark",
            fixture_id="daily_benchmark:%s" % row["benchmark_bundle_id"],
            captured_at=row["created_at"], artifact=artifact,
            expected_output=row.get("benchmark_bundle_hash") or artifact,
            receipt_root=row.get("anchored_hash") or row.get("benchmark_bundle_hash"),
            metadata={"benchmark_date": str(row["benchmark_date"])},
        ))

    usable = next((row for row in reversed(scores) if "private_holdout_gate" not in _promotion_input(row.get("score_bundle_doc") or {})), scores[-1])
    rejected = next(
        (
            row
            for row in reversed(scores)
            if promotion_gate_decision(
                _promotion_input(row.get("score_bundle_doc") or {}),
                candidate_kind="image_build",
                candidate_parent="sha256:historical-parent",
                active_parent="sha256:historical-parent",
                threshold_points=-1_000_000.0,
                auto_promotion_enabled=True,
            ).status
            == "rejected_basis_unavailable"
        ),
        None,
    )
    base_input = _promotion_input(usable.get("score_bundle_doc") or {})
    rejected_input = _promotion_input((rejected or usable).get("score_bundle_doc") or {})
    if rejected is None:
        rejected_input = {
            "aggregates": base_input.get("aggregates", {}),
            "private_holdout_gate": {
                "decision": "private_holdout_rejected",
                "private_holdout_evaluated": True,
            },
        }
    branch_inputs = {
        "disabled": (base_input, {"auto_promotion_enabled": False}),
        "rejected_legacy_patch_candidate": (base_input, {"candidate_kind": "patch"}),
        "rejected_basis_unavailable": (rejected_input, {}),
        "rejected_below_threshold": (base_input, {"threshold_points": 1_000_000.0}),
        "stale_parent_needs_rescore": (base_input, {"active_parent": "sha256:different-parent"}),
        "promotion_passed": (base_input, {"threshold_points": -1_000_000.0}),
    }
    captured_at = usable["created_at"]
    source_root = usable.get("anchored_hash") or usable.get("score_bundle_hash")
    for expected_status in sorted(REQUIRED_PROMOTION_BRANCHES):
        score_input, overrides = branch_inputs[expected_status]
        arguments = {
            "candidate_kind": "image_build",
            "candidate_parent": "sha256:historical-parent",
            "active_parent": "sha256:historical-parent",
            "threshold_points": -1_000_000.0,
            "auto_promotion_enabled": True,
            **overrides,
        }
        decision = promotion_gate_decision(score_input, **arguments).to_dict()
        if decision["status"] != expected_status:
            raise AcceptanceCorpusBootstrapV2Error(
                "authoritative promotion replay did not reach %s" % expected_status
            )
        artifact = {
            "schema_version": "leadpoet.v2_acceptance_promotion_replay.v1",
            "source_score_bundle_hash": _normalized_hash(usable.get("score_bundle_hash"), fallback=score_input),
            "score_bundle": score_input,
            "arguments": arguments,
            "expected_decision": decision,
        }
        fixtures.append(_write_fixture(
            root=root, kind="promotion_branch",
            fixture_id="promotion_branch:%s" % expected_status,
            captured_at=captured_at, artifact=artifact,
            expected_output=decision, receipt_root=source_root,
            metadata={"status": expected_status},
        ))

    loops = reader.rows(
        "research_lab_auto_research_loop_events",
        columns=("event_id", "run_id", "created_at", "anchored_hash", "candidate_artifact_hash", "candidate_patch_hash", "event_type", "loop_status"),
        order="created_at.desc",
        filters=(("event_type", "eq", "loop_completed"),),
        limit=1,
    )
    _require(loops, 1, "completed autoresearch")
    row = loops[0]
    artifact = {name: row.get(name) for name in ("event_id", "run_id", "event_type", "loop_status", "candidate_artifact_hash", "candidate_patch_hash")}
    fixtures.append(_write_fixture(root=root, kind="autoresearch_run", fixture_id="autoresearch_run:%s" % row["run_id"], captured_at=row["created_at"], artifact=artifact, expected_output=artifact, receipt_root=row.get("anchored_hash"), metadata={}))

    providers = reader.rows(
        "research_lab_provider_usage_ledger",
        columns=("usage_row_id", "recorded_at", "request_fingerprint", "provider_id", "endpoint_class", "status", "est_cost_microusd"),
        order="recorded_at.desc",
        limit=1,
    )
    _require(providers, 1, "provider-tape")
    row = providers[0]
    artifact = dict(row)
    fixtures.append(_write_fixture(root=root, kind="provider_tape", fixture_id="provider_tape:%s" % row["usage_row_id"], captured_at=row["recorded_at"], artifact=artifact, expected_output=artifact, receipt_root=row.get("request_fingerprint"), metadata={}))

    allocations = reader.rows(
        "research_lab_emission_allocation_snapshots",
        columns=("allocation_id", "created_at", "allocation_hash", "input_hash", "epoch", "netuid", "snapshot_status", "champion_alpha_percent", "reimbursement_alpha_percent", "source_add_alpha_percent", "unallocated_alpha_percent"),
        order="created_at.desc",
        limit=1,
    )
    _require(allocations, 1, "reward-allocation")
    row = allocations[0]
    artifact = dict(row)
    fixtures.append(_write_fixture(root=root, kind="reward_allocation", fixture_id="reward_allocation:%s" % row["allocation_id"], captured_at=row["created_at"], artifact=artifact, expected_output=row.get("allocation_hash") or artifact, receipt_root=row.get("input_hash") or row.get("allocation_hash"), metadata={}))

    weights = reader.rows(
        "research_lab_signed_audit_bundles",
        columns=("audit_bundle_id", "created_at", "epoch", "audit_bundle_hash", "anchored_hash", "schema_version"),
        order="epoch.desc,created_at.desc",
        limit=500,
    )
    by_epoch: Dict[int, Mapping[str, Any]] = {}
    for row in weights:
        epoch = row.get("epoch")
        if isinstance(epoch, int) and not isinstance(epoch, bool):
            by_epoch.setdefault(epoch, row)
    selected_weights = [by_epoch[epoch] for epoch in sorted(by_epoch)[-50:]]
    _require(selected_weights, 50, "unique weight-epoch")
    for row in selected_weights:
        artifact = dict(row)
        fixtures.append(_write_fixture(root=root, kind="weight_epoch", fixture_id="weight_epoch:%s" % row["epoch"], captured_at=row["created_at"], artifact=artifact, expected_output=row.get("audit_bundle_hash") or artifact, receipt_root=row.get("anchored_hash") or row.get("audit_bundle_hash"), metadata={"epoch_id": row["epoch"]}))

    return fixtures


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--supabase-url", default=os.environ.get("SUPABASE_URL", ""))
    parser.add_argument("--service-role-key", default=os.environ.get("SUPABASE_SERVICE_ROLE_KEY", ""))
    parser.add_argument("--corpus-root", required=True, type=Path)
    parser.add_argument("--fixture-index", required=True, type=Path)
    parser.add_argument("--captured-from", required=True)
    parser.add_argument("--captured-through", required=True)
    parser.add_argument("--signing-key", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args(argv)
    fixtures = build_fixture_index(
        PostgrestHistoryReader(url=args.supabase_url, service_role_key=args.service_role_key),
        corpus_root=args.corpus_root,
    )
    args.fixture_index.parent.mkdir(parents=True, exist_ok=True)
    args.fixture_index.write_bytes(_canonical(fixtures) + b"\n")
    args.fixture_index.chmod(0o600)
    from gateway.tee.acceptance_corpus_v2 import _load_private_key
    key = _load_private_key(args.signing_key)
    manifest = build_acceptance_corpus_from_index_v2(
        fixture_index=fixtures,
        corpus_root=args.corpus_root,
        captured_from=args.captured_from,
        captured_through=args.captured_through,
        signing_key=key,
    )
    signer_hash = _hash(bytes.fromhex(manifest["signing_pubkey_hex"]))
    validate_acceptance_corpus_v2(manifest, corpus_root=args.corpus_root, expected_signing_pubkey_hash=signer_hash)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, sort_keys=True, indent=2)
        handle.write("\n")
    args.manifest.chmod(0o600)
    print(json.dumps({"fixture_count": len(fixtures), "manifest": str(args.manifest), "signer_hash": signer_hash}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
