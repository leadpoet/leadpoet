#!/usr/bin/env python3
"""Create a read-only Research Lab gateway loop monitoring report."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from typing import Any
from urllib import parse, request


DEFAULT_SSH_TARGET = "ec2-user@52.91.135.79"
DEFAULT_SSH_KEY = "~/Downloads/leadpoet-gateway-tee-main.pem"
DEFAULT_GATEWAY_ROOT = "/home/ec2-user/gateway"
DEFAULT_OUTPUT_DIR = "/Users/pranav/Downloads"

SECRET_PATTERNS = (
    (re.compile(r"sk-or-v1-[A-Za-z0-9_-]{16,}"), "[redacted-openrouter-key]"),
    (re.compile(r"sb_secret_[A-Za-z0-9_-]+"), "[redacted-supabase-service-key]"),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "[redacted-aws-access-key]"),
    (
        re.compile(r"(?i)([A-Z0-9_]*(?:SECRET|API_KEY|TOKEN|PASSWORD)[A-Z0-9_]*\s*[=:]\s*)[^\s'\"`]+"),
        r"\1[redacted-secret-value]",
    ),
    (
        re.compile(r"(?i)([?&](?:signature|token|api_key|apikey|key)=)[^&\s\"']+"),
        r"\1[redacted-query-secret]",
    ),
    (
        re.compile(r"(?i)(aws_secret_access_key\s*[=:]\s*)[A-Za-z0-9/+=]{24,}"),
        r"\1[redacted-aws-secret-key]",
    ),
    (
        re.compile(r"(?i)(service_role_key\s*[=:]\s*)[A-Za-z0-9._-]{20,}"),
        r"\1[redacted-service-role-key]",
    ),
    (re.compile(r"://([^/\s:@]+):([^/@\s]+)@"), "://[redacted-credentials]@"),
    (
        re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S),
        "[redacted-private-key]",
    ),
    (
        re.compile(r"\b\d{12}\.dkr\.ecr\.[A-Za-z0-9.-]+\.amazonaws\.com/[^\s'\"<>]+"),
        "[redacted-ecr-uri]",
    ),
    (re.compile(r"\b[\w./:-]+@sha256:[0-9a-f]{64}\b"), "[redacted-image-digest]"),
)

REPORT_SECRET_RE = re.compile(
    r"sk-or-v1-|sb_secret_|AKIA[0-9A-Z]{16}|BEGIN [A-Z ]*PRIVATE KEY|://[^/\s:@]+:[^/@\s]+@"
    r"|(?i:[A-Z0-9_]*(?:SECRET|API_KEY|TOKEN|PASSWORD)[A-Z0-9_]*\s*[=:]\s*(?!\[redacted-)[^\s'\"`]+)",
    re.I,
)

LOG_PATTERNS: tuple[tuple[str, str, re.Pattern[str]], ...] = (
    (
        "openrouter_credit",
        "OpenRouter credit or payment block",
        re.compile(r"openrouter.*(402|insufficient|credit|balance|payment required)|blocked_for_credit", re.I),
    ),
    ("exa_429", "Exa rate limiting", re.compile(r"\bexa\b.*(429|rate limit|too many requests)", re.I)),
    (
        "docker_ecr_build",
        "Docker/ECR/image build failure",
        re.compile(r"\b(docker|ecr|overlay2|buildkit)\b|image build|docker cp|docker pull|no space left", re.I),
    ),
    (
        "kms_s3",
        "KMS/S3 artifact failure",
        re.compile(r"\bkms\b|accessdeniedexception|access denied|s3://|putobject|getobject|signaturedoesnotmatch", re.I),
    ),
    (
        "stale_parent",
        "Stale-parent rebase/rescore",
        re.compile(r"stale_parent|parent.*changed|rebase_queued|needs_rescore", re.I),
    ),
    ("baseline_wait", "Baseline not ready", re.compile(r"baseline_not_ready|matching_completed_private_baseline", re.I)),
    (
        "patch_apply",
        "Patch apply/repair/build issue",
        re.compile(r"patch.*(apply|repair|failed|invalid)|git apply|candidate_patch|malformed|corrupt", re.I),
    ),
    (
        "loop_projection",
        "Loop projection or event sequence issue",
        re.compile(r"terminal_loop_projection|seq conflict|duplicate key|loop_current|projection", re.I),
    ),
    ("traceback", "Python traceback/error", re.compile(r"traceback|exception|error:", re.I)),
)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.self_test_redaction:
        return run_redaction_self_test()

    output = Path(args.output).expanduser() if args.output else default_output_path()
    output.parent.mkdir(parents=True, exist_ok=True)

    log_result = collect_logs(args)
    db_result = collect_supabase(args)
    analysis = analyze(log_result, db_result, args)
    report = render_report(log_result, db_result, analysis, args)
    report = redact(report)
    if REPORT_SECRET_RE.search(report):
        raise SystemExit("refusing to write report: redaction self-check found secret-like material")
    output.write_text(report, encoding="utf-8")
    print(str(output))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssh-target", default=DEFAULT_SSH_TARGET)
    parser.add_argument("--ssh-key", default=DEFAULT_SSH_KEY)
    parser.add_argument("--gateway-root", default=DEFAULT_GATEWAY_ROOT)
    parser.add_argument("--output", default="")
    parser.add_argument("--since-hours", type=int, default=24)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--ticket-id", default="")
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--log-lines", type=int, default=6000)
    parser.add_argument("--journal-lines", type=int, default=800)
    parser.add_argument("--self-test-redaction", action="store_true")
    return parser


def default_output_path() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path(DEFAULT_OUTPUT_DIR) / f"loop_monitoring_report_{stamp}.md"


def collect_logs(args: argparse.Namespace) -> dict[str, Any]:
    if args.local_only:
        return collect_local_logs(Path(args.gateway_root), args.log_lines)
    key_path = str(Path(args.ssh_key).expanduser())
    remote_script = remote_log_script(args.gateway_root, args.log_lines, args.journal_lines)
    cmd = [
        "ssh",
        "-i",
        key_path,
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        args.ssh_target,
        remote_script,
    ]
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=90, check=False)
    except Exception as exc:
        return {
            "ok": False,
            "source": "ssh",
            "error": redact(str(exc)),
            "text": "",
            "lines": [],
            "command": redact(" ".join(shlex.quote(part) for part in cmd[:5] + ["..."])),
        }
    text = redact((proc.stdout or "") + ("\nSTDERR:\n" + proc.stderr if proc.stderr else ""))
    return {
        "ok": proc.returncode == 0,
        "source": "ssh",
        "returncode": proc.returncode,
        "error": redact(proc.stderr[:1000]) if proc.stderr else "",
        "text": text,
        "lines": [line for line in text.splitlines() if line.strip()],
    }


def remote_log_script(gateway_root: str, log_lines: int, journal_lines: int) -> str:
    root = shlex.quote(gateway_root)
    log_lines = max(100, min(int(log_lines), 50000))
    journal_lines = max(0, min(int(journal_lines), 5000))
    return f"""set -e
ROOT={root}
echo '### remote_host'
hostname || true
echo '### gateway_root'
printf '%s\\n' "$ROOT"
for f in "$ROOT/gateway.log" "$ROOT/nohup.out"; do
  if [ -f "$f" ]; then
    echo "### file:$f"
    stat -c 'size=%s mtime=%y path=%n' "$f" 2>/dev/null || ls -l "$f"
    tail -n {log_lines} "$f"
  else
    echo "### missing:$f"
  fi
done
if command -v journalctl >/dev/null 2>&1; then
  echo '### journalctl'
  journalctl --no-pager -n {journal_lines} 2>/dev/null | grep -Eai 'research_lab|Research Lab|OpenRouter|Exa|docker|ECR|KMS|candidate|autoresearch|gateway' || true
fi
"""


def collect_local_logs(root: Path, log_lines: int) -> dict[str, Any]:
    chunks: list[str] = []
    for path in (root / "gateway.log", root / "nohup.out"):
        if not path.exists():
            chunks.append(f"### missing:{path}")
            continue
        chunks.append(f"### file:{path}")
        chunks.append(f"size={path.stat().st_size} mtime={datetime.fromtimestamp(path.stat().st_mtime).isoformat()}")
        chunks.extend(path.read_text(encoding="utf-8", errors="replace").splitlines()[-max(100, log_lines) :])
    text = redact("\n".join(chunks))
    return {"ok": True, "source": "local", "text": text, "lines": [line for line in text.splitlines() if line.strip()]}


def collect_supabase(args: argparse.Namespace) -> dict[str, Any]:
    url = os.getenv("SUPABASE_URL", "").rstrip("/")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
    if not url or not key:
        return {"ok": False, "available": False, "error": "SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY not set"}
    since = (datetime.now(timezone.utc) - timedelta(hours=max(1, int(args.since_hours)))).isoformat()
    client = SupabaseReader(url, key)
    tables: dict[str, list[dict[str, Any]]] = {}
    errors: dict[str, str] = {}
    queries = {
        "queue": (
            "research_loop_run_queue_current",
            {
                "select": "run_id,ticket_id,current_queue_status,current_reason,current_status_at,worker_ref",
                "order": "current_status_at.desc",
                "limit": "1000",
            },
        ),
        "loops": (
            "research_lab_auto_research_loop_current",
            {
                "select": "run_id,ticket_id,current_loop_status,current_event_type,current_status_at",
                "order": "current_status_at.desc",
                "limit": "1000",
            },
        ),
        "candidates": (
            "research_lab_candidate_evaluation_current",
            {
                "select": "candidate_id,run_id,ticket_id,current_candidate_status,current_reason,current_status_at,miner_hotkey",
                "order": "current_status_at.desc",
                "limit": "1000",
            },
        ),
        "score_bundles": (
            "research_evaluation_score_bundle_current",
            {
                "select": "score_bundle_id,candidate_id,run_id,evaluation_epoch,current_status_at,created_at,score_bundle_doc",
                "order": "current_status_at.desc",
                "limit": "200",
            },
        ),
        "promotions": (
            "research_lab_candidate_promotion_events",
            {
                "select": "candidate_id,derived_candidate_id,event_type,promotion_status,created_at",
                "created_at": f"gte.{since}",
                "order": "created_at.desc",
                "limit": "500",
            },
        ),
        "public_cards": (
            "research_lab_public_loop_card_current",
            {
                "select": "ticket_id,current_run_id,current_outcome_label,current_outcome_band,current_candidate_count,current_scored_candidate_count,current_last_activity_at",
                "order": "current_last_activity_at.desc",
                "limit": "1000",
            },
        ),
        "arweave": (
            "research_lab_arweave_epoch_audit_anchor_current",
            {
                "select": "epoch,audit_kind,current_anchor_status,arweave_tx_id,current_status_at",
                "order": "current_status_at.desc",
                "limit": "100",
            },
        ),
    }
    for name, (table, params) in queries.items():
        try:
            rows = client.get(table, params)
            tables[name] = rows
        except Exception as exc:
            errors[name] = redact(str(exc)[:500])
            tables[name] = []
    return {"ok": not errors, "available": True, "tables": tables, "errors": errors}


class SupabaseReader:
    def __init__(self, base_url: str, service_role_key: str):
        self.base_url = base_url.rstrip("/") + "/rest/v1/"
        self.headers = {
            "apikey": service_role_key,
            "Authorization": f"Bearer {service_role_key}",
            "Accept": "application/json",
        }

    def get(self, table: str, params: dict[str, str]) -> list[dict[str, Any]]:
        query = parse.urlencode(params, doseq=True, safe="*,.:->")
        req = request.Request(self.base_url + table + "?" + query, headers=self.headers)
        with request.urlopen(req, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))


def analyze(log_result: dict[str, Any], db_result: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    issues: dict[str, dict[str, Any]] = {}
    evidence_rows: list[dict[str, str]] = []

    def add_issue(
        key: str,
        title: str,
        classification: str,
        severity: str,
        evidence: str,
        recommendation: str,
    ) -> None:
        item = issues.setdefault(
            key,
            {
                "key": key,
                "title": title,
                "classification": classification,
                "severity": severity,
                "evidence": [],
                "recommendation": recommendation,
            },
        )
        if evidence and evidence not in item["evidence"]:
            item["evidence"].append(evidence)
        evidence_rows.append({"issue": title, "severity": severity, "evidence": evidence})

    log_matches = analyze_log_patterns(log_result.get("lines") or [])
    for key, matches in log_matches.items():
        if not matches:
            continue
        title = pattern_title(key)
        severity = {
            "docker_ecr_build": "critical",
            "openrouter_credit": "high",
            "exa_429": "high",
            "traceback": "medium",
        }.get(key, "medium")
        classification = "infra_blocker" if key in {"docker_ecr_build", "openrouter_credit", "exa_429", "kms_s3"} else "likely_bug"
        add_issue(
            key,
            title,
            classification,
            severity,
            matches[0],
            recommendation_for_pattern(key),
        )

    db_summary = analyze_db(db_result, add_issue)
    return {
        "issues": sorted(issues.values(), key=lambda item: severity_rank(item["severity"])),
        "evidence_rows": evidence_rows[:80],
        "log_matches": log_matches,
        "db_summary": db_summary,
    }


def analyze_log_patterns(lines: list[str]) -> dict[str, list[str]]:
    matches: dict[str, list[str]] = {key: [] for key, _title, _pattern in LOG_PATTERNS}
    for raw_line in lines:
        line = redact(raw_line.strip())
        if not line:
            continue
        for key, _title, pattern in LOG_PATTERNS:
            if pattern.search(line) and len(matches[key]) < 12:
                matches[key].append(line[:500])
    return matches


def analyze_db(db_result: dict[str, Any], add_issue: Any) -> dict[str, Any]:
    if not db_result.get("available"):
        return {"available": False, "error": db_result.get("error")}
    tables = db_result.get("tables") or {}
    queue = tables.get("queue", [])
    loops = tables.get("loops", [])
    candidates = tables.get("candidates", [])
    score_bundles = tables.get("score_bundles", [])
    promotions = tables.get("promotions", [])
    public_cards = tables.get("public_cards", [])
    arweave = tables.get("arweave", [])

    queue_by_run = {str(row.get("run_id") or ""): row for row in queue if row.get("run_id")}
    mismatches = []
    for loop in loops:
        run_id = str(loop.get("run_id") or "")
        qrow = queue_by_run.get(run_id)
        if not qrow:
            continue
        queue_status = str(qrow.get("current_queue_status") or "")
        loop_status = str(loop.get("current_loop_status") or "")
        if queue_status in {"completed", "failed"} and loop_status not in {"completed", "failed"}:
            mismatches.append((run_id, queue_status, loop_status, str(loop.get("current_event_type") or "")))
    if mismatches:
        add_issue(
            "terminal_queue_nonterminal_loop",
            "Terminal queue rows have nonterminal loop projections",
            "confirmed_bug",
            "high",
            f"{len(mismatches)} mismatch(es), e.g. run={compact(mismatches[0][0])} queue={mismatches[0][1]} loop={mismatches[0][2]} event={mismatches[0][3]}",
            "Run the terminal projection reconciler dry-run, then append loop_completed/loop_failed repairs if correct.",
        )

    candidate_reason_counts = counts(candidates, "current_reason")
    stale_parent = [
        row
        for row in candidates
        if row.get("current_candidate_status") == "rejected"
        and row.get("current_reason") == "stale_parent_needs_rescore"
    ]
    if stale_parent:
        add_issue(
            "stale_parent_backlog",
            "Stale-parent candidates need rebase/rescore",
            "operator_action",
            "high",
            f"{len(stale_parent)} rejected candidate(s), e.g. candidate={compact(stale_parent[0].get('candidate_id'))}",
            "Run stale-parent rebase dry-run in small batches and verify derived candidates are queued.",
        )

    baseline_wait = [
        row
        for row in candidates
        if row.get("current_candidate_status") == "queued" and row.get("current_reason") == "baseline_not_ready"
    ]
    if baseline_wait:
        add_issue(
            "baseline_not_ready",
            "Candidates are waiting for private baseline readiness",
            "operator_action",
            "medium",
            f"{len(baseline_wait)} queued candidate(s), e.g. candidate={compact(baseline_wait[0].get('candidate_id'))}",
            "Display as waiting_for_baseline; requeue only after matching baseline exists.",
        )

    credit_blocked = [
        row
        for row in queue
        if row.get("current_queue_status") == "paused" and row.get("current_reason") == "blocked_for_credit"
    ]
    if credit_blocked:
        add_issue(
            "credit_blocked_queue",
            "Runs are blocked on miner OpenRouter credit",
            "infra_blocker",
            "high",
            f"{len(credit_blocked)} paused run(s), e.g. run={compact(credit_blocked[0].get('run_id'))}",
            "Do not generic-resume. Use explicit credit-blocked resume only after OpenRouter key preflight passes.",
        )

    active_scoring = [
        row for row in candidates if str(row.get("current_candidate_status") or "") in {"assigned", "evaluating"}
    ]
    public_no_score = [
        row
        for row in public_cards
        if str(row.get("current_outcome_label") or "") == "candidate_generation_complete"
        and int(row.get("current_candidate_count") or 0) > int(row.get("current_scored_candidate_count") or 0)
    ]
    if public_no_score:
        add_issue(
            "candidate_generation_complete_no_score",
            "Public cards show generated candidates with no score",
            "likely_bug",
            "medium",
            f"{len(public_no_score)} card(s), e.g. ticket={compact(public_no_score[0].get('ticket_id'))}",
            "Classify by candidate reason: scoring, waiting_for_baseline, needs_rescore, or failed.",
        )

    return {
        "available": True,
        "errors": db_result.get("errors") or {},
        "queue_status_counts": counts(queue, "current_queue_status"),
        "queue_reason_counts": counts(queue, "current_reason"),
        "loop_status_counts": counts(loops, "current_loop_status"),
        "candidate_status_counts": counts(candidates, "current_candidate_status"),
        "candidate_reason_counts": candidate_reason_counts,
        "promotion_status_counts": counts(promotions, "promotion_status"),
        "public_outcome_counts": counts(public_cards, "current_outcome_label"),
        "arweave_anchor_status_counts": counts(arweave, "current_anchor_status"),
        "terminal_queue_nonterminal_loop_mismatches": len(mismatches),
        "stale_parent_backlog": len(stale_parent),
        "baseline_not_ready_queued": len(baseline_wait),
        "credit_blocked_paused": len(credit_blocked),
        "active_scoring": len(active_scoring),
        "recent_score_bundle_count": len(score_bundles),
        "candidate_generation_complete_no_score_cards": len(public_no_score),
    }


def render_report(
    log_result: dict[str, Any],
    db_result: dict[str, Any],
    analysis: dict[str, Any],
    args: argparse.Namespace,
) -> str:
    generated_at = datetime.now(timezone.utc).isoformat()
    db_summary = analysis["db_summary"]
    issues = analysis["issues"]
    confirmed = [item for item in issues if item["classification"] == "confirmed_bug"]
    potential = [item for item in issues if item["classification"] != "confirmed_bug"]
    lines: list[str] = []
    lines.append("# Research Lab Loop Monitoring Report")
    lines.append("")
    lines.append(f"Generated: `{generated_at}`")
    lines.append("")
    lines.append("Data sources:")
    lines.append(f"- Gateway logs: `{args.gateway_root}` via `{('local' if args.local_only else args.ssh_target)}`")
    lines.append(f"- Supabase read-only: `{'available' if db_result.get('available') else 'not available'}`")
    lines.append(f"- Window hint: last `{args.since_hours}` hours where timestamps are available")
    if args.run_id:
        lines.append(f"- Focus run_id: `{redact(args.run_id)}`")
    if args.ticket_id:
        lines.append(f"- Focus ticket_id: `{redact(args.ticket_id)}`")
    lines.append("")

    lines.append("## Executive Summary")
    if issues:
        lines.append(f"- Issues found: `{len(issues)}`")
        lines.append(f"- Confirmed bugs: `{len(confirmed)}`")
        lines.append(f"- Potential/operator/infra issues: `{len(potential)}`")
        top = ", ".join(f"{item['severity']}:{item['key']}" for item in issues[:5])
        lines.append(f"- Highest priority: {top}")
    else:
        lines.append("- No high-signal issues found in the sampled logs/current views.")
    if not db_result.get("available"):
        lines.append(f"- Supabase evidence unavailable: `{db_result.get('error')}`")
    if not log_result.get("ok"):
        lines.append(f"- Gateway log collection had errors: `{log_result.get('error')}`")
    lines.append("")

    lines.append("## Production Flow Health")
    if db_summary.get("available"):
        for key in (
            "queue_status_counts",
            "loop_status_counts",
            "candidate_status_counts",
            "candidate_reason_counts",
            "public_outcome_counts",
            "promotion_status_counts",
            "arweave_anchor_status_counts",
        ):
            lines.append(f"- `{key}`: `{json.dumps(db_summary.get(key, {}), sort_keys=True)}`")
        for key in (
            "terminal_queue_nonterminal_loop_mismatches",
            "stale_parent_backlog",
            "baseline_not_ready_queued",
            "credit_blocked_paused",
            "active_scoring",
            "recent_score_bundle_count",
            "candidate_generation_complete_no_score_cards",
        ):
            lines.append(f"- `{key}`: `{db_summary.get(key, 0)}`")
    else:
        lines.append("- DB/current-view health skipped because Supabase env was not present.")
    lines.append("")

    lines.append("## Confirmed Bugs")
    append_issue_section(lines, confirmed)
    lines.append("")
    lines.append("## Potential Issues")
    append_issue_section(lines, potential)
    lines.append("")

    lines.append("## Evidence Table")
    evidence_rows = analysis.get("evidence_rows") or []
    if evidence_rows:
        lines.append("| Issue | Severity | Evidence |")
        lines.append("|---|---|---|")
        for row in evidence_rows[:50]:
            lines.append(
                f"| {md(row['issue'])} | {md(row['severity'])} | {md(row['evidence'])} |"
            )
    else:
        lines.append("- No evidence rows emitted.")
    lines.append("")

    lines.append("## Log Pattern Samples")
    for key, matches in (analysis.get("log_matches") or {}).items():
        if not matches:
            continue
        lines.append(f"### {pattern_title(key)}")
        for sample in matches[:8]:
            lines.append(f"- `{md(sample)}`")
        lines.append("")

    lines.append("## Recommended Fixes")
    if issues:
        for item in issues:
            lines.append(f"- `{item['severity']}` `{item['classification']}` {item['title']}: {item['recommendation']}")
    else:
        lines.append("- Keep monitoring; no immediate code or operator fix identified from this sample.")
    lines.append("")

    lines.append("## Do Not Resume Blindly")
    lines.append("- Do not resume `blocked_for_credit` rows without explicit OpenRouter key preflight.")
    lines.append("- Do not requeue stale-parent candidates directly; rebase to current parent first.")
    lines.append("- Do not treat `candidate_generation_complete` as scored. Verify candidate and score bundle rows.")
    lines.append("- Do not infer model improvement without a score bundle or promotion event.")
    lines.append("")

    lines.append("## Operator Commands To Run Next")
    lines.append("Read-only first:")
    lines.append("")
    lines.append("```bash")
    lines.append("python3 scripts/check_research_lab_operator_health.py")
    lines.append("python3 -m gateway.research_lab.admin status")
    lines.append("python3 -m gateway.research_lab.admin reconcile-loop-projections --dry-run")
    lines.append("python3 -m gateway.research_lab.admin repair-public-cards --dry-run")
    lines.append("python3 -m gateway.research_lab.admin rebase-stale-candidates --dry-run --limit 5")
    lines.append("```")
    lines.append("")
    lines.append("Only run write commands after reviewing dry-run output.")
    lines.append("")
    return "\n".join(lines) + "\n"


def append_issue_section(lines: list[str], items: list[dict[str, Any]]) -> None:
    if not items:
        lines.append("- None found in this sample.")
        return
    for item in items:
        lines.append(f"### {item['title']}")
        lines.append(f"- Classification: `{item['classification']}`")
        lines.append(f"- Severity: `{item['severity']}`")
        for evidence in item.get("evidence", [])[:5]:
            lines.append(f"- Evidence: `{md(evidence)}`")
        lines.append(f"- Suggested fix: {item['recommendation']}")


def pattern_title(key: str) -> str:
    for item_key, title, _pattern in LOG_PATTERNS:
        if item_key == key:
            return title
    return key.replace("_", " ").title()


def recommendation_for_pattern(key: str) -> str:
    return {
        "openrouter_credit": "Pause/block the run, preserve checkpoint/reimbursement, and require explicit credit preflight before resume.",
        "exa_429": "Confirm EXA_MAX_RPS/backoff is live in the scoring worker runtime and reduce burst concurrency if 429s continue.",
        "docker_ecr_build": "Check disk, Docker daemon health, ECR auth, and candidate image cleanup before resuming builds.",
        "kms_s3": "Classify as infrastructure/auth failure; verify KMS/S3 permissions and retry only after config is corrected.",
        "stale_parent": "Rebase stored diff onto current parent and score derived candidate from scratch.",
        "baseline_wait": "Show waiting_for_baseline and requeue only after matching baseline exists.",
        "patch_apply": "Use patch normalization/repair; reject unsafe or unread-file diffs with diagnostics.",
        "loop_projection": "Run terminal loop projection reconciler; keep append-only event semantics.",
        "traceback": "Inspect adjacent run/candidate ids and classify as infra, provider, validation, or code bug before repair.",
    }.get(key, "Inspect evidence and add a targeted test before changing production behavior.")


def severity_rank(value: str) -> int:
    return {"critical": 0, "high": 1, "medium": 2, "low": 3}.get(value, 4)


def counts(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(Counter(str(row.get(field) or "") for row in rows))


def compact(value: object, limit: int = 12) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return f"{text[:limit]}..."


def redact(value: object) -> str:
    text = str(value)
    for pattern, replacement in SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def md(value: object) -> str:
    text = redact(value).replace("|", "\\|")
    return " ".join(text.split())[:900]


def run_redaction_self_test() -> int:
    sample = "\n".join(
        [
            "OPENROUTER_API_KEY=" + "sk-or-" + "v1-" + "abcdefghijklmnopqrstuvwxyz123456",
            "SUPABASE_SERVICE_ROLE_KEY=" + "sb_" + "secret_" + "abcdefghi",
            "LEADPOET_INTERNAL_SECRET=" + "1e158f11b80c0d6c0ac24111e6ae3f714b21e912855d648dd945759d46e3838a",
            "MINIO_SECRET_KEY=" + "Zr8tQp9xL2vB7mNj",
            "/fulfillment/scoring?signature=abc123&nonce=ok",
            "AWS_ACCESS_KEY_ID=" + "AKIA" + "1234567890ABCDEF",
            "AWS_" + "SECRET_ACCESS_KEY=" + "abc123abc123abc123abc123abc123abc123abc1",
            "http://" + "user:pass@" + "example.com:8080",
            "493765492819.dkr.ecr.us-east-1.amazonaws.com/repo@sha256:"
            + "a" * 64,
            "-----BEGIN " + "PRIVATE KEY-----abc-----END " + "PRIVATE KEY-----",
        ]
    )
    redacted = redact(sample)
    if REPORT_SECRET_RE.search(redacted):
        print(redacted)
        return 1
    print("redaction self-test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
