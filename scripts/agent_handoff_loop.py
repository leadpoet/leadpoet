#!/usr/bin/env python3
"""Bounded Codex -> Claude -> Codex handoff loop for local phase work.

The loop is intentionally local and auditable:

1. Codex implements the phase prompt.
2. Claude reviews the resulting diff and Codex's final message as a read-only
   critic, returning a strict JSON verdict.
3. If Claude requests revisions, the review is fed back to Codex.

No production SQL is applied, no commits are made, and Claude receives the diff
as text rather than write access to the workspace.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional


REVIEW_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["verdict", "summary", "issues", "follow_up_prompt"],
    "properties": {
        "verdict": {"type": "string", "enum": ["approve", "revise"]},
        "summary": {"type": "string"},
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "severity",
                    "title",
                    "body",
                    "file",
                    "line",
                    "must_fix",
                    "fix_prompt",
                ],
                "properties": {
                    "severity": {
                        "type": "string",
                        "enum": ["P0", "P1", "P2", "P3"],
                    },
                    "title": {"type": "string"},
                    "body": {"type": "string"},
                    "file": {"type": "string"},
                    "line": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                    "must_fix": {"type": "boolean"},
                    "fix_prompt": {"type": "string"},
                },
            },
        },
        "follow_up_prompt": {"type": "string"},
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a bounded local Codex/Claude handoff loop."
    )
    parser.add_argument(
        "--phase-prompt",
        required=True,
        type=Path,
        help="Markdown/text file containing the phase implementation prompt.",
    )
    parser.add_argument(
        "--phase-name",
        default="phase",
        help="Short name used for the transcript directory.",
    )
    parser.add_argument(
        "--workspace",
        default=Path.cwd(),
        type=Path,
        help="Repository root passed to Codex.",
    )
    parser.add_argument(
        "--cycles",
        default=2,
        type=int,
        help="Maximum Codex/Claude cycles. Defaults to 2.",
    )
    parser.add_argument(
        "--log-dir",
        default=Path(".agent_handoffs"),
        type=Path,
        help="Directory for prompts, raw outputs, diffs, and JSON reviews.",
    )
    parser.add_argument("--codex-cmd", default="codex", help="Codex CLI command.")
    parser.add_argument("--claude-cmd", default="claude", help="Claude CLI command.")
    parser.add_argument("--codex-model", help="Optional Codex model override.")
    parser.add_argument("--claude-model", help="Optional Claude model override.")
    parser.add_argument(
        "--codex-sandbox",
        default="workspace-write",
        choices=["read-only", "workspace-write", "danger-full-access"],
        help="Sandbox mode for Codex. Defaults to workspace-write.",
    )
    parser.add_argument(
        "--codex-approval",
        default="never",
        choices=["untrusted", "on-failure", "on-request", "never"],
        help="Codex approval policy. Defaults to never for unattended runs.",
    )
    parser.add_argument(
        "--claude-tools",
        default="",
        help='Claude tools list. Default "" disables tools for review-only mode.',
    )
    parser.add_argument(
        "--claude-max-budget-usd",
        help="Optional per-review Claude budget, passed to --max-budget-usd.",
    )
    parser.add_argument(
        "--diff-char-limit",
        default=120_000,
        type=int,
        help="Maximum characters of git diff included in Claude prompt.",
    )
    parser.add_argument(
        "--timeout-seconds",
        default=None,
        type=int,
        help="Optional timeout for each agent command.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write prompts and schema only; do not invoke Codex or Claude.",
    )
    return parser.parse_args()


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip()).strip("-")
    return slug.lower() or "phase"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def require_command(name: str) -> None:
    if shutil.which(name) is None:
        raise SystemExit(f"Required command not found on PATH: {name}")


def run_command(
    command: List[str],
    cwd: Path,
    *,
    input_text: Optional[str] = None,
    timeout_seconds: Optional[int] = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(cwd),
        input=input_text,
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
    )


def truncate_middle(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    head_len = limit // 2
    tail_len = limit - head_len
    omitted = len(text) - limit
    return (
        text[:head_len]
        + f"\n\n[... truncated {omitted} characters from middle ...]\n\n"
        + text[-tail_len:]
    )


def git_output(workspace: Path, args: List[str]) -> str:
    result = run_command(["git", *args], workspace)
    if result.returncode == 0:
        return result.stdout.strip()
    return (result.stdout + result.stderr).strip()


def text_preview(path: Path, limit: int) -> str:
    try:
        data = path.read_bytes()
    except OSError as exc:
        return f"[could not read {path.name}: {exc}]"

    if b"\0" in data[:4096]:
        return f"[binary file omitted: {path.name}]"

    text = data.decode("utf-8", errors="replace")
    return truncate_middle(text, limit)


def capture_untracked_snapshot(
    workspace: Path, untracked_files: List[str], char_limit: int
) -> str:
    chunks: List[str] = []
    remaining = char_limit

    for rel_path in untracked_files:
        if remaining <= 0:
            chunks.append("\n[untracked file snapshot truncated]\n")
            break

        path = workspace / rel_path
        if not path.is_file():
            continue

        header = f"\n--- /dev/null\n+++ b/{rel_path}\n@@ untracked file @@\n"
        body_limit = max(0, remaining - len(header))
        body = text_preview(path, body_limit)
        chunk = header + body
        chunks.append(chunk)
        remaining -= len(chunk)

    return "\n".join(chunks).strip()


def capture_git_snapshot(workspace: Path, run_dir: Path, diff_limit: int) -> Dict[str, str]:
    status = git_output(workspace, ["status", "--short"])
    diff_stat = git_output(workspace, ["diff", "--stat"])
    staged_diff_stat = git_output(workspace, ["diff", "--cached", "--stat"])
    diff_names = git_output(workspace, ["diff", "--name-only"])
    staged_diff_names = git_output(workspace, ["diff", "--cached", "--name-only"])
    untracked_names = git_output(workspace, ["ls-files", "--others", "--exclude-standard"])
    untracked_files = [name for name in untracked_names.splitlines() if name.strip()]
    diff = git_output(workspace, ["diff", "--no-ext-diff", "--"])
    staged_diff = git_output(workspace, ["diff", "--cached", "--no-ext-diff", "--"])
    untracked_snapshot = capture_untracked_snapshot(
        workspace, untracked_files, diff_limit // 2
    )

    write_text(run_dir / "git_status.txt", status + "\n")
    write_text(run_dir / "git_diff_stat.txt", diff_stat + "\n")
    write_text(run_dir / "git_staged_diff_stat.txt", staged_diff_stat + "\n")
    write_text(run_dir / "git_diff_name_only.txt", diff_names + "\n")
    write_text(run_dir / "git_staged_diff_name_only.txt", staged_diff_names + "\n")
    write_text(run_dir / "git_untracked_name_only.txt", untracked_names + "\n")
    write_text(run_dir / "git_diff.patch", diff + "\n")
    write_text(run_dir / "git_staged_diff.patch", staged_diff + "\n")
    write_text(run_dir / "git_untracked_snapshot.patch", untracked_snapshot + "\n")

    return {
        "status": status,
        "diff_stat": "\n".join(
            part for part in (diff_stat, staged_diff_stat) if part
        ),
        "diff_names": "\n".join(
            part for part in (diff_names, staged_diff_names, untracked_names) if part
        ),
        "diff": truncate_middle(
            "\n\n".join(part for part in (diff, staged_diff) if part),
            diff_limit,
        ),
        "untracked_snapshot": truncate_middle(untracked_snapshot, diff_limit // 2),
    }


def build_codex_prompt(
    phase_prompt: str,
    *,
    phase_name: str,
    cycle: int,
    claude_review: Optional[Dict[str, Any]],
) -> str:
    base = f"""
You are Codex running an unattended local implementation cycle for Leadpoet.

Phase name: {phase_name}
Cycle: {cycle}

Operational rules:
- Work only inside the repository workspace.
- Keep the Phase 1 local-only / prod-disabled posture unless the phase prompt
  explicitly says otherwise.
- Do not apply production SQL, start schedulers, send outreach, spend budget,
  run paid production workflows, commit, push, or call Claude.
- Prefer focused changes that match the existing tracker and verifier style.
- Run the narrowest relevant local verification commands you can run without
  secrets or live production access.
- Finish with a concise summary of files changed and verification results.

Primary phase prompt:

{phase_prompt.strip()}
"""
    if claude_review is None:
        return textwrap.dedent(base).strip() + "\n"

    review_text = json.dumps(claude_review, indent=2, sort_keys=True)
    revision = f"""

Claude review from the previous cycle:

```json
{review_text}
```

Apply the must-fix items from Claude's review. Treat P3 items as optional unless
they reveal a real correctness, safety, or verification problem. Do not broaden
scope beyond the original phase prompt.
"""
    return textwrap.dedent(base + revision).strip() + "\n"


def build_claude_prompt(
    phase_prompt: str,
    codex_last_message: str,
    git_snapshot: Dict[str, str],
    *,
    phase_name: str,
    cycle: int,
) -> str:
    return textwrap.dedent(
        f"""
        You are Claude reviewing an unattended Codex implementation pass for
        Leadpoet. Act as a strict read-only code reviewer.

        Phase name: {phase_name}
        Cycle: {cycle}

        Review priorities:
        - Bugs, behavioral regressions, broken contracts, and missing tests.
        - Violations of the repository's Phase 1 local-only / prod-disabled
          policy.
        - Accidental production writes, live workflow enablement, payment,
          scheduler, outreach, public miner, or live champion exposure paths.
        - Schema/fixture/verifier drift against the tracker style.

        Do not request unrelated refactors or cosmetic rewrites. Return only a
        JSON object matching the provided schema. Use verdict "approve" only
        when there are no must-fix issues. Put a complete Codex-ready revision
        prompt in follow_up_prompt when verdict is "revise"; otherwise use an
        empty string.

        Primary phase prompt:

        {phase_prompt.strip()}

        Codex final message:

        {codex_last_message.strip()}

        Current git status:

        {git_snapshot["status"]}

        Current git diff stat:

        {git_snapshot["diff_stat"]}

        Current git diff file list:

        {git_snapshot["diff_names"]}

        Current git diff, possibly truncated:

        ```diff
        {git_snapshot["diff"]}
        ```

        Current untracked text-file snapshot, possibly truncated:

        ```diff
        {git_snapshot["untracked_snapshot"]}
        ```
        """
    ).strip() + "\n"


def run_codex(
    args: argparse.Namespace,
    workspace: Path,
    prompt: str,
    cycle_dir: Path,
) -> subprocess.CompletedProcess[str]:
    output_path = cycle_dir / "codex_last_message.md"
    command = [
        args.codex_cmd,
        "exec",
        "-C",
        str(workspace),
        "--sandbox",
        args.codex_sandbox,
        "--ask-for-approval",
        args.codex_approval,
        "-o",
        str(output_path),
        "-",
    ]
    if args.codex_model:
        command.extend(["--model", args.codex_model])

    result = run_command(
        command,
        workspace,
        input_text=prompt,
        timeout_seconds=args.timeout_seconds,
    )
    write_text(cycle_dir / "codex_stdout.log", result.stdout)
    write_text(cycle_dir / "codex_stderr.log", result.stderr)
    if not output_path.exists():
        write_text(output_path, result.stdout.strip() + "\n")
    return result


def run_claude(
    args: argparse.Namespace,
    workspace: Path,
    prompt: str,
    cycle_dir: Path,
) -> subprocess.CompletedProcess[str]:
    schema_path = cycle_dir / "claude_review_schema.json"
    write_text(schema_path, json.dumps(REVIEW_SCHEMA, indent=2, sort_keys=True) + "\n")

    command = [
        args.claude_cmd,
        "-p",
        "--output-format",
        "json",
        "--json-schema",
        str(schema_path),
        "--permission-mode",
        "plan",
        "--tools",
        args.claude_tools,
    ]
    if args.claude_model:
        command.extend(["--model", args.claude_model])
    if args.claude_max_budget_usd:
        command.extend(["--max-budget-usd", args.claude_max_budget_usd])

    result = run_command(
        command,
        workspace,
        input_text=prompt,
        timeout_seconds=args.timeout_seconds,
    )
    write_text(cycle_dir / "claude_stdout.json", result.stdout)
    write_text(cycle_dir / "claude_stderr.log", result.stderr)
    return result


def json_from_text(text: str) -> Any:
    stripped = text.strip()
    if not stripped:
        raise ValueError("empty output")

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for index, char in enumerate(stripped):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(stripped[index:])
            return value
        except json.JSONDecodeError:
            continue
    raise ValueError("no JSON object found")


def extract_review(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict) and "verdict" in value:
        return value

    if isinstance(value, dict):
        for key in ("result", "content", "message", "text", "output"):
            nested = value.get(key)
            if isinstance(nested, str):
                return extract_review(json_from_text(nested))
            if isinstance(nested, dict):
                return extract_review(nested)
            if isinstance(nested, list):
                combined = "\n".join(
                    item if isinstance(item, str) else json.dumps(item)
                    for item in nested
                )
                return extract_review(json_from_text(combined))

    raise ValueError("Claude output did not contain a review verdict")


def validate_review(review: Dict[str, Any]) -> None:
    if review.get("verdict") not in {"approve", "revise"}:
        raise ValueError("review verdict must be approve or revise")
    if not isinstance(review.get("issues"), list):
        raise ValueError("review issues must be a list")
    if review["verdict"] == "revise" and not review.get("follow_up_prompt"):
        raise ValueError("revise verdict must include follow_up_prompt")


def main() -> int:
    args = parse_args()
    if args.cycles < 1:
        raise SystemExit("--cycles must be at least 1")

    workspace = args.workspace.resolve()
    phase_prompt_path = args.phase_prompt.resolve()
    if not phase_prompt_path.exists():
        raise SystemExit(f"Phase prompt not found: {phase_prompt_path}")

    require_command(args.codex_cmd)
    require_command(args.claude_cmd)

    phase_prompt = phase_prompt_path.read_text(encoding="utf-8")
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = (workspace / args.log_dir / f"{timestamp}-{slugify(args.phase_name)}")
    run_dir.mkdir(parents=True, exist_ok=True)

    write_text(run_dir / "phase_prompt.md", phase_prompt)
    write_text(run_dir / "initial_git_status.txt", git_output(workspace, ["status", "--short"]) + "\n")
    write_text(run_dir / "README.txt", f"Agent handoff transcript for {args.phase_name}\n")

    print(f"Transcript: {run_dir}")
    if args.dry_run:
        dry_prompt = build_codex_prompt(
            phase_prompt,
            phase_name=args.phase_name,
            cycle=1,
            claude_review=None,
        )
        write_text(run_dir / "cycle_01" / "codex_prompt.md", dry_prompt)
        write_text(
            run_dir / "cycle_01" / "claude_review_schema.json",
            json.dumps(REVIEW_SCHEMA, indent=2, sort_keys=True) + "\n",
        )
        print("Dry run complete. No agents were invoked.")
        return 0

    claude_review: Optional[Dict[str, Any]] = None
    for cycle in range(1, args.cycles + 1):
        cycle_dir = run_dir / f"cycle_{cycle:02d}"
        cycle_dir.mkdir(parents=True, exist_ok=True)

        codex_prompt = build_codex_prompt(
            phase_prompt,
            phase_name=args.phase_name,
            cycle=cycle,
            claude_review=claude_review,
        )
        write_text(cycle_dir / "codex_prompt.md", codex_prompt)

        print(f"Cycle {cycle}: running Codex...")
        codex_result = run_codex(args, workspace, codex_prompt, cycle_dir)
        if codex_result.returncode != 0:
            print(f"Codex failed in cycle {cycle}; see {cycle_dir}", file=sys.stderr)
            return codex_result.returncode

        codex_last_message = (cycle_dir / "codex_last_message.md").read_text(
            encoding="utf-8"
        )
        git_snapshot = capture_git_snapshot(
            workspace, cycle_dir, args.diff_char_limit
        )
        claude_prompt = build_claude_prompt(
            phase_prompt,
            codex_last_message,
            git_snapshot,
            phase_name=args.phase_name,
            cycle=cycle,
        )
        write_text(cycle_dir / "claude_prompt.md", claude_prompt)

        print(f"Cycle {cycle}: running Claude review...")
        claude_result = run_claude(args, workspace, claude_prompt, cycle_dir)
        if claude_result.returncode != 0:
            print(f"Claude failed in cycle {cycle}; see {cycle_dir}", file=sys.stderr)
            return claude_result.returncode

        try:
            claude_review = extract_review(json_from_text(claude_result.stdout))
            validate_review(claude_review)
        except ValueError as exc:
            print(f"Could not parse Claude review: {exc}", file=sys.stderr)
            print(f"Raw output is saved in {cycle_dir / 'claude_stdout.json'}", file=sys.stderr)
            return 2

        write_text(
            cycle_dir / "claude_review.parsed.json",
            json.dumps(claude_review, indent=2, sort_keys=True) + "\n",
        )
        print(f"Cycle {cycle}: Claude verdict = {claude_review['verdict']}")

        if claude_review["verdict"] == "approve":
            write_text(run_dir / "final_verdict.json", json.dumps(claude_review, indent=2) + "\n")
            print("Approved by Claude.")
            return 0

    assert claude_review is not None
    write_text(run_dir / "final_verdict.json", json.dumps(claude_review, indent=2) + "\n")
    print(
        f"Reached cycle limit ({args.cycles}) with verdict "
        f"{claude_review['verdict']}. See {run_dir / 'final_verdict.json'}."
    )
    return 1 if claude_review["verdict"] == "revise" else 0


if __name__ == "__main__":
    raise SystemExit(main())
