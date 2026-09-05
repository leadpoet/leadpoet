"""Release the gateway builder from superseded automatic full-parity runs."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from urllib import error, parse, request


API_VERSION = "2022-11-28"
MAX_RESPONSE_BYTES = 2_000_000
WORKFLOW_PATH = ".github/workflows/physical-v2-staging.yml"
ACTIVE_INVENTORY_STATUSES = (
    "in_progress",
    "pending",
    "queued",
    "requested",
    "waiting",
)
NONTERMINAL_STATUSES = {"in_progress", "pending", "queued", "requested", "waiting"}


class GatewayRunnerAdmissionError(RuntimeError):
    pass


class GitHubActionsClient:
    def __init__(self, *, repository: str, token: str) -> None:
        if re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository) is None:
            raise GatewayRunnerAdmissionError("GitHub repository identity is invalid")
        if not token:
            raise GatewayRunnerAdmissionError("GitHub Actions token is unavailable")
        self._base = f"https://api.github.com/repos/{repository}"
        self._headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": API_VERSION,
        }

    def _json(self, path: str) -> dict:
        api_request = request.Request(self._base + path, headers=self._headers)
        with request.urlopen(api_request, timeout=30) as response:
            if response.headers.get("Link"):
                raise GatewayRunnerAdmissionError(
                    "GitHub workflow inventory pagination is ambiguous"
                )
            payload = response.read(MAX_RESPONSE_BYTES + 1)
        if len(payload) > MAX_RESPONSE_BYTES:
            raise GatewayRunnerAdmissionError("GitHub API response is oversized")
        value = json.loads(payload.decode("utf-8"))
        if not isinstance(value, dict):
            raise GatewayRunnerAdmissionError("GitHub API response is invalid")
        return value

    def runs(self, status: str) -> list[dict]:
        query = parse.urlencode({"status": status, "per_page": 100})
        value = self._json(
            f"/actions/workflows/{WORKFLOW_PATH.rsplit('/', 1)[-1]}/runs?" + query
        )
        runs = value.get("workflow_runs")
        total = value.get("total_count")
        if (
            not isinstance(runs, list)
            or len(runs) > 100
            or not isinstance(total, int)
            or isinstance(total, bool)
            or total != len(runs)
        ):
            raise GatewayRunnerAdmissionError(
                "GitHub workflow inventory is incomplete"
            )
        return runs

    def cancel(self, run_id: int) -> None:
        api_request = request.Request(
            f"{self._base}/actions/runs/{run_id}/cancel",
            data=b"",
            headers=self._headers,
            method="POST",
        )
        try:
            with request.urlopen(api_request, timeout=30) as response:
                if response.status != 202:
                    raise GatewayRunnerAdmissionError(
                        "GitHub run cancellation was not accepted"
                    )
        except error.HTTPError as exc:
            if exc.code in {409, 422} and self.status(run_id) == "completed":
                return
            raise GatewayRunnerAdmissionError(
                "GitHub run cancellation failed"
            ) from exc

    def status(self, run_id: int) -> str:
        value = self._json(f"/actions/runs/{run_id}")
        status = value.get("status")
        if value.get("id") != run_id:
            raise GatewayRunnerAdmissionError("GitHub run readback identity differs")
        if status != "completed" and status not in NONTERMINAL_STATUSES:
            raise GatewayRunnerAdmissionError("GitHub run status is invalid")
        return str(status)


def select_superseded_runs(
    runs: list[dict], *, current_sha: str, current_run_id: int
) -> tuple[int, ...]:
    observed = {}
    active = set()
    for run in runs:
        if not isinstance(run, dict):
            raise GatewayRunnerAdmissionError("GitHub workflow run is invalid")
        run_id = run.get("id")
        head_sha = run.get("head_sha")
        status = run.get("status")
        workflow_path = run.get("path")
        if (
            not isinstance(run_id, int)
            or isinstance(run_id, bool)
            or run_id <= 0
            or not isinstance(head_sha, str)
            or re.fullmatch(r"[0-9a-f]{40}", head_sha) is None
        ):
            raise GatewayRunnerAdmissionError("GitHub workflow run identity is invalid")
        if status != "completed" and status not in ACTIVE_INVENTORY_STATUSES:
            raise GatewayRunnerAdmissionError("GitHub workflow run status is invalid")
        allowed_paths = {
            WORKFLOW_PATH,
            WORKFLOW_PATH + "@main",
            WORKFLOW_PATH + "@refs/heads/main",
        }
        if workflow_path is not None and workflow_path not in allowed_paths:
            raise GatewayRunnerAdmissionError("GitHub workflow path differs")
        identity = (
            head_sha,
            run.get("event"),
            run.get("head_branch"),
            workflow_path,
        )
        if run_id in observed and observed[run_id] != identity:
            raise GatewayRunnerAdmissionError("GitHub workflow run identity conflicts")
        observed[run_id] = identity
        if status in ACTIVE_INVENTORY_STATUSES:
            active.add(run_id)

    selected = []
    for run_id, identity in observed.items():
        head_sha, event, head_branch, _workflow_path = identity
        if (
            run_id in active
            and event == "workflow_run"
            and head_branch == "main"
            and run_id < current_run_id
            and head_sha != current_sha
        ):
            selected.append(run_id)
    return tuple(sorted(selected))


def admit(
    client: GitHubActionsClient,
    *,
    current_sha: str,
    current_run_id: int,
    timeout_seconds: int,
    monotonic=time.monotonic,
    sleep=time.sleep,
) -> tuple[int, ...]:
    inventory = [
        run
        for status in ACTIVE_INVENTORY_STATUSES
        for run in client.runs(status)
    ]
    targets = select_superseded_runs(
        inventory, current_sha=current_sha, current_run_id=current_run_id
    )
    for run_id in targets:
        client.cancel(run_id)
    deadline = monotonic() + timeout_seconds
    pending = set(targets)
    while pending:
        pending = {run_id for run_id in pending if client.status(run_id) != "completed"}
        if not pending:
            break
        if monotonic() >= deadline:
            raise GatewayRunnerAdmissionError(
                "superseded full-parity run did not terminate before admission deadline"
            )
        sleep(10)
    return targets


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--current-sha", required=True)
    parser.add_argument("--current-run-id", required=True, type=int)
    parser.add_argument("--current-ref", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    args = parser.parse_args()
    if args.current_ref != "refs/heads/main":
        raise GatewayRunnerAdmissionError("gateway admission requires main")
    if re.fullmatch(r"[0-9a-f]{40}", args.current_sha) is None:
        raise GatewayRunnerAdmissionError("current commit identity is invalid")
    if args.current_run_id <= 0 or not 60 <= args.timeout_seconds <= 3600:
        raise GatewayRunnerAdmissionError("gateway admission bound is invalid")
    targets = admit(
        GitHubActionsClient(
            repository=args.repository,
            token=os.environ.get("GITHUB_TOKEN", ""),
        ),
        current_sha=args.current_sha,
        current_run_id=args.current_run_id,
        timeout_seconds=args.timeout_seconds,
    )
    print(f"superseded_full_parity_runs_terminal={len(targets)}")
    print("resource_cleanup_success_not_inferred=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
