import json
from urllib import error

import pytest
import yaml

import scripts.admit_gateway_runner_v2 as admission_module
from scripts.admit_gateway_runner_v2 import (
    GatewayRunnerAdmissionError,
    GitHubActionsClient,
    admit,
    select_superseded_runs,
)


CURRENT = "b" * 40
OLD = "a" * 40


def _run(
    run_id,
    *,
    sha=OLD,
    event="workflow_run",
    branch="main",
    status="in_progress",
    path=".github/workflows/physical-v2-staging.yml",
):
    return {
        "id": run_id,
        "head_sha": sha,
        "event": event,
        "head_branch": branch,
        "status": status,
        "path": path,
    }


def test_selection_cancels_only_older_automatic_main_runs_for_other_shas():
    assert select_superseded_runs(
        [
            _run(9),
            _run(8, sha=CURRENT),
            _run(7, event="workflow_dispatch"),
            _run(6, branch="feature"),
            _run(10),
            _run(11),
        ],
        current_sha=CURRENT,
        current_run_id=10,
    ) == (9,)


def test_selection_deduplicates_a_consistent_status_transition():
    assert select_superseded_runs(
        [_run(9, status="queued"), _run(9, status="in_progress")],
        current_sha=CURRENT,
        current_run_id=10,
    ) == (9,)


def test_selection_preserves_completed_and_rejects_conflicting_identity():
    assert (
        select_superseded_runs(
            [_run(9, status="completed")],
            current_sha=CURRENT,
            current_run_id=10,
        )
        == ()
    )
    with pytest.raises(GatewayRunnerAdmissionError, match="identity conflicts"):
        select_superseded_runs(
            [_run(9, status="queued"), _run(9, status="in_progress", sha=CURRENT)],
            current_sha=CURRENT,
            current_run_id=10,
        )


@pytest.mark.parametrize(
    "run, match",
    [
        (_run(9, status="failure"), "status is invalid"),
        (_run(9, path=".github/workflows/other.yml"), "path differs"),
    ],
)
def test_selection_rejects_invalid_status_or_workflow_path(run, match):
    with pytest.raises(GatewayRunnerAdmissionError, match=match):
        select_superseded_runs(
            [run], current_sha=CURRENT, current_run_id=10
        )


def test_selection_accepts_bounded_main_workflow_path_suffix():
    assert select_superseded_runs(
        [_run(9, path=".github/workflows/physical-v2-staging.yml@main")],
        current_sha=CURRENT,
        current_run_id=10,
    ) == (9,)


class _Client:
    def __init__(self, runs, statuses):
        self.inventory = list(runs)
        self.statuses = {key: iter(value) for key, value in statuses.items()}
        self.cancelled = []

    def runs(self, status):
        return self.inventory if status == "in_progress" else []

    def cancel(self, run_id):
        self.cancelled.append(run_id)

    def status(self, run_id):
        return next(self.statuses[run_id])


def test_admission_waits_for_selected_run_to_be_terminal():
    client = _Client([_run(9)], {9: ["in_progress", "completed"]})
    clock = iter([0, 1])
    sleeps = []
    assert admit(
        client,
        current_sha=CURRENT,
        current_run_id=10,
        timeout_seconds=30,
        monotonic=lambda: next(clock),
        sleep=sleeps.append,
    ) == (9,)
    assert client.cancelled == [9]
    assert sleeps == [10]


def test_admission_fails_closed_when_cancelled_run_does_not_terminate():
    client = _Client([_run(9)], {9: ["in_progress", "in_progress"]})
    clock = iter([0, 31])
    with pytest.raises(GatewayRunnerAdmissionError, match="admission deadline"):
        admit(
            client,
            current_sha=CURRENT,
            current_run_id=10,
            timeout_seconds=30,
            monotonic=lambda: next(clock),
            sleep=lambda _seconds: None,
        )


class _Response:
    def __init__(self, value, *, link=None, status=200):
        self.payload = json.dumps(value).encode("utf-8")
        self.headers = {"Link": link} if link is not None else {}
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self, _limit):
        return self.payload


@pytest.mark.parametrize(
    "response, match",
    [
        (
            _Response({"total_count": 101, "workflow_runs": [_run(9)] * 101}),
            "inventory is incomplete",
        ),
        (
            _Response({"total_count": 2, "workflow_runs": [_run(9)]}),
            "inventory is incomplete",
        ),
        (
            _Response(
                {"total_count": 1, "workflow_runs": [_run(9)]},
                link='<https://api.github.com/next>; rel="next"',
            ),
            "pagination is ambiguous",
        ),
    ],
)
def test_inventory_fails_closed_when_response_is_incomplete_or_paginated(
    monkeypatch, response, match
):
    monkeypatch.setattr(
        admission_module.request,
        "urlopen",
        lambda *_args, **_kwargs: response,
    )
    client = GitHubActionsClient(
        repository="leadpoet/leadpoet", token="test-token"
    )
    with pytest.raises(GatewayRunnerAdmissionError, match=match):
        client.runs("in_progress")


def test_cancel_race_accepts_only_a_run_that_is_already_terminal(monkeypatch):
    responses = iter(
        [
            error.HTTPError("https://api.github.test", 409, "conflict", {}, None),
            _Response({"id": 9, "status": "completed"}),
        ]
    )

    def urlopen(*_args, **_kwargs):
        response = next(responses)
        if isinstance(response, BaseException):
            raise response
        return response

    monkeypatch.setattr(admission_module.request, "urlopen", urlopen)
    client = GitHubActionsClient(
        repository="leadpoet/leadpoet", token="test-token"
    )
    client.cancel(9)


def test_terminal_readback_rejects_a_different_run_id(monkeypatch):
    monkeypatch.setattr(
        admission_module.request,
        "urlopen",
        lambda *_args, **_kwargs: _Response({"id": 10, "status": "completed"}),
    )
    client = GitHubActionsClient(
        repository="leadpoet/leadpoet", token="test-token"
    )
    with pytest.raises(GatewayRunnerAdmissionError, match="identity differs"):
        client.status(9)


def test_attested_workflow_scopes_mutation_and_gates_only_gateway_parent():
    source = open(".github/workflows/attested-v2-release.yml", encoding="utf-8").read()
    workflow = yaml.safe_load(source)
    jobs = workflow["jobs"]
    admission = jobs["gateway-runner-admission"]

    assert admission["runs-on"] == "ubuntu-latest"
    assert admission["permissions"] == {"actions": "write", "contents": "read"}
    assert workflow["permissions"] == {"contents": "read"}
    assert jobs["gateway-parent"]["needs"] == "gateway-runner-admission"
    assert "needs" not in jobs["validator-parent"]
    step_names = [step["name"] for step in admission["steps"]]
    assert step_names.index("Require exact main source before runner admission") < (
        step_names.index("Await cleanup of superseded automatic full parity")
    )
    identity = admission["steps"][1]["run"]
    assert 'test "$GITHUB_REF" = refs/heads/main' in identity
    assert 'test "$(git rev-parse HEAD)" = "$GITHUB_SHA"' in identity
    assert "refs/heads/main:refs/remotes/origin/main" in identity
    assert 'test "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)"' in identity
    command = admission["steps"][-1]["run"]
    assert "scripts/admit_gateway_runner_v2.py" in command
    assert "--timeout-seconds 1800" in command
