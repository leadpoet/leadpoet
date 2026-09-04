from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from tests.restart_rehearsal import production_workflow_runner as workflow


HELPER_SOURCE = r'''#!/usr/bin/env python3
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--scenario", required=True)
parser.add_argument("--result", type=Path, required=True)
parser.add_argument("--token", required=True)
parser.add_argument("--ordinal", type=int, required=True)
args = parser.parse_args()
track = Path(os.environ["BEHAVIOR_TEST_TRACK"])
lock = track.with_suffix(".lock")

def update(mutator):
    with lock.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        data = json.loads(track.read_text()) if track.exists() else {
            "active": 0,
            "events": [],
            "maximum": 0,
        }
        mutator(data)
        track.write_text(json.dumps(data, sort_keys=True))
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

state_root = os.environ["REHEARSAL_STATE_ROOT"]
source_root = os.environ["REHEARSAL_SOURCE_ROOT"]
assert os.environ["BEHAVIOR_TEST_SITECUSTOMIZE_STATE_ROOT"] == state_root
assert os.environ["BEHAVIOR_TEST_SITECUSTOMIZE_SOURCE_ROOT"] == source_root
Path(state_root).mkdir(parents=True)
update(lambda data: (
    data.__setitem__("active", data["active"] + 1),
    data.__setitem__("maximum", max(data["maximum"], data["active"])),
    data["events"].append({
        "event": "started",
        "pid": os.getpid(),
        "scenario": args.scenario,
        "source_root": source_root,
        "state_root": state_root,
    }),
))
print(f"stdout:{args.scenario}", flush=True)
print(f"stderr:{args.scenario}", file=sys.stderr, flush=True)
action_started_at = time.monotonic()
try:
    if args.scenario.startswith("slow"):
        time.sleep(0.20)
    elif args.scenario.startswith("medium"):
        time.sleep(0.08)
    elif args.scenario.startswith("hang"):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        time.sleep(30)
    else:
        time.sleep(0.02)
    if args.scenario.startswith("crash"):
        os._exit(7)
    common = {
        "duration_seconds": time.monotonic() - action_started_at,
        "ordinal": args.ordinal,
        "scenario": args.scenario,
        "schema_version": "leadpoet.workflow_behavior_worker.v1",
        "source_root": source_root,
        "state_root": state_root,
        "token": args.token,
    }
    if args.scenario.startswith("bad-duration"):
        common["duration_seconds"] = "invalid"
    if args.scenario.startswith("wrong"):
        common["scenario"] = "wrong-result"
    if args.scenario.startswith("fail"):
        payload = {
            **common,
            "error": "deliberate action failure",
            "error_type": "DeliberateFailure",
            "status": "failed",
            "traceback": "deliberate traceback",
        }
    else:
        payload = {
            **common,
            "status": "passed",
            "value": {
                "scenario": args.scenario,
                "source_root": source_root,
                "state_root": state_root,
            },
        }
    if args.scenario.startswith("malformed"):
        args.result.write_text("{malformed\n")
    elif args.scenario.startswith("duplicate"):
        line = json.dumps(payload, sort_keys=True) + "\n"
        args.result.write_text(line + line)
    else:
        args.result.write_text(json.dumps(payload, sort_keys=True) + "\n")
finally:
    update(lambda data: data.__setitem__("active", data["active"] - 1))
'''


@pytest.fixture
def behavior_helper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    helper = tmp_path / "behavior_helper.py"
    helper.write_text(HELPER_SOURCE, encoding="utf-8")
    (tmp_path / "sitecustomize.py").write_text(
        "\n".join(
            (
                "import os",
                "os.environ['BEHAVIOR_TEST_SITECUSTOMIZE_STATE_ROOT'] = "
                "os.environ['REHEARSAL_STATE_ROOT']",
                "os.environ['BEHAVIOR_TEST_SITECUSTOMIZE_SOURCE_ROOT'] = "
                "os.environ['REHEARSAL_SOURCE_ROOT']",
                "",
            )
        ),
        encoding="utf-8",
    )
    track = tmp_path / "track.json"
    monkeypatch.setenv("BEHAVIOR_TEST_TRACK", str(track))
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    monkeypatch.setattr(workflow, "SOURCE_ROOT", Path("/exact/source"))

    def configure(scenarios: list[str]) -> Path:
        monkeypatch.setattr(
            workflow,
            "BEHAVIOR_ACTIONS",
            {scenario: (lambda: {}) for scenario in scenarios},
        )
        monkeypatch.setattr(
            workflow,
            "_behavior_worker_command",
            lambda *, scenario, result_path, token, ordinal: [
                sys.executable,
                str(helper),
                "--scenario",
                scenario,
                "--result",
                str(result_path),
                "--token",
                token,
                "--ordinal",
                str(ordinal),
            ],
        )
        return track

    return configure


def _track(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_processes_reaped(path: Path, scenarios: set[str]) -> None:
    events = [
        item
        for item in _track(path)["events"]
        if item["scenario"] in scenarios
    ]
    assert {item["scenario"] for item in events} == scenarios
    for event in events:
        with pytest.raises(ProcessLookupError):
            os.kill(event["pid"], 0)


def test_behavior_workers_merge_out_of_order_results_canonically(
    behavior_helper,
    capsys: pytest.CaptureFixture[str],
) -> None:
    scenarios = [
        "slow-first",
        "fail-fast-second",
        "medium-third",
        "fast-fourth",
    ]
    track = behavior_helper(scenarios)
    stages: list[dict] = []

    evidence = workflow._run_behavior_actions(
        scenarios=scenarios,
        stages=stages,
    )

    captured = capsys.readouterr()
    assert list(evidence) == [
        "slow-first",
        "medium-third",
        "fast-fourth",
    ]
    assert [item["stage"] for item in stages] == [
        f"behavior:{scenario}" for scenario in scenarios
    ]
    assert [item["status"] for item in stages] == [
        "passed",
        "failed",
        "passed",
        "passed",
    ]
    passed_scenarios = [
        "slow-first",
        "medium-third",
        "fast-fourth",
    ]
    assert [
        evidence[scenario]["scenario"] for scenario in passed_scenarios
    ] == passed_scenarios
    assert len(
        {evidence[scenario]["state_root"] for scenario in passed_scenarios}
    ) == 3
    assert {
        evidence[scenario]["source_root"] for scenario in passed_scenarios
    } == {"/exact/source"}
    stage_by_name = {item["stage"]: item for item in stages}
    assert stage_by_name["behavior:slow-first"]["duration_seconds"] > 0.15
    assert (
        stage_by_name["behavior:slow-first"]["duration_seconds"]
        > stage_by_name["behavior:medium-third"]["duration_seconds"]
        > stage_by_name["behavior:fast-fourth"]["duration_seconds"]
        > 0.0
    )
    failed_stage = stage_by_name["behavior:fail-fast-second"]
    assert failed_stage["duration_seconds"] > 0.0
    assert failed_stage["error_type"] == "DeliberateFailure"
    assert failed_stage["error"] == "deliberate action failure"
    assert failed_stage["traceback"] == "deliberate traceback"
    assert [
        captured.out.index(f"stdout:{scenario}") for scenario in scenarios
    ] == sorted(
        captured.out.index(f"stdout:{scenario}") for scenario in scenarios
    )
    assert [
        captured.err.index(f"stderr:{scenario}") for scenario in scenarios
    ] == sorted(
        captured.err.index(f"stderr:{scenario}") for scenario in scenarios
    )
    tracked = _track(track)
    assert workflow._BEHAVIOR_WORKER_LIMIT == 3
    assert tracked["maximum"] == 3
    assert tracked["active"] == 0
    assert len({item["state_root"] for item in tracked["events"]}) == 4
    assert {item["source_root"] for item in tracked["events"]} == {
        "/exact/source"
    }


def test_behavior_workers_preserve_the_exact_candidate_inventory(
    behavior_helper,
) -> None:
    contract = workflow.build_rehearsal_behavior_contract_v2(
        source_root=Path(__file__).resolve().parents[2],
        candidate_sha="f" * 40,
        profile="prepush",
        epoch_count=1,
    )
    scenarios = list(contract["behavior_scenarios"])
    assert set(scenarios) == set(workflow.BEHAVIOR_ACTIONS)
    behavior_helper(scenarios)
    stages: list[dict] = []

    evidence = workflow._run_behavior_actions(
        scenarios=scenarios,
        stages=stages,
    )

    assert list(evidence) == scenarios
    assert [item["stage"] for item in stages] == [
        stage
        for stage in contract["required_stage_ids"]
        if stage.startswith("behavior:")
    ]
    assert [item["status"] for item in stages] == ["passed"] * len(scenarios)
    assert all(item["duration_seconds"] > 0.0 for item in stages)


def test_behavior_workers_reject_bad_results_and_continue_later_stages(
    behavior_helper,
) -> None:
    scenarios = [
        "fail-action",
        "fast-after-failure",
        "bad-duration",
        "malformed-result",
        "wrong-binding",
        "duplicate-result",
        "crash-result",
        "fast-final",
    ]
    track = behavior_helper(scenarios)
    stages: list[dict] = []

    evidence = workflow._run_behavior_actions(
        scenarios=scenarios,
        stages=stages,
    )

    assert [item["stage"] for item in stages] == [
        f"behavior:{scenario}" for scenario in scenarios
    ]
    assert [item["status"] for item in stages] == [
        "failed",
        "passed",
        "failed",
        "failed",
        "failed",
        "failed",
        "failed",
        "passed",
    ]
    assert stages[0]["error_type"] == "DeliberateFailure"
    assert stages[0]["error"] == "deliberate action failure"
    assert stages[0]["traceback"] == "deliberate traceback"
    assert stages[0]["duration_seconds"] > 0.0
    assert {
        item["error_type"] for item in stages[2:7]
    } == {"_BehaviorWorkerProtocolError"}
    assert list(evidence) == ["fast-after-failure", "fast-final"]
    _assert_processes_reaped(track, {"crash-result"})


def test_behavior_worker_timeout_kills_reaps_and_continues(
    behavior_helper,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenarios = [
        "hang-timeout-a",
        "hang-timeout-b",
        "hang-timeout-c",
        "fast-later",
    ]
    track = behavior_helper(scenarios)
    monkeypatch.setattr(
        workflow,
        "_BEHAVIOR_WORKER_TERMINATE_GRACE_SECONDS",
        0.05,
    )
    stages: list[dict] = []

    evidence = workflow._run_behavior_actions(
        scenarios=scenarios,
        stages=stages,
        worker_timeout_seconds=0.15,
    )

    assert [item["status"] for item in stages] == [
        "failed",
        "failed",
        "failed",
        "passed",
    ]
    assert stages[0]["error_type"] == "_BehaviorWorkerProtocolError"
    assert all(item["duration_seconds"] >= 0.15 for item in stages[:3])
    assert list(evidence) == ["fast-later"]
    tracked = _track(track)
    assert workflow._BEHAVIOR_WORKER_LIMIT == 3
    event_scenarios = [item["scenario"] for item in tracked["events"]]
    assert set(event_scenarios[:3]) == set(scenarios[:3])
    assert event_scenarios[3:] == ["fast-later"]
    _assert_processes_reaped(
        track,
        {"hang-timeout-a", "hang-timeout-b", "hang-timeout-c"},
    )


def test_behavior_scheduler_deadline_cleans_every_worker_and_reraises(
    behavior_helper,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenarios = ["hang-deadline-a", "hang-deadline-b", "hang-deadline-c"]
    track = behavior_helper(scenarios)
    monkeypatch.setattr(
        workflow,
        "_BEHAVIOR_WORKER_TERMINATE_GRACE_SECONDS",
        0.05,
    )

    with pytest.raises(TimeoutError, match="scheduler deadline exceeded"):
        workflow._run_behavior_actions(
            scenarios=scenarios,
            stages=[],
            deadline_monotonic=time.monotonic() + 0.20,
        )

    _assert_processes_reaped(track, set(scenarios))


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
def test_behavior_scheduler_signal_cleans_every_worker_and_reraises(
    behavior_helper,
    monkeypatch: pytest.MonkeyPatch,
    signum: int,
) -> None:
    scenarios = ["hang-signal-a", "hang-signal-b", "hang-signal-c"]
    track = behavior_helper(scenarios)
    monkeypatch.setattr(
        workflow,
        "_BEHAVIOR_WORKER_TERMINATE_GRACE_SECONDS",
        0.05,
    )
    signaler = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import os,signal,time; time.sleep(0.3); "
                f"os.kill({os.getpid()}, {signum})"
            ),
        ]
    )
    try:
        expected = (
            KeyboardInterrupt
            if signum == signal.SIGINT
            else workflow._BehaviorWorkerSignal
        )
        with pytest.raises(expected) as caught:
            workflow._run_behavior_actions(
                scenarios=scenarios,
                stages=[],
            )
    finally:
        signaler.wait(timeout=2)

    if signum == signal.SIGTERM:
        assert caught.value.signum == signal.SIGTERM
        assert caught.value.code == 128 + signal.SIGTERM
    _assert_processes_reaped(track, set(scenarios))
