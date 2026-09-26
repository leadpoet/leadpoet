from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from scripts import verify_arena_trajectory as verification


ROUND = "arena-2026-09-25-trajectory-live"


def _specs():
    return [
        ("baseline", "primary", "run-baseline-primary"),
        ("baseline", "external-like", "run-baseline-external"),
        ("miner", "primary", "run-miner-primary"),
        ("miner", "external-like", "run-miner-external"),
    ]


class Store:
    def __init__(self):
        self.round = {
            "round_id": ROUND,
            "status": "stage1",
            "configuration_doc": {
                "mode": "shadow", "rewards_enabled": False,
                "stage_1_icp_count": 1, "stage_2_icp_count": 1,
            },
            "reward_activated_at": None,
            "publication_doc": None,
            "participants": [{"is_king": True}, {"is_king": False}],
        }
        self.runs = {}
        self.submissions = {}
        self.trajectories = {}
        self.ledger = {}
        provider_by_run = {
            "run-baseline-primary": "openrouter",
            "run-baseline-external": "deepline",
            "run-miner-primary": "scrapingdog",
            "run-miner-external": None,
        }
        for index, (role, path, run_id) in enumerate(_specs()):
            submission_id = "sub-" + role
            miner_hotkey = "miner-" + role
            runner_hotkey = "runner-a" if path == "primary" else "runner-b"
            run = {
                "run_id": run_id, "round_id": ROUND,
                "submission_id": submission_id, "miner_hotkey": miner_hotkey,
                "runner_hotkey": runner_hotkey,
                "assignment_id": "assignment-%d" % index,
                "stage": 1, "icp_position": index % 2, "attempt": 1,
                "kind": "execute", "status": "accepted",
                "result_doc": {"terminal_status": "accepted"},
            }
            self.runs[run_id] = run
            self.submissions[submission_id] = {
                "submission_id": submission_id, "miner_hotkey": miner_hotkey,
                "is_king": role == "baseline",
            }
            identity = {
                key: run[key] for key in verification.IDENTITY_FIELDS
            } | {
                "icp_identifier": "%s:icp:%s" % (ROUND, run["icp_position"]),
                "run_kind": "execute", "model_role": role,
            }
            events = [
                identity | {"event_kind": "runtime.started", "content": {}},
                identity | {
                    "event_kind": "runtime.finished",
                    "content": {"status": "accepted"},
                },
            ]
            provider = provider_by_run[run_id]
            ledger = []
            if provider:
                events += [
                    identity | {
                        "event_kind": "provider.request",
                        "content": {"provider": provider},
                    },
                    identity | {
                        "event_kind": "provider.response",
                        "content": {"call": {"provider": provider}},
                    },
                ]
                for entry_kind in ("reservation", "dispatch", "settlement"):
                    ledger.append({
                        "run_id": run_id, "round_id": ROUND,
                        "submission_id": submission_id,
                        "miner_hotkey": miner_hotkey,
                        "provider": provider,
                        "call_identity": "call-" + provider,
                        "entry_kind": entry_kind,
                    })
            self.trajectories[run_id] = events
            self.ledger[run_id] = ledger

    def get_round(self, round_id):
        return self.round if round_id == ROUND else None

    def get_run(self, run_id):
        return self.runs.get(run_id)

    def get_submission(self, submission_id):
        return self.submissions.get(submission_id)

    def list_trajectory_events(self, run_id):
        return self.trajectories[run_id]

    def list_ledger(self, *, run_id):
        return self.ledger[run_id]


def _service():
    return SimpleNamespace(store=Store())


def test_parser_requires_explicit_trajectory_round_suffix():
    with pytest.raises(SystemExit):
        verification.parser().parse_args([
            "verify", "--round-id", "arena-2026-09-25",
            "--environment-file", "/tmp/env", "--status-file", "/tmp/out",
            "--run", "baseline:primary:run-1",
        ])


def test_runner_environment_rejects_ambient_database_and_provider_secrets(
    monkeypatch, tmp_path
):
    environment = tmp_path / "validator.env"
    environment.write_text("LAB_ARENA_MODE=shadow\n")
    environment.chmod(0o600)
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "secret")
    monkeypatch.setenv("DEEPLINE_API_KEY", "secret")
    with pytest.raises(verification.VerificationError, match="credentials"):
        verification._load_runner_environment(environment)


def test_public_preflight_requires_two_roles_and_expected_submission():
    row = {
        "round_id": ROUND, "status": "stage1", "benchmark_icp_count": 2,
        "participants": [
            {"submission_id": "baseline", "is_baseline": True},
            {"submission_id": "miner", "is_baseline": False},
        ],
    }
    verification._public_participant(row, ROUND, "miner", "miner")
    with pytest.raises(verification.VerificationError):
        verification._public_participant(row, ROUND, "miner", "another")
    scoring = dict(row, status="stage1_scoring")
    verification._public_participant(scoring, ROUND, "baseline", "baseline", "score")


def test_proof_covers_roles_validators_canonical_rows_and_all_providers():
    result = verification._proof(_service(), ROUND, _specs())
    assert result["proof"] == {"complete": True, "errors": []}
    assert result["validator_path_hotkeys"] == {
        "primary": "runner-a", "external-like": "runner-b",
    }
    assert {row["model_terminal_status"] for row in result["runs"]} == {"accepted"}
    assert "content" not in str(result)

    published = _service()
    published.store.round["status"] = "published"
    published.store.round["publication_doc"] = {"private_shadow": True}
    assert verification._proof(published, ROUND, _specs())["proof"]["complete"] is True


@pytest.mark.parametrize("mutation", ["reward", "identity", "provider", "runner"])
def test_proof_fails_closed_on_safety_or_evidence_gap(mutation):
    service = _service()
    store = service.store
    if mutation == "reward":
        store.round["configuration_doc"]["rewards_enabled"] = True
    elif mutation == "identity":
        store.trajectories["run-miner-primary"][0]["runner_hotkey"] = "forged"
    elif mutation == "provider":
        store.trajectories["run-miner-primary"] = store.trajectories[
            "run-miner-primary"
        ][:2]
        store.ledger["run-miner-primary"] = []
    else:
        store.runs["run-miner-external"]["runner_hotkey"] = "runner-a"
        for row in store.trajectories["run-miner-external"]:
            row["runner_hotkey"] = "runner-a"
    with pytest.raises(verification.VerificationError):
        verification._proof(service, ROUND, _specs())


def test_runtime_log_is_optional_unless_requested():
    verification._proof(_service(), ROUND, _specs())
    with pytest.raises(verification.VerificationError, match="runtime trajectory"):
        verification._proof(_service(), ROUND, _specs(), logs=True)


def test_fixture_bounds_fail_before_service_or_production_access(monkeypatch):
    from lab_arena import wiring

    reached = []
    monkeypatch.setattr(
        wiring, "build_service_from_environment",
        lambda _mode: reached.append(True),
    )
    args = SimpleNamespace(
        icp_position=[0, 1], runner_hotkey=["only-one-runner"],
        cutoff_minutes=5, tick_seconds=5, port=19125,
    )
    with pytest.raises(verification.VerificationError, match="bounds"):
        verification._fixture_service(args)
    assert reached == []
