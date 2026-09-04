from pathlib import Path

import pytest

from lab_arena import contracts
from lab_arena.miner_submit import (
    MinerSubmissionError,
    submit_agent_source,
    validate_agent_source,
)


HOTKEY = "5" * 48


class _Response:
    def __init__(self, status_code, document):
        self.status_code = status_code
        self._document = document

    def json(self):
        return self._document


class _Session:
    def __init__(self, current):
        self.current = current
        self.posts = []

    def get(self, url, timeout):
        assert url == "https://arena.example/arena/v1/current"
        assert timeout == 30
        return _Response(200, self.current)

    def post(self, url, json, timeout):
        self.posts.append((url, json, timeout))
        return _Response(200, {"status": "uploaded", "submission_id": "sub-123"})


class _Keypair:
    ss58_address = HOTKEY

    def sign(self, message):
        assert isinstance(message, bytes)
        return b"\x11" * 64


def _agent_source(tmp_path: Path, harness: str = "def run_icp(icp):\n    return []\n") -> Path:
    (tmp_path / "harness.py").write_text(harness, encoding="utf-8")
    (tmp_path / "Dockerfile").write_text("FROM python:3.11-slim\n", encoding="utf-8")
    return tmp_path


def test_source_validation_checks_callable_without_importing_code(tmp_path):
    source = _agent_source(tmp_path, "raise RuntimeError('must not run')\ndef run_icp(icp):\n    return []\n")
    assert validate_agent_source(source) == source.resolve()

    (source / "harness.py").write_text("def other(value):\n    return value\n", encoding="utf-8")
    with pytest.raises(MinerSubmissionError, match="run_icp_missing"):
        validate_agent_source(source)


@pytest.mark.parametrize(
    "harness,code",
    [
        ("async def run_icp(icp):\n    return []\n", "run_icp_must_be_sync"),
        ("def run_icp():\n    return []\n", "run_icp_input_missing"),
        ("def run_icp(icp, secret):\n    return []\n", "run_icp_has_required_extra_inputs"),
    ],
)
def test_source_validation_rejects_incompatible_callable(tmp_path, harness, code):
    source = _agent_source(tmp_path, harness)
    with pytest.raises(MinerSubmissionError) as caught:
        validate_agent_source(source)
    assert caught.value.code == code


def test_source_submission_builds_pushes_and_uses_signed_arena_endpoint(tmp_path):
    source = _agent_source(tmp_path)
    session = _Session(
        {"open_round": {"round_id": "arena-2026-09-04", "status": "open"}}
    )
    commands = []

    def run(command, check, timeout):
        commands.append((command, check, timeout))

    result = submit_agent_source(
        source_dir=source,
        image_reference="ghcr.io/example/my-agent:v1",
        api_base_url="https://arena.example/",
        keypair=_Keypair(),
        session=session,
        command_runner=run,
        now=lambda: 1_788_480_000,
    )

    assert commands == [
        (
            [
                "docker",
                "build",
                "--platform",
                "linux/amd64",
                "--tag",
                "ghcr.io/example/my-agent:v1",
                str(source),
            ],
            True,
            1800,
        ),
        (["docker", "push", "ghcr.io/example/my-agent:v1"], True, 1800),
    ]
    assert result["submission_id"] == "sub-123"
    assert result["round_id"] == "arena-2026-09-04"
    url, envelope, timeout = session.posts[0]
    assert url == "https://arena.example/arena/v1/submissions"
    assert timeout == 60
    assert envelope["body"] == {
        "image_reference": "ghcr.io/example/my-agent:v1",
        "consent": {"public_rerun": True},
    }
    contracts.validate_signed_request(
        envelope,
        expected_scope=contracts.SCOPE_SUBMISSION,
        expected_round_id="arena-2026-09-04",
        now=1_788_480_000,
        verify_signature=lambda hotkey, signature, message: hotkey == HOTKEY
        and signature == "0x" + "11" * 64,
    )


def test_closed_round_stops_before_docker_build(tmp_path):
    source = _agent_source(tmp_path)
    session = _Session({"open_round": None})
    commands = []
    with pytest.raises(MinerSubmissionError) as caught:
        submit_agent_source(
            source_dir=source,
            image_reference="ghcr.io/example/my-agent:v1",
            api_base_url="https://arena.example",
            keypair=_Keypair(),
            session=session,
            command_runner=lambda *args, **kwargs: commands.append((args, kwargs)),
        )
    assert caught.value.code == "submission_window_closed"
    assert commands == []


def test_miner_menu_has_agent_submission_and_no_autoresearch_choice():
    miner = (Path(__file__).resolve().parents[2] / "neurons" / "miner.py").read_text(encoding="utf-8")
    menu = miner.split("# MINER MODE SELECTION", 1)[1].split("# Create miner and run it properly", 1)[0]
    assert "Agent Competition" in menu
    assert "Submit your model/agent source code" in menu
    assert "Auto Research" not in menu
    assert "Resume Credit-Blocked" not in menu
