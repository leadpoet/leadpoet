from pathlib import Path

import pytest

from lab_arena import contracts, source_bundle
from lab_arena.miner_submit import MinerSubmissionError, submit_agent_source, validate_agent_source


HOTKEY = "5" * 48
NOW = 1_788_480_000


class _Response:
    def __init__(self, status_code, document=None):
        self.status_code = status_code
        self._document = document or {}

    def json(self):
        return self._document


class _Session:
    def __init__(self, current, *, upload_status=200):
        self.current = current
        self.upload_status = upload_status
        self.posts = []
        self.uploads = []

    def get(self, url, timeout):
        assert url == "https://arena.example/arena/v1/current" and timeout == 30
        return _Response(200, self.current)

    def post(self, url, json, timeout):
        self.posts.append((url, json, timeout))
        if url.endswith("/presign"):
            return _Response(
                200,
                {
                    "status": "upload_ready",
                    "submission_id": "sub-0123456789abcdef0123456789abcdef",
                    "source_ref": "arena/arena-2026-09-04/sources/sub-0123456789abcdef0123456789abcdef.tar.gz",
                    "upload_url": "https://uploads.example/source",
                    "upload_headers": {
                        "content-type": source_bundle.SOURCE_CONTENT_TYPE,
                        "content-length": str(json["body"]["source_size_bytes"]),
                    },
                    "expires_in_seconds": 900,
                },
            )
        return _Response(
            200,
            {
                "status": "accepted",
                "submission_id": "sub-0123456789abcdef0123456789abcdef",
            },
        )

    def put(self, url, data, headers, timeout):
        self.uploads.append((url, data.read(), dict(headers), timeout))
        return _Response(self.upload_status)


class _Keypair:
    ss58_address = HOTKEY

    def sign(self, message):
        assert isinstance(message, bytes)
        return b"\x11" * 64


def _agent_source(tmp_path: Path, harness: str = "def run_icp(icp):\n    return []\n") -> Path:
    (tmp_path / "harness.py").write_text(harness, encoding="utf-8")
    (tmp_path / "agent.py").write_text("VALUE = 1\n", encoding="utf-8")
    return tmp_path


def _valid_signature(envelope, scope):
    return contracts.validate_signed_request(
        envelope,
        expected_scope=scope,
        expected_round_id="arena-2026-09-04",
        now=NOW,
        verify_signature=lambda hotkey, signature, _message: hotkey == HOTKEY
        and signature == "0x" + "11" * 64,
    )


def test_source_validation_checks_syntax_without_importing_code(tmp_path):
    source = _agent_source(
        tmp_path,
        "raise RuntimeError('must not run')\ndef run_icp(icp):\n    return []\n",
    )
    assert validate_agent_source(source) == source.resolve()
    (source / "harness.py").write_text(
        "from package.adapter import run_icp\n", encoding="utf-8"
    )
    assert validate_agent_source(source) == source.resolve()
    (source / "harness.py").write_text("def run_icp(:\n", encoding="utf-8")
    with pytest.raises(MinerSubmissionError, match="harness_invalid"):
        validate_agent_source(source)


def test_source_submission_archives_uploads_and_finalizes_signed_bytes(tmp_path):
    source = _agent_source(tmp_path)
    session = _Session(
        {"open_round": {"round_id": "arena-2026-09-04", "status": "open"}}
    )
    result = submit_agent_source(
        source_dir=source,
        api_base_url="https://arena.example/",
        keypair=_Keypair(),
        session=session,
        now=lambda: NOW,
    )

    assert result == {
        "status": "accepted",
        "submission_id": "sub-0123456789abcdef0123456789abcdef",
        "round_id": "arena-2026-09-04",
    }
    presign = _valid_signature(
        session.posts[0][1], contracts.SCOPE_SUBMISSION_PRESIGN
    )
    assert contracts.validate_submission_presign_body(presign["body"])[
        "consent"
    ] == {"public_rerun": True}
    url, archive, headers, timeout = session.uploads[0]
    assert url == "https://uploads.example/source" and timeout == 300
    assert int(headers["content-length"]) == len(archive)
    facts = source_bundle.validate_source_archive(archive)
    assert facts["source_size_bytes"] == len(archive)
    assert presign["body"] == {
        "source_size_bytes": len(archive),
        "consent": {"public_rerun": True},
    }
    finalize = _valid_signature(
        session.posts[1][1], contracts.SCOPE_SUBMISSION_FINALIZE
    )
    assert finalize["body"] == {
        "submission_id": result["submission_id"],
        "source_ref": "arena/arena-2026-09-04/sources/%s.tar.gz"
        % result["submission_id"],
        "source_size_bytes": len(archive),
    }


def test_closed_round_stops_before_archive_or_upload(tmp_path):
    source = _agent_source(tmp_path)
    session = _Session({"open_round": None})
    with pytest.raises(MinerSubmissionError) as caught:
        submit_agent_source(
            source_dir=source,
            api_base_url="https://arena.example",
            keypair=_Keypair(),
            session=session,
        )
    assert caught.value.code == "submission_window_closed"
    assert session.posts == [] and session.uploads == []


def test_retry_can_finalize_when_the_write_once_upload_already_exists(tmp_path):
    session = _Session(
        {"open_round": {"round_id": "arena-2026-09-04", "status": "open"}},
        upload_status=412,
    )
    result = submit_agent_source(
        source_dir=_agent_source(tmp_path),
        api_base_url="https://arena.example",
        keypair=_Keypair(),
        session=session,
        now=lambda: NOW,
    )
    assert result["status"] == "accepted"
    assert len(session.posts) == 2


def test_miner_menu_has_agent_submission_and_no_autoresearch_choice():
    miner = (Path(__file__).resolve().parents[2] / "neurons" / "miner.py").read_text(
        encoding="utf-8"
    )
    menu = miner.split("# MINER MODE SELECTION", 1)[1].split(
        "# Create miner and run it properly", 1
    )[0]
    assert "Agent Competition" in menu
    assert "Submit your model/agent source code" in menu
    assert "Auto Research" not in menu
