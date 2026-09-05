import ast
import getpass
from pathlib import Path
import warnings

import pytest

from lab_arena import miner_submit
from lab_arena import contracts, source_bundle
from lab_arena.miner_submit import (
    MinerSubmissionError,
    prompt_submission_credentials,
    submit_agent_source,
    validate_agent_source,
)


HOTKEY = "5" * 48
NOW = 1_788_480_000
CREDENTIALS = {
    "openrouter_api_key": "openrouter-execution-secret",
    "openrouter_management_key": "openrouter-management-secret",
    "deepline_api_key": "deepline-execution-secret",
}


def _load_neuron_function(name: str):
    source_path = Path(__file__).resolve().parents[2] / "neurons" / "miner.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == name
    )
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {}
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace[name]


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

    def post(self, url, json, timeout, allow_redirects):
        assert allow_redirects is False
        self.posts.append((url, json, timeout, allow_redirects))
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
        credentials=CREDENTIALS,
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
        "credentials": CREDENTIALS,
    }
    assert all(secret.encode() not in archive for secret in CREDENTIALS.values())


@pytest.mark.parametrize(
    "credentials",
    (
        None,
        {},
        {
            "openrouter_api_key": "value",
            "openrouter_management_key": "value",
        },
        {
            "openrouter_api_key": "value",
            "openrouter_management_key": "value",
            "deepline_api_key": "",
        },
        {**CREDENTIALS, "deepline_api_key": " deepline-execution-secret"},
        {**CREDENTIALS, "deepline_api_key": "x" * 4097},
        {**CREDENTIALS, "unexpected_key": "value"},
    ),
)
def test_invalid_credentials_fail_before_network(tmp_path, credentials):
    source = _agent_source(tmp_path)

    class _NoNetwork:
        def get(self, *_args, **_kwargs):  # pragma: no cover - must not run
            raise AssertionError("network call attempted")

    with pytest.raises(MinerSubmissionError) as caught:
        submit_agent_source(
            source_dir=source,
            api_base_url="https://arena.example",
            keypair=_Keypair(),
            credentials=credentials,
            session=_NoNetwork(),
        )
    assert caught.value.code == "submission_credentials_required"


def test_credential_prompts_are_masked_and_environment_values_skip_prompts():
    prompts = []
    prompted = prompt_submission_credentials(
        environ={},
        getpass_fn=lambda prompt: prompts.append(prompt) or "masked-secret-value",
    )
    assert prompted == {name: "masked-secret-value" for name in CREDENTIALS}
    assert prompts == [
        "OpenRouter API key: ",
        "OpenRouter management key: ",
        "Deepline API key: ",
    ]

    def fail_prompt(_prompt):  # pragma: no cover - must not run
        raise AssertionError("environment credentials should not prompt")

    assert prompt_submission_credentials(
        environ={
            "OPENROUTER_API_KEY": CREDENTIALS["openrouter_api_key"],
            "OPENROUTER_MANAGEMENT_KEY": CREDENTIALS[
                "openrouter_management_key"
            ],
            "DEEPLINE_API_KEY": CREDENTIALS["deepline_api_key"],
        },
        getpass_fn=fail_prompt,
    ) == CREDENTIALS


def test_getpass_warning_fails_instead_of_echoing_credentials():
    def unsafe_prompt(_prompt):
        warnings.warn("terminal unavailable", getpass.GetPassWarning)
        raise AssertionError("warning must stop the prompt")  # pragma: no cover

    with pytest.raises(MinerSubmissionError) as caught:
        prompt_submission_credentials(environ={}, getpass_fn=unsafe_prompt)
    assert caught.value.code == "credential_prompt_unavailable"


def test_interactive_submission_keeps_credentials_out_of_input_and_output(monkeypatch):
    ordinary_prompts = []
    masked_prompts = []
    output = []
    submitted = {}
    answers = iter(("./agent", "yes"))
    secrets = iter(CREDENTIALS.values())

    def fake_submit(**kwargs):
        submitted.update(kwargs)
        return {
            "status": "accepted",
            "submission_id": "sub-0123456789abcdef0123456789abcdef",
            "round_id": "arena-2026-09-04",
        }

    monkeypatch.setattr(miner_submit, "submit_agent_source", fake_submit)
    assert miner_submit.run_interactive_submission(
        _Keypair(),
        "https://arena.example",
        input_fn=lambda prompt: ordinary_prompts.append(prompt) or next(answers),
        output_fn=output.append,
        getpass_fn=lambda prompt: masked_prompts.append(prompt) or next(secrets),
        environ={},
    )
    assert len(ordinary_prompts) == 2
    assert len(masked_prompts) == 3
    assert submitted["credentials"] == CREDENTIALS
    rendered = "\n".join(output + ordinary_prompts + masked_prompts)
    assert all(secret not in rendered for secret in CREDENTIALS.values())


def test_server_error_text_is_not_propagated():
    secret = "server-echoed-secret"
    response = _Response(
        500,
        {"detail": {"credentials": {"openrouter_api_key": secret}}},
    )
    with pytest.raises(MinerSubmissionError) as caught:
        miner_submit._json_response(response, "submission_finalize")
    assert caught.value.code == "arena_request_failed"
    assert secret not in str(caught.value)


@pytest.mark.parametrize("code", ["hotkey_unregistered", "submission_rate_limited", "submission_rejected:openrouter_management_key_invalid"])
def test_known_admission_errors_are_actionable_without_echoing_details(code):
    response = _Response(400, {"code": code, "detail": "server-echoed-secret"})
    with pytest.raises(MinerSubmissionError) as caught:
        miner_submit._json_response(response, "submission_finalize")
    assert caught.value.code == code
    assert "server-echoed-secret" not in str(caught.value)


def test_unknown_error_code_is_never_echoed():
    response = _Response(400, {"code": "submission_rejected:server-echoed-secret"})
    with pytest.raises(MinerSubmissionError) as caught:
        miner_submit._json_response(response, "submission_finalize")
    assert caught.value.code == "arena_request_failed"
    assert "server-echoed-secret" not in str(caught.value)


def test_cli_rejects_submitted_credentials_embedded_in_source_before_upload(tmp_path):
    source = _agent_source(tmp_path)
    (source / "agent.py").write_text(
        "KEY = %r\n" % CREDENTIALS["deepline_api_key"],
        encoding="utf-8",
    )
    session = _Session(
        {"open_round": {"round_id": "arena-2026-09-04", "status": "open"}}
    )
    with pytest.raises(MinerSubmissionError) as caught:
        submit_agent_source(
            source_dir=source,
            api_base_url="https://arena.example",
            keypair=_Keypair(),
            credentials=CREDENTIALS,
            session=session,
            now=lambda: NOW,
        )
    assert caught.value.code == "source_contains_credentials"
    assert session.posts == []
    assert session.uploads == []


def test_closed_round_stops_before_archive_or_upload(tmp_path):
    source = _agent_source(tmp_path)
    session = _Session({"open_round": None})
    with pytest.raises(MinerSubmissionError) as caught:
        submit_agent_source(
            source_dir=source,
            api_base_url="https://arena.example",
            keypair=_Keypair(),
            credentials=CREDENTIALS,
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
        credentials=CREDENTIALS,
        session=session,
        now=lambda: NOW,
    )
    assert result["status"] == "accepted"
    assert len(session.posts) == 2


def test_miner_menu_has_two_primary_submission_actions():
    miner = (Path(__file__).resolve().parents[2] / "neurons" / "miner.py").read_text(
        encoding="utf-8"
    )
    menu = miner.split("def _choose_primary_miner_mode", 1)[1].split(
        "def main", 1
    )[0]
    assert "Submit SOURCE_ADD" in menu
    assert "Submit Model" in menu
    assert "Fulfillment —" not in menu
    assert "Check my submissions" in menu
    assert "Auto Research" not in menu


@pytest.mark.parametrize(
    ("answers", "expected"),
    (
        (("",), "agent_competition"),
        (("2",), "agent_competition"),
        (("1", ""), "research_lab_source_add"),
        (("1", "1"), "research_lab_source_add"),
        (("1", "2"), "research_lab_source_add_status"),
    ),
)
def test_miner_menu_routes_two_primary_actions_and_source_add_status(
    answers,
    expected,
):
    choose = _load_neuron_function("_choose_primary_miner_mode")
    values = iter(answers)
    output = []
    assert choose(lambda _prompt: next(values), output.append) == expected
    primary_lines = [
        line for line in output if line.startswith("  1.") or line.startswith("  2.")
    ][:2]
    assert primary_lines == [
        "  1. Submit SOURCE_ADD — Submit or check an API/source candidate",
        "  2. Submit Model — Submit model source and run credentials (default)",
    ]
