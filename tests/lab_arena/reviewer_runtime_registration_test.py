"""Runtime activation uses the existing scoped store, never rewrites a round."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import httpx
import pytest

from lab_arena import store
from scripts import register_arena_reviewer_runtime as command


GRADING = "registry.example/lab/scorer@sha256:" + "a" * 64
RUNTIME = "registry.example/lab/scorer@sha256:" + "b" * 64


@pytest.mark.parametrize("runtime", ["registry.example/lab/scorer:latest",
                                      "registry.example/other@sha256:" + "b" * 64])
def test_unpinned_or_different_repository_rejected(runtime):
    with pytest.raises(ValueError):
        command.registration_arguments(GRADING, runtime)


def setup_args(monkeypatch, *, apply=False):
    monkeypatch.setattr(sys, "argv", ["register", "--grading-image", GRADING,
        "--runtime-image", RUNTIME, "--environment-file", "/protected/env"] +
        (["--apply"] if apply else []))


def test_plan_never_reads_credentials_or_calls_store(monkeypatch, capsys):
    setup_args(monkeypatch)
    monkeypatch.setattr(command, "load_scoped_environment", lambda _: pytest.fail("credential read"))
    monkeypatch.setattr(command, "PostgrestTransport", lambda *a, **k: pytest.fail("store called"))
    assert command.main() == 0
    assert '"writes": false' in capsys.readouterr().out


def test_child_plan_imports_without_creating_bytecode(tmp_path):
    root = tmp_path / "reviewed-release"
    (root / "scripts").mkdir(parents=True)
    (root / "lab_arena").mkdir()
    source = Path(command.__file__)
    shutil.copy2(source, root / "scripts" / source.name)
    (root / "scripts" / "__init__.py").write_text("")
    (root / "lab_arena" / "__init__.py").write_text("")
    (root / "lab_arena" / "images.py").write_text(
        "import sys\nassert sys.flags.dont_write_bytecode == 0\n"
        "assert sys.dont_write_bytecode\n"
        "from types import SimpleNamespace\n"
        "def parse_reference(value):\n"
        "    return SimpleNamespace(registry='registry.example', "
        "repository='lab/scorer', digest=value.split('@sha256:')[1])\n"
    )
    (root / "lab_arena" / "store.py").write_text(
        "import sys\nassert sys.dont_write_bytecode\n"
        "class PostgrestTransport:\n"
        "    def __init__(self, *args, **kwargs):\n"
        "        raise AssertionError('plan reached the store')\n"
    )
    (root / "scripts" / "run_lab_arena_service.py").write_text(
        "import sys\nassert sys.dont_write_bytecode\n"
        "def load_scoped_environment(path):\n"
        "    raise AssertionError('plan read credentials')\n"
    )
    env = os.environ.copy()
    for name in ("PYTHONDONTWRITEBYTECODE", "PYTHONPYCACHEPREFIX", "PYTHONPATH"):
        env.pop(name, None)
    result = subprocess.run(
        [sys.executable, str(root / "scripts" / source.name),
         "--grading-image", GRADING, "--runtime-image", RUNTIME,
         "--environment-file", str(root / "absent.env")],
        cwd=root, env=env, text=True, capture_output=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["writes"] is False
    assert not list(root.rglob("__pycache__"))
    assert not list(root.rglob("*.pyc"))


def test_apply_requires_operator_intent_before_credential_read(monkeypatch):
    setup_args(monkeypatch, apply=True)
    monkeypatch.delenv("LEADPOET_ARENA_REVIEWER_RUNTIME_APPLY", raising=False)
    monkeypatch.setattr(command, "load_scoped_environment", lambda _: pytest.fail("credential read"))
    with pytest.raises(ValueError, match="operator"):
        command.main()


@pytest.mark.parametrize("mismatch", [False, True])
def test_apply_exact_registration_readback_and_transport_cleanup(monkeypatch, mismatch):
    setup_args(monkeypatch, apply=True)
    monkeypatch.setenv("LEADPOET_ARENA_REVIEWER_RUNTIME_APPLY", "1")
    monkeypatch.setenv("LAB_ARENA_SUPABASE_URL", "https://store.example")
    monkeypatch.setenv("LAB_ARENA_SERVICE_KEY", "sb_secret_test")
    monkeypatch.setattr(command, "load_scoped_environment", lambda _: None)
    calls = []
    class Transport:
        def __init__(self, url, *, service_key):
            assert url == "https://store.example" and service_key == "sb_secret_test"
        def rpc(self, name, arguments):
            calls.append(name)
            assert arguments == command.registration_arguments(GRADING, RUNTIME)
            result = {key.removeprefix("p_"): value for key, value in arguments.items()}
            result["updated_at"] = "2026-10-16T23:00:00Z"
            if mismatch:
                result["runtime_image_reference"] = GRADING
            return result
        def close(self):
            calls.append("close")
    monkeypatch.setattr(command, "PostgrestTransport", Transport)
    if mismatch:
        with pytest.raises(ValueError, match="readback"):
            command.main()
    else:
        assert command.main() == 0
    assert calls == ["lab_arena_register_reviewer_runtime_v1", "close"]


@pytest.mark.parametrize("http_status", [200, 503])
def test_apply_uses_real_postgrest_rpc_and_closes_transport(
    monkeypatch, capsys, http_status,
):
    setup_args(monkeypatch, apply=True)
    monkeypatch.setenv("LEADPOET_ARENA_REVIEWER_RUNTIME_APPLY", "1")
    monkeypatch.setenv("LAB_ARENA_SUPABASE_URL", "https://store.example")
    monkeypatch.setenv("LAB_ARENA_SERVICE_KEY", "sb_secret_test")
    monkeypatch.setattr(command, "load_scoped_environment", lambda _: None)
    arguments = command.registration_arguments(GRADING, RUNTIME)
    registered = {key.removeprefix("p_"): value for key, value in arguments.items()}
    registered["updated_at"] = "2026-10-16T23:00:00Z"
    requests = []

    def reply(request):
        requests.append(request)
        assert request.method == "POST"
        assert str(request.url) == (
            "https://store.example/rest/v1/rpc/lab_arena_register_reviewer_runtime_v1"
        )
        assert json.loads(request.content) == arguments
        assert request.headers["apikey"] == "sb_secret_test"
        return httpx.Response(
            http_status,
            json=registered if http_status == 200 else {"message": "synthetic unavailable"},
        )

    client = httpx.Client(transport=httpx.MockTransport(reply))
    monkeypatch.setattr(store, "create_http1_client", lambda _timeout, **_kwargs: client)
    assert command.PostgrestTransport is store.PostgrestTransport
    try:
        if http_status == 200:
            assert command.main() == 0
            assert json.loads(capsys.readouterr().out) == registered
        else:
            with pytest.raises(store.ArenaStoreError, match="HTTP 503"):
                command.main()
            assert capsys.readouterr().out == ""
        assert len(requests) == 1
        assert client.is_closed
    finally:
        client.close()
