"""Real HTTP/Unix broker boundary and optional native Codex tool-loop proof."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest

from lab_arena import broker as br, operations as ops, runner, runtime
from lab_arena import lab_arena_codex as codex
from tests.lab_arena.test_lab_arena_broker import CONTEXT, FakeTransport, FakeLedgerStore, make_broker
from tests.lab_arena.test_lab_arena_runner import lease


def response(output=None, **changes):
    return dict({
        "id": "gen-codex-1", "object": "response", "created_at": 1789488000,
        "status": "completed", "model": "openai/gpt-4o-mini", "error": None,
        "output": output or [{"id": "msg-1", "type": "message", "role": "assistant", "status": "completed", "content": [{"type": "output_text", "text": "ARENA_CODEX_OK", "annotations": []}]}],
        "usage": {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20, "cost": 0.000012},
    }, **changes)


@contextmanager
def broker_socket(monkeypatch, transport=None, store=None):
    broker, store, transport = make_broker(transport=transport, store=store)

    class Api:
        def provider(self, run_id, lease_token, frame):
            return broker.execute(CONTEXT, **frame).to_document()

    with tempfile.TemporaryDirectory(prefix="codex-test-", dir="/tmp") as directory:
        path = Path(directory) / "worker.sock"
        state = runner.RunState(lease=lease("r1"), lease_token="tok-r1")
        server = runner.WorkerSocketServer(path, Api(), state)
        server.start()
        monkeypatch.setenv("LAB_ARENA_WORKER_SOCKET", str(path))
        monkeypatch.setenv("LAB_ARENA_WEB_EGRESS_SOCKET", str(Path(directory) / "web.sock"))
        try:
            yield store, transport, path
        finally:
            server.stop()


@pytest.mark.parametrize("extra", [
    {"stream": True}, {"store": True}, {"previous_response_id": "resp-other"},
    {"provider": {"allow_fallbacks": True}}, {"background": True},
    {"tools": [{"type": "web_search"}]}, {"tools": [{"type": "mcp", "server_url": "https://example.com"}]},
    {"input": [{"type": "item_reference", "id": "remote-item"}]},
    {"input": [{"role": "user", "content": [{"type": "input_image", "image_url": "https://example.com/a.png"}]}]},
    {"max_output_tokens": 4097}, {"input": True}, {"tools": {}},
])
def test_closed_responses_operation_rejects_unpriced_or_remote_work(extra):
    with pytest.raises(ops.OperationRequestError):
        ops.validate_operation_request("openrouter.responses", {"model": "openai/gpt-4o-mini", "input": "hi", **extra})


def test_responses_native_tool_history_and_caps():
    params = ops.validate_operation_request("openrouter.responses", {
        "model": "openai/gpt-4o-mini",
        "input": [
            {"role": "user", "content": [{"type": "input_text", "text": "research"}]},
            {"type": "function_call", "id": "fc-1", "call_id": "call-1", "name": "shell", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call-1", "output": "tool evidence"},
            {"type": "reasoning", "id": "rs-1", "summary": [], "encrypted_content": "opaque"},
        ],
        "tools": [{"type": "function", "name": "shell", "parameters": {"type": "object", "properties": {"url": {"type": "string"}}}}],
    })
    normalized = br.normalized_request("openrouter.responses", params)
    assert normalized["max_output_tokens"] == 4096
    assert "max_tokens" not in normalized
    outbound = ops.build_outbound_request("openrouter.responses", normalized)
    assert outbound.url == "https://openrouter.ai/api/v1/responses"
    body = json.loads(outbound.body)
    assert body["stream"] is False and body["store"] is False
    assert body["provider"] == dict(ops.OPENROUTER_STRICT_PROVIDER_POLICY)


def test_bridge_bills_real_worker_and_replays_sse(monkeypatch):
    with broker_socket(monkeypatch, FakeTransport([(200, response())])) as (store, transport, path), codex.ResponsesBridge(str(path)) as bridge:
        with httpx.Client(trust_env=False) as client:
            reply = client.post(bridge.base_url + "/responses", headers={"Authorization": "Bearer " + bridge.token}, json={"model": "openai/gpt-4o-mini", "input": "hi", "stream": True, "store": False})
        assert reply.status_code == 200, reply.text
        assert "event: response.completed" in reply.text
        assert "ARENA_CODEX_OK" in reply.text
        assert len(transport.sent) == 1
        assert next(iter(store.calls.values()))["actual"] == 12
        assert next(iter(store.calls.values()))["terminal"]["call_succeeded"] is True


def test_bridge_auth_and_stateful_requests_never_dispatch(monkeypatch):
    with broker_socket(monkeypatch) as (store, transport, path), codex.ResponsesBridge(str(path)) as bridge:
        with httpx.Client(trust_env=False) as client:
            assert client.post(bridge.base_url + "/responses", json={}).status_code == 401
            assert client.post(bridge.base_url + "/responses", headers={"Authorization": "Bearer " + bridge.token}, json={"store": True}).status_code == 400
            assert client.post(bridge.base_url + "/chat/completions", json={}).status_code == 404
        assert not store.calls and not transport.sent


def test_responses_budget_refusal_and_incomplete_billing(monkeypatch):
    with broker_socket(monkeypatch, store=FakeLedgerStore(openrouter_capacity=0)) as (store, transport, path):
        status, _ = codex._dispatch(str(path), {"model": "openai/gpt-4o-mini", "input": "hi"})
        assert status >= 400
        assert not transport.sent
    broker, store, _ = make_broker(transport=FakeTransport([(200, response(status="incomplete"))]))
    result = broker.execute(CONTEXT, operation_id="openrouter.responses", parameters={"model": "openai/gpt-4o-mini", "input": "hi"}, action_sequence=1, timeout_ms=120000)
    assert next(iter(store.calls.values()))["terminal"]["call_succeeded"] is False
    assert next(iter(store.calls.values()))["actual"] == 12


@pytest.mark.parametrize("scrapingdog_value", ["lab-arena-brokered-scrapingdog", "test-secret"])
def test_session_isolates_login_and_provider_keys(monkeypatch, scrapingdog_value):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret")
    monkeypatch.setenv("SCRAPINGDOG_API_KEY", scrapingdog_value)
    monkeypatch.setenv("CODEX_HOME", "/personal/config")
    with broker_socket(monkeypatch), codex.session(model="openai/gpt-4o-mini") as env:
        assert "OPENROUTER_API_KEY" not in env and "OPENAI_API_KEY" not in env
        if scrapingdog_value == "lab-arena-brokered-scrapingdog":
            assert env["SCRAPINGDOG_API_KEY"] == scrapingdog_value
        else:
            assert "SCRAPINGDOG_API_KEY" not in env
        assert env["CODEX_HOME"] != "/personal/config"
        home = Path(env["CODEX_HOME"])
        assert not (home / "auth.json").exists()
        assert 'request_max_retries = 0' in (home / "config.toml").read_text()
    assert not home.exists()


@pytest.mark.skipif(not os.environ.get("ARENA_TEST_CODEX_BINARY"), reason="set ARENA_TEST_CODEX_BINARY for the real CLI protocol proof")
def test_real_codex_tool_call_and_continuation(monkeypatch, tmp_path):
    request_errors = []
    original_dispatch = codex._dispatch

    def checked_dispatch(socket_path, document):
        try:
            ops.validate_operation_request("openrouter.responses", document)
        except ops.OperationRequestError as exc:
            request_errors.append((str(exc), list(document), document.get("tools")))
        return original_dispatch(socket_path, document)

    monkeypatch.setattr(codex, "_dispatch", checked_dispatch)
    class ToolTransport(FakeTransport):
        def send(self, **kwargs):
            document = json.loads(kwargs["body"])
            if not self.sent:
                names = [tool["name"] for tool in document["tools"]]
                name = "exec_command" if "exec_command" in names else "shell_command"
                assert name in names, names
                command = "python3 -c 'import lab_arena_codex; from pathlib import Path; Path(\"codex-tool-proof.txt\").write_text(lab_arena_codex.CODEX_VERSION); print(\"ARENA_TOOL_OK\")'"
                arguments = {"cmd": command, "max_output_tokens": 100} if name == "exec_command" else {"command": command}
                self.responses.append((200, response(output=[{"type": "function_call", "id": "fc-1", "call_id": "call-1", "name": name, "arguments": json.dumps(arguments), "status": "completed"}])))
            else:
                assert any(item.get("type") == "function_call_output" and "ARENA_TOOL_OK" in item["output"] for item in document["input"])
                self.responses.append((200, response(id="gen-codex-2")))
            return super().send(**kwargs)

    monkeypatch.setattr(codex, "CODEX_BINARY", os.environ["ARENA_TEST_CODEX_BINARY"])
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        monkeypatch.setenv(name, "http://127.0.0.1:9")
    monkeypatch.setenv("NO_PROXY", "")
    monkeypatch.setenv("no_proxy", "")
    with broker_socket(monkeypatch, ToolTransport()) as (store, transport, path):
        try:
            text = codex.run("Run the shell command provided by the model, then report success.", model="openai/gpt-4o-mini", cwd=tmp_path, timeout_seconds=60)
        except codex.CodexRuntimeError as exc:
            pytest.fail(exc.diagnostics[-8000:] + "\n" + repr(request_errors))
        assert "ARENA_CODEX_OK" in text
        assert (tmp_path / "codex-tool-proof.txt").read_text() == codex.CODEX_VERSION
        assert len(transport.sent) == 2
        assert sum(call["actual"] for call in store.calls.values()) == 24


def test_malformed_reply_fails_closed():
    document = response(output=[{"type": "message", "content": ["bad"]}])
    with pytest.raises(codex.CodexRuntimeError, match="invalid Responses content"):
        list(codex.response_events(document))


def test_noisy_cli_failure_keeps_only_a_bounded_tail(monkeypatch, tmp_path):
    executable = tmp_path / "fake-codex"
    executable.write_text("#!" + sys.executable + "\nimport sys\nsys.stdout.write('x' * 300000)\nsys.exit(7)\n")
    executable.chmod(0o755)
    monkeypatch.setattr(codex, "CODEX_BINARY", str(executable))
    with broker_socket(monkeypatch), pytest.raises(codex.CodexRuntimeError, match="status 7") as error:
        codex.run("test", model="openai/gpt-4o-mini", cwd=tmp_path, timeout_seconds=5)
    assert error.value.diagnostics == "x" * codex.MAX_LOG_BYTES


def test_cli_timeout_stops_the_process(monkeypatch, tmp_path):
    executable = tmp_path / "fake-codex"
    executable.write_text("#!" + sys.executable + "\nimport time\ntime.sleep(60)\n")
    executable.chmod(0o755)
    monkeypatch.setattr(codex, "CODEX_BINARY", str(executable))
    with broker_socket(monkeypatch), pytest.raises(subprocess.TimeoutExpired):
        codex.run("test", model="openai/gpt-4o-mini", cwd=tmp_path, timeout_seconds=0.1)


@pytest.mark.skipif(sys.platform != "linux" or not os.environ.get("ARENA_TEST_CODEX_ROOTFS"), reason="requires Linux root, runsc and the built Arena rootfs")
def test_codex_in_real_gvisor(monkeypatch):
    """Execute the shipped entrypoint/helper/CLI in the actual Arena sandbox."""
    root = Path(__file__).resolve().parents[2]
    tool = {"id": "fc-1", "type": "function_call", "call_id": "call-1", "name": "exec_command", "arguments": json.dumps({"cmd": "python3 -c 'import lab_arena_codex; print(\"ARENA_TOOL_OK\")'", "max_output_tokens": 100}), "status": "completed"}

    class ToolTransport(FakeTransport):
        def send(self, **kwargs):
            if not self.sent:
                self.responses.append((200, response(output=[tool])))
            else:
                body = json.loads(kwargs["body"])
                assert any(item.get("type") == "function_call_output" and "ARENA_TOOL_OK" in item["output"] for item in body["input"])
                self.responses.append((200, response(id="gen-codex-2")))
            return super().send(**kwargs)

    with broker_socket(monkeypatch, ToolTransport()) as (store, transport, socket_path), tempfile.TemporaryDirectory(prefix="arena-codex-gvisor-", dir="/tmp") as directory:
        work = Path(directory)
        work.chmod(0o755)
        for name in ("source", "deps", "input", "output", "runtime"):
            (work / name).mkdir(mode=0o755)
        (work / "input/icp.json").write_text('{"icp":{},"company_limit":5}')
        (work / "source/harness.py").write_text(
            "import lab_arena_codex\n"
            "def run_icp(icp):\n"
            "    result = lab_arena_codex.run('Run the requested tool, then finish.', model='openai/gpt-4o-mini', cwd='/tmp', timeout_seconds=60)\n"
            "    assert 'ARENA_CODEX_OK' in result\n"
            "    return []\n"
        )
        for name in ("agent_entrypoint.py", "lab_arena_checkpoint.py", "lab_arena_codex.py", "web_egress_bridge.py"):
            shutil.copyfile(root / "lab_arena" / name, work / name)
            (work / name).chmod(0o444)
        spec = runtime.SandboxSpec(
            sandbox_id="arena-codex-probe", rootfs_path=Path(os.environ["ARENA_TEST_CODEX_ROOTFS"]),
            input_dir=work / "input", output_dir=work / "output", socket_path=socket_path,
            source_dir=work / "source", dependency_dir=work / "deps",
            agent_entrypoint_path=work / "agent_entrypoint.py", checkpoint_module_path=work / "lab_arena_checkpoint.py",
            codex_module_path=work / "lab_arena_codex.py", web_bridge_path=work / "web_egress_bridge.py",
            entry_command=runtime.AGENT_ENTRY_COMMAND, working_dir=runtime.AGENT_WORKING_DIR,
            evaluation_date="2026-09-15", random_seed=1, wall_clock_seconds=90,
        )
        result = runtime.RunscRuntime(runtime.RuntimeConfig(
            runsc_path=Path(os.environ.get("ARENA_TEST_RUNSC", "/usr/local/bin/runsc")), work_dir=work / "runtime",
        )).run_icp(spec)
        assert result.exit_code == 0 and not result.timed_out, result.stderr.decode(errors="replace")
        assert result.output_bytes == b'{"companies":[]}'
        assert len(transport.sent) == 2
        assert sum(call["actual"] for call in store.calls.values()) == 24
