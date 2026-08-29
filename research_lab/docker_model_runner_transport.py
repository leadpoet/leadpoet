"""Credential-free OCI calls to the common champion runner adapter."""

from __future__ import annotations

from dataclasses import replace
import json
import math
import os
import queue
import re
import subprocess
import threading
import time
import uuid
from typing import Any, Mapping, Sequence

from research_lab.canonical import sha256_json
from research_lab.eval.private_runtime import (
    DockerPrivateModelRunner,
    PrivateModelRuntimeError,
    PROVIDER_COST_EVALUATION_SCOPE_ENV,
    _build_docker_process_env,
    _collect_incontainer_trace,
    _docker_env_args,
    _docker_lifecycle_remaining_seconds,
    _docker_platform_args,
    _docker_private_model_lifecycle,
    _loads_adapter_stdout,
    _raise_on_empty_provider_error,
    _remove_private_model_container,
    _sanitize_text,
)


_COMMON_RUNNER_BOOTSTRAP = r"""
import contextlib
import hashlib
import importlib
import json
from pathlib import Path
import sys

module_name, operation = sys.argv[1:3]
payload = json.load(sys.stdin)
module = importlib.import_module(module_name)

def declared_member(role, legacy_field):
    name = payload.get("member_name")
    if not isinstance(name, str) or not name.isidentifier():
        raise RuntimeError("common runner member name is invalid")
    metadata = module.adapter_metadata()
    champion = metadata.get("champion_execution")
    if not isinstance(champion, dict):
        raise RuntimeError("common runner champion metadata is invalid")
    role_contract = champion.get("runner_role_contract")
    if isinstance(role_contract, dict):
        roles = role_contract.get("roles")
        role_entry = roles.get(role) if isinstance(roles, dict) else None
        declared = (
            role_entry.get("adapter_member")
            if isinstance(role_entry, dict)
            else None
        )
    else:
        declared = champion.get(legacy_field)
    # The frozen v2 generation predates explicit start-member metadata.  Its
    # exact consumer contract remains the only authority for that one member.
    if declared is not None and declared != name:
        raise RuntimeError("common runner member differs from artifact metadata")
    if isinstance(role_contract, dict) and declared is None:
        raise RuntimeError("common runner semantic role is unavailable")
    member = getattr(module, name, None)
    if not callable(member):
        raise RuntimeError("common runner member is unavailable")
    return member

if operation == "runner_protocol_generation":
    metadata = module.adapter_metadata()
    contract_path = Path(module.__file__).resolve().parent / "sourcing_model" / "consumer_contract.json"
    contract_bytes = contract_path.read_bytes()
    contract_sha256 = hashlib.sha256(contract_bytes).hexdigest()
    if contract_sha256 != payload["release_identity"]["consumer_contract_sha256"]:
        raise RuntimeError("common runner consumer contract differs from release")
    contract = json.loads(contract_bytes)
    champion = metadata.get("champion_execution")
    role_contract = (
        champion.get("runner_role_contract")
        if isinstance(champion, dict)
        else None
    )
    contract_functions = contract.get("functions", {})
    contract_signatures = contract.get("exact_signatures", [])
    contract_full_parameters = contract.get("full_parameters", {})
    contract_keyword_only = contract.get("required_keyword_only", {})
    contract_asyncness = contract.get("frozen_asyncness", {})
    if isinstance(role_contract, dict):
        roles = role_contract.get("roles")
        if not isinstance(roles, dict):
            raise RuntimeError("common runner semantic roles are unavailable")
        functions = {}
        exact_signatures = []
        full_parameters = {}
        required_keyword_only = {}
        frozen_asyncness = {}
        for role, entry in sorted(roles.items()):
            if not isinstance(entry, dict):
                raise RuntimeError("common runner semantic role is invalid")
            member = entry.get("adapter_member")
            signature = entry.get("consumer_signature")
            path = (
                signature.get("consumer_contract_path")
                if isinstance(signature, dict)
                else None
            )
            if (
                not isinstance(member, str)
                or not member.isidentifier()
                or not isinstance(path, str)
                or ":" not in path
                or path.rsplit(":", 1)[1] != member
            ):
                raise RuntimeError("common runner role member path is invalid")
            source_path = path.rsplit(":", 1)[0]
            source_functions = (
                contract_functions.get(source_path)
                if isinstance(contract_functions, dict)
                else None
            )
            if not isinstance(source_functions, dict):
                raise RuntimeError("common runner role functions are unavailable")
            functions[member] = source_functions.get(member)
            if path in contract_signatures:
                exact_signatures.append(path)
            full_parameters[member] = contract_full_parameters.get(path)
            if path in contract_keyword_only:
                required_keyword_only[member] = contract_keyword_only[path]
            # Consumer contracts historically list asynchronous members and
            # may omit synchronous ``False`` entries.  The signed role
            # signature remains the exact asyncness authority.
            frozen_asyncness[member] = contract_asyncness.get(path, False)
        consumer_contract = {
            "schema_version": contract.get("schema_version"),
            "contract_id": contract.get("contract_id"),
            "functions": functions,
            "exact_signatures": sorted(exact_signatures),
            "full_parameters": full_parameters,
            "required_keyword_only": required_keyword_only,
            "exact_constants": {
                key: value
                for key, value in contract.get("exact_constants", {}).items()
                if key in {
                    "sourcing_model/model_runner.py",
                    "sourcing_model/raw_icp_normalization.py",
                }
            },
            "extensions": contract.get("extensions"),
            "frozen_asyncness": frozen_asyncness,
        }
    else:
        adapter_path = "research_lab_adapter.py"
        prefix = adapter_path + ":"
        functions = (
            contract_functions.get(adapter_path)
            if isinstance(contract_functions, dict)
            else None
        )
        if not isinstance(functions, dict):
            raise RuntimeError("common runner consumer functions are unavailable")
        consumer_contract = {
            "schema_version": contract.get("schema_version"),
            "contract_id": contract.get("contract_id"),
            "functions": functions,
            "exact_signatures": sorted(
                item for item in contract_signatures
                if isinstance(item, str) and item.startswith(prefix)
            ),
            "full_parameters": {
                key[len(prefix):]: value
                for key, value in contract_full_parameters.items()
                if isinstance(key, str) and key.startswith(prefix)
            },
            "required_keyword_only": {
                key[len(prefix):]: value
                for key, value in contract_keyword_only.items()
                if isinstance(key, str) and key.startswith(prefix)
            },
            "exact_constants": {
                key: value
                for key, value in contract.get("exact_constants", {}).items()
                if key in {
                    "sourcing_model/model_runner.py",
                    "sourcing_model/raw_icp_normalization.py",
                }
            },
        }
    result = {
        "schema_version": "leadpoet.research_lab.artifact_runner_declaration.v1",
        "champion_execution": metadata.get("champion_execution"),
        "consumer_contract_sha256": contract_sha256,
        "consumer_contract": consumer_contract,
    }
elif operation == "build_raw_runner_input":
    result = declared_member("raw_icp_input", "raw_icp_entrypoint")(
        payload["payload"],
        source_schema=payload["source_schema"],
    )
elif operation == "build_runner_start":
    result = declared_member("start", "start_entrypoint")(
        input=payload["input"],
        execution_mode=payload["execution_mode"],
        target_count=payload["target_count"],
        evaluated_on=payload["evaluated_on"],
        host_capability_manifest=payload["host_capability_manifest"],
        release_identity=payload["release_identity"],
    )
elif operation == "continue_runner":
    result = declared_member("continuation", "entrypoint")(
        payload["start_request"],
        expected_release_identity=payload["expected_release_identity"],
        continuation=payload.get("continuation"),
        completion=payload.get("completion"),
    )
elif operation == "build_runner_completion":
    result = declared_member("completion", "completion_entrypoint")(
        payload["action"],
        payload["result"],
    )
elif operation == "build_runner_provider_receipt_binding":
    result = declared_member(
        "provider_receipt_binding", "provider_receipt_binding_entrypoint"
    )(
        payload["action"],
        payload["result"],
    )
elif operation == "build_host_capability_manifest":
    result = declared_member(
        "host_capability_manifest", "host_capability_manifest_entrypoint"
    )(
        payload["bindings"],
    )
elif operation == "project_runner_result_for_benchmark":
    result = declared_member(
        "benchmark_projection", "benchmark_projection_entrypoint"
    )(
        payload["value"],
        start_request=payload["start_request"],
        expected_release_identity=payload["expected_release_identity"],
    )
elif operation == "build_official_baseline_execution":
    result = declared_member(
        "official_baseline_execution", "official_baseline_execution_entrypoint"
    )(
        release_identity=payload["release_identity"],
        protocol_generation_sha256=payload["protocol_generation_sha256"],
        protected_action_authority_sha256=payload[
            "protected_action_authority_sha256"
        ],
    )
elif operation == "prepare_runner_provider_request":
    result = declared_member("provider_prepare", "provider_prepare_entrypoint")(
        payload["action"],
    )
elif operation == "ingest_runner_provider_response":
    result = declared_member(
        "provider_response_ingestion",
        "provider_response_ingestion_entrypoint",
    )(
        payload["action"],
        payload["host_response"],
    )
elif operation == "prepare_runner_normalization_request":
    champion = module.adapter_metadata().get("champion_execution", {})
    role_contract = (
        champion.get("runner_role_contract")
        if isinstance(champion, dict)
        else None
    )
    if isinstance(role_contract, dict):
        roles = role_contract.get("roles")
        role_entry = (
            roles.get("normalization_prepare_legacy")
            if isinstance(roles, dict)
            else None
        )
        declared = (
            role_entry.get("adapter_member")
            if isinstance(role_entry, dict)
            else None
        )
    else:
        normalization = (
            champion.get("normalization_action")
            if isinstance(champion, dict)
            else None
        )
        declared = (
            normalization.get("dispatch_entrypoint")
            if isinstance(normalization, dict)
            else None
        )
    if declared != payload["member_name"]:
        raise RuntimeError(
            "common runner normalization member differs from artifact metadata"
        )
    member = getattr(module, payload["member_name"], None)
    if not callable(member):
        raise RuntimeError("common runner normalization member is unavailable")
    result = member(payload["action"])
elif operation == "model_runner_provider_compiler_inventory":
    result = declared_member(
        "provider_compiler_inventory", "provider_compiler_inventory_entrypoint"
    )()
elif operation == "runner_provider_compiler_preflight":
    result = declared_member(
        "provider_compiler_preflight", "provider_compiler_preflight_entrypoint"
    )(
        payload["host_capability_manifest"],
    )
elif operation == "execute_runner_verifier_action":
    result = declared_member(
        "verifier_execution", "verifier_execution_entrypoint"
    )(
        payload["action"],
    )
elif operation == "runner_official_host_binding_catalog":
    result = declared_member(
        "official_host_binding_catalog", "official_host_binding_catalog_entrypoint"
    )()
elif operation == "build_runner_official_host_capability_manifest":
    result = declared_member(
        "official_host_capability_manifest",
        "official_host_capability_manifest_entrypoint",
    )(payload["availability"])
elif operation == "runner_preflight":
    result = declared_member("preflight", "preflight_entrypoint")(
        host_capability_manifest=payload["host_capability_manifest"],
        release_identity=payload["release_identity"],
        execution_mode=payload["execution_mode"],
    )
elif operation == "validate_runner_preflight":
    result = declared_member(
        "preflight_validation", "preflight_validation_entrypoint"
    )(
        payload["value"],
        host_capability_manifest=payload["host_capability_manifest"],
        release_identity=payload["release_identity"],
        execution_mode=payload["execution_mode"],
    )
elif operation == "validate_runner_result":
    result = declared_member(
        "result_validation", "result_validation_entrypoint"
    )(
        payload["value"],
        start_request=payload["start_request"],
        expected_release_identity=payload["expected_release_identity"],
    )
else:
    raise RuntimeError("unsupported common runner operation")
with contextlib.redirect_stdout(sys.stderr):
    encoded = json.dumps(result, sort_keys=True, separators=(",", ":"))
sys.stdout.write(encoded)
"""


_COMMON_RUNNER_SESSION_SCHEMA_VERSION = (
    "leadpoet.research_lab.common_runner_session.v1"
)
_COMMON_RUNNER_MAX_PARALLEL_SESSIONS = 4
_COMMON_RUNNER_SESSION_BOOTSTRAP = r"""
import contextlib
import importlib
import io
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback

schema_version = "leadpoet.research_lab.common_runner_session.v1"
common_bootstrap, module_name = sys.argv[1:3]
if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]{0,191}", module_name) is None:
    raise RuntimeError("common runner session module name is invalid")

session_provider_cost_scope = str(
    os.environ.get("RESEARCH_LAB_PROVIDER_COST_EVALUATION_SCOPE") or ""
).strip()
if re.fullmatch(r"sha256:[0-9a-f]{64}", session_provider_cost_scope) is None:
    raise RuntimeError("common runner session provider cost scope is invalid")
os.environ["RESEARCH_LAB_PROVIDER_COST_SCOPE"] = session_provider_cost_scope

common_code = compile(common_bootstrap, "<common_runner_bootstrap>", "exec")
fork_enabled = callable(getattr(os, "fork", None))
if fork_enabled:
    preload_stdout = io.StringIO()
    preload_stderr = io.StringIO()
    with contextlib.redirect_stdout(preload_stdout):
        with contextlib.redirect_stderr(preload_stderr):
            importlib.import_module(module_name)
    fork_enabled = threading.active_count() == 1
    task_root = "/proc/self/task"
    if fork_enabled and os.path.isdir(task_root):
        try:
            fork_enabled = len(os.listdir(task_root)) == 1
        except OSError:
            fork_enabled = False


def anonymous_binary_file(name):
    create = getattr(os, "memfd_create", None)
    if callable(create):
        descriptor = create(name, flags=getattr(os, "MFD_CLOEXEC", 0))
        return os.fdopen(descriptor, "w+b", buffering=0)
    return tempfile.TemporaryFile(mode="w+b")


def run_in_fresh_fork(operation, payload, timeout_seconds, provider_cost_scope):
    encoded_payload = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    with (
        anonymous_binary_file("leadpoet-runner-stdin") as stdin_file,
        anonymous_binary_file("leadpoet-runner-stdout") as stdout_file,
        anonymous_binary_file("leadpoet-runner-stderr") as stderr_file,
    ):
        stdin_file.write(encoded_payload)
        stdin_file.seek(0)
        sys.stdout.flush()
        sys.stderr.flush()
        child_pid = os.fork()
        if child_pid == 0:
            exit_code = 1
            try:
                try:
                    os.setsid()
                except OSError:
                    pass
                os.dup2(stdin_file.fileno(), 0)
                os.dup2(stdout_file.fileno(), 1)
                os.dup2(stderr_file.fileno(), 2)
                sys.stdin = open(
                    0,
                    mode="r",
                    encoding="utf-8",
                    errors="strict",
                    closefd=False,
                )
                sys.stdout = open(
                    1,
                    mode="w",
                    encoding="utf-8",
                    errors="strict",
                    buffering=1,
                    closefd=False,
                )
                sys.stderr = open(
                    2,
                    mode="w",
                    encoding="utf-8",
                    errors="replace",
                    buffering=1,
                    closefd=False,
                )
                sys.__stdin__ = sys.stdin
                sys.__stdout__ = sys.stdout
                sys.__stderr__ = sys.stderr
                sys.argv = ["-c", module_name, operation]
                os.environ["RESEARCH_LAB_PROVIDER_COST_SCOPE"] = (
                    provider_cost_scope
                )
                for loaded_module in tuple(sys.modules.values()):
                    namespace = getattr(loaded_module, "__dict__", None)
                    if (
                        isinstance(namespace, dict)
                        and "_research_lab_provider_cost_scope" in namespace
                    ):
                        namespace["_research_lab_provider_cost_scope"] = (
                            provider_cost_scope
                        )
                exec(common_code, {"__name__": "__main__"})
                exit_code = 0
            except BaseException:
                traceback.print_exc()
            finally:
                try:
                    sys.stdout.flush()
                    sys.stderr.flush()
                except BaseException:
                    pass
                os._exit(exit_code)

        timed_out = False
        child_status = 0
        deadline = time.monotonic() + float(timeout_seconds)
        while True:
            try:
                waited_pid, child_status = os.waitpid(child_pid, os.WNOHANG)
            except InterruptedError:
                continue
            if waited_pid == child_pid:
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                try:
                    os.killpg(child_pid, signal.SIGKILL)
                except OSError:
                    try:
                        os.kill(child_pid, signal.SIGKILL)
                    except OSError:
                        pass
                while True:
                    try:
                        os.waitpid(child_pid, 0)
                        break
                    except InterruptedError:
                        continue
                    except ChildProcessError:
                        break
                break
            time.sleep(min(0.01, remaining))

        stdout_file.seek(0)
        stderr_file.seek(0)
        stdout = stdout_file.read().decode("utf-8", "replace")
        stderr = stderr_file.read().decode("utf-8", "replace")
        if timed_out:
            return None, stdout, stderr, True
        if os.WIFEXITED(child_status):
            returncode = os.WEXITSTATUS(child_status)
        elif os.WIFSIGNALED(child_status):
            returncode = -os.WTERMSIG(child_status)
        else:
            returncode = -1
        return returncode, stdout, stderr, False

sys.stdout.write(json.dumps({
    "schema_version": schema_version,
    "status": "ready",
}, sort_keys=True, separators=(",", ":")) + "\n")
sys.stdout.flush()

for line in sys.stdin:
    request_id = ""
    try:
        request = json.loads(line)
        if not isinstance(request, dict) or set(request) != {
            "schema_version",
            "request_id",
            "operation",
            "payload",
            "timeout_seconds",
            "provider_cost_scope",
        }:
            raise RuntimeError("common runner session request is not closed")
        request_id = request.get("request_id")
        operation = request.get("operation")
        payload = request.get("payload")
        timeout_seconds = request.get("timeout_seconds")
        provider_cost_scope = request.get("provider_cost_scope")
        if (
            request.get("schema_version") != schema_version
            or not isinstance(request_id, str)
            or re.fullmatch(r"[0-9a-f]{32}", request_id) is None
            or not isinstance(operation, str)
            or re.fullmatch(r"[a-z][a-z0-9_]{0,95}", operation) is None
            or not isinstance(payload, dict)
            or isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not 0.1 <= float(timeout_seconds) <= 900000.0
            or not isinstance(provider_cost_scope, str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", provider_cost_scope) is None
        ):
            raise RuntimeError("common runner session request is invalid")
        environment = dict(os.environ)
        environment["RESEARCH_LAB_PROVIDER_COST_SCOPE"] = provider_cost_scope
        if fork_enabled:
            returncode, stdout, stderr, timed_out = run_in_fresh_fork(
                operation,
                payload,
                float(timeout_seconds),
                provider_cost_scope,
            )
            completed = None
        else:
            try:
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        common_bootstrap,
                        module_name,
                        operation,
                    ],
                    input=json.dumps(
                        payload,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    text=True,
                    capture_output=True,
                    timeout=float(timeout_seconds),
                    env=environment,
                    check=False,
                )
                returncode = int(completed.returncode)
                stdout = str(completed.stdout or "")
                stderr = str(completed.stderr or "")
                timed_out = False
            except subprocess.TimeoutExpired as exc:
                stdout = exc.stdout
                stderr = exc.stderr
                if isinstance(stdout, bytes):
                    stdout = stdout.decode("utf-8", "replace")
                if isinstance(stderr, bytes):
                    stderr = stderr.decode("utf-8", "replace")
                returncode = None
                stdout = str(stdout or "")
                stderr = str(stderr or "")
                timed_out = True
        if timed_out:
            response = {
                "schema_version": schema_version,
                "request_id": request_id,
                "status": "timeout",
                "returncode": None,
                "stdout": str(stdout or "")[-8000:],
                "stderr": str(stderr or "")[-8000:],
            }
        else:
            response = {
                "schema_version": schema_version,
                "request_id": request_id,
                "status": (
                    "succeeded" if returncode == 0 else "failed"
                ),
                "returncode": int(returncode),
                "stdout": str(stdout or ""),
                "stderr": str(stderr or ""),
            }
    except BaseException as exc:
        response = {
            "schema_version": schema_version,
            "request_id": request_id,
            "status": "server_error",
            "error_class": type(exc).__name__,
            "error": str(exc)[:1200],
        }
    sys.stdout.write(json.dumps(
        response,
        sort_keys=True,
        separators=(",", ":"),
    ) + "\n")
    sys.stdout.flush()
"""


class _DockerModelRunnerSessionError(PrivateModelRuntimeError):
    """The reusable credential-free OCI process became unavailable."""


def _common_runner_session_capacity() -> int:
    """Use available CPUs without allowing an unbounded OCI session fanout."""

    affinity = getattr(os, "sched_getaffinity", None)
    if callable(affinity):
        try:
            processor_count = len(affinity(0))
        except (OSError, TypeError, ValueError):
            processor_count = 0
    else:
        processor_count = 0
    if processor_count <= 0:
        processor_count = os.cpu_count() or 1
    return max(
        1,
        min(_COMMON_RUNNER_MAX_PARALLEL_SESSIONS, int(processor_count)),
    )


class _DockerModelRunnerSession:
    """Reuse one immutable OCI container while keeping each call process-fresh."""

    def __init__(self, runner: DockerPrivateModelRunner) -> None:
        if not isinstance(runner, DockerPrivateModelRunner):
            raise PrivateModelRuntimeError("Docker model runner is required")
        self._runner = runner
        self._responses: queue.Queue[str | None] = queue.Queue()
        self._stderr_lock = threading.Lock()
        self._stderr_tail = ""
        self._call_lock = threading.Lock()
        self._closed = False
        self._container_name = (
            "leadpoet-private-model-session-" + uuid.uuid4().hex
        )
        self._process: subprocess.Popen[str] | None = None
        self._start()

    def _start(self) -> None:
        spec = self._runner.spec
        session_scope = sha256_json({
            "schema_version": _COMMON_RUNNER_SESSION_SCHEMA_VERSION,
            "image_digest": spec.image_digest,
            "module_name": spec.module_name,
        })
        command = [
            spec.docker_executable,
            "run",
            "--rm",
            "--name",
            self._container_name,
            "-i",
            *_docker_platform_args(spec),
            *(["--network", "none"] if spec.network_disabled else []),
            *_docker_env_args(spec),
            "-e",
            f"{PROVIDER_COST_EVALUATION_SCOPE_ENV}={session_scope}",
            spec.image_digest,
            "python",
            "-u",
            "-c",
            _COMMON_RUNNER_SESSION_BOOTSTRAP,
            _COMMON_RUNNER_BOOTSTRAP,
            spec.module_name,
        ]
        try:
            with _docker_private_model_lifecycle(spec) as deadline:
                self._process = subprocess.Popen(
                    command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    env=_build_docker_process_env(spec),
                )
                threading.Thread(
                    target=self._read_stdout,
                    name="common-runner-session-stdout",
                    daemon=True,
                ).start()
                threading.Thread(
                    target=self._read_stderr,
                    name="common-runner-session-stderr",
                    daemon=True,
                ).start()
                ready_line = self._next_response(
                    _docker_lifecycle_remaining_seconds(deadline)
                )
        except BaseException:
            self.close()
            raise
        try:
            ready = json.loads(ready_line)
        except (TypeError, json.JSONDecodeError) as exc:
            self.close()
            raise _DockerModelRunnerSessionError(
                "common model runner session returned invalid readiness"
            ) from exc
        if ready != {
            "schema_version": _COMMON_RUNNER_SESSION_SCHEMA_VERSION,
            "status": "ready",
        }:
            self.close()
            raise _DockerModelRunnerSessionError(
                "common model runner session readiness differs"
            )

    def _read_stdout(self) -> None:
        process = self._process
        stream = process.stdout if process is not None else None
        try:
            if stream is not None:
                for line in stream:
                    self._responses.put(line)
        finally:
            self._responses.put(None)

    def _read_stderr(self) -> None:
        process = self._process
        stream = process.stderr if process is not None else None
        if stream is None:
            return
        for line in stream:
            with self._stderr_lock:
                self._stderr_tail = (self._stderr_tail + line)[-8000:]

    def _next_response(self, timeout_seconds: float) -> str:
        try:
            value = self._responses.get(timeout=max(0.1, timeout_seconds))
        except queue.Empty as exc:
            raise _DockerModelRunnerSessionError(
                "common model runner session response timed out"
            ) from exc
        if value is None:
            with self._stderr_lock:
                stderr = _sanitize_text(self._stderr_tail)[-1200:]
            raise _DockerModelRunnerSessionError(
                "common model runner session exited before responding: "
                + stderr
            )
        return value

    def call(
        self,
        operation: str,
        payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        # The idle session may coexist with Docker maintenance, but one child
        # operation keeps the established shared lifecycle exclusion. If
        # maintenance removed an idle session, the transport replaces it once.
        with _docker_private_model_lifecycle(self._runner.spec):
            return self._call_with_lifecycle(operation, payload)

    def _call_with_lifecycle(
        self,
        operation: str,
        payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        with self._call_lock:
            if self._closed:
                raise _DockerModelRunnerSessionError(
                    "common model runner session is closed"
                )
            process = self._process
            stream = process.stdin if process is not None else None
            if process is None or process.poll() is not None or stream is None:
                raise _DockerModelRunnerSessionError(
                    "common model runner session is unavailable"
                )
            request_id = uuid.uuid4().hex
            timeout_seconds = max(
                0.1,
                float(self._runner.spec.timeout_seconds),
            )
            provider_cost_scope = sha256_json({
                "schema_version": _COMMON_RUNNER_SESSION_SCHEMA_VERSION,
                "image_digest": self._runner.spec.image_digest,
                "module_name": self._runner.spec.module_name,
                "operation": operation,
                "payload": dict(payload),
            })
            request = {
                "schema_version": _COMMON_RUNNER_SESSION_SCHEMA_VERSION,
                "request_id": request_id,
                "operation": operation,
                "payload": dict(payload),
                "timeout_seconds": timeout_seconds,
                "provider_cost_scope": provider_cost_scope,
            }
            try:
                stream.write(json.dumps(
                    request,
                    sort_keys=True,
                    separators=(",", ":"),
                ) + "\n")
                stream.flush()
            except (BrokenPipeError, OSError, ValueError) as exc:
                raise _DockerModelRunnerSessionError(
                    "common model runner session request failed"
                ) from exc
            response_line = self._next_response(timeout_seconds + 5.0)
            try:
                response = json.loads(response_line)
            except json.JSONDecodeError as exc:
                raise _DockerModelRunnerSessionError(
                    "common model runner session response is invalid"
                ) from exc
            if (
                not isinstance(response, Mapping)
                or response.get("schema_version")
                != _COMMON_RUNNER_SESSION_SCHEMA_VERSION
                or response.get("request_id") != request_id
            ):
                raise _DockerModelRunnerSessionError(
                    "common model runner session response identity differs"
                )
            status = response.get("status")
            if status == "server_error":
                if set(response) != {
                    "schema_version",
                    "request_id",
                    "status",
                    "error_class",
                    "error",
                }:
                    raise _DockerModelRunnerSessionError(
                        "common model runner session error response is invalid"
                    )
                detail = _sanitize_text(str(response.get("error") or ""))[-1200:]
                raise _DockerModelRunnerSessionError(
                    "common model runner session failed: " + detail
                )
            if set(response) != {
                "schema_version",
                "request_id",
                "status",
                "returncode",
                "stdout",
                "stderr",
            }:
                raise _DockerModelRunnerSessionError(
                    "common model runner session result is not closed"
                )
            stderr_text = _collect_incontainer_trace(
                str(response.get("stderr") or "")
            )
            if status == "timeout":
                raise PrivateModelRuntimeError(
                    "common model runner operation timed out"
                )
            if status != "succeeded" or response.get("returncode") != 0:
                stderr = _sanitize_text(stderr_text)[-1200:]
                raise PrivateModelRuntimeError(
                    "common model runner operation failed with code "
                    f"{response.get('returncode')}: {stderr}"
                )
            stdout = str(response.get("stdout") or "")
            try:
                result = _loads_adapter_stdout(stdout)
            except json.JSONDecodeError as exc:
                safe_stdout = _sanitize_text(stdout)[-800:]
                safe_stderr = _sanitize_text(stderr_text)[-800:]
                raise PrivateModelRuntimeError(
                    "common model runner returned invalid JSON: "
                    f"stdout={safe_stdout!r} stderr={safe_stderr!r}"
                ) from exc
            _raise_on_empty_provider_error(
                result,
                stderr_text,
                context_label="common model runner",
            )
            if not isinstance(result, Mapping):
                raise PrivateModelRuntimeError(
                    "common model runner returned a non-object response"
                )
            return dict(result)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        process = self._process
        self._process = None
        if process is None:
            return
        try:
            if process.stdin is not None:
                process.stdin.close()
        except (OSError, ValueError):
            pass
        try:
            process.wait(timeout=5)
            return
        except (OSError, subprocess.TimeoutExpired):
            pass
        _remove_private_model_container(
            self._runner.spec,
            container_name=self._container_name,
        )
        try:
            process.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            try:
                process.kill()
            except OSError:
                pass

    def __del__(self) -> None:
        try:
            self.close()
        except BaseException:
            pass


class DockerModelRunnerTransport:
    """Invoke only runner APIs, without forwarding provider credentials."""

    def __init__(self, runner: DockerPrivateModelRunner) -> None:
        if not isinstance(runner, DockerPrivateModelRunner):
            raise PrivateModelRuntimeError("Docker model runner is required")
        isolated_spec = replace(
            runner.spec,
            env_passthrough=(),
            extra_env={},
            pull_before_run=False,
        )
        self._runner = DockerPrivateModelRunner(isolated_spec)
        self._pool_capacity = _common_runner_session_capacity()
        self._pool_slots = threading.BoundedSemaphore(self._pool_capacity)
        self._pool_lock = threading.Lock()
        self._idle_sessions: list[_DockerModelRunnerSession] = []
        self._sessions: set[_DockerModelRunnerSession] = set()
        self._call_timing = threading.local()
        self._closed = False

    def _checkout_session(self) -> _DockerModelRunnerSession:
        with self._pool_lock:
            if self._closed:
                raise _DockerModelRunnerSessionError(
                    "common model runner transport is closed"
                )
            if self._idle_sessions:
                return self._idle_sessions.pop()
        session = _DockerModelRunnerSession(self._runner)
        with self._pool_lock:
            if self._closed:
                session.close()
                raise _DockerModelRunnerSessionError(
                    "common model runner transport is closed"
                )
            self._sessions.add(session)
        return session

    def _return_session(self, session: _DockerModelRunnerSession) -> None:
        close_session = False
        with self._pool_lock:
            if session not in self._sessions:
                return
            if self._closed:
                self._sessions.remove(session)
                close_session = True
            else:
                self._idle_sessions.append(session)
        if close_session:
            session.close()

    def _discard_session(self, session: _DockerModelRunnerSession) -> None:
        with self._pool_lock:
            if session in self._sessions:
                self._sessions.remove(session)
            if session in self._idle_sessions:
                self._idle_sessions.remove(session)
        session.close()

    def _call(
        self,
        operation: str,
        payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if re.fullmatch(r"[a-z][a-z0-9_]{0,95}", operation) is None:
            raise PrivateModelRuntimeError(
                "common model runner operation is invalid"
            )
        self._call_timing.execution_latency_ms = None
        self._pool_slots.acquire()
        session: _DockerModelRunnerSession | None = None
        execution_seconds = 0.0
        try:
            for session_attempt in range(2):
                session = self._checkout_session()
                execution_started = time.monotonic()
                try:
                    result = session.call(operation, payload)
                except _DockerModelRunnerSessionError:
                    self._discard_session(session)
                    session = None
                    if session_attempt:
                        raise
                except BaseException:
                    self._return_session(session)
                    session = None
                    raise
                else:
                    self._return_session(session)
                    session = None
                    return result
                finally:
                    execution_seconds += max(
                        0.0,
                        time.monotonic() - execution_started,
                    )
            raise PrivateModelRuntimeError(
                "common model runner session did not return"
            )
        finally:
            self._call_timing.execution_latency_ms = max(
                0,
                int(math.ceil(execution_seconds * 1000.0)),
            )
            if session is not None:
                self._return_session(session)
            self._pool_slots.release()

    def last_call_execution_latency_ms(self) -> int | None:
        """Return this thread's admitted OCI execution time for its last call.

        Pool admission can wait behind unrelated model operations for hours
        during a wide rebenchmark. It is scheduling delay, not execution
        latency for the verifier action persisted by the official authority.
        Thread-local storage keeps concurrent ICP measurements independent.
        """

        value = getattr(self._call_timing, "execution_latency_ms", None)
        if type(value) is not int or value < 0:
            return None
        return value

    def close(self) -> None:
        with self._pool_lock:
            if self._closed:
                return
            self._closed = True
        for _ in range(self._pool_capacity):
            self._pool_slots.acquire()
        try:
            with self._pool_lock:
                sessions = list(self._sessions)
                self._sessions.clear()
                self._idle_sessions.clear()
            for session in sessions:
                session.close()
        finally:
            for _ in range(self._pool_capacity):
                self._pool_slots.release()

    def __del__(self) -> None:
        try:
            self.close()
        except BaseException:
            pass

    def continue_runner(
        self,
        start_request: Mapping[str, Any],
        *,
        expected_release_identity: Mapping[str, Any],
        continuation: Mapping[str, Any] | None,
        completion: Mapping[str, Any] | None,
        member_name: str,
    ) -> Mapping[str, Any]:
        return self._call(
            "continue_runner",
            {
                "start_request": dict(start_request),
                "expected_release_identity": dict(
                    expected_release_identity
                ),
                "continuation": (
                    None if continuation is None else dict(continuation)
                ),
                "completion": (
                    None if completion is None else dict(completion)
                ),
                "member_name": member_name,
            },
        )

    def runner_protocol_generation(
        self,
        *,
        release_identity: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        return self._call(
            "runner_protocol_generation",
            {"release_identity": dict(release_identity)},
        )

    def build_raw_runner_input(
        self,
        payload: Mapping[str, Any],
        *,
        source_schema: str,
        member_name: str,
    ) -> Mapping[str, Any]:
        return self._call(
            "build_raw_runner_input",
            {
                "payload": dict(payload),
                "source_schema": source_schema,
                "member_name": member_name,
            },
        )

    def build_runner_start(
        self,
        *,
        input: Mapping[str, Any],
        execution_mode: str,
        target_count: int,
        evaluated_on: str,
        host_capability_manifest: Mapping[str, Any],
        release_identity: Mapping[str, Any],
        member_name: str,
    ) -> Mapping[str, Any]:
        return self._call(
            "build_runner_start",
            {
                "input": dict(input),
                "execution_mode": execution_mode,
                "target_count": target_count,
                "evaluated_on": evaluated_on,
                "host_capability_manifest": dict(
                    host_capability_manifest
                ),
                "release_identity": dict(release_identity),
                "member_name": member_name,
            },
        )

    def build_runner_completion(
        self,
        action: Mapping[str, Any],
        result: Mapping[str, Any],
        *,
        member_name: str,
    ) -> Mapping[str, Any]:
        return self._call(
            "build_runner_completion",
            {
                "action": dict(action),
                "result": dict(result),
                "member_name": member_name,
            },
        )

    def build_runner_provider_receipt_binding(
        self,
        action: Mapping[str, Any],
        result: Mapping[str, Any],
        *,
        member_name: str,
    ) -> Mapping[str, Any]:
        return self._call(
            "build_runner_provider_receipt_binding",
            {
                "action": dict(action),
                "result": dict(result),
                "member_name": member_name,
            },
        )

    def build_host_capability_manifest(
        self,
        *,
        bindings: Sequence[Mapping[str, Any]],
        member_name: str,
    ) -> Mapping[str, Any]:
        return self._call(
            "build_host_capability_manifest",
            {
                "bindings": [dict(value) for value in bindings],
                "member_name": member_name,
            },
        )

    def project_runner_result_for_benchmark(
        self,
        value: Mapping[str, Any],
        *,
        start_request: Mapping[str, Any],
        expected_release_identity: Mapping[str, Any],
        member_name: str,
    ) -> Mapping[str, Any]:
        return self._call(
            "project_runner_result_for_benchmark",
            {
                "value": dict(value),
                "start_request": dict(start_request),
                "expected_release_identity": dict(expected_release_identity),
                "member_name": member_name,
            },
        )

    def prepare_runner_provider_request(
        self,
        action: Mapping[str, Any],
        *,
        member_name: str,
    ) -> Mapping[str, Any]:
        return self._call(
            "prepare_runner_provider_request",
            {"action": dict(action), "member_name": member_name},
        )

    def ingest_runner_provider_response(
        self,
        action: Mapping[str, Any],
        host_response: Mapping[str, Any],
        *,
        member_name: str,
    ) -> Mapping[str, Any]:
        return self._call(
            "ingest_runner_provider_response",
            {
                "action": dict(action),
                "host_response": dict(host_response),
                "member_name": member_name,
            },
        )

    def prepare_runner_normalization_request(
        self,
        action: Mapping[str, Any],
        *,
        member_name: str,
    ) -> Mapping[str, Any]:
        return self._call(
            "prepare_runner_normalization_request",
            {"action": dict(action), "member_name": member_name},
        )

    def model_runner_provider_compiler_inventory(
        self,
        *,
        member_name: str,
    ) -> Mapping[str, Any]:
        return self._call(
            "model_runner_provider_compiler_inventory",
            {"member_name": member_name},
        )

    def runner_provider_compiler_preflight(
        self,
        host_capability_manifest: Mapping[str, Any],
        *,
        member_name: str,
    ) -> Mapping[str, Any]:
        return self._call(
            "runner_provider_compiler_preflight",
            {
                "host_capability_manifest": dict(host_capability_manifest),
                "member_name": member_name,
            },
        )

    def execute_runner_verifier_action(
        self,
        action: Mapping[str, Any],
        *,
        member_name: str,
    ) -> Mapping[str, Any]:
        return self._call(
            "execute_runner_verifier_action",
            {"action": dict(action), "member_name": member_name},
        )

    def runner_official_host_binding_catalog(
        self,
        *,
        member_name: str,
    ) -> Mapping[str, Any]:
        return self._call(
            "runner_official_host_binding_catalog",
            {"member_name": member_name},
        )

    def build_runner_official_host_capability_manifest(
        self,
        availability: Mapping[str, Any],
        *,
        member_name: str,
    ) -> Mapping[str, Any]:
        return self._call(
            "build_runner_official_host_capability_manifest",
            {
                "availability": dict(availability),
                "member_name": member_name,
            },
        )

    def build_official_baseline_execution(
        self,
        *,
        release_identity: Mapping[str, Any],
        protocol_generation_sha256: str,
        protected_action_authority_sha256: str,
        member_name: str,
    ) -> Mapping[str, Any]:
        return self._call(
            "build_official_baseline_execution",
            {
                "release_identity": dict(release_identity),
                "protocol_generation_sha256": protocol_generation_sha256,
                "protected_action_authority_sha256": (
                    protected_action_authority_sha256
                ),
                "member_name": member_name,
            },
        )

    def runner_preflight(
        self,
        *,
        host_capability_manifest: Mapping[str, Any],
        release_identity: Mapping[str, Any],
        execution_mode: str,
        member_name: str,
    ) -> Mapping[str, Any]:
        return self._call(
            "runner_preflight",
            {
                "host_capability_manifest": dict(host_capability_manifest),
                "release_identity": dict(release_identity),
                "execution_mode": execution_mode,
                "member_name": member_name,
            },
        )

    def validate_runner_preflight(
        self,
        value: Mapping[str, Any],
        *,
        host_capability_manifest: Mapping[str, Any],
        release_identity: Mapping[str, Any],
        execution_mode: str,
        member_name: str,
    ) -> Mapping[str, Any]:
        return self._call(
            "validate_runner_preflight",
            {
                "value": dict(value),
                "host_capability_manifest": dict(host_capability_manifest),
                "release_identity": dict(release_identity),
                "execution_mode": execution_mode,
                "member_name": member_name,
            },
        )

    def validate_runner_result(
        self,
        value: Mapping[str, Any],
        *,
        start_request: Mapping[str, Any],
        expected_release_identity: Mapping[str, Any],
        member_name: str,
    ) -> Mapping[str, Any]:
        return self._call(
            "validate_runner_result",
            {
                "value": dict(value),
                "start_request": dict(start_request),
                "expected_release_identity": dict(expected_release_identity),
                "member_name": member_name,
            },
        )


__all__ = ["DockerModelRunnerTransport"]
