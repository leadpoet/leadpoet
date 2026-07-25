#!/usr/bin/env python3.11
"""Strict boundary adapters for isolated restart testing.

The gateway and validator restart shell scripts execute unchanged. Privileged
external services may be adapted. A repository module, script, or long-lived
process that is substituted is recorded explicitly and invalidates a complete
restart rehearsal. Substitutions are permitted only in a clearly labelled
targeted regression run.
"""

from __future__ import annotations

import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tarfile
import time
from typing import Any, Iterable


STATE_ROOT = Path(os.environ.get("REHEARSAL_STATE_ROOT", "/rehearsal-state"))
STATE_PATH = STATE_ROOT / "state.json"
EVENT_PATH = STATE_ROOT / "events.jsonl"
LOCK_PATH = STATE_ROOT / "adapter.lock"
REAL_PYTHON = "/usr/bin/python3.11"
REAL_BASH = "/bin/bash"
PCR0 = hashlib.sha384(b"leadpoet-local-restart-rehearsal").hexdigest()
HASH64 = hashlib.sha256(b"leadpoet-local-restart-rehearsal").hexdigest()
ACCOUNT = "493765492819"
TARGETED_REGRESSION_SCOPE = "weight_readiness_regression"


def _ensure_state() -> None:
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    LOCK_PATH.touch(exist_ok=True)
    if not STATE_PATH.exists():
        STATE_PATH.write_text(
            json.dumps(
                {
                    "component": os.environ.get("REHEARSAL_COMPONENT", ""),
                    "candidate_sha": os.environ.get("REHEARSAL_CANDIDATE_SHA", ""),
                    "images": {},
                    "containers": {},
                    "enclaves": [],
                    "processes": {},
                    "docker_ready": True,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )


def _locked_state() -> tuple[io.TextIOWrapper, dict[str, Any]]:
    _ensure_state()
    handle = LOCK_PATH.open("r+", encoding="utf-8")
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    try:
        value = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        value = {}
    return handle, value


def _save_state(handle: io.TextIOWrapper, value: dict[str, Any]) -> None:
    temporary = STATE_PATH.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, STATE_PATH)
    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    handle.close()


def _event(kind: str, argv: Iterable[str], **details: Any) -> None:
    _ensure_state()
    if kind in {"aws", "curl", "docker", "nitro"}:
        details.setdefault("fixture_authenticity", "synthetic")
    payload = {
        "at_ns": time.time_ns(),
        "kind": kind,
        "argv": list(argv),
        **details,
    }
    line = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    descriptor = os.open(
        EVENT_PATH,
        os.O_WRONLY | os.O_CREAT | os.O_APPEND,
        0o600,
    )
    try:
        os.write(descriptor, line.encode("utf-8"))
    finally:
        os.close(descriptor)


def _fail(kind: str, argv: list[str], message: str) -> int:
    _event(kind, argv, status="rejected", reason=message)
    print(f"REHEARSAL CONTRACT ERROR [{kind}]: {message}: {argv!r}", file=sys.stderr)
    return 97


def _arg_value(argv: list[str], name: str, default: str = "") -> str:
    try:
        return argv[argv.index(name) + 1]
    except (ValueError, IndexError):
        return default


def _candidate_sha() -> str:
    configured = os.environ.get("REHEARSAL_CANDIDATE_SHA", "").strip()
    if re.fullmatch(r"[0-9a-f]{40}", configured):
        return configured
    repo = Path("/home/ec2-user/leadpoet_repo")
    if not repo.exists():
        repo = Path("/home/ec2-user/leadpoet/leadpoet")
    result = subprocess.run(
        ["/usr/bin/git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _rehearsal_scope() -> str:
    return os.environ.get("REHEARSAL_SCOPE", "exact").strip()


def _targeted_substitutions_allowed() -> bool:
    return _rehearsal_scope() == TARGETED_REGRESSION_SCOPE


def _candidate_root() -> Path:
    gateway_root = Path("/home/ec2-user/leadpoet_repo")
    if gateway_root.is_dir():
        return gateway_root
    validator_root = Path("/home/ec2-user/leadpoet/leadpoet")
    if validator_root.is_dir():
        return validator_root
    raise RuntimeError("candidate checkout is unavailable")


def _candidate_git_path(resolved: Path, root: Path) -> tuple[Path, str]:
    if resolved == root or root in resolved.parents:
        return resolved.relative_to(root), "candidate_checkout"

    for parent in resolved.parents:
        if (
            parent.parent == Path("/tmp")
            and re.fullmatch(r"gateway-v2-preflight\.[A-Za-z0-9]+", parent.name)
        ):
            return resolved.relative_to(parent), "candidate_archive"

    raise RuntimeError(
        "production source is outside the candidate checkout or a recognized "
        f"candidate archive: {resolved} (candidate root: {root})"
    )


def _source_identity(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    root = _candidate_root().resolve()
    if not resolved.is_file():
        raise RuntimeError(f"production source is unavailable: {resolved}")
    relative, source_kind = _candidate_git_path(resolved, root)
    candidate_sha = _candidate_sha()
    result = subprocess.run(
        [
            "/usr/bin/git",
            "-C",
            str(root),
            "show",
            f"{candidate_sha}:{relative.as_posix()}",
        ],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "production source is not present at the frozen candidate SHA: "
            f"{relative.as_posix()}"
        )
    source_bytes = resolved.read_bytes()
    if source_bytes != result.stdout:
        raise RuntimeError(
            "production source bytes differ from the frozen candidate SHA: "
            f"{relative.as_posix()}"
        )
    return {
        "candidate_sha": candidate_sha,
        "source_path": str(resolved),
        "source_git_path": relative.as_posix(),
        "source_kind": source_kind,
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
    }


def _module_source(module: str) -> Path:
    root = _candidate_root()
    module_path = root.joinpath(*module.split(".")).with_suffix(".py")
    if module_path.is_file():
        return module_path
    package_main = root.joinpath(*module.split("."), "__main__.py")
    if package_main.is_file():
        return package_main
    raise RuntimeError("candidate module source is unavailable: %s" % module)


def _record_production_module(module: str, argv: list[str]) -> None:
    _event(
        "python-module",
        argv,
        status="started",
        module=module,
        implementation="production_module",
        **_source_identity(_module_source(module)),
    )


def _record_production_script(path: Path, argv: list[str]) -> None:
    _event(
        "python-script",
        argv,
        status="started",
        script=path.name,
        implementation="production_script",
        **_source_identity(path),
    )


def _record_internal_substitution(
    *,
    kind: str,
    argv: list[str],
    module: str = "",
    script: str = "",
    process: str = "",
    substitution: str = "",
) -> int:
    details = {
        "status": "substituted",
        "implementation": "internal_substitution",
        "scope": _rehearsal_scope(),
    }
    if module:
        details["module"] = module
    if script:
        details["script"] = script
    if process:
        details["process"] = process
    if substitution:
        details["substitution"] = substitution
    _event(kind, argv, **details)
    if _targeted_substitutions_allowed():
        return 0
    return _fail(
        kind,
        argv,
        "repository implementation substitution invalidates exact rehearsal",
    )


def _write_json(path: str | Path, value: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(value, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _gateway_secret() -> dict[str, str]:
    return {
        "AWS_REGION": "us-east-1",
        "AWS_DEFAULT_REGION": "us-east-1",
        "GITHUB_REPO_URL": "/srv/origin.git",
        "GITHUB_BRANCH": "main",
        "SUPABASE_URL": "https://example.invalid",
        "SUPABASE_ANON_KEY": "rehearsal-public",
        "SUPABASE_SERVICE_ROLE_KEY": "rehearsal-secret",
        "OPENROUTER_API_KEY": "rehearsal-openrouter",
        "EXA_API_KEY": "rehearsal-exa",
        "SCRAPINGDOG_API_KEY": "rehearsal-scrapingdog",
        "DEEPLINE_API_KEY": "rehearsal-deepline",
        "TRUELIST_API_KEY": "rehearsal-truelist",
        "RESEARCH_LAB_TEE_PROTOCOL": "v2",
        "GATEWAY_PYTHON_BIN": "/home/ec2-user/venv311/bin/python3",
        "GATEWAY_PRIVATE_KEY_PATH": (
            "/home/ec2-user/gateway/secrets/gateway_private_key.pem"
        ),
        "ARWEAVE_KEYFILE_PATH": (
            "/home/ec2-user/gateway/secrets/arweave_keyfile.json"
        ),
        "GATEWAY_TEE_TOPOLOGY_MODE": "full",
        "GATEWAY_TEE_ROLE_READY_TIMEOUT_SECONDS": "5",
        "GATEWAY_TEE_ROLE_READY_RETRY_SECONDS": "1",
        "NO_PROXY": "127.0.0.1,localhost",
    }


def _validator_secret() -> dict[str, str]:
    values = {
        "ENABLE_FULFILLMENT": "true",
        "ENABLE_QUALIFICATION_EVALUATION": "true",
        "LEADPOET_WRAPPER_ACTIVE": "1",
        "GATEWAY_URL": "http://gateway.invalid:8000",
        "VALIDATOR_V2_GATEWAY_URL": "http://gateway.invalid:8000",
        "SUPABASE_URL": "https://example.invalid",
        "SUPABASE_ANON_KEY": "rehearsal-public",
        "SUPABASE_SERVICE_ROLE_KEY": "rehearsal-secret",
        "OPENROUTER_API_KEY": "rehearsal-openrouter",
        "QUALIFICATION_OPENROUTER_API_KEY": "rehearsal-openrouter",
        "FULFILLMENT_OPENROUTER_API_KEY": "rehearsal-openrouter",
        "EXA_API_KEY": "rehearsal-exa",
        "SCRAPINGDOG_API_KEY": "rehearsal-scrapingdog",
        "QUALIFICATION_SCRAPINGDOG_API_KEY": "rehearsal-scrapingdog",
        "AWS_REGION": "us-east-1",
        "AWS_DEFAULT_REGION": "us-east-1",
        "RESEARCH_LAB_VALIDATOR_FETCH_ENABLED": "true",
        "RESEARCH_LAB_VALIDATOR_SHADOW_VERIFY_ENABLED": "true",
        "RESEARCH_LAB_VALIDATOR_EVALUATION_VERIFY_ENABLED": "true",
        "RESEARCH_LAB_REQUIRE_SHADOW_VERIFICATION_BEFORE_SUBMIT": "true",
        "RESEARCH_LAB_REQUIRE_EVALUATION_VERIFICATION_BEFORE_SUBMIT": "true",
        "RESEARCH_LAB_INTERNAL_API_KEY": "rehearsal-internal",
        "RESEARCH_LAB_SCORE_BUNDLE_KMS_KEY_ID": "rehearsal-kms",
        "RESEARCH_LAB_WEIGHT_MUTATION_ENABLED": "true",
        "RESEARCH_LAB_SUBMIT_ON_CHAIN_ENABLED": "true",
        "QUALIFICATION_WEBSHARE_PROXY_1": "http://proxy.invalid",
        "EXPECTED_CHAIN": "wss://entrypoint-finney.opentensor.ai:443",
        "NO_PROXY": "127.0.0.1,localhost",
        "VALIDATOR_WEIGHT_PROTOCOL": "authoritative_v2",
    }
    return values


def command_aws(argv: list[str]) -> int:
    if argv[:2] == ["secretsmanager", "get-secret-value"]:
        component = os.environ.get("REHEARSAL_COMPONENT", "")
        secret = _gateway_secret() if component == "gateway" else _validator_secret()
        _event("aws", argv, status="ok", operation="secretsmanager")
        print(json.dumps(secret, sort_keys=True))
        return 0
    if argv[:2] == ["sts", "get-caller-identity"]:
        _event("aws", argv, status="ok", operation="sts")
        print(ACCOUNT)
        return 0
    if argv[:2] == ["ecr", "get-login-password"]:
        _event("aws", argv, status="ok", operation="ecr_login")
        print("rehearsal-ecr-password")
        return 0
    return _fail("aws", argv, "unknown AWS operation")


def _image_id(name: str) -> str:
    return "sha256:" + hashlib.sha256(name.encode("utf-8")).hexdigest()


def _docker_save(path: Path) -> None:
    layer_bytes = b"leadpoet restart rehearsal layer\n"
    layer_buffer = io.BytesIO()
    with tarfile.open(fileobj=layer_buffer, mode="w") as layer_tar:
        info = tarfile.TarInfo("rehearsal.txt")
        info.size = len(layer_bytes)
        info.mtime = 0
        layer_tar.addfile(info, io.BytesIO(layer_bytes))
    layer_data = layer_buffer.getvalue()
    layer_hash = hashlib.sha256(layer_data).hexdigest()
    config = {
        "created": "1970-01-01T00:00:00Z",
        "rootfs": {"type": "layers", "diff_ids": [f"sha256:{layer_hash}"]},
        "history": [{"created": "1970-01-01T00:00:00Z"}],
    }
    config_data = json.dumps(config, separators=(",", ":")).encode("utf-8")
    config_hash = hashlib.sha256(config_data).hexdigest()
    manifest = [
        {
            "Config": f"blobs/sha256/{config_hash}",
            "RepoTags": ["validator-tee-enclave:raw"],
            "Layers": [f"blobs/sha256/{layer_hash}"],
        }
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w") as archive:
        for name, data in (
            ("manifest.json", json.dumps(manifest).encode("utf-8")),
            (f"blobs/sha256/{config_hash}", config_data),
            (f"blobs/sha256/{layer_hash}", layer_data),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mtime = 0
            archive.addfile(info, io.BytesIO(data))


def command_docker(argv: list[str]) -> int:
    handle, state = _locked_state()
    images = state.setdefault("images", {})
    containers = state.setdefault("containers", {})
    operation = " ".join(argv[:3])
    status = 0
    output = ""
    try:
        if not argv:
            return _fail("docker", argv, "missing Docker operation")
        if argv[0] == "info":
            if not state.get("docker_ready", True):
                status = 1
            elif "--format" in argv:
                output = "/var/lib/docker"
            else:
                output = "Server Version: rehearsal"
        elif argv[:2] in (["images", "-q"], ["image", "ls"]):
            target = argv[-1] if argv[:2] == ["images", "-q"] and len(argv) > 2 else ""
            if target:
                output = images.get(target, "")
            else:
                output = "\n".join(sorted(set(images.values())))
        elif argv[:2] == ["image", "inspect"]:
            target = argv[-1]
            output = images.get(target, _image_id(target))
        elif argv[0] == "inspect":
            target = argv[-1]
            row = containers.get(target, {})
            template = _arg_value(argv, "-f")
            if ".State.Running" in template:
                output = "true" if row.get("running") else "false"
            elif ".RestartCount" in template:
                output = str(row.get("restart_count", 0))
            else:
                output = json.dumps([row])
        elif argv[0] == "build":
            tag = _arg_value(argv, "-t")
            if not tag:
                return _fail("docker", argv, "Docker build omitted -t")
            images[tag] = _image_id(tag)
        elif argv[0] in {"rmi", "rm", "stop"}:
            for item in argv[1:]:
                if item.startswith("-"):
                    continue
                if argv[0] == "rmi":
                    images.pop(item, None)
                elif argv[0] == "stop" and item in containers:
                    containers[item]["running"] = False
                elif argv[0] == "rm":
                    containers.pop(item, None)
        elif argv[:2] in (
            ["container", "prune"],
            ["builder", "prune"],
            ["system", "prune"],
            ["image", "prune"],
        ):
            output = "rehearsal prune complete"
        elif argv[:2] == ["system", "df"]:
            output = "TYPE TOTAL ACTIVE SIZE RECLAIMABLE"
        elif argv[:2] == ["volume", "ls"]:
            output = ""
        elif argv[0] == "ps":
            output = ""
        elif argv[0] == "login":
            sys.stdin.read()
            output = "Login Succeeded"
        elif argv[0] == "save":
            destination = _arg_value(argv, "-o")
            if not destination:
                return _fail("docker", argv, "Docker save omitted -o")
            _docker_save(Path(destination))
        elif argv[0] == "load":
            images["validator-tee-enclave:latest"] = _image_id(
                "validator-tee-enclave:latest"
            )
            output = "Loaded image: validator-tee-enclave:latest"
        elif argv[0] == "tag":
            if len(argv) < 3:
                return _fail("docker", argv, "Docker tag is incomplete")
            images[argv[2]] = _image_id(argv[2])
        elif argv[0] == "run":
            if "-c" in argv:
                output = "sha256:" + HASH64
            else:
                output = ""
        elif argv[0] == "logs":
            output = "rehearsal validator container log"
        else:
            return _fail("docker", argv, "unknown Docker operation")
        _event("docker", argv, status="ok" if status == 0 else "failed", operation=operation)
        if output:
            print(output)
        return status
    finally:
        _save_state(handle, state)


def command_nitro(argv: list[str]) -> int:
    handle, state = _locked_state()
    enclaves = state.setdefault("enclaves", [])
    try:
        if argv[:2] == ["terminate-enclave", "--all"]:
            enclaves.clear()
            _event("nitro", argv, status="ok", operation="terminate")
            return 0
        if argv and argv[0] == "build-enclave":
            output = _arg_value(argv, "--output-file")
            image = _arg_value(argv, "--docker-uri")
            if not output or not image:
                return _fail("nitro", argv, "build-enclave arguments are incomplete")
            destination = Path(output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(b"leadpoet-rehearsal-eif\n")
            _event("nitro", argv, status="ok", operation="build")
            print(
                json.dumps(
                    {
                        "Measurements": {
                            "PCR0": PCR0,
                            "PCR1": hashlib.sha384(b"pcr1").hexdigest(),
                            "PCR2": hashlib.sha384(b"pcr2").hexdigest(),
                        }
                    },
                    sort_keys=True,
                )
            )
            return 0
        if argv and argv[0] == "run-enclave":
            cid = _arg_value(argv, "--enclave-cid", "16")
            eif = _arg_value(argv, "--eif-path")
            if not eif or not Path(eif).is_file():
                return _fail("nitro", argv, "run-enclave EIF is unavailable")
            row = {
                "EnclaveCID": int(cid),
                "EnclaveID": f"rehearsal-{cid}",
                "State": "RUNNING",
                "Measurements": {"PCR0": PCR0},
            }
            enclaves.append(row)
            _event("nitro", argv, status="ok", operation="run")
            print(json.dumps(row, sort_keys=True))
            return 0
        if argv and argv[0] == "describe-eif":
            eif = _arg_value(argv, "--eif-path")
            if not eif or not Path(eif).is_file():
                return _fail("nitro", argv, "describe-eif EIF is unavailable")
            _event("nitro", argv, status="ok", operation="describe-eif")
            print(
                json.dumps(
                    {
                        "Measurements": {
                            "PCR0": PCR0,
                            "PCR1": hashlib.sha384(b"pcr1").hexdigest(),
                            "PCR2": hashlib.sha384(b"pcr2").hexdigest(),
                        }
                    },
                    sort_keys=True,
                )
            )
            return 0
        if argv and argv[0] == "describe-enclaves":
            _event("nitro", argv, status="ok", operation="describe")
            print(json.dumps(enclaves, sort_keys=True))
            return 0
        return _fail("nitro", argv, "unknown Nitro operation")
    finally:
        _save_state(handle, state)


def command_systemctl(argv: list[str]) -> int:
    accepted = {"start", "stop", "restart", "reset-failed", "is-active"}
    if not argv or argv[0] not in accepted:
        return _fail("systemctl", argv, "unknown systemctl operation")
    if _record_internal_substitution(
        kind="systemctl",
        argv=argv,
        substitution="host.systemd",
    ) != 0:
        return 97
    _event("systemctl", argv, status="ok")
    return 0


def command_curl(argv: list[str]) -> int:
    output_path = _arg_value(argv, "--output") or _arg_value(argv, "-o")
    urls = [arg for arg in argv if re.match(r"^https?://", arg)]
    if len(urls) != 1:
        return _fail("curl", argv, "curl must contain exactly one URL")
    url = urls[0]
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"rehearsal-downloaded-artifact\n")
        _event("curl", argv, status="ok", operation="download", url=url)
        return 0
    if url.startswith(("http://localhost", "http://127.0.0.1")):
        if _record_internal_substitution(
            kind="curl",
            argv=argv,
            substitution="http.local_gateway",
        ) != 0:
            return 97
    if url.endswith("/build-info"):
        print(json.dumps({"git_commit": _candidate_sha()}, sort_keys=True))
    elif url.endswith("/health/v2-authority"):
        print(
            json.dumps(
                {
                    "schema_version": "leadpoet.gateway_v2_authority_health.v2",
                    "status": "ready",
                    "commit_sha": _candidate_sha(),
                },
                sort_keys=True,
            )
        )
    elif re.search(r"/weights/v2/release-evidence/[0-9a-f]{40}$", url):
        print(
            json.dumps(
                {
                    "schema_version": "leadpoet.auditor_release_evidence.v2",
                    "commit_sha": _candidate_sha(),
                    "release_channel_version_id": "rehearsal-version",
                    "release_channel_get_url": "https://release.invalid/get",
                    "release_channel_head_url": "https://release.invalid/head",
                },
                sort_keys=True,
            )
        )
    elif url.endswith(("/health", "/research-lab/status", "/attest")):
        print(json.dumps({"status": "ok"}, sort_keys=True))
    else:
        return _fail("curl", argv, "unknown HTTP endpoint")
    _event("curl", argv, status="ok", operation="http", url=url)
    return 0


def command_sudo(argv: list[str]) -> int:
    while argv and argv[0].startswith("-"):
        argv = argv[1:]
    if not argv:
        return _fail("sudo", argv, "sudo command is missing")
    _event("sudo", argv, status="delegated")
    os.execvpe(argv[0], argv, os.environ.copy())
    return 127


def command_df(argv: list[str]) -> int:
    if _record_internal_substitution(
        kind="host-command",
        argv=argv,
        substitution="host.filesystem_capacity",
    ) != 0:
        return 97
    if any("output=avail" in arg for arg in argv):
        print("Avail")
        print("107374182400" if "-B1" in argv else "104857600")
        return 0
    print("Filesystem Size Used Avail Use% Mounted on")
    print("rehearsal 120G 1G 119G 1% /")
    return 0


def command_getconf(argv: list[str]) -> int:
    if _record_internal_substitution(
        kind="host-command",
        argv=argv,
        substitution="host.cpu_capacity",
    ) != 0:
        return 97
    if argv == ["_NPROCESSORS_CONF"]:
        print("16")
        return 0
    return _fail("getconf", argv, "unknown getconf query")


def command_awk(argv: list[str]) -> int:
    if argv and argv[-1] == "/proc/meminfo" and "MemTotal" in " ".join(argv):
        if _record_internal_substitution(
            kind="host-command",
            argv=argv,
            substitution="host.memory_capacity",
        ) != 0:
            return 97
        print("131072")
        return 0
    os.execv("/usr/bin/awk", ["awk", *argv])
    return 127


def command_sleep(argv: list[str]) -> int:
    if _record_internal_substitution(
        kind="host-command",
        argv=argv,
        substitution="host.timing",
    ) != 0:
        return 97
    _event("sleep", argv, status="shortened")
    time.sleep(0.01)
    return 0


def command_ss(argv: list[str]) -> int:
    if _record_internal_substitution(
        kind="host-command",
        argv=argv,
        substitution="host.socket_state",
    ) != 0:
        return 97
    _event("ss", argv, status="ok")
    return 0


def command_ctr(argv: list[str]) -> int:
    allowed_tokens = {"containers", "tasks", "namespaces", "list", "-q", "-n", "moby"}
    if any(item not in allowed_tokens for item in argv):
        return _fail("ctr", argv, "unknown containerd operation")
    if _record_internal_substitution(
        kind="host-command",
        argv=argv,
        substitution="host.containerd_state",
    ) != 0:
        return 97
    _event("ctr", argv, status="ok")
    return 0


def command_nsenter(argv: list[str]) -> int:
    if "--" not in argv:
        return _fail("nsenter", argv, "nsenter omitted --")
    command = argv[argv.index("--") + 1 :]
    if not command:
        return _fail("nsenter", argv, "nsenter command is empty")
    if _record_internal_substitution(
        kind="host-command",
        argv=argv,
        substitution="host.mount_namespace",
    ) != 0:
        return 97
    _event("nsenter", argv, status="delegated")
    os.execvpe(command[0], command, os.environ.copy())
    return 127


def command_pgrep(argv: list[str]) -> int:
    if _record_internal_substitution(
        kind="host-command",
        argv=argv,
        substitution="host.process_lookup",
    ) != 0:
        return 97
    pattern = argv[-1] if argv else ""
    handle, state = _locked_state()
    try:
        if "containerd-shim-runc-v2" in pattern:
            if "-c" in "".join(argv) or "-fc" in argv:
                print("0")
                return 0
            return 1
        process_key = ""
        if "gateway[.]main" in pattern or "gateway.main" in pattern:
            process_key = "gateway.main"
        elif "chain_relay_v2" in pattern:
            process_key = "validator.chain_relay"
        pid = state.get("processes", {}).get(process_key)
        if pid and Path(f"/proc/{pid}").exists():
            print(pid)
            return 0
        return 1
    finally:
        _save_state(handle, state)


def command_pkill(argv: list[str]) -> int:
    if _record_internal_substitution(
        kind="host-command",
        argv=argv,
        substitution="host.process_termination",
    ) != 0:
        return 97
    pattern = argv[-1] if argv else ""
    handle, state = _locked_state()
    try:
        processes = state.setdefault("processes", {})
        for key, pid in list(processes.items()):
            if key in pattern or ("gateway" in key and "gateway" in pattern):
                try:
                    os.kill(int(pid), signal.SIGTERM)
                except (ProcessLookupError, ValueError):
                    pass
                processes.pop(key, None)
        _event("pkill", argv, status="ok")
        return 0
    finally:
        _save_state(handle, state)


def _long_lived_process(key: str, argv: list[str]) -> int:
    if _record_internal_substitution(
        kind="process",
        argv=argv,
        process=key,
    ) != 0:
        return 97
    handle, state = _locked_state()
    state.setdefault("processes", {})[key] = os.getpid()
    _save_state(handle, state)
    _event(
        "process",
        argv,
        status="started",
        process=key,
        pid=os.getpid(),
        implementation="internal_substitution",
        scope=_rehearsal_scope(),
    )

    def stop(_signum: int, _frame: Any) -> None:
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    while True:
        time.sleep(60)


def _release_manifest(role: str) -> dict[str, Any]:
    return {
        "schema_version": f"leadpoet.{role}_release_manifest.v2",
        "commit_sha": _candidate_sha(),
        "pcr0": PCR0,
        "release_hash": "sha256:" + HASH64,
        "release_manifest_hash": "sha256:" + HASH64,
        "verified_build_count": 6,
    }


def _module_release_channel(argv: list[str]) -> int:
    expected = _arg_value(argv, "--expected-commit")
    if expected != _candidate_sha():
        return _fail("python-module", argv, "release expected commit differs")
    gateway_output = _arg_value(argv, "--gateway-output")
    validator_output = _arg_value(argv, "--validator-output")
    lineage_output = _arg_value(argv, "--lineage-output")
    if gateway_output:
        _write_json(gateway_output, _release_manifest("gateway"))
    if validator_output:
        _write_json(validator_output, _release_manifest("validator"))
    if lineage_output:
        _write_json(
            lineage_output,
            {
                "schema_version": "leadpoet.gateway_release_lineage.v2",
                "commit_sha": expected,
                "lineage_hash": "sha256:" + HASH64,
                "releases": [{"commit_sha": expected}],
            },
        )
    print(json.dumps({"status": "local_verified", "commit_sha": expected}))
    return 0


def _module_restart_gate(argv: list[str]) -> int:
    capture = _arg_value(argv, "--capture-output")
    report = {
        "schema_version": "leadpoet.restart_epoch_start.v1",
        "maximum_restart_epoch_block": 300,
        "restart_allowed": True,
        "snapshot": {
            "netuid": 71,
            "epoch_id": 99999,
            "epoch_block": 42,
            "tempo": 360,
            "block_hash": "0x" + "1" * 64,
        },
    }
    if capture:
        _write_json(capture, report)
    captured = _arg_value(argv, "--captured-report")
    if captured and not Path(captured).is_file():
        return _fail("python-module", argv, "captured restart report is missing")
    print(json.dumps(report, sort_keys=True))
    return 0


def _module_envelopes(argv: list[str]) -> int:
    output_dir = Path(_arg_value(argv, "--output-dir"))
    deploy_commit = _arg_value(argv, "--deploy-commit")
    if deploy_commit != _candidate_sha():
        return _fail("python-module", argv, "envelope commit differs")
    output_dir.mkdir(parents=True, exist_ok=True)
    names = (
        "artifact_master_key",
        "openrouter",
        "exa",
        "scrapingdog",
        "deepline",
        "supabase_service_role",
        "truelist",
    )
    for name in names:
        _write_json(
            output_dir / f"{name}.json",
            {
                "schema_version": "leadpoet.kms_credential_envelope.v2",
                "deploy_commit": deploy_commit,
                "ciphertext_b64": "cmVoZWFyc2Fs",
                "credential_reference_hash": "sha256:" + HASH64,
            },
        )
    corpus = output_dir / "acceptance-corpus-v2"
    corpus.mkdir(parents=True, exist_ok=True)
    (corpus / "fixture.json").write_text("{}\n", encoding="utf-8")
    _write_json(
        output_dir / "acceptance-corpus-v2.json",
        {
            "schema_version": "leadpoet.acceptance_corpus.v2",
            "deploy_commit": deploy_commit,
            "corpus_hash": "sha256:" + HASH64,
        },
    )
    _write_json(
        output_dir / "gateway-v2-env-transition.json",
        {"schema_version": "leadpoet.gateway_env_transition.v2", "status": "ready"},
    )
    print(json.dumps({"status": "installed", "deploy_commit": deploy_commit}))
    return 0


def _module_stage_artifacts(argv: list[str]) -> int:
    output = _arg_value(argv, "--output-dir")
    if not output:
        return _fail("python-module", argv, "artifact output directory is missing")
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    lock_path = _arg_value(argv, "--lock")
    if lock_path and Path(lock_path).is_file():
        try:
            lock = json.loads(Path(lock_path).read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            lock = {}
        for row in (lock.get("artifacts") or {}).values():
            filename = str(row.get("filename") or "")
            if filename:
                (destination / filename).write_bytes(b"rehearsal-runtime-artifact\n")
    print(json.dumps({"status": "staged", "output_dir": str(destination)}))
    return 0


def _module_weight_ready(argv: list[str]) -> int:
    if "--repair" not in argv and not _arg_value(argv, "--gateway-url"):
        return _fail("python-module", argv, "weight readiness mode is unknown")
    result = subprocess.run(
        [
            REAL_PYTHON,
            "/harness/weight_readiness_runner.py",
            *argv,
        ],
        check=False,
    )
    return int(result.returncode)


def _scrub_parent_env(argv: list[str]) -> int:
    if len(argv) < 2:
        return _fail("python-inline", argv, "scrub-parent-env arguments are missing")
    env_path = Path(argv[0])
    report_path = Path(argv[1])
    secret_names = {
        "OPENROUTER_API_KEY",
        "EXA_API_KEY",
        "SCRAPINGDOG_API_KEY",
        "DEEPLINE_API_KEY",
        "TRUELIST_API_KEY",
    }
    kept = []
    for line in env_path.read_text(encoding="utf-8").splitlines():
        candidate = line.removeprefix("export ").strip()
        key = candidate.split("=", 1)[0] if "=" in candidate else ""
        if key not in secret_names:
            kept.append(line)
    env_path.write_text("\n".join(kept) + "\n", encoding="utf-8")
    _write_json(
        report_path,
        {"schema_version": "leadpoet.gateway_env_transition.v2", "status": "scrubbed"},
    )
    print("Scrubbed commit-bound provider plaintext from prepared parent environment")
    return 0


def _python_inline(argv: list[str]) -> int:
    source = sys.stdin.read()
    if "scrub_parent_environment_file_v2" in source:
        if _record_internal_substitution(
            kind="python-inline",
            argv=argv,
            substitution="python.scrub_parent_environment",
        ) != 0:
            return 97
        return _scrub_parent_env(argv[1:])
    _event("python-inline", argv, status="real")
    result = subprocess.run([REAL_PYTHON, *argv], input=source, text=True, check=False)
    return result.returncode


def _python_script(argv: list[str]) -> int | None:
    path = Path(argv[0])
    name = path.name
    if path.resolve() == Path("/tmp/get-pip.py"):
        expected = b"rehearsal-downloaded-artifact\n"
        if not path.is_file() or path.read_bytes() != expected:
            return _fail(
                "python-script",
                argv,
                "downloaded get-pip.py does not match the strict rehearsal fixture",
            )
        if _record_internal_substitution(
            kind="python-script",
            argv=argv,
            script=name,
            substitution="python_dependencies.bootstrap",
        ) != 0:
            return 97
        return 0
    if name in {"gateway_git_deploy.py", "write_gateway_build_info.py"}:
        return None
    if name == "host_memory_guard_v2.py":
        if _record_internal_substitution(
            kind="python-script",
            argv=argv,
            script=name,
        ) != 0:
            return 97
        print(
            json.dumps(
                {
                    "schema_version": "leadpoet.gateway_host_memory_guard.v2",
                    "status": "ready",
                    "available_memory_mib": 32768,
                    "minimum_available_memory_mib": 16384,
                    "cleaned_disposable_tests": [],
                    "top_processes": [],
                },
                sort_keys=True,
            )
        )
        return 0
    if name == "scoring_wheelhouse.py":
        if _record_internal_substitution(
            kind="python-script",
            argv=argv,
            script=name,
        ) != 0:
            return 97
        print(json.dumps({"status": "verified", "script": name}))
        return 0
    if name == "sandbox_runtime_artifact.py":
        if _record_internal_substitution(
            kind="python-script",
            argv=argv,
            script=name,
        ) != 0:
            return 97
        operation = argv[1] if len(argv) > 1 else ""
        lock_path = Path(_arg_value(argv, "--lock"))
        if operation == "verify":
            artifact_path = Path(_arg_value(argv, "--artifact"))
            if not lock_path.is_file():
                return _fail(
                    "python-script",
                    argv,
                    "sandbox runtime lock is unavailable",
                )
            if not artifact_path.is_file():
                _event(
                    "python-script",
                    argv,
                    status="failed",
                    script=name,
                    reason="artifact_missing",
                )
                return 1
            _event(
                "python-script",
                argv,
                status="ok",
                script=name,
                operation=operation,
            )
            print(json.dumps({"status": "verified", "script": name}))
            return 0
        return _fail(
            "python-script",
            argv,
            "unknown sandbox runtime artifact operation",
        )
    if name in {
        "verify_release_artifacts_v2.py",
        "verify_topology.py",
        "docker_image_normalizer_v2.py",
        "release_manifest_v2.py",
    }:
        if _record_internal_substitution(
            kind="python-script",
            argv=argv,
            script=name,
        ) != 0:
            return 97
        if name == "docker_image_normalizer_v2.py":
            normalized = _arg_value(argv, "--normalized-image")
            handle, state = _locked_state()
            state.setdefault("images", {})[normalized] = _image_id(normalized)
            _save_state(handle, state)
        _event("python-script", argv, status="ok", script=name)
        print(json.dumps({"status": "verified", "script": name}))
        return 0
    if name == "stage_runtime_artifacts_v2.py":
        if _record_internal_substitution(
            kind="python-script",
            argv=argv,
            script=name,
        ) != 0:
            return 97
        return _module_stage_artifacts(argv[1:])
    return None


def command_python(argv: list[str]) -> int:
    if not argv:
        return _python_inline(argv)
    if argv[0] == "-u":
        if len(argv) == 1:
            return _fail("python", argv, "-u omitted a Python operation")
        return command_python(argv[1:])
    if argv[0] == "-":
        return _python_inline(argv)
    if argv[0] == "-m":
        if len(argv) < 2:
            return _fail("python", argv, "-m omitted a module")
        module = argv[1]
        module_argv = argv[2:]
        if module == "pip":
            if module_argv and module_argv[0] == "download":
                if _record_internal_substitution(
                    kind="python-dependencies",
                    argv=module_argv,
                    substitution="python_dependencies.download",
                ) != 0:
                    return 97
                destination_value = _arg_value(module_argv, "--dest")
                if not destination_value:
                    return _fail("pip", module_argv, "pip download omitted --dest")
                destination = Path(destination_value)
                destination.mkdir(parents=True, exist_ok=True)
                (destination / "restart_rehearsal-0-py3-none-any.whl").write_bytes(
                    b"restart-rehearsal-wheel\n"
                )
                _event("pip", module_argv, status="ok", operation="download")
                return 0
            if module_argv and module_argv[0] == "install":
                if _record_internal_substitution(
                    kind="python-dependencies",
                    argv=module_argv,
                    substitution="python_dependencies.install",
                ) != 0:
                    return 97
                requirement_paths: list[Path] = []
                for option in ("--requirement", "-r"):
                    start = 0
                    while True:
                        try:
                            index = module_argv.index(option, start)
                        except ValueError:
                            break
                        if index + 1 >= len(module_argv):
                            return _fail(
                                "pip",
                                module_argv,
                                f"{option} omitted its requirement path",
                            )
                        requirement_paths.append(Path(module_argv[index + 1]))
                        start = index + 2
                if not requirement_paths:
                    return _fail(
                        "pip",
                        module_argv,
                        "offline install contract omitted a requirement file",
                    )
                missing = [
                    str(path)
                    for path in requirement_paths
                    if not path.is_file() or path.stat().st_size == 0
                ]
                if missing:
                    return _fail(
                        "pip",
                        module_argv,
                        f"requirement file is unavailable: {missing}",
                    )
                result = subprocess.run(
                    [REAL_PYTHON, "-m", "pip", "check"],
                    check=False,
                )
                _event(
                    "pip",
                    module_argv,
                    status="ok" if result.returncode == 0 else "failed",
                    operation="offline-install-contract",
                    requirement_paths=[str(path) for path in requirement_paths],
                )
                return result.returncode
            if module_argv and module_argv[0] == "uninstall":
                if _record_internal_substitution(
                    kind="python-dependencies",
                    argv=module_argv,
                    substitution="python_dependencies.uninstall",
                ) != 0:
                    return 97
                _event(
                    "pip",
                    module_argv,
                    status="ok",
                    operation="offline-uninstall-contract",
                )
                return 0
            _event("pip", module_argv, status="real")
            os.execv(REAL_PYTHON, [REAL_PYTHON, *argv])
        if module == "Leadpoet.utils.restart_epoch_gate":
            if _record_internal_substitution(
                kind="python-module",
                argv=argv,
                module=module,
            ) != 0:
                return 97
            result = _module_restart_gate(module_argv)
        elif module == "gateway.tee.release_channel_v2":
            if _record_internal_substitution(
                kind="python-module",
                argv=argv,
                module=module,
            ) != 0:
                return 97
            result = _module_release_channel(module_argv)
        elif module == "gateway.tee.prepare_gateway_envelopes_v2":
            if _record_internal_substitution(
                kind="python-module",
                argv=argv,
                module=module,
            ) != 0:
                return 97
            result = _module_envelopes(module_argv)
        elif module in {
            "gateway.tee.restart_preflight_v2",
            "validator_tee.host.docker_operation_guard_v2",
            "gateway.research_lab.provider_profiles_v2",
            "gateway.utils.tee_v2_bootstrap",
            "gateway.utils.tee_kms_provision_v2",
            "gateway.tee.verify_v2_runtime_ready",
            "validator_tee.host.refresh_hotkey_config_v2",
            "validator_tee.host.restart_preflight_v2",
            "validator_tee.host.verify_chain_signing_profile_v2",
            "validator_tee.host.verify_release_gate_v2",
            "validator_tee.host.release_archive_v2",
            "validator_tee.host.runtime_v2_bootstrap",
            "validator_tee.host.hotkey_bootstrap_v2",
            "gateway.tee.release_archive_v2",
        }:
            if _record_internal_substitution(
                kind="python-module",
                argv=argv,
                module=module,
            ) != 0:
                return 97
            print(json.dumps({"status": "ready", "module": module}, sort_keys=True))
            result = 0
        elif module == "validator_tee.scripts.stage_runtime_artifacts_v2":
            if _record_internal_substitution(
                kind="python-module",
                argv=argv,
                module=module,
            ) != 0:
                return 97
            result = _module_stage_artifacts(module_argv)
        elif module == "gateway.tee.verify_weight_submission_ready_v2":
            return _module_weight_ready(module_argv)
        elif module == "gateway.main":
            return _long_lived_process("gateway.main", argv)
        elif module == "gateway.utils.tee_egress_forwarder":
            return _long_lived_process("gateway.tee_egress", argv)
        elif module == "gateway.utils.tee_inter_enclave_relay":
            return _long_lived_process("gateway.tee_relay", argv)
        elif module == "validator_tee.host.chain_relay_v2":
            return _long_lived_process("validator.chain_relay", argv)
        else:
            _record_production_module(module, argv)
            os.execv(REAL_PYTHON, [REAL_PYTHON, *argv])
        _event(
            "python-module",
            argv,
            status="ok" if result == 0 else "failed",
            module=module,
            implementation="internal_substitution",
            scope=_rehearsal_scope(),
        )
        return result
    if argv[0].endswith(".py"):
        intercepted = _python_script(argv)
        if intercepted is not None:
            return intercepted
        _record_production_script(Path(argv[0]), argv)
    if Path(argv[0]).name == "validator.py":
        if _record_internal_substitution(
            kind="validator-process",
            argv=argv,
            substitution="python.validator_coordinator",
        ) != 0:
            return 97
        handle, state = _locked_state()
        state.setdefault("containers", {})["leadpoet-validator-main"] = {
            "running": True,
            "restart_count": 0,
        }
        _save_state(handle, state)
        _event("validator-process", argv, status="started")
        print("rehearsal validator coordinator started")
        return 0
    _event("python", argv, status="real")
    os.execv(REAL_PYTHON, [REAL_PYTHON, *argv])
    return 127


def command_bash(argv: list[str]) -> int:
    if not argv:
        os.execv(REAL_BASH, [REAL_BASH])
    script = Path(argv[0]).name
    if script == "build_drand_cabi_v2.sh":
        if _record_internal_substitution(
            kind="bash",
            argv=argv,
            substitution="bash.build_drand_cabi_v2",
        ) != 0:
            return 97
        if len(argv) < 4:
            return _fail("bash", argv, "drand build arguments are incomplete")
        output = Path(argv[2])
        expected_hash = Path(argv[3]).read_text(encoding="utf-8").strip().split()[0]
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"rehearsal-drand-library\n")
        if not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
            return _fail("bash", argv, "drand expected hash is invalid")
        _event("bash", argv, status="ok", operation="drand-build-contract")
        return 0
    _event("bash", argv, status="real", script=script)
    os.execv(REAL_BASH, [REAL_BASH, *argv])
    return 127


COMMANDS = {
    "aws": command_aws,
    "docker": command_docker,
    "nitro-cli": command_nitro,
    "systemctl": command_systemctl,
    "curl": command_curl,
    "sudo": command_sudo,
    "df": command_df,
    "getconf": command_getconf,
    "awk": command_awk,
    "sleep": command_sleep,
    "ss": command_ss,
    "ctr": command_ctr,
    "nsenter": command_nsenter,
    "pgrep": command_pgrep,
    "pkill": command_pkill,
    "python3": command_python,
    "python3.11": command_python,
    "bash": command_bash,
}


def main() -> int:
    if len(sys.argv) < 2:
        print("adapter command is missing", file=sys.stderr)
        return 2
    command = sys.argv[1]
    argv = sys.argv[2:]
    handler = COMMANDS.get(command)
    if handler is None:
        return _fail("adapter", sys.argv[1:], "unknown adapter command")
    return int(handler(argv))


if __name__ == "__main__":
    raise SystemExit(main())
