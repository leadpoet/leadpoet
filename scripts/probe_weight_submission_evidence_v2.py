#!/usr/bin/env python3
"""Emit bounded, strictly read-only proof of finalized SN71 weights.

Repository identity and import safety are established with the standard library
before any repository module is imported. External access is limited to HTTPS
GET/HEAD and four read-only, finalized-state WebSocket JSON-RPC methods.
The command-line entry point must be invoked with Python isolated mode (``-I``).
"""

from __future__ import annotations

import sys


if __name__ == "__main__" and not sys.flags.isolated:
    sys.stderr.write("ERROR:isolated_python_required\n")
    raise SystemExit(2)


import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
from types import ModuleType, SimpleNamespace
from typing import Any, Callable, Dict, Mapping, Optional, Sequence
from urllib.parse import parse_qs, unquote, urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]

SCHEMA_VERSION = "leadpoet.weight_submission_evidence_probe.v2"
DEFAULT_GATEWAY_URL = "https://gateway.subnet71.com"
PROFILE_RELATIVE_PATH = "validator_tee/enclave/chain_signing_profile_v2.json"
CUTOVER_RELATIVE_PATH = "config/stateful-epoch-cutover-sn71.json"
BINDING_RELATIVE_PATH = "leadpoet_canonical/binding.py"
MAX_GATEWAY_DOCUMENT_BYTES = 8 * 1024 * 1024
MAX_RELEASE_EVIDENCE_BYTES = 64 * 1024
MAX_SOURCE_BYTES = 8 * 1024 * 1024
MAX_GIT_OUTPUT_BYTES = 32 * 1024 * 1024
MAX_URL_BYTES = 8192
MAX_PATH_BYTES = 1024
MAX_JSON_DEPTH = 96
MAX_JSON_NODES = 250_000
MAX_AUDITOR_HOTKEYS = 16
MAX_HOTKEY_INPUT_BYTES = 64
MAX_WORKTREE_ENTRIES = 100_000
MAX_WORKTREE_FILE_BYTES = 32 * 1024 * 1024
MAX_RELEASE_CHANNEL_BYTES = 2 * 1024 * 1024
MAX_VERSION_ID_BYTES = 1024
TRUSTED_GIT = Path("/usr/bin/git")
RELEASE_BUCKET = "leadpoet-attested-v2-artifacts-493765492819"
RELEASE_PREFIX = "attested-v2/releases"
RELEASE_EVIDENCE_SCHEMA_VERSION = "leadpoet.auditor_release_evidence.v2"
RELEASE_CHANNEL_SCHEMA_VERSION = "leadpoet.attested_release_channel.v2"
IDENTITY_CACHE_SCHEMA_VERSION = "leadpoet.independent_pcr0_identities.v2"
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_HOTKEY_RE = re.compile(r"^[1-9A-HJ-NP-Za-km-z]{32,64}$")
_READ_ONLY_RPC_METHODS = frozenset(
    {"chain_getFinalizedHead", "chain_getHeader", "state_call", "state_getStorage"}
)
_REPOSITORY_IMPORT_ROOTS = frozenset(
    {"Leadpoet", "gateway", "leadpoet_canonical", "validator_tee"}
)
_RUNTIME_MODULE_NAMES = (
    "Leadpoet.utils.subnet_epoch",
    "leadpoet_canonical.ancestry_checkpoint_v2",
    "leadpoet_canonical.chain_source_v2",
    "leadpoet_canonical.compact_auditor_authority_v2",
    "leadpoet_canonical.hotkey_authority_v2",
)
_NORMAL_IMPORT_DENY_ROOTS = frozenset({"gateway", "validator_tee"})
_FORBIDDEN_RUNTIME_PREFIXES = (
    "bittensor",
    "bittensor_wallet",
    "gateway",
    "scalecodec",
    "substrate_interface",
    "substrateinterface",
    "validator_tee",
)
_FORBIDDEN_RUNTIME_COMPONENTS = frozenset(
    {"publication", "publisher", "signer", "signing", "transport", "vsock", "wallet"}
)
_PRIVATE_RELEASE_MODULE_PREFIX = "_leadpoet_probe_release_"
_PUBLIC_VERIFY_WALLET_MODULES = frozenset(
    {"bittensor_wallet", "bittensor_wallet.bittensor_wallet"}
)
_EXPECTED_RELEASE_IDENTITY_ROLES = {
    "gateway_coordinator": "gateway_coordinator",
    "gateway_scoring": "gateway_scoring",
    "validator_weights": "validator_weights",
}
_RELEASE_IDENTITY_FIELDS = {
    "physical_role",
    "role",
    "commit_sha",
    "pcr0",
    "build_manifest_hash",
    "dependency_lock_hash",
    "verified_build_count",
}
_CHAIN_STATE_FIELDS = {
    "block_hash",
    "block",
    "subnet_epoch_index",
    "metagraph_hotkeys",
    "validators",
}
_VALIDATOR_STATE_FIELDS = {
    "hotkey",
    "uid",
    "mechanism_id",
    "last_update",
    "weights",
}


class WeightSubmissionEvidenceProbeError(RuntimeError):
    """A fixed, redacted failure from the read-only evidence boundary."""

    def __init__(self, code: str):
        normalized = str(code or "probe_failed").strip().lower()
        if not re.fullmatch(r"[a-z0-9_]{3,80}", normalized):
            normalized = "probe_failed"
        self.code = normalized
        super().__init__(normalized)


def _fail(code: str) -> None:
    raise WeightSubmissionEvidenceProbeError(code)


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError, RecursionError):
        _fail("evidence_serialization_invalid")


def _sha256_json(value: Any) -> str:
    encoded = _canonical_json(value).encode("ascii")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _bounded_ascii(value: Any, *, maximum_bytes: int, code: str) -> str:
    if type(value) is not str:
        _fail(code)
    try:
        encoded = value.encode("ascii")
    except (UnicodeEncodeError, MemoryError):
        _fail(code)
    if not encoded or len(encoded) > maximum_bytes:
        _fail(code)
    return value


def _strict_json(
    payload: bytes,
    *,
    maximum_bytes: int,
    invalid_code: str = "gateway_document_invalid",
) -> Dict[str, Any]:
    if type(payload) is not bytes or not payload or len(payload) > int(maximum_bytes):
        _fail(invalid_code)

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                _fail("gateway_document_duplicate_key")
            result[key] = value
        return result

    def reject_constant(_value):
        _fail(invalid_code)

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=reject_constant,
        )
    except WeightSubmissionEvidenceProbeError:
        raise
    except (Exception, MemoryError):
        _fail(invalid_code)
    if not isinstance(value, Mapping):
        _fail(invalid_code)
    stack = [(value, 0)]
    nodes = 0
    while stack:
        current, depth = stack.pop()
        nodes += 1
        if nodes > MAX_JSON_NODES or depth > MAX_JSON_DEPTH:
            _fail(invalid_code)
        if isinstance(current, Mapping):
            stack.extend((item, depth + 1) for item in current.values())
        elif isinstance(current, list):
            stack.extend((item, depth + 1) for item in current)
        elif isinstance(current, float) and not math.isfinite(current):
            _fail(invalid_code)
    return dict(value)


def _bounded_url(value: Any, *, scheme: str, code: str, allow_query: bool) -> str:
    text = _bounded_ascii(value, maximum_bytes=MAX_URL_BYTES, code=code).strip()
    try:
        parsed = urlsplit(text)
        port = parsed.port
    except (Exception, MemoryError):
        _fail(code)
    if (
        parsed.scheme != scheme
        or not parsed.hostname
        or port not in (None, 443)
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or (not allow_query and bool(parsed.query))
    ):
        _fail(code)
    return text


def _gateway_origin(value: str) -> str:
    text = _bounded_url(
        value, scheme="https", code="gateway_origin_invalid", allow_query=False
    )
    parsed = urlsplit(text)
    if parsed.path not in ("", "/"):
        _fail("gateway_origin_invalid")
    return urlunsplit(("https", parsed.netloc, "", "", ""))


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class ReadOnlyHttp:
    """No-proxy, no-redirect HTTP boundary accepting only GET and HEAD."""

    def __init__(self, gateway_url: str, *, opener: Any = None):
        self.gateway_url = _gateway_origin(gateway_url)
        self._opener = opener or build_opener(ProxyHandler({}), _NoRedirect())

    def open_exact_url(self, url: str, *, method: str):
        normalized_method = str(method or "").upper()
        if normalized_method not in {"GET", "HEAD"}:
            _fail("write_capable_http_call_rejected")
        exact_url = _bounded_url(
            url, scheme="https", code="external_url_invalid", allow_query=True
        )
        request = Request(
            exact_url,
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "identity",
                "User-Agent": "leadpoet-weight-evidence-v2",
            },
            method=normalized_method,
        )
        try:
            return self._opener.open(request, timeout=30)
        except WeightSubmissionEvidenceProbeError:
            raise
        except (Exception, MemoryError):
            _fail("gateway_read_unavailable")

    def get_json(self, path: str, *, maximum_bytes: int) -> Dict[str, Any]:
        normalized_path = _bounded_ascii(
            path, maximum_bytes=MAX_PATH_BYTES, code="gateway_path_invalid"
        )
        if (
            not normalized_path.startswith("/")
            or "?" in normalized_path
            or "#" in normalized_path
            or "//" in normalized_path
        ):
            _fail("gateway_path_invalid")
        url = self.gateway_url + normalized_path
        try:
            with self.open_exact_url(url, method="GET") as response:
                if int(getattr(response, "status", 0)) != 200:
                    _fail("gateway_read_unavailable")
                observed_url = _bounded_ascii(
                    str(response.geturl()),
                    maximum_bytes=MAX_URL_BYTES,
                    code="gateway_redirect_rejected",
                )
                if observed_url != url:
                    _fail("gateway_redirect_rejected")
                payload = response.read(int(maximum_bytes) + 1)
        except WeightSubmissionEvidenceProbeError:
            raise
        except (Exception, MemoryError):
            _fail("gateway_read_unavailable")
        if not isinstance(payload, bytes):
            _fail("gateway_document_invalid")
        return _strict_json(payload, maximum_bytes=int(maximum_bytes))


def _run_git(
    root: Path,
    args: Sequence[str],
    *,
    allowed_returncodes: Sequence[int] = (0,),
    maximum_output_bytes: int = MAX_GIT_OUTPUT_BYTES,
) -> tuple[int, bytes]:
    try:
        git_path = TRUSTED_GIT
        git_stat = git_path.lstat()
        if (
            not git_path.is_absolute()
            or not stat.S_ISREG(git_stat.st_mode)
            or stat.S_ISLNK(git_stat.st_mode)
            or git_stat.st_uid != 0
            or git_stat.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
            or not git_stat.st_mode & stat.S_IXUSR
        ):
            _fail("trusted_git_invalid")
    except WeightSubmissionEvidenceProbeError:
        raise
    except (Exception, MemoryError):
        _fail("trusted_git_invalid")
    environment = {
        "GIT_ATTR_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
    }
    try:
        completed = subprocess.run(
            [
                str(git_path),
                "--no-pager",
                "-c",
                "core.fsmonitor=false",
                "-c",
                "core.untrackedCache=false",
                "-c",
                "core.hooksPath=/dev/null",
                *args,
            ],
            cwd=root,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=15,
            check=False,
            env=environment,
        )
    except (Exception, MemoryError):
        _fail("repository_identity_unavailable")
    if (
        completed.returncode not in set(allowed_returncodes)
        or len(completed.stdout) > maximum_output_bytes
        or len(completed.stderr) > maximum_output_bytes
    ):
        _fail("repository_identity_unavailable")
    return int(completed.returncode), bytes(completed.stdout)


def _decode_git_line(value: bytes) -> str:
    try:
        text = value.decode("ascii").strip()
    except (UnicodeDecodeError, MemoryError):
        _fail("repository_identity_unavailable")
    if not text or "\n" in text or "\r" in text:
        _fail("repository_identity_unavailable")
    return text


def _repository_path(root: Path, value: bytes) -> Path:
    text = _decode_git_line(value)
    try:
        path = Path(text)
        if not path.is_absolute():
            path = root / path
        return path.resolve(strict=True)
    except (Exception, MemoryError):
        _fail("repository_identity_unavailable")


def _reject_repository_overlays(root: Path) -> None:
    common = _repository_path(
        root, _run_git(root, ("rev-parse", "--git-common-dir"))[1]
    )
    for relative in (
        "info/attributes",
        "info/grafts",
        "objects/info/alternates",
        "objects/info/http-alternates",
        "shallow",
    ):
        if os.path.lexists(common / relative):
            _fail("repository_history_or_attributes_override")
    if _run_git(root, ("for-each-ref", "--format=%(refname)", "refs/replace"))[1]:
        _fail("repository_history_or_attributes_override")
    shallow = _decode_git_line(
        _run_git(root, ("rev-parse", "--is-shallow-repository"))[1]
    )
    if shallow not in {"true", "false"}:
        _fail("repository_identity_unavailable")
    if shallow == "true":
        _fail("repository_history_or_attributes_override")
    configured, _output = _run_git(
        root,
        (
            "config",
            "--includes",
            "--name-only",
            "--get-regexp",
            r"^(filter\.|core\.attributesfile$|include\.|includeif\.)",
        ),
        allowed_returncodes=(0, 1),
    )
    if configured == 0:
        _fail("repository_filter_configuration")


def _parse_tree_inventory(payload: bytes) -> Dict[str, tuple[str, str]]:
    inventory: Dict[str, tuple[str, str]] = {}
    try:
        for record in payload.split(b"\0"):
            if not record:
                continue
            metadata, raw_path = record.split(b"\t", 1)
            mode, object_type, object_id = metadata.decode("ascii").split(" ")
            path = raw_path.decode("utf-8")
            parts = path.split("/")
            if (
                object_type != "blob"
                or mode not in {"100644", "100755", "120000"}
                or not _SHA_RE.fullmatch(object_id)
                or not path
                or path.startswith("/")
                or any(part in {"", ".", ".."} for part in parts)
                or path in inventory
                or len(inventory) >= MAX_WORKTREE_ENTRIES
            ):
                _fail("repository_identity_unavailable")
            inventory[path] = (mode, object_id)
    except WeightSubmissionEvidenceProbeError:
        raise
    except (Exception, MemoryError):
        _fail("repository_identity_unavailable")
    if not inventory:
        _fail("repository_identity_unavailable")
    return inventory


def _tree_inventory(root: Path, candidate: str) -> Dict[str, tuple[str, str]]:
    return _parse_tree_inventory(
        _run_git(root, ("ls-tree", "-r", "-z", "--full-tree", candidate))[1]
    )


def _index_inventory(root: Path) -> Dict[str, tuple[str, str]]:
    payload = _run_git(root, ("ls-files", "--stage", "-z"))[1]
    inventory: Dict[str, tuple[str, str]] = {}
    try:
        for record in payload.split(b"\0"):
            if not record:
                continue
            metadata, raw_path = record.split(b"\t", 1)
            mode, object_id, stage_number = metadata.decode("ascii").split(" ")
            path = raw_path.decode("utf-8")
            if (
                stage_number != "0"
                or mode not in {"100644", "100755", "120000"}
                or not _SHA_RE.fullmatch(object_id)
                or path in inventory
                or len(inventory) >= MAX_WORKTREE_ENTRIES
            ):
                _fail("candidate_sha_mismatch")
            inventory[path] = (mode, object_id)
    except WeightSubmissionEvidenceProbeError:
        raise
    except (Exception, MemoryError):
        _fail("candidate_sha_mismatch")
    return inventory


def _blob_digest(payload: bytes) -> str:
    digest = hashlib.sha1()
    digest.update(("blob %d\0" % len(payload)).encode("ascii"))
    digest.update(payload)
    return digest.hexdigest()


def _read_regular_blob(path: Path, *, maximum_bytes: int) -> tuple[bytes, str, bool]:
    try:
        before = path.lstat()
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size < 0
            or before.st_size > maximum_bytes
        ):
            _fail("candidate_sha_mismatch")
        payload = path.read_bytes()
        after = path.lstat()
        if (
            len(payload) != before.st_size
            or (before.st_dev, before.st_ino, before.st_mode, before.st_size)
            != (after.st_dev, after.st_ino, after.st_mode, after.st_size)
            or getattr(before, "st_mtime_ns", None)
            != getattr(after, "st_mtime_ns", None)
        ):
            _fail("candidate_sha_mismatch")
    except WeightSubmissionEvidenceProbeError:
        raise
    except (Exception, MemoryError):
        _fail("candidate_sha_mismatch")
    return payload, _blob_digest(payload), bool(before.st_mode & 0o111)


def _worktree_blob(root: Path, relative: str, mode: str) -> tuple[Optional[bytes], str]:
    path = root / relative
    if mode == "120000":
        try:
            before = path.lstat()
            if not stat.S_ISLNK(before.st_mode):
                _fail("candidate_sha_mismatch")
            payload = os.fsencode(os.readlink(path))
            after = path.lstat()
            if (before.st_dev, before.st_ino, before.st_mode) != (
                after.st_dev,
                after.st_ino,
                after.st_mode,
            ):
                _fail("candidate_sha_mismatch")
        except WeightSubmissionEvidenceProbeError:
            raise
        except (Exception, MemoryError):
            _fail("candidate_sha_mismatch")
        return None, _blob_digest(payload)
    payload, object_id, executable = _read_regular_blob(
        path, maximum_bytes=MAX_WORKTREE_FILE_BYTES
    )
    if executable != (mode == "100755"):
        _fail("candidate_sha_mismatch")
    return payload, object_id


def _verify_raw_worktree(root: Path, inventory: Mapping[str, tuple[str, str]]) -> None:
    expected_directories = set()
    for relative in inventory:
        parts = relative.split("/")
        expected_directories.update(
            "/".join(parts[:index]) for index in range(1, len(parts))
        )
    seen = set()
    stack = [(root, "")]
    scanned = 0
    try:
        while stack:
            directory, prefix = stack.pop()
            with os.scandir(directory) as entries:
                for entry in entries:
                    if not prefix and entry.name == ".git":
                        continue
                    scanned += 1
                    if scanned > MAX_WORKTREE_ENTRIES:
                        _fail("candidate_sha_mismatch")
                    relative = entry.name if not prefix else prefix + "/" + entry.name
                    if entry.is_dir(follow_symlinks=False):
                        if relative not in expected_directories:
                            _fail("candidate_sha_mismatch")
                        stack.append((Path(entry.path), relative))
                        continue
                    expected = inventory.get(relative)
                    if expected is None or relative in seen:
                        _fail("candidate_sha_mismatch")
                    _payload, object_id = _worktree_blob(root, relative, expected[0])
                    if object_id != expected[1]:
                        _fail("candidate_sha_mismatch")
                    seen.add(relative)
    except WeightSubmissionEvidenceProbeError:
        raise
    except (Exception, MemoryError):
        _fail("candidate_sha_mismatch")
    if seen != set(inventory):
        _fail("candidate_sha_mismatch")


def _wallet_import_nodes(tree: ast.AST) -> list[ast.AST]:
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(
                alias.name == "bittensor_wallet"
                or alias.name.startswith("bittensor_wallet.")
                for alias in node.names
            ):
                imports.append(node)
        elif isinstance(node, ast.ImportFrom):
            module = str(node.module or "")
            if module == "bittensor_wallet" or module.startswith("bittensor_wallet."):
                imports.append(node)
    return imports


def _audit_binding_public_key_verifier(source: bytes) -> None:
    """Prove the candidate binding uses the wallet only for public verification."""

    try:
        tree = ast.parse(source, filename="<candidate:%s>" % BINDING_RELATIVE_PATH)
        parents = {
            child: parent
            for parent in ast.walk(tree)
            for child in ast.iter_child_nodes(parent)
        }
        functions = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "verify_binding_message"
        ]
        if len(functions) != 1:
            raise ValueError
        function = functions[0]
        arguments = function.args
        if (
            [item.arg for item in arguments.posonlyargs + arguments.args]
            != [
                "binding_msg",
                "signature_hex",
                "hotkey",
                "expected_netuid",
                "expected_chain",
                "expected_enclave_pubkey",
                "expected_code_hash",
            ]
            or arguments.vararg is not None
            or arguments.kwarg is not None
            or arguments.kwonlyargs
            or arguments.defaults
            or arguments.kw_defaults
        ):
            raise ValueError

        wallet_imports = _wallet_import_nodes(tree)
        if len(wallet_imports) != 1:
            raise ValueError
        wallet_import = wallet_imports[0]
        if (
            not isinstance(wallet_import, ast.ImportFrom)
            or wallet_import.level != 0
            or wallet_import.module != "bittensor_wallet"
            or [(alias.name, alias.asname) for alias in wallet_import.names]
            != [("Keypair", None)]
        ):
            raise ValueError
        ancestor = parents.get(wallet_import)
        while ancestor is not None and not isinstance(
            ancestor, (ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            ancestor = parents.get(ancestor)
        if ancestor is not function:
            raise ValueError

        keypair_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "Keypair"
        ]
        if len(keypair_calls) != 1:
            raise ValueError
        constructor = keypair_calls[0]
        if (
            constructor.args
            or len(constructor.keywords) != 1
            or constructor.keywords[0].arg != "ss58_address"
            or not isinstance(constructor.keywords[0].value, ast.Name)
            or constructor.keywords[0].value.id != "hotkey"
        ):
            raise ValueError
        assignment = parents.get(constructor)
        if (
            not isinstance(assignment, ast.Assign)
            or len(assignment.targets) != 1
            or not isinstance(assignment.targets[0], ast.Name)
            or assignment.targets[0].id != "keypair"
        ):
            raise ValueError

        verification_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "verify"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "keypair"
        ]
        if len(verification_calls) != 1:
            raise ValueError
        verification = verification_calls[0]
        if verification.keywords or len(verification.args) != 2:
            raise ValueError
        encoded, signature = verification.args
        if (
            not isinstance(encoded, ast.Call)
            or encoded.args
            or encoded.keywords
            or not isinstance(encoded.func, ast.Attribute)
            or encoded.func.attr != "encode"
            or not isinstance(encoded.func.value, ast.Name)
            or encoded.func.value.id != "binding_msg"
            or not isinstance(signature, ast.Call)
            or len(signature.args) != 1
            or signature.keywords
            or not isinstance(signature.func, ast.Attribute)
            or signature.func.attr != "fromhex"
            or not isinstance(signature.func.value, ast.Name)
            or signature.func.value.id != "bytes"
            or not isinstance(signature.args[0], ast.Name)
            or signature.args[0].id != "signature_hex"
            or not isinstance(parents.get(verification), ast.Return)
        ):
            raise ValueError

        keypair_names = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and node.id == "Keypair"
        ]
        keypair_instances = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Name) and node.id == "keypair"
        ]
        keypair_attributes = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "keypair"
        ]
        if (
            len(keypair_names) != 1
            or keypair_names[0] is not constructor.func
            or len(keypair_instances) != 2
            or {type(node.ctx) for node in keypair_instances} != {ast.Load, ast.Store}
            or keypair_attributes != [verification.func]
        ):
            raise ValueError

        forbidden_calls = {
            "__import__",
            "compile",
            "eval",
            "exec",
            "getattr",
            "setattr",
        }
        forbidden_components = {
            "coldkey",
            "config",
            "keyfile",
            "mnemonic",
            "mock",
            "path",
            "private",
            "seed",
            "secret",
            "wallet",
        }
        for node in ast.walk(function):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id.lower() in forbidden_calls:
                    raise ValueError
            if isinstance(node, (ast.Name, ast.Attribute)):
                identifier = node.id if isinstance(node, ast.Name) else node.attr
                components = set(re.split(r"[_-]", identifier.lower()))
                if components & (forbidden_components | {"create", "sign"}):
                    raise ValueError
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = (
                    [alias.name for alias in node.names]
                    if isinstance(node, ast.Import)
                    else [str(node.module or "")]
                )
                if any(
                    name == "mock"
                    or name.startswith("mock.")
                    or name == "unittest.mock"
                    or name.startswith("unittest.mock.")
                    for name in names
                ) or any(
                    alias.name == "mock" or alias.asname == "mock"
                    for alias in node.names
                ):
                    raise ValueError
    except WeightSubmissionEvidenceProbeError:
        raise
    except (Exception, MemoryError):
        _fail("candidate_public_key_policy_invalid")


def _audit_candidate_wallet_surface(
    relative_path: str,
    source: bytes,
    *,
    binding_policy: Optional[Any],
    object_id: str,
) -> None:
    try:
        tree = ast.parse(source, filename="<candidate:%s>" % relative_path)
    except (Exception, MemoryError):
        _fail("repository_import_invalid")
    if relative_path == BINDING_RELATIVE_PATH:
        if (
            binding_policy is None
            or binding_policy.binding_object_id != object_id
            or binding_policy.binding_source_hash != hashlib.sha256(source).hexdigest()
        ):
            _fail("candidate_public_key_policy_invalid")
        _audit_binding_public_key_verifier(source)
        return
    if _wallet_import_nodes(tree):
        _fail("candidate_public_key_policy_invalid")
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if "bittensor_wallet" in node.value.lower():
                _fail("candidate_public_key_policy_invalid")
        if isinstance(node, ast.Name) and node.id in {"Keyfile", "Keypair", "Wallet"}:
            _fail("candidate_public_key_policy_invalid")


class _CandidateRepository:
    """Exact candidate tree plus dependency-free source/import binding."""

    def __init__(
        self,
        *,
        root: Path,
        candidate_sha: str,
        tree_sha: str,
        inventory: Mapping[str, tuple[str, str]],
    ):
        self.root = root
        self.candidate_sha = candidate_sha
        self.tree_sha = tree_sha
        self.inventory = dict(inventory)
        self.public_key_verifier_policy = None

    @classmethod
    def preflight(cls, root: Path, candidate_sha: str) -> "_CandidateRepository":
        if type(candidate_sha) is not str:
            _fail("candidate_sha_invalid")
        candidate = candidate_sha.lower()
        if not _SHA_RE.fullmatch(candidate):
            _fail("candidate_sha_invalid")
        try:
            resolved_root = Path(root).resolve(strict=True)
        except (Exception, MemoryError):
            _fail("repository_identity_unavailable")
        top = _decode_git_line(
            _run_git(resolved_root, ("rev-parse", "--show-toplevel"))[1]
        )
        try:
            if Path(top).resolve(strict=True) != resolved_root:
                _fail("candidate_sha_mismatch")
        except WeightSubmissionEvidenceProbeError:
            raise
        except (Exception, MemoryError):
            _fail("repository_identity_unavailable")
        _reject_repository_overlays(resolved_root)
        if (
            _decode_git_line(
                _run_git(resolved_root, ("rev-parse", "--show-object-format"))[1]
            )
            != "sha1"
        ):
            _fail("repository_identity_unavailable")
        head = _decode_git_line(_run_git(resolved_root, ("rev-parse", "HEAD"))[1])
        head_tree = _decode_git_line(
            _run_git(resolved_root, ("rev-parse", "HEAD^{tree}"))[1]
        )
        candidate_tree = _decode_git_line(
            _run_git(resolved_root, ("rev-parse", candidate + "^{tree}"))[1]
        )
        if head != candidate or head_tree != candidate_tree:
            _fail("candidate_sha_mismatch")
        inventory = _tree_inventory(resolved_root, candidate)
        if _index_inventory(resolved_root) != inventory:
            _fail("candidate_sha_mismatch")
        _verify_raw_worktree(resolved_root, inventory)
        probe_path = "scripts/probe_weight_submission_evidence_v2.py"
        if probe_path not in inventory:
            _fail("candidate_sha_mismatch")
        binding = cls(
            root=resolved_root,
            candidate_sha=candidate,
            tree_sha=candidate_tree,
            inventory=inventory,
        )
        binding.read_bound_file(probe_path, maximum_bytes=MAX_SOURCE_BYTES)
        return binding

    def recheck(self) -> None:
        _reject_repository_overlays(self.root)
        head = _decode_git_line(_run_git(self.root, ("rev-parse", "HEAD"))[1])
        tree = _decode_git_line(_run_git(self.root, ("rev-parse", "HEAD^{tree}"))[1])
        if head != self.candidate_sha or tree != self.tree_sha:
            _fail("candidate_sha_mismatch")
        inventory = _tree_inventory(self.root, self.candidate_sha)
        if inventory != self.inventory or _index_inventory(self.root) != self.inventory:
            _fail("candidate_sha_mismatch")
        _verify_raw_worktree(self.root, self.inventory)

    def read_bound_file(self, relative_path: str, *, maximum_bytes: int) -> bytes:
        entry = self.inventory.get(relative_path)
        if entry is None or entry[0] not in {"100644", "100755"}:
            _fail("candidate_source_binding_invalid")
        if maximum_bytes <= 0 or maximum_bytes > MAX_WORKTREE_FILE_BYTES:
            _fail("candidate_source_binding_invalid")
        try:
            path = self.root / relative_path
            current, object_id, executable = _read_regular_blob(
                path, maximum_bytes=maximum_bytes
            )
        except WeightSubmissionEvidenceProbeError:
            _fail("candidate_source_binding_invalid")
        if not current or object_id != entry[1] or executable != (entry[0] == "100755"):
            _fail("candidate_source_binding_invalid")
        return current

    def _bind_loaded_module(self, name: str, module: Any) -> None:
        top = str(name).partition(".")[0]
        file_value = getattr(module, "__file__", None) if module is not None else None
        if file_value is None:
            if top in _REPOSITORY_IMPORT_ROOTS and module is not None:
                paths = list(getattr(module, "__path__", ()) or ())
                if not paths:
                    _fail("repository_import_binding_invalid")
                for path_value in paths:
                    try:
                        Path(path_value).resolve(strict=True).relative_to(self.root)
                    except (Exception, MemoryError):
                        _fail("repository_import_binding_invalid")
            return
        try:
            module_path = Path(file_value).resolve(strict=True)
        except (Exception, MemoryError):
            if top in _REPOSITORY_IMPORT_ROOTS:
                _fail("repository_import_binding_invalid")
            return
        try:
            relative = module_path.relative_to(self.root)
        except ValueError:
            if top in _REPOSITORY_IMPORT_ROOTS:
                _fail("repository_import_binding_invalid")
            return
        if module_path.suffix in {".pyc", ".pyo"}:
            try:
                module_path = Path(
                    importlib.util.source_from_cache(str(module_path))
                ).resolve(strict=True)
                relative = module_path.relative_to(self.root)
            except (Exception, MemoryError):
                _fail("repository_import_binding_invalid")
        self.read_bound_file(relative.as_posix(), maximum_bytes=MAX_SOURCE_BYTES)

    def bind_loaded_modules(self) -> None:
        for name, module in tuple(sys.modules.items()):
            self._bind_loaded_module(name, module)


class _CapturedCandidateSourceLoader(importlib.abc.Loader):
    """Execute only source bytes already bound to the candidate Git blob."""

    def __init__(
        self,
        *,
        binding: _CandidateRepository,
        fullname: str,
        relative_path: str,
        is_package: bool,
    ):
        entry = binding.inventory.get(relative_path)
        if entry is None or entry[0] not in {"100644", "100755"}:
            _fail("repository_import_invalid")
        source = binding.read_bound_file(relative_path, maximum_bytes=MAX_SOURCE_BYTES)
        _audit_candidate_wallet_surface(
            relative_path,
            source,
            binding_policy=binding.public_key_verifier_policy,
            object_id=entry[1],
        )
        self.fullname = fullname
        self.relative_path = relative_path
        self.filename = str(binding.root / relative_path)
        self.is_package = bool(is_package)
        self.source = bytes(source)

    def create_module(self, spec):
        del spec
        return None

    def exec_module(self, module) -> None:
        try:
            spec = getattr(module, "__spec__", None)
            if (
                spec is None
                or spec.name != self.fullname
                or spec.loader is not self
                or str(getattr(module, "__file__", "")) != self.filename
            ):
                _fail("repository_import_invalid")
            code = compile(
                self.source,
                "<candidate:%s>" % self.relative_path,
                "exec",
                dont_inherit=True,
                optimize=0,
            )
            exec(code, module.__dict__)
        except WeightSubmissionEvidenceProbeError:
            raise
        except (Exception, MemoryError):
            _fail("repository_import_invalid")


class _PublicKeyVerifierPolicy:
    __slots__ = (
        "binding_object_id",
        "binding_source_hash",
        "extension_module",
        "extension_origin",
        "keypair",
        "public_module",
        "public_origin",
        "verify_method",
    )

    def __init__(
        self,
        *,
        binding_object_id: str,
        binding_source_hash: str,
        extension_module: ModuleType,
        extension_origin: str,
        keypair: Any,
        public_module: ModuleType,
        public_origin: str,
        verify_method: Any,
    ):
        self.binding_object_id = binding_object_id
        self.binding_source_hash = binding_source_hash
        self.extension_module = extension_module
        self.extension_origin = extension_origin
        self.keypair = keypair
        self.public_module = public_module
        self.public_origin = public_origin
        self.verify_method = verify_method


class _CandidateSourceFinder(importlib.abc.MetaPathFinder):
    """Resolve repository packages only from candidate-tracked Python source."""

    def __init__(self, binding: _CandidateRepository):
        self.binding = binding

    def find_spec(self, fullname, path=None, target=None):
        del path, target
        top = str(fullname).partition(".")[0]
        if top in _NORMAL_IMPORT_DENY_ROOTS:
            raise ModuleNotFoundError("candidate repository module denied")
        if top not in _REPOSITORY_IMPORT_ROOTS:
            return None
        stem = str(fullname).replace(".", "/")
        package_path = stem + "/__init__.py"
        module_path = stem + ".py"
        if self.binding.inventory.get(package_path, (None,))[0] in {
            "100644",
            "100755",
        }:
            source = self.binding.root / package_path
            loader = _CapturedCandidateSourceLoader(
                binding=self.binding,
                fullname=fullname,
                relative_path=package_path,
                is_package=True,
            )
            spec = importlib.machinery.ModuleSpec(
                fullname, loader, origin=str(source), is_package=True
            )
            spec.has_location = True
            spec.submodule_search_locations = [str(source.parent)]
            return spec
        if self.binding.inventory.get(module_path, (None,))[0] in {
            "100644",
            "100755",
        }:
            source = self.binding.root / module_path
            loader = _CapturedCandidateSourceLoader(
                binding=self.binding,
                fullname=fullname,
                relative_path=module_path,
                is_package=False,
            )
            spec = importlib.machinery.ModuleSpec(
                fullname, loader, origin=str(source), is_package=False
            )
            spec.has_location = True
            return spec
        prefix = stem + "/"
        if any(candidate.startswith(prefix) for candidate in self.binding.inventory):
            spec = importlib.machinery.ModuleSpec(
                fullname, loader=None, is_package=True
            )
            spec.submodule_search_locations = [str(self.binding.root / stem)]
            return spec
        raise ModuleNotFoundError("candidate repository module unavailable")


def _validate_public_key_verifier_runtime(
    observed: Mapping[str, Any], policy: _PublicKeyVerifierPolicy
) -> None:
    try:
        names = {
            str(name)
            for name in observed
            if str(name) == "bittensor_wallet"
            or str(name).startswith("bittensor_wallet.")
        }
        public_module = observed.get("bittensor_wallet")
        extension_module = observed.get("bittensor_wallet.bittensor_wallet")
        public_names = {
            name for name in vars(public_module) if not str(name).startswith("_")
        }
        if (
            names != _PUBLIC_VERIFY_WALLET_MODULES
            or public_module is not policy.public_module
            or extension_module is not policy.extension_module
            or public_names != {"Keypair"}
            or getattr(public_module, "Keypair", None) is not policy.keypair
            or getattr(extension_module, "Keypair", None) is not policy.keypair
            or getattr(policy.keypair, "verify", None) is not policy.verify_method
            or getattr(public_module, "__file__", None) != policy.public_origin
            or getattr(extension_module, "__file__", None) != policy.extension_origin
            or not isinstance(
                getattr(extension_module, "__loader__", None),
                importlib.machinery.ExtensionFileLoader,
            )
        ):
            raise ValueError
    except (Exception, MemoryError):
        _fail("public_key_verifier_runtime_invalid")


def _reject_forbidden_runtime_modules(
    modules: Optional[Mapping[str, Any]] = None,
    *,
    public_key_verifier_policy: Optional[_PublicKeyVerifierPolicy] = None,
) -> None:
    observed = sys.modules if modules is None else modules
    for raw_name in tuple(observed):
        name = str(raw_name or "").lower()
        if not name or name.startswith(_PRIVATE_RELEASE_MODULE_PREFIX):
            continue
        if (
            public_key_verifier_policy is not None
            and name in _PUBLIC_VERIFY_WALLET_MODULES
        ):
            continue
        if any(
            name == prefix or name.startswith(prefix + ".")
            for prefix in _FORBIDDEN_RUNTIME_PREFIXES
        ):
            _fail("forbidden_runtime_module_loaded")
        components = {
            component
            for dotted in name.split(".")
            for component in re.split(r"[_-]", dotted)
            if component
        }
        if components & _FORBIDDEN_RUNTIME_COMPONENTS:
            _fail("forbidden_runtime_module_loaded")
    if public_key_verifier_policy is not None:
        _validate_public_key_verifier_runtime(observed, public_key_verifier_policy)


def _sanitized_dependency_paths(root: Path) -> list[str]:
    trusted_roots = set()
    for attribute in ("base_exec_prefix", "base_prefix", "exec_prefix", "prefix"):
        value = getattr(sys, attribute, None)
        if type(value) is not str or not value:
            continue
        try:
            trusted_roots.add(Path(value).resolve(strict=True))
        except (Exception, MemoryError):
            continue
    excluded = set()
    for entry in str(os.environ.get("PYTHONPATH") or "").split(os.pathsep):
        if not entry:
            continue
        try:
            excluded.add(Path(entry).resolve())
        except (Exception, MemoryError):
            continue
    retained = []
    seen = set()
    for entry in sys.path:
        if type(entry) is not str or not entry or not Path(entry).is_absolute():
            continue
        try:
            resolved = Path(entry).resolve(strict=True)
            details = resolved.stat()
            resolved.relative_to(root)
        except ValueError:
            pass
        except (Exception, MemoryError):
            continue
        else:
            continue
        if (
            resolved in excluded
            or details.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
            or not (stat.S_ISDIR(details.st_mode) or stat.S_ISREG(details.st_mode))
            or resolved in seen
        ):
            continue
        if not any(
            resolved == trusted
            or (
                resolved.is_relative_to(trusted)
                if hasattr(resolved, "is_relative_to")
                else str(resolved).startswith(str(trusted) + os.sep)
            )
            for trusted in trusted_roots
        ):
            continue
        seen.add(resolved)
        retained.append(str(resolved))
    return retained


def _activate_candidate_imports(binding: _CandidateRepository) -> None:
    for name in tuple(sys.modules):
        if str(name).partition(".")[0] in _REPOSITORY_IMPORT_ROOTS:
            _fail("repository_import_preloaded")
    _reject_forbidden_runtime_modules()
    sys.path[:] = [str(binding.root), *_sanitized_dependency_paths(binding.root)]
    if Path(sys.path[0]).resolve() != binding.root:
        _fail("repository_import_binding_invalid")
    sys.path_importer_cache.clear()
    sys.meta_path.insert(0, _CandidateSourceFinder(binding))
    importlib.invalidate_caches()


def _trusted_dependency_origin(module: Any, *, extension: bool) -> str:
    try:
        value = getattr(module, "__file__", None)
        if type(value) is not str or not value or not Path(value).is_absolute():
            raise ValueError
        unresolved = Path(value)
        if unresolved.is_symlink():
            raise ValueError
        resolved = unresolved.resolve(strict=True)
        details = resolved.stat()
        dependency_roots = []
        for entry in sys.path[1:]:
            if type(entry) is not str or not entry or not Path(entry).is_absolute():
                continue
            candidate = Path(entry).resolve(strict=True)
            if candidate.is_dir():
                dependency_roots.append(candidate)
        if (
            not stat.S_ISREG(details.st_mode)
            or details.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
            or not any(
                resolved == root
                or (
                    resolved.is_relative_to(root)
                    if hasattr(resolved, "is_relative_to")
                    else str(resolved).startswith(str(root) + os.sep)
                )
                for root in dependency_roots
            )
            or (not extension and resolved.suffix != ".py")
            or extension
            != any(
                str(resolved).endswith(suffix)
                for suffix in importlib.machinery.EXTENSION_SUFFIXES
            )
        ):
            raise ValueError
        return str(resolved)
    except (Exception, MemoryError):
        _fail("public_key_verifier_dependency_invalid")


def _preflight_public_key_verifier_dependency(
    binding: _CandidateRepository,
    *,
    module_importer: Optional[Callable[[str], Any]] = None,
) -> _PublicKeyVerifierPolicy:
    entry = binding.inventory.get(BINDING_RELATIVE_PATH)
    source = binding.read_bound_file(
        BINDING_RELATIVE_PATH, maximum_bytes=MAX_SOURCE_BYTES
    )
    if entry is None or entry[0] not in {"100644", "100755"}:
        _fail("candidate_public_key_policy_invalid")
    _audit_binding_public_key_verifier(source)
    binding.recheck()
    prior = {
        name: module
        for name, module in tuple(sys.modules.items())
        if name == "bittensor_wallet" or name.startswith("bittensor_wallet.")
    }
    if prior:
        _fail("forbidden_runtime_module_loaded")
    importer = module_importer or importlib.import_module

    def restore() -> None:
        for name in tuple(sys.modules):
            if name == "bittensor_wallet" or name.startswith("bittensor_wallet."):
                sys.modules.pop(name, None)
        sys.modules.update(prior)
        binding.public_key_verifier_policy = None

    try:
        imported = importer("bittensor_wallet")
        names = {
            name
            for name in sys.modules
            if name == "bittensor_wallet" or name.startswith("bittensor_wallet.")
        }
        extension_module = sys.modules.get("bittensor_wallet.bittensor_wallet")
        if (
            names != _PUBLIC_VERIFY_WALLET_MODULES
            or imported is not sys.modules.get("bittensor_wallet")
            or not isinstance(extension_module, ModuleType)
        ):
            raise ValueError
        public_origin = _trusted_dependency_origin(imported, extension=False)
        extension_origin = _trusted_dependency_origin(extension_module, extension=True)
        keypair = getattr(imported, "Keypair", None)
        verify_method = getattr(keypair, "verify", None)
        if (
            keypair is None
            or getattr(extension_module, "Keypair", None) is not keypair
            or not callable(keypair)
            or not callable(verify_method)
            or not isinstance(
                getattr(extension_module, "__loader__", None),
                importlib.machinery.ExtensionFileLoader,
            )
        ):
            raise ValueError
        public_module = ModuleType("bittensor_wallet")
        public_module.__file__ = public_origin
        public_module.__package__ = "bittensor_wallet"
        public_module.__path__ = ()
        public_module.Keypair = keypair
        sys.modules["bittensor_wallet"] = public_module
        policy = _PublicKeyVerifierPolicy(
            binding_object_id=entry[1],
            binding_source_hash=hashlib.sha256(source).hexdigest(),
            extension_module=extension_module,
            extension_origin=extension_origin,
            keypair=keypair,
            public_module=public_module,
            public_origin=public_origin,
            verify_method=verify_method,
        )
        binding.public_key_verifier_policy = policy
        _reject_forbidden_runtime_modules(public_key_verifier_policy=policy)
        binding.recheck()
        return policy
    except WeightSubmissionEvidenceProbeError as exc:
        restore()
        if exc.code in {
            "candidate_sha_mismatch",
            "candidate_source_binding_invalid",
            "repository_history_or_attributes_override",
        }:
            raise
        _fail("public_key_verifier_dependency_invalid")
    except (Exception, MemoryError):
        restore()
        _fail("public_key_verifier_dependency_invalid")


def _compile_private_module(
    *,
    name: str,
    relative_path: str,
    source: bytes,
    topology_alias: Optional[str] = None,
) -> ModuleType:
    try:
        if topology_alias is None:
            code = compile(
                source,
                "<candidate:%s>" % relative_path,
                "exec",
                dont_inherit=True,
                optimize=0,
            )
        else:
            tree = ast.parse(source, filename="<candidate:%s>" % relative_path)
            rewritten = 0
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.ImportFrom)
                    and node.level == 0
                    and node.module == "gateway.tee.topology"
                ):
                    names = {(item.name, item.asname) for item in node.names}
                    if names != {("ROLE_SPECS", None), ("topology_hash", None)}:
                        raise ValueError
                    node.module = topology_alias
                    rewritten += 1
            if rewritten != 1:
                raise ValueError
            ast.fix_missing_locations(tree)
            code = compile(
                tree,
                "<candidate:%s>" % relative_path,
                "exec",
                dont_inherit=True,
                optimize=0,
            )
        module = ModuleType(name)
        module.__file__ = "<candidate:%s>" % relative_path
        module.__package__ = ""
        sys.modules[name] = module
        exec(code, module.__dict__)
        return module
    except (Exception, MemoryError):
        sys.modules.pop(name, None)
        _fail("candidate_release_validator_invalid")


def _load_candidate_release_validators(
    binding: _CandidateRepository,
) -> SimpleNamespace:
    topology_path = "gateway/tee/topology.py"
    gateway_path = "gateway/tee/release_manifest_v2.py"
    validator_path = "validator_tee/host/release_v2.py"
    topology_alias = _PRIVATE_RELEASE_MODULE_PREFIX + "topology"
    gateway_alias = _PRIVATE_RELEASE_MODULE_PREFIX + "gateway_manifest"
    validator_alias = _PRIVATE_RELEASE_MODULE_PREFIX + "validator_manifest"
    topology = _compile_private_module(
        name=topology_alias,
        relative_path=topology_path,
        source=binding.read_bound_file(topology_path, maximum_bytes=MAX_SOURCE_BYTES),
    )
    gateway = _compile_private_module(
        name=gateway_alias,
        relative_path=gateway_path,
        source=binding.read_bound_file(gateway_path, maximum_bytes=MAX_SOURCE_BYTES),
        topology_alias=topology_alias,
    )
    validator = _compile_private_module(
        name=validator_alias,
        relative_path=validator_path,
        source=binding.read_bound_file(validator_path, maximum_bytes=MAX_SOURCE_BYTES),
    )
    binding.recheck()
    _reject_forbidden_runtime_modules()
    return SimpleNamespace(topology=topology, gateway=gateway, validator=validator)


def _candidate_release_contract(validators: SimpleNamespace) -> Dict[str, Any]:
    try:
        role_specs = validators.topology.ROLE_SPECS
        roles = {
            str(physical_role): str(spec["service_role"])
            for physical_role, spec in role_specs.items()
        }
        roles["validator_weights"] = "validator_weights"
        gateway_domains = validators.gateway.BUILDER_DOMAINS
        gateway_per_domain = validators.gateway.BUILDS_PER_DOMAIN
        validator_domains = validators.validator.VALIDATOR_BUILDER_DOMAINS
        validator_per_domain = validators.validator.VALIDATOR_BUILDS_PER_DOMAIN
        if (
            gateway_domains != frozenset({"gateway", "validator"})
            or validator_domains != frozenset({"gateway", "validator"})
            or type(gateway_per_domain) is not int
            or type(validator_per_domain) is not int
        ):
            raise TypeError
        gateway_count = len(gateway_domains) * gateway_per_domain
        validator_count = len(validator_domains) * validator_per_domain
    except (Exception, MemoryError):
        _fail("candidate_release_contract_invalid")
    if (
        roles != _EXPECTED_RELEASE_IDENTITY_ROLES
        or type(gateway_count) is not int
        or type(validator_count) is not int
        or gateway_count != 6
        or validator_count != 6
    ):
        _fail("candidate_release_contract_invalid")
    return {
        "roles": roles,
        "build_counts": {
            **{role: gateway_count for role in role_specs},
            "validator_weights": validator_count,
        },
    }


def _release_channel_url(value: Any, *, commit: str, version_id: str) -> str:
    text = _bounded_ascii(
        value,
        maximum_bytes=MAX_URL_BYTES,
        code="object_locked_release_invalid",
    )
    try:
        parsed = urlsplit(text)
        port = parsed.port
        query = {
            key.lower(): values
            for key, values in parse_qs(
                parsed.query,
                keep_blank_values=True,
                strict_parsing=True,
                max_num_fields=16,
            ).items()
        }
    except (Exception, MemoryError):
        _fail("object_locked_release_invalid")
    required = {
        "x-amz-algorithm",
        "x-amz-credential",
        "x-amz-date",
        "x-amz-expires",
        "x-amz-signedheaders",
        "x-amz-signature",
        "versionid",
    }
    if (
        parsed.scheme != "https"
        or parsed.hostname
        not in {
            "%s.s3.amazonaws.com" % RELEASE_BUCKET,
            "%s.s3.us-east-1.amazonaws.com" % RELEASE_BUCKET,
        }
        or port not in (None, 443)
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or unquote(parsed.path)
        != "/%s/%s/release-channel-v2.json" % (RELEASE_PREFIX, commit)
        or set(query) - (required | {"x-amz-security-token"})
        or not required.issubset(query)
        or any(len(values) != 1 or not values[0] for values in query.values())
    ):
        _fail("object_locked_release_invalid")
    try:
        expires = int(query["x-amz-expires"][0])
    except (TypeError, ValueError, IndexError):
        _fail("object_locked_release_invalid")
    if (
        query["x-amz-algorithm"] != ["AWS4-HMAC-SHA256"]
        or query["versionid"] != [version_id]
        or "host" not in query["x-amz-signedheaders"][0].split(";")
        or not 1 <= expires <= 300
    ):
        _fail("object_locked_release_invalid")
    return text


def _response_header(response: Any, name: str) -> str:
    try:
        headers = response.headers
        value = headers.get(name)
        if value is None:
            value = headers.get(name.title())
    except (Exception, MemoryError):
        _fail("object_locked_release_invalid")
    return _bounded_ascii(
        value,
        maximum_bytes=MAX_VERSION_ID_BYTES,
        code="object_locked_release_invalid",
    )


def _release_identity_from_channel(
    channel: Mapping[str, Any],
    *,
    expected_commit: str,
    validators: SimpleNamespace,
) -> Dict[str, Any]:
    fields = {
        "schema_version",
        "commit_sha",
        "gateway_release_manifest",
        "validator_release_manifest",
        "channel_hash",
    }
    if (
        not isinstance(channel, Mapping)
        or set(channel) != fields
        or channel.get("schema_version") != RELEASE_CHANNEL_SCHEMA_VERSION
        or channel.get("commit_sha") != expected_commit
    ):
        _fail("object_locked_release_invalid")
    try:
        gateway = validators.gateway.validate_release_manifest(
            channel["gateway_release_manifest"]
        )
        validator = validators.validator.validate_validator_release_manifest(
            channel["validator_release_manifest"]
        )
    except (Exception, MemoryError):
        _fail("object_locked_release_invalid")
    body = {
        "schema_version": RELEASE_CHANNEL_SCHEMA_VERSION,
        "commit_sha": gateway["commit_sha"],
        "gateway_release_manifest": gateway,
        "validator_release_manifest": validator,
    }
    if (
        gateway["commit_sha"] != expected_commit
        or validator["release"]["commit_sha"] != expected_commit
        or channel.get("channel_hash") != _sha256_json(body)
    ):
        _fail("object_locked_release_invalid")
    entries = []
    for physical_role, summary in sorted(gateway["roles"].items()):
        entries.append(
            {
                "physical_role": physical_role,
                "role": summary["service_role"],
                "commit_sha": summary["commit_sha"],
                "pcr0": summary["pcr0"],
                "build_manifest_hash": summary["execution_manifest_hash"],
                "dependency_lock_hash": summary["dependency_lock_hash"],
                "verified_build_count": summary["verified_build_count"],
            }
        )
    release = validator["release"]
    entries.append(
        {
            "physical_role": "validator_weights",
            "role": "validator_weights",
            "commit_sha": release["commit_sha"],
            "pcr0": release["pcr0"],
            "build_manifest_hash": release["app_manifest_hash"],
            "dependency_lock_hash": release["dependency_lock_hash"],
            "verified_build_count": validator["verified_build_count"],
        }
    )
    return {"schema_version": IDENTITY_CACHE_SCHEMA_VERSION, "entries": entries}


def _load_immutable_release_identity(
    evidence: Mapping[str, Any],
    *,
    http_open: Callable[..., Any],
    validators: SimpleNamespace,
) -> Dict[str, Any]:
    fields = {
        "schema_version",
        "commit_sha",
        "release_channel_version_id",
        "release_channel_get_url",
        "release_channel_head_url",
    }
    if (
        not isinstance(evidence, Mapping)
        or set(evidence) != fields
        or evidence.get("schema_version") != RELEASE_EVIDENCE_SCHEMA_VERSION
    ):
        _fail("object_locked_release_invalid")
    commit = evidence.get("commit_sha")
    version_id = evidence.get("release_channel_version_id")
    if type(commit) is not str or not _SHA_RE.fullmatch(commit):
        _fail("object_locked_release_invalid")
    version_id = _bounded_ascii(
        version_id,
        maximum_bytes=MAX_VERSION_ID_BYTES,
        code="object_locked_release_invalid",
    )
    get_url = _release_channel_url(
        evidence.get("release_channel_get_url"),
        commit=commit,
        version_id=version_id,
    )
    head_url = _release_channel_url(
        evidence.get("release_channel_head_url"),
        commit=commit,
        version_id=version_id,
    )
    try:
        with http_open(head_url, method="HEAD") as response:
            if (
                type(getattr(response, "status", None)) is not int
                or response.status != 200
            ):
                _fail("object_locked_release_invalid")
            if response.geturl() != head_url:
                _fail("object_locked_release_invalid")
            lock_mode = _response_header(response, "x-amz-object-lock-mode")
            retain_text = _response_header(
                response, "x-amz-object-lock-retain-until-date"
            )
            head_version = _response_header(response, "x-amz-version-id")
        with http_open(get_url, method="GET") as response:
            if (
                type(getattr(response, "status", None)) is not int
                or response.status != 200
            ):
                _fail("object_locked_release_invalid")
            if response.geturl() != get_url:
                _fail("object_locked_release_invalid")
            get_version = _response_header(response, "x-amz-version-id")
            payload = response.read(MAX_RELEASE_CHANNEL_BYTES + 1)
    except WeightSubmissionEvidenceProbeError:
        raise
    except (Exception, MemoryError):
        _fail("object_locked_release_invalid")
    if (
        lock_mode.upper() != "COMPLIANCE"
        or head_version != version_id
        or get_version != version_id
        or type(payload) is not bytes
        or not payload
        or len(payload) > MAX_RELEASE_CHANNEL_BYTES
    ):
        _fail("object_locked_release_invalid")
    try:
        retain_until = datetime.fromisoformat(retain_text.replace("Z", "+00:00"))
        if retain_until.tzinfo is None or retain_until <= datetime.now(timezone.utc):
            raise ValueError
    except (TypeError, ValueError, OverflowError):
        _fail("object_locked_release_invalid")
    channel = _strict_json(
        payload,
        maximum_bytes=MAX_RELEASE_CHANNEL_BYTES,
        invalid_code="object_locked_release_invalid",
    )
    return _release_identity_from_channel(
        channel, expected_commit=commit, validators=validators
    )


def _load_candidate_runtime(binding: _CandidateRepository) -> SimpleNamespace:
    _activate_candidate_imports(binding)
    release_validators = _load_candidate_release_validators(binding)
    release_contract = _candidate_release_contract(release_validators)
    public_key_verifier_policy = _preflight_public_key_verifier_dependency(binding)
    try:
        modules = {
            name: importlib.import_module(name) for name in _RUNTIME_MODULE_NAMES
        }
    except (Exception, MemoryError):
        _fail("repository_import_invalid")
    binding.recheck()
    binding.bind_loaded_modules()
    _reject_forbidden_runtime_modules(
        public_key_verifier_policy=public_key_verifier_policy
    )

    def load_release_identity(evidence, *, http_open):
        return _load_immutable_release_identity(
            evidence,
            http_open=http_open,
            validators=release_validators,
        )

    return SimpleNamespace(
        SubnetEpochCutover=modules["Leadpoet.utils.subnet_epoch"].SubnetEpochCutover,
        derive_ancestry_lineage_id_v2=modules[
            "leadpoet_canonical.ancestry_checkpoint_v2"
        ].derive_ancestry_lineage_id_v2,
        identity_cache_schema=IDENTITY_CACHE_SCHEMA_VERSION,
        load_immutable_release_identity=load_release_identity,
        release_contract=release_contract,
        chain=modules["leadpoet_canonical.chain_source_v2"],
        authority_schema=modules[
            "leadpoet_canonical.compact_auditor_authority_v2"
        ].COMPACT_PUBLISHED_WEIGHT_AUTHORITY_SCHEMA_VERSION,
        verify_authority=modules[
            "leadpoet_canonical.compact_auditor_authority_v2"
        ].verify_compact_published_weight_authority_v2,
        validate_chain_signing_profile=modules[
            "leadpoet_canonical.hotkey_authority_v2"
        ].validate_chain_signing_profile,
    )


def _load_candidate_configuration(
    binding: _CandidateRepository, runtime: SimpleNamespace
) -> tuple[Dict[str, Any], Any]:
    profile_mapping = _strict_json(
        binding.read_bound_file(PROFILE_RELATIVE_PATH, maximum_bytes=1024 * 1024),
        maximum_bytes=1024 * 1024,
        invalid_code="candidate_configuration_invalid",
    )
    cutover_mapping = _strict_json(
        binding.read_bound_file(CUTOVER_RELATIVE_PATH, maximum_bytes=1024 * 1024),
        maximum_bytes=1024 * 1024,
        invalid_code="candidate_configuration_invalid",
    )
    try:
        profile = runtime.validate_chain_signing_profile(profile_mapping)
        cutover = runtime.SubnetEpochCutover.from_mapping(cutover_mapping)
    except (Exception, MemoryError):
        _fail("candidate_configuration_invalid")
    binding.recheck()
    binding.bind_loaded_modules()
    binding.read_bound_file(PROFILE_RELATIVE_PATH, maximum_bytes=1024 * 1024)
    binding.read_bound_file(CUTOVER_RELATIVE_PATH, maximum_bytes=1024 * 1024)
    return profile, cutover


class FinalizedWebSocketChainReader:
    """Read exact finalized storage through four allowlisted RPC methods."""

    def __init__(
        self,
        endpoint: str,
        *,
        chain_api: Any,
        connector: Optional[Callable[..., Any]] = None,
        timeout_seconds: float = 30.0,
    ):
        text = _bounded_url(
            endpoint,
            scheme="wss",
            code="chain_endpoint_invalid",
            allow_query=False,
        )
        parsed = urlsplit(text)
        if parsed.path not in ("", "/"):
            _fail("chain_endpoint_invalid")
        self.endpoint = urlunsplit(("wss", parsed.netloc, "/", "", ""))
        self.chain = chain_api
        self._connector = connector
        self._timeout_seconds = float(timeout_seconds)

    def _connect(self):
        if self._connector is not None:
            return self._connector(
                self.endpoint,
                open_timeout=self._timeout_seconds,
                close_timeout=5,
                max_size=self.chain.CHAIN_MAX_RPC_RESPONSE_BYTES,
                proxy=None,
            )
        try:
            from websockets.sync.client import connect
        except (Exception, MemoryError):
            _fail("chain_websocket_dependency_unavailable")
        return connect(
            self.endpoint,
            open_timeout=self._timeout_seconds,
            close_timeout=5,
            max_size=self.chain.CHAIN_MAX_RPC_RESPONSE_BYTES,
            proxy=None,
        )

    def _call_on_connection(
        self,
        connection: Any,
        *,
        method: str,
        params: Sequence[Any],
        request_id: int,
    ) -> Any:
        if str(method) not in _READ_ONLY_RPC_METHODS:
            _fail("write_capable_chain_call_rejected")
        try:
            request_body = self.chain.json_rpc_request(
                str(method), list(params), request_id
            )
            connection.send(request_body.decode("ascii"))
            received = connection.recv(timeout=self._timeout_seconds)
            response_body = (
                bytes(received)
                if isinstance(received, (bytes, bytearray))
                else str(received).encode("utf-8")
            )
            if (
                not response_body
                or len(response_body) > self.chain.CHAIN_MAX_RPC_RESPONSE_BYTES
            ):
                _fail("chain_response_size_invalid")
            return self.chain.parse_json_rpc_response(response_body, request_id)
        except WeightSubmissionEvidenceProbeError:
            raise
        except (Exception, MemoryError):
            _fail("chain_read_unavailable")

    def call(self, method: str, params: Sequence[Any]) -> Any:
        if str(method) not in _READ_ONLY_RPC_METHODS:
            _fail("write_capable_chain_call_rejected")
        try:
            with self._connect() as connection:
                return self._call_on_connection(
                    connection, method=method, params=params, request_id=1
                )
        except WeightSubmissionEvidenceProbeError:
            raise
        except (Exception, MemoryError):
            _fail("chain_read_unavailable")

    def read_finalized_state(
        self, *, netuid: int, hotkeys: Sequence[str]
    ) -> Dict[str, Any]:
        request_id = 0
        try:
            with self._connect() as connection:

                def rpc(method: str, params: Sequence[Any]) -> Any:
                    nonlocal request_id
                    request_id += 1
                    return self._call_on_connection(
                        connection,
                        method=method,
                        params=params,
                        request_id=request_id,
                    )

                finalized = "0x" + self.chain.normalize_raw_hash(
                    rpc("chain_getFinalizedHead", ()), "finalized head"
                )
                header = self.chain.parse_finalized_header(
                    rpc("chain_getHeader", (finalized,))
                )
                metagraph = self.chain.decode_selective_metagraph_result(
                    rpc(
                        "state_call",
                        (
                            self.chain.CHAIN_RPC_METHOD,
                            self.chain.encode_selective_metagraph_params(
                                netuid=netuid, mechid=0
                            ),
                            finalized,
                        ),
                    )
                )
                subnet_epoch_index = self.chain.decode_subnet_epoch_storage(
                    rpc(
                        "state_getStorage",
                        (
                            self.chain.subnet_epoch_storage_key(
                                storage_name="SubnetEpochIndex", netuid=netuid
                            ),
                            finalized,
                        ),
                    ),
                    storage_name="SubnetEpochIndex",
                )
                last_updates = self.chain.decode_last_update_storage(
                    rpc(
                        "state_getStorage",
                        (self.chain.last_update_storage_key(netuid=netuid), finalized),
                    )
                )
                metagraph_hotkeys = list(metagraph["hotkeys"])
                if (
                    int(metagraph["netuid"]) != netuid
                    or int(metagraph["block"]) != int(header["block"])
                    or len(metagraph_hotkeys) != len(set(metagraph_hotkeys))
                ):
                    _fail("finalized_metagraph_invalid")
                validators = []
                for position, hotkey in enumerate(hotkeys):
                    matches = [
                        index
                        for index, observed in enumerate(metagraph_hotkeys)
                        if observed == hotkey
                    ]
                    if len(matches) != 1:
                        _fail(
                            "auditor_hotkey_missing"
                            if position
                            else "primary_hotkey_missing"
                        )
                    uid = matches[0]
                    if uid >= len(last_updates):
                        _fail("stale_validator_uid")
                    weights = self.chain.decode_weights_storage(
                        rpc(
                            "state_getStorage",
                            (
                                self.chain.weights_storage_key(
                                    netuid=netuid, validator_uid=uid
                                ),
                                finalized,
                            ),
                        )
                    )
                    validators.append(
                        {
                            "hotkey": hotkey,
                            "uid": uid,
                            "mechanism_id": 0,
                            "last_update": int(last_updates[uid]),
                            "weights": [list(pair) for pair in weights],
                        }
                    )
                return {
                    "block_hash": finalized,
                    "block": int(header["block"]),
                    "subnet_epoch_index": int(subnet_epoch_index),
                    "metagraph_hotkeys": metagraph_hotkeys,
                    "validators": validators,
                }
        except WeightSubmissionEvidenceProbeError:
            raise
        except (Exception, MemoryError):
            _fail("chain_read_unavailable")


def _bounded_auditors(values: Any) -> list[str]:
    try:
        iterator = iter(values)
    except (Exception, MemoryError):
        _fail("auditor_hotkeys_invalid")
    auditors = []
    for _index in range(MAX_AUDITOR_HOTKEYS + 1):
        try:
            value = next(iterator)
        except StopIteration:
            break
        except (Exception, MemoryError):
            _fail("auditor_hotkeys_invalid")
        if len(auditors) == MAX_AUDITOR_HOTKEYS:
            _fail("auditor_hotkey_limit_exceeded")
        if type(value) is not str:
            _fail("auditor_hotkeys_invalid")
        auditors.append(
            _bounded_ascii(
                value,
                maximum_bytes=MAX_HOTKEY_INPUT_BYTES,
                code="auditor_hotkeys_invalid",
            ).strip()
        )
    return auditors


def _normalize_inputs(
    *, candidate_sha: str, netuid: int, epoch_id: int, auditor_hotkeys: Any
) -> tuple[str, int, int, list[str]]:
    if type(candidate_sha) is not str:
        _fail("candidate_sha_invalid")
    candidate = candidate_sha.lower()
    if not _SHA_RE.fullmatch(candidate):
        _fail("candidate_sha_invalid")
    if type(netuid) is not int or not 0 < netuid <= 0xFFFF:
        _fail("netuid_invalid")
    if type(epoch_id) is not int or not 0 <= epoch_id < 1 << 64:
        _fail("epoch_id_invalid")
    auditors = _bounded_auditors(auditor_hotkeys)
    if (
        not auditors
        or any(not _HOTKEY_RE.fullmatch(value) for value in auditors)
        or len(auditors) != len(set(auditors))
    ):
        _fail("auditor_hotkeys_invalid")
    return candidate, netuid, epoch_id, auditors


def _validate_release_identity_cache(
    identity_cache: Mapping[str, Any],
    *,
    candidate_sha: str,
    identity_cache_schema: str,
    release_contract: Mapping[str, Any],
) -> None:
    if (
        not isinstance(identity_cache, Mapping)
        or set(identity_cache) != {"schema_version", "entries"}
        or identity_cache.get("schema_version") != identity_cache_schema
    ):
        _fail("release_identity_invalid")
    roles = dict(release_contract["roles"])
    counts = dict(release_contract["build_counts"])
    entries = identity_cache.get("entries")
    if not isinstance(entries, list) or len(entries) != len(roles):
        _fail("release_identity_invalid")
    observed: Dict[str, str] = {}
    for item in entries:
        if not isinstance(item, Mapping) or set(item) != _RELEASE_IDENTITY_FIELDS:
            _fail("release_identity_invalid")
        physical_role = str(item.get("physical_role") or "")
        if physical_role in observed:
            _fail("release_identity_invalid")
        if str(item.get("commit_sha") or "").lower() != candidate_sha:
            _fail("release_sha_mismatch")
        service_role = str(item.get("role") or "")
        if (
            roles.get(physical_role) != service_role
            or item.get("verified_build_count") != counts.get(physical_role)
            or counts.get(physical_role) != 6
        ):
            _fail("release_identity_invalid")
        observed[physical_role] = service_role
    if observed != roles or roles != _EXPECTED_RELEASE_IDENTITY_ROLES:
        _fail("release_identity_invalid")


def _extract_primary_reveal_index(
    authority: Mapping[str, Any], *, epoch_id: int, netuid: int
) -> int:
    try:
        authorization = authority["finalization"]["compact_submission"]["finalization"][
            "extrinsic_authorization"
        ]
        if not isinstance(authorization, Mapping):
            raise TypeError
        if (
            type(authorization["epoch_id"]) is not int
            or type(authorization["netuid"]) is not int
        ):
            raise TypeError
        if authorization["epoch_id"] != epoch_id or authorization["netuid"] != netuid:
            _fail("authority_identity_mismatch")
        index = authorization["subnet_epoch_index"]
        if type(index) is not int or not 0 <= index < 1 << 64:
            raise TypeError
        return index
    except WeightSubmissionEvidenceProbeError:
        raise
    except (KeyError, TypeError, ValueError):
        _fail("authority_reveal_identity_missing")


def _normalized_expected_vector(
    verified: Mapping[str, Any],
) -> tuple[tuple[int, int], ...]:
    try:
        uids = list(verified["uids"])
        weights = list(verified["weights_u16"])
        if any(type(value) is not int for value in [*uids, *weights]):
            raise TypeError
        pairs = tuple(zip(uids, weights))
    except (KeyError, TypeError, ValueError):
        _fail("canonical_vector_invalid")
    if (
        not pairs
        or len(pairs) != len(uids)
        or len(pairs) != len(weights)
        or pairs != tuple(sorted(pairs))
        or len({uid for uid, _weight in pairs}) != len(pairs)
        or any(
            uid < 0 or uid > 0xFFFF or weight <= 0 or weight > 0xFFFF
            for uid, weight in pairs
        )
    ):
        _fail("canonical_vector_invalid")
    return pairs


def _validate_chain_readback(
    chain_state: Mapping[str, Any],
    *,
    chain_api: Any,
    primary_hotkey: str,
    auditor_hotkeys: Sequence[str],
    expected_vector: tuple[tuple[int, int], ...],
    bundle_block: int,
    primary_finalized_block: int,
    target_subnet_epoch_index: int,
    reveal_period_epochs: int,
) -> tuple[list[Dict[str, Any]], int, str]:
    if not isinstance(chain_state, Mapping) or set(chain_state) != _CHAIN_STATE_FIELDS:
        _fail("finalized_chain_state_invalid")
    try:
        block_hash = "0x" + chain_api.normalize_raw_hash(
            chain_state["block_hash"], "finalized head"
        )
        block = chain_state["block"]
        current_index = chain_state["subnet_epoch_index"]
        metagraph = list(chain_state["metagraph_hotkeys"])
        raw_states = list(chain_state["validators"])
        if type(block) is not int or type(current_index) is not int:
            raise TypeError
    except (KeyError, TypeError, ValueError):
        _fail("finalized_chain_state_invalid")
    requested = [primary_hotkey, *auditor_hotkeys]
    if (
        block < primary_finalized_block
        or current_index < 0
        or len(metagraph) != len(set(metagraph))
        or len(raw_states) != len(requested)
    ):
        _fail("finalized_chain_state_invalid")
    reveal_index = target_subnet_epoch_index + reveal_period_epochs
    if current_index < reveal_index:
        _fail("reveal_pending")
    expected_hash = _sha256_json(
        {"mechanism_id": 0, "weights": [list(pair) for pair in expected_vector]}
    )
    by_hotkey = {}
    for raw in raw_states:
        if not isinstance(raw, Mapping) or set(raw) != _VALIDATOR_STATE_FIELDS:
            _fail("finalized_validator_state_invalid")
        hotkey = str(raw.get("hotkey") or "")
        if hotkey in by_hotkey or hotkey not in requested:
            _fail("finalized_validator_state_invalid")
        by_hotkey[hotkey] = raw
    summaries = []
    for position, hotkey in enumerate(requested):
        if hotkey not in metagraph:
            _fail("auditor_hotkey_missing" if position else "primary_hotkey_missing")
        state = by_hotkey.get(hotkey)
        if state is None:
            _fail("auditor_hotkey_missing" if position else "primary_hotkey_missing")
        try:
            uid = state["uid"]
            mechanism_id = state["mechanism_id"]
            last_update = state["last_update"]
            if any(
                type(value) is not int for value in (uid, mechanism_id, last_update)
            ):
                raise TypeError
            observed_vector = tuple(
                (pair[0], pair[1])
                for pair in state["weights"]
                if type(pair[0]) is int and type(pair[1]) is int
            )
            if len(observed_vector) != len(state["weights"]):
                raise TypeError
        except (KeyError, TypeError, ValueError, IndexError):
            _fail("finalized_validator_state_invalid")
        if uid != metagraph.index(hotkey):
            _fail("stale_validator_uid")
        if mechanism_id != 0:
            _fail("mechanism_mismatch")
        if (
            last_update <= bundle_block
            or last_update < primary_finalized_block
            or last_update > block
        ):
            _fail("last_update_not_advanced")
        if not observed_vector:
            _fail("reveal_pending")
        if observed_vector != expected_vector:
            _fail("vector_divergence")
        summaries.append(
            {
                "role": "primary" if position == 0 else "auditor",
                "hotkey_hash": _sha256_json({"hotkey": hotkey}),
                "uid": uid,
                "mechanism_id": 0,
                "last_update": last_update,
                "destination_count": len(observed_vector),
                "vector_hash": expected_hash,
            }
        )
    return summaries, block, block_hash


def _verify_weight_submission_evidence_core(
    *,
    candidate: str,
    netuid: int,
    epoch_id: int,
    auditors: Sequence[str],
    gateway_url: str,
    runtime: SimpleNamespace,
    profile: Mapping[str, Any],
    cutover: Any,
    release_contract: Mapping[str, Any],
    binding: Optional[_CandidateRepository] = None,
    http: Optional[Any] = None,
    chain_reader: Optional[Any] = None,
    release_identity_loader: Optional[Callable[..., Mapping[str, Any]]] = None,
    authority_verifier: Optional[Callable[..., Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    if (
        profile.get("mechid") != 0
        or cutover.netuid != netuid
        or str(profile.get("genesis_hash") or "").lower()
        != str(cutover.network_genesis_hash).lower().removeprefix("0x")
    ):
        _fail("candidate_configuration_invalid")
    readonly_http = http or ReadOnlyHttp(gateway_url)
    load_release = release_identity_loader or runtime.load_immutable_release_identity
    verify_authority = authority_verifier or runtime.verify_authority

    if binding is not None:
        binding.recheck()
        binding.bind_loaded_modules()
        _reject_forbidden_runtime_modules(
            public_key_verifier_policy=binding.public_key_verifier_policy
        )
    build_info = readonly_http.get_json("/build-info", maximum_bytes=64 * 1024)
    if (
        build_info.get("is_commit_known") is not True
        or str(build_info.get("git_commit") or "").lower() != candidate
    ):
        _fail("candidate_sha_mismatch")

    if binding is not None:
        binding.recheck()
        binding.bind_loaded_modules()
        _reject_forbidden_runtime_modules(
            public_key_verifier_policy=binding.public_key_verifier_policy
        )
    release_evidence = readonly_http.get_json(
        "/weights/v2/immutable-release-evidence/" + candidate,
        maximum_bytes=MAX_RELEASE_EVIDENCE_BYTES,
    )
    if str(release_evidence.get("commit_sha") or "").lower() != candidate:
        _fail("release_sha_mismatch")
    try:
        identity_cache = dict(
            load_release(release_evidence, http_open=readonly_http.open_exact_url)
        )
    except WeightSubmissionEvidenceProbeError:
        raise
    except (Exception, MemoryError):
        _fail("object_locked_release_invalid")
    if binding is not None:
        binding.recheck()
        binding.bind_loaded_modules()
        _reject_forbidden_runtime_modules(
            public_key_verifier_policy=binding.public_key_verifier_policy
        )
    _validate_release_identity_cache(
        identity_cache,
        candidate_sha=candidate,
        identity_cache_schema=runtime.identity_cache_schema,
        release_contract=release_contract,
    )

    authority = readonly_http.get_json(
        "/weights/v2/published-compact/%d/%d" % (netuid, epoch_id),
        maximum_bytes=MAX_GATEWAY_DOCUMENT_BYTES,
    )
    if (
        authority.get("schema_version") != runtime.authority_schema
        or authority.get("authority_stage") != "finalized"
        or not isinstance(authority.get("finalization"), Mapping)
    ):
        _fail("authority_not_finalized")
    lineage_id = runtime.derive_ancestry_lineage_id_v2(
        cutover_mapping_hash=str(cutover.mapping_hash),
        network_genesis_hash=str(cutover.network_genesis_hash),
        netuid=netuid,
    )
    try:
        verified = dict(
            verify_authority(
                authority,
                identity_cache=identity_cache,
                chain_signing_profile=profile,
                expected_lineage_id=lineage_id,
                expected_chain=str(profile["chain_endpoint"]),
            )
        )
    except WeightSubmissionEvidenceProbeError:
        raise
    except (Exception, MemoryError):
        _fail("authority_verification_failed")
    if (
        type(verified.get("netuid")) is not int
        or type(verified.get("epoch_id")) is not int
        or verified["netuid"] != netuid
        or verified["epoch_id"] != epoch_id
        or verified.get("authority_stage") != "finalized"
    ):
        _fail("authority_identity_mismatch")
    primary_hotkey = str(verified.get("validator_hotkey") or "")
    if not _HOTKEY_RE.fullmatch(primary_hotkey) or primary_hotkey in auditors:
        _fail("primary_hotkey_invalid")
    target_index = _extract_primary_reveal_index(
        authority, epoch_id=epoch_id, netuid=netuid
    )
    try:
        if cutover.settlement_epoch_id(target_index) != epoch_id:
            _fail("authority_epoch_mapping_mismatch")
    except WeightSubmissionEvidenceProbeError:
        raise
    except (Exception, MemoryError):
        _fail("authority_epoch_mapping_mismatch")
    expected_vector = _normalized_expected_vector(verified)
    primary_finalized_block = verified.get("finalized_block")
    bundle_block = verified.get("block")
    if (
        type(primary_finalized_block) is not int
        or type(bundle_block) is not int
        or primary_finalized_block <= bundle_block
    ):
        _fail("authority_finalization_invalid")

    if binding is not None:
        binding.recheck()
        binding.bind_loaded_modules()
        _reject_forbidden_runtime_modules(
            public_key_verifier_policy=binding.public_key_verifier_policy
        )
    reader = chain_reader or FinalizedWebSocketChainReader(
        str(profile["chain_endpoint"]), chain_api=runtime.chain
    )
    try:
        chain_state = reader.read_finalized_state(
            netuid=netuid, hotkeys=[primary_hotkey, *auditors]
        )
    except WeightSubmissionEvidenceProbeError:
        raise
    except (Exception, MemoryError):
        _fail("chain_read_unavailable")
    validators, head_block, head_hash = _validate_chain_readback(
        chain_state,
        chain_api=runtime.chain,
        primary_hotkey=primary_hotkey,
        auditor_hotkeys=auditors,
        expected_vector=expected_vector,
        bundle_block=bundle_block,
        primary_finalized_block=primary_finalized_block,
        target_subnet_epoch_index=target_index,
        reveal_period_epochs=profile["subnet_reveal_period_epochs"],
    )
    body = {
        "schema_version": SCHEMA_VERSION,
        "candidate_sha": candidate,
        "netuid": netuid,
        "epoch_id": epoch_id,
        "release_channel_version_hash": _sha256_json(
            {"version_id": str(release_evidence["release_channel_version_id"])}
        ),
        "release_identity_hash": _sha256_json(identity_cache),
        "authority_hash": str(verified["authority_hash"]),
        "bundle_hash": str(verified["bundle_hash"]),
        "weights_hash": str(verified["weights_hash"]),
        "weight_finalization_event_hash": str(
            verified["weight_finalization_event_hash"]
        ),
        "primary_finalized_block": primary_finalized_block,
        "finalized_head_block": head_block,
        "finalized_head_hash": head_hash,
        "target_subnet_epoch_index": target_index,
        "auditor_count": len(auditors),
        "validator_count": len(validators),
        "destination_count": len(expected_vector),
        "validators": validators,
    }
    return {**body, "evidence_hash": _sha256_json(body)}


def verify_weight_submission_evidence_v2(
    *,
    candidate_sha: str,
    netuid: int,
    epoch_id: int,
    auditor_hotkeys: Any,
    gateway_url: str = DEFAULT_GATEWAY_URL,
) -> Dict[str, Any]:
    if not sys.flags.isolated:
        _fail("isolated_python_required")
    candidate, normalized_netuid, normalized_epoch, auditors = _normalize_inputs(
        candidate_sha=candidate_sha,
        netuid=netuid,
        epoch_id=epoch_id,
        auditor_hotkeys=auditor_hotkeys,
    )
    binding = _CandidateRepository.preflight(ROOT, candidate)
    runtime = _load_candidate_runtime(binding)
    profile, cutover = _load_candidate_configuration(binding, runtime)
    binding.recheck()
    binding.bind_loaded_modules()
    _reject_forbidden_runtime_modules(
        public_key_verifier_policy=binding.public_key_verifier_policy
    )
    return _verify_weight_submission_evidence_core(
        candidate=candidate,
        netuid=normalized_netuid,
        epoch_id=normalized_epoch,
        auditors=auditors,
        gateway_url=gateway_url,
        runtime=runtime,
        profile=profile,
        cutover=cutover,
        release_contract=runtime.release_contract,
        binding=binding,
    )


def _cli_uint(value: str, *, maximum: int, code: str) -> int:
    if type(value) is not str:
        _fail(code)
    text = value
    if not re.fullmatch(r"[0-9]{1,20}", text):
        _fail(code)
    parsed = int(text, 10)
    if not 0 <= parsed <= maximum:
        _fail(code)
    return parsed


class _FixedArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        del message
        _fail("arguments_invalid")


class _BoundedAuditorAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        del parser, option_string
        current = list(getattr(namespace, self.dest, None) or ())
        if len(current) >= MAX_AUDITOR_HOTKEYS:
            _fail("auditor_hotkey_limit_exceeded")
        current.append(
            _bounded_ascii(
                values,
                maximum_bytes=MAX_HOTKEY_INPUT_BYTES,
                code="auditor_hotkeys_invalid",
            )
        )
        setattr(namespace, self.dest, current)


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        parser = _FixedArgumentParser(
            description="Prove finalized primary/auditor canonical weights read-only."
        )
        parser.add_argument("--candidate-sha", required=True)
        parser.add_argument("--netuid", required=True)
        parser.add_argument("--epoch-id", required=True)
        parser.add_argument(
            "--auditor-hotkey", required=True, action=_BoundedAuditorAction
        )
        parser.add_argument("--gateway-url", default=DEFAULT_GATEWAY_URL)
        args = parser.parse_args(argv)
        evidence = verify_weight_submission_evidence_v2(
            candidate_sha=args.candidate_sha,
            netuid=_cli_uint(args.netuid, maximum=0xFFFF, code="netuid_invalid"),
            epoch_id=_cli_uint(
                args.epoch_id, maximum=(1 << 64) - 1, code="epoch_id_invalid"
            ),
            auditor_hotkeys=args.auditor_hotkey,
            gateway_url=args.gateway_url,
        )
        serialized = _canonical_json(evidence)
        sys.stdout.write(serialized + "\n")
    except WeightSubmissionEvidenceProbeError as exc:
        try:
            sys.stderr.write("ERROR:" + exc.code + "\n")
        except Exception:
            pass
        return 1
    except (Exception, MemoryError):
        try:
            sys.stderr.write("ERROR:probe_failed\n")
        except Exception:
            pass
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
