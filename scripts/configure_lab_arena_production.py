#!/usr/bin/env python3
"""Safely prepare or apply the fixed Lab Arena production environment.

Secret documents are read and changed on their owning hosts. The complete
documents and the scoped service key travel only on SSH stdin and stay in memory.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence

GATEWAY_HOST = "ec2-user@52.91.135.79"
VALIDATOR_HOST = "ec2-user@100.59.201.156"
GATEWAY_SECRET = "leadpoet/prod/gateway/env"
VALIDATOR_SECRET = "leadpoet/prod/validator/env"
DEFAULT_SSH_KEY = Path.home() / ".ssh" / "leadpoet-2026-07-28.pem"
AUTH_ENV = "LEADPOET_LAB_ARENA_PRODUCTION_APPLY"
KNOWN_ACCOUNTS = frozenset({"187445349696", "493765492819"})
KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class ConfigurationError(RuntimeError):
    pass


def _dotenv_key(line: str) -> str:
    candidate = line.strip()
    if not candidate or candidate.startswith("#"):
        return ""
    if candidate.startswith("export "):
        candidate = candidate[7:].strip()
    key = candidate.partition("=")[0].strip()
    return key if KEY_RE.fullmatch(key) else ""


def update_document(raw: str, updates: Mapping[str, str]) -> tuple[str, str]:
    """Merge values while retaining the source JSON or dotenv format."""
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        parsed = None
    if isinstance(parsed, dict):
        parsed.update(updates)
        return json.dumps(parsed, separators=(",", ":")), "json"
    if parsed is not None:
        raise ConfigurationError("secret document JSON must be an object")

    pending = dict(updates)
    lines: list[str] = []
    for line in raw.replace("\x00", "\n").splitlines():
        key = _dotenv_key(line)
        if key in pending:
            lines.append("export %s=%s" % (key, shlex.quote(pending.pop(key))))
        else:
            lines.append(line)
    for key, value in pending.items():
        lines.append("export %s=%s" % (key, shlex.quote(value)))
    return "\n".join(lines) + "\n", "dotenv"


def _required_alias(values: Mapping[str, str], name: str) -> str:
    value = str(values.get(name) or "").strip()
    if not value:
        raise ConfigurationError("required existing value is missing: %s" % name)
    return value


def gateway_updates(values: Mapping[str, str], args: argparse.Namespace, service_key: str) -> dict[str, str]:
    _validate_service_key(service_key)
    return {
        "LAB_ARENA_MODE": "live",
        "LAB_ARENA_REWARDS_ENABLED": "false",
        "LAB_ARENA_OPENROUTER_API_KEY": _required_alias(values, "OPENROUTER_API_KEY"),
        "LAB_ARENA_SCRAPINGDOG_API_KEY": _required_alias(values, "SCRAPINGDOG_API_KEY"),
        "LAB_ARENA_DEEPLINE_API_KEY": _required_alias(values, "DEEPLINE_API_KEY"),
        "LAB_ARENA_SUPABASE_URL": _required_alias(values, "SUPABASE_URL"),
        "LAB_ARENA_SUPABASE_ANON_KEY": _required_alias(values, "SUPABASE_ANON_KEY"),
        "LAB_ARENA_SERVICE_KEY": service_key,
        "LAB_ARENA_BUCKET": args.bucket,
        "LAB_ARENA_SCORER_IMAGE": args.scorer_image,
        "LAB_ARENA_RUNNER_HOTKEYS": args.runner_hotkey,
        "LAB_ARENA_BASELINE_HOTKEY": args.baseline_hotkey,
        "LAB_ARENA_CHAIN_ENDPOINT": args.chain_endpoint,
        "LAB_ARENA_DAILY_CUTOFF_UTC": str(args.daily_cutoff_utc),
    }


def _registry_repository(image: str) -> str:
    image_name = image.split("@", 1)[0]
    last = image_name.rsplit("/", 1)[-1]
    if ":" in last:
        image_name = image_name.rsplit(":", 1)[0]
    return image_name


def gateway_nonsecret_updates(args: argparse.Namespace) -> dict[str, str]:
    image_name = _registry_repository(args.scorer_image)
    return {
        "LAB_ARENA_MODE": "live",
        "LAB_ARENA_REWARDS_ENABLED": "false",
        "LAB_ARENA_BUCKET": args.bucket,
        "LAB_ARENA_SCORER_IMAGE": args.scorer_image,
        "LAB_ARENA_REGISTRY_REPOSITORY": image_name,
        "LAB_ARENA_RUNNER_HOTKEYS": args.runner_hotkey,
        "LAB_ARENA_BASELINE_HOTKEY": args.baseline_hotkey,
        "LAB_ARENA_CHAIN_ENDPOINT": args.chain_endpoint,
        "LAB_ARENA_DAILY_CUTOFF_UTC": str(args.daily_cutoff_utc),
    }


def validator_updates(args: argparse.Namespace) -> dict[str, str]:
    # Wallet, hotkey, path, work directory, and runsc use validator_restart.sh defaults.
    return {
        "LAB_ARENA_MODE": "live",
        "LAB_ARENA_REWARDS_ENABLED": "false",
        "LAB_ARENA_API_BASE_URL": args.api_base_url,
        "LAB_ARENA_REGISTRY_REPOSITORY": _registry_repository(args.scorer_image),
        "LAB_ARENA_WALLET_NAME": "arena_runner",
        "LAB_ARENA_HOTKEY_NAME": "default",
        "LAB_ARENA_WALLET_PATH": "/var/lib/lab-arena/runner-wallets",
    }


def _validate_service_key(service_key: str) -> None:
    if not service_key.startswith("sb_secret_") or any(ch.isspace() for ch in service_key):
        raise ConfigurationError("scoped service key is malformed")


_REMOTE = r'''
import json, re, shlex, subprocess, sys, uuid

def fail(code):
    print(json.dumps({"ok": False, "code": code}, separators=(",", ":")))
    raise SystemExit(1)

def aws(command, input_text=None):
    result = subprocess.run(["aws"] + command, input=input_text, text=True,
                            capture_output=True, check=False)
    if result.returncode:
        fail("aws_command_failed")
    return result.stdout

def get(secret_id, version_id=""):
    command = ["secretsmanager", "get-secret-value", "--secret-id", secret_id,
               "--output", "json"]
    if version_id:
        command += ["--version-id", version_id]
    try:
        value = json.loads(aws(command))
        return str(value["VersionId"]), str(value["SecretString"])
    except Exception:
        fail("secret_read_invalid")

request = json.loads(sys.stdin.read())
identity = json.loads(aws(["sts", "get-caller-identity", "--output", "json"]))
account = str(identity.get("Account") or "")
if account not in request["allowed_accounts"]:
    fail("account_not_allowed")
secret_id = request["secret_id"]
before_id, raw = get(secret_id)

def key_of(line):
    value = line.strip()
    if value.startswith("export "):
        value = value[7:].strip()
    key = value.partition("=")[0].strip()
    return key if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) else ""

try:
    values = json.loads(raw)
except Exception:
    values = None
occurrences = {}
if isinstance(values, dict):
    document_format = "json"
    occurrences = {str(key): 1 for key in values}
else:
    if values is not None:
        fail("secret_document_invalid")
    document_format = "dotenv"
    values = {}
    for line in raw.replace("\x00", "\n").splitlines():
        key = key_of(line)
        if not key:
            continue
        value = line.strip()
        if value.startswith("export "):
            value = value[7:].strip()
        raw_value = value.split("=", 1)[1]
        occurrences[key] = occurrences.get(key, 0) + 1
        if key in values and values[key] != raw_value:
            fail("secret_document_conflicting_duplicate")
        values[key] = raw_value

updates = dict(request["updates"])
if request["role"] == "gateway":
    for source, target in request["aliases"].items():
        raw_value = str(values.get(source) or "").strip()
        try:
            parts = shlex.split("VALUE=" + raw_value, comments=True, posix=True)
        except ValueError:
            fail("required_existing_value_invalid")
        if len(parts) != 1 or not parts[0].startswith("VALUE="):
            fail("required_existing_value_invalid")
        value = parts[0].split("=", 1)[1]
        if not value.strip():
            fail("required_existing_value_missing")
        updates[target] = value
    service_key = str(request.get("service_key") or "")
    if not service_key.startswith("sb_secret_") or any(ch.isspace() for ch in service_key):
        fail("service_key_malformed")
    updates["LAB_ARENA_SERVICE_KEY"] = service_key

for key in updates:
    if occurrences.get(key, 0) > 1:
        fail("target_key_duplicate")

if document_format == "json":
    merged = dict(values)
    merged.update(updates)
    updated = json.dumps(merged, separators=(",", ":"))
else:
    pending = dict(updates)
    lines = []
    for line in raw.replace("\x00", "\n").splitlines():
        key = key_of(line)
        if key in pending:
            lines.append("export %s=%s" % (key, shlex.quote(pending.pop(key))))
        else:
            lines.append(line)
    for key, value in pending.items():
        lines.append("export %s=%s" % (key, shlex.quote(str(value))))
    updated = "\n".join(lines) + "\n"

print_result = {"ok": True, "account": account, "format": document_format,
                "changed_keys": sorted(updates), "applied": False,
                "before_version": before_id}
if updated == raw:
    print_result["unchanged"] = True
    print(json.dumps(print_result, separators=(",", ":")))
    raise SystemExit(0)
if not request["apply"]:
    print(json.dumps(print_result, separators=(",", ":")))
    raise SystemExit(0)
candidate = str(uuid.uuid4())
put = ["secretsmanager", "put-secret-value", "--secret-id", secret_id,
       "--client-request-token", candidate, "--version-stages", "AWSCURRENT",
       "--secret-string", "file:///dev/stdin", "--output", "json"]
current_id, current_raw = get(secret_id)
if current_id != before_id or current_raw != raw:
    fail("version_race")
aws(put, updated)
after_id, after_raw = get(secret_id)
if after_id != candidate or after_raw != updated:
    fail("post_apply_mismatch")
print_result.update({"applied": True, "after_version": candidate})
print(json.dumps(print_result, separators=(",", ":")))
'''


_RUNNER_WALLET_REMOTE = r'''
import contextlib, json, os, stat, sys
from pathlib import Path
from bittensor_wallet import Keypair, Wallet

root = Path("/var/lib/lab-arena/runner-wallets")
wallet_dir = root / "arena_runner"
hotkeys_dir = wallet_dir / "hotkeys"
hotkey_file = hotkeys_dir / "default"
for path in (root.parent, root, wallet_dir, hotkeys_dir, hotkey_file):
    if path.is_symlink():
        print(json.dumps({"ok": False, "code": "runner_wallet_symlink"}))
        raise SystemExit(1)
exists = hotkey_file.is_file()
if not exists and not bool(int(sys.argv[1])):
    print(json.dumps({"ok": True, "exists": False, "created": False}))
    raise SystemExit(0)
if not exists:
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    wallet_dir.mkdir(mode=0o700, exist_ok=True)
    hotkeys_dir.mkdir(mode=0o700, exist_ok=True)
    os.chmod(root, 0o700)
    os.chmod(wallet_dir, 0o700)
    os.chmod(hotkeys_dir, 0o700)
    with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        wallet = Wallet(path=str(root), name="arena_runner", hotkey="default")
        wallet.set_hotkey(
            Keypair.create_from_mnemonic(Keypair.generate_mnemonic()),
            encrypt=False, overwrite=False,
        )
    if not hotkey_file.is_file() or hotkey_file.is_symlink():
        print(json.dumps({"ok": False, "code": "runner_wallet_create_failed"}))
        raise SystemExit(1)
    os.chmod(hotkey_file, 0o600)
else:
    metadata = hotkey_file.stat()
    mode = stat.S_IMODE(metadata.st_mode)
    if metadata.st_uid != 0 or mode != 0o600:
        print(json.dumps({"ok": False, "code": "runner_wallet_permissions"}))
        raise SystemExit(1)
for directory in (root, wallet_dir, hotkeys_dir):
    metadata = directory.stat()
    if metadata.st_uid != 0 or stat.S_IMODE(metadata.st_mode) != 0o700:
        print(json.dumps({"ok": False, "code": "runner_wallet_permissions"}))
        raise SystemExit(1)
with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
    wallet = Wallet(path=str(root), name="arena_runner", hotkey="default")
    address = wallet.hotkey.ss58_address
print(json.dumps({"ok": True, "exists": True, "created": not exists,
                  "ss58_address": address}, separators=(",", ":")))
'''


_ACCOUNT_REMOTE = r'''
import json, subprocess, sys
result = subprocess.run(
    ["aws", "sts", "get-caller-identity", "--output", "json"],
    text=True, capture_output=True, check=False,
)
if result.returncode:
    print(json.dumps({"ok": False, "code": "account_check_failed"}))
    raise SystemExit(1)
try:
    account = str(json.loads(result.stdout)["Account"])
except Exception:
    print(json.dumps({"ok": False, "code": "account_check_invalid"}))
    raise SystemExit(1)
if account not in json.loads(sys.argv[1]):
    print(json.dumps({"ok": False, "code": "account_not_allowed"}))
    raise SystemExit(1)
print(json.dumps({"ok": True, "account": account}, separators=(",", ":")))
'''


def _read_fd(fd: int) -> str:
    with os.fdopen(os.dup(fd), "r", encoding="utf-8") as handle:
        value = handle.read().strip()
    if not value:
        raise ConfigurationError("scoped service key input is empty")
    return value


def _ssh(host: str, key: Path, request: Mapping[str, object]) -> dict[str, object]:
    result = subprocess.run(
        ["ssh", "-i", str(key), "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes",
         "-o", "ConnectTimeout=15", host, "python3 -c " + shlex.quote(_REMOTE)],
        input=json.dumps(request, separators=(",", ":")), text=True,
        capture_output=True, check=False, timeout=60,
    )
    if result.returncode != 0:
        try:
            code = json.loads(result.stdout).get("code", "remote_failed")
        except Exception:
            code = "remote_failed"
        raise ConfigurationError("%s: %s" % (host, code))
    try:
        response = json.loads(result.stdout)
    except ValueError as exc:
        raise ConfigurationError("remote response is invalid") from exc
    if not response.get("ok"):
        raise ConfigurationError("remote operation failed")
    return response


def _prepare_runner_wallet(host: str, key: Path, *, apply: bool) -> dict[str, object]:
    result = subprocess.run(
        ["ssh", "-i", str(key), "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes",
         "-o", "ConnectTimeout=15", host,
         "sudo /home/ec2-user/venv311/bin/python3 -c " + shlex.quote(_RUNNER_WALLET_REMOTE) + " " + ("1" if apply else "0")],
        text=True, capture_output=True, check=False, timeout=60,
    )
    if result.returncode != 0:
        try:
            code = json.loads(result.stdout).get("code", "runner_wallet_failed")
        except Exception:
            code = "runner_wallet_failed"
        raise ConfigurationError("runner wallet: %s" % code)
    try:
        response = json.loads(result.stdout)
    except ValueError as exc:
        raise ConfigurationError("runner wallet response is invalid") from exc
    return response


def _check_remote_account(host: str, key: Path, allowed_accounts: Sequence[str]) -> dict[str, object]:
    command = "python3 -c %s %s" % (
        shlex.quote(_ACCOUNT_REMOTE), shlex.quote(json.dumps(list(allowed_accounts)))
    )
    result = subprocess.run(
        ["ssh", "-i", str(key), "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes",
         "-o", "ConnectTimeout=15", host, command],
        text=True, capture_output=True, check=False, timeout=30,
    )
    if result.returncode != 0:
        try:
            code = json.loads(result.stdout).get("code", "account_check_failed")
        except Exception:
            code = "account_check_failed"
        raise ConfigurationError("validator account: %s" % code)
    return json.loads(result.stdout)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check or apply Lab Arena production environment")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="check only (the default)")
    mode.add_argument("--apply", action="store_true", help="apply after an authorized check")
    parser.add_argument("--prepare-runner", action="store_true", help="inspect or create the dedicated host-only runner signer")
    parser.add_argument("--service-key-fd", "--service-jwt-fd", dest="service_key_fd", type=int, help="inherited descriptor containing only the scoped service key")
    parser.add_argument("--ssh-key", type=Path, default=Path(os.getenv("LEADPOET_LAB_ARENA_SSH_KEY") or DEFAULT_SSH_KEY))
    parser.add_argument("--gateway-host", default=GATEWAY_HOST)
    parser.add_argument("--validator-host", default=VALIDATOR_HOST)
    parser.add_argument("--allowed-account", action="append", required=True)
    parser.add_argument("--bucket")
    parser.add_argument("--scorer-image")
    parser.add_argument("--runner-hotkey")
    parser.add_argument("--baseline-hotkey")
    parser.add_argument("--chain-endpoint")
    parser.add_argument("--api-base-url")
    parser.add_argument("--daily-cutoff-utc", type=int, default=0)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    accounts = set(args.allowed_account)
    if not accounts or not accounts <= KNOWN_ACCOUNTS:
        raise ConfigurationError("allowed account must be in the repository allowlist")
    if args.apply and os.getenv(AUTH_ENV) != "1":
        raise ConfigurationError("--apply requires %s=1" % AUTH_ENV)
    if not 0 <= args.daily_cutoff_utc <= 23:
        raise ConfigurationError("--daily-cutoff-utc must be between 0 and 23")
    if not args.ssh_key.is_file():
        raise ConfigurationError("SSH key does not exist")
    if not args.prepare_runner and args.service_key_fd is None:
        raise ConfigurationError("--service-key-fd is required for configuration")
    for name in (() if args.prepare_runner else ("bucket", "scorer_image", "runner_hotkey", "baseline_hotkey", "chain_endpoint", "api_base_url")):
        raw_value = getattr(args, name)
        value = str(raw_value or "")
        if not value.strip() or any(ch in value for ch in "\r\n\x00"):
            raise ConfigurationError("invalid argument: %s" % name)
    for host in (args.gateway_host, args.validator_host):
        if host.startswith("-") or any(ch in host for ch in "\r\n\x00"):
            raise ConfigurationError("invalid SSH host")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        _validate_args(args)
        if args.prepare_runner:
            account = _check_remote_account(
                args.validator_host, args.ssh_key, args.allowed_account
            )
            runner_wallet = _prepare_runner_wallet(
                args.validator_host, args.ssh_key, apply=args.apply
            )
            print(json.dumps({
                "ok": True,
                "mode": "apply" if args.apply else "check",
                "account": account,
                "runner_wallet": runner_wallet,
            }, separators=(",", ":")))
            return 0
        service_key = _read_fd(args.service_key_fd)
        _validate_service_key(service_key)
        gateway = gateway_nonsecret_updates(args)
        requests = [
            (args.gateway_host, {
                "secret_id": GATEWAY_SECRET, "allowed_accounts": args.allowed_account,
                "apply": args.apply, "role": "gateway", "updates": gateway,
                "service_key": service_key,
                "aliases": {"OPENROUTER_API_KEY": "LAB_ARENA_OPENROUTER_API_KEY",
                            "SCRAPINGDOG_API_KEY": "LAB_ARENA_SCRAPINGDOG_API_KEY",
                            "DEEPLINE_API_KEY": "LAB_ARENA_DEEPLINE_API_KEY",
                            "SUPABASE_URL": "LAB_ARENA_SUPABASE_URL",
                            "SUPABASE_ANON_KEY": "LAB_ARENA_SUPABASE_ANON_KEY"},
            }),
            (args.validator_host, {
                "secret_id": VALIDATOR_SECRET, "allowed_accounts": args.allowed_account,
                "apply": args.apply, "role": "validator",
                "updates": validator_updates(args), "aliases": {},
            }),
        ]
        # Both read-only preflights must pass before the first possible write.
        preflight = []
        for host, request in requests:
            probe = dict(request)
            probe["apply"] = False
            preflight.append(_ssh(host, args.ssh_key, probe))
        results = preflight
        if args.apply:
            results = []
            for host, request in requests:
                try:
                    results.append(_ssh(host, args.ssh_key, request))
                except Exception as exc:
                    raise ConfigurationError(
                        "apply failed after %d target(s); inspect version stages before retry: %s"
                        % (len(results), exc)
                    ) from exc
        print(json.dumps({"ok": True, "mode": "apply" if args.apply else "check", "targets": results}, separators=(",", ":")))
        return 0
    except (ConfigurationError, subprocess.TimeoutExpired) as exc:
        print("configuration failed: %s" % exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
