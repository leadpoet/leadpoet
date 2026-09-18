"""Use an operator's existing sudo permission for the rootful scoring host.

No permissions are installed or granted here. Check permission without a
password prompt before replacing this process, and preserve the caller's
wallet, journal, interpreter and configuration. If sudo is unavailable, the
ordinary validator still starts its independent weight loop.
"""

from __future__ import annotations

import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence

SUDO_CHECK_TIMEOUT_SECONDS = 5
_CONFIG_NAME = re.compile(
    r"(?:LAB_ARENA_[A-Z0-9_]+|QUALIFICATION_WEBSHARE_PROXY_[1-9][0-9]*|"
    r"LEADPOET_SUBNET_EPOCH_CUTOVER_(?:JSON|PATH))\Z"
)
_ROOT_PROBE = "import os; raise SystemExit(0 if os.geteuid() == 0 else 1)"


def _absolute_path(value: str) -> str:
    # Do not resolve symlinks: in particular, resolving the venv interpreter
    # would silently select the system interpreter instead.
    return str(Path(value).expanduser().absolute())


def rootful_startup_command(args, argv: Sequence[str], environment: Mapping[str, str]):
    """Freeze user-relative paths before sudo changes HOME or environment."""
    from lab_arena.validator_proxy_environment import validator_proxy_environment

    # A root process must not re-read a user-owned proxy file under a different
    # ownership rule. Validate it as the caller and carry only proxy settings.
    configured = validator_proxy_environment(environment)
    child_environment = dict(environment)
    child_environment.update({
        name: value for name, value in configured.items()
        if _CONFIG_NAME.fullmatch(name)
    })
    child_environment.pop("LAB_ARENA_PROXY_ENV_FILE", None)
    child_environment["LAB_ARENA_VALIDATOR_STATE_DIR"] = _absolute_path(
        environment.get("LAB_ARENA_VALIDATOR_STATE_DIR", "/var/lib/leadpoet/arena-validator")
    )
    cutover = child_environment.get("LEADPOET_SUBNET_EPOCH_CUTOVER_PATH")
    if cutover:
        child_environment["LEADPOET_SUBNET_EPOCH_CUTOVER_PATH"] = _absolute_path(cutover)
    names = sorted(name for name in child_environment if _CONFIG_NAME.fullmatch(name))
    command = [
        _absolute_path(sys.executable), "-B", str(Path(__file__).resolve()), *argv,
        "--wallet.path", _absolute_path(args.wallet_path),
        "--arena-work-dir", _absolute_path(args.work_dir),
        "--arena-runsc-path", args.runsc_path,
    ]
    return command, child_environment, "--preserve-env=" + ",".join(names)


def maybe_reexec_rootful(args, argv: Sequence[str]) -> None:
    """Replace this process only when existing sudo policy permits startup."""
    if (platform.system() != "Linux" or os.geteuid() == 0
            or args.check_only or args.check_scoring_only):
        return
    sudo = next((path for path in ("/usr/bin/sudo", "/bin/sudo")
                 if os.path.isfile(path) and os.access(path, os.X_OK)), None)
    if sudo is None:
        return
    try:
        command, environment, preserve = rootful_startup_command(args, argv, os.environ)
        checks = (
            # Listing the exact command does not execute it. A restricted
            # sudo rule must not authorize a different Python invocation.
            [sudo, "-n", "-l", "--", *command],
            # -I -S makes this a side-effect-free identity probe, without
            # sitecustomize, wallet loading, chain access or validator work.
            [sudo, "-n", preserve, "--", command[0], "-I", "-S", "-c", _ROOT_PROBE],
        )
        for check in checks:
            result = subprocess.run(
                check, env=environment, stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=SUDO_CHECK_TIMEOUT_SECONDS, check=False,
            )
            if result.returncode != 0:
                return
    except (OSError, ValueError, subprocess.TimeoutExpired):
        # Scoring setup retains its normal bounded diagnostic. Never print
        # proxy values, sudo output, or arbitrary configuration exceptions.
        return
    print("Arena validator startup: using existing non-interactive sudo permission; "
          "wallet and state paths preserved", flush=True)
    try:
        # Replacing, rather than spawning a second validator, preserves the
        # supervisor's process/signal boundary and prevents duplicate signers.
        os.execve(sudo, [sudo, "-n", preserve, "--", *command], environment)
    except OSError:
        print("Arena validator startup: sudo launch unavailable; weight loop continues",
              file=sys.stderr, flush=True)


if __name__ == "__main__":
    # Absolute script invocation works from any CWD and from installed wheels.
    root = str(Path(__file__).resolve().parents[1])
    if root not in sys.path:
        sys.path.insert(0, root)
    from lab_arena.validator import main

    raise SystemExit(main())
