"""Exercise the checked-in operator include with real OpenSSH, without AWS/network."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
TARGETS = [
    ("52.91.135.79", "i-07e945bb2653c2e8f", "52.91.135.79"),
    ("gateway", "i-07e945bb2653c2e8f", "52.91.135.79"),
    ("leadpoet-gateway-ssm", "i-07e945bb2653c2e8f", "52.91.135.79"),
    ("100.59.201.156", "i-07ab2b90c537d0f28", "100.59.201.156"),
    ("validator", "i-07ab2b90c537d0f28", "100.59.201.156"),
    ("leadpoet-validator-ssm", "i-07ab2b90c537d0f28", "100.59.201.156"),
    ("13.218.210.185", "i-0e428b150faa9d20f", "13.218.210.185"),
    ("subnet71", "i-0e428b150faa9d20f", "13.218.210.185"),
    ("leadpoet-dashboard-ssm", "i-0e428b150faa9d20f", "13.218.210.185"),
]


@pytest.fixture
def operator(tmp_path):
    ssh = shutil.which("ssh")
    if ssh is None:
        pytest.skip("OpenSSH is required to validate the real config parser")
    home = tmp_path / "home"
    ssh_dir = home / ".ssh"
    ssh_dir.mkdir(parents=True)
    include = ssh_dir / "leadpoet-ssm.conf"
    include.write_bytes((ROOT / "deploy/ssh-ssm.conf").read_bytes())
    # A fixture record is enough: parsing must preserve it, not invent trust.
    known_hosts = ssh_dir / "known_hosts"
    known_hosts.write_text("fixture-existing-host-key-must-stay-unchanged\n")
    existing = (
        "Host unrelated-host\n    HostName unrelated.internal\n"
        "    User other-operator\n    ProxyCommand none\n"
        "Host *\n    Port 2222\n    ConnectTimeout 41\n    ForwardAgent yes\n"
    )
    config = ssh_dir / "config"
    # An absolute include keeps the fixture isolated even on systems where ssh
    # resolves ~ through the account database rather than the HOME environment.
    config.write_text(f"Include {include}\nHost *\n" + existing)
    env = {**os.environ, "HOME": str(home), "AWS_CONFIG_FILE": str(home / "aws-config"),
           "AWS_SHARED_CREDENTIALS_FILE": str(home / "aws-credentials"),
           "AWS_EC2_METADATA_DISABLED": "true"}
    return ssh, config, env, known_hosts, existing


def resolved(operator, host):
    ssh, config, env, *_ = operator
    result = subprocess.run([ssh, "-G", "-F", str(config), host], env=env,
                            capture_output=True, text=True, check=True, timeout=10)
    return dict(line.split(" ", 1) for line in result.stdout.splitlines())


@pytest.mark.parametrize("host,instance,ip", TARGETS)
def test_aliases_and_existing_ips_pin_transport_identity_and_trust(operator, host, instance, ip):
    values = resolved(operator, host)
    assert values["hostname"] == instance
    assert values["hostkeyalias"] == ip
    assert values["user"] == "ec2-user"
    assert values["port"] == "22"
    assert values["identityfile"] == "~/.ssh/leadpoet-2026-07-28.pem"
    assert values["identitiesonly"] == "yes"
    assert values["stricthostkeychecking"] == "true"
    assert values["forwardagent"] == "no"
    assert values["connecttimeout"] == "15"
    assert values["serveraliveinterval"] == "30"
    assert values["serveralivecountmax"] == "3"
    assert values["proxycommand"] == (
        "aws --profile leadpoet-lab --region us-east-1 ssm start-session "
        "--document-name AWS-StartSSHSession --target %h --parameters portNumber=%p"
    )


def test_install_preserves_unrelated_configuration_and_known_hosts(operator):
    _, config, _, known_hosts, existing = operator
    before = known_hosts.read_bytes()
    values = resolved(operator, "unrelated-host")
    assert values["hostname"] == "unrelated.internal"
    assert values["user"] == "other-operator"
    assert values["port"] == "2222"
    assert values["connecttimeout"] == "41"
    assert values["forwardagent"] == "yes"
    assert "proxycommand" not in values  # OpenSSH omits the explicit 'none'.
    assert config.read_text().endswith(existing)
    assert known_hosts.read_bytes() == before
    # Similar names and unrelated IPs must not acquire these routes.
    for host in ("gateway-extra", "subnet71-extra", "192.0.2.20"):
        values = resolved(operator, host)
        assert values["hostname"] == host
        assert values["port"] == "2222"
        assert "hostkeyalias" not in values
        assert "proxycommand" not in values


@pytest.mark.parametrize("failure", ["missing-profile", "expired-sso"])
def test_proxy_failure_stops_real_ssh_without_direct_fallback(operator, tmp_path, failure):
    ssh, config, env, *_ = operator
    binary_dir = tmp_path / "bin"
    binary_dir.mkdir()
    aws = binary_dir / "aws"
    invocation = tmp_path / "aws-invocation.json"
    aws.write_text(
        f"#!{sys.executable}\nimport json, os, sys\n"
        "from pathlib import Path\n"
        "Path(os.environ['SSM_TEST_INVOCATION']).write_text(json.dumps(sys.argv[1:]))\n"
        "print(os.environ['SSM_TEST_FAILURE'], file=sys.stderr)\nraise SystemExit(42)\n"
    )
    aws.chmod(0o700)
    env = {**env, "PATH": str(binary_dir), "SSM_TEST_INVOCATION": str(invocation),
           "SSM_TEST_FAILURE": failure, "AWS_PROFILE": "wrong-ambient-profile",
           "AWS_REGION": "wrong-ambient-region"}
    result = subprocess.run(
        [ssh, "-vv", "-F", str(config), "-o", "BatchMode=yes", "leadpoet-gateway-ssm", "true"],
        env=env, capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 255
    assert failure in result.stderr
    assert json.loads(invocation.read_text()) == [
        "--profile", "leadpoet-lab", "--region", "us-east-1", "ssm", "start-session",
        "--document-name", "AWS-StartSSHSession", "--target", "i-07e945bb2653c2e8f",
        "--parameters", "portNumber=22",
    ]
    assert "Executing proxy command" in result.stderr
    assert "Connecting to " not in result.stderr


@pytest.mark.parametrize("aws_exit,account", [(1, ""), (0, "000000000000"), (0, "493765492819")])
def test_documented_account_preflight_gates_ssh(tmp_path, aws_exit, account):
    document = (ROOT / "docs/production-access.md").read_text()
    command = document.split("# access-preflight\n", 1)[1].split("\n```", 1)[0]
    binary_dir = tmp_path / "bin"
    binary_dir.mkdir()
    marker = tmp_path / "ssh-called"
    for name, body in {
        "aws": f"printf '%s\\n' '{account}'\nexit {aws_exit}\n",
        "ssh": 'printf called > "$SSM_TEST_SSH_MARKER"\n',
    }.items():
        executable = binary_dir / name
        executable.write_text("#!/bin/sh\n" + body)
        executable.chmod(0o700)
    result = subprocess.run(["/bin/sh", "-c", command], capture_output=True, text=True,
                            env={"PATH": str(binary_dir), "HOME": str(tmp_path),
                                 "SSM_TEST_SSH_MARKER": str(marker)}, timeout=10)
    allowed = aws_exit == 0 and account == "493765492819"
    assert marker.exists() is allowed
    assert (result.returncode == 0) is allowed
