"""Canonical restarts retain the request evidence needed to diagnose pickup."""

import os
from pathlib import Path
import re
import subprocess

import pytest


@pytest.mark.parametrize("variable", ["GATEWAY_LOG_FILE", "LAB_ARENA_SERVICE_LOG_FILE"])
def test_restart_log_redirection_preserves_prior_run(tmp_path, variable):
    source = (Path(__file__).resolve().parents[1] / "gw_restart.sh").read_text()
    redirects = re.findall(r'(>{1,2}) "\$' + variable + r'" 2>&1 < /dev/null', source)
    assert len(redirects) == 1
    log = tmp_path / "service.log"
    log.write_text("prior request evidence\n")
    environment = dict(os.environ, **{variable: str(log)})
    command = 'printf "new process started\\n" ' + redirects[0] + ' "$' + variable + '" 2>&1 < /dev/null'
    subprocess.run(["bash", "-c", command], env=environment, check=True, timeout=5)
    assert log.read_text() == "prior request evidence\nnew process started\n"
    assert redirects == [">>"]
