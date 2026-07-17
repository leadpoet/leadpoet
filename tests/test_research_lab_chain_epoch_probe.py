from __future__ import annotations

import subprocess

import pytest

from gateway.research_lab import chain


def test_direct_epoch_probe_runs_in_killable_proxy_free_subprocess(monkeypatch):
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                "bittensor log\n"
                "LEADPOET_EPOCH_RESULT={\"epoch\":23978,\"block\":8632213,"
                "\"network\":\"finney\",\"official_subnet_epoch_id\":23913,"
                "\"epoch_ref\":\"sha256:fixture\"}\n"
            ),
            stderr="",
        )

    monkeypatch.setenv("BITTENSOR_NETWORK", "finney")
    monkeypatch.setenv("HTTPS_PROXY", "http://secret-proxy.example")
    monkeypatch.setattr(chain.subprocess, "run", fake_run)

    assert chain._fetch_current_chain_epoch_direct() == (23978, 8632213, "finney")
    assert captured["command"][:2] == [chain.sys.executable, "-c"]
    probe = captured["command"][2]
    assert "snapshot = read_subnet_epoch_snapshot(" in probe
    assert "subtensor," in probe
    assert "finalized=True" in probe
    assert "validate_cutover_anchor_from_archive(cutover)" in probe
    assert (
        "assert_legacy_epoch_namespace_open(epoch, force_refresh=True)"
        in probe
    )
    assert "validate_subnet_epoch_cutover_anchor" not in probe
    assert "HTTPS_PROXY" not in captured["env"]
    assert captured["timeout"] == 19.0
    assert captured["capture_output"] is True


def test_direct_epoch_probe_timeout_is_visible(monkeypatch):
    def fake_run(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(chain.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="timed out after 19.0s"):
        chain._fetch_current_chain_epoch_direct()


@pytest.mark.parametrize(
    "stdout",
    (
        "",
        "LEADPOET_EPOCH_RESULT=not-json\n",
        "LEADPOET_EPOCH_RESULT={\"block\":0,\"network\":\"finney\"}\n",
        "LEADPOET_EPOCH_RESULT={\"block\":8632213,\"network\":\"test\"}\n",
    ),
)
def test_direct_epoch_probe_rejects_invalid_or_inconsistent_output(monkeypatch, stdout):
    monkeypatch.setenv("BITTENSOR_NETWORK", "finney")
    monkeypatch.setattr(
        chain.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 0, stdout=stdout, stderr=""
        ),
    )

    with pytest.raises(RuntimeError, match="invalid output|inconsistent output"):
        chain._fetch_current_chain_epoch_direct()
