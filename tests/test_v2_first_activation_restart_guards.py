from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_gateway_first_activation_exits_before_process_shutdown():
    content = (ROOT / "gw_restart.sh").read_text(encoding="utf-8")
    assert '"status": "bootstrap_pending"' in content
    assert '"production_shutdown_started": False' in content
    assert content.index("if report_gateway_v2_bootstrap_pending; then") < content.index(
        "gateway/tee/scoped_shutdown_v2.py"
    )
    assert content.index("Acquiring the independently built V2 release channel") < content.index(
        "gateway/tee/scoped_shutdown_v2.py"
    )


def test_validator_first_activation_exits_before_process_shutdown():
    content = (ROOT / "validator_restart.sh").read_text(encoding="utf-8")
    assert '"status": "bootstrap_pending"' in content
    assert '"production_shutdown_started": False' in content
    missing_inputs = content.index(
        'if [ "${#VALIDATOR_V2_MISSING_INPUTS[@]}" -gt 0 ]; then'
    )
    destructive_phase = content.index("VALIDATOR_DESTRUCTIVE_PHASE_STARTED=1")
    shutdown = content.index(
        'echo "Stopping validator processes and containers"',
        destructive_phase,
    )
    first_shutdown_signal = content.index(
        'sudo pkill -TERM -f ".auto_update_wrapper.sh"',
        shutdown,
    )
    assert missing_inputs < destructive_phase < shutdown < first_shutdown_signal
    assert (
        content.index("Acquiring the independently built V2 release channel")
        < destructive_phase
    )
