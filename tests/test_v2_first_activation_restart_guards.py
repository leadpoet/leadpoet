from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_gateway_first_activation_exits_before_process_shutdown():
    content = (ROOT / "gw_restart.sh").read_text(encoding="utf-8")
    assert '"status": "bootstrap_pending"' in content
    assert '"production_shutdown_started": False' in content
    assert content.index("if report_gateway_v2_bootstrap_pending; then") < content.index(
        'pkill -9 -f "python3 main.py"'
    )
    assert content.index("Acquiring the independently built V2 release channel") < content.index(
        'pkill -9 -f "python3 main.py"'
    )


def test_validator_first_activation_exits_before_process_shutdown():
    content = (ROOT / "validator_restart.sh").read_text(encoding="utf-8")
    assert '"status": "bootstrap_pending"' in content
    assert '"production_shutdown_started": False' in content
    assert content.index(
        'if [ "${#VALIDATOR_V2_MISSING_INPUTS[@]}" -gt 0 ]; then'
    ) < content.index('sudo pkill -TERM -f ".auto_update_wrapper.sh"')
    assert content.index("Acquiring the independently built V2 release channel") < content.index(
        'sudo pkill -TERM -f ".auto_update_wrapper.sh"'
    )
