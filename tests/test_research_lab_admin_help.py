import pytest

from gateway.research_lab import admin
from gateway.research_lab.admin import build_parser


def test_research_lab_admin_help_formats_every_subcommand(capsys):
    parser = build_parser()

    parser.print_help()

    output = capsys.readouterr().out
    assert "Require complete V2 receipt coverage" in output
    assert "pause-scoring" not in output
    assert "resume-scoring" not in output
    assert "resume-restart-maintenance" not in output


@pytest.mark.parametrize(
    "command",
    ("pause-scoring", "resume-scoring", "resume-restart-maintenance"),
)
def test_research_lab_admin_rejects_retired_scoring_controls(command):
    with pytest.raises(SystemExit):
        build_parser().parse_args([command])


@pytest.mark.asyncio
async def test_research_lab_admin_status_only_reports_source_add(monkeypatch):
    expected = {"action": "source-add status", "paused": False}

    async def source_add_status():
        return expected

    monkeypatch.setattr(admin, "_source_add_status", source_add_status)

    result = await admin._run(build_parser().parse_args(["status"]))

    assert result == {"ok": True, "source_add": expected}
