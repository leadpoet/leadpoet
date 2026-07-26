from gateway.research_lab.admin import build_parser


def test_research_lab_admin_help_formats_every_subcommand(capsys):
    parser = build_parser()

    parser.print_help()

    output = capsys.readouterr().out
    assert "Require 100% V2 receipt coverage" in output
