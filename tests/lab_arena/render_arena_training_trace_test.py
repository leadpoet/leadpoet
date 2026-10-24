import json
import stat

from lab_arena import contracts
from scripts.render_arena_training_trace import main
from tests.lab_arena.training_thinking_test import full_example


def _example_file(tmp_path):
    source = tmp_path / "example.json"
    source.write_text(contracts.canonical_json(full_example()) + "\n", encoding="utf-8")
    return source


def test_renders_hash_verified_private_raw_and_readable_views(tmp_path):
    source = _example_file(tmp_path)
    output = tmp_path / "views"
    assert main(["--example", str(source), "--output", str(output)]) == 0
    assert stat.S_IMODE(output.stat().st_mode) == 0o700
    assert {path.name for path in output.iterdir()} == {
        "raw-export.json", "readable-trace.json", "readable-trace.md"}
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in output.iterdir())
    assert (output / "raw-export.json").read_bytes() == source.read_bytes()
    report = json.loads((output / "readable-trace.json").read_text())
    assert report["example_hash"] == json.loads(source.read_text())["example_hash"]
    markdown = (output / "readable-trace.md").read_text()
    assert "[Open the raw private export](<raw-export.json>)" in markdown
    assert "Inspect the retained page, then answer." in markdown


def test_rejects_tampered_export_before_creating_output(tmp_path):
    source = _example_file(tmp_path)
    example = json.loads(source.read_text())
    example["icp"]["industry"] = "tampered"
    source.write_text(contracts.canonical_json(example), encoding="utf-8")
    output = tmp_path / "views"
    assert main(["--example", str(source), "--output", str(output)]) == 1
    assert not output.exists()


def test_never_overwrites_existing_output_directory(tmp_path):
    source = _example_file(tmp_path)
    output = tmp_path / "views"
    output.mkdir()
    sentinel = output / "raw-export.json"
    sentinel.write_text("existing", encoding="utf-8")
    assert main(["--example", str(source), "--output", str(output)]) == 1
    assert sentinel.read_text() == "existing"
    assert list(output.iterdir()) == [sentinel]
