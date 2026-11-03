import gzip
import json
import weakref

import pytest

from scripts import prepare_arena_training_dataset as cli


def test_gzip_input_preserves_complete_strings_and_stable_directory_order(tmp_path):
    documents = [{"text": "unchanged Ω\n" * 100}, {"text": "second"}]
    (tmp_path / "b.json").write_text(json.dumps(documents[1]))
    (tmp_path / "a.json.gz").write_bytes(gzip.compress(json.dumps(documents[0]).encode()))
    assert list(cli._examples([tmp_path], max_example_bytes=10000)) == documents


@pytest.mark.parametrize("compressed", [False, True])
def test_decompressed_limit_is_exact_and_source_untouched(tmp_path, compressed):
    raw = json.dumps({"text": "a" * 10000}).encode()
    path = tmp_path / ("example.json.gz" if compressed else "example.json")
    original = gzip.compress(raw) if compressed else raw
    path.write_bytes(original)
    assert list(cli._examples([path], max_example_bytes=len(raw))) == [json.loads(raw)]
    with pytest.raises(ValueError, match="example_too_large"):
        list(cli._examples([path], max_example_bytes=len(raw) - 1))
    assert path.read_bytes() == original


@pytest.mark.parametrize("payload", [b'{"text":', b'{} {}', b'[]'])
def test_rejects_incomplete_multiple_or_nonobject_json(tmp_path, payload):
    path = tmp_path / "example.json.gz"
    path.write_bytes(gzip.compress(payload))
    with pytest.raises(ValueError):
        list(cli._examples([path]))


def test_rejects_corrupt_gzip_and_compression_bomb(tmp_path):
    path = tmp_path / "example.json.gz"
    path.write_bytes(gzip.compress(b'{"text":"' + b'a' * 1000000 + b'"}'))
    with pytest.raises(ValueError, match="example_too_large"):
        list(cli._examples([path], max_example_bytes=1000))
    path.write_bytes(path.read_bytes()[:-5])
    with pytest.raises((EOFError, OSError)):
        list(cli._examples([path], max_example_bytes=2000000))


def test_multiple_episodes_release_previous_document_before_next_parse(tmp_path, monkeypatch):
    for index in range(3):
        (tmp_path / f'{index}.json').write_text(json.dumps({"index": index}))
    class TrackedDict(dict):
        pass

    refs = []
    loads = cli.json.loads

    def tracked_loads(raw):
        assert not any(ref() is not None for ref in refs)
        document = TrackedDict(loads(raw))
        refs.append(weakref.ref(document))
        return document

    monkeypatch.setattr(cli.json, "loads", tracked_loads)
    iterator = cli._examples([tmp_path], max_example_bytes=100)
    for index in range(3):
        document = next(iterator)
        assert document["index"] == index
        del document
    with pytest.raises(StopIteration):
        next(iterator)
    assert not any(ref() is not None for ref in refs)


@pytest.mark.parametrize("valid_final_source", [False, True])
def test_grouping_sources_release_before_next_parse_and_record_iteration(valid_final_source):
    from lab_arena import training_dataset
    from tests.lab_arena.training_dataset_test import example, row, claim

    class TrackedDict(dict):
        pass

    refs = []

    def grouping_sources():
        for index in range(3):
            assert not any(ref() is not None for ref in refs)
            source = TrackedDict(example(f"source-{index}", [row(0, claim(), "qualified")]))
            if index == 2 and not valid_final_source:
                source["example_hash"] = "invalid"
            refs.append(weakref.ref(source))
            yield source
            del source

    def record_sources():
        assert not any(ref() is not None for ref in refs)
        yield example("selected", [row(0, claim(), "qualified")])

    splits, _ = training_dataset.build(record_sources(), split_sources=grouping_sources())
    assert sum(len(splits[key]) for key in ("train", "eval")) == 1
    assert not any(ref() is not None for ref in refs)


def test_report_batch_limit_counts_decompressed_bytes(tmp_path):
    paths = []
    raw = json.dumps({"text": "a" * 1000}).encode()
    for index in range(2):
        path = tmp_path / f"{index}.json.gz"
        path.write_bytes(gzip.compress(raw))
        paths.append(path)
    with pytest.raises(ValueError, match="example_too_large"):
        list(cli._examples(paths, max_example_bytes=2 * len(raw), max_total_bytes=len(raw)))


@pytest.mark.parametrize("module_name", [
    "prepare_arena_training_dataset", "prepare_arena_agent_dataset",
    "prepare_arena_full_thinking_dataset", "report_arena_training",
])
def test_cli_later_corrupt_episode_leaves_source_and_no_partial_output(tmp_path, module_name):
    import importlib

    module = importlib.import_module("scripts." + module_name)
    source = tmp_path / "sources"
    source.mkdir()
    (source / "a.json").write_text('{}')
    bad = source / "b.json.gz"
    original = gzip.compress(b'{"incomplete":')
    bad.write_bytes(original)
    output = tmp_path / "output"
    assert module.main(["--examples", str(source), "--output", str(output),
                        "--max-example-bytes", "10000"]) == 1
    assert not output.exists()
    assert bad.read_bytes() == original


def test_full_thinking_multi_episode_global_split_and_manifest_golden():
    import copy
    from lab_arena import training_thinking
    from tests.lab_arena.training_thinking_test import full_example
    from tests.lab_arena.training_steps_test import seal

    first = full_example()
    second = copy.deepcopy(first)
    second["round_id"] = "arena-20261028-000002"
    seal(second)
    splits, manifest = training_thinking.build(iter([first, second, first]))
    # Bind the shared source graph, cross-source dedup, and coverage.
    assert manifest["dataset_version"] == "sha256:4b8067822b5508c8c787facf305445b7d22d7af138859acba3598bd7a72693f5"
    assert manifest["coverage"]["distinct_verified_exports"] == 2
    assert not (splits["train"] and splits["eval"])
