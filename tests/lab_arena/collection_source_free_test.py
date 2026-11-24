"""Full-task experiments start from the frozen builder without a student transcript."""
import pytest

from lab_arena import collection_experiment as experiment, collection_intake as funding
from tests.lab_arena.collection_experiment_test import settings


def source_free(repeats=1, *, max_slots=12):
    raw = settings()
    raw.pop("provenance_sha256")
    raw["cases"] = [{"scenario_id": "P7-s40401"}]
    raw["arms"] = [dict(raw["arms"][0], role="expert", repeats=repeats)]
    raw["max_slots"] = max_slots
    raw["budget_microusd"] = raw["arms"][0]["budget_microusd"] * repeats
    return raw


def test_repeat_pass_rate_arm_starts_at_initial_state_and_binds_size():
    raw = source_free(16, max_slots=16)
    spec = experiment.load_experiment(raw, {})
    assert [repeat for _, _, repeat in spec.slots] == list(range(1, 17))
    assert spec.batch_id != experiment.load_experiment(dict(raw, max_slots=17), {}).batch_id
    with pytest.raises(ValueError):
        experiment.load_experiment(dict(raw, max_slots=15), {})
    raw["arms"][0]["role"] = "continuation"
    with pytest.raises(ValueError):
        experiment.load_experiment(raw, {})


def test_shared_terms_bind_the_mode_and_scope_of_the_actual_signature(monkeypatch):
    for var in ("LAB_ARENA_CORRECTION_TERMS_SHA256", "LAB_ARENA_LIVE_CORRECTION_TERMS_SHA256"):
        monkeypatch.delenv(var, raising=False)
    offers = [funding.terms(mode, scope) for mode in ("pilot", "live")
              for scope in (funding.TASK_SCOPE, funding.CORRECTION_SCOPE)]
    assert len({o["terms_sha256"] for o in offers}) == 1
    assert len({(o["mode"], o["task_scope"]) for o in offers}) == 4
    assert "train" in offers[0]["terms"]


def test_prepared_repeat_pack_reconstructs_frozen_initial_state(tmp_path):
    import json
    import os
    from lab_arena.collection_workflow import workflow_sources
    from scripts.prepare_collection_experiment import main

    agent = os.environ.get("COLLECTION_TEST_AGENT_REPO")
    poet = os.environ.get("COLLECTION_TEST_POET_REPO")
    if not agent or not poet:
        pytest.skip("requires the pinned agent and scenario repositories")
    raw = source_free()
    _, revision = experiment.source_revisions(raw)
    with workflow_sources(agent, poet, corrections=True, builder_revision=revision) as sources:
        split = sources[1].training_partition("P7")
        partition = tmp_path / "split.json"
        partition.write_text(json.dumps(split))
        partition.chmod(0o600)
        pack = tmp_path / "prepared"
        model = funding.MODELS[0]
        assert main(["prepare-expert", "--scenario-id", "P7-s40401", "--model", model,
                     "--repeat", "16", "--max-slots", "16", "--experiment-id", "pass-rate",
                     "--question", "Does the assigned model pass repeatedly?", "--pass-mark", "passes",
                     "--terms-sha256", funding.SYNTHETIC_TRAINING_TERMS_HASH,
                     "--cap-microusd", "300000", "--max-steps", "20", "--max-generated-tokens", "8192",
                     "--split", str(partition), "--output", str(pack)]) == 0
        batch = json.loads((pack / "correction-batch.json").read_bytes())
        spec = experiment.load_experiment(batch, {})
        resolved, actual_split, provenance = experiment.read_inputs(spec, pack / "correction-inputs", sources)
        assert actual_split == split and provenance is None and len(spec.slots) == 16
        student, stamp = resolved["P7-s40401"]
        assert student is None and stamp["template_id"] == "P7"
