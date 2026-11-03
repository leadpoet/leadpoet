"""Baseline-first HTTP execution preserves costs, publication and reward weights."""

from dataclasses import replace

from leadpoet_canonical import arena_weights
from tests.lab_arena import proxy_model_capacity401_e2e_test as fullflow
from tests.lab_arena import test_lab_arena_service_round as fixtures
from tests.lab_arena.baseline_cost_zero_round_test import _accepted_state
from tests.lab_arena.parallel_twenty_runtime_e2e_test import (
    bounded_socket_shutdown_poll,
)
from tests.lab_arena.proxy_model_capacity401_e2e_test import complete_application_database


def test_ten_slot_baseline_first_round_publishes_promotes_and_derives_weights(
    complete_application_database, tmp_path, monkeypatch,
):
    database = complete_application_database
    install = fullflow._install_company_only_sandbox
    make_runner = fullflow._http_runner
    observed = {}

    def capture(harness):
        harness.service.config.defaults = replace(
            harness.service.config.defaults, runner_slot_ceiling=251,
            stage_minutes={
                **harness.service.config.defaults.stage_minutes,
                "stage_2": 240,
            },
        )
        observed["harness"] = harness
        return install(harness)

    def ten_slot_runner(*args, **kwargs):
        kwargs["local_capacity"] = 10
        return make_runner(*args, **kwargs)

    monkeypatch.setattr(fullflow, "_install_company_only_sandbox", capture)
    monkeypatch.setattr(fullflow, "_http_runner", ten_slot_runner)
    # The shared current-schema fixture has already installed the claim chain.
    fullflow._exercise_company_only_round(
        database=database, migrated=True, tmp_path=tmp_path,
    )
    h = observed["harness"]
    published = h.service.store.get_round(h.round_id)
    configuration = published["configuration_doc"]
    assert configuration["execution_sequence_policy"] == "baseline_scored_first_v1"
    assert configuration["runner_slot_ceiling"] == 251
    assert (configuration["stage_1_icp_count"], configuration["stage_2_icp_count"]) == (5, 5)
    assert "parallel_twenty_icp_execution" not in configuration
    runs = h.service.store.list_runs(h.round_id, kind="execute")
    assert len(runs) == 30 and all(run["status"] == "accepted" for run in runs)
    for run in runs:
        egress = run["result_doc"]["resource_summary"]["web_egress"]
        assert 0 <= egress["worker_slot"] < 10
        assert egress["exit_fingerprint"]
    assert h.service.activate_reward(h.round_id)["status"] == "activated"
    activated = h.service.store.get_round(h.round_id)
    winner = activated["reward_basis_doc"]["king_hotkey"]
    burn = fixtures.keypair("idle-worker-claims-burn").ss58_address
    state = _accepted_state(h, int(activated["effective_reward_epoch"]), burn)
    weights = arena_weights.derive_arena_weights(state, [burn, winner])
    # This future-dated round uses the current signed 50-percent miner pot.
    assert activated["reward_basis_doc"]["slot_policy"]["pool_percent"] == 50
    assert weights["champion_share_ppb"] == 500_000_000
    assert weights["burned_residual_ppb"] == 500_000_000
    fixtures.assert_canary_absent(h, lambda: database[0].connect(**database[1]))
