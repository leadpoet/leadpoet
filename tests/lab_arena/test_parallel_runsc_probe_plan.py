"""The production-style probe must stay bounded and credential-free."""

import ast

import pytest

from lab_arena import shim
from lab_arena.runner import RunState, WorkerSocketServer
from scripts import probe_arena_parallel_runsc as probe
from scripts._lab_arena_runsc_probe_ci import ProbeApi


@pytest.mark.parametrize("workers", (2, 9, 20))
def test_parallel_probe_plan(workers, capsys):
    assert probe.main(["--dry-run", "--workers", str(workers)]) == 0
    assert '"sandboxes": %d' % (2 * workers) in capsys.readouterr().out
    for index in range(workers):
        reference = probe._source(index, parallel=False)
        measured = probe._source(index, parallel=True)
        ast.parse(reference)
        ast.parse(measured)
        assert '"query": "probe-%d"' % index in reference
        assert '"Probe %d"' % index in measured
    assert "INTENTIONAL_FAILURE" in probe._source(0, parallel=True)
    assert "time.sleep(600)" in probe._source(1, parallel=True)


@pytest.mark.parametrize("workers", (1, 21))
def test_parallel_probe_rejects_out_of_range_workers(workers):
    with pytest.raises(SystemExit):
        probe.main(["--dry-run", "--workers", str(workers)])


def test_local_provider_call_identity_is_independent_of_run_id(tmp_path):
    frame = shim.build_operation_frame("exa.search", {"query": "probe-2"}, 5000)
    identities = []
    for run_id in ("reference2", "parallel2"):
        api = ProbeApi()
        state = RunState(lease={"run_id": run_id}, lease_token="probe")
        server = WorkerSocketServer(tmp_path / "worker.sock", api, state)
        assert server.handle_frame(frame)
        identities.append(state.calls[0]["call_identity"])
    assert identities[0] == identities[1]
